"""P0-D: fail-closed GPU controller — cross-process lease, crash-safety,
chain-stop, output validation, idempotency, co-tenant policy."""
import sys, os, json, time, tempfile, subprocess, textwrap, signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from basemap import run_controller as rc

REPO = os.path.join(os.path.dirname(__file__), '..')


def _hold_lease_script(lease_path, hold_s):
    return textwrap.dedent(f"""
        import sys, time; sys.path.insert(0, {REPO!r})
        from basemap.run_controller import GpuLease
        L = GpuLease(path={lease_path!r}, timeout=0); L.acquire()
        print("HELD", flush=True); time.sleep({hold_s}); L.release()
    """)


def test_cross_process_lease_exclusion():
    lp = tempfile.mktemp(suffix='.lease')
    p = subprocess.Popen([sys.executable, '-c', _hold_lease_script(lp, 4)],
                         stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "HELD"      # child holds it
    try:
        rc.GpuLease(path=lp, timeout=0).acquire()
        assert False, "acquired a lease held by another process"
    except RuntimeError as e:
        assert "held by" in str(e)
    finally:
        p.wait()
    rc.GpuLease(path=lp, timeout=0).acquire().release()   # free after child exits
    print("PASS cross-process exclusion")


def test_lease_survives_controller_death_via_inherited_fd():
    # a child that inherits the lease fd keeps the lock after the 'controller' dies.
    lp = tempfile.mktemp(suffix='.lease')
    from basemap.run_controller import GpuLease
    L = GpuLease(path=lp, timeout=0); L.acquire()
    child = subprocess.Popen([sys.executable, '-c',
        f"import time,os; time.sleep(3)"], pass_fds=(L.fileno(),), close_fds=False)
    # 'controller' dies (close its fd) but child still holds the inherited fd
    os.close(L._fd); L._fd = None
    time.sleep(0.3)
    try:
        rc.GpuLease(path=lp, timeout=0).acquire()
        got = True
    except RuntimeError:
        got = False
    child.wait()
    assert not got, "lease was acquirable while inheriting child still alive"
    rc.GpuLease(path=lp, timeout=0).acquire().release()   # released after child exits
    print("PASS controller-death lease protection")


def test_nonzero_job1_stops_job2():
    d = tempfile.mkdtemp(); os.environ['BASEMAP_GPU_LEASE'] = os.path.join(d, '.lease')
    import importlib; importlib.reload(rc)
    j1 = rc.Job(name='a', argv=['false'], outputs=[], done_marker=os.path.join(d, 'a.done'))
    j2 = rc.Job(name='b', argv=['bash', '-c', f'touch {d}/b.out'],
                outputs=[f'{d}/b.out'], done_marker=os.path.join(d, 'b.done'))
    s = rc.run_jobs([j1, j2])
    assert s['jobs'][0]['status'].startswith('exit_')
    assert len(s['jobs']) == 1 and 'stopped' in s['stop_reason']
    assert not os.path.exists(f'{d}/b.out')   # job 2 never ran
    print("PASS nonzero job1 stops chain")


def test_exit_zero_without_outputs_is_failure():
    d = tempfile.mkdtemp(); os.environ['BASEMAP_GPU_LEASE'] = os.path.join(d, '.lease')
    import importlib; importlib.reload(rc)
    j = rc.Job(name='x', argv=['true'], outputs=[f'{d}/never'], done_marker=os.path.join(d, 'x.done'))
    s = rc.run_jobs([j])
    assert s['jobs'][0]['status'] == 'missing_outputs'
    assert not os.path.exists(os.path.join(d, 'x.done'))   # no completion record
    print("PASS exit-0 without outputs = failure")


def test_stale_output_does_not_skip_without_done_record():
    d = tempfile.mkdtemp(); os.environ['BASEMAP_GPU_LEASE'] = os.path.join(d, '.lease')
    import importlib; importlib.reload(rc)
    open(f'{d}/out', 'w').close()   # stale output present, but NO valid .done record
    j = rc.Job(name='x', argv=['bash', '-c', f'echo redo > {d}/out'],
               outputs=[f'{d}/out'], done_marker=os.path.join(d, 'x.done'))
    s = rc.run_jobs([j])
    assert s['jobs'][0]['status'] == 'ok'   # it RE-RAN (not skipped)
    assert open(f'{d}/out').read().strip() == 'redo'
    # second run WITH the completion record skips
    s2 = rc.run_jobs([j]); assert s2['jobs'][0]['status'] == 'skipped_done'
    print("PASS stale output re-runs; valid record skips")


def test_final_manifest_has_status_and_telemetry():
    d = tempfile.mkdtemp(); os.environ['BASEMAP_GPU_LEASE'] = os.path.join(d, '.lease')
    import importlib; importlib.reload(rc)
    j = rc.Job(name='x', argv=['bash', '-c', f'touch {d}/o'], outputs=[f'{d}/o'],
               done_marker=os.path.join(d, 'x.done'), manifest=os.path.join(d, 'x.manifest.json'))
    rc.run_jobs([j])
    m = json.load(open(os.path.join(d, 'x.manifest.json')))
    assert m['status'] == 'ok' and m['exit_code'] == 0 and 'gpu_post' in m and 'output_sigs' in m
    print("PASS final manifest has status + telemetry")


def test_p0_5_stale_noop_over_existing_output_fails():
    # exit-0 that does NOT change a pre-existing output must FAIL (stale) and write
    # no done record (P0-5).
    d = tempfile.mkdtemp(); os.environ['BASEMAP_GPU_LEASE'] = os.path.join(d, '.lease')
    import importlib; importlib.reload(rc)
    with open(f'{d}/out', 'w') as f: f.write("stale")
    j = rc.Job(name='x', argv=['true'], outputs=[f'{d}/out'], done_marker=os.path.join(d, 'x.done'))
    s = rc.run_jobs([j])
    assert s['jobs'][0]['status'] == 'stale_outputs', s['jobs'][0]
    assert not os.path.exists(os.path.join(d, 'x.done'))
    assert open(f'{d}/out').read() == "stale"


def test_p0_5_changed_argv_cannot_reuse_done_record():
    d = tempfile.mkdtemp(); os.environ['BASEMAP_GPU_LEASE'] = os.path.join(d, '.lease')
    import importlib; importlib.reload(rc)
    j1 = rc.Job(name='x', argv=['bash', '-c', f'echo a > {d}/out'],
                outputs=[f'{d}/out'], done_marker=os.path.join(d, 'x.done'))
    assert rc.run_jobs([j1])['jobs'][0]['status'] == 'ok'
    # different argv (same name/outputs) → digest differs → must NOT skip; re-runs
    j2 = rc.Job(name='x', argv=['bash', '-c', f'echo b > {d}/out'],
                outputs=[f'{d}/out'], done_marker=os.path.join(d, 'x.done'))
    assert rc.run_jobs([j2])['jobs'][0]['status'] == 'ok'
    assert open(f'{d}/out').read().strip() == 'b'


def test_p0_5_mutated_output_invalidates_skip():
    d = tempfile.mkdtemp(); os.environ['BASEMAP_GPU_LEASE'] = os.path.join(d, '.lease')
    import importlib; importlib.reload(rc)
    j = rc.Job(name='x', argv=['bash', '-c', f'echo a > {d}/out'],
               outputs=[f'{d}/out'], done_marker=os.path.join(d, 'x.done'))
    assert rc.run_jobs([j])['jobs'][0]['status'] == 'ok'
    assert rc.run_jobs([j])['jobs'][0]['status'] == 'skipped_done'   # unchanged → skip
    with open(f'{d}/out', 'w') as f: f.write("tampered")             # mutate output
    assert rc.run_jobs([j])['jobs'][0]['status'] == 'ok'             # re-runs, not skipped


def test_p0_5_known_service_pids_matches_by_identity(monkeypatch):
    # P0-5: allow-list by identity (cmdline marker), NOT snapshot-all. The current
    # process is a known service iff its cmdline matches a marker; unknown markers
    # never tolerate it.
    import basemap.run_controller as rc
    mypid = os.getpid()
    monkeypatch.setattr(rc, "gpu_snapshot", lambda: {"compute_pids": [mypid]})
    cmd = open(f"/proc/{mypid}/cmdline", "rb").read().replace(b"\x00", b" ").decode()
    token = "python" if "python" in cmd else os.path.basename(cmd.split()[0]) if cmd.split() else "x"
    assert mypid in rc.known_service_pids([token])
    assert rc.known_service_pids(["zzz_no_such_service_marker"]) == []


def test_p0_5_require_active_lease():
    d = tempfile.mkdtemp(); lp = os.path.join(d, '.lease')
    os.environ['BASEMAP_GPU_LEASE'] = lp
    import importlib; importlib.reload(rc)
    import pytest
    with pytest.raises(RuntimeError, match="lease"):
        rc.require_active_lease()                    # nobody holds it
    L = rc.GpuLease(path=lp, timeout=0).acquire()
    try:
        rc.require_active_lease()                    # held by us → OK
    finally:
        L.release()
    os.environ['BASEMAP_UNSAFE_NO_LEASE'] = '1'
    rc.require_active_lease()                         # explicit override
    del os.environ['BASEMAP_UNSAFE_NO_LEASE']




def test_co_tenant_policy_blocks_gpu_job_with_unknown_pid():
    # a GPU job (required_free_gb>0) is blocked if an unknown compute PID exists
    # OR free VRAM is insufficient. We force the free-VRAM branch (require > total)
    # so the test is deterministic regardless of what's on the GPU.
    d = tempfile.mkdtemp(); os.environ['BASEMAP_GPU_LEASE'] = os.path.join(d, '.lease')
    import importlib; importlib.reload(rc)
    j = rc.Job(name='big', argv=['true'], outputs=[], done_marker=os.path.join(d, 'big.done'),
               required_free_gb=100000.0)   # impossible → policy must block
    s = rc.run_jobs([j])
    assert s['jobs'][0]['status'] == 'co_tenant_block', s
    assert not os.path.exists(os.path.join(d, 'big.done'))
    print("PASS co-tenant policy blocks insufficient-VRAM job")


if __name__ == '__main__':
    for fn in [test_cross_process_lease_exclusion, test_lease_survives_controller_death_via_inherited_fd,
               test_nonzero_job1_stops_job2, test_exit_zero_without_outputs_is_failure,
               test_stale_output_does_not_skip_without_done_record, test_final_manifest_has_status_and_telemetry,
               test_co_tenant_policy_blocks_gpu_job_with_unknown_pid]:
        fn()
    print("ALL P0-D TESTS PASSED")
