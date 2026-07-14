"""P0.10: atomic GPU lease + idempotent jobs."""
import sys, os, tempfile, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from basemap.run_controller import GpuLease, run_jobs, Job


def test_lease_is_exclusive():
    lease_path = tempfile.mktemp(suffix='.lease')
    a = GpuLease(path=lease_path, timeout=0)
    a.acquire()
    # second non-blocking acquirer must fail fast
    b = GpuLease(path=lease_path, timeout=0)
    try:
        b.acquire()
        assert False, "second lease should not have been granted"
    except RuntimeError as e:
        assert "held by" in str(e)
    a.release()
    # now it can be acquired
    b.acquire(); b.release()
    print("PASS lease exclusivity + release")


def test_lease_context_manager_releases_on_exit():
    lp = tempfile.mktemp(suffix='.lease')
    with GpuLease(path=lp, timeout=0):
        pass
    with GpuLease(path=lp, timeout=0):  # would raise if not released
        pass
    print("PASS context-manager release")


def test_idempotent_job_skipped():
    d = tempfile.mkdtemp()
    marker = os.path.join(d, 'done')
    open(marker, 'w').close()  # pretend already done
    os.environ['BASEMAP_GPU_LEASE'] = os.path.join(d, '.lease')
    import importlib, basemap.run_controller as rc
    importlib.reload(rc)
    job = rc.Job(name='x', argv=['false'], done_marker=marker)  # 'false' would fail if run
    s = rc.run_jobs([job])
    assert s['jobs'][0]['status'] == 'skipped_idempotent', s
    print("PASS idempotent skip (job with existing marker not launched)")


def test_job_runs_and_records():
    d = tempfile.mkdtemp()
    marker = os.path.join(d, 'out.txt')
    os.environ['BASEMAP_GPU_LEASE'] = os.path.join(d, '.lease')
    import importlib, basemap.run_controller as rc
    importlib.reload(rc)
    job = rc.Job(name='touch', argv=['bash', '-c', f'echo hi > {marker}'],
                 done_marker=marker, manifest=os.path.join(d, 'manifest.json'))
    s = rc.run_jobs([job])
    assert s['jobs'][0]['status'] == 'ok', s
    assert os.path.exists(marker)
    import json; m = json.load(open(os.path.join(d, 'manifest.json')))
    assert 'gpu_pre' in m and 'controller_id' in m
    print("PASS job runs + manifest records gpu snapshot")


if __name__ == '__main__':
    test_lease_is_exclusive()
    test_lease_context_manager_releases_on_exit()
    test_idempotent_job_skipped()
    test_job_runs_and_records()
    print("ALL RUN-CONTROLLER TESTS PASSED")
