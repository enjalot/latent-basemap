"""Single fail-closed GPU run-controller with an atomic, crash-safe lease (P0-D).

Guarantees one GPU job at a time and, critically, keeps the lease held for the
LIFETIME OF THE GPU CHILD even if the controller process is killed:

- the lease is an exclusive `flock` on an open file description; the controller
  passes that FD (inheritable) into the child and starts the child in its own
  process group, so the OS lock persists until the child exits even if the
  controller dies uncatchably. A new controller then cannot overlap the orphan.
- jobs are idempotent via an explicit controller-written `.done.json` completion
  record (NOT the output file), created only after exit-0 AND declared-output
  validation. A stale/partial output never counts as done.
- the chain STOPS on nonzero exit, missing outputs, or an invalid completion
  record, unless a job sets `continue_on_failure`.
- before a launch the controller enforces a co-tenant policy: unknown compute
  PIDs or insufficient free VRAM fail (or wait), rather than launching anyway.

Route ALL GPU entry points (training, scoring, graph/index validation, benches,
chains) through here; direct launches are unsafe.
"""
from __future__ import annotations
import os, sys, json, time, uuid, fcntl, signal, subprocess, datetime, dataclasses, hashlib
from typing import Optional

_DEFAULT_LEASE = "/data/latent-basemap/.gpu_lease"


def _lease_path() -> str:
    """Resolve the lease path at CALL time so BASEMAP_GPU_LEASE set after import
    (e.g. in tests, or per-launch) is honoured."""
    return os.environ.get("BASEMAP_GPU_LEASE", _DEFAULT_LEASE)


LEASE_PATH = _lease_path()   # back-compat module constant (import-time value)


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)   # atomic


def _file_sha(path: str, cap=1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            h.update(f.read(cap))
        return h.hexdigest()[:16]
    except Exception:
        return None


def _output_sig(path: str) -> Optional[dict]:
    """Content signature of an output (size + full-stream sha), or None if absent."""
    try:
        st = os.stat(path)
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return {"size": st.st_size, "sha": h.hexdigest()[:16]}
    except FileNotFoundError:
        return None


def _git_state():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        c = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True, timeout=10).strip()[:12]
        d = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True, timeout=10).strip())
        return f"{c}{'-dirty' if d else ''}"
    except Exception:
        return "nogit"


def _job_spec_digest(job) -> str:
    """Bind a done record to the job's FULL identity (P1): argv + cwd + declared
    outputs + the repo commit/dirty state + the content hashes of every declared
    input_path (config, scorer/trainer code, input artifacts). A changed scorer or
    config at the same argv therefore invalidates a stale done marker."""
    payload = json.dumps({
        "argv": list(job.argv), "cwd": job.cwd or "", "outputs": sorted(job.outputs),
        "code": _git_state(),
        "inputs": {p: _output_sig(p) for p in sorted(getattr(job, "input_paths", []) or [])},
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


_OWNED_LEASE_FDS = set()   # P1: fds of GpuLeases acquired in THIS process


def require_active_lease(path: str = None) -> None:
    """Refuse to run a GPU entry point unless THIS process OWNS or INHERITED the
    GPU lease (P1 — the old any-lock-holder-passes check was ownership-blind, so a
    stray direct process passed while a controller job held the lease). Proof is:
      (a) an in-process GpuLease this process acquired (registered fd whose inode
          matches the lease file), or
      (b) an inherited lease fd the controller passed via BASEMAP_GPU_LEASE_FD
          (open in this process, inode matches the lease file).
    Bypass only with BASEMAP_UNSAFE_NO_LEASE=1 for explicit unsafe diagnostics."""
    if os.environ.get("BASEMAP_UNSAFE_NO_LEASE") == "1":
        return
    path = path or _lease_path()
    try:
        lease_ino = os.stat(path).st_ino
    except OSError:
        lease_ino = None
    # (a) in-process owner
    for fd in list(_OWNED_LEASE_FDS):
        try:
            if lease_ino is not None and os.fstat(fd).st_ino == lease_ino:
                return
        except OSError:
            _OWNED_LEASE_FDS.discard(fd)
    # (b) inherited fd from a controller parent
    envfd = os.environ.get("BASEMAP_GPU_LEASE_FD")
    if envfd is not None and lease_ino is not None:
        try:
            if os.fstat(int(envfd)).st_ino == lease_ino:
                return
        except (OSError, ValueError):
            pass
    raise RuntimeError(
        f"no owned/inherited GPU lease for {path} — refuse to run a GPU entry point "
        f"(P1). Launch via run_controller (which passes BASEMAP_GPU_LEASE_FD) or hold "
        f"an in-process GpuLease; a lease held by another process does NOT count. Set "
        f"BASEMAP_UNSAFE_NO_LEASE=1 for an explicit unsafe diagnostic run.")


def gpu_snapshot() -> dict:
    def q(kind, fields, extra=()):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits", *extra],
                text=True, timeout=15)
            return [l.strip() for l in out.strip().splitlines() if l.strip()]
        except Exception as e:
            return [f"err:{e}"]
    gpu = q("gpu", "memory.free,memory.used,utilization.gpu,power.draw")
    apps = q("compute-apps", "pid,used_memory")
    pids = []
    for a in apps:
        try:
            pids.append(int(a.split(",")[0]))
        except Exception:
            pass
    free_mb = None
    try:
        free_mb = float(gpu[0].split(",")[0])
    except Exception:
        pass
    return {"at": _utcnow(), "gpu": gpu[0] if gpu else None, "compute_apps": apps,
            "compute_pids": pids, "free_mb": free_mb, "n_co_tenants": len(pids)}


class GpuLease:
    """Exclusive, atomic, crash-safe GPU lease. The lock is held by the OPEN
    FILE DESCRIPTION, so passing the fd to a child keeps it held past controller
    death. ``timeout=0`` fails fast, ``None`` blocks, ``N`` retries for N s."""
    def __init__(self, path: str = None, timeout: Optional[float] = 0,
                 controller_id: Optional[str] = None):
        self.path = path or _lease_path()
        self.timeout = timeout
        self.controller_id = controller_id or f"ctl-{uuid.uuid4().hex[:8]}"
        self._fd = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        os.set_inheritable(self._fd, True)   # child inherits → lock survives controller death
        deadline = None if self.timeout is None else time.time() + self.timeout
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.ftruncate(self._fd, 0)
                os.write(self._fd, f"{self.controller_id} pid={os.getpid()} at={_utcnow()}\n".encode())
                os.fsync(self._fd)
                _OWNED_LEASE_FDS.add(self._fd)   # P1: this process owns this lease
                return self
            except BlockingIOError:
                if deadline is not None and time.time() >= deadline:
                    held = os.pread(self._fd, 256, 0).decode(errors="replace").strip()
                    os.close(self._fd); self._fd = None
                    raise RuntimeError(f"GPU lease held by [{held}]; not launching (P0-D).")
                time.sleep(1.0)

    def fileno(self):
        return self._fd

    def release(self):
        if self._fd is not None:
            _OWNED_LEASE_FDS.discard(self._fd)
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd); self._fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


DEFAULT_SERVICE_MARKERS = ("ls-serve", "moonshine-web")   # known always-on viewers


def known_service_pids(markers=DEFAULT_SERVICE_MARKERS) -> list:
    """Compute PIDs whose /proc cmdline matches a KNOWN background-service marker
    (P0-5). Unlike snapshotting every observed PID, this allow-lists only named
    services by identity — an unknown training process is NOT auto-tolerated."""
    out = []
    for pid in gpu_snapshot()["compute_pids"]:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode(errors="replace")
            if any(m in cmd for m in markers):
                out.append(int(pid))
        except Exception:
            pass
    return out


def check_co_tenants(required_free_gb: float, allowed_pids=(), wait_s: float = 0) -> dict:
    """Enforce the co-tenant policy before a GPU launch. Fails (or waits up to
    ``wait_s``) if an unknown compute PID is present or free VRAM < requirement.
    ``allowed_pids`` are tolerated background services (e.g. a viewer)."""
    deadline = time.time() + wait_s
    allowed = set(int(p) for p in allowed_pids)
    while True:
        snap = gpu_snapshot()
        unknown = [p for p in snap["compute_pids"] if p not in allowed]
        free_gb = (snap["free_mb"] or 0) / 1024.0
        ok = (not unknown) and free_gb >= required_free_gb
        if ok or time.time() >= deadline:
            if not ok:
                raise RuntimeError(
                    f"co-tenant policy blocked launch: unknown GPU PIDs {unknown}, "
                    f"free {free_gb:.1f} GB < required {required_free_gb} GB (P0-D). snapshot={snap['gpu']}")
            return snap
        time.sleep(2.0)


@dataclasses.dataclass
class Job:
    name: str
    argv: list
    outputs: list                    # declared output paths; ALL must exist for success
    done_marker: str                 # controller-written completion record (.done.json)
    deps: list = dataclasses.field(default_factory=list)   # names that must be done first
    cwd: Optional[str] = None
    log: Optional[str] = None
    manifest: Optional[str] = None
    required_free_gb: float = 0.0
    continue_on_failure: bool = False
    input_paths: list = dataclasses.field(default_factory=list)   # P1: files whose
    # content binds the job identity (config, scorer/trainer code, input artifacts).
    # A change to any of them (or the repo commit) invalidates a stale done marker.
    certifying: bool = True   # S1: a certifying job MUST declare outputs; outputs=[]
    # is only allowed for an explicitly non-certifying job (certifying=False).
    predicted_wall_s: float = 0.0   # S2: predicted wall time; >CANARY_REQUIRED_WALL_S
    # forces a passing canary dependency before the run is admitted.
    canary_dep: Optional[str] = None   # S2: name of the perf-canary job this long
    # run depends on. Must also appear in `deps` so a sub-floor canary blocks it.


CANARY_REQUIRED_WALL_S = 600.0   # S2: >10 predicted minutes ⇒ canary is mandatory


def run_jobs(jobs: list, controller_id: Optional[str] = None, allowed_pids=(),
             summary_path: Optional[str] = None) -> dict:
    cid = controller_id or f"ctl-{uuid.uuid4().hex[:8]}"
    summary = {"controller_id": cid, "controller_pid": os.getpid(), "started": _utcnow(), "jobs": []}
    done = set()
    with GpuLease(controller_id=cid, timeout=0) as lease:
        for job in jobs:
            rec = {"name": job.name}
            # S1: a certifying job MUST declare outputs (exit-0 alone can't certify);
            # and every declared input MUST exist (fail on missing input).
            if job.certifying and not job.outputs:
                rec["status"] = "config_error:certifying_job_without_outputs"
                summary["jobs"].append(rec)
                if not job.continue_on_failure:
                    summary["stop_reason"] = f"{job.name}: certifying job declares no outputs (S1)"; break
                continue
            missing_inputs = [p for p in (job.input_paths or []) if not os.path.exists(p)]
            if missing_inputs:
                rec["status"] = f"missing_inputs:{missing_inputs}"
                summary["jobs"].append(rec)
                if not job.continue_on_failure:
                    summary["stop_reason"] = f"{job.name}: missing declared inputs {missing_inputs} (S1)"; break
                continue
            # S2: a certifying run predicted to exceed 10 minutes MUST depend on a
            # passing performance canary in this same batch (critique #4: nothing
            # made the canary a mandatory dependency). A sub-floor canary exits
            # non-zero → not in `done` → the deps check below blocks this run.
            if (job.certifying and job.predicted_wall_s > CANARY_REQUIRED_WALL_S
                    and (not job.canary_dep or job.canary_dep not in (job.deps or []))):
                rec["status"] = "config_error:long_run_without_canary_dep"
                rec["predicted_wall_s"] = job.predicted_wall_s
                summary["jobs"].append(rec)
                if not job.continue_on_failure:
                    summary["stop_reason"] = (f"{job.name}: predicted {job.predicted_wall_s:.0f}s "
                                              f"> {CANARY_REQUIRED_WALL_S:.0f}s needs a canary_dep "
                                              f"in deps (S2)"); break
                continue
            spec_digest = _job_spec_digest(job)
            # idempotency: skip only if the done record matches THIS job's spec
            # digest AND every output still matches its recorded signature (P0-5:
            # a changed argv/config or a mutated output invalidates the skip).
            if os.path.exists(job.done_marker):
                try:
                    m = json.load(open(job.done_marker))
                    sigs_ok = all(_output_sig(o) == (m.get("output_sigs") or {}).get(o)
                                  and _output_sig(o) is not None for o in job.outputs)
                    if (m.get("status") == "ok" and m.get("spec_digest") == spec_digest
                            and sigs_ok):
                        rec["status"] = "skipped_done"; done.add(job.name)
                        summary["jobs"].append(rec); continue
                except Exception:
                    pass  # partial/corrupt marker → re-run
            # dependencies
            missing = [d for d in job.deps if d not in done]
            if missing:
                rec["status"] = f"blocked_deps:{missing}"
                summary["jobs"].append(rec)
                if not job.continue_on_failure:
                    summary["stop_reason"] = f"unsatisfied deps for {job.name}"; break
                continue
            # co-tenant policy — enforced only for jobs that declare a GPU need
            # (required_free_gb>0); required_free_gb==0 marks a CPU job.
            try:
                snap_pre = (check_co_tenants(job.required_free_gb, allowed_pids)
                            if job.required_free_gb > 0 else gpu_snapshot())
            except RuntimeError as e:
                rec["status"] = "co_tenant_block"; rec["error"] = str(e)
                summary["jobs"].append(rec)
                if not job.continue_on_failure:
                    summary["stop_reason"] = str(e); break
                continue
            rec["gpu_pre"] = snap_pre
            if job.manifest:
                _atomic_write_json(job.manifest, {"controller_id": cid, "job": job.name,
                                                  "argv": job.argv, "gpu_pre": snap_pre,
                                                  "status": "running", "started": _utcnow()})
            # snapshot pre-existing output signatures — an exit-0 no-op that
            # leaves a stale output unchanged must NOT be certified (P0-5).
            pre_sigs = {o: _output_sig(o) for o in job.outputs}
            logf = open(job.log, "w") if job.log else None
            t0 = time.time()
            # child in its own process group. pass_fds keeps the inherited lease
            # fd open (lock survives controller death); close_fds stays default
            # True so no other fds leak (the old close_fds=False warned every run).
            # P1: pass the inherited lease fd number so the child's
            # require_active_lease() can PROVE ownership (inode match), not merely
            # observe the global lock is held.
            child_env = dict(os.environ)
            if lease.fileno() is not None:
                child_env["BASEMAP_GPU_LEASE_FD"] = str(lease.fileno())
            p = subprocess.Popen(job.argv, cwd=job.cwd, stdout=logf, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=child_env,
                                 pass_fds=(lease.fileno(),) if lease.fileno() is not None else ())
            rec["child_pid"] = p.pid; rec["pgid"] = os.getpgid(p.pid)
            rc = p.wait()
            if logf:
                logf.close()
            post_sigs = {o: _output_sig(o) for o in job.outputs}
            outs_ok = all(post_sigs[o] is not None for o in job.outputs)
            # every declared output must be freshly produced or changed vs pre-run
            fresh_ok = all(post_sigs[o] is not None and post_sigs[o] != pre_sigs[o]
                           for o in job.outputs)
            stale = outs_ok and not fresh_ok
            success = (rc == 0) and outs_ok and fresh_ok
            rec["status"] = ("ok" if success else
                             ("exit_%d" % rc if rc != 0 else
                              "stale_outputs" if stale else "missing_outputs"))
            rec["seconds"] = round(time.time() - t0, 1)
            rec["outputs_present"] = outs_ok
            snap_post = gpu_snapshot()
            final = {"controller_id": cid, "job": job.name, "argv": job.argv,
                     "status": rec["status"], "exit_code": rc, "seconds": rec["seconds"],
                     "spec_digest": spec_digest, "output_sigs": post_sigs,
                     "gpu_pre": snap_pre, "gpu_post": snap_post, "finished": _utcnow()}
            if job.manifest:
                _atomic_write_json(job.manifest, final)
            if success:
                # completion record is controller-written, AFTER validation, and
                # bound to the spec digest + output signatures.
                _atomic_write_json(job.done_marker, {"status": "ok", "job": job.name,
                                                     "finished": _utcnow(), "exit_code": 0,
                                                     "spec_digest": spec_digest,
                                                     "output_sigs": post_sigs})
                done.add(job.name)
            summary["jobs"].append(rec)
            if not success and not job.continue_on_failure:
                summary["stop_reason"] = f"{job.name} failed ({rec['status']}) — chain stopped"; break
    summary.setdefault("stop_reason", "completed")
    summary["finished"] = _utcnow()
    if summary_path:
        _atomic_write_json(summary_path, summary)
    return summary


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    jobs = [Job(**j) for j in spec.get("jobs", spec)]
    out = run_jobs(jobs, allowed_pids=spec.get("allowed_pids", ()) if isinstance(spec, dict) else (),
                   summary_path=sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(out, indent=1))
