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

LEASE_PATH = os.environ.get("BASEMAP_GPU_LEASE", "/data/latent-basemap/.gpu_lease")


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
    def __init__(self, path: str = LEASE_PATH, timeout: Optional[float] = 0,
                 controller_id: Optional[str] = None):
        self.path = path
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
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd); self._fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


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


def run_jobs(jobs: list, controller_id: Optional[str] = None, allowed_pids=(),
             summary_path: Optional[str] = None) -> dict:
    cid = controller_id or f"ctl-{uuid.uuid4().hex[:8]}"
    summary = {"controller_id": cid, "controller_pid": os.getpid(), "started": _utcnow(), "jobs": []}
    done = set()
    with GpuLease(controller_id=cid, timeout=0) as lease:
        for job in jobs:
            rec = {"name": job.name}
            # idempotency: a VALID completion record (not a stray output) skips.
            if os.path.exists(job.done_marker):
                try:
                    m = json.load(open(job.done_marker))
                    if m.get("status") == "ok" and all(os.path.exists(o) for o in job.outputs):
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
            logf = open(job.log, "w") if job.log else None
            t0 = time.time()
            # child in its own process group, inheriting the lease fd (lock survives us)
            p = subprocess.Popen(job.argv, cwd=job.cwd, stdout=logf, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=False,
                                 pass_fds=(lease.fileno(),) if lease.fileno() is not None else ())
            rec["child_pid"] = p.pid; rec["pgid"] = os.getpgid(p.pid)
            rc = p.wait()
            if logf:
                logf.close()
            outs_ok = all(os.path.exists(o) for o in job.outputs)
            success = (rc == 0) and outs_ok
            rec["status"] = "ok" if success else ("exit_%d" % rc if rc != 0 else "missing_outputs")
            rec["seconds"] = round(time.time() - t0, 1)
            rec["outputs_present"] = outs_ok
            snap_post = gpu_snapshot()
            final = {"controller_id": cid, "job": job.name, "argv": job.argv,
                     "status": rec["status"], "exit_code": rc, "seconds": rec["seconds"],
                     "outputs": {o: _file_sha(o) for o in job.outputs},
                     "gpu_pre": snap_pre, "gpu_post": snap_post, "finished": _utcnow()}
            if job.manifest:
                _atomic_write_json(job.manifest, final)
            if success:
                # completion record is controller-written, AFTER validation.
                _atomic_write_json(job.done_marker, {"status": "ok", "job": job.name,
                                                     "finished": _utcnow(), "exit_code": 0})
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
