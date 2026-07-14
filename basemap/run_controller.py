"""Single idempotent GPU run-controller with an atomic lease (P0.10).

Replaces the ad-hoc shell watchers/orchestrators that let multiple trainers +
scorers contend on one GPU (observed 2026-07-13: 4 co-tenants, 30.6/32.6 GB,
the coordination file listing only one). Guarantees:

- **Atomic lease**: exactly one GPU job runs at a time. `GpuLease` takes an
  exclusive `flock`; a second acquirer fails fast instead of co-launching.
- **Idempotent jobs**: each job declares a `done_marker`; if it exists the job
  is skipped. Two triggers firing on the same dependency cannot double-launch.
- **Co-tenancy record**: every launch snapshots free VRAM, GPU util, power,
  and the PIDs already on the GPU into the job manifest, so throughput
  evidence is never silently confounded again.

Not a scheduler — a deliberately small sequential controller. Submit an ordered
job list; each runs under the lease after its predecessors' markers appear.
"""
from __future__ import annotations
import os, sys, json, time, uuid, fcntl, signal, subprocess, datetime, dataclasses
from typing import Optional

LEASE_PATH = os.environ.get("BASEMAP_GPU_LEASE", "/data/latent-basemap/.gpu_lease")


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def gpu_snapshot() -> dict:
    """Free VRAM / util / power + the PIDs currently holding the GPU."""
    def q(fields, extra=()):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-{fields[0]}={fields[1]}",
                 "--format=csv,noheader,nounits", *extra], text=True, timeout=15)
            return [l.strip() for l in out.strip().splitlines() if l.strip()]
        except Exception as e:
            return [f"err:{e}"]
    gpu = q(("gpu", "memory.free,utilization.gpu,power.draw"))
    apps = q(("compute-apps", "pid,used_memory"))
    return {"at": _utcnow(), "gpu": gpu[0] if gpu else None,
            "compute_apps": apps, "n_co_tenants": len(apps)}


class GpuLease:
    """Exclusive, atomic, auto-releasing GPU lease via ``flock``.

    Usage::

        with GpuLease(timeout=0) as lease:   # non-blocking; raises if held
            ... launch GPU work ...

    ``timeout=None`` blocks until acquired; ``timeout=0`` fails immediately;
    ``timeout=N`` retries for N seconds. The lock is an OS advisory lock on
    ``LEASE_PATH`` — it survives only while this process holds the fd, so a
    crashed controller frees the GPU automatically.
    """
    def __init__(self, path: str = LEASE_PATH, timeout: Optional[float] = 0,
                 controller_id: Optional[str] = None):
        self.path = path
        self.timeout = timeout
        self.controller_id = controller_id or f"ctl-{uuid.uuid4().hex[:8]}"
        self._fd = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
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
                    raise RuntimeError(f"GPU lease held by [{held}]; not launching (P0.10).")
                time.sleep(1.0)

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


@dataclasses.dataclass
class Job:
    name: str
    argv: list          # process argv, run under the lease
    done_marker: str    # path whose existence means "already completed" (idempotent)
    cwd: Optional[str] = None
    log: Optional[str] = None
    manifest: Optional[str] = None   # where to write the co-tenancy manifest


def run_jobs(jobs: list, controller_id: Optional[str] = None) -> dict:
    """Run ``jobs`` in order, each under the atomic lease, idempotently.

    A job whose ``done_marker`` already exists is skipped. Each launched job
    gets a manifest with the pre-launch GPU snapshot + controller id. Returns a
    summary dict. One controller == one lease held across the whole batch, so no
    other controller/scorer can co-launch on this GPU while it runs.
    """
    cid = controller_id or f"ctl-{uuid.uuid4().hex[:8]}"
    summary = {"controller_id": cid, "started": _utcnow(), "jobs": []}
    with GpuLease(controller_id=cid, timeout=0):   # fail fast if another controller runs
        for job in jobs:
            rec = {"name": job.name}
            if os.path.exists(job.done_marker):
                rec["status"] = "skipped_idempotent"
                summary["jobs"].append(rec); continue
            snap = gpu_snapshot()
            rec["gpu_pre"] = snap
            if job.manifest:
                os.makedirs(os.path.dirname(job.manifest), exist_ok=True)
                json.dump({"controller_id": cid, "job": job.name, "argv": job.argv,
                           "gpu_pre": snap, "started": _utcnow()},
                          open(job.manifest, "w"), indent=1)
            logf = open(job.log, "w") if job.log else None
            t0 = time.time()
            p = subprocess.Popen(job.argv, cwd=job.cwd, stdout=logf, stderr=subprocess.STDOUT)
            rc = p.wait()
            rec["status"] = "ok" if rc == 0 else f"exit_{rc}"
            rec["seconds"] = round(time.time() - t0, 1)
            rec["done_marker_present"] = os.path.exists(job.done_marker)
            if logf:
                logf.close()
            summary["jobs"].append(rec)
    summary["finished"] = _utcnow()
    return summary


if __name__ == "__main__":
    # CLI: run_controller.py jobs.json  (a JSON list of Job dicts)
    spec = json.load(open(sys.argv[1]))
    jobs = [Job(**j) for j in spec]
    print(json.dumps(run_jobs(jobs), indent=1))
