"""S2 — canary performance instrumentation (closure review §S2).

`CanaryProfiler` turns the admission canary into a regression *contract*: it
partitions the post-warmup bench window into >=5 rate windows, times the three
hot phases (sample+gather+H2D, forward+loss, backward+optimizer) with SAMPLED
CUDA events (never a per-step sync), snapshots peak VRAM/RSS/util/power/IO/
co-tenants/lease, and — the safety property — ABORTS after `subfloor_patience`
consecutive windows below the throughput floor. A 7x slowdown trips this in a
handful of windows instead of wasting a 28-minute run.

The profiler is only attached in canary/profile mode, so production steps are
untouched. On CPU (tests) it degrades to perf_counter timing so the abort logic
is verifiable without a GPU.
"""
from __future__ import annotations
import os, time, resource, subprocess, logging, statistics


def _cuda_ok(device):
    try:
        import torch
        return torch.cuda.is_available() and "cuda" in str(device)
    except Exception:
        return False


class _EventPair:
    """A sampled phase measurement: a CUDA event pair (GPU) or wall stamps (CPU)."""
    __slots__ = ("phase", "start", "end", "cuda", "_t0", "_t1")

    def __init__(self, phase, cuda):
        self.phase, self.cuda = phase, cuda
        if cuda:
            import torch
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
        self._t0 = self._t1 = None

    def mark_start(self):
        if self.cuda:
            self.start.record()
        else:
            self._t0 = time.perf_counter()

    def mark_end(self):
        if self.cuda:
            self.end.record()
        else:
            self._t1 = time.perf_counter()

    def elapsed_ms(self):
        if self.cuda:
            return self.start.elapsed_time(self.end)   # forces caller-side sync
        return (self._t1 - self._t0) * 1e3


class _NullPhase:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class CanaryProfiler:
    def __init__(self, *, warmup, max_steps, floor, warn_rate=None, device="cuda",
                 n_windows=6, subfloor_patience=3, sample_every=25, baseline_key=None):
        self.warmup = int(warmup)
        self.max_steps = int(max_steps)
        self.floor = float(floor)
        self.warn_rate = float(warn_rate) if warn_rate else None
        self.device = device
        self.cuda = _cuda_ok(device)
        self.sample_every = max(1, int(sample_every))
        self.subfloor_patience = int(subfloor_patience)
        self.baseline_key = baseline_key or {}
        span = max(1, self.max_steps - self.warmup)
        n_windows = max(5, int(n_windows))
        # boundary update-counts (positive-LR updates) that close each window
        step = span / n_windows
        self.bounds = [self.warmup + int(round(step * (i + 1))) for i in range(n_windows)]
        self.bounds[-1] = self.max_steps
        # per-window accumulators
        self._win_idx = 0
        self._win_t0 = None
        self._win_u0 = None
        self.windows = []              # [{idx, updates, wall_s, rate}]
        self.consecutive_subfloor = 0
        self.abort = False
        self.abort_reason = None
        # phase timing (sampled)
        self._pending = []             # unread _EventPair list, drained at window close
        self._phase_ms = {}            # phase -> [ms samples]
        self._active = None
        self._rusage0 = resource.getrusage(resource.RUSAGE_SELF)
        self._t_start = time.perf_counter()

    # ── phase timing (context managers, sampled) ────────────────────────────
    def phase(self, name, global_step):
        """Time `name` on sampled steps only; a no-op context otherwise."""
        if global_step <= self.warmup or (global_step % self.sample_every) != 0:
            return _NullPhase()
        ev = _EventPair(name, self.cuda)
        self._active = ev
        return self  # __enter__/__exit__ below drive this ev

    def __enter__(self):
        self._active.mark_start()
        return self

    def __exit__(self, *exc):
        self._active.mark_end()
        self._pending.append(self._active)
        self._active = None
        return False

    def _drain_phases(self):
        # One sync per window (reading a CUDA event forces it); never per-step.
        if self.cuda and self._pending:
            import torch
            torch.cuda.synchronize(self.device)
        for ev in self._pending:
            self._phase_ms.setdefault(ev.phase, []).append(ev.elapsed_ms())
        self._pending = []

    # ── rate-window bookkeeping (called once per successful update) ──────────
    def on_update(self, global_step, positive_updates):
        """Advance window state; returns True if training must ABORT now."""
        if positive_updates <= self.warmup:
            return False
        if self._win_t0 is None:                       # first post-warmup update
            if self.cuda:
                import torch; torch.cuda.synchronize(self.device)
            self._win_t0 = time.perf_counter()
            self._win_u0 = positive_updates
            return False
        if positive_updates >= self.bounds[self._win_idx]:
            if self.cuda:
                import torch; torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            du = positive_updates - self._win_u0
            dt = now - self._win_t0
            rate = du / dt if dt > 0 else 0.0
            self._drain_phases()
            self.windows.append({"idx": self._win_idx, "updates": du,
                                 "wall_s": round(dt, 4), "rate": round(rate, 1)})
            if rate < self.floor:
                self.consecutive_subfloor += 1
                logging.warning("S2 canary: window %d rate %.0f upd/s < floor %.0f "
                                "(%d consecutive sub-floor)", self._win_idx, rate,
                                self.floor, self.consecutive_subfloor)
                if self.consecutive_subfloor >= self.subfloor_patience:
                    self.abort = True
                    self.abort_reason = (f"{self.consecutive_subfloor} consecutive windows "
                                         f"below floor {self.floor:.0f} upd/s "
                                         f"(last {rate:.0f}); aborting to avoid a wasted run")
                    return True
            else:
                if self.warn_rate and rate < self.warn_rate:
                    logging.warning("S2 canary: window %d rate %.0f upd/s below warn %.0f",
                                    self._win_idx, rate, self.warn_rate)
                self.consecutive_subfloor = 0
            self._win_idx = min(self._win_idx + 1, len(self.bounds) - 1)
            self._win_t0 = now
            self._win_u0 = positive_updates
        return False

    # ── environment snapshot ────────────────────────────────────────────────
    def _gpu_snapshot(self):
        snap = {}
        try:
            import torch
            if self.cuda:
                snap["peak_vram_alloc_gb"] = round(torch.cuda.max_memory_allocated(self.device) / 1e9, 3)
                snap["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved(self.device) / 1e9, 3)
        except Exception:
            pass
        # nvidia-smi: utilization/power + co-tenant compute PIDs (best-effort, once).
        try:
            u = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if u.returncode == 0:
                parts = [p.strip() for p in u.stdout.splitlines()[0].split(",")]
                snap["gpu_util_pct"], snap["power_w"], snap["mem_used_mib"] = (
                    float(parts[0]), float(parts[1]), float(parts[2]))
            a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if a.returncode == 0:
                pids = [ln.split(",")[0].strip() for ln in a.stdout.splitlines() if ln.strip()]
                mypid = str(os.getpid())
                snap["co_tenant_pids"] = [p for p in pids if p != mypid]
                snap["n_compute_apps"] = len(pids)
        except Exception:
            pass
        return snap

    def finalize(self, *, bench_seconds=None, setup_seconds=None):
        self._drain_phases()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rates = [w["rate"] for w in self.windows]
        phase_ms = {k: round(statistics.median(v), 3) for k, v in self._phase_ms.items() if v}
        out = {
            "baseline_key": self.baseline_key,
            "floor": self.floor, "warn_rate": self.warn_rate,
            "n_windows": len(self.windows), "rate_windows": self.windows,
            "rate_median": round(statistics.median(rates), 1) if rates else None,
            "rate_min": round(min(rates), 1) if rates else None,
            "rate_max": round(max(rates), 1) if rates else None,
            "phase_ms_median": phase_ms,
            "phase_samples": {k: len(v) for k, v in self._phase_ms.items()},
            "aborted": self.abort, "abort_reason": self.abort_reason,
            "setup_seconds": round(setup_seconds, 3) if setup_seconds else None,
            "bench_seconds": round(bench_seconds, 4) if bench_seconds else None,
            "rss_peak_gb": round(ru.ru_maxrss / 1024**2, 3),
            "minor_faults": ru.ru_minflt - self._rusage0.ru_minflt,
            "major_faults": ru.ru_majflt - self._rusage0.ru_majflt,
            "block_in": ru.ru_inblock - self._rusage0.ru_inblock,
            "block_out": ru.ru_oublock - self._rusage0.ru_oublock,
            "lease_id": os.environ.get("BASEMAP_GPU_LEASE_FD"),
            "controller_id": os.environ.get("BASEMAP_CONTROLLER_ID"),
            **self._gpu_snapshot(),
        }
        # A regressed phase is the one whose median dominates the step budget.
        if phase_ms:
            worst = max(phase_ms.items(), key=lambda kv: kv[1])
            out["dominant_phase"] = worst[0]
            total = sum(phase_ms.values())
            out["phase_fractions"] = {k: round(v / total, 3) for k, v in phase_ms.items()} if total else {}
        return out


def build_baseline_key(*, model, n, d, n_edges, batch_size, use_amp, kernel,
                       pipeline_info, device="cuda"):
    """Content key that makes a canary rate comparable ONLY across identical
    hardware/driver/torch + problem shape + architecture + pipeline/semantics."""
    key = {"n": int(n), "d": int(d), "n_edges": int(n_edges),
           "hidden_dim": getattr(model, "hidden_dim", None),
           "n_components": getattr(model, "n_components", None),
           "n_layers": getattr(model, "n_layers", None),
           "batch_size": int(batch_size), "use_amp": bool(use_amp), "kernel": kernel,
           "pipeline": pipeline_info.get("pipeline"),
           "sampler_class": pipeline_info.get("sampler_class"),
           "positive_sampling": pipeline_info.get("positive_sampling"),
           "x_residency": pipeline_info.get("x_residency")}
    try:
        import torch
        key["torch"] = torch.__version__
        if _cuda_ok(device):
            key["gpu_name"] = torch.cuda.get_device_name(device)
            key["cuda"] = torch.version.cuda
            cc = torch.cuda.get_device_capability(device)
            key["compute_capability"] = f"{cc[0]}.{cc[1]}"
    except Exception:
        pass
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if drv.returncode == 0:
            key["driver"] = drv.stdout.splitlines()[0].strip()
    except Exception:
        pass
    return key
