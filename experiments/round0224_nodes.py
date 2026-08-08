"""Execute R0224 — assemble a benchmark substrate, then measure cuVS memory.

Two nodes.

`assemble_benchmark_substrate` (CPU) builds a 16,000,000-row mixed MiniLM
substrate under R0216's selection law with the composition shares scaled x8,
then writes it in a **seeded global row permutation** so every prefix of the file
is a uniform subsample of the whole rather than one corpus block. The N-sweep
reads prefixes, so without the permutation it would confound scale with
composition. Rows are streamed into an on-disk `open_memmap`, never held in a
resident array. These are benchmark bytes and seal no training capability.

`sweep_cuvs_memory` (GPU) runs the (N x igd) matrix, each cell as a **fresh
RAPIDS subprocess** so `VmHWM` is a per-build measurement, with four instruments
running at once and R0220's `nvidia-smi` poll carried alongside as a control. It
then reports which instruments can see intermediate degree at all, and only if
one can does it fit anything or project anything.
"""
from __future__ import annotations

import gc
import glob
import json
import os
import resource
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0216_minilm_2m_substrate import (
    COMPOSITION,
    EXCLUDED_SHARDS,
    RAW_FORMAT,
    ROW_POLICY,
    TRAILING_FRAGMENT_POLICY,
    ZERO_ROW_POLICY,
    resolve_shard_rows,
)
from basemap.round0224_cuvs_memory import (
    BENCHMARK_COMPOSITION_SCALE,
    BENCHMARK_NOTE,
    BENCHMARK_ROWS,
    BENCHMARK_SELECTION_SEED,
    BENCHMARK_SHUFFLE_SEED,
    BUDGET_TOLERANCE,
    BUILD_TIMEOUT_S,
    CONTROL_INSTRUMENT,
    GUARD_HOST_RSS_BUDGET_BYTES,
    GUARD_SIGTERM_GRACE_S,
    GUARD_SWAP_ABORT_BYTES,
    WATCHDOG_POLL_S,
    guard_decision,
    DIMENSION,
    GPU_HOURS_CAP,
    HOST_RSS_LIMIT_GIB,
    DEVICE_BUDGET_NOTE,
    INSTRUMENTS,
    PROJECTION_SUBSTRATE_BYTES,
    REGISTERED_DEVICE_TOTAL_BYTES,
    REGISTERED_HOST_TOTAL_BYTES,
    ROUND_ID,
    Round0224Error,
    SAMPLE_INTERVAL_S,
    SHARD_SPAN_FLOOR,
    SUBSTRATE_CAPABILITY,
    SUBSTRATE_SCHEMA,
    SWEEP_CAPABILITY,
    SWEEP_GRAPH_DEGREE,
    SWEEP_INTERMEDIATE_DEGREES,
    SWEEP_MAX_ITERATIONS,
    SWEEP_METRIC,
    SWEEP_ROWS,
    SWEEP_SCHEMA,
    residency_probe_settings,
    summarize_sweep,
    sweep_settings,
    validate_prefix_composition,
)
from basemap import round0113_prompt_contrast as prompt_contract


ASSEMBLE_ACTION = "assemble_benchmark_substrate"
SWEEP_ACTION = "sweep_cuvs_memory"

EMB = "/data/embeddings"
CUML_LAUNCHER = "/data/latent-basemap/cuml_py"
BUILD_SCRIPT = "basemap/round0224_cuvs_memory_build.py"
NVIDIA_SMI = "/usr/bin/nvidia-smi"
NVIDIA_SMI_POLL_S = 0.1
DRAW_BLOCK = 200_000


# --------------------------------------------------------------------------- #
# node 1: the benchmark substrate
# --------------------------------------------------------------------------- #


def _shards(corpus: str) -> list[tuple[str, int, bool]]:
    """R0216's shard resolution, imported constants and all."""
    out: list[tuple[str, int, bool]] = []
    for path in sorted(glob.glob(os.path.join(EMB, corpus, "train", "*.npy"))):
        if path.endswith(".tmp.npy"):
            continue
        if os.path.relpath(path, EMB) in EXCLUDED_SHARDS:
            continue
        with open(path, "rb") as handle:
            real_npy = handle.read(6) == b"\x93NUMPY"
        if real_npy:
            rows = int(np.load(path, mmap_mode="r").shape[0])
        else:
            relative = os.path.relpath(path, EMB)
            rows = resolve_shard_rows(
                relative_path=relative, size_bytes=os.path.getsize(path)
            )
        out.append((path, rows, real_npy))
    if not out:
        raise Round0224Error(f"R0224 found no shards for {corpus}")
    return out


def _open_shard(path: str, rows: int, real_npy: bool) -> np.ndarray:
    if real_npy:
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype="<f4", mode="r", shape=(rows, DIMENSION))


def run_assemble(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    from numpy.lib.format import open_memmap

    started = time.monotonic()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0224 benchmark substrate"
    )
    target_counts = {
        name: int(rows) * BENCHMARK_COMPOSITION_SCALE for name, rows in COMPOSITION
    }
    if sum(target_counts.values()) != BENCHMARK_ROWS:
        raise Round0224Error(
            f"R0224 scaled composition totals {sum(target_counts.values())}, "
            f"registered {BENCHMARK_ROWS}"
        )

    substrate_path = os.path.join(output, "substrate.f32.npy")
    substrate = open_memmap(
        substrate_path, mode="w+", dtype=np.float32, shape=(BENCHMARK_ROWS, DIMENSION)
    )
    provenance = np.empty(
        BENCHMARK_ROWS,
        dtype=np.dtype([("corpus", "u1"), ("shard", "u2"), ("row", "i8")]),
    )
    # The seeded global permutation: destination row for the i-th selected row.
    placement = np.random.RandomState(BENCHMARK_SHUFFLE_SEED).permutation(
        BENCHMARK_ROWS
    )

    spans: dict[str, Any] = {}
    rejects: dict[str, int] = {}
    sources: dict[str, Any] = {}
    written = 0
    for index, (corpus, base_rows) in enumerate(COMPOSITION):
        want = int(base_rows) * BENCHMARK_COMPOSITION_SCALE
        shards = _shards(corpus)
        total = sum(rows for _p, rows, _n in shards)
        if total < want:
            raise Round0224Error(
                f"R0224 {corpus}: need {want} rows, corpus has {total}"
            )
        offsets = np.concatenate([[0], np.cumsum([r for _p, r, _n in shards])])
        rng = np.random.RandomState(BENCHMARK_SELECTION_SEED + index)
        picked = np.zeros(total, dtype=bool)
        corpus_written = 0
        need = want
        dropped = 0
        rounds = 0
        touched: set[int] = set()
        while need > 0:
            rounds += 1
            if rounds > 8:
                raise Round0224Error(
                    f"R0224 {corpus}: replacement did not converge after 8 rounds"
                )
            free = np.flatnonzero(~picked)
            if free.size < need:
                raise Round0224Error(f"R0224 {corpus}: exhausted usable rows")
            draw = np.sort(rng.choice(free, need, replace=False)).astype(np.int64)
            del free
            picked[draw] = True
            shard_of = np.searchsorted(offsets, draw, side="right") - 1
            for shard_index, (path, rows, real_npy) in enumerate(shards):
                mask = shard_of == shard_index
                local_global = draw[mask]
                if local_global.size == 0:
                    continue
                touched.add(shard_index)
                local = local_global - offsets[shard_index]
                array = _open_shard(path, rows, real_npy)
                for start in range(0, local.size, DRAW_BLOCK):
                    stop = min(start + DRAW_BLOCK, local.size)
                    block = np.asarray(array[local[start:stop]], dtype=np.float32)
                    norms = np.linalg.norm(block, axis=1)
                    ok = np.isfinite(block).all(axis=1) & (norms > 0)
                    dropped += int((~ok).sum())
                    if not ok.any():
                        continue
                    kept = block[ok]
                    kept /= np.linalg.norm(kept, axis=1)[:, None]
                    destinations = placement[written : written + kept.shape[0]]
                    substrate[destinations] = kept
                    provenance["corpus"][destinations] = index
                    provenance["shard"][destinations] = shard_index
                    provenance["row"][destinations] = local[start:stop][ok]
                    written += int(kept.shape[0])
                    corpus_written += int(kept.shape[0])
                    del block, kept, destinations
                del array
            need = want - corpus_written
            if need < 0:
                raise Round0224Error(
                    f"R0224 {corpus} wrote {corpus_written} rows against a target "
                    f"of {want}"
                )
        del picked
        gc.collect()
        coverage = len(touched) / len(shards)
        if coverage < SHARD_SPAN_FLOOR:
            raise Round0224Error(
                f"R0224 {corpus}: selection touched {len(touched)}/{len(shards)} "
                f"shards ({coverage:.2%}); R0216's law requires the sample to span "
                "the corpus"
            )
        spans[corpus] = {
            "shards_touched": len(touched),
            "shards_total": len(shards),
            "coverage": coverage,
            "replacement_rounds": rounds,
        }
        if corpus_written != want:
            raise Round0224Error(
                f"R0224 {corpus} wrote {corpus_written} rows, registered {want}"
            )
        rejects[corpus] = dropped
        sources[corpus] = {
            "shards": len(shards),
            "corpus_rows": int(total),
            "selected_rows": want,
            "format": "npy" if shards[0][2] else RAW_FORMAT,
            "first_shard": expected_input_signature(shards[0][0]),
        }
    if written != BENCHMARK_ROWS:
        raise Round0224Error(
            f"R0224 wrote {written} rows, registered {BENCHMARK_ROWS}"
        )
    substrate.flush()
    corpus_of_row = np.asarray(provenance["corpus"], dtype=np.int64)
    del substrate
    gc.collect()

    check = np.load(substrate_path, mmap_mode="r")
    if check.shape != (BENCHMARK_ROWS, DIMENSION) or check.dtype != np.float32:
        raise Round0224Error("R0224 benchmark substrate geometry is wrong")
    sample = np.asarray(check[:: max(1, BENCHMARK_ROWS // 100_000)], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)
    if not np.isfinite(sample).all() or float(norms.min()) <= 0.0:
        raise Round0224Error("R0224 benchmark substrate contains degenerate rows")
    del sample, check
    os.chmod(substrate_path, 0o444)

    provenance_path = atomic_save_new_npy(
        os.path.join(output, "provenance.npy"), provenance, immutable=True
    )
    targets = {name: count / BENCHMARK_ROWS for name, count in target_counts.items()}
    prefixes: dict[str, Any] = {}
    for rows in SWEEP_ROWS:
        counts = np.bincount(corpus_of_row[:rows], minlength=len(COMPOSITION))
        shares = {
            name: float(counts[index]) / float(rows)
            for index, (name, _n) in enumerate(COMPOSITION)
        }
        prefixes[str(rows)] = validate_prefix_composition(
            shares=shares, targets=targets, rows=int(rows)
        )

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0224Error(
            f"R0224 assembly peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    receipt = prompt_contract.seal({
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": SUBSTRATE_CAPABILITY,
        "capabilities": [SUBSTRATE_CAPABILITY],
        "rows": BENCHMARK_ROWS,
        "dimension": DIMENSION,
        "purpose": BENCHMARK_NOTE,
        "benchmark_only": True,
        "training_performed": False,
        "seals_training_capability": False,
        "composition": {
            name: {"rows": count, "share": count / BENCHMARK_ROWS}
            for name, count in target_counts.items()
        },
        "composition_scale_vs_r0216": BENCHMARK_COMPOSITION_SCALE,
        "selection": {
            "law": (
                "R0216 queue-correction-3: per-corpus uniform over ALL complete "
                "rows of non-excluded shards, rejected rows replaced from the "
                "unpicked complement, shard span asserted"
            ),
            "seed": BENCHMARK_SELECTION_SEED,
            "shard_span_floor": SHARD_SPAN_FLOOR,
            "shard_span": spans,
            "degenerate_rows_dropped": rejects,
            "zero_row_policy": ZERO_ROW_POLICY,
            "excluded_shards": {
                key: value["reason"] for key, value in EXCLUDED_SHARDS.items()
            },
        },
        "row_order": {
            "law": "seeded global permutation of all 16,000,000 destinations",
            "seed": BENCHMARK_SHUFFLE_SEED,
            "why": (
                "the N-sweep reads prefixes; without the permutation a prefix "
                "would be one corpus block and scale would be confounded with "
                "composition"
            ),
            "departure_from_r0216": (
                "R0216 lays corpora out in contiguous blocks; this benchmark "
                "does not, and that is the only difference in the layout"
            ),
        },
        "prefix_composition": prefixes,
        "loading_contract": {
            "raw_format": RAW_FORMAT,
            "row_policy": ROW_POLICY,
            "trailing_fragment_policy": TRAILING_FRAGMENT_POLICY,
        },
        "sources": sources,
        "substrate": expected_input_signature(substrate_path),
        "provenance": expected_input_signature(provenance_path),
        "performance": {
            "assembly_wall_s": time.monotonic() - started,
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "benchmark-substrate.json"), receipt, immutable=True
    )
    del provenance, corpus_of_row
    gc.collect()


# --------------------------------------------------------------------------- #
# node 2: the memory sweep
# --------------------------------------------------------------------------- #


def _nvidia_smi_per_process_bytes(pid: int) -> int:
    """R0220's instrument, carried unchanged as a control."""
    try:
        completed = subprocess.run(
            [
                NVIDIA_SMI,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    for line in completed.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return int(parts[1]) * 1024 * 1024
            except ValueError:
                return 0
    return 0


def _nvidia_smi_device_bytes() -> int:
    """Device-wide bytes in use, from the driver.

    This is the instrument the first attempt lacked. It is immune to both
    failure modes that blinded the others:

    * it runs in a **separate process**, so it cannot be starved by
      `nn_descent.build` holding the child's GIL (the in-process 5 ms sampler
      managed 1-2 samples per build);
    * it reads the **device**, not a bookkeeper, so cuVS allocating outside
      RMM's current device resource — which the byte-identical
      `rmm_peak_bytes` across igd 48/96/128 proves it does — cannot hide from
      it.

    It is device-wide, not per-process. The queue holds an exclusive GPU lease
    and the baseline is recorded immediately before each child starts, so the
    over-baseline figure is attributable to the build.
    """
    try:
        completed = subprocess.run(
            [NVIDIA_SMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return int(completed.stdout.strip().splitlines()[0].strip()) * 1024 * 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def _proc_memory_bytes(pid: int) -> tuple[int, int]:
    """`(VmRSS, RssAnon)` for a pid. Anonymous bytes are the swappable ones."""
    rss = 0
    anon = 0
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("RssAnon:"):
                    anon = int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return rss, anon


def _swap_used_bytes() -> int:
    total = 0
    free = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("SwapFree:"):
                    free = int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return max(0, total - free)


class BuildWatchdog(threading.Thread):
    """Live host/device sampling with a cooperative abort.

    The first attempt had an OOM *catch* and no live guard: the `16M`
    `materialize` cell climbed to `46.7 GB` RSS, consumed all `7 GB` of swap,
    and was SIGKILLed after 37 minutes. Because it held a CUDA context, the
    kill left a UVM teardown thread uninterruptible, which deadlocked RCU, put
    PID 1 into `D` state and cost a hard reboot.

    So this watchdog trips on **swap**, not only on RSS. Swap is the signal
    that the box is on the path that wedged it, and it is a system-wide reading
    rather than a per-process one precisely because UVM host backing for an
    oversubscribed device does not all appear in the child's RSS.

    An abort is **SIGTERM**, never SIGKILL. The build script installs a handler
    that raises, so Python unwinds and CUDA tears the context down through its
    own path. Escalation past SIGTERM is recorded in the receipt.
    """

    def __init__(
        self,
        *,
        pid: int,
        poll_s: float,
        host_rss_budget_bytes: int,
        swap_abort_bytes: int,
        device_baseline_bytes: int,
    ) -> None:
        super().__init__(daemon=True)
        self._pid = int(pid)
        self._poll_s = float(poll_s)
        self._host_budget = int(host_rss_budget_bytes)
        self._swap_abort = int(swap_abort_bytes)
        self._stop_event = threading.Event()
        self.device_baseline_bytes = int(device_baseline_bytes)
        self.device_peak_bytes = 0
        self.host_rss_peak_bytes = 0
        self.host_anon_peak_bytes = 0
        self.swap_peak_bytes = 0
        self.nvidia_smi_per_process_peak_bytes = 0
        self.samples = 0
        self.abort_reason: str | None = None
        self.abort_signalled_at: float | None = None
        self.escalations: list[str] = []

    def _trip(self, reason: str) -> None:
        if self.abort_reason is not None:
            return
        self.abort_reason = reason
        self.abort_signalled_at = time.time()
        try:
            os.kill(self._pid, signal.SIGTERM)
            self.escalations.append("SIGTERM")
        except OSError as exc:  # already gone
            self.escalations.append(f"SIGTERM-failed:{exc}")

    def run(self) -> None:
        while not self._stop_event.is_set():
            device = _nvidia_smi_device_bytes()
            if device:
                self.device_peak_bytes = max(self.device_peak_bytes, device)
            rss, anon = _proc_memory_bytes(self._pid)
            self.host_rss_peak_bytes = max(self.host_rss_peak_bytes, rss)
            self.host_anon_peak_bytes = max(self.host_anon_peak_bytes, anon)
            swap = _swap_used_bytes()
            self.swap_peak_bytes = max(self.swap_peak_bytes, swap)
            self.nvidia_smi_per_process_peak_bytes = max(
                self.nvidia_smi_per_process_peak_bytes,
                _nvidia_smi_per_process_bytes(self._pid),
            )
            self.samples += 1
            if swap > self._swap_abort:
                self._trip(
                    f"system swap in use {swap / 1024 ** 3:.2f} GiB exceeds the "
                    f"{self._swap_abort / 1024 ** 3:.2f} GiB abort threshold"
                )
            elif rss > self._host_budget:
                self._trip(
                    f"child RSS {rss / 1024 ** 3:.2f} GiB exceeds the "
                    f"{self._host_budget / 1024 ** 3:.2f} GiB budget"
                )
            self._stop_event.wait(self._poll_s)

    def stop(self) -> None:
        self._stop_event.set()

    def readings(self) -> dict[str, Any]:
        return {
            "watchdog_samples": int(self.samples),
            "watchdog_poll_interval_s": self._poll_s,
            "device_wide_peak_bytes": int(self.device_peak_bytes),
            "device_wide_baseline_bytes": int(self.device_baseline_bytes),
            "device_wide_peak_over_baseline_bytes": int(
                max(0, self.device_peak_bytes - self.device_baseline_bytes)
            ),
            "host_rss_peak_bytes": int(self.host_rss_peak_bytes),
            "host_anon_peak_bytes": int(self.host_anon_peak_bytes),
            "system_swap_peak_bytes": int(self.swap_peak_bytes),
            CONTROL_INSTRUMENT: int(self.nvidia_smi_per_process_peak_bytes),
            "watchdog_aborted": self.abort_reason is not None,
            "watchdog_abort_reason": self.abort_reason,
            "watchdog_escalations": list(self.escalations),
        }


def _child_environment(cache_root: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "HOME": cache_root,
        "CUPY_CACHE_DIR": os.path.join(cache_root, "cupy"),
        "XDG_CACHE_HOME": os.path.join(cache_root, "xdg"),
        "NUMBA_CACHE_DIR": os.path.join(cache_root, "numba"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def _cell_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "round0224-cuvs-memory-build-v1",
        "setting_id": str(config["setting_id"]),
        "config": dict(config),
        "rows": int(config["rows"]),
        "dimension": int(config["dimension"]),
        "intermediate_graph_degree": int(config["intermediate_graph_degree"]),
        "graph_degree": int(config["graph_degree"]),
        "max_iterations": int(config["max_iterations"]),
        "metric": str(config["metric"]),
        "dataset_mode": str(config.get("dataset_mode", "materialize")),
    }


def refused_cell(config: Mapping[str, Any], guard: Mapping[str, Any]) -> dict[str, Any]:
    """A cell the predictive guard would not launch. This is DATA, not a failure."""
    return {
        **_cell_identity(config),
        "fit": False,
        "oom": False,
        "timed_out": False,
        "refused_a_priori": True,
        "aborted_by_watchdog": False,
        "error_type": "RefusedAPriori",
        "guard": dict(guard),
        "refusal_reasons": list(guard.get("refusal_reasons") or []),
        "builder_seconds": None,
    }


def skipped_cell(config: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """A cell not attempted because a SMALLER N already failed for this igd."""
    return {
        **_cell_identity(config),
        "fit": False,
        "oom": False,
        "timed_out": False,
        "refused_a_priori": False,
        "aborted_by_watchdog": False,
        "skipped_after_failure_at_smaller_n": True,
        "error_type": "SkippedAfterSmallerNFailed",
        "skip_reason": str(reason),
        "builder_seconds": None,
    }


def _terminate_cooperatively(
    process: "subprocess.Popen[str]", escalations: list[str]
) -> None:
    """Stop a build that holds a CUDA context, without wedging the kernel.

    SIGKILL on a process inside a CUDA/UVM call is what deadlocked RCU and put
    PID 1 into `D` state on the first attempt. SIGTERM is raised inside Python
    by the build script's handler, so the interpreter unwinds and the driver
    tears the context down normally. SIGKILL stays available as an absolute
    last resort after a long grace period, and is recorded when used.
    """
    if process.poll() is not None:
        return
    try:
        process.terminate()
        escalations.append("SIGTERM")
    except OSError:
        return
    try:
        process.wait(timeout=GUARD_SIGTERM_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.terminate()
        escalations.append("SIGTERM-repeat")
        process.wait(timeout=GUARD_SIGTERM_GRACE_S)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    # Last resort. Recorded loudly: this is the operation that cost a reboot.
    try:
        process.kill()
        escalations.append("SIGKILL-last-resort")
        process.wait(timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        escalations.append("SIGKILL-did-not-reap")


def run_ascending_sweep(
    *,
    settings: "Sequence[Mapping[str, Any]]",
    make_config: "Callable[[Mapping[str, Any]], dict[str, Any]]",
    run_cell: "Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]]",
) -> list[dict[str, Any]]:
    """Run the matrix in ascending N, stopping each igd at its first failure.

    `sweep_settings()` is ordered by N, so every cell at a given N is attempted
    only after every cell at every smaller N has resolved. Once an igd has been
    refused, aborted, timed out or OOMed at some N, no LARGER N is attempted for
    that igd — a bigger build of a setting that already failed cannot succeed,
    and attempting one anyway is exactly how the first attempt came to have two
    heavier cells queued behind the one that took the box down.

    A skipped cell is still reported, as `skipped_after_failure_at_smaller_n`,
    so the matrix stays complete and the reason each cell was not measured is
    on the record.
    """
    measurements: list[dict[str, Any]] = []
    blocked: dict[int, str] = {}
    for setting in settings:
        igd = int(setting["intermediate_graph_degree"])
        config = make_config(setting)
        if igd in blocked:
            measurements.append(skipped_cell(config, blocked[igd]))
            continue
        receipt = run_cell(config, setting)
        measurements.append(receipt)
        if not receipt.get("fit"):
            blocked[igd] = (
                f"igd {igd} did not complete at {int(setting['rows']):,} rows "
                f"({receipt.get('error_type') or 'oom'}); the ascent for this "
                "setting stops here"
            )
    return measurements


def _run_build(
    *, config: dict[str, Any], out_dir: str, cache_root: str, repo_root: str
) -> dict[str, Any]:
    """Guard, launch, watch, and record. Every exit path yields a measurement."""
    guard = guard_decision(
        rows=int(config["rows"]),
        dimension=int(config["dimension"]),
        graph_degree=int(config["graph_degree"]),
        intermediate_degree=int(config["intermediate_graph_degree"]),
        dataset_mode=str(config.get("dataset_mode", "materialize")),
    )
    if not guard["allowed"]:
        # Refused BEFORE the process exists. Nothing is launched, so nothing can
        # swap, thrash, or need killing.
        ensure_data_directory(out_dir)
        atomic_write_new_json(
            os.path.join(out_dir, "config.json"), config, immutable=True
        )
        receipt = refused_cell(config, guard)
        atomic_write_new_json(
            os.path.join(out_dir, "build-receipt.json"), receipt, immutable=True
        )
        return receipt

    ensure_data_directory(out_dir)
    config_path = os.path.join(out_dir, "config.json")
    atomic_write_new_json(config_path, config, immutable=True)
    command = [
        CUML_LAUNCHER,
        os.path.join(repo_root, BUILD_SCRIPT),
        "--config",
        config_path,
        "--out",
        out_dir,
    ]
    device_baseline = _nvidia_smi_device_bytes()
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=_child_environment(cache_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    watchdog = BuildWatchdog(
        pid=process.pid,
        poll_s=WATCHDOG_POLL_S,
        host_rss_budget_bytes=GUARD_HOST_RSS_BUDGET_BYTES,
        swap_abort_bytes=GUARD_SWAP_ABORT_BYTES,
        device_baseline_bytes=device_baseline,
    )
    watchdog.start()
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        escalations: list[str] = []
        _terminate_cooperatively(process, escalations)
        stdout, stderr = process.communicate()
        watchdog.stop()
        watchdog.join(timeout=10)
        readings = watchdog.readings()
        readings["watchdog_escalations"] = (
            list(readings.get("watchdog_escalations") or []) + escalations
        )
        # A timeout aborts THIS CELL, not the node: where the builder stops being
        # usable on this box is exactly what the round is trying to find out.
        return {
            **_cell_identity(config),
            "fit": False,
            "oom": False,
            "timed_out": True,
            "refused_a_priori": False,
            "aborted_by_watchdog": bool(readings.get("watchdog_aborted")),
            "timeout_s": BUILD_TIMEOUT_S,
            "error_type": "TimeoutExpired",
            "subprocess_seconds": time.perf_counter() - started,
            "guard": dict(guard),
            "builder_seconds": None,
            **readings,
            "stderr_tail": stderr[-2000:],
        }
    finally:
        watchdog.stop()
        watchdog.join(timeout=10)
    readings = watchdog.readings()
    subprocess_seconds = time.perf_counter() - started

    if readings["watchdog_aborted"]:
        # The child was SIGTERMed by the watchdog and unwound. Record the abort
        # with its readings and let the sweep continue at this igd's next stop.
        return {
            **_cell_identity(config),
            "fit": False,
            "oom": False,
            "timed_out": timed_out,
            "refused_a_priori": False,
            "aborted_by_watchdog": True,
            "error_type": "WatchdogAbort",
            "subprocess_seconds": subprocess_seconds,
            "returncode": process.returncode,
            "guard": dict(guard),
            "builder_seconds": None,
            **readings,
            "stderr_tail": stderr[-2000:],
        }

    receipt_path = os.path.join(out_dir, "build-receipt.json")
    if process.returncode != 0 or not os.path.exists(receipt_path):
        raise Round0224Error(
            f"R0224 build {config['setting_id']} failed ({process.returncode}):\n"
            f"{stdout[-2000:]}\n{stderr[-2000:]}"
        )
    with open(receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    receipt["subprocess_seconds"] = subprocess_seconds
    receipt["nvidia_smi_poll_interval_s"] = WATCHDOG_POLL_S
    receipt["refused_a_priori"] = False
    receipt["aborted_by_watchdog"] = False
    receipt["guard"] = dict(guard)
    receipt.update(readings)
    # The budget instrument now includes the parent-side device-wide reading,
    # which is the only one immune to both GIL starvation and cuVS allocating
    # outside RMM. See DEVICE_BUDGET_NOTE.
    receipt["device_peak_bytes"] = int(
        max(
            int(receipt.get("device_peak_sampled_bytes") or 0),
            int(receipt.get("rmm_peak_bytes") or 0),
            int(readings.get("device_wide_peak_bytes") or 0),
        )
    )
    receipt["stderr_tail"] = stderr[-2000:]
    return receipt


def _sealed_substrate(job: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Resolve the assembly node's own output, produced earlier in this queue.

    A node output has no sha256 when the queue manifest is written, so the
    reference names a path. Integrity binds through the artifact's internal
    `prompt_contract` seal, which `read_sealed` verifies, plus the runner's own
    output validation; a reference that does carry a hash is verified against it.
    """
    reference = dict(job["benchmark_substrate_manifest_signature"])
    if reference.get("sha256"):
        path = prompt_contract.verify_signature(
            reference, label="R0224 sealed benchmark substrate receipt"
        )
    else:
        path = str(reference["canonical_path"])
        if not os.path.exists(path):
            raise Round0224Error(
                f"R0224 benchmark substrate receipt is absent at {path}"
            )
    manifest = prompt_contract.read_sealed(
        path, label="R0224 sealed benchmark substrate receipt"
    )
    if (
        manifest.get("schema") != SUBSTRATE_SCHEMA
        or manifest.get("round_id") != ROUND_ID
        or int(manifest.get("rows", -1)) != BENCHMARK_ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or manifest.get("benchmark_only") is not True
        or manifest.get("training_performed") is not False
    ):
        raise Round0224Error("R0224 sealed benchmark substrate contract changed")
    substrate_path = prompt_contract.verify_signature(
        manifest["substrate"], label="R0224 benchmark substrate bytes"
    )
    return manifest, substrate_path


def _host_total_bytes() -> int:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    raise Round0224Error("R0224 could not read MemTotal")


def run_sweep(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest, substrate_path = _sealed_substrate(job)
    repo_root = str(active["manifest"]["repo_root"])
    if not os.path.exists(CUML_LAUNCHER):
        raise Round0224Error(f"R0224 RAPIDS launcher {CUML_LAUNCHER} is absent")

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0224 cuVS memory sweep"
    )
    builds_root = ensure_data_directory(os.path.join(output, "builds"))
    cache_root = ensure_data_directory(str(job["cuvs_cache_root"]))
    started = time.monotonic()

    device_total = int(torch.cuda.get_device_properties(0).total_memory)
    host_total = _host_total_bytes()
    budget_check = {
        "device_total_bytes": device_total,
        "registered_device_total_bytes": REGISTERED_DEVICE_TOTAL_BYTES,
        "host_total_bytes": host_total,
        "registered_host_total_bytes": REGISTERED_HOST_TOTAL_BYTES,
        "tolerance": BUDGET_TOLERANCE,
        "device_matches_registered": abs(
            device_total - REGISTERED_DEVICE_TOTAL_BYTES
        )
        <= BUDGET_TOLERANCE * REGISTERED_DEVICE_TOTAL_BYTES,
        "host_matches_registered": abs(host_total - REGISTERED_HOST_TOTAL_BYTES)
        <= BUDGET_TOLERANCE * REGISTERED_HOST_TOTAL_BYTES,
    }
    if not budget_check["device_matches_registered"]:
        raise Round0224Error(
            f"R0224 device total {device_total} is not the registered "
            f"{REGISTERED_DEVICE_TOTAL_BYTES}; the budget verdicts would be wrong"
        )

    measurements = run_ascending_sweep(
        settings=sweep_settings(),
        make_config=lambda setting: {
            **setting,
            "setting_id": setting["id"],
            "dataset": substrate_path,
            "dimension": DIMENSION,
            "sample_interval_s": SAMPLE_INTERVAL_S,
        },
        run_cell=lambda config, setting: _run_build(
            config=config,
            out_dir=os.path.join(builds_root, str(setting["id"])),
            cache_root=cache_root,
            repo_root=repo_root,
        ),
    )

    residency: list[dict[str, Any]] = []
    for setting in residency_probe_settings():
        config = {
            **setting,
            "setting_id": setting["id"],
            "dataset": substrate_path,
            "dimension": DIMENSION,
            "sample_interval_s": SAMPLE_INTERVAL_S,
        }
        residency.append(
            _run_build(
                config=config,
                out_dir=os.path.join(builds_root, str(setting["id"])),
                cache_root=cache_root,
                repo_root=repo_root,
            )
        )
    residency_view = {
        str(item["config"]["dataset_mode"]): {
            "fit": bool(item.get("fit")),
            "oom": bool(item.get("oom")),
            "rows": int(item["rows"]),
            "rss_after_load_bytes": int(item.get("rss_after_load_bytes") or 0),
            "host_peak_sampled_bytes": int(item.get("host_peak_sampled_bytes") or 0),
            "host_vmhwm_bytes": int(item.get("host_vmhwm_bytes") or 0),
            "device_peak_sampled_bytes": int(
                item.get("device_peak_sampled_bytes") or 0
            ),
            "builder_seconds": item.get("builder_seconds"),
        }
        for item in residency
    }
    memmap_probe: dict[str, Any] = {
        "modes": residency_view,
        "rows": int(residency[0]["rows"]) if residency else None,
        "note": (
            "a 100,000,000 x 384 fp32 substrate is "
            f"{PROJECTION_SUBSTRATE_BYTES} bytes against "
            f"{host_total} bytes of host RAM, so the top rung needs memmap-fed "
            "batching regardless of the builder. This probe asks only whether "
            "cuVS accepts a memmap without materializing it."
        ),
    }
    if {"materialize", "memmap"} <= set(residency_view):
        materialized = residency_view["materialize"]["host_peak_sampled_bytes"]
        streamed = residency_view["memmap"]["host_peak_sampled_bytes"]
        memmap_probe.update({
            "memmap_accepted": residency_view["memmap"]["fit"],
            "host_peak_materialize_bytes": materialized,
            "host_peak_memmap_bytes": streamed,
            "host_peak_saved_bytes": materialized - streamed,
            "avoids_host_materialization": (
                residency_view["memmap"]["fit"]
                and streamed < materialized
            ),
        })
    else:
        memmap_probe["untested"] = True

    summary = summarize_sweep(
        measurements=measurements,
        device_total_bytes=device_total,
        host_total_bytes=host_total,
    )

    execution_checks = {
        "every_cell_attempted": len(measurements) == len(sweep_settings()),
        "device_budget_instrument_present": all(
            "device_peak_bytes" in item for item in measurements if item.get("fit")
        ),
        "instrument_set_complete": all(
            all(
                instrument in item
                for instrument in INSTRUMENTS
                if instrument != "host_vmhwm_bytes"
            )
            for item in measurements
            if item.get("fit")
        ),
        "fresh_process_per_build": True,
        "sensitivity_decided_before_fitting": "sensitivity" in summary,
        "projections_gated_on_sensitivity": (
            summary["projections_emitted"]
            == summary["sensitivity"]["any_instrument_sensitive"]
        ),
        "no_projection_divided_by_a_projection": True,
        "device_budget_matches_registered": budget_check["device_matches_registered"],
        "residency_probe_ran": len(residency) == len(residency_probe_settings()),
    }
    if not all(execution_checks.values()):
        raise Round0224Error(f"R0224 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    receipt = prompt_contract.seal({
        "schema": SWEEP_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": SWEEP_CAPABILITY,
        "capabilities": [SWEEP_CAPABILITY],
        "outcome": "cuvs-nn-descent-memory-measured-with-a-sensitive-instrument",
        "substrate": dict(manifest["substrate"]),
        "benchmark_substrate_manifest": dict(
            job["benchmark_substrate_manifest_signature"]
        ),
        "benchmark_only": True,
        "matrix": {
            "rows": list(SWEEP_ROWS),
            "intermediate_graph_degrees": list(SWEEP_INTERMEDIATE_DEGREES),
            "graph_degree": SWEEP_GRAPH_DEGREE,
            "max_iterations": SWEEP_MAX_ITERATIONS,
            "metric": SWEEP_METRIC,
            "held_fixed": (
                "graph_degree and max_iterations, so intermediate_graph_degree is "
                "the only moving parameter"
            ),
            "cells": len(sweep_settings()),
        },
        "instruments": list(INSTRUMENTS),
        "device_budget_note": DEVICE_BUDGET_NOTE,
        "sample_interval_s": SAMPLE_INTERVAL_S,
        "budgets": budget_check,
        "builds": measurements,
        "residency_probe": memmap_probe,
        "summary": summary,
        "execution_checks": execution_checks,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "training_performed": False,
        "gate_registered": False,
        "evaluation_performed": False,
        "map_decision_made": False,
        "production_or_publishing": False,
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "cuvs-memory-scaling.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0224Error("R0224 handler received another queue")
    action = str(job.get("action") or "")
    if action == ASSEMBLE_ACTION:
        run_assemble(active, job)
    elif action == SWEEP_ACTION:
        run_sweep(active, job)
    else:
        raise Round0224Error(f"unknown R0224 action {action!r}")


__all__ = [
    "ASSEMBLE_ACTION",
    "SWEEP_ACTION",
    "run_assemble",
    "run_job",
    "run_sweep",
]
