"""Execute R0226 — qualify a graph builder that can actually reach 100M.

Two nodes.

`qualify_graph_builders` (GPU) runs both candidate ladders in ascending `N`,
each cell as a **fresh subprocess** — candidate A under the RAPIDS env, candidate
B under the release venv — behind a predictive guard, a live swap-growth
watchdog and a cooperative SIGTERM abort. A refusal, an abort and a timeout are
all recorded as measurements; a ladder stops at its candidate's first one.

`evaluate_recall_and_verdict` (GPU) scores the 2,000,000-row graphs each
candidate emitted against R0220's sealed exact k15 truth — strict *and*
tie-aware containment, with the candidate cosines **recomputed from the
substrate here** rather than taken from the builder's own accumulator (the
independence rule of review-0216-01) — runs R0215's degree-zero tripwire, and
applies the registered 100M device rule and per-rung recommendation.

Nothing in this round trains a map, registers a gate, or seals a graph for
downstream use. Qualification is not adoption.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0220_cuvs_qualification import (
    graph_validity,
    strict_containment_rows,
    summarize,
    tie_aware_rows,
)
from basemap.round0226_graph_builders import (
    A_CLUSTER_CAPACITY_ROWS,
    A_CLUSTER_TARGET_ROWS,
    A_GRAPH_DEGREE,
    A_INTERMEDIATE_DEGREE,
    A_MAX_ITERATIONS,
    A_METRIC,
    A_SCRATCH_BUDGET_BYTES,
    A_SEED,
    A_SPILL,
    B_NLIST,
    B_NPROBE,
    B_QUERY_BLOCK,
    B_SEARCH_K,
    B_SHARD_ROWS,
    B_TRAIN_ROWS,
    BUILD_TIMEOUT_S,
    CANDIDATES,
    CANDIDATE_A,
    CANDIDATE_B,
    DEVICE_BUDGET_INSTRUMENT,
    DEVICE_INSTRUMENT_QUANTUM_BYTES,
    DEVICE_TOTAL_BYTES,
    DIMENSION,
    FLATNESS_TOLERANCE,
    GPU_HOURS_CAP,
    GRAPH_K,
    GUARD_BUDGET_NOTE,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SIGTERM_GRACE_S,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    INSTRUMENTS,
    INSTRUMENT_APPLICABILITY,
    INSTRUMENT_NOTE,
    LADDER_ROWS,
    METRIC_EQUIVALENCE,
    NORM_TOLERANCE,
    PHASE2_RUNGS,
    PROJECTION_ROWS,
    QUALIFICATION_CAPABILITY,
    QUALIFICATION_SCHEMA,
    RECALL_ROWS,
    RECALL_SCHEMA,
    ROUND_ID,
    Round0226Error,
    SAMPLE_INTERVAL_S,
    SENSITIVITY_ARGUMENT,
    SUBSTRATE_2M_PATH,
    TRUTH_COS_PATH,
    TRUTH_IDS_PATH,
    TRUTH_RECEIPT_PATH,
    TRUTH_SCHEMA,
    WATCHDOG_POLL_S,
    a_cluster_count,
    a_spill_groups,
    b_shard_count,
    device_verdict_at_100m,
    flatness,
    guard_decision,
    ladder_settings,
    power_law,
    project_wall,
    rung_recommendation,
)
from basemap import round0113_prompt_contrast as prompt_contract


QUALIFY_ACTION = "qualify_graph_builders"
EVALUATE_ACTION = "evaluate_recall_and_verdict"

NVIDIA_SMI = "/usr/bin/nvidia-smi"
CUML_LAUNCHER = "/data/latent-basemap/cuml_py"
RELEASE_PYTHON = "/home/enjalot/code/latent-basemap-run/.venv/bin/python"
BUILD_SCRIPT_BY_CANDIDATE: dict[str, str] = {
    CANDIDATE_A: "basemap/round0226_cluster_spill_build.py",
    CANDIDATE_B: "basemap/round0226_sharded_ivf_build.py",
}
#: The recall floor Phase 2 requires of any graph builder
#: (`guides/plan-minilm-100m-v2.md`: exact sharded graphs with >= 0.90 recall
#: qualification). Registered here before measurement.
RECALL_MEAN_FLOOR = 0.90
RECALL_P10_FLOOR = 0.80
EVAL_BLOCK = 16_384


# --------------------------------------------------------------------------- #
# instruments, watchdog, cooperative abort
# --------------------------------------------------------------------------- #
def _nvidia_smi_device_bytes() -> int:
    """Device-wide bytes in use, from the driver, in a separate process.

    The budget instrument. It runs outside the child so a builder holding the
    GIL cannot starve it, and it reads the device rather than a bookkeeper so an
    allocation made outside RMM (or by FAISS, which never touches RMM) cannot
    hide from it. The queue holds an exclusive GPU lease and the baseline is read
    immediately before each child starts, so the over-baseline figure is
    attributable to the build.
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


def _nvidia_smi_per_process_bytes(pid: int) -> int:
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


def _proc_memory_bytes(pid: int) -> tuple[int, int]:
    """`(VmRSS, RssAnon)`. Anonymous bytes are the swappable ones."""
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
    """Live sampling with a cooperative abort.

    Trips on **swap growth over a pre-launch baseline**, never on an absolute
    swap level: this box already holds swap from idle daemons, so an absolute
    threshold would abort every cell before it started. What took the box down in
    R0224's first attempt was a build driving swap from its baseline to
    exhaustion, and growth is exactly what sees that.

    It also trips on the host **anonymous** budget. Anonymous bytes are the
    swappable ones; file-backed pages are clean page cache and are evicted, not
    swapped (review-0224-01), so they are measured but never used to abort.

    An abort is **SIGTERM**, never SIGKILL. The build scripts install a handler
    that raises inside Python so the interpreter unwinds and the driver tears the
    CUDA context down through its normal path. Escalation past SIGTERM is
    recorded in the receipt.
    """

    def __init__(
        self,
        *,
        pid: int,
        poll_s: float,
        host_anon_budget_bytes: int,
        swap_growth_abort_bytes: int,
        device_baseline_bytes: int,
        swap_baseline_bytes: int,
    ) -> None:
        super().__init__(daemon=True)
        self._pid = int(pid)
        self._poll = float(poll_s)
        self._anon_budget = int(host_anon_budget_bytes)
        self._swap_abort = int(swap_growth_abort_bytes)
        self._swap_baseline = int(swap_baseline_bytes)
        self._halt = threading.Event()
        self.device_baseline_bytes = int(device_baseline_bytes)
        self.swap_baseline_bytes = int(swap_baseline_bytes)
        self.device_peak_bytes = 0
        self.per_process_peak_bytes = 0
        self.rss_peak_bytes = 0
        self.anon_peak_bytes = 0
        self.swap_peak_bytes = int(swap_baseline_bytes)
        self.samples = 0
        self.aborted = False
        self.abort_reason: str | None = None
        self.escalations: list[str] = []

    def _trip(self, reason: str) -> None:
        if self.aborted:
            return
        self.aborted = True
        self.abort_reason = reason
        try:
            os.kill(self._pid, 15)
            self.escalations.append("SIGTERM")
        except OSError as error:
            self.escalations.append(f"SIGTERM-failed:{error}")

    def run(self) -> None:
        while not self._halt.is_set():
            self.samples += 1
            self.device_peak_bytes = max(
                self.device_peak_bytes, _nvidia_smi_device_bytes()
            )
            self.per_process_peak_bytes = max(
                self.per_process_peak_bytes, _nvidia_smi_per_process_bytes(self._pid)
            )
            rss, anon = _proc_memory_bytes(self._pid)
            self.rss_peak_bytes = max(self.rss_peak_bytes, rss)
            self.anon_peak_bytes = max(self.anon_peak_bytes, anon)
            swap = _swap_used_bytes()
            self.swap_peak_bytes = max(self.swap_peak_bytes, swap)
            if swap - self._swap_baseline > self._swap_abort:
                self._trip(
                    f"system swap grew "
                    f"{(swap - self._swap_baseline) / 1024 ** 3:.2f} GiB over its "
                    f"{self._swap_baseline / 1024 ** 3:.2f} GiB pre-launch "
                    f"baseline, exceeding the "
                    f"{self._swap_abort / 1024 ** 3:.2f} GiB growth threshold"
                )
            elif anon > self._anon_budget:
                self._trip(
                    f"child anonymous memory {anon / 1024 ** 3:.2f} GiB exceeds "
                    f"the {self._anon_budget / 1024 ** 3:.2f} GiB budget"
                )
            self._halt.wait(self._poll)

    def halt(self) -> None:
        self._halt.set()
        self.join(timeout=10)

    def readings(self) -> dict[str, Any]:
        return {
            "device_wide_peak_bytes": int(self.device_peak_bytes),
            "device_wide_baseline_bytes": int(self.device_baseline_bytes),
            "device_wide_peak_over_baseline_bytes": int(
                max(0, self.device_peak_bytes - self.device_baseline_bytes)
            ),
            "nvidia_smi_per_process_peak_bytes": int(self.per_process_peak_bytes),
            "host_rss_peak_bytes": int(self.rss_peak_bytes),
            "host_anon_peak_bytes": int(self.anon_peak_bytes),
            "system_swap_baseline_bytes": int(self.swap_baseline_bytes),
            "system_swap_peak_bytes": int(self.swap_peak_bytes),
            "system_swap_growth_bytes": int(
                max(0, self.swap_peak_bytes - self.swap_baseline_bytes)
            ),
            "watchdog_samples": int(self.samples),
            "watchdog_aborted": bool(self.aborted),
            "watchdog_abort_reason": self.abort_reason,
            "watchdog_escalations": list(self.escalations),
        }


def _terminate_cooperatively(
    process: "subprocess.Popen[str]", escalations: list[str]
) -> None:
    """Stop a build that holds a CUDA context without wedging the kernel."""
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
    # Last resort, recorded loudly: this is the operation that cost a reboot.
    try:
        process.kill()
        escalations.append("SIGKILL-last-resort")
        process.wait(timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        escalations.append("SIGKILL-did-not-reap")


# --------------------------------------------------------------------------- #
# cell records
# --------------------------------------------------------------------------- #
def _cell_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "setting_id": str(config["setting_id"]),
        "candidate": str(config["candidate"]),
        "rows": int(config["rows"]),
        "dimension": int(config["dimension"]),
        "k": int(config["k"]),
        "substrate": str(config["substrate"]),
        "config": dict(config),
    }


def _null_instruments(candidate: str) -> dict[str, Any]:
    """Every registered instrument, present, with `null` where inapplicable."""
    out: dict[str, Any] = {}
    for instrument in INSTRUMENTS:
        applies = INSTRUMENT_APPLICABILITY[instrument]
        out[instrument] = None if applies not in ("both", candidate) else 0
    return out


def refused_cell(config: Mapping[str, Any], guard: Mapping[str, Any]) -> dict[str, Any]:
    """A cell the predictive guard would not launch. DATA, not a failure."""
    return {
        **_cell_identity(config),
        **_null_instruments(str(config["candidate"])),
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
    """A cell not attempted because a SMALLER N already failed for this builder."""
    return {
        **_cell_identity(config),
        **_null_instruments(str(config["candidate"])),
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


def run_ascending_ladder(
    *,
    settings: Sequence[Mapping[str, Any]],
    make_config: "Callable[[Mapping[str, Any]], dict[str, Any]]",
    run_cell: "Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]]",
) -> list[dict[str, Any]]:
    """Ascend N per candidate, stopping that candidate at its first failure.

    A larger build of a configuration that already failed cannot succeed, and
    queueing one anyway is how R0224's first attempt came to hold the GPU for
    1.30235 h and cost a hard reboot. A skipped cell is still reported so the
    matrix stays complete and the reason each cell was not measured is on the
    record.
    """
    measurements: list[dict[str, Any]] = []
    stopped: dict[str, str] = {}
    for setting in settings:
        candidate = str(setting["candidate"])
        config = make_config(setting)
        if candidate in stopped:
            measurements.append(skipped_cell(config, stopped[candidate]))
            continue
        record = run_cell(config, setting)
        measurements.append(record)
        if not record.get("fit"):
            stopped[candidate] = (
                f"{candidate} did not fit at {int(setting['rows'])} rows "
                f"({record.get('error_type') or 'no error type'})"
            )
    return measurements


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


def _run_build(
    *, config: dict[str, Any], out_dir: str, cache_root: str, repo_root: str
) -> dict[str, Any]:
    """Guard, launch, watch, record. Every exit path yields a measurement."""
    candidate = str(config["candidate"])
    guard = guard_decision(candidate=candidate, rows=int(config["rows"]))
    if not guard["allowed"]:
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
    launcher = CUML_LAUNCHER if candidate == CANDIDATE_A else RELEASE_PYTHON
    command = [
        launcher,
        os.path.join(repo_root, BUILD_SCRIPT_BY_CANDIDATE[candidate]),
        "--config",
        config_path,
        "--out",
        out_dir,
    ]
    device_baseline = _nvidia_smi_device_bytes()
    swap_baseline = _swap_used_bytes()
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
        host_anon_budget_bytes=GUARD_HOST_ANON_BUDGET_BYTES,
        swap_growth_abort_bytes=GUARD_SWAP_GROWTH_ABORT_BYTES,
        device_baseline_bytes=device_baseline,
        swap_baseline_bytes=swap_baseline,
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
        watchdog.halt()
        readings = watchdog.readings()
        readings["watchdog_escalations"] = (
            list(readings.get("watchdog_escalations") or []) + escalations
        )
        return {
            **_cell_identity(config),
            **_null_instruments(candidate),
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
        watchdog.halt()
    readings = watchdog.readings()
    subprocess_seconds = time.perf_counter() - started

    if readings["watchdog_aborted"]:
        return {
            **_cell_identity(config),
            **_null_instruments(candidate),
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
        raise Round0226Error(
            f"R0226 build {config['setting_id']} failed ({process.returncode}):\n"
            f"{stdout[-2000:]}\n{stderr[-2000:]}"
        )
    with open(receipt_path, encoding="utf-8") as handle:
        child = json.load(handle)
    record = {
        **_cell_identity(config),
        **_null_instruments(candidate),
        **{
            key: value
            for key, value in child.items()
            if key not in ("config", "setting_id")
        },
        **readings,
        "subprocess_seconds": subprocess_seconds,
        "nvidia_smi_poll_interval_s": WATCHDOG_POLL_S,
        "refused_a_priori": False,
        "aborted_by_watchdog": False,
        "timed_out": bool(child.get("timed_out", False)),
        "guard": dict(guard),
        "stderr_tail": stderr[-2000:],
    }
    return record


# --------------------------------------------------------------------------- #
# node 1: the ladders
# --------------------------------------------------------------------------- #
def _assert_unit_norm(path: str, *, rows: int) -> dict[str, Any]:
    """Cheap seeded probe that the metric-equivalence argument actually holds."""
    array = np.load(path, mmap_mode="r")
    rng = np.random.default_rng(226)
    probe = np.sort(rng.choice(min(rows, int(array.shape[0])), size=4096, replace=False))
    sample = np.asarray(array[probe], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))
    if worst > NORM_TOLERANCE:
        raise Round0226Error(
            f"R0226 substrate {path} is not unit-normalised (worst |norm-1| "
            f"{worst}), so the sqeuclidean/inner-product/cosine equivalence "
            "the round registers does not hold"
        )
    return {
        "path": path,
        "probe_rows": int(probe.size),
        "worst_abs_norm_deviation": worst,
        "tolerance": NORM_TOLERANCE,
    }


def run_qualify(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0226 qualification")
    builds_root = ensure_data_directory(os.path.join(output, "builds"))
    cache_root = ensure_data_directory(str(job["cuvs_cache_root"]))
    scratch_root = ensure_data_directory(str(job["scratch_root"]))
    repo_root = str(active["manifest"]["repo_root"])

    settings = ladder_settings()
    norm_checks = [
        _assert_unit_norm(path, rows=rows)
        for path, rows in sorted({
            (str(item["substrate"]), int(item["rows"])) for item in settings
        })
    ]

    def make_config(setting: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "setting_id": str(setting["id"]),
            "candidate": str(setting["candidate"]),
            "rows": int(setting["rows"]),
            "dimension": int(setting["dimension"]),
            "k": int(setting["k"]),
            "substrate": str(setting["substrate"]),
            "emit_graph": bool(setting["emit_graph"]),
            "scratch_root": scratch_root,
            "sample_interval_s": SAMPLE_INTERVAL_S,
        }

    def run_cell(config: dict[str, Any], _setting: Mapping[str, Any]) -> dict[str, Any]:
        return _run_build(
            config=config,
            out_dir=os.path.join(builds_root, str(config["setting_id"])),
            cache_root=cache_root,
            repo_root=repo_root,
        )

    measurements = run_ascending_ladder(
        settings=settings, make_config=make_config, run_cell=run_cell
    )

    per_candidate: dict[str, Any] = {}
    for candidate in CANDIDATES:
        cells = [
            item
            for item in measurements
            if item["candidate"] == candidate and item.get("fit")
        ]
        entry: dict[str, Any] = {
            "cells_measured": len(cells),
            "rows_measured": [int(item["rows"]) for item in cells],
            "device_wide_peak_bytes": [
                int(item["device_wide_peak_bytes"]) for item in cells
            ],
            "device_wide_peak_over_baseline_bytes": [
                int(item["device_wide_peak_over_baseline_bytes"]) for item in cells
            ],
            "builder_seconds": [float(item["builder_seconds"]) for item in cells],
            "zero_degree_rows": [int(item["zero_degree_rows"]) for item in cells],
        }
        if len(cells) >= 2:
            entry["device_flatness"] = flatness(entry["device_wide_peak_bytes"])
            entry["device_verdict_100m"] = device_verdict_at_100m(
                candidate=candidate,
                rows=entry["rows_measured"],
                device_peaks=entry["device_wide_peak_bytes"],
            )
            wall_fit = power_law(entry["rows_measured"], entry["builder_seconds"])
            entry["wall_power_law"] = wall_fit
            entry["wall_projection_100m"] = project_wall(wall_fit)
        else:
            entry["device_flatness"] = None
            entry["device_verdict_100m"] = None
            entry["wall_power_law"] = None
            entry["wall_projection_100m"] = None
        per_candidate[candidate] = entry

    execution_checks = {
        "every_cell_resolved": len(measurements) == len(settings),
        "instrument_set_complete": all(
            all(instrument in item for instrument in INSTRUMENTS)
            for item in measurements
        ),
        "budget_instrument_present_on_every_fit": all(
            item.get(DEVICE_BUDGET_INSTRUMENT) is not None
            for item in measurements
            if item.get("fit")
        ),
        "fresh_process_per_build": all(
            os.path.exists(
                os.path.join(builds_root, str(item["setting_id"]), "config.json")
            )
            for item in measurements
            if not item.get("skipped_after_failure_at_smaller_n")
        ),
        "no_process_sigkilled": all(
            "SIGKILL-last-resort" not in (item.get("watchdog_escalations") or [])
            for item in measurements
        ),
        "no_swap_growth_beyond_threshold": all(
            int(item.get("system_swap_growth_bytes") or 0)
            <= GUARD_SWAP_GROWTH_ABORT_BYTES
            for item in measurements
        ),
        "substrates_unit_normalised": bool(norm_checks) and all(
            float(check["worst_abs_norm_deviation"]) <= NORM_TOLERANCE
            for check in norm_checks
        ),
        # cuVS 25.02.01 has no all_neighbors, so paper 0197's builder is not
        # available directly and candidate A implements the design by hand. The
        # check computes the negative rather than asserting it.
        "cuvs_all_neighbors_absent": all(
            item.get("cuvs_has_all_neighbors") is not True for item in measurements
        ),
    }
    if not all(execution_checks.values()):
        raise Round0226Error(f"R0226 execution checks failed: {execution_checks}")

    receipt = prompt_contract.seal({
        "schema": QUALIFICATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": QUALIFICATION_CAPABILITY,
        "capabilities": [QUALIFICATION_CAPABILITY],
        "outcome": "graph-builder-device-footprint-and-wall-measured-at-fixed-capacity",
        "qualification_not_adoption": (
            "this round qualifies builders; it trains no map, registers no gate "
            "and seals no graph for downstream use"
        ),
        "candidates": {
            CANDIDATE_A: {
                "name": CANDIDATE_A,
                "design": (
                    "paper 0197 out-of-core all-neighbours, implemented by hand "
                    "because cuvs 25.02.01 ships no all_neighbors and no "
                    "cuvs.cluster"
                ),
                "spill": A_SPILL,
                "cluster_target_rows": A_CLUSTER_TARGET_ROWS,
                "cluster_capacity_rows": A_CLUSTER_CAPACITY_ROWS,
                "scratch_budget_bytes": A_SCRATCH_BUDGET_BYTES,
                "seed": A_SEED,
                "graph_degree": A_GRAPH_DEGREE,
                "intermediate_graph_degree": A_INTERMEDIATE_DEGREE,
                "max_iterations": A_MAX_ITERATIONS,
                "metric": A_METRIC,
                "clusters_by_rows": {
                    str(rows): a_cluster_count(rows) for rows in LADDER_ROWS
                },
                "spill_groups_by_rows": {
                    str(rows): a_spill_groups(rows) for rows in LADDER_ROWS
                },
                "recall_cost": (
                    "a true neighbour is reachable only if it lands in one of "
                    "the s clusters the query was assigned to; cross-cluster "
                    "neighbours are lost and that is measured, not assumed"
                ),
            },
            CANDIDATE_B: {
                "name": CANDIDATE_B,
                "design": "R0171 sharded fp32 IVF with exact global top-k merge",
                "shard_rows": B_SHARD_ROWS,
                "nlist": B_NLIST,
                "nprobe": B_NPROBE,
                "search_k": B_SEARCH_K,
                "query_block": B_QUERY_BLOCK,
                "train_rows": B_TRAIN_ROWS,
                "shards_by_rows": {
                    str(rows): b_shard_count(rows) for rows in LADDER_ROWS
                },
                "recall_cost": (
                    "none from sharding: searching row-disjoint shards and "
                    "merging is the same candidate operation as searching their "
                    "union; the gap to exact truth is the ordinary IVF nprobe gap"
                ),
            },
        },
        "ladder_rows": list(LADDER_ROWS),
        "cells": len(settings),
        "instruments": list(INSTRUMENTS),
        "instrument_applicability": dict(INSTRUMENT_APPLICABILITY),
        "instrument_note": INSTRUMENT_NOTE,
        "device_budget_instrument": DEVICE_BUDGET_INSTRUMENT,
        "device_instrument_quantum_bytes": DEVICE_INSTRUMENT_QUANTUM_BYTES,
        "sensitivity_argument": SENSITIVITY_ARGUMENT,
        "flatness_tolerance": FLATNESS_TOLERANCE,
        "metric_equivalence": METRIC_EQUIVALENCE,
        "substrate_norm_checks": norm_checks,
        "guard": {
            "device_budget_bytes": GUARD_DEVICE_BUDGET_BYTES,
            "host_anon_budget_bytes": GUARD_HOST_ANON_BUDGET_BYTES,
            "swap_growth_abort_bytes": GUARD_SWAP_GROWTH_ABORT_BYTES,
            "watchdog_poll_s": WATCHDOG_POLL_S,
            "sigterm_grace_s": GUARD_SIGTERM_GRACE_S,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "note": GUARD_BUDGET_NOTE,
        },
        "device_total_bytes": DEVICE_TOTAL_BYTES,
        "projection_rows": PROJECTION_ROWS,
        "builds": measurements,
        "per_candidate": per_candidate,
        "execution_checks": execution_checks,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "training_performed": False,
        "gate_registered": False,
        "evaluation_performed": False,
        "map_decision_made": False,
        "production_or_publishing": False,
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2),
        },
    })
    atomic_write_new_json(
        os.path.join(output, "graph-builder-qualification.json"),
        receipt,
        immutable=True,
    )


# --------------------------------------------------------------------------- #
# node 2: recall against exact truth, tripwire, verdict
# --------------------------------------------------------------------------- #
def _recompute_cosines(torch: Any, tensor: Any, ids: np.ndarray) -> np.ndarray:
    """Exact cosines for emitted neighbour ids, computed HERE, not by the builder.

    Review-0216-01: an in-node recall probe that shares the builder's accumulator
    is not independent. These cosines come from the substrate and the emitted
    ids alone.
    """
    rows, width = ids.shape
    out = np.empty((rows, width), dtype=np.float32)
    for begin in range(0, rows, EVAL_BLOCK):
        end = min(begin + EVAL_BLOCK, rows)
        block = ids[begin:end]
        safe = np.where(block < 0, 0, block).astype(np.int64)
        neighbours = tensor[torch.from_numpy(safe).to(tensor.device)]
        queries = tensor[begin:end]
        scores = torch.einsum("bd,bkd->bk", queries, neighbours)
        values = scores.float().cpu().numpy()
        values[block < 0] = -np.inf
        out[begin:end] = values
        del neighbours, queries, scores
    return out


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0226 recall")
    qualification_path = str(job["qualification_manifest"])
    qualification = prompt_contract.read_sealed(
        qualification_path, label="R0226 qualification"
    )
    if (
        qualification.get("schema") != QUALIFICATION_SCHEMA
        or qualification.get("round_id") != ROUND_ID
    ):
        raise Round0226Error("R0226 qualification receipt contract changed")

    truth_receipt = prompt_contract.read_sealed(
        TRUTH_RECEIPT_PATH, label="R0220 exact k15 truth"
    )
    if (
        truth_receipt.get("schema") != TRUTH_SCHEMA
        or truth_receipt.get("round_id") != "0220"
        or not truth_receipt["probe"]["passed"]
    ):
        raise Round0226Error("R0226 refuses a truth that did not pass its own probe")
    truth_ids = np.load(TRUTH_IDS_PATH)
    truth_cos = np.load(TRUTH_COS_PATH)
    if truth_ids.shape != (RECALL_ROWS, GRAPH_K) or truth_cos.shape != (
        RECALL_ROWS,
        GRAPH_K,
    ):
        raise Round0226Error("R0226 truth arrays have the wrong shape")

    substrate = np.load(SUBSTRATE_2M_PATH, mmap_mode="r")[:RECALL_ROWS]
    tensor = torch.from_numpy(np.ascontiguousarray(substrate, dtype=np.float32)).to(
        "cuda"
    )
    kth = truth_cos[:, GRAPH_K - 1].astype(np.float64)

    builds_root = str(job["builds_root"])
    recalls: dict[str, Any] = {}
    for candidate in CANDIDATES:
        cells = [
            item
            for item in qualification["builds"]
            if item["candidate"] == candidate and item.get("fit")
        ]
        largest_wall = None
        if cells:
            largest = max(cells, key=lambda item: int(item["rows"]))
            largest_wall = float(largest["builder_seconds"])
        graph_path = os.path.join(
            builds_root, f"{candidate}-n{RECALL_ROWS}", "graph-k15-ids.i32.npy"
        )
        if not os.path.exists(graph_path):
            recalls[candidate] = {
                "graph_available": False,
                "reason": (
                    f"{candidate} produced no {RECALL_ROWS}-row graph, so it has "
                    "no recall measurement and cannot qualify"
                ),
                "clears_recall_floor": False,
                "zero_degree_rows": None,
                "wall_seconds_at_largest_n": largest_wall,
            }
            continue
        ids = np.load(graph_path)
        if ids.shape != (RECALL_ROWS, GRAPH_K):
            raise Round0226Error(f"R0226 {candidate} graph has the wrong shape")
        cosines = _recompute_cosines(torch, tensor, ids)
        strict = summarize(
            strict_containment_rows(ids, truth_ids), label=f"{candidate} strict"
        )
        tie = summarize(
            tie_aware_rows(cosines, ids, kth, k=GRAPH_K),
            label=f"{candidate} tie-aware",
        )
        validity = graph_validity(ids, rows=RECALL_ROWS)
        recalls[candidate] = {
            "graph_available": True,
            "graph_path": graph_path,
            "strict_containment": strict,
            "tie_aware_containment": tie,
            "graph_validity": validity,
            "zero_degree_rows": int(validity["zero_degree_rows"]),
            "rows_below_k": int(validity["rows_below_k"]),
            "cosines_recomputed_here": True,
            "independence_note": (
                "candidate cosines are recomputed in this node from the sealed "
                "substrate and the emitted ids; the builder's own accumulator is "
                "not consulted (review-0216-01)"
            ),
            "recall_mean_floor": RECALL_MEAN_FLOOR,
            "recall_p10_floor": RECALL_P10_FLOOR,
            "clears_recall_floor": bool(
                tie["mean"] >= RECALL_MEAN_FLOOR and tie["p10"] >= RECALL_P10_FLOOR
            ),
            "wall_seconds_at_largest_n": largest_wall,
        }

    verdicts = {
        candidate: (qualification["per_candidate"][candidate]["device_verdict_100m"] or {})
        for candidate in CANDIDATES
    }
    recommendation = rung_recommendation(verdicts=verdicts, recalls=recalls)

    tripwire_ok = all(
        entry.get("zero_degree_rows") in (0, None) for entry in recalls.values()
    )
    execution_checks = {
        "truth_probe_passed": bool(truth_receipt["probe"]["passed"]),
        "both_candidates_scored_or_explained": len(recalls) == len(CANDIDATES),
        "strict_and_tie_aware_both_reported": all(
            ("strict_containment" in entry and "tie_aware_containment" in entry)
            or not entry.get("graph_available")
            for entry in recalls.values()
        ),
        "degree_zero_tripwire_evaluated": all(
            "zero_degree_rows" in entry for entry in recalls.values()
        ),
        "no_candidate_with_edgeless_rows_recommended": all(
            (recalls.get(item["builder"], {}).get("zero_degree_rows") == 0)
            for item in recommendation["rungs"].values()
            if item.get("builder")
        ),
    }
    if not all(execution_checks.values()):
        raise Round0226Error(f"R0226 evaluation checks failed: {execution_checks}")

    receipt = prompt_contract.seal({
        "schema": RECALL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [],
        "outcome": "graph-builder-recall-tripwire-and-phase2-rung-recommendation",
        "qualification_manifest": expected_input_signature(qualification_path),
        "truth": {
            "receipt": expected_input_signature(TRUTH_RECEIPT_PATH),
            "identity_sha256": truth_receipt["identity_sha256"],
            "probe": truth_receipt["probe"],
            "note": (
                "R0220's recomputed exact k15 truth over R0216's "
                "queue-correction-3 substrate; its registered probe against "
                "R0216's sealed adjacency passed and review-0220-01 reproduced "
                "it independently"
            ),
        },
        "recall_rows": RECALL_ROWS,
        "recall_mean_floor": RECALL_MEAN_FLOOR,
        "recall_p10_floor": RECALL_P10_FLOOR,
        "duplicate_caveat": (
            "the substrate provably contains exact-duplicate clusters, one with "
            "1,377 members, so strict containment understates a builder's "
            "quality; both measures are reported and neither is dropped"
        ),
        "recalls": recalls,
        "device_verdicts_100m": verdicts,
        "degree_zero_tripwire_clean": bool(tripwire_ok),
        "phase2_rungs": list(PHASE2_RUNGS),
        "recommendation": recommendation,
        "execution_checks": execution_checks,
        "training_performed": False,
        "gate_registered": False,
        "map_decision_made": False,
        "production_or_publishing": False,
        "adoption_claimed": False,
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2),
        },
    })
    atomic_write_new_json(
        os.path.join(output, "graph-builder-recall-and-verdict.json"),
        receipt,
        immutable=True,
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0226Error("R0226 handler received another queue")
    action = str(job.get("action") or "")
    if action == QUALIFY_ACTION:
        run_qualify(active, job)
    elif action == EVALUATE_ACTION:
        run_evaluate(active, job)
    else:
        raise Round0226Error(f"unknown R0226 action {action!r}")


__all__ = [
    "BuildWatchdog",
    "EVALUATE_ACTION",
    "QUALIFY_ACTION",
    "RECALL_MEAN_FLOOR",
    "RECALL_P10_FLOOR",
    "refused_cell",
    "run_ascending_ladder",
    "run_evaluate",
    "run_job",
    "run_qualify",
    "skipped_cell",
]
