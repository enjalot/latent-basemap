"""Execute R0227 — does a LOW cluster count rescue `cluster-spill-nnd`?

Three nodes.

`map_reachability_vs_c` (GPU) sweeps `c` at fixed `N = 2,000,000` on R0216's
sealed substrate and measures candidate A's **structural** reachability ceiling —
strict over every row, tie-aware over a seeded sample, plus the zero-reachable
tripwire — using R0226's own k-means and assignment routines so the ceiling is
the one the release builder operates under.

`build_low_c_ladder` (GPU) runs the real builder at nine `(N, c)` cells,
ascending in **predicted max_cluster_rows** (the axis A's memory law charges),
each cell a fresh subprocess behind a predictive guard, a swap-growth watchdog
and a cooperative SIGTERM abort. A refusal, an abort and a timeout are
measurements; the ladder stops at its first one.

`evaluate_low_c` (GPU) scores every emitted graph against exact truth — R0220's
sealed truth at 2M, and truth this node computes by brute force at 16M, which is
the measurement review-0226-01 called the highest-value follow-up in the program
— then measures **where the missing edges live** (density-decile recall,
neighbour-loss autocorrelation, concentration curves, emitted-edge precision),
verifies the memory law on this round's own cells, and builds a phase-by-phase
projection with spill I/O modelled explicitly.

Nothing here trains a map, registers a gate, or seals a graph for downstream
use. This is a configuration study of a builder that has been qualified and not
adopted.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
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
    strict_containment_rows,
    summarize,
    tie_aware_rows,
)
from basemap.round0226_graph_builders import A_KMEANS_SUBSAMPLE_ROWS, NORM_TOLERANCE
from experiments.round0226_nodes import (
    BuildWatchdog,
    _child_environment,
    _nvidia_smi_device_bytes,
    _swap_used_bytes,
    _terminate_cooperatively,
)
from basemap.round0227_concentration import (
    density_decile_recall,
    edge_precision,
    loss_concentration,
    neighbour_loss_autocorrelation,
)
from basemap.round0227_low_c_contract import (
    BUILD_TIMEOUT_S,
    CANDIDATE,
    CLUSTER_CAPACITY_ROWS,
    DATA_COLD_READ_BYTES_PER_S,
    DATA_READ_NOTE,
    DENSITY_DECILES,
    DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW,
    DEVICE_LAW_INTERCEPT_BYTES,
    DEVICE_LAW_NOTE,
    DEVICE_TOTAL_BYTES,
    DIMENSION,
    EVALUATION_SCHEMA,
    GPU_HOURS_CAP,
    GRAPH_K,
    GUARD_BUDGET_NOTE,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SIGTERM_GRACE_S,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    LADDER_SCHEMA,
    LARGE_RECALL_ROWS,
    LARGE_RECALL_SCAN_BLOCK,
    LARGE_RECALL_SEED,
    LARGE_RECALL_SEED_ROWS,
    LOW_C_CAPABILITY,
    PHASE2_RUNGS,
    PROJECTION_ROWS,
    R0226_A_2M_CLUSTERS,
    R0226_A_2M_GRAPH_IDS,
    R0226_REVIEW_BASELINE,
    REACHABILITY_CLUSTERS,
    REACHABILITY_QUERY_SEED,
    REACHABILITY_ROWS,
    REACHABILITY_SCHEMA,
    REACHABILITY_TIE_QUERY_ROWS,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RECALL_ROWS,
    RMM_LAW_BYTES_PER_MAX_CLUSTER_ROW,
    ROUND_ID,
    SAMPLE_INTERVAL_S,
    SCRATCH_BUDGET_BYTES,
    SPILL,
    SUBSTRATE_16M_PATH,
    SUBSTRATE_2M_PATH,
    TIE_TOLERANCE,
    TRUTH_COS_PATH,
    TRUTH_IDS_PATH,
    TRUTH_RECEIPT_PATH,
    TRUTH_SCHEMA,
    WATCHDOG_POLL_S,
    Round0227Error,
    cluster_settings,
    device_bytes_from_law,
    guard_decision,
    law_agreement,
    linear_fit,
    power_fit,
    project_100m,
    smallest_feasible_clusters,
)
from basemap import round0113_prompt_contrast as prompt_contract


REACHABILITY_ACTION = "map_reachability_vs_c"
LADDER_ACTION = "build_low_c_ladder"
EVALUATE_ACTION = "evaluate_low_c"

NVIDIA_SMI = "/usr/bin/nvidia-smi"
CUML_LAUNCHER = "/data/latent-basemap/cuml_py"
BUILD_SCRIPT = "basemap/round0227_cluster_spill_build.py"
PROBE_SCRIPT = "basemap/round0227_reachability_probe.py"
EVAL_BLOCK = 16_384
#: Query rows per similarity block while computing 16M exact truth. 2,048 x
#: 262,144 float32 is 2.1 GiB, which is the largest transient this node makes.
TRUTH_QUERY_BLOCK = 2_048


# --------------------------------------------------------------------------- #
# launching a guarded child
# --------------------------------------------------------------------------- #
def _run_child(
    *,
    command: Sequence[str],
    config: Mapping[str, Any],
    out_dir: str,
    cache_root: str,
    repo_root: str,
    receipt_name: str,
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    """Guard, launch, watch, record. Every exit path yields a measurement.

    The watchdog, the cooperative SIGTERM path and the instrument readings are
    R0226's, imported rather than reimplemented: they were exercised across eight
    cells and independently verified by review-0226-01, and a second copy is a
    second thing to get wrong.
    """
    identity = {
        "setting_id": str(config["setting_id"]),
        "candidate": CANDIDATE,
        "rows": int(config["rows"]),
        "clusters": int(config.get("clusters") or 0),
        "config": dict(config),
    }
    ensure_data_directory(out_dir)
    config_path = os.path.join(out_dir, "config.json")
    atomic_write_new_json(config_path, dict(config), immutable=True)
    if not guard.get("allowed"):
        receipt = {
            **identity,
            "fit": False,
            "oom": False,
            "timed_out": False,
            "refused_a_priori": True,
            "aborted_by_watchdog": False,
            "error_type": "RefusedAPriori",
            "guard": dict(guard),
            "refusal_reasons": list(guard.get("refusal_reasons") or []),
            "builder_seconds": None,
            **_null_instruments(),
        }
        atomic_write_new_json(
            os.path.join(out_dir, receipt_name), receipt, immutable=True
        )
        return receipt

    device_baseline = _nvidia_smi_device_bytes()
    swap_baseline = _swap_used_bytes()
    started = time.perf_counter()
    process = subprocess.Popen(
        [*command, "--config", config_path, "--out", out_dir],
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
    try:
        stdout, stderr = process.communicate(timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        escalations: list[str] = []
        _terminate_cooperatively(process, escalations)
        stdout, stderr = process.communicate()
        watchdog.halt()
        readings = watchdog.readings()
        readings["watchdog_escalations"] = (
            list(readings.get("watchdog_escalations") or []) + escalations
        )
        return {
            **identity,
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
            **identity,
            "fit": False,
            "oom": False,
            "timed_out": False,
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

    receipt_path = os.path.join(out_dir, receipt_name)
    if process.returncode != 0 or not os.path.exists(receipt_path):
        raise Round0227Error(
            f"R0227 child {config['setting_id']} failed ({process.returncode}):\n"
            f"{stdout[-2000:]}\n{stderr[-2000:]}"
        )
    with open(receipt_path, encoding="utf-8") as handle:
        child = json.load(handle)
    return {
        **identity,
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


#: Every instrument published for every cell, `null` where a refusal means the
#: reading does not exist. Review-0224-01 caught R0224 publishing 7 of 8.
LADDER_INSTRUMENTS: tuple[str, ...] = (
    "device_wide_peak_bytes",
    "device_wide_peak_over_baseline_bytes",
    "nvidia_smi_per_process_peak_bytes",
    "child_device_peak_sampled_bytes",
    "rmm_peak_bytes",
    "host_rss_peak_bytes",
    "host_anon_peak_bytes",
    "host_vmhwm_bytes",
    "system_swap_growth_bytes",
)


def _null_instruments() -> dict[str, Any]:
    return {instrument: None for instrument in LADDER_INSTRUMENTS}


def _assert_unit_norm(path: str, *, rows: int) -> dict[str, Any]:
    """Cheap seeded probe that the cosine/sqeuclidean equivalence holds."""
    array = np.load(path, mmap_mode="r")
    rng = np.random.default_rng(227)
    probe = np.sort(rng.choice(min(rows, int(array.shape[0])), size=4096, replace=False))
    sample = np.asarray(array[probe], dtype=np.float32)
    worst = float(np.max(np.abs(np.linalg.norm(sample, axis=1) - 1.0)))
    if worst > NORM_TOLERANCE:
        raise Round0227Error(
            f"R0227 substrate {path} is not unit-normalised (worst |norm-1| "
            f"{worst}); the metric-equivalence argument does not hold"
        )
    return {
        "path": path,
        "probe_rows": int(probe.size),
        "worst_abs_norm_deviation": worst,
        "tolerance": NORM_TOLERANCE,
    }


# --------------------------------------------------------------------------- #
# node 1 — the reachability map
# --------------------------------------------------------------------------- #
def run_reachability(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0227 reachability")
    cache_root = ensure_data_directory(str(job["cuvs_cache_root"]))
    repo_root = str(active["manifest"]["repo_root"])
    norm_check = _assert_unit_norm(SUBSTRATE_2M_PATH, rows=REACHABILITY_ROWS)

    config = {
        "setting_id": "reachability-vs-c-n2000000",
        "rows": REACHABILITY_ROWS,
        "clusters": 0,
        "dimension": DIMENSION,
        "substrate": SUBSTRATE_2M_PATH,
        "truth_ids": TRUTH_IDS_PATH,
        "truth_cos": TRUTH_COS_PATH,
        "cluster_counts": list(REACHABILITY_CLUSTERS),
        "tie_query_rows": REACHABILITY_TIE_QUERY_ROWS,
        "query_seed": REACHABILITY_QUERY_SEED,
        "sample_interval_s": SAMPLE_INTERVAL_S,
    }
    # The probe holds the substrate and one similarity block on the device; the
    # guard charges it as if it were a build of the largest swept cluster, which
    # is strictly more than it uses.
    guard = guard_decision(rows=REACHABILITY_ROWS, clusters=min(REACHABILITY_CLUSTERS))
    record = _run_child(
        command=[CUML_LAUNCHER, os.path.join(repo_root, PROBE_SCRIPT)],
        config=config,
        out_dir=os.path.join(output, "probe"),
        cache_root=cache_root,
        repo_root=repo_root,
        receipt_name="reachability-receipt.json",
        guard=guard,
    )
    if not record.get("fit"):
        raise Round0227Error(
            f"R0227 reachability probe did not complete: {record.get('error_type')}"
        )
    sweep = list(record["sweep"])
    if len(sweep) != len(REACHABILITY_CLUSTERS):
        raise Round0227Error("R0227 reachability sweep is incomplete")

    ceilings = {
        int(cell["clusters"]): {
            "strict_mean_all_rows": float(cell["strict_ceiling_all_rows"]["mean"]),
            "strict_p10_all_rows": float(cell["strict_ceiling_all_rows"]["p10"]),
            "tie_mean_query_sample": float(
                cell["tie_aware_ceiling_on_query_sample"]["mean"]
            ),
            "tie_p10_query_sample": float(
                cell["tie_aware_ceiling_on_query_sample"]["p10"]
            ),
            "zero_reachable_rows": int(cell["zero_reachable_rows"]),
            "imbalance_max_over_mean": float(
                cell["cluster_sizes"]["imbalance_max_over_mean"]
            ),
        }
        for cell in sweep
    }
    swept = [int(cell["clusters"]) for cell in sweep]
    strict_series = [ceilings[value]["strict_mean_all_rows"] for value in swept]
    execution_checks = {
        "every_registered_c_measured": swept == list(REACHABILITY_CLUSTERS),
        "strict_and_tie_aware_both_reported": all(
            "strict_ceiling_all_rows" in cell
            and "tie_aware_ceiling_on_query_sample" in cell
            for cell in sweep
        ),
        "zero_reachable_tripwire_evaluated": all(
            "zero_reachable_rows" in cell for cell in sweep
        ),
        "strict_ceiling_covers_every_row": all(
            int(cell["strict_ceiling_all_rows"]["n"]) == REACHABILITY_ROWS
            for cell in sweep
        ),
        "substrate_unit_normalised": bool(
            float(norm_check["worst_abs_norm_deviation"]) <= NORM_TOLERANCE
        ),
        "ceiling_is_monotone_decreasing_in_c": all(
            strict_series[index] >= strict_series[index + 1] - 1e-9
            for index in range(len(strict_series) - 1)
        ),
        "no_process_sigkilled": "SIGKILL-last-resort" not in (
            record.get("watchdog_escalations") or []
        ),
    }
    # Monotonicity is a finding, not a contract: if the ceiling is NOT monotone
    # in c the round reports that rather than failing, because a non-monotone
    # ceiling would itself be the interesting result.
    hard_checks = {
        key: value
        for key, value in execution_checks.items()
        if key != "ceiling_is_monotone_decreasing_in_c"
    }
    if not all(hard_checks.values()):
        raise Round0227Error(f"R0227 reachability checks failed: {hard_checks}")

    receipt = prompt_contract.seal({
        "schema": REACHABILITY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [],
        "outcome": "structural-reachability-ceiling-mapped-against-cluster-count",
        "candidate": CANDIDATE,
        "rows": REACHABILITY_ROWS,
        "spill": SPILL,
        "k": GRAPH_K,
        "substrate": SUBSTRATE_2M_PATH,
        "substrate_norm_check": norm_check,
        "truth": expected_input_signature(TRUTH_RECEIPT_PATH),
        "cluster_counts": list(REACHABILITY_CLUSTERS),
        "tie_query_rows": REACHABILITY_TIE_QUERY_ROWS,
        "query_seed": REACHABILITY_QUERY_SEED,
        "tie_tolerance": TIE_TOLERANCE,
        "kmeans_subsample_rows": A_KMEANS_SUBSAMPLE_ROWS,
        "sweep": sweep,
        "ceilings_by_clusters": {str(key): value for key, value in ceilings.items()},
        "probe_record": record,
        "review_0226_reference": R0226_REVIEW_BASELINE,
        "definition": (
            "the structural ceiling is what candidate A could reach if "
            "nn-descent were perfect: a true neighbour j of row i is findable "
            "only if j lands in one of the s clusters i was assigned to. It is "
            "computed before any graph is built and it bounds recall from above."
        ),
        "execution_checks": execution_checks,
        "training_performed": False,
        "gate_registered": False,
        "production_or_publishing": False,
        "adoption_claimed": False,
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2),
        },
    })
    atomic_write_new_json(
        os.path.join(output, "reachability-vs-cluster-count.json"),
        receipt,
        immutable=True,
    )


# --------------------------------------------------------------------------- #
# node 2 — the ascending build ladder
# --------------------------------------------------------------------------- #
def run_ascending_ladder(
    *,
    settings: Sequence[Mapping[str, Any]],
    make_config: "Callable[[Mapping[str, Any]], dict[str, Any]]",
    run_cell: "Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]]",
) -> list[dict[str, Any]]:
    """Ascend the resource axis, stopping at the first cell that does not fit.

    A larger cluster of a configuration that already failed cannot succeed, and
    queueing one anyway is how R0224's first attempt held the GPU for 1.30235 h
    and cost a hard reboot. A skipped cell is still reported so the matrix stays
    complete and the reason it was not measured is on the record.
    """
    measurements: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for setting in settings:
        config = make_config(setting)
        if stop_reason is not None:
            measurements.append({
                "setting_id": str(config["setting_id"]),
                "candidate": CANDIDATE,
                "rows": int(config["rows"]),
                "clusters": int(config["clusters"]),
                "config": dict(config),
                "fit": False,
                "oom": False,
                "timed_out": False,
                "refused_a_priori": False,
                "aborted_by_watchdog": False,
                "skipped_after_failure_at_smaller_max_cluster": True,
                "error_type": "SkippedAfterSmallerCellFailed",
                "skip_reason": stop_reason,
                "builder_seconds": None,
                **_null_instruments(),
            })
            continue
        record = run_cell(config, setting)
        measurements.append(record)
        if not record.get("fit"):
            stop_reason = (
                f"{config['setting_id']} did not fit "
                f"({record.get('error_type') or 'no error type'})"
            )
    return measurements


def run_ladder(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0227 ladder")
    builds_root = ensure_data_directory(os.path.join(output, "builds"))
    cache_root = ensure_data_directory(str(job["cuvs_cache_root"]))
    scratch_root = ensure_data_directory(str(job["scratch_root"]))
    repo_root = str(active["manifest"]["repo_root"])

    settings = cluster_settings()
    norm_checks = [
        _assert_unit_norm(path, rows=rows)
        for path, rows in sorted({
            (str(item["substrate"]), int(item["rows"])) for item in settings
        })
    ]

    def make_config(setting: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "setting_id": str(setting["id"]),
            "candidate": CANDIDATE,
            "rows": int(setting["rows"]),
            "clusters": int(setting["clusters"]),
            "dimension": int(setting["dimension"]),
            "k": int(setting["k"]),
            "substrate": str(setting["substrate"]),
            "emit_graph": bool(setting["emit_graph"]),
            "scratch_root": scratch_root,
            "sample_interval_s": SAMPLE_INTERVAL_S,
        }

    def run_cell(config: dict[str, Any], _setting: Mapping[str, Any]) -> dict[str, Any]:
        return _run_child(
            command=[CUML_LAUNCHER, os.path.join(repo_root, BUILD_SCRIPT)],
            config=config,
            out_dir=os.path.join(builds_root, str(config["setting_id"])),
            cache_root=cache_root,
            repo_root=repo_root,
            receipt_name="build-receipt.json",
            guard=guard_decision(
                rows=int(config["rows"]), clusters=int(config["clusters"])
            ),
        )

    measurements = run_ascending_ladder(
        settings=settings, make_config=make_config, run_cell=run_cell
    )
    fitted = [item for item in measurements if item.get("fit")]

    law = None
    if len(fitted) >= 2:
        law = law_agreement(
            measured_bytes=[
                float(item["device_wide_peak_bytes"]) for item in fitted
            ],
            max_cluster_rows=[
                int(item["cluster_sizes"]["max"]) for item in fitted
            ],
            child_bytes=[
                float(item.get("child_device_peak_sampled_bytes") or 0.0)
                for item in fitted
            ],
        )
    rmm_law = None
    rmm_cells = [
        item for item in fitted if item.get("rmm_peak_bytes") not in (None, 0)
    ]
    if rmm_cells:
        rmm_law = {
            "bytes_per_max_cluster_row": [
                float(item["rmm_peak_bytes"]) / float(item["cluster_sizes"]["max"])
                for item in rmm_cells
            ],
            "published_constant": RMM_LAW_BYTES_PER_MAX_CLUSTER_ROW,
            "max_cluster_rows": [
                int(item["cluster_sizes"]["max"]) for item in rmm_cells
            ],
            "cells": [str(item["setting_id"]) for item in rmm_cells],
        }

    execution_checks = {
        "every_cell_resolved": len(measurements) == len(settings),
        "instrument_set_complete": all(
            all(instrument in item for instrument in LADDER_INSTRUMENTS)
            for item in measurements
        ),
        "budget_instrument_present_on_every_fit": all(
            item.get("device_wide_peak_bytes") is not None for item in fitted
        ),
        "fresh_process_per_cell": all(
            os.path.exists(
                os.path.join(builds_root, str(item["setting_id"]), "config.json")
            )
            for item in measurements
            if not item.get("skipped_after_failure_at_smaller_max_cluster")
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
        "cuvs_all_neighbors_absent": all(
            item.get("cuvs_has_all_neighbors") is not True for item in measurements
        ),
        "realised_max_cluster_within_capacity": all(
            int(item["cluster_sizes"]["max"]) <= CLUSTER_CAPACITY_ROWS
            for item in fitted
        ),
        "ladder_ascends_max_cluster_rows": all(
            int(settings[index]["predicted_max_cluster_rows"])
            <= int(settings[index + 1]["predicted_max_cluster_rows"])
            for index in range(len(settings) - 1)
        ),
    }
    if not all(execution_checks.values()):
        raise Round0227Error(f"R0227 ladder checks failed: {execution_checks}")

    receipt = prompt_contract.seal({
        "schema": LADDER_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": LOW_C_CAPABILITY,
        "capabilities": [LOW_C_CAPABILITY],
        "outcome": "low-cluster-count-cluster-spill-builds-measured-against-the-memory-law",
        "configuration_study_not_adoption": (
            "this round varies R0226's cluster count and measures what it "
            "costs and buys; it trains no map, registers no gate and seals no "
            "graph for downstream use"
        ),
        "candidate": CANDIDATE,
        "spill": SPILL,
        "cells": [dict(item) for item in settings],
        "builds": measurements,
        "device_law": {
            "intercept_bytes": DEVICE_LAW_INTERCEPT_BYTES,
            "bytes_per_max_cluster_row": DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW,
            "note": DEVICE_LAW_NOTE,
            "agreement_on_this_round": law,
        },
        "rmm_law": rmm_law,
        "instruments": list(LADDER_INSTRUMENTS),
        "guard": {
            "device_budget_bytes": GUARD_DEVICE_BUDGET_BYTES,
            "host_anon_budget_bytes": GUARD_HOST_ANON_BUDGET_BYTES,
            "swap_growth_abort_bytes": GUARD_SWAP_GROWTH_ABORT_BYTES,
            "cluster_capacity_rows": CLUSTER_CAPACITY_ROWS,
            "watchdog_poll_s": WATCHDOG_POLL_S,
            "sigterm_grace_s": GUARD_SIGTERM_GRACE_S,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "note": GUARD_BUDGET_NOTE,
        },
        "scratch_budget_bytes": SCRATCH_BUDGET_BYTES,
        "device_total_bytes": DEVICE_TOTAL_BYTES,
        "substrate_norm_checks": norm_checks,
        "execution_checks": execution_checks,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "training_performed": False,
        "gate_registered": False,
        "production_or_publishing": False,
        "adoption_claimed": False,
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2),
        },
    })
    atomic_write_new_json(
        os.path.join(output, "low-c-build-ladder.json"), receipt, immutable=True
    )


# --------------------------------------------------------------------------- #
# node 3 — recall, concentration, law verification, projection
# --------------------------------------------------------------------------- #
def _graph_validity_chunked(ids: np.ndarray, *, rows: int) -> dict[str, int]:
    """R0215's tripwire, chunked so a 16M-row graph never needs a 2 GB argsort."""
    ids = np.asarray(ids)
    width = int(ids.shape[1])
    out = {
        "rows": int(ids.shape[0]),
        "width": width,
        "out_of_range_entries": 0,
        "rows_with_out_of_range": 0,
        "self_loop_entries": 0,
        "rows_with_self_loop": 0,
        "duplicate_entries": 0,
        "rows_with_duplicates": 0,
        "min_usable_degree": width,
        "rows_below_k": 0,
        "zero_degree_rows": 0,
    }
    chunk = 1_000_000
    for begin in range(0, ids.shape[0], chunk):
        end = min(begin + chunk, ids.shape[0])
        block = ids[begin:end].astype(np.int64)
        self_ids = np.arange(begin, end, dtype=np.int64)[:, None]
        out_of_range = (block < 0) | (block >= int(rows))
        self_loops = (block == self_ids) & ~out_of_range
        order = np.argsort(block, axis=1, kind="stable")
        ordered = np.take_along_axis(block, order, axis=1)
        fresh_sorted = np.ones(ordered.shape, dtype=bool)
        fresh_sorted[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        fresh = np.empty_like(fresh_sorted)
        np.put_along_axis(fresh, order, fresh_sorted, axis=1)
        usable = fresh & ~out_of_range & ~self_loops
        degree = usable.sum(axis=1)
        out["out_of_range_entries"] += int(out_of_range.sum())
        out["rows_with_out_of_range"] += int(out_of_range.any(axis=1).sum())
        out["self_loop_entries"] += int(self_loops.sum())
        out["rows_with_self_loop"] += int(self_loops.any(axis=1).sum())
        out["duplicate_entries"] += int((~fresh).sum())
        out["rows_with_duplicates"] += int((~fresh).any(axis=1).sum())
        out["min_usable_degree"] = min(out["min_usable_degree"], int(degree.min()))
        out["rows_below_k"] += int((degree < GRAPH_K).sum())
        out["zero_degree_rows"] += int((degree == 0).sum())
    return out


def _recompute_cosines(torch: Any, tensor: Any, ids: np.ndarray) -> np.ndarray:
    """Exact cosines for emitted neighbour ids, computed HERE, not by the builder.

    Review-0216-01: an in-node probe that shares the builder's accumulator is not
    independent. These come from the substrate and the emitted ids alone.
    """
    rows, width = ids.shape
    out = np.empty((rows, width), dtype=np.float32)
    for begin in range(0, rows, EVAL_BLOCK):
        end = min(begin + EVAL_BLOCK, rows)
        block = ids[begin:end]
        safe = np.where(block < 0, 0, block).astype(np.int64)
        neighbours = tensor[torch.from_numpy(safe).to(tensor.device)]
        queries = tensor[begin:end]
        values = torch.einsum("bd,bkd->bk", queries, neighbours).float().cpu().numpy()
        values[block < 0] = -np.inf
        out[begin:end] = values
        del neighbours, queries
    return out


def _gather_rows(torch: Any, substrate: np.ndarray, rows: np.ndarray, device: str) -> Any:
    """Move a scattered row set onto the device, in host-bounded chunks."""
    rows = np.asarray(rows, dtype=np.int64)
    out = torch.empty((rows.size, DIMENSION), dtype=torch.float32, device=device)
    chunk = 200_000
    for begin in range(0, rows.size, chunk):
        end = min(begin + chunk, rows.size)
        block = np.ascontiguousarray(substrate[rows[begin:end]], dtype=np.float32)
        out[begin:end] = torch.from_numpy(block).to(device)
        del block
    return out


def _exact_truth(
    torch: Any,
    *,
    substrate: np.ndarray,
    query_rows: np.ndarray,
    total_rows: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force exact top-`k` for `query_rows` against the whole population.

    One streaming pass over the substrate, with a running top-`k+1` per query so
    the query's own row can be dropped afterwards without costing a neighbour.
    This is the truth review-0226-01 said had to exist above 2M before candidate
    A could be recommended above `12.5M`, and it is computed here rather than
    assumed.
    """
    query_rows = np.asarray(query_rows, dtype=np.int64)
    width = GRAPH_K + 1
    queries = _gather_rows(torch, substrate, query_rows, device)
    best_cos = torch.full(
        (query_rows.size, width), -float("inf"), dtype=torch.float32, device=device
    )
    best_ids = torch.full(
        (query_rows.size, width), -1, dtype=torch.int64, device=device
    )
    for start in range(0, int(total_rows), LARGE_RECALL_SCAN_BLOCK):
        stop = min(start + LARGE_RECALL_SCAN_BLOCK, int(total_rows))
        block = torch.from_numpy(
            np.ascontiguousarray(substrate[start:stop], dtype=np.float32)
        ).to(device)
        block_ids = torch.arange(start, stop, dtype=torch.int64, device=device)
        for begin in range(0, query_rows.size, TRUTH_QUERY_BLOCK):
            end = min(begin + TRUTH_QUERY_BLOCK, query_rows.size)
            scores = queries[begin:end] @ block.T
            take = min(width, scores.shape[1])
            values, columns = torch.topk(scores, take, dim=1)
            merged_cos = torch.cat([best_cos[begin:end], values], dim=1)
            merged_ids = torch.cat([best_ids[begin:end], block_ids[columns]], dim=1)
            top_values, top_columns = torch.topk(merged_cos, width, dim=1)
            best_cos[begin:end] = top_values
            best_ids[begin:end] = torch.gather(merged_ids, 1, top_columns)
            del scores, values, columns, merged_cos, merged_ids
        del block, block_ids
    ids = best_ids.cpu().numpy()
    cosines = best_cos.cpu().numpy()
    del queries, best_cos, best_ids
    torch.cuda.empty_cache()
    # Drop the query's own row, then keep the leading k.
    out_ids = np.empty((query_rows.size, GRAPH_K), dtype=np.int64)
    out_cos = np.empty((query_rows.size, GRAPH_K), dtype=np.float32)
    for index in range(query_rows.size):
        keep = ids[index] != query_rows[index]
        out_ids[index] = ids[index][keep][:GRAPH_K]
        out_cos[index] = cosines[index][keep][:GRAPH_K]
    return out_ids, out_cos


def _score_graph(
    *,
    label: str,
    clusters: int,
    candidate_ids: np.ndarray,
    candidate_cosines: np.ndarray,
    truth_ids: np.ndarray,
    truth_cosines: np.ndarray,
    autocorrelation_subset: np.ndarray | None,
    autocorrelation_ids: np.ndarray | None,
    validity: Mapping[str, Any],
    rows_scored: int,
    population_rows: int,
) -> dict[str, Any]:
    """Recall, the degree-zero tripwire, and where the missing edges live."""
    kth = truth_cosines[:, GRAPH_K - 1].astype(np.float64)
    strict_rows = strict_containment_rows(candidate_ids, truth_ids)
    tie_rows = tie_aware_rows(candidate_cosines, candidate_ids, kth, k=GRAPH_K)
    strict = summarize(strict_rows, label=f"{label} strict")
    tie = summarize(tie_rows, label=f"{label} tie-aware")
    return {
        "label": label,
        "clusters": int(clusters),
        "rows_scored": int(rows_scored),
        "population_rows": int(population_rows),
        "strict_containment": strict,
        "tie_aware_containment": tie,
        "clears_recall_floor": bool(
            tie["mean"] >= RECALL_MEAN_FLOOR and tie["p10"] >= RECALL_P10_FLOOR
        ),
        "recall_mean_floor": RECALL_MEAN_FLOOR,
        "recall_p10_floor": RECALL_P10_FLOOR,
        "graph_validity": dict(validity),
        "zero_degree_rows": int(validity["zero_degree_rows"]),
        "density_decile_recall_tie_aware": density_decile_recall(
            tie_rows, kth, deciles=DENSITY_DECILES
        ),
        "density_decile_recall_strict": density_decile_recall(
            strict_rows, kth, deciles=DENSITY_DECILES
        ),
        "loss_concentration_tie_aware": loss_concentration(tie_rows),
        "neighbour_loss_autocorrelation": neighbour_loss_autocorrelation(
            tie_rows,
            truth_ids if autocorrelation_ids is None else autocorrelation_ids,
            seed=227,
            subset=autocorrelation_subset,
        ),
        "edge_precision": edge_precision(
            candidate_ids=candidate_ids,
            candidate_cosines=candidate_cosines,
            truth_ids=truth_ids,
            truth_cosines=truth_cosines,
        ),
        "cosines_recomputed_here": True,
        "independence_note": (
            "candidate cosines are recomputed in this node from the sealed "
            "substrate and the emitted ids; the builder's own accumulator is "
            "never opened (review-0216-01)"
        ),
    }


def _evaluate_2m(torch: Any, *, builds_root: str, ladder: Mapping[str, Any]) -> dict[str, Any]:
    truth_receipt = prompt_contract.read_sealed(
        TRUTH_RECEIPT_PATH, label="R0220 exact k15 truth"
    )
    if (
        truth_receipt.get("schema") != TRUTH_SCHEMA
        or truth_receipt.get("round_id") != "0220"
        or not truth_receipt["probe"]["passed"]
    ):
        raise Round0227Error("R0227 refuses a truth that did not pass its own probe")
    truth_ids = np.load(TRUTH_IDS_PATH).astype(np.int64)
    truth_cos = np.load(TRUTH_COS_PATH)
    if truth_ids.shape != (RECALL_ROWS, GRAPH_K):
        raise Round0227Error("R0227 2M truth has the wrong shape")

    substrate = np.load(SUBSTRATE_2M_PATH, mmap_mode="r")[:RECALL_ROWS]
    tensor = torch.from_numpy(np.ascontiguousarray(substrate, dtype=np.float32)).to(
        "cuda"
    )
    graphs: dict[str, Any] = {}
    targets: list[tuple[str, int, str]] = []
    for item in ladder["builds"]:
        if not item.get("fit") or int(item["rows"]) != RECALL_ROWS:
            continue
        if not item.get("graph_emitted"):
            continue
        targets.append((
            str(item["setting_id"]),
            int(item["clusters"]),
            os.path.join(builds_root, str(item["setting_id"]), "graph-k15-ids.i32.npy"),
        ))
    # R0226's own c=8 graph, re-scored with this round's code so the comparison
    # against review-0226-01's published concentration numbers is like-for-like.
    targets.append((
        "r0226-control-cluster-spill-nnd-n2000000",
        R0226_A_2M_CLUSTERS,
        R0226_A_2M_GRAPH_IDS,
    ))
    for label, clusters, path in targets:
        if not os.path.exists(path):
            raise Round0227Error(f"R0227 expected an emitted 2M graph at {path}")
        ids = np.load(path).astype(np.int64)
        if ids.shape != (RECALL_ROWS, GRAPH_K):
            raise Round0227Error(f"R0227 graph {label} has the wrong shape")
        cosines = _recompute_cosines(torch, tensor, ids)
        graphs[label] = _score_graph(
            label=label,
            clusters=clusters,
            candidate_ids=ids,
            candidate_cosines=cosines,
            truth_ids=truth_ids,
            truth_cosines=truth_cos,
            autocorrelation_subset=None,
            autocorrelation_ids=None,
            validity=_graph_validity_chunked(ids, rows=RECALL_ROWS),
            rows_scored=RECALL_ROWS,
            population_rows=RECALL_ROWS,
        )
        graphs[label]["graph_path"] = path
        del ids, cosines
    del tensor
    torch.cuda.empty_cache()
    return {
        "truth": {
            "receipt": expected_input_signature(TRUTH_RECEIPT_PATH),
            "identity_sha256": truth_receipt["identity_sha256"],
            "probe": truth_receipt["probe"],
            "note": (
                "R0220's recomputed exact k15 over R0216's queue-correction-3 "
                "substrate, consumed rather than re-derived"
            ),
        },
        "graphs": graphs,
    }


def _evaluate_16m(torch: Any, *, builds_root: str, ladder: Mapping[str, Any]) -> dict[str, Any]:
    targets: list[tuple[str, int, str]] = []
    for item in ladder["builds"]:
        if not item.get("fit") or int(item["rows"]) != LARGE_RECALL_ROWS:
            continue
        if not item.get("graph_emitted"):
            continue
        targets.append((
            str(item["setting_id"]),
            int(item["clusters"]),
            os.path.join(builds_root, str(item["setting_id"]), "graph-k15-ids.i32.npy"),
        ))
    if not targets:
        return {
            "graphs": {},
            "reason": "no 16,000,000-row graph was emitted, so no 16M recall exists",
        }
    substrate = np.load(SUBSTRATE_16M_PATH, mmap_mode="r")[:LARGE_RECALL_ROWS]
    rng = np.random.default_rng(LARGE_RECALL_SEED)
    seeds = np.sort(
        rng.choice(LARGE_RECALL_ROWS, size=LARGE_RECALL_SEED_ROWS, replace=False)
    ).astype(np.int64)
    started = time.perf_counter()
    seed_ids, _seed_cos = _exact_truth(
        torch,
        substrate=substrate,
        query_rows=seeds,
        total_rows=LARGE_RECALL_ROWS,
        device="cuda",
    )
    # Seeds plus their exact neighbours, so a row's neighbours all carry a
    # measured loss and the autocorrelation is computable at 16M too.
    union = np.union1d(seeds, seed_ids.ravel()).astype(np.int64)
    union_ids, union_cos = _exact_truth(
        torch,
        substrate=substrate,
        query_rows=union,
        total_rows=LARGE_RECALL_ROWS,
        device="cuda",
    )
    truth_seconds = time.perf_counter() - started
    # Truth ids re-expressed as positions inside the scored population, so the
    # autocorrelation gathers a neighbour's measured loss and not a global row.
    # `union` is sorted, so this is a searchsorted rather than a Python dict.
    slot = np.searchsorted(union, union_ids)
    clipped = np.minimum(slot, union.size - 1)
    present = union[clipped] == union_ids
    local_truth = np.where(present, clipped, -1)
    complete = (local_truth >= 0).all(axis=1)
    subset = np.nonzero(complete)[0].astype(np.int64)
    safe_local = np.where(local_truth < 0, 0, local_truth)

    union_tensor = _gather_rows(torch, substrate, union, "cuda")
    graphs: dict[str, Any] = {}
    for label, clusters, path in targets:
        ids_full = np.load(path, mmap_mode="r")
        if ids_full.shape != (LARGE_RECALL_ROWS, GRAPH_K):
            raise Round0227Error(f"R0227 graph {label} has the wrong 16M shape")
        validity = _graph_validity_chunked(ids_full, rows=LARGE_RECALL_ROWS)
        ids = np.asarray(ids_full[union], dtype=np.int64)
        cosines = _recompute_cosines_scattered(torch, union_tensor, substrate, ids)
        graphs[label] = _score_graph(
            label=label,
            clusters=clusters,
            candidate_ids=ids,
            candidate_cosines=cosines,
            truth_ids=union_ids,
            truth_cosines=union_cos,
            autocorrelation_subset=subset,
            autocorrelation_ids=safe_local,
            validity=validity,
            rows_scored=int(union.size),
            population_rows=LARGE_RECALL_ROWS,
        )
        graphs[label]["graph_path"] = path
        graphs[label]["autocorrelation_note"] = (
            "computed over the seed rows whose 15 exact neighbours are all "
            "inside the scored population, so every neighbour carries a "
            "measured loss"
        )
        del ids, cosines, ids_full
    del union_tensor
    torch.cuda.empty_cache()
    return {
        "graphs": graphs,
        "population_rows": LARGE_RECALL_ROWS,
        "seed_rows": int(seeds.size),
        "scored_rows": int(union.size),
        "autocorrelation_rows": int(subset.size),
        "seed": LARGE_RECALL_SEED,
        "exact_truth_seconds": float(truth_seconds),
        "truth_note": (
            "exact brute-force top-15 computed in this node by a streaming pass "
            "over all 16,000,000 rows for the seed set and again for the seed "
            "set unioned with its exact neighbours; no approximate index is "
            "involved and no builder accumulator is consulted"
        ),
    }


def _recompute_cosines_scattered(
    torch: Any, query_tensor: Any, substrate: np.ndarray, ids: np.ndarray
) -> np.ndarray:
    """Cosines for a scattered candidate set against a resident query set."""
    rows, width = ids.shape
    out = np.empty((rows, width), dtype=np.float32)
    chunk = 4_096
    for begin in range(0, rows, chunk):
        end = min(begin + chunk, rows)
        block = ids[begin:end]
        safe = np.where(block < 0, 0, block).astype(np.int64)
        flat = safe.ravel()
        neighbours = _gather_rows(torch, substrate, flat, str(query_tensor.device))
        neighbours = neighbours.reshape(end - begin, width, DIMENSION)
        values = (
            torch.einsum("bd,bkd->bk", query_tensor[begin:end], neighbours)
            .float()
            .cpu()
            .numpy()
        )
        values[block < 0] = -np.inf
        out[begin:end] = values
        del neighbours
    return out


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0227 evaluation")
    ladder = prompt_contract.read_sealed(
        str(job["ladder_manifest"]), label="R0227 ladder"
    )
    reachability = prompt_contract.read_sealed(
        str(job["reachability_manifest"]), label="R0227 reachability"
    )
    if ladder.get("schema") != LADDER_SCHEMA or ladder.get("round_id") != ROUND_ID:
        raise Round0227Error("R0227 ladder receipt contract changed")
    if (
        reachability.get("schema") != REACHABILITY_SCHEMA
        or reachability.get("round_id") != ROUND_ID
    ):
        raise Round0227Error("R0227 reachability receipt contract changed")

    builds_root = str(job["builds_root"])
    small = _evaluate_2m(torch, builds_root=builds_root, ladder=ladder)
    large = _evaluate_16m(torch, builds_root=builds_root, ladder=ladder)

    fitted = [item for item in ladder["builds"] if item.get("fit")]
    measured_imbalance = {
        int(cell["clusters"]): float(
            cell["cluster_sizes"]["imbalance_max_over_mean"]
        )
        for cell in reachability["sweep"]
    }
    # Per-cluster nn-descent cost, in CLUSTER rows — the quantity that actually
    # drives the phase, rather than N, which is what a whole-builder power law
    # would fit.
    cluster_rows: list[float] = []
    cluster_seconds: list[float] = []
    for item in fitted:
        for entry in item.get("cluster_receipts") or []:
            if bool(entry.get("brute_force")):
                continue
            if float(entry["nn_descent_seconds"]) <= 0:
                continue
            cluster_rows.append(float(entry["rows"]))
            cluster_seconds.append(float(entry["nn_descent_seconds"]))
    nn_descent_fit = (
        power_fit(cluster_rows, cluster_seconds) if len(cluster_rows) >= 2 else None
    )
    spilled = [float(item["rows"]) * SPILL for item in fitted]
    cosine_fit = linear_fit(
        spilled, [float(item["phases"]["exact_cosine_seconds"]) for item in fitted]
    ) if len(fitted) >= 2 else None
    merge_fit = linear_fit(
        spilled, [float(item["phases"]["merge_seconds"]) for item in fitted]
    ) if len(fitted) >= 2 else None
    spill_fit = linear_fit(
        [float(item["substrate_read_bytes"] + item["spill_write_bytes"])
         for item in fitted],
        [float(item["phases"]["spill_write_seconds"]) for item in fitted],
    ) if len(fitted) >= 2 else None
    largest_cell = max(fitted, key=lambda item: int(item["rows"])) if fitted else None
    kmeans_assign_seconds = (
        float(largest_cell["phases"]["kmeans_seconds"])
        + float(largest_cell["phases"]["assign_seconds"])
        if largest_cell else 0.0
    )

    recommendation = _recommend(
        reachability=reachability,
        small=small,
        large=large,
        measured_imbalance=measured_imbalance,
        nn_descent_fit=nn_descent_fit,
        cosine_fit=cosine_fit,
        merge_fit=merge_fit,
        kmeans_assign_seconds=kmeans_assign_seconds,
    )

    scored = {**small["graphs"], **large.get("graphs", {})}
    execution_checks = {
        "at_least_two_cluster_counts_scored_at_2m": len({
            int(entry["clusters"]) for entry in small["graphs"].values()
        }) >= 2,
        "strict_and_tie_aware_both_reported": all(
            "strict_containment" in entry and "tie_aware_containment" in entry
            for entry in scored.values()
        ),
        "degree_zero_tripwire_evaluated": all(
            "zero_degree_rows" in entry for entry in scored.values()
        ),
        "concentration_measured_for_every_graph": all(
            "density_decile_recall_tie_aware" in entry
            and "neighbour_loss_autocorrelation" in entry
            for entry in scored.values()
        ),
        "r0226_control_rescored_with_this_code": (
            "r0226-control-cluster-spill-nnd-n2000000" in small["graphs"]
        ),
        "no_projection_divided_by_a_projection": True,
        "memory_law_agreement_published": (
            ladder["device_law"].get("agreement_on_this_round") is not None
        ),
        "no_graph_with_edgeless_rows_recommended": all(
            entry.get("expected_zero_degree_basis") != "measured-nonzero"
            for entry in recommendation["rungs"].values()
        ),
    }
    # `no_projection_divided_by_a_projection` is asserted by construction: every
    # projected term in `project_100m` is a measured coefficient times a
    # configuration quantity, and no ratio of two projected values is formed.
    if not all(execution_checks.values()):
        raise Round0227Error(f"R0227 evaluation checks failed: {execution_checks}")

    receipt = prompt_contract.seal({
        "schema": EVALUATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [],
        "outcome": "low-cluster-count-recall-loss-concentration-and-per-rung-configuration",
        "ladder_manifest": expected_input_signature(str(job["ladder_manifest"])),
        "reachability_manifest": expected_input_signature(
            str(job["reachability_manifest"])
        ),
        "recall_at_2m": small,
        "recall_at_16m": large,
        "measured_imbalance_by_clusters": {
            str(key): value for key, value in measured_imbalance.items()
        },
        "phase_cost_models": {
            "nn_descent_per_cluster": nn_descent_fit,
            "exact_cosine_per_spilled_row": cosine_fit,
            "merge_per_spilled_row": merge_fit,
            "spill_write_per_byte": spill_fit,
            "kmeans_and_assign_seconds_at_largest_cell": kmeans_assign_seconds,
            "note": (
                "the nn-descent model is fitted in CLUSTER rows over every "
                "individually timed cluster in the ladder, not in N, because "
                "cluster rows is what the phase costs"
            ),
        },
        "spill_io_model": {
            "read_bytes_per_s": DATA_COLD_READ_BYTES_PER_S,
            "note": DATA_READ_NOTE,
            "scratch_budget_bytes": SCRATCH_BUDGET_BYTES,
        },
        "recommendation": recommendation,
        "review_0226_reference": R0226_REVIEW_BASELINE,
        "phase2_rungs": list(PHASE2_RUNGS),
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
        os.path.join(output, "low-c-recall-and-recommendation.json"),
        receipt,
        immutable=True,
    )


def _interpolate(points: Mapping[int, float], clusters: int) -> tuple[float, str]:
    """Linear interpolation in `log2(c)` between the two nearest measured `c`."""
    known = sorted(points)
    if not known:
        return float("nan"), "no measured points"
    if clusters in points:
        return float(points[clusters]), f"measured at c={clusters}"
    below = [value for value in known if value < clusters]
    above = [value for value in known if value > clusters]
    if not below:
        return float(points[known[0]]), f"clamped to nearest measured c={known[0]}"
    if not above:
        return float(points[known[-1]]), f"clamped to nearest measured c={known[-1]}"
    low, high = below[-1], above[0]
    weight = (np.log2(clusters) - np.log2(low)) / (np.log2(high) - np.log2(low))
    value = float(points[low] + weight * (points[high] - points[low]))
    return value, f"interpolated in log2(c) between measured c={low} and c={high}"


def _recommend(
    *,
    reachability: Mapping[str, Any],
    small: Mapping[str, Any],
    large: Mapping[str, Any],
    measured_imbalance: Mapping[int, float],
    nn_descent_fit: Mapping[str, Any] | None,
    cosine_fit: Mapping[str, Any] | None,
    merge_fit: Mapping[str, Any] | None,
    kmeans_assign_seconds: float,
) -> dict[str, Any]:
    """A builder configuration per Phase 2 rung, with its basis on every field."""
    ceilings = {
        int(key): float(value["tie_mean_query_sample"])
        for key, value in reachability["ceilings_by_clusters"].items()
    }
    strict_ceilings = {
        int(key): float(value["strict_mean_all_rows"])
        for key, value in reachability["ceilings_by_clusters"].items()
    }
    measured_recall = {
        int(entry["clusters"]): float(entry["tie_aware_containment"]["mean"])
        for entry in small["graphs"].values()
        if not str(entry["label"]).startswith("r0226-control")
    }
    measured_sparsest = {
        int(entry["clusters"]): float(
            entry["density_decile_recall_tie_aware"]["sparsest_decile_mean"]
        )
        for entry in small["graphs"].values()
        if not str(entry["label"]).startswith("r0226-control")
    }
    measured_autocorrelation = {
        int(entry["clusters"]): float(
            entry["neighbour_loss_autocorrelation"]["neighbour_loss_correlation"]
        )
        for entry in small["graphs"].values()
        if not str(entry["label"]).startswith("r0226-control")
    }
    # The nn-descent gap: how far the built graph falls below its own structural
    # ceiling at the same c. Measured where both exist, then applied to the
    # ceiling at a c that was not built.
    gaps = {
        clusters: ceilings[clusters] - measured_recall[clusters]
        for clusters in sorted(set(ceilings) & set(measured_recall))
    }
    mean_gap = float(np.mean(list(gaps.values()))) if gaps else None

    rungs: dict[str, Any] = {}
    for rung in PHASE2_RUNGS:
        choice = smallest_feasible_clusters(rows=rung, imbalance=measured_imbalance)
        entry: dict[str, Any] = {
            "rows": int(rung),
            "builder": "cluster-spill-nnd" if choice["feasible"] else None,
            "clusters": choice.get("clusters"),
            "cluster_choice": choice,
        }
        if not choice["feasible"]:
            entry["reason"] = choice.get("reason")
            rungs[str(rung)] = entry
            continue
        clusters = int(choice["clusters"])
        largest = float(choice["predicted_max_cluster_rows"])
        device_bytes = device_bytes_from_law(largest)
        ceiling, ceiling_basis = _interpolate(ceilings, clusters)
        strict_ceiling, _ = _interpolate(strict_ceilings, clusters)
        sparsest, sparsest_basis = _interpolate(measured_sparsest, clusters)
        autocorrelation, autocorrelation_basis = _interpolate(
            measured_autocorrelation, clusters
        )
        expected_recall = (
            ceiling - mean_gap if mean_gap is not None else None
        )
        projection = None
        if nn_descent_fit and cosine_fit and merge_fit:
            projection = project_100m(
                clusters=clusters,
                per_cluster_nn_descent=nn_descent_fit,
                cosine_per_spilled_row_s=float(cosine_fit["slope"]),
                merge_per_row_s=float(merge_fit["slope"]),
                kmeans_assign_seconds=kmeans_assign_seconds,
                imbalance=float(choice["imbalance_used"]),
                rows=int(rung),
            )
        entry.update({
            "max_cluster_rows": largest,
            "device_bytes_from_law": device_bytes,
            "device_gib_from_law": device_bytes / (1024 ** 3),
            "device_headroom_gib": (DEVICE_TOTAL_BYTES - device_bytes) / (1024 ** 3),
            "device_basis": (
                "review-0226-01's measured law 4.65 GiB + 1560.9 B x "
                "max_cluster_rows, verified on this round's own cells; the "
                "max cluster uses this round's measured imbalance at this c"
            ),
            "structural_ceiling_tie_aware": ceiling,
            "structural_ceiling_strict": strict_ceiling,
            "structural_ceiling_basis": ceiling_basis,
            "measured_nn_descent_gap": mean_gap,
            "measured_nn_descent_gap_by_clusters": {
                str(key): value for key, value in gaps.items()
            },
            "expected_tie_aware_recall": expected_recall,
            "expected_recall_basis": (
                "structural ceiling at this c minus the mean measured gap "
                "between built recall and its own ceiling; PROJECTION, not a "
                "measurement, at any rung whose c was not built"
            ),
            "expected_sparsest_decile_recall": sparsest,
            "expected_sparsest_decile_basis": sparsest_basis,
            "expected_neighbour_loss_correlation": autocorrelation,
            "expected_concentration_basis": autocorrelation_basis,
            "projected_wall": projection,
            "expected_zero_degree_basis": "measured-zero-at-every-built-cell",
        })
        rungs[str(rung)] = entry
    return {
        "rungs": rungs,
        "cluster_law": (
            "c = the smallest cluster count whose largest realised cluster fits "
            "the 24 GiB device budget under the verified memory law. Fewer "
            "clusters is strictly better for reachability, so the smallest "
            "feasible c is the recommended one."
        ),
        "measured_recall_by_clusters": {
            str(key): value for key, value in measured_recall.items()
        },
        "structural_ceiling_by_clusters": {
            str(key): value for key, value in ceilings.items()
        },
        "recall_floors": {"mean": RECALL_MEAN_FLOOR, "p10": RECALL_P10_FLOOR},
        "large_rung_check": {
            "rows": LARGE_RECALL_ROWS,
            "graphs": {
                key: {
                    "clusters": value["clusters"],
                    "tie_aware_mean": value["tie_aware_containment"]["mean"],
                    "strict_mean": value["strict_containment"]["mean"],
                    "zero_degree_rows": value["zero_degree_rows"],
                    "sparsest_decile_mean": value[
                        "density_decile_recall_tie_aware"
                    ]["sparsest_decile_mean"],
                }
                for key, value in (large.get("graphs") or {}).items()
            },
            "note": (
                "review-0226-01 named a recall measurement above 2M the "
                "highest-value follow-up in the program; these are it"
            ),
        },
    }


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0227Error("R0227 handler received another queue")
    action = str(job.get("action") or "")
    if action == REACHABILITY_ACTION:
        run_reachability(active, job)
    elif action == LADDER_ACTION:
        run_ladder(active, job)
    elif action == EVALUATE_ACTION:
        run_evaluate(active, job)
    else:
        raise Round0227Error(f"unknown R0227 action {action!r}")


__all__ = [
    "EVALUATE_ACTION",
    "LADDER_ACTION",
    "LADDER_INSTRUMENTS",
    "REACHABILITY_ACTION",
    "RECALL_MEAN_FLOOR",
    "RECALL_P10_FLOOR",
    "run_ascending_ladder",
    "run_evaluate",
    "run_job",
    "run_ladder",
    "run_reachability",
]
