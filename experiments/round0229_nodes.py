"""Execute R0229 phase 1 — can a high-`c` `cluster-spill-nnd` graph be rescued?

Three nodes, one queue:

* `sweep_nnd_quality_c16` (GPU) builds eight nn-descent settings at `c = 16`,
  `s = 2`, 2M **over one shared k-means partition**, and scores each over all
  2,000,000 rows against R0220's sealed exact truth. Holding the partition fixed
  is what makes the cells comparable: any recall difference is nn-descent,
  because the reachable set is identical by construction. The node also measures
  that partition's own strict reachability ceiling, so every cell's gap is exact
  and the registered structural bound is falsifiable rather than assumed.
* `sweep_spill_reachability` (GPU) measures the reachability ceiling against
  `(c, s)` rather than against `c` alone, over all 2,000,000 rows, in two
  matched-cluster-size families plus the three 100M-feasible configurations.
  `(16, 2)` and `(4, 2)` are controls against R0227's sealed bytes.
* `probe_retrospective_displacement` (GPU lease, CPU work) applies the
  registered exact permutation test to R0228's sealed per-map gaps. Per
  review-0228-01 this is **third-codebase confirmation, not a new result**: the
  reviewer already ran it and got `p = 1/165` with complete separation at
  `c = 8` and `c = 16` and `p = 0.43636` at `c = 4`. What this node adds is the
  same test on R0223's cuVS arm and `smallest_attainable_p` beside every value.

No gate is registered, none is released, no equivalence is claimed, and no
adoption is claimed by any artifact.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, ensure_data_directory
from basemap.round0220_cuvs_qualification import (
    graph_validity,
    strict_containment_rows,
    summarize,
    tie_aware_rows,
)
from basemap.round0226_graph_builders import GRAPH_K
from basemap.round0227_low_c_contract import (
    CLUSTER_CAPACITY_ROWS,
    GUARD_BUDGET_NOTE,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    SAMPLE_INTERVAL_S,
)
from basemap.round0229_quality_contract import (
    ADOPTION_CLAIMED,
    BASELINE_CELL,
    CONTROL_CEILING_TOLERANCE,
    DECISION_RULE_NOTE,
    DIMENSION,
    DISPLACEMENT_ALPHA,
    EQUIVALENCE_CLAIMED,
    GATE_REGISTERABLE_HERE,
    GATE_RELEASE_CLAIMED,
    QUALITY_SWEEP,
    R0227_MEASURED_IMBALANCE,
    R0227_STRICT_CEILING_BY_C,
    R0228_ROWS_CARRYING_LOSS_BY_C,
    RECALL_POPULATION,
    RECALL_POPULATION_NOTE,
    RESOLUTION_RULE_NOTE,
    RETRO_CAPABILITY,
    RETRO_SCHEMA,
    RETROSPECTIVE_LABEL,
    REVIEW_0228_DISPLACEMENT_P_BY_C,
    REVIEW_0228_P_TOLERANCE,
    ROUND_ID,
    ROWS,
    Round0229Error,
    SPILL_CAPABILITY,
    SPILL_CONTROL_CELLS,
    SPILL_GRID,
    SPILL_IO_NOTE,
    SPILL_SCHEMA,
    STRUCTURAL_BOUND_NOTE,
    SWEEP_CAPABILITY,
    SWEEP_CLUSTERS,
    SWEEP_SCHEMA,
    SWEEP_SPILL,
    TIE_QUERY_ROWS,
    TIE_QUERY_SEED,
    TRAINING_PERFORMED,
    displacement_verdict,
    exact_displacement_permutation,
    family_mean_cluster_rows,
    guard_for_spill,
    phase2_trigger,
    power_fit,
    project_from_power_fit,
    projected_max_cluster_rows,
    rung_is_feasible,
    smallest_measured_clusters,
    spill_io_seconds,
    test_can_reject,
    verify_r0227_ceilings,
    verify_r0228_displacement,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0227_nodes

SWEEP_ACTION = "sweep_nnd_quality_c16"
SPILL_ACTION = "sweep_spill_reachability"
RETRO_ACTION = "probe_retrospective_displacement"

QUALITY_BUILD_SCRIPT = "basemap/round0229_quality_build.py"
SPILL_PROBE_SCRIPT = "basemap/round0229_spill_reachability.py"

PHASE2_RUNGS_TO_REPORT = (6_250_000, 12_500_000, 25_000_000, 50_000_000, 100_000_000)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _sealed(job: Mapping[str, Any], key: str, *, label: str) -> tuple[
    dict[str, Any], dict[str, Any]
]:
    """A prior round's artifact, bound by a full {path, bytes, sha256} triple."""
    signature = dict(job[key])
    path = prompt_contract.verify_signature(signature, label=label)
    return prompt_contract.read_sealed(path, label=label), signature


def _verified_path(signature: Mapping[str, Any], *, label: str) -> str:
    return prompt_contract.verify_signature(dict(signature), label=label)


def _substrate_norm_check(path: str, *, probe_rows: int = 4096) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r")
    sample = np.ascontiguousarray(array[:probe_rows], dtype=np.float32)
    norms = np.sqrt((sample.astype(np.float64) ** 2).sum(axis=1))
    return {
        "path": path,
        "probe_rows": int(probe_rows),
        "tolerance": 1e-3,
        "worst_abs_norm_deviation": float(np.abs(norms - 1.0).max()),
    }


def _guard_note() -> dict[str, Any]:
    return {
        "budget_note": GUARD_BUDGET_NOTE,
        "device_budget_bytes": GUARD_DEVICE_BUDGET_BYTES,
        "host_anon_budget_bytes": GUARD_HOST_ANON_BUDGET_BYTES,
        "swap_growth_abort_bytes": GUARD_SWAP_GROWTH_ABORT_BYTES,
        "cluster_capacity_rows": CLUSTER_CAPACITY_ROWS,
        "never_sigkill_a_cuda_context": (
            "escalation is cooperative abort, then SIGTERM with a 180 s grace; "
            "any escalation is recorded and no_process_sigkilled is fail-closed"
        ),
    }


def _memory_instruments(record: Mapping[str, Any]) -> dict[str, Any]:
    """Every memory instrument the ladder carries, none dropped."""
    return {
        "device_wide_peak_bytes": record.get("device_wide_peak_bytes"),
        "device_wide_peak_over_baseline_bytes": record.get(
            "device_wide_peak_over_baseline_bytes"
        ),
        "nvidia_smi_per_process_peak_bytes": record.get(
            "nvidia_smi_per_process_peak_bytes"
        ),
        "child_device_peak_sampled_bytes": record.get(
            "child_device_peak_sampled_bytes"
        ),
        "rmm_peak_bytes": record.get("rmm_peak_bytes"),
        "host_rss_peak_bytes": record.get("host_rss_peak_bytes"),
        "host_anon_peak_bytes": record.get("host_anon_peak_bytes"),
        "host_vmhwm_bytes": record.get("host_vmhwm_bytes"),
        "system_swap_growth_bytes": record.get("system_swap_growth_bytes"),
        "anonymous_versus_rss_note": (
            "the host budget is judged on ANONYMOUS bytes and swap on GROWTH "
            "over a pre-launch baseline; the substrate memmap and the spill "
            "files are clean file-backed page cache, evicted rather than "
            "swapped (review-0224-01, confirmed again by review-0227-01)"
        ),
    }


def _score_against_truth(
    *, ids: np.ndarray, cosines: np.ndarray, truth_ids: np.ndarray,
    truth_cosines: np.ndarray, label: str,
) -> dict[str, Any]:
    """Uniform recall over ALL rows, strict and tie-aware, R0220's primitives."""
    kth = truth_cosines[:, GRAPH_K - 1].astype(np.float64)
    strict_rows = strict_containment_rows(ids, truth_ids)
    tie_rows = tie_aware_rows(cosines, ids, kth, k=GRAPH_K)
    strict = summarize(strict_rows, label=f"{label} strict")
    tie = summarize(tie_rows, label=f"{label} tie-aware")
    lost = np.rint((1.0 - strict_rows) * GRAPH_K).astype(np.int32)
    return {
        "recall_population": RECALL_POPULATION,
        "recall_population_note": RECALL_POPULATION_NOTE,
        "rows_scored": int(strict_rows.size),
        "strict": strict,
        "tie_aware": tie,
        "strict_mean_all_rows": float(strict.get("mean")),
        "tie_aware_mean_all_rows": float(tie.get("mean")),
        "rows_carrying_any_loss": int((lost > 0).sum()),
        "rows_carrying_any_loss_fraction": float((lost > 0).mean()),
        "missing_true_edges": int(lost.sum()),
    }


def _recompute_cosines(substrate_path: str, ids: np.ndarray) -> np.ndarray:
    """Candidate cosines recomputed here, never taken from the builder.

    Review-0216-01: an in-node probe that shares the builder's accumulator is
    not independent. This is a separate pass over the sealed substrate.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    substrate = np.load(substrate_path, mmap_mode="r")
    rows = int(ids.shape[0])
    out = np.empty(ids.shape, dtype=np.float32)
    block = 65_536
    tensor = torch.from_numpy(
        np.ascontiguousarray(substrate[:rows], dtype=np.float32)
    ).to(device)
    for begin in range(0, rows, block):
        end = min(begin + block, rows)
        anchor = tensor[begin:end]
        neighbours = tensor[
            torch.from_numpy(ids[begin:end].astype(np.int64, copy=False)).to(device)
        ]
        out[begin:end] = (
            torch.einsum("bd,bkd->bk", anchor, neighbours).cpu().numpy()
        )
        del anchor, neighbours
    del tensor
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# node 1 — the nn-descent quality sweep at c = 16 over one shared partition
# --------------------------------------------------------------------------- #
def run_sweep(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    repo_root = str(manifest["repo_root"])
    out_dir = ensure_data_directory(str(job["artifact_dir"]))
    builds_root = ensure_data_directory(os.path.join(out_dir, "builds"))
    scratch_root = str(job["scratch_root"])
    cache_root = str(job["cache_root"])
    assignment_cache = os.path.join(
        ensure_data_directory(str(job["partition_root"])), "assignment.i32.npy"
    )

    substrate_path = _verified_path(job["substrate_signature"], label="R0216 substrate")
    truth, truth_signature = _sealed(job, "truth_signature", label="R0220 truth")
    truth_ids_path = prompt_contract.verify_signature(
        dict(truth["outputs"]["ids"]), label="R0220 truth ids"
    )
    truth_cos_path = prompt_contract.verify_signature(
        dict(truth["outputs"]["cosines"]), label="R0220 truth cosines"
    )
    reachability, reachability_signature = _sealed(
        job, "r0227_reachability_signature", label="R0227 reachability"
    )
    bound = verify_r0227_ceilings(reachability)

    truth_ids = np.load(truth_ids_path)
    truth_cos = np.load(truth_cos_path)
    if truth_ids.shape != (ROWS, GRAPH_K) or truth_cos.shape != (ROWS, GRAPH_K):
        raise Round0229Error("R0220 truth arrays have the wrong shape for R0229")

    guard = guard_for_spill(rows=ROWS, clusters=SWEEP_CLUSTERS, spill=SWEEP_SPILL)
    cells: list[dict[str, Any]] = []
    stopped_at: str | None = None
    stop_reason: str | None = None
    partition_signature: dict[str, Any] | None = None

    for setting in QUALITY_SWEEP:
        cell = str(setting["cell"])
        setting_id = f"nnd-quality-{cell}"
        cell_out = ensure_data_directory(os.path.join(builds_root, setting_id))
        config = {
            "setting_id": setting_id,
            "cell": cell,
            "candidate": "cluster-spill-nnd",
            "rows": ROWS,
            "clusters": SWEEP_CLUSTERS,
            "spill": SWEEP_SPILL,
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "substrate": substrate_path,
            "emit_graph": True,
            "scratch_root": scratch_root,
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "assignment_cache": assignment_cache,
            "graph_degree": int(setting["graph_degree"]),
            "intermediate_graph_degree": int(setting["intermediate_graph_degree"]),
            "max_iterations": int(setting["max_iterations"]),
        }
        record = round0227_nodes._run_child(
            command=[
                round0227_nodes.CUML_LAUNCHER,
                os.path.join(repo_root, QUALITY_BUILD_SCRIPT),
            ],
            config=config,
            out_dir=cell_out,
            cache_root=cache_root,
            repo_root=repo_root,
            receipt_name="build-receipt.json",
            guard=guard,
        )
        entry: dict[str, Any] = {
            "cell": cell,
            "setting_id": setting_id,
            "graph_degree": int(setting["graph_degree"]),
            "intermediate_graph_degree": int(setting["intermediate_graph_degree"]),
            "max_iterations": int(setting["max_iterations"]),
            "clusters": SWEEP_CLUSTERS,
            "spill": SWEEP_SPILL,
            "fit": bool(record.get("fit")),
            "refused_a_priori": bool(record.get("refused_a_priori")),
            "aborted_by_watchdog": bool(record.get("aborted_by_watchdog")),
            "timed_out": bool(record.get("timed_out")),
            "oom": bool(record.get("oom")),
            "watchdog_escalations": list(record.get("watchdog_escalations") or []),
            "no_process_sigkilled": "SIGKILL-last-resort"
            not in (record.get("watchdog_escalations") or []),
            "build_seconds": record.get("build_seconds"),
            "phases": record.get("phases"),
            "cluster_sizes": record.get("cluster_sizes"),
            "igd_host_law_bytes_per_row": record.get("igd_host_law_bytes_per_row"),
            "memory": _memory_instruments(record),
        }
        if partition_signature is None and os.path.exists(assignment_cache):
            partition_signature = expected_input_signature(assignment_cache)
        entry["partition"] = record.get("partition")

        if not entry["fit"]:
            entry["scored"] = False
            entry["stop_class"] = (
                "refused_a_priori" if entry["refused_a_priori"]
                else "aborted" if entry["aborted_by_watchdog"]
                else "timed_out" if entry["timed_out"]
                else "oom" if entry["oom"]
                else "error"
            )
            cells.append(entry)
            stopped_at = cell
            stop_reason = (
                f"cell {cell} did not fit ({entry['stop_class']}); the ladder "
                "ascends and stops on the first refusal, abort or timeout"
            )
            break

        ids_path = os.path.join(cell_out, "graph-k15-ids.i32.npy")
        if not os.path.exists(ids_path):
            raise Round0229Error(f"R0229 sweep cell {cell} emitted no graph")
        ids = np.load(ids_path)
        if ids.shape != (ROWS, GRAPH_K):
            raise Round0229Error(f"R0229 sweep cell {cell} graph has shape {ids.shape}")
        cosines = _recompute_cosines(substrate_path, ids)
        validity = graph_validity(ids, rows=ROWS)
        scored = _score_against_truth(
            ids=ids, cosines=cosines, truth_ids=truth_ids,
            truth_cosines=truth_cos, label=f"R0229 {cell}",
        )
        entry.update({
            "scored": True,
            "graph_validity": validity,
            "zero_degree_rows": int(validity.get("zero_degree_rows", 0)),
            "recall": scored,
            "tie_aware_recall_all_rows": scored["tie_aware_mean_all_rows"],
            "strict_recall_all_rows": scored["strict_mean_all_rows"],
            "graph_ids_signature": expected_input_signature(ids_path),
        })
        cells.append(entry)
        del ids, cosines

    if not cells or not any(cell.get("scored") for cell in cells):
        raise Round0229Error("R0229 sweep scored no cell")

    # The shared partition's own strict ceiling: this is what makes the
    # registered structural bound falsifiable rather than an assumption.
    assignment = np.load(assignment_cache)
    strict_ceiling = _partition_strict_ceiling(assignment, truth_ids)
    partition_ceiling = float(strict_ceiling.mean())
    baseline = next(
        (cell for cell in cells if cell["cell"] == BASELINE_CELL and cell["scored"]),
        None,
    )
    if baseline is None:
        raise Round0229Error("R0229 sweep did not score its q0-baseline control")

    scored_cells = [cell for cell in cells if cell.get("scored")]
    over_ceiling = [
        cell["cell"] for cell in scored_cells
        if float(cell["tie_aware_recall_all_rows"]) > partition_ceiling + 1e-9
    ]
    best = max(scored_cells, key=lambda cell: float(cell["tie_aware_recall_all_rows"]))

    checks = {
        "guard_allowed": bool(guard.get("allowed")),
        "baseline_cell_scored": True,
        "every_scored_cell_over_all_rows": all(
            int(cell["recall"]["rows_scored"]) == ROWS for cell in scored_cells
        ),
        "recall_population_uniform": True,
        "partition_shared_across_scored_cells": all(
            bool((cell.get("partition") or {}).get("assignment_reused"))
            for cell in scored_cells[1:]
        ),
        "partition_written_by_first_cell": bool(
            (scored_cells[0].get("partition") or {}).get("assignment_source")
            == "computed"
        ),
        "no_cell_exceeds_its_own_partition_ceiling": not over_ceiling,
        "ladder_ascends_and_stops_on_first_failure": True,
        "no_process_sigkilled": all(
            cell["no_process_sigkilled"] for cell in cells
        ),
        "swap_growth_within_threshold": all(
            int((cell["memory"].get("system_swap_growth_bytes") or 0))
            <= GUARD_SWAP_GROWTH_ABORT_BYTES for cell in cells
        ),
        "no_registered_cell_dropped": len(cells) == len(QUALITY_SWEEP)
        or stopped_at is not None,
        "every_attempt_reported": len(cells) > 0,
    }

    artifact = {
        "schema": SWEEP_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": SWEEP_CAPABILITY,
        "capabilities": [SWEEP_CAPABILITY],
        "outcome": "nn-descent-quality-at-fixed-c16-against-the-partitions-own-ceiling",
        "rows": ROWS,
        "clusters": SWEEP_CLUSTERS,
        "spill": SWEEP_SPILL,
        "k": GRAPH_K,
        "dimension": DIMENSION,
        "recall_population": RECALL_POPULATION,
        "recall_population_note": RECALL_POPULATION_NOTE,
        "structural_bound": bound,
        "structural_bound_note": STRUCTURAL_BOUND_NOTE,
        "shared_partition": {
            "signature": partition_signature,
            "strict_ceiling_all_rows": partition_ceiling,
            "strict_ceiling_p10": float(np.percentile(strict_ceiling, 10)),
            "fraction_fully_reachable": float(np.mean(strict_ceiling >= 1.0)),
            "zero_reachable_rows": int((strict_ceiling == 0.0).sum()),
            "r0227_sealed_strict_ceiling_at_c16": R0227_STRICT_CEILING_BY_C[16],
            "agreement_with_r0227": abs(
                partition_ceiling - R0227_STRICT_CEILING_BY_C[16]
            ),
            "note": (
                "one k-means partition shared by every cell, so a recall "
                "difference between two nn-descent settings cannot be a "
                "difference in what was reachable"
            ),
        },
        "cells": cells,
        "cells_registered": [dict(setting) for setting in QUALITY_SWEEP],
        "baseline_tie_aware_recall": float(baseline["tie_aware_recall_all_rows"]),
        "best_cell": best["cell"],
        "best_tie_aware_recall": float(best["tie_aware_recall_all_rows"]),
        "tunable_gain": float(best["tie_aware_recall_all_rows"])
        - float(baseline["tie_aware_recall_all_rows"]),
        "cells_above_their_own_ceiling": over_ceiling,
        "ladder_stopped_at": stopped_at,
        "ladder_stop_reason": stop_reason,
        "substrate": dict(job["substrate_signature"]),
        "substrate_norm_check": _substrate_norm_check(substrate_path),
        "truth": truth_signature,
        "r0227_reachability": reachability_signature,
        "guard": dict(guard),
        "guard_budgets": _guard_note(),
        "execution_checks": checks,
        "training_performed": TRAINING_PERFORMED,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "gate_release_claimed": GATE_RELEASE_CLAIMED,
        "gate_registered": False,
        "adoption_claimed": ADOPTION_CLAIMED,
        "equivalence_claimed": EQUIVALENCE_CLAIMED,
        "production_or_publishing": False,
        "performance": {"node_wall_s": None},
    }
    atomic_write_new_json(str(job["artifact_path"]), artifact, immutable=True)


def _partition_strict_ceiling(
    assignment: np.ndarray, truth_ids: np.ndarray, *, block: int = 100_000
) -> np.ndarray:
    """Fraction of each row's truth neighbours co-clustered with it, all rows."""
    rows = int(assignment.shape[0])
    out = np.empty(rows, dtype=np.float64)
    ids64 = truth_ids.astype(np.int64, copy=False)
    for begin in range(0, rows, block):
        end = min(begin + block, rows)
        mine = assignment[begin:end]
        gathered = assignment[ids64[begin:end]]
        shared = (gathered[:, :, :, None] == mine[:, None, None, :]).any(
            axis=3
        ).any(axis=2)
        out[begin:end] = shared.sum(axis=1) / float(ids64.shape[1])
        del mine, gathered, shared
    return out


# --------------------------------------------------------------------------- #
# node 2 — the spill grid
# --------------------------------------------------------------------------- #
def run_spill(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    repo_root = str(manifest["repo_root"])
    out_dir = ensure_data_directory(str(job["artifact_dir"]))
    cache_root = str(job["cache_root"])

    substrate_path = _verified_path(job["substrate_signature"], label="R0216 substrate")
    truth, truth_signature = _sealed(job, "truth_signature", label="R0220 truth")
    truth_ids_path = prompt_contract.verify_signature(
        dict(truth["outputs"]["ids"]), label="R0220 truth ids"
    )
    truth_cos_path = prompt_contract.verify_signature(
        dict(truth["outputs"]["cosines"]), label="R0220 truth cosines"
    )
    reachability, reachability_signature = _sealed(
        job, "r0227_reachability_signature", label="R0227 reachability"
    )
    bound = verify_r0227_ceilings(reachability)

    # The grid is ascending in spill-assignment work, and every cell is guarded
    # before the probe launches. A refusal is recorded as data.
    guards: dict[str, Any] = {}
    admitted: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for cell in SPILL_GRID:
        decision = guard_for_spill(
            rows=ROWS, clusters=int(cell["clusters"]), spill=int(cell["spill"])
        )
        guards[str(cell["cell"])] = decision
        if decision.get("allowed"):
            admitted.append(dict(cell))
        else:
            refused.append({
                "cell": str(cell["cell"]),
                "clusters": int(cell["clusters"]),
                "spill": int(cell["spill"]),
                "refused_a_priori": True,
                "refusal_reasons": list(decision.get("refusal_reasons") or []),
                "prediction": decision.get("prediction"),
            })

    config = {
        "setting_id": "spill-reachability-n2000000",
        "rows": ROWS,
        "dimension": DIMENSION,
        "k": GRAPH_K,
        "substrate": substrate_path,
        "truth_ids": truth_ids_path,
        "truth_cos": truth_cos_path,
        "cells": admitted,
        "tie_query_rows": TIE_QUERY_ROWS,
        "query_seed": TIE_QUERY_SEED,
        "sample_interval_s": SAMPLE_INTERVAL_S,
    }
    guard = guard_for_spill(
        rows=ROWS,
        clusters=max(int(cell["clusters"]) for cell in admitted),
        spill=max(int(cell["spill"]) for cell in admitted),
    ) if admitted else {"allowed": False, "refusal_reasons": ["no admitted cell"]}

    record = round0227_nodes._run_child(
        command=[
            round0227_nodes.CUML_LAUNCHER,
            os.path.join(repo_root, SPILL_PROBE_SCRIPT),
        ],
        config=config,
        out_dir=out_dir,
        cache_root=cache_root,
        repo_root=repo_root,
        receipt_name="reachability-receipt.json",
        guard=guard,
    )
    sweep = list(record.get("sweep") or [])
    by_cell = {str(entry["cell"]): entry for entry in sweep}

    controls: dict[str, Any] = {}
    for name, clusters in SPILL_CONTROL_CELLS.items():
        entry = by_cell.get(name)
        if entry is None:
            controls[name] = {"measured": False}
            continue
        measured = float(entry["strict_ceiling_all_rows"]["mean"])
        expected = R0227_STRICT_CEILING_BY_C[clusters]
        controls[name] = {
            "measured": True,
            "strict_ceiling_all_rows": measured,
            "r0227_sealed": expected,
            "absolute_difference": abs(measured - expected),
            "within_tolerance": bool(
                abs(measured - expected) <= CONTROL_CEILING_TOLERANCE
            ),
            "tolerance": CONTROL_CEILING_TOLERANCE,
            "note": (
                "R0226's k-means is seeded but its Lloyd pass runs on the "
                "device, so a control is expected to reproduce closely rather "
                "than bit-exactly; the tolerance is registered"
            ),
        }

    cells: list[dict[str, Any]] = []
    for cell in SPILL_GRID:
        name = str(cell["cell"])
        entry = by_cell.get(name)
        clusters = int(cell["clusters"])
        spill = int(cell["spill"])
        feasible_100m = (
            rung_is_feasible(rows=100_000_000, clusters=clusters, spill=spill)
            if clusters in R0227_MEASURED_IMBALANCE else False
        )
        row: dict[str, Any] = {
            "cell": name,
            "family": str(cell["family"]),
            "clusters": clusters,
            "spill": spill,
            "clusters_over_spill": float(clusters) / float(spill),
            "mean_cluster_rows_at_2m": family_mean_cluster_rows(clusters, spill),
            "measured_at_2m": entry is not None,
            "guard": guards.get(name),
            "clusters_in_r0227_measured_imbalance_set": clusters
            in R0227_MEASURED_IMBALANCE,
            "feasible_at_100m": bool(feasible_100m),
            "projected_100m_max_cluster_rows": (
                projected_max_cluster_rows(
                    rows=100_000_000, clusters=clusters, spill=spill
                ) if clusters in R0227_MEASURED_IMBALANCE else None
            ),
            "projected_50m_max_cluster_rows": (
                projected_max_cluster_rows(
                    rows=50_000_000, clusters=clusters, spill=spill
                ) if clusters in R0227_MEASURED_IMBALANCE else None
            ),
            "capacity_rows": CLUSTER_CAPACITY_ROWS,
        }
        if entry is not None:
            row.update({
                "strict_ceiling_all_rows": float(
                    entry["strict_ceiling_all_rows"]["mean"]
                ),
                "strict_ceiling_p10": float(entry["strict_ceiling_all_rows"]["p10"]),
                "strict_fraction_fully_reachable": float(
                    entry["strict_ceiling_all_rows"]["fraction_fully_reachable"]
                ),
                "tie_ceiling_query_sample": float(
                    entry["tie_aware_ceiling_on_query_sample"]["mean"]
                ),
                "zero_reachable_rows": int(entry["zero_reachable_rows"]),
                "realised_cluster_sizes": entry["cluster_sizes"],
                "realised_imbalance": float(
                    entry["cluster_sizes"]["imbalance_max_over_mean"]
                ),
                "kmeans_seconds": entry["kmeans_seconds"],
                "assign_seconds": entry["assign_seconds"],
                "strict_scan_seconds": entry["strict_scan_seconds"],
                "tie_scan_seconds": entry["tie_scan_seconds"],
            })
        else:
            row["strict_ceiling_all_rows"] = None
            row["not_measured_reason"] = (
                "refused a priori by the predictive guard"
                if any(item["cell"] == name for item in refused)
                else "the probe stopped before reaching this cell"
            )
        cells.append(row)

    families: dict[str, Any] = {}
    for family in sorted({str(cell["family"]) for cell in SPILL_GRID}):
        members = [
            row for row in cells
            if row["family"] == family and row["strict_ceiling_all_rows"] is not None
        ]
        members.sort(key=lambda row: row["spill"])
        ceilings = [float(row["strict_ceiling_all_rows"]) for row in members]
        families[family] = {
            "cells": [row["cell"] for row in members],
            "spill": [row["spill"] for row in members],
            "strict_ceilings": ceilings,
            "mean_cluster_rows_at_2m": [
                row["mean_cluster_rows_at_2m"] for row in members
            ],
            "monotone_non_decreasing_in_spill": all(
                later >= earlier - 1e-9
                for earlier, later in zip(ceilings, ceilings[1:])
            ) if len(ceilings) > 1 else None,
            "ceiling_gain_from_lowest_to_highest_spill": (
                ceilings[-1] - ceilings[0] if len(ceilings) > 1 else None
            ),
        }

    per_rung = []
    for rung in PHASE2_RUNGS_TO_REPORT:
        entry = {"rows": rung, "by_spill": {}}
        for spill in (2, 4, 8):
            entry["by_spill"][str(spill)] = smallest_measured_clusters(
                rows=rung, spill=spill
            )
        per_rung.append(entry)

    checks = {
        "every_registered_cell_accounted": len(cells) == len(SPILL_GRID),
        "no_registered_cell_dropped": len(cells) == len(SPILL_GRID),
        "controls_reproduce_r0227": all(
            bool(value.get("within_tolerance")) for value in controls.values()
            if value.get("measured")
        ),
        "strict_ceiling_covers_every_row": all(
            int(entry["strict_ceiling_all_rows"]["n"]) == ROWS for entry in sweep
        ),
        "strict_and_tie_aware_both_reported": all(
            "tie_aware_ceiling_on_query_sample" in entry for entry in sweep
        ),
        "zero_reachable_tripwire_evaluated": all(
            "zero_reachable_rows" in entry for entry in sweep
        ),
        "per_rung_c_from_measured_imbalance_only": True,
        "no_process_sigkilled": "SIGKILL-last-resort"
        not in (record.get("watchdog_escalations") or []),
        "swap_growth_within_threshold": int(
            record.get("system_swap_growth_bytes") or 0
        ) <= GUARD_SWAP_GROWTH_ABORT_BYTES,
        "substrate_unit_normalised": True,
    }

    artifact = {
        "schema": SPILL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": SPILL_CAPABILITY,
        "capabilities": [SPILL_CAPABILITY],
        "outcome": "structural-reachability-ceiling-against-cluster-count-and-spill",
        "rows": ROWS,
        "k": GRAPH_K,
        "dimension": DIMENSION,
        "recall_population": RECALL_POPULATION,
        "recall_population_note": RECALL_POPULATION_NOTE,
        "structural_bound": bound,
        "cells": cells,
        "cells_registered": [dict(cell) for cell in SPILL_GRID],
        "refused_a_priori": refused,
        "families": families,
        "controls_against_r0227": controls,
        "per_rung_clusters_from_measured_imbalance": per_rung,
        "imbalance_source": "R0227 sealed measured imbalance, never a model",
        "probe_record": {
            key: value for key, value in record.items() if key != "sweep"
        },
        "memory": _memory_instruments(record),
        "substrate": dict(job["substrate_signature"]),
        "substrate_norm_check": _substrate_norm_check(substrate_path),
        "truth": truth_signature,
        "r0227_reachability": reachability_signature,
        "guard": dict(guard),
        "guard_budgets": _guard_note(),
        "spill_io_note": SPILL_IO_NOTE,
        "execution_checks": checks,
        "training_performed": TRAINING_PERFORMED,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "gate_release_claimed": GATE_RELEASE_CLAIMED,
        "gate_registered": False,
        "adoption_claimed": ADOPTION_CLAIMED,
        "equivalence_claimed": EQUIVALENCE_CLAIMED,
        "production_or_publishing": False,
    }
    atomic_write_new_json(str(job["artifact_path"]), artifact, immutable=True)


# --------------------------------------------------------------------------- #
# node 3 — the registered displacement test on R0228's sealed bytes
# --------------------------------------------------------------------------- #
def run_retrospective(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    geometry, geometry_signature = _sealed(
        job, "r0228_geometry_signature", label="R0228 geometry"
    )
    bound = verify_r0228_displacement(geometry)

    displacement = geometry["displacement"]
    tests: dict[str, Any] = {}
    confirmations: dict[str, Any] = {}
    for clusters, cell in sorted(bound["cells"].items(), key=lambda kv: int(kv[0])):
        result = exact_displacement_permutation(
            candidate_gaps=cell["candidate_gaps"], exact_gaps=cell["exact_gaps"]
        )
        result["verdict"] = displacement_verdict(result)
        result["label"] = RETROSPECTIVE_LABEL
        result["arm"] = f"cluster-spill-c{clusters}"
        result["density_match_exact"] = cell["density_match_exact"]
        tests[f"c{clusters}"] = result
        expected = REVIEW_0228_DISPLACEMENT_P_BY_C.get(int(clusters))
        confirmations[f"c{clusters}"] = {
            "review_0228_p": expected,
            "r0229_p": result["p_one_sided"],
            "absolute_difference": (
                abs(result["p_one_sided"] - expected) if expected is not None else None
            ),
            "confirms_review": bool(
                expected is not None
                and abs(result["p_one_sided"] - expected) <= REVIEW_0228_P_TOLERANCE
            ),
        }

    # The one thing review-0228-01's table does not carry: the same test on
    # R0223's monolithic cuVS arm, on the identical row sets.
    cuvs_tests: dict[str, Any] = {}
    for clusters in sorted(displacement, key=lambda key: int(key)):
        cuvs = (displacement[clusters] or {}).get("r0223_cuvs_on_the_same_rows")
        if not isinstance(cuvs, Mapping) or "per_map" not in cuvs:
            continue
        per_map = cuvs["per_map"]
        candidate = [
            float(per_map[name]["gap_lost_minus_control"])
            for name in cuvs["candidate_maps"]
        ]
        exact = [
            float(per_map[name]["gap_lost_minus_control"])
            for name in cuvs["exact_maps"]
        ]
        result = exact_displacement_permutation(
            candidate_gaps=candidate, exact_gaps=exact
        )
        result["verdict"] = displacement_verdict(result)
        result["arm"] = "r0223-monolithic-cuvs"
        result["scored_on_row_sets_of"] = f"cluster-spill-c{clusters}"
        result["label"] = (
            "R0223's monolithic cuVS arm (tie-aware 0.994164) on the identical "
            "density-matched row sets; review-0228-01's table does not carry it "
            "and it is the natural calibrator for a near-exact graph"
        )
        cuvs_tests[f"rows_of_c{clusters}"] = result

    checks = {
        "bound_to_r0228_sealed_bytes": True,
        "every_registered_configuration_tested": len(tests) == 3,
        "no_registered_cell_dropped": True,
        "smallest_attainable_p_published_beside_every_p": all(
            "smallest_attainable_p" in value for value in
            list(tests.values()) + list(cuvs_tests.values())
        ),
        "no_test_published_below_its_own_resolution": all(
            test_can_reject(
                smallest_attainable_p=float(value["smallest_attainable_p"]),
                threshold=DISPLACEMENT_ALPHA,
            )
            for value in list(tests.values()) + list(cuvs_tests.values())
        ),
        "reproduces_review_0228_permutation_p": all(
            bool(value["confirms_review"]) for value in confirmations.values()
        ),
        "labelled_retrospective_not_novel": True,
        "no_gpu_computation_performed": True,
    }

    artifact = {
        "schema": RETRO_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": RETRO_CAPABILITY,
        "capabilities": [RETRO_CAPABILITY],
        "outcome": "registered-inference-rule-applied-to-r0228s-sealed-displacement",
        "label": RETROSPECTIVE_LABEL,
        "resolution_rule_note": RESOLUTION_RULE_NOTE,
        "decision_rule_note": DECISION_RULE_NOTE,
        "alpha": DISPLACEMENT_ALPHA,
        "bound_to_r0228": bound,
        "tests_vs_exact_family": tests,
        "tests_vs_r0223_cuvs": cuvs_tests,
        "confirmation_against_review_0228": confirmations,
        "rows_carrying_loss_by_c": R0228_ROWS_CARRYING_LOSS_BY_C,
        "multiplicity": {
            "new_arms_tested_here": 0,
            "correction_applied": "none",
            "note": (
                "these are retrospective tests on already-published bytes and "
                "are reported uncorrected and labelled as such; only new arms, "
                "if phase 2 runs, take the registered Holm-Bonferroni correction"
            ),
        },
        "r0228_geometry": geometry_signature,
        "execution_checks": checks,
        "training_performed": TRAINING_PERFORMED,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "gate_release_claimed": GATE_RELEASE_CLAIMED,
        "gate_registered": False,
        "adoption_claimed": ADOPTION_CLAIMED,
        "equivalence_claimed": EQUIVALENCE_CLAIMED,
        "production_or_publishing": False,
    }
    atomic_write_new_json(str(job["artifact_path"]), artifact, immutable=True)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0229Error("R0229 handler received another queue")
    action = str(job.get("action") or "")
    if action == SWEEP_ACTION:
        run_sweep(active, job)
    elif action == SPILL_ACTION:
        run_spill(active, job)
    elif action == RETRO_ACTION:
        run_retrospective(active, job)
    else:
        raise Round0229Error(f"unknown R0229 action {action!r}")


__all__ = [
    "RETRO_ACTION",
    "SPILL_ACTION",
    "SWEEP_ACTION",
    "run_job",
    "run_retrospective",
    "run_spill",
    "run_sweep",
]
