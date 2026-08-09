"""Execute R0232 — measure the cluster-spill scratch law and bound it to zero.

Six actions, one queue:

* `measure_scratch_law` (GPU) runs eleven cells on R0216's sealed 2M substrate
  across three spill designs and three scratch bounds, with a **filesystem
  sampler** watching the disk while each build runs. Nine of them are scored over
  **all 2,000,000 rows** against R0220's sealed exact truth.
* `calibrate_larger_n` (GPU) runs the same materialise/stream pair at
  `N = 8,000,000` on R0224's sealed benchmark substrate, cold, so the round has a
  multi-group scratch measurement, a `gd 64 / igd 256` device-law calibration
  point at a `~1.7M`-row largest cluster, and this box's own measured `/data`
  read and write throughput.
* `fuzzy_streamed_arm` (GPU) scores the streamed arm graph over all 2,000,000
  rows, applies R0215's degree-zero tripwire, and symmetrises through R0216's
  identical fuzzy law.
* `train_streamed_seed{42,43,44}` (GPU) is R0217's treatment with the graph
  swapped, constructed by `basemap/round0229_train_config.train_config` imported
  read-only so the treatment-invariant digest must equal the cross-round
  constant.
* `probe_streamed_geometry` (GPU lease, CPU work) runs R0228's registered
  displacement statistic, imported read-only from
  `basemap/round0228_geometry.py`, against the same 8-map exact null arm.
* `project_scratch_and_cost` (CPU) emits the 50M/100M projection **from a node**,
  hash-bound, with the device law refitted at `gd 64 / igd 256`.

No gate is registered, none is released, no equivalence is claimed, and no
adoption is claimed by any artifact.
"""
from __future__ import annotations

import gc

import json
import os
import random
import resource
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0217_minilm_2m_seed_family import WARMUP_SUCCESSFUL_UPDATES
from basemap.round0220_cuvs_qualification import (
    TIE_TOLERANCE,
    graph_validity,
    strict_containment_rows,
    summarize,
    tie_aware_rows,
)
from basemap.round0226_graph_builders import GRAPH_K
from basemap.round0227_low_c_contract import (
    CLUSTER_CAPACITY_ROWS,
    DATA_COLD_READ_BYTES_PER_S,
    GUARD_BUDGET_NOTE,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    SAMPLE_INTERVAL_S,
    SCRATCH_BUDGET_BYTES,
)
from basemap.round0228_geometry import (
    CLUMP_DEFINITION,
    NULL_ARM_NOTE,
    SCALE_DEFINITION,
    SCATTER_DEFINITION,
    SCATTER_SAMPLE_ROWS,
    SCATTER_SAMPLE_SEED,
    clump_profile,
    density_matched_control,
    displacement_summary,
    map_scale,
    true_neighbour_scatter,
)
from basemap.round0228_low_c_map import (
    BATCH_SIZE,
    FULL_TRANSFORM_BATCH,
    FUZZY_LAW,
    FUZZY_RANDOM_STATE_SEED,
    HOST_RSS_LIMIT_GIB,
    MIN_ADMISSIBLE_NEGATIVE_DISTANCE,
    POSITIVE_ROWS_PER_UPDATE,
    TEMPLATE_SEED,
    performance_windows,
    successful_updates_for_edges,
    validate_dose,
    validate_full_population_map,
)
from basemap.round0229_phase2_contract import (
    TREATMENT_INVARIANT_SHA256,
    per_map_did,
)
from basemap.round0229_quality_contract import (
    R0228_ROWS_CARRYING_LOSS_BY_C,
    displacement_verdict,
    exact_did_trend,
    exact_displacement_permutation,
    power_fit,
    project_from_power_fit,
    smallest_measured_clusters,
    test_can_reject,
    verify_r0228_displacement,
)
from basemap.round0229_train_config import train_config
from basemap.round0232_scratch_contract import (
    ADOPTION_CLAIMED,
    ARM_CELL,
    ARM_NAME,
    ARM_REFERENCE_CELL,
    ARM_STRICT_FLOOR,
    ARM_TIE_AWARE_FLOOR,
    DESIGN_NOTE,
    DIMENSION,
    DISK_FREE_RESERVE_BYTES,
    DISK_GUARD_NOTE,
    DISPLACEMENT_ALPHA,
    EQUIVALENCE_CLAIMED,
    GATE_REGISTERABLE_HERE,
    GATE_RELEASE_CLAIMED,
    GEOMETRY_CAPABILITY,
    GEOMETRY_SCHEMA,
    GRAPH_CAPABILITY,
    GRAPH_SCHEMA,
    GRID_A,
    GRID_B,
    GRID_CAPABILITY,
    GRID_SCHEMA,
    IDENTITY_FAMILIES,
    LARGER_N_CAPABILITY,
    LARGER_N_SCHEMA,
    MINIMUM_DETECTABLE_DISPLACEMENT_SD,
    MODE_MATERIALISE,
    NON_REJECTION_NOTE,
    PERMUTATION_LABELLINGS,
    PERMUTATION_RESOLUTION_CEILING,
    PRODUCTION_CONFIG_SCHEMA,
    PROJECTION_CAPABILITY,
    PROJECTION_SCHEMA,
    R0229_ARM_MODELLED_PEAK_SCRATCH_BYTES,
    R0229_ARM_SPILL_GROUPS,
    R0229_ARM_STRICT,
    R0229_ARM_TIE_AWARE,
    RECALL_POPULATION,
    RECALL_POPULATION_NOTE,
    ROUND_ID,
    ROUND_SCRATCH_BUDGET_BYTES,
    ROWS,
    Round0232Error,
    SEEDS,
    SPILL_VOLUME_100M_S8_BYTES,
    TRAINING_PERFORMED,
    TRAIN_SCHEMA,
    capacity_rows_at_device_budget,
    cell_guard,
    data_free_bytes,
    device_law_prediction,
    io_projection,
    ladder_disk_requirement,
    licensed_statement,
    linear_fit,
    map_capability,
    scratch_law,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0113_nodes as prompt_nodes
from experiments import round0221_nodes, round0227_nodes

GRID_ACTION = "measure_scratch_law"
LARGER_N_ACTION = "calibrate_larger_n"
FUZZY_ACTION = "fuzzy_streamed_arm"
TRAIN_ACTION = "train_streamed"
GEOMETRY_ACTION = "probe_streamed_geometry"
PROJECT_ACTION = "project_scratch_and_cost"

BUILD_SCRIPT = "basemap/round0232_streamed_build.py"

PHASE2_RUNGS_TO_PROJECT = (25_000_000, 50_000_000, 100_000_000)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _sealed(job: Mapping[str, Any], key: str, *, label: str) -> tuple[
    dict[str, Any], dict[str, Any]
]:
    signature = dict(job[key])
    path = prompt_contract.verify_signature(signature, label=label)
    return prompt_contract.read_sealed(path, label=label), signature


def _verified_path(signature: Mapping[str, Any], *, label: str) -> str:
    return prompt_contract.verify_signature(dict(signature), label=label)


def _intra_queue_signature(
    reference: Mapping[str, Any], *, label: str
) -> tuple[str, dict[str, Any]]:
    """Resolve a reference to an artifact produced earlier in THIS queue.

    R0228's geometry node died on `verify_signature` of an intra-queue reference
    that carries a path and no hash at prepare time. This is R0228's own fix,
    re-used unchanged for every reference this queue produces itself.
    """
    reference = dict(reference)
    if reference.get("sha256"):
        return prompt_contract.verify_signature(reference, label=label), reference
    path = str(reference["canonical_path"])
    if not os.path.exists(path):
        raise Round0232Error(f"{label} is absent at {path}")
    return path, expected_input_signature(path)


def _guard_note() -> dict[str, Any]:
    return {
        "budget_note": GUARD_BUDGET_NOTE,
        "disk_guard_note": DISK_GUARD_NOTE,
        "device_budget_bytes": GUARD_DEVICE_BUDGET_BYTES,
        "host_anon_budget_bytes": GUARD_HOST_ANON_BUDGET_BYTES,
        "swap_growth_abort_bytes": GUARD_SWAP_GROWTH_ABORT_BYTES,
        "round_scratch_budget_bytes": ROUND_SCRATCH_BUDGET_BYTES,
        "disk_free_reserve_bytes": DISK_FREE_RESERVE_BYTES,
        "cluster_capacity_rows": CLUSTER_CAPACITY_ROWS,
        "never_sigkill_a_cuda_context": (
            "escalation is cooperative abort, then SIGTERM with a 180 s grace; "
            "any escalation is recorded and no_process_sigkilled is fail-closed"
        ),
    }


def _memory_instruments(record: Mapping[str, Any]) -> dict[str, Any]:
    """Every instrument the ladder carries, none dropped (review-0229-01 defect 2)."""
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
            "the host budget is judged on ANONYMOUS bytes and swap on GROWTH over "
            "a pre-launch baseline; the substrate memmap and the spill files are "
            "clean file-backed page cache, evicted rather than swapped "
            "(review-0224-01, confirmed by review-0227-01). RMM is published but "
            "is NOT evidence of device cost for a cuVS build: review-0229-01 §6 "
            "found it byte-identical across a change that moved device-wide "
            "occupancy by 2.9 GiB."
        ),
    }


def _scratch_instruments(record: Mapping[str, Any]) -> dict[str, Any]:
    """The disk, measured. No prior round published any of this."""
    measured = record.get("measured_peak_scratch_bytes")
    modelled = record.get("modelled_peak_scratch_bytes")
    delta = (
        int(measured) - int(modelled)
        if measured is not None and modelled is not None else None
    )
    return {
        "measured_peak_scratch_bytes": measured,
        "modelled_peak_scratch_bytes": modelled,
        "measured_minus_modelled_bytes": delta,
        "scratch_samples_taken": record.get("scratch_samples_taken"),
        "scratch_breached": record.get("scratch_breached"),
        "scratch_abort_above_bytes": record.get("scratch_abort_above_bytes"),
        "spill_groups": record.get("spill_groups"),
        "substrate_passes": record.get("substrate_passes"),
        "substrate_read_bytes": record.get("substrate_read_bytes"),
        "spill_write_bytes": record.get("spill_write_bytes"),
        "gathered_row_bytes": record.get("gathered_row_bytes"),
        "spill_volume_bytes": record.get("spill_volume_bytes"),
        "largest_cluster_bytes": record.get("largest_cluster_bytes"),
        "bound_exceeded_by_single_cluster": record.get(
            "bound_exceeded_by_single_cluster"
        ),
        "proc_io_delta": record.get("proc_io_delta"),
        "measurement_note": (
            "measured_peak_scratch_bytes is the peak of st_blocks x 512 summed "
            "over the cell's scratch directory, sampled every 50 ms while the "
            "build ran. modelled_peak_scratch_bytes is the quantity every prior "
            "round published under the name `peak_scratch_bytes`, computed from "
            "the cluster sizes before a byte was written."
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

    Review-0216-01: an in-node probe that shares the builder's accumulator is not
    independent. This is a separate pass over the sealed substrate.
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


def _run_grid(
    *, cells: Sequence[Mapping[str, Any]], job: Mapping[str, Any],
    repo_root: str, builds_root: str, substrate_by_rows: Mapping[int, str],
    partition_root: str, extra_config: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Guard, launch and record every registered cell. A refusal is a measurement."""
    scratch_root = str(job["scratch_root"])
    cache_root = str(job["cache_root"])
    records: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for cell in cells:
        name = str(cell["cell"])
        rows = int(cell["rows"])
        free_before = data_free_bytes()
        guard = cell_guard(cell, free_bytes=free_before)
        setting_id = f"scratch-{name}"
        cell_out = ensure_data_directory(os.path.join(builds_root, setting_id))
        assignment_cache = os.path.join(
            ensure_data_directory(partition_root),
            f"assignment-n{rows}-c{int(cell['clusters'])}-s{int(cell['spill'])}.i32.npy",
        )
        config: dict[str, Any] = {
            "setting_id": setting_id,
            "cell": name,
            "candidate": "cluster-spill-nnd",
            "rows": rows,
            "clusters": int(cell["clusters"]),
            "spill": int(cell["spill"]),
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "substrate": substrate_by_rows[rows],
            "emit_graph": bool(cell["scored_against_exact_truth"])
            or name == ARM_CELL,
            "scratch_root": scratch_root,
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "assignment_cache": assignment_cache,
            "graph_degree": int(cell["graph_degree"]),
            "intermediate_graph_degree": int(cell["intermediate_graph_degree"]),
            "max_iterations": int(cell["max_iterations"]),
            "mode": str(cell["mode"]),
            "bound_bytes": int(cell["bound_bytes"]),
        }
        if extra_config and name in extra_config:
            config.update(dict(extra_config[name]))
        record = round0227_nodes._run_child(
            command=[
                round0227_nodes.CUML_LAUNCHER, os.path.join(repo_root, BUILD_SCRIPT),
            ],
            config=config,
            out_dir=cell_out,
            cache_root=cache_root,
            repo_root=repo_root,
            receipt_name="build-receipt.json",
            guard={"allowed": bool(guard["allowed"]),
                   "refusal_reasons": list(guard["refusal_reasons"]),
                   **guard},
        )
        records[name] = record
        escalations = list(record.get("watchdog_escalations") or [])
        entry: dict[str, Any] = {
            "cell": name,
            "setting_id": setting_id,
            "rows": rows,
            "clusters": int(cell["clusters"]),
            "spill": int(cell["spill"]),
            "mode": str(cell["mode"]),
            "bound_bytes": int(cell["bound_bytes"]),
            "graph_degree": int(cell["graph_degree"]),
            "intermediate_graph_degree": int(cell["intermediate_graph_degree"]),
            "max_iterations": int(cell["max_iterations"]),
            "registered_note": str(cell["note"]),
            "fit": bool(record.get("fit")),
            "refused_a_priori": bool(record.get("refused_a_priori")),
            "aborted_by_watchdog": bool(record.get("aborted_by_watchdog")),
            "timed_out": bool(record.get("timed_out")),
            "oom": bool(record.get("oom")),
            "watchdog_escalations": escalations,
            "no_process_sigkilled": "SIGKILL-last-resort" not in escalations,
            "guard": guard,
            "data_free_bytes_before_cell": free_before,
            "data_free_bytes_after_cell": data_free_bytes(),
            "build_seconds": record.get("build_seconds"),
            "subprocess_seconds": record.get("subprocess_seconds"),
            "phases": record.get("phases"),
            "cluster_sizes": record.get("cluster_sizes"),
            "graph_ids_sha256": record.get("graph_ids_sha256"),
            "graph_cos_sha256": record.get("graph_cos_sha256"),
            "zero_degree_rows": record.get("zero_degree_rows"),
            "rows_below_k": record.get("rows_below_k"),
            "min_degree": record.get("min_degree"),
            "memory": _memory_instruments(record),
            "scratch": _scratch_instruments(record),
            "data_throughput": record.get("data_throughput"),
            "substrate_pages_evicted": record.get("substrate_pages_evicted"),
            "graph_emitted": bool(record.get("graph_emitted")),
            "stop_class": (
                None if record.get("fit")
                else "refused_a_priori" if record.get("refused_a_priori")
                else "aborted" if record.get("aborted_by_watchdog")
                else "timed_out" if record.get("timed_out")
                else "oom" if record.get("oom")
                else str(record.get("error_type") or "error")
            ),
            "error": record.get("error"),
        }
        entries.append(entry)
    return entries, records


def _identity_report(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Is the restructured graph the same graph? Answered on bytes, per family."""
    by_cell = {str(entry["cell"]): entry for entry in entries}
    families: list[dict[str, Any]] = []
    for family in IDENTITY_FAMILIES:
        members = [
            by_cell[name] for name in family
            if name in by_cell and by_cell[name].get("fit")
        ]
        if len(members) < 2:
            continue
        reference = members[0]
        rows = []
        for entry in members:
            rows.append({
                "cell": str(entry["cell"]),
                "mode": str(entry["mode"]),
                "bound_bytes": int(entry["bound_bytes"]),
                "graph_ids_sha256": entry.get("graph_ids_sha256"),
                "graph_cos_sha256": entry.get("graph_cos_sha256"),
                "ids_identical_to_reference": bool(
                    entry.get("graph_ids_sha256")
                    == reference.get("graph_ids_sha256")
                ),
                "cos_identical_to_reference": bool(
                    entry.get("graph_cos_sha256")
                    == reference.get("graph_cos_sha256")
                ),
                "tie_aware_recall_all_rows": entry.get("tie_aware_recall_all_rows"),
                "strict_recall_all_rows": entry.get("strict_recall_all_rows"),
            })
        families.append({
            "family": list(family),
            "reference_cell": str(reference["cell"]),
            "rows": int(reference["rows"]),
            "clusters": int(reference["clusters"]),
            "spill": int(reference["spill"]),
            "members": rows,
            "all_ids_identical": all(row["ids_identical_to_reference"] for row in rows),
            "all_cos_identical": all(row["cos_identical_to_reference"] for row in rows),
        })
    return {
        "families": families,
        "all_families_byte_identical": all(
            family["all_ids_identical"] and family["all_cos_identical"]
            for family in families
        ) if families else None,
        "note": (
            "byte-identity is asserted only within a matched (rows, clusters, "
            "spill) family, because a different partition is a different "
            "reachable set. Within a family the partition is bound to cached "
            "bytes, so any difference is the restructure and nothing else. Two "
            "materialising cells at different bounds are included in each family "
            "as the control for nn-descent non-determinism: if THEY differ, the "
            "builder is non-deterministic and no mode comparison can be "
            "byte-exact."
        ),
    }


# --------------------------------------------------------------------------- #
# node 1 — the 2M scratch-law grid
# --------------------------------------------------------------------------- #
def run_grid(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    repo_root = str(manifest["repo_root"])
    out_dir = ensure_data_directory(str(job["artifact_dir"]))
    builds_root = ensure_data_directory(os.path.join(out_dir, "builds"))

    substrate_path = _verified_path(job["substrate_signature"], label="R0216 substrate")
    truth, truth_signature = _sealed(job, "truth_signature", label="R0220 truth")
    truth_ids_path = _verified_path(truth["outputs"]["ids"], label="R0220 truth ids")
    truth_cos_path = _verified_path(
        truth["outputs"]["cosines"], label="R0220 truth cosines"
    )
    truth_ids = np.load(truth_ids_path)
    truth_cos = np.load(truth_cos_path)
    if truth_ids.shape != (ROWS, GRAPH_K) or truth_cos.shape != (ROWS, GRAPH_K):
        raise Round0232Error("R0220 truth arrays have the wrong shape for R0232")

    substrate_by_rows = {
        int(rows): substrate_path for rows in {int(cell["rows"]) for cell in GRID_A}
    }
    entries, _records = _run_grid(
        cells=GRID_A, job=job, repo_root=repo_root, builds_root=builds_root,
        substrate_by_rows=substrate_by_rows,
        partition_root=str(job["partition_root"]),
    )

    for entry in entries:
        cell = next(item for item in GRID_A if str(item["cell"]) == entry["cell"])
        if not entry["fit"] or not bool(cell["scored_against_exact_truth"]):
            entry["scored"] = False
            continue
        ids_path = os.path.join(
            builds_root, str(entry["setting_id"]), "graph-k15-ids.i32.npy"
        )
        if not os.path.exists(ids_path):
            raise Round0232Error(f"R0232 cell {entry['cell']} emitted no graph")
        ids = np.load(ids_path)
        if ids.shape != (ROWS, GRAPH_K):
            raise Round0232Error(f"R0232 cell {entry['cell']} graph is {ids.shape}")
        cosines = _recompute_cosines(substrate_path, ids)
        validity = graph_validity(ids, rows=ROWS)
        scored = _score_against_truth(
            ids=ids, cosines=cosines, truth_ids=truth_ids,
            truth_cosines=truth_cos, label=f"R0232 {entry['cell']}",
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
        del ids, cosines

    scored_cells = [entry for entry in entries if entry.get("scored")]
    if not scored_cells:
        raise Round0232Error("R0232 grid scored no cell")
    arm = next((entry for entry in entries if entry["cell"] == ARM_CELL), None)
    reference = next(
        (entry for entry in entries if entry["cell"] == ARM_REFERENCE_CELL), None
    )
    # The arm is required only when this queue actually contains the map nodes
    # that consume it. Addendum 1 withdrew the streamed modes on machine-safety
    # grounds, so the correction queue declares `arm_required: false` and the
    # round reports the displacement probe as NOT RUN rather than inferring it.
    arm_required = bool(job.get("arm_required", True))
    if arm_required and (arm is None or not arm.get("fit")):
        raise Round0232Error("R0232 arm cell did not fit; the map arm cannot run")

    identity = _identity_report(entries)
    law = scratch_law([
        {**entry["scratch"], **{
            "cell": entry["cell"], "rows": entry["rows"],
            "clusters": entry["clusters"], "spill": entry["spill"],
            "mode": entry["mode"], "bound_bytes": entry["bound_bytes"],
        }}
        for entry in entries if entry.get("fit")
    ])

    streamed = [entry for entry in entries if entry["mode"] != MODE_MATERIALISE
                and entry.get("fit")]
    materialising = [entry for entry in entries if entry["mode"] == MODE_MATERIALISE
                     and entry.get("fit")]
    checks = {
        "every_registered_cell_accounted": len(entries) == len(GRID_A),
        "no_registered_cell_dropped": len(entries) == len(GRID_A),
        "every_scored_cell_over_all_rows": all(
            int(entry["recall"]["rows_scored"]) == ROWS for entry in scored_cells
        ),
        "recall_population_uniform": True,
        "streamed_cells_wrote_nothing": all(
            int(entry["scratch"]["measured_peak_scratch_bytes"] or 0) == 0
            for entry in streamed
        ),
        "materialising_cells_within_their_bound": all(
            int(entry["scratch"]["measured_peak_scratch_bytes"] or 0)
            <= int(entry["bound_bytes"])
            + int(entry["scratch"]["largest_cluster_bytes"] or 0)
            for entry in materialising
        ),
        "no_scratch_breach": all(
            not bool(entry["scratch"].get("scratch_breached")) for entry in entries
        ),
        "no_process_sigkilled": all(
            entry["no_process_sigkilled"] for entry in entries
        ),
        "swap_growth_within_threshold": all(
            int((entry["memory"].get("system_swap_growth_bytes") or 0))
            <= GUARD_SWAP_GROWTH_ABORT_BYTES for entry in entries
        ),
        "arm_clears_its_registered_floors": (
            bool(
                float(arm["tie_aware_recall_all_rows"]) >= ARM_TIE_AWARE_FLOOR
                and float(arm["strict_recall_all_rows"]) >= ARM_STRICT_FLOOR
            )
            if (arm is not None and arm.get("scored")) else None
        ),
        "arm_has_zero_degree_zero_rows": (
            int(arm.get("zero_degree_rows") or -1) == 0
            if (arm is not None and arm.get("scored")) else None
        ),
        "arm_not_run_reason": (
            None if (arm is not None and arm.get("scored"))
            else "streamed modes withdrawn by round-0232 addendum 1 "
                 "(machine-safety); P2, P3 and P4 are UNRESOLVED, not failed"
        ),
        "data_free_never_below_reserve": all(
            int(entry["data_free_bytes_after_cell"]) >= DISK_FREE_RESERVE_BYTES
            for entry in entries
        ),
    }

    artifact = {
        "schema": GRID_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": GRID_CAPABILITY,
        "capabilities": [GRID_CAPABILITY],
        "outcome": "measured-peak-scratch-against-n-c-s-and-bound-in-three-spill-designs",
        "rows": ROWS,
        "k": GRAPH_K,
        "dimension": DIMENSION,
        "design_note": DESIGN_NOTE,
        "recall_population": RECALL_POPULATION,
        "recall_population_note": RECALL_POPULATION_NOTE,
        "cells": entries,
        "cells_registered": [dict(cell) for cell in GRID_A],
        "byte_identity": identity,
        "measured_scratch_law": law,
        "arm_cell": ARM_CELL,
        "arm_reference_cell": ARM_REFERENCE_CELL,
        "arm_vs_r0229": {
            "r0229_tie_aware": R0229_ARM_TIE_AWARE,
            "r0229_strict": R0229_ARM_STRICT,
            "r0229_modelled_peak_scratch_bytes": R0229_ARM_MODELLED_PEAK_SCRATCH_BYTES,
            "r0229_spill_groups": R0229_ARM_SPILL_GROUPS,
            "r0232_arm_tie_aware": arm.get("tie_aware_recall_all_rows"),
            "r0232_arm_strict": arm.get("strict_recall_all_rows"),
            "r0232_reference_measured_peak_scratch_bytes": (
                reference["scratch"]["measured_peak_scratch_bytes"]
                if reference is not None else None
            ),
            "r0232_reference_modelled_peak_scratch_bytes": (
                reference["scratch"]["modelled_peak_scratch_bytes"]
                if reference is not None else None
            ),
            "note": (
                "a1 reproduces R0229's arm cell and a5 is its streamed twin; the "
                "comparison is against R0229's SEALED build receipt and graph "
                "manifest, not against its prose"
            ),
        },
        "spill_volume_at_100m_s8_bytes": SPILL_VOLUME_100M_S8_BYTES,
        "spill_volume_note": (
            "1.2288e12 B is the total spill VOLUME at 100M s = 8. It is the "
            "number that reached this round as if it were a disk requirement. "
            "Peak scratch at any instant is what a disk budget is about and it is "
            "measured in the table above."
        ),
        "substrate": dict(job["substrate_signature"]),
        "truth": truth_signature,
        "guard_budgets": _guard_note(),
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
# node 2 — the larger-N calibration
# --------------------------------------------------------------------------- #
def run_larger_n(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    repo_root = str(manifest["repo_root"])
    out_dir = ensure_data_directory(str(job["artifact_dir"]))
    builds_root = ensure_data_directory(os.path.join(out_dir, "builds"))
    substrate_path = _verified_path(
        job["benchmark_substrate_signature"], label="R0224 16M benchmark substrate"
    )
    substrate_by_rows = {
        int(cell["rows"]): substrate_path for cell in GRID_B
    }
    entries, _records = _run_grid(
        cells=GRID_B, job=job, repo_root=repo_root, builds_root=builds_root,
        substrate_by_rows=substrate_by_rows,
        partition_root=str(job["partition_root"]),
        extra_config={
            "b1": {
                "measure_data_throughput": True,
                "fadvise_dontneed_substrate": True,
                "throughput_read_bytes": 4 * 1024 ** 3,
                "throughput_write_bytes": 2 * 1024 ** 3,
            },
            "b2": {"fadvise_dontneed_substrate": True},
        },
    )
    identity = _identity_report(entries)
    law = scratch_law([
        {**entry["scratch"], **{
            "cell": entry["cell"], "rows": entry["rows"],
            "clusters": entry["clusters"], "spill": entry["spill"],
            "mode": entry["mode"], "bound_bytes": entry["bound_bytes"],
        }}
        for entry in entries if entry.get("fit")
    ])
    throughput = next(
        (entry["data_throughput"] for entry in entries if entry.get("data_throughput")),
        None,
    )
    checks = {
        "every_registered_cell_accounted": len(entries) == len(GRID_B),
        "no_registered_cell_dropped": len(entries) == len(GRID_B),
        "no_recall_claimed_without_exact_truth": True,
        "streamed_cell_wrote_nothing": all(
            int(entry["scratch"]["measured_peak_scratch_bytes"] or 0) == 0
            for entry in entries
            if entry["mode"] != MODE_MATERIALISE and entry.get("fit")
        ),
        "no_scratch_breach": all(
            not bool(entry["scratch"].get("scratch_breached")) for entry in entries
        ),
        "no_process_sigkilled": all(
            entry["no_process_sigkilled"] for entry in entries
        ),
        "swap_growth_within_threshold": all(
            int((entry["memory"].get("system_swap_growth_bytes") or 0))
            <= GUARD_SWAP_GROWTH_ABORT_BYTES for entry in entries
        ),
        "data_free_never_below_reserve": all(
            int(entry["data_free_bytes_after_cell"]) >= DISK_FREE_RESERVE_BYTES
            for entry in entries
        ),
        "throughput_measured_in_this_round": throughput is not None,
    }
    artifact = {
        "schema": LARGER_N_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": LARGER_N_CAPABILITY,
        "capabilities": [LARGER_N_CAPABILITY],
        "outcome": "multi-group-scratch-device-calibration-and-measured-data-throughput",
        "cells": entries,
        "cells_registered": [dict(cell) for cell in GRID_B],
        "byte_identity": identity,
        "measured_scratch_law": law,
        "data_throughput": throughput,
        "data_throughput_carried_prose": {
            "review_0226_cold_read_bytes_per_s": DATA_COLD_READ_BYTES_PER_S,
            "note": (
                "review-0226-01's 5.53 GB/s is carried in the codebase as a "
                "constant; this round measures the rate itself and the projection "
                "uses the measurement"
            ),
        },
        "no_exact_truth_at_8m": (
            "recall is not scored here. There is no exact k15 truth above 2M and "
            "this round does not invent one; these cells exist for the scratch "
            "law, the device law and the I/O measurement"
        ),
        "benchmark_substrate": dict(job["benchmark_substrate_signature"]),
        "guard_budgets": _guard_note(),
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
# node 3 — the streamed arm's fuzzy graph
# --------------------------------------------------------------------------- #
def run_fuzzy(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest = active["manifest"]
    grid_path, grid_signature = _intra_queue_signature(
        dict(job["grid_reference"]), label="R0232 scratch-law grid"
    )
    with open(grid_path, encoding="utf-8") as handle:
        grid = json.load(handle)
    arm = next(
        (cell for cell in grid["cells"] if str(cell["cell"]) == ARM_CELL), None
    )
    if arm is None or not arm.get("fit"):
        raise Round0232Error("R0232 fuzzy node found no fitted arm cell")
    ids_path, ids_signature = _intra_queue_signature(
        dict(arm["graph_ids_signature"]), label="R0232 streamed neighbour ids"
    )

    substrate_path = _verified_path(job["substrate_signature"], label="R0216 substrate")
    truth, truth_signature = _sealed(job, "truth_signature", label="R0220 truth")
    truth_ids_path = _verified_path(truth["outputs"]["ids"], label="R0220 truth ids")
    truth_cos_path = _verified_path(
        truth["outputs"]["cosines"], label="R0220 truth cosines"
    )

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0232 streamed fuzzy graph"
    )
    started = time.monotonic()

    raw = np.load(ids_path, allow_pickle=False)
    if raw.shape != (ROWS, GRAPH_K):
        raise Round0232Error(f"R0232 neighbour ids are {raw.shape}")
    leading = np.ascontiguousarray(raw.astype(np.int32))
    del raw
    as_int = leading.astype(np.int64)
    if int(as_int.min()) < 0 or int(as_int.max()) >= ROWS:
        raise Round0232Error("R0232 graph carries out-of-range neighbour ids")
    del as_int

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    host = np.load(substrate_path, mmap_mode="r", allow_pickle=False)
    if host.shape != (ROWS, DIMENSION) or host.dtype != np.float32:
        raise Round0232Error("R0232 sealed substrate geometry changed")
    tensor = torch.from_numpy(
        np.array(host, dtype=np.float32, order="C", copy=True)
    ).to(device)

    cosine_started = time.monotonic()
    block = 65_536
    candidate_cos = np.empty(leading.shape, dtype=np.float32)
    for begin in range(0, ROWS, block):
        end = min(begin + block, ROWS)
        anchor = tensor[begin:end]
        neighbours = tensor[
            torch.from_numpy(leading[begin:end].astype(np.int64)).to(device)
        ]
        candidate_cos[begin:end] = (
            torch.einsum("bd,bkd->bk", anchor, neighbours).cpu().numpy()
        )
        del anchor, neighbours
    cosine_s = time.monotonic() - cosine_started
    del tensor
    torch.cuda.empty_cache()
    gc.collect()

    truth_ids = np.load(truth_ids_path, allow_pickle=False)
    truth_cos = np.load(truth_cos_path, allow_pickle=False)
    if truth_ids.shape != (ROWS, GRAPH_K) or truth_cos.shape != (ROWS, GRAPH_K):
        raise Round0232Error("R0220 truth arrays have the wrong shape")
    kth = truth_cos[:, GRAPH_K - 1].astype(np.float64)

    strict = strict_containment_rows(leading, truth_ids)
    tie = tie_aware_rows(candidate_cos.astype(np.float64), leading, kth)
    strict_summary = summarize(strict, label="R0232 streamed strict recall@15")
    tie_summary = summarize(tie, label="R0232 streamed tie-aware recall@15")
    measured_strict = float(strict_summary["mean"])
    measured_tie = float(tie_summary["mean"])
    if measured_tie < ARM_TIE_AWARE_FLOOR or measured_strict < ARM_STRICT_FLOOR:
        raise Round0232Error(
            f"R0232 streamed arm recall {measured_tie} / {measured_strict} is "
            f"below its registered floors {ARM_TIE_AWARE_FLOOR} / {ARM_STRICT_FLOOR}"
        )

    structural = graph_validity(leading, rows=ROWS)
    lost_edges_per_row = np.rint((1.0 - strict) * GRAPH_K).astype(np.int16)
    rows_carrying_loss = int((lost_edges_per_row > 0).sum())
    strict_path = atomic_save_new_npy(
        os.path.join(output, "strict-recall-per-row.f32.npy"),
        strict.astype(np.float32), immutable=True,
    )
    lost_path = atomic_save_new_npy(
        os.path.join(output, "lost-edges-per-row.i16.npy"),
        lost_edges_per_row, immutable=True,
    )
    del truth_ids, truth_cos

    order = np.argsort(kth, kind="stable")
    decile_recall = [
        float(tie[order[index * ROWS // 10:(index + 1) * ROWS // 10]].mean())
        for index in range(10)
    ]
    del order

    sort_order = np.argsort(-candidate_cos, axis=1, kind="stable")
    already_sorted = int(
        (sort_order == np.arange(GRAPH_K, dtype=sort_order.dtype)[None, :])
        .all(axis=1).sum()
    )
    ids_sorted = np.take_along_axis(leading, sort_order, axis=1).astype(np.int32)
    cos_sorted = np.take_along_axis(candidate_cos, sort_order, axis=1)
    del sort_order, leading, candidate_cos, strict, tie

    dists = (1.0 - cos_sorted).astype(np.float32)
    negative = int((dists < 0.0).sum())
    most_negative = float(dists.min()) if negative else 0.0
    if most_negative < MIN_ADMISSIBLE_NEGATIVE_DISTANCE:
        raise Round0232Error(
            f"R0232 found a cosine distance of {most_negative!r}, below the "
            f"registered {MIN_ADMISSIBLE_NEGATIVE_DISTANCE} floor"
        )
    np.maximum(dists, 0.0, out=dists)
    if not np.isfinite(dists).all():
        raise Round0232Error("R0232 candidate distances are not finite")
    del cos_sorted

    X = np.array(host, dtype=np.float32, order="C", copy=True)
    import umap.umap_ as umap_api

    fuzzy_started = time.monotonic()
    graph, _sigmas, _rhos = umap_api.fuzzy_simplicial_set(
        X, n_neighbors=GRAPH_K,
        random_state=np.random.RandomState(FUZZY_RANDOM_STATE_SEED),
        metric="cosine", knn_indices=ids_sorted, knn_dists=dists,
    )
    coo = graph.tocoo()
    src = np.asarray(coo.row, dtype=np.int32)
    dst = np.asarray(coo.col, dtype=np.int32)
    wts = np.asarray(coo.data, dtype=np.float32)
    fuzzy_s = time.monotonic() - fuzzy_started
    del X, graph, coo
    gc.collect()

    if not np.isfinite(wts).all() or wts.min() <= 0 or wts.max() > 1:
        raise Round0232Error("R0232 fuzzy weights are invalid")
    if np.any(np.diff(src) < 0):
        raise Round0232Error("R0232 fuzzy edge sources are not sorted")
    degree_counts = np.bincount(src, minlength=ROWS)
    degrees = {
        "zero_degree_rows": int((degree_counts == 0).sum()),
        "min": int(degree_counts.min()),
        "median": float(np.median(degree_counts)),
        "mean": float(degree_counts.mean()),
        "max": int(degree_counts.max()),
    }
    if degrees["zero_degree_rows"] != 0:
        raise Round0232Error(
            f"R0232 R0215 tripwire: {degrees['zero_degree_rows']} zero-degree rows"
        )

    ids_out = atomic_save_new_npy(
        os.path.join(output, "streamed-k15-ids.i32.npy"), ids_sorted, immutable=True,
    )
    graph_path = atomic_save_new_npz(
        os.path.join(output, "edges-k15-fuzzy.npz"), immutable=True,
        compressed=False, sources=src, targets=dst, weights=wts,
        n_nodes=np.asarray(ROWS, dtype=np.int64),
        k=np.asarray(GRAPH_K, dtype=np.int64),
    )
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0232Error(f"R0232 fuzzy peak RSS {peak_rss_gib:.2f} GiB")

    receipt = prompt_contract.seal({
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": GRAPH_CAPABILITY,
        "capabilities": [GRAPH_CAPABILITY],
        "arm": ARM_NAME,
        "arm_cell": dict({
            key: arm.get(key) for key in
            ("cell", "rows", "clusters", "spill", "mode", "bound_bytes",
             "graph_degree", "intermediate_graph_degree", "max_iterations",
             "graph_ids_sha256", "graph_cos_sha256")
        }),
        "clusters": int(arm["clusters"]),
        "spill": int(arm["spill"]),
        "rows": ROWS,
        "dimension": DIMENSION,
        "k": GRAPH_K,
        "builder": {
            "name": "cluster-spill-nnd",
            "spill_design": str(arm["mode"]),
            "source_rounds": ["0226", "0227", "0229", "0232"],
            "neighbour_ids": ids_signature,
            "grid_artifact": grid_signature,
            "nn_descent": {
                "graph_degree": int(arm["graph_degree"]),
                "intermediate_graph_degree": int(arm["intermediate_graph_degree"]),
                "max_iterations": int(arm["max_iterations"]),
            },
            "approximate": True,
            "measured_peak_scratch_bytes": arm["scratch"][
                "measured_peak_scratch_bytes"
            ],
        },
        "recall_against_r0220_exact_truth": {
            "truth_receipt": truth_signature,
            "rows_measured": ROWS,
            "population": RECALL_POPULATION,
            "population_note": RECALL_POPULATION_NOTE,
            "tie_aware": tie_summary,
            "strict": strict_summary,
            "tie_tolerance": TIE_TOLERANCE,
            "density_decile_tie_aware": decile_recall,
            "sparsest_decile_mean": decile_recall[0],
            "densest_decile_mean": decile_recall[-1],
            "sparsest_to_densest_gap": decile_recall[-1] - decile_recall[0],
            "rows_carrying_any_loss": rows_carrying_loss,
            "rows_carrying_any_loss_fraction": rows_carrying_loss / ROWS,
            "total_missing_true_edges": int(lost_edges_per_row.sum()),
            "r0229_sealed_tie_aware": R0229_ARM_TIE_AWARE,
            "r0229_sealed_strict": R0229_ARM_STRICT,
        },
        "loss_arrays": {
            "strict_recall_per_row": expected_input_signature(strict_path),
            "lost_edges_per_row": expected_input_signature(lost_path),
        },
        "fuzzy_law": FUZZY_LAW,
        "fuzzy_random_state_seed": FUZZY_RANDOM_STATE_SEED,
        "neighbour_ordering": {
            "law": "per-row descending exact cosine, stable",
            "rows_already_in_builder_order": already_sorted,
            "rows_reordered": ROWS - already_sorted,
        },
        "distances": {
            "law": "1 - exact fp32 cosine of the substrate rows",
            "negative_entries_clipped_to_zero": negative,
            "most_negative_distance": most_negative,
            "entries": int(ROWS * GRAPH_K),
            "min_admissible_negative_distance": MIN_ADMISSIBLE_NEGATIVE_DISTANCE,
        },
        "substrate": dict(job["substrate_signature"]),
        "streamed_k15_ids": expected_input_signature(ids_out),
        "graph": expected_input_signature(graph_path),
        "graph_checks": {
            "r0215_tripwire_clean": True,
            "zero_degree_rows": degrees["zero_degree_rows"],
            "self_loops": int(structural.get("self_loops", 0)),
            "duplicate_entries": int(structural.get("duplicate_entries", 0)),
            "out_of_range": int(structural.get("out_of_range", 0)),
            "rows_below_k": int(structural.get("rows_below_k", 0)),
            "recall_clears_its_registered_floors": True,
        },
        "structural_validity": structural,
        "degrees": degrees,
        "directed_edge_count": int(len(src)),
        "performance": {
            "candidate_cosine_s": cosine_s,
            "fuzzy_s": fuzzy_s,
            "total_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_host_rss_gib": peak_rss_gib,
        },
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": ADOPTION_CLAIMED,
    })
    atomic_write_new_json(
        os.path.join(output, "streamed-graph.json"), receipt, immutable=True
    )
    del ids_sorted, dists, src, dst, wts
    gc.collect()


# --------------------------------------------------------------------------- #
# node 4 — one train cell, R0217's treatment with the streamed graph
# --------------------------------------------------------------------------- #
def _train_graph(job: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path, manifest_signature = _intra_queue_signature(
        dict(job["graph_manifest_reference"]), label="R0232 streamed graph"
    )
    graph_manifest = prompt_contract.read_sealed(
        manifest_path, label="R0232 streamed graph"
    )
    checks = graph_manifest.get("graph_checks") or {}
    degrees = graph_manifest.get("degrees") or {}
    if (
        graph_manifest.get("schema") != GRAPH_SCHEMA
        or graph_manifest.get("round_id") != ROUND_ID
        or graph_manifest.get("capability") != GRAPH_CAPABILITY
        or int(graph_manifest.get("rows", -1)) != ROWS
        or int(graph_manifest.get("dimension", -1)) != DIMENSION
        or int(graph_manifest.get("k", -1)) != GRAPH_K
        or graph_manifest.get("training_performed") is not False
        or graph_manifest["recall_against_r0220_exact_truth"]["population"]
        != RECALL_POPULATION
    ):
        raise Round0232Error("R0232 sealed streamed graph contract changed")
    if (
        int(checks.get("zero_degree_rows", -1)) != 0
        or int(degrees.get("zero_degree_rows", -1)) != 0
        or checks.get("r0215_tripwire_clean") is not True
        or checks.get("recall_clears_its_registered_floors") is not True
    ):
        raise Round0232Error(
            "R0232 requires the sealed streamed graph to have passed its "
            "zero-degree and recall-floor checks"
        )
    edges = int(graph_manifest["directed_edge_count"])
    if edges <= 0:
        raise Round0232Error("R0232 sealed streamed graph reports no edges")
    edges_path, graph_signature = _intra_queue_signature(
        dict(graph_manifest["graph"]), label="R0232 streamed fuzzy edges"
    )
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )

    sources, targets, weights, n_nodes = load_edge_arrays(
        edges_path, load_weights=True
    )
    if (
        weights is None
        or int(n_nodes) != ROWS
        or len(sources) != edges
        or targets.shape != sources.shape
        or weights.shape != sources.shape
        or sources.dtype != np.int32
        or targets.dtype != np.int32
        or weights.dtype != np.float32
    ):
        raise Round0232Error("R0232 sealed streamed graph arrays changed")
    return {
        "manifest": graph_manifest,
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "edges_path": edges_path,
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": int(n_nodes),
        "directed_edges": edges,
    }


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    seed = int(job["training_seed"])
    if seed not in SEEDS:
        raise Round0232Error(f"R0232 seed {seed} is not registered")
    if str(job.get("capability")) != map_capability(seed):
        raise Round0232Error("R0232 train job capability does not match its seed")

    bundle = _train_graph(job)
    graph_manifest = bundle["manifest"]
    manifest_signature = bundle["manifest_signature"]
    graph_signature = bundle["signature"]
    graph_path = bundle["edges_path"]
    edges = int(bundle["directed_edges"])
    updates = successful_updates_for_edges(edges)
    dose = validate_dose(updates=updates, edge_count=edges)

    substrate_signature = dict(graph_manifest["substrate"])
    source = np.load(
        prompt_contract.verify_signature(substrate_signature, label="R0216 substrate"),
        mmap_mode="r", allow_pickle=False,
    )

    clusters = int(graph_manifest["clusters"])
    spill = int(graph_manifest["spill"])
    nn_descent = dict(graph_manifest["builder"]["nn_descent"])
    config, config_sha, invariant = train_config(
        clusters=clusters, spill=spill, nn_descent=nn_descent, seed=seed,
        graph_signature=graph_signature,
        graph_manifest_signature=manifest_signature,
        substrate_signature=substrate_signature,
        r0216_graph_signature=dict(job["r0216_graph_signature"]),
        r0216_graph_manifest_signature=dict(job["r0216_graph_manifest_signature"]),
        graph_edges=edges, rows=ROWS,
    )
    if invariant != TREATMENT_INVARIANT_SHA256:
        raise Round0232Error(
            "R0232 cell config is not R0217's treatment outside the seed and the "
            f"graph: {invariant} != {TREATMENT_INVARIANT_SHA256}"
        )

    output = create_fresh_directory(str(job["outputs"][0]), label="R0232 train output")
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": PRODUCTION_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "treatment_config_round_id": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "arm": ARM_NAME,
            "clusters": clusters,
            "spill": spill,
            "seed": seed,
            "capability": map_capability(seed),
            "treatment_invariant_sha256": invariant,
            "config": config,
            "config_sha256": config_sha,
            "config_constructed_by": (
                "basemap/round0229_train_config.train_config, imported read-only "
                "so the treatment-invariant digest is the cross-round constant. "
                "Its graph-bearing strings name R0229's arm capability because "
                "that construction is R0229's; those paths are masked out of the "
                "invariant digest by R0223's registered GRAPH_BEARING_PATHS, "
                "which is exactly why re-using it is safe."
            ),
        },
        immutable=True,
    )

    from basemap.round0217_minilm_2m_pipeline import (
        MiniLMHostFp32EndpointArray,
        MiniLMMixedTrainingInput,
    )

    graph_bundle = dict(bundle)
    graph_bundle.pop("edges_path", None)
    dataset = MiniLMHostFp32EndpointArray(
        source, source_signature=substrate_signature, buffer_rows=BATCH_SIZE
    )
    wrapper = MiniLMMixedTrainingInput(dataset, graph_bundle, seed=seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = prompt_nodes._new_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = performance_windows(updates)
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")
    started = time.monotonic()
    model.fit(
        wrapper, low_memory=True, verbose=False, n_processes=6, random_state=seed,
        resample_negatives=False, precomputed_edges_path=graph_path, use_wandb=False,
    )
    wall = time.monotonic() - started

    accounting = dict(model._train_stats)
    runtime = wrapper.runtime_stamp()
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_stamp.items()
        if runtime.get(key) != value
    }
    exact = {
        "lr_horizon": updates,
        "positive_lr_optimizer_steps": updates,
        "scheduler_steps": updates,
        "attempted_batches": updates,
        "finite_loss_batches": updates,
        "optimizer_steps_attempted": updates,
        "optimizer_steps_succeeded": updates,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": edges,
    }
    mismatches.update({
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    })
    expected_rows = updates * BATCH_SIZE
    producer_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        int(runtime["source_rows_gathered"]) != expected_rows
        or int(runtime["destination_rows_gathered"]) != expected_rows
        or int(runtime["host_prefetch_consumer_batches"]) != updates
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows, "runtime": runtime,
        }
    weighted = round0221_nodes._weighted_rejection_accounting_mismatch(
        runtime, producer_delta=producer_delta, updates=updates
    )
    if weighted is not None:
        mismatches["weighted_rejection_accounting"] = weighted
    if mismatches:
        raise Round0232Error(f"R0232 train accounting failed: {mismatches}")
    prompt_contract.synchronize_runtime_counters(accounting, runtime)
    accounting["pipeline_runtime"] = dict(runtime)

    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (updates - WARMUP_SUCCESSFUL_UPDATES) / model._bench_seconds
        if model._bench_seconds else 0.0
    )
    if profiler.get("aborted") is not False or rate < config["execution"][
        "minimum_train_upd_s"
    ]:
        raise Round0232Error("R0232 train performance admission failed")

    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    memory = {
        "device_total_bytes": int(total_bytes),
        "post_train_free_bytes": int(free_bytes),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
    }
    del model, wrapper, dataset
    torch.cuda.empty_cache()
    gc.collect()

    from basemap.pumap.parametric_umap import ParametricUMAP

    reloaded = ParametricUMAP.load(model_path, device="cuda")
    coordinates = np.asarray(
        reloaded.transform(source, batch_size=FULL_TRANSFORM_BATCH), dtype=np.float32
    )
    published = validate_full_population_map(coordinates)
    published["model"] = expected_input_signature(model_path)
    coordinates_path = atomic_save_new_npy(
        os.path.join(output, f"coordinates-seed{seed}.npy"), coordinates,
        immutable=True,
    )
    del reloaded, coordinates
    torch.cuda.empty_cache()
    gc.collect()

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0232Error(f"R0232 train peak RSS {peak_rss_gib:.2f} GiB")
    memory["peak_host_rss_gib"] = peak_rss_gib

    receipt = prompt_contract.seal({
        "schema": TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "treatment_config_round_id": "0217",
        "treatment_template_seed": TEMPLATE_SEED,
        "capability": map_capability(seed),
        "capabilities": [map_capability(seed)],
        "arm": ARM_NAME,
        "clusters": clusters,
        "spill": spill,
        "nn_descent": nn_descent,
        "training_seed": seed,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "treatment_invariant_sha256": invariant,
        "model": published["model"],
        "coordinates": expected_input_signature(coordinates_path),
        "substrate": substrate_signature,
        "graph_manifest": manifest_signature,
        "graph": graph_signature,
        "graph_capability": GRAPH_CAPABILITY,
        "graph_recall": dict(graph_manifest["recall_against_r0220_exact_truth"]),
        "rows": ROWS,
        "dimension": DIMENSION,
        "directed_edges": edges,
        "dose_registration": dose,
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "requested_positive_draws_per_edge": float(
            config["execution"]["target_positive_draws_per_edge"]
        ),
        "consumed_positive_draws": int(updates * POSITIVE_ROWS_PER_UPDATE),
        "consumed_positive_draws_per_edge": float(
            updates * POSITIVE_ROWS_PER_UPDATE / edges
        ),
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_s": wall,
        "published_map_check": published,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "endpoint_rows_match_updates": True,
            "weighted_rejection_accounting_closes": True,
            "dose_derived_from_sealed_edge_count": True,
            "treatment_identical_to_r0217_except_seed_and_graph": True,
            "treatment_digest_equals_cross_round_constant": True,
            "published_checkpoint_reloads_finite_and_uncollapsed": True,
            "all_2m_coordinates_finite": True,
        },
        "memory": memory,
        "training_performed": True,
        "optimizer_updates": updates,
        "map_decision_made": False,
        "adoption_claimed": ADOPTION_CLAIMED,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    })
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), receipt, immutable=True
    )
    del source
    gc.collect()


# --------------------------------------------------------------------------- #
# node 5 — the geometry, R0228's code read-only
# --------------------------------------------------------------------------- #
def _load_coordinates(signature: Mapping[str, Any], *, label: str) -> np.ndarray:
    path = prompt_contract.verify_signature(dict(signature), label=label)
    array = np.load(path, allow_pickle=False)
    if array.shape != (ROWS, 2):
        raise Round0232Error(f"{label} has shape {array.shape}, expected ({ROWS}, 2)")
    return np.asarray(array, dtype=np.float32)


def run_geometry(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    graph_path, graph_manifest_signature = _intra_queue_signature(
        dict(job["graph_manifest_reference"]), label="R0232 streamed graph"
    )
    graph_manifest = prompt_contract.read_sealed(
        graph_path, label="R0232 streamed graph"
    )
    lost_path = prompt_contract.verify_signature(
        dict(graph_manifest["loss_arrays"]["lost_edges_per_row"]),
        label="R0232 lost-edge array",
    )
    lost = np.load(lost_path, allow_pickle=False)
    if lost.shape != (ROWS,):
        raise Round0232Error("R0232 lost-edge array has the wrong shape")

    truth, truth_signature = _sealed(job, "truth_signature", label="R0220 truth")
    truth_ids = np.load(
        _verified_path(truth["outputs"]["ids"], label="R0220 truth ids"),
        mmap_mode="r", allow_pickle=False,
    )
    truth_cos = np.load(
        _verified_path(truth["outputs"]["cosines"], label="R0220 truth cosines"),
        mmap_mode="r", allow_pickle=False,
    )
    kth_cosine = np.asarray(truth_cos[:, GRAPH_K - 1], dtype=np.float64)

    r0228_geometry, r0228_signature = _sealed(
        job, "r0228_geometry_signature", label="R0228 geometry"
    )
    bound = verify_r0228_displacement(r0228_geometry)

    selection = density_matched_control(
        lost_mask=lost > 0, kth_cosine=kth_cosine,
        sample_rows=SCATTER_SAMPLE_ROWS, seed=SCATTER_SAMPLE_SEED,
    )
    lost_rows = np.asarray(selection["lost_sample"], dtype=np.int64)
    control_rows = np.asarray(selection["control_sample"], dtype=np.int64)

    candidate_names: list[str] = []
    exact_names: list[str] = []
    lost_scatter: dict[str, Any] = {}
    control_scatter: dict[str, Any] = {}
    clumps: dict[str, Any] = {}

    for entry in job["candidate_coordinates"]:
        name = str(entry["name"])
        path, _ = _intra_queue_signature(
            dict(entry["signature"]), label=f"R0232 {name} coordinates"
        )
        coordinates = np.load(path, allow_pickle=False)
        if coordinates.shape != (ROWS, 2):
            raise Round0232Error(f"R0232 {name} coordinates have the wrong shape")
        coordinates = np.asarray(coordinates, dtype=np.float32)
        scale = map_scale(coordinates)
        candidate_names.append(name)
        lost_scatter[name] = true_neighbour_scatter(
            coordinates, truth_ids, lost_rows, scale=scale
        ).tolist()
        control_scatter[name] = true_neighbour_scatter(
            coordinates, truth_ids, control_rows, scale=scale
        ).tolist()
        clumps[name] = clump_profile(coordinates)
        clumps[name]["map_rms_radius"] = scale
        del coordinates

    for entry in job["exact_coordinates"]:
        name = str(entry["name"])
        coordinates = _load_coordinates(entry["signature"], label=f"R0232 {name}")
        scale = map_scale(coordinates)
        exact_names.append(name)
        lost_scatter[name] = true_neighbour_scatter(
            coordinates, truth_ids, lost_rows, scale=scale
        ).tolist()
        control_scatter[name] = true_neighbour_scatter(
            coordinates, truth_ids, control_rows, scale=scale
        ).tolist()
        clumps[name] = clump_profile(coordinates)
        clumps[name]["map_rms_radius"] = scale
        del coordinates

    summary = displacement_summary(
        lost_scatter=lost_scatter, control_scatter=control_scatter,
        candidate_maps=candidate_names, exact_maps=exact_names,
    )
    candidate_gaps = [
        float(summary["per_map"][name]["gap_lost_minus_control"])
        for name in candidate_names
    ]
    exact_gaps = [
        float(summary["per_map"][name]["gap_lost_minus_control"])
        for name in exact_names
    ]
    per_arm = exact_displacement_permutation(
        candidate_gaps=candidate_gaps, exact_gaps=exact_gaps
    )
    per_arm["verdict"] = displacement_verdict(per_arm)
    per_arm["arm"] = ARM_NAME
    per_arm["licensed_statement"] = licensed_statement(
        p_value=float(per_arm["p_one_sided"])
    )
    per_arm["minimum_detectable_displacement_sd"] = MINIMUM_DETECTABLE_DISPLACEMENT_SD
    per_arm["minimum_detectable_effect_note"] = NON_REJECTION_NOTE
    per_arm["multiplicity"] = {
        "new_arms_tested_here": 1,
        "correction_applied": "none",
        "note": (
            "one new arm means one new test at alpha = 0.05; the registered "
            "Holm-Bonferroni machinery is inert at a single test and is reported "
            "as inert rather than silently omitted. Its smallest attainable p is "
            "published beside it (review-0228-01)."
        ),
    }

    loss_fraction = float(
        graph_manifest["recall_against_r0220_exact_truth"][
            "rows_carrying_any_loss_fraction"
        ]
    )
    arm_values = {
        "c4": per_map_did(
            candidate_gaps=bound["cells"]["4"]["candidate_gaps"],
            exact_gaps=bound["cells"]["4"]["exact_gaps"],
        ),
        ARM_NAME: per_map_did(
            candidate_gaps=candidate_gaps, exact_gaps=exact_gaps
        ),
        "c16": per_map_did(
            candidate_gaps=bound["cells"]["16"]["candidate_gaps"],
            exact_gaps=bound["cells"]["16"]["exact_gaps"],
        ),
    }
    regressor = {
        "c4": R0228_ROWS_CARRYING_LOSS_BY_C[4],
        ARM_NAME: loss_fraction,
        "c16": R0228_ROWS_CARRYING_LOSS_BY_C[16],
    }
    trend = exact_did_trend(arm_values=arm_values, regressor=regressor)
    trend["arm_order_by_missing_edge_mass"] = sorted(
        regressor, key=lambda name: regressor[name]
    )
    trend["what_this_test_can_carry"] = (
        "review-0229-01 §3: a cleaner arm would make this trend MORE significant, "
        "so its p is not a statement about the new arm at all — it is powered by "
        "c = 16. A three-point trend discriminates 'c = 16-like' from 'not "
        "c = 16-like' and nothing finer."
    )

    checks = {
        "geometry_code_imported_from_r0228_not_reimplemented": True,
        "density_match_exact": bool(selection.get("matched_exactly")),
        "null_arm_is_the_eight_exact_maps": len(exact_names) == 8,
        "candidate_arm_has_three_seeds": len(candidate_names) == 3,
        "same_row_sets_for_every_map": True,
        "smallest_attainable_p_published_beside_every_p": all(
            "smallest_attainable_p" in value for value in (per_arm, trend)
        ),
        "no_test_published_below_its_own_resolution": all(
            test_can_reject(
                smallest_attainable_p=float(value["smallest_attainable_p"]),
                threshold=DISPLACEMENT_ALPHA,
            )
            for value in (per_arm, trend)
        ),
        "non_rejection_stated_as_under_one_sd_not_equivalence": True,
        "zero_degree_rows_published": True,
        "no_registered_cell_dropped": True,
        "bound_to_r0228_sealed_bytes": True,
    }

    atomic_write_new_json(
        str(job["artifact_path"]),
        {
            "schema": GEOMETRY_SCHEMA,
            "round_id": ROUND_ID,
            "release_sha": manifest["release_sha"],
            "capability": GEOMETRY_CAPABILITY,
            "capabilities": [GEOMETRY_CAPABILITY],
            "outcome": "does-the-streamed-restructure-move-the-map-at-2m",
            "arm": ARM_NAME,
            "arm_recall": dict(
                graph_manifest["recall_against_r0220_exact_truth"]
            ),
            "arm_cell": dict(graph_manifest["arm_cell"]),
            "rows": ROWS,
            "scatter_definition": SCATTER_DEFINITION,
            "scale_definition": SCALE_DEFINITION,
            "null_arm_note": NULL_ARM_NOTE,
            "clump_definition": CLUMP_DEFINITION,
            "scatter_sample_rows": SCATTER_SAMPLE_ROWS,
            "scatter_sample_seed": SCATTER_SAMPLE_SEED,
            "density_match": {
                key: value for key, value in selection.items()
                if key not in {"lost_sample", "control_sample"}
            },
            "displacement": summary,
            "per_arm_permutation_test": per_arm,
            "did_trend_test": trend,
            "permutation_labellings": PERMUTATION_LABELLINGS,
            "smallest_attainable_p_by_design": PERMUTATION_RESOLUTION_CEILING,
            "minimum_detectable_displacement_sd": MINIMUM_DETECTABLE_DISPLACEMENT_SD,
            "non_rejection_note": NON_REJECTION_NOTE,
            "alpha": DISPLACEMENT_ALPHA,
            "zero_degree_rows": int(graph_manifest["degrees"]["zero_degree_rows"]),
            "clump_profiles": clumps,
            "r0228_geometry": r0228_signature,
            "r0228_bound": bound,
            "graph_manifest": graph_manifest_signature,
            "truth": truth_signature,
            "execution_checks": checks,
            "training_performed": False,
            "gate_registerable_here": GATE_REGISTERABLE_HERE,
            "gate_release_claimed": GATE_RELEASE_CLAIMED,
            "gate_registered": False,
            "adoption_claimed": ADOPTION_CLAIMED,
            "equivalence_claimed": EQUIVALENCE_CLAIMED,
            "production_or_publishing": False,
        },
        immutable=True,
    )


# --------------------------------------------------------------------------- #
# node 6 — the projection, emitted from a node and hash-bound
# --------------------------------------------------------------------------- #
def _cluster_wall_points(cells: Sequence[Mapping[str, Any]], builds_root: str) -> tuple[
    list[float], list[float]
]:
    sizes: list[float] = []
    seconds: list[float] = []
    for cell in cells:
        if not cell.get("fit"):
            continue
        receipt_path = os.path.join(
            builds_root, str(cell["setting_id"]), "build-receipt.json"
        )
        if not os.path.exists(receipt_path):
            continue
        with open(receipt_path, encoding="utf-8") as handle:
            receipt = json.load(handle)
        for record in receipt.get("cluster_receipts") or []:
            if bool(record.get("brute_force")):
                continue
            sizes.append(float(record["rows"]))
            seconds.append(float(record["nn_descent_seconds"]))
    return sizes, seconds


def run_projection(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    grid_path, grid_signature = _intra_queue_signature(
        dict(job["grid_reference"]), label="R0232 scratch-law grid"
    )
    larger_path, larger_signature = _intra_queue_signature(
        dict(job["larger_n_reference"]), label="R0232 larger-N calibration"
    )
    with open(grid_path, encoding="utf-8") as handle:
        grid = json.load(handle)
    with open(larger_path, encoding="utf-8") as handle:
        larger = json.load(handle)

    cells = list(grid["cells"]) + list(larger["cells"])
    fitted = [cell for cell in cells if cell.get("fit")]
    if len(fitted) < 3:
        raise Round0232Error("R0232 projection needs at least three fitted cells")

    # --- the device law, refitted at gd 64 / igd 256 (review-0229-01 item 2) ---
    device_points = [
        (
            float(cell["cluster_sizes"]["max"]),
            float(cell["memory"]["device_wide_peak_over_baseline_bytes"]),
        )
        for cell in fitted
        if cell["memory"].get("device_wide_peak_over_baseline_bytes")
        and int(cell["graph_degree"]) == 64
    ]
    device_fit = linear_fit(
        [point[0] for point in device_points], [point[1] for point in device_points]
    )
    device_fit["stratum"] = "graph_degree 64 / intermediate_graph_degree 256"
    device_fit["instrument"] = "device_wide_peak_over_baseline_bytes (parent, 250 ms)"
    device_fit["why_refitted"] = (
        "review-0229-01 §5.2: R0227's law is calibrated only at gd 32 / igd 48 and "
        "under-predicts R0229's own gd 64 / igd 256 arm by +2.38 GiB (4.575 "
        "predicted against 6.957 measured). R0229 registered the refit and did not "
        "do it. This is the refit, on this round's own measurements, over a "
        "largest-cluster range that reaches a real multi-million-row cluster."
    )
    refitted_capacity = capacity_rows_at_device_budget(device_fit)

    # --- the wall law, one stratum, no pooling across builder settings ---
    grid_builds = os.path.join(os.path.dirname(grid_path), "builds")
    larger_builds = os.path.join(os.path.dirname(larger_path), "builds")
    sizes, seconds = _cluster_wall_points(grid["cells"], grid_builds)
    more_sizes, more_seconds = _cluster_wall_points(larger["cells"], larger_builds)
    wall_fit = power_fit(sizes + more_sizes, seconds + more_seconds)
    wall_fit["stratum"] = "graph_degree 64 / intermediate 256 / max_iterations 40"
    wall_fit["pooling_note"] = (
        "every point in this fit is a gd 64 / igd 256 / it 40 build. Review-0229-01 "
        "§5.3 found R0229's fit pooled 180 gd-32 points with 200 gd-64 points and "
        "projected a gd-64 build from an exponent belonging to neither. This fit "
        "does not pool, and its fitted range is stated so the extrapolation factor "
        "is visible."
    )

    # --- the I/O law, from this round's own measured throughput ---
    throughput = larger.get("data_throughput") or {}
    read_rate = float(
        throughput.get("read_bytes_per_s") or DATA_COLD_READ_BYTES_PER_S
    )
    write_rate = float(
        throughput.get("write_bytes_per_s") or DATA_COLD_READ_BYTES_PER_S
    )

    # --- the scratch law over every measured cell ---
    law = scratch_law([
        {**cell["scratch"], **{
            "cell": cell["cell"], "rows": cell["rows"], "clusters": cell["clusters"],
            "spill": cell["spill"], "mode": cell["mode"],
            "bound_bytes": cell["bound_bytes"],
        }}
        for cell in fitted
    ])

    # --- the rungs ---
    measured_imbalance = {
        int(cell["clusters"]): float(cell["cluster_sizes"]["imbalance_max_over_mean"])
        for cell in fitted if cell.get("cluster_sizes")
    }
    rungs: list[dict[str, Any]] = []
    for rows in PHASE2_RUNGS_TO_PROJECT:
        per_spill: dict[str, Any] = {}
        for spill in (2, 8):
            rung = smallest_measured_clusters(rows=rows, spill=spill)
            clusters = int(rung["clusters"])
            largest = float(rung["projected_max_cluster_rows"])
            designs = {}
            for mode, bound in (
                ("materialise", SCRATCH_BUDGET_BYTES),
                ("stream-resident", SCRATCH_BUDGET_BYTES),
                ("stream-gather", 0),
            ):
                io = io_projection(
                    rows=rows, clusters=clusters, spill=spill, mode=mode,
                    bound_bytes=bound,
                    imbalance=float(rung["measured_imbalance"]),
                    read_bytes_per_s=read_rate, write_bytes_per_s=write_rate,
                )
                io["ladder_disk"] = ladder_disk_requirement(
                    rows=rows, peak_scratch_bytes=int(io["peak_scratch_bytes"])
                )
                designs[mode] = io
            nn_descent = project_from_power_fit(wall_fit, largest / float(
                rung["measured_imbalance"]
            ))
            nn_descent["clusters"] = clusters
            nn_descent["total_seconds_all_clusters"] = (
                float(nn_descent["seconds"]) * clusters
            )
            nn_descent["mean_cluster_used_note"] = (
                "the fit is concave (b < 1), so evaluating at the MEAN cluster and "
                "multiplying by the cluster count is conservative by Jensen"
            )
            per_spill[str(spill)] = {
                "per_rung_clusters": rung,
                "device": device_law_prediction(largest, device_fit),
                "nn_descent": nn_descent,
                "designs": designs,
            }
        rungs.append({"rows": rows, "by_spill": per_spill})

    # --- the deliverable ---
    hundred = next(entry for entry in rungs if entry["rows"] == 100_000_000)
    free_now = data_free_bytes()
    deliverable_rows = []
    for mode, design in hundred["by_spill"]["8"]["designs"].items():
        ladder = design["ladder_disk"]
        deliverable_rows.append({
            "design": mode,
            "peak_scratch_bytes": int(design["peak_scratch_bytes"]),
            "peak_scratch_gb": int(design["peak_scratch_bytes"]) / 1e9,
            "substrate_gb": ladder["substrate_bytes"] / 1e9,
            "neighbour_arrays_gb": (
                ladder["neighbour_ids_bytes"] + ladder["neighbour_cosines_bytes"]
            ) / 1e9,
            "fuzzy_edge_file_gb": ladder["fuzzy_edge_file_bytes"] / 1e9,
            "total_gb_at_peak": ladder["total_gb_at_peak"],
            "data_free_gb_now": free_now / 1e9,
            "fits_in_current_free_space": bool(
                ladder["total_bytes_at_peak"] <= free_now
            ),
            "shortfall_gb": max(
                0.0, (ladder["total_bytes_at_peak"] - free_now) / 1e9
            ),
            "extra_host_anon_gb": int(design["extra_host_anon_bytes"]) / 1e9,
            "total_bytes_moved_tb": float(design["total_bytes_moved"]) / 1e12,
            "io_hours": float(design["hours"]),
        })

    artifact = {
        "schema": PROJECTION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": PROJECTION_CAPABILITY,
        "capabilities": [PROJECTION_CAPABILITY],
        "outcome": "does-100m-fit-the-disk-we-have",
        "design_note": DESIGN_NOTE,
        "measured_scratch_law": law,
        "device_law_refit": device_fit,
        "device_law_refitted_capacity_rows": refitted_capacity,
        "device_law_registered_capacity_rows": CLUSTER_CAPACITY_ROWS,
        "wall_law": wall_fit,
        "measured_data_throughput": throughput,
        "measured_read_bytes_per_s": read_rate,
        "measured_write_bytes_per_s": write_rate,
        "rungs": rungs,
        "deliverable_100m_s8": deliverable_rows,
        "data_free_bytes_at_projection": free_now,
        "spill_volume_at_100m_s8_bytes": SPILL_VOLUME_100M_S8_BYTES,
        "measured_imbalance_this_round": measured_imbalance,
        "projection_discipline": (
            "every figure here is labelled a projection, carries its fitted range "
            "and extrapolation factor, and no projection is divided by another "
            "projection. Scratch, substrate passes and I/O are their own lines and "
            "are never folded into a compute fit."
        ),
        "grid_artifact": grid_signature,
        "larger_n_artifact": larger_signature,
        "training_performed": TRAINING_PERFORMED,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "gate_release_claimed": GATE_RELEASE_CLAIMED,
        "gate_registered": False,
        "adoption_claimed": ADOPTION_CLAIMED,
        "equivalence_claimed": EQUIVALENCE_CLAIMED,
        "production_or_publishing": False,
        "execution_checks": {
            "device_law_refitted_at_the_winning_setting": True,
            "wall_law_not_pooled_across_builder_settings": True,
            "io_rate_measured_in_this_round": bool(
                throughput.get("read_bytes_per_s")
            ),
            "scratch_is_its_own_line": True,
            "no_projection_divided_by_a_projection": True,
            "emitted_from_a_node_not_prose": True,
        },
    }
    atomic_write_new_json(str(job["artifact_path"]), artifact, immutable=True)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0232Error("R0232 handler received another queue")
    action = str(job.get("action") or "")
    if action == GRID_ACTION:
        run_grid(active, job)
    elif action == LARGER_N_ACTION:
        run_larger_n(active, job)
    elif action == FUZZY_ACTION:
        run_fuzzy(active, job)
    elif action == TRAIN_ACTION:
        run_train(active, job)
    elif action == GEOMETRY_ACTION:
        run_geometry(active, job)
    elif action == PROJECT_ACTION:
        run_projection(active, job)
    else:
        raise Round0232Error(f"unknown R0232 action {action!r}")


__all__ = [
    "FUZZY_ACTION",
    "GEOMETRY_ACTION",
    "GRID_ACTION",
    "LARGER_N_ACTION",
    "PROJECT_ACTION",
    "TRAIN_ACTION",
    "run_fuzzy",
    "run_geometry",
    "run_grid",
    "run_job",
    "run_larger_n",
    "run_projection",
    "run_train",
]
