"""Execute R0229 phase 2 — build, train and test the spill-lifted arm.

Six nodes, one queue:

* `build_spill_lifted` (GPU) builds the selected `(c, s)` configuration at 2M
  with the quality sweep's winning nn-descent setting, under R0227's guard and
  watchdog, using this round's own build script.
* `fuzzy_spill_lifted` (GPU) scores it over **all 2,000,000 rows** against
  R0220's sealed exact truth, applies the R0215 tripwire, records which rows lost
  which edges, and symmetrises through R0216's identical fuzzy law. This is
  R0228's fuzzy node with the arm's identity swapped for its cluster count.
* `train_spill_lifted_seed{42,43,44}` (GPU) is R0228's train node with the graph
  swapped. Each cell rebuilds its config from R0217's own template and refuses to
  train unless the treatment-invariant digest equals the cross-round constant
  `c28cfd61...` that R0217, R0221, R0223 and R0228 all carry.
* `probe_spill_lifted_geometry` (GPU lease, CPU work) runs R0228's registered
  displacement statistic, imported read-only from `basemap/round0228_geometry.py`
  so the numbers are directly comparable, and both registered inference rules:
  the per-arm exact permutation test and the three-arm DiD trend test across
  `{c = 4, spill-lifted, c = 16}` that review-0228-01 recommended.

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
from collections.abc import Mapping
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
from basemap.round0227_low_c_contract import SAMPLE_INTERVAL_S
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
from basemap.round0229_train_config import train_config
from basemap.round0229_phase2_contract import (
    ADOPTION_CLAIMED,
    ARM_NAME,
    DIMENSION,
    EQUIVALENCE_CLAIMED,
    GATE_REGISTERABLE_HERE,
    GATE_RELEASE_CLAIMED,
    GEOMETRY_CAPABILITY,
    GEOMETRY_SCHEMA,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    PRODUCTION_CONFIG_SCHEMA,
    PRODUCTION_CONFIG_SCHEMA as CONFIG_SCHEMA,
    ROUND_ID,
    ROWS,
    SEEDS,
    TRAIN_SCHEMA,
    TREATMENT_INVARIANT_SHA256,
    TREND_ARMS,
    map_capability,
    per_map_did,
    select_arm,
)
from basemap.round0229_quality_contract import (
    DECISION_RULE_NOTE,
    DISPLACEMENT_ALPHA,
    R0228_ROWS_CARRYING_LOSS_BY_C,
    RECALL_POPULATION,
    RECALL_POPULATION_NOTE,
    RESOLUTION_RULE_NOTE,
    Round0229Error,
    displacement_verdict,
    exact_did_trend,
    exact_displacement_permutation,
    test_can_reject,
    verify_r0228_displacement,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0113_nodes as prompt_nodes
from experiments import round0221_nodes, round0227_nodes

BUILD_ACTION = "build_spill_lifted"
FUZZY_ACTION = "fuzzy_spill_lifted"
TRAIN_ACTION = "train_spill_lifted"
GEOMETRY_ACTION = "probe_spill_lifted_geometry"

QUALITY_BUILD_SCRIPT = "basemap/round0229_quality_build.py"


def _sealed(job: Mapping[str, Any], key: str, *, label: str) -> tuple[
    dict[str, Any], dict[str, Any]
]:
    """A prior round's identity-sealed artifact, bound by its signature."""
    signature = dict(job[key])
    path = prompt_contract.verify_signature(signature, label=label)
    return prompt_contract.read_sealed(path, label=label), signature


def _verified_json(job: Mapping[str, Any], key: str, *, label: str) -> tuple[
    dict[str, Any], dict[str, Any]
]:
    """A hash-bound JSON artifact that carries no `prompt_contract` identity seal.

    R0229's own phase-1 artifacts are written with `atomic_write_new_json` and
    are therefore bound by their `{path, bytes, sha256}` signature rather than by
    an identity seal. The signature is the guarantee `roundreport` verifies, so
    it is verified here and the seal is not demanded of an artifact that never
    claimed one.
    """
    signature = dict(job[key])
    path = prompt_contract.verify_signature(signature, label=label)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle), signature


def _intra_queue_signature(
    reference: Mapping[str, Any], *, label: str
) -> tuple[str, dict[str, Any]]:
    """Resolve a reference to an artifact produced earlier in THIS queue.

    R0228's geometry node died on `verify_signature` of an intra-queue reference
    that carries a path and no hash at prepare time. This is R0228's own fix,
    used for every reference this queue produces itself.
    """
    reference = dict(reference)
    if reference.get("sha256"):
        return prompt_contract.verify_signature(reference, label=label), reference
    path = str(reference["canonical_path"])
    if not os.path.exists(path):
        raise Round0229Error(f"{label} is absent at {path}")
    return path, expected_input_signature(path)


def _verified_arm(job: Mapping[str, Any]) -> dict[str, Any]:
    """Re-run the registered selection rule against phase 1's sealed artifacts."""
    sweep, sweep_signature = _verified_json(
        job, "sweep_signature", label="R0229 sweep"
    )
    spill, spill_signature = _verified_json(
        job, "spill_signature", label="R0229 spill grid"
    )
    chosen = select_arm(sweep=sweep, spill=spill)
    declared = dict(job["arm"])
    for key in ("cell", "clusters", "spill"):
        if chosen[key] != declared.get(key):
            raise Round0229Error(
                f"R0229 phase 2 queue binds arm {declared.get(key)!r} for {key} "
                f"but the registered rule selects {chosen[key]!r}"
            )
    for key in ("graph_degree", "intermediate_graph_degree", "max_iterations"):
        if chosen["nn_descent"][key] != declared.get("nn_descent", {}).get(key):
            raise Round0229Error(
                f"R0229 phase 2 queue binds a different nn-descent {key}"
            )
    chosen["sweep_signature"] = sweep_signature
    chosen["spill_signature"] = spill_signature
    return chosen


# --------------------------------------------------------------------------- #
# node 1 — build the selected configuration
# --------------------------------------------------------------------------- #
def run_build(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    repo_root = str(manifest["repo_root"])
    arm = _verified_arm(job)
    out_dir = ensure_data_directory(str(job["artifact_dir"]))
    substrate_path = prompt_contract.verify_signature(
        dict(job["substrate_signature"]), label="R0216 substrate"
    )
    guard = dict(job["guard"])
    config = {
        "setting_id": f"spill-lifted-{arm['cell']}",
        "cell": arm["cell"],
        "candidate": "cluster-spill-nnd",
        "rows": ROWS,
        "clusters": int(arm["clusters"]),
        "spill": int(arm["spill"]),
        "dimension": DIMENSION,
        "k": GRAPH_K,
        "substrate": substrate_path,
        "emit_graph": True,
        "scratch_root": str(job["scratch_root"]),
        "sample_interval_s": SAMPLE_INTERVAL_S,
        "assignment_cache": None,
        "graph_degree": int(arm["nn_descent"]["graph_degree"]),
        "intermediate_graph_degree": int(
            arm["nn_descent"]["intermediate_graph_degree"]
        ),
        "max_iterations": int(arm["nn_descent"]["max_iterations"]),
    }
    record = round0227_nodes._run_child(
        command=[
            round0227_nodes.CUML_LAUNCHER,
            os.path.join(repo_root, QUALITY_BUILD_SCRIPT),
        ],
        config=config,
        out_dir=out_dir,
        cache_root=str(job["cache_root"]),
        repo_root=repo_root,
        receipt_name="build-receipt.json",
        guard=guard,
    )
    if not record.get("fit"):
        raise Round0229Error(
            f"R0229 spill-lifted build did not fit: {record.get('error_type')}"
        )
    ids_path = os.path.join(out_dir, "graph-k15-ids.i32.npy")
    if not os.path.exists(ids_path):
        raise Round0229Error("R0229 spill-lifted build emitted no graph")
    escalations = list(record.get("watchdog_escalations") or [])
    atomic_write_new_json(
        str(job["artifact_path"]),
        {
            "schema": "round0229-spill-lifted-build-v1",
            "round_id": ROUND_ID,
            "release_sha": manifest["release_sha"],
            "arm": arm,
            "build": record,
            "neighbour_ids": expected_input_signature(ids_path),
            "guard": guard,
            "execution_checks": {
                "guard_allowed": bool(guard.get("allowed")),
                "cell_fit": True,
                "not_refused_a_priori": not bool(record.get("refused_a_priori")),
                "not_aborted_by_watchdog": not bool(
                    record.get("aborted_by_watchdog")
                ),
                "not_timed_out": not bool(record.get("timed_out")),
                "no_process_sigkilled": "SIGKILL-last-resort" not in escalations,
                "arm_matches_the_registered_selection_rule": True,
                "graph_emitted": True,
            },
            "training_performed": False,
            "gate_registered": False,
            "adoption_claimed": ADOPTION_CLAIMED,
        },
        immutable=True,
    )


# --------------------------------------------------------------------------- #
# node 2 — uniform recall, the tripwire, the loss arrays, R0216's fuzzy law
# --------------------------------------------------------------------------- #
def run_fuzzy(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest = active["manifest"]
    arm = _verified_arm(job)
    build_path, build_signature = _intra_queue_signature(
        dict(job["build_reference"]), label="R0229 spill-lifted build"
    )
    build = prompt_contract.read_sealed(build_path, label="R0229 spill-lifted build")
    ids_path, ids_signature = _intra_queue_signature(
        dict(build["neighbour_ids"]), label="R0229 spill-lifted neighbour ids"
    )

    substrate_path = prompt_contract.verify_signature(
        dict(job["substrate_signature"]), label="R0216 substrate"
    )
    truth, truth_signature = _sealed(job, "truth_signature", label="R0220 truth")
    truth_ids_path = prompt_contract.verify_signature(
        dict(truth["outputs"]["ids"]), label="R0220 truth ids"
    )
    truth_cos_path = prompt_contract.verify_signature(
        dict(truth["outputs"]["cosines"]), label="R0220 truth cosines"
    )

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0229 spill-lifted fuzzy graph"
    )
    started = time.monotonic()

    raw = np.load(ids_path, allow_pickle=False)
    if raw.shape != (ROWS, GRAPH_K):
        raise Round0229Error(f"R0229 neighbour ids are {raw.shape}")
    leading = np.ascontiguousarray(raw.astype(np.int32))
    del raw
    as_int = leading.astype(np.int64)
    if int(as_int.min()) < 0 or int(as_int.max()) >= ROWS:
        raise Round0229Error("R0229 graph carries out-of-range neighbour ids")
    del as_int

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    host = np.load(substrate_path, mmap_mode="r", allow_pickle=False)
    if host.shape != (ROWS, DIMENSION) or host.dtype != np.float32:
        raise Round0229Error("R0229 sealed substrate geometry changed")
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
        raise Round0229Error("R0220 truth arrays have the wrong shape")
    kth = truth_cos[:, GRAPH_K - 1].astype(np.float64)

    strict = strict_containment_rows(leading, truth_ids)
    tie = tie_aware_rows(candidate_cos.astype(np.float64), leading, kth)
    strict_summary = summarize(strict, label="R0229 spill-lifted strict recall@15")
    tie_summary = summarize(tie, label="R0229 spill-lifted tie-aware recall@15")

    ceiling = float(arm["strict_ceiling_all_rows"])
    measured_strict = float(strict_summary["mean"])
    if measured_strict > ceiling + 1e-6:
        raise Round0229Error(
            f"R0229 spill-lifted strict recall {measured_strict} exceeds its own "
            f"measured structural ceiling {ceiling}; the ceiling instrument is wrong"
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
        raise Round0229Error(
            f"R0229 found a cosine distance of {most_negative!r}, below the "
            f"registered {MIN_ADMISSIBLE_NEGATIVE_DISTANCE} floor"
        )
    np.maximum(dists, 0.0, out=dists)
    if not np.isfinite(dists).all():
        raise Round0229Error("R0229 candidate distances are not finite")
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
        raise Round0229Error("R0229 fuzzy weights are invalid")
    if np.any(np.diff(src) < 0):
        raise Round0229Error("R0229 fuzzy edge sources are not sorted")
    degree_counts = np.bincount(src, minlength=ROWS)
    degrees = {
        "zero_degree_rows": int((degree_counts == 0).sum()),
        "min": int(degree_counts.min()),
        "median": float(np.median(degree_counts)),
        "mean": float(degree_counts.mean()),
        "max": int(degree_counts.max()),
    }
    if degrees["zero_degree_rows"] != 0:
        raise Round0229Error(
            f"R0229 R0215 tripwire: {degrees['zero_degree_rows']} zero-degree rows"
        )

    ids_out = atomic_save_new_npy(
        os.path.join(output, "spill-lifted-k15-ids.i32.npy"), ids_sorted,
        immutable=True,
    )
    graph_path = atomic_save_new_npz(
        os.path.join(output, "edges-k15-fuzzy.npz"), immutable=True,
        compressed=False, sources=src, targets=dst, weights=wts,
        n_nodes=np.asarray(ROWS, dtype=np.int64),
        k=np.asarray(GRAPH_K, dtype=np.int64),
    )
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0229Error(f"R0229 fuzzy peak RSS {peak_rss_gib:.2f} GiB")

    receipt = prompt_contract.seal({
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": GRAPH_CAPABILITY,
        "capabilities": [GRAPH_CAPABILITY],
        "arm": arm,
        "clusters": int(arm["clusters"]),
        "spill": int(arm["spill"]),
        "rows": ROWS,
        "dimension": DIMENSION,
        "k": GRAPH_K,
        "builder": {
            "name": "cluster-spill-nnd",
            "source_rounds": ["0226", "0227", "0229"],
            "neighbour_ids": ids_signature,
            "build_receipt": build_signature,
            "nn_descent": dict(arm["nn_descent"]),
            "approximate": True,
        },
        "recall_against_r0220_exact_truth": {
            "truth_receipt": truth_signature,
            "rows_measured": ROWS,
            "population": RECALL_POPULATION,
            "population_note": RECALL_POPULATION_NOTE,
            "tie_aware": tie_summary,
            "strict": strict_summary,
            "tie_tolerance": TIE_TOLERANCE,
            "structural_ceiling_strict_all_rows": ceiling,
            "gap_to_structural_ceiling": ceiling - measured_strict,
            "density_decile_tie_aware": decile_recall,
            "sparsest_decile_mean": decile_recall[0],
            "densest_decile_mean": decile_recall[-1],
            "sparsest_to_densest_gap": decile_recall[-1] - decile_recall[0],
            "rows_carrying_any_loss": rows_carrying_loss,
            "rows_carrying_any_loss_fraction": rows_carrying_loss / ROWS,
            "total_missing_true_edges": int(lost_edges_per_row.sum()),
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
        "spill_lifted_k15_ids": expected_input_signature(ids_out),
        "graph": expected_input_signature(graph_path),
        "graph_checks": {
            "r0215_tripwire_clean": True,
            "zero_degree_rows": degrees["zero_degree_rows"],
            "self_loops": int(structural.get("self_loops", 0)),
            "duplicate_entries": int(structural.get("duplicate_entries", 0)),
            "out_of_range": int(structural.get("out_of_range", 0)),
            "rows_below_k": int(structural.get("rows_below_k", 0)),
            "recall_does_not_exceed_its_own_ceiling": True,
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
        os.path.join(output, "spill-lifted-graph.json"), receipt, immutable=True
    )
    del ids_sorted, dists, src, dst, wts
    gc.collect()


# --------------------------------------------------------------------------- #
# node 3 — one train cell, R0217's treatment with the graph swapped
# --------------------------------------------------------------------------- #
def _train_graph(job: Mapping[str, Any]) -> dict[str, Any]:
    """The sealed fuzzy graph this cell trains on, bound by its real bytes."""
    manifest_path, manifest_signature = _intra_queue_signature(
        dict(job["graph_manifest_reference"]), label="R0229 spill-lifted graph"
    )
    graph_manifest = prompt_contract.read_sealed(
        manifest_path, label="R0229 spill-lifted graph"
    )
    edges_path, graph_signature = _intra_queue_signature(
        dict(graph_manifest["graph"]), label="R0229 spill-lifted fuzzy edges"
    )
    return {
        "manifest": graph_manifest,
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "edges_path": edges_path,
        "directed_edges": int(graph_manifest["directed_edge_count"]),
    }


def _open_substrate(graph_manifest: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """R0216's sealed substrate as a memmap, never a resident array."""
    substrate_signature = dict(graph_manifest["substrate"])
    path = prompt_contract.verify_signature(
        substrate_signature, label="R0216 substrate"
    )
    return np.load(path, mmap_mode="r", allow_pickle=False), substrate_signature


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    seed = int(job["training_seed"])
    if seed not in SEEDS:
        raise Round0229Error(f"R0229 seed {seed} is not registered")
    if str(job.get("capability")) != map_capability(seed):
        raise Round0229Error("R0229 train job capability does not match its seed")

    bundle = _train_graph(job)
    graph_manifest = bundle["manifest"]
    manifest_signature = bundle["manifest_signature"]
    graph_signature = bundle["signature"]
    graph_path = bundle["edges_path"]
    edges = int(bundle["directed_edges"])
    updates = successful_updates_for_edges(edges)
    dose = validate_dose(updates=updates, edge_count=edges)

    source, substrate_signature = _open_substrate(graph_manifest)

    clusters = int(graph_manifest["clusters"])
    spill = int(graph_manifest["spill"])
    nn_descent = dict(graph_manifest["builder"]["nn_descent"])
    config, config_sha, invariant = train_config(
        clusters=clusters,
        spill=spill,
        nn_descent=nn_descent,
        seed=seed,
        graph_signature=graph_signature,
        graph_manifest_signature=manifest_signature,
        substrate_signature=substrate_signature,
        r0216_graph_signature=dict(job["r0216_graph_signature"]),
        r0216_graph_manifest_signature=dict(job["r0216_graph_manifest_signature"]),
        graph_edges=edges,
        rows=ROWS,
    )
    if invariant != TREATMENT_INVARIANT_SHA256:
        raise Round0229Error(
            "R0229 cell config is not R0217's treatment outside the seed and the "
            f"graph: {invariant} != {TREATMENT_INVARIANT_SHA256}"
        )

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0229 train output"
    )
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "treatment_config_round_id": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "arm": ARM_NAME,
            "clusters": clusters,
            "seed": seed,
            "capability": map_capability(seed),
            "treatment_invariant_sha256": invariant,
            "config": config,
            "config_sha256": config_sha,
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
        resample_negatives=False, precomputed_edges_path=graph_path,
        use_wandb=False,
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
        raise Round0229Error(f"R0229 train accounting failed: {mismatches}")
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
        raise Round0229Error("R0229 train performance admission failed")

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
        raise Round0229Error(f"R0229 train peak RSS {peak_rss_gib:.2f} GiB")
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
        "graph_recall": dict(
            graph_manifest["recall_against_r0220_exact_truth"]
        ),
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
# node 4 — the geometry, and both registered inference rules
# --------------------------------------------------------------------------- #
def _load_coordinates(signature: Mapping[str, Any], *, label: str) -> np.ndarray:
    path = prompt_contract.verify_signature(dict(signature), label=label)
    array = np.load(path, allow_pickle=False)
    if array.shape != (ROWS, 2):
        raise Round0229Error(f"{label} has shape {array.shape}, expected ({ROWS}, 2)")
    return np.asarray(array, dtype=np.float32)


def run_geometry(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    arm = _verified_arm(job)

    graph_path, graph_manifest_signature = _intra_queue_signature(
        dict(job["graph_manifest_reference"]), label="R0229 spill-lifted graph"
    )
    graph_manifest = prompt_contract.read_sealed(
        graph_path, label="R0229 spill-lifted graph"
    )
    lost_path = prompt_contract.verify_signature(
        dict(graph_manifest["loss_arrays"]["lost_edges_per_row"]),
        label="R0229 lost-edge array",
    )
    lost = np.load(lost_path, allow_pickle=False)
    if lost.shape != (ROWS,):
        raise Round0229Error("R0229 lost-edge array has the wrong shape")

    truth, truth_signature = _sealed(job, "truth_signature", label="R0220 truth")
    truth_ids = np.load(
        prompt_contract.verify_signature(
            dict(truth["outputs"]["ids"]), label="R0220 truth ids"
        ),
        mmap_mode="r", allow_pickle=False,
    )
    truth_cos = np.load(
        prompt_contract.verify_signature(
            dict(truth["outputs"]["cosines"]), label="R0220 truth cosines"
        ),
        mmap_mode="r", allow_pickle=False,
    )
    kth_cosine = np.asarray(truth_cos[:, GRAPH_K - 1], dtype=np.float64)

    r0228_geometry, r0228_signature = _sealed(
        job, "r0228_geometry_signature", label="R0228 geometry"
    )
    bound = verify_r0228_displacement(r0228_geometry)

    # Row sets: this arm's own lost rows, and a control matched on deciles of the
    # row's own true 15th cosine. R0228's function, R0228's constants.
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
        path_signature, _ = _intra_queue_signature(
            dict(entry["signature"]), label=f"R0229 {name} coordinates"
        )
        coordinates = np.load(path_signature, allow_pickle=False)
        if coordinates.shape != (ROWS, 2):
            raise Round0229Error(f"R0229 {name} coordinates have the wrong shape")
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
        coordinates = _load_coordinates(entry["signature"], label=f"R0229 {name}")
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
    per_arm["multiplicity"] = {
        "new_arms_tested_here": 1,
        "correction_applied": "none",
        "note": (
            "one new arm means one new test at alpha = 0.05; the registered "
            "Holm-Bonferroni machinery is inert at a single test. Its smallest "
            "attainable p is published beside it, so the test is shown to be "
            "capable of rejecting before its result is read (review-0228-01)."
        ),
    }

    # The registered three-arm DiD trend test (review-0228-01 recommendation #8).
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
            "outcome": "does-a-spill-lifted-100m-feasible-graph-move-the-map-at-2m",
            "arm": arm,
            "arm_recall": dict(
                graph_manifest["recall_against_r0220_exact_truth"]
            ),
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
            "trend_arms": list(TREND_ARMS),
            "decision_rule_note": DECISION_RULE_NOTE,
            "resolution_rule_note": RESOLUTION_RULE_NOTE,
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


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0229Error("R0229 phase-2 handler received another queue")
    action = str(job.get("action") or "")
    if action == BUILD_ACTION:
        run_build(active, job)
    elif action == FUZZY_ACTION:
        run_fuzzy(active, job)
    elif action == TRAIN_ACTION:
        run_train(active, job)
    elif action == GEOMETRY_ACTION:
        run_geometry(active, job)
    else:
        raise Round0229Error(f"unknown R0229 phase-2 action {action!r}")


__all__ = [
    "BUILD_ACTION",
    "FUZZY_ACTION",
    "GEOMETRY_ACTION",
    "TRAIN_ACTION",
    "run_build",
    "run_fuzzy",
    "run_geometry",
    "run_job",
    "run_train",
]
