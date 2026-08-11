"""Execute the R0218 MiniLM 2M four-seed panel.

One node, four scored cells. The reason it is one node rather than four is the
shared high-D reference: the exact top-`k_hit` truth, the approximate `k_frac`
membership, the exact density radii and the centroid labels are all
**map-independent**, so they are computed once over the sealed R0216 substrate
and reused — by content key, re-verified inside every `score_panel` call — for
all four maps. Four separate nodes would either recompute that reference four
times or pass it between processes without the verification that makes reuse
safe.

Nothing about the evaluator is re-implemented here. `basemap/panel_v2.py` does
the scoring, `experiments/score_complete_panel.frozen_centroids` builds the
purity vocabularies, and the panel configuration is the accepted R0113 one.

The round makes no quality claim and registers no gate; the checks below are
execution checks. Every one of them aborts the node rather than emitting a
degraded receipt.
"""
from __future__ import annotations

import gc
import math
import os
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0218_minilm_2m_panel import (
    CAPABILITY,
    CENTROID_ITERS,
    CENTROID_KS,
    CENTROID_SEED,
    CORPORA,
    CORPUS_ROWS,
    CORPUS_SLUGS,
    DIMENSION,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    HOST_RSS_LIMIT_GIB,
    MAP_TRAIN_SCHEMA,
    REFERENCE_NOTE,
    ROUND_ID,
    ROWS,
    Round0218Error,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    build_family_panel_evidence,
    corpus_ffr_view,
    map_capability,
    panel_execution_ok,
    panel_metric_view,
)
from basemap import round0113_prompt_contrast as prompt_contract


ACTION = "score_minilm_mixed_2m_seed_family_panel"

#: The row-order / distance / anchor conventions this universe's high-D
#: reference is bound to. Any drift changes the content key and fails closed.
REFERENCE_CONVENTION = {
    "row_order": "R0216 queue-correction-3 mixed 2M substrate order",
    "distance": "cosine via fp32 L2-normalized squared L2",
    "self_exclusion": True,
    "anchor_namespace": "R0216 substrate row IDs",
    "embedding_prompt": "none (all-MiniLM-L6-v2 raw sentence embeddings)",
}


def _sealed_substrate(job: Mapping[str, Any]) -> dict[str, Any]:
    """Bind R0216's sealed receipt and prove it is the corrected, exact graph."""
    manifest_signature = dict(job["graph_manifest_signature"])
    manifest_path = prompt_contract.verify_signature(
        manifest_signature, label="R0218 sealed R0216 substrate+graph receipt"
    )
    manifest = prompt_contract.read_sealed(
        manifest_path, label="R0218 sealed R0216 substrate+graph receipt"
    )
    checks = manifest.get("graph_checks") or {}
    degrees = manifest.get("degrees") or {}
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != GRAPH_SOURCE_ROUND_ID
        or manifest.get("capability") != GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or manifest.get("training_performed") is not False
    ):
        raise Round0218Error("R0218 sealed R0216 substrate+graph contract changed")
    if (
        int(checks.get("zero_degree_rows", -1)) != 0
        or int(degrees.get("zero_degree_rows", -1)) != 0
    ):
        raise Round0218Error("R0218 requires the sealed zero-degree tripwire to hold")
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if edges != SEALED_DIRECTED_EDGES:
        raise Round0218Error(
            f"R0218 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    substrate_signature = dict(manifest["substrate"])
    provenance_signature = dict(manifest["provenance"])
    return {
        "manifest": manifest,
        "manifest_signature": manifest_signature,
        "substrate_signature": substrate_signature,
        "provenance_signature": provenance_signature,
        "directed_edges": edges,
        "ordered_substrate_sha256": str(manifest["ordered_substrate_sha256"]),
    }


def _open_substrate(sealed: Mapping[str, Any]) -> np.ndarray:
    """Serve the 3.07 GB substrate lazily and authenticate its ordered bytes."""
    path = prompt_contract.verify_signature(
        sealed["substrate_signature"], label="R0218 sealed R0216 substrate"
    )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (ROWS, DIMENSION) or array.dtype != np.float32:
        raise Round0218Error("R0218 sealed R0216 substrate geometry changed")
    observed = ordered_array_sha256(array)
    if observed != sealed["ordered_substrate_sha256"]:
        raise Round0218Error(
            "R0218 substrate ordered-array identity does not match the sealed "
            f"R0216 receipt: {observed} != {sealed['ordered_substrate_sha256']}"
        )
    return array


def _corpus_of_row(sealed: Mapping[str, Any]) -> np.ndarray:
    """Per-row corpus ids from R0216's provenance, checked against COMPOSITION."""
    path = prompt_contract.verify_signature(
        sealed["provenance_signature"], label="R0218 sealed R0216 provenance"
    )
    provenance = np.load(path, mmap_mode="r", allow_pickle=False)
    if provenance.shape != (ROWS,) or provenance.dtype.names != (
        "corpus",
        "shard",
        "row",
    ):
        raise Round0218Error("R0218 sealed R0216 provenance layout changed")
    corpus = np.asarray(provenance["corpus"], dtype=np.int64)
    counts = np.bincount(corpus, minlength=len(CORPORA))
    if len(counts) != len(CORPORA):
        raise Round0218Error("R0218 provenance carries an unregistered corpus id")
    for index, slug, _name, rows in CORPORA:
        if int(counts[index]) != int(rows):
            raise Round0218Error(
                f"R0218 corpus {slug} has {int(counts[index])} rows, registered "
                f"{int(rows)}"
            )
    del provenance
    return corpus


def _authenticate_map(
    cell: Mapping[str, Any], sealed: Mapping[str, Any]
) -> tuple[int, dict[str, Any], dict[str, Any], str]:
    """Bind one accepted R0217 map to the exact substrate this panel scores."""
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0218Error(f"R0218 cell seed {seed!r} is not a registered cell")
    capability = map_capability(seed)
    if str(cell.get("capability") or "") != capability:
        raise Round0218Error(f"R0218 seed-{seed} cell capability changed")
    receipt_signature = dict(cell["train_receipt"])
    receipt_path = prompt_contract.verify_signature(
        receipt_signature, label=f"R0217 seed-{seed} train receipt"
    )
    receipt = prompt_contract.read_sealed(
        receipt_path, label=f"R0217 seed-{seed} train receipt"
    )
    train_checks = receipt.get("train_checks") or {}
    if (
        receipt.get("schema") != MAP_TRAIN_SCHEMA
        or receipt.get("round_id") != "0217"
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != seed
        or receipt.get("training_performed") is not True
        or receipt.get("gate_registerable_here") is not False
        or receipt.get("map_decision_made") is not False
        or int(receipt.get("rows", -1)) != ROWS
        or int(receipt.get("dimension", -1)) != DIMENSION
        or int(receipt.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
        or receipt.get("graph_capability") != GRAPH_CAPABILITY
        or not train_checks
        or not all(bool(value) for value in train_checks.values())
    ):
        raise Round0218Error(f"R0217 seed-{seed} train receipt contract changed")
    if (
        dict(receipt.get("substrate") or {}) != dict(sealed["substrate_signature"])
        or dict(receipt.get("graph_manifest") or {})
        != dict(sealed["manifest_signature"])
    ):
        raise Round0218Error(
            f"R0217 seed-{seed} was not trained on the substrate this panel scores"
        )
    model_path = prompt_contract.verify_signature(
        receipt["model"], label=f"R0217 seed-{seed} published map"
    )
    return seed, receipt, receipt_signature, model_path


def _anchor_corpus_labels(
    corpus_of_row: np.ndarray, anchors: np.ndarray
) -> np.ndarray:
    labels = np.asarray(
        [CORPUS_SLUGS[int(value)] for value in corpus_of_row[anchors]], dtype="U16"
    )
    present = set(labels.tolist())
    if present != set(CORPUS_SLUGS):
        raise Round0218Error(
            f"R0218 anchor sample misses corpora {sorted(set(CORPUS_SLUGS) - present)}"
        )
    return labels


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch
    from basemap.panel_v2 import (
        build_hiD_reference,
        reset_process_cuda_peak,
        sample_anchors,
        save_hiD_reference,
        score_panel,
    )
    from basemap.pumap.parametric_umap import ParametricUMAP
    from experiments.score_complete_panel import frozen_centroids

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0218Error("R0218 panel handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0218Error("R0218 panel requires CUDA")

    sealed = _sealed_substrate(job)
    source = _open_substrate(sealed)
    corpus_of_row = _corpus_of_row(sealed)

    cells_in = job.get("cells")
    if not isinstance(cells_in, list) or {
        int(cell.get("seed", -1)) for cell in cells_in
    } != set(SEEDS):
        raise Round0218Error("R0218 four-cell input matrix changed")
    authenticated = {}
    for cell in cells_in:
        seed, receipt, receipt_signature, model_path = _authenticate_map(cell, sealed)
        authenticated[seed] = {
            "receipt": receipt,
            "receipt_signature": receipt_signature,
            "model_path": model_path,
        }
    invariants = {
        str(authenticated[seed]["receipt"]["seed_invariant_sha256"]) for seed in SEEDS
    }
    if len(invariants) != 1:
        raise Round0218Error(
            "R0218 scored family is not commensurate: the four R0217 cells carry "
            f"{len(invariants)} seed-invariant config digests"
        )
    model_hashes = {
        str(authenticated[seed]["receipt"]["model"]["sha256"]) for seed in SEEDS
    }
    if len(model_hashes) != len(SEEDS):
        raise Round0218Error("R0218 scored family contains a duplicated checkpoint")

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0218 MiniLM 2M seed-family panel"
    )
    started = time.monotonic()
    reset_process_cuda_peak()

    cfg = prompt_contract.panel_config()
    centroid_root = create_fresh_directory(
        os.path.join(output, "centroids"), label="R0218 purity centroids"
    )
    centroids = frozen_centroids(
        source, CENTROID_KS, centroid_root, seed=CENTROID_SEED, iters=CENTROID_ITERS
    )
    centroid_signatures = {
        str(k): expected_input_signature(
            os.path.join(centroid_root, f"centroids_k{k}.npy")
        )
        for k in CENTROID_KS
    }
    data_identity = {
        "kind": "ordered_array",
        "shape": [ROWS, DIMENSION],
        "dtype": np.dtype("<f4").str,
        "sha256": sealed["ordered_substrate_sha256"],
    }
    reference_identity = {
        "data_identity": data_identity,
        "convention": dict(REFERENCE_CONVENTION),
    }
    anchors = sample_anchors(ROWS, cfg)
    reference = build_hiD_reference(
        source, anchors, cfg, centroids, **reference_identity
    )
    reference_path = os.path.join(output, "minilm-2m-high-d-reference.npz")
    save_hiD_reference(reference, reference_path)
    reference_signature = expected_input_signature(reference_path)
    anchor_labels = _anchor_corpus_labels(corpus_of_row, anchors)
    anchor_corpus_counts = {
        slug: int((anchor_labels == slug).sum()) for slug in CORPUS_SLUGS
    }

    cells: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        entry = authenticated[seed]
        model = ParametricUMAP.load(entry["model_path"], device="cuda")
        coordinates = np.asarray(
            model.transform(source, batch_size=8192), dtype=np.float32
        )
        if coordinates.shape != (ROWS, 2):
            raise Round0218Error(
                f"R0218 seed-{seed} transform produced {coordinates.shape}, "
                f"expected ({ROWS}, 2)"
            )
        if not np.isfinite(coordinates).all():
            raise Round0218Error(
                f"R0218 seed-{seed} transform over {ROWS} rows is not finite"
            )
        coordinates_path = os.path.join(output, f"coordinates-seed{seed}.npy")
        atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
        coordinates_signature = expected_input_signature(coordinates_path)
        panel = score_panel(
            source,
            coordinates,
            config=cfg,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=reference_identity,
            ffr_group_labels=anchor_labels,
            scale_admission=None,
            provenance={
                "round_id": ROUND_ID,
                "seed": seed,
                "capability": map_capability(seed),
                "universe": "R0216-queue-correction-3-minilm-mixed-2m",
                "substrate": dict(sealed["substrate_signature"]),
                "provenance_array": dict(sealed["provenance_signature"]),
                "train_receipt": dict(entry["receipt_signature"]),
                "coordinates": coordinates_signature,
                "shared_high_d_reference": reference_signature,
            },
        )
        if not panel_execution_ok(panel):
            raise Round0218Error(f"R0218 seed-{seed} panel is collapsed or nonfinite")
        cells[seed] = {
            "seed": seed,
            "capability": map_capability(seed),
            "train_receipt": dict(entry["receipt_signature"]),
            "model": dict(entry["receipt"]["model"]),
            "coordinates": coordinates_signature,
            "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
            "panel": panel,
            "panel_metrics": panel_metric_view(panel),
            "corpus_ffr": corpus_ffr_view(panel),
            "panel_finite_noncollapsed": True,
            "transform_rows_finite": ROWS,
        }
        del model, coordinates
        torch.cuda.empty_cache()
        gc.collect()

    evidence = build_family_panel_evidence(cells)
    execution_checks = {
        "all_four_cells_scored": set(cells) == set(SEEDS),
        "every_metric_finite": all(
            math.isfinite(float(value))
            for cell in cells.values()
            for value in (
                *cell["panel_metrics"].values(),
                *(slice_["ffr"] for slice_ in cell["corpus_ffr"].values()),
            )
        ),
        "no_collapsed_panel": all(
            bool(cell["panel_finite_noncollapsed"]) for cell in cells.values()
        ),
        "map_transform_finite_over_all_rows": all(
            int(cell["transform_rows_finite"]) == ROWS for cell in cells.values()
        ),
        "per_corpus_ffr_slices_complete": all(
            set(cell["corpus_ffr"]) == set(CORPUS_SLUGS) for cell in cells.values()
        ),
        "family_commensurate_one_seed_invariant_digest": len(invariants) == 1,
        "four_distinct_checkpoints": len(model_hashes) == len(SEEDS),
        "shared_reference_reused_by_content_key": all(
            bool(cell["panel"]["provenance"]["hiD_reference_reused"])
            for cell in cells.values()
        ),
    }
    if not all(execution_checks.values()):
        raise Round0218Error(f"R0218 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0218Error(
            f"R0218 panel peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    receipt = prompt_contract.seal({
        **evidence,
        "capabilities": [CAPABILITY],
        "release_sha": active["manifest"]["release_sha"],
        "lineage": {
            "graph_manifest": dict(sealed["manifest_signature"]),
            "substrate": dict(sealed["substrate_signature"]),
            "provenance": dict(sealed["provenance_signature"]),
            "ordered_substrate_sha256": sealed["ordered_substrate_sha256"],
        },
        "panel_configuration": {
            "source": "accepted R0113 panel_config()",
            "frac": cfg.frac,
            "k_hit": cfg.k_hit,
            "k_density": cfg.k_density,
            "k_frac_effective": int(reference["kf"]),
            "n_anchors": int(len(anchors)),
            "anchor_seed": cfg.anchor_seed,
            "formula_version": cfg.formula_version,
            "centroid_ks": list(CENTROID_KS),
            "centroid_recipe": (
                f"GPU Lloyd k-means over all {ROWS} rows; seed {CENTROID_SEED}; "
                f"{CENTROID_ITERS} iterations"
            ),
        },
        "shared_high_d_reference": reference_signature,
        "high_d_reference_key": str(reference["key"]),
        "high_d_reference_content_sha256": str(reference["content_sha256"]),
        "reference_convention": dict(REFERENCE_CONVENTION),
        "reference_reuse_note": REFERENCE_NOTE,
        "centroids": centroid_signatures,
        "anchor_corpus_counts": anchor_corpus_counts,
        "corpus_rows": dict(CORPUS_ROWS),
        "seed_invariant_sha256": sorted(invariants)[0],
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "map_decision_made": False,
        "production_or_publishing": False,
        "performance": {
            "panel_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "seed-family-panel.json"), receipt, immutable=True
    )
    del source, corpus_of_row, reference, centroids
    gc.collect()


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != ACTION:
        raise Round0218Error("R0218 authorizes only the MiniLM 2M seed-family panel")
    run_panel(active, job)


__all__ = ["ACTION", "REFERENCE_CONVENTION", "run_job", "run_panel"]
