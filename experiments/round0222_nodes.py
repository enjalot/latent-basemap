"""Execute R0222: score seeds 46-49 on R0218's frozen panel, then gate at n=8.

One node, and deliberately so. The eight cells are only poolable if seeds 46-49
are scored against the **byte-identical** high-D reference that seeds 42-45 were
scored against, so this node does not rebuild the reference — it loads R0218's
published `minilm-2m-high-d-reference.npz`, checks its file signature against
R0218's sealed value, re-derives the content-addressed key from the substrate,
anchors, config and centroids, and requires all three to agree. If any of them
does not, the node aborts with `REFERENCE_MISMATCH_MESSAGE`: the eight cells are
not comparable and no gate may be registered from them.

Everything else about the low-D scoring path is R0218's, imported rather than
re-implemented: the sealed-substrate binding, the ordered-array authentication,
the provenance corpus labels, the reference convention and `score_panel` itself.
The purity vocabularies are R0218's published centroid arrays, loaded from disk,
so the k-means is not re-run and cannot drift.

The gate arithmetic then runs on the pooled eight cells — R0218's four read out
of its sealed receipt, R0221's four scored here — over the accepted six-metric
set restricted to what this panel computes, `density_v2` included.
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
    CAPABILITY as PANEL_CAPABILITY,
    CENTROID_KS,
    CORPUS_SLUGS,
    DIMENSION,
    HOST_RSS_LIMIT_GIB,
    PANEL_METRICS,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS as R0218_SEEDS,
    corpus_ffr_view,
    panel_execution_ok,
    panel_metric_view,
)
from basemap.round0221_minilm_2m_seed_extension import (
    GRAPH_CAPABILITY,
    POOLED_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    SEEDS as R0221_SEEDS,
    TRAIN_SCHEMA as R0221_TRAIN_SCHEMA,
    capability_for_seed as r0221_capability_for_seed,
)
from basemap.round0222_minilm_2m_gate_n8 import (
    CAPABILITY,
    GATE_METRICS,
    PANEL_EXTENSION_CAPABILITY,
    PANEL_SCHEMA,
    ROUND_ID,
    Round0222Error,
    UNAVAILABLE_METRICS,
    register_minilm_gates_n8,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0218_nodes


ACTION = "register_minilm_mixed_2m_quality_gates_n8"

REFERENCE_MISMATCH_MESSAGE = (
    "R0222 high-D reference is not byte-identical to R0218's. The eight cells "
    "are NOT comparable and no n=8 gate may be registered from them."
)


def _sealed_panel(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind R0218's sealed four-cell panel: the reference and four of the cells."""
    panel_path = str(job["panel_evidence"])
    signature = expected_input_signature(panel_path)
    panel = prompt_contract.read_sealed(
        panel_path, label="R0218 MiniLM 2M four-seed panel"
    )
    checks = panel.get("execution_checks") or {}
    if (
        panel.get("schema") != PANEL_SCHEMA
        or panel.get("round_id") != "0218"
        or panel.get("capabilities") != [PANEL_CAPABILITY]
        or panel.get("seeds") != list(R0218_SEEDS)
        or int(panel.get("n", -1)) != len(R0218_SEEDS)
        or panel.get("evaluation_performed") is not True
        or panel.get("gate_registered") is not False
        or panel.get("metrics") != list(PANEL_METRICS)
        or panel.get("seed_invariant_sha256") != R0217_SEED_INVARIANT_SHA256
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise Round0222Error("R0218 panel receipt contract changed")
    return panel, signature


def _load_reference(panel: Mapping[str, Any], reference_identity: Mapping[str, Any]):
    """Load R0218's published reference and prove it is byte-identical."""
    from basemap.panel_v2 import hiD_reference_key, load_hiD_reference

    signature = dict(panel["shared_high_d_reference"])
    path = prompt_contract.verify_signature(
        signature, label="R0218 shared high-D reference"
    )
    observed = expected_input_signature(path)
    if observed != signature:
        raise Round0222Error(f"{REFERENCE_MISMATCH_MESSAGE} file signature drift")
    reference = load_hiD_reference(path)
    if (
        str(reference["key"]) != str(panel["high_d_reference_key"])
        or str(reference["content_sha256"])
        != str(panel["high_d_reference_content_sha256"])
    ):
        raise Round0222Error(f"{REFERENCE_MISMATCH_MESSAGE} content key drift")
    return reference, signature, hiD_reference_key


def _load_centroids(panel: Mapping[str, Any]) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """R0218's published purity vocabularies, loaded rather than recomputed."""
    declared = dict(panel.get("centroids") or {})
    if set(declared) != {str(k) for k in CENTROID_KS}:
        raise Round0222Error("R0218 centroid vocabularies changed")
    centroids: dict[int, np.ndarray] = {}
    signatures: dict[str, Any] = {}
    for k in CENTROID_KS:
        signature = dict(declared[str(k)])
        path = prompt_contract.verify_signature(
            signature, label=f"R0218 purity centroids k{k}"
        )
        array = np.load(path, allow_pickle=False)
        if array.shape != (k, DIMENSION) or array.dtype != np.dtype("float32"):
            raise Round0222Error(f"R0218 centroids k{k} geometry changed")
        centroids[k] = array
        signatures[str(k)] = signature
    return centroids, signatures


def _authenticate_r0221_map(
    cell: Mapping[str, Any], sealed: Mapping[str, Any]
) -> tuple[int, dict[str, Any], dict[str, Any], str]:
    """Bind one R0221 map to the exact substrate R0218's panel scored."""
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in R0221_SEEDS:
        raise Round0222Error(f"R0222 cell seed {seed!r} is not an R0221 cell")
    capability = r0221_capability_for_seed(seed)
    if str(cell.get("capability") or "") != capability:
        raise Round0222Error(f"R0222 seed-{seed} cell capability changed")
    receipt_signature = dict(cell["train_receipt"])
    receipt_path = prompt_contract.verify_signature(
        receipt_signature, label=f"R0221 seed-{seed} train receipt"
    )
    receipt = prompt_contract.read_sealed(
        receipt_path, label=f"R0221 seed-{seed} train receipt"
    )
    train_checks = receipt.get("train_checks") or {}
    if (
        receipt.get("schema") != R0221_TRAIN_SCHEMA
        or receipt.get("round_id") != "0221"
        or receipt.get("treatment_config_round_id") != "0217"
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != seed
        or receipt.get("training_performed") is not True
        or receipt.get("gate_registerable_here") is not False
        or receipt.get("map_decision_made") is not False
        or int(receipt.get("rows", -1)) != ROWS
        or int(receipt.get("dimension", -1)) != DIMENSION
        or int(receipt.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
        or receipt.get("graph_capability") != GRAPH_CAPABILITY
        or str(receipt.get("seed_invariant_sha256") or "")
        != R0217_SEED_INVARIANT_SHA256
        or not train_checks
        or not all(bool(value) for value in train_checks.values())
    ):
        raise Round0222Error(f"R0221 seed-{seed} train receipt contract changed")
    if (
        dict(receipt.get("substrate") or {}) != dict(sealed["substrate_signature"])
        or dict(receipt.get("graph_manifest") or {})
        != dict(sealed["manifest_signature"])
    ):
        raise Round0222Error(
            f"R0221 seed-{seed} was not trained on the substrate this panel scores"
        )
    model_path = prompt_contract.verify_signature(
        receipt["model"], label=f"R0221 seed-{seed} published map"
    )
    return seed, receipt, receipt_signature, model_path


def _precedent_gates(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read R0161's and R0193's sealed gate artifacts, unmodified."""
    artifacts: dict[str, Any] = {}
    signatures: dict[str, Any] = {}
    declared = dict(job["precedent_gate_signatures"])
    for round_id in sorted(declared):
        signature = dict(declared[round_id])
        path = prompt_contract.verify_signature(
            signature, label=f"R{round_id} sealed quality-gate artifact"
        )
        artifacts[round_id] = prompt_contract.read_sealed(
            path, label=f"R{round_id} sealed quality-gate artifact"
        )
        signatures[round_id] = signature
    return artifacts, signatures


def run_registration(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch
    from basemap.panel_v2 import reset_process_cuda_peak, sample_anchors, score_panel
    from basemap.pumap.parametric_umap import ParametricUMAP

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0222Error("R0222 handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0222Error("R0222 scoring requires CUDA")

    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    corpus_of_row = round0218_nodes._corpus_of_row(sealed)
    panel_evidence, panel_signature = _sealed_panel(job)

    cells_in = job.get("cells")
    if not isinstance(cells_in, list) or {
        int(cell.get("seed", -1)) for cell in cells_in
    } != set(R0221_SEEDS):
        raise Round0222Error("R0222 four-cell input matrix changed")
    authenticated = {}
    for cell in cells_in:
        seed, receipt, receipt_signature, model_path = _authenticate_r0221_map(
            cell, sealed
        )
        authenticated[seed] = {
            "receipt": receipt,
            "receipt_signature": receipt_signature,
            "model_path": model_path,
        }
    invariants = {
        str(authenticated[seed]["receipt"]["seed_invariant_sha256"])
        for seed in R0221_SEEDS
    } | {R0217_SEED_INVARIANT_SHA256}
    if len(invariants) != 1:
        raise Round0222Error(
            "R0222 pooled family is not commensurate: R0221's cells do not "
            "carry R0217's seed-invariant config digest"
        )
    r0221_model_hashes = {
        str(authenticated[seed]["receipt"]["model"]["sha256"]) for seed in R0221_SEEDS
    }
    r0217_model_hashes = {
        str(panel_evidence["cells"][str(seed)]["model"]["sha256"])
        for seed in R0218_SEEDS
    }
    if len(r0221_model_hashes | r0217_model_hashes) != len(POOLED_SEEDS):
        raise Round0222Error("R0222 pooled family contains a duplicated checkpoint")

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0222 n=8 gate registration"
    )
    started = time.monotonic()
    reset_process_cuda_peak()

    cfg = prompt_contract.panel_config()
    centroids, centroid_signatures = _load_centroids(panel_evidence)
    data_identity = {
        "kind": "ordered_array",
        "shape": [ROWS, DIMENSION],
        "dtype": np.dtype("<f4").str,
        "sha256": sealed["ordered_substrate_sha256"],
    }
    reference_identity = {
        "data_identity": data_identity,
        "convention": dict(round0218_nodes.REFERENCE_CONVENTION),
    }
    reference, reference_signature, reference_key_fn = _load_reference(
        panel_evidence, reference_identity
    )
    anchors = sample_anchors(ROWS, cfg)
    if not np.array_equal(
        np.asarray(anchors, dtype=np.int64),
        np.asarray(reference["anchor_ids"], dtype=np.int64),
    ):
        raise Round0222Error(f"{REFERENCE_MISMATCH_MESSAGE} anchor drift")
    rederived_key, _parts = reference_key_fn(
        source, anchors, cfg, centroids, kf=int(reference["kf"]), **reference_identity
    )
    if str(rederived_key) != str(reference["key"]):
        raise Round0222Error(f"{REFERENCE_MISMATCH_MESSAGE} re-derived key drift")
    anchor_labels = round0218_nodes._anchor_corpus_labels(corpus_of_row, anchors)
    anchor_corpus_counts = {
        slug: int((anchor_labels == slug).sum()) for slug in CORPUS_SLUGS
    }
    if anchor_corpus_counts != dict(panel_evidence["anchor_corpus_counts"]):
        raise Round0222Error(f"{REFERENCE_MISMATCH_MESSAGE} anchor corpus drift")

    cells: dict[int, dict[str, Any]] = {}
    for seed in R0221_SEEDS:
        entry = authenticated[seed]
        model = ParametricUMAP.load(entry["model_path"], device="cuda")
        coordinates = np.asarray(
            model.transform(source, batch_size=8192), dtype=np.float32
        )
        if coordinates.shape != (ROWS, 2):
            raise Round0222Error(
                f"R0222 seed-{seed} transform produced {coordinates.shape}, "
                f"expected ({ROWS}, 2)"
            )
        if not np.isfinite(coordinates).all():
            raise Round0222Error(
                f"R0222 seed-{seed} transform over {ROWS} rows is not finite"
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
                "capability": r0221_capability_for_seed(seed),
                "universe": "R0216-queue-correction-3-minilm-mixed-2m",
                "substrate": dict(sealed["substrate_signature"]),
                "provenance_array": dict(sealed["provenance_signature"]),
                "train_receipt": dict(entry["receipt_signature"]),
                "coordinates": coordinates_signature,
                "shared_high_d_reference": reference_signature,
                "reference_source_round": "0218",
            },
        )
        if not panel_execution_ok(panel):
            raise Round0222Error(f"R0222 seed-{seed} panel is collapsed or nonfinite")
        if not bool(panel["provenance"]["hiD_reference_reused"]):
            raise Round0222Error(f"{REFERENCE_MISMATCH_MESSAGE} seed-{seed} recomputed")
        cells[seed] = {
            "seed": seed,
            "capability": r0221_capability_for_seed(seed),
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

    #: The high-D side of every ratio must be literally the same array for all
    #: eight cells; R0218 published its four, so compare rather than assume.
    for seed in R0221_SEEDS:
        mine = cells[seed]["panel"]["purity_numerators"]
        theirs = panel_evidence["cells"][str(R0218_SEEDS[0])]["panel"][
            "purity_numerators"
        ]
        for k in ("k256", "k1024"):
            if float(mine[k]["hi_D_agreement"]) != float(theirs[k]["hi_D_agreement"]):
                raise Round0222Error(
                    f"{REFERENCE_MISMATCH_MESSAGE} seed-{seed} hi-D agreement {k}"
                )

    pooled_cells: dict[str, dict[str, float]] = {}
    pooled_corpus: dict[str, dict[str, dict[str, float]]] = {}
    for seed in R0218_SEEDS:
        pooled_cells[str(seed)] = {
            key: float(value)
            for key, value in panel_evidence["panel_metric_cells"][str(seed)].items()
        }
        pooled_corpus[str(seed)] = {
            slug: {
                "anchors": int(
                    panel_evidence["corpus_ffr_cells"][str(seed)][slug]["anchors"]
                ),
                "ffr": float(
                    panel_evidence["corpus_ffr_cells"][str(seed)][slug]["ffr"]
                ),
            }
            for slug in CORPUS_SLUGS
        }
    for seed in R0221_SEEDS:
        pooled_cells[str(seed)] = dict(cells[seed]["panel_metrics"])
        pooled_corpus[str(seed)] = dict(cells[seed]["corpus_ffr"])

    precedent_artifacts, precedent_signatures = _precedent_gates(job)
    registration = register_minilm_gates_n8(
        pooled_cells=pooled_cells,
        corpus_cells=pooled_corpus,
        precedents=precedent_artifacts,
    )

    execution_checks = {
        "all_four_new_cells_scored": set(cells) == set(R0221_SEEDS),
        "eight_pooled_cells": len(pooled_cells) == len(POOLED_SEEDS),
        "every_metric_finite": all(
            math.isfinite(float(value))
            for cell in pooled_cells.values()
            for value in cell.values()
        ),
        "no_collapsed_panel": all(
            bool(cell["panel_finite_noncollapsed"]) for cell in cells.values()
        ),
        "map_transform_finite_over_all_rows": all(
            int(cell["transform_rows_finite"]) == ROWS for cell in cells.values()
        ),
        "per_corpus_ffr_slices_complete": all(
            set(pooled_corpus[str(seed)]) == set(CORPUS_SLUGS)
            for seed in POOLED_SEEDS
        ),
        "pooled_family_one_seed_invariant_digest": len(invariants) == 1,
        "eight_distinct_checkpoints": (
            len(r0221_model_hashes | r0217_model_hashes) == len(POOLED_SEEDS)
        ),
        "reference_byte_identical_to_r0218": True,
        "shared_reference_reused_by_content_key": all(
            bool(cell["panel"]["provenance"]["hiD_reference_reused"])
            for cell in cells.values()
        ),
        "density_v2_is_gated": "density_v2" in GATE_METRICS,
        "no_computed_metric_excluded_by_judgement": not registration[
            "excluded_by_judgement"
        ],
        "precedent_density_floors_read_from_sealed_artifacts": bool(
            registration["r0219_retraction"]["density_v2_gated_in_both_precedents"]
        ),
    }
    if not all(execution_checks.values()):
        raise Round0222Error(f"R0222 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0222Error(
            f"R0222 peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )

    receipt = prompt_contract.seal({
        **registration,
        "capabilities": [PANEL_EXTENSION_CAPABILITY, CAPABILITY],
        "release_sha": active["manifest"]["release_sha"],
        "panel_evidence": panel_signature,
        "panel_capability": PANEL_CAPABILITY,
        "panel_release_sha": panel_evidence.get("release_sha"),
        "precedent_gate_artifacts": precedent_signatures,
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
            "centroid_source": "R0218 published frozen centroid arrays, loaded",
        },
        "shared_high_d_reference": reference_signature,
        "high_d_reference_key": str(reference["key"]),
        "high_d_reference_content_sha256": str(reference["content_sha256"]),
        "reference_convention": dict(round0218_nodes.REFERENCE_CONVENTION),
        "reference_source_round": "0218",
        "reference_byte_identical_to_r0218": True,
        "centroids": centroid_signatures,
        "anchor_corpus_counts": anchor_corpus_counts,
        "new_cells": {str(seed): cells[seed] for seed in R0221_SEEDS},
        "pooled_panel_metric_cells": pooled_cells,
        "pooled_corpus_ffr_cells": pooled_corpus,
        "unavailable_metrics": dict(UNAVAILABLE_METRICS),
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "gate_registered": True,
        "map_decision_made": False,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "decision": {
            "outcome": "minilm-mixed-2m-quality-gates-registered-at-n8",
            "gated_metrics": list(GATE_METRICS),
            "unavailable_metrics": sorted(UNAVAILABLE_METRICS),
            "excluded_by_judgement": [],
            "applies_to": (
                "future byte-commensurate maps of the R0216 queue-correction-3 "
                "mixed MiniLM 2M universe under the registered R0217 recipe"
            ),
            "does_not_apply_to": (
                "jina universes, differently composed or differently sized "
                "MiniLM universes, PQ-derived graphs, or any map scored on a "
                "different panel configuration"
            ),
        },
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "minilm-quality-gates-n8.json"), receipt, immutable=True
    )
    del source, corpus_of_row, reference, centroids
    gc.collect()


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != ACTION:
        raise Round0222Error("R0222 authorizes only the n=8 gate registration")
    run_registration(active, job)


__all__ = ["ACTION", "REFERENCE_MISMATCH_MESSAGE", "run_job", "run_registration"]
