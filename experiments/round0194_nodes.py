"""Execute R0194's CPU-only three-seed Pile loss localization."""
from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import atomic_save_new_npz, atomic_write_new_json, create_fresh_directory
from basemap.panel_v2 import load_hiD_reference
from basemap.round0108_evaluation import seal
from basemap.round0113_prompt_contrast import read_sealed
from basemap.round0194_pile_loss_localization import (
    ANCHORS,
    CAPABILITY,
    CLUSTER_KS,
    K_FRAC,
    K_HIT,
    ROUND_ID,
    SEEDS,
    Round0194Error,
    per_anchor_ffr,
    synthesize,
)


ROWS = 1_988_104
DIMENSION = 768
PILE_START = 1_421_764
PILE_STOP = ROWS
PILE_ROWS = PILE_STOP - PILE_START
TREE_WORKERS = 12
TREE_OVERSELECT = 16
CENTROID_CHUNK_ROWS = 65_536


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0194Error(f"{label} bytes changed")
    return actual


def _read_evaluation(expected: Mapping[str, Any], *, seed: int, rung: str) -> dict[str, Any]:
    signature = _signature(expected, label=f"seed {seed} {rung} evaluation")
    value = read_sealed(signature["canonical_path"], label=f"R0194 seed {seed} {rung}")
    expected_schema = {
        42: "round0187-composition-nested-common-core-evaluation-v1",
        43: "round0188-composition-boundary-evaluation-v1",
        44: "round0189-composition-boundary-evaluation-v1",
    }[seed]
    expected_round = {42: "0187", 43: "0188", 44: "0189"}[seed]
    if (
        value.get("schema") != expected_schema
        or value.get("round_id") != expected_round
        or value.get("rung") != rung
        or int((value.get("corpus_panels") or {}).get("pile", {}).get("n", -1))
        != PILE_ROWS
        or (value.get("corpus_panels") or {}).get("pile", {}).get("k_frac") != K_FRAC
        or (value.get("corpus_panels") or {}).get("pile", {}).get("n_anchors") != ANCHORS
    ):
        raise Round0194Error(f"seed {seed} {rung} common-core contract changed")
    return value


def _exact_low_fraction(
    coordinates: np.ndarray, anchor_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(coordinates, dtype=np.float32)
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    if (
        values.shape != (PILE_ROWS, 2)
        or anchors.shape != (ANCHORS,)
        or not np.isfinite(values).all()
    ):
        raise Round0194Error("Pile coordinate geometry changed")
    tree = cKDTree(values, compact_nodes=True, balanced_tree=True)
    _distances, candidates = tree.query(
        values[anchors],
        k=K_FRAC + TREE_OVERSELECT + 1,
        eps=0.0,
        p=2,
        workers=TREE_WORKERS,
    )
    filtered = np.empty((ANCHORS, K_FRAC + TREE_OVERSELECT), dtype=np.int64)
    for row, anchor in enumerate(anchors):
        selected = candidates[row][candidates[row] != anchor]
        if len(selected) < K_FRAC + TREE_OVERSELECT:
            raise Round0194Error("cKDTree self exclusion did not close")
        filtered[row] = selected[: K_FRAC + TREE_OVERSELECT]
    query = values[anchors]
    candidate_values = values[filtered]
    squared = np.sum(
        (candidate_values - query[:, None, :]) ** 2,
        axis=2,
        dtype=np.float32,
    )
    order = np.argsort(squared, axis=1, kind="stable")
    ordered_ids = np.take_along_axis(filtered, order, axis=1)
    ordered_squared = np.take_along_axis(squared, order, axis=1)
    gap = ordered_squared[:, K_FRAC] - ordered_squared[:, K_FRAC - 1]
    if np.any(gap < 0) or not np.isfinite(gap).all():
        raise Round0194Error("low-dimensional boundary guard failed")
    return np.ascontiguousarray(ordered_ids[:, :K_FRAC]), {
        "algorithm": "scipy-cKDTree-exact-query-plus-fp32-candidate-rerank",
        "scipy_version": scipy.__version__,
        "workers": TREE_WORKERS,
        "overselect": TREE_OVERSELECT,
        "minimum_boundary_gap_squared_l2": float(gap.min()),
        "zero_boundary_gaps": int(np.count_nonzero(gap == 0)),
    }


def _unit(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms < 0.90) or np.any(norms > 1.10) or not np.isfinite(norms).all():
        raise Round0194Error("source embedding normalization guard failed")
    return np.ascontiguousarray(array / norms[:, None], dtype=np.float32)


def _predictor_arrays(
    source: np.memmap, anchor_ids: np.ndarray, hi_hit: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    pile = source[PILE_START:PILE_STOP]
    anchor = _unit(pile[anchor_ids])
    nearest_two = _unit(pile[hi_hit[:, :2].reshape(-1)]).reshape(ANCHORS, 2, DIMENSION)
    distance = np.sqrt(np.maximum(
        2.0 - 2.0 * np.sum(anchor[:, None, :] * nearest_two, axis=2), 1e-12
    ))
    log_ratio = np.log(distance[:, 1] / distance[:, 0])
    valid_log = np.isfinite(log_ratio) & (log_ratio > 1e-12)
    if int(valid_log.sum()) < int(0.90 * ANCHORS):
        raise Round0194Error("local TwoNN valid-row guard failed")

    with threadpool_limits(limits=TREE_WORKERS, user_api="blas"):
        similarity = np.asarray(anchor @ anchor.T, dtype=np.float32)
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argpartition(similarity, -K_HIT, axis=1)[:, -K_HIT:]
    occurrences = np.bincount(nearest.reshape(-1), minlength=ANCHORS).astype(np.int64)

    accumulator = np.zeros(DIMENSION, dtype=np.float64)
    for start in range(0, ROWS, CENTROID_CHUNK_ROWS):
        accumulator += np.asarray(
            source[start : start + CENTROID_CHUNK_ROWS], dtype=np.float32
        ).sum(axis=0, dtype=np.float64)
    centroid = accumulator / ROWS
    centroid_norm = float(np.linalg.norm(centroid))
    if not math.isfinite(centroid_norm) or centroid_norm <= 0:
        raise Round0194Error("mixture centroid collapsed")
    centroid /= centroid_norm
    centroid_distance = np.maximum(
        1.0 - np.asarray(anchor @ centroid.astype(np.float32), dtype=np.float64),
        0.0,
    )
    return {
        "log_r2_r1": np.asarray(log_ratio, dtype=np.float64),
        "hubness_occurrence": occurrences,
        "mixture_centroid_distance": centroid_distance,
    }, {
        "twonn": {
            "definition": "local log(r2/r1) from the sealed exact high-D top10; group ID is 1/mean(valid log ratio)",
            "valid_anchors": int(valid_log.sum()),
        },
        "hubness": {
            "definition": "k10 occurrence count in exact cosine kNN among the 4,000 frozen Pile anchors",
            "mean_occurrence": float(occurrences.mean()),
            "maximum_occurrence": int(occurrences.max()),
        },
        "mixture_centroid_distance": {
            "definition": "one minus cosine to the normalized arithmetic mean of all 1,988,104 mixed-quarter embeddings",
            "centroid_sha256": ordered_array_sha256(centroid),
            "source_rows": ROWS,
        },
    }


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if (
        str(job.get("action") or "") != "localize_pile_boundary_loss"
        or active.get("manifest", {}).get("round_id") != ROUND_ID
    ):
        raise Round0194Error("unknown R0194 action or queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0194Error("R0194 is CPU-only")
    started = time.monotonic()
    source_signature = _signature(job["source"], label="mixed-quarter source")
    if source_signature["bytes"] != ROWS * DIMENSION * 2:
        raise Round0194Error("mixed-quarter source byte geometry changed")
    _signature(job["population"], label="mixed-quarter population")
    reference_signature = _signature(job["reference"], label="Pile high-D reference")
    reference = load_hiD_reference(reference_signature["canonical_path"])
    anchors = np.asarray(reference["anchor_ids"], dtype=np.int64)
    hi_hit = np.asarray(reference["hi_hit"], dtype=np.int64)
    if (
        reference["kf"] != K_FRAC
        or anchors.shape != (ANCHORS,)
        or hi_hit.shape != (ANCHORS, K_HIT)
        or set(reference["labels"]) != set(CLUSTER_KS)
    ):
        raise Round0194Error("Pile high-D reference contract changed")
    source = np.memmap(
        source_signature["canonical_path"],
        dtype="<f2",
        mode="r",
        shape=(ROWS, DIMENSION),
    )
    predictors, predictor_receipt = _predictor_arrays(source, anchors, hi_hit)

    scores: dict[int, dict[str, np.ndarray]] = {}
    search_receipts: dict[str, Any] = {}
    evaluation_receipts: dict[str, Any] = {}
    for seed in SEEDS:
        scores[seed] = {}
        evaluation_receipts[str(seed)] = {}
        for rung in ("half", "full"):
            spec = job["cells"][str(seed)][rung]
            evaluation = _read_evaluation(spec["evaluation"], seed=seed, rung=rung)
            coordinate_signature = _signature(
                spec["coordinates"], label=f"seed {seed} {rung} coordinates"
            )
            if evaluation.get("coordinates") != coordinate_signature:
                raise Round0194Error(
                    f"seed {seed} {rung} evaluation/coordinate binding changed"
                )
            coordinates = np.load(
                coordinate_signature["canonical_path"], mmap_mode="r", allow_pickle=False
            )
            if coordinates.shape != (ROWS, 2) or coordinates.dtype != np.float32:
                raise Round0194Error(f"seed {seed} {rung} coordinates changed")
            low, search = _exact_low_fraction(
                coordinates[PILE_START:PILE_STOP], anchors
            )
            score = per_anchor_ffr(hi_hit, low)
            reported = float((evaluation.get("corpus_panels") or {})["pile"]["ffr"])
            if round(float(score.mean()), 4) != reported:
                raise Round0194Error(
                    f"seed {seed} {rung} Pile FFR failed exact reproduction: "
                    f"{score.mean()} versus {reported}"
                )
            scores[seed][rung] = score
            search_receipts[f"seed{seed}_{rung}"] = {
                **search,
                "coordinates": coordinate_signature,
                "per_anchor_ffr_sha256": ordered_array_sha256(score),
                "reproduced_panel_ffr": reported,
            }
            evaluation_receipts[str(seed)][rung] = dict(spec["evaluation"])

    labels = {
        k: np.asarray(reference["labels"][k][anchors], dtype=np.int32)
        for k in CLUSTER_KS
    }
    synthesis = synthesize(scores, labels, predictors)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0194 Pile loss localization"
    )
    array_path = os.path.join(output, "per-anchor-evidence.npz")
    arrays: dict[str, np.ndarray] = {
        "anchor_ids": anchors,
        "labels_256": labels[256],
        "labels_1024": labels[1024],
        **predictors,
    }
    for seed in SEEDS:
        arrays[f"seed{seed}_half_ffr"] = scores[seed]["half"]
        arrays[f"seed{seed}_full_ffr"] = scores[seed]["full"]
        arrays[f"seed{seed}_delta"] = scores[seed]["full"] - scores[seed]["half"]
    atomic_save_new_npz(array_path, immutable=True, **arrays)
    receipt = seal({
        **synthesis,
        "release_sha": active["manifest"]["release_sha"],
        "source": source_signature,
        "population": dict(job["population"]),
        "high_d_reference": reference_signature,
        "reference_content_sha256": reference["content_sha256"],
        "evaluation_receipts": evaluation_receipts,
        "search_receipts": search_receipts,
        "predictor_receipt": predictor_receipt,
        "per_anchor_evidence": expected_input_signature(array_path),
        "accepted_reviews": [dict(value) for value in job["accepted_reviews"]],
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "pile-loss-localization.json"), receipt, immutable=True
    )


__all__ = ["run_job"]
