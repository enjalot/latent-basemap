"""Frozen descriptive summaries for R0194 Pile loss localization."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import ordered_array_sha256
from basemap.round0146_projection_predictors import spearman_rho


ROUND_ID = "0194"
CAPABILITY = "jina-mixed-pile-boundary-loss-localization-v1"
SEEDS = (42, 43, 44)
ANCHORS = 4_000
K_HIT = 10
K_FRAC = 567
CLUSTER_KS = (256, 1024)


class Round0194Error(RuntimeError):
    """The registered descriptive Pile analysis changed or is invalid."""


def per_anchor_ffr(hi_hit: np.ndarray, low_fraction: np.ndarray) -> np.ndarray:
    high = np.asarray(hi_hit, dtype=np.int64)
    low = np.asarray(low_fraction, dtype=np.int64)
    if (
        high.ndim != 2
        or high.shape[1] != K_HIT
        or low.ndim != 2
        or low.shape != (len(high), K_FRAC)
    ):
        raise Round0194Error("per-anchor FFR geometry changed")
    hits = np.any(high[:, :, None] == low[:, None, :], axis=2)
    return np.asarray(hits.mean(axis=1), dtype=np.float64)


def cluster_summary(labels: np.ndarray, mean_delta: np.ndarray, *, k: int) -> dict[str, Any]:
    group = np.asarray(labels, dtype=np.int64)
    delta = np.asarray(mean_delta, dtype=np.float64)
    if (
        group.shape != delta.shape
        or group.ndim != 1
        or len(group) != ANCHORS
        or np.any(group < 0)
        or np.any(group >= k)
        or not np.isfinite(delta).all()
    ):
        raise Round0194Error(f"k{k} cluster inputs changed")
    negative = np.maximum(-delta, 0.0)
    rows = []
    for cluster in np.unique(group):
        mask = group == cluster
        rows.append({
            "cluster": int(cluster),
            "anchors": int(mask.sum()),
            "mean_delta": float(delta[mask].mean()),
            "median_delta": float(np.median(delta[mask])),
            "losing_fraction": float(np.mean(delta[mask] < 0)),
            "negative_loss_mass": float(negative[mask].sum()),
        })
    rows.sort(key=lambda value: (value["mean_delta"], value["cluster"]))
    masses = sorted((row["negative_loss_mass"] for row in rows), reverse=True)
    total_mass = float(sum(masses))
    top_count = max(1, int(math.ceil(0.10 * len(rows))))
    top_share = float(sum(masses[:top_count]) / total_mass) if total_mass > 0 else 0.0
    losing_clusters = sum(row["losing_fraction"] > 0 for row in rows)
    return {
        "k": k,
        "occupied_clusters": len(rows),
        "clusters_with_any_losing_anchor": losing_clusters,
        "losing_cluster_coverage": float(losing_clusters / len(rows)),
        "top_decile_cluster_count": top_count,
        "top_decile_negative_loss_mass_share": top_share,
        "rows": rows,
    }


def _skew(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    centered = array - array.mean()
    scale = float(np.sqrt(np.mean(centered * centered)))
    return 0.0 if scale == 0 else float(np.mean((centered / scale) ** 3))


def predictor_groups(
    mean_delta: np.ndarray,
    log_r2_r1: np.ndarray,
    hubness_occurrence: np.ndarray,
    mixture_centroid_distance: np.ndarray,
) -> dict[str, Any]:
    delta = np.asarray(mean_delta, dtype=np.float64)
    log_ratio = np.asarray(log_r2_r1, dtype=np.float64)
    hubness = np.asarray(hubness_occurrence, dtype=np.float64)
    centroid = np.asarray(mixture_centroid_distance, dtype=np.float64)
    if any(value.shape != (ANCHORS,) for value in (delta, log_ratio, hubness, centroid)):
        raise Round0194Error("predictor array geometry changed")
    valid_log = np.isfinite(log_ratio) & (log_ratio > 1e-12)
    if (
        not np.isfinite(delta).all()
        or int(valid_log.sum()) < int(0.90 * ANCHORS)
        or not np.isfinite(hubness).all()
        or not np.isfinite(centroid).all()
        or np.any(hubness < 0)
        or np.any(centroid < 0)
    ):
        raise Round0194Error("predictor arrays are invalid")
    groups = {}
    for name, mask in (("losing", delta < 0), ("retaining", delta >= 0)):
        if int(mask.sum()) < 32:
            raise Round0194Error(f"{name} predictor group is too small")
        group_valid = mask & valid_log
        if int(group_valid.sum()) < 32:
            raise Round0194Error(f"{name} valid TwoNN group is too small")
        groups[name] = {
            "anchors": int(mask.sum()),
            "twonn_valid_anchors": int(group_valid.sum()),
            "mean_delta": float(delta[mask].mean()),
            "twonn_intrinsic_dimension": float(1.0 / log_ratio[group_valid].mean()),
            "hubness_occurrence_mean": float(hubness[mask].mean()),
            "hubness_occurrence_skew": _skew(hubness[mask]),
            "mixture_centroid_distance_p50": float(np.median(centroid[mask])),
            "mixture_centroid_distance_p90": float(np.percentile(centroid[mask], 90)),
        }
    correlations = {
        "local_log_r2_r1_vs_mean_delta": spearman_rho(
            log_ratio[valid_log], delta[valid_log]
        ),
        "hubness_occurrence_vs_mean_delta": spearman_rho(hubness, delta),
        "mixture_centroid_distance_vs_mean_delta": spearman_rho(centroid, delta),
    }
    return {
        "twonn_valid_anchors": int(valid_log.sum()),
        "groups": groups,
        "spearman": correlations,
    }


def synthesize(
    scores: Mapping[int, Mapping[str, np.ndarray]],
    labels: Mapping[int, np.ndarray],
    predictors: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if set(scores) != set(SEEDS) or set(labels) != set(CLUSTER_KS):
        raise Round0194Error("seed or cluster family changed")
    deltas = []
    cells = {}
    for seed in SEEDS:
        half = np.asarray(scores[seed]["half"], dtype=np.float64)
        full = np.asarray(scores[seed]["full"], dtype=np.float64)
        if half.shape != (ANCHORS,) or full.shape != (ANCHORS,):
            raise Round0194Error(f"seed {seed} per-anchor geometry changed")
        delta = full - half
        deltas.append(delta)
        cells[str(seed)] = {
            "half_ffr": float(half.mean()),
            "full_ffr": float(full.mean()),
            "mean_delta": float(delta.mean()),
            "losing_anchors": int(np.count_nonzero(delta < 0)),
            "retaining_anchors": int(np.count_nonzero(delta >= 0)),
        }
    delta_matrix = np.stack(deltas)
    mean_delta = delta_matrix.mean(axis=0)
    consensus = np.sum(delta_matrix < 0, axis=0)
    clusters = {
        str(k): cluster_summary(np.asarray(labels[k]), mean_delta, k=k)
        for k in CLUSTER_KS
    }
    k256 = clusters["256"]
    top_share = float(k256["top_decile_negative_loss_mass_share"])
    coverage = float(k256["losing_cluster_coverage"])
    if top_share <= 0.35 and coverage >= 0.75:
        pattern = "diffuse"
    elif top_share >= 0.50:
        pattern = "cluster-concentrated"
    else:
        pattern = "mixed-or-unresolved"
    predictor_result = predictor_groups(
        mean_delta,
        np.asarray(predictors["log_r2_r1"]),
        np.asarray(predictors["hubness_occurrence"]),
        np.asarray(predictors["mixture_centroid_distance"]),
    )
    return {
        "schema": "round0194-pile-boundary-loss-localization-v1",
        "round_id": ROUND_ID,
        "capabilities": [CAPABILITY],
        "seeds": list(SEEDS),
        "anchor_count": ANCHORS,
        "per_seed": cells,
        "across_seed": {
            "mean_delta": float(mean_delta.mean()),
            "anchors_mean_losing": int(np.count_nonzero(mean_delta < 0)),
            "anchors_negative_in_at_least_two_seeds": int(np.count_nonzero(consensus >= 2)),
            "mean_delta_sha256": ordered_array_sha256(mean_delta),
            "negative_seed_count_sha256": ordered_array_sha256(consensus.astype(np.int8)),
        },
        "cluster_localization": clusters,
        "predictors": predictor_result,
        "descriptive_pattern": pattern,
        "steering": (
            "prioritize mixture/subdomain cap-or-reweight probes"
            if pattern == "cluster-concentrated"
            else "prioritize capacity/global-geometry explanations"
            if pattern == "diffuse"
            else "do not choose between mixture and capacity from this analysis alone"
        ),
        "scope": {
            "hypothesis_generating": True,
            "quality_gate": False,
            "causal_claim": False,
            "training_performed": False,
        },
    }


__all__ = ["ANCHORS", "CAPABILITY", "CLUSTER_KS", "K_FRAC", "K_HIT", "ROUND_ID", "Round0194Error", "SEEDS", "cluster_summary", "per_anchor_ffr", "predictor_groups", "synthesize"]
