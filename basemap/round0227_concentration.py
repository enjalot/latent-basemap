"""R0227 — where a sharded builder's missing edges LIVE, not just how many.

Review-0226-01's central scientific finding was that candidate A's `2.9%` missing
edges are not spread across the substrate. They are:

* **monotone in local density** — tie-aware recall `0.9130` in the sparsest tenth
  against `0.9957` in the densest, a `20x` heavier loss where a UMAP graph has the
  fewest alternate paths;
* **spatially autocorrelated** — a row's loss correlates with the mean loss of
  its 15 *true* neighbours at `r = 0.6216`, against a shuffled null of `~0`;
* **carried by a minority** — `19.57%` of rows carry `100%` of the loss and the
  worst `1%` of rows carry `19.3%` of it.

R0215 showed sparse regions are exactly where the v1 150M map broke, so a
configuration whose *mean* recall improves while its loss stays concentrated in
the sparse tail has not actually been fixed. These functions measure the
concentration so that claim can be tested rather than asserted, and they are
written to produce the same statistics review-0226-01 published so the
comparison is like-for-like.

One reassuring measurement from that review is reproduced here too: the mean
cosine of an *emitted* edge against the mean cosine of a *true* edge. R0215's v1
graph scored `~0.47` edge precision; R0226's builders scored `0.97/0.95`, which
says the substitutions are near-misses rather than garbage. It is the single
strongest positive signal about this builder family and it belongs beside the
bad news.
"""
from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_DECILES = 10


class Round0227ConcentrationError(RuntimeError):
    """A concentration statistic was asked for on inputs it cannot describe."""


def density_decile_recall(
    per_row_recall: np.ndarray,
    kth_cosine: np.ndarray,
    *,
    deciles: int = DEFAULT_DECILES,
) -> dict[str, Any]:
    """Mean recall inside each decile of local density.

    Local density is the row's own true k-th best cosine: a row whose 15th
    neighbour is far away sits in a sparse neighbourhood. Decile `0` is the
    sparsest. Ranking is done on the cosine directly, so the bins are the same
    bins review-0226-01 used.
    """
    recall = np.asarray(per_row_recall, dtype=np.float64)
    density = np.asarray(kth_cosine, dtype=np.float64)
    if recall.shape != density.shape or recall.ndim != 1:
        raise Round0227ConcentrationError(
            "density-decile recall needs matched 1-D recall and density arrays"
        )
    if recall.size < int(deciles):
        raise Round0227ConcentrationError("not enough rows to form the deciles")
    order = np.argsort(density, kind="stable")
    edges = np.linspace(0, recall.size, int(deciles) + 1).astype(np.int64)
    means: list[float] = []
    counts: list[int] = []
    cosine_bounds: list[list[float]] = []
    for index in range(int(deciles)):
        rows = order[edges[index] : edges[index + 1]]
        means.append(float(recall[rows].mean()))
        counts.append(int(rows.size))
        cosine_bounds.append(
            [float(density[rows].min()), float(density[rows].max())]
        )
    return {
        "deciles": int(deciles),
        "decile_mean_recall": means,
        "decile_rows": counts,
        "decile_kth_cosine_bounds": cosine_bounds,
        "sparsest_decile_mean": means[0],
        "densest_decile_mean": means[-1],
        "sparsest_to_densest_gap": float(means[-1] - means[0]),
        "monotone_nondecreasing": bool(
            all(means[index] <= means[index + 1] + 1e-12 for index in range(len(means) - 1))
        ),
        "definition": (
            "deciles of the row's own true k-th best cosine; decile 0 is the "
            "sparsest local neighbourhood, decile 9 the densest"
        ),
    }


def neighbour_loss_autocorrelation(
    per_row_recall: np.ndarray,
    truth_ids: np.ndarray,
    *,
    seed: int = 227,
    subset: np.ndarray | None = None,
) -> dict[str, Any]:
    """Correlate a row's loss with the mean loss of its 15 TRUE neighbours.

    A shuffled null is computed by permuting the loss vector and repeating the
    same gather, which destroys the spatial relationship while preserving the
    loss distribution exactly. `subset` restricts the *queries* to rows whose
    neighbours all have a measured loss, which is what makes this computable on
    a sampled population.
    """
    recall = np.asarray(per_row_recall, dtype=np.float64)
    ids = np.asarray(truth_ids, dtype=np.int64)
    if ids.ndim != 2 or ids.shape[0] != recall.size:
        raise Round0227ConcentrationError(
            "autocorrelation needs a (rows, k) truth table matching the recall rows"
        )
    loss = 1.0 - recall
    query = (
        np.arange(recall.size, dtype=np.int64) if subset is None
        else np.asarray(subset, dtype=np.int64)
    )
    if query.size < 2:
        raise Round0227ConcentrationError("autocorrelation needs >= 2 query rows")
    neighbour_loss = loss[ids[query]].mean(axis=1)
    own_loss = loss[query]
    observed = _pearson(own_loss, neighbour_loss)
    rng = np.random.default_rng(seed)
    shuffled = loss.copy()
    rng.shuffle(shuffled)
    null = _pearson(shuffled[query], shuffled[ids[query]].mean(axis=1))
    return {
        "query_rows": int(query.size),
        "neighbour_loss_correlation": observed,
        "shuffled_null_correlation": null,
        "mean_loss": float(own_loss.mean()),
        "definition": (
            "Pearson r between a row's own missing-edge fraction and the mean "
            "missing-edge fraction of its 15 exact-truth neighbours; the null "
            "permutes the loss vector and repeats the identical gather"
        ),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() == 0.0 or right.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def loss_concentration(per_row_recall: np.ndarray) -> dict[str, Any]:
    """How much of the total loss the worst rows carry."""
    recall = np.asarray(per_row_recall, dtype=np.float64)
    if recall.ndim != 1 or recall.size == 0:
        raise Round0227ConcentrationError("loss concentration needs a 1-D recall array")
    loss = 1.0 - recall
    total = float(loss.sum())
    ordered = np.sort(loss)[::-1]
    def _share(fraction: float) -> float:
        take = max(1, int(round(recall.size * fraction)))
        return float(ordered[:take].sum() / total) if total > 0 else 0.0
    return {
        "rows": int(recall.size),
        "mean_recall": float(recall.mean()),
        "total_loss": total,
        "rows_carrying_any_loss": float(np.mean(loss > 0.0)),
        "worst_1pct_share_of_loss": _share(0.01),
        "worst_5pct_share_of_loss": _share(0.05),
        "worst_10pct_share_of_loss": _share(0.10),
        "definition": (
            "loss is 1 - per-row recall; shares are of the summed loss, so a "
            "uniform loss gives 0.01/0.05/0.10 and a fully concentrated one "
            "gives shares near 1.0"
        ),
    }


def edge_precision(
    *,
    candidate_ids: np.ndarray,
    candidate_cosines: np.ndarray,
    truth_ids: np.ndarray,
    truth_cosines: np.ndarray,
) -> dict[str, Any]:
    """Mean cosine of an emitted edge, a missed true edge and a substituted edge.

    R0215's v1 150M graph had edge precision `~0.47`: half its edges were simply
    wrong. If a builder's substitutions are near-misses instead, its emitted-edge
    cosine sits within a percent of truth's, and that is a different — far more
    benign — failure mode. Rows are matched positionally, so this must be handed
    the same row population on both sides.
    """
    ids = np.asarray(candidate_ids, dtype=np.int64)
    cosines = np.asarray(candidate_cosines, dtype=np.float64)
    true_ids = np.asarray(truth_ids, dtype=np.int64)
    true_cos = np.asarray(truth_cosines, dtype=np.float64)
    if ids.shape != cosines.shape or true_ids.shape != true_cos.shape:
        raise Round0227ConcentrationError("edge precision needs matched id/cosine tables")
    if ids.shape[0] != true_ids.shape[0]:
        raise Round0227ConcentrationError("edge precision needs the same rows on both sides")
    # Chunked: the pairwise (rows, k, k) comparison is 450 MB per boolean array
    # at 2,000,000 rows, and materialising two of them at once is exactly the
    # kind of avoidable host allocation this program has been bitten by.
    emitted_sum = 0.0
    emitted_count = 0
    missed_sum = 0.0
    missed_count = 0
    substituted_sum = 0.0
    substituted_count = 0
    total_slots = 0
    chunk = 100_000
    for begin in range(0, ids.shape[0], chunk):
        end = min(begin + chunk, ids.shape[0])
        block_ids = ids[begin:end]
        block_cos = cosines[begin:end]
        block_true_ids = true_ids[begin:end]
        block_true_cos = true_cos[begin:end]
        same = block_true_ids[:, :, None] == block_ids[:, None, :]
        emitted = np.isfinite(block_cos) & (block_ids >= 0)
        hit = same.any(axis=2)
        found = same.any(axis=1)
        substituted = emitted & ~found
        emitted_sum += float(block_cos[emitted].sum())
        emitted_count += int(emitted.sum())
        missed_sum += float(block_true_cos[~hit].sum())
        missed_count += int((~hit).sum())
        substituted_sum += float(block_cos[substituted].sum())
        substituted_count += int(substituted.sum())
        total_slots += int(substituted.size)
        del same, emitted, hit, found, substituted
    return {
        "rows": int(ids.shape[0]),
        "mean_true_edge_cosine": float(true_cos.mean()),
        "mean_emitted_edge_cosine": (
            emitted_sum / emitted_count if emitted_count else None
        ),
        "emitted_over_true_ratio": (
            (emitted_sum / emitted_count) / float(true_cos.mean())
            if emitted_count and true_cos.mean() != 0 else None
        ),
        "mean_missed_true_edge_cosine": (
            missed_sum / missed_count if missed_count else None
        ),
        "mean_substituted_edge_cosine": (
            substituted_sum / substituted_count if substituted_count else None
        ),
        "substituted_edge_fraction": (
            substituted_count / total_slots if total_slots else 0.0
        ),
        "r0215_reference_edge_precision": 0.47,
        "definition": (
            "an emitted edge is any non-sentinel candidate id; a missed true "
            "edge is a truth id absent from the candidate row; a substituted "
            "edge is a candidate id absent from the truth row"
        ),
    }


__all__ = [
    "DEFAULT_DECILES",
    "Round0227ConcentrationError",
    "density_decile_recall",
    "edge_precision",
    "loss_concentration",
    "neighbour_loss_autocorrelation",
]
