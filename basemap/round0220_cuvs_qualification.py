"""R0220 — qualify cuVS as a k15 all-neighbours graph builder for Phase 2.

This module holds the registered constants and the *pure* measurement code for
Round 0220. It qualifies a builder against exact ground truth; it registers no
gate, trains nothing, and makes no map-quality claim.

Ground truth is R0216's `queue-correction-3` release
(`minilm-mixed-2m-substrate-and-exact-k15-graph-v1`). R0216 did **not** persist
its `nbr`/`dist` arrays (review-0216-02, required correction 3), so the exact
k15 neighbour lists are *recomputed* here from the sealed substrate under the
identical blocked fp32 cosine law, and the recomputation is then validated
against the sealed `edges-k15-fuzzy.npz` adjacency with the tie-aware measure
review-0216-02 used. The substrate contains exact-duplicate clusters (one of
1,377 byte-identical rows), so a strict set comparison understates any correct
builder; both strict and tie-aware numbers are reported everywhere.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ROUND_ID = "0220"
CAPABILITY = "cuvs-k15-graph-builder-qualification-v1"
TRUTH_SCHEMA = "round0220-exact-k15-truth-rebuild-v1"
QUALIFICATION_SCHEMA = "round0220-cuvs-k15-builder-qualification-v1"

#: The R0216 release this round measures against. Hashes are the ones
#: review-0216-02 published when it released the capability.
GRAPH_SOURCE_ROUND_ID = "0216"
GRAPH_CAPABILITY = "minilm-mixed-2m-substrate-and-exact-k15-graph-v1"
GRAPH_SCHEMA = "round0216-minilm-mixed-2m-exact-k15-graph-v1"
SUBSTRATE_SHA256 = "372fbec511c0e9fa3b8e141529ecccaad975e469fe7b8296c019698b340b3660"
EDGES_SHA256 = "8d99aa0ef20c465b9f873d878ef2a5a3147265dc71a2aaaec24ee98b1b7faad7"
SEALED_DIRECTED_EDGES = 48_344_648

ROWS = 2_000_000
DIMENSION = 384
GRAPH_K = 15

#: R0216's exact-search blocking law, reused unchanged so the recomputed truth
#: is the same object R0216 built.
SEARCH_BLOCK = 100_000
QUERY_BLOCK = 16_384

#: Two float32 cosines within this tolerance are treated as a tie. Review
#: 0216-02 used the same 1e-6 scale when it identified duplicate clusters.
TIE_TOLERANCE = 1e-6

#: Registered *in advance*: the probe used to check the recomputed truth
#: against R0216's sealed adjacency.
TRUTH_PROBE_ROWS = 65_536
TRUTH_PROBE_SEED = 220
TRUTH_VALIDITY_MEAN_FLOOR = 0.999
TRUTH_VALIDITY_P10_FLOOR = 0.99

#: cuVS recall is measured over the **full** 2,000,000 rows, not a probe.
RECALL_ROWS = ROWS

#: The substrate is unit-normalised (R0216 checked norms in
#: [0.99999988, 1.00000012]), so squared-Euclidean ordering is identical to
#: cosine ordering: ||a-b||^2 = 2 - 2*cos. cuVS nn-descent/CAGRA expose
#: `sqeuclidean`, so the registered metric is `sqeuclidean` and the ranking is
#: the same object R0216's cosine search ranks.
CUVS_METRIC = "sqeuclidean"
METRIC_EQUIVALENCE = (
    "substrate rows are unit-normalised, so sqeuclidean = 2 - 2*cosine is a "
    "strictly decreasing function of cosine and induces the identical ranking"
)

#: The swept settings. Three nn-descent points give the curve for the direct
#: all-neighbours kernel (the paper's per-cluster builder); two CAGRA points
#: contrast a search-time recall knob against a build-time one.
SWEEP: tuple[dict[str, Any], ...] = (
    {
        "id": "nnd-gd32-igd48-it20",
        "algo": "nn_descent",
        "graph_degree": 32,
        "intermediate_graph_degree": 48,
        "max_iterations": 20,
    },
    {
        "id": "nnd-gd64-igd96-it20",
        "algo": "nn_descent",
        "graph_degree": 64,
        "intermediate_graph_degree": 96,
        "max_iterations": 20,
    },
    {
        "id": "nnd-gd64-igd128-it40",
        "algo": "nn_descent",
        "graph_degree": 64,
        "intermediate_graph_degree": 128,
        "max_iterations": 40,
    },
    {
        "id": "cagra-gd32-igd64-itopk64",
        "algo": "cagra",
        "graph_degree": 32,
        "intermediate_graph_degree": 64,
        "itopk_size": 64,
        "search_width": 1,
    },
    {
        "id": "cagra-gd32-igd64-itopk256",
        "algo": "cagra",
        "graph_degree": 32,
        "intermediate_graph_degree": 64,
        "itopk_size": 256,
        "search_width": 4,
    },
)

#: The setting whose wall time is measured at several N to fit the scaling
#: exponent used for the 100M projection.
SCALING_SETTING_ID = "nnd-gd64-igd96-it20"
SCALING_ROWS: tuple[int, ...] = (250_000, 500_000, 1_000_000, 2_000_000)
PROJECTION_ROWS = 100_000_000

#: Review-0216-02's published calibration for the exact brute-force kernel:
#: 0.107 GPU-h at 2M, ~78 GPU-h at 100M (quadratic in N). Cited as the
#: baseline this round is compared against; not remeasured here.
BRUTE_FORCE_2M_GPU_HOURS = 0.10710089
BRUTE_FORCE_100M_GPU_HOURS = 78.0
BRUTE_FORCE_BASELINE_SOURCE = "review-0216-2026-08-08-02.md, performance review"

#: cuVS 25.02 ships no `cuvs.neighbors.all_neighbors`; the out-of-core batched
#: builder of 0197-out-of-core-umap-gpu.md is not available in this env. Any
#: projection past what fits in 32 GiB of VRAM is therefore a compute-only
#: extrapolation and must be labelled as such.
REQUIRED_CUVS_MODULES = ("nn_descent", "cagra")
OUT_OF_CORE_MODULE = "all_neighbors"

GPU_HOURS_CAP = 3.0


class Round0220Error(RuntimeError):
    """Registered R0220 failure."""


def _finite(values: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise Round0220Error(f"{label}: non-finite or empty measurement")
    return array


def summarize(values: np.ndarray, *, label: str) -> dict[str, float]:
    """Mean / p10 / min / max / perfect-row fraction, with a finiteness gate."""
    array = _finite(values, label=label)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "min": float(array.min()),
        "max": float(array.max()),
        "fraction_perfect": float(np.mean(array >= 1.0)),
    }


def _first_occurrence_mask(ids: np.ndarray) -> np.ndarray:
    """Per-row mask marking the first occurrence of each id (dedup within row)."""
    order = np.argsort(ids, axis=1, kind="stable")
    ordered = np.take_along_axis(ids, order, axis=1)
    fresh = np.ones(ordered.shape, dtype=bool)
    fresh[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    mask = np.empty_like(fresh)
    np.put_along_axis(mask, order, fresh, axis=1)
    return mask


def strict_containment_rows(
    candidates: np.ndarray, truth: np.ndarray, *, chunk: int = 100_000
) -> np.ndarray:
    """Fraction of each row's exact top-k that appears in the candidate row.

    Containment of truth in the candidate set, so duplicated candidate ids
    cannot inflate the score.
    """
    candidates = np.asarray(candidates)
    truth = np.asarray(truth)
    if candidates.shape[0] != truth.shape[0]:
        raise Round0220Error("candidate and truth row counts differ")
    k = truth.shape[1]
    out = np.empty(truth.shape[0], dtype=np.float64)
    for start in range(0, truth.shape[0], chunk):
        stop = min(start + chunk, truth.shape[0])
        hit = (
            truth[start:stop, :, None].astype(np.int64)
            == candidates[start:stop, None, :].astype(np.int64)
        ).any(axis=2)
        out[start:stop] = hit.sum(axis=1) / k
    return out


def tie_aware_rows(
    candidate_cosines: np.ndarray,
    candidate_ids: np.ndarray,
    kth_cosine: np.ndarray,
    *,
    k: int = GRAPH_K,
    tolerance: float = TIE_TOLERANCE,
) -> np.ndarray:
    """Validity of a candidate row against exact truth, tie-aware.

    A candidate is valid when its true cosine is at least the true k-th best
    cosine (within `tolerance`). Duplicated ids inside a row count once and the
    per-row count is capped at `k`, exactly as review-0216-02 measured the
    sealed adjacency.
    """
    cosines = np.asarray(candidate_cosines, dtype=np.float64)
    fresh = _first_occurrence_mask(np.asarray(candidate_ids))
    valid = (cosines >= (np.asarray(kth_cosine, dtype=np.float64)[:, None] - tolerance))
    counts = np.minimum((valid & fresh).sum(axis=1), k)
    return counts / float(k)


def graph_validity(ids: np.ndarray, *, rows: int) -> dict[str, int]:
    """Structural validity of a raw k-NN graph, including the R0215 tripwire."""
    ids = np.asarray(ids)
    self_ids = np.arange(ids.shape[0], dtype=np.int64)[:, None]
    out_of_range = (ids.astype(np.int64) < 0) | (ids.astype(np.int64) >= rows)
    self_loops = (ids.astype(np.int64) == self_ids) & ~out_of_range
    fresh = _first_occurrence_mask(ids)
    usable = fresh & ~out_of_range & ~self_loops
    degree = usable.sum(axis=1)
    return {
        "rows": int(ids.shape[0]),
        "width": int(ids.shape[1]),
        "out_of_range_entries": int(out_of_range.sum()),
        "rows_with_out_of_range": int((out_of_range.any(axis=1)).sum()),
        "self_loop_entries": int(self_loops.sum()),
        "rows_with_self_loop": int(self_loops.any(axis=1).sum()),
        "duplicate_entries": int((~fresh).sum()),
        "rows_with_duplicates": int((~fresh).any(axis=1).sum()),
        "min_usable_degree": int(degree.min()),
        "rows_below_k": int((degree < GRAPH_K).sum()),
        "zero_degree_rows": int((degree == 0).sum()),
    }


def power_law(sizes: Sequence[int], seconds: Sequence[float]) -> dict[str, Any]:
    """Least-squares log-log fit `t = a * N**b`, with its own R^2."""
    n = _finite(np.asarray(sizes, dtype=np.float64), label="scaling sizes")
    t = _finite(np.asarray(seconds, dtype=np.float64), label="scaling seconds")
    if n.size != t.size or n.size < 2:
        raise Round0220Error("power-law fit needs at least two matched points")
    if np.any(n <= 0) or np.any(t <= 0):
        raise Round0220Error("power-law fit needs strictly positive points")
    x = np.log(n)
    y = np.log(t)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "exponent": float(slope),
        "coefficient": float(math.exp(intercept)),
        "r_squared": float(1.0 - residual / total) if total > 0 else 1.0,
        "points": [
            {"rows": int(a), "seconds": float(b)} for a, b in zip(n.tolist(), t.tolist())
        ],
    }


def project_cost(fit: Mapping[str, Any], *, rows: int = PROJECTION_ROWS) -> dict[str, Any]:
    """Extrapolate the fitted power law. A projection, never a measurement."""
    seconds = float(fit["coefficient"]) * float(rows) ** float(fit["exponent"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise Round0220Error("projected cost is not a finite positive time")
    gpu_hours = seconds / 3600.0
    return {
        "is_measurement": False,
        "kind": "projection",
        "rows": int(rows),
        "assumption": (
            "wall time follows the power law t = a*N**b fitted to measured "
            f"builds at {list(SCALING_ROWS)} rows for setting "
            f"{SCALING_SETTING_ID}; the exponent is measured on this box, not "
            "assumed"
        ),
        "caveat": (
            "the fit covers only the regime where the dataset and its "
            "intermediate graph fit in 32 GiB of VRAM. 100,000,000 x 384 fp32 "
            "is 153.6 GB, so a real 100M build needs the out-of-core batched "
            "builder of arXiv 0197, which cuvs 25.02 does not ship. The "
            "projection is therefore a compute-only lower bound and excludes "
            "any clustering, spilling, host-device transfer, or merge cost"
        ),
        "projected_seconds": seconds,
        "projected_gpu_hours": gpu_hours,
        "brute_force_baseline_gpu_hours": BRUTE_FORCE_100M_GPU_HOURS,
        "brute_force_baseline_source": BRUTE_FORCE_BASELINE_SOURCE,
        "speedup_vs_brute_force": (
            BRUTE_FORCE_100M_GPU_HOURS / gpu_hours if gpu_hours > 0 else None
        ),
    }


def validate_truth_probe(
    *, tie_aware: Mapping[str, float], strict: Mapping[str, float]
) -> dict[str, Any]:
    """Fail closed unless the recomputed truth reproduces R0216's sealed graph."""
    checks = {
        "tie_aware_mean": float(tie_aware["mean"]),
        "tie_aware_p10": float(tie_aware["p10"]),
        "strict_mean": float(strict["mean"]),
        "strict_p10": float(strict["p10"]),
        "mean_floor": TRUTH_VALIDITY_MEAN_FLOOR,
        "p10_floor": TRUTH_VALIDITY_P10_FLOOR,
        "probe_rows": TRUTH_PROBE_ROWS,
        "probe_seed": TRUTH_PROBE_SEED,
    }
    passed = (
        checks["tie_aware_mean"] >= TRUTH_VALIDITY_MEAN_FLOOR
        and checks["tie_aware_p10"] >= TRUTH_VALIDITY_P10_FLOOR
    )
    checks["passed"] = passed
    if not passed:
        raise Round0220Error(
            "recomputed k15 truth does not reproduce R0216's sealed adjacency: "
            f"tie-aware mean {checks['tie_aware_mean']:.6f} / p10 "
            f"{checks['tie_aware_p10']:.6f} against floors "
            f"{TRUTH_VALIDITY_MEAN_FLOOR} / {TRUTH_VALIDITY_P10_FLOOR}"
        )
    return checks


def setting(setting_id: str) -> dict[str, Any]:
    for item in SWEEP:
        if item["id"] == setting_id:
            return dict(item)
    raise Round0220Error(f"unknown R0220 sweep setting {setting_id!r}")


__all__ = [
    "BRUTE_FORCE_100M_GPU_HOURS",
    "BRUTE_FORCE_2M_GPU_HOURS",
    "BRUTE_FORCE_BASELINE_SOURCE",
    "CAPABILITY",
    "CUVS_METRIC",
    "DIMENSION",
    "EDGES_SHA256",
    "GPU_HOURS_CAP",
    "GRAPH_CAPABILITY",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "GRAPH_SOURCE_ROUND_ID",
    "METRIC_EQUIVALENCE",
    "OUT_OF_CORE_MODULE",
    "PROJECTION_ROWS",
    "QUALIFICATION_SCHEMA",
    "QUERY_BLOCK",
    "RECALL_ROWS",
    "REQUIRED_CUVS_MODULES",
    "ROUND_ID",
    "ROWS",
    "Round0220Error",
    "SCALING_ROWS",
    "SCALING_SETTING_ID",
    "SEALED_DIRECTED_EDGES",
    "SEARCH_BLOCK",
    "SUBSTRATE_SHA256",
    "SWEEP",
    "TIE_TOLERANCE",
    "TRUTH_PROBE_ROWS",
    "TRUTH_PROBE_SEED",
    "TRUTH_SCHEMA",
    "TRUTH_VALIDITY_MEAN_FLOOR",
    "TRUTH_VALIDITY_P10_FLOOR",
    "graph_validity",
    "power_law",
    "project_cost",
    "setting",
    "strict_containment_rows",
    "summarize",
    "tie_aware_rows",
    "validate_truth_probe",
]
