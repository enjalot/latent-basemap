"""R0261 — price the 4M exact k15 graph by building it, on the n = 29 universe.

`design-0260-reference-anchored-quality-criterion.md` §5 opens option 3 (a
reference-anchored quality criterion `D_k = |log purity_ratio_k|`) and makes its
registrability conditional on **same-universe evidence at more than one N**.
Its E2 reuses the sealed `n = 29` 2M family for free; its E3 needs a *second*
rung that is neither 2M nor a ladder rung, and nominates 4M. The design does not
price it and says so:

    "Total new GPU: approximately 0.8-1.0 GPU-h for the seeds and scoring, plus
     the 4M substrate and exact k15 graph. The graph is the real cost and is not
     estimated here ... That estimate is the first thing the design round must
     produce."

review-0260-2026-08-12-01 §H raised exactly this to a finding — the design
"under-headlines an unpriced 4M exact graph" — and §H's recommendation is to
"price the graph first", because the cheaper alternative (spending the three
`6250k` maps as calibration cells) forecloses option 3's own motivating test
case: the `k256` drift at `1.0982-1.1022` that `T_k` exists to see.

**This round buys the measurement rather than extrapolating to it.** Two sealed
prior measurements bound the answer well enough that a build is affordable
inside the 3.0 GPU-h cap, so the price is *measured* and the artifact E3 needs
exists afterwards. That choice is registered in `round-0261-2026-08-12.md` §2
before the queue runs, together with the numeric prediction this module holds.

The universe is R0216's, unchanged: same four corpora, same 40/25/25/10 shares,
same R0025 loading contract, same excluded damaged shard, same span-sampling
law, same exact brute-force fp32 cosine builder at the *same block sizes*. Only
`ROWS` and the selection seed move. Holding the implementation fixed is what
makes the registered 2M back-check (§`back_check_at_2m`) a real check rather
than a comparison of two different programs.

Two qualification bars are asserted, both required by the round:

* **recall against exact truth** — and the probe that gates is a *separate*
  plain-NumPy CPU pass, because "an in-node recall probe that shares the
  builder's accumulator is not independent" (review-0216-01, promoted to a
  standing rule in `plan-minilm-100m-v2.md`). R0216's own probe re-ran the
  builder's kernel and unsurprisingly returned `1.0`. Here the builder's GPU
  probe is still published, but the CPU probe is the one with authority.
* **the degree-zero tripwire** — R0215 identified edgeless rows as the
  mechanism behind the v1 map's clumps (v1 carried `2,779,481` of them). Zero is
  the only acceptable count.

Every guard in this module ships a positive control that plants the defect into
the **shipped** function and proves the shipped function refuses it; review-0260
was downgraded partly because "both its ordering guards shipped with no positive
control while 19 tests covered lesser claims".

The `AbortPollGate` class is under an owner stop and this module does not touch
it. Stop-latency evidence here is `PollRecorder` + `gap_report` +
`CoverageLedger`, and every node publishes `observed_span_s` at its top level.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.round0216_minilm_2m_substrate import (
    DIMENSION,
    EXCLUDED_SHARDS,
    GRAPH_K,
    KNOWN_TRAILING_FRAGMENTS,
    MAX_ZERO_DEGREE_ROWS,
    MEAN_RECALL_FLOOR,
    P10_RECALL_FLOOR,
    RAW_FORMAT,
    ROW_POLICY,
    TRAILING_FRAGMENT_POLICY,
    ZERO_ROW_POLICY,
    Round0216Error,
    resolve_shard_rows,
    validate_graph,
)

ROUND_ID = "0261"

CAPABILITY = "minilm-mixed-4m-substrate-and-exact-k15-graph-v1"
PRICE_CAPABILITY = "minilm-mixed-4m-exact-k15-graph-measured-price-v1"
SUBSTRATE_SCHEMA = "round0261-minilm-mixed-4m-substrate-v1"
GRAPH_SCHEMA = "round0261-minilm-mixed-4m-exact-k15-graph-v1"
PREDICTION_SCHEMA = "round0261-four-m-exact-graph-cost-prediction-v1"
PRICE_SCHEMA = "round0261-four-m-exact-graph-measured-price-v1"

ROWS = 4_000_000

#: R0216's table at exactly 2x. Shares are unchanged: 0.40 / 0.25 / 0.25 / 0.10.
COMPOSITION: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2", 1_600_000),
    ("RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2", 1_000_000),
    ("pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2", 1_000_000),
    ("starcoderdata-code-chunked-120-all-MiniLM-L6-v2", 400_000),
)
TARGET_SHARES = {name: n / ROWS for name, n in COMPOSITION}

#: A distinct selection seed. The 4M substrate is deliberately NOT nested inside
#: R0216's 2M one: `rng.choice(free, need)` with a different `need` draws a
#: different set anyway, and E3 needs a second N on the same universe, not a
#: superset of the first. Stated here so no later round reads nestedness in.
SELECTION_SEED = 261
NESTED_IN_R0216 = False

#: Identical to R0216's builder, on purpose. The 2M back-check below compares a
#: law fitted at 4M against R0216's sealed 2M wall; if the block geometry moved,
#: that comparison would measure the geometry change instead of the N scaling.
QUERY_BLOCK = 16_384
SEARCH_BLOCK = 100_000

#: The builder's own GPU probe, kept for continuity with R0216. It shares the
#: builder's kernel and accumulator and therefore CANNOT establish exactness;
#: it is published as a self-consistency figure and never gates.
GPU_PROBE_ROWS = 4_096
GPU_PROBE_SEED = 217
GPU_PROBE_IS_INDEPENDENT = False

#: The probe that gates. Plain NumPy on the host, different seed, different
#: rows, different reduction order, no CUDA involved.
CPU_PROBE_ROWS = 2_048
CPU_PROBE_SEED = 2_611
CPU_PROBE_BLOCK = 50_000

#: The program-wide graph bar from `plan-minilm-100m-v2.md` ("exact sharded-fp32
#: GPU graphs with >= 0.90 recall qualification"). R0216's exact-build floors
#: (0.999 mean / 0.99 p10) are far stricter and are also asserted; both are
#: reported so the round's clearance of the program bar is legible.
PROGRAM_RECALL_FLOOR = 0.90

#: fp32 cosine ties. The CPU probe and the GPU builder reduce 384 products in
#: different orders, so the k-th and (k+1)-th neighbour can swap on rows whose
#: cosines agree to fp32. R0220's registered tolerance, reused unchanged.
TIE_TOLERANCE = 1e-6

# --------------------------------------------------------------------------- #
# Sealed prior measurements. Every one is copied from an artifact on disk, and
# the node re-reads the two that the prediction is built from rather than
# trusting these literals (see `experiments/round0261_nodes.py`).
# --------------------------------------------------------------------------- #

#: R0216 queue-correction-3, `substrate-graph.json`, `performance`.
R0216_ROWS = 2_000_000
R0216_EXACT_SEARCH_S = 112.18968716729432
R0216_FUZZY_S = 6.974188329651952
R0216_TOTAL_WALL_S = 382.67656675493345
R0216_PEAK_ALLOCATED_BYTES = 9_717_483_008
R0216_DIRECTED_EDGES = 48_344_648
R0216_EDGES_NPZ_BYTES = 580_136_932
R0216_SUBSTRATE_BYTES = 3_072_000_128

#: R0233 queue-correction-1, `exact-k15-truth.json`. A DIFFERENT implementation:
#: fully vectorised, no per-row Python postprocess, `query_block` 8,192 and
#: `search_block` 131,072. It is used only to bound the quadratic coefficient,
#: never as an implementation-matched point.
R0233_ROWS = 6_250_000
R0233_EXACT_SEARCH_S = 1048.0414850809998
R0233_PEAK_ALLOCATED_BYTES = 13_950_027_264
R0233_QUERY_BLOCK = 8_192
R0233_SEARCH_BLOCK = 131_072
R0233_IS_IMPLEMENTATION_MATCHED = False

#: The registered tolerance on the 2M back-check. The law is fitted from the 4M
#: node's own decomposed timers and then extrapolated DOWN a factor of two to a
#: wall measured eleven weeks-of-rounds earlier on a different page-cache state;
#: 20% is loose enough that a pass is not luck and tight enough that a
#: qualitatively wrong law fails. Registered before the run.
BACK_CHECK_REL_TOL = 0.20

#: Host anonymous budget for the build node, as arithmetic:
#: X `4,000,000 x 384` fp32 = `6,144,000,000` B; the per-corpus assembly holds
#: at most `1,600,000 x 384` fp32 twice (chunk list + concatenated copy) =
#: `4,915,200,000` B; `nbr`/`dist` are `4,000,000 x 15 x 4` = `240,000,000` B
#: each; `fuzzy_simplicial_set` builds COO/CSR copies over ~`96.7M` edges at
#: 4-12 B/edge. `40` GiB is comfortably above the sum and well under the host.
NODE_ANON_BUDGET_BYTES = 40 * (1 << 30)
MIN_MEM_AVAILABLE_BYTES = 56 * (1 << 30)

#: Device budget, as arithmetic: `Xt` `6,144,000,000` B plus one similarity tile
#: `16,384 x 100,000 x 4 = 6,553,600,000` B plus top-k temporaries. R0216
#: measured `9,717,483,008` B for `3,072,000,128 + 6,553,600,000` plus `91.9` MB
#: of temporaries; the same arithmetic at 4M gives `12,789,483,008` B.
PREDICTED_PEAK_ALLOCATED_BYTES = 12_789_483_008
DEVICE_BUDGET_BYTES = 20 * (1 << 30)


class Round0261Error(RuntimeError):
    """R0261 fails closed."""


# --------------------------------------------------------------------------- #
# 1. the pre-registered price, and the four named models it spans
# --------------------------------------------------------------------------- #


def two_term_fit(
    *, rows_a: int, seconds_a: float, rows_b: int, seconds_b: float
) -> dict[str, float]:
    """Solve `t = a*N^2 + b*N` exactly through two measured points.

    Two terms because the builder has exactly two shapes of work: the blocked
    GEMM + top-k over every (query, candidate) pair, which is quadratic, and the
    per-row Python postprocess that drops the self-match and writes `nbr`/`dist`,
    which is linear. A single power law would blend them into an exponent that
    is neither, and `power_law` in R0220 is available for exactly that reading
    when it is wanted.
    """
    na, nb = float(rows_a), float(rows_b)
    ta, tb = float(seconds_a), float(seconds_b)
    if na <= 0 or nb <= 0 or na == nb:
        raise Round0261Error("R0261 two-term fit needs two distinct positive Ns")
    # a*na^2 + b*na = ta ; a*nb^2 + b*nb = tb  ->  a = (tb/nb - ta/na)/(nb - na)
    a = ((tb / nb) - (ta / na)) / (nb - na)
    b = (ta / na) - a * na
    return {"quadratic_s_per_pair": float(a), "linear_s_per_row": float(b)}


def predict_search_s(*, quadratic_s_per_pair: float, linear_s_per_row: float,
                     rows: int) -> float:
    n = float(rows)
    return float(quadratic_s_per_pair * n * n + linear_s_per_row * n)


def cost_prediction(*, rows: int = ROWS) -> dict[str, Any]:
    """The pre-registered price of the 4M exact search, as four named models.

    Registered BEFORE the build (node `predict_0261` seals this, node
    `build_0261` refuses to start until that seal exists). The interval is the
    span of the four models, not a confidence band: each endpoint is a model a
    reader can re-derive from the two sealed walls quoted above.

    * `M1_pure_quadratic_from_r0216` — the builder's wall is entirely quadratic.
      `112.18968716729432 * (4/2)^2`. The upper anchor.
    * `M2_two_term_r0216_r0233` — solve `a*N^2 + b*N` through both sealed
      points. Mixes two implementations and says so.
    * `M3_r0233_quadratic_plus_r0216_linear` — take `a` from R0233 alone
      (assuming its vectorised wall is pure quadratic) and let R0216's residual
      be the linear term.
    * `M4_half_of_r0216_is_linear` — a deliberately pessimistic-on-scaling lower
      anchor: split R0216's wall 50/50 between the two terms. Nothing suggests
      the split is that linear; it exists to bound the interval from below.
    """
    n = float(rows)
    scale = n / float(R0216_ROWS)
    m1 = R0216_EXACT_SEARCH_S * scale * scale

    fit = two_term_fit(rows_a=R0216_ROWS, seconds_a=R0216_EXACT_SEARCH_S,
                       rows_b=R0233_ROWS, seconds_b=R0233_EXACT_SEARCH_S)
    m2 = predict_search_s(rows=rows, **fit)

    a3 = R0233_EXACT_SEARCH_S / (float(R0233_ROWS) ** 2)
    b3 = (R0216_EXACT_SEARCH_S - a3 * float(R0216_ROWS) ** 2) / float(R0216_ROWS)
    m3 = predict_search_s(quadratic_s_per_pair=a3, linear_s_per_row=b3, rows=rows)

    half = R0216_EXACT_SEARCH_S / 2.0
    m4 = half * scale * scale + half * scale

    models = {
        "M1_pure_quadratic_from_r0216": float(m1),
        "M2_two_term_r0216_r0233": float(m2),
        "M3_r0233_quadratic_plus_r0216_linear": float(m3),
        "M4_half_of_r0216_is_linear": float(m4),
    }
    values = sorted(models.values())
    return {
        "schema": PREDICTION_SCHEMA,
        "rows": int(rows),
        "quantity": "exact_search_s",
        "label": "prediction",
        "models": models,
        "point_estimate_s": models["M2_two_term_r0216_r0233"],
        "point_estimate_model": "M2_two_term_r0216_r0233",
        "interval_s": [values[0], values[-1]],
        "m2_fit": fit,
        "m3_fit": {"quadratic_s_per_pair": float(a3), "linear_s_per_row": float(b3)},
        "sources": {
            "r0216": {"rows": R0216_ROWS, "exact_search_s": R0216_EXACT_SEARCH_S,
                      "query_block": QUERY_BLOCK, "search_block": SEARCH_BLOCK,
                      "implementation_matched": True},
            "r0233": {"rows": R0233_ROWS, "exact_search_s": R0233_EXACT_SEARCH_S,
                      "query_block": R0233_QUERY_BLOCK,
                      "search_block": R0233_SEARCH_BLOCK,
                      "implementation_matched": R0233_IS_IMPLEMENTATION_MATCHED},
        },
        "registered_check": (
            "the build node measures exact_search_s at 4,000,000 rows with the "
            "R0216 builder at the R0216 block sizes and publishes it against "
            "this interval. A measurement outside the interval is published as "
            "outside it; no model is refitted after the fact."
        ),
        "what_this_is_not": (
            "not a confidence interval. The four models are not draws from a "
            "distribution and the interval carries no coverage claim; it is the "
            "range spanned by four stated readings of two sealed walls, one of "
            "which (R0233) is a different implementation."
        ),
    }


def other_predictions() -> dict[str, Any]:
    """Pre-registered non-wall quantities, each falsifiable at seal time."""
    scale = float(ROWS) / float(R0216_ROWS)
    substrate_bytes = ROWS * DIMENSION * 4 + 128
    edges_point = int(round(R0216_DIRECTED_EDGES * scale))
    npz_point = int(round(R0216_EDGES_NPZ_BYTES * scale))
    return {
        "label": "prediction",
        "substrate_bytes": {
            "point": int(substrate_bytes),
            "basis": "exactly 4,000,000 x 384 x 4 B plus the 128 B .npy header; "
                     "R0216's 2,000,000-row substrate is 3,072,000,128 B",
            "is_deterministic": True,
        },
        "directed_edges": {
            "point": edges_point,
            "interval": [int(round(edges_point * 0.95)), int(round(edges_point * 1.05))],
            "basis": "R0216 sealed 48,344,648 directed edges over 2,000,000 rows "
                     "(24.172324 mean degree); the fuzzy symmetrisation of a k15 "
                     "graph is close to linear in N at fixed k",
        },
        "edges_npz_bytes": {
            "point": npz_point,
            "interval": [int(round(npz_point * 0.95)), int(round(npz_point * 1.05))],
            "basis": "12 B per directed edge, uncompressed (R0216: 580,136,932 B "
                     "over 48,344,648 edges)",
        },
        "peak_allocated_bytes": {
            "point": int(PREDICTED_PEAK_ALLOCATED_BYTES),
            "interval": [int(PREDICTED_PEAK_ALLOCATED_BYTES * 0.95),
                         int(DEVICE_BUDGET_BYTES)],
            "basis": "Xt 6,144,000,000 B + one 16,384 x 100,000 fp32 tile "
                     "6,553,600,000 B + R0216's measured 91,882,880 B of top-k "
                     "temporaries",
        },
        "fuzzy_s": {
            "point": float(R0216_FUZZY_S * scale),
            "interval": [float(R0216_FUZZY_S * scale * 0.8),
                         float(R0216_FUZZY_S * scale * 3.0)],
            "basis": "R0216 6.974188329651952 s at 2M; the interval is wide "
                     "upward because umap.fuzzy_simplicial_set's smooth-knn "
                     "solve is numba-parallel and its scaling here is untested",
        },
        "node_wall_s": {
            "interval": [900.0, 1900.0],
            "basis": "the exact_search_s interval, plus 2x R0216's non-search "
                     "wall (382.67656675493345 - 112.18968716729432 = "
                     "270.48687958764 s) at +/-50%, plus 20-240 s for the new "
                     "independent CPU probe, which has no precedent to scale "
                     "from",
        },
    }


def back_check_at_2m(
    *, quadratic_s_per_pair: float, linear_s_per_row: float,
    measured_2m_s: float = R0216_EXACT_SEARCH_S, rel_tol: float = BACK_CHECK_REL_TOL,
) -> dict[str, Any]:
    """Does a law fitted at 4M reproduce R0216's sealed 2M wall?

    This is the round's registered falsifiable check on its own extrapolation.
    It is scored and published either way: a miss does not fail the build (the
    4M measurement stands on its own), it withdraws the law's licence to price
    any OTHER N.
    """
    predicted = predict_search_s(quadratic_s_per_pair=quadratic_s_per_pair,
                                 linear_s_per_row=linear_s_per_row,
                                 rows=R0216_ROWS)
    measured = float(measured_2m_s)
    if measured <= 0:
        raise Round0261Error("R0261 back-check needs a positive measured wall")
    rel = (predicted - measured) / measured
    holds = abs(rel) <= float(rel_tol)
    return {
        "rows": R0216_ROWS,
        "predicted_s": float(predicted),
        "measured_s": measured,
        "relative_error": float(rel),
        "registered_rel_tol": float(rel_tol),
        "holds": bool(holds),
        "if_it_fails": (
            "the two-term law is not licensed to price any N other than the one "
            "measured here; the 4,000,000-row measurement itself is unaffected."
        ),
    }


def price_other_rungs(
    *, quadratic_s_per_pair: float, linear_s_per_row: float,
    rungs: Sequence[int] = (2_000_000, 3_000_000, 4_000_000, 5_000_000,
                            6_250_000, 8_000_000),
) -> dict[str, Any]:
    """Search-wall price of an exact k15 graph at other candidate second-Ns.

    Every entry except the one at `ROWS` carries `label: prediction`, per the
    standing rule that no measured quantity is carried to another N without a
    measurement there or an explicit prediction label.
    """
    out: dict[str, Any] = {}
    for rung in rungs:
        out[str(int(rung))] = {
            "rows": int(rung),
            "predicted_exact_search_s": predict_search_s(
                quadratic_s_per_pair=quadratic_s_per_pair,
                linear_s_per_row=linear_s_per_row, rows=int(rung)),
            "substrate_bytes": int(rung) * DIMENSION * 4 + 128,
            "predicted_edges_npz_bytes": int(round(
                R0216_EDGES_NPZ_BYTES * (float(rung) / float(R0216_ROWS)))),
            "label": "measurement" if int(rung) == int(ROWS) else "prediction",
        }
    return out


# --------------------------------------------------------------------------- #
# 2. the guards, each shipped as a function a control can plant into
# --------------------------------------------------------------------------- #


def validate_composition(counts: Mapping[str, int]) -> dict[str, Any]:
    """Fail closed unless the assembled universe is exactly the 4M mix."""
    total = sum(int(v) for v in counts.values())
    if total != ROWS:
        raise Round0261Error(f"substrate has {total} rows, registered {ROWS}")
    observed: dict[str, Any] = {}
    for name, want in COMPOSITION:
        got = int(counts.get(name, 0))
        if got != want:
            raise Round0261Error(f"{name}: assembled {got} rows, registered {want}")
        observed[name] = {"rows": got, "share": got / ROWS,
                          "registered_share": TARGET_SHARES[name]}
    return observed


#: The span rule R0216 earned: an oversample-then-stop-at-quota loop sampled only
#: the leading 90-94% of each corpus and the defect was invisible in the output.
MIN_SHARD_COVERAGE = 0.999


def assert_shard_span(*, corpus: str, shards_touched: int,
                      shards_total: int) -> dict[str, Any]:
    """Refuse a selection that did not span the corpus."""
    if int(shards_total) <= 0:
        raise Round0261Error(f"{corpus}: no shards to span")
    coverage = int(shards_touched) / int(shards_total)
    if coverage < MIN_SHARD_COVERAGE:
        raise Round0261Error(
            f"{corpus}: selection touched {shards_touched}/{shards_total} shards "
            f"({coverage:.2%}); the registered law requires the sample to span "
            "every shard"
        )
    return {"shards_touched": int(shards_touched), "shards_total": int(shards_total),
            "coverage": float(coverage), "floor": MIN_SHARD_COVERAGE}


def degree_census(sources: np.ndarray, *, rows: int = ROWS) -> dict[str, Any]:
    """Out-degree of every row of the symmetrised fuzzy graph, and the tripwire.

    This is the shipped path the build node uses; the R0215 tripwire control
    plants an edgeless row into THIS function rather than into a copy of it.
    """
    src = np.asarray(sources)
    if src.ndim != 1:
        raise Round0261Error("R0261 degree census needs a 1-D source array")
    if src.size and (int(src.min()) < 0 or int(src.max()) >= int(rows)):
        raise Round0261Error("R0261 degree census saw an out-of-range source id")
    deg = np.bincount(src.astype(np.int64), minlength=int(rows))
    return {
        "zero_degree_rows": int((deg == 0).sum()),
        "min": int(deg.min()),
        "median": float(np.median(deg)),
        "mean": float(deg.mean()),
        "max": int(deg.max()),
    }


def validate_exact_graph(
    *, degrees: Mapping[str, Any], gating_recall: Mapping[str, float],
    builder_recall: Mapping[str, float], edges: int,
) -> dict[str, Any]:
    """Assert exactness on the INDEPENDENT probe, then assert the R0215 tripwire.

    `validate_graph` is R0216's shipped judge, reused unchanged so the two rungs
    are qualified by one instrument. What changes is *which* probe is handed to
    it: R0216 handed it a GPU probe that shared the builder's accumulator, which
    review-0216-01 established is not independent. Here the gating probe is the
    plain-NumPy CPU pass, and the builder's own GPU probe is reported beside it
    with `is_independent: false`.
    """
    checks = validate_graph(degrees=degrees, recall=gating_recall, edges=edges)
    program = {}
    for label, recall in (("gating_cpu_probe", gating_recall),
                          ("builder_gpu_probe", builder_recall)):
        mean = float(recall["mean_recall_at_k"])
        if mean < PROGRAM_RECALL_FLOOR:
            raise Round0261Error(
                f"{label} mean recall {mean:.6f} is below the program floor "
                f"{PROGRAM_RECALL_FLOOR} from plan-minilm-100m-v2.md"
            )
        program[label] = {
            "mean_recall_at_k": mean,
            "p10_recall_at_k": float(recall["p10_recall_at_k"]),
            "clears_program_floor": True,
            "program_floor": PROGRAM_RECALL_FLOOR,
        }
    checks["program_floor_clearance"] = program
    checks["gating_probe"] = "independent CPU brute-force pass"
    checks["builder_probe_is_independent"] = GPU_PROBE_IS_INDEPENDENT
    checks["tie_tolerance"] = TIE_TOLERANCE
    return checks


def assert_prediction_precedes_build(
    *, prediction: Mapping[str, Any], build_started_unix: float,
) -> dict[str, Any]:
    """Refuse a build whose price prediction was not sealed first.

    Two refusal branches, both controlled:

    1. the prediction's `sealed_at_unix` is not strictly before the build node's
       start, so the "prediction" could have been written knowing the answer;
    2. the prediction artifact carries a measured 4M quantity, so it is not a
       prediction at all.
    """
    sealed_at = prediction.get("sealed_at_unix")
    if not isinstance(sealed_at, (int, float)):
        raise Round0261Error(
            "R0261 refuses to build: the price prediction carries no seal time"
        )
    if not float(sealed_at) < float(build_started_unix):
        raise Round0261Error(
            "R0261 refuses to build: the price prediction was not sealed before "
            f"the build started ({sealed_at} !< {build_started_unix})"
        )
    forbidden = sorted(k for k in ("measured_exact_search_s", "measured_node_wall_s",
                                   "measured_directed_edges", "measured")
                       if k in prediction)
    if forbidden:
        raise Round0261Error(
            "R0261 refuses to build: the price prediction binds measured 4M "
            f"quantities {forbidden}, so it is not a prediction"
        )
    return {
        "prediction_sealed_at_unix": float(sealed_at),
        "build_started_unix": float(build_started_unix),
        "seconds_between": float(build_started_unix) - float(sealed_at),
        "prediction_precedes_build": True,
        "no_measured_quantity_is_bound_to_the_prediction": True,
    }


def score_prediction(*, prediction: Mapping[str, Any], measured_s: float) -> dict[str, Any]:
    """Publish the measurement against the registered interval, without refitting."""
    low, high = (float(v) for v in prediction["interval_s"])
    point = float(prediction["point_estimate_s"])
    measured = float(measured_s)
    return {
        "measured_exact_search_s": measured,
        "registered_interval_s": [low, high],
        "registered_point_estimate_s": point,
        "inside_the_registered_interval": bool(low <= measured <= high),
        "relative_error_against_point": (measured - point) / point,
        "closest_model": min(
            prediction["models"].items(), key=lambda kv: abs(float(kv[1]) - measured)
        )[0],
        "models": dict(prediction["models"]),
    }


# --------------------------------------------------------------------------- #
# 3. the independent CPU recall probe
# --------------------------------------------------------------------------- #


def cpu_exact_topk(
    X: np.ndarray, probe: np.ndarray, *, k: int = GRAPH_K,
    block: int = CPU_PROBE_BLOCK, poll: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force exact top-k for `probe` rows, in plain NumPy on the host.

    Deliberately *not* the builder's algorithm: a blocked `float32` GEMM whose
    running top-k is maintained by `np.argpartition` over a concatenated
    candidate buffer, reduced in a different order from the GPU's `topk` +
    `argsort` merge. Self-matches are removed after the fact, exactly as the
    builder does, so the comparison is like-for-like on content.
    """
    Xa = np.asarray(X)
    rows = int(Xa.shape[0])
    q = np.ascontiguousarray(Xa[probe], dtype=np.float32)
    width = k + 1
    best_s = np.full((q.shape[0], width), -np.inf, dtype=np.float32)
    best_i = np.full((q.shape[0], width), -1, dtype=np.int64)
    for start in range(0, rows, int(block)):
        stop = min(start + int(block), rows)
        sims = q @ np.asarray(Xa[start:stop], dtype=np.float32).T
        take = min(width, stop - start)
        part = np.argpartition(-sims, take - 1, axis=1)[:, :take]
        cand_s = np.take_along_axis(sims, part, axis=1)
        cand_i = part.astype(np.int64) + start
        merged_s = np.concatenate([best_s, cand_s], axis=1)
        merged_i = np.concatenate([best_i, cand_i], axis=1)
        order = np.argsort(-merged_s, axis=1, kind="stable")[:, :width]
        best_s = np.take_along_axis(merged_s, order, axis=1)
        best_i = np.take_along_axis(merged_i, order, axis=1)
        if poll is not None:
            poll(f"R0261 CPU probe block {start}")
    ids = np.empty((q.shape[0], k), dtype=np.int64)
    cos = np.empty((q.shape[0], k), dtype=np.float32)
    for n, row in enumerate(np.asarray(probe, dtype=np.int64)):
        keep = [(int(i), float(s)) for i, s in zip(best_i[n], best_s[n])
                if int(i) != int(row)][:k]
        if len(keep) < k:
            raise Round0261Error(f"CPU probe row {row} found {len(keep)} neighbours")
        ids[n] = [i for i, _ in keep]
        cos[n] = [s for _, s in keep]
    return ids, cos


def score_cpu_probe(
    *, truth_ids: np.ndarray, truth_cos: np.ndarray,
    builder_ids: np.ndarray, builder_cos: np.ndarray, k: int = GRAPH_K,
) -> dict[str, Any]:
    """Strict containment and tie-aware validity of the builder's rows.

    Both are R0220's shipped estimators, reused unchanged. Strict is a set test
    over ids and is immune to float noise; tie-aware is a value test against the
    truth's k-th cosine and is what gates, because an fp32 tie at rank 15 is a
    property of the data, not a defect in the builder.
    """
    from basemap.round0220_cuvs_qualification import (
        strict_containment_rows, summarize, tie_aware_rows,
    )

    strict = strict_containment_rows(np.asarray(builder_ids), np.asarray(truth_ids))
    kth = np.asarray(truth_cos, dtype=np.float64)[:, k - 1]
    tie = tie_aware_rows(np.asarray(builder_cos), np.asarray(builder_ids), kth,
                         k=k, tolerance=TIE_TOLERANCE)
    return {
        "probe_rows": int(np.asarray(truth_ids).shape[0]),
        "strict": summarize(strict, label="R0261 CPU probe strict"),
        "tie_aware": summarize(tie, label="R0261 CPU probe tie-aware"),
        "tie_tolerance": TIE_TOLERANCE,
        "gating_estimator": "tie_aware",
        "why_tie_aware_gates": (
            "the CPU probe reduces 384 fp32 products in a different order from "
            "the GPU builder, so a pair of neighbours whose cosines agree to "
            "fp32 can swap at rank 15. That is an ordering of equals, not a "
            "missed neighbour. Strict containment is published beside it."
        ),
    }


def gating_recall_block(scored: Mapping[str, Any]) -> dict[str, float]:
    """The two numbers `validate_exact_graph` gates on, named as R0216 names them."""
    tie = scored["tie_aware"]
    return {"mean_recall_at_k": float(tie["mean"]),
            "p10_recall_at_k": float(tie["p10"])}


# --------------------------------------------------------------------------- #
# 4. the positive controls
# --------------------------------------------------------------------------- #


def _plant(label: str, fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run a planted defect through a SHIPPED function and require a refusal."""
    try:
        fn(*args, **kwargs)
    except (Round0261Error, Round0216Error) as error:
        return {"plant": label, "refused": True,
                "error": f"{type(error).__name__}: {error}"}
    raise Round0261Error(
        f"R0261 positive control {label!r} was ACCEPTED by the shipped path; "
        "the guard is untested at its only job"
    )


def _accept(label: str, fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """The negative arm: the clean input must pass, or the guard is a rejector."""
    value = fn(*args, **kwargs)
    return {"control": label, "accepted": True,
            "returned_keys": sorted(value.keys()) if isinstance(value, Mapping) else None}


def _clean_recall() -> dict[str, float]:
    return {"mean_recall_at_k": 1.0, "p10_recall_at_k": 1.0}


def _clean_degrees(zero: int = 0) -> dict[str, Any]:
    return {"zero_degree_rows": int(zero), "min": 5, "median": 19.0,
            "mean": 24.17, "max": 1394}


def assert_degree_zero_tripwire_controls() -> dict[str, Any]:
    """Plant edgeless rows into the shipped tripwire, both entry points.

    R0215's mechanism, planted twice: once as a degree census over a real source
    array with a row that has no outgoing edges, and once as the census block
    the judge receives. `MAX_ZERO_DEGREE_ROWS` is `0`, so `1` must refuse.
    """
    # A four-row graph in which row 2 emits nothing.
    sources = np.array([0, 0, 1, 1, 3, 3], dtype=np.int32)
    census = degree_census(sources, rows=4)
    if int(census["zero_degree_rows"]) != 1:
        raise Round0261Error(
            "R0261 degree census did not see the planted edgeless row: "
            f"{census['zero_degree_rows']}"
        )
    one = _plant("one_edgeless_row_reaches_the_judge", validate_exact_graph,
                 degrees=census, gating_recall=_clean_recall(),
                 builder_recall=_clean_recall(), edges=int(sources.size))
    many = _plant("v1_scale_edgeless_population", validate_exact_graph,
                  degrees=_clean_degrees(zero=2_779_481),
                  gating_recall=_clean_recall(), builder_recall=_clean_recall(),
                  edges=48_344_648)
    clean = _accept("zero_edgeless_rows_passes", validate_exact_graph,
                    degrees=_clean_degrees(zero=0), gating_recall=_clean_recall(),
                    builder_recall=_clean_recall(), edges=48_344_648)
    return {
        "guard": "degree_zero_tripwire",
        "shipped_entry_points": ["degree_census", "validate_exact_graph"],
        "census_over_a_planted_graph": census,
        "plants": [one, many],
        "negative_arm": clean,
        "max_zero_degree_rows": MAX_ZERO_DEGREE_ROWS,
        "defects_planted": 2,
    }


def assert_recall_floor_controls() -> dict[str, Any]:
    """Plant sub-floor recall into the shipped judge, on both floors and bars."""
    plants = [
        _plant("mean_below_the_exact_floor", validate_exact_graph,
               degrees=_clean_degrees(), builder_recall=_clean_recall(),
               gating_recall={"mean_recall_at_k": MEAN_RECALL_FLOOR - 1e-6,
                              "p10_recall_at_k": 1.0},
               edges=48_344_648),
        _plant("p10_below_the_exact_floor", validate_exact_graph,
               degrees=_clean_degrees(), builder_recall=_clean_recall(),
               gating_recall={"mean_recall_at_k": 1.0,
                              "p10_recall_at_k": P10_RECALL_FLOOR - 1e-6},
               edges=48_344_648),
        _plant("gating_probe_below_the_program_bar", validate_exact_graph,
               degrees=_clean_degrees(), builder_recall=_clean_recall(),
               gating_recall={"mean_recall_at_k": PROGRAM_RECALL_FLOOR - 1e-6,
                              "p10_recall_at_k": 0.0},
               edges=48_344_648),
        _plant("builder_probe_below_the_program_bar", validate_exact_graph,
               degrees=_clean_degrees(), gating_recall=_clean_recall(),
               builder_recall={"mean_recall_at_k": PROGRAM_RECALL_FLOOR - 1e-6,
                               "p10_recall_at_k": 1.0},
               edges=48_344_648),
        _plant("no_edges_at_all", validate_exact_graph, degrees=_clean_degrees(),
               gating_recall=_clean_recall(), builder_recall=_clean_recall(),
               edges=0),
    ]
    boundary = _accept("exactly_at_both_exact_floors", validate_exact_graph,
                       degrees=_clean_degrees(),
                       gating_recall={"mean_recall_at_k": MEAN_RECALL_FLOOR,
                                      "p10_recall_at_k": P10_RECALL_FLOOR},
                       builder_recall=_clean_recall(), edges=48_344_648)
    return {
        "guard": "exact_recall_floors_and_program_bar",
        "floors": {"mean": MEAN_RECALL_FLOOR, "p10": P10_RECALL_FLOOR,
                   "program": PROGRAM_RECALL_FLOOR},
        "plants": plants,
        "negative_arm": boundary,
        "defects_planted": len(plants),
        "note": ("the builder's GPU probe is checked against the program bar "
                 "only. It cannot gate the exact floors because it shares the "
                 "builder's accumulator (review-0216-01)."),
    }


def assert_selection_law_controls() -> dict[str, Any]:
    """Plant a prefix selection and a wrong mix into the shipped selection guards."""
    plants = [
        _plant("fineweb_prefix_at_94_percent", assert_shard_span,
               corpus="fineweb", shards_touched=93, shards_total=99),
        _plant("code_corpus_touched_one_shard", assert_shard_span,
               corpus="code", shards_touched=1, shards_total=10),
        _plant("composition_short_by_one_row", validate_composition,
               {name: (n - 1 if i == 0 else n)
                for i, (name, n) in enumerate(COMPOSITION)}),
        _plant("composition_rebalanced_to_the_same_total", validate_composition,
               {COMPOSITION[0][0]: COMPOSITION[0][1] + 1,
                COMPOSITION[1][0]: COMPOSITION[1][1] - 1,
                COMPOSITION[2][0]: COMPOSITION[2][1],
                COMPOSITION[3][0]: COMPOSITION[3][1]}),
    ]
    clean = [
        _accept("full_span_passes", assert_shard_span, corpus="fineweb",
                shards_touched=99, shards_total=99),
        _accept("registered_mix_passes", validate_composition,
                {name: n for name, n in COMPOSITION}),
    ]
    return {
        "guard": "selection_law",
        "plants": plants,
        "negative_arms": clean,
        "defects_planted": len(plants),
        "min_shard_coverage": MIN_SHARD_COVERAGE,
    }


def assert_ordering_controls() -> dict[str, Any]:
    """Plant BOTH refusal branches of the prediction-precedes-build guard.

    review-0260-01 §D.2 and §K: R0260's ordering guard shipped with neither
    refusal branch exercised while nineteen tests covered lesser claims. This
    control exists so that finding is not repeated in the same shape.
    """
    good = {"sealed_at_unix": 1000.0, "interval_s": [1.0, 2.0]}
    plants = [
        _plant("prediction_sealed_after_the_build_started",
               assert_prediction_precedes_build,
               prediction={"sealed_at_unix": 2000.0}, build_started_unix=1000.0),
        _plant("prediction_sealed_at_exactly_the_build_start",
               assert_prediction_precedes_build,
               prediction={"sealed_at_unix": 1000.0}, build_started_unix=1000.0),
        _plant("prediction_carries_no_seal_time",
               assert_prediction_precedes_build,
               prediction={"interval_s": [1.0, 2.0]}, build_started_unix=1000.0),
        _plant("prediction_binds_a_measured_4m_wall",
               assert_prediction_precedes_build,
               prediction={"sealed_at_unix": 1.0,
                           "measured_exact_search_s": 434.0},
               build_started_unix=1000.0),
    ]
    clean = _accept("a_genuine_prediction_passes", assert_prediction_precedes_build,
                    prediction=good, build_started_unix=1001.0)
    #: Which refusal BRANCH each plant actually took, read off the message the
    #: shipped function raised. review-0260-01 §D.2's finding was that neither
    #: branch was exercised, so "both branches" must be a count over refusals,
    #: never a declared literal.
    branch_of = {"not sealed before": "not_sealed_first",
                 "no seal time": "not_sealed_first",
                 "not a prediction": "binds_a_measured_quantity"}
    branches_hit = sorted({
        name for plant in plants for text, name in branch_of.items()
        if text in str(plant["error"])
    })
    return {
        "guard": "prediction_precedes_build",
        "refusal_branches": ["not_sealed_first", "binds_a_measured_quantity"],
        "branches_exercised": branches_hit,
        "plants": plants,
        "negative_arm": clean,
        "defects_planted": len(plants),
        "both_branches_exercised": branches_hit == [
            "binds_a_measured_quantity", "not_sealed_first"],
    }


def assert_back_check_controls() -> dict[str, Any]:
    """Prove the 2M back-check can fail, by fitting laws that miss it."""
    honest = two_term_fit(rows_a=R0216_ROWS, seconds_a=R0216_EXACT_SEARCH_S,
                          rows_b=R0233_ROWS, seconds_b=R0233_EXACT_SEARCH_S)
    passing = back_check_at_2m(**honest)
    if not passing["holds"]:
        raise Round0261Error(
            "R0261 back-check control: the law fitted THROUGH the 2M point does "
            "not reproduce it, so the check is broken rather than strict"
        )
    wrong_high = back_check_at_2m(
        quadratic_s_per_pair=honest["quadratic_s_per_pair"] * 1.5,
        linear_s_per_row=honest["linear_s_per_row"])
    wrong_low = back_check_at_2m(
        quadratic_s_per_pair=honest["quadratic_s_per_pair"] * 0.5,
        linear_s_per_row=honest["linear_s_per_row"])
    if wrong_high["holds"] or wrong_low["holds"]:
        raise Round0261Error(
            "R0261 back-check accepted a law 50% wrong in the quadratic term; "
            f"high={wrong_high['relative_error']} low={wrong_low['relative_error']}"
        )
    return {
        "guard": "back_check_at_2m",
        "registered_rel_tol": BACK_CHECK_REL_TOL,
        "law_through_the_point_passes": passing,
        "plants": [
            {"plant": "quadratic_term_inflated_50_percent", "refused": True,
             "relative_error": wrong_high["relative_error"], "holds": False},
            {"plant": "quadratic_term_halved", "refused": True,
             "relative_error": wrong_low["relative_error"], "holds": False},
        ],
        "defects_planted": 2,
        "note": ("this control is arithmetic on sealed walls only; it reads no "
                 "4M measurement and runs before the build."),
    }


def assert_prediction_scoring_controls() -> dict[str, Any]:
    """Prove `score_prediction` can report OUTSIDE, so an inside verdict means something."""
    prediction = cost_prediction()
    low, high = (float(v) for v in prediction["interval_s"])
    inside = score_prediction(prediction=prediction,
                              measured_s=prediction["point_estimate_s"])
    below = score_prediction(prediction=prediction, measured_s=low * 0.5)
    above = score_prediction(prediction=prediction, measured_s=high * 2.0)
    if not inside["inside_the_registered_interval"]:
        raise Round0261Error("R0261 scoring control: the point estimate reads outside")
    if below["inside_the_registered_interval"] or above["inside_the_registered_interval"]:
        raise Round0261Error("R0261 scoring control: a value outside reads inside")
    return {
        "guard": "score_prediction",
        "plants": [
            {"plant": "half_the_lower_bound", "reads_inside": False,
             "measured_s": below["measured_exact_search_s"]},
            {"plant": "twice_the_upper_bound", "reads_inside": False,
             "measured_s": above["measured_exact_search_s"]},
        ],
        "negative_arm": {"control": "the_point_estimate_reads_inside",
                         "accepted": True},
        "defects_planted": 2,
    }


def all_controls() -> dict[str, Any]:
    """Every guard this round ships, each with its plants run through the shipped path."""
    blocks = {
        "degree_zero_tripwire": assert_degree_zero_tripwire_controls(),
        "recall_floors": assert_recall_floor_controls(),
        "selection_law": assert_selection_law_controls(),
        "ordering": assert_ordering_controls(),
        "back_check": assert_back_check_controls(),
        "prediction_scoring": assert_prediction_scoring_controls(),
    }
    refusals = sum(len(block.get("plants") or ()) for block in blocks.values())
    return {
        "schema": "round0261-positive-controls-v1",
        "guards": blocks,
        "guards_shipped": len(blocks),
        "defects_planted": sum(int(b["defects_planted"]) for b in blocks.values()),
        "refusals_recorded": refusals,
        "guards_with_at_least_one_plant": sum(
            1 for block in blocks.values() if int(block["defects_planted"]) > 0),
        "every_guard_has_a_plant": all(
            int(block["defects_planted"]) > 0 for block in blocks.values()),
        "what_a_plant_proves": (
            "the defect was fed to the SHIPPED function -- degree_census, "
            "validate_exact_graph, assert_shard_span, validate_composition, "
            "assert_prediction_precedes_build, back_check_at_2m, "
            "score_prediction -- and the shipped function refused it. A guard "
            "whose test suite contains no failing input is untested at its only "
            "job (AGENT_STARTUP.md; the shape that let a SIGKILL ship under a "
            "'no signal delivered' receipt in R0238)."
        ),
    }


__all__ = [
    "BACK_CHECK_REL_TOL",
    "CAPABILITY",
    "COMPOSITION",
    "CPU_PROBE_BLOCK",
    "CPU_PROBE_ROWS",
    "CPU_PROBE_SEED",
    "DEVICE_BUDGET_BYTES",
    "DIMENSION",
    "EXCLUDED_SHARDS",
    "GPU_PROBE_IS_INDEPENDENT",
    "GPU_PROBE_ROWS",
    "GPU_PROBE_SEED",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "KNOWN_TRAILING_FRAGMENTS",
    "MAX_ZERO_DEGREE_ROWS",
    "MEAN_RECALL_FLOOR",
    "MIN_MEM_AVAILABLE_BYTES",
    "MIN_SHARD_COVERAGE",
    "NESTED_IN_R0216",
    "NODE_ANON_BUDGET_BYTES",
    "P10_RECALL_FLOOR",
    "PREDICTED_PEAK_ALLOCATED_BYTES",
    "PREDICTION_SCHEMA",
    "PRICE_CAPABILITY",
    "PRICE_SCHEMA",
    "PROGRAM_RECALL_FLOOR",
    "QUERY_BLOCK",
    "R0216_DIRECTED_EDGES",
    "R0216_EDGES_NPZ_BYTES",
    "R0216_EXACT_SEARCH_S",
    "R0216_FUZZY_S",
    "R0216_PEAK_ALLOCATED_BYTES",
    "R0216_ROWS",
    "R0216_SUBSTRATE_BYTES",
    "R0216_TOTAL_WALL_S",
    "R0233_EXACT_SEARCH_S",
    "R0233_IS_IMPLEMENTATION_MATCHED",
    "R0233_PEAK_ALLOCATED_BYTES",
    "R0233_QUERY_BLOCK",
    "R0233_ROWS",
    "R0233_SEARCH_BLOCK",
    "RAW_FORMAT",
    "ROUND_ID",
    "ROWS",
    "ROW_POLICY",
    "SEARCH_BLOCK",
    "SELECTION_SEED",
    "SUBSTRATE_SCHEMA",
    "TARGET_SHARES",
    "TIE_TOLERANCE",
    "TRAILING_FRAGMENT_POLICY",
    "ZERO_ROW_POLICY",
    "Round0261Error",
    "all_controls",
    "assert_back_check_controls",
    "assert_degree_zero_tripwire_controls",
    "assert_ordering_controls",
    "assert_prediction_precedes_build",
    "assert_prediction_scoring_controls",
    "assert_recall_floor_controls",
    "assert_selection_law_controls",
    "assert_shard_span",
    "back_check_at_2m",
    "cost_prediction",
    "cpu_exact_topk",
    "degree_census",
    "gating_recall_block",
    "other_predictions",
    "predict_search_s",
    "price_other_rungs",
    "resolve_shard_rows",
    "score_cpu_probe",
    "score_prediction",
    "two_term_fit",
    "validate_composition",
    "validate_exact_graph",
]
