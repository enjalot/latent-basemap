"""R0244 — the three prerequisites review-0243-01 §8 put in front of a map.

`basemap/round0244_guard.py` carries the fourth (the host watchdog). This
module carries:

* **the edge list as a sampling distribution** (§8.4). `30 GB` across three
  memmappable `.npy` arrays plus a header is the right *shape*; nothing has
  loaded it as a distribution over `875,131,479.5054033` total weight. A
  trainer never materialises a `2.5e9`-entry CDF (`20 GB` of anonymous memory),
  so the demonstration here is the two-level scheme it would actually use — a
  block CDF over `float64` block sums plus a within-block CDF built on demand —
  and its output is checked against the distribution it is supposed to be, with
  a mis-sampler as the positive control.
* **R0228's displacement DiD, pre-registered** (§8.1). The inference rule, the
  populations, the null, the permutation design and the decision map are fixed
  here, in module constants at the release commit, before any displacement
  exists. What is NOT here is a displacement measurement: R0228's statistic
  consumes trained `(n, 2)` map coordinates and this round trains no map. The
  cost of the round that can run it is computed rather than asserted.
* **near-duplicate classification for cluster `168`** (§8.2). Tie tolerance
  certifies equal *cosine to the query*, not equal *location* — two rows at
  identical cosine to a query lie on a cone, not on each other. So the decisive
  quantity is the cosine between the missed true neighbour and the substitute
  that tied it, published beside the two texts.

Nothing here touches the GPU, starts a child, or signals anything.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from basemap.round0228_geometry import (  # noqa: F401  (re-exported deliberately)
    DENSITY_DECILES,
    SCATTER_DEFINITION,
    SCATTER_SAMPLE_ROWS,
    density_matched_control,
)
from basemap.round0229_quality_contract import (  # noqa: F401
    exact_displacement_permutation,
    holm_bonferroni,
)
from basemap.round0238_rung5 import GRAPH_K
from basemap.round0244_guard import ROUND_ID, ROWS, Round0244Error
from basemap.round0247_registry import clamp as _clamp_r0247
from basemap.round0247_registry import override_records as _override_records

# --------------------------------------------------------------------------- #
# D. the edge list as a sampling distribution
# --------------------------------------------------------------------------- #
#: R0243's sealed symmetrised graph, as published in `fuzzy-graph.json`. Bound
#: here so "the array I loaded is the array R0243 sealed" is a check and not a
#: path convention.
R0243_DIRECTED_EDGES = 2_511_103_254
R0243_TOTAL_WEIGHT = 875131479.5054033
R0243_WEIGHT_MAX = 1.0
R0243_WEIGHT_MIN = 1.401298464324817e-45
R0243_ENTRIES_AT_OR_ABOVE_ONE = 164_364_036

#: Digests published in `result-0243-2026-08-10.md`'s Outputs table and in
#: R0238's sealed `substrate.json`, bound here so this round reads the same
#: bytes the reviewed rounds sealed. `strict-c400.f64.npy` is the vector
#: review-0243-01 section 7 found missing from every Inputs table.
R0238_REACHABILITY_VECTOR_SHA256 = (
    "2cce1f2abc2d404a41d2877340c59812e2a15631528a3ff6b88edf188894261c"
)
R0238_PROVENANCE_SHA256 = (
    "35bc0c65f365daba4edecde43553f40c5aa9a7f6c13d11854fa68fd240caa87c"
)
R0242_PROBE_STRICT_RECALL_SHA256 = (
    "4e54b4b636399566f9132315bdfd701d31a0c4a26b41df99be4f8a051e77ac3c"
)
R0242_PROBE_BUILDER_MISSING_SHA256 = (
    "1b405d882f8a261f1572bbf65a0b546c76d2a962d1ed9aba4a9def063a522b15"
)
R0242_PROBE_CLUSTER_SHA256 = (
    "04f0b05c5b8969b635b7a9c67d95ba55c1995dfd57a84da64b9e3e5889f022e4"
)
R0243_TIE_AWARE_BUILDER_MISSING_SHA256 = (
    "4dae9a12026d6e29d58947a7a7a43ec17c8f69d922eb9d567b0d7d16f6a63450"
)
R0243_EDGES_SRC_SHA256 = (
    "dfb3f4c25f8024614e0738456fc02dd5ec8e5589020ea3eb2f9e977c7fca52be"
)
R0243_EDGES_DST_SHA256 = (
    "cfd3011055e9f6f1a48f2c6897f3d4c1bfc4417fb59537fd31289563766d87ca"
)
R0243_EDGES_WTS_SHA256 = (
    "ea7cffb2c8db3bb27f1cd4890b6a9b8b0bb0facc58c2240c984119c38eb16be8"
)
R0243_EDGES_HEADER_SHA256 = (
    "eaacff947fcff8e53ed00e72943239ca31ff6252047cc449466695fc53aa2e53"
)
R0243_FUZZY_RECEIPT_SHA256 = (
    "7d6c0a24556f9127116fd8c0ec1a7cffa3e80a2fa228f3f87d5df0be7f75ec14"
)

#: One block is `1,048,576` edges = `4 MiB` of `float32`. `2,395` blocks at
#: `100M`. The block sums are `float64` and fit in `19 KB`, which is the whole
#: point: the trainer holds the index, never the distribution.
SAMPLER_BLOCK_EDGES = 1 << 20
SAMPLER_DRAWS = 20_000_000
SAMPLER_SEED = 244_001
SAMPLER_WEIGHT_BINS = 20

#: Registered acceptance for the fidelity check, stated before it is computed.
SAMPLER_MAX_ABS_Z = 5.0
SAMPLER_MIN_CHI_SQUARE_P = 1e-4
SAMPLER_MAX_ANONYMOUS_BYTES = 4 * (1 << 30)
SAMPLER_MIN_DRAWS_PER_S = 100_000.0

#: UMAP's edge schedule is `n_epochs * w / w.max()`; an edge whose weight falls
#: below `w.max() / n_epochs` is never sampled in a run of `n_epochs`. That is a
#: property of THIS distribution that a trainer needs before it commits `10 h`.
SAMPLER_EPOCHS = 200

SAMPLER_NOTE = (
    "the trainer's contract with this artifact is: memmap three arrays, never "
    "materialise a 2.5e9-entry CDF (20 GB anonymous), draw edges with "
    "probability proportional to weight, and gather (src, dst) for the drawn "
    "edges. All four are exercised here, with the anonymous footprint measured "
    "rather than asserted."
)


def weight_block_profile(
    weights: np.ndarray,
    *,
    block: int = SAMPLER_BLOCK_EDGES,
    bins: int = SAMPLER_WEIGHT_BINS,
    epochs: int = SAMPLER_EPOCHS,
    abort_check: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """One streaming pass: block sums, moments, a weight histogram.

    `block_sums` is the only thing kept, and it is `float64` at one value per
    `1,048,576` edges. Everything else is a scalar or a `bins`-long vector.
    """
    array = np.asarray(weights)
    if array.ndim != 1 or array.size == 0:
        raise Round0244Error("R0244 sampler needs a non-empty 1-D weight array")
    block = int(block)
    if block <= 0:
        raise Round0244Error("R0244 sampler block must be positive")
    edges = array.size
    blocks = -(-edges // block)
    block_sums = np.zeros(blocks, dtype=np.float64)
    bin_mass = np.zeros(int(bins), dtype=np.float64)
    bin_counts = np.zeros(int(bins), dtype=np.int64)
    total_squares = 0.0
    smallest = math.inf
    largest = -math.inf
    non_finite = 0
    non_positive = 0
    epoch_floor = 1.0 / float(epochs)
    sampled_once = 0
    at_or_above_one = 0
    for index in range(blocks):
        if abort_check is not None:
            abort_check(f"R0244 weight profile block {index}")
        chunk = np.asarray(
            array[index * block:min((index + 1) * block, edges)],
            dtype=np.float64,
        )
        finite = np.isfinite(chunk)
        non_finite += int(chunk.size - finite.sum())
        non_positive += int((chunk <= 0.0).sum())
        block_sums[index] = float(chunk.sum())
        total_squares += float(np.square(chunk).sum())
        smallest = min(smallest, float(chunk.min()))
        largest = max(largest, float(chunk.max()))
        sampled_once += int((chunk >= epoch_floor).sum())
        at_or_above_one += int((chunk >= 1.0).sum())
        slot = np.clip(
            (chunk * int(bins)).astype(np.int64), 0, int(bins) - 1
        )
        bin_mass += np.bincount(slot, weights=chunk, minlength=int(bins))
        bin_counts += np.bincount(slot, minlength=int(bins))
        del chunk, finite, slot
    total = float(block_sums.sum())
    return {
        "edges": int(edges),
        "block": block,
        "blocks": int(blocks),
        "block_sums": block_sums,
        "total_weight": total,
        "sum_of_squares": float(total_squares),
        "min_weight": float(smallest),
        "max_weight": float(largest),
        "mean_weight": total / float(edges),
        "non_finite_entries": int(non_finite),
        "non_positive_entries": int(non_positive),
        "bins": int(bins),
        "bin_mass": bin_mass,
        "bin_counts": bin_counts,
        "epochs": int(epochs),
        "epoch_weight_floor": float(epoch_floor),
        "edges_sampled_at_least_once_in_a_run": int(sampled_once),
        "edges_never_sampled_in_a_run": int(edges - sampled_once),
        "entries_at_or_above_one": int(at_or_above_one),
        "block_sums_bytes": int(block_sums.nbytes),
        "note": (
            "one streaming pass; the resident state is block_sums "
            f"({block_sums.nbytes} B) plus {bins} histogram cells"
        ),
    }


#: review-0245-01 section C, the blocker. The widest abort-read gap in R0245's
#: sampler node was `24.46713631998864` s and it sat inside THIS function: the
#: `abort_check` fired once per `128` blocks, and the three unbroken bulk
#: operations after the last one (a `40M` `argsort`, a `40M` random gather from
#: a `10 GB` memmap, and a `40M` `np.unique`) ran with no cooperative-abort read
#: at all. Against the `11,767,996,416` B/s worst case that gap permits
#: `287,929,172,523` B of growth after the guard has decided to stop, on a box
#: with `123` GiB of RAM. Every loop below now polls once per unit, and the
#: three unbroken operations are chunked so a unit is bounded work rather than
#: bounded block count.
SAMPLER_POLL_CHUNK_DRAWS = 2_000_000


def _stable_counting_order(
    keys: np.ndarray,
    *,
    key_count: int,
    chunk: int,
    abort_check: Callable[[str], None] | None,
) -> np.ndarray:
    """`np.argsort(keys, kind="stable")` for small-integer keys, in polled chunks.

    A stable sort's permutation is unique, so this returns exactly what
    `argsort(kind="stable")` returns — it is a re-expression of the same
    operation whose only purpose is that it can be interrupted. The R0246
    contract test asserts the equality element-by-element, and the R0246
    sampler node reproduces R0245's sealed `distinct_edges_drawn` and all four
    sealed fidelity statistics to the digit, which is the same assertion at
    `40,000,000` draws.
    """
    counts = np.bincount(keys, minlength=int(key_count))
    cursor = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    order = np.empty(keys.size, dtype=np.int64)
    for lo in range(0, keys.size, int(chunk)):
        if abort_check is not None:
            abort_check(f"R0244 stable counting order from draw {lo}")
        hi = min(lo + int(chunk), keys.size)
        segment = keys[lo:hi]
        inside = np.argsort(segment, kind="stable")
        sorted_segment = segment[inside]
        local = np.bincount(sorted_segment, minlength=int(key_count))
        group_start = np.concatenate([[0], np.cumsum(local)[:-1]]).astype(
            np.int64
        )
        within = (
            np.arange(sorted_segment.size, dtype=np.int64)
            - group_start[sorted_segment]
        )
        order[cursor[sorted_segment] + within] = lo + inside
        cursor += local
        del segment, inside, sorted_segment, local, group_start, within
    return order


def two_level_weight_sample(
    weights: np.ndarray,
    *,
    profile: Mapping[str, Any],
    draws: int = SAMPLER_DRAWS,
    seed: int = SAMPLER_SEED,
    abort_check: Callable[[str], None] | None = None,
    poll_chunk_draws: int = SAMPLER_POLL_CHUNK_DRAWS,
) -> dict[str, Any]:
    """Draw `draws` edges with probability proportional to weight.

    Block first, from a `float64` CDF over block sums; then within the chosen
    block, from a CDF built on the `4 MiB` the block occupies and discarded.
    This is the scheme a `100M` trainer must use, so it is the scheme that gets
    demonstrated.

    The draw itself is unchanged — the random stream is consumed in exactly the
    same order and the returned `edge_index` is identical. What changed is that
    every loop polls the cooperative abort flag once per unit and no unit is
    larger than `poll_chunk_draws` draws.
    """
    array = np.asarray(weights)
    block = int(profile["block"])
    block_sums = np.asarray(profile["block_sums"], dtype=np.float64)
    total = float(profile["total_weight"])
    if total <= 0.0:
        raise Round0244Error("R0244 sampler needs positive total weight")
    #: R0247: `poll_chunk_draws` sets how many draws pass between two
    #: cooperative-abort reads, so it is a safety parameter in exactly the
    #: sense R0247 registers. A caller passing 40,000,000 restores R0245's
    #: single-chunk behaviour and with it the 24.46713631998864 s gap this
    #: family of rounds exists to close. It is clamped at the registry and the
    #: attempt is recorded on the returned evidence.
    chunk_effective, chunk_record = _clamp_r0247(
        "sampler_poll_chunk_draws", poll_chunk_draws,
        site="two_level_weight_sample(poll_chunk_draws=)",
        label="R0244 two-level weight sample",
    )
    chunk_draws = max(int(chunk_effective), 1)
    rng = np.random.default_rng(int(seed))
    block_cdf = np.cumsum(block_sums)
    block_cdf[-1] = total
    if abort_check is not None:
        abort_check("R0244 two-level sample block CDF")
    uniforms = rng.random(int(draws))
    if abort_check is not None:
        abort_check("R0244 two-level sample block uniforms")
    chosen_block = np.empty(int(draws), dtype=np.int64)
    for lo in range(0, int(draws), chunk_draws):
        if abort_check is not None:
            abort_check(f"R0244 two-level sample block choice from draw {lo}")
        hi = min(lo + chunk_draws, int(draws))
        chosen_block[lo:hi] = np.searchsorted(
            block_cdf, uniforms[lo:hi] * total, side="right"
        )
    del uniforms
    np.clip(chosen_block, 0, block_sums.size - 1, out=chosen_block)
    if abort_check is not None:
        abort_check("R0244 two-level sample block clip")
    order = _stable_counting_order(
        chosen_block, key_count=block_sums.size, chunk=chunk_draws,
        abort_check=abort_check,
    )
    sorted_blocks = chosen_block[order]
    if abort_check is not None:
        abort_check("R0244 two-level sample sorted blocks")
    boundaries = np.flatnonzero(np.diff(sorted_blocks)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [sorted_blocks.size]])
    if abort_check is not None:
        abort_check("R0244 two-level sample block boundaries")
    edge_index = np.empty(int(draws), dtype=np.int64)
    #: Distinct edges, accumulated per block group. The groups occupy disjoint
    #: half-open ranges `[index * block, index * block + block)`, so the sum of
    #: the per-group distinct counts IS the global distinct count — computed
    #: inside the polled loop instead of by a `40M` `np.unique` after it.
    distinct = 0
    for start, end in zip(starts, ends):
        index = int(sorted_blocks[start])
        if abort_check is not None:
            abort_check(f"R0244 two-level sample block {index}")
        lo = index * block
        hi = min(lo + block, array.size)
        chunk = np.asarray(array[lo:hi], dtype=np.float64)
        within = np.cumsum(chunk)
        span = float(within[-1])
        picks = np.searchsorted(
            within, rng.random(end - start) * span, side="right"
        )
        np.clip(picks, 0, chunk.size - 1, out=picks)
        edge_index[order[start:end]] = lo + picks
        distinct += int(np.unique(picks).size)
        del chunk, within, picks
    if abort_check is not None:
        abort_check("R0244 two-level sample within-block draws complete")
    sampled = np.empty(int(draws), dtype=np.float64)
    for lo in range(0, int(draws), chunk_draws):
        if abort_check is not None:
            abort_check(f"R0244 two-level sample weight gather from draw {lo}")
        hi = min(lo + chunk_draws, int(draws))
        sampled[lo:hi] = np.asarray(array[edge_index[lo:hi]], dtype=np.float64)
    if abort_check is not None:
        abort_check("R0244 two-level sample weight gather complete")
    return {
        "draws": int(draws),
        "seed": int(seed),
        "edge_index": edge_index,
        "sampled_weights": sampled,
        "chosen_block": chosen_block,
        "distinct_edges_drawn": int(distinct),
        "block_cdf_bytes": int(block_cdf.nbytes),
        "poll_chunk_draws": chunk_draws,
        "declared_poll_chunk_draws": int(poll_chunk_draws),
        "safety_overrides": [
            dict(record) for record in _override_records([chunk_record])
        ],
    }


def _chi_square_p(statistic: float, dof: int) -> float:
    from scipy import stats

    return float(stats.chi2.sf(float(statistic), int(dof)))


def sampling_fidelity(
    *,
    profile: Mapping[str, Any],
    sample: Mapping[str, Any],
    max_abs_z: float = SAMPLER_MAX_ABS_Z,
    min_chi_square_p: float = SAMPLER_MIN_CHI_SQUARE_P,
) -> dict[str, Any]:
    """Is what came out the distribution the weights describe?

    Three arms, all registered before the run:

    1. **The mean drawn weight.** Under weight-proportional sampling
       `E[w] = sum(w^2) / sum(w)`, which is a strictly larger number than the
       arithmetic mean whenever the weights are not constant — so this arm is
       the one that separates a weight-proportional sampler from a uniform one.
    2. **Weight-bin mass**, `bins` cells, chi-square against the mass share the
       full stream measured.
    3. **Block occupancy**, one cell per block, chi-square against the block's
       share of total weight. This is the arm that would catch a block CDF that
       is right in aggregate and wrong in placement.
    """
    total = float(profile["total_weight"])
    draws = int(sample["draws"])
    sampled = np.asarray(sample["sampled_weights"], dtype=np.float64)
    expected_mean = float(profile["sum_of_squares"]) / total
    observed_mean = float(sampled.mean())
    standard_error = float(sampled.std(ddof=1) / math.sqrt(draws))
    z = (
        math.inf if standard_error == 0.0
        else (observed_mean - expected_mean) / standard_error
    )

    bins = int(profile["bins"])
    expected_bin = np.asarray(profile["bin_mass"], dtype=np.float64) / total * draws
    slot = np.clip((sampled * bins).astype(np.int64), 0, bins - 1)
    observed_bin = np.bincount(slot, minlength=bins).astype(np.float64)
    live_bin = expected_bin > 0
    bin_chi = float(np.sum(
        np.square(observed_bin[live_bin] - expected_bin[live_bin])
        / expected_bin[live_bin]
    ))
    bin_dof = int(live_bin.sum()) - 1
    bin_p = _chi_square_p(bin_chi, max(bin_dof, 1))

    block_sums = np.asarray(profile["block_sums"], dtype=np.float64)
    expected_block = block_sums / total * draws
    observed_block = np.bincount(
        np.asarray(sample["chosen_block"], dtype=np.int64),
        minlength=block_sums.size,
    ).astype(np.float64)
    live_block = expected_block > 0
    block_chi = float(np.sum(
        np.square(observed_block[live_block] - expected_block[live_block])
        / expected_block[live_block]
    ))
    block_dof = int(live_block.sum()) - 1
    block_p = _chi_square_p(block_chi, max(block_dof, 1))

    arms = {
        "mean_weight_z_within_bound": bool(abs(z) <= float(max_abs_z)),
        "weight_bin_chi_square_p_above_floor": bool(
            bin_p >= float(min_chi_square_p)
        ),
        "block_occupancy_chi_square_p_above_floor": bool(
            block_p >= float(min_chi_square_p)
        ),
    }
    return {
        "expected_mean_weight": expected_mean,
        "observed_mean_weight": observed_mean,
        "arithmetic_mean_weight": float(profile["mean_weight"]),
        "mean_weight_standard_error": standard_error,
        "mean_weight_z": z,
        "max_abs_z": float(max_abs_z),
        "weight_bin_chi_square": bin_chi,
        "weight_bin_dof": bin_dof,
        "weight_bin_p": bin_p,
        "block_chi_square": block_chi,
        "block_dof": block_dof,
        "block_p": block_p,
        "min_chi_square_p": float(min_chi_square_p),
        "arms": arms,
        "holds": all(arms.values()),
    }


def uniform_sample_control(
    weights: np.ndarray,
    *,
    profile: Mapping[str, Any],
    draws: int = 2_000_000,
    seed: int = SAMPLER_SEED + 1,
) -> dict[str, Any]:
    """The mis-sampler the fidelity check must reject.

    A uniform draw over edge positions is the most plausible way to get this
    wrong — it loads, it runs, it produces edges, and it trains a different
    graph. `sampling_fidelity` must fail on it, or the check is decoration.
    """
    array = np.asarray(weights)
    rng = np.random.default_rng(int(seed))
    edge_index = rng.integers(0, array.size, size=int(draws), dtype=np.int64)
    block = int(profile["block"])
    sample = {
        "draws": int(draws),
        "seed": int(seed),
        "edge_index": edge_index,
        "sampled_weights": np.asarray(array[edge_index], dtype=np.float64),
        "chosen_block": (edge_index // block).astype(np.int64),
    }
    verdict = sampling_fidelity(profile=profile, sample=sample)
    return {
        "control": "round0244-uniform-mis-sampler-v1",
        "draws": int(draws),
        "fidelity": {
            key: value for key, value in verdict.items() if key != "arms"
        },
        "arms": verdict["arms"],
        "rejected": not verdict["holds"],
        "note": (
            "a uniform sampler over edge POSITIONS. If sampling_fidelity "
            "accepts this, it is not measuring what it claims to measure."
        ),
    }


# --------------------------------------------------------------------------- #
# B. R0228's displacement DiD — the inference rule, registered in advance
# --------------------------------------------------------------------------- #
#: Family-wise alpha over the two stratifications. Fixed here, at the release
#: commit, before any displacement number exists anywhere.
DID_ALPHA = 0.01
DID_TESTS_IN_FAMILY = 2
DID_MAPS_PER_ARM = 5
DID_SAMPLE_ROWS = 20_000
DID_DECILES = DENSITY_DECILES
DID_SEED = 244_228
DID_SD_EQUIVALENCE_BOUND = 1.0

#: The three row populations, defined on R0238's `500,000`-row uniform probe —
#: the only rows in this rung that have exact 15-NN truth, which R0228's
#: statistic requires.
DID_POPULATIONS = (
    (
        "genuine",
        "probe rows with tie-aware builder-missing edges > 0: loss the builder "
        "did NOT cover with a tie-equivalent substitute",
    ),
    (
        "tie_forgiven",
        "probe rows with strict builder-missing > 0 AND tie-aware "
        "builder-missing == 0: loss entirely covered by a substitute within "
        "TIE_TOLERANCE of the true k-th cosine",
    ),
    (
        "control",
        "probe rows with strict builder-missing == 0, density-matched to each "
        "treated population on deciles of the row's own true 15th-best cosine, "
        "and split in half so one half serves as the placebo arm",
    ),
)

DID_STATISTIC = (
    "within one map: s(r) = mean over the row's 15 EXACT high-D neighbours of "
    "||y_r - y_j||, divided by that map's own RMS radius about its centroid "
    "(R0228 `true_neighbour_scatter`, imported, never re-typed). Per map m, "
    "gap_P(m) = mean s over population P minus mean s over its density-matched "
    "control half C1. The difference in differences over an arm of n maps is "
    "DiD_P = mean_m gap_P(m) - mean_m gap_placebo(m), where the placebo gap is "
    "the second control half C2 scored against C1."
)

DID_NULL_ARM = (
    "R0228's null arm was eight maps trained on the EXACT k-NN graph. That arm "
    "is NOT CONSTRUCTIBLE at 100,000,000 rows: no exact 100M k15 graph exists, "
    "building one is the reason cluster-spill exists, and R0238's exact truth "
    "covers 500,000 probe rows only. The null is therefore re-specified as a "
    "PLACEBO SPLIT of the density-matched control inside the same maps, which "
    "estimates the gap statistic's own bias and dispersion under 'these two "
    "row sets differ in nothing'. This is a WEAKER null than R0228's in one "
    "specific way and the difference is registered here rather than discovered "
    "later: it controls for the statistic, not for 'a map trained on a lossy "
    "graph displaces everything'. No result computed under it may be reported "
    "as if the exact-graph arm had been run."
)

DID_PERMUTATION = (
    "one-sided exact permutation over every relabelling of the pooled per-map "
    "gaps (R0229 `exact_displacement_permutation`, imported). With n maps in "
    "the treated arm and n in the placebo arm the design enumerates "
    "C(2n, n) labellings and its smallest attainable p is 1/C(2n, n). The two "
    "stratifications form a family of 2 and are corrected by Holm-Bonferroni "
    "(R0229 `holm_bonferroni`, imported) at alpha = 0.01."
)

#: The decision map. Registered BEFORE the measurement exists. The third clause
#: is the one that matters: this program has shipped five checks that could not
#: fail, and an underpowered null read as "harmless" would be the sixth.
DID_DECISION_RULE = (
    "HARMFUL - the residual displaces rows, training on this graph stops: "
    "DiD_genuine > 0 with a Holm-adjusted one-sided p <= 0.01. "
    "| "
    "HARMLESS IN THE MEASURED SENSE: DiD_genuine has Holm-adjusted p > 0.01 "
    "AND |DiD_genuine| <= 1.0 sd of the placebo arm AND the instrument is "
    "shown on the SAME maps to have the power to detect a displacement of the "
    "size at issue - a planted-displacement positive control that the test "
    "does reject. "
    "| "
    "INDETERMINATE - anything else, including both arms null without the power "
    "demonstration, and including both arms significant. A null from a design "
    "whose power was not demonstrated is INDETERMINATE, never harmless."
)

DID_WHAT_IT_CANNOT_SETTLE = (
    "R0244 trains no map, so it computes no displacement. R0228's statistic "
    "consumes trained (n, 2) map coordinates - `map_scale` and "
    "`true_neighbour_scatter` both raise on anything else - and parametric "
    "UMAP is not identifiable across seeds, so the statistic must be computed "
    "WITHIN one map and therefore cannot be simulated from the graph. The "
    "populations, the density match, the rule and the resolution are fixed "
    "here; the measurement belongs to the first round that trains maps at this "
    "rung. No proxy is substituted."
)


def permutation_resolution(
    *, treated_maps: int, null_maps: int, alpha: float = DID_ALPHA,
    tests_in_family: int = DID_TESTS_IN_FAMILY,
) -> dict[str, Any]:
    """Can a design of this size clear its own correction? Computed, not claimed.

    review-0228-01 found the program's per-configuration permutation family had
    a smallest attainable `p` of `1/165` against a `0.05/12` threshold, so no
    outcome could ever have cleared it. The same arithmetic is done here for
    each candidate arm size BEFORE the arm size is chosen.
    """
    total = math.comb(int(treated_maps) + int(null_maps), int(treated_maps))
    floor = 1.0 / float(total)
    strictest = float(alpha) / float(tests_in_family)
    return {
        "treated_maps": int(treated_maps),
        "null_maps": int(null_maps),
        "labellings": int(total),
        "smallest_attainable_p": floor,
        "strictest_holm_threshold": strictest,
        "can_reject_under_the_family_correction": bool(floor <= strictest),
    }


def did_registration(
    *,
    arm_sizes: Sequence[int] = (3, 4, 5, 6, 8),
    maps_per_arm: int = DID_MAPS_PER_ARM,
) -> dict[str, Any]:
    """The full pre-registered rule, plus the resolution table behind it."""
    table = [
        permutation_resolution(treated_maps=size, null_maps=size)
        for size in arm_sizes
    ]
    smallest_workable = next(
        (row["treated_maps"] for row in table
         if row["can_reject_under_the_family_correction"]),
        None,
    )
    selected = permutation_resolution(
        treated_maps=int(maps_per_arm), null_maps=int(maps_per_arm)
    )
    if not selected["can_reject_under_the_family_correction"]:
        raise Round0244Error(
            "R0244 refuses to register a DiD design whose smallest attainable "
            f"p ({selected['smallest_attainable_p']}) cannot clear its own "
            f"family correction ({selected['strictest_holm_threshold']})"
        )
    return {
        "instrument": "round0244-displacement-did-registration-v1",
        "round_id": ROUND_ID,
        "rows": ROWS,
        "k": GRAPH_K,
        "statistic": DID_STATISTIC,
        "scatter_definition": SCATTER_DEFINITION,
        "populations": [
            {"name": name, "definition": text} for name, text in DID_POPULATIONS
        ],
        "density_match": (
            "R0228 `density_matched_control`, imported: deciles of the row's "
            "own true 15th-best cosine, exact per-decile counts, seed "
            f"{DID_SEED}"
        ),
        "sample_rows_per_population": int(DID_SAMPLE_ROWS),
        "deciles": int(DID_DECILES),
        "seed": int(DID_SEED),
        "null_arm": DID_NULL_ARM,
        "permutation_design": DID_PERMUTATION,
        "alpha_family_wise": float(DID_ALPHA),
        "tests_in_family": int(DID_TESTS_IN_FAMILY),
        "equivalence_bound_in_placebo_sd": float(DID_SD_EQUIVALENCE_BOUND),
        "decision_rule": DID_DECISION_RULE,
        "resolution_table": table,
        "smallest_arm_that_can_reject": smallest_workable,
        "selected_design": selected,
        "what_this_round_cannot_settle": DID_WHAT_IT_CANNOT_SETTLE,
    }


def did_requirement(
    *,
    maps_per_arm: int = DID_MAPS_PER_ARM,
    hours_per_map: float = 10.0,
    rung_gpu_hours_cap: float = 12.0,
) -> dict[str, Any]:
    """What the first training round must produce for the DiD to be computable.

    The placebo null lives inside the same maps, so the arm count is `n`, not
    `2n`. The statistic reads 2-D coordinates and R0238's sealed exact truth,
    so it performs NO substrate gather at all — R0243's sorted-gather price is
    the right instrument for a different question and does not apply here.
    """
    maps = int(maps_per_arm)
    gpu_hours = float(maps) * float(hours_per_map)
    return {
        "maps_required": maps,
        "why_not_two_arms": (
            "the placebo null is a second control half inside the SAME maps, "
            "so no separate null-arm training is needed. R0228 needed 8 extra "
            "maps because its null was a different graph; that graph does not "
            "exist at this rung."
        ),
        "seeds": [f"seed{index}" for index in range(42, 42 + maps)],
        "rows_scored": "the 500,000 R0238 uniform probe rows",
        "why_only_the_probe": (
            "R0228's s(r) averages the distance to the row's EXACT high-D "
            "neighbours. Exact 15-NN exists at this rung for the 500,000 probe "
            "rows and nowhere else."
        ),
        "gather_bytes": 0,
        "gather_note": (
            "no substrate gather: the statistic reads (n, 2) coordinates and "
            "R0238's sealed 30,000,128-byte truth ids. R0243's 0.5918x sorted "
            "gather prices a 384-d substrate read, which this instrument does "
            "not perform."
        ),
        "hours_per_map_assumed": float(hours_per_map),
        "estimated_gpu_hours": gpu_hours,
        "rung_gpu_hours_cap": float(rung_gpu_hours_cap),
        "fits_in_the_rung_cap": bool(gpu_hours <= float(rung_gpu_hours_cap)),
        "multiple_of_the_rung_cap": gpu_hours / float(rung_gpu_hours_cap),
        "consequence": (
            "at 100,000,000 rows the registered design costs about "
            f"{gpu_hours:.1f} GPU-h against a {rung_gpu_hours_cap:.1f} h rung "
            "cap. It therefore CANNOT be run at this rung under the current "
            "cap. Either the cap moves, or the DiD runs first at a rung where "
            "a map is affordable and the 100M rung inherits the finding with "
            "its N-extrapolation stated. This is a cost fact, not a reason to "
            "substitute a cheaper instrument."
        ),
        "label": "prediction",
    }


def did_populations(
    *,
    strict_builder_missing: np.ndarray,
    tie_aware_builder_missing: np.ndarray,
    kth_cosine: np.ndarray,
    sample_rows: int = DID_SAMPLE_ROWS,
    deciles: int = DID_DECILES,
    seed: int = DID_SEED,
) -> dict[str, Any]:
    """Build and seal the DiD's row sets. This computes NO displacement.

    Everything here is substrate-side: `density_matched_control` needs only a
    per-row loss mask and the row's own true `k`-th cosine, both of which R0242
    and R0243 sealed. Sealing them now makes the training round a join rather
    than a re-measurement, and fixes the row sets before any map exists.
    """
    strict = np.asarray(strict_builder_missing, dtype=np.int64)
    tie = np.asarray(tie_aware_builder_missing, dtype=np.int64)
    cosine = np.asarray(kth_cosine, dtype=np.float64)
    if not (strict.shape == tie.shape == cosine.shape) or strict.ndim != 1:
        raise Round0244Error(
            "R0244 DiD populations need three matched 1-D probe vectors"
        )
    genuine_mask = tie > 0
    forgiven_mask = (strict > 0) & (tie == 0)
    intact_mask = strict == 0
    if not genuine_mask.any() or not forgiven_mask.any():
        raise Round0244Error(
            "R0244 DiD populations are empty; the probe carries no genuine or "
            "no tie-forgiven loss"
        )
    genuine = density_matched_control(
        lost_mask=genuine_mask, kth_cosine=cosine,
        sample_rows=int(sample_rows), deciles=int(deciles), seed=int(seed),
    )
    forgiven = density_matched_control(
        lost_mask=forgiven_mask, kth_cosine=cosine,
        sample_rows=int(sample_rows), deciles=int(deciles), seed=int(seed) + 1,
    )
    #: The placebo split. The control pool is everything with zero strict
    #: builder loss; halving it at a fixed seed gives the arm that estimates
    #: the gap statistic's own bias.
    rng = np.random.default_rng(int(seed) + 2)
    intact_rows = np.flatnonzero(intact_mask)
    shuffled = rng.permutation(intact_rows)
    half = shuffled.size // 2
    placebo_a = np.sort(shuffled[:half])
    placebo_b = np.sort(shuffled[half:2 * half])
    overlap = np.intersect1d(placebo_a, placebo_b).size
    return {
        "probe_rows": int(strict.size),
        "genuine_rows_total": int(genuine_mask.sum()),
        "tie_forgiven_rows_total": int(forgiven_mask.sum()),
        "intact_rows_total": int(intact_mask.sum()),
        "genuine": genuine,
        "tie_forgiven": forgiven,
        "placebo_a": placebo_a,
        "placebo_b": placebo_b,
        "placebo_halves_disjoint": bool(overlap == 0),
        "matched_exactly": bool(
            genuine.get("matched_exactly", False)
            and forgiven.get("matched_exactly", False)
        ),
        "displacement_computed": False,
        "note": (
            "row sets only. No map exists, so no s(r), no gap and no DiD is "
            "computed here, and none is approximated by anything else."
        ),
    }


# --------------------------------------------------------------------------- #
# C. cluster 168's text — what a "near-duplicate" actually is here
# --------------------------------------------------------------------------- #
#: Registered before any text is read. A pair falls in exactly one class.
NEAR_DUPLICATE_CATEGORIES = (
    ("identical", "the two chunk texts are byte-identical"),
    ("near_identical", "character 5-gram Jaccard >= 0.80"),
    ("substantial_overlap", "0.40 <= Jaccard < 0.80"),
    ("some_overlap", "0.10 <= Jaccard < 0.40"),
    ("distinct_text", "Jaccard < 0.10"),
)
NEAR_IDENTICAL_JACCARD = 0.80
SUBSTANTIAL_JACCARD = 0.40
SOME_OVERLAP_JACCARD = 0.10
SHINGLE_N = 5

#: The text a row claims to be must be VERIFIED, not assumed: the chunk is
#: re-embedded with the same `all-MiniLM-L6-v2` that produced the substrate and
#: must reproduce the substrate row.
TEXT_BINDING_COSINE_FLOOR = 0.999
TEXT_EXCERPT_CHARS = 320
TEXT_SAMPLE_PAIRS = 12
TEXT_SAMPLE_SEED = 244_168

TEXT_NOTE = (
    "tie tolerance certifies equal COSINE TO THE QUERY, not equal LOCATION: "
    "two rows at identical cosine to a query lie on a cone, not on each other. "
    "So the decisive number in this section is cos(missed truth neighbour, "
    "tie-forgiven substitute) - not cos(query, either of them), which tie "
    "tolerance makes equal by construction."
)


def char_shingles(text: str, *, n: int = SHINGLE_N) -> set[str]:
    body = " ".join(str(text).split())
    if len(body) < n:
        return {body} if body else set()
    return {body[index:index + n] for index in range(len(body) - n + 1)}


def jaccard(left: str, right: str, *, n: int = SHINGLE_N) -> float:
    a = char_shingles(left, n=n)
    b = char_shingles(right, n=n)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def classify_text_pair(left: str, right: str) -> dict[str, Any]:
    """One registered category per pair. Descriptive, not a gate."""
    if str(left) == str(right):
        score = 1.0
        category = "identical"
    else:
        score = jaccard(left, right)
        if score >= NEAR_IDENTICAL_JACCARD:
            category = "near_identical"
        elif score >= SUBSTANTIAL_JACCARD:
            category = "substantial_overlap"
        elif score >= SOME_OVERLAP_JACCARD:
            category = "some_overlap"
        else:
            category = "distinct_text"
    return {
        "jaccard_char_5gram": float(score),
        "category": category,
        "left_chars": len(str(left)),
        "right_chars": len(str(right)),
    }


def excerpt(text: str, *, chars: int = TEXT_EXCERPT_CHARS) -> str:
    body = " ".join(str(text).split())
    return body if len(body) <= chars else body[:chars] + " ..."


__all__ = [
    "DID_ALPHA",
    "DID_DECISION_RULE",
    "DID_MAPS_PER_ARM",
    "DID_NULL_ARM",
    "DID_PERMUTATION",
    "DID_POPULATIONS",
    "DID_SAMPLE_ROWS",
    "DID_SEED",
    "DID_STATISTIC",
    "DID_WHAT_IT_CANNOT_SETTLE",
    "NEAR_DUPLICATE_CATEGORIES",
    "R0243_DIRECTED_EDGES",
    "R0243_TOTAL_WEIGHT",
    "R0243_WEIGHT_MAX",
    "SAMPLER_BLOCK_EDGES",
    "SAMPLER_DRAWS",
    "SAMPLER_EPOCHS",
    "SAMPLER_MAX_ABS_Z",
    "SAMPLER_MAX_ANONYMOUS_BYTES",
    "SAMPLER_MIN_CHI_SQUARE_P",
    "SAMPLER_MIN_DRAWS_PER_S",
    "SAMPLER_NOTE",
    "SAMPLER_POLL_CHUNK_DRAWS",
    "SAMPLER_SEED",
    "SAMPLER_WEIGHT_BINS",
    "TEXT_BINDING_COSINE_FLOOR",
    "TEXT_EXCERPT_CHARS",
    "TEXT_NOTE",
    "TEXT_SAMPLE_PAIRS",
    "TEXT_SAMPLE_SEED",
    "char_shingles",
    "classify_text_pair",
    "did_populations",
    "did_registration",
    "did_requirement",
    "excerpt",
    "jaccard",
    "permutation_resolution",
    "sampling_fidelity",
    "two_level_weight_sample",
    "uniform_sample_control",
    "weight_block_profile",
]
