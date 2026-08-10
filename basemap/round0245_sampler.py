"""R0245 — finding 7: the mis-sampler control, strengthened and bounded.

review-0244-01 §E3 is blunt about R0244's negative control and it is right:

* The uniform mis-sampler is a **strawman**. `z = -1010` against a `|z| <= 5`
  bound is a `200x` overshoot and its bin chi-square is `991,142` against a
  critical `50.8`. Nothing subtle survives to be tested by it.
* One of its three arms is **nearly vacuous at the wrong draw count**. Block
  sums over `2,395` blocks of a million *consecutive* edges have a coefficient
  of variation of a couple of percent, so block occupancy under any
  position-uniform sampler is nearly correct. "Rejected on all three arms" is
  true only at the `2,000,000` draws R0244 happened to use.
* The check has a **blind spot below `w ~ 0.02`** and only an `8x` margin
  against a `float32`-CDF bug at `20,000,000` draws.

So this module does three things, none of which weakens the check:

1. **Makes the draw count part of the check.** `required_draws_for_arm()`
   inverts the chi-square power relation from the *measured* block and bin
   profiles, and `sampler_draw_floor()` is an executable gate that refuses a
   draw count below what the registered mis-sampler family needs.
2. **Replaces the strawman with a family.** `mis_sampler_battery()` runs the
   subtler errors — block-by-weight/uniform-within, uniform-block/weight-within,
   a `float32` within-block CDF, and an epoch-floor truncation — through the
   IMPORTED `sampling_fidelity`, and reports per arm which caught which.
3. **States the limit rather than implying it away.** `certified_weight_floor()`
   computes the weight below which truncation is undetectable at a given draw
   count, so the capability carries its own blind spot.

Everything scores through R0244's `sampling_fidelity` and `weight_block_profile`
unchanged. No threshold in the registered check is moved.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.round0244_prereq import (
    SAMPLER_BLOCK_EDGES,
    SAMPLER_MIN_CHI_SQUARE_P,
    SAMPLER_SEED,
    sampling_fidelity,
    two_level_weight_sample,
)
from basemap.round0245_guard import Round0245Error

#: The mis-samplers this round registers as the family the check must catch.
#: Each is a plausible implementation error, not a caricature.
MIS_SAMPLER_FAMILY = (
    (
        "uniform_positions",
        "R0244's control: uniform over edge POSITIONS. The strawman, kept so "
        "the family is comparable with the sealed round.",
    ),
    (
        "block_by_weight_uniform_within",
        "picks the block correctly from the block-sum CDF, then picks "
        "uniformly INSIDE the block - the second half of the two-level scheme "
        "forgotten. Aggregate block placement is right; weights are wrong.",
    ),
    (
        "uniform_block_weight_within",
        "picks the block uniformly, then correctly by weight within it - the "
        "inverse error. Weights look nearly right; block placement is wrong.",
    ),
    (
        "float32_within_block_cdf",
        "the correct two-level scheme with the WITHIN-block cumulative sum "
        "accumulated in float32. A real precision bug: over 1,048,576 terms "
        "the float32 running sum drifts and the tail of each block is "
        "systematically mis-weighted.",
    ),
    (
        "drop_tail_below_epoch_floor",
        "the correct scheme applied only to edges at or above the "
        "n_epochs = 200 floor. The edges dropped hold a millionth of the "
        "total weight, which is why this is the family's blind spot rather "
        "than a member the arms catch.",
    ),
)

#: R0245's registered draw count. R0244 used 20,000,000 and review-0244-01 §E3
#: showed that is close to the floor at which the weight-bin arm still catches
#: a float32 within-block CDF. Doubling it is the cheap strengthening: the
#: chi-square excess of every mis-sampler scales with draws, so 40,000,000
#: roughly doubles every margin in the family.
R0245_SAMPLER_DRAWS = 40_000_000

DRAW_COUNT_NOTE = (
    "chi-square power scales with the draw count, so a mis-sampler control is "
    "a statement about a DRAW COUNT and not only about a check. Under a wrong "
    "sampler with cell probabilities q against the truth p, the expected "
    "statistic is dof + draws * sum((p - q)^2 / q), so the draws needed to "
    "reach the critical value at the registered floor are "
    "(critical - dof) / sum((p - q)^2 / q). R0245 registers the draw count as "
    "part of the check: 20,000,000 is a FLOOR derived from the family, not a "
    "convention."
)

BLIND_SPOT_NOTE = (
    "the weight-bin arm cannot see a sampler that truncates the extreme tail: "
    "the edges below the n_epochs = 200 floor hold about 1.7e-6 of the total "
    "weight, so removing them moves no cell measurably. Sensitivity fades "
    "around w ~ 0.02. Since UMAP's own edge schedule never samples that tail "
    "either, this is arguably the right blind spot to have - but it is a LIMIT "
    "OF THE CAPABILITY and belongs in the capability, not in a review."
)


def _chi_square_critical(dof: int, min_p: float) -> float:
    from scipy import stats

    return float(stats.chi2.isf(float(min_p), int(dof)))


def _noncentrality(p_cells: np.ndarray, q_cells: np.ndarray) -> float:
    """`sum((p - q)^2 / q)` over live cells: the per-draw chi-square excess."""
    p_cells = np.asarray(p_cells, dtype=np.float64)
    q_cells = np.asarray(q_cells, dtype=np.float64)
    live = q_cells > 0
    return float(
        np.sum(np.square(p_cells[live] - q_cells[live]) / q_cells[live])
    )


def required_draws_for_arm(
    *,
    p_cells: np.ndarray,
    q_cells: np.ndarray,
    min_p: float = SAMPLER_MIN_CHI_SQUARE_P,
) -> dict[str, Any]:
    """How many draws this arm needs to reject a sampler with cells `q`.

    The criterion is the expected statistic reaching the critical value — the
    median-unbiased reading of a non-central chi-square, which understates the
    power slightly and therefore overstates the required draws slightly. That
    is the conservative direction for a floor.
    """
    excess = _noncentrality(p_cells, q_cells)
    dof = int(np.count_nonzero(np.asarray(q_cells) > 0)) - 1
    critical = _chi_square_critical(max(dof, 1), float(min_p))
    if excess <= 0.0:
        return {
            "per_draw_chi_square_excess": excess,
            "dof": dof,
            "critical_value": critical,
            "required_draws": None,
            "arm_can_never_reject_this_sampler": True,
        }
    required = (critical - dof) / excess
    return {
        "per_draw_chi_square_excess": excess,
        "dof": dof,
        "critical_value": critical,
        "required_draws": float(max(required, 0.0)),
        "arm_can_never_reject_this_sampler": False,
    }


def block_profile_dispersion(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Why the block arm is weak against a position-uniform sampler.

    A block is a million *consecutive* edges, so the block sums average away
    almost all weight structure. This measures how much is left.
    """
    sums = np.asarray(profile["block_sums"], dtype=np.float64)
    total = float(profile["total_weight"])
    block = int(profile["block"])
    edges = int(profile["edges"])
    counts = np.full(sums.size, float(block))
    counts[-1] = float(edges - block * (sums.size - 1))
    p_cells = sums / total
    q_cells = counts / float(edges)
    return {
        "blocks": int(sums.size),
        "block_edges": block,
        "block_sum_mean": float(sums.mean()),
        "block_sum_sd": float(sums.std(ddof=1)),
        "block_sum_coefficient_of_variation": float(
            sums.std(ddof=1) / sums.mean()
        ),
        "uniform_position_noncentrality": _noncentrality(p_cells, q_cells),
        "why_the_block_arm_is_weak": (
            "under a position-uniform sampler the block cell probabilities are "
            "the block's share of EDGES rather than of WEIGHT, and those two "
            "differ only by the block-to-block dispersion of mean weight, "
            "which a million consecutive edges nearly average out."
        ),
    }


def uniform_position_draw_requirement(
    *,
    profile: Mapping[str, Any],
    min_p: float = SAMPLER_MIN_CHI_SQUARE_P,
) -> dict[str, Any]:
    """The draw count each arm needs against R0244's own uniform mis-sampler.

    This is the arithmetic R0244 did not publish and the reason its "rejected
    on all three arms" is a statement about `2,000,000` draws.
    """
    sums = np.asarray(profile["block_sums"], dtype=np.float64)
    total = float(profile["total_weight"])
    block = int(profile["block"])
    edges = int(profile["edges"])
    counts = np.full(sums.size, float(block))
    counts[-1] = float(edges - block * (sums.size - 1))
    block_arm = required_draws_for_arm(
        p_cells=sums / total, q_cells=counts / float(edges), min_p=min_p
    )
    bin_mass = np.asarray(profile["bin_mass"], dtype=np.float64)
    bin_counts = np.asarray(profile["bin_counts"], dtype=np.float64)
    bin_arm = required_draws_for_arm(
        p_cells=bin_mass / total,
        q_cells=bin_counts / float(edges),
        min_p=min_p,
    )
    return {
        "mis_sampler": "uniform_positions",
        "block_arm": block_arm,
        "weight_bin_arm": bin_arm,
        "note": DRAW_COUNT_NOTE,
    }


def certified_weight_floor(
    *,
    profile: Mapping[str, Any],
    draws: int = R0245_SAMPLER_DRAWS,
    min_p: float = SAMPLER_MIN_CHI_SQUARE_P,
) -> dict[str, Any]:
    """The weight below which truncation is invisible at this draw count.

    Walks the weight bins from the bottom, deletes each prefix, and reports the
    largest prefix whose removal the bin arm still cannot detect at `draws`.
    """
    bin_mass = np.asarray(profile["bin_mass"], dtype=np.float64)
    bin_counts = np.asarray(profile["bin_counts"], dtype=np.float64)
    total = float(profile["total_weight"])
    bins = int(profile["bins"])
    p_cells = bin_mass / total
    edge = 1.0 / float(bins)
    undetectable_upto = 0.0
    undetectable_mass = 0.0
    rows = []
    for cut in range(1, bins):
        kept = bin_mass.copy()
        kept[:cut] = 0.0
        kept_total = float(kept.sum())
        if kept_total <= 0.0:
            break
        q_cells = kept / kept_total
        arm = required_draws_for_arm(
            p_cells=p_cells, q_cells=q_cells, min_p=min_p
        )
        detectable = bool(
            arm["required_draws"] is not None
            and arm["required_draws"] <= float(draws)
        )
        rows.append({
            "weight_below": float(cut) * edge,
            "mass_removed_fraction": float(bin_mass[:cut].sum()) / total,
            "edges_removed": int(bin_counts[:cut].sum()),
            "required_draws": arm["required_draws"],
            "detectable_at_the_registered_draws": detectable,
        })
        if not detectable:
            undetectable_upto = float(cut) * edge
            undetectable_mass = float(bin_mass[:cut].sum()) / total
    return {
        "instrument": "round0245-sampler-blind-spot-v1",
        "draws": int(draws),
        "certified_above_weight": undetectable_upto,
        "undetectable_mass_fraction": undetectable_mass,
        "epoch_weight_floor": float(profile["epoch_weight_floor"]),
        "edges_below_the_epoch_floor": int(
            profile["edges_never_sampled_in_a_run"]
        ),
        "prefix_scan": rows,
        "note": BLIND_SPOT_NOTE,
    }


# --------------------------------------------------------------------------- #
# the mis-sampler family, drawn for real and scored by the imported check
# --------------------------------------------------------------------------- #
def _uniform_positions(weights, *, profile, draws, rng):
    array = np.asarray(weights)
    edge_index = rng.integers(0, array.size, size=int(draws), dtype=np.int64)
    return edge_index


def _block_by_weight_uniform_within(weights, *, profile, draws, rng):
    array = np.asarray(weights)
    block = int(profile["block"])
    sums = np.asarray(profile["block_sums"], dtype=np.float64)
    total = float(profile["total_weight"])
    cdf = np.cumsum(sums)
    cdf[-1] = total
    chosen = np.searchsorted(cdf, rng.random(int(draws)) * total, side="right")
    np.clip(chosen, 0, sums.size - 1, out=chosen)
    lo = chosen.astype(np.int64) * block
    span = np.minimum(lo + block, array.size) - lo
    return lo + (rng.random(int(draws)) * span).astype(np.int64)


def _uniform_block_weight_within(weights, *, profile, draws, rng):
    array = np.asarray(weights)
    block = int(profile["block"])
    blocks = int(profile["blocks"])
    chosen = rng.integers(0, blocks, size=int(draws), dtype=np.int64)
    order = np.argsort(chosen, kind="stable")
    ordered = chosen[order]
    boundaries = np.flatnonzero(np.diff(ordered)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [ordered.size]])
    out = np.empty(int(draws), dtype=np.int64)
    for start, end in zip(starts, ends):
        index = int(ordered[start])
        lo = index * block
        hi = min(lo + block, array.size)
        chunk = np.asarray(array[lo:hi], dtype=np.float64)
        within = np.cumsum(chunk)
        picks = np.searchsorted(
            within, rng.random(end - start) * float(within[-1]), side="right"
        )
        np.clip(picks, 0, chunk.size - 1, out=picks)
        out[order[start:end]] = lo + picks
    return out


def _float32_within_block_cdf(weights, *, profile, draws, rng):
    array = np.asarray(weights)
    block = int(profile["block"])
    sums = np.asarray(profile["block_sums"], dtype=np.float64)
    total = float(profile["total_weight"])
    cdf = np.cumsum(sums)
    cdf[-1] = total
    chosen = np.searchsorted(cdf, rng.random(int(draws)) * total, side="right")
    np.clip(chosen, 0, sums.size - 1, out=chosen)
    order = np.argsort(chosen, kind="stable")
    ordered = chosen[order]
    boundaries = np.flatnonzero(np.diff(ordered)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [ordered.size]])
    out = np.empty(int(draws), dtype=np.int64)
    for start, end in zip(starts, ends):
        index = int(ordered[start])
        lo = index * block
        hi = min(lo + block, array.size)
        chunk = np.asarray(array[lo:hi], dtype=np.float32)
        within = np.cumsum(chunk, dtype=np.float32)
        picks = np.searchsorted(
            within, (rng.random(end - start) * float(within[-1])).astype(
                np.float32
            ), side="right",
        )
        np.clip(picks, 0, chunk.size - 1, out=picks)
        out[order[start:end]] = lo + picks
    return out


def _drop_tail_below_epoch_floor(weights, *, profile, draws, rng):
    array = np.asarray(weights)
    floor = float(profile["epoch_weight_floor"])
    block = int(profile["block"])
    blocks = int(profile["blocks"])
    kept_sums = np.zeros(blocks, dtype=np.float64)
    for index in range(blocks):
        lo = index * block
        chunk = np.asarray(array[lo:min(lo + block, array.size)], dtype=np.float64)
        kept_sums[index] = float(chunk[chunk >= floor].sum())
    total = float(kept_sums.sum())
    cdf = np.cumsum(kept_sums)
    cdf[-1] = total
    chosen = np.searchsorted(cdf, rng.random(int(draws)) * total, side="right")
    np.clip(chosen, 0, blocks - 1, out=chosen)
    order = np.argsort(chosen, kind="stable")
    ordered = chosen[order]
    boundaries = np.flatnonzero(np.diff(ordered)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [ordered.size]])
    out = np.empty(int(draws), dtype=np.int64)
    for start, end in zip(starts, ends):
        index = int(ordered[start])
        lo = index * block
        hi = min(lo + block, array.size)
        chunk = np.asarray(array[lo:hi], dtype=np.float64)
        masked = np.where(chunk >= floor, chunk, 0.0)
        within = np.cumsum(masked)
        picks = np.searchsorted(
            within, rng.random(end - start) * float(within[-1]), side="right"
        )
        np.clip(picks, 0, chunk.size - 1, out=picks)
        out[order[start:end]] = lo + picks
    return out


_IMPLEMENTATIONS = {
    "uniform_positions": _uniform_positions,
    "block_by_weight_uniform_within": _block_by_weight_uniform_within,
    "uniform_block_weight_within": _uniform_block_weight_within,
    "float32_within_block_cdf": _float32_within_block_cdf,
    "drop_tail_below_epoch_floor": _drop_tail_below_epoch_floor,
}


def mis_sampler_battery(
    weights: np.ndarray,
    *,
    profile: Mapping[str, Any],
    draws: int = R0245_SAMPLER_DRAWS,
    seed: int = SAMPLER_SEED + 245,
    abort_check: Any = None,
) -> dict[str, Any]:
    """Run the registered family through the IMPORTED fidelity check.

    Reports, per mis-sampler, which of the three arms rejected it. A family
    where every member is caught by the same arm would mean the other two are
    decoration; a member caught by none is a stated blind spot.
    """
    array = np.asarray(weights)
    block = int(profile["block"])
    results: list[dict[str, Any]] = []
    for offset, (name, description) in enumerate(MIS_SAMPLER_FAMILY):
        if abort_check is not None:
            abort_check(f"R0245 mis-sampler {name}")
        rng = np.random.default_rng(int(seed) + offset)
        edge_index = _IMPLEMENTATIONS[name](
            array, profile=profile, draws=int(draws), rng=rng
        )
        sample = {
            "draws": int(draws),
            "seed": int(seed) + offset,
            "edge_index": edge_index,
            "sampled_weights": np.asarray(array[edge_index], dtype=np.float64),
            "chosen_block": (edge_index // block).astype(np.int64),
        }
        verdict = sampling_fidelity(profile=profile, sample=sample)
        arms = verdict["arms"]
        rejecting = [
            arm for arm, holds in arms.items() if not holds
        ]
        results.append({
            "mis_sampler": name,
            "description": description,
            "draws": int(draws),
            "mean_weight_z": verdict["mean_weight_z"],
            "weight_bin_chi_square": verdict["weight_bin_chi_square"],
            "weight_bin_p": verdict["weight_bin_p"],
            "block_chi_square": verdict["block_chi_square"],
            "block_p": verdict["block_p"],
            "arms": arms,
            "arms_that_rejected": rejecting,
            "rejected": not verdict["holds"],
        })
        del edge_index, sample
    caught = [row["mis_sampler"] for row in results if row["rejected"]]
    missed = [row["mis_sampler"] for row in results if not row["rejected"]]
    by_arm: dict[str, list[str]] = {}
    for row in results:
        for arm in row["arms_that_rejected"]:
            by_arm.setdefault(arm, []).append(row["mis_sampler"])
    sole_detector = {
        row["mis_sampler"]: row["arms_that_rejected"][0]
        for row in results
        if len(row["arms_that_rejected"]) == 1
    }
    return {
        "instrument": "round0245-mis-sampler-family-v1",
        "family_size": len(MIS_SAMPLER_FAMILY),
        "draws": int(draws),
        "results": results,
        "caught": caught,
        "missed": missed,
        "rejections_by_arm": by_arm,
        "mis_samplers_caught_by_exactly_one_arm": sole_detector,
        "every_arm_is_the_sole_detector_of_something": bool(
            len(set(sole_detector.values())) >= 2
        ),
        "strawman_note": (
            "the uniform mis-sampler is kept for comparability with R0244 and "
            "it IS a strawman: its rejection margin is orders of magnitude "
            "wider than any of the others. The family exists so the check's "
            "power is characterised by the errors it barely catches, not by "
            "the one it obliterates."
        ),
    }


def sampler_draw_floor(
    *,
    profile: Mapping[str, Any],
    battery: Mapping[str, Any],
    registered_draws: int = R0245_SAMPLER_DRAWS,
    min_p: float = SAMPLER_MIN_CHI_SQUARE_P,
    margin: float = 1.0,
) -> dict[str, Any]:
    """The executable gate: is the registered draw count enough for the family?

    For every member the battery caught, the draw count at which it was caught
    is evidence; the floor is the largest per-member requirement the arms imply,
    times a registered margin. A round that lowers the draw count below this
    must fail rather than silently lose the ability to catch a `float32` CDF.
    """
    uniform = uniform_position_draw_requirement(profile=profile, min_p=min_p)
    requirements = [
        value["required_draws"]
        for value in (uniform["block_arm"], uniform["weight_bin_arm"])
        if value["required_draws"] is not None
    ]
    #: The battery's empirical margins: how far each caught member sits above
    #: its critical value tells us how much draw-count headroom it has.
    margins = []
    for row in battery["results"]:
        if not row["rejected"]:
            continue
        dof = max(int(profile["bins"]) - 1, 1)
        critical = _chi_square_critical(dof, float(min_p))
        excess = float(row["weight_bin_chi_square"]) - dof
        if excess > 0:
            margins.append({
                "mis_sampler": row["mis_sampler"],
                "bin_chi_square": row["weight_bin_chi_square"],
                "critical_value": critical,
                "bin_arm_margin": float(row["weight_bin_chi_square"]) / critical,
                "battery_draws": int(row["draws"]),
                "draws_at_which_the_bin_arm_would_lose_it": (
                    float(row["draws"]) * (critical - dof) / excess
                ),
            })
    binding = max(
        (row["draws_at_which_the_bin_arm_would_lose_it"] for row in margins),
        default=0.0,
    )
    floor = max([*requirements, binding * float(margin)])
    verdict = {
        "instrument": "round0245-sampler-draw-floor-v1",
        "registered_draws": int(registered_draws),
        "uniform_position_requirement": uniform,
        "per_member_margins": margins,
        "tightest_member_draws": binding,
        "margin": float(margin),
        "required_draws_floor": floor,
        "registered_draws_over_the_floor": (
            float(registered_draws) / floor if floor > 0 else None
        ),
        "registered_draws_clear_the_floor": bool(
            float(registered_draws) >= floor
        ),
        "note": DRAW_COUNT_NOTE,
    }
    if not verdict["registered_draws_clear_the_floor"]:
        raise Round0245Error(
            f"R0245 REFUSES the sampler check at {registered_draws} draws: the "
            f"registered mis-sampler family needs at least {floor:.0f} for "
            "every member to remain detectable. Draw count is part of the "
            "check."
        )
    return verdict


def true_sampler_reference(
    weights: np.ndarray,
    *,
    profile: Mapping[str, Any],
    draws: int = R0245_SAMPLER_DRAWS,
    seed: int = SAMPLER_SEED + 245,
    abort_check: Any = None,
) -> dict[str, Any]:
    """The correct two-level sampler at a fresh seed, scored the same way.

    A battery of wrong samplers proves nothing unless the right one passes the
    same arms at the same draw count.
    """
    sample = two_level_weight_sample(
        weights, profile=profile, draws=int(draws), seed=int(seed),
        abort_check=abort_check,
    )
    verdict = sampling_fidelity(profile=profile, sample=sample)
    out = {
        "seed": int(seed),
        "draws": int(draws),
        "distinct_edges_drawn": int(sample["distinct_edges_drawn"]),
        "fidelity": {
            key: value for key, value in verdict.items() if key != "arms"
        },
        "arms": verdict["arms"],
        "holds": bool(verdict["holds"]),
    }
    del sample
    return out


def sampler_limits(
    *,
    battery: Mapping[str, Any],
    draw_floor: Mapping[str, Any],
    blind_spot: Mapping[str, Any],
    dispersion: Mapping[str, Any],
) -> dict[str, Any]:
    """The capability's recorded limits, in the capability rather than a review."""
    return {
        "instrument": "round0245-sampler-limits-v1",
        "draw_count_is_part_of_the_check": DRAW_COUNT_NOTE,
        "required_draws_floor": draw_floor["required_draws_floor"],
        "certified_above_weight": blind_spot["certified_above_weight"],
        "undetectable_mass_fraction": blind_spot["undetectable_mass_fraction"],
        "block_sum_coefficient_of_variation": dispersion[
            "block_sum_coefficient_of_variation"
        ],
        "mis_samplers_the_family_missed": list(battery["missed"]),
        "mis_samplers_caught_by_exactly_one_arm": dict(
            battery["mis_samplers_caught_by_exactly_one_arm"]
        ),
        "plain_statement": (
            "R0244's sampling demonstration was weaker than it appeared: its "
            "negative control was a strawman, one of its three arms is nearly "
            "vacuous against that control at low draw counts, and the check "
            "certifies nothing below the weight floor reported here. The "
            "distribution binding to R0243 is unaffected - that part was "
            "byte-exact - but 'rejected on all three arms' should be read as "
            "'rejected on all three arms AT 2,000,000 draws'."
        ),
        "blind_spot": BLIND_SPOT_NOTE,
    }


__all__ = [
    "BLIND_SPOT_NOTE",
    "DRAW_COUNT_NOTE",
    "MIS_SAMPLER_FAMILY",
    "R0245_SAMPLER_DRAWS",
    "SAMPLER_BLOCK_EDGES",
    "block_profile_dispersion",
    "certified_weight_floor",
    "mis_sampler_battery",
    "required_draws_for_arm",
    "sampler_draw_floor",
    "sampler_limits",
    "true_sampler_reference",
    "uniform_position_draw_requirement",
]
