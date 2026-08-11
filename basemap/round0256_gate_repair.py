"""R0256 — repair the evidence under R0255's registered `n = 29` `MAD_n` gate.

Review-0255 registered the gate and blocked four claims made *about* it. Nothing
here changes a floor value: every floor and band re-fits from R0255's sealed panel
to the last bit. What changes is whether the statements published beside them were
established.

**1. The independence control did not run.** R0255's `independence_control` built
a `perturbed` copy of the held-out cells, counted it into the artifact, and then
called `_fit(series, log_series)` on the **unperturbed family**. The arms compared
a function of the family to itself, so `every_floor_bitwise_identical: true` was
structurally incapable of being false -- it would have reported `true` for a fit
that genuinely read held-out cells. This module rebuilds the control so the
perturbation reaches the fit: the fit's inputs are **derived from cells** by a
shipped builder, the perturbation is applied to the cells, and the builder runs
again. It then ships positive controls that plant *"the fit reads a held-out
cell"* in that builder and require the arms to report `false`.

**2. Two-sided bands were published with one-sided power.** `summarise_two_sided`
in R0234's calibration module never computed detection power, so both purity rows
carried the `ffr` floor's one-sided figure. The two-sided power of a band is
`E[Phi(L + delta) + 1 - Phi(U + delta)]`, which is smaller: `0.19381748` at
`-2 sigma`, not `0.30035090`.

**3. The panel false-alarm rate was never compounded.** A three-criterion panel
is not its per-metric rate.

**4. `registered_floors` was backwards.** `density_v2` -- descriptive-only by
standing rule, and the metric the R0161/R0193 audit found five downstream
artifacts consuming as a gate -- was the one entry left populated, while the two
gated purity criteria were nulled. The registry here publishes every gated
criterion in the form it actually gates (a scalar floor for `ffr`, a two-sided
ratio band for each purity metric) and refuses to carry a descriptive metric at
all, with positive controls for both directions.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from .round0234_calibrated_floors import (
    DESCRIPTIVE_METRICS,
    GATED_METRICS,
    PURITY_METRICS,
)
from .round0255_gate_n29 import (
    EXACT_FAMILY_SEEDS,
    GATE_CAPABILITY as R0255_GATE_CAPABILITY,
    OWNER_RULING_ESTIMATOR,
    family_series_from_cells,
    independence_control,
)
from .round0255_treatment import HELD_OUT_CELL_IDS, exact_cell_id


ROUND_ID = "0256"

GATE_CAPABILITY = "minilm-mixed-2m-calibrated-madn-floors-n29-v2"
GATE_SCHEMA = "round0256-minilm-mixed-2m-calibrated-madn-floors-n29-v2"
SUPERSEDES_CAPABILITY = R0255_GATE_CAPABILITY

#: The one estimator the owner ruled on. R0255's gate node calibrated all six --
#: including the two O(n^2) pairwise ones -- for five estimators the ruling
#: forbade acting on, and paid a 160.83 s uninterruptible stretch for the table.
REGISTERED_ESTIMATOR_NAMES: tuple[str, ...] = (OWNER_RULING_ESTIMATOR,)

DEFINING_CELL_IDS: tuple[str, ...] = tuple(
    exact_cell_id(seed) for seed in EXACT_FAMILY_SEEDS
)

#: The cell the owner ruling names, and the one arm 1 drives.
RULING_HELD_OUT_CELL = "cluster-spill-c8-seed42"


class Round0256RepairError(RuntimeError):
    """The R0256 repaired-evidence contract changed."""


# --------------------------------------------------------------------------- #
# A. the plants: builders that DO read a held-out cell
# --------------------------------------------------------------------------- #
#
# `family_series_from_cells` -- the builder under test -- lives beside the
# control it feeds, in `round0255_gate_n29`, because that is the shipped path.
# What lives here is the thing R0255 never shipped: builders that plant the
# defect, and a harness that requires the arms to report `false` against them.


def _builder_appends_every_held_out_cell(
    cells: Sequence[Mapping[str, Any]], *, defining_cell_ids: Sequence[str]
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """PLANT: the family grows to include the cells it judges (the v0 defect)."""
    held_out = [
        str(cell["cell_id"])
        for cell in cells
        if str(cell["cell_id"]) in set(HELD_OUT_CELL_IDS)
    ]
    return family_series_from_cells(
        cells, defining_cell_ids=list(defining_cell_ids) + held_out
    )


def _builder_substitutes_c8_seed42_for_a_family_cell(
    cells: Sequence[Mapping[str, Any]], *, defining_cell_ids: Sequence[str]
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """PLANT: one family slot is silently filled by the ruling's held-out cell."""
    wanted = list(defining_cell_ids)
    wanted[0] = RULING_HELD_OUT_CELL
    return family_series_from_cells(cells, defining_cell_ids=wanted)


def _builder_selects_every_cell_named_seed42(
    cells: Sequence[Mapping[str, Any]], *, defining_cell_ids: Sequence[str]
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """PLANT: a selection bug matching a seed suffix rather than the cell id."""
    extra = [
        str(cell["cell_id"])
        for cell in cells
        if str(cell["cell_id"]).endswith("seed42")
        and str(cell["cell_id"]) not in set(defining_cell_ids)
    ]
    return family_series_from_cells(
        cells, defining_cell_ids=list(defining_cell_ids) + extra
    )


PLANTED_BUILDERS: tuple[tuple[str, Callable[..., Any], str], ...] = (
    (
        "the_family_grows_to_include_every_cell_it_judges",
        _builder_appends_every_held_out_cell,
        "the v0 defect: R0195/R0204 qualified on R0161's own fitting cells",
    ),
    (
        "one_family_slot_is_filled_by_the_ruling_held_out_cell",
        _builder_substitutes_c8_seed42_for_a_family_cell,
        "the single-cell version: c8-seed42 becomes one of its own judges",
    ),
    (
        "the_selection_matches_a_seed_suffix_instead_of_a_cell_id",
        _builder_selects_every_cell_named_seed42,
        "a plausible selection bug that admits cuvs-igd48-seed42 and the c-arms",
    ),
)


# --------------------------------------------------------------------------- #
# B. the positive control R0255 did not ship
# --------------------------------------------------------------------------- #


def independence_positive_controls(
    *,
    estimator: str,
    multiplier_one_sided: float,
    multiplier_two_sided: float,
    family_cells: Sequence[Mapping[str, Any]],
    held_out_cells: Sequence[Mapping[str, Any]],
    defining_cell_ids: Sequence[str] = DEFINING_CELL_IDS,
) -> dict[str, Any]:
    """Plant "the fit reads a held-out cell" and require the arms to report false.

    Each plant swaps in a builder that selects one or more held-out cells into
    the fit, runs the SHIPPED `independence_control` unchanged against it, and
    requires arms 1 and 2 to go `false`. The honest builder runs in the same call
    as the negative control, so a plant that "catches" because everything fails
    is visible.
    """

    def run(build: Callable[..., Any]) -> dict[str, Any]:
        return independence_control(
            estimator=estimator,
            multiplier_one_sided=multiplier_one_sided,
            multiplier_two_sided=multiplier_two_sided,
            family_cells=family_cells,
            held_out_cells=held_out_cells,
            defining_cell_ids=defining_cell_ids,
            build=build,
        )

    honest = run(family_series_from_cells)
    plants: list[dict[str, Any]] = []
    for name, build, why in PLANTED_BUILDERS:
        # `build` is the planted object and it reaches the function under test
        # on the line below; `under_the_plant` is that call's OUTCOME.
        under_the_plant = run(build)
        arm1 = bool(under_the_plant["arms"][0]["every_floor_bitwise_identical"])
        arm2 = bool(under_the_plant["arms"][1]["every_floor_bitwise_identical"])
        caught = (not arm1) or (not arm2)
        plants.append({
            "plant": name,
            "why_this_is_the_defect": why,
            "builder": build.__name__,
            "arm_1_every_floor_bitwise_identical": arm1,
            "arm_2_every_floor_bitwise_identical": arm2,
            "the_control_reports_the_fit_independent": bool(
                under_the_plant["the_fit_is_independent_of_every_held_out_cell"]
            ),
            "caught": caught,
            "baseline_ffr_floor_under_the_plant": (
                under_the_plant["baseline_floors"]["ffr"]
            ),
        })
    return {
        "honest_builder": family_series_from_cells.__name__,
        "the_honest_builder_still_reports_independent": bool(
            honest["the_fit_is_independent_of_every_held_out_cell"]
        ),
        "the_honest_builder_is_not_inert": bool(honest["the_fit_is_not_inert"]),
        "plants": plants,
        "planted_defects": len(plants),
        "planted_defects_caught": sum(1 for row in plants if row["caught"]),
        "every_planted_defect_is_caught": all(row["caught"] for row in plants),
        "holds": bool(
            all(row["caught"] for row in plants)
            and honest["the_fit_is_independent_of_every_held_out_cell"]
            and honest["the_fit_is_not_inert"]
        ),
        "why_this_exists": (
            "Standing rule: every guard ships a positive control that plants the "
            "defect and is verified to fail when the shipped path is broken. "
            "R0255's independence control shipped none, which is why review-0255 "
            "could show its arms cannot fail."
        ),
    }


# --------------------------------------------------------------------------- #
# C. two-sided detection power
# --------------------------------------------------------------------------- #


def two_sided_detection_power(
    centre: np.ndarray,
    scale: np.ndarray,
    multiplier: float,
    *,
    alternatives: Sequence[float] = (1.0, 2.0, 3.0),
) -> dict[str, float]:
    """`E[Phi(L + delta) + 1 - Phi(U + delta)]` for the band `centre +/- k*scale`.

    R0234's `summarise_two_sided` reports delivered confidence and the false-fail
    rate but no power, so R0255 published the ONE-SIDED power beside both bands.
    A two-sided criterion judged by one-sided power overstates its sensitivity by
    `1.55x` at `-2 sigma`.
    """
    lower = np.asarray(centre) - float(multiplier) * np.asarray(scale)
    upper = np.asarray(centre) + float(multiplier) * np.asarray(scale)
    return {
        f"minus_{delta:g}_sigma": float(
            (
                stats.norm.cdf(lower + float(delta))
                + 1.0
                - stats.norm.cdf(upper + float(delta))
            ).mean()
        )
        for delta in alternatives
    }


def panel_false_alarm_rate(
    *, one_sided_rate: float, two_sided_rate: float, two_sided_criteria: int = 2
) -> dict[str, Any]:
    """Compound the per-metric rate across the gated panel, as the rule requires.

    `plan-minilm-100m-v2.md`: *"Compound the false-alarm rate across the panel
    before trusting it. A panel gate is not its per-metric rate."* No such field
    existed anywhere in R0255's sealed artifact.
    """
    one = float(one_sided_rate)
    two = float(two_sided_rate)
    compounded = 1.0 - (1.0 - one) * (1.0 - two) ** int(two_sided_criteria)
    return {
        "gated_criteria": ["ffr"] + list(PURITY_METRICS),
        "one_sided_per_metric_rate": one,
        "two_sided_per_metric_rate": two,
        "panel_false_alarm_rate_under_independence": compounded,
        "expression": (
            f"1 - (1 - {one!r}) * (1 - {two!r})**{int(two_sided_criteria)}"
        ),
        "ratio_to_the_one_sided_per_metric_rate": compounded / one if one else math.inf,
        "this_is_an_upper_bound": True,
        "why_it_is_an_upper_bound": (
            "the three gated criteria are correlated on this family, so the true "
            "panel rate is at or below the independence product. It is still the "
            "number a rung map actually faces, and R0255 published none."
        ),
    }


# --------------------------------------------------------------------------- #
# D. the registry: descriptive metrics are not floors; gated metrics are
# --------------------------------------------------------------------------- #


def registered_criteria(
    *,
    floors: Mapping[str, float],
    bands: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Every gated criterion in the form it gates. No descriptive metric appears."""
    out: dict[str, Any] = {}
    for metric in GATED_METRICS:
        if metric in PURITY_METRICS:
            band = bands[metric]
            out[metric] = {
                "kind": "two_sided_ratio_band",
                "ratio_lower": float(band["ratio_lower"]),
                "ratio_upper": float(band["ratio_upper"]),
                "gates_a_map": True,
                "read_as": (
                    "the UNFOLDED purity ratio must lie inside [ratio_lower, "
                    "ratio_upper]; there is no folded one-sided purity floor in "
                    "the decision path"
                ),
            }
            continue
        out[metric] = {
            "kind": "one_sided_lower_floor",
            "floor": float(floors[metric]),
            "gates_a_map": True,
            "read_as": "the metric must be at or above `floor`",
        }
    return out


def assert_registry_is_readable_only_as_registered(
    criteria: Mapping[str, Any], scalar_floors: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse a registry that publishes a descriptive metric as a gate.

    R0255's `registered_floors` carried `density_v2: 0.41643957035196294` while
    nulling the two gated purity entries -- exactly backwards relative to risk.
    `AUDIT-r0161-r0193-old-floors.md` records five downstream artifacts consuming
    such a value as a gate.
    """
    problems: list[str] = []
    for metric in DESCRIPTIVE_METRICS:
        if metric in criteria:
            problems.append(f"{metric} is descriptive-only and may not be a criterion")
        if scalar_floors.get(metric) is not None:
            problems.append(
                f"{metric} is descriptive-only and may not carry a readable floor"
            )
    for metric in GATED_METRICS:
        if metric not in criteria:
            problems.append(f"{metric} gates a map and must be registered")
        if metric in PURITY_METRICS:
            if criteria.get(metric, {}).get("kind") != "two_sided_ratio_band":
                problems.append(f"{metric} must be registered as a two-sided band")
            if scalar_floors.get(metric) is not None:
                problems.append(
                    f"{metric} is a band; a scalar floor for it is a folded misread"
                )
        else:
            if criteria.get(metric, {}).get("kind") != "one_sided_lower_floor":
                problems.append(f"{metric} must be registered as a one-sided floor")
            if scalar_floors.get(metric) is None:
                problems.append(f"{metric} is a scalar floor and must be readable")
    if problems:
        raise Round0256RepairError(
            "R0256 registry refuses: " + "; ".join(sorted(set(problems)))
        )
    return {
        "holds": True,
        "gated_criteria_registered": sorted(criteria),
        "descriptive_metrics_absent_from_the_registry": sorted(DESCRIPTIVE_METRICS),
        "scalar_floor_keys": sorted(scalar_floors),
    }


def registry_controls(
    *,
    criteria: Mapping[str, Any],
    scalar_floors: Mapping[str, Any],
    descriptive_values: Mapping[str, float],
) -> dict[str, Any]:
    """Positive controls for the registry guard, both directions."""
    honest_ok = True
    honest_error = None
    try:
        assert_registry_is_readable_only_as_registered(criteria, scalar_floors)
    except Round0256RepairError as error:
        honest_ok = False
        honest_error = f"{type(error).__name__}: {error}"

    plants: list[dict[str, Any]] = []

    def plant(name: str, crit: Mapping[str, Any], floors: Mapping[str, Any], why: str):
        refused = False
        message = None
        try:
            assert_registry_is_readable_only_as_registered(crit, floors)
        except Round0256RepairError as error:
            refused = True
            message = str(error)
        plants.append({
            "plant": name,
            "why_this_is_the_defect": why,
            "refused": refused,
            "error": message,
        })

    descriptive_metric = DESCRIPTIVE_METRICS[0]
    plant(
        "density_v2_left_readable_as_a_floor",
        criteria,
        {**dict(scalar_floors), descriptive_metric: float(
            descriptive_values[descriptive_metric]
        )},
        "R0255's shipped registry: the descriptive metric was the one entry left "
        "populated in a dict named registered_floors",
    )
    plant(
        "density_v2_registered_as_a_criterion",
        {**dict(criteria), descriptive_metric: {"kind": "one_sided_lower_floor"}},
        scalar_floors,
        "R0161 and R0193 both gated density_v2; the audit found five downstream "
        "artifacts consuming it as a gate",
    )
    plant(
        "a_gated_purity_band_dropped_from_the_registry",
        {k: v for k, v in criteria.items() if k != PURITY_METRICS[0]},
        scalar_floors,
        "the other half of R0255's inversion: the gating metrics were nulled",
    )
    plant(
        "a_purity_band_republished_as_a_folded_scalar_floor",
        criteria,
        {**dict(scalar_floors), PURITY_METRICS[0]: 0.962048853686814},
        "a folded one-sided purity floor is not the registered criterion and "
        "R0234 published one that was read as a gate",
    )
    plant(
        "the_ffr_floor_made_unreadable",
        criteria,
        {k: v for k, v in scalar_floors.items() if k != "ffr"},
        "nulling a gated scalar floor is the failure that made R0255's registry "
        "unusable in the other direction",
    )
    return {
        "the_honest_registry_passes": honest_ok,
        "honest_registry_error": honest_error,
        "plants": plants,
        "planted_defects": len(plants),
        "planted_defects_refused": sum(1 for row in plants if row["refused"]),
        "every_planted_defect_is_refused": all(row["refused"] for row in plants),
        "holds": bool(honest_ok and all(row["refused"] for row in plants)),
    }


__all__ = [
    "DEFINING_CELL_IDS",
    "GATE_CAPABILITY",
    "GATE_SCHEMA",
    "PLANTED_BUILDERS",
    "REGISTERED_ESTIMATOR_NAMES",
    "ROUND_ID",
    "RULING_HELD_OUT_CELL",
    "Round0256RepairError",
    "SUPERSEDES_CAPABILITY",
    "assert_registry_is_readable_only_as_registered",
    "independence_positive_controls",
    "panel_false_alarm_rate",
    "registered_criteria",
    "registry_controls",
    "two_sided_detection_power",
]
