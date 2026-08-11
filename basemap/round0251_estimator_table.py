"""R0251 — the six-candidate joint table, with the `c8-seed42` column attached.

review-0250-01 §B.4 and §B.5 make one finding that R0250 presented as two
independent questions:

* §B4 asked the owner whether R0234's power-first tie-break should stand at
  `n = 16` when it selects `S_n`, the *least* invariant qualifier, or whether the
  continuity choice `MAD_n` should be registered instead;
* §B6 published the `c8-seed42` reproduction — R0228's released `ffr` failure,
  un-failed by R0231 and R0234, failing again under the `n = 16` floor — as the
  round's strongest single result.

The review showed these are the same decision. `S_n` (`0.3116760`), `Q_n`
(`0.3163961`), sample-`s` (`0.3134620`) and 1-trimmed (`0.3109132`) all fail
`c8-seed42`; `MAD_n` (`0.3055370`) and `IQR_n` (`0.3050150`) do not. **Choosing
the more invariant standing instrument gives the reproduction back.**

This module builds that as ONE table so the coupling is visible rather than
inferred, and it deliberately **does not choose**. It re-derives every column
from the Gaussian null and the sealed cells — nothing is copied from R0250's
artifact, though every re-derived value is checked against it — and it adds the
column R0250's artifact did not have.

**The honesty clause this module exists to enforce.** Once the `c8-seed42`
column is in front of the decision, the reproduction stops being independent
evidence for the estimator that produces it. An estimator selected in part
because it reproduces a released failure cannot then have that reproduction
counted as confirmation of the selection; that is the same circularity R0234 was
criticised for when it retained R0225's floor to keep the failure alive. The
statement is published in the artifact as
`the_c8_seed42_reproduction_is_not_independent_evidence`, with the reason,
rather than left for a reviewer to supply.

**Pre-registration.** `PRE_REGISTERED_CRITERIA` fixes the criteria on which a
dominance claim may be made, and it is fixed BEFORE the `c8-seed42` column
exists — it names only quantities R0234 registered at `n = 13` and R0250
published at `n = 16`. If one candidate dominates on those, this module says so
and shows which criteria it was judged on; the `c8-seed42` column is excluded
from the dominance evaluation by construction.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .round0234_calibrated_floors import (
    CANDIDATE_ORDER,
    GATED_METRICS,
    PURITY_METRICS,
    REQUIRED_INVARIANCE_DEPTH,
    score_cell_metric,
)
from .round0250_gate_n16 import GATE_CAPABILITY as R0250_GATE_CAPABILITY, N_EXACT


ROUND_ID = "0251"

TABLE_CAPABILITY = "round0251-estimator-coupling-joint-table-n16-v1"
TABLE_SCHEMA = "round0251-estimator-coupling-joint-table-n16-v1"

#: The held-out cell the coupling turns on. R0228 published its `ffr` failure as
#: "the first cell in this program to fall below a released floor".
COUPLING_CELL_ID = "cluster-spill-c8-seed42"
COUPLING_CELL_CLUSTERS = 8
COUPLING_CELL_SEED = 42
COUPLING_METRIC = "ffr"

#: The second cell R0250 reported failing its own family, on `k1024`. Carried in
#: the table because it is the only other cell whose verdict this round's family
#: changes, so a reader can see whether the coupling is specific to one cell.
SECOND_CELL_ID = "cluster-spill-c8-seed43"
SECOND_CELL_SEED = 43
SECOND_METRIC = "purity_fidelity_k1024"

#: Fixed BEFORE the coupling column is computed. Every entry is a quantity
#: R0234 registered as a selection input at `n = 13`; none of them reads a cell
#: of the held-out set. A dominance claim may only be made on these.
PRE_REGISTERED_CRITERIA: tuple[dict[str, str], ...] = (
    {
        "criterion": "requirement_1_coverage",
        "direction": "must hold",
        "source": "R0234 SELECTION_RULE requirement 1",
    },
    {
        "criterion": "minimum_exact_invariance_depth",
        "direction": "higher is better",
        "source": f"R0234 SELECTION_RULE requirement 2, bar = {REQUIRED_INVARIANCE_DEPTH}",
    },
    {
        "criterion": "requirement_3_attainability",
        "direction": "must hold",
        "source": "R0234 SELECTION_RULE requirement 3",
    },
    {
        "criterion": "detection_power_at_minus_2_sigma",
        "direction": "higher is better",
        "source": "R0234 SELECTION_RULE tie-break 1",
    },
    {
        "criterion": "asymptotic_breakdown_point_at_this_n",
        "direction": "higher is better",
        "source": "R0234 SELECTION_RULE tie-break 2",
    },
    {
        "criterion": "new_cell_false_fail_rate_one_sided",
        "direction": "lower is better",
        "source": "R0234 calibration report, published beside every multiplier",
    },
    {
        "criterion": "qualifies_at_n13_as_well_as_n16",
        "direction": "higher is better",
        "source": (
            "review-0250-01 §B.5.1: a standing instrument re-derived at every rung "
            "must not change identity when the next cell lands. Measured from "
            "R0234's own sealed n=13 selection and this round's n=16 one."
        ),
    },
)
#: Explicitly NOT a dominance criterion, and named so the exclusion is auditable.
EXCLUDED_FROM_DOMINANCE: tuple[str, ...] = (
    "fails_the_coupling_cell",
    "fails_the_second_cell",
)

COUPLING_STATEMENT = (
    "The §B4 continuity question and the §B6 headline are ONE decision. Fitted "
    f"at n = {N_EXACT} on the same sixteen cells, the six candidates' `ffr` "
    f"floors straddle {COUPLING_CELL_ID}'s sealed R0228 value: the four with the "
    "higher floors fail it and the two with the lower floors pass it, and the "
    "more invariant standing instrument is on the passing side. So registering "
    "the estimator that review-0250-01 judges the better standing instrument "
    "GIVES BACK the reproduction R0250 called its strongest result, and "
    "registering the estimator that reproduces it keeps a family that clears the "
    "invariance bar by the minimum possible margin at this n. The owner cannot "
    "have both, and R0250 put the two to the owner as independent questions."
)

NOT_INDEPENDENT_EVIDENCE = (
    f"The {COUPLING_CELL_ID} reproduction is NOT independent evidence for any "
    "estimator whose selection is influenced by it. Once this table is in front "
    "of the decision, an estimator chosen partly because its floor reproduces a "
    "released failure cannot also count that reproduction as confirmation: the "
    "floor was fitted to the sixteen defining cells, the cell is held out, and "
    "the reproduction is a consequence of where the floor landed, not a test the "
    "floor passed. R0234 was criticised for exactly this shape when it retained "
    "R0225's floor to keep the same failure alive."
)

REGISTERS_NOTHING = (
    "R0251 registers no estimator, supersedes nothing, and does not recommend a "
    "choice. It publishes the joint table and the coupling; the selection is the "
    "owner's."
)


class Round0251TableError(RuntimeError):
    """The registered R0251 joint-table contract changed."""


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise Round0251TableError(f"R0251 {label} is not finite: {value!r}")
    return number


def held_out_cell(
    comparison: Mapping[str, Any], *, clusters: int, seed: int
) -> dict[str, Any]:
    """One cluster-spill cell, read from R0228's sealed comparison bytes."""
    cells = dict(comparison["candidate_panel_metric_cells"])[str(clusters)]
    ratios = dict(comparison["candidate_purity_ratios"])[str(clusters)]
    key = str(seed)
    return {
        "cell_id": f"cluster-spill-c{clusters}-seed{seed}",
        "family": f"cluster-spill-c{clusters}",
        "values": {
            metric: _finite(cells[key][metric], f"R0228 c{clusters} seed-{seed} {metric}")
            for metric in dict(cells[key])
        },
        "ratios": {
            ratio_key: _finite(
                ratios[key][ratio_key], f"R0228 c{clusters} seed-{seed} {ratio_key}"
            )
            for ratio_key in ("k256", "k1024")
        },
    }


def coupling_column(
    *,
    cell: Mapping[str, Any],
    floors: Mapping[str, float],
    bands: Mapping[str, tuple[float, float]],
    metric: str,
) -> dict[str, Any]:
    """Does this candidate's fitted criterion fail this held-out cell?

    Scored with R0234's own released `score_cell_metric`, on the raw unfolded
    ratio for a purity metric and the value for `ffr`, so the arithmetic is the
    gate's rather than this module's.
    """
    if metric in PURITY_METRICS:
        ratio_key = "k256" if metric.endswith("k256") else "k1024"
        band = tuple(dict(bands)[metric])
        verdict = score_cell_metric(
            value=float(dict(cell["values"])[metric]),
            floor=float(dict(floors)[metric]) if metric in dict(floors) else None,
            ratio=float(dict(cell["ratios"])[ratio_key]),
            band=(float(band[0]), float(band[1])),
        )
    else:
        verdict = score_cell_metric(
            value=float(dict(cell["values"])[metric]),
            floor=float(dict(floors)[metric]),
        )
    return {
        "cell_id": str(cell["cell_id"]),
        "metric": metric,
        "verdict": dict(verdict),
        "fails": not bool(verdict["passes"]),
    }


def joint_table(
    *,
    selection: Mapping[str, Any],
    fitted: Mapping[str, Any],
    n13_selection: Mapping[str, Any],
    coupling_cell: Mapping[str, Any],
    second_cell: Mapping[str, Any],
) -> dict[str, Any]:
    """One row per candidate: every selection input, the floor, and the coupling.

    `selection` and `fitted` are this round's own re-derivation at `n = 16`;
    `n13_selection` is R0234's sealed `selection.candidates`, used only for the
    `qualifies_at_n13_as_well_as_n16` column that review-0250-01 §B.5.1 asked for.
    """
    rows: list[dict[str, Any]] = []
    for name in CANDIDATE_ORDER:
        item = dict(selection["candidates"][name])
        fit = dict(fitted[name])
        floors = {metric: float(fit["floors"][metric]) for metric in GATED_METRICS}
        bands = {
            metric: (
                float(fit["two_sided_ratio_bands"][metric]["ratio_lower"]),
                float(fit["two_sided_ratio_bands"][metric]["ratio_upper"]),
            )
            for metric in PURITY_METRICS
        }
        coupling = coupling_column(
            cell=coupling_cell, floors=floors, bands=bands, metric=COUPLING_METRIC
        )
        second = coupling_column(
            cell=second_cell, floors=floors, bands=bands, metric=SECOND_METRIC
        )
        n13 = dict(n13_selection).get(name) or {}
        qualified_at_13 = n13.get("qualifies")
        if qualified_at_13 is None:
            raise Round0251TableError(
                f"R0234's sealed selection has no `qualifies` for {name}"
            )
        rows.append({
            "estimator": name,
            "n": N_EXACT,
            "calibrated_one_sided_multiplier": float(
                item["calibrated_one_sided_multiplier"]
            ),
            "calibrated_two_sided_multiplier": float(
                item["calibrated_two_sided_multiplier"]
            ),
            "delivered_one_sided_coverage": float(item["delivered_one_sided_coverage"]),
            "delivered_two_sided_confidence": float(
                item["delivered_two_sided_confidence"]
            ),
            "new_cell_false_fail_rate_one_sided": float(
                item["new_cell_false_fail_rate_one_sided"]
            ),
            "new_cell_false_fail_rate_two_sided": float(
                item["new_cell_false_fail_rate_two_sided"]
            ),
            "detection_power_at_minus_2_sigma": float(
                item["detection_power_at_selection_alternative"]
            ),
            "minimum_exact_invariance_depth": int(
                item["minimum_exact_invariance_depth"]
            ),
            "exact_invariance_depth_by_series": dict(
                item["exact_invariance_depth_by_series"]
            ),
            "asymptotic_breakdown_point_at_this_n": float(
                item["asymptotic_breakdown_point_at_this_n"]
            ),
            "requirement_1_coverage": bool(item["requirement_1_coverage"]),
            "requirement_2_invariance": bool(item["requirement_2_invariance"]),
            "requirement_3_attainability": bool(item["requirement_3_attainability"]),
            "qualifies_at_n16": bool(item["qualifies"]),
            "qualifies_at_n13": bool(qualified_at_13),
            "qualifies_at_n13_as_well_as_n16": bool(
                bool(item["qualifies"]) and bool(qualified_at_13)
            ),
            "fitted_ffr_floor_at_n16": float(fit["floors"]["ffr"]),
            "fitted_purity_bands_at_n16": {
                metric: {
                    "ratio_lower": bands[metric][0],
                    "ratio_upper": bands[metric][1],
                }
                for metric in PURITY_METRICS
            },
            "coupling_cell": COUPLING_CELL_ID,
            "coupling_cell_value": float(
                dict(coupling_cell["values"])[COUPLING_METRIC]
            ),
            "fails_the_coupling_cell": bool(coupling["fails"]),
            "coupling_cell_verdict": coupling["verdict"],
            "second_cell": SECOND_CELL_ID,
            "fails_the_second_cell": bool(second["fails"]),
            "second_cell_verdict": second["verdict"],
        })
    return {
        "n": N_EXACT,
        "rows": rows,
        "coupling_cell": COUPLING_CELL_ID,
        "coupling_metric": COUPLING_METRIC,
        "coupling_cell_sealed_value": float(
            dict(coupling_cell["values"])[COUPLING_METRIC]
        ),
        "candidates_failing_the_coupling_cell": [
            row["estimator"] for row in rows if row["fails_the_coupling_cell"]
        ],
        "candidates_passing_the_coupling_cell": [
            row["estimator"] for row in rows if not row["fails_the_coupling_cell"]
        ],
        "coupling_statement": COUPLING_STATEMENT,
        "the_c8_seed42_reproduction_is_not_independent_evidence": (
            NOT_INDEPENDENT_EVIDENCE
        ),
        "registers_nothing": REGISTERS_NOTHING,
    }


def _better(criterion: str, direction: str, left: Any, right: Any) -> int:
    """1 if `left` beats `right` on this criterion, -1 if worse, 0 if equal."""
    if direction == "must hold":
        left_ok, right_ok = bool(left), bool(right)
        if left_ok == right_ok:
            return 0
        return 1 if left_ok else -1
    left_value = float(left)
    right_value = float(right)
    if left_value == right_value:
        return 0
    if direction == "higher is better":
        return 1 if left_value > right_value else -1
    return 1 if left_value < right_value else -1


def dominance(table: Mapping[str, Any]) -> dict[str, Any]:
    """Does any candidate dominate on the PRE-REGISTERED criteria alone?

    Dominance means: at least as good on every pre-registered criterion and
    strictly better on at least one, against every other candidate. The coupling
    columns are excluded by construction — they are not in
    `PRE_REGISTERED_CRITERIA` and this function never reads them.
    """
    rows = {str(row["estimator"]): dict(row) for row in table["rows"]}
    criteria = [dict(item) for item in PRE_REGISTERED_CRITERIA]
    for row in rows.values():
        for excluded in EXCLUDED_FROM_DOMINANCE:
            if excluded in {item["criterion"] for item in criteria}:
                raise Round0251TableError(
                    f"R0251 dominance may not read {excluded}"
                )
    dominators: list[str] = []
    pairwise: dict[str, Any] = {}
    for name, row in rows.items():
        comparisons: dict[str, Any] = {}
        dominates_all = True
        for other_name, other in rows.items():
            if other_name == name:
                continue
            signs = {
                item["criterion"]: _better(
                    item["criterion"],
                    item["direction"],
                    row[item["criterion"]],
                    other[item["criterion"]],
                )
                for item in criteria
            }
            never_worse = all(value >= 0 for value in signs.values())
            strictly_better = any(value > 0 for value in signs.values())
            comparisons[other_name] = {
                "signs": signs,
                "dominates": bool(never_worse and strictly_better),
            }
            if not (never_worse and strictly_better):
                dominates_all = False
        pairwise[name] = comparisons
        if dominates_all:
            dominators.append(name)
    return {
        "pre_registered_criteria": criteria,
        "excluded_from_dominance": list(EXCLUDED_FROM_DOMINANCE),
        "pairwise": pairwise,
        "dominating_candidates": dominators,
        "a_candidate_dominates": bool(dominators),
        "note": (
            "dominance here is Pareto dominance over the pre-registered criteria "
            "only. R0234's SELECTION_RULE is a LEXICOGRAPHIC rule over a subset "
            "of them, so a rule-selected estimator need not dominate and a "
            "dominating estimator need not be rule-selected. Both are reported; "
            "neither is a recommendation."
        ),
    }


def reproduction_checks(
    *,
    table: Mapping[str, Any],
    r0250_gate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Every re-derived column against R0250's sealed artifact, at tolerance 0.

    This is what makes the table an independent instrument rather than a copy:
    it is built from the Gaussian null and the sealed cells, and only then
    compared. A mismatch is a failure of this round, not of R0250.
    """
    sealed_selection = dict(dict(r0250_gate["selection"])["candidates"])
    sealed_fitted = dict(r0250_gate["fitted_candidates"])
    if str(r0250_gate.get("capability") or "") != R0250_GATE_CAPABILITY:
        raise Round0251TableError("R0251 was handed a non-R0250 gate artifact")
    checks: list[dict[str, Any]] = []
    for row in table["rows"]:
        name = str(row["estimator"])
        sealed = dict(sealed_selection[name])
        fit = dict(sealed_fitted[name])
        for key, observed, target in (
            (
                "calibrated_one_sided_multiplier",
                row["calibrated_one_sided_multiplier"],
                sealed["calibrated_one_sided_multiplier"],
            ),
            (
                "calibrated_two_sided_multiplier",
                row["calibrated_two_sided_multiplier"],
                sealed["calibrated_two_sided_multiplier"],
            ),
            (
                "detection_power_at_minus_2_sigma",
                row["detection_power_at_minus_2_sigma"],
                sealed["detection_power_at_selection_alternative"],
            ),
            (
                "new_cell_false_fail_rate_one_sided",
                row["new_cell_false_fail_rate_one_sided"],
                sealed["new_cell_false_fail_rate_one_sided"],
            ),
            (
                "minimum_exact_invariance_depth",
                row["minimum_exact_invariance_depth"],
                sealed["minimum_exact_invariance_depth"],
            ),
            (
                "fitted_ffr_floor_at_n16",
                row["fitted_ffr_floor_at_n16"],
                fit["floors"]["ffr"],
            ),
        ):
            delta = abs(float(observed) - float(target))
            checks.append({
                "estimator": name,
                "key": key,
                "observed": float(observed),
                "r0250_sealed": float(target),
                "delta": delta,
                "tolerance": 0.0,
                "reproduced": delta == 0.0,
            })
    return checks


__all__ = [
    "COUPLING_CELL_CLUSTERS",
    "COUPLING_CELL_ID",
    "COUPLING_CELL_SEED",
    "COUPLING_METRIC",
    "COUPLING_STATEMENT",
    "EXCLUDED_FROM_DOMINANCE",
    "NOT_INDEPENDENT_EVIDENCE",
    "PRE_REGISTERED_CRITERIA",
    "REGISTERS_NOTHING",
    "ROUND_ID",
    "Round0251TableError",
    "SECOND_CELL_ID",
    "SECOND_CELL_SEED",
    "SECOND_METRIC",
    "TABLE_CAPABILITY",
    "TABLE_SCHEMA",
    "coupling_column",
    "dominance",
    "held_out_cell",
    "joint_table",
    "reproduction_checks",
]
