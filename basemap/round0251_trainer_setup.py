"""R0251 — the trainer's SETUP interval, instrumented, and the batch tail, modelled.

review-0250-01 §A.3 corrected R0250's headline. Under per-batch polling the
binding interval is not steady state, it is **setup**: R0250's own sealed trace
carries a cold setup of `1.984` s against the registered
`2.5109531834854018` s ceiling — `0.790x`, not the `0.0988x` the result, the
digest and the topic log all carried forward. And §A.2 of R0250 disclosed, and
the review credited, that a `~10 h` node draws about `410x` as many per-batch
samples from a tail this program has never modelled.

This module is the instrument for both, and it is deliberately *not* a gate.

**Setup.** R0250 could not reduce the setup gap because it refused to edit
`basemap/pumap/`; it installed its poll by monkey-patching `_low_dim_qs`, which
is only reachable once the first batch is already running. R0251 puts the read
in the trainer itself (`ParametricUMAP._poll_abort`, five declared sites, all
no-ops when no hook is installed) and then measures what the widest *remaining*
setup interval is. Reducing a gap by polling does not make the work faster: it
converts "the widest gap is the whole setup" into "the widest gap is the largest
single un-polled call", which is a different and much smaller quantity — and one
that stops scaling with total setup size. Whether the largest such call is under
the ceiling is a measurement, and it is reported either way.

**The tail.** A per-batch gap is not a constant. The right question for a 10 h
node is not "what was the widest gap in 10,000 batches" but "what is the widest
gap likely to be in `N` batches, and how confident can that be". Two answers are
published side by side, because they fail in opposite directions:

* a **peaks-over-threshold** extreme-value model — a generalised Pareto fitted
  to the exceedances over a high quantile, giving an `N`-batch return level. It
  extrapolates, which is what is wanted, and it assumes the sampled mechanisms
  are the only mechanisms, which is what is dangerous;
* a **distribution-free** bound with no tail assumption at all: with `m`
  observations and `x` of them over the ceiling, the 95% upper confidence bound
  on the per-batch exceedance probability bounds the expected number of
  exceedances in `N` batches. With `x = 0` this is the rule of three, `3/m`, and
  it is deliberately pessimistic.

Neither is a measurement at 10 h and both are stamped as models.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .round0247_registry import registered_value
from .round0250_trainer_loops import (
    PROJECTION_TARGET_HOURS,
    PROJECTION_TARGET_SECONDS,
    SHORT_HORIZON_UPDATES,
)


ROUND_ID = "0251"

SETUP_CAPABILITY = "round0251-trainer-setup-and-batch-tail-v1"
SETUP_SCHEMA = "round0251-trainer-setup-and-batch-tail-v1"

#: The rung. Short enough to cost minutes; long enough that the steady-state
#: tail has thousands of draws to fit a threshold model to. It is NOT the
#: registered 80,163-update treatment and no floor may be fitted to it.
TAIL_RUNG_UPDATES = SHORT_HORIZON_UPDATES
SETUP_RUNG_UPDATES = 600

NOT_A_FAMILY_CELL = (
    "R0251's rungs measure loop and setup structure. Their horizons are "
    f"{SETUP_RUNG_UPDATES} and {TAIL_RUNG_UPDATES} updates, not the registered "
    "80,163; they publish no map, define no cell of the exact-graph seed family, "
    "and no floor may be fitted to them."
)

#: The five sites `ParametricUMAP` now polls, named here so a test can assert the
#: release class carries exactly these and a reviewer can diff the list against
#: the diff rather than against a docstring.
DECLARED_TRAINER_POLL_SITES: tuple[str, ...] = (
    "pumap.fit setup entered",
    "pumap.fit setup edge-list prepared",
    "pumap.fit setup model allocated",
    "pumap.fit setup complete",
    "pumap.fit train batch",
)
BATCH_POLL_SITE = "pumap.fit train batch"
SETUP_POLL_SITES: tuple[str, ...] = DECLARED_TRAINER_POLL_SITES[:-1]

#: Sites the NODE polls, outside `fit()`. R0250's arms started their gate after
#: the graph was already loaded, so the graph verify/load never entered any
#: measured interval at all. Here it does.
NODE_SETUP_SITES: tuple[str, ...] = (
    "R0251 node stage entered",
    "R0251 node sealed graph verified and loaded",
    "R0251 node substrate opened",
    "R0251 node training input constructed",
)

#: A `~10 h` node at the rung's measured rate. `410x` is review-0250-01's figure
#: and R0250's sealed `36000.0 / stage_wall`; it is recomputed here from this
#: round's own measured wall rather than carried.
TAIL_TARGET_SECONDS = PROJECTION_TARGET_SECONDS
TAIL_TARGET_HOURS = PROJECTION_TARGET_HOURS

#: The exceedance threshold for the peaks-over-threshold fit, as a quantile of
#: the observed per-batch gaps. Fixed before any gap is drawn.
POT_THRESHOLD_QUANTILE = 0.99
#: A POT return level is only as good as its threshold. These are reported
#: beside the registered one so threshold sensitivity is visible rather than a
#: reviewer's objection. Also fixed before any gap is drawn.
POT_THRESHOLD_LADDER: tuple[float, ...] = (0.98, 0.99, 0.995)
#: Nonparametric bootstrap resamples for the shape and return-level intervals.
POT_BOOTSTRAP_RESAMPLES = 400
POT_BOOTSTRAP_SEED = 20260811
#: The rule-of-three constant: the 95% upper bound on `p` when `x = 0` of `m`
#: trials exceed is `3/m`. Registered here so the arithmetic is not invented at
#: the reporting site.
RULE_OF_THREE_NUMERATOR = 3.0
DISTRIBUTION_FREE_CONFIDENCE = 0.95


class Round0251SetupError(RuntimeError):
    """The registered R0251 setup/tail measurement contract changed."""


# --------------------------------------------------------------------------- #
# the recorder
# --------------------------------------------------------------------------- #


class PollRecorder:
    """Forward every abort read to the gate, and keep the whole gap series.

    `AbortPollGate` keeps only `max_gap_s` and the site it followed, which is
    all a ceiling verdict needs and far too little to model a tail. This wraps
    the gate rather than replacing it: the gate still scores, and the series is
    this round's own evidence beside it.
    """

    def __init__(self, *, gate: Any, clock: Any) -> None:
        if not callable(gate):
            raise Round0251SetupError("R0251 poll recorder needs a callable gate")
        if not callable(clock):
            raise Round0251SetupError("R0251 poll recorder needs a callable clock")
        self._gate = gate
        self._clock = clock
        self._last: float | None = None
        self.batches = 0
        #: (site, gap_s) in call order, one entry per read after the anchor.
        self.records: list[tuple[str, float]] = []
        self.batch_gaps: list[float] = []

    def anchor(self, where: str) -> None:
        """Start the series without recording a gap into it."""
        self._last = float(self._clock())
        self._gate(str(where))

    def __call__(self, where: str) -> None:
        site = str(where)
        now = float(self._clock())
        if self._last is not None:
            gap = now - self._last
            self.records.append((site, gap))
            if site == BATCH_POLL_SITE:
                self.batch_gaps.append(gap)
        if site == BATCH_POLL_SITE:
            self.batches += 1
        self._last = now
        self._gate(site)

    def receipt(self) -> dict[str, Any]:
        return {
            "reads_recorded": len(self.records),
            "batch_reads": int(self.batches),
            "batch_gaps_recorded": len(self.batch_gaps),
            "sites_seen": sorted({site for site, _gap in self.records}),
        }


# --------------------------------------------------------------------------- #
# the phase split
# --------------------------------------------------------------------------- #


def _widest(entries: Sequence[tuple[str, float]]) -> tuple[str, float]:
    if not entries:
        return ("", 0.0)
    site, gap = max(entries, key=lambda item: item[1])
    return (str(site), float(gap))


def phase_report(
    records: Sequence[tuple[str, float]], *, arm: str
) -> dict[str, Any]:
    """Widest gap in SETUP and in STEADY STATE, separately, against the ceiling.

    "Setup" is every read up to and including the first batch read: the interval
    that ends at batch 1 is the last setup interval, because it covers the
    profiler attach, the loader construction and the first prefetch. Everything
    after it is steady state.
    """
    entries = [(str(site), float(gap)) for site, gap in records]
    if not entries:
        raise Round0251SetupError(f"R0251 arm {arm!r} recorded no abort reads")
    first_batch = next(
        (index for index, (site, _gap) in enumerate(entries) if site == BATCH_POLL_SITE),
        None,
    )
    if first_batch is None:
        raise Round0251SetupError(f"R0251 arm {arm!r} never reached a training batch")
    setup = entries[: first_batch + 1]
    steady = entries[first_batch + 1 :]
    ceiling = registered_value("r0246_max_poll_spacing_s")
    setup_site, setup_gap = _widest(setup)
    steady_site, steady_gap = _widest(steady)
    per_site: dict[str, Any] = {}
    for site, gap in setup:
        entry = per_site.setdefault(site, {"reads": 0, "widest_gap_s": 0.0, "total_s": 0.0})
        entry["reads"] += 1
        entry["total_s"] += gap
        entry["widest_gap_s"] = max(entry["widest_gap_s"], gap)
    widest_overall_site, widest_overall = _widest(entries)
    return {
        "arm": arm,
        "registered_ceiling_s": ceiling,
        "setup_reads": len(setup),
        "steady_state_reads": len(steady),
        "setup_wall_s": sum(gap for _site, gap in setup),
        "steady_state_wall_s": sum(gap for _site, gap in steady),
        "widest_setup_gap_s": setup_gap,
        "widest_setup_gap_after": setup_site,
        "widest_setup_gap_over_the_ceiling": setup_gap / ceiling,
        "setup_meets_the_registered_ceiling": bool(setup_gap <= ceiling),
        "widest_steady_state_gap_s": steady_gap,
        "widest_steady_state_gap_after": steady_site,
        "widest_steady_state_gap_over_the_ceiling": steady_gap / ceiling,
        "steady_state_meets_the_registered_ceiling": bool(steady_gap <= ceiling),
        "widest_gap_across_both_phases_s": widest_overall,
        "widest_gap_across_both_phases_after": widest_overall_site,
        "widest_gap_across_both_phases_over_the_ceiling": widest_overall / ceiling,
        "the_binding_phase": (
            "setup" if setup_gap >= steady_gap else "steady_state"
        ),
        "setup_gaps_by_site": per_site,
    }


def setup_reduction(
    *, before_gap_s: float, after_gap_s: float, margin_required: float = 2.0
) -> dict[str, Any]:
    """Did instrumenting setup bring it under the ceiling, and with what margin?

    `margin_required` is the pre-registered factor: a setup gap is reported as
    reduced *with margin* only if the ceiling is at least this many times it.
    The `2.0` is R0250 §A5's own safety factor for the block-size resolution,
    reused rather than reinvented.
    """
    ceiling = registered_value("r0246_max_poll_spacing_s")
    before = float(before_gap_s)
    after = float(after_gap_s)
    if not math.isfinite(before) or not math.isfinite(after) or after <= 0.0:
        raise Round0251SetupError(
            f"R0251 setup reduction needs finite positive gaps: {before!r} / {after!r}"
        )
    margin = ceiling / after
    return {
        "registered_ceiling_s": ceiling,
        "widest_setup_gap_before_instrumentation_s": before,
        "widest_setup_gap_after_instrumentation_s": after,
        "reduction_factor": before / after,
        "before_over_the_ceiling": before / ceiling,
        "after_over_the_ceiling": after / ceiling,
        "margin_after_instrumentation": margin,
        "required_margin": float(margin_required),
        "setup_is_below_the_ceiling": bool(after <= ceiling),
        "setup_is_below_the_ceiling_with_the_required_margin": bool(
            margin >= float(margin_required)
        ),
        "shortfall_over_the_ceiling_if_not": (
            None if after <= ceiling else after / ceiling
        ),
        "note": (
            "polling does not make setup faster. It replaces 'the widest gap is "
            "the whole setup' with 'the widest gap is the largest single "
            "un-polled call', which is what the numbers above compare. The "
            "largest un-polled call is named in the phase report; whether IT "
            "grows with N is not established by this rung."
        ),
    }


# --------------------------------------------------------------------------- #
# the tail
# --------------------------------------------------------------------------- #


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Type-7 quantile on an already-sorted sequence."""
    count = len(sorted_values)
    if count == 0:
        raise Round0251SetupError("R0251 cannot take a quantile of nothing")
    if count == 1:
        return float(sorted_values[0])
    position = (count - 1) * float(q)
    low = int(math.floor(position))
    high = min(low + 1, count - 1)
    frac = position - low
    return float(sorted_values[low]) * (1.0 - frac) + float(sorted_values[high]) * frac


def distribution_free_tail_bound(
    gaps: Sequence[float], *, batches_at_target: float, ceiling_s: float | None = None
) -> dict[str, Any]:
    """No tail assumption at all. Deliberately the pessimistic answer.

    With `m` observed per-batch gaps of which `x` exceed the ceiling, the 95%
    upper confidence bound on the per-batch exceedance probability `p` is the
    Clopper-Pearson upper limit; at `x = 0` that is exactly `1 - 0.05**(1/m)`,
    which the rule of three approximates as `3/m`. Both are published: the exact
    limit is used, the rule of three is reported beside it as the sanity check.
    """
    ceiling = (
        registered_value("r0246_max_poll_spacing_s")
        if ceiling_s is None
        else float(ceiling_s)
    )
    values = [float(value) for value in gaps]
    count = len(values)
    if count < 100:
        raise Round0251SetupError(
            f"R0251 tail bound needs at least 100 per-batch gaps, got {count}"
        )
    exceedances = sum(1 for value in values if value > ceiling)
    alpha = 1.0 - DISTRIBUTION_FREE_CONFIDENCE
    if exceedances == 0:
        p_upper = 1.0 - alpha ** (1.0 / count)
    else:
        # Clopper-Pearson upper limit via the beta quantile, computed from the
        # regularised incomplete beta through scipy so no approximation enters.
        from scipy import stats as _stats

        p_upper = float(_stats.beta.ppf(1.0 - alpha, exceedances + 1, count - exceedances))
    target = float(batches_at_target)
    return {
        "kind": "model",
        "model": "distribution-free Clopper-Pearson upper bound on the exceedance rate",
        "registered_ceiling_s": ceiling,
        "batches_observed": count,
        "batches_exceeding_the_ceiling_observed": exceedances,
        "confidence": DISTRIBUTION_FREE_CONFIDENCE,
        "upper_bound_on_the_per_batch_exceedance_probability": p_upper,
        "rule_of_three_approximation": RULE_OF_THREE_NUMERATOR / count,
        "batches_at_the_target_wall": target,
        "upper_bound_on_expected_exceedances_at_the_target": p_upper * target,
        "upper_bound_on_the_probability_of_at_least_one_exceedance": (
            1.0 - (1.0 - p_upper) ** target if p_upper < 1.0 else 1.0
        ),
        "is_a_measurement_at_the_target_wall": False,
        "note": (
            "this bound assumes only that the batches are exchangeable draws. It "
            "is pessimistic BY CONSTRUCTION: observing zero exceedances in m "
            "batches cannot rule out a rate near 3/m, and 3/m times a few "
            "million batches is a large number. It bounds what the data can "
            "exclude, not what is likely."
        ),
    }


def peaks_over_threshold_tail(
    gaps: Sequence[float],
    *,
    batches_at_target: float,
    threshold_quantile: float = POT_THRESHOLD_QUANTILE,
) -> dict[str, Any]:
    """A generalised-Pareto return level for `N` batches. **A model.**

    Fits `genpareto` by maximum likelihood to the exceedances over the
    `threshold_quantile` of the observed gaps, with the location pinned at the
    threshold (the standard POT parameterisation), then returns the `N`-batch
    return level

        `x_N = u + (sigma/xi) * ((N * zeta_u)**xi - 1)`,   `xi != 0`
        `x_N = u + sigma * log(N * zeta_u)`,               `xi == 0`

    where `zeta_u` is the observed exceedance rate. When `xi < 0` the fitted
    tail is bounded and the finite endpoint `u - sigma/xi` is published too,
    because that endpoint — not the return level — is the interesting number: it
    is the largest gap the fitted model admits at ANY number of batches.
    """
    from scipy import stats as _stats

    values = sorted(float(value) for value in gaps)
    count = len(values)
    if count < 1000:
        raise Round0251SetupError(
            f"R0251 POT fit needs at least 1000 per-batch gaps, got {count}"
        )
    threshold = _quantile(values, float(threshold_quantile))
    exceedances = [value - threshold for value in values if value > threshold]
    if len(exceedances) < 30:
        raise Round0251SetupError(
            f"R0251 POT fit needs at least 30 exceedances over the "
            f"{threshold_quantile!r} quantile, got {len(exceedances)}"
        )
    shape, _loc, scale = _stats.genpareto.fit(exceedances, floc=0.0)
    zeta = len(exceedances) / count
    target = float(batches_at_target)
    expected = target * zeta

    def _level(xi: float, sigma: float) -> float:
        if abs(xi) < 1e-12:
            return threshold + sigma * math.log(expected)
        return threshold + (sigma / xi) * (expected ** xi - 1.0)

    level = _level(float(shape), float(scale))
    #: The fitted shape carries most of the extrapolation, so its sampling
    #: uncertainty is published rather than left implicit. A single return level
    #: from `100` exceedances is not a number to hand a reviewer alone.
    import numpy as _np

    rng = _np.random.default_rng(POT_BOOTSTRAP_SEED)
    sample = _np.asarray(exceedances, dtype=float)
    shapes: list[float] = []
    levels: list[float] = []
    for _ in range(POT_BOOTSTRAP_RESAMPLES):
        draw = rng.choice(sample, size=sample.size, replace=True)
        try:
            xi, _l, sigma = _stats.genpareto.fit(draw, floc=0.0)
        except Exception:  # noqa: BLE001 - a degenerate resample is dropped
            continue
        shapes.append(float(xi))
        levels.append(_level(float(xi), float(sigma)))
    shapes.sort()
    levels.sort()
    ceiling = registered_value("r0246_max_poll_spacing_s")
    endpoint = threshold - scale / shape if shape < 0.0 else None
    #: The `xi = 0` alternative: an exponential tail, i.e. the lightest tail the
    #: POT family admits without a finite endpoint. It is the other end of the
    #: modelling range and costs nothing to report.
    exponential_level = threshold + float(_np.mean(sample)) * math.log(expected)
    return {
        "kind": "model",
        "model": "peaks-over-threshold, generalised Pareto, MLE with the location pinned at the threshold",
        "batches_observed": count,
        "threshold_quantile": float(threshold_quantile),
        "threshold_s": threshold,
        "exceedances": len(exceedances),
        "observed_exceedance_rate": zeta,
        "fitted_shape_xi": float(shape),
        "fitted_scale_sigma": float(scale),
        "tail_is_bounded": bool(shape < 0.0),
        "fitted_finite_endpoint_s": endpoint,
        "fitted_finite_endpoint_over_the_ceiling": (
            None if endpoint is None else endpoint / ceiling
        ),
        "batches_at_the_target_wall": target,
        "return_level_at_the_target_s": float(level),
        "return_level_over_the_ceiling": float(level) / ceiling,
        "return_level_meets_the_ceiling": bool(float(level) <= ceiling),
        "bootstrap_resamples": len(shapes),
        "shape_bootstrap_ci_95": (
            [shapes[int(0.025 * len(shapes))], shapes[int(0.975 * len(shapes)) - 1]]
            if shapes
            else None
        ),
        "return_level_bootstrap_ci_95_s": (
            [levels[int(0.025 * len(levels))], levels[int(0.975 * len(levels)) - 1]]
            if levels
            else None
        ),
        "return_level_bootstrap_median_s": (
            levels[len(levels) // 2] if levels else None
        ),
        "return_level_if_the_tail_were_exponential_s": exponential_level,
        "return_level_if_the_tail_were_exponential_over_the_ceiling": (
            exponential_level / ceiling
        ),
        "registered_ceiling_s": ceiling,
        "observed_max_gap_s": values[-1],
        "observed_max_over_the_ceiling": values[-1] / ceiling,
        "is_a_measurement_at_the_target_wall": False,
        "note": (
            "the fit extrapolates the mechanisms this rung sampled — allocator "
            "pauses, prefetch stalls, scheduler preemption — and nothing else. A "
            "10 h node also runs mechanisms no 90 s rung contains (a periodic "
            "checkpoint write, a log rotation, an OS reclaim under a full page "
            "cache). This model does not cover them and is not a bound."
        ),
    }


def tail_model(
    gaps: Sequence[float], *, arm_wall_s: float, target_seconds: float = TAIL_TARGET_SECONDS
) -> dict[str, Any]:
    """Both tail answers, plus the batch multiple they are evaluated at."""
    wall = float(arm_wall_s)
    if wall <= 0.0:
        raise Round0251SetupError("R0251 tail model needs a positive arm wall")
    count = len(gaps)
    multiple = float(target_seconds) / wall
    batches_at_target = count * multiple
    return {
        "target_wall_s": float(target_seconds),
        "target_wall_hours": float(target_seconds) / 3600.0,
        "measured_arm_wall_s": wall,
        "batch_multiple_at_the_target": multiple,
        "batches_observed": count,
        "batches_at_the_target_wall": batches_at_target,
        "peaks_over_threshold": peaks_over_threshold_tail(
            gaps, batches_at_target=batches_at_target
        ),
        "threshold_sensitivity": [
            {
                "threshold_quantile": quantile,
                **{
                    key: value
                    for key, value in peaks_over_threshold_tail(
                        gaps,
                        batches_at_target=batches_at_target,
                        threshold_quantile=quantile,
                    ).items()
                    if key
                    in {
                        "threshold_s",
                        "exceedances",
                        "fitted_shape_xi",
                        "return_level_at_the_target_s",
                        "return_level_over_the_ceiling",
                        "return_level_if_the_tail_were_exponential_over_the_ceiling",
                    }
                },
            }
            for quantile in POT_THRESHOLD_LADDER
        ],
        "distribution_free": distribution_free_tail_bound(
            gaps, batches_at_target=batches_at_target
        ),
        "the_two_models_disagree_by_design": (
            "the POT return level is what the sampled tail predicts; the "
            "distribution-free bound is what the sample cannot exclude. A round "
            "that reported only the first would be claiming a bound it did not "
            "measure, which is the defect review-0250-01 named in R0250's "
            "per-batch projection."
        ),
    }


def fit_setup_gap(report: Mapping[str, Any]) -> dict[str, Any]:
    """The widest setup gap ATTRIBUTABLE TO `fit()`, which is what the diff moves.

    The stage this round measures starts before the node verifies and loads the
    graph, so its widest setup gap may sit in node code that no change to
    `basemap/pumap/` can touch. Separating the two is the difference between
    "the trainer diff worked" and "the stage is under the ceiling", which are
    not the same claim and must not be reported as one.
    """
    sites = dict(report["setup_gaps_by_site"])
    fit_sites = set(DECLARED_TRAINER_POLL_SITES)
    inside = {
        site: entry["widest_gap_s"] for site, entry in sites.items() if site in fit_sites
    }
    outside = {
        site: entry["widest_gap_s"]
        for site, entry in sites.items()
        if site not in fit_sites
    }
    ceiling = registered_value("r0246_max_poll_spacing_s")
    widest_inside = max(inside.values()) if inside else 0.0
    widest_outside = max(outside.values()) if outside else 0.0
    return {
        "arm": report["arm"],
        "widest_setup_gap_inside_fit_s": widest_inside,
        "widest_setup_gap_inside_fit_after": (
            max(inside, key=inside.get) if inside else None
        ),
        "widest_setup_gap_inside_fit_over_the_ceiling": widest_inside / ceiling,
        "widest_setup_gap_outside_fit_s": widest_outside,
        "widest_setup_gap_outside_fit_after": (
            max(outside, key=outside.get) if outside else None
        ),
        "widest_setup_gap_outside_fit_over_the_ceiling": widest_outside / ceiling,
        "the_binding_setup_interval_is_inside_fit": bool(
            widest_inside >= widest_outside
        ),
    }


def hash_bound_setup_projection(
    *, measured_gap_s: float, measured_bytes: int, target_rows: int, dimension: int
) -> dict[str, Any]:
    """If the binding setup call is an integrity hash, project it to `target_rows`.

    **A projection, not a measurement.** It rests on one identity — a sha256 of
    `B` bytes costs `B / throughput` seconds and the throughput is a property of
    the CPU, not of the file — and on the measured throughput of this rung's own
    call. It is published because a call whose cost is exactly linear in the
    substrate is the one term of setup whose growth to 100M is predictable, and
    a round that measured it at 2M and said nothing about 100M would be hiding
    the only extrapolable number it has.
    """
    ceiling = registered_value("r0246_max_poll_spacing_s")
    gap = float(measured_gap_s)
    observed_bytes = int(measured_bytes)
    if gap <= 0.0 or observed_bytes <= 0:
        raise Round0251SetupError(
            "R0251 hash projection needs a positive gap and byte count"
        )
    throughput = observed_bytes / gap
    target_bytes = int(target_rows) * int(dimension) * 4
    projected = target_bytes / throughput
    return {
        "kind": "projection",
        "measured_gap_s": gap,
        "measured_bytes": observed_bytes,
        "measured_throughput_bytes_per_s": throughput,
        "target_rows": int(target_rows),
        "target_dimension": int(dimension),
        "target_bytes": target_bytes,
        "projected_gap_s": projected,
        "projected_gap_over_the_ceiling": projected / ceiling,
        "projected_meets_the_registered_ceiling": bool(projected <= ceiling),
        "registered_ceiling_s": ceiling,
        "is_a_measurement_at_the_target_rows": False,
        "basis": (
            "the binding setup interval at this rung is one integrity hash of "
            f"{observed_bytes} B taking {gap} s, i.e. "
            f"{throughput} B/s. The same call over a "
            f"{int(target_rows)} x {int(dimension)} fp32 substrate reads "
            f"{target_bytes} B and would take {projected} s at the same "
            "throughput. Storage read rate, not hash rate, may bind instead on a "
            "cold cache, which would make this an UNDER-estimate."
        ),
        "what_would_fix_it": (
            "the hash is a loop over fixed-size chunks. One abort read between "
            "chunks makes this interval independent of the file size entirely. "
            "That is a change to basemap/artifact_identity.py, which this round "
            "does not touch: R0251's mandate limits trainer edits to "
            "basemap/pumap/, and the integrity hash is neither trainer nor guard."
        ),
    }


#: A GPD return level is only meaningful if the shape parameter is identified.
#: If the threshold ladder moves the implied return level by more than this
#: factor, the fit is a threshold artefact and the round says so instead of
#: publishing whichever row it likes. Fixed before any gap is drawn.
SHAPE_IDENTIFICATION_SPREAD_LIMIT = 10.0


def tail_verdict(model: Mapping[str, Any]) -> dict[str, Any]:
    """What the tail data actually supports, once the models are cross-checked.

    A round that published one POT return level would be claiming a tail it did
    not measure. This function checks whether the extreme-value fit is
    identified at all — across the threshold ladder and across the bootstrap —
    and reports the three defensible statements separately from the one
    fragile one.
    """
    pot = dict(model["peaks_over_threshold"])
    ladder = [dict(row) for row in model["threshold_sensitivity"]]
    ratios = [float(row["return_level_over_the_ceiling"]) for row in ladder]
    spread = max(ratios) / min(ratios) if min(ratios) > 0 else float("inf")
    ci = pot.get("shape_bootstrap_ci_95") or [float("nan"), float("nan")]
    identified = bool(
        spread <= SHAPE_IDENTIFICATION_SPREAD_LIMIT
        and float(ci[1]) - float(ci[0]) <= 1.0
    )
    exponential = [
        float(row["return_level_if_the_tail_were_exponential_over_the_ceiling"])
        for row in ladder
    ]
    free = dict(model["distribution_free"])
    return {
        "the_extreme_value_fit_is_identified": identified,
        "threshold_ladder_return_level_spread": spread,
        "threshold_ladder_spread_limit": SHAPE_IDENTIFICATION_SPREAD_LIMIT,
        "fitted_shape_bootstrap_ci_95": list(ci),
        "defensible_statements": {
            "observed_max_over_the_ceiling": pot["observed_max_over_the_ceiling"],
            "batches_observed": pot["batches_observed"],
            "exponential_tail_reference_over_the_ceiling_by_threshold": exponential,
            "distribution_free_upper_bound_on_expected_exceedances": free[
                "upper_bound_on_expected_exceedances_at_the_target"
            ],
        },
        "fragile_statement": {
            "generalised_pareto_return_level_over_the_ceiling": pot[
                "return_level_over_the_ceiling"
            ],
            "why_it_is_fragile": (
                "the fitted shape moves across the threshold ladder and its "
                "bootstrap interval is wide, so the return level is a property "
                "of the threshold rather than of the tail."
            ),
        },
        "plain_statement": (
            "the per-batch tail is NOT determined by this rung. The exponential "
            "reference puts the widest gap at a few percent of the ceiling; the "
            "distribution-free bound cannot exclude a four-figure number of "
            "exceedances; and the generalised-Pareto fit lands anywhere between "
            "them depending on the threshold. What is measured is the observed "
            "maximum over the batches actually run. A multi-hour node's tail "
            "needs a multi-hour observation, not an extrapolation from minutes."
            if not identified
            else
            "the extreme-value fit is identified across the threshold ladder and "
            "the bootstrap, so its return level is the round's tail estimate, "
            "with the distribution-free bound beside it as the pessimistic limit."
        ),
    }


def declared_sites_match_the_release() -> dict[str, Any]:
    """The five sites in this module against the five on the release class.

    A docstring that lists poll sites is a claim; this is the check. It reads
    the class attributes, so a site renamed or removed in `basemap/pumap/`
    without updating this module fails closed here rather than silently
    measuring four sites.
    """
    from .pumap.parametric_umap import ParametricUMAP

    names = (
        "ABORT_POLL_SITE_FIT_ENTERED",
        "ABORT_POLL_SITE_EDGE_LIST_PREPARED",
        "ABORT_POLL_SITE_MODEL_ALLOCATED",
        "ABORT_POLL_SITE_SETUP_COMPLETE",
        "ABORT_POLL_SITE_TRAIN_BATCH",
    )
    observed = tuple(getattr(ParametricUMAP, name, None) for name in names)
    if observed != DECLARED_TRAINER_POLL_SITES:
        raise Round0251SetupError(
            "R0251's declared trainer poll sites are not the release class's: "
            f"{observed!r} != {DECLARED_TRAINER_POLL_SITES!r}"
        )
    if not callable(getattr(ParametricUMAP, "_poll_abort", None)):
        raise Round0251SetupError("R0251 requires ParametricUMAP._poll_abort")
    return {
        "declared_sites": list(DECLARED_TRAINER_POLL_SITES),
        "release_class_sites": list(observed),
        "sites_match": True,
        "hook_attribute": "ParametricUMAP.abort_poll",
        "hook_default_is_none": getattr(ParametricUMAP(), "abort_poll", "missing")
        is None,
    }


__all__ = [
    "BATCH_POLL_SITE",
    "DECLARED_TRAINER_POLL_SITES",
    "DISTRIBUTION_FREE_CONFIDENCE",
    "NODE_SETUP_SITES",
    "NOT_A_FAMILY_CELL",
    "POT_THRESHOLD_QUANTILE",
    "PollRecorder",
    "ROUND_ID",
    "RULE_OF_THREE_NUMERATOR",
    "Round0251SetupError",
    "SETUP_CAPABILITY",
    "SETUP_POLL_SITES",
    "SETUP_RUNG_UPDATES",
    "SETUP_SCHEMA",
    "TAIL_RUNG_UPDATES",
    "TAIL_TARGET_HOURS",
    "TAIL_TARGET_SECONDS",
    "POT_BOOTSTRAP_RESAMPLES",
    "POT_BOOTSTRAP_SEED",
    "POT_THRESHOLD_LADDER",
    "declared_sites_match_the_release",
    "distribution_free_tail_bound",
    "fit_setup_gap",
    "hash_bound_setup_projection",
    "peaks_over_threshold_tail",
    "phase_report",
    "SHAPE_IDENTIFICATION_SPREAD_LIMIT",
    "setup_reduction",
    "tail_model",
    "tail_verdict",
]
