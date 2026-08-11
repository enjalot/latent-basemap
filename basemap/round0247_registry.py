"""R0247 — no safety parameter may be overridable from its registered value.

Three consecutive rounds shipped guards and three consecutive reviews defeated
them **in the same shape**:

* R0245 shipped three fixes; review-0245-01 defeated all three.
* R0246 closed those three plus fifteen novel attacks; review-0246-01 defeated
  it again, twice, in a shape R0246 believed it had closed. R0246 clamped the
  worst-case slope (`a2-novel-4`) and the declared headroom (`a2-novel-3`) and
  then left `max_poll_spacing_s` as an unclamped constructor keyword. The
  reviewer pushed R0245's own `24.46713631998864` s blocker gap through it with
  `training_performed=True`, and the receipt published the override as

      registered_max_poll_spacing_s: 1000000.0

**The generalisable defect is not those five parameters. It is the class:**

    A registered safety bound that reaches the decision through a constructor
    keyword, a function keyword, a self-declared interval, or a mutable module
    global is a bound the caller owns; and a receipt that echoes the caller's
    value under a `registered_*` key makes the substitution invisible.

This module is the fix. Every number in R0244-R0246 that participates in a
safety decision is enumerated here **once**, with its direction, its basis and
the exact construction path by which a caller could previously have replaced it.
Guarded code no longer reads its bound from its own arguments or from a module
global; it reads it from here, through `clamp()`, which

1. returns the **registered** value whenever the caller asked for a weaker one
   (so the bound is non-overridable in the weakening direction),
2. returns the caller's value when it is *stricter* (a node is always free to
   hold itself to more), and
3. **records the attempt either way**, so an override is a recorded violation
   that fails the gate rather than a number the receipt reports as registered.

`registered_bounds()` is the only sanctioned source of a `registered_*` receipt
field, so a receipt cannot name a caller's value as registered even by accident.
`registry_fingerprint()` is a SHA-256 over the whole inventory, pinned at
`REGISTERED_REGISTRY_SHA256` and verified by every gate, so mutating the
registry itself - the obvious next door over - fails closed instead of
succeeding quietly.

Nothing here signals anything, starts a child process, touches the GPU, or
imports a GPU array library.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

ROUND_ID = "0247"
ROWS = 100_000_000
GPU_HOURS_CAP = 1.0


class Round0247Error(RuntimeError):
    """R0247 fails closed."""


SAFETY_PARAMETER_CLASS_NOTE = (
    "REGISTERED 2026-08-10 (R0247). A safety parameter is any number, flag or "
    "callable that participates in a decision about whether a stage may run, "
    "may continue, or may publish. No safety parameter may be overridable from "
    "its registered value in the weakening direction. A caller may always be "
    "STRICTER than the registry; a caller that asks to be weaker gets the "
    "registered value AND a recorded violation that fails the gate. A receipt "
    "field named registered_* is produced only by registered_bounds() and "
    "therefore always reports the registry, never the caller. This replaces the "
    "R0244-R0246 pattern in which the bound was a constructor keyword with a "
    "registered DEFAULT and no floor, which review-0245-01 defeated on the "
    "slope and review-0246-01 defeated again on the poll-spacing ceiling."
)

#: `ceiling` — the registry value is the largest permitted; a caller asking for
#: MORE is asking to be weaker. `floor` — the registry value is the smallest
#: permitted; a caller asking for LESS is asking to be weaker.
CEILING = "ceiling"
FLOOR = "floor"


@dataclass(frozen=True)
class SafetyParameter:
    """One registered bound. Frozen: the entry itself cannot be edited."""

    name: str
    module: str
    symbol: str
    value: float
    direction: str
    role: str
    basis: str
    override_path: str
    what_it_does_not_catch: str
    #: `refused` — a bound on the GUARD. Asking for a weaker one is an attempt
    #: to defeat the guard, so the registered value is used AND the attempt
    #: fails the gate. `clamped` — a DECLARATION of the node's own resource
    #: envelope (its budget, its headroom). Clamping it to the registry can
    #: only make the node stricter and leaves no residual weakening, so the
    #: attempt is recorded in the receipt and does not by itself fail the gate.
    #: The distinction is registered rather than ad hoc: a bound belongs to the
    #: guard, an envelope belongs to the node, and only the first is a safety
    #: decision the node is not entitled to make.
    enforcement: str = "refused"

    def weaker_than_registered(self, requested: float) -> bool:
        if self.direction == CEILING:
            return float(requested) > float(self.value)
        return float(requested) < float(self.value)

    def stricter_than_registered(self, requested: float) -> bool:
        if self.direction == CEILING:
            return float(requested) < float(self.value)
        return float(requested) > float(self.value)


def _p(**kwargs: Any) -> SafetyParameter:
    return SafetyParameter(**kwargs)


#: R0244's sealed numbers, restated here so the registry does not import the
#: modules it governs (which would make the import graph circular).
R0244_SEALED_BUDGET_HEADROOM_BYTES = 29_548_888_064
R0244_SEALED_MEASURED_SLOPE_BYTES_PER_S = 11_767_996_416
#: `headroom / slope`. The one inequality the whole guard family rests on.
DERIVED_MAX_OBSERVATION_GAP_S = float(
    R0244_SEALED_BUDGET_HEADROOM_BYTES
) / float(R0244_SEALED_MEASURED_SLOPE_BYTES_PER_S)

_PARAMETERS: tuple[SafetyParameter, ...] = (
    # ---- R0244: the host watchdog ---------------------------------------- #
    _p(
        name="watchdog_sample_interval_s",
        module="basemap.round0244_guard",
        symbol="WATCHDOG_SAMPLE_INTERVAL_S",
        value=0.25,
        direction=CEILING,
        role=(
            "how long the host guard may be blind between two observations of "
            "/proc"
        ),
        basis=(
            "0.25 s at the registered worst-case slope 11,767,996,416 B/s "
            "permits 2,941,999,104 B of growth before a trip is even seen, "
            "which is 0.0996 of the 29,548,888,064 B of sealed headroom"
        ),
        override_path=(
            "ThreadedHostWatchdog(interval_s=...) - review-0246-01 A: a node "
            "declaring interval_s = 5.0 sampled faithfully, reported "
            "thread_sample_coverage 0.9994, passed require_live_sampler, and "
            "bought an observation gap of 5.0 * 11,767,996,416 = 5.884e10 B, "
            "1.99x the sealed headroom"
        ),
        what_it_does_not_catch=(
            "a burst inside one sampling interval. The interval bounds how "
            "late a trip is SEEN, never how much is allocated between two "
            "observations"
        ),
    ),
    _p(
        name="watchdog_max_consecutive_sample_failures",
        module="basemap.round0244_guard",
        symbol="WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES",
        value=3.0,
        direction=CEILING,
        role="how many consecutive /proc read failures are tolerated",
        basis=(
            "n consecutive failures blind the guard for n * 0.25 s, i.e. "
            "n * 2,941,999,104 B at the registered slope. Holding that inside "
            "29,548,888,064 B of sealed headroom admits n <= 10.04; the "
            "registered 3 is 3.35x stricter"
        ),
        override_path=(
            "a mutable module global read inside _sample_once(). Not a keyword "
            "in R0246, but assignable: round0244_guard."
            "WATCHDOG_MAX_CONSECUTIVE_SAMPLE_FAILURES = 10**9 disarmed the arm "
            "from any importer"
        ),
        what_it_does_not_catch=(
            "a sampler failing fewer than four samples in a row forever. That "
            "is the observation-gap arm's job"
        ),
    ),
    _p(
        name="watchdog_max_observation_gap_s",
        module="basemap.round0247_registry",
        symbol="WATCHDOG_MAX_OBSERVATION_GAP_S",
        value=DERIVED_MAX_OBSERVATION_GAP_S,
        direction=CEILING,
        role=(
            "the widest MEASURED interval between two successful thread "
            "samples that a node may publish - the denominator-free "
            "replacement for a coverage ratio"
        ),
        basis=(
            "29,548,888,064 B of R0244 sealed headroom over 11,767,996,416 "
            "B/s of R0244 sealed measured slope. It is the same inequality the "
            "poll-spacing ceiling is derived from, applied to the OBSERVATION "
            "gap instead of the ENFORCEMENT gap, and it is measured in seconds "
            "of wall clock rather than as a ratio to anything the node declares"
        ),
        override_path=(
            "new in R0247. It exists because review-0246-01 A showed that "
            "thread_sample_coverage is a ratio to a self-declared interval and "
            "is therefore not a bound at all"
        ),
        what_it_does_not_catch=(
            "a burst inside one observation gap, and a fabricated receipt. The "
            "arm is computed from timestamps the sampling thread takes itself"
        ),
    ),
    _p(
        name="watchdog_max_mean_observation_gap_s",
        module="basemap.round0247_registry",
        symbol="WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S",
        value=0.5,
        direction=CEILING,
        role=(
            "the mean measured interval between thread samples, in seconds - "
            "what R0246's 0.50 coverage floor was TRYING to bound"
        ),
        basis=(
            "R0246 registered a coverage floor of 0.50 against a registered "
            "interval of 0.25 s, i.e. a mean observation gap of 0.25 / 0.50 = "
            "0.5 s. This is that same number expressed in seconds so that it "
            "cannot be satisfied by moving the denominator. It is 5.02x "
            "stricter than the derived safety bound 2.5109531834854018 s"
        ),
        override_path=(
            "new in R0247; R0246 expressed this bound only as a ratio to "
            "interval_s, which the caller supplied"
        ),
        what_it_does_not_catch=(
            "one long blind interval inside an otherwise dense trace. That is "
            "watchdog_max_observation_gap_s's job; the two arms are kept "
            "separate on purpose"
        ),
    ),
    _p(
        name="min_thread_sample_coverage",
        module="basemap.round0246_guard",
        symbol="MIN_THREAD_SAMPLE_COVERAGE",
        value=0.50,
        direction=FLOOR,
        role=(
            "the fraction of the samples the REGISTERED interval implies that "
            "a node's host guard must actually have taken"
        ),
        basis=(
            "derived safety floor 0.25 * 11,767,996,416 / 29,548,888,064 = "
            "0.09956; measured healthy value on R0245's sealed 114.54487995902309 s "
            "sampler node (439 - 4) / 458.17951983609237 = 0.9494. The "
            "registered 0.50 is 5.02x stricter than the safety floor and half "
            "the healthy value"
        ),
        override_path=(
            "require_live_sampler(min_thread_sample_coverage=...) - a keyword "
            "with a registered default and no floor"
        ),
        what_it_does_not_catch=(
            "anything at all, once the denominator is the caller's. R0247 "
            "recomputes the coverage against the REGISTERED interval and keeps "
            "this arm only as a cross-check on the two second-valued arms"
        ),
    ),
    _p(
        name="coverage_gate_min_expected_samples",
        module="basemap.round0246_guard",
        symbol="COVERAGE_GATE_MIN_EXPECTED_SAMPLES",
        value=20.0,
        direction=CEILING,
        role=(
            "the stage length, in expected samples, above which the coverage "
            "arm applies at all"
        ),
        basis=(
            "at 20 expected samples one missed sample moves coverage by 0.05, "
            "so quantisation cannot carry a healthy 0.95 to the 0.50 floor. "
            "Below 20 the ratio is noise"
        ),
        override_path=(
            "require_live_sampler(min_expected_samples=...). R0247's own "
            "attack a1-r0247-2: passing 1e9 makes coverage_gate_applies false "
            "for every stage this program will ever run, disabling the arm "
            "without touching the floor. review-0246-01 did not find this one"
        ),
        what_it_does_not_catch=(
            "a dead sampler inside a stage shorter than 20 intervals; the "
            "death arms still catch that"
        ),
    ),
    _p(
        name="max_declared_anonymous_budget_bytes",
        module="basemap.round0242_locality",
        symbol="WATCHDOG_ANON_BYTES",
        value=float(60 * (1 << 30)),
        direction=CEILING,
        role=(
            "the largest anonymous budget a node may declare for itself before "
            "the node-declared-budget arm of the host guard stops meaning "
            "anything"
        ),
        basis=(
            "R0242's machine-level anonymous pressure threshold is 60 GiB on a "
            "123 GiB box. A node-declared budget above the level at which the "
            "machine itself is in trouble cannot trip before the machine rule "
            "does, so it is not a budget"
        ),
        override_path=(
            "ThreadedHostWatchdog(anonymous_budget_bytes=...). R0247's own "
            "attack a1-r0247-3: R0246's OWN controls pass 1 << 62, which "
            "disables the budget arm entirely - legitimate in a control that "
            "isolates the death arm, and a blank cheque for a trainer"
        ),
        what_it_does_not_catch=(
            "a node declaring 59 GiB on a stage that needs 4. Declaring a "
            "budget smaller than you need is the node's own business; "
            "declaring one the machine cannot survive is not"
        ),
        enforcement="clamped",
    ),
    # ---- R0245 / R0246: the poll-spacing gate ----------------------------- #
    _p(
        name="r0246_max_poll_spacing_s",
        module="basemap.round0246_guard",
        symbol="R0246_MAX_POLL_SPACING_S",
        value=DERIVED_MAX_OBSERVATION_GAP_S,
        direction=CEILING,
        role=(
            "the widest interval between two cooperative-abort READS that any "
            "stage may publish"
        ),
        basis=(
            "29,548,888,064 B of R0244 sealed headroom over 11,767,996,416 B/s "
            "of R0244 sealed measured slope"
        ),
        override_path=(
            "AbortPollGate(max_poll_spacing_s=...) - review-0246-01 C, the "
            "sixteenth attack. With headroom 1 << 50 and ceiling 1e6 the "
            "reviewer passed R0245's own 24.46713631998864 s blocker gap "
            "through all eight arms with training_performed=True, and the "
            "receipt published registered_max_poll_spacing_s: 1000000.0"
        ),
        what_it_does_not_catch=(
            "an allocation burst inside one unit of work. The gate bounds the "
            "spacing of reads, never what happens between two of them"
        ),
    ),
    _p(
        name="min_enforcement_polls",
        module="basemap.round0246_guard",
        symbol="MIN_ENFORCEMENT_POLLS",
        value=2.0,
        direction=FLOOR,
        role="the smallest number of abort reads from which a spacing exists",
        basis=(
            "two points define one interval. The binding requirement is the "
            "derived ceil(wall / ceiling) + 1, of which this is the floor"
        ),
        override_path=(
            "AbortPollGate(min_polls=...). Passing 0 restores R0245's "
            "zero-poll stage, which held on max_gap 0.0"
        ),
        what_it_does_not_catch=(
            "a stage that meets the count with reads bunched at one end. The "
            "start and end anchors are what catch that"
        ),
    ),
    _p(
        name="min_binding_slope_bytes_per_s",
        module="basemap.round0245_guard",
        symbol="MIN_BINDING_SLOPE_BYTES_PER_S",
        value=float(R0244_SEALED_MEASURED_SLOPE_BYTES_PER_S),
        direction=FLOOR,
        role="the allocation slope a poll-spacing verdict is scored at",
        basis=(
            "the steepest one-second rise in R0244's sealed "
            "anonymous_trace_by_second - the fastest allocation slope this "
            "machine has been measured at. A one-second difference cannot see "
            "a burst inside a single bin, so a measured 0.0 B/s is not "
            "evidence of a slow stage"
        ),
        override_path=(
            "AbortPollTracker(slope_bytes_per_s=...). R0246 caught this one "
            "with a failure ARM (a2-novel-4) but still scored and published "
            "worst_case_requirement at the caller's slope"
        ),
        what_it_does_not_catch=(
            "a stage that really allocates faster than 11,767,996,416 B/s "
            "without its own one-second trace showing it"
        ),
    ),
    _p(
        name="max_declared_headroom_bytes",
        module="basemap.round0245_guard",
        symbol="R0244_BUDGET_HEADROOM_BYTES",
        value=float(R0244_SEALED_BUDGET_HEADROOM_BYTES),
        direction=CEILING,
        role=(
            "the largest headroom a node may claim when it converts a poll "
            "spacing into permitted growth"
        ),
        basis=(
            "R0244's sealed budget_headroom_bytes: the node declared "
            "68,719,476,736 B and the sampling thread measured a peak of "
            "39,170,588,672 B. review-0245-01 B: a headroom is a measurement "
            "from a COMPLETED run, so a node that has not run cannot declare a "
            "larger one"
        ),
        override_path=(
            "AbortPollTracker(headroom_bytes=...). R0246 closed this at the "
            "ceiling arm (a2-novel-3) but left the headroom itself unclamped, "
            "so the reviewer combined 1 << 50 of headroom with an overridden "
            "ceiling and both arms fell together"
        ),
        what_it_does_not_catch=(
            "a node declaring a headroom SMALLER than the registered one, "
            "which is stricter and is permitted and recorded"
        ),
        enforcement="clamped",
    ),
    _p(
        name="slope_control_min_bytes_per_s",
        module="basemap.round0245_guard",
        symbol="SLOPE_CONTROL_MIN_BYTES_PER_S",
        value=float(R0244_SEALED_MEASURED_SLOPE_BYTES_PER_S),
        direction=FLOOR,
        role=(
            "the allocation rate a synthetic slope CONTROL must beat to be a "
            "control for this machine at all"
        ),
        basis=(
            "a control against a defect gentler than the one already measured "
            "proves nothing. The bar is the measured slope itself"
        ),
        override_path=(
            "run_allocation_slope_positive_control(min_slope_bytes_per_s=...)."
            " Lowering it lets a slow synthetic stage certify the guard"
        ),
        what_it_does_not_catch=(
            "a control that meets the rate and tests the wrong arm. Only the "
            "breaching arm's overshoot shows the requirement is non-vacuous"
        ),
    ),
    # ---- R0244: the sampler ---------------------------------------------- #
    _p(
        name="sampler_poll_chunk_draws",
        module="basemap.round0244_prereq",
        symbol="SAMPLER_POLL_CHUNK_DRAWS",
        value=2_000_000.0,
        direction=CEILING,
        role=(
            "how many draws two_level_weight_sample processes between two "
            "cooperative-abort reads"
        ),
        basis=(
            "the measured widest abort-read gap at this chunk size is "
            "0.9261989569931757 s, which is 0.369 of the registered "
            "2.5109531834854018 s ceiling. Doubling it would spend the margin"
        ),
        override_path=(
            "two_level_weight_sample(poll_chunk_draws=...). A caller passing "
            "40,000,000 restores R0245's single-chunk behaviour and with it "
            "the 24.46713631998864 s gap this whole family of rounds exists to "
            "close - and the poll-spacing gate would then simply MEASURE the "
            "wide gap, which is the correct outcome but only after the fact"
        ),
        what_it_does_not_catch=(
            "the allocation inside one chunk. Two million draws of int64 "
            "indices is 16 MB, far inside any headroom"
        ),
    ),
    _p(
        name="sampler_max_anonymous_bytes",
        module="basemap.round0244_prereq",
        symbol="SAMPLER_MAX_ANONYMOUS_BYTES",
        value=float(4 * (1 << 30)),
        direction=CEILING,
        role="the anonymous peak the sampler node is permitted to reach",
        basis=(
            "R0244 registered 4 GiB for a stage whose architectural accounting "
            "is under 1 GiB at 40,000,000 draws"
        ),
        override_path=(
            "a module global compared in the node's verdict_arms. Assignable "
            "by any importer"
        ),
        what_it_does_not_catch=(
            "a peak between two thread samples; the observation-gap arms bound "
            "how long that can hide"
        ),
    ),
    # ---- R0246: the tie-aware estimator ----------------------------------- #
    _p(
        name="tie_tolerance",
        module="basemap.round0227_low_c_contract",
        symbol="TIE_TOLERANCE",
        value=1e-6,
        direction=CEILING,
        role=(
            "the slack in the tie-aware value test cos >= kth - tolerance, and "
            "therefore the width of the band inside which a verdict is noise"
        ),
        basis=(
            "registered by R0227 and consumed by every published R0241 and "
            "R0243 tie-aware figure. R0247 does not move it"
        ),
        override_path=(
            "tie_aware_precision_profile(tolerance=...). R0247's own attack "
            "tie-r0247-1: raising the tolerance widens the band, so FEWER "
            "verdicts differ between float32 and float64, so the MEASURED flip "
            "rate falls and more published claims 'survive'. The measurement "
            "that adjudicates the claims is scored at a threshold the caller "
            "supplies. review-0246-01 did not find this one"
        ),
        what_it_does_not_catch=(
            "the fact that a float32 value test at 1e-6 sits within 1.86x of "
            "its own noise floor. That is what the aggregate-only rule is for"
        ),
    ),
    _p(
        name="tie_claim_max_expected_flips_over_margin",
        module="basemap.round0246_tie",
        symbol="TIE_CLAIM_MAX_EXPECTED_FLIPS_OVER_MARGIN",
        value=1.0,
        direction=CEILING,
        role=(
            "the RETROSPECTIVE criterion: a published claim survives when its "
            "expected flipped decisions are at most this multiple of the "
            "margin that would change it"
        ),
        basis=(
            "the margin is already defined as the number of verdicts that "
            "would have to change for the claim to read differently AT THE "
            "PRECISION IT IS PUBLISHED TO, so a ratio of 1.0 asks exactly 'is "
            "this published number expected to be wrong?' and nothing else"
        ),
        override_path=(
            "adjudicate_tie_aware_claims(margin_fraction=...). Passing 1e6 "
            "makes every claim survive"
        ),
        what_it_does_not_catch=(
            "a claim whose margin is itself optimistic. review-0246-01 E is "
            "right that the ledger's margins are author-chosen"
        ),
    ),
    _p(
        name="tie_use_max_expected_flips_over_margin",
        module="basemap.round0246_tie",
        symbol="TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN",
        value=0.01,
        direction=CEILING,
        role=(
            "the PROSPECTIVE criterion: a NEW consumption of the tie-aware "
            "vector is admitted only this far below its own margin"
        ),
        basis=(
            "TIE_AGGREGATE_ONLY_RULE, sealed verbatim into every R0246 "
            "receipt, says a claim may consume the vector where the expected "
            "flips are 'below 1% of the margin'. review-0246-01 H5 found the "
            "code applying 1.0. R0247 registers the sealed 1% as the criterion "
            "for ADMITTING a new use - the question 'will this be right, with "
            "margin?' - and keeps 1.0 for ADJUDICATING an existing published "
            "claim - the question 'is this wrong?'. The two functions were "
            "always asking different questions; only one rule text was written"
        ),
        override_path=(
            "require_aggregate_tie_aware_use(margin_fraction=...)"
        ),
        what_it_does_not_catch=(
            "a use whose declared decision count is understated. The ledger's "
            "PROBE_CANDIDATE_DECISIONS exists because a flip anywhere in the "
            "probe can move a small selected set"
        ),
    ),
    _p(
        name="tie_precision_min_rows",
        module="basemap.round0246_tie",
        symbol="TIE_PRECISION_ROWS",
        value=20_000.0,
        direction=FLOOR,
        role=(
            "the smallest probe sample from which a flip rate may be "
            "published, because the rule-of-three bound is a property of the "
            "sample size and nothing else"
        ),
        basis=(
            "R0246 measured 0 flips in 300,000 decisions and retracted eight "
            "published precisions on a bound of 3/300000 = 1e-05 that the "
            "sample size alone produced. 20,000 rows is R0246's own sample and "
            "is registered as the FLOOR, not the target: R0247 scores the "
            "whole 500,000-row probe"
        ),
        override_path=(
            "tie_aware_precision_profile(sample_rows=...). Note the direction: "
            "a SMALLER sample gives a WIDER bound and retracts MORE claims, so "
            "under-sampling is self-punishing rather than self-serving. It is "
            "registered as a floor for reproducibility, not because it is the "
            "attack surface"
        ),
        what_it_does_not_catch=(
            "the fact that 7,500,000 is the entire decision population of this "
            "probe. Tightening the bound further needs a larger truth probe, "
            "not a larger sample"
        ),
    ),
    # ---- R0248: the declarations that waive gate arms -------------------- #
    _p(
        name="replay",
        module="basemap.round0245_guard",
        symbol="AbortPollTracker.replay",
        value=0.0,
        direction=CEILING,
        role=(
            "whether a poll-spacing gate's verdict is a REPLAY - a "
            "demonstration scored on a scripted clock through a callable that "
            "need not read the cooperative abort flag - rather than a "
            "measurement of a stage that actually ran. 0.0 is False"
        ),
        basis=(
            "review-0247-01 A.6: replay=True waives BOTH "
            "the_clock_is_the_registered_monotonic_clock and "
            "the_gate_wraps_a_registered_abort_reader inside require() itself. "
            "R0247 retired training_performed on the stated principle that a "
            "self-declared bool which switches off an arm is a safety "
            "parameter, and then shipped a self-declared bool that switches "
            "off two. The registered value is False: no gate is a replay "
            "unless it says so, and saying so is published"
        ),
        override_path=(
            "AbortPollTracker(replay=True) / AbortPollGate(replay=True). It is "
            "not clamped to False, because the reviewer-shaped replays are "
            "legitimate demonstrations and forcing them to False would make "
            "every replay control unrunnable. It is REGISTERED, so the "
            "declaration is recorded in safety_overrides, published as "
            "declared_replay beside registered_replay, listed in "
            "gate_arms_waived_by_declaration, and refused at the sealing "
            "boundary by require_enforcement_evidence"
        ),
        what_it_does_not_catch=(
            "a node that declares replay=True and never calls "
            "require_enforcement_evidence. Nothing in-process closes that; "
            "what closes it is that the node module is the diff a reviewer "
            "reads, and that the arms it waived are now named in the receipt"
        ),
        enforcement="clamped",
    ),
    _p(
        name="tie_bound_confidence",
        module="basemap.round0247_precision",
        symbol="TIE_BOUND_CONFIDENCE",
        value=0.95,
        direction=FLOOR,
        role=(
            "the confidence level of the upper bound on the flip rate that the "
            "claim adjudication is run at"
        ),
        basis=(
            "the rule of three, 3/n, is the 95% one-sided Poisson upper limit "
            "at zero observed events. R0247 registers the confidence rather "
            "than the formula so that a non-zero observation is bounded by the "
            "same rule instead of by a point estimate"
        ),
        override_path=(
            "any caller of the bound helper. A lower confidence gives a "
            "tighter bound and more surviving claims"
        ),
        what_it_does_not_catch=(
            "systematic error. A Poisson bound assumes the flips are "
            "independent draws, which candidates inside one probe row are not"
        ),
    ),
)

REGISTERED_SAFETY_PARAMETERS: Mapping[str, SafetyParameter] = MappingProxyType(
    {parameter.name: parameter for parameter in _PARAMETERS}
)

#: Convenience aliases for the two bounds R0247 introduces, so guarded code can
#: read them as constants without going through the mapping.
WATCHDOG_MAX_OBSERVATION_GAP_S = REGISTERED_SAFETY_PARAMETERS[
    "watchdog_max_observation_gap_s"
].value
WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S = REGISTERED_SAFETY_PARAMETERS[
    "watchdog_max_mean_observation_gap_s"
].value
TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN = REGISTERED_SAFETY_PARAMETERS[
    "tie_use_max_expected_flips_over_margin"
].value
TIE_BOUND_CONFIDENCE = REGISTERED_SAFETY_PARAMETERS[
    "tie_bound_confidence"
].value


# --------------------------------------------------------------------------- #
# construction paths that are not numbers
# --------------------------------------------------------------------------- #
#: A poll-spacing gate measures the spacing of calls to ITSELF. If the callable
#: it wraps does not read the cooperative abort flag, the gate publishes a
#: perfect spacing for a stage that never checks anything. R0246's own controls
#: pass `inner=lambda _where: None`; so could a trainer.
REGISTERED_ABORT_READERS: frozenset[str] = frozenset({
    "experiments.round0238_nodes._check_runner_abort",
    "basemap.gpu_child_supervision.runner_abort_reason",
    #: R0248 gap 3: the two control readers that were previously sanctioned by
    #: the runtime marker are now sanctioned BY NAME, in source, in this set.
    "basemap.round0247_guard._sanctioned_reader",
    "basemap.round0248_guard._sanctioned_reader",
})

REGISTERED_ABORT_READER_ATTRIBUTE = "_r0247_registered_abort_reader"

#: R0248 gap 3, review-0247-01 A.4. `registered_abort_reader` was an exported
#: one-call decorator that set the attribute below on ANY callable, and
#: `is_registered_abort_reader` honoured it — so one function call made a no-op
#: pass both `require()` and `require_enforcement_evidence()`. An allowlist a
#: caller can extend at runtime is not an allowlist. The attribute no longer
#: sanctions anything: sanction is by qualified name against
#: `REGISTERED_ABORT_READERS`, which is source. Applying the retired marker is
#: still possible (the function is still importable, and pretending otherwise
#: would only push the attack to `setattr`), but it is now a RECORDED
#: VIOLATION: every application is appended here and published, and a reader
#: carrying the marker without a registered name is reported as
#: `unsanctioned`.
MARKER_APPLICATIONS: list[dict[str, Any]] = []

RETIRED_MARKER_NOTE = (
    "RETIRED 2026-08-11 (R0248, review-0247-01 A.4). The "
    "_r0247_registered_abort_reader attribute no longer sanctions a "
    "cooperative-abort reader. Sanction is by qualified name against the "
    "source-level REGISTERED_ABORT_READERS set. Every application of the "
    "retired marker is recorded in MARKER_APPLICATIONS and published in the "
    "gate's verdict under abort_reader_sanction, so a reader that was marked "
    "at runtime is visible as marked-and-unsanctioned rather than passing."
)

CONSTRUCTION_PATH_NOTE = (
    "Three of the escape hatches R0247 closes are not numbers. (1) `clock` - an "
    "AbortPollGate built with a scripted clock measures whatever the script "
    "says, so a stage could report a 1 ms widest gap by supplying its own "
    "clock; a scripted clock is now permitted only on a gate explicitly "
    "declared replay=True, whose verdict is marked replay_only and which a node "
    "may not seal as enforcement evidence. (2) `inner` - the gate times calls "
    "to itself, so a gate wrapping a no-op publishes a perfect spacing for a "
    "stage that never reads the flag; `inner` must now be a registered abort "
    "reader. (3) `training_performed` - a self-declared bool that switched off "
    "the worst-case-slope arm; the arm is now unconditional, so the "
    "declaration no longer participates in any safety decision."
)


def abort_reader_qualified_name(inner: Any) -> str:
    module = getattr(inner, "__module__", "") or ""
    qualname = (
        getattr(inner, "__qualname__", "")
        or getattr(inner, "__name__", "")
    )
    return f"{module}.{qualname}"


def abort_reader_sanction(inner: Any) -> dict[str, Any]:
    """WHICH mechanism sanctioned this reader — published in every verdict.

    review-0247-01 H5: "an allowlist that a caller can extend at runtime is not
    an allowlist. If the marker must stay, publish *which* mechanism sanctioned
    the reader (name match vs. marker) in the verdict, so a reviewer can see
    it." R0248 does both: the marker no longer sanctions, and the mechanism is
    published either way.
    """
    if inner is None:
        return {
            "qualified_name": None,
            "sanctioned": False,
            "mechanism": "no_reader",
            "carries_the_retired_marker": False,
            "note": RETIRED_MARKER_NOTE,
        }
    name = abort_reader_qualified_name(inner)
    marked = bool(getattr(inner, REGISTERED_ABORT_READER_ATTRIBUTE, False))
    sanctioned = name in REGISTERED_ABORT_READERS
    return {
        "qualified_name": name,
        "sanctioned": sanctioned,
        "mechanism": "registered_name" if sanctioned else (
            "retired_marker_only" if marked else "unsanctioned"
        ),
        "carries_the_retired_marker": marked,
        "marked_and_unsanctioned": bool(marked and not sanctioned),
        "note": RETIRED_MARKER_NOTE,
    }


def is_registered_abort_reader(inner: Any) -> bool:
    """Is this callable one of the sanctioned cooperative-abort readers?

    R0248 gap 3: BY NAME ONLY. The runtime marker is ignored.
    """
    return bool(abort_reader_sanction(inner)["sanctioned"])


def registered_abort_reader(function: Any) -> Any:
    """RETIRED. Marking a callable no longer sanctions it; it records it.

    Kept importable so that (a) the retired-marker attack has a call site to be
    controlled through and (b) an old caller fails visibly rather than
    silently. The returned callable is unchanged and is NOT sanctioned unless
    its qualified name is in `REGISTERED_ABORT_READERS`.
    """
    MARKER_APPLICATIONS.append({
        "qualified_name": abort_reader_qualified_name(function),
        "sanctioned_by_name": (
            abort_reader_qualified_name(function) in REGISTERED_ABORT_READERS
        ),
        "note": RETIRED_MARKER_NOTE,
    })
    setattr(function, REGISTERED_ABORT_READER_ATTRIBUTE, True)
    return function


def marker_applications() -> tuple[dict[str, Any], ...]:
    """Every application of the retired marker in this process."""
    return tuple(dict(row) for row in MARKER_APPLICATIONS)


def unsanctioned_marker_applications() -> tuple[dict[str, Any], ...]:
    """Marker applications that did NOT correspond to a registered name."""
    return tuple(
        dict(row) for row in MARKER_APPLICATIONS
        if not row.get("sanctioned_by_name")
    )


# --------------------------------------------------------------------------- #
# the fingerprint — mutating the registry is the next door over
# --------------------------------------------------------------------------- #
def registry_rows() -> tuple[dict[str, Any], ...]:
    """The rows the fingerprint covers.

    R0248, review-0247-01 H1. This used to iterate `_PARAMETERS`, the
    module-level TUPLE, while `clamp()`, `registered_bounds()` and every guard
    resolved `REGISTERED_SAFETY_PARAMETERS`, a DIFFERENT module-level name.
    Rebinding the mapping moved every bound at once and `verify_registry()`
    still returned `holds: true` under the pinned digest, because the
    fingerprint was looking at the other object. It now hashes **the object the
    decisions read**, sorted by name so the digest does not depend on
    insertion order.

    This is a correctness fix to the fingerprint's own claim; it is **not** a
    defence. `gc.get_referents()` still hands out the dict behind the
    `MappingProxyType`, and mutating it moves the bound with the digest intact.
    No in-process Python guard can constrain the process it runs in. The
    external bound R0248 adds is `basemap.round0248_external`.
    """
    return tuple(
        {
            "name": parameter.name,
            "module": parameter.module,
            "symbol": parameter.symbol,
            "value": repr(float(parameter.value)),
            "direction": parameter.direction,
            "enforcement": parameter.enforcement,
        }
        for parameter in sorted(
            REGISTERED_SAFETY_PARAMETERS.values(), key=lambda row: row.name
        )
    )


def registry_fingerprint() -> str:
    """SHA-256 over every registered name, symbol, value and direction."""
    payload = json.dumps(
        {
            "rows": list(registry_rows()),
            "abort_readers": sorted(REGISTERED_ABORT_READERS),
            "retired_marker": RETIRED_MARKER_NOTE,
        },
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


#: Pinned. `verify_registry()` fails closed when the inventory moves, so
#: `REGISTERED_SAFETY_PARAMETERS["r0246_max_poll_spacing_s"] = ...` - the
#: obvious next attack after the keywords are clamped - cannot succeed quietly.
#: Changing a registered bound therefore requires changing this digest in the
#: same commit, which is exactly the "register the change and its basis" rule.
#: Re-pinned 2026-08-11 by R0248. R0247's digest was
#: `35128f85e10ccf74387391b403bd1f18f33732863fc893b7b087db37c826d23f` over
#: `_PARAMETERS`; this one is over `REGISTERED_SAFETY_PARAMETERS` — the object
#: the decisions read — with `replay` registered as the twentieth parameter and
#: the retired-marker note folded in. Changing a registered bound still
#: requires changing this digest in the same commit.
REGISTERED_REGISTRY_SHA256 = (
    "bb8f7b395f06d19db062e1698ad8a332858028dab56eff005baf535771fc253b"
)


def verify_registry(*, label: str = "R0247") -> dict[str, Any]:
    """Fail closed if the registry itself has been edited at runtime."""
    observed = registry_fingerprint()
    if observed != REGISTERED_REGISTRY_SHA256:
        raise Round0247Error(
            f"R0247 STOP: {label} ran against a safety-parameter registry "
            f"whose fingerprint is {observed}, not the registered "
            f"{REGISTERED_REGISTRY_SHA256}. A bound was changed without "
            "registering the change. " + SAFETY_PARAMETER_CLASS_NOTE
        )
    return {
        "registry_fingerprint": observed,
        "registered_registry_sha256": REGISTERED_REGISTRY_SHA256,
        "parameters": len(REGISTERED_SAFETY_PARAMETERS),
        "the_fingerprint_covers_the_object_the_decisions_read": True,
        "what_it_does_not_catch": (
            "gc.get_referents() on the MappingProxyType, and any other "
            "deliberate in-process mutation. See basemap.round0248_external "
            "for the bound that does not live in this process."
        ),
        "holds": True,
    }


# --------------------------------------------------------------------------- #
# the clamp
# --------------------------------------------------------------------------- #
def clamp(
    name: str, requested: Any, *, site: str, label: str = "",
    population: float | None = None,
) -> tuple[float, dict[str, Any] | None]:
    """Return the effective bound and, when the caller asked, the record.

    * `requested is None` — the caller took the registered value; no record.
    * the caller asked to be **weaker** — the REGISTERED value is returned and a
      `weakening` record is produced. The record is what fails the gate; the
      returned value is what makes the gate safe even if nobody reads the
      record.
    * the caller asked to be **stricter** — the caller's value is returned and a
      `stricter` record is produced, because a receipt should say when a node
      held itself to more than the registry.
    """
    parameter = REGISTERED_SAFETY_PARAMETERS.get(name)
    if parameter is None:
        raise Round0247Error(
            f"R0247 STOP: {site} asked to clamp an unregistered safety "
            f"parameter {name!r}. Every bound that participates in a safety "
            "decision must be in the registry. " + SAFETY_PARAMETER_CLASS_NOTE
        )
    registered = float(parameter.value)
    #: A FLOOR on a sample size cannot exceed the population that exists. A
    #: 400-row probe cannot be sampled 20,000 times, so the effective floor
    #: there is "the whole probe", which is strictly stricter than the
    #: registered number and is not an override. The registry value is still
    #: what the receipt reports as registered.
    population_capped = False
    if (
        population is not None
        and parameter.direction == FLOOR
        and float(population) < registered
    ):
        registered = float(population)
        population_capped = True
    if requested is None:
        return registered, None
    asked = float(requested)
    if asked == registered:
        return registered, None
    weakening = parameter.weaker_than_registered(asked)
    effective = registered if weakening else asked
    return effective, {
        "parameter": parameter.name,
        "module": parameter.module,
        "symbol": parameter.symbol,
        "direction": parameter.direction,
        "registered_value": float(parameter.value),
        "effective_floor_after_population_cap": (
            registered if population_capped else None
        ),
        "requested_value": asked,
        "effective_value": effective,
        "kind": "weakening" if weakening else "stricter",
        "enforcement": parameter.enforcement,
        "site": str(site),
        "label": str(label),
        "basis": parameter.basis,
        "note": (
            "the registered value was used and the attempt is recorded"
            if weakening
            else "the caller asked to be stricter than the registry"
        ),
    }


def registered_value(name: str) -> float:
    """Read a registered bound **at the comparison site**.

    R0248 gaps 1 and 2, review-0247-01 A.3. Three bounds were *registered* and
    then *enforced* against a module-level name: `WATCHDOG_MAX_OBSERVATION_GAP_S`
    and `WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S` at `round0246_guard:278,289`, and
    `SAMPLER_MAX_ANONYMOUS_BYTES` at `round0244_prereq:112`. Registering a
    number changes nothing if the `if` statement reads a mirror of it: two
    assignments passed review-0246-01 A's `5.0` s attack while the receipt still
    published `2.5109531834854018`.

    So every gate comparison now calls this, which resolves the name in the
    registry at the moment of the comparison. A module-global assignment is no
    longer a decision surface. `basemap.round0248_inventory` derives the set of
    module-level constants any gate compares against **mechanically** and fails
    if one of them is a registered symbol read as a bare name.
    """
    verify_registry(label=f"R0248 comparison site for {name}")
    parameter = REGISTERED_SAFETY_PARAMETERS.get(name)
    if parameter is None:
        raise Round0247Error(
            f"R0247 STOP: no registered safety parameter {name!r} to read at a "
            "comparison site. " + SAFETY_PARAMETER_CLASS_NOTE
        )
    return float(parameter.value)


def record_declaration(
    name: str, declared: Any, *, site: str, label: str = ""
) -> tuple[float, dict[str, Any] | None]:
    """Record a self-declared flag against its registered value.

    Unlike `clamp()`, this does **not** substitute the registered value. A
    replay gate really is a replay gate, and a receipt that reported otherwise
    would be the `registered_*`-echoes-the-caller defect with the sign flipped.
    What it does is make the declaration *visible*: the record lands in
    `safety_overrides`, the verdict names the arms the declaration waived, and
    the sealing gate refuses it.
    """
    parameter = REGISTERED_SAFETY_PARAMETERS.get(name)
    if parameter is None:
        raise Round0247Error(
            f"R0247 STOP: {site} declared an unregistered safety parameter "
            f"{name!r}. " + SAFETY_PARAMETER_CLASS_NOTE
        )
    asked = float(declared)
    registered = float(parameter.value)
    if asked == registered:
        return asked, None
    weakening = parameter.weaker_than_registered(asked)
    return asked, {
        "parameter": parameter.name,
        "module": parameter.module,
        "symbol": parameter.symbol,
        "direction": parameter.direction,
        "registered_value": registered,
        "effective_floor_after_population_cap": None,
        "requested_value": asked,
        #: the declaration STANDS. That is the difference from clamp().
        "effective_value": asked,
        "kind": "weakening" if weakening else "stricter",
        "enforcement": parameter.enforcement,
        "site": str(site),
        "label": str(label),
        "basis": parameter.basis,
        "note": (
            "a self-declared flag that waives a gate arm. It is not clamped - "
            "the declaration stands and is published - and it is refused at "
            "the sealing boundary by require_enforcement_evidence()"
            if weakening
            else "the caller declared something stricter than the registry"
        ),
    }


def clamp_int(
    name: str, requested: Any, *, site: str, label: str = "",
    population: float | None = None,
) -> tuple[int, dict[str, Any] | None]:
    value, record = clamp(
        name, requested, site=site, label=label, population=population
    )
    return int(round(value)), record


def weakening_overrides(
    records: Iterable[Mapping[str, Any] | None],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(record) for record in records
        if record is not None
        and record.get("kind") == "weakening"
        and record.get("enforcement", "refused") == "refused"
    )


def override_records(
    records: Iterable[Mapping[str, Any] | None],
) -> tuple[dict[str, Any], ...]:
    return tuple(dict(record) for record in records if record is not None)


def require_no_weakening_overrides(
    records: Iterable[Mapping[str, Any] | None], *, label: str
) -> tuple[dict[str, Any], ...]:
    """An override is a recorded violation that FAILS the gate."""
    attempts = override_records(records)
    weakening = weakening_overrides(attempts)
    if weakening:
        names = [str(record["parameter"]) for record in weakening]
        raise Round0247Error(
            f"R0247 STOP: {label} attempted to weaken registered safety "
            f"bound(s) {names}. Each attempt was refused and the registered "
            "value used; the attempt itself is the violation. "
            f"{[dict(record) for record in weakening]} "
            + SAFETY_PARAMETER_CLASS_NOTE
        )
    return attempts


def registered_bounds(names: Sequence[str]) -> dict[str, Any]:
    """The ONLY sanctioned source of a `registered_*` receipt field.

    review-0246-01 C: "A field named `registered_` that reports whatever the
    node passed is worse than no field." Every value here comes from the
    registry and none of them can come from a caller.
    """
    block: dict[str, Any] = {}
    for name in names:
        parameter = REGISTERED_SAFETY_PARAMETERS.get(name)
        if parameter is None:
            raise Round0247Error(
                f"R0247 STOP: no registered safety parameter {name!r}"
            )
        block[f"registered_{parameter.name}"] = float(parameter.value)
        block[f"registered_{parameter.name}_direction"] = parameter.direction
        block[f"registered_{parameter.name}_enforcement"] = parameter.enforcement
        block[f"registered_{parameter.name}_basis"] = parameter.basis
    block["registry_fingerprint"] = registry_fingerprint()
    block["registered_registry_sha256"] = REGISTERED_REGISTRY_SHA256
    return block


def safety_parameter_inventory() -> dict[str, Any]:
    """The published inventory: every safety parameter across R0244-R0247."""
    return {
        "instrument": "round0247-safety-parameter-registry-v1",
        "class_note": SAFETY_PARAMETER_CLASS_NOTE,
        "construction_path_note": CONSTRUCTION_PATH_NOTE,
        "parameters": [
            {
                "name": parameter.name,
                "module": parameter.module,
                "symbol": parameter.symbol,
                "registered_value": float(parameter.value),
                "direction": parameter.direction,
                "enforcement": parameter.enforcement,
                "role": parameter.role,
                "basis": parameter.basis,
                "override_path_before_r0247": parameter.override_path,
                "what_it_does_not_catch": parameter.what_it_does_not_catch,
            }
            for parameter in _PARAMETERS
        ],
        "parameter_count": len(_PARAMETERS),
        "registered_abort_readers": sorted(REGISTERED_ABORT_READERS),
        "retired_marker_note": RETIRED_MARKER_NOTE,
        "retired_marker_applications": list(marker_applications()),
        "unsanctioned_marker_applications": list(
            unsanctioned_marker_applications()
        ),
        "examined_and_not_safety_parameters": EXAMINED_NOT_SAFETY,
        "registry_fingerprint": registry_fingerprint(),
        "registered_registry_sha256": REGISTERED_REGISTRY_SHA256,
    }


#: Enumerated, examined, and deliberately NOT registered - with the reason.
#: A registry that quietly omits a number is the same defect one level up, so
#: the things that were looked at and left out are published too.
EXAMINED_NOT_SAFETY: tuple[dict[str, str], ...] = (
    {
        "symbol": "basemap.round0244_guard.WATCHDOG_TRACE_SECONDS",
        "value": "4000",
        "why_not": (
            "it bounds how long the anonymous trace is kept, and the trace "
            "feeds measured_slope_from_trace. A shorter trace can only make "
            "the OWN-slope branch report less, and the binding slope is "
            "floored at the registered worst case, so it cannot weaken a "
            "verdict"
        ),
    },
    {
        "symbol": "experiments.round0242_nodes.WATCHDOG_SWAP_GROWTH_BYTES",
        "value": "4 GiB",
        "why_not": (
            "R0242's conjunctive machine rule and its three thresholds are "
            "inherited unchanged through two rounds and are not reachable "
            "through any keyword. They are registered in R0242 and R0247 does "
            "not restate them; the ONE of the three a node can reach - the "
            "anonymous budget it declares for itself - is registered above"
        ),
    },
    {
        "symbol": "basemap.round0246_tie.TIE_CONTROL_PLANTED_FLIP_RATE",
        "value": "1e-06",
        "why_not": (
            "it is a PLANT for the routing control, deliberately not the "
            "measurement. A control that only fires at the rate the round "
            "happened to observe is not a control, so this number must be "
            "free to move"
        ),
    },
    {
        "symbol": "basemap.round0246_tie.PROBE_CANDIDATE_DECISIONS",
        "value": "7,500,000",
        "why_not": (
            "it is the size of the probe's decision population, a fact about "
            "R0238's truth build (500,000 rows x k=15), not a bound anybody "
            "chose. It is asserted against the truth array's own shape in the "
            "contract tests instead"
        ),
    },
    {
        "symbol": "basemap.round0244_guard.WATCHDOG_DEFAULT_ANON_BUDGET_BYTES",
        "value": "64 GiB",
        "why_not": (
            "it is the DEFAULT a node inherits when it declares nothing, and "
            "it is now clamped on the way in by "
            "max_declared_anonymous_budget_bytes like any other declaration"
        ),
    },
)


__all__ = [
    "CEILING",
    "CONSTRUCTION_PATH_NOTE",
    "MARKER_APPLICATIONS",
    "RETIRED_MARKER_NOTE",
    "abort_reader_qualified_name",
    "abort_reader_sanction",
    "marker_applications",
    "record_declaration",
    "registered_value",
    "unsanctioned_marker_applications",
    "DERIVED_MAX_OBSERVATION_GAP_S",
    "EXAMINED_NOT_SAFETY",
    "FLOOR",
    "GPU_HOURS_CAP",
    "REGISTERED_ABORT_READERS",
    "REGISTERED_ABORT_READER_ATTRIBUTE",
    "REGISTERED_REGISTRY_SHA256",
    "REGISTERED_SAFETY_PARAMETERS",
    "ROUND_ID",
    "ROWS",
    "Round0247Error",
    "SAFETY_PARAMETER_CLASS_NOTE",
    "SafetyParameter",
    "TIE_BOUND_CONFIDENCE",
    "TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN",
    "WATCHDOG_MAX_MEAN_OBSERVATION_GAP_S",
    "WATCHDOG_MAX_OBSERVATION_GAP_S",
    "clamp",
    "clamp_int",
    "is_registered_abort_reader",
    "override_records",
    "registered_abort_reader",
    "registered_bounds",
    "registry_fingerprint",
    "registry_rows",
    "require_no_weakening_overrides",
    "safety_parameter_inventory",
    "verify_registry",
    "weakening_overrides",
]
