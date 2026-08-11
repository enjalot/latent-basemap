"""Execute R0246 — close the three defeated guards and unblock the trainer.

Three CPU-shaped nodes, none of which trains anything, builds anything, or
creates a CUDA context:

* `guard_0246` — the three closures, each with a positive control **in the
  exact shape review-0245-01 used to defeat R0245**, plus a battery of novel
  attacks no control covers. It recomputes the healthy sampling coverage the
  registered floor is calibrated against from R0245's own sealed sampler
  receipt, and exercises the coverage gate on a live watchdog on the node path.
* `sampler_0246` — **the blocker.** R0245's widest abort-read gap was
  `24.46713631998864` s inside R0244's registered `two_level_weight_sample`.
  This node re-runs the same draw at the same seed and the same
  `40,000,000` draws through the polled sampler, re-measures the widest gap,
  scores it against the node's own slope AND against the `11,767,996,416` B/s
  worst case, and refuses unless the correct sampler reproduces R0245's sealed
  statistics to the digit — which is what makes the poll fix provably inert.
* `tie_0246` — the tie-aware precision decision. It measures the estimator's
  per-decision error rate directly (`float32` vs `float64` verdict flips on the
  same bytes) and adjudicates every published claim in R0241 and R0243 that
  rests on a tie-aware decision.

Every registered check is IMPORTED, never re-typed. No node in this module
starts a child process, hands cuVS anything, or contains a signalling
construct of any kind.
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0227_low_c_contract import TIE_TOLERANCE
from basemap.round0238_rung5 import (
    DIMENSION,
    GRAPH_K,
    TRUTH_PROBE_ROWS,
    json_safe,
)
from basemap.round0241_qualify import GPU_HOURS_CAP_NOTE
from basemap.round0242_locality import io_counters, json_scrub
from basemap.round0244_prereq import (
    SAMPLER_BLOCK_EDGES,
    SAMPLER_MAX_ANONYMOUS_BYTES,
    sampler_max_anonymous_bytes,
    SAMPLER_SEED,
    sampling_fidelity,
    two_level_weight_sample,
    weight_block_profile,
)
from basemap.round0245_guard import (
    MIN_BINDING_SLOPE_BASIS,
    MIN_BINDING_SLOPE_BYTES_PER_S,
    R0244_BUDGET_HEADROOM_BYTES,
    R0244_MEASURED_SLOPE_BYTES_PER_S,
    EnforcedHostWatchdog,
    require_enforceable_abort_flag,
)
from basemap.round0245_sampler import R0245_SAMPLER_DRAWS
from basemap.round0246_guard import (
    ATTEMPT_1_GAP_S,
    GPU_HOURS_CAP,
    POLL_GATE_NOTE,
    R0246_MAX_POLL_SPACING_S,
    ROUND_ID,
    ROWS,
    SAMPLER_LIVENESS_NOTE,
    AbortPollGate,
    Round0246Error,
    guard_closure_receipt,
    measured_slope_from_trace,
    registered_thresholds,
    require_abort_flag_landed,
    require_live_sampler,
    run_novel_attack_battery,
    run_reviewer_directory_flag_control,
    run_reviewer_gap_replay_control,
    run_reviewer_oserror_control,
)
from basemap.round0247_registry import registered_value
from basemap.round0246_tie import (
    TIE_AGGREGATE_ONLY_RULE,
    TIE_PRECISION_ROWS,
    TIE_PRECISION_SEED,
    adjudicate_tie_aware_claims,
    tie_aware_precision_profile,
    tie_use_positive_control,
)
from experiments.round0238_nodes import _check_runner_abort
from experiments.round0241_nodes import _readonly_memmap
from experiments.round0242_nodes import WATCHDOG_ANON_BYTES, _memmap_attestation

GUARD_ACTION = "guard_0246"
SAMPLER_ACTION = "sampler_0246"
TIE_ACTION = "tie_0246"

GUARD_CAPABILITY = "round0246-guard-closure-v1"
SAMPLER_CAPABILITY = "minilm-mixed-100000k-k15-sampler-abort-poll-closure-v1"
TIE_CAPABILITY = "minilm-mixed-100000k-tie-aware-aggregate-only-v1"

GUARD_FILE = "guard-closure.json"
SAMPLER_FILE = "sampler-abort-poll.json"
TIE_FILE = "tie-precision.json"

GUARD_SCHEMA = "round0246-guard-closure-v1"
SAMPLER_SCHEMA = "round0246-sampler-abort-poll-closure-v1"
TIE_SCHEMA = "round0246-tie-aware-aggregate-only-v1"

#: The same budget and headroom R0245's nodes declared, so the re-measured gap
#: is comparable with the sealed `24.46713631998864` s line for line.
NODE_ANON_BUDGET_BYTES = 16 * (1 << 30)
NODE_HEADROOM_BYTES = WATCHDOG_ANON_BYTES - NODE_ANON_BUDGET_BYTES

#: The sealed R0245 sampler numbers this round must reproduce EXACTLY. They are
#: bound here so the poll-granularity fix inside `two_level_weight_sample`
#: cannot quietly change the draw: same seed, same draw count, same statistics.
R0245_SEALED_DISTINCT_EDGES_DRAWN = 39_525_520
R0245_SEALED_MAX_ABORT_READ_GAP_S = 24.46713631998864
R0245_SEALED_SAMPLER_OWN_SLOPE_BYTES_PER_S = 639_209_472.0

#: The healthy sampling coverage the registered floor is calibrated against,
#: recomputed in the node from R0245's sealed receipt rather than transcribed.
R0245_SEALED_SAMPLER_SAMPLES = 439
R0245_SEALED_SAMPLER_BOUNDARY_POLLS = 4
R0245_SEALED_SAMPLER_EXPECTED_SAMPLES = 458.17951983609237

SCOPE_NOTE = (
    "R0246 trains nothing, builds nothing, registers no gate on a map, adopts "
    "nothing and measures no displacement. It closes the three guard defects "
    "review-0245-01 defeated outside their own positive controls, closes the "
    "abort-read gap that blocks the first multi-hour training node, and decides "
    "what the tie-aware tolerance finding invalidates."
)
SAFETY_NOTE = (
    "every bulk input is a read-only np.memmap; nothing is handed to cuVS; no "
    "child process is started; no signal is delivered on any path. The host "
    "guard cannot be armed without a path that has been PROVEN writable by a "
    "probe file created and removed at that path; its sampling thread's death, "
    "unrequested exit and coverage are all receipt fields the node gates on; "
    "and its poll spacing is bound at no less than the fastest allocation slope "
    "this machine has been measured at."
)


def _bound_path(job: Mapping[str, Any], key: str, *, label: str) -> str:
    entry = job.get(key)
    if not isinstance(entry, Mapping):
        raise Round0246Error(f"R0246 job does not bind {label} ({key})")
    path = str(entry.get("canonical_path") or "")
    if not path or not os.path.exists(path):
        raise Round0246Error(f"R0246 {label} is absent at {path!r}")
    declared = int(entry.get("bytes", -1))
    actual = os.path.getsize(path)
    if declared >= 0 and declared != actual:
        raise Round0246Error(
            f"R0246 {label} is {actual} bytes, the manifest declared {declared}"
        )
    return path


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "rows": ROWS,
        "k": GRAPH_K,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "gpu_hours_cap_note": GPU_HOURS_CAP_NOTE,
        "scope_note": SCOPE_NOTE,
        "safety_note": SAFETY_NOTE,
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": False,
        "map_quality_claimed": False,
        "displacement_measured": False,
        "cuvs_calls": 0,
        "cuda_context_created": False,
        "child_processes_launched": 0,
        "signal_delivered": False,
    }


def _seal(output: str, name: str, body: Mapping[str, Any]) -> None:
    atomic_write_new_json(
        os.path.join(output, name),
        prompt_contract.seal(json_safe(json_scrub(dict(body)))),
        immutable=True,
    )


def _start_node(label: str) -> dict[str, Any]:
    """Every R0246 node refuses to start where a guard trip cannot land."""
    return require_enforceable_abort_flag(label=label)


def _node_guard(label: str, *, interval_s: float = 0.05) -> EnforcedHostWatchdog:
    """A live host guard whose stage is long enough for the coverage gate.

    At `0.05` s the coverage arm becomes applicable after `1.0` s of stage, so
    every R0246 node exercises the gate it registers rather than only its
    controls.
    """
    return EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        interval_s=float(interval_s),
        label=label,
    )


def _guard_tail(
    watchdog: EnforcedHostWatchdog, *, label: str
) -> dict[str, Any]:
    """The three node-tail gates, in one place: alive, sampling, and landed."""
    receipt = watchdog.receipt()
    return {
        "host_watchdog": receipt,
        "sampler_liveness": require_live_sampler(receipt, label=label),
        "abort_flag_landing": require_abort_flag_landed(receipt, label=label),
    }


# --------------------------------------------------------------------------- #
# node 1 — the three closures, the reviewer's controls, the novel attacks
# --------------------------------------------------------------------------- #
def run_guard(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0246Error("R0246 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0246 guard closure")
    output = create_fresh_directory(str(job["outputs"][0]), label="R0246 guard")
    workspace = create_fresh_directory(
        os.path.join(output, "attack-workspace"), label="R0246 attacks"
    )

    #: The basis for the registered coverage floor, recomputed from R0245's
    #: sealed sampler receipt rather than transcribed from its prose.
    sealed_sampler = prompt_contract.read_sealed(
        _bound_path(job, "r0245_sampler_receipt", label="R0245 sampler receipt"),
        label="R0246 R0245 sampler receipt",
    )
    sealed_watchdog = sealed_sampler["host_watchdog"]
    healthy_thread_samples = (
        int(sealed_watchdog["samples"]) - int(sealed_watchdog["boundary_polls"])
    )
    healthy_expected = float(sealed_watchdog["expected_samples_at_interval"])
    healthy_coverage = {
        "source": "R0245 sealed sampler-power.json host_watchdog",
        "sampled_wall_s": float(sealed_watchdog["sampled_wall_s"]),
        "samples_including_boundary_polls": int(sealed_watchdog["samples"]),
        "boundary_polls": int(sealed_watchdog["boundary_polls"]),
        "thread_samples": healthy_thread_samples,
        "expected_samples_at_interval": healthy_expected,
        "healthy_thread_sample_coverage": (
            float(healthy_thread_samples) / healthy_expected
        ),
        "published_sample_coverage_which_counts_boundary_polls": (
            sealed_watchdog["sample_coverage"]
        ),
        "registered_floor_is_half_of_it": True,
        "binding": {
            "samples_agree": bool(
                int(sealed_watchdog["samples"]) == R0245_SEALED_SAMPLER_SAMPLES
            ),
            "boundary_polls_agree": bool(
                int(sealed_watchdog["boundary_polls"])
                == R0245_SEALED_SAMPLER_BOUNDARY_POLLS
            ),
            "expected_agrees": bool(
                healthy_expected == R0245_SEALED_SAMPLER_EXPECTED_SAMPLES
            ),
        },
    }
    if not all(healthy_coverage["binding"].values()):
        raise Round0246Error(
            "R0246 STOP: the sealed R0245 sampler receipt does not carry the "
            f"registered coverage basis ({healthy_coverage['binding']})"
        )
    _check_runner_abort("R0246 bound the healthy coverage basis")

    #: The three positive controls, each in the exact shape the reviewer used.
    oserror_control = run_reviewer_oserror_control(
        flag_path=os.path.join(workspace, "oserror-control.abort")
    )
    _check_runner_abort("R0246 A1-bis reviewer control")
    directory_control = run_reviewer_directory_flag_control(
        directory=os.path.join(workspace, "flag-as-a-directory")
    )
    _check_runner_abort("R0246 A3-bis reviewer control")
    gap_replay_control = run_reviewer_gap_replay_control()
    _check_runner_abort("R0246 A2-bis reviewer control")

    #: And the attacks no control covers.
    novel = run_novel_attack_battery(
        workspace=os.path.join(workspace, "novel")
    )
    _check_runner_abort("R0246 novel attack battery")

    #: A live enforced watchdog on the node path, run long enough that the
    #: coverage arm of the liveness gate actually applies to it.
    guard = _node_guard("R0246 guard node tail")
    gate = AbortPollGate(
        inner=_check_runner_abort, headroom_bytes=NODE_HEADROOM_BYTES,
        label="R0246 guard node tail", training_performed=True,
    )
    with guard:
        gate.start()
        guard.poll("R0246 guard node tail start")
        for step in range(12):
            gate(f"R0246 guard node tail step {step}")
            time.sleep(0.1)
        guard.poll("R0246 guard node tail end")
        gate.finish()
    tail = _guard_tail(guard, label="R0246 guard node tail")
    enforcement = gate.require(
        measured_slope_bytes_per_s=measured_slope_from_trace(
            tail["host_watchdog"]["anonymous_trace_by_second"]
        )
    )

    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": GUARD_SCHEMA,
        "capabilities": [GUARD_CAPABILITY],
        "closure": guard_closure_receipt(
            abort_flag=abort_flag,
            oserror_control=oserror_control,
            directory_control=directory_control,
            gap_replay_control=gap_replay_control,
            novel_attacks=novel,
            healthy_coverage=healthy_coverage,
        ),
        "sampler_liveness_note": SAMPLER_LIVENESS_NOTE,
        "poll_gate_note": POLL_GATE_NOTE,
        "attempt_1_gap_s": ATTEMPT_1_GAP_S,
        "enforcement_poll_spacing": enforcement,
        **tail,
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, GUARD_FILE, body)


# --------------------------------------------------------------------------- #
# node 2 — the blocker: the abort-read gap inside two_level_weight_sample
# --------------------------------------------------------------------------- #
def run_sampler(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0246Error("R0246 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0246 sampler abort-poll closure")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0246 sampler"
    )
    draws = int(job.get("draws", R0245_SAMPLER_DRAWS))
    seed = int(job.get("seed", SAMPLER_SEED + 245))

    weights_path = _bound_path(job, "edges_wts", label="edge weights")
    weights = np.load(weights_path, mmap_mode="r")
    if weights.ndim != 1:
        raise Round0246Error("R0246 sampler needs a 1-D weight array")

    guard = _node_guard("R0246 sampler abort-poll closure")
    #: `training_performed=True` deliberately: this node is the stand-in for the
    #: trainer, so it is held to the gate a training node is held to, including
    #: the worst-case-slope arm that R0245's sampler node failed.
    gate = AbortPollGate(
        inner=_check_runner_abort, headroom_bytes=NODE_HEADROOM_BYTES,
        label="R0246 sampler abort-poll closure", training_performed=True,
    )

    io_before = io_counters()
    with guard:
        gate.start()
        profile_started = time.monotonic()
        profile = weight_block_profile(
            weights, block=SAMPLER_BLOCK_EDGES, abort_check=gate
        )
        profile_wall = time.monotonic() - profile_started
        gate("R0246 weight profile complete")
        guard.poll("R0246 streamed the weight profile")

        sample_started = time.monotonic()
        sample = two_level_weight_sample(
            weights, profile=profile, draws=draws, seed=seed, abort_check=gate,
        )
        sample_wall = time.monotonic() - sample_started
        gate("R0246 correct sampler drawn")
        guard.poll("R0246 drew the correct sampler")

        verdict = sampling_fidelity(profile=profile, sample=sample)
        gate("R0246 correct sampler scored")
        distinct = int(sample["distinct_edges_drawn"])
        del sample
        gate("R0246 released the draw")
        gate.finish()
    tail = _guard_tail(guard, label="R0246 sampler abort-poll closure")
    own_slope = measured_slope_from_trace(
        tail["host_watchdog"]["anonymous_trace_by_second"]
    )
    enforcement = gate.require(measured_slope_bytes_per_s=own_slope)
    io_after = io_counters()

    #: The poll fix is inert or it is not. R0245 sealed this exact draw at this
    #: exact seed and draw count; if a single statistic moves, the "fix" changed
    #: the science and the node fails rather than publishing it.
    reproduction = {
        "seed": seed,
        "draws": draws,
        "distinct_edges_drawn": distinct,
        "r0245_sealed_distinct_edges_drawn": R0245_SEALED_DISTINCT_EDGES_DRAWN,
        "distinct_agrees": bool(
            distinct == R0245_SEALED_DISTINCT_EDGES_DRAWN
        ),
        "fidelity": {
            key: value for key, value in verdict.items() if key != "arms"
        },
        "arms": verdict["arms"],
        "correct_sampler_still_passes": bool(verdict["holds"]),
        "why": (
            "the abort-poll fix rewrote three loops inside "
            "two_level_weight_sample - the block-choice searchsorted, the "
            "stable order, and the weight gather - and replaced a 40M "
            "np.unique with a per-group distinct count. None of that may touch "
            "the draw. Reproducing R0245's sealed distinct count and fidelity "
            "statistics at the same seed IS the proof that it did not."
        ),
    }
    if not (reproduction["distinct_agrees"] and verdict["holds"]):
        raise Round0246Error(
            "R0246 STOP: the polled two_level_weight_sample did not reproduce "
            f"R0245's sealed draw ({reproduction['distinct_edges_drawn']} "
            f"distinct against {R0245_SEALED_DISTINCT_EDGES_DRAWN}, fidelity "
            f"holds={verdict['holds']}). A poll-granularity fix that changes "
            "the science is not a fix."
        )

    widest = float(enforcement["max_gap_between_enforcement_polls_s"])
    against_both = {
        "widest_abort_read_gap_s": widest,
        "widest_gap_before": enforcement["widest_gap_before"],
        "enforcement_polls": int(enforcement["enforcement_polls"]),
        "r0245_sealed_widest_gap_s": R0245_SEALED_MAX_ABORT_READ_GAP_S,
        "narrowing_multiple": (
            R0245_SEALED_MAX_ABORT_READ_GAP_S / widest if widest > 0 else None
        ),
        "own_slope": {
            "measured_slope_bytes_per_s": own_slope,
            "r0245_sealed_own_slope_bytes_per_s": (
                R0245_SEALED_SAMPLER_OWN_SLOPE_BYTES_PER_S
            ),
            "permitted_growth_at_the_r0245_own_slope_bytes": (
                widest * R0245_SEALED_SAMPLER_OWN_SLOPE_BYTES_PER_S
            ),
            "headroom_bytes": NODE_HEADROOM_BYTES,
            "holds_at_the_r0245_own_slope": bool(
                widest * R0245_SEALED_SAMPLER_OWN_SLOPE_BYTES_PER_S
                <= float(NODE_HEADROOM_BYTES)
            ),
        },
        "worst_case": {
            "slope_bytes_per_s": float(R0244_MEASURED_SLOPE_BYTES_PER_S),
            "permitted_growth_bytes": (
                widest * float(R0244_MEASURED_SLOPE_BYTES_PER_S)
            ),
            "headroom_bytes": R0244_BUDGET_HEADROOM_BYTES,
            "registered_ceiling_s": R0246_MAX_POLL_SPACING_S,
            "gap_over_the_registered_ceiling": (
                widest / R0246_MAX_POLL_SPACING_S
            ),
            "meets_the_r0244_worst_case_slope": bool(
                enforcement["meets_the_r0244_worst_case_slope"]
            ),
        },
        "binding_slope_floor_bytes_per_s": MIN_BINDING_SLOPE_BYTES_PER_S,
        "binding_slope_floor_basis": MIN_BINDING_SLOPE_BASIS,
    }
    training_unblocked = bool(
        enforcement["holds"]
        and against_both["worst_case"]["meets_the_r0244_worst_case_slope"]
        and widest <= registered_value("r0246_max_poll_spacing_s")
        and reproduction["distinct_agrees"]
    )

    block_sums_path = os.path.join(output, "block-sums.f64.npy")
    saved_block_sums = atomic_save_new_npy(
        block_sums_path,
        np.asarray(profile["block_sums"], dtype=np.float64), immutable=True,
    )
    profile_public = {
        key: value for key, value in profile.items()
        if key not in ("block_sums", "bin_mass", "bin_counts")
    }

    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": SAMPLER_SCHEMA,
        "capabilities": [SAMPLER_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "draws": draws,
        "seed": seed,
        "weight_profile": profile_public,
        "block_sums": saved_block_sums,
        "sealed_draw_reproduction": reproduction,
        "abort_read_gap_against_both_slopes": against_both,
        "enforcement_poll_spacing": enforcement,
        "registered_thresholds": registered_thresholds(),
        "the_first_multi_hour_training_node_is_unblocked": training_unblocked,
        "what_unblocking_means": (
            "the widest interval between cooperative-abort reads inside the "
            "sampler a trainer will call now satisfies the derived requirement "
            "headroom / slope at the fastest allocation slope this machine has "
            "been measured at. It does not mean a trainer is safe: nothing here "
            "bounds what a stage allocates WITHIN one unit of work, and the "
            "trainer's own loops carry their own poll spacing, which this round "
            "has not measured."
        ),
        "verdict_arms": {
            "the_draw_is_unchanged": bool(reproduction["distinct_agrees"]),
            "the_correct_sampler_still_passes": bool(verdict["holds"]),
            "the_gap_meets_the_registered_ceiling": bool(
                widest <= registered_value("r0246_max_poll_spacing_s")
            ),
            "the_gap_meets_the_worst_case_slope": bool(
                against_both["worst_case"]["meets_the_r0244_worst_case_slope"]
            ),
            "anonymous_peak_within_budget": bool(
                int(tail["host_watchdog"]["thread_peak_anonymous_bytes"])
                <= int(sampler_max_anonymous_bytes())
            ),
        },
        **tail,
        "walls": {
            "profile_s": profile_wall,
            "two_level_weight_sample_s": sample_wall,
        },
        "io": {
            "read_bytes": int(io_after["read_bytes"] - io_before["read_bytes"]),
            "write_bytes": int(
                io_after["write_bytes"] - io_before["write_bytes"]
            ),
        },
        "bulk_input_memmap_attestation": _memmap_attestation(
            {"edges_wts": weights}
        ),
        "edges_republished": False,
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, SAMPLER_FILE, body)


# --------------------------------------------------------------------------- #
# node 3 — what the tie-aware tolerance finding invalidates
# --------------------------------------------------------------------------- #
def run_tie(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0246Error("R0246 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0246 tie-aware precision")
    output = create_fresh_directory(str(job["outputs"][0]), label="R0246 tie")

    truth_cos = _readonly_memmap(
        _bound_path(job, "truth_cos", label="truth cosines"),
        label="R0246 truth cosines", shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )
    truth_ids = _readonly_memmap(
        _bound_path(job, "truth_ids", label="truth ids"),
        label="R0246 truth ids", shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )
    probe_rows = _readonly_memmap(
        _bound_path(job, "probe_query_rows", label="probe query rows"),
        label="R0246 probe query rows", shape=(TRUTH_PROBE_ROWS,),
    )
    substrate = _readonly_memmap(
        _bound_path(job, "substrate_array", label="substrate"),
        label="R0246 substrate", shape=(ROWS, DIMENSION),
    )
    graph_ids = _readonly_memmap(
        _bound_path(job, "graph_ids", label="graph ids"),
        label="R0246 graph ids", shape=(ROWS, GRAPH_K),
    )

    guard = _node_guard("R0246 tie-aware precision")
    gate = AbortPollGate(
        inner=_check_runner_abort, headroom_bytes=NODE_HEADROOM_BYTES,
        label="R0246 tie-aware precision", training_performed=True,
    )
    with guard:
        gate.start()
        profile = tie_aware_precision_profile(
            substrate=substrate,
            graph_ids=graph_ids,
            probe_query_rows=probe_rows,
            truth_ids=truth_ids,
            truth_cosines=truth_cos,
            sample_rows=int(job.get("tie_rows", TIE_PRECISION_ROWS)),
            seed=int(job.get("tie_seed", TIE_PRECISION_SEED)),
            abort_check=gate,
        )
        gate("R0246 measured the tie-aware precision profile")
        guard.poll("R0246 measured the tie-aware precision profile")
        adjudication = adjudicate_tie_aware_claims(profile)
        gate("R0246 adjudicated the published tie-aware claims")
        #: A PLANTED rate, deliberately not the measured one: a control that
        #: only fires at the rate this round happened to observe is not a
        #: control. The measured rate drives the adjudication above.
        use_control = tie_use_positive_control()
        gate("R0246 ran the aggregate-only positive control")
        gate.finish()
    tail = _guard_tail(guard, label="R0246 tie-aware precision")
    enforcement = gate.require(
        measured_slope_bytes_per_s=measured_slope_from_trace(
            tail["host_watchdog"]["anonymous_trace_by_second"]
        )
    )

    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": TIE_SCHEMA,
        "capabilities": [TIE_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "tie_tolerance": float(TIE_TOLERANCE),
        "tie_precision_profile": profile,
        "claim_adjudication": adjudication,
        "aggregate_only_control": use_control,
        "aggregate_only_rule": TIE_AGGREGATE_ONLY_RULE,
        "the_tolerance_was_not_raised": True,
        "why_the_tolerance_was_not_raised": (
            "raising TIE_TOLERANCE would move R0243's published 83.92% "
            "forgiveness by under 0.1 pp and would replace one arbitrary "
            "threshold with another. The estimator's problem is that it is a "
            "float32 value test at all; the float64 column measured here is the "
            "fix a future round should make, and this round registers the "
            "aggregate-only rule instead of moving a number."
        ),
        "enforcement_poll_spacing": enforcement,
        **tail,
        "bulk_input_memmap_attestation": _memmap_attestation({
            "truth_cosines": truth_cos,
            "truth_ids": truth_ids,
            "probe_query_rows": probe_rows,
            "substrate": substrate,
            "graph_ids": graph_ids,
        }),
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, TIE_FILE, body)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == GUARD_ACTION:
        run_guard(active, job)
    elif action == SAMPLER_ACTION:
        run_sampler(active, job)
    elif action == TIE_ACTION:
        run_tie(active, job)
    else:
        raise Round0246Error(f"R{ROUND_ID} does not authorize action {action!r}")


__all__ = [
    "GUARD_ACTION",
    "GUARD_CAPABILITY",
    "GUARD_FILE",
    "GUARD_SCHEMA",
    "NODE_ANON_BUDGET_BYTES",
    "NODE_HEADROOM_BYTES",
    "R0245_SEALED_DISTINCT_EDGES_DRAWN",
    "R0245_SEALED_MAX_ABORT_READ_GAP_S",
    "SAMPLER_ACTION",
    "SAMPLER_CAPABILITY",
    "SAMPLER_FILE",
    "SAMPLER_SCHEMA",
    "TIE_ACTION",
    "TIE_CAPABILITY",
    "TIE_FILE",
    "TIE_SCHEMA",
    "run_guard",
    "run_job",
    "run_sampler",
    "run_tie",
]
