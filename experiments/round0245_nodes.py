"""Execute R0245 — the five defects review-0244-01 said stand before a long node.

Three nodes, none of which trains anything, builds anything, or creates a CUDA
context:

* `guard_0245` — the three guard fixes and their three positive controls. It
  re-derives the allocation slope from R0244's **sealed** trace rather than
  trusting the constant, scores R0243's own stripe spacing against the derived
  requirement, and runs a synthetic stage faster than that slope in two arms:
  one that meets the requirement and is stopped inside its headroom, one that
  violates it and overshoots. The node also demonstrates the failure it fixes:
  a sampler killed by a non-`OSError`, and a node started with
  `ROUNDRUN_ABORT_FLAG` unset.
* `did_0245` — the decision map as an executable gate with its underpowered-null
  positive control; the arm-assignment fix on the real sealed vectors, with
  disjointness enforced rather than inspected; the forensic recompute that
  explains why two probe rows carry more tie-aware loss than strict loss; and
  the permutation design's cost, simulated.
* `sampler_0245` — finding 7: the mis-sampler family, the draw-count floor as an
  executable gate, and the check's blind spot stated as a limit of the
  capability.

Every registered check is IMPORTED, never re-typed: `sampling_fidelity`,
`weight_block_profile`, `two_level_weight_sample`, `did_populations`,
`did_registration`, `exact_displacement_permutation`, `holm_bonferroni`,
`_readonly_memmap`, `_memmap_attestation`, `_check_runner_abort`, `io_counters`,
`json_scrub`. R0244's `ThreadedHostWatchdog` is subclassed, and the one edit to
it — catch `BaseException`, record the death, raise from `poll()` — changes no
threshold and no rule.

No node in this module starts a child process, hands cuVS anything, or contains
a signalling construct of any kind.
"""
from __future__ import annotations

import gc
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
    did_registration,
    did_requirement,
    weight_block_profile,
)
from basemap.round0245_did import (
    DECISION_NOTE,
    TIE_EFFECTIVE_RULE,
    TIE_SCORING_NOTE,
    cosine_agreement_profile,
    did_decision_positive_controls,
    did_populations_v2,
    permutation_design_cost,
    tie_effective_builder_missing,
    tie_monotonicity_audit,
    tie_violation_forensics,
)
from basemap.round0245_guard import (
    GPU_HOURS_CAP,
    POLL_SPACING_NOTE,
    R0244_BUDGET_HEADROOM_BYTES,
    R0244_MEASURED_SLOPE_BYTES_PER_S,
    R0244_STRIPE_LATENCY_S,
    ROUND_ID,
    ROWS,
    AbortPollTracker,
    EnforcedHostWatchdog,
    Round0245Error,
    guard_receipt,
    r0244_stripe_verdict,
    require_enforceable_abort_flag,
    run_allocation_slope_positive_control,
    run_missing_flag_positive_control,
    run_thread_death_positive_control,
    slope_from_trace,
)
from basemap.round0245_sampler import (
    MIS_SAMPLER_FAMILY,
    R0245_SAMPLER_DRAWS,
    block_profile_dispersion,
    certified_weight_floor,
    mis_sampler_battery,
    sampler_draw_floor,
    sampler_limits,
    true_sampler_reference,
    uniform_position_draw_requirement,
)
from experiments.round0238_nodes import _check_runner_abort
from experiments.round0241_nodes import _readonly_memmap
from experiments.round0242_nodes import WATCHDOG_ANON_BYTES, _memmap_attestation

GUARD_ACTION = "guard_0245"
DID_ACTION = "did_0245"
SAMPLER_ACTION = "sampler_0245"

GUARD_CAPABILITY = "round0245-guard-closure-v1"
DID_CAPABILITY = "minilm-mixed-100000k-displacement-did-decision-gate-v1"
SAMPLER_CAPABILITY = "minilm-mixed-100000k-k15-sampler-power-and-limits-v1"

GUARD_FILE = "guard-closure.json"
DID_FILE = "did-decision-gate.json"
SAMPLER_FILE = "sampler-power.json"

GUARD_SCHEMA = "round0245-guard-closure-v1"
DID_SCHEMA = "round0245-did-decision-gate-and-arm-assignment-v1"
SAMPLER_SCHEMA = "round0245-sampler-power-and-limits-v1"

#: The sealed R0244 values this round re-derives rather than trusts.
R0244_TRIP_TRACE_SLOPE_BYTES_PER_S = R0244_MEASURED_SLOPE_BYTES_PER_S
R0244_WATCHDOG_RECEIPT_PEAK_BYTES = 39_170_588_672

#: R0243's sealed per-row vectors, at the totals reviews 0243-01 and 0244-01
#: both recomputed. Bound so a different vector cannot be re-analysed as if it
#: were this one.
R0243_STRICT_BUILDER_MISSING_EDGES = 18_129
R0243_TIE_AWARE_BUILDER_MISSING_EDGES = 2_915

COSINE_AGREEMENT_ROWS = 20_000
COSINE_AGREEMENT_SEED = 245_001

#: What every R0245 node declares as its own anonymous budget, and the headroom
#: that budget leaves against R0242's machine-level anonymous threshold. The
#: poll-spacing requirement is evaluated against THIS headroom, so the number a
#: node must beat follows from the budget it declares rather than from taste.
NODE_ANON_BUDGET_BYTES = 16 * (1 << 30)
NODE_HEADROOM_BYTES = WATCHDOG_ANON_BYTES - NODE_ANON_BUDGET_BYTES

SCOPE_NOTE = (
    "R0245 trains nothing, builds nothing, registers no gate on a map, adopts "
    "nothing and measures no displacement. It closes the three guard defects "
    "review-0244-01 put in front of a ~10 h node, makes the DiD decision map "
    "executable, fixes the arm assignment two probe rows corrupted, and "
    "documents the two findings the review said must not be silently accepted."
)
SAFETY_NOTE = (
    "every bulk input is a read-only np.memmap, re-measured on the live "
    "objects at seal time; nothing is handed to cuVS; no child process is "
    "started; no signal is delivered on any path. The host guard cannot be "
    "armed without a writable cooperative abort flag, its sampling thread's "
    "death is a receipt field a node gates on, and its poll spacing is checked "
    "against the measured allocation slope before the stage runs."
)


def _bound_path(job: Mapping[str, Any], key: str, *, label: str) -> str:
    entry = job.get(key)
    if not isinstance(entry, Mapping):
        raise Round0245Error(f"R0245 job does not bind {label} ({key})")
    path = str(entry.get("canonical_path") or "")
    if not path or not os.path.exists(path):
        raise Round0245Error(f"R0245 {label} is absent at {path!r}")
    declared = int(entry.get("bytes", -1))
    actual = os.path.getsize(path)
    if declared >= 0 and declared != actual:
        raise Round0245Error(
            f"R0245 {label} is {actual} bytes, the manifest declared {declared}"
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
    """Every R0245 node refuses to start where a guard trip cannot land."""
    return require_enforceable_abort_flag(label=label)


def _require_live_sampler(receipt: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """A node may not publish a memory receipt written by a dead sampler.

    review-0244-01 A1: the death was discoverable only after the run, by
    reading `sample_coverage`, and no node gated on it. This is the gate.
    """
    state = {
        "label": label,
        "sampling_thread_alive": bool(receipt["sampling_thread_alive"]),
        "thread_death": receipt["thread_death"],
        "samples": int(receipt["samples"]),
        "sample_coverage": receipt["sample_coverage"],
    }
    if not state["sampling_thread_alive"]:
        raise Round0245Error(
            f"R0245 STOP: {label} ran behind a host guard whose sampling "
            f"thread died ({receipt['thread_death']}). Its memory receipt is "
            "not evidence and the node must not publish it."
        )
    return state


# --------------------------------------------------------------------------- #
# node 1 — the three guard fixes and their three positive controls
# --------------------------------------------------------------------------- #
def run_guard(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0245Error("R0245 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0245 guard closure")
    output = create_fresh_directory(str(job["outputs"][0]), label="R0245 guard")

    #: 1. Re-derive the slope from R0244's SEALED trace. The registered
    #: constant is a registration; this is the measurement behind it.
    receipt_path = _bound_path(
        job, "r0244_watchdog_receipt", label="R0244 watchdog receipt"
    )
    sealed = prompt_contract.read_sealed(
        receipt_path, label="R0245 R0244 watchdog receipt"
    )
    watchdog_receipt = sealed["host_watchdog"]
    slope = slope_from_trace(watchdog_receipt["anonymous_trace_by_second"])
    binding = {
        "sealed_receipt": receipt_path,
        "recomputed_slope_bytes_per_s": slope["max_rise_bytes_per_s"],
        "registered_slope_bytes_per_s": float(
            R0244_TRIP_TRACE_SLOPE_BYTES_PER_S
        ),
        "slope_agrees": bool(
            slope["max_rise_bytes_per_s"]
            == float(R0244_TRIP_TRACE_SLOPE_BYTES_PER_S)
        ),
        "recomputed_peak_bytes": slope["peak_bytes"],
        "registered_peak_bytes": R0244_WATCHDOG_RECEIPT_PEAK_BYTES,
        "peak_agrees": bool(
            slope["peak_bytes"] == R0244_WATCHDOG_RECEIPT_PEAK_BYTES
        ),
        "sealed_headroom_bytes": int(watchdog_receipt["budget_headroom_bytes"]),
        "registered_headroom_bytes": R0244_BUDGET_HEADROOM_BYTES,
        "headroom_agrees": bool(
            int(watchdog_receipt["budget_headroom_bytes"])
            == R0244_BUDGET_HEADROOM_BYTES
        ),
        "stripe_latency_s": R0244_STRIPE_LATENCY_S,
        "sealed_stripes": int(sealed["fuzzy"]["stripes"]),
        "sealed_fuzzy_wall_s": float(
            sealed["stage_walls"]["fuzzy_symmetrisation_s"]
        ),
        "trace": slope,
    }
    if not (
        binding["slope_agrees"]
        and binding["peak_agrees"]
        and binding["headroom_agrees"]
    ):
        raise Round0245Error(
            "R0245 STOP: the slope, peak or headroom re-derived from R0244's "
            f"sealed trace does not match the registered constants ({binding})"
        )
    _check_runner_abort("R0245 rederived the allocation slope")

    #: 2. Score R0243's own stripe spacing against the derived requirement.
    stripe = r0244_stripe_verdict()

    #: 3. The three positive controls. Each plants the defect.
    thread_death = run_thread_death_positive_control(
        flag_path=os.path.join(output, "thread-death-control.abort")
    )
    _check_runner_abort("R0245 thread-death control")
    missing_flag = run_missing_flag_positive_control()
    _check_runner_abort("R0245 missing-flag control")
    slope_control = run_allocation_slope_positive_control(
        flag_path=os.path.join(output, "slope-control.abort")
    )
    _check_runner_abort("R0245 allocation-slope control")
    gc.collect()

    #: 4. A live enforced watchdog over the node's own tail, so the fixed guard
    #: is exercised on the node path and not only in its controls.
    guard = EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        label="R0245 guard node tail",
    )
    with guard:
        guard.poll("R0245 guard node tail start")
        time.sleep(0.6)
        guard.poll("R0245 guard node tail end")
    guard_state = guard.receipt()

    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": GUARD_SCHEMA,
        "capabilities": [GUARD_CAPABILITY],
        "closure": guard_receipt(
            abort_flag=abort_flag,
            thread_death=thread_death,
            missing_flag=missing_flag,
            slope=slope_control,
            stripe=stripe,
        ),
        "r0244_slope_binding": binding,
        "poll_spacing_note": POLL_SPACING_NOTE,
        "node_tail_watchdog": guard_state,
        "sampler_liveness": _require_live_sampler(
            guard_state, label="R0245 guard node tail"
        ),
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, GUARD_FILE, body)


# --------------------------------------------------------------------------- #
# node 2 — the DiD decision gate and the arm assignment
# --------------------------------------------------------------------------- #
def run_did(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0245Error("R0245 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0245 DiD decision gate")
    output = create_fresh_directory(str(job["outputs"][0]), label="R0245 DiD")
    guard = EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        label="R0245 DiD decision gate",
    )
    tracker = AbortPollTracker(
        inner=_check_runner_abort, headroom_bytes=NODE_HEADROOM_BYTES,
        label="R0245 DiD decision gate",
    )

    strict_builder = _readonly_memmap(
        _bound_path(
            job, "r0242_probe_builder_missing_edges", label="builder missing"
        ),
        label="R0245 strict builder-missing", shape=(TRUTH_PROBE_ROWS,),
    )
    tie_builder = _readonly_memmap(
        _bound_path(
            job, "r0243_probe_tie_aware_builder_missing_edges",
            label="tie-aware builder missing",
        ),
        label="R0245 tie-aware builder-missing", shape=(TRUTH_PROBE_ROWS,),
    )
    truth_cos = _readonly_memmap(
        _bound_path(job, "truth_cos", label="truth cosines"),
        label="R0245 truth cosines", shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )
    truth_ids = _readonly_memmap(
        _bound_path(job, "truth_ids", label="truth ids"),
        label="R0245 truth ids", shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )
    probe_rows = _readonly_memmap(
        _bound_path(job, "probe_query_rows", label="probe query rows"),
        label="R0245 probe query rows", shape=(TRUTH_PROBE_ROWS,),
    )
    substrate = _readonly_memmap(
        _bound_path(job, "substrate_array", label="substrate"),
        label="R0245 substrate", shape=(ROWS, DIMENSION),
    )
    graph_ids = _readonly_memmap(
        _bound_path(job, "graph_ids", label="graph ids"),
        label="R0245 graph ids", shape=(ROWS, GRAPH_K),
    )

    with guard:
        strict = np.asarray(strict_builder, dtype=np.int64)
        tie = np.asarray(tie_builder, dtype=np.int64)
        vector_binding = {
            "strict_builder_missing_edges": int(strict.sum()),
            "r0243_strict_builder_missing_edges": (
                R0243_STRICT_BUILDER_MISSING_EDGES
            ),
            "tie_aware_builder_missing_edges": int(tie.sum()),
            "r0243_tie_aware_builder_missing_edges": (
                R0243_TIE_AWARE_BUILDER_MISSING_EDGES
            ),
            "strict_agrees": bool(
                int(strict.sum()) == R0243_STRICT_BUILDER_MISSING_EDGES
            ),
            "tie_agrees": bool(
                int(tie.sum()) == R0243_TIE_AWARE_BUILDER_MISSING_EDGES
            ),
        }
        if not (vector_binding["strict_agrees"] and vector_binding["tie_agrees"]):
            raise Round0245Error(
                "R0245 STOP: the bound per-row vectors do not sum to R0243's "
                f"sealed totals ({vector_binding}); this is not that analysis"
            )
        guard.poll("R0245 bound the sealed loss vectors")

        #: 4. The decision map, as a gate, with its branches planted.
        decision_controls = did_decision_positive_controls()
        tracker("R0245 DiD decision controls")

        #: 5. The arm assignment, on the real vectors.
        audit = tie_monotonicity_audit(
            strict_builder_missing=strict, tie_aware_builder_missing=tie
        )
        effective = tie_effective_builder_missing(
            strict_builder_missing=strict, tie_aware_builder_missing=tie
        )
        populations = did_populations_v2(
            strict_builder_missing=strict,
            tie_aware_builder_missing=tie,
            kth_cosine=np.asarray(truth_cos[:, GRAPH_K - 1], dtype=np.float64),
        )
        guard.poll("R0245 rebuilt the DiD populations on the clamped vector")

        #: The forensics: why the two rows disagree, from the substrate bytes.
        forensics = tie_violation_forensics(
            probe_rows_to_explain=audit["violating_rows"],
            substrate=substrate,
            graph_ids=graph_ids,
            probe_query_rows=probe_rows,
            truth_ids=truth_ids,
            truth_cosines=truth_cos,
        )
        tracker("R0245 tie violation forensics")
        agreement = cosine_agreement_profile(
            substrate=substrate,
            graph_ids=graph_ids,
            probe_query_rows=probe_rows,
            truth_ids=truth_ids,
            truth_cosines=truth_cos,
            sample_rows=COSINE_AGREEMENT_ROWS,
            seed=COSINE_AGREEMENT_SEED,
            abort_check=tracker,
        )
        guard.poll("R0245 measured the cosine agreement floor")

        #: 6. What the unrestricted permutation costs.
        permutation = permutation_design_cost(abort_check=tracker)
        tracker("R0245 priced the permutation design")
        guard.poll("R0245 priced the permutation design")
    did_guard_receipt = guard.receipt()
    enforcement = tracker.require(
        measured_slope_bytes_per_s=slope_from_trace(
            did_guard_receipt["anonymous_trace_by_second"]
        )["max_rise_bytes_per_s"]
    )

    vector_dir = create_fresh_directory(
        os.path.join(output, "vectors"), label="R0245 DiD vectors"
    )
    saved = {
        "tie_effective_builder_missing": atomic_save_new_npy(
            os.path.join(
                vector_dir, "probe-tie-effective-builder-missing-edges.i16.npy"
            ),
            effective.astype(np.int16), immutable=True,
        ),
        "genuine_lost_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-v2-genuine-lost-rows.i64.npy"),
            np.asarray(populations["genuine"]["lost_sample"], dtype=np.int64),
            immutable=True,
        ),
        "genuine_control_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-v2-genuine-control-rows.i64.npy"),
            np.asarray(
                populations["genuine"]["control_sample"], dtype=np.int64
            ),
            immutable=True,
        ),
        "tie_forgiven_lost_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-v2-tie-forgiven-lost-rows.i64.npy"),
            np.asarray(
                populations["tie_forgiven"]["lost_sample"], dtype=np.int64
            ),
            immutable=True,
        ),
        "tie_forgiven_control_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-v2-tie-forgiven-control-rows.i64.npy"),
            np.asarray(
                populations["tie_forgiven"]["control_sample"], dtype=np.int64
            ),
            immutable=True,
        ),
        "placebo_a_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-v2-placebo-a-rows.i64.npy"),
            np.asarray(populations["placebo_a"], dtype=np.int64),
            immutable=True,
        ),
        "placebo_b_rows": atomic_save_new_npy(
            os.path.join(vector_dir, "did-v2-placebo-b-rows.i64.npy"),
            np.asarray(populations["placebo_b"], dtype=np.int64),
            immutable=True,
        ),
    }

    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": DID_SCHEMA,
        "capabilities": [DID_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "registration_inherited": did_registration(),
        "requirement_inherited": did_requirement(),
        "decision_gate": decision_controls,
        "decision_note": DECISION_NOTE,
        "sealed_vector_binding": vector_binding,
        "tie_monotonicity_audit": audit,
        "tie_violation_forensics": forensics,
        "cosine_agreement_profile": agreement,
        "tie_tolerance": float(TIE_TOLERANCE),
        "tie_scoring_note": TIE_SCORING_NOTE,
        "arm_assignment_rule": TIE_EFFECTIVE_RULE,
        "arm_disjointness": populations["arm_disjointness"],
        "populations": {
            key: value for key, value in populations.items()
            if key not in (
                "genuine", "tie_forgiven", "placebo_a", "placebo_b",
                "tie_monotonicity_audit", "arm_disjointness",
            )
        },
        "population_sizes": {
            "genuine_lost_sample": int(
                np.asarray(populations["genuine"]["lost_sample"]).size
            ),
            "genuine_control_sample": int(
                np.asarray(populations["genuine"]["control_sample"]).size
            ),
            "tie_forgiven_lost_sample": int(
                np.asarray(populations["tie_forgiven"]["lost_sample"]).size
            ),
            "tie_forgiven_control_sample": int(
                np.asarray(populations["tie_forgiven"]["control_sample"]).size
            ),
            "placebo_a": int(np.asarray(populations["placebo_a"]).size),
            "placebo_b": int(np.asarray(populations["placebo_b"]).size),
        },
        "permutation_design_cost": permutation,
        "enforcement_poll_spacing": enforcement,
        "sealed_vectors": saved,
        "sampler_liveness": _require_live_sampler(
            did_guard_receipt, label="R0245 DiD decision gate"
        ),
        "host_watchdog": did_guard_receipt,
        "bulk_input_memmap_attestation": _memmap_attestation({
            "strict_builder_missing": strict_builder,
            "tie_aware_builder_missing": tie_builder,
            "truth_cosines": truth_cos,
            "truth_ids": truth_ids,
            "probe_query_rows": probe_rows,
            "substrate": substrate,
            "graph_ids": graph_ids,
        }),
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, DID_FILE, body)


# --------------------------------------------------------------------------- #
# node 3 — the sampler's power and its limits
# --------------------------------------------------------------------------- #
def run_sampler(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0245Error("R0245 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0245 sampler power")
    output = create_fresh_directory(str(job["outputs"][0]), label="R0245 sampler")
    guard = EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        label="R0245 sampler power",
    )
    tracker = AbortPollTracker(
        inner=_check_runner_abort, headroom_bytes=NODE_HEADROOM_BYTES,
        label="R0245 sampler power",
    )
    draws = int(job.get("draws", R0245_SAMPLER_DRAWS))

    weights_path = _bound_path(job, "edges_wts", label="edge weights")
    weights = np.load(weights_path, mmap_mode="r")
    if weights.ndim != 1:
        raise Round0245Error("R0245 sampler needs a 1-D weight array")

    io_before = io_counters()
    with guard:
        profile_started = time.monotonic()
        profile = weight_block_profile(
            weights, block=SAMPLER_BLOCK_EDGES, abort_check=tracker
        )
        profile_wall = time.monotonic() - profile_started
        guard.poll("R0245 streamed the weight profile")

        dispersion = block_profile_dispersion(profile)
        uniform_requirement = uniform_position_draw_requirement(profile=profile)
        blind_spot = certified_weight_floor(profile=profile, draws=draws)
        guard.poll("R0245 computed the arm power curves")

        reference = true_sampler_reference(
            weights, profile=profile, draws=draws, abort_check=tracker,
        )
        if not reference["holds"]:
            raise Round0245Error(
                "R0245 STOP: the CORRECT two-level sampler failed the fidelity "
                f"check at a fresh seed ({reference['fidelity']}); a battery of "
                "wrong samplers proves nothing if the right one does not pass"
            )
        guard.poll("R0245 scored the correct sampler at a fresh seed")

        battery = mis_sampler_battery(
            weights, profile=profile, draws=draws, abort_check=tracker,
        )
        guard.poll("R0245 scored the mis-sampler family")
        floor = sampler_draw_floor(
            profile=profile, battery=battery, registered_draws=draws
        )
        limits = sampler_limits(
            battery=battery, draw_floor=floor, blind_spot=blind_spot,
            dispersion=dispersion,
        )
    sampler_guard_receipt = guard.receipt()
    enforcement = tracker.require(
        measured_slope_bytes_per_s=slope_from_trace(
            sampler_guard_receipt["anonymous_trace_by_second"]
        )["max_rise_bytes_per_s"]
    )
    io_after = io_counters()

    profile_public = {
        key: value for key, value in profile.items()
        if key not in ("block_sums", "bin_mass", "bin_counts")
    }
    profile_public["bin_counts"] = [
        int(value) for value in np.asarray(profile["bin_counts"])
    ]
    profile_public["bin_mass"] = [
        float(value) for value in np.asarray(profile["bin_mass"])
    ]
    block_sums_path = os.path.join(output, "block-sums.f64.npy")
    saved_block_sums = atomic_save_new_npy(
        block_sums_path,
        np.asarray(profile["block_sums"], dtype=np.float64), immutable=True,
    )

    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": SAMPLER_SCHEMA,
        "capabilities": [SAMPLER_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "draws": draws,
        "weight_profile": profile_public,
        "block_sums": saved_block_sums,
        "block_dispersion": dispersion,
        "uniform_position_draw_requirement": uniform_requirement,
        "blind_spot": blind_spot,
        "correct_sampler_reference": reference,
        "mis_sampler_family": [name for name, _text in MIS_SAMPLER_FAMILY],
        "mis_sampler_battery": battery,
        "draw_floor": floor,
        "limits": limits,
        "enforcement_poll_spacing": enforcement,
        "sampler_liveness": _require_live_sampler(
            sampler_guard_receipt, label="R0245 sampler power"
        ),
        "verdict_arms": {
            "correct_sampler_passes_at_a_fresh_seed": bool(reference["holds"]),
            "family_members_the_arms_missed": len(battery["missed"]),
            "registered_draws_clear_the_floor": bool(
                floor["registered_draws_clear_the_floor"]
            ),
            "anonymous_peak_within_budget": bool(
                int(sampler_guard_receipt["thread_peak_anonymous_bytes"])
                <= int(SAMPLER_MAX_ANONYMOUS_BYTES)
            ),
        },
        "host_watchdog": sampler_guard_receipt,
        "walls": {"profile_s": profile_wall},
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


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == GUARD_ACTION:
        run_guard(active, job)
    elif action == DID_ACTION:
        run_did(active, job)
    elif action == SAMPLER_ACTION:
        run_sampler(active, job)
    else:
        raise Round0245Error(f"R{ROUND_ID} does not authorize action {action!r}")


__all__ = [
    "COSINE_AGREEMENT_ROWS",
    "DID_ACTION",
    "DID_CAPABILITY",
    "DID_FILE",
    "DID_SCHEMA",
    "GUARD_ACTION",
    "GUARD_CAPABILITY",
    "GUARD_FILE",
    "GUARD_SCHEMA",
    "SAMPLER_ACTION",
    "SAMPLER_CAPABILITY",
    "SAMPLER_FILE",
    "SAMPLER_SCHEMA",
    "run_did",
    "run_guard",
    "run_job",
    "run_sampler",
]
