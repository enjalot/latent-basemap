"""Execute R0247 — fix the class, not the five instances, and seal the ledger.

Three CPU-shaped nodes, none of which trains anything, builds anything, or
creates a CUDA context:

* `paramguard_0247` — the safety-parameter inventory, one positive control per
  registered parameter through its own construction path, the `5.0` s coverage
  attack review-0246-01 A published, the reviewer's sixteenth attack in its
  exact construction, and a battery of attacks on R0247's own fix with every
  residual stated. A live enforced guard runs on the node path so the node is
  held to the gates it registers.
* `truthcos_0247` — the precision fix review-0246-01 F showed was cheap.
  R0238's truth **ids** are sealed, so the cosines are recomputed in `float64`
  from the same substrate bytes, the storage/arithmetic decomposition is
  measured, and the tolerance the new floor supports is stated (not applied).
* `tie_0247` — the flip rate re-measured against the recomputed `float64`
  reference over the **whole** `500,000`-row probe (`7,500,000` decisions, the
  probe's entire decision population), the bound generalised from the rule of
  three, and the ledger adjudication sealed as a receipt at both registered
  criteria.

Every registered check is IMPORTED, never re-typed. No node in this module
starts a child process, hands cuVS anything, or contains a signalling construct
of any kind.
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
from basemap.round0245_guard import (
    EnforcedHostWatchdog,
    require_enforceable_abort_flag,
)
from basemap.round0246_guard import (
    AbortPollGate,
    measured_slope_from_trace,
    require_abort_flag_landed,
    require_enforcement_evidence,
    require_live_sampler,
)
from basemap.round0246_tie import (
    TIE_AGGREGATE_ONLY_RULE,
    TIE_AWARE_CLAIM_LEDGER,
    TIE_PRECISION_ROWS,
    TIE_PRECISION_SEED,
    adjudicate_tie_aware_claims,
    tie_aware_precision_profile,
    tie_use_positive_control,
)
from basemap.round0247_guard import (
    run_call_site_controls,
    run_clamp_controls,
    run_coverage_denominator_control,
    run_reviewer_sixteenth_attack_control,
    run_self_attack_battery,
    safety_closure_receipt,
)
from basemap.round0247_precision import (
    PRECISION_NOTE,
    cosine_noise_floor,
    defensible_tolerance,
    flip_rate_bound,
    recompute_truth_cosines_f64,
    sealed_bound_adjudication,
)
from basemap.round0247_registry import (
    GPU_HOURS_CAP,
    ROUND_ID,
    ROWS,
    SAFETY_PARAMETER_CLASS_NOTE,
    Round0247Error,
    registered_bounds,
    registry_fingerprint,
    safety_parameter_inventory,
    verify_registry,
)
from experiments.round0238_nodes import _check_runner_abort
from experiments.round0241_nodes import _readonly_memmap
from experiments.round0242_nodes import _memmap_attestation

PARAMGUARD_ACTION = "paramguard_0247"
TRUTHCOS_ACTION = "truthcos_0247"
TIE_ACTION = "tie_0247"

PARAMGUARD_CAPABILITY = "round0247-safety-parameter-registry-v1"
TRUTHCOS_CAPABILITY = "minilm-mixed-100000k-uniform-probe-k15-truth-cos-f64-v1"
TIE_CAPABILITY = "minilm-mixed-100000k-tie-aware-sealed-bound-adjudication-v1"

PARAMGUARD_FILE = "safety-parameter-closure.json"
TRUTHCOS_FILE = "truth-cosine-precision.json"
TIE_FILE = "tie-sealed-bound.json"

PARAMGUARD_SCHEMA = "round0247-safety-parameter-closure-v1"
TRUTHCOS_SCHEMA = "round0247-truth-cosine-precision-v1"
TIE_SCHEMA = "round0247-tie-sealed-bound-v1"

TRUTH_COS_F64_FILE = "truth-k15-cos.f64.npy"

#: The node's own anonymous budget. Its headroom is now the registered one -
#: R0246's nodes declared `WATCHDOG_ANON_BYTES - budget = 47,244,640,256` B,
#: which is above R0244's sealed `29,548,888,064` B and is clamped to it.
NODE_ANON_BUDGET_BYTES = 16 * (1 << 30)

#: The whole probe. `500,000 x 15 = 7,500,000` is the entire decision
#: population of R0238's truth build, so this is the tightest rule-of-three
#: bound this probe can produce; tightening further needs a larger probe, not a
#: larger sample. review-0246-01 H7 asked for `>= 10^7` decisions and `7.5e6` is
#: the ceiling that exists.
TIE_FULL_PROBE_ROWS = TRUTH_PROBE_ROWS

SCOPE_NOTE = (
    "R0247 trains nothing, builds nothing, registers no gate on a map, adopts "
    "nothing and measures no displacement. It makes every safety parameter in "
    "R0244-R0246 non-overridable from its registered value, replaces the "
    "self-declared coverage denominator with two measured second-valued bounds, "
    "recomputes R0238's truth cosines in float64 from the sealed truth ids, and "
    "seals the tie-aware bound adjudication that R0246 published as prose."
)
SAFETY_NOTE = (
    "every bulk input is a read-only np.memmap; nothing is handed to cuVS; no "
    "child process is started; no signal is delivered on any path. Every bound "
    "the host guard and the poll gate apply comes from the R0247 registry, "
    "which is a read-only mapping of frozen entries under a pinned SHA-256 that "
    "every gate verifies; a caller may be stricter and may not be weaker; and "
    "every registered_* field in every receipt is produced by "
    "registered_bounds(), which reads the registry and cannot read a caller."
)


def _bound_path(job: Mapping[str, Any], key: str, *, label: str) -> str:
    entry = job.get(key)
    if not isinstance(entry, Mapping):
        raise Round0247Error(f"R0247 job does not bind {label} ({key})")
    path = str(entry.get("canonical_path") or "")
    if not path or not os.path.exists(path):
        raise Round0247Error(f"R0247 {label} is absent at {path!r}")
    declared = int(entry.get("bytes", -1))
    actual = os.path.getsize(path)
    if declared >= 0 and declared != actual:
        raise Round0247Error(
            f"R0247 {label} is {actual} bytes, the manifest declared {declared}"
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
        "safety_parameter_class_note": SAFETY_PARAMETER_CLASS_NOTE,
        "registry_fingerprint": registry_fingerprint(),
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
    """Every R0247 node verifies the registry and its abort path first."""
    verify_registry(label=label)
    return require_enforceable_abort_flag(label=label)


def _node_guard(label: str, *, interval_s: float = 0.05) -> EnforcedHostWatchdog:
    return EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        interval_s=float(interval_s),
        label=label,
    )


def _node_gate(label: str) -> AbortPollGate:
    """The registered headroom, the registered ceiling, the registered reader.

    Nothing here is a keyword any more: the headroom comes from the registry,
    the ceiling has no constructor argument this node supplies, and `inner` is
    the one function in the release that reads the cooperative abort flag.
    """
    return AbortPollGate(
        inner=_check_runner_abort,
        headroom_bytes=int(
            registered_bounds(["max_declared_headroom_bytes"])[
                "registered_max_declared_headroom_bytes"
            ]
        ),
        label=label,
        training_performed=True,
    )


def _guard_tail(watchdog: EnforcedHostWatchdog, *, label: str) -> dict[str, Any]:
    receipt = watchdog.receipt()
    return {
        "host_watchdog": receipt,
        "sampler_liveness": require_live_sampler(receipt, label=label),
        "abort_flag_landing": require_abort_flag_landed(receipt, label=label),
    }


def _close_gate(
    gate: AbortPollGate, tail: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Score the gate, then prove the verdict is enforcement evidence."""
    enforcement = gate.require(
        measured_slope_bytes_per_s=measured_slope_from_trace(
            tail["host_watchdog"]["anonymous_trace_by_second"]
        )
    )
    enforcement["enforcement_evidence"] = require_enforcement_evidence(
        enforcement, label=label
    )
    return enforcement


# --------------------------------------------------------------------------- #
# node 1 — the class fix, its controls, and the attacks on it
# --------------------------------------------------------------------------- #
def run_paramguard(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0247Error("R0247 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0247 safety-parameter closure")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0247 paramguard"
    )
    workspace = create_fresh_directory(
        os.path.join(output, "control-workspace"), label="R0247 controls"
    )
    flag_path = os.path.join(workspace, "r0247-control.abort")

    inventory = safety_parameter_inventory()
    _check_runner_abort("R0247 published the safety-parameter inventory")
    clamp_controls = run_clamp_controls()
    _check_runner_abort("R0247 clamp controls")
    call_site_controls = run_call_site_controls(flag_path=flag_path)
    _check_runner_abort("R0247 call-site controls")
    denominator_control = run_coverage_denominator_control()
    _check_runner_abort("R0247 coverage denominator control")
    sixteenth = run_reviewer_sixteenth_attack_control()
    _check_runner_abort("R0247 reviewer sixteenth-attack control")
    self_attacks = run_self_attack_battery(flag_path=flag_path)
    _check_runner_abort("R0247 self-attack battery")

    guard = _node_guard("R0247 paramguard node tail")
    gate = _node_gate("R0247 paramguard node tail")
    with guard:
        gate.start()
        guard.poll("R0247 paramguard node tail start")
        for step in range(12):
            gate(f"R0247 paramguard node tail step {step}")
            time.sleep(0.1)
        guard.poll("R0247 paramguard node tail end")
        gate.finish()
    tail = _guard_tail(guard, label="R0247 paramguard node tail")
    enforcement = _close_gate(
        gate, tail, label="R0247 paramguard node tail"
    )

    closure = safety_closure_receipt(
        inventory=inventory,
        clamp_controls=clamp_controls,
        call_site_controls=call_site_controls,
        denominator_control=denominator_control,
        sixteenth_attack=sixteenth,
        self_attacks=self_attacks,
    )
    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": PARAMGUARD_SCHEMA,
        "capabilities": [PARAMGUARD_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "closure": closure,
        "enforcement_poll_spacing": enforcement,
        **tail,
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, PARAMGUARD_FILE, body)


# --------------------------------------------------------------------------- #
# node 2 — the precision fix review-0246-01 F showed was cheap
# --------------------------------------------------------------------------- #
def run_truthcos(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0247Error("R0247 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0247 truth cosine precision")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0247 truthcos"
    )
    rows = int(job.get("truth_rows", TRUTH_PROBE_ROWS))

    truth_cos = _readonly_memmap(
        _bound_path(job, "truth_cos", label="truth cosines"),
        label="R0247 truth cosines", shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )
    truth_ids = _readonly_memmap(
        _bound_path(job, "truth_ids", label="truth ids"),
        label="R0247 truth ids", shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )
    probe_rows = _readonly_memmap(
        _bound_path(job, "probe_query_rows", label="probe query rows"),
        label="R0247 probe query rows", shape=(TRUTH_PROBE_ROWS,),
    )
    substrate = _readonly_memmap(
        _bound_path(job, "substrate_array", label="substrate"),
        label="R0247 substrate", shape=(ROWS, DIMENSION),
    )

    io_before = io_counters()
    guard = _node_guard("R0247 truth cosine precision")
    gate = _node_gate("R0247 truth cosine precision")
    with guard:
        gate.start()
        recompute_started = time.monotonic()
        recompute = recompute_truth_cosines_f64(
            substrate=substrate,
            probe_query_rows=probe_rows[:rows],
            truth_ids=truth_ids[:rows],
            abort_check=gate,
        )
        recompute_wall = time.monotonic() - recompute_started
        gate("R0247 recomputed the truth cosines in float64")
        guard.poll("R0247 recomputed the truth cosines in float64")

        cosines = recompute.pop("cosines")
        floor = cosine_noise_floor(
            stored_f32=truth_cos[:rows],
            recomputed_f64=cosines,
            substrate=substrate,
            probe_query_rows=probe_rows[:rows],
            truth_ids=truth_ids[:rows],
            abort_check=gate,
        )
        gate("R0247 measured the cosine noise floor")
        guard.poll("R0247 measured the cosine noise floor")
        tolerance = defensible_tolerance(floor)
        gate("R0247 stated the defensible tolerance")
        gate.finish()
    tail = _guard_tail(guard, label="R0247 truth cosine precision")
    enforcement = _close_gate(gate, tail, label="R0247 truth cosine precision")
    io_after = io_counters()

    saved = atomic_save_new_npy(
        os.path.join(output, TRUTH_COS_F64_FILE), cosines, immutable=True
    )
    del cosines

    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": TRUTHCOS_SCHEMA,
        "capabilities": [TRUTHCOS_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "truth_rows_recomputed": rows,
        "recompute": recompute,
        "truth_cosines_f64": saved,
        "cosine_noise_floor": floor,
        "defensible_tolerance": tolerance,
        "precision_note": PRECISION_NOTE,
        "what_this_replaces": (
            "result-0246 priced the fix as 'a 100M-row GPU job ... it should be "
            "priced before it is promised'. review-0246-01 F showed that prices "
            "the wrong operation: re-deriving the truth IDS needs the exact "
            "search, re-deriving the truth COSINES needs a gather of the sealed "
            "ids. This node ran the gather, on CPU, and it created no CUDA "
            "context."
        ),
        "enforcement_poll_spacing": enforcement,
        **tail,
        "walls": {"recompute_s": recompute_wall},
        "io": {
            "read_bytes": int(io_after["read_bytes"] - io_before["read_bytes"]),
            "write_bytes": int(
                io_after["write_bytes"] - io_before["write_bytes"]
            ),
        },
        "bulk_input_memmap_attestation": _memmap_attestation({
            "truth_cosines": truth_cos,
            "truth_ids": truth_ids,
            "probe_query_rows": probe_rows,
            "substrate": substrate,
        }),
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, TRUTHCOS_FILE, body)


# --------------------------------------------------------------------------- #
# node 3 — the sealed bound, over the whole probe
# --------------------------------------------------------------------------- #
def run_tie(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0247Error("R0247 handler received another queue")
    started = time.monotonic()
    abort_flag = _start_node("R0247 tie-aware sealed bound")
    output = create_fresh_directory(str(job["outputs"][0]), label="R0247 tie")

    truth_cos = _readonly_memmap(
        _bound_path(job, "truth_cos", label="truth cosines"),
        label="R0247 truth cosines", shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )
    truth_ids = _readonly_memmap(
        _bound_path(job, "truth_ids", label="truth ids"),
        label="R0247 truth ids", shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )
    probe_rows = _readonly_memmap(
        _bound_path(job, "probe_query_rows", label="probe query rows"),
        label="R0247 probe query rows", shape=(TRUTH_PROBE_ROWS,),
    )
    substrate = _readonly_memmap(
        _bound_path(job, "substrate_array", label="substrate"),
        label="R0247 substrate", shape=(ROWS, DIMENSION),
    )
    graph_ids = _readonly_memmap(
        _bound_path(job, "graph_ids", label="graph ids"),
        label="R0247 graph ids", shape=(ROWS, GRAPH_K),
    )
    truth_cos_f64 = _readonly_memmap(
        _bound_path(job, "truth_cos_f64", label="float64 truth cosines"),
        label="R0247 float64 truth cosines",
        shape=(TRUTH_PROBE_ROWS, GRAPH_K),
    )

    replication_rows = int(job.get("replication_rows", TIE_PRECISION_ROWS))
    full_rows = int(job.get("full_probe_rows", TIE_FULL_PROBE_ROWS))
    seed = int(job.get("tie_seed", TIE_PRECISION_SEED))

    guard = _node_guard("R0247 tie-aware sealed bound")
    gate = _node_gate("R0247 tie-aware sealed bound")
    with guard:
        gate.start()
        #: A — R0246's exact measurement, at R0246's rows and seed, against the
        #: stored float32 truth. It must reproduce, or the comparison below is
        #: not a comparison.
        replication = tie_aware_precision_profile(
            substrate=substrate, graph_ids=graph_ids,
            probe_query_rows=probe_rows, truth_ids=truth_ids,
            truth_cosines=truth_cos, sample_rows=replication_rows, seed=seed,
            abort_check=gate,
        )
        gate("R0247 replicated R0246's profile against the stored truth")
        guard.poll("R0247 replicated R0246's profile")

        #: B — the same rows, the same seed, against the RECOMPUTED float64
        #: truth. The only difference is the reference, so the delta is what
        #: the precision fix bought.
        against_f64 = tie_aware_precision_profile(
            substrate=substrate, graph_ids=graph_ids,
            probe_query_rows=probe_rows, truth_ids=truth_ids,
            truth_cosines=truth_cos_f64, sample_rows=replication_rows,
            seed=seed, abort_check=gate,
        )
        gate("R0247 measured the same rows against the float64 truth")
        guard.poll("R0247 measured the same rows against the float64 truth")

        #: C — the WHOLE probe against the float64 truth. 7,500,000 decisions
        #: is the entire decision population, so this is the tightest bound the
        #: probe can produce.
        full_started = time.monotonic()
        full = tie_aware_precision_profile(
            substrate=substrate, graph_ids=graph_ids,
            probe_query_rows=probe_rows, truth_ids=truth_ids,
            truth_cosines=truth_cos_f64, sample_rows=full_rows, seed=seed,
            abort_check=gate,
        )
        full_wall = time.monotonic() - full_started
        gate("R0247 measured the whole probe")
        guard.poll("R0247 measured the whole probe")

        sealed = sealed_bound_adjudication(full)
        gate("R0247 sealed the bound adjudication")
        r0246_scale = sealed_bound_adjudication(replication)
        gate("R0247 reproduced R0246's bound at R0246's sample size")
        use_control = tie_use_positive_control()
        gate("R0247 ran the aggregate-only routing control")
        gate.finish()
    tail = _guard_tail(guard, label="R0247 tie-aware sealed bound")
    enforcement = _close_gate(gate, tail, label="R0247 tie-aware sealed bound")

    r0246_eight = list(
        r0246_scale["claims_that_do_not_survive_at_the_bound_names"]
    )
    surviving_now = [
        name for name in r0246_eight
        if name not in sealed["claims_that_do_not_survive_at_the_bound_names"]
    ]
    body = dict(_receipt_envelope(manifest))
    body.update({
        "schema": TIE_SCHEMA,
        "capabilities": [TIE_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "tie_tolerance": float(TIE_TOLERANCE),
        "ledger_size": len(TIE_AWARE_CLAIM_LEDGER),
        "replication_against_the_stored_float32_truth": replication,
        "same_rows_against_the_recomputed_float64_truth": against_f64,
        "whole_probe_against_the_recomputed_float64_truth": full,
        "sealed_bound_adjudication": sealed,
        "at_r0246_sample_size": r0246_scale,
        "the_r0246_non_survivors": r0246_eight,
        "how_many_claims_r0246_retracted": len(r0246_eight),
        "of_those_that_now_survive": surviving_now,
        "of_those_that_still_do_not_survive": [
            name for name in r0246_eight if name in
            sealed["claims_that_do_not_survive_at_the_bound_names"]
        ],
        "point_estimate_adjudication": adjudicate_tie_aware_claims(full),
        "flip_rate_bound": flip_rate_bound(full),
        "aggregate_only_control": use_control,
        "aggregate_only_rule": TIE_AGGREGATE_ONLY_RULE,
        "the_tolerance_was_not_moved": True,
        "decision_population_note": (
            "7,500,000 is the ENTIRE decision population of R0238's truth "
            "probe (500,000 rows x k=15). review-0246-01 H7 asked for a "
            "re-measurement on >= 10^7 decisions; 7.5e6 is the ceiling this "
            "probe has, and tightening the bound further needs a larger truth "
            "probe rather than a larger sample."
        ),
        "enforcement_poll_spacing": enforcement,
        **tail,
        "walls": {"whole_probe_s": full_wall},
        "bulk_input_memmap_attestation": _memmap_attestation({
            "truth_cosines": truth_cos,
            "truth_cosines_f64": truth_cos_f64,
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
    if action == PARAMGUARD_ACTION:
        run_paramguard(active, job)
    elif action == TRUTHCOS_ACTION:
        run_truthcos(active, job)
    elif action == TIE_ACTION:
        run_tie(active, job)
    else:
        raise Round0247Error(f"R{ROUND_ID} does not authorize action {action!r}")


__all__ = [
    "NODE_ANON_BUDGET_BYTES",
    "PARAMGUARD_ACTION",
    "PARAMGUARD_CAPABILITY",
    "PARAMGUARD_FILE",
    "PARAMGUARD_SCHEMA",
    "TIE_ACTION",
    "TIE_CAPABILITY",
    "TIE_FILE",
    "TIE_FULL_PROBE_ROWS",
    "TIE_SCHEMA",
    "TRUTHCOS_ACTION",
    "TRUTHCOS_CAPABILITY",
    "TRUTHCOS_FILE",
    "TRUTHCOS_SCHEMA",
    "TRUTH_COS_F64_FILE",
    "run_job",
    "run_paramguard",
    "run_tie",
    "run_truthcos",
]
