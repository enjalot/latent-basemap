"""Execute R0250 — measure the trainer's loops, and take the gate to n = 16.

Six nodes in one queue.

**A. the blocker, measured.**

* `trainloop_0250` (GPU) — the training path's inner loops, instrumented the way
  R0246/R0249 instrumented the qualification path. Two arms on one short rung:
  `as_shipped`, which measures the widest abort-read gap a training node actually
  has today (nothing inside `ParametricUMAP.fit` reads the cooperative abort flag,
  so it is the whole `fit()` wall), and `per_batch_poll`, which installs one read
  per training batch and measures what the gap would become. Both are scored by a
  real `AbortPollGate` against `registered_value("r0246_max_poll_spacing_s")`, with
  the allocation slope measured from the live host watchdog's own anonymous trace,
  and both are projected to a `~10 h` node with the projection labelled as one.
* `blocksize_0250` (CPU) — the standing `truthcos_0247` item. R0247's gather block
  is a keyword default of `2_000` that left a `0.906`-of-ceiling gap and is not in
  the registry. The node walks the same released gather over the same sealed probe
  at four block sizes, measures the per-row cost across cold and warm page-cache
  states, and applies a pre-registered three-part rule to decide whether a
  registered block-size ceiling is the right instrument. It registers nothing and
  edits no guard module.

**B. the gate at n >= 16.**

* three GPU trains (`seeds 55/56/57`) under R0217's treatment with the seed as the
  only free variable, structurally identical to R0221's and R0230's;
* `score_minilm_mixed_2m_panel_n16` (GPU) — the three new maps scored on R0218's
  frozen panel, the thirteen existing cells read from R0230's sealed pooled table
  and not rescored, and the high-D reference proved byte-identical on the same five
  components. If it is not, the node raises and the round stops: sixteen cells that
  are not poolable is the finding, not an obstacle;
* `register_calibrated_robust_floors_n16` (CPU) — R0234's calibrated method applied
  at `n = 16`, the multiplier derived rather than carried, and the JOINT criteria
  applied: R0234's floors together with the retained R0225/R0231 ones.

Every registered check is IMPORTED and called. The memory guard, the watchdog, the
sealed-graph reader, the substrate opener and the accounting closure come from
`round0230_nodes`; the calibration, invariance, attainability and scoring come from
`round0234_calibrated_floors`; the poll gate, the host watchdog and the three
`require_*` gates come from R0244-R0247. Nothing in this module signals any
process, hands cuVS anything, or wraps a child in a timeout.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import resource
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap import round0234_calibration as calibration
from basemap.round0217_minilm_2m_pipeline import (
    MiniLMHostFp32EndpointArray,
    MiniLMMixedTrainingInput,
)
from basemap.round0217_minilm_2m_seed_family import (
    WARMUP_SUCCESSFUL_UPDATES,
    train_config as r0217_train_config,
)
from basemap.round0242_locality import io_counters, json_scrub
from basemap.round0238_rung5 import json_safe
from basemap.round0245_guard import (
    EnforcedHostWatchdog,
    require_enforceable_abort_flag,
)
from basemap.round0246_guard import (
    AbortPollGate,
    Round0246Error,
    measured_slope_from_trace,
    require_abort_flag_landed,
    require_enforcement_evidence,
    require_live_sampler,
)
from basemap.round0247_registry import (
    Round0247Error,
    registered_bounds,
    registry_fingerprint,
    verify_registry,
)
from basemap.round0250_blocksize import (
    ADAPTIVE_TARGET_FRACTION,
    BLOCKSIZE_CAPABILITY,
    BLOCKSIZE_NOTE,
    BLOCKSIZE_SCHEMA,
    BLOCK_SIZES_PROBED,
    PER_ROW_COST_STABILITY_LIMIT,
    Round0250BlockSizeError,
    adaptive_gather_probe,
    measure_block,
    observed_default_block,
    resolve as resolve_block_size,
)
from basemap.round0250_gate_n16 import (
    CANDIDATES,
    CANDIDATE_CLUSTER_COUNTS,
    CANDIDATE_ORDER,
    CANDIDATE_SEEDS,
    COVERAGE_TARGET,
    COVERAGE_TOLERANCE,
    CUVS_FAMILY_SEEDS,
    DENSITY_V2_DEFECT,
    DESCRIPTIVE_METRICS,
    EXACT_FAMILY_SEEDS,
    GATED_METRICS,
    GATE_CAPABILITY,
    GATE_SCHEMA,
    IDENTITY_BOUND_AT_N,
    JOINT_CRITERIA_RULE,
    METRICS,
    N_EXACT,
    N_HELD_OUT,
    POWER_MATERIALITY,
    POWER_SELECTION_ALTERNATIVE,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
    REQUIRED_INVARIANCE_DEPTH,
    RETAINED_FAMILY_SOURCES,
    Round0250GateError,
    SELECTION_RULE,
    THIS_FAMILY,
    TOLERANCE_CONFIDENCE,
    TOLERANCE_CONTENT,
    attainability,
    band_at,
    centre_and_scale,
    degenerate_witness,
    falsifiability_statement,
    floor_at,
    identity_bound,
    injection_ladder,
    joint_criteria_from_sealed,
    positive_scale_witness,
    score_joint,
    score_population,
    verdict_changes,
)
from basemap.round0250_panel_n16 import (
    CENTROID_KS,
    CORPUS_SLUGS,
    DENSITY_V2_STATUS,
    PANEL_CAPABILITY,
    PANEL_CAPABILITY_N13,
    PANEL_CAPABILITY_N16,
    PANEL_METRICS,
    PANEL_SCHEMA_N13,
    PANEL_SCHEMA_N16,
    POOLED_CELL_SOURCES,
    REFERENCE_MISMATCH_MESSAGE,
    Round0250PanelError,
    assert_hi_d_agreement,
    assert_reference_identity,
    corpus_ffr_view,
    descriptive_family_summary,
    panel_execution_ok,
    panel_metric_view,
    pool_sixteen_cells,
    raw_purity_ratios,
)
from basemap.round0250_seed_extension_n16 import (
    assert_reconstructs_r0217_template,
    BATCH_SIZE,
    CAPABILITY_TEMPLATE,
    DEVICE_BUDGET_BYTES,
    DIMENSION,
    FULL_TRANSFORM_BATCH,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    HOST_ANON_BUDGET_BYTES,
    HOST_RSS_LIMIT_GIB,
    IDENTITY_BOUND_AT_N16,
    MEMORY_POLICY,
    POOLED_SEEDS,
    POSITIVE_ROWS_PER_UPDATE,
    PRODUCTION_CONFIG_SCHEMA,
    R0217_SEED_INVARIANT_SHA256,
    R0230_POOLED_SEEDS,
    REGISTERED_SUCCESSFUL_UPDATES,
    ROUND_ID,
    ROWS,
    Round0250Error,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    STANDING_MINIMUM_N,
    TEMPLATE_SEED,
    TRAIN_SCHEMA,
    capability_for_seed,
    masked_config_bytes,
    performance_windows,
    predict_cell_footprint,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config,
    validate_full_population_map,
    validate_registered_dose,
)
from basemap.round0250_trainer_loops import (
    ARMS,
    ARM_AS_SHIPPED,
    ARM_PER_BATCH,
    NOT_A_FAMILY_CELL,
    PER_BATCH_HOOK,
    PROJECTION_TARGET_HOURS,
    PROJECTION_TARGET_SECONDS,
    PerBatchAbortPoll,
    Round0250TrainerLoopError,
    SHORT_HORIZON_UPDATES,
    TRAINER_LOOP_CAPABILITY,
    TRAINER_LOOP_NOTE,
    TRAINER_LOOP_SCHEMA,
    ceiling_report,
    project_to_hours,
    short_rung_config,
)
from basemap import round0113_prompt_contrast as _prompt_contract_alias  # noqa: F401
from experiments import round0218_nodes
from experiments.round0230_nodes import (
    CellWatchdog,
    _open_substrate,
    _sealed_graph,
    _weighted_rejection_accounting_mismatch,
)
from experiments.round0113_nodes import _new_model
from experiments.round0238_nodes import _check_runner_abort
from experiments.round0241_nodes import _readonly_memmap
from experiments.round0242_nodes import _memmap_attestation
from basemap.round0253_stop_hooks import install_stop_hooks


TRAINLOOP_ACTION = "trainloop_0250"
BLOCKSIZE_ACTION = "blocksize_0250"
TRAIN_ACTION = "train_minilm_mixed_2m_seed_extension_n16"
PANEL_ACTION = "score_minilm_mixed_2m_panel_n16"
GATE_ACTION = "register_calibrated_robust_floors_n16"

#: The instrumentation node's own anonymous budget, matching R0247's nodes. Its
#: headroom comes from the registry, never from a keyword.
NODE_ANON_BUDGET_BYTES = 16 * (1 << 30)

SAFETY_NOTE = (
    "no node in this module signals any process, starts a child process, hands "
    "cuVS anything, or wraps a subprocess in a timeout. Every bulk input is a "
    "read-only np.memmap. The trainer instrumentation installs one abort read per "
    "batch through a context manager that restores the release attribute in a "
    "finally; nothing in `basemap/pumap/` is edited. Every bound the poll gate and "
    "the host guard apply is read from the R0247 registry at the comparison site."
)


# --------------------------------------------------------------------------- #
# shared node scaffolding, imported rather than re-typed where it exists
# --------------------------------------------------------------------------- #


def _read_json(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise Round0250Error(f"R0250 {label} is absent at {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _bound_path(job: Mapping[str, Any], key: str, *, label: str) -> str:
    entry = job.get(key)
    if not isinstance(entry, Mapping):
        raise Round0250Error(f"R0250 job does not bind {label} ({key})")
    path = str(entry.get("canonical_path") or "")
    if not path or not os.path.exists(path):
        raise Round0250Error(f"R0250 {label} is absent at {path!r}")
    declared = int(entry.get("bytes", -1))
    actual = os.path.getsize(path)
    if declared >= 0 and declared != actual:
        raise Round0250Error(
            f"R0250 {label} is {actual} bytes, the manifest declared {declared}"
        )
    return path


def _start_node(label: str) -> dict[str, Any]:
    """Verify the registry and the cooperative abort path before anything else."""
    verify_registry(label=label)
    return require_enforceable_abort_flag(label=label)


def _node_guard(label: str, *, interval_s: float = 0.05) -> EnforcedHostWatchdog:
    return EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        interval_s=float(interval_s),
        label=label,
    )


def _node_gate(label: str, *, training_performed: bool) -> AbortPollGate:
    """The registered headroom, the registered ceiling, the registered reader.

    `inner` is `experiments.round0238_nodes._check_runner_abort`, one of the four
    names in `REGISTERED_ABORT_READERS`; `clock` is left unset so the gate uses
    `time.monotonic`; `replay` is left False. Those three together are what
    `require_enforcement_evidence` demands before a verdict may be sealed as
    evidence, and no keyword here can weaken a registered bound.
    """
    return AbortPollGate(
        inner=_check_runner_abort,
        headroom_bytes=int(
            registered_bounds(["max_declared_headroom_bytes"])[
                "registered_max_declared_headroom_bytes"
            ]
        ),
        label=label,
        training_performed=bool(training_performed),
    )


def _guard_tail(watchdog: EnforcedHostWatchdog, *, label: str) -> dict[str, Any]:
    receipt = watchdog.receipt()
    return {
        "host_watchdog": receipt,
        "sampler_liveness": require_live_sampler(receipt, label=label),
        "abort_flag_landing": require_abort_flag_landed(receipt, label=label),
    }


def _guard_tail_reported(
    watchdog: EnforcedHostWatchdog, *, label: str
) -> dict[str, Any]:
    """The same tail, published even when a tail gate refuses.

    The trainer-loop node exists to find out whether the training path holds the
    registered bounds. A tail gate that refuses IS the answer, so it is recorded
    with its exact error rather than aborting the node before the numbers are
    sealed. The gate is still called, and its refusal is still published.
    """
    receipt = watchdog.receipt()
    tail: dict[str, Any] = {"host_watchdog": receipt}
    for key, gate in (
        ("sampler_liveness", require_live_sampler),
        ("abort_flag_landing", require_abort_flag_landed),
    ):
        try:
            tail[key] = gate(receipt, label=label)
        except (Round0246Error, Round0247Error) as error:
            tail[key] = {
                "holds": False,
                "raised": f"{type(error).__name__}: {error}",
                "reported_rather_than_aborting": True,
            }
    return tail


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


def _score_gate_without_raising(
    gate: AbortPollGate, tail: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """The same score, published even when the arm FAILS the ceiling.

    `as_shipped` is expected to fail, and "the trainer does not satisfy the
    ceiling" is the round's finding rather than its error. `verdict()` never
    raises, so the numbers are always published; `require()` is then called inside
    a try so the exact failing arm names are published too.
    """
    slope = measured_slope_from_trace(
        tail["host_watchdog"]["anonymous_trace_by_second"]
    )
    verdict = gate.verdict(measured_slope_bytes_per_s=slope)
    outcome: dict[str, Any] = {
        "measured_slope_bytes_per_s": slope,
        "require_raised": False,
        "require_error": None,
    }
    try:
        required = gate.require(measured_slope_bytes_per_s=slope)
    except (Round0246Error, Round0247Error) as error:
        outcome["require_raised"] = True
        outcome["require_error"] = f"{type(error).__name__}: {error}"
        required = dict(verdict)
        required["failures"] = [
            arm
            for arm in (
                "meets_the_registered_ceiling",
                "meets_the_required_poll_count",
                "measured_spacing_is_non_zero",
            )
            if verdict.get(arm) is False
        ]
        required["holds"] = not required["failures"]
    try:
        required["enforcement_evidence"] = require_enforcement_evidence(
            required, label=label
        )
    except (Round0246Error, Round0247Error) as error:
        required["enforcement_evidence"] = {
            "holds": False,
            "error": f"{type(error).__name__}: {error}",
        }
    required["outcome"] = outcome
    return required


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "registry_fingerprint": registry_fingerprint(),
        "safety_note": SAFETY_NOTE,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": False,
    }


def _seal(output: str, name: str, body: Mapping[str, Any]) -> None:
    atomic_write_new_json(
        os.path.join(output, name),
        prompt_contract.seal(json_safe(json_scrub(dict(body)))),
        immutable=True,
    )


# --------------------------------------------------------------------------- #
# A1. the trainer's own loops
# --------------------------------------------------------------------------- #


def _short_rung_template(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """R0217's canonical config for the sealed substrate, then the short rung."""
    graph = _sealed_graph(job)
    source, substrate_signature = _open_substrate(graph)
    template, _sha = r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        substrate_signature=substrate_signature,
        graph_edges=graph["directed_edges"],
        rows=ROWS,
    )
    return template, {
        "graph": graph,
        "source": source,
        "substrate_signature": substrate_signature,
    }


def _run_trainer_arm(
    *,
    arm: str,
    config: Mapping[str, Any],
    bound: Mapping[str, Any],
    updates: int,
    gate: AbortPollGate,
) -> dict[str, Any]:
    """One short fit, instrumented according to the arm, with the gate live."""
    import torch

    seed = int(config["seed"])
    dataset = MiniLMHostFp32EndpointArray(
        bound["source"],
        source_signature=bound["substrate_signature"],
        buffer_rows=BATCH_SIZE,
    )
    wrapper = MiniLMMixedTrainingInput(dataset, bound["graph"], seed=seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = _new_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = performance_windows(updates)
    model._abort_on_first_nonfinite = True

    installer = PerBatchAbortPoll(poll=gate, label=f"R0250 {arm} batch")
    started = time.monotonic()
    if arm == ARM_PER_BATCH:
        with installer:
            gate(f"R0250 {arm} fit() entered")
            model.fit(
                wrapper,
                low_memory=True,
                verbose=False,
                n_processes=6,
                random_state=seed,
                resample_negatives=False,
                precomputed_edges_path=bound["graph"]["signature"]["canonical_path"],
                use_wandb=False,
            )
        gate(f"R0250 {arm} fit() returned")
    else:
        gate(f"R0250 {arm} fit() entered")
        model.fit(
            wrapper,
            low_memory=True,
            verbose=False,
            n_processes=6,
            random_state=seed,
            resample_negatives=False,
            precomputed_edges_path=bound["graph"]["signature"]["canonical_path"],
            use_wandb=False,
        )
        gate(f"R0250 {arm} fit() returned")
    wall = time.monotonic() - started

    accounting = dict(model._train_stats)
    runtime = wrapper.runtime_stamp()
    exact = {
        "optimizer_steps_succeeded": updates,
        "positive_lr_optimizer_steps": updates,
        "attempted_batches": updates,
        "finite_loss_batches": updates,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "n_pos_edges": SEALED_DIRECTED_EDGES,
    }
    mismatches = {
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    }
    if mismatches:
        raise Round0250TrainerLoopError(
            f"R0250 {arm} short rung did not close its accounting: {mismatches}"
        )
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (updates - WARMUP_SUCCESSFUL_UPDATES) / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
    }
    del model, wrapper, dataset
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "arm": arm,
        "short_horizon_updates": updates,
        "fit_wall_s": wall,
        "setup_seconds": profiler.get("setup_seconds"),
        "steady_updates_per_s": rate,
        "performance_profile": profiler,
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "memory": memory,
        "instrumentation": installer.receipt(),
        "is_a_family_cell": False,
        "not_a_family_cell_note": NOT_A_FAMILY_CELL,
    }


def run_trainloop(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0250TrainerLoopError("R0250 trainloop handler received another queue")
    started = time.monotonic()
    label = "R0250 trainer loop instrumentation"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0250 trainloop")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0250TrainerLoopError("R0250 trainer-loop measurement requires CUDA")

    updates = int(job.get("short_horizon_updates", SHORT_HORIZON_UPDATES))
    template, bound = _short_rung_template(job)
    config, config_sha, rung_identity = short_rung_config(template, updates=updates)
    atomic_write_new_json(
        os.path.join(output, "short-rung-config.json"),
        {
            "round_id": ROUND_ID,
            "capability": TRAINER_LOOP_CAPABILITY,
            "config": config,
            "config_sha256": config_sha,
            "identity": rung_identity,
        },
        immutable=True,
    )

    io_before = io_counters()
    arms: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    projections: dict[str, Any] = {}
    ceilings: dict[str, Any] = {}
    tails: dict[str, Any] = {}
    for arm in ARMS:
        guard = _node_guard(f"{label} {arm}")
        gate = _node_gate(f"{label} {arm}", training_performed=True)
        with guard:
            gate.start()
            arms[arm] = _run_trainer_arm(
                arm=arm, config=config, bound=bound, updates=updates, gate=gate
            )
            guard.poll(f"R0250 {arm} fit complete")
            gate.finish(f"R0250 {arm} stage end")
        tail = _guard_tail_reported(guard, label=f"{label} {arm}")
        scored = _score_gate_without_raising(gate, tail, label=f"{label} {arm}")
        gates[arm] = scored
        tails[arm] = tail
        ceilings[arm] = ceiling_report(
            arm=arm,
            widest_gap_s=float(scored["max_gap_between_enforcement_polls_s"]),
            stage_wall_s=float(scored["stage_wall_s"]),
            polls=int(scored["enforcement_polls"]),
        )
        projections[arm] = project_to_hours(
            arm=arm,
            widest_gap_s=float(scored["max_gap_between_enforcement_polls_s"]),
            stage_wall_s=float(scored["stage_wall_s"]),
        )
    io_after = io_counters()

    verdict = {
        "the_trainer_as_shipped_meets_the_registered_ceiling": bool(
            ceilings[ARM_AS_SHIPPED]["meets_the_registered_ceiling"]
        ),
        "a_per_batch_poll_meets_the_registered_ceiling": bool(
            ceilings[ARM_PER_BATCH]["meets_the_registered_ceiling"]
        ),
        "as_shipped_gap_over_the_ceiling": ceilings[ARM_AS_SHIPPED][
            "gap_over_the_registered_ceiling"
        ],
        "per_batch_gap_over_the_ceiling": ceilings[ARM_PER_BATCH][
            "gap_over_the_registered_ceiling"
        ],
        "widest_gap_sits_at": {
            arm: gates[arm]["widest_gap_before"] for arm in ARMS
        },
        "measured_allocation_slope_bytes_per_s": {
            arm: gates[arm]["outcome"]["measured_slope_bytes_per_s"] for arm in ARMS
        },
        "binding_slope_floor_bytes_per_s": gates[ARM_AS_SHIPPED][
            "binding_slope_floor_bytes_per_s"
        ],
        # Measured, not asserted: the gate counts calls to ITSELF, and the
        # `as_shipped` arm makes exactly the two the node makes on either side of
        # fit(). Any abort read inside fit() would have raised that count.
        "abort_reads_made_by_the_as_shipped_arm": int(
            gates[ARM_AS_SHIPPED]["enforcement_polls"]
        ),
        "abort_reads_the_node_itself_makes": 2,
        "no_abort_read_exists_inside_fit": (
            int(gates[ARM_AS_SHIPPED]["enforcement_polls"]) == 2
        ),
        "as_shipped_gap_equals_its_own_stage_wall_to_within": abs(
            float(gates[ARM_AS_SHIPPED]["max_gap_between_enforcement_polls_s"])
            - float(gates[ARM_AS_SHIPPED]["stage_wall_s"])
        ),
        "evidence_for_that": (
            "grep over basemap/pumap/ for runner_abort_reason, _check_runner_abort "
            "and ROUNDRUN_ABORT_FLAG returns nothing; the two numbers above are "
            "the runtime confirmation"
        ),
        "projection_target_hours": PROJECTION_TARGET_HOURS,
        "projections_are_projections_not_measurements": all(
            not bool(item["is_a_measurement_at_the_target_wall"])
            for item in projections.values()
        ),
    }

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": TRAINER_LOOP_SCHEMA,
        "capability": TRAINER_LOOP_CAPABILITY,
        "capabilities": [TRAINER_LOOP_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "short_rung": rung_identity,
        "short_rung_config_sha256": config_sha,
        "per_batch_hook": dict(PER_BATCH_HOOK),
        "arms": arms,
        "enforcement_poll_spacing": gates,
        "ceiling_reports": ceilings,
        "projections": projections,
        "guard_tails": tails,
        "verdict": verdict,
        "trainer_loop_note": TRAINER_LOOP_NOTE,
        "training_performed": True,
        "map_decision_made": False,
        "gate_registered": False,
        "map_quality_claimed": False,
        "published_a_map": False,
        "io": {
            "read_bytes": int(io_after["read_bytes"] - io_before["read_bytes"]),
            "write_bytes": int(io_after["write_bytes"] - io_before["write_bytes"]),
        },
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 ** 2),
        "performance": {
            "total_wall_s": time.monotonic() - started,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        },
    })
    _seal(output, "trainer-loop-poll-spacing.json", body)
    print(json.dumps({
        "capability": TRAINER_LOOP_CAPABILITY,
        "as_shipped_gap_over_ceiling": verdict["as_shipped_gap_over_the_ceiling"],
        "per_batch_gap_over_ceiling": verdict["per_batch_gap_over_the_ceiling"],
    }))


# --------------------------------------------------------------------------- #
# A2. the truthcos block size
# --------------------------------------------------------------------------- #


def run_blocksize(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0250BlockSizeError("R0250 blocksize handler received another queue")
    started = time.monotonic()
    label = "R0250 truthcos block-size resolution"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0250 blocksize")

    probe_rows = int(job["truth_probe_rows"])
    arm_rows = int(job["arm_rows"])
    truth_ids = _readonly_memmap(
        _bound_path(job, "truth_ids", label="truth ids"),
        label="R0250 truth ids",
        shape=(probe_rows, int(job["graph_k"])),
    )
    query_rows = _readonly_memmap(
        _bound_path(job, "probe_query_rows", label="probe query rows"),
        label="R0250 probe query rows",
        shape=(probe_rows,),
    )
    substrate = _readonly_memmap(
        _bound_path(job, "substrate_array", label="substrate"),
        label="R0250 substrate",
        shape=(int(job["substrate_rows"]), int(job["substrate_dimension"])),
    )
    r0247 = prompt_contract.read_sealed(
        _bound_path(job, "r0247_truthcos", label="R0247 sealed truthcos receipt"),
        label="R0247 sealed truthcos receipt",
    )
    published_gap = float(
        r0247["enforcement_poll_spacing"]["max_gap_between_enforcement_polls_s"]
    )
    default_block = observed_default_block()

    io_before = io_counters()
    guard = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    cold_arms: list[dict[str, Any]] = []
    with guard:
        gate.start()
        # Cold arms: disjoint row ranges, so no arm reads pages another warmed.
        cursor = 0
        for block in BLOCK_SIZES_PROBED:
            arm = measure_block(
                substrate=substrate,
                truth_ids=truth_ids,
                probe_query_rows=query_rows,
                start=cursor,
                rows=arm_rows,
                block=block,
                abort_check=gate,
                label=f"{label} cold",
            )
            arm["page_cache_state"] = "cold: a row range no earlier arm touched"
            cold_arms.append(arm)
            cursor += arm_rows
            guard.poll(f"R0250 cold block {block}")
        # Warm arm: the FIRST cold range again, at R0247's own block.
        warm = measure_block(
            substrate=substrate,
            truth_ids=truth_ids,
            probe_query_rows=query_rows,
            start=0,
            rows=arm_rows,
            block=default_block,
            abort_check=gate,
            label=f"{label} warm",
        )
        warm["page_cache_state"] = (
            "warm: the first cold arm's row range, read a second time"
        )
        guard.poll("R0250 warm arm")
        adaptive = adaptive_gather_probe(
            substrate=substrate,
            truth_ids=truth_ids,
            probe_query_rows=query_rows,
            start=cursor,
            rows=arm_rows,
            initial_block=max(BLOCK_SIZES_PROBED),
            abort_check=gate,
            label=f"{label} adaptive",
        )
        gate("R0250 adaptive gather probe complete")
        gate.finish(f"{label} stage end")
    # The blocksize node exists to find out WHETHER a block breaches the
    # ceiling, so a breach is its finding and not its error. Attempt 1 of this
    # round used the raising `_close_gate` here and died on
    # `meets_the_registered_ceiling` at the warm block=2,000 arm, destroying the
    # measurement it had just made. The gate is unchanged and still refuses; the
    # refusal is now recorded as data. (The trainer-loop node already did this;
    # this node should have from the start.)
    tail = _guard_tail_reported(guard, label=label)
    enforcement = _score_gate_without_raising(gate, tail, label=label)
    io_after = io_counters()

    arms = [*cold_arms, warm]
    resolution = resolve_block_size(
        arms=arms,
        declared_default_block=default_block,
        r0247_widest_gap_s=published_gap,
        r0247_block_rows=default_block,
        adaptive=adaptive,
    )

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": BLOCKSIZE_SCHEMA,
        "capability": BLOCKSIZE_CAPABILITY,
        "capabilities": [BLOCKSIZE_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "r0247_sealed_receipt": expected_input_signature(
            _bound_path(job, "r0247_truthcos", label="R0247 sealed truthcos receipt")
        ),
        "r0247_published_widest_gap_s": published_gap,
        "r0247_declared_default_block_rows": default_block,
        "block_sizes_probed": list(BLOCK_SIZES_PROBED),
        "arm_rows": arm_rows,
        "per_row_cost_stability_limit": PER_ROW_COST_STABILITY_LIMIT,
        "adaptive_target_fraction": ADAPTIVE_TARGET_FRACTION,
        "cold_arms": cold_arms,
        "warm_arm": warm,
        "adaptive_probe": adaptive,
        "resolution": resolution,
        "enforcement_poll_spacing": enforcement,
        "the_probe_itself_met_the_registered_ceiling": bool(
            enforcement.get("meets_the_registered_ceiling")
        ),
        "the_probe_widest_gap_s": enforcement.get(
            "max_gap_between_enforcement_polls_s"
        ),
        "the_probe_widest_gap_sits_after": enforcement.get("widest_gap_before"),
        "probe_gate_note": (
            "this probe walks the released gather at four block sizes, so a "
            "block wide enough to breach the ceiling makes the probe's own gate "
            "refuse. That refusal is the measurement, recorded here rather than "
            "raised: the gate is unchanged and its verdict stands."
        ),
        **tail,
        "note": BLOCKSIZE_NOTE,
        "training_performed": False,
        "cuda_context_created": False,
        "registry_mutated": False,
        "guard_module_edited": False,
        "io": {
            "read_bytes": int(io_after["read_bytes"] - io_before["read_bytes"]),
            "write_bytes": int(io_after["write_bytes"] - io_before["write_bytes"]),
        },
        "bulk_input_memmap_attestation": _memmap_attestation({
            "truth_ids": truth_ids,
            "probe_query_rows": query_rows,
            "substrate": substrate,
        }),
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 ** 2),
        "performance": {"total_wall_s": time.monotonic() - started},
    })
    _seal(output, "truthcos-block-size-resolution.json", body)
    print(json.dumps({
        "capability": BLOCKSIZE_CAPABILITY,
        "resolution": resolution["resolution"],
        "per_row_cost_spread": resolution["per_row_cost_spread"],
    }))


# --------------------------------------------------------------------------- #
# B1. the three train nodes
# --------------------------------------------------------------------------- #


def _seed(job: Mapping[str, Any]) -> int:
    seed = job.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0250Error(f"R0250 job seed {seed!r} is not a registered cell")
    if str(job.get("capability") or "") != capability_for_seed(seed):
        raise Round0250Error("R0250 job capability does not match its seed")
    return int(seed)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0253 round0250_nodes.run_train")
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0250Error("R0250 train handler received another queue")
    seed = _seed(job)

    prediction = predict_cell_footprint(seed)
    declared = job.get("memory_prediction")
    if declared is not None and dict(declared) != prediction:
        raise Round0250Error(
            "R0250 cell prediction differs from the one sealed at prepare time"
        )
    if prediction["refused_a_priori"]:
        raise Round0250Error(
            f"R0250 seed-{seed} refused a priori: predicted "
            f"{prediction['predicted_peak_device_bytes']} device bytes and "
            f"{prediction['predicted_peak_host_anonymous_bytes']} host anonymous "
            f"bytes against budgets {DEVICE_BUDGET_BYTES} / {HOST_ANON_BUDGET_BYTES}"
        )

    graph = _sealed_graph(job)
    edges = graph["directed_edges"]
    updates = successful_updates_for_edges(edges)
    dose = validate_registered_dose(updates=updates, edge_count=edges)
    declared_bound = job.get("registered_dose_bound")
    if declared_bound is not None and updates > int(declared_bound):
        raise Round0250Error(
            "R0250 derived update horizon exceeds the registered round bound"
        )
    source, substrate_signature = _open_substrate(graph)
    config, config_sha = train_config(
        seed=seed,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        substrate_signature=substrate_signature,
        graph_edges=edges,
        rows=ROWS,
    )
    declared_invariant = str(job.get("family_seed_invariant_sha256") or "")
    observed_invariant = seed_invariant_sha256(config)
    masked = masked_config_bytes(config)
    if (
        not declared_invariant
        or observed_invariant != declared_invariant
        or observed_invariant != R0217_SEED_INVARIANT_SHA256
    ):
        raise Round0250Error(
            "R0250 cell config is not R0217's treatment outside the seed: "
            f"{observed_invariant} != {declared_invariant} / "
            f"{R0217_SEED_INVARIANT_SHA256}"
        )
    if int(config["optimizer"]["successful_positive_lr_updates"]) != updates:
        raise Round0250Error("R0250 horizon did not reach the train config")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0250 train output")
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": PRODUCTION_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "treatment_config_round_id": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "seed": seed,
            "capability": capability_for_seed(seed),
            "seed_invariant_sha256": observed_invariant,
            "masked_config_bytes": len(masked),
            "config": config,
            "config_sha256": config_sha,
        },
        immutable=True,
    )

    dataset = MiniLMHostFp32EndpointArray(
        source, source_signature=substrate_signature, buffer_rows=BATCH_SIZE
    )
    wrapper = MiniLMMixedTrainingInput(dataset, graph, seed=seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = _new_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = performance_windows(updates)
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")

    watchdog = CellWatchdog()
    watchdog.start()
    started = time.monotonic()
    try:
        model.fit(
            wrapper,
            low_memory=True,
            verbose=False,
            n_processes=6,
            random_state=seed,
            resample_negatives=False,
            precomputed_edges_path=graph["signature"]["canonical_path"],
            use_wandb=False,
        )
    except KeyboardInterrupt as error:
        watchdog_state = watchdog.stop()
        raise Round0250Error(
            f"R0250 seed-{seed} aborted by the memory watchdog: "
            f"{watchdog_state.get('trip_reason')!r}"
        ) from error
    wall = time.monotonic() - started

    accounting = dict(model._train_stats)
    runtime = wrapper.runtime_stamp()
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_stamp.items()
        if runtime.get(key) != value
    }
    exact = {
        "lr_horizon": updates,
        "positive_lr_optimizer_steps": updates,
        "scheduler_steps": updates,
        "attempted_batches": updates,
        "finite_loss_batches": updates,
        "optimizer_steps_attempted": updates,
        "optimizer_steps_succeeded": updates,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": edges,
    }
    mismatches.update({
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    })
    expected_rows = updates * BATCH_SIZE
    producer_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        int(runtime["source_rows_gathered"]) != expected_rows
        or int(runtime["destination_rows_gathered"]) != expected_rows
        or int(runtime["host_prefetch_consumer_batches"]) != updates
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows,
            "runtime": runtime,
        }
    weighted = _weighted_rejection_accounting_mismatch(
        runtime, producer_delta=producer_delta, updates=updates
    )
    if weighted is not None:
        mismatches["weighted_rejection_accounting"] = weighted
    if mismatches:
        watchdog.stop()
        raise Round0250Error(f"R0250 train accounting failed: {mismatches}")
    prompt_contract.synchronize_runtime_counters(accounting, runtime)
    accounting["pipeline_runtime"] = dict(runtime)

    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (updates - WARMUP_SUCCESSFUL_UPDATES) / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    if profiler.get("aborted") is not False or rate < config["execution"][
        "minimum_train_upd_s"
    ]:
        watchdog.stop()
        raise Round0250Error("R0250 train performance admission failed")

    model_path = os.path.join(output, "model.pt")
    from basemap.output_safety import atomic_build_new_file

    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    memory = {
        "device_total_bytes": int(total_bytes),
        "post_train_free_bytes": int(free_bytes),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
    }
    del model, wrapper, dataset
    torch.cuda.empty_cache()
    gc.collect()

    from basemap.pumap.parametric_umap import ParametricUMAP

    reloaded = ParametricUMAP.load(model_path, device="cuda")
    coordinates = np.asarray(
        reloaded.transform(source, batch_size=FULL_TRANSFORM_BATCH), dtype=np.float32
    )
    published = validate_full_population_map(coordinates)
    published["model"] = expected_input_signature(model_path)
    del reloaded, coordinates
    torch.cuda.empty_cache()
    gc.collect()

    watchdog_state = watchdog.stop()
    if watchdog_state["tripped"]:
        raise Round0250Error(
            f"R0250 seed-{seed} watchdog tripped: {watchdog_state['trip_reason']!r}"
        )
    if int(memory["peak_reserved_bytes"]) > DEVICE_BUDGET_BYTES:
        raise Round0250Error(
            f"R0250 seed-{seed} peak reserved device bytes "
            f"{memory['peak_reserved_bytes']} exceed the {DEVICE_BUDGET_BYTES} budget"
        )
    if int(watchdog_state["peak_anonymous_bytes"]) > HOST_ANON_BUDGET_BYTES:
        raise Round0250Error(
            f"R0250 seed-{seed} peak anonymous bytes "
            f"{watchdog_state['peak_anonymous_bytes']} exceed the "
            f"{HOST_ANON_BUDGET_BYTES} budget"
        )
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0250Error(
            f"R0250 train peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    memory["peak_host_rss_gib"] = peak_rss_gib
    memory["peak_host_anonymous_bytes"] = int(watchdog_state["peak_anonymous_bytes"])

    receipt = prompt_contract.seal({
        "schema": TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "treatment_config_round_id": "0217",
        "treatment_template_seed": TEMPLATE_SEED,
        "pooled_seed_family": list(POOLED_SEEDS),
        "standing_minimum_n": STANDING_MINIMUM_N,
        "capability": capability_for_seed(seed),
        "capabilities": [capability_for_seed(seed)],
        "training_seed": seed,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "seed_invariant_sha256": observed_invariant,
        "masked_config_bytes": len(masked),
        "r0217_published_seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "model": published["model"],
        "substrate": substrate_signature,
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "graph_capability": GRAPH_CAPABILITY,
        "rows": ROWS,
        "dimension": DIMENSION,
        "directed_edges": edges,
        "dose_registration": dose,
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "requested_positive_draws_per_edge": float(
            config["execution"]["target_positive_draws_per_edge"]
        ),
        "consumed_positive_draws": int(updates * POSITIVE_ROWS_PER_UPDATE),
        "consumed_positive_draws_per_edge": float(
            updates * POSITIVE_ROWS_PER_UPDATE / edges
        ),
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_s": wall,
        "published_map_check": published,
        "memory_prediction": prediction,
        "memory_watchdog": watchdog_state,
        "memory_policy": MEMORY_POLICY,
        # Every entry is the expression that decided it, not a literal. The node
        # has already raised on each of these, so they read True here -- but the
        # value is produced by a comparison that could have gone the other way,
        # which is what `vacuouscheck`'s literal-check rule exists to require.
        "train_checks": {
            "exact_update_closure": len(mismatches) == 0
            and int(accounting["optimizer_steps_succeeded"]) == updates,
            "zero_numerical_skips": (
                int(accounting["amp_overflow_skips"]) == 0
                and int(accounting["nonfinite_loss_skips"]) == 0
                and int(accounting["nonfinite_gradient_skips"]) == 0
            ),
            "no_pipeline_stamp_drift": all(
                runtime.get(key) == value for key, value in expected_stamp.items()
            ),
            "endpoint_rows_match_updates": (
                int(runtime["source_rows_gathered"]) == expected_rows
                and int(runtime["destination_rows_gathered"]) == expected_rows
            ),
            "weighted_rejection_accounting_closes": _weighted_rejection_accounting_mismatch(
                runtime, producer_delta=producer_delta, updates=updates
            )
            is None,
            "dose_derived_from_sealed_edge_count": (
                int(dose["active_graph_edges"]) == edges
            ),
            "dose_landed_on_registered_ceil_value": bool(
                dose["landed_on_registered_ceil_value"]
            ),
            "treatment_identical_to_r0217_except_seed": (
                observed_invariant == R0217_SEED_INVARIANT_SHA256
            ),
            "reconstructs_r0217_template_byte_for_byte": bool(
                assert_reconstructs_r0217_template(
                    config,
                    r0217_train_config(
                        seed=TEMPLATE_SEED,
                        graph_signature=graph["signature"],
                        graph_manifest_signature=graph["manifest_signature"],
                        substrate_signature=substrate_signature,
                        graph_edges=edges,
                        rows=ROWS,
                    )[0],
                )["byte_equal"]
            ),
            "published_checkpoint_reloads_finite_and_uncollapsed": bool(
                published["coordinates_finite"]
            )
            and not bool(published["collapsed"]),
            "all_2m_coordinates_finite": (
                int(published["transform_rows_finite"]) == ROWS
            ),
            "predicted_before_launch": "predicted_peak_device_bytes" in prediction,
            "not_refused_a_priori": not bool(prediction["refused_a_priori"]),
            "watchdog_did_not_trip": not bool(watchdog_state["tripped"]),
        },
        "memory": memory,
        "training_performed": True,
        "optimizer_updates": updates,
        "map_decision_made": False,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    })
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), receipt, immutable=True
    )
    del source, graph
    gc.collect()


# --------------------------------------------------------------------------- #
# B2. the n = 16 panel
# --------------------------------------------------------------------------- #


def _intra_queue_signature(
    reference: Mapping[str, Any], *, label: str
) -> tuple[str, dict[str, Any]]:
    reference = dict(reference)
    if reference.get("sha256"):
        return prompt_contract.verify_signature(reference, label=label), reference
    path = str(reference["canonical_path"])
    if not os.path.exists(path):
        raise Round0250PanelError(f"{label} is absent at {path}")
    return path, expected_input_signature(path)


def _sealed_panel_evidence(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """R0218's four-cell panel: the reference bytes and the frozen centroids."""
    panel_path = str(job["panel_evidence"])
    signature = expected_input_signature(panel_path)
    panel = prompt_contract.read_sealed(
        panel_path, label="R0218 MiniLM 2M four-seed panel"
    )
    checks = panel.get("execution_checks") or {}
    if (
        panel.get("round_id") != "0218"
        or panel.get("capabilities") != [PANEL_CAPABILITY]
        or panel.get("evaluation_performed") is not True
        or panel.get("gate_registered") is not False
        or panel.get("metrics") != list(PANEL_METRICS)
        or panel.get("seed_invariant_sha256") != R0217_SEED_INVARIANT_SHA256
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise Round0250PanelError("R0218 panel receipt contract changed")
    return panel, signature


def _sealed_n13_panel(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """R0230's thirteen pooled cells, read rather than rescored."""
    path, signature = _intra_queue_signature(
        job["panel_n13_signature"], label="R0230 sealed thirteen-cell panel"
    )
    panel = prompt_contract.read_sealed(path, label="R0230 sealed thirteen-cell panel")
    if (
        panel.get("schema") != PANEL_SCHEMA_N13
        or panel.get("round_id") != "0230"
        or panel.get("capabilities") != [PANEL_CAPABILITY_N13]
        or int(panel.get("n", -1)) != len(R0230_POOLED_SEEDS)
        or panel.get("gate_registerable_here") is not False
        or str(panel.get("family_seed_invariant_sha256") or "")
        != R0217_SEED_INVARIANT_SHA256
    ):
        raise Round0250PanelError("R0230 thirteen-cell panel receipt contract changed")
    want = {str(seed) for seed in R0230_POOLED_SEEDS}
    for key in ("panel_metric_cells", "raw_purity_ratios", "corpus_ffr_cells"):
        if set(panel.get(key) or {}) != want:
            raise Round0250PanelError(
                f"R0230 panel's {key} is not exactly the thirteen cells 42-54"
            )
    return panel, signature


def _load_centroids(panel: Mapping[str, Any]):
    declared = dict(panel.get("centroids") or {})
    if set(declared) != {str(k) for k in CENTROID_KS}:
        raise Round0250PanelError("R0218 centroid vocabularies changed")
    centroids: dict[int, np.ndarray] = {}
    signatures: dict[str, Any] = {}
    for k in CENTROID_KS:
        signature = dict(declared[str(k)])
        path = prompt_contract.verify_signature(
            signature, label=f"R0218 purity centroids k{k}"
        )
        array = np.load(path, allow_pickle=False)
        if array.shape != (k, DIMENSION) or array.dtype != np.dtype("float32"):
            raise Round0250PanelError(f"R0218 centroids k{k} geometry changed")
        centroids[k] = array
        signatures[str(k)] = signature
    return centroids, signatures


def _authenticate_map(
    cell: Mapping[str, Any], sealed: Mapping[str, Any]
) -> tuple[int, dict[str, Any], dict[str, Any], str]:
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0250PanelError(f"R0250 cell seed {seed!r} is not an R0250 cell")
    capability = capability_for_seed(seed)
    if str(cell.get("capability") or "") != capability:
        raise Round0250PanelError(f"R0250 seed-{seed} cell capability changed")
    receipt_path, receipt_signature = _intra_queue_signature(
        cell["train_receipt"], label=f"R0250 seed-{seed} train receipt"
    )
    receipt = prompt_contract.read_sealed(
        receipt_path, label=f"R0250 seed-{seed} train receipt"
    )
    train_checks = receipt.get("train_checks") or {}
    if (
        receipt.get("schema") != TRAIN_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("treatment_config_round_id") != "0217"
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != seed
        or receipt.get("training_performed") is not True
        or receipt.get("gate_registerable_here") is not False
        or receipt.get("map_decision_made") is not False
        or int(receipt.get("rows", -1)) != ROWS
        or int(receipt.get("dimension", -1)) != DIMENSION
        or int(receipt.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
        or int(receipt.get("optimizer_updates", -1)) != REGISTERED_SUCCESSFUL_UPDATES
        or receipt.get("graph_capability") != GRAPH_CAPABILITY
        or str(receipt.get("seed_invariant_sha256") or "")
        != R0217_SEED_INVARIANT_SHA256
        or not train_checks
        or not all(bool(value) for value in train_checks.values())
    ):
        raise Round0250PanelError(f"R0250 seed-{seed} train receipt contract changed")
    if (
        dict(receipt.get("substrate") or {}) != dict(sealed["substrate_signature"])
        or dict(receipt.get("graph_manifest") or {}) != dict(sealed["manifest_signature"])
    ):
        raise Round0250PanelError(
            f"R0250 seed-{seed} was not trained on the substrate this panel scores"
        )
    model_path = prompt_contract.verify_signature(
        receipt["model"], label=f"R0250 seed-{seed} published map"
    )
    return seed, receipt, receipt_signature, model_path


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0253 round0250_nodes.run_panel")
    import torch
    from basemap.panel_v2 import (
        hiD_reference_key,
        load_hiD_reference,
        reset_process_cuda_peak,
        sample_anchors,
        score_panel,
    )
    from basemap.pumap.parametric_umap import ParametricUMAP

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0250PanelError("R0250 panel handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0250PanelError("R0250 panel scoring requires CUDA")

    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    corpus_of_row = round0218_nodes._corpus_of_row(sealed)
    panel_evidence, panel_signature = _sealed_panel_evidence(job)
    panel_n13, panel_n13_signature = _sealed_n13_panel(job)

    prior_cells = dict(panel_n13["panel_metric_cells"])
    prior_ratios = dict(panel_n13["raw_purity_ratios"])
    prior_corpus = dict(panel_n13["corpus_ffr_cells"])

    cells_in = job.get("cells")
    if not isinstance(cells_in, list) or {
        int(cell.get("seed", -1)) for cell in cells_in
    } != set(SEEDS):
        raise Round0250PanelError("R0250 three-cell input matrix changed")
    authenticated: dict[int, dict[str, Any]] = {}
    for cell in cells_in:
        seed, receipt, receipt_signature, model_path = _authenticate_map(cell, sealed)
        authenticated[seed] = {
            "receipt": receipt,
            "receipt_signature": receipt_signature,
            "model_path": model_path,
        }
    invariants = {
        str(authenticated[seed]["receipt"]["seed_invariant_sha256"]) for seed in SEEDS
    } | {R0217_SEED_INVARIANT_SHA256}
    if len(invariants) != 1:
        raise Round0250PanelError(
            "R0250 pooled family is not commensurate: the new cells do not carry "
            "R0217's seed-invariant config digest"
        )
    prior_model_hashes = dict(job.get("prior_model_sha256_by_seed") or {})
    if set(prior_model_hashes) != {str(seed) for seed in R0230_POOLED_SEEDS}:
        raise Round0250PanelError(
            "R0250 panel needs the thirteen prior checkpoint digests, one per seed "
            "42-54, read from their sealed train receipts at prepare time"
        )
    model_hashes = {
        str(authenticated[seed]["receipt"]["model"]["sha256"]) for seed in SEEDS
    }
    model_hashes |= {str(value) for value in prior_model_hashes.values()}
    if len(model_hashes) != len(POOLED_SEEDS):
        raise Round0250PanelError(
            f"R0250 pooled family has {len(model_hashes)} distinct checkpoints, "
            f"expected {len(POOLED_SEEDS)}"
        )

    output = create_fresh_directory(str(job["outputs"][0]), label="R0250 n=16 panel")
    started = time.monotonic()
    reset_process_cuda_peak()

    cfg = prompt_contract.panel_config()
    centroids, centroid_signatures = _load_centroids(panel_evidence)
    data_identity = {
        "kind": "ordered_array",
        "shape": [ROWS, DIMENSION],
        "dtype": np.dtype("<f4").str,
        "sha256": sealed["ordered_substrate_sha256"],
    }
    reference_identity = {
        "data_identity": data_identity,
        "convention": dict(round0218_nodes.REFERENCE_CONVENTION),
    }
    reference_signature = dict(panel_evidence["shared_high_d_reference"])
    reference_path = prompt_contract.verify_signature(
        reference_signature, label="R0218 shared high-D reference"
    )
    observed_signature = expected_input_signature(reference_path)
    if observed_signature != reference_signature:
        raise Round0250PanelError(f"{REFERENCE_MISMATCH_MESSAGE} file signature drift")
    reference = load_hiD_reference(reference_path)
    anchors = sample_anchors(ROWS, cfg)
    if not np.array_equal(
        np.asarray(anchors, dtype=np.int64),
        np.asarray(reference["anchor_ids"], dtype=np.int64),
    ):
        raise Round0250PanelError(f"{REFERENCE_MISMATCH_MESSAGE} anchor drift")
    rederived_key, _parts = hiD_reference_key(
        source, anchors, cfg, centroids, kf=int(reference["kf"]), **reference_identity
    )
    anchor_labels = round0218_nodes._anchor_corpus_labels(corpus_of_row, anchors)
    anchor_corpus_counts = {
        slug: int((anchor_labels == slug).sum()) for slug in CORPUS_SLUGS
    }
    reference_identity_receipt = assert_reference_identity(
        file_signature=observed_signature,
        key=str(reference["key"]),
        content_sha256=str(reference["content_sha256"]),
        rederived_key=str(rederived_key),
        anchor_corpus_counts=anchor_corpus_counts,
    )
    if (
        str(reference["key"]) != str(panel_evidence["high_d_reference_key"])
        or str(reference["content_sha256"])
        != str(panel_evidence["high_d_reference_content_sha256"])
        or str(reference["key"]) != str(panel_n13["high_d_reference_key"])
    ):
        raise Round0250PanelError(
            f"{REFERENCE_MISMATCH_MESSAGE} the reference R0218 and R0230 used is "
            "not the one loaded here"
        )

    cells: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        entry = authenticated[seed]
        model = ParametricUMAP.load(entry["model_path"], device="cuda")
        coordinates = np.asarray(
            model.transform(source, batch_size=8192), dtype=np.float32
        )
        if coordinates.shape != (ROWS, 2):
            raise Round0250PanelError(
                f"R0250 seed-{seed} transform produced {coordinates.shape}, "
                f"expected ({ROWS}, 2)"
            )
        if not np.isfinite(coordinates).all():
            raise Round0250PanelError(
                f"R0250 seed-{seed} transform over {ROWS} rows is not finite"
            )
        coordinates_path = os.path.join(output, f"coordinates-seed{seed}.npy")
        atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
        coordinates_signature = expected_input_signature(coordinates_path)
        panel = score_panel(
            source,
            coordinates,
            config=cfg,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=reference_identity,
            ffr_group_labels=anchor_labels,
            scale_admission=None,
            provenance={
                "round_id": ROUND_ID,
                "seed": seed,
                "capability": capability_for_seed(seed),
                "universe": "R0216-queue-correction-3-minilm-mixed-2m",
                "substrate": dict(sealed["substrate_signature"]),
                "provenance_array": dict(sealed["provenance_signature"]),
                "train_receipt": dict(entry["receipt_signature"]),
                "coordinates": coordinates_signature,
                "shared_high_d_reference": reference_signature,
                "reference_source_round": "0218",
            },
        )
        if not panel_execution_ok(panel):
            raise Round0250PanelError(
                f"R0250 seed-{seed} panel is collapsed or nonfinite"
            )
        if not bool(panel["provenance"]["hiD_reference_reused"]):
            raise Round0250PanelError(
                f"{REFERENCE_MISMATCH_MESSAGE} seed-{seed} recomputed"
            )
        agreement = assert_hi_d_agreement(seed, panel["purity_numerators"])
        cells[seed] = {
            "seed": seed,
            "capability": capability_for_seed(seed),
            "train_receipt": dict(entry["receipt_signature"]),
            "model": dict(entry["receipt"]["model"]),
            "coordinates": coordinates_signature,
            "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
            "panel": panel,
            "panel_metrics": panel_metric_view(panel),
            "raw_purity_ratios": raw_purity_ratios(panel),
            "hi_d_agreement": agreement,
            "corpus_ffr": corpus_ffr_view(panel),
            "panel_finite_noncollapsed": True,
            "transform_rows_finite": ROWS,
        }
        del model, coordinates
        torch.cuda.empty_cache()
        gc.collect()

    pooled_cells: dict[str, dict[str, float]] = {}
    pooled_ratios: dict[str, dict[str, float]] = {}
    pooled_corpus: dict[str, dict[str, dict[str, float]]] = {}
    for seed in R0230_POOLED_SEEDS:
        key = str(seed)
        pooled_cells[key] = {
            name: float(value) for name, value in prior_cells[key].items()
        }
        pooled_ratios[key] = {
            name: float(value) for name, value in prior_ratios[key].items()
        }
        pooled_corpus[key] = {
            slug: {
                "anchors": int(prior_corpus[key][slug]["anchors"]),
                "ffr": float(prior_corpus[key][slug]["ffr"]),
            }
            for slug in CORPUS_SLUGS
        }
    for seed in SEEDS:
        key = str(seed)
        pooled_cells[key] = dict(cells[seed]["panel_metrics"])
        pooled_ratios[key] = dict(cells[seed]["raw_purity_ratios"])
        pooled_corpus[key] = dict(cells[seed]["corpus_ffr"])

    pooled = pool_sixteen_cells(
        cells=pooled_cells,
        ratios=pooled_ratios,
        corpus=pooled_corpus,
        sources=POOLED_CELL_SOURCES,
    )
    summary = descriptive_family_summary({
        metric: [pooled_cells[str(seed)][metric] for seed in POOLED_SEEDS]
        for metric in PANEL_METRICS
    })

    execution_checks = {
        "all_three_new_cells_scored": set(cells) == set(SEEDS),
        "sixteen_pooled_cells": len(pooled_cells) == len(POOLED_SEEDS),
        "reaches_the_standing_minimum_n": len(POOLED_SEEDS) >= STANDING_MINIMUM_N,
        "every_metric_finite": all(
            math.isfinite(float(value))
            for cell in pooled_cells.values()
            for value in cell.values()
        ),
        "every_raw_ratio_positive_and_finite": all(
            math.isfinite(float(value)) and float(value) > 0.0
            for cell in pooled_ratios.values()
            for value in cell.values()
        ),
        "no_collapsed_panel": all(
            bool(cell["panel_finite_noncollapsed"]) for cell in cells.values()
        ),
        "map_transform_finite_over_all_rows": all(
            int(cell["transform_rows_finite"]) == ROWS for cell in cells.values()
        ),
        "per_corpus_ffr_slices_complete": all(
            set(pooled_corpus[str(seed)]) == set(CORPUS_SLUGS) for seed in POOLED_SEEDS
        ),
        "pooled_family_one_seed_invariant_digest": len(invariants) == 1,
        "sixteen_distinct_checkpoints": len(model_hashes) == len(POOLED_SEEDS),
        "reference_byte_identical_to_r0218": bool(
            reference_identity_receipt["reference_byte_identical_to_r0218"]
        ),
        "hi_d_agreement_identical_in_every_new_cell": all(
            cell["hi_d_agreement"] == {"k256": 0.3828, "k1024": 0.2385}
            for cell in cells.values()
        ),
        "shared_reference_reused_by_content_key": all(
            bool(cell["panel"]["provenance"]["hiD_reference_reused"])
            for cell in cells.values()
        ),
        "prior_thirteen_cells_not_rescored": (
            set(cells) == set(SEEDS)
            and set(prior_cells) == {str(seed) for seed in R0230_POOLED_SEEDS}
        ),
        "no_floor_registered_here": not pooled["gate_registerable_here"],
    }
    if not all(execution_checks.values()):
        raise Round0250PanelError(f"R0250 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0250PanelError(
            f"R0250 panel peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )

    receipt = prompt_contract.seal({
        **pooled,
        "schema": PANEL_SCHEMA_N16,
        "round_id": ROUND_ID,
        "capability": PANEL_CAPABILITY_N16,
        "capabilities": [PANEL_CAPABILITY_N16],
        "release_sha": active["manifest"]["release_sha"],
        "panel_evidence": panel_signature,
        "panel_n13_evidence": panel_n13_signature,
        "panel_capability": PANEL_CAPABILITY,
        "panel_n13_capability": PANEL_CAPABILITY_N13,
        "lineage": {
            "graph_manifest": dict(sealed["manifest_signature"]),
            "substrate": dict(sealed["substrate_signature"]),
            "provenance": dict(sealed["provenance_signature"]),
            "ordered_substrate_sha256": sealed["ordered_substrate_sha256"],
        },
        "panel_configuration": {
            "source": "accepted R0113 panel_config()",
            "frac": cfg.frac,
            "k_hit": cfg.k_hit,
            "k_density": cfg.k_density,
            "k_frac_effective": int(reference["kf"]),
            "n_anchors": int(len(anchors)),
            "anchor_seed": cfg.anchor_seed,
            "formula_version": cfg.formula_version,
            "centroid_ks": list(CENTROID_KS),
            "centroid_source": "R0218 published frozen centroid arrays, loaded",
        },
        "shared_high_d_reference": reference_signature,
        "high_d_reference_key": str(reference["key"]),
        "high_d_reference_content_sha256": str(reference["content_sha256"]),
        "reference_convention": dict(round0218_nodes.REFERENCE_CONVENTION),
        "reference_identity": reference_identity_receipt,
        "centroids": centroid_signatures,
        "anchor_corpus_counts": anchor_corpus_counts,
        "new_cells": {str(seed): cells[seed] for seed in SEEDS},
        "descriptive_family_summary": summary,
        "density_v2_status": DENSITY_V2_STATUS,
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gate_registered": False,
        "gate_registerable_here": False,
        "map_decision_made": False,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "outcome": "minilm-mixed-2m-seed-family-pooled-at-n16-on-r0218s-frozen-panel",
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "seed-family-panel-n16.json"), receipt, immutable=True
    )
    print(json.dumps({
        "capability": PANEL_CAPABILITY_N16,
        "n": len(POOLED_SEEDS),
        "reference_byte_identical_to_r0218": True,
    }))
    del source, corpus_of_row, reference, centroids
    gc.collect()


# --------------------------------------------------------------------------- #
# B3. the n = 16 registration
# --------------------------------------------------------------------------- #


def _exact_cell_id(seed: int) -> str:
    return f"exact-seed{int(seed)}"


def calibrate_at_n16(
    sealed_ladder: Sequence[Mapping[str, Any]],
    sealed_n13_candidates: Mapping[str, Any],
) -> dict[str, Any]:
    """Calibrate at 16, and check the harness against published outside numbers.

    Three independent external checks, none of them self-consistency:

    * R0234's **sealed** power-parity ladder row at `n = 16`, reproduced bit for
      bit by calling the same released `power_ladder` with the same defaults;
    * R0234's **sealed** calibrated multipliers at `n = 13`, reproduced bit for
      bit by calling the same released `calibrate` at 13;
    * the two closed forms at `n = 16` -- the noncentral-t one-sided factor and
      Howe's two-sided factor -- against the calibrated sample-sd multipliers, at
      R0234's own registered tolerances.
    """
    at16 = calibration.calibrate(N_EXACT)
    arrays16 = at16.pop("_arrays")
    nct16 = calibration.nct_one_sided_factor(N_EXACT)
    howe16 = calibration.howe_two_sided_factor(N_EXACT)

    ladder16 = calibration.power_ladder("median_minus_k_madn", sizes=(N_EXACT,))[0]
    sealed16 = next(
        (row for row in sealed_ladder if int(row["n"]) == N_EXACT), None
    )
    if sealed16 is None:
        raise Round0250GateError(
            f"R0234's sealed power ladder has no n = {N_EXACT} row to check against"
        )

    at13 = calibration.calibrate(len(R0230_POOLED_SEEDS))
    at13.pop("_arrays")

    checks: list[dict[str, Any]] = [
        {
            "key": "r0234_sealed_power_ladder_n16_multiplier",
            "source": "R0234 sealed power_parity_ladder (600,000 families)",
            "target": float(sealed16["calibrated_multiplier"]),
            "observed": float(ladder16["calibrated_multiplier"]),
            "tolerance": 1e-12,
        },
        {
            "key": "r0234_sealed_power_ladder_n16_power_at_minus_2_sigma",
            "source": "R0234 sealed power_parity_ladder (600,000 families)",
            "target": float(sealed16["detection_power_at_minus_2_sigma"]),
            "observed": float(ladder16["detection_power_at_minus_2_sigma"]),
            "tolerance": 1e-12,
        },
        {
            "key": "n16_calibrated_sample_sd_k_vs_nct",
            "source": "closed form: k = t'_{n-1,z*sqrt(n)}(gamma)/sqrt(n) at n = 16",
            "target": 0.0,
            "observed": float(
                at16["candidates"]["mean_minus_k_sample_sd"]["one_sided"][
                    "calibrated_multiplier"
                ]
            )
            - nct16,
            "tolerance": 0.006,
        },
        {
            "key": "n16_calibrated_sample_sd_k2_vs_howe",
            "source": "closed form: Howe (1969) two-sided factor at n = 16",
            "target": 0.0,
            "observed": float(
                at16["candidates"]["mean_minus_k_sample_sd"]["two_sided"][
                    "calibrated_multiplier"
                ]
            )
            - howe16,
            "tolerance": 0.025,
        },
    ]
    for name in CANDIDATE_ORDER:
        for side, key in (
            ("one_sided", "calibrated_one_sided_multiplier"),
            ("two_sided", "calibrated_two_sided_multiplier"),
        ):
            checks.append({
                "key": f"r0234_sealed_n13_{side}_k::{name}",
                "source": "R0234 sealed selection.candidates (4,000,000 families)",
                "target": float(sealed_n13_candidates[name][key]),
                "observed": float(
                    at13["candidates"][name][side]["calibrated_multiplier"]
                ),
                "tolerance": 0.0,
            })
    for item in checks:
        item["delta"] = abs(float(item["observed"]) - float(item["target"]))
        item["reproduced"] = bool(item["delta"] <= float(item["tolerance"]))
    if not all(item["reproduced"] for item in checks):
        raise Round0250GateError(
            "R0250 calibration harness did not reproduce every external reference: "
            f"{[item['key'] for item in checks if not item['reproduced']]}"
        )
    return {
        "n": N_EXACT,
        "n16": at16,
        "n13_reproduction": at13,
        "arrays": arrays16,
        "closed_forms": {
            "nct_one_sided_factor_at_n16": nct16,
            "howe_two_sided_factor_at_n16": howe16,
            "identity_bound_at_n16": IDENTITY_BOUND_AT_N,
        },
        "power_ladder_n16": ladder16,
        "r0234_sealed_power_ladder_n16": dict(sealed16),
        "external_reference_checks": checks,
        "external_references_all_reproduced": True,
        "content": TOLERANCE_CONTENT,
        "confidence": TOLERANCE_CONFIDENCE,
    }


def evaluate_selection_n16(
    *,
    calibrated: Mapping[str, Any],
    series: Mapping[str, Sequence[float]],
    log_series: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """R0234's pre-registered rule, evaluated at `n = 16`.

    Every primitive is R0234's released one -- `injection_ladder`, `attainability`,
    `positive_scale_witness`, `degenerate_witness`, and the three requirement
    thresholds. Only `n` differs, which is the whole point of the round.
    """
    candidates: dict[str, Any] = {}
    for name in CANDIDATE_ORDER:
        entry = calibrated["n16"]["candidates"][name]
        k_one = float(entry["one_sided"]["calibrated_multiplier"])
        k_two = float(entry["two_sided"]["calibrated_multiplier"])
        coverage_one = float(entry["one_sided"]["delivered_coverage"])
        coverage_two = float(entry["two_sided"]["delivered_confidence_at_content"])
        coverage_ok = (
            abs(coverage_one - COVERAGE_TARGET) <= COVERAGE_TOLERANCE
            and abs(coverage_two - COVERAGE_TARGET) <= COVERAGE_TOLERANCE
        )

        ladders: dict[str, Any] = {}
        for metric in GATED_METRICS:
            ladders[metric] = injection_ladder(name, series[metric], k_one, side="lower")
        for metric, logs in log_series.items():
            ladders[f"log_ratio::{metric}::lower"] = injection_ladder(
                name, logs, k_two, side="lower"
            )
            ladders[f"log_ratio::{metric}::upper"] = injection_ladder(
                name, logs, k_two, side="upper"
            )
        descriptive_ladders = {
            metric: injection_ladder(name, series[metric], k_one, side="lower")
            for metric in DESCRIPTIVE_METRICS
        }
        depths = {key: item["exact_invariance_depth"] for key, item in ladders.items()}
        invariance_ok = min(depths.values()) >= REQUIRED_INVARIANCE_DEPTH

        bound = attainability(name, n=N_EXACT, multiplier=k_one)
        bound_two = attainability(name, n=N_EXACT, multiplier=k_two)
        witness = positive_scale_witness(name, k_one, n=N_EXACT)
        attainable_ok = (
            bool(bound["every_defining_cell_can_fail"])
            and bool(bound_two["every_defining_cell_can_fail"])
            and bool(witness["scale_is_strictly_positive"])
            and bool(witness["lowest_cell_fails_its_own_floor"])
        )
        candidates[name] = {
            "estimator": name,
            "n": N_EXACT,
            "centre": CANDIDATES[name]["centre"],
            "scale": CANDIDATES[name]["scale_name"],
            "asymptotic_breakdown_point": CANDIDATES[name]["breakdown_point"],
            # R0234's table computes the trimmed family's breakdown point as
            # TRIM_COUNT / 13, its own n. At n = 16 it is 1/16. The corrected value
            # is published beside the imported one rather than by editing R0234;
            # it changes no outcome, because the breakdown point is only a
            # tie-break among candidates that already qualify and 1/16 loses to
            # MAD_n's and S_n's and Q_n's 0.50 either way.
            "asymptotic_breakdown_point_at_this_n": (
                1.0 / float(N_EXACT)
                if name == "trimmed1_mean_minus_k_trimmed_sd"
                else CANDIDATES[name]["breakdown_point"]
            ),
            "gaussian_efficiency": CANDIDATES[name]["gaussian_efficiency"],
            "calibrated_one_sided_multiplier": k_one,
            "calibrated_two_sided_multiplier": k_two,
            "delivered_one_sided_coverage": coverage_one,
            "delivered_two_sided_confidence": coverage_two,
            "new_cell_false_fail_rate_one_sided": float(
                entry["one_sided"]["new_cell_false_fail_rate"]
            ),
            "new_cell_false_fail_rate_two_sided": float(
                entry["two_sided"]["new_cell_false_fail_rate"]
            ),
            "detection_power": entry["one_sided"]["detection_power"],
            "detection_power_at_selection_alternative": float(
                entry["one_sided"]["detection_power"][
                    f"minus_{POWER_SELECTION_ALTERNATIVE:g}_sigma"
                ]
            ),
            "invariance_ladders": ladders,
            "invariance_ladders_descriptive": descriptive_ladders,
            "exact_invariance_depth_by_series": depths,
            "minimum_exact_invariance_depth": min(depths.values()),
            "attainability_one_sided": bound,
            "attainability_two_sided": bound_two,
            "positive_scale_witness": witness,
            "degenerate_witness_r0231_used": degenerate_witness(name, k_one),
            "requirement_1_coverage": coverage_ok,
            "requirement_2_invariance": invariance_ok,
            "requirement_3_attainability": attainable_ok,
            "qualifies": bool(coverage_ok and invariance_ok and attainable_ok),
        }

    qualifying = [name for name, item in candidates.items() if item["qualifies"]]
    chosen = None
    reasoning: list[str] = []
    if qualifying:
        best = max(
            candidates[name]["detection_power_at_selection_alternative"]
            for name in qualifying
        )
        tied = [
            name
            for name in qualifying
            if best - candidates[name]["detection_power_at_selection_alternative"]
            <= POWER_MATERIALITY
        ]
        reasoning.append(
            f"qualifying: {qualifying}; best detection power at "
            f"-{POWER_SELECTION_ALTERNATIVE:g} sigma is {best!r}; within the "
            f"{POWER_MATERIALITY!r} materiality band: {tied}"
        )
        top_breakdown = max(
            candidates[name]["asymptotic_breakdown_point_at_this_n"] for name in tied
        )
        tied = [
            name
            for name in tied
            if candidates[name]["asymptotic_breakdown_point_at_this_n"]
            == top_breakdown
        ]
        reasoning.append(
            f"highest asymptotic breakdown point {top_breakdown!r}: {tied}"
        )
        if len(tied) > 1:
            smallest = min(
                candidates[name]["calibrated_one_sided_multiplier"] for name in tied
            )
            tied = [
                name
                for name in tied
                if candidates[name]["calibrated_one_sided_multiplier"] == smallest
            ]
            reasoning.append(f"smallest calibrated multiplier {smallest!r}: {tied}")
        chosen = tied[0]
        reasoning.append(f"registered: {chosen}")
    else:
        reasoning.append("no candidate qualifies at n = 16; register NOTHING")
    return {
        "rule": SELECTION_RULE,
        "n": N_EXACT,
        "candidates": candidates,
        "qualifying": qualifying,
        "chosen_estimator": chosen,
        "reasoning": reasoning,
    }


def run_gate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0250GateError("R0250 gate handler received another queue")
    started = time.monotonic()
    label = "R0250 n=16 calibrated robust floor registration"
    verify_registry(label=label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0250 gate")

    r0234 = prompt_contract.read_sealed(
        _bound_path(job, "r0234_gate", label="R0234 sealed calibrated gate"),
        label="R0234 sealed calibrated gate",
    )
    # 1. the Gaussian null, before any sealed CELL is opened.
    calibrated = calibrate_at_n16(
        list(r0234["power_parity_ladder"]["ladder"]),
        dict(r0234["selection"]["candidates"]),
    )

    panel = prompt_contract.read_sealed(
        _bound_path(job, "panel_n16", label="R0250 sealed sixteen-cell panel"),
        label="R0250 sealed sixteen-cell panel",
    )
    r0223 = prompt_contract.read_sealed(
        _bound_path(job, "r0223_comparison", label="R0223 sealed cuVS comparison"),
        label="R0223 sealed cuVS comparison",
    )
    r0225 = prompt_contract.read_sealed(
        _bound_path(job, "r0225_gate", label="R0225 sealed n=8 tolerance gate"),
        label="R0225 sealed n=8 tolerance gate",
    )
    r0228 = prompt_contract.read_sealed(
        _bound_path(job, "r0228_comparison", label="R0228 sealed cluster-spill"),
        label="R0228 sealed cluster-spill comparison",
    )
    r0231 = prompt_contract.read_sealed(
        _bound_path(job, "r0231_gate", label="R0231 sealed n=13 robust gate"),
        label="R0231 sealed n=13 robust gate",
    )

    exact_cells = panel["panel_metric_cells"]
    exact_ratios = panel["raw_purity_ratios"]
    if tuple(sorted(int(seed) for seed in exact_cells)) != EXACT_FAMILY_SEEDS:
        raise Round0250GateError("R0250 exact family is not seeds 42-57")
    if int(panel["n"]) != N_EXACT or N_EXACT < STANDING_MINIMUM_N:
        raise Round0250GateError(
            f"R0250 requires the sixteen-cell panel and n >= {STANDING_MINIMUM_N}"
        )
    if panel.get("gate_registerable_here") is not False:
        raise Round0250GateError("R0250's panel claims to register a gate; it must not")

    seeds = [str(seed) for seed in EXACT_FAMILY_SEEDS]
    series = {
        metric: [float(exact_cells[seed][metric]) for seed in seeds]
        for metric in METRICS
    }
    log_series = {
        metric: [
            math.log(float(exact_ratios[seed][PURITY_RATIO_KEYS[metric]]))
            for seed in seeds
        ]
        for metric in PURITY_METRICS
    }

    # 2. the pre-registered rule at the new n.
    selection = evaluate_selection_n16(
        calibrated=calibrated, series=series, log_series=log_series
    )
    chosen = selection["chosen_estimator"]
    if chosen is None:
        raise Round0250GateError(
            "R0250 registers nothing: no candidate qualifies at n = 16"
        )

    # 3. every candidate's floors and bands on the sixteen cells.
    fitted: dict[str, Any] = {}
    for name in CANDIDATE_ORDER:
        item = selection["candidates"][name]
        k_one = float(item["calibrated_one_sided_multiplier"])
        k_two = float(item["calibrated_two_sided_multiplier"])
        floors = {metric: floor_at(name, series[metric], k_one) for metric in METRICS}
        bands = {
            metric: band_at(name, log_series[metric], k_two)
            for metric in PURITY_METRICS
        }
        centres = {
            metric: dict(
                zip(("centre", "scale"), centre_and_scale(name, series[metric]))
            )
            for metric in METRICS
        }
        log_centres = {
            metric: dict(
                zip(("centre", "scale"), centre_and_scale(name, log_series[metric]))
            )
            for metric in PURITY_METRICS
        }
        fitted[name] = {
            "estimator": name,
            "one_sided_multiplier": k_one,
            "two_sided_multiplier": k_two,
            "floors": floors,
            "descriptive_folded_purity_floors": {
                metric: floors[metric] for metric in PURITY_METRICS
            },
            "two_sided_ratio_bands": {
                metric: {
                    "ratio_lower": math.exp(bands[metric][0]),
                    "ratio_upper": math.exp(bands[metric][1]),
                    "log_lower": bands[metric][0],
                    "log_upper": bands[metric][1],
                    "log_centre": log_centres[metric]["centre"],
                    "log_scale": log_centres[metric]["scale"],
                    "ratio_geometric_centre": math.exp(log_centres[metric]["centre"]),
                    "quantisation_note": (
                        "panel_v2 rounds each purity ratio to four decimals inside "
                        "the scorer, so this band inherits +/- 5e-5 in r"
                    ),
                }
                for metric in PURITY_METRICS
            },
            "centre_and_scale_by_metric": centres,
            "log_centre_and_scale_by_metric": log_centres,
            "scale_over_sample_sd": {
                metric: centres[metric]["scale"] / statistics.stdev(series[metric])
                for metric in METRICS
            },
            "effective_sigma_multiplier": {
                metric: (
                    statistics.fmean(series[metric]) - floors[metric]
                ) / statistics.stdev(series[metric])
                for metric in METRICS
            },
        }

    # 4. the twenty-eight cells.
    exact_scoring_cells = [
        {
            "cell_id": _exact_cell_id(seed),
            "family": "exact-graph",
            "values": {
                metric: float(exact_cells[str(seed)][metric]) for metric in METRICS
            },
            "ratios": {
                key: float(exact_ratios[str(seed)][key]) for key in ("k256", "k1024")
            },
        }
        for seed in EXACT_FAMILY_SEEDS
    ]
    held_out_cells: list[dict[str, Any]] = []
    cuvs_cells = r0223["cuvs_panel_metric_cells"]
    cuvs_ratios = r0223["cuvs_purity_ratios"]
    if tuple(sorted(int(seed) for seed in cuvs_cells)) != CUVS_FAMILY_SEEDS:
        raise Round0250GateError("R0250 cuVS family is not seeds 42-44")
    for seed in CUVS_FAMILY_SEEDS:
        held_out_cells.append({
            "cell_id": f"cuvs-igd48-seed{seed}",
            "family": "cuvs-igd48",
            "values": {
                metric: float(cuvs_cells[str(seed)][metric]) for metric in METRICS
            },
            "ratios": {
                key: float(cuvs_ratios[str(seed)][key]) for key in ("k256", "k1024")
            },
        })
    candidate_cells = r0228["candidate_panel_metric_cells"]
    candidate_ratios = r0228["candidate_purity_ratios"]
    for clusters in CANDIDATE_CLUSTER_COUNTS:
        arm = candidate_cells[str(clusters)]
        arm_ratios = candidate_ratios[str(clusters)]
        if tuple(sorted(int(seed) for seed in arm)) != CANDIDATE_SEEDS:
            raise Round0250GateError(f"R0250 candidate arm c{clusters} is not 42-44")
        for seed in CANDIDATE_SEEDS:
            held_out_cells.append({
                "cell_id": f"cluster-spill-c{clusters}-seed{seed}",
                "family": f"cluster-spill-c{clusters}",
                "values": {
                    metric: float(arm[str(seed)][metric]) for metric in METRICS
                },
                "ratios": {
                    key: float(arm_ratios[str(seed)][key]) for key in ("k256", "k1024")
                },
            })
    if len(held_out_cells) != N_HELD_OUT:
        raise Round0250GateError("R0250 held-out set is not twelve cells")
    all_cells = list(exact_scoring_cells) + list(held_out_cells)
    defining_ids = [_exact_cell_id(seed) for seed in EXACT_FAMILY_SEEDS]

    scoring: dict[str, Any] = {}
    for name in CANDIDATE_ORDER:
        item = fitted[name]
        floors = {metric: item["floors"][metric] for metric in GATED_METRICS}
        bands = {
            metric: (
                item["two_sided_ratio_bands"][metric]["ratio_lower"],
                item["two_sided_ratio_bands"][metric]["ratio_upper"],
            )
            for metric in PURITY_METRICS
        }
        can_fail = bool(
            selection["candidates"][name]["attainability_one_sided"][
                "every_defining_cell_can_fail"
            ]
        )
        scoring[name] = {
            "all_cells": score_population(
                cells=all_cells,
                floors=floors,
                bands=bands,
                metrics=GATED_METRICS,
                defining_cell_ids=defining_ids,
                every_defining_cell_can_fail=can_fail,
            ),
            "exact_sixteen": score_population(
                cells=exact_scoring_cells,
                floors=floors,
                bands=bands,
                metrics=GATED_METRICS,
                defining_cell_ids=defining_ids,
                every_defining_cell_can_fail=can_fail,
            ),
            "held_out_twelve": score_population(
                cells=held_out_cells,
                floors=floors,
                bands=bands,
                metrics=GATED_METRICS,
                defining_cell_ids=defining_ids,
                every_defining_cell_can_fail=can_fail,
            ),
        }

    winner = fitted[chosen]
    this_round = {
        "floors": {"ffr": winner["floors"]["ffr"]},
        "bands": {
            metric: (
                winner["two_sided_ratio_bands"][metric]["ratio_lower"],
                winner["two_sided_ratio_bands"][metric]["ratio_upper"],
            )
            for metric in PURITY_METRICS
        },
        "gate_status": "registered-and-contingent-pending-review",
    }
    joint_families = joint_criteria_from_sealed(
        r0225=r0225, r0231=r0231, r0234=r0234, this_round=this_round
    )
    can_fail = bool(
        selection["candidates"][chosen]["attainability_one_sided"][
            "every_defining_cell_can_fail"
        ]
    )
    joint = score_joint(
        cells=all_cells,
        families=joint_families,
        defining_cell_ids=defining_ids,
        every_defining_cell_can_fail=can_fail,
    )
    falsifiability = falsifiability_statement(
        multiplier_one_sided=winner["one_sided_multiplier"],
        multiplier_two_sided=winner["two_sided_multiplier"],
    )
    changes = verdict_changes(
        chosen=scoring[chosen]["all_cells"],
        published={
            name: entry["scoring"]
            for name, entry in joint["per_family"].items()
            if name != THIS_FAMILY
        },
    )

    execution_checks = {
        "n_is_at_least_the_standing_minimum": N_EXACT >= STANDING_MINIMUM_N,
        "every_external_reference_reproduced": bool(
            calibrated["external_references_all_reproduced"]
        ),
        "multiplier_was_derived_at_this_n": (
            float(winner["one_sided_multiplier"])
            != float(
                calibrated["n13_reproduction"]["candidates"][chosen]["one_sided"][
                    "calibrated_multiplier"
                ]
            )
        ),
        "a_candidate_qualifies": chosen is not None,
        "density_v2_is_not_gated_anywhere": not (
            set(DESCRIPTIVE_METRICS) & set(GATED_METRICS)
        ),
        "purity_is_gated_on_the_unfolded_two_sided_ratio": all(
            "ratio_lower" in winner["two_sided_ratio_bands"][metric]
            for metric in PURITY_METRICS
        ),
        "joint_criteria_cover_every_retained_family": (
            len(joint["families"]) == len(RETAINED_FAMILY_SOURCES) + 1
        ),
        "every_defining_cell_can_fail": bool(
            falsifiability["registered_family_every_defining_cell_can_fail"]
        ),
        "count_is_attainable": bool(joint["count_is_attainable"]),
        "twenty_eight_cells_scored": len(all_cells) == N_EXACT + N_HELD_OUT,
    }
    if not all(execution_checks.values()):
        raise Round0250GateError(f"R0250 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": GATE_SCHEMA,
        "capability": GATE_CAPABILITY,
        "capabilities": [GATE_CAPABILITY],
        "n": N_EXACT,
        "standing_minimum_n": STANDING_MINIMUM_N,
        "identity_bound_at_n": IDENTITY_BOUND_AT_N,
        "selection_rule": SELECTION_RULE,
        "selection": selection,
        "calibration": {
            key: value
            for key, value in calibrated.items()
            if key not in {"arrays"}
        },
        "chosen_estimator": chosen,
        "fitted_candidates": fitted,
        "registered_floors": {
            metric: (
                winner["floors"][metric] if metric not in PURITY_METRICS else None
            )
            for metric in METRICS
        },
        "registered_floors_all_metrics_including_descriptive": winner["floors"],
        "registered_floors_purity_entries_are_descriptive": (
            "the purity entries of registered_floors are published as null: the "
            "gated purity criterion is the two-sided band on the UNFOLDED ratio "
            "below. A consumer reading a folded one-sided purity floor would fail "
            "cells this gate passes -- the trap review-0231-01 found in R0231."
        ),
        "registered_two_sided_bands": winner["two_sided_ratio_bands"],
        "descriptive_folded_purity_floors": winner["descriptive_folded_purity_floors"],
        "gated_metrics": list(GATED_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        "density_v2_defect": DENSITY_V2_DEFECT,
        "density_v2_status": DENSITY_V2_STATUS,
        "scoring_by_candidate": scoring,
        "joint_criteria": joint,
        "joint_criteria_rule": JOINT_CRITERIA_RULE,
        "retained_family_sources": [dict(item) for item in RETAINED_FAMILY_SOURCES],
        "falsifiability": falsifiability,
        "verdict_changes_versus_retained_families": changes,
        "exact_family_seeds": list(EXACT_FAMILY_SEEDS),
        "held_out_cells": [cell["cell_id"] for cell in held_out_cells],
        "sources": {
            "r0223_cuvs_comparison": expected_input_signature(
                _bound_path(job, "r0223_comparison", label="R0223")
            ),
            "r0225_tolerance_gate_n8": expected_input_signature(
                _bound_path(job, "r0225_gate", label="R0225")
            ),
            "r0228_cluster_spill_comparison": expected_input_signature(
                _bound_path(job, "r0228_comparison", label="R0228")
            ),
            "r0231_robust_gate_n13": expected_input_signature(
                _bound_path(job, "r0231_gate", label="R0231")
            ),
            "r0234_calibrated_gate_n13": expected_input_signature(
                _bound_path(job, "r0234_gate", label="R0234")
            ),
            "r0250_panel_n16": expected_input_signature(
                _bound_path(job, "panel_n16", label="R0250 panel")
            ),
        },
        "supersedes_capability": None,
        "supersession_note": (
            "supersedes NOTHING. Every criterion R0225, R0231 and R0234 released "
            "stays in force alongside this one; see joint_criteria."
        ),
        "gate_status": "registered-and-contingent-pending-review",
        "gate_registered": True,
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gpu_used": False,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "peak_host_rss_gib": peak_rss_gib,
        "wall_seconds": time.monotonic() - started,
    })
    _seal(output, "minilm-calibrated-robust-floors-n16.json", body)
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "n": N_EXACT,
        "chosen_estimator": chosen,
        "one_sided_multiplier": winner["one_sided_multiplier"],
        "cells_clearing_the_joint_criteria": joint["cells_clearing_the_joint_criteria"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == TRAINLOOP_ACTION:
        run_trainloop(active, job)
        return
    if action == BLOCKSIZE_ACTION:
        run_blocksize(active, job)
        return
    if action == TRAIN_ACTION:
        run_train(active, job)
        return
    if action == PANEL_ACTION:
        run_panel(active, job)
        return
    if action == GATE_ACTION:
        run_gate(active, job)
        return
    raise Round0250Error(f"R0250 unknown action {action!r}")


__all__ = [
    "BLOCKSIZE_ACTION",
    "GATE_ACTION",
    "PANEL_ACTION",
    "TRAINLOOP_ACTION",
    "TRAIN_ACTION",
    "calibrate_at_n16",
    "evaluate_selection_n16",
    "run_blocksize",
    "run_gate",
    "run_job",
    "run_panel",
    "run_train",
    "run_trainloop",
]
