"""Execute R0255 — thirteen new 2M cells, and the Phase 3 gate on `MAD_n` at n = 29.

Fifteen nodes in one queue.

* thirteen GPU trains (`seeds 58-70`) under R0217's treatment with the seed as the
  only free variable, structurally identical to R0221's, R0230's and R0250's;
* one GPU **replay control** (`seed 42`) that retrains R0217's own canonical cell on
  this release. It is **not a family cell**: seed 42 is already in the family, and
  the family-purity guard refuses a family that contains the control. Its job is to
  test the *training path* the way R0251's rescore tested the *scoring path*;
* `score_minilm_mixed_2m_panel_n29` (GPU) — the thirteen new maps and the replay
  control scored on R0218's frozen panel, the sixteen existing cells read from
  R0250's sealed pooled table and not rescored, the high-D reference proved
  byte-identical on the same five components. If it is not, the node raises and the
  round stops: twenty-nine cells that are not poolable is the finding;
* `register_calibrated_madn_floors_n29` (CPU) — R0234's calibrated method at
  `n = 29` on the estimator the **owner ruled**, with the poolability test, the
  attainability and detection power beside every floor, the independence control,
  and the JOINT criteria.

Every registered check is IMPORTED and called. The memory guard, the watchdog, the
sealed-graph reader, the substrate opener and the accounting closure come from
`round0230_nodes`; the calibration, invariance, attainability and scoring come from
`round0234_calibrated_floors`; the poll gate, the host watchdog and the three
`require_*` gates come from R0244-R0247; the coverage ledger from R0253. Nothing in
this module signals any process, hands cuVS anything, or wraps a child in a timeout.

**Coverage is earned by the layout, not by emitting the field (R0254).** Every node
constructs its gate and its coverage window BEFORE its expensive work, and the train
nodes hand the wrapped poll to `ParametricUMAP.abort_poll`, so `fit()` -- the
overwhelming majority of a training node's wall -- is inside the window and polls
once per batch. The full-population transform is walked in chunks with a poll
between them for the same reason.
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
from basemap.round0242_locality import json_scrub
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
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report
from basemap.round0253_coverage import CoverageLedger
from basemap.round0253_stop_hooks import install_stop_hooks
from basemap.round0255_gate_n29 import (
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
    NO_TUNING_STATEMENT,
    N_EXACT,
    N_HELD_OUT,
    OWNER_RULING,
    OWNER_RULING_ESTIMATOR,
    POWER_MATERIALITY,
    POWER_SELECTION_ALTERNATIVE,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
    REQUIRED_INVARIANCE_DEPTH,
    RETAINED_FAMILY_SOURCES,
    Round0255GateError,
    SELECTION_RULE,
    THIS_FAMILY,
    TOLERANCE_CONFIDENCE,
    TOLERANCE_CONTENT,
    attainability,
    attainability_and_power,
    band_at,
    centre_and_scale,
    degenerate_witness,
    exact_cell_id,
    falsifiability_statement,
    floor_at,
    independence_control,
    injection_ladder,
    joint_criteria_from_sealed,
    owner_ruling_registration,
    poolability_shift_test,
    positive_scale_witness,
    score_joint,
    score_population,
    verdict_changes,
)
from basemap.round0255_panel_n29 import (
    CENTROID_KS,
    CORPUS_SLUGS,
    DENSITY_V2_STATUS,
    PANEL_CAPABILITY,
    PANEL_CAPABILITY_N16,
    PANEL_CAPABILITY_N29,
    PANEL_METRICS,
    PANEL_SCHEMA_N16,
    PANEL_SCHEMA_N29,
    POOLED_CELL_SOURCES,
    REFERENCE_MISMATCH_MESSAGE,
    Round0255PanelError,
    assert_hi_d_agreement,
    assert_reference_identity,
    corpus_ffr_view,
    descriptive_family_summary,
    panel_execution_ok,
    panel_metric_view,
    pool_twenty_nine_cells,
    raw_purity_ratios,
    replay_control_comparison,
)
from basemap.round0255_seed_extension_n29 import (
    BATCH_SIZE,
    CAPABILITY_TEMPLATE,
    DEVICE_BUDGET_BYTES,
    DIMENSION,
    FULL_TRANSFORM_BATCH,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    HOST_ANON_BUDGET_BYTES,
    HOST_RSS_LIMIT_GIB,
    IDENTITY_BOUND_AT_N29,
    MEMORY_POLICY,
    OWNER_RULING_N,
    POOLED_SEEDS,
    POSITIVE_ROWS_PER_UPDATE,
    PRODUCTION_CONFIG_SCHEMA,
    R0217_SEED_INVARIANT_SHA256,
    R0250_POOLED_SEEDS,
    REGISTERED_SUCCESSFUL_UPDATES,
    REPLAY_CONTROL_CAPABILITY,
    REPLAY_CONTROL_SEED,
    REPLAY_SCHEMA,
    ROUND_ID,
    ROWS,
    Round0255Error,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    STANDING_MINIMUM_N,
    TEMPLATE_SEED,
    TRAIN_SCHEMA,
    assert_reconstructs_r0217_template,
    capability_for_seed,
    masked_config_bytes,
    performance_windows,
    predict_cell_footprint,
    replay_control_config,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config,
    validate_full_population_map,
    validate_registered_dose,
)
from basemap.round0255_treatment import (
    Round0255FamilyError,
    Round0255TreatmentError,
    TRAIN_CLOSURE_MODULES,
    assert_family_is_2m_only,
    assert_runtime_closure_matches_seal,
    family_purity_controls,
    runtime_closure_hashes,
    treatment_closure_controls,
)
from experiments import round0218_nodes
from experiments.round0230_nodes import (
    CellWatchdog,
    _open_substrate,
    _sealed_graph,
    _weighted_rejection_accounting_mismatch,
)
from experiments.round0113_nodes import _new_model
from experiments.round0238_nodes import _check_runner_abort


TRAIN_ACTION = "train_minilm_mixed_2m_seed_extension_n29"
PANEL_ACTION = "score_minilm_mixed_2m_panel_n29"
GATE_ACTION = "register_calibrated_madn_floors_n29"

#: Matches R0250's nodes. Its headroom comes from the registry, never a keyword.
NODE_ANON_BUDGET_BYTES = 16 * (1 << 30)

#: Rows per full-population transform chunk. The chunk exists so the transform is
#: inside the coverage window and polls between chunks; it changes nothing
#: numerically, because the projection is row-wise.
TRANSFORM_CHUNK_ROWS = 100_000

#: One `panel_v2` rounding quantum. Fixed here, before the run, as the replay
#: control's criterion -- the same tolerance R0251 fixed for its scorer-side control.
REPLAY_TOLERANCE = 1e-4

SAFETY_NOTE = (
    "no node in this module signals any process, starts a child process, hands "
    "cuVS anything, or wraps a subprocess in a timeout. Every bulk input is a "
    "read-only np.memmap opened by R0230's reader. The per-batch abort read is the "
    "release's own `ParametricUMAP.abort_poll` attribute, set to this node's "
    "recorder and cleared in a finally. Every bound the poll gate and the host "
    "guard apply is read from the R0247 registry at the comparison site."
)


# --------------------------------------------------------------------------- #
# shared node scaffolding, imported rather than re-typed where it exists
# --------------------------------------------------------------------------- #


def _read_json(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise Round0255Error(f"{label} is absent at {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _bound_path(job: Mapping[str, Any], key: str, *, label: str) -> str:
    reference = job.get(key)
    if not isinstance(reference, Mapping):
        raise Round0255Error(f"R0255 job is missing the bound input {key!r} ({label})")
    if reference.get("sha256"):
        return prompt_contract.verify_signature(dict(reference), label=label)
    path = str(reference["canonical_path"])
    if not os.path.exists(path):
        raise Round0255Error(f"{label} is absent at {path}")
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
    """The registered headroom, the registered ceiling, the registered reader."""
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


def _guard_tail_reported(
    watchdog: EnforcedHostWatchdog, *, label: str
) -> dict[str, Any]:
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


def _score_gate_without_raising(
    gate: AbortPollGate, tail: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Score the gate and publish the numbers whether or not the ceiling holds.

    A ceiling breach in a training node is this round's *measurement*, not its
    error: R0250 established that a training node's widest gap is its whole `fit()`
    wall without a per-batch poll, and this round installs one. Whether that is
    enough at this rung is the thing being measured, so the verdict is published
    rather than raised on.
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


def _closure_evidence(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The prepare-time source seal, and the bytes this process actually imported."""
    sealed = prompt_contract.read_sealed(
        _bound_path(job, "treatment_closure", label="R0255 treatment closure seal"),
        label="R0255 treatment closure seal",
    )
    observed = runtime_closure_hashes(TRAIN_CLOSURE_MODULES)
    verdict = assert_runtime_closure_matches_seal(sealed=sealed, observed=observed)
    return sealed, {
        "runtime_closure": observed,
        "verdict": verdict,
        "controls": treatment_closure_controls(sealed=sealed, observed=observed),
    }


def _transform_in_chunks(model: Any, source: Any, poll: Any) -> np.ndarray:
    """Project all 2,000,000 rows, polling between chunks.

    Row-wise projection, so chunking is numerically inert; it exists so the longest
    non-`fit()` stage of a training node is inside the coverage window instead of
    being a silent uninstrumented span (review-0252-01 §H).
    """
    parts: list[np.ndarray] = []
    for start in range(0, ROWS, TRANSFORM_CHUNK_ROWS):
        stop = min(start + TRANSFORM_CHUNK_ROWS, ROWS)
        block = np.asarray(
            model.transform(source[start:stop], batch_size=FULL_TRANSFORM_BATCH),
            dtype=np.float32,
        )
        parts.append(block)
        poll(f"R0255 transform rows {start}-{stop}")
    return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------- #
# the thirteen cells, and the replay control
# --------------------------------------------------------------------------- #


def _seed(job: Mapping[str, Any]) -> tuple[int, bool]:
    replay = bool(job.get("is_replay_control"))
    seed = job.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Round0255Error(f"R0255 job seed {seed!r} is not an integer")
    if replay:
        if seed != REPLAY_CONTROL_SEED:
            raise Round0255Error(
                f"R0255 replay control is seed {REPLAY_CONTROL_SEED}, not {seed!r}"
            )
        if str(job.get("capability") or "") != REPLAY_CONTROL_CAPABILITY:
            raise Round0255Error("R0255 replay control capability changed")
        return seed, True
    if seed not in SEEDS:
        raise Round0255Error(f"R0255 job seed {seed!r} is not a registered cell")
    if str(job.get("capability") or "") != capability_for_seed(seed):
        raise Round0255Error("R0255 job capability does not match its seed")
    return int(seed), False


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0255 round0255_nodes.run_train")
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0255Error("R0255 train handler received another queue")
    seed, replay = _seed(job)
    capability = REPLAY_CONTROL_CAPABILITY if replay else capability_for_seed(seed)
    node_id = str(active.get("node_id") or f"train_seed{seed}")
    label = f"R0255 {capability}"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)

    prediction = predict_cell_footprint(seed, replay_control=replay)
    declared = job.get("memory_prediction")
    if declared is not None and dict(declared) != prediction:
        raise Round0255Error(
            "R0255 cell prediction differs from the one sealed at prepare time"
        )
    if prediction["refused_a_priori"]:
        raise Round0255Error(
            f"R0255 seed-{seed} refused a priori: predicted "
            f"{prediction['predicted_peak_device_bytes']} device bytes and "
            f"{prediction['predicted_peak_host_anonymous_bytes']} host anonymous "
            f"bytes against budgets {DEVICE_BUDGET_BYTES} / {HOST_ANON_BUDGET_BYTES}"
        )

    closure_seal, closure = _closure_evidence(job)
    if not closure["controls"]["every_planted_defect_was_refused"]:
        raise Round0255TreatmentError(
            "R0255 the shipped closure guard did not refuse every planted defect: "
            f"{closure['controls']['controls']}"
        )
    if not closure["controls"]["the_honest_closure_still_passes"]:
        raise Round0255TreatmentError(
            "R0255 the shipped closure guard rejects the honest closure, so its "
            "refusals prove nothing"
        )

    graph = _sealed_graph(job)
    edges = graph["directed_edges"]
    updates = successful_updates_for_edges(edges)
    dose = validate_registered_dose(updates=updates, edge_count=edges)
    declared_bound = job.get("registered_dose_bound")
    if declared_bound is not None and updates > int(declared_bound):
        raise Round0255Error(
            "R0255 derived update horizon exceeds the registered round bound"
        )
    source, substrate_signature = _open_substrate(graph)
    if replay:
        config, config_sha = replay_control_config(
            graph_signature=graph["signature"],
            graph_manifest_signature=graph["manifest_signature"],
            substrate_signature=substrate_signature,
            graph_edges=edges,
            rows=ROWS,
        )
    else:
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
        raise Round0255Error(
            "R0255 cell config is not R0217's treatment outside the seed: "
            f"{observed_invariant} != {declared_invariant} / "
            f"{R0217_SEED_INVARIANT_SHA256}"
        )
    if int(config["optimizer"]["successful_positive_lr_updates"]) != updates:
        raise Round0255Error("R0255 horizon did not reach the train config")

    replay_targets: dict[str, Any] = {}
    if replay:
        r0217_receipt = prompt_contract.read_sealed(
            _bound_path(job, "r0217_receipt", label="R0217 sealed seed-42 receipt"),
            label="R0217 sealed seed-42 receipt",
        )
        replay_targets = {
            "r0217_production_config_sha256": str(
                r0217_receipt["production_config_sha256"]
            ),
            "r0217_model_sha256": str(dict(r0217_receipt["model"])["sha256"]),
            "r0217_model_bytes": int(dict(r0217_receipt["model"])["bytes"]),
            "r0217_train_receipt_round_id": str(r0217_receipt["round_id"]),
        }
        if config_sha != replay_targets["r0217_production_config_sha256"]:
            raise Round0255Error(
                "R0255 replay control config digest "
                f"{config_sha} != R0217's sealed "
                f"{replay_targets['r0217_production_config_sha256']}: the replay is "
                "not R0217's cell"
            )

    output = create_fresh_directory(str(job["outputs"][0]), label="R0255 train output")
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": PRODUCTION_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "treatment_config_round_id": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "seed": seed,
            "capability": capability,
            "is_a_family_cell": not replay,
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

    # The gate and the coverage window are constructed BEFORE the work, and the
    # release's own per-batch poll site is handed this node's recorder, so `fit()`
    # is inside the window rather than beside it.
    window = ledger.window(f"R0255 {capability} train stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=True)
    watchdog = CellWatchdog()
    watchdog.start()
    started = time.monotonic()
    try:
        with guard_ctx:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor(f"R0255 {capability} stage entered")
            wrapped = window.wrap(recorder)
            model.abort_poll = wrapped
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
            finally:
                model.abort_poll = None
            wall = time.monotonic() - started
            wrapped("R0255 fit() returned")
            accounting = dict(model._train_stats)
            runtime = wrapper.runtime_stamp()
            wrapped("R0255 train accounting read")
            model_path = os.path.join(output, "model.pt")
            from basemap.output_safety import atomic_build_new_file

            atomic_build_new_file(model_path, model.save, immutable=True)
            wrapped("R0255 checkpoint published")
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
            wrapped("R0255 training objects released")

            from basemap.pumap.parametric_umap import ParametricUMAP

            reloaded = ParametricUMAP.load(model_path, device="cuda")
            wrapped("R0255 checkpoint reloaded")
            coordinates = _transform_in_chunks(reloaded, source, wrapped)
            published = validate_full_population_map(coordinates)
            published["model"] = expected_input_signature(model_path)
            coordinates_ordered_sha256 = ordered_array_sha256(coordinates)
            del reloaded, coordinates
            torch.cuda.empty_cache()
            gc.collect()
            gate.finish(f"R0255 {capability} stage end")
        window.close()
        tail = _guard_tail_reported(guard_ctx, label=label)
        scored = _score_gate_without_raising(gate, tail, label=label)
        gaps = gap_report(recorder.records, arm=node_id)
    finally:
        watchdog_state = watchdog.stop()

    if watchdog_state["tripped"]:
        raise Round0255Error(
            f"R0255 seed-{seed} watchdog tripped: {watchdog_state['trip_reason']!r}"
        )

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
        raise Round0255Error(f"R0255 train accounting failed: {mismatches}")
    prompt_contract.synchronize_runtime_counters(accounting, runtime)
    accounting["pipeline_runtime"] = dict(runtime)

    if int(memory["peak_reserved_bytes"]) > DEVICE_BUDGET_BYTES:
        raise Round0255Error(
            f"R0255 seed-{seed} peak reserved device bytes "
            f"{memory['peak_reserved_bytes']} exceed the {DEVICE_BUDGET_BYTES} budget"
        )
    if int(watchdog_state["peak_anonymous_bytes"]) > HOST_ANON_BUDGET_BYTES:
        raise Round0255Error(
            f"R0255 seed-{seed} peak anonymous bytes "
            f"{watchdog_state['peak_anonymous_bytes']} exceed the "
            f"{HOST_ANON_BUDGET_BYTES} budget"
        )
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0255Error(
            f"R0255 train peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    memory["peak_host_rss_gib"] = peak_rss_gib
    memory["peak_host_anonymous_bytes"] = int(watchdog_state["peak_anonymous_bytes"])

    replay_receipt: dict[str, Any] = {}
    if replay:
        model_sha = str(published["model"]["sha256"])
        replay_receipt = {
            **replay_targets,
            "r0255_production_config_sha256": config_sha,
            "r0255_model_sha256": model_sha,
            "r0255_model_bytes": int(published["model"]["bytes"]),
            "r0255_coordinates_ordered_sha256": coordinates_ordered_sha256,
            "production_config_is_byte_identical_to_r0217": (
                config_sha == replay_targets["r0217_production_config_sha256"]
            ),
            "checkpoint_is_byte_identical_to_r0217": (
                model_sha == replay_targets["r0217_model_sha256"]
            ),
            "checkpoint_byte_length_matches": (
                int(published["model"]["bytes"])
                == replay_targets["r0217_model_bytes"]
            ),
            "what_a_checkpoint_difference_would_and_would_not_mean": (
                "a difference is NOT by itself treatment drift: a GPU training path "
                "is not required to be bit-reproducible across releases, and this "
                "round makes no such claim. The criterion is the panel comparison "
                "the n=29 panel node runs on this map against R0218's four sealed "
                "seed-42 values at one rounding quantum. The checkpoint digest is "
                "published for what it is: the strongest possible outcome if it "
                "matches, and an unresolved question if it does not."
            ),
        }

    receipt_body = {
        "schema": REPLAY_SCHEMA if replay else TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "treatment_config_round_id": "0217",
        "treatment_template_seed": TEMPLATE_SEED,
        "pooled_seed_family": list(POOLED_SEEDS),
        "owner_ruling_n": OWNER_RULING_N,
        "standing_minimum_n": STANDING_MINIMUM_N,
        "capability": capability,
        "capabilities": [capability],
        "training_seed": seed,
        "is_a_family_cell": not replay,
        "release_sha": active["manifest"]["release_sha"],
        "abort_flag_precondition": abort_flag,
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "seed_invariant_sha256": observed_invariant,
        "masked_config_bytes": len(masked),
        "r0217_published_seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "treatment_closure": closure["verdict"],
        "treatment_closure_controls": closure["controls"],
        "treatment_closure_seal": expected_input_signature(
            _bound_path(job, "treatment_closure", label="R0255 treatment closure seal")
        ),
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
        "train_wall_s": wall,
        "published_map_check": published,
        "coordinates_ordered_sha256": coordinates_ordered_sha256,
        "memory_prediction": prediction,
        "memory_watchdog": watchdog_state,
        "memory_policy": MEMORY_POLICY,
        "replay_control": replay_receipt or None,
        # Every entry is the expression that decided it, not a literal.
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
            "training_closure_ran_the_sealed_release_bytes": bool(
                closure["verdict"]["every_module_ran_the_sealed_bytes"]
            ),
            "every_planted_closure_defect_was_refused": bool(
                closure["controls"]["every_planted_defect_was_refused"]
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
        "gap_report": gaps,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "training_performed": True,
        "optimizer_updates": updates,
        "map_decision_made": False,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    }
    coverage = ledger.receipt()
    receipt_body["poll_coverage"] = coverage
    receipt_body["observed_span_s"] = coverage["observed_span_s"]
    receipt_body["node_wall_s"] = coverage["node_wall_s"]
    receipt_body["node"] = node_id
    receipt = prompt_contract.seal(json_safe(json_scrub(receipt_body)))
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), receipt, immutable=True
    )
    del source, graph
    gc.collect()
    print(json.dumps({
        "capability": capability,
        "seed": seed,
        "is_a_family_cell": not replay,
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


# --------------------------------------------------------------------------- #
# the n = 29 panel
# --------------------------------------------------------------------------- #


def _intra_queue_signature(
    reference: Mapping[str, Any], *, label: str
) -> tuple[str, dict[str, Any]]:
    reference = dict(reference)
    if reference.get("sha256"):
        return prompt_contract.verify_signature(reference, label=label), reference
    path = str(reference["canonical_path"])
    if not os.path.exists(path):
        raise Round0255PanelError(f"{label} is absent at {path}")
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
        raise Round0255PanelError("R0218 panel receipt contract changed")
    return panel, signature


def _sealed_n16_panel(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """R0250's sixteen pooled cells, read rather than rescored."""
    path, signature = _intra_queue_signature(
        job["panel_n16_signature"], label="R0250 sealed sixteen-cell panel"
    )
    panel = prompt_contract.read_sealed(path, label="R0250 sealed sixteen-cell panel")
    if (
        panel.get("schema") != PANEL_SCHEMA_N16
        or panel.get("round_id") != "0250"
        or panel.get("capabilities") != [PANEL_CAPABILITY_N16]
        or int(panel.get("n", -1)) != len(R0250_POOLED_SEEDS)
        or panel.get("gate_registerable_here") is not False
        or str(panel.get("family_seed_invariant_sha256") or "")
        != R0217_SEED_INVARIANT_SHA256
    ):
        raise Round0255PanelError("R0250 sixteen-cell panel receipt contract changed")
    want = {str(seed) for seed in R0250_POOLED_SEEDS}
    for key in ("panel_metric_cells", "raw_purity_ratios", "corpus_ffr_cells"):
        if set(panel.get(key) or {}) != want:
            raise Round0255PanelError(
                f"R0250 panel's {key} is not exactly the sixteen cells 42-57"
            )
    return panel, signature


def _load_centroids(panel: Mapping[str, Any]):
    declared = dict(panel.get("centroids") or {})
    if set(declared) != {str(k) for k in CENTROID_KS}:
        raise Round0255PanelError("R0218 centroid vocabularies changed")
    centroids: dict[int, np.ndarray] = {}
    signatures: dict[str, Any] = {}
    for k in CENTROID_KS:
        signature = dict(declared[str(k)])
        path = prompt_contract.verify_signature(
            signature, label=f"R0218 purity centroids k{k}"
        )
        array = np.load(path, allow_pickle=False)
        if array.shape != (k, DIMENSION) or array.dtype != np.dtype("float32"):
            raise Round0255PanelError(f"R0218 centroids k{k} geometry changed")
        centroids[k] = array
        signatures[str(k)] = signature
    return centroids, signatures


def _authenticate_map(
    cell: Mapping[str, Any], sealed: Mapping[str, Any]
) -> tuple[int, bool, dict[str, Any], dict[str, Any], str]:
    replay = bool(cell.get("is_replay_control"))
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Round0255PanelError(f"R0255 cell seed {seed!r} is not an integer")
    if replay:
        if seed != REPLAY_CONTROL_SEED:
            raise Round0255PanelError("R0255 replay control cell is not seed 42")
        capability = REPLAY_CONTROL_CAPABILITY
        schema = REPLAY_SCHEMA
    else:
        if seed not in SEEDS:
            raise Round0255PanelError(f"R0255 cell seed {seed!r} is not an R0255 cell")
        capability = capability_for_seed(seed)
        schema = TRAIN_SCHEMA
    if str(cell.get("capability") or "") != capability:
        raise Round0255PanelError(f"R0255 seed-{seed} cell capability changed")
    receipt_path, receipt_signature = _intra_queue_signature(
        cell["train_receipt"], label=f"R0255 seed-{seed} train receipt"
    )
    receipt = prompt_contract.read_sealed(
        receipt_path, label=f"R0255 seed-{seed} train receipt"
    )
    train_checks = receipt.get("train_checks") or {}
    if (
        receipt.get("schema") != schema
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("treatment_config_round_id") != "0217"
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != seed
        or bool(receipt.get("is_a_family_cell")) is replay
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
        raise Round0255PanelError(f"R0255 seed-{seed} train receipt contract changed")
    if (
        dict(receipt.get("substrate") or {}) != dict(sealed["substrate_signature"])
        or dict(receipt.get("graph_manifest") or {}) != dict(sealed["manifest_signature"])
    ):
        raise Round0255PanelError(
            f"R0255 seed-{seed} was not trained on the substrate this panel scores"
        )
    model_path = prompt_contract.verify_signature(
        receipt["model"], label=f"R0255 seed-{seed} published map"
    )
    return seed, replay, receipt, receipt_signature, model_path


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0255 round0255_nodes.run_panel")
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
        raise Round0255PanelError("R0255 panel handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0255PanelError("R0255 panel scoring requires CUDA")
    node_id = str(active.get("node_id") or "score_minilm_mixed_2m_panel_n29")
    label = "R0255 n=29 panel"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)

    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    corpus_of_row = round0218_nodes._corpus_of_row(sealed)
    panel_evidence, panel_signature = _sealed_panel_evidence(job)
    panel_n16, panel_n16_signature = _sealed_n16_panel(job)

    prior_cells = dict(panel_n16["panel_metric_cells"])
    prior_ratios = dict(panel_n16["raw_purity_ratios"])
    prior_corpus = dict(panel_n16["corpus_ffr_cells"])

    cells_in = job.get("cells")
    if not isinstance(cells_in, list):
        raise Round0255PanelError("R0255 cell input matrix changed")
    family_seeds = {
        int(cell.get("seed", -1))
        for cell in cells_in
        if not bool(cell.get("is_replay_control"))
    }
    if family_seeds != set(SEEDS):
        raise Round0255PanelError("R0255 thirteen-cell input matrix changed")
    authenticated: dict[str, dict[str, Any]] = {}
    for cell in cells_in:
        seed, replay, receipt, receipt_signature, model_path = _authenticate_map(
            cell, sealed
        )
        key = REPLAY_CONTROL_CAPABILITY if replay else str(seed)
        authenticated[key] = {
            "seed": seed,
            "replay": replay,
            "receipt": receipt,
            "receipt_signature": receipt_signature,
            "model_path": model_path,
            "capability": REPLAY_CONTROL_CAPABILITY if replay else capability_for_seed(seed),
        }
    invariants = {
        str(entry["receipt"]["seed_invariant_sha256"])
        for entry in authenticated.values()
    } | {R0217_SEED_INVARIANT_SHA256}
    if len(invariants) != 1:
        raise Round0255PanelError(
            "R0255 pooled family is not commensurate: the new cells do not carry "
            "R0217's seed-invariant config digest"
        )
    prior_model_hashes = dict(job.get("prior_model_sha256_by_seed") or {})
    if set(prior_model_hashes) != {str(seed) for seed in R0250_POOLED_SEEDS}:
        raise Round0255PanelError(
            "R0255 panel needs the sixteen prior checkpoint digests, one per seed "
            "42-57, read from their sealed train receipts at prepare time"
        )
    model_hashes = {
        str(entry["receipt"]["model"]["sha256"])
        for key, entry in authenticated.items()
        if not entry["replay"]
    }
    model_hashes |= {str(value) for value in prior_model_hashes.values()}
    if len(model_hashes) != len(POOLED_SEEDS):
        raise Round0255PanelError(
            f"R0255 pooled family has {len(model_hashes)} distinct checkpoints, "
            f"expected {len(POOLED_SEEDS)}"
        )

    output = create_fresh_directory(str(job["outputs"][0]), label="R0255 n=29 panel")
    started = time.monotonic()
    reset_process_cuda_peak()

    window = ledger.window("R0255 n=29 panel scoring stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0255 panel stage entered")
        wrapped = window.wrap(recorder)

        cfg = prompt_contract.panel_config()
        centroids, centroid_signatures = _load_centroids(panel_evidence)
        wrapped("R0255 frozen centroids loaded")
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
            raise Round0255PanelError(
                f"{REFERENCE_MISMATCH_MESSAGE} file signature drift"
            )
        reference = load_hiD_reference(reference_path)
        wrapped("R0255 high-D reference loaded")
        anchors = sample_anchors(ROWS, cfg)
        if not np.array_equal(
            np.asarray(anchors, dtype=np.int64),
            np.asarray(reference["anchor_ids"], dtype=np.int64),
        ):
            raise Round0255PanelError(f"{REFERENCE_MISMATCH_MESSAGE} anchor drift")
        rederived_key, _parts = hiD_reference_key(
            source, anchors, cfg, centroids, kf=int(reference["kf"]), **reference_identity
        )
        wrapped("R0255 reference key re-derived")
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
            or str(reference["key"]) != str(panel_n16["high_d_reference_key"])
        ):
            raise Round0255PanelError(
                f"{REFERENCE_MISMATCH_MESSAGE} the reference R0218 and R0250 used is "
                "not the one loaded here"
            )
        wrapped("R0255 reference identity proved")

        cells: dict[str, dict[str, Any]] = {}
        for key in [str(seed) for seed in SEEDS] + [REPLAY_CONTROL_CAPABILITY]:
            if key not in authenticated:
                continue
            entry = authenticated[key]
            seed = int(entry["seed"])
            model = ParametricUMAP.load(entry["model_path"], device="cuda")
            coordinates = np.asarray(
                model.transform(source, batch_size=8192), dtype=np.float32
            )
            if coordinates.shape != (ROWS, 2):
                raise Round0255PanelError(
                    f"R0255 {key} transform produced {coordinates.shape}, "
                    f"expected ({ROWS}, 2)"
                )
            if not np.isfinite(coordinates).all():
                raise Round0255PanelError(
                    f"R0255 {key} transform over {ROWS} rows is not finite"
                )
            suffix = "replay-seed42" if entry["replay"] else f"seed{seed}"
            coordinates_path = os.path.join(output, f"coordinates-{suffix}.npy")
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
                    "capability": entry["capability"],
                    "is_a_family_cell": not entry["replay"],
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
                raise Round0255PanelError(f"R0255 {key} panel is collapsed or nonfinite")
            if not bool(panel["provenance"]["hiD_reference_reused"]):
                raise Round0255PanelError(
                    f"{REFERENCE_MISMATCH_MESSAGE} {key} recomputed"
                )
            agreement = assert_hi_d_agreement(seed, panel["purity_numerators"])
            cells[key] = {
                "seed": seed,
                "capability": entry["capability"],
                "is_a_family_cell": not entry["replay"],
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
            wrapped(f"R0255 {key} scored")
        gate.finish("R0255 panel stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    pooled_cells: dict[str, dict[str, float]] = {}
    pooled_ratios: dict[str, dict[str, float]] = {}
    pooled_corpus: dict[str, dict[str, dict[str, float]]] = {}
    for seed in R0250_POOLED_SEEDS:
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
        pooled_cells[key] = dict(cells[key]["panel_metrics"])
        pooled_ratios[key] = dict(cells[key]["raw_purity_ratios"])
        pooled_corpus[key] = dict(cells[key]["corpus_ffr"])

    pooled = pool_twenty_nine_cells(
        cells=pooled_cells,
        ratios=pooled_ratios,
        corpus=pooled_corpus,
        sources=POOLED_CELL_SOURCES,
    )
    summary = descriptive_family_summary({
        metric: [pooled_cells[str(seed)][metric] for seed in POOLED_SEEDS]
        for metric in PANEL_METRICS
    })

    replay: dict[str, Any] = {}
    if REPLAY_CONTROL_CAPABILITY in cells:
        control = cells[REPLAY_CONTROL_CAPABILITY]
        sealed_seed42 = {
            metric: float(prior_cells["42"][metric]) for metric in PANEL_METRICS
        }
        sealed_seed42_ratios = {
            key: float(prior_ratios["42"][key]) for key in ("k256", "k1024")
        }
        replay = replay_control_comparison(
            observed=dict(control["panel_metrics"]),
            sealed_r0218=sealed_seed42,
            observed_ratios=dict(control["raw_purity_ratios"]),
            sealed_ratios=sealed_seed42_ratios,
            tolerance=REPLAY_TOLERANCE,
        )
        replay["checkpoint_comparison"] = dict(
            control["train_receipt"] and authenticated[REPLAY_CONTROL_CAPABILITY][
                "receipt"
            ].get("replay_control")
            or {}
        )
        replay["coordinates_ordered_sha256"] = control["coordinates_ordered_sha256"]
        replay["r0218_seed42_coordinates_ordered_sha256"] = str(
            job.get("r0218_seed42_coordinates_ordered_sha256") or ""
        )
        replay["coordinates_byte_identical_to_r0218"] = (
            str(control["coordinates_ordered_sha256"])
            == str(job.get("r0218_seed42_coordinates_ordered_sha256") or "")
        )

    family_purity = assert_family_is_2m_only(
        [exact_cell_id(seed) for seed in POOLED_SEEDS]
    )
    purity_controls = family_purity_controls()

    execution_checks = {
        "all_thirteen_new_cells_scored": {
            key for key in cells if key != REPLAY_CONTROL_CAPABILITY
        } == {str(seed) for seed in SEEDS},
        "twenty_nine_pooled_cells": len(pooled_cells) == len(POOLED_SEEDS),
        "reaches_the_owner_ruling_n": len(POOLED_SEEDS) == OWNER_RULING_N,
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
        "twenty_nine_distinct_checkpoints": len(model_hashes) == len(POOLED_SEEDS),
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
        "prior_sixteen_cells_not_rescored": (
            {key for key in cells if key != REPLAY_CONTROL_CAPABILITY}
            == {str(seed) for seed in SEEDS}
            and set(prior_cells) == {str(seed) for seed in R0250_POOLED_SEEDS}
        ),
        "the_replay_control_is_not_in_the_pooled_family": (
            REPLAY_CONTROL_CAPABILITY not in pooled_cells
            and bool(family_purity["family_is_the_2m_universe_only"])
        ),
        "every_planted_family_defect_was_refused": bool(
            purity_controls["every_planted_defect_was_refused"]
        ),
        "the_honest_family_still_passes": bool(
            purity_controls["the_honest_family_still_passes"]
        ),
        "no_floor_registered_here": not pooled["gate_registerable_here"],
    }
    if not all(execution_checks.values()):
        raise Round0255PanelError(f"R0255 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0255PanelError(
            f"R0255 panel peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )

    coverage = ledger.receipt()
    body = {
        **pooled,
        **_receipt_envelope(active["manifest"]),
        "schema": PANEL_SCHEMA_N29,
        "capability": PANEL_CAPABILITY_N29,
        "capabilities": [PANEL_CAPABILITY_N29],
        "node": node_id,
        "abort_flag_precondition": abort_flag,
        "panel_evidence": panel_signature,
        "panel_n16_evidence": panel_n16_signature,
        "panel_capability": PANEL_CAPABILITY,
        "panel_n16_capability": PANEL_CAPABILITY_N16,
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
        "new_cells": {str(seed): cells[str(seed)] for seed in SEEDS},
        "replay_control_cell": cells.get(REPLAY_CONTROL_CAPABILITY),
        "replay_control": replay or None,
        "family_purity": family_purity,
        "family_purity_controls": purity_controls,
        "descriptive_family_summary": summary,
        "density_v2_status": DENSITY_V2_STATUS,
        "gap_report": gaps,
        "enforcement_poll_spacing": scored_gate,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gate_registered": False,
        "gate_registerable_here": False,
        "map_decision_made": False,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "outcome": "minilm-mixed-2m-seed-family-pooled-at-n29-on-r0218s-frozen-panel",
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
    }
    _seal(output, "seed-family-panel-n29.json", body)
    print(json.dumps({
        "capability": PANEL_CAPABILITY_N29,
        "n": len(POOLED_SEEDS),
        "reference_byte_identical_to_r0218": True,
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))
    del source, corpus_of_row, reference, centroids
    gc.collect()


# --------------------------------------------------------------------------- #
# the n = 29 registration
# --------------------------------------------------------------------------- #


def calibrate_at_n29(
    sealed_ladder: Sequence[Mapping[str, Any]],
    sealed_n13_candidates: Mapping[str, Any],
    sealed_n16_candidates: Mapping[str, Any],
) -> dict[str, Any]:
    """Calibrate at 29, and check the harness against published outside numbers.

    Four independent external checks, none of them self-consistency: R0234's sealed
    power-parity ladder row at `n = 29`; R0234's sealed calibrated multipliers at
    `n = 13`; R0250's sealed calibrated multipliers at `n = 16`; and the two closed
    forms at `n = 29` -- the noncentral-t one-sided factor and Howe's two-sided
    factor -- against the calibrated sample-sd multipliers, at R0234's own registered
    tolerances.
    """
    at29 = calibration.calibrate(N_EXACT)
    arrays29 = at29.pop("_arrays")
    nct29 = calibration.nct_one_sided_factor(N_EXACT)
    howe29 = calibration.howe_two_sided_factor(N_EXACT)

    ladder29 = calibration.power_ladder(
        OWNER_RULING_ESTIMATOR, sizes=(N_EXACT,)
    )[0]
    sealed29 = next((row for row in sealed_ladder if int(row["n"]) == N_EXACT), None)
    if sealed29 is None:
        raise Round0255GateError(
            f"R0234's sealed power ladder has no n = {N_EXACT} row to check against"
        )

    at13 = calibration.calibrate(13)
    at13.pop("_arrays")
    at16 = calibration.calibrate(16)
    at16.pop("_arrays")

    checks: list[dict[str, Any]] = [
        {
            "key": "r0234_sealed_power_ladder_n29_multiplier",
            "source": "R0234 sealed power_parity_ladder (600,000 families)",
            "target": float(sealed29["calibrated_multiplier"]),
            "observed": float(ladder29["calibrated_multiplier"]),
            "tolerance": 1e-12,
        },
        {
            "key": "r0234_sealed_power_ladder_n29_power_at_minus_2_sigma",
            "source": "R0234 sealed power_parity_ladder (600,000 families)",
            "target": float(sealed29["detection_power_at_minus_2_sigma"]),
            "observed": float(ladder29["detection_power_at_minus_2_sigma"]),
            "tolerance": 1e-12,
        },
        {
            "key": "n29_calibrated_sample_sd_k_vs_nct",
            "source": "closed form: k = t'_{n-1,z*sqrt(n)}(gamma)/sqrt(n) at n = 29",
            "target": 0.0,
            "observed": float(
                at29["candidates"]["mean_minus_k_sample_sd"]["one_sided"][
                    "calibrated_multiplier"
                ]
            )
            - nct29,
            "tolerance": 0.006,
        },
        {
            "key": "n29_calibrated_sample_sd_k2_vs_howe",
            "source": "closed form: Howe (1969) two-sided factor at n = 29",
            "target": 0.0,
            "observed": float(
                at29["candidates"]["mean_minus_k_sample_sd"]["two_sided"][
                    "calibrated_multiplier"
                ]
            )
            - howe29,
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
            checks.append({
                "key": f"r0250_sealed_n16_{side}_k::{name}",
                "source": "R0250 sealed selection.candidates (4,000,000 families)",
                "target": float(sealed_n16_candidates[name][key]),
                "observed": float(
                    at16["candidates"][name][side]["calibrated_multiplier"]
                ),
                "tolerance": 0.0,
            })
    for item in checks:
        item["delta"] = abs(float(item["observed"]) - float(item["target"]))
        item["reproduced"] = bool(item["delta"] <= float(item["tolerance"]))
    if not all(item["reproduced"] for item in checks):
        raise Round0255GateError(
            "R0255 calibration harness did not reproduce every external reference: "
            f"{[item['key'] for item in checks if not item['reproduced']]}"
        )
    return {
        "n": N_EXACT,
        "n29": at29,
        "n13_reproduction": at13,
        "n16_reproduction": at16,
        "arrays": arrays29,
        "closed_forms": {
            "nct_one_sided_factor_at_n29": nct29,
            "howe_two_sided_factor_at_n29": howe29,
            "identity_bound_at_n29": IDENTITY_BOUND_AT_N,
        },
        "power_ladder_n29": ladder29,
        "r0234_sealed_power_ladder_n29": dict(sealed29),
        "external_reference_checks": checks,
        "external_references_all_reproduced": True,
        "content": TOLERANCE_CONTENT,
        "confidence": TOLERANCE_CONFIDENCE,
    }


def evaluate_selection_n29(
    *,
    calibrated: Mapping[str, Any],
    series: Mapping[str, Sequence[float]],
    log_series: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """R0234's pre-registered rule, evaluated at `n = 29` and REPORTED, not applied.

    The owner ruled the estimator. This function exists so the round can say what
    the rule would have chosen, and whether the ruled estimator qualifies under it.
    """
    candidates: dict[str, Any] = {}
    for name in CANDIDATE_ORDER:
        entry = calibrated["n29"]["candidates"][name]
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
        reasoning.append(f"the rule would register: {chosen}")
    else:
        reasoning.append("no candidate qualifies at n = 29 under the rule")
    reasoning.append(
        f"NOT APPLIED: the owner ruled {OWNER_RULING_ESTIMATOR!r} on 2026-08-11 and "
        "this round registers that, whatever the rule would have chosen"
    )
    return {
        "rule": SELECTION_RULE,
        "rule_is_reported_not_applied": True,
        "n": N_EXACT,
        "candidates": candidates,
        "qualifying": qualifying,
        "chosen_estimator": chosen,
        "reasoning": reasoning,
    }


def run_gate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0255 round0255_nodes.run_gate")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0255GateError("R0255 gate handler received another queue")
    node_id = str(active.get("node_id") or "register_calibrated_madn_floors_n29")
    label = "R0255 n=29 calibrated MAD_n floor registration"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0255 gate")

    window = ledger.window("R0255 gate registration stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0255 gate stage entered")
        wrapped = window.wrap(recorder)

        r0234 = prompt_contract.read_sealed(
            _bound_path(job, "r0234_gate", label="R0234 sealed calibrated gate"),
            label="R0234 sealed calibrated gate",
        )
        r0250 = prompt_contract.read_sealed(
            _bound_path(job, "r0250_gate", label="R0250 sealed n=16 gate"),
            label="R0250 sealed n=16 gate",
        )
        wrapped("R0255 prior calibrated gates read")

        # 1. the Gaussian null, before any sealed CELL is opened.
        calibrated = calibrate_at_n29(
            list(r0234["power_parity_ladder"]["ladder"]),
            dict(r0234["selection"]["candidates"]),
            dict(r0250["selection"]["candidates"]),
        )
        wrapped("R0255 calibration at n=29 complete")

        panel = prompt_contract.read_sealed(
            _bound_path(job, "panel_n29", label="R0255 sealed twenty-nine-cell panel"),
            label="R0255 sealed twenty-nine-cell panel",
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
        wrapped("R0255 sealed cells and retained families read")

        exact_cells = panel["panel_metric_cells"]
        exact_ratios = panel["raw_purity_ratios"]
        if tuple(sorted(int(seed) for seed in exact_cells)) != EXACT_FAMILY_SEEDS:
            raise Round0255GateError("R0255 exact family is not seeds 42-70")
        if int(panel["n"]) != N_EXACT or N_EXACT != OWNER_RULING_N:
            raise Round0255GateError(
                f"R0255 requires the twenty-nine-cell panel at n = {OWNER_RULING_N}"
            )
        if panel.get("gate_registerable_here") is not False:
            raise Round0255GateError("R0255's panel claims to register a gate")

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

        # 2. poolability of the thirteen new cells against the sixteen existing.
        poolability = poolability_shift_test(
            panel_metric_cells=exact_cells, raw_purity_ratios=exact_ratios
        )
        wrapped("R0255 poolability test complete")

        # 3. the pre-registered rule at the new n, REPORTED; the ruling APPLIED.
        selection = evaluate_selection_n29(
            calibrated=calibrated, series=series, log_series=log_series
        )
        registration = owner_ruling_registration(selection=selection)
        chosen = registration["registered_estimator"]
        wrapped("R0255 owner ruling applied")

        # 4. every candidate's floors and bands on the twenty-nine cells.
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
                            "panel_v2 rounds each purity ratio to four decimals "
                            "inside the scorer, so this band inherits +/- 5e-5 in r"
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
        wrapped("R0255 every candidate fitted on the twenty-nine cells")

        # 5. the forty-one cells.
        exact_scoring_cells = [
            {
                "cell_id": exact_cell_id(seed),
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
            raise Round0255GateError("R0255 cuVS family is not seeds 42-44")
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
                raise Round0255GateError(f"R0255 candidate arm c{clusters} is not 42-44")
            for seed in CANDIDATE_SEEDS:
                held_out_cells.append({
                    "cell_id": f"cluster-spill-c{clusters}-seed{seed}",
                    "family": f"cluster-spill-c{clusters}",
                    "values": {
                        metric: float(arm[str(seed)][metric]) for metric in METRICS
                    },
                    "ratios": {
                        key: float(arm_ratios[str(seed)][key])
                        for key in ("k256", "k1024")
                    },
                })
        if len(held_out_cells) != N_HELD_OUT:
            raise Round0255GateError("R0255 held-out set is not twelve cells")
        all_cells = list(exact_scoring_cells) + list(held_out_cells)
        defining_ids = [exact_cell_id(seed) for seed in EXACT_FAMILY_SEEDS]
        family_purity = assert_family_is_2m_only(defining_ids)
        purity_controls = family_purity_controls()
        wrapped("R0255 forty-one cells assembled and the family proved 2M-only")

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
                "exact_twenty_nine": score_population(
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
        wrapped("R0255 every candidate scored on forty-one cells")

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
            r0225=r0225, r0231=r0231, r0234=r0234, r0250=r0250, this_round=this_round
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
            estimator=chosen,
            multiplier_one_sided=winner["one_sided_multiplier"],
            multiplier_two_sided=winner["two_sided_multiplier"],
        )
        power = attainability_and_power(
            estimator=chosen,
            n=N_EXACT,
            multiplier_one_sided=winner["one_sided_multiplier"],
            multiplier_two_sided=winner["two_sided_multiplier"],
            calibrated_entry=calibrated["n29"]["candidates"][chosen],
            floors=winner["floors"],
            bands={
                metric: (
                    winner["two_sided_ratio_bands"][metric]["ratio_lower"],
                    winner["two_sided_ratio_bands"][metric]["ratio_upper"],
                )
                for metric in PURITY_METRICS
            },
            series=series,
            log_series=log_series,
        )
        independence = independence_control(
            estimator=chosen,
            multiplier_one_sided=winner["one_sided_multiplier"],
            multiplier_two_sided=winner["two_sided_multiplier"],
            series=series,
            log_series=log_series,
            held_out_cells=held_out_cells,
        )
        changes = verdict_changes(
            chosen=scoring[chosen]["all_cells"],
            published={
                name: entry["scoring"]
                for name, entry in joint["per_family"].items()
                if name != THIS_FAMILY
            },
        )
        wrapped("R0255 joint criteria, power and independence control complete")

        # 6. the c8-seed42 outcome, stated plainly either way.
        c8_row = next(
            row
            for row in scoring[chosen]["all_cells"]["cells"]
            if row["cell_id"] == "cluster-spill-c8-seed42"
        )
        c8_joint = next(
            row for row in joint["cells"] if row["cell_id"] == "cluster-spill-c8-seed42"
        )
        c8_seed42 = {
            "cell_id": "cluster-spill-c8-seed42",
            "is_a_held_out_cell": True,
            "sealed_r0228_ffr": float(
                next(
                    cell for cell in held_out_cells
                    if cell["cell_id"] == "cluster-spill-c8-seed42"
                )["values"]["ffr"]
            ),
            "r0255_n29_ffr_floor": winner["floors"]["ffr"],
            "clears_this_rounds_own_family": bool(c8_row["clears_every_gated_metric"]),
            "per_metric_this_family": c8_row["metrics"],
            "clears_the_joint_criteria": bool(c8_joint["clears_the_joint_criteria"]),
            "joint_binding_failures": c8_joint["binding_failures"],
            "the_ruling": (
                "'The c8-seed42 coupling resolves however the n = 29 fit lands; do "
                "not tune anything to preserve the published pass.' This field is "
                "the answer, whichever way it reads."
            ),
            "no_tuning_statement": NO_TUNING_STATEMENT,
        }
        gate.finish("R0255 gate stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    execution_checks = {
        "n_is_the_owner_ruling_n": N_EXACT == OWNER_RULING_N,
        "every_external_reference_reproduced": bool(
            calibrated["external_references_all_reproduced"]
        ),
        "multiplier_was_derived_at_this_n": (
            float(winner["one_sided_multiplier"])
            != float(
                calibrated["n16_reproduction"]["candidates"][chosen]["one_sided"][
                    "calibrated_multiplier"
                ]
            )
        ),
        "the_registered_estimator_is_the_owner_ruling": (
            chosen == OWNER_RULING_ESTIMATOR
        ),
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
        "the_multiplier_is_below_the_identity_bound": bool(
            falsifiability["one_sided_multiplier_below_the_identity_bound"]
            and falsifiability["two_sided_multiplier_below_the_identity_bound"]
        ),
        "count_is_attainable": bool(joint["count_is_attainable"]),
        "the_family_is_2m_only": bool(family_purity["family_is_the_2m_universe_only"]),
        "every_planted_family_defect_was_refused": bool(
            purity_controls["every_planted_defect_was_refused"]
        ),
        "the_fit_is_independent_of_every_held_out_cell": bool(
            independence["the_fit_is_independent_of_every_held_out_cell"]
        ),
        "the_fit_is_not_inert": bool(independence["the_fit_is_not_inert"]),
        "forty_one_cells_scored": len(all_cells) == N_EXACT + N_HELD_OUT,
    }
    if not all(execution_checks.values()):
        raise Round0255GateError(f"R0255 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    coverage = ledger.receipt()
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": GATE_SCHEMA,
        "capability": GATE_CAPABILITY,
        "capabilities": [GATE_CAPABILITY],
        "node": node_id,
        "abort_flag_precondition": abort_flag,
        "n": N_EXACT,
        "owner_ruling": OWNER_RULING,
        "owner_ruling_n": OWNER_RULING_N,
        "owner_ruling_registration": registration,
        "no_tuning_statement": NO_TUNING_STATEMENT,
        "standing_minimum_n": STANDING_MINIMUM_N,
        "identity_bound_at_n": IDENTITY_BOUND_AT_N,
        "identity_bound_at_n_expression": f"(n-1)/sqrt(n) = 28/sqrt(29)",
        "selection_rule": SELECTION_RULE,
        "selection_reported_not_applied": selection,
        "poolability": poolability,
        "calibration": {
            key: value for key, value in calibrated.items() if key not in {"arrays"}
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
        "attainability_and_detection_power": power,
        "independence_control": independence,
        "family_purity": family_purity,
        "family_purity_controls": purity_controls,
        "gated_metrics": list(GATED_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        "density_v2_defect": DENSITY_V2_DEFECT,
        "density_v2_status": DENSITY_V2_STATUS,
        "scoring_by_candidate": scoring,
        "joint_criteria": joint,
        "joint_criteria_rule": JOINT_CRITERIA_RULE,
        "retained_family_sources": [dict(item) for item in RETAINED_FAMILY_SOURCES],
        "falsifiability": falsifiability,
        "c8_seed42_outcome": c8_seed42,
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
            "r0250_calibrated_gate_n16": expected_input_signature(
                _bound_path(job, "r0250_gate", label="R0250")
            ),
            "r0255_panel_n29": expected_input_signature(
                _bound_path(job, "panel_n29", label="R0255 panel")
            ),
        },
        "supersedes_capability": None,
        "supersession_note": (
            "supersedes NOTHING. Every criterion R0225, R0231, R0234 and R0250 "
            "registered stays in force alongside this one; see joint_criteria. The "
            "owner's ruling replaces the ESTIMATOR for the Phase 3 gate, which is a "
            "statement about what this round registers, not a retraction of an "
            "earlier registration."
        ),
        "gate_status": "registered-and-contingent-pending-review",
        "gate_registered": True,
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gpu_used": False,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "gap_report": gaps,
        "enforcement_poll_spacing": scored_gate,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "peak_host_rss_gib": peak_rss_gib,
        "wall_seconds": time.monotonic() - started,
    })
    _seal(output, "minilm-calibrated-madn-floors-n29.json", body)
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "n": N_EXACT,
        "chosen_estimator": chosen,
        "one_sided_multiplier": winner["one_sided_multiplier"],
        "ffr_floor": winner["floors"]["ffr"],
        "c8_seed42_clears_this_family": c8_seed42["clears_this_rounds_own_family"],
        "cells_clearing_the_joint_criteria": joint["cells_clearing_the_joint_criteria"],
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0255 round0255_nodes.run_job")
    action = str(job.get("action") or "")
    if action == TRAIN_ACTION:
        run_train(active, job)
        return
    if action == PANEL_ACTION:
        run_panel(active, job)
        return
    if action == GATE_ACTION:
        run_gate(active, job)
        return
    raise Round0255Error(f"R0255 unknown action {action!r}")


__all__ = [
    "GATE_ACTION",
    "PANEL_ACTION",
    "TRAIN_ACTION",
    "calibrate_at_n29",
    "evaluate_selection_n29",
    "run_gate",
    "run_job",
    "run_panel",
    "run_train",
]
