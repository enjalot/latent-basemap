"""Execute R0266 — one host-int8 2M cell of the promoted fneg recipe, its panel, and the
map-level int8-fidelity gate.

Three nodes in one queue, reusing R0265's machinery wherever possible:

1. `train_minilm_fneg_2m_hostint8` (GPU) — ONE cell, seed 42, the EXACT R0265 pinned
   recipe plus the single delta `x_residency="host_int8"` (int8 rows + fp16 per-row
   scales on the uniform DeviceEdgeSampler path). The config is built and proved by
   `round0266_int8_treatment`; the model is R0265's `_build_fneg_model` with
   `model.x_residency` set to `host_int8` and asserted; the pipeline-stamp tripwire
   requires `weighted_effective=False`, `positive_sampling="uniform"` AND
   `x_residency="host_int8"` so a silent fp32 fallback fails closed.
2. `score_minilm_fneg_2m_hostint8_panel` (GPU) — the int8 map scored on the SAME
   instruments R0265's panel uses (held-out FFR, purity k256/k1024, collapse, fog),
   via R0265's `score_one_map` and `panel_v2.score_panel`.
3. `register_hostint8_map_fidelity_gate` (CPU) — the map-level fidelity gate. Reads
   (a) R0265's SEALED family floors artifact and requires the int8 cell's five gated
   metrics INSIDE the family bands; (b) R0265's SEALED n=13 panel for the seed-42 fp32
   sibling and the 13-seed observed range per metric, and requires
   `|int8 - fp32_seed42| <= family 13-seed range`. Every band and every range is READ /
   COMPUTED from the sealed inputs at gate time -- never a typed literal. This cell IS
   the map-level int8 fidelity evidence review-0262-01 §H.1 demanded and left unproven.

Nothing in this module signals a process, starts a child, hands cuVS anything, or wraps
a subprocess in a timeout. Every bulk input is a read-only np.memmap.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0217_minilm_2m_seed_family import (
    WARMUP_SUCCESSFUL_UPDATES,
    validate_published_map,
)
from basemap.round0242_locality import json_scrub
from basemap.round0238_rung5 import json_safe
from basemap.round0234_calibrated_floors import identity_bound
from basemap.round0253_coverage import CoverageLedger
from basemap.round0253_stop_hooks import install_stop_hooks

from basemap import round0266_int8_treatment as T
from basemap.round0266_int8_treatment import (
    CAPABILITY,
    ROUND_ID,
    SEED,
    TRAIN_CLOSURE_MODULES,
    X_RESIDENCY,
    Round0266RecipeError,
    assert_registered_int8_recipe,
    assert_runtime_closure_matches_seal,
    int8_train_config,
    runtime_closure_hashes,
    treatment_closure_controls,
)

import experiments.round0218_nodes as round0218_nodes
import experiments.round0265_nodes as R0265N
from experiments.round0230_nodes import CellWatchdog, _open_substrate, _sealed_graph
from experiments.round0263_nodes import (
    _judge_k256_two_sided,
    _judge_one_sided_floor,
)
from experiments.round0264_nodes import (
    VERDICT_NOT_MEASURABLE,
    _judge_collapse,
    _judge_fog,
)

# Reuse R0265's shared node scaffolding verbatim (guards, gate, transform, scoring).
from experiments.round0265_nodes import (
    BATCH_SIZE,  # noqa: F401  (documented reuse)
    COLLAPSE_METRIC,
    DIMENSION,
    FOG_BINS,
    FOG_METRIC,
    FULL_TRANSFORM_BATCH,
    HELDOUT_FFR_METRIC,
    K_TRUE,
    N_PROBES,
    PURITY_K1024_METRIC,
    PURITY_K256_METRIC,
    ROWS,
    _bound_path,
    _build_fneg_model,
    _intra_queue_signature,
    _load_centroids,
    _node_gate,
    _node_guard,
    _score_gate_without_raising,
    _start_node,
    _transform_in_chunks,
    score_one_map,
)
from experiments.round0238_nodes import _check_runner_abort  # noqa: F401
from basemap.round0245_guard import require_enforceable_abort_flag  # noqa: F401
from basemap.round0246_guard import (  # noqa: F401
    require_abort_flag_landed,
    require_live_sampler,
)
from basemap.round0247_registry import registry_fingerprint
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report


TRAIN_ACTION = "train_minilm_fneg_2m_hostint8"
PANEL_ACTION = "score_minilm_fneg_2m_hostint8_panel"
GATE_ACTION = "register_hostint8_map_fidelity_gate"

TRAIN_SCHEMA = "round0266-minilm-fneg-2m-hostint8-train-receipt-v1"
PANEL_CAPABILITY = "minilm-fneg-2m-hostint8-cell-panel-v1"
PANEL_SCHEMA = "round0266-minilm-fneg-2m-hostint8-cell-panel-v1"
GATE_CAPABILITY = "minilm-fneg-2m-hostint8-map-fidelity-gate-v1"
GATE_SCHEMA = "round0266-minilm-fneg-2m-hostint8-map-fidelity-gate-v1"

#: The five gated metrics, named as they key the panel metric table and R0265's floors.
#: Criterion (a) bounds each against a family band; criterion (b) bounds each cell's
#: |int8 - fp32| against the family's 13-seed observed range for that metric.
GATE_METRICS: tuple[str, ...] = (
    HELDOUT_FFR_METRIC, PURITY_K256_METRIC, PURITY_K1024_METRIC, COLLAPSE_METRIC, FOG_METRIC,
)

DEVICE_BUDGET_BYTES = 30 * (1 << 30)
HOST_RSS_LIMIT_GIB = 40.0
POSITIVE_ROWS_PER_UPDATE = 409

SAFETY_NOTE = (
    "no node in this module signals any process, starts a child process, hands cuVS "
    "anything, or wraps a subprocess in a timeout. Every bulk input is a read-only "
    "np.memmap. The per-batch abort read is the release's own ParametricUMAP.abort_poll "
    "attribute, set to this node's recorder and cleared in a finally."
)


class Round0266NodeError(RuntimeError):
    """The R0266 node contract changed."""


# --------------------------------------------------------------------------- #
# shared scaffolding local to R0266 (mirrors R0265's, with ROUND_ID="0266")
# --------------------------------------------------------------------------- #


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


def _guard_tail_reported(watchdog, *, label: str) -> dict[str, Any]:
    return R0265N._guard_tail_reported(watchdog, label=label)


def _closure_evidence(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sealed = prompt_contract.read_sealed(
        _bound_path(job, "treatment_closure", label="R0266 treatment closure seal"),
        label="R0266 treatment closure seal",
    )
    observed = runtime_closure_hashes(TRAIN_CLOSURE_MODULES)
    verdict = assert_runtime_closure_matches_seal(sealed=sealed, observed=observed)
    return sealed, {
        "runtime_closure": observed,
        "verdict": verdict,
        "controls": treatment_closure_controls(sealed=sealed, observed=observed),
    }


def _build_int8_model(config: Mapping[str, Any]):
    """R0265's `_build_fneg_model` plus the one int8 delta: `model.x_residency`.

    `_new_model` predates the x_residency constructor arg (the config->model bridge does
    not thread it), so -- exactly as `_build_fneg_model` sets the fneg fields and the
    manifest posture on the instance -- this sets `model.x_residency = "host_int8"` on
    the built model and asserts it (fail closed). The recipe invariant has already proved
    the config carries the host-int8 delta.
    """
    model = _build_fneg_model(config)
    model.x_residency = X_RESIDENCY
    if getattr(model, "x_residency", None) != X_RESIDENCY:
        raise Round0266NodeError(
            f"R0266 model x_residency is {getattr(model, 'x_residency', None)!r}, "
            f"expected {X_RESIDENCY!r} (the host-int8 delta did not take)"
        )
    return model


# --------------------------------------------------------------------------- #
# the train node — one host-int8 map, seed 42
# --------------------------------------------------------------------------- #


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0266 round0266_nodes.run_train")
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0266NodeError("R0266 train handler received another queue")
    seed = int(job.get("training_seed", -1))
    if seed != SEED:
        raise Round0266NodeError(f"R0266 is a single seed-{SEED} cell; got {seed!r}")
    if str(job.get("capability") or "") != CAPABILITY:
        raise Round0266NodeError("R0266 job capability does not match the int8 cell")
    node_id = str(active.get("node_id") or f"train_hostint8_seed{seed}")
    label = f"R0266 {CAPABILITY}"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)

    closure_seal, closure = _closure_evidence(job)
    if not closure["controls"]["every_planted_defect_was_refused"]:
        raise Round0266NodeError(
            "R0266 closure guard did not refuse every planted defect: "
            f"{closure['controls']['controls']}"
        )
    if not closure["controls"]["the_honest_closure_still_passes"]:
        raise Round0266NodeError("R0266 closure guard rejects the honest closure")

    graph = _sealed_graph(job)
    edges = graph["directed_edges"]
    source, substrate_signature = _open_substrate(graph)
    config, config_sha = int8_train_config(
        seed=seed,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        substrate_signature=substrate_signature,
        graph_edges=edges,
        rows=ROWS,
    )
    recipe = assert_registered_int8_recipe(config)
    observed_invariant = R0265N.fneg_seed_invariant_sha256(config)
    declared_invariant = str(job.get("cell_seed_invariant_sha256") or "")
    if not declared_invariant or observed_invariant != declared_invariant:
        raise Round0266NodeError(
            "R0266 cell config is not the sealed host-int8 recipe: "
            f"{observed_invariant} != {declared_invariant}"
        )
    updates = int(config["optimizer"]["successful_positive_lr_updates"])
    if updates != R0265N.DOSE_MULTIPLIER * int(job.get("base_horizon", -1)):
        raise Round0266NodeError("R0266 horizon does not match the sealed x4 base horizon")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0266 train output")
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": T.INT8_TRAIN_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "seed": seed,
            "capability": CAPABILITY,
            "recipe": recipe,
            "seed_invariant_sha256": observed_invariant,
            "config": config,
            "config_sha256": config_sha,
        },
        immutable=True,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = _build_int8_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")

    window = ledger.window(f"R0266 {CAPABILITY} train stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=True)
    watchdog = CellWatchdog()
    watchdog.start()
    started = time.monotonic()
    try:
        with guard_ctx:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor(f"R0266 {CAPABILITY} stage entered")
            wrapped = window.wrap(recorder)
            model.abort_poll = wrapped
            try:
                model.fit(
                    source,
                    random_state=seed,
                    precomputed_edges_path=graph["signature"]["canonical_path"],
                )
            finally:
                model.abort_poll = None
            wall = time.monotonic() - started
            wrapped("R0266 fit() returned")
            accounting = dict(model._train_stats)
            runtime = dict(getattr(model, "_pipeline_info", None) or {})
            if not runtime:
                raise Round0266NodeError(
                    "R0266 fit() left no _pipeline_info stamp -- cannot prove the sampler"
                )
            # FAIL-CLOSED TRIPWIRE: uniform positive sampling AND the host-int8 residency.
            # A silent fp32 fallback (x_residency=device_fp16) or a weighted regression can
            # never seal this cell.
            if (
                runtime.get("weighted_effective") is not False
                or runtime.get("positive_sampling") != "uniform"
                or runtime.get("x_residency") != X_RESIDENCY
            ):
                raise Round0266NodeError(
                    "R0266 trained off the host-int8 uniform path (silent fallback): "
                    f"weighted_effective={runtime.get('weighted_effective')!r}, "
                    f"positive_sampling={runtime.get('positive_sampling')!r}, "
                    f"x_residency={runtime.get('x_residency')!r}, "
                    f"pipeline={runtime.get('pipeline')!r}"
                )
            fneg_telemetry = dict(model.fneg_telemetry) if model.fneg_telemetry else None
            model_path = os.path.join(output, "model.pt")
            from basemap.output_safety import atomic_build_new_file

            atomic_build_new_file(model_path, model.save, immutable=True)
            wrapped("R0266 checkpoint published")
            free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
            memory = {
                "device_total_bytes": int(total_bytes),
                "post_train_free_bytes": int(free_bytes),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            }
            del model
            torch.cuda.empty_cache()
            gc.collect()
            wrapped("R0266 training objects released")

            from basemap.pumap.parametric_umap import ParametricUMAP

            reloaded = ParametricUMAP.load(model_path, device="cuda")
            checkpoint_fneg_roundtrip = (
                float(reloaded.fneg_weight) == R0265N.FNEG_WEIGHT
                and float(reloaded.fneg_lo) == R0265N.FNEG_LO
                and float(reloaded.fneg_hi) == R0265N.FNEG_HI
            )
            if not checkpoint_fneg_roundtrip:
                raise Round0266NodeError("R0266 checkpoint did not round-trip the fneg params")
            wrapped("R0266 checkpoint reloaded")
            coordinates = _transform_in_chunks(reloaded, source, wrapped)
            validate_published_map(coordinates)
            coordinates_path = os.path.join(output, "coordinates.npy")
            atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
            coordinates_ordered_sha256 = ordered_array_sha256(coordinates)
            transform_rows_finite = int(np.isfinite(coordinates).all(axis=1).sum())
            del reloaded, coordinates
            torch.cuda.empty_cache()
            gc.collect()
            gate.finish(f"R0266 {CAPABILITY} stage end")
        window.close()
        tail = _guard_tail_reported(guard_ctx, label=label)
        scored = _score_gate_without_raising(gate, tail, label=label)
        gaps = gap_report(recorder.records, arm=node_id)
    finally:
        watchdog_state = watchdog.stop()

    if watchdog_state["tripped"]:
        raise Round0266NodeError(
            f"R0266 hostint8 watchdog tripped: {watchdog_state['trip_reason']!r}"
        )
    if int(memory["peak_reserved_bytes"]) > DEVICE_BUDGET_BYTES:
        raise Round0266NodeError("R0266 peak reserved bytes exceed the budget")
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0266NodeError(f"R0266 train peak RSS {peak_rss_gib:.2f} GiB exceeds {HOST_RSS_LIMIT_GIB}")
    memory["peak_host_rss_gib"] = peak_rss_gib

    residency = {
        "x_residency": runtime.get("x_residency"),
        "weighted_effective": runtime.get("weighted_effective"),
        "positive_sampling": runtime.get("positive_sampling"),
        "uniform_with_replacement": runtime.get("uniform_with_replacement"),
        "sampler_pipeline": runtime.get("pipeline"),
        "sampler_class": runtime.get("sampler_class"),
    }
    coverage = ledger.receipt()
    receipt_body = {
        "schema": TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "capabilities": [CAPABILITY],
        "training_seed": seed,
        "is_the_int8_map_fidelity_cell": True,
        "release_sha": active["manifest"]["release_sha"],
        "abort_flag_precondition": abort_flag,
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "seed_invariant_sha256": observed_invariant,
        "recipe": recipe,
        "x_residency": X_RESIDENCY,
        "treatment_closure": closure["verdict"],
        "treatment_closure_controls": closure["controls"],
        "treatment_closure_seal": expected_input_signature(
            _bound_path(job, "treatment_closure", label="R0266 treatment closure seal")
        ),
        "model": expected_input_signature(model_path),
        "coordinates": expected_input_signature(coordinates_path),
        "coordinates_ordered_sha256": coordinates_ordered_sha256,
        "substrate": substrate_signature,
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "rows": ROWS,
        "dimension": DIMENSION,
        "directed_edges": edges,
        "optimizer_updates": updates,
        "base_horizon": int(job.get("base_horizon", -1)),
        "dose_multiplier": R0265N.DOSE_MULTIPLIER,
        "consumed_positive_draws_per_edge": float(updates * POSITIVE_ROWS_PER_UPDATE / edges),
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "host_int8_residency": residency,
        "fneg_telemetry": fneg_telemetry,
        "train_wall_s": wall,
        "memory": memory,
        "memory_watchdog": watchdog_state,
        "gap_report": gaps,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "training_performed": True,
        "gate_registerable_here": False,
        "map_decision_made": False,
        "train_checks": {
            "recipe_is_the_registered_hostint8_recipe": (
                observed_invariant == declared_invariant
            ),
            "every_planted_closure_defect_was_refused": bool(
                closure["controls"]["every_planted_defect_was_refused"]
            ),
            "closure_ran_the_sealed_bytes": bool(
                closure["verdict"]["every_module_ran_the_sealed_bytes"]
            ),
            "fneg_reweighting_was_active": fneg_telemetry is not None,
            "checkpoint_round_trips_fneg_params": checkpoint_fneg_roundtrip,
            "all_2m_coordinates_finite": transform_rows_finite == ROWS,
            "host_int8_residency_stamp_verified": (
                residency["x_residency"] == X_RESIDENCY
                and residency["weighted_effective"] is False
                and residency["positive_sampling"] == "uniform"
            ),
            "watchdog_did_not_trip": not bool(watchdog_state["tripped"]),
            "zero_numerical_skips": (
                int(accounting.get("amp_overflow_skips", 0)) == 0
                and int(accounting.get("nonfinite_loss_skips", 0)) == 0
                and int(accounting.get("nonfinite_gradient_skips", 0)) == 0
            ),
        },
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "node": node_id,
    }
    _seal(output, "train-receipt.json", receipt_body)
    del source, graph
    gc.collect()
    print(json.dumps({
        "capability": CAPABILITY,
        "seed": seed,
        "x_residency": residency["x_residency"],
        "fneg_active": fneg_telemetry is not None,
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


# --------------------------------------------------------------------------- #
# the single-cell panel — reuse R0265's instruments
# --------------------------------------------------------------------------- #


def _authenticate_int8_map(cell: Mapping[str, Any], sealed: Mapping[str, Any]) -> dict[str, Any]:
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != SEED:
        raise Round0266NodeError(f"R0266 cell seed {seed!r} is not the int8 seed-{SEED} cell")
    if str(cell.get("capability") or "") != CAPABILITY:
        raise Round0266NodeError("R0266 cell capability changed")
    receipt_path, receipt_signature = _intra_queue_signature(
        cell["train_receipt"], label="R0266 int8 train receipt"
    )
    receipt = prompt_contract.read_sealed(receipt_path, label="R0266 int8 train receipt")
    train_checks = receipt.get("train_checks") or {}
    if (
        receipt.get("schema") != TRAIN_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("capability") != CAPABILITY
        or int(receipt.get("training_seed", -1)) != SEED
        or receipt.get("training_performed") is not True
        or receipt.get("x_residency") != X_RESIDENCY
        or not train_checks
        or not all(bool(value) for value in train_checks.values())
    ):
        raise Round0266NodeError("R0266 int8 train receipt contract changed")
    if (
        dict(receipt.get("substrate") or {}) != dict(sealed["substrate_signature"])
        or dict(receipt.get("graph_manifest") or {}) != dict(sealed["manifest_signature"])
    ):
        raise Round0266NodeError("R0266 int8 cell was not trained on the panel's substrate")
    model_path = prompt_contract.verify_signature(receipt["model"], label="R0266 int8 map")
    return {
        "seed": SEED,
        "capability": CAPABILITY,
        "receipt": receipt,
        "receipt_signature": receipt_signature,
        "model_path": model_path,
    }


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0266 round0266_nodes.run_panel")
    import torch
    from basemap.panel_v2 import (
        load_hiD_reference,
        reset_process_cuda_peak,
        sample_anchors,
        score_panel,
    )
    from basemap.pumap.parametric_umap import ParametricUMAP

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0266NodeError("R0266 panel handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0266NodeError("R0266 panel scoring requires CUDA")
    node_id = str(active.get("node_id") or PANEL_ACTION)
    label = "R0266 host-int8 cell panel"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)

    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    panel_evidence = prompt_contract.read_sealed(
        str(job["panel_evidence"]), label="R0218 MiniLM 2M frozen panel"
    )
    centroid_ks = [int(k) for k in job["centroid_ks"]]
    cfg = prompt_contract.panel_config()

    probes = np.load(_bound_path(job, "heldout_probes", label="R0266 held-out probes"), allow_pickle=False)
    truth_top10 = np.load(_bound_path(job, "heldout_truth", label="R0266 held-out truth top10"), allow_pickle=False)
    probes = np.asarray(probes, dtype=np.float32)
    truth_top10 = np.asarray(truth_top10, dtype=np.int64)
    if probes.shape != (N_PROBES, DIMENSION) or truth_top10.shape != (N_PROBES, K_TRUE):
        raise Round0266NodeError("R0266 held-out truth geometry changed")

    cell_in = job.get("cell")
    if not isinstance(cell_in, Mapping):
        raise Round0266NodeError("R0266 cell input changed")
    entry = _authenticate_int8_map(cell_in, sealed)

    output = create_fresh_directory(str(job["outputs"][0]), label="R0266 host-int8 panel")
    started = time.monotonic()
    reset_process_cuda_peak()

    window = ledger.window("R0266 host-int8 panel scoring stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0266 panel stage entered")
        wrapped = window.wrap(recorder)

        centroids, centroid_signatures = _load_centroids(panel_evidence, centroid_ks)
        reference_signature = dict(panel_evidence["shared_high_d_reference"])
        reference_path = prompt_contract.verify_signature(reference_signature, label="R0218 shared high-D reference")
        reference = load_hiD_reference(reference_path)
        _anchors = sample_anchors(ROWS, cfg)
        reference_identity = {
            "data_identity": {
                "kind": "ordered_array",
                "shape": [ROWS, DIMENSION],
                "dtype": np.dtype("<f4").str,
                "sha256": sealed["ordered_substrate_sha256"],
            },
            "convention": dict(round0218_nodes.REFERENCE_CONVENTION),
        }
        wrapped("R0266 frozen reference + centroids loaded")

        model = ParametricUMAP.load(entry["model_path"], device="cuda")
        coordinates = np.asarray(model.transform(source, batch_size=FULL_TRANSFORM_BATCH), dtype=np.float32)
        if coordinates.shape != (ROWS, 2) or not np.isfinite(coordinates).all():
            raise Round0266NodeError("R0266 int8 transform is not a finite 2M map")
        panel = score_panel(
            source,
            coordinates,
            config=cfg,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=reference_identity,
            provenance={
                "round_id": ROUND_ID,
                "seed": SEED,
                "capability": entry["capability"],
                "treatment": "fneg-x4-md000-hostint8",
            },
        )
        purity_ratios = {"k256": float(panel["purity"]["k256"]), "k1024": float(panel["purity"]["k1024"])}
        placed = np.asarray(model.transform(probes, batch_size=FULL_TRANSFORM_BATCH), dtype=np.float32)
        scored_map = score_one_map(
            coordinates=coordinates,
            probes_placed=placed,
            truth_top10=truth_top10,
            purity_ratios=purity_ratios,
        )
        cell = {
            "seed": SEED,
            "capability": entry["capability"],
            "train_receipt": dict(entry["receipt_signature"]),
            "model": dict(entry["receipt"]["model"]),
            "x_residency": X_RESIDENCY,
            "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
            "metrics": scored_map,
            "panel_purity_numerators": panel.get("purity_numerators"),
        }
        del model, coordinates, placed
        torch.cuda.empty_cache()
        gc.collect()
        wrapped("R0266 int8 cell scored")
        gate.finish("R0266 panel stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    # The seed-42 cross-check against the sandbox host-int8 arm, reported either way.
    cross_check = R0265N.seed42_cross_check(cell["metrics"])

    metric_summary = {
        "heldout_ffr": cell["metrics"]["heldout_ffr"],
        "purity_fidelity_k256": cell["metrics"]["purity_fidelity_k256"],
        "purity_fidelity_k1024": cell["metrics"]["purity_fidelity_k1024"],
        "collapse": cell["metrics"]["collapse"],
        "fog": cell["metrics"]["fog"],
        "resolution_levels": cell["metrics"]["resolution_levels"],
        "degenerate": cell["metrics"]["degenerate"],
    }
    execution_checks = {
        "int8_cell_scored": True,
        "every_metric_finite": all(
            math.isfinite(float(metric_summary[m]))
            for m in ("heldout_ffr", "collapse", "fog", "purity_fidelity_k256", "purity_fidelity_k1024")
        ),
        "purity_ratios_positive": (
            metric_summary["purity_fidelity_k256"] > 0 and metric_summary["purity_fidelity_k1024"] > 0
        ),
        "no_gate_registered_here": not job.get("gate_registerable_here", False),
    }
    if not all(execution_checks.values()):
        raise Round0266NodeError(f"R0266 panel execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    coverage = ledger.receipt()
    body = {
        **_receipt_envelope(active["manifest"]),
        "schema": PANEL_SCHEMA,
        "capability": PANEL_CAPABILITY,
        "capabilities": [PANEL_CAPABILITY],
        "node": node_id,
        "abort_flag_precondition": abort_flag,
        "seed": SEED,
        "x_residency": X_RESIDENCY,
        "cell": cell,
        "metric_summary": metric_summary,
        "seed42_cross_check": cross_check,
        "reference": reference_signature,
        "centroids": centroid_signatures,
        "gated_metrics": list(GATE_METRICS),
        "lineage": {
            "graph_manifest": dict(sealed["manifest_signature"]),
            "substrate": dict(sealed["substrate_signature"]),
            "ordered_substrate_sha256": sealed["ordered_substrate_sha256"],
        },
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
        "peak_host_rss_gib": peak_rss_gib,
        "performance": {"node_wall_s": time.monotonic() - started},
    }
    _seal(output, "hostint8-cell-panel.json", body)
    print(json.dumps({
        "capability": PANEL_CAPABILITY,
        "seed": SEED,
        "heldout_ffr": cell["metrics"]["heldout_ffr"],
        "collapse": cell["metrics"]["collapse"],
        "fog": cell["metrics"]["fog"],
        "seed42_cross_check_metric_close": cross_check["metric_close"],
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))
    del source, reference, centroids
    gc.collect()


# --------------------------------------------------------------------------- #
# the map-level int8-fidelity gate: criteria (a) + (b), bound from sealed inputs
# --------------------------------------------------------------------------- #


def read_family_bands(floors_path: str) -> dict[str, Any]:
    """Read R0265's five family gate bands from the SEALED floors artifact.

    Every value here is READ from `registered_criteria` in the sealed R0265 gate
    artifact -- NEVER a typed literal. Substituting a hardcoded constant for any read
    would be detected by the constants-discipline contract test, which mutates the
    artifact on disk and asserts the returned band tracks the file.
    """
    d = prompt_contract.read_sealed(floors_path, label="R0265 sealed family floors")
    if (
        d.get("capability") != R0265N.GATE_CAPABILITY
        or d.get("schema") != R0265N.GATE_SCHEMA
        or d.get("gate_registered") is not True
    ):
        raise Round0266NodeError("R0265 sealed family floors contract changed")
    rc = dict(d["registered_criteria"])
    bands = {
        "heldout_ffr_floor": float(rc[HELDOUT_FFR_METRIC]["floor"]),
        "collapse_floor": float(rc[COLLAPSE_METRIC]["floor"]),
        "fog_ceiling": float(rc[FOG_METRIC]["ceiling"]),
        "k1024_floor": float(rc[PURITY_K1024_METRIC]["ratio_floor"]),
        "k256_band": {
            "ratio_lower": float(rc[PURITY_K256_METRIC]["ratio_lower"]),
            "ratio_upper": float(rc[PURITY_K256_METRIC]["ratio_upper"]),
            "log_centre": float(rc[PURITY_K256_METRIC]["log_centre"]),
            "log_scale": float(rc[PURITY_K256_METRIC]["log_scale"]),
        },
    }
    return {
        "bands": bands,
        "source_capability": d.get("capability"),
        "source_schema": d.get("schema"),
        "source_identity_sha256": d.get("identity_sha256"),
    }


def read_fp32_sibling_and_ranges(panel_path: str) -> dict[str, Any]:
    """Read the seed-42 fp32 sibling metrics AND the 13-seed observed range per metric.

    Both are READ / COMPUTED from R0265's SEALED n=13 panel at gate time -- never typed.
    The fp32 sibling is `panel_metric_table["42"]`; the range for each metric is
    `max - min` over all thirteen seed values in the same table.
    """
    d = prompt_contract.read_sealed(panel_path, label="R0265 sealed n=13 panel")
    if d.get("capability") != R0265N.PANEL_CAPABILITY or int(d.get("n", -1)) != R0265N.N_FAMILY:
        raise Round0266NodeError("R0265 sealed n=13 panel contract changed")
    table = dict(d["panel_metric_table"])
    seeds = sorted(int(k) for k in table)
    if SEED not in seeds:
        raise Round0266NodeError("R0265 panel carries no seed-42 fp32 sibling")
    sibling = dict(table[str(SEED)])
    fp32 = {m: float(sibling[m]) for m in GATE_METRICS}
    ranges: dict[str, float] = {}
    per_metric_values: dict[str, list[float]] = {}
    for m in GATE_METRICS:
        vals = [float(table[str(s)][m]) for s in seeds]
        per_metric_values[m] = vals
        ranges[m] = max(vals) - min(vals)
    return {
        "fp32_seed42": fp32,
        "family_13seed_range": ranges,
        "per_metric_values": per_metric_values,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "source_capability": d.get("capability"),
        "source_schema": d.get("schema"),
        "source_identity_sha256": d.get("identity_sha256"),
    }


def _int8_metrics_from_panel(panel: Mapping[str, Any]) -> dict[str, Any]:
    if panel.get("capability") != PANEL_CAPABILITY or panel.get("schema") != PANEL_SCHEMA:
        raise Round0266NodeError("R0266 int8 cell panel contract changed")
    metrics = dict(dict(panel["cell"])["metrics"])
    out = {m: float(metrics[m]) for m in GATE_METRICS}
    out["fog_detail"] = dict(metrics.get("fog_detail") or {})
    out["resolution_levels"] = int(metrics["resolution_levels"])
    out["degenerate"] = bool(metrics["degenerate"])
    return out


def score_criterion_a(*, int8_metrics: Mapping[str, Any], bands: Mapping[str, Any]) -> dict[str, Any]:
    """PRIMARY: the int8 cell's five gated metrics fall INSIDE R0265's family bands.

    Reuses R0265's judge mechanics (`_judge_collapse`, `_judge_fog`,
    `_judge_k256_two_sided`, `_judge_one_sided_floor`) against the FAMILY bands read from
    the sealed floors artifact, plus a one-sided FFR floor. AND the five verdicts.
    """
    ffr = float(int8_metrics[HELDOUT_FFR_METRIC])
    ffr_floor = float(bands["heldout_ffr_floor"])
    ffr_pass = ffr >= ffr_floor

    collapse_v = _judge_collapse(float(int8_metrics[COLLAPSE_METRIC]), float(bands["collapse_floor"]))
    fog_result = {
        "fog": float(int8_metrics[FOG_METRIC]),
        "resolution_levels": int(int8_metrics["resolution_levels"]),
        "degenerate": bool(int8_metrics["degenerate"]),
        "peak_bin_count": dict(int8_metrics.get("fog_detail") or {}).get("peak_bin_count"),
    }
    fog_v = _judge_fog(fog_result, float(bands["fog_ceiling"]))
    k256_v = _judge_k256_two_sided(float(int8_metrics[PURITY_K256_METRIC]), dict(bands["k256_band"]))
    k1024_v = _judge_one_sided_floor(float(int8_metrics[PURITY_K1024_METRIC]), float(bands["k1024_floor"]))

    by_metric = {
        HELDOUT_FFR_METRIC: {
            "verdict": "PASS" if ffr_pass else "FAIL", "passes": ffr_pass,
            "value": ffr, "floor": ffr_floor, "margin": ffr - ffr_floor,
        },
        COLLAPSE_METRIC: collapse_v,
        FOG_METRIC: fog_v,
        PURITY_K256_METRIC: k256_v,
        PURITY_K1024_METRIC: k1024_v,
    }
    passes = all(bool(v["passes"]) for v in by_metric.values())
    return {
        "criterion": "a_inside_family_bands",
        "by_metric": by_metric,
        "passes": passes,
        "bands_used": dict(bands),
        "any_fog_not_measurable": fog_v.get("verdict") == VERDICT_NOT_MEASURABLE,
    }


def score_criterion_b(
    *, int8_metrics: Mapping[str, Any], fp32_metrics: Mapping[str, Any], ranges: Mapping[str, Any]
) -> dict[str, Any]:
    """SECONDARY: |int8 - fp32_seed42| per metric <= the family's 13-seed observed range.

    The int8 perturbation must be no larger than the seed perturbation the family already
    spans. Both the fp32 sibling and every range are read/computed from the sealed R0265
    panel; nothing here is typed.
    """
    by_metric: dict[str, Any] = {}
    for m in GATE_METRICS:
        int8_v = float(int8_metrics[m])
        fp32_v = float(fp32_metrics[m])
        rng = float(ranges[m])
        delta = abs(int8_v - fp32_v)
        by_metric[m] = {
            "int8": int8_v,
            "fp32_seed42": fp32_v,
            "abs_delta": delta,
            "family_13seed_range": rng,
            "passes": delta <= rng,
            "margin": rng - delta,
        }
    passes = all(bool(v["passes"]) for v in by_metric.values())
    return {
        "criterion": "b_delta_within_family_seed_range",
        "by_metric": by_metric,
        "passes": passes,
        "rationale": (
            "the int8 perturbation on each metric must be no larger than the seed "
            "perturbation the R0265 fneg family already spans (its 13-seed max-min range)"
        ),
    }


def run_gate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0266 round0266_nodes.run_gate")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0266NodeError("R0266 gate handler received another queue")
    node_id = str(active.get("node_id") or GATE_ACTION)
    label = "R0266 host-int8 map fidelity gate"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0266 gate")

    window = ledger.window("R0266 gate stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0266 gate stage entered")
        wrapped = window.wrap(recorder)

        # 1. the int8 cell metrics (intra-queue panel).
        int8_panel = prompt_contract.read_sealed(
            _bound_path(job, "int8_panel", label="R0266 int8 cell panel"),
            label="R0266 int8 cell panel",
        )
        int8_metrics = _int8_metrics_from_panel(int8_panel)
        wrapped("R0266 int8 cell metrics read")

        # 2. criterion (a) bands, READ from R0265's sealed floors artifact.
        floors = read_family_bands(
            _bound_path(job, "r0265_floors", label="R0265 sealed family floors")
        )
        bands = floors["bands"]
        criterion_a = score_criterion_a(int8_metrics=int8_metrics, bands=bands)
        wrapped("R0266 criterion (a) scored from sealed family bands")

        # 3. criterion (b) fp32 sibling + 13-seed ranges, READ/COMPUTED from R0265 panel.
        sibling = read_fp32_sibling_and_ranges(
            _bound_path(job, "r0265_panel", label="R0265 sealed n=13 panel")
        )
        criterion_b = score_criterion_b(
            int8_metrics=int8_metrics,
            fp32_metrics=sibling["fp32_seed42"],
            ranges=sibling["family_13seed_range"],
        )
        wrapped("R0266 criterion (b) scored from sealed panel ranges")
        gate.finish("R0266 gate stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    both_pass = bool(criterion_a["passes"] and criterion_b["passes"])
    execution_checks = {
        "criterion_a_bands_read_from_sealed_floors": (
            floors["source_capability"] == R0265N.GATE_CAPABILITY
            and floors["source_schema"] == R0265N.GATE_SCHEMA
        ),
        "criterion_b_ranges_read_from_sealed_panel": (
            sibling["source_capability"] == R0265N.PANEL_CAPABILITY
            and sibling["n_seeds"] == R0265N.N_FAMILY
        ),
        "five_gated_metrics_present": set(criterion_a["by_metric"]) == set(GATE_METRICS),
        "criterion_b_covers_five_metrics": set(criterion_b["by_metric"]) == set(GATE_METRICS),
        "int8_panel_is_host_int8": int8_panel.get("x_residency") == X_RESIDENCY,
        "no_typed_band_literals": True,  # every band/range is a read from a sealed input
    }
    if not all(execution_checks.values()):
        raise Round0266NodeError(f"R0266 gate execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    coverage = ledger.receipt()
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": GATE_SCHEMA,
        "capability": GATE_CAPABILITY,
        "capabilities": [GATE_CAPABILITY],
        "node": node_id,
        "abort_flag_precondition": abort_flag,
        "identity_bound_at_n": identity_bound(R0265N.N_FAMILY),
        "purpose": (
            "the map-level int8 fidelity evidence review-0262-01 §H.1 demanded and left "
            "unproven (R0262 proved fidelity only at the gradient level on a 600-update "
            "proxy). This cell is the BLOCKING precondition before step-5's x2 dose "
            "staged through 50M on the host-int8 path."
        ),
        "int8_cell": {
            "capability": CAPABILITY,
            "x_residency": X_RESIDENCY,
            "panel": expected_input_signature(
                _bound_path(job, "int8_panel", label="R0266 int8 cell panel")
            ),
            "metrics": {m: float(int8_metrics[m]) for m in GATE_METRICS},
        },
        "criterion_a_inside_family_bands": criterion_a,
        "criterion_b_delta_within_family_seed_range": criterion_b,
        "family_bands_source": floors,
        "fp32_sibling_and_ranges_source": {
            "fp32_seed42": sibling["fp32_seed42"],
            "family_13seed_range": sibling["family_13seed_range"],
            "n_seeds": sibling["n_seeds"],
            "seeds": sibling["seeds"],
            "source_capability": sibling["source_capability"],
            "source_schema": sibling["source_schema"],
            "source_identity_sha256": sibling["source_identity_sha256"],
        },
        "pre_registered_pass_criteria": {
            "a": (
                "int8 cell's five gated metrics INSIDE R0265's sealed family bands "
                "(collapse >= floor; fog <= ceiling; FFR >= floor; k1024 >= floor; "
                "k256 in band) -- every band READ from the sealed floors artifact"
            ),
            "b": (
                "|int8 - fp32_seed42| per metric <= the family's 13-seed observed range "
                "(max-min), both READ/COMPUTED from the sealed R0265 panel"
            ),
            "constants_discipline": (
                "no band or range is a typed literal; all are bound from sealed inputs "
                "by sha256 and read at gate time (the R0264/R0265 constants-bug class)"
            ),
        },
        "map_level_int8_fidelity": {
            "criterion_a_passes": criterion_a["passes"],
            "criterion_b_passes": criterion_b["passes"],
            "both_criteria_pass": both_pass,
            "verdict": "MAP_LEVEL_INT8_FIDELITY_ESTABLISHED" if both_pass else "NOT_ESTABLISHED",
        },
        "r0262_map_level_obligation": (
            "review-0262-01 §H.1: 'The train should not run until int8 fidelity is "
            "validated at the map level.' This gate IS that validation for the promoted "
            "fneg recipe, scored against R0265's family bands and 13-seed range."
        ),
        "gate_status": "registered-and-contingent-pending-review",
        "gate_registered": True,
        "evaluation_performed": True,
        "training_performed": False,
        "gpu_used": False,
        "upstream_review_state": dict(job.get("upstream_review_state") or {}),
        "execution_checks": execution_checks,
        "gap_report": gaps,
        "enforcement_poll_spacing": scored_gate,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "peak_host_rss_gib": peak_rss_gib,
        "wall_seconds": time.monotonic() - started,
    })
    _seal(output, "hostint8-map-fidelity-gate.json", body)
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "criterion_a_passes": criterion_a["passes"],
        "criterion_b_passes": criterion_b["passes"],
        "both_criteria_pass": both_pass,
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0266 round0266_nodes.run_job")
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
    raise Round0266NodeError(f"R0266 unknown action {action!r}")


__all__ = [
    "GATE_ACTION",
    "GATE_CAPABILITY",
    "GATE_METRICS",
    "GATE_SCHEMA",
    "PANEL_ACTION",
    "PANEL_CAPABILITY",
    "PANEL_SCHEMA",
    "Round0266NodeError",
    "TRAIN_ACTION",
    "TRAIN_SCHEMA",
    "read_family_bands",
    "read_fp32_sibling_and_ranges",
    "run_gate",
    "run_job",
    "run_panel",
    "run_train",
    "score_criterion_a",
    "score_criterion_b",
]
