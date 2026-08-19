"""Train and score the matched 50M fneg-off control.

The heavy data path is deliberately shared with R0267: the sealed R0237 graph
and substrate, the verified first-50M prefix of R0262's host-int8 substrate,
the model bridge, chunked transform, and quality instruments are the same
implementation.  ``basemap.baseline_50m_fneg_off`` proves the sole treatment
difference is ``fneg_weight=0``.
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
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap import baseline_50m_fneg_off as C
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0217_minilm_2m_seed_family import (
    WARMUP_SUCCESSFUL_UPDATES,
    validate_published_map,
)
from basemap.round0238_rung5 import json_safe
from basemap.round0242_locality import json_scrub
from basemap.round0247_registry import registry_fingerprint
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report
from basemap.round0253_coverage import CoverageLedger
from basemap.round0253_stop_hooks import install_stop_hooks
from experiments.round0230_nodes import CellWatchdog
from experiments.round0265_nodes import (
    DIMENSION,
    FULL_TRANSFORM_BATCH,
    _bound_path,
    _intra_queue_signature,
    _node_gate,
    _node_guard,
    _score_gate_without_raising,
    _start_node,
    score_one_map,
)
from experiments.round0267_nodes import (
    DESCRIPTIVE_PURITY_LINEAGE_CAVEAT,
    HOST_RSS_LIMIT_GIB,
    PREFIX_ROWS,
    R0267_ANON_BUDGET_BYTES,
    _build_int8_50m_model,
    _build_prefix_purity_centroids,
    _guard_tail_reported,
    _open_50m_substrate,
    _read_int8_slice_manifest,
    _sealed_50m_graph,
    _sealed_50m_substrate,
    _transform_50m_in_chunks,
    build_hostint8_dataset_from_slice,
)


TRAIN_ACTION = "train_minilm_fneg_off_50m_x2_hostint8"
PANEL_ACTION = "score_minilm_fneg_off_50m_x2_panel"
COMPARE_ACTION = "compare_minilm_50m_fneg_off_to_fneg"

TRAIN_SCHEMA = "baseline-minilm-fneg-off-50m-x2-hostint8-train-receipt-v1"
PANEL_SCHEMA = "baseline-minilm-fneg-off-50m-x2-hostint8-panel-v1"
COMPARE_SCHEMA = "baseline-minilm-50m-fneg-off-vs-fneg-comparison-v1"
PANEL_CAPABILITY = "minilm-fneg-off-50m-x2-hostint8-panel-v1"
COMPARE_CAPABILITY = "minilm-50m-fneg-off-vs-fneg-comparison-v1"

DEVICE_BUDGET_BYTES = 30 * (1 << 30)
POSITIVE_ROWS_PER_UPDATE = 409
SAFETY_NOTE = (
    "nodes do not signal another process, start a child process, or hand cuVS data. "
    "Bulk inputs are read-only memmaps; the only treatment delta from R0267 is "
    "optimizer.fneg_weight=0."
)


class Baseline50MNodeError(RuntimeError):
    """The matched-control execution contract changed."""


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round_id": C.ROUND_ID,
        "study_id": C.STUDY_ID,
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


def _closure_evidence(job: Mapping[str, Any]) -> dict[str, Any]:
    path = _bound_path(job, "treatment_closure", label="50M fneg-off treatment closure")
    sealed = prompt_contract.read_sealed(path, label="50M fneg-off treatment closure")
    observed = C.runtime_closure_hashes()
    verdict = C.assert_runtime_closure_matches_seal(sealed=sealed, observed=observed)
    controls = C.treatment_closure_controls(sealed=sealed, observed=observed)
    if not controls["every_planted_defect_was_refused"]:
        raise Baseline50MNodeError("50M fneg-off closure controls did not all refuse")
    if not controls["the_honest_closure_still_passes"]:
        raise Baseline50MNodeError("50M fneg-off closure rejected its sealed source")
    return {"verdict": verdict, "controls": controls, "runtime_closure": observed}


def _seed(job: Mapping[str, Any]) -> int:
    seed = job.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in C.SEEDS:
        raise Baseline50MNodeError(f"invalid 50M fneg-off seed {seed!r}")
    if str(job.get("capability") or "") != C.capability_for_seed(seed):
        raise Baseline50MNodeError("50M fneg-off capability does not match its seed")
    return seed


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="baseline_50m_fneg_off_nodes.run_train")
    import torch

    if active.get("manifest", {}).get("round_id") != C.ROUND_ID:
        raise Baseline50MNodeError("50M fneg-off handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Baseline50MNodeError("50M fneg-off train requires a leased CUDA device")
    if not torch.cuda.is_available():
        raise Baseline50MNodeError("50M fneg-off train environment cannot see CUDA")
    seed = _seed(job)
    capability = C.capability_for_seed(seed)
    node_id = str(active.get("node_id") or f"{TRAIN_ACTION}_seed{seed}")
    label = f"50M fneg-off control seed {seed}"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    closure = _closure_evidence(job)

    graph = _sealed_50m_graph(job)
    substrate = _sealed_50m_substrate(job)
    source = _open_50m_substrate(substrate)
    int8_slice = _read_int8_slice_manifest(job)
    config, config_sha = C.control_train_config(
        seed=seed,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        substrate_signature=substrate["substrate_signature"],
        graph_edges=graph["directed_edges"],
        rows=C.ROWS,
    )
    recipe = C.assert_registered_control(config)
    invariant = C.control_seed_invariant_sha256(config)
    if invariant != str(job.get("cell_seed_invariant_sha256") or ""):
        raise Baseline50MNodeError("50M fneg-off seed-invariant digest changed")
    updates = int(config["optimizer"]["successful_positive_lr_updates"])
    if updates != C.DOSE_MULTIPLIER * int(job.get("base_horizon", -1)):
        raise Baseline50MNodeError("50M fneg-off horizon is not the matched x2 dose")

    output = create_fresh_directory(str(job["outputs"][0]), label=label)
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": C.TRAIN_CONFIG_SCHEMA,
            "round_id": C.ROUND_ID,
            "study_id": C.STUDY_ID,
            "seed": seed,
            "capability": capability,
            "recipe": recipe,
            "seed_invariant_sha256": invariant,
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
    model = _build_int8_50m_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")

    int8_dataset, int8_receipt = build_hostint8_dataset_from_slice(
        int8_slice["manifest"], model.device
    )
    if tuple(int8_dataset.shape) != (C.ROWS, DIMENSION):
        raise Baseline50MNodeError("verified host-int8 slice has the wrong geometry")

    window = ledger.window(f"{label} train stage")
    guard_ctx = _node_guard(label, anonymous_budget_bytes=R0267_ANON_BUDGET_BYTES)
    gate = _node_gate(label, training_performed=True)
    watchdog = CellWatchdog()
    watchdog.start()
    started = time.monotonic()
    try:
        with guard_ctx:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor(f"{label} entered")
            poll = window.wrap(recorder)
            model.abort_poll = poll
            try:
                model.fit(
                    int8_dataset,
                    random_state=seed,
                    precomputed_edges_path=graph["signature"]["canonical_path"],
                )
            finally:
                model.abort_poll = None
            train_wall_s = time.monotonic() - started
            poll(f"{label} fit returned")
            accounting = dict(model._train_stats)
            runtime = dict(getattr(model, "_pipeline_info", None) or {})
            if (
                runtime.get("weighted_effective") is not False
                or runtime.get("positive_sampling") != "uniform"
                or runtime.get("x_residency") != C.X_RESIDENCY
            ):
                raise Baseline50MNodeError(
                    f"50M fneg-off control left the matched sampler/residency path: {runtime}"
                )
            if model.fneg_telemetry is not None:
                raise Baseline50MNodeError("fneg-off control emitted active fneg telemetry")
            fneg_reweighting_was_inactive = model.fneg_telemetry is None

            from basemap.output_safety import atomic_build_new_file

            model_path = os.path.join(output, "model.pt")
            atomic_build_new_file(model_path, model.save, immutable=True)
            poll(f"{label} checkpoint published")
            free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
            memory = {
                "device_total_bytes": int(total_bytes),
                "post_train_free_bytes": int(free_bytes),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            }
            del model, int8_dataset
            torch.cuda.empty_cache()
            gc.collect()
            poll(f"{label} training objects released")

            from basemap.pumap.parametric_umap import ParametricUMAP

            reloaded = ParametricUMAP.load(model_path, device="cuda")
            checkpoint_control_roundtrip = (
                float(reloaded.fneg_weight) == C.FNEG_WEIGHT
                and float(reloaded.fneg_lo) == float(config["optimizer"]["fneg_lo"])
                and float(reloaded.fneg_hi) == float(config["optimizer"]["fneg_hi"])
            )
            if not checkpoint_control_roundtrip:
                raise Baseline50MNodeError("checkpoint did not preserve the fneg-off control")
            coordinates = _transform_50m_in_chunks(reloaded, source, poll)
            validate_published_map(coordinates)
            coordinates_path = os.path.join(output, "coordinates.npy")
            atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
            coordinates_ordered_sha256 = ordered_array_sha256(coordinates)
            finite_rows = int(np.isfinite(coordinates).all(axis=1).sum())
            del reloaded, coordinates
            torch.cuda.empty_cache()
            gc.collect()
            gate.finish(f"{label} stage end")
        window.close()
        tail = _guard_tail_reported(guard_ctx, label=label)
        gate_receipt = _score_gate_without_raising(gate, tail, label=label)
        gaps = gap_report(recorder.records, arm=node_id)
    finally:
        watchdog_state = watchdog.stop()

    if watchdog_state["tripped"]:
        raise Baseline50MNodeError(
            f"50M fneg-off seed {seed} watchdog tripped: {watchdog_state['trip_reason']!r}"
        )
    if int(memory["peak_reserved_bytes"]) > DEVICE_BUDGET_BYTES:
        raise Baseline50MNodeError("50M fneg-off device-memory budget exceeded")
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Baseline50MNodeError(
            f"50M fneg-off peak RSS {peak_rss_gib:.2f} GiB exceeds {HOST_RSS_LIMIT_GIB}"
        )
    memory["peak_host_rss_gib"] = peak_rss_gib
    coverage = ledger.receipt()
    checks = {
        "one_axis_control_recipe_verified": recipe["only_treatment_delta"]
        == {"path": "optimizer.fneg_weight", "parent": 1.0, "control": 0.0},
        "fneg_reweighting_was_inactive": fneg_reweighting_was_inactive,
        "checkpoint_round_trips_control": checkpoint_control_roundtrip,
        "pre_sealed_int8_slice_verified": bool(
            int8_receipt.get("verified_against_sealed_manifest")
            and int8_receipt.get("re_encoded_at_train_time") is False
        ),
        "all_50m_coordinates_finite": finite_rows == C.ROWS,
        "matched_uniform_host_int8_path": (
            runtime.get("x_residency") == C.X_RESIDENCY
            and runtime.get("weighted_effective") is False
            and runtime.get("positive_sampling") == "uniform"
        ),
        "successful_update_budget_satisfied": bool(accounting.get("budget_satisfied")),
        "zero_numerical_skips": (
            int(accounting.get("amp_overflow_skips", 0)) == 0
            and int(accounting.get("nonfinite_loss_skips", 0)) == 0
            and int(accounting.get("nonfinite_gradient_skips", 0)) == 0
        ),
        "watchdog_did_not_trip": not bool(watchdog_state["tripped"]),
    }
    if not all(checks.values()):
        raise Baseline50MNodeError(f"50M fneg-off train checks failed: {checks}")

    body = {
        **_receipt_envelope(active["manifest"]),
        "schema": TRAIN_SCHEMA,
        "capability": capability,
        "training_seed": seed,
        "training_performed": True,
        "evaluation_performed": False,
        "control_of": "R0267 50M x2 host-int8 fneg treatment",
        "only_treatment_delta": recipe["only_treatment_delta"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "seed_invariant_sha256": invariant,
        "recipe": recipe,
        "treatment_closure": closure,
        "model": expected_input_signature(model_path),
        "coordinates": expected_input_signature(coordinates_path),
        "coordinates_ordered_sha256": coordinates_ordered_sha256,
        "substrate": substrate["substrate_signature"],
        "substrate_manifest": substrate["manifest_signature"],
        "ordered_substrate_sha256": substrate["ordered_substrate_sha256"],
        "graph": graph["signature"],
        "graph_manifest": graph["manifest_signature"],
        "int8_substrate_manifest": int8_slice["manifest_signature"],
        "int8_substrate_slice": int8_receipt,
        "rows": C.ROWS,
        "dimension": DIMENSION,
        "directed_edges": graph["directed_edges"],
        "optimizer_updates": updates,
        "base_horizon": int(job["base_horizon"]),
        "dose_multiplier": C.DOSE_MULTIPLIER,
        "consumed_positive_draws_per_edge": float(
            updates * POSITIVE_ROWS_PER_UPDATE / graph["directed_edges"]
        ),
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "fneg_telemetry": None,
        "train_wall_s": train_wall_s,
        "memory": memory,
        "memory_watchdog": watchdog_state,
        "gap_report": gaps,
        "enforcement_poll_spacing": gate_receipt,
        "guard_tail": tail,
        "train_checks": checks,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "node": node_id,
    }
    _seal(output, "train-receipt.json", body)
    del source, graph
    gc.collect()
    print(json.dumps({"capability": capability, "seed": seed, "fneg_active": False}))


def _authenticate_cell(
    cell: Mapping[str, Any], substrate: Mapping[str, Any]
) -> dict[str, Any]:
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in C.SEEDS:
        raise Baseline50MNodeError(f"invalid panel seed {seed!r}")
    capability = C.capability_for_seed(seed)
    if str(cell.get("capability") or "") != capability:
        raise Baseline50MNodeError("panel cell capability does not match its seed")
    path, signature = _intra_queue_signature(
        cell["train_receipt"], label=f"50M fneg-off seed {seed} train receipt"
    )
    receipt = prompt_contract.read_sealed(path, label=f"50M fneg-off seed {seed} receipt")
    checks = dict(receipt.get("train_checks") or {})
    if (
        receipt.get("schema") != TRAIN_SCHEMA
        or receipt.get("round_id") != C.ROUND_ID
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != seed
        or receipt.get("training_performed") is not True
        or receipt.get("only_treatment_delta")
        != {"path": "optimizer.fneg_weight", "parent": 1.0, "control": 0.0}
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise Baseline50MNodeError(f"seed {seed} train receipt contract changed")
    if receipt.get("ordered_substrate_sha256") != substrate["ordered_substrate_sha256"]:
        raise Baseline50MNodeError(f"seed {seed} was trained on another substrate")
    return {
        "seed": seed,
        "capability": capability,
        "receipt": receipt,
        "receipt_signature": signature,
        "model_path": prompt_contract.verify_signature(
            receipt["model"], label=f"50M fneg-off seed {seed} model"
        ),
        "coordinates_path": prompt_contract.verify_signature(
            receipt["coordinates"], label=f"50M fneg-off seed {seed} coordinates"
        ),
    }


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="baseline_50m_fneg_off_nodes.run_panel")
    import torch
    from basemap.panel_v2 import reset_process_cuda_peak, score_panel
    from basemap.pumap.parametric_umap import ParametricUMAP

    if active.get("manifest", {}).get("round_id") != C.ROUND_ID:
        raise Baseline50MNodeError("50M fneg-off panel received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Baseline50MNodeError("50M fneg-off panel requires CUDA")
    node_id = str(active.get("node_id") or PANEL_ACTION)
    label = "50M fneg-off matched-control panel"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)

    substrate = _sealed_50m_substrate(job)
    source = _open_50m_substrate(substrate)
    reserve = np.load(
        _bound_path(job, "heldout_reserve", label="50M held-out reserve"),
        mmap_mode="r",
        allow_pickle=False,
    )
    reserve_rows = np.load(
        _bound_path(job, "reserve_query_rows", label="50M reserve query rows"),
        allow_pickle=False,
    ).astype(np.int64, copy=False)
    reserve_embeddings = np.asarray(reserve[reserve_rows], dtype=np.float32)
    reserve_truth = np.load(
        _bound_path(job, "reserve_truth", label="50M reserve-neighbour truth"),
        allow_pickle=False,
    ).astype(np.int64, copy=False)
    if reserve_embeddings.ndim != 2 or reserve_embeddings.shape[1] != DIMENSION:
        raise Baseline50MNodeError("50M held-out reserve geometry changed")
    if reserve_truth.ndim != 2 or reserve_truth.shape[0] != reserve_embeddings.shape[0]:
        raise Baseline50MNodeError("50M reserve truth geometry changed")

    cells_in = job.get("cells")
    if not isinstance(cells_in, list) or not cells_in:
        raise Baseline50MNodeError("50M fneg-off panel has no cells")
    authenticated = [_authenticate_cell(cell, substrate) for cell in cells_in]
    seeds = sorted(entry["seed"] for entry in authenticated)
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset(C.SEEDS):
        raise Baseline50MNodeError("50M fneg-off panel seed set is invalid")
    invariants = {entry["receipt"]["seed_invariant_sha256"] for entry in authenticated}
    if len(invariants) != 1:
        raise Baseline50MNodeError("50M fneg-off panel mixes recipes")

    output = create_fresh_directory(str(job["outputs"][0]), label=label)
    reset_process_cuda_peak()
    started = time.monotonic()
    window = ledger.window(f"{label} scoring stage")
    guard_ctx = _node_guard(label, anonymous_budget_bytes=R0267_ANON_BUDGET_BYTES)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor(f"{label} entered")
        poll = window.wrap(recorder)
        centroid_ks = [int(value) for value in job["centroid_ks"]]
        centroids, centroid_signatures = _build_prefix_purity_centroids(
            source[:PREFIX_ROWS],
            centroid_ks,
            cache_dir=os.path.join(output, "descriptive-prefix-centroids"),
        )
        poll("50M fneg-off descriptive purity centroids built")

        cfg = prompt_contract.panel_config()
        cells: dict[str, dict[str, Any]] = {}
        for entry in sorted(authenticated, key=lambda value: value["seed"]):
            seed = entry["seed"]
            coordinates = np.load(entry["coordinates_path"], mmap_mode="r", allow_pickle=False)
            if coordinates.shape != (C.ROWS, 2) or coordinates.dtype != np.float32:
                raise Baseline50MNodeError(f"seed {seed} coordinates have changed geometry")
            if not np.isfinite(coordinates).all():
                raise Baseline50MNodeError(f"seed {seed} coordinates are not finite")
            model = ParametricUMAP.load(entry["model_path"], device="cuda")
            if float(model.fneg_weight) != C.FNEG_WEIGHT:
                raise Baseline50MNodeError(f"seed {seed} panel loaded an fneg-active model")
            purity_panel = score_panel(
                source[:PREFIX_ROWS],
                coordinates[:PREFIX_ROWS],
                config=cfg,
                centroids_by_k=centroids,
                hiD_reference=None,
                provenance={
                    "round_id": C.ROUND_ID,
                    "study_id": C.STUDY_ID,
                    "seed": seed,
                    "capability": entry["capability"],
                    "treatment": "fneg-off-x2-md000-hostint8-50m",
                    "pass": "r0237-prefix-descriptive-purity",
                    "descriptive": True,
                    "gated": False,
                    "lineage_caveat": DESCRIPTIVE_PURITY_LINEAGE_CAVEAT,
                },
            )
            purity = {
                "k256": float(purity_panel["purity"]["k256"]),
                "k1024": float(purity_panel["purity"]["k1024"]),
            }
            placed = np.asarray(
                model.transform(reserve_embeddings, batch_size=FULL_TRANSFORM_BATCH),
                dtype=np.float32,
            )
            metrics = score_one_map(
                coordinates=coordinates,
                probes_placed=placed,
                truth_top10=reserve_truth,
                purity_ratios=purity,
                disc=int(C.ROWS * 0.001),
            )
            cells[str(seed)] = {
                "seed": seed,
                "capability": entry["capability"],
                "train_receipt": entry["receipt_signature"],
                "model": entry["receipt"]["model"],
                "coordinates": entry["receipt"]["coordinates"],
                "coordinates_ordered_sha256": entry["receipt"][
                    "coordinates_ordered_sha256"
                ],
                "seed_invariant_sha256": entry["receipt"]["seed_invariant_sha256"],
                "metrics": metrics,
                "descriptive_purity": {
                    "descriptive": True,
                    "gated": False,
                    "prefix_rows": PREFIX_ROWS,
                    "reference": "r0237-prefix-inline",
                    "k256": purity["k256"],
                    "k1024": purity["k1024"],
                    "lineage_caveat": DESCRIPTIVE_PURITY_LINEAGE_CAVEAT,
                },
            }
            del model, coordinates, placed
            torch.cuda.empty_cache()
            gc.collect()
            poll(f"50M fneg-off seed {seed} scored")
        gate.finish(f"{label} stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    gate_receipt = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)
    coverage = ledger.receipt()

    metric_table = {
        str(seed): {
            key: cells[str(seed)]["metrics"][key]
            for key in (
                "heldout_ffr",
                "regressor_ffr",
                "net_minus_regressor",
                "purity_fidelity_k256",
                "purity_fidelity_k1024",
                "collapse",
                "fog",
                "resolution_levels",
                "degenerate",
            )
        }
        for seed in seeds
    }
    checks = {
        "every_requested_seed_scored": set(cells) == {str(seed) for seed in seeds},
        "one_recipe": len(invariants) == 1,
        "every_gated_metric_finite": all(
            math.isfinite(float(row[key]))
            for row in metric_table.values()
            for key in ("heldout_ffr", "collapse", "fog")
        ),
        "result_is_descriptive_not_a_gate": job.get("gate_registerable_here") is False,
    }
    if not all(checks.values()):
        raise Baseline50MNodeError(f"50M fneg-off panel checks failed: {checks}")

    body = {
        **_receipt_envelope(active["manifest"]),
        "schema": PANEL_SCHEMA,
        "capability": PANEL_CAPABILITY,
        "node": node_id,
        "abort_flag_precondition": abort_flag,
        "n": len(seeds),
        "seeds": seeds,
        "control_of": "R0267 50M x2 host-int8 fneg treatment",
        "only_treatment_delta": {
            "path": "optimizer.fneg_weight",
            "parent": 1.0,
            "control": 0.0,
        },
        "seed_invariant_sha256": next(iter(invariants)),
        "panel_metric_table": metric_table,
        "cells": cells,
        "descriptive_purity_centroids": centroid_signatures,
        "descriptive_purity": {
            "descriptive": True,
            "gated": False,
            "prefix_rows": PREFIX_ROWS,
            "reference": "r0237-prefix-inline",
            "lineage_caveat": DESCRIPTIVE_PURITY_LINEAGE_CAVEAT,
        },
        "heldout_ffr_instrument": {
            "probe_design": "out_of_substrate_reserve_projection",
            "disc": int(C.ROWS * 0.001),
            "disc_rule": "int(rows * 0.001)",
            "n_reserve_probes": int(reserve_embeddings.shape[0]),
            "reserve_query_rows": dict(job["reserve_query_rows"]),
            "reserve_truth": dict(job["reserve_truth"]),
        },
        "lineage": {
            "substrate_manifest": substrate["manifest_signature"],
            "substrate": substrate["substrate_signature"],
            "ordered_substrate_sha256": substrate["ordered_substrate_sha256"],
        },
        "execution_checks": checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gate_registered": False,
        "gate_registerable_here": False,
        "gap_report": gaps,
        "enforcement_poll_spacing": gate_receipt,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "performance": {"node_wall_s": time.monotonic() - started},
    }
    _seal(output, "fneg-off-50m-x2-panel.json", body)
    print(json.dumps({"capability": PANEL_CAPABILITY, "seeds": seeds}))


def _treated_metric_table(gate: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    rows = ((gate.get("criteria_1_3_per_seed_backstops") or {}).get("cells") or [])
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        seed = str(int(row["seed"]))
        metrics = dict(row["metrics"])
        purity = dict(row.get("descriptive_purity") or {})
        table[seed] = {
            "heldout_ffr": float(metrics["heldout_ffr"]["value"]),
            "collapse": float(metrics["collapse"]["value"]),
            "fog": float(metrics["fog"]["value"]),
            "purity_fidelity_k256": float(purity["purity_fidelity_k256"]["raw_ratio"]),
            "purity_fidelity_k1024": float(purity["purity_fidelity_k1024"]["raw_ratio"]),
        }
    return table


def run_compare(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="baseline_50m_fneg_off_nodes.run_compare")
    if active.get("manifest", {}).get("round_id") != C.ROUND_ID:
        raise Baseline50MNodeError("50M fneg-off comparison received another queue")
    abort_flag = _start_node("50M fneg-off comparison")
    panel = prompt_contract.read_sealed(
        _bound_path(job, "control_panel", label="50M fneg-off panel"),
        label="50M fneg-off panel",
    )
    treated_gate = prompt_contract.read_sealed(
        _bound_path(job, "treated_gate", label="R0267 superseding 50M gate"),
        label="R0267 superseding 50M gate",
    )
    if panel.get("schema") != PANEL_SCHEMA or panel.get("gate_registered") is not False:
        raise Baseline50MNodeError("50M fneg-off comparison received another panel")
    if (treated_gate.get("fifty_m_decision") or {}).get("verdict") != "50M_PASS":
        raise Baseline50MNodeError("bound R0267 artifact is not the superseding 50M pass")
    control = dict(panel["panel_metric_table"])
    treated = _treated_metric_table(treated_gate)
    seeds = sorted(set(control) & set(treated), key=int)
    if seeds != sorted(control, key=int):
        raise Baseline50MNodeError("R0267 comparison is missing a requested control seed")
    metrics = (
        "heldout_ffr",
        "collapse",
        "fog",
        "purity_fidelity_k256",
        "purity_fidelity_k1024",
    )
    paired = {
        seed: {
            metric: {
                "fneg_off": float(control[seed][metric]),
                "fneg_on": float(treated[seed][metric]),
                "off_minus_on": float(control[seed][metric]) - float(treated[seed][metric]),
            }
            for metric in metrics
        }
        for seed in seeds
    }
    means = {
        metric: {
            arm: statistics.mean(paired[seed][metric][arm] for seed in seeds)
            for arm in ("fneg_off", "fneg_on", "off_minus_on")
        }
        for metric in metrics
    }
    output = create_fresh_directory(str(job["outputs"][0]), label="50M control comparison")
    _seal(
        output,
        "fneg-off-vs-fneg-50m.json",
        {
            **_receipt_envelope(active["manifest"]),
            "schema": COMPARE_SCHEMA,
            "capability": COMPARE_CAPABILITY,
            "abort_flag_precondition": abort_flag,
            "seeds": [int(seed) for seed in seeds],
            "n": len(seeds),
            "paired_by_seed": paired,
            "mean_by_metric": means,
            "difference_convention": "fneg_off - fneg_on",
            "interpretation": {
                "heldout_ffr": "higher is better",
                "collapse": "lower can indicate collapse; use the registered floor",
                "fog": "lower is better",
                "purity": "descriptive only at 50M",
            },
            "decision_made": False,
            "gate_registered": False,
            "training_performed": False,
            "evaluation_performed": True,
            "control_panel": dict(job["control_panel"]),
            "treated_gate": dict(job["treated_gate"]),
        },
    )
    print(json.dumps({"capability": COMPARE_CAPABILITY, "seeds": seeds}))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="baseline_50m_fneg_off_nodes.run_job")
    action = str(job.get("action") or "")
    if action == TRAIN_ACTION:
        run_train(active, job)
    elif action == PANEL_ACTION:
        run_panel(active, job)
    elif action == COMPARE_ACTION:
        run_compare(active, job)
    else:
        raise Baseline50MNodeError(f"unknown 50M fneg-off action {action!r}")


__all__ = [
    "COMPARE_ACTION",
    "COMPARE_CAPABILITY",
    "COMPARE_SCHEMA",
    "PANEL_ACTION",
    "PANEL_CAPABILITY",
    "PANEL_SCHEMA",
    "TRAIN_ACTION",
    "TRAIN_SCHEMA",
    "run_compare",
    "run_job",
    "run_panel",
    "run_train",
]
