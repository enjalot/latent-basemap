#!/usr/bin/env python3
"""Slim-runner handlers for the R0034 canary and core training nodes."""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0034_pipeline import (
    HostInt8MaterializedArray,
    Round0034PipelineError,
    Round0034TrainingInput,
    load_canonical_graph,
    run_two_endpoint_no_update_canary,
)
from basemap.round0034_program import (
    INT8_PATH,
    INT8_SHA256,
    SCALES_PATH,
    SCALES_SHA256,
    train_config_from_capabilities,
)
from experiments.build_round0034_graph import load_released_eligibility


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _load_pipeline(job: dict[str, Any], *, device: str) -> tuple[
        Round0034TrainingInput, dict[str, Any], dict[str, Any], str]:
    eligibility = load_released_eligibility(
        job["eligibility_path"], job["eligibility_sha256"], 150_000_000
    )
    graph = load_canonical_graph(
        job["canonical_graph_manifest"],
        expected_sha256=job["canonical_graph_manifest_sha256"],
        expected_eligibility_sha256=job["eligibility_sha256"],
        row_count=150_000_000,
    )
    dataset = HostInt8MaterializedArray.from_files(
        int8_path=INT8_PATH,
        int8_sha256=INT8_SHA256,
        scales_path=SCALES_PATH,
        scales_sha256=SCALES_SHA256,
        row_count=150_000_000,
        dimension=384,
        device=device,
        buffer_rows=int(job.get("batch_size", 8192)),
    )
    wrapper = Round0034TrainingInput(dataset, graph, eligibility)
    config, config_sha256 = train_config_from_capabilities(
        graph["manifest"],
        canonical_graph_manifest_path=graph["signature"]["canonical_path"],
        canonical_graph_manifest_sha256=graph["signature"]["sha256"],
        eligibility_sha256=job["eligibility_sha256"],
    )
    if (
        job.get("train_config_sha256") != config_sha256
        or int(job.get("successful_positive_lr_updates", -1))
        != config["optimizer"]["successful_positive_lr_updates"]
    ):
        raise Round0034PipelineError("R0034 queue/config update horizon changed")
    return wrapper, graph, config, config_sha256


def _run_canary(job: dict[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0], label="R0034 two-endpoint canary output"
    )
    wrapper, _graph, config, config_sha256 = _load_pipeline(job, device="cuda")
    receipt = run_two_endpoint_no_update_canary(
        wrapper,
        graph_manifest_path=job["canonical_graph_manifest"],
        batch_size=config["optimizer"]["batch_size"],
        pos_ratio=config["optimizer"]["positive_ratio"],
        random_state=config["optimizer"]["seed"],
        warmup_steps=int(job.get("canary_warmup_steps", 20)),
        measured_steps=int(job.get("canary_measured_steps", 100)),
        minimum_batches_per_second=config["execution"][
            "minimum_canary_train_step_equivalents_per_second"
        ],
        minimum_headroom_gib=config["execution"][
            "minimum_post_setup_headroom_gib"
        ],
    )
    receipt["train_config_sha256"] = config_sha256
    # Re-seal after binding the dynamic config.
    receipt = _seal({key: value for key, value in receipt.items()
                     if key != "identity_sha256"})
    path = os.path.join(output, "verdict.json")
    atomic_write_new_json(path, receipt, immutable=True)
    if receipt.get("passed") is not True:
        raise Round0034PipelineError(
            "R0034 canary failed its registered admission thresholds"
        )
    return {**receipt, "verdict": expected_input_signature(path)}


def _exact_model(config: dict[str, Any]):
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = config["model"]
    optimizer = config["optimizer"]
    execution = config["execution"]
    return ParametricUMAP(
        n_components=model["output_dimension"],
        hidden_dim=model["hidden_dimension"],
        n_layers=model["hidden_layers"],
        n_neighbors=15,
        a=model["a"],
        b=model["b"],
        low_dim_kernel=model["low_dim_kernel"],
        correlation_weight=optimizer["correlation_weight"],
        learning_rate=optimizer["learning_rate"],
        n_epochs=1,
        batch_size=optimizer["batch_size"],
        device="cuda",
        use_batchnorm=model["use_batchnorm"],
        use_dropout=model["use_dropout"],
        clip_grad_norm=optimizer["clip_grad_norm"],
        pos_ratio=optimizer["positive_ratio"],
        architecture=model["architecture"],
        correlation_distance_transform="raw",
        lr_schedule="cosine",
        warmup_steps=optimizer["warmup_successful_updates"],
        total_steps_estimate=optimizer["successful_positive_lr_updates"],
        require_full_budget=True,
        require_graph_manifest=True,
        required_input_pipeline=execution["required_pipeline"],
        use_amp=optimizer["use_amp"],
        positive_target_mode=optimizer["positive_target_mode"],
        reject_neighbors=optimizer["reject_neighbors"],
        anchored_init="none",
        anchor_hold_weight=0.0,
        midnear_enabled=False,
        weighted_edge_sampling=False,
        gpu_resident_data=False,
    )


def _run_train(job: dict[str, Any]) -> dict[str, Any]:
    canary_path = os.path.join(job["canary_output"], "verdict.json")
    with open(canary_path, encoding="utf-8") as handle:
        canary = json.load(handle)
    canary_body = {key: value for key, value in canary.items()
                   if key != "identity_sha256"}
    if (
        canary.get("identity_sha256") != sha256_bytes(canonical_json(canary_body))
        or canary.get("passed") is not True
        or canary.get("optimizer_updates") != 0
    ):
        raise Round0034PipelineError("R0034 canary did not pass unchanged")
    canary_signature = expected_input_signature(canary_path)
    output = create_fresh_directory(job["outputs"][0], label="R0034 train output")
    wrapper, graph, config, config_sha256 = _load_pipeline(job, device="cuda")
    if canary.get("train_config_sha256") != config_sha256:
        raise Round0034PipelineError("R0034 canary and train config differ")
    atomic_write_new_json(
        os.path.join(output, "production-config.json"),
        {
            "schema": "round0034-production-config-receipt-v1",
            "config": config,
            "config_sha256": config_sha256,
        },
        immutable=True,
    )

    import torch

    seed = int(config["optimizer"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    instance = _exact_model(config)
    instance._abort_on_first_nonfinite = True
    instance._admission_artifact_path = os.path.join(output, "admission.json")
    instance.fit(
        wrapper,
        low_memory=True,
        random_state=seed,
        verbose=False,
        precomputed_edges_path=graph["signature"]["canonical_path"],
        use_wandb=False,
    )
    accounting = dict(instance._train_stats)
    required_updates = int(config["optimizer"]["successful_positive_lr_updates"])
    if (
        accounting.get("budget_satisfied") is not True
        or accounting.get("positive_lr_optimizer_steps") != required_updates
        or accounting.get("optimizer_steps_attempted") != required_updates
        or accounting.get("optimizer_steps_succeeded") != required_updates
        or accounting.get("amp_overflow_skips") != 0
        or accounting.get("nonfinite_loss_skips") != 0
        or accounting.get("nonfinite_gradient_skips") != 0
    ):
        raise Round0034PipelineError(
            f"R0034 successful-update accounting failed: {accounting}"
        )
    runtime = wrapper.runtime_stamp()
    expected_endpoint_rows = accounting["attempted_batches"] * config["optimizer"][
        "batch_size"
    ]
    if (
        runtime["source_rows_gathered"] != expected_endpoint_rows
        or runtime["destination_rows_gathered"] != expected_endpoint_rows
    ):
        raise Round0034PipelineError("R0034 endpoint gather accounting failed")

    model_path = os.path.join(output, "model.pt")

    def write_model(path: str) -> None:
        state = {
            name: value.detach().cpu()
            for name, value in instance.model.state_dict().items()
        }
        torch.save(
            {
                "state_dict": state,
                "production_config": config,
                "production_config_sha256": config_sha256,
            },
            path,
        )

    atomic_build_new_file(model_path, write_model, immutable=True)
    body = {
        "schema": "round0034-train-receipt-v1",
        "round_id": "0034",
        "model": expected_input_signature(model_path),
        "train_config": config,
        "train_config_sha256": config_sha256,
        "canary": canary_signature,
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "graph": graph["signature"],
        "eligibility": wrapper.eligibility["signature"],
    }
    receipt = _seal(body)
    receipt_path = os.path.join(output, "train-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != "0034":
        raise RuntimeError("R0034 handler received another queue")
    job = job if job is not None else active.get("job") or {}
    if len(job.get("outputs") or []) != 1:
        raise RuntimeError("R0034 job output contract changed")
    action = job.get("action")
    if action == "canary":
        return _run_canary(job)
    if action == "train":
        return _run_train(job)
    raise RuntimeError(f"unknown R0034 action: {action!r}")
