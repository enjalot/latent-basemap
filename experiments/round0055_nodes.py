"""Fresh-process handler for the matched balanced-30M int8 control."""
from __future__ import annotations

import os
import random
import time
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.int8_eligibility import load_int8_eligibility
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0034_pipeline import (
    HostInt8MaterializedArray,
    Round0034PipelineError,
    load_canonical_graph,
)
from basemap.round0053_program import (
    DIMENSION,
    ROW_COUNT,
    validate_control_substrate,
)
from basemap.round0055_program import (
    PIPELINE_SCHEMA,
    ROUND_ID,
    SAMPLER_CLASS,
    SEED,
    SUCCESSFUL_UPDATES,
    train_config_from_capabilities,
)
from experiments.round0052_nodes import (
    HostInt8BalancedCanonicalSampler,
    Round0052TrainingInput,
)


class HostInt8Balanced30mCanonicalSampler(
    HostInt8BalancedCanonicalSampler
):
    """R0034's efficient sampler with exact 30M control semantics."""

    def execution_stamp(self) -> dict[str, Any]:
        stamp = super().execution_stamp()
        stamp.update({
            "schema": PIPELINE_SCHEMA,
            "sampler_class": SAMPLER_CLASS,
            "positive_destination_policy": (
                "native-balanced-30m-representative-only-k15;self-removed"
            ),
            "negative_sampling": (
                "uniform-balanced-30m-retained-rows-nonself"
            ),
        })
        return stamp


class Round0055TrainingInput(Round0052TrainingInput):
    """Construct the exact registered 30M host-int8 sampler."""

    def prepare_round0034_training(
        self,
        *,
        edges_path: str,
        batch_size: int,
        pos_ratio: float,
        random_state: int,
        positive_target_mode: str,
        weighted_edge_sampling: bool,
        reject_neighbors: bool,
        required_input_pipeline: str | None,
    ) -> tuple[
        "Round0055TrainingInput",
        HostInt8Balanced30mCanonicalSampler,
        int,
        dict[str, Any],
        dict[str, Any],
    ]:
        manifest = self.graph["manifest"]
        signature = self.graph["signature"]
        if os.path.realpath(edges_path) != signature["canonical_path"]:
            raise Round0034PipelineError(
                "R0055 trainer graph is not the loaded manifest"
            )
        if (
            positive_target_mode != "binary"
            or weighted_edge_sampling
            or reject_neighbors
            or required_input_pipeline != "host_int8_canonical"
        ):
            raise Round0034PipelineError(
                "R0055 requires binary uniform sampling on host int8"
            )
        summary = manifest["summary"]
        sampler = HostInt8Balanced30mCanonicalSampler(
            self.dataset,
            targets=self.graph["targets"],
            degrees=self.graph["degrees"],
            excluded_rows=self.eligibility["excluded_rows"],
            positive_source_count=summary[
                "retained_positive_source_count"
            ],
            valid_edge_count=summary["valid_canonical_edge_count"],
            batch_size=batch_size,
            pos_ratio=pos_ratio,
            random_state=random_state,
            graph_signature=signature,
            eligibility_signature=self.eligibility["signature"],
        )
        self._last_sampler = sampler
        stamp = sampler.execution_stamp()
        verified = {
            "canonical_graph_manifest": signature,
            "canonical_targets": manifest["outputs"]["targets"],
            "canonical_degrees": manifest["outputs"]["degrees"],
            "eligibility": self.eligibility["signature"],
            "int8": self.dataset.signatures.get("int8"),
            "scales": self.dataset.signatures.get("scales"),
        }
        return self, sampler, sampler.n_pos, stamp, verified


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {
        **value,
        "identity_sha256": sha256_bytes(canonical_json(value)),
    }


def _load_pipeline(
    job: Mapping[str, Any],
) -> tuple[
    Round0055TrainingInput,
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
]:
    substrate = validate_control_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    outputs = substrate["manifest"]["outputs"]
    eligibility = load_int8_eligibility(
        outputs["eligibility"]["canonical_path"],
        expected_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    graph = load_canonical_graph(
        str(job["canonical_graph_manifest"]),
        expected_sha256=str(job["canonical_graph_manifest_sha256"]),
        expected_eligibility_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    dataset = HostInt8MaterializedArray.from_files(
        int8_path=outputs["int8"]["canonical_path"],
        int8_sha256=outputs["int8"]["sha256"],
        scales_path=outputs["scales"]["canonical_path"],
        scales_sha256=outputs["scales"]["sha256"],
        row_count=ROW_COUNT,
        dimension=DIMENSION,
        device="cuda",
        buffer_rows=int(job["batch_size"]),
    )
    wrapper = Round0055TrainingInput(
        dataset,
        graph,
        eligibility,
    )
    config, config_sha256 = train_config_from_capabilities(
        graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
        substrate_manifest=substrate["manifest"],
        substrate_manifest_path=substrate["signature"]["canonical_path"],
        substrate_manifest_sha256=substrate["signature"]["sha256"],
    )
    if (
        job.get("train_config_sha256") != config_sha256
        or int(job.get("successful_updates", -1))
        != SUCCESSFUL_UPDATES
    ):
        raise Round0034PipelineError(
            "R0055 queue/config identity changed"
        )
    return wrapper, graph, config, config_sha256, substrate


def run_train(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    round_id = str(active.get("manifest", {}).get("round_id"))
    wrapper, graph, config, config_sha256, substrate = (
        _load_pipeline(job)
    )
    output = create_fresh_directory(
        job["outputs"][0],
        label=f"Round {round_id} train output",
    )
    atomic_write_new_json(
        os.path.join(output, "production-config.json"),
        {
            "schema": str(job.get(
                "production_config_receipt_schema",
                "round0055-production-config-receipt-v1",
            )),
            "config": config,
            "config_sha256": config_sha256,
        },
        immutable=True,
    )
    import torch
    from experiments.run_round0034_node import _exact_model

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats("cuda")
    instance = _exact_model(config)
    instance._max_train_steps = SUCCESSFUL_UPDATES
    instance._bench_warmup = 200
    instance._perf_profile = True
    instance._perf_floor = float(
        config["execution"]["minimum_train_upd_s"]
    )
    instance._perf_warn_rate = float(
        config["execution"]["warning_train_upd_s"]
    )
    instance._perf_subfloor_patience = int(
        config["execution"]["performance_subfloor_patience"]
    )
    instance._perf_n_windows = int(
        config["execution"]["performance_windows"]
    )
    instance._abort_on_first_nonfinite = True
    instance._admission_artifact_path = os.path.join(
        output,
        "admission.json",
    )
    started = time.monotonic()
    instance.fit(
        wrapper,
        low_memory=True,
        verbose=False,
        n_processes=6,
        random_state=SEED,
        resample_negatives=False,
        precomputed_edges_path=graph["signature"]["canonical_path"],
        use_wandb=False,
    )
    wall_seconds = time.monotonic() - started
    accounting = dict(instance._train_stats)
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    expected = {
        "lr_horizon": SUCCESSFUL_UPDATES,
        "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
        "scheduler_steps": SUCCESSFUL_UPDATES,
        "attempted_batches": SUCCESSFUL_UPDATES,
        "finite_loss_batches": SUCCESSFUL_UPDATES,
        "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
        "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": expected_stamp["valid_canonical_edge_count"],
        **{
            f"pipeline_{key}": value
            for key, value in expected_stamp.items()
        },
    }
    mismatches = {
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in expected.items()
        if accounting.get(key) != value
    }
    if mismatches:
        raise Round0034PipelineError(
            f"R0055 exact train accounting failed: {mismatches}"
        )
    runtime = wrapper.runtime_stamp()
    runtime_mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_stamp.items()
        if runtime.get(key) != value
    }
    expected_endpoint_rows = (
        SUCCESSFUL_UPDATES * config["optimizer"]["batch_size"]
    )
    prefetch_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        runtime_mismatches
        or accounting.get("pipeline_runtime") != runtime
        or runtime.get("source_rows_gathered")
        != expected_endpoint_rows
        or runtime.get("destination_rows_gathered")
        != expected_endpoint_rows
        or runtime.get("host_prefetch_consumer_batches")
        != SUCCESSFUL_UPDATES
        or prefetch_delta not in {0, 1}
    ):
        raise Round0034PipelineError(
            "R0055 runtime endpoint/pipeline accounting changed: "
            f"{runtime_mismatches}"
        )
    profiler = instance._canary_profiler.finalize(
        bench_seconds=instance._bench_seconds,
        setup_seconds=getattr(instance, "_setup_seconds", None),
    )
    if (
        profiler.get("n_windows")
        != config["execution"]["performance_windows"]
        or len(profiler.get("rate_windows") or [])
        != config["execution"]["performance_windows"]
    ):
        raise Round0034PipelineError(
            "R0055 profiler did not close every window"
        )
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
        "schema": str(job.get(
            "train_receipt_schema",
            "round0055-train-receipt-v1",
        )),
        "round_id": round_id,
        "release_sha": active["manifest"]["release_sha"],
        "model": expected_input_signature(model_path),
        "production_config": config,
        "production_config_sha256": config_sha256,
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "performance_profile": profiler,
        "graph": graph["signature"],
        "eligibility": wrapper.eligibility["signature"],
        "substrate": substrate["signature"],
        "train_wall_seconds": wall_seconds,
        "seed": SEED,
        "retry_count": 0,
    }
    receipt = _seal(body)
    path = os.path.join(output, "train-receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {
        **receipt,
        "receipt": expected_input_signature(path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    round_id = active.get("manifest", {}).get("round_id")
    if round_id not in {ROUND_ID, "0061"}:
        raise RuntimeError("matched-30M trainer received another queue")
    selected = job if job is not None else active.get("job") or {}
    if (
        selected.get("action") != "train"
        or len(selected.get("outputs") or []) != 1
    ):
        raise RuntimeError("R0055 accepts one train job")
    return run_train(active, selected)
