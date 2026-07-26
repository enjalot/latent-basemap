"""Train the R0064-selected next rung on the exact host-int8 pipeline."""
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
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0034_pipeline import (
    HostInt8CanonicalSampler,
    HostInt8MaterializedArray,
    Round0034PipelineError,
    Round0034TrainingInput,
    load_canonical_graph,
)
from basemap.round0049_program import DIMENSION
from basemap.round0065_substrates import (
    SUBSETS,
    validate_scale_substrate,
)
from basemap.round0066_quality import load_scale_decision
from basemap.round0068_training import (
    PIPELINE_SCHEMA,
    ROUND_ID,
    SAMPLER_CLASS,
    SEED,
    train_config_from_capabilities,
)


class HostInt8SelectedCanonicalSampler(HostInt8CanonicalSampler):
    """Generic host-int8 sampling with the exact selected-tier semantics."""

    def __init__(self, *args: Any, tier: str, **kwargs: Any) -> None:
        self.tier = tier
        super().__init__(*args, **kwargs)

    def execution_stamp(self) -> dict[str, Any]:
        stamp = super().execution_stamp()
        stamp.update({
            "schema": PIPELINE_SCHEMA,
            "sampler_class": SAMPLER_CLASS,
            "positive_sampling": (
                "uniform-retained-positive-source-then-uniform-native-"
                "representative-destination-with-replacement"
            ),
            "positive_destination_policy": (
                f"native-balanced-{self.tier}-representative-only-k15;"
                "self-removed"
            ),
            "negative_sampling": (
                f"uniform-balanced-{self.tier}-retained-rows-nonself"
            ),
            "graph_degree": (
                "fixed-15-for-every-retained-source;"
                "excluded-sources-degree-zero"
            ),
            "uniform_with_replacement": True,
            "positive_with_replacement": True,
            "weighted_requested": False,
            "weighted_effective": False,
            "source_edge_uniform_equivalent": True,
        })
        return stamp


class Round0068TrainingInput(Round0034TrainingInput):
    """Bind the efficient canonical trainer to the selected graph."""

    def __init__(
        self,
        dataset: HostInt8MaterializedArray,
        graph: Mapping[str, Any],
        eligibility: Mapping[str, Any],
        *,
        tier: str,
    ) -> None:
        self.tier = tier
        super().__init__(dataset, graph, eligibility)

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
        "Round0068TrainingInput",
        HostInt8SelectedCanonicalSampler,
        int,
        dict[str, Any],
        dict[str, Any],
    ]:
        manifest = self.graph["manifest"]
        signature = self.graph["signature"]
        if os.path.realpath(edges_path) != signature["canonical_path"]:
            raise Round0034PipelineError(
                "R0068 trainer graph is not the loaded manifest"
            )
        if (
            positive_target_mode != "binary"
            or weighted_edge_sampling
            or reject_neighbors
            or required_input_pipeline != "host_int8_canonical"
        ):
            raise Round0034PipelineError(
                "R0068 requires binary uniform sampling on host int8"
            )
        summary = manifest["summary"]
        sampler = HostInt8SelectedCanonicalSampler(
            self.dataset,
            tier=self.tier,
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
    Round0068TrainingInput,
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
]:
    decision = load_scale_decision(
        str(job["scale_comparison"]),
        expected_sha256=str(job["scale_comparison_sha256"]),
    )
    tier = decision["tier"]
    if tier != str(job["tier"]):
        raise Round0034PipelineError(
            "R0068 materialized tier differs from R0064"
        )
    spec = SUBSETS[tier]
    row_count = int(spec["row_count"])
    substrate = validate_scale_substrate(
        str(job["substrate_manifest"]),
        tier=tier,
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    outputs = substrate["manifest"]["outputs"]
    graph = load_canonical_graph(
        str(job["canonical_graph_manifest"]),
        expected_sha256=str(job["canonical_graph_manifest_sha256"]),
        expected_eligibility_sha256=outputs["eligibility"]["sha256"],
        row_count=row_count,
    )
    dataset = HostInt8MaterializedArray.from_files(
        int8_path=outputs["int8"]["canonical_path"],
        int8_sha256=outputs["int8"]["sha256"],
        scales_path=outputs["scales"]["canonical_path"],
        scales_sha256=outputs["scales"]["sha256"],
        row_count=row_count,
        dimension=DIMENSION,
        device="cuda",
        buffer_rows=int(job["batch_size"]),
    )
    wrapper = Round0068TrainingInput(
        dataset,
        graph,
        substrate["eligibility"],
        tier=tier,
    )
    config, config_sha256 = train_config_from_capabilities(
        tier=tier,
        graph_manifest=graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
        substrate_manifest=substrate["manifest"],
        substrate_manifest_path=substrate["signature"]["canonical_path"],
        substrate_manifest_sha256=substrate["signature"]["sha256"],
    )
    updates = int(
        config["optimizer"]["successful_positive_lr_updates"]
    )
    if (
        job.get("train_config_sha256") != config_sha256
        or int(job.get("successful_updates", -1)) != updates
    ):
        raise Round0034PipelineError("R0068 queue/config identity changed")
    return wrapper, graph, config, config_sha256, substrate


def run_train(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    wrapper, graph, config, config_sha256, substrate = _load_pipeline(job)
    updates = int(config["optimizer"]["successful_positive_lr_updates"])
    output = create_fresh_directory(
        job["outputs"][0],
        label="Round 0068 train output",
    )
    atomic_write_new_json(
        os.path.join(output, "production-config.json"),
        {
            "schema": "round0068-production-config-receipt-v1",
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
    instance._max_train_steps = updates
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
            f"R0068 exact train accounting failed: {mismatches}"
        )
    runtime = wrapper.runtime_stamp()
    runtime_mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_stamp.items()
        if runtime.get(key) != value
    }
    expected_endpoint_rows = updates * config["optimizer"]["batch_size"]
    prefetch_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        runtime_mismatches
        or accounting.get("pipeline_runtime") != runtime
        or runtime.get("source_rows_gathered") != expected_endpoint_rows
        or runtime.get("destination_rows_gathered") != expected_endpoint_rows
        or runtime.get("host_prefetch_consumer_batches") != updates
        or prefetch_delta not in {0, 1}
    ):
        raise Round0034PipelineError(
            "R0068 runtime endpoint/pipeline accounting changed: "
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
            "R0068 profiler did not close every window"
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
        "schema": "round0068-train-receipt-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "tier": wrapper.tier,
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
    receipt_path = os.path.join(output, "train-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {
        **receipt,
        "receipt": expected_input_signature(receipt_path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("selected-tier trainer received another queue")
    selected = job if job is not None else active.get("job") or {}
    if (
        selected.get("action") != "train_selected_tier"
        or len(selected.get("outputs") or []) != 1
    ):
        raise RuntimeError("R0068 accepts one train job")
    return run_train(active, selected)
