"""Frozen contracts for the prompted-English 8M scale rung."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from . import round0113_prompt_contrast as r0113


ROUND_ID = "0166"
CAPABILITY = "jina-document-english-8m-prompted-map-seed42-v1"
SEED = 42
DIMENSION = 768
SUCCESSFUL_UPDATES = 500_000
QUERY_CANDIDATES = 4_096
QUERY_ROWS = 2_000
GRAPH_K = 50
GRAPH_NLIST = 8_192
GRAPH_NPROBE = 64
GRAPH_NPROBE_GRID = (16, 32, 64, 128, 256)
GRAPH_TRAIN_ROWS = 262_144
GRAPH_TRAIN_SEED = 113
GRAPH_QUALITY_ROWS = 4_096
GRAPH_QUALITY_SEED = 114
GRAPH_MEAN_RECALL_FLOOR = 0.90
GRAPH_P10_RECALL_FLOOR = 0.80
RETENTION_RATIO = 0.97
HOST_RSS_LIMIT_GIB = 90.0
METRICS = (
    "density_v2",
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
    "projection_ffr",
    "heldout_recall_at_10",
)
NATIVE_ABSOLUTE_METRICS = (
    "density_v2",
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
)
MULTIPLICITY_POLICY = (
    "prompted-source-text-and-document-fp16-union-representative-only"
)


class Round0166Error(RuntimeError):
    """Raised when the prompted 8M scale contract changes."""


class ScalePromptTrainingInput(r0113.PromptTrainingInput):
    """R0113's exact sampler adapter with a dynamic prompted population."""

    def __init__(
        self,
        dataset: r0113.HostFp16EndpointArray,
        graph: Mapping[str, Any],
        *,
        arm: str = "document",
    ) -> None:
        self.dataset = dataset
        self.graph = dict(graph)
        self.arm = arm
        self.shape = dataset.shape
        self._last_sampler = None
        if (
            arm != "document"
            or self.shape[1:] != (DIMENSION,)
            or self.shape[0] <= 2_000_000
            or int(graph.get("n_nodes", -1)) != len(dataset)
        ):
            raise Round0166Error("R0166 training input geometry changed")

    @staticmethod
    def _patch_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(runtime)
        value["multiplicity_policy"] = MULTIPLICITY_POLICY
        return value

    def prepare_round0034_training(self, **kwargs: Any):
        dataset, sampler, edges, runtime, provenance = super().prepare_round0034_training(
            **kwargs
        )
        return dataset, sampler, edges, self._patch_runtime(runtime), provenance

    def runtime_stamp(self) -> dict[str, Any]:
        return self._patch_runtime(super().runtime_stamp())


def scale_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Clone the accepted R0115 recipe, changing only population bindings."""
    if retained_rows <= 2_000_000 or graph_edges <= 0:
        raise Round0166Error("R0166 train config population is invalid")
    config, _ = r0113.train_config(
        "document",
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=r0113.RETAINED_ROWS,
        seed=SEED,
    )
    config = copy.deepcopy(config)
    config["schema"] = "round0166-prompted-8m-train-config-v1"
    config["paired_invariant"] = {
        "rows": retained_rows,
        "dimension": DIMENSION,
        "seed": SEED,
        "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
        "dose_rule": "same fixed 500,000 successful-update dose as R0115",
        "graph_policy": "same fuzzy-k50 builder parameters and seeds as R0115",
        "sampler": r0113.SAMPLER_CLASS,
    }
    config["input"].update({
        "rows": retained_rows,
        "representation": "prompted-document-host-fp16",
        "multiplicity_policy": MULTIPLICITY_POLICY,
    })
    expected = config["execution"]["expected_pipeline_stamp"]
    expected["negative_sampling"] = (
        f"uniform-{retained_rows}-compact-representatives-nonself"
    )
    expected["compact_retained_rows"] = retained_rows
    expected["multiplicity_policy"] = MULTIPLICITY_POLICY
    config["optimizer"]["successful_positive_lr_updates"] = SUCCESSFUL_UPDATES
    config["execution"]["scale_change"] = (
        "population only; recipe, k50 graph law, seed, and fixed dose unchanged"
    )
    return config, sha256_bytes(canonical_json(config))


def scale_decision(
    *,
    native: Mapping[str, float],
    matched_2m: Mapping[str, float],
    baseline_2m: Mapping[str, float],
    prompted_floors: Mapping[str, float],
) -> dict[str, Any]:
    inputs = (native, matched_2m, baseline_2m, prompted_floors)
    if any(set(value) != set(METRICS) for value in inputs):
        raise Round0166Error("R0166 scale decision metric set changed")
    if not all(
        np.isfinite(float(value))
        for table in inputs
        for value in table.values()
    ):
        raise Round0166Error("R0166 scale decision contains nonfinite metrics")
    native_gates = {
        metric: {
            "observed": float(native[metric]),
            "floor": float(prompted_floors[metric]),
            "passed": float(native[metric]) >= float(prompted_floors[metric]),
        }
        for metric in NATIVE_ABSOLUTE_METRICS
    }
    retention_gates: dict[str, Any] = {}
    for metric in METRICS:
        baseline = float(baseline_2m[metric])
        observed = float(matched_2m[metric])
        if baseline <= 0:
            raise Round0166Error("R0166 baseline metric must be positive")
        ratio = observed / baseline
        retention_gates[metric] = {
            "observed": observed,
            "baseline_seed42": baseline,
            "ratio": ratio,
            "minimum_ratio": RETENTION_RATIO,
            "passed": ratio >= RETENTION_RATIO,
        }
    passed = all(cell["passed"] for cell in native_gates.values()) and all(
        cell["passed"] for cell in retention_gates.values()
    )
    return {
        "passed": passed,
        "native_absolute_gates": native_gates,
        "matched_2m_retention_gates": retention_gates,
        "native_projection_metrics_role": "diagnostic; held-out corpus and N changed",
        "registered_retention_ratio": RETENTION_RATIO,
    }
