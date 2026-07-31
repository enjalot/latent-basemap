"""Frozen contract for the R0124 native 2M Jina graph-degree bridge."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, expected_input_signature, sha256_bytes
from .round0113_prompt_contrast import (
    BATCH_SIZE,
    DIMENSION,
    NEGATIVE_RNG_SEED_OFFSET,
    PERFORMANCE_WARMUP_UPDATES,
    PERFORMANCE_WINDOWS,
    PIPELINE,
    PIPELINE_SCHEMA,
    POSITIVE_RATIO,
    POSITIVE_ROWS_PER_UPDATE,
    RETAINED_ROWS,
    SAMPLER_CLASS,
    SEED,
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
    TRAIN_WARNING_UPDATES_PER_S,
    HostFp16EndpointArray,
    PromptWeightedJinaSampler,
    read_sealed,
    verify_signature,
)


ROUND_ID = "0124"
ARM = "raw"

# R0115 called its 50-search-neighbor (49 nonself) graph "k50".  This bridge
# deliberately follows the later R0106/R0107 convention: k is the number of
# distinct nonself neighbors and the search/fuzzy tuple includes self.
GRAPH_DEGREE = 15
GRAPH_SEARCH_NEIGHBORS = 16
GRAPH_NLIST = 8_192
GRAPH_TRAIN_ROWS = 262_144
GRAPH_TRAIN_SEED = 113
GRAPH_QUALITY_ROWS = 4_096
GRAPH_QUALITY_SEED = 114
GRAPH_NPROBE_GRID = (16, 32, 64, 128, 256)
GRAPH_NPROBE = 64
GRAPH_MEAN_RECALL_FLOOR = 0.90
GRAPH_P10_RECALL_FLOOR = 0.80

GRAPH_SCHEMA = "round0124-fineweb-2m-k15-fuzzy-graph-v1"
TRAIN_CONFIG_SCHEMA = "round0124-fineweb-2m-k15-train-config-v1"
PRODUCTION_CONFIG_SCHEMA = "round0124-production-config-v1"
TRAIN_RECEIPT_SCHEMA = "round0124-fineweb-2m-k15-train-receipt-v1"
DIAGNOSTIC_SCHEMA = "round0124-fineweb-2m-k15-diagnostics-v1"
NATIVE_DENSITY_SCHEMA = "round0124-fineweb-2m-k15-native-density-score-v1"
DECISION_SCHEMA = "round0124-fineweb-2m-degree-bridge-decision-v1"

BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 12_401
BOOTSTRAP_CI_LEVEL = 0.99
MATERIAL_DENSITY_DEGRADATION = 0.03
NATIVE_DENSITY_ANCHORS = 4_000
NATIVE_ANCHOR_SEED = 123
OUTCOME_MATERIAL = "k15-materially-degrades-native-density"
OUTCOME_NOT_MATERIAL = "k15-does-not-materially-degrade-native-density"
OUTCOME_INCONCLUSIVE = "k15-native-density-effect-inconclusive"


class Round0124Error(RuntimeError):
    """The frozen R0124 single-variable contrast was violated."""


def training_loop_plan(
    *,
    graph_edges: int,
    positive_rows_per_update: int = POSITIVE_ROWS_PER_UPDATE,
    successful_updates: int = SUCCESSFUL_UPDATES,
) -> dict[str, Any]:
    """Return the minimal epoch plan needed to cover the registered horizon."""
    if graph_edges <= 0 or positive_rows_per_update <= 0 or successful_updates <= 0:
        raise Round0124Error("R0124 training loop plan inputs are invalid")
    batches_per_epoch = int(math.ceil(graph_edges / positive_rows_per_update))
    n_epochs = int(math.ceil(successful_updates / batches_per_epoch))
    return {
        "schema": "round0124-training-loop-plan-v1",
        "rule": (
            "ceil(successful_updates / "
            "ceil(graph_edges / positive_rows_per_update))"
        ),
        "graph_edges": int(graph_edges),
        "positive_rows_per_update": int(positive_rows_per_update),
        "batches_per_epoch": batches_per_epoch,
        "successful_positive_lr_updates": int(successful_updates),
        "n_epochs": n_epochs,
        "planned_loop_iters": batches_per_epoch * n_epochs,
    }


def graph_degree_stamp() -> dict[str, Any]:
    return {
        "topology": "variable-symmetric-fuzzy-k15-topology",
        "selected_nonself_neighbors": GRAPH_DEGREE,
        "search_neighbors_including_self": GRAPH_SEARCH_NEIGHBORS,
        "fuzzy_neighbors_including_self": GRAPH_SEARCH_NEIGHBORS,
    }


def expected_pipeline_stamp(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
) -> dict[str, Any]:
    return {
        "schema": PIPELINE_SCHEMA,
        "pipeline": PIPELINE,
        "sampler_class": SAMPLER_CLASS,
        "arm": ARM,
        "positive_sampling": (
            "fuzzy_weight_proportional_with_replacement_via_exact_"
            "uniform_envelope_rejection"
        ),
        "positive_destination_policy": "separate-raw-fp16-fuzzy-k15-graph",
        "negative_sampling": (
            f"uniform-{RETAINED_ROWS}-compact-representatives-nonself"
        ),
        "rng_stream_policy": (
            "separate-positive-rejection-and-negative-pair-streams"
        ),
        "positive_rng_seed": SEED,
        "negative_rng_seed": SEED + NEGATIVE_RNG_SEED_OFFSET,
        "negative_row_pairs_identical_across_arms": True,
        "graph_degree": "variable-symmetric-fuzzy-k15-topology",
        "graph_search_neighbors_including_self": GRAPH_SEARCH_NEIGHBORS,
        "graph_nonself_degree": GRAPH_DEGREE,
        "host_prefetch": "single-producer-two-pinned-slot",
        "endpoint_forward": "fused-source-destination",
        "weighted_requested": True,
        "weighted_effective": True,
        "uniform_with_replacement": False,
        "positive_with_replacement": True,
        "weight_sampler": "uniform-envelope-rejection-max-weight-one",
        "weight_uniform_dtype": np.dtype("float64").str,
        "valid_canonical_edge_count": graph_edges,
        "compact_retained_rows": RETAINED_ROWS,
        "multiplicity_policy": (
            "shared-source-raw-document-union-representative-only"
        ),
        "source_representation": "raw-fp16",
        "feature_residency": "host-contiguous-compact-fp16-memmap",
        "device_conversion": "device-fp32-from-exact-fp16",
        "graph": {
            "graph": dict(graph_signature),
            "manifest": dict(graph_manifest_signature),
        },
    }


def train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    if graph_edges <= 0 or retained_rows != RETAINED_ROWS:
        raise Round0124Error("R0124 train-config graph geometry changed")
    pipeline = expected_pipeline_stamp(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
    )
    loop_plan = training_loop_plan(graph_edges=graph_edges)
    config = {
        "schema": TRAIN_CONFIG_SCHEMA,
        "arm": ARM,
        "causal_invariant": {
            "control": "R0115 raw seed-42 map",
            "changed_factor": "fuzzy graph neighbor degree only",
            "population_rows": RETAINED_ROWS,
            "dimension": DIMENSION,
            "seed": SEED,
            "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
            "control_topology": "R0115 variable-symmetric fuzzy k50",
            "treatment_topology": graph_degree_stamp(),
        },
        "input": {
            "rows": retained_rows,
            "dimension": DIMENSION,
            "representation": "fresh-local-raw-fp16",
            "multiplicity_policy": (
                "shared-source-raw-document-union-representative-only"
            ),
        },
        "graph": {
            "path": str(graph_signature["canonical_path"]),
            "sha256": str(graph_signature["sha256"]),
            "manifest_path": str(graph_manifest_signature["canonical_path"]),
            "manifest_sha256": str(graph_manifest_signature["sha256"]),
            "k": GRAPH_DEGREE,
            "n_neighbors_including_self": GRAPH_SEARCH_NEIGHBORS,
            "nprobe": GRAPH_NPROBE,
            "directed_edges": graph_edges,
            "sampling": "fuzzy-weight-proportional-with-replacement",
            "positive_target_mode": "binary",
        },
        "model": {
            "architecture": "residual_bottleneck",
            "input_dimension": DIMENSION,
            "hidden_dimension": 2048,
            "hidden_layers": 3,
            "output_dimension": 2,
            "use_batchnorm": False,
            "use_dropout": False,
            "low_dim_kernel": "legacy_lp",
            "a": 1.0,
            "b": 1.0,
        },
        "optimizer": {
            "seed": SEED,
            "learning_rate": 0.001,
            "batch_size": BATCH_SIZE,
            "positive_ratio": POSITIVE_RATIO,
            "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
            "positive_rng_seed": SEED,
            "negative_rng_seed": SEED + NEGATIVE_RNG_SEED_OFFSET,
            "positive_target_mode": "binary",
            "weighted_edge_sampling": True,
            "correlation_weight": 0.0,
            "clip_grad_norm": 1.0,
            "use_amp": "bf16",
            "schedule": "cosine-v3-positive-budget",
            "warmup_successful_updates": PERFORMANCE_WARMUP_UPDATES,
            "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
            "reject_neighbors": False,
        },
        "execution": {
            "device_count": 1,
            "required_pipeline": PIPELINE,
            "gpu_resident_data": False,
            "gpu_resident_vram_budget_gb": 0.0,
            "minimum_train_upd_s": TRAIN_MINIMUM_UPDATES_PER_S,
            "warning_train_upd_s": TRAIN_WARNING_UPDATES_PER_S,
            "performance_subfloor_patience": 2,
            "performance_windows": PERFORMANCE_WINDOWS,
            "training_loop_plan": loop_plan,
            "expected_pipeline_stamp": pipeline,
        },
    }
    return config, sha256_bytes(canonical_json(config))


def load_graph(
    manifest_path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    signature = expected_input_signature(manifest_path)
    if signature["sha256"] != expected_sha256:
        raise Round0124Error("R0124 graph manifest bytes changed")
    manifest = read_sealed(manifest_path, label="R0124 k15 graph manifest")
    search = manifest.get("search_qualification") or {}
    fixed = (search.get("cells") or {}).get(str(GRAPH_NPROBE)) or {}
    degree = manifest.get("degree") or {}
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != ROUND_ID
        or manifest.get("arm") != ARM
        or int(manifest.get("retained_rows", -1)) != RETAINED_ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or degree != graph_degree_stamp()
        or int(manifest.get("directed_edge_count", -1)) <= 0
        or int(search.get("selected_nprobe", -1)) != GRAPH_NPROBE
        or fixed.get("passed") is not True
    ):
        raise Round0124Error("R0124 k15 graph contract changed")
    graph_path = verify_signature(manifest["graph"], label="R0124 k15 graph")
    from .pumap.parametric_umap.datasets.edge_list_dataset import load_edge_arrays

    sources, targets, weights, n_nodes = load_edge_arrays(
        graph_path, load_weights=True
    )
    if (
        weights is None
        or int(n_nodes) != RETAINED_ROWS
        or len(sources) != int(manifest["directed_edge_count"])
        or not np.isfinite(weights).all()
    ):
        raise Round0124Error("R0124 graph arrays changed")
    return {
        "manifest": manifest,
        "manifest_signature": signature,
        "signature": dict(manifest["graph"]),
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": int(n_nodes),
    }


class DegreeBridgeTrainingInput:
    """The R0115 host-fp16 input/sampler with only its graph degree changed."""

    round0034_host_int8 = True

    def __init__(
        self,
        dataset: HostFp16EndpointArray,
        graph: Mapping[str, Any],
    ) -> None:
        self.dataset = dataset
        self.graph = dict(graph)
        self.shape = dataset.shape
        self._last_sampler: PromptWeightedJinaSampler | None = None
        if (
            self.shape != (RETAINED_ROWS, DIMENSION)
            or int(graph.get("n_nodes", -1)) != len(dataset)
        ):
            raise Round0124Error("R0124 training input geometry changed")

    def __len__(self) -> int:
        return len(self.dataset)

    def to(self, _device: str) -> "DegreeBridgeTrainingInput":
        return self

    def index_select(self, rows: Any):
        return self.dataset.index_select(rows)

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
    ):
        if (
            expected_input_signature(edges_path) != self.graph["signature"]
            or positive_target_mode != "binary"
            or not weighted_edge_sampling
            or reject_neighbors
            or required_input_pipeline != PIPELINE
        ):
            raise Round0124Error("R0124 trainer pipeline request changed")
        sampler = PromptWeightedJinaSampler(
            self.dataset,
            sources=self.graph["sources"],
            targets=self.graph["targets"],
            weights=self.graph["weights"],
            n_nodes=self.graph["n_nodes"],
            batch_size=batch_size,
            pos_ratio=pos_ratio,
            random_state=random_state,
            graph_signatures={
                "graph": self.graph["signature"],
                "manifest": self.graph["manifest_signature"],
            },
            arm=ARM,
            graph_search_neighbors_including_self=GRAPH_SEARCH_NEIGHBORS,
            graph_nonself_degree=GRAPH_DEGREE,
            graph_degree_label=GRAPH_DEGREE,
        )
        self._last_sampler = sampler
        runtime = sampler.execution_stamp()
        return (
            self,
            sampler,
            sampler.n_pos,
            runtime,
            {
                "graph": self.graph["signature"],
                "graph_manifest": self.graph["manifest_signature"],
                "source_representation": runtime["source_representation"],
            },
        )

    def runtime_stamp(self) -> dict[str, Any]:
        if self._last_sampler is None:
            raise Round0124Error("R0124 sampler has not been constructed")
        return self._last_sampler.execution_stamp()


def classify_degree_bridge(
    *,
    control_density: float,
    treatment_density: float,
    delta_ci_low: float,
    delta_ci_high: float,
) -> dict[str, Any]:
    values = (
        control_density,
        treatment_density,
        delta_ci_low,
        delta_ci_high,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise Round0124Error("R0124 density selector is nonfinite")
    if delta_ci_low > delta_ci_high:
        raise Round0124Error("R0124 density interval is reversed")
    threshold = -MATERIAL_DENSITY_DEGRADATION
    if delta_ci_high <= threshold:
        outcome = OUTCOME_MATERIAL
    elif delta_ci_low > threshold:
        outcome = OUTCOME_NOT_MATERIAL
    else:
        outcome = OUTCOME_INCONCLUSIVE
    return {
        "outcome": outcome,
        "control_density": float(control_density),
        "treatment_density": float(treatment_density),
        "treatment_minus_control": float(treatment_density - control_density),
        "paired_bootstrap_delta_ci": [
            float(delta_ci_low),
            float(delta_ci_high),
        ],
        "paired_bootstrap_ci_level": BOOTSTRAP_CI_LEVEL,
        "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_bootstrap_seed": BOOTSTRAP_SEED,
        "material_degradation_threshold": threshold,
        "selector_metrics": ["native-density-correlation-delta"],
        "core_and_ood_diagnostics_can_rescue_or_fail": False,
        "legacy_density_floor_used": False,
        "single_cause_beyond_graph_degree_claimed": False,
    }


def paired_density_bootstrap(
    *,
    high_radius: np.ndarray,
    control_low_radius: np.ndarray,
    treatment_low_radius: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired anchor bootstrap for the native density-correlation contrast."""
    high = np.asarray(high_radius, dtype=np.float64)
    control = np.asarray(control_low_radius, dtype=np.float64)
    treatment = np.asarray(treatment_low_radius, dtype=np.float64)
    if (
        high.ndim != 1
        or control.shape != high.shape
        or treatment.shape != high.shape
        or len(high) < 3
        or draws != BOOTSTRAP_DRAWS
        or seed != BOOTSTRAP_SEED
        or not np.isfinite(high).all()
        or not np.isfinite(control).all()
        or not np.isfinite(treatment).all()
        or np.any(high < 0)
        or np.any(control < 0)
        or np.any(treatment < 0)
    ):
        raise Round0124Error("R0124 paired native-density inputs changed")
    eps = 1e-12
    log_high = np.log(high + eps)
    log_control = np.log(control + eps)
    log_treatment = np.log(treatment + eps)

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        value = float(np.corrcoef(left, right)[0, 1])
        if not np.isfinite(value):
            raise Round0124Error("R0124 native-density correlation is nonfinite")
        return value

    control_density = correlation(log_high, log_control)
    treatment_density = correlation(log_high, log_treatment)
    rng = np.random.RandomState(seed)
    deltas = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.randint(0, len(high), size=len(high))
        deltas[draw] = (
            correlation(log_high[selected], log_treatment[selected])
            - correlation(log_high[selected], log_control[selected])
        )
    alpha = (1.0 - BOOTSTRAP_CI_LEVEL) / 2.0
    ci_low, ci_high = np.quantile(
        deltas,
        [alpha, 1.0 - alpha],
        method="linear",
    )
    selector = classify_degree_bridge(
        control_density=control_density,
        treatment_density=treatment_density,
        delta_ci_low=float(ci_low),
        delta_ci_high=float(ci_high),
    )
    return {
        **selector,
        "bootstrap_delta_mean": float(deltas.mean()),
        "bootstrap_delta_standard_deviation": float(deltas.std(ddof=1)),
        "bootstrap_quantile_method": "linear",
        "anchor_count": int(len(high)),
        "bootstrap_deltas": deltas,
    }
