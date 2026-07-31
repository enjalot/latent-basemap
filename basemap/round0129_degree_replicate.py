"""Frozen contract for the conditional R0129 seed-43 k15 replicate."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .round0113_prompt_contrast import NEGATIVE_RNG_SEED_OFFSET
from .round0124_degree_bridge import (
    ARM,
    ATTEMPT1_EVIDENCE,
    ATTEMPT1_RELEASE_SHA,
    GRAPH_DEGREE,
    GRAPH_SEARCH_NEIGHBORS,
    RETAINED_ROWS,
    SUCCESSFUL_UPDATES,
    load_graph,
    read_sealed,
    train_config as r0124_train_config,
)


ROUND_ID = "0129"
TRAINING_SEED = 43
GRAPH_PROVENANCE_SCHEMA = "round0129-r0124-attempt1-k15-graph-v1"
TRAIN_CONFIG_SCHEMA = "round0129-seed43-k15-train-config-v1"
PRODUCTION_CONFIG_SCHEMA = "round0129-production-config-v1"
TRAIN_RECEIPT_SCHEMA = "round0129-seed43-k15-train-receipt-v1"
DIAGNOSTIC_SCHEMA = "round0129-seed43-k15-diagnostics-v1"
NATIVE_DENSITY_SCHEMA = "round0129-seed43-native-density-score-v1"
DECISION_SCHEMA = "round0129-seed43-degree-replicate-decision-v1"
CAPABILITY = "jina-fineweb-2m-native-k15-degree-bridge-seed43-v1"
R0117_RELEASE_SHA = "c53aae050ec16596a1279176c1694e769fd2c70c"
R0117_TORCH_VERSION = "2.11.0+cu128"
SEED43_INITIAL_STATE_SHA256 = (
    "efafadb7d2e92951503bf76e9f29cd8253ea3ce55179f683dd0fb805639dea42"
)
SEED43_PARAMETER_COUNT = 12_595_714
R0117_R0129_MLP_GIT_BLOB_SHA1 = (
    "6aa2f60d9cba8ed0bcc6b7998097166f96ffc29f"
)
R0117_R0129_NEW_MODEL_SOURCE_SHA256 = (
    "c71476cfa2ab31b96688d083b13b821eb112673e09689cfe2b2e06ed82e31ec4"
)
R0117_R0129_NEW_MODEL_SOURCE_BYTES = 1_943
R0117_R0129_INIT_MODEL_SOURCE_SHA256 = (
    "6d923f7fae09990cbdb7198fab6672eb53e6091b5540de3d4512e3d6d41b7853"
)
R0117_R0129_INIT_MODEL_SOURCE_BYTES = 882
SAMPLER_ROWS_SOURCE_SHA256 = (
    "b20960374eebddaf1627a1cca683c3d948d7b6e0635acf742b4460d2130160c5"
)
SAMPLER_ROWS_SOURCE_BYTES = 697
WEIGHTED_DRAW_SOURCE_SHA256 = (
    "a2982ffc7137cbbffa5317258810e899ce0b1ecd66c6c7d59e7f1eeab472e36e"
)
WEIGHTED_DRAW_SOURCE_BYTES = 1_936


class Round0129Error(RuntimeError):
    """The conditional seed-43 graph-degree replicate changed."""


def _source_segment_identity(value: Any) -> dict[str, Any]:
    payload = inspect.getsource(value).strip().encode("utf-8")
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _git_blob_sha1(path: str) -> str:
    with open(path, "rb") as handle:
        payload = handle.read()
    digest = hashlib.sha1()
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def constructor_source_identity() -> dict[str, Any]:
    """Prove the model constructor sources are exact across R0117/R0129."""
    from basemap.pumap.parametric_umap import ParametricUMAP
    from basemap.pumap.parametric_umap.models.mlp import ResidualBottleneckMLP
    from experiments.round0113_nodes import _new_model

    source_path = inspect.getsourcefile(ResidualBottleneckMLP)
    if not source_path:
        raise Round0129Error("ResidualBottleneckMLP source path is missing")
    observed = {
        "mlp_module_git_blob_sha1": _git_blob_sha1(source_path),
        "new_model_source": _source_segment_identity(_new_model),
        "init_model_source": _source_segment_identity(ParametricUMAP._init_model),
    }
    expected = {
        "mlp_module_git_blob_sha1": R0117_R0129_MLP_GIT_BLOB_SHA1,
        "new_model_source": {
            "bytes": R0117_R0129_NEW_MODEL_SOURCE_BYTES,
            "sha256": R0117_R0129_NEW_MODEL_SOURCE_SHA256,
        },
        "init_model_source": {
            "bytes": R0117_R0129_INIT_MODEL_SOURCE_BYTES,
            "sha256": R0117_R0129_INIT_MODEL_SOURCE_SHA256,
        },
    }
    if observed != expected:
        raise Round0129Error("R0117/R0129 model constructor source changed")
    return {
        "schema": "round0129-constructor-source-identity-v1",
        "historical_release": R0117_RELEASE_SHA,
        "historical_and_treatment_source_equal": True,
        "full_core_blob_identity_claimed": False,
        "source_file": os.path.relpath(
            source_path, os.path.dirname(os.path.dirname(__file__))
        ),
        **observed,
    }


def sampling_mechanism_source_identity() -> dict[str, Any]:
    """Bind unchanged sampler algorithms without claiming equal distributions."""
    from .round0113_prompt_contrast import PromptWeightedJinaSampler

    observed = {
        "rows_source": _source_segment_identity(PromptWeightedJinaSampler._rows),
        "weighted_draw_source": _source_segment_identity(
            PromptWeightedJinaSampler._draw_weighted_edge_ids
        ),
    }
    expected = {
        "rows_source": {
            "bytes": SAMPLER_ROWS_SOURCE_BYTES,
            "sha256": SAMPLER_ROWS_SOURCE_SHA256,
        },
        "weighted_draw_source": {
            "bytes": WEIGHTED_DRAW_SOURCE_BYTES,
            "sha256": WEIGHTED_DRAW_SOURCE_SHA256,
        },
    }
    if observed != expected:
        raise Round0129Error("R0117/R0129 sampler mechanism source changed")
    return {
        "schema": "round0129-sampler-mechanism-source-identity-v1",
        "historical_release": R0117_RELEASE_SHA,
        "historical_and_treatment_mechanism_source_equal": True,
        **observed,
    }


def model_state_sha256(model: Any) -> str:
    """Hash ordered tensor names, dtypes, shapes, and values canonically."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        for payload in (
            key.encode(),
            str(array.dtype).encode(),
            repr(tuple(array.shape)).encode(),
            array.tobytes(order="C"),
        ):
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return digest.hexdigest()


def deterministic_seed43_reconstruction(
    *, r0117_torch_version: str,
) -> dict[str, Any]:
    """Reconstruct, but never mislabel, R0117's deterministic initial state."""
    import torch
    from .pumap.parametric_umap.models.mlp import ResidualBottleneckMLP

    if (
        r0117_torch_version != R0117_TORCH_VERSION
        or torch.__version__ != R0117_TORCH_VERSION
    ):
        raise Round0129Error("R0117/R0129 torch constructor environment changed")
    source_identity = constructor_source_identity()
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    model = ResidualBottleneckMLP(
        input_dim=768,
        hidden_dim=2048,
        output_dim=2,
        num_layers=3,
    )
    observed = model_state_sha256(model)
    parameters = sum(value.numel() for value in model.parameters())
    if observed != SEED43_INITIAL_STATE_SHA256 or parameters != SEED43_PARAMETER_COUNT:
        raise Round0129Error("seed-43 deterministic model reconstruction changed")
    del model
    return {
        "schema": "round0129-seed43-initial-state-reconstruction-v1",
        "algorithm": (
            "sha256 over sorted state_dict key/dtype/shape/contiguous-bytes; "
            "each payload length-prefixed uint64 little-endian"
        ),
        "observed_sha256": observed,
        "expected_seed43_sha256": SEED43_INITIAL_STATE_SHA256,
        "parameter_count": parameters,
        "seed": TRAINING_SEED,
        "torch_version": torch.__version__,
        "constructor": "ResidualBottleneckMLP(768,2048,2,3)",
        "constructor_source_identity": source_identity,
        "historical_evidence_kind": (
            "deterministic-reconstruction-not-original-reviewed-receipt"
        ),
        "r0117_original_pre_update_model_receipt_exists": False,
        "matches_deterministic_r0117_source_reconstruction": True,
    }


def verify_seed43_initial_state(
    model: Any,
    *,
    expected_reconstruction: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the actual R0129 state at the pre-optimizer initialization hook."""
    observed = model_state_sha256(model)
    parameters = sum(value.numel() for value in model.parameters())
    source_identity = constructor_source_identity()
    if (
        observed != SEED43_INITIAL_STATE_SHA256
        or parameters != SEED43_PARAMETER_COUNT
        or expected_reconstruction.get("observed_sha256") != observed
        or expected_reconstruction.get("constructor_source_identity")
        != source_identity
        or expected_reconstruction.get("historical_evidence_kind")
        != "deterministic-reconstruction-not-original-reviewed-receipt"
    ):
        raise Round0129Error("actual R0129 seed-43 initial model state changed")
    return {
        "schema": "round0129-actual-pre-update-model-state-v1",
        "algorithm": expected_reconstruction["algorithm"],
        "observed_sha256": observed,
        "expected_seed43_sha256": SEED43_INITIAL_STATE_SHA256,
        "parameter_count": parameters,
        "seed": TRAINING_SEED,
        "constructor_source_identity": source_identity,
        "captured_immediately_after_init_model": True,
        "captured_before_optimizer_construction_and_update_zero": True,
        "historical_evidence_kind": (
            "deterministic-reconstruction-not-original-reviewed-receipt"
        ),
        "same_release_reconstruction": dict(expected_reconstruction),
        "actual_matches_deterministic_reconstruction": True,
        "actual_historical_r0117_bytes_claimed": False,
    }


def graph_provenance() -> dict[str, Any]:
    """Bind the exact successful R0124 attempt-1 graph node and bytes."""
    keep = {
        key: dict(ATTEMPT1_EVIDENCE[key])
        for key in (
            "queue_manifest",
            "runner_terminal",
            "graph_done_marker",
            "graph_manifest",
            "graph",
            "topology_probe",
        )
    }
    value = {
        "schema": GRAPH_PROVENANCE_SCHEMA,
        "source_round_id": "0124",
        "source_attempt": 1,
        "source_release_sha": ATTEMPT1_RELEASE_SHA,
        "source_terminal_verdict": "failed-after-successful-graph-node",
        "graph_rebuilt": False,
        "evidence": keep,
    }
    return verify_graph_provenance(value)


def _read_json(path: str, *, label: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise Round0129Error(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise Round0129Error(f"{label} is not a JSON object")
    return value


def verify_graph_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Round0129Error("R0129 graph provenance is missing")
    evidence = value.get("evidence")
    expected = {
        key: ATTEMPT1_EVIDENCE[key]
        for key in (
            "queue_manifest",
            "runner_terminal",
            "graph_done_marker",
            "graph_manifest",
            "graph",
            "topology_probe",
        )
    }
    if (
        value.get("schema") != GRAPH_PROVENANCE_SCHEMA
        or value.get("source_round_id") != "0124"
        or value.get("source_attempt") != 1
        or value.get("source_release_sha") != ATTEMPT1_RELEASE_SHA
        or value.get("source_terminal_verdict")
        != "failed-after-successful-graph-node"
        or value.get("graph_rebuilt") is not False
        or not isinstance(evidence, Mapping)
        or dict(evidence) != expected
    ):
        raise Round0129Error("R0129 graph provenance contract changed")
    for label, signature in expected.items():
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise Round0129Error(f"R0129 source graph {label} bytes changed")
    queue = _read_json(
        expected["queue_manifest"]["canonical_path"],
        label="R0124 attempt-1 queue",
    )
    terminal = _read_json(
        expected["runner_terminal"]["canonical_path"],
        label="R0124 attempt-1 terminal",
    )
    done = _read_json(
        expected["graph_done_marker"]["canonical_path"],
        label="R0124 attempt-1 graph done marker",
    )
    manifest = read_sealed(
        expected["graph_manifest"]["canonical_path"],
        label="R0124 attempt-1 k15 graph manifest",
    )
    queue_sha = expected["queue_manifest"]["sha256"]
    if (
        queue.get("schema")
        != "round0124-fineweb-2m-degree-bridge-queue-v1"
        or queue.get("round_id") != "0124"
        or queue.get("release_sha") != ATTEMPT1_RELEASE_SHA
        or terminal.get("schema") != "slim-runner-terminal-v3"
        or terminal.get("round_id") != "0124"
        or terminal.get("verdict") != "failed"
        or terminal.get("queue_manifest_sha256") != queue_sha
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("boundary_problems") != []
        or "build_k15_graph" not in (terminal.get("completed_jobs") or [])
        or done.get("schema") != "slim-runner-done-v2"
        or done.get("node") != "build_k15_graph"
        or done.get("returncode") != 0
        or done.get("queue_manifest_sha256") != queue_sha
        or done.get("release_sha") != ATTEMPT1_RELEASE_SHA
        or manifest.get("round_id") != "0124"
        or manifest.get("release_sha") != ATTEMPT1_RELEASE_SHA
        or manifest.get("graph") != expected["graph"]
        or manifest.get("topology_probe") != expected["topology_probe"]
    ):
        raise Round0129Error("R0124 source graph execution linkage changed")
    return dict(value)


def load_k15_graph(provenance: Mapping[str, Any]) -> dict[str, Any]:
    verified = verify_graph_provenance(provenance)
    evidence = verified["evidence"]
    return load_graph(
        evidence["graph_manifest"]["canonical_path"],
        expected_manifest_signature=evidence["graph_manifest"],
        expected_graph_signature=evidence["graph"],
        expected_topology_probe_signature=evidence["topology_probe"],
        expected_release_sha=ATTEMPT1_RELEASE_SHA,
    )


def train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Derive the seed-43 treatment from the reviewed R0124 recipe."""
    base, _base_sha = r0124_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
    )
    config = copy.deepcopy(base)
    config["schema"] = TRAIN_CONFIG_SCHEMA
    config["causal_invariant"] = {
        "control": "exact accepted R0117 raw seed-43 k49 map",
        "changed_factor": "fuzzy graph neighbor degree only",
        "population_rows": RETAINED_ROWS,
        "dimension": 768,
        "seed": TRAINING_SEED,
        "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
        "control_topology": "R0115/R0117 variable-symmetric fuzzy k50",
        "treatment_topology": "R0124 exact immutable k15 graph bytes",
        "non_graph_config_equal": True,
        "sampling_mechanism_equal_conditioned_on_graph": True,
        "positive_edge_distribution_equal": False,
        "registered_distributional_intervention": (
            "weighted graph topology/edge population/weights induced by k49-to-k15"
        ),
        "negative_sampling_distribution_equal": True,
        "identical_realized_negative_pairs_claimed": False,
        "identical_realized_edge_draws_claimed": False,
        "initial_state_contract": {
            "seed": TRAINING_SEED,
            "expected_sha256": SEED43_INITIAL_STATE_SHA256,
            "parameter_count": SEED43_PARAMETER_COUNT,
            "constructor_source": {
                "mlp_git_blob_sha1": R0117_R0129_MLP_GIT_BLOB_SHA1,
                "new_model_sha256": R0117_R0129_NEW_MODEL_SOURCE_SHA256,
                "init_model_sha256": R0117_R0129_INIT_MODEL_SOURCE_SHA256,
            },
            "historical_evidence_kind": (
                "deterministic-reconstruction-not-original-reviewed-receipt"
            ),
            "actual_pre_update_hook_required": True,
        },
    }
    optimizer = config["optimizer"]
    optimizer["seed"] = TRAINING_SEED
    optimizer["positive_rng_seed"] = TRAINING_SEED
    optimizer["negative_rng_seed"] = (
        TRAINING_SEED + NEGATIVE_RNG_SEED_OFFSET
    )
    pipeline = config["execution"]["expected_pipeline_stamp"]
    pipeline["positive_rng_seed"] = TRAINING_SEED
    pipeline["negative_rng_seed"] = (
        TRAINING_SEED + NEGATIVE_RNG_SEED_OFFSET
    )
    return config, sha256_bytes(canonical_json(config))


def config_equivalence(
    *,
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove graph-only non-graph config/mechanism equality without law claims."""
    if treatment.get("schema") != TRAIN_CONFIG_SCHEMA:
        raise Round0129Error("R0129 treatment train config schema changed")
    if control.get("schema") != "round0113-prompt-arm-train-config-v1":
        raise Round0129Error("R0117 control train config schema changed")
    exact_sections = ("arm", "input", "model", "optimizer")
    if any(treatment.get(key) != control.get(key) for key in exact_sections):
        raise Round0129Error("R0129 non-graph train config differs from R0117")

    treatment_execution = copy.deepcopy(dict(treatment["execution"]))
    control_execution = copy.deepcopy(dict(control["execution"]))
    treatment_execution.pop("training_loop_plan", None)
    treatment_pipeline = treatment_execution.pop("expected_pipeline_stamp")
    control_pipeline = control_execution.pop("expected_pipeline_stamp")
    if treatment_execution != control_execution:
        raise Round0129Error("R0129 non-graph execution config differs")
    graph_pipeline_fields = {
        "positive_destination_policy",
        "graph_degree",
        "graph_search_neighbors_including_self",
        "graph_nonself_degree",
        "valid_canonical_edge_count",
        "feature_residency",
        "device_conversion",
        "graph",
    }
    treatment_sampling = {
        key: value
        for key, value in treatment_pipeline.items()
        if key not in graph_pipeline_fields
    }
    control_sampling = {
        key: value
        for key, value in control_pipeline.items()
        if key not in graph_pipeline_fields
    }
    if treatment_sampling != control_sampling:
        raise Round0129Error("R0129 sampler mechanism differs beyond graph degree")
    treatment_graph = treatment["graph"]
    control_graph = control["graph"]
    if any(
        treatment_graph.get(key) != control_graph.get(key)
        for key in ("nprobe", "sampling", "positive_target_mode")
    ):
        raise Round0129Error("R0129 graph sampling policy changed")
    if (
        treatment_graph.get("k") != GRAPH_DEGREE
        or treatment_graph.get("n_neighbors_including_self")
        != GRAPH_SEARCH_NEIGHBORS
        or control_graph.get("k") != 50
    ):
        raise Round0129Error("R0129 graph-degree contrast changed")
    mechanism_identity = sampling_mechanism_source_identity()
    return {
        "schema": "round0129-treatment-isolation-v2",
        "exact_equal_sections": list(exact_sections),
        "non_graph_config_equal": True,
        "non_graph_execution_equal": True,
        "sampling_mechanism_equal_conditioned_on_graph": True,
        "sampling_mechanism_source_identity": mechanism_identity,
        "graph_sampler_policy_fields_equal": True,
        "positive_edge_distribution_equal": False,
        "registered_distributional_intervention": (
            "weighted graph topology/edge population/weights induced by k49-to-k15"
        ),
        "negative_sampling_distribution_equal": True,
        "identical_realized_negative_pairs_claimed": False,
        "negative_realization_caveat": (
            "the distribution and RNG policy match, but R0117 did not seal a "
            "cross-run full negative-pair trace; seed equality is not execution evidence"
        ),
        "only_registered_non_distributional_config_difference": (
            "graph topology/bytes/degree metadata"
        ),
        "training_seed": TRAINING_SEED,
        "successful_updates": SUCCESSFUL_UPDATES,
        "identical_realized_edge_draws_claimed": False,
        "reason_realized_draws_are_not_paired": (
            "different weighted graph populations and weights define different "
            "positive categorical distributions"
        ),
        "initial_state_contract": dict(
            treatment["causal_invariant"]["initial_state_contract"]
        ),
    }
