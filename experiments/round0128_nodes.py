"""Execute the direct 25M k49-versus-k15 graph-degree rescue."""
from __future__ import annotations

import gc
import json
import os
import random
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0040_program import (
    RepresentativeArrayView,
    RepresentativeRowSelector,
    load_jina_census,
)
from basemap.round0105_search import (
    BOUNDARY_TIE_ATOL,
    EVERY_GROUP_MEAN_FLOOR,
    GLOBAL_MEAN_FLOOR,
    GROUPS,
    K as R0105_K,
    QUALITY_GROUP_IDS_SHA256,
    QUALITY_ROWS,
    QUALITY_ROWS_PER_GROUP,
    QUALITY_SAMPLE_SHA256,
    RETAINED_ROWS,
    group_ranges,
    sample_stratified_rows,
)
from basemap.round0107_training import validate_seal as validate_train_seal
from basemap.round0108_evaluation import (
    CALIBRATION_SCHEMA,
    FAMILY_SIZE_CUTOFF,
    K_DENSITY,
    Round0108Error,
    map_family_sizes,
    read_sealed,
    seal,
    verify_signature,
)
from basemap.round0128_k49_rescue import (
    CORE_SCHEMA,
    DECISION_SCHEMA,
    DENSITY_BOOTSTRAP_DRAWS,
    DENSITY_BOOTSTRAP_SEED,
    FIXED_SUCCESSFUL_UPDATES,
    GRAPH_K,
    GRAPH_PART_SCHEMA,
    GRAPH_SCHEMA,
    GRAPH_SHARD_SCHEMA,
    MAP_KEY,
    MAP_LABEL,
    MATCHED_DENSITY_SCHEMA,
    N_NEIGHBORS,
    OOD_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    PRODUCTION_CONFIG_SCHEMA,
    ROUND_ID,
    Round0128Error,
    TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA,
    assert_treatment_isolation,
    k49_train_config,
    noninferiority_checks,
    paired_density_materiality,
    verify_r0107_seed42_initial_state,
)
from experiments import round0108_nodes as evaluation_nodes
from experiments.round0105_nodes import (
    _exact_truth,
    _gpu_options,
    _load_sealed,
    _normalized_rows,
    _require_geometry,
    _search_and_rerank,
    _substrate_arrays,
)
from experiments.round0106_nodes import (
    GraphNodeContract,
    _load_search,
)
from experiments.round0107_nodes import run_train_contract
from experiments.round0110_nodes import _matched_density_cell


QUALIFICATION_SCHEMA = "round0128-k49-fixed-policy-qualification-v1"
GRAPH_CONTRACT = GraphNodeContract(
    round_id=ROUND_ID,
    k=GRAPH_K,
    n_neighbors=N_NEIGHBORS,
    shard_schema=GRAPH_SHARD_SCHEMA,
    part_schema=GRAPH_PART_SCHEMA,
    graph_schema=GRAPH_SCHEMA,
)
GRAPH_CONTRACT_JOB = {
    "round_id": GRAPH_CONTRACT.round_id,
    "k": GRAPH_CONTRACT.k,
    "n_neighbors": GRAPH_CONTRACT.n_neighbors,
    "shard_schema": GRAPH_CONTRACT.shard_schema,
    "part_schema": GRAPH_CONTRACT.part_schema,
    "graph_schema": GRAPH_CONTRACT.graph_schema,
}
EVALUATION_CONTRACT = evaluation_nodes.EvaluationNodeContract(
    round_id=ROUND_ID,
    map_key=MAP_KEY,
    map_label=MAP_LABEL,
    graph_round_id=ROUND_ID,
    graph_k=GRAPH_K,
    core_schema=CORE_SCHEMA,
    ood_schema=OOD_SCHEMA,
    train_round_id=ROUND_ID,
    train_receipt_schema=TRAIN_RECEIPT_SCHEMA,
    production_config_schema=PRODUCTION_CONFIG_SCHEMA,
    seed=42,
    graph_schema=GRAPH_SCHEMA,
)
EVALUATION_CONTRACT_JOB = {
    key: getattr(EVALUATION_CONTRACT, key)
    for key in EVALUATION_CONTRACT.__dataclass_fields__
}


def _signature(path: str, expected_sha256: str, *, label: str) -> dict[str, Any]:
    value = expected_input_signature(path)
    if value["sha256"] != expected_sha256:
        raise Round0128Error(f"{label} bytes changed")
    return value


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0128Error(f"{path} is not a JSON object")
    return value


def _k49_policy_metrics(
    selected: np.ndarray,
    exact: np.ndarray,
    *,
    group_ids: np.ndarray,
    unambiguous: np.ndarray,
) -> dict[str, Any]:
    candidate = np.asarray(selected, dtype=np.int64)
    truth = np.asarray(exact, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=np.uint8)
    clear = np.asarray(unambiguous, dtype=bool)
    if (
        candidate.shape != (QUALITY_ROWS, GRAPH_K)
        or truth.shape != candidate.shape
        or groups.shape != (QUALITY_ROWS,)
        or clear.shape != (QUALITY_ROWS,)
        or np.any(candidate < 0)
        or np.any(truth < 0)
        or np.any(np.diff(np.sort(candidate, axis=1), axis=1) == 0)
        or not np.any(clear)
    ):
        raise Round0128Error("k49 qualification arrays are malformed")
    overlap = (
        candidate[:, :, None] == truth[:, None, :]
    ).any(axis=2).sum(axis=1) / GRAPH_K
    by_group: dict[str, Any] = {}
    group_passes: list[bool] = []
    for group_id, group in enumerate(GROUPS):
        registered = groups == group_id
        mask = registered & clear
        values = overlap[mask]
        mean = float(values.mean()) if len(values) else None
        passed = mean is not None and mean >= EVERY_GROUP_MEAN_FLOOR
        group_passes.append(passed)
        by_group[group] = {
            "registered_rows": int(registered.sum()),
            "boundary_ties_excluded": int((registered & ~clear).sum()),
            "unambiguous_rows": int(mask.sum()),
            "mean_recall_at_49_unambiguous": mean,
            "passes_floor": passed,
        }
    values = overlap[clear]
    global_mean = float(values.mean()) if len(values) else None
    checks = {
        "all_rows_complete_and_unique": True,
        "every_group_has_unambiguous_rows": all(
            value["unambiguous_rows"] > 0 for value in by_group.values()
        ),
        "global_mean_recall_at_49_at_least_0p90": (
            global_mean is not None and global_mean >= GLOBAL_MEAN_FLOOR
        ),
        "every_group_mean_recall_at_49_at_least_0p84": all(group_passes),
    }
    return {
        "mean_recall_at_49": float(overlap.mean()),
        "mean_recall_at_49_unambiguous": global_mean,
        "p10_recall_at_49_unambiguous": float(np.percentile(values, 10)),
        "by_group": by_group,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_qualify_k49(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    """Qualify only R0105's selected nprobe=64,width=128 policy at k49."""
    import faiss
    import torch

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0128 fixed-policy k49 qualification"
    )
    started = time.monotonic()
    substrate, excluded, encoded, scales, search = _load_search(active, job)
    selected_policy = search["selected"]
    if (
        int(selected_policy.get("nprobe", -1)) != 64
        or int(selected_policy.get("shortlist_width", -1)) != 128
    ):
        raise Round0128Error("R0105 selected policy is not nprobe64/width128")

    ranges = group_ranges(substrate["manifest"])
    sample, group_ids = sample_stratified_rows(excluded, ranges)
    if (
        sha256_bytes(sample.tobytes()) != QUALITY_SAMPLE_SHA256
        or sha256_bytes(group_ids.tobytes()) != QUALITY_GROUP_IDS_SHA256
        or any(
            np.count_nonzero(group_ids == index) != QUALITY_ROWS_PER_GROUP
            for index in range(len(GROUPS))
        )
    ):
        raise Round0128Error("registered R0105 stratified sample changed")

    accepted_truth_signature = _signature(
        str(job["r0105_truth"]),
        str(job["r0105_truth_sha256"]),
        label="R0105 exact top15 truth",
    )
    with np.load(str(job["r0105_truth"]), allow_pickle=False) as archive:
        accepted_sample = np.asarray(archive["sample_rows"], dtype=np.int64)
        accepted_group_ids = np.asarray(archive["group_ids"], dtype=np.uint8)
        accepted_exact15 = np.asarray(
            archive["exact_neighbors"], dtype=np.int64
        )
    if (
        not np.array_equal(accepted_sample, sample)
        or not np.array_equal(accepted_group_ids, group_ids)
        or accepted_exact15.shape != (QUALITY_ROWS, R0105_K)
    ):
        raise Round0128Error("accepted R0105 exact-truth arrays changed")

    exact49, ties49, margins49, truth_timing = _exact_truth(
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        sample=sample,
        k=GRAPH_K,
    )
    if not np.array_equal(exact49[:, :R0105_K], accepted_exact15):
        raise Round0128Error("exact k49 truth does not extend R0105 exact top15")
    queries = _normalized_rows(encoded, scales, sample)
    cpu = faiss.read_index(str(job["index"]))
    _require_geometry(cpu, ntotal=RETAINED_ROWS)
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    gpu = faiss.index_cpu_to_gpu(resources, 0, cpu, _gpu_options())
    selected49, execution49 = _search_and_rerank(
        gpu,
        nprobe=64,
        width=128,
        queries=queries,
        sample=sample,
        excluded=excluded,
        encoded=encoded,
        scales=scales,
        k=GRAPH_K,
    )
    # R0105 did not persist each policy's selected top15 IDs. Reconstruct the
    # accepted selected policy once under the same reviewed release and report
    # that limitation explicitly; this is not represented as a byte proof.
    reconstructed15, execution15 = _search_and_rerank(
        gpu,
        nprobe=64,
        width=128,
        queries=queries,
        sample=sample,
        excluded=excluded,
        encoded=encoded,
        scales=scales,
        k=R0105_K,
    )
    prefix_equal = np.array_equal(selected49[:, :R0105_K], reconstructed15)
    metrics = _k49_policy_metrics(
        selected49,
        exact49,
        group_ids=group_ids,
        unambiguous=~ties49,
    )
    checks = {
        **metrics["checks"],
        "exact_truth_first15_equals_accepted_r0105_truth": True,
        "selected_k49_first15_equals_same_release_r0105_policy_reconstruction": (
            prefix_equal
        ),
        "one_local_cuda_device": (
            faiss.get_num_gpus() == 1 and torch.cuda.device_count() == 1
        ),
        "no_policy_sweep_or_tuning": True,
        "no_graph_or_map_built": True,
    }
    if not all(checks.values()):
        raise Round0128Error(
            "fixed-policy k49 qualification failed: "
            + ", ".join(key for key, value in checks.items() if not value)
        )
    arrays_path = os.path.join(output, "k49-qualification-arrays.npz")
    atomic_save_new_npz(
        arrays_path,
        immutable=True,
        sample_rows=sample,
        group_ids=group_ids,
        accepted_exact_top15=accepted_exact15,
        exact_neighbors_top49=exact49,
        selected_neighbors_top49=selected49,
        reconstructed_selected_top15=reconstructed15,
        boundary_ties_at_49=ties49,
        boundary_margins_at_49=margins49.astype(np.float32),
    )
    receipt = seal({
        "schema": QUALIFICATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "substrate": substrate["signature"],
        "r0105_search": search,
        "accepted_r0105_truth": accepted_truth_signature,
        "fixed_policy": {
            "nprobe": 64,
            "shortlist_width": 128,
            "selected_neighbors": GRAPH_K,
            "candidate_universe": RETAINED_ROWS,
            "tuning_or_sweep_performed": False,
        },
        "boundary_tie_policy": {
            "atol": BOUNDARY_TIE_ATOL,
            "boundary": "exact ranks 49 and 50",
            "ties_excluded_from_registered_means": True,
            "tie_rows": int(ties49.sum()),
        },
        "selected_top15_prefix_evidence": {
            "historical_selected_rows_stored_by_r0105": False,
            "direct_byte_comparison_available": False,
            "limitation": (
                "R0105 stored exact truth and aggregate policy metrics, not "
                "per-row selected-policy neighbors"
            ),
            "evidence_kind": "same-release-deterministic-reconstruction",
            "prefix_equal": prefix_equal,
            "reconstructed_top15_ordered_sha256": ordered_array_sha256(
                reconstructed15
            ),
            "k49_first15_ordered_sha256": ordered_array_sha256(
                selected49[:, :R0105_K]
            ),
            "reconstruction_execution": execution15,
        },
        "quality": metrics,
        "checks": checks,
        "outcome": "qualified-fixed-r0105-policy-at-k49",
        "graph_build_released": True,
        "arrays": expected_input_signature(arrays_path),
        "performance": {
            "exact_truth": truth_timing,
            "selected_k49": execution49,
            "wall_seconds": time.monotonic() - started,
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "map_decision_made": False,
    })
    receipt_path = os.path.join(output, "k49-qualification.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del gpu, resources, cpu, encoded, scales
    gc.collect()
    torch.cuda.empty_cache()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _quality_admission(job: Mapping[str, Any]) -> dict[str, Any]:
    path = os.path.join(str(job["quality_output"]), "k49-qualification.json")
    receipt = read_sealed(
        path, label="R0128 k49 qualification", schema=QUALIFICATION_SCHEMA
    )
    if (
        receipt.get("round_id") != ROUND_ID
        or receipt.get("outcome") != "qualified-fixed-r0105-policy-at-k49"
        or receipt.get("graph_build_released") is not True
        or not all((receipt.get("checks") or {}).values())
    ):
        raise Round0128Error("k49 graph quality admission is not positive")
    return expected_input_signature(path)


def run_graph_part(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    from experiments.round0106_nodes import run_build_part

    selected = dict(job)
    selected["graph_node_contract"] = GRAPH_CONTRACT_JOB
    selected["graph_quality_admission"] = _quality_admission(job)
    return run_build_part(active, selected)


def run_graph_assembly(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    from experiments.round0106_nodes import run_assemble

    selected = dict(job)
    selected["graph_node_contract"] = GRAPH_CONTRACT_JOB
    result = run_assemble(active, selected)
    if result.get("graph_quality_admission") != _quality_admission(job):
        raise Round0128Error("assembled graph lost k49 quality admission")
    return result


def _legacy_initial_reconstruction() -> dict[str, Any]:
    import torch
    from basemap.pumap.parametric_umap.models.mlp import ResidualBottleneckMLP

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = ResidualBottleneckMLP(
        input_dim=768,
        hidden_dim=2048,
        output_dim=2,
        num_layers=3,
    )
    receipt = verify_r0107_seed42_initial_state(model)
    receipt["reconstruction_runtime"] = {
        "torch_version": torch.__version__,
        "device": "cpu",
        "seed": 42,
        "parameters": sum(value.numel() for value in model.parameters()),
    }
    del model
    return receipt


def _model_class(expected_reconstruction: Mapping[str, Any]) -> type:
    from basemap.pumap.parametric_umap import ParametricUMAP

    expected = dict(expected_reconstruction)

    class Round0128ParametricUMAP(ParametricUMAP):
        def _init_model(self, input_dim: int) -> None:
            super()._init_model(input_dim)
            actual = verify_r0107_seed42_initial_state(self.model)
            if actual["observed_sha256"] != expected["observed_sha256"]:
                raise Round0128Error(
                    "actual treatment initialization differs from the "
                    "same-release R0107 deterministic reconstruction"
                )
            self._initial_model_state_receipt = {
                **actual,
                "same_release_legacy_default_reconstruction": expected,
                "same_release_byte_equal": True,
                "capture_hook": (
                    "round-specific ParametricUMAP._init_model override after "
                    "graph/input admission and before optimizer construction"
                ),
            }

    return Round0128ParametricUMAP


def _control_config(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = str(job["r0107_production_config"])
    signature = _signature(
        path,
        str(job["r0107_production_config_sha256"]),
        label="accepted R0107 production config",
    )
    value = _load_json(path)
    config = value.get("config")
    if (
        value.get("schema") != "round0107-production-config-v1"
        or value.get("round_id") != "0107"
        or not isinstance(config, Mapping)
        or sha256_bytes(
            json.dumps(
                config, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        != value.get("config_sha256")
    ):
        raise Round0128Error("accepted R0107 production config changed")
    return dict(config), signature


def _accepted_r0107_lineage(
    job: Mapping[str, Any],
    *,
    control_config: Mapping[str, Any],
) -> dict[str, Any]:
    path = str(job["r0107_train_receipt"])
    signature = _signature(
        path,
        str(job["r0107_train_receipt_sha256"]),
        label="accepted R0107 train receipt",
    )
    receipt = _load_json(path)
    validate_train_seal(receipt, label="accepted R0107 train receipt")
    stamp = receipt.get("exact_execution_receipt") or {}
    expected_stamp = (control_config.get("execution") or {}).get(
        "expected_pipeline_stamp"
    )
    if (
        receipt.get("schema")
        != "round0107-diverse-jina-train-receipt-v1"
        or receipt.get("round_id") != "0107"
        or receipt.get("production_config_sha256")
        != sha256_bytes(
            json.dumps(
                control_config,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        or int(receipt.get("optimizer_updates", -1))
        != FIXED_SUCCESSFUL_UPDATES
        or int((control_config.get("optimizer") or {}).get("seed", -1)) != 42
        or not isinstance(expected_stamp, Mapping)
        or any(stamp.get(key) != value for key, value in expected_stamp.items())
        or (receipt.get("train_checks") or {}).get(
            "exact_update_closure"
        ) is not True
        or (receipt.get("train_checks") or {}).get(
            "no_pipeline_stamp_drift"
        ) is not True
    ):
        raise Round0128Error("accepted R0107 seed/config/runtime lineage changed")
    model = receipt.get("model")
    if not isinstance(model, Mapping):
        raise Round0128Error("accepted R0107 model signature is missing")
    return {
        "train_receipt": signature,
        "release_sha": receipt["release_sha"],
        "model": dict(model),
        "seed": 42,
        "successful_updates": FIXED_SUCCESSFUL_UPDATES,
        "actual_pipeline_stamp": {
            key: stamp[key] for key in expected_stamp
        },
        "historical_initial_state_evidence": (
            "deterministic reconstruction; the accepted R0107 receipt did "
            "not store pre-update model bytes"
        ),
    }


def run_train(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    graph_path = str(job["graph_manifest"])
    graph_signature = expected_input_signature(graph_path)
    graph = read_sealed(graph_path, label="R0128 k49 graph", schema=GRAPH_SCHEMA)
    if (
        graph.get("round_id") != ROUND_ID
        or int(graph.get("k_real", -1)) != GRAPH_K
        or graph.get("graph_quality_admission") != _quality_admission(job)
    ):
        raise Round0128Error("R0128 k49 graph contract changed")
    control, control_signature = _control_config(job)
    historical_lineage = _accepted_r0107_lineage(
        job, control_config=control
    )
    treatment, treatment_sha256 = k49_train_config(
        graph_manifest=graph,
        graph_signature=graph_signature,
    )
    isolation = assert_treatment_isolation(control, treatment)
    reconstruction = _legacy_initial_reconstruction()
    selected = dict(job)
    selected["graph_manifest_sha256"] = graph_signature["sha256"]
    receipt = run_train_contract(
        active,
        selected,
        round_id=ROUND_ID,
        seed=42,
        train_config_schema=TRAIN_CONFIG_SCHEMA,
        production_config_schema=PRODUCTION_CONFIG_SCHEMA,
        train_receipt_schema=TRAIN_RECEIPT_SCHEMA,
        output_label="R0128 diverse-Jina k49 treatment",
        graph_load_kwargs={
            "expected_graph_schema": GRAPH_SCHEMA,
            "expected_graph_round_id": ROUND_ID,
            "expected_k_real": GRAPH_K,
            "successful_updates": FIXED_SUCCESSFUL_UPDATES,
        },
        train_config_kwargs={
            "n_neighbors_including_self": N_NEIGHBORS,
            "successful_updates": FIXED_SUCCESSFUL_UPDATES,
            "update_rule": "fixed-R0107-dose-1459722-successful-updates",
            "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
            "graph_degree": "variable-symmetric-fuzzy-k49-topology",
        },
        training_input_kwargs={
            "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
            "graph_degree": "variable-symmetric-fuzzy-k49-topology",
        },
        model_class=_model_class(reconstruction),
        require_initial_model_state=True,
    )
    if (
        receipt.get("production_config_sha256") != treatment_sha256
        or int(receipt.get("optimizer_updates", -1))
        != FIXED_SUCCESSFUL_UPDATES
    ):
        raise Round0128Error("R0128 fixed treatment config/dose drifted")
    # The immutable receipt cannot be edited after generic publication. Publish
    # a separate sealed isolation receipt that binds both configs and the train.
    output = str(job["outputs"][0])
    isolation_receipt = seal({
        "schema": "round0128-k49-treatment-isolation-v1",
        "round_id": ROUND_ID,
        "accepted_r0107_production_config": control_signature,
        "r0128_production_config": expected_input_signature(
            os.path.join(output, "production-config.json")
        ),
        "r0128_train_receipt": expected_input_signature(
            os.path.join(output, "train-receipt.json")
        ),
        "comparison": isolation,
        "frozen_dose_source": {
            "round_id": "0107",
            "successful_updates": FIXED_SUCCESSFUL_UPDATES,
            "derivation_for_r0128": "fixed-not-edge-derived",
        },
        "initial_state_evidence": receipt["initial_model_state"],
        "accepted_r0107_lineage": historical_lineage,
        "treatment_isolated": True,
    })
    isolation_path = os.path.join(output, "treatment-isolation.json")
    atomic_write_new_json(isolation_path, isolation_receipt, immutable=True)
    return {
        **receipt,
        "treatment_isolation": expected_input_signature(isolation_path),
    }


def _evaluation_job(job: Mapping[str, Any]) -> dict[str, Any]:
    selected = dict(job)
    graph_path = str(selected["graph_manifest"])
    graph_signature = expected_input_signature(graph_path)
    graph = read_sealed(
        graph_path, label="R0128 evaluation graph", schema=GRAPH_SCHEMA
    )
    if (
        graph.get("round_id") != ROUND_ID
        or int(graph.get("k_real", -1)) != GRAPH_K
        or graph.get("graph_quality_admission") != _quality_admission(job)
    ):
        raise Round0128Error("R0128 evaluation graph lineage changed")
    selected["graph_manifest_sha256"] = graph_signature["sha256"]
    selected["evaluation_node_contract"] = EVALUATION_CONTRACT_JOB
    selected["graph_node_contract"] = GRAPH_CONTRACT_JOB
    return selected


def run_transform(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    return evaluation_nodes.run_transform(active, _evaluation_job(job))


def run_core(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    return evaluation_nodes.run_core_score(active, _evaluation_job(job))


def run_ood(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    return evaluation_nodes.run_ood(active, _evaluation_job(job))


def run_matched_density(
    _active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    """Score the exact R0040 universe as a diagnostic, never a gate."""
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0128 matched R0040 diagnostic"
    )
    started = time.monotonic()
    calibration_path = os.path.join(
        str(job["calibration_output"]), "jina-density-calibration.json"
    )
    calibration = read_sealed(
        calibration_path,
        label="R0108 frozen Jina calibration",
        schema=CALIBRATION_SCHEMA,
    )
    floor = float((calibration.get("floor_calibration") or {})["registered_floor"])
    census_path = str(job["census_receipt"])
    census_signature = _signature(
        census_path,
        str(job["census_receipt_sha256"]),
        label="R0040 census receipt",
    )
    census = load_jina_census(census_path)
    source_path = verify_signature(
        census["receipt"].get("source"), label="R0040 Jina source"
    )
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    selector = RepresentativeRowSelector(
        census["arrays"]["excluded_rows"],
        row_count=2_000_000,
        source=census["signature"],
        policy="R0040 exact nonzero fp16 family; minimum row representative",
    )
    representatives = RepresentativeArrayView(source, selector)
    reference_path = str(job["representative_reference"])
    reference_signature = _signature(
        reference_path,
        str(job["representative_reference_sha256"]),
        label="R0040 high-D reference",
    )
    with np.load(reference_path, allow_pickle=False) as archive:
        anchors = np.asarray(archive["anchor_ids"], dtype=np.int64)
        high_radius = np.asarray(archive["r_hd"], dtype=np.float64)
        reference_key = str(archive["key"].item())
    global_rows = selector.compact_to_global(anchors)
    family_sizes = map_family_sizes(
        global_rows,
        census["arrays"]["representative_rows"],
        census["arrays"]["family_counts"],
    )
    if (
        anchors.shape != (10_000,)
        or high_radius.shape != anchors.shape
        or np.any(family_sizes >= FAMILY_SIZE_CUTOFF)
        or reference_key != calibration.get("representative_reference_key")
        or ordered_array_sha256(anchors)
        != (calibration.get("anchors") or {}).get("compact_rows_sha256")
    ):
        raise Round0128Error("R0040 matched diagnostic universe changed")
    bundle = evaluation_nodes._load_contract_model(
        _evaluation_job(job), EVALUATION_CONTRACT
    )
    cell, arrays = _matched_density_cell(
        key="r0128-k49-seed42",
        seed=42,
        bundle=bundle,
        representatives=representatives,
        anchors=anchors,
        high_radius=high_radius,
    )
    arrays_path = os.path.join(output, "matched-density-arrays.npz")
    atomic_save_new_npz(
        arrays_path,
        immutable=True,
        anchor_compact_rows=anchors,
        anchor_global_rows=global_rows,
        high_radius=high_radius,
        family_sizes=family_sizes,
        **arrays,
    )
    receipt = seal({
        "schema": MATCHED_DENSITY_SCHEMA,
        "round_id": ROUND_ID,
        "role": "diagnostic-only; excluded from every registered selector",
        "census_receipt": census_signature,
        "representative_reference": reference_signature,
        "registered_floor": floor,
        "cell": {
            **cell,
            "clears_unchanged_registered_floor": (
                float(cell["density_v2"]["correlation"]) >= floor
            ),
        },
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "map_decision_made": False,
        "wall_seconds": time.monotonic() - started,
    })
    receipt_path = os.path.join(output, "matched-density.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del bundle["model"], representatives, source
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _read_exact_receipt(
    path: str,
    expected_sha256: str,
    *,
    label: str,
    schema: str,
    round_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = _signature(path, expected_sha256, label=label)
    value = read_sealed(path, label=label, schema=schema)
    if value.get("round_id") != round_id:
        raise Round0128Error(f"{label} round changed")
    return value, signature


def run_decision(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0128 k49 rescue decision"
    )
    control_core, control_core_signature = _read_exact_receipt(
        str(job["control_core"]),
        str(job["control_core_sha256"]),
        label="accepted R0108 control core",
        schema="round0108-diverse-jina-core-geometry-v1",
        round_id="0108",
    )
    control_ood, control_ood_signature = _read_exact_receipt(
        str(job["control_ood"]),
        str(job["control_ood_sha256"]),
        label="accepted R0108 control OOD",
        schema="round0108-diverse-jina-ood-evaluation-v1",
        round_id="0108",
    )
    treatment_core_path = os.path.join(
        str(job["core_output"]), "core-geometry.json"
    )
    treatment_ood_path = os.path.join(
        str(job["ood_output"]), "ood-evaluation.json"
    )
    treatment_core = read_sealed(
        treatment_core_path, label="R0128 treatment core", schema=CORE_SCHEMA
    )
    treatment_ood = read_sealed(
        treatment_ood_path, label="R0128 treatment OOD", schema=OOD_SCHEMA
    )
    train_receipt_path = os.path.join(
        str(job["train_output"]), "train-receipt.json"
    )
    train_receipt = read_sealed(
        train_receipt_path,
        label="R0128 treatment train receipt",
        schema=TRAIN_RECEIPT_SCHEMA,
    )
    train_signature = expected_input_signature(train_receipt_path)
    isolation_path = os.path.join(
        str(job["train_output"]), "treatment-isolation.json"
    )
    isolation = read_sealed(
        isolation_path,
        label="R0128 treatment isolation",
        schema="round0128-k49-treatment-isolation-v1",
    )
    initial_state = train_receipt.get("initial_model_state") or {}
    if (
        train_receipt.get("round_id") != ROUND_ID
        or int(train_receipt.get("optimizer_updates", -1))
        != FIXED_SUCCESSFUL_UPDATES
        or (train_receipt.get("update_derivation") or {}).get("rule")
        != "fixed-R0107-dose-1459722-successful-updates"
        or initial_state.get("observed_sha256")
        != isolation.get("initial_state_evidence", {}).get("observed_sha256")
        or isolation.get("r0128_train_receipt") != train_signature
        or isolation.get("treatment_isolated") is not True
        or treatment_core.get("train_receipt") != train_signature
        or treatment_ood.get("train_receipt") != train_signature
    ):
        raise Round0128Error("R0128 treatment isolation/dose lineage changed")
    control_arrays_path = verify_signature(
        control_core.get("arrays"), label="R0108 control core arrays"
    )
    treatment_arrays_path = verify_signature(
        treatment_core.get("arrays"), label="R0128 treatment core arrays"
    )
    with np.load(control_arrays_path, allow_pickle=False) as archive:
        control_arrays = {
            key: np.asarray(archive[key])
            for key in (
                "global_anchor_rows",
                "compact_anchor_rows",
                "group_ids",
                "high_radius",
                "low_radius",
                "anchor_family_sizes",
            )
        }
    with np.load(treatment_arrays_path, allow_pickle=False) as archive:
        treatment_arrays = {
            key: np.asarray(archive[key])
            for key in control_arrays
        }
    for key in (
        "global_anchor_rows",
        "compact_anchor_rows",
        "group_ids",
        "high_radius",
        "anchor_family_sizes",
    ):
        if not np.array_equal(control_arrays[key], treatment_arrays[key]):
            raise Round0128Error(f"paired native anchor {key} changed")
    density_mask = control_arrays["anchor_family_sizes"] < FAMILY_SIZE_CUTOFF
    density, bootstrap = paired_density_materiality(
        control_high_radius=control_arrays["high_radius"][density_mask],
        control_low_radius=control_arrays["low_radius"][density_mask],
        treatment_high_radius=treatment_arrays["high_radius"][density_mask],
        treatment_low_radius=treatment_arrays["low_radius"][density_mask],
        draws=DENSITY_BOOTSTRAP_DRAWS,
        seed=DENSITY_BOOTSTRAP_SEED,
    )
    noninferiority = noninferiority_checks(
        control_core=control_core,
        treatment_core=treatment_core,
        control_ood=control_ood,
        treatment_ood=treatment_ood,
    )
    treatment_floor_check = bool(
        ((treatment_core.get("decision") or {}).get("checks") or {}).get(
            "density_v2_clears_registered_jina_floor"
        )
    )
    material = (
        density["outcome"] == "k49-materially-improves-native-density"
    )
    causal_release = material and bool(noninferiority["passed"])
    floor_rescue = causal_release and treatment_floor_check
    matched_path = os.path.join(
        str(job["matched_density_output"]), "matched-density.json"
    )
    matched = read_sealed(
        matched_path,
        label="R0128 matched R0040 diagnostic",
        schema=MATCHED_DENSITY_SCHEMA,
    )
    arrays_path = os.path.join(output, "decision-arrays.npz")
    atomic_save_new_npz(
        arrays_path,
        immutable=True,
        density_anchor_mask=density_mask,
        control_high_radius=control_arrays["high_radius"][density_mask],
        control_low_radius=control_arrays["low_radius"][density_mask],
        treatment_high_radius=treatment_arrays["high_radius"][density_mask],
        treatment_low_radius=treatment_arrays["low_radius"][density_mask],
        paired_density_bootstrap=bootstrap,
    )
    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "control": {
            "round_id": "0108",
            "nonself_graph_degree": 15,
            "core": control_core_signature,
            "ood": control_ood_signature,
        },
        "treatment": {
            "round_id": ROUND_ID,
            "nonself_graph_degree": GRAPH_K,
            "core": expected_input_signature(treatment_core_path),
            "ood": expected_input_signature(treatment_ood_path),
            "train_receipt": train_signature,
            "treatment_isolation": expected_input_signature(isolation_path),
        },
        "paired_native_density": {
            **density,
            "high_dimensional_radius_neighbors": K_DENSITY,
            "density_eligible_family_size_cutoff_exclusive": (
                FAMILY_SIZE_CUTOFF
            ),
            "anchors": int(density_mask.sum()),
        },
        "frozen_native_floor_clearance": {
            "passed": treatment_floor_check,
            "role": (
                "separate certification result; not substituted for paired "
                "delta materiality"
            ),
        },
        "native_and_ood_noninferiority": noninferiority,
        "matched_r0040_density": {
            "receipt": expected_input_signature(matched_path),
            "correlation": matched["cell"]["density_v2"]["correlation"],
            "clears_unchanged_registered_floor": matched["cell"][
                "clears_unchanged_registered_floor"
            ],
            "role": "diagnostic-only; excluded from selector",
        },
        "projection_ffr": {
            "role": "diagnostic-only; excluded from selector",
            "control": (control_ood.get("cross_atlas_alignment") or {}),
            "treatment": (treatment_ood.get("cross_atlas_alignment") or {}),
        },
        "decision": {
            "graph_degree_causal_rescue_supported": causal_release,
            "native_floor_rescue_supported": floor_rescue,
            "outcome": (
                "k49-material-density-rescue-with-noninferior-quality"
                if causal_release
                else "k49-rescue-not-released"
            ),
        },
        "capability_releases": [
            *(
                ["capability:jina-diverse-25m-native-k49-degree-rescue-v1"]
                if causal_release else []
            ),
            *(
                ["capability:jina-diverse-25m-k49-atlas-quality-v1"]
                if floor_rescue else []
            ),
        ],
        "registry_or_render_publication_performed": False,
        "registry_or_render_follow_up_required": floor_rescue,
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "map_decision_made": True,
    })
    path = os.path.join(output, "k49-rescue-decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0128Error("R0128 handler requires its exact round/job")
    handlers = {
        "qualify_k49_selected_policy": run_qualify_k49,
        "build_graph_part": run_graph_part,
        "assemble_graph": run_graph_assembly,
        "train_k49_treatment": run_train,
        "transform_retained_map": run_transform,
        "score_core_geometry": run_core,
        "score_ood": run_ood,
        "score_matched_r0040_density": run_matched_density,
        "decide_k49_rescue": run_decision,
    }
    try:
        handler = handlers[str(job["action"])]
    except KeyError as exc:
        raise Round0128Error(f"unknown R0128 action: {job.get('action')!r}") from exc
    return handler(active, job)
