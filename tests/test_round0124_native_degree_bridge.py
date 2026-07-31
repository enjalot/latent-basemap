"""Contract and bounded CPU smoke tests for R0124."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.panel_v2 import PanelV2Config, score_panel
from basemap.round0113_prompt_contrast import (
    _DYNAMIC_PIPELINE_COUNTERS,
    read_sealed,
    train_config as r0115_train_config,
)
from basemap.round0124_degree_bridge import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DIAGNOSTIC_SCHEMA,
    GRAPH_DEGREE,
    GRAPH_SEARCH_NEIGHBORS,
    NATIVE_DENSITY_SCHEMA,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_MATERIAL,
    OUTCOME_NOT_MATERIAL,
    classify_degree_bridge,
    paired_density_bootstrap,
    train_config,
    training_loop_plan,
)
from experiments import round0124_nodes


def _signature(path: str = "/tmp/r0124-graph") -> dict[str, Any]:
    return {
        "kind": "file",
        "canonical_path": path,
        "bytes": 123,
        "sha256": "a" * 64,
    }


def test_only_registered_training_change_is_graph_degree() -> None:
    graph = _signature()
    manifest = _signature("/tmp/r0124-manifest")
    control, _ = r0115_train_config(
        "raw",
        graph_signature=graph,
        graph_manifest_signature=manifest,
        graph_edges=100,
        retained_rows=1_993_761,
    )
    treatment, _ = train_config(
        graph_signature=graph,
        graph_manifest_signature=manifest,
        graph_edges=100,
        retained_rows=1_993_761,
    )
    assert treatment["input"] == control["input"]
    assert treatment["model"] == control["model"]
    assert treatment["optimizer"] == control["optimizer"]
    assert {
        key: value
        for key, value in treatment["graph"].items()
        if key not in {"k", "n_neighbors_including_self"}
    } == {
        key: value
        for key, value in control["graph"].items()
        if key != "k"
    }
    assert treatment["graph"]["k"] == GRAPH_DEGREE
    assert (
        treatment["graph"]["n_neighbors_including_self"]
        == GRAPH_SEARCH_NEIGHBORS
    )
    for key in (
        "device_count",
        "required_pipeline",
        "gpu_resident_data",
        "gpu_resident_vram_budget_gb",
        "minimum_train_upd_s",
        "warning_train_upd_s",
        "performance_subfloor_patience",
        "performance_windows",
    ):
        assert treatment["execution"][key] == control["execution"][key]
    stamp = treatment["execution"]["expected_pipeline_stamp"]
    control_stamp = control["execution"]["expected_pipeline_stamp"]
    shared_stamp_changes = {
        key
        for key in set(stamp) & set(control_stamp)
        if stamp[key] != control_stamp[key]
    }
    assert shared_stamp_changes == {
        "graph_degree",
        "positive_destination_policy",
    }
    assert set(stamp) - set(control_stamp) == {
        "device_conversion",
        "feature_residency",
        "graph",
        "graph_nonself_degree",
        "graph_search_neighbors_including_self",
    }
    assert stamp["source_representation"] == control_stamp[
        "source_representation"
    ]
    assert stamp["multiplicity_policy"] == control_stamp[
        "multiplicity_policy"
    ]
    assert stamp["sampler_class"] == "PromptWeightedJinaSampler"
    assert stamp["positive_sampling"] == control_stamp["positive_sampling"]
    assert stamp["graph_degree"] == "variable-symmetric-fuzzy-k15-topology"


def test_k15_training_loop_plan_covers_registered_horizon() -> None:
    plan = training_loop_plan(graph_edges=46_065_518)
    assert plan["batches_per_epoch"] == 112_630
    assert plan["n_epochs"] == 5
    assert plan["planned_loop_iters"] == 563_150
    assert plan["planned_loop_iters"] >= plan["successful_positive_lr_updates"]


def test_explicit_self_knn_enforces_15_distinct_nonself() -> None:
    rows = np.asarray([4, 20], dtype=np.int64)
    ids = np.vstack(
        (
            np.asarray([9, 4, *range(10, 24)], dtype=np.int64),
            np.asarray([20, *range(30, 45)], dtype=np.int64),
        )
    )
    similarities = np.linspace(1.0, 0.5, ids.size).reshape(ids.shape)
    canonical, distances, nonself = round0124_nodes._explicit_self_knn(
        ids, similarities, rows
    )
    assert canonical.shape == (2, 16)
    assert np.array_equal(canonical[:, 0], rows)
    assert nonself.shape == (2, 15)
    assert not np.any(nonself == rows[:, None])
    assert np.all(distances[:, 0] == 0)

    malformed = ids.copy()
    malformed[0, 1] = 8
    with pytest.raises(
        round0124_nodes.Round0124Error, match="exactly one self"
    ):
        round0124_nodes._explicit_self_knn(
            malformed, similarities, rows
        )


@pytest.mark.parametrize(
    ("interval", "outcome"),
    [
        ((-0.05, -0.03), OUTCOME_MATERIAL),
        ((-0.029, -0.01), OUTCOME_NOT_MATERIAL),
        ((-0.03, -0.01), OUTCOME_INCONCLUSIVE),
        ((-0.04, -0.02), OUTCOME_INCONCLUSIVE),
    ],
)
def test_degree_selector_is_native_delta_only(
    interval: tuple[float, float], outcome: str
) -> None:
    result = classify_degree_bridge(
        control_density=0.23,
        treatment_density=0.19,
        delta_ci_low=interval[0],
        delta_ci_high=interval[1],
    )
    assert result["outcome"] == outcome
    assert result["core_and_ood_diagnostics_can_rescue_or_fail"] is False
    assert result["legacy_density_floor_used"] is False


def test_paired_native_density_bootstrap_is_frozen_and_deterministic() -> None:
    high = np.geomspace(0.2, 4.0, 64)
    control = high * np.linspace(0.8, 1.2, 64)
    treatment = np.random.RandomState(124).lognormal(size=64)
    first = paired_density_bootstrap(
        high_radius=high,
        control_low_radius=control,
        treatment_low_radius=treatment,
    )
    second = paired_density_bootstrap(
        high_radius=high,
        control_low_radius=control,
        treatment_low_radius=treatment,
    )
    assert first["paired_bootstrap_draws"] == BOOTSTRAP_DRAWS
    assert first["paired_bootstrap_seed"] == BOOTSTRAP_SEED
    assert np.array_equal(
        first["bootstrap_deltas"], second["bootstrap_deltas"]
    )
    assert first["paired_bootstrap_delta_ci"] == second[
        "paired_bootstrap_delta_ci"
    ]
    assert first["outcome"] == OUTCOME_MATERIAL


def test_decision_diagnostics_cannot_rescue_density(
    tmp_path: Path,
) -> None:
    density_root = tmp_path / "density"
    diagnostic_root = tmp_path / "diagnostic"
    density_root.mkdir()
    diagnostic_root.mkdir()
    from basemap.round0113_prompt_contrast import seal

    selector = classify_degree_bridge(
        control_density=0.23,
        treatment_density=0.18,
        delta_ci_low=-0.06,
        delta_ci_high=-0.04,
    )
    diagnostic = seal(
        {
            "schema": DIAGNOSTIC_SCHEMA,
            "round_id": "0124",
            "release_sha": "b" * 40,
            "arm": "raw",
            "execution_gates": {
                "finite_noncollapsed_coordinates": True,
                "transductive_recall50_gt_recall10": True,
                "matched_projection_recall50_gt_recall10": True,
                "exact_update_closure": True,
                "zero_numerical_skips": True,
                "no_pipeline_stamp_drift": True,
            },
            # Deliberately excellent diagnostics; they cannot alter density.
            "metrics": {"density": 1.0, "ffr": 1.0},
            "ood": {"pol_Latn": {"matched": 1.0}},
        }
    )
    (diagnostic_root / "score.json").write_text(
        json.dumps(diagnostic), encoding="utf-8"
    )
    density = seal(
        {
            "schema": NATIVE_DENSITY_SCHEMA,
            "round_id": "0124",
            "release_sha": "b" * 40,
            "control": {"density": 0.23},
            "treatment": {
                "density": 0.18,
                "diagnostics": expected_input_signature(
                    str(diagnostic_root / "score.json")
                ),
            },
            "registered_selector": selector,
        }
    )
    (density_root / "native-density-score.json").write_text(
        json.dumps(density), encoding="utf-8"
    )
    result = round0124_nodes.run_decision(
        {"manifest": {"round_id": "0124", "release_sha": "b" * 40}},
        {
            "outputs": [str(tmp_path / "decision")],
            "density_output": str(density_root),
            "diagnostic_output": str(diagnostic_root),
        },
    )
    assert result["registered_selector"]["outcome"] == OUTCOME_MATERIAL
    assert result["diagnostics_can_rescue_or_fail_selector"] is False
    assert result["scale_contribution_excluded"] is True
    read_sealed(
        str(tmp_path / "decision" / "decision.json"),
        label="R0124 smoke decision",
    )


def test_decision_rejects_missing_execution_gates(tmp_path: Path) -> None:
    from basemap.round0113_prompt_contrast import seal

    density_root = tmp_path / "density"
    diagnostic_root = tmp_path / "diagnostic"
    density_root.mkdir()
    diagnostic_root.mkdir()
    selector = classify_degree_bridge(
        control_density=0.23,
        treatment_density=0.23,
        delta_ci_low=-0.01,
        delta_ci_high=0.01,
    )
    (diagnostic_root / "score.json").write_text(
        json.dumps(
            seal(
                {
                    "schema": DIAGNOSTIC_SCHEMA,
                    "round_id": "0124",
                    "release_sha": "b" * 40,
                    "arm": "raw",
                    "execution_gates": {},
                }
            )
        ),
        encoding="utf-8",
    )
    (density_root / "native-density-score.json").write_text(
        json.dumps(
            seal(
                {
                    "schema": NATIVE_DENSITY_SCHEMA,
                    "round_id": "0124",
                    "release_sha": "b" * 40,
                    "control": {"density": 0.23},
                    "treatment": {
                        "density": 0.23,
                        "diagnostics": expected_input_signature(
                            str(diagnostic_root / "score.json")
                        ),
                    },
                    "registered_selector": selector,
                }
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        round0124_nodes.Round0124Error,
        match="decision evidence changed",
    ):
        round0124_nodes.run_decision(
            {
                "manifest": {
                    "round_id": "0124",
                    "release_sha": "b" * 40,
                }
            },
            {
                "outputs": [str(tmp_path / "decision")],
                "density_output": str(density_root),
                "diagnostic_output": str(diagnostic_root),
            },
        )


def test_train_seal_reload_panel_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Catch post-training receipt/panel crashes without allocating CUDA."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch

    updates = 201
    batch_size = 16
    positive_rows = 4
    expected_rows = updates * batch_size
    graph_signature = _signature(str(tmp_path / "graph.npz"))
    manifest_path = tmp_path / "graph-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest_signature = expected_input_signature(str(manifest_path))
    assembly_signature = _signature(str(tmp_path / "assembly.json"))
    graph = {
        "signature": graph_signature,
        "manifest_signature": manifest_signature,
        "manifest": {},
        "sources": np.arange(64, dtype=np.int32),
        "targets": np.roll(np.arange(64, dtype=np.int32), -1),
        "weights": np.ones(64, dtype=np.float32),
        "n_nodes": 240,
    }
    expected_stamp = {
        "schema": "round0124-cpu-smoke-v1",
        "pipeline": "host_weighted_jina_prompt_contrast",
        "sampler_class": "PromptWeightedJinaSampler",
    }
    config = {
        "model": {
            "architecture": "residual_bottleneck",
            "input_dimension": 8,
            "hidden_dimension": 16,
            "hidden_layers": 1,
            "output_dimension": 2,
            "use_batchnorm": False,
            "use_dropout": False,
            "low_dim_kernel": "legacy_lp",
            "a": 1.0,
            "b": 1.0,
        },
        "execution": {
            "expected_pipeline_stamp": expected_stamp,
            "minimum_train_upd_s": 1.0,
            "warning_train_upd_s": 1.0,
        },
    }
    accepted = updates * positive_rows + 3
    runtime = {
        **expected_stamp,
        "positive_sampling": (
            "fuzzy_weight_proportional_with_replacement_via_exact_"
            "uniform_envelope_rejection"
        ),
        "multiplicity_policy": (
            "shared-source-raw-document-union-representative-only"
        ),
        "feature_residency": "host-contiguous-compact-fp16-memmap",
        "source_representation": "raw-fp16",
        "device_conversion": "device-fp32-from-exact-fp16",
        "endpoint_gather_calls": updates,
        "source_rows_gathered": expected_rows,
        "destination_rows_gathered": expected_rows,
        "host_prefetch_batches_filled": updates,
        "host_prefetch_producer_batches": updates,
        "host_prefetch_consumer_batches": updates,
        "host_prefetch_source_rows_filled": expected_rows,
        "host_prefetch_destination_rows_filled": expected_rows,
        "weight_proposals": accepted + 1_000,
        "weight_acceptances": accepted,
        "weight_emitted_draws": updates * positive_rows,
        "weight_buffered_draws": 3,
        "weight_acceptance_rate": accepted / (accepted + 1_000),
        "weight_rejection_iterations": updates,
    }
    accounting: dict[str, Any] = {
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
        "n_pos_edges": len(graph["sources"]),
    }
    accounting["pipeline_runtime"] = dict(runtime)
    accounting.update({f"pipeline_{key}": value for key, value in runtime.items()})
    assert set(_DYNAMIC_PIPELINE_COUNTERS).issubset(runtime)

    class SmokeDataset:
        def __init__(self, *args, **kwargs) -> None:
            self.features = np.random.default_rng(121).normal(
                size=(240, 8)
            ).astype(np.float32)

    class SmokeWrapper:
        def __init__(self, dataset, graph) -> None:
            self.dataset = dataset

        def runtime_stamp(self) -> dict[str, Any]:
            return dict(runtime)

    class SmokeProfiler:
        def finalize(self, **kwargs) -> dict[str, Any]:
            return {"aborted": False, "cpu_smoke": True}

    class SmokeModel:
        def __init__(self) -> None:
            self.layer = torch.nn.Linear(8, 2)
            self._canary_profiler = SmokeProfiler()
            self._bench_seconds = 0.01
            self._setup_seconds = 0.001

        def fit(self, wrapper, **kwargs) -> None:
            X = torch.from_numpy(wrapper.dataset.features)
            target = X[:, :2] - 0.2 * X[:, 2:4]
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.05)
            for _ in range(4):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(self.layer(X), target)
                loss.backward()
                optimizer.step()
            self._train_stats = dict(accounting)

        def save(self, path: str) -> None:
            torch.save(
                {
                    "weight": self.layer.weight.detach(),
                    "bias": self.layer.bias.detach(),
                },
                path,
            )

    assembly = {"outputs": {"raw": _signature("/smoke/raw")}, "mapping": {}}
    monkeypatch.setattr(
        round0124_nodes.prompt_nodes,
        "_load_assembly",
        lambda job: (assembly, assembly_signature),
    )
    monkeypatch.setattr(
        round0124_nodes, "load_graph", lambda *args, **kwargs: graph
    )
    monkeypatch.setattr(
        round0124_nodes,
        "train_config",
        lambda **kwargs: (config, "c" * 64),
    )
    monkeypatch.setattr(
        round0124_nodes.prompt_nodes,
        "_open_compact",
        lambda assembly, arm: np.empty((240, 8), dtype=np.float16),
    )
    monkeypatch.setattr(round0124_nodes, "HostFp16EndpointArray", SmokeDataset)
    monkeypatch.setattr(
        round0124_nodes, "DegreeBridgeTrainingInput", SmokeWrapper
    )
    monkeypatch.setattr(
        round0124_nodes, "_new_model", lambda config: SmokeModel()
    )
    monkeypatch.setattr(round0124_nodes, "SUCCESSFUL_UPDATES", updates)
    monkeypatch.setattr(round0124_nodes, "BATCH_SIZE", batch_size)
    monkeypatch.setattr(
        round0124_nodes, "PERFORMANCE_WARMUP_UPDATES", 1
    )
    monkeypatch.setattr(
        round0124_nodes.prompt_nodes,
        "_weighted_rejection_accounting_mismatch",
        lambda runtime, producer_delta: None,
    )
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: (1_000_000_000, 2_000_000_000),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1_024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2_048)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    output = tmp_path / "train"
    result = round0124_nodes.run_train(
        {"manifest": {"round_id": "0124", "release_sha": "d" * 40}},
        {
            "outputs": [str(output)],
            "graph_manifest": str(tmp_path / "graph-manifest.json"),
        },
    )
    assert result["optimizer_updates"] == updates
    receipt = read_sealed(
        str(output / "train-receipt.json"), label="R0124 CPU smoke train"
    )
    assert receipt["train_checks"]["weighted_rejection_accounting_closes"]

    checkpoint = torch.load(
        output / "model.pt", map_location="cpu", weights_only=True
    )
    features = SmokeDataset().features
    coordinates = torch.nn.functional.linear(
        torch.from_numpy(features),
        checkpoint["weight"],
        checkpoint["bias"],
    ).detach().numpy()
    panel = score_panel(
        features,
        coordinates,
        config=PanelV2Config(
            frac=0.1,
            k_hit=3,
            k_density=3,
            n_anchors=24,
            corpus_chunk=64,
            overselect=4,
            block_elems=100_000,
            rerank_byte_cap=8_000_000,
            peak_byte_cap=16_000_000,
        ),
        provenance={"round": "0124", "mode": "cpu-smoke"},
    )
    assert panel["guards"]["coords_finite"] is True
    assert panel["guards"]["coords_collapsed"] is False

    # Exercise the actual post-fit authentication and R0115 diagnostic
    # adapter. Only model compute and corpus size are reduced.
    from basemap.panel_v2 import build_hiD_reference
    from basemap.pumap.parametric_umap import ParametricUMAP

    class LoadedSmokeModel:
        architecture = "residual_bottleneck"
        input_dim = 8
        hidden_dim = 16
        n_layers = 1
        n_components = 2
        use_batchnorm = False
        use_dropout = False
        low_dim_kernel = "legacy_lp"
        a = 1.0
        b = 1.0

        def __init__(self, checkpoint: Mapping[str, Any]) -> None:
            self.weight = checkpoint["weight"].numpy()
            self.bias = checkpoint["bias"].numpy()

        def transform(self, values, batch_size: int) -> np.ndarray:
            data = np.asarray(values, dtype=np.float32)
            return data @ self.weight.T + self.bias

    monkeypatch.setattr(
        ParametricUMAP,
        "load",
        staticmethod(
            lambda path, device: LoadedSmokeModel(
                torch.load(path, map_location="cpu", weights_only=True)
            )
        ),
    )
    auth_job = {
        "graph_manifest": str(tmp_path / "graph-manifest.json"),
        "train_output": str(output),
    }
    loaded, authenticated, _assembly, _graph = (
        round0124_nodes._authenticate_treatment_model(
            {"manifest": {"round_id": "0124", "release_sha": "d" * 40}},
            auth_job,
        )
    )
    assert authenticated["optimizer_updates"] == updates
    assert loaded.input_dim == 8

    prompt = round0124_nodes.prompt_nodes
    monkeypatch.setattr(prompt, "RETAINED_ROWS", 240)
    monkeypatch.setattr(prompt, "DIMENSION", 8)
    monkeypatch.setattr(prompt, "QUERY_CANDIDATES", 30)
    monkeypatch.setattr(prompt, "QUERY_ROWS", 20)
    monkeypatch.setattr(prompt, "POLISH_QUERY_ROWS", 10)
    panel_config = PanelV2Config(
        frac=0.1,
        k_clust=(),
        k_hit=10,
        k_density=GRAPH_DEGREE,
        n_anchors=24,
        anchor_seed=123,
        corpus_chunk=64,
        overselect=4,
        block_elems=100_000,
        rerank_byte_cap=8_000_000,
        peak_byte_cap=16_000_000,
    )
    monkeypatch.setattr(prompt, "panel_config", lambda: panel_config)
    data_identity = {
        "kind": "ordered_array",
        "shape": [240, 8],
        "dtype": np.dtype("<f4").str,
        "sha256": ordered_array_sha256(features),
    }
    monkeypatch.setattr(
        prompt, "_data_identity", lambda assembly, arm: data_identity
    )
    convention = {
        "row_order": (
            "R0113 shared source/raw/document union-representative compact "
            "order"
        ),
        "distance": "cosine via fp32-L2-normalized squared L2",
        "self_exclusion": True,
        "anchor_namespace": "R0113 compact IDs",
        "embedding_prompt": "raw",
    }
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    reference = build_hiD_reference(
        normalized,
        np.sort(
            np.random.RandomState(123)
            .choice(240, 24, replace=False)
            .astype(np.int64)
        ),
        panel_config,
        centroids_by_k=None,
        data_identity=data_identity,
        convention=convention,
    )
    from basemap.panel_v2 import save_hiD_reference

    reference_path = tmp_path / "high-d-reference.npz"
    save_hiD_reference(reference, str(reference_path))
    reference_signature = expected_input_signature(str(reference_path))
    graph["manifest"] = {
        "high_d_reference": reference_signature,
        "high_d_reference_key": reference["key"],
    }
    import basemap.panel_v2 as panel_v2

    monkeypatch.setattr(
        panel_v2,
        "load_hiD_reference",
        lambda path, expected_key: reference,
    )
    query_rng = np.random.default_rng(1210)
    query_signatures: dict[str, Any] = {}
    polish_signatures: dict[str, Any] = {}
    for arm in ("raw", "document"):
        query_path = tmp_path / f"{arm}-queries.npy"
        polish_path = tmp_path / f"{arm}-polish.npy"
        np.save(
            query_path,
            query_rng.normal(size=(30, 8)).astype(np.float16),
        )
        np.save(
            polish_path,
            query_rng.normal(size=(10, 8)).astype(np.float16),
        )
        query_signatures[arm] = expected_input_signature(str(query_path))
        polish_signatures[arm] = expected_input_signature(str(polish_path))
    polish_rows_path = tmp_path / "polish-rows.npy"
    global_rows_path = tmp_path / "query-global-rows.npy"
    np.save(polish_rows_path, np.arange(10, dtype=np.int64))
    np.save(global_rows_path, np.arange(20, dtype=np.int64))
    query_receipt = {
        "outputs": query_signatures,
        "ood": {
            "pol_Latn": {
                "outputs": polish_signatures,
                "query_rows": expected_input_signature(str(polish_rows_path)),
                "source_text": {"name": "cpu-smoke-polish"},
            }
        },
    }
    query_receipt_signature = _signature("/smoke/query-receipt.json")
    selection = {
        "identity_sha256": "e" * 64,
        "global_rows": expected_input_signature(str(global_rows_path)),
    }
    selection_root = tmp_path / "selection"
    selection_root.mkdir()
    (selection_root / "query-selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    monkeypatch.setattr(
        prompt,
        "_load_query_reserve",
        lambda job: (query_receipt, query_receipt_signature),
    )
    monkeypatch.setattr(
        prompt,
        "_load_query_selection",
        lambda job: (selection, np.arange(20, dtype=np.int64)),
    )
    monkeypatch.setattr(
        prompt,
        "_open_compact",
        lambda assembly, arm: features,
    )
    diagnostic_root = tmp_path / "diagnostics"
    diagnostic = round0124_nodes.run_diagnostics(
        {"manifest": {"round_id": "0124", "release_sha": "d" * 40}},
        {
            **auth_job,
                "outputs": [str(diagnostic_root)],
                "arm": "raw",
                "query_output": "/smoke/query",
                "query_selection_output": str(selection_root),
        },
    )
    assert diagnostic["schema"] == DIAGNOSTIC_SCHEMA
    assert all(diagnostic["execution_gates"].values())
    read_sealed(
        str(diagnostic_root / "score.json"),
        label="R0124 CPU smoke diagnostics",
    )

    # Finish the real post-fit path: authenticate the sealed coordinates,
    # recompute matched native radii for both arms, bootstrap the paired
    # density delta, seal it, and consume it in the decision node.
    treatment_coordinates = np.load(
        diagnostic["coordinates"]["training"]["canonical_path"],
        allow_pickle=False,
    )
    control_coordinates_path = tmp_path / "control-coordinates.npy"
    np.save(control_coordinates_path, treatment_coordinates)
    control_coordinates_signature = expected_input_signature(
        str(control_coordinates_path)
    )
    control_score = {
        "coordinates": {"training": control_coordinates_signature},
        "metrics": {"density": diagnostic["metrics"]["density"]},
    }
    control_graph = {
        "high_d_reference": reference_signature,
        "high_d_reference_key": reference["key"],
    }
    monkeypatch.setattr(round0124_nodes, "RETAINED_ROWS", 240)
    monkeypatch.setattr(round0124_nodes, "NATIVE_DENSITY_ANCHORS", 24)
    monkeypatch.setattr(
        round0124_nodes,
        "_r0115_native_evidence",
        lambda job: (
            control_graph,
            _signature("/smoke/r0115-control-graph.json"),
            {"round_id": "0115"},
            _signature("/smoke/r0115-control-train.json"),
            control_score,
            _signature("/smoke/r0115-control-score.json"),
        ),
    )
    monkeypatch.setattr(
        round0124_nodes,
        "_context_evidence",
        lambda job: (
            _signature("/smoke/r0106-context.json"),
            _signature("/smoke/r0108-context.json"),
        ),
    )
    density_root = tmp_path / "native-density"
    density = round0124_nodes.run_native_density(
        {"manifest": {"round_id": "0124", "release_sha": "d" * 40}},
        {
            **auth_job,
            "outputs": [str(density_root)],
            "diagnostic_output": str(diagnostic_root),
        },
    )
    assert density["schema"] == NATIVE_DENSITY_SCHEMA
    assert density["registered_selector"]["outcome"] == OUTCOME_NOT_MATERIAL
    assert density["legacy_density_floor_used"] is False
    arrays_path = density["arrays"]["canonical_path"]
    with np.load(arrays_path, allow_pickle=False) as archive:
        assert archive["paired_bootstrap_deltas"].shape == (
            BOOTSTRAP_DRAWS,
        )
    read_sealed(
        str(density_root / "native-density-score.json"),
        label="R0124 CPU smoke native density",
    )
    decision_root = tmp_path / "decision"
    decision = round0124_nodes.run_decision(
        {"manifest": {"round_id": "0124", "release_sha": "d" * 40}},
        {
            "outputs": [str(decision_root)],
            "density_output": str(density_root),
            "diagnostic_output": str(diagnostic_root),
        },
    )
    assert decision["registered_selector"]["outcome"] == OUTCOME_NOT_MATERIAL
    assert decision["diagnostics_can_rescue_or_fail_selector"] is False
    assert decision["scale_contribution_excluded"] is True
    read_sealed(
        str(decision_root / "decision.json"),
        label="R0124 CPU smoke final decision",
    )
