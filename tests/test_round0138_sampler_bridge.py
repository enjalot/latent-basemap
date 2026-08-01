from __future__ import annotations

from copy import deepcopy

import pytest

from basemap.round0104_training import preprocessing_stamp, train_config as host_config
from basemap.round0138_sampler_bridge import (
    CELL_ORDER,
    CONTROL,
    HISTORICAL,
    PIPELINE,
    SAMPLER_CLASS,
    TREATMENT,
    Round0138Error,
    build_decision,
    train_config,
)
from experiments.prepare_round0138_queue import (
    GPU_HOURS_EXPECTED,
    GPU_HOURS_MAXIMUM,
    GPU_HOURS_MINIMUM,
    GPU_HOURS_P90,
    REVIEW_CAPABILITIES,
    R0134_DECISION,
    R0134_PANEL,
    R0104_SHARED,
    _preissuance_cpu_smoke,
    _read_json,
    _write_device_manifest,
)


GRAPH = {
    "canonical_path": "/tmp/graph.npz",
    "kind": "file",
    "bytes": 123,
    "sha256": "a" * 64,
}
MANIFEST = {
    "canonical_path": "/tmp/manifest.json",
    "kind": "file",
    "bytes": 456,
    "sha256": "b" * 64,
}
EDGES = 151_202_984


def _cell(value: float = 0.5) -> dict:
    return {
        "panel": {
            "ffr": value,
            "purity": {"k256": 1.0, "k1024": 1.0},
        },
        "projection": {"ffr": value, "recall_at_10": value},
    }


def _cells(*, historical: float, control: float, treatment: float) -> dict:
    return {
        HISTORICAL: _cell(historical),
        CONTROL: _cell(control),
        TREATMENT: _cell(treatment),
    }


def test_device_sampler_config_changes_only_registered_runtime_fields():
    host, _ = host_config(
        "fp16_control",
        graph_signature=GRAPH,
        graph_manifest_signature=MANIFEST,
        graph_edges=EDGES,
    )
    treatment, digest = train_config(
        graph_signature=GRAPH,
        graph_manifest_signature=MANIFEST,
        graph_edges=EDGES,
    )
    normalized = deepcopy(treatment)
    normalized["schema"] = host["schema"]
    normalized["arm"] = host["arm"]
    normalized.pop("causal_change")
    normalized["paired_invariant"]["sampler"] = host["paired_invariant"]["sampler"]
    normalized["execution"] = host["execution"]
    assert normalized == host
    assert treatment["execution"]["required_pipeline"] == PIPELINE == "device"
    assert treatment["execution"]["gpu_resident_data"] is True
    stamp = treatment["execution"]["expected_pipeline_stamp"]
    assert stamp["sampler_class"] == SAMPLER_CLASS == "DeviceEdgeSampler"
    assert stamp["positive_sampling"] == "weighted_with_replacement"
    assert stamp["positive_with_replacement"] is True
    assert stamp["weighted_effective"] is True
    assert stamp["x_residency"] == "device_fp16"
    assert stamp["identity_sha256"] == preprocessing_stamp("fp16_control")[
        "identity_sha256"
    ]
    assert len(digest) == 64


def test_pareto_restoration_stops_after_device_sampler():
    decision = build_decision(
        _cells(historical=0.6, control=0.5, treatment=0.6)
    )
    assert decision["outcome"] == "device-sampler-sufficient-to-restore-function"
    assert decision["device_sampler_sufficient"] is True
    assert decision["density_recalibration_authorized"] is False


def test_sampler_tradeoff_is_not_mislabeled_as_restoration():
    cells = _cells(historical=0.5, control=0.5, treatment=0.5)
    cells[HISTORICAL]["panel"]["purity"]["k256"] = 2.0
    cells[TREATMENT]["panel"]["purity"]["k256"] = 2.0
    decision = build_decision(cells)
    assert decision["restores_historical_on_all_metrics"] is True
    assert decision["preserves_current_control_on_all_metrics"] is False
    assert decision["device_sampler_sufficient"] is False
    assert decision["outcome"] == "device-sampler-regresses-current-control"


def test_sampler_nonrestoration_is_explicit():
    decision = build_decision(
        _cells(historical=0.6, control=0.5, treatment=0.55)
    )
    assert decision["outcome"] == "device-sampler-insufficient-to-restore-function"
    assert decision["device_sampler_sufficient"] is False


def test_canonical_json_key_order_restores_preregistered_cell_order():
    cells = _cells(historical=0.6, control=0.5, treatment=0.6)
    assert tuple(cells) == CELL_ORDER
    assert build_decision(dict(reversed(list(cells.items())))) == build_decision(cells)
    del cells[HISTORICAL]
    with pytest.raises(Round0138Error, match="missing or unexpected"):
        build_decision(cells)


def test_dependencies_and_budget_are_bounded():
    assert set(REVIEW_CAPABILITIES) == {
        "0037",
        "0103",
        "0104",
        "0122",
        "0134",
        "0137",
    }
    assert (
        GPU_HOURS_MINIMUM,
        GPU_HOURS_EXPECTED,
        GPU_HOURS_P90,
        GPU_HOURS_MAXIMUM,
    ) == (1.20, 1.55, 2.00, 2.50)


def test_exact_corrected_r0134_receipts_are_bound():
    assert "queue-attempt-3-exact-views" in R0134_PANEL
    assert "queue-attempt-5-decision-recovery-a3adb61" in R0134_DECISION


def test_preissuance_cpu_smoke_closes_train_receipt_to_selector_path(tmp_path):
    shared = _read_json(R0104_SHARED)
    graph = shared["graph"]
    parent = shared["graph_manifest"]
    manifest = _write_device_manifest(
        root=str(tmp_path), graph=graph, parent_signature=parent
    )
    receipt = _preissuance_cpu_smoke(
        graph=graph,
        device_manifest=manifest,
        graph_edges=int(shared["graph_edges"]),
    )
    assert receipt["passed"] is True
    assert receipt["cuda_used"] is False
    assert receipt["source_probe_shape"] == [32, 2]
    assert receipt["query_probe_shape"] == [32, 2]
    assert receipt["canonical_selector_outcome"] == (
        "device-sampler-insufficient-to-restore-function"
    )
