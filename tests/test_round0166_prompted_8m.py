"""Pure tests for the prompted-English 8M scale contract."""
from __future__ import annotations

from basemap.round0166_prompted_8m import METRICS, scale_decision, scale_train_config
from experiments.round0166_nodes import (
    _data_identity,
    _full_text_layouts,
    _query_payload_inputs,
)


def _metrics(value: float) -> dict[str, float]:
    return {name: value for name in METRICS}


def test_scale_config_changes_population_bindings_not_dose() -> None:
    signature = {"canonical_path": "/tmp/graph", "sha256": "a" * 64}
    config, digest = scale_train_config(
        graph_signature=signature,
        graph_manifest_signature={"canonical_path": "/tmp/manifest", "sha256": "b" * 64},
        graph_edges=123,
        retained_rows=7_950_000,
    )
    assert config["input"]["rows"] == 7_950_000
    assert config["optimizer"]["successful_positive_lr_updates"] == 500_000
    assert config["graph"]["k"] == 50
    assert len(digest) == 64


def test_scale_decision_requires_native_and_retention_gates() -> None:
    accepted = scale_decision(
        native=_metrics(1.0),
        matched_2m=_metrics(0.98),
        baseline_2m=_metrics(1.0),
        prompted_floors=_metrics(0.9),
    )
    assert accepted["passed"] is True
    failed = scale_decision(
        native=_metrics(1.0),
        matched_2m={**_metrics(0.98), "heldout_recall_at_10": 0.96},
        baseline_2m=_metrics(1.0),
        prompted_floors=_metrics(0.9),
    )
    assert failed["passed"] is False
    assert failed["matched_2m_retention_gates"]["heldout_recall_at_10"]["passed"] is False


def test_full_text_layout_reaches_post_8m_query_reserve() -> None:
    signature_a = {"canonical_path": "/a.parquet", "bytes": 1, "sha256": "a" * 64}
    signature_b = {"canonical_path": "/b.parquet", "bytes": 2, "sha256": "b" * 64}
    first = {"source_layout": [{
        "corpus_global_row_start": 0,
        "corpus_global_row_stop": 5_727_340,
        "shard_row_start": 0,
        "shard_rows": 5_727_340,
        "text": signature_a,
        "text_column": "text",
    }]}
    second = {"source_layout": [{
        "r0087_global_row_start": 5_727_340,
        "r0087_global_row_stop": 9_126_376,
        "shard_row_start": 0,
        "shard_rows": 3_399_036,
        "text": signature_b,
        "text_column": "text",
    }]}
    text_layout = _full_text_layouts(first, second)
    document = {"canonical_path": "/document.npy", "bytes": 3, "sha256": "c" * 64}
    layout = {"chunks": [{
        "canonical_row_range": [7_990_000, 8_010_000],
        "staged_output": document,
    }]}
    assert _query_payload_inputs(layout, text_layout) == [document, signature_b]


def test_native_reference_identity_binds_corrected_population_bytes() -> None:
    population = {
        "retained_rows": 7_952_419,
        "document_compact": {
            "canonical_path": "/data/document-compact.f16",
            "bytes": 12_214_915_584,
            "sha256": "d" * 64,
        },
    }
    assert _data_identity(population) == {
        "kind": "ordered_shards",
        "shape": [7_952_419, 768],
        "dtype": "<f2",
        "shards": [{
            "position": 0,
            "name": "document-compact.f16",
            "bytes": 12_214_915_584,
            "sha256": "d" * 64,
        }],
    }
