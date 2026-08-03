"""Contract tests for the training-disjoint R0178 recovery."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from basemap.artifact_identity import expected_input_signature, sha256_bytes
from basemap import round0167_prompted_universality as contract_base
from basemap.round0178_prompted_universality import (
    CAPABILITY,
    PROMPTED_MAP_ORDER,
    ROUND_ID,
)
from experiments import round0167_nodes, round0176_nodes, round0178_nodes


def _hashes(texts: list[str]) -> np.ndarray:
    return np.asarray(
        [hashlib.sha256(text.encode("utf-8")).digest() for text in texts],
        dtype="V32",
    )


def test_r0178_text_copy_identity_catches_byte_fragility() -> None:
    mask, audit = round0178_nodes._text_copy_mask(
        ["same", "other"], ["same", "heldout"]
    )
    assert mask.tolist() == [True, False]
    assert audit["query_rows_with_corpus_copy"] == 1
    assert audit["corpus_query_disjoint"] is False

    corpus_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    query_vectors = np.asarray(
        [[1.0, np.float16(2 ** -14)], [0.5, 0.5]], dtype=np.float16
    )
    byte_mask, _byte_audit = round0167_nodes.exact_reference_copy_mask(
        corpus_vectors, query_vectors
    )
    assert byte_mask.tolist() == [False, False]


def test_r0178_selector_is_ordered_unique_and_training_disjoint(
    tmp_path, monkeypatch
) -> None:
    texts = ["a", "training", "b", "a", "c", "unused"]
    parquet = tmp_path / "source.parquet"
    pq.write_table(pa.table({"chunk_text": texts}), parquet)
    training_values = np.sort(_hashes(["training"]), kind="stable")
    training = tmp_path / "training.npy"
    np.save(training, training_values, allow_pickle=False)
    expected_rows = np.asarray([0, 2, 4], dtype=np.int64)
    monkeypatch.setattr(round0178_nodes, "CONTROL_ROWS", 3)
    monkeypatch.setattr(
        round0178_nodes, "EXPECTED_CONTROL_ROWS_SCANNED", 5
    )
    monkeypatch.setattr(
        round0178_nodes, "EXPECTED_CONTROL_TRAINING_TEXT_REJECTS", 1
    )
    monkeypatch.setattr(
        round0178_nodes, "EXPECTED_CONTROL_DUPLICATE_TEXT_REJECTS", 1
    )
    monkeypatch.setattr(
        round0178_nodes,
        "EXPECTED_CONTROL_SELECTION_SHA256",
        sha256_bytes(expected_rows.astype("<i8").tobytes(order="C")),
    )
    output = tmp_path / "output"
    round0178_nodes.run_select_disjoint_control(
        {"manifest": {"release_sha": "a" * 40}},
        {
            "outputs": [str(output)],
            "text_source": expected_input_signature(parquet),
            "training_text_hashes": {
                "training": expected_input_signature(training)
            },
        },
    )
    with open(output / "selector.json", encoding="utf-8") as handle:
        receipt = json.load(handle)
    rows = np.load(output / "selected-source-rows.i64.npy")
    selected_hashes = np.load(output / "selected-text-sha256.v32.npy")
    assert np.array_equal(rows, expected_rows)
    assert np.array_equal(selected_hashes, _hashes(["a", "b", "c"]))
    assert receipt["training_text_rejects"] == 1
    assert receipt["within_control_duplicate_text_rejects"] == 1
    assert receipt["training_text_disjoint"] is True
    assert receipt["source_text_unique"] is True


def test_r0178_dispatch_enables_external_text_sensitivity(monkeypatch) -> None:
    contract_names = (
        "ROUND_ID",
        "CAPABILITY",
        "PROMPTED_MAP_ORDER",
        "Round0167Error",
    )
    node_names = (
        *contract_names,
        "CANARY_SCHEMA",
        "PROBE_SCHEMA",
        "CONTROL_SCHEMA",
        "MAP_PANEL_SCHEMA",
        "ALLOW_CROSS_SPLIT_FAMILIES",
        "DUPLICATE_SENSITIVITY",
    )
    contract_before = {
        name: getattr(contract_base, name) for name in contract_names
    }
    node_before = {
        name: getattr(round0167_nodes, name) for name in node_names
    }
    audit_names = (
        "ROUND_ID",
        "CAPABILITY",
        "PROMPTED_MAP_ORDER",
        "Round0176Error",
        "exact_training_overlap_report",
    )
    audit_before = {
        name: getattr(round0176_nodes, name) for name in audit_names
    }
    observed = {}
    monkeypatch.setattr(
        round0167_nodes,
        "run_job",
        lambda active, job: observed.update({
            "round": round0167_nodes.ROUND_ID,
            "maps": round0167_nodes.PROMPTED_MAP_ORDER,
            "allow": round0167_nodes.ALLOW_CROSS_SPLIT_FAMILIES,
            "sensitivity": round0167_nodes.DUPLICATE_SENSITIVITY,
            "schema": round0167_nodes.MAP_PANEL_SCHEMA,
        }),
    )
    try:
        round0178_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}}, {"action": "assemble"}
        )
        assert observed == {
            "round": ROUND_ID,
            "maps": PROMPTED_MAP_ORDER,
            "allow": True,
            "sensitivity": True,
            "schema": "round0178-prompted-universality-map-panel-v1",
        }
        assert CAPABILITY == "jina-prompted-universality-panel-v1"
    finally:
        for name, value in contract_before.items():
            setattr(contract_base, name, value)
        for name, value in node_before.items():
            setattr(round0167_nodes, name, value)
        for name, value in audit_before.items():
            setattr(round0176_nodes, name, value)
