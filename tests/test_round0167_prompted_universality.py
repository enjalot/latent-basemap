from __future__ import annotations

import json

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0142_jina_universality import PROBE_ORDER
from basemap.round0167_prompted_universality import (
    CONTROL_QUERY_ID_OFFSET,
    PROMPTED_MAP_ORDER,
    QUERY_ID_OFFSET,
    Round0167Error,
    control_rows_from_coordinate_archive,
    retention_verdict,
    source_rows_from_coordinate_archive,
    twonn_correlations,
)
from experiments.round0167_nodes import _arrow_texts, _exact_family_audit


def test_source_rows_decode_exact_r0142_offsets() -> None:
    corpus, queries = source_rows_from_coordinate_archive(
        np.array([1, 4, 8, 12, 20] * 10, dtype=np.int64)
        + np.repeat(np.arange(10, dtype=np.int64) * 100, 5),
        QUERY_ID_OFFSET + np.arange(1000, 1010, dtype=np.int64),
        label="probe",
    )
    assert len(corpus) == 50
    assert np.array_equal(queries, np.arange(1000, 1010))


def test_source_rows_reject_cross_split_overlap() -> None:
    with pytest.raises(Round0167Error, match="overlap"):
        source_rows_from_coordinate_archive(
            np.arange(50, dtype=np.int64),
            QUERY_ID_OFFSET + np.arange(10, dtype=np.int64),
            label="probe",
        )


def test_separate_sources_allow_equal_numeric_rows() -> None:
    corpus, queries = source_rows_from_coordinate_archive(
        np.arange(50, dtype=np.int64),
        QUERY_ID_OFFSET + np.arange(10, dtype=np.int64),
        label="beir",
        separate_sources=True,
    )
    assert np.array_equal(corpus[:10], queries)


def test_control_rows_decode_1p5b_offset() -> None:
    corpus, queries = control_rows_from_coordinate_archive(
        np.arange(50, dtype=np.int64),
        CONTROL_QUERY_ID_OFFSET + np.arange(100, 110, dtype=np.int64),
        label="control",
    )
    assert np.array_equal(corpus, np.arange(50))
    assert np.array_equal(queries, np.arange(100, 110))


def test_retention_verdict_boundaries() -> None:
    assert retention_verdict(0.70) == "pass"
    assert retention_verdict(0.50) == "amber"
    assert retention_verdict(0.499) == "named-failure"
    with pytest.raises(Round0167Error):
        retention_verdict(float("nan"))


def test_twonn_correlations_require_complete_map_probe_matrix() -> None:
    cells = []
    for map_index, map_key in enumerate(PROMPTED_MAP_ORDER):
        for index, probe in enumerate(PROBE_ORDER):
            cells.append({
                "map": map_key,
                "probe": probe,
                "twonn_intrinsic_dimension": float(index + 1),
                "ffr_retention": float(100 - index + map_index),
                "recall10_retention": float(50 - index + map_index),
            })
    output = twonn_correlations(cells)
    assert len(output) == 8
    assert all(
        np.isclose(item["spearman_rho"], -1.0)
        for item in output
        if item["scope"] != "pooled-descriptive"
    )
    assert all(
        item["spearman_rho"] < 0
        for item in output
        if item["scope"] == "pooled-descriptive"
    )
    with pytest.raises(Round0167Error, match="incomplete"):
        twonn_correlations(cells[:-1])


def test_exact_family_audit_rejects_cross_split_copy() -> None:
    rng = np.random.RandomState(167)
    corpus = rng.normal(size=(50, 768)).astype(np.float16)
    queries = rng.normal(size=(10, 768)).astype(np.float16)
    receipt = _exact_family_audit(corpus, queries)
    assert receipt["cross_split_family_overlap"] == 0
    queries[0] = corpus[0]
    with pytest.raises(Round0167Error, match="overlap"):
        _exact_family_audit(corpus, queries)


def test_arrow_text_loader_preserves_requested_id_order(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    path = tmp_path / "corpus.arrow"
    table = pa.table({
        "_id": ["a", "b", "c"],
        "title": ["A", "B", "C"],
        "text": ["alpha", "beta", "gamma"],
    })
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    signature = expected_input_signature(str(path))
    assert _arrow_texts(
        signature,
        wanted_ids=["c", "a"],
        label="test",
        include_title=True,
    ) == ["C gamma", "A alpha"]


def test_beir_id_file_contract_is_json_string_list(tmp_path) -> None:
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(["x", "y"]), encoding="utf-8")
    assert expected_input_signature(str(path))["bytes"] > 0
