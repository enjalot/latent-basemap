from __future__ import annotations

import copy

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import basemap.round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import ordered_array_sha256
from basemap.round0113_prompt_contrast import (
    DECISION_METRICS,
    DIMENSION,
    HostFp16EndpointArray,
    NONINFERIORITY_RATIO,
    POLISH_QUERY_ROWS_SHA256,
    PromptWeightedJinaSampler,
    QUERY_CANDIDATES,
    QUERY_SCAN_START,
    RETAINED_ROWS,
    Round0113Error,
    compact_mapping,
    paired_decision,
    polish_query_rows,
    query_candidate_rows,
    query_source_layout,
    seal,
    train_config,
    validate_seal,
)
from experiments.round0113_nodes import (
    _data_identity,
    _exact_duplicate_audit,
    _fetch_parquet_rows,
    _exact_text_families,
    _require_unique_stored_rows,
    _sorted_hash_membership,
    _text_row_hashes,
    _union_prompt_exclusions,
)


def _signature(path: str, digest: str) -> dict[str, object]:
    return {
        "canonical_path": path,
        "kind": "file",
        "bytes": 1,
        "sha256": digest * 64,
    }


def _score(value: float, *, execution: bool = True) -> dict[str, object]:
    return {
        "metrics": {metric: value for metric in DECISION_METRICS},
        "execution_gates": {
            "finite": execution,
            "accounting": execution,
        },
    }


def test_seal_detects_contract_mutation():
    receipt = seal({"schema": "test", "rows": 2_000_000})
    validate_seal(receipt, label="test receipt")
    changed = copy.deepcopy(receipt)
    changed["rows"] = 1
    with pytest.raises(Round0113Error, match="identity seal"):
        validate_seal(changed, label="test receipt")


def test_real_query_reserve_is_fixed_and_training_family_clean():
    rows, report = query_candidate_rows()
    assert rows.shape == (QUERY_CANDIDATES,)
    assert rows.dtype == np.int64
    assert rows[0] == QUERY_SCAN_START
    assert rows[-1] == 2_004_123
    assert report["rejected_training_family_rows"] == 28
    assert report["rejected_reserve_family_rows"] == 0


def test_query_source_layout_binds_only_the_consumed_shard():
    rows, _report = query_candidate_rows()
    layout = query_source_layout(rows)
    assert len(layout) == 1
    assert layout[0]["global_row_start"] == int(rows[0])
    assert layout[0]["global_row_stop"] == int(rows[-1]) + 1
    assert layout[0]["embedding"]["sha256"]
    assert layout[0]["text"]["sha256"]


def test_polish_ood_query_panel_reproduces_round0108():
    rows = polish_query_rows()
    assert rows.shape == (500,)
    assert rows[0] == 24_263
    assert rows[-1] == 1_999_953
    assert POLISH_QUERY_ROWS_SHA256 == (
        "ae06ba5dd3e5ce3b1aafd18604b80c8d8575ea45a367f68871be6c80a99aa36b"
    )


def test_sparse_parquet_text_fetch_preserves_requested_order(tmp_path):
    path = tmp_path / "texts.parquet"
    pq.write_table(
        pa.table({"chunk_text": ["zero", "one", "two", "three", "four"]}),
        path,
        row_group_size=2,
    )
    assert _fetch_parquet_rows(
        str(path), np.asarray([1, 4], dtype=np.int64), expected_rows=5
    ) == ["one", "four"]


def test_polish_stored_row_uniqueness_guard_uses_complete_fp16_bytes():
    values = np.zeros((3, DIMENSION), dtype=np.float32)
    values[0, 0] = 1
    values[1, 1] = 1
    values[2, 2] = 1
    _require_unique_stored_rows(values, label="unique fixture")
    values[2] = values[1]
    with pytest.raises(Round0113Error, match="exact repeated rows"):
        _require_unique_stored_rows(values, label="duplicate fixture")


def test_retained_duplicate_audit_reports_complete_byte_families():
    values = np.zeros((4, 64), dtype=np.float16)
    values[0, 0] = values[2, 0] = 1
    values[1, 1] = 1
    values[3, 3] = 1
    report = _exact_duplicate_audit(
        values, mapping=np.asarray([10, 11, 12, 13], dtype=np.int64)
    )
    assert report["exact_nontrivial_family_count"] == 1
    assert report["rows_in_exact_nontrivial_families"] == 2
    assert report["example_global_families"] == [[10, 12]]
    assert report["passed_no_retained_exact_duplicates"] is False

    values[2, 2] = 1
    assert _exact_duplicate_audit(values)[
        "passed_no_retained_exact_duplicates"
    ] is True


def test_prompt_family_union_uses_one_shared_transitive_representative():
    extra, report = _union_prompt_exclusions(
        {
            "raw": [[10, 20], [40, 50]],
            "document": [[20, 30], [40, 60]],
            "text": [[10, 20, 30]],
        },
        np.asarray([10, 20, 30, 40, 50, 60, 70], dtype=np.int64),
    )
    np.testing.assert_array_equal(extra, [20, 30, 50, 60])
    assert report["union_families_global_rows"] == [
        [10, 20, 30],
        [40, 50, 60],
    ]
    assert report["embedding_family_text_relation"]["raw"] == {
        "exact_embedding_families": 2,
        "source_text_explained_families": 1,
        "cross_source_text_families": 1,
        "cross_source_text_families_global_rows": [[40, 50]],
    }


def test_text_family_census_verifies_complete_utf8_bytes(tmp_path, monkeypatch):
    path = tmp_path / "training-texts.parquet"
    pq.write_table(
        pa.table({"chunk_text": ["same", "different", "same", "other"]}),
        path,
        row_group_size=2,
    )
    layout = [
        {
            "global_row_start": 0,
            "global_row_stop": 4,
            "shard_row_start": 0,
            "shard_row_stop": 4,
            "shard_rows": 4,
            "text": {"canonical_path": str(path)},
        }
    ]
    import experiments.round0113_nodes as nodes

    monkeypatch.setattr(nodes, "BASELINE_RETAINED_ROWS", 4)
    families, report, hashes = _exact_text_families(
        layout, np.arange(4, dtype=np.int64)
    )
    assert families == [[0, 2]]
    assert report["exact_nontrivial_family_count"] == 1
    np.testing.assert_array_equal(
        hashes, _text_row_hashes(["same", "different", "same", "other"])
    )


def test_sorted_text_hash_membership_finds_exact_training_copies():
    reference = np.sort(
        _text_row_hashes(["alpha", "beta", "gamma"]), kind="stable"
    )
    observed = _sorted_hash_membership(
        reference, _text_row_hashes(["other", "beta", "alpha"])
    )
    np.testing.assert_array_equal(observed, [False, True, True])


def test_compact_mapping_closes_registered_population(monkeypatch):
    excluded = np.arange(5_366, dtype=np.int64)
    extra = np.asarray([6_000, 7_000], dtype=np.int64)
    monkeypatch.setattr(
        prompt_contract, "PROMPT_UNION_EXTRA_EXCLUDED_ROWS", len(extra)
    )
    monkeypatch.setattr(
        prompt_contract,
        "PROMPT_UNION_EXTRA_EXCLUSIONS_SHA256",
        ordered_array_sha256(extra),
    )
    monkeypatch.setattr(prompt_contract, "RETAINED_ROWS", 1_994_632)
    mapping = compact_mapping(excluded, extra)
    assert mapping.shape == (1_994_632,)
    assert mapping[0] == 5_366
    assert mapping[-1] == 1_999_999
    assert not np.isin(extra, mapping).any()


def test_data_identity_uses_panel_v2_ordered_shard_contract():
    assembly = {
        "outputs": {
            "raw": _signature("/data/raw-compact.f16", "a"),
            "document": _signature("/data/document-compact.f16", "b"),
        },
        "mapping": _signature("/data/compact-to-global.i64.npy", "c"),
        "substrate": _signature("/data/native8192-substrate-v2.json", "d"),
    }
    identity = _data_identity(assembly, arm="raw")
    assert set(identity) == {"kind", "shape", "dtype", "shards"}
    assert identity["kind"] == "ordered_shards"
    assert identity["shape"] == [RETAINED_ROWS, DIMENSION]
    assert identity["dtype"] == "<f2"
    assert identity["shards"] == [
        {
            "position": 0,
            "name": "raw-compact.f16",
            "bytes": 1,
            "sha256": "a" * 64,
        }
    ]


def test_fp16_endpoint_gather_preserves_requested_pairs():
    source = np.arange(4 * DIMENSION, dtype=np.float16).reshape(4, DIMENSION)
    view = HostFp16EndpointArray(
        source,
        arm="raw",
        source_signature={"sha256": "a" * 64},
        mapping_signature={"sha256": "b" * 64},
        buffer_rows=3,
        device="cpu",
    )
    left, right = view.gather_pairs(
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
    )
    np.testing.assert_array_equal(left.numpy(), source[[0, 2]].astype(np.float32))
    np.testing.assert_array_equal(
        right.numpy(), source[[1, 3]].astype(np.float32)
    )
    stamp = view.execution_stamp()
    assert stamp["endpoint_gather_calls"] == 1
    assert stamp["source_rows_gathered"] == 2
    assert stamp["destination_rows_gathered"] == 2


def test_arm_configs_share_recipe_but_bind_separate_graphs():
    raw, raw_digest = train_config(
        "raw",
        graph_signature=_signature("/data/raw-graph.npz", "a"),
        graph_manifest_signature=_signature("/data/raw-manifest.json", "b"),
        graph_edges=123,
        retained_rows=RETAINED_ROWS,
    )
    document, document_digest = train_config(
        "document",
        graph_signature=_signature("/data/document-graph.npz", "c"),
        graph_manifest_signature=_signature(
            "/data/document-manifest.json", "d"
        ),
        graph_edges=123,
        retained_rows=RETAINED_ROWS,
    )
    assert raw["paired_invariant"] == document["paired_invariant"]
    assert raw["model"] == document["model"]
    assert raw["optimizer"] == document["optimizer"]
    assert raw["graph"]["sha256"] != document["graph"]["sha256"]
    assert raw["graph"]["nprobe"] == document["graph"]["nprobe"] == 64
    assert raw["optimizer"]["seed"] == document["optimizer"]["seed"] == 42
    assert (
        raw["execution"]["expected_pipeline_stamp"][
            "negative_row_pairs_identical_across_arms"
        ]
        is True
    )
    assert raw_digest != document_digest


def test_prompt_sampler_keeps_negative_pairs_paired_across_arm_weights():
    class Dataset:
        device = "cpu"

        def __len__(self):
            return 8

        def execution_stamp(self):
            return {}

    sources = np.arange(8, dtype=np.int32)
    targets = np.roll(sources, -1)
    common = {
        "dataset": Dataset(),
        "sources": sources,
        "targets": targets,
        "n_nodes": 8,
        "batch_size": 20,
        "pos_ratio": 0.2,
        "random_state": 42,
        "graph_signatures": {},
    }
    raw = PromptWeightedJinaSampler(
        **common, weights=np.ones(8, dtype=np.float32), arm="raw"
    )
    document = PromptWeightedJinaSampler(
        **common,
        weights=np.linspace(0.1, 1.0, 8, dtype=np.float32),
        arm="document",
    )
    raw_left, raw_right = raw._rows()
    document_left, document_right = document._rows()
    np.testing.assert_array_equal(
        raw_left[raw.num_pos :], document_left[document.num_pos :]
    )
    np.testing.assert_array_equal(
        raw_right[raw.num_pos :], document_right[document.num_pos :]
    )


def test_registered_noninferiority_is_inclusive_and_excludes_projection_ffr():
    control = _score(1.0)
    at_margin = _score(NONINFERIORITY_RATIO)
    decision = paired_decision(control, at_margin)
    assert decision["passed"] is True
    assert set(decision["metric_gates"]) == set(DECISION_METRICS)
    assert "projection_ffr" not in decision["metric_gates"]
    assert decision["projection_ffr_role"] == "diagnostic-only"

    below = _score(NONINFERIORITY_RATIO - 1e-6)
    assert paired_decision(control, below)["passed"] is False
    assert paired_decision(control, _score(1.0, execution=False))["passed"] is False
