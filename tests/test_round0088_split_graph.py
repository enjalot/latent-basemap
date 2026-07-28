from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0034_pipeline import GRAPH_SCHEMA
from basemap.round0096_larger_nlist import (
    DECISION_SCHEMA,
    INDEX_SCHEMA,
    POLICY_GRID,
    QUALITY_SAMPLE_SHA256,
    QUALIFICATION_SCHEMA,
)
from basemap.round0088_graph import (
    CORPUS_SPECS,
    ASSEMBLED_GRAPH_SCHEMA,
    PART_SCHEMA,
    ROUND_BY_CORPUS,
    R0081_SELECTED_TOTAL_SECONDS_PER_QUERY,
    Round0088Error,
    projected_corpus_wall_seconds,
    seal,
    selected_benchmark_seconds_per_query,
    validate_decision,
    validate_index_receipt,
    validate_part_receipt,
    validate_qualification,
)
from experiments import round0088_nodes as nodes
from experiments import prepare_round0088_0091_queue as prepare


def _write_json(path: Path, value: dict) -> str:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return expected_input_signature(str(path))["sha256"]


def test_corpus_partition_closes_exactly() -> None:
    assert ASSEMBLED_GRAPH_SCHEMA == GRAPH_SCHEMA
    assert [value["start"] for value in CORPUS_SPECS.values()] == [
        0, 50_000_000, 100_000_000
    ]
    assert [value["stop"] for value in CORPUS_SPECS.values()] == [
        50_000_000, 100_000_000, 150_000_000
    ]
    assert sum(value["excluded_rows"] for value in CORPUS_SPECS.values()) == 2_778_243
    assert sum(value["retained_rows"] for value in CORPUS_SPECS.values()) == 147_221_757


@pytest.mark.parametrize("corpus", list(CORPUS_SPECS))
def test_projection_replays_r0078_rate_at_unit_ratio(corpus: str) -> None:
    spec = CORPUS_SPECS[corpus]
    observed = (
        spec["retained_rows"]
        * spec["r0078_wall_us_per_retained_source"]
        / 1_000_000
    )
    projected = projected_corpus_wall_seconds(
        corpus,
        selected_total_seconds_per_query=R0081_SELECTED_TOTAL_SECONDS_PER_QUERY,
    )
    assert projected == pytest.approx(observed * 1.25 + 300.0)
    assert projected < 7.5 * 3600


def test_projection_scales_with_measured_search_cost() -> None:
    baseline = projected_corpus_wall_seconds(
        "fineweb",
        selected_total_seconds_per_query=R0081_SELECTED_TOTAL_SECONDS_PER_QUERY,
    )
    slower = projected_corpus_wall_seconds(
        "fineweb",
        selected_total_seconds_per_query=2
        * R0081_SELECTED_TOTAL_SECONDS_PER_QUERY,
    )
    assert slower == pytest.approx(2 * (baseline - 300.0) + 300.0)
    with pytest.raises(Round0088Error):
        projected_corpus_wall_seconds(
            "fineweb", selected_total_seconds_per_query=0.0
        )


def test_selected_benchmark_rates_keep_search_and_rerank_separate() -> None:
    selected = {
        "benchmark": {
            "queries": 8_192,
            "median_wall_seconds_per_query": 0.00025,
        }
    }
    assert selected_benchmark_seconds_per_query(selected) == 0.00025


def test_r0096_index_qualification_and_decision_are_content_bound(
    tmp_path: Path,
) -> None:
    substrate = {
        "canonical_path": "/tmp/substrate.json",
        "bytes": 10,
        "sha256": "a" * 64,
        "kind": "file",
    }
    index = {
        "canonical_path": "/tmp/index.ivfpq",
        "bytes": 20,
        "sha256": "b" * 64,
        "kind": "file",
    }
    index_receipt_value = seal({
        "schema": INDEX_SCHEMA,
        "round_id": "0096",
        "substrate": substrate,
        "index": index,
        "geometry": {
            "class": "IndexIVFPQ",
            "dimension": 384,
            "nlist": 32_768,
            "ntotal": 147_221_757,
            "code_size": 48,
            "pq_m": 48,
            "pq_bits": 8,
            "metric_type": 0,
        },
        "id_validation": {
            "global_ids_unique": True,
            "excluded_rows_absent": True,
            "seen_retained_rows": 147_221_757,
        },
        "training_performed": False,
        "optimizer_updates": 0,
    })
    index_receipt_path = tmp_path / "index-receipt.json"
    index_receipt_sha = _write_json(
        index_receipt_path, index_receipt_value,
    )
    loaded_index = validate_index_receipt(
        str(index_receipt_path),
        expected_sha256=index_receipt_sha,
        substrate_signature=substrate,
        index_signature=index,
    )

    def cell(nprobe: int, width: int) -> dict:
        return {
            "nprobe": nprobe,
            "shortlist_width": width,
            "passes_global_floor": True,
            "passes_every_corpus_floor": True,
            "mean_recall_at_15_unambiguous": 0.91,
            "by_corpus": {
                corpus: {
                    "mean_recall_at_15_unambiguous": 0.90,
                    "passes_floor": True,
                    "unambiguous_rows": 1_000,
                }
                for corpus in CORPUS_SPECS
            },
            "benchmark": {
                "queries": 8_192,
                "median_wall_seconds_per_query": (
                    0.0003
                    if (nprobe, width) == (128, 512)
                    else 0.0004 + nprobe / 1e9 + width / 1e10
                ),
            },
        }

    cells = {
        f"nprobe-{nprobe}-width-{width}": cell(nprobe, width)
        for nprobe, width in POLICY_GRID
    }
    selected = cells["nprobe-128-width-512"]
    qualification = seal({
        "schema": QUALIFICATION_SCHEMA,
        "round_id": "0096",
        "substrate": substrate,
        "index": index,
        "index_receipt": loaded_index["signature"],
        "cells": cells,
        "selected": selected,
        "quality": {
            "sample_rows": 4_096,
            "sample_seed": 86,
            "sample_sha256": QUALITY_SAMPLE_SHA256,
            "global_mean_floor": 0.90,
            "per_corpus_mean_floor": 0.84,
        },
        "checks": {"all_registered_cells_present": True},
        "failed_checks": [],
        "geometry": {"nlist": 32_768, "ntotal": 147_221_757},
        "validity_passed": True,
        "training_performed": False,
        "scale_decision_made": False,
        "optimizer_updates": 0,
    })
    qualification_path = tmp_path / "qualification.json"
    qualification_sha = _write_json(qualification_path, qualification)
    loaded = validate_qualification(
        str(qualification_path),
        expected_sha256=qualification_sha,
        substrate_signature=substrate,
        index_signature=index,
        index_receipt_signature=loaded_index["signature"],
    )
    assert loaded["selected"] == selected
    decision = seal({
        "schema": DECISION_SCHEMA,
        "round_id": "0096",
        "qualification": loaded["signature"],
        "selected": selected,
        "outcome": "qualified",
        "validity_passed": True,
        "graph_build_released": True,
        "training_performed": False,
        "scale_decision_made": False,
        "optimizer_updates": 0,
    })
    decision_path = tmp_path / "decision.json"
    decision_sha = _write_json(decision_path, decision)
    assert validate_decision(
        str(decision_path),
        expected_sha256=decision_sha,
        qualification_signature=loaded["signature"],
        selected=selected,
    )["receipt"]["graph_build_released"] is True


@pytest.mark.parametrize("corpus", list(CORPUS_SPECS))
def test_part_receipt_requires_exact_registered_counts(
    tmp_path: Path,
    corpus: str,
) -> None:
    spec = CORPUS_SPECS[corpus]
    value = seal({
        "schema": PART_SCHEMA,
        "round_id": ROUND_BY_CORPUS[corpus],
        "corpus": corpus,
        "start": spec["start"],
        "stop": spec["stop"],
        "retained_sources": spec["retained_rows"],
        "excluded_sources": spec["excluded_rows"],
        "valid_edges": spec["retained_rows"] * 15,
        "quality": {
            "selected_global_mean_recall": 0.91,
            "selected_by_corpus": {
                corpus: {"mean_recall_at_15_unambiguous": 0.90},
            },
            "global_mean_floor": 0.90,
            "per_corpus_mean_floor": 0.84,
            "qualification_sample_rows": 4_096,
            "qualification_sample_seed": 86,
            "qualification_sample_sha256": QUALITY_SAMPLE_SHA256,
        },
    })
    path = tmp_path / f"{corpus}.json"
    sha = _write_json(path, value)
    assert validate_part_receipt(
        str(path), expected_sha256=sha
    )["receipt"]["corpus"] == corpus

    value["valid_edges"] += 1
    bad = tmp_path / f"{corpus}-bad.json"
    bad_sha = _write_json(bad, seal({
        key: item for key, item in value.items()
        if key != "identity_sha256"
    }))
    with pytest.raises(Round0088Error):
        validate_part_receipt(str(bad), expected_sha256=bad_sha)


def _write_shard(
    root: Path,
    *,
    round_id: str,
    start: int,
    stop: int,
    values: np.ndarray,
    nprobe: int = 128,
    search_width: int = 256,
) -> None:
    root.mkdir(parents=True)
    target, receipt = nodes._shard_paths(str(root), start // nodes.SHARD_ROWS)
    np.save(target, np.asarray(values, dtype="<i4"), allow_pickle=False)
    body = {
        "schema": "round0049-exact-rerank-graph-shard-v2",
        "round_id": round_id,
        "start": start,
        "stop": stop,
        "nprobe": nprobe,
        "search_width": search_width,
        "index_search_width": search_width + 1,
        "targets": expected_input_signature(target),
    }
    Path(receipt).write_text(
        json.dumps(seal(body), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_small_three_part_assembly_preserves_global_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiny_specs = {
        "fineweb": {"start": 0, "stop": 2},
        "redpajama": {"start": 2, "stop": 4},
        "pile": {"start": 4, "stop": 6},
    }
    monkeypatch.setattr(nodes, "ROW_COUNT", 6)
    monkeypatch.setattr(nodes, "RETAINED_ROWS", 5)
    monkeypatch.setattr(nodes, "CORPUS_SPECS", tiny_specs)
    roots = {}
    expected = []
    for index, (corpus, spec) in enumerate(tiny_specs.items()):
        root = tmp_path / corpus
        roots[corpus] = str(root.parent / corpus)
        values = np.full((2, 15), index + 10, dtype="<i4")
        expected.append(values)
        _write_shard(
            root / "shards",
            round_id=ROUND_BY_CORPUS[corpus],
            start=spec["start"],
            stop=spec["stop"],
            values=values,
        )
    output = tmp_path / "assembled"
    output.mkdir()
    targets, degrees = nodes._assemble_part_roots(
        output=str(output),
        roots=roots,
        excluded=np.asarray([3], dtype=np.int64),
        nprobe=128,
        search_width=256,
    )
    target_values = np.memmap(
        targets["canonical_path"], dtype="<i4", mode="r", shape=(6, 15)
    )
    degree_values = np.memmap(
        degrees["canonical_path"], dtype="u1", mode="r", shape=(6,)
    )
    assert np.array_equal(target_values, np.concatenate(expected))
    assert degree_values.tolist() == [15, 15, 15, 0, 15, 15]


def test_r0100_queue_declares_every_shard_target_as_an_input() -> None:
    source = inspect.getsource(prepare.prepare_assembly_queue)
    assert "shard_targets" in source
    assert "observed_target != target" in source
    assert "*shard_targets" in source
    assert '"required_reviews"] = ["0097", "0098", "0099"]' in source
