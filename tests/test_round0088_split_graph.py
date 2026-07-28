from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0034_pipeline import GRAPH_SCHEMA
from basemap.round0086_program import (
    FILTER_RECEIPT_SCHEMA,
    POLICY_GRID,
    QUALIFICATION_SCHEMA,
    seal as seal86,
)
from basemap.round0088_graph import (
    CORPUS_SPECS,
    ASSEMBLED_GRAPH_SCHEMA,
    PART_SCHEMA,
    ROUND_BY_CORPUS,
    R0081_SELECTED_RERANK_SECONDS_PER_QUERY,
    R0081_SELECTED_SEARCH_SECONDS_PER_QUERY,
    Round0088Error,
    projected_corpus_wall_seconds,
    seal,
    selected_benchmark_seconds_per_query,
    validate_filter_receipt,
    validate_part_receipt,
    validate_qualification,
)
from basemap.round0093_policy import (
    POLICY_GRID as R0093_POLICY_GRID,
    QUALIFICATION_SCHEMA as R0093_QUALIFICATION_SCHEMA,
    seal as seal93,
)
from experiments import round0088_nodes as nodes
from experiments import prepare_round0088_0091_queue as preparer


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
        selected_search_seconds_per_query=R0081_SELECTED_SEARCH_SECONDS_PER_QUERY,
        selected_rerank_seconds_per_query=R0081_SELECTED_RERANK_SECONDS_PER_QUERY,
    )
    assert projected == pytest.approx(observed * 1.25 + 300.0)
    assert projected < 7.5 * 3600


def test_projection_scales_with_measured_search_cost() -> None:
    baseline = projected_corpus_wall_seconds(
        "fineweb",
        selected_search_seconds_per_query=R0081_SELECTED_SEARCH_SECONDS_PER_QUERY,
        selected_rerank_seconds_per_query=R0081_SELECTED_RERANK_SECONDS_PER_QUERY,
    )
    slower = projected_corpus_wall_seconds(
        "fineweb",
        selected_search_seconds_per_query=2 * R0081_SELECTED_SEARCH_SECONDS_PER_QUERY,
        selected_rerank_seconds_per_query=R0081_SELECTED_RERANK_SECONDS_PER_QUERY,
    )
    assert slower == pytest.approx(2 * (baseline - 300.0) + 300.0)
    with pytest.raises(Round0088Error):
        projected_corpus_wall_seconds(
            "fineweb", selected_search_seconds_per_query=0.0
        )


def test_selected_benchmark_rates_keep_search_and_rerank_separate() -> None:
    selected = {
        "benchmark": {
            "rows": 10_000,
            "median_search_seconds": 2.5,
            "median_rerank_seconds": 1.5,
            "median_total_seconds": 9.0,
        }
    }
    assert selected_benchmark_seconds_per_query(selected) == (
        0.00025,
        0.00015,
    )


def test_qualification_and_filter_receipts_are_content_bound(
    tmp_path: Path,
) -> None:
    substrate = {
        "canonical_path": "/tmp/substrate.json",
        "bytes": 10,
        "sha256": "a" * 64,
        "kind": "file",
    }
    filtered = {
        "canonical_path": "/tmp/index.ivfpq",
        "bytes": 20,
        "sha256": "b" * 64,
        "kind": "file",
    }
    filter_value = seal86({
        "schema": FILTER_RECEIPT_SCHEMA,
        "round_id": "0086",
        "substrate": substrate,
        "filtered_index": filtered,
    })
    filter_path = tmp_path / "filter.json"
    filter_sha = _write_json(filter_path, filter_value)
    loaded_filter = validate_filter_receipt(
        str(filter_path),
        expected_sha256=filter_sha,
        substrate_signature=substrate,
        filtered_index_signature=filtered,
    )
    assert loaded_filter["receipt"]["round_id"] == "0086"

    cell = {
        "nprobe": 128,
        "shortlist_width": 256,
        "passes_mean_floor": True,
        "mean_recall_at_15_unambiguous": 0.91,
        "benchmark": {
            "rows": 10_000,
            "median_search_seconds": 2.0,
            "median_rerank_seconds": 1.0,
            "median_wall_seconds_per_query": 0.0003,
        },
    }
    cells = {
        f"nprobe-{nprobe}-width-{width}": (
            cell if (nprobe, width) == POLICY_GRID[0] else None
        )
        for nprobe, width in POLICY_GRID
    }
    qualification = seal86({
        "schema": QUALIFICATION_SCHEMA,
        "round_id": "0086",
        "substrate": substrate,
        "filtered_index": filtered,
        "validity_passed": True,
        "training_performed": False,
        "quality": {"floor": 0.90},
        "checks": {"passing_policy_selected": True},
        "cells": cells,
        "selected": cell,
    })
    qualification_path = tmp_path / "qualification.json"
    qualification_sha = _write_json(qualification_path, qualification)
    loaded = validate_qualification(
        str(qualification_path),
        expected_sha256=qualification_sha,
        substrate_signature=substrate,
        filtered_index_signature=filtered,
    )
    assert loaded["selected"] == cell
    assert loaded["policy_round_id"] == "0086"

    r0093_cells = {
        f"nprobe-{nprobe}-width-{width}": (
            {
                **cell,
                "nprobe": nprobe,
                "shortlist_width": width,
                "mean_recall_at_15_unambiguous": 0.85,
            }
            if (nprobe, width) == R0093_POLICY_GRID[0]
            else None
        )
        for nprobe, width in R0093_POLICY_GRID
    }
    r0093_cell = r0093_cells[
        f"nprobe-{R0093_POLICY_GRID[0][0]}-"
        f"width-{R0093_POLICY_GRID[0][1]}"
    ]
    r0093 = seal93({
        "schema": R0093_QUALIFICATION_SCHEMA,
        "round_id": "0093",
        "substrate": substrate,
        "filtered_index": filtered,
        "validity_passed": True,
        "training_performed": False,
        "quality": {"floor": 0.84},
        "checks": {"passing_policy_selected": True},
        "cells": r0093_cells,
        "selected": r0093_cell,
    })
    r0093_path = tmp_path / "r0093-qualification.json"
    r0093_sha = _write_json(r0093_path, r0093)
    loaded_r0093 = validate_qualification(
        str(r0093_path),
        expected_sha256=r0093_sha,
        substrate_signature=substrate,
        filtered_index_signature=filtered,
    )
    assert loaded_r0093["selected"] == r0093_cell
    assert loaded_r0093["policy_round_id"] == "0093"
    assert loaded_r0093["mean_recall_floor"] == 0.84

    qualification["selected"] = {**cell, "nprobe": 192}
    bad_path = tmp_path / "bad-qualification.json"
    bad_sha = _write_json(bad_path, seal86({
        key: value
        for key, value in qualification.items()
        if key != "identity_sha256"
    }))
    with pytest.raises(Round0088Error):
        validate_qualification(
            str(bad_path),
            expected_sha256=bad_sha,
            substrate_signature=substrate,
            filtered_index_signature=filtered,
        )


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
            "mean_recall_at_15_unambiguous": 0.91,
            "floor": 0.90,
            "qualification_sample_rows": 4_096,
            "qualification_sample_seed": 86,
        },
    })
    path = tmp_path / f"{corpus}.json"
    sha = _write_json(path, value)
    assert validate_part_receipt(
        str(path), expected_sha256=sha
    )["receipt"]["corpus"] == corpus

    lower = {
        key: item
        for key, item in value.items()
        if key != "identity_sha256"
    }
    lower["quality"] = {
        **lower["quality"],
        "mean_recall_at_15_unambiguous": 0.85,
        "floor": 0.84,
    }
    lower_path = tmp_path / f"{corpus}-lower.json"
    lower_sha = _write_json(lower_path, seal(lower))
    assert validate_part_receipt(
        str(lower_path), expected_sha256=lower_sha
    )["receipt"]["quality"]["floor"] == 0.84

    value["valid_edges"] += 1
    bad = tmp_path / f"{corpus}-bad.json"
    bad_sha = _write_json(bad, seal({
        key: item for key, item in value.items()
        if key != "identity_sha256"
    }))
    with pytest.raises(Round0088Error):
        validate_part_receipt(str(bad), expected_sha256=bad_sha)


def test_part_queue_binds_reviewed_r0093_decision() -> None:
    source = inspect.getsource(preparer.prepare_part_queue)
    assert 'manifest["required_reviews"] = ["0086", "0093"]' in source
    assert "minilm-graph-recall-operational-floor-0p84-v1" in source
    assert "search-qualified-low-recall-v1" in source
    assert '"policy_decision": staged["policy_decision"]["signature"]' in source
    node_source = inspect.getsource(nodes.run_build_part)
    assert "load_r0093_decision" in node_source
    assert '"floor": qualification["mean_recall_floor"]' in node_source


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
