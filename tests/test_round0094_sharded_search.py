from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0094_sharded_search import (
    MAX_MEDIAN_SECONDS_PER_QUERY,
    MEAN_RECALL_FLOOR,
    POLICY_GRID,
    RETAINED_ROWS,
    ROUND_ID,
    SHARD_SPECS,
    SPLIT_SCHEMA,
    Round0094Error,
    cell_key,
    load_split_receipt,
    seal,
    select_cell,
)
from experiments import round0094_nodes as nodes
from experiments.round0094_nodes import _search_and_rerank


def test_registered_shards_close_the_reviewed_universe() -> None:
    assert ROUND_ID == "0094"
    assert MEAN_RECALL_FLOOR == 0.84
    assert MAX_MEDIAN_SECONDS_PER_QUERY == 0.001
    assert list(SHARD_SPECS) == ["fineweb", "redpajama", "pile"]
    assert sum(
        value["retained_rows"] for value in SHARD_SPECS.values()
    ) == RETAINED_ROWS
    assert [
        (value["start"], value["stop"])
        for value in SHARD_SPECS.values()
    ] == [
        (0, 50_000_000),
        (50_000_000, 100_000_000),
        (100_000_000, 150_000_000),
    ]


def test_policy_grid_is_fixed_and_bounded() -> None:
    assert POLICY_GRID == (
        (32, 64),
        (40, 64),
        (64, 64),
        (96, 64),
        (32, 128),
        (40, 128),
        (64, 128),
        (96, 128),
        (32, 256),
        (40, 256),
        (64, 256),
        (96, 256),
    )


def test_selector_requires_quality_and_speed() -> None:
    cells = {}
    for index, (nprobe, width) in enumerate(POLICY_GRID):
        cells[cell_key(nprobe, width)] = {
            "nprobe_per_shard": nprobe,
            "width_per_shard": width,
            "total_shortlist_width": width * 3,
            "passes_mean_floor": index != 0,
            "passes_performance_ceiling": index not in {0, 1},
            "benchmark": {
                "median_wall_seconds_per_query": 0.0009 + index / 1e6,
            },
        }
    selected = select_cell({"cells": cells})
    assert selected is cells[cell_key(64, 64)]
    cells[cell_key(64, 64)]["passes_performance_ceiling"] = False
    assert select_cell({"cells": cells}) is cells[cell_key(96, 64)]


def test_split_receipt_binds_all_three_index_shards(
    tmp_path: Path,
) -> None:
    source = {
        "canonical_path": "/data/source.ivfpq",
        "bytes": 123,
        "sha256": "a" * 64,
        "kind": "file",
    }
    shards = {
        name: {
            **spec,
            "ntotal": spec["retained_rows"],
            "index": {
                "canonical_path": f"/data/{name}.ivfpq",
                "bytes": 10,
                "sha256": str(index + 1) * 64,
                "kind": "file",
            },
        }
        for index, (name, spec) in enumerate(SHARD_SPECS.items())
    }
    receipt = seal({
        "schema": SPLIT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": "b" * 40,
        "source_index": source,
        "shards": shards,
        "global_ids_preserved": True,
        "disjoint_complete_id_ranges": True,
        "training_performed": False,
    })
    path = tmp_path / "split.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    loaded = load_split_receipt(
        str(path),
        expected_source=source,
        expected_release_sha="b" * 40,
    )
    assert loaded["receipt"]["shards"]["fineweb"]["ntotal"] == 48_529_276
    bad = dict(receipt)
    bad["shards"] = dict(shards)
    bad["shards"]["pile"] = dict(shards["pile"], ntotal=1)
    body = {key: value for key, value in bad.items() if key != "identity_sha256"}
    bad["identity_sha256"] = sha256_bytes(canonical_json(body))
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(Round0094Error):
        load_split_receipt(
            str(path),
            expected_source=source,
            expected_release_sha="b" * 40,
        )


class _FakeIndex:
    def __init__(self, raw: np.ndarray) -> None:
        self.raw = raw
        self.index = SimpleNamespace(nprobe=0)

    def search(
        self,
        queries: np.ndarray,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert width == self.raw.shape[1]
        assert len(queries) == len(self.raw)
        return np.zeros_like(self.raw, dtype=np.float32), self.raw.copy()


def test_sharded_search_preserves_each_shard_quota_before_global_rerank() -> None:
    dimension = 384
    encoded = np.ones((100, dimension), dtype=np.int8)
    encoded[:, 0] = np.arange(1, 101, dtype=np.int8)
    scales = np.ones(100, dtype="<f2")
    queries = encoded[[0, 1]].astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    sources = np.asarray([0, 1], dtype=np.int64)
    indices = [
        _FakeIndex(np.asarray([
            [0, *range(2, 17)],
            [1, *range(2, 17)],
        ], dtype=np.int64)),
        _FakeIndex(np.asarray([
            list(range(20, 36)),
            list(range(20, 36)),
        ], dtype=np.int64)),
        _FakeIndex(np.asarray([
            list(range(40, 56)),
            list(range(40, 56)),
        ], dtype=np.int64)),
    ]
    selected, receipt = _search_and_rerank(
        indices=indices,
        nprobe=40,
        width_per_shard=15,
        queries=queries,
        sources=sources,
        encoded=encoded,
        scales=scales,
    )
    assert selected.shape == (2, 15)
    assert receipt["total_shortlist_width"] == 45
    assert receipt["nprobe_per_shard"] == 40
    assert all(index.index.nprobe == 40 for index in indices)
    assert receipt["self_returned"] == 2


def test_peak_rss_uses_python_resource_module(monkeypatch) -> None:
    class Usage:
        ru_maxrss = 3 * 1024**2

    monkeypatch.setattr(
        nodes.resource,
        "getrusage",
        lambda who: Usage() if who == nodes.resource.RUSAGE_SELF else None,
    )
    assert nodes._peak_rss_gib() == 3.0
