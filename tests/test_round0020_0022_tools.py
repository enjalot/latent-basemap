from __future__ import annotations

import json

import numpy as np

from basemap.duplicate_census import save_cap_npz
from basemap.duplicate_multiplicity import load_duplicate_cap
from experiments import run_round0014_node as node
from experiments.universality_panel import _ffr_from_neighbors


def test_round0020_global_cap_uses_duplicate_cap_schema(tmp_path):
    representatives = np.asarray([10, 20], dtype=np.int64)
    family_counts = np.asarray([3, 2], dtype=np.int64)
    member_rows = np.asarray([10, 11, 12, 20, 21], dtype=np.int64)
    offsets = np.asarray([0, 3, 5], dtype=np.int64)
    path = tmp_path / "global-cap-v1.npz"
    save_cap_npz(
        str(path),
        representative_rows=representatives,
        family_counts=family_counts,
        member_rows=member_rows,
        family_offsets=offsets,
        census_signature={"sha256": "0" * 64},
    )
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    loaded = load_duplicate_cap(
        str(path),
        expected_sha256=digest,
        row_count=30_000_000,
        fixed_edges_per_source=15,
    )
    assert loaded["representative_rows"].tolist() == [10, 20]
    assert loaded["excluded_rows"].tolist() == [11, 12, 21]
    assert loaded["metadata"]["retained_row_count"] == 29_999_997


def test_round0022_ffr_math_counts_true_neighbors_inside_hit_set():
    true_top10 = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]])
    hit_neighbors = np.asarray([[2, 4, 6, 8, 10, 12], [1, 2, 3, 4, 5, 6]])
    ffr, per_query = _ffr_from_neighbors(true_top10, hit_neighbors)
    assert per_query.tolist() == [0.5, 0.0]
    assert ffr == 0.25


def test_round0020_0022_configure_existing_runner_node_names():
    node.configure_round0020()
    assert node.ROUND_ID == "0020"
    assert node._run_canary.__name__ == "_run_round0020_duplicate_census"
    node.configure_round0022()
    assert node.ROUND_ID == "0022"
    assert node._run_canary.__name__ == "_run_round0022_canary"
    assert node._run_panel.__name__ == "_run_round0022_panel"
    assert node._run_semantic_renders.__name__ == "_run_round0022_renders"

