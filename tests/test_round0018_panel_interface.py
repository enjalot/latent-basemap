"""Tiny CPU fixture for the exact production coordinate/panel interface."""
from __future__ import annotations

import numpy as np

from basemap.panel_v2 import PanelV2Config, QueryTruthCache, score_panel
from basemap.artifact_identity import ordered_array_sha256
from experiments import run_round0014_node as node
from experiments.score_complete_panel import score_query_bundle


def _streamed_fixture(tmp_path, values: np.ndarray):
    view = node.StreamedCoordinateArray.__new__(node.StreamedCoordinateArray)
    view.root = str(tmp_path)
    view.shape = values.shape
    view.dtype = np.dtype("<f4")
    view._members = []
    cursor = 0
    for index, chunk in enumerate(np.array_split(values, 3)):
        path = tmp_path / f"coords-{index}.npy"
        np.save(path, np.asarray(chunk, dtype="<f4"), allow_pickle=False)
        stop = cursor + len(chunk)
        view._members.append(
            {
                "global_row_start": cursor,
                "global_row_stop": stop,
                "path": str(path),
            }
        )
        cursor = stop
    view.shard_paths = [member["path"] for member in view._members]
    return view


def test_streamed_coordinates_cover_nd_reduce_and_complete_panel(tmp_path):
    rng = np.random.RandomState(18)
    values = rng.randn(180, 2).astype("<f4")
    coordinates = _streamed_fixture(tmp_path, values)

    nd_rows = np.array([[0, 61, 179], [90, 2, 125]], dtype=np.int64)
    np.testing.assert_array_equal(coordinates[nd_rows], values[nd_rows])
    np.testing.assert_array_equal(coordinates[nd_rows, 1], values[nd_rows, 1])
    np.testing.assert_array_equal(coordinates.min(axis=0), values.min(axis=0))
    np.testing.assert_array_equal(coordinates.max(axis=0), values.max(axis=0))
    assert coordinates.min() == values.min()
    assert coordinates.max() == values.max()

    X = rng.randn(len(values), 8).astype("<f4")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    config = PanelV2Config(
        frac=0.1,
        k_hit=3,
        k_density=3,
        n_anchors=18,
        corpus_chunk=64,
        overselect=4,
        block_elems=100_000,
        rerank_byte_cap=8_000_000,
        peak_byte_cap=16_000_000,
    )
    panel = score_panel(X, coordinates, config=config, provenance={"fixture": True})
    assert panel["n"] == len(values)
    assert panel["guards"]["coords_finite"] is True
    assert panel["guards"]["coords_collapsed"] is False

    Xq = rng.randn(6, X.shape[1]).astype("<f4")
    Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
    Zq = rng.randn(len(Xq), 2).astype("<f4")
    cache = QueryTruthCache(cache_dir=None, enabled=False)
    cache.get_or_build(
        Xq,
        X,
        cfg=config,
        corpus_identity={"sha256": ordered_array_sha256(X)},
        query_identity={"sha256": ordered_array_sha256(Xq)},
        k=15,
    )
    projection = score_query_bundle(
        X=X,
        Z=coordinates,
        Xq=Xq,
        Zq=Zq,
        cfg=config,
        truth_cache=cache,
        label="streamed-production-fixture",
        random_seed=18,
    )
    assert set(projection) == {
        "proj_ffr",
        "proj_recall@k",
        "proj_knn_regressor_ffr",
        "proj_random_floor_ffr",
        "proj_beats_knn",
        "proj_margin_over_knn",
    }


def test_registered_selector_requires_real_guard_fields():
    panel = {
        "ffr": 0.4645,
        "density": 0.7856,
        "purity": {"k256": 1.1152, "k1024": 0.9317},
        "recall@k": 0.00365,
        "guards": {
            "coords_finite": True,
            "coords_collapsed": False,
            "emb_finite": True,
            "emb_zero_rows": 0,
        },
    }
    projection = {"proj_ffr": 0.4288}
    ratio, checks = node._registered_panel_decision(
        panel, projection, 0.00456, {"passed": True}, 0.0004
    )
    assert ratio == 1072.0
    assert all(checks.values())

    for field, invalid in (
        ("coords_finite", None),
        ("coords_collapsed", True),
        ("emb_finite", None),
        ("emb_zero_rows", 1),
    ):
        mutated = {**panel, "guards": dict(panel["guards"])}
        mutated["guards"][field] = invalid
        _, failed = node._registered_panel_decision(
            mutated, projection, 0.00456, {"passed": True}, 0.0004
        )
        assert failed["numerical_guards"] is False
