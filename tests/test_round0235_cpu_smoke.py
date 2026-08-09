"""R0235 CUDA-hidden CPU smoke — reach the code paths a GPU queue would.

The protocol's preparation validation: run the real seal/publish/reload path and
the real downstream interfaces on toy inputs, so a late NameError, an accounting
shape drift or a serialization failure surfaces before hours of GPU time.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.round0220_cuvs_qualification import (
    graph_validity,
    strict_containment_rows,
    tie_aware_rows,
)
from basemap.round0235_rung2 import (
    GRAPH_K,
    IMBALANCE_PROBE_CLUSTERS,
    Round0235Error,
    fit_device_law,
    imbalance_drift,
    io_projection,
    rung_derivation,
)
import basemap.round0235_build as r0235_build
from basemap import round0113_prompt_contrast as prompt_contract
import experiments.round0235_nodes as nodes
import experiments.prepare_round0235_queue as prepare


R0233_SUBSTRATE_MANIFEST = prepare.R0233_SUBSTRATE_MANIFEST
R0233_LADDER = prepare.R0233_LADDER
R0229_SWEEP = prepare.R0229_SWEEP
R0229_ARM = prepare.R0229_ARM
R0229_REACHABILITY = prepare.R0229_REACHABILITY


def test_node_dispatch_refuses_an_unregistered_action():
    with pytest.raises(Round0235Error, match="does not authorize"):
        nodes.run_job({"manifest": {"round_id": "0235"}}, {"action": "train"})


def test_build_wrapper_requires_a_capacity_and_rebinds_exactly_one_constant(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"setting_id": "x"}), encoding="utf-8")
    with pytest.raises(Round0235Error, match="cluster_capacity_rows"):
        r0235_build.main(["--config", str(config), "--out", str(tmp_path)])
    import basemap.round0233_build as r0233_build

    before = int(r0233_build.CLUSTER_CAPACITY_ROWS)
    assert before == r0235_build.EXPECTED_R0233_CAPACITY_ROWS
    try:
        assert r0235_build.rebind_capacity(9_000_000) == 9_000_000
        assert int(r0233_build.CLUSTER_CAPACITY_ROWS) == 9_000_000
        with pytest.raises(Round0235Error):
            r0235_build.rebind_capacity(0)
    finally:
        r0233_build.CLUSTER_CAPACITY_ROWS = before


@pytest.mark.skipif(
    not os.path.exists(R0233_SUBSTRATE_MANIFEST), reason="R0233 substrate absent"
)
def test_parent_manifest_verifies_and_carries_what_nesting_needs():
    sealed = prompt_contract.read_sealed(
        R0233_SUBSTRATE_MANIFEST, label="R0233 substrate manifest"
    )
    assert sealed["round_id"] == "0233" and sealed["rows"] == 6_250_000
    assert len(sealed["ordered_substrate_sha256"]) == 64
    for key in ("substrate", "provenance", "reserve_substrate",
                "reserve_provenance", "reserve_query_rows"):
        assert expected_input_signature(
            sealed[key]["canonical_path"]
        ) == dict(sealed[key])


@pytest.mark.skipif(
    not (os.path.exists(R0229_SWEEP) and os.path.exists(R0229_ARM)
         and os.path.exists(R0233_LADDER)),
    reason="inherited sealed artifacts absent",
)
def test_inherited_law_points_are_homogeneous_and_fit():
    job = {
        "r0229_sweep": expected_input_signature(R0229_SWEEP),
        "r0229_arm": expected_input_signature(R0229_ARM),
        "r0233_ladder": expected_input_signature(R0233_LADDER),
    }
    points = nodes._inherited_law_points(job)
    law = fit_device_law(points, label="inherited")
    assert law["n_points"] == 6
    settings = {
        (p["graph_degree"], p["intermediate_graph_degree"], p["max_iterations"])
        for p in law["points"]
    }
    assert settings == {(64, 256, 40)}
    assert law["slope_bytes_per_max_cluster_row"] == pytest.approx(1678.2678, rel=1e-4)
    # The q6 cell R0233 mislabelled is present in the sweep and refused here.
    refused = {p["cell"] for p in law["points_refused"]}
    assert "q6" in refused


@pytest.mark.skipif(
    not (os.path.exists(R0229_REACHABILITY) and os.path.exists(R0233_LADDER)),
    reason="inherited sealed artifacts absent",
)
def test_inherited_imbalance_reads_2m_and_6250k_and_shows_the_c32_gap():
    job = {
        "r0229_reachability": expected_input_signature(R0229_REACHABILITY),
        "r0233_ladder": expected_input_signature(R0233_LADDER),
    }
    series = nodes._inherited_imbalance(job)
    at_2m = series[2_000_000]
    at_6250k = series[6_250_000]
    assert set(at_2m) == {16, 64, 200}
    assert 32 not in at_2m
    assert set(at_6250k) == {16, 32, 64, 200}
    drift = imbalance_drift({
        key: value for key, value in series.items() if isinstance(key, int)
    })
    assert drift["by_clusters"]["64"]["drift_relative"] > 0.03
    assert abs(drift["by_clusters"]["200"]["drift_relative"]) < 0.01


def test_qualification_and_fuzzy_law_close_on_a_tiny_memmapped_substrate(tmp_path):
    rows, dim, k = 512, 8, GRAPH_K
    rng = np.random.default_rng(235)
    data = rng.standard_normal((rows, dim)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    path = tmp_path / "substrate.f32.npy"
    np.save(path, data)
    host = np.load(path, mmap_mode="r")
    assert isinstance(host, np.memmap)

    sims = data @ data.T
    np.fill_diagonal(sims, -np.inf)
    truth_ids = np.argsort(-sims, axis=1)[:, :k].astype(np.int32)
    truth_cos = np.take_along_axis(sims, truth_ids.astype(np.int64), axis=1)
    kth = truth_cos[:, k - 1].astype(np.float64)
    candidate_cos = np.take_along_axis(sims, truth_ids.astype(np.int64), axis=1)

    strict = strict_containment_rows(truth_ids, truth_ids)
    tie = tie_aware_rows(candidate_cos.astype(np.float64), truth_ids, kth)
    assert float(strict.mean()) == 1.0 and float(tie.mean()) == 1.0
    assert graph_validity(truth_ids, rows=rows)["zero_degree_rows"] == 0

    import umap.umap_ as umap_api

    order = np.argsort(-candidate_cos, axis=1, kind="stable")
    ids_sorted = np.take_along_axis(truth_ids, order, axis=1).astype(np.int32)
    dists = np.maximum(
        (1.0 - np.take_along_axis(candidate_cos, order, axis=1)).astype(np.float32),
        0.0,
    )
    graph, _s, _r = umap_api.fuzzy_simplicial_set(
        host, n_neighbors=k, random_state=np.random.RandomState(42),
        metric="cosine", knn_indices=ids_sorted, knn_dists=dists,
    )
    coo = graph.tocoo()
    assert np.isfinite(coo.data).all() and coo.data.min() > 0
    assert int((np.bincount(coo.row, minlength=rows) == 0).sum()) == 0
    assert len(ordered_array_sha256(host)) == 64


def test_seal_publish_and_reload_round_trips_the_receipt_shape(tmp_path):
    laws = [fit_device_law([
        {"max_cluster_rows": rows, "device_bytes": device, "graph_degree": 64,
         "intermediate_graph_degree": 256, "max_iterations": 40}
        for rows, device in ((170_504, 7.47e9), (3_656_227, 1.35245e10))
    ], label="two")]
    receipt = prompt_contract.seal({
        "schema": "round0235-smoke",
        "round_id": "0235",
        "per_rung_derivation": {
            str(rung): {
                "with_margin": rung_derivation(
                    rung=rung, imbalance_by_c={16: 1.17, 64: 1.6, 200: 2.13},
                    imbalance_source="smoke", laws=laws, apply_margin=True,
                ),
                "point_estimate_no_margin": rung_derivation(
                    rung=rung, imbalance_by_c={16: 1.17, 64: 1.6, 200: 2.13},
                    imbalance_source="smoke", laws=laws, apply_margin=False,
                ),
            }
            for rung in (12_500_000, 25_000_000, 50_000_000, 100_000_000)
        },
        "io_term": io_projection(rows=12_500_000, substrate_passes=8),
        "probe_clusters": list(IMBALANCE_PROBE_CLUSTERS),
        "training_performed": False,
    })
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    reloaded = prompt_contract.read_sealed(str(path), label="smoke")
    assert reloaded["per_rung_derivation"]["25000000"]["with_margin"]["feasible"]
    assert reloaded["io_term"]["io_hours"] > 0
