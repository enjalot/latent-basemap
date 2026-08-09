"""R0236 CUDA-hidden CPU smoke — reach the code paths a GPU queue would.

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
from basemap.round0235_rung2 import rung_derivation
from basemap.round0236_rung3 import (
    GRAPH_K,
    IMBALANCE_PROBE_CLUSTERS,
    PRIMARY_IMBALANCE_SEED,
    ROWS,
    Round0236Error,
    architectural_io,
    fit_device_law,
    imbalance_tolerance,
    io_scaling_fit,
    physical_io_prediction,
    replicate_drift,
    truth_probe_query_rows,
)
import basemap.round0236_build as r0236_build
from basemap import round0113_prompt_contrast as prompt_contract
import experiments.round0236_nodes as nodes
import experiments.prepare_round0236_queue as prepare


R0235_SUBSTRATE_MANIFEST = prepare.R0235_SUBSTRATE_MANIFEST
R0235_LADDER = prepare.R0235_LADDER
R0233_LADDER = prepare.R0233_LADDER
R0229_SWEEP = prepare.R0229_SWEEP
R0229_ARM = prepare.R0229_ARM
R0229_REACHABILITY = prepare.R0229_REACHABILITY


def test_node_dispatch_refuses_an_unregistered_action():
    with pytest.raises(Round0236Error, match="does not authorize"):
        nodes.run_job({"manifest": {"round_id": "0236"}}, {"action": "train"})


def test_build_delegation_is_fail_closed_and_needs_a_capacity(tmp_path):
    import basemap.round0235_build as r0235_build

    assert r0236_build.assert_reviewed_build_path()[
        "r0235_expected_r0233_capacity_rows"
    ] == r0235_build.EXPECTED_R0233_CAPACITY_ROWS
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"setting_id": "x"}), encoding="utf-8")
    with pytest.raises(Round0236Error, match="cluster_capacity_rows"):
        r0236_build.main(["--config", str(config), "--out", str(tmp_path)])

    before = int(r0235_build.EXPECTED_R0233_CAPACITY_ROWS)
    try:
        r0235_build.EXPECTED_R0233_CAPACITY_ROWS = before + 1
        with pytest.raises(Round0236Error, match="reviewed build path has moved"):
            r0236_build.assert_reviewed_build_path()
    finally:
        r0235_build.EXPECTED_R0233_CAPACITY_ROWS = before


def test_io_sampler_reads_only_and_reports_the_registered_fields():
    sampler = nodes._IoSampler(pid=os.getpid())
    reading = nodes._proc_io(os.getpid())
    assert set(reading) >= {"rchar", "wchar", "read_bytes", "write_bytes"}
    sampler.peak = dict(reading)
    published = sampler.readings()
    assert published["child_io_read_bytes"] == reading["read_bytes"]
    assert published["child_io_write_bytes"] == reading["write_bytes"]
    assert nodes._proc_io(-1) == {}


def test_block_device_and_meminfo_instruments_answer_on_this_box():
    stat = nodes._data_block_device_stat()
    assert stat and stat["bytes_read"] >= 0 and stat["bytes_written"] >= 0
    assert nodes._mem_available_bytes() > 0


@pytest.mark.skipif(
    not os.path.exists(R0235_SUBSTRATE_MANIFEST), reason="R0235 substrate absent"
)
def test_parent_manifest_verifies_and_carries_what_nesting_needs():
    sealed = prompt_contract.read_sealed(
        R0235_SUBSTRATE_MANIFEST, label="R0235 substrate manifest"
    )
    assert sealed["round_id"] == "0235" and sealed["rows"] == 12_500_000
    assert len(sealed["ordered_substrate_sha256"]) == 64
    for key in ("substrate", "provenance", "reserve_substrate",
                "reserve_provenance", "reserve_query_rows"):
        assert expected_input_signature(
            sealed[key]["canonical_path"]
        ) == dict(sealed[key])
    # The reserve this rung inherits is the one R0233 drew, unchanged.
    assert sealed["reserve"]["inherited_from_round"] == "0233"


@pytest.mark.skipif(
    not (os.path.exists(R0229_SWEEP) and os.path.exists(R0229_ARM)
         and os.path.exists(R0233_LADDER) and os.path.exists(R0235_LADDER)),
    reason="inherited sealed artifacts absent",
)
def test_inherited_law_points_are_homogeneous_and_reproduce_the_reviewed_fit():
    job = {
        "r0229_sweep": expected_input_signature(R0229_SWEEP),
        "r0229_arm": expected_input_signature(R0229_ARM),
        "r0233_ladder": expected_input_signature(R0233_LADDER),
        "r0235_ladder": expected_input_signature(R0235_LADDER),
    }
    points = nodes._inherited_law_points(job)
    law = fit_device_law(points, label="all-sealed-gd64-igd256-it40")
    assert law["n_points"] == 8
    settings = {
        (p["graph_degree"], p["intermediate_graph_degree"], p["max_iterations"])
        for p in law["points"]
    }
    assert settings == {(64, 256, 40)}
    # review-0235-01 verified this fit independently.
    assert law["slope_bytes_per_max_cluster_row"] == pytest.approx(
        1600.461148, rel=1e-6
    )
    assert law["intercept_bytes"] == pytest.approx(7.617621e9, rel=1e-5)
    assert law["r_squared"] == pytest.approx(0.99633161, rel=1e-6)
    assert "q6" in {p["cell"] for p in law["points_refused"]}


@pytest.mark.skipif(
    not (os.path.exists(R0233_LADDER) and os.path.exists(R0235_LADDER)),
    reason="inherited sealed artifacts absent",
)
def test_inherited_io_points_carry_the_sealed_substrate_reads():
    job = {
        "r0233_ladder": expected_input_signature(R0233_LADDER),
        "r0235_ladder": expected_input_signature(R0235_LADDER),
    }
    points = nodes._inherited_io_points(job)
    by_rows = {int(point["rows"]) for point in points}
    assert by_rows == {6_250_000, 12_500_000}
    assert all(point["measured_physical_io"] is False for point in points)
    biggest = max(points, key=lambda point: point["substrate_read_bytes"])
    assert biggest["substrate_read_bytes"] == 153_600_000_000
    fit = io_scaling_fit(points)
    assert fit["n_points"] == len(points)
    assert fit["exponent"] > 1.0


@pytest.mark.skipif(
    not os.path.exists(R0229_REACHABILITY), reason="R0229 reachability absent"
)
def test_inherited_2m_imbalance_stays_a_single_realisation():
    job = {"r0229_reachability": expected_input_signature(R0229_REACHABILITY)}
    series = nodes._inherited_imbalance(job)
    at_2m = series["by_rows"][2_000_000]
    assert set(at_2m) == {16, 64, 200}
    assert 32 not in at_2m
    table = replicate_drift(
        {ROWS: {200: {226: 1.98, 236: 2.00, 1236: 1.99}}},
        inherited=series["by_rows"],
    )
    assert table["by_clusters"]["200"]["by_rows"]["2000000"]["replicated"] is False
    assert table["by_clusters"]["200"]["by_rows"][str(ROWS)]["replicated"] is True


def test_qualification_and_fuzzy_law_close_on_a_tiny_memmapped_substrate(tmp_path):
    rows, dim, k = 512, 8, GRAPH_K
    rng = np.random.default_rng(236)
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

    strict = strict_containment_rows(truth_ids, truth_ids)
    tie = tie_aware_rows(truth_cos.astype(np.float64), truth_ids, kth)
    assert float(strict.mean()) == 1.0 and float(tie.mean()) == 1.0
    structural = graph_validity(truth_ids, rows=rows)
    assert structural["zero_degree_rows"] == 0

    import umap.umap_ as umap_api

    order = np.argsort(-truth_cos, axis=1, kind="stable")
    ids_sorted = np.take_along_axis(truth_ids, order, axis=1).astype(np.int32)
    dists = np.maximum(
        (1.0 - np.take_along_axis(truth_cos, order, axis=1)).astype(np.float32), 0.0
    )
    graph, _s, _r = umap_api.fuzzy_simplicial_set(
        host, n_neighbors=k, random_state=np.random.RandomState(42),
        metric="cosine", knn_indices=ids_sorted, knn_dists=dists,
    )
    coo = graph.tocoo()
    assert np.isfinite(coo.data).all() and coo.data.min() > 0
    # Both directions, which is what this round's tripwire checks.
    assert int((np.bincount(coo.row, minlength=rows) == 0).sum()) == 0
    assert int((np.bincount(coo.col, minlength=rows) == 0).sum()) == 0
    assert len(ordered_array_sha256(host)) == 64


def test_probe_scoring_path_closes_on_a_toy_graph():
    rng = np.random.default_rng(1236)
    rows, k = 4_000, GRAPH_K
    probe = np.sort(rng.choice(rows, size=400, replace=False)).astype(np.int64)
    # Distinct ids within each row, so a perfect graph scores exactly 1.0 and a
    # shortfall cannot be an artefact of within-row duplication.
    ids = np.stack(
        [rng.choice(rows, size=k, replace=False) for _ in range(rows)]
    ).astype(np.int32)
    cos = rng.random((rows, k)).astype(np.float32)
    truth_ids = ids[probe].astype(np.int32)
    kth = np.zeros(probe.size, dtype=np.float64)
    summary = nodes._score_probe(
        ids=ids, candidate_cos=cos, probe_rows=probe,
        truth_ids=truth_ids, kth=kth,
    )
    assert summary["rows_measured"] == 400
    assert summary["strict"]["mean"] == pytest.approx(1.0)
    assert summary["tie_aware"]["mean"] == pytest.approx(1.0)
    assert len(summary["density_decile_tie_aware"]) == 10
    assert summary["missing_true_edges"] == 0
    assert summary["rows_carrying_any_loss"] == 0

    # And a graph that drops one true neighbour per probe row must show it.
    damaged = ids.copy()
    damaged[probe, 0] = (damaged[probe, 0] + 1) % rows
    lossy = nodes._score_probe(
        ids=damaged, candidate_cos=cos, probe_rows=probe,
        truth_ids=truth_ids, kth=kth,
    )
    assert lossy["strict"]["mean"] < 1.0
    assert lossy["missing_true_edges"] == 400


def test_seal_publish_and_reload_round_trips_the_receipt_shape(tmp_path):
    laws = [fit_device_law([
        {"max_cluster_rows": rows, "device_bytes": device, "graph_degree": 64,
         "intermediate_graph_degree": 256, "max_iterations": 40}
        for rows, device in (
            (170_504, 7_470_055_424.0), (7_275_244, 19_107_151_872.0),
        )
    ], label="two")]
    imbalance = {64: 1.6486848, 128: 1.7144294, 200: 1.98499}
    receipt = prompt_contract.seal({
        "schema": "round0236-smoke",
        "round_id": "0236",
        "probe_rows": int(truth_probe_query_rows(rows=1000, size=10, seed=1).size),
        "per_rung_derivation": {
            str(rung): {
                "with_margin": rung_derivation(
                    rung=rung, imbalance_by_c=imbalance, imbalance_source="smoke",
                    laws=laws, apply_margin=True,
                ),
                "imbalance_tolerance_by_c": {
                    str(c): imbalance_tolerance(
                        rung=rung, clusters=c, imbalance=value, laws=laws
                    )
                    for c, value in imbalance.items()
                },
                "io": physical_io_prediction(
                    rows=rung,
                    substrate_passes=max(1, rung // 2_000_000),
                ),
            }
            for rung in (25_000_000, 50_000_000, 100_000_000)
        },
        "io_scaling": io_scaling_fit([
            {"rows": rows, "substrate_read_bytes": architectural_io(
                rows=rows, substrate_passes=max(1, rows // 2_000_000)
            )["substrate_read_bytes"]}
            for rows in (6_250_000, 12_500_000, 25_000_000)
        ]),
        "probe_clusters": list(IMBALANCE_PROBE_CLUSTERS),
        "primary_seed": PRIMARY_IMBALANCE_SEED,
        "training_performed": False,
    })
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    reloaded = prompt_contract.read_sealed(str(path), label="smoke")
    at_25m = reloaded["per_rung_derivation"]["25000000"]
    assert at_25m["with_margin"]["feasible"]
    assert at_25m["imbalance_tolerance_by_c"]["64"][
        "tolerance_to_adverse_imbalance"
    ] > 0
    assert at_25m["io"]["substrate_fits_page_cache"] is True
    assert reloaded["per_rung_derivation"]["100000000"]["io"][
        "substrate_fits_page_cache"
    ] is False
    assert reloaded["io_scaling"]["exponent"] > 1.5
