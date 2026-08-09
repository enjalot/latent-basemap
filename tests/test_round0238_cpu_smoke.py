"""R0238 CUDA-hidden CPU smoke — reach the code paths a GPU queue would.

The protocol's preparation validation: run the real seal/publish/reload path and
the real downstream interfaces on toy inputs, so a late NameError, an accounting
shape drift or a serialization failure surfaces before hours of GPU time.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from basemap.gpu_child_supervision import CooperativeChildResult

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.round0220_cuvs_qualification import (
    graph_validity,
    strict_containment_rows,
    tie_aware_rows,
)
from basemap.round0238_rung5 import (
    GRAPH_K,
    IMBALANCE_PROBE_CLUSTERS,
    IMBALANCE_REPLICATE_SEEDS,
    PRIMARY_IMBALANCE_SEED,
    ROWS,
    Round0238Error,
    architectural_io,
    fit_device_law,
    imbalance_tolerance,
    io_scaling_fit,
    json_safe,
    physical_io_prediction,
    reachability_cell_summary,
    replicate_grid_table,
    rung_derivation,
    truth_probe_query_rows,
)
import basemap.round0238_rung5 as rung5
import basemap.round0238_build as r0238_build
from basemap import round0113_prompt_contrast as prompt_contract
import experiments.round0238_nodes as nodes
import experiments.prepare_round0238_queue as prepare


R0237_SUBSTRATE_MANIFEST = prepare.R0237_SUBSTRATE_MANIFEST
R0237_LADDER = prepare.R0237_LADDER
R0236_LADDER = prepare.R0236_LADDER
R0237_REACHABILITY = prepare.R0237_REACHABILITY
R0235_LADDER = prepare.R0235_LADDER
R0233_LADDER = prepare.R0233_LADDER
R0229_SWEEP = prepare.R0229_SWEEP
R0229_ARM = prepare.R0229_ARM
R0229_REACHABILITY = prepare.R0229_REACHABILITY


def test_node_dispatch_refuses_an_unregistered_action():
    with pytest.raises(Round0238Error, match="does not authorize"):
        nodes.run_job({"manifest": {"round_id": "0238"}}, {"action": "train"})


def test_build_delegation_is_fail_closed_and_needs_a_capacity(tmp_path):
    import basemap.round0236_build as r0236_build
    import basemap.round0237_build as r0237_build

    checked = r0238_build.assert_reviewed_build_path()
    # The whole delegation chain is checked, link by link, not just the nearest.
    assert checked["r0237_expected_r0236_capacity_rows"] == (
        r0237_build.EXPECTED_R0236_EXPECTED_CAPACITY_ROWS
    )
    assert checked["r0236_expected_r0235_capacity_rows"] == (
        r0236_build.EXPECTED_R0235_EXPECTED_CAPACITY_ROWS
    )
    assert checked["r0235_expected_r0233_capacity_rows"] == 5_204_724
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"setting_id": "x"}), encoding="utf-8")
    with pytest.raises(Round0238Error, match="cluster_capacity_rows"):
        r0238_build.main(["--config", str(config), "--out", str(tmp_path)])

    before = int(r0237_build.EXPECTED_R0236_EXPECTED_CAPACITY_ROWS)
    try:
        r0237_build.EXPECTED_R0236_EXPECTED_CAPACITY_ROWS = before + 1
        with pytest.raises(Round0238Error, match="reviewed build path has moved"):
            r0238_build.assert_reviewed_build_path()
    finally:
        r0237_build.EXPECTED_R0236_EXPECTED_CAPACITY_ROWS = before


def test_io_sampler_reads_only_and_reports_the_registered_fields():
    sampler = nodes._IoSampler(pid=os.getpid())
    reading = nodes._proc_io(os.getpid())
    assert set(reading) >= {"rchar", "wchar", "read_bytes", "write_bytes"}
    sampler.peak = dict(reading)
    published = sampler.readings()
    assert published["child_io_read_bytes"] == reading["read_bytes"]
    assert published["child_io_write_bytes"] == reading["write_bytes"]
    assert nodes._proc_io(-1) == {}


@pytest.fixture
def data_tmp_path():
    """A scratch directory under `/data`, which `output_safety` requires."""
    import shutil
    import tempfile

    root = tempfile.mkdtemp(prefix="round0238-test-", dir="/data/latent-basemap")
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_imbalance_grid_returns_the_record_and_the_grid_the_ladder_unpacks(
    data_tmp_path, monkeypatch
):
    """The first R0238 ladder attempt died here on an arity mismatch.

    The grid runs in a RAPIDS child, so no CPU test reached this function and a
    plain `return` arity error survived to the GPU. This stubs the child - it
    writes the artifact the real probe writes - and calls the real function, so
    the contract between it and `run_ladder` is exercised on CPU.
    """
    written: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        script = command[-1]
        probe_dir = os.path.dirname(script)
        assert os.path.exists(script), "the probe script must be written first"
        body = open(script, encoding="utf-8").read()
        # The child must assert its dataset is a read-only memmap before k-means.
        assert "isinstance(dataset, np.memmap)" in body
        assert "not dataset.flags.writeable" in body
        # R0239: the child must poll the cooperative flag, because the parent
        # no longer has any way to signal it.
        assert "_check_abort()" in body
        assert "except _CooperativeAbort:" in body
        cells = [
            {"rows": rows, "clusters": clusters, "seed": seed, "spill": 8,
             "min": 1, "max": 2, "mean": 1.5, "median": 1.5,
             "empty_clusters": 0, "imbalance_max_over_mean": 1.5 + 1e-5 * seed,
             "elapsed_s": 0.1}
            for rows in (ROWS,)
            for clusters in (128, 200)
            for seed in (226, 236, 1236, 2236, 3236)
        ]
        with open(os.path.join(probe_dir, "imbalance.json"), "w") as handle:
            json.dump({"spill": 8, "cells": cells}, handle)
        written["cells"] = len(cells)

        # R0239: return the real supervision result, so this smoke exercises
        # the receipt shape the node now seals rather than a stub that would
        # drift away from it silently.
        return CooperativeChildResult(
            returncode=0, stdout="", stderr="",
            elapsed_s=0.0, deadline_s=float(kwargs["deadline_s"]),
            flag_written=False, flag_written_at_s=None, escalations=[],
            io_readings={"child_io_samples": 0},
            snapshot_before={"probe": "before"},
            snapshot_after={"probe": "after"},
        )

    monkeypatch.setattr(nodes, "run_gpu_child_cooperative", _fake_run)
    record, grid = nodes._measure_imbalance_grid(
        substrate_path=os.path.join(data_tmp_path, "substrate.f32.npy"),
        output=data_tmp_path, repo_root=data_tmp_path,
        cache_root=data_tmp_path,
    )
    assert written["cells"] == 10
    assert record["primary_seed"] == PRIMARY_IMBALANCE_SEED
    assert record["replicate_seeds"] == [226, 236, 1236, 2236, 3236]
    assert set(grid) == {ROWS}
    assert set(grid[ROWS]) == {128, 200}
    assert set(grid[ROWS][128]) == set(IMBALANCE_REPLICATE_SEEDS)
    # The exact consumption `run_ladder` performs, on the returned grid.
    worst = {c: max(s.values()) for c, s in grid[ROWS].items()}
    primary = {c: s[PRIMARY_IMBALANCE_SEED] for c, s in grid[ROWS].items()}
    assert worst[128] > primary[128]
    assert record["summary"][str(ROWS)]["128"]["n"] == 5
    # And the arity assertion `run_ladder` performs before unpacking.
    result = (record, grid)
    assert isinstance(result, tuple) and len(result) == 2


def test_a_sealed_receipt_with_int_keyed_maps_survives_the_json_round_trip(tmp_path):
    """R0238's first correction queue lost a 48.7-minute build cell to this.

    `seal` hashes `canonical_json` of the in-memory payload; `read_sealed` hashes
    `canonical_json` of what `json.load` returns. A mapping keyed by `int` is
    stringified by `json`, and a canonical sort then orders `'16','200','32'`
    lexicographically where the original ordered `16,32,200` numerically — so the
    seal is computed over one ordering and validated against another. The
    artifact is intact and unreadable at the same time.
    """
    # Plain JSON types, so the ONLY defect in play is the key ordering.
    payload = {
        "schema": "round0238-seal-round-trip",
        "worst_seed_imbalance_at_this_rung": {
            16: 1.17, 32: 1.55, 64: 1.61, 128: 1.79, 200: 2.03, 400: 2.17,
        },
        "nested": {"by_rows": {6_250_000: {64: {226: 1.6}}}},
        "training_performed": False,
    }
    # The defect, demonstrated: sealing the raw payload does NOT survive.
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps(prompt_contract.seal(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="identity seal is invalid"):
        prompt_contract.read_sealed(str(raw), label="raw")

    # The fix: json_safe first, and the same artifact reads back.
    safe = tmp_path / "safe.json"
    safe.write_text(
        json.dumps(prompt_contract.seal(json_safe(payload)), indent=2,
                   sort_keys=True),
        encoding="utf-8",
    )
    reloaded = prompt_contract.read_sealed(str(safe), label="safe")
    assert reloaded["worst_seed_imbalance_at_this_rung"]["200"] == 2.03
    assert reloaded["nested"]["by_rows"]["6250000"]["64"]["226"] == 1.6

    # numpy leaks into receipts through bincount/mean; json_safe unwraps them.
    numpy_safe = json_safe({
        "scalar": np.float32(1.5), "count": np.int64(7),
        "array": np.arange(3, dtype=np.int64),
    })
    assert numpy_safe == {"scalar": pytest.approx(1.5), "count": 7,
                          "array": [0, 1, 2]}
    json.dumps(numpy_safe)


def test_every_r0236_seal_site_goes_through_json_safe():
    """Structural: no receipt in this round may be sealed unguarded."""
    source = open(nodes.__file__, encoding="utf-8").read()
    assert source.count("prompt_contract.seal(") == 5
    assert source.count("prompt_contract.seal(json_safe(") == 5


def test_block_device_and_meminfo_instruments_answer_on_this_box():
    stat = nodes._data_block_device_stat()
    assert stat and stat["bytes_read"] >= 0 and stat["bytes_written"] >= 0
    assert nodes._mem_available_bytes() > 0


@pytest.mark.skipif(
    not os.path.exists(R0237_SUBSTRATE_MANIFEST), reason="R0237 substrate absent"
)
def test_parent_manifest_verifies_and_carries_what_nesting_needs():
    sealed = prompt_contract.read_sealed(
        R0237_SUBSTRATE_MANIFEST, label="R0237 substrate manifest"
    )
    assert sealed["round_id"] == "0237" and sealed["rows"] == 50_000_000
    assert len(sealed["ordered_substrate_sha256"]) == 64
    for key in ("provenance", "reserve_substrate",
                "reserve_provenance", "reserve_query_rows"):
        assert expected_input_signature(
            sealed[key]["canonical_path"]
        ) == dict(sealed[key])
    # The 76.8 GB substrate is size-checked here, not re-hashed: the queue's
    # own preflight and the assemble node's prefix hash both bind its bytes,
    # and a CPU smoke must stay under two minutes (protocol v2.1).
    assert os.path.getsize(sealed["substrate"]["canonical_path"]) == int(
        sealed["substrate"]["bytes"]
    ) == 76_800_000_128
    # The reserve this rung inherits is the one R0233 drew, unchanged.
    assert sealed["reserve"]["inherited_from_round"] == "0236"
    # R0237's manifest names R0235 here because R0235 wrote its own grandparent
    # id into the field and R0236 and R0237 inherited the expression; R0233
    # drew these rows. R0238 seals the literal and does not rewrite the past.
    assert sealed["reserve"]["originally_drawn_by_round"] == "0235"
    assert rung5.RESERVE_DRAWN_BY_ROUND_ID == "0233"
    # R0237's own rung hash is the 50,000,000-row prefix this round must
    # reproduce from its own bytes, and its ladder chain is the rest.
    assert sealed["ordered_substrate_sha256"] == (
        "e7ccf848e0f42e7ac4efa84636b243c9fa75d8e8e0644c21973179cdde1002a9"
    )
    ladder = sealed["nesting"]["ladder_prefix_ordered_sha256"]
    assert ladder["6250000"] == rung5.INHERITED_PREFIX_SHA256[6_250_000]
    assert ladder["12500000"] == rung5.INHERITED_PREFIX_SHA256[12_500_000]
    assert ladder["25000000"] == rung5.INHERITED_PREFIX_SHA256[25_000_000]
    assert ladder["50000000"] == sealed["ordered_substrate_sha256"]
    # The code corpus R0237 sealed is the PRE-extension pool, which is exactly
    # what this round's registered pool change is defined against.
    code = sealed["sources"][rung5.CODE_CORPUS]
    assert int(code["corpus_rows"]) == rung5.CODE_POOL_PARENT_ROWS
    assert int(code["shards"]) == rung5.CODE_POOL_PARENT_SHARDS


@pytest.mark.skipif(
    not (os.path.exists(R0229_SWEEP) and os.path.exists(R0229_ARM)
         and os.path.exists(R0233_LADDER) and os.path.exists(R0235_LADDER)
         and os.path.exists(R0236_LADDER)
         and os.path.exists(R0237_LADDER)),
    reason="inherited sealed artifacts absent",
)
def test_inherited_law_points_are_homogeneous_and_reproduce_the_reviewed_fit():
    job = {
        "r0229_sweep": expected_input_signature(R0229_SWEEP),
        "r0229_arm": expected_input_signature(R0229_ARM),
        "r0233_ladder": expected_input_signature(R0233_LADDER),
        "r0235_ladder": expected_input_signature(R0235_LADDER),
        "r0236_ladder": expected_input_signature(R0236_LADDER),
        "r0237_ladder": expected_input_signature(R0237_LADDER),
    }
    points = nodes._inherited_law_points(job)
    law = fit_device_law(points, label="all-sealed-gd64-igd256-it40")
    assert law["n_points"] == 10
    settings = {
        (p["graph_degree"], p["intermediate_graph_degree"], p["max_iterations"])
        for p in law["points"]
    }
    assert settings == {(64, 256, 40)}
    # The TEN-point law, which is R0237's own refit with its 50M cell added,
    # and which review-0237-01 reproduced to twelve significant figures under
    # its own least squares.
    assert law["slope_bytes_per_max_cluster_row"] == pytest.approx(
        1598.3515811373481, rel=1e-9
    )
    assert law["intercept_bytes"] == pytest.approx(
        7_619_114_558.335568, rel=1e-9
    )
    assert law["r_squared"] == pytest.approx(0.9972485950132305, rel=1e-9)
    assert "q6" in {p["cell"] for p in law["points_refused"]}


@pytest.mark.skipif(
    not (os.path.exists(R0233_LADDER) and os.path.exists(R0236_LADDER)
         and os.path.exists(R0237_LADDER)),
    reason="inherited sealed artifacts absent",
)
def test_inherited_io_points_carry_the_sealed_substrate_reads():
    job = {
        "r0233_ladder": expected_input_signature(R0233_LADDER),
        "r0235_ladder": expected_input_signature(R0235_LADDER),
        "r0236_ladder": expected_input_signature(R0236_LADDER),
        "r0237_ladder": expected_input_signature(R0237_LADDER),
    }
    points = nodes._inherited_io_points(job)
    by_rows = {int(point["rows"]) for point in points}
    assert by_rows == {6_250_000, 12_500_000, 25_000_000, 50_000_000}
    assert all(point["measured_physical_io"] is False for point in points)
    biggest = max(points, key=lambda point: point["substrate_read_bytes"])
    assert biggest["substrate_read_bytes"] == 2_073_600_000_000
    # review-0236-01 F5: every one of these is the identity
    # passes x N x 1536, which is why no exponent fitted across them is cited
    # as a measured scaling.
    for point in points:
        assert point["substrate_read_bytes"] == (
            int(point["substrate_passes"]) * int(point["rows"]) * 1536
        )
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
    table = replicate_grid_table(
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
    truth_best = np.ones(probe.size, dtype=np.float64)
    summary = nodes._score_probe(
        ids=ids, candidate_cos=cos, probe_rows=probe,
        truth_ids=truth_ids, kth=kth, truth_best=truth_best,
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
        truth_ids=truth_ids, kth=kth, truth_best=truth_best,
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
        "schema": "round0238-smoke",
        "round_id": "0238",
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


# --------------------------------------------------------------------------- #
# the instruments new at this rung
# --------------------------------------------------------------------------- #
def test_page_cache_residency_answers_on_a_real_file(tmp_path):
    """`mincore` residency — the direct test of the flip-point assumption."""
    path = tmp_path / "block.bin"
    path.write_bytes(b"\x00" * (8 * 1024 * 1024))
    reading = nodes._page_cache_resident_bytes(str(path))
    assert "error" not in reading
    assert reading["bytes"] == 8 * 1024 * 1024
    assert 0.0 <= float(reading["resident_fraction"]) <= 1.0
    assert reading["resident_bytes"] <= reading["bytes"]
    # A file that does not exist is reported, not raised.
    assert "error" in nodes._page_cache_resident_bytes(
        str(tmp_path / "absent.bin")
    )


def test_device_free_admission_refuses_when_the_shared_card_is_busy(monkeypatch):
    """The card is shared; a 24 GiB budget is not a statement about free memory."""
    monkeypatch.setattr(nodes, "_device_free_bytes", lambda: 8 * 1024 ** 3)
    monkeypatch.setattr(
        nodes, "_foreign_gpu_processes",
        lambda *a, **k: [{"pid": 1234, "used_bytes": 6_638_242_304,
                          "process_name": "embed_marlin.py"}],
    )
    refused = nodes._device_free_admission(18.6 * 1024 ** 3)
    assert refused["admitted"] is False
    assert "free on the card" in refused["reason"]
    assert refused["foreign_compute_processes"][0]["pid"] == 1234
    assert refused["foreign_device_bytes"] == 6_638_242_304

    monkeypatch.setattr(nodes, "_device_free_bytes", lambda: 31 * 1024 ** 3)
    monkeypatch.setattr(nodes, "_foreign_gpu_processes", lambda *a, **k: [])
    assert nodes._device_free_admission(18.6 * 1024 ** 3)["admitted"] is True

    # An unavailable reading must not silently refuse a legal cell.
    monkeypatch.setattr(nodes, "_device_free_bytes", lambda: -1)
    unavailable = nodes._device_free_admission(18.6 * 1024 ** 3)
    assert unavailable["admitted"] is True
    assert unavailable["measurement_unavailable"] is True


def test_the_watchdog_is_conjunctive_and_still_cannot_signal(monkeypatch):
    """R0236 registered this change in advance; it must be auditable, not silent."""
    watchdog = nodes.MemAwareFlagWatchdog(
        flag_path="/dev/null", pid=os.getpid(), poll_s=0.01,
        host_anon_budget_bytes=60 * 1024 ** 3,
        swap_growth_abort_bytes=1 * 1024 ** 3,
        device_baseline_bytes=0, swap_baseline_bytes=0,
    )
    # benign: swap grows under memmap page-cache pressure, box is healthy
    monkeypatch.setattr(nodes, "_nvidia_smi_device_bytes", lambda: 0)
    monkeypatch.setattr(nodes, "_nvidia_smi_per_process_bytes", lambda pid: 0)
    monkeypatch.setattr(nodes, "_proc_memory_bytes", lambda pid: (60 * 1024 ** 3, 2 * 1024 ** 3))
    monkeypatch.setattr(nodes, "_swap_used_bytes", lambda: 3 * 1024 ** 3)
    monkeypatch.setattr(nodes, "_mem_available_bytes", lambda: 120 * 1024 ** 3)
    watchdog.sample_once()
    assert watchdog.aborted is False
    assert watchdog.swap_only_rule_would_have_fired is True
    readings = watchdog.readings()
    assert readings["swap_only_rule_would_have_fired"] is True
    assert readings["mem_available_min_bytes_sampled"] == 120 * 1024 ** 3
    assert readings["watchdog_escalations"] == []

    # genuine pressure: growth AND a large anonymous footprint
    hostile = nodes.MemAwareFlagWatchdog(
        flag_path="/dev/null", pid=os.getpid(), poll_s=0.01,
        host_anon_budget_bytes=60 * 1024 ** 3,
        swap_growth_abort_bytes=1 * 1024 ** 3,
        device_baseline_bytes=0, swap_baseline_bytes=0,
    )
    monkeypatch.setattr(
        nodes, "_proc_memory_bytes", lambda pid: (90 * 1024 ** 3, 45 * 1024 ** 3)
    )
    hostile.sample_once()
    assert hostile.aborted is True
    assert hostile.escalations == ["cooperative-flag"]
    assert "memory pressure" in str(hostile.abort_reason)

    # and no signal path exists on any of it: strip comments and docstrings
    # first, because the prose in this module discusses exactly these calls.
    import ast

    tree = ast.parse(open(nodes.__file__, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            node.value = ast.Constant(value="")
    code = ast.unparse(tree)
    calls = {
        ast.unparse(node.func) for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
    }
    for forbidden in ("os.kill", "signal.signal", "os.killpg"):
        assert forbidden not in calls, forbidden
    assert not any(
        call.endswith((".terminate", ".kill", ".send_signal")) for call in calls
    ), sorted(c for c in calls if c.endswith((".terminate", ".kill")))


def test_host_sampler_reports_the_node_level_fields():
    sampler = nodes._HostSampler(poll_s=0.01)
    sampler.start()
    sampler.halt()
    readings = sampler.readings()
    assert readings["node_host_rss_peak_bytes"] > 0
    assert readings["node_mem_available_min_bytes"] > 0


def test_the_reachability_node_is_registered_and_reachable_by_dispatch():
    assert nodes.REACHABILITY_ACTION == "reachability_100000k"
    with pytest.raises(Round0238Error, match="does not authorize"):
        nodes.run_job(
            {"manifest": {"round_id": "0238"}}, {"action": "reachability_wrong"}
        )


@pytest.mark.skipif(
    not os.path.exists(R0237_REACHABILITY), reason="R0237 reachability absent"
)
def test_the_reachability_node_carries_r0237s_sealed_25m_ceilings():
    """The registered trend reference, read from the artifact not from prose."""
    sealed = prompt_contract.read_sealed(
        R0237_REACHABILITY, label="R0237 high-c reachability"
    )
    assert sealed["round_id"] == "0237" and sealed["rows"] == 25_000_000
    by_c = {
        int(cell["clusters"]): float(cell["strict_ceiling_mean"])
        for cell in sealed["cells"]
    }
    for clusters, expected in rung5.R0237_25M_S8_CEILING_REFERENCE.items():
        assert by_c[int(clusters)] == pytest.approx(expected, abs=5e-7)
    # The partition this rung builds, and the floor it must clear.
    assert by_c[400] >= rung5.REACHABILITY_CONCERN_FLOOR
    # review-0237-01 F1: 1 and 3 rows per million had NO reachable true
    # neighbour at c = 200 / 400, which is why this round reports the count.
    zero = {
        int(cell["clusters"]): int(cell["rows_with_zero_reachable"])
        for cell in sealed["cells"]
    }
    assert zero[64] == 0 and zero[200] == 1 and zero[400] == 3


def test_the_strict_reachability_inner_loop_is_what_the_child_computes():
    """The inner loop, on a toy partition, exactly as the child computes it."""
    assignment = np.array([[0, 1], [0, 2], [1, 2], [3, 4]], dtype=np.int32)
    probe = np.array([0, 3], dtype=np.int64)
    truth_ids = np.array([[1, 2], [0, 2]], dtype=np.int64)
    mine = assignment[probe]
    gathered = assignment[truth_ids]
    shared = (gathered[:, :, :, None] == mine[:, None, None, :]).any(3).any(2)
    strict = shared.sum(axis=1) / 2.0
    assert strict.tolist() == [1.0, 0.0]
    summary = reachability_cell_summary(strict, clusters=400, spill=8)
    assert summary["rows_with_zero_reachable"] == 1
    assert summary["strict_ceiling_min"] == 0.0


def test_intra_queue_references_are_resolved_by_the_intra_resolver():
    """The defect that failed R0238's first queue, and R0233's before it.

    `prompt_contract.verify_signature` requires a sha256 and refuses a reference
    that has none. A first-attempt queue cannot know the sha256 of an artifact
    its own node has not produced yet, so those references carry only a
    `canonical_path` — and must be resolved with `_intra_signature`, which
    checks the hash when one is present and the path when one is not.
    """
    import ast

    tree = ast.parse(open(nodes.__file__, encoding="utf-8").read())
    qualify = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_qualify"
    )
    intra_keys, bound_keys = set(), set()
    for node in ast.walk(qualify):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if not node.args or not isinstance(node.args[-1], ast.Constant):
            continue
        key = node.args[-1].value
        if node.func.id == "_intra":
            intra_keys.add(key)
        elif node.func.id == "_read_bound":
            bound_keys.add(key)
    # Every reference this round's own nodes produce is intra-resolved.
    assert {"truth_reference", "ladder_reference", "reachability_reference"} <= (
        intra_keys
    )
    # And none of them is resolved by the full-signature reader.
    assert not (
        {"truth_reference", "ladder_reference", "reachability_reference"}
        & bound_keys
    )


def test_a_correction_queue_inherits_every_sealed_artifact_it_can():
    """`--inherit-from` must drop a node whose artifact already exists."""
    assert set(prepare.INHERITABLE) == {
        "reachability", "substrate", "truth", "ladder"
    }
    assert set(prepare.INHERIT_NODE) == set(prepare.INHERITABLE)
    for key, node_id in prepare.INHERIT_NODE.items():
        assert key in prepare.INHERITABLE
        assert node_id.startswith(("reachability_", "assemble_", "truth_", "ladder_"))


# --------------------------------------------------------------------------- #
# the two host-memory-bounded re-implementations, checked against the reviewed
# originals rather than argued for. These are the only new science-path code in
# this round and both exist because 100M does not fit the naive path.
# --------------------------------------------------------------------------- #
def test_draw_streaming_is_identical_to_r0233s_reviewed_draw(tmp_path):
    """Same RNG calls, same rejections, same rows, same vectors, same order."""
    from experiments.round0233_nodes import _draw

    rng_state = np.random.RandomState(7)
    shards = []
    rows_per_shard = 400
    for index in range(3):
        block = rng_state.normal(size=(rows_per_shard, 384)).astype(np.float32)
        # a handful of degenerate rows, so the replacement path is exercised
        block[5] = 0.0
        block[11] = np.nan
        path = str(tmp_path / f"shard-{index}.npy")
        np.save(path, block)
        shards.append((path, rows_per_shard, True))
    offsets = nodes._corpus_offsets(shards)
    total = int(offsets[-1])
    want = 900

    picked_a = np.zeros(total, dtype=bool)
    ids_a, vectors_a, dropped_a, rounds_a = _draw(
        shards=shards, offsets=offsets, picked=picked_a,
        rng=np.random.RandomState(238), want=want, corpus="test",
    )
    picked_b = np.zeros(total, dtype=bool)
    stage = str(tmp_path / "stage.npy")
    ids_b, order_b, dropped_b, rounds_b = nodes._draw_streaming(
        shards=shards, offsets=offsets, picked=picked_b,
        rng=np.random.RandomState(238), want=want, corpus="test",
        stage_path=stage,
    )
    assert dropped_a == dropped_b > 0 and rounds_a == rounds_b > 1
    assert np.array_equal(ids_a, ids_b)
    assert np.array_equal(picked_a, picked_b)
    staged = np.load(stage, mmap_mode="r")
    gathered = nodes._gather_staged(
        staged=staged, order=order_b, begin=0, end=want
    )
    assert np.array_equal(vectors_a, gathered)
    # and blockwise, the way the writer actually consumes it
    pieces = [
        nodes._gather_staged(
            staged=staged, order=order_b, begin=begin,
            end=min(begin + 137, want),
        )
        for begin in range(0, want, 137)
    ]
    assert np.array_equal(vectors_a, np.concatenate(pieces, axis=0))


def test_fuzzy_symmetrise_blocked_reproduces_umap_exactly():
    """Bit-identical to `umap.fuzzy_simplicial_set` on the same inputs."""
    import shutil
    import tempfile

    import scipy.sparse as sp
    import umap.umap_ as umap_api

    # `ensure_data_directory` refuses anything outside /data, which is the point
    # of it, so the scratch this stage needs lives there for the test too.
    work = tempfile.mkdtemp(prefix="round0238-fuzzy-", dir="/data/latent-basemap")
    try:
        _fuzzy_case(work, sp, umap_api)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _fuzzy_case(work, sp, umap_api):

    rows, k = 4_000, GRAPH_K
    rng = np.random.RandomState(11)
    data = rng.normal(size=(rows, 96)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    sims = data @ data.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1)[:, :k]
    knn_indices = order.astype(np.int32)
    knn_dists = np.maximum(
        1.0 - np.take_along_axis(sims, order, axis=1), 0.0
    ).astype(np.float32)

    reference, _sigmas, _rhos = umap_api.fuzzy_simplicial_set(
        data, n_neighbors=k, random_state=np.random.RandomState(0),
        metric="cosine", knn_indices=knn_indices, knn_dists=knn_dists.copy(),
    )
    reference = sp.csr_matrix(reference)
    reference.sort_indices()
    reference.eliminate_zeros()

    blocked = nodes._fuzzy_symmetrise_blocked(
        knn_indices=knn_indices, knn_dists=knn_dists.copy(), rows=rows, k=k,
        umap_api=umap_api, out_dir=work, stripe_rows=397,
    )
    mine = sp.coo_matrix(
        (
            np.asarray(blocked["weights"]),
            (np.asarray(blocked["src"]), np.asarray(blocked["dst"])),
        ),
        shape=(rows, rows),
    ).tocsr()
    mine.sort_indices()
    mine.eliminate_zeros()
    assert blocked["stripes"] == int(np.ceil(rows / 397))
    assert mine.nnz == reference.nnz == blocked["directed_edges"]
    assert np.array_equal(mine.indptr, reference.indptr)
    assert np.array_equal(mine.indices, reference.indices)
    # EXACT, not approximate: the arithmetic is scipy's, restricted to rows.
    assert np.array_equal(mine.data, reference.data)


def test_graph_validity_blocked_matches_the_reviewed_function():
    rng = np.random.RandomState(3)
    rows = 5_000
    ids = rng.randint(0, rows, size=(rows, GRAPH_K)).astype(np.int32)
    ids[7, 0] = 7           # a self loop
    ids[9, 1] = ids[9, 0]   # a duplicate
    ids[11, :] = 11         # an entirely edgeless row
    reference = graph_validity(ids, rows=rows)
    blocked = nodes._graph_validity_blocked(ids, rows=rows, block=613)
    assert blocked == reference
    assert reference["zero_degree_rows"] == 1


def test_blocked_descending_sort_matches_the_whole_array_sort():
    rng = np.random.RandomState(5)
    rows = 3_000
    ids = rng.randint(0, rows, size=(rows, GRAPH_K)).astype(np.int32)
    cos = rng.random_sample((rows, GRAPH_K)).astype(np.float32)
    order = np.argsort(-cos, axis=1, kind="stable")
    want_ids = np.take_along_axis(ids, order, axis=1).astype(np.int32)
    want_cos = np.take_along_axis(cos, order, axis=1)
    got_ids, got_cos = nodes._blocked_descending_sort(
        ids=ids, cosines=cos, rows=rows, block=511
    )
    assert np.array_equal(got_ids, want_ids)
    assert np.array_equal(got_cos, want_cos)


# --------------------------------------------------------------------------- #
# the registered selection-law change
# --------------------------------------------------------------------------- #
def test_pool_verification_accepts_the_live_code_corpus_and_records_the_change():
    from experiments.round0233_nodes import _shards

    shards = _shards(rung5.CODE_CORPUS)
    total = int(sum(rows for _p, rows, _n in shards))
    sealed = {
        "corpus_rows": rung5.CODE_POOL_PARENT_ROWS,
        "shards": rung5.CODE_POOL_PARENT_SHARDS,
    }
    record = nodes._verify_pool(
        corpus=rung5.CODE_CORPUS, shards=shards, total=total,
        sealed_source=sealed,
    )
    assert record["pool_extended"] is True
    assert record["corpus_rows"] == rung5.CODE_POOL_ROWS == total
    assert record["shards"] == rung5.CODE_POOL_SHARDS == len(shards)
    assert record["parent_corpus_rows"] == rung5.CODE_POOL_PARENT_ROWS
    assert record["parent_shards_are_a_prefix"] is True
    assert record["corpus_offset_prefix_preserved"] is True
    assert record["appended_shards"] == [
        {"name": "data-00020-of-00021.npy", "rows": 100_000,
         "bytes": 153_600_128, "sorts_after_parent": True}
    ]
    # the arithmetic the registration publishes rather than asserts
    assert record["marginal_inclusion_pre_extension_row"] == pytest.approx(
        0.9950743818100403, rel=1e-12
    )
    assert record["marginal_inclusion_appended_row"] == pytest.approx(
        0.9900990099009901, rel=1e-12
    )
    assert record["appended_row_relative_under_representation"] == pytest.approx(
        0.005, abs=1e-12
    )
    assert record["expected_shortfall_rows"] == pytest.approx(492.5866, rel=1e-5)
    assert record["shortfall_fraction_of_rung"] < 1e-5


def test_pool_verification_refuses_a_corpus_whose_row_ids_would_move(tmp_path):
    sealed = {
        "corpus_rows": rung5.CODE_POOL_PARENT_ROWS,
        "shards": rung5.CODE_POOL_PARENT_SHARDS,
    }
    base = [
        (f"/x/data-{i:05d}-of-00020.npy", 500_000, True) for i in range(20)
    ]
    # an appended shard that sorts FIRST would shift every global row id
    bad = [("/x/data-00000-of-00021.npy", 100_000, True), *base]
    with pytest.raises(Round0238Error):
        nodes._verify_pool(
            corpus=rung5.CODE_CORPUS, shards=bad, total=10_100_000,
            sealed_source=sealed,
        )
    # a base corpus may not change at all
    with pytest.raises(Round0238Error, match="nesting would be a lie"):
        nodes._verify_pool(
            corpus="pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2",
            shards=base, total=10_000_001,
            sealed_source={"corpus_rows": 10_000_000, "shards": 20},
        )


def test_the_registered_adverse_drift_is_carried_not_assumed_safe():
    assert rung5.PREDICTION_IMBALANCE_AT_C400 == 2.456543
    assert rung5.PREDICTION_TOLERANCE_AT_C400 == pytest.approx(0.738255)
    note = rung5.ADVERSE_DRIFT_NOTE
    assert "+3.15%" in note and "+10.69%" in note
    assert "ADVERSE" in note
    # the round must not repeat the retracted "safe direction" claim
    assert "safe direction" not in rung5.REPLICATE_NOTE
    assert "OPPOSITE" in rung5.REPLICATE_NOTE
