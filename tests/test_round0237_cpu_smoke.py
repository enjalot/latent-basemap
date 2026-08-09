"""R0237 CUDA-hidden CPU smoke — reach the code paths a GPU queue would.

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
from basemap.round0237_rung4 import (
    GRAPH_K,
    IMBALANCE_PROBE_CLUSTERS,
    IMBALANCE_REPLICATE_SEEDS,
    PRIMARY_IMBALANCE_SEED,
    ROWS,
    Round0237Error,
    architectural_io,
    fit_device_law,
    imbalance_tolerance,
    io_scaling_fit,
    json_safe,
    physical_io_prediction,
    replicate_grid_table,
    rung_derivation,
    truth_probe_query_rows,
)
import basemap.round0237_build as r0237_build
from basemap import round0113_prompt_contrast as prompt_contract
import experiments.round0237_nodes as nodes
import experiments.prepare_round0237_queue as prepare


R0236_SUBSTRATE_MANIFEST = prepare.R0236_SUBSTRATE_MANIFEST
R0236_TRUTH_MANIFEST = prepare.R0236_TRUTH_MANIFEST
R0236_LADDER = prepare.R0236_LADDER
R0235_LADDER = prepare.R0235_LADDER
R0233_LADDER = prepare.R0233_LADDER
R0229_SWEEP = prepare.R0229_SWEEP
R0229_ARM = prepare.R0229_ARM
R0229_REACHABILITY = prepare.R0229_REACHABILITY


def test_node_dispatch_refuses_an_unregistered_action():
    with pytest.raises(Round0237Error, match="does not authorize"):
        nodes.run_job({"manifest": {"round_id": "0237"}}, {"action": "train"})


def test_build_delegation_is_fail_closed_and_needs_a_capacity(tmp_path):
    import basemap.round0236_build as r0236_build

    checked = r0237_build.assert_reviewed_build_path()
    assert checked["r0236_expected_r0235_capacity_rows"] == (
        r0236_build.EXPECTED_R0235_EXPECTED_CAPACITY_ROWS
    )
    # the whole delegation chain is checked, not just the nearest link
    assert checked["r0235_expected_r0233_capacity_rows"] == 5_204_724
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"setting_id": "x"}), encoding="utf-8")
    with pytest.raises(Round0237Error, match="cluster_capacity_rows"):
        r0237_build.main(["--config", str(config), "--out", str(tmp_path)])

    before = int(r0236_build.EXPECTED_R0235_EXPECTED_CAPACITY_ROWS)
    try:
        r0236_build.EXPECTED_R0235_EXPECTED_CAPACITY_ROWS = before + 1
        with pytest.raises(Round0237Error, match="reviewed build path has moved"):
            r0237_build.assert_reviewed_build_path()
    finally:
        r0236_build.EXPECTED_R0235_EXPECTED_CAPACITY_ROWS = before


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

    root = tempfile.mkdtemp(prefix="round0237-test-", dir="/data/latent-basemap")
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_imbalance_grid_returns_the_record_and_the_grid_the_ladder_unpacks(
    data_tmp_path, monkeypatch
):
    """The first R0237 ladder attempt died here on an arity mismatch.

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

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr(nodes.subprocess, "run", _fake_run)
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
    """R0237's first correction queue lost a 48.7-minute build cell to this.

    `seal` hashes `canonical_json` of the in-memory payload; `read_sealed` hashes
    `canonical_json` of what `json.load` returns. A mapping keyed by `int` is
    stringified by `json`, and a canonical sort then orders `'16','200','32'`
    lexicographically where the original ordered `16,32,200` numerically — so the
    seal is computed over one ordering and validated against another. The
    artifact is intact and unreadable at the same time.
    """
    # Plain JSON types, so the ONLY defect in play is the key ordering.
    payload = {
        "schema": "round0237-seal-round-trip",
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
    not os.path.exists(R0236_SUBSTRATE_MANIFEST), reason="R0236 substrate absent"
)
def test_parent_manifest_verifies_and_carries_what_nesting_needs():
    sealed = prompt_contract.read_sealed(
        R0236_SUBSTRATE_MANIFEST, label="R0236 substrate manifest"
    )
    assert sealed["round_id"] == "0236" and sealed["rows"] == 25_000_000
    assert len(sealed["ordered_substrate_sha256"]) == 64
    for key in ("provenance", "reserve_substrate",
                "reserve_provenance", "reserve_query_rows"):
        assert expected_input_signature(
            sealed[key]["canonical_path"]
        ) == dict(sealed[key])
    # The 38.4 GB substrate is size-checked here, not re-hashed: the queue's
    # own preflight and the assemble node's prefix hash both bind its bytes,
    # and a CPU smoke must stay under two minutes (protocol v2.1).
    assert os.path.getsize(sealed["substrate"]["canonical_path"]) == int(
        sealed["substrate"]["bytes"]
    ) == 38_400_000_128
    # The reserve this rung inherits is the one R0233 drew, unchanged.
    assert sealed["reserve"]["inherited_from_round"] == "0235"
    assert sealed["reserve"]["originally_drawn_by_round"] == "0233"
    # And R0236's own prefix hashes are what this round must reproduce.
    assert sealed["nesting"]["prefix_ordered_sha256"] == (
        "bd004db8511c9e3ea44bbc1471f739cdcc5d78adb35f7cb53c96919638ec7ad5"
    )
    assert sealed["nesting"]["grandparent"]["prefix_ordered_sha256"] == (
        "5d976ab6d895db45095967afd5ce7dd078a6242bc62edfff941c120fec473e36"
    )


@pytest.mark.skipif(
    not (os.path.exists(R0229_SWEEP) and os.path.exists(R0229_ARM)
         and os.path.exists(R0233_LADDER) and os.path.exists(R0235_LADDER)
         and os.path.exists(R0236_LADDER)),
    reason="inherited sealed artifacts absent",
)
def test_inherited_law_points_are_homogeneous_and_reproduce_the_reviewed_fit():
    job = {
        "r0229_sweep": expected_input_signature(R0229_SWEEP),
        "r0229_arm": expected_input_signature(R0229_ARM),
        "r0233_ladder": expected_input_signature(R0233_LADDER),
        "r0235_ladder": expected_input_signature(R0235_LADDER),
        "r0236_ladder": expected_input_signature(R0236_LADDER),
    }
    points = nodes._inherited_law_points(job)
    law = fit_device_law(points, label="all-sealed-gd64-igd256-it40")
    assert law["n_points"] == 9
    settings = {
        (p["graph_degree"], p["intermediate_graph_degree"], p["max_iterations"])
        for p in law["points"]
    }
    assert settings == {(64, 256, 40)}
    # The nine-point law R0236 refitted with its own cell added, and
    # review-0236-01 reproduced under its own least squares.
    assert law["slope_bytes_per_max_cluster_row"] == pytest.approx(
        1599.9675148288215, rel=1e-9
    )
    assert law["intercept_bytes"] == pytest.approx(
        7_617_825_807.764802, rel=1e-9
    )
    assert law["r_squared"] == pytest.approx(0.9968238977523114, rel=1e-9)
    assert "q6" in {p["cell"] for p in law["points_refused"]}


@pytest.mark.skipif(
    not (os.path.exists(R0233_LADDER) and os.path.exists(R0236_LADDER)),
    reason="inherited sealed artifacts absent",
)
def test_inherited_io_points_carry_the_sealed_substrate_reads():
    job = {
        "r0233_ladder": expected_input_signature(R0233_LADDER),
        "r0235_ladder": expected_input_signature(R0235_LADDER),
        "r0236_ladder": expected_input_signature(R0236_LADDER),
    }
    points = nodes._inherited_io_points(job)
    by_rows = {int(point["rows"]) for point in points}
    assert by_rows == {6_250_000, 12_500_000, 25_000_000}
    assert all(point["measured_physical_io"] is False for point in points)
    biggest = max(points, key=lambda point: point["substrate_read_bytes"])
    assert biggest["substrate_read_bytes"] == 537_600_000_000
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
        "schema": "round0237-smoke",
        "round_id": "0237",
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
    assert nodes.REACHABILITY_ACTION == "reachability_high_c_25000k"
    with pytest.raises(Round0237Error, match="does not authorize"):
        nodes.run_job(
            {"manifest": {"round_id": "0237"}}, {"action": "reachability_wrong"}
        )


@pytest.mark.skipif(
    not (os.path.exists(R0236_TRUTH_MANIFEST)
         and os.path.exists(R0236_SUBSTRATE_MANIFEST)),
    reason="R0236 artifacts absent",
)
def test_the_reachability_node_reads_r0236s_sealed_probe_and_truth():
    truth = prompt_contract.read_sealed(
        R0236_TRUTH_MANIFEST, label="R0236 probe truth manifest"
    )
    assert truth["round_id"] == "0236" and truth["rows"] == 25_000_000
    assert truth["probe_rows"] == 1_000_000
    ids = np.load(truth["outputs"]["ids"]["canonical_path"], mmap_mode="r")
    rows = np.load(truth["outputs"]["query_rows"]["canonical_path"], mmap_mode="r")
    assert ids.shape == (1_000_000, GRAPH_K)
    assert rows.shape == (1_000_000,)
    assert int(rows.min()) >= 0 and int(rows.max()) < 25_000_000
    # The strict-reachability inner loop, on a toy partition, exactly as the
    # child computes it.
    assignment = np.array([[0, 1], [0, 2], [1, 2], [3, 4]], dtype=np.int32)
    probe = np.array([0, 3], dtype=np.int64)
    truth_ids = np.array([[1, 2], [0, 2]], dtype=np.int64)
    mine = assignment[probe]
    gathered = assignment[truth_ids]
    shared = (gathered[:, :, :, None] == mine[:, None, None, :]).any(3).any(2)
    strict = shared.sum(axis=1) / 2.0
    assert strict.tolist() == [1.0, 0.0]
