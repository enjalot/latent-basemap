from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.round0049_nodes import (
    Round0049Error,
    _validate_shard,
    _write_shard,
)
from basemap.round0065_substrates import subset_spec
from experiments import prepare_round0078_queue, round0078_nodes


def test_round0078_graph_is_fixed_balanced_120m() -> None:
    spec = subset_spec("120m")
    source = inspect.getsource(round0078_nodes.run_build_graph)
    assert "ROW_COUNT" in source
    assert "INTERVALS" in source
    assert "load_gpu_policy_qualification(" in source
    assert "load_scale_decision" not in source
    assert '"scale_decision_made": False' in source
    assert round0078_nodes.ROW_COUNT == 120_000_000
    assert round0078_nodes.INTERVALS == tuple(spec["intervals"])
    assert round0078_nodes.ELIGIBILITY_SUMMARY[
        "retained_row_count"
    ] == 118_067_492


def test_round0078_queue_is_one_resumable_no_training_job() -> None:
    source = inspect.getsource(prepare_round0078_queue.prepare_round0078)
    assert "gpu_hours_cap=8.0" in source
    assert '"p90_wall_s": 28_800.0' in source
    assert '"overlap_adjusted_projection_hours"' in source
    assert "max(" in source
    assert source.count(
        '"action": "build_balanced_120m_gpu_graph"'
    ) == 1
    assert '"shard_rows": 100_000' in source
    assert '"resumable_shards": True' in source
    assert '"rerank_workers": RERANK_WORKERS' in source
    assert '"rerank_blas_threads_per_worker": RERANK_BLAS_THREADS' in source
    assert '"max_pending_reranks": MAX_PENDING_RERANKS' in source
    assert '"gpu_search_cpu_rerank_overlap": True' in source
    assert '"no_training": True' in source
    assert '"no_scale_decision": True' in source
    assert '"required_reviews"] = ["0065", "0081"]' in source


def test_round0078_gpu_clone_preserves_qualified_precision() -> None:
    source = inspect.getsource(round0078_nodes._to_gpu)
    assert "INDICES_64_BIT" in source
    assert "useFloat16 = False" in source
    assert "usePrecomputed = True" in source
    assert "setTempMemory(1 << 30)" in source


def test_round0078_pipelined_rerank_matches_serial_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.RandomState(78)
    encoded = rng.randint(
        -127,
        128,
        size=(200, 384),
        dtype=np.int16,
    ).astype(np.int8)
    scales = np.ones(200, dtype="<f2")
    excluded = np.asarray([2, 17, 150], dtype=np.int64)
    candidate_ids = np.asarray(
        [row for row in range(200) if row not in set(excluded)][:129],
        dtype=np.int64,
    )

    class FakeIndex:
        def search(self, queries, width, *, params=None):
            assert width == 129
            assert params.nprobe == 48
            raw = np.tile(
                candidate_ids,
                (len(queries), 1),
            )
            return np.zeros(raw.shape, dtype=np.float32), raw

    class Parameters:
        nprobe = 48

    identity = lambda rows: np.asarray(rows, dtype=np.int64)
    serial = tmp_path / "serial"
    pipelined = tmp_path / "pipelined"
    serial.mkdir()
    pipelined.mkdir()
    serial_receipt = _write_shard(
        index=FakeIndex(),
        parameters=Parameters(),
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        shard_root=str(serial),
        shard=0,
        start=0,
        stop=40,
        nprobe=48,
        round_id="0078",
        compact_to_global_fn=identity,
        global_to_compact_fn=identity,
        source_rows=200,
    )
    monkeypatch.setattr(round0078_nodes, "SEARCH_BATCH_ROWS", 10)
    pipelined_receipt = round0078_nodes._write_pipelined_shard(
        index=FakeIndex(),
        parameters=Parameters(),
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        shard_root=str(pipelined),
        shard=0,
        start=0,
        stop=40,
        nprobe=48,
        search_width=128,
        compact_to_global_fn=identity,
        global_to_compact_fn=identity,
        source_rows=200,
    )
    serial_targets = np.load(serial / "targets-0000.npy")
    pipelined_targets = np.load(pipelined / "targets-0000.npy")
    assert np.array_equal(serial_targets, pipelined_targets)
    assert np.all(pipelined_targets[[2, 17]] == -1)
    retained_targets = pipelined_targets[
        [row for row in range(40) if row not in {2, 17}]
    ]
    assert not np.any(np.isin(retained_targets, excluded))
    assert (
        serial_receipt["targets"]["sha256"]
        == pipelined_receipt["targets"]["sha256"]
    )
    assert pipelined_receipt["rerank_workers"] == 2
    assert pipelined_receipt["rerank_blas_threads_per_worker"] >= 1
    assert pipelined_receipt["max_pending_reranks"] == 3
    assert pipelined_receipt["search_rerank_overlap"] is True
    with open(pipelined / "receipt-0000.json", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["targets"] == pipelined_receipt["targets"]
    with pytest.raises(
        Round0049Error,
        match="completed graph shard identity changed",
    ):
        _validate_shard(
            target_path=str(pipelined / "targets-0000.npy"),
            receipt_path=str(pipelined / "receipt-0000.json"),
            start=0,
            stop=40,
            nprobe=48,
            search_width=256,
            round_id="0078",
        )


def test_round0078_accepts_only_an_issued_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    round_file = tmp_path / "round-0078.md"
    monkeypatch.setattr(
        prepare_round0078_queue,
        "ROUND_FILE",
        str(round_file),
    )
    round_file.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    prepare_round0078_queue._require_issued_round()
    round_file.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="remains draft"):
        prepare_round0078_queue._require_issued_round()


def test_round0078_modules_do_not_mutate_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0078_graph")
    importlib.import_module("experiments.round0078_nodes")
    importlib.import_module("experiments.prepare_round0078_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
