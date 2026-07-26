from __future__ import annotations

import importlib
import inspect
import os

import numpy as np

from basemap.round0053_program import (
    EXPECTED_EXCLUDED_ROWS,
    EXPECTED_RETAINED_ROWS,
    GLOBAL_150M_INTERVALS,
)
from experiments import prepare_round0060_queue, round0060_nodes


def test_r0060_is_two_stage_no_training_gpu_graph() -> None:
    prep = inspect.getsource(prepare_round0060_queue.prepare_round0060)
    node = inspect.getsource(round0060_nodes)
    assert 'gpu_hours_cap=2.0' in prep
    assert '"qualify_gpu_index_balanced_30m"' in prep
    assert '"build_gpu_native_graph_balanced_30m"' in prep
    assert '"deps": ["qualify_gpu_index_balanced_30m"]' in prep
    assert '"training_performed": False' in prep
    assert '"optimizer_updates": 0' in node
    assert "GpuIndexIVFPQ" in node
    assert "exact_rerank" in node
    assert prepare_round0060_queue.RELEASE_ROOT == (
        "/home/enjalot/code/latent-basemap-run"
    )
    assert round0060_nodes.RUNTIME_SPEC.endswith(
        "round0060_runtime.json"
    )


def test_r0060_keeps_registered_30m_geometry() -> None:
    assert round0060_nodes.NPROBE == 64
    assert round0060_nodes.MAX_PROJECTED_GRAPH_HOURS == 2.0
    assert GLOBAL_150M_INTERVALS == (
        (0, 10_000_000),
        (50_000_000, 60_000_000),
        (100_000_000, 110_000_000),
    )
    assert EXPECTED_RETAINED_ROWS + EXPECTED_EXCLUDED_ROWS == 30_000_000


def test_filtered_index_physically_removes_control_exclusions(
    tmp_path,
    monkeypatch,
) -> None:
    class ProductQuantizer:
        M = 48
        nbits = 8

    class IndexIVFPQ:
        d = 384
        nlist = 8192
        code_size = 48
        pq = ProductQuantizer()

        def __init__(self, ntotal: int) -> None:
            self.ntotal = ntotal

        def reset(self) -> None:
            self.ntotal = 0

        def copy_subset_to(self, destination, _kind, start, stop) -> None:
            destination.ntotal += stop - start

        def remove_ids(self, selector) -> int:
            removed = len(selector.ids)
            self.ntotal -= removed
            return removed

    class Selector:
        def __init__(self, ids) -> None:
            self.ids = np.asarray(ids)

    class InvertedLists:
        SUBSET_TYPE_ID_RANGE = 2

    class FakeFaiss:
        @staticmethod
        def read_index(_path):
            return IndexIVFPQ(150_000_000)

        @staticmethod
        def clone_index(index):
            return IndexIVFPQ(index.ntotal)

        @staticmethod
        def write_index(index, path):
            with open(path, "wb") as handle:
                handle.write(str(index.ntotal).encode())

    FakeFaiss.InvertedLists = InvertedLists
    FakeFaiss.IDSelectorBatch = Selector
    monkeypatch.setattr(round0060_nodes, "INDEX_PATH", "/sealed/index")
    excluded = np.arange(EXPECTED_EXCLUDED_ROWS, dtype=np.int64)
    destination = tmp_path / "filtered.ivfpq"
    index, performance = round0060_nodes._build_filtered_index(
        faiss=FakeFaiss,
        destination_path=str(destination),
        excluded=excluded,
    )
    assert index.ntotal == EXPECTED_RETAINED_ROWS
    assert performance["balanced_range_rows"] == 30_000_000
    assert performance["physically_removed_rows"] == EXPECTED_EXCLUDED_ROWS
    assert performance["filtered_ntotal"] == EXPECTED_RETAINED_ROWS
    assert destination.stat().st_mode & 0o222 == 0


def test_r0060_import_does_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.reload(round0060_nodes)
    importlib.reload(prepare_round0060_queue)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
