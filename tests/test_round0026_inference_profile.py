from __future__ import annotations

import numpy as np

from experiments import prepare_round0026_queue
from experiments import run_round0014_node
from experiments.export_inference_profile import _latency_summary, _make_batch


def test_round0026_make_batch_tiles_sample_without_changing_dtype():
    sample = np.arange(12, dtype=np.float32).reshape(3, 4)
    batch = _make_batch(sample, 8)
    assert batch.shape == (8, 4)
    assert batch.dtype == np.dtype(np.float32)
    assert batch[:3].tolist() == sample.tolist()
    assert batch[3:6].tolist() == sample.tolist()
    assert batch.flags.c_contiguous


def test_round0026_latency_summary_reports_median_rows_per_second():
    summary = _latency_summary([0.2, 0.1, 0.3], rows=10)
    assert summary["repetitions"] == 3
    assert summary["latency_seconds_p50"] == 0.2
    assert summary["rows_per_second_median"] == 50.0


def test_round0026_queue_hides_cuda_until_gpu_handler():
    env = prepare_round0026_queue._child_environment("/tmp/r0026-queue")
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["BASEMAP_ROUND0026_GPU_UUID"].startswith("GPU-")


def test_round0026_configure_selects_profile_runtime():
    run_round0014_node.configure_round0026()
    assert run_round0014_node.ROUND_ID == "0026"
    assert run_round0014_node.RUNTIME_SCRIPT == "experiments/export_inference_profile.py"
    assert hasattr(run_round0014_node, "_run_gpu_inference_profile")
