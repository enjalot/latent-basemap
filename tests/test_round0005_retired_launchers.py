"""No superseded or generic scale entry point may reach a child/CUDA silently."""
from __future__ import annotations

import importlib
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from basemap.round0005_retirement import (ADMITTED_GPU_ENTRYPOINTS,
                                          EXECUTABLE_GPU_ENTRYPOINTS,
                                          RETIRED_LAUNCHERS,
                                          RetiredLauncherError)


RETIRED_MAIN_MODULES = (
    "experiments.dag_template",
    "experiments.run_o2_frontier",
    "experiments.run_o2_4m",
    "experiments.run_g0",
    "experiments.run_backfill_2m_s44",
    "experiments.run_o1_prompted",
    "experiments.run_overnight_program",
    "experiments.score_a3_rescore",
    "experiments.build_reference",
    "experiments.run_8m_canary",
    "experiments.run_r1_kernel",
    "experiments.run_r1_ablation",
    "experiments.run_experiment",
    "experiments.run_canary",
    "experiments.build_prompted_graph",
    "experiments.embed_prompted_200k",
    "experiments.golden_validate",
    "experiments.run_round0001_gpu_canary",
    "experiments.bench_input_pipeline",
    "experiments.profile_pipeline",
    "experiments.measure_knn_cost",
    "experiments.build_testbed",
    "experiments.score_8m_bridge",
    "experiments.space_passport",
    "scale_experiment",
    "train_local",
    "validate_umap",
)

REMOTE_MODULE_PATHS = (
    "train_dedicated_modal.py",
    "build_150m_index_modal.py",
    "train_15m_modal.py",
    "train_and_project_modal.py",
    "sweep_global_modal.py",
    "sweep_structure_modal.py",
    "sweep_v3_modal.py",
    "bench_knn_modal.py",
    "bench_query_a100.py",
    "bench_scale_modal.py",
    "bench_throughput_modal.py",
    "bench_train_gpu_modal.py",
    "build_faiss_index_modal.py",
    "debug_faiss_modal.py",
    "train_combined_modal.py",
    "train_modal.py",
)

LOCAL_EXECUTABLE_PATHS = (
    "autoresearch/prepare.py",
    "project_local.py",
)


@pytest.mark.parametrize("module_name", RETIRED_MAIN_MODULES)
def test_retired_main_stops_before_controller_or_output(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    calls = []
    if hasattr(module, "run_jobs"):
        monkeypatch.setattr(
            module, "run_jobs", lambda *args, **kwargs: calls.append("run_jobs"))
    if hasattr(module, "run_single_experiment"):
        monkeypatch.setattr(
            module, "run_single_experiment",
            lambda *args, **kwargs: calls.append("run_single_experiment"))
    with pytest.raises(RetiredLauncherError, match="RETIRED for Round 0005"):
        module.main()
    assert calls == []


def test_retired_job_builders_cannot_be_reused_directly(tmp_path):
    dag = importlib.import_module("experiments.dag_template")
    with pytest.raises(RetiredLauncherError):
        dag.build_g1_dag(
            train_cfg="unused", work_dir=str(tmp_path / "dag"), reference="unused",
            legacy_run_dir="unused", testbed_2m="unused", kernel_run_2m="unused",
            decision_out="unused", gate_out="unused")
    assert not (tmp_path / "dag").exists()

    g0 = importlib.import_module("experiments.run_g0")
    with pytest.raises(RetiredLauncherError):
        g0.score_job("2m", "unused", (42,), "unused", "unused")

    overnight = importlib.import_module("experiments.run_overnight_program")
    with pytest.raises(RetiredLauncherError):
        overnight.phase_8m_program("unused", {})


@pytest.mark.parametrize("relative_path", REMOTE_MODULE_PATHS)
def test_remote_scale_module_retires_before_modal_import_or_spawn(relative_path):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(RetiredLauncherError, match="RETIRED for Round 0005"):
        runpy.run_path(str(root / relative_path), run_name="round0005_retirement_probe")


@pytest.mark.parametrize("relative_path", LOCAL_EXECUTABLE_PATHS)
def test_local_executable_retires_before_argument_work(relative_path):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(RetiredLauncherError, match="RETIRED for Round 0005"):
        runpy.run_path(str(root / relative_path), run_name="__main__")


def test_autoresearch_train_function_is_hard_retired():
    module = importlib.import_module("autoresearch.train")
    with pytest.raises(RetiredLauncherError, match="RETIRED for Round 0005"):
        module.train()


def test_generic_experiment_reopens_rows_and_requires_certificate_before_cuda(
        fresh_data_root):
    from experiments.run_experiment import _round0005_scale_preflight

    embedding_root = os.path.join(fresh_data_root, "generic-8m")
    os.mkdir(embedding_root)
    embedding = os.path.join(embedding_root, "data-00000.npy")
    matrix = np.lib.format.open_memmap(
        embedding, mode="w+", dtype=np.float32, shape=(8_000_000, 1))
    matrix.flush()
    del matrix
    cfg = SimpleNamespace(data=SimpleNamespace(
        source="memmap", memmap_dirs=[embedding_root], input_dim=1, n_samples=None))
    with pytest.raises(RuntimeError, match="requires --performance-gate"):
        _round0005_scale_preflight(
            cfg, performance_gate=None, release_sha=None)


def test_generic_canary_retires_before_child_or_scratch(
        fresh_data_root, monkeypatch):
    module = importlib.import_module("experiments.run_canary")
    child_calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: child_calls.append(1))
    out = os.path.join(fresh_data_root, "must-not-exist", "verdict.json")
    monkeypatch.setattr(sys, "argv", [
        "run_canary.py", "--train-config", "unused", "--run-dir",
        os.path.join(fresh_data_root, "run"), "--out", out,
    ])
    with pytest.raises(RetiredLauncherError, match="RETIRED for Round 0005"):
        module.main()
    assert child_calls == []
    assert not os.path.lexists(os.path.dirname(out))


class _EightMillionRows:
    def __len__(self):
        return 8_000_000


def test_complete_scorer_requires_controller_admission_before_rows_or_cuda(
        fresh_data_root, monkeypatch):
    module = importlib.import_module("experiments.score_complete_panel")
    out_root = os.path.join(fresh_data_root, "generic-scorer")
    os.mkdir(out_root)
    monkeypatch.setattr(module, "load_embeddings", lambda *args, **kwargs: _EightMillionRows())
    monkeypatch.setattr(sys, "argv", [
        "score_complete_panel.py", "--runs", "map=/unused", "--testbed", "/unused",
        "--no-model", "--out-root", out_root,
    ])
    with pytest.raises(RuntimeError, match="requires controller admission identity"):
        module.main()
    assert os.listdir(out_root) == []


def test_panel_cli_retires_from_actual_rows_before_coords_or_output(
        fresh_data_root, monkeypatch):
    import basemap.panel_v2 as panel

    out = os.path.join(fresh_data_root, "panel-must-not-exist.json")
    monkeypatch.setattr(panel, "load_embeddings", lambda *args, **kwargs: _EightMillionRows())
    monkeypatch.setattr(sys, "argv", [
        "panel_v2.py", "--emb", "/unused", "--coords", "/unused", "--out", out,
    ])
    with pytest.raises(RetiredLauncherError, match="evaluator CLI"):
        panel.main()
    assert not os.path.lexists(out)


def test_legacy_eval_retires_actual_scale_input_before_output(fresh_data_root):
    import basemap.eval as evaluator

    embedding = os.path.join(fresh_data_root, "legacy-eval-8m.npy")
    matrix = np.lib.format.open_memmap(
        embedding, mode="w+", dtype=np.float32, shape=(8_000_000, 1))
    matrix.flush()
    del matrix
    coords = os.path.join(fresh_data_root, "must-not-open.parquet")
    output = os.path.join(fresh_data_root, "legacy-eval-must-not-exist.json")
    with pytest.raises(RetiredLauncherError, match="RETIRED for Round 0005"):
        evaluator.main([
            "score", "--coords", coords, "--embeddings", embedding,
            "--out", output,
        ])
    assert not os.path.lexists(output)


def test_legacy_eval_direct_command_function_is_also_retired():
    import basemap.eval as evaluator

    with pytest.raises(RetiredLauncherError, match="RETIRED for Round 0005"):
        evaluator.cmd_score(SimpleNamespace())


@pytest.mark.parametrize("module_name", ["basemap.panel_v2", "basemap.eval"])
def test_retired_scoring_modules_reject_import_as_main_and_alternate_argv(
        module_name, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module_name, "--unexpected", "value"])
    with pytest.raises(RetiredLauncherError, match="RETIRED for Round 0005"):
        runpy.run_module(module_name, run_name="__main__")


def test_subscale_panel_cuda_visibility_requires_capability_before_torch_probe(
        monkeypatch):
    import basemap.panel_v2 as panel

    calls = []

    class NeverCuda:
        @staticmethod
        def is_available():
            calls.append("cuda-probed")
            raise AssertionError("CUDA availability must not be queried")

    fake_torch = SimpleNamespace(cuda=NeverCuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-adversarial")
    monkeypatch.setattr(
        panel, "_require_cuda_scoring_admission",
        lambda: (_ for _ in ()).throw(RuntimeError("genuine child capability required")))
    X = np.zeros((7, 3), dtype=np.float32)
    Z = np.zeros((7, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="genuine child capability"):
        panel.score_panel(
            X, Z, config=panel.PanelV2Config(n_anchors=2),
            provenance={"test": "subscale-cuda-visible"})
    assert calls == []


def test_complete_panel_centroid_device_boundary_requires_live_child_before_torch(
        tmp_path, monkeypatch):
    import experiments.score_complete_panel as scorer

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-adversarial")
    monkeypatch.setattr(
        scorer, "_torch_scoring_device",
        lambda: (_ for _ in ()).throw(RuntimeError("live child required")))
    with pytest.raises(RuntimeError, match="live child required"):
        scorer.frozen_centroids(
            np.zeros((8, 3), dtype=np.float32), (2,), str(tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("helper", ["load_model", "embed_texts", "embed_outer_chunks"])
def test_imported_embedding_gpu_helpers_require_live_child_before_torch(
        monkeypatch, helper):
    import basemap.run_controller as controller
    import experiments.embed_prompted_200k as embedder

    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-adversarial")
    monkeypatch.setattr(
        controller, "require_active_round0005_child_admission",
        lambda: (_ for _ in ()).throw(RuntimeError("live child required")))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: calls.append("cuda-probed"))))
    with pytest.raises(RuntimeError, match="live child required"):
        if helper == "load_model":
            embedder.load_model(device="cuda")
        elif helper == "embed_texts":
            embedder.embed_texts(object(), ["text"])
        else:
            embedder.embed_outer_chunks(
                object(), sample_indices=np.array([], dtype=np.int64),
                out_train="/data/not-created-embeddings",
                receipt_dir="/data/not-created-receipts", text_dir="/data/unused",
                text_shards=[], offsets=[], model_commit="a" * 40,
                compute_dtype="float32")
    assert calls == []


def test_imported_legacy_eval_gpu_boundary_rejects_before_torch_or_cuda(monkeypatch):
    import basemap.eval as evaluator
    import basemap.run_controller as controller

    calls = []
    monkeypatch.setattr(
        controller, "require_active_round0005_child_admission",
        lambda: (_ for _ in ()).throw(RuntimeError("genuine child required")))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        from_numpy=lambda *_args: calls.append("torch-used")))
    with pytest.raises(RuntimeError, match="genuine child required"):
        evaluator._gpu_knn_ids(
            np.zeros((8, 3), dtype=np.float32), np.array([0]), 1,
            device="cuda")
    assert calls == []


def test_forged_fixture_child_capability_cannot_leak_into_cuda_scoring(monkeypatch):
    import basemap.panel_v2 as panel
    import basemap.run_controller as controller

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-adversarial")
    monkeypatch.setattr(controller, "_ACTIVE_CHILD_ADMISSION", {
        "node_id": "scalar_equivalence", "fixture_only": True,
        "child_pid": os.getpid(),
    })
    with pytest.raises(RuntimeError, match="forged|incomplete|genuine"):
        panel.score_panel(
            np.zeros((7, 3), dtype=np.float32),
            np.zeros((7, 2), dtype=np.float32),
            config=panel.PanelV2Config(n_anchors=2),
            provenance={"test": "forged-capability"})


def test_self_acquired_public_lease_cannot_launch_canonical_scorer_child(
        fresh_data_root):
    root = Path(__file__).resolve().parents[1]
    output_root = Path(fresh_data_root) / "direct-child-output"
    output_root.mkdir()
    lease_path = Path(fresh_data_root) / "self-acquired.lease"
    script = root / "experiments" / "score_complete_panel.py"
    bootstrap = (
        "import os,sys; sys.path.insert(0," + repr(str(root)) + "); "
        "from basemap.run_controller import GpuLease; "
        "lease=GpuLease(path=" + repr(str(lease_path)) + ",timeout=0).acquire(); "
        "env=dict(os.environ); env['BASEMAP_GPU_LEASE_FD']=str(lease.fileno()); "
        "env['BASEMAP_GPU_LEASE_TOKEN']=lease.token; "
        "os.execve(sys.executable,[sys.executable," + repr(str(script)) +
        ",'--runs','map=/unused','--testbed','/unused','--no-model','--out-root'," +
        repr(str(output_root)) + "],env)"
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", bootstrap], env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    assert proc.returncode != 0
    assert "requires controller admission identity" in proc.stderr
    assert list(output_root.iterdir()) == []


def test_score_panel_library_rejects_actual_scale_file_and_row_lies_before_work(
        fresh_data_root, monkeypatch):
    import basemap.panel_v2 as panel

    scale_path = os.path.join(fresh_data_root, "library-scale-8m.npy")
    matrix = np.lib.format.open_memmap(
        scale_path, mode="w+", dtype=np.float32, shape=(8_000_000, 1))
    matrix.flush()
    calls = []
    monkeypatch.setattr(
        panel, "align_x_to_z",
        lambda *args, **kwargs: calls.append("scientific-work"))
    cfg = panel.PanelV2Config(n_anchors=1)
    with pytest.raises(RuntimeError, match="requires exact replayable scale admission"):
        panel.score_panel(
            matrix, np.empty((0, 2), dtype=np.float32), config=cfg,
            provenance={"test": "scale-boundary"})
    lied = {
        "performance_gate": "/data/unused", "release_sha": "0" * 40,
        "row_derivation": {"scientific_rows": 0, "dimensions": 1,
                           "embedding_input": {"canonical_path": scale_path}},
        "scale_policy": {},
    }
    with pytest.raises(RuntimeError, match="not the reopened signed embedding input"):
        panel.score_panel(
            matrix, np.empty((0, 2), dtype=np.float32), config=cfg,
            scale_admission=lied, provenance={"test": "row-lie"})
    assert calls == []
    del matrix


def test_gpu_entrypoint_inventory_has_no_unclassified_direct_lane():
    root = Path(__file__).resolve().parents[1]
    gpu_markers = (
        "torch.cuda", "import torch", "import cupy", "import cuml",
        "faiss-gpu", "modal.App", "gpu=",
    )
    executable_markers = (
        "if __name__ == \"__main__\"", "if __name__ == '__main__'",
        "def main(", "@app.local_entrypoint",
    )
    discovered = set()
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("tests/"):
            continue
        source = path.read_text(errors="replace")
        if any(marker in source for marker in gpu_markers) and any(
                marker in source for marker in executable_markers):
            discovered.add(relative)
    assert discovered <= EXECUTABLE_GPU_ENTRYPOINTS
    assert (EXECUTABLE_GPU_ENTRYPOINTS - ADMITTED_GPU_ENTRYPOINTS <=
            set(RETIRED_LAUNCHERS))
    assert "experiments/score_8m_bridge.py" in RETIRED_LAUNCHERS
    assert set(REMOTE_MODULE_PATHS).issubset(RETIRED_LAUNCHERS)
