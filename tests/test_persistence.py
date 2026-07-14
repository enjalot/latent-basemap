"""
Run-persistence tests (Work Package 2, task 3).

Verifies that every run writes coords.parquet (x, y float32 + ls_index int64)
and a model.pt checkpoint that reloads and reproduces the projection.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_edgelist_smoke import write_blobs_dataset  # noqa: E402


def _run(tmp_path):
    from experiments.experiment_config import load_config
    from experiments.run_experiment import run_single_experiment

    cfg_path = REPO_ROOT / "experiments" / "configs" / "test_edgelist_smoke.yaml"
    memmap_dir, edges_path, labels = write_blobs_dataset(tmp_path)
    cfg = load_config(str(cfg_path))
    cfg.data.memmap_dirs = [memmap_dir]
    cfg.data.precomputed_edges_path = edges_path
    cfg.logging.results_dir = str(tmp_path / "results")
    cfg.train.require_graph_manifest = False
    cfg.train.require_full_budget = False
    run_single_experiment(cfg)

    results_root = Path(cfg.logging.results_dir)
    run_dir = sorted(results_root.glob(f"{cfg.name}_*"))[-1]
    return run_dir, memmap_dir


def test_coords_parquet_schema_and_dtypes(tmp_path):
    import pyarrow.parquet as pq

    run_dir, _ = _run(tmp_path)
    coords_path = run_dir / "coords.parquet"
    assert coords_path.exists()

    schema = pq.read_schema(coords_path)
    fields = {f.name: str(f.type) for f in schema}
    assert fields == {"x": "float", "y": "float", "ls_index": "int64"}, fields

    df = pq.read_table(coords_path).to_pandas()
    assert len(df) == 5000
    # ls_index is a permutation of 0..n-1 (all training rows, precomputed mode).
    assert sorted(df["ls_index"].tolist()) == list(range(5000))
    assert np.isfinite(df[["x", "y"]].to_numpy()).all()


def test_model_pt_reloads_and_reproduces(tmp_path):
    import pyarrow.parquet as pq
    from basemap.pumap.parametric_umap import ParametricUMAP
    from basemap.data_loader import MemmapArrayConcatenator

    run_dir, memmap_dir = _run(tmp_path)
    model_path = run_dir / "model.pt"
    assert model_path.exists()

    # Reload the checkpoint (residual_bottleneck arch must round-trip).
    pumap = ParametricUMAP.load(str(model_path), device="cpu")
    assert pumap.is_fitted
    assert pumap.architecture == "residual_bottleneck"
    assert pumap.positive_target_mode == "binary"

    # Reloaded model reproduces the persisted coordinates exactly.
    X = np.asarray(MemmapArrayConcatenator([memmap_dir], 32)[:], dtype=np.float32)
    Z = pumap.transform(X)

    df = pq.read_table(run_dir / "coords.parquet").to_pandas().sort_values("ls_index")
    Z_saved = df[["x", "y"]].to_numpy()
    np.testing.assert_allclose(Z[df["ls_index"].to_numpy()], Z_saved,
                               rtol=0, atol=1e-4)
