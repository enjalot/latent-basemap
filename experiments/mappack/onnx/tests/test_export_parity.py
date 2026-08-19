"""Parity gate for the ONNX map-head export (PLAN6 in-browser projection).

Two checks, both CPU-only:

1. torch vs onnxruntime on 10k random unit-norm 384-d rows -> max |d| < 1e-4
   (the fp32 gate; R0026 used 1e-3 over 20k rows on the R0019 head).
2. onnxruntime vs the checkpoint's *sealed* coordinates on 1k real substrate
   rows.  Those rows are training rows, so the sealed coordinates came from the
   very same forward pass; the only slack is CPU-vs-GPU fp32 accumulation.

Run:  CUDA_VISIBLE_DEVICES="" pytest experiments/mappack/onnx/tests -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# The onnx toolchain lives in the scratch venv at
# /data/latent-basemap/envs/mappack-onnx (the repo .venv is deliberately
# untouched) — skip cleanly instead of erroring when run from the main venv.
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.mappack.onnx.export_map_head import (  # noqa: E402
    export_onnx, load_head, onnx_run, parity_sealed, random_unit_rows,
)

CHECKPOINT = Path("/data/latent-basemap/sandbox/2m-knobs/umap-md000-x4-fneg10/model.pt")
COORDS = Path("/data/latent-basemap/sandbox/2m-knobs/umap-md000-x4-fneg10/coordinates.npy")
SUBSTRATE = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
                 "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy")

RANDOM_ROWS = 10_000
SEALED_ROWS = 1_000
TOL_MAX_ABS = 1e-4          # torch vs onnxruntime, fp32
TOL_SEALED_FRAC = 1e-5      # onnx vs sealed, as a fraction of the extent diagonal

requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason=f"missing checkpoint {CHECKPOINT}")
requires_substrate = pytest.mark.skipif(
    not (SUBSTRATE.is_file() and COORDS.is_file()),
    reason="missing substrate/coordinates")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    pumap, meta, _ = load_head(CHECKPOINT)
    out = tmp_path_factory.mktemp("onnx") / "map_head.onnx"
    export_onnx(pumap, out)
    return pumap, meta, out


@requires_checkpoint
def test_random_unit_parity(exported):
    pumap, meta, out = exported
    x = random_unit_rows(RANDOM_ROWS, meta["input_dim"], seed=0)
    torch_xy = pumap.transform(x, batch_size=4096)
    onnx_xy = onnx_run(out, x)
    assert onnx_xy.shape == (RANDOM_ROWS, meta["n_components"])
    max_abs = float(np.abs(torch_xy - onnx_xy).max())
    print(f"\nrandom-unit max |d| = {max_abs:.3e} "
          f"(coord |max| = {np.abs(torch_xy).max():.2f})")
    assert max_abs < TOL_MAX_ABS


@requires_checkpoint
def test_dynamic_batch(exported):
    """The export must accept any batch size and be batch-invariant."""
    _, meta, out = exported
    x = random_unit_rows(1024, meta["input_dim"], seed=7)
    full = onnx_run(out, x, batch=1024)
    for bs in (1, 3, 97, 512):
        chunked = onnx_run(out, x, batch=bs)
        assert chunked.shape == full.shape
        assert float(np.abs(chunked - full).max()) < 1e-6


@requires_checkpoint
@requires_substrate
def test_sealed_coordinate_parity(exported):
    pumap, _, out = exported
    res = parity_sealed(pumap, out, SUBSTRATE, COORDS, SEALED_ROWS, seed=0)
    print("\nsealed parity:", res)
    assert res["onnx_vs_sealed"]["max_frac_of_extent_diagonal"] < TOL_SEALED_FRAC
    assert res["onnx_vs_sealed"]["max_abs_diff"] < 1e-3
    # onnx and torch must agree at least as well as either agrees with the seal
    assert res["onnx_vs_torch"]["max_abs_diff"] < TOL_MAX_ABS


@requires_checkpoint
def test_graph_is_fp32_and_web_safe(exported):
    """onnxruntime-web has no fp16 CPU kernels for this graph; stay fp32, and
    keep the op set to the plain MLP ops every ORT-web build ships."""
    import onnx

    _, _, out = exported
    proto = onnx.load(str(out))
    ops = {n.op_type for n in proto.graph.node}
    assert ops <= {"Gemm", "MatMul", "Add", "Relu"}, ops
    for init in proto.graph.initializer:
        assert init.data_type == onnx.TensorProto.FLOAT, init.name
    inp = proto.graph.input[0]
    dim0 = inp.type.tensor_type.shape.dim[0]
    assert dim0.dim_param, "batch axis must be dynamic"
