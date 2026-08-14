"""Export a trained map head (parametric-UMAP projector) to ONNX for the
in-browser projection path of PLAN6.

Why this is small: `fneg`, the low-dim UMAP kernel, `a`/`b`, `kernel_alpha` and
every other sandbox knob shape the *training loss* only.  Inference is a plain
MLP forward (`ResidualBottleneckMLP`: proj_in -> down -> residual blocks -> up
-> proj_out), so the export is just `torch.onnx.export` of `pumap.model` with a
dynamic batch axis.  The checkpoint's kernel/fneg fields are carried into the
sidecar JSON as provenance, never as computation.

Precedent: round-0026 exported the R0019 h2048 head and passed a 20k-row fp32
parity gate at max |dcoord| = 9.3e-4 <= 1e-3; naive fp16 export did not hold
parity.  This exporter therefore writes **fp32 only**.

Usage
-----
    CUDA_VISIBLE_DEVICES="" python experiments/mappack/onnx/export_map_head.py \
        --checkpoint /data/latent-basemap/sandbox/2m-knobs/umap-md000-x4-fneg10/model.pt \
        --out /data/latent-basemap/mappacks/<map_id>/model/map_head.onnx \
        --map-id <map_id> \
        --coordinates /data/latent-basemap/sandbox/2m-knobs/umap-md000-x4-fneg10/coordinates.npy \
        --substrate  /data/.../substrate.f32.npy

Writes `map_head.onnx` plus a `models.json` sidecar next to it carrying the
sha256s, param count, source-checkpoint provenance, the map frame (extent /
quantization contract from PLAN6) and the parity numbers measured at export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The renderer is the authority on the map frame; import it so a pack's extent
# can never drift from the density PNG it is drawn over.
from experiments.map_renders import BINS, EXTENT_PCT, robust_extent  # noqa: E402

SCHEMA = "mappack-models-v1"
DEFAULT_OPSET = 17          # onnxruntime-web >= 1.14 supports opset 17 everywhere
INPUT_NAME = "embedding"
OUTPUT_NAME = "xy"
ENCODER = {
    "name": "sentence-transformers/all-MiniLM-L6-v2",
    "browser_name": "Xenova/all-MiniLM-L6-v2",
    "dim": 384,
    "pooling": "mean",
    "normalize": "l2",
    "max_seq_length": 256,
}


# --------------------------------------------------------------------------- io

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": f"gsv:{path}", "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


# ---------------------------------------------------------------------- export

def load_head(checkpoint: Path):
    """Load the checkpoint through ParametricUMAP.load (restores kernel/fneg
    provenance) and hand back the inference module + its metadata."""
    import torch
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    pumap = ParametricUMAP.load(str(checkpoint), device="cpu")
    pumap.model.eval()
    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    meta = {
        "architecture": raw.get("architecture", "mlp"),
        "input_dim": int(raw.get("input_dim") or pumap.model.proj_in.in_features),
        "hidden_dim": int(raw["hidden_dim"]),
        "n_layers": int(raw["n_layers"]),
        "n_components": int(raw["n_components"]),
        "params": int(sum(p.numel() for p in pumap.model.parameters())),
    }
    training = {k: raw.get(k) for k in (
        "low_dim_kernel", "a", "b", "kernel_alpha", "n_neighbors",
        "correlation_weight", "pos_ratio", "positive_target_mode",
        "density_weight", "fneg_weight", "fneg_lo", "fneg_hi",
        "use_batchnorm", "use_dropout")}
    return pumap, meta, training


def export_onnx(pumap, out: Path, opset: int = DEFAULT_OPSET) -> Path:
    """fp32, dynamic batch, ONNX opset `opset`."""
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    module = pumap.model.float().eval()
    dummy = torch.zeros(1, module.proj_in.in_features, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            module,
            (dummy,),
            str(out),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,          # TorchScript exporter: stable graph, no onnxscript
        )
    return out


def onnx_run(onnx_path: Path, x: np.ndarray, batch: int = 4096) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = [sess.run(None, {name: np.ascontiguousarray(x[i:i + batch], dtype=np.float32)})[0]
            for i in range(0, len(x), batch)]
    return np.concatenate(outs, axis=0)


def random_unit_rows(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def parity_random(pumap, onnx_path: Path, n: int, dim: int, seed: int = 0) -> dict:
    x = random_unit_rows(n, dim, seed)
    torch_xy = pumap.transform(x, batch_size=4096)
    onnx_xy = onnx_run(onnx_path, x)
    d = np.abs(torch_xy - onnx_xy)
    return {"rows": int(n), "seed": seed,
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "p99_abs_diff": float(np.percentile(d, 99))}


def parity_sealed(pumap, onnx_path: Path, substrate: Path, coords: Path,
                  n: int, seed: int = 0) -> dict:
    """Real substrate rows vs the checkpoint's sealed coordinates.

    The sealed coordinates came from the same forward pass, so agreement is
    limited only by CPU-vs-GPU fp32 accumulation order.  Reported in absolute
    units and relative to the map's own extent diagonal.
    """
    X = np.load(substrate, mmap_mode="r")
    Y = np.load(coords, mmap_mode="r")
    rows = min(len(X), len(Y))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(rows, size=min(n, rows), replace=False))
    # Gather a small sample only: never materialise the memmap (>= 2 GB rule).
    xs = np.asarray(X[idx], dtype=np.float32)
    sealed = np.asarray(Y[idx], dtype=np.float32)
    torch_xy = pumap.transform(xs, batch_size=4096)
    onnx_xy = onnx_run(onnx_path, xs)
    ext = robust_extent(Y)
    diag = float(np.hypot(ext[1] - ext[0], ext[3] - ext[2]))
    out = {"rows": int(len(idx)), "seed": seed, "extent_diagonal": diag}
    for tag, arr in (("onnx_vs_sealed", onnx_xy), ("torch_vs_sealed", torch_xy)):
        d = np.abs(arr - sealed)
        out[tag] = {"max_abs_diff": float(d.max()),
                    "mean_abs_diff": float(d.mean()),
                    "max_frac_of_extent_diagonal": float(d.max() / diag)}
    d = np.abs(onnx_xy - torch_xy)
    out["onnx_vs_torch"] = {"max_abs_diff": float(d.max()),
                            "mean_abs_diff": float(d.mean())}
    return out


# ----------------------------------------------------------------------- frame

def frame_block(coords: Path | None, density_png: Path | None) -> dict:
    """The pack's frame contract (PLAN6): extent, PNG placement, quantization.

    `robust_extent` + `binned_counts` + `render_png` in experiments/map_renders.py
    define the image on disk.  Verified empirically against the 2M density.png:
    the render is 180-degrees rotated from the natural orientation, so
    col = (1 - u) * width and row = v * height with
    u = (x - x0) / (x1 - x0), v = (y - y0) / (y1 - y0).
    """
    frame = {
        "extent_percentiles": list(EXTENT_PCT),
        "extent_pad_frac": 0.02,
        "bins": BINS,
        "normalize": {"u": "(x - x0) / (x1 - x0)", "v": "(y - y0) / (y1 - y0)"},
        "png_placement": {"col_px": "(1 - u) * width", "row_px": "v * height",
                          "x_flipped": True, "y_down": True,
                          "note": "matches experiments/map_renders.render_png"},
        "quantization": {"xy": "u16", "levels": 65536,
                         "encode": "round(u * 65535), round(v * 65535)"},
    }
    if coords is not None:
        Y = np.load(coords, mmap_mode="r")
        ext = robust_extent(Y)
        frame["extent"] = [float(v) for v in ext]
        frame["n_rows"] = int(len(Y))
        frame["coordinates"] = {"path": f"gsv:{coords}", "bytes": coords.stat().st_size}
    if density_png is not None and density_png.is_file():
        frame["density_png"] = file_record(density_png)
    return frame


# ------------------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="output .onnx path (models.json is written beside it)")
    ap.add_argument("--map-id", default=None)
    ap.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    ap.add_argument("--coordinates", type=Path, default=None,
                    help="sealed coordinates.npy -> frame extent + sealed parity")
    ap.add_argument("--substrate", type=Path, default=None,
                    help="substrate.f32.npy -> sealed-coordinate parity check")
    ap.add_argument("--density-png", type=Path, default=None)
    ap.add_argument("--random-rows", type=int, default=10_000)
    ap.add_argument("--sealed-rows", type=int, default=1_000)
    ap.add_argument("--tolerance", type=float, default=1e-4,
                    help="max abs torch-vs-onnx difference accepted (fp32 gate)")
    ap.add_argument("--models-json", type=Path, default=None)
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args(argv)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    pumap, meta, training = load_head(args.checkpoint)
    export_onnx(pumap, args.out, opset=args.opset)

    import onnx
    import onnxruntime as ort
    model_proto = onnx.load(str(args.out))
    ops = sorted({n.op_type for n in model_proto.graph.node})

    parity = {}
    if not args.skip_parity:
        parity["random_unit"] = parity_random(pumap, args.out, args.random_rows,
                                              meta["input_dim"])
        if args.substrate and args.coordinates:
            parity["sealed"] = parity_sealed(pumap, args.out, args.substrate,
                                             args.coordinates, args.sealed_rows)
        parity["gate"] = {"threshold_max_abs": args.tolerance,
                          "passed": parity["random_unit"]["max_abs_diff"] < args.tolerance}

    map_id = args.map_id or args.out.parent.parent.name
    doc = {
        "schema": SCHEMA,
        "map_id": map_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "map_head": {
            **file_record(args.out),
            "file": args.out.name,
            "opset": args.opset,
            "ir_version": int(model_proto.ir_version),
            "precision": "fp32",
            "op_types": ops,
            "params": meta["params"],
            "architecture": meta,
            "input": {"name": INPUT_NAME, "shape": ["batch", meta["input_dim"]],
                      "dtype": "float32"},
            "output": {"name": OUTPUT_NAME, "shape": ["batch", meta["n_components"]],
                       "dtype": "float32"},
            "source_checkpoint": file_record(args.checkpoint),
            "training_params": training,
            "training_params_note": ("kernel / fneg / a / b affect the training "
                                     "loss only; inference is a plain MLP forward"),
        },
        "encoder": dict(ENCODER),
        "frame": frame_block(args.coordinates, args.density_png),
        "parity": parity,
        "toolchain": {"torch": torch.__version__, "onnx": onnx.__version__,
                      "onnxruntime": ort.__version__,
                      "exporter": "torchscript (dynamo=False)"},
        "exported_by": "experiments/mappack/onnx/export_map_head.py",
    }
    models_json = args.models_json or (args.out.parent / "models.json")
    models_json.parent.mkdir(parents=True, exist_ok=True)
    models_json.write_text(json.dumps(doc, indent=2) + "\n")

    print(json.dumps({"onnx": str(args.out), "models_json": str(models_json),
                      "params": meta["params"], "parity": parity}, indent=2))
    if parity and not parity["gate"]["passed"]:
        print("PARITY GATE FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
