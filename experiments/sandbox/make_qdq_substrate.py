#!/usr/bin/env python
"""Build the int8 quant->dequant (QDQ) substrate for the int8-tax factorization.

Purpose
-------
Isolate the *pure quantization damage* half of the host-int8 residency FFR tax by
baking the R0262 int8 rounding into a plain fp32 substrate that then trains through
the normal fp16-resident (``x_residency="auto"``) loader. Comparing three arms on
the champion-bs16k recipe:

  (i)   fp16-device control  = auto residency on the RAW 500K substrate
  (ii)  quant->dequant       = auto residency on THIS QDQ substrate  (pure quant)
  (iii) host-int8            = host_int8 residency on the RAW 500K substrate

  (i)->(ii)  = quantization damage        (same auto loader, values differ)
  (ii)->(iii)= loader-path damage         (same quantized values, loader differs)

Scheme match (bit-for-bit)
--------------------------
The trainer (image_map_pipeline.train) L2-normalizes the substrate with ``_norm``
BEFORE ``HostInt8ArrayDataset`` quantizes it. So arm (iii) feeds the model
``dequant(quant(_norm(raw)))``. This script reproduces that EXACTLY:

  1. Load the raw fp32 substrate.
  2. Apply the identical ``_norm`` (L2 row-normalize; zero-norm rows -> divide by 1).
  3. Encode with ``quantize_int8_rows`` (imported from the SAME module the trainer
     uses -> byte-identical int8 codes + fp16 per-row scales; R0262 encoder:
     ``scale = fp16(max|row| / 127)``, ``q = clip(rint(row / scale_fp32), -127, 127)``).
  4. Dequant with the identical arithmetic ``HostInt8ArrayDataset.index_select``
     performs: ``int8.float() * scale.float()`` (both cast to fp32), i.e. in numpy
     ``encoded.astype(f32) * scales.astype(f32)[:,None]``.

The stored QDQ substrate is therefore byte-identical to the values arm (iii)
dequantizes and feeds the model. (The only in-training residual is the pipeline's
mandatory second ``_norm`` applied to this substrate in arm (ii); because
``dequant(quant(unit))`` has norm ~= 1 +/- 5e-5, that renorm is a ~5e-5 per-row
rescale, ~200x smaller than the ~1e-2 quantization error being measured, and the
fp16 device-store it then passes through is itself part of the auto loader path
that (ii)->(iii) is designed to measure.)

CPU-only. Write-once (refuses to overwrite an existing output).
"""
import json
import sys
from pathlib import Path

import numpy as np

# Import the EXACT encoder the trainer uses, so int8 codes + fp16 scales are
# byte-identical to the host_int8 path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (  # noqa: E402
    quantize_int8_rows,
)

RAW = Path("/data/latent-basemap/substrates/minilm-mix-500k/substrate.f32.npy")
OUT_DIR = Path("/data/latent-basemap/substrates/minilm-mix-500k-qdq")
OUT = OUT_DIR / "substrate.f32.npy"
CHUNK = 100_000


def _norm(x: np.ndarray) -> np.ndarray:
    """EXACT copy of image_map_pipeline._norm (L2 row-normalize)."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def main() -> int:
    if OUT.exists():
        print(f"REFUSING to overwrite existing {OUT} (write-once).")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = np.load(RAW, mmap_mode="r")
    n, d = int(raw.shape[0]), int(raw.shape[1])
    print(f"raw substrate: {RAW}  shape=({n},{d}) dtype={raw.dtype}")

    out = np.empty((n, d), dtype=np.float32)
    abs_err_sum = 0.0          # sum of |dequant - normalized| over all elements
    n_elem = 0
    raw_unit_dev = 0.0         # max |1 - ||normalized row|||  (sanity on _norm)
    qdq_norm_min, qdq_norm_max = np.inf, -np.inf

    for i in range(0, n, CHUNK):
        end = min(i + CHUNK, n)
        blk = np.asarray(raw[i:end], dtype=np.float32)
        xn = _norm(blk)                                   # matches trainer pre-quant
        enc, sc = quantize_int8_rows(xn)                  # R0262 int8 + fp16 scale
        # Dequant EXACTLY as HostInt8ArrayDataset.index_select: int8.f32 * fp16.f32
        deq = enc.astype(np.float32) * sc.astype(np.float32)[:, None]
        out[i:end] = deq
        abs_err_sum += float(np.abs(deq - xn).sum())
        n_elem += deq.size
        raw_unit_dev = max(raw_unit_dev,
                           float(np.abs(1.0 - np.linalg.norm(xn, axis=1)).max()))
        deq_norms = np.linalg.norm(deq, axis=1)
        qdq_norm_min = min(qdq_norm_min, float(deq_norms.min()))
        qdq_norm_max = max(qdq_norm_max, float(deq_norms.max()))

    mean_abs_err = abs_err_sum / n_elem

    np.save(OUT, out)

    manifest = {
        "name": "minilm-mix-500k-qdq",
        "source": str(RAW),
        "rows": n,
        "dim": d,
        "dtype": "float32",
        "purpose": "int8-tax factorization arm (ii): pure quant->dequant, "
                   "trained through the normal auto (fp16-resident) loader.",
        "scheme": (
            "dequant(quant(_norm(raw))). _norm = L2 row-normalize (zero->1), "
            "identical to image_map_pipeline._norm. quant = R0262 "
            "quantize_int8_rows (scale=fp16(max|row|/127), "
            "q=clip(rint(row/scale_fp32),-127,127) int8). dequant = "
            "int8.astype(f32)*scale.astype(f32)[:,None], identical to "
            "HostInt8ArrayDataset.index_select. Values are byte-identical to what "
            "the host_int8 arm dequantizes and feeds the model."
        ),
        "encoder_module": "basemap.pumap.parametric_umap.datasets."
                          "edge_list_dataset.quantize_int8_rows",
        "mean_abs_quant_error": mean_abs_err,
        "normalized_row_norm_max_dev_from_1": raw_unit_dev,
        "qdq_row_norm_min": qdq_norm_min,
        "qdq_row_norm_max": qdq_norm_max,
        "owner": "int8-tax factorization @500K (2026-08-25)",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"wrote {OUT}  shape={out.shape} dtype={out.dtype}")
    print(f"mean abs quant error (|dequant - normalized|): {mean_abs_err:.6e}")
    print(f"normalized-row-norm max dev from 1 (pre-quant sanity): {raw_unit_dev:.3e}")
    print(f"qdq row-norm range: [{qdq_norm_min:.6f}, {qdq_norm_max:.6f}]")
    print(f"wrote manifest {OUT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
