#!/usr/bin/env python
"""Blockwise normalize+quantize builder for a SEALED int8 substrate.

Purpose
-------
Phase C2 of the 30M admission-engineering effort. At 30M rows the
``image_map_pipeline.py`` ``_norm(load())`` idiom materialises the whole fp32
matrix (30M x 768 x 4 = 92 GB) and blows past RAM. The trainer's
``HostInt8ArrayDataset`` instead consumes a SEALED int8 substrate — int8 codes +
per-row fp16 scales on disk (exactly how the R0034 150M pool is stored:
``minilm-int8-150m/embeddings.i8`` + ``scales.f16``). This builder produces that
sealed format from ANY fp32/fp16 substrate WITHOUT ever holding the full fp32
matrix: it streams the input memmap in BLOCKS, and only one block of fp32 is
resident at a time.

Sealed on-disk layout (must match the loader EXACTLY)
-----------------------------------------------------
``HostInt8MaterializedArray.from_files`` (round0034_pipeline.py:813-814) reads::

    encoded = np.fromfile(int8_path, dtype=np.int8).reshape(row_count, dimension)
    scales  = np.fromfile(scales_path, dtype="<f2", count=row_count)

so the sealed files are RAW, headerless, C-order:

  * ``<out>/embeddings.i8`` — raw signed int8, row-major, logical shape (N, D).
    Byte size == N * D. No .npy header, no manifest embedded.
  * ``<out>/scales.f16``   — raw little-endian fp16 (``<f2``), shape (N,).
    Byte size == N * 2.

``int8_eligibility.py:303-304`` memmaps the same two files with
``dtype=np.int8, shape=(rows, dimension)`` and ``dtype="<u2", shape=(rows,)``
(uint16 = the raw fp16 bits) — same bytes, confirming a headerless raw layout.
The loader's invariants (round0034_pipeline.py:745-755 /
edge_list_dataset.py:208-216): int8 2-D rows, little-endian fp16 scales, all
finite, all strictly positive.

Quantization scheme (byte-for-byte the trainer's)
-------------------------------------------------
The trainer L2-normalizes the substrate with ``_norm`` BEFORE
``HostInt8ArrayDataset`` quantizes it (see make_qdq_substrate.py and
image_map_pipeline._norm). This builder replicates that exactly:

  1. ``_norm``  — L2 row-normalize, zero-norm rows divided by 1 (guarded).
  2. ``quantize_int8_rows`` — imported from the SAME module the trainer uses, so
     the int8 codes AND fp16 scale bits are byte-identical to the host_int8 path
     (R0262 encoder: scale = fp16(max|row|/127); q = clip(rint(row/scale_fp32),
     -127, 127) int8).

Because the builder normalizes first and seals the EXACT trainer codes/scales,
loading the sealed files and dequantizing (``int8.float()*scale.float()``)
yields the SAME bytes the trainer would see if fed ``_norm(raw)`` directly.

CPU-only. Write-once (refuses to overwrite existing output files).

Usage
-----
Build::

    .venv/bin/python experiments/sandbox/build_int8_substrate.py \
        <input_substrate.npy> <out_dir> [--block 500000] [--limit N]

Self-validate on the first 200K rows of the 2M MiniLM substrate::

    .venv/bin/python experiments/sandbox/build_int8_substrate.py --validate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Import the EXACT encoder the trainer uses so int8 codes + fp16 scales are
# byte-identical to the host_int8 (HostInt8ArrayDataset / HostInt8Materialized)
# path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (  # noqa: E402
    quantize_int8_rows,
)

# Defaults for the --validate self-check (first N rows of the sealed 2M MiniLM
# substrate; only the first VALIDATE_LIMIT rows are processed to keep it fast).
VALIDATE_SRC = Path(
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy"
)
VALIDATE_OUT = Path(
    "/tmp/claude-1000/-home-enjalot-code/"
    "44761e3d-6bf2-4b87-bc1a-0a2230694374/scratchpad/int8-substrate-validate"
)
VALIDATE_LIMIT = 200_000
DEFAULT_BLOCK = 500_000


def _norm(x: np.ndarray) -> np.ndarray:
    """EXACT copy of image_map_pipeline._norm (L2 row-normalize; zero->1)."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def build(src_path: Path, out_dir: Path, block: int, limit: int | None) -> dict:
    """Stream ``src_path`` in blocks -> sealed int8 codes + fp16 scales.

    Only one ``block``-row slice of fp32 is ever resident. Returns the manifest
    dict that was written to ``<out_dir>/manifest.json``.
    """
    i8_path = out_dir / "embeddings.i8"
    scales_path = out_dir / "scales.f16"
    manifest_path = out_dir / "manifest.json"
    for p in (i8_path, scales_path, manifest_path):
        if p.exists():
            raise SystemExit(f"REFUSING to overwrite existing {p} (write-once).")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(src_path, mmap_mode="r")
    if raw.ndim != 2:
        raise SystemExit(f"input must be 2-D, got shape {raw.shape}")
    n_total, d = int(raw.shape[0]), int(raw.shape[1])
    n = n_total if limit is None else min(limit, n_total)
    print(f"source: {src_path}")
    print(f"  shape=({n_total},{d}) dtype={raw.dtype}  processing rows=0..{n}")
    print(f"  block={block}  -> {out_dir}")

    max_abs_quant_err = 0.0     # max |dequant - normalized| over all elements
    scale_min, scale_max = np.inf, -np.inf
    rows_written = 0

    # Append raw C-order bytes block by block. int8 codes are already C-order
    # from quantize_int8_rows; scales are cast to little-endian fp16 (<f2) to
    # match np.fromfile(..., dtype="<f2") on the loader side regardless of host
    # byte order.
    with open(i8_path, "wb") as fi8, open(scales_path, "wb") as fsc:
        for i in range(0, n, block):
            end = min(i + block, n)
            blk = np.asarray(raw[i:end], dtype=np.float32)   # only fp32 held
            xn = _norm(blk)                                  # trainer pre-quant
            enc, sc = quantize_int8_rows(xn)                 # R0262 int8+fp16

            enc = np.ascontiguousarray(enc, dtype=np.int8)
            sc = np.ascontiguousarray(sc.astype("<f2"))
            fi8.write(enc.tobytes(order="C"))
            fsc.write(sc.tobytes(order="C"))
            rows_written += end - i

            # Accounting: dequant EXACTLY as HostInt8ArrayDataset.index_select
            # (int8.float() * scale.float()) and compare to the normalized rows.
            deq = enc.astype(np.float32) * sc.astype(np.float32)[:, None]
            max_abs_quant_err = max(max_abs_quant_err,
                                    float(np.abs(deq - xn).max()))
            scale_min = min(scale_min, float(sc.astype(np.float32).min()))
            scale_max = max(scale_max, float(sc.astype(np.float32).max()))
            print(f"  rows {i:>10}..{end:<10}  "
                  f"i8={i8_path.stat().st_size}B  f16={scales_path.stat().st_size}B")

    # Byte-size self-check against the loader's geometry contract.
    i8_bytes = i8_path.stat().st_size
    sc_bytes = scales_path.stat().st_size
    assert i8_bytes == rows_written * d, (i8_bytes, rows_written * d)
    assert sc_bytes == rows_written * 2, (sc_bytes, rows_written * 2)

    manifest = {
        "name": out_dir.name,
        "source": str(src_path),
        "rows": rows_written,
        "dim": d,
        "dtype": "int8+f16",
        "files": {
            "codes": "embeddings.i8",
            "scales": "scales.f16",
        },
        "layout": (
            "RAW headerless C-order. embeddings.i8 = np.fromfile(dtype=int8)."
            "reshape(rows,dim); scales.f16 = np.fromfile(dtype='<f2', count=rows)."
            " Matches HostInt8MaterializedArray.from_files "
            "(round0034_pipeline.py:813-814) and int8_eligibility.py:303-304."
        ),
        "scheme": (
            "_norm (L2 row-normalize, zero-norm->divide by 1; identical to "
            "image_map_pipeline._norm) THEN R0262 quantize_int8_rows: "
            "scale = fp16(max|row|/127) stored little-endian fp16; "
            "q = clip(rint(row/scale_fp32),-127,127) int8. Dequant on the "
            "trainer side is int8.float()*scale.float(); sealing the exact "
            "trainer codes+scales makes the loaded dequant byte-identical to "
            "feeding the trainer _norm(raw)."
        ),
        "encoder_module": (
            "basemap.pumap.parametric_umap.datasets.edge_list_dataset."
            "quantize_int8_rows"
        ),
        "normalized_before_quant": True,
        "i8_bytes": i8_bytes,
        "scales_bytes": sc_bytes,
        "max_abs_quant_error_vs_normalized": max_abs_quant_err,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "block_rows": block,
        "builder": "experiments/sandbox/build_int8_substrate.py",
    }
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {i8_path} ({i8_bytes} B) + {scales_path} ({sc_bytes} B)")
    print(f"wrote {manifest_path}")
    print(f"max abs quant error vs normalized: {max_abs_quant_err:.6e}")
    print(f"scale range: [{scale_min:.6e}, {scale_max:.6e}]")
    return manifest


def validate(src_path: Path, out_dir: Path, limit: int) -> int:
    """Build the first ``limit`` rows, then assert the sealed-file dequant is
    BITWISE-identical to what the trainer's ``HostInt8ArrayDataset`` would feed
    the model given ``_norm(raw rows)``.
    """
    import torch  # CPU-only; no GPU touched.

    build(src_path, out_dir, block=DEFAULT_BLOCK, limit=limit)

    d = int(np.load(src_path, mmap_mode="r").shape[1])

    # 1) Load the sealed files EXACTLY as HostInt8MaterializedArray.from_files.
    encoded = np.fromfile(out_dir / "embeddings.i8",
                          dtype=np.int8).reshape(limit, d)
    scales = np.fromfile(out_dir / "scales.f16", dtype="<f2", count=limit)
    print(f"\nloaded sealed: encoded{encoded.shape} {encoded.dtype}, "
          f"scales{scales.shape} {scales.dtype}")
    assert encoded.shape == (limit, d)
    assert scales.shape == (limit,)
    assert scales.dtype == np.dtype("<f2")
    assert np.all(np.isfinite(scales)) and np.all(scales.astype(np.float32) > 0)

    # 2) Reference path: feed the trainer loader _norm(raw rows) and let IT
    #    quantize + dequantize (encoded=None -> internal quantize_int8_rows).
    raw = np.load(src_path, mmap_mode="r")
    rows_fp32 = np.asarray(raw[:limit], dtype=np.float32)
    xn = _norm(rows_fp32)
    ref_ds = HostInt8ArrayDataset(xn, device="cpu")

    # Spot-check rows spanning the block boundary + a zero-guard case.
    idx = np.array([0, 1, 7, 12345, 199999,
                    DEFAULT_BLOCK - 1 if limit > DEFAULT_BLOCK else limit - 1],
                   dtype=np.int64)
    idx = np.unique(idx[idx < limit])

    ref = ref_ds.index_select(torch.from_numpy(idx))          # fp32 torch rows
    mine = torch.from_numpy(
        encoded[idx].astype(np.float32) * scales[idx].astype(np.float32)[:, None]
    )

    # Bitwise identity: the sealed dequant == the trainer-loader dequant.
    equal_bytes = bool(torch.equal(ref, mine))
    max_abs_diff = float((ref - mine).abs().max())

    # Also confirm the sealed int8 codes/scales themselves match the loader's
    # internal encode of the same normalized rows (byte-for-byte encoder match).
    ref_i8 = ref_ds._i8.index_select(0, torch.from_numpy(idx)).numpy()
    ref_sc = ref_ds._scales.index_select(0, torch.from_numpy(idx)).numpy()
    codes_equal = bool(np.array_equal(ref_i8, encoded[idx]))
    scales_equal = bool(np.array_equal(
        ref_sc.view(np.uint16), scales[idx].view(np.uint16)))

    # Quant error magnitude vs the normalized fp32 (the physical damage).
    max_quant_err = float(np.abs(mine.numpy() - xn[idx]).max())

    print("\n=== VALIDATION ===")
    print(f"rows checked: {idx.tolist()}")
    print(f"sealed dequant == trainer HostInt8 dequant (bitwise): {equal_bytes}")
    print(f"  max abs diff between the two dequant paths: {max_abs_diff:.3e}")
    print(f"int8 codes byte-equal:  {codes_equal}")
    print(f"fp16 scale bits equal:  {scales_equal}")
    print(f"max abs reconstruction error vs _norm(raw): {max_quant_err:.6e}")

    ok = equal_bytes and codes_equal and scales_equal and max_abs_diff == 0.0
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="input substrate .npy (fp32/fp16)")
    ap.add_argument("out_dir", nargs="?", help="output dir for sealed files")
    ap.add_argument("--block", type=int, default=DEFAULT_BLOCK,
                    help=f"rows per block (default {DEFAULT_BLOCK})")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N rows (default: all)")
    ap.add_argument("--validate", action="store_true",
                    help="run the 200K self-check on the 2M MiniLM substrate")
    args = ap.parse_args()

    if args.validate:
        return validate(VALIDATE_SRC, VALIDATE_OUT, VALIDATE_LIMIT)

    if not args.input or not args.out_dir:
        ap.error("input and out_dir are required (or pass --validate)")
    build(Path(args.input), Path(args.out_dir), args.block, args.limit)
    return 0


# HostInt8ArrayDataset is imported lazily inside validate() to avoid importing
# torch on the build path (CPU-only builds should not need torch at all).
def _lazy_import_dataset():
    global HostInt8ArrayDataset
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        HostInt8ArrayDataset as _H,
    )
    HostInt8ArrayDataset = _H


if "--validate" in sys.argv:
    _lazy_import_dataset()

if __name__ == "__main__":
    raise SystemExit(main())
