#!/usr/bin/env -S uv run --with numpy --with pillow python
"""make_fixture_pack.py — synthesize a tiny map pack matching the PLAN6
interactive-map data contract, so the viewer can be developed and smoke-tested
before real packs exist.

The real builder is `experiments/mappack/map_pack.py`; this script is only a
fixture generator. It is the *reference implementation of the contract* as the
viewer understands it — if the real builder disagrees, reconcile here first.

Usage:
    ./make_fixture_pack.py --out ~/.agent/basemap-maps/packs
    ./make_fixture_pack.py --out /tmp/packs --n 50000 --zmax 2

Emits (per pack):

    <out>/index.json                     list of packs in this directory
    <out>/<map_id>/manifest.json
    <out>/<map_id>/density/z{z}/{x}_{y}.{corpus}.u32     256*256 u32, row-major, y-down
    <out>/<map_id>/density/z{z}/{x}_{y}.png              combined YlGnBu render
    <out>/<map_id>/points/xy_id.bin                      8B records: u16 x, u16 y, u32 (id<<0 | corpus<<28)
    <out>/<map_id>/points/tile_index.u64                 (4^Z + 1) u64 record offsets, tile-major (y*2^Z + x)
    <out>/<map_id>/points/lod.bin                        9B records: 8B point + u8 min_zoom
    <out>/<map_id>/bins/samples_z{k}.json                {"z_binx_biny": [row_id, ...]}
    <out>/<map_id>/bins/snippets_z{k}.json               {"z_binx_biny": ["text", ...]}
    <out>/<map_id>/text/offsets.u64                      (rows + 1) u64 byte offsets into blob
    <out>/<map_id>/text/blob.utf8                        concatenated utf-8 chunk texts

Chunked-parts fallback (for static hosts that ignore HTTP Range, e.g. the
python http.server behind gsv.local:8800): every file listed in
manifest.chunking.files is ALSO written as `<path>.part{N}` of exactly
`part_bytes` bytes (last part short). The viewer picks range vs chunked mode
per data source at runtime.

Coordinate conventions (the viewer depends on these):
  * world unit square u,v in [0,1]; u = (x - xmin)/(xmax - xmin),
    v = (ymax - y)/(ymax - ymin)   -- i.e. v is Y-DOWN, data-y-up flipped.
  * tile (z, tx, ty) covers u in [tx/2^z, (tx+1)/2^z], v likewise. y-down.
  * bin (bx, by) within a tile is row-major y-down: index = by*256 + bx.
  * global bin coords at level z: gbx = tx*256 + bx, gby = ty*256 + by.
  * quantized point xy: qx = round(u * 65535), qy = round(v * 65535).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

TILE = 256
PART_BYTES_DEFAULT = 4 * 1024 * 1024

# Corpus codes fit in 4 bits (0..15); row ids in the low 28 bits.
CORPORA = [
    {"code": 0, "name": "fineweb", "label": "FineWeb-Edu", "color": "#3987e5"},
    {"code": 1, "name": "redpajama", "label": "RedPajama-V2", "color": "#eb6834"},
    {"code": 2, "name": "pile", "label": "The Pile", "color": "#0ca30c"},
    {"code": 3, "name": "code", "label": "Code", "color": "#a05fd3"},
]

# YlGnBu (ColorBrewer 9-class) — the combined-density look on the research pages.
YLGNBU = [
    (255, 255, 217), (237, 248, 177), (199, 233, 180), (127, 205, 187),
    (65, 182, 196), (29, 145, 192), (34, 94, 168), (37, 52, 148), (8, 29, 88),
]

WORDS = (
    "transformer embedding gradient corpus manifold neighbor entropy kernel "
    "tokenizer inference latent projection cluster density retrieval attention "
    "scaling parametric autoencoder sparse feature semantic geometry topology "
    "wikipedia reddit arxiv github recipe legal medicine finance history poem"
).split()


# ---------------------------------------------------------------------------
# synthetic point cloud
# ---------------------------------------------------------------------------

def synth_points(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # A handful of blobs with corpus-correlated placement, plus a diffuse field,
    # so corpus toggles and dominant-corpus mode both look like something.
    blobs = [
        # (cx, cy, sx, sy, share, corpus mix)
        (-3.2, 1.8, 0.55, 0.42, 0.22, [0.75, 0.12, 0.10, 0.03]),
        (1.9, 2.6, 0.70, 0.55, 0.20, [0.10, 0.72, 0.15, 0.03]),
        (0.4, -2.1, 0.48, 0.90, 0.18, [0.12, 0.18, 0.66, 0.04]),
        (3.4, -1.4, 0.35, 0.35, 0.12, [0.05, 0.06, 0.09, 0.80]),
        (-1.4, -0.3, 1.10, 0.80, 0.16, [0.30, 0.30, 0.30, 0.10]),
    ]
    xs, ys, cs = [], [], []
    used = 0
    for i, (cx, cy, sx, sy, share, mix) in enumerate(blobs):
        k = int(n * share)
        used += k
        xs.append(rng.normal(cx, sx, k))
        ys.append(rng.normal(cy, sy, k))
        cs.append(rng.choice(len(CORPORA), size=k, p=np.array(mix) / sum(mix)))
    k = n - used
    if k > 0:  # diffuse background
        xs.append(rng.uniform(-4.5, 4.5, k))
        ys.append(rng.uniform(-3.5, 3.5, k))
        cs.append(rng.choice(len(CORPORA), size=k, p=[0.35, 0.3, 0.25, 0.1]))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    c = np.concatenate(cs).astype(np.uint8)
    return np.stack([x, y], axis=1), c


def synth_text(rng: np.random.Generator, row: int, corpus: int) -> str:
    nw = int(rng.integers(8, 16))
    body = " ".join(WORDS[int(i)] for i in rng.integers(0, len(WORDS), nw))
    return f"[{CORPORA[corpus]['name']}#{row}] {body}."


# ---------------------------------------------------------------------------
# png (avoid a hard Pillow dep at read time; Pillow is used when available)
# ---------------------------------------------------------------------------

def write_png(path: Path, rgba: np.ndarray) -> None:
    try:
        from PIL import Image
        Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)
        return
    except Exception:
        pass
    import zlib
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def ylgnbu_rgba(counts: np.ndarray, norm: float) -> np.ndarray:
    """log1p-normalized YlGnBu render; count==0 is fully transparent."""
    ramp = np.array(YLGNBU, dtype=np.float32)
    t = np.log1p(counts.astype(np.float32)) / max(math.log1p(norm), 1e-6)
    t = np.clip(t, 0.0, 1.0)
    idx = t * (len(YLGNBU) - 1)
    lo = np.floor(idx).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(YLGNBU) - 1)
    f = (idx - lo)[..., None]
    rgb = ramp[lo] * (1 - f) + ramp[hi] * f
    out = np.zeros(counts.shape + (4,), dtype=np.uint8)
    out[..., :3] = rgb.astype(np.uint8)
    out[..., 3] = np.where(counts > 0, 255, 0)
    return out


# ---------------------------------------------------------------------------
# chunked parts
# ---------------------------------------------------------------------------

def write_parts(path: Path, part_bytes: int) -> int:
    data = path.read_bytes()
    n = max(1, math.ceil(len(data) / part_bytes))
    for i in range(n):
        (path.parent / f"{path.name}.part{i}").write_bytes(
            data[i * part_bytes:(i + 1) * part_bytes])
    return n


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------

def build_pack(out_root: Path, map_id: str, title: str, n: int, zmax: int,
               seed: int, part_bytes: int, lod_target: int) -> dict:
    pack = out_root / map_id
    for sub in ("density", "points", "bins", "text"):
        (pack / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    xy, corpus = synth_points(n, seed)
    n = xy.shape[0]

    xmin, ymin = float(xy[:, 0].min()), float(xy[:, 1].min())
    xmax, ymax = float(xy[:, 0].max()), float(xy[:, 1].max())
    # pad 1% so nothing sits exactly on the boundary
    px, py = (xmax - xmin) * 0.01, (ymax - ymin) * 0.01
    xmin, xmax, ymin, ymax = xmin - px, xmax + px, ymin - py, ymax + py

    u = (xy[:, 0] - xmin) / (xmax - xmin)
    v = (ymax - xy[:, 1]) / (ymax - ymin)          # Y-DOWN
    u = np.clip(u, 0.0, 1.0 - 1e-9)
    v = np.clip(v, 0.0, 1.0 - 1e-9)

    qx = np.round(u * 65535).astype(np.uint16)
    qy = np.round(v * 65535).astype(np.uint16)

    # ---- the one sort: tile order at the finest zoom -----------------------
    ntx = 1 << zmax
    tx = np.minimum((u * ntx).astype(np.int64), ntx - 1)
    ty = np.minimum((v * ntx).astype(np.int64), ntx - 1)
    tile_id = ty * ntx + tx
    order = np.argsort(tile_id, kind="stable")

    qx, qy, corpus, tile_id = qx[order], qy[order], corpus[order], tile_id[order]
    row_id = order.astype(np.uint32)               # original row -> text sidecar row

    # ---- text sidecar (per substrate in real packs; per pack here) ---------
    corpus_by_row = np.empty(n, dtype=np.uint8)
    corpus_by_row[row_id] = corpus
    texts = [synth_text(rng, i, int(corpus_by_row[i])) for i in range(n)]

    blob = bytearray()
    offsets = np.zeros(n + 1, dtype=np.uint64)
    for i, t in enumerate(texts):
        offsets[i] = len(blob)
        blob += t.encode("utf-8")
    offsets[n] = len(blob)
    (pack / "text" / "blob.utf8").write_bytes(bytes(blob))
    (pack / "text" / "offsets.u64").write_bytes(offsets.tobytes())

    # ---- deep points: xy_id.bin + tile_index.u64 --------------------------
    packed = (row_id & 0x0FFFFFFF) | (corpus.astype(np.uint32) << 28)
    rec = np.zeros(n, dtype=np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<u4")]))
    rec["x"], rec["y"], rec["p"] = qx, qy, packed
    (pack / "points" / "xy_id.bin").write_bytes(rec.tobytes())

    ntiles = ntx * ntx
    counts_per_tile = np.bincount(tile_id, minlength=ntiles)
    tile_index = np.zeros(ntiles + 1, dtype=np.uint64)
    tile_index[1:] = np.cumsum(counts_per_tile)
    (pack / "points" / "tile_index.u64").write_bytes(tile_index.tobytes())

    # ---- LOD points: density-stratified, min-zoom tagged ------------------
    # A point's min_zoom is the coarsest level at which it survives a per-bin
    # quota. Cheap approximation: reservoir per bin per level.
    min_zoom = np.full(n, 255, dtype=np.uint8)
    remaining = np.arange(n)
    quota_at = {}
    for z in range(zmax + 1):
        nb = (1 << z) * TILE
        bx = np.minimum((u[order][remaining] * nb).astype(np.int64), nb - 1)
        by = np.minimum((v[order][remaining] * nb).astype(np.int64), nb - 1)
        bid = by * nb + bx
        # keep the first `quota` points per bin at this level
        quota = 1 if z < zmax else 4
        o2 = np.argsort(bid, kind="stable")
        b_sorted = bid[o2]
        first = np.ones(len(b_sorted), dtype=bool)
        first[1:] = b_sorted[1:] != b_sorted[:-1]
        rank = np.arange(len(b_sorted)) - np.maximum.accumulate(
            np.where(first, np.arange(len(b_sorted)), 0))
        keep_mask = np.zeros(len(remaining), dtype=bool)
        keep_mask[o2[rank < quota]] = True
        chosen = remaining[keep_mask]
        min_zoom[chosen] = z
        quota_at[z] = int(len(chosen))
        remaining = remaining[~keep_mask]
        if len(remaining) == 0:
            break
    # everything still unassigned only ever shows at the deepest LOD level
    min_zoom[min_zoom == 255] = zmax

    sel = np.arange(n)
    if lod_target and n > lod_target:
        # keep all coarse points, subsample the deepest band
        deep = sel[min_zoom == zmax]
        coarse = sel[min_zoom < zmax]
        keep_deep = rng.choice(deep, size=max(0, lod_target - len(coarse)),
                               replace=False) if len(deep) else deep
        sel = np.sort(np.concatenate([coarse, keep_deep]))

    lod_order = np.lexsort((tile_id[sel], min_zoom[sel]))   # min-zoom, then tile
    sel = sel[lod_order]
    lod = np.zeros(len(sel), dtype=np.dtype(
        [("x", "<u2"), ("y", "<u2"), ("p", "<u4"), ("z", "u1")]))
    lod["x"], lod["y"] = qx[sel], qy[sel]
    lod["p"], lod["z"] = packed[sel], min_zoom[sel]
    (pack / "points" / "lod.bin").write_bytes(lod.tobytes())

    lod_zoom_offsets = []
    mz = min_zoom[sel]
    for z in range(zmax + 1):
        lod_zoom_offsets.append(int(np.searchsorted(mz, z, side="left")))
    lod_zoom_offsets.append(int(len(sel)))

    # ---- density pyramid + bin samples/snippets ---------------------------
    levels = []
    for z in range(zmax + 1):
        nt = 1 << z
        nb = nt * TILE
        bx = np.minimum((u[order] * nb).astype(np.int64), nb - 1)
        by = np.minimum((v[order] * nb).astype(np.int64), nb - 1)
        gbin = by * nb + bx

        level_dir = pack / "density" / f"z{z}"
        level_dir.mkdir(parents=True, exist_ok=True)

        # per-corpus count planes per tile
        tile_list = []
        # global max for a stable ramp across tiles at this level
        total_counts = np.bincount(gbin, minlength=nb * nb)
        norm = float(total_counts.max())

        samples: dict[str, list[int]] = {}
        snippets: dict[str, list[str]] = {}

        for t_y in range(nt):
            for t_x in range(nt):
                in_tile = (bx // TILE == t_x) & (by // TILE == t_y)
                if not in_tile.any():
                    continue
                lb_x = (bx[in_tile] - t_x * TILE)
                lb_y = (by[in_tile] - t_y * TILE)
                lidx = lb_y * TILE + lb_x
                cor = corpus[in_tile]
                present = []
                combined = np.zeros(TILE * TILE, dtype=np.uint32)
                for cinfo in CORPORA:
                    code = cinfo["code"]
                    m = cor == code
                    if not m.any():
                        continue
                    plane = np.bincount(lidx[m], minlength=TILE * TILE).astype(np.uint32)
                    (level_dir / f"{t_x}_{t_y}.{code}.u32").write_bytes(plane.tobytes())
                    combined += plane
                    present.append(code)
                write_png(level_dir / f"{t_x}_{t_y}.png",
                          ylgnbu_rgba(combined.reshape(TILE, TILE), norm))
                tile_list.append({"x": t_x, "y": t_y, "corpora": present,
                                  "max": int(combined.max())})

                # bin samples/snippets: up to K rows per non-empty bin.
                # Coarse levels are covered densely (this IS the preview tier);
                # fine levels only keep the busiest bins, and the viewer walks
                # up the pyramid for anything not covered.
                K = 3 if z == 0 else 2
                cap = 20000 if z == 0 else 3000
                nz = np.flatnonzero(combined)
                if len(nz) > cap:
                    nz = nz[np.argsort(-combined[nz])[:cap]]
                idx_in_tile = np.flatnonzero(in_tile)
                order_by_bin = np.argsort(lidx, kind="stable")
                lidx_sorted = lidx[order_by_bin]
                starts = np.searchsorted(lidx_sorted, nz, side="left")
                ends = np.searchsorted(lidx_sorted, nz, side="right")
                for b, s, e in zip(nz, starts, ends):
                    rows = idx_in_tile[order_by_bin[s:min(e, s + K)]]
                    rid = [int(row_id[r]) for r in rows]
                    key = f"{z}_{t_x * TILE + int(b) % TILE}_{t_y * TILE + int(b) // TILE}"
                    samples[key] = rid
                    snippets[key] = [texts[i][:110] for i in rid]

        (pack / "bins" / f"samples_z{z}.json").write_text(
            json.dumps(samples, separators=(",", ":")))
        (pack / "bins" / f"snippets_z{z}.json").write_text(
            json.dumps(snippets, separators=(",", ":")))

        levels.append({
            "z": z, "tiles": nt, "bins": nb, "max_count": int(norm),
            "tile_list": [f'{t["x"]}_{t["y"]}' for t in tile_list],
            "tile_corpora": {f'{t["x"]}_{t["y"]}': t["corpora"] for t in tile_list},
        })

    # ---- chunked parts ----------------------------------------------------
    chunked_files = ["points/xy_id.bin", "points/lod.bin",
                     "points/tile_index.u64", "text/offsets.u64", "text/blob.utf8"]
    chunk_info = {}
    for rel in chunked_files:
        p = pack / rel
        chunk_info[rel] = {"bytes": p.stat().st_size,
                           "parts": write_parts(p, part_bytes)}

    manifest = {
        "schema": "basemap-map-pack-v1",
        "map_id": map_id,
        "title": title,
        "description": "Synthetic fixture pack — not real embeddings.",
        "synthetic": True,
        "N": int(n),
        "extent": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
        "tile_size": TILE,
        "zoom": {"min": 0, "max": zmax},
        "corpora": CORPORA,
        "quantization": {"xy_bits": 16, "xy_max": 65535,
                         "id_bits": 28, "corpus_bits": 4, "y_down": True},
        "density": {
            "pattern": "density/z{z}/{x}_{y}.{corpus}.u32",
            "png": "density/z{z}/{x}_{y}.png",
            "plane_dtype": "u32", "plane_bytes": TILE * TILE * 4,
            "levels": levels,
        },
        "points": {
            "lod": {
                "path": "points/lod.bin", "record_bytes": 9,
                "count": int(len(sel)), "bytes": int(len(sel) * 9),
                "zoom_offsets": lod_zoom_offsets,
            },
            "deep": {
                "path": "points/xy_id.bin", "record_bytes": 8,
                "count": int(n), "bytes": int(n * 8),
                "tile_index": {"path": "points/tile_index.u64", "z": zmax,
                               "entries": ntiles + 1, "unit": "records"},
            },
        },
        "bins": {
            "samples": {"pattern": "bins/samples_z{z}.json",
                        "levels": list(range(zmax + 1))},
            "snippets": {"pattern": "bins/snippets_z{z}.json",
                         "levels": list(range(zmax + 1))},
            "key": "{z}_{global_bin_x}_{global_bin_y}",
        },
        "text": {
            "offsets": "text/offsets.u64", "blob": "text/blob.utf8",
            "rows": int(n), "encoding": "utf-8",
        },
        "chunking": {"part_bytes": part_bytes, "suffix": ".part{n}",
                     "files": chunk_info},
    }
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1))

    total = sum(f.stat().st_size for f in pack.rglob("*") if f.is_file())
    return {"map_id": map_id, "title": title, "path": map_id,
            "N": int(n), "zmax": zmax, "synthetic": True,
            "bytes": total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="packs root directory")
    ap.add_argument("--n", type=int, default=250_000)
    ap.add_argument("--zmax", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--part-bytes", type=int, default=256 * 1024,
                    help=f"chunked-mode part size (real packs: {PART_BYTES_DEFAULT})")
    ap.add_argument("--lod-target", type=int, default=60_000)
    ap.add_argument("--map-id", default="fixture-blobs")
    ap.add_argument("--extra", action="store_true",
                    help="also build a second, smaller pack to exercise the switcher")
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    entries = [build_pack(out, args.map_id, "Fixture — five blobs (synthetic)",
                          args.n, args.zmax, args.seed, args.part_bytes,
                          args.lod_target)]
    if args.extra:
        entries.append(build_pack(out, "fixture-small",
                                  "Fixture — small (synthetic)",
                                  args.n // 8, max(1, args.zmax - 1),
                                  args.seed + 1, args.part_bytes,
                                  args.lod_target // 8))

    index_path = out / "index.json"
    prior = {}
    if index_path.exists():
        try:
            prior = {p["map_id"]: p for p in json.loads(index_path.read_text())["packs"]}
        except Exception:
            prior = {}
    for e in entries:
        prior[e["map_id"]] = e
    index_path.write_text(json.dumps(
        {"schema": "basemap-pack-index-v1",
         "packs": sorted(prior.values(), key=lambda p: p["map_id"])}, indent=1))

    for e in entries:
        print(f"{e['map_id']}: N={e['N']} zmax={e['zmax']} "
              f"{e['bytes'] / 1e6:.1f} MB -> {out / e['path']}")
    print(f"index: {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
