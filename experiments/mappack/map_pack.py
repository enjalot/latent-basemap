#!/usr/bin/env python
"""Build "map packs" — static, range-requestable artifacts for the interactive
map viewer (PLAN6, ``experiments/sandbox/PLAN6-interactive-maps.md``).

A pack is one directory per map under ``/data/latent-basemap/mappacks/<map_id>/``::

    manifest.json                 identity, frame, quantization, inventory
    density/z{z}/{x}_{y}.{c}.u32  per-corpus u32 count planes (256x256, y-down)
    density/z{z}/{x}_{y}.png      combined log1p render (YlGnBu)
    density/z{z}/index.json       which tiles/planes exist at this level
    points/xy_id.bin              8 B/point, tile-sorted (Morton within tile)
    points/tile_index.u64         N_tiles+1 byte offsets into xy_id.bin
    points/lod.bin                9 B/point density-stratified LOD sample
    bins/samples_z0.json          reservoir K=4 row ids per occupied coarse bin
    bins/snippets_z0.json         140-char text snippets for those rows

Text lives in a per-SUBSTRATE sidecar shared by every map trained on it::

    /data/latent-basemap/textsidecar/<substrate_key>/{offsets.u64,blob.utf8,manifest.json}

Everything is CPU-only and streams over read-only memmaps; no array >= 2 GB is
ever materialized.

CLI::

    map_pack.py build   --coords ... --substrate-dir ... --map-id ... [--skip-text]
    map_pack.py sidecar --substrate-dir ...
    map_pack.py validate --pack ... [--full]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Contract constants (see manifest["contract"]; deviations are recorded there)
# --------------------------------------------------------------------------
PACK_FORMAT_VERSION = "1"
TILE_BINS = 256           # bins per tile side
QUANT_LEVELS = 65536      # u16 quantization of the full extent
MAX_ZOOM_CAP = 5          # >= ~1 bin per ~50 points; capped so 100M -> z=5
CORE_RADIUS_PCT = 99.5    # trimmed-core radius percentile
EXTENT_PCT = (0.1, 99.9)  # percentile extent of the core
PAD_FRAC = 0.02
CMAP = "YlGnBu"
SNIPPET_CHARS = 140
BIN_SAMPLE_K = 4
COARSE_ZOOM = 0           # bins/*_z{k}.json level: 256^2 total bins
DEFAULT_SEED = 0
ID_BITS = 28              # packed u32 = corpus<<28 | row_id
CHUNK_ROWS = 2_000_000

PACKS_ROOT = Path("/data/latent-basemap/mappacks")
SIDECAR_ROOT = Path("/data/latent-basemap/textsidecar")
EMB_ROOT = Path("/data/embeddings")
CHUNK_ROOT = Path("/data/chunks")

# corpus embedding-dataset name -> chunk-text dataset dir under /data/chunks.
# The embed pipeline writes <stem>.npy next to <stem>.parquet (latent-data-modal
# embed-tei.py: out = file.replace(".parquet", ".npy")), so shards match by stem
# for the three text corpora; starcoderdata was embedded by
# embed_after_chunk.py, which maps sorted-parquet position i -> data-{i:05d}.npy.
CHUNK_DATASETS = {
    "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2":
        ("fineweb-edu-sample-10BT-chunked-120", "stem"),
    "RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2":
        ("RedPajama-Data-V2-sample-10B-chunked-120", "stem"),
    "pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2":
        ("pile-uncopyrighted-chunked-120", "stem"),
    "starcoderdata-code-chunked-120-all-MiniLM-L6-v2":
        ("starcoderdata-code-chunked-120", "position"),
}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def choose_max_zoom(n: int) -> int:
    """Smallest z whose finest grid has >= n bins, capped at MAX_ZOOM_CAP.

    Reproduces the spec's table: 2M -> 3, 12.5M -> 4, 100M-class -> 5.
    """
    z = 0
    while z < MAX_ZOOM_CAP and (TILE_BINS * (1 << z)) ** 2 < n:
        z += 1
    return z


def morton8(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interleave the low 8 bits of x and y (x in even bit positions)."""
    def spread(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint32) & np.uint32(0xFF)
        v = (v | (v << np.uint32(4))) & np.uint32(0x0F0F)
        v = (v | (v << np.uint32(2))) & np.uint32(0x3333)
        v = (v | (v << np.uint32(1))) & np.uint32(0x5555)
        return v
    return (spread(x) | (spread(y) << np.uint32(1))).astype(np.uint32)


def rank_within(gids: np.ndarray, priority: np.ndarray) -> np.ndarray:
    """Rank of each element inside its group, ordered by ``priority``."""
    order = np.lexsort((priority, gids))
    g = gids[order]
    new = np.empty(len(g), dtype=bool)
    new[0] = True
    np.not_equal(g[1:], g[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    lengths = np.diff(np.append(starts, len(g)))
    ranks_sorted = np.arange(len(g), dtype=np.int64) - np.repeat(starts, lengths)
    out = np.empty(len(g), dtype=np.int64)
    out[order] = ranks_sorted
    return out


def cap_for_budget(counts: np.ndarray, budget: int) -> int:
    """Largest per-bin cap c with sum(min(count, c)) <= budget (>= 1)."""
    total = int(counts.sum())
    if total <= budget:
        return int(counts.max()) if counts.size else 1
    lo, hi = 1, int(counts.max())
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        got = int(np.minimum(counts, mid).sum())
        if got <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# --------------------------------------------------------------------------
# substrate: manifest, corpus codes, shard -> source-file resolution
# --------------------------------------------------------------------------
class Substrate:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        cands = [self.dir / "substrate.json", self.dir / "substrate-graph.json"]
        self.manifest_path = next((p for p in cands if p.is_file()), None)
        if self.manifest_path is None:
            raise FileNotFoundError(f"no substrate.json/substrate-graph.json in {self.dir}")
        self.meta = json.loads(self.manifest_path.read_text())
        self.key = self.meta["capability"]
        self.rows = int(self.meta["rows"])
        self.provenance_path = self.dir / "provenance.npy"
        self.excluded = set((self.meta.get("selection") or {}).get("excluded_shards") or {})
        self._shard_lists: dict[str, list[str]] = {}

    # -- provenance ---------------------------------------------------------
    def provenance(self) -> np.ndarray:
        prov = np.load(self.provenance_path, mmap_mode="r")
        if len(prov) != self.rows:
            raise ValueError(f"provenance rows {len(prov)} != manifest rows {self.rows}")
        return prov

    # -- shard lists --------------------------------------------------------
    def _source_size_manifest(self) -> dict:
        ssm = self.meta.get("source_size_manifest")
        path = None
        if isinstance(ssm, dict) and ssm.get("canonical_path"):
            path = Path(ssm["canonical_path"])
        if path is None or not path.is_file():
            path = self.dir.parent.parent / "source-size-manifest.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text()).get("corpora", {})

    def shard_files(self, corpus: str) -> list[str]:
        """Ordered shard list for a corpus; index i == provenance ``shard`` == i.

        The round's source-size manifest is authoritative (it fixes the order and
        the membership); excluded shards are dropped, which is what makes the
        provenance shard index a position in the *non-excluded* list. Verified
        empirically against the substrate vectors (fineweb shard 37 is excluded,
        and provenance shard 37 resolves to file data-00038-of-00099.npy).
        """
        if corpus in self._shard_lists:
            return self._shard_lists[corpus]
        ssm = self._source_size_manifest()
        entry = ssm.get(corpus)
        if entry and entry.get("shard_sizes"):
            files = [k for k in entry["shard_sizes"] if k not in self.excluded]
        else:  # fallback: sorted local glob
            files = sorted(
                f"{corpus}/train/{p.name}"
                for p in (EMB_ROOT / corpus / "train").glob("*.npy")
            )
            files = [f for f in files if f not in self.excluded]
        expected = int((self.meta.get("sources", {}).get(corpus) or {}).get("shards", len(files)))
        if len(files) != expected:
            raise ValueError(f"{corpus}: resolved {len(files)} shards, manifest says {expected}")
        self._shard_lists[corpus] = files
        return files

    # -- corpus code -> name ------------------------------------------------
    def corpus_map(self, prov: np.ndarray | None = None) -> dict[int, str]:
        """Match provenance corpus codes to names by (row count, shard count).

        Never assumes 0..3 = fineweb/redpajama/pile/starcoder; the pairing has
        to be a bijection or we refuse to build.
        """
        prov = self.provenance() if prov is None else prov
        codes = np.asarray(prov["corpus"])
        shards = np.asarray(prov["shard"])
        obs = {}
        for code in np.unique(codes):
            sel = codes == code
            obs[int(code)] = (int(sel.sum()), int(np.unique(shards[sel]).size))
        want = {}
        for name, comp in self.meta["composition"].items():
            n_shards = int(self.meta["sources"][name]["shards"])
            want[name] = (int(comp["rows"]), n_shards)
        mapping: dict[int, str] = {}
        for code, sig in obs.items():
            hits = [n for n, s in want.items() if s == sig]
            if len(hits) != 1:
                raise ValueError(
                    f"corpus code {code} signature {sig} matches {hits} — "
                    "cannot resolve corpus names unambiguously")
            mapping[code] = hits[0]
        if len(set(mapping.values())) != len(mapping) or len(mapping) != len(want):
            raise ValueError(f"corpus code map is not a bijection: {mapping}")
        return mapping


# --------------------------------------------------------------------------
# frame
# --------------------------------------------------------------------------
def robust_extent(pts: np.ndarray) -> list[float]:
    lo, hi = EXTENT_PCT
    x0, x1 = np.percentile(pts[:, 0], [lo, hi])
    y0, y1 = np.percentile(pts[:, 1], [lo, hi])
    pad_x = PAD_FRAC * (x1 - x0) or 1.0
    pad_y = PAD_FRAC * (y1 - y0) or 1.0
    return [float(x0 - pad_x), float(x1 + pad_x), float(y0 - pad_y), float(y1 + pad_y)]


def squarify(extent: list[float]) -> list[float]:
    """Grow the shorter axis about its centre so bins are square in data units."""
    x0, x1, y0, y1 = extent
    w, h = x1 - x0, y1 - y0
    side = max(w, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2]


def compute_frame(coords: np.ndarray, sample_rows: int = 25_000_000) -> dict:
    """Trimmed-core extent, then squared — the build_kernels_page frame logic."""
    n = len(coords)
    if n > sample_rows:
        idx = np.linspace(0, n - 1, sample_rows).astype(np.int64)
        pts = np.asarray(coords[idx], dtype=np.float32)
    else:
        pts = np.asarray(coords, dtype=np.float32)
    radii = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
    core = pts[radii <= np.percentile(radii, CORE_RADIUS_PCT)]
    raw = robust_extent(core)
    return {
        "raw_extent": raw,
        "extent": squarify(raw),
        "core_radius_pct": CORE_RADIUS_PCT,
        "extent_pct": list(EXTENT_PCT),
        "pad_frac": PAD_FRAC,
        "extent_sample_rows": int(len(pts)),
        "squared": True,
    }


def quantize(coords: np.ndarray, extent: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """u16 quantization of the full extent; y measured downward from y1.

    ``floor((v - lo) / (hi - lo) * 65536)`` clamped to [0, 65535] — floor (not
    round) so a bin index at any level is an exact right-shift of the stored
    value, which lets the validator re-derive tiles from the pack alone.
    """
    x0, x1, y0, y1 = extent
    n = len(coords)
    qx = np.empty(n, dtype=np.uint16)
    qy = np.empty(n, dtype=np.uint16)
    for i in range(0, n, CHUNK_ROWS):
        c = np.asarray(coords[i:i + CHUNK_ROWS], dtype=np.float64)
        fx = (c[:, 0] - x0) / (x1 - x0) * QUANT_LEVELS
        fy = (y1 - c[:, 1]) / (y1 - y0) * QUANT_LEVELS
        np.clip(np.floor(fx), 0, QUANT_LEVELS - 1, out=fx)
        np.clip(np.floor(fy), 0, QUANT_LEVELS - 1, out=fy)
        qx[i:i + len(c)] = fx.astype(np.uint16)
        qy[i:i + len(c)] = fy.astype(np.uint16)
    return qx, qy


def bins_at(q: np.ndarray, z: int) -> np.ndarray:
    """Bin index along one axis at zoom z, from the u16 quantized coordinate."""
    shift = 16 - (8 + z)
    if shift < 0:
        raise ValueError("zoom too deep for 16-bit quantization")
    return (q >> np.uint16(shift)).astype(np.int64)


# --------------------------------------------------------------------------
# density pyramid
# --------------------------------------------------------------------------
def render_png(counts: np.ndarray, out_path: Path, peak_log: float) -> None:
    """log1p counts through YlGnBu; empty bins white. counts is (y, x), y-down."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    from matplotlib.image import imsave

    logc = np.log1p(counts.astype(np.float64))
    rgba = colormaps[CMAP](logc / (peak_log or 1.0))
    rgba[counts == 0] = [1.0, 1.0, 1.0, 1.0]
    imsave(out_path, rgba)


def build_density(out_dir: Path, qx, qy, corpus, n_corpora: int, max_zoom: int) -> dict:
    """One counting pass at the finest level, then 2x2 sums up the pyramid."""
    side = TILE_BINS * (1 << max_zoom)
    counts = np.zeros((n_corpora, side, side), dtype=np.uint32)
    ix = bins_at(qx, max_zoom)
    iy = bins_at(qy, max_zoom)
    flat = iy * side + ix
    del ix, iy
    for c in range(n_corpora):
        sel = np.flatnonzero(corpus == c)
        if sel.size == 0:
            continue
        for i in range(0, sel.size, CHUNK_ROWS):
            part = np.bincount(flat[sel[i:i + CHUNK_ROWS]], minlength=side * side)
            counts[c] += part.reshape(side, side).astype(np.uint32)
            del part
        del sel
    del flat

    levels = []
    cur = counts
    for z in range(max_zoom, -1, -1):
        lvl_dir = out_dir / "density" / f"z{z}"
        lvl_dir.mkdir(parents=True, exist_ok=True)
        combined = cur.sum(axis=0, dtype=np.uint64)
        peak_log = float(np.log1p(combined.max()))
        tiles_per_side = 1 << z
        index = {}
        planes = 0
        for ty in range(tiles_per_side):
            for tx in range(tiles_per_side):
                ys = slice(ty * TILE_BINS, (ty + 1) * TILE_BINS)
                xs = slice(tx * TILE_BINS, (tx + 1) * TILE_BINS)
                tile_total = int(combined[ys, xs].sum())
                if tile_total == 0:
                    continue
                present = []
                for c in range(n_corpora):
                    plane = np.ascontiguousarray(cur[c, ys, xs])
                    if not plane.any():
                        continue
                    (lvl_dir / f"{tx}_{ty}.{c}.u32").write_bytes(plane.tobytes())
                    present.append(c)
                    planes += 1
                render_png(np.asarray(combined[ys, xs]), lvl_dir / f"{tx}_{ty}.png", peak_log)
                index[f"{tx}_{ty}"] = {"n": tile_total, "corpora": present}
        (lvl_dir / "index.json").write_text(json.dumps(
            {"z": z, "tiles_per_side": tiles_per_side, "bin_bytes": TILE_BINS * TILE_BINS * 4,
             "png_log_peak": peak_log, "tiles": index}, separators=(",", ":")))
        levels.append({
            "z": z, "tiles_per_side": tiles_per_side, "bins_per_side": TILE_BINS * tiles_per_side,
            "tiles_written": len(index), "planes_written": planes,
            "png_log_peak": peak_log, "total_count": int(combined.sum()),
        })
        if z > 0:
            half = cur.shape[1] // 2
            cur = cur.reshape(n_corpora, half, 2, half, 2).sum(axis=(2, 4), dtype=np.uint64
                                                               ).astype(np.uint32)
    levels.reverse()
    return {"levels": levels, "finest_counts": counts.sum(axis=0, dtype=np.uint64)}


# --------------------------------------------------------------------------
# points, tile index, LOD
# --------------------------------------------------------------------------
POINT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("packed", "<u4")])
LOD_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("packed", "<u4"), ("minz", "u1")])


def sort_key(qx, qy, max_zoom: int) -> tuple[np.ndarray, np.ndarray]:
    """(tile row-major id, Morton within tile) packed into one u64 key."""
    ix, iy = bins_at(qx, max_zoom), bins_at(qy, max_zoom)
    t = 1 << max_zoom
    tile_id = (iy // TILE_BINS) * t + (ix // TILE_BINS)
    morton = morton8((ix % TILE_BINS).astype(np.uint32), (iy % TILE_BINS).astype(np.uint32))
    key = (tile_id.astype(np.uint64) << np.uint64(16)) | morton.astype(np.uint64)
    return tile_id, key


def build_points(out_dir: Path, qx, qy, packed, tile_id, key, max_zoom: int) -> dict:
    pdir = out_dir / "points"
    pdir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(key, kind="stable")
    rec = np.empty(len(order), dtype=POINT_DTYPE)
    rec["x"] = qx[order]
    rec["y"] = qy[order]
    rec["packed"] = packed[order]
    rec.tofile(pdir / "xy_id.bin")
    del rec

    n_tiles = (1 << max_zoom) ** 2
    per_tile = np.bincount(tile_id, minlength=n_tiles).astype(np.uint64)
    offsets = np.zeros(n_tiles + 1, dtype="<u8")
    offsets[1:] = np.cumsum(per_tile) * POINT_DTYPE.itemsize
    offsets.tofile(pdir / "tile_index.u64")
    return {"record_bytes": POINT_DTYPE.itemsize, "n_points": int(len(order)),
            "n_tiles": n_tiles, "order": order, "tile_counts": per_tile}


def build_lod(out_dir: Path, qx, qy, packed, tile_id, max_zoom: int,
              finest_counts: np.ndarray, seed: int) -> dict:
    n = len(qx)
    budget = int(min(n // 4, 2_000_000))
    if budget < 1:
        budget = n
    rng = np.random.default_rng(seed)
    prio = rng.random(n)

    # Stratify on the deepest grid whose occupied-bin count still fits the budget
    # — at the finest level there are usually more occupied bins than the budget,
    # so a per-bin cap there could not shrink the set at all.
    occ = {}
    cur = np.asarray(finest_counts)
    occ[max_zoom] = cur
    for z in range(max_zoom - 1, -1, -1):
        half = cur.shape[0] // 2
        cur = cur.reshape(half, 2, half, 2).sum(axis=(1, 3))
        occ[z] = cur
    strat_z = 0
    for z in range(max_zoom, -1, -1):
        if int((occ[z] > 0).sum()) <= budget:
            strat_z = z
            break
    counts_s = occ[strat_z].reshape(-1)
    cap = cap_for_budget(counts_s[counts_s > 0].astype(np.int64), budget)
    side_s = TILE_BINS << strat_z
    strat_bin = bins_at(qy, strat_z) * side_s + bins_at(qx, strat_z)
    ranks = rank_within(strat_bin, prio)
    sel = np.flatnonzero(ranks < cap)
    if sel.size > budget:  # tiny maps: even cap=1 overshoots; trim deterministically
        sel = sel[np.lexsort((prio[sel], ranks[sel]))[:budget]]
        sel.sort()
    del occ, cur, ranks, strat_bin

    # min-zoom: the coarsest level at which this point is revealed. Level z gets
    # ~budget / 4**(Z-z) points, capped per level-z bin; nesting of the bins makes
    # visibility monotone in z.
    minz = np.full(sel.size, max_zoom, dtype=np.uint8)
    prev_cap = 0
    for z in range(0, max_zoom):
        gid = bins_at(qy[sel], z) * (TILE_BINS << z) + bins_at(qx[sel], z)
        counts_z = np.bincount(gid)
        counts_z = counts_z[counts_z > 0]
        target = max(1, budget // (4 ** (max_zoom - z)))
        cz = max(prev_cap, cap_for_budget(counts_z, target))
        prev_cap = cz
        vis = rank_within(gid, prio[sel]) < cz
        minz[vis & (minz > z)] = z

    order = np.lexsort((tile_id[sel], minz))
    rec = np.empty(sel.size, dtype=LOD_DTYPE)
    s = sel[order]
    rec["x"] = qx[s]
    rec["y"] = qy[s]
    rec["packed"] = packed[s]
    rec["minz"] = minz[order]
    rec.tofile(out_dir / "points" / "lod.bin")
    counts = np.bincount(rec["minz"], minlength=max_zoom + 1)
    starts = np.zeros(max_zoom + 2, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    return {"record_bytes": LOD_DTYPE.itemsize, "n_points": int(sel.size),
            "budget": budget, "stratify_zoom": int(strat_z), "stratify_cap": int(cap),
            "min_zoom_counts": [int(v) for v in counts],
            "min_zoom_offsets": [int(v) * LOD_DTYPE.itemsize for v in starts]}


# --------------------------------------------------------------------------
# coarse-bin samples + snippets
# --------------------------------------------------------------------------
def build_bin_samples(out_dir: Path, qx, qy, z: int, seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed + 1000 + z)
    prio = rng.random(len(qx))
    side = TILE_BINS << z
    gid = bins_at(qy, z) * side + bins_at(qx, z)
    keep = np.flatnonzero(rank_within(gid, prio) < BIN_SAMPLE_K)
    keep = keep[np.lexsort((prio[keep], gid[keep]))]
    samples: dict[str, list[int]] = {}
    for row, g in zip(keep.tolist(), gid[keep].tolist()):
        samples.setdefault(f"{g % side}_{g // side}", []).append(int(row))
    bdir = out_dir / "bins"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / f"samples_z{z}.json").write_text(json.dumps(samples, separators=(",", ":")))
    return samples, {"z": z, "bins_per_side": side, "occupied_bins": len(samples),
                     "k": BIN_SAMPLE_K, "sampled_rows": int(keep.size)}


def build_snippets(out_dir: Path, samples: dict, sidecar: Path | None, z: int) -> dict:
    if sidecar is None or not (sidecar / "blob.utf8").is_file():
        return {"text_available": False, "reason": "no text sidecar for this substrate"}
    offsets = np.memmap(sidecar / "offsets.u64", dtype="<u8", mode="r")
    blob = np.memmap(sidecar / "blob.utf8", dtype=np.uint8, mode="r")
    out: dict[str, list[str]] = {}
    empty = 0
    total = 0
    for kbin, rows in samples.items():
        snips = []
        for r in rows:
            a, b = int(offsets[r]), int(offsets[r + 1])
            total += 1
            if b <= a:
                empty += 1
                snips.append("")
                continue
            snips.append(bytes(blob[a:b]).decode("utf-8", "replace")[:SNIPPET_CHARS])
        out[kbin] = snips
    (out_dir / "bins" / f"snippets_z{z}.json").write_text(json.dumps(out, separators=(",", ":")))
    return {"text_available": True, "sampled_rows": total, "empty_rows": empty,
            "coverage": (total - empty) / total if total else 0.0}


# --------------------------------------------------------------------------
# text sidecar (per substrate)
# --------------------------------------------------------------------------
def _chunk_file_for(sub: Substrate, corpus: str, shard: int) -> Path | None:
    entry = CHUNK_DATASETS.get(corpus)
    if entry is None:
        return None
    ds, mode = entry
    cdir = CHUNK_ROOT / ds / "train"
    if not cdir.is_dir():
        return None
    shard_files = sub.shard_files(corpus)
    if shard >= len(shard_files):
        return None
    if mode == "stem":
        stem = Path(shard_files[shard]).stem
        p = cdir / f"{stem}.parquet"
        return p if p.is_file() else None
    parquets = sorted(cdir.glob("*.parquet"))
    return parquets[shard] if shard < len(parquets) else None


def _single_array(col):
    """Collapse a pyarrow ChunkedArray/Array into one contiguous Array."""
    import pyarrow as pa
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks == 1:
            return col.chunk(0)
        if col.num_chunks == 0:
            return pa.array([], type=col.type)
        return pa.concat_arrays([c for c in col.chunks])
    return col


def _emb_rows(path: Path) -> int:
    """Row count of an embedding shard (raw headerless f32 or real .npy)."""
    try:
        return int(np.load(path, mmap_mode="r").shape[0])
    except Exception:
        return path.stat().st_size // (384 * 4)


def build_sidecar(substrate_dir: str | Path, out_root: Path = SIDECAR_ROOT,
                  force: bool = False, verbose: bool = True) -> dict:
    import pyarrow.parquet as pq
    import pyarrow as pa

    t0 = time.time()
    sub = Substrate(substrate_dir)
    out_dir = out_root / sub.key
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file() and not force:
        return json.loads(manifest_path.read_text())

    prov = sub.provenance()
    codes = np.asarray(prov["corpus"]).astype(np.int64)
    shards = np.asarray(prov["shard"]).astype(np.int64)
    rows = np.asarray(prov["row"]).astype(np.int64)
    cmap = sub.corpus_map(prov)
    n = len(prov)

    order = np.lexsort((rows, shards, codes))
    lengths = np.zeros(n, dtype=np.int64)
    tmp_off = np.zeros(n, dtype=np.int64)

    tmp_path = out_dir / "blob.tmp"
    coverage = {name: {"rows": 0, "text_rows": 0, "shards": 0, "shards_missing": 0,
                       "row_count_checked": 0, "row_count_mismatch": 0}
                for name in cmap.values()}
    missing_shards: list[str] = []
    pos = 0
    with open(tmp_path, "wb") as tmp:
        boundaries = np.flatnonzero(np.diff(codes[order] * 100000 + shards[order])) + 1
        groups = np.split(order, boundaries)
        for gi, grp in enumerate(groups):
            code = int(codes[grp[0]])
            shard = int(shards[grp[0]])
            name = cmap[code]
            coverage[name]["rows"] += len(grp)
            coverage[name]["shards"] += 1
            pq_path = _chunk_file_for(sub, name, shard)
            if pq_path is None:
                coverage[name]["shards_missing"] += 1
                missing_shards.append(f"{name}#{shard}")
                continue
            emb = EMB_ROOT / sub.shard_files(name)[shard]
            pf = pq.ParquetFile(pq_path)
            n_parquet = pf.metadata.num_rows
            if emb.is_file():
                coverage[name]["row_count_checked"] += 1
                if _emb_rows(emb) != n_parquet:
                    coverage[name]["row_count_mismatch"] += 1
            col = _single_array(pq.read_table(pq_path, columns=["chunk_text"]).column(0))
            want = rows[grp]
            take = _single_array(col.take(pa.array(want)))
            bufs = take.buffers()
            off_dtype = np.int64 if pa.types.is_large_string(take.type) else np.int32
            offs = np.frombuffer(bufs[1], dtype=off_dtype, count=len(want) + 1).astype(np.int64)
            data = np.frombuffer(bufs[2], dtype=np.uint8)[offs[0]:offs[-1]]
            lengths[grp] = np.diff(offs)
            tmp_off[grp] = pos + (offs[:-1] - offs[0])
            tmp.write(memoryview(data))
            pos += int(data.size)
            coverage[name]["text_rows"] += len(grp)
            del col, take, data
            if verbose and gi % 25 == 0:
                print(f"  sidecar {gi + 1}/{len(groups)} shards, {pos / 1e9:.2f} GB "
                      f"({time.time() - t0:.0f}s)", flush=True)

    offsets = np.zeros(n + 1, dtype="<u8")
    np.cumsum(lengths, out=offsets[1:].view(np.int64))
    total = int(offsets[-1])
    (out_dir / "offsets.u64").write_bytes(offsets.tobytes())

    # Reorder the temp blob (written in (corpus, shard, row) order) into substrate
    # row order. tmp_off is monotone along ``order``, so each batch's source bytes
    # are one contiguous run; only the destination is scattered.
    blob = np.memmap(out_dir / "blob.utf8", dtype=np.uint8, mode="w+", shape=(max(total, 1),))
    budget = 8 << 20
    cum_len = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths[order], out=cum_len[1:])
    i = 0
    with open(tmp_path, "rb") as tmp:
        while i < n:
            j = int(np.searchsorted(cum_len, cum_len[i] + budget, side="left"))
            j = max(j, i + 1)
            j = min(j, n)
            grp = order[i:j]
            grp = grp[lengths[grp] > 0]
            if grp.size:
                src_a = int(tmp_off[grp[0]])
                nbytes = int(lengths[grp].sum())
                tmp.seek(src_a)
                src = np.frombuffer(tmp.read(nbytes), dtype=np.uint8)
                lens = lengths[grp]
                cum = np.zeros(len(grp), dtype=np.int64)
                np.cumsum(lens[:-1], out=cum[1:])
                dst = np.repeat(offsets[grp].astype(np.int64) - cum, lens) + np.arange(nbytes)
                blob[dst] = src
                del dst, src
            i = j
    blob.flush()
    del blob
    tmp_path.unlink()

    manifest = {
        "pack_format_version": PACK_FORMAT_VERSION,
        "substrate_key": sub.key,
        "substrate_dir": str(sub.dir),
        "rows": n,
        "blob_bytes": total,
        "corpus_codes": {str(k): v for k, v in cmap.items()},
        "coverage": coverage,
        "missing_shards": missing_shards[:50],
        "chunk_datasets": {k: v[0] for k, v in CHUNK_DATASETS.items()},
        "build_wall_s": round(time.time() - t0, 1),
        "files": {p.name: file_entry(p) for p in
                  (out_dir / "offsets.u64", out_dir / "blob.utf8")},
    }
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def build_pack(coords_path: str | Path, substrate_dir: str | Path, map_id: str,
               out_root: Path = PACKS_ROOT, skip_text: bool = False,
               seed: int = DEFAULT_SEED, sidecar_root: Path = SIDECAR_ROOT,
               verbose: bool = True) -> dict:
    t0 = time.time()
    coords_path = Path(coords_path)
    sub = Substrate(substrate_dir)
    out_dir = out_root / map_id
    for stale in ("density", "points", "bins"):  # rebuilds must not leave orphans
        if (out_dir / stale).exists():
            shutil.rmtree(out_dir / stale)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = np.load(coords_path, mmap_mode="r")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{coords_path}: expected (N, 2), got {coords.shape}")
    n = len(coords)
    if n != sub.rows:
        raise ValueError(f"coordinates rows {n} != substrate rows {sub.rows}")
    if n >= (1 << ID_BITS):
        raise ValueError(f"{n} rows exceeds the {ID_BITS}-bit packed id field")

    prov = sub.provenance()
    cmap = sub.corpus_map(prov)
    corpus = np.asarray(prov["corpus"]).astype(np.int64)
    n_corpora = int(corpus.max()) + 1
    if n_corpora > 16:
        raise ValueError("more than 16 corpora does not fit the 4-bit corpus field")

    frame = compute_frame(coords)
    max_zoom = choose_max_zoom(n)
    if verbose:
        print(f"[{map_id}] N={n:,} Z={max_zoom} extent={frame['extent']}", flush=True)

    qx, qy = quantize(coords, frame["extent"])
    packed = ((corpus.astype(np.uint32) << np.uint32(ID_BITS))
              | np.arange(n, dtype=np.uint32)).astype(np.uint32)

    t_density = time.time()
    dens = build_density(out_dir, qx, qy, corpus, n_corpora, max_zoom)
    t_density = time.time() - t_density
    if verbose:
        print(f"[{map_id}] density {t_density:.0f}s", flush=True)

    tile_id, key = sort_key(qx, qy, max_zoom)
    t_points = time.time()
    pts = build_points(out_dir, qx, qy, packed, tile_id, key, max_zoom)
    order = pts.pop("order")
    tile_counts = pts.pop("tile_counts")
    del order, key
    t_points = time.time() - t_points

    t_lod = time.time()
    lod = build_lod(out_dir, qx, qy, packed, tile_id, max_zoom,
                    dens["finest_counts"], seed)
    t_lod = time.time() - t_lod
    if verbose:
        print(f"[{map_id}] points {t_points:.0f}s lod {t_lod:.0f}s", flush=True)

    samples, sample_meta = build_bin_samples(out_dir, qx, qy, COARSE_ZOOM, seed)

    sidecar_dir = sidecar_root / sub.key
    text_meta: dict
    if skip_text:
        text_meta = {"text_available": False, "reason": "--skip-text"}
    else:
        if not (sidecar_dir / "manifest.json").is_file():
            if verbose:
                print(f"[{map_id}] building text sidecar for {sub.key}", flush=True)
            build_sidecar(sub.dir, out_root=sidecar_root, verbose=verbose)
        text_meta = build_snippets(out_dir, samples, sidecar_dir, COARSE_ZOOM)
        sm = sidecar_dir / "manifest.json"
        if sm.is_file():
            text_meta["sidecar"] = {
                "dir": str(sidecar_dir),
                "manifest_sha256": sha256_file(sm),
                "coverage": json.loads(sm.read_text())["coverage"],
            }

    counts_by_corpus = np.bincount(corpus, minlength=n_corpora)
    manifest = {
        "pack_format_version": PACK_FORMAT_VERSION,
        "map_id": map_id,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_points": int(n),
        "substrate_key": sub.key,
        "substrate_dir": str(sub.dir),
        "substrate_manifest": {"path": str(sub.manifest_path),
                               "sha256": sha256_file(sub.manifest_path)},
        "provenance": {"path": str(sub.provenance_path),
                       "sha256": sha256_file(sub.provenance_path)},
        "source_coordinates": {"path": str(coords_path), **file_entry(coords_path)},
        "corpus_codes": {str(k): v for k, v in cmap.items()},
        "corpus_counts": {str(c): int(counts_by_corpus[c]) for c in range(n_corpora)},
        "frame": frame,
        "quantization": {
            "levels": QUANT_LEVELS, "bits": 16,
            "formula": "q = clip(floor((v - lo) / (hi - lo) * 65536), 0, 65535)",
            "x_axis": "left-to-right over extent[0..1]",
            "y_axis": "top-to-bottom, measured downward from extent[3]",
            "bin_from_q": "bin_z = q >> (8 - z)",
        },
        "tiles": {
            "scheme": "square grid over the squared trimmed-core extent",
            "tile_bins": TILE_BINS, "max_zoom": max_zoom,
            "zoom_rule": "smallest z with (256*2^z)^2 >= N, capped at 5",
            "tile_id": "row-major, ty * 2^z + tx, y-down",
            "levels": dens["levels"],
        },
        "points": {
            **pts,
            "sort": "primary key = finest-level tile id (row-major); "
                    "secondary = Morton interleave of the in-tile 8-bit bin coords",
            "record": "x:u16, y:u16, packed:u32 (little-endian, 8 B, no padding)",
            "packed": f"corpus << {ID_BITS} | row_id",
            "nonempty_tiles": int((tile_counts > 0).sum()),
        },
        "lod": {**lod,
                "record": "x:u16, y:u16, packed:u32, min_zoom:u8 (9 B, no padding)",
                "order": "min_zoom, then finest tile id",
                "sampling": "density-stratified: per-finest-bin cap chosen so the "
                            "total hits min(N/4, 2M); min_zoom from nested per-level caps",
                "seed": seed},
        "bins": {**sample_meta, "sampling": "priority-reservoir (K smallest seeded "
                                            "uniform priorities per bin)",
                 "seed": seed, "snippet_chars": SNIPPET_CHARS},
        "text": text_meta,
        "timings_s": {"density": round(t_density, 1), "points": round(t_points, 1),
                      "lod": round(t_lod, 1)},
        "contract": {
            "density_tile": "density/z{z}/{x}_{y}.{corpus}.u32 — 256*256 u32 LE, "
                            "row-major, y-down; all-zero planes are omitted",
            "density_png": "density/z{z}/{x}_{y}.png — log1p(count)/png_log_peak "
                           "through YlGnBu, empty bins white",
            "tile_index": "points/tile_index.u64 — N_tiles+1 u64 LE byte offsets "
                          "into points/xy_id.bin, row-major tile order",
        },
        "deviations": [
            "extent is squared (shorter axis grown about its centre) so that bins "
            "are square in data units; frame.raw_extent keeps the unsquared box",
            "quantization uses floor (not round) so bin indices are exact "
            "right-shifts of the stored u16 — makes the pack self-validating",
            "all-zero corpus planes and empty tiles are omitted; density/z{z}/"
            "index.json lists what exists",
            "bins/terms_z{k}.json (TF-IDF) not built in v1",
        ],
    }
    manifest["files"] = inventory(out_dir)
    manifest["build_wall_s"] = round(time.time() - t0, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if verbose:
        print(f"[{map_id}] done in {manifest['build_wall_s']:.0f}s -> {out_dir}", flush=True)
    return manifest


def inventory(out_dir: Path) -> dict:
    files = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files[str(p.relative_to(out_dir))] = file_entry(p)
    return files


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------
def validate_pack(pack_dir: str | Path, full: bool = False, sample_tiles: int = 32,
                  seed: int = 0) -> dict:
    """Re-check a built pack's invariants. Returns {"ok":bool,"checks":[...]}

    A broken pack must produce a failing report, never a traceback, so anything
    that blows up mid-check is folded into the report as a failure.
    """
    checks: list[dict] = []
    try:
        return _validate_pack(Path(pack_dir), full, sample_tiles, seed, checks)
    except Exception as exc:  # truncated/garbled files land here
        checks.append({"check": "validator_exception", "ok": False, "detail": repr(exc)})
        return {"ok": False, "pack": str(pack_dir), "checks": checks,
                "failed": [c for c in checks if not c["ok"]]}


def _validate_pack(pack: Path, full: bool, sample_tiles: int, seed: int,
                   checks: list[dict]) -> dict:
    man = json.loads((pack / "manifest.json").read_text())

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        return ok

    check("pack_format_version", man.get("pack_format_version") == PACK_FORMAT_VERSION,
          man.get("pack_format_version"))

    # --- inventory: sizes always, sha256 when --full ----------------------
    missing, wrong_size, wrong_hash = [], [], []
    for rel, ent in man["files"].items():
        p = pack / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if p.stat().st_size != ent["bytes"]:
            wrong_size.append(rel)
        elif full and sha256_file(p) != ent["sha256"]:
            wrong_hash.append(rel)
    on_disk = {str(p.relative_to(pack)) for p in pack.rglob("*")
               if p.is_file() and p.name != "manifest.json"}
    extra = sorted(on_disk - set(man["files"]))
    check("inventory_present", not missing, missing[:5])
    check("inventory_sizes", not wrong_size, wrong_size[:5])
    check("inventory_no_extras", not extra, extra[:5])
    if full:
        check("inventory_sha256", not wrong_hash, wrong_hash[:5])
    if missing:  # nothing downstream can be trusted
        return {"ok": False, "pack": str(pack), "checks": checks,
                "failed": [c for c in checks if not c["ok"]]}

    n = man["n_points"]
    max_zoom = man["tiles"]["max_zoom"]
    t = 1 << max_zoom
    side = TILE_BINS * t
    n_corpora = len(man["corpus_codes"])

    # --- points ------------------------------------------------------------
    xy = np.memmap(pack / "points" / "xy_id.bin", dtype=POINT_DTYPE, mode="r")
    off = np.memmap(pack / "points" / "tile_index.u64", dtype="<u8", mode="r")
    check("points_count", len(xy) == n, f"{len(xy)} vs {n}")
    check("tile_index_len", len(off) == t * t + 1, f"{len(off)} vs {t * t + 1}")
    check("tile_index_monotone", bool(np.all(np.diff(off.astype(np.int64)) >= 0)))
    check("tile_index_end", int(off[-1]) == len(xy) * POINT_DTYPE.itemsize,
          f"{int(off[-1])} vs {len(xy) * POINT_DTYPE.itemsize}")
    check("tile_index_start", int(off[0]) == 0)

    ix = (xy["x"] >> np.uint16(8 - max_zoom)).astype(np.int64)
    iy = (xy["y"] >> np.uint16(8 - max_zoom)).astype(np.int64)
    tile_id = (iy // TILE_BINS) * t + (ix // TILE_BINS)
    morton = morton8((ix % TILE_BINS).astype(np.uint32), (iy % TILE_BINS).astype(np.uint32))
    key = (tile_id.astype(np.uint64) << np.uint64(16)) | morton.astype(np.uint64)
    check("points_sorted", bool(np.all(np.diff(key.astype(np.int64)) >= 0)),
          "sort key must be nondecreasing (tile row-major, then Morton in tile)")

    # every tile's declared byte run must contain exactly that tile's points
    starts = (off[:-1] // POINT_DTYPE.itemsize).astype(np.int64)
    ends = (off[1:] // POINT_DTYPE.itemsize).astype(np.int64)
    boundary_ok = True
    rng = np.random.default_rng(seed)
    nonempty = np.flatnonzero(ends > starts)
    probe = nonempty if (full or nonempty.size <= sample_tiles) else rng.choice(
        nonempty, size=sample_tiles, replace=False)
    bad = []
    for ti in np.atleast_1d(probe):
        run = tile_id[starts[ti]:ends[ti]]
        if run.size == 0 or not np.all(run == ti):
            bad.append(int(ti))
            boundary_ok = False
    check("tile_index_delimits_tiles", boundary_ok, bad[:5])
    check("tile_index_covers_all", int((ends - starts).sum()) == n)

    ids = (xy["packed"] & np.uint32((1 << ID_BITS) - 1)).astype(np.int64)
    check("ids_unique", int(np.unique(ids).size) == n)
    check("ids_in_range", bool(ids.max() < n and ids.min() >= 0))
    corp = (xy["packed"] >> np.uint32(ID_BITS)).astype(np.int64)
    got = np.bincount(corp, minlength=n_corpora)
    want = np.array([man["corpus_counts"][str(c)] for c in range(n_corpora)])
    check("corpus_bits_roundtrip", bool(np.array_equal(got[:n_corpora], want)),
          f"{got[:n_corpora].tolist()} vs {want.tolist()}")

    # --- density -----------------------------------------------------------
    per_level_ok = True
    for lvl in man["tiles"]["levels"]:
        z = lvl["z"]
        idx = json.loads((pack / "density" / f"z{z}" / "index.json").read_text())
        total = 0
        planes = 0
        for tk, ent in idx["tiles"].items():
            tx, ty = (int(v) for v in tk.split("_"))
            tsum = 0
            for c in ent["corpora"]:
                raw = (pack / "density" / f"z{z}" / f"{tx}_{ty}.{c}.u32").read_bytes()
                if len(raw) != TILE_BINS * TILE_BINS * 4:
                    per_level_ok = False
                arr = np.frombuffer(raw, dtype="<u4")
                tsum += int(arr.sum())
                planes += 1
            if tsum != ent["n"]:
                per_level_ok = False
            total += tsum
        if total != n:
            per_level_ok = False
            check(f"density_z{z}_total", False, f"{total} vs {n}")
        else:
            check(f"density_z{z}_total", True, f"{total} points, {planes} planes")
    check("density_pyramid_consistent", per_level_ok)

    # cross-check the finest density level against the point file
    z = max_zoom
    idx = json.loads((pack / "density" / f"z{z}" / "index.json").read_text())
    tile_counts = np.bincount(tile_id, minlength=t * t)
    dens_counts = np.zeros(t * t, dtype=np.int64)
    for tk, ent in idx["tiles"].items():
        tx, ty = (int(v) for v in tk.split("_"))
        dens_counts[ty * t + tx] = ent["n"]
    check("density_matches_points", bool(np.array_equal(tile_counts, dens_counts)))

    # a spot bin-level check: recompute one tile's plane from the point file
    if nonempty.size:
        ti = int(np.atleast_1d(probe)[0])
        tx, ty = ti % t, ti // t
        run = slice(starts[ti], ends[ti])
        bx = (ix[run] % TILE_BINS).astype(np.int64)
        by = (iy[run] % TILE_BINS).astype(np.int64)
        cc = corp[run]
        ok = True
        for c in json.loads((pack / "density" / f"z{z}" / "index.json").read_text()
                            )["tiles"][f"{tx}_{ty}"]["corpora"]:
            plane = np.frombuffer((pack / "density" / f"z{z}" / f"{tx}_{ty}.{c}.u32"
                                   ).read_bytes(), dtype="<u4").reshape(TILE_BINS, TILE_BINS)
            m = cc == c
            recon = np.bincount(by[m] * TILE_BINS + bx[m],
                                minlength=TILE_BINS * TILE_BINS).reshape(TILE_BINS, TILE_BINS)
            ok = ok and bool(np.array_equal(plane.astype(np.int64), recon))
        check("density_plane_matches_points", ok, f"tile {tx}_{ty} at z{z}")

    # --- lod ---------------------------------------------------------------
    lod = np.memmap(pack / "points" / "lod.bin", dtype=LOD_DTYPE, mode="r")
    check("lod_count", len(lod) == man["lod"]["n_points"])
    check("lod_minz_sorted", bool(np.all(np.diff(lod["minz"].astype(np.int64)) >= 0)))
    check("lod_minz_range", bool(lod["minz"].max() <= max_zoom))
    lod_ids = (lod["packed"] & np.uint32((1 << ID_BITS) - 1)).astype(np.int64)
    check("lod_ids_unique", int(np.unique(lod_ids).size) == len(lod))
    check("lod_offsets", man["lod"]["min_zoom_offsets"][-1] == len(lod) * LOD_DTYPE.itemsize)

    # --- bins + text -------------------------------------------------------
    z = man["bins"]["z"]
    samples = json.loads((pack / "bins" / f"samples_z{z}.json").read_text())
    over = [k for k, v in samples.items() if len(v) > BIN_SAMPLE_K]
    check("bin_samples_k_cap", not over, over[:5])
    check("bin_samples_occupied", len(samples) == man["bins"]["occupied_bins"])
    # every sampled row must actually live in the bin it is filed under
    pos_of_id = np.empty(n, dtype=np.int64)
    pos_of_id[ids] = np.arange(n, dtype=np.int64)
    flat_rows, want_bx, want_by = [], [], []
    for kbin, rows in samples.items():
        bx, by = (int(v) for v in kbin.split("_"))
        flat_rows.extend(rows)
        want_bx.extend([bx] * len(rows))
        want_by.extend([by] * len(rows))
    if flat_rows:
        p = pos_of_id[np.asarray(flat_rows, dtype=np.int64)]
        gx = (ix[p] >> (max_zoom - z))
        gy = (iy[p] >> (max_zoom - z))
        misfiled = int((gx != np.asarray(want_bx)).sum() + (gy != np.asarray(want_by)).sum())
        check("bin_samples_in_their_bin", misfiled == 0, f"{misfiled} misfiled")
        check("bin_samples_unique", len(set(flat_rows)) == len(flat_rows))

    snip_path = pack / "bins" / f"snippets_z{z}.json"
    if man["text"].get("text_available"):
        snips = json.loads(snip_path.read_text())
        check("snippets_keys_match", set(snips) == set(samples))
        check("snippets_lengths", all(
            len(v) == len(samples[k]) for k, v in snips.items()))
        check("snippets_char_cap", all(
            len(s) <= SNIPPET_CHARS for v in snips.values() for s in v))
    else:
        check("snippets_skipped", True, man["text"].get("reason", ""))

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "pack": str(pack), "checks": checks,
            "failed": [c for c in checks if not c["ok"]]}


def validate_sidecar(sidecar_dir: str | Path, substrate_dir: str | Path | None = None,
                     sample: int = 64, seed: int = 0) -> dict:
    side = Path(sidecar_dir)
    man = json.loads((side / "manifest.json").read_text())
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    offsets = np.memmap(side / "offsets.u64", dtype="<u8", mode="r")
    blob = np.memmap(side / "blob.utf8", dtype=np.uint8, mode="r")
    n = man["rows"]
    check("offsets_len", len(offsets) == n + 1, f"{len(offsets)} vs {n + 1}")
    check("offsets_monotone", bool(np.all(np.diff(offsets.astype(np.int64)) >= 0)))
    check("offsets_end", int(offsets[-1]) == man["blob_bytes"],
          f"{int(offsets[-1])} vs {man['blob_bytes']}")
    check("blob_bytes", len(blob) >= man["blob_bytes"])
    rng = np.random.default_rng(seed)
    rows = rng.choice(n, size=min(sample, n), replace=False)
    bad = 0
    for r in rows:
        a, b = int(offsets[r]), int(offsets[r + 1])
        if b < a:
            bad += 1
            continue
        try:
            bytes(blob[a:b]).decode("utf-8")
        except UnicodeDecodeError:
            bad += 1
    check("sampled_rows_decode", bad == 0, f"{bad}/{len(rows)} failed")
    return {"ok": all(c["ok"] for c in checks), "sidecar": str(side), "checks": checks,
            "failed": [c for c in checks if not c["ok"]]}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build", help="build a map pack")
    b.add_argument("--coords", required=True)
    b.add_argument("--substrate-dir", required=True)
    b.add_argument("--map-id", required=True)
    b.add_argument("--out-root", default=str(PACKS_ROOT))
    b.add_argument("--sidecar-root", default=str(SIDECAR_ROOT))
    b.add_argument("--skip-text", action="store_true")
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)

    s = sp.add_parser("sidecar", help="build a per-substrate text sidecar")
    s.add_argument("--substrate-dir", required=True)
    s.add_argument("--out-root", default=str(SIDECAR_ROOT))
    s.add_argument("--force", action="store_true")

    v = sp.add_parser("validate", help="re-check an existing pack's invariants")
    v.add_argument("--pack", required=True)
    v.add_argument("--full", action="store_true", help="hash every file, probe every tile")

    vs = sp.add_parser("validate-sidecar")
    vs.add_argument("--sidecar", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "build":
        build_pack(a.coords, a.substrate_dir, a.map_id, out_root=Path(a.out_root),
                   skip_text=a.skip_text, seed=a.seed, sidecar_root=Path(a.sidecar_root))
        return 0
    if a.cmd == "sidecar":
        m = build_sidecar(a.substrate_dir, out_root=Path(a.out_root), force=a.force)
        print(json.dumps({k: v for k, v in m.items() if k != "files"}, indent=1))
        return 0
    if a.cmd == "validate":
        res = validate_pack(a.pack, full=a.full)
        for c in res["checks"]:
            print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']} {c['detail']}")
        print("PASS" if res["ok"] else "FAIL")
        return 0 if res["ok"] else 1
    if a.cmd == "validate-sidecar":
        res = validate_sidecar(a.sidecar)
        for c in res["checks"]:
            print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']} {c['detail']}")
        print("PASS" if res["ok"] else "FAIL")
        return 0 if res["ok"] else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
