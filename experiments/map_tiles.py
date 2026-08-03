"""map_tiles.py — binning / sampling / text-resolution library + CLI for the
interactive basemap viewer (Component A).

Produces the frozen data-contract artifacts consumed by the viewer:
  grid-<layer>-<L>.bin        sparse binned counts (magic BIN1)
  samples-<layer>-<sx>_<sy>.json  per-super-tile bin text samples
  points-<layer>.bin          raw xy for small layers (magic PTS1)

Design doc: scratchpad/map-viz-design.md (schema `basemap-viewer-manifest-v1`).

Two map kinds:
  jina-25m   — deduped compact order (24,948,663 rows). coord row -> global row
               via compact-to-global.i64.npy, then R0087 inventory range table
               -> parquet shard + shard row (chunk_text column, strip [CLS]).
  minilm-150m— identity order (coord row == global row). 3 blocks of 50M
               (fineweb / redpajama / pile). block row -> shard via a parquet
               num_rows offset table (parquet metadata is authoritative; the
               fineweb-120 npy shard data-00037 is truncated — never trust npy
               sizes). Offset tables cached as json sidecars under the OUT dir.

No GPU. Everything streams via mmap / chunked bincount. Only reads under /data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Binary format magic numbers (little-endian u32), from the frozen contract.
# ---------------------------------------------------------------------------
GRID_MAGIC = 0x42494E31   # "BIN1"
PTS_MAGIC = 0x50545331    # "PTS1"
ANCHOR_MAGIC = 0x414E4331  # "ANC1"  (written by component C, reader provided here)

# Addendum v3: a level whose whole sparse BIN1 file would exceed this is
# written as split x split spatial tiles instead of one plain file.
TILE_THRESHOLD_BYTES = 2_500_000
TILE_SPLIT = 4

# ---------------------------------------------------------------------------
# Fixed data-source locations (read-only, under /data).
# ---------------------------------------------------------------------------
JINA_INVENTORY = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/"
    "jina-diverse-25m-inventory/jina-diverse-25m-inventory-v1.json"
)
JINA_COMPACT_TO_GLOBAL = (
    "/data/latent-basemap/runs/round-0106/queue-attempt-3/artifacts/"
    "canonical-fuzzy-graph/compact-to-global.i64.npy"
)

# minilm-150m: 3 blocks of 50M, concatenated in this order (design doc:
# [0,50M) fineweb, [50M,100M) redpajama, [100M,150M) pile). chunked-120 parquet.
MINILM_BLOCKS = [
    ("fineweb", 0, 50_000_000,
     "/data/chunks/fineweb-edu-sample-10BT-chunked-120/train"),
    ("redpajama", 50_000_000, 100_000_000,
     "/data/chunks/RedPajama-Data-V2-sample-10B-chunked-120/train"),
    ("pile", 100_000_000, 150_000_000,
     "/data/chunks/pile-uncopyrighted-chunked-120/train"),
]

# Named coord-space subset ranges for minilm-150m (identity order -> coord == global).
MINILM_SUBSET_RANGES = {
    "fineweb": (0, 50_000_000),
    "redpajama": (50_000_000, 100_000_000),
    "pile": (100_000_000, 150_000_000),
}

CLS_PREFIX = "[CLS] "
TEXT_MAX_CHARS = 200


# ===========================================================================
# MapSource
# ===========================================================================
class MapSource:
    """Opens a coordinates dir (chunk-*/coordinates.npy, mmap) as one logical
    f32 [N,2] array. Exposes extent() (cached), iter_chunks(), and row lookup.
    """

    def __init__(self, coords_dir, cache_dir=None):
        self.coords_dir = os.path.realpath(coords_dir)
        self.cache_dir = os.path.realpath(cache_dir) if cache_dir else None
        chunk_dirs = sorted(
            d for d in os.listdir(self.coords_dir)
            if d.startswith("chunk-")
            and os.path.exists(os.path.join(self.coords_dir, d, "coordinates.npy"))
        )
        if not chunk_dirs:
            # allow a flat coordinates.npy directly in coords_dir
            flat = os.path.join(self.coords_dir, "coordinates.npy")
            if os.path.exists(flat):
                self._paths = [flat]
            else:
                raise FileNotFoundError(
                    f"no chunk-*/coordinates.npy under {self.coords_dir}")
        else:
            self._paths = [
                os.path.join(self.coords_dir, d, "coordinates.npy")
                for d in chunk_dirs
            ]
        self._arrays = [np.load(p, mmap_mode="r") for p in self._paths]
        self._sizes = [int(a.shape[0]) for a in self._arrays]
        # cumulative start offset (coord-space global row index) per chunk
        self._starts = np.concatenate([[0], np.cumsum(self._sizes)]).astype(np.int64)
        self.nrows = int(self._starts[-1])
        self._extent = None

    def iter_chunks(self):
        """Yield (chunk_start_row, array_mmap) in coord order."""
        for i, a in enumerate(self._arrays):
            yield int(self._starts[i]), a

    def _cache_key(self):
        h = hashlib.sha1()
        h.update(self.coords_dir.encode())
        for p, n in zip(self._paths, self._sizes):
            h.update(f"{p}:{n}".encode())
        return h.hexdigest()[:16]

    def extent(self):
        """Exact (xmin, ymin, xmax, ymax) over all chunks. Cached json sidecar
        under cache_dir when provided (coords dir is read-only)."""
        if self._extent is not None:
            return self._extent
        cache_path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(
                self.cache_dir, f"extent-{self._cache_key()}.json")
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    self._extent = tuple(json.load(f)["extent"])
                    return self._extent
        xmin = ymin = np.inf
        xmax = ymax = -np.inf
        for _, a in self.iter_chunks():
            # chunked pass over rows to bound memory on huge mmaps
            n = a.shape[0]
            step = 2_000_000
            for s in range(0, n, step):
                blk = np.asarray(a[s:s + step], dtype=np.float64)
                xmin = min(xmin, float(blk[:, 0].min()))
                xmax = max(xmax, float(blk[:, 0].max()))
                ymin = min(ymin, float(blk[:, 1].min()))
                ymax = max(ymax, float(blk[:, 1].max()))
        self._extent = (xmin, ymin, xmax, ymax)
        if cache_path:
            with open(cache_path, "w") as f:
                json.dump({"extent": list(self._extent),
                           "coords_dir": self.coords_dir,
                           "nrows": self.nrows}, f)
        return self._extent

    def read_rows(self, coord_rows):
        """Fetch xy for an array of coord-space rows. Returns f32 [M,2]."""
        coord_rows = np.asarray(coord_rows, dtype=np.int64)
        out = np.empty((coord_rows.shape[0], 2), dtype=np.float32)
        # locate chunk per row
        chunk_of = np.searchsorted(self._starts, coord_rows, "right") - 1
        for ci in np.unique(chunk_of):
            mask = chunk_of == ci
            local = coord_rows[mask] - self._starts[ci]
            out[mask] = np.asarray(self._arrays[ci][local], dtype=np.float32)
        return out


# ===========================================================================
# Cell-index math
# ===========================================================================
def _cell_xy(xy, level, extent):
    """Map f32 [M,2] coords to integer (cx, cy) cell coords at `level`.
    y computed in DATA space; the JS viewer flips for screen."""
    xmin, ymin, xmax, ymax = extent
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0
    u = (xy[:, 0].astype(np.float64) - xmin) / xspan
    v = (xy[:, 1].astype(np.float64) - ymin) / yspan
    cx = np.clip(np.floor(u * level).astype(np.int64), 0, level - 1)
    cy = np.clip(np.floor(v * level).astype(np.int64), 0, level - 1)
    return cx, cy


def _flat_cell(xy, level, extent):
    cx, cy = _cell_xy(xy, level, extent)
    return cy * level + cx  # row-major, cy outer


# ===========================================================================
# Binning
# ===========================================================================
def _row_filter_bounds(row_filter):
    """Return ('slice', lo, hi) or ('ids', sorted_id_array) or None."""
    if row_filter is None:
        return None
    if isinstance(row_filter, slice):
        return ("slice", int(row_filter.start or 0), int(row_filter.stop))
    if isinstance(row_filter, tuple) and len(row_filter) == 2:
        return ("slice", int(row_filter[0]), int(row_filter[1]))
    ids = np.asarray(row_filter, dtype=np.int64)
    return ("ids", np.sort(ids))


def _chunk_selected_xy(chunk_start, arr, rf):
    """Return (xy, coord_rows) for the rows of this chunk that pass the filter."""
    n = arr.shape[0]
    if rf is None:
        xy = np.asarray(arr, dtype=np.float32)
        rows = np.arange(chunk_start, chunk_start + n, dtype=np.int64)
        return xy, rows
    kind = rf[0]
    if kind == "slice":
        lo, hi = rf[1], rf[2]
        a = max(lo, chunk_start)
        b = min(hi, chunk_start + n)
        if b <= a:
            return None, None
        xy = np.asarray(arr[a - chunk_start:b - chunk_start], dtype=np.float32)
        rows = np.arange(a, b, dtype=np.int64)
        return xy, rows
    # ids
    ids = rf[1]
    lo = np.searchsorted(ids, chunk_start, "left")
    hi = np.searchsorted(ids, chunk_start + n, "left")
    if hi <= lo:
        return None, None
    sel = ids[lo:hi]
    local = sel - chunk_start
    xy = np.asarray(arr[local], dtype=np.float32)
    return xy, sel


def bin_counts(source, level, extent, row_filter=None):
    """Sparse binned counts at a single `level` via chunked np.bincount.
    Returns (idx u32[], cnt u32[]) of nonempty cells (idx = cy*level+cx)."""
    rf = _row_filter_bounds(row_filter)
    dense = np.zeros(level * level, dtype=np.int64)
    for cs, arr in source.iter_chunks():
        xy, rows = _chunk_selected_xy(cs, arr, rf)
        if xy is None or xy.shape[0] == 0:
            continue
        flat = _flat_cell(xy, level, extent)
        dense += np.bincount(flat, minlength=level * level)
    idx = np.nonzero(dense)[0].astype(np.uint32)
    cnt = dense[idx].astype(np.uint32)
    return idx, cnt


def bin_all_levels(source, levels, extent, row_filter=None):
    """Compute the FINEST requested level via bincount, then derive coarser
    levels by exact 2x sum-pooling. `levels` must be ascending powers of two
    each 2x the previous. Returns {level: (idx u32[], cnt u32[])}."""
    levels = sorted(levels)
    for a, b in zip(levels, levels[1:]):
        if b != a * 2:
            raise ValueError(f"levels must double: {levels}")
    finest = levels[-1]
    rf = _row_filter_bounds(row_filter)
    dense = np.zeros(finest * finest, dtype=np.int64)
    for cs, arr in source.iter_chunks():
        xy, rows = _chunk_selected_xy(cs, arr, rf)
        if xy is None or xy.shape[0] == 0:
            continue
        flat = _flat_cell(xy, finest, extent)
        dense += np.bincount(flat, minlength=finest * finest)
    out = {}
    cur = dense.reshape(finest, finest)  # [cy, cx]
    cur_level = finest
    for lvl in reversed(levels):
        while cur_level > lvl:
            half = cur_level // 2
            cur = cur.reshape(half, 2, half, 2).sum(axis=(1, 3))
            cur_level = half
        flat = cur.reshape(-1)
        idx = np.nonzero(flat)[0].astype(np.uint32)
        cnt = flat[idx].astype(np.uint32)
        out[lvl] = (idx, cnt)
    return out


# ===========================================================================
# Sampling
# ===========================================================================
def sample_bins(source, level, extent, k=3, row_filter=None, rng_seed=0,
                max_candidates=4_000_000):
    """Two-pass per-cell sampling.

    Pass 1: random candidate subsample (~max_candidates rows or all if fewer),
            lexsort by (cell, hash), keep first k coord-rows per cell.
    Pass 2: targeted sweep for nonempty cells that pass 1 missed (rare cells):
            chunked scan collecting up to k rows for exactly those cells.

    Returns {cell_idx: [coord_row, ...]} with <= k rows per cell.
    """
    rf = _row_filter_bounds(row_filter)
    rng = np.random.default_rng(rng_seed)

    # Determine sampling probability from selected row count.
    n_sel = 0
    for cs, arr in source.iter_chunks():
        xy, rows = _chunk_selected_xy(cs, arr, rf)
        if xy is not None:
            n_sel += xy.shape[0]
    if n_sel == 0:
        return {}
    p = min(1.0, max_candidates / float(n_sel))

    cand_cells = []
    cand_rows = []
    cand_hash = []
    for cs, arr in source.iter_chunks():
        xy, rows = _chunk_selected_xy(cs, arr, rf)
        if xy is None or xy.shape[0] == 0:
            continue
        if p < 1.0:
            mask = rng.random(xy.shape[0]) < p
            if not mask.any():
                continue
            xy = xy[mask]
            rows = rows[mask]
        flat = _flat_cell(xy, level, extent)
        cand_cells.append(flat)
        cand_rows.append(rows)
        cand_hash.append(rng.random(rows.shape[0]))

    result = {}
    if cand_cells:
        cells = np.concatenate(cand_cells)
        rows = np.concatenate(cand_rows)
        hsh = np.concatenate(cand_hash)
        order = np.lexsort((hsh, cells))  # primary: cells, secondary: hash
        cells = cells[order]
        rows = rows[order]
        # first k per contiguous cell group
        boundaries = np.nonzero(np.diff(cells))[0] + 1
        group_starts = np.concatenate([[0], boundaries])
        group_ends = np.concatenate([boundaries, [len(cells)]])
        for gs, ge in zip(group_starts, group_ends):
            cell = int(cells[gs])
            take = rows[gs:min(ge, gs + k)]
            result[cell] = take.tolist()

    # Pass 2: nonempty cells with no candidate.
    idx, _ = bin_counts(source, level, extent, row_filter=row_filter)
    nonempty = set(int(c) for c in idx)
    missing = nonempty - set(result.keys())
    if missing:
        need = {c: k for c in missing}
        for cs, arr in source.iter_chunks():
            if not need:
                break
            xy, rows = _chunk_selected_xy(cs, arr, rf)
            if xy is None or xy.shape[0] == 0:
                continue
            flat = _flat_cell(xy, level, extent)
            # only rows whose cell is still needed
            want_mask = np.isin(flat, np.fromiter(need.keys(), dtype=np.int64))
            if not want_mask.any():
                continue
            for cell, row in zip(flat[want_mask], rows[want_mask]):
                cell = int(cell)
                if need.get(cell, 0) <= 0:
                    continue
                result.setdefault(cell, []).append(int(row))
                need[cell] -= 1
                if need[cell] <= 0:
                    del need[cell]
    return result


# ===========================================================================
# Subset range helpers
# ===========================================================================
_JINA_INVENTORY_CACHE = None
_JINA_C2G_CACHE = None


def _load_jina_inventory():
    global _JINA_INVENTORY_CACHE
    if _JINA_INVENTORY_CACHE is None:
        with open(JINA_INVENTORY) as f:
            inv = json.load(f)
        ranges = inv["selection"]["ranges"]
        _JINA_INVENTORY_CACHE = {
            "ranges": ranges,
            "starts": np.array([r["global_row_start"] for r in ranges],
                               dtype=np.int64),
        }
    return _JINA_INVENTORY_CACHE


def _load_jina_c2g():
    global _JINA_C2G_CACHE
    if _JINA_C2G_CACHE is None:
        _JINA_C2G_CACHE = np.load(JINA_COMPACT_TO_GLOBAL, mmap_mode="r")
    return _JINA_C2G_CACHE


def _lang_key(dataset, language):
    """Human-facing subset key + label for a jina source group."""
    if language:
        code = language.split("_")[0]
        return f"lang-{code}", language
    # English corpus
    if dataset.startswith("fineweb-edu"):
        return "corpus-fineweb", "FineWeb-Edu (English)"
    if dataset.startswith("RedPajama"):
        return "corpus-redpajama", "RedPajama (English)"
    if dataset.startswith("pile"):
        return "corpus-pile", "Pile (English)"
    return f"corpus-{dataset.split('-')[0]}", dataset


def jina_subset_ranges():
    """Programmatically derive per-source/per-language coord-space ranges for
    the jina-25m map. compact-to-global is strictly increasing, so each source
    group's global span maps to a CONTIGUOUS compact/coord slice.

    Returns {key: {"label", "coord_start", "coord_stop", "global_start",
    "global_stop", "dataset", "language"}}. Provenance: R0087 inventory
    selection.ranges + compact-to-global.i64.npy searchsorted.
    """
    inv = _load_jina_inventory()
    c2g = _load_jina_c2g()
    # aggregate global spans per (dataset, language)
    spans = {}
    for r in inv["ranges"]:
        key = (r["dataset"], r["language"])
        gs, ge = r["global_row_start"], r["global_row_stop"]
        if key not in spans:
            spans[key] = [gs, ge]
        else:
            spans[key][0] = min(spans[key][0], gs)
            spans[key][1] = max(spans[key][1], ge)
    out = {}
    for (dataset, language), (gs, ge) in spans.items():
        cs = int(np.searchsorted(c2g, gs, "left"))
        ce = int(np.searchsorted(c2g, ge, "left"))
        key, label = _lang_key(dataset, language)
        out[key] = {
            "label": label,
            "coord_start": cs,
            "coord_stop": ce,
            "global_start": int(gs),
            "global_stop": int(ge),
            "dataset": dataset,
            "language": language,
        }
    return out


def subset_ranges(map_kind):
    """Named coord-space ranges for a map kind, as {key: (lo, hi)} plus labels
    where available."""
    if map_kind == "jina-25m":
        jr = jina_subset_ranges()
        return {k: (v["coord_start"], v["coord_stop"]) for k, v in jr.items()}
    if map_kind == "minilm-150m":
        return dict(MINILM_SUBSET_RANGES)
    raise ValueError(f"unknown map_kind {map_kind}")


# ===========================================================================
# Text resolution
# ===========================================================================
def _clean_text(t):
    if t is None:
        return ""
    if t.startswith(CLS_PREFIX):
        t = t[len(CLS_PREFIX):]
    elif t.startswith("[CLS]"):
        t = t[len("[CLS]"):].lstrip()
    t = " ".join(t.split())
    return t[:TEXT_MAX_CHARS]


def _jina_parquet_path(npy_canonical_path):
    return (npy_canonical_path
            .replace("/data/embeddings/", "/data/chunks/")
            .replace("-jina-v5-nano", "")
            .replace(".npy", ".parquet"))


def _resolve_jina(coord_rows):
    import pyarrow.parquet as pq
    inv = _load_jina_inventory()
    ranges = inv["ranges"]
    starts = inv["starts"]
    c2g = _load_jina_c2g()

    coord_rows = np.asarray(coord_rows, dtype=np.int64)
    globals_ = np.asarray(c2g[coord_rows], dtype=np.int64)
    range_idx = np.searchsorted(starts, globals_, "right") - 1

    # group by parquet path
    by_path = {}  # path -> list of (out_position, shard_row, group_label, global_row)
    for pos in range(len(coord_rows)):
        r = ranges[int(range_idx[pos])]
        g = int(globals_[pos])
        shard_row = r["shard_row_start"] + (g - r["global_row_start"])
        path = _jina_parquet_path(r["shard"]["canonical_path"])
        _, label = _lang_key(r["dataset"], r["language"])
        by_path.setdefault(path, []).append((pos, shard_row, label, g))

    out = [None] * len(coord_rows)
    for path, items in by_path.items():
        col = pq.ParquetFile(path).read(columns=["chunk_text"]).column("chunk_text")
        for pos, shard_row, label, g in items:
            txt = _clean_text(col[shard_row].as_py())
            out[pos] = {"t": txt, "g": label, "r": g}
    return out


# minilm offset tables (parquet num_rows), cached json under OUT dir.
def _minilm_offset_table(block_dir, cache_dir=None):
    """Return dict {"files":[...], "cumrows":[...], "total": int} where cumrows[i]
    is the first global-within-block row of file i (cumrows[-1] == total)."""
    import pyarrow.parquet as pq
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        tag = hashlib.sha1(block_dir.encode()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, f"minilm-offsets-{tag}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)
    files = sorted(f for f in os.listdir(block_dir) if f.endswith(".parquet"))
    cum = [0]
    for f in files:
        nr = pq.ParquetFile(os.path.join(block_dir, f)).metadata.num_rows
        cum.append(cum[-1] + nr)
    table = {"block_dir": block_dir, "files": files, "cumrows": cum,
             "total": cum[-1]}
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(table, f)
    return table


def _resolve_minilm(coord_rows, cache_dir=None):
    import pyarrow.parquet as pq
    coord_rows = np.asarray(coord_rows, dtype=np.int64)
    out = [None] * len(coord_rows)
    # group rows by block
    for name, bstart, bstop, block_dir in MINILM_BLOCKS:
        mask = (coord_rows >= bstart) & (coord_rows < bstop)
        if not mask.any():
            continue
        positions = np.nonzero(mask)[0]
        block_rows = coord_rows[positions] - bstart
        table = _minilm_offset_table(block_dir, cache_dir=cache_dir)
        cum = np.asarray(table["cumrows"], dtype=np.int64)
        files = table["files"]
        file_idx = np.searchsorted(cum, block_rows, "right") - 1
        # group by file
        by_file = {}
        for pos, br, fi in zip(positions, block_rows, file_idx):
            if fi < 0 or fi >= len(files):
                out[pos] = {"t": "", "g": name, "r": int(coord_rows[pos])}
                continue
            shard_row = int(br - cum[fi])
            by_file.setdefault(int(fi), []).append((int(pos), shard_row))
        for fi, items in by_file.items():
            path = os.path.join(block_dir, files[fi])
            col = pq.ParquetFile(path).read(columns=["chunk_text"]).column("chunk_text")
            for pos, shard_row in items:
                txt = _clean_text(col[shard_row].as_py())
                out[pos] = {"t": txt, "g": name, "r": int(coord_rows[pos])}
    return out


def resolve_texts(map_kind, coord_rows, cache_dir=None):
    """Resolve coord-space rows to [{t, g, r}] (text, group label, global row).
    Batches per parquet shard. Strips [CLS]; truncates to 200 chars."""
    coord_rows = list(coord_rows)
    if len(coord_rows) == 0:
        return []
    if map_kind == "jina-25m":
        return _resolve_jina(coord_rows)
    if map_kind == "minilm-150m":
        return _resolve_minilm(coord_rows, cache_dir=cache_dir)
    raise ValueError(f"unknown map_kind {map_kind}")


# ===========================================================================
# Binary / JSON writers + readers
# ===========================================================================
def write_grid(path, level, idx, cnt):
    """grid-<layer>-<L>.bin: 16-byte header (magic, level, ncells, reserved),
    then u32[ncells] cell indices, then u32[ncells] counts. All little-endian."""
    idx = np.ascontiguousarray(np.asarray(idx, dtype="<u4"))
    cnt = np.ascontiguousarray(np.asarray(cnt, dtype="<u4"))
    if idx.shape != cnt.shape:
        raise ValueError("idx/cnt length mismatch")
    ncells = int(idx.shape[0])
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", GRID_MAGIC, int(level), ncells, 0))
        f.write(idx.tobytes())
        f.write(cnt.tobytes())
    return path


def read_grid(path):
    """Round-trip reader for grid-<layer>-<L>.bin -> (level, idx u32[], cnt u32[])."""
    with open(path, "rb") as f:
        magic, level, ncells, _ = struct.unpack("<IIII", f.read(16))
        if magic != GRID_MAGIC:
            raise ValueError(f"bad grid magic {magic:#x} in {path}")
        idx = np.frombuffer(f.read(4 * ncells), dtype="<u4").copy()
        cnt = np.frombuffer(f.read(4 * ncells), dtype="<u4").copy()
    return int(level), idx, cnt


def grid_file_bytes(ncells):
    """Whole-file size of a plain sparse BIN1 grid with `ncells` nonempty cells."""
    return 16 + 8 * int(ncells)


def write_grid_tiled(out_dir, layer, level, idx, cnt, split=TILE_SPLIT):
    """Split one level's sparse bins into split x split spatial tile files.

    Files: grid-<layer>-<L>-<tx>_<ty>.bin — the SAME BIN1 format as write_grid,
    with GLOBAL row-major cell indices for the full LxL grid (the tiling is a
    fetch-granularity split, not a re-indexing). A cell (cx, cy) belongs to
    tile tx = cx // (L/split), ty = cy // (L/split). ALL split*split tiles are
    written (empty tiles are 16-byte headers) so the viewer never 404s on a
    viewport fetch. Returns the list of written paths in (ty, tx) order.
    """
    if level % split != 0:
        raise ValueError(f"level {level} not divisible by split {split}")
    idx = np.ascontiguousarray(np.asarray(idx, dtype="<u4"))
    cnt = np.ascontiguousarray(np.asarray(cnt, dtype="<u4"))
    if idx.shape != cnt.shape:
        raise ValueError("idx/cnt length mismatch")
    span = level // split  # cells per tile edge
    cx = idx % np.uint32(level)
    cy = idx // np.uint32(level)
    tx = cx // np.uint32(span)
    ty = cy // np.uint32(span)
    tile_key = ty * np.uint32(split) + tx
    paths = []
    for t_y in range(split):
        for t_x in range(split):
            sel = tile_key == np.uint32(t_y * split + t_x)
            path = os.path.join(out_dir, f"grid-{layer}-{level}-{t_x}_{t_y}.bin")
            write_grid(path, level, idx[sel], cnt[sel])
            paths.append(path)
    return paths


def write_grid_auto(out_dir, layer, level, idx, cnt, *,
                    threshold=TILE_THRESHOLD_BYTES, split=TILE_SPLIT):
    """Write one level plain or tiled by the addendum-v3 size rule.

    Plain grid-<layer>-<L>.bin when the whole sparse file fits within
    `threshold` bytes; otherwise split x split tile files via write_grid_tiled.
    Returns {"level", "tiled", "split" (only when tiled), "paths"}.
    """
    if grid_file_bytes(len(idx)) > threshold:
        paths = write_grid_tiled(out_dir, layer, level, idx, cnt, split=split)
        return {"level": int(level), "tiled": True, "split": int(split),
                "paths": paths}
    path = write_grid(os.path.join(out_dir, f"grid-{layer}-{level}.bin"),
                      level, idx, cnt)
    return {"level": int(level), "tiled": False, "paths": [path]}


def write_points(path, xy):
    """points-<layer>.bin: header (magic PTS1, npoints), then f32 x,y pairs."""
    xy = np.ascontiguousarray(np.asarray(xy, dtype="<f4"))
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be [N,2]")
    npoints = int(xy.shape[0])
    with open(path, "wb") as f:
        f.write(struct.pack("<II", PTS_MAGIC, npoints))
        f.write(xy.tobytes())
    return path


def read_points(path):
    with open(path, "rb") as f:
        magic, npoints = struct.unpack("<II", f.read(8))
        if magic != PTS_MAGIC:
            raise ValueError(f"bad points magic {magic:#x} in {path}")
        xy = np.frombuffer(f.read(8 * npoints), dtype="<f4").reshape(-1, 2).copy()
    return xy


def read_anchors(path):
    """Reader for metrics-anchors.bin (written by component C): magic ANC1,
    u32 count, then f32 triples (x, y, score)."""
    with open(path, "rb") as f:
        magic, count = struct.unpack("<II", f.read(8))
        if magic != ANCHOR_MAGIC:
            raise ValueError(f"bad anchor magic {magic:#x} in {path}")
        arr = np.frombuffer(f.read(12 * count), dtype="<f4").reshape(-1, 3).copy()
    return arr


def write_samples(out_dir, layer, samples_by_cell, sample_level, super_tile,
                  map_kind, cache_dir=None):
    """Resolve sampled cells to text and write per-super-tile JSON files.

    samples_by_cell: {cell_idx: [coord_row, ...]} at `sample_level`.
    Writes samples-<layer>-<sx>_<sy>.json where the sample_level grid is split
    into super_tile x super_tile blocks. Returns list of written paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not samples_by_cell:
        return []
    span = sample_level // super_tile  # cells per super-tile edge

    # Resolve all coord rows in one batched pass (grouped by parquet inside).
    flat_rows = []
    row_owner = []  # (cell, position-in-cell)
    for cell, rows in samples_by_cell.items():
        for row in rows:
            flat_rows.append(row)
            row_owner.append(cell)
    resolved = resolve_texts(map_kind, flat_rows, cache_dir=cache_dir)

    # assemble per-cell sample lists
    cell_samples = {}
    for cell, rec in zip(row_owner, resolved):
        cell_samples.setdefault(cell, []).append(rec)

    # bucket cells into super-tiles
    tiles = {}  # (sx, sy) -> {"cells": {str(cell): [rec,...]}}
    for cell, recs in cell_samples.items():
        cx = cell % sample_level
        cy = cell // sample_level
        sx = cx // span
        sy = cy // span
        tiles.setdefault((sx, sy), {"cells": {}})["cells"][str(cell)] = recs

    written = []
    for (sx, sy), payload in tiles.items():
        path = os.path.join(out_dir, f"samples-{layer}-{sx}_{sy}.json")
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        written.append(path)
    return written


# ===========================================================================
# CLI
# ===========================================================================
def _build_cli(args):
    coords = args.coords
    out = args.out
    data_dir = os.path.join(out, "data")
    os.makedirs(data_dir, exist_ok=True)
    cache_dir = os.path.join(out, "_cache")
    source = MapSource(coords, cache_dir=cache_dir)
    extent = source.extent()
    levels = [int(x) for x in args.levels.split(",")]
    print(f"map: {coords}", file=sys.stderr)
    print(f"rows: {source.nrows}  extent: {extent}", file=sys.stderr)

    # base "all" layer grids
    grids = bin_all_levels(source, levels, extent)
    for lvl, (idx, cnt) in grids.items():
        p = write_grid(os.path.join(data_dir, f"grid-all-{lvl}.bin"), lvl, idx, cnt)
        print(f"wrote {p}  ncells={len(idx)}", file=sys.stderr)

    # samples for base layer at sample_level
    sl = args.sample_level
    samples = sample_bins(source, sl, extent, k=3, rng_seed=args.seed)
    written = write_samples(data_dir, "all", samples, sl, args.super_tile,
                            args.map_kind, cache_dir=cache_dir)
    print(f"wrote {len(written)} sample supertiles", file=sys.stderr)

    # explicit subset layers
    for spec in (args.layer or []):
        key, _, rangespec = spec.partition(":")
        named = subset_ranges(args.map_kind)
        if rangespec in named:
            lo, hi = named[rangespec]
        else:
            lo, hi = (int(x) for x in rangespec.split("-"))
        rf = slice(lo, hi)
        g = bin_all_levels(source, levels, extent, row_filter=rf)
        for lvl, (idx, cnt) in g.items():
            write_grid(os.path.join(data_dir, f"grid-{key}-{lvl}.bin"), lvl, idx, cnt)
        print(f"wrote subset {key} rows={hi - lo}", file=sys.stderr)

    print("done", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description="basemap tile builder (component A)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build grids/samples for a map")
    b.add_argument("--coords", required=True, help="coordinates dir (chunk-*/coordinates.npy)")
    b.add_argument("--out", required=True, help="output viewer dir")
    b.add_argument("--map-kind", required=True, choices=["jina-25m", "minilm-150m"])
    b.add_argument("--levels", default="64,128,256,512,1024")
    b.add_argument("--sample-level", type=int, default=256)
    b.add_argument("--super-tile", type=int, default=16)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--layer", action="append", help="subset spec key:rangename or key:lo-hi")
    b.set_defaults(func=_build_cli)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
