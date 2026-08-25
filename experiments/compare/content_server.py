#!/usr/bin/env python3
"""Row-content API for the compare page (port 8801).

GET /content?rung=<rung>&ids=1,2,3   -> {"items":[{"id","text","thumb","meta"}]}
GET /thumb?rung=bl-siglip-1m&id=123  -> webp bytes

Resolution follows the 2026-08-25 provenance spec (agent survey; see the
DATASETS registry in experiments/sandbox/image_map_pipeline.py for row
orders). Everything is lazy: sidecars are memmapped, parquet chunk_text
columns load per-file into a small LRU. Nothing >~200MB is ever resident.
"""
from __future__ import annotations

import glob
import json
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

SIDECAR = Path("/data/latent-basemap/textsidecar/"
               "minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
CHUNKS = Path("/data/chunks")
BL = Path("/data/images/british-library-book-images")
BL_ROWS = Path("/data/latent-basemap/substrates/bl-siglip2-1m/rows.parquet")
DEDUP = Path("/data/latent-basemap/substrates/sisap-clip-2m-dedup")

_JINA_LANGS = ("arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek",
               "fra_Latn", "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan",
               "kor_Hang", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl",
               "spa_Latn", "swe_Latn", "tha_Thai", "tur_Latn", "vie_Latn")
BL_SPANS = [("covers", 0, 61_548), ("medium", 61_548, 278_648),
            ("embellishments", 278_648, 695_583),
            ("plates", 695_583, 1_080_814)]

# ---------- lazy stores ----------
_cache: dict = {}


def sidecar():
    if "sc" not in _cache:
        off = np.memmap(SIDECAR / "offsets.u64", dtype="<u8", mode="r")
        _cache["sc"] = (off, open(SIDECAR / "blob.utf8", "rb"))
    return _cache["sc"]


def sidecar_text(row: int) -> str:
    off, blob = sidecar()
    a, b = int(off[row]), int(off[row + 1])
    blob.seek(a)
    t = blob.read(b - a).decode("utf-8", "replace")
    return t[6:] if t.startswith("[CLS] ") else t


TRUNC = 800  # chars kept per text in the row-group cache


class GroupLRU:
    """Decompressed (truncated) chunk_text per parquet row group.

    Groups run up to ~1M rows; truncation keeps a resident group under
    ~250MB and maxn bounds the total. First hit on a big group costs a
    decompress (~seconds); sidecars (below) bypass this entirely.
    """

    def __init__(self, maxn=3):
        self.d: OrderedDict = OrderedDict()
        self.maxn = maxn

    def get(self, f: str, g: int) -> list:
        key = (f, g)
        if key not in self.d:
            import pyarrow.parquet as pq
            col = pq.ParquetFile(f).read_row_group(
                g, columns=["chunk_text"])["chunk_text"].to_pylist()
            self.d[key] = [t[:TRUNC] if t else "" for t in col]
            while len(self.d) > self.maxn:
                self.d.popitem(last=False)
        self.d.move_to_end(key)
        return self.d[key]


COLS = GroupLRU()


def dir_index(chunk_dir: str):
    """per-file and per-row-group cumulative row counts of a dir."""
    key = ("idx", chunk_dir)
    if key not in _cache:
        import pyarrow.parquet as pq
        files = sorted(glob.glob(f"{CHUNKS}/{chunk_dir}/train/*.parquet"))
        counts, groups = [], []
        for f in files:
            md = pq.ParquetFile(f).metadata
            counts.append(md.num_rows)
            groups.append(np.cumsum(
                [0] + [md.row_group(i).num_rows
                       for i in range(md.num_row_groups)]))
        _cache[key] = (files, np.cumsum([0] + counts), groups)
    return _cache[key]


def dir_text(chunk_dir: str, grow: int) -> str:
    # a prebuilt sidecar for this dir wins (O(1), no decompress)
    sc = compare_sidecar(chunk_dir)
    if sc:
        off, blob = sc
        if grow < len(off) - 1:
            a, b = int(off[grow]), int(off[grow + 1])
            blob.seek(a)
            return blob.read(b - a).decode("utf-8", "replace")
    files, cum, groups = dir_index(chunk_dir)
    s = int(np.searchsorted(cum, grow, side="right") - 1)
    local = grow - int(cum[s])
    gc = groups[s]
    g = int(np.searchsorted(gc, local, side="right") - 1)
    t = COLS.get(files[s], g)[local - int(gc[g])]
    return t[6:] if t.startswith("[CLS] ") else t


def compare_sidecar(chunk_dir: str):
    """offsets.u64+blob.utf8 built by build_compare_sidecars.py, or None."""
    key = ("csc", chunk_dir)
    if key not in _cache:
        d = Path("/data/latent-basemap/textsidecar/compare") / chunk_dir
        if (d / "offsets.u64").exists() and (d / "blob.utf8").exists():
            _cache[key] = (np.memmap(d / "offsets.u64", dtype="<u8",
                                     mode="r"), open(d / "blob.utf8", "rb"))
        else:
            _cache[key] = None
    return _cache[key]


def strided_text(chunk_dir: str, i: int, stride: int) -> str:
    return dir_text(chunk_dir, stride * i)


# ---------- per-rung resolvers: id -> {"text":..} | {"thumb":..} ----------
def r_sealed2m(i, mul=1):
    return {"text": sidecar_text(mul * i)}


def r_redditmix(i):
    blocks = [(0, 640_000, 0, 800_000), (640_000, 1_040_000, 800_000, 500_000),
              (1_040_000, 1_440_000, 1_300_000, 500_000),
              (1_440_000, 1_600_000, 1_800_000, 200_000)]
    for b0, b1, S, L in blocks:
        if b0 <= i < b1:
            k = b1 - b0
            sealed = S + int(round((i - b0) * (L - 1) / (k - 1)))
            return {"text": sidecar_text(sealed)}
    return {"text": dir_text("reddit-tldr17-chunked-120", i - 1_600_000)}


def r_jina_multi_2m(i):
    if i < 333_334:
        return {"text": dir_text("fineweb-edu-sample-10BT-chunked-500", i)}
    if i < 666_667:
        return {"text": dir_text("RedPajama-Data-V2-sample-10B-chunked-500",
                                 i - 333_334)}
    if i < 1_000_000:
        return {"text": dir_text("pile-uncopyrighted-chunked-500",
                                 i - 666_667)}
    j = i - 1_000_000
    lang = _JINA_LANGS[j // 50_000]
    return {"text": dir_text(f"fineweb2-{lang}-chunked-500", j % 50_000),
            "meta": lang}


def bl_lookup(i):
    for subset, s0, s1 in BL_SPANS:
        if s0 <= i < s1:
            return subset, i - s0
    return None, None


def r_bl(i):
    subset, j = bl_lookup(i)
    if subset is None:
        return {"text": "(row out of range)"}
    if "bl_meta" not in _cache:
        import pyarrow.parquet as pq
        t = pq.read_table(BL_ROWS, columns=["fname", "date", "image_type"])
        _cache["bl_meta"] = (t["fname"].to_pylist(), t["date"].to_pylist())
    fn, dt = _cache["bl_meta"]
    return {"thumb": f"/thumb?rung=bl-siglip-1m&id={i}",
            "text": f"{fn[i]} · {dt[i]} · {subset}"}


def r_sisap(i, dedup=False):
    meta = ""
    if dedup:
        if "reps" not in _cache:
            _cache["reps"] = (np.load(DEDUP / "representatives.npy",
                                      mmap_mode="r"),
                              np.load(DEDUP / "multiplicity.npy",
                                      mmap_mode="r"))
        reps, mult = _cache["reps"]
        meta = (f"sisap row {int(reps[i])} · h5 row {15 * int(reps[i])} · "
                f"dup-group size {int(mult[i])}")
    else:
        meta = f"h5 row {15 * i}"
    return {"text": f"(LAION image — no metadata on disk; {meta})"}


def r_curated(i):
    if "selrows" not in _cache:
        _cache["selrows"] = np.load(
            "/data/latent-basemap/substrates/minilm-curated-2m/"
            "selected_candidate_rows.npy", mmap_mode="r")
    c = int(_cache["selrows"][i])
    corpus = ("fineweb" if c < 1_600_000 else
              "redpajama" if c < 2_600_000 else
              "pile" if c < 3_600_000 else "starcoder")
    return {"text": f"({corpus} — exact text needs RNG replay; "
                    f"candidate row {c})"}


RESOLVERS = {
    "2m-knobs": r_sealed2m,
    "minilm-mix-1m": lambda i: r_sealed2m(i, 2),
    "minilm-mix-500k": lambda i: r_sealed2m(i, 4),
    "minilm-redditmix-2m": r_redditmix,
    "jina-multi-2m": r_jina_multi_2m,
    "reddit-2m": lambda i: {"text": strided_text(
        "reddit-tldr17-chunked-120", i, 5)},
    "communityarchive-2m": lambda i: {"text": strided_text(
        "communityarchive-tweets", i, 8)},
    "bl-siglip-1m": r_bl,
    "sisap-clip-2m": r_sisap,
    "sisap-clip-2m-dedup": lambda i: r_sisap(i, dedup=True),
    "minilm-curated-2m": r_curated,
    "minilm-random-2m": lambda i: {"text": "(random substrate — provenance "
                                           "needs RNG replay; pending)"},
}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/content":
                rung, ids = q["rung"], q["ids"].split(",")[:200]
                res = RESOLVERS.get(rung)
                items = []
                for s in ids:
                    i = int(s)
                    try:
                        it = res(i) if res else {
                            "text": f"(no resolver for {rung})"}
                    except Exception as ex:
                        it = {"text": f"(resolve error: {ex})"}
                    it["id"] = i
                    items.append(it)
                self._send(200, json.dumps({"items": items}).encode())
            elif u.path == "/thumb":
                subset, j = bl_lookup(int(q["id"]))
                p = BL / "thumbs" / subset / f"{j:08d}.webp"
                self._send(200, p.read_bytes(), "image/webp")
            else:
                self._send(404, b"{}")
        except Exception as ex:
            self._send(500, json.dumps({"error": str(ex)}).encode())


if __name__ == "__main__":
    print("content server on :8801", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8801), H).serve_forever()
