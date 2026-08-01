#!/usr/bin/env python3
"""Emit a tiny synthetic ``data/`` directory matching the frozen
``basemap-viewer-manifest-v1`` contract, so the viewer can be exercised
end-to-end (headless playwright gate) without any real map artifacts.

Usage:
    uv run python experiments/viewer_assets/make_fixture.py --out /tmp/viewerfix/data

Writes: manifest.json, grid-<layer>-<L>.bin, samples-all-<sx>_<sy>.json,
points-<layer>.bin, metrics-anchors.bin, metrics-queries.json — all in the
exact little-endian byte layout the viewer's parsers expect. The byte formats
here are the single source of truth the integrator/tests reuse.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

MAGIC_GRID = 0x42494E31  # "BIN1"
MAGIC_PTS = 0x50545331   # "PTS1"
MAGIC_ANC = 0x414E4331   # anchors

EXTENT = (-10.0, -8.0, 10.0, 8.0)  # xmin, ymin, xmax, ymax
LEVELS = [64, 128, 256, 512]
SAMPLE_LEVEL = 256
SUPER_TILE = 16

_WORDS = ("model embeddings cluster corpus token vector density retrieval "
          "neighbor manifold projection latent semantic sample probe language "
          "fineweb redpajama pile polish query anchor recall").split()


def _rng(seed):
    return np.random.default_rng(seed)


def make_points(seed, n, centers):
    r = _rng(seed)
    parts = []
    per = n // len(centers)
    for cx, cy, s in centers:
        parts.append(r.normal([cx, cy], s, size=(per, 2)))
    xy = np.concatenate(parts, axis=0).astype(np.float32)
    x0, y0, x1, y1 = EXTENT
    xy[:, 0] = np.clip(xy[:, 0], x0 + 1e-3, x1 - 1e-3)
    xy[:, 1] = np.clip(xy[:, 1], y0 + 1e-3, y1 - 1e-3)
    return xy


def bin_cells(xy, level):
    x0, y0, x1, y1 = EXTENT
    w, h = (x1 - x0), (y1 - y0)
    cx = np.clip(((xy[:, 0] - x0) / (w / level)).astype(np.int64), 0, level - 1)
    cy = np.clip(((xy[:, 1] - y0) / (h / level)).astype(np.int64), 0, level - 1)
    idx = cy * level + cx
    uniq, counts = np.unique(idx, return_counts=True)
    return uniq.astype(np.uint32), counts.astype(np.uint32)


def write_grid(path, level, cells, counts):
    order = np.argsort(cells)
    cells, counts = cells[order], counts[order]
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC_GRID, level, len(cells), 0))
        f.write(cells.astype("<u4").tobytes())
        f.write(counts.astype("<u4").tobytes())


def write_points(path, xy):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC_PTS, len(xy)))
        f.write(xy.astype("<f4").tobytes())


def write_anchors(path, xy, score):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC_ANC, len(xy)))
        trip = np.empty((len(xy), 3), np.float32)
        trip[:, :2] = xy
        trip[:, 2] = score
        f.write(trip.astype("<f4").tobytes())


def sample_text(r):
    k = int(r.integers(6, 16))
    return "[CLS] " + " ".join(r.choice(_WORDS, size=k)).strip()


def write_samples(out: Path, xy, groups):
    """Per-supertile JSON of up to 3 text samples per nonempty sample_level cell."""
    r = _rng(99)
    x0, y0, x1, y1 = EXTENT
    w, h = (x1 - x0), (y1 - y0)
    L = SAMPLE_LEVEL
    per = L // SUPER_TILE
    cx = np.clip(((xy[:, 0] - x0) / (w / L)).astype(np.int64), 0, L - 1)
    cy = np.clip(((xy[:, 1] - y0) / (h / L)).astype(np.int64), 0, L - 1)
    cell = cy * L + cx
    tiles = {}
    seen = {}
    order = r.permutation(len(xy))
    for i in order:
        c = int(cell[i])
        if seen.get(c, 0) >= 3:
            continue
        seen[c] = seen.get(c, 0) + 1
        sx, sy = int(cx[i] // per), int(cy[i] // per)
        key = f"{sx}_{sy}"
        tiles.setdefault(key, {}).setdefault(str(c), []).append(
            {"t": sample_text(r)[:200], "g": groups[i % len(groups)], "r": int(i)}
        )
    for key, cells in tiles.items():
        (out / f"samples-all-{key}.json").write_text(json.dumps({"cells": cells}))


def write_queries(out: Path):
    r = _rng(7)
    probes = []
    for pk, plabel, base in (("pol_Latn", "Polish (held-out language)", (-6, 4)),
                             ("trec_covid", "TREC-COVID (OOD corpus)", (5, -3))):
        queries = []
        for _ in range(24):
            qx = float(np.clip(r.normal(base[0], 2.0), EXTENT[0] + 0.2, EXTENT[2] - 0.2))
            qy = float(np.clip(r.normal(base[1], 1.6), EXTENT[1] + 0.2, EXTENT[3] - 0.2))
            neigh, hits, texts = [], [], []
            nhit = int(r.integers(1, 8))
            for j in range(10):
                nx = float(np.clip(qx + r.normal(0, 0.9), EXTENT[0], EXTENT[2]))
                ny = float(np.clip(qy + r.normal(0, 0.9), EXTENT[1], EXTENT[3]))
                neigh.append([round(nx, 3), round(ny, 3)])
                hits.append(j < nhit)
                texts.append(sample_text(r)[:120])
            queries.append({
                "xy": [round(qx, 3), round(qy, 3)],
                "neighbors": neigh,
                "hits": hits,
                "recall": round(nhit / 10.0, 3),
                "text": sample_text(r)[:120],
                "neighbor_texts": texts,
            })
        recall50 = round(float(np.mean([q["recall"] for q in queries])), 4)
        probes.append({"key": pk, "label": plabel, "recall50": recall50, "queries": queries})
    (out / "metrics-queries.json").write_text(json.dumps({"probes": probes}))
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # base "all" corpus: three blobs
    all_xy = make_points(1, 12000, [(-5, 3, 1.6), (4, -2, 2.0), (-2, -4, 1.2)])
    groups_all = ["fineweb", "redpajama", "pile"]
    row_groups = np.array([groups_all[i % 3] for i in range(len(all_xy))])

    for L in LEVELS:
        cells, counts = bin_cells(all_xy, L)
        write_grid(out / f"grid-all-{L}.bin", L, cells, counts)

    # one grid subset layer (fineweb blob only)
    fw_xy = all_xy[row_groups == "fineweb"]
    for L in LEVELS:
        cells, counts = bin_cells(fw_xy, L)
        write_grid(out / f"grid-fineweb-{L}.bin", L, cells, counts)

    # one points subset layer (held-out language probe corpus)
    pol_xy = make_points(3, 480, [(-6, 4, 1.0)])
    write_points(out / "points-lang-pol.bin", pol_xy)

    write_samples(out, all_xy, list(row_groups))

    # anchors: scattered, score in [0,1]
    anc_xy = make_points(5, 320, [(0, 0, 5.0)])
    r = _rng(6)
    anc_score = r.random(len(anc_xy)).astype(np.float32)
    write_anchors(out / "metrics-anchors.bin", anc_xy, anc_score)

    probes = write_queries(out)

    manifest = {
        "schema": "basemap-viewer-manifest-v1",
        "map_id": "fixture-synthetic",
        "round_id": "0000",
        "title": "Synthetic fixture map — viewer gate",
        "rows_total": int(len(all_xy)),
        "rows_note": "synthetic fixture, exact count — not a sample",
        "extent": list(EXTENT),
        "levels": LEVELS,
        "sample_level": SAMPLE_LEVEL,
        "super_tile": SUPER_TILE,
        "layers": [
            {"key": "all", "label": "Full corpus", "kind": "grid",
             "rows": int(len(all_xy)), "levels": LEVELS},
            {"key": "fineweb", "label": "FineWeb-Edu", "kind": "grid",
             "rows": int(len(fw_xy)), "levels": LEVELS, "group": "corpora"},
            {"key": "lang-pol", "label": "Polish (held-out)", "kind": "points",
             "rows": int(len(pol_xy)), "group": "held-out languages"},
        ],
        "metrics": {
            "anchors": {"file": "metrics-anchors.bin", "count": int(len(anc_xy)),
                        "score": "local expansion (log2 vs median)",
                        "summary": {"ffr": 0.6386}},
            "probes": [{"key": p["key"], "label": p["label"],
                        "queries": len(p["queries"]), "recall50": p["recall50"]}
                       for p in probes],
        },
        "provenance": {"training_round": "0000", "eval_round": "0000",
                       "evidence_status": "accepted (synthetic)",
                       "panel": {}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote fixture to {out} ({len(list(out.iterdir()))} files)")


if __name__ == "__main__":
    main()
