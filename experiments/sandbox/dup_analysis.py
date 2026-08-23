#!/usr/bin/env python3
"""Near-duplicate analysis + dedup substrate builder (owner order 2026-08-23).

Detection is FREE: every image/text suite saved its exact k15 kNN
(knn_indices/knn_dists, cosine distance). Near-dup groups = connected
components over knn edges with distance < EPS (union-find, CPU). For each
dataset: group count/size distribution, mass inside dup groups, and — where a
trained map exists — whether dup groups ARE the far-flung satellites
(mean group position radius vs the bulk).

`build` then writes a dup-aware substrate for a dataset: one representative
per group (the member closest to the group feature mean) + a multiplicity
column, so a rebuild trains on unique rows while FFR truth stays the
original. (Graph rebuild for the deduped substrate needs GPU — queued
separately.)

Usage: dup_analysis.py analyze [eps] | build <dataset> [eps]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")
EPS_DEFAULT = 0.005   # cosine distance < 0.005  (cos sim > 0.995)
DATASETS = ["sisap-clip-2m", "bl-siglip-1m", "jina-en-2m", "jina-multi-2m",
            "reddit-2m", "communityarchive-2m", "minilm-redditmix-2m"]


def _find(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def dup_groups(ds: str, eps: float):
    idx = np.load(SANDBOX / ds / "knn_indices.npy")
    dst = np.load(SANDBOX / ds / "knn_dists.npy")
    n, k = idx.shape
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    tgt = idx.reshape(-1).astype(np.int64)
    close = dst.reshape(-1) < eps
    parent = np.arange(n, dtype=np.int64)
    for s, t in zip(src[close], tgt[close]):
        rs, rt = _find(parent, s), _find(parent, t)
        if rs != rt:
            parent[rt] = rs
    roots = np.array([_find(parent, i) for i in range(n)], dtype=np.int64)
    _, labels, counts = np.unique(roots, return_inverse=True,
                                  return_counts=True)
    return labels, counts


def analyze(eps: float) -> int:
    report = {}
    for ds in DATASETS:
        if not (SANDBOX / ds / "knn_dists.npy").exists():
            continue
        t0 = time.time()
        labels, counts = dup_groups(ds, eps)
        n = len(labels)
        gsz = counts[labels]
        in_dup = gsz > 1
        big = counts[counts > 1]
        entry = {
            "rows": int(n),
            "dup_groups_gt1": int(len(big)),
            "rows_in_dup_groups": int(in_dup.sum()),
            "dup_mass_frac": float(in_dup.mean()),
            "largest_group": int(counts.max()),
            "p99_group": int(np.percentile(big, 99)) if len(big) else 1,
            "unique_rows_after_dedup": int(len(counts)),
            "wall_s": round(time.time() - t0, 1),
        }
        # satellite correlation: are dup groups far from the map center?
        for arm_dir in sorted((SANDBOX / ds).iterdir()):
            cpath = arm_dir / "coordinates.npy"
            if not cpath.exists():
                continue
            xy = np.load(cpath, mmap_mode="r")
            if xy.shape[0] != n:
                continue
            xy = np.asarray(xy)
            center = np.median(xy, axis=0)
            r = np.linalg.norm(xy - center, axis=1)
            bulk = np.percentile(r, 99)
            far = r > bulk
            entry["satellite_check"] = {
                "arm": arm_dir.name,
                "far_points": int(far.sum()),
                "far_points_in_dup_groups_frac": float(in_dup[far].mean())
                if far.any() else 0.0,
                "overall_dup_frac": float(in_dup.mean()),
            }
            break
        report[ds] = entry
        print(f"{ds}: {entry['dup_mass_frac']:.1%} mass in dup groups "
              f"({entry['dup_groups_gt1']:,} groups, largest "
              f"{entry['largest_group']:,}); dedup -> "
              f"{entry['unique_rows_after_dedup']:,} rows", flush=True)
        if "satellite_check" in entry:
            sc = entry["satellite_check"]
            print(f"  satellites ({sc['arm']}): {sc['far_points']:,} far pts, "
                  f"{sc['far_points_in_dup_groups_frac']:.1%} in dup groups "
                  f"(base rate {sc['overall_dup_frac']:.1%})", flush=True)
    out = SANDBOX / "dup-analysis.json"
    out.write_text(json.dumps({"eps": eps, "datasets": report}, indent=1))
    print(f"report: {out}")
    return 0


def build(ds: str, eps: float) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from image_map_pipeline import DATASETS as DS, _norm

    labels, counts = dup_groups(ds, eps)
    x = _norm(DS[ds]["load"]())
    n = x.shape[0]
    order = np.argsort(labels, kind="stable")
    reps = np.empty(len(counts), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for g, (s, c) in enumerate(zip(starts, counts)):
        members = order[s:s + c]
        if c == 1:
            reps[g] = members[0]
        else:
            mean = x[members].mean(axis=0)
            reps[g] = members[np.argmax(x[members] @ mean)]
    out = Path("/data/latent-basemap/substrates") / f"{ds}-dedup"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "substrate.f32.npy", x[reps])
    np.save(out / "multiplicity.npy", counts.astype(np.int32))
    np.save(out / "representatives.npy", reps)
    np.save(out / "labels.npy", labels.astype(np.int32))
    (out / "manifest.json").write_text(json.dumps({
        "source_dataset": ds, "eps": eps, "rows_before": int(n),
        "rows_after": int(len(counts)),
        "note": "one representative per near-dup group (closest to group "
                "mean); multiplicity preserved for dup-aware weighting; FFR "
                "truth stays the ORIGINAL graph.",
    }, indent=1))
    print(f"{ds}: {n:,} -> {len(counts):,} rows at {out}")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "analyze":
        raise SystemExit(analyze(float(sys.argv[2]) if len(sys.argv) > 2 else EPS_DEFAULT))
    raise SystemExit(build(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else EPS_DEFAULT))
