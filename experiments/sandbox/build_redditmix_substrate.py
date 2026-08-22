#!/usr/bin/env python3
"""Build the minilm-redditmix-2m substrate (owner order 2026-08-22).

New-mixture experiment: 80% of the sealed 2M substrate (per-corpus strided
keep, preserving fineweb-edu 40 / RPJ 25 / pile 25 / starcoder 10 shares) +
400K reddit-tldr17 chunks (20% conversational register — the register MiniLM
was mostly trained on, near-zero in the old mix). Output:

  /data/latent-basemap/substrates/minilm-redditmix-2m/{substrate.f32.npy,
  subsets.npy, manifest.json}

Trained by image_map_pipeline (promoted control + composed-x8) on its own
exact-k15 fuzzy graph; before/after comparisons vs the old-mix maps use
own-truth FFR + the frozen-map register probes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SRC = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
           "minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
OUT = Path("/data/latent-basemap/substrates/minilm-redditmix-2m")
CORPUS_NAMES = {0: None, 1: None, 2: None, 3: None}  # resolved from manifest
KEEP_FRAC = 0.8
REDDIT_ROWS = 400_000


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "substrate.f32.npy").exists():
        print("exists, skip")
        return 0
    sub = np.load(SRC / "substrate.f32.npy", mmap_mode="r")
    prov = np.load(SRC / "provenance.npy")
    graph_doc = json.loads((SRC / "substrate-graph.json").read_text())
    comp = graph_doc["composition"]
    # corpus codes are assigned in composition-key order (u1 codes 0..3);
    # verify counts match to bind code -> name safely.
    names = sorted(comp)  # deterministic
    counts = {c: int((prov["corpus"] == c).sum()) for c in range(len(names))}
    by_rows = {n: comp[n]["rows"] for n in names}
    code_to_name = {}
    used = set()
    for c in range(len(names)):
        match = [n for n in names if by_rows[n] == counts[c] and n not in used]
        assert match, (c, counts[c], by_rows)
        code_to_name[c] = match[0]
        used.add(match[0])
    short = {n: n.split("-chunked")[0].split("-sample")[0].lower()
             for n in names}

    keep_idx = []
    labels = []
    for c, name in code_to_name.items():
        rows = np.nonzero(prov["corpus"] == c)[0]
        k = int(round(len(rows) * KEEP_FRAC))
        sel = rows[np.linspace(0, len(rows) - 1, k).astype(np.int64)]
        keep_idx.append(sel)
        labels.extend([short[name]] * k)
        print(f"{short[name]}: keep {k:,}/{len(rows):,}")
    keep_idx = np.concatenate(keep_idx)
    base = np.asarray(sub[np.sort(keep_idx)], dtype=np.float32)
    # labels must follow the SORTED row order
    order = np.argsort(keep_idx)
    labels = np.asarray(labels, dtype=object)[order]

    import glob
    shards = sorted(glob.glob(
        "/data/embeddings/reddit-tldr17-chunked-120-all-MiniLM-L6-v2/"
        "train/*.npy"))
    parts, got = [], 0
    for f in shards:
        a = np.load(f, mmap_mode="r")
        take = min(REDDIT_ROWS - got, a.shape[0])
        parts.append(np.asarray(a[:take], dtype=np.float32))
        got += take
        if got >= REDDIT_ROWS:
            break
    assert got == REDDIT_ROWS
    reddit = np.concatenate(parts)

    full = np.concatenate([base, reddit])
    subsets = np.concatenate([labels,
                              np.asarray(["reddit"] * REDDIT_ROWS, dtype=object)])
    assert full.shape[0] == subsets.shape[0]
    np.save(OUT / "substrate.f32.npy", full)
    np.save(OUT / "subsets.npy", subsets)
    (OUT / "manifest.json").write_text(json.dumps({
        "rows": int(full.shape[0]),
        "base": str(SRC), "keep_frac": KEEP_FRAC,
        "reddit_rows": REDDIT_ROWS,
        "composition_note": "80% of each sealed-substrate corpus (strided) + "
                            "20% reddit-tldr17 (register experiment)",
    }, indent=1))
    print(f"built {full.shape} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
