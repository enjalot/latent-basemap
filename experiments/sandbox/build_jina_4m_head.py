#!/usr/bin/env python3
"""P1.6 head-size experiment (4th-review, delegate 2026-08-29): build the nested
composition-matched 4M jina head substrate.

The 4M is a NESTED subset of the jina-prompted 6.25M substrate: a seed-42 stratified draw
of 64% of EACH span (3 EN corpora + 20 languages), so composition is matched to the 6.25M
(the only variable vs the 2M head and the direct 6.25M reference is HEAD SIZE). Span boundaries
are reconstructed exactly from the component block sizes (en/en2 + ml/ml2 per corpus/lang),
which sum to exactly 6,250,000; 64% per span sums to exactly 4,000,000.

Outputs (to /data/latent-basemap/substrates/jina-4m-head/):
  substrate.f16.npy   (4,000,000 x 768) f16 — the head's training substrate
  member_indices.npy  (int64, 4,000,000) — GLOBAL indices into the 6.25M (the member set for
                      projection member/unseen scoring)
  subsets.npy         (str, 4,000,000) — per-row span label
  manifest.json       per-span counts + seed + provenance

Memory-safe: the 6.25M f16 mmap (9.6 GB) is never materialized; rows are gathered in <=500K
chunks into a preallocated output memmap.
"""
import json
from pathlib import Path

import numpy as np

P = Path("/data/latent-basemap/substrates/jina-prompted")
OUT = Path("/data/latent-basemap/substrates/jina-4m-head")
SEED = 42
FRAC = 0.64  # 4.0M / 6.25M
EN = ("fineweb-edu", "redpajama", "pile")
LANGS = ("arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek", "fra_Latn",
         "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan", "kor_Hang", "nld_Latn",
         "pol_Latn", "por_Latn", "rus_Cyrl", "spa_Latn", "swe_Latn", "tha_Thai",
         "tur_Latn", "vie_Latn")


def _rows(f):
    return int(np.load(P / f, mmap_mode="r").shape[0])


def _spans():
    """Exact (label, length) spans of the 6.25M in build order (EN x3 then lang x20)."""
    spans = []
    for c in EN:
        spans.append((f"en-{c}", _rows(f"en-{c}.f16.npy") + _rows(f"en2-{c}.f16.npy")))
    for l in LANGS:
        spans.append((f"ml-{l}", _rows(f"ml-{l}.f16.npy") + _rows(f"ml2-{l}.f16.npy")))
    return spans


def main():
    sub6 = np.load(P / "substrate-6250k.f16.npy", mmap_mode="r")
    n6 = int(sub6.shape[0])
    d = int(sub6.shape[1])
    spans = _spans()
    assert sum(l for _, l in spans) == n6, f"span sum {sum(l for _,l in spans)} != {n6}"

    rng = np.random.default_rng(SEED)
    off = 0
    picks = []          # list of (label, sorted global-index array)
    for label, length in spans:
        k = int(round(FRAC * length))
        local = np.sort(rng.choice(length, size=k, replace=False))
        picks.append((label, (local + off).astype(np.int64)))
        off += length
    member_idx = np.concatenate([g for _, g in picks])   # already globally sorted (contiguous spans)
    n4 = int(member_idx.shape[0])
    subsets = np.concatenate([np.full(len(g), label, dtype=object) for label, g in picks])
    assert n4 == member_idx.shape[0] == subsets.shape[0]
    assert np.all(np.diff(member_idx) > 0), "member indices must be strictly increasing (unique, sorted)"

    OUT.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(OUT / "substrate.f16.npy", mode="w+",
                                    dtype=np.float16, shape=(n4, d))
    CH = 500_000
    for i in range(0, n4, CH):
        idx = member_idx[i:i + CH]
        out[i:i + len(idx)] = np.asarray(sub6[idx], dtype=np.float16)
        print(f"  gathered {min(i+len(idx), n4):,}/{n4:,}", flush=True)
    out.flush()
    del out
    np.save(OUT / "member_indices.npy", member_idx)
    np.save(OUT / "subsets.npy", subsets, allow_pickle=True)
    manifest = {
        "schema": "jina-4m-head-2026-08-29", "seed": SEED, "frac": FRAC,
        "source_substrate": str(P / "substrate-6250k.f16.npy"), "source_rows": n6,
        "rows": n4, "dim": d,
        "spans": [{"label": lb, "source_len": ln, "drawn": int(round(FRAC * ln))}
                  for (lb, ln) in spans],
        "note": ("NESTED seed-42 stratified 64%-per-span draw of the jina-prompted 6.25M. "
                 "member_indices.npy = global indices into the 6.25M (the head's training rows) "
                 "-> member mask for P1.6 projection member/unseen scoring. Composition matched "
                 "to the 6.25M; head SIZE is the sole variable vs the 2M head + direct 6.25M ref."),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {OUT}/substrate.f16.npy ({n4:,} x {d}) + member_indices + subsets + manifest",
          flush=True)
    print(f"  spans: {len(spans)}, total drawn {n4:,} (target 4,000,000)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
