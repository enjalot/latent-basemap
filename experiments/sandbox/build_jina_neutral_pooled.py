#!/usr/bin/env python3
"""Build the jina-neutral-pooled projector register (JINA_SWEEP_PROPOSAL.md
2026-08-28). A FIXED-SEED (42) pooled uniform draw of ~120,000 rows ACROSS the
P-A/P-B/P-C holdout substrates — a single NEUTRAL jina-space register spanning
every OOD probe corpus, disjoint from every jina TRAINING substrate (its inputs
are themselves the holdout/probe registers).

Sources (27 holdout substrates; all document-prompted f16 (N,768)):
  P-A social 250k probes : reddit-jina-250k, ca-jina-250k,
                           twitter-jina-250k, bluesky-jina-250k
  P-B language probes    : probe-lang-<lang>-jina  x20  (_JINA_LANGS)
  P-C EN holdout probes  : probe-fineweb-jina, probe-rpj-jina, probe-pile-jina

DRAW: pooled uniform without replacement over the CONCATENATION of all source
rows (each pooled row equally likely -> per-source count is proportional to the
source's size), np.random.default_rng(42). Deterministic given fixed inputs.
[If the parent prefers a balanced (equal-per-source) draw instead, set
env POOL_MODE=balanced — proportional is the default.]

Output (f16, write-once):
  substrates/jina-neutral-pooled/substrate.f16.npy   (~120000, 768)
  substrates/jina-neutral-pooled/subsets.npy         per-row source label
  substrates/jina-neutral-pooled/manifest.json       sources, per-source counts, seed

CPU-only (samples existing f16 substrates; imports numpy only). Its knn+fuzzy
TRUTH is built later by image_map_pipeline.py jina-neutral-pooled knn|fuzzy.

The P-A/P-B/P-C substrates are GPU prereqs and may be PENDING; this builder is
import/compile-clean now and RUNS once they exist. It PRINTS which inputs exist
vs are pending and, if ANY source is missing, exits cleanly (0) without writing.

Usage:
  build_jina_neutral_pooled.py   # build (or report pending inputs)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SEED = 42
TOTAL = int(os.environ.get("N_TOTAL", "120000"))
DIM = 768
SUBSTRATES = Path("/data/latent-basemap/substrates")
OUT_DIR = SUBSTRATES / "jina-neutral-pooled"

_JINA_LANGS = ("arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek",
               "fra_Latn", "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan",
               "kor_Hang", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl",
               "spa_Latn", "swe_Latn", "tha_Thai", "tur_Latn", "vie_Latn")

# source register name -> substrate path. Order fixed for deterministic gather.
SOURCES = (
    ["reddit-jina-250k", "ca-jina-250k", "twitter-jina-250k", "bluesky-jina-250k"]
    + [f"probe-lang-{l}-jina" for l in _JINA_LANGS]
    + ["probe-fineweb-jina", "probe-rpj-jina", "probe-pile-jina"]
)


def _src_path(name: str) -> Path:
    return SUBSTRATES / name / "substrate.f16.npy"


def _report_inputs() -> tuple[dict, list[str]]:
    status, missing = {}, []
    print(f"[inputs] {len(SOURCES)} neutral-pool sources:")
    for name in SOURCES:
        p = _src_path(name)
        ok = p.exists()
        rows = int(np.load(p, mmap_mode="r").shape[0]) if ok else 0
        status[name] = {"path": str(p), "exists": ok, "rows": rows}
        if not ok:
            missing.append(name)
        print(f"    {'OK  ' if ok else 'PEND'} {name}: {p}"
              f"{'' if not ok else f' (rows={rows:,})'}")
    return status, missing


def _plan_counts(rows_per_source: list[int], mode: str, rng) -> list[int]:
    """Per-source draw counts summing EXACTLY to TOTAL.

    proportional: uniform-without-replacement over the pooled row space (each
      pooled row equally likely) -> multivariate-hypergeometric per-source counts.
    balanced: as-equal-as-possible per source, capped at each source's size."""
    ns = np.asarray(rows_per_source, dtype=np.int64)
    if ns.sum() < TOTAL:
        raise SystemExit(f"pool too small: {int(ns.sum())} rows < {TOTAL}")
    if mode == "balanced":
        counts = np.zeros(len(ns), dtype=np.int64)
        remaining = TOTAL
        # water-fill equal shares, capped by source size
        active = np.ones(len(ns), dtype=bool)
        while remaining > 0 and active.any():
            share = remaining // int(active.sum())
            if share == 0:
                # distribute the remainder one-by-one to the largest-headroom sources
                head = np.where(active, ns - counts, -1)
                order = np.argsort(-head)
                for i in order[:remaining]:
                    counts[i] += 1
                remaining = 0
                break
            for i in np.where(active)[0]:
                take = min(share, int(ns[i] - counts[i]))
                counts[i] += take
            remaining = TOTAL - int(counts.sum())
            active = counts < ns
        return counts.tolist()
    # proportional: draw TOTAL global indices without replacement, bin by source.
    cum = np.concatenate([[0], np.cumsum(ns)])
    gidx = rng.choice(int(ns.sum()), size=TOTAL, replace=False)
    src_of = np.searchsorted(cum, gidx, side="right") - 1
    counts = np.bincount(src_of, minlength=len(ns))
    return counts.astype(np.int64).tolist()


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    mode = os.environ.get("POOL_MODE", "proportional")
    if mode not in ("proportional", "balanced"):
        raise SystemExit(f"POOL_MODE must be proportional|balanced, got {mode!r}")

    status, missing = _report_inputs()
    if missing:
        print(f"\n[pending] {len(missing)} source substrate(s) not yet built "
              f"(GPU prereqs P-A/P-B/P-C): {missing}\n"
              f"Nothing written; re-run once they exist.")
        return 0

    sub_path = OUT_DIR / "substrate.f16.npy"
    if sub_path.exists():
        raise SystemExit(f"REFUSE overwrite: {sub_path} already exists")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)   # single rng(42): count plan then per-source draws
    rows_per_source = [status[n]["rows"] for n in SOURCES]
    counts = _plan_counts(rows_per_source, mode, rng)
    assert sum(counts) == TOTAL, (sum(counts), TOTAL)

    t0 = time.time()
    out = np.empty((TOTAL, DIM), dtype=np.float16)
    labels = np.empty(TOTAL, dtype=object)
    per_source = {}
    pos = 0
    for name, n_src, k in zip(SOURCES, rows_per_source, counts):
        if k == 0:
            per_source[name] = 0
            continue
        mm = np.load(_src_path(name), mmap_mode="r")
        assert mm.shape[1] == DIM and mm.dtype == np.float16, (name, mm.shape, mm.dtype)
        pick = np.sort(rng.choice(n_src, size=k, replace=False))  # ascending for locality
        out[pos:pos + k] = np.asarray(mm[pick], dtype=np.float16)
        labels[pos:pos + k] = name
        per_source[name] = int(k)
        pos += k
        del mm
    assert pos == TOTAL, pos

    # shuffle so source blocks are not contiguous (same rng(42))
    perm = rng.permutation(TOTAL)
    out = out[perm]
    labels = labels[perm]

    finite = bool(np.isfinite(out).all())
    assert finite, "non-finite rows in neutral-pooled substrate"

    tmp = sub_path.with_suffix(".tmp.npy")
    np.save(tmp, out)
    os.replace(tmp, sub_path)
    np.save(OUT_DIR / "subsets.npy", labels)

    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "name": "jina-neutral-pooled",
        "role": ("NEUTRAL jina-space projector register: a fixed-seed pooled "
                 "uniform draw across every P-A/P-B/P-C holdout substrate; "
                 "disjoint from every jina TRAINING substrate. knn+fuzzy truth "
                 "built later by image_map_pipeline.py jina-neutral-pooled."),
        "rows": TOTAL, "dim": DIM, "dtype": "float16", "seed": SEED,
        "draw_mode": mode,
        "shuffled": True, "finite": finite,
        "n_sources": len(SOURCES),
        "sources": list(SOURCES),
        "rows_per_source_available": {n: r for n, r in zip(SOURCES, rows_per_source)},
        "per_source_counts": per_source,
        "counts_sum": int(sum(per_source.values())),
        "draw_note": (
            "proportional = uniform-without-replacement over the concatenation of "
            "all source rows (per-source count ~ source size); balanced = "
            "as-equal-as-possible per source. POOL_MODE selects; default proportional."),
        "outputs": {"substrate": str(sub_path),
                    "subsets": str(OUT_DIR / "subsets.npy")},
        "input_status": status,
    }, indent=1))
    print(f"\n[jina-neutral-pooled] built ({mode}) shape={out.shape} "
          f"dtype={out.dtype} finite={finite} in {(time.time()-t0)/60:.1f} min "
          f"-> {sub_path}")
    top = sorted(per_source.items(), key=lambda kv: -kv[1])[:6]
    print(f"[per-source top] {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
