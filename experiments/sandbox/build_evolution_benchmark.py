#!/usr/bin/env python3
"""Corpus-evolution benchmark substrate timeline (plan-evolution-benchmark.md §2, owner priority
2026-08-31). CPU-only builder + disjointness proofs. md000 kernel; scale set by the cuVS check.

Timeline: T0 (seeded 40/25/25/10 MiniLM draw) + 5 tranches of 20%-of-T0 each. T1/T2/T4/T5 are
IN-DISTRIBUTION growth (same 40/25/25/10 mix); T3 = OOD reddit injection (holdout-disjoint from the
reddit probe truth). final corpus = 2xT0. Every snapshot Sk = T0 ∪ T1..Tk is scored on its own exact
truth (built later, GPU). All tranches + T0 are MUTUALLY DISJOINT by one-permutation-per-corpus
construction, and disjoint from the eval probe truths (a1-common-neutral by content, probe-code-heldout
by coord, reddit probe by the 300k holdout offset).

Env: T0_SCALE (default 5_000_000). tranche = T0_SCALE//5. Outputs to
/data/latent-basemap/substrates/evolbench/{T0,T1,T2,T3,T4,T5}/ (+ evolbench-proofs.json).
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_draw_universality import (  # vetted helpers
    _corpus_shards, _shard_offsets, _prov_globals, _draw_distinct, _void_view, _gather,
    DIM, PROV_DTYPE, CORPUS_CODE as BASE_CODE, BASELINE_PROV, CODE_HELDOUT_PROV, A1_PROV)

E = "/data/embeddings"
SEED = 42
T0_SCALE = int(os.environ.get("T0_SCALE", "5000000"))
TRANCHE = T0_SCALE // 5                       # 20% of T0
MIX = {"fineweb": 0.40, "redpajama": 0.25, "pile": 0.25, "starcoder": 0.10}
BASE_ORDER = ("fineweb", "redpajama", "pile", "starcoder")
BASE_PATHS = {
    "fineweb":   f"{E}/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train",
    "redpajama": f"{E}/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2/train",
    "pile":      f"{E}/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2/train",
    "starcoder": f"{E}/starcoderdata-code-chunked-120-all-MiniLM-L6-v2/train",
}
REDDIT_PATH = f"{E}/reddit-tldr17-chunked-120-all-MiniLM-L6-v2/train"
REDDIT_CODE = 4
REDDIT_HOLDOUT = 300_000                       # front slice reserved for the reddit probe
IN_DIST = ("T0", "T1", "T2", "T4", "T5")       # base-corpus snapshots (T0 5x, tranches 1x)
OUT = Path("/data/latent-basemap/substrates/evolbench")
# per base-corpus counts: T0 = MIX*T0_SCALE; each in-dist tranche = MIX*TRANCHE.
EXCLUDE = {c: ([A1_PROV] + ([CODE_HELDOUT_PROV] if c == "starcoder" else [])) for c in BASE_ORDER}


def _cnt(scale, corpus):  # exact per-corpus count for a snapshot of `scale` rows at the mix
    if corpus == "pile":  # give pile the rounding remainder so the sum is exact
        return scale - sum(int(round(MIX[c] * scale)) for c in BASE_ORDER if c != "pile")
    return int(round(MIX[corpus] * scale))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for t in ("T0", "T1", "T2", "T3", "T4", "T5"):
        if (OUT / t / "substrate.f32.npy").exists():
            raise SystemExit(f"REFUSE overwrite: {OUT/t}")
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    # per base-corpus: draw the TOTAL across all in-dist snapshots in one call (disjoint), content-clean
    # of a1; then slice T0 (5x) + T1/T2/T4/T5 (1x each) in order.
    a1_rows = np.asarray(np.load(A1_PROV.parent / "substrate.f32.npy", mmap_mode="r"), dtype=np.float32)
    a1_void = _void_view(a1_rows)
    seg = {t: {"rows": [], "prov": []} for t in ("T0", "T1", "T2", "T3", "T4", "T5")}
    meta = {}
    for c in BASE_ORDER:
        shards = _corpus_shards(c); offs, sr = _shard_offsets(shards); T = int(offs[-1]); code = BASE_CODE[c]
        excl = np.empty(0, np.int64)
        for src in EXCLUDE[c]:
            if Path(src).exists():
                excl = np.union1d(excl, _prov_globals(src, code, offs))
        need = _cnt(T0_SCALE, c) + 4 * _cnt(TRANCHE, c)
        buf = int(need * 1.3) + excl.shape[0] + 200_000
        cand = _draw_distinct(rng, T, min(buf, T))
        cand = cand[~np.isin(cand, np.unique(excl), assume_unique=False)]
        rows, shof, loc = _gather(shards, cand, offs)
        keep = ~np.isin(_void_view(rows), a1_void); a1hits = int((~keep).sum())
        rows, shof, loc = rows[keep][:need], shof[keep][:need], loc[keep][:need]
        if rows.shape[0] < need:
            raise SystemExit(f"{c}: short {rows.shape[0]}<{need}")
        # partition: T0 gets 5x, each tranche 1x
        sizes = [(_cnt(T0_SCALE, c), "T0")] + [(_cnt(TRANCHE, c), t) for t in ("T1", "T2", "T4", "T5")]
        off = 0
        for n, t in sizes:
            r, s, l = rows[off:off+n], shof[off:off+n], loc[off:off+n]; off += n
            pr = np.empty(n, PROV_DTYPE); pr["corpus"] = code; pr["shard"] = s; pr["row"] = l
            seg[t]["rows"].append(r); seg[t]["prov"].append(pr)
        meta[c] = {"pool": T, "a1_hits_removed": a1hits, "coord_excluded": int(np.unique(excl).shape[0])}
        print(f"  {c}: {need:,} drawn (a1 hits {a1hits})", flush=True)

    # T3 = reddit OOD, offset >= holdout (disjoint from the reddit probe)
    import glob as _g
    rfiles = sorted(_g.glob(os.path.join(REDDIT_PATH, "*.npy")))
    rsh = [np.load(f, mmap_mode="r") if open(f, "rb").read(6) == b"\x93NUMPY"
           else np.memmap(f, np.float32, "r", shape=(os.path.getsize(f)//(DIM*4), DIM)) for f in rfiles]
    roffs = np.concatenate([[0], np.cumsum([int(s.shape[0]) for s in rsh])]).astype(np.int64); rT = int(roffs[-1])
    r_need = TRANCHE
    r_cand = _draw_distinct(rng, rT - REDDIT_HOLDOUT, r_need) + REDDIT_HOLDOUT  # offset past holdout
    r_rows, r_sh, r_lo = _gather(rsh, np.sort(r_cand), roffs)
    pr = np.empty(r_need, PROV_DTYPE); pr["corpus"] = REDDIT_CODE; pr["shard"] = r_sh; pr["row"] = r_lo
    seg["T3"]["rows"].append(r_rows); seg["T3"]["prov"].append(pr)
    meta["reddit"] = {"pool": rT, "holdout_offset": REDDIT_HOLDOUT}
    print(f"  reddit T3: {r_need:,} drawn (offset>={REDDIT_HOLDOUT:,})", flush=True)

    # write snapshots
    prov_by_t = {}
    for t in ("T0", "T1", "T2", "T3", "T4", "T5"):
        d = OUT / t; d.mkdir(parents=True, exist_ok=True)
        sub = np.concatenate(seg[t]["rows"]); prov = np.concatenate(seg[t]["prov"])
        np.save(d / "substrate.f32.npy", sub); np.save(d / "provenance.npy", prov)
        prov_by_t[t] = prov
        print(f"  {t}: {sub.shape[0]:,} rows written", flush=True)
        del sub

    # PROOFS: pairwise tranche/T0 disjoint (exact coord) + vs eval probes
    def _keys(p): return set(zip(p["corpus"].tolist(), p["shard"].tolist(), p["row"].tolist()))
    ks = {t: _keys(prov_by_t[t]) for t in prov_by_t}
    tlist = list(prov_by_t)
    pairwise = {f"{a}∩{b}": len(ks[a] & ks[b]) for i, a in enumerate(tlist) for b in tlist[i+1:]}
    # content-proof vs a1 (CHUNKED, memory-safe — never materialize the full base union) for all base
    # snapshots; coord vs code-heldout; reddit vs its holdout.
    a1_overlap = 0
    for t in IN_DIST:
        mm = np.load(OUT / t / "substrate.f32.npy", mmap_mode="r")
        for i in range(0, mm.shape[0], 500_000):
            a1_overlap += int(np.isin(_void_view(np.asarray(mm[i:i+500_000], dtype=np.float32)),
                                      a1_void).sum())
    ch = np.load(CODE_HELDOUT_PROV, mmap_mode="r"); chk = _keys(ch)
    codeheldout_overlap = {t: len(ks[t] & chk) for t in IN_DIST}
    reddit_holdout_ok = bool(int((prov_by_t["T3"]["row"] < REDDIT_HOLDOUT).sum()) == 0)
    proofs = {
        "schema": "evolbench-proofs-2026-08-31", "seed": SEED, "T0_scale": T0_SCALE, "tranche": TRANCHE,
        "final_corpus": T0_SCALE * 2, "mix": MIX,
        "snapshot_rows": {t: int(prov_by_t[t].shape[0]) for t in prov_by_t},
        "pairwise_intersections": pairwise, "pairwise_disjoint": all(v == 0 for v in pairwise.values()),
        "a1_content_overlap_all_base": a1_overlap,
        "codeheldout_coord_overlap": codeheldout_overlap,
        "reddit_disjoint_from_holdout": reddit_holdout_ok,
        "all_disjoint": bool(all(v == 0 for v in pairwise.values()) and a1_overlap == 0
                             and all(v == 0 for v in codeheldout_overlap.values()) and reddit_holdout_ok),
        "corpus_meta": meta, "wall_s": round(time.time()-t0, 1),
        "note": ("T3 is reddit OOD (drift trigger MUST fire at T3, not T1/T2). Base snapshots content-"
                 "exclude a1-common-neutral + coord-exclude probe-code-heldout; reddit offset>=300k "
                 "holdout. Snapshots are mutually disjoint (fresh growth, no re-draw)."),
    }
    (OUT / "evolbench-proofs.json").write_text(json.dumps(proofs, indent=1))
    for t in prov_by_t:
        (OUT/t/"manifest.json").write_text(json.dumps(
            {"snapshot": t, "rows": int(prov_by_t[t].shape[0]), "T0_scale": T0_SCALE,
             "role": ("OOD reddit injection" if t == "T3" else "in-distribution"),
             "pairwise_disjoint": proofs["pairwise_disjoint"]}, indent=1))
    print(f"\n=== PROOFS === pairwise_disjoint={proofs['pairwise_disjoint']} a1_overlap={a1_overlap} "
          f"codeheldout={codeheldout_overlap} reddit_holdout_ok={reddit_holdout_ok} "
          f"-> ALL_DISJOINT={proofs['all_disjoint']} ({proofs['wall_s']}s)", flush=True)
    return 0 if proofs["all_disjoint"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
