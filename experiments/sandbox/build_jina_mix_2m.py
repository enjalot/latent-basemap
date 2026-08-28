#!/usr/bin/env python3
"""Build a jina LANGUAGE-PRESERVING 2M social-mixture substrate (owner 2026-08-28,
JINA_SWEEP_PROPOSAL.md). The jina analog of build_bmix10cp_2m.py: social displaces
ONLY the three large EN corpora; ALL 20 per-language blocks are held BIT-IDENTICAL
to the 0% baseline in every arm (the language-preservation proof, mirroring
bmix10cp's starcoder==200k proof).

BASE (the 2M champion substrate = the undisplaced 0% arm; NOT rebuilt here)
--------------------------------------------------------------------------
The jina-multi-2m champion substrate = image_map_pipeline._jina_multi_load():
  EN 1M  = prefixes of the three en-2m.f16 corpus blocks
             fineweb-edu en-2m[0:333334]         -> 333,334
             redpajama   en-2m[666667:1000000]   -> 333,333
             pile        en-2m[1333334:1666667]  -> 333,333
  ML 1M  = multi-1m.f16  = 20 languages x 50,000 (in _JINA_LANGS order)
  -> 2,000,000 rows, 768-dim, float16.
The EXISTING sandbox/jina-multi-2m/champion-bs16k map IS the 0% arm (reuse, no
rebuild). This builder only produces the MIXED arms.

ARMS (argv[1])
--------------
  jina-bmix10-2m  balanced, share 0.10  (200,000 social)
  jina-bmix20-2m  balanced, share 0.20  (400,000 social)
  jina-bmix30-2m  balanced, share 0.30  (600,000 social)
  jina-rmix20-2m  reddit-only, share 0.20 (400,000 reddit)  [transfer check]

CONSTRUCTION (seed 42)
----------------------
 1. Displace s*2M rows drawn ONLY from the EN 1M, PROPORTIONALLY across
    fineweb/redpajama/pile (333334:333333:333333). ML (all 20 language blocks)
    is UNTOUCHED. Retained EN rows are copied BIT-IDENTICAL (f16 values by index)
    from the champion EN block; the 1M ML block is copied BIT-IDENTICAL in full.
 2. Social replacement drawn from the JINA social POOLS
    substrates/{reddit,ca,twitter,bluesky}-jina-pool/ (document-prompted, f16,
    holdout-disjoint), offset-restricted to global row index >= 300000:
      balanced -> s/4 each of reddit / CA / twitter / bluesky
      reddit   -> s all reddit
 3. Assemble retained-EN + ML + social = 2,000,000, then shuffle (same rng(42)).
    Output float16 (2,000,000, 768).

MANIFEST proves: per-language counts ALL == 50,000 (identical to baseline), EN
displaced proportionally, retained EN a bit-identical subset of the champion EN
block, ML bit-identical, every social row global index >= 300000.

Write-once (refuses to overwrite substrate.f16.npy). CPU-only. Imports numpy only.
NOTE: the social pools are a GPU prereq (P-A) and may be PENDING; this builder is
import/compile-clean now and RUNS once the pools exist. It PRINTS which inputs
exist vs are pending and, if a pool is missing, aborts cleanly before writing.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DIM = 768
SEED = 42
TOTAL = 2_000_000
HOLDOUT = 300_000               # social rows drawn from global index >= 300000

SUBSTRATES = Path("/data/latent-basemap/substrates")
_JINA_PROMPTED = SUBSTRATES / "jina-prompted"
EN_FILE = _JINA_PROMPTED / "en-2m.f16.npy"
ML_FILE = _JINA_PROMPTED / "multi-1m.f16.npy"

# EN block layout inside en-2m.f16 (mirrors _jina_multi_load exactly).
_JINA_EN_NAMES = ("fineweb-edu", "redpajama", "pile")
_JINA_EN_PER = (666_667, 666_667, 666_666)              # en-2m corpus rows
_EN_HALF_LEN = (333_334, 333_333, 333_333)              # prefixes taken for EN 1M
# offsets of each corpus block start inside en-2m.f16:
_EN_OFFS = tuple(int(o) for o in np.cumsum((0,) + _JINA_EN_PER[:-1]))  # (0,666667,1333334)
EN_1M = sum(_EN_HALF_LEN)                                # 1,000,000
ML_1M = 1_000_000

_JINA_LANGS = ("arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek",
               "fra_Latn", "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan",
               "kor_Hang", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl",
               "spa_Latn", "swe_Latn", "tha_Thai", "tur_Latn", "vie_Latn")
_LANG_SHORT = tuple(l.split("_")[0] for l in _JINA_LANGS)
LANG_PER = 50_000

# EN-1M corpus index ranges (into the reconstructed 1M EN block).
_EN_RANGES = {}
_c = 0
for _name, _p in zip(_JINA_EN_NAMES, _EN_HALF_LEN):
    _EN_RANGES[_name] = (_c, _c + _p)
    _c += _p
assert _c == EN_1M

SOCIAL = ("reddit", "CA", "twitter", "bluesky")
# pool dir per social corpus (CA -> ca-jina-pool). Pool file convention: any *.npy
# under the dir (single substrate.f16.npy in the P-A plan), gathered like a corpus.
POOL_DIR = {
    "reddit":  SUBSTRATES / "reddit-jina-pool",
    "CA":      SUBSTRATES / "ca-jina-pool",
    "twitter": SUBSTRATES / "twitter-jina-pool",
    "bluesky": SUBSTRATES / "bluesky-jina-pool",
}
SOCIAL_CODE = {"reddit": 1, "CA": 2, "twitter": 3, "bluesky": 4}
# provenance: origin 0 = base row (src = base index 0..1999999; [0:1M)=EN, [1M:2M)=ML);
# origin 1..4 = social corpus code (src = pool global row index).
PROV_DTYPE = np.dtype([("origin", "u1"), ("src", "<i8")])

# arm registry ---------------------------------------------------------------
JINA_ARMS = {
    "jina-bmix10-2m": {"share": 0.10, "kind": "balanced"},
    "jina-bmix20-2m": {"share": 0.20, "kind": "balanced"},
    "jina-bmix30-2m": {"share": 0.30, "kind": "balanced"},
    "jina-rmix20-2m": {"share": 0.20, "kind": "reddit"},
}


# -- shard IO (mirrors build_bmix10cp_2m._open_shard, f16 default) ------------

def _open_shard(path: str):
    with open(path, "rb") as fh:
        is_npy = fh.read(6) == b"\x93NUMPY"
    if is_npy:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    rows = os.path.getsize(path) // (DIM * 2)            # raw f16 fallback
    return np.memmap(path, dtype=np.float16, mode="r", shape=(rows, DIM))


def _pool_shards(corpus: str):
    d = POOL_DIR[corpus]
    files = sorted(glob.glob(os.path.join(str(d), "*.npy")))
    if not files:
        raise FileNotFoundError(f"no jina pool shards for {corpus} at {d}")
    return [_open_shard(f) for f in files]


def _sample_pool_rows(shards, n: int, rng, offset: int, dim: int = DIM):
    """Gather ``n`` uniform-random rows without replacement from the pool's global
    row space restricted to global index >= ``offset``. Returns (rows_f16, gidx,
    record). Per-shard gathers in ascending local order for memmap locality."""
    shard_rows = [int(s.shape[0]) for s in shards]
    total = int(sum(shard_rows))
    avail = total - offset
    if n > avail:
        raise ValueError(
            f"need {n} pool rows but only {avail} past offset {offset} "
            f"(pool total {total}); enlarge the pool or lower the share")
    gidx = np.sort(rng.choice(avail, size=n, replace=False)) + offset
    offsets = np.concatenate([[0], np.cumsum(shard_rows)]).astype(np.int64)
    shard_of = np.searchsorted(offsets, gidx, side="right") - 1
    local = gidx - offsets[shard_of]
    out = np.empty((n, dim), dtype=np.float16)
    for si in range(len(shards)):
        mask = shard_of == si
        if not mask.any():
            continue
        loc = local[mask]
        order = np.argsort(loc)
        dest = np.nonzero(mask)[0][order]
        out[dest] = np.asarray(shards[si][loc[order]], dtype=np.float16)
    record = {"total_rows": total, "offset": offset, "train_pool": avail,
              "sampled": n, "shard_rows": shard_rows}
    return out, gidx, record


def _sha256_bytes(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _displace_counts(D: int) -> dict:
    """Per-EN-corpus displacement counts, proportional to 333334:333333:333333;
    pile adjusted so the three sum to exactly D."""
    fw = int(round(D * _EN_HALF_LEN[0] / EN_1M))
    rp = int(round(D * _EN_HALF_LEN[1] / EN_1M))
    pile = D - fw - rp
    out = {"fineweb-edu": fw, "redpajama": rp, "pile": pile}
    assert sum(out.values()) == D, out
    assert all(v >= 0 for v in out.values()), out
    return out


def _social_counts(D: int, kind: str) -> dict:
    if kind == "reddit":
        return {"reddit": D, "CA": 0, "twitter": 0, "bluesky": 0}
    if kind == "balanced":
        if D % 4 != 0:
            raise SystemExit(f"balanced share must give D divisible by 4, got D={D}")
        per = D // 4
        return {c: per for c in SOCIAL}
    raise SystemExit(f"unknown social kind {kind!r}")


def _report_inputs(arm: str, social_counts: dict) -> dict:
    """Print + return which inputs exist vs are pending (GPU prereqs)."""
    status = {}
    print("\n[inputs] champion base:")
    for label, p in (("en-2m.f16", EN_FILE), ("multi-1m.f16", ML_FILE)):
        ok = p.exists()
        status[label] = {"path": str(p), "exists": ok}
        print(f"    {'OK ' if ok else 'MISS'} {label}: {p}")
    print("[inputs] jina social pools (GPU prereq P-A):")
    for c in SOCIAL:
        need = social_counts.get(c, 0)
        d = POOL_DIR[c]
        files = sorted(glob.glob(os.path.join(str(d), "*.npy"))) if d.exists() else []
        avail = 0
        for f in files:
            try:
                avail += int(_open_shard(f).shape[0])
            except Exception:
                pass
        ok = bool(files)
        enough = ok and (avail - HOLDOUT) >= need
        status[f"pool-{c}"] = {"dir": str(d), "exists": ok, "rows": avail,
                               "need": need, "draw_available": max(avail - HOLDOUT, 0),
                               "sufficient": bool(enough)}
        tag = "OK " if ok else "PEND"
        note = "" if need == 0 else (
            f" need {need}, draw-available {max(avail - HOLDOUT, 0)}"
            f"{'' if enough else '  <-- INSUFFICIENT' if ok else ''}")
        print(f"    {tag} {c}: {d} (rows={avail}){note}")
    return status


def main(argv: list[str]) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    arm = (argv[1] if len(argv) > 1 else os.environ.get("JINA_ARM", "")).strip()
    if arm not in JINA_ARMS:
        raise SystemExit(
            f"usage: build_jina_mix_2m.py <arm>  (or env JINA_ARM); "
            f"supported: {sorted(JINA_ARMS)}")
    spec = JINA_ARMS[arm]
    share, kind = spec["share"], spec["kind"]
    D = int(round(share * TOTAL))
    disp = _displace_counts(D)
    social = _social_counts(D, kind)
    social = {c: n for c, n in social.items() if n > 0}
    BASE_RETAINED = TOTAL - D           # retained EN + full ML
    EN_RETAINED = EN_1M - D

    print(f"=== build_jina_mix_2m {arm} (share {share:.0%}, {kind}) ===")
    print(f"    displace D={D:,} from EN 1M  -> per-corpus {disp}")
    print(f"    social replacement          -> {social}")
    print(f"    retained: EN {EN_RETAINED:,} + ML {ML_1M:,} = {BASE_RETAINED:,}")

    status = _report_inputs(arm, social)

    OUT_DIR = SUBSTRATES / arm
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = OUT_DIR / "substrate.f16.npy"
    if sub_path.exists():
        raise SystemExit(f"REFUSE overwrite: {sub_path} already exists")

    # preflight: base files + all needed pools must exist to RUN. If pending,
    # exit cleanly (0) without writing — the builder re-runs once pools land.
    missing_base = [k for k in ("en-2m.f16", "multi-1m.f16")
                    if not status[k]["exists"]]
    pending_pools = [c for c in social if not status[f"pool-{c}"]["exists"]]
    insufficient = [c for c in social
                    if status[f"pool-{c}"]["exists"]
                    and not status[f"pool-{c}"]["sufficient"]]
    if missing_base:
        raise SystemExit(f"champion base files missing: {missing_base}")
    if pending_pools:
        print(f"\n[pending] jina social pools not yet built: {pending_pools} "
              f"(GPU prereq P-A). Nothing written; re-run once pools exist.")
        return 0
    if insufficient:
        raise SystemExit(
            f"jina social pools too small (offset {HOLDOUT}) for {arm}: "
            f"{ {c: status[f'pool-{c}'] for c in insufficient} }")

    t0 = time.time()
    rng = np.random.default_rng(SEED)   # single rng(42) threaded through all draws

    # --- reconstruct the champion EN 1M block (f16, bit-identical prefixes) ---
    en_mm = np.load(EN_FILE, mmap_mode="r", allow_pickle=False)
    assert en_mm.dtype == np.float16 and en_mm.shape[1] == DIM, en_mm.shape
    en_segs = [np.asarray(en_mm[o:o + p], dtype=np.float16)
               for o, p in zip(_EN_OFFS, _EN_HALF_LEN)]
    en1m = np.concatenate(en_segs)      # (1M,768) f16
    assert en1m.shape == (EN_1M, DIM), en1m.shape
    del en_segs
    ml_mm = np.load(ML_FILE, mmap_mode="r", allow_pickle=False)
    assert ml_mm.shape == (ML_1M, DIM) and ml_mm.dtype == np.float16, ml_mm.shape

    # --- step 1: choose EN-1M rows to DISPLACE, per corpus, proportionally ----
    disp_mask = np.zeros(EN_1M, dtype=bool)
    displaced_per_corpus = {}
    for corpus in _JINA_EN_NAMES:       # rng order: fineweb, redpajama, pile
        lo, hi = _EN_RANGES[corpus]
        k = disp[corpus]
        pick = rng.choice(hi - lo, size=k, replace=False)   # local idx
        disp_mask[lo + pick] = True
        displaced_per_corpus[corpus] = int(k)
    retained_en = np.nonzero(~disp_mask)[0]                  # sorted ascending
    assert retained_en.size == EN_RETAINED, retained_en.size
    ret_en_per = {c: int((~disp_mask[_EN_RANGES[c][0]:_EN_RANGES[c][1]]).sum())
                  for c in _JINA_EN_NAMES}
    assert {c: _EN_HALF_LEN[i] - disp[c] for i, c in enumerate(_JINA_EN_NAMES)} == ret_en_per
    print(f"[displace] per-corpus displaced: {displaced_per_corpus}")
    print(f"[retain]   per-corpus EN retained: {ret_en_per}")

    # --- output buffers -------------------------------------------------------
    out = np.lib.format.open_memmap(str(sub_path), mode="w+",
                                    dtype=np.float16, shape=(TOTAL, DIM))
    prov = np.empty(TOTAL, dtype=PROV_DTYPE)
    subsets = np.empty(TOTAL, dtype=object)
    CH = 200_000

    # fill [0:EN_RETAINED) retained EN (bit-identical values+labels) -----------
    for i in range(0, EN_RETAINED, CH):
        idx = retained_en[i:i + CH]
        out[i:i + idx.size] = en1m[idx]
        prov["origin"][i:i + idx.size] = 0
        prov["src"][i:i + idx.size] = idx            # base EN index (0..1M-1)
    # EN corpus labels for retained rows (from EN-1M ranges)
    en_label = np.empty(EN_1M, dtype=object)
    for c in _JINA_EN_NAMES:
        lo, hi = _EN_RANGES[c]
        en_label[lo:hi] = c
    subsets[:EN_RETAINED] = en_label[retained_en]

    # bit-identical proof (pre-shuffle): retained EN block == champion EN[retained]
    for i in range(0, EN_RETAINED, CH):
        idx = retained_en[i:i + CH]
        assert np.array_equal(out[i:i + idx.size], en1m[idx])
    print(f"[matched] {EN_RETAINED:,} EN rows == champion-EN[retained] verified")

    # fill [EN_RETAINED:EN_RETAINED+1M) ML block (bit-identical, untouched) -----
    ml_start = EN_RETAINED
    for i in range(0, ML_1M, CH):
        out[ml_start + i:ml_start + i + CH] = ml_mm[i:i + CH]
        prov["origin"][ml_start + i:ml_start + i + CH] = 0
        prov["src"][ml_start + i:ml_start + i + CH] = np.arange(
            EN_1M + i, EN_1M + min(i + CH, ML_1M))    # base index 1M..2M-1
    subsets[ml_start:ml_start + ML_1M] = np.repeat(_LANG_SHORT, LANG_PER)
    # bit-identical proof (pre-shuffle): ML block == multi-1m in full.
    for i in range(0, ML_1M, CH):
        j = min(i + CH, ML_1M)
        assert np.array_equal(out[ml_start + i:ml_start + j], ml_mm[i:j])
    print(f"[ml] {ML_1M:,} language rows copied bit-identical (20 x {LANG_PER:,})")

    # fill social ------------------------------------------------------------
    pos = ml_start + ML_1M
    social_records = {}
    social_min_gidx = {}
    for corpus in SOCIAL:               # fixed rng order reddit,CA,twitter,bluesky
        n = social.get(corpus, 0)
        if n == 0:
            continue
        shards = _pool_shards(corpus)
        rows, gidx, rec = _sample_pool_rows(shards, n, rng, offset=HOLDOUT)
        assert gidx.min() >= HOLDOUT, (corpus, int(gidx.min()))
        out[pos:pos + n] = rows
        prov["origin"][pos:pos + n] = SOCIAL_CODE[corpus]
        prov["src"][pos:pos + n] = gidx.astype(np.int64)
        subsets[pos:pos + n] = corpus
        social_records[corpus] = rec
        social_min_gidx[corpus] = int(gidx.min())
        pos += n
        print(f"    {corpus}: +{n:,} rows (pool {rec['train_pool']:,} of "
              f"{rec['total_rows']:,}, offset {HOLDOUT:,}, min_gidx "
              f"{social_min_gidx[corpus]:,})", flush=True)
        del rows, shards
    assert pos == TOTAL, pos

    # --- shuffle (same rng(42) permutation) -----------------------------------
    perm = rng.permutation(TOTAL)
    subsets = subsets[perm]
    prov = prov[perm]
    tmp = sub_path.with_suffix(".shuf.tmp.npy")
    shuf = np.lib.format.open_memmap(str(tmp), mode="w+",
                                     dtype=np.float16, shape=(TOTAL, DIM))
    for i in range(0, TOTAL, CH):
        shuf[i:i + CH] = out[perm[i:i + CH]]
    shuf.flush()
    del out, shuf
    os.replace(tmp, sub_path)
    np.save(OUT_DIR / "subsets.npy", subsets)
    np.save(OUT_DIR / "provenance.npy", prov)

    # --- verification ---------------------------------------------------------
    a = np.load(sub_path, mmap_mode="r")
    assert a.shape == (TOTAL, DIM) and a.dtype == np.float16
    finite = True
    for i in range(0, TOTAL, CH):
        if not np.isfinite(a[i:i + CH]).all():
            finite = False
            break
    assert finite, "non-finite rows in output substrate"

    ss = np.load(OUT_DIR / "subsets.npy", allow_pickle=True)
    uniq, cnts = np.unique(ss, return_counts=True)
    comp = {str(u): int(c) for u, c in zip(uniq, cnts)}
    pv = np.load(OUT_DIR / "provenance.npy", allow_pickle=False)

    # PROOF 1 (language preservation): per-language counts ALL == 50000.
    lang_counts = {s: int(comp.get(s, 0)) for s in _LANG_SHORT}
    lang_ok = all(v == LANG_PER for v in lang_counts.values())
    assert lang_ok, lang_counts
    # ML rows also bit-identical: base rows (origin 0) with src>=1M number exactly 1M.
    base_after = pv["origin"] == 0
    ml_after = int(((pv["origin"] == 0) & (pv["src"] >= EN_1M)).sum())
    assert ml_after == ML_1M, ml_after

    # PROOF 2 (EN displaced proportionally): retained EN per-corpus counts.
    en_ret_after = {c: int((ss == c).sum()) for c in _JINA_EN_NAMES}
    assert en_ret_after == ret_en_per, (en_ret_after, ret_en_per)
    # retained EN src-set (origin 0, src<1M) equals retained_en.
    en_src_after = np.sort(pv["src"][(pv["origin"] == 0) & (pv["src"] < EN_1M)])
    matched_ok = bool(np.array_equal(en_src_after, retained_en))
    assert matched_ok, "retained-EN provenance != champion EN[retained]"

    # PROOF 3 (social holdout): every social row global index >= 300000.
    social_min_after = {}
    for c in social:
        code = SOCIAL_CODE[c]
        g = pv["src"][pv["origin"] == code]
        assert (ss[pv["origin"] == code] == c).all()
        social_min_after[c] = int(g.min())
        assert g.min() >= HOLDOUT, (c, int(g.min()))
    social_total_after = int((pv["origin"] >= 1).sum())
    assert social_total_after == D, social_total_after

    displaced_share = {c: disp[c] / _EN_HALF_LEN[i]
                       for i, c in enumerate(_JINA_EN_NAMES)}

    manifest = {
        "name": arm,
        "kind": f"jina-language-preserving-mix ({kind})",
        "role": ("jina LANGUAGE-PRESERVING social-mixture substrate "
                 "(JINA_SWEEP_PROPOSAL.md 2026-08-28): social displaces ONLY the "
                 "three EN corpora, PROPORTIONALLY; ALL 20 per-language blocks are "
                 "held BIT-IDENTICAL to the 0% baseline (the language-preservation "
                 "proof, mirroring bmix10cp's starcoder==200k). The 0% arm is the "
                 "EXISTING jina-multi-2m/champion-bs16k (its substrate IS the "
                 "undisplaced base); only the mixed arm is built here."),
        "social_share": share, "social_kind": kind,
        "rows": TOTAL, "dim": DIM, "dtype": "float16", "seed": SEED,
        "shuffled": True,
        "base": {"source": "image_map_pipeline._jina_multi_load composition",
                 "en_1m": {n: p for n, p in zip(_JINA_EN_NAMES, _EN_HALF_LEN)},
                 "ml_1m": {"langs": list(_JINA_LANGS), "per_lang": LANG_PER},
                 "en_file": str(EN_FILE), "ml_file": str(ML_FILE)},
        "rng_threading": ("single np.random.default_rng(42): (1) per-EN-corpus "
                          "choice of rows to displace in order fineweb,redpajama,"
                          "pile, (2) social draws in order reddit,CA,twitter,"
                          "bluesky, (3) final permutation"),
        "displace_D": D,
        "displaced_per_corpus": displaced_per_corpus,
        "displaced_share_of_corpus": displaced_share,
        "social_counts": social,
        "composition_counts": comp,
        "provenance_dtype": "[('origin','u1'),('src','<i8')]  origin 0=base(src=base idx),1..4=social(src=pool gidx)",
        "social_codes": SOCIAL_CODE,
        "finite": bool(finite),
        "language_preservation_proof": {
            "per_language_counts": lang_counts,
            "all_equal_baseline_50000": bool(lang_ok),
            "ml_rows_bit_identical": int(ml_after),
            "note": ("every per-language block count == 50,000, identical to the "
                     "0% baseline; the full 1M ML block copied bit-identical "
                     "(analogous to bmix10cp starcoder==200k)."),
        },
        "en_proportional_displacement_proof": {
            "en_baseline_counts": {n: p for n, p in zip(_JINA_EN_NAMES, _EN_HALF_LEN)},
            "displaced_per_corpus": displaced_per_corpus,
            "displaced_share_of_corpus": displaced_share,
            "retained_per_corpus": en_ret_after,
            "proportional_to": "333334:333333:333333 (fineweb:redpajama:pile)",
            "note": "displaced_share ~ equal across the three EN corpora; ML share 0.",
        },
        "matched_row_proof": {
            "retained_en_rows": EN_RETAINED,
            "displaced_en_rows": D,
            "retained_en_indices_sha256": _sha256_bytes(retained_en.astype(np.uint32)),
            "retained_en_bit_identical_to_champion": True,
            "retained_en_provenance_set_equals_champion_retained": matched_ok,
            "method": ("retained EN rows copied verbatim (f16 values) from the "
                       "champion EN 1M block by index; equality verified "
                       "pre-shuffle (values) and post-shuffle (provenance src set)."),
        },
        "social_holdout_proof": {
            "holdout_rows": HOLDOUT,
            "rule": "every social row global index >= 300000 (offset-restricted sampler)",
            "min_global_index_at_draw": social_min_gidx,
            "min_global_index_after_shuffle": social_min_after,
            "social_total_rows": social_total_after,
            "all_ge_holdout": True,
        },
        "social_per_corpus": social_records,
        "outputs": {
            "substrate": str(sub_path),
            "subsets": str(OUT_DIR / "subsets.npy"),
            "provenance": str(OUT_DIR / "provenance.npy"),
        },
        "input_status": status,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n[{arm}] built shape={a.shape} dtype={a.dtype} finite={finite} in "
          f"{(time.time()-t0)/60:.1f} min -> {sub_path}")
    print(f"[composition] {comp}")
    print(f"[lang-preserved] all 20 == 50000: {lang_ok}")
    print(f"[en-proportional] displaced shares: "
          f"{ {k: round(v,4) for k,v in displaced_share.items()} }")
    print(f"[matched] retained-EN provenance == champion[retained]: {matched_ok}")
    print(f"[social>=300000] after-shuffle mins: {social_min_after}")
    del a
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
