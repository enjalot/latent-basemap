#!/usr/bin/env python3
"""Build the `minilm-bmix10cp-2m` substrate: the CODE-PRESERVING social probe
(owner 2026-08-28). 10% BALANCED social (reddit/CA/twitter/bluesky) added at 2M
with MATCHED rows against the 2M baseline, BUT — unlike bmix30 — the displaced
rows are drawn ONLY from fineweb/redpajama/pile, PROPORTIONALLY to their baseline
counts (800:500:500). STARCODER IS UNTOUCHED: all 200,000 baseline code rows are
retained. The whole point is a matched contrast where the social 10% is paid for
entirely out of the web/pile budget while the code budget is PRESERVED.

MATCHED-ROW SPEC
----------------
Start from the 2M baseline substrate + provenance:
  /data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/
    minilm-mixed-2m-substrate-and-exact-k15-graph-v1/{substrate.f32.npy,provenance.npy}
  Baseline composition: fineweb 800k / redpajama 500k / pile 500k / starcoder 200k
  (provenance corpus codes 0/1/2/3).

  1. Deterministically (rng=default_rng(42)) choose 200,000 baseline rows to
     DISPLACE, drawn ONLY from fineweb/redpajama/pile, PROPORTIONALLY to their
     baseline counts (800:500:500 -> displace fineweb 88,889 / redpajama 55,556 /
     pile 55,555; the last adjusted so the sum is exactly 200,000). STARCODER
     contributes 0 displacements. The remaining 1,800,000 rows (fineweb 711,111 /
     redpajama 444,444 / pile 444,445 / starcoder 200,000) are RETAINED verbatim —
     their embedding values AND their (corpus,shard,row) provenance are copied
     straight from the baseline (they are the SAME rows).
  2. Draw 200,000 balanced social rows = 50,000 each of reddit / CA / twitter /
     bluesky, holdout-disjoint (global offset >= 300000), without replacement,
     from the SAME rng(42) threaded in FILL_ORDER.
  3. Assemble 1.8M retained base + 200k social = 2,000,000 rows, then shuffle
     (same rng(42) permutation). Output float32 (2_000_000, 384).

Baseline provenance corpus codes (u1): 0=fineweb 1=redpajama 2=pile 3=starcoder.
Social codes extend this: 4=reddit 5=CA 6=twitter 7=bluesky.

subsets.npy label convention: retained base rows -> "base"; social rows -> their
corpus name (reddit / CA / twitter / bluesky). The per-base-corpus breakdown is
still recoverable from provenance codes 0-3.

Write-once (refuses to overwrite substrate.f32.npy). CPU-only. ~1-2 GB RAM via
memmap gather + chunked shuffle.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

DIM = 384
SEED = 42
TOTAL = 2_000_000
SOCIAL_TOTAL = 200_000            # 10% of 2M
PER_SOCIAL = SOCIAL_TOTAL // 4    # 50,000 each of 4 social corpora
BASE_RETAINED = TOTAL - SOCIAL_TOTAL  # 1,800,000
HOLDOUT = 300_000

# per-base-corpus DISPLACEMENT counts, proportional to baseline 800:500:500
# (fineweb/redpajama/pile). starcoder is UNTOUCHED (0 displacement). The last
# entry (pile) is adjusted so the three sum to exactly SOCIAL_TOTAL (200,000).
DISPLACE_PER = {"fineweb": 88_889, "redpajama": 55_556, "pile": 55_555}
assert sum(DISPLACE_PER.values()) == SOCIAL_TOTAL, sum(DISPLACE_PER.values())
# expected retained per base corpus (baseline counts minus displacements).
BASELINE_BASE_COUNTS = {"fineweb": 800_000, "redpajama": 500_000,
                        "pile": 500_000, "starcoder": 200_000}
RETAINED_PER = {c: BASELINE_BASE_COUNTS[c] - DISPLACE_PER.get(c, 0)
                for c in BASELINE_BASE_COUNTS}
assert sum(RETAINED_PER.values()) == BASE_RETAINED, sum(RETAINED_PER.values())

SUBSTRATES = Path("/data/latent-basemap/substrates")
OUT_DIR = SUBSTRATES / "minilm-bmix10cp-2m"

BASELINE_DIR = Path(
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
BASELINE_SUB = BASELINE_DIR / "substrate.f32.npy"
BASELINE_PROV = BASELINE_DIR / "provenance.npy"

E = "/data/embeddings"
CORPORA = {
    "reddit":  f"{E}/reddit-tldr17-chunked-120-all-MiniLM-L6-v2/train",
    "CA":      f"{E}/communityarchive-tweets-all-MiniLM-L6-v2/train",
    "twitter": f"{E}/twitter100m-chunked-120-all-MiniLM-L6-v2/train",
    "bluesky": f"{E}/bluesky-5m-chunked-120-all-MiniLM-L6-v2/train",
}
SOCIAL = ("reddit", "CA", "twitter", "bluesky")
# base displacement order (rng threaded in this order): fineweb, redpajama, pile.
BASE_DISPLACE_ORDER = ("fineweb", "redpajama", "pile")
# provenance corpus codes: base codes MATCH the 2M baseline; social codes extend.
CORPUS_CODE = {"fineweb": 0, "redpajama": 1, "pile": 2, "starcoder": 3,
               "reddit": 4, "CA": 5, "twitter": 6, "bluesky": 7}

PROV_DTYPE = np.dtype([("corpus", "u1"), ("shard", "<u2"), ("row", "<i8")])


# -- shard IO (mirrors build_mixture_substrates._open_shard) ------------------

def _open_shard(path: str):
    with open(path, "rb") as fh:
        is_npy = fh.read(6) == b"\x93NUMPY"
    if is_npy:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    rows = os.path.getsize(path) // (DIM * 4)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(rows, DIM))


def _corpus_shards(corpus: str):
    files = sorted(glob.glob(os.path.join(CORPORA[corpus], "*.npy")))
    if not files:
        raise FileNotFoundError(f"no shards for corpus {corpus} at {CORPORA[corpus]}")
    return [_open_shard(f) for f in files]


# -- offset-aware sampler that also returns global indices --------------------

def _sample_corpus_rows(shards, n: int, rng, offset: int, dim: int = DIM):
    """Gather ``n`` uniform-random rows without replacement from the global row
    space of ``shards`` restricted to global index >= ``offset`` (holdout front
    slice). Returns (rows, gidx, shard_of, local, record). Per-shard gathers in
    ascending local order for memmap locality (matches the reference)."""
    shard_rows = [int(s.shape[0]) for s in shards]
    total = int(sum(shard_rows))
    avail = total - offset
    if n > avail:
        raise ValueError(f"need {n} rows but only {avail} past offset {offset}")
    gidx = np.sort(rng.choice(avail, size=n, replace=False)) + offset
    offsets = np.concatenate([[0], np.cumsum(shard_rows)]).astype(np.int64)
    shard_of = np.searchsorted(offsets, gidx, side="right") - 1
    local = gidx - offsets[shard_of]
    out = np.empty((n, dim), dtype=np.float32)
    for si in range(len(shards)):
        mask = shard_of == si
        if not mask.any():
            continue
        loc = local[mask]
        order = np.argsort(loc)
        dest = np.nonzero(mask)[0][order]
        out[dest] = np.asarray(shards[si][loc[order]], dtype=np.float32)
    record = {"total_rows": total, "offset": offset, "train_pool": avail,
              "sampled": n, "shard_rows": shard_rows}
    return out, gidx, shard_of, local, record


def _sha256_bytes(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = OUT_DIR / "substrate.f32.npy"
    if sub_path.exists():
        raise SystemExit(f"REFUSE overwrite: {sub_path} already exists")

    t0 = time.time()
    rng = np.random.default_rng(SEED)  # single rng(42) threaded through all draws

    # --- baseline memmaps -----------------------------------------------------
    base_sub = np.load(BASELINE_SUB, mmap_mode="r", allow_pickle=False)
    base_prov = np.load(BASELINE_PROV, mmap_mode="r", allow_pickle=False)
    assert base_sub.shape == (TOTAL, DIM), base_sub.shape
    assert base_sub.dtype == np.float32, base_sub.dtype
    assert base_prov.shape == (TOTAL,) and base_prov.dtype == PROV_DTYPE

    # --- step 1: choose baseline rows to DISPLACE, per base corpus,
    #     PROPORTIONALLY to baseline counts; STARCODER untouched. -------------
    disp_mask = np.zeros(TOTAL, dtype=bool)
    displaced_per_corpus = {}
    for corpus in BASE_DISPLACE_ORDER:  # fineweb, redpajama, pile (rng order)
        code = CORPUS_CODE[corpus]
        corpus_idx = np.nonzero(np.asarray(base_prov["corpus"]) == code)[0]
        assert corpus_idx.size == BASELINE_BASE_COUNTS[corpus], (
            corpus, corpus_idx.size)
        k = DISPLACE_PER[corpus]
        pick = rng.choice(corpus_idx.size, size=k, replace=False)  # local idx
        disp_mask[corpus_idx[pick]] = True
        displaced_per_corpus[corpus] = int(k)
    # starcoder (code 3): 0 displacements -> fully retained.
    displaced_per_corpus["starcoder"] = 0

    displaced = np.nonzero(disp_mask)[0]
    retained = np.nonzero(~disp_mask)[0]            # sorted ascending
    assert displaced.size == SOCIAL_TOTAL, displaced.size
    assert retained.size == BASE_RETAINED, retained.size

    # code-untouched proof: no starcoder (code 3) row is displaced.
    sc_code = CORPUS_CODE["starcoder"]
    disp_codes = np.asarray(base_prov["corpus"])[displaced]
    assert not (disp_codes == sc_code).any(), "starcoder rows were displaced!"
    ret_codes = np.asarray(base_prov["corpus"])[retained]
    sc_retained = int((ret_codes == sc_code).sum())
    assert sc_retained == BASELINE_BASE_COUNTS["starcoder"], sc_retained
    # per-corpus retained counts (from baseline provenance among retained rows).
    retained_per_observed = {}
    for corpus, code in (("fineweb", 0), ("redpajama", 1), ("pile", 2),
                         ("starcoder", 3)):
        retained_per_observed[corpus] = int((ret_codes == code).sum())
    assert retained_per_observed == RETAINED_PER, (retained_per_observed,
                                                   RETAINED_PER)
    print(f"[displace] per-corpus displaced: {displaced_per_corpus}")
    print(f"[retain]   per-corpus retained:  {retained_per_observed}")
    print(f"[code]     starcoder retained {sc_retained} == baseline "
          f"{BASELINE_BASE_COUNTS['starcoder']} (UNTOUCHED)")

    # --- output buffers -------------------------------------------------------
    out = np.lib.format.open_memmap(str(sub_path), mode="w+",
                                    dtype=np.float32, shape=(TOTAL, DIM))
    prov = np.empty(TOTAL, dtype=PROV_DTYPE)
    subsets = np.empty(TOTAL, dtype=object)

    # --- fill 0:1.8M with retained baseline rows (verbatim values+provenance) -
    CH = 200_000
    for i in range(0, BASE_RETAINED, CH):
        idx = retained[i:i + CH]
        out[i:i + idx.size] = base_sub[idx]
        prov[i:i + idx.size] = base_prov[idx]
    subsets[:BASE_RETAINED] = "base"

    # matched-subset assertion: base block bit-identical to baseline[retained]
    for i in range(0, BASE_RETAINED, CH):
        idx = retained[i:i + CH]
        assert np.array_equal(out[i:i + idx.size], base_sub[idx])
    print(f"[matched] 1,800,000 base rows == baseline[retained] verified")

    # --- fill 1.8M:2M with balanced social, offset>=300000 --------------------
    pos = BASE_RETAINED
    social_records = {}
    social_min_gidx = {}
    for corpus in SOCIAL:
        shards = _corpus_shards(corpus)
        rows, gidx, shard_of, local, rec = _sample_corpus_rows(
            shards, PER_SOCIAL, rng, offset=HOLDOUT)
        assert gidx.min() >= HOLDOUT, (corpus, int(gidx.min()))
        out[pos:pos + PER_SOCIAL] = rows
        prov["corpus"][pos:pos + PER_SOCIAL] = CORPUS_CODE[corpus]
        prov["shard"][pos:pos + PER_SOCIAL] = shard_of.astype(np.uint16)
        prov["row"][pos:pos + PER_SOCIAL] = local.astype(np.int64)
        subsets[pos:pos + PER_SOCIAL] = corpus
        social_records[corpus] = rec
        social_min_gidx[corpus] = int(gidx.min())
        pos += PER_SOCIAL
        print(f"    {corpus}: +{PER_SOCIAL:,} rows (pool {rec['train_pool']:,} "
              f"of {rec['total_rows']:,}, offset {HOLDOUT:,}, "
              f"min_gidx {social_min_gidx[corpus]:,})", flush=True)
        del rows, shards
    assert pos == TOTAL, pos

    # --- shuffle (same rng(42) permutation), chunked to temp memmap -----------
    perm = rng.permutation(TOTAL)
    subsets = subsets[perm]
    prov = prov[perm]
    tmp = sub_path.with_suffix(".shuf.tmp.npy")
    shuf = np.lib.format.open_memmap(str(tmp), mode="w+",
                                     dtype=np.float32, shape=(TOTAL, DIM))
    for i in range(0, TOTAL, CH):
        shuf[i:i + CH] = out[perm[i:i + CH]]
    shuf.flush()
    del out, shuf
    os.replace(tmp, sub_path)
    np.save(OUT_DIR / "subsets.npy", subsets)
    np.save(OUT_DIR / "provenance.npy", prov)

    # --- verification ---------------------------------------------------------
    a = np.load(sub_path, mmap_mode="r")
    assert a.shape == (TOTAL, DIM) and a.dtype == np.float32
    finite = True
    for i in range(0, TOTAL, CH):
        if not np.isfinite(a[i:i + CH]).all():
            finite = False
            break
    assert finite, "non-finite rows in output substrate"
    ss = np.load(OUT_DIR / "subsets.npy", allow_pickle=True)
    uniq, cnts = np.unique(ss, return_counts=True)
    comp = {str(u): int(c) for u, c in zip(uniq, cnts)}
    assert comp.get("base") == BASE_RETAINED
    for c in SOCIAL:
        assert comp.get(c) == PER_SOCIAL, (c, comp.get(c))

    # matched-subset proof after shuffle: base-row provenance set == baseline
    # provenance at retained indices (as sets of (corpus,shard,row) tuples).
    base_after = ss == "base"
    pv = np.load(OUT_DIR / "provenance.npy", allow_pickle=False)
    got = pv[base_after]
    exp = base_prov[retained]
    # robust set-equality via sorted structured compare
    def _sortkey(x):
        return np.lexsort((x['row'], x['shard'], x['corpus']))
    g = got[_sortkey(got)]
    e = np.asarray(exp)[_sortkey(np.asarray(exp))]
    matched_ok = bool(np.array_equal(g['corpus'], e['corpus']) and
                      np.array_equal(g['shard'], e['shard']) and
                      np.array_equal(g['row'], e['row']))
    assert matched_ok, "base-row provenance does not match baseline[retained]"

    # code-untouched proof after shuffle: all 200,000 starcoder base rows present.
    sc_after = int((pv[base_after]['corpus'] == sc_code).sum())
    assert sc_after == BASELINE_BASE_COUNTS["starcoder"], sc_after
    # per-base-corpus retained counts after shuffle (from provenance codes 0-3).
    retained_per_after = {}
    for corpus, code in (("fineweb", 0), ("redpajama", 1), ("pile", 2),
                         ("starcoder", 3)):
        retained_per_after[corpus] = int((pv[base_after]['corpus'] == code).sum())
    assert retained_per_after == RETAINED_PER, (retained_per_after, RETAINED_PER)

    # social>=300000 proof after shuffle, per corpus, via reconstructed gidx
    social_shard_offsets = {}
    for c in SOCIAL:
        sr = social_records[c]["shard_rows"]
        social_shard_offsets[c] = np.concatenate([[0], np.cumsum(sr)]).astype(np.int64)
    social_min_after = {}
    for c in SOCIAL:
        m = ss == c
        code = CORPUS_CODE[c]
        rows_c = pv[m]
        assert (rows_c['corpus'] == code).all()
        g = social_shard_offsets[c][rows_c['shard']] + rows_c['row']
        social_min_after[c] = int(g.min())
        assert g.min() >= HOLDOUT, (c, int(g.min()))

    # proportional-displacement proof: displaced/baseline share equal across
    # fineweb/redpajama/pile (within rounding), starcoder share == 0.
    displaced_share = {c: DISPLACE_PER.get(c, 0) / BASELINE_BASE_COUNTS[c]
                       for c in BASELINE_BASE_COUNTS}

    manifest = {
        "name": "minilm-bmix10cp-2m",
        "kind": "bmix-code-preserving",
        "role": ("CODE-PRESERVING social probe (owner 2026-08-28): 10% BALANCED "
                 "social at 2M with MATCHED rows vs the minilm-mixed-2m baseline, "
                 "where the displaced 200k are drawn ONLY from fineweb/redpajama/"
                 "pile PROPORTIONALLY (800:500:500) and STARCODER IS UNTOUCHED "
                 "(all 200k code rows retained). The social 10% is paid entirely "
                 "from the web/pile budget; the code budget is preserved."),
        "social_share": 0.10,
        "rows": TOTAL, "dim": DIM, "dtype": "float32", "seed": SEED,
        "shuffled": True,
        "rng_threading": ("single np.random.default_rng(42): (1) per-corpus "
                          "choice of baseline rows to displace in order "
                          "fineweb,redpajama,pile (starcoder: none), (2) social "
                          "draws in order reddit,CA,twitter,bluesky, (3) final "
                          "permutation"),
        "base_mix_note": ("base rows are the baseline's fineweb/redpajama/pile/"
                          "starcoder rows minus the proportional web/pile "
                          "displacement; recoverable from provenance codes 0-3."),
        "corpus_codes": CORPUS_CODE,
        "composition_counts": {
            "base": BASE_RETAINED,
            "reddit": PER_SOCIAL, "CA": PER_SOCIAL,
            "twitter": PER_SOCIAL, "bluesky": PER_SOCIAL,
        },
        "base_corpus_retained_counts": RETAINED_PER,
        "base_corpus_displaced_counts": displaced_per_corpus,
        "subsets_observed": comp,
        "finite": bool(finite),
        "code_preserved_proof": {
            "starcoder_baseline_count": BASELINE_BASE_COUNTS["starcoder"],
            "starcoder_displaced": displaced_per_corpus["starcoder"],
            "starcoder_retained_at_draw": sc_retained,
            "starcoder_retained_after_shuffle": sc_after,
            "starcoder_untouched": bool(
                displaced_per_corpus["starcoder"] == 0 and
                sc_after == BASELINE_BASE_COUNTS["starcoder"]),
            "note": ("all 200,000 baseline starcoder rows retained; 0 displaced "
                     "-> code budget identical to baseline."),
        },
        "proportional_displacement_proof": {
            "baseline_base_counts": BASELINE_BASE_COUNTS,
            "displaced_per_corpus": displaced_per_corpus,
            "displaced_share_of_corpus": displaced_share,
            "displaced_only_from": list(BASE_DISPLACE_ORDER),
            "proportional_to": "800:500:500 (fineweb:redpajama:pile)",
            "note": ("displaced_share ~0.1111 for each of fineweb/redpajama/pile "
                     "(equal fraction -> proportional); starcoder share 0.0. "
                     "pile rounded down so the three sum to exactly 200,000."),
            "retained_per_corpus_after_shuffle": retained_per_after,
        },
        "matched_row_proof": {
            "baseline_dir": str(BASELINE_DIR),
            "retained_baseline_rows": BASE_RETAINED,
            "displaced_baseline_rows": SOCIAL_TOTAL,
            "retained_is_strict_subset_of_2M": True,
            "retained_indices_sha256": _sha256_bytes(retained.astype(np.uint32)),
            "displaced_indices_sha256": _sha256_bytes(np.sort(displaced).astype(np.uint32)),
            "base_block_bit_identical_to_baseline_retained": True,
            "base_provenance_set_equals_baseline_retained": matched_ok,
            "method": ("base rows copied verbatim (values+provenance) from "
                       "baseline[retained]; equality verified pre- and "
                       "post-shuffle."),
        },
        "social_holdout_proof": {
            "holdout_rows": HOLDOUT,
            "rule": "every social row global index >= 300000 (offset-restricted sampler)",
            "min_global_index_at_draw": social_min_gidx,
            "min_global_index_after_shuffle": social_min_after,
            "all_ge_holdout": True,
        },
        "social_per_corpus": social_records,
        "holdout_note": (
            "The 4 social corpora reserve their FIRST 300,000 global rows as the "
            "probe OOD holdout; all social rows here drawn from global offset "
            ">= 300000. See social-holdout-partition.json."),
        "provenance_dtype": "[('corpus','u1'),('shard','<u2'),('row','<i8')]",
        "outputs": {
            "substrate": str(sub_path),
            "subsets": str(OUT_DIR / "subsets.npy"),
            "provenance": str(OUT_DIR / "provenance.npy"),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n[minilm-bmix10cp-2m] built shape={a.shape} dtype={a.dtype} "
          f"finite={finite} in {(time.time()-t0)/60:.1f} min -> {sub_path}")
    print(f"[composition] {comp}")
    print(f"[code-preserved] starcoder retained after shuffle: {sc_after} "
          f"(displaced {displaced_per_corpus['starcoder']})")
    print(f"[proportional] displaced shares: "
          f"{ {k: round(v,4) for k,v in displaced_share.items()} }")
    print(f"[matched] base provenance == baseline[retained]: {matched_ok}")
    print(f"[social>=300000] after-shuffle mins: {social_min_after}")
    del a
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
