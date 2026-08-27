#!/usr/bin/env python3
"""Build the `minilm-bmix30-2m` substrate: FINALIST CONFIRMATION of the
social-mixture sweep winner (bmix30 = 30% balanced social) at 2M rows with
MATCHED rows against the 2M baseline, so the only difference vs the baseline is
the social 30% (a clean matched contrast).

MATCHED-ROW SPEC
----------------
Start from the 2M baseline substrate + provenance:
  /data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/
    minilm-mixed-2m-substrate-and-exact-k15-graph-v1/{substrate.f32.npy,provenance.npy}

  1. Deterministically (rng=default_rng(42)) choose 600,000 (30%) of the
     2,000,000 baseline rows to DISPLACE. The remaining 1,400,000 rows are
     RETAINED verbatim -- their embedding values AND their (corpus,shard,row)
     provenance are copied straight from the baseline (they are the SAME rows).
  2. Draw 600,000 balanced social rows = 150,000 each of reddit / CA / twitter /
     bluesky, holdout-disjoint (global offset >= 300000), without replacement,
     from the SAME rng(42) threaded in FILL_ORDER.
  3. Assemble 1.4M retained base + 600k social = 2,000,000 rows, then shuffle
     (same rng(42) permutation). Output float32 (2_000_000, 384).

Baseline provenance corpus codes (u1): 0=fineweb 1=redpajama 2=pile 3=starcoder
(decoded from counts 800k/500k/500k/200k and shard spans 98/150/177/20). Social
codes extend this: 4=reddit 5=CA 6=twitter 7=bluesky.

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
SOCIAL_TOTAL = 600_000            # 30% of 2M
PER_SOCIAL = SOCIAL_TOTAL // 4    # 150,000 each of 4 social corpora
BASE_RETAINED = TOTAL - SOCIAL_TOTAL  # 1,400,000
HOLDOUT = 300_000

SUBSTRATES = Path("/data/latent-basemap/substrates")
OUT_DIR = SUBSTRATES / "minilm-bmix30-2m"

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

    # --- step 1: choose 600k baseline rows to DISPLACE; retain the rest -------
    displaced = rng.choice(TOTAL, size=SOCIAL_TOTAL, replace=False)
    disp_mask = np.zeros(TOTAL, dtype=bool)
    disp_mask[displaced] = True
    retained = np.nonzero(~disp_mask)[0]            # sorted ascending
    assert retained.size == BASE_RETAINED, retained.size
    assert np.unique(displaced).size == SOCIAL_TOTAL

    # --- output buffers -------------------------------------------------------
    out = np.lib.format.open_memmap(str(sub_path), mode="w+",
                                    dtype=np.float32, shape=(TOTAL, DIM))
    prov = np.empty(TOTAL, dtype=PROV_DTYPE)
    subsets = np.empty(TOTAL, dtype=object)

    # --- fill 0:1.4M with retained baseline rows (verbatim values+provenance) -
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
    print(f"[matched] 1,400,000 base rows == baseline[retained] verified")

    # --- fill 1.4M:2M with balanced social, offset>=300000 --------------------
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

    manifest = {
        "name": "minilm-bmix30-2m",
        "kind": "bmix",
        "role": ("FINALIST CONFIRMATION of the social-mixture sweep winner "
                 "(bmix30 = 30% balanced social) at 2M with MATCHED rows vs the "
                 "2M baseline; the only difference vs baseline is the social 30%."),
        "social_share": 0.30,
        "rows": TOTAL, "dim": DIM, "dtype": "float32", "seed": SEED,
        "shuffled": True,
        "rng_threading": ("single np.random.default_rng(42): (1) choice of 600k "
                          "baseline rows to displace, (2) social draws in order "
                          "reddit,CA,twitter,bluesky, (3) final permutation"),
        "base_mix_note": ("base rows are the baseline's own 40/25/25/10 mix "
                          "(fineweb/redpajama/pile/starcoder); recoverable from "
                          "provenance codes 0-3."),
        "corpus_codes": CORPUS_CODE,
        "composition_counts": {
            "base": BASE_RETAINED,
            "reddit": PER_SOCIAL, "CA": PER_SOCIAL,
            "twitter": PER_SOCIAL, "bluesky": PER_SOCIAL,
        },
        "subsets_observed": comp,
        "finite": bool(finite),
        "matched_row_proof": {
            "baseline_dir": str(BASELINE_DIR),
            "baseline_substrate_sha256_ref":
                "372fbec511c0e9fa3b8e141529ecccaad975e469fe7b8296c019698b340b3660",
            "baseline_provenance_sha256_ref":
                "07e41fdde4e6aeefaddbc03004f52f66fd82ff3d029ec145dbe698850a1c7980",
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
    print(f"\n[minilm-bmix30-2m] built shape={a.shape} dtype={a.dtype} "
          f"finite={finite} in {(time.time()-t0)/60:.1f} min -> {sub_path}")
    print(f"[composition] {comp}")
    print(f"[matched] base provenance == baseline[retained]: {matched_ok}")
    print(f"[social>=300000] after-shuffle mins: {social_min_after}")
    del a
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
