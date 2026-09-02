"""Build an OOD-battery substrate (owner item 4, overseer 2026-09-02). Fixed-w=0.02 OOD test: same T0+T1+T2
as the evolbench draw1 timeline, but swap the T3 injection corpus. Materializes substrates/evolbench-ood-
<name>/ with T0/T1/T2 SYMLINKED to evolbench (bit-identical base) and T3 = 800k rows from the swap corpus
(cast to f32, disjoint front slice). Writes provenance (corpus code, shard, local row) so the blob forensics
can resolve text, and a proofs.json (T3 content-disjoint from the base union, sampled).
Usage: build_ood_tranche.py <name> <emb_glob> <corpus_code> [n=800000]."""
import sys, glob, json, time
from pathlib import Path
import numpy as np

import os
# base dir env-parameterized: MiniLM OOD uses evolbench (N_BASE 5.6M); jina OOD uses evolbench-d768 (2.8M).
EVOL = Path(os.environ.get("EVOLBENCH_BASE_DIR", "/data/latent-basemap/substrates/evolbench"))
N_BASE = int(os.environ.get("EVOLBENCH_N_BASE", "5600000")); SHARD = 500_000


def _void(a):
    return np.ascontiguousarray(a).view([('', a.dtype)] * a.shape[1]).ravel()


def main():
    name, emb_glob, code = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 800_000
    out = Path(f"/data/latent-basemap/substrates/evolbench-ood-{name}")
    t0 = time.time()
    # T0/T1/T2 symlinks (bit-identical base)
    for t in ("T0", "T1", "T2"):
        d = out / t; d.mkdir(parents=True, exist_ok=True)
        for f in ("substrate.f32.npy", "provenance.npy"):
            lk = d / f
            if not lk.exists():
                lk.symlink_to(EVOL / t / f)
    # T3' = first n rows of the swap corpus (fp16/f32 -> f32), disjoint front slice
    files = sorted(glob.glob(emb_glob))
    if not files:
        raise SystemExit(f"no embeddings at {emb_glob}")
    # POOL_OFFSET: when the embeddings are a pre-sliced social POOL (e.g. ca-jina-pool = communityarchive-
    # tweets[offset:]), map each pool row back to its GLOBAL text-corpus row so provenance -> chunk_text is
    # correct: global = POOL_OFFSET + running_index -> shard = global//SHARD, row = global%SHARD.
    pool_offset = int(os.environ.get("EVOLBENCH_POOL_OFFSET", "0"))
    rows = []; prov = []; got = 0
    for si, f in enumerate(files):
        a = np.asarray(np.load(f, mmap_mode="r"), dtype=np.float32)
        take = min(len(a), n - got)
        rows.append(a[:take]);
        pr = np.empty(take, dtype=[("corpus", "u1"), ("shard", "<u2"), ("row", "<i8")])
        pr["corpus"] = code
        if pool_offset:
            g = pool_offset + got + np.arange(take, dtype=np.int64)   # global text-corpus row
            pr["shard"] = (g // SHARD).astype("<u2"); pr["row"] = (g % SHARD)
        else:
            pr["shard"] = si; pr["row"] = np.arange(take, dtype=np.int64)
        prov.append(pr); got += take
        if got >= n:
            break
    if got < n:
        raise SystemExit(f"corpus {name} only {got:,} rows < {n:,} requested")
    sub = np.concatenate(rows).astype(np.float32); prov = np.concatenate(prov)
    d3 = out / "T3"; d3.mkdir(parents=True, exist_ok=True)
    np.save(d3 / "substrate.f32.npy", sub); np.save(d3 / "provenance.npy", prov)
    # proofs: T3' content-disjoint from base union (sampled void-view membership)
    base_sample = np.asarray(np.load(EVOL / "T0" / "substrate.f32.npy", mmap_mode="r")[:500_000], np.float32)
    overlap = int(np.isin(_void(sub[:50_000]), _void(base_sample)).sum())
    proofs = {"schema": "evolbench-ood-tranche-2026-09-02", "name": name, "corpus_code": code,
              "T3_rows": int(len(sub)), "base_rows": N_BASE, "t3_base_content_overlap_sampled": overlap,
              "disjoint_ok": bool(overlap == 0), "emb_glob": emb_glob, "wall_s": round(time.time() - t0, 1),
              "note": "T0/T1/T2 symlinked to evolbench (bit-identical); T3 swapped for the OOD corpus."}
    (out / "evolbench-ood-proofs.json").write_text(json.dumps(proofs, indent=1))
    print(f"OOD tranche {name}: T3 {len(sub):,}x{sub.shape[1]} overlap={overlap} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
