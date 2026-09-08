"""Seal v2 — INSTRUMENT CORRECTION of the common eval set (owner via overseer 2026-09-08). NOT a gate change: the
worst-cohort gate (no cohort loses >0.01) stands. The v1 seal's per-source floor (min 100) left diffusion-aesthetic-4k
at 99 val queries, whose bootstrap SD (~0.036) cannot resolve a 0.01 gate at any confidence — the instrument was
mis-sized for the preregistered gate. v2 keeps v1's TRAIN + 250K REFERENCE + diagnostic UNCHANGED (so the truth and
every arm's training are identical and comparable), and ONLY re-draws the VALIDATION queries to ~1200/source (laion
capped) so every cohort clears ≥1000 and can resolve the gate. v1 is preserved on disk. Re-scoring stored heads
against v2 applies the SAME unchanged gates.

Output /data2/monet/eval-common-v2/: train_idx/ref_idx/ref_hd/diag_idx/diag_knn_hd/train_hd COPIED from v1;
NEW val_idx/val_source/val_hd + truth_val (exact k15 among the SAME v1 250K reference). Usage: eval_seal_v2.py
"""
import json, shutil, time
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m"); V1 = Path("/data2/monet/eval-common"); OUT = Path("/data2/monet/eval-common-v2")
DIM = 1536; K_TRUTH = 15; PER_SOURCE = 1200; SEED = 4242


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import faiss
    dino = np.load(POOL / "dino1536.f16.npy", mmap_mode="r"); source = np.load(POOL / "source.npy", allow_pickle=True)
    # carry v1's train/ref/diagnostic verbatim (truth + training unchanged -> comparable)
    for f in ("train_idx.npy", "ref_idx.npy", "ref_hd.f16.npy", "train_hd.f16.npy", "diag_idx.npy", "diag_knn_hd.npy"):
        shutil.copy(V1 / f, OUT / f)
    ref_hd = np.asarray(np.load(OUT / "ref_hd.f16.npy"), np.float32)
    used = np.concatenate([np.load(V1 / f) for f in ("train_idx.npy", "ref_idx.npy", "val_idx.npy", "test_idx.npy")])
    used_set = np.zeros(dino.shape[0], bool); used_set[used] = True         # exclude v1 train/ref/val/test
    rng = np.random.default_rng(SEED)

    picks, srcs = [], np.unique(source)
    for s in srcs:
        cand = np.where((source == s) & (~used_set))[0]
        take = min(PER_SOURCE, cand.size)
        picks.append(rng.choice(cand, take, replace=False))
        print(f"[seal-v2] {s}: {take} val (of {cand.size:,} available)", flush=True)
    val_idx = np.sort(np.concatenate(picks))
    val_hd = np.asarray(dino[val_idx], np.float32)
    assert len(set(val_idx.tolist()) & set(np.load(OUT / "ref_idx.npy").tolist())) == 0

    t0 = time.time(); hdx = faiss.IndexFlatIP(DIM); hdx.add(np.ascontiguousarray(ref_hd))
    _, truth_val = hdx.search(np.ascontiguousarray(val_hd), K_TRUTH)         # exact k15 among the SAME 250K reference
    np.save(OUT / "val_idx.npy", val_idx); np.save(OUT / "val_source.npy", source[val_idx])
    np.save(OUT / "val_hd.f16.npy", val_hd.astype(np.float16)); np.save(OUT / "truth_val.npy", truth_val.astype(np.int32))
    vs, vc = np.unique(source[val_idx], return_counts=True)
    (OUT / "manifest.json").write_text(json.dumps({
        "schema": "eval-common-sealed-v2-2026-09-08", "kind": "INSTRUMENT CORRECTION of v1 (gate unchanged)",
        "unchanged_from_v1": "train (500K), reference (250K), diagnostic, original-D 1536 truth definition",
        "changed": "validation queries re-drawn to ~%d/source (min ≥1000, laion no longer dominant) so every cohort "
                   "can resolve the preregistered 0.01 worst-cohort gate; v1's diffusion-aesthetic-4k had 99 (SD ~0.036)" % PER_SOURCE,
        "n_val": int(val_idx.size), "val_by_source": {str(s): int(c) for s, c in zip(vs, vc)}, "seed": SEED,
        "why": "the 99-query cohort tripped C's worst-cohort gate at ~1 bootstrap SD (noise). Re-scoring stored heads "
               "against v2 applies the SAME gates; v1 preserved at /data2/monet/eval-common."}, indent=1))
    print(f"[seal-v2] DONE {val_idx.size} val ({time.time()-t0:.0f}s) -> {OUT} | by source: {dict(zip(vs.tolist(), vc.tolist()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
