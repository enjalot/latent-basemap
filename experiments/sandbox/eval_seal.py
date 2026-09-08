"""Seal the COMMON evaluation set for the efficiency-experiment pilots (owner plan
plan-global-map-efficiency-experiments.md "common decision instrument", 2026-09-08). CPU-only (faiss + gather;
GPU untouched). This is the GATE — every pilot arm scores against THIS sealed set so compression/loss/sampler
changes are judged on identical neighborhood targets, not each arm's own truth.

Contract (from the plan):
  * one sealed TRAINING subset (500K), a shared REFERENCE subset (250K), disjoint VALIDATION + TEST queries (5K
    each) — all DISJOINT, drawn from the pool-20m DINO-1536 column, stratified BY PROVENANCE (source.npy, 9 sources).
  * ORIGINAL full-D (1536) encoder truth for every compression arm: each query's exact k=15 nearest reference
    rows in 1536-D (faiss IP). Query self-matches excluded (queries are disjoint from the reference by construction).
  * a fixed DIAGNOSTIC sample (5K reference rows) with full-D k=50 NN among the reference, for
    trustworthiness/continuity in the scorer.
  * preprocessing (PCA etc.) is fit by each ARM on train rows only — NOT here; the sealed truth is original-D.

Output /data2/monet/eval-common/: {train_idx,ref_idx,val_idx,test_idx}.npy (pool positions), {val,test}_source.npy,
{train,ref,val,test}_hd.f16.npy (the 1536-D vectors, so arms/scorer never re-gather from 19.3M), truth_val.npy +
truth_test.npy (Q×15 REFERENCE-LOCAL indices), diag_idx.npy (ref-local) + diag_knn_hd.npy (5K×50 ref-local),
manifest.json. Usage: eval_seal.py [SEED=42].
"""
import json, sys, time
from pathlib import Path
import numpy as np

POOL = Path("/data2/monet/pool-20m"); OUT = Path("/data2/monet/eval-common")
DIM = 1536; N_TRAIN, N_REF, N_VAL, N_TEST, N_DIAG = 500_000, 250_000, 5_000, 5_000, 5_000
K_TRUTH, K_DIAG = 15, 50; SRC_FLOOR = 100        # per-source floor for val/test so every cohort is represented


def _gather(dino, positions):
    """Gather DINO f16 at SORTED pool positions -> f32 (positions are small sets here)."""
    return np.asarray(dino[np.sort(positions)], np.float32)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    OUT.mkdir(parents=True, exist_ok=True)
    import faiss
    dino = np.load(POOL / "dino1536.f16.npy", mmap_mode="r"); N = dino.shape[0]
    source = np.load(POOL / "source.npy", allow_pickle=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    # disjoint partition off the shuffled pool
    train_idx = np.sort(perm[:N_TRAIN])
    ref_idx = np.sort(perm[N_TRAIN:N_TRAIN + N_REF])
    pool_rest = perm[N_TRAIN + N_REF:]
    # provenance-stratified val + test off the remainder: proportional to source frequency, floor SRC_FLOOR/source
    rest_src = source[pool_rest]; sources = np.unique(source)

    def strat(n_target, exclude):
        picks = []
        avail = pool_rest[~np.isin(pool_rest, exclude)] if exclude.size else pool_rest
        avail_src = source[avail]
        # proportional allocation with a floor
        counts = {s: int((avail_src == s).sum()) for s in sources}
        total = sum(counts.values())
        alloc = {s: max(SRC_FLOOR, int(round(n_target * counts[s] / total))) if counts[s] else 0 for s in sources}
        for s in sources:
            cand = avail[avail_src == s]
            take = min(alloc[s], cand.size)
            if take: picks.append(rng.choice(cand, take, replace=False))
        out = np.concatenate(picks); rng.shuffle(out)
        return np.sort(out[:n_target]) if out.size > n_target else np.sort(out)

    val_idx = strat(N_VAL, np.array([], int))
    test_idx = strat(N_TEST, val_idx)
    assert len(set(train_idx) & set(ref_idx)) == 0 and len(set(val_idx) & set(ref_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0 and len(set(train_idx) & set(val_idx)) == 0

    t0 = time.time()
    ref_hd = _gather(dino, ref_idx); val_hd = _gather(dino, val_idx); test_hd = _gather(dino, test_idx)
    train_hd = _gather(dino, train_idx)
    print(f"[seal] gathered train {train_hd.shape} ref {ref_hd.shape} val {val_hd.shape} test {test_hd.shape} in {time.time()-t0:.0f}s", flush=True)

    # ORIGINAL full-D truth: each query's exact k=15 NN among the 250K reference (faiss IP; queries disjoint -> no self)
    t0 = time.time(); hdx = faiss.IndexFlatIP(DIM); hdx.add(np.ascontiguousarray(ref_hd))
    _, truth_val = hdx.search(np.ascontiguousarray(val_hd), K_TRUTH)
    _, truth_test = hdx.search(np.ascontiguousarray(test_hd), K_TRUTH)
    # diagnostic sample: 5K reference rows + their full-D k=50 NN AMONG the reference (exclude self at col0)
    diag_idx = np.sort(rng.choice(N_REF, N_DIAG, replace=False))
    _, diag_knn = hdx.search(np.ascontiguousarray(ref_hd[diag_idx]), K_DIAG + 1)
    diag_knn = diag_knn[:, 1:]                    # drop self (diag rows ARE in the reference)
    print(f"[seal] truth (val/test k{K_TRUTH}) + diag k{K_DIAG} in {time.time()-t0:.0f}s", flush=True)

    np.save(OUT / "train_idx.npy", train_idx); np.save(OUT / "ref_idx.npy", ref_idx)
    np.save(OUT / "val_idx.npy", val_idx); np.save(OUT / "test_idx.npy", test_idx)
    np.save(OUT / "val_source.npy", source[val_idx]); np.save(OUT / "test_source.npy", source[test_idx])
    for nm, arr in [("train_hd", train_hd), ("ref_hd", ref_hd), ("val_hd", val_hd), ("test_hd", test_hd)]:
        np.save(OUT / f"{nm}.f16.npy", arr.astype(np.float16))
    np.save(OUT / "truth_val.npy", truth_val.astype(np.int32)); np.save(OUT / "truth_test.npy", truth_test.astype(np.int32))
    np.save(OUT / "diag_idx.npy", diag_idx.astype(np.int32)); np.save(OUT / "diag_knn_hd.npy", diag_knn.astype(np.int32))
    vs, vc = np.unique(source[val_idx], return_counts=True)
    (OUT / "manifest.json").write_text(json.dumps({
        "schema": "eval-common-sealed-2026-09-08", "seed": seed, "substrate": "pool-20m/dino1536 (1536-d, L2=1.0)",
        "n_train": int(train_idx.size), "n_ref": int(ref_idx.size), "n_val": int(val_idx.size),
        "n_test": int(test_idx.size), "n_diag": int(diag_idx.size), "k_truth": K_TRUTH, "k_diag": K_DIAG,
        "truth": "ORIGINAL full-D (1536) exact k15 NN among the 250K reference; self excluded (queries disjoint from ref)",
        "val_by_source": {str(s): int(c) for s, c in zip(vs, vc)},
        "contract": "recall(k=15,B) = |H_15(q) ∩ M_B(q)|/15 with M_B = B nearest REFERENCE map points to the query's "
                    "projected coord; B fixed at 250 and 2000 (0.1% kept secondary). Arms fit preprocessing on train "
                    "rows only. Score+deploy f_theta(x), never Z.",
        "note": "GATE for every efficiency pilot. Original-D truth is arm-independent so compression is judged fairly."}, indent=1))
    print(f"[seal] DONE -> {OUT} | val by source: {dict(zip(vs.tolist(), vc.tolist()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
