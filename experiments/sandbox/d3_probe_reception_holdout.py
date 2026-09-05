"""D3: probe-reception redo with a TRUE holdout (owner D-block, 2026-09-05). CPU (device=cpu transforms — no GPU
contention). Fixes the leakage in p_monet_probe_reception.py: its 50K probe was drawn from the whole pool, so
some probe rows sat INSIDE an arm's training draw (an arm being scored partly on rows it trained on). D3 draws
the probe from the TRUE holdout — pool rows in NO arm's training set (pool minus the union of random/sscd/
annfaiss/theirfaiss draw idx) — so every arm is scored purely on unseen data. Identical probe + identical
CLIP-kNN truth across arms (the fair cross-arm generalization exam). Recall@15 = |CLIP-15NN ∩ 2D-15NN|/15.

Output: monet-probe-reception-holdout-20260905.json (+ old-vs-holdout delta if the leaky run's json is present).
"""
import json, sys
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); POOL = Path("/data2/monet/pool-20m"); DRAWS = Path("/data2/monet/draws")
ARMS = ["random", "sscd", "annfaiss", "theirfaiss"]; NPROBE = 50_000; K = 15; SEED = 7


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    sys.path.insert(0, "experiments/sandbox")
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    from scipy.spatial import cKDTree
    clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r"); N = clip.shape[0]

    # TRUE holdout: pool positions in NO arm's training draw
    trained = np.zeros(N, bool); arm_sizes = {}
    for arm in ARMS:
        idx = np.load(DRAWS / f"{arm}.idx.npy"); arm_sizes[arm] = int(idx.shape[0])
        trained[idx] = True
    holdout = np.where(~trained)[0]
    print(f"[d3] pool {N:,} | trained-union {int(trained.sum()):,} | true holdout {holdout.size:,} "
          f"({holdout.size/N*100:.1f}%)", flush=True)
    probe_idx = np.sort(np.random.default_rng(SEED).choice(holdout, NPROBE, replace=False))
    P = _norm(np.array(clip[probe_idx], dtype=np.float32))

    def _knn_cos(X, k, block=5000):
        n = X.shape[0]; out = np.empty((n, k), np.int64)
        for s0 in range(0, n, block):
            sim = X[s0:s0+block] @ X.T
            sim[np.arange(sim.shape[0]), s0 + np.arange(sim.shape[0])] = -np.inf
            out[s0:s0+block] = np.argpartition(-sim, k, axis=1)[:, :k]
        return out
    tset = [set(row) for row in _knn_cos(P, K)]

    old = {}
    op = SB / "monet-probe-reception.json"
    if op.exists():
        old = {a: v.get("probe_recall_at_15") for a, v in json.loads(op.read_text()).get("arms", {}).items()}

    out = {"schema": "monet-probe-reception-holdout-2026-09-05", "n_probe": NPROBE, "k": K, "seed": SEED,
           "true_holdout": "pool minus union of all arm training idx (no leakage)",
           "arm_train_sizes": arm_sizes, "n_true_holdout": int(holdout.size), "arms": {}}
    for arm in ARMS:
        mp = SB / f"monet-draw-{arm}-clip" / "champion-bs16k" / "model.pt"
        if not mp.exists():
            out["arms"][arm] = {"_status": "PENDING"}; continue
        m = ParametricUMAP.load(str(mp), device="cpu")
        xy = np.asarray(m.transform(P, batch_size=8192), dtype=np.float32)
        _, pnn = cKDTree(xy).query(xy, k=K + 1, workers=8); pnn = pnn[:, 1:]
        rec = float(np.mean([len(tset[i] & set(pnn[i])) / K for i in range(NPROBE)]))
        row = {"probe_recall_at_15_holdout": round(rec, 4)}
        if arm in old and old[arm] is not None:
            row["leaky_recall_at_15"] = old[arm]; row["holdout_minus_leaky"] = round(rec - old[arm], 4)
        out["arms"][arm] = row
        print(f"[d3] {arm}: holdout recall@15 = {rec:.4f}" +
              (f" (leaky was {old[arm]}, delta {rec-old[arm]:+.4f})" if arm in old and old[arm] is not None else ""), flush=True)
    (SB / "monet-probe-reception-holdout-20260905.json").write_text(json.dumps(out, indent=1))
    print(f"[d3] DONE -> {SB/'monet-probe-reception-holdout-20260905.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
