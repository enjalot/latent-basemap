"""MONET probe reception (owner MONET, overseer 2026-09-04) — the FAIREST cross-arm metric: one IDENTICAL
held-out probe set (50K random pool rows, seeded, same for every arm) transformed through EACH arm's champion
head, scored on the SAME CLIP-kNN truth. Recall@15 = |CLIP-15NN ∩ 2D-15NN|/15 averaged. Because the exam is
identical, it isolates how well each head GENERALIZES to unseen data — unlike per-arm FFR (each arm scored on
its own truth graph, and a density-flattened draw has intrinsically sparser neighborhoods = a harder exam).
CPU transform (no GPU contention). Output: monet-probe-reception.json."""
import json, sys
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); POOL = Path("/data2/monet/pool-20m")
ARMS = ["random", "sscd", "annfaiss", "theirfaiss"]; NPROBE = 50_000; K = 15; SEED = 7


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    sys.path.insert(0, str(SB.parent / "experiments/sandbox")); sys.path.insert(0, "experiments/sandbox")
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    from scipy.spatial import cKDTree
    clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r"); N = clip.shape[0]
    probe_idx = np.sort(np.random.default_rng(SEED).choice(N, NPROBE, replace=False))
    P = _norm(np.array(clip[probe_idx], dtype=np.float32))          # identical probe set for all arms
    def _knn_cos(X, k, block=5000):          # chunked cosine (normalized -> dot); fast vs kd-tree in 512-d
        n = X.shape[0]; out = np.empty((n, k), np.int64)
        for s0 in range(0, n, block):
            sim = X[s0:s0+block] @ X.T
            sim[np.arange(sim.shape[0]), s0 + np.arange(sim.shape[0])] = -np.inf   # exclude self
            out[s0:s0+block] = np.argpartition(-sim, k, axis=1)[:, :k]
        return out
    # CLIP-space truth (identical across arms)
    tnn = _knn_cos(P, K); tset = [set(row) for row in tnn]
    out = {"schema": "monet-probe-reception-2026-09-04", "n_probe": NPROBE, "k": K, "seed": SEED,
           "metric": "recall@15 of CLIP-NN preserved in each arm-head's 2D of an IDENTICAL held-out probe set",
           "arms": {}}
    for arm in ARMS:
        mp = SB / f"monet-draw-{arm}-clip" / "champion-bs16k" / "model.pt"
        if not mp.exists():
            out["arms"][arm] = {"_status": "PENDING"}; continue
        m = ParametricUMAP.load(str(mp), device="cpu")
        xy = np.asarray(m.transform(P, batch_size=8192), dtype=np.float32)
        _, pnn = cKDTree(xy).query(xy, k=K + 1, workers=8); pnn = pnn[:, 1:]
        rec = np.mean([len(tset[i] & set(pnn[i])) / K for i in range(NPROBE)])
        out["arms"][arm] = {"probe_recall_at_15": round(float(rec), 4)}
        print(f"{arm}: probe recall@15 = {rec:.4f}", flush=True)
    (SB / "monet-probe-reception.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {SB/'monet-probe-reception.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
