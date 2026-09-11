"""Card010 edge-set builder (CPU, off-flock) — EXPLORATORY continuation per OC/card010-floor-ruling.md.
Consumes ONLY the corrected estimator's persisted arrays (kgraph.npy from card010_kstar.py, self-
inclusive knn_idx.npy col0=self). Builds the three matched directed binary-edge graphs:
  fixed15  : k=15
  adaptive : k_i = kgraph[i]  (ABIDE k* clamped [5,60] — the intended clamped intervention)
  fixed_mean: k=12 (round(kgraph_mean); the "more edges" degree control)
Each node i emits its first k_i REAL neighbors (knn_idx[i, 1:k_i+1]; self excluded). Asserts edge
counts + n_nodes match the frozen card010-kstar.json arm_diagnostics (identity guard) and fails closed
on any mismatch. Does NOT recompute k* or touch the immutable failed prereg. Usage: build_card010_edges.py
"""
import os, sys, json, hashlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np

OUT = Path("/data/latent-basemap/substrates/card010-adaptive")
OC = Path("/data/latent-basemap/sandbox/overseer-codex")
FIXED_MEAN_K = 12  # from corrected kgraph_mean 11.6225 -> round = 12 (== card010-kstar.json fixed_mean_k)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(name, kfunc, idx, n, maxk):
    src = np.empty(0, np.int32); dst_parts = []; src_parts = []
    ks = kfunc if isinstance(kfunc, np.ndarray) else np.full(n, int(kfunc), np.int32)
    ks = ks.astype(np.int32)
    assert ks.min() >= 1 and ks.max() <= maxk - 1, f"{name}: k out of [1,{maxk-1}]"
    for i in range(n):
        k = int(ks[i])
        src_parts.append(np.full(k, i, np.int32))
        dst_parts.append(idx[i, 1:k + 1].astype(np.int32))  # col0 is self -> skip
    s = np.concatenate(src_parts); d = np.concatenate(dst_parts)
    # no self-loops (self was column 0, excluded) and targets in range
    assert (d != s).all(), f"{name}: self-loop leaked"
    assert d.min() >= 0 and d.max() < n, f"{name}: target index out of range"
    return s, d, ks


def main():
    idx = np.load(OUT / "knn_idx.npy"); kgraph = np.load(OUT / "kgraph.npy")
    kraw = np.load(OUT / "kstar_raw.npy"); draw = np.load(OUT / "draw_ids.npy")
    n, maxk = idx.shape
    assert kgraph.shape[0] == n == draw.shape[0], "row-count mismatch across persisted arrays"
    assert (idx[:, 0] == np.arange(n)).all(), "knn_idx col0 must be self (self-inclusive convention)"
    assert int(round(float(kraw.mean()))) in (FIXED_MEAN_K, FIXED_MEAN_K - 1) or \
        int(round(float(kgraph.mean()))) == FIXED_MEAN_K, "fixed_mean_k does not match persisted means"

    ref = json.load(open(OC / "card010-kstar.json"))["arm_diagnostics"]
    specs = [("fixed15", np.int32(15)), ("adaptive", kgraph), ("fixed_mean", np.int32(FIXED_MEAN_K))]
    ident = {}
    for name, kf in specs:
        s, d, ks = build(name, kf, idx, n, maxk)
        exp = int(ref[name]["edges"])
        assert s.shape[0] == exp, f"{name}: edge count {s.shape[0]} != frozen diagnostic {exp}"
        p = OUT / f"edges-{name}.npz"
        np.savez(p, sources=s, targets=d, weights=np.ones(s.shape[0], np.float32), n_nodes=np.int64(n))
        ident[name] = {"edges": int(s.shape[0]), "mean_k": round(float(ks.mean()), 4),
                       "sha256": _sha(p), "matches_diagnostic": True}
        print(f"edges-{name}: {s.shape[0]} directed edges (mean k {ks.mean():.3f})  sha {ident[name]['sha256'][:16]}", flush=True)

    manifest = {"schema": "card010-edges-2026-09-11", "n_nodes": int(n), "maxk_incl_self": int(maxk),
                "fixed_mean_k": FIXED_MEAN_K, "source": "corrected card010_kstar.py kgraph (clamped [5,60])",
                "consumes": {"knn_idx.sha256": _sha(OUT / "knn_idx.npy")[:16],
                             "kgraph.sha256": _sha(OUT / "kgraph.npy")[:16],
                             "kstar_raw.sha256": _sha(OUT / "kstar_raw.npy")[:16],
                             "draw_ids.sha256": _sha(OUT / "draw_ids.npy")[:16]},
                "edges": ident,
                "note": "EXPLORATORY per card010-floor-ruling.md; original prereg viability stays FAIL. Binary weights."}
    (OC / "card010-edges-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
