"""Durable MiniLM substrate passport (P0-F).

Replaces the machine-local hand-edited `minilm_passport.json`: a deterministic
generator that records the exact sample row IDs, ordered shards, the
total-variance formula, the PCA/SVD method, the seed, and code/data hashes.

The headline number is `top2_pc_var` — the fraction of TOTAL variance (all 384
dims) explained by the top two principal components. The earlier 0.298/0.2823
figure used a top-10 singular-value denominator, which inflates the fraction;
`top2_total_variance_fraction` below uses the full-spectrum denominator and is
covered by a synthetic-covariance test that fails on the top-10-denominator bug.

Run:
  python experiments/minilm_passport.py \
     --emb /data/latent-basemap/minilm-15m/train/data-00000.npy \
     --dim 384 --n 201000 --out experiments/evidence/minilm_passport.json
"""
from __future__ import annotations
import argparse, os, sys, json, hashlib, subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.panel_v2 import load_embeddings, _ids_hash


def top2_total_variance_fraction(sample: np.ndarray):
    """Fraction of TOTAL variance (all dims) in the top-2 PCs of ``sample``.

    Centered SVD; denominator is the sum of ALL squared singular values, not the
    leading ten (the R6 bug). Returns (fraction, method_dict)."""
    Xc = sample.astype(np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)   # singular values
    var = s ** 2                                                   # ∝ variance per component
    total = float(var.sum())
    frac = float(var[:2].sum() / total) if total > 0 else float("nan")
    bug = float(var[:2].sum() / var[:10].sum()) if var[:10].sum() > 0 else float("nan")
    return frac, {"method": "centered_svd_full_spectrum",
                  "formula": "(s0^2+s1^2)/sum(all s^2)", "n_components": int(len(s)),
                  "top2_over_top10_denom_bug": round(bug, 4)}


def _sha_file(p, cap=1 << 20):
    try:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            h.update(f.read(cap))
        return h.hexdigest()[:16]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="file | dir | comma-list of ordered shards")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--n", type=int, default=201000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/evidence/minilm_passport.json")
    args = ap.parse_args()

    shards = args.emb.split(",") if "," in args.emb else args.emb
    X = load_embeddings(shards, dim=args.dim)
    N = len(X)
    rng = np.random.RandomState(args.seed)
    ids = np.sort(rng.choice(N, min(args.n, N), replace=False))
    S = np.asarray(X[ids], dtype=np.float32)                       # deterministic sample

    norms = np.linalg.norm(S, axis=1)
    # random-pair cosine baseline
    a = rng.randint(0, len(S), 20000); b = rng.randint(0, len(S), 20000)
    Sa = S[a] / (np.linalg.norm(S[a], axis=1, keepdims=True) + 1e-12)
    Sb = S[b] / (np.linalg.norm(S[b], axis=1, keepdims=True) + 1e-12)
    rand_cos = float((Sa * Sb).sum(1).mean())

    frac, method = top2_total_variance_fraction(S)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()[:12]
        dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip())
    except Exception:
        commit = dirty = None
    shard_list = shards if isinstance(shards, list) else [shards]

    out = {"substrate": "MiniLM-384-mixed", "n_total": int(N), "n_sample": int(len(ids)),
           "seed": args.seed, "dim": args.dim,
           "norm_mean": round(float(norms.mean()), 4), "norm_std": round(float(norms.std()), 4),
           "unit_norm": bool(abs(norms.mean() - 1.0) < 1e-2 and norms.std() < 1e-2),
           "mean_random_cosine": round(rand_cos, 4),
           "top2_pc_var": round(frac, 4), "pca": method,
           "sample_ids_hash": _ids_hash(ids), "sample_ids_head": ids[:8].tolist(),
           "ordered_shards": [os.path.basename(s) for s in shard_list],
           "shard_sha": {os.path.basename(s): _sha_file(s) for s in shard_list},
           "code_commit": commit, "code_dirty": dirty,
           "generated_cmd": "python " + " ".join(sys.argv)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
