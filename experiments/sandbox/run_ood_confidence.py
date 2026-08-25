#!/usr/bin/env python3
"""Driver: does a cheap per-point novelty score predict per-point placement
error for parametric-UMAP maps? CPU-ONLY.

Wires the on-disk inputs -> experiments/metrics/ood_confidence.py -> JSON + PNGs.

For each test set (CA worst-OOD, reddit mid-OOD, in-corpus holdout), the 50k
MiniLM queries are projected through the champion map on CPU, then scored with:
  Signal A  novelty_score           (cheap)
  truth     pointwise_placement_error
  Signal B  seed_ensemble_spread    (3-seed family, one procrustes alignment)

Reports Spearman + Pearson + AUC (top-quartile error = positive class) of
score-vs-error, plus a calibration table (median novelty / median error). Writes
results.json and scatter/overlay PNGs to /data/latent-basemap/sandbox/ood-confidence/.

Usage:  CUDA_VISIBLE_DEVICES="" python run_ood_confidence.py
"""
from __future__ import annotations

# ── HARD CPU LOCK (must precede torch import) ────────────────────────────────
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "32")

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))            # experiments/
sys.path.insert(0, str(HERE.parents[2]))            # repo root (basemap package)

from metrics import ood_confidence as ooc  # noqa: E402

import torch  # noqa: E402
torch.set_num_threads(32)

from basemap.pumap.parametric_umap.core import ParametricUMAP  # noqa: E402

# ── paths ────────────────────────────────────────────────────────────────────
SUBSTRATE = ("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
             "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy")
RADII = "/data/latent-basemap/sandbox/density-radii-2m.npy"
CHAMP_DIR = "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x4bs16k-winner"
SEED_DIRS = [
    "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x2-fneg10-tanh4",
    "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x2-fneg10-tanh4-seed43",
    "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x2-fneg10-tanh4-seed44",
]
CA_GLOB = "/data/embeddings/communityarchive-tweets-all-MiniLM-L6-v2/train/*.npy"
REDDIT_GLOB = "/data/embeddings/reddit-tldr17-chunked-120-all-MiniLM-L6-v2/train/*.npy"
FINEWEB_GLOB = "/data/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train/*.npy"

OUT = Path("/data/latent-basemap/sandbox/ood-confidence")
OUT.mkdir(parents=True, exist_ok=True)

N_QUERIES = 50_000
SEED = 42
K = 15


def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# ── test-set samplers ─────────────────────────────────────────────────────────

def _sample_npy_shards(pattern: str, n: int, seed: int) -> np.ndarray:
    """Sample n rows (seeded) from a set of proper .npy shards, concatenated."""
    files = sorted(glob.glob(pattern))
    mms = [np.load(f, mmap_mode="r") for f in files]
    counts = np.array([m.shape[0] for m in mms])
    total = int(counts.sum())
    offs = np.concatenate([[0], np.cumsum(counts)])
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(total, size=min(n, total), replace=False))
    out = np.empty((len(pick), mms[0].shape[1]), dtype=np.float32)
    for i, g in enumerate(pick):
        s = int(np.searchsorted(offs, g, side="right") - 1)
        out[i] = np.asarray(mms[s][g - offs[s]], dtype=np.float32)
    return out


def _sample_fineweb_holdout(pattern: str, n: int, seed: int,
                            skip_rows: int = 3_000_000) -> np.ndarray:
    """In-corpus holdout: headerless raw f32 dim-384 fineweb shards, sampled from
    an offset well beyond the 2M used in the training substrate."""
    files = sorted(glob.glob(pattern))
    dim = 384
    mms = []
    for f in files:
        with open(f, "rb") as fh:
            magic = fh.read(6)
        if magic == b"\x93NUMPY":
            mms.append(np.load(f, mmap_mode="r"))
        else:
            sz = os.path.getsize(f)
            rows = sz // (dim * 4)
            mms.append(np.memmap(f, dtype=np.float32, mode="r", shape=(rows, dim)))
    counts = np.array([m.shape[0] for m in mms])
    offs = np.concatenate([[0], np.cumsum(counts)])
    total = int(counts.sum())
    lo = min(skip_rows, max(total - n, 0))
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(np.arange(lo, total), size=min(n, total - lo), replace=False))
    out = np.empty((len(pick), dim), dtype=np.float32)
    for i, g in enumerate(pick):
        s = int(np.searchsorted(offs, g, side="right") - 1)
        out[i] = np.asarray(mms[s][g - offs[s]], dtype=np.float32)
    return out


# ── stats ──────────────────────────────────────────────────────────────────

def _spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def _pearson(a, b):
    from scipy.stats import pearsonr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(pearsonr(a[m], b[m])[0])


def _auc_top_quartile(score, error):
    """AUC with positive class = top-quartile error; ranking var = score."""
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(score) & np.isfinite(error)
    score, error = score[m], error[m]
    thr = np.quantile(error, 0.75)
    y = (error >= thr).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, score))


def main() -> int:
    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "", \
        "CUDA must be disabled"
    t_all = time.time()
    print(f"[cpu] faiss={ooc._HAVE_FAISS} threads=32", flush=True)

    print("[load] substrate + radii + champion coords ...", flush=True)
    substrate = np.load(SUBSTRATE, mmap_mode="r")
    radii = np.load(RADII)
    champ_xy = np.load(f"{CHAMP_DIR}/coordinates.npy")           # (2M, 2) training positions

    print("[index] building CPU kNN index over 2M substrate ...", flush=True)
    t0 = time.time()
    index = ooc.build_index(substrate, threads=32)
    print(f"[index] built in {time.time()-t0:.1f}s (ntotal={index.ntotal})", flush=True)

    print("[kdtree] building cKDTree over champion 2D coords ...", flush=True)
    from scipy.spatial import cKDTree
    t0 = time.time()
    champ_tree = cKDTree(champ_xy)
    print(f"[kdtree] built in {time.time()-t0:.1f}s", flush=True)

    print("[model] loading champion (CPU) ...", flush=True)
    champ = ParametricUMAP.load(f"{CHAMP_DIR}/model.pt", device="cpu")

    print("[model] loading seed family (CPU) ...", flush=True)
    seed_models = [ParametricUMAP.load(f"{d}/model.pt", device="cpu") for d in SEED_DIRS]
    seed_train_xy = [np.load(f"{d}/coordinates.npy") for d in SEED_DIRS]

    # test sets
    test_sets = {}
    print("[sample] CA (worst OOD) ...", flush=True)
    test_sets["CA"] = _norm(_sample_npy_shards(CA_GLOB, N_QUERIES, SEED))
    print("[sample] reddit (mid OOD) ...", flush=True)
    test_sets["reddit"] = _norm(_sample_npy_shards(REDDIT_GLOB, N_QUERIES, SEED))
    print("[sample] in-corpus fineweb holdout ...", flush=True)
    test_sets["in_corpus"] = _norm(_sample_fineweb_holdout(FINEWEB_GLOB, N_QUERIES, SEED))

    # in-corpus novelty ratios first -> reference distribution for percentiles
    print("[A] in_corpus novelty (reference distribution) ...", flush=True)
    ref = ooc.novelty_score(test_sets["in_corpus"], index, radii, k=K)
    ref_ratios = ref.ratio

    results = {}
    per_point = {}
    for name, X in test_sets.items():
        print(f"\n==== {name} (n={len(X)}) ====", flush=True)
        # Signal A
        t0 = time.time()
        nov = ooc.novelty_score(X, index, radii, k=K, ref_ratios=ref_ratios)
        print(f"[A] novelty {time.time()-t0:.1f}s", flush=True)

        # champion projection (CPU)
        t0 = time.time()
        xy = np.asarray(champ.transform(X, batch_size=8192), dtype=np.float32)
        print(f"[proj] champion transform {time.time()-t0:.1f}s", flush=True)

        # ground-truth pointwise error
        t0 = time.time()
        err = ooc.pointwise_placement_error(
            X, xy, substrate, champ_xy, index, k=K, kdtree=champ_tree)
        print(f"[truth] pointwise error {time.time()-t0:.1f}s", flush=True)

        # Signal B: project X through each seed, align, spread
        t0 = time.time()
        seed_proj = [np.asarray(m.transform(X, batch_size=8192), dtype=np.float32)
                     for m in seed_models]
        spread = ooc.seed_ensemble_spread(seed_proj, seed_train_xy)
        print(f"[B] ensemble spread {time.time()-t0:.1f}s", flush=True)

        results[name] = {
            "n": int(len(X)),
            "spearman_A": _spearman(nov.ratio, err),
            "pearson_A": _pearson(nov.ratio, err),
            "auc_A": _auc_top_quartile(nov.ratio, err),
            "spearman_B": _spearman(spread, err),
            "pearson_B": _pearson(spread, err),
            "auc_B": _auc_top_quartile(spread, err),
            "median_novelty": float(np.nanmedian(nov.ratio)),
            "median_error": float(np.nanmedian(err)),
            "median_novelty_percentile": float(np.nanmedian(nov.percentile)),
            "median_spread": float(np.nanmedian(spread)),
        }
        per_point[name] = {"novelty": nov.ratio, "error": err, "spread": spread}
        print(json.dumps(results[name], indent=2), flush=True)

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[write] {OUT/'results.json'}", flush=True)

    _plots(per_point)
    print(f"\n[done] total {time.time()-t_all:.1f}s", flush=True)
    return 0


def _plots(per_point: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"CA": "#d1495b", "reddit": "#edae49", "in_corpus": "#2e86ab"}

    # per-set scatter: novelty vs error
    for name, d in per_point.items():
        nov, err = d["novelty"], d["error"]
        m = np.isfinite(nov) & np.isfinite(err)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(nov[m], err[m], s=3, alpha=0.15, color=colors.get(name, "#444"))
        ax.set_xlabel("novelty ratio (Signal A)")
        ax.set_ylabel("pointwise placement error (1 - recall)")
        ax.set_title(f"{name}: novelty vs error")
        fig.tight_layout()
        fig.savefig(OUT / f"scatter_{name}.png", dpi=110)
        plt.close(fig)

    # novelty distribution overlay
    fig, ax = plt.subplots(figsize=(7, 5))
    lo = min(np.nanpercentile(d["novelty"], 1) for d in per_point.values())
    hi = max(np.nanpercentile(d["novelty"], 99) for d in per_point.values())
    bins = np.linspace(lo, hi, 80)
    for name, d in per_point.items():
        ax.hist(d["novelty"][np.isfinite(d["novelty"])], bins=bins, density=True,
                histtype="step", lw=2, color=colors.get(name, "#444"), label=name)
    ax.axvline(1.0, color="#888", ls="--", lw=1, label="ratio=1 (local density match)")
    ax.set_xlabel("novelty ratio (Signal A)")
    ax.set_ylabel("density")
    ax.set_title("novelty distribution: CA vs reddit vs in-corpus")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "novelty_overlay.png", dpi=110)
    plt.close(fig)

    # error distribution overlay (bonus calibration read)
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 60)
    for name, d in per_point.items():
        ax.hist(d["error"][np.isfinite(d["error"])], bins=bins, density=True,
                histtype="step", lw=2, color=colors.get(name, "#444"), label=name)
    ax.set_xlabel("pointwise placement error")
    ax.set_ylabel("density")
    ax.set_title("error distribution: CA vs reddit vs in-corpus")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "error_overlay.png", dpi=110)
    plt.close(fig)
    print(f"[write] PNGs -> {OUT}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
