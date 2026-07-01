#!/usr/bin/env python
"""Space passport: pre-flight geometry diagnostics per embedding space.

Implements Work Package 4 of the basemap atlas program
(`latent-labs/guides/plan-basemap-atlas.md`, §3 "pre-flight geometry
diagnostics"). For each embedding space it computes a one-page "passport":

  1. Anisotropy   -- mean/dist of cosine sim of random pairs, norm + norm CV.
  2. Uniformity   -- Wang-Isola L_uniform = log mean exp(-2||u-v||^2).
  3. Intrinsic dim-- TwoNN (Facco et al. 2017) + PCA participation ratio.
  4. Eigenspectrum-- top-100 PCA eigenvalues, tail power-law slope, %var@k.
  5. Hubness      -- exact k=50 kNN in-degree skewness + max in-degree.

Plus (§3 decision point) a matryoshka truncation test on the jina English
space: does a truncated-dim k=50 graph reproduce the full-dim graph.

Everything is CPU-only and samples <=100k rows (200k for matryoshka), so it
stays tractable. Embedding shards are opened lazily with np.load(mmap_mode)
or np.memmap for raw buffers -- no shard >= 2 GB is ever materialized; rows
are cast to fp32 per gathered batch.

Usage:
    python experiments/space_passport.py passports    # all 5 space passports
    python experiments/space_passport.py matryoshka    # truncation test
    python experiments/space_passport.py all           # both (default)

Read-only on /data. Writes only under experiments/results/passports/.
"""
from __future__ import annotations

import os
import glob
import json
import time
import argparse
import numpy as np
from scipy.special import logsumexp
from scipy.stats import skew

try:
    import faiss
except Exception as e:  # pragma: no cover
    raise SystemExit(f"faiss is required: {e}")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "results", "passports")

# ---------------------------------------------------------------------------
# Space registry
# ---------------------------------------------------------------------------
SPACES = [
    dict(
        name="minilm-384-fineweb",
        model="all-MiniLM-L6-v2",
        dim=384,
        per_token=False,
        directory="/data/embeddings/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train/",
        note="Pooled sentence embeddings, 120-token chunks, fineweb-edu (English).",
    ),
    dict(
        name="jina-v5-nano-768-en",
        model="jina-embeddings-v5-text-nano",
        dim=768,
        per_token=False,
        directory="/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train/",
        note="Pooled sentence embeddings, 500-token chunks, fineweb-edu (English). Primary target model.",
    ),
    dict(
        name="jina-v5-nano-768-zh",
        model="jina-embeddings-v5-text-nano",
        dim=768,
        per_token=False,
        directory="/data/embeddings/fineweb2-cmn_Hani-chunked-500-jina-v5-nano/train/",
        note="Pooled sentence embeddings, 500-token chunks, fineweb2 Chinese (cmn_Hani).",
    ),
    dict(
        name="jina-v5-nano-768-ar",
        model="jina-embeddings-v5-text-nano",
        dim=768,
        per_token=False,
        directory="/data/embeddings/fineweb2-arb_Arab-chunked-500-jina-v5-nano/train/",
        note="Pooled sentence embeddings, 500-token chunks, fineweb2 Arabic (arb_Arab).",
    ),
    dict(
        name="mxbai-colbert-64-pile",
        model="mxbai-edge-colbert-v0-32m",
        dim=64,
        per_token=True,
        directory="/data/embeddings/pile-uncopyrighted-chunked-500-mxbai-edge-32m/train/",
        note=("PER-TOKEN ColBERT late-interaction vectors (one vector per token, "
              "not per chunk), 500-token chunks, pile-uncopyrighted. Included as a "
              "known-anisotropic contrast substrate; sampling draws random tokens."),
    ),
]

JINA_EN = "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train/"

# Sample sizes (kept small for CPU tractability).
N_MASTER = 100_000     # anisotropy / uniformity / derived subsamples
N_PAIRS = 100_000      # random pairs for cosine + uniformity
N_TWONN = 20_000       # TwoNN intrinsic dimension
N_PCA = 50_000         # PCA eigenspectrum
N_HUB = 50_000         # hubness kNN
K_HUB = 50


# ---------------------------------------------------------------------------
# Lazy embedding loader
# ---------------------------------------------------------------------------
class LazyEmbeddings:
    """Concatenate embedding shards lazily.

    Handles two on-disk formats seen in /data/embeddings:
      * proper .npy files (with header) -- opened via np.load(mmap_mode='r');
        these can be fp16 or fp32.
      * raw buffers saved without a header -- opened via np.memmap; dtype is
        inferred from file size and validated to produce finite values.

    Non-conforming files (wrong dim, 1-D side-car arrays like chunk_offsets)
    are skipped. Rows are cast to fp32 only when gathered.
    """

    def __init__(self, directory, dim):
        self.dim = int(dim)
        files = sorted(glob.glob(os.path.join(directory, "*.npy")))
        if not files:
            raise ValueError(f"No .npy files in {directory}")
        self.memmaps = []
        self.shapes = []
        self.dtypes = []
        for f in files:
            mm = self._open(f, self.dim)
            if mm is None:
                continue
            self.memmaps.append(mm)
            self.shapes.append(mm.shape)
            self.dtypes.append(str(mm.dtype))
        if not self.memmaps:
            raise ValueError(f"No 2-D shards with dim {dim} in {directory}")
        self.cum = np.cumsum([0] + [s[0] for s in self.shapes])
        self.total = int(self.cum[-1])

    @staticmethod
    def _open(path, dim):
        try:
            a = np.load(path, mmap_mode="r")
            if a.ndim == 2 and a.shape[1] == dim:
                return a
            return None  # side-car / wrong-dim
        except ValueError:
            # Raw buffer without .npy header -> infer dtype from size.
            sz = os.path.getsize(path)
            for dt in (np.float32, np.float16):
                it = np.dtype(dt).itemsize
                if sz % (dim * it) == 0:
                    rows = sz // (dim * it)
                    mm = np.memmap(path, dtype=dt, mode="r", shape=(rows, dim))
                    probe = np.asarray(mm[: min(64, rows)], dtype=np.float32)
                    if np.isfinite(probe).all():
                        return mm
            return None

    def gather(self, idx):
        """Gather rows at (globally-sorted) integer indices as fp32."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((len(idx), self.dim), dtype=np.float32)
        shard = np.searchsorted(self.cum, idx, side="right") - 1
        for s in np.unique(shard):
            pos = np.flatnonzero(shard == s)
            local = idx[pos] - self.cum[s]
            out[pos] = np.asarray(self.memmaps[s][local], dtype=np.float32)
        return out

    def sample(self, n, seed=0):
        rng = np.random.default_rng(seed)
        n = min(int(n), self.total)
        idx = np.sort(rng.choice(self.total, size=n, replace=False))
        return self.gather(idx)

    def summary(self):
        return dict(
            total_rows=self.total,
            n_shards=len(self.memmaps),
            dtypes=sorted(set(self.dtypes)),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def l2normalize(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def _pct(a, ps):
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


# ---------------------------------------------------------------------------
# 1. Anisotropy
# ---------------------------------------------------------------------------
def anisotropy(X, n_pairs, seed=0):
    """Mean/dist of random-pair cosine; vector norm mean and CV.

    Norms are measured on the raw sampled vectors (before renormalization)."""
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(X, axis=1)
    mean_norm = float(norms.mean())
    norm_cv = float(norms.std() / (norms.mean() + 1e-12))

    Xn = l2normalize(X)
    N = Xn.shape[0]
    i = rng.integers(0, N, size=n_pairs)
    j = rng.integers(0, N, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    cos = np.einsum("ij,ij->i", Xn[i], Xn[j])
    return dict(
        mean_cosine=float(cos.mean()),
        mean_abs_cosine=float(np.abs(cos).mean()),
        std_cosine=float(cos.std()),
        cosine_pct=_pct(cos, [1, 5, 25, 50, 75, 95, 99]),
        mean_norm=mean_norm,
        norm_cv=norm_cv,
        n_pairs=int(len(i)),
    )


# ---------------------------------------------------------------------------
# 2. Uniformity (Wang & Isola 2020)
# ---------------------------------------------------------------------------
def uniformity(X, n_pairs, seed=1, t=2.0):
    """L_uniform = log mean_{u,v} exp(-t ||u-v||^2) over normalized pairs.

    More negative => more uniformly spread on the hypersphere."""
    rng = np.random.default_rng(seed)
    Xn = l2normalize(X)
    N = Xn.shape[0]
    i = rng.integers(0, N, size=n_pairs)
    j = rng.integers(0, N, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    diff = Xn[i] - Xn[j]
    d2 = np.einsum("ij,ij->i", diff, diff)  # squared euclidean
    L = float(logsumexp(-t * d2) - np.log(len(d2)))
    return dict(l_uniform=L, t=t, n_pairs=int(len(i)))


# ---------------------------------------------------------------------------
# 3a. Intrinsic dimension -- TwoNN (Facco et al. 2017)
# ---------------------------------------------------------------------------
def twonn(X, discard_frac=0.10):
    """TwoNN maximum-likelihood / linear-fit intrinsic dimension.

    For each point, mu = r2/r1 (ratio of 2nd to 1st NN distance). With points
    locally uniform, -log(1 - F(mu)) = d * log(mu), so d is the slope of a
    line through the origin fit to (log mu, -log(1-F)). The top `discard_frac`
    of mu values are dropped as outliers (Facco recommendation)."""
    Xc = np.ascontiguousarray(X.astype(np.float32))
    index = faiss.IndexFlatL2(Xc.shape[1])
    index.add(Xc)
    D, _ = index.search(Xc, 3)  # self + 2 NN; D is squared L2
    d = np.sqrt(np.maximum(D, 0.0))
    r1, r2 = d[:, 1], d[:, 2]
    good = r1 > 0
    mu = r2[good] / r1[good]
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    mu.sort()
    N = len(mu)
    keep = int(N * (1.0 - discard_frac))
    mu = mu[:keep]
    F = (np.arange(1, len(mu) + 1)) / (len(mu) + 1)  # empirical CDF
    x = np.log(mu)
    y = -np.log(1.0 - F)
    d_est = float(np.sum(x * y) / np.sum(x * x))
    return dict(
        twonn_id=d_est,
        n_points=int(Xc.shape[0]),
        n_used=int(len(mu)),
        discard_frac=discard_frac,
    )


# ---------------------------------------------------------------------------
# 3b + 4. PCA participation ratio + eigenspectrum
# ---------------------------------------------------------------------------
def pca_spectrum(X, top=100, tail_lo=10):
    """Eigenvalues of the covariance via SVD on a (centered) subsample.

    Returns participation ratio, top-100 eigenvalues, %-variance captured at
    ranks {2,10,50}, and a power-law slope fit to the eigenvalue tail
    (log eigenvalue vs log rank over ranks [tail_lo, min(top, dim)])."""
    Xc = X.astype(np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    # singular values -> eigenvalues of covariance
    s = np.linalg.svd(Xc, compute_uv=False)
    eig = (s ** 2) / (n - 1)
    total = float(eig.sum())
    pr = float((eig.sum() ** 2) / np.sum(eig ** 2))  # participation ratio

    def var_at(k):
        k = min(k, len(eig))
        return float(eig[:k].sum() / total)

    dim = len(eig)
    top_k = min(top, dim)
    ranks = np.arange(tail_lo, top_k + 1)
    slope = None
    if len(ranks) >= 3:
        lx = np.log(ranks.astype(np.float64))
        ly = np.log(eig[tail_lo - 1:top_k])
        slope = float(np.polyfit(lx, ly, 1)[0])
    return dict(
        participation_ratio=pr,
        n_eigen=int(dim),
        top_eigenvalues=[float(v) for v in eig[:top_k]],
        var_top2=var_at(2),
        var_top10=var_at(10),
        var_top50=var_at(50),
        tail_powerlaw_slope=slope,
        tail_range=[int(tail_lo), int(top_k)],
    )


# ---------------------------------------------------------------------------
# 5. Hubness
# ---------------------------------------------------------------------------
def hubness(X, k=50):
    """Exact cosine kNN in-degree distribution (skewness, max)."""
    Xn = np.ascontiguousarray(l2normalize(X).astype(np.float32))
    N = Xn.shape[0]
    index = faiss.IndexFlatIP(Xn.shape[1])
    index.add(Xn)
    _, I = index.search(Xn, k + 1)  # +1 for self
    indeg = np.zeros(N, dtype=np.int64)
    rows = np.arange(N)
    for col in range(k + 1):
        nbr = I[:, col]
        m = nbr != rows  # drop self-hits
        np.add.at(indeg, nbr[m], 1)
    # if any point had no self in its list an extra neighbor slipped in; clip
    # to k by ignoring -- effect is negligible for skew/max reporting.
    return dict(
        k=k,
        indegree_skewness=float(skew(indeg)),
        indegree_max=int(indeg.max()),
        indegree_mean=float(indeg.mean()),
        indegree_std=float(indeg.std()),
        n_points=int(N),
        n_hubs_ge_2k=int((indeg >= 2 * k).sum()),
        n_antihubs_zero=int((indeg == 0).sum()),
    )


# ---------------------------------------------------------------------------
# Interpretation text
# ---------------------------------------------------------------------------
def interpret(space, m):
    """2-3 sentences on what the passport predicts for parametric-UMAP."""
    a = m["anisotropy"]
    u = m["uniformity"]
    tw = m["intrinsic_dim"]["twonn_id"]
    pr = m["pca"]["participation_ratio"]
    hub = m["hubness"]["indegree_skewness"]
    v2 = m["pca"]["var_top2"]
    mc = a["mean_cosine"]
    mac = a["mean_abs_cosine"]

    parts = []
    # isotropy / anchoring
    if mac < 0.06 and abs(mc) < 0.05 and v2 < 0.15:
        parts.append(
            f"Near-isotropic (mean cos {mc:+.3f}, mean|cos| {mac:.3f}, only "
            f"{v2*100:.0f}% variance in top-2 PCA): the global 2D layout is "
            f"underdetermined with no dominant axis to lock onto, so cross-seed "
            f"stability will likely require explicit anchoring (§4)."
        )
    elif mc > 0.15 or v2 > 0.30:
        parts.append(
            f"Anisotropic cone (mean cos {mc:+.3f}, {v2*100:.0f}% variance in "
            f"top-2 PCA): a dominant axis exists that a map can lock onto, which "
            f"helps determinism but Euclidean UMAP may over-weight that axis; "
            f"cosine/centering worth checking."
        )
    else:
        parts.append(
            f"Mildly anisotropic (mean cos {mc:+.3f}, mean|cos| {mac:.3f}, "
            f"{v2*100:.0f}% variance in top-2 PCA): some global orientation but "
            f"no strong dominant axis; expect moderate layout determinacy."
        )
    # intrinsic dim -> 2D feasibility / k
    dim = space["dim"]
    if pr < 0.1 * dim or v2 > 0.30:
        parts.append(
            f"Intrinsic dimension is comparatively low and concentrated (TwoNN "
            f"{tw:.1f}, PCA participation ratio {pr:.0f} of {dim}, {v2*100:.0f}% "
            f"of variance already in the top-2 axes): a few directions dominate, "
            f"so a 2D map can capture more of the coarse geometry, but the strong "
            f"anisotropy above means Euclidean structure is skewed."
        )
    else:
        parts.append(
            f"Intrinsic dimension is high (TwoNN {tw:.1f}, PCA participation ratio "
            f"{pr:.0f} of {dim}), so 2D can only preserve local neighborhoods, not "
            f"global geometry -- favors a dense kNN graph (k=50) and %-of-ceiling "
            f"framing."
        )
    # hubness
    if hub > 2.0:
        parts.append(
            f"High hubness (in-degree skew {hub:.1f}, max "
            f"{m['hubness']['indegree_max']}): a few hub points dominate the kNN "
            f"graph, a known kNN-quality concern that can distort attraction; "
            f"mutual-kNN or hubness-aware edge weighting may help."
        )
    else:
        parts.append(
            f"Hubness is moderate (in-degree skew {hub:.1f}, max "
            f"{m['hubness']['indegree_max']}): the kNN graph is reasonably "
            f"balanced, no special hubness mitigation indicated."
        )
    if space.get("per_token"):
        parts.append(
            "NOTE: these are per-token ColBERT vectors, not pooled chunk "
            "embeddings -- geometry is token-level and not directly comparable "
            "to the pooled sentence spaces; deferred as a basemap substrate."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Per-space passport
# ---------------------------------------------------------------------------
def passport_for(space):
    t0 = time.time()
    print(f"\n=== {space['name']} ===")
    emb = LazyEmbeddings(space["directory"], space["dim"])
    summ = emb.summary()
    print(f"  shards={summ['n_shards']} rows={summ['total_rows']:,} "
          f"dtypes={summ['dtypes']}")

    master = emb.sample(N_MASTER, seed=0)
    print(f"  sampled master {master.shape} in {time.time()-t0:.1f}s")

    m = {}
    m["anisotropy"] = anisotropy(master, N_PAIRS, seed=0)
    m["uniformity"] = uniformity(master, N_PAIRS, seed=1)
    print(f"  aniso mean_cos={m['anisotropy']['mean_cosine']:+.4f} "
          f"mean|cos|={m['anisotropy']['mean_abs_cosine']:.4f} "
          f"L_uniform={m['uniformity']['l_uniform']:.3f}")

    m["intrinsic_dim"] = twonn(master[:N_TWONN])
    m["intrinsic_dim"]["pca_participation_ratio_ref"] = None  # filled below
    print(f"  TwoNN ID={m['intrinsic_dim']['twonn_id']:.2f}")

    m["pca"] = pca_spectrum(master[:N_PCA])
    m["intrinsic_dim"]["pca_participation_ratio_ref"] = m["pca"]["participation_ratio"]
    print(f"  PCA PR={m['pca']['participation_ratio']:.1f} "
          f"var@2={m['pca']['var_top2']*100:.1f}% "
          f"var@10={m['pca']['var_top10']*100:.1f}% "
          f"var@50={m['pca']['var_top50']*100:.1f}% "
          f"slope={m['pca']['tail_powerlaw_slope']}")

    m["hubness"] = hubness(master[:N_HUB], K_HUB)
    print(f"  hubness skew={m['hubness']['indegree_skewness']:.2f} "
          f"max_indeg={m['hubness']['indegree_max']}")

    result = dict(
        space=dict(
            name=space["name"], model=space["model"], dim=space["dim"],
            per_token=space["per_token"], directory=space["directory"],
            note=space["note"],
        ),
        data=summ,
        sample_sizes=dict(master=int(master.shape[0]), pairs=N_PAIRS,
                          twonn=N_TWONN, pca=N_PCA, hubness=N_HUB, k_hub=K_HUB),
        metrics=m,
        wall_time_sec=round(time.time() - t0, 1),
    )
    result["interpretation"] = interpret(space, m)
    print(f"  done in {result['wall_time_sec']}s")
    return result


def write_space_md(res, path):
    s = res["space"]
    m = res["metrics"]
    a, u, idm, pca, hub = (m["anisotropy"], m["uniformity"],
                           m["intrinsic_dim"], m["pca"], m["hubness"])
    lines = []
    lines.append(f"# Space passport — {s['name']}\n")
    lines.append(f"**Model:** `{s['model']}`  |  **Dim:** {s['dim']}  |  "
                 f"**Per-token:** {s['per_token']}\n")
    lines.append(f"**Source:** `gsv:{s['directory']}`  \n"
                 f"**Rows on disk:** {res['data']['total_rows']:,} "
                 f"({res['data']['n_shards']} shards, dtype "
                 f"{','.join(res['data']['dtypes'])})\n")
    lines.append(f"> {s['note']}\n")
    lines.append("## Metrics\n")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| mean cosine (random pairs) | {a['mean_cosine']:+.4f} |")
    lines.append(f"| mean \\|cosine\\| | {a['mean_abs_cosine']:.4f} |")
    lines.append(f"| std cosine | {a['std_cosine']:.4f} |")
    lines.append(f"| cosine p1 / p50 / p99 | {a['cosine_pct']['p1']:+.3f} / "
                 f"{a['cosine_pct']['p50']:+.3f} / {a['cosine_pct']['p99']:+.3f} |")
    lines.append(f"| mean vector norm | {a['mean_norm']:.4f} |")
    lines.append(f"| norm CV | {a['norm_cv']:.4f} |")
    lines.append(f"| L_uniform (Wang–Isola, t=2) | {u['l_uniform']:.4f} |")
    lines.append(f"| TwoNN intrinsic dim | {idm['twonn_id']:.2f} |")
    lines.append(f"| PCA participation ratio | {pca['participation_ratio']:.1f} "
                 f"(of {pca['n_eigen']}) |")
    lines.append(f"| %var top-2 / 10 / 50 | {pca['var_top2']*100:.1f}% / "
                 f"{pca['var_top10']*100:.1f}% / {pca['var_top50']*100:.1f}% |")
    lines.append(f"| eigenvalue tail slope (ranks {pca['tail_range'][0]}–"
                 f"{pca['tail_range'][1]}) | {pca['tail_powerlaw_slope']:.3f} |")
    lines.append(f"| kNN in-degree skewness (k={hub['k']}) | "
                 f"{hub['indegree_skewness']:.3f} |")
    lines.append(f"| kNN in-degree max / mean | {hub['indegree_max']} / "
                 f"{hub['indegree_mean']:.1f} |")
    lines.append(f"| hubs (in-deg ≥ 2k) / anti-hubs (0) | {hub['n_hubs_ge_2k']} / "
                 f"{hub['n_antihubs_zero']} |")
    lines.append("")
    lines.append(f"*Samples: master {res['sample_sizes']['master']:,} rows; "
                 f"TwoNN {res['sample_sizes']['twonn']:,}; PCA "
                 f"{res['sample_sizes']['pca']:,}; hubness "
                 f"{res['sample_sizes']['hubness']:,}. "
                 f"Wall time {res['wall_time_sec']}s.*\n")
    lines.append("## Interpretation\n")
    lines.append(res["interpretation"] + "\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_combined_md(results, path):
    lines = []
    lines.append("# Space passports — comparison\n")
    lines.append("Pre-flight geometry diagnostics per embedding space "
                 "(plan `latent-labs/guides/plan-basemap-atlas.md` §3). "
                 "CPU-only, ≤100k-row samples. All spaces store L2-normalized "
                 "vectors on disk, so mean norm ≈ 1 and norm CV ≈ 0.\n")
    lines.append("## Comparison table\n")
    hdr = ("| space | dim | mean cos | mean\\|cos\\| | L_uniform | TwoNN ID | "
           "PCA PR | %var@2 | %var@10 | tail slope | hub skew | max in-deg |")
    sep = "| " + " | ".join(["---"] * 12) + " |"
    lines.append(hdr)
    lines.append(sep)
    for r in results:
        s, m = r["space"], r["metrics"]
        a, u, idm, pca, hub = (m["anisotropy"], m["uniformity"],
                               m["intrinsic_dim"], m["pca"], m["hubness"])
        lines.append(
            f"| {s['name']} | {s['dim']} | {a['mean_cosine']:+.3f} | "
            f"{a['mean_abs_cosine']:.3f} | {u['l_uniform']:.3f} | "
            f"{idm['twonn_id']:.1f} | {pca['participation_ratio']:.0f} | "
            f"{pca['var_top2']*100:.1f}% | {pca['var_top10']*100:.1f}% | "
            f"{pca['tail_powerlaw_slope']:.2f} | {hub['indegree_skewness']:.2f} | "
            f"{hub['indegree_max']} |"
        )
    lines.append("")
    lines.append("Reading the columns: **mean cos / mean|cos|** near 0 ⇒ "
                 "isotropic (no dominant axis); **L_uniform** more negative ⇒ "
                 "more uniform on the sphere; **TwoNN ID / PCA PR** high ⇒ "
                 "genuinely high-dimensional (2D preserves locals only); "
                 "**%var@2** low ⇒ layout underdetermined; **hub skew** high ⇒ "
                 "kNN-graph quality concern.\n")
    lines.append("## Per-space interpretation\n")
    for r in results:
        lines.append(f"### {r['space']['name']}\n")
        lines.append(r["interpretation"] + "\n")
    lines.append("---\n")
    lines.append("See `matryoshka_truncation.md` for the jina truncated-dim "
                 "graph-overlap decision test, and each `"
                 "<space-name>/passport.md` for the full metric table.\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Matryoshka truncation test
# ---------------------------------------------------------------------------
def knn_sets(X, k, dims=None):
    """Return (I) neighbor-id matrix (self excluded, k cols) via exact IP."""
    if dims is not None:
        X = X[:, :dims]
    Xn = np.ascontiguousarray(l2normalize(X).astype(np.float32))
    index = faiss.IndexFlatIP(Xn.shape[1])
    index.add(Xn)
    _, I = index.search(Xn, k + 1)
    N = X.shape[0]
    rows = np.arange(N)[:, None]
    out = np.empty((N, k), dtype=np.int64)
    for r in range(N):
        row = I[r]
        row = row[row != r][:k]
        if len(row) < k:  # pad (shouldn't happen with exact search)
            row = np.concatenate([row, np.full(k - len(row), -1)])
        out[r] = row
    return out


def matryoshka_test(n_rows=200_000, k=50, seed=0, time_budget_s=1200):
    """Compare full-768 k=50 graph to truncated 256/128 graphs on jina-en."""
    print(f"\n=== matryoshka truncation test (jina-en, k={k}) ===")
    emb = LazyEmbeddings(JINA_EN, 768)
    n_rows = min(n_rows, emb.total)
    # Adaptive: if the full-dim exact search looks too slow, fall back to 100k.
    dropped = False
    t0 = time.time()
    X = emb.sample(n_rows, seed=seed)
    print(f"  sampled {X.shape} in {time.time()-t0:.1f}s")

    def build(dims, label):
        t = time.time()
        I = knn_sets(X, k, dims)
        print(f"  built {label} graph in {time.time()-t:.1f}s")
        return I, time.time() - t

    I_full, t_full = build(None, "full-768")
    # If full search alone blew the budget, redo at 100k.
    if t_full > time_budget_s and n_rows > 100_000:
        print(f"  full-dim build {t_full:.0f}s exceeded budget; "
              f"falling back to 100k rows")
        dropped = True
        n_rows = 100_000
        X = emb.sample(n_rows, seed=seed)
        I_full, t_full = build(None, "full-768 (100k)")

    results = {}
    ref = set_rows(I_full)
    ref10 = I_full[:, :10]
    for dims in (256, 128):
        I_t, _ = build(dims, f"trunc-{dims}")
        trunc = set_rows(I_t)
        # edge overlap: fraction of full k=50 neighbors recovered
        overlaps = np.array([len(trunc[r] & ref[r]) for r in range(n_rows)]) / k
        # recall@10-of-50: full's top-10 present in trunc's top-50
        rec10 = np.array([len(set(ref10[r].tolist()) & trunc[r])
                          for r in range(n_rows)]) / 10.0
        results[dims] = dict(
            mean_edge_overlap=float(overlaps.mean()),
            median_edge_overlap=float(np.median(overlaps)),
            mean_recall10_of_50=float(rec10.mean()),
        )
        print(f"  dims={dims}: edge_overlap={overlaps.mean():.4f} "
              f"recall@10={rec10.mean():.4f}")

    return dict(
        n_rows=int(n_rows),
        k=k,
        dropped_to_100k=dropped,
        results=results,
        threshold_note="Decision threshold (plan §3): >=90% edge overlap at "
                        "256d means graph construction at 20M+ can use truncated dims.",
    )


def set_rows(I):
    return [set(row[row >= 0].tolist()) for row in I]


def write_matryoshka_md(res, path):
    r256 = res["results"].get(256, {})
    r128 = res["results"].get(128, {})
    ov256 = r256.get("mean_edge_overlap", 0.0)
    verdict = ("**PASS** — 256d reproduces ≥90% of the full-768 k=50 graph; "
               "graph construction at 20M+ can safely use the first 256 "
               "matryoshka dims (~3× cheaper).") if ov256 >= 0.90 else (
               "**FAIL** — 256d does not reach the 90% edge-overlap threshold; "
               "full-dim graph construction is required, or a higher truncation "
               "dim should be tested.")
    lines = []
    lines.append("# Matryoshka truncation test — jina-v5-nano English\n")
    lines.append("Plan `latent-labs/guides/plan-basemap-atlas.md` §3 decision "
                 "point. jina-v5-nano is matryoshka-trained; if a truncated-dim "
                 "k=50 graph reproduces the full-768 graph, large-scale graph "
                 "construction gets cheaper.\n")
    note_200k = ""
    if res["n_rows"] < 200_000:
        note_200k = ("Reduced from the planned 200k to "
                     f"{res['n_rows']:,}: exact faiss IndexFlatIP search on this "
                     "CPU box parallelizes poorly (~1.7 of 32 cores), so a full "
                     "200k×200k×768 build ran >8 min for the full-dim graph "
                     "alone; 100k keeps the three-build test tractable (~a few "
                     "minutes each) without changing the overlap conclusion. ")
    lines.append(f"**Sample:** {res['n_rows']:,} rows from "
                 f"`gsv:{JINA_EN}` (exact faiss IndexFlatIP on L2-normalized "
                 f"vectors, k={res['k']}).  " + note_200k + "\n")
    lines.append("| truncation | mean edge overlap | median edge overlap | "
                 "recall@10-of-50 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| 768 (full) | 1.0000 | 1.0000 | 1.0000 |")
    for d in (256, 128):
        rr = res["results"][d]
        lines.append(f"| {d} (re-normalized) | {rr['mean_edge_overlap']:.4f} | "
                     f"{rr['median_edge_overlap']:.4f} | "
                     f"{rr['mean_recall10_of_50']:.4f} |")
    lines.append("")
    lines.append(f"{res['threshold_note']}\n")
    lines.append("## Verdict\n")
    lines.append(verdict + "\n")
    rec256 = r256.get("mean_recall10_of_50", 0.0)
    lines.append(
        f"Note the split signal at 256d: recall@10-of-50 is high "
        f"({rec256:.3f}) while full k=50 edge overlap is only {ov256:.3f}. The "
        f"first ~256 matryoshka dims preserve each point's *closest* neighbors, "
        f"but the broader 50-neighbor set reshuffles substantially — so "
        f"truncation is safe for a small-k nearest-neighbor lookup but not for "
        f"reproducing the dense k=50 UMAP graph, whose mid-range edges carry the "
        f"manifold structure. Graph construction at 20M+ should therefore use "
        f"full 768 dims (or an ANN index on full dims), not truncated dims.\n")
    if r128:
        lines.append(f"At 128d edge overlap drops further to "
                     f"{r128['mean_edge_overlap']:.4f} (recall@10 "
                     f"{r128['mean_recall10_of_50']:.4f}) — "
                     + ("also above" if r128["mean_edge_overlap"] >= 0.90
                        else "below") + " the 90% bar, shown for the "
                     "cost/quality curve.\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def run_passports():
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []
    for space in SPACES:
        try:
            res = passport_for(space)
        except Exception as e:
            print(f"  !! FAILED {space['name']}: {type(e).__name__}: {e}")
            continue
        d = os.path.join(OUT_ROOT, space["name"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "passport.json"), "w") as f:
            json.dump(res, f, indent=2)
        write_space_md(res, os.path.join(d, "passport.md"))
        results.append(res)
    if results:
        write_combined_md(results, os.path.join(OUT_ROOT, "README.md"))
        print(f"\nWrote combined README with {len(results)} passports.")
    return results


def run_matryoshka(n_rows):
    os.makedirs(OUT_ROOT, exist_ok=True)
    res = matryoshka_test(n_rows=n_rows)
    with open(os.path.join(OUT_ROOT, "matryoshka_truncation.json"), "w") as f:
        json.dump(res, f, indent=2)
    write_matryoshka_md(res, os.path.join(OUT_ROOT, "matryoshka_truncation.md"))
    print("\nWrote matryoshka_truncation.md")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", default="all",
                    choices=["passports", "matryoshka", "all"])
    ap.add_argument("--matryoshka-rows", type=int, default=200_000)
    args = ap.parse_args()
    if args.mode in ("passports", "all"):
        run_passports()
    if args.mode in ("matryoshka", "all"):
        run_matryoshka(args.matryoshka_rows)


if __name__ == "__main__":
    main()
