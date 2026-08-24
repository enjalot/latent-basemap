#!/usr/bin/env python3
"""Image-embedding map suites (owner orders 2026-08-21): exact graph + best-3 arms.

Datasets:
  bl-siglip-1m   BL SigLIP2, 1,080,814 x 1152 fp16 (substrate export)
  sisap-clip-2m  LAION CLIP768v2 (SISAP 30M h5), every 15th row -> 2,024,617
                 x 768 fp16 unit-norm. NOTE: this corpus is CLIP ViT-L/14
                 ("clip768v2"), not SigLIP — named accordingly.

Per dataset, three phases (two envs):
  knn    (.venv, GPU): exact k=15 cosine kNN, chunked fp16 matmul w/ fp32
         accumulate; db resident (2.5 GB / 3.1 GB VRAM).
  fuzzy  (umap06dev-env, CPU): umap-learn fuzzy_simplicial_set on the
         precomputed kNN (+self col, n_neighbors=16 — the 2M-artifact
         convention) -> trainer-format edges npz.
  train  (.venv, GPU): the window's best-3 recipes, config parity via
         knobs_2m BASE_KWARGS/MD; dose x2 horizon = 2 x 0.6782 draws/edge
         (R0217 base-dose law). quick-FFR vs the dataset's own graph +
         density render (+ per-subset overlays when subsets exist);
         summaries land in /data/latent-basemap/sandbox/<dataset>/<arm>/.

Usage: image_map_pipeline.py <dataset> knn|fuzzy|train
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

K = 15
SEED = 42
SANDBOX = Path("/data/latent-basemap/sandbox")

ARMS = {
    "promoted-fneg10": {"md": "000", "extra": {"fneg_weight": 1.0}},
    "fneg10-tanh4": {"md": "000", "extra": {"fneg_weight": 1.0,
                                            "neg_tanh_gamma": 4.0}},
    "md005-fneg10": {"md": "005", "extra": {"fneg_weight": 1.0}},
    # owner 2026-08-22: the missing kernel x tanh cell (anti-collapse look
    # with the tanh win). Re-running train on a dataset adds just this arm.
    "md005-fneg10-tanh4": {"md": "005", "extra": {"fneg_weight": 1.0,
                                                  "neg_tanh_gamma": 4.0}},
    # owner 2026-08-23: the confirmed winner (x8+tanh4+pos10) at the airier
    # md010 kernel, for the image/jina spaces.
    "composed-x8-md010": {"md": "010", "dose": 8,
                          "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                    "pos_ratio": 0.10}},
    # owner 2026-08-23 v2: the best-efficiency guess (see knobs_2m
    # umap-md010-h1024L2mlp-bs16k-x4-winner).
    "efficiency-x4-md010": {"md": "010", "dose": 4,
                            "extra": {"fneg_weight": 1.0,
                                      "neg_tanh_gamma": 4.0,
                                      "pos_ratio": 0.10,
                                      "rankneg_window": 500_000,
                                      "hidden_dim": 1024, "n_layers": 2,
                                      "architecture": "mlp",
                                      "batch_size": 16384}},
}


def _bl_load() -> np.ndarray:
    sub = Path("/data/latent-basemap/substrates/bl-siglip2-1m")
    return np.array(np.load(sub / "substrate.f16.npy", mmap_mode="r"),
                    dtype=np.float32)


def _bl_subsets():
    import pyarrow.parquet as pq
    sub = Path("/data/latent-basemap/substrates/bl-siglip2-1m")
    col = pq.read_table(sub / "rows.parquet", columns=["subset"])
    return col["subset"].to_numpy()


def _sisap_load() -> np.ndarray:
    import h5py
    with h5py.File("/data/embeddings/laion2b-en-clip768v2-sisap/"
                   "laion2B-en-clip768v2-n=30M.h5", "r") as f:
        return np.asarray(f["emb"][::15], dtype=np.float32)


# jina substrates come from the PROMPTED re-embed (owner ruling 2026-08-22:
# the original corpora were embedded raw, the known July problem — "Document: "
# shifts cosine to 0.73–0.94, so raw-trained maps can't receive normally-
# embedded datasets). embed_jina_prompted_subsets.py produces these.
_JINA_PROMPTED = Path("/data/latent-basemap/substrates/jina-prompted")
_JINA_EN_NAMES = ("fineweb-edu", "redpajama", "pile")
_JINA_EN_PER = (666_667, 666_667, 666_666)
_JINA_LANGS = ("arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek",
               "fra_Latn", "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan",
               "kor_Hang", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl",
               "spa_Latn", "swe_Latn", "tha_Thai", "tur_Latn", "vie_Latn")


def _jina_en_load() -> np.ndarray:
    return np.array(np.load(_JINA_PROMPTED / "en-2m.f16.npy", mmap_mode="r"),
                    dtype=np.float32)


def _jina_en_subsets():
    return np.repeat(_JINA_EN_NAMES, _JINA_EN_PER)


def _jina_multi_load() -> np.ndarray:
    # half English (per-corpus prefixes of the en blocks), half multilingual.
    en = np.load(_JINA_PROMPTED / "en-2m.f16.npy", mmap_mode="r")
    offs = np.cumsum((0,) + _JINA_EN_PER[:-1])
    en_half = [np.asarray(en[o:o + p], dtype=np.float32)
               for o, p in zip(offs, (333_334, 333_333, 333_333))]
    ml = np.array(np.load(_JINA_PROMPTED / "multi-1m.f16.npy", mmap_mode="r"),
                  dtype=np.float32)
    return np.concatenate(en_half + [ml])


def _jina_multi_subsets():
    en = np.repeat(_JINA_EN_NAMES, (333_334, 333_333, 333_333))
    ml = np.repeat([l.split("_")[0] for l in _JINA_LANGS], 50_000)
    return np.concatenate([en, ml])


def _ca_load() -> np.ndarray:
    # Community Archive tweets (MiniLM), strided to <=2M for the register probe.
    import glob as _g
    shards = [np.load(f, mmap_mode="r") for f in sorted(_g.glob(
        "/data/embeddings/communityarchive-tweets-all-MiniLM-L6-v2/"
        "train/*.npy"))]
    n = sum(s.shape[0] for s in shards)
    stride = max(1, -(-n // 2_000_000))
    parts, offset = [], 0
    for s in shards:
        start = (-offset) % stride
        parts.append(np.asarray(s[start::stride], dtype=np.float32))
        offset += s.shape[0]
    return np.concatenate(parts)


_REDDITMIX = Path("/data/latent-basemap/substrates/minilm-redditmix-2m")


def _redditmix_load() -> np.ndarray:
    return np.array(np.load(_REDDITMIX / "substrate.f32.npy", mmap_mode="r"))


def _redditmix_subsets():
    return np.load(_REDDITMIX / "subsets.npy", allow_pickle=True)


def _reddit_load() -> np.ndarray:
    # every 5th row of the 10M reddit-tldr17 MiniLM embeddings -> 2M sample.
    # Used for knn/fuzzy TRUTH ONLY (the OOD probe projects these rows through
    # frozen maps; no map is trained on reddit).
    import glob as _g
    shards = [np.load(f, mmap_mode="r") for f in sorted(_g.glob(
        "/data/embeddings/reddit-tldr17-chunked-120-all-MiniLM-L6-v2/"
        "train/*.npy"))]
    parts, offset = [], 0
    for s in shards:
        start = (-offset) % 5
        parts.append(np.asarray(s[start::5], dtype=np.float32))
        offset += s.shape[0]
    return np.concatenate(parts)


#: the redditmix suite trains the promoted control + the composed winner
#: (dose is per-arm here; default elsewhere stays x2).
_REDDITMIX_ARMS = {
    "promoted-fneg10": {"md": "000", "extra": {"fneg_weight": 1.0}},
    "composed-x8": {"md": "000", "dose": 8,
                    "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                              "pos_ratio": 0.10}},
}

def _sisap_dedup_load() -> np.ndarray:
    return np.array(np.load(
        "/data/latent-basemap/substrates/sisap-clip-2m-dedup/substrate.f32.npy",
        mmap_mode="r"))


#: dedup comparison arms — SAME two recipes the original sisap suite ran, so
#: the render/satellite comparison is apples-to-apples. NOTE the FFR caveat:
#: dup rows have trivially-findable neighbors (their own copies), so the
#: original suite's FFR is inflated; expect dedup FFR lower for metric
#: reasons even when the map is better.
_SISAP_DEDUP_ARMS = {
    "promoted-fneg10": {"md": "000", "extra": {"fneg_weight": 1.0}},
    "fneg10-tanh4": {"md": "000", "extra": {"fneg_weight": 1.0,
                                            "neg_tanh_gamma": 4.0}},
}

_CHAMPION_500K = {"md": "000", "dose": 4,
                  "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                            "pos_ratio": 0.10, "rankneg_window": 500_000,
                            "batch_size": 16384}}
_CHAMPION_270K = {"md": "000", "dose": 4,
                  "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                            "pos_ratio": 0.10, "rankneg_window": 270_000,
                            "batch_size": 16384}}

DATASETS = {
    "bl-siglip-1m": {"load": _bl_load, "subsets": _bl_subsets,
                     "arms": {**ARMS, "champion-bs16k": _CHAMPION_270K}},
    "sisap-clip-2m": {"load": _sisap_load, "subsets": None,
                      "arms": {**ARMS, "champion-bs16k": _CHAMPION_500K}},
    "jina-en-2m": {"load": _jina_en_load, "subsets": _jina_en_subsets},
    "jina-multi-2m": {"load": _jina_multi_load, "subsets": _jina_multi_subsets,
                      "arms": {**ARMS, "champion-bs16k": _CHAMPION_500K}},
    "reddit-2m": {"load": _reddit_load, "subsets": None},
    "communityarchive-2m": {"load": _ca_load, "subsets": None},
    "minilm-redditmix-2m": {"load": _redditmix_load,
                            "subsets": _redditmix_subsets,
                            "arms": _REDDITMIX_ARMS},
    "sisap-clip-2m-dedup": {"load": _sisap_dedup_load, "subsets": None,
                            "arms": _SISAP_DEDUP_ARMS},
    "jina-multi-6m": {
        "load": lambda: np.array(np.load(
            _JINA_PROMPTED / "substrate-6250k.f16.npy", mmap_mode="r"),
            dtype=np.float32),
        "subsets": None,
        "arms": {
            # champion-class at scale (bs16k acquitted: 0.4646 vs 0.4600).
            # rankneg REMOVED (2026-08-24 verdict: fixed 500K window = 8%
            # fraction at 6.25M dragged FFR -0.07; norank scales cleanly).
            "champion-bs16k": {"md": "000", "dose": 4,
                               "extra": {"fneg_weight": 1.0,
                                         "neg_tanh_gamma": 4.0,
                                         "pos_ratio": 0.10,
                                         "batch_size": 16384,
                                         "gpu_resident_vram_budget_gb": 22.0}},
            # does the efficiency recipe survive scale on the easy space?
            "efficiency-x4-md010": {"md": "010", "dose": 4,
                                    "extra": {"fneg_weight": 1.0,
                                              "neg_tanh_gamma": 4.0,
                                              "pos_ratio": 0.10,
                                              "hidden_dim": 1024,
                                              "n_layers": 2,
                                              "architecture": "mlp",
                                              "batch_size": 16384,
                                              "gpu_resident_vram_budget_gb": 22.0}},
        }},
}


def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def knn(ds: str) -> int:
    import torch

    out = SANDBOX / ds
    out.mkdir(parents=True, exist_ok=True)
    x = _norm(DATASETS[ds]["load"]())
    n = x.shape[0]
    db = torch.from_numpy(x).half().cuda()
    del x
    idx_out = np.empty((n, K), dtype=np.int32)
    dist_out = np.empty((n, K), dtype=np.float32)
    q_chunk, d_chunk = 4096, 262_144
    t0 = time.time()
    with torch.no_grad():
        for qs in range(0, n, q_chunk):
            q = db[qs:qs + q_chunk].float()
            best_s = torch.full((q.shape[0], K + 1), -2.0, device="cuda")
            best_i = torch.zeros((q.shape[0], K + 1), dtype=torch.long,
                                 device="cuda")
            for dstart in range(0, n, d_chunk):
                sims = q @ db[dstart:dstart + d_chunk].float().T
                s, i = torch.topk(sims, min(K + 1, sims.shape[1]), dim=1)
                best_s, sel = torch.topk(
                    torch.cat([best_s, s], dim=1), K + 1, dim=1)
                best_i = torch.gather(
                    torch.cat([best_i, i + dstart], dim=1), 1, sel)
            rows = torch.arange(qs, qs + q.shape[0], device="cuda")
            is_self = best_i == rows.unsqueeze(1)
            # drop the self hit; when self was not in top-K+1 (duplicate-heavy
            # rows), drop the worst instead. Vectorized keep-mask.
            keep = ~is_self
            no_self = keep.all(dim=1)
            keep[no_self, K] = False       # drop the (K+1)-th
            sel_i = best_i[keep].view(q.shape[0], K)
            sel_s = best_s[keep].view(q.shape[0], K)
            idx_out[qs:qs + q.shape[0]] = sel_i.cpu().numpy().astype(np.int32)
            dist_out[qs:qs + q.shape[0]] = (
                1.0 - sel_s.cpu().numpy()).astype(np.float32)
    np.save(out / "knn_indices.npy", idx_out)
    np.save(out / "knn_dists.npy", np.clip(dist_out, 0.0, None))
    print(f"{ds}: exact k{K} for {n:,} rows in {(time.time()-t0)/60:.1f} min")
    return 0


def fuzzy(ds: str) -> int:
    from umap.umap_ import fuzzy_simplicial_set

    out = SANDBOX / ds
    idx = np.load(out / "knn_indices.npy")
    dst = np.load(out / "knn_dists.npy")
    n = idx.shape[0]
    knn_i = np.concatenate(
        [np.arange(n, dtype=np.int64)[:, None], idx.astype(np.int64)], axis=1)
    knn_d = np.concatenate([np.zeros((n, 1), dtype=np.float32), dst], axis=1)
    g, _, _ = fuzzy_simplicial_set(
        X=np.empty((n, 1), dtype=np.float32), n_neighbors=K + 1,
        random_state=SEED, metric="euclidean",
        knn_indices=knn_i, knn_dists=knn_d)
    g = g.tocoo()
    np.savez(out / "edges-k15-fuzzy.npz",
             sources=g.row.astype(np.int32), targets=g.col.astype(np.int32),
             weights=g.data.astype(np.float32), n_nodes=np.int64(n))
    print(f"{ds}: fuzzy edges {len(g.row):,} directed ({len(g.row)/n:.1f}/node)")
    return 0


def train(ds: str) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from knobs_2m import BASE_KWARGS, MD, quick_ffr
    from map_renders import binned_counts, render_png, robust_extent

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    out = SANDBOX / ds
    edges = out / "edges-k15-fuzzy.npz"
    e = int(len(np.load(edges, mmap_mode="r")["sources"]))
    n_pos_per_batch = max(1, int(round(
        BASE_KWARGS["batch_size"] * BASE_KWARGS["pos_ratio"])))
    x = _norm(DATASETS[ds]["load"]())
    subsets = DATASETS[ds]["subsets"]() if DATASETS[ds]["subsets"] else None

    for arm, spec in DATASETS[ds].get("arms", ARMS).items():
        dose = spec.get("dose", 2)
        horizon = int(round(dose * 0.6782 * e / n_pos_per_batch))
        arm_batch = spec.get("extra", {}).get("batch_size",
                                              BASE := 8192)
        arm_pos = spec.get("extra", {}).get("pos_ratio", 0.05)
        n_epochs = max(1, math.ceil(horizon * arm_batch * arm_pos / e))
        d = out / arm
        if (d / "summary.json").exists():
            print(f"{ds}/{arm}: done, skip")
            continue
        d.mkdir(parents=True, exist_ok=True)
        kwargs = dict(BASE_KWARGS)
        kwargs.update({"low_dim_kernel": "umap", **MD[spec["md"]],
                       **spec["extra"], "n_epochs": n_epochs,
                       "total_steps_estimate": horizon})
        model = ParametricUMAP(**kwargs)
        t0 = time.time()
        model.fit(x, precomputed_edges_path=str(edges), random_state=SEED)
        xy = np.asarray(model.transform(x, batch_size=8192), dtype=np.float32)
        wall = time.time() - t0
        np.save(d / "coordinates.npy", xy)
        model.save(str(d / "model.pt"))
        frame = robust_extent(xy)
        render_png(binned_counts(xy, frame), d / "density.png")
        if subsets is not None:
            for name in np.unique(subsets):
                render_png(binned_counts(xy[subsets == name], frame),
                           d / f"density-{name}.png")
        ffr = quick_ffr(xy, edges, x.shape[0])
        (d / "summary.json").write_text(json.dumps({
            "arm": f"{ds}--{arm}", "rung": ds,
            "overrides": {"low_dim_kernel": "umap", **MD[spec["md"]],
                          **spec["extra"]},
            "seed": SEED, "dose_multiplier": dose, "horizon_updates": horizon,
            "wall_s": wall, "quick_ffr_at_0.1pct": float(ffr),
            "edges": str(edges),
            "note": f"{ds} sandbox map; truth = own exact-k15 fuzzy graph; "
                    "no sealed claim.",
        }, indent=1))
        print(f"{ds}/{arm}: FFR {ffr:.4f} in {wall/60:.1f} min")
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(
        {"knn": knn, "fuzzy": fuzzy, "train": train}[sys.argv[2]](sys.argv[1]))
