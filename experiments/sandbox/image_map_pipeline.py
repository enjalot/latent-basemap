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
import os
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
#: no-rankneg variants — the manager runs THESE instead if the fraction-
#: scaled rankneg test at 6.25M fails (owner decision rule 2026-08-24).
_CHAMPION_NORANK = {"md": "000", "dose": 4,
                    "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                              "pos_ratio": 0.10, "batch_size": 16384}}

DATASETS = {
    "bl-siglip-1m": {"load": _bl_load, "subsets": _bl_subsets,
                     "arms": {**ARMS, "champion-bs16k": _CHAMPION_270K,
                              "champion-bs16k-norank": _CHAMPION_NORANK}},
    "sisap-clip-2m": {"load": _sisap_load, "subsets": None,
                      "arms": {**ARMS, "champion-bs16k": _CHAMPION_500K,
                               "champion-bs16k-norank": _CHAMPION_NORANK}},
    "jina-en-2m": {"load": _jina_en_load, "subsets": _jina_en_subsets},
    "jina-multi-2m": {"load": _jina_multi_load, "subsets": _jina_multi_subsets,
                      "arms": {**ARMS, "champion-bs16k": _CHAMPION_500K,
                               "champion-bs16k-norank": _CHAMPION_NORANK,
                               # #2 dose-vs-width decomposition (external review 2026-08-25);
                               # champion-identical except one lever, rank25=500K. Baseline 0.6426.
                               "champion-x8-h2048": {"md": "000", "dose": 8,       # exposure lever
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384}},
                               "champion-x4-h3072": {"md": "000", "dose": 4,       # capacity lever
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384, "hidden_dim": 3072}},
                               "champion-x8-h3072": {"md": "000", "dose": 8,       # escalation: both levers
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384, "hidden_dim": 3072}},
                               "champion-x8-h4096": {"md": "000", "dose": 8,       # escalation: h3072 capacity-limited
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384, "hidden_dim": 4096,
                                             "gpu_resident_vram_budget_gb": 22.0}},
                               # aesthetics track (owner 2026-08-26): md010 (looser kernel,
                               # a=1.577/b=0.8951) counterpart of the 0.6871 champion-x8-h2048
                               # exposure arm — does the spread-ier md010 trade a little FFR for a
                               # less island-y look at full dose8? Same champion stack otherwise.
                               "champion-md010-x8-h2048": {"md": "010", "dose": 8,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384}},
                               # Phase A3 matched-positive: constant TOTAL positive pairs vs
                               # champion-x8-h2048 (H=592187, ~970M pairs), fewer updates via
                               # higher pos_ratio (H = 592187 * 0.10/pos_ratio). Explicit horizon.
                               "champion-x8-h2048-pos15": {"md": "000", "dose": 0, "horizon": 394_791,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.15, "rankneg_window": 500_000,
                                             "batch_size": 16384}},
                               "champion-x8-h2048-pos20": {"md": "000", "dose": 0, "horizon": 296_094,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.20, "rankneg_window": 500_000,
                                             "batch_size": 16384}},
                               # Phase A4 tapered neck: champion-x8-h3072 (0.6999, residual_bottleneck)
                               # verbatim, only neck_fraction changes (default 0.75->2304; 0.50->1536, 0.625->1920).
                               "champion-x8-h3072-neck50": {"md": "000", "dose": 8,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384, "hidden_dim": 3072,
                                             "neck_fraction": 0.50}},
                               "champion-x8-h3072-neck625": {"md": "000", "dose": 8,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384, "hidden_dim": 3072,
                                             "neck_fraction": 0.625}}}},
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
            # rankneg RESTORED with the FRACTION-SCALED window (2026-08-24 rank25 verdict:
            # fixed 500K/8% dragged -0.07, but 1.5625M/25% = 0.4981 > norank 0.4848, +0.0133;
            # the pitfall was the FIXED window, not rankneg — so 25% of 6.25M = 1,562,500).
            "champion-bs16k": {"md": "000", "dose": 4,
                               "extra": {"fneg_weight": 1.0,
                                         "neg_tanh_gamma": 4.0,
                                         "pos_ratio": 0.10,
                                         "rankneg_window": 1_562_500,
                                         "batch_size": 16384,
                                         "gpu_resident_vram_budget_gb": 22.0}},
            # does the efficiency recipe survive scale on the easy space?
            "efficiency-x4-md010": {"md": "010", "dose": 4,
                                    "extra": {"fneg_weight": 1.0,
                                              "neg_tanh_gamma": 4.0,
                                              "pos_ratio": 0.10,
                                              "rankneg_window": 1_562_500,
                                              "hidden_dim": 1024,
                                              "n_layers": 2,
                                              "architecture": "mlp",
                                              "batch_size": 16384,
                                              "gpu_resident_vram_budget_gb": 22.0}},
            # Phase A2 6.25M horizon ladder (explicit "horizon" override, NOT dose): champion
            # h2048 knobs at EXPLICIT H (draws/edge H320k~3.67, H640k~7.35; champion dose4~10.84).
            "champion-h2048-H320k": {"md": "000", "dose": 0, "horizon": 320_000,
                "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0, "pos_ratio": 0.10,
                          "rankneg_window": 1_562_500, "batch_size": 16384,
                          "hidden_dim": 2048, "gpu_resident_vram_budget_gb": 22.0}},
            "champion-h2048-H640k": {"md": "000", "dose": 0, "horizon": 640_000,
                "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0, "pos_ratio": 0.10,
                          "rankneg_window": 1_562_500, "batch_size": 16384,
                          "hidden_dim": 2048, "gpu_resident_vram_budget_gb": 22.0}},
        }},
    # P4 mini-ladders (owner 2026-08-25): nested MiniLM sub-2M substrates
    # (every-kth-row, mixture-preserving) for (i) rankneg-fraction @1M and
    # (ii) dose-vs-N @500K+1M. Champion recipe = fneg10+tanh4+pos10+bs16k, md000.
    # rankneg held at the validated 25% fraction across the dose ladder so dose is
    # the only variable; P4(i) sweeps the fraction itself (12.5/25/50%) at fixed dose4.
    "minilm-mix-1m": {
        "load": lambda: np.load(
            "/data/latent-basemap/substrates/minilm-mix-1m/substrate.f32.npy",
            mmap_mode="r"),
        "subsets": None,
        "arms": {
            # P4(i) rankneg-fraction @1M: 12.5% / 25% / 50% windows (125K/250K/500K).
            "rankfrac-12p5": {"md": "000", "dose": 4,
                              "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                        "pos_ratio": 0.10, "rankneg_window": 125_000,
                                        "batch_size": 16384}},
            "rankfrac-25": {"md": "000", "dose": 4,      # = dose-x4 @1M (P4ii reuses this)
                            "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                      "pos_ratio": 0.10, "rankneg_window": 250_000,
                                      "batch_size": 16384}},
            "rankfrac-50": {"md": "000", "dose": 4,
                            "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                      "pos_ratio": 0.10, "rankneg_window": 500_000,
                                      "batch_size": 16384}},
            # P4(ii) dose-vs-N @1M (dose4 == rankfrac-25 above): add dose2 + dose8, rankneg 25%.
            "dose-x2-rf25": {"md": "000", "dose": 2,
                             "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                       "pos_ratio": 0.10, "rankneg_window": 250_000,
                                       "batch_size": 16384}},
            "dose-x8-rf25": {"md": "000", "dose": 8,
                             "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                       "pos_ratio": 0.10, "rankneg_window": 250_000,
                                       "batch_size": 16384}},
        }},
    "minilm-mix-500k": {
        "load": lambda: np.load(
            "/data/latent-basemap/substrates/minilm-mix-500k/substrate.f32.npy",
            mmap_mode="r"),
        "subsets": None,
        "arms": {
            # P4(ii) dose-vs-N @500K: dose 2/4/8, rankneg 25% of N = 125K.
            "dose-x2-rf25": {"md": "000", "dose": 2,
                             "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                       "pos_ratio": 0.10, "rankneg_window": 125_000,
                                       "batch_size": 16384}},
            "dose-x4-rf25": {"md": "000", "dose": 4,
                             "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                       "pos_ratio": 0.10, "rankneg_window": 125_000,
                                       "batch_size": 16384}},
            "dose-x8-rf25": {"md": "000", "dose": 8,
                             "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                       "pos_ratio": 0.10, "rankneg_window": 125_000,
                                       "batch_size": 16384}},
            # #3 int8-tax factorization @500K (external review 2026-08-25): champion recipe
            # (rankneg 125K = 25% of 500K), differ ONLY in the X path. (i) fp16 control +
            # (iii) host-int8 here; (ii) quant-dequant lives on minilm-mix-500k-qdq (shares this
            # dataset's graph via symlink). Factorization: (i)->(ii) quant damage, (ii)->(iii) loader.
            "int8fac-fp16": {"md": "000", "dose": 4,
                             "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                       "pos_ratio": 0.10, "rankneg_window": 125_000,
                                       "batch_size": 16384, "x_residency": "auto"}},
            "int8fac-hostint8": {"md": "000", "dose": 4,
                                 "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                           "pos_ratio": 0.10, "rankneg_window": 125_000,
                                           "batch_size": 16384, "x_residency": "host_int8"}},
        }},
    "minilm-mix-500k-qdq": {   # #3 (ii): int8 quant->dequant substrate (graph symlinked to 500k)
        "load": lambda: np.load(
            "/data/latent-basemap/substrates/minilm-mix-500k-qdq/substrate.f32.npy",
            mmap_mode="r"),
        "subsets": None,
        "arms": {
            "int8fac-qdq": {"md": "000", "dose": 4,
                            "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                      "pos_ratio": 0.10, "rankneg_window": 125_000,
                                      "batch_size": 16384, "x_residency": "auto"}},
        }},
    # MiniLM-2M capacity-probe dataset (owner 2026-08-25): completes the 4-space width
    # table. load = the sealed R0216 (a) substrate; teacher + edges symlinked into
    # sandbox/minilm-mix-2m/{upstream-06dev/coordinates.npy, edges-k15-fuzzy.npz} (row-aligned,
    # 0.6dev 0.4798). Probe-only via space_capacity_probe.py minilm-mix-2m (no train arms).
    "minilm-mix-2m": {
        "load": lambda: np.load(
            "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
            "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy",
            mmap_mode="r"),
        "subsets": None},
    # P3 curation validation loop (owner 2026-08-25): (b) curated + (c) random 2M
    # substrates (built by p3_build_and_scorecard.py from the same f32 shards as (a),
    # mix 40/25/25/10 — precision constant across the triple). Trained with the SAME
    # champion recipe as (a) (_CHAMPION_500K: rankneg 500K = 25% of 2M) so curation is
    # the only variable. (a) = existing 2m-knobs/umap-md000-x4bs16k-winner (not re-registered).
    "minilm-random-2m": {
        "load": lambda: np.load(
            "/data/latent-basemap/substrates/minilm-random-2m/substrate.f32.npy",
            mmap_mode="r"),
        "subsets": None,
        "arms": {"champion-bs16k": _CHAMPION_500K}},
    "minilm-curated-2m": {
        "load": lambda: np.load(
            "/data/latent-basemap/substrates/minilm-curated-2m/substrate.f32.npy",
            mmap_mode="r"),
        "subsets": None,
        "arms": {"champion-bs16k": _CHAMPION_500K}},
    # P2 jina-champion OOD register corpora (owner 2026-08-25): document-prompted
    # jina-v5-nano embeds of ~250K reddit + ~250K CA chunks (built by p2_jina_embed.py).
    # Probe-only — knn+fuzzy build their jina-space truth graphs; no train arms. p2_jina_probe.py
    # projects them through the jina-multi-2m champion for per-register OOD-FFR.
    "reddit-jina-250k": {
        "load": lambda: np.asarray(np.load(
            "/data/latent-basemap/substrates/reddit-jina-250k/substrate.f16.npy",
            mmap_mode="r"), dtype=np.float32),
        "subsets": None},
    "ca-jina-250k": {
        "load": lambda: np.asarray(np.load(
            "/data/latent-basemap/substrates/ca-jina-250k/substrate.f16.npy",
            mmap_mode="r"), dtype=np.float32),
        "subsets": None},
    # ---- #5 mixture sweep (owner 2026-08-26): 6 social-mix 1M substrates (rmix=reddit-only,
    # bmix=balanced reddit/CA/twitter/bluesky; share 10/20/30%). champion-bs16k = dose4, rankneg
    # 250K (25% of 1M). 0% baseline reuses minilm-mix-1m/rankfrac-25. Scored on the broad probe suite.
    **{ds: {"load": (lambda p=f"/data/latent-basemap/substrates/{ds}/substrate.f32.npy":
                     np.load(p, mmap_mode="r")),
            "subsets": None,
            "arms": {"champion-bs16k": {"md": "000", "dose": 4,
                                        "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                                  "pos_ratio": 0.10, "rankneg_window": 250_000,
                                                  "batch_size": 16384}}}}
       for ds in ("minilm-rmix10-1m", "minilm-rmix20-1m", "minilm-rmix30-1m",
                  "minilm-bmix10-1m", "minilm-bmix20-1m", "minilm-bmix30-1m")},
    # ---- #5 broad probe-register suite (heldout, disjoint from sweep training). Probe-only (no arms);
    # the orchestrator builds each register's knn+fuzzy truth graph, frozen before any sweep train seals.
    **{ds: {"load": (lambda p=f"/data/latent-basemap/substrates/{ds}/substrate.f32.npy":
                     np.load(p, mmap_mode="r")),
            "subsets": None}
       for ds in ("probe-reddit", "probe-ca", "probe-twitter", "probe-bluesky",
                  "probe-wiki", "probe-ccweb", "probe-ccscience", "probe-code")},
}

# capacity-curve calibration (owner 2026-08-24, DEFERRED-to-LAST 2026-08-25): IDENTICAL to
# champion-bs16k (rankneg_window 1.5625M, bs16k, dose x4) + hidden_dim=4096. ~14h (width mult
# ~3.1-3.3x). ENV-GATED so the branch-suite's no-ONLY_ARM jina-6m stage (trains all dict arms)
# does NOT pick it up — h4096 must run LAST, after the P1->P4->P3-GPU cheaper experiments.
# The dedicated final unit sets ENABLE_H4096=1 ONLY_ARM=champion-bs16k-h4096.
# Resident decision is hidden_dim-independent (need = X fp16 + edges only), so it inherits the
# proven rank25 resident+rankneg path; a HostStream fallback would raise (core.py:1416, fails
# closed, never silent-norank).
# HOLD guard (owner 2026-08-25, external-review directive): h4096@6.25M is HELD behind a
# preregistered gate (runs only if a 2M width arm beats the best 2M exposure arm by >=+0.010 FFR
# without hurting the worst-register probe). While /data/latent-basemap/sandbox/H4096_HOLD exists,
# the arm is NOT injected, so the P-plan's h4096 stage is a no-op (ONLY_ARM finds no match). Python
# is imported fresh per stage, so toggling the hold file needs no code edit. Delete the file to release.
if os.environ.get("ENABLE_H4096") and not os.path.exists(
        "/data/latent-basemap/sandbox/H4096_HOLD"):
    DATASETS["jina-multi-6m"]["arms"]["champion-bs16k-h4096"] = {
        "md": "000", "dose": 4,
        "extra": {"fneg_weight": 1.0,
                  "neg_tanh_gamma": 4.0,
                  "pos_ratio": 0.10,
                  "rankneg_window": 1_562_500,
                  "hidden_dim": 4096,
                  "batch_size": 16384,
                  "gpu_resident_vram_budget_gb": 22.0}}


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

    only = os.environ.get("ONLY_ARM")
    for arm, spec in DATASETS[ds].get("arms", ARMS).items():
        if only and arm != only:
            continue
        dose = spec.get("dose", 2)
        horizon = int(round(dose * 0.6782 * e / n_pos_per_batch))
        # Phase A explicit update-horizon override: an arm may set "horizon" to
        # train at exactly that many positive-LR updates instead of the dose-derived
        # budget. train() threads horizon -> total_steps_estimate (cosine LR/stop
        # horizon, core.py) and n_epochs below derives from it, so it's honored end-to-end.
        if spec.get("horizon") is not None:
            horizon = int(spec["horizon"])
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
        # Realized positive-pair bookkeeping (adopt everywhere): actual per-batch
        # positive count is int(batch*pos_ratio) (sampler num_pos; 1638 for
        # bs16k*0.10, NOT 1638.4), and the run takes positive_lr_optimizer_steps
        # updates. Stamp realized totals, not just the dose label.
        pos_per_batch = max(1, int(arm_batch * arm_pos))
        realized_updates = int((getattr(model, "_train_stats", {}) or {}).get(
            "positive_lr_optimizer_steps", horizon) or horizon)
        actual_positive_pairs = realized_updates * pos_per_batch
        (d / "summary.json").write_text(json.dumps({
            "arm": f"{ds}--{arm}", "rung": ds,
            "overrides": {"low_dim_kernel": "umap", **MD[spec["md"]],
                          **spec["extra"]},
            "seed": SEED, "dose_multiplier": dose, "horizon_updates": horizon,
            "positive_lr_updates": realized_updates,
            "pos_per_batch": pos_per_batch,
            "actual_positive_pairs": actual_positive_pairs,
            "draws_per_edge": actual_positive_pairs / e,
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
