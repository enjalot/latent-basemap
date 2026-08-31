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


def _bmix30_2m_subsets():
    return np.load("/data/latent-basemap/substrates/minilm-bmix30-2m/subsets.npy",
                   allow_pickle=True)


def _bmix10cp_2m_subsets():
    return np.load("/data/latent-basemap/substrates/minilm-bmix10cp-2m/subsets.npy",
                   allow_pickle=True)


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
                               # #12 gate-3 parity twin (external review 2026-08-27): champion-bs16k
                               # with the SOLE delta x_residency=host_int8, so resident (0.6426 /
                               # v2-rescored) vs hostint8 measures the TRUE jina-D768 int8 tax at 2M —
                               # the jina-shape number that governs 30M gate 3, replacing the
                               # MiniLM-shape points. Queue AFTER Phase A (needs an idle GPU, ~1.5h).
                               "champion-bs16k-hostint8": {**_CHAMPION_500K,
                                   "extra": {**_CHAMPION_500K["extra"],
                                             "x_residency": "host_int8"}},
                               # device-int8 quality-parity (2026-08-28): sole delta
                               # x_residency=device_int8 (zero-transport). Shares the quant scheme +
                               # sampler draws with hostint8, so FFR MUST reproduce champion-bs16k-
                               # hostint8's 0.6964 (v2) within |Δ|<0.002 — a larger deviation is a
                               # gather-misorder/dequant bug. Confirms device-int8 (resident-class
                               # throughput per the 4-way A/B) is also quality-correct before 30M.
                               "champion-bs16k-deviceint8": {**_CHAMPION_500K,
                                   "extra": {**_CHAMPION_500K["extra"],
                                             "x_residency": "device_int8"}},
                               # 4th-review P0.1 INT8-FLOOR twins (delegate 2026-08-29): resident came
                               # back BITWISE-DETERMINISTIC at both D384+D768 (floor=0), so per the decision
                               # tree we now band the device-int8 parity with the int8-path floor. Same H=200K
                               # short-horizon as the resident-D768 twins (apples-to-apples), seed 42. host×2
                               # gives the host_int8 self-floor; device×2 gives the device_int8 self-floor;
                               # device-vs-host FFR delta (RE-measured seeded, replacing the unseeded 0.0025)
                               # is judged against those floors. If host_a==host_b and dev_a==dev_b by trained
                               # hash, both int8 paths are deterministic and the device-vs-host delta is a REAL
                               # (bug-or-genuine) divergence; else the nonzero self-floor bands it.
                               "floor-hostint8-h200k-a": {"md": "000", "dose": 0, "horizon": 200_000,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0, "pos_ratio": 0.10,
                                             "rankneg_window": 500_000, "batch_size": 16384,
                                             "x_residency": "host_int8"}},
                               "floor-hostint8-h200k-b": {"md": "000", "dose": 0, "horizon": 200_000,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0, "pos_ratio": 0.10,
                                             "rankneg_window": 500_000, "batch_size": 16384,
                                             "x_residency": "host_int8"}},
                               "floor-deviceint8-h200k-a": {"md": "000", "dose": 0, "horizon": 200_000,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0, "pos_ratio": 0.10,
                                             "rankneg_window": 500_000, "batch_size": 16384,
                                             "x_residency": "device_int8"}},
                               "floor-deviceint8-h200k-b": {"md": "000", "dose": 0, "horizon": 200_000,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0, "pos_ratio": 0.10,
                                             "rankneg_window": 500_000, "batch_size": 16384,
                                             "x_residency": "device_int8"}},
                               # determinism CONTROL (2026-08-28): identical to champion-bs16k-hostint8
                               # (seed 42, host_int8), distinct out-dir. Its FFR vs champion-bs16k-hostint8
                               # measures the run-to-run GPU non-determinism FLOOR at fixed config — the
                               # yardstick for judging device-int8's 0.0025 vs hostint8 (GPU-noise if the
                               # self-vs-self floor is comparable, real divergence if the floor is ~0).
                               "champion-bs16k-hostint8-rep2": {**_CHAMPION_500K,
                                   "extra": {**_CHAMPION_500K["extra"],
                                             "x_residency": "host_int8"}},
                               # gate-3 SEED-43 replicate PAIR (delegate 2026-08-28): the jina-D768
                               # int8 SIGN-FLIP (int8 +0.0134 vs resident @ seed 42) doesn't enter the
                               # 30M spec on one seed. Resident + hostint8 twins at seed 43, otherwise
                               # champion-bs16k-identical -> confirm int8 >= resident (sealed) or flip
                               # back (seed-noise). Distinct out dirs (seed in the name) avoid write-once skip.
                               "champion-bs16k-seed43": {**_CHAMPION_500K, "seed": 43},
                               "champion-bs16k-hostint8-seed43": {**_CHAMPION_500K, "seed": 43,
                                   "extra": {**_CHAMPION_500K["extra"],
                                             "x_residency": "host_int8"}},
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
                                             "neck_fraction": 0.625}},
                               # P2 seeded ARCH PAIR (delegate 2026-08-30): the existing arch arms were
                               # UNSEEDED (pre-fix); these seed-42 twins seal the width-ladder (+0.0128)
                               # + neck (−0.0008) deltas with exact numbers. Same recipes, seed 42.
                               "p2-x8-h2048-s42": {"md": "000", "dose": 8,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384}},
                               "p2-x8-h3072n625-s42": {"md": "000", "dose": 8,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384, "hidden_dim": 3072,
                                             "neck_fraction": 0.625}},
                               # P2 arch SEED-43 replicate (delegate 2026-08-30): the maximin win
                               # (+0.0083) is at seed-variance scale + cross-arch, so a 2nd seed confirms
                               # the sign before the 17h flagship. Verdict: sign agreement -> seal h3072.
                               "p2-x8-h2048-s43": {"md": "000", "dose": 8, "seed": 43,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384}},
                               "p2-x8-h3072n625-s43": {"md": "000", "dose": 8, "seed": 43,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384, "hidden_dim": 3072,
                                             "neck_fraction": 0.625}},
                               # 4th-review P0.1 FLOOR (delegate 2026-08-29): resident-D768 determinism
                               # twins via the explicit-horizon mechanism (H=200K, ~30-60min each vs the
                               # full champion wall). Same readout as the D384 twins — trained_state_sha256
                               # equality => the resident floor is ZERO at D768 too; init hash must match.
                               # Seed defaults to 42 (no override). resident (no x_residency).
                               "floor-resident-h200k-a": {"md": "000", "dose": 0, "horizon": 200_000,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384}},
                               "floor-resident-h200k-b": {"md": "000", "dose": 0, "horizon": 200_000,
                                   "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0,
                                             "pos_ratio": 0.10, "rankneg_window": 500_000,
                                             "batch_size": 16384}}}},
    # ---- bmix30-2m FINALIST CONFIRMATION (owner 2026-08-27): social-mixture sweep winner
    # (bmix30 = 30% balanced social) replicated at 2M with MATCHED rows vs minilm-mixed-2m — 1.4M
    # base rows IDENTICAL to a subset of that baseline + 600K balanced social (150K each
    # reddit/CA/twitter/bluesky, holdout-disjoint offset>=300000), so the social 30% is the SOLE
    # delta. champion-bs16k = _CHAMPION_500K (dose4, rankneg 500K = 25% of 2M), same recipe as the
    # 2M baseline. Scored on the full probe-register suite (incl. probe-code-heldout).
    "minilm-bmix30-2m": {
        "load": lambda: np.load(
            "/data/latent-basemap/substrates/minilm-bmix30-2m/substrate.f32.npy",
            mmap_mode="r"),
        "subsets": _bmix30_2m_subsets,
        "arms": {"champion-bs16k": _CHAMPION_500K}},
    # ---- bmix10cp-2m CODE-PRESERVING social probe (owner 2026-08-28): 10% BALANCED social at 2M
    # with MATCHED rows vs minilm-mixed-2m — 1.8M base rows IDENTICAL to a subset of that baseline
    # (fineweb 711,111 / redpajama 444,444 / pile 444,445 / STARCODER 200,000 UNTOUCHED) + 200K
    # balanced social (50K each reddit/CA/twitter/bluesky, holdout-disjoint offset>=300000). The
    # 200K displacement is drawn ONLY from fineweb/redpajama/pile PROPORTIONALLY (800:500:500), so
    # the social 10% is paid entirely from the web/pile budget and the code budget is preserved.
    # champion-bs16k = _CHAMPION_500K (dose4, rankneg 500K = 25% of 2M), same recipe as the 2M
    # baseline. Scored on the full probe-register suite (incl. probe-code-heldout).
    "minilm-bmix10cp-2m": {
        "load": lambda: np.load(
            "/data/latent-basemap/substrates/minilm-bmix10cp-2m/substrate.f32.npy",
            mmap_mode="r"),
        "subsets": _bmix10cp_2m_subsets,
        "arms": {"champion-bs16k": _CHAMPION_500K,
                 # P1.5 bmix10cp@43 (the seed-43 treatment cell; @42 = floor-resident-a reuse).
                 "p15-bmix10cp-s43": {**_CHAMPION_500K, "seed": 43},
                 # 4th-review P0.1 FLOOR (delegate 2026-08-29): resident-D384 determinism twins.
                 # Two SEEDED (42) same-config champion runs into distinct dirs. PRIMARY readout is
                 # trained_state_sha256 EQUALITY — identical => the resident floor is literally ZERO
                 # (every same-seed cross-config comparison becomes exact, no statistical banding);
                 # |Δmaximin| is the fallback floor only if the trained hashes differ. init_state_sha256
                 # MUST match across the twins either way (the seeding proof). Determinism is a PATH
                 # property, so this cheap D384 champion arm answers the resident question for all D384.
                 "floor-resident-a": _CHAMPION_500K,
                 "floor-resident-b": _CHAMPION_500K}},
    # ---- substrate-draw universality (owner 2026-08-30): 3 disjoint composition-matched 2M slices
    # (build_draw_universality.py; proofs in draw-univ-proofs.json). Same champion recipe + seed 42 ->
    # the three init_state_sha256 MUST be equal (validity gate); any output difference is the DATA DRAW.
    **{f"draw-univ-{s}": {
        "load": (lambda ss=s: np.load(
            f"/data/latent-basemap/substrates/draw-univ-{ss}/substrate.f32.npy", mmap_mode="r")),
        "subsets": (lambda ss=s: np.load(
            f"/data/latent-basemap/substrates/draw-univ-{ss}/provenance.npy", allow_pickle=False)["corpus"]),
        "arms": {"champion-bs16k": _CHAMPION_500K}}
       for s in ("A", "B", "C")},
    # ---- image-space universality (owner 2026-08-30): sisap-CLIP D768, 4 disjoint 2M slices
    # (build_draw_univ_image.py; img-univ-proofs.json). A/B/C = heads (champion@42, rankneg 500K),
    # D = shared neutral eval (truth only, no train). Validity gate: the 3 head inits equal (D768).
    **{f"img-univ-{s}": {
        "load": (lambda ss=s: np.load(
            f"/data/latent-basemap/substrates/img-univ-{ss}/substrate.f32.npy", mmap_mode="r")),
        "subsets": None,
        **({"arms": {"champion-bs16k": _CHAMPION_500K}} if s in ("A", "B", "C") else {})}
       for s in ("A", "B", "C", "D")},
    "reddit-2m": {"load": _reddit_load, "subsets": None},
    "communityarchive-2m": {"load": _ca_load, "subsets": None},
    "minilm-redditmix-2m": {"load": _redditmix_load,
                            "subsets": _redditmix_subsets,
                            "arms": _REDDITMIX_ARMS},
    "sisap-clip-2m-dedup": {"load": _sisap_dedup_load, "subsets": None,
                            "arms": _SISAP_DEDUP_ARMS},
    # ---- jina LANGUAGE-PRESERVING social-mixture sweep (JINA_SWEEP_PROPOSAL.md 2026-08-28):
    # social displaces ONLY the EN 1M (proportional fw/rp/pile); all 20 language blocks held
    # BIT-IDENTICAL to the 0% baseline (jina-multi-2m/champion-bs16k, reused as the 0% arm). f16 (2M,768).
    # champion-bs16k = _CHAMPION_500K (dose4, rankneg 500K = 25% of 2M), same recipe as jina-multi-2m.
    **{arm: {
        "load": (lambda a=arm: np.asarray(np.load(
            f"/data/latent-basemap/substrates/{a}/substrate.f16.npy",
            mmap_mode="r"), dtype=np.float32)),
        "subsets": (lambda a=arm: np.load(
            f"/data/latent-basemap/substrates/{a}/subsets.npy", allow_pickle=True)),
        "arms": {"champion-bs16k": _CHAMPION_500K}}
       for arm in ("jina-bmix10-2m", "jina-bmix20-2m", "jina-bmix30-2m", "jina-rmix20-2m")},
    # ---- jina maximin probe registers (JINA_SWEEP_PROPOSAL.md P-A/P-B/P-C, 2026-08-28):
    # document-prompted jina-v5-nano holdout probes; probe-only (no arms). The orchestrator builds
    # each register's knn+fuzzy TRUTH graph; no map trains on them. f16 (N,768) -> float32.
    **{ds: {
        "load": (lambda a=ds: np.asarray(np.load(
            f"/data/latent-basemap/substrates/{a}/substrate.f16.npy",
            mmap_mode="r"), dtype=np.float32)),
        "subsets": None}
       for ds in (
           "twitter-jina-250k", "bluesky-jina-250k",
           "probe-fineweb-jina", "probe-rpj-jina", "probe-pile-jina",
           *(f"probe-lang-{l}-jina" for l in _JINA_LANGS))},
    "jina-neutral-pooled": {
        "load": lambda: np.asarray(np.load(
            "/data/latent-basemap/substrates/jina-neutral-pooled/substrate.f16.npy",
            mmap_mode="r"), dtype=np.float32),
        "subsets": None},
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
            # P1.6 SEEDED direct 6.25M reference (delegate 2026-08-29): the existing champion-bs16k
            # here is UNSEEDED (0.6686 artifact); retrain seeded so the head-size comparison is exact.
            # Same champion recipe (rankneg 1,562,500 = 25% of 6.25M). Becomes the canonical jina-6m map.
            "p16-ref-s42": {"md": "000", "dose": 4,
                "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0, "pos_ratio": 0.10,
                          "rankneg_window": 1_562_500, "batch_size": 16384,
                          "gpu_resident_vram_budget_gb": 22.0}},
        }},
    # P1.6 head-size experiment (delegate 2026-08-29): the nested composition-matched 4M head
    # (build_jina_4m_head.py -> seed-42 64%-per-span draw of the 6.25M; member_indices.npy records
    # exact 6.25M membership). champion recipe fraction-scaled to 25% of 4M = rankneg 1,000,000, seed 42.
    "jina-4m-head": {
        "load": lambda: np.array(np.load(
            "/data/latent-basemap/substrates/jina-4m-head/substrate.f16.npy",
            mmap_mode="r"), dtype=np.float32),
        "subsets": lambda: np.load(
            "/data/latent-basemap/substrates/jina-4m-head/subsets.npy", allow_pickle=True),
        "arms": {"champion-bs16k": {"md": "000", "dose": 4,
                     "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0, "pos_ratio": 0.10,
                               "rankneg_window": 1_000_000, "batch_size": 16384,
                               "gpu_resident_vram_budget_gb": 22.0}},
                 # P1.6 near-boundary replicate: 4M is within 2x of the retention gate (94.8% vs 97%),
                 # so the preregistered rule calls a seed-43 replicate of THIS decisive cell.
                 "champion-s43": {"md": "000", "dose": 4, "seed": 43,
                     "extra": {"fneg_weight": 1.0, "neg_tanh_gamma": 4.0, "pos_ratio": 0.10,
                               "rankneg_window": 1_000_000, "batch_size": 16384,
                               "gpu_resident_vram_budget_gb": 22.0}}}},
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
        "subsets": None,
        # P1.5 finalist BASELINE (0% social) two-seed cells, trained through THIS runner so the
        # same-seed comparison vs the bmix10cp treatment is EXACT (resident floor=0). full champion.
        "arms": {"p15-baseline-s42": _CHAMPION_500K,
                 "p15-baseline-s43": {**_CHAMPION_500K, "seed": 43}}},
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
                  "minilm-bmix10-1m", "minilm-bmix20-1m", "minilm-bmix30-1m",
                  # social-CEILING arms (owner 2026-08-27): balanced family only, extend past 30%.
                  "minilm-bmix40-1m", "minilm-bmix50-1m")},
    # ---- #5 broad probe-register suite (heldout, disjoint from sweep training). Probe-only (no arms);
    # the orchestrator builds each register's knn+fuzzy truth graph, frozen before any sweep train seals.
    **{ds: {"load": (lambda p=f"/data/latent-basemap/substrates/{ds}/substrate.f32.npy":
                     np.load(p, mmap_mode="r")),
            "subsets": None}
       for ds in ("probe-reddit", "probe-ca", "probe-twitter", "probe-bluesky",
                  "probe-wiki", "probe-ccweb", "probe-ccscience", "probe-code",
                  # #8 fix: HELD-OUT code register (250K from the complement of the 2M
                  # baseline's starcoder rows; 0 overlap proven, incl. bmix30-2m ⊆ baseline).
                  # Replaces the contaminated probe-code at 2M-confirmation decision time.
                  "probe-code-heldout")},
    # Phase A1 cross-scale audit: one frozen common sample (250K of the 2M pool). Probe-only;
    # the orchestrator builds its knn+fuzzy truth, then a1_audit.py scores every MiniLM head on it.
    "a1-common": {
        "load": lambda: np.load(
            "/data/latent-basemap/substrates/a1-common/substrate.f32.npy",
            mmap_mode="r"),
        "subsets": None},
    # Phase A1 cross-scale audit (Bug #5 fix): NEUTRAL common probe (250K sampled from the
    # source MiniLM shards MINUS every head's training rows; held out for ALL heads at 0.000000%
    # overlap). Probe-only; the orchestrator builds its knn+fuzzy truth, then a1_audit.py scores
    # every MiniLM head on it. Supersedes "a1-common" (which was the 2M train set, 100% overlap).
    "a1-common-neutral": {
        "load": lambda: np.load(
            "/data/latent-basemap/substrates/a1-common-neutral/substrate.f32.npy",
            mmap_mode="r"),
        "subsets": None},
}

# ---- P1.5 JINA finalist two-seed cells (4th-review, delegate 2026-08-29). Full champion, same
# runner (image_map_pipeline) so same-seed comparisons are EXACT (resident-D768 floor=0). The
# existing champion-bs16k on each substrate was UNSEEDED (pre-fix) + the h200k floor twins are
# short-horizon, so none is reusable on the jina side — all 4 cells are fresh SEEDED full-champion
# trains: baseline (0% social, jina-multi-2m) @{42,43} + bmix10 (jina-bmix10-2m) @{42,43}. Added
# post-construction because jina-bmix10-2m's arms come from the substrate comprehension above. ----
DATASETS["jina-multi-2m"]["arms"].update({
    "p15-baseline-s42": _CHAMPION_500K,
    "p15-baseline-s43": {**_CHAMPION_500K, "seed": 43}})
DATASETS["jina-bmix10-2m"]["arms"] = {
    **DATASETS["jina-bmix10-2m"]["arms"],
    "p15-bmix10-s42": _CHAMPION_500K,
    "p15-bmix10-s43": {**_CHAMPION_500K, "seed": 43}}

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
    assert not getattr(x, "sealed_int8", False), (
        "_norm() called on a sealed pre-normalized int8 substrate — "
        "double-normalize guard (build_int8_substrate already _norm'd it)")
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
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_sealed_int8_substrate)

    out = SANDBOX / ds
    edges = out / "edges-k15-fuzzy.npz"
    e = int(len(np.load(edges, mmap_mode="r")["sources"]))
    n_pos_per_batch = max(1, int(round(
        BASE_KWARGS["batch_size"] * BASE_KWARGS["pos_ratio"])))
    sealed = DATASETS[ds].get("sealed_int8_path")
    if sealed:
        # Pre-sealed int8 substrate: already _norm'd + quantized at build time.
        # Load a dequantizing memmap view and DO NOT call _norm (see §4 guard).
        x = load_sealed_int8_substrate(sealed, dim=DATASETS[ds].get("int8_dim"))
        assert getattr(x, "_prenormalized", False) is True, (
            "sealed_int8_path substrate is not pre-normalized; refusing to train")
        # NB: _norm(...) is intentionally NOT applied on this branch.
    else:
        x = _norm(DATASETS[ds]["load"]())
    subsets = DATASETS[ds]["subsets"]() if DATASETS[ds]["subsets"] else None

    only = os.environ.get("ONLY_ARM")
    _arms = DATASETS[ds].get("arms", ARMS)
    # #1 (external review 2026-08-27): a typo'd/missing ONLY_ARM must HARD-ERROR, not silently
    # "succeed" by training nothing (that is how the held-h4096 stage logged DONE).
    if only and only not in _arms:
        raise SystemExit(
            f"ONLY_ARM={only!r} not in {ds} arms {list(_arms)} — refusing silent no-op (typo?)")
    for arm, spec in _arms.items():
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
        arm_seed = int(spec.get("seed", SEED))   # per-arm seed override (e.g. seed-43 replicate)
        import torch, hashlib as _hashlib   # train() has no module-level torch (the :611 import is elsewhere)
        # 4th-review P0.1 (2026-08-28): SEED the torch init before construction/fit. This runner
        # previously NEVER seeded (unlike knobs_2m:643), so every image-runner arm trained from a
        # random init — conflating init variance into every same-seed comparison. Seed here so the
        # init (in ParametricUMAP/_init_model, called INSIDE fit) is reproducible.
        torch.manual_seed(arm_seed)
        torch.cuda.manual_seed_all(arm_seed)
        model = ParametricUMAP(**kwargs)
        t0 = time.time()
        # env-gated resumable checkpointing (#11): CHECKPOINT_EVERY_EPOCHS>0 writes per-arm epoch
        # checkpoints to <arm>/ckpt and auto-resumes from the newest one (bitwise-invisible). Opt-in
        # so normal short runs pay nothing; enable for long/preemptible runs (arch, flagship, scale).
        _ck_every = int(os.environ.get("CHECKPOINT_EVERY_EPOCHS", "0") or 0)
        _fit_kw = {}
        if _ck_every > 0:
            _ckd = d / "ckpt"; _ckd.mkdir(parents=True, exist_ok=True)
            _existing = sorted(_ckd.glob("ckpt-epoch*.pt"),
                               key=lambda p: int(p.stem.split("epoch")[1]))
            _fit_kw = {"checkpoint_every_epochs": _ck_every, "checkpoint_dir": str(_ckd)}
            if _existing:
                _fit_kw["resume_from"] = str(_existing[-1])
                print(f"{ds}/{arm}: RESUMING from {_existing[-1].name}", flush=True)
        model.fit(x, precomputed_edges_path=str(edges), random_state=arm_seed, **_fit_kw)
        # Seeded-ness fingerprint: model.init_state_sha256 (core.py hook, landed bb6e1e7) is the true
        # PRE-init hash — reproducible at fixed seed, isolates init variance. This post-fit hash of the
        # TRAINED weights is the complementary end-to-end fingerprint: same seed+config reproduces it
        # IFF init AND training are both deterministic (the resident-floor question the twins measure).
        state_sha = None
        try:
            _h = _hashlib.sha256()
            for p in model.model.parameters():
                _h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
            state_sha = _h.hexdigest()[:16]
        except Exception as _e:
            state_sha = f"err:{type(_e).__name__}"
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
        # #13 (external review 2026-08-27): record REALIZED NEGATIVE totals too.
        # edge_list_dataset fills each batch to batch_size with num_pos positives
        # + num_neg = batch_size - num_pos negatives (edge_list_dataset.py:304-305).
        # A "matched-positive" arm (higher pos_ratio, shorter horizon holding
        # actual_positive_pairs constant) therefore trains on FEWER total
        # negatives and a different pos:neg batch balance -> it is a CHANGED
        # OBJECTIVE (less repulsion), NOT the same objective with fewer steps.
        neg_per_batch = max(0, arm_batch - pos_per_batch)
        actual_negative_pairs = realized_updates * neg_per_batch
        (d / "summary.json").write_text(json.dumps({
            "arm": f"{ds}--{arm}", "rung": ds,
            "overrides": {"low_dim_kernel": "umap", **MD[spec["md"]],
                          **spec["extra"]},
            "seed": arm_seed, "torch_seeded": True,
            "init_state_sha256": getattr(model, "init_state_sha256", None),  # P0.1 core hook (pre-init, isolates init)
            "trained_state_sha256": state_sha,  # post-fit fingerprint (init+training)
            "dose_multiplier": dose, "horizon_updates": horizon,
            "positive_lr_updates": realized_updates,
            "pos_per_batch": pos_per_batch,
            "neg_per_batch": neg_per_batch,
            "actual_positive_pairs": actual_positive_pairs,
            "actual_negative_pairs": actual_negative_pairs,
            "draws_per_edge": actual_positive_pairs / e,
            "negatives_per_edge": actual_negative_pairs / e,
            "pos_to_neg_per_batch": (f"1:{neg_per_batch / pos_per_batch:.2f}"
                                     if pos_per_batch else None),
            "objective_note": (
                "matched-positive arms (higher pos_ratio at a shorter horizon) hold "
                "actual_positive_pairs ~constant but train on FEWER total negatives "
                "(neg_per_batch = batch - int(batch*pos_ratio)); the pos:neg balance "
                "and total repulsion change -> compare as a CHANGED OBJECTIVE, not "
                "'same objective, fewer steps'. Sweep arms use unmatched (screening) "
                "rows; finalists must replicate at 2M with matched rows."),
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
