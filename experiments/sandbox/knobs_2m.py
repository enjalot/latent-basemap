"""One-variable kernel/dose arms on sealed substrates (2M and 6.25M rungs).

Owner-directed sandbox (PLAN.md, PLAN2-umap-kernel.md, PLAN3-kernels.md).
Not rounds: no capability, no gate, no sealed claim. Each arm re-runs the
registered treatment of its rung (R0217 at 2M, R0257 at 6.25M) with the
arm's named knobs changed and renders the result for visual comparison.

Arms may set:
  dose          multiplier on the rung's registered draws-per-edge (default 1)
  kernel_alpha  gcauchy tail exponent (alpha=1 is exactly the umap kernel)
  any ParametricUMAP constructor field (a, b, low_dim_kernel, pos_ratio, ...)

Safety: refuses a busy GPU (issued rounds keep priority) and existing output
dirs (write-once; delete an arm dir to re-run).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

BATCH = 8192
POS_RATIO = 0.05
SEED = 42
SITE_DIR = Path.home() / ".agent/basemap-maps/sandbox/2m-knobs"

RUNGS: dict[str, dict] = {
    "2m": {
        "rows": 2_000_000,
        "substrate": Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts"
                          "/minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy"),
        "edges": Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts"
                      "/minilm-mixed-2m-substrate-and-exact-k15-graph-v1/edges-k15-fuzzy.npz"),
        "directed_edges": 48_344_648,
        "sealed_checkpoint": Path("/data/latent-basemap/runs/round-0217/queue-correction-1/artifacts"
                                  "/minilm-mixed-2m-map-seed42-low-dose-v1/model.pt"),
        "base_horizon": 80_163,          # R0217: 0.6782 draws/edge
        "out_root": Path("/data/latent-basemap/sandbox/2m-knobs"),
    },
    "6250k": {
        "rows": 6_250_000,
        "substrate": Path("/data/latent-basemap/runs/round-0233/queue/artifacts"
                          "/minilm-mixed-6250k-substrate-and-reserves-v1/substrate.f32.npy"),
        "edges": None,   # resolved at runtime: fuzzy npz inside the R0233 graph artifact
        "edges_glob": ("/data/latent-basemap/runs/round-0233/queue-correction-1/artifacts"
                       "/minilm-mixed-6250k-cluster-spill-k15-fuzzy-graph-v1/*.npz"),
        "directed_edges": 153_872_500,
        "sealed_checkpoint": Path("/data/latent-basemap/runs/round-0257/queue/artifacts"
                                  "/minilm-mixed-6250k-ladder-map-seed42-v1/train_seed42-model.pt"),
        "base_horizon": 255_142,         # R0257: 0.6782 draws/edge
        "out_root": Path("/data/latent-basemap/sandbox/6250k-knobs"),
    },
    "12500k": {
        # P1 drift cell (promotion plan): third N point at x2 on the sealed
        # R0235 nested substrate + cluster-spill fuzzy graph. No 12.5M ladder
        # map was sealed (R0235 built substrate+graph only), so the recipe
        # reference is R0257's checkpoint — the registered treatment is
        # rung-invariant and receipt_diff only compares recipe fields
        # (a, b, kernel, pos_ratio, ...), never rung-specific quantities.
        "rows": 12_500_000,
        "substrate": Path("/data/latent-basemap/runs/round-0235/queue/artifacts"
                          "/minilm-mixed-12500k-nested-substrate-and-reserves-v1/substrate.f32.npy"),
        "edges": Path("/data/latent-basemap/runs/round-0235/queue/artifacts"
                      "/minilm-mixed-12500k-cluster-spill-k15-fuzzy-graph-v1/edges-k15-fuzzy.npz"),
        "directed_edges": 310_512_394,
        "sealed_checkpoint": Path("/data/latent-basemap/runs/round-0257/queue/artifacts"
                                  "/minilm-mixed-6250k-ladder-map-seed42-v1/train_seed42-model.pt"),
        "base_horizon": 514_889,         # 0.6782 draws/edge at 310.5M directed edges
        "out_root": Path("/data/latent-basemap/sandbox/12500k-knobs"),
    },
    "25000k": {
        # P1 analysis-v2 4th x2 point (owner-approved 2026-08-13): 25M on the
        # sealed R0236 nested substrate + cluster-spill fuzzy graph. Recipe
        # reference R0257 (rung-invariant; receipt_diff checks recipe fields only).
        "rows": 25_000_000,
        "substrate": Path("/data/latent-basemap/runs/round-0236/queue-correction-2/artifacts"
                          "/minilm-mixed-25000k-nested-substrate-and-reserves-v1/substrate.f32.npy"),
        "edges": Path("/data/latent-basemap/runs/round-0236/queue-correction-2/artifacts"
                      "/minilm-mixed-25000k-cluster-spill-k15-fuzzy-graph-v1/edges-k15-fuzzy.npz"),
        "directed_edges": 625_148_276,
        "sealed_checkpoint": Path("/data/latent-basemap/runs/round-0257/queue/artifacts"
                                  "/minilm-mixed-6250k-ladder-map-seed42-v1/train_seed42-model.pt"),
        "base_horizon": 1_036_615,       # 0.6782 draws/edge at 625.1M directed edges
        "out_root": Path("/data/latent-basemap/sandbox/25000k-knobs"),
    },
}

# The registered treatment shared by R0217 and R0257 (seed aside) — EXCEPT
# sampling: this trainer's device edge-list path samples positives UNIFORMLY,
# while R0217's sealed receipt says fuzzy-weight-proportional. That silent
# difference was caught by R0265's seed-42 cross-check (2026-08-14). All
# sandbox arms shared the uniform sampler, so arm-vs-arm contrasts are valid,
# and UNIFORM is now the promoted treatment's registered sampling field (see
# guides/plan-umap-fneg-promotion.md). The checkpoint dict does not persist a
# sampling field, so receipt_diff cannot check it — the round program's
# treatment closure pins weighted_edge_sampling=False with a refusal plant.
BASE_KWARGS = dict(
    n_components=2, hidden_dim=2048, n_layers=3, n_neighbors=15,
    a=1.0, b=1.0, low_dim_kernel="legacy_lp", kernel_alpha=1.0,
    correlation_weight=0.0, learning_rate=1e-3, n_epochs=1, batch_size=BATCH,
    pos_ratio=POS_RATIO, architecture="residual_bottleneck",
    positive_target_mode="binary", lr_schedule="cosine", use_amp=True,
    use_batchnorm=False, use_dropout=False,
    require_full_budget=False,   # exploratory: the horizon break ends the run
    device="cuda",
)

# umap-kernel (a, b) fits at spread=1 (scipy curve_fit, verified 2026-08-12).
MD = {
    "000": {"a": 1.9328, "b": 0.7905},   # min_dist 0.0
    "0025": {"a": 1.8404, "b": 0.8160},  # min_dist 0.025 (phase 3c; fit verified vs published md010 before minting)
    "005": {"a": 1.7502, "b": 0.8421},   # min_dist 0.05
    "010": {"a": 1.5769, "b": 0.8951},   # min_dist 0.1
    "020": {"a": 1.2621, "b": 1.0030},   # min_dist 0.2
    "035": {"a": 0.8741, "b": 1.1673},   # min_dist 0.35
    "050": {"a": 0.5830, "b": 1.3342},   # min_dist 0.5
}


def _umap(md: str, **extra) -> dict:
    return {"low_dim_kernel": "umap", **MD[md], **extra}


def _gc(md: str, alpha: float, **extra) -> dict:
    return {"low_dim_kernel": "gcauchy", "kernel_alpha": alpha, **MD[md], **extra}


ARMS: dict[str, dict] = {
    # phase 1/2 (PLAN.md, PLAN2)
    "replay-baseline": {},
    "dose-x2": {"dose": 2},
    "kernel-a4": {"a": 4.0},
    "kernel-b2": {"b": 2.0},
    "umap-kernel": _umap("010"),
    "umap-dose-x2": _umap("010", dose=2),
    "umap-dose-x4": _umap("010", dose=4),
    "umap-mind0": _umap("000"),
    "umap-mind05": _umap("050"),
    # phase 3 (PLAN3-kernels.md) — min_dist sweep at the established dose x2
    "umap-md000-x2": _umap("000", dose=2),
    "umap-md005-x2": _umap("005", dose=2),
    "umap-md020-x2": _umap("020", dose=2),
    "umap-md035-x2": _umap("035", dose=2),
    "umap-md050-x2": _umap("050", dose=2),
    # phase 3 — tail exponent (gcauchy; alpha=1 would equal the umap arms)
    "gc-a05-md000-x2": _gc("000", 0.5, dose=2),
    "gc-a05-md010-x2": _gc("010", 0.5, dose=2),
    "gc-a2-md000-x2": _gc("000", 2.0, dose=2),
    "gc-a2-md010-x2": _gc("010", 2.0, dose=2),
    # phase 3 — attraction/repulsion spectrum probes (dose covaries with
    # pos_ratio at fixed horizon; exploratory, documented in PLAN3)
    "umap-pos02-x2": _umap("010", dose=2, pos_ratio=0.02),
    "umap-pos15-x2": _umap("010", dose=2, pos_ratio=0.15),
    # phase 3c reserve (code-71 decision 2026-08-12): finer min_dist between the
    # two best (md000/md005), and md000 at dose x4 to test if convergence keeps
    # buying fidelity or creeps toward collapse.
    "umap-md0025-x2": _umap("0025", dose=2),
    "umap-md000-x4": _umap("000", dose=4),
    # phase 4A (PLAN4-fog): mid-near PaCMAP attractive term on the winner base.
    # Existing dormant core machinery (midnear_enabled/mn_pairs_per_batch);
    # values bracket the ~409 positives/batch (small/≈match/large). One
    # mechanism per arm. Loss semantics stamped in the arm summary.
    "umap-md000-x4-mn100":  _umap("000", dose=4, midnear_enabled=True, mn_pairs_per_batch=100),
    "umap-md000-x4-mn410":  _umap("000", dose=4, midnear_enabled=True, mn_pairs_per_batch=410),
    "umap-md000-x4-mn1600": _umap("000", dose=4, midnear_enabled=True, mn_pairs_per_batch=1600),
    # phase 4B (PLAN4-fog): densMAP-style local-density term on the winner base.
    # Opt-in core params (default-off), log-spaced density_weight. r_hd =
    # per-row mean cosine distance to sealed-graph neighbors (precomputed).
    "umap-md000-x4-dw01": _umap("000", dose=4, density_weight=0.1,
                                density_radii_path="/data/latent-basemap/sandbox/density-radii-2m.npy"),
    "umap-md000-x4-dw03": _umap("000", dose=4, density_weight=0.3,
                                density_radii_path="/data/latent-basemap/sandbox/density-radii-2m.npy"),
    "umap-md000-x4-dw10": _umap("000", dose=4, density_weight=1.0,
                                density_radii_path="/data/latent-basemap/sandbox/density-radii-2m.npy"),
    # phase 4B faithful discriminator: same density term but the 2D radius r_2d
    # is measured to each anchor's TRUE nearest graph neighbour (precomputed
    # density-nn-2m.npy), not the crude nearest-of-6-random surrogate. If the
    # crude sweep's fog INCREASE is a surrogate artifact, these should behave
    # differently (fog flat or down); if the term genuinely fails here, they
    # confirm it. One knob changed vs the matching dw arm (density_nn_path).
    "umap-md000-x4-dw01-nn": _umap("000", dose=4, density_weight=0.1,
                                   density_radii_path="/data/latent-basemap/sandbox/density-radii-2m.npy",
                                   density_nn_path="/data/latent-basemap/sandbox/density-nn-2m.npy"),
    "umap-md000-x4-dw03-nn": _umap("000", dose=4, density_weight=0.3,
                                   density_radii_path="/data/latent-basemap/sandbox/density-radii-2m.npy",
                                   density_nn_path="/data/latent-basemap/sandbox/density-nn-2m.npy"),
    # phase 4C (PLAN4-fog): fog-targeted negatives — the only repulsion-class
    # lever (4A/4B are attraction/matching, both inflate fog). Up-weights the
    # BCE of negatives whose current 2D pair distance is in [0.1, 0.4] x p90
    # map radius, by (1 + fneg_weight). Reweighting, not resampling. One arm.
    "umap-md000-x4-fneg10": _umap("000", dose=4, fneg_weight=1.0),
    # 4C follow-up: is fog a monotone dial in fneg_weight? fneg10 Pareto-beat
    # the base (heldFFR 0.411, tissue 0.371); push the weight to chase the
    # clean corner and bracket the sweet spot.
    "umap-md000-x4-fneg30": _umap("000", dose=4, fneg_weight=3.0),
    # 4C SCALE CHECK (code-71 owner call): fneg at 6.25M dose-x2, matched to the
    # existing no-fneg control umap-md000-x2 at 6250k. Tests whether fneg10's
    # 2M Pareto-win (fog down + fidelity up, healthy) reproduces at scale — the
    # program law that a 2M result never carries upward without same-N evidence.
    "umap-md000-x2-fneg10": _umap("000", dose=2, fneg_weight=1.0),
    # P5 SCALE UNBLOCK (opt-in host-int8 X residency): the fp16/fp32 feature
    # matrix will not fit VRAM at the 25M rung (fp32 X = 35.76 GiB > 32 GB card).
    # x_residency="host_int8" keeps X in host RAM as int8 rows + fp16 scales
    # (R0262 encoding, verified bit-for-bit vs the sealed 100M substrate) and
    # dequantises per batch on device. One knob vs the matching fneg arm.
    # NOTE: run only on the 25000k rung. GPU run pending owner review of the
    # host-int8 integration report.
    "umap-md000-x4-hostint8": _umap("000", dose=4, x_residency="host_int8"),
    "umap-md000-x4-fneg10-hostint8": _umap("000", dose=4, fneg_weight=1.0,
                                           x_residency="host_int8"),
    # x2 int8 siblings (P5 pass, 2026-08-14): the 6.25M int8-vs-fp32 delta cell
    # and the 25M 4th-x2-point cell (fp32 OOMs >20M). Same DeviceEdgeSampler/
    # uwr=False as the fp32 x2 device cells — isolates X precision only.
    "umap-md000-x2-fneg10-hostint8": _umap("000", dose=2, fneg_weight=1.0,
                                           x_residency="host_int8"),
    # PLAN7 phase A (owner-triggered 2026-08-16): the promoted recipe with a
    # 3D output. Kernel/fneg math is dimension-general in core; collapse/fog
    # numbers in the summary are DESCRIPTIVE ONLY for 3D (r10 normalization is
    # the 2D packing convention — PLAN7 phase B owns the N^(1/3) re-derivation).
    # Renders = three orthogonal projections.
    "umap-md000-x4-fneg10-3d": _umap("000", dose=4, fneg_weight=1.0,
                                     n_components=3),

    # Aesthetic-revival cross (owner picks 2026-08-21, review page): the looser
    # kernels that scored below the promoted recipe but looked better, now WITH
    # the promoted fneg mechanism — does fneg recover the FFR the soft kernels
    # gave up, and does the "landmass" look survive fneg? All dose x2.
    "umap-md005-x2-fneg10": _umap("005", dose=2, fneg_weight=1.0),
    "umap-md020-x2-fneg10": _umap("020", dose=2, fneg_weight=1.0),
    "gc-a2-md000-x2-fneg10": _gc("000", 2.0, dose=2, fneg_weight=1.0),
    "gc-a2-md005-x2-fneg10": _gc("005", 2.0, dose=2, fneg_weight=1.0),
    "gc-a2-md020-x2-fneg10": _gc("020", 2.0, dose=2, fneg_weight=1.0),

    # umap-0.6dev sweep (plan-gpu-window-2026-08-21.md §3; upstream review in
    # scratchpad/umap-06dev-review.md). Rank-window hard negatives REPLACE the
    # fneg band (same job, different mechanism) so those arms run fneg-off,
    # plus one combined arm; tanh cap + kernel annealing ride the promoted
    # baseline. gamma=4.0 mirrors upstream's ±4 repulsion clip.
    "umap-md000-x2-rankneg100k": _umap("000", dose=2, rankneg_window=100_000),
    "umap-md000-x2-rankneg200k": _umap("000", dose=2, rankneg_window=200_000),
    "umap-md000-x2-rankneg500k": _umap("000", dose=2, rankneg_window=500_000),
    "umap-md000-x2-rankneg200k-xn": _umap("000", dose=2, rankneg_window=200_000,
                                          rankneg_exclude_neighbors=True),
    "umap-md000-x2-rankneg200k-fneg10": _umap("000", dose=2,
                                              rankneg_window=200_000,
                                              fneg_weight=1.0),
    "umap-md000-x2-fneg10-tanh4": _umap("000", dose=2, fneg_weight=1.0,
                                        neg_tanh_gamma=4.0),
    # gamma mini-sweep (owner 2026-08-21, after tanh4 beat promoted 0.3690 vs
    # 0.3378): is gamma a dial? Seed replicates of tanh4 run via --seed 43/44.
    "umap-md000-x2-fneg10-tanh2": _umap("000", dose=2, fneg_weight=1.0,
                                        neg_tanh_gamma=2.0),
    "umap-md000-x2-fneg10-tanh8": _umap("000", dose=2, fneg_weight=1.0,
                                        neg_tanh_gamma=8.0),
    "umap-md000-x2-fneg10-anneal25": _umap("000", dose=2, fneg_weight=1.0,
                                           kernel_anneal_frac=0.25),

    # gap-closure sweep (owner first-principles review 2026-08-22): the levers
    # upstream 0.6dev (0.4798 @ 2M) uses that our recipe has never tried under
    # the current kernel/fneg era. Next GPU window.
    "umap-md000-x2-rankneg500k-fneg10": _umap("000", dose=2,
                                              rankneg_window=500_000,
                                              fneg_weight=1.0),
    # weighted edge sampling: upstream's make_epochs_per_sample IS weighted
    # attraction; our promoted treatment pins uniform. Sandbox re-open.
    "umap-md000-x2-fneg10-wes": _umap("000", dose=2, fneg_weight=1.0,
                                      weighted_edge_sampling=True),
    # fuzzy memberships as BCE targets instead of binary 1s.
    "umap-md000-x2-fneg10-probt": _umap("000", dose=2, fneg_weight=1.0,
                                        positive_target_mode="probability"),
    # best mechanisms at the family dose, and the dose ceiling: upstream's
    # effective positive dose is ~10-15x ours.
    "umap-md000-x4-fneg10-tanh4": _umap("000", dose=4, fneg_weight=1.0,
                                        neg_tanh_gamma=4.0),
    "umap-md000-x8-fneg10": _umap("000", dose=8, fneg_weight=1.0),
    # force balance under the current best mechanisms (crude version of
    # upstream's global/local auto-calibration).
    "umap-md000-x2-fneg10-tanh4-pos02": _umap("000", dose=2, fneg_weight=1.0,
                                              neg_tanh_gamma=4.0,
                                              pos_ratio=0.02),
    "umap-md000-x2-fneg10-tanh4-pos10": _umap("000", dose=2, fneg_weight=1.0,
                                              neg_tanh_gamma=4.0,
                                              pos_ratio=0.10),
    # composed best (2026-08-22 gap-closure verdict: dose dominates, tanh +
    # pos10 stack): all three winning levers at once.
    "umap-md000-x8-fneg10-tanh4-pos10": _umap("000", dose=8, fneg_weight=1.0,
                                              neg_tanh_gamma=4.0,
                                              pos_ratio=0.10),
    "umap-md000-x16-fneg10-tanh4-pos10": _umap("000", dose=16, fneg_weight=1.0,
                                               neg_tanh_gamma=4.0,
                                               pos_ratio=0.10),
    # dose/composition factorial fill (external review 2026-08-22): pos10
    # DOUBLES positive draws/update, so x8+pos10 is ~x16 POSITIVE exposure vs
    # the pos05 recipe. These controls separate positive dose from total dose
    # and complete the tanh x pos factorial.
    "umap-md000-x4-fneg10-tanh4-pos10": _umap("000", dose=4, fneg_weight=1.0,
                                              neg_tanh_gamma=4.0,
                                              pos_ratio=0.10),
    "umap-md000-x8-fneg10-tanh4": _umap("000", dose=8, fneg_weight=1.0,
                                        neg_tanh_gamma=4.0),
    "umap-md000-x2-fneg10-pos10": _umap("000", dose=2, fneg_weight=1.0,
                                        pos_ratio=0.10),
    "umap-md000-x4-fneg10-tanh2": _umap("000", dose=4, fneg_weight=1.0,
                                        neg_tanh_gamma=2.0),
    # gamma-2 promotion path (tanh2 beat tanh4 at x4: 0.3981 vs 0.3936).
    "umap-md000-x4-fneg10-tanh2-pos10": _umap("000", dose=4, fneg_weight=1.0,
                                              neg_tanh_gamma=2.0,
                                              pos_ratio=0.10),
    "umap-md000-x8-fneg10-tanh2-pos10": _umap("000", dose=8, fneg_weight=1.0,
                                              neg_tanh_gamma=2.0,
                                              pos_ratio=0.10),
    # architecture sweep (owner 2026-08-23): width/depth/arch under the x2
    # composed core (ref 0.3734 at h2048/L3/residual_bottleneck). Smaller nets
    # are ALSO the real throughput lever (the loop is encoder-GPU-bound).
    "core-h512": _umap("000", dose=2, fneg_weight=1.0, neg_tanh_gamma=4.0,
                       pos_ratio=0.10, hidden_dim=512),
    "core-h1024": _umap("000", dose=2, fneg_weight=1.0, neg_tanh_gamma=4.0,
                        pos_ratio=0.10, hidden_dim=1024),
    "core-h3072": _umap("000", dose=2, fneg_weight=1.0, neg_tanh_gamma=4.0,
                        pos_ratio=0.10, hidden_dim=3072),
    "core-h4096": _umap("000", dose=2, fneg_weight=1.0, neg_tanh_gamma=4.0,
                        pos_ratio=0.10, hidden_dim=4096),
    "core-L2": _umap("000", dose=2, fneg_weight=1.0, neg_tanh_gamma=4.0,
                     pos_ratio=0.10, n_layers=2),
    "core-L4": _umap("000", dose=2, fneg_weight=1.0, neg_tanh_gamma=4.0,
                     pos_ratio=0.10, n_layers=4),
    "core-L5": _umap("000", dose=2, fneg_weight=1.0, neg_tanh_gamma=4.0,
                     pos_ratio=0.10, n_layers=5),
    "core-mlp": _umap("000", dose=2, fneg_weight=1.0, neg_tanh_gamma=4.0,
                      pos_ratio=0.10, architecture="mlp"),

    # best-efficiency guess (owner 2026-08-23 v2): champion stack at md010,
    # with EVERY efficiency winner from the sweeps — h1024 (2.2x faster),
    # 2 layers (shallower won), plain mlp (beat the bottleneck), bs16k
    # (+15% edges/s), dose x4-in-updates = champion positive draws/edge at a
    # fraction of the wall clock.
    "umap-md010-h1024L2mlp-bs16k-x4-winner": _umap(
        "010", dose=4, fneg_weight=1.0, neg_tanh_gamma=4.0, pos_ratio=0.10,
        rankneg_window=500_000, hidden_dim=1024, n_layers=2,
        architecture="mlp", batch_size=16384),
    "umap-md020-h1024L2mlp-bs16k-x4-winner": _umap(
        "020", dose=4, fneg_weight=1.0, neg_tanh_gamma=4.0, pos_ratio=0.10,
        rankneg_window=500_000, hidden_dim=1024, n_layers=2,
        architecture="mlp", batch_size=16384),

    # night12 winner set (candidate confirmed 2026-08-23: x8+tanh4+pos10).
    "umap-md000-x8-fneg10-tanh4-pos10-rankneg500k": _umap(
        "000", dose=8, fneg_weight=1.0, neg_tanh_gamma=4.0, pos_ratio=0.10,
        rankneg_window=500_000),
    # anti-collapse ladder (owner 2026-08-22: "maps slightly too collapsed"):
    # looser kernels x the best candidates — cheap x2 looks first, then the
    # composed winner at md005/md010.
    "umap-md005-x2-fneg10-tanh4": _umap("005", dose=2, fneg_weight=1.0,
                                        neg_tanh_gamma=4.0),
    "umap-md010-x2-fneg10-tanh4": _umap("010", dose=2, fneg_weight=1.0,
                                        neg_tanh_gamma=4.0),
    "umap-md005-x8-fneg10-tanh4-pos10": _umap("005", dose=8, fneg_weight=1.0,
                                              neg_tanh_gamma=4.0,
                                              pos_ratio=0.10),
    "umap-md010-x8-fneg10-tanh4-pos10": _umap("010", dose=8, fneg_weight=1.0,
                                              neg_tanh_gamma=4.0,
                                              pos_ratio=0.10),
    # rejected-in-isolation composition screen (owner 2026-08-22): the x2
    # composed core (fneg10+tanh4+pos10, ref 0.3734) + one rejected lever
    # each. Beat the ref -> earns an x8 slot.
    "umap-md000-x2-core-rankneg500k": _umap("000", dose=2, fneg_weight=1.0,
                                            neg_tanh_gamma=4.0, pos_ratio=0.10,
                                            rankneg_window=500_000),
    "umap-md000-x2-core-wes": _umap("000", dose=2, fneg_weight=1.0,
                                    neg_tanh_gamma=4.0, pos_ratio=0.10,
                                    weighted_edge_sampling=True),
    "umap-md000-x2-core-anneal25": _umap("000", dose=2, fneg_weight=1.0,
                                         neg_tanh_gamma=4.0, pos_ratio=0.10,
                                         kernel_anneal_frac=0.25),
    "umap-md005-x2-core": _umap("005", dose=2, fneg_weight=1.0,
                                neg_tanh_gamma=4.0, pos_ratio=0.10),
}


def gpu_is_free() -> bool:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        raise SystemExit("nvidia-smi failed; do not touch a possibly-sick driver")
    return not out.stdout.strip()


def resolve_edges(rung: dict) -> Path:
    if rung["edges"] is not None:
        return rung["edges"]
    import glob
    hits = [p for p in sorted(glob.glob(rung["edges_glob"]))]
    if not hits:
        raise SystemExit(f"no fuzzy edge npz at {rung['edges_glob']}")
    return Path(hits[0])


def receipt_diff(kwargs: dict, sealed_checkpoint: Path) -> list[str]:
    """Fields where our constructor departs from the rung's sealed checkpoint."""
    import torch
    ck = torch.load(sealed_checkpoint, map_location="cpu", weights_only=False)
    mismatches = []
    # sampling mode is a treatment field (added 2026-08-14 after R0265's seed-42 cross-check)
    for key in ("a", "b", "low_dim_kernel", "correlation_weight", "learning_rate",
                "pos_ratio", "positive_target_mode", "use_batchnorm", "use_dropout",
                "architecture", "weighted_edge_sampling"):
        if kwargs.get(key) != ck.get(key):
            mismatches.append(f"{key}: ours={kwargs.get(key)!r} sealed={ck.get(key)!r}")
    return mismatches


def quick_ffr(xy: np.ndarray, edges_path: Path, rows: int,
              n_queries: int = 20_000, k_true: int = 15) -> float:
    """FFR@0.1% with the rung's sealed graph edges as high-D truth."""
    from scipy.spatial import cKDTree

    with np.load(edges_path) as z:
        names = z.files
        src_name = next(n for n in ("sources", "src", "rows") if n in names)
        dst_name = next(n for n in ("destinations", "dst", "cols", "targets") if n in names)
        sources = z[src_name]
        dests = z[dst_name]
    order = np.argsort(sources, kind="stable")
    sources, dests = sources[order], dests[order]
    starts = np.searchsorted(sources, np.arange(rows))
    ends = np.searchsorted(sources, np.arange(rows), side="right")

    rng = np.random.default_rng(0)
    queries = rng.choice(rows, size=n_queries, replace=False)
    disc = max(int(rows * 0.001), 1)
    tree = cKDTree(xy)
    _, near = tree.query(xy[queries], k=disc, workers=8)
    hits = 0
    total = 0
    for qi, q in enumerate(queries):
        truth = dests[starts[q]:ends[q]][:k_true]
        if len(truth) == 0:
            continue
        hits += np.isin(truth, near[qi]).sum()
        total += len(truth)
    return hits / max(total, 1)


def run_arm(arm: str, dry_run: bool, seed: int = SEED, rung_name: str = "2m") -> int:
    rung = RUNGS[rung_name]
    overrides = dict(ARMS[arm])
    dose_mult = overrides.pop("dose", 1)
    kwargs = {**BASE_KWARGS, **overrides}
    horizon = round(dose_mult * rung["base_horizon"])
    num_pos = int(kwargs.get("batch_size", BATCH) * kwargs["pos_ratio"])
    steps_per_epoch = math.ceil(rung["directed_edges"] / num_pos)
    kwargs["total_steps_estimate"] = horizon
    kwargs["n_epochs"] = max(1, math.ceil(horizon / steps_per_epoch))
    dose = horizon * num_pos / rung["directed_edges"]

    edges_path = resolve_edges(rung)
    for path in (rung["substrate"], edges_path, rung["sealed_checkpoint"]):
        if not path.is_file():
            raise SystemExit(f"missing sealed input: {path}")

    expected_knobs = set(overrides) | {"kernel_alpha"}
    departures = [d for d in receipt_diff(kwargs, rung["sealed_checkpoint"])
                  if d.split(":")[0] not in expected_knobs]
    print(f"rung={rung_name} arm={arm} seed={seed} horizon={horizon} updates "
          f"dose={dose:.4f} draws/edge n_epochs={kwargs['n_epochs']}")
    if departures:
        raise SystemExit("treatment departs from the sealed rung recipe outside "
                         "the arm's knobs:\n  " + "\n  ".join(departures))
    print("receipt check: all non-arm fields match the sealed checkpoint")
    if dry_run:
        return 0

    out_dir = rung["out_root"] / (arm if seed == SEED else f"{arm}-seed{seed}")
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; arms are write-once (delete to re-run)")
    if not gpu_is_free():
        raise SystemExit("GPU busy (round runner has priority); try again later")
    out_dir.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(out_dir / "fit.log"),
                                  logging.StreamHandler()])

    import torch
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    torch.manual_seed(seed)
    started = datetime.datetime.now(datetime.timezone.utc)
    X = np.load(rung["substrate"], mmap_mode="r")
    model = ParametricUMAP(**kwargs)
    model.fit(X, precomputed_edges_path=str(edges_path), random_state=seed)
    model.save(str(out_dir / "model.pt"))
    xy = model.transform(X, batch_size=8192).astype(np.float32)
    np.save(out_dir / "coordinates.npy", xy)
    wall_s = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds()

    ffr = quick_ffr(xy, edges_path, rung["rows"])
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    sample = xy[rng.choice(len(xy), 20_000, replace=False)]
    dists, _ = cKDTree(xy).query(sample, k=11, workers=8)
    radius = np.percentile(np.linalg.norm(xy - xy.mean(axis=0), axis=1), 90)
    r10 = float(np.median(dists[:, 10]) / radius)
    from map_renders import robust_extent, binned_counts, render_png
    if xy.shape[1] == 2:
        render_png(binned_counts(xy, robust_extent(xy)), out_dir / "density.png")
    else:
        # 3D arms: three orthogonal projections (PLAN7 rendering-for-the-record);
        # the xy view doubles as density.png so the legacy page still shows it.
        import shutil
        for name, (i, j) in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
            proj = np.ascontiguousarray(xy[:, (i, j)])
            render_png(binned_counts(proj, robust_extent(proj)),
                       out_dir / f"density-{name}.png")
        shutil.copy2(out_dir / "density-xy.png", out_dir / "density.png")

    summary = {
        "arm": arm, "rung": rung_name, "overrides": overrides, "seed": seed,
        "n_components": kwargs.get("n_components", 2),
        "dose_multiplier": dose_mult, "horizon_updates": horizon,
        "draws_per_edge": dose, "wall_s": wall_s, "quick_ffr_at_0.1pct": ffr,
        "r10_over_map_radius_median": r10,
        "substrate": str(rung["substrate"]), "edges": str(edges_path),
        "started_utc": started.isoformat(),
        "note": "sandbox artifact; not a round, no sealed claim",
    }
    fneg_tel = getattr(model, "fneg_telemetry", None)
    if fneg_tel is not None:
        summary["fneg_telemetry"] = fneg_tel
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    build_page()
    return 0


def build_page() -> None:
    """Legacy phase-1/2 page (2M only). The kernel program's dedicated page is
    built by build_kernels_page.py and covers both rungs."""
    import html as html_mod
    import shutil

    OUT_ROOT = RUNGS["2m"]["out_root"]
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    cards = []
    sealed = Path("/data/latent-basemap/render-cache"
                  "/round-0217-minilm-mixed-2m-map-seed42-low-dose-v1/density.png")
    if sealed.is_file():
        shutil.copy2(sealed, SITE_DIR / "sealed-r0217-seed42.png")
        cards.append(("sealed R0217 seed 42", "sealed-r0217-seed42.png",
                      "the registered low-dose baseline"))
    for arm_dir in sorted(OUT_ROOT.iterdir()) if OUT_ROOT.is_dir() else []:
        s_path = arm_dir / "summary.json"
        png = arm_dir / "density.png"
        if not (s_path.is_file() and png.is_file()):
            continue
        s = json.loads(s_path.read_text())
        label = arm_dir.name
        name = f"{label}.png"
        shutil.copy2(png, SITE_DIR / name)
        # external-baseline summaries (e.g. paramrepulsor-upstream) carry no
        # horizon/draws fields — show what exists rather than crashing the
        # whole arm run at page-build time.
        bits = [f"quick-ffr {s['quick_ffr_at_0.1pct']:.4f}"]
        if "horizon_updates" in s:
            bits.append(f"{s['horizon_updates']:,} updates")
        if "draws_per_edge" in s:
            bits.append(f"{s['draws_per_edge']:.3f} draws/edge")
        cards.append((label, name, " · ".join(bits)))
    figs = "".join(
        f'<figure style="margin:0"><img src="{f}" style="width:100%;border:1px solid #ddd;border-radius:6px">'
        f'<figcaption><b>{html_mod.escape(t)}</b><br><small>{html_mod.escape(c)}</small></figcaption></figure>'
        for t, f, c in cards)
    (SITE_DIR / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>2M knob sandbox</title>"
        '<body style="font-family:system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem">'
        "<h1>2M visual-quality knobs (sandbox)</h1>"
        "<p>One knob per arm on the sealed R0216 substrate/graph, seed 42. "
        "Not rounds; quick-ffr uses sealed graph edges as truth. "
        'Kernel program page: <a href="../kernels/">sandbox/kernels</a>.</p>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px">{figs}</div>')
    print(f"page: {SITE_DIR}/index.html")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=sorted(ARMS), required=False)
    ap.add_argument("--rung", choices=sorted(RUNGS), default="2m")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild-page", action="store_true")
    args = ap.parse_args()
    if args.rebuild_page:
        build_page()
        return 0
    if not args.arm:
        raise SystemExit("pass --arm (or --rebuild-page)")
    return run_arm(args.arm, args.dry_run, seed=args.seed, rung_name=args.rung)


if __name__ == "__main__":
    sys.exit(main())
