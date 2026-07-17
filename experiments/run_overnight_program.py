"""Overnight autonomous GPU program: R0.1 closure items 3 + 4, then the 6-hour
8M block (kernel comparison + capacity + 3D). ONE process holds ONE GpuLease for
the whole run (blocking until any prior lease-holder releases), so it chains
straight through with no idle GPU. Each phase is fault-isolated (a failure logs
and the program continues) and writes durable JSON so partial progress survives.

Phases:
  3a. kNN-regressor index/query cost (200k + 2M) — the NUMAP-baseline cost the
      review wants measured, not assumed free.
  3b. 2M golden gate EXTENDED to purity + projection (real substrate).
  4 . matched-budget 8M bridge — legacy nomn+weighted, 2 seeds (the "ffr flat
      with scale" headline point), transductive panel (ffr/purity/density).
  6h. 8M kernel comparison (umap a=b=1, umap std-curve) + capacity (h2048) + 3D,
      all legacy-recipe substrate, matched 500k-update v3 budget.

8M maps are scored TRANSDUCTIVELY (ffr/purity/density with the FineWeb purity
mask) — the 8M block has no sample_indices, so held-out projection is N/A there
(consistent with the A3 8M scoring).

Run (via the orchestrator, which waits for the A3 job to free the lease):
  BASEMAP_GPU_LEASE=/data/latent-basemap/.gpu_lease HF_HOME=/data/hf \
    python experiments/run_overnight_program.py --out /data/latent-basemap/overnight/summary.json
"""
from __future__ import annotations
import argparse, os, sys, json, time, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.experiment_config import load_config
from experiments.run_experiment import run_single_experiment
from basemap.run_controller import GpuLease, known_service_pids, check_co_tenants
from basemap.round0005_retirement import refuse_retired_launcher
from basemap.panel_v2 import (score_panel, PanelV2Config, load_embeddings, load_coords,
                              cross_knn, ffr_from_neighbors, sample_anchors)

STD_A, STD_B = 1.57694346, 0.895060879
SRC = "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train"
# 8M scoring assets (reuse the A3 frozen centroids + FineWeb labels)
TRAIN8M = "/data/latent-basemap/jina-en-8M-nested/train"
LABELS8M = "/data/latent-basemap/jina-en-8m/corpus_labels.npy"
C256 = "/data/latent-basemap/track1/audit_centroids_k256_s0.npy"
C1024 = "/data/latent-basemap/track1/centroids_fineweb_k1024.npy"

# 8M program: (name, kernel, a, b, hidden_dim, n_components, seed)
PROGRAM_8M = [
    ("bridge_legacy_h1024_s42", "legacy_lp", 1.0, 1.0, 1024, 2, 42),   # item 4 bridge
    ("bridge_legacy_h1024_s43", "legacy_lp", 1.0, 1.0, 1024, 2, 43),   # 2nd seed
    ("umap_a1b1_h1024_s42",     "umap", 1.0, 1.0, 1024, 2, 42),        # 8M kernel cmp
    ("umap_stdcurve_h1024_s42", "umap", STD_A, STD_B, 1024, 2, 42),    # 8M kernel cmp
    ("legacy_h2048_s42",        "legacy_lp", 1.0, 1.0, 2048, 2, 42),   # capacity A1/A2
    ("legacy_3d_h1024_s42",     "legacy_lp", 1.0, 1.0, 1024, 3, 42),   # 3D A4
]


def _phase(summary, name, fn, out_path):
    t0 = time.time()
    try:
        summary[name] = {"status": "ok", "result": fn()}
    except Exception as e:
        summary[name] = {"status": "error", "error": repr(e), "trace": traceback.format_exc()[-1500:]}
        print(f"[overnight] PHASE {name} FAILED: {e}", flush=True)
    summary[name]["wall_s"] = round(time.time() - t0, 1)
    json.dump(summary, open(out_path, "w"), indent=1)
    print(f"[overnight] phase {name} {summary[name]['status']} ({summary[name]['wall_s']}s)", flush=True)


# ── phase 3a: kNN-regressor cost ──────────────────────────────────────────────

def phase_knn_cost():
    cfg = PanelV2Config(frac=0.001, n_anchors=10000, corpus_chunk=500_000)
    out = {}
    for name, testbed, dim in [("200k", "/data/latent-basemap/jina-en-200k", 768),
                               ("2m", "/data/latent-basemap/jina-en-2m", 768)]:
        X = load_embeddings(os.path.join(testbed, "train"), dim=dim)
        si = np.load(os.path.join(testbed, "sample_indices.npy"))
        src = load_embeddings(SRC, dim=dim)
        comp = np.setdiff1d(np.arange(len(src), dtype=np.int64), np.asarray(sorted(set(int(i) for i in si)), np.int64))
        rng = np.random.RandomState(123)
        held = np.sort(rng.choice(comp, 20000, replace=False))
        Xq = np.asarray(src[held], dtype=np.float32)
        # cost = time to find each held-out query's k nearest TRAIN rows in high-D
        # (the kNN-regressor's query step; the "index" is the training X itself).
        t0 = time.time()
        _ = cross_knn(Xq, X, 15, cfg, hi_dim=True)
        q_s = time.time() - t0
        out[name] = {"n_train": int(len(X)), "n_query": int(len(Xq)),
                     "knn_query_wall_s": round(q_s, 2),
                     "queries_per_s": round(len(Xq) / q_s, 1),
                     "note": "kNN-regressor query = hi-D k-NN over the full train set per held-out point; "
                             "NOT free vs the parametric map's constant-time transform (~430k rows/s)."}
        print(f"[overnight] knn_cost {name}: {out[name]['knn_query_wall_s']}s for {len(Xq)} queries "
              f"over {len(X):,} train", flush=True)
    return out


# ── phase 3b: 2M golden extended to purity + projection ───────────────────────

def phase_golden_2m_extended():
    from experiments.score_complete_panel import frozen_centroids, projection_ffr, knn_regress_coords
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    cfg = PanelV2Config(frac=0.001, n_anchors=2000, corpus_chunk=500_000)
    testbed = "/data/latent-basemap/jina-en-2m"
    X = load_embeddings(os.path.join(testbed, "train"), dim=768)
    # a representative trained 2M legacy map (seed 42) — use its coords + model
    import glob
    rd = sorted(glob.glob("experiments/results/r1_kernel_2m_legacy_a1b1_s42_*"))[-1]
    Z, z_ids = load_coords(os.path.join(rd, "coords.parquet"))
    cents = frozen_centroids(X, (256, 1024), testbed)
    panel = score_panel(X, Z, config=cfg, z_ids=z_ids, centroids_by_k=cents,
                        provenance={"gate": "golden_2m_extended", "run": os.path.basename(rd)})
    # projection on real held-out queries
    si = np.load(os.path.join(testbed, "sample_indices.npy"))
    src = load_embeddings(SRC, dim=768)
    comp = np.setdiff1d(np.arange(len(src), dtype=np.int64), np.asarray(sorted(set(int(i) for i in si)), np.int64))
    held = np.sort(np.random.RandomState(7).choice(comp, 20000, replace=False))
    Xq = np.asarray(src[held], dtype=np.float32)
    Xa = X[np.asarray(z_ids, np.int64)] if z_ids is not None else X
    model = ParametricUMAP.load(os.path.join(rd, "model.pt"), device="cuda")
    Zq = np.asarray(model.transform(Xq), dtype=np.float32)
    pf, pr = projection_ffr(Xa, Z, Xq, Zq, cfg)
    return {"gate": "golden_2m_extended", "run": os.path.basename(rd),
            "ffr": panel["ffr"], "purity": panel.get("purity"), "density": panel["density"],
            "proj_ffr": pf, "covers": ["ffr", "purity", "density", "projection"],
            "n_holdout_unique": int(len(np.unique(held)))}


# ── phases 4 + 6h: the 8M training program ────────────────────────────────────

def _build_8m_cfg(name, kernel, a, b, hidden_dim, ncomp, seed):
    cfg = load_config("experiments/configs/jina_en_8m_nested.yaml", {
        "model.low_dim_kernel": kernel, "model.a": a, "model.b": b,
        "model.hidden_dim": hidden_dim, "model.n_components": ncomp,
        "data.random_seed": seed,
        "train.total_steps_estimate": 500000, "train.lr_schedule": "cosine",
        "train.warmup_steps": 200, "train.midnear_enabled": False,
        "train.weighted_edge_sampling": True,
        "train.require_graph_manifest": True, "train.require_full_budget": True,
    })
    cfg.name = f"r1_8m_{name}"
    cfg.eval.metrics = ["panel_v2"] if "panel_v2" not in cfg.eval.metrics else cfg.eval.metrics
    # 8M block has no sample_indices → transductive scoring only; the runner's
    # panel_v2 metric runs on the full X/Z (ffr/recall/density, no projection).
    return cfg


def phase_8m_program(out_path, summary):
    refuse_retired_launcher("experiments/run_overnight_program.py")
    import glob
    X = load_embeddings(TRAIN8M, dim=768)
    n = len(X)
    labels = np.asarray(np.load(LABELS8M, mmap_mode="r")[:n])
    cents = {256: np.load(C256), 1024: np.load(C1024)}
    # 2000 anchors (not 10k): the k_frac≈8000 exact-rerank gathers ~n_anchors×8009
    # rows from the 8M memmap; 10k anchors is CPU-bound for many minutes/map. 2k is
    # 5× fewer tiles, statistically ample for the scale/recipe comparison (the
    # golden gate used 512), and keeps the overnight program within budget.
    cfg_score = PanelV2Config(frac=0.001, n_anchors=2000, corpus_chunk=500_000, k_clust=(256, 1024))
    aidx = sample_anchors(n, cfg_score)
    fineweb_mask = (labels[aidx] == 0)
    results = {}
    for spec in PROGRAM_8M:
        name, kernel, a, b, hidden_dim, ncomp, seed = spec
        rec = {"kernel": kernel, "a": a, "b": b, "hidden_dim": hidden_dim, "n_components": ncomp, "seed": seed}
        try:
            cfg = _build_8m_cfg(*spec)
            assert cfg.model.low_dim_kernel == kernel
            print(f"\n[overnight] === 8M train {cfg.name} kernel={kernel} h={hidden_dim} "
                  f"ncomp={ncomp} seed={seed} ===", flush=True)
            t0 = time.time()
            res = run_single_experiment(cfg)
            rec["train_s"] = round(time.time() - t0, 1)
            rec["train_accounting"] = (res.get("run_manifest") or {}).get("train_accounting", {})
            rd = sorted(glob.glob(os.path.join("experiments/results", f"{cfg.name}_*")))[-1]
            rec["run_dir"] = os.path.basename(rd)
            Z, z_ids = load_coords(os.path.join(rd, "coords.parquet"))
            panel = score_panel(X, Z, config=cfg_score, centroids_by_k=cents,
                                anchor_masks={"ffr": None, "purity": fineweb_mask, "density": None},
                                provenance={"prog": "8m", "run": os.path.basename(rd)})
            rec.update({"ffr": panel["ffr"], "recall@k": panel["recall@k"],
                        "purity": panel.get("purity"), "density": panel["density"],
                        "n_dims_lo": panel["n_dims_lo"]})
            print(f"[overnight] 8M {name}: ffr={panel['ffr']} purity={panel.get('purity')} "
                  f"dens={panel['density']} ({rec['train_s']}s)", flush=True)
        except Exception as e:
            rec["status"] = "error"; rec["error"] = repr(e)
            rec["trace"] = traceback.format_exc()[-1500:]
            print(f"[overnight] 8M {name} FAILED: {e}", flush=True)
        results[name] = rec
        summary["phase_8m"] = {"status": "partial", "runs": results}
        json.dump(summary, open(out_path, "w"), indent=1)
    return results


def main():
    refuse_retired_launcher("experiments/run_overnight_program.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/latent-basemap/overnight/summary.json")
    ap.add_argument("--required-free-gb", type=float, default=14.0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # ONE lease for the whole program; block until any prior holder (the A3 job)
    # releases it, so the GPU is never idle between phases.
    print("[overnight] acquiring GPU lease (blocking until free)…", flush=True)
    with GpuLease(timeout=None) as lease:
        check_co_tenants(args.required_free_gb, allowed_pids=known_service_pids(), wait_s=60)
        print("[overnight] lease acquired; starting program.", flush=True)
        _phase(summary, "phase_3a_knn_cost", phase_knn_cost, args.out)
        _phase(summary, "phase_3b_golden_extended", phase_golden_2m_extended, args.out)
        # phases 4 + 6h (writes incrementally itself)
        try:
            phase_8m_program(args.out, summary)
            summary["phase_8m"]["status"] = "done"
        except Exception as e:
            summary["phase_8m"] = {"status": "error", "error": repr(e)}
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json.dump(summary, open(args.out, "w"), indent=1)
    print(f"[overnight] program complete → {args.out}", flush=True)


if __name__ == "__main__":
    main()
