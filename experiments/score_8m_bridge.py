"""P4 — score the retrained WEIGHTED 8M bridge maps under panel v2.2 and bind the
ACTUAL training pipeline/semantics into the evidence (closure item 4). Scoring
runs in this fresh process (one GPU phase per process); it reads each seed's
training results.json for the stamped pipeline + positive-LR update count, then
scores the coords transductively (ffr/purity/density, FineWeb purity mask).

Usage (under a held lease):
  python experiments/score_8m_bridge.py --out experiments/evidence/r1_8m/bridge_weighted.json
"""
from __future__ import annotations
import argparse, os, sys, json, glob, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.panel_v2 import (score_panel, PanelV2Config, load_embeddings, load_coords,
                              sample_anchors, FORMULA_VERSION, build_hiD_reference,
                              save_hiD_reference, load_hiD_reference, hiD_reference_key)
import numpy as _np

TRAIN8M = "/data/latent-basemap/jina-en-8M-nested/train"
LABELS8M = "/data/latent-basemap/jina-en-8m/corpus_labels.npy"
C256 = "/data/latent-basemap/track1/audit_centroids_k256_s0.npy"
C1024 = "/data/latent-basemap/track1/centroids_fineweb_k1024.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-anchors", type=int, default=2000)
    ap.add_argument("--reference", default=None, help="L0.4 shared hi-D reference .npz")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="optional label=run_dir pairs (e.g. legacy=<dir> stdcurve=<dir>); "
                         "default = the two weighted bridge seeds")
    ap.add_argument("--out", default="experiments/evidence/r1_8m/bridge_weighted.json")
    args = ap.parse_args()
    # S1: GPU scorer must run under a held/inherited lease (controller or in-process).
    import torch as _t
    if _t.cuda.is_available():
        from basemap.run_controller import require_active_lease
        require_active_lease()
    X = load_embeddings(TRAIN8M, dim=768); n = len(X)
    labels = np.asarray(np.load(LABELS8M, mmap_mode="r")[:n])
    cents = {256: np.load(C256), 1024: np.load(C1024)}
    cfg = PanelV2Config(frac=0.001, n_anchors=args.n_anchors, corpus_chunk=500_000, k_clust=(256, 1024))
    aidx = sample_anchors(n, cfg); fw = (labels[aidx] == 0)
    # L0.4: build/load + key-verify the ONE shared 8M reference; every map reuses it.
    ref = None
    if args.reference:
        kf = max(cfg.k_hit, int(_np.ceil(cfg.frac * n)))
        want_key, _ = hiD_reference_key(X, aidx, cfg, cents, kf=kf)
        if os.path.exists(args.reference) or os.path.exists(args.reference + ".npz"):
            ref = load_hiD_reference(args.reference)
            if ref["key"] != want_key:
                raise ValueError(f"8M reference key {ref['key']} != recomputed {want_key} (L0.4).")
        else:
            ref = build_hiD_reference(X, aidx, cfg, cents); save_hiD_reference(ref, args.reference)
        print(f"[bridge] shared reference key={ref['key']}", flush=True)
    out = {"substrate": "mixed-8M (fineweb+rpj+pile)", "recipe": "legacy nomn+weighted",
           "budget": "500k updates v3", "n_anchors": args.n_anchors,
           "formula_version": FORMULA_VERSION,
           "hiD_reference_key": (ref["key"] if ref is not None else None), "seeds": {}}
    # default: the two weighted bridge seeds; optional --runs label=dir pairs (G1 pair).
    run_map = (dict(kv.split("=", 1) for kv in args.runs) if args.runs
               else {f"s{s}": None for s in (42, 43)})
    for label, rd_arg in run_map.items():
        seed = label
        if rd_arg is None:
            sd = label[1:] if label.startswith("s") else label
            rds = sorted(glob.glob(f"experiments/results/r1_8m_bridge_weighted_s{sd}_*"))
            if not rds:
                out["seeds"][label] = {"error": "no run dir"}; continue
            rd = rds[-1]
        else:
            rd = rd_arg
        tr = json.load(open(os.path.join(rd, "results.json")))
        acct = (tr.get("run_manifest") or {}).get("train_accounting", {})
        Z, zid = load_coords(os.path.join(rd, "coords.parquet"))
        t0 = time.time()
        p = score_panel(X, Z, config=cfg, centroids_by_k=cents, hiD_reference=ref,
                        anchor_masks={"ffr": None, "purity": fw, "density": None},
                        provenance={"gate": "weighted_8m_bridge", "run": os.path.basename(rd)})
        if ref is not None and not p["provenance"].get("hiD_reference_reused"):
            raise ValueError(f"map {label} did not reuse the shared 8M reference (L0.4).")
        out["seeds"][label] = {
            "run_dir": os.path.basename(rd), "score_wall_s": round(time.time() - t0, 1),
            # ACTUAL execution pipeline/semantics (P1 stamp) — the review's core requirement
            "pipeline": acct.get("pipeline_pipeline"),
            "positive_sampling": acct.get("pipeline_positive_sampling"),
            "sampler_class": acct.get("pipeline_sampler_class"),
            "x_residency": acct.get("pipeline_x_residency"),
            "positive_lr_updates": acct.get("positive_lr_optimizer_steps"),
            "budget_satisfied": acct.get("budget_satisfied"),
            "schedule_version": acct.get("schedule_version"),
            "updates_per_s": acct.get("updates_per_s"),
            "formula_version": p["formula_version"],
            "hiD_reference_key": p["provenance"].get("hiD_reference_key"),
            "hiD_reference_reused": p["provenance"].get("hiD_reference_reused"),
            "ffr": p["ffr"], "recall@k": p["recall@k"], "purity": p.get("purity"), "density": p["density"],
            "exactness": p["provenance"].get("exactness"),
        }
        s = out["seeds"][label]
        print(f"[bridge] {label}: pipeline={s['pipeline']}/{s['positive_sampling']} "
              f"updates={s['positive_lr_updates']} | ffr={s['ffr']} purity={s['purity']} "
              f"density={s['density']} ({s['updates_per_s']} upd/s)", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[bridge] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
