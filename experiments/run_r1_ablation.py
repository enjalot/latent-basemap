"""R1 2×2 mid-near × weighted-sampling ablation (plan §R1) — un-confounds A3.

A3 flipped mid-near AND weighted sampling together, so its "mid-near trades local
for density" read was confounded. This isolates the two factors on the DECIDED
kernel (legacy_lp), 200k, matched-budget, on the HISTORICAL doses:

  mid-near ∈ {off (nomn, scale 1), on (mn4, scale 4.0 — the A3 dose)}
  weighted_edge_sampling ∈ {false, true}

IMPORTANT (P0-1, overnight review 2026-07-14): the historical A3 `mn4` level uses
`mn_weight_scale=4.0`. Earlier ablation runs left the default `1.0` while calling
the on level "mn4" — those are a *scale-1* pilot and CANNOT estimate the A3
mid-near factor. This launcher sets `mn_weight_scale=4.0` on the on level and
STAMPS the dose into every tag/assertion/summary so a dose can never go unlabeled.

Reuse: cells whose exact config (by config hash) already trained are reused from
disk, so only the corrected scale-4 on cells need to run.

Usage:
  python experiments/run_r1_ablation.py --budget 500000 --seed 42 \
     --base experiments/configs/jina_en_200k_k50_fuzzy.yaml \
     --out /data/latent-basemap/r1_ablation/summary_mn4.json
"""
from __future__ import annotations
import argparse, os, sys, json, glob, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.experiment_config import load_config
from experiments.run_experiment import run_single_experiment
from experiments.run_r1_kernel import extract_panel
from basemap.run_controller import GpuLease, check_co_tenants, known_service_pids

MN_ON_SCALE = 4.0                                   # the historical A3 mn4 dose
CELLS = [(mn, ws) for mn in (False, True) for ws in (False, True)]


def build_cfg(base, mn, ws, seed, budget):
    scale = MN_ON_SCALE if mn else 1.0
    tag = f"mn{'1' if mn else '0'}_s{str(scale).replace('.0','')}_ws{'1' if ws else '0'}"
    cfg = load_config(base, {
        "model.low_dim_kernel": "legacy_lp",
        "model.a": 1.0, "model.b": 1.0,
        "data.random_seed": seed,
        "train.total_steps_estimate": budget,
        "train.lr_schedule": "cosine", "train.warmup_steps": 200,
        "train.midnear_enabled": mn,
        "train.mn_weight_scale": scale,            # STAMP the dose (P0-1)
        "train.weighted_edge_sampling": ws,
    })
    cfg.name = f"r1_abl_{tag}_s{seed}"
    if "panel_v2" not in cfg.eval.metrics:
        cfg.eval.metrics = list(cfg.eval.metrics) + ["panel_v2"]
    return tag, scale, cfg


def _find_reusable(cfg):
    """Return an existing run dir whose config hash matches this exact cfg (so a
    correctly-configured completed cell is reused, never a differently-dosed one)."""
    for d in sorted(glob.glob(os.path.join("experiments/results", f"{cfg.name}_*_{cfg.config_hash()}"))):
        if os.path.exists(os.path.join(d, "results.json")):
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="experiments/configs/jina_en_200k_k50_fuzzy.yaml")
    ap.add_argument("--budget", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/data/latent-basemap/r1_ablation/summary_mn4.json")
    ap.add_argument("--cells", default="all", help="comma tags to force-run, or 'all'")
    ap.add_argument("--reuse", action="store_true", help="reuse config-hash-matched completed cells")
    ap.add_argument("--required-free-gb", type=float, default=6.0)
    args = ap.parse_args()

    built = [(tag, scale, cfg) for (mn, ws) in CELLS
             for (tag, scale, cfg) in [build_cfg(args.base, mn, ws, args.seed, args.budget)]]
    want = None if args.cells == "all" else set(args.cells.split(","))

    print(f"[abl] {len(built)} cells, budget={args.budget}, seed={args.seed}, mn_on_scale={MN_ON_SCALE}", flush=True)
    for tag, scale, cfg in built:
        assert cfg.model.low_dim_kernel == "legacy_lp", cfg.name
        assert (cfg.train.mn_weight_scale == MN_ON_SCALE) == cfg.train.midnear_enabled, \
            f"{cfg.name}: dose {cfg.train.mn_weight_scale} inconsistent with midnear={cfg.train.midnear_enabled}"
        print(f"[abl] preflight {tag:14s} midnear={cfg.train.midnear_enabled} "
              f"mn_scale={cfg.train.mn_weight_scale} weighted={cfg.train.weighted_edge_sampling} "
              f"kernel={cfg.model.low_dim_kernel} hash={cfg.config_hash()}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary = {"budget": args.budget, "seed": args.seed, "base": args.base,
               "mn_on_scale": MN_ON_SCALE, "runs": {},
               "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def record(tag, scale, cfg, res, wall_s, run_dir, reused):
        p = extract_panel(res) or {}
        acct = (res.get("run_manifest") or {}).get("train_accounting", {})
        summary["runs"][tag] = {
            "name": cfg.name, "midnear": bool(cfg.train.midnear_enabled),
            "mn_weight_scale": float(cfg.train.mn_weight_scale),
            "weighted": bool(cfg.train.weighted_edge_sampling),
            "config_hash": cfg.config_hash(), "run_dir": run_dir, "reused": reused,
            "wall_s": round(wall_s, 1) if wall_s else None,
            "ffr": p.get("ffr"), "recall@k": p.get("recall@k"), "density": p.get("density"),
            "stop_reason": acct.get("stop_reason"),
            "positive_lr_updates": acct.get("positive_lr_optimizer_steps")}
        json.dump(summary, open(args.out, "w"), indent=1)

    # reuse pass (no GPU) — pull completed correctly-dosed cells from disk
    to_run = []
    for tag, scale, cfg in built:
        if want is not None and tag not in want:
            rd = _find_reusable(cfg)
            if args.reuse and rd:
                res = {"metrics_train": json.load(open(os.path.join(rd, "results.json"))).get("metrics_train", {}),
                       "run_manifest": json.load(open(os.path.join(rd, "results.json"))).get("run_manifest", {})}
                record(tag, scale, cfg, res, None, os.path.basename(rd), reused=True)
                print(f"[abl] reused {tag} <- {os.path.basename(rd)}", flush=True)
            continue
        rd = _find_reusable(cfg) if args.reuse else None
        if rd:
            res = {"metrics_train": json.load(open(os.path.join(rd, "results.json"))).get("metrics_train", {}),
                   "run_manifest": json.load(open(os.path.join(rd, "results.json"))).get("run_manifest", {})}
            record(tag, scale, cfg, res, None, os.path.basename(rd), reused=True)
            print(f"[abl] reused {tag} <- {os.path.basename(rd)}", flush=True)
        else:
            to_run.append((tag, scale, cfg))

    if to_run:
        allowed = known_service_pids()   # P0-5: allow-list known viewers by identity, not snapshot-all
        with GpuLease(timeout=0) as lease:
            check_co_tenants(args.required_free_gb, allowed_pids=allowed)
            for tag, scale, cfg in to_run:
                t0 = time.time()
                print(f"\n[abl] === train {cfg.name} (mn_scale={scale}) ===", flush=True)
                res = run_single_experiment(cfg)
                rd = _find_reusable(cfg)
                record(tag, scale, cfg, res, time.time() - t0, os.path.basename(rd) if rd else None, reused=False)
                r = summary["runs"][tag]
                print(f"[abl] {tag}: ffr={r['ffr']} density={r['density']} ({r['wall_s']}s)", flush=True)

    summary["effects"] = compute_effects(summary["runs"])
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(summary, open(args.out, "w"), indent=1)
    print("\n[abl] effects:", json.dumps(summary["effects"], indent=1), flush=True)


def compute_effects(runs, metrics=("ffr", "density", "recall@k")):
    """Marginal main effects AND the 2×2 interaction. REQUIRES all four cells with
    a non-null metric — a missing cell raises, never silently becomes 0 (P0-1)."""
    def cell(mn, ws):
        for r in runs.values():
            if r["midnear"] == mn and r["weighted"] == ws:
                return r
        raise ValueError(f"missing ablation cell midnear={mn} weighted={ws}; "
                         f"cannot compute effects (P0-1)")
    out = {}
    for m in metrics:
        c00, c01, c10, c11 = cell(False, False)[m], cell(False, True)[m], cell(True, False)[m], cell(True, True)[m]
        if any(v is None for v in (c00, c01, c10, c11)):
            raise ValueError(f"cell missing metric {m}; refuse to compute effects (P0-1)")
        out[m] = {
            "weighted_main": round(((c01 + c11) - (c00 + c10)) / 2, 4),
            "midnear_main": round(((c10 + c11) - (c00 + c01)) / 2, 4),
            "interaction": round((c11 - c10) - (c01 - c00), 4)}
    return out


if __name__ == "__main__":
    main()
