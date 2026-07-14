"""R1 2×2 mid-near × weighted-sampling ablation (plan §R1) — un-confounds A3.

A3 flipped mid-near AND weighted sampling together, so its "mid-near trades local
for density" read was confounded. This isolates the two factors on the DECIDED
kernel (legacy_lp, per the R1 kernel decision), 200k, matched 500k-update budget:

  midnear ∈ {off (nomn), on (mn4, weight 4)}  ×  weighted_edge_sampling ∈ {false, true}

Config discipline: kernel asserted before the lease; one lease for the batch;
durable per-run summary. Transductive panel_v2 metrics here; run
score_complete_panel.py afterwards for purity/projection on the same 4 runs.

Usage:
  python experiments/run_r1_ablation.py --budget 500000 --seed 42 \
     --base experiments/configs/jina_en_200k_k50_fuzzy.yaml \
     --out /data/latent-basemap/r1_ablation/summary.json
"""
from __future__ import annotations
import argparse, os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.experiment_config import load_config
from experiments.run_experiment import run_single_experiment
from experiments.run_r1_kernel import extract_panel
from basemap.run_controller import GpuLease, check_co_tenants, gpu_snapshot

CELLS = [(mn, ws) for mn in (False, True) for ws in (False, True)]


def build_cfg(base, mn, ws, seed, budget):
    tag = f"mn{'1' if mn else '0'}_ws{'1' if ws else '0'}"
    cfg = load_config(base, {
        "model.low_dim_kernel": "legacy_lp",   # the decided kernel
        "model.a": 1.0, "model.b": 1.0,
        "data.random_seed": seed,
        "train.total_steps_estimate": budget,
        "train.lr_schedule": "cosine", "train.warmup_steps": 200,
        "train.midnear_enabled": mn,
        "train.weighted_edge_sampling": ws,
    })
    cfg.name = f"r1_abl_{tag}_s{seed}"
    if "panel_v2" not in cfg.eval.metrics:
        cfg.eval.metrics = list(cfg.eval.metrics) + ["panel_v2"]
    return tag, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="experiments/configs/jina_en_200k_k50_fuzzy.yaml")
    ap.add_argument("--budget", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/data/latent-basemap/r1_ablation/summary.json")
    ap.add_argument("--required-free-gb", type=float, default=6.0)
    args = ap.parse_args()

    built = [build_cfg(args.base, mn, ws, args.seed, args.budget) for mn, ws in CELLS]
    print(f"[abl] {len(built)} cells, budget={args.budget}, seed={args.seed}", flush=True)
    for tag, cfg in built:
        assert cfg.model.low_dim_kernel == "legacy_lp", cfg.name
        print(f"[abl] pre-lease OK {cfg.name:20s} midnear={cfg.train.midnear_enabled} "
              f"weighted={cfg.train.weighted_edge_sampling} kernel={cfg.model.low_dim_kernel}", flush=True)

    allowed = gpu_snapshot()["compute_pids"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary = {"budget": args.budget, "seed": args.seed, "base": args.base, "runs": {},
               "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with GpuLease(timeout=0) as lease:
        check_co_tenants(args.required_free_gb, allowed_pids=allowed)
        for tag, cfg in built:
            t0 = time.time()
            print(f"\n[abl] === {cfg.name} ===", flush=True)
            res = run_single_experiment(cfg)
            p = extract_panel(res) or {}
            acct = (res.get("run_manifest") or {}).get("train_accounting", {})
            summary["runs"][tag] = {
                "name": cfg.name, "midnear": cfg.train.midnear_enabled,
                "weighted": cfg.train.weighted_edge_sampling, "wall_s": round(time.time() - t0, 1),
                "ffr": p.get("ffr"), "recall@k": p.get("recall@k"), "density": p.get("density"),
                "run_dir": None, "stop_reason": acct.get("stop_reason"),
                "positive_lr_updates": acct.get("positive_lr_optimizer_steps")}
            json.dump(summary, open(args.out, "w"), indent=1)
            print(f"[abl] {tag}: ffr={p.get('ffr')} density={p.get('density')} "
                  f"({summary['runs'][tag]['wall_s']}s)", flush=True)
    # main effects
    def mean(pred, m):
        v = [r[m] for r in summary["runs"].values() if pred(r) and r.get(m) is not None]
        return round(float(np.mean(v)), 4) if v else None
    summary["main_effects"] = {
        "midnear_on_minus_off": {m: round((mean(lambda r: r["midnear"], m) or 0) -
                                          (mean(lambda r: not r["midnear"], m) or 0), 4)
                                 for m in ("ffr", "density")},
        "weighted_on_minus_off": {m: round((mean(lambda r: r["weighted"], m) or 0) -
                                           (mean(lambda r: not r["weighted"], m) or 0), 4)
                                  for m in ("ffr", "density")}}
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(summary, open(args.out, "w"), indent=1)
    print("\n[abl] main effects:", json.dumps(summary["main_effects"]), flush=True)


if __name__ == "__main__":
    main()
