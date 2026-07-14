"""R1 kernel rebaseline launcher (plan-basemap-100m.md §R1, item 1).

Two pre-registered, SEPARATE questions at the 200k jina-en testbed, 3 seeds each:

  Q1  formula isolation — legacy_lp vs umap at IDENTICAL a=b=1.
      Isolates the kernel FORM (Lp `1/(1+a‖Δ‖_{2b})` vs UMAP `1/(1+a·r²^b)`)
      with everything else held fixed.
  Q2  standard UMAP curve — umap kernel at the fitted a≈1.5769, b≈0.8951.
      The kernel we would actually ship.

Config discipline (R1): the code default is `legacy_lp`, so a copied config
silently runs legacy. This launcher PRINTS + ASSERTS every run's kernel BEFORE
acquiring the GPU lease, and stamps kernel/a/b/budget/schedule into the summary.
All runs share one explicit active-update budget (the matched-ladder unit that
replaces the retracted "shared 500k" claim) and one GPU lease held for the batch.

Decisions come from the canonical panel (basemap.panel_v2) via the runner's
`panel_v2` eval metric: transductive ffr + separate recall@k + density + guards.

Usage:
  python experiments/run_r1_kernel.py --budget 500000 \
     --base experiments/configs/jina_en_200k_k50_fuzzy.yaml \
     --out /data/latent-basemap/r1_kernel/summary.json
"""
from __future__ import annotations
import argparse, os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.experiment_config import load_config
from experiments.run_experiment import run_single_experiment
from basemap.run_controller import GpuLease, check_co_tenants, gpu_snapshot

# fitted standard-UMAP curve (min_dist=0.1) — the shipped kernel for Q2.
STD_A, STD_B = 1.57694346, 0.895060879
SEEDS = (42, 43, 44)


def run_specs(seeds=SEEDS, tag=""):
    t = f"{tag}_" if tag else ""
    specs = []
    for s in seeds:                        # Q1 formula isolation @ a=b=1
        specs.append(dict(q="Q1_formula_isolation", kernel="legacy_lp", a=1.0, b=1.0, seed=s,
                          name=f"r1_kernel_{t}legacy_a1b1_s{s}"))
        specs.append(dict(q="Q1_formula_isolation", kernel="umap", a=1.0, b=1.0, seed=s,
                          name=f"r1_kernel_{t}umap_a1b1_s{s}"))
    for s in seeds:                        # Q2 standard UMAP curve
        specs.append(dict(q="Q2_std_umap_curve", kernel="umap", a=STD_A, b=STD_B, seed=s,
                          name=f"r1_kernel_{t}umap_stdcurve_s{s}"))
    return specs


def build_cfg(base, spec, budget):
    cfg = load_config(base, {
        "model.low_dim_kernel": spec["kernel"],
        "model.a": spec["a"], "model.b": spec["b"],
        "data.random_seed": spec["seed"],
        "train.total_steps_estimate": budget,
        "train.lr_schedule": "cosine",
        "train.warmup_steps": 200,
        "train.midnear_enabled": False,        # frozen recipe = nomn
        "train.weighted_edge_sampling": True,  # fuzzy weights
    })
    cfg.name = spec["name"]
    if "panel_v2" not in cfg.eval.metrics:
        cfg.eval.metrics = list(cfg.eval.metrics) + ["panel_v2"]
    return cfg


def extract_panel(results):
    m = results.get("metrics_test") or {}
    if "panel_v2" not in m:
        m = results.get("metrics_train") or {}
    return m.get("panel_v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="experiments/configs/jina_en_200k_k50_fuzzy.yaml")
    ap.add_argument("--budget", type=int, default=500000, help="active-update budget (LR horizon)")
    ap.add_argument("--out", default="/data/latent-basemap/r1_kernel/summary.json")
    ap.add_argument("--required-free-gb", type=float, default=6.0)
    ap.add_argument("--seeds", default="42,43,44", help="comma list of seeds")
    ap.add_argument("--tag", default="", help="run-name tag (e.g. 2m) to distinguish rungs")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    specs = run_specs(seeds=seeds, tag=args.tag)
    cfgs = [(s, build_cfg(args.base, s, args.budget)) for s in specs]

    # CONFIG DISCIPLINE: print + assert every kernel BEFORE the lease.
    print(f"[R1] {len(cfgs)} runs, budget={args.budget} updates, base={args.base}", flush=True)
    for spec, cfg in cfgs:
        got = cfg.model.low_dim_kernel
        assert got == spec["kernel"], f"{cfg.name}: kernel {got} != expected {spec['kernel']}"
        print(f"[R1] pre-lease OK  {cfg.name:32s} kernel={got:9s} a={cfg.model.a:.5f} "
              f"b={cfg.model.b:.5f} seed={cfg.data.random_seed}", flush=True)

    allowed = gpu_snapshot()["compute_pids"]   # tolerate the current viewer PID
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary = {"budget": args.budget, "base": args.base, "std_curve": [STD_A, STD_B],
               "seeds": list(seeds), "tag": args.tag, "runs": {},
               "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    with GpuLease(timeout=0) as lease:
        check_co_tenants(args.required_free_gb, allowed_pids=allowed)
        for i, (spec, cfg) in enumerate(cfgs):
            t0 = time.time()
            print(f"\n[R1] === run {i+1}/{len(cfgs)}: {cfg.name} ===", flush=True)
            results = run_single_experiment(cfg)
            panel = extract_panel(results)
            acct = (results.get("run_manifest") or {}).get("train_accounting", {})
            summary["runs"][cfg.name] = {
                "q": spec["q"], "kernel": spec["kernel"], "a": spec["a"], "b": spec["b"],
                "seed": spec["seed"], "wall_s": round(time.time() - t0, 1),
                "panel": panel, "train_accounting": acct,
                "throughput": (results.get("run_manifest") or {}).get("samples_per_sec")}
            json.dump(summary, open(args.out, "w"), indent=1)   # durable after EACH run
            p = panel or {}
            print(f"[R1] done {cfg.name}: ffr={p.get('ffr')} recall@k={p.get('recall@k')} "
                  f"density={p.get('density')} ({summary['runs'][cfg.name]['wall_s']}s)", flush=True)

    # 3-seed means + the two pre-registered comparisons
    def mean_over(pred, metric):
        vals = [r["panel"][metric] for r in summary["runs"].values()
                if pred(r) and r.get("panel") and r["panel"].get(metric) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    groups = {
        "Q1_legacy_a1b1": lambda r: r["q"] == "Q1_formula_isolation" and r["kernel"] == "legacy_lp",
        "Q1_umap_a1b1":   lambda r: r["q"] == "Q1_formula_isolation" and r["kernel"] == "umap",
        "Q2_umap_stdcurve": lambda r: r["q"] == "Q2_std_umap_curve",
    }
    summary["group_means"] = {g: {m: mean_over(pred, m) for m in ("ffr", "recall@k", "density")}
                              for g, pred in groups.items()}
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(summary, open(args.out, "w"), indent=1)
    print("\n[R1] group means:", json.dumps(summary["group_means"], indent=1), flush=True)
    print(f"[R1] summary -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
