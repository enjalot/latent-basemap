"""P3 — 8M admission canary (closure review). Proves the RESIDENT weighted path
is admitted (not the legacy uniform fallback that trained the overnight bridge)
and meets a pre-registered throughput floor BEFORE any 500k-update run.

Exact bridge config: 8M mixed substrate, legacy kernel, h1024, 2D, nomn, weighted
sampling, pos_ratio 0.20, batch 8192, v3 schedule — with:
  - gpu_resident_vram_budget_gb raised to admit the ~21.97 GB resident need
    (X fp16 12.3 + edges 4.8 + weighted CDF 4.8), leaving model/activation headroom;
  - required_input_pipeline="device" (a legacy fallback RAISES, not warns);
  - canary_max_steps to stop after ~1.2k steps and time the steady-state window.

Runs in a FRESH controller child (no VRAM pollution). Aborts if the pipeline is
not device+weighted or the steady rate is below the floor.

Usage:
  python experiments/run_8m_canary.py --floor 200 --max-steps 1200
"""
from __future__ import annotations
import argparse, os, sys, json, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.experiment_config import load_config
from basemap.run_controller import run_jobs, Job, known_service_pids

BASE = "experiments/configs/jina_en_8m_nested.yaml"


def build_canary_config(max_steps, warmup, budget_gb):
    cfg = load_config(BASE, {
        "model.low_dim_kernel": "legacy_lp", "model.a": 1.0, "model.b": 1.0,
        "model.hidden_dim": 1024, "model.n_components": 2,
        "data.random_seed": 42,
        "train.total_steps_estimate": 500000, "train.lr_schedule": "cosine",
        "train.warmup_steps": 200, "train.midnear_enabled": False,
        "train.weighted_edge_sampling": True, "train.pos_ratio": 0.20,
        "train.batch_size": 8192, "train.require_graph_manifest": True,
        "train.require_full_budget": False,            # canary stops early by design
        "train.required_input_pipeline": "device",     # legacy fallback RAISES
        "train.gpu_resident_data": "auto",
        "train.gpu_resident_vram_budget_gb": float(budget_gb),
        "train.canary_max_steps": int(max_steps), "train.canary_warmup": int(warmup),
    })
    cfg.name = "r1_8m_canary"
    cfg.eval.metrics = []          # no scoring
    cfg.logging.save_model = False
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=200.0, help="steady upd/s abort floor")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--budget-gb", type=float, default=26.0)
    ap.add_argument("--out", default="/data/latent-basemap/closure/canary_8m.json")
    args = ap.parse_args()

    cfg = build_canary_config(args.max_steps, args.warmup, args.budget_gb)
    cfg_path = "experiments/configs/_canary_8m.yaml"
    cfg.to_yaml(cfg_path)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    job = Job(name="canary_8m",
              argv=[".venv/bin/python", "experiments/run_experiment.py", cfg_path],
              outputs=[], done_marker="/data/latent-basemap/closure/canary.done.json",
              log="/data/latent-basemap/closure/canary.log",
              manifest="/data/latent-basemap/closure/canary.manifest.json",
              cwd=os.getcwd(), required_free_gb=28.0,
              input_paths=[cfg_path, "experiments/run_experiment.py",
                           "basemap/pumap/parametric_umap/core.py"])
    print(f"[canary] launching (floor {args.floor} upd/s, {args.max_steps} steps, "
          f"budget {args.budget_gb} GB, required device+weighted)…", flush=True)
    summary = run_jobs([job], allowed_pids=known_service_pids(),
                       summary_path="/data/latent-basemap/closure/canary_ctl.json")
    rec = summary["jobs"][0]
    verdict = {"controller": summary.get("stop_reason"), "job_status": rec["status"], "floor": args.floor}
    rd = sorted(glob.glob("experiments/results/r1_8m_canary_*"))
    if rd and os.path.exists(os.path.join(rd[-1], "results.json")):
        c = json.load(open(os.path.join(rd[-1], "results.json"))).get("canary", {})
        pipe = c.get("pipeline", {})
        rate = c.get("steady_updates_per_s")
        passed = (rec["status"] == "ok" and pipe.get("pipeline") == "device"
                  and pipe.get("positive_sampling") == "weighted_with_replacement"
                  and rate is not None and rate >= args.floor)
        verdict.update({"passed": bool(passed), "steady_updates_per_s": rate,
                        "pipeline": pipe.get("pipeline"),
                        "positive_sampling": pipe.get("positive_sampling"),
                        "x_residency": pipe.get("x_residency"),
                        "bench_seconds": c.get("bench_seconds"), "run_dir": os.path.basename(rd[-1]),
                        "est_500k_minutes": round((500000 / rate) / 60, 1) if rate else None})
    else:
        verdict["passed"] = False; verdict["error"] = "no canary results (job likely raised on admission)"
    json.dump(verdict, open(args.out, "w"), indent=1)
    print(f"[canary] {'PASS' if verdict.get('passed') else 'FAIL'}: {json.dumps(verdict)}", flush=True)
    sys.exit(0 if verdict.get("passed") else 2)


if __name__ == "__main__":
    main()
