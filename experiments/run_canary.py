"""L0.1 — verdict-bearing performance canary.

The canary is the admission contract for a long run, so its PASS must be earned,
not inferred from a bare exit 0 of the raw experiment runner. This wrapper:

  1. derives the canary config from the EXACT train config, changing only the
     registered canary differences (canary steps/warmup/floor/warn, no budget
     requirement, no model/coords). It never overwrites a tracked YAML — the
     derived config is written to a scratch path;
  2. runs `run_experiment.py` as a child with a DETERMINISTIC run dir;
  3. INDEPENDENTLY re-judges from the child's persisted `results.json` (aborted,
     missing rate, below floor, pipeline/sampler mismatch, config-family
     mismatch), writes a stable verdict JSON, and exits nonzero on any failure.

Usage (under a held/inherited GPU lease — the controller supplies it):
  python experiments/run_canary.py --train-config <cfg.yaml> \
      --run-dir /data/latent-basemap/closure/<name>_canary_run \
      --out /data/latent-basemap/closure/<name>_canary_verdict.json \
      --floor 200 --warn 250 --max-steps 1200 --warmup 200
"""
from __future__ import annotations
import argparse, os, sys, json, shutil, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.experiment_config import load_config

# Config keys that MUST be identical between the canary and its train config —
# the "config family". A mismatch means the canary did not exercise the real run.
FAMILY = [
    ("model", "low_dim_kernel"), ("model", "a"), ("model", "b"),
    ("model", "hidden_dim"), ("model", "n_components"), ("model", "n_layers"),
    ("train", "batch_size"), ("train", "pos_ratio"), ("train", "weighted_edge_sampling"),
    ("train", "required_input_pipeline"), ("train", "gpu_resident_data"),
    ("train", "gpu_resident_vram_budget_gb"), ("train", "midnear_enabled"),
    ("data", "precomputed_edges_path"), ("data", "random_seed"),
]


def _family(cfg):
    out = {}
    for sect, key in FAMILY:
        out[f"{sect}.{key}"] = getattr(getattr(cfg, sect), key, None)
    return out


def derive_canary_config(train_cfg_path, run_dir, max_steps, warmup, floor, warn):
    train = load_config(train_cfg_path, {})
    canary = load_config(train_cfg_path, {
        # the ONLY registered canary differences:
        "train.canary_max_steps": int(max_steps), "train.canary_warmup": int(warmup),
        "train.canary_floor": float(floor), "train.canary_warn_rate": float(warn),
        "train.require_full_budget": False,
        "logging.save_model": False, "logging.run_dir_override": run_dir,
    })
    canary.name = f"{train.name}_canary"
    canary.eval.metrics = []
    # config-family must be byte-identical to the train config (proves the canary
    # exercises the same shape/arch/pipeline/graph as the real run).
    tf, cf = _family(train), _family(canary)
    if tf != cf:
        diff = {k: (tf[k], cf[k]) for k in tf if tf[k] != cf[k]}
        raise SystemExit(f"canary config-family diverged from train config: {diff}")
    return canary, tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--run-dir", required=True, help="deterministic child run dir")
    ap.add_argument("--out", required=True, help="stable verdict JSON (declared output)")
    ap.add_argument("--scratch-config", default="", help="where to write the derived canary yaml")
    ap.add_argument("--floor", type=float, default=200.0)
    ap.add_argument("--warn", type=float, default=250.0)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args()

    cfg, family = derive_canary_config(args.train_config, args.run_dir,
                                       args.max_steps, args.warmup, args.floor, args.warn)
    scratch_cfg = args.scratch_config or os.path.join(
        os.path.dirname(args.out) or ".", f"{cfg.name}_derived.yaml")
    os.makedirs(os.path.dirname(scratch_cfg) or ".", exist_ok=True)
    # never overwrite a tracked config: refuse if the target is under experiments/configs
    if os.path.abspath(scratch_cfg).startswith(os.path.abspath("experiments/configs")):
        raise SystemExit("refuse to write a derived canary config into experiments/configs "
                         "(tracked); use a scratch path (L0.1).")
    cfg.to_yaml(scratch_cfg)
    if os.path.isdir(args.run_dir) and os.listdir(args.run_dir):
        shutil.rmtree(args.run_dir)      # fresh deterministic dir for this canary

    # Run the raw experiment runner as a child. Its exit code is NOT trusted for
    # success (we re-judge below); we only surface a hard crash. The inherited GPU
    # lease fd (BASEMAP_GPU_LEASE_FD from the controller) must be kept open across
    # the exec so the child's require_active_lease() passes.
    env = dict(os.environ, BASEMAP_RUN_DIR=args.run_dir)
    pass_fds = ()
    lease_fd = os.environ.get("BASEMAP_GPU_LEASE_FD")
    if lease_fd and lease_fd.isdigit():
        pass_fds = (int(lease_fd),)
    proc = subprocess.run([sys.executable, "experiments/run_experiment.py", scratch_cfg],
                          env=env, close_fds=True, pass_fds=pass_fds)
    child_rc = proc.returncode

    rj = os.path.join(args.run_dir, "results.json")
    verdict = {"gate": "perf_canary", "train_config": os.path.abspath(args.train_config),
               "run_dir": args.run_dir, "floor": args.floor, "child_rc": child_rc,
               "config_family": family}
    reasons = []
    c = (json.load(open(rj)).get("canary") if os.path.exists(rj) else None)
    if not c:
        reasons.append("no canary block in results.json (child likely raised on admission)")
    else:
        pipe = c.get("pipeline") or {}
        rate = c.get("steady_updates_per_s")
        bkey = c.get("baseline_key") or {}
        if c.get("aborted"):
            reasons.append("aborted (sub-floor windows)")
        if rate is None:
            reasons.append("missing steady rate")
        elif rate < args.floor:
            reasons.append(f"rate {rate} < floor {args.floor}")
        # pipeline/sampler must match the train config's intent
        want_pipe = family.get("train.required_input_pipeline")
        if want_pipe and want_pipe not in ("any",) and pipe.get("pipeline") != want_pipe:
            reasons.append(f"pipeline {pipe.get('pipeline')} != required {want_pipe}")
        if family.get("train.weighted_edge_sampling") and \
                pipe.get("positive_sampling") != "weighted_with_replacement":
            reasons.append(f"sampler {pipe.get('positive_sampling')} != weighted_with_replacement")
        # baseline/config-family cross-check (shape/arch/pipeline as measured)
        for k, fk in [("kernel", "model.low_dim_kernel"), ("hidden_dim", "model.hidden_dim"),
                      ("batch_size", "train.batch_size"), ("pipeline", "train.required_input_pipeline")]:
            if fk in family and bkey.get(k) is not None and family[fk] not in (None, "any") \
                    and bkey.get(k) != family[fk]:
                reasons.append(f"baseline {k}={bkey.get(k)} != config-family {family[fk]}")
        verdict.update({"steady_updates_per_s": rate, "aborted": bool(c.get("aborted")),
                        "pipeline": pipe.get("pipeline"),
                        "positive_sampling": pipe.get("positive_sampling"),
                        "rate_median": (c.get("profile") or {}).get("rate_median"),
                        "dominant_phase": (c.get("profile") or {}).get("dominant_phase"),
                        "baseline_key": bkey})
    verdict["passed"] = (len(reasons) == 0)
    verdict["reasons"] = reasons
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = f"{args.out}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(verdict, f, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, args.out)
    print(f"[canary] {'PASS' if verdict['passed'] else 'FAIL'} -> {args.out}: {reasons}", flush=True)
    sys.exit(0 if verdict["passed"] else 2)


if __name__ == "__main__":
    main()
