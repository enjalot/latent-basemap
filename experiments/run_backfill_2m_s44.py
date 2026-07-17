"""Optional backfill (plan): the missing 2M seed-44 kernel replication, upgrading
the 2M kernel evidence from 2 seeds to the program's 3-seed standard. Runs through
the fail-stop controller DAG: one perf canary (2M device+weighted) gates three
seed-44 trainings (legacy_a1b1 / umap_a1b1 / umap_stdcurve, byte-faithful to their
s42 configs except seed+name), then all NINE maps (3 arms × seeds 42/43/44) are
rescored through ONE shared v2.2 reference.
"""
from __future__ import annotations
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.run_controller import Job, run_jobs, known_service_pids
from basemap.round0005_retirement import refuse_retired_launcher

PY = ".venv/bin/python"
W = "/data/latent-basemap/closure/bf_2m"
EVID = "experiments/evidence/r1_kernel"
SRC = "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train"
ARMS = ["legacy_a1b1", "umap_a1b1", "umap_stdcurve"]


def _one(pat):
    d = sorted(glob.glob(os.path.join("experiments/results", pat)))
    if not d:
        raise SystemExit(f"backfill: no run dir for {pat}")
    return d[-1]


def main():
    refuse_retired_launcher("experiments/run_backfill_2m_s44.py")
    os.makedirs(W, exist_ok=True)
    verdict = f"{W}/perf_canary_verdict.json"
    legacy_cfg = "experiments/configs/_bf_2m_legacy_a1b1_s44.yaml"
    jobs = [Job(
        name="perf_canary_2m", certifying=True, outputs=[verdict],
        argv=[PY, "experiments/run_canary.py", "--train-config", legacy_cfg,
              "--run-dir", f"{W}/canary_run", "--out", verdict,
              "--floor", "200", "--warn", "250", "--max-steps", "1200", "--warmup", "200"],
        done_marker=f"{W}/canary.done.json", log=f"{W}/canary.log",
        manifest=f"{W}/canary.manifest.json", cwd=os.getcwd(), required_free_gb=12.0,
        input_paths=[legacy_cfg, "experiments/run_canary.py", "experiments/run_experiment.py",
                     "basemap/pumap/parametric_umap/core.py"])]
    train_dirs = {}
    for arm in ARMS:
        cfg = f"experiments/configs/_bf_2m_{arm}_s44.yaml"
        rd = f"{W}/train_{arm}_s44"
        train_dirs[arm] = rd
        jobs.append(Job(
            name=f"train_{arm}_s44", certifying=True,
            outputs=[os.path.join(rd, "coords.parquet"), os.path.join(rd, "model.pt"),
                     os.path.join(rd, "results.json")],
            argv=[PY, "experiments/run_experiment.py", cfg,
                  "--override", f"logging.run_dir_override={rd}"],
            done_marker=f"{W}/train_{arm}_s44.done.json", log=f"{W}/train_{arm}_s44.log",
            manifest=f"{W}/train_{arm}_s44.manifest.json", cwd=os.getcwd(),
            required_free_gb=12.0, deps=["perf_canary_2m"], canary_dep="perf_canary_2m",
            require_passing_verdict=verdict, predicted_wall_s=1750.0,
            input_paths=[cfg, "experiments/run_experiment.py",
                         "basemap/pumap/parametric_umap/core.py", verdict]))
    # score all 9 (existing s42/s43 + new s44) through ONE shared reference
    runs = []
    for arm in ARMS:
        runs.append(f"{arm}_s42={_one(f'r1_kernel_2m_{arm}_s42_*')}")
        runs.append(f"{arm}_s43={_one(f'r1_kernel_2m_{arm}_s43_*')}")
        runs.append(f"{arm}_s44={train_dirs[arm]}")
    out9 = f"{EVID}/complete_2m_3seed_v22.json"
    jobs.append(Job(
        name="rescore_2m_3seed", certifying=True, outputs=[out9],
        argv=[PY, "experiments/score_complete_panel.py", "--runs", *runs,
              "--testbed", "/data/latent-basemap/jina-en-2m", "--source", SRC,
              "--reference", f"{W}/ref_2m.npz", "--out", out9],
        done_marker=f"{W}/rescore.done.json", log=f"{W}/rescore.log",
        manifest=f"{W}/rescore.manifest.json", cwd=os.getcwd(), required_free_gb=24.0,
        deps=[f"train_{a}_s44" for a in ARMS],
        input_paths=["experiments/score_complete_panel.py", "basemap/panel_v2.py"]))
    summary = run_jobs(jobs, allowed_pids=known_service_pids(), summary_path=f"{W}/bf_ctl.json")
    import json
    print(json.dumps({j["name"]: j["status"] for j in summary["jobs"]}, indent=1))
    bad = [j for j in summary["jobs"] if j["status"] not in ("ok", "skipped_done")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
