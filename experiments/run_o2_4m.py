"""O2 phase 1 — matched unanchored 4M controls via the fail-stop DAG.

device perf canary (4M device+weighted) -> train 3 unanchored 4M controls
(legacy_lp, seeds 42/43/44, recipe-matched, verdict-gated) -> transductive v2.2
score through one shared 4M reference. These are the stability/quality baseline the
sparse-hold frontier (phase 2, run_o2_frontier.py) is measured against.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.run_controller import Job, run_jobs, known_service_pids
from basemap.round0005_retirement import refuse_retired_launcher

PY = ".venv/bin/python"
W = "/data/latent-basemap/closure/o2"
TESTBED = "/data/latent-basemap/jina-en-4M-nested"
EVID = "experiments/evidence/r1_o2"


def main():
    refuse_retired_launcher("experiments/run_o2_4m.py")
    os.makedirs(W, exist_ok=True); os.makedirs(EVID, exist_ok=True)
    d = lambda p: os.path.join(W, p)
    verdict = d("perf_canary_verdict.json")
    cfg42 = "experiments/configs/_o2_4m_control_s42.yaml"
    jobs = [Job(
        name="perf_canary_4m", certifying=True, outputs=[verdict],
        argv=[PY, "experiments/run_canary.py", "--train-config", cfg42,
              "--run-dir", d("canary_run"), "--out", verdict,
              "--floor", "200", "--warn", "250", "--max-steps", "1200", "--warmup", "200"],
        done_marker=d("canary.done.json"), log=d("canary.log"), manifest=d("canary.manifest.json"),
        cwd=os.getcwd(), required_free_gb=18.0,
        input_paths=[cfg42, "experiments/run_canary.py", "experiments/run_experiment.py",
                     "basemap/pumap/parametric_umap/core.py"])]
    train_dirs = {}
    for seed in (42, 43, 44):
        cfg = f"experiments/configs/_o2_4m_control_s{seed}.yaml"
        rd = d(f"control_s{seed}")
        train_dirs[seed] = rd
        jobs.append(Job(
            name=f"control_s{seed}", certifying=True,
            outputs=[os.path.join(rd, "coords.parquet"), os.path.join(rd, "model.pt"),
                     os.path.join(rd, "results.json")],
            argv=[PY, "experiments/run_experiment.py", cfg,
                  "--override", f"logging.run_dir_override={rd}"],
            done_marker=d(f"control_s{seed}.done.json"), log=d(f"control_s{seed}.log"),
            manifest=d(f"control_s{seed}.manifest.json"), cwd=os.getcwd(), required_free_gb=18.0,
            deps=["perf_canary_4m"], canary_dep="perf_canary_4m", require_passing_verdict=verdict,
            predicted_wall_s=1750.0,
            input_paths=[cfg, "experiments/run_experiment.py",
                         "basemap/pumap/parametric_umap/core.py", verdict]))
    # transductive v2.2 score of the 3 controls (coord-only; 4M has no held-out
    # projection set). Reduced anchors keep 4M hi-D neighbour cost tractable.
    runs = [f"control_s{s}={os.path.join(train_dirs[s], 'coords.parquet')}" for s in (42, 43, 44)]
    out = f"{EVID}/complete_4m_controls_v22.json"
    jobs.append(Job(
        name="score_controls", certifying=True, outputs=[out],
        argv=[PY, "experiments/score_complete_panel.py", "--runs", *runs,
              "--testbed", TESTBED, "--no-model", "--n-anchors", "2000",
              "--reference", d("ref_4m.npz"), "--out", out],
        done_marker=d("score.done.json"), log=d("score.log"), manifest=d("score.manifest.json"),
        cwd=os.getcwd(), required_free_gb=18.0,
        deps=[f"control_s{s}" for s in (42, 43, 44)],
        input_paths=["experiments/score_complete_panel.py", "basemap/panel_v2.py"]))
    summary = run_jobs(jobs, allowed_pids=known_service_pids(), summary_path=d("o2_ctl.json"))
    print(json.dumps({j["name"]: j["status"] for j in summary["jobs"]}, indent=1))
    bad = [j for j in summary["jobs"] if j["status"] not in ("ok", "skipped_done")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
