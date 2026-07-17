"""O2 phase 2 — sparse-hold frontier via the fail-stop DAG.

build_anchors (from the seed-42 control) -> perf canary (sparse path) -> train the
w in {2,10,50} sparse-hold arms (166,667 updates each, verdict-gated) -> score them
through the shared 4M reference -> select (overlap >= 0.5 AND KPIs >= 90% control).

Run AFTER run_o2_4m.py (needs control_s42/coords.parquet + ref_4m.npz).
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
WEIGHTS = [2, 10, 50]


def main():
    refuse_retired_launcher("experiments/run_o2_frontier.py")
    d = lambda p: os.path.join(W, p)
    control42 = d("control_s42")
    if not os.path.exists(os.path.join(control42, "coords.parquet")):
        raise SystemExit("run_o2_frontier: control_s42 not found — run run_o2_4m.py first")
    anchors = d("anchors_from_control_s42.npz")
    verdict = d("frontier_canary_verdict.json")
    ref = d("ref_4m.npz")

    jobs = [Job(
        name="build_anchors", certifying=True, outputs=[anchors],
        argv=[PY, "experiments/build_o2_anchors.py", "--control", control42,
              "--n-landmarks", "2000000", "--out", anchors],
        done_marker=d("anchors.done.json"), log=d("anchors.log"), manifest=d("anchors.manifest.json"),
        cwd=os.getcwd(), required_free_gb=0.0,
        input_paths=["experiments/build_o2_anchors.py", os.path.join(control42, "coords.parquet")])]

    cfg2 = "experiments/configs/_o2_4m_sparse_w2_s42.yaml"
    jobs.append(Job(
        name="frontier_canary", certifying=True, outputs=[verdict],
        argv=[PY, "experiments/run_canary.py", "--train-config", cfg2,
              "--run-dir", d("frontier_canary_run"), "--out", verdict,
              "--floor", "200", "--warn", "250", "--max-steps", "1200", "--warmup", "200"],
        done_marker=d("frontier_canary.done.json"), log=d("frontier_canary.log"),
        manifest=d("frontier_canary.manifest.json"), cwd=os.getcwd(), required_free_gb=18.0,
        deps=["build_anchors"],
        input_paths=[cfg2, "experiments/run_canary.py", "experiments/run_experiment.py",
                     "basemap/pumap/parametric_umap/core.py", anchors]))

    for w in WEIGHTS:
        cfg = f"experiments/configs/_o2_4m_sparse_w{w}_s42.yaml"
        rd = d(f"sparse_w{w}_s42")
        jobs.append(Job(
            name=f"sparse_w{w}", certifying=True,
            outputs=[os.path.join(rd, "coords.parquet"), os.path.join(rd, "model.pt"),
                     os.path.join(rd, "results.json")],
            argv=[PY, "experiments/run_experiment.py", cfg,
                  "--override", f"logging.run_dir_override={rd}"],
            done_marker=d(f"sparse_w{w}.done.json"), log=d(f"sparse_w{w}.log"),
            manifest=d(f"sparse_w{w}.manifest.json"), cwd=os.getcwd(), required_free_gb=18.0,
            deps=["frontier_canary", "build_anchors"], canary_dep="frontier_canary",
            require_passing_verdict=verdict, predicted_wall_s=650.0,
            input_paths=[cfg, "experiments/run_experiment.py",
                         "basemap/pumap/parametric_umap/core.py", verdict, anchors]))

    runs = [f"sparse_w{w}={os.path.join(d(f'sparse_w{w}_s42'), 'coords.parquet')}" for w in WEIGHTS]
    fscore = f"{EVID}/complete_4m_frontier_v22.json"
    jobs.append(Job(
        name="score_frontier", certifying=True, outputs=[fscore],
        argv=[PY, "experiments/score_complete_panel.py", "--runs", *runs,
              "--testbed", TESTBED, "--no-model", "--n-anchors", "2000",
              "--reference", ref, "--out", fscore],
        done_marker=d("score_frontier.done.json"), log=d("score_frontier.log"),
        manifest=d("score_frontier.manifest.json"), cwd=os.getcwd(), required_free_gb=18.0,
        deps=[f"sparse_w{w}" for w in WEIGHTS],
        input_paths=["experiments/score_complete_panel.py", "basemap/panel_v2.py", ref]))

    sel = f"{EVID}/o2_frontier_selection.json"
    jobs.append(Job(
        name="select", certifying=True, outputs=[sel],
        argv=[PY, "experiments/o2_frontier_select.py", "--control", control42,
              "--anchors", anchors, "--frontier-scores", fscore,
              "--control-scores", f"{EVID}/complete_4m_controls_v22.json",
              "--weights", ",".join(str(w) for w in WEIGHTS), "--out", sel],
        done_marker=d("select.done.json"), log=d("select.log"), manifest=d("select.manifest.json"),
        cwd=os.getcwd(), required_free_gb=18.0, deps=["score_frontier"],
        input_paths=["experiments/o2_frontier_select.py", fscore,
                     f"{EVID}/complete_4m_controls_v22.json", anchors]))

    summary = run_jobs(jobs, allowed_pids=known_service_pids(), summary_path=d("o2_frontier_ctl.json"))
    print(json.dumps({j["name"]: j["status"] for j in summary["jobs"]}, indent=1))
    bad = [j for j in summary["jobs"] if j["status"] not in ("ok", "skipped_done")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
