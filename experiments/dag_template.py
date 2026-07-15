"""S1.3 / S2.3 — the ONE canonical subprocess DAG template.

Encodes the required dependency contract for a trained-map extension so no future
run can skip a safeguard:

    perf_canary ─┬─▶ train_s42 ─┐
                 └─▶ train_s43 ─┤
                                ├─▶ build_reference ─▶ score ─┬─▶ scoring_regression
                                                              └─▶ gate

Rules baked in (enforced by `basemap.run_controller.run_jobs`):
  • Every training node is certifying, declares its coords/model/results outputs,
    predicts a >10-minute wall, and therefore MUST carry `canary_dep=perf_canary`
    with `perf_canary` in its `deps` (S2 mandatory-canary rule). A sub-floor canary
    exits non-zero → blocks all trains.
  • `build_reference` produces the single content-addressed high-D reference; the
    scorer reuses it (fail-closed on drift).
  • `score` feeds BOTH the numeric gate and the 2M scoring-regression gate.
  • Each GPU node routes through the lease guard; the whole DAG runs in ONE
    controller invocation (no same-process multi-GPU-phase launcher).

`--dry-run` validates the DAG (edges + the mandatory-canary contract) WITHOUT
launching anything, so the template is testable off-GPU.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.run_controller import Job, run_jobs, known_service_pids, CANARY_REQUIRED_WALL_S

PY = ".venv/bin/python"
CLO = "/data/latent-basemap/closure"


def build_dag(seeds=(42, 43), *, canary_cfg, train_cfgs, reference_out, scores_out,
              gate_out, regression_out, predicted_train_s=1680.0):
    """Return the ordered Job list. `train_cfgs` maps seed -> (config_path, run_dir,
    coords_out, model_out, results_out)."""
    jobs = []
    canary = Job(
        name="perf_canary", certifying=False, outputs=[],
        argv=[PY, "experiments/run_experiment.py", canary_cfg],
        done_marker=f"{CLO}/dag_canary.done.json", log=f"{CLO}/dag_canary.log",
        manifest=f"{CLO}/dag_canary.manifest.json", cwd=os.getcwd(),
        required_free_gb=28.0, input_paths=[canary_cfg, "experiments/run_experiment.py",
                                            "basemap/pumap/parametric_umap/core.py"])
    jobs.append(canary)
    train_names = []
    for s in seeds:
        cfg, rundir, coords, model, results = train_cfgs[s]
        t = Job(name=f"train_s{s}", certifying=True,
                outputs=[coords, model, results],
                argv=[PY, "experiments/run_experiment.py", cfg],
                done_marker=f"{CLO}/dag_train_s{s}.done.json", log=f"{CLO}/dag_train_s{s}.log",
                manifest=f"{CLO}/dag_train_s{s}.manifest.json", cwd=os.getcwd(),
                required_free_gb=28.0, deps=["perf_canary"], canary_dep="perf_canary",
                predicted_wall_s=predicted_train_s,
                input_paths=[cfg, "experiments/run_experiment.py",
                             "basemap/pumap/parametric_umap/core.py"])
        jobs.append(t); train_names.append(t.name)
    ref = Job(name="build_reference", certifying=True, outputs=[reference_out],
              argv=[PY, "experiments/build_reference.py", "--out", reference_out],
              done_marker=f"{CLO}/dag_reference.done.json", log=f"{CLO}/dag_reference.log",
              manifest=f"{CLO}/dag_reference.manifest.json", cwd=os.getcwd(),
              required_free_gb=24.0, deps=train_names,
              input_paths=["experiments/build_reference.py", "basemap/panel_v2.py"])
    score = Job(name="score", certifying=True, outputs=[scores_out],
                argv=[PY, "experiments/score_8m_bridge.py", "--out", scores_out],
                done_marker=f"{CLO}/dag_score.done.json", log=f"{CLO}/dag_score.log",
                manifest=f"{CLO}/dag_score.manifest.json", cwd=os.getcwd(),
                required_free_gb=24.0, deps=["build_reference"] + train_names,
                input_paths=["experiments/score_8m_bridge.py", "basemap/panel_v2.py", reference_out])
    regression = Job(name="scoring_regression", certifying=True, outputs=[regression_out],
                     argv=[PY, "experiments/scoring_regression_gate.py", "--out", regression_out],
                     done_marker=f"{CLO}/dag_regression.done.json", log=f"{CLO}/dag_regression.log",
                     manifest=f"{CLO}/dag_regression.manifest.json", cwd=os.getcwd(),
                     required_free_gb=0.0, deps=["score"],
                     input_paths=["experiments/scoring_regression_gate.py"])
    gate = Job(name="gate", certifying=True, outputs=[gate_out],
               argv=[PY, "experiments/gate_summary.py", "--out", gate_out],
               done_marker=f"{CLO}/dag_gate.done.json", log=f"{CLO}/dag_gate.log",
               manifest=f"{CLO}/dag_gate.manifest.json", cwd=os.getcwd(),
               required_free_gb=0.0, deps=["score"],
               input_paths=["experiments/gate_summary.py"])
    return jobs + [ref, score, regression, gate]


def validate_dag(jobs):
    """Static contract check — raises on a violated edge/rule (S2 mandatory canary,
    declared outputs, dependency closure)."""
    names = {j.name for j in jobs}
    errs = []
    for j in jobs:
        for d in (j.deps or []):
            if d not in names:
                errs.append(f"{j.name}: dep '{d}' not in DAG")
        if j.certifying and not j.outputs:
            errs.append(f"{j.name}: certifying job declares no outputs")
        if j.certifying and j.predicted_wall_s > CANARY_REQUIRED_WALL_S:
            if not j.canary_dep or j.canary_dep not in (j.deps or []):
                errs.append(f"{j.name}: long run ({j.predicted_wall_s:.0f}s) lacks a canary_dep in deps")
    if errs:
        raise ValueError("DAG contract violations:\n  " + "\n  ".join(errs))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="validate + print the DAG, do not launch")
    ap.add_argument("--seeds", default="42,43")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    # Placeholder paths for the template shape; real launches fill these in.
    train_cfgs = {s: (f"experiments/configs/_bridge_w_s{s}.yaml",
                      f"experiments/results/train_s{s}",
                      f"{CLO}/train_s{s}/coords.parquet", f"{CLO}/train_s{s}/model.pt",
                      f"{CLO}/train_s{s}/results.json") for s in seeds}
    jobs = build_dag(seeds, canary_cfg="experiments/configs/_canary_8m.yaml",
                     train_cfgs=train_cfgs, reference_out=f"{CLO}/hiD_reference_8m.npz",
                     scores_out="experiments/evidence/r1_8m/bridge_weighted.json",
                     gate_out="experiments/evidence/r0_1_gate_summary.json",
                     regression_out="experiments/evidence/r1_rescore/scoring_regression.json")
    validate_dag(jobs)
    print("DAG (validated):")
    for j in jobs:
        dep = f" deps={j.deps}" if j.deps else ""
        cn = f" canary_dep={j.canary_dep}" if j.canary_dep else ""
        wall = f" wall~{j.predicted_wall_s:.0f}s" if j.predicted_wall_s else ""
        print(f"  {j.name:20s} certifying={j.certifying} outputs={len(j.outputs)}{dep}{cn}{wall}")
    if args.dry_run:
        print("\n--dry-run: contract valid; not launching.")
        return
    run_jobs(jobs, allowed_pids=known_service_pids(), summary_path=f"{CLO}/dag_ctl.json")


if __name__ == "__main__":
    main()
