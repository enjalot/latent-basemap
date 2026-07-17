"""L0.3 — the ONE runnable, job-specific, fail-stop science DAG.

Unlike the retired placeholder version, this builds a DAG whose every input/output
resolves to the exact path the invoked command reads/writes (no placeholder run
dirs, no latest-run globs), and whose canary→train edge is bound by CONTENT (a
passing verdict JSON), not by a job merely being named `perf_canary`.

G1 standard-curve chain (canary→train→reference→score→{regression,gate}):

    eval_canary_2m ─▶ (regression gate, before any 8M scoring)
    perf_canary ────▶ train ─┐
    build_reference ─────────┼─▶ score ─┬─▶ scoring_regression
                             └───────────┴─▶ decision

Contract (enforced by basemap.run_controller.run_jobs):
  • every job is certifying with declared, deterministic outputs;
  • continue_on_failure is FALSE for all;
  • the train job predicts >10 min, so it MUST carry canary_dep + a
    require_passing_verdict pointing at the perf-canary verdict — a `touch` job
    named perf_canary cannot release it;
  • the score/decision reuse the ONE shared reference.

`--dry-run` validates the contract and resolves every path WITHOUT launching.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.run_controller import (Job, run_jobs, known_service_pids,
                                     CANARY_REQUIRED_WALL_S)
from basemap.round0005_retirement import refuse_retired_launcher

PY = ".venv/bin/python"


def build_g1_dag(*, train_cfg, work_dir, reference, legacy_run_dir, testbed_2m,
                 kernel_run_2m, decision_out, gate_out, predicted_train_s=1750.0,
                 required_free_gb=30.0):
    """A fully-resolved G1 DAG. `train_cfg` is the tracked stdcurve train YAML;
    `legacy_run_dir` is the accepted seed-42 legacy bridge run dir (scored as the
    paired control); `kernel_run_2m` is a persisted 2M map for the fresh evaluator
    canary."""
    refuse_retired_launcher("experiments/dag_template.py")
    os.makedirs(work_dir, exist_ok=True)
    train_run_dir = os.path.join(work_dir, "train_stdcurve_s42")
    canary_run_dir = os.path.join(work_dir, "perf_canary_run")
    canary_verdict = os.path.join(work_dir, "perf_canary_verdict.json")
    eval_canary_out = os.path.join(work_dir, "eval_canary_2m.json")
    regression_out = os.path.join(work_dir, "scoring_regression.json")
    scores_out = os.path.join(work_dir, "stdcurve_pair_scores.json")
    d = lambda p: os.path.join(work_dir, p)

    jobs = []

    # 1) fresh current-commit 2M evaluator canary → wall/peak verdict, BEFORE 8M scoring.
    # --no-model → the run value is a coords.parquet PATH (not a run dir).
    eval_canary = Job(
        name="eval_canary_2m", certifying=True, outputs=[eval_canary_out],
        argv=[PY, "experiments/score_complete_panel.py",
              "--runs", f"legacy2m={os.path.join(kernel_run_2m, 'coords.parquet')}",
              "--testbed", testbed_2m,
              "--no-model", "--reference", d("ref_2m.npz"), "--out", eval_canary_out],
        done_marker=d("eval_canary_2m.done.json"), log=d("eval_canary_2m.log"),
        manifest=d("eval_canary_2m.manifest.json"), cwd=os.getcwd(),
        required_free_gb=required_free_gb,
        input_paths=["experiments/score_complete_panel.py", "basemap/panel_v2.py"])
    jobs.append(eval_canary)

    regression = Job(
        name="scoring_regression", certifying=True, outputs=[regression_out],
        argv=[PY, "experiments/scoring_regression_gate.py",
              "--golden", eval_canary_out, "--out", regression_out],
        done_marker=d("scoring_regression.done.json"), log=d("scoring_regression.log"),
        manifest=d("scoring_regression.manifest.json"), cwd=os.getcwd(),
        required_free_gb=0.0, deps=["eval_canary_2m"],
        input_paths=["experiments/scoring_regression_gate.py", eval_canary_out])
    jobs.append(regression)

    # 2) perf canary derived from the EXACT train config → verdict JSON.
    canary = Job(
        name="perf_canary", certifying=True, outputs=[canary_verdict],
        argv=[PY, "experiments/run_canary.py", "--train-config", train_cfg,
              "--run-dir", canary_run_dir, "--out", canary_verdict,
              "--floor", "200", "--warn", "250", "--max-steps", "1200", "--warmup", "200"],
        done_marker=d("perf_canary.done.json"), log=d("perf_canary.log"),
        manifest=d("perf_canary.manifest.json"), cwd=os.getcwd(),
        required_free_gb=required_free_gb, deps=["scoring_regression"],
        input_paths=[train_cfg, "experiments/run_canary.py", "experiments/run_experiment.py",
                     "basemap/pumap/parametric_umap/core.py"])
    jobs.append(canary)

    # 3) build/verify the shared 8M reference.
    ref = Job(
        name="build_reference", certifying=True, outputs=[reference],
        argv=[PY, "experiments/build_reference.py", "--out", reference],
        done_marker=d("build_reference.done.json"), log=d("build_reference.log"),
        manifest=d("build_reference.manifest.json"), cwd=os.getcwd(),
        required_free_gb=required_free_gb, deps=["perf_canary"],
        input_paths=["experiments/build_reference.py", "basemap/panel_v2.py"])
    jobs.append(ref)

    # 4) the 500k train — verdict-gated, deterministic run dir + declared outputs.
    train = Job(
        name="train_stdcurve_s42", certifying=True,
        outputs=[os.path.join(train_run_dir, "coords.parquet"),
                 os.path.join(train_run_dir, "model.pt"),
                 os.path.join(train_run_dir, "results.json")],
        argv=[PY, "experiments/run_experiment.py", train_cfg,
              "--override", f"logging.run_dir_override={train_run_dir}"],
        done_marker=d("train_stdcurve_s42.done.json"), log=d("train_stdcurve_s42.log"),
        manifest=d("train_stdcurve_s42.manifest.json"), cwd=os.getcwd(),
        required_free_gb=required_free_gb, deps=["perf_canary", "build_reference"],
        canary_dep="perf_canary", require_passing_verdict=canary_verdict,
        predicted_wall_s=predicted_train_s,
        input_paths=[train_cfg, "experiments/run_experiment.py",
                     "basemap/pumap/parametric_umap/core.py", canary_verdict])
    jobs.append(train)

    # 5) score the new stdcurve map + the paired legacy map through the SAME reference.
    score = Job(
        name="score_pair", certifying=True, outputs=[scores_out],
        argv=[PY, "experiments/score_8m_bridge.py", "--reference", reference,
              "--runs", f"stdcurve={train_run_dir}", f"legacy={legacy_run_dir}",
              "--out", scores_out],
        done_marker=d("score_pair.done.json"), log=d("score_pair.log"),
        manifest=d("score_pair.manifest.json"), cwd=os.getcwd(),
        required_free_gb=required_free_gb, deps=["train_stdcurve_s42", "build_reference"],
        input_paths=["experiments/score_8m_bridge.py", "basemap/panel_v2.py", reference])
    jobs.append(score)

    # 6) CPU decision node (paired deltas + closeness bands) — content-bound.
    decision = Job(
        name="decision", certifying=True, outputs=[decision_out],
        argv=[PY, "experiments/stdcurve_decision.py", "--scores", scores_out,
              "--out", decision_out],
        done_marker=d("decision.done.json"), log=d("decision.log"),
        manifest=d("decision.manifest.json"), cwd=os.getcwd(),
        required_free_gb=0.0, deps=["score_pair"],
        input_paths=["experiments/stdcurve_decision.py", scores_out])
    jobs.append(decision)
    return jobs


def validate_dag(jobs, *, require_paths=False):
    names = {j.name for j in jobs}
    errs = []
    for j in jobs:
        if j.continue_on_failure:
            errs.append(f"{j.name}: continue_on_failure must be False (fail-stop DAG)")
        for dep in (j.deps or []):
            if dep not in names:
                errs.append(f"{j.name}: dep '{dep}' not in DAG")
        if j.certifying and not j.outputs:
            errs.append(f"{j.name}: certifying job declares no outputs")
        if j.certifying and j.predicted_wall_s > CANARY_REQUIRED_WALL_S:
            if not j.canary_dep or j.canary_dep not in (j.deps or []):
                errs.append(f"{j.name}: long run lacks a canary_dep in deps")
            if not j.require_passing_verdict:
                errs.append(f"{j.name}: long run lacks require_passing_verdict (content gate)")
        # every output/input must be a concrete resolved path (no placeholders/globs)
        for p in list(j.outputs) + list(j.input_paths or []):
            if any(tok in p for tok in ("*", "<", ">", "PLACEHOLDER", "latest")):
                errs.append(f"{j.name}: unresolved path token in '{p}'")
        if require_paths:
            for p in (j.input_paths or []):
                # code/config inputs must exist at plan time; run-produced inputs may not
                if p.endswith((".py", ".yaml")) and not os.path.exists(p):
                    errs.append(f"{j.name}: declared code/config input '{p}' does not exist")
    if errs:
        raise ValueError("DAG contract violations:\n  " + "\n  ".join(errs))
    return True


def main():
    refuse_retired_launcher("experiments/dag_template.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--train-config", default="experiments/configs/_stdcurve_s42.yaml")
    ap.add_argument("--work-dir", default="/data/latent-basemap/closure/g1")
    ap.add_argument("--reference", default="/data/latent-basemap/closure/hiD_reference_8m.npz")
    ap.add_argument("--legacy-run-dir", required=False, default="")
    ap.add_argument("--testbed-2m", default="/data/latent-basemap/jina-en-2m")
    ap.add_argument("--kernel-run-2m", default="")
    args = ap.parse_args()
    jobs = build_g1_dag(
        train_cfg=args.train_config, work_dir=args.work_dir, reference=args.reference,
        legacy_run_dir=args.legacy_run_dir or "/data/latent-basemap/closure/legacy_s42",
        testbed_2m=args.testbed_2m,
        kernel_run_2m=args.kernel_run_2m or "/data/latent-basemap/closure/kernel_2m_s42",
        decision_out=os.path.join(args.work_dir, "stdcurve_decision.json"),
        gate_out="experiments/evidence/r0_1_gate_summary.json")
    validate_dag(jobs, require_paths=args.dry_run)
    print("G1 DAG (validated):")
    for j in jobs:
        dep = f" deps={j.deps}" if j.deps else ""
        cn = f" canary_dep={j.canary_dep}" if j.canary_dep else ""
        vg = " verdict-gated" if j.require_passing_verdict else ""
        wall = f" wall~{j.predicted_wall_s:.0f}s" if j.predicted_wall_s else ""
        print(f"  {j.name:20s} out={len(j.outputs)}{dep}{cn}{vg}{wall}")
    if args.dry_run:
        print("\n--dry-run: contract valid; not launching.")
        return
    run_jobs(jobs, allowed_pids=known_service_pids(),
             summary_path=os.path.join(args.work_dir, "dag_ctl.json"))


if __name__ == "__main__":
    main()
