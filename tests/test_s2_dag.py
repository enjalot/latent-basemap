"""L0.3 / S2 — the canonical G1 DAG must encode the fail-stop + verdict contract."""
import pytest
import experiments.dag_template as dag


def _g1(tmp_path):
    return dag.build_g1_dag(
        train_cfg="experiments/configs/_stdcurve_s42.yaml", work_dir=str(tmp_path),
        reference=str(tmp_path / "ref.npz"), legacy_run_dir=str(tmp_path / "legacy"),
        testbed_2m=str(tmp_path / "tb2m"), kernel_run_2m=str(tmp_path / "k2m"),
        decision_out=str(tmp_path / "dec.json"), gate_out=str(tmp_path / "gate.json"))


def test_g1_dag_is_valid(tmp_path):
    jobs = _g1(tmp_path)
    assert dag.validate_dag(jobs) is True
    by = {j.name: j for j in jobs}
    # the long train is verdict-gated + canary-dep'd, fail-stop, deterministic outputs
    t = by["train_stdcurve_s42"]
    assert t.canary_dep == "perf_canary" and "perf_canary" in t.deps
    assert t.require_passing_verdict and t.require_passing_verdict.endswith("perf_canary_verdict.json")
    assert t.predicted_wall_s > dag.CANARY_REQUIRED_WALL_S
    assert len(t.outputs) == 3 and all("*" not in o for o in t.outputs)
    assert all(j.continue_on_failure is False for j in jobs)
    # the fresh 2M evaluator canary precedes the perf canary + train
    assert by["scoring_regression"].deps == ["eval_canary_2m"]
    assert "scoring_regression" in by["perf_canary"].deps


def test_g1_dag_rejects_missing_verdict_gate(tmp_path):
    jobs = _g1(tmp_path)
    t = next(j for j in jobs if j.name == "train_stdcurve_s42")
    t.require_passing_verdict = None
    with pytest.raises(ValueError, match="require_passing_verdict"):
        dag.validate_dag(jobs)


def test_g1_dag_rejects_placeholder_path(tmp_path):
    jobs = _g1(tmp_path)
    next(j for j in jobs if j.name == "score_pair").outputs = ["experiments/results/*/coords.parquet"]
    with pytest.raises(ValueError, match="unresolved path token"):
        dag.validate_dag(jobs)


def test_g1_dag_rejects_continue_on_failure(tmp_path):
    jobs = _g1(tmp_path)
    jobs[0].continue_on_failure = True
    with pytest.raises(ValueError, match="continue_on_failure"):
        dag.validate_dag(jobs)
