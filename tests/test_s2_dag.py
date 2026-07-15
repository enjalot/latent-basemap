"""S2/S1.3 — the canonical DAG template must encode the safeguard contract."""
import pytest
import experiments.dag_template as dag


def _cfgs(seeds):
    return {s: (f"cfg_s{s}.yaml", f"run_s{s}", f"c_s{s}.parquet",
                f"m_s{s}.pt", f"r_s{s}.json") for s in seeds}


def test_template_dag_is_valid():
    jobs = dag.build_dag((42, 43), canary_cfg="canary.yaml", train_cfgs=_cfgs((42, 43)),
                         reference_out="ref.npz", scores_out="scores.json",
                         gate_out="gate.json", regression_out="reg.json")
    assert dag.validate_dag(jobs) is True
    names = [j.name for j in jobs]
    assert names[0] == "perf_canary"
    # every train depends on the canary and declares it as canary_dep
    for j in jobs:
        if j.name.startswith("train_s"):
            assert "perf_canary" in j.deps and j.canary_dep == "perf_canary"
            assert j.predicted_wall_s > dag.CANARY_REQUIRED_WALL_S
            assert len(j.outputs) == 3
    # gate + regression both hang off score
    by = {j.name: j for j in jobs}
    assert by["gate"].deps == ["score"]
    assert by["scoring_regression"].deps == ["score"]
    assert set(by["score"].deps) >= {"build_reference", "train_s42", "train_s43"}


def test_validate_rejects_long_run_without_canary():
    jobs = dag.build_dag((42,), canary_cfg="c.yaml", train_cfgs=_cfgs((42,)),
                         reference_out="ref.npz", scores_out="s.json",
                         gate_out="g.json", regression_out="r.json")
    # strip the canary safeguard off the training node -> contract must fail
    train = next(j for j in jobs if j.name == "train_s42")
    train.canary_dep = None
    with pytest.raises(ValueError, match="lacks a canary_dep"):
        dag.validate_dag(jobs)


def test_validate_rejects_dangling_dep():
    jobs = dag.build_dag((42,), canary_cfg="c.yaml", train_cfgs=_cfgs((42,)),
                         reference_out="ref.npz", scores_out="s.json",
                         gate_out="g.json", regression_out="r.json")
    next(j for j in jobs if j.name == "gate").deps = ["nonexistent"]
    with pytest.raises(ValueError, match="not in DAG"):
        dag.validate_dag(jobs)
