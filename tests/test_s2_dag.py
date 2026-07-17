"""The historical G1 DAG is superseded by exact Round 0005 admission."""
import pytest

from basemap.round0005_retirement import (RetiredLauncherError,
                                          retirement_message)
import experiments.dag_template as dag


def _build(work_dir):
    return dag.build_g1_dag(
        train_cfg="experiments/configs/_stdcurve_s42.yaml", work_dir=str(work_dir),
        reference=str(work_dir / "ref.npz"),
        legacy_run_dir=str(work_dir / "legacy"),
        testbed_2m=str(work_dir / "tb2m"),
        kernel_run_2m=str(work_dir / "k2m"),
        decision_out=str(work_dir / "dec.json"),
        gate_out=str(work_dir / "gate.json"))


def test_g1_dag_builder_retires_before_output_or_job_construction(tmp_path):
    work_dir = tmp_path / "must-not-exist"
    with pytest.raises(RetiredLauncherError, match="RETIRED for Round 0005"):
        _build(work_dir)
    assert not work_dir.exists()


def test_g1_dag_main_retires_before_controller_child(monkeypatch):
    calls = []
    monkeypatch.setattr(dag, "run_jobs", lambda *args, **kwargs: calls.append("child"))
    with pytest.raises(RetiredLauncherError, match="exact signed manifest"):
        dag.main()
    assert calls == []


def test_g1_dag_retirement_names_scale_certificate_and_row_derivation():
    message = retirement_message("experiments/dag_template.py")
    assert "round0005_performance_certificate.v3" in message
    assert "reopened-input row derivation" in message


def test_g1_dag_has_no_environment_override_bypass(tmp_path, monkeypatch):
    monkeypatch.setenv("BASEMAP_UNSAFE_SAME_PROCESS", "1")
    work_dir = tmp_path / "still-must-not-exist"
    with pytest.raises(RetiredLauncherError):
        _build(work_dir)
    assert not work_dir.exists()
