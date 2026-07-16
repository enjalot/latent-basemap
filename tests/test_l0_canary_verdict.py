"""L0.1 — the canary derives from the exact train config (family-preserving) and
the verdict wrapper judges independently of the raw runner's exit code."""
import os, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
import experiments.run_canary as rcan

STDCURVE = "experiments/configs/_stdcurve_s42.yaml"


def test_canary_derives_only_registered_diffs():
    run_dir = tempfile.mkdtemp()
    cfg, family = rcan.derive_canary_config(STDCURVE, run_dir, 1200, 200, 200.0, 250.0)
    # registered canary diffs applied
    assert cfg.train.canary_max_steps == 1200 and cfg.train.canary_warmup == 200
    assert cfg.train.canary_floor == 200.0 and cfg.train.canary_warn_rate == 250.0
    assert cfg.train.require_full_budget is False
    assert cfg.logging.save_model is False
    assert cfg.logging.run_dir_override == run_dir
    assert cfg.name.endswith("_canary")
    # config family preserved (proves the canary exercises the same run)
    assert family["model.low_dim_kernel"] == "umap"
    assert family["train.weighted_edge_sampling"] is True
    assert family["train.required_input_pipeline"] == "device"


def test_canary_refuses_to_write_tracked_config(monkeypatch, tmp_path):
    # writing the derived config into experiments/configs (tracked) must be refused
    import subprocess
    argv = ["run_canary.py", "--train-config", STDCURVE, "--run-dir", str(tmp_path / "r"),
            "--out", str(tmp_path / "v.json"),
            "--scratch-config", "experiments/configs/_should_not_write.yaml"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="refuse to write"):
        rcan.main()
    assert not os.path.exists("experiments/configs/_should_not_write.yaml")


def test_canary_family_mismatch_raises(tmp_path, monkeypatch):
    # a train config whose family cannot be reproduced by the canary overrides must
    # be caught. Simulate by making derive apply an unexpected extra key.
    orig = rcan.load_config
    def patched(path, overrides):
        cfg = orig(path, overrides)
        if overrides:                       # only mutate the canary variant
            cfg.model.hidden_dim = 999      # break the family
        return cfg
    monkeypatch.setattr(rcan, "load_config", patched)
    with pytest.raises(SystemExit, match="config-family diverged"):
        rcan.derive_canary_config(STDCURVE, str(tmp_path), 1200, 200, 200.0, 250.0)
