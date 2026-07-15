"""S2 acceptance tests — the performance-regression admission contract.

The controlling property (closure review §S2 "done when"): a synthetic ~7x
slowdown ABORTS automatically, and the canary artifact DIAGNOSES which phase
regressed. These run on CPU with a fake clock — no GPU needed.
"""
import types
import pytest

from basemap.pumap.parametric_umap import perf as perfmod
from basemap.pumap.parametric_umap.perf import CanaryProfiler, build_baseline_key


@pytest.fixture
def fake_clock(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(perfmod.time, "perf_counter", lambda: t["now"])
    return t


def _run(prof, per_update_dt, clock, n=200):
    aborted_at = None
    for u in range(1, n + 1):
        clock["now"] += per_update_dt
        if prof.on_update(global_step=u, positive_updates=u):
            aborted_at = u
            break
    return aborted_at


def test_healthy_rate_does_not_abort(fake_clock):
    # 0.001 s/update -> ~1000 upd/s, well above the 100 floor.
    prof = CanaryProfiler(warmup=5, max_steps=55, floor=100.0, device="cpu",
                          subfloor_patience=3)
    aborted_at = _run(prof, 0.001, fake_clock)
    assert aborted_at is None
    assert prof.abort is False
    out = prof.finalize()
    assert out["n_windows"] >= 5
    assert out["rate_median"] > 100
    assert out["aborted"] is False


def test_seven_x_slowdown_aborts(fake_clock):
    # Baseline ~1000 upd/s; a 7x slowdown -> ~140 upd/s. Set the floor at 200 so
    # the slow run is sub-floor and must abort within patience windows.
    prof = CanaryProfiler(warmup=5, max_steps=105, floor=200.0, device="cpu",
                          subfloor_patience=3)
    # 7.1 ms/update -> ~140 upd/s < 200 floor
    aborted_at = _run(prof, 1.0 / 140.0, fake_clock)
    assert aborted_at is not None, "a 7x slowdown must trip the abort"
    assert prof.abort is True
    assert prof.consecutive_subfloor >= 3
    assert "below floor" in prof.abort_reason
    out = prof.finalize()
    assert out["aborted"] is True
    # It aborts EARLY — not after the full window budget was spent.
    assert out["n_windows"] < 5 + 1


def test_abort_is_prompt_not_after_full_budget(fake_clock):
    # With patience=2 the abort should land on the 2nd sub-floor window.
    prof = CanaryProfiler(warmup=0, max_steps=100, floor=1000.0, device="cpu",
                          subfloor_patience=2, n_windows=5)
    _run(prof, 0.01, fake_clock)      # 100 upd/s, far below 1000
    assert prof.abort is True
    assert len(prof.windows) == 2     # stopped at the 2nd window, not the 5th


def test_phase_timing_diagnoses_dominant_phase(fake_clock):
    # CPU phase timing uses perf_counter; make "forward" the fat phase.
    prof = CanaryProfiler(warmup=0, max_steps=100, floor=1.0, device="cpu",
                          sample_every=1)
    for step in range(1, 12):
        with prof.phase("sample", step):
            fake_clock["now"] += 0.001
        with prof.phase("forward", step):
            fake_clock["now"] += 0.010     # 10x the others -> dominant
        with prof.phase("backward", step):
            fake_clock["now"] += 0.002
    out = prof.finalize()
    assert set(out["phase_ms_median"]) == {"sample", "forward", "backward"}
    assert out["dominant_phase"] == "forward"
    assert out["phase_fractions"]["forward"] > 0.5
    assert all(v >= 1 for v in out["phase_samples"].values())


def test_baseline_key_has_shape_and_pipeline():
    model = types.SimpleNamespace(hidden_dim=1024, n_components=2, n_layers=3)
    key = build_baseline_key(
        model=model, n=8_000_000, d=768, n_edges=120_000_000, batch_size=8192,
        use_amp=True, kernel="legacy_lp",
        pipeline_info={"pipeline": "device", "sampler_class": "DeviceEdgeSampler",
                       "positive_sampling": "weighted_with_replacement",
                       "x_residency": "device_fp16"}, device="cpu")
    for f in ("n", "d", "n_edges", "hidden_dim", "n_components", "batch_size",
              "use_amp", "kernel", "pipeline", "positive_sampling", "x_residency"):
        assert f in key, f"baseline key missing {f}"
    assert key["pipeline"] == "device"
    assert key["positive_sampling"] == "weighted_with_replacement"
    assert "torch" in key   # torch is importable in the env


def test_finalize_reports_env_and_faults(fake_clock):
    prof = CanaryProfiler(warmup=0, max_steps=50, floor=1.0, device="cpu")
    _run(prof, 0.001, fake_clock, n=60)
    out = prof.finalize(bench_seconds=1.0, setup_seconds=12.0)
    assert out["rss_peak_gb"] > 0
    assert "minor_faults" in out and "major_faults" in out
    assert out["setup_seconds"] == 12.0
    assert "lease_id" in out          # None off-controller, present on-controller
