"""Tests for the COLLAPSE and FOG map-quality metrics.

The program's standing rule: a guard whose test suite contains no failing
input is untested at its only job. So the first two tests below are POSITIVE
CONTROLS -- synthetic maps built to be broken, which the guard must reject:

  (a) a bead-collapsed map must FAIL the collapse floor;
  (b) a uniform-noise haze map must FAIL the fog ceiling;

followed by a negative control (c) a healthy map must clear both -- and clear
fog with a real measurement rather than the 0.0000 degeneracy -- plus the
invariance / determinism / memmap properties the metrics claim.

Note on (b): the positive control buries clusters in uniform noise rather
than using pure uniform noise, because pure noise CANNOT fail the fog ceiling
by arithmetic. `test_pure_uniform_noise_is_a_documented_blind_spot_of_fog`
pins that limitation down instead of hiding it; see REPORT.md section 7.

The band constants below are the provisional bounds proposed in REPORT.md
section 6. They are not registered gates.

Run:
  CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest experiments/metrics/tests/ \
      -q -p no:cacheprovider
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collapse_fog import (  # noqa: E402
    MADN_CONSISTENCY,
    load_coords,
    madn,
    map_collapse,
    map_fog,
    robust_ceiling,
    robust_floor,
)

# Provisional reference bands (see REPORT.md). The tests exercise the guard,
# not the bands' evidentiary status -- these are the numbers a gate would use.
COLLAPSE_FLOOR = 0.6444  # REPORT.md section 6, core umap family, k = 6.05513
FOG_CEILING = 0.7330     # REPORT.md section 6, same family and multiplier


# --- synthetic maps -----------------------------------------------------

def collapsed_map(n=200_000, n_beads=40, seed=1):
    """Bead collapse: all mass in a few point-like clumps spread over a wide
    extent. This is the failure the registered kernel produces and that FFR
    rewards -- neighbourhoods are tight, the map extent is not."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-10, 10, size=(n_beads, 2))
    which = rng.integers(0, n_beads, size=n)
    jitter = rng.normal(0.0, 1e-4, size=(n, 2))
    return (centres[which] + jitter).astype(np.float32)


def uniform_noise_map(n=2_000_000, seed=2):
    """Pure haze: no clusters at all, mass spread evenly over the extent."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-10, 10, size=(n, 2)).astype(np.float32)


def haze_map(n=1_000_000, n_blobs=10, sigma=0.05, background=0.80, seed=4):
    """The failure FOG exists to catch: real clusters buried in uniform noise.

    Dense cores survive, so the 1%-of-peak cutoff has resolution, but 80% of
    the mass sits in near-empty bins between the clusters.
    """
    rng = np.random.default_rng(seed)
    n_bg = int(n * background)
    n_fg = n - n_bg
    centres = rng.uniform(-8, 8, size=(n_blobs, 2))
    which = rng.integers(0, n_blobs, size=n_fg)
    fg = centres[which] + rng.normal(0.0, sigma, size=(n_fg, 2))
    bg = rng.uniform(-10, 10, size=(n_bg, 2))
    xy = np.vstack([fg, bg]).astype(np.float32)
    rng.shuffle(xy)
    return xy


def healthy_map(n=1_000_000, n_blobs=25, sigma=0.45, background=0.002,
                dup_pile=500, seed=3):
    """Gaussian blobs of finite radius with only a trace of background.

    Tuned to land where the real min_dist = 0.00 umap arms land: collapse
    ~1.6 and fog ~0.52 against their measured 1.05-1.23 and 0.45-0.54. The
    duplicate pile mirrors the real substrate, whose peak bin is a group of
    1377 byte-identical rows -- without a pile of that kind a synthetic map
    at this N has a peak bin in the tens and fog degenerates to 0.0000, which
    would make the negative control vacuous rather than informative.
    """
    rng = np.random.default_rng(seed)
    dup_pile = max(0, min(dup_pile, n // 100))
    n_bg = int(n * background)
    n_fg = n - n_bg - dup_pile
    centres = rng.uniform(-8, 8, size=(n_blobs, 2))
    which = rng.integers(0, n_blobs, size=n_fg)
    fg = centres[which] + rng.normal(0.0, sigma, size=(n_fg, 2))
    bg = rng.uniform(-10, 10, size=(n_bg, 2))
    pile = np.repeat(centres[:1], dup_pile, axis=0)
    xy = np.vstack([fg, bg, pile]).astype(np.float32)
    rng.shuffle(xy)
    return xy


# --- (a) POSITIVE CONTROL: collapse ------------------------------------

def test_positive_control_collapsed_map_fails_the_collapse_floor():
    xy = collapsed_map()
    stat = map_collapse(xy)["r10_over_radius_times_sqrt_n"]
    assert stat < COLLAPSE_FLOOR, (
        f"collapse guard did not reject a bead-collapsed map: {stat:.4f}")
    # And the raw (un-adjusted) ratio is tiny, as the sandbox observed.
    assert map_collapse(xy)["r10_over_map_radius_median"] < 1e-3


# --- (b) POSITIVE CONTROL: fog -----------------------------------------

def test_positive_control_uniform_noise_haze_fails_the_fog_ceiling():
    """Clusters buried in uniform noise -- the haze failure fog exists to
    catch. This is the required positive control; see the test below for why
    it uses noise-over-clusters rather than pure noise."""
    f = map_fog(haze_map())
    assert not f["degenerate"], f
    assert f["fog"] > FOG_CEILING, (
        f"fog guard did not reject a noise-buried map: {f['fog']:.4f}")


def test_pure_uniform_noise_is_a_documented_blind_spot_of_fog():
    """A structureless uniform map reports fog EXACTLY 0.0000 and cannot be
    made to fail, by arithmetic: bin counts are integers and the cutoff is
    1% of the peak bin, so unless the peak exceeds 100 counts no occupied bin
    can be 'low density'. At 2M points over 1024^2 bins the peak is ~13.

    This is a property of the inherited definition, not of this
    implementation. The metric reports `degenerate` so a caller can refuse
    the measurement, and `occupied_bin_fraction` separates the case cleanly
    (~0.81 for uniform noise vs 0.04-0.23 for real maps)."""
    f = map_fog(uniform_noise_map())
    assert f["fog"] == 0.0
    assert f["degenerate"] is True
    assert f["absolute_floor_binding"] is True
    assert f["occupied_bin_fraction"] > 0.7
    # and collapse does not catch it either -- neither metric covers this mode
    assert map_collapse(uniform_noise_map(n=200_000))["r10_over_radius_times_sqrt_n"] \
        > COLLAPSE_FLOOR


def test_fog_degeneracy_flag_matches_the_measured_sandbox_case():
    """umap-md035-x2 has peak bin 99 and reports fog 0.0000 while being one
    of the haziest maps in the sandbox. Reproduce that regime synthetically:
    a haze map thin enough that its peak bin holds < 100 points is flagged
    degenerate rather than silently scored as perfect."""
    f = map_fog(haze_map(n=200_000, sigma=0.6, background=0.55))
    assert f["peak_bin_count"] < 100
    assert f["degenerate"] is True
    assert f["fog"] == 0.0


# --- (c) NEGATIVE CONTROL ----------------------------------------------

def test_healthy_map_passes_both():
    xy = healthy_map()
    c = map_collapse(xy)
    f = map_fog(xy)
    assert c["r10_over_radius_times_sqrt_n"] >= COLLAPSE_FLOOR, c
    assert f["fog"] <= FOG_CEILING, f
    # and the fog pass must be a real measurement, not the 0.0000 degeneracy
    assert not f["degenerate"], f
    assert f["fog"] > 0.0, f


def test_the_two_metrics_are_different_failure_directions():
    """A collapsed map must NOT be caught by fog, and a hazy map must NOT be
    caught by collapse -- that is why the gate needs both."""
    collapsed = collapsed_map()
    haze = haze_map()
    assert map_fog(collapsed)["fog"] <= FOG_CEILING
    assert map_collapse(haze)["r10_over_radius_times_sqrt_n"] >= COLLAPSE_FLOOR


# --- (d) N-invariance ---------------------------------------------------

def test_collapse_statistic_is_n_invariant_under_4x_subsampling():
    xy = healthy_map()
    full = map_collapse(xy)["r10_over_radius_times_sqrt_n"]
    rng = np.random.default_rng(11)
    idx = rng.choice(len(xy), size=len(xy) // 4, replace=False)
    quarter = map_collapse(np.ascontiguousarray(xy[idx]))["r10_over_radius_times_sqrt_n"]
    rel = abs(quarter - full) / full
    assert rel < 0.10, f"N-adjustment failed: {full:.4f} vs {quarter:.4f} (rel {rel:.3%})"


def test_raw_ratio_is_not_n_invariant():
    """Control for the test above: without the sqrt(N) factor the statistic
    moves by roughly 2x under 4x subsampling, which is why it is applied."""
    xy = healthy_map()
    full = map_collapse(xy)["r10_over_map_radius_median"]
    rng = np.random.default_rng(11)
    idx = rng.choice(len(xy), size=len(xy) // 4, replace=False)
    quarter = map_collapse(np.ascontiguousarray(xy[idx]))["r10_over_map_radius_median"]
    assert quarter / full > 1.5


# --- (e) determinism ----------------------------------------------------

def test_determinism_same_seed_identical_results():
    xy = healthy_map(n=100_000)
    a, b = map_collapse(xy, rng_seed=0), map_collapse(xy, rng_seed=0)
    assert a == b
    assert map_fog(xy) == map_fog(xy)  # fog uses a fixed stride, no rng


def test_different_seed_changes_the_sample_but_not_the_verdict():
    xy = healthy_map(n=100_000)
    a = map_collapse(xy, rng_seed=0)["r10_over_radius_times_sqrt_n"]
    b = map_collapse(xy, rng_seed=7)["r10_over_radius_times_sqrt_n"]
    assert abs(a - b) / a < 0.05


# --- (f) memmap path ----------------------------------------------------

def test_memmap_path_matches_in_memory(tmp_path):
    xy = healthy_map(n=100_000)
    p = tmp_path / "coordinates.npy"
    np.save(p, xy)
    mm = load_coords(p)
    assert isinstance(mm, np.memmap)
    assert map_collapse(mm) == map_collapse(xy)
    assert map_fog(mm) == map_fog(xy)


def test_memmap_subsampled_tree_path(tmp_path):
    """Force the >MAX_TREE_ROWS branch with a small threshold and check the
    N-adjusted statistic survives it."""
    xy = healthy_map(n=120_000)
    p = tmp_path / "coordinates.npy"
    np.save(p, xy)
    mm = load_coords(p)
    full = map_collapse(mm)
    sub = map_collapse(mm, max_tree_rows=30_000)
    assert sub["subsampled_for_tree"] is True
    assert sub["n_effective"] == 30_000
    assert full["subsampled_for_tree"] is False
    rel = abs(sub["r10_over_radius_times_sqrt_n"] - full["r10_over_radius_times_sqrt_n"]) \
        / full["r10_over_radius_times_sqrt_n"]
    assert rel < 0.10, f"subsampled tree path drifted {rel:.3%}"


# --- protocol / shape guards -------------------------------------------

def test_returns_protocol_parameters():
    c = map_collapse(healthy_map(n=5_000))
    for key in ("n_rows", "n_effective", "sample_size", "rng_seed", "k_neighbor",
                "radius_pct", "r10_median", "map_radius_p90",
                "r10_over_map_radius_median", "r10_over_radius_times_sqrt_n",
                "gate_direction"):
        assert key in c
    f = map_fog(healthy_map(n=5_000))
    for key in ("bins", "extent_pct", "threshold", "extent", "peak_bin_count",
                "occupied_bin_fraction", "low_density_mass_fraction", "fog",
                "gate_direction"):
        assert key in f


def test_sqrt_n_factor_is_exactly_applied():
    c = map_collapse(healthy_map(n=5_000))
    assert c["r10_over_radius_times_sqrt_n"] == pytest.approx(
        c["r10_over_map_radius_median"] * math.sqrt(c["n_effective"]), rel=1e-12)


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        map_collapse(np.zeros((100, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        map_fog(np.zeros((100, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        map_collapse(np.zeros((5, 2), dtype=np.float32))


def test_fog_absolute_floor_prevents_spurious_fog_on_a_sparse_map():
    """peak * 0.01 < 1 must not make every single-count bin 'low density'."""
    rng = np.random.default_rng(5)
    xy = rng.uniform(-1, 1, size=(500, 2)).astype(np.float32)
    f = map_fog(xy)
    assert f["low_bin_cutoff"] == 1.0
    assert f["fog"] == 0.0


# --- robust statistics --------------------------------------------------

def test_madn_and_floors():
    v = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert madn(v) == 0.0
    v = np.arange(9, dtype=float)
    assert madn(v) == pytest.approx(MADN_CONSISTENCY * 2.0)
    assert robust_floor(v, 2.0) == pytest.approx(4.0 - 2.0 * MADN_CONSISTENCY * 2.0)
    assert robust_ceiling(v, 2.0) == pytest.approx(4.0 + 2.0 * MADN_CONSISTENCY * 2.0)


def test_calibrator_reproduces_r0234_sealed_n13_multiplier():
    """Cheap smoke version of the REPORT's validation gate (1M families is
    the full protocol; this uses 200k and a looser tolerance so the suite
    stays fast). The sealed value is read, never written."""
    import null_calibration as nc

    if not nc.R0234_N13_SEALED_PATH.is_file():
        pytest.skip("sealed R0234 artifact not present on this machine")
    sealed = nc.sealed_multiplier(nc.R0234_N13_SEALED_PATH, "n13")
    got = nc.calibrate_one_sided(13, families=1_000_000, seed=20260809)
    rel = abs(got["calibrated_multiplier"] - sealed) / sealed
    assert rel < 0.02, f"{got['calibrated_multiplier']} vs sealed {sealed}"
