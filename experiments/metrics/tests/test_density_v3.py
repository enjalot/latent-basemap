"""Tests for density_v3 — the repaired density metric.

The mandatory positive controls are ``test_planted_degenerate_*``: they build a
synthetic corpus that contains the exact defect review-0225 found in the real
one (a duplicate family whose members have ``r_hd == 0`` and, because they also
collapse to a single map coordinate, ``r_2d == 0``), then check that

  * the v2-style statistic really does move by tens of percent because of those
    anchors — i.e. the failure fires on this construction, so the test is not
    vacuous; and
  * density_v3 moves by less than 2%.

Run with:  CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
             experiments/metrics/tests/test_density_v3.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from density_v3 import (  # noqa: E402
    DensityV3Error,
    density_v2_legacy,
    density_v3,
    draw_anchor_pool,
    high_d_radii,
    low_d_radii,
    pearson,
    spearman,
)

N_ROWS = 12_000
N_ANCHORS = 2_000
N_DUPLICATES = 60
N_POOL_DEGENERATE = 1   # one degenerate anchor, exactly as on the real corpus
SEED = 0
# The real corpus correlates weakly (density_v2 with the degenerate anchor
# dropped is 0.06-0.18), which is precisely why one extreme diagonal point could
# dominate.  A synthetic corpus that already correlates at 0.99 leaves no room
# for the failure to show, so the fixture noise is tuned to land in the same
# regime: Spearman ~0.25 before any degeneracy is planted.
NOISE = 1.1


# ── fixtures ────────────────────────────────────────────────────────────────

def _blobs(rng: np.random.Generator, n: int) -> np.ndarray:
    """2-D points from blobs of deliberately different local density."""
    sizes = np.array([0.35, 0.7, 1.4, 2.8, 5.6])
    centres = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0],
                        [12.0, 12.0], [24.0, 6.0]])
    per = n // len(sizes)
    parts = [rng.normal(centres[i], sizes[i], size=(per, 2))
             for i in range(len(sizes))]
    parts.append(rng.normal(centres[0], sizes[0], size=(n - per * len(sizes), 2)))
    return np.concatenate(parts).astype(np.float64)


def _corpus(with_duplicates: bool, noise: float = NOISE):
    """A map plus per-row high-D radii that agree with it up to rank noise.

    ``N_DUPLICATES`` rows — including exactly ONE anchor the v3 pool draws, to
    mirror the real corpus where a single anchor of 4,000 was degenerate — are
    moved into an isolated satellite far from every blob.  In the clean
    corpus that satellite is an ordinary Gaussian cluster; in the degenerate one
    it collapses to a single exact coordinate, so those rows get ``r_2d == 0``
    and therefore ``r_hd == 0``.  This is exactly the degeneracy that supplied
    two thirds of density_v2's value on the real 2M corpus (substrate row
    1449227 with its 1,377 duplicates).

    The two corpora are identical everywhere else — same geometry draw, same
    noise draw, same rows moved into the satellite — so the *only* difference
    between their scores is the collapse.
    """
    geometry = np.random.default_rng(1234)
    picker = np.random.default_rng(4321)
    xy = _blobs(geometry, N_ROWS)
    log_noise = geometry.normal(size=N_ROWS)

    pool = draw_anchor_pool(N_ROWS, N_ANCHORS, SEED)
    outside = np.setdiff1d(np.arange(N_ROWS), pool)
    duplicate_rows = np.sort(np.concatenate([
        pool[:N_POOL_DEGENERATE],
        picker.choice(outside, size=N_DUPLICATES - N_POOL_DEGENERATE,
                      replace=False),
    ]))
    satellite = np.array([60.0, 60.0])
    if with_duplicates:
        xy[duplicate_rows] = satellite
    else:
        xy[duplicate_rows] = satellite + picker.normal(
            size=(len(duplicate_rows), 2))

    r_2d_all = low_d_radii(xy, np.arange(N_ROWS), threads=2)
    # High-D radii are built by *rank* mixing rather than multiplicative noise,
    # so that (like the real MiniLM substrate) they occupy a bounded band with
    # no near-zero tail — the only zeros are the planted duplicates.  ``noise``
    # controls how much the high-D ordering disagrees with the map's.
    order = np.argsort(np.argsort(r_2d_all)) / (N_ROWS - 1)
    mixed = np.argsort(np.argsort(order + noise * log_noise)) / (N_ROWS - 1)
    r_hd_all = 0.20 + 0.90 * mixed
    r_hd_all[r_2d_all == 0.0] = 0.0
    return xy, r_hd_all, pool, duplicate_rows


@pytest.fixture(scope="module")
def clean_corpus():
    return _corpus(with_duplicates=False)


@pytest.fixture(scope="module")
def degenerate_corpus():
    return _corpus(with_duplicates=True)


def _v3(xy, radii, **kwargs):
    return density_v3(xy, radii, anchor_seed=SEED, n_anchors=N_ANCHORS,
                      threads=2, **kwargs)


# ── (a) the mandatory positive control ──────────────────────────────────────

def test_planted_degeneracy_is_actually_present(degenerate_corpus):
    """Sanity: the construction really does produce r_hd == 0 anchors."""
    xy, r_hd_all, pool, duplicate_rows = degenerate_corpus
    assert len(duplicate_rows) == N_DUPLICATES
    assert np.all(r_hd_all[duplicate_rows] == 0.0)
    in_pool = np.intersect1d(pool, duplicate_rows)
    assert len(in_pool) == N_POOL_DEGENERATE, in_pool
    # and they collapse in 2-D too, which is what makes them leverage points
    r_2d = low_d_radii(xy, in_pool, threads=2)
    assert np.all(r_2d == 0.0)


def test_v2_style_failure_fires_on_the_planted_degeneracy(degenerate_corpus):
    """The v2 statistic must be badly moved by the degenerate anchors.

    Without this the v3 test below would be vacuous.  On the real 2M corpus one
    anchor in 4,000 moved density_v2 from 0.4377 to 0.1681 (62%).
    """
    xy, r_hd_all, pool, _ = degenerate_corpus
    v2 = density_v2_legacy(xy, r_hd_all, anchor_seed=SEED, n_anchors=N_ANCHORS,
                           threads=2)
    assert v2["n_degenerate_hd_included"] == N_POOL_DEGENERATE

    eligible = pool[r_hd_all[pool] > 1e-3][:N_ANCHORS]
    r_2d = low_d_radii(xy, eligible, threads=2)
    eps = 1e-12
    without = pearson(np.log(r_hd_all[eligible] + eps), np.log(r_2d + eps))
    shift = abs(v2["value"] - without) / abs(v2["value"])
    assert shift > 0.20, (
        f"v2 only moved {shift:.1%}; the positive control did not fire "
        f"(with={v2['value']:.4f} without={without:.4f})"
    )
    # and that single anchor dominates the leave-one-out sweep, exactly as
    # row 1449227 does on the real corpus (61.6%-91.4% there)
    assert v2["leave_one_out"]["pearson_log"]["max_relative_shift"] > 0.20


def test_v3_resists_the_planted_degeneracy(degenerate_corpus, clean_corpus):
    """density_v3 must move less than 2% when the degeneracy is planted."""
    xy_bad, r_hd_bad, _, _ = degenerate_corpus
    xy_ok, r_hd_ok, _, _ = clean_corpus
    bad, ok = _v3(xy_bad, r_hd_bad), _v3(xy_ok, r_hd_ok)
    assert bad["n_excluded_degenerate_hd"] == N_POOL_DEGENERATE
    assert ok["n_excluded_degenerate_hd"] == 0
    shift = abs(bad["value"] - ok["value"]) / abs(ok["value"])
    assert shift < 0.02, (
        f"density_v3 moved {shift:.2%} (clean={ok['value']:.4f} "
        f"planted={bad['value']:.4f})"
    )


def test_v3_survives_even_without_the_exclusion(degenerate_corpus):
    """Defence in depth: with eps_hd disabled the winsorization + rank
    statistic still keep the shift under 2%."""
    xy, r_hd_all, _, _ = degenerate_corpus
    excluded = _v3(xy, r_hd_all)
    included = _v3(xy, r_hd_all, eps_hd=-1.0)   # nothing is excluded
    assert included["n_excluded_degenerate_hd"] == 0
    assert included["n_anchors"] == N_ANCHORS
    shift = abs(included["value"] - excluded["value"]) / abs(excluded["value"])
    assert shift < 0.02, f"policy-free v3 moved {shift:.2%}"


# ── (b)/(c) structure controls ──────────────────────────────────────────────

def test_preserved_density_structure_scores_high():
    """A map whose 2-D crowding is a monotone function of high-D crowding."""
    rng = np.random.default_rng(7)
    xy = _blobs(rng, N_ROWS)
    r_2d_all = low_d_radii(xy, np.arange(N_ROWS), threads=2)
    exact = _v3(xy, r_2d_all)                       # perfect agreement
    monotone = _v3(xy, r_2d_all ** 1.7)             # monotone but nonlinear
    noisy = _v3(xy, r_2d_all * np.exp(0.25 * rng.normal(size=N_ROWS)))
    assert exact["spearman"] > 0.99
    assert monotone["spearman"] > 0.99, "Spearman must be transform-invariant"
    assert noisy["spearman"] > 0.60
    assert exact["pearson_log"] > 0.95


def test_shuffled_map_scores_near_zero(clean_corpus):
    xy, r_hd_all, _, _ = clean_corpus
    rng = np.random.default_rng(99)
    shuffled = xy[rng.permutation(len(xy))]
    result = _v3(shuffled, r_hd_all)
    assert abs(result["spearman"]) < 0.05, result["spearman"]
    assert abs(result["pearson_log"]) < 0.05, result["pearson_log"]
    # ... while the same corpus scored clearly positive before shuffling
    assert _v3(xy, r_hd_all)["spearman"] > 0.15


# ── (d) determinism ─────────────────────────────────────────────────────────

def test_anchor_draw_is_deterministic_and_seed_dependent():
    a = draw_anchor_pool(N_ROWS, N_ANCHORS, anchor_seed=0)
    b = draw_anchor_pool(N_ROWS, N_ANCHORS, anchor_seed=0)
    c = draw_anchor_pool(N_ROWS, N_ANCHORS, anchor_seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(np.diff(a) > 0), "pool must be sorted and distinct"
    assert len(a) == int(np.ceil(N_ANCHORS * 1.25))


def test_metric_is_deterministic_by_seed(clean_corpus):
    xy, r_hd_all, _, _ = clean_corpus
    first, second = _v3(xy, r_hd_all), _v3(xy, r_hd_all)
    assert first["spearman"] == second["spearman"]
    assert first["pearson_log"] == second["pearson_log"]
    assert first["anchor_ids_sha256_prefix"] == second["anchor_ids_sha256_prefix"]

    other = density_v3(xy, r_hd_all, anchor_seed=1, n_anchors=N_ANCHORS, threads=2)
    assert other["anchor_ids_sha256_prefix"] != first["anchor_ids_sha256_prefix"]
    assert abs(other["spearman"] - first["spearman"]) < 0.10, (
        "different anchor draws must agree to within sampling error"
    )


# ── (e) leave-one-out stability ─────────────────────────────────────────────

def test_leave_one_out_stability_bound(clean_corpus, degenerate_corpus):
    for corpus in (clean_corpus, degenerate_corpus):
        xy, r_hd_all, _, _ = corpus
        result = _v3(xy, r_hd_all)
        loo = result["leave_one_out"]
        assert loo["spearman"]["max_relative_shift"] < 0.02, loo["spearman"]
        assert loo["spearman"]["max_absolute_shift"] < 0.01, loo["spearman"]
        assert loo["pearson_log"]["max_absolute_shift"] < 0.02, loo["pearson_log"]


def test_leave_one_out_is_exact():
    """The O(n) leave-one-out Pearson must equal a brute-force recomputation."""
    from density_v3 import _loo_pearson

    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    y = 0.7 * x + rng.normal(size=200)
    fast = _loo_pearson(x, y)
    slow = np.array([
        pearson(np.delete(x, i), np.delete(y, i)) for i in range(len(x))
    ])
    assert np.allclose(fast, slow, atol=1e-12)


# ── radius kernels ──────────────────────────────────────────────────────────

def test_high_d_radii_match_brute_force_and_detect_duplicates():
    """Chunked selection + per-dimension rerank must equal a naive computation,
    and must report an EXACT zero for duplicate rows.

    The naive ``2 - 2s`` matmul shortcut reports ~1e-4 for duplicate unit-norm
    rows; that cancellation is what made density_v2's degeneracy invisible to
    two independent reviewers, so it is tested explicitly.
    """
    rng = np.random.default_rng(11)
    n, dims = 3_000, 16
    substrate = rng.normal(size=(n, dims))
    substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
    substrate[100:140] = substrate[100]          # a duplicate family
    anchors = np.array([5, 100, 101, 900, 2999], dtype=np.int64)

    got = high_d_radii(substrate, anchors, k=15, corpus_chunk=317,
                       anchor_tile=2, threads=2)

    expect = np.empty(len(anchors))
    for position, anchor in enumerate(anchors):
        diff = substrate - substrate[anchor]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        dist = np.delete(dist, anchor)
        expect[position] = np.sort(dist)[:15].mean()
    assert np.allclose(got, expect, atol=1e-6), (got, expect)
    assert got[1] == 0.0 and got[2] == 0.0, "duplicate family must give r_hd == 0"
    assert got[0] > 0.1


def test_low_d_radii_match_brute_force():
    rng = np.random.default_rng(5)
    xy = rng.normal(size=(2_000, 2))
    xy[7] = xy[8] = xy[9]                        # coordinate duplicates
    anchors = np.array([0, 7, 8, 1_999], dtype=np.int64)
    got = low_d_radii(xy, anchors, k=15, threads=2)
    for position, anchor in enumerate(anchors):
        dist = np.linalg.norm(xy - xy[anchor], axis=1)
        dist = np.delete(dist, anchor)
        assert abs(got[position] - np.sort(dist)[:15].mean()) < 1e-12


def test_substrate_input_mode_matches_precomputed_radii():
    """Mode (b) (compute radii from a substrate) must equal mode (a)."""
    rng = np.random.default_rng(13)
    n, dims = 4_000, 12
    substrate = rng.normal(size=(n, dims)).astype(np.float32)
    substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
    xy = substrate[:, :2] * 3.0

    from_substrate = density_v3(xy, substrate, anchor_seed=SEED, n_anchors=400,
                                threads=2, leave_one_out=False,
                                radii_kwargs={"corpus_chunk": 997,
                                              "anchor_tile": 64})
    pool = draw_anchor_pool(n, 400, SEED)
    radii_all = np.zeros(n)
    radii_all[pool] = high_d_radii(substrate, pool, threads=2)
    from_radii = density_v3(xy, radii_all, anchor_seed=SEED, n_anchors=400,
                            threads=2, leave_one_out=False)
    assert from_substrate["high_d_mode"] == "substrate"
    assert from_radii["high_d_mode"] == "radii_all_rows"
    assert abs(from_substrate["spearman"] - from_radii["spearman"]) < 1e-12


def test_substrate_path_input(tmp_path):
    rng = np.random.default_rng(17)
    substrate = rng.normal(size=(1_500, 8)).astype(np.float32)
    substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
    path = tmp_path / "substrate.npy"
    np.save(path, substrate)
    result = density_v3(substrate[:, :2], str(path), anchor_seed=SEED,
                        n_anchors=200, threads=2, leave_one_out=False)
    assert result["high_d_mode"] == "substrate"
    assert -1.0 <= result["spearman"] <= 1.0


# ── reporting + failure modes ───────────────────────────────────────────────

def test_report_contents(degenerate_corpus):
    xy, r_hd_all, _, _ = degenerate_corpus
    result = _v3(xy, r_hd_all)
    for key in ("spearman", "pearson_log", "n_anchors",
                "n_excluded_degenerate_hd", "anchor_rule", "leave_one_out",
                "winsorization", "eps_hd", "anchor_shortfall"):
        assert key in result, key
    assert result["primary_statistic"] == "spearman"
    assert result["value"] == result["spearman"]
    assert result["n_anchors"] == N_ANCHORS
    assert result["anchor_shortfall"] == 0


def test_anchor_count_is_at_least_eight_thousand_by_default():
    from density_v3 import DEFAULT_N_ANCHORS

    assert DEFAULT_N_ANCHORS >= 8_000


def test_spearman_is_transform_invariant():
    rng = np.random.default_rng(23)
    x = np.exp(rng.normal(size=500))
    y = np.exp(rng.normal(size=500)) * x
    assert abs(spearman(x, y) - spearman(np.log(x), y ** 3)) < 1e-12


def test_malformed_inputs_fail_closed():
    rng = np.random.default_rng(29)
    xy = rng.normal(size=(500, 2))
    with pytest.raises(DensityV3Error):
        density_v3(xy, np.zeros(499), anchor_seed=SEED, n_anchors=100)
    with pytest.raises(DensityV3Error):
        density_v3(xy[:, :1], np.ones(500), anchor_seed=SEED, n_anchors=100)
    with pytest.raises(DensityV3Error):
        density_v3(xy, np.zeros(500), anchor_seed=SEED, n_anchors=100)
    with pytest.raises(DensityV3Error):
        low_d_radii(xy, np.array([500]))
