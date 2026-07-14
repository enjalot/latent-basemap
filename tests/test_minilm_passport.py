"""P0-F: MiniLM passport variance formula. The synthetic-covariance test fails if
the top-2 fraction is computed against a top-10 singular-value denominator (the
R6 bug) instead of the full spectrum."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.minilm_passport import top2_total_variance_fraction


def _sample_with_spectrum(eigs, n=4000, seed=0):
    """Draw n rows from a zero-mean Gaussian whose per-dim variances are ``eigs``
    (already in a PCA basis → covariance is diagonal, singular values ∝ sqrt(eig))."""
    rng = np.random.RandomState(seed)
    d = len(eigs)
    return (rng.randn(n, d) * np.sqrt(eigs)[None, :]).astype(np.float32)


def test_uses_full_spectrum_not_top10_denominator():
    # Spectrum: two big components + a long tail. Top-2/total is small; top-2/top-10
    # is large. The correct function must return the small (total) fraction.
    eigs = np.array([50.0, 40.0] + [1.0] * 200)      # tail dominates total variance
    S = _sample_with_spectrum(eigs, n=6000, seed=1)
    frac, method = top2_total_variance_fraction(S)
    total = eigs.sum()
    expected_total = (eigs[0] + eigs[1]) / total                     # ~0.31
    expected_top10_bug = (eigs[0] + eigs[1]) / eigs[:10].sum()       # ~0.918
    assert abs(frac - expected_total) < 0.05, (frac, expected_total)
    # the reported fraction must NOT be the inflated top-10-denominator value
    assert abs(frac - expected_top10_bug) > 0.3, frac
    # and the generator records what the buggy denominator WOULD have given
    assert abs(method["top2_over_top10_denom_bug"] - expected_top10_bug) < 0.05


def test_fraction_bounds_and_monotonic():
    S = _sample_with_spectrum(np.array([10.0, 5.0, 1.0, 1.0, 1.0]), seed=2)
    frac, _ = top2_total_variance_fraction(S)
    assert 0.0 < frac < 1.0
    # concentrating variance into the top-2 raises the fraction
    S2 = _sample_with_spectrum(np.array([100.0, 50.0, 1.0, 1.0, 1.0]), seed=2)
    frac2, _ = top2_total_variance_fraction(S2)
    assert frac2 > frac


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-q'])
