"""Guardrail unit tests for the Vendi score implementation.

Run directly:  python test_vendi.py   (or via pytest).
"""

from __future__ import annotations

import numpy as np

from vendi import vendi_cosine, vendi_from_eigs


def _vendi_direct(X: np.ndarray) -> float:
    """Reference impl via the full N x N kernel (small N only)."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    K = (Xn @ Xn.T) / X.shape[0]
    eigs = np.linalg.eigvalsh(K)
    return vendi_from_eigs(eigs)


def test_identical_vectors():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(384)
    X = np.tile(v, (100, 1))
    score = vendi_cosine(X)
    assert abs(score - 1.0) < 1e-6, f"identical -> {score}, expected ~1.0"


def test_orthonormal_vectors():
    N = 50
    X = np.eye(N)
    score = vendi_cosine(X)
    assert abs(score - N) < 1e-6, f"orthonormal -> {score}, expected ~{N}"


def test_scale_invariance():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((500, 384))
    a = vendi_cosine(X)
    b = vendi_cosine(3.7 * X)
    assert abs(a - b) < 1e-9, f"scale-variant: {a} vs {b}"


def test_gram_trick_matches_direct():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 384))
    direct = _vendi_direct(X)
    trick = vendi_cosine(X)
    assert abs(direct - trick) < 1e-6, f"direct {direct} vs trick {trick}"


def test_monotonicity_dupes_vs_independent():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(384)
    dupes = base[None, :] + 1e-4 * rng.standard_normal((1000, 384))
    independent = rng.standard_normal((1000, 384))
    v_dupes = vendi_cosine(dupes)
    v_indep = vendi_cosine(independent)
    assert v_dupes < v_indep, f"dupes {v_dupes} !< indep {v_indep}"
    # near-duplicates should collapse to a tiny effective count
    assert v_dupes < 10.0, f"dupes Vendi unexpectedly large: {v_dupes}"


def test_chunking_matches_single_pass():
    """Chunked accumulation must equal a single-pass computation."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((1000, 384))
    a = vendi_cosine(X, chunk=1000)
    b = vendi_cosine(X, chunk=137)
    assert abs(a - b) < 1e-9, f"chunking mismatch: {a} vs {b}"


def _run_all():
    tests = [
        test_identical_vectors,
        test_orthonormal_vectors,
        test_scale_invariance,
        test_gram_trick_matches_direct,
        test_monotonicity_dupes_vs_independent,
        test_chunking_matches_single_pass,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")

    # Print a few illustrative numbers.
    rng = np.random.default_rng(3)
    base = rng.standard_normal(384)
    dupes = base[None, :] + 1e-4 * rng.standard_normal((1000, 384))
    independent = rng.standard_normal((1000, 384))
    print(f"  dupes Vendi     = {vendi_cosine(dupes):.4f}")
    print(f"  independent Vendi = {vendi_cosine(independent):.4f}")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_all()
