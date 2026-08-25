"""Vendi score: an "effective number of distinct items" diversity metric.

For N vectors, form the trace-normalized cosine kernel K (an N x N matrix with
unit-norm rows and K = (1/N) Xn Xn^T so trace(K) = 1). Its eigenvalues lambda_i
sum to 1, and the Vendi score is exp(Shannon entropy of that distribution):

    Vendi = exp(-sum_i lambda_i * log lambda_i)   (with 0*log0 := 0)

Identical items -> 1; N mutually orthogonal items -> N.

Scalability: we never form the N x N matrix. For the linear/cosine kernel the
nonzero eigenvalues of K = (1/N) Xn Xn^T are EXACTLY the eigenvalues of the
D x D matrix S = (1/N) Xn^T Xn (both trace-normalized to 1); the big matrix only
carries extra zero eigenvalues, which contribute 0 to the entropy. So we build
S in O(N*D^2) by chunking rows and eig it in O(D^3). Feasible at N=2M, D=384.
"""

from __future__ import annotations

import numpy as np


def vendi_from_eigs(eigs) -> float:
    """Vendi score from an array of eigenvalues (need not be pre-normalized).

    Clamps tiny negative (numerical) eigenvalues to 0, renormalizes to sum 1,
    then returns exp(Shannon entropy), using 0*log0 := 0.
    """
    eigs = np.asarray(eigs, dtype=np.float64)
    eigs = np.clip(eigs, 0.0, None)
    total = eigs.sum()
    if total <= 0.0:
        return 0.0
    p = eigs / total
    p = p[p > 0.0]  # 0*log0 := 0
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def vendi_cosine(X, *, chunk: int = 200_000) -> float:
    """Vendi score under the cosine kernel, computed via the D x D Gram trick.

    Parameters
    ----------
    X : array-like or np.memmap, shape (N, D)
        Embedding vectors (float32/float64). Cast to float64 per chunk.
    chunk : int
        Max number of rows materialized at once. Never forms an N x N matrix.
    """
    N = X.shape[0]
    D = X.shape[1]
    if N == 0:
        return 0.0

    # S = (1/N) sum_i xhat_i xhat_i^T, accumulated in float64 by chunks.
    S = np.zeros((D, D), dtype=np.float64)
    for start in range(0, N, chunk):
        Xc = np.asarray(X[start:start + chunk], dtype=np.float64)
        norms = np.linalg.norm(Xc, axis=1, keepdims=True)
        # Guard zero rows: leave them as zero so they contribute nothing to S.
        np.divide(Xc, norms, out=Xc, where=norms > 0.0)
        S += Xc.T @ Xc
    S /= N

    eigs = np.linalg.eigvalsh(S)
    return vendi_from_eigs(eigs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python vendi.py <path-to.npy>", file=sys.stderr)
        sys.exit(1)
    X = np.load(sys.argv[1], mmap_mode="r")
    print(vendi_cosine(X))
