"""P0.1: low_dim_kernel switch — value + gradient tests for legacy_lp vs umap
at b in {1.0, 0.895}, zero distance, and large distance."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP


def _qs(kernel, a, b, delta):
    m = ParametricUMAP(a=a, b=b, low_dim_kernel=kernel, device='cpu')
    src = torch.zeros(1, len(delta))
    dst = torch.tensor([delta], dtype=torch.float32)
    return m._low_dim_qs(src, dst)[0]


def test_values_b1():
    # delta=(3,4): r=5, r2=25. legacy: 1/(1+1*5)=1/6; umap: 1/(1+1*25)=1/26
    a, b, d = 1.0, 1.0, [3.0, 4.0]
    assert abs(_qs('legacy_lp', a, b, d).item() - 1/6) < 1e-6
    assert abs(_qs('umap', a, b, d).item() - 1/26) < 1e-6


def test_values_b0895():
    # b=0.895, delta=(3,4). legacy: norm p=2b=1.79 of (3,4); umap: (r2)^b = 25^0.895
    a, b, d = 1.0, 0.895, [3.0, 4.0]
    legacy_norm = (3.0**1.79 + 4.0**1.79) ** (1/1.79)
    assert abs(_qs('legacy_lp', a, b, d).item() - 1/(1 + legacy_norm)) < 1e-5
    umap_radial = 25.0 ** 0.895
    assert abs(_qs('umap', a, b, d).item() - 1/(1 + umap_radial)) < 1e-5
    # they must differ (the whole point of P0.1)
    assert abs(_qs('legacy_lp', a, b, d).item() - _qs('umap', a, b, d).item()) > 1e-3


def test_zero_distance():
    # both kernels -> q=1 at zero distance, finite gradient
    for k in ('legacy_lp', 'umap'):
        m = ParametricUMAP(a=1.0, b=1.0, low_dim_kernel=k, device='cpu')
        src = torch.zeros(1, 2, requires_grad=True)
        dst = torch.zeros(1, 2, requires_grad=True)
        q, _ = m._low_dim_qs(src, dst)
        assert abs(q.item() - 1.0) < 1e-6
        q.sum().backward()
        assert torch.isfinite(src.grad).all(), f"{k}: non-finite grad at zero dist"


def test_large_distance_and_grad():
    for k in ('legacy_lp', 'umap'):
        m = ParametricUMAP(a=1.0, b=1.0, low_dim_kernel=k, device='cpu')
        src = torch.zeros(1, 2, requires_grad=True)
        dst = torch.tensor([[1e3, 1e3]], requires_grad=True)
        q, _ = m._low_dim_qs(src, dst)
        assert 0.0 <= q.item() < 1e-3, f"{k}: q should -> 0 at large dist"
        q.sum().backward()
        assert torch.isfinite(src.grad).all() and torch.isfinite(dst.grad).all()


def test_gradient_matches_autograd_umap():
    # umap q=1/(1+a r2); dq/dr2 = -a/(1+a r2)^2. Check autograd via r2.
    a = 1.0
    src = torch.zeros(1, 2)
    dst = torch.tensor([[0.6, 0.8]], requires_grad=True)   # r2=1.0
    m = ParametricUMAP(a=a, b=1.0, low_dim_kernel='umap', device='cpu')
    q, _ = m._low_dim_qs(src, dst)
    q.backward()
    # analytic dq/ddst = dq/dr2 * dr2/ddst = (-a/(1+a)^2) * 2*(dst-src)
    r2 = 1.0
    dqdr2 = -a / (1 + a*r2)**2
    expected = dqdr2 * 2 * torch.tensor([[0.6, 0.8]])
    assert torch.allclose(dst.grad, expected, atol=1e-6), (dst.grad, expected)


if __name__ == '__main__':
    for fn in [test_values_b1, test_values_b0895, test_zero_distance,
               test_large_distance_and_grad, test_gradient_matches_autograd_umap]:
        fn(); print(f"PASS {fn.__name__}")
    print("ALL KERNEL TESTS PASSED")


# ── P0-A regression: zero-distance safety at the standard b≈0.895 ────────────────
B_STD = 0.895060879


def test_umap_zero_distance_grad_finite_b_std():
    m = ParametricUMAP(a=1.5769, b=B_STD, low_dim_kernel='umap', device='cpu')
    src = torch.zeros(4, 2, requires_grad=True); dst = torch.zeros(4, 2, requires_grad=True)
    q, _ = m._low_dim_qs(src, dst)
    assert torch.allclose(q, torch.ones_like(q))
    q.sum().backward()
    assert torch.isfinite(src.grad).all() and torch.isfinite(dst.grad).all()
    assert torch.allclose(src.grad, torch.zeros_like(src.grad))   # zero grad at equality


def test_umap_small_nonzero_grad_finite_and_curved_b_std():
    m = ParametricUMAP(a=1.5769, b=B_STD, low_dim_kernel='umap', device='cpu')
    for r in (1e-4, 1e-2, 0.5, 2.0):
        src = torch.zeros(1, 2); dst = torch.tensor([[r, 0.0]], requires_grad=True)
        q, radial = m._low_dim_qs(src, dst)
        q.backward()
        assert torch.isfinite(dst.grad).all()
        # radial follows r^(2b), not flattened
        assert abs(radial.item() - (r * r) ** B_STD) < 1e-5 * max(1.0, (r*r)**B_STD)


def test_umap_self_edges_end_to_end_finite_b_std():
    # a batch with exact self-edges (dup rows) must leave params + coords finite.
    import numpy as np
    n, d = 400, 8
    X = np.random.RandomState(3).randn(n, d).astype(np.float32)
    X[10:20] = X[10]                       # exact duplicates (self-edge-like)
    src = np.array([10, 11, 12, 13, 14, 0, 1, 2] * 200, dtype=np.int32)
    dst = np.array([10, 11, 12, 13, 14, 1, 2, 3] * 200, dtype=np.int32)  # incl. i==i self-edges
    w = np.ones(len(src), dtype=np.float32)
    np.savez('/tmp/_pa_edges.npz', sources=src, targets=dst, weights=w, n_nodes=n, k=15)
    m = ParametricUMAP(a=1.5769, b=B_STD, low_dim_kernel='umap', correlation_weight=0.0,
                       n_epochs=2, batch_size=64, total_steps_estimate=200, lr_schedule='cosine',
                       device='cpu', positive_target_mode='binary', gpu_resident_data=False, use_amp=False)
    m.fit(X, precomputed_edges_path='/tmp/_pa_edges.npz')
    for p in m.model.parameters():
        assert torch.isfinite(p).all(), "non-finite parameter after self-edge training"
    Z = m.transform(X)
    assert np.isfinite(Z).all(), "non-finite coords after self-edge training"


if __name__ == '__main__':
    for fn in [test_values_b1, test_values_b0895, test_zero_distance,
               test_large_distance_and_grad, test_gradient_matches_autograd_umap,
               test_umap_zero_distance_grad_finite_b_std,
               test_umap_small_nonzero_grad_finite_and_curved_b_std,
               test_umap_self_edges_end_to_end_finite_b_std]:
        fn(); print(f"PASS {fn.__name__}")
    print("ALL KERNEL TESTS PASSED")
