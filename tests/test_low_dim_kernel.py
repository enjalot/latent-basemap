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
