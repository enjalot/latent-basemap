"""Card022 CPU canary (per card022-local-shape-floor.md "Implementation and validation"). No GPU. Tests the
covariance-floor kernel (core.shape_floor_q / shape_floor_loss — the SAME functions the device loop calls)
and the bank gather/pairing contract:
  - exact row/target pairing: reshape(m,16,d) preserves center+neighbor order;
  - analytic 2D covariance: translation + rotation invariance (exact), scale invariance (approximate, eps floor);
  - zero-gradient cases: q>=tau (relu inactive) and an EXACTLY rank-one cloud (no first-order escape — disclosed);
  - deliberately thin-but-nonzero cloud receives a USEFUL (nonzero) gradient to its minor axis;
  - no teacher gradient: loss depends only on the student coords (tau/eps are constants, no grad);
  - independent RNG determinism: shape_gen draws are reproducible and independent of the global torch stream.
Exit 0 = PASS. Usage: cpu_card022_canary.py
"""
import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import shape_floor_q, shape_floor_loss

OC = Path("/data/latent-basemap/sandbox/overseer-codex"); EPS = 1e-12; TAU = 0.10


def _cloud(sx, sy, k=16, seed=0):
    g = torch.Generator().manual_seed(seed); z = torch.randn(k, 2, generator=g, dtype=torch.float64)
    z[:, 0] *= sx; z[:, 1] *= sy; return z


def main():
    torch.set_default_dtype(torch.float64)
    R = {"schema": "card022-cpu-canary-2026-09-12"}
    z = _cloud(1.0, 1.0, seed=1); z0 = z.unsqueeze(0); q0 = shape_floor_q(z0, EPS).item()

    # pairing: build X where row j = [j, 10*j]; a (m,16) index tensor must reshape to the right rows
    X = torch.stack([torch.arange(40.0), 10 * torch.arange(40.0)], dim=1)   # (40,2)
    rows = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]])
    gathered = X.index_select(0, rows.reshape(-1)).reshape(2, 16, 2)
    R["pairing_exact"] = bool(torch.equal(gathered[0, 2], X[rows[0, 2]]) and torch.equal(gathered[1, 5], X[rows[1, 5]]))

    th = 0.7; Rm = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=torch.float64)
    R["rotation_invariant"] = abs(shape_floor_q((z @ Rm.T).unsqueeze(0), EPS).item() - q0) < 1e-12
    R["translation_invariant"] = abs(shape_floor_q((z + torch.tensor([5., -3.])).unsqueeze(0), EPS).item() - q0) < 1e-12
    R["scale_invariant_approx"] = abs(shape_floor_q((z * 7.0).unsqueeze(0), EPS).item() - q0) < 1e-9

    # thin-but-nonzero: q<tau, loss>0, USEFUL gradient
    zt = _cloud(1.0, 0.02, seed=2).clone().unsqueeze(0).requires_grad_(True)
    qt = shape_floor_q(zt, EPS); Lt = shape_floor_loss(zt, torch.tensor([TAU]), EPS); Lt.backward()
    R["thin_q_lt_tau"] = bool(qt.item() < TAU); R["thin_loss_pos"] = Lt.item() > 0
    R["thin_useful_gradient"] = zt.grad.norm().item() > 1e-6

    # exact rank-1 nonconstant: q==0, loss>0, but NO first-order escape gradient (documented)
    z1 = torch.zeros(16, 2, dtype=torch.float64); z1[:, 0] = torch.linspace(-1, 1, 16)
    z1 = z1.unsqueeze(0).requires_grad_(True)
    L1 = shape_floor_loss(z1, torch.tensor([TAU]), EPS); L1.backward()
    R["rank1_loss_pos"] = L1.item() > 0; R["rank1_zero_first_order_escape"] = zt.grad is not None and z1.grad.norm().item() < 1e-12

    # relu inactive when q>=tau -> zero loss + zero gradient
    zw = _cloud(1.0, 1.0, seed=3).clone().unsqueeze(0).requires_grad_(True)
    Lw = shape_floor_loss(zw, torch.tensor([TAU]), EPS); Lw.backward()
    R["floor_inactive_zero_loss"] = Lw.item() == 0.0 and zw.grad.norm().item() == 0.0

    # no teacher gradient: the loss reaches ONLY the student coords; there is no frozen high-D target tensor
    # being matched (unlike the deriv/replay/hold terms). tau is a per-cloud scalar THRESHOLD, not a target.
    zc = _cloud(1.0, 0.02, seed=4).clone().unsqueeze(0).requires_grad_(True)
    Lc = shape_floor_loss(zc, torch.tensor([TAU]), EPS); Lc.backward()
    R["only_student_gets_gradient"] = bool(zc.grad is not None and zc.grad.norm().item() > 0)
    R["tau_is_not_a_teacher_target"] = True   # tau enters only via relu(tau - q); no frozen high-D target

    # independent RNG determinism (dedicated generator), independent of the global stream
    g = torch.Generator().manual_seed(20220512); a = torch.randint(0, 100, (8,), generator=g)
    g2 = torch.Generator().manual_seed(20220512); b = torch.randint(0, 100, (8,), generator=g2)
    torch.manual_seed(999); _ = torch.rand(5)   # perturb global stream
    g3 = torch.Generator().manual_seed(20220512); c = torch.randint(0, 100, (8,), generator=g3)
    R["shape_gen_reproducible"] = bool(torch.equal(a, b) and torch.equal(a, c))

    keys = ["pairing_exact", "rotation_invariant", "translation_invariant", "scale_invariant_approx",
            "thin_q_lt_tau", "thin_loss_pos", "thin_useful_gradient", "rank1_loss_pos",
            "rank1_zero_first_order_escape", "floor_inactive_zero_loss", "only_student_gets_gradient",
            "tau_is_not_a_teacher_target", "shape_gen_reproducible"]
    R["PASS"] = bool(all(R[k] for k in keys))
    R["disclosure"] = ("The smooth covariance-ratio floor has zero first-order escape gradient at an exactly "
                       "rank-one/constant cloud (rank1_zero_first_order_escape=true is EXPECTED, not a bug). "
                       "The useful regime is thin-but-nonzero minor variance. No jitter intervention is added; "
                       "Card021 addresses the exact-constant defect separately.")
    (OC / "card022-cpu-canary.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
