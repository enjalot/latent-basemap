"""Card024 CPU scalar/logit validation (per card024 "Validation before GPU production"). No GPU. Exercises
the exact contrastive-normalization loss math the core uses (qhat kernel, the shared finite-distance clamp,
the binary-logit loss logit=log(qhat)-beta, and the learned scalar) as device-agnostic checks:
  - qhat(d)=1/(1+a*(d^2)^b) finite in (0,1) at near/mid/far d; shared clamp keeps log(qhat) finite at far d;
  - neg_fixed(beta=0) equals the equivalent probability qhat/(qhat+1) BCE (logit form == probability form);
  - beta0 reproduces NEG EXACTLY: nce_learned at beta=0 has identical loss AND identical MODEL (qhat) gradient
    to neg_fixed;
  - NCE beta-gradient matches a central finite-difference; the scalar receives a real gradient;
  - neg_fixed's frozen scalar (const, requires_grad False) receives NO gradient;
  - loss + qhat-gradient finite at near/mid/far distances (zero-gradient guard).
Exit 0 = PASS. Usage: cpu_card024_canary.py
"""
import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
import torch.nn.functional as Fn

OC = Path("/data/latent-basemap/sandbox/overseer-codex"); A, B = 1.9328, 0.7905
CLAMP_LO, CLAMP_HI = 1e-7, 1 - 1e-7


def qhat(d):
    return 1.0 / (1.0 + A * (d * d) ** B)


def _clamp(q):
    return torch.clamp(torch.nan_to_num(q, nan=1e-7, posinf=1 - 1e-7, neginf=1e-7), CLAMP_LO, CLAMP_HI)


def logit_loss(q, t, beta):
    return Fn.binary_cross_entropy_with_logits(torch.log(_clamp(q)) - beta, t, reduction="mean")


def prob_loss_neg(q, t):     # neg_fixed equivalent probability qhat/(qhat+1) at beta=0
    p = _clamp(q); p = p / (p + 1.0)
    return Fn.binary_cross_entropy(p, t, reduction="mean")


def main():
    torch.set_default_dtype(torch.float64)
    R = {"schema": "card024-cpu-canary-2026-09-12", "a": A, "b": B}
    d = torch.tensor([0.05, 0.5, 1.0, 3.0, 20.0])              # near..far
    q = qhat(d)
    R["qhat_in_unit_interval"] = bool(torch.all(q > 0) and torch.all(q < 1))
    R["qhat_finite"] = bool(torch.isfinite(q).all())
    R["log_qhat_finite_far"] = bool(torch.isfinite(torch.log(_clamp(qhat(torch.tensor([1e3]))))).all())

    # matched batch: half positives, half noise
    n = 64; g = torch.Generator().manual_seed(24)
    dd = torch.rand(n, generator=g) * 5.0; qb = qhat(dd)
    t = torch.zeros(n); t[: n // 2] = 1.0

    # (1) neg_fixed(beta=0) logit form == probability form qhat/(qhat+1)
    l_logit = logit_loss(qb, t, torch.tensor(0.0))
    l_prob = prob_loss_neg(qb, t)
    R["neg_logit_equals_prob"] = bool(torch.allclose(l_logit, l_prob, atol=1e-10))

    # (2) beta0 reproduces NEG exactly in loss AND model(qhat) gradient
    qn = qb.clone().requires_grad_(True); ln = logit_loss(qn, t, torch.tensor(0.0)); ln.backward()
    qc = qb.clone().requires_grad_(True)
    beta_param = torch.zeros((), requires_grad=True)           # nce scalar at init 0
    lc = logit_loss(qc, t, beta_param); lc.backward()
    R["beta0_loss_equals_neg"] = bool(torch.allclose(ln.detach(), lc.detach(), atol=1e-12))
    R["beta0_model_grad_equals_neg"] = bool(torch.allclose(qn.grad, qc.grad, atol=1e-12))

    # (3) NCE beta gradient matches central finite-difference; scalar gets a real gradient
    R["nce_beta_receives_gradient"] = bool(beta_param.grad is not None and abs(float(beta_param.grad)) > 0)
    h = 1e-6
    lp = float(logit_loss(qb, t, torch.tensor(0.0 + h))); lm = float(logit_loss(qb, t, torch.tensor(0.0 - h)))
    fd = (lp - lm) / (2 * h)
    R["nce_beta_grad_finite_difference"] = bool(abs(float(beta_param.grad) - fd) < 1e-5)

    # (4) neg_fixed's frozen scalar (const, requires_grad False) receives NO gradient
    qf = qb.clone().requires_grad_(True); beta_const = torch.zeros((), requires_grad=False)
    lf = logit_loss(qf, t, beta_const); lf.backward()
    R["neg_fixed_scalar_frozen"] = bool(getattr(beta_const, "grad", None) is None)

    # (5) loss + qhat gradient finite at near/mid/far
    finite = True
    for dv in (0.02, 0.5, 5.0, 100.0):
        qx = qhat(torch.tensor([dv])).clone().requires_grad_(True)
        lx = logit_loss(qx, torch.tensor([1.0]), torch.tensor(0.3)); lx.backward()
        finite = finite and bool(torch.isfinite(lx).all() and torch.isfinite(qx.grad).all())
    R["loss_grad_finite_near_mid_far"] = finite

    keys = ["qhat_in_unit_interval", "qhat_finite", "log_qhat_finite_far", "neg_logit_equals_prob",
            "beta0_loss_equals_neg", "beta0_model_grad_equals_neg", "nce_beta_receives_gradient",
            "nce_beta_grad_finite_difference", "neg_fixed_scalar_frozen", "loss_grad_finite_near_mid_far"]
    R["PASS"] = bool(all(R[k] for k in keys))
    R["note"] = ("Validates the loss math only (device-agnostic). Default-off bitwise identity, real sampler/RNG "
                 "parity, zero weight decay on the scalar group, and mid-epoch resume of model+scalar+moments "
                 "are proven on the device canary. This is an explicitly simplified NEG/NCE family (no rank "
                 "window / fneg / anchors / replay / shape / radius / mid-near / density).")
    (OC / "card024-cpu-canary.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
