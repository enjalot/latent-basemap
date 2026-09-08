"""Experiment C — CPU gradient-identity verification vs the PRODUCTION loss (owner plan 2026-09-08, gate before any
GPU microbenchmark). Extends the plan's 8e-18 fixture to the EXACT production per-batch loss (core.py _low_dim_qs
umap kernel + BCE + neg_tanh cap + rankneg (2W/N)^0.75 scale + fneg mid-range-band up-weight + total-weight
normalization), and checks that forwarding each UNIQUE endpoint once + gathering back to pair order preserves the
parameter gradient. Pointwise MLP, float64, BN/dropout OFF (those invalidate the identity). No speed/convergence
claim follows — this only licenses the dedup as gradient-neutral so the GPU microbenchmark measures throughput,
not a changed objective. Usage: exp_c_verify.py
"""
import copy, json
import torch
import torch.nn.functional as F

torch.set_num_threads(2); torch.manual_seed(20260908); DT = torch.float64
A, B, NEG_TANH, RANKNEG_SCALE, FNEG_W, FNEG_LO, FNEG_HI = 1.9328, 0.7905, 4.0, 0.5 ** 0.75, 1.0, 0.2, 0.8  # md000 champion


def production_loss(src_emb, dst_emb, targets):
    """EXACT replica of core.py's per-element weighted UMAP loss (the _per_elem_loss path)."""
    delta = src_emb - dst_emb; r2 = delta.square().sum(1)
    tiny = torch.finfo(r2.dtype).tiny
    radial = torch.where(r2 == 0, torch.zeros_like(r2), r2.clamp_min(tiny).pow(B))     # umap kernel radial, r2=0 guard
    qs = torch.pow(1 + A * radial, -1.0).clamp(1e-7, 1 - 1e-7)
    per = F.binary_cross_entropy(qs, targets, reduction="none")
    neg = targets < 0.5
    per = torch.where(neg, NEG_TANH * torch.tanh(per / NEG_TANH), per)                 # capped negative loss
    w = torch.ones_like(per)
    w = torch.where(neg, w * RANKNEG_SCALE, w)                                         # rankneg (2W/N)^0.75
    with torch.no_grad():                                                              # detached batch geometry (fog)
        endpts = torch.cat([src_emb, dst_emb], 0); center = endpts.mean(0)
        R = torch.quantile((endpts - center).norm(dim=1), 0.9)
        r2d = (src_emb - dst_emb).norm(dim=1)
        band = neg & (r2d >= FNEG_LO * R) & (r2d <= FNEG_HI * R)
    w = w + band.to(DT) * FNEG_W
    return (per * w).sum() / w.sum()


def main():
    x = torch.randn(512, 32, dtype=DT)
    mlp = lambda: torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.SiLU(), torch.nn.Linear(64, 2)).to(DT)
    src = torch.randint(0, len(x), (4096,)); dst = (src + torch.randint(1, len(x), (4096,))) % len(x)
    targets = (torch.arange(4096) < 410).to(DT)                                        # ~10% positive

    direct = mlp(); unique = copy.deepcopy(direct)                                    # identical init
    # (a) direct: forward each pair endpoint
    ld = production_loss(direct(x[src]), direct(x[dst]), targets); ld.backward()
    # (b) unique: forward each UNIQUE endpoint once, gather back to pair order
    ids, inverse = torch.unique(torch.cat([src, dst]), return_inverse=True)
    z = unique(x[ids])
    lu = production_loss(z[inverse[:len(src)]], z[inverse[len(src):]], targets); lu.backward()

    gmax = max((a.grad - b.grad).abs().max().item() for a, b in zip(direct.parameters(), unique.parameters()))
    lmax = abs(ld.item() - lu.item())
    result = {"schema": "exp-c-gradient-identity-2026-09-08", "device": "cpu", "dtype": str(DT),
              "rows": len(x), "pairs": len(src), "unique_endpoints": int(len(ids)),
              "dedup_ratio": round(2 * len(src) / len(ids), 3),
              "loss_direct": ld.item(), "loss_unique": lu.item(), "loss_abs_diff": lmax,
              "max_param_grad_abs_diff": gmax, "tol": 1e-12,
              "PASS": bool(gmax < 1e-12 and lmax < 1e-12),
              "kernel": "umap a=%.4f b=%.4f" % (A, B), "loss_terms": "BCE + neg_tanh(%.1f) + rankneg_scale(%.4f) + fneg_band(%.1f) + total-weight-norm" % (NEG_TANH, RANKNEG_SCALE, FNEG_W),
              "note": "gradient identity of unique-endpoint dedup vs the PRODUCTION loss; licenses the dedup as "
                      "gradient-neutral. Pointwise MLP, BN/dropout OFF. No speed claim."}
    print(json.dumps(result, indent=1))
    Path = __import__("pathlib").Path
    Path("/data/latent-basemap/sandbox/exp-c-gradient-identity.json").write_text(json.dumps(result, indent=1))
    return 0 if result["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
