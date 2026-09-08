"""Experiment C — GPU throughput microbenchmark (owner plan 2026-09-08, the ≤30min profiler gate). Times the
PRODUCTION model forward+backward under four endpoint-forwarding strategies on a real substrate, to see whether
the gradient-neutral dedup (CPU-verified in exp_c_verify.py, PASS) actually buys throughput on the 5090:
  (a) direct         — model(src) + model(dst)              (2*B endpoint forwards)         [baseline]
  (b) fused          — model(cat(src,dst))                  (2*B in one call)               [current fused path]
  (c) unique         — model(unique(cat(src,dst))) + gather (dedup pair endpoints)          [exact-pair reuse control]
  (d) grouped+unique — 4 neg tails per neg SOURCE, then unique+gather                        [the 1.51x-bound test]
NO quality claim (identity already licensed the gradient); THROUGHPUT ONLY. Promote to a C quality run only at a
measured >=1.25x time-to-quality; else STOP. Uses the sealed eval-common train subset (real DINO-1536), champion
model (residual_bottleneck 2048), champion loss shape, batch 16384, pos_ratio 0.10. Usage: exp_c_microbench.py
"""
import json, time
from pathlib import Path
import numpy as np

SEAL = Path("/data2/monet/eval-common"); OUT = Path("/data/latent-basemap/sandbox/exp-c-microbench.json")
BATCH, POS_RATIO, STEPS, WARMUP, K = 16384, 0.10, 220, 20, 15
A, B_, NEG_TANH, RANKNEG_SCALE, FNEG_W, FNEG_LO, FNEG_HI = 1.9328, 0.7905, 4.0, 0.5 ** 0.75, 1.0, 0.2, 0.8


def main():
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch
    dev = "cuda"; torch.manual_seed(42)
    feats = torch.from_numpy(np.asarray(np.load(SEAL / "train_hd.f16.npy"), np.float16)).to(dev)   # 500K x1536
    n = feats.shape[0]; dim = feats.shape[1]
    pu = ParametricUMAP(n_components=2, hidden_dim=2048, n_layers=3, architecture="residual_bottleneck",
                        low_dim_kernel="umap", a=A, b=B_, use_amp=True, device=dev)
    pu._init_model(dim); model = pu.model; model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_pos = int(BATCH * POS_RATIO); n_neg = BATCH - n_pos

    def loss_fn(se, de, tgt):
        delta = (se - de).float(); r2 = delta.square().sum(1)
        tiny = torch.finfo(r2.dtype).tiny
        radial = torch.where(r2 == 0, torch.zeros_like(r2), r2.clamp_min(tiny).pow(B_))
        qs = torch.pow(1 + A * radial, -1.0).clamp(1e-7, 1 - 1e-7)
        per = torch.nn.functional.binary_cross_entropy(qs, tgt, reduction="none"); neg = tgt < 0.5
        per = torch.where(neg, NEG_TANH * torch.tanh(per / NEG_TANH), per)
        w = torch.where(neg, torch.full_like(per, RANKNEG_SCALE), torch.ones_like(per))
        with torch.no_grad():
            ep = torch.cat([se, de], 0).float(); R = torch.quantile((ep - ep.mean(0)).norm(dim=1), 0.9)
            r2d = (se - de).float().norm(dim=1); band = neg & (r2d >= FNEG_LO * R) & (r2d <= FNEG_HI * R)
        w = w + band.float() * FNEG_W
        return (per * w).sum() / w.sum()

    def make_batch(grouped):
        pos_s = torch.randint(0, n, (n_pos,), device=dev); pos_d = (pos_s + torch.randint(1, n, (n_pos,), device=dev)) % n
        if grouped:                                                          # 4 neg tails per unique neg source
            ns = torch.randint(0, n, ((n_neg + 3) // 4,), device=dev).repeat_interleave(4)[:n_neg]
        else:
            ns = torch.randint(0, n, (n_neg,), device=dev)
        nd = torch.randint(0, n, (n_neg,), device=dev)
        src = torch.cat([pos_s, ns]); dst = torch.cat([pos_d, nd])
        tgt = torch.cat([torch.ones(n_pos, device=dev), torch.zeros(n_neg, device=dev)])
        return src, dst, tgt

    def run(variant):
        grouped = variant == "grouped+unique"
        uniq_counts = []; torch.cuda.reset_peak_memory_stats(); t_steady = 0.0
        for step in range(STEPS):
            src, dst, tgt = make_batch(grouped)
            if step == WARMUP: torch.cuda.synchronize(); t0 = time.time()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):               # forward mixed-precision (like production)
                if variant == "direct":
                    se, de = model(feats[src]), model(feats[dst])
                elif variant == "fused":
                    both = model(torch.cat([feats[src], feats[dst]], 0)); se, de = both[:BATCH], both[BATCH:]
                else:                                                        # unique / grouped+unique
                    ids, inv = torch.unique(torch.cat([src, dst]), return_inverse=True)
                    z = model(feats[ids]); se, de = z[inv[:BATCH]], z[inv[BATCH:]]
                    uniq_counts.append(ids.numel())
            loss = loss_fn(se, de, tgt)                                      # BCE in fp32 OUTSIDE autocast (unsafe under it)
            loss.backward(); opt.step()
        torch.cuda.synchronize(); t_steady = time.time() - t0
        its = (STEPS - WARMUP) / t_steady
        return {"it_per_s": round(its, 1), "steady_s": round(t_steady, 2),
                "mean_unique_endpoints": int(np.mean(uniq_counts)) if uniq_counts else 2 * BATCH,
                "dedup_ratio": round(2 * BATCH / np.mean(uniq_counts), 3) if uniq_counts else 1.0,
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}

    res = {}
    for v in ["direct", "fused", "unique", "grouped+unique"]:
        res[v] = run(v); print(f"[c-bench] {v}: {res[v]['it_per_s']} it/s | dedup {res[v]['dedup_ratio']}x | {res[v]['peak_gb']} GB", flush=True)
    base = res["direct"]["it_per_s"]
    for v in res: res[v]["speedup_vs_direct"] = round(res[v]["it_per_s"] / base, 3)
    best = max(res, key=lambda v: res[v]["it_per_s"])
    out = {"schema": "exp-c-microbench-2026-09-08", "kind": "THROUGHPUT ONLY (gradient identity already CPU-verified, PASS)",
           "substrate": "eval-common train 500K DINO-1536", "model": "residual_bottleneck hidden2048",
           "batch": BATCH, "pos_ratio": POS_RATIO, "steps": STEPS, "warmup": WARMUP,
           "variants": res, "fastest": best, "fastest_speedup": res[best]["speedup_vs_direct"],
           "gate": ">=1.25x time-to-quality to promote to a C quality run; else STOP",
           "verdict": ("PROMOTE-candidate: %s at %.2fx" % (best, res[best]["speedup_vs_direct"])) if res[best]["speedup_vs_direct"] >= 1.25
                      else "STOP: best %s only %.2fx < 1.25x — dedup/gather overhead consumes the saving at this batch/negative mix" % (best, res[best]["speedup_vs_direct"])}
    OUT.write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
