"""Experiment C — grouped-negatives QUALITY run (owner plan + overseer spec 2026-09-08). The microbenchmark showed
grouped negatives (4 tails/unique source) give 1.42x throughput, but grouping changes within-batch negative
correlations so it is NOT gradient-neutral (unlike the plain unique dedup). This settles the only remaining
question: does grouped-negative training reach the SAME quality as independent (uniform) negatives?

Two head-based arms, SAME init/seed/horizon, differing ONLY in the negative sampler; both scored on the common
evaluator. PROMOTE only if global recall within 0.005 AND no cohort (esp. tail) loses >0.01 — the changed
within-batch correlation is exactly what shows up in worst-cohort, which the evaluator surfaces. Reuses exp_a's
loss/model/score. Usage: exp_c_quality.py [HORIZON=20000]
"""
import json, sys, time
from pathlib import Path
import numpy as np

SEAL = Path("/data2/monet/eval-common"); GRAPH = Path("/data/latent-basemap/sandbox/eval-common-train/edges-k15-fuzzy.npz")
OUT = Path("/data/latent-basemap/sandbox/exp-c-quality"); BATCH, POS_RATIO = 16384, 0.10
A, B_, NEG_TANH, FNEG_W, FNEG_LO, FNEG_HI, CLIP = 1.9328, 0.7905, 4.0, 1.0, 0.2, 0.8, 1.0


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import eval_common, torch
    dev = "cuda"
    feats = torch.from_numpy(np.asarray(np.load(SEAL / "train_hd.f16.npy"), np.float32)).to(dev)
    n, dim = feats.shape
    g = np.load(GRAPH); src_all = torch.from_numpy(g["sources"].astype(np.int64)).to(dev)
    dst_all = torch.from_numpy(g["targets"].astype(np.int64)).to(dev); n_edges = src_all.numel()
    seal = eval_common._load_seal(); n_pos = int(BATCH * POS_RATIO); n_neg = BATCH - n_pos

    def loss_fn(se, de, tgt):
        delta = (se - de).float(); r2 = delta.square().sum(1); tiny = torch.finfo(r2.dtype).tiny
        radial = torch.where(r2 == 0, torch.zeros_like(r2), r2.clamp_min(tiny).pow(B_))
        qs = torch.nan_to_num(torch.pow(1 + A * radial, -1.0), nan=1e-7, posinf=1 - 1e-7, neginf=1e-7).clamp(1e-7, 1 - 1e-7)
        per = torch.nn.functional.binary_cross_entropy(qs, tgt, reduction="none"); neg = tgt < 0.5
        per = torch.where(neg, NEG_TANH * torch.tanh(per / NEG_TANH), per); w = torch.ones_like(per)
        with torch.no_grad():
            ep = torch.cat([se, de], 0).float(); R = torch.quantile((ep - ep.mean(0)).norm(dim=1), 0.9)
            r2d = (se - de).float().norm(dim=1); band = neg & (r2d >= FNEG_LO * R) & (r2d <= FNEG_HI * R)
        return (per * (w + band.float() * FNEG_W)).sum() / (w + band.float() * FNEG_W).sum()

    def sample_edges(grouped):
        pe = torch.randint(0, n_edges, (n_pos,), device=dev); ps, pd = src_all[pe], dst_all[pe]
        if grouped:                                                          # 4 neg tails per unique neg SOURCE
            ns = torch.randint(0, n, ((n_neg + 3) // 4,), device=dev).repeat_interleave(4)[:n_neg]
        else:
            ns = torch.randint(0, n, (n_neg,), device=dev)
        nd = torch.randint(0, n, (n_neg,), device=dev)
        src = torch.cat([ps, ns]); dst = torch.cat([pd, nd])
        tgt = torch.cat([torch.ones(n_pos, device=dev), torch.zeros(n_neg, device=dev)])
        return src, dst, tgt

    def train_and_score(grouped, label):
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)                # SAME init/seed for both arms
        pu = ParametricUMAP(n_components=2, hidden_dim=2048, n_layers=3, architecture="residual_bottleneck",
                            low_dim_kernel="umap", a=A, b=B_, use_amp=True, device=dev)
        pu._init_model(dim); opt = torch.optim.Adam(pu.model.parameters(), lr=1e-3); pu.model.train()
        t0 = time.time()
        for step in range(horizon):
            src, dst, tgt = sample_edges(grouped); opt.zero_grad(set_to_none=True)
            se, de = pu.model(feats[src]), pu.model(feats[dst])
            loss = loss_fn(se, de, tgt); loss.backward()
            torch.nn.utils.clip_grad_norm_(pu.model.parameters(), CLIP); opt.step()
        torch.cuda.synchronize(); wall = time.time() - t0
        pu.model.eval()
        with torch.no_grad():
            rc = pu.model(torch.from_numpy(seal["ref_hd"]).to(dev)).cpu().numpy().astype(np.float32)
            vc = pu.model(torch.from_numpy(seal["val_hd"]).to(dev)).cpu().numpy().astype(np.float32)
        s = eval_common.score(rc, vc, seal, label); s["wall_s"] = round(wall, 1)
        (OUT / label).mkdir(parents=True, exist_ok=True)                    # PERSIST head (standing rule) so the v2
        pu.save(str(OUT / label / "model.pt")); s["pca_k"] = None           # re-verdict is a CPU rescore, not a retrain
        print(f"[C-qual {label}] {wall:.0f}s B2000 {s['recall@k15_B2000']['micro']} worst {s['recall@k15_B2000']['worst_cohort_recall']}", flush=True)
        return s

    uni = train_and_score(False, "uniform-neg")
    grp = train_and_score(True, "grouped-neg")

    d250 = grp["recall@k15_B250"]["micro"] - uni["recall@k15_B250"]["micro"]
    d2000 = grp["recall@k15_B2000"]["micro"] - uni["recall@k15_B2000"]["micro"]
    cohort_losses = {r: round(uni["recall@k15_B2000"]["per_source"][r] - grp["recall@k15_B2000"]["per_source"][r], 4)
                     for r in uni["recall@k15_B2000"]["per_source"]}         # uniform − grouped (+ = grouped lost)
    worst_cohort_loss = max(cohort_losses.values())
    passes = abs(d2000) <= 0.005 and worst_cohort_loss <= 0.01
    out = {"schema": "exp-c-quality-2026-09-08", "horizon": horizon,
           "protocol": "grouped-neg (4 tails/source) vs uniform-neg, SAME init/seed/horizon, common evaluator",
           "uniform": uni, "grouped": grp,
           "delta_recall_B250": round(d250, 4), "delta_recall_B2000": round(d2000, 4),
           "cohort_loss_uniform_minus_grouped": cohort_losses, "worst_cohort_loss": worst_cohort_loss,
           "verdict": ("PROMOTE: grouped negatives preserve quality (Δrecall %.4f within 0.005, worst-cohort loss %.4f ≤ 0.01) "
                       "— the 1.42x throughput compounds into every run" % (d2000, worst_cohort_loss)) if passes
                      else "STOP: grouped negatives harm quality (Δrecall %.4f or worst-cohort loss %.4f)" % (d2000, worst_cohort_loss),
           "note": "settles C: the microbenchmark's 1.42x is real ONLY if grouping doesn't move worst-cohort. gradient "
                   "identity of the plain dedup was already CPU-verified; this tests the grouped sampler's quality."}
    (OUT / "result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
