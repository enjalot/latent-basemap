"""Experiment B — compress the input, preserve the problem (owner plan 2026-09-08, 3h cap). Separates "information
discarded by the network INPUT" from "a changed neighborhood OBJECTIVE": all arms train the head against the SAME
original-D (1536) graph and are scored on the common evaluator against the SAME original-D truth; ONLY the head's
INPUT changes (full-1536 / PCA768 / PCA384). PCA is fit on TRAIN rows only, then applied (+renorm) to train/ref/val.
Output size, hidden arch, recipe, horizon, sampler all fixed; input size necessarily changes first-layer params.

Promote (plan): >=20% reduction in time-to-quality OR a useful residency-threshold crossing, with global recall
within 0.005 AND no cohort loss >0.01. Stop if the advantage exists only against PCA's own truth. Usage:
  exp_b_compress.py [HORIZON=20000]
"""
import json, sys, time
from pathlib import Path
import numpy as np

SEAL = Path("/data2/monet/eval-common"); GRAPH = Path("/data/latent-basemap/sandbox/eval-common-train/edges-k15-fuzzy.npz")
OUT = Path("/data/latent-basemap/sandbox/exp-b"); BATCH, POS_RATIO = 16384, 0.10
A, B_, NEG_TANH, FNEG_W, FNEG_LO, FNEG_HI, CLIP = 1.9328, 0.7905, 4.0, 1.0, 0.2, 0.8, 1.0


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import eval_common, torch
    dev = "cuda"
    train_hd = torch.from_numpy(np.asarray(np.load(SEAL / "train_hd.f16.npy"), np.float32)).to(dev)
    ref_hd = torch.from_numpy(np.asarray(np.load(SEAL / "ref_hd.f16.npy"), np.float32)).to(dev)
    val_hd = torch.from_numpy(np.asarray(np.load(SEAL / "val_hd.f16.npy"), np.float32)).to(dev)
    n = train_hd.shape[0]
    g = np.load(GRAPH); src_all = torch.from_numpy(g["sources"].astype(np.int64)).to(dev)
    dst_all = torch.from_numpy(g["targets"].astype(np.int64)).to(dev); n_edges = src_all.numel()
    seal = eval_common._load_seal(); n_pos = int(BATCH * POS_RATIO); n_neg = BATCH - n_pos

    # PCA fit on TRAIN only (exact eigh of the 1536x1536 covariance)
    with torch.no_grad():
        mu = train_hd.mean(0); Xc = train_hd - mu
        cov = (Xc.T @ Xc) / n; cov = (cov + cov.T) / 2
        evals, evecs = torch.linalg.eigh(cov)                               # ascending
        comp = evecs.flip(1)                                                # descending; comp[:, :k] = top-k
    def make_tf(k):
        if k is None:
            return (lambda X: torch.nn.functional.normalize(X, dim=1)), train_hd.shape[1]
        Ck = comp[:, :k].contiguous()
        return (lambda X: torch.nn.functional.normalize((X - mu) @ Ck, dim=1)), k

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

    def sample_edges():
        pe = torch.randint(0, n_edges, (n_pos,), device=dev); ps, pd = src_all[pe], dst_all[pe]
        ns = torch.randint(0, n, (n_neg,), device=dev); nd = torch.randint(0, n, (n_neg,), device=dev)
        return torch.cat([ps, ns]), torch.cat([pd, nd]), torch.cat([torch.ones(n_pos, device=dev), torch.zeros(n_neg, device=dev)])

    def run(k, label):
        tf, in_dim = make_tf(k)
        with torch.no_grad():
            tr = tf(train_hd).contiguous(); rf = tf(ref_hd).contiguous(); vl = tf(val_hd).contiguous()
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)               # same init/seed; first-layer dim differs by design
        pu = ParametricUMAP(n_components=2, hidden_dim=2048, n_layers=3, architecture="residual_bottleneck",
                            low_dim_kernel="umap", a=A, b=B_, use_amp=True, device=dev)
        pu._init_model(in_dim); opt = torch.optim.Adam(pu.model.parameters(), lr=1e-3); pu.model.train()
        t0 = time.time()
        for step in range(horizon):
            src, dst, tgt = sample_edges(); opt.zero_grad(set_to_none=True)
            se, de = pu.model(tr[src]), pu.model(tr[dst])
            loss = loss_fn(se, de, tgt); loss.backward()
            torch.nn.utils.clip_grad_norm_(pu.model.parameters(), CLIP); opt.step()
        torch.cuda.synchronize(); wall = time.time() - t0
        pu.model.eval()
        with torch.no_grad():
            rc = pu.model(rf).cpu().numpy().astype(np.float32); vc = pu.model(vl).cpu().numpy().astype(np.float32)
        s = eval_common.score(rc, vc, seal, label); s["wall_s"] = round(wall, 1); s["in_dim"] = in_dim
        (OUT / label).mkdir(parents=True, exist_ok=True)                    # PERSIST the head + PCA k so this arm can be
        pu.save(str(OUT / label / "model.pt")); s["pca_k"] = k              # rescored against a re-sealed eval set (no retrain)
        print(f"[B {label}] in_dim {in_dim} {wall:.0f}s B2000 {s['recall@k15_B2000']['micro']} worst {s['recall@k15_B2000']['worst_cohort_recall']}", flush=True)
        return s

    arms = {"full-1536": run(None, "full-1536"), "pca768": run(768, "pca768"), "pca384": run(384, "pca384")}
    base = arms["full-1536"]; br = base["recall@k15_B2000"]["micro"]
    out = {"schema": "exp-b-compress-2026-09-08", "horizon": horizon,
           "protocol": "full/PCA768/PCA384 INPUT, all vs the SAME original-D graph, scored on the common evaluator "
                       "(original-D 1536 truth). PCA fit on train only. Isolates input-info loss from objective change.",
           "arms": arms}
    for a in arms:
        r = arms[a]; out[a] = {"in_dim": r["in_dim"], "B2000": r["recall@k15_B2000"]["micro"],
                               "worst_cohort": r["recall@k15_B2000"]["worst_cohort_recall"], "wall_s": r["wall_s"],
                               "delta_vs_full": round(r["recall@k15_B2000"]["micro"] - br, 4),
                               "worst_cohort_loss_vs_full": round(base["recall@k15_B2000"]["worst_cohort_recall"] - r["recall@k15_B2000"]["worst_cohort_recall"], 4)}
    def ok(a): return abs(out[a]["delta_vs_full"]) <= 0.005 and out[a]["worst_cohort_loss_vs_full"] <= 0.01
    out["pca768_preserves"] = ok("pca768"); out["pca384_preserves"] = ok("pca384")
    out["verdict"] = ("PCA768 preserves original geometry (Δ %.4f, worst-cohort loss %.4f) at in_dim 768 vs 1536 — "
                      "input compressible; %s" % (out["pca768"]["delta_vs_full"], out["pca768"]["worst_cohort_loss_vs_full"],
                      "PCA384 also preserves" if ok("pca384") else "PCA384 does NOT (stop at 768)")) if ok("pca768") \
                     else "PCA768 does NOT preserve original geometry (Δ %.4f) — input carries geometry the compression discards" % out["pca768"]["delta_vs_full"]
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("full-1536", "pca768", "pca384", "verdict")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
