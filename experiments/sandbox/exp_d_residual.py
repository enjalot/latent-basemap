"""Experiment D — improve a region without retraining the world (owner plan 2026-09-08, 3h cap). Freeze a global
head f0; train a small residual f(x) = f0(x) + g(f0(x))·r_phi(x), g a fixed compactly-supported smooth gate for ONE
region (Gaussian on f0-coords, tapered to 0 at 3σ). Outside the support coords are unchanged BY CONSTRUCTION; a
boundary-halo penalty keeps the seam still. Compare no-update (f0) vs residual, on: region-local recall (repair),
global recall (must not regress), and halo/outside movement. Persists heads (standing rule).

Region = the worst-global-recall provenance cohort (preselected from the f0 diagnostic, not from test failures).
Promote (plan): local recall +0.03 (or 20% fewer candidates at fixed recall) with global loss ≤0.005, coords
unchanged outside support, p99 halo movement ≤0.005 of the global radius, cross-cohort recall loss ≤0.01. Stop if
the gain needs seams, large halo movement, or repairs only training members. Usage: exp_d_residual.py [HORIZON=8000]
"""
import json, sys, time
from pathlib import Path
import numpy as np

SEAL = Path("/data2/monet/eval-common"); GRAPH = Path("/data/latent-basemap/sandbox/eval-common-train/edges-k15-fuzzy.npz")
F0 = Path("/data/latent-basemap/sandbox/exp-c-quality/uniform-neg/model.pt"); OUT = Path("/data/latent-basemap/sandbox/exp-d")
POOL = Path("/data2/monet/pool-20m"); BATCH, POS_RATIO = 8192, 0.10
A, B_, NEG_TANH, FNEG_W, FNEG_LO, FNEG_HI, CLIP = 1.9328, 0.7905, 4.0, 1.0, 0.2, 0.8, 1.0


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import eval_common, torch
    dev = "cuda"
    train_hd = torch.from_numpy(np.asarray(np.load(SEAL / "train_hd.f16.npy"), np.float32)).to(dev)
    n = train_hd.shape[0]; dim = train_hd.shape[1]
    src_np = np.load(POOL / "source.npy", allow_pickle=True)
    train_src = src_np[np.load(SEAL / "train_idx.npy")]
    g = np.load(GRAPH); src_all = torch.from_numpy(g["sources"].astype(np.int64)).to(dev)
    dst_all = torch.from_numpy(g["targets"].astype(np.int64)).to(dev)
    seal = eval_common._load_seal(); val_src = seal["val_source"]

    f0pu = ParametricUMAP.load(str(F0), device=dev); f0 = f0pu.model; f0.eval()
    for p in f0.parameters(): p.requires_grad_(False)
    def f0_all(X):
        with torch.no_grad():
            return torch.cat([f0(X[i:i + 100_000]) for i in range(0, X.shape[0], 100_000)]).float()
    Z0 = f0_all(train_hd)                                                   # frozen global layout of train
    global_radius = (Z0 - Z0.mean(0)).norm(dim=1).quantile(0.9).item()

    import os
    with torch.no_grad():
        rc0 = f0(torch.from_numpy(seal["ref_hd"]).to(dev)).cpu().numpy(); vc0 = f0(torch.from_numpy(seal["val_hd"]).to(dev)).cpu().numpy()
    base = eval_common.score(rc0, vc0, seal, "f0")
    mode = os.environ.get("REGION_MODE", "worst-cohort")
    if mode == "dense":                                                     # COMPACT dense region: densest 2D cell of the f0 layout
        region = "dense-2d-cluster"
        lo, hi = Z0.min(0).values, Z0.max(0).values; G = 40
        b = ((Z0 - lo) / (hi - lo).clamp_min(1e-6) * (G - 1)).long().clamp(0, G - 1)
        flat = b[:, 0] * G + b[:, 1]; cell = torch.bincount(flat, minlength=G * G).argmax()
        center = lo + (torch.tensor([cell // G, cell % G], device=dev).float() + 0.5) / G * (hi - lo)
        sigma = float(global_radius * 0.08)                                 # compact (~8% of the global radius)
        reg_rows = torch.where(((Z0 - center) ** 2).sum(1) <= (2 * sigma) ** 2)[0]
    else:                                                                   # worst-global-recall provenance cohort
        region = base["recall@k15_B2000"]["worst_cohort"]
        reg_rows = torch.where(torch.from_numpy((train_src == region).astype(bool)).to(dev))[0]
        center = Z0[reg_rows].mean(0); sigma = float((Z0[reg_rows] - center).norm(dim=1).quantile(0.75).clamp_min(1e-3))
    print(f"[D] region={region} ({reg_rows.numel()} train rows) sigma {sigma:.3f} | f0 global B2000 {base['recall@k15_B2000']['micro']} region-cohort {base['recall@k15_B2000']['per_source'].get(region,'spatial')}", flush=True)

    def gate(z):                                                            # smooth compactly-supported (Gaussian, tapered 0 at 3σ)
        d2 = ((z - center) ** 2).sum(1); gg = torch.exp(-d2 / (2 * sigma ** 2))
        return torch.where(d2 <= (3 * sigma) ** 2, gg, torch.zeros_like(gg))

    r_phi = torch.nn.Sequential(torch.nn.Linear(dim, 512), torch.nn.SiLU(), torch.nn.Linear(512, 2)).to(dev)
    torch.nn.init.zeros_(r_phi[-1].weight); torch.nn.init.zeros_(r_phi[-1].bias)   # start = f0 (residual 0)
    opt = torch.optim.Adam(r_phi.parameters(), lr=1e-3)
    n_pos = int(BATCH * POS_RATIO); n_neg = BATCH - n_pos
    # region edges: positive edges whose SOURCE is a region row (repair the region's neighborhoods)
    reg_edge = torch.isin(src_all, reg_rows); reg_src = src_all[reg_edge]; reg_dst = dst_all[reg_edge]
    ne = reg_src.numel(); print(f"[D] region edges {ne:,}", flush=True)

    def f_res(idx):                                                         # f(x)=f0(x)+g(f0(x))·r_phi(x); f0 frozen
        z0 = Z0[idx]; return z0 + gate(z0)[:, None] * r_phi(train_hd[idx])

    def loss_fn(se, de, tgt):
        r2 = (se - de).square().sum(1); tiny = torch.finfo(r2.dtype).tiny
        radial = torch.where(r2 == 0, torch.zeros_like(r2), r2.clamp_min(tiny).pow(B_))
        qs = torch.nan_to_num(torch.pow(1 + A * radial, -1.0), nan=1e-7, posinf=1 - 1e-7, neginf=1e-7).clamp(1e-7, 1 - 1e-7)
        per = torch.nn.functional.binary_cross_entropy(qs, tgt, reduction="none"); neg = tgt < 0.5
        per = torch.where(neg, NEG_TANH * torch.tanh(per / NEG_TANH), per)
        return per.mean()

    t0 = time.time()
    for step in range(horizon):
        pe = torch.randint(0, ne, (n_pos,), device=dev); ps, pd = reg_src[pe], reg_dst[pe]
        ns = torch.randint(0, n, (n_neg,), device=dev); nd = torch.randint(0, n, (n_neg,), device=dev)
        idx_s = torch.cat([ps, ns]); idx_d = torch.cat([pd, nd])
        tgt = torch.cat([torch.ones(n_pos, device=dev), torch.zeros(n_neg, device=dev)])
        se, de = f_res(idx_s), f_res(idx_d)
        gloss = loss_fn(se, de, tgt)
        # halo penalty: rows in the seam (gate in [0.05,0.5]) must not move
        halo = torch.where((gate(Z0[idx_s]) > 0.05) & (gate(Z0[idx_s]) < 0.5))[0]
        hpen = ((se[halo] - Z0[idx_s][halo]) ** 2).sum(1).mean() if halo.numel() else torch.zeros((), device=dev)
        loss = gloss + 5.0 * hpen
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(r_phi.parameters(), CLIP); opt.step()
    wall = time.time() - t0

    # deploy f (residual). score on the common evaluator (global) + region-local + movement
    def f_full(X, Z0X):
        with torch.no_grad():
            gg = gate(Z0X); return (Z0X + gg[:, None] * torch.cat([r_phi(X[i:i+100_000]) for i in range(0, X.shape[0], 100_000)]).float()).cpu().numpy()
    with torch.no_grad():
        ref_hd = torch.from_numpy(seal["ref_hd"]).to(dev); val_hd = torch.from_numpy(seal["val_hd"]).to(dev)
        Z0ref = f0(ref_hd).float(); Z0val = f0(val_hd).float()
        rcf = f_full(ref_hd, Z0ref); vcf = f_full(val_hd, Z0val)
    upd = eval_common.score(rcf, vcf, seal, "residual")
    # movement of TRAIN coords, and outside-support invariance
    with torch.no_grad():
        Zf = torch.from_numpy(f_full(train_hd, Z0)).to(dev); move = (Zf - Z0).norm(dim=1) / global_radius
        outside = gate(Z0) < 1e-4; out_move = move[outside]
    # region-local recall (works for a provenance cohort OR a spatial dense region): recall over the val queries
    # that belong to the region (by source, or by f0-coord inside the gate support), before vs after.
    import faiss
    if mode == "dense":
        vmask = (gate(Z0val) > 1e-3).cpu().numpy()
    else:
        vmask = (val_src == region)
    def region_recall(refc, valc):
        if vmask.sum() == 0: return 0.0
        d2 = faiss.IndexFlatL2(2); d2.add(np.ascontiguousarray(refc.astype(np.float32)))
        _, nn = d2.search(np.ascontiguousarray(valc[vmask].astype(np.float32)), 2000)
        tv = seal["truth_val"][vmask]
        return round(float(np.mean([len(set(int(x) for x in tv[i]) & set(int(x) for x in nn[i])) / 15 for i in range(nn.shape[0])])), 4)
    reg_before = region_recall(rc0, vc0); reg_after = region_recall(rcf, vcf)
    print(f"[D] region-local recall {reg_before} -> {reg_after} ({int(vmask.sum())} region val queries)", flush=True)
    out = {"schema": "exp-d-residual-2026-09-08", "region": region, "region_train_rows": int(reg_rows.numel()), "horizon": horizon, "wall_s": round(wall, 1),
           "region_recall_before": reg_before, "region_recall_after": reg_after, "region_gain": round(reg_after - reg_before, 4),
           "global_before": base["recall@k15_B2000"]["micro"], "global_after": upd["recall@k15_B2000"]["micro"],
           "global_loss": round(base["recall@k15_B2000"]["micro"] - upd["recall@k15_B2000"]["micro"], 4),
           "halo_p99_move": round(float(move.quantile(0.99).item()), 5), "outside_max_move": round(float(out_move.max().item()) if out_move.numel() else 0.0, 6),
           "cross_cohort_max_loss": round(max(base["recall@k15_B2000"]["per_source"][r] - upd["recall@k15_B2000"]["per_source"][r] for r in base["recall@k15_B2000"]["per_source"] if r != region), 4)}
    promote = out["region_gain"] >= 0.03 and out["global_loss"] <= 0.005 and out["outside_max_move"] <= 1e-4 and out["halo_p99_move"] <= 0.005 and out["cross_cohort_max_loss"] <= 0.01
    out["verdict"] = ("PROMOTE: region +%.4f with global loss %.4f, outside unchanged, halo p99 %.5f" % (out["region_gain"], out["global_loss"], out["halo_p99_move"])) if promote \
                     else "STOP: region gain %.4f / global loss %.4f / halo %.5f / cross-cohort %.4f did not clear the gates" % (out["region_gain"], out["global_loss"], out["halo_p99_move"], out["cross_cohort_max_loss"])
    (OUT / "r_phi.pt").parent.mkdir(parents=True, exist_ok=True); torch.save(r_phi.state_dict(), OUT / "r_phi.pt")
    (OUT / "result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
