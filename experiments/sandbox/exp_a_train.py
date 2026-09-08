"""Experiment A — alternating coordinate/head optimization (owner plan 2026-09-08, the main research bet, 4h cap).
Three arms, SAME init + SAME common sampler, on the sealed eval-common 500K DINO-1536 train subset + its full-D
graph; each DEPLOYS + is scored on f_theta (NEVER the free Z), via the common evaluator:

  direct       — standard training: model(x[src]), model(x[dst]) -> production edge loss -> step.          [control]
  teacher      — fixed teacher: optimize free Z on the graph loss (no head), FREEZE it, regress model->Z*.  [old distill]
  alternating  — Z init from head; alternate cheap coord SGD on L_graph(Z)+rho/2N·||Z−f(x).detach()||² with
                 head regression model->Z.detach(), rho ramped. Deploy f_theta; record head-vs-Z discrepancy.

Common sampler across arms (isolates the mechanism): pos edges from the graph, UNIFORM negatives at pos_ratio; the
production umap kernel + BCE + neg_tanh + fneg-band + total-weight norm (rankneg OFF — a separate axis, held fixed
across arms). All arms get a matched update budget; wall + f_theta quality recorded. NO claim that the 2015 MAC
convergence theory transfers to our stop-grad reweighted loss. Usage: exp_a_train.py [HORIZON=20000]
"""
import json, sys, time
from pathlib import Path
import numpy as np

SEAL = Path("/data2/monet/eval-common"); GRAPH = Path("/data/latent-basemap/sandbox/eval-common-train/edges-k15-fuzzy.npz")
OUT = Path("/data/latent-basemap/sandbox/exp-a"); BATCH, POS_RATIO = 16384, 0.10
A, B_, NEG_TANH, FNEG_W, FNEG_LO, FNEG_HI = 1.9328, 0.7905, 4.0, 1.0, 0.2, 0.8


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import eval_common, torch
    dev = "cuda"; torch.manual_seed(42)
    # fp32 forwards (no autocast) — A's pilot prioritizes stability over fp16 throughput; all arms identical so the
    # comparison stays fair. Removes the fp16 coord-overflow -> NaN that killed the first run.
    feats = torch.from_numpy(np.asarray(np.load(SEAL / "train_hd.f16.npy"), np.float32)).to(dev)
    n, dim = feats.shape; CLIP = 1.0                                        # production clip_grad_norm=1.0
    g = np.load(GRAPH); src_all = torch.from_numpy(g["sources"].astype(np.int64)).to(dev)
    dst_all = torch.from_numpy(g["targets"].astype(np.int64)).to(dev); n_edges = src_all.numel()
    seal = eval_common._load_seal()
    n_pos = int(BATCH * POS_RATIO); n_neg = BATCH - n_pos

    def loss_fn(se, de, tgt):
        delta = (se - de).float(); r2 = delta.square().sum(1); tiny = torch.finfo(r2.dtype).tiny
        radial = torch.where(r2 == 0, torch.zeros_like(r2), r2.clamp_min(tiny).pow(B_))
        qs = torch.pow(1 + A * radial, -1.0)
        qs = torch.nan_to_num(qs, nan=1e-7, posinf=1 - 1e-7, neginf=1e-7).clamp(1e-7, 1 - 1e-7)  # nan_to_num BEFORE clamp (clamp can't fix NaN) — matches production
        per = torch.nn.functional.binary_cross_entropy(qs, tgt, reduction="none"); neg = tgt < 0.5
        per = torch.where(neg, NEG_TANH * torch.tanh(per / NEG_TANH), per); w = torch.ones_like(per)
        with torch.no_grad():
            ep = torch.cat([se, de], 0).float(); R = torch.quantile((ep - ep.mean(0)).norm(dim=1), 0.9)
            r2d = (se - de).float().norm(dim=1); band = neg & (r2d >= FNEG_LO * R) & (r2d <= FNEG_HI * R)
        w = w + band.float() * FNEG_W
        return (per * w).sum() / w.sum()

    def sample_edges():
        pe = torch.randint(0, n_edges, (n_pos,), device=dev); ps, pd = src_all[pe], dst_all[pe]
        ns = torch.randint(0, n, (n_neg,), device=dev); nd = torch.randint(0, n, (n_neg,), device=dev)
        src = torch.cat([ps, ns]); dst = torch.cat([pd, nd])
        tgt = torch.cat([torch.ones(n_pos, device=dev), torch.zeros(n_neg, device=dev)])
        return src, dst, tgt

    def fresh_model():
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        pu = ParametricUMAP(n_components=2, hidden_dim=2048, n_layers=3, architecture="residual_bottleneck",
                            low_dim_kernel="umap", a=A, b=B_, use_amp=True, device=dev)
        pu._init_model(dim); return pu

    def head_all(model):                                                    # fp32 forward over all train rows (no autocast)
        with torch.no_grad():
            return torch.cat([model(feats[i:i + 100_000].float()) for i in range(0, n, 100_000)]).float()

    def pca2d_init():                                                        # spread structure-preserving free-Z init
        with torch.no_grad():                                               # (UMAP-standard alternative to spectral); the
            mu = feats.mean(0); X = feats - mu                              # tight normal(0.1) init collapsed the 1st run
            _, _, V = torch.pca_lowrank(X, q=2, niter=4)
            Z0 = X @ V[:, :2]; Z0 = Z0 / (Z0.std(0) + 1e-6) * 10.0          # scale to a UMAP-ish coord spread (std~10)
        return Z0.contiguous()

    def score(pu, label):
        pu.model.eval()
        with torch.no_grad():
            rc = pu.model(torch.from_numpy(seal["ref_hd"]).to(dev)).cpu().numpy().astype(np.float32)
            vc = pu.model(torch.from_numpy(seal["val_hd"]).to(dev)).cpu().numpy().astype(np.float32)
        pu.model.train()
        return eval_common.score(rc, vc, seal, label)

    results = {}

    # ---- arm: direct control ----
    pu = fresh_model(); opt = torch.optim.Adam(pu.model.parameters(), lr=1e-3); pu.model.train()
    t0 = time.time()
    for step in range(horizon):
        src, dst, tgt = sample_edges(); opt.zero_grad(set_to_none=True)
        se, de = pu.model(feats[src]), pu.model(feats[dst])
        loss = loss_fn(se, de, tgt); loss.backward()
        torch.nn.utils.clip_grad_norm_(pu.model.parameters(), CLIP); opt.step()
        if step % 5000 == 0:
            with torch.no_grad(): ff = torch.isfinite(se).all(1).float().mean().item()
            print(f"[A direct] step {step} loss {loss.item():.4f} coord-finite {ff:.4f}", flush=True)
    torch.cuda.synchronize(); wall = time.time() - t0
    results["direct"] = {"wall_s": round(wall, 1), "horizon": horizon, **score(pu, "direct")}
    print(f"[A direct] {wall:.0f}s B2000 {results['direct']['recall@k15_B2000']['micro']}", flush=True)

    # ---- arm: fixed teacher + regression ----
    t0 = time.time()
    Z = pca2d_init().clone().requires_grad_()                               # spread PCA-2D init (was normal(0.1) -> collapse)
    optZ = torch.optim.Adam([Z], lr=1e-2)
    for step in range(horizon):                                             # optimize FREE coords on the graph loss
        src, dst, tgt = sample_edges(); optZ.zero_grad(set_to_none=True)
        loss = loss_fn(Z[src], Z[dst], tgt); loss.backward()
        optZ.step()                                                          # no Z clip: Adam is scale-robust; clip crippled spread
        if step % 5000 == 0: print(f'[A teacher] Zopt step {step} loss {loss.item():.4f} Z.std {Z.detach().std().item():.3f}', flush=True)
    Zstar = Z.detach()
    pu = fresh_model(); opt = torch.optim.Adam(pu.model.parameters(), lr=1e-3); pu.model.train()
    reg_steps = horizon                                                     # matched budget on the regression side
    for step in range(reg_steps):
        idx = torch.randint(0, n, (BATCH,), device=dev); opt.zero_grad(set_to_none=True)
        pred = pu.model(feats[idx])
        loss = ((pred - Zstar[idx]) ** 2).sum(1).mean(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pu.model.parameters(), CLIP); opt.step()
    torch.cuda.synchronize(); wall = time.time() - t0
    with torch.no_grad():
        disc = (head_all(pu.model) - Zstar).norm(dim=1).mean().item()
    results["teacher"] = {"wall_s": round(wall, 1), "head_teacher_discrepancy": round(disc, 4), **score(pu, "teacher")}
    print(f"[A teacher] {wall:.0f}s B2000 {results['teacher']['recall@k15_B2000']['micro']} disc {disc:.3f}", flush=True)

    # ---- arm: alternating (proximal penalty) ----
    t0 = time.time(); pu = fresh_model(); opt = torch.optim.Adam(pu.model.parameters(), lr=1e-3); pu.model.train()
    Z = pca2d_init().clone().requires_grad_()                               # PCA-2D init; rho ramps 0->2
    optZ = torch.optim.Adam([Z], lr=1e-2)                                  # bind the NEW tensor, not the teacher's Z
    assert optZ.param_groups[0]["params"][0] is Z
    outer = horizon // 200; rho0, rho1 = 0.0, 2.0                           # rho ramp; calibrated below to loss scale
    for o in range(outer):
        rho = rho0 + (rho1 - rho0) * (o / max(outer - 1, 1))
        fx = head_all(pu.model)                                             # head output (detached, fp32)
        for _ in range(100):                                                # cheap coord phase: graph loss + rho·proximal
            src, dst, tgt = sample_edges(); optZ.zero_grad(set_to_none=True)
            gl = loss_fn(Z[src], Z[dst], tgt)
            prox = rho / (2 * n) * ((Z - fx) ** 2).sum()
            (gl + prox).backward(); optZ.step()                              # no Z clip (Adam scale-robust)
        Zc = Z.detach()
        if o % 20 == 0: print(f'[A alternating] outer {o} rho {rho:.2f} Z.std {Zc.std().item():.3f}', flush=True)
        for _ in range(100):                                                # regression phase: head -> Z
            idx = torch.randint(0, n, (BATCH,), device=dev); opt.zero_grad(set_to_none=True)
            pred = pu.model(feats[idx])
            loss = ((pred - Zc[idx]) ** 2).sum(1).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(pu.model.parameters(), CLIP); opt.step()
    torch.cuda.synchronize(); wall = time.time() - t0
    with torch.no_grad():
        disc = (head_all(pu.model) - Z.detach()).norm(dim=1).mean().item()
    results["alternating"] = {"wall_s": round(wall, 1), "head_Z_discrepancy": round(disc, 4), "rho_final": rho1, **score(pu, "alternating")}
    print(f"[A alternating] {wall:.0f}s B2000 {results['alternating']['recall@k15_B2000']['micro']} disc {disc:.3f}", flush=True)

    # ---- promote/stop (plan rule) ----
    dc, al = results["direct"], results["alternating"]
    dr, ar = dc["recall@k15_B2000"]["micro"], al["recall@k15_B2000"]["micro"]
    cohort_losses = {c: dc["recall@k15_B2000"]["per_source"][c] - al["recall@k15_B2000"]["per_source"][c]
                     for c in dc["recall@k15_B2000"]["per_source"]}
    worst_cohort_loss = max(cohort_losses.values())
    half_time = al["wall_s"] <= 0.5 * dc["wall_s"] and ar >= dr - 0.005 and worst_cohort_loss <= 0.01
    better = ar >= dr + 0.02 and al["wall_s"] <= dc["wall_s"] and worst_cohort_loss <= 0.01
    verdict = ("PROMOTE: alternating reaches control quality in ≤half time" if half_time else
               "PROMOTE: alternating +0.02 recall at matched wall" if better else
               "STOP: alternating did not beat control on the plan's rule (Δrecall %.4f, wall %.0f vs %.0f, worst-cohort Δ %.4f)"
               % (ar - dr, al["wall_s"], dc["wall_s"], worst_cohort_loss))
    out = {"schema": "exp-a-2026-09-08", "substrate": "eval-common train 500K DINO-1536", "horizon": horizon,
           "sampler": "pos from graph + UNIFORM negatives (rankneg OFF, held fixed across arms)",
           "arms": results, "verdict": verdict,
           "note": "scored f_theta NEVER Z; deploy the head. head-vs-Z discrepancy recorded. First A pilot; if only "
                   "Z improved and the head can't match it, that's a layout the head can't yet deliver."}
    (OUT / "result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
