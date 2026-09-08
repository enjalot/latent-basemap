"""B1 — corrected B rescore (follow-up review 2026-09-08). Fixes exp_b/rescore_b_v2's min-vs-min cohort bug and adds
the missing rigor: PER-COHORT max-loss max_c[full(c)−arm(c)] (not min_c(full)−min_c(arm)), PER-QUERY paired
differences (paired base-vs-arm on identical queries → the right uncertainty for small losses), BOTH budgets
(B250+B2000), source-level counts, and it RECOVERS+PERSISTS the real PCA mean/components (deterministic eigh) with
a reconstruction faithfulness check (rescored v1 must reproduce exp_b's stored v1 B2000). CPU (faiss + a CPU head
projection). Scores the saved B heads (full-1536/pca768/pca384) on seal-v2. Usage: exp_b1_rescore.py
"""
import json
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); BDIR = SB / "exp-b"; V1 = Path("/data2/monet/eval-common"); V2 = Path("/data2/monet/eval-common-v2")
ARMS = {"full-1536": None, "pca768": 768, "pca384": 384}; K = 15; BUDGETS = (250, 2000)


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def per_query_recall(ref_coords, val_coords, truth, B):
    import faiss
    d2 = faiss.IndexFlatL2(ref_coords.shape[1]); d2.add(np.ascontiguousarray(ref_coords.astype(np.float32)))
    _, nn = d2.search(np.ascontiguousarray(val_coords.astype(np.float32)), B)
    return np.array([len(set(int(x) for x in truth[i]) & set(int(x) for x in nn[i])) / K for i in range(nn.shape[0])])


def main():
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    dev = "cuda"                                                            # MUST match exp_b's CUDA eigh — CPU eigh flips
    stored_v1 = json.loads((BDIR / "result.json").read_text())              # eigenvector signs → wrong transform (B1 v1 bug)
    train = torch.from_numpy(np.asarray(np.load(V1 / "train_hd.f16.npy"), np.float32)).to(dev)
    with torch.no_grad():                                                   # deterministic PCA refit on CUDA — recover + persist
        mu = train.mean(0); Xc = train - mu; cov = (Xc.T @ Xc) / train.shape[0]; cov = (cov + cov.T) / 2
        _, evecs = torch.linalg.eigh(cov); comp = evecs.flip(1)
    np.savez(BDIR / "pca-basis-recovered.npz", mean=mu.cpu().numpy(), comp768=comp[:, :768].cpu().numpy(), comp384=comp[:, :384].cpu().numpy())
    del train, Xc, cov

    def tf(X, k):
        Xt = torch.from_numpy(np.asarray(X, np.float32)).to(dev)
        Y = torch.nn.functional.normalize(Xt if k is None else (Xt - mu) @ comp[:, :k], dim=1)
        return Y.cpu().numpy().astype(np.float32)

    def arm_coords(arm, k, seal):
        m = ParametricUMAP.load(str(BDIR / arm / "model.pt"), device=dev); m.model.eval()
        with torch.no_grad():
            rc = m.model(torch.from_numpy(tf(seal["ref_hd"], k)).to(dev)).cpu().numpy().astype(np.float32)
            vc = m.model(torch.from_numpy(tf(seal["val_hd"], k)).to(dev)).cpu().numpy().astype(np.float32)
        return rc, vc

    def load_seal(d):
        L = lambda n, **kw: np.load(d / n, **kw)
        return dict(ref_hd=np.asarray(L("ref_hd.f16.npy"), np.float32), val_hd=np.asarray(L("val_hd.f16.npy"), np.float32),
                    truth=L("truth_val.npy"), src=L("val_source.npy", allow_pickle=True))

    out = {"schema": "exp-b1-rescore-2026-09-08", "corrects": "min-vs-min cohort bug → per-cohort max-loss + per-query paired diffs + both budgets; PCA basis recovered+persisted",
           "pca_basis": str(BDIR / "pca-basis-recovered.npz"), "v1_reconstruction_check": {}, "v2": {}}
    # v1 faithfulness: rescored full/pca768/pca384 v1 B2000 must reproduce exp_b's stored v1 B2000
    sv1 = load_seal(V1)
    for arm, k in ARMS.items():
        rc, vc = arm_coords(arm, k, sv1); pq = per_query_recall(rc, vc, sv1["truth"], 2000)
        out["v1_reconstruction_check"][arm] = {"rescored_B2000": round(float(pq.mean()), 4), "stored_B2000": stored_v1[arm]["B2000"],
                                                "faithful": bool(abs(float(pq.mean()) - stored_v1[arm]["B2000"]) <= 0.01)}
    # v2 corrected scoring
    sv2 = load_seal(V2); sources = np.unique(sv2["src"])
    pq_arm = {}
    for B in BUDGETS:
        rec = {}
        for arm, k in ARMS.items():
            rc, vc = arm_coords(arm, k, sv2); pq = per_query_recall(rc, vc, sv2["truth"], B); pq_arm[(arm, B)] = pq
            rec[arm] = {"micro": round(float(pq.mean()), 4),
                        "per_source": {str(s): round(float(pq[sv2["src"] == s].mean()), 4) for s in sources}}
        full = rec["full-1536"]
        for arm in ("pca768", "pca384"):
            per_c_loss = {s: round(full["per_source"][str(s)] - rec[arm]["per_source"][str(s)], 4) for s in sources}
            worst = max(per_c_loss.values()); worst_s = max(per_c_loss, key=per_c_loss.get)
            paired = pq_arm[("full-1536", B)] - pq_arm[(arm, B)]           # per-query paired diff (full − arm), same queries
            rng = np.random.default_rng(0); boot = [paired[rng.integers(0, paired.size, paired.size)].mean() for _ in range(1000)]
            rec[arm]["vs_full"] = {"agg_delta": round(float(paired.mean()), 4),
                                   "agg_delta_ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                                   "worst_cohort_loss": worst, "worst_cohort": str(worst_s), "per_cohort_loss": per_c_loss,
                                   "gate_pass": bool(abs(float(paired.mean())) <= 0.005 and worst <= 0.01)}
        out["v2"][f"B{B}"] = rec
    (BDIR / "b1-rescore.json").write_text(json.dumps(out, indent=1))
    print("v1 faithful:", {a: out["v1_reconstruction_check"][a]["faithful"] for a in ARMS})
    for B in BUDGETS:
        for arm in ("pca768", "pca384"):
            v = out["v2"][f"B{B}"][arm]["vs_full"]
            print(f"  B{B} {arm}: aggΔ {v['agg_delta']} CI {v['agg_delta_ci95']} worst-cohort {v['worst_cohort_loss']}@{v['worst_cohort']} PASS {v['gate_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
