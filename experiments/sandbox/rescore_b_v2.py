"""B arms RE-SCORED on seal-v2 (one-ruler, owner via overseer 2026-09-08). CPU. B saved heads + pca_k but not the
PCA components; eigh is deterministic on the same train rows, so we REFIT the PCA (same as exp_b's make_tf) and
SELF-VALIDATE by rescoring each arm on seal-v1 and checking it reproduces B's stored v1 B2000 (if the refit sign/
components matched, v1 reproduces; then the v2 number is trustworthy). Applies the same recall/cohort reporting.
Usage: rescore_b_v2.py
"""
import json, os
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); BDIR = SB / "exp-b"; V1 = Path("/data2/monet/eval-common")
ARMS = {"full-1536": None, "pca768": 768, "pca384": 384}


def main():
    import torch
    train = torch.from_numpy(np.asarray(np.load(V1 / "train_hd.f16.npy"), np.float32)).cuda()
    with torch.no_grad():                                                   # refit PCA exactly as exp_b (deterministic eigh)
        mu = train.mean(0); Xc = train - mu; cov = (Xc.T @ Xc) / train.shape[0]; cov = (cov + cov.T) / 2
        _, evecs = torch.linalg.eigh(cov); comp = evecs.flip(1)
    del train, Xc, cov
    stored_v1 = json.loads((BDIR / "result.json").read_text())

    def tf(X, k):
        X = torch.from_numpy(np.asarray(X, np.float32)).cuda()
        Y = torch.nn.functional.normalize(X if k is None else (X - mu) @ comp[:, :k].contiguous(), dim=1)
        return Y.cpu().numpy().astype(np.float32)

    def score_arm(arm, k, seal_dir):
        os.environ["EVAL_SEAL_DIR"] = seal_dir
        import importlib, eval_common; importlib.reload(eval_common); seal = eval_common._load_seal()
        head = BDIR / arm / "model.pt"
        import torch as _t
        from _paths import ensure_paths; ensure_paths()
        from basemap.pumap.parametric_umap.core import ParametricUMAP
        m = ParametricUMAP.load(str(head), device="cuda"); m.model.eval()
        with _t.no_grad():
            rc = m.model(_t.from_numpy(tf(seal["ref_hd"], k)).cuda()).cpu().numpy().astype(np.float32)
            vc = m.model(_t.from_numpy(tf(seal["val_hd"], k)).cuda()).cpu().numpy().astype(np.float32)
        return eval_common.score(rc, vc, seal, f"{arm}-{Path(seal_dir).name}")

    out = {"schema": "exp-b-rescore-v2-2026-09-08", "note": "PCA refit via deterministic eigh; v1 rescore validates the refit reproduces B's in-process scores.", "arms": {}}
    for arm, k in ARMS.items():
        v1 = score_arm(arm, k, "/data2/monet/eval-common"); v2 = score_arm(arm, k, "/data2/monet/eval-common-v2")
        v1_stored = stored_v1[arm]["B2000"]; v1_repro = v1["recall@k15_B2000"]["micro"]
        faithful = abs(v1_repro - v1_stored) <= 0.01
        out["arms"][arm] = {"v1_stored_B2000": v1_stored, "v1_rescored_B2000": v1_repro, "refit_faithful": faithful,
                            "v2_B2000": v2["recall@k15_B2000"]["micro"], "v2_worst_cohort": v2["recall@k15_B2000"]["worst_cohort_recall"]}
        print(f"[B-v2 {arm}] v1 stored {v1_stored} vs rescored {v1_repro} faithful {faithful} | v2 {v2['recall@k15_B2000']['micro']} worst {v2['recall@k15_B2000']['worst_cohort_recall']}", flush=True)
    full = out["arms"]["full-1536"]
    for arm in ("pca768", "pca384"):
        a = out["arms"][arm]; a["v2_delta_vs_full"] = round(a["v2_B2000"] - full["v2_B2000"], 4)
        a["v2_worst_cohort_loss_vs_full"] = round(full["v2_worst_cohort"] - a["v2_worst_cohort"], 4)
    (BDIR / "rescore-v2.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out["arms"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
