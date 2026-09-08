"""E1 — the honest cross-rung ladder (follow-up review 2026-09-08). Scores the EXISTING 2M/6M/12M DINO heads on the
common set of seal-v2 validation queries EXCLUDED from BOTH the 6M and 12M training draws (~8,661), against
original-D (1536) truth over the seal's 250K reference at fixed B, with each head's CORRECT preprocessing (2M:
DINO-1536 identity; 6M/12M: PCA-768 via the recovered basis). Required before funding any further rung — the
own-truth 2M→6M→12M ladder doubles its inspection budget across rungs and is not a fair cross-rung read. Reports
source-balanced (mean of per-source recalls) AND fixed-population-weighted (micro) aggregates SEPARATELY, with
source support after exclusion. CPU. Usage: exp_e1_ladder.py [B=2000]
"""
import json, sys
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); V2 = Path("/data2/monet/eval-common-v2")
PCA = SB / "exp-b" / "pca-basis-recovered.npz"; K = 15
HEADS = {"2M": (SB / "monet-random-dino-2m/champion-bs16k/model.pt", None),        # input DINO-1536 (no PCA)
         "6M": (SB / "monet-random-dino-6m-pca768/champion-bs16k/model.pt", 768),  # PCA-768 (recovered basis)
         "12M": (SB / "monet-random-dino-12m-pca768/champion-bs16k/model.pt", 768)}
N_POOL = 19_344_847


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    import torch, faiss
    sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    val_idx = np.load(V2 / "val_idx.npy"); val_hd = np.asarray(np.load(V2 / "val_hd.f16.npy"), np.float32)
    ref_hd = np.asarray(np.load(V2 / "ref_hd.f16.npy"), np.float32); truth = np.load(V2 / "truth_val.npy")
    src = np.load(V2 / "val_source.npy", allow_pickle=True)
    # both-excluded: seal-v2 val (pool positions) NOT in the 6M NOR 12M draw (full-corpus positions; pool part < N_POOL)
    fp6 = np.load("/data2/monet/random-dino-6m/full_pos.npy"); fp12 = np.load("/data2/monet/random-dino-12m/full_pos.npy")
    in6 = np.isin(val_idx, fp6); in12 = np.isin(val_idx, fp12); both_excl = ~(in6 | in12)
    ex = np.where(both_excl)[0]
    print(f"[E1] val {val_idx.size} | in6 {int(in6.sum())} in12 {int(in12.sum())} | both-excluded {ex.size}", flush=True)
    vex = val_hd[ex]; tex = truth[ex]; sex = src[ex]

    pm = np.load(PCA); mean = pm["mean"].astype(np.float32); comp768 = pm["comp768"].astype(np.float32)

    def preprocess(X, k):
        return _norm(X) if k is None else _norm((X - mean) @ comp768)

    def head_recall(head, k):
        m = ParametricUMAP.load(str(head), device="cpu"); m.model.eval()
        with torch.no_grad():
            rc = m.model(torch.from_numpy(preprocess(ref_hd, k))).numpy().astype(np.float32)
            vc = m.model(torch.from_numpy(preprocess(vex, k))).numpy().astype(np.float32)
        d2 = faiss.IndexFlatL2(rc.shape[1]); d2.add(np.ascontiguousarray(rc))
        _, nn = d2.search(np.ascontiguousarray(vc), B)
        pq = np.array([len(set(int(x) for x in tex[i]) & set(int(x) for x in nn[i])) / K for i in range(nn.shape[0])])
        sources = np.unique(sex); per_src = {str(s): round(float(pq[sex == s].mean()), 4) for s in sources}
        return {"micro_fixed_pop_weighted": round(float(pq.mean()), 4),
                "source_balanced": round(float(np.mean(list(per_src.values()))), 4),
                "per_source": per_src, "support": {str(s): int((sex == s).sum()) for s in sources}}

    out = {"schema": "exp-e1-ladder-2026-09-08", "B": B, "n_both_excluded": int(ex.size),
           "excluded_in_6M": int(in6.sum()), "excluded_in_12M": int(in12.sum()),
           "note": "cross-rung ladder on queries excluded from BOTH 6M+12M training; original-D truth over the seal "
                   "250K reference (retained, may overlap large-head training); each head's correct preprocessing "
                   "(2M DINO-1536, 6M/12M PCA-768 recovered basis). Source-balanced AND fixed-pop-weighted separate.",
           "rungs": {}}
    for name, (head, k) in HEADS.items():
        if not head.exists(): out["rungs"][name] = {"error": "head missing"}; print(f"[E1] {name} head MISSING", flush=True); continue
        out["rungs"][name] = head_recall(head, k)
        r = out["rungs"][name]; print(f"[E1] {name}: micro {r['micro_fixed_pop_weighted']} source-balanced {r['source_balanced']}", flush=True)
    (SB / "exp-e1-ladder.json").write_text(json.dumps(out, indent=1))
    print("[E1] ladder (both-excluded, B=%d):" % B, {n: out["rungs"][n].get("micro_fixed_pop_weighted") for n in HEADS}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
