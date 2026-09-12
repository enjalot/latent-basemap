"""Card034 InfoNCE init-scale calibration (per card034 "Before any GPU production, freeze ONE multiplicative
InfoNCE loss coefficient to match initial MODEL gradient scale to grouped_umap"). CPU: 16 predetermined
64-group batches drawn only from the training graph (shared grouped sampler recipe), actual stored FP16->FP32
inputs and the shared fresh initial weights (589895f0). For each batch compute the UMAP and the (raw) InfoNCE
global-L2 MODEL gradient norms; coefficient = sum(UMAP norms) / sum(InfoNCE norms). Record every norm and the
coefficient; FAIL on any nonfinite or zero. Declared optimization-scale calibration at initialization — not a
quality tune, not equality of later gradients; NCE's scale is untouched. Usage: calibrate_card034.py
"""
import os, sys, json, math, time, datetime as dt
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card034_validate as V
import card034_grouped as G

ROOT = Path(__file__).resolve().parents[2]
OC = V.OC; CHAMPION = V.CHAMPION; SUB = V.SUB; GRAPH = V.GRAPH; INIT = V.INIT
CALIB_SEED = 3434; N_BATCHES = 16; GROUPS_PER_BATCH = 64


def _grad_l2(loss, params):
    import torch
    gs = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    return float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs if g is not None)))


def main():
    import torch
    fok, bad = V.runtime_manifest_check(ROOT); assert fok, f"frozen runtime mismatch {bad}"
    assert V.full_sha(SUB) == V.SUB_SHA256 and V.full_sha(GRAPH) == V.GRAPH_SHA256, "data identity"
    init_obj = torch.load(str(INIT), map_location="cpu", weights_only=False)
    assert init_obj["init_state_sha256"] == V.INIT_SHA, "fresh 2D init identity"; warm = init_obj["model_state"]

    from basemap.pumap.parametric_umap.core import ParametricUMAP
    p = ParametricUMAP.load(str(CHAMPION), device="cpu"); p.model = None; p.n_components = V.NC
    p._init_model(1536); p.model.load_state_dict(warm); model = p.model.eval()
    for prm in model.parameters(): prm.requires_grad_(True)
    params = [prm for prm in model.parameters() if prm.requires_grad]

    ez = np.load(GRAPH); src = ez["sources"]; tgt = ez["targets"]
    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)     # unit-norm rows; fp16 store cast to fp32 below
    sampler = G.GroupedSampler(V.N, src, tgt, seed=CALIB_SEED, block_pos=GROUPS_PER_BATCH)

    t0 = time.time(); rows = []
    for bi in range(N_BATCHES):
        heads, tails = sampler.next_block()
        h = torch.tensor(np.asarray(X[heads], np.float16).astype(np.float32))          # stored fp16 -> fp32
        tl = torch.tensor(np.asarray(X[tails.reshape(-1)], np.float16).astype(np.float32))
        head_emb = model(h); tail_emb = model(tl).reshape(h.shape[0], G.GROUP, V.NC)
        phi = G.phi_from_emb(head_emb, tail_emb)
        umap_n = _grad_l2(G.grouped_umap_loss(phi), params)
        # recompute phi (graph freed after grad) for the InfoNCE norm on the SAME batch
        head_emb2 = model(h); tail_emb2 = model(tl).reshape(h.shape[0], G.GROUP, V.NC)
        info_n = _grad_l2(G.grouped_infonce_loss(G.phi_from_emb(head_emb2, tail_emb2)), params)
        rows.append({"batch": bi, "umap_grad_l2": umap_n, "infonce_grad_l2": info_n})

    all_finite = all(math.isfinite(r["umap_grad_l2"]) and math.isfinite(r["infonce_grad_l2"]) and
                     r["umap_grad_l2"] > 0 and r["infonce_grad_l2"] > 0 for r in rows)
    sum_u = sum(r["umap_grad_l2"] for r in rows); sum_i = sum(r["infonce_grad_l2"] for r in rows)
    coeff = (sum_u / sum_i) if (all_finite and sum_i > 0) else None
    stable = bool(all_finite and coeff is not None and math.isfinite(coeff) and coeff > 0)
    R = {"schema": "card034-calibration-2026-09-12", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
         "calibration_seed": CALIB_SEED, "n_batches": N_BATCHES, "groups_per_batch": GROUPS_PER_BATCH,
         "per_batch": rows, "sum_umap_grad_l2": sum_u, "sum_infonce_grad_l2": sum_i, "coefficient": coeff,
         "all_finite_positive": all_finite, "wall_s": round(time.time() - t0, 2),
         "method": "coefficient = sum(UMAP global-L2 model grad) / sum(InfoNCE global-L2 model grad) over 16 "
                   "fixed 64-group training-graph batches at the shared fresh init; init-scale only, not a "
                   "quality tune; NCE scale untouched; production/scalar canaries must use this exact value.",
         "PASS": stable}
    if not stable: R["stop_reason"] = "nonfinite/zero gradient norm — STOP; no retuning"
    (OC / "card034-calibration.json").write_text(json.dumps(R, indent=2))
    print(json.dumps({"coefficient": coeff, "all_finite_positive": all_finite, "PASS": stable}, indent=1), flush=True)
    return 0 if stable else 3


if __name__ == "__main__":
    raise SystemExit(main())
