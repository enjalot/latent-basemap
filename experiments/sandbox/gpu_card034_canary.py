"""Card034 device canary (real 300K substrate, short dose) — per card034 "Mandatory real-device full-data
canaries". Root runs on GPU. For each arm (grouped_umap/grouped_nce/grouped_infonce): exact fresh init;
continuous vs genuine mid-epoch RESUMED training BITWISE equal, including an EPOCH BOUNDARY and (nce) the
scalar/Adam/scaler state; wrong objective / coefficient / seed / scalar identity REJECTS before restore. To
cross epoch boundaries within a short canary, a small fixed edge subset is used (block_pos small) so an epoch
is a few steps — this exercises the real GroupedSampler epoch-reshuffle + epoch-ckpt + resume path; the
production dose uses the full graph. Exit 0 = PASS. Usage: gpu_card034_canary.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card034_validate as V
import card034_grouped as G
import torch, torch.nn as nn
from torch.optim import AdamW
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = V.SB; OC = V.OC; CHAMPION = V.CHAMPION; SUB = V.SUB; GRAPH = V.GRAPH; INIT = V.INIT
SEED = V.SEED; SHORT = 12; CKPT_AT = 5; BLOCK = 100; SUBSET = 400   # small subset -> epoch = 4 steps
_X = None; _WARM = None; _EDGES = None


def _prep():
    global _X, _WARM, _EDGES
    if _X is None: _X = torch.tensor(np.asarray(np.load(SUB, mmap_mode="r"), np.float16), device="cuda")
    if _WARM is None: _WARM = torch.load(str(INIT), map_location="cpu", weights_only=False)["model_state"]
    if _EDGES is None:
        ez = np.load(GRAPH); _EDGES = (ez["sources"][:SUBSET].copy(), ez["targets"][:SUBSET].copy())


def _identity(arm, seed, coeff): return {"card": "card034-canary", "arm": arm, "seed": int(seed), "coeff": float(coeff)}


def _run(arm, short, seed=SEED, coeff=None, resume_from=None, ckpt_at=None, ckpt_dir=None, ident_override=None):
    if coeff is None: coeff = 16779.48203543921 if arm == "grouped_infonce" else 1.0
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    p = ParametricUMAP.load(str(CHAMPION), device="cuda"); p.model = None; p.n_components = V.NC
    p._init_model(1536); p.model.load_state_dict(_WARM); model = p.model
    beta = None; groups = [{"params": list(model.parameters()), "lr": V.LR, "weight_decay": V.WEIGHT_DECAY}]
    if arm == "grouped_nce":
        beta = nn.Parameter(torch.zeros((), device="cuda")); groups.append({"params": [beta], "lr": V.LR, "weight_decay": 0.0})
    opt = AdamW(groups); scaler = torch.amp.GradScaler("cuda", enabled=True)
    sampler = G.GroupedSampler(V.N, _EDGES[0], _EDGES[1], seed=seed, block_pos=BLOCK)
    ident = _identity(arm, seed, coeff); ident.update(ident_override or {})
    success = 0; last_epoch = 0

    def ck(name):
        st = {"schema": "card034-ckpt-2026-09-12", "arm": arm, "global_step": int(success), "epoch": int(sampler.epoch),
              "step_checkpoint": name.startswith("step"), "identity": ident, "coeff": coeff, "model": model.state_dict(),
              "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "beta": (beta.detach().cpu() if beta is not None else None),
              "sampler_state": sampler.state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        torch.save(st, Path(ckpt_dir) / f"ckpt-{name}.pt")

    if resume_from is not None:
        cks = torch.load(resume_from, map_location="cuda", weights_only=False)
        assert cks.get("identity") == ident, "resume identity mismatch — reject before restore"   # fail closed
        model.load_state_dict(cks["model"]); opt.load_state_dict(cks["optimizer"]); scaler.load_state_dict(cks["scaler"])
        if beta is not None and cks.get("beta") is not None:
            with torch.no_grad(): beta.copy_(cks["beta"].to("cuda"))
        sampler.load_state(cks["sampler_state"]); torch.set_rng_state(cks["torch_rng"].to("cpu", torch.uint8))
        torch.cuda.set_rng_state_all([s.to("cpu", torch.uint8) for s in cks["cuda_rng"]]); success = int(cks["global_step"]); last_epoch = int(sampler.epoch)

    while success < short:
        heads, tails = sampler.next_block()
        if sampler.epoch > last_epoch and ckpt_dir: ck(f"epoch{sampler.epoch}"); last_epoch = sampler.epoch
        h = torch.as_tensor(heads, device="cuda"); tl = torch.as_tensor(tails, device="cuda")
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            he = model(_X.index_select(0, h)); te = model(_X.index_select(0, tl.reshape(-1))).reshape(h.shape[0], G.GROUP, V.NC)
        phi = G.phi_from_emb(he.float(), te.float())
        loss = (G.grouped_umap_loss(phi) if arm == "grouped_umap" else
                G.grouped_nce_loss(phi, beta) if arm == "grouped_nce" else coeff * G.grouped_infonce_loss(phi))
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), V.CLIP)
        prev = scaler.get_scale(); scaler.step(opt); scaler.update()
        if scaler.get_scale() >= prev:
            success += 1
            if ckpt_at and success == ckpt_at and ckpt_dir: ck(f"step{success}")
    return V.state_sha(model.state_dict()), (float(beta.detach()) if beta is not None else None)


def main():
    _prep(); R = {"schema": "card034-canary-2026-09-12", "short": SHORT, "ckpt_at": CKPT_AT, "epoch_len_steps": SUBSET // BLOCK}
    import tempfile
    for arm in V.ARMS:
        with tempfile.TemporaryDirectory(dir=str(SB)) as td:
            sha_full, beta_full = _run(arm, SHORT, ckpt_at=CKPT_AT, ckpt_dir=td)
            ck = Path(td) / f"ckpt-step{CKPT_AT}.pt"; assert ck.exists(), f"{arm} step ckpt not written"
            o = torch.load(ck, map_location="cpu", weights_only=False)
            R[f"{arm}_ckpt_full_state"] = all(o.get(k) is not None for k in ["model", "optimizer", "scaler", "sampler_state", "torch_rng", "cuda_rng"])
            R[f"{arm}_epoch_ckpt_present"] = any(Path(td).glob("ckpt-epoch*.pt"))   # small subset crosses epochs
            sha_res, beta_res = _run(arm, SHORT, resume_from=ck, ckpt_dir=td)
            R[f"{arm}_resume_twin_bitwise"] = bool(sha_res == sha_full)
            R[f"{arm}_resume_recovers_scalar"] = bool((beta_full is None and beta_res is None) or (beta_full is not None and abs(beta_res - beta_full) < 1e-12))
            if arm == "grouped_nce":
                R["nce_scalar_moved"] = bool(beta_full is not None and abs(beta_full) > 0)

            def _reject(**ov):
                try:
                    _run(arm, SHORT, resume_from=ck, ckpt_dir=td, **ov); return False
                except AssertionError: return True
                except Exception: return False
            R[f"{arm}_wrong_seed_rejected"] = _reject(seed=SEED + 1)
            R[f"{arm}_wrong_coeff_rejected"] = _reject(coeff=(1.0 if arm == "grouped_infonce" else 2.0))
            R[f"{arm}_wrong_objective_rejected"] = _reject(ident_override={"arm": "grouped_umap" if arm != "grouped_umap" else "grouped_nce"})

    keys = [k for k in R if isinstance(R[k], bool)]
    R["PASS"] = bool(all(R[k] for k in keys) and len(keys) >= 3 * len(V.ARMS))
    (OC / "card034-canary.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
