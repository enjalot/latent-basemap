"""Card034 device canary (real 300K substrate, short dose) — per production review items 2/4. Root runs on
GPU. Drives the SAME production engine (card034_engine.GroupedEngine) as the trainer/preflight. For each arm:
exact fresh init; continuous vs genuine MID-EPOCH resumed training BITWISE equal; resume from an EPOCH-BOUNDARY
checkpoint and cross a LATER boundary BITWISE equal; the checkpoint payload is deep-validated
(card034_validate.validate_ckpt_payload) before restore; wrong objective / coefficient / seed / scalar
identity REJECTS before restore; the frozen calibration coefficient (V.calibrated_coeff) is used verbatim. A
small fixed edge subset makes an epoch a few steps so boundaries are crossed cheaply. Exit 0 = PASS.
Usage: gpu_card034_canary.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card034_validate as V
import card034_grouped as G
import card034_engine as E
import torch

SB = V.SB; OC = V.OC; CHAMPION = V.CHAMPION; SUB = V.SUB; GRAPH = V.GRAPH; INIT = V.INIT
SEED = V.SEED; SHORT = 12; STEP_AT = 5; EPOCH_STEP = 4; BLOCK = 100; SUBSET = 400   # epoch = 4 steps
_X = None; _WARM = None; _EDGES = None


def _prep():
    global _X, _WARM, _EDGES
    if _X is None: _X = torch.tensor(np.asarray(np.load(SUB, mmap_mode="r"), np.float16), device="cuda")
    if _WARM is None: _WARM = torch.load(str(INIT), map_location="cpu", weights_only=False)["model_state"]
    if _EDGES is None:
        ez = np.load(GRAPH); _EDGES = (ez["sources"][:SUBSET].copy(), ez["targets"][:SUBSET].copy())


def _ident(arm, seed, coeff): return {"card": "card034-canary", "arm": arm, "seed": int(seed), "coeff": float(coeff)}


def _run(arm, short, seed=SEED, coeff=None, resume_from=None, ckpt_targets=(), ckpt_dir=None, ident_override=None):
    if coeff is None: coeff = V.calibrated_coeff() if arm == "grouped_infonce" else 1.0
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    ident = _ident(arm, seed, coeff); ident.update(ident_override or {})
    engine = E.GroupedEngine(arm, ident, coeff, "cuda", CHAMPION, _WARM)
    sampler = G.GroupedSampler(V.N, _EDGES[0], _EDGES[1], seed=seed, block_pos=BLOCK)
    if resume_from is not None:
        ck = torch.load(resume_from, map_location="cuda", weights_only=False)
        engine.restore(ck, sampler, str(Path(__file__).resolve().parents[2]), V.N)   # deep-validate then restore
    prev_epoch = sampler.epoch; pending = False
    while engine.success < short:
        heads, tails = sampler.next_block()
        if sampler.epoch > prev_epoch: pending = True; prev_epoch = sampler.epoch
        if engine.step(_X, heads, tails):
            s = engine.success
            if pending and ckpt_dir: torch.save(engine.ckpt_dict(sampler, True), Path(ckpt_dir) / f"ckpt-epoch{sampler.epoch}.pt"); pending = False
            if s in ckpt_targets and ckpt_dir: torch.save(engine.ckpt_dict(sampler, True), Path(ckpt_dir) / f"ckpt-step{s}.pt")
    return V.state_sha(engine.model.state_dict()), engine.final_beta()


def main():
    _prep(); ROOT = str(Path(__file__).resolve().parents[2]); R = {"schema": "card034-canary-2026-09-12", "short": SHORT, "epoch_len_steps": SUBSET // BLOCK}
    import tempfile
    for arm in V.ARMS:
        with tempfile.TemporaryDirectory(dir=str(SB)) as td:
            sha_full, beta_full = _run(arm, SHORT, ckpt_targets={STEP_AT, EPOCH_STEP}, ckpt_dir=td)
            step_ck = Path(td) / f"ckpt-step{STEP_AT}.pt"; epoch_ck = sorted(Path(td).glob("ckpt-epoch*.pt"))
            assert step_ck.exists() and epoch_ck, f"{arm} checkpoints not written"
            # (a) MID-EPOCH resume bitwise
            sha_mid, beta_mid = _run(arm, SHORT, resume_from=step_ck, ckpt_dir=td)
            R[f"{arm}_midepoch_resume_bitwise"] = bool(sha_mid == sha_full)
            R[f"{arm}_midepoch_scalar_recovered"] = bool((beta_full is None and beta_mid is None) or abs(beta_mid - beta_full) < 1e-12)
            # (b) EPOCH-BOUNDARY resume crossing a LATER boundary, bitwise
            sha_ep, beta_ep = _run(arm, SHORT, resume_from=epoch_ck[0], ckpt_dir=td)
            R[f"{arm}_epoch_resume_crosses_bitwise"] = bool(sha_ep == sha_full)
            # (c) deep payload validation succeeds on a real ckpt
            V.validate_ckpt_payload(torch.load(step_ck, map_location="cpu", weights_only=False), arm, ROOT, _ident(arm, SEED, (V.calibrated_coeff() if arm == "grouped_infonce" else 1.0)), V.N, expect_beta=(arm == "grouped_nce"), expect_step=STEP_AT)
            R[f"{arm}_deep_payload_valid"] = True
            if arm == "grouped_nce": R["nce_scalar_moved"] = bool(beta_full is not None and abs(beta_full) > 0)

            def _reject(**ov):
                try:
                    _run(arm, SHORT, resume_from=step_ck, ckpt_dir=td, **ov); return False
                except AssertionError: return True
                except Exception: return False
            R[f"{arm}_wrong_seed_rejected"] = _reject(seed=SEED + 1)
            R[f"{arm}_wrong_coeff_rejected"] = _reject(coeff=(1.0 if arm == "grouped_infonce" else 2.0))
            R[f"{arm}_wrong_objective_rejected"] = _reject(ident_override={"arm": "grouped_umap" if arm != "grouped_umap" else "grouped_nce"})

    keys = [k for k in R if isinstance(R[k], bool)]
    R["PASS"] = bool(all(R[k] for k in keys) and len(keys) >= 6 * len(V.ARMS))
    (OC / "card034-canary.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
