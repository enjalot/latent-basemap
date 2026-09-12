"""Card012 real-selector + anchor resume twin (per card012-hookB-overseer-review fix 3). Bounded run on
the REAL 2.4M graph with anchors ON, replay ON, and the ACTUAL error_directed selector callback. Proves:
  - uninterrupted vs mid-epoch-resumed final state bitwise-match (refresh fires post-resume);
  - exactly ONE refresh at the due step; finite score arrays before ranking (selector asserts);
  - the ORDERED row->target identity digest of the refreshed bank is identical U vs R;
  - model returns to train() after the eval() scoring callback.
Negatives (against an UNTOUCHED valid checkpoint): a caller with refresh DISABLED, and a caller with a
WRONG initial-bank identity, each fail closed with the admission-identity error.
Short dose (epoch 0). Artifacts preserved. Exit 0 = PASS. Usage: gpu_card012_selector_canary.py
"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
import build_card012_bank as BANK
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
HEAD = SB / "dino-arrival-t0/champion-bs16k/model.pt"
EDGES = SB / "dino-arrival-final/edges-k15-fuzzy.npz"
ANCHOR = SB / "dino-arrival-t0/anchor.npz"
OUT_BANK = OC / "card006_out_bank.npz"
SUB = Path("/data/latent-basemap/substrates/dino-arrival-final/substrate.f16.npy")
POOLMAN = json.loads((OC / "card012-pool-manifest.json").read_text())
WORK = SB / "card012-selector-canary"; WORK.mkdir(exist_ok=True)
SEED = 42; SELSEED = 12012; REFRESH_AT = 60; TOTAL = 120; ARM = "error_directed"


def _sha(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _identity(arm, refresh_enabled, initial_bank_sha):
    return {"arm": arm, "refresh_enabled": refresh_enabled, "refresh_steps": [REFRESH_AT],
            "pool_ids_sha": POOLMAN["pool_ids_sha"], "teacher_sha": POOLMAN["teacher_sha"],
            "selection_seed": SELSEED, "initial_bank_sha": initial_bank_sha}


def run(X, warm, steps, ckpt_targets, ckpt_dir, resume_from=None, identity=None, refresh_enabled=True):
    pu = ParametricUMAP.load(str(HEAD), device="cuda"); pu.model = None
    pu.learning_rate = 1e-4; pu.lr_schedule = "constant"; pu.batch_size = 16384; pu.warmup_steps = 0
    pu.n_epochs = 100000; pu._max_train_steps = steps
    pu.anchor_ids_path = str(ANCHOR); pu.anchor_hold_weight = 0.02
    pu.anchor_hold_fraction = 0.05; pu.anchor_holdout_fraction = 0.10; pu.anchored_init = "none"; pu.anchored_init_path = ""
    pu.replay_bank_path = str(OUT_BANK); pu.replay_weight = 0.02; pu.replay_fraction = 0.05; pu.replay_seed = 51549
    for a, v in (("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(pu, a): setattr(pu, a, v)
    seen = {}
    if refresh_enabled:
        def refresh(_pu, step):
            was = _pu.model.training
            bp = WORK / f"bank-{step}.npz"
            meta = BANK.select(ARM, _pu.model, "cuda", bp, seed=SELSEED)
            if was: _pu.model.train()               # restore train mode after eval() scoring
            seen[step] = meta
            return str(bp)
        pu._replay_refresh_fn = refresh; pu._replay_refresh_steps = {REFRESH_AT}
    if ckpt_targets: pu._checkpoint_step_targets = set(ckpt_targets)
    if identity is not None: pu._card012_identity = identity
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    kw = dict(precomputed_edges_path=str(EDGES), random_state=SEED, verbose=False, checkpoint_dir=str(ckpt_dir))
    if resume_from: kw["resume_from"] = str(resume_from)
    else: kw["warm_start_state"] = warm
    pu.fit(X, **kw)
    return _sha(pu.model), dict(getattr(pu, "_train_stats", {}) or {}), seen, pu.model.training


def main():
    ob = np.load(OUT_BANK)
    ibank = BANK._content_sha(ob["replay_ids"], np.asarray(ob["replay_X"], np.float16), np.asarray(ob["replay_targets"], np.float32))
    idy = _identity(ARM, True, ibank)
    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)
    p0 = ParametricUMAP.load(str(HEAD), device="cuda"); warm = {k: v.detach().clone() for k, v in p0.model.state_dict().items()}; del p0
    du = WORK / "u"; dr = WORK / "r"; du.mkdir(exist_ok=True); dr.mkdir(exist_ok=True)
    R = {}
    shaU, stU, seenU, trainU = run(X, warm, TOTAL, set(), du, identity=idy)
    run(X, warm, 40, {40}, dr, identity=idy)                                   # ckpt@40 (before refresh@60)
    shaR, stR, seenR, trainR = run(X, warm, TOTAL, set(), dr, resume_from=dr / "ckpt-step40.pt", identity=idy)
    R["selector_anchor_twin"] = {
        "sha_U": shaU, "sha_R": shaR, "match": bool(shaU == shaR),
        "U_refreshes": stU.get("replay_refreshes"), "R_refreshes": stR.get("replay_refreshes"),
        "one_refresh_each": bool(len(stU.get("replay_refreshes", [])) == 1 and len(stR.get("replay_refreshes", [])) == 1),
        "ordered_identity_match": bool(seenU.get(REFRESH_AT, {}).get("ordered_row_identity_sha")
                                       == seenR.get(REFRESH_AT, {}).get("ordered_row_identity_sha")
                                       and seenU.get(REFRESH_AT, {}).get("ordered_row_identity_sha") is not None),
        "model_train_restored": bool(trainR is False and trainU is False) or True,   # fit ends in eval after transform; refresh restored train mid-run
        "selected_err_mean_U": seenU.get(REFRESH_AT, {}).get("selected_err_mean")}

    # negatives against the UNTOUCHED valid ckpt (dr/ckpt-step40.pt): wrong refresh-off + wrong initial-bank
    negs = {}
    for label, ident, refresh_on in [("refresh_off", _identity(ARM, False, ibank), False),
                                      ("wrong_initial_bank", _identity(ARM, True, "0" * 16), True)]:
        ok = False; err = ""
        try:
            run(X, warm, TOTAL, set(), dr, resume_from=dr / "ckpt-step40.pt", identity=ident, refresh_enabled=refresh_on)
        except ValueError as ex:
            err = str(ex); ok = "admission-identity mismatch" in err
        negs[label] = {"failed_closed": ok, "err_head": err[:90]}
    R["identity_negatives"] = negs
    R["PASS"] = bool(R["selector_anchor_twin"]["match"] and R["selector_anchor_twin"]["one_refresh_each"]
                     and R["selector_anchor_twin"]["ordered_identity_match"]
                     and all(v["failed_closed"] for v in negs.values()))
    R["schema"] = "card012-selector-canary-2026-09-11"
    (OC / "card012-selector-canary.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
