"""Card012 Hook B boundary canary (per card012-hookB-overseer-review, fixes 1+2). A SMALL representative
PERM graph (multi-epoch cheaply) proves:
  T1  LEGACY epoch-resume with Hook B OFF is bitwise-invisible (regression proof: cards 006-009 unaffected).
  T2  mid-epoch STEP resume that CROSSES an epoch boundary is bitwise-invisible (with a bank refresh).
  T3  refresh/admission IDENTITY negative: resuming a VALID untouched ckpt with a WRONG arm identity fails
      closed with the EXPECTED admission-identity error (not any ValueError).
Artifacts are PRESERVED under card012-boundary-canary/. Exit 0 = PASS. Usage: gpu_card012_boundary_canary.py
"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
HEAD = SB / "dino-arrival-t0/champion-bs16k/model.pt"
WORK = SB / "card012-boundary-canary"; WORK.mkdir(exist_ok=True)
SEED = 42; N = 3000; K = 15


def _state_sha(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def build_inputs():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, 1536)).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
    np.save(WORK / "sub.npy", X.astype(np.float16))
    src = np.repeat(np.arange(N), K); dst = rng.integers(0, N, N * K).astype(np.int64)
    np.savez(WORK / "edges.npz", sources=src.astype(np.int64), targets=dst,
             weights=np.ones(N * K, np.float32), n_nodes=np.int64(N))
    # small replay banks (initial + alternate) from a T0 projection of a subset
    m = ParametricUMAP.load(str(HEAD), device="cpu").model.eval()
    idx = np.sort(rng.choice(N, 500, replace=False))
    with torch.no_grad():
        tgt = m(torch.from_numpy(X[idx])).float().numpy().astype(np.float32)
    for nm, perm in (("bank0", np.arange(500)), ("bank1", rng.permutation(500))):
        np.savez(WORK / f"{nm}.npz", replay_X=X[idx][perm].astype(np.float16), replay_targets=tgt[perm],
                 replay_ids=idx[perm].astype(np.int64), source=np.array(["s"] * 500))


def run(max_steps, ckpt_epochs=0, step_targets=None, refresh_at=None, refresh_bank=None,
        resume_from=None, ckpt_dir=None, identity=None, hookb=True):
    X = np.asarray(np.load(WORK / "sub.npy", mmap_mode="r"), np.float32)
    pu = ParametricUMAP.load(str(HEAD), device="cuda"); warm = {k: v.detach().clone() for k, v in pu.model.state_dict().items()}
    pu.model = None
    pu.learning_rate = 1e-4; pu.lr_schedule = "constant"; pu.batch_size = 16384; pu.warmup_steps = 0
    pu.n_epochs = 10000; pu._max_train_steps = max_steps
    pu.rankneg_window = 750   # valid for N=3000 (T0's 500000 exceeds n_nodes); keeps rank negatives ACTIVE
    pu.replay_bank_path = str(WORK / "bank0.npz"); pu.replay_weight = 0.02; pu.replay_fraction = 0.05; pu.replay_seed = 51549
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(pu, a): setattr(pu, a, v)
    if hookb:
        if step_targets: pu._checkpoint_step_targets = set(step_targets)
        if refresh_at is not None:
            pu._replay_refresh_fn = lambda _pu, _s: str(WORK / f"{refresh_bank}.npz")
            pu._replay_refresh_steps = {refresh_at}
        if identity is not None: pu._card012_identity = identity
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    kw = dict(precomputed_edges_path=str(WORK / "edges.npz"), random_state=SEED, verbose=False,
              checkpoint_dir=str(ckpt_dir) if ckpt_dir else None, checkpoint_every_epochs=ckpt_epochs)
    if resume_from: kw["resume_from"] = str(resume_from)
    else: kw["warm_start_state"] = warm
    pu.fit(X, **kw)
    return _state_sha(pu.model), dict(getattr(pu, "_train_stats", {}) or {})


def main():
    build_inputs()
    R = {}
    # epoch length ~ ceil(45000/1638)=28 batches; 3 epochs ~84 steps.
    # T1: LEGACY epoch resume, Hook B OFF. SEPARATE clean dirs for U vs stop vs resume, and resume from the
    # STOPPED run's OWN explicit epoch checkpoint (not U's) — provenance fix per overseer review.
    d1u = WORK / "t1_u"; d1s = WORK / "t1_stop"; d1u.mkdir(exist_ok=True); d1s.mkdir(exist_ok=True)
    shaU1, _ = run(84, ckpt_epochs=1, ckpt_dir=d1u, hookb=False)              # U: full 0->84, own dir
    run(60, ckpt_epochs=1, ckpt_dir=d1s, hookb=False)                        # stopped 0->60 -> enters epoch2, writes ckpt-epoch2@56
    stop_ck = d1s / "ckpt-epoch2.pt"
    assert stop_ck.exists(), "stopped run did not write ckpt-epoch2"
    _sc = torch.load(str(stop_ck), map_location="cpu", weights_only=False)
    stop_step = int(_sc["global_step"]); assert not _sc.get("step_checkpoint", False), "T1 must use an EPOCH ckpt"
    shaR1, _ = run(84, ckpt_epochs=1, ckpt_dir=d1s, resume_from=stop_ck, hookb=False)   # resume 56->84 (>=1 epoch)
    R["T1_legacy_epoch_resume_off"] = {"sha_U": shaU1, "sha_R": shaR1, "match": bool(shaU1 == shaR1),
                                       "resumed_from": stop_ck.name, "resume_start_step": stop_step,
                                       "resume_end_step": 84, "resumed_work_steps": 84 - stop_step,
                                       "provenance_ok": bool(stop_step < 84 and (84 - stop_step) >= 28
                                                             and str(d1s) != str(d1u))}
    # T2: mid-epoch STEP resume crossing an epoch boundary, refresh@40 (epoch2)
    idy = {"arm": "uniform", "refresh_enabled": True, "refresh_steps": [40], "pool_ids_sha": "x", "selection_seed": 1}
    d2 = WORK / "t2"; d2.mkdir(exist_ok=True)
    shaU2, stU2 = run(84, step_targets={40}, refresh_at=40, refresh_bank="bank1", ckpt_dir=d2, identity=idy)
    run(35, step_targets={35}, refresh_at=40, refresh_bank="bank1", ckpt_dir=d2, identity=idy)  # ckpt@35 (epoch2, before refresh@40)
    shaR2, stR2 = run(84, refresh_at=40, refresh_bank="bank1", ckpt_dir=d2, resume_from=d2 / "ckpt-step35.pt", identity=idy)
    R["T2_midepoch_cross_boundary"] = {"sha_U": shaU2, "sha_R": shaR2, "match": bool(shaU2 == shaR2),
                                       "U_refreshes": stU2.get("replay_refreshes"), "R_refreshes": stR2.get("replay_refreshes")}
    # T3: refresh/admission IDENTITY negative — resume a VALID untouched ckpt with a WRONG arm
    neg_ok = False; err = ""
    try:
        wrong = dict(idy); wrong["arm"] = "error_directed"
        run(84, refresh_at=40, refresh_bank="bank1", ckpt_dir=d2, resume_from=d2 / "ckpt-step35.pt", identity=wrong)
    except ValueError as ex:
        err = str(ex); neg_ok = "admission-identity mismatch" in err
    R["T3_identity_negative"] = {"failed_closed": neg_ok, "expected_error": bool(neg_ok), "err_head": err[:120]}
    R["PASS"] = bool(R["T1_legacy_epoch_resume_off"]["match"] and R["T1_legacy_epoch_resume_off"]["provenance_ok"]
                     and R["T2_midepoch_cross_boundary"]["match"] and neg_ok)
    R["schema"] = "card012-boundary-canary-2026-09-11"
    (OC / "card012-boundary-canary.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
