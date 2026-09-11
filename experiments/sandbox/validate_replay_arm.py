"""Fail-closed completion validator for a cards006/007 replay arm (review ref 3).
Truncated/timed-out training is NOT done; a scientific-gate outcome is decided elsewhere.
Requires: fresh manifest/coords/model (mtime >= run_start), EXACT expected attempted steps
(140000), warm_start changed, finite coords + model outputs, all 3 inference snapshots present
+ fresh + LOADABLE, replay bank content sha == admission, warm hash == admission, the produced
active/holdout anchor IDs identical to the ORIGINAL card005/004 update (same split), and full
train_stats persisted. Exit 0 = validated DONE; nonzero = fail-closed STOP.
Usage: validate_replay_arm.py OUTD tag run_start_epoch expected_steps orig_active_ids orig_holdout_ids [snapshots_csv]
(snapshots_csv defaults to 35000,70000,140000; card008 70K arms pass 35000,70000)
"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np


def main():
    OUTD, tag, rs, exp = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    orig_active, orig_holdout = sys.argv[5], sys.argv[6]
    snaps = tuple(int(s) for s in (sys.argv[7] if len(sys.argv) > 7 else "35000,70000,140000").split(",") if s.strip())
    errs = []

    def fresh(p):
        return Path(p).exists() and Path(p).stat().st_mtime >= rs

    mp = OUTD / f"manifest-{tag}.json"; cp = OUTD / f"coords-{tag}.npy"; mdl = OUTD / f"model-{tag}.pt"
    adp = OUTD / f"admission-{tag}.json"; snapdir = OUTD / f"snapshots-{tag}"
    for p in (mp, cp, mdl, adp):
        if not fresh(p): errs.append(f"missing/stale {p.name}")
    if errs:
        print(json.dumps({"tag": tag, "PASS": False, "errors": errs})); return 3
    m = json.load(open(mp)); adm = json.load(open(adp))

    if int(m.get("executed_steps", -1)) != exp:
        errs.append(f"executed_steps {m.get('executed_steps')} != {exp}")
    if m.get("warm_start_changed") is not True:
        errs.append("warm_start_changed not True")
    if m.get("replay_bank_sha") != adm.get("replay_bank_content_sha"):
        errs.append(f"replay bank sha {m.get('replay_bank_sha')} != admission {adm.get('replay_bank_content_sha')}")
    if m.get("warm_start_hash") != adm.get("warm_start_hash"):
        errs.append("warm hash != admission")
    # Intervention + LR binding (frozen per-arm expected config; a deriv arm with the flag
    # accidentally OFF must FAIL here). Env: EXPECT_LR, EXPECT_DERIV_WEIGHT, EXPECT_DERIV_SUBBATCH,
    # EXPECT_DERIV_BANK_SHA (checked only when a derivative intervention is expected).
    exp_lr = float(os.environ.get("EXPECT_LR", "0.0001"))
    if abs(float(m.get("learning_rate", -1)) - exp_lr) > 1e-12:
        errs.append(f"learning_rate {m.get('learning_rate')} != expected {exp_lr}")
    exp_dw = float(os.environ.get("EXPECT_DERIV_WEIGHT", "0"))
    if abs(float(m.get("deriv_weight", 0) or 0) - exp_dw) > 1e-9:
        errs.append(f"deriv_weight {m.get('deriv_weight')} != expected {exp_dw} (deriv intervention mis-set)")
    if exp_dw > 0:
        exp_sha = os.environ.get("EXPECT_DERIV_BANK_SHA", "")
        if exp_sha and m.get("deriv_bank_sha") != exp_sha:
            errs.append(f"deriv_bank_sha {m.get('deriv_bank_sha')} != expected {exp_sha}")
        if int(m.get("deriv_subbatch", 0) or 0) != int(os.environ.get("EXPECT_DERIV_SUBBATCH", "128")):
            errs.append(f"deriv_subbatch {m.get('deriv_subbatch')} != expected")
    ts = m.get("train_stats") or {}
    if not ts or "executed_iters" not in ts:
        errs.append("train_stats not persisted")
    # finite coords
    c = np.load(cp)
    if not np.isfinite(c).all():
        errs.append("coords non-finite")
    # all expected snapshots present, fresh, loadable + finite model output
    for s in snaps:
        sp = snapdir / f"model-step{s}.pt"
        if not fresh(sp):
            errs.append(f"snapshot model-step{s}.pt missing/stale"); continue
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
            from basemap.pumap.parametric_umap.core import ParametricUMAP
            import torch
            mo = ParametricUMAP.load(str(sp), device="cpu"); mo.model.eval()
            d = next(m.in_features for m in mo.model.modules() if isinstance(m, torch.nn.Linear))
            with torch.no_grad():
                probe = torch.nn.functional.normalize(torch.randn(8, d), dim=1)
                z = mo.model(probe).float().numpy()
            if not np.isfinite(z).all():
                errs.append(f"snapshot model-step{s}.pt produced non-finite output")
        except Exception as e:
            errs.append(f"snapshot model-step{s}.pt not loadable: {e}")
    # produced anchor ids identical to the ORIGINAL update's split
    for produced, orig, nm in ((OUTD / f"anchor_active_ids-{tag}.npy", orig_active, "active"),
                               (OUTD / f"anchor_holdout_ids-{tag}.npy", orig_holdout, "holdout")):
        if not fresh(produced):
            errs.append(f"{nm} ids missing/stale"); continue
        if not Path(orig).exists():
            errs.append(f"original {nm} ids not found at {orig}"); continue
        a = np.sort(np.load(produced)); b = np.sort(np.load(orig))
        if a.shape != b.shape or not np.array_equal(a, b):
            errs.append(f"{nm} anchor ids differ from original card005/004 split")

    ok = not errs
    print(json.dumps({"tag": tag, "PASS": ok, "executed_steps": m.get("executed_steps"),
                      "snapshots_ok": all(f"model-step{s}" not in e for s in snaps for e in errs),
                      "errors": errs}, indent=1))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
