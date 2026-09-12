"""Card012 Hook B resume-twin canary. Proves the step-based checkpoint + mid-run bank refresh give a
BITWISE-INVISIBLE mid-epoch resume, tested BEFORE and AFTER a refresh, on the REAL fit() path with replay
ON (2.4M final graph, T0 warm-start, rankneg active). Also a fail-closed negative control (wrong bank).
Small step targets keep everything inside epoch 0. Exit 0 = PASS. Usage: gpu_card012_resume_canary.py
"""
import os, sys, json, hashlib, tempfile, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
HEAD = SB / "dino-arrival-t0/champion-bs16k/model.pt"
EDGES = SB / "dino-arrival-final/edges-k15-fuzzy.npz"
SUB = Path("/data/latent-basemap/substrates/dino-arrival-final/substrate.f16.npy")
OUT_BANK = OC / "card006_out_bank.npz"
SEED = 42; REFRESH_AT = 100; TOTAL = 200
TMP = Path(tempfile.mkdtemp(prefix="card012canary_", dir="/data/latent-basemap/sandbox"))
ALT_BANK = TMP / "altbank.npz"


def _state_sha(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _make_alt_bank():
    z = np.load(OUT_BANK); rng = np.random.default_rng(777)
    p = rng.permutation(z["replay_ids"].shape[0])                       # deterministic row permutation -> different sha
    np.savez(ALT_BANK, replay_X=z["replay_X"][p], replay_targets=z["replay_targets"][p],
             replay_ids=z["replay_ids"][p], source=z["source"][p])


def _refresh_fn(pu, step):
    return str(ALT_BANK)                                                # deterministic alternate bank


def _cfg(pu):
    pu.learning_rate = 1e-4; pu.lr_schedule = "constant"; pu.batch_size = 16384
    pu.warmup_steps = 0; pu.n_epochs = 10000
    pu.replay_bank_path = str(OUT_BANK); pu.replay_weight = 0.02; pu.replay_fraction = 0.05; pu.replay_seed = 51549
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(pu, a): setattr(pu, a, v)


def run_seg(X, warm, max_steps, ckpt_targets, ckpt_dir, resume_from=None, refresh=True):
    pu = ParametricUMAP.load(str(HEAD), device="cuda"); pu.model = None
    _cfg(pu); pu._max_train_steps = max_steps
    if refresh:
        pu._replay_refresh_fn = _refresh_fn; pu._replay_refresh_steps = {REFRESH_AT}
    if ckpt_targets:
        pu._checkpoint_step_targets = set(ckpt_targets)
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    kw = dict(precomputed_edges_path=str(EDGES), random_state=SEED, verbose=False,
              checkpoint_dir=str(ckpt_dir))
    if resume_from:
        kw["resume_from"] = str(resume_from)
    else:
        kw["warm_start_state"] = warm
    pu.fit(X, **kw)
    return _state_sha(pu.model), dict(getattr(pu, "_train_stats", {}) or {})


def main():
    _make_alt_bank()
    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)
    p0 = ParametricUMAP.load(str(HEAD), device="cuda")
    warm = {k: v.detach().clone() for k, v in p0.model.state_dict().items()}; del p0
    R = {}

    # Twin U: uninterrupted, refresh@100
    du = TMP / "cu"; sha_U, st_U = run_seg(X, warm, TOTAL, set(), du)
    R["sha_U"] = sha_U; R["U_refreshes"] = st_U.get("replay_refreshes")

    # Twin R-afterrefresh: ckpt@140 (after the 100 refresh), resume -> 200
    da = TMP / "ca"; run_seg(X, warm, 140, {140}, da)
    sha_A, st_A = run_seg(X, warm, TOTAL, set(), da, resume_from=da / "ckpt-step140.pt")
    R["sha_after"] = sha_A; R["after_match"] = bool(sha_A == sha_U)

    # Twin R-straddle: ckpt@60 (before refresh), resume -> 200 (refresh@100 fires post-resume)
    ds = TMP / "cs"; run_seg(X, warm, 60, {60}, ds)
    sha_S, st_S = run_seg(X, warm, TOTAL, set(), ds, resume_from=ds / "ckpt-step60.pt")
    R["sha_straddle"] = sha_S; R["straddle_match"] = bool(sha_S == sha_U)
    R["straddle_refreshes"] = st_S.get("replay_refreshes")

    # Negative control: resume the @140 ckpt but tamper current_replay_bank_path -> wrong bank => fail-closed sha match
    neg_ok = False
    try:
        ck = torch.load(str(da / "ckpt-step140.pt"), map_location="cpu", weights_only=False)
        ck["current_replay_bank_path"] = str(OUT_BANK)                 # WRONG (should be alt post-refresh)
        bad = TMP / "ckpt-bad.pt"; torch.save(ck, bad)
        run_seg(X, warm, TOTAL, set(), da, resume_from=bad)
    except (ValueError, AssertionError) as ex:
        neg_ok = True; R["neg_control_error"] = str(ex)[:160]
    R["negative_control_failed_closed"] = neg_ok

    R["PASS"] = bool(R["after_match"] and R["straddle_match"] and neg_ok)
    R["schema"] = "card012-resume-canary-2026-09-11"
    (OC / "card012-resume-canary.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1), flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
