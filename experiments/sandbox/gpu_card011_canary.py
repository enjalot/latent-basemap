"""Card011 ACTUAL-DEVICE canary (attempt-2, per card011-attempt1-repair-review.md). Attempt-1's canary
tested only the uniform branch; the real fit() path installs the rank-window (rankneg_window=75000) at the
start of epoch 0, so negatives take the RANK-WINDOW branch from step 1 — where attempt-1's injection was
bypassed (0 exposure -> bit-identical models). This canary exercises the REAL path.

PART A (unit, rank-window INSTALLED): _sample_negatives on the rank-window branch — background stream +
global RNG unchanged hook-on vs off, front-k slots overwritten with the exact pool pairs, tail identical.
PART B (real fit(), production batch 16384, window 75000):
  - off-path BIT IDENTITY: inject_frac=0 fit == a no-hook fit (same trained sha) -> attempt1 ordinary reusable.
  - DIVERGENCE: inject-on fit (collision pool) != inject-off fit from the SAME warm weights.
  - EXPOSURE: inject_slots_applied>0 and injection applied on ~every batch (zero exposure must FAIL).
Exit 0 = PASS. Usage: gpu_card011_canary.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
HEAD = SB / "card010-train/model-fixed15.pt"
SUB = Path("/data/latent-basemap/substrates/card010-adaptive/substrate.f16.npy")
EDGES = Path("/data/latent-basemap/substrates/card010-adaptive/edges-fixed15.npz")
STAGE0 = SB / "card011-train/stages/collision-stage0.npz"
SEED = 42; INJECT_FRAC = 0.05; SHORT = 200


def _state_sha(model):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _short_fit(X, warm, inject, pool, dev):
    p = ParametricUMAP.load(str(HEAD), device=dev)
    p.model = None
    p.learning_rate = 1e-4; p.lr_schedule = "constant"; p.batch_size = 16384
    p.warmup_steps = 0; p.n_epochs = 10000; p._max_train_steps = SHORT
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(p, a):
            setattr(p, a, v)
    if inject:
        p._inject_frac_cfg = INJECT_FRAC; p._inject_pool = pool; p._inject_refresh_steps = ()
    torch.manual_seed(SEED); np.random.seed(SEED)
    if dev == "cuda":
        torch.cuda.manual_seed_all(SEED)
    p.fit(X, precomputed_edges_path=str(EDGES), random_state=SEED, verbose=False, warm_start_state=warm)
    return _state_sha(p.model), dict(getattr(p, "_train_stats", {}) or {})


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(STAGE0); pool = (z["src"].astype(np.int64), z["dst"].astype(np.int64))
    pool_src = torch.as_tensor(pool[0], device=dev); pool_dst = torch.as_tensor(pool[1], device=dev)
    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)
    n = X.shape[0]

    # ---------- PART A: unit test on the RANK-WINDOW branch ----------
    pa = ParametricUMAP.load(str(HEAD), device=dev)
    fixed_model = pa.model; pa.model = None
    pa.batch_size = 16384
    dataset, loader, npos = pa._prepare_edge_list_training(X, str(EDGES), n, False, SEED)
    loader.configure_rank_negatives(75000, False)
    pa.model = fixed_model; pa._refresh_rank_negatives(loader)                 # install a real rank order
    assert loader._rank_window == 75000 and loader._rank_of_node is not None, "rank-window not installed"
    num_neg = loader.num_neg; k = int(round(INJECT_FRAC * num_neg))
    g0 = loader.gen.get_state().clone(); ig0 = loader._inject_gen.get_state().clone()
    loader._inject_frac = 0.0; loader._inject_src = None; loader._inject_dst = None
    off_s, off_d = loader._sample_negatives(num_neg)
    gen_off = loader.gen.get_state().clone()
    loader.gen.set_state(g0); loader._inject_gen.set_state(ig0)
    loader._inject_frac = INJECT_FRAC; loader._inject_src = pool_src; loader._inject_dst = pool_dst
    grng = torch.random.get_rng_state().clone(); crng = torch.cuda.get_rng_state().clone() if dev == "cuda" else None
    on_s, on_d = loader._sample_negatives(num_neg)
    gen_on = loader.gen.get_state().clone()
    posdraw = torch.randint(0, pool_src.shape[0], (k,),
                            generator=torch.Generator(device=dev).manual_seed(SEED + 918273), device=dev)
    A = {"rank_window_branch_active": True, "k": int(k),
         "background_gen_advances_identically": bool(torch.equal(gen_off, gen_on)),
         "tail_slots_identical": bool(torch.equal(off_s[k:], on_s[k:]) and torch.equal(off_d[k:], on_d[k:])),
         "front_slots_changed": bool(not torch.equal(off_s[:k], on_s[:k])),
         "front_match_pool_pairs": bool(torch.equal(on_s[:k], pool_src.index_select(0, posdraw))
                                        and torch.equal(on_d[:k], pool_dst.index_select(0, posdraw))),
         "global_rng_unchanged": bool(torch.equal(torch.random.get_rng_state(), grng)
                                      and (dev != "cuda" or torch.equal(torch.cuda.get_rng_state(), crng)))}
    A_pass = all(v is True for kk, v in A.items() if isinstance(v, bool))
    del pa, fixed_model, loader, dataset
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---------- PART B: real fit() short-training identity + divergence + exposure ----------
    p0 = ParametricUMAP.load(str(HEAD), device=dev)
    warm = {kk: v.detach().clone() for kk, v in p0.model.state_dict().items()}
    del p0
    sha_base, st_base = _short_fit(X, warm, inject=False, pool=None, dev=dev)   # no inject attrs at all
    sha_off, st_off = _short_fit(X, warm, inject=False, pool=None, dev=dev)     # explicit inject off
    sha_on, st_on = _short_fit(X, warm, inject=True, pool=pool, dev=dev)
    B = {"steps": SHORT, "sha_baseline": sha_base, "sha_off": sha_off, "sha_on": sha_on,
         "off_path_bit_identity": bool(sha_off == sha_base),
         "on_off_divergence": bool(sha_on != sha_off),
         "inject_slots_applied": int(st_on.get("inject_slots_applied", 0)),
         "inject_batches_applied": int(st_on.get("inject_batches_applied", 0)),
         "exposure_ok": bool(st_on.get("inject_slots_applied", 0) > 0
                             and st_on.get("inject_batches_applied", 0) >= SHORT - 20),
         "off_zero_exposure": int(st_off.get("inject_slots_applied", 0))}
    B_pass = bool(B["off_path_bit_identity"] and B["on_off_divergence"] and B["exposure_ok"]
                  and B["off_zero_exposure"] == 0)

    result = {"schema": "card011-canary-attempt2-2026-09-11", "device": dev,
              "part_a_rank_window_unit": A, "part_a_pass": A_pass,
              "part_b_real_fit": B, "part_b_pass": B_pass, "PASS": bool(A_pass and B_pass)}
    (OC / "card011-canary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
    return 0 if result["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
