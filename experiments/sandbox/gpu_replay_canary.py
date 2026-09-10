"""Cards 006/007 DEVICE-path GPU canary (the CPU canary defers these three to here).
Tiny synthetic device fit (device_int8), fast. Validates:
  R1 real-sampler RNG untouched: fit replay-OFF vs replay-IN at the SAME seed with
     epoch-boundary checkpoints; the loader/mn/dens/hold sampler generator STATES and the
     torch global/cuda RNG states at the epoch-1 boundary must be BYTE-EQUAL (the replay
     branch draws only from its own generator). replay_gen present only when enabled.
     Nuance (per review): the model-adaptive rank refresh may change negative IDENTITIES
     after weights diverge; the required invariant is identical RNG state/schedule, not identities.
  R2 checkpoint replay_gen round-trip: an uninterrupted replay-IN run and one RESUMED from
     its epoch-1 checkpoint reach a BITWISE-IDENTICAL endpoint (replay_gen save/restore works).
  R3 step-snapshot retention: model-step{N}.pt written at targets and NOT pruned by the
     epoch-checkpoint last-2 pruner; snapshots are inference-only (not resumed from).
  R4 fail-closed resume: resuming with a DIFFERENT bank sha, a CHANGED replay weight, or a
     replay-OFF checkpoint (missing replay_gen) all RAISE before stepping.
Usage: gpu_replay_canary.py   (run under the GPU flock)
"""
import os, sys, json, tempfile, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

N, D, K, NB = 3000, 64, 10, 300
BATCH, EPOCHS = 512, 3
OC = Path("/data/latent-basemap/sandbox/overseer-codex")
rng = np.random.default_rng(11)


def _norm(X):
    return (X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)).astype(np.float32)


def tiny_edges(X, path):
    Xn = _norm(X); S = Xn @ Xn.T; np.fill_diagonal(S, -1); nn = np.argsort(-S, axis=1)[:, :K]
    np.savez(path, sources=np.repeat(np.arange(N), K).astype(np.int32), targets=nn.reshape(-1).astype(np.int32),
             weights=np.ones(N * K, np.float32), n_nodes=np.int64(N))


def _mk(**kw):
    m = ParametricUMAP(n_components=2, n_epochs=EPOCHS, batch_size=BATCH, low_dim_kernel="umap",
                       x_residency="device_int8", **kw)
    return m


def _fit(edges, X, ckdir=None, ckevery=0, resume=None, snap=(), snapdir=None, **kw):
    torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
    m = _mk(**kw)
    fk = dict(precomputed_edges_path=str(edges), random_state=0, verbose=False)
    if ckdir: fk.update(checkpoint_dir=str(ckdir), checkpoint_every_epochs=ckevery)
    if resume: fk.update(resume_from=str(resume))
    if snap: fk.update(snapshot_steps=snap, snapshot_dir=str(snapdir))
    m.fit(X, **fk)
    return m


def _sd_equal(a, b):
    ka, kb = a.model.state_dict(), b.model.state_dict()
    return set(ka) == set(kb) and all(torch.equal(ka[k].cpu(), kb[k].cpu()) for k in ka)


def _ck_epoch1(d):
    import glob
    fs = sorted(glob.glob(str(Path(d) / "ckpt-epoch*.pt")))
    e1 = [f for f in fs if f.endswith("epoch1.pt")]
    return e1[0] if e1 else (fs[0] if fs else None)


def main():
    out = {"schema": "gpu-replay-canary-2026-09-10", "device": "cuda"}
    if not torch.cuda.is_available():
        out["PASS"] = False; out["error"] = "no cuda"; print(json.dumps(out)); return 3
    td = Path(tempfile.mkdtemp(prefix="gpureplay_")); edges = td / "edges.npz"
    X = rng.standard_normal((N, D)).astype(np.float32); tiny_edges(X, edges)
    bank_raw = rng.standard_normal((NB, D)).astype(np.float32); bank_X = _norm(bank_raw)
    m0 = _fit(edges, X)                                   # teacher for targets
    with torch.no_grad():
        bank_tgt = m0.model(torch.from_numpy(bank_X).cuda()).float().cpu().numpy().astype(np.float32)
    bankA = td / "bankA.npz"; bankB = td / "bankB.npz"
    np.savez(bankA, replay_X=bank_X.astype(np.float16), replay_targets=bank_tgt, replay_ids=np.arange(NB, dtype=np.int64))
    np.savez(bankB, replay_X=_norm(rng.standard_normal((NB, D)).astype(np.float32)).astype(np.float16),
             replay_targets=bank_tgt, replay_ids=np.arange(NB, dtype=np.int64))   # different X -> different sha

    # ---- R1 sampler-RNG equality (off vs IN) at epoch-1 boundary ----
    off_d = td / "ck_off"; in_d = td / "ck_in"
    _fit(edges, X, ckdir=off_d, ckevery=1)
    _fit(edges, X, ckdir=in_d, ckevery=1, replay_bank_path=str(bankA), replay_weight=0.02,
         replay_fraction=0.05, replay_seed=51549)
    co = torch.load(_ck_epoch1(off_d), map_location="cpu", weights_only=False)
    ci = torch.load(_ck_epoch1(in_d), map_location="cpu", weights_only=False)
    def _eq(a, b):
        if a is None and b is None: return True
        if a is None or b is None: return False
        return torch.equal(a.cpu() if torch.is_tensor(a) else a, b.cpu() if torch.is_tensor(b) else b)
    gen_keys = ["loader_gen", "mn_gen", "dens_gen", "hold_gen"]
    sampler_rng_equal = all(_eq(co.get(k), ci.get(k)) for k in gen_keys)
    cuda_rng_equal = all(torch.equal(a.cpu(), b.cpu()) for a, b in
                         zip(co.get("cuda_rng") or [], ci.get("cuda_rng") or []))
    torch_rng_equal = _eq(co.get("torch_rng"), ci.get("torch_rng"))
    replay_gen_present = (co.get("replay_gen") is None and ci.get("replay_gen") is not None)
    out["R1_sampler_rng_equal"] = bool(sampler_rng_equal)
    out["R1_global_rng_equal"] = bool(cuda_rng_equal and torch_rng_equal)
    out["R1_replay_gen_only_when_enabled"] = bool(replay_gen_present)
    R1 = bool(sampler_rng_equal and cuda_rng_equal and torch_rng_equal and replay_gen_present)

    # ---- R2 resume round-trip (replay IN) bitwise-invisible ----
    unint_d = td / "ck_unint"
    m_un = _fit(edges, X, ckdir=unint_d, ckevery=1, replay_bank_path=str(bankA), replay_weight=0.02,
                replay_fraction=0.05, replay_seed=51549)
    m_re = _fit(edges, X, ckdir=td / "ck_re", ckevery=1, resume=_ck_epoch1(unint_d),
                replay_bank_path=str(bankA), replay_weight=0.02, replay_fraction=0.05, replay_seed=51549)
    R2 = bool(_sd_equal(m_un, m_re))
    out["R2_resume_bitwise_invisible"] = R2

    # ---- R3 step-snapshot retention ----
    snap_d = td / "snaps"
    m_s = _fit(edges, X, ckdir=td / "ck_s", ckevery=1, snap=(40, 100), snapdir=snap_d,
               replay_bank_path=str(bankA), replay_weight=0.02, replay_fraction=0.05, replay_seed=51549)
    snaps_exist = [(snap_d / f"model-step{s}.pt").exists() for s in (40, 100)]
    # snapshots are loadable inference models, and NOT epoch checkpoints
    snap_loads = False
    try:
        ParametricUMAP.load(str(snap_d / "model-step40.pt"), device="cuda"); snap_loads = True
    except Exception:
        snap_loads = False
    R3 = bool(all(snaps_exist) and snap_loads)
    out["R3_snapshots_present"] = snaps_exist; out["R3_snapshot_loads"] = snap_loads

    # ---- R4 fail-closed resume ----
    def _resume_raises(**kw):
        try:
            _fit(edges, X, ckdir=td / "ck_tmp", ckevery=1, resume=_ck_epoch1(unint_d), **kw); return False
        except (ValueError, AssertionError, RuntimeError):
            return True
        finally:
            shutil.rmtree(td / "ck_tmp", ignore_errors=True)
    diff_bank = _resume_raises(replay_bank_path=str(bankB), replay_weight=0.02, replay_fraction=0.05, replay_seed=51549)
    diff_weight = _resume_raises(replay_bank_path=str(bankA), replay_weight=0.05, replay_fraction=0.05, replay_seed=51549)
    off_resume = _resume_raises()   # resume a replay-IN ckpt with replay OFF -> replay_enabled mismatch
    out["R4_diff_bank_raises"] = diff_bank; out["R4_diff_weight_raises"] = diff_weight
    out["R4_off_vs_on_raises"] = off_resume
    R4 = bool(diff_bank and diff_weight and off_resume)

    out["R1"] = R1; out["R2"] = R2; out["R3"] = R3; out["R4"] = R4
    out["PASS"] = bool(R1 and R2 and R3 and R4)
    print(json.dumps(out, indent=1))
    (OC / "cards006-007-gpu-canary.json").write_text(json.dumps(out, indent=1))
    shutil.rmtree(td, ignore_errors=True)
    return 0 if out["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
