#!/usr/bin/env python3
"""#11 checkpointing ACCEPTANCE TEST (2026-08-30). GPU.

The gate: a resume must be BITWISE-INVISIBLE. Train a short SEEDED device-path run with
checkpoint_every_epochs=1 (full run -> TW + ckpts); then a fresh model resumes from a mid ckpt and
finishes -> TR. ASSERT TR == TW (post-fit param sha256, exact). If any RNG stream was missed the hash
diverges — that's the point. Fixture: probe-lang-arb_Arab-jina (100k x 768), champion recipe, 4 epochs.
"""
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

SB = Path("/data/latent-basemap/sandbox")
SUBSTRATES = Path("/data/latent-basemap/substrates")


def _param_sha(model):
    h = hashlib.sha256()
    for p in model.model.parameters():
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from knobs_2m import BASE_KWARGS, MD
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    ds = "probe-lang-arb_Arab-jina"
    sub = SUBSTRATES / ds / "substrate.f16.npy"
    edges = SB / ds / "edges-k15-fuzzy.npz"
    X = np.asarray(np.load(sub, mmap_mode="r"), dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0; X = X / n
    SEED = 42
    N_EPOCHS = 4

    def _kwargs():
        k = dict(BASE_KWARGS)
        k.update({"low_dim_kernel": "umap", **MD["000"], "n_epochs": N_EPOCHS,
                  "batch_size": 4096, "pos_ratio": 0.10, "fneg_weight": 1.0,
                  "neg_tanh_gamma": 4.0, "rankneg_window": 25_000,
                  "total_steps_estimate": 100_000})
        return k

    tmp = Path(tempfile.mkdtemp(prefix="ckpt-test-"))
    ck_full = tmp / "full"; ck_full.mkdir()

    # --- FULL run (uninterrupted) -> TW + checkpoints ---
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    m_full = ParametricUMAP(**_kwargs())
    m_full.fit(X, precomputed_edges_path=str(edges), random_state=SEED,
               verbose=False, checkpoint_every_epochs=1, checkpoint_dir=str(ck_full))
    TW = _param_sha(m_full)
    init_full = getattr(m_full, "init_state_sha256", None)
    cks = sorted(ck_full.glob("ckpt-epoch*.pt"))
    print(f"full run: trained_sha={TW} init={init_full}; ckpts={[c.name for c in cks]}", flush=True)
    assert cks, "no checkpoints written"
    # pick the earliest surviving ckpt (keep-last-2 leaves epoch2,3 for N_EPOCHS=4)
    resume_ck = cks[0]

    # --- RESUME run from a mid checkpoint -> TR ---
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    m_res = ParametricUMAP(**_kwargs())
    m_res.fit(X, precomputed_edges_path=str(edges), random_state=SEED,
              verbose=False, checkpoint_every_epochs=1, checkpoint_dir=str(tmp / "resume"),
              resume_from=str(resume_ck))
    TR = _param_sha(m_res)
    print(f"resume run (from {resume_ck.name}): trained_sha={TR}", flush=True)

    ok = TR == TW
    print(f"\n=== ACCEPTANCE: TR({TR}) {'==' if ok else '!='} TW({TW}) -> "
          f"{'PASS (resume is bitwise-invisible)' if ok else 'FAIL (a RNG stream was missed)'} ===",
          flush=True)
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
