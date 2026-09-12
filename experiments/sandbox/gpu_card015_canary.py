"""Card015 Hook C canary. On the REAL fit path (card010 300K draw + fixed15 graph + shared init, LR 0.001):
  - radii == ONES reproduces the baseline fit BITWISE (validates the id-stash gather + detached rescale as a
    true no-op at unit scale);
  - actual radii DIVERGE from baseline (the local-scale kernel is active);
  - off-path (no radii) == baseline (default-off).
Short dose. Exit 0 = PASS. Usage: gpu_card013_canary.py
"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
CHAMPION = SB / "dino-arrival-t0/champion-bs16k/model.pt"
SUBD = Path("/data/latent-basemap/substrates/card010-adaptive")
SUB = SUBD / "substrate.f16.npy"; EDGES = SUBD / "edges-fixed15.npz"; INIT = SB / "card015-init/init-card015-3d.pt"
SEED = 42; SHORT = 200


def _sha(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def run(radii):
    init = torch.load(str(INIT), map_location="cpu", weights_only=False)["model_state"]
    p = ParametricUMAP.load(str(CHAMPION), device="cuda"); p.model = None; p.n_components = 3
    p.learning_rate = 0.001; p.lr_schedule = "constant"; p.batch_size = 16384; p.warmup_steps = 0
    p.n_epochs = 10000; p._max_train_steps = SHORT; p.rankneg_window = 75000
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(p, a): setattr(p, a, v)
    if radii is not None:
        p._card013_radii = radii
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)
    p.fit(X, precomputed_edges_path=str(EDGES), random_state=SEED, verbose=False, warm_start_state=init)
    assert p._train_stats["positive_lr_optimizer_steps"] == SHORT
    assert p.model.proj_out.out_features == 3
    assert all(torch.isfinite(t).all() for t in p.model.state_dict().values())
    return _sha(p.model)


def main():
    ones = np.ones(300000, np.float32)
    actual = np.load(SB / "card013-radii/r_actual.npy").astype(np.float32)
    sha_base = run(None)
    sha_ones = run(ones)
    sha_act = run(actual)
    R = {"schema": "card015-canary-2026-09-12", "sha_baseline": sha_base, "sha_ones": sha_ones, "sha_actual": sha_act,
         "radius1_bitwise_baseline": bool(sha_ones == sha_base),
         "actual_diverges": bool(sha_act != sha_base)}
    R["PASS"] = bool(R["radius1_bitwise_baseline"] and R["actual_diverges"])
    (OC / "card015-canary.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
