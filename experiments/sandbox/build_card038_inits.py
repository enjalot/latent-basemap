"""Card038 fresh 3D init construction. Builds a ResidualBottleneckMLP (3 outputs, 3 layers, neck 0.75) at
each width (2048, 1024) independently at torch seed 42 and persists the init state + full hashes BEFORE any
GPU use (different-width states cannot be bitwise identical). CPU. Usage: build_card038_inits.py
"""
import os, sys, json, hashlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
CHAMPION = SB / "dino-arrival-t0/champion-bs16k/model.pt"
OUT = SB / "card038-init"; OUT.mkdir(parents=True, exist_ok=True)
WIDTH = {"wide2048": 2048, "compact1024": 1024}; SEED = 42


def _state_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd): h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]
def full_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def main():
    rec = {"schema": "card038-inits-2026-09-12", "seed": SEED, "n_components": 3, "arms": {}}
    for arm, hd in WIDTH.items():
        p = ParametricUMAP.load(str(CHAMPION), device="cpu"); p.model = None; p.n_components = 3; p.hidden_dim = hd
        torch.manual_seed(SEED); np.random.seed(SEED); p._init_model(1536)
        sd = {k: v.detach().clone() for k, v in p.model.state_dict().items()}
        nparam = int(sum(t.numel() for t in p.model.parameters()))
        assert p.model.proj_out.out_features == 3 and p.neck_fraction == 0.75 and p.n_layers == 3
        named = _state_sha(sd); path = OUT / f"init-{arm}.pt"
        torch.save({"model_state": sd, "init_state_sha256": named, "n_components": 3, "hidden_dim": hd, "n_params": nparam}, path)
        rec["arms"][arm] = {"hidden_dim": hd, "n_params": nparam, "init_state_sha256": named, "init_file_sha256": full_sha(path), "path": str(path)}
        print(f"{arm}: H{hd} params={nparam} init_state_sha={named} file_sha={rec['arms'][arm]['init_file_sha256'][:16]}")
    assert rec["arms"]["wide2048"]["n_params"] == 14170627 and rec["arms"]["compact1024"]["n_params"] == 4332803, "param count drift"
    assert rec["arms"]["wide2048"]["init_state_sha256"] != rec["arms"]["compact1024"]["init_state_sha256"], "widths must differ"
    (OC / "card038-inits.json").write_text(json.dumps(rec, indent=2))
    print("wrote", OC / "card038-inits.json"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
