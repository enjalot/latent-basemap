"""Card015 shared 3D initialization. Builds a 3-output ParametricUMAP head (champion arch) and constructs
the shared init by COPYING the card010/013 2D fresh init (589895f037d406ae) hidden weights + first-2 output
rows/biases, retaining ONE independently-seeded 3rd output row/bias. Asserts every shared tensor + the first
two output rows match the 2D init exactly; hashes + saves the common 3D init. If the arch cannot support the
exact construction, writes the concrete reason (no silent fallback). CPU. Usage: build_card015_init.py
"""
import os, sys, json, hashlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()

SB = Path("/data/latent-basemap/sandbox")
CHAMPION = SB / "dino-arrival-t0/champion-bs16k/model.pt"
INIT2D = Path("/data/latent-basemap/substrates/card010-adaptive/init-card010.pt")
OUT = SB / "card015-init"; OUT.mkdir(parents=True, exist_ok=True); OC = SB / "overseer-codex"
SEED = 42


def _state_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def main():
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch
    init2d = torch.load(str(INIT2D), map_location="cpu", weights_only=False)
    sd2 = init2d["model_state"]; assert init2d["init_state_sha256"] == "589895f037d406ae", "2D init sha mismatch"
    # build a fresh 3D head (champion config, n_components=3, seed 42)
    p = ParametricUMAP.load(str(CHAMPION), device="cpu"); p.model = None; p.n_components = 3
    torch.manual_seed(SEED); np.random.seed(SEED); p._init_model(1536)
    sd3_fresh = {k: v.detach().clone() for k, v in p.model.state_dict().items()}

    # verify the arch supports the exact construction: shared tensors identical shape; proj_out differs only
    # in output rows (2 -> 3). Identify the output layer (proj_out).
    reason = None
    out_w, out_b = "proj_out.weight", "proj_out.bias"
    if out_w not in sd2 or out_w not in sd3_fresh:
        reason = f"no proj_out.weight in state dicts (keys2d={list(sd2)[:6]})"
    elif sd2[out_w].shape[0] != 2 or sd3_fresh[out_w].shape[0] != 3 or sd2[out_w].shape[1] != sd3_fresh[out_w].shape[1]:
        reason = f"proj_out shape mismatch: 2D {tuple(sd2[out_w].shape)} vs 3D {tuple(sd3_fresh[out_w].shape)}"
    else:
        for k in sd2:
            if k in (out_w, out_b): continue
            if k not in sd3_fresh or sd2[k].shape != sd3_fresh[k].shape:
                reason = f"shared tensor {k} missing/shape-mismatch in 3D"; break
    if reason is not None:
        (OC / "card015-init.json").write_text(json.dumps({"schema": "card015-init-2026-09-12", "VIABLE": False,
            "reason": reason, "prospective_recipe": "Instantiate n_components=3 head, copy 2D hidden + proj_out[:2]; if proj_out is fused/named differently, map by output-dim slice; refuse until construction is exact."}, indent=1))
        print("CONSTRUCTION NOT SUPPORTED:", reason); return 3

    # construct shared 3D init: 2D hidden + 2D proj_out[:2]; fresh 3rd output row/bias
    sd3 = {}
    for k in sd3_fresh:
        if k == out_w:
            w = sd2[out_w].clone(); w = torch.cat([w, sd3_fresh[out_w][2:3]], dim=0); sd3[k] = w   # rows 0,1 from 2D; row2 fresh
        elif k == out_b:
            b = sd2[out_b].clone(); b = torch.cat([b, sd3_fresh[out_b][2:3]], dim=0); sd3[k] = b
        else:
            sd3[k] = sd2[k].clone()   # shared hidden tensors from the 2D init exactly

    # assert shared tensors + first-two output rows == 2D init exactly
    A = {}
    A["shared_hidden_exact"] = all(bool(torch.equal(sd3[k], sd2[k])) for k in sd2 if k not in (out_w, out_b))
    A["out_first2_rows_exact"] = bool(torch.equal(sd3[out_w][:2], sd2[out_w]) and torch.equal(sd3[out_b][:2], sd2[out_b]))
    A["out_third_row_is_fresh"] = bool(torch.equal(sd3[out_w][2:3], sd3_fresh[out_w][2:3]) and torch.equal(sd3[out_b][2:3], sd3_fresh[out_b][2:3]))
    A["out_shape_3"] = bool(sd3[out_w].shape[0] == 3 and sd3[out_b].shape[0] == 3)
    ok = all(A.values())
    if ok:
        p.model.load_state_dict(sd3); init3d_sha = _state_sha(sd3)
        torch.save({"model_state": sd3, "init_state_sha256": init3d_sha, "n_components": 3,
                    "derived_from_2d_sha": "589895f037d406ae"}, OUT / "init-card015-3d.pt")
    result = {"schema": "card015-init-2026-09-12", "VIABLE": bool(ok), "asserts": A,
              "init3d_sha256": (init3d_sha if ok else None), "derived_from_2d": "589895f037d406ae",
              "construction": "champion arch n_components=3; hidden + proj_out[:2] copied from 2D init; proj_out[2] fresh (seed42)"}
    (OC / "card015-init.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
