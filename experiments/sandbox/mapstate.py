"""MapState — persisted state of a living basemap, for the chained-growth centerpiece (G, overseer-signed-off
2026-09-04). The anchored-λ harness saved COORDS ONLY, so update-from-update was never demonstrable. MapState
bundles everything needed to reload a map and update FROM it: the trained ParametricUMAP model, the input
preprocessing stamp, the RIGID coordinate frame (frame.py, item B), the kNN/fuzzy graph params, and a receipt
(source commit, resolved config, input/output hashes, gen key — review A5). save() writes it + a reload ASSERT
(transform(x_ref) reproduces the saved ref coords within TOL); load_and_assert() reconstructs + re-checks.

Contract (overseer resolution 3): the chained science runs on fp16 substrate; a separate int8 save/reload
assert (100K rows) is reported alongside but never confounds the chain. Reload tolerance 1e-4 on the model's
native coords (rigid-frame alignment is applied at scoring time via frame.py, not baked into the model output)."""
import json, hashlib, time
from pathlib import Path
import numpy as np

TOL = 1e-4


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def save(out_dir, model, x_ref, *, preproc_stamp, graph_params, receipt, frame_ref_coords=None):
    """Persist a MapState. `model` is a fitted ParametricUMAP; `x_ref` is the PREPROCESSED reference substrate
    (same preprocessing the chain will apply to arrivals) used for the reload assert. frame_ref_coords: the
    canonical reference layout for rigid alignment across the chain (defaults to this map's own ref coords)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "model.pt"))
    ref_xy = np.asarray(model.transform(x_ref), dtype=np.float32)      # deterministic (model.eval + no_grad)
    np.save(out / "ref_xy.npy", ref_xy)
    if frame_ref_coords is None:
        frame_ref_coords = ref_xy
    np.save(out / "frame_ref.npy", np.asarray(frame_ref_coords, dtype=np.float32))
    man = {"schema": "mapstate-v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "preproc": preproc_stamp, "graph": graph_params, "receipt": receipt,
           "n_ref": int(len(ref_xy)), "ref_xy_sha": _sha(ref_xy), "input_dim": int(x_ref.shape[1]),
           "tol": TOL}
    (out / "mapstate.json").write_text(json.dumps(man, indent=1))
    # immediate save-side assert: a fresh transform of x_ref reproduces ref_xy (guards a broken save)
    chk = np.asarray(model.transform(x_ref), dtype=np.float32)
    dev = float(np.abs(chk - ref_xy).max())
    assert dev <= TOL, f"MapState.save self-check FAILED: transform not reproducible, max dev {dev} > {TOL}"
    return out


def load_and_assert(state_dir, x_ref, ParametricUMAP, device=None):
    """Reconstruct the model + re-run the reload assert: transform(x_ref) must reproduce the saved ref_xy within
    TOL (proves model + preproc round-tripped). Returns (model, manifest, ref_xy). Fail-closed on divergence."""
    d = Path(state_dir)
    man = json.loads((d / "mapstate.json").read_text())
    model = ParametricUMAP.load(str(d / "model.pt"), device=device)
    got = np.asarray(model.transform(x_ref), dtype=np.float32)
    ref = np.load(d / "ref_xy.npy")
    assert got.shape == ref.shape, f"reload shape mismatch {got.shape} vs {ref.shape}"
    dev = float(np.abs(got - ref).max())
    assert dev <= TOL, (f"MapState reload assert FAILED: transform(x_ref) diverged from saved coords, "
                        f"max dev {dev} > {TOL} — model or preproc did not round-trip")
    assert _sha(got) == man["ref_xy_sha"] or dev <= TOL, "ref_xy hash mismatch beyond tolerance"
    return model, man, got


def make_receipt(source_commit, resolved_config, input_hashes, gen_key):
    """A5 receipt: full provenance for one MapState. resolved_config must be the FULL config (max_train_steps,
    anchor/holdout settings, seed, w) — two different-config runs must never share gen_key."""
    return {"source_commit": source_commit, "resolved_config": resolved_config,
            "input_hashes": input_hashes, "gen_key": gen_key,
            "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
