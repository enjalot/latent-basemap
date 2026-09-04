"""Text-chain stage helpers (owner plan-basemap-chain, 2026-09-04). CPU-testable pieces of the orchestrator:
transform an arrival tranche through a persisted MapState, and build the anchor (previous live layout) for an
anchored update. The GPU steps (cumulative graph build, anchored update) are shelled to image_map_pipeline /
p_evolbench_lambda under the flock by the driver. Fixed-T0-frame scoring lives in text_chain_score.py."""
import json
from pathlib import Path
import numpy as np

SUB = Path("/data/latent-basemap/substrates")


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def load_tranches(paths):
    """Ordered L2-normed concat of substrate paths (matches image_map_pipeline._norm_concat + the fit loader)."""
    return _norm(np.concatenate([np.asarray(np.load(p, mmap_mode="r"), dtype=np.float32) for p in paths]))


def transform(mapstate_dir, tranche_paths, ParametricUMAP, device="cuda", batch_size=8192):
    """Transform the (normed) arrival tranches through the persisted MapState's model -> 2D coords. Used for the
    Ordinary arrival stages (by the current MapState) AND the frozen control (MapState_0 over the whole timeline)."""
    import mapstate  # noqa: F401 (kept for symmetry; model loaded directly here)
    model = ParametricUMAP.load(str(Path(mapstate_dir) / "model.pt"), device=device)
    X = load_tranches(tranche_paths)
    return np.asarray(model.transform(X, batch_size=batch_size), dtype=np.float32)


def build_anchor(prior_layout_coords, out_npz):
    """Anchor = pin every previously-visible cumulative row to its immediately-preceding live layout position.
    prior_layout_coords: (n_visible, 2) coords of the rows visible BEFORE this update (in the prior MapState's
    frame). anchor_ids = 0..n_visible-1 (the update's substrate is ordered [visible..., new...]), targets = those
    coords. The new (OOD) rows beyond n_visible are unanchored."""
    c = np.asarray(prior_layout_coords, dtype=np.float32)
    np.savez(out_npz, anchor_ids=np.arange(c.shape[0], dtype=np.int64), anchor_targets=c)
    return c.shape[0]


def stage_row_ranges(manifest_path):
    """From the resolved chain manifest, the [lo,hi) row range of each tranche within each stage's ordered concat
    (for cohort scoring). Returns {stage: {tranche: (lo,hi)}} using the ordered tranche list + per-tranche rows."""
    m = json.loads(Path(manifest_path).read_text())
    rows = {t: m["tranches"][t]["rows"] for t in m["tranches"]}
    out = {}
    for s, sd in m["stages"].items():
        r = {}; off = 0
        for t in sd["tranches_ordered"]:
            r[t] = (off, off + rows[t]); off += rows[t]
        out[s] = r
    return out
