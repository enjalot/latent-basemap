"""Binned density renders for registered maps, generated from sealed artifacts.

Rendering used to happen inside rounds (`semantic-renders/*.png` sealed as
artifacts), so when the slim protocol stopped sealing renders the registry
pages went image-less and the 20k compare sample became the only visual. This
module makes renders a derived view the registry regenerates on its own — like
`maps.json` itself — from each map's sealed full-N coordinates, so every map
gets a visual whether or not its round rendered one.

Per map (cache at ``/data/latent-basemap/render-cache/<map_id>/``):

  density.png        every sealed coordinate row (2M, 6.25M, ... 100M), binned
                     on a 1024x1024 grid, log-scaled counts
  heldout.png        the substrate's sealed held-out reserve projected through
                     the map's own checkpoint (CPU), binned on the same grid --
                     rows the model never trained on
  train-matched.png  a training subsample of the same size as the reserve on
                     the same grid, so heldout vs train is a fair side-by-side

heldout/train-matched exist only for maps whose universe sealed a reserve
(the Phase 2 rung substrates; the 2M family predates reserves). Binning
streams over a read-only memmap in chunks -- never materialize the coordinate
array (the >= 2 GB rule).

Cache is keyed on the coordinates file identity plus RENDER_VERSION; delete a
map's cache dir (or bump the version) to force a re-render.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RENDER_CACHE = Path("/data/latent-basemap/render-cache")
RENDER_VERSION = 1
BINS = 1024
CHUNK_ROWS = 2_000_000
EXTENT_PCT = (0.1, 99.9)
CMAP = "YlGnBu"


def _coords_path(m: dict) -> Path | None:
    f = ((m.get("coordinates") or {}).get("file") or "").removeprefix("gsv:")
    p = Path(f)
    return p if f and p.is_file() else None


def _load_coords(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{path}: expected (N, 2), got {arr.shape}")
    return arr


def robust_extent(coords: np.ndarray, sample_rows: int = 2_000_000) -> list[float]:
    """Percentile extent so one outlier island cannot shrink the whole map."""
    n = len(coords)
    if n > sample_rows:
        idx = np.linspace(0, n - 1, sample_rows).astype(np.int64)
        pts = np.asarray(coords[idx], dtype=np.float32)
    else:
        pts = np.asarray(coords, dtype=np.float32)
    lo, hi = EXTENT_PCT
    x0, x1 = np.percentile(pts[:, 0], [lo, hi])
    y0, y1 = np.percentile(pts[:, 1], [lo, hi])
    pad_x = 0.02 * (x1 - x0) or 1.0
    pad_y = 0.02 * (y1 - y0) or 1.0
    return [float(x0 - pad_x), float(x1 + pad_x), float(y0 - pad_y), float(y1 + pad_y)]


def binned_counts(coords: np.ndarray, extent: list[float], bins: int = BINS) -> np.ndarray:
    x0, x1, y0, y1 = extent
    edges_x = np.linspace(x0, x1, bins + 1)
    edges_y = np.linspace(y0, y1, bins + 1)
    counts = np.zeros((bins, bins), dtype=np.int64)
    for i in range(0, len(coords), CHUNK_ROWS):
        chunk = np.asarray(coords[i:i + CHUNK_ROWS], dtype=np.float32)
        # Clip so off-extent outliers accumulate in the edge bins instead of
        # silently vanishing from the census.
        cx = np.clip(chunk[:, 0], x0, x1)
        cy = np.clip(chunk[:, 1], y0, y1)
        h, _, _ = np.histogram2d(cx, cy, bins=[edges_x, edges_y])
        counts += h.astype(np.int64)
    return counts


def render_png(counts: np.ndarray, out_path: Path) -> None:
    """log-scaled counts through a sequential colormap; empty bins stay white."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    from matplotlib.image import imsave

    logc = np.log1p(counts.astype(np.float64))
    peak = logc.max() or 1.0
    rgba = colormaps[CMAP](logc / peak)
    rgba[counts == 0] = [1.0, 1.0, 1.0, 1.0]
    # histogram2d puts x on axis 0; images want x horizontal, y up.
    imsave(out_path, np.rot90(rgba.transpose(1, 0, 2), 2)[::-1])


def find_reserve(m: dict) -> Path | None:
    """The sealed held-out reserve of the map's own training universe."""
    model_dir = Path(((m.get("model") or {}).get("path") or "").removeprefix("gsv:")).parent
    for cfg_path in sorted(model_dir.glob("*production-config.json")):
        cfg = json.loads(cfg_path.read_text())
        substrate = (cfg.get("config", {}).get("input", {}) or {}).get("substrate_path")
        if substrate:
            reserve = Path(substrate).parent / "reserve.f32.npy"
            if reserve.is_file():
                return reserve
    return None


def project_reserve(m: dict, reserve: Path) -> np.ndarray | None:
    model_path = Path(((m.get("model") or {}).get("path") or "").removeprefix("gsv:"))
    if not model_path.is_file():
        return None
    import torch
    torch.set_num_threads(8)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    model = ParametricUMAP.load(str(model_path), device="cpu")
    vectors = np.load(reserve, mmap_mode="r")
    return model.transform(vectors, batch_size=8192)


def _cache_key(coords: Path) -> dict:
    st = coords.stat()
    return {"version": RENDER_VERSION, "coords": str(coords),
            "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "bins": BINS}


def build_map(m: dict, *, force: bool = False) -> str:
    coords_path = _coords_path(m)
    if coords_path is None:
        return "no-coordinates"
    out_dir = RENDER_CACHE / m["map_id"]
    meta_path = out_dir / "meta.json"
    key = _cache_key(coords_path)
    if not force and meta_path.is_file():
        try:
            if json.loads(meta_path.read_text()).get("key") == key:
                return "cached"
        except (json.JSONDecodeError, OSError):
            pass
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = _load_coords(coords_path)
    extent = robust_extent(coords)
    counts = binned_counts(coords, extent)
    render_png(counts, out_dir / "density.png")
    meta = {
        "key": key, "extent": extent, "n_rows": int(len(coords)),
        "renders": {"density": {"rows": int(len(coords))}},
    }

    reserve = find_reserve(m)
    if reserve is not None:
        heldout_xy = project_reserve(m, reserve)
        if heldout_xy is not None:
            n_res = len(heldout_xy)
            render_png(binned_counts(heldout_xy, extent), out_dir / "heldout.png")
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(len(coords), size=min(n_res, len(coords)),
                                     replace=False))
            train_xy = np.asarray(coords[idx], dtype=np.float32)
            render_png(binned_counts(train_xy, extent), out_dir / "train-matched.png")
            meta["renders"]["heldout"] = {"rows": n_res, "reserve": str(reserve)}
            meta["renders"]["train_matched"] = {"rows": int(len(idx))}

    meta_path.write_text(json.dumps(meta, indent=1))
    return "rendered"


def build(registry: dict, *, force: bool = False, only: str | None = None) -> None:
    outcomes: dict[str, int] = {}
    for m in registry["maps"]:
        if m.get("kind") != "round-map":
            continue
        if only and only not in m["map_id"]:
            continue
        try:
            outcome = build_map(m, force=force)
        except Exception as exc:  # one bad map must not stop the sweep
            outcome = "error"
            print(f"  {m['map_id']}: ERROR {exc}", file=sys.stderr)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    print("map_renders:", ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", help="substring filter on map_id")
    ap.add_argument("--registry", default="/data/latent-basemap/maps.json")
    args = ap.parse_args()
    registry = json.loads(Path(args.registry).read_text())
    build(registry, force=args.force, only=args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
