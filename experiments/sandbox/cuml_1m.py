"""1M cuML UMAP reference beside our parametric maps, on identical rows.

Owner-directed sandbox (see PLAN.md). Three stages so the GPU stage is a
single short command run under the cuml env:

  sample   CPU  draw 1M uniform rows (seed 0) from the sealed 2M substrate,
                copy their embeddings out chunked (never materialize the 2M)
  umap     GPU  cuML UMAP (n_neighbors=15, min_dist=0.1) on the 1M — runs
                itself under /data/latent-basemap/cuml_py
  page     CPU  binned renders of cuML vs our exact-graph and cuVS-graph
                parametric maps restricted to the same rows, one HTML page

cuML is the non-parametric reference: it may legitimately tear inter-cluster
tissue apart because nothing forces it to be a continuous function of the
embedding. If its picture shows the same filaments, they are in the data/graph;
if not, they are the parametric price (or a recipe knob — see knobs_2m.py).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))

SUBSTRATE = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts"
                 "/minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy")
REGISTRY = Path("/data/latent-basemap/maps.json")
EXACT_MAP_ID = "round-0217-minilm-mixed-2m-map-seed42-low-dose-v1"
CUVS_MAP_ID = "round-0223-minilm-mixed-2m-cuvs-igd48-map-seed42-low-dose-v1"
CUML_PY = Path("/data/latent-basemap/cuml_py")
OUT = Path("/data/latent-basemap/sandbox/cuml-1m")
SITE_DIR = Path.home() / ".agent/basemap-maps/sandbox/cuml-1m"
N_SAMPLE = 1_000_000
SEED = 0


def _registry_coords(map_id: str) -> Path:
    """The sealed coordinates file the registry indexed for this map."""
    registry = json.loads(REGISTRY.read_text())
    for m in registry["maps"]:
        if m.get("map_id") == map_id:
            p = Path(((m.get("coordinates") or {}).get("file") or "").removeprefix("gsv:"))
            if p.is_file():
                return p
            raise SystemExit(f"{map_id}: coordinates missing on disk: {p}")
    raise SystemExit(f"{map_id}: not in {REGISTRY}")


def gpu_is_free() -> bool:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                          "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit("nvidia-smi failed; do not touch a possibly-sick driver")
    return not out.stdout.strip()


def stage_sample() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(SUBSTRATE, mmap_mode="r")
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(len(X), size=N_SAMPLE, replace=False))
    np.save(OUT / "rows.npy", rows)
    emb = np.lib.format.open_memmap(OUT / "emb.f32.npy", mode="w+",
                                    dtype=np.float32, shape=(N_SAMPLE, 384))
    step = 200_000
    for i in range(0, N_SAMPLE, step):
        emb[i:i + step] = X[rows[i:i + step]]
    emb.flush()
    print(f"sampled {N_SAMPLE:,} rows -> {OUT}/emb.f32.npy")


def stage_umap() -> None:
    """Re-exec under the cuml env if needed, then fit cuML UMAP."""
    try:
        import cuml  # noqa: F401
    except ImportError:
        if not CUML_PY.exists():
            raise SystemExit(f"cuml launcher missing: {CUML_PY}")
        if not gpu_is_free():
            raise SystemExit("GPU busy (round runner has priority); try again later")
        print("re-exec under cuml env ...")
        raise SystemExit(subprocess.run(
            [str(CUML_PY), __file__, "umap"]).returncode)

    import time
    from cuml.manifold import UMAP

    emb = np.load(OUT / "emb.f32.npy", mmap_mode="r")
    t0 = time.time()
    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                   random_state=SEED, verbose=True)
    xy = np.asarray(reducer.fit_transform(np.asarray(emb)), dtype=np.float32)
    wall = time.time() - t0
    np.save(OUT / "cuml-xy.npy", xy)
    (OUT / "cuml-summary.json").write_text(json.dumps({
        "rows": N_SAMPLE, "n_neighbors": 15, "min_dist": 0.1,
        "wall_s": wall, "note": "sandbox artifact; non-parametric reference",
    }, indent=1))
    print(f"cuML UMAP on {N_SAMPLE:,} rows in {wall:.1f}s -> {OUT}/cuml-xy.npy")


def stage_page() -> None:
    import html as html_mod
    from map_renders import robust_extent, binned_counts, render_png

    rows = np.load(OUT / "rows.npy")
    panels = [("cuML UMAP (non-parametric, trained on these 1M rows)",
               np.load(OUT / "cuml-xy.npy")),
              ("parametric · exact graph (R0217 seed 42, 2M-trained, viewed on the rows)",
               np.asarray(np.load(_registry_coords(EXACT_MAP_ID), mmap_mode="r")[rows])),
              ("parametric · cuVS graph (R0223 seed 42, 2M-trained, viewed on the rows)",
               np.asarray(np.load(_registry_coords(CUVS_MAP_ID), mmap_mode="r")[rows]))]
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, (title, xy) in enumerate(panels):
        extent = robust_extent(xy)
        name = f"panel-{i}.png"
        render_png(binned_counts(xy, extent), SITE_DIR / name)
        cards.append((title, name))
    figs = "".join(
        f'<figure style="margin:0"><img src="{f}" style="width:100%;border:1px solid #ddd;border-radius:6px">'
        f"<figcaption><small>{html_mod.escape(t)}</small></figcaption></figure>"
        for t, f in cards)
    (SITE_DIR / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>cuML 1M reference</title>"
        '<body style="font-family:system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem">'
        "<h1>1M rows, three layouts (sandbox)</h1>"
        "<p>Identical 1M-row uniform sample of the sealed 2M substrate, binned "
        "identically. cuML is free to tear tissue apart; the parametric maps "
        "must stay continuous functions of the embedding. Caveat: cuML trains "
        "on the subset, the parametric maps trained on the full 2M and are "
        "viewed on the subset.</p>"
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:16px">{figs}</div>')
    print(f"page: {SITE_DIR}/index.html  (http://gsv.local:8800/basemap-maps/sandbox/cuml-1m/)")


def main() -> int:
    stages = {"sample": stage_sample, "umap": stage_umap, "page": stage_page}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        raise SystemExit(f"usage: cuml_1m.py {{{'|'.join(stages)}}}")
    stages[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
