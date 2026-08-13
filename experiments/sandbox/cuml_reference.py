"""cuML (RAPIDS out-of-core UMAP, lit 0197) benchmark maps on the sealed rung substrates.

Owner-directed sandbox (2026-08-13): run the non-parametric reference method
on the EXACT substrates our scale-ladder maps train on, so every rung has a
"what would their method draw" card scored with our own instruments
(quick-FFR against the sealed exact edges, collapse r10·√N, fog) and aligned
into the same frame on the kernels page.

  .venv/bin/python experiments/sandbox/cuml_reference.py umap --rung 2m
  .venv/bin/python experiments/sandbox/cuml_reference.py umap --rung 6250k
  .venv/bin/python experiments/sandbox/cuml_reference.py eval --rung 2m

The `umap` stage re-execs itself under /data/latent-basemap/cuml_py.
Settings are the method's own defaults (n_neighbors=15 to match our graphs,
min_dist=0.1, random_state=0) — this is a benchmark of THEIR method, not a
tuned competitor; stated on the page. cuML has no projector, so held-out
FFR does not apply (no graph/index is kept either — that is the product
difference, not an omission).

Rung 12500k is configured for when the promotion plan's P1 drift cell runs;
note 12.5M x 384 fp32 is ~19.2 GB on a 31.4 GiB card — expect to need the
paper's batched NN-descent (nnd) parameters; measure, don't assume.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO / "experiments"), str(REPO / "experiments/sandbox")):
    if p not in sys.path:
        sys.path.insert(0, p)

CUML_PY = Path("/data/latent-basemap/cuml_py")
OUT_ROOT = Path("/data/latent-basemap/sandbox/cuml-ref")

SUBSTRATES = {
    "2m": Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts"
               "/minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy"),
    "6250k": Path("/data/latent-basemap/runs/round-0233/queue/artifacts"
                  "/minilm-mixed-6250k-substrate-and-reserves-v1/substrate.f32.npy"),
    "12500k": Path("/data/latent-basemap/runs/round-0235/queue/artifacts"
                   "/minilm-mixed-12500k-nested-substrate-and-reserves-v1/substrate.f32.npy"),
}


def gpu_is_free() -> bool:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                          "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit("nvidia-smi failed; do not touch a possibly-sick driver")
    return not out.stdout.strip()


def stage_umap(rung: str) -> None:
    try:
        import cuml  # noqa: F401
    except ImportError:
        if not gpu_is_free():
            raise SystemExit("GPU busy (round runner has priority); try again later")
        print("re-exec under cuml env ...")
        raise SystemExit(subprocess.run(
            [str(CUML_PY), __file__, "umap", "--rung", rung]).returncode)

    import time
    from cuml.manifold import UMAP

    out = OUT_ROOT / rung
    out.mkdir(parents=True, exist_ok=True)
    if (out / "coordinates.npy").is_file():
        raise SystemExit(f"{out}/coordinates.npy exists; write-once (delete to re-run)")
    substrate = SUBSTRATES[rung]
    X = np.asarray(np.load(substrate, mmap_mode="r"), dtype=np.float32)
    t0 = time.time()
    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                   random_state=0, verbose=True)
    xy = np.asarray(reducer.fit_transform(X), dtype=np.float32)
    wall = time.time() - t0
    np.save(out / "coordinates.npy", xy)
    (out / "summary.json").write_text(json.dumps({
        "arm": f"cuml-reference-{rung}", "rung": rung, "rows": int(len(xy)),
        "method": "RAPIDS cuML UMAP (lit 0197 method family)",
        "settings": {"n_neighbors": 15, "min_dist": 0.1, "random_state": 0},
        "substrate": str(substrate), "wall_s": wall,
        "note": "non-parametric benchmark on the sealed rung substrate; "
                "no projector, no held-out FFR",
    }, indent=1))
    print(f"cuML {rung}: {len(xy):,} rows in {wall:.0f}s -> {out}")


def stage_eval(rung: str) -> None:
    from knobs_2m import quick_ffr, RUNGS, resolve_edges
    sys.path.insert(0, str(REPO / "experiments/metrics"))
    from collapse_fog import map_collapse, map_fog

    out = OUT_ROOT / rung
    xy = np.asarray(np.load(out / "coordinates.npy", mmap_mode="r"), dtype=np.float32)
    summary = json.loads((out / "summary.json").read_text())
    rung_cfg = RUNGS.get(rung)
    if rung_cfg is not None:
        edges = resolve_edges(rung_cfg)
        summary["quick_ffr_at_0.1pct"] = quick_ffr(xy, edges, rung_cfg["rows"])
    collapse = map_collapse(xy)
    fog = map_fog(xy)
    summary["r10_over_map_radius_median"] = collapse["r10_over_map_radius_median"]
    summary["collapse_r10_sqrtN"] = collapse["r10_over_radius_times_sqrt_n"]
    summary["low_density_mass_fraction"] = fog["low_density_mass_fraction"]
    summary["fog_detail"] = {k: fog[k] for k in
                             ("fog", "degenerate", "resolution_levels",
                              "occupied_bin_fraction", "peak_bin_count")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary.get(k) for k in
                      ("arm", "quick_ffr_at_0.1pct", "collapse_r10_sqrtN")},
                     indent=1, default=float))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("umap", "eval"))
    ap.add_argument("--rung", choices=sorted(SUBSTRATES), required=True)
    args = ap.parse_args()
    if args.stage == "umap":
        stage_umap(args.rung)
    else:
        stage_eval(args.rung)
    return 0


if __name__ == "__main__":
    sys.exit(main())
