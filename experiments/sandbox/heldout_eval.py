"""Held-out projection quality and the kNN-regressor guard, per sandbox arm.

Owner-directed sandbox (see PLAN2-umap-kernel.md). Two stages:

  truth   GPU, once (~2 min): exact top-10 cosine neighbors in the sealed 2M
          substrate for a 20k subset of R0233's sealed 200k held-out reserve.
  eval    CPU, per arm: project the 20k held-out rows through each arm's
          checkpoint and report
            heldout_ffr   FFR@0.1% of the projected rows against the truth
            regressor_ffr the same rows placed at the mean 2D position of
                          their true neighbors instead (the July guard: if a
                          dumb regressor matches the net, the map is a
                          picture, not a projector)
            tissue        occupied-bin fraction and low-density mass of the
                          arm's full 2M map (registrable "filament" numbers)

Caveat printed with every result: the reserve's per-corpus composition
(25/25/25/25) differs from the training mix (40/25/25/10) — numbers compare
arms against each other, they are not absolute panel values.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

SUBSTRATE = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts"
                 "/minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy")
RESERVE = Path("/data/latent-basemap/runs/round-0233/queue/artifacts"
               "/minilm-mixed-6250k-substrate-and-reserves-v1/reserve.f32.npy")
OUT_ROOT = Path("/data/latent-basemap/sandbox/2m-knobs")
TRUTH_DIR = Path("/data/latent-basemap/sandbox/heldout-truth-2m")
SITE_DIR = Path.home() / ".agent/basemap-maps/sandbox/2m-knobs"

N_PROBES = 20_000
K_TRUE = 10
ROWS = 2_000_000
DISC = int(ROWS * 0.001)


def gpu_is_free() -> bool:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                          "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit("nvidia-smi failed; do not touch a possibly-sick driver")
    return not out.stdout.strip()


def stage_truth() -> None:
    if (TRUTH_DIR / "truth-top10.npy").is_file():
        print("truth already built; delete", TRUTH_DIR, "to rebuild")
        return
    if not gpu_is_free():
        raise SystemExit("GPU busy (round runner has priority); try again later")
    import torch

    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    reserve = np.load(RESERVE, mmap_mode="r")
    probe_rows = np.sort(rng.choice(len(reserve), size=N_PROBES, replace=False))
    probes = np.asarray(reserve[probe_rows], dtype=np.float32)
    q = torch.from_numpy(probes).cuda()
    q = torch.nn.functional.normalize(q, dim=1)

    X = np.load(SUBSTRATE, mmap_mode="r")
    best_val = torch.full((N_PROBES, K_TRUE), -2.0, device="cuda")
    best_idx = torch.zeros((N_PROBES, K_TRUE), dtype=torch.int64, device="cuda")
    chunk = 40_000  # sized for the 32 GB card: sims is (N_PROBES, chunk); exact top-K is chunk-invariant
    for start in range(0, ROWS, chunk):
        block = torch.from_numpy(np.asarray(X[start:start + chunk], dtype=np.float32)).cuda()
        block = torch.nn.functional.normalize(block, dim=1)
        sims = q @ block.T
        merged_val = torch.cat([best_val, sims], dim=1)
        merged_idx = torch.cat([
            best_idx,
            torch.arange(start, start + block.shape[0], device="cuda").expand(N_PROBES, -1),
        ], dim=1)
        top = torch.topk(merged_val, K_TRUE, dim=1)
        best_val = top.values
        best_idx = torch.gather(merged_idx, 1, top.indices)
        del block, sims, merged_val, merged_idx
    np.save(TRUTH_DIR / "probe-rows.npy", probe_rows)
    np.save(TRUTH_DIR / "probes.f32.npy", probes)
    np.save(TRUTH_DIR / "truth-top10.npy", best_idx.cpu().numpy())
    (TRUTH_DIR / "meta.json").write_text(json.dumps({
        "n_probes": N_PROBES, "k_true": K_TRUE, "metric": "cosine, exact",
        "reserve": str(RESERVE), "substrate": str(SUBSTRATE),
        "note": "sandbox truth; reserve composition differs from training mix",
    }, indent=1))
    print(f"truth built: {N_PROBES} probes x top-{K_TRUE} -> {TRUTH_DIR}")


def tissue_metrics(xy: np.ndarray) -> dict:
    from map_renders import robust_extent, binned_counts
    from scipy.spatial import cKDTree

    counts = binned_counts(xy, robust_extent(xy))
    occupied = counts > 0
    peak = counts.max()
    low = occupied & (counts < max(peak * 0.01, 1))
    # Collapse metric (measured 2026-08-12): median 2D radius to the 10th
    # neighbor over the p90 map radius. legacy_lp maps sit near 1e-4 (bead
    # collapse); umap-kernel ~1.1e-3; cuML ~1.7e-3. The kernel is the only
    # knob that moves it — graph choice, dose, and legacy a/b do not.
    rng = np.random.default_rng(0)
    sample = xy[rng.choice(len(xy), min(20_000, len(xy)), replace=False)]
    dists, _ = cKDTree(xy).query(sample, k=11, workers=8)
    radius = np.percentile(np.linalg.norm(xy - xy.mean(axis=0), axis=1), 90)
    return {
        "occupied_bin_fraction": float(occupied.mean()),
        "low_density_mass_fraction": float(counts[low].sum() / counts.sum()),
        "r10_over_map_radius_median": float(np.median(dists[:, 10]) / radius),
    }


def _ffr(tree, placed: np.ndarray, truth: np.ndarray, map_xy_len: int) -> float:
    _, near = tree.query(placed, k=DISC, workers=8)
    hits = sum(np.isin(truth[i], near[i]).sum() for i in range(len(placed)))
    return float(hits / truth.size)


def stage_eval() -> None:
    from scipy.spatial import cKDTree
    import torch
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    torch.set_num_threads(8)
    probes = np.load(TRUTH_DIR / "probes.f32.npy")
    truth = np.load(TRUTH_DIR / "truth-top10.npy")

    results = {}
    for arm_dir in sorted(OUT_ROOT.iterdir()):
        coords_path = arm_dir / "coordinates.npy"
        model_path = arm_dir / "model.pt"
        if not (coords_path.is_file() and model_path.is_file()):
            continue
        xy = np.load(coords_path, mmap_mode="r")
        xy_arr = np.asarray(xy, dtype=np.float32)
        tree = cKDTree(xy_arr)
        model = ParametricUMAP.load(str(model_path), device="cpu")
        placed = model.transform(probes, batch_size=8192).astype(np.float32)
        heldout_ffr = _ffr(tree, placed, truth, ROWS)
        regressor = xy_arr[truth].mean(axis=1)
        regressor_ffr = _ffr(tree, regressor, truth, ROWS)
        results[arm_dir.name] = {
            "heldout_ffr": heldout_ffr,
            "regressor_ffr": regressor_ffr,
            "net_minus_regressor": heldout_ffr - regressor_ffr,
            **tissue_metrics(xy_arr),
        }
        print(f"{arm_dir.name:22s} heldout={heldout_ffr:.4f} "
              f"regressor={regressor_ffr:.4f} "
              f"delta={heldout_ffr - regressor_ffr:+.4f} "
              f"occupied={results[arm_dir.name]['occupied_bin_fraction']:.3f} "
              f"tissue={results[arm_dir.name]['low_density_mass_fraction']:.4f}")

    # cuML reference: tissue only (it has no projector to evaluate).
    cuml_xy = Path("/data/latent-basemap/sandbox/cuml-1m/cuml-xy.npy")
    if cuml_xy.is_file():
        results["cuml-1m-reference"] = {
            "heldout_ffr": None, "regressor_ffr": None, "net_minus_regressor": None,
            **tissue_metrics(np.load(cuml_xy)),
        }

    out = OUT_ROOT / "heldout-eval.json"
    out.write_text(json.dumps({
        "results": results,
        "caveat": "reserve composition 25/25/25/25 vs training 40/25/25/10; "
                  "compare arms to each other, not to panel absolutes",
    }, indent=1))
    print("wrote", out)
    _append_to_page(results)


def _append_to_page(results: dict) -> None:
    import html as html_mod
    page = SITE_DIR / "index.html"
    if not page.is_file():
        return
    rows = "".join(
        f"<tr><td>{html_mod.escape(k)}</td>"
        + "".join(
            f'<td style="text-align:right">'
            + (f"{v[c]:.4f}" if isinstance(v[c], float) else "—") + "</td>"
            for c in ("heldout_ffr", "regressor_ffr", "net_minus_regressor",
                      "occupied_bin_fraction", "low_density_mass_fraction",
                      "r10_over_map_radius_median"))
        + "</tr>"
        for k, v in sorted(results.items()))
    table = ('<h2 id="heldout">Held-out projection + regressor guard</h2>'
             '<p><small>20k sealed reserve rows; truth = exact cosine top-10 in the 2M '
             'substrate; regressor places a row at its true neighbors\' mean 2D position. '
             'Reserve composition differs from the training mix: compare arms, not absolutes.'
             "</small></p>"
             '<table border="0" cellpadding="6" style="border-collapse:collapse">'
             "<tr><th>arm</th><th>heldout ffr</th><th>regressor ffr</th><th>net − reg</th>"
             "<th>occupied bins</th><th>tissue mass</th><th>r10/R (collapse)</th></tr>"
             + rows + "</table>")
    text = page.read_text()
    marker = '<h2 id="heldout">'
    text = text.split(marker)[0] + table
    page.write_text(text)
    print("appended results table to", page)


def main() -> int:
    stages = {"truth": stage_truth, "eval": stage_eval}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        raise SystemExit(f"usage: heldout_eval.py {{{'|'.join(stages)}}}")
    stages[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
