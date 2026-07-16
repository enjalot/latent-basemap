"""Deterministic fixed-sample/fixed-axis basemap comparison renderer.

The input is a JSON spec with ``comparisons``.  Each comparison declares a
substrate and a list of maps (label + coords path).  Every map on a substrate is
indexed by the same saved sample-ID set; axes are the union extent across the
comparison and coordinates are never normalized per map.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.artifact_identity import (canonical_json, git_checkout_state, path_signature,
                                       sha256_bytes)
from basemap.panel_v2 import load_coords


def _ids_hash(values) -> str:
    values = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return sha256_bytes(canonical_json({"length": len(values), "dtype": "int64"}) +
                        values.tobytes())


def _diagnostics(points: np.ndarray, bins: int = 64) -> dict:
    finite = np.isfinite(points).all(axis=1)
    pts = points[finite]
    if len(pts) == 0:
        return {"finite_fraction": 0.0, "collapsed": True, "archipelago_components": None}
    std = pts.std(axis=0)
    span = pts.max(axis=0) - pts.min(axis=0)
    collapsed = bool(np.any(std <= 1e-8) or np.any(span <= 1e-7))
    hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins)
    occupied = hist > 0
    seen = np.zeros_like(occupied, dtype=bool)
    sizes = []
    for i, j in zip(*np.nonzero(occupied)):
        if seen[i, j]:
            continue
        queue = deque([(int(i), int(j))]); seen[i, j] = True; size = 0
        while queue:
            x, y = queue.popleft(); size += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < bins and 0 <= ny < bins and occupied[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True; queue.append((nx, ny))
        sizes.append(size)
    occupied_count = int(occupied.sum())
    largest = max(sizes, default=0)
    return {
        "finite_fraction": round(float(finite.mean()), 8),
        "axis_std": [float(v) for v in std],
        "axis_span": [float(v) for v in span],
        "collapsed": collapsed,
        "occupied_grid_cells": occupied_count,
        "archipelago_components": len(sizes),
        "largest_component_fraction": (round(largest / occupied_count, 6)
                                       if occupied_count else None),
    }


def render(spec: dict, *, spec_path: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.realpath(spec["output_dir"])
    if not out_dir.startswith("/data/"):
        raise ValueError("render output_dir must be under /data")
    os.makedirs(out_dir, exist_ok=True)
    seed = int(spec.get("sample_seed", 20260716))
    sample_size = int(spec.get("sample_size", 50000))
    substrates = {}
    entries = []

    for comparison in spec.get("comparisons", []):
        maps = comparison.get("maps") or []
        if not maps:
            raise ValueError(f"comparison {comparison.get('id')} has no maps")
        loaded = []
        for item in maps:
            coords, coord_ids = load_coords(item["coords"])
            if coords.shape[1] < 2:
                raise ValueError(f"map {item['label']} has fewer than two coordinate dimensions")
            loaded.append((item, coords, coord_ids))
        lengths = {len(coords) for _, coords, _ in loaded}
        if len(lengths) != 1:
            raise ValueError(f"comparison {comparison['id']} map lengths differ: {sorted(lengths)}")
        n = lengths.pop()
        substrate = comparison["substrate"]
        if substrate in substrates:
            ids = substrates[substrate]
            if len(ids) and int(ids.max()) >= n:
                raise ValueError(f"substrate {substrate} sample IDs do not fit comparison {comparison['id']}")
        else:
            rng = np.random.RandomState(seed)
            ids = np.sort(rng.choice(n, min(sample_size, n), replace=False)).astype(np.int64)
            substrates[substrate] = ids
            np.save(os.path.join(out_dir, f"sample_ids_{substrate}.npy"), ids)

        sampled = [np.asarray(coords[ids, :2], dtype=np.float32) for _, coords, _ in loaded]
        union = np.concatenate(sampled, axis=0)
        if not np.isfinite(union).all():
            raise ValueError(f"comparison {comparison['id']} contains non-finite sampled coordinates")
        lo, hi = union.min(axis=0), union.max(axis=0)
        span = np.maximum(hi - lo, 1e-8)
        pad = span * float(spec.get("axis_padding_fraction", 0.02))
        xlim = [float(lo[0] - pad[0]), float(hi[0] + pad[0])]
        ylim = [float(lo[1] - pad[1]), float(hi[1] + pad[1])]

        ncols = min(int(spec.get("max_columns", 3)), len(maps))
        nrows = int(math.ceil(len(maps) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        map_entries = []
        flat_axes = axes.reshape(-1)
        for ax, (item, coords, coord_ids), points in zip(flat_axes, loaded, sampled):
            ax.scatter(points[:, 0], points[:, 1], s=0.15, alpha=0.35,
                       linewidths=0, rasterized=True)
            ax.set_title(item["label"])
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
            sources = [path_signature(item["coords"])]
            for source in item.get("sources", []):
                sources.append(path_signature(source))
            map_entries.append({
                "label": item["label"],
                "coordinate_dimensions_rendered": [0, 1],
                "coordinate_signature": sources[0],
                "source_signatures": sources[1:],
                "coordinate_ids_hash": (_ids_hash(coord_ids) if coord_ids is not None else None),
                "diagnostics": _diagnostics(points),
            })
        for ax in flat_axes[len(maps):]:
            ax.axis("off")
        fig.suptitle(comparison.get("title", comparison["id"]))
        fig.tight_layout()
        image_path = os.path.join(out_dir, f"{comparison['id']}.png")
        fig.savefig(image_path, dpi=int(spec.get("dpi", 180)), bbox_inches="tight")
        plt.close(fig)
        entries.append({
            "comparison_id": comparison["id"],
            "substrate": substrate,
            "sample_ids_path": os.path.join(out_dir, f"sample_ids_{substrate}.npy"),
            "sample_ids_sha256": _ids_hash(ids),
            "sample_count": len(ids),
            "axis_policy": "shared union extent; no per-map normalization",
            "xlim": xlim,
            "ylim": ylim,
            "image": path_signature(image_path),
            "maps": map_entries,
        })

    return {
        "schema": "fixed_comparison_render_manifest.v1",
        "spec": path_signature(spec_path),
        "renderer": path_signature(__file__),
        "checkout": git_checkout_state(os.path.dirname(os.path.dirname(__file__))),
        "sample_seed": seed,
        "sample_size_requested": sample_size,
        "comparisons": entries,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--manifest")
    args = parser.parse_args(argv)
    with open(args.spec, encoding="utf-8") as handle:
        spec = json.load(handle)
    manifest = render(spec, spec_path=args.spec)
    out = args.manifest or os.path.join(spec["output_dir"], "render-manifest.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
