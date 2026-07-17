"""Deterministic fixed-sample/fixed-axis basemap comparison renderer.

The input is a JSON spec with ``comparisons``.  Each comparison declares a
substrate and a list of maps (label + coords path).  Every map on a substrate is
indexed by the same saved sample-ID set; axes are the union extent across the
comparison and coordinates are never normalized per map.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.artifact_identity import (canonical_json, git_checkout_state, path_signature,
                                       expected_input_signature, ordered_array_sha256,
                                       sha256_bytes)
from basemap.output_safety import (atomic_build_new_file, atomic_save_new_npy,
                                   atomic_write_new_json, create_fresh_directory,
                                   ensure_data_directory, refuse_existing)
from basemap.panel_v2 import load_coords
from basemap.round0005_staging import validate_semantic_namespace


ROUND0005_RENDER_SPEC_SCHEMA = "round0005_fixed_comparison_render_spec.v1"
ROUND0005_RENDER_MANIFEST_SCHEMA = "round0005_fixed_comparison_render_manifest.v3"
ROUND0005_HISTORICAL_MANIFEST = \
    "/data/latent-basemap/runs/round-0001/renders/render-manifest.json"
ROUND0005_FIXED_COMPARISONS = (
    {
        "id": "g1_fixed_pair",
        "title": "G1 accepted 8M pair — fixed sample and axes",
        "substrate": "jina-8m-mixed",
        "maps": (
            ("legacy_lp s42",
             "/data/latent-basemap/closure/historical/r1_8m_bridge_weighted_s42_20260715_163719_f4934f5b"),
            ("umap std-curve s42", "/data/latent-basemap/closure/g1/train_stdcurve_s42"),
        ),
    },
    {
        "id": "o1_fixed_six_map",
        "title": "O1 diagnostic raw/prompted maps — fixed sample and axes",
        "substrate": "jina-200k-o1-row-universe",
        "maps": (
            ("raw s42", "/data/latent-basemap/closure/historical/r1_kernel_legacy_a1b1_s42_20260714_041129_d8943c8d"),
            ("prompted s42", "/data/latent-basemap/closure/o1/train_prompted_s42"),
            ("raw s43", "/data/latent-basemap/closure/historical/r1_kernel_legacy_a1b1_s43_20260714_050821_983a311d"),
            ("prompted s43", "/data/latent-basemap/closure/o1/train_prompted_s43"),
            ("raw s44", "/data/latent-basemap/closure/historical/r1_kernel_legacy_a1b1_s44_20260714_060516_7be55b53"),
            ("prompted s44", "/data/latent-basemap/closure/o1/train_prompted_s44"),
        ),
    },
    {
        "id": "o2_fixed_controls_and_sparse",
        "title": "O2 engineering diagnostics — fixed sample and axes",
        "substrate": "jina-4m-nested-o2",
        "maps": (
            ("control s42", "/data/latent-basemap/closure/o2/control_s42"),
            ("control s43", "/data/latent-basemap/closure/o2/control_s43"),
            ("control s44", "/data/latent-basemap/closure/o2/control_s44"),
            ("sparse w2 s42", "/data/latent-basemap/closure/o2/sparse_w2_s42"),
            ("sparse w10 s42", "/data/latent-basemap/closure/o2/sparse_w10_s42"),
            ("sparse w50 s42", "/data/latent-basemap/closure/o2/sparse_w50_s42"),
        ),
    },
)


def _ids_hash(values) -> str:
    values = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return sha256_bytes(canonical_json({"length": len(values), "dtype": "int64"}) +
                        values.tobytes())


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "map"


def _registered_input_paths(registry=ROUND0005_FIXED_COMPARISONS) -> list[str]:
    paths = []
    for comparison in registry:
        for _, run_dir in comparison["maps"]:
            paths.extend([
                os.path.join(run_dir, "coords.parquet"),
                os.path.join(run_dir, "config.yaml"),
                os.path.join(run_dir, "results.json"),
            ])
    return paths


def _compact_historical_signature(value: dict) -> dict:
    path = value.get("path")
    return {
        "canonical_path": os.path.realpath(path),
        "kind": value.get("kind"),
        "bytes": value.get("bytes"),
        "sha256": value.get("sha256"),
    }


def _validate_historical_registration(registry, historical_manifest_path: str) -> dict:
    with open(historical_manifest_path, encoding="utf-8") as handle:
        historical = json.load(handle)
    by_comparison = {item.get("comparison_id"): item
                     for item in historical.get("comparisons", [])}
    for comparison in registry:
        old = by_comparison.get(comparison["id"])
        if not isinstance(old, dict):
            raise ValueError(f"historical render manifest lacks {comparison['id']}")
        old_maps = {item.get("label"): item for item in old.get("maps", [])}
        for label, run_dir in comparison["maps"]:
            item = old_maps.get(label)
            if not isinstance(item, dict):
                raise ValueError(
                    f"historical render manifest lacks {comparison['id']}/{label}")
            expected_paths = [os.path.join(run_dir, name)
                              for name in ("coords.parquet", "config.yaml", "results.json")]
            old_signatures = [item.get("coordinate_signature"),
                              *(item.get("source_signatures") or [])]
            if len(old_signatures) != 3:
                raise ValueError(f"historical registration is incomplete for {label}")
            for expected_path, old_signature in zip(expected_paths, old_signatures):
                if (not isinstance(old_signature, dict) or
                        os.path.realpath(old_signature.get("path", "")) !=
                        os.path.realpath(expected_path)):
                    raise ValueError(f"historical registered path drift for {label}")
                observed = expected_input_signature(expected_path)
                if observed != _compact_historical_signature(old_signature):
                    raise ValueError(f"historical registered bytes drift for {expected_path}")
    return expected_input_signature(historical_manifest_path)


def _namespace_for_registered_comparison(comparison: dict) -> dict:
    loaded_ids = []
    coordinate_signatures = []
    for label, run_dir in comparison["maps"]:
        coords_path = os.path.join(run_dir, "coords.parquet")
        _, ids = load_coords(coords_path)
        if ids is None:
            raise ValueError(f"Round 0005 registered map {label} has no semantic IDs")
        ids = np.asarray(ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Round 0005 registered map {label} has duplicate IDs")
        loaded_ids.append(ids)
        coordinate_signatures.append(expected_input_signature(coords_path))
    universe = np.sort(loaded_ids[0])
    for ids in loaded_ids[1:]:
        if not np.array_equal(np.sort(ids), universe):
            raise ValueError(
                f"Round 0005 registered comparison {comparison['id']} has mismatched ID universes")
    universe_sha = ordered_array_sha256(universe)
    corpus_identity = sha256_bytes(canonical_json({
        "schema": "round0005_registered_render_substrate.v1",
        "substrate": comparison["substrate"],
        "row_count": len(universe),
        "semantic_universe_sha256": universe_sha,
        "registered_coordinate_inputs": coordinate_signatures,
    }))
    return {
        "schema": "basemap_semantic_id_namespace.v1",
        "name": f"round0005/{comparison['substrate']}/coordinate-row-id",
        "kind": "coordinate_semantic_id",
        "corpus_identity_sha256": corpus_identity,
        "universe_sha256": universe_sha,
        "row_count": len(universe),
    }


def build_round0005_fixed_spec(*, output_dir: str,
                               registry=ROUND0005_FIXED_COMPARISONS,
                               historical_manifest_path: str =
                               ROUND0005_HISTORICAL_MANIFEST) -> dict:
    """Build the one exact G1/O1/O2 Round 0005 render spec from reviewed paths."""
    output_dir = os.path.realpath(output_dir)
    if not output_dir.startswith("/data/latent-basemap/runs/round-0005/"):
        raise ValueError("Round 0005 render root must be below its /data run root")
    if output_dir.startswith("/data/latent-basemap/runs/round-0001/"):
        raise ValueError("Round 0005 must never overwrite Round 0001 renders")
    historical_signature = _validate_historical_registration(
        registry, historical_manifest_path)
    comparisons = []
    for registered in registry:
        namespace = _namespace_for_registered_comparison(registered)
        comparisons.append({
            "id": registered["id"], "title": registered["title"],
            "substrate": registered["substrate"],
            "semantic_id_namespace": namespace,
            "maps": [{
                "label": label,
                "coords": os.path.join(run_dir, "coords.parquet"),
                "sources": [os.path.join(run_dir, "config.yaml"),
                            os.path.join(run_dir, "results.json")],
                "semantic_id_namespace": namespace,
            } for label, run_dir in registered["maps"]],
        })
    input_paths = [historical_manifest_path, *_registered_input_paths(registry)]
    spec = {
        "schema": ROUND0005_RENDER_SPEC_SCHEMA,
        "round_id": "0005",
        "output_dir": output_dir,
        "sample_seed": 20260716,
        "sample_size": 50000,
        "axis_padding_fraction": 0.02,
        "max_columns": 3,
        "dpi": 180,
        "historical_manifest": historical_signature,
        "input_signatures": [expected_input_signature(path) for path in input_paths],
        "comparisons": comparisons,
    }
    validate_round0005_fixed_spec(spec, registry=registry,
                                  historical_manifest_path=historical_manifest_path)
    return spec


def validate_round0005_fixed_spec(spec: dict, *,
                                  registry=ROUND0005_FIXED_COMPARISONS,
                                  historical_manifest_path: str =
                                  ROUND0005_HISTORICAL_MANIFEST) -> dict:
    required = {"schema", "round_id", "output_dir", "sample_seed", "sample_size",
                "axis_padding_fraction", "max_columns", "dpi", "historical_manifest",
                "input_signatures", "comparisons"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError(f"Round 0005 fixed render spec fields must be {sorted(required)}")
    if (spec["schema"] != ROUND0005_RENDER_SPEC_SCHEMA or spec["round_id"] != "0005" or
            spec["sample_seed"] != 20260716 or spec["sample_size"] != 50000 or
            spec["axis_padding_fraction"] != 0.02 or spec["max_columns"] != 3 or
            spec["dpi"] != 180):
        raise ValueError("Round 0005 fixed render policy changed")
    output_dir = os.path.realpath(spec["output_dir"])
    if (not output_dir.startswith("/data/latent-basemap/runs/round-0005/") or
            output_dir.startswith("/data/latent-basemap/runs/round-0001/")):
        raise ValueError("Round 0005 fixed render output root is invalid")
    if spec["historical_manifest"] != expected_input_signature(historical_manifest_path):
        raise ValueError("Round 0005 historical render registration signature changed")
    expected_paths = [historical_manifest_path, *_registered_input_paths(registry)]
    expected_signatures = [expected_input_signature(path) for path in expected_paths]
    if spec["input_signatures"] != expected_signatures:
        raise ValueError("Round 0005 fixed render expected input signatures changed")
    comparisons = spec["comparisons"]
    if not isinstance(comparisons, list) or len(comparisons) != 3:
        raise ValueError("Round 0005 fixed render must contain exactly three comparisons")
    for actual, expected in zip(comparisons, registry):
        if (actual.get("id") != expected["id"] or actual.get("title") != expected["title"] or
                actual.get("substrate") != expected["substrate"]):
            raise ValueError("Round 0005 comparison registry/order changed")
        namespace = validate_semantic_namespace(actual.get("semantic_id_namespace"))
        maps = actual.get("maps")
        if not isinstance(maps, list) or len(maps) != len(expected["maps"]):
            raise ValueError(f"Round 0005 comparison {expected['id']} map count changed")
        for item, (label, run_dir) in zip(maps, expected["maps"]):
            wanted = {
                "label": label, "coords": os.path.join(run_dir, "coords.parquet"),
                "sources": [os.path.join(run_dir, "config.yaml"),
                            os.path.join(run_dir, "results.json")],
                "semantic_id_namespace": namespace,
            }
            if item != wanted:
                raise ValueError(f"Round 0005 registered map changed: {expected['id']}/{label}")
    return spec


def _semantic_ids(item: dict, coord_ids, n: int, coordinate_signature: dict,
                  expected_namespace: dict):
    """Return proved semantic IDs and their identity declaration.

    Native coordinate IDs are preferred.  A table without IDs is admissible
    only with an explicit row-position declaration bound both to the exact
    coordinate bytes and to the ordered ``0..N-1`` universe hash.
    """
    if coord_ids is not None:
        item_namespace = validate_semantic_namespace(item.get("semantic_id_namespace"))
        if item_namespace != expected_namespace:
            raise ValueError(f"map {item['label']} semantic namespace mismatch")
        if item_namespace["kind"] != "coordinate_semantic_id":
            raise ValueError(f"map {item['label']} coordinate IDs need coordinate-ID namespace")
        ids = np.asarray(coord_ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"map {item['label']} has duplicate coordinate IDs")
        return ids, {"kind": "coordinate_ids", "namespace": item_namespace,
                     "ids_sha256": ordered_array_sha256(ids)}
    declaration = item.get("positional_identity")
    if not isinstance(declaration, dict):
        raise ValueError(
            f"map {item['label']} has no coordinate IDs and no explicit "
            "content-bound positional_identity declaration")
    required = {"kind", "namespace", "row_count", "coordinate_sha256", "universe_sha256"}
    if set(declaration) != required:
        raise ValueError(
            f"map {item['label']} positional_identity fields must be exactly {sorted(required)}")
    if declaration["kind"] != "row_position":
        raise ValueError(f"map {item['label']} positional identity kind must be row_position")
    declaration_namespace = validate_semantic_namespace(declaration["namespace"])
    if declaration_namespace != expected_namespace:
        raise ValueError(f"map {item['label']} positional semantic namespace mismatch")
    if declaration_namespace["kind"] != "row_position":
        raise ValueError(f"map {item['label']} positional identity needs row-position namespace")
    if (not isinstance(declaration["row_count"], int) or
            isinstance(declaration["row_count"], bool) or declaration["row_count"] != n):
        raise ValueError(f"map {item['label']} positional row_count does not match coordinates")
    if declaration["coordinate_sha256"] != coordinate_signature["sha256"]:
        raise ValueError(f"map {item['label']} positional declaration is not bound to coordinate bytes")
    ids = np.arange(n, dtype=np.int64)
    if declaration["universe_sha256"] != ordered_array_sha256(np.sort(ids)):
        raise ValueError(f"map {item['label']} positional universe hash is invalid")
    return ids, {"kind": "positional_identity", "declaration": declaration,
                 "declaration_sha256": sha256_bytes(canonical_json(declaration))}


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


def _spec_input_paths(spec: dict) -> list[str]:
    paths = []
    for comparison in spec.get("comparisons") or []:
        for item in comparison.get("maps") or []:
            paths.append(item["coords"])
            paths.extend(item.get("sources") or [])
    return paths


def _published_signature(work_path: str, *, work_root: str, published_root: str) -> dict:
    signature = path_signature(work_path)
    relative = os.path.relpath(work_path, work_root)
    published = os.path.join(published_root, relative)
    signature["path"] = published
    signature["resolved_path"] = published
    return signature


def _publish_directory_noreplace(work_dir: str, published_dir: str) -> None:
    """Atomically publish a directory while refusing even an empty racing root."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd, os.fsencode(work_dir), at_fdcwd, os.fsencode(published_dir),
        rename_noreplace)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"refuse racing render output root: {published_dir}")
    raise OSError(error, os.strerror(error), published_dir)


def _render_into(spec: dict, *, spec_path: str, work_dir: str, published_dir: str,
                 pre_signatures: dict[str, dict], spec_signature: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = work_dir
    seed = int(spec.get("sample_seed", 20260716))
    sample_size = int(spec.get("sample_size", 50000))
    substrates = {}
    entries = []

    comparisons = spec.get("comparisons", [])
    comparison_ids = [comparison.get("id") for comparison in comparisons]
    if (any(not isinstance(value, str) or not value for value in comparison_ids)
            or len(comparison_ids) != len(set(comparison_ids))
            or len({_slug(value) for value in comparison_ids}) != len(comparison_ids)):
        raise ValueError("render comparison IDs must be nonempty and collision-proof")
    for comparison in comparisons:
        maps = comparison.get("maps") or []
        if not maps:
            raise ValueError(f"comparison {comparison.get('id')} has no maps")
        labels = [item.get("label") for item in maps]
        if (any(not isinstance(label, str) or not label for label in labels) or
                len(labels) != len(set(labels))):
            raise ValueError(f"comparison {comparison.get('id')} map labels must be unique")
        expected_namespace = validate_semantic_namespace(
            comparison.get("semantic_id_namespace"))
        loaded = []
        for item in maps:
            coords, coord_ids = load_coords(item["coords"])
            if coords.shape[1] < 2:
                raise ValueError(f"map {item['label']} has fewer than two coordinate dimensions")
            coordinate_signature = pre_signatures[os.path.realpath(item["coords"])]
            semantic_ids, identity = _semantic_ids(
                item, coord_ids, len(coords), coordinate_signature,
                expected_namespace)
            loaded.append((item, coords, semantic_ids, identity, coordinate_signature))
        lengths = {len(coords) for _, coords, _, _, _ in loaded}
        if len(lengths) != 1:
            raise ValueError(f"comparison {comparison['id']} map lengths differ: {sorted(lengths)}")
        n = lengths.pop()
        canonical_universe = np.sort(loaded[0][2])
        universe_hash = ordered_array_sha256(canonical_universe)
        if expected_namespace["universe_sha256"] != universe_hash or \
                expected_namespace["row_count"] != n:
            raise ValueError(
                f"comparison {comparison['id']} semantic namespace does not bind its universe")
        for item, _, semantic_ids, _, _ in loaded[1:]:
            candidate = np.sort(semantic_ids)
            if not np.array_equal(candidate, canonical_universe):
                missing = np.setdiff1d(canonical_universe, candidate, assume_unique=True)
                extra = np.setdiff1d(candidate, canonical_universe, assume_unique=True)
                raise ValueError(
                    f"comparison {comparison['id']} map {item['label']} semantic ID universe "
                    f"differs: missing={missing[:5].tolist()} extra={extra[:5].tolist()}")
        substrate = comparison["substrate"]
        if substrate in substrates:
            substrate_state = substrates[substrate]
            if (substrate_state["universe_sha256"] != universe_hash or
                    substrate_state["semantic_id_namespace"] != expected_namespace):
                raise ValueError(
                    f"substrate {substrate} semantic namespace/universe differs in comparison "
                    f"{comparison['id']}")
            ids = substrate_state["sample_ids"]
        else:
            rng = np.random.RandomState(seed)
            selected = np.sort(rng.choice(n, min(sample_size, n), replace=False)).astype(np.int64)
            ids = canonical_universe[selected]
            sample_work_path = os.path.join(
                out_dir, f"sample_semantic_ids_{_slug(substrate)}.npy")
            atomic_save_new_npy(sample_work_path, ids, immutable=True)
            substrates[substrate] = {
                "sample_ids": ids,
                "universe_sha256": universe_hash,
                "semantic_id_namespace": expected_namespace,
                "sample_work_path": sample_work_path,
                "sample_path": os.path.join(
                    published_dir, os.path.basename(sample_work_path)),
            }

        sampled = []
        gathered_rows = []
        for item, coords, semantic_ids, _, _ in loaded:
            order = np.argsort(semantic_ids, kind="mergesort")
            sorted_ids = semantic_ids[order]
            positions = np.searchsorted(sorted_ids, ids)
            if (len(positions) != len(ids) or np.any(positions >= len(sorted_ids))
                    or not np.array_equal(sorted_ids[positions], ids)):
                raise ValueError(f"map {item['label']} cannot gather sampled semantic IDs")
            rows = order[positions].astype(np.int64, copy=False)
            gathered_rows.append(rows)
            sampled.append(np.asarray(coords[rows, :2], dtype=np.float32))
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
        for map_index, (ax, loaded_item, points, rows) in enumerate(
                zip(flat_axes, loaded, sampled, gathered_rows)):
            item, coords, coord_ids, identity, coordinate_signature = loaded_item
            ax.scatter(points[:, 0], points[:, 1], s=0.15, alpha=0.35,
                       linewidths=0, rasterized=True)
            ax.set_title(item["label"])
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
            sources = [coordinate_signature]
            for source in item.get("sources", []):
                sources.append(pre_signatures[os.path.realpath(source)])
            rows_work_path = os.path.join(
                out_dir, f"gathered_rows_{_slug(comparison['id'])}_{map_index:02d}_{_slug(item['label'])}.npy")
            atomic_save_new_npy(rows_work_path, rows, immutable=True)
            map_entries.append({
                "label": item["label"],
                "coordinate_dimensions_rendered": [0, 1],
                "coordinate_signature": sources[0],
                "source_signatures": sources[1:],
                "semantic_identity": identity,
                "coordinate_ids_hash": _ids_hash(coord_ids),
                "semantic_universe_sha256": universe_hash,
                "gathered_row_positions": _published_signature(
                    rows_work_path, work_root=work_dir, published_root=published_dir),
                "gathered_row_positions_sha256": _ids_hash(rows),
                "diagnostics": _diagnostics(points),
            })
        for ax in flat_axes[len(maps):]:
            ax.axis("off")
        fig.suptitle(comparison.get("title", comparison["id"]))
        fig.tight_layout()
        image_work_path = os.path.join(out_dir, f"{_slug(comparison['id'])}.png")
        atomic_build_new_file(
            image_work_path,
            lambda tmp: fig.savefig(
                tmp, format="png", dpi=int(spec.get("dpi", 180)), bbox_inches="tight"),
            immutable=True)
        plt.close(fig)
        entries.append({
            "comparison_id": comparison["id"],
            "substrate": substrate,
            "semantic_universe_sha256": universe_hash,
            "sample_ids_path": substrates[substrate]["sample_path"],
            "sample_ids_sha256": _ids_hash(ids),
            "sample_count": len(ids),
            "axis_policy": "shared union extent; no per-map normalization",
            "xlim": xlim,
            "ylim": ylim,
            "image": _published_signature(
                image_work_path, work_root=work_dir, published_root=published_dir),
            "maps": map_entries,
        })

    return {
        "schema": (ROUND0005_RENDER_MANIFEST_SCHEMA
                   if spec.get("schema") == ROUND0005_RENDER_SPEC_SCHEMA
                   else "fixed_comparison_render_manifest.v3"),
        "spec": spec_signature,
        "renderer": pre_signatures[os.path.realpath(__file__)],
        "checkout": git_checkout_state(os.path.dirname(os.path.dirname(__file__))),
        "sample_seed": seed,
        "sample_size_requested": sample_size,
        "comparisons": entries,
    }


def render(spec: dict, *, spec_path: str, expected_spec_signature: dict | None = None) -> dict:
    """Render into a temporary sibling, verify input stability, publish one root."""
    published_dir = os.path.realpath(spec["output_dir"])
    if not published_dir.startswith("/data/"):
        raise ValueError("render output_dir must be under /data")
    refuse_existing(published_dir, label="render output root")
    if spec.get("schema") == ROUND0005_RENDER_SPEC_SCHEMA:
        validate_round0005_fixed_spec(spec)

    spec_pre = path_signature(spec_path)
    if expected_spec_signature is not None and spec_pre != expected_spec_signature:
        raise ValueError("render spec changed while it was being loaded")
    paths = [*_spec_input_paths(spec)]
    if spec.get("schema") == ROUND0005_RENDER_SPEC_SCHEMA:
        # Historical registration is a scientific input too.  Keep it inside
        # the same pre/post read window as every registered map source.
        paths.append(spec["historical_manifest"]["canonical_path"])
    paths.append(__file__)
    canonical_paths = [os.path.realpath(path) for path in paths]
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ValueError("render input paths contain a canonical duplicate")
    pre_signatures = {canonical: path_signature(canonical)
                      for canonical in canonical_paths}
    if spec.get("schema") == ROUND0005_RENDER_SPEC_SCHEMA:
        declared = spec["input_signatures"]
        declared_by_path = {item["canonical_path"]: item for item in declared}
        for path in _spec_input_paths(spec):
            canonical = os.path.realpath(path)
            if declared_by_path.get(canonical) != expected_input_signature(canonical):
                raise ValueError(f"Round 0005 render input expectation mismatch: {canonical}")

    parent = ensure_data_directory(
        os.path.dirname(published_dir), label="render output parent")
    work_dir = tempfile.mkdtemp(prefix=f".{os.path.basename(published_dir)}.tmp.", dir=parent)
    published = False
    try:
        manifest = _render_into(
            spec, spec_path=spec_path, work_dir=work_dir,
            published_dir=published_dir, pre_signatures=pre_signatures,
            spec_signature=spec_pre)
        # Re-hash after all reads and image construction.  Neither the spec nor a
        # coordinate/config/result/renderer byte may change across the render.
        spec_post = path_signature(spec_path)
        post_signatures = {canonical: path_signature(canonical)
                           for canonical in canonical_paths}
        if spec_post != spec_pre:
            raise ValueError("render spec changed during rendering (TOCTOU)")
        changed = [path for path in canonical_paths
                   if post_signatures[path] != pre_signatures[path]]
        if changed:
            raise ValueError(f"render input changed during rendering (TOCTOU): {changed}")
        manifest["input_toctou"] = {
            "schema": "fixed_render_input_toctou.v1",
            "pre_signatures": [pre_signatures[path] for path in canonical_paths],
            "post_matches_pre": True,
            "spec_pre_signature": spec_pre,
            "spec_post_matches_pre": True,
        }
        if spec.get("schema") == ROUND0005_RENDER_SPEC_SCHEMA:
            images = [item["image"]["path"] for item in manifest["comparisons"]]
            expected_images = [os.path.join(published_dir, f"{item['id']}.png")
                               for item in ROUND0005_FIXED_COMPARISONS]
            if images != expected_images or len(manifest["comparisons"]) != 3:
                raise ValueError("Round 0005 renderer did not produce its exact three-image set")
        atomic_write_new_json(
            os.path.join(work_dir, "render-manifest.json"), manifest, immutable=True)
        os.chmod(work_dir, 0o555)
        # Linux renameat2(RENAME_NOREPLACE) publishes the complete sibling in one
        # namespace operation and refuses even an empty destination created by a
        # racing process.  Plain os.rename would silently replace that empty root.
        _publish_directory_noreplace(work_dir, published_dir)
        published = True
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return manifest
    finally:
        if not published and os.path.isdir(work_dir):
            # Publication makes the root read-only.  If the no-replace rename is
            # rejected by a racing destination, restore owner write permission
            # only on our private temporary root so it can be removed completely.
            os.chmod(work_dir, 0o755)
            shutil.rmtree(work_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", nargs="?")
    parser.add_argument("--build-round0005-spec", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--manifest")
    args = parser.parse_args(argv)
    if args.build_round0005_spec:
        if not args.spec or not args.output_dir or args.manifest:
            parser.error("--build-round0005-spec requires SPEC --output-dir and no --manifest")
        spec = build_round0005_fixed_spec(output_dir=args.output_dir)
        atomic_write_new_json(args.spec, spec, immutable=True)
        print(args.spec)
        return 0
    if not args.spec or args.output_dir:
        parser.error("render mode requires SPEC and does not accept --output-dir")
    spec_signature = path_signature(args.spec)
    with open(args.spec, encoding="utf-8") as handle:
        spec = json.load(handle)
    if path_signature(args.spec) != spec_signature:
        raise ValueError("render spec changed while it was being parsed")
    out = os.path.join(os.path.realpath(spec["output_dir"]), "render-manifest.json")
    if args.manifest and os.path.realpath(args.manifest) != out:
        parser.error("atomic render manifest path is fixed inside output_dir")
    render(spec, spec_path=args.spec, expected_spec_signature=spec_signature)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
