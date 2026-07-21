#!/usr/bin/env python3
"""Round 0023 cross-seed layout-disparity scorer."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import atomic_build_new_file, atomic_write_new_json, create_fresh_directory
from basemap.round0023_program import R0019_REFERENCE


SCHEMA = "round0023-layout-disparity-v1"
R0019_COORDINATE_ROOT = "/data/latent-basemap/runs/round-0019/queue/artifacts/coordinates"
R0019_MODEL = "/data/latent-basemap/runs/round-0019/queue/artifacts/train/model.pt"
R0019_PANEL = "/data/latent-basemap/runs/round-0019/queue/artifacts/panel/panel.json"
R0019_RENDER = (
    "/data/latent-basemap/runs/round-0019/queue/artifacts/semantic-renders/"
    "render-manifest.json"
)
R0019_SAMPLE_IDS = (
    "/data/latent-basemap/runs/round-0019/queue/artifacts/semantic-renders/"
    "sample-semantic-ids.npy"
)


@dataclass(frozen=True)
class CoordinateStream:
    root: str
    receipt: dict[str, Any]
    members: tuple[dict[str, Any], ...]

    @classmethod
    def open(cls, root: str) -> "CoordinateStream":
        canonical = os.path.realpath(root)
        if canonical != root or not os.path.isdir(canonical) or os.path.islink(root):
            raise ValueError(f"coordinate root is not canonical: {root}")
        receipt_path = os.path.join(canonical, "actual-transform.json")
        with open(receipt_path, encoding="utf-8") as handle:
            receipt = json.load(handle)
        body = {key: receipt[key] for key in receipt if key != "identity_sha256"}
        if receipt.get("identity_sha256") != sha256_bytes(canonical_json(body)):
            raise ValueError(f"coordinate transform receipt seal changed: {receipt_path}")
        stream = receipt.get("stream_capability")
        capability = stream.get("capability_payload") if isinstance(stream, dict) else None
        if (
            not isinstance(capability, dict)
            or stream.get("capability_sha256") != sha256_bytes(canonical_json(capability))
        ):
            raise ValueError(f"coordinate stream capability seal changed: {receipt_path}")
        plan = capability.get("plan")
        ordered = capability.get("ordered_chunks")
        if (
            capability.get("schema") != "round0013-stream-output-v1"
            or not isinstance(plan, dict)
            or plan.get("output", {}).get("shape") != [30_000_000, 2]
            or plan.get("output", {}).get("dtype") != "<f4"
            or not isinstance(ordered, list)
            or len(ordered) != 30
        ):
            raise ValueError(f"coordinate stream geometry changed: {receipt_path}")
        members: list[dict[str, Any]] = []
        cursor = 0
        for position, item in enumerate(ordered):
            start = int(item["global_row_start"])
            stop = int(item["global_row_stop"])
            if (
                int(item["chunk_index"]) != position
                or start != cursor
                or stop != min(cursor + 1_000_000, 30_000_000)
            ):
                raise ValueError(f"coordinate stream order changed: {receipt_path}")
            path = os.path.join(canonical, f"chunk-{position:05d}", "coordinates.npy")
            signature = expected_input_signature(path)
            if (
                signature["sha256"] != item["sha256"]
                or int(signature["bytes"]) != int(item["size_bytes"])
            ):
                raise ValueError(f"coordinate chunk signature changed: {path}")
            members.append({**item, "path": path, "signature": signature})
            cursor = stop
        if cursor != 30_000_000:
            raise ValueError(f"coordinate stream row coverage changed: {receipt_path}")
        return cls(root=canonical, receipt=receipt, members=tuple(members))

    def gather(self, rows: np.ndarray) -> np.ndarray:
        indices = np.asarray(rows, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("coordinate gather requires a nonempty 1-D index")
        if np.any(indices < 0) or np.any(indices >= 30_000_000):
            raise IndexError("coordinate gather index out of bounds")
        out = np.empty((len(indices), 2), dtype="<f4")
        for member in self.members:
            lo = int(member["global_row_start"])
            hi = int(member["global_row_stop"])
            selected = np.flatnonzero((indices >= lo) & (indices < hi))
            if not len(selected):
                continue
            chunk = np.load(member["path"], mmap_mode="r", allow_pickle=False)
            if chunk.shape != (hi - lo, 2) or chunk.dtype.str != "<f4":
                raise ValueError(f"coordinate chunk geometry changed: {member['path']}")
            out[selected] = chunk[indices[selected] - lo]
            del chunk
        if not np.isfinite(out).all():
            raise ValueError("coordinate sample contains non-finite values")
        return out


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def optimal_similarity(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if reference.shape != moving.shape or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("Procrustes inputs must both be shaped (N, 2)")
    ref_mean = reference.mean(axis=0)
    mov_mean = moving.mean(axis=0)
    ref_centered = reference - ref_mean
    mov_centered = moving - mov_mean
    denom = float(np.square(mov_centered).sum())
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError("moving layout is degenerate")
    u, singular, vt = np.linalg.svd(mov_centered.T @ ref_centered, full_matrices=False)
    rotation = u @ vt
    scale = float(singular.sum() / denom)
    aligned = scale * (mov_centered @ rotation) + ref_mean
    ref_norm = float(np.linalg.norm(ref_centered))
    mov_norm = float(np.linalg.norm(mov_centered))
    if ref_norm <= 0.0 or mov_norm <= 0.0:
        raise ValueError("layout is degenerate")
    norm_aligned = (mov_centered / mov_norm) @ rotation
    norm_ref = ref_centered / ref_norm
    disparity = float(np.square(norm_ref - norm_aligned).sum())
    return aligned.astype(np.float64), {
        "scale": scale,
        "rotation": rotation.tolist(),
        "reference_mean": ref_mean.tolist(),
        "moving_mean": mov_mean.tolist(),
        "normalized_sum_squared_disparity": disparity,
    }


def local_radius(points: np.ndarray, *, k: int = 15) -> np.ndarray:
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) <= k:
        raise ValueError("local radius input must be shaped (N, 2) with N > k")
    distances, indices = cKDTree(points).query(points, k=k + 1, workers=-1)
    radii = []
    for row, (dist_row, index_row) in enumerate(zip(distances, indices)):
        kept = [float(distance) for distance, index in zip(dist_row, index_row) if int(index) != row]
        if len(kept) < k:
            raise ValueError("could not compute self-excluded local radius")
        radii.append(kept[k - 1])
    values = np.asarray(radii, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("seed-42 local radii must be finite nonnegative")
    return values


def knn_indices(points: np.ndarray, *, k: int = 15) -> np.ndarray:
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    distances, indices = cKDTree(points).query(points, k=k + 1, workers=-1)
    del distances
    out = np.empty((len(points), k), dtype=np.int64)
    for row, index_row in enumerate(indices):
        kept = [int(index) for index in index_row if int(index) != row]
        if len(kept) < k:
            raise ValueError("could not compute self-excluded nearest neighbors")
        out[row] = kept[:k]
    return out


def retention_and_jaccard(left: np.ndarray, right: np.ndarray, *, k: int = 15) -> dict[str, Any]:
    if left.shape != right.shape or left.shape[1] != k:
        raise ValueError("neighbor matrices must share shape (N, k)")
    counts = np.empty(left.shape[0], dtype=np.float64)
    for row in range(left.shape[0]):
        intersection = len(set(map(int, left[row])).intersection(map(int, right[row])))
        counts[row] = intersection
    retention = counts / float(k)
    jaccard = counts / (float(2 * k) - counts)
    return {
        "k": k,
        "mean_retention": float(retention.mean()),
        "median_retention": float(np.median(retention)),
        "p10_retention": float(np.quantile(retention, 0.10)),
        "mean_jaccard": float(jaccard.mean()),
        "median_jaccard": float(np.median(jaccard)),
        "p10_jaccard": float(np.quantile(jaccard, 0.10)),
    }


def _quantile_with_infinite(values: np.ndarray, q: float) -> tuple[float | None, bool]:
    value = float(np.quantile(values, q))
    return (value if np.isfinite(value) else None), bool(np.isposinf(value))


def pair_metrics(
    reference_points: np.ndarray,
    moving_points: np.ndarray,
    radius15_seed42: np.ndarray,
    reference_knn: np.ndarray,
    moving_knn: np.ndarray,
    *,
    pair: tuple[str, str],
) -> dict[str, Any]:
    aligned, transform = optimal_similarity(reference_points, moving_points)
    residual = np.linalg.norm(np.asarray(reference_points, dtype=np.float64) - aligned, axis=1)
    radius = np.asarray(radius15_seed42, dtype=np.float64)
    if radius.shape != residual.shape:
        raise ValueError(f"local radius shape mismatch for pair {pair}")
    positive = radius > 0.0
    drift = np.empty_like(residual)
    drift[positive] = residual[positive] / radius[positive]
    zero_residual = residual <= 1e-12
    drift[~positive & zero_residual] = 0.0
    drift[~positive & ~zero_residual] = np.inf
    if np.isnan(drift).any() or np.isneginf(drift).any():
        raise ValueError(f"invalid local-r15 drift for pair {pair}")
    median, median_inf = _quantile_with_infinite(drift, 0.50)
    p90, p90_inf = _quantile_with_infinite(drift, 0.90)
    p95, p95_inf = _quantile_with_infinite(drift, 0.95)
    finite = np.isfinite(drift)
    return {
        "pair": list(pair),
        "procrustes": transform,
        "zero_seed42_r15_count": int((~positive).sum()),
        "undefined_infinite_drift_count": int(np.isposinf(drift).sum()),
        "zero_radius_zero_residual_count": int((~positive & zero_residual).sum()),
        "finite_drift_count": int(finite.sum()),
        "drift_quantile_policy": (
            "zero seed42 r15 with nonzero residual is +infinity; quantiles are "
            "computed over all sample rows and represented as null only if infinite"
        ),
        "median_drift_local_r15": median,
        "median_drift_local_r15_is_infinite": median_inf,
        "p90_drift_local_r15": p90,
        "p90_drift_local_r15_is_infinite": p90_inf,
        "p95_drift_local_r15": p95,
        "p95_drift_local_r15_is_infinite": p95_inf,
        "finite_max_drift_local_r15": float(drift[finite].max()) if finite.any() else None,
        "max_drift_local_r15_is_infinite": bool(np.isposinf(drift).any()),
        "neighbor_overlap": retention_and_jaccard(reference_knn, moving_knn, k=15),
    }


def _assert_pinned(path: str, expected_sha256: str) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise ValueError(f"pinned artifact changed: {path}")
    return signature


def _load_sample_ids(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    signature = _assert_pinned(path, R0019_REFERENCE["semantic_sample_ids_file_sha256"])
    sample_ids = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
    if (
        sample_ids.ndim != 1
        or len(sample_ids) != 50_000
        or not np.array_equal(sample_ids, np.unique(sample_ids))
        or sample_ids[0] < 0
        or sample_ids[-1] >= 30_000_000
    ):
        raise ValueError("Round 0019 semantic sample IDs changed")
    return sample_ids, signature


def draw_fixed_axis(points_by_seed: dict[str, np.ndarray], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_points = np.vstack([points_by_seed[key] for key in ("42", "43", "44")])
    lo = all_points.min(axis=0)
    hi = all_points.max(axis=0)
    pad = np.maximum((hi - lo) * 0.03, 1e-6)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for axis, seed in zip(axes, ("42", "43", "44")):
        pts = points_by_seed[seed]
        axis.scatter(pts[:, 0], pts[:, 1], s=0.12, alpha=0.35, linewidths=0, rasterized=True)
        axis.set_title(f"seed {seed}")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(float(lo[0] - pad[0]), float(hi[0] + pad[0]))
        axis.set_ylim(float(lo[1] - pad[1]), float(hi[1] + pad[1]))
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Round 0023 fixed-axis 50k semantic sample")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_layout_disparity(
    *,
    output_root: str,
    coordinate_roots: dict[str, str],
    train_roots: dict[str, str],
    panel_roots: dict[str, str],
    render_roots: dict[str, str],
    release_sha: str,
    implementation_base_commit: str,
    sample_ids_path: str = R0019_SAMPLE_IDS,
) -> dict[str, Any]:
    started = time.monotonic()
    output = create_fresh_directory(output_root, label="Round 0023 layout-disparity output")
    sample_ids, sample_signature = _load_sample_ids(sample_ids_path)

    _assert_pinned(R0019_MODEL, R0019_REFERENCE["seed42_model_sha256"])
    _assert_pinned(
        os.path.join(R0019_COORDINATE_ROOT, "actual-transform.json"),
        R0019_REFERENCE["seed42_coordinate_receipt_sha256"],
    )
    _assert_pinned(R0019_PANEL, R0019_REFERENCE["seed42_panel_sha256"])

    streams = {
        "42": CoordinateStream.open(R0019_COORDINATE_ROOT),
        "43": CoordinateStream.open(coordinate_roots["43"]),
        "44": CoordinateStream.open(coordinate_roots["44"]),
    }
    points_by_seed = {seed: stream.gather(sample_ids) for seed, stream in streams.items()}
    radius15 = local_radius(points_by_seed["42"], k=15)
    knn_by_seed = {seed: knn_indices(points, k=15) for seed, points in points_by_seed.items()}
    pairs = [
        pair_metrics(
            points_by_seed["42"], points_by_seed["43"], radius15,
            knn_by_seed["42"], knn_by_seed["43"], pair=("42", "43")
        ),
        pair_metrics(
            points_by_seed["42"], points_by_seed["44"], radius15,
            knn_by_seed["42"], knn_by_seed["44"], pair=("42", "44")
        ),
        pair_metrics(
            points_by_seed["43"], points_by_seed["44"], radius15,
            knn_by_seed["43"], knn_by_seed["44"], pair=("43", "44")
        ),
    ]

    render_path = os.path.join(output, "fixed-axis-three-seed-sample.png")
    atomic_build_new_file(render_path, lambda path: draw_fixed_axis(points_by_seed, path), immutable=True)
    radius_path = os.path.join(output, "seed42-local-r15.npy")
    from basemap.output_safety import atomic_save_new_npy

    atomic_save_new_npy(radius_path, radius15.astype("<f4"), immutable=True)

    body = {
        "schema": SCHEMA,
        "round_id": "0023",
        "release_sha": release_sha,
        "implementation_base_commit": implementation_base_commit,
        "scientific_reference": dict(R0019_REFERENCE),
        "sample": {
            "semantic_ids": sample_signature,
            "semantic_ids_array_sha256": ordered_array_sha256(sample_ids),
            "sample_size": int(len(sample_ids)),
            "sample_universe": "30m-minilm-row-position",
        },
        "artifacts": {
            "42": {
                "model": expected_input_signature(R0019_MODEL),
                "coordinate_receipt": expected_input_signature(
                    os.path.join(R0019_COORDINATE_ROOT, "actual-transform.json")
                ),
                "panel": expected_input_signature(R0019_PANEL),
                "render": expected_input_signature(R0019_RENDER),
            },
            "43": {
                "model": expected_input_signature(os.path.join(train_roots["43"], "model.pt")),
                "train_receipt": expected_input_signature(
                    os.path.join(train_roots["43"], "train-receipt.json")
                ),
                "coordinate_receipt": expected_input_signature(
                    os.path.join(coordinate_roots["43"], "actual-transform.json")
                ),
                "panel": expected_input_signature(os.path.join(panel_roots["43"], "panel.json")),
                "render": expected_input_signature(
                    os.path.join(render_roots["43"], "render-manifest.json")
                ),
            },
            "44": {
                "model": expected_input_signature(os.path.join(train_roots["44"], "model.pt")),
                "train_receipt": expected_input_signature(
                    os.path.join(train_roots["44"], "train-receipt.json")
                ),
                "coordinate_receipt": expected_input_signature(
                    os.path.join(coordinate_roots["44"], "actual-transform.json")
                ),
                "panel": expected_input_signature(os.path.join(panel_roots["44"], "panel.json")),
                "render": expected_input_signature(
                    os.path.join(render_roots["44"], "render-manifest.json")
                ),
            },
        },
        "local_radius": {
            "definition": "seed42 self-excluded 15th nearest neighbor distance on fixed 50k sample",
            "radius_file": expected_input_signature(radius_path),
            "zero_radius_count": int((radius15 <= 0.0).sum()),
            "zero_radius_fraction": float((radius15 <= 0.0).mean()),
            "median": float(np.median(radius15)),
            "p10": float(np.quantile(radius15, 0.10)),
            "p90": float(np.quantile(radius15, 0.90)),
        },
        "pairs": pairs,
        "fixed_axis_render": expected_input_signature(render_path),
        "wall_seconds": round(time.monotonic() - started, 6),
    }
    receipt = _seal(body)
    path = os.path.join(output, "layout-disparity-v1.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed43-coordinates", required=True)
    parser.add_argument("--seed44-coordinates", required=True)
    parser.add_argument("--seed43-train", required=True)
    parser.add_argument("--seed44-train", required=True)
    parser.add_argument("--seed43-panel", required=True)
    parser.add_argument("--seed44-panel", required=True)
    parser.add_argument("--seed43-render", required=True)
    parser.add_argument("--seed44-render", required=True)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--implementation-base-commit", required=True)
    parser.add_argument("--sample-ids", default=R0019_SAMPLE_IDS)
    args = parser.parse_args(argv)
    receipt = run_layout_disparity(
        output_root=args.output_root,
        coordinate_roots={"43": args.seed43_coordinates, "44": args.seed44_coordinates},
        train_roots={"43": args.seed43_train, "44": args.seed44_train},
        panel_roots={"43": args.seed43_panel, "44": args.seed44_panel},
        render_roots={"43": args.seed43_render, "44": args.seed44_render},
        release_sha=args.release_sha,
        implementation_base_commit=args.implementation_base_commit,
        sample_ids_path=args.sample_ids,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
