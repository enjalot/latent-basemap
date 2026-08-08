"""Execute the R0215 v1 150M map forensic."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0215_v1_forensic import (
    CAPABILITY,
    CLUMP_DENSITY_PERCENTILE,
    COORD_TRIM_PERCENTILE,
    DIMENSION,
    EXACT_REFERENCE,
    FILAMENT_BACKGROUND_RATIO,
    FILAMENT_CORRIDOR_BINS,
    FILAMENT_SEGMENT_OCCUPANCY,
    FORENSIC_SCHEMA,
    GRAPH_K,
    HEATMAP_BINS,
    POPULATIONS,
    PREDICTIONS,
    ROUND_ID,
    ROWS,
    Round0215Error,
    SAMPLE_ROWS_PER_POPULATION,
    SAMPLE_SEED,
    SEGMENT_INTERIOR,
    TOP_CLUMPS_FOR_SEGMENTS,
    classify_summary,
    population_stats,
    verdict,
)
from basemap import round0113_prompt_contrast as prompt_contract

CORPUS_BLOCK = 2_000_000


def _sig(value: Any, *, label: str) -> dict[str, Any]:
    return dict(
        expected_input_signature(
            prompt_contract.verify_signature(value, label=label)
        )
    )


def _load_coordinates(job: Mapping[str, Any]) -> np.ndarray:
    chunks = list(job["coordinate_chunks"])
    out = np.empty((ROWS, 2), dtype=np.float32)
    at = 0
    for entry in chunks:
        path = prompt_contract.verify_signature(entry, label="v1 coordinate chunk")
        a = np.load(path, mmap_mode="r")
        out[at:at + len(a)] = np.asarray(a, dtype=np.float32)
        at += len(a)
    if at != ROWS or not np.isfinite(out).all():
        raise Round0215Error("R0215 v1 coordinates are incomplete or nonfinite")
    return out


def _classify(coords: np.ndarray) -> dict[str, Any]:
    """Bin the map and split it into clump / filament / density-matched field."""
    from scipy import ndimage

    lo_x, hi_x = np.percentile(coords[:, 0], list(COORD_TRIM_PERCENTILE))
    lo_y, hi_y = np.percentile(coords[:, 1], list(COORD_TRIM_PERCENTILE))
    H, xe, ye = np.histogram2d(
        coords[:, 0], coords[:, 1], bins=HEATMAP_BINS,
        range=[[lo_x, hi_x], [lo_y, hi_y]],
    )
    occupied = H[H > 0]
    background = float(np.median(occupied))
    clump_threshold = float(np.percentile(occupied, CLUMP_DENSITY_PERCENTILE))
    clump_mask = H >= clump_threshold
    labels, n_components = ndimage.label(clump_mask)
    if n_components == 0:
        raise Round0215Error("R0215 found no clump components")
    sizes = ndimage.sum(clump_mask, labels, range(1, n_components + 1))
    keep = np.argsort(sizes)[::-1][:TOP_CLUMPS_FOR_SEGMENTS] + 1
    centroids = np.asarray(
        ndimage.center_of_mass(clump_mask, labels, list(keep)), dtype=np.float64
    )

    filament_mask = np.zeros_like(clump_mask)
    segments: list[dict[str, Any]] = []
    lo_t, hi_t = SEGMENT_INTERIOR
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            p, q = centroids[i], centroids[j]
            t = np.linspace(lo_t, hi_t, 96)
            rr = np.clip((p[0] + (q[0] - p[0]) * t).astype(int), 0, HEATMAP_BINS - 1)
            cc = np.clip((p[1] + (q[1] - p[1]) * t).astype(int), 0, HEATMAP_BINS - 1)
            seg = H[rr, cc]
            occupancy = float((seg > 0).mean())
            median = float(np.median(seg))
            ratio = median / background if background else 0.0
            accepted = bool(
                occupancy >= FILAMENT_SEGMENT_OCCUPANCY
                and ratio >= FILAMENT_BACKGROUND_RATIO
            )
            segments.append({
                "i": int(i), "j": int(j), "occupancy": occupancy,
                "median_bin_count": median, "ratio_to_background": ratio,
                "accepted": accepted,
            })
            if not accepted:
                continue
            for dr in range(-FILAMENT_CORRIDOR_BINS, FILAMENT_CORRIDOR_BINS + 1):
                for dc in range(-FILAMENT_CORRIDOR_BINS, FILAMENT_CORRIDOR_BINS + 1):
                    filament_mask[
                        np.clip(rr + dr, 0, HEATMAP_BINS - 1),
                        np.clip(cc + dc, 0, HEATMAP_BINS - 1),
                    ] = True
    filament_mask &= ~clump_mask
    if not filament_mask.any():
        raise Round0215Error(
            "R0215 accepted no filament corridor; the selector calibrated on the "
            "probe did not reproduce on the full population"
        )

    # Field control, density-matched to the filament bins: same per-bin count
    # band, neither clump nor filament.
    fil_counts = H[filament_mask]
    band_lo, band_hi = float(fil_counts.min()), float(fil_counts.max())
    field_mask = (
        (H >= band_lo) & (H <= band_hi) & (~clump_mask) & (~filament_mask) & (H > 0)
    )
    if not field_mask.any():
        raise Round0215Error("R0215 found no density-matched field control bins")
    return {
        "H": H, "xe": xe, "ye": ye,
        "clump_mask": clump_mask, "filament_mask": filament_mask,
        "field_mask": field_mask,
        "background": background, "clump_threshold": clump_threshold,
        "clump_components": int(n_components),
        "segments": segments,
        "accepted_segments": int(sum(1 for s in segments if s["accepted"])),
        "field_band": [band_lo, band_hi],
    }


def _sample_rows(coords: np.ndarray, cls: Mapping[str, Any]) -> dict[str, np.ndarray]:
    xe, ye = cls["xe"], cls["ye"]
    bx = np.clip(np.digitize(coords[:, 0], xe) - 1, 0, HEATMAP_BINS - 1)
    by = np.clip(np.digitize(coords[:, 1], ye) - 1, 0, HEATMAP_BINS - 1)
    rng = np.random.RandomState(SAMPLE_SEED)
    out: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for population, mask in (
        ("clump", cls["clump_mask"]),
        ("filament", cls["filament_mask"]),
        ("field", cls["field_mask"]),
    ):
        member = mask[bx, by]
        rows = np.flatnonzero(member).astype(np.int64)
        counts[population] = int(rows.size)
        if rows.size == 0:
            continue
        take = min(SAMPLE_ROWS_PER_POPULATION, rows.size)
        out[population] = np.sort(rng.choice(rows, take, replace=False))
    classify_summary(counts)
    return out, counts


def _raw_graph_neighbours(job: Mapping[str, Any], rows: np.ndarray) -> np.ndarray:
    """Row i's k15 edges live at [i*k, (i+1)*k); verified, not assumed."""
    path = prompt_contract.verify_signature(job["v1_graph"], label="v1 k15 graph")
    with np.load(path, mmap_mode="r") as z:
        sources = z["sources"]
        targets = z["targets"]
        if len(sources) != ROWS * GRAPH_K:
            raise Round0215Error("R0215 v1 graph is not exactly k15 source-major")
        probe = rows[:: max(1, len(rows) // 32)][:32]
        for r in probe:
            block = np.asarray(sources[int(r) * GRAPH_K:(int(r) + 1) * GRAPH_K])
            if not np.all(block == int(r)):
                raise Round0215Error(
                    "R0215 v1 graph source-major layout assumption failed"
                )
        out = np.empty((len(rows), GRAPH_K), dtype=np.int64)
        for n, r in enumerate(rows):
            out[n] = np.asarray(
                targets[int(r) * GRAPH_K:(int(r) + 1) * GRAPH_K], dtype=np.int64
            )
    return out


def _exact_neighbours(job: Mapping[str, Any], rows: np.ndarray) -> np.ndarray:
    """Exact fp32 cosine top-k over the dequantized int8 corpus, one pass."""
    import torch

    i8_path = prompt_contract.verify_signature(job["int8_corpus"], label="int8 corpus")
    sc_path = prompt_contract.verify_signature(job["int8_scales"], label="int8 scales")
    corpus = np.memmap(i8_path, dtype=np.int8, mode="r", shape=(ROWS, DIMENSION))
    scales = np.memmap(sc_path, dtype="<f2", mode="r", shape=(ROWS,))
    device = torch.device("cuda")

    def dequant(lo: int, hi: int) -> "torch.Tensor":
        block = torch.from_numpy(np.ascontiguousarray(corpus[lo:hi])).to(
            device, non_blocking=True
        ).to(torch.float32)
        s = torch.from_numpy(
            np.ascontiguousarray(scales[lo:hi]).astype(np.float32)
        ).to(device)
        block *= s[:, None]
        return torch.nn.functional.normalize(block, dim=1)

    queries = torch.cat(
        [dequant(int(r), int(r) + 1) for r in rows], dim=0
    ).contiguous()
    best_sim = torch.full(
        (len(rows), GRAPH_K + 1), -float("inf"), device=device, dtype=torch.float32
    )
    best_idx = torch.full(
        (len(rows), GRAPH_K + 1), -1, device=device, dtype=torch.int64
    )
    for lo in range(0, ROWS, CORPUS_BLOCK):
        hi = min(lo + CORPUS_BLOCK, ROWS)
        sims = queries @ dequant(lo, hi).T
        k = min(GRAPH_K + 1, hi - lo)
        top_sim, top_loc = torch.topk(sims, k, dim=1)
        cat_sim = torch.cat([best_sim, top_sim], dim=1)
        cat_idx = torch.cat([best_idx, top_loc.to(torch.int64) + lo], dim=1)
        order = torch.argsort(cat_sim, dim=1, descending=True)[:, : GRAPH_K + 1]
        best_sim = torch.gather(cat_sim, 1, order)
        best_idx = torch.gather(cat_idx, 1, order)
        del sims, top_sim, top_loc, cat_sim, cat_idx, order
    exact = best_idx.cpu().numpy()
    # Drop self, keep k.
    out = np.empty((len(rows), GRAPH_K), dtype=np.int64)
    for n, r in enumerate(rows):
        keep = [int(v) for v in exact[n] if int(v) != int(r)][:GRAPH_K]
        if len(keep) < GRAPH_K:
            raise Round0215Error("R0215 exact search returned too few neighbours")
        out[n] = keep
    return out


def run_forensic(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0215Error("R0215 forensic handler received another queue")
    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0215 forensic")

    coords = _load_coordinates(job)
    cls = _classify(coords)
    samples, counts = _sample_rows(coords, cls)
    del coords

    census_path = prompt_contract.verify_signature(
        job["r0033_eligibility"], label="accepted R0033 eligibility census"
    )
    census = np.load(census_path)
    duplicate_members = np.zeros(ROWS, dtype=bool)
    duplicate_members[census["member_rows"].astype(np.int64)] = True
    excluded = np.zeros(ROWS, dtype=bool)
    excluded[census["excluded_rows"].astype(np.int64)] = True
    degrees = np.memmap(
        prompt_contract.verify_signature(job["canonical_degrees"], label="degrees"),
        dtype=np.uint8, mode="r", shape=(ROWS,),
    )

    edge_precision: dict[str, Any] = {}
    duplicate_rate: dict[str, float] = {}
    per_population: dict[str, Any] = {}
    for population in POPULATIONS:
        rows = samples[population]
        raw = _raw_graph_neighbours(job, rows)
        exact = _exact_neighbours(job, rows)
        precision = np.asarray(
            [len(set(raw[n].tolist()) & set(exact[n].tolist())) / GRAPH_K
             for n in range(len(rows))], dtype=np.float64
        )
        edge_precision[population] = population_stats(precision.tolist())
        duplicate_rate[population] = float(duplicate_members[rows].mean())
        deg = np.asarray(degrees[rows], dtype=np.int64)
        per_population[population] = {
            "sampled_rows": int(len(rows)),
            "population_rows_in_map": int(counts[population]),
            "edge_precision": edge_precision[population],
            "duplicate_family_membership_rate": duplicate_rate[population],
            "excluded_row_rate": float(excluded[rows].mean()),
            "canonical_degree": population_stats(deg.tolist()),
            "canonical_degree_zero_rate": float((deg == 0).mean()),
        }
        atomic_save_new_npz(
            os.path.join(output, f"{population}-sample.npz"),
            immutable=True, compressed=False,
            rows=rows, raw_neighbours=raw, exact_neighbours=exact,
            edge_precision=precision,
        )

    decision = verdict(edge_precision=edge_precision, duplicate_rate=duplicate_rate)
    receipt = prompt_contract.seal({
        "schema": FORENSIC_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "capabilities": [CAPABILITY],
        "map": "r0034-150m-seed42 (the v1 150M map)",
        "predictions": dict(PREDICTIONS),
        "selector": {
            "heatmap_bins": HEATMAP_BINS,
            "clump_density_percentile": CLUMP_DENSITY_PERCENTILE,
            "clump_threshold_count": cls["clump_threshold"],
            "background_median_bin": cls["background"],
            "filament_background_ratio": FILAMENT_BACKGROUND_RATIO,
            "filament_corridor_bins": FILAMENT_CORRIDOR_BINS,
            "clump_components": cls["clump_components"],
            "accepted_segments": cls["accepted_segments"],
            "total_segments": len(cls["segments"]),
            "segments": cls["segments"],
            "field_control": "density-matched to filament bins",
            "field_band_counts": cls["field_band"],
        },
        "populations": per_population,
        "verdict": decision,
        "exact_reference": EXACT_REFERENCE,
        "graph_k": GRAPH_K,
        "rows": ROWS,
        "training_performed": False,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "v1-forensic.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "forensic_v1_150m":
        raise Round0215Error("R0215 authorizes only the v1 150M forensic")
    run_forensic(active, job)


__all__ = ["run_job", "run_forensic"]
