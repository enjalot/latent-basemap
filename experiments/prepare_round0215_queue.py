#!/usr/bin/env python3
"""Prepare, but never launch, the R0215 v1 150M map forensic."""
from __future__ import annotations

import argparse, glob, json, os, re, subprocess, sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json, create_fresh_directory, ensure_data_directory,
)
from basemap.round0215_v1_forensic import (
    CAPABILITY, CLUMP_DENSITY_PERCENTILE, EXACT_REFERENCE,
    FILAMENT_BACKGROUND_RATIO, GRAPH_K, HEATMAP_BINS, POPULATIONS,
    ROUND_ID, ROWS, SAMPLE_ROWS_PER_POPULATION,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter

ROUND_ROOT = "/data/latent-basemap/runs/round-0215"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0215-2026-08-08.md")
COORD_DIR = "/data/latent-basemap/runs/round-0036/queue/artifacts/coordinates"
V1_GRAPH = "/data/checkpoints/pumap/edges_150m_k15.npz"
R0033_ELIGIBILITY = (
    "/data/latent-basemap/runs/round-0033/queue/artifacts/eligibility/"
    "minilm-150m-row-eligibility-v1.npz"
)
CANONICAL_DEGREES = (
    "/data/latent-basemap/runs/round-0034/preflight/canonical-graph-v1/valid-degrees.u8"
)
INT8_CORPUS = (
    "/data/latent-basemap/runs/round-0025/queue/artifacts/int8-shards/"
    "minilm-int8-150m/embeddings.i8"
)
INT8_SCALES_GLOB = (
    "/data/latent-basemap/runs/round-0025/queue/artifacts/int8-shards/"
    "minilm-int8-150m/*scale*"
)
GPU_HOURS_CAP = 1.5


def _issued_round(release_sha: str) -> dict[str, Any]:
    fm = _frontmatter(ROUND_FILE)
    base = str(fm.get("base_commit") or "")
    ok = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base, release_sha],
        check=False, timeout=10,
    ).returncode == 0
    if fm.get("round_id") != ROUND_ID or fm.get("status") != "issued" or not ok:
        raise RuntimeError("R0215 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def prepare_round0215(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0215 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    chunks = sorted(glob.glob(os.path.join(COORD_DIR, "chunk-*/coordinates.npy")))
    if len(chunks) != 30:
        raise RuntimeError(f"R0215 expected 30 coordinate chunks, found {len(chunks)}")
    coordinate_chunks = [expected_input_signature(p) for p in chunks]
    scales = sorted(glob.glob(INT8_SCALES_GLOB))
    if len(scales) != 1:
        raise RuntimeError(f"R0215 expected one int8 scale array, found {len(scales)}")
    primaries = {
        "v1_graph": expected_input_signature(V1_GRAPH),
        "r0033_eligibility": expected_input_signature(R0033_ELIGIBILITY),
        "canonical_degrees": expected_input_signature(CANONICAL_DEGREES),
        "int8_corpus": expected_input_signature(INT8_CORPUS),
        "int8_scales": expected_input_signature(scales[0]),
        "transform_receipt": expected_input_signature(
            os.path.join(COORD_DIR, "actual-transform.json")
        ),
    }
    expected_inputs = _dedupe(
        [round_signature, *coordinate_chunks, *primaries.values()]
    )
    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0215 GPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    job = {
        "id": "forensic_v1_150m",
        "action": "forensic_v1_150m",
        "handler_module": "experiments.round0215_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, CAPABILITY)],
        "done_marker": os.path.join(artifacts, "v1-forensic.done.json"),
        "expected_inputs": expected_inputs,
        "coordinate_chunks": coordinate_chunks,
        **primaries,
        "p90_wall_s": 3_600.0,
        "node_policy": {"gpu_required": True, "training_performed": False, "cpu_heavy": True},
    }
    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0215-v1-150m-forensic-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-evaluation",
        "required_reviews": [],
        "capability_dependencies": [],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "p90_gpu_seconds": {"forensic_v1_150m": 3_600.0, "total": 3_600.0},
        "scientific_contract": {
            "map": "r0034-150m-seed42 (v1 150M)",
            "rows": ROWS,
            "graph_k": GRAPH_K,
            "heatmap_bins": HEATMAP_BINS,
            "clump_density_percentile": CLUMP_DENSITY_PERCENTILE,
            "filament_background_ratio": FILAMENT_BACKGROUND_RATIO,
            "populations": list(POPULATIONS),
            "field_control": "density-matched to filament bins",
            "sample_rows_per_population": SAMPLE_ROWS_PER_POPULATION,
            "exact_reference": EXACT_REFERENCE,
            "selector_fixed_before_measurement": True,
            "training_performed": False,
            "production_or_publishing": False,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-sha", required=True)
    ap.add_argument("--queue-root", default=QUEUE_ROOT)
    a = ap.parse_args(argv)
    print(json.dumps({"queue_manifest": prepare_round0215(
        release_sha=a.release_sha, queue_root=a.queue_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
