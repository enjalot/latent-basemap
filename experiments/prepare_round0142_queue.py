#!/usr/bin/env python3
"""Materialize, but never launch, the R0142 Jina universality queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0112_prompt_substrate import (
    first2m_layout,
    model_member_signatures,
)
from basemap.round0114_prompt_recovery import source_chunk_path
from basemap.round0141_prompted_multilingual import environment_freeze_receipt
from basemap.round0142_jina_universality import (
    CAPABILITY,
    COMMON_CORPUS_ROWS,
    EMBED_MINIMUM_ROWS_PER_S,
    MAP_ORDER,
    PROBE_ORDER,
    ROUND_ID,
    seal,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _accepted_review, _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0142"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0142-*.md")

REVIEW_CAPABILITIES = {
    "0107": "jina-diverse-25m-full768-trained-map-seed42-v1",
    "0108": "jina-diverse-25m-map-registry-v1",
    "0114": "jina-fineweb-2m-dual-prompt-native8192-substrate-v2",
    "0132": "jina-diverse-12p5m-25m-scale-policy-geometry-v1",
}

MAPS = {
    "r0107-25m-seed42": (
        "/data/latent-basemap/runs/round-0107/queue/artifacts/"
        "train-diverse-jina-25m/model.pt"
    ),
    "r0132-12p5m-seed42": (
        "/data/latent-basemap/runs/round-0132/queue/artifacts/"
        "train-half-seed42/model.pt"
    ),
}
CONTROL = (
    "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-"
    "jina-v5-nano-heldout/train/data-00000.npy"
)
DADABASE = "/data/embeddings/dadabase/jina-v5-nano.npy"
DADABASE_TEXTS = "/data/embeddings/dadabase/jokes.parquet"
BEIR = {
    name: {
        "corpus": f"/data/embeddings/beir/{name}-pooled-jina-v5-nano/corpus_vectors.npy",
        "queries": f"/data/embeddings/beir/{name}-pooled-jina-v5-nano/query_vectors.npy",
        "corpus_ids": f"/data/embeddings/beir/{name}-pooled-jina-v5-nano/corpus_ids.json",
        "query_ids": f"/data/embeddings/beir/{name}-pooled-jina-v5-nano/query_ids.json",
    }
    for name in ("scifact", "trec-covid")
}

GPU_HOURS_MINIMUM = 0.45
GPU_HOURS_EXPECTED = 0.90
GPU_HOURS_P90 = 1.55
GPU_HOURS_MAXIMUM = 2.50


def _issued_round(release_sha: str) -> tuple[str, dict[str, Any]]:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0142 requires exactly one issued round; found {len(candidates)}"
        )
    if _frontmatter(candidates[0]).get("base_commit") != release_sha:
        raise RuntimeError("R0142 issued base_commit differs from release")
    return candidates[0], expected_input_signature(candidates[0])


def _common_sources() -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    values: dict[str, dict[str, Any]] = {}
    for name, rows in COMMON_CORPUS_ROWS.items():
        path = (
            f"/data/chunks/common-corpus-{name}-chunked-120/train/"
            "000_00000.parquet"
        )
        if int(pq.ParquetFile(path).metadata.num_rows) != rows:
            raise RuntimeError(f"Common Corpus {name} row count changed")
        values[name] = {
            "rows": rows,
            "selected_rows": min(rows, 50_000),
            "signature": expected_input_signature(path),
        }
    return values


def _canary_inputs() -> dict[str, Any]:
    import pyarrow.parquet as pq

    layout = first2m_layout()
    text_path = str(layout[0]["text_path"])
    raw_path = source_chunk_path("raw", 0)
    table = pq.read_table(
        text_path, columns=["chunk_token_count", "chunk_text"]
    ).slice(0, 25_000)
    counts = np.asarray(table.column("chunk_token_count").to_numpy(), dtype=np.int64)
    positions = np.flatnonzero(counts <= 400)[:32].astype(np.int64)
    if positions.shape != (32,):
        raise RuntimeError("R0142 could not select 32 short raw canary rows")
    return {
        "text": expected_input_signature(text_path),
        "raw": expected_input_signature(raw_path),
        "positions": positions.tolist(),
    }


def _cpu_smoke(
    maps: dict[str, dict[str, Any]], control: dict[str, Any]
) -> dict[str, Any]:
    """Exercise both reviewed model reload/transform paths with CUDA hidden."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("R0142 CPU smoke requires CUDA_VISIBLE_DEVICES='' or '-1'")
    from basemap.pumap.parametric_umap import ParametricUMAP

    source = np.load(control["canonical_path"], mmap_mode="r", allow_pickle=False)
    sample = np.asarray(source[:32], dtype=np.float32)
    cells: dict[str, Any] = {}
    for map_key, signature in maps.items():
        model = ParametricUMAP.load(signature["canonical_path"], device="cpu")
        coordinates = np.asarray(
            model.transform(sample, batch_size=16), dtype=np.float32
        )
        if coordinates.shape != (32, 2) or not np.isfinite(coordinates).all():
            raise RuntimeError(f"R0142 CPU smoke failed for {map_key}")
        cells[map_key] = {
            "model": signature,
            "rows": 32,
            "coordinates_finite": True,
            "coordinate_min": float(coordinates.min()),
            "coordinate_max": float(coordinates.max()),
        }
    return seal({
        "schema": "round0142-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scope": "both reviewed models reload -> transform real raw-Jina rows",
        "source": control,
        "cells": cells,
    })


def prepare_round0142(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0142 release SHA must be one full commit")
    round_file, round_signature = _issued_round(release_sha)
    reviews: list[dict[str, Any]] = []
    for round_id, capability in REVIEW_CAPABILITIES.items():
        reviews.extend(_accepted_review(round_id, capability))
    sources = _common_sources()
    canary = _canary_inputs()
    model_members = model_member_signatures()
    environment = environment_freeze_receipt()
    maps = {name: expected_input_signature(path) for name, path in MAPS.items()}
    if tuple(maps) != MAP_ORDER:
        raise RuntimeError("R0142 map order changed")
    control = expected_input_signature(CONTROL)
    dadabase = expected_input_signature(DADABASE)
    dadabase_texts = expected_input_signature(DADABASE_TEXTS)
    beir = {
        name: {key: expected_input_signature(path) for key, path in paths.items()}
        for name, paths in BEIR.items()
    }
    common_input_signatures = [item["signature"] for item in sources.values()]
    external_inputs = _dedupe([
        round_signature,
        *reviews,
        *model_members,
        canary["text"],
        canary["raw"],
        *common_input_signatures,
        *maps.values(),
        control,
        dadabase,
        dadabase_texts,
        *[
            signature
            for probe in beir.values()
            for signature in probe.values()
        ],
    ])

    queue_root = create_fresh_directory(queue_root, label="R0142 universality queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "cpu-smoke.json")
    atomic_write_new_json(smoke_path, _cpu_smoke(maps, control), immutable=True)
    smoke_signature = expected_input_signature(smoke_path)
    external_inputs = _dedupe([*external_inputs, smoke_signature])
    canary_output = os.path.join(artifacts, "raw-model-canary")
    jobs: list[dict[str, Any]] = [{
        "id": "raw_jina_model_canary",
        "action": "raw_model_canary",
        "handler_module": "experiments.round0142_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [canary_output],
        "done_marker": os.path.join(artifacts, "raw-model-canary.done.json"),
        "expected_inputs": _dedupe([
            round_signature,
            *reviews,
            *model_members,
            canary["text"],
            canary["raw"],
            smoke_signature,
        ]),
        "p90_wall_s": 180.0,
        "canary_text": canary["text"],
        "canary_raw": canary["raw"],
        "canary_positions": canary["positions"],
        "model_members": model_members,
        "environment_freeze": environment,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }]
    common_outputs: dict[str, str] = {}
    embed_ids: list[str] = []
    for name, source in sources.items():
        job_id = f"embed_common_{name}_raw_jina"
        output = os.path.join(artifacts, f"common-{name}-raw-jina")
        common_outputs[name] = output
        embed_ids.append(job_id)
        p90 = max(120.0, source["selected_rows"] / EMBED_MINIMUM_ROWS_PER_S + 90.0)
        jobs.append({
            "id": job_id,
            "action": "embed_common_corpus",
            "handler_module": "experiments.round0142_nodes",
            "handler_callable": "run_job",
            "deps": ["raw_jina_model_canary"],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{job_id}.done.json"),
            "expected_inputs": _dedupe([
                round_signature,
                *reviews,
                *model_members,
                source["signature"],
                canary["text"],
                canary["raw"],
                smoke_signature,
            ]),
            "p90_wall_s": p90,
            "probe": name,
            "source": source["signature"],
            "source_rows": source["rows"],
            "selected_rows": source["selected_rows"],
            "canary_output": canary_output,
            "model_members": model_members,
            "environment_freeze": environment,
            "node_policy": {"gpu_required": True, "training_performed": False},
        })

    map_outputs: dict[str, str] = {}
    score_ids: list[str] = []
    for map_key in MAP_ORDER:
        job_id = f"score_universality_{map_key}"
        output = os.path.join(artifacts, map_key)
        map_outputs[map_key] = output
        score_ids.append(job_id)
        jobs.append({
            "id": job_id,
            "action": "score_map",
            "handler_module": "experiments.round0142_nodes",
            "handler_callable": "run_job",
            "deps": embed_ids,
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{job_id}.done.json"),
            "expected_inputs": external_inputs,
            "p90_wall_s": 900.0,
            "map_key": map_key,
            "model": maps[map_key],
            "common_outputs": common_outputs,
            "control_embeddings": control,
            "dadabase": dadabase,
            "dadabase_texts": dadabase_texts,
            "beir": beir,
            "node_policy": {"gpu_required": True, "training_performed": False},
        })

    final_output = os.path.join(artifacts, CAPABILITY)
    jobs.append({
        "id": "assemble_retention_table",
        "action": "assemble",
        "handler_module": "experiments.round0142_nodes",
        "handler_callable": "run_job",
        "deps": score_ids,
        "outputs": [final_output],
        "done_marker": os.path.join(artifacts, "assemble-retention-table.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 120.0,
        "map_outputs": map_outputs,
        "node_policy": {"gpu_required": False, "training_performed": False},
    })

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0142-jina-universality-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-evaluation",
        "required_reviews": list(REVIEW_CAPABILITIES),
        "capability_dependencies": list(REVIEW_CAPABILITIES.values()),
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": jobs,
        "scientific_contract": {
            "question": "which OOD domains retain raw-Jina map neighborhoods?",
            "maps": maps,
            "probe_order": list(PROBE_ORDER),
            "common_corpus_source_rows": COMMON_CORPUS_ROWS,
            "common_corpus_selection": "first min(N, 50000) exact source rows",
            "common_corpus_embedding": {
                "model_members": model_members,
                "prompt_applied": False,
                "prompt_semantics": "raw/unprompted to match both maps",
                "canary": canary,
            },
            "duplicate_policy": (
                "canonicalize exact stored-vector families before splitting; "
                "corpus/query exact families are disjoint"
            ),
            "control": {
                "source": control,
                "policy": "canonical raw FineWeb-heldout, exactly shape matched per probe",
            },
            "metrics": ["probe FFR", "control FFR", "FFR retention", "recall10 retention"],
            "thresholds": {"pass_at_least": 0.70, "failure_below": 0.50},
            "diagnostic_only": True,
            "no_map_training": True,
            "no_atlas_gate_change": True,
            "no_prompt_transfer_claim": True,
            "no_production_or_publishing_claim": True,
            "environment_freeze": environment,
            "cpu_smoke": smoke_signature,
        },
    })
    gpu_p90 = {
        str(job["id"]): float(job["p90_wall_s"])
        for job in jobs
        if job["node_policy"]["gpu_required"]
    }
    queue["p90_gpu_seconds"] = {**gpu_p90, "total": sum(gpu_p90.values())}
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {"queue_manifest": prepare_round0142(
                release_sha=args.release_sha, queue_root=args.queue_root
            )},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
