#!/usr/bin/env python3
"""Prepare, but never launch, the R0167 prompted universality queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0112_prompt_substrate import first2m_layout, model_member_signatures
from basemap.round0114_prompt_recovery import source_chunk_path
from basemap.round0116_prompted_corpus import environment_freeze_receipt
from basemap.round0142_jina_universality import PROBE_ORDER
from basemap.round0167_prompted_universality import (
    CAPABILITY,
    EMBED_MINIMUM_ROWS_PER_S,
    PROMPTED_MAP_ORDER,
    ROUND_ID,
    control_rows_from_coordinate_archive,
    seal,
    source_rows_from_coordinate_archive,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _accepted_review, _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0167"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0167-2026-08-03.md")
R0142_ROOT = "/data/latent-basemap/runs/round-0142/queue/artifacts"
R0142_MAP = "r0107-25m-seed42"
R0142_TABLE = os.path.join(
    R0142_ROOT, "jina-diverse-universality-panel-v1", "retention-table.json"
)
R0146_PREDICTORS = (
    "/data/latent-basemap/runs/round-0146/queue/artifacts/"
    "jina-diverse-projection-loss-predictors-v1/projection-loss-predictors.json"
)
CONTROL_TEXT = (
    "/data/chunks/fineweb-edu-sample-10BT-chunked-500/heldout/"
    "data-00090-of-00099.parquet"
)
MAPS = {
    "r0115-prompted-2m-seed42": (
        "/data/latent-basemap/runs/round-0115/queue-attempt-2/artifacts/"
        "document/train/model.pt"
    ),
    "r0117-prompted-2m-seed43": (
        "/data/latent-basemap/runs/round-0117/queue/artifacts/document/train/model.pt"
    ),
    "r0166-prompted-8m-seed42": (
        "/data/latent-basemap/runs/round-0166/queue/artifacts/seed42-train/model.pt"
    ),
}
REVIEW_CAPABILITIES = {
    "0115": "jina-fineweb-2m-prompt-map-contrast-v1",
    "0117": "jina-fineweb-2m-prompt-map-seed43-contrast-v1",
    "0142": "jina-diverse-universality-panel-v1",
    "0146": "jina-diverse-projection-loss-predictors-v1",
}

GPU_HOURS_MINIMUM = 0.20
GPU_HOURS_EXPECTED = 0.40
GPU_HOURS_MAXIMUM = 2.00
Q2_ROUND_ID = "0166"
Q2_CAPABILITY: str | None = "jina-document-english-8m-prompted-map-seed42-v1"
Q2_MAP_ROLE = "accepted positive Q2 map capability"
TRAINING_AUDIT_PATHS: dict[str, tuple[str, int]] = {}
TRAINING_AUDIT_POLICY = "not requested"
HANDLER_MODULE = "experiments.round0167_nodes"
QUEUE_SCHEMA = "round0167-prompted-universality-queue-v1"
QUEUE_LABEL = "R0167 prompted universality queue"


def _one_document(prefix: str, round_id: str, *, status: str) -> dict[str, Any]:
    paths = [
        path
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, f"{prefix}-{round_id}-*.md")))
        if _frontmatter(path).get("status") == status
    ]
    if len(paths) != 1:
        raise RuntimeError(
            f"R0167 requires one {status} {prefix} for R{round_id}; found {len(paths)}"
        )
    return expected_input_signature(paths[0])


def _accepted_any_review(round_id: str) -> list[dict[str, Any]]:
    review = _one_document("review", round_id, status="accepted")
    frontmatter = _frontmatter(review["canonical_path"])
    result_path = os.path.join(LAB_ROOT, frontmatter.get("result") or "")
    round_path = os.path.join(LAB_ROOT, frontmatter.get("round") or "")
    result = expected_input_signature(result_path)
    issued = expected_input_signature(round_path)
    if (
        result["sha256"] != frontmatter.get("result_sha256")
        or issued["sha256"] != frontmatter.get("round_sha256")
    ):
        raise RuntimeError(f"Review {round_id} bindings changed")
    if Q2_CAPABILITY is not None:
        releases = frontmatter.get("releases") or []
        expected_release = f"capability:{Q2_CAPABILITY}"
        if not isinstance(releases, list) or expected_release not in releases:
            raise RuntimeError(
                f"Review {round_id} did not release required {expected_release}"
            )
    return [issued, result, review]


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if frontmatter.get("status") != "issued" or not descendant:
        raise RuntimeError("R0167 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def _unique_arrow(dataset: str, split: str) -> str:
    paths = sorted(glob.glob(f"/data/hf/datasets/mteb___{dataset}/{split}/**/*.arrow", recursive=True))
    if len(paths) != 1:
        raise RuntimeError(f"R0167 requires one {dataset} {split} Arrow; found {len(paths)}")
    return paths[0]


def _coordinate_signature(name: str, *, control: bool = False) -> dict[str, Any]:
    suffix = "__fineweb-control" if control else ""
    return expected_input_signature(
        os.path.join(R0142_ROOT, R0142_MAP, f"{name}{suffix}-coordinates.npz")
    )


def _selection_rows(
    signature: Mapping[str, Any],
    *,
    label: str,
    control: bool = False,
    separate_sources: bool = False,
) -> tuple[int, int]:
    with np.load(signature["canonical_path"], allow_pickle=False) as archive:
        corpus_ids = np.asarray(archive["probe_corpus_ids"], dtype=np.int64)
        query_ids = np.asarray(archive["probe_query_ids"], dtype=np.int64)
    if control:
        corpus, queries = control_rows_from_coordinate_archive(corpus_ids, query_ids, label=label)
    else:
        corpus, queries = source_rows_from_coordinate_archive(
            corpus_ids,
            query_ids,
            label=label,
            separate_sources=separate_sources,
        )
    return len(corpus), len(queries)


def _source_specs() -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for name in PROBE_ORDER[:8]:
        values[name] = {
            "source_kind": "common-parquet",
            "text_source": expected_input_signature(
                f"/data/chunks/common-corpus-{name}-chunked-120/train/000_00000.parquet"
            ),
        }
    values["dadabase"] = {
        "source_kind": "dadabase-parquet",
        "text_source": expected_input_signature("/data/embeddings/dadabase/jokes.parquet"),
    }
    for name in ("scifact", "trec-covid"):
        root = f"/data/embeddings/beir/{name}-pooled-jina-v5-nano"
        values[name] = {
            "source_kind": "beir-arrow",
            "corpus_text_source": expected_input_signature(_unique_arrow(name, "corpus")),
            "query_text_source": expected_input_signature(_unique_arrow(name, "queries")),
            "corpus_ids": expected_input_signature(os.path.join(root, "corpus_ids.json")),
            "query_ids": expected_input_signature(os.path.join(root, "query_ids.json")),
        }
    if set(values) != set(PROBE_ORDER):
        raise RuntimeError("R0167 probe source inventory changed")
    return values


def _canary_inputs() -> dict[str, Any]:
    import pyarrow.parquet as pq

    text_path = str(first2m_layout()[0]["text_path"])
    table = pq.read_table(text_path, columns=["chunk_token_count"]).slice(0, 25_000)
    counts = np.asarray(table.column("chunk_token_count").to_numpy(), dtype=np.int64)
    positions = np.flatnonzero(counts <= 400)[:32].astype(np.int64)
    if positions.shape != (32,):
        raise RuntimeError("R0167 could not select 32 prompt-canary rows")
    return {
        "text": expected_input_signature(text_path),
        "document": expected_input_signature(source_chunk_path("document", 0)),
        "positions": positions.tolist(),
    }


def _release_cpu_smoke(release_sha: str, maps: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0167 release checkout differs from requested release")
    started = time.monotonic()
    from basemap.pumap.parametric_umap import ParametricUMAP

    source = np.memmap(
        "/data/latent-basemap/runs/round-0113/queue/artifacts/compact-arrays/document-compact.f16",
        dtype="<f2",
        mode="r",
        shape=(1_993_761, 768),
    )
    sample = np.asarray(source[:32], dtype=np.float32)
    cells: dict[str, Any] = {}
    for map_key, signature in maps.items():
        model = ParametricUMAP.load(signature["canonical_path"], device="cpu")
        coordinates = np.asarray(model.transform(sample, batch_size=16), dtype=np.float32)
        if coordinates.shape != (32, 2) or not np.isfinite(coordinates).all():
            raise RuntimeError(f"R0167 CPU smoke failed for {map_key}")
        cells[map_key] = {
            "model": dict(signature),
            "coordinates_finite": True,
            "coordinate_minimum": float(coordinates.min()),
            "coordinate_maximum": float(coordinates.max()),
        }
    return seal({
        "schema": f"round{ROUND_ID}-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scope": "all three prompted models reload and transform real prompted rows",
        "cells": cells,
        "wall_seconds": time.monotonic() - started,
    })


def prepare_round0167(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0167 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews: list[dict[str, Any]] = []
    for round_id, capability in REVIEW_CAPABILITIES.items():
        reviews.extend(_accepted_review(round_id, capability))
    reviews.extend(_accepted_any_review(Q2_ROUND_ID))

    maps = {key: expected_input_signature(path) for key, path in MAPS.items()}
    if tuple(maps) != PROMPTED_MAP_ORDER:
        raise RuntimeError("R0167 prompted map order changed")
    training_audit_sources = {
        label: {
            "signature": expected_input_signature(path),
            "rows": int(rows),
        }
        for label, (path, rows) in TRAINING_AUDIT_PATHS.items()
    }
    for label, source in training_audit_sources.items():
        expected_bytes = int(source["rows"]) * 768 * 2
        if source["signature"]["bytes"] != expected_bytes:
            raise RuntimeError(
                f"R0167 {label} training-audit source size changed"
            )
    sources = _source_specs()
    model_members = model_member_signatures()
    environment = environment_freeze_receipt()
    canary = _canary_inputs()
    raw_table = expected_input_signature(R0142_TABLE)
    raw_predictors = expected_input_signature(R0146_PREDICTORS)
    control_text = expected_input_signature(CONTROL_TEXT)
    probe_coordinates = {name: _coordinate_signature(name) for name in PROBE_ORDER}
    control_coordinates = {
        name: _coordinate_signature(name, control=True) for name in PROBE_ORDER
    }

    external_inputs = _dedupe([
        round_signature,
        *reviews,
        *maps.values(),
        *[source["signature"] for source in training_audit_sources.values()],
        *model_members,
        canary["text"],
        canary["document"],
        raw_table,
        raw_predictors,
        control_text,
        *probe_coordinates.values(),
        *control_coordinates.values(),
        *[
            signature
            for source in sources.values()
            for signature in source.values()
            if isinstance(signature, Mapping)
        ],
    ])

    queue_root = create_fresh_directory(queue_root, label=QUEUE_LABEL)
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha, maps), immutable=True)
    smoke_signature = expected_input_signature(smoke_path)
    external_inputs = _dedupe([*external_inputs, smoke_signature])
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    canary_output = os.path.join(artifacts, "prompt-model-canary")
    jobs: list[dict[str, Any]] = [{
        "id": "prompt_model_canary",
        "action": "prompt_canary",
        "handler_module": HANDLER_MODULE,
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [canary_output],
        "done_marker": os.path.join(artifacts, "prompt-model-canary.done.json"),
        "expected_inputs": _dedupe([
            round_signature, *reviews, *model_members, canary["text"], canary["document"], smoke_signature
        ]),
        "p90_wall_s": 180.0,
        "canary_text": canary["text"],
        "canary_document": canary["document"],
        "canary_positions": canary["positions"],
        "model_members": model_members,
        "environment_freeze": environment,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }]

    probe_outputs: dict[str, str] = {}
    embed_ids: list[str] = []
    for name in PROBE_ORDER:
        corpus_rows, query_rows = _selection_rows(
            probe_coordinates[name],
            label=name,
            separate_sources=sources[name]["source_kind"] == "beir-arrow",
        )
        job_id = f"embed_prompted_{name.replace('-', '_')}"
        output = os.path.join(artifacts, f"prompted-{name}")
        probe_outputs[name] = output
        embed_ids.append(job_id)
        source = sources[name]
        source_signatures = [value for value in source.values() if isinstance(value, Mapping)]
        jobs.append({
            "id": job_id,
            "action": "embed_probe",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["prompt_model_canary"],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{job_id}.done.json"),
            "expected_inputs": _dedupe([
                round_signature, *reviews, *model_members, probe_coordinates[name], *source_signatures, smoke_signature
            ]),
            "p90_wall_s": max(120.0, (corpus_rows + query_rows) / EMBED_MINIMUM_ROWS_PER_S + 90.0),
            "probe": name,
            "r0142_coordinates": probe_coordinates[name],
            **source,
            "canary_output": canary_output,
            "model_members": model_members,
            "environment_freeze": environment,
            "node_policy": {"gpu_required": True, "training_performed": False},
        })

    control_output = os.path.join(artifacts, "prompted-fineweb-control")
    control_job_id = "embed_prompted_fineweb_control"
    jobs.append({
        "id": control_job_id,
        "action": "embed_control",
        "handler_module": HANDLER_MODULE,
        "handler_callable": "run_job",
        "deps": ["prompt_model_canary"],
        "outputs": [control_output],
        "done_marker": os.path.join(artifacts, "prompted-fineweb-control.done.json"),
        "expected_inputs": _dedupe([
            round_signature, *reviews, *model_members, control_text, smoke_signature
        ]),
        "p90_wall_s": 60_000 / EMBED_MINIMUM_ROWS_PER_S + 90.0,
        "text_source": control_text,
        "canary_output": canary_output,
        "model_members": model_members,
        "environment_freeze": environment,
        "node_policy": {"gpu_required": True, "training_performed": False},
    })

    audit_output: str | None = None
    audit_job_id: str | None = None
    if training_audit_sources:
        audit_job_id = "audit_prompted_rows_against_map_training"
        audit_output = os.path.join(artifacts, "prompted-training-disjoint-audit")
        jobs.append({
            "id": audit_job_id,
            "action": "audit_training_disjoint",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": [*embed_ids, control_job_id],
            "outputs": [audit_output],
            "done_marker": os.path.join(
                artifacts, "prompted-training-disjoint-audit.done.json"
            ),
            "expected_inputs": external_inputs,
            "p90_wall_s": 1_800.0,
            "training_sources": training_audit_sources,
            "probe_outputs": probe_outputs,
            "control_output": control_output,
            "control_coordinates": control_coordinates,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": True,
            },
        })

    map_outputs: dict[str, str] = {}
    score_ids: list[str] = []
    for map_key in PROMPTED_MAP_ORDER:
        job_id = f"score_{map_key}"
        output = os.path.join(artifacts, map_key)
        map_outputs[map_key] = output
        score_ids.append(job_id)
        jobs.append({
            "id": job_id,
            "action": "score_map",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": [
                *embed_ids,
                control_job_id,
                *([audit_job_id] if audit_job_id is not None else []),
            ],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{job_id}.done.json"),
            "expected_inputs": external_inputs,
            "p90_wall_s": 900.0,
            "map_key": map_key,
            "model": maps[map_key],
            "probe_outputs": probe_outputs,
            "control_output": control_output,
            "control_coordinates": control_coordinates,
            "node_policy": {"gpu_required": True, "training_performed": False},
        })

    final_output = os.path.join(artifacts, CAPABILITY)
    jobs.append({
        "id": "assemble_prompted_universality",
        "action": "assemble",
        "handler_module": HANDLER_MODULE,
        "handler_callable": "run_job",
        "deps": score_ids,
        "outputs": [final_output],
        "done_marker": os.path.join(artifacts, "assemble-prompted-universality.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 1_800.0,
        "map_outputs": map_outputs,
        "probe_outputs": probe_outputs,
        "raw_retention_table": raw_table,
        "raw_predictors": raw_predictors,
        **(
            {"training_disjoint_audit": audit_output}
            if audit_output is not None
            else {}
        ),
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
    })

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": QUEUE_SCHEMA,
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-evaluation",
        "required_reviews": [*REVIEW_CAPABILITIES, Q2_ROUND_ID],
        "ordering_dependencies": [Q2_ROUND_ID],
        "capability_dependencies": [
            *REVIEW_CAPABILITIES.values(),
            *([Q2_CAPABILITY] if Q2_CAPABILITY is not None else []),
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            **{
                str(job["id"]): float(job["p90_wall_s"])
                for job in jobs
                if job["node_policy"]["gpu_required"]
            },
            "total": sum(
                float(job["p90_wall_s"])
                for job in jobs
                if job["node_policy"]["gpu_required"]
            ),
        },
        "scientific_contract": {
            "question": "does prompted OOD retention and its TwoNN association persist at 2M and 8M?",
            "maps": list(PROMPTED_MAP_ORDER),
            "probes": list(PROBE_ORDER),
            "source_selection": "exact accepted R0142 corpus/query row IDs",
            "embedding_convention": "literal Document: prefix on every probe and control text",
            "control": "exact R0142 per-probe row selection re-embedded from first 60000 heldout FineWeb texts",
            "metrics": ["probe FFR", "control FFR", "FFR retention", "recall10 retention"],
            "twonn": "R0146 exact 2048-row estimator recomputed in prompted geometry",
            "raw_comparison": "accepted R0142/R0146, descriptive because map scales differ",
            "q2_map_evidence_role": Q2_MAP_ROLE,
            "training_overlap_audit": TRAINING_AUDIT_POLICY,
            "diagnostic_only": True,
            "no_causal_prompt_claim": True,
            "no_universal_map_claim": True,
            "no_quality_gate_change": True,
            "no_training": True,
            "release_cpu_smoke": smoke_signature,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0167(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
