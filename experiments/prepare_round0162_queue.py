#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0162 prompted-English staging."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import canonical_json, expected_input_signature, sha256_bytes
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory
from basemap.round0162_prompted_english_staging import (
    CAPABILITY,
    ROUND_ID,
    TOTAL_ROWS,
    VIEW_CAPABILITY,
    VIEW_ROWS,
    first_view,
    layout_identity,
    ordered_chunks,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0162"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0162-2026-08-03.md")
R0116_MANIFEST = (
    "/data/latent-basemap/runs/round-0116/queue/artifacts/"
    "jina-document-english-fineweb-rpj-5p727m-v1/"
    "jina-document-english-fineweb-rpj-5p727m-v1.json"
)
R0120_MANIFEST = (
    "/data/latent-basemap/runs/round-0120/queue/artifacts/"
    "jina-document-pile-english-3p399m-v1/"
    "jina-document-pile-english-3p399m-v1.json"
)
REVIEWS = (
    os.path.join(LAB_ROOT, "review-0116-2026-07-31.md"),
    os.path.join(LAB_ROOT, "review-0120-2026-07-31.md"),
)


def _read_sealed(path: str, *, expected_schema: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if (
        value.get("schema") != expected_schema
        or value.get("identity_sha256") != sha256_bytes(canonical_json(body))
    ):
        raise RuntimeError(f"R0162 accepted source manifest changed: {path}")
    return value, signature


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if frontmatter.get("status") != "issued" or frontmatter.get("base_commit") != release_sha:
        raise RuntimeError("R0162 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def _accepted_reviews() -> list[dict[str, Any]]:
    signatures = []
    for path, round_id in zip(REVIEWS, ("0116", "0120"), strict=True):
        frontmatter = _frontmatter(path)
        if frontmatter.get("status") != "accepted" or frontmatter.get("round_id") != round_id:
            raise RuntimeError(f"R0162 required Review {round_id} is not accepted")
        signatures.append(expected_input_signature(path))
    return signatures


def prepare_round0162(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0162 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews = _accepted_reviews()
    r0116, r0116_signature = _read_sealed(
        R0116_MANIFEST, expected_schema="jina-document-english-fineweb-rpj-5p727m-v1"
    )
    r0120, r0120_signature = _read_sealed(
        R0120_MANIFEST, expected_schema="jina-document-pile-english-3p399m-v1"
    )
    chunks = ordered_chunks(r0116, r0120)
    view = first_view(chunks)
    identity = layout_identity(
        r0116_signature=r0116_signature,
        r0120_signature=r0120_signature,
        chunks=chunks,
    )
    output = os.path.join(
        "/data/latent-basemap/canonical/jina-document-english-9p126m-v1",
        identity,
    )
    queue_root = create_fresh_directory(queue_root, label="R0162 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    expected_inputs = _dedupe([round_signature, *reviews, r0116_signature, r0120_signature])
    job = {
        "id": "stage_prompted_english_corpus",
        "action": "stage_prompted_english_corpus",
        "handler_module": "experiments.round0162_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "prompted-english-staging.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 900.0,
        "r0116_manifest": r0116_signature,
        "r0120_manifest": r0120_signature,
        "accepted_reviews": reviews,
        "layout_identity": identity,
        "node_policy": {"gpu_required": False, "training_performed": False, "cpu_heavy": True},
    }
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    queue.update({
        "schema": "round0162-prompted-english-staging-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0116", "0120"],
        "capability_dependencies": [
            "jina-document-english-fineweb-rpj-5p727m-v1",
            "jina-document-pile-english-3p399m-v1",
        ],
        "capabilities_produced": [CAPABILITY, VIEW_CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "what exact canonical prompted-English layout feeds the 8M scale rung?",
            "rows": TOTAL_ROWS,
            "first_view_rows": VIEW_ROWS,
            "source_order": list(view["source_order"]),
            "dataset_canonical_row_ranges": view["dataset_canonical_row_ranges"],
            "ordered_selection_sha256": view["ordered_selection_sha256"],
            "layout_identity": identity,
            "immutable_content_addressed_hardlinks": True,
            "symlinks": False,
            "source_manifests_preserved": True,
            "ordered_fp16_payload_sha256_required": True,
            "no_graph": True,
            "no_training": True,
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
    print(json.dumps({"queue_manifest": prepare_round0162(release_sha=args.release_sha, queue_root=args.queue_root)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
