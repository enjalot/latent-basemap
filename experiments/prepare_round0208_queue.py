#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0208 prompted OOD reserve repair."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0108_evaluation import IN_MIX_LANGUAGES, POLISH
from basemap.round0168_prompted_diverse_staging import (
    CAPABILITY as STAGING_CAPABILITY,
)
from basemap.round0208_prompted_ood_repair import (
    CAPABILITY,
    RETAINED_CORPUS_ROWS,
    RETAINED_QUERY_ROWS,
    ROUND_ID,
    SOURCE_PACK_ROOT,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0191_queue import _embedded_signatures


LANGUAGES = (*IN_MIX_LANGUAGES, POLISH)
ROUND_ROOT = "/data/latent-basemap/runs/round-0208"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0208-2026-08-07.md")
R0173_REVIEW = os.path.join(LAB_ROOT, "review-0173-2026-08-03.md")
R0168_REVIEW = os.path.join(LAB_ROOT, "review-0168-2026-08-03-01.md")
STAGING_MANIFEST = (
    "/data/latent-basemap/runs/round-0168/queue/artifacts/prompted-diverse-u12/"
    "prompted-u12-manifest.json"
)
R0173_AUDIT = os.path.join(
    SOURCE_PACK_ROOT, "jina-prompted-u12-ood-probe-pack-v1", "audit.json"
)
R0173_CANARY = os.path.join(SOURCE_PACK_ROOT, "prompt-model-canary", "canary.json")
R0132_MAPPING = (
    "/data/latent-basemap/runs/round-0132/queue/artifacts/half-subset/"
    "compact-to-global.i64.npy"
)
R0087_INVENTORY = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/jina-diverse-25m-inventory/"
    "jina-diverse-25m-inventory-v1.json"
)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0208 round is not issued for this exact release")
    return expected_input_signature(ROUND_FILE)


def _accepted_review(round_id: str, review_path: str) -> list[dict[str, Any]]:
    """Bind an accepted review and the round/result bytes it verified.

    R0173's queue reached a registered *failed* terminal state, so this helper
    deliberately does not require a ``succeeded`` verdict; it requires an
    accepted review whose round/result hashes still reproduce, plus the queue
    manifest and terminal receipt that review verified.
    """
    review = _frontmatter(review_path)
    if review.get("round_id") != round_id or review.get("status") != "accepted":
        raise RuntimeError(f"R{round_id} review is not accepted")
    round_signature = expected_input_signature(
        os.path.join(LAB_ROOT, str(review.get("round") or ""))
    )
    result_signature = expected_input_signature(
        os.path.join(LAB_ROOT, str(review.get("result") or ""))
    )
    if (
        review.get("round_sha256") != round_signature["sha256"]
        or review.get("result_sha256") != result_signature["sha256"]
    ):
        raise RuntimeError(f"R{round_id} review binding changed")
    signatures = [
        expected_input_signature(review_path),
        round_signature,
        result_signature,
    ]
    for name in ("queue/queue.json", "queue/runner-terminal.json"):
        path = f"/data/latent-basemap/runs/round-{round_id}/{name}"
        if os.path.exists(path):
            signatures.append(expected_input_signature(path))
    return signatures


def _probe_inputs() -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for language in LANGUAGES:
        receipt = os.path.join(SOURCE_PACK_ROOT, f"prompted-{language}", "receipt.json")
        signatures.append(expected_input_signature(receipt))
        with open(receipt, encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"R0173 {language} probe receipt changed")
        for key in (
            "corpus_embeddings",
            "query_embeddings",
            "corpus_source_rows",
            "query_source_rows",
        ):
            _embedded_signatures(value[key], signatures)
    return signatures


def _accepted_inputs() -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = [
        *_accepted_review("0173", R0173_REVIEW),
        *_accepted_review("0168", R0168_REVIEW),
        expected_input_signature(STAGING_MANIFEST),
        expected_input_signature(R0173_AUDIT),
        expected_input_signature(R0173_CANARY),
        expected_input_signature(R0132_MAPPING),
        expected_input_signature(R0087_INVENTORY),
        *_probe_inputs(),
    ]
    with open(STAGING_MANIFEST, encoding="utf-8") as handle:
        staging = json.load(handle)
    _embedded_signatures(staging["host_fp16"], signatures)
    return _dedupe(signatures)


def prepare_round0208(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0208 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    expected_inputs = _dedupe([round_signature, *_accepted_inputs()])
    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0208 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "repair_prompted_ood_pack",
        "action": "repair_prompted_ood_pack",
        "handler_module": "experiments.round0208_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "prompted-ood-repair.done.json"),
        "expected_inputs": expected_inputs,
        "source_pack_root": SOURCE_PACK_ROOT,
        "staging_manifest": expected_input_signature(STAGING_MANIFEST),
        "r0173_audit": expected_input_signature(R0173_AUDIT),
        "r0173_canary": expected_input_signature(R0173_CANARY),
        "r0132_mapping": expected_input_signature(R0132_MAPPING),
        "r0087_inventory": expected_input_signature(R0087_INVENTORY),
        "p90_wall_s": 300.0,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
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
        "schema": "round0208-prompted-ood-repair-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0168", "0173"],
        "capability_dependencies": [STAGING_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "population": "exact accepted R0168 12,474,331-row prompted U12 matrix",
            "probe_source": "exact immutable R0173 prompted language arrays",
            "embedding_performed": False,
            "rows_reselected": 0,
            "repair_policy": "removal-only explicit exclusion; no reselection, no re-embedding",
            "identities": [
                "complete stored prompted-fp16 row bytes versus U12",
                "complete stored prompted-fp16 row bytes within the pack",
                "exact per-language source-row membership of U12",
            ],
            "retained_corpus_rows_per_language": RETAINED_CORPUS_ROWS,
            "retained_query_rows_per_language": RETAINED_QUERY_ROWS,
            "query_ids_unchanged_from_r0173": True,
            "held_out_language": POLISH,
            "training_performed": False,
            "production_or_publishing": False,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0208(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
