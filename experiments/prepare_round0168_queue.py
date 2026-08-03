#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0168 prompted U12 staging."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0103_substrate import validate_inventory
from basemap.round0168_prompted_diverse_staging import (
    CAPABILITY,
    ROUND_ID,
    U12_ROWS,
    prompted_selection,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0168"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0168-2026-08-03.md")
SUBSET_MANIFEST = (
    "/data/latent-basemap/runs/round-0132/queue/artifacts/half-subset/"
    "subset-manifest.json"
)
U12_MAPPING = (
    "/data/latent-basemap/runs/round-0132/queue/artifacts/half-subset/"
    "compact-to-global.i64.npy"
)
PROMPTED_MANIFESTS = {
    "0116": "/data/latent-basemap/runs/round-0116/queue/artifacts/jina-document-english-fineweb-rpj-5p727m-v1/jina-document-english-fineweb-rpj-5p727m-v1.json",
    "0120": "/data/latent-basemap/runs/round-0120/queue/artifacts/jina-document-pile-english-3p399m-v1/jina-document-pile-english-3p399m-v1.json",
    "0126": "/data/latent-basemap/runs/round-0126/queue/artifacts/jina-document-multilingual-arb-ces-cmn-2p506m-v1/jina-document-multilingual-arb-ces-cmn-2p506m-v1.json",
    "0127": "/data/latent-basemap/runs/round-0127/queue/artifacts/jina-document-multilingual-deu-ell-fra-2p506m-v1/jina-document-multilingual-deu-ell-fra-2p506m-v1.json",
    "0139": "/data/latent-basemap/runs/round-0139/queue/artifacts/jina-document-multilingual-hin-ind-ita-2p506m-v1/jina-document-multilingual-hin-ind-ita-2p506m-v1.json",
    "0141": "/data/latent-basemap/runs/round-0141/queue/artifacts/jina-document-multilingual-jpn-kor-nld-2p506m-v1/jina-document-multilingual-jpn-kor-nld-2p506m-v1.json",
    "0143": "/data/latent-basemap/runs/round-0143/queue-refresh-20260802/artifacts/jina-document-multilingual-pol-por-rus-2p506m-v1/jina-document-multilingual-pol-por-rus-2p506m-v1.json",
    "0144": "/data/latent-basemap/runs/round-0144/queue-refresh-20260802/artifacts/jina-document-multilingual-spa-swe-tha-2p506m-v1/jina-document-multilingual-spa-swe-tha-2p506m-v1.json",
    "0145": "/data/latent-basemap/runs/round-0145/queue-refresh-20260802/artifacts/jina-document-multilingual-tur-vie-1p671m-v1/jina-document-multilingual-tur-vie-1p671m-v1.json",
}
REVIEWS = {
    "0087": "review-0087-2026-07-28.md",
    "0116": "review-0116-2026-07-31.md",
    "0120": "review-0120-2026-07-31.md",
    "0126": "review-0126-2026-07-31.md",
    "0127": "review-0127-2026-08-01.md",
    "0132": "review-0132-2026-08-01.md",
    "0139": "review-0139-2026-08-01.md",
    "0141": "review-0141-2026-08-02.md",
    "0143": "review-0143-2026-08-02.md",
    "0144": "review-0144-2026-08-02.md",
    "0145": "review-0145-2026-08-02.md",
}


def _read_sealed(path: str, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if not isinstance(value, dict) or value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise RuntimeError(f"{label} identity seal changed")
    return value, signature


def _accepted_reviews() -> list[dict[str, Any]]:
    signatures = []
    for round_id, filename in REVIEWS.items():
        path = os.path.join(LAB_ROOT, filename)
        frontmatter = _frontmatter(path)
        if frontmatter.get("round_id") != round_id or frontmatter.get("status") != "accepted":
            raise RuntimeError(f"R0168 required Review {round_id} is not accepted")
        signatures.append(expected_input_signature(path))
    return signatures


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if frontmatter.get("status") != "issued" or not descendant:
        raise RuntimeError("R0168 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def prepare_round0168(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0168 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews = _accepted_reviews()
    inventory = validate_inventory()
    prompt_values: dict[str, dict[str, Any]] = {}
    prompt_signatures: dict[str, dict[str, Any]] = {}
    for round_id, path in PROMPTED_MANIFESTS.items():
        value, signature = _read_sealed(path, label=f"accepted R{round_id} prompted tranche")
        prompt_values[round_id] = value
        prompt_signatures[round_id] = signature
    selection = prompted_selection(inventory["manifest"], prompt_values)

    subset, subset_signature = _read_sealed(SUBSET_MANIFEST, label="accepted R0132 subset")
    mapping_signature = expected_input_signature(U12_MAPPING)
    if (
        subset.get("round_id") != "0132"
        or int(subset.get("selected_rows", -1)) != U12_ROWS
        or subset.get("mapping") != mapping_signature
        or (subset.get("checks") or {}).get("mapping_strictly_increasing") is not True
    ):
        raise RuntimeError("accepted R0132 U12 population contract changed")
    mapping = np.load(U12_MAPPING, mmap_mode="r", allow_pickle=False)
    if mapping.dtype != np.dtype("<i8") or mapping.shape != (U12_ROWS,):
        raise RuntimeError("accepted R0132 U12 mapping geometry changed")
    mapping_hash = ordered_array_sha256(mapping)

    queue_root = create_fresh_directory(queue_root, label="R0168 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, "prompted-diverse-u12")
    expected_inputs = _dedupe([
        round_signature,
        *reviews,
        inventory["signature"],
        subset_signature,
        mapping_signature,
        *prompt_signatures.values(),
        *selection["source_signatures"],
    ])
    job = {
        "id": "stage_prompted_diverse_u12",
        "action": "stage_prompted_diverse_u12",
        "handler_module": "experiments.round0168_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "prompted-diverse-u12.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 1800.0,
        "r0087_inventory": inventory["signature"],
        "r0132_subset_manifest": subset_signature,
        "u12_mapping": mapping_signature,
        "u12_ordered_array_sha256": mapping_hash,
        "prompted_manifests": prompt_signatures,
        "ordered_selection_sha256": selection["ordered_selection_sha256"],
        "accepted_reviews": reviews,
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
        "schema": "round0168-prompted-diverse-u12-staging-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": list(REVIEWS),
        "capability_dependencies": [
            "jina-diverse-25m-inventory-v1",
            "jina-diverse-12p5m-map-registry-v1",
            "jina-document-20-language-corpus-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "what byte-exact prompted matrix represents the frozen R0132 U12 population?",
            "population_rows": U12_ROWS,
            "population_law": "exact accepted R0132 compact-to-global rows, ascending",
            "representation": "reviewed local Jina-v5 Document: fp16 tranches",
            "r0087_dataset_row_mapping_must_match_exactly": True,
            "polish_held_out": True,
            "ordered_selection_sha256": selection["ordered_selection_sha256"],
            "u12_ordered_array_sha256": mapping_hash,
            "contiguous_immutable_host_fp16_required": True,
            "duplicate_census": "diagnostic only; do not alter the exact U12 population",
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
    print(json.dumps({"queue_manifest": prepare_round0168(release_sha=args.release_sha, queue_root=args.queue_root)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
