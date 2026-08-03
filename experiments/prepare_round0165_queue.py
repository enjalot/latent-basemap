#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0165 frozen-prefix derivation."""
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

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0165_frozen_prefix_population import (
    CAPABILITY,
    HOST_CAPABILITY,
    PREFIX_STOP,
    ROUND_ID,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0163_queue import (
    LAYOUT_PATH,
    R0113_MAPPING,
    _dedupe_signatures,
)
from experiments.round0163_nodes import _read_sealed, _source_layouts


ROUND_ROOT = "/data/latent-basemap/runs/round-0165"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0165-2026-08-03.md")
RESULT_0164 = os.path.join(LAB_ROOT, "result-0164-2026-08-03.md")
REVIEWS = {
    "0113": os.path.join(LAB_ROOT, "review-0113-2026-07-30.md"),
    "0163": os.path.join(LAB_ROOT, "review-0163-2026-08-03.md"),
    "0164": os.path.join(LAB_ROOT, "review-0164-2026-08-03.md"),
}
R0163_MAPPING = (
    "/data/latent-basemap/runs/round-0163/queue-correction-1/artifacts/"
    "prompted-english-8m-representatives/compact-to-canonical.i64.npy"
)
R0164_ROOT = (
    "/data/latent-basemap/runs/round-0164/queue/artifacts/"
    "prompted-english-8m-population-v2"
)
R0164_POPULATION = os.path.join(R0164_ROOT, "prompted-population.json")


def _accepted_review(round_id: str) -> dict[str, Any]:
    path = REVIEWS[round_id]
    frontmatter = _frontmatter(path)
    if frontmatter.get("status") != "accepted" or frontmatter.get("round_id") != round_id:
        raise RuntimeError(f"R0165 required Review {round_id} is not accepted")
    return expected_input_signature(path)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if frontmatter.get("status") != "issued" or not descendant:
        raise RuntimeError("R0165 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def _touched_text_inputs(
    text_layout: list[Mapping[str, Any]], dropped: np.ndarray
) -> list[dict[str, Any]]:
    selected = []
    for item in text_layout:
        left = int(np.searchsorted(dropped, int(item["canonical_start"]), side="left"))
        right = int(np.searchsorted(dropped, int(item["canonical_stop"]), side="left"))
        if right > left:
            selected.append(dict(item["text"]))
    return _dedupe_signatures(selected)


def prepare_round0165(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0165 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews = [_accepted_review(round_id) for round_id in ("0113", "0163", "0164")]
    result0164 = expected_input_signature(RESULT_0164)
    population0164 = expected_input_signature(R0164_POPULATION)
    prior = _read_sealed(population0164, label="accepted R0164 population")
    mapping0164 = dict(prior["mapping"])
    compact0164 = dict(prior["document_compact"])
    hashes0164 = dict(prior["source_text_hash_index"])
    mapping0163 = expected_input_signature(R0163_MAPPING)
    mapping0113 = expected_input_signature(R0113_MAPPING)

    prompted = np.load(mapping0164["canonical_path"], mmap_mode="r", allow_pickle=False)
    accepted = np.load(mapping0113["canonical_path"], mmap_mode="r", allow_pickle=False)
    dropped = np.setdiff1d(
        prompted[prompted < PREFIX_STOP], accepted, assume_unique=True
    )
    if len(dropped) != 7:
        raise RuntimeError("R0165 registered seven-row prefix delta changed")

    layout_signature = expected_input_signature(LAYOUT_PATH)
    layout = _read_sealed(layout_signature, label="accepted R0162 layout")
    source_manifests = layout.get("source_manifests") or {}
    r0116_signature = dict(source_manifests["0116"])
    r0120_signature = dict(source_manifests["0120"])
    r0116 = _read_sealed(r0116_signature, label="accepted R0116 manifest")
    r0120 = _read_sealed(r0120_signature, label="accepted R0120 manifest")
    text_layout, _raw, _source_inputs = _source_layouts(r0116, r0120)
    payload_inputs = _touched_text_inputs(text_layout, dropped)
    if not payload_inputs:
        raise RuntimeError("R0165 dropped rows did not select a text payload")

    queue_root = create_fresh_directory(queue_root, label="R0165 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, "prompted-english-8m-frozen-prefix")
    expected_inputs = _dedupe([
        round_signature,
        *reviews,
        result0164,
        population0164,
        mapping0164,
        compact0164,
        hashes0164,
        mapping0163,
        mapping0113,
        layout_signature,
        r0116_signature,
        r0120_signature,
        *payload_inputs,
    ])
    job = {
        "id": "derive_frozen_prefix_population",
        "action": "derive_frozen_prefix_population",
        "handler_module": "experiments.round0165_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "frozen-prefix-population.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 300.0,
        "r0164_population": population0164,
        "r0163_mapping": mapping0163,
        "r0113_mapping": mapping0113,
        "r0116_manifest": r0116_signature,
        "r0120_manifest": r0120_signature,
        "payload_inputs": payload_inputs,
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
        "schema": "round0165-frozen-prefix-population-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0113", "0163", "0164"],
        "capability_dependencies": [],
        "capabilities_produced": [CAPABILITY, HOST_CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "source_rows": 8_000_000,
            "selection_rule": (
                "byte-exact accepted R0113 representatives below canonical 2M; "
                "R0164 prompted-only representatives at or above canonical 2M"
            ),
            "registered_dropped_r0164_prefix_rows": 7,
            "registered_added_over_r0163_extension_rows": 205,
            "accepted_r0113_first2m_prefix_must_remain_exact": True,
            "mapping_must_be_r0164_subset": True,
            "mapping_must_be_strict_r0163_superset": True,
            "raw_unprompted_embedding_relation_used": False,
            "multiplicity_is_metadata": True,
            "no_census": True,
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
    print(json.dumps({
        "queue_manifest": prepare_round0165(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
