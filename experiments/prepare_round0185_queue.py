#!/usr/bin/env python3
"""Prepare, but never launch, the conditional CPU-only R0185 probe repair."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0168_prompted_diverse_staging import CAPABILITY as STAGING_CAPABILITY
from basemap.round0180_dose_matched_8m import CAPABILITY as R0180_CAPABILITY
from basemap.round0185_prompted_ood_disjoint_pack import (
    CAPABILITY,
    EXPECTED_REMOVALS,
    RETAINED_PROBE_ROWS,
    ROUND_ID,
    SOURCE_PROBE_ROWS,
    TRAINING_ROWS,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import (
    _accepted_review,
    _frontmatter,
    _frontmatter_list,
)
from experiments.prepare_round0184_queue import _accepted_terminal_review


ROUND_ROOT = "/data/latent-basemap/runs/round-0185"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0185-2026-08-03.md")
R0168_MANIFEST = (
    "/data/latent-basemap/runs/round-0168/queue/artifacts/"
    "prompted-diverse-u12/prompted-u12-manifest.json"
)
R0173_AUDIT = (
    "/data/latent-basemap/runs/round-0173/queue/artifacts/"
    "jina-prompted-u12-ood-probe-pack-v1/audit.json"
)
R0168_REVIEW = os.path.join(LAB_ROOT, "review-0168-2026-08-03-01.md")


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0185 issued round binding changed")
    return expected_input_signature(ROUND_FILE)


def _accepted_r0168_review() -> list[dict[str, Any]]:
    """Bind the append-only correction that supersedes the first R0168 review."""
    review = _frontmatter(R0168_REVIEW)
    if (
        review.get("round_id") != "0168"
        or review.get("status") != "accepted"
        or f"capability:{STAGING_CAPABILITY}"
        not in _frontmatter_list(review, "releases")
    ):
        raise RuntimeError("corrected R0168 review is not accepted")
    round_path = os.path.join(LAB_ROOT, review.get("round") or "")
    result_path = os.path.join(LAB_ROOT, review.get("result") or "")
    issued = expected_input_signature(round_path)
    result = expected_input_signature(result_path)
    if (
        issued["sha256"] != review.get("round_sha256")
        or result["sha256"] != review.get("result_sha256")
    ):
        raise RuntimeError("corrected R0168 review binding changed")
    return [issued, result, expected_input_signature(R0168_REVIEW)]


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0185 CPU checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0185_prompted_ood_disjoint_pack.py",
        "tests/test_round0173_prompted_ood_pack.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0185-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
        "path_exercised": (
            "failed-audit authentication -> registered exact-family removal -> "
            "retained-position sealing -> complete retained/training byte audit"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0185 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _artifact_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest_signature = expected_input_signature(R0168_MANIFEST)
    manifest = prompt_contract.read_sealed(
        R0168_MANIFEST, label="accepted R0168 staging manifest"
    )
    if (
        manifest.get("round_id") != "0168"
        or int(manifest.get("rows", -1)) != TRAINING_ROWS
        or not isinstance(manifest.get("host_fp16"), Mapping)
    ):
        raise RuntimeError("R0168 staging manifest changed")
    training_signature = expected_input_signature(
        manifest["host_fp16"]["canonical_path"]
    )
    if training_signature != manifest["host_fp16"]:
        raise RuntimeError("R0168 prompted matrix changed")

    audit_signature = expected_input_signature(R0173_AUDIT)
    audit = prompt_contract.read_sealed(R0173_AUDIT, label="accepted failed R0173 audit")
    if (
        audit.get("round_id") != "0173"
        or audit.get("passed") is not False
        or int(audit.get("probe_rows", -1)) != SOURCE_PROBE_ROWS
        or int(audit.get("exact_training_family_overlap_count", -1))
        != len(EXPECTED_REMOVALS)
    ):
        raise RuntimeError("R0173 failed audit changed")
    probe_inputs: list[dict[str, Any]] = []
    for language, values in sorted((audit.get("language_outputs") or {}).items()):
        if not isinstance(values, Mapping):
            raise RuntimeError(f"R0173 {language} output binding changed")
        for key in (
            "receipt",
            "corpus_embeddings",
            "query_embeddings",
            "corpus_source_rows",
            "query_source_rows",
        ):
            expected = values.get(key)
            if not isinstance(expected, Mapping):
                raise RuntimeError(f"R0173 {language} {key} is missing")
            observed = expected_input_signature(expected["canonical_path"])
            if observed != expected:
                raise RuntimeError(f"R0173 {language} {key} changed")
            probe_inputs.append(observed)
    return manifest_signature, audit_signature, _dedupe([
        manifest_signature,
        training_signature,
        audit_signature,
        *probe_inputs,
    ])


def prepare_round0185(
    *, release_sha: str, queue_root: str = QUEUE_ROOT
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0185 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    r0168_evidence = _accepted_r0168_review()
    r0173_evidence = _accepted_terminal_review("0173")
    r0180_evidence = _accepted_review("0180", R0180_CAPABILITY)
    manifest_signature, audit_signature, artifact_inputs = _artifact_inputs()

    queue_root = create_fresh_directory(queue_root, label="R0185 CPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    smoke_signature = expected_input_signature(smoke_path)
    expected_inputs = _dedupe([
        round_signature,
        *r0168_evidence,
        *r0173_evidence,
        *r0180_evidence,
        *artifact_inputs,
        smoke_signature,
    ])
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "filter_and_audit_prompted_ood_pack",
        "action": "filter_and_audit_prompted_ood_pack",
        "handler_module": "experiments.round0185_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "source_audit": audit_signature,
        "staging_manifest": manifest_signature,
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "filter-and-audit.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 180.0,
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
        "schema": "round0185-prompted-u12-ood-disjoint-pack-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-heavy",
        "required_reviews": ["0168", "0173", "0180"],
        "capability_dependencies": [STAGING_CAPABILITY, R0180_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "operation": (
                "remove exactly the five Review-0173 accepted stored-fp16 training "
                "families, then independently rescan every retained probe row "
                "against all R0168 training rows"
            ),
            "source_probe_rows": SOURCE_PROBE_ROWS,
            "removed_probe_rows": len(EXPECTED_REMOVALS),
            "retained_probe_rows": RETAINED_PROBE_ROWS,
            "expected_removals": [list(item) for item in EXPECTED_REMOVALS],
            "queries_must_remain_unchanged": True,
            "replacement_rows": 0,
            "reembedding": False,
            "retained_overlap_floor": 0,
            "failed_round_reuse_boundary": (
                "consume only Review-0173-authenticated successful embedding payloads; "
                "do not consume its failed pack as a capability; revalidate every "
                "payload and independently mint a new filtered-view receipt"
            ),
            "memory_basis": {
                "nearest_same_shape_round": "0173",
                "same_training_rows": TRAINING_ROWS,
                "same_source_probe_rows": SOURCE_PROBE_ROWS,
                "measured_audit_wall_s": 31.07894393801689,
                "scaling_argument": (
                    "same fingerprint/full-byte scan plus forty small position arrays; "
                    "no embedding or full probe-matrix copy"
                ),
            },
            "release_cpu_smoke": smoke_signature,
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
        "queue_manifest": prepare_round0185(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
