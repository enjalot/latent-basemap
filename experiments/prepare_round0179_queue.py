#!/usr/bin/env python3
"""Prepare, but never launch, the R0179 NUMAP 200k baseline queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0179_numap_baseline import CAPABILITY, ROUND_ID, ROWS
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _accepted_review, _frontmatter
from experiments.prepare_round0175_queue import _source_signatures
from experiments.round0179_nodes import (
    REFERENCE_SCRIPT,
    TESTBED_ROOT,
    TOOLCHAIN_PYTHON,
    TOOLCHAIN_ROOT,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0179"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0179-2026-08-03.md")
AUMAP_SYNTHESIS = (
    "/data/latent-basemap/runs/round-0175/queue/artifacts/"
    "jina-aumap-oos-baseline-v1/synthesis.json"
)
GPU_HOURS_MAXIMUM = 3.0
GPU_P90_SECONDS = 7_200.0


def _issued_round(release_sha: str) -> dict[str, Any]:
    if not os.path.isfile(ROUND_FILE):
        raise RuntimeError("R0179 issued round file is absent")
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        [
            "git",
            "-C",
            RELEASE_ROOT,
            "merge-base",
            "--is-ancestor",
            base_commit,
            release_sha,
        ],
        check=False,
        timeout=10,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0179 issued round binding changed")
    return expected_input_signature(ROUND_FILE)


def _package_files() -> list[str]:
    site = os.path.join(TOOLCHAIN_ROOT, "lib", "python3.12", "site-packages")
    paths = sorted(
        glob.glob(os.path.join(site, "numap", "**", "*.py"), recursive=True)
        + glob.glob(os.path.join(site, "grease", "**", "*.py"), recursive=True)
    )
    for distribution in ("numap-0.2.3.dist-info", "grease_embeddings-0.1.5.dist-info"):
        paths.extend(
            os.path.join(site, distribution, name)
            for name in ("METADATA", "RECORD")
        )
    if len(paths) != 28 or not all(os.path.isfile(path) for path in paths):
        raise RuntimeError(
            f"R0179 package source closure changed: found {len(paths)} files"
        )
    return paths


def prepare_round0179(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    round_signature = _issued_round(release_sha)
    review_inputs = _accepted_review("0175", "jina-aumap-oos-baseline-v1")
    source_manifest, source_shards = _source_signatures()
    testbed_embeddings = expected_input_signature(
        os.path.join(TESTBED_ROOT, "train", "data-00000.npy")
    )
    sample_indices = expected_input_signature(
        os.path.join(TESTBED_ROOT, "sample_indices.npy")
    )
    if testbed_embeddings["bytes"] != ROWS * 768 * 4 + 128:
        raise RuntimeError("R0179 200k embedding matrix size changed")
    reference_script = expected_input_signature(REFERENCE_SCRIPT)
    package_files = [expected_input_signature(path) for path in _package_files()]
    toolchain_python = {
        "invocation_path": TOOLCHAIN_PYTHON,
        "resolved_interpreter": expected_input_signature(os.path.realpath(TOOLCHAIN_PYTHON)),
        "pyvenv_config": expected_input_signature(os.path.join(TOOLCHAIN_ROOT, "pyvenv.cfg")),
    }
    aumap_synthesis = expected_input_signature(AUMAP_SYNTHESIS)

    queue_root = create_fresh_directory(queue_root, label="R0179 NUMAP queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    expected_inputs = _dedupe([
        round_signature,
        *review_inputs,
        source_manifest,
        *source_shards,
        testbed_embeddings,
        sample_indices,
        reference_script,
        toolchain_python["resolved_interpreter"],
        toolchain_python["pyvenv_config"],
        *package_files,
        aumap_synthesis,
    ])

    cell_output = os.path.join(artifacts, "numap-200k")
    synthesis_output = os.path.join(artifacts, CAPABILITY)
    jobs: list[dict[str, Any]] = [
        {
            "id": "fit_and_score_numap_200k",
            "action": "numap_cell",
            "handler_module": "experiments.round0179_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "testbed_embeddings": testbed_embeddings,
            "sample_indices": sample_indices,
            "source_manifest": source_manifest,
            "source_shards": source_shards,
            "reference_script": reference_script,
            "toolchain_python": toolchain_python,
            "package_files": package_files,
            "outputs": [cell_output],
            "done_marker": os.path.join(artifacts, "fit-and-score-numap-200k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": GPU_P90_SECONDS,
            "node_policy": {"gpu_required": True, "training_performed": True},
        },
        {
            "id": "synthesize_numap_baseline",
            "action": "synthesis",
            "handler_module": "experiments.round0179_nodes",
            "handler_callable": "run_job",
            "deps": ["fit_and_score_numap_200k"],
            "cell_output": cell_output,
            "aumap_synthesis": aumap_synthesis,
            "outputs": [synthesis_output],
            "done_marker": os.path.join(artifacts, "synthesize-numap.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 60.0,
            "node_policy": {"gpu_required": False, "training_performed": False},
        },
    ]

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
        "schema": "round0179-numap-oos-baseline-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": ["0175"],
        "capability_dependencies": ["jina-aumap-oos-baseline-v1"],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "fit_and_score_numap_200k": GPU_P90_SECONDS,
            "total": GPU_P90_SECONDS,
        },
        "scientific_contract": {
            "question": (
                "what held-out neighborhood fidelity does the unmodified NUMAP "
                "0.2.3 official-example GrEASE/residual path achieve at 200k?"
            ),
            "rows": ROWS,
            "queries": N_QUERIES,
            "treatment": (
                "unmodified numap==0.2.3 + grease-embeddings==0.1.5; cosine "
                "k10, GrEASE se_dim5, residual PUMAP, package-example lr/epochs/batch"
            ),
            "comparison": (
                "reviewed R0175 aUMAP 200k on identical rows, held IDs, high "
                "truth, and metric formulas; diagnostic only"
            ),
            "quality_role": "diagnostic only; no method-winner branch",
            "map_registry_state_changed": False,
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
    path = prepare_round0179(release_sha=args.release_sha, queue_root=args.queue_root)
    print(json.dumps({"path": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
