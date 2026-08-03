#!/usr/bin/env python3
"""Prepare, but never launch, the R0175 approximate-UMAP OOS baseline."""
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
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory
from basemap.round0175_aumap_baseline import CAPABILITY, ROUND_ID, ROWS, SCALES
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _accepted_review, _frontmatter
from experiments.round0175_nodes import SOURCE_ROOT, TESTBED_ROOTS, TOOLCHAIN_PYTHON


ROUND_ROOT = "/data/latent-basemap/runs/round-0175"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0175-2026-08-03.md")
TOOLCHAIN_ROOT = "/data/latent-basemap/toolchains/aumap-v0.2.0-py312-r0175"
PACKAGE_FILES = (
    os.path.join(
        TOOLCHAIN_ROOT,
        "lib/python3.12/site-packages/approx_umap/approx_umap.py",
    ),
    os.path.join(
        TOOLCHAIN_ROOT,
        "lib/python3.12/site-packages/approx_umap-0.2.0.dist-info/METADATA",
    ),
    os.path.join(
        TOOLCHAIN_ROOT,
        "lib/python3.12/site-packages/approx_umap-0.2.0.dist-info/RECORD",
    ),
)
EVIDENCE_200K = os.path.join(
    RELEASE_ROOT, "experiments/evidence/r1_rescore/complete_200k.json"
)
EVIDENCE_2M = os.path.join(
    RELEASE_ROOT, "experiments/evidence/r1_rescore/complete_2m.json"
)
GPU_HOURS_MAXIMUM = 0.75


def _issued_round(release_sha: str) -> dict[str, Any]:
    if not os.path.isfile(ROUND_FILE):
        raise RuntimeError("R0175 issued round file is absent")
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
        raise RuntimeError("R0175 issued round binding changed")
    return expected_input_signature(ROUND_FILE)


def _source_signatures() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = expected_input_signature(os.path.join(os.path.dirname(SOURCE_ROOT), "manifest.json"))
    shards = [
        expected_input_signature(path)
        for path in sorted(glob.glob(os.path.join(SOURCE_ROOT, "data-*.npy")))
    ]
    if len(shards) != 11:
        raise RuntimeError(f"R0175 expected 11 source shards; found {len(shards)}")
    return manifest, shards


def prepare_round0175(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    round_signature = _issued_round(release_sha)
    review_inputs = _accepted_review("0037", "jina-mrl-seed42-screen-v1")
    source_manifest, source_shards = _source_signatures()
    probe_source = expected_input_signature(
        os.path.join(RELEASE_ROOT, "experiments", "round0175_official_probe.py")
    )
    toolchain_python = {
        "invocation_path": TOOLCHAIN_PYTHON,
        "resolved_interpreter": expected_input_signature(os.path.realpath(TOOLCHAIN_PYTHON)),
        "pyvenv_config": expected_input_signature(os.path.join(TOOLCHAIN_ROOT, "pyvenv.cfg")),
    }
    package_files = [expected_input_signature(path) for path in PACKAGE_FILES]
    evidence_200k = expected_input_signature(EVIDENCE_200K)
    evidence_2m = expected_input_signature(EVIDENCE_2M)

    testbeds: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        root = TESTBED_ROOTS[scale]
        embeddings = expected_input_signature(os.path.join(root, "train", "data-00000.npy"))
        sample_indices = expected_input_signature(os.path.join(root, "sample_indices.npy"))
        teacher = expected_input_signature(os.path.join(root, "ceiling_umaplearn_k50.parquet"))
        if embeddings["bytes"] != ROWS[scale] * 768 * 4 + 128:
            raise RuntimeError(f"R0175 {scale} testbed embedding size changed")
        testbeds[scale] = {
            "testbed_embeddings": embeddings,
            "sample_indices": sample_indices,
            "teacher": teacher,
        }

    queue_root = create_fresh_directory(queue_root, label="R0175 aUMAP queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    expected_inputs = _dedupe([
        round_signature,
        *review_inputs,
        source_manifest,
        *source_shards,
        probe_source,
        toolchain_python["resolved_interpreter"],
        toolchain_python["pyvenv_config"],
        *package_files,
        evidence_200k,
        evidence_2m,
        *[
            signature
            for cell in testbeds.values()
            for signature in cell.values()
        ],
    ])

    probe_output = os.path.join(artifacts, "official-formula-probe")
    jobs: list[dict[str, Any]] = [{
        "id": "validate_official_approx_umap_formula",
        "action": "official_probe",
        "handler_module": "experiments.round0175_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "toolchain_python": toolchain_python,
        "package_files": package_files,
        "probe_source": probe_source,
        "outputs": [probe_output],
        "done_marker": os.path.join(artifacts, "official-formula-probe.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 120.0,
        "node_policy": {"gpu_required": False, "training_performed": False},
    }]
    scale_outputs: dict[str, str] = {}
    p90_by_scale = {"200k": 180.0, "500k": 300.0, "2m": 720.0}
    for scale in SCALES:
        output = os.path.join(artifacts, f"aumap-{scale}")
        scale_outputs[scale] = output
        jobs.append({
            "id": f"measure_aumap_{scale}",
            "action": "scale",
            "scale": scale,
            "handler_module": "experiments.round0175_nodes",
            "handler_callable": "run_job",
            "deps": ["validate_official_approx_umap_formula"],
            **testbeds[scale],
            "source_manifest": source_manifest,
            "source_shards": source_shards,
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"measure-aumap-{scale}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": p90_by_scale[scale],
            "node_policy": {"gpu_required": True, "training_performed": False},
        })
    synthesis_output = os.path.join(artifacts, CAPABILITY)
    jobs.append({
        "id": "synthesize_aumap_baseline",
        "action": "synthesis",
        "handler_module": "experiments.round0175_nodes",
        "handler_callable": "run_job",
        "deps": [f"measure_aumap_{scale}" for scale in SCALES],
        "official_probe_output": probe_output,
        "scale_outputs": scale_outputs,
        "evidence_200k": evidence_200k,
        "evidence_2m": evidence_2m,
        "outputs": [synthesis_output],
        "done_marker": os.path.join(artifacts, "synthesize-aumap.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 60.0,
        "node_policy": {"gpu_required": False, "training_performed": False},
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
        "schema": "round0175-aumap-oos-baseline-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-evaluation",
        "required_reviews": ["0037"],
        "capability_dependencies": ["jina-mrl-seed42-screen-v1"],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            f"measure_aumap_{scale}": p90_by_scale[scale] for scale in SCALES
        } | {"total": sum(p90_by_scale.values())},
        "scientific_contract": {
            "question": (
                "how well does official aUMAP inverse-distance k15 projection "
                "preserve held-out neighborhoods on frozen standard-UMAP teachers?"
            ),
            "scales": list(SCALES),
            "teacher": "existing standard umap-learn k50 transductive coordinates",
            "queries": "same deterministic held-out source IDs as the R1 panel",
            "neighbor_search": "exact fp32 cosine GPU IndexFlatIP",
            "treatment": "approx-umap==0.2.0 fn=inv, k=1, n_neighbors=15",
            "paired_control": "unweighted mean over the identical k15 neighbors",
            "quality_role": "diagnostic only",
            "training_performed": False,
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
    path = prepare_round0175(release_sha=args.release_sha, queue_root=args.queue_root)
    print(json.dumps({"path": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
