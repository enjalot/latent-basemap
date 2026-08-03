#!/usr/bin/env python3
"""Prepare, but never launch, the R0173 prompted U12 OOD probe pack."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0112_prompt_substrate import model_member_signatures
from basemap.round0116_prompted_corpus import environment_freeze_receipt
from basemap.round0168_prompted_diverse_staging import (
    CAPABILITY as STAGING_CAPABILITY,
    MANIFEST_SCHEMA as STAGING_SCHEMA,
)
from basemap.round0173_prompted_ood_pack import CAPABILITY, ROUND_ID
from basemap import round0113_prompt_contrast as prompt_contract
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0169_queue import (
    LANGUAGES,
    SELECTION_PATH,
    STAGING_MANIFEST,
    _accepted_bundle,
    _canary_inputs,
    _language_sources,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0173"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0173-2026-08-03.md")
GPU_HOURS_MAXIMUM = 2.0


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if frontmatter.get("status") != "issued" or not descendant:
        raise RuntimeError("R0173 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0173 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0173_prompted_ood_pack.py",
        "tests/test_round0169_prompted_diverse.py",
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
        "schema": "round0173-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
        "path_exercised": "prompt canary/embed/audit dispatch and immutable probe-pack contract",
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0173 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def prepare_round0173(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0173 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    dependencies = [
        *_accepted_bundle("0108"),
        *_accepted_bundle("0114"),
        *_accepted_bundle(
            "0168",
            review_path=os.path.join(LAB_ROOT, "review-0168-2026-08-03-01.md"),
        ),
    ]
    staging_signature = expected_input_signature(STAGING_MANIFEST)
    staging = prompt_contract.read_sealed(
        prompt_contract.verify_signature(staging_signature, label="accepted R0168 staging"),
        label="accepted R0168 staging",
    )
    if (
        staging.get("schema") != STAGING_SCHEMA
        or staging.get("round_id") != "0168"
        or staging.get("capability") != STAGING_CAPABILITY
        or int(staging.get("rows", -1)) != 12_474_331
        or (staging.get("population") or {}).get("polish_held_out") is not True
    ):
        raise RuntimeError("R0173 accepted staging contract changed")
    staging_inputs = [
        staging_signature,
        dict(staging["host_fp16"]),
        dict(staging["population"]["mapping"]),
    ]
    selection_signature = expected_input_signature(SELECTION_PATH)
    sources = _language_sources(selection_signature)
    model_members = model_member_signatures()
    environment = environment_freeze_receipt()
    canary = _canary_inputs()

    queue_root = create_fresh_directory(queue_root, label="R0173 prompted OOD queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )
    smoke_signature = expected_input_signature(smoke_path)
    common = _dedupe([
        round_signature,
        *dependencies,
        *staging_inputs,
        selection_signature,
        *sources.values(),
        *model_members,
        canary["text"],
        canary["document"],
        smoke_signature,
    ])
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    canary_output = os.path.join(artifacts, "prompt-model-canary")
    jobs: list[dict[str, Any]] = [{
        "id": "prompt_model_canary",
        "action": "prompt_canary",
        "handler_module": "experiments.round0173_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [canary_output],
        "done_marker": os.path.join(artifacts, "prompt-model-canary.done.json"),
        "expected_inputs": common,
        "p90_wall_s": 180.0,
        "canary_text": canary["text"],
        "canary_document": canary["document"],
        "canary_positions": canary["positions"],
        "model_members": model_members,
        "environment_freeze": environment,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }]
    language_outputs: dict[str, str] = {}
    embed_ids: list[str] = []
    for language in LANGUAGES:
        node_id = f"embed_prompted_{language}"
        output = os.path.join(artifacts, f"prompted-{language}")
        embed_ids.append(node_id)
        language_outputs[language] = output
        jobs.append({
            "id": node_id,
            "action": "embed_language_probe",
            "handler_module": "experiments.round0173_nodes",
            "handler_callable": "run_job",
            "deps": ["prompt_model_canary"],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
            "expected_inputs": common,
            "p90_wall_s": 300.0,
            "language": language,
            "selection": selection_signature,
            "text_source": sources[language],
            "canary_output": canary_output,
            "model_members": model_members,
            "environment_freeze": environment,
            "node_policy": {"gpu_required": True, "training_performed": False},
        })
    audit_output = os.path.join(artifacts, CAPABILITY)
    jobs.append({
        "id": "audit_prompted_ood_training_disjoint",
        "action": "audit_probe_training_disjoint",
        "handler_module": "experiments.round0173_nodes",
        "handler_callable": "run_job",
        "deps": embed_ids,
        "outputs": [audit_output],
        "done_marker": os.path.join(artifacts, "ood-training-audit.done.json"),
        "expected_inputs": common,
        "p90_wall_s": 1_800.0,
        "staging_manifest": staging_signature,
        "language_outputs": language_outputs,
        "canary_output": canary_output,
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
        "schema": "round0173-prompted-u12-ood-pack-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-evaluation",
        "required_reviews": ["0108", "0114", "0168"],
        "capability_dependencies": [
            "jina-diverse-25m-map-registry-v1",
            "jina-fineweb-2m-dual-prompt-native8192-substrate-v2",
            STAGING_CAPABILITY,
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
            "population": "exact accepted R0132 U12 prompted training rows",
            "selectors": "exact accepted R0108 49,500 corpus + 500 query rows per language",
            "languages": list(LANGUAGES),
            "embedding_convention": "literal Document: prefix, local native-8192 Jina-v5, fp16 storage",
            "exact_training_disjointness": "complete stored prompted-fp16 row bytes",
            "post_embedding_replacements": False,
            "training_performed": False,
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
        "queue_manifest": prepare_round0173(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
