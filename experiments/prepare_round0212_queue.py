#!/usr/bin/env python3
"""Prepare, but never launch, the R0212 seed-43 prompted-diverse U12 replay."""
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
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0168_prompted_diverse_staging import (
    CAPABILITY as STAGING_CAPABILITY,
)
from basemap.round0209_prompted_diverse_graph import (
    CAPABILITY as GRAPH_CAPABILITY,
    GRAPH_SCHEMA,
    plausible_directed_edges,
)
from basemap.round0210_prompted_diverse_low_dose import (
    HIDDEN_DIMENSION,
    HOST_RSS_LIMIT_GIB,
    ROWS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    achieved_draws_per_edge,
    successful_updates_for_edges,
)
from basemap.round0212_prompted_diverse_seed43 import (
    CANONICAL_SEED,
    CAPABILITY,
    PAIRED_CAPABILITY,
    ROUND_ID,
    SEED,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0169_queue import (
    STAGING_MANIFEST,
    _accepted_bundle,
    _read_sealed,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0212"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0212-2026-08-07.md")
R0168_REVIEW = os.path.join(LAB_ROOT, "review-0168-2026-08-03-01.md")
# R0209's first queue reached a registered terminal `failed` state and sealed no
# manifest; the graph was sealed by its dated `queue-correction-1` relaunch.
GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0209/queue-correction-1/artifacts/"
    "fuzzy-k50-graph-and-reference/graph-manifest.json"
)
GPU_HOURS_CAP = 8.0
#: Refuse to launch if the sealed graph implies a horizon this queue's budget
#: cannot honour. Registered in the round file alongside the GPU bound.
REGISTERED_UPDATE_BOUND = 2_100_000


def _issued_round(release_sha: str) -> dict[str, Any]:
    """Accept the issued round at its base commit or any descendant release.

    The R0173 ancestor test rather than exact equality: an issued round file is
    append-only, so its `base_commit` cannot be rewritten when a setup-class
    correction elsewhere in the campaign advances the shared release.
    """
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0212 round is not issued for this release")
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
        raise RuntimeError("R0212 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0212_prompted_diverse_seed43.py",
        "tests/test_round0166_cpu_smoke.py",
        "tests/test_round0169_prompted_diverse.py",
        "tests/test_panel_v2.py",
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
        timeout=180,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0212-release-cpu-smoke-v1",
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
            "low-dose horizon derivation, train config seal, post-fit accounting, "
            "checkpoint publish/reload, and the downstream panel interface"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0212 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _sealed_graph() -> tuple[dict[str, Any], dict[str, Any], int]:
    signature = expected_input_signature(GRAPH_MANIFEST)
    manifest = _read_sealed(signature, label="sealed R0209 prompted-diverse graph")
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != "0209"
        or int(manifest.get("retained_rows", -1)) != ROWS
        or manifest.get("training_performed") is not False
    ):
        raise RuntimeError("R0212 sealed R0209 graph contract changed")
    qualification = manifest.get("search_qualification") or {}
    selected = str(qualification.get("selected_nprobe") or "")
    if ((qualification.get("cells") or {}).get(selected) or {}).get("passed") is not True:
        raise RuntimeError("R0212 requires a qualified R0209 graph")
    edges = int(manifest.get("directed_edge_count", 0))
    if edges <= 0 or not plausible_directed_edges(edges):
        raise RuntimeError("R0212 sealed R0209 directed-edge count is implausible")
    return signature, manifest, edges


def prepare_round0210(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0212 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    dependencies = [
        *_accepted_bundle("0132"),
        *_accepted_bundle("0168", review_path=R0168_REVIEW),
        *_accepted_bundle("0209"),
    ]
    staging_signature = expected_input_signature(STAGING_MANIFEST)
    staging = _read_sealed(staging_signature, label="accepted R0168 staging")
    if int(staging.get("rows", -1)) != ROWS:
        raise RuntimeError("R0212 accepted staging contract changed")
    staging_inputs = [
        staging_signature,
        dict(staging["host_fp16"]),
        dict(staging["population"]["mapping"]),
        dict(staging["duplicate_control"]["arrays"]),
    ]
    graph_signature, graph_manifest, edges = _sealed_graph()
    updates = successful_updates_for_edges(edges)
    if updates > REGISTERED_UPDATE_BOUND:
        raise RuntimeError(
            f"R0212 derived horizon {updates} exceeds the registered bound "
            f"{REGISTERED_UPDATE_BOUND}"
        )
    graph_inputs = [
        graph_signature,
        dict(graph_manifest["graph"]),
        dict(graph_manifest["source"]),
        dict(graph_manifest["high_d_reference"]),
        dict(graph_manifest["topology_probe"]),
        *[dict(value) for value in (graph_manifest["centroids"] or {}).values()],
    ]

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0212 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    expected_inputs = _dedupe([
        round_signature,
        *dependencies,
        *staging_inputs,
        *graph_inputs,
        expected_input_signature(smoke_path),
    ])
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    train_output = os.path.join(artifacts, "seed43-low-dose-train")
    job = {
        "id": "train_prompted_diverse_u12_seed43",
        "action": "train_prompted_diverse_u12_seed43",
        "handler_module": "experiments.round0212_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [train_output],
        "done_marker": os.path.join(artifacts, "seed43-train.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 21_600.0,
        "staging_manifest": staging_signature,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_signature,
        "registered_dose_bound": REGISTERED_UPDATE_BOUND,
        "node_policy": {
            "gpu_required": True,
            "training_performed": True,
            "cpu_heavy": False,
        },
    }
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0212-prompted-diverse-u12-seed43-train-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": ["0132", "0168", "0209"],
        "capability_dependencies": [STAGING_CAPABILITY, GRAPH_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": [job],
        "p90_gpu_seconds": {
            "train_prompted_diverse_u12_seed43": 21_600.0,
            "total": 21_600.0,
        },
        "scientific_contract": {
            "population": "exact accepted R0168 12,474,331-row prompted U12 matrix",
            "graph": "sealed R0209 fuzzy k50 four-shard fp32 graph",
            "sealed_directed_edges": edges,
            "hidden_dimension": HIDDEN_DIMENSION,
            "seed": SEED,
            "paired_seed": CANONICAL_SEED,
            "paired_capability": PAIRED_CAPABILITY,
            "seed_family_gate_registerable_here": False,
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "successful_positive_lr_updates": updates,
            "achieved_positive_draws_per_edge": achieved_draws_per_edge(
                updates=updates, edge_count=edges
            ),
            "dose_rule": (
                "ceil(R0184_successful_updates * active_edges / R0184_directed_edges)"
            ),
            "polish_held_out": True,
            "multiplicity_policy": "metadata only; never a sampler weight",
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "evaluation_performed": False,
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
        "queue_manifest": prepare_round0210(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
