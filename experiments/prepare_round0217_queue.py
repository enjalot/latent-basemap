#!/usr/bin/env python3
"""Prepare, but never launch, the R0217 MiniLM 2M four-seed family queue.

Four train nodes in one queue. The script reads the sealed R0216 receipt, builds
all four cell configs *here*, proves they differ only by the seed, and stamps the
resulting seed-invariant digest into every job — so each node can re-derive it at
runtime and refuse to train if its own config drifted.
"""
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
from basemap.round0216_minilm_2m_substrate import CAPABILITY as R0216_CAPABILITY
from basemap.round0217_minilm_2m_seed_family import (
    CAPABILITIES,
    DIMENSION,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    HIDDEN_DIMENSION,
    HOST_RSS_LIMIT_GIB,
    ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    SEEDS_REQUIRED_FOR_FAMILY_GATE,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    USE_AMP,
    achieved_draws_per_edge,
    assert_family_differs_only_by_seed,
    capability_for_seed,
    successful_updates_for_edges,
    train_config,
    validate_dose,
)
from experiments.round0217_nodes import ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0217"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0217-2026-08-08.md")
#: The sealed R0216 artifact directory. `queue-correction-3` is the terminal
#: queue: `queue/` and `queue-correction-1/` failed and sealed nothing, and
#: `queue-correction-2/` succeeded but was downgraded to partial by review
#: 0216-01 for taking a leading prefix of each corpus instead of spanning it.
#: Only correction-3's selection covers 100% of every corpus's shards.
R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
#: ~0.2 GPU-h per seed at ~110 upd/s; four cells plus headroom.
GPU_HOURS_CAP = 1.5
NODE_P90_WALL_S = 1_800.0
#: Refuse to launch if the sealed graph implies a horizon this budget cannot
#: honour. Registered in the round file alongside the GPU bound.
REGISTERED_UPDATE_BOUND = 120_000


def _issued_round(release_sha: str) -> dict[str, Any]:
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
        raise RuntimeError("R0217 round is not issued for this release")
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
        raise RuntimeError("R0217 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0217_minilm_2m_seed_family.py",
        "tests/test_round0217_cpu_smoke.py",
        "tests/test_round0166_cpu_smoke.py",
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
        "schema": "round0217-release-cpu-smoke-v1",
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
            "sealed-edge-count horizon derivation, dose assertion, four-cell "
            "config identity, train config seal, post-fit accounting, checkpoint "
            "publish, reload and collapse check, and the receipt seal"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0217 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _sealed_graph() -> tuple[dict[str, Any], dict[str, Any], int]:
    signature = expected_input_signature(GRAPH_MANIFEST)
    manifest = prompt_contract.read_sealed(
        signature["canonical_path"], label="sealed R0216 substrate+graph receipt"
    )
    checks = manifest.get("graph_checks") or {}
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != GRAPH_SOURCE_ROUND_ID
        or manifest.get("capability") != GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or manifest.get("training_performed") is not False
    ):
        raise RuntimeError("R0217 sealed R0216 substrate+graph contract changed")
    if int(checks.get("zero_degree_rows", -1)) != 0:
        raise RuntimeError("R0217 requires a graph with zero degree-zero rows")
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if edges != SEALED_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0217 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    return signature, manifest, edges


def prepare_round0217(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0217 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    graph_manifest_signature, graph_manifest, edges = _sealed_graph()
    updates = successful_updates_for_edges(edges)
    if updates > REGISTERED_UPDATE_BOUND:
        raise RuntimeError(
            f"R0217 derived horizon {updates} exceeds the registered bound "
            f"{REGISTERED_UPDATE_BOUND}"
        )
    dose = validate_dose(updates=updates, edge_count=edges)

    substrate_signature = dict(graph_manifest["substrate"])
    graph_signature = dict(graph_manifest["graph"])
    provenance_signature = dict(graph_manifest["provenance"])

    configs = {}
    for seed in SEEDS:
        config, _sha = train_config(
            seed=seed,
            graph_signature=graph_signature,
            graph_manifest_signature=graph_manifest_signature,
            substrate_signature=substrate_signature,
            graph_edges=edges,
            rows=ROWS,
        )
        configs[seed] = config
    family = assert_family_differs_only_by_seed(configs)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0217 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    family_path = os.path.join(preflight, "seed-family-configs.json")
    atomic_write_new_json(
        family_path,
        prompt_contract.seal({
            "schema": "round0217-seed-family-config-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "sealed_directed_edges": edges,
            "successful_positive_lr_updates": updates,
            "dose_registration": dose,
            "family": family,
            "configs": {str(seed): configs[seed] for seed in SEEDS},
        }),
        immutable=True,
    )
    expected_inputs = _dedupe([
        round_signature,
        graph_manifest_signature,
        substrate_signature,
        graph_signature,
        provenance_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(family_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs = []
    p90 = {}
    for seed in SEEDS:
        capability = capability_for_seed(seed)
        node = f"train_minilm_mixed_2m_seed{seed}"
        jobs.append({
            "id": node,
            "action": ACTION,
            "handler_module": "experiments.round0217_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [os.path.join(artifacts, capability)],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": NODE_P90_WALL_S,
            "training_seed": int(seed),
            "capability": capability,
            "graph_manifest": GRAPH_MANIFEST,
            "graph_manifest_signature": graph_manifest_signature,
            "family_seed_invariant_sha256": family["seed_invariant_sha256"],
            "registered_dose_bound": REGISTERED_UPDATE_BOUND,
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
                "cpu_heavy": False,
            },
        })
        p90[node] = NODE_P90_WALL_S
    p90["total"] = NODE_P90_WALL_S * len(SEEDS)

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
        "schema": "round0217-minilm-mixed-2m-seed-family-train-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": ["0216"],
        "capability_dependencies": [GRAPH_CAPABILITY],
        "capabilities_produced": list(CAPABILITIES),
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": p90,
        "scientific_contract": {
            "population": "sealed R0216 2,000,000-row mixed MiniLM substrate",
            "graph": "sealed R0216 exact k15 fuzzy graph (recall 1.000000, 0 zero-degree rows)",
            "sealed_directed_edges": edges,
            "hidden_dimension": HIDDEN_DIMENSION,
            "input_dimension": DIMENSION,
            "precision": USE_AMP,
            "seeds": list(SEEDS),
            "cells": len(SEEDS),
            "capabilities_by_seed": {
                str(seed): capability_for_seed(seed) for seed in SEEDS
            },
            "family_seed_invariant_sha256": family["seed_invariant_sha256"],
            "per_seed_config_sha256": family["per_seed_config_sha256"],
            "only_treatment_between_cells": "the training seed",
            "cells_required_for_gate": SEEDS_REQUIRED_FOR_FAMILY_GATE,
            "gate_registerable_here": GATE_REGISTERABLE_HERE,
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "successful_positive_lr_updates": updates,
            "achieved_positive_draws_per_edge": achieved_draws_per_edge(
                updates=updates, edge_count=edges
            ),
            "dose_quantum_draws_per_edge": dose["dose_quantum_draws_per_edge"],
            "dose_rule": dose["dose_rule"],
            "registered_update_bound": REGISTERED_UPDATE_BOUND,
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
        "queue_manifest": prepare_round0217(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
