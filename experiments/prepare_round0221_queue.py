#!/usr/bin/env python3
"""Prepare, but never launch, the R0221 MiniLM 2M seed-extension queue.

Four train nodes in one queue, seeds 46-49. The script reads the sealed R0216
`queue-correction-3` receipt, builds all four cell configs *here* from R0217's
own `train_config`, proves each one reproduces R0217's **published**
seed-invariant digest, proves the four new full-config digests are distinct from
each other and from R0217's four, and stamps the shared digest into every job so
each node can re-derive it and refuse to train if its own config drifted.
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
    HIDDEN_DIMENSION,
    TRAIN_SCHEMA as R0217_TRAIN_SCHEMA,
    capability_for_seed as r0217_capability_for_seed,
)
from basemap.round0221_minilm_2m_seed_extension import (
    CAPABILITIES,
    DIMENSION,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    HOST_RSS_LIMIT_GIB,
    POOLED_SEEDS,
    R0217_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    REGISTERED_ACHIEVED_DRAWS_PER_EDGE,
    REGISTERED_SUCCESSFUL_UPDATES,
    REGISTERED_UPDATE_BOUND,
    ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    TEMPLATE_SEED,
    USE_AMP,
    assert_extension_differs_only_by_seed,
    capability_for_seed,
    successful_updates_for_edges,
    train_config,
    validate_registered_dose,
)
from experiments.round0221_nodes import ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0221"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0221-2026-08-08.md")
#: R0216's terminal, corpus-spanning queue — the same bytes R0217 trained on.
R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
#: R0217's terminal queue: the four cells this round extends.
R0217_ARTIFACTS = "/data/latent-basemap/runs/round-0217/queue-correction-1/artifacts"
#: R0217 measured 0.197 GPU-h per cell; four cells is ~0.79 GPU-h. The cap is
#: the round's registered 2.0 h, which also covers the added full-population
#: transform in every cell.
GPU_HOURS_CAP = 2.0
NODE_P90_WALL_S = 1_800.0


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
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
        raise RuntimeError("R0221 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0221 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0221 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0221_minilm_2m_seed_extension.py",
        "tests/test_round0221_cpu_smoke.py",
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
        "schema": "round0221-release-cpu-smoke-v1",
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
            "R0217-template config construction, seed-invariant digest equality, "
            "registered ceil-derived dose assertion, train config seal, post-fit "
            "accounting, checkpoint publish, full-population reload and "
            "finiteness check, and the receipt seal"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0221 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
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
        raise RuntimeError("R0221 sealed R0216 substrate+graph contract changed")
    if int(checks.get("zero_degree_rows", -1)) != 0:
        raise RuntimeError("R0221 requires a graph with zero degree-zero rows")
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if edges != SEALED_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0221 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    return signature, manifest, edges


def _r0217_family(
    manifest_signature: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Read R0217's four sealed cells: the treatment these four must match."""
    invariants: set[str] = set()
    config_hashes: dict[str, str] = {}
    model_hashes: dict[str, str] = {}
    for seed in R0217_SEEDS:
        capability = r0217_capability_for_seed(seed)
        receipt_path = os.path.join(R0217_ARTIFACTS, capability, "train-receipt.json")
        receipt = prompt_contract.read_sealed(
            receipt_path, label=f"R0217 seed-{seed} train receipt"
        )
        train_checks = receipt.get("train_checks") or {}
        if (
            receipt.get("schema") != R0217_TRAIN_SCHEMA
            or receipt.get("round_id") != "0217"
            or receipt.get("capability") != capability
            or int(receipt.get("training_seed", -1)) != seed
            or int(receipt.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
            or receipt.get("training_performed") is not True
            or not train_checks
            or not all(bool(value) for value in train_checks.values())
        ):
            raise RuntimeError(f"R0217 seed-{seed} train receipt contract changed")
        if (
            dict(receipt.get("substrate") or {}) != dict(manifest["substrate"])
            or dict(receipt.get("graph_manifest") or {}) != manifest_signature
        ):
            raise RuntimeError(
                f"R0217 seed-{seed} was not trained on the substrate R0221 extends"
            )
        if int(receipt.get("optimizer_updates", -1)) != REGISTERED_SUCCESSFUL_UPDATES:
            raise RuntimeError(
                f"R0217 seed-{seed} horizon is not the registered "
                f"{REGISTERED_SUCCESSFUL_UPDATES}"
            )
        invariants.add(str(receipt["seed_invariant_sha256"]))
        config_hashes[str(seed)] = str(receipt["production_config_sha256"])
        model_hashes[str(seed)] = str(receipt["model"]["sha256"])
    if invariants != {R0217_SEED_INVARIANT_SHA256}:
        raise RuntimeError(
            "R0217's sealed cells do not carry the published seed-invariant "
            f"digest: {sorted(invariants)}"
        )
    if len(set(model_hashes.values())) != len(R0217_SEEDS):
        raise RuntimeError("R0217 family contains a duplicated checkpoint")
    return {
        "seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "config_sha256_by_seed": config_hashes,
        "model_sha256_by_seed": model_hashes,
    }


def prepare_round0221(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0221 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    graph_manifest_signature, graph_manifest, edges = _sealed_graph()
    updates = successful_updates_for_edges(edges)
    if updates > REGISTERED_UPDATE_BOUND:
        raise RuntimeError(
            f"R0221 derived horizon {updates} exceeds the registered bound "
            f"{REGISTERED_UPDATE_BOUND}"
        )
    dose = validate_registered_dose(updates=updates, edge_count=edges)
    r0217_family = _r0217_family(graph_manifest_signature, graph_manifest)

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
    family = assert_extension_differs_only_by_seed(
        configs, expected_seed_invariant=r0217_family["seed_invariant_sha256"]
    )
    if not family["matches_r0217_published_seed_invariant"]:
        raise RuntimeError(
            "R0221 seed-invariant digest does not match R0217's published value"
        )
    pooled_config_hashes = {
        **r0217_family["config_sha256_by_seed"],
        **family["per_seed_config_sha256"],
    }
    if len(set(pooled_config_hashes.values())) != len(POOLED_SEEDS):
        raise RuntimeError(
            "R0221 cell configs collide with R0217's: the eight cells must be "
            "eight distinct configs sharing one seed-invariant digest"
        )

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0221 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    family_path = os.path.join(preflight, "seed-extension-configs.json")
    atomic_write_new_json(
        family_path,
        prompt_contract.seal({
            "schema": "round0221-seed-extension-config-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "sealed_directed_edges": edges,
            "successful_positive_lr_updates": updates,
            "dose_registration": dose,
            "r0217_family": r0217_family,
            "family": family,
            "pooled_seed_family": list(POOLED_SEEDS),
            "pooled_config_sha256_by_seed": pooled_config_hashes,
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
            "handler_module": "experiments.round0221_nodes",
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
        "schema": "round0221-minilm-mixed-2m-seed-extension-train-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [GRAPH_CAPABILITY],
        "capabilities_produced": list(CAPABILITIES),
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": p90,
        "scientific_contract": {
            "question": (
                "do four more seeds under R0217's exact treatment extend the "
                "MiniLM mixed-2M family from n=4 to n=8, so a mean - 2 sigma "
                "gate can be tested by its own defining cells?"
            ),
            "population": "sealed R0216 2,000,000-row mixed MiniLM substrate",
            "graph": (
                "sealed R0216 exact k15 fuzzy graph (recall 1.000000, 0 "
                "zero-degree rows)"
            ),
            "sealed_directed_edges": edges,
            "hidden_dimension": HIDDEN_DIMENSION,
            "input_dimension": DIMENSION,
            "precision": USE_AMP,
            "seeds": list(SEEDS),
            "cells": len(SEEDS),
            "pooled_seed_family": list(POOLED_SEEDS),
            "treatment_source_round": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "capabilities_by_seed": {
                str(seed): capability_for_seed(seed) for seed in SEEDS
            },
            "family_seed_invariant_sha256": family["seed_invariant_sha256"],
            "r0217_published_seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
            "per_seed_config_sha256": family["per_seed_config_sha256"],
            "pooled_config_sha256_by_seed": pooled_config_hashes,
            "r0217_model_sha256_by_seed": r0217_family["model_sha256_by_seed"],
            "only_treatment_between_cells": "the training seed",
            "gate_registerable_here": GATE_REGISTERABLE_HERE,
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "successful_positive_lr_updates": updates,
            "registered_successful_positive_lr_updates": (
                REGISTERED_SUCCESSFUL_UPDATES
            ),
            "achieved_positive_draws_per_edge": dose[
                "achieved_positive_draws_per_edge"
            ],
            "registered_achieved_positive_draws_per_edge": (
                REGISTERED_ACHIEVED_DRAWS_PER_EDGE
            ),
            "dose_quantum_draws_per_edge": dose["dose_quantum_draws_per_edge"],
            "dose_rule": dose["dose_rule"],
            "registered_update_bound": REGISTERED_UPDATE_BOUND,
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "full_population_transform_rows": ROWS,
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
        "queue_manifest": prepare_round0221(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
