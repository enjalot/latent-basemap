#!/usr/bin/env python3
"""Prepare, but never launch, the R0223 cuVS-graph map queue.

Five nodes in one queue: one cuVS fuzzy-graph build, three train cells (seeds
42/43/44), one panel comparison. The graph node runs first because its edge
count determines every cell's horizon, so the train configs cannot be built here
— they are built inside each node from R0217's template against the sealed cuVS
receipt. What *is* proved here, before launch, is that the construction is
sound: the script builds a cell config against a *hypothetical* edge count and
requires it to reproduce the treatment-invariant digest of the unmodified R0217
template, so a drifted `GRAPH_BEARING_PATHS` register fails at prepare time
rather than after a graph build.
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
    GRAPH_CAPABILITY as R0216_GRAPH_CAPABILITY,
    GRAPH_SCHEMA as R0216_GRAPH_SCHEMA,
    HIDDEN_DIMENSION,
    train_config as r0217_train_config,
)
from basemap.round0218_minilm_2m_panel import CAPABILITY as R0218_PANEL_CAPABILITY
from basemap.round0223_cuvs_graph_map import (
    COMPARISON_CAPABILITY,
    CUVS_GRAPH_CAPABILITY,
    CUVS_INTERMEDIATE_GRAPH_DEGREE,
    CUVS_SETTING_ID,
    DIMENSION,
    EVIDENCE_LIMITS,
    FLOOR_STATUS,
    GATE_REGISTERABLE_HERE,
    GATE_RELEASE_CLAIMED,
    GRAPH_BEARING_PATHS,
    GRAPH_BEARING_REASONS,
    GRAPH_K,
    HOST_RSS_LIMIT_GIB,
    MAP_CAPABILITIES,
    PIPELINE_STAMP_LABEL_CARRYOVER,
    R0216_SEALED_DIRECTED_EDGES,
    R0220_CUVS_GRAPH_SIGNATURE,
    R0220_QUALIFICATION_SIGNATURE,
    R0220_TRUTH_RECEIPT_SIGNATURE,
    R0222_GATE_ARTIFACT_ROOT,
    R0222_POOLED_SEEDS,
    REGISTERED_UPDATE_BOUND,
    ROUND_ID,
    ROWS,
    SEEDS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    TEMPLATE_SEED,
    USE_AMP,
    map_capability,
    train_config,
    treatment_invariant_sha256,
)
from experiments.round0223_nodes import COMPARE_ACTION, GRAPH_ACTION, TRAIN_ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0223"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0223-2026-08-08.md")

R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
R0216_GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
R0218_PANEL_EVIDENCE = (
    "/data/latent-basemap/runs/round-0218/queue/artifacts/"
    f"{R0218_PANEL_CAPABILITY}/seed-family-panel.json"
)
R0222_GATE_ARTIFACT = os.path.join(
    R0222_GATE_ARTIFACT_ROOT, "minilm-quality-gates-n8.json"
)

#: R0217 measured ~0.197 GPU-h per 2M cell. Three cells plus a graph build and a
#: three-cell panel is ~0.65 GPU-h; the cap is the round's registered 1.5.
GPU_HOURS_CAP = 1.5
GRAPH_NODE_P90_WALL_S = 900.0
TRAIN_NODE_P90_WALL_S = 1_800.0
COMPARE_NODE_P90_WALL_S = 900.0

#: Used only to prove the config construction at prepare time. It is never the
#: horizon any cell trains at; the node derives that from the sealed cuVS graph.
CONSTRUCTION_PROBE_EDGES = 48_000_000


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
        raise RuntimeError("R0223 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0223 round must declare its required reviews")
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
        raise RuntimeError("R0223 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0223_cuvs_graph_map.py",
        "tests/test_round0223_cpu_smoke.py",
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
        "schema": "round0223-release-cpu-smoke-v1",
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
            "R0217-template config construction with the graph swapped, "
            "treatment-invariant digest equality, ceil-derived dose, fuzzy "
            "graph validation, published-map validation, panel metric view, "
            "comparison arithmetic against a synthetic eight-cell family, "
            "config seal, post-fit accounting, checkpoint publish, reload and "
            "downstream panel call"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0223 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _sealed_r0216() -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(R0216_GRAPH_MANIFEST)
    manifest = prompt_contract.read_sealed(
        signature["canonical_path"], label="sealed R0216 substrate+graph receipt"
    )
    checks = manifest.get("graph_checks") or {}
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if (
        manifest.get("schema") != R0216_GRAPH_SCHEMA
        or manifest.get("capability") != R0216_GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or int(checks.get("zero_degree_rows", -1)) != 0
        or edges != R0216_SEALED_DIRECTED_EDGES
    ):
        raise RuntimeError("R0223 sealed R0216 substrate+graph contract changed")
    return signature, manifest


def _construction_proof(
    *,
    substrate_signature: dict[str, Any],
    r0216_graph_signature: dict[str, Any],
    r0216_manifest_signature: dict[str, Any],
) -> dict[str, Any]:
    """Prove, before any GPU work, that only the registered paths move."""
    template, _sha = r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=r0216_graph_signature,
        graph_manifest_signature=r0216_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=R0216_SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )
    template_invariant = treatment_invariant_sha256(template)
    probe_graph_signature = {
        "kind": "file",
        "canonical_path": "/data/latent-basemap/runs/round-0223/<pending>/edges-k15-fuzzy.npz",
        "bytes": 1,
        "sha256": "0" * 64,
    }
    probe_manifest_signature = {
        "kind": "file",
        "canonical_path": "/data/latent-basemap/runs/round-0223/<pending>/cuvs-graph.json",
        "bytes": 1,
        "sha256": "1" * 64,
    }
    invariants = set()
    seed_digests = set()
    for seed in SEEDS:
        config, _config_sha, invariant = train_config(
            seed=seed,
            graph_signature=probe_graph_signature,
            graph_manifest_signature=probe_manifest_signature,
            substrate_signature=substrate_signature,
            r0216_graph_signature=r0216_graph_signature,
            r0216_graph_manifest_signature=r0216_manifest_signature,
            graph_edges=CONSTRUCTION_PROBE_EDGES,
            rows=ROWS,
        )
        invariants.add(invariant)
        seed_digests.add(
            json.dumps(config["seed_family"]["this_capability"], sort_keys=True)
        )
    if invariants != {template_invariant}:
        raise RuntimeError(
            "R0223 construction does not reproduce R0217's treatment-invariant "
            f"digest: {sorted(invariants)} != {template_invariant}"
        )
    if len(seed_digests) != len(SEEDS):
        raise RuntimeError("R0223 cells do not carry distinct capabilities")
    return {
        "schema": "round0223-treatment-construction-proof-v1",
        "round_id": ROUND_ID,
        "template_round": "0217",
        "template_seed": TEMPLATE_SEED,
        "treatment_invariant_sha256": template_invariant,
        "graph_bearing_paths": [".".join(path) for path in GRAPH_BEARING_PATHS],
        "graph_bearing_reasons": dict(GRAPH_BEARING_REASONS),
        "pipeline_stamp_label_carryover": PIPELINE_STAMP_LABEL_CARRYOVER,
        "construction_probe_edges": CONSTRUCTION_PROBE_EDGES,
        "construction_probe_note": (
            "a hypothetical edge count used only to prove the construction; the "
            "real horizon is derived inside each node from the sealed cuVS graph"
        ),
        "seeds": list(SEEDS),
    }


def prepare_round0223(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0223 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    r0216_manifest_signature, r0216_manifest = _sealed_r0216()
    substrate_signature = dict(r0216_manifest["substrate"])
    provenance_signature = dict(r0216_manifest["provenance"])
    r0216_graph_signature = dict(r0216_manifest["graph"])

    cuvs_graph_signature = expected_input_signature(
        R0220_CUVS_GRAPH_SIGNATURE["canonical_path"]
    )
    qualification_signature = expected_input_signature(
        R0220_QUALIFICATION_SIGNATURE["canonical_path"]
    )
    truth_signature = expected_input_signature(
        R0220_TRUTH_RECEIPT_SIGNATURE["canonical_path"]
    )
    for label, observed, registered in (
        ("cuVS graph", cuvs_graph_signature, R0220_CUVS_GRAPH_SIGNATURE),
        ("cuVS qualification", qualification_signature, R0220_QUALIFICATION_SIGNATURE),
        ("cuVS truth receipt", truth_signature, R0220_TRUTH_RECEIPT_SIGNATURE),
    ):
        if dict(observed) != dict(registered):
            raise RuntimeError(
                f"R0223 {label} is not the registered R0220 artifact: {observed!r}"
            )
    panel_signature = expected_input_signature(R0218_PANEL_EVIDENCE)
    gate_signature = expected_input_signature(R0222_GATE_ARTIFACT)

    proof = _construction_proof(
        substrate_signature=substrate_signature,
        r0216_graph_signature=r0216_graph_signature,
        r0216_manifest_signature=r0216_manifest_signature,
    )

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0223 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    proof_path = os.path.join(preflight, "treatment-construction-proof.json")
    atomic_write_new_json(proof_path, prompt_contract.seal(proof), immutable=True)

    expected_inputs = _dedupe([
        round_signature,
        r0216_manifest_signature,
        substrate_signature,
        provenance_signature,
        r0216_graph_signature,
        cuvs_graph_signature,
        qualification_signature,
        truth_signature,
        panel_signature,
        gate_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(proof_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    graph_output = os.path.join(artifacts, CUVS_GRAPH_CAPABILITY)
    graph_manifest_path = os.path.join(graph_output, "cuvs-graph.json")
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}

    graph_node = "build_cuvs_igd48_fuzzy_graph"
    jobs.append({
        "id": graph_node,
        "action": GRAPH_ACTION,
        "handler_module": "experiments.round0223_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [graph_output],
        "done_marker": os.path.join(artifacts, f"{graph_node}.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": GRAPH_NODE_P90_WALL_S,
        "capability": CUVS_GRAPH_CAPABILITY,
        "graph_manifest_signature": r0216_manifest_signature,
        "cuvs_graph_signature": cuvs_graph_signature,
        "cuvs_qualification_signature": qualification_signature,
        "truth_receipt_signature": truth_signature,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": False,
        },
    })
    p90[graph_node] = GRAPH_NODE_P90_WALL_S

    train_nodes = []
    for seed in SEEDS:
        capability = map_capability(seed)
        node = f"train_cuvs_igd48_seed{seed}"
        train_nodes.append(node)
        jobs.append({
            "id": node,
            "action": TRAIN_ACTION,
            "handler_module": "experiments.round0223_nodes",
            "handler_callable": "run_job",
            "deps": [graph_node],
            "outputs": [os.path.join(artifacts, capability)],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": TRAIN_NODE_P90_WALL_S,
            "training_seed": int(seed),
            "capability": capability,
            "cuvs_graph_manifest_signature": {
                "kind": "file",
                "canonical_path": graph_manifest_path,
            },
            "r0216_graph_signature": r0216_graph_signature,
            "r0216_graph_manifest_signature": r0216_manifest_signature,
            "treatment_invariant_sha256": proof["treatment_invariant_sha256"],
            "registered_dose_bound": REGISTERED_UPDATE_BOUND,
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
                "cpu_heavy": False,
            },
        })
        p90[node] = TRAIN_NODE_P90_WALL_S

    compare_node = "compare_cuvs_graph_map_panel"
    jobs.append({
        "id": compare_node,
        "action": COMPARE_ACTION,
        "handler_module": "experiments.round0223_nodes",
        "handler_callable": "run_job",
        "deps": list(train_nodes),
        "outputs": [os.path.join(artifacts, COMPARISON_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{compare_node}.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": COMPARE_NODE_P90_WALL_S,
        "capability": COMPARISON_CAPABILITY,
        "graph_manifest_signature": r0216_manifest_signature,
        "cuvs_graph_manifest_signature": {
            "kind": "file",
            "canonical_path": graph_manifest_path,
        },
        "panel_evidence": R0218_PANEL_EVIDENCE,
        "r0222_gate_signature": gate_signature,
        "cells": [
            {
                "seed": int(seed),
                "capability": map_capability(seed),
                "train_receipt": {
                    "kind": "file",
                    "canonical_path": os.path.join(
                        artifacts, map_capability(seed), "train-receipt.json"
                    ),
                },
            }
            for seed in SEEDS
        ],
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": False,
        },
    })
    p90[compare_node] = COMPARE_NODE_P90_WALL_S
    p90["total"] = sum(
        value for key, value in p90.items() if key != "total"
    )

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
        "schema": "round0223-minilm-mixed-2m-cuvs-graph-map-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            R0216_GRAPH_CAPABILITY,
            "cuvs-k15-graph-builder-qualification-v1",
            R0218_PANEL_CAPABILITY,
            "minilm-mixed-2m-quality-gates-n8-v1",
        ],
        "capabilities_produced": [
            CUVS_GRAPH_CAPABILITY,
            *MAP_CAPABILITIES,
            COMPARISON_CAPABILITY,
        ],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": p90,
        "scientific_contract": {
            "question": (
                "does a map trained on the R0220-qualified cuVS igd48 k15 graph "
                "score like the exact-graph family on the same frozen panel?"
            ),
            "population": "sealed R0216 queue-correction-3 mixed MiniLM 2M substrate",
            "graph": (
                f"cuVS nn-descent {CUVS_SETTING_ID} "
                f"(intermediate_graph_degree {CUVS_INTERMEDIATE_GRAPH_DEGREE}), "
                "symmetrised through R0216's identical fuzzy law"
            ),
            "control": (
                "R0222's sealed eight-cell exact-graph family (seeds 42-49) on "
                "the byte-identical R0218 high-D reference"
            ),
            "only_treatment_vs_control": "the k-NN graph",
            "hidden_dimension": HIDDEN_DIMENSION,
            "input_dimension": DIMENSION,
            "precision": USE_AMP,
            "seeds": list(SEEDS),
            "cells": len(SEEDS),
            "exact_family_seeds": list(R0222_POOLED_SEEDS),
            "treatment_source_round": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "treatment_invariant_sha256": proof["treatment_invariant_sha256"],
            "graph_bearing_paths": [".".join(path) for path in GRAPH_BEARING_PATHS],
            "pipeline_stamp_label_carryover": PIPELINE_STAMP_LABEL_CARRYOVER,
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "dose_rule": (
                "ceil(R0184_successful_updates * active_edges / "
                "R0184_directed_edges), derived in-node from the sealed cuVS "
                "graph's edge count"
            ),
            "registered_update_bound": REGISTERED_UPDATE_BOUND,
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "pending_n8_floor_status": FLOOR_STATUS,
            "gate_registerable_here": GATE_REGISTERABLE_HERE,
            "gate_release_claimed": GATE_RELEASE_CLAIMED,
            "equivalence_claimed": False,
            "evidence_limits": EVIDENCE_LIMITS,
            "evaluation_performed": True,
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
        "queue_manifest": prepare_round0223(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
