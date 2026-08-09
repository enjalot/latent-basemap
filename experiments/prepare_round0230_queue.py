#!/usr/bin/env python3
"""Prepare, but never launch, the R0230 n=13 seed-extension + panel queue.

Six nodes in one queue: five GPU trains (seeds 50-54) and one GPU panel node that
depends on all five and pools thirteen cells.

The script builds all five cell configs **here**, from R0217's own `train_config`,
proves each reproduces R0217's **published** seed-invariant digest, proves each
reconstructs R0217's canonical config **byte for byte** once the nine seed-bearing
paths are restored, proves the thirteen full-config digests are thirteen distinct
values, and stamps the shared digest into every job so each node re-derives it and
refuses to train if its own config drifted.

It also runs the predictive memory guard for every cell at prepare time and seals
the prediction — including `refused_a_priori` — so a refused cell is recorded as
data rather than as an absence.

The five train receipts do not exist when this queue is written, so the panel
node's references to them are **intra-queue**: a canonical path with no hash,
resolved by `_intra_queue_signature` inside the node. That is R0229's fix for the
failure that killed R0228's geometry node.
"""
from __future__ import annotations

import argparse
import glob
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
    train_config as r0217_train_config,
)
from basemap.round0221_minilm_2m_seed_extension import (
    TRAIN_SCHEMA as R0221_TRAIN_SCHEMA,
    capability_for_seed as r0221_capability_for_seed,
)
from basemap.round0230_minilm_2m_seed_extension_n13 import (
    CAPABILITIES,
    DEVICE_BUDGET_BYTES,
    DIMENSION,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    HOST_ANON_BUDGET_BYTES,
    HOST_RSS_LIMIT_GIB,
    IDENTITY_BOUND_AT_N13,
    MEMORY_POLICY,
    N_TARGET,
    POOLED_SEEDS,
    R0217_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    R0221_SEEDS,
    REGISTERED_ACHIEVED_DRAWS_PER_EDGE,
    REGISTERED_SUCCESSFUL_UPDATES,
    REGISTERED_UPDATE_BOUND,
    ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    SWAP_GROWTH_ABORT_BYTES,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    TEMPLATE_SEED,
    USE_AMP,
    assert_extension_differs_only_by_seed,
    assert_reconstructs_r0217_template,
    capability_for_seed,
    predict_cell_footprint,
    successful_updates_for_edges,
    train_config,
    validate_registered_dose,
)
from basemap.round0230_minilm_2m_panel_n13 import (
    ANCHOR_CORPUS_COUNTS,
    CENTROID_KS,
    CORPUS_SLUGS,
    DENSITY_V2_STATUS,
    HI_D_AGREEMENT,
    PANEL_CAPABILITY,
    PANEL_CAPABILITY_N13,
    PANEL_METRICS,
    PANEL_SCHEMA,
    POOLED_CELL_SOURCES,
    REFERENCE_BYTES,
    REFERENCE_CONTENT_SHA256,
    REFERENCE_KEY,
    REFERENCE_SHA256,
)
from experiments.round0230_nodes import PANEL_ACTION, TRAIN_ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0230"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0230-2026-08-09.md")

R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
R0217_ARTIFACTS = "/data/latent-basemap/runs/round-0217/queue-correction-1/artifacts"
R0218_ARTIFACTS = (
    f"/data/latent-basemap/runs/round-0218/queue/artifacts/{PANEL_CAPABILITY}"
)
R0218_PANEL = os.path.join(R0218_ARTIFACTS, "seed-family-panel.json")
R0221_ARTIFACTS = "/data/latent-basemap/runs/round-0221/queue/artifacts"
R0222_GATE = (
    "/data/latent-basemap/runs/round-0222/queue/artifacts/"
    "minilm-mixed-2m-quality-gates-n8-v1/minilm-quality-gates-n8.json"
)
R0223_COMPARISON = (
    "/data/latent-basemap/runs/round-0223/queue-correction-3/artifacts/"
    "minilm-mixed-2m-cuvs-graph-map-comparison-v1/cuvs-graph-map-comparison.json"
)

#: R0221 measured 0.19765 GPU-h per cell (711.2-713.6 s) under this identical
#: treatment, so five cells is ~0.99 GPU-h; R0222 scored four cells against a
#: loaded reference in 0.005 GPU-h and this node scores five. The registered cap
#: is the round's 2.5 h, deliberately above the estimate rather than tight to it.
GPU_HOURS_CAP = 2.5
TRAIN_P90_WALL_S = 1_800.0
PANEL_P90_WALL_S = 1_800.0


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
        raise RuntimeError("R0230 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0230 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _upstream_review_state(required: list[str]) -> dict[str, Any]:
    """Record, rather than assume, the state of every required review."""
    state: dict[str, Any] = {}
    contingent: list[str] = []
    for round_id in required:
        reviews = []
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))):
            frontmatter = _frontmatter(path)
            reviews.append({
                "file": os.path.basename(path),
                "status": frontmatter.get("status"),
                "sha256": expected_input_signature(path)["sha256"],
            })
        accepted = [item for item in reviews if item["status"] == "accepted"]
        state[round_id] = {
            "reviews_present": reviews,
            "accepted_reviews": len(accepted),
        }
        if not accepted:
            contingent.append(round_id)
    return {
        "required_reviews": list(required),
        "by_round": state,
        "rounds_without_an_accepted_review": contingent,
        "claims_contingent_on": contingent,
        "note": (
            "Review is post-hoc: it blocks the downstream claim, not the launch. "
            "R0230 registers no floor, so nothing here is released by running; "
            "R0231's floors carry the contingency."
        ),
    }


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0230 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0230_seed_extension_n13.py",
        "tests/test_round0230_cpu_smoke.py",
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
        timeout=300,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0230-release-cpu-smoke-v1",
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
            "byte-for-byte reconstruction of R0217's canonical config, the "
            "predictive memory guard including its refusal branch, the registered "
            "ceil-derived dose assertion, the train config seal, post-fit "
            "accounting, checkpoint publish, full-population reload and finiteness "
            "check, the reference-identity assertions and the n=13 pooling"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0230 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
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
        raise RuntimeError("R0230 sealed R0216 substrate+graph contract changed")
    if int(checks.get("zero_degree_rows", -1)) != 0:
        raise RuntimeError("R0230 requires a graph with zero degree-zero rows")
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if edges != SEALED_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0230 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    return signature, manifest, edges


def _prior_family(
    manifest_signature: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Read the eight sealed cells R0230 extends: R0217's four and R0221's four."""
    invariants: set[str] = set()
    config_hashes: dict[str, str] = {}
    model_hashes: dict[str, str] = {}
    sources = (
        [(seed, "0217", R0217_ARTIFACTS, R0217_TRAIN_SCHEMA, r0217_capability_for_seed)
         for seed in R0217_SEEDS]
        + [(seed, "0221", R0221_ARTIFACTS, R0221_TRAIN_SCHEMA, r0221_capability_for_seed)
           for seed in R0221_SEEDS]
    )
    for seed, round_id, root, schema, capability_fn in sources:
        capability = capability_fn(seed)
        receipt_path = os.path.join(root, capability, "train-receipt.json")
        receipt = prompt_contract.read_sealed(
            receipt_path, label=f"R{round_id} seed-{seed} train receipt"
        )
        train_checks = receipt.get("train_checks") or {}
        if (
            receipt.get("schema") != schema
            or receipt.get("round_id") != round_id
            or receipt.get("capability") != capability
            or int(receipt.get("training_seed", -1)) != seed
            or int(receipt.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
            or receipt.get("training_performed") is not True
            or not train_checks
            or not all(bool(value) for value in train_checks.values())
        ):
            raise RuntimeError(f"R{round_id} seed-{seed} train receipt contract changed")
        if (
            dict(receipt.get("substrate") or {}) != dict(manifest["substrate"])
            or dict(receipt.get("graph_manifest") or {}) != manifest_signature
        ):
            raise RuntimeError(
                f"R{round_id} seed-{seed} was not trained on the substrate R0230 "
                "extends"
            )
        if int(receipt.get("optimizer_updates", -1)) != REGISTERED_SUCCESSFUL_UPDATES:
            raise RuntimeError(
                f"R{round_id} seed-{seed} horizon is not the registered "
                f"{REGISTERED_SUCCESSFUL_UPDATES}"
            )
        invariants.add(str(receipt["seed_invariant_sha256"]))
        config_hashes[str(seed)] = str(receipt["production_config_sha256"])
        model_hashes[str(seed)] = str(receipt["model"]["sha256"])
    if invariants != {R0217_SEED_INVARIANT_SHA256}:
        raise RuntimeError(
            "the eight prior cells do not carry one published seed-invariant "
            f"digest: {sorted(invariants)}"
        )
    if len(set(model_hashes.values())) != 8:
        raise RuntimeError("the eight prior cells contain a duplicated checkpoint")
    return {
        "seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "config_sha256_by_seed": config_hashes,
        "model_sha256_by_seed": model_hashes,
    }


def _sealed_panel() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """R0218's panel, its reference and its centroids — the frozen panel bytes."""
    signature = expected_input_signature(R0218_PANEL)
    panel = prompt_contract.read_sealed(
        R0218_PANEL, label="R0218 MiniLM 2M four-seed panel"
    )
    checks = panel.get("execution_checks") or {}
    if (
        panel.get("schema") != PANEL_SCHEMA
        or panel.get("round_id") != "0218"
        or panel.get("capabilities") != [PANEL_CAPABILITY]
        or panel.get("seeds") != list(R0217_SEEDS)
        or panel.get("metrics") != list(PANEL_METRICS)
        or panel.get("seed_invariant_sha256") != R0217_SEED_INVARIANT_SHA256
        or panel.get("evaluation_performed") is not True
        or panel.get("gate_registered") is not False
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise RuntimeError("R0218 panel receipt contract changed")
    reference = dict(panel["shared_high_d_reference"])
    if expected_input_signature(reference["canonical_path"]) != reference:
        raise RuntimeError(
            "R0218's published high-D reference bytes changed; the thirteen cells "
            "would not be poolable"
        )
    if (
        int(reference["bytes"]) != REFERENCE_BYTES
        or str(reference["sha256"]) != REFERENCE_SHA256
        or str(panel["high_d_reference_key"]) != REFERENCE_KEY
        or str(panel["high_d_reference_content_sha256"]) != REFERENCE_CONTENT_SHA256
    ):
        raise RuntimeError(
            "R0218's reference identity is not the registered one; STOP — the "
            "thirteen cells are not poolable"
        )
    if dict(panel["anchor_corpus_counts"]) != dict(ANCHOR_CORPUS_COUNTS):
        raise RuntimeError("R0218's anchor corpus counts are not the registered ones")
    for seed in R0217_SEEDS:
        numerators = panel["cells"][str(seed)]["panel"]["purity_numerators"]
        for key, expected in HI_D_AGREEMENT.items():
            if float(numerators[key]["hi_D_agreement"]) != float(expected):
                raise RuntimeError(
                    f"R0218 seed-{seed} hi-D agreement {key} is not {expected}"
                )
    inputs = [signature, reference]
    declared = dict(panel.get("centroids") or {})
    if set(declared) != {str(k) for k in CENTROID_KS}:
        raise RuntimeError("R0218 centroid vocabularies changed")
    for k in CENTROID_KS:
        centroid = dict(declared[str(k)])
        if expected_input_signature(centroid["canonical_path"]) != centroid:
            raise RuntimeError(f"R0218 published centroids k{k} bytes changed")
        inputs.append(centroid)
    return panel, signature, inputs


def _prior_scored_cells() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """R0222's eight pooled cells and R0223's eight raw ratios, read from bytes."""
    r0222_signature = expected_input_signature(R0222_GATE)
    r0222 = prompt_contract.read_sealed(R0222_GATE, label="R0222 sealed n=8 gate")
    r0223_signature = expected_input_signature(R0223_COMPARISON)
    r0223 = prompt_contract.read_sealed(
        R0223_COMPARISON, label="R0223 sealed cuVS comparison"
    )
    want = {str(seed) for seed in R0217_SEEDS} | {str(seed) for seed in R0221_SEEDS}
    if set(r0222.get("pooled_panel_metric_cells") or {}) != want:
        raise RuntimeError("R0222 does not carry exactly the eight pooled cells")
    if set(r0223.get("exact_family_purity_ratios") or {}) != want:
        raise RuntimeError("R0223 does not carry exactly the eight raw purity ratios")
    if (
        str(r0222.get("high_d_reference_key")) != REFERENCE_KEY
        or str(r0223.get("high_d_reference_key")) != REFERENCE_KEY
    ):
        raise RuntimeError(
            "R0222 or R0223 scored against a different high-D reference; STOP"
        )
    return r0222_signature, r0223_signature, {
        "r0222_pooled_panel_metric_cells": dict(r0222["pooled_panel_metric_cells"]),
        "r0223_exact_family_purity_ratios": dict(r0223["exact_family_purity_ratios"]),
    }


def prepare_round0230(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0230 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    graph_manifest_signature, graph_manifest, edges = _sealed_graph()
    updates = successful_updates_for_edges(edges)
    if updates > REGISTERED_UPDATE_BOUND:
        raise RuntimeError(
            f"R0230 derived horizon {updates} exceeds the registered bound "
            f"{REGISTERED_UPDATE_BOUND}"
        )
    dose = validate_registered_dose(updates=updates, edge_count=edges)
    prior = _prior_family(graph_manifest_signature, graph_manifest)
    panel, panel_signature, panel_inputs = _sealed_panel()
    r0222_signature, r0223_signature, prior_scored = _prior_scored_cells()
    review_state = _upstream_review_state(list(required_reviews))

    substrate_signature = dict(graph_manifest["substrate"])
    graph_signature = dict(graph_manifest["graph"])
    provenance_signature = dict(graph_manifest["provenance"])

    template, _template_sha = r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=edges,
        rows=ROWS,
    )
    configs: dict[int, dict[str, Any]] = {}
    reconstructions: dict[str, Any] = {}
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
        reconstructions[str(seed)] = assert_reconstructs_r0217_template(
            config, template
        )
    family = assert_extension_differs_only_by_seed(
        configs, expected_seed_invariant=prior["seed_invariant_sha256"]
    )
    if not family["matches_r0217_published_seed_invariant"]:
        raise RuntimeError(
            "R0230 seed-invariant digest does not match R0217's published value"
        )
    pooled_config_hashes = {
        **prior["config_sha256_by_seed"],
        **family["per_seed_config_sha256"],
    }
    if len(set(pooled_config_hashes.values())) != len(POOLED_SEEDS):
        raise RuntimeError(
            "R0230 cell configs collide with the prior eight: the thirteen cells "
            "must be thirteen distinct configs sharing one seed-invariant digest"
        )

    predictions = {str(seed): predict_cell_footprint(seed) for seed in SEEDS}
    refused = sorted(
        int(seed) for seed, item in predictions.items() if item["refused_a_priori"]
    )

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0230 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    family_path = os.path.join(preflight, "seed-extension-n13-identity.json")
    atomic_write_new_json(
        family_path,
        prompt_contract.seal({
            "schema": "round0230-seed-extension-n13-config-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "sealed_directed_edges": edges,
            "successful_positive_lr_updates": updates,
            "dose_registration": dose,
            "prior_family": prior,
            "family": family,
            "byte_for_byte_reconstruction_of_r0217": reconstructions,
            "pooled_seed_family": list(POOLED_SEEDS),
            "n_pooled": len(POOLED_SEEDS),
            "identity_bound_at_n_pooled": IDENTITY_BOUND_AT_N13,
            "pooled_config_sha256_by_seed": pooled_config_hashes,
            "memory_predictions": predictions,
            "refused_a_priori": refused,
            "memory_policy": MEMORY_POLICY,
            "configs": {str(seed): configs[seed] for seed in SEEDS},
        }),
        immutable=True,
    )
    shared_inputs = _dedupe([
        round_signature,
        graph_manifest_signature,
        substrate_signature,
        graph_signature,
        provenance_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(family_path),
    ])
    panel_only_inputs = _dedupe([
        *panel_inputs,
        r0222_signature,
        r0223_signature,
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}
    panel_cells: list[dict[str, Any]] = []
    for seed in SEEDS:
        if predictions[str(seed)]["refused_a_priori"]:
            continue
        capability = capability_for_seed(seed)
        node = f"train_minilm_mixed_2m_seed{seed}"
        output = os.path.join(artifacts, capability)
        jobs.append({
            "id": node,
            "action": TRAIN_ACTION,
            "handler_module": "experiments.round0230_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": shared_inputs,
            "p90_wall_s": TRAIN_P90_WALL_S,
            "training_seed": int(seed),
            "capability": capability,
            "graph_manifest": GRAPH_MANIFEST,
            "graph_manifest_signature": graph_manifest_signature,
            "family_seed_invariant_sha256": family["seed_invariant_sha256"],
            "registered_dose_bound": REGISTERED_UPDATE_BOUND,
            "memory_prediction": predictions[str(seed)],
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
                "cpu_heavy": False,
            },
        })
        p90[node] = TRAIN_P90_WALL_S
        panel_cells.append({
            "seed": int(seed),
            "capability": capability,
            # Intra-queue: these bytes do not exist yet, so the reference carries
            # a path and no hash and the node resolves it (R0229's fix).
            "train_receipt": {
                "kind": "file",
                "canonical_path": os.path.join(output, "train-receipt.json"),
            },
        })
    if not jobs:
        raise RuntimeError("R0230 refused every cell a priori; nothing to run")

    panel_node = "score_minilm_mixed_2m_panel_n13"
    jobs.append({
        "id": panel_node,
        "action": PANEL_ACTION,
        "handler_module": "experiments.round0230_nodes",
        "handler_callable": "run_job",
        "deps": [job["id"] for job in jobs],
        "outputs": [os.path.join(artifacts, PANEL_CAPABILITY_N13)],
        "done_marker": os.path.join(artifacts, f"{panel_node}.done.json"),
        "expected_inputs": _dedupe([*shared_inputs, *panel_only_inputs]),
        "p90_wall_s": PANEL_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "panel_evidence": R0218_PANEL,
        "r0222_gate_signature": r0222_signature,
        "r0223_comparison_signature": r0223_signature,
        "cells": panel_cells,
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": False,
        },
    })
    p90[panel_node] = PANEL_P90_WALL_S
    p90["total"] = sum(value for key, value in p90.items() if key != "total")

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
        "schema": "round0230-minilm-mixed-2m-seed-extension-n13-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            GRAPH_CAPABILITY,
            PANEL_CAPABILITY,
            "minilm-mixed-2m-quality-gates-n8-v1",
            "minilm-mixed-2m-cuvs-graph-map-comparison-v1",
            *(r0217_capability_for_seed(seed) for seed in R0217_SEEDS),
            *(r0221_capability_for_seed(seed) for seed in R0221_SEEDS),
        ],
        "capabilities_produced": [*CAPABILITIES, PANEL_CAPABILITY_N13],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": p90,
        "scientific_contract": {
            "question": (
                "does extending the exact-graph MiniLM 2M family from eight seeds "
                "to thirteen, under R0217's treatment with the seed as the only "
                "free variable, produce a family in which a defining cell CAN fail "
                "its own floor?"
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
            "n_pooled": N_TARGET,
            "identity_bound_at_n_pooled": IDENTITY_BOUND_AT_N13,
            "identity_bound_note": (
                "max|x - xbar|/s <= (n-1)/sqrt(n) = 3.3282 at n = 13, which "
                "exceeds mean-2s's 2.0, the one-sided 95/95 factor 2.6705 and "
                "Howe's two-sided 3.1008. At n = 4 the bound is 1.5 and at n = 8 "
                "it is 2.4749, which is why R0219's '4/4 pass' and R0225's "
                "'0 failures under k = 3.187' were theorems."
            ),
            "treatment_source_round": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "capabilities_by_seed": {
                str(seed): capability_for_seed(seed) for seed in SEEDS
            },
            "family_seed_invariant_sha256": family["seed_invariant_sha256"],
            "r0217_published_seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
            "per_seed_config_sha256": family["per_seed_config_sha256"],
            "masked_config_identity": family["masked_config_identity"],
            "byte_for_byte_reconstruction_of_r0217": reconstructions,
            "pooled_config_sha256_by_seed": pooled_config_hashes,
            "prior_model_sha256_by_seed": prior["model_sha256_by_seed"],
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
            "device_budget_bytes": DEVICE_BUDGET_BYTES,
            "host_anonymous_budget_bytes": HOST_ANON_BUDGET_BYTES,
            "swap_growth_abort_bytes": SWAP_GROWTH_ABORT_BYTES,
            "memory_policy": MEMORY_POLICY,
            "memory_predictions": predictions,
            "refused_a_priori": refused,
            "full_population_transform_rows": ROWS,
            "panel_config_source": "accepted R0113 panel_config()",
            "panel_metrics": list(PANEL_METRICS),
            "corpus_ffr_slices": list(CORPUS_SLUGS),
            "shared_high_d_reference": dict(panel["shared_high_d_reference"]),
            "reference_source_round": "0218",
            "reference_must_be_byte_identical_to_r0218": True,
            "reference_key": REFERENCE_KEY,
            "reference_content_sha256": REFERENCE_CONTENT_SHA256,
            "hi_d_agreement_required": dict(HI_D_AGREEMENT),
            "anchor_corpus_counts": dict(ANCHOR_CORPUS_COUNTS),
            "prior_cells_read_not_rescored": dict(POOLED_CELL_SOURCES),
            "prior_scored_cells": prior_scored,
            "density_v2_status": DENSITY_V2_STATUS,
            "upstream_review_state": review_state,
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
        "queue_manifest": prepare_round0230(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
