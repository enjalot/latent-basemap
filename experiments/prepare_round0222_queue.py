#!/usr/bin/env python3
"""Prepare, but never launch, the R0222 n=8 MiniLM gate-registration queue.

One GPU node. The script binds, by hash and before the GPU is touched: R0216's
sealed substrate receipt, R0218's sealed panel plus **its** shared high-D
reference and frozen centroid arrays, the four R0221 train receipts and their
maps, and R0161's and R0193's sealed quality-gate artifacts — the last two
because R0222 refutes R0219's `density_v2` claim by reading those artifacts
rather than by quoting them.

It proves here that the eight cells share one seed-invariant config digest and
carry eight distinct checkpoints, and it records — rather than asserts — the
state of R0221's and R0218's independent reviews.
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
from basemap.round0218_minilm_2m_panel import (
    CAPABILITY as PANEL_CAPABILITY,
    CENTROID_KS,
    CORPUS_SLUGS,
    EVALUATION_SCHEMA as PANEL_SCHEMA,
    GRAPH_CAPABILITY,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    HOST_RSS_LIMIT_GIB,
    PANEL_METRICS,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS as R0218_SEEDS,
)
from basemap.round0221_minilm_2m_seed_extension import (
    DIMENSION,
    GRAPH_K,
    POOLED_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    SEEDS as R0221_SEEDS,
    TRAIN_SCHEMA as R0221_TRAIN_SCHEMA,
    capability_for_seed as r0221_capability_for_seed,
)
from basemap.round0222_minilm_2m_gate_n8 import (
    ACCEPTED_SIX_METRIC_SET,
    CAPABILITY,
    FORMULA,
    GATE_METRICS,
    MULTIPLIER,
    N_REQUIRED,
    PANEL_EXTENSION_CAPABILITY,
    PRECEDENT_CAPABILITIES,
    PRECEDENT_GATE_ARTIFACTS,
    PRECISION_NOTE,
    RETRACTED_CLAIM,
    ROUND_ID,
    SD_DDOF,
    UNAVAILABLE_METRICS,
)
from experiments.round0222_nodes import ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0222"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0222-2026-08-08.md")
R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
R0218_ARTIFACTS = (
    f"/data/latent-basemap/runs/round-0218/queue/artifacts/{PANEL_CAPABILITY}"
)
R0218_PANEL = os.path.join(R0218_ARTIFACTS, "seed-family-panel.json")
R0221_ARTIFACTS = "/data/latent-basemap/runs/round-0221/queue/artifacts"
#: R0218 scored four cells with a freshly built reference in 0.00914 GPU-h.
#: R0222 scores four cells against a *loaded* reference, so it is strictly
#: cheaper; the cap is the round's registered 0.5 h.
GPU_HOURS_CAP = 0.5
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
        raise RuntimeError("R0222 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0222 round must declare its required reviews")
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
        "gate_release_contingent_on": contingent,
        "note": (
            "The slim protocol allows an already-registered experiment to run "
            "before its upstream review lands; it blocks the downstream claim. "
            "The n=8 floors are registered here and are released only once every "
            "listed round carries an accepted review."
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
        raise RuntimeError("R0222 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0222_minilm_2m_gate_n8.py",
        "tests/test_round0222_cpu_smoke.py",
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
        "schema": "round0222-release-cpu-smoke-v1",
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
            "R0221 map authentication, byte-identical reference binding, panel "
            "metric and per-corpus views, the n=8 and n=4 gate arithmetic, the "
            "jackknife, the precedent-artifact retraction check and the seal"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0222 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _sealed_substrate() -> tuple[dict[str, Any], dict[str, Any], int]:
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
        or int(checks.get("zero_degree_rows", -1)) != 0
    ):
        raise RuntimeError("R0222 sealed R0216 substrate+graph contract changed")
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if edges != SEALED_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0222 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    return signature, manifest, edges


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
        or panel.get("seeds") != list(R0218_SEEDS)
        or panel.get("metrics") != list(PANEL_METRICS)
        or panel.get("seed_invariant_sha256") != R0217_SEED_INVARIANT_SHA256
        or panel.get("evaluation_performed") is not True
        or panel.get("gate_registered") is not False
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise RuntimeError("R0218 panel receipt contract changed")
    if set(panel.get("panel_metric_cells") or {}) != {
        str(seed) for seed in R0218_SEEDS
    }:
        raise RuntimeError("R0218 panel does not carry its four metric cells")
    reference = dict(panel["shared_high_d_reference"])
    if expected_input_signature(reference["canonical_path"]) != reference:
        raise RuntimeError(
            "R0218's published high-D reference bytes changed; the eight cells "
            "would not be comparable"
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


def _r0221_maps(
    manifest_signature: dict[str, Any], manifest: dict[str, Any], panel: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Bind the four R0221 maps and prove the pooled eight are one family."""
    cells: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    invariants: set[str] = set()
    model_hashes: dict[str, str] = {}
    for seed in R0221_SEEDS:
        capability = r0221_capability_for_seed(seed)
        receipt_path = os.path.join(R0221_ARTIFACTS, capability, "train-receipt.json")
        receipt_signature = expected_input_signature(receipt_path)
        receipt = prompt_contract.read_sealed(
            receipt_path, label=f"R0221 seed-{seed} train receipt"
        )
        train_checks = receipt.get("train_checks") or {}
        if (
            receipt.get("schema") != R0221_TRAIN_SCHEMA
            or receipt.get("round_id") != "0221"
            or receipt.get("treatment_config_round_id") != "0217"
            or receipt.get("capability") != capability
            or int(receipt.get("training_seed", -1)) != seed
            or int(receipt.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
            or receipt.get("gate_registerable_here") is not False
            or str(receipt.get("seed_invariant_sha256") or "")
            != R0217_SEED_INVARIANT_SHA256
            or not train_checks
            or not all(bool(value) for value in train_checks.values())
        ):
            raise RuntimeError(f"R0221 seed-{seed} train receipt contract changed")
        if (
            dict(receipt.get("substrate") or {}) != dict(manifest["substrate"])
            or dict(receipt.get("graph_manifest") or {}) != manifest_signature
        ):
            raise RuntimeError(
                f"R0221 seed-{seed} was not trained on the substrate R0222 scores"
            )
        model_signature = dict(receipt["model"])
        if expected_input_signature(model_signature["canonical_path"]) != model_signature:
            raise RuntimeError(f"R0221 seed-{seed} published map bytes changed")
        invariants.add(str(receipt["seed_invariant_sha256"]))
        model_hashes[str(seed)] = str(model_signature["sha256"])
        cells.append({
            "seed": int(seed),
            "capability": capability,
            "train_receipt": receipt_signature,
        })
        inputs.extend([receipt_signature, model_signature])
    for seed in R0218_SEEDS:
        model_hashes[str(seed)] = str(panel["cells"][str(seed)]["model"]["sha256"])
    if invariants != {R0217_SEED_INVARIANT_SHA256}:
        raise RuntimeError(
            "R0222 refuses to pool an incommensurate family: R0221's cells do "
            f"not carry R0217's seed-invariant digest ({sorted(invariants)})"
        )
    if len(set(model_hashes.values())) != len(POOLED_SEEDS):
        raise RuntimeError("R0222 pooled family contains a duplicated checkpoint")
    family = {
        "seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "model_sha256_by_seed": model_hashes,
        "pooled_seed_family": list(POOLED_SEEDS),
        "n": len(POOLED_SEEDS),
    }
    return cells, inputs, family


def _precedent_gates() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """R0161's and R0193's sealed gate artifacts: the retraction's evidence."""
    signatures: dict[str, Any] = {}
    floors: dict[str, float] = {}
    inputs: list[dict[str, Any]] = []
    for round_id, path in sorted(PRECEDENT_GATE_ARTIFACTS.items()):
        signature = expected_input_signature(path)
        artifact = prompt_contract.read_sealed(
            path, label=f"R{round_id} sealed quality-gate artifact"
        )
        gates = artifact.get("gates") or {}
        if (
            artifact.get("capability") != PRECEDENT_CAPABILITIES[round_id]
            or tuple(sorted(gates)) != tuple(sorted(ACCEPTED_SIX_METRIC_SET))
            or "density_v2" not in gates
        ):
            raise RuntimeError(
                f"R{round_id} gate artifact does not carry the accepted "
                "six-metric set; R0222's retraction premise is wrong and the "
                "round must not run"
            )
        signatures[round_id] = signature
        floors[round_id] = float(gates["density_v2"]["floor"])
        inputs.append(signature)
    return signatures, floors, inputs


def prepare_round0222(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0222 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    manifest_signature, manifest, edges = _sealed_substrate()
    substrate_signature = dict(manifest["substrate"])
    provenance_signature = dict(manifest["provenance"])
    panel, panel_signature, panel_inputs = _sealed_panel()
    cells, map_inputs, family = _r0221_maps(manifest_signature, manifest, panel)
    precedent_signatures, precedent_floors, precedent_inputs = _precedent_gates()
    review_state = _upstream_review_state(list(required_reviews))

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0222 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    family_path = os.path.join(preflight, "pooled-family-identity.json")
    atomic_write_new_json(
        family_path,
        prompt_contract.seal({
            "schema": "round0222-pooled-family-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "sealed_directed_edges": edges,
            "substrate": substrate_signature,
            "provenance": provenance_signature,
            "panel_evidence": panel_signature,
            "shared_high_d_reference": dict(panel["shared_high_d_reference"]),
            "high_d_reference_key": panel["high_d_reference_key"],
            "high_d_reference_content_sha256": panel[
                "high_d_reference_content_sha256"
            ],
            "family": family,
            "cells": cells,
            "precedent_density_v2_floors": precedent_floors,
            "upstream_review_state": review_state,
        }),
        immutable=True,
    )
    expected_inputs = _dedupe([
        round_signature,
        manifest_signature,
        substrate_signature,
        provenance_signature,
        *panel_inputs,
        *map_inputs,
        *precedent_inputs,
        expected_input_signature(smoke_path),
        expected_input_signature(family_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    node = "register_minilm_mixed_2m_quality_gates_n8"
    job = {
        "id": node,
        "action": ACTION,
        "handler_module": "experiments.round0222_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{node}.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": NODE_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": manifest_signature,
        "panel_evidence": R0218_PANEL,
        "cells": cells,
        "precedent_gate_signatures": precedent_signatures,
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
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
        "schema": "round0222-minilm-mixed-2m-quality-gates-n8-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-evaluation",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            GRAPH_CAPABILITY,
            PANEL_CAPABILITY,
            *(r0221_capability_for_seed(seed) for seed in R0221_SEEDS),
        ],
        "capabilities_produced": [PANEL_EXTENSION_CAPABILITY, CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "p90_gpu_seconds": {node: NODE_P90_WALL_S, "total": NODE_P90_WALL_S},
        "scientific_contract": {
            "question": (
                "what are the mean - 2 sigma floors for this universe at n = 8, "
                "over the accepted metric set with density_v2 included?"
            ),
            "population": "sealed R0216 2,000,000-row mixed MiniLM substrate",
            "sealed_directed_edges": edges,
            "seed_family": list(POOLED_SEEDS),
            "n": len(POOLED_SEEDS),
            "n_required_by_plan": N_REQUIRED,
            "source_rounds": {"0217": list(R0218_SEEDS), "0221": list(R0221_SEEDS)},
            "formula": FORMULA,
            "multiplier": MULTIPLIER,
            "sample_standard_deviation_ddof": SD_DDOF,
            "precision_note": PRECISION_NOTE,
            "accepted_six_metric_set": list(ACCEPTED_SIX_METRIC_SET),
            "panel_metrics_available": list(PANEL_METRICS),
            "gate_metrics": list(GATE_METRICS),
            "unavailable_metrics": sorted(UNAVAILABLE_METRICS),
            "excluded_by_judgement": [],
            "density_v2_is_gated": True,
            "retracted_claim": RETRACTED_CLAIM,
            "precedent_density_v2_floors": precedent_floors,
            "corpus_ffr_slices": list(CORPUS_SLUGS),
            "panel_config_source": "accepted R0113 panel_config()",
            "shared_high_d_reference": dict(panel["shared_high_d_reference"]),
            "reference_source_round": "0218",
            "reference_must_be_byte_identical_to_r0218": True,
            "family_seed_invariant_sha256": family["seed_invariant_sha256"],
            "model_sha256_by_seed": family["model_sha256_by_seed"],
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "upstream_review_state": review_state,
            "gate_registerable_here": True,
            "evaluation_performed": True,
            "training_performed": False,
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
        "queue_manifest": prepare_round0222(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
