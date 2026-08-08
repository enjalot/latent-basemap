#!/usr/bin/env python3
"""Prepare, but never launch, the R0218 MiniLM 2M four-seed panel queue.

One GPU node, four scored cells. The script binds R0216's sealed substrate
receipt and all four accepted R0217 train receipts by hash, proves here — before
the GPU is touched — that the four maps were trained on the exact bytes this
panel will score against and that they form one commensurate family, and stamps
the resulting evidence into the queue's scientific contract.
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
from basemap.round0218_minilm_2m_panel import (
    CAPABILITY,
    CENTROID_KS,
    CORPUS_SLUGS,
    DIAGNOSTIC_METRICS,
    DIMENSION,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    HOST_RSS_LIMIT_GIB,
    MAP_TRAIN_SCHEMA,
    PANEL_METRICS,
    ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    map_capability,
)
from experiments.round0218_nodes import ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0218"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0218-2026-08-08.md")
#: R0216's terminal, corpus-spanning queue. `queue-correction-2` is superseded.
R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
#: R0217's terminal queue: the four released maps.
R0217_ARTIFACTS = "/data/latent-basemap/runs/round-0217/queue-correction-1/artifacts"
#: One shared reference plus four low-D passes and four full-population
#: transforms. Expected ~0.3 GPU-h; the cap is deliberately loose because no
#: MiniLM panel has been timed on this box.
GPU_HOURS_CAP = 1.0
NODE_P90_WALL_S = 3_600.0


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
        raise RuntimeError("R0218 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0218 round must declare its required reviews")
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
        raise RuntimeError("R0218 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0218_minilm_2m_panel.py",
        "tests/test_round0218_cpu_smoke.py",
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
        "schema": "round0218-release-cpu-smoke-v1",
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
            "sealed-substrate binding, map authentication, panel metric and "
            "per-corpus FFR views, execution checks, family evidence and the "
            "receipt seal"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0218 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
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
        raise RuntimeError("R0218 sealed R0216 substrate+graph contract changed")
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if edges != SEALED_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0218 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    return signature, manifest, edges


def _accepted_maps(
    manifest_signature: dict[str, Any], manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Bind the four R0217 maps and prove the family is one commensurate unit."""
    cells: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    invariants: set[str] = set()
    model_hashes: dict[str, str] = {}
    for seed in SEEDS:
        capability = map_capability(seed)
        receipt_path = os.path.join(R0217_ARTIFACTS, capability, "train-receipt.json")
        receipt_signature = expected_input_signature(receipt_path)
        receipt = prompt_contract.read_sealed(
            receipt_path, label=f"R0217 seed-{seed} train receipt"
        )
        train_checks = receipt.get("train_checks") or {}
        if (
            receipt.get("schema") != MAP_TRAIN_SCHEMA
            or receipt.get("round_id") != "0217"
            or receipt.get("capability") != capability
            or int(receipt.get("training_seed", -1)) != seed
            or int(receipt.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
            or receipt.get("gate_registerable_here") is not False
            or not train_checks
            or not all(bool(value) for value in train_checks.values())
        ):
            raise RuntimeError(f"R0217 seed-{seed} train receipt contract changed")
        if (
            dict(receipt.get("substrate") or {}) != dict(manifest["substrate"])
            or dict(receipt.get("graph_manifest") or {}) != manifest_signature
        ):
            raise RuntimeError(
                f"R0217 seed-{seed} was not trained on the substrate R0218 scores"
            )
        model_signature = dict(receipt["model"])
        if expected_input_signature(model_signature["canonical_path"]) != model_signature:
            raise RuntimeError(f"R0217 seed-{seed} published map bytes changed")
        invariants.add(str(receipt["seed_invariant_sha256"]))
        model_hashes[str(seed)] = str(model_signature["sha256"])
        cells.append({
            "seed": int(seed),
            "capability": capability,
            "train_receipt": receipt_signature,
        })
        inputs.extend([receipt_signature, model_signature])
    if len(invariants) != 1:
        raise RuntimeError(
            "R0218 refuses to score an incommensurate family: the four R0217 "
            f"cells carry {len(invariants)} seed-invariant config digests"
        )
    if len(set(model_hashes.values())) != len(SEEDS):
        raise RuntimeError("R0218 family contains a duplicated checkpoint")
    family = {
        "seed_invariant_sha256": sorted(invariants)[0],
        "model_sha256_by_seed": model_hashes,
    }
    return cells, inputs, family


def prepare_round0218(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0218 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    manifest_signature, manifest, edges = _sealed_substrate()
    substrate_signature = dict(manifest["substrate"])
    provenance_signature = dict(manifest["provenance"])
    cells, map_inputs, family = _accepted_maps(manifest_signature, manifest)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0218 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    family_path = os.path.join(preflight, "scored-family-identity.json")
    atomic_write_new_json(
        family_path,
        prompt_contract.seal({
            "schema": "round0218-scored-family-identity-v1",
            "round_id": ROUND_ID,
            "release_sha": release_sha,
            "sealed_directed_edges": edges,
            "substrate": substrate_signature,
            "provenance": provenance_signature,
            "family": family,
            "cells": cells,
        }),
        immutable=True,
    )
    expected_inputs = _dedupe([
        round_signature,
        manifest_signature,
        substrate_signature,
        provenance_signature,
        *map_inputs,
        expected_input_signature(smoke_path),
        expected_input_signature(family_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    node = "score_minilm_mixed_2m_seed_family_panel"
    job = {
        "id": node,
        "action": ACTION,
        "handler_module": "experiments.round0218_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{node}.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": NODE_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": manifest_signature,
        "cells": cells,
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
        "schema": "round0218-minilm-mixed-2m-seed-family-panel-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-evaluation",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            GRAPH_CAPABILITY,
            *(map_capability(seed) for seed in SEEDS),
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "p90_gpu_seconds": {node: NODE_P90_WALL_S, "total": NODE_P90_WALL_S},
        "scientific_contract": {
            "question": (
                "what does the frozen panel measure for four commensurate maps "
                "over the sealed R0216 2M mixed MiniLM substrate?"
            ),
            "population": "sealed R0216 2,000,000-row mixed MiniLM substrate",
            "sealed_directed_edges": edges,
            "seeds": list(SEEDS),
            "cells": len(SEEDS),
            "metrics": list(PANEL_METRICS),
            "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
            "corpus_ffr_slices": list(CORPUS_SLUGS),
            "centroid_ks": list(CENTROID_KS),
            "panel_config_source": "accepted R0113 panel_config()",
            "shared_high_d_reference": True,
            "family_seed_invariant_sha256": family["seed_invariant_sha256"],
            "model_sha256_by_seed": family["model_sha256_by_seed"],
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "gate_registerable_here": GATE_REGISTERABLE_HERE,
            "quality_claim_made_here": False,
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
        "queue_manifest": prepare_round0218(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
