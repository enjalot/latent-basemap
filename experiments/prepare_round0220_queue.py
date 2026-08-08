#!/usr/bin/env python3
"""Prepare, but never launch, the R0220 cuVS k15 builder qualification queue.

Two GPU nodes on one queue: rebuild the exact k15 truth from R0216's sealed
`queue-correction-3` bytes, then measure cuVS against it over the full 2M rows.

Everything the round binds is proved here, before the GPU is touched: the
substrate and edge-file hashes are R0216's released ones, the RAPIDS env
actually imports the cuVS modules the sweep needs, and the round's
`required_reviews` list is copied into the manifest (review-0216-02's required
correction 1 — three releasing queues in a row shipped it empty).
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
from basemap.round0220_cuvs_qualification import (
    CAPABILITY,
    CUVS_METRIC,
    DIMENSION,
    EDGES_SHA256,
    GPU_HOURS_CAP,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    METRIC_EQUIVALENCE,
    OUT_OF_CORE_MODULE,
    PROJECTION_ROWS,
    REQUIRED_CUVS_MODULES,
    ROUND_ID,
    ROWS,
    SCALING_ROWS,
    SCALING_SETTING_ID,
    SEALED_DIRECTED_EDGES,
    SUBSTRATE_SHA256,
    SWEEP,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
    TRUTH_VALIDITY_MEAN_FLOOR,
    TRUTH_VALIDITY_P10_FLOOR,
)
from experiments.round0220_nodes import CUML_LAUNCHER, QUALIFY_ACTION, TRUTH_ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0220"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0220-2026-08-08.md")
R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{GRAPH_CAPABILITY}"
)
GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
TRUTH_NODE = "rebuild_exact_k15_truth"
QUALIFY_NODE = "qualify_cuvs_k15_builder"
TRUTH_P90_WALL_S = 1_800.0
QUALIFY_P90_WALL_S = 5_400.0


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
        raise RuntimeError("R0220 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0220 round must declare its required reviews")
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
        raise RuntimeError("R0220 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0220_cuvs_qualification.py",
        "tests/test_round0220_cpu_smoke.py",
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
        "schema": "round0220-release-cpu-smoke-v1",
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
            "sealed-manifest binding, recall/tie-aware/validity math, the "
            "power-law fit and projection labelling, the registered truth-probe "
            "floors, and the receipt seal path both nodes end on"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0220 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _rapids_environment() -> dict[str, Any]:
    """Prove cuVS imports here, before a GPU node depends on it."""
    if not os.path.exists(CUML_LAUNCHER):
        raise RuntimeError(f"RAPIDS launcher {CUML_LAUNCHER} is absent")
    program = (
        "import json, cuvs, cuvs.neighbors as n\n"
        "print(json.dumps({'version': str(cuvs.__version__), 'modules': "
        "sorted(x for x in dir(n) if not x.startswith('_'))}))\n"
    )
    completed = subprocess.run(
        [CUML_LAUNCHER, "-c", program],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"RAPIDS env does not import cuVS:\n{completed.stdout}\n{completed.stderr}"
        )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    missing = [name for name in REQUIRED_CUVS_MODULES if name not in payload["modules"]]
    if missing:
        raise RuntimeError(f"RAPIDS env is missing cuVS modules {missing}")
    return prompt_contract.seal({
        "schema": "round0220-rapids-environment-v1",
        "round_id": ROUND_ID,
        "launcher": CUML_LAUNCHER,
        "cuvs_version": payload["version"],
        "cuvs_neighbors_modules": payload["modules"],
        "required_modules": list(REQUIRED_CUVS_MODULES),
        "out_of_core_module": OUT_OF_CORE_MODULE,
        "out_of_core_module_available": OUT_OF_CORE_MODULE in payload["modules"],
        "note": (
            "cuvs 25.02 predates the out-of-core all-neighbours entry point of "
            "0197-out-of-core-umap-gpu.md; its absence is recorded, not "
            "worked around"
        ),
    })


def _sealed_graph() -> tuple[dict[str, Any], dict[str, Any]]:
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
        or int(checks.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
        or int(checks.get("zero_degree_rows", -1)) != 0
        or manifest.get("training_performed") is not False
    ):
        raise RuntimeError("R0220 sealed R0216 substrate+graph contract changed")
    substrate = dict(manifest["substrate"])
    graph = dict(manifest["graph"])
    if substrate.get("sha256") != SUBSTRATE_SHA256 or graph.get("sha256") != EDGES_SHA256:
        raise RuntimeError(
            "R0220 refuses to measure against bytes that are not the released "
            "R0216 queue-correction-3 substrate and exact k15 edge file"
        )
    if expected_input_signature(substrate["canonical_path"]) != substrate:
        raise RuntimeError("R0216 substrate bytes changed on disk")
    if expected_input_signature(graph["canonical_path"]) != graph:
        raise RuntimeError("R0216 edge-file bytes changed on disk")
    return signature, manifest


def prepare_round0220(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0220 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    manifest_signature, manifest = _sealed_graph()
    substrate_signature = dict(manifest["substrate"])
    edges_signature = dict(manifest["graph"])

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0220 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    rapids_path = os.path.join(preflight, "rapids-environment.json")
    atomic_write_new_json(rapids_path, _rapids_environment(), immutable=True)

    expected_inputs = _dedupe([
        round_signature,
        manifest_signature,
        substrate_signature,
        edges_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(rapids_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    cache_root = os.path.join(queue_root, "cache", "cuvs")
    truth_output = os.path.join(artifacts, "exact-k15-truth")
    truth_job = {
        "id": TRUTH_NODE,
        "action": TRUTH_ACTION,
        "handler_module": "experiments.round0220_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [truth_output],
        "done_marker": os.path.join(artifacts, f"{TRUTH_NODE}.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": TRUTH_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": manifest_signature,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": False,
        },
    }
    qualify_job = {
        "id": QUALIFY_NODE,
        "action": QUALIFY_ACTION,
        "handler_module": "experiments.round0220_nodes",
        "handler_callable": "run_job",
        "deps": [TRUTH_NODE],
        "outputs": [os.path.join(artifacts, CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{QUALIFY_NODE}.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": QUALIFY_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": manifest_signature,
        "truth_receipt": os.path.join(truth_output, "truth-rebuild.json"),
        "cuvs_cache_root": cache_root,
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
        "schema": "round0220-cuvs-k15-builder-qualification-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-qualification",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [GRAPH_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [truth_job, qualify_job],
        "scientific_contract": {
            "question": (
                "can cuVS build a k15 all-neighbours graph over the R0216 2M "
                "substrate at high recall against exact truth, for a small "
                "fraction of the exact kernel's cost?"
            ),
            "truth": (
                "exact fp32 cosine top-15 recomputed from the sealed R0216 "
                "substrate under R0216's identical blocked law, because R0216 "
                "persisted no nbr/dist arrays"
            ),
            "truth_probe": {
                "rows": TRUTH_PROBE_ROWS,
                "seed": TRUTH_PROBE_SEED,
                "measure": "tie-aware validity of R0216's sealed adjacency",
                "mean_floor": TRUTH_VALIDITY_MEAN_FLOOR,
                "p10_floor": TRUTH_VALIDITY_P10_FLOOR,
            },
            "recall_population": ROWS,
            "recall_measures": ["strict", "tie-aware"],
            "tie_handling": (
                "the substrate contains exact-duplicate clusters, one of 1,377 "
                "byte-identical rows (review-0216-02), so strict set equality "
                "understates any correct builder; both are reported"
            ),
            "metric": CUVS_METRIC,
            "metric_equivalence": METRIC_EQUIVALENCE,
            "sweep": [dict(item) for item in SWEEP],
            "scaling_setting_id": SCALING_SETTING_ID,
            "scaling_rows": list(SCALING_ROWS),
            "projection_rows": PROJECTION_ROWS,
            "projection_is_a_measurement": False,
            "zero_degree_tripwire_reported_not_gated": True,
            "gate_registered": False,
            "map_quality_claim": False,
            "no_training": True,
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
        "queue_manifest": prepare_round0220(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
