#!/usr/bin/env python3
"""Prepare, but never launch, the R0233 queue — the 6.25M rung.

Four nodes: assemble (substrate + reserves), exact truth over all rows, the
`cluster-spill-nnd` build ladder at `s = 8`, and qualification + the device-law
refit. No training node exists in this queue by design.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0233_substrate import (
    BUILD_TIMEOUT_S,
    COMPOSITION,
    DIMENSION,
    GPU_HOURS_CAP,
    GRAPH_CAPABILITY,
    LADDER_CAPABILITY,
    LADDER_CLUSTERS,
    NN_DESCENT_SETTING,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RECALL_POPULATION,
    RESERVE_ROWS,
    ROUND_ID,
    ROWS,
    SPILL,
    SUBSTRATE_CAPABILITY,
    TRUTH_CAPABILITY,
)
from experiments.round0233_nodes import (
    ASSEMBLE_ACTION,
    LADDER_ACTION,
    QUALIFY_ACTION,
    TRUTH_ACTION,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0233"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0233-2026-08-09.md")
EMB = "/data/embeddings"

ASSEMBLE_P90_WALL_S = 1_500.0
TRUTH_P90_WALL_S = 2_000.0
LADDER_P90_WALL_S = 4_800.0
QUALIFY_P90_WALL_S = 2_000.0


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor",
         base_commit, release_sha],
        check=False, timeout=10,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0233 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0233 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(f"R0233 release checkout is at {observed}, not {release_sha}")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0233_contract.py", "tests/test_round0233_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0233 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0233_contract.py "
            "tests/test_round0233_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "selection law with a forced prefix (must raise)",
            "reserve split closure and disjointness",
            "c selection from measured imbalance, C_MIN = 2s",
            "memmap precondition for cuVS inputs (must raise on an ndarray)",
            "signal-free abort policy (must raise on SIGTERM)",
            "device-law refit, per-rung derivation, I/O term",
            "R0216 fuzzy law on a tiny memmapped substrate",
        ],
    }


def _source_size_manifest(queue_root: str) -> str:
    corpora: dict[str, Any] = {}
    for corpus, _rows in COMPOSITION:
        sizes: dict[str, int] = {}
        root = os.path.join(EMB, corpus, "train")
        for name in sorted(os.listdir(root)):
            if not name.endswith(".npy") or name.endswith(".tmp.npy"):
                continue
            path = os.path.join(root, name)
            sizes[os.path.relpath(path, EMB)] = os.path.getsize(path)
        if not sizes:
            raise RuntimeError(f"R0233 found no shards for {corpus}")
        corpora[corpus] = {"shard_sizes": sizes}
    path = os.path.join(queue_root, "source-size-manifest.json")
    atomic_write_new_json(
        path, {"schema": "round0233-source-size-manifest-v1", "corpora": corpora},
        immutable=True,
    )
    return path


def prepare_round0233(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    reuse_substrate: str | None = None,
) -> str:
    """Build the queue.

    `reuse_substrate` names an already-sealed `substrate.json` from an earlier
    queue of THIS round. A setup-class re-run must not pay for the substrate
    twice, and re-deriving it would also produce a second set of bytes for the
    same registered selection. When it is given the assemble node is dropped and
    the sealed receipt is declared as a queue input at its full signature, so the
    lineage is explicit rather than implied.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0233 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0233 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    manifest_path = _source_size_manifest(queue_root)

    reused: dict[str, Any] | None = None
    if reuse_substrate:
        if not os.path.exists(reuse_substrate):
            raise RuntimeError(f"R0233 reused substrate absent at {reuse_substrate}")
        with open(reuse_substrate, encoding="utf-8") as handle:
            sealed = json.load(handle)
        if sealed.get("round_id") != ROUND_ID or sealed.get("rows") != ROWS:
            raise RuntimeError("R0233 reused substrate is not this round's")
        for key in ("substrate", "provenance", "reserve_substrate",
                    "reserve_provenance", "reserve_query_rows"):
            declared = dict(sealed[key])
            path = str(declared["canonical_path"])
            if expected_input_signature(path) != declared:
                raise RuntimeError(f"R0233 reused substrate {key} bytes changed")
        reused = expected_input_signature(reuse_substrate)

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(manifest_path),
        *( [reused] if reused else [] ),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    cache_root = os.path.join(ROUND_ROOT, "child-cache")
    scratch_root = os.path.join(ROUND_ROOT, "spill-scratch")

    substrate_dir = os.path.join(artifacts, SUBSTRATE_CAPABILITY)
    substrate_manifest = (
        reuse_substrate if reuse_substrate
        else os.path.join(substrate_dir, "substrate.json")
    )
    truth_dir = os.path.join(artifacts, TRUTH_CAPABILITY)
    truth_manifest = os.path.join(truth_dir, "exact-k15-truth.json")
    ladder_dir = os.path.join(artifacts, LADDER_CAPABILITY)
    ladder_manifest = os.path.join(ladder_dir, "build-ladder.json")
    graph_dir = os.path.join(artifacts, GRAPH_CAPABILITY)

    intra = lambda path: {"kind": "file", "canonical_path": path}  # noqa: E731
    policy = {"gpu_required": True, "training_performed": False, "cpu_heavy": False}

    substrate_reference = (
        expected_input_signature(reuse_substrate) if reuse_substrate
        else intra(substrate_manifest)
    )
    jobs: list[dict[str, Any]] = []
    if not reuse_substrate:
        jobs.append({
            "id": "assemble_6250k", "action": ASSEMBLE_ACTION,
            "handler_module": "experiments.round0233_nodes",
            "handler_callable": "run_job", "deps": [],
            "outputs": [substrate_dir],
            "done_marker": os.path.join(artifacts, "assemble_6250k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": ASSEMBLE_P90_WALL_S,
            "capability": SUBSTRATE_CAPABILITY,
            "source_size_manifest": expected_input_signature(manifest_path),
            "node_policy": {**policy, "cpu_heavy": True},
        })
    jobs.extend([
        {
            "id": "truth_6250k", "action": TRUTH_ACTION,
            "handler_module": "experiments.round0233_nodes",
            "handler_callable": "run_job",
            "deps": [] if reuse_substrate else ["assemble_6250k"],
            "outputs": [truth_dir],
            "done_marker": os.path.join(artifacts, "truth_6250k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": TRUTH_P90_WALL_S,
            "capability": TRUTH_CAPABILITY,
            "substrate_manifest": substrate_reference,
            "node_policy": dict(policy),
        },
        {
            "id": "ladder_6250k", "action": LADDER_ACTION,
            "handler_module": "experiments.round0233_nodes",
            "handler_callable": "run_job", "deps": ["truth_6250k"],
            "outputs": [ladder_dir],
            "done_marker": os.path.join(artifacts, "ladder_6250k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": LADDER_P90_WALL_S,
            "capability": LADDER_CAPABILITY,
            "substrate_manifest": substrate_reference,
            "scratch_root": scratch_root,
            "cache_root": cache_root,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "node_policy": dict(policy),
        },
        {
            "id": "qualify_6250k", "action": QUALIFY_ACTION,
            "handler_module": "experiments.round0233_nodes",
            "handler_callable": "run_job", "deps": ["ladder_6250k"],
            "outputs": [graph_dir],
            "done_marker": os.path.join(artifacts, "qualify_6250k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": QUALIFY_P90_WALL_S,
            "capability": GRAPH_CAPABILITY,
            "substrate_manifest": substrate_reference,
            "truth_reference": intra(truth_manifest),
            "ladder_reference": intra(ladder_manifest),
            "node_policy": dict(policy),
        },
    ])

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0233-6250k-rung-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-graph",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [],
        "capabilities_produced": [
            SUBSTRATE_CAPABILITY, TRUTH_CAPABILITY, LADDER_CAPABILITY,
            GRAPH_CAPABILITY,
        ],
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "total": (
                ASSEMBLE_P90_WALL_S + TRUTH_P90_WALL_S + LADDER_P90_WALL_S
                + QUALIFY_P90_WALL_S
            )
        },
        "scientific_contract": {
            "question": (
                "can the 6.25M rung be assembled with spanning selection and "
                "held-out reserves, and does cluster-spill-nnd at s = 8 with c "
                "derived from MEASURED imbalance produce a k15 graph that "
                "reproduces brute-force truth over every row with no edgeless "
                "row?"
            ),
            "rows": ROWS,
            "reserve_rows": RESERVE_ROWS,
            "dimension": DIMENSION,
            "spill": SPILL,
            "nn_descent_setting": NN_DESCENT_SETTING,
            "ladder_clusters": list(LADDER_CLUSTERS),
            "recall_population": RECALL_POPULATION,
            "floors": {
                "tie_aware_mean": RECALL_MEAN_FLOOR,
                "tie_aware_p10": RECALL_P10_FLOOR,
                "zero_degree_rows": 0,
            },
            "safety": (
                "every buffer handed to cuVS is a read-only np.memmap, asserted "
                "on the substrate view and on every intermediate per-cluster "
                "spill file; the abort path is an in-band cooperative flag and "
                "no signal is ever delivered to a build process"
            ),
            "no_training": True,
            "no_gate_registered": True,
            "no_adoption_claimed": True,
        },
    })
    queue_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(queue_path, queue, immutable=True)
    return queue_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument(
        "--reuse-substrate", default=None,
        help=(
            "path to an already-sealed substrate.json from an earlier queue of "
            "this round; drops the assemble node and binds those bytes instead"
        ),
    )
    args = parser.parse_args(argv)
    path = prepare_round0233(
        release_sha=args.release_sha, queue_root=args.queue_root,
        reuse_substrate=args.reuse_substrate,
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
