#!/usr/bin/env python3
"""Prepare, but never launch, the R0236 queue — the nested 25M rung.

Four nodes: assemble (nested substrate + inherited reserve), exact truth for the
registered uniform probe, the imbalance replicate grid + one `cluster-spill-nnd`
cell with the block-layer I/O instrument, and qualification + the per-rung
re-derivation with its tolerance to adverse imbalance. No training node exists in
this queue by design.

Every inherited scientific input is declared at its FULL signature — R0235's
sealed substrate manifest and build ladder, R0233's sealed build ladder, R0229's
sealed nn-descent quality sweep, adopted-arm build receipt and spill-reachability
artifact — so that no device point and no imbalance figure in this round can come
from anywhere but a hash-bound artifact (review-0233-01 D3, upheld by
review-0235-01).
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
from basemap.round0236_rung3 import (
    BUILD_TIMEOUT_S,
    COMPOSITION,
    C_BUILD_MIN,
    C_BUILD_MIN_NOTE,
    DIMENSION,
    GPU_HOURS_CAP,
    GRAPH_CAPABILITY,
    IMBALANCE_CAPABILITY,
    IMBALANCE_PROBE_CLUSTERS,
    IMBALANCE_PROBE_ROWS,
    IMBALANCE_REPLICATE_SEEDS,
    IO_CAPABILITY,
    IO_REGIME_NOTE,
    LADDER_CAPABILITY,
    NN_DESCENT_SETTING,
    PARENT_ROWS,
    PRIMARY_IMBALANCE_SEED,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RECALL_POPULATION,
    REPLICATE_NOTE,
    RESERVE_ROWS,
    ROUND_ID,
    ROWS,
    SELECTION_CANDIDATES,
    SELECTION_LAW,
    SPILL,
    STRUCTURAL_POPULATION,
    SUBSTRATE_CAPABILITY,
    TRUTH_AFFORDABILITY_NOTE,
    TRUTH_CAPABILITY,
    TRUTH_METHOD,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
)
from experiments.round0236_nodes import (
    ASSEMBLE_ACTION,
    LADDER_ACTION,
    QUALIFY_ACTION,
    TRUTH_ACTION,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0236"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0236-2026-08-09.md")
EMB = "/data/embeddings"

#: Inherited, sealed, hash-bound scientific inputs.
R0235_SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0235/queue/artifacts/"
    "minilm-mixed-12500k-nested-substrate-and-reserves-v1/substrate.json"
)
R0235_LADDER = (
    "/data/latent-basemap/runs/round-0235/queue/artifacts/"
    "minilm-mixed-12500k-cluster-spill-build-ladder-v1/build-ladder.json"
)
R0233_LADDER = (
    "/data/latent-basemap/runs/round-0233/queue-correction-1/artifacts/"
    "minilm-mixed-6250k-cluster-spill-build-ladder-v1/build-ladder.json"
)
R0229_SWEEP = (
    "/data/latent-basemap/runs/round-0229/queue-correction-1/artifacts/"
    "minilm-mixed-2m-nnd-quality-sweep-v1/nnd-quality-sweep.json"
)
R0229_ARM = (
    "/data/latent-basemap/runs/round-0229/queue-phase2-correction-3/artifacts/"
    "spill-lifted-build/spill-lifted-build.json"
)
R0229_REACHABILITY = (
    "/data/latent-basemap/runs/round-0229/queue-correction-1/artifacts/"
    "minilm-mixed-2m-spill-reachability-v1/spill-reachability.json"
)

ASSEMBLE_P90_WALL_S = 2_400.0
TRUTH_P90_WALL_S = 2_400.0
LADDER_P90_WALL_S = 6_000.0
QUALIFY_P90_WALL_S = 2_400.0


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
        raise RuntimeError("R0236 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0236 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(f"R0236 release checkout is at {observed}, not {release_sha}")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0236_contract.py", "tests/test_round0236_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0236 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0236_contract.py "
            "tests/test_round0236_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "nesting assertion on provenance keys (must raise when a parent row "
            "is absent and when the prefix is permuted)",
            "reserve disjointness assertion (must raise on one shared row)",
            "shard span assertion on a forced prefix (must raise)",
            "the registered truth probe (uniform, distinct, ascending, "
            "reproducible from the seed alone)",
            "the replicate drift table with its within-N spread and the "
            "drift_exceeds_spread verdict",
            "the tolerance-to-adverse-imbalance arithmetic against "
            "review-0235-01's published 25M / 50M / 100M figures",
            "the two-regime physical I/O prediction and the quadratic scaling fit",
            "device-law homogeneity filter (an igd 128 / it 20 point is refused)",
            "the imbalance margin reaching BOTH the guard and the rung derivation",
            "the delegated build path (must raise without a config capacity)",
            "memmap precondition for cuVS inputs (must raise on an ndarray)",
            "signal-free abort policy (must raise on SIGTERM)",
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
            raise RuntimeError(f"R0236 found no shards for {corpus}")
        corpora[corpus] = {"shard_sizes": sizes}
    path = os.path.join(queue_root, "source-size-manifest.json")
    atomic_write_new_json(
        path, {"schema": "round0236-source-size-manifest-v1", "corpora": corpora},
        immutable=True,
    )
    return path


def prepare_round0236(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    """Build the queue."""
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0236 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    inherited = {
        "parent_substrate": R0235_SUBSTRATE_MANIFEST,
        "r0235_ladder": R0235_LADDER,
        "r0233_ladder": R0233_LADDER,
        "r0229_sweep": R0229_SWEEP,
        "r0229_arm": R0229_ARM,
        "r0229_reachability": R0229_REACHABILITY,
    }
    signatures = {}
    for key, path in inherited.items():
        if not os.path.exists(path):
            raise RuntimeError(f"R0236 inherited input absent: {key} at {path}")
        signatures[key] = expected_input_signature(path)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0236 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    manifest_path = _source_size_manifest(queue_root)

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(manifest_path),
        *signatures.values(),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    cache_root = os.path.join(ROUND_ROOT, "child-cache")
    scratch_root = os.path.join(ROUND_ROOT, "spill-scratch")

    substrate_dir = os.path.join(artifacts, SUBSTRATE_CAPABILITY)
    substrate_manifest = os.path.join(substrate_dir, "substrate.json")
    truth_dir = os.path.join(artifacts, TRUTH_CAPABILITY)
    truth_manifest = os.path.join(truth_dir, "probe-k15-truth.json")
    ladder_dir = os.path.join(artifacts, LADDER_CAPABILITY)
    ladder_manifest = os.path.join(ladder_dir, "build-ladder.json")
    graph_dir = os.path.join(artifacts, GRAPH_CAPABILITY)

    intra = lambda path: {"kind": "file", "canonical_path": path}  # noqa: E731
    policy = {"gpu_required": True, "training_performed": False, "cpu_heavy": False}

    jobs: list[dict[str, Any]] = [
        {
            "id": "assemble_25000k", "action": ASSEMBLE_ACTION,
            "handler_module": "experiments.round0236_nodes",
            "handler_callable": "run_job", "deps": [],
            "outputs": [substrate_dir],
            "done_marker": os.path.join(artifacts, "assemble_25000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": ASSEMBLE_P90_WALL_S,
            "capability": SUBSTRATE_CAPABILITY,
            "source_size_manifest": expected_input_signature(manifest_path),
            "parent_substrate_manifest": signatures["parent_substrate"],
            "node_policy": {**policy, "cpu_heavy": True},
        },
        {
            "id": "truth_probe_25000k", "action": TRUTH_ACTION,
            "handler_module": "experiments.round0236_nodes",
            "handler_callable": "run_job", "deps": ["assemble_25000k"],
            "outputs": [truth_dir],
            "done_marker": os.path.join(artifacts, "truth_probe_25000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": TRUTH_P90_WALL_S,
            "capability": TRUTH_CAPABILITY,
            "substrate_manifest": intra(substrate_manifest),
            "probe_rows": TRUTH_PROBE_ROWS,
            "probe_seed": TRUTH_PROBE_SEED,
            "node_policy": dict(policy),
        },
        {
            "id": "ladder_25000k", "action": LADDER_ACTION,
            "handler_module": "experiments.round0236_nodes",
            "handler_callable": "run_job", "deps": ["truth_probe_25000k"],
            "outputs": [ladder_dir],
            "done_marker": os.path.join(artifacts, "ladder_25000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": LADDER_P90_WALL_S,
            "capability": LADDER_CAPABILITY,
            "substrate_manifest": intra(substrate_manifest),
            "r0229_sweep": signatures["r0229_sweep"],
            "r0229_arm": signatures["r0229_arm"],
            "r0233_ladder": signatures["r0233_ladder"],
            "r0235_ladder": signatures["r0235_ladder"],
            "scratch_root": scratch_root,
            "cache_root": cache_root,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "node_policy": dict(policy),
        },
        {
            "id": "qualify_25000k", "action": QUALIFY_ACTION,
            "handler_module": "experiments.round0236_nodes",
            "handler_callable": "run_job", "deps": ["ladder_25000k"],
            "outputs": [graph_dir],
            "done_marker": os.path.join(artifacts, "qualify_25000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": QUALIFY_P90_WALL_S,
            "capability": GRAPH_CAPABILITY,
            "substrate_manifest": intra(substrate_manifest),
            "truth_reference": intra(truth_manifest),
            "ladder_reference": intra(ladder_manifest),
            "r0229_reachability": signatures["r0229_reachability"],
            "r0233_ladder": signatures["r0233_ladder"],
            "r0235_ladder": signatures["r0235_ladder"],
            "probe_rows": TRUTH_PROBE_ROWS,
            "probe_seed": TRUTH_PROBE_SEED,
            "node_policy": dict(policy),
        },
    ]

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0236-25000k-nested-rung-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-graph",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-12500k-nested-substrate-and-reserves-v1",
            "minilm-mixed-12500k-cluster-spill-build-ladder-v1",
            "minilm-mixed-cluster-spill-s8-imbalance-drift-v1",
        ],
        "capabilities_produced": [
            SUBSTRATE_CAPABILITY, TRUTH_CAPABILITY, LADDER_CAPABILITY,
            GRAPH_CAPABILITY, IMBALANCE_CAPABILITY, IO_CAPABILITY,
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
                "can rung 3 be assembled so that it CONTAINS rung 2's 12,500,000 "
                "training rows while still spanning every corpus, does "
                "cluster-spill-nnd at s = 8 with c chosen under the round's own "
                "imbalance margin reproduce brute-force truth on a registered "
                "uniform probe with no edgeless row in either direction, does a "
                "three-seed replicate grid at three nested N separate the draw "
                "channel from the N channel in cluster imbalance, and what do "
                "measured substrate reads say the 50M and 100M I/O terms cost?"
            ),
            "rows": ROWS,
            "parent_rows": PARENT_ROWS,
            "reserve_rows": RESERVE_ROWS,
            "reserve_source": "inherited verbatim through R0235 from R0233",
            "dimension": DIMENSION,
            "spill": SPILL,
            "nn_descent_setting": NN_DESCENT_SETTING,
            "selection_law": SELECTION_LAW,
            "selection_candidates": list(SELECTION_CANDIDATES),
            "c_build_min": C_BUILD_MIN,
            "c_build_min_note": C_BUILD_MIN_NOTE,
            "imbalance_probe_clusters": list(IMBALANCE_PROBE_CLUSTERS),
            "imbalance_probe_rows": list(IMBALANCE_PROBE_ROWS),
            "imbalance_replicate_seeds": list(IMBALANCE_REPLICATE_SEEDS),
            "primary_imbalance_seed": PRIMARY_IMBALANCE_SEED,
            "replicate_note": REPLICATE_NOTE,
            "recall_population": RECALL_POPULATION,
            "structural_population": STRUCTURAL_POPULATION,
            "truth_method": TRUTH_METHOD,
            "truth_probe_rows": TRUTH_PROBE_ROWS,
            "truth_probe_seed": TRUTH_PROBE_SEED,
            "truth_affordability": TRUTH_AFFORDABILITY_NOTE,
            "io_regime_note": IO_REGIME_NOTE,
            "floors": {
                "tie_aware_mean": RECALL_MEAN_FLOOR,
                "tie_aware_p10": RECALL_P10_FLOOR,
                "zero_degree_rows_in_and_out": 0,
                "shard_coverage_per_corpus": 0.999,
                "nested_parent_rows_missing": 0,
                "reserve_training_intersection_rows": 0,
                "increment_reserve_intersection_rows": 0,
            },
            "safety": (
                "every buffer handed to cuVS is a read-only np.memmap, asserted "
                "on the substrate view and on every intermediate per-cluster "
                "spill file; the abort path is an in-band cooperative flag and "
                "no signal is ever delivered to a build process; the build path "
                "is R0233's reviewed script reached through R0235's reviewed "
                "one-constant rebind, with a fail-closed delegation and no new "
                "code inside the child"
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
    args = parser.parse_args(argv)
    path = prepare_round0236(
        release_sha=args.release_sha, queue_root=args.queue_root
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
