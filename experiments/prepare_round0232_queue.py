#!/usr/bin/env python3
"""Prepare, but never launch, the R0232 queue.

Eight jobs, one queue: the 2M scratch-law grid, the 8M calibration, the
projection (emitted early, because it is the deliverable), the streamed arm's
fuzzy graph, three train cells and the geometry probe.
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
from basemap.round0227_low_c_contract import (
    BUILD_TIMEOUT_S,
    GPU_HOURS_CAP,
    GUARD_BUDGET_NOTE,
    SUBSTRATE_2M_PATH,
    TRUTH_RECEIPT_PATH,
    WATCHDOG_POLL_S,
)
from basemap.round0232_scratch_contract import (
    ADOPTION_CLAIMED,
    ARM_CELL,
    ARM_REFERENCE_CELL,
    ARM_STRICT_FLOOR,
    ARM_TIE_AWARE_FLOOR,
    DESIGN_NOTE,
    DISK_FREE_RESERVE_BYTES,
    DISK_GUARD_NOTE,
    DISPLACEMENT_ALPHA,
    GEOMETRY_CAPABILITY,
    GRAPH_CAPABILITY,
    GRID_A,
    GRID_B,
    GRID_CAPABILITY,
    IDENTITY_FAMILIES,
    LARGER_N_CAPABILITY,
    MINIMUM_DETECTABLE_DISPLACEMENT_SD,
    NON_REJECTION_NOTE,
    PERMUTATION_LABELLINGS,
    PERMUTATION_RESOLUTION_CEILING,
    PROJECTION_CAPABILITY,
    RECALL_POPULATION,
    RECALL_POPULATION_NOTE,
    ROUND_ID,
    ROUND_SCRATCH_BUDGET_BYTES,
    ROWS,
    SEEDS,
    SPILL_VOLUME_100M_S8_BYTES,
    STREAMED_MODES_ADMITTED,
    STREAMED_REFUSAL_REASON,
    SUBSTRATE_16M_PATH,
    cell_guard,
    data_free_bytes,
    map_capability,
)
from experiments.round0232_nodes import (
    FUZZY_ACTION,
    GEOMETRY_ACTION,
    GRID_ACTION,
    LARGER_N_ACTION,
    PROJECT_ACTION,
    TRAIN_ACTION,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0232"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0232-2026-08-09.md")

R0228_GEOMETRY_PATH = (
    "/data/latent-basemap/runs/round-0228/queue-correction-1/artifacts/"
    "minilm-mixed-2m-cluster-spill-map-geometry-v1/cluster-spill-map-geometry.json"
)
R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1"
)
R0216_GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
R0229_ARM_BUILD_RECEIPT = (
    "/data/latent-basemap/runs/round-0229/queue-phase2-correction-3/artifacts/"
    "spill-lifted-build/build-receipt.json"
)
R0229_ARM_GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0229/queue-phase2-correction-3/artifacts/"
    "minilm-mixed-2m-spill-lifted-k15-fuzzy-graph-v1/spill-lifted-graph.json"
)

#: The eight exact-graph maps: R0218 seeds 42-45, R0222 seeds 46-49, declared as
#: DIRECT queue inputs (review-0228-01).
EXACT_COORDINATES = {
    42: "/data/latent-basemap/runs/round-0218/queue/artifacts/"
        "minilm-mixed-2m-seed-family-panel-v1/coordinates-seed42.npy",
    43: "/data/latent-basemap/runs/round-0218/queue/artifacts/"
        "minilm-mixed-2m-seed-family-panel-v1/coordinates-seed43.npy",
    44: "/data/latent-basemap/runs/round-0218/queue/artifacts/"
        "minilm-mixed-2m-seed-family-panel-v1/coordinates-seed44.npy",
    45: "/data/latent-basemap/runs/round-0218/queue/artifacts/"
        "minilm-mixed-2m-seed-family-panel-v1/coordinates-seed45.npy",
    46: "/data/latent-basemap/runs/round-0222/queue/artifacts/"
        "minilm-mixed-2m-quality-gates-n8-v1/coordinates-seed46.npy",
    47: "/data/latent-basemap/runs/round-0222/queue/artifacts/"
        "minilm-mixed-2m-quality-gates-n8-v1/coordinates-seed47.npy",
    48: "/data/latent-basemap/runs/round-0222/queue/artifacts/"
        "minilm-mixed-2m-quality-gates-n8-v1/coordinates-seed48.npy",
    49: "/data/latent-basemap/runs/round-0222/queue/artifacts/"
        "minilm-mixed-2m-quality-gates-n8-v1/coordinates-seed49.npy",
}

GRID_P90_WALL_S = 3_000.0
LARGER_N_P90_WALL_S = 2_000.0
PROJECT_P90_WALL_S = 200.0
FUZZY_P90_WALL_S = 400.0
TRAIN_P90_WALL_S = 900.0
GEOMETRY_P90_WALL_S = 600.0


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
        raise RuntimeError("R0232 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0232 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    """The model-producing release's CUDA-hidden CPU smoke.

    R0232's own contract, grids, guard and statistics, plus R0229's phase-2 CPU
    smoke run **unmodified**: R0232's train node is R0229's train path with the
    graph swapped, and R0229's smoke is the one that reaches the actual post-fit
    accounting, checkpoint publish, full-population reload, coordinate
    publication and receipt seal.
    """
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0232 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0232_scratch.py",
         "tests/test_round0229_phase2_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0232 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0232_scratch.py "
            "tests/test_round0229_phase2_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "R0232 grids, disk guard and scratch law",
            "ceil-derived dose", "R0217-template config construction",
            "cross-round treatment-digest equality", "post-fit accounting",
            "checkpoint publish", "full-population reload",
            "coordinate publication", "receipt seal",
        ],
        "note": (
            "preparation validation, not a queue node and not a scientific result"
        ),
    }


def prepare_round0232(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0232 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    for path in (
        SUBSTRATE_2M_PATH, SUBSTRATE_16M_PATH, TRUTH_RECEIPT_PATH,
        R0228_GEOMETRY_PATH, R0216_GRAPH_MANIFEST, R0229_ARM_BUILD_RECEIPT,
        R0229_ARM_GRAPH_MANIFEST, *EXACT_COORDINATES.values(),
    ):
        if not os.path.exists(path):
            raise RuntimeError(f"R0232 input is absent at {path}")

    # Every registered cell is guarded at prepare time as well as at run time, so
    # a configuration that cannot launch is visible before the GPU is taken.
    free_now = data_free_bytes()
    guards = {
        str(cell["cell"]): cell_guard(cell, free_bytes=free_now)
        for cell in (*GRID_A, *GRID_B)
    }
    if not any(guard["allowed"] for guard in guards.values()):
        raise RuntimeError("R0232 has no launchable cell under the current guards")

    substrate_signature = expected_input_signature(SUBSTRATE_2M_PATH)
    benchmark_signature = expected_input_signature(SUBSTRATE_16M_PATH)
    truth_signature = expected_input_signature(TRUTH_RECEIPT_PATH)
    geometry_signature = expected_input_signature(R0228_GEOMETRY_PATH)
    r0216_manifest_signature = expected_input_signature(R0216_GRAPH_MANIFEST)
    with open(R0216_GRAPH_MANIFEST, encoding="utf-8") as handle:
        r0216_graph_signature = dict(json.load(handle)["graph"])
    r0229_build_signature = expected_input_signature(R0229_ARM_BUILD_RECEIPT)
    r0229_graph_signature = expected_input_signature(R0229_ARM_GRAPH_MANIFEST)
    exact_signatures = {
        seed: expected_input_signature(path)
        for seed, path in sorted(EXACT_COORDINATES.items())
    }

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0232 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    guard_path = os.path.join(preflight, "cell-guards.json")
    atomic_write_new_json(
        guard_path,
        {
            "data_free_bytes_at_prepare": free_now,
            "round_scratch_budget_bytes": ROUND_SCRATCH_BUDGET_BYTES,
            "disk_free_reserve_bytes": DISK_FREE_RESERVE_BYTES,
            "guards": guards,
            "note": (
                "the predictive guard evaluated at prepare time on all thirteen "
                "registered cells; every cell is guarded AGAIN immediately before "
                "it launches, against /data free space re-read at that moment"
            ),
        },
        immutable=True,
    )

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(guard_path),
        substrate_signature, benchmark_signature, truth_signature,
        geometry_signature, r0216_manifest_signature,
        r0229_build_signature, r0229_graph_signature,
        *exact_signatures.values(),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    cache_root = os.path.join(ROUND_ROOT, "child-cache")
    scratch_root = os.path.join(ROUND_ROOT, "spill-scratch")
    partition_root = os.path.join(ROUND_ROOT, "shared-partition")

    grid_dir = os.path.join(artifacts, GRID_CAPABILITY)
    grid_path = os.path.join(grid_dir, "scratch-law.json")
    larger_dir = os.path.join(artifacts, LARGER_N_CAPABILITY)
    larger_path = os.path.join(larger_dir, "larger-n-calibration.json")
    projection_dir = os.path.join(artifacts, PROJECTION_CAPABILITY)
    projection_path = os.path.join(projection_dir, "scratch-and-cost-projection.json")
    graph_dir = os.path.join(artifacts, GRAPH_CAPABILITY)
    graph_manifest = os.path.join(graph_dir, "streamed-graph.json")
    geometry_dir = os.path.join(artifacts, GEOMETRY_CAPABILITY)
    geometry_path = os.path.join(geometry_dir, "streamed-geometry.json")

    intra = lambda path: {"kind": "file", "canonical_path": path}  # noqa: E731

    common = {
        "substrate_signature": substrate_signature,
        "truth_signature": truth_signature,
    }

    jobs: list[dict[str, Any]] = [
        {
            "id": "measure_scratch_law", "action": GRID_ACTION,
            "handler_module": "experiments.round0232_nodes",
            "handler_callable": "run_job", "deps": [],
            "outputs": [grid_dir],
            "done_marker": os.path.join(artifacts, "measure_scratch_law.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": GRID_P90_WALL_S,
            "capability": GRID_CAPABILITY,
            "artifact_dir": grid_dir, "artifact_path": grid_path,
            "cache_root": cache_root, "cuvs_cache_root": cache_root,
            "scratch_root": scratch_root, "partition_root": partition_root,
            **common,
            "node_policy": {
                "gpu_required": True, "training_performed": False, "cpu_heavy": False,
            },
        },
        {
            "id": "calibrate_larger_n", "action": LARGER_N_ACTION,
            "handler_module": "experiments.round0232_nodes",
            "handler_callable": "run_job", "deps": ["measure_scratch_law"],
            "outputs": [larger_dir],
            "done_marker": os.path.join(artifacts, "calibrate_larger_n.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": LARGER_N_P90_WALL_S,
            "capability": LARGER_N_CAPABILITY,
            "artifact_dir": larger_dir, "artifact_path": larger_path,
            "cache_root": cache_root, "cuvs_cache_root": cache_root,
            "scratch_root": scratch_root, "partition_root": partition_root,
            "benchmark_substrate_signature": benchmark_signature,
            **common,
            "node_policy": {
                "gpu_required": True, "training_performed": False, "cpu_heavy": False,
            },
        },
        {
            "id": "project_scratch_and_cost", "action": PROJECT_ACTION,
            "handler_module": "experiments.round0232_nodes",
            "handler_callable": "run_job",
            "deps": ["measure_scratch_law", "calibrate_larger_n"],
            "outputs": [projection_dir],
            "done_marker": os.path.join(
                artifacts, "project_scratch_and_cost.done.json"
            ),
            "expected_inputs": expected_inputs,
            "p90_wall_s": PROJECT_P90_WALL_S,
            "capability": PROJECTION_CAPABILITY,
            "artifact_dir": projection_dir, "artifact_path": projection_path,
            "grid_reference": intra(grid_path),
            "larger_n_reference": intra(larger_path),
            **common,
            "node_policy": {
                "gpu_required": True, "training_performed": False, "cpu_heavy": False,
            },
        },
    ]
    jobs[0]["arm_required"] = bool(STREAMED_MODES_ADMITTED)

    # Addendum 1: with the streamed modes withdrawn on machine-safety grounds the
    # arm graph cannot be built, so the map arm does not run and the registered
    # displacement probe is reported as NOT RUN rather than inferred.
    map_jobs: list[dict[str, Any]] = [
        {
            "id": "fuzzy_streamed_arm", "action": FUZZY_ACTION,
            "handler_module": "experiments.round0232_nodes",
            "handler_callable": "run_job", "deps": ["project_scratch_and_cost"],
            "outputs": [graph_dir],
            "done_marker": os.path.join(artifacts, "fuzzy_streamed_arm.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": FUZZY_P90_WALL_S,
            "capability": GRAPH_CAPABILITY,
            "grid_reference": intra(grid_path),
            **common,
            "node_policy": {
                "gpu_required": True, "training_performed": False, "cpu_heavy": False,
            },
        },
    ]

    train_nodes: list[str] = []
    for seed in SEEDS:
        node = f"train_streamed_seed{seed}"
        train_nodes.append(node)
        output = os.path.join(artifacts, map_capability(seed))
        map_jobs.append({
            "id": node, "action": TRAIN_ACTION,
            "handler_module": "experiments.round0232_nodes",
            "handler_callable": "run_job",
            "deps": ["fuzzy_streamed_arm"] + train_nodes[:-1],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": TRAIN_P90_WALL_S,
            "capability": map_capability(seed),
            "training_seed": seed,
            "graph_manifest_reference": intra(graph_manifest),
            "r0216_graph_signature": r0216_graph_signature,
            "r0216_graph_manifest_signature": r0216_manifest_signature,
            **common,
            "node_policy": {
                "gpu_required": True, "training_performed": True, "cpu_heavy": False,
            },
        })

    map_jobs.append({
        "id": "probe_streamed_geometry", "action": GEOMETRY_ACTION,
        "handler_module": "experiments.round0232_nodes",
        "handler_callable": "run_job", "deps": list(train_nodes),
        "outputs": [geometry_dir],
        "done_marker": os.path.join(
            artifacts, "probe_streamed_geometry.done.json"
        ),
        "expected_inputs": expected_inputs,
        "p90_wall_s": GEOMETRY_P90_WALL_S,
        "capability": GEOMETRY_CAPABILITY,
        "artifact_dir": geometry_dir, "artifact_path": geometry_path,
        "graph_manifest_reference": intra(graph_manifest),
        "r0228_geometry_signature": geometry_signature,
        "candidate_coordinates": [
            {
                "name": f"{'streamed-spill'}-seed{seed}",
                "signature": intra(
                    os.path.join(
                        artifacts, map_capability(seed), f"coordinates-seed{seed}.npy"
                    )
                ),
            }
            for seed in SEEDS
        ],
        "exact_coordinates": [
            {"name": f"exact-seed{seed}", "signature": exact_signatures[seed]}
            for seed in sorted(EXACT_COORDINATES)
        ],
        **common,
        "node_policy": {
            "gpu_required": True, "training_performed": False, "cpu_heavy": False,
        },
    })

    if STREAMED_MODES_ADMITTED:
        jobs.extend(map_jobs)
        produced = [
            GRID_CAPABILITY, LARGER_N_CAPABILITY, PROJECTION_CAPABILITY,
            GRAPH_CAPABILITY, *[map_capability(seed) for seed in SEEDS],
            GEOMETRY_CAPABILITY,
        ]
    else:
        produced = [GRID_CAPABILITY, LARGER_N_CAPABILITY, PROJECTION_CAPABILITY]

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0232-scratch-law-and-streamed-builder-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-train",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-2m-substrate-and-exact-k15-graph-v1",
            "minilm-mixed-16m-benchmark-substrate-v1",
            "minilm-100m-low-cluster-count-graph-configuration-v1",
            "minilm-mixed-2m-cluster-spill-map-geometry-v1",
            "minilm-mixed-2m-spill-lifted-k15-fuzzy-graph-v1",
        ],
        "capabilities_produced": produced,
        "training_performed": bool(STREAMED_MODES_ADMITTED),
        "jobs": jobs,
        "p90_gpu_seconds": {
            "measure_scratch_law": GRID_P90_WALL_S,
            "calibrate_larger_n": LARGER_N_P90_WALL_S,
            "project_scratch_and_cost": PROJECT_P90_WALL_S,
            "total": (
                GRID_P90_WALL_S + LARGER_N_P90_WALL_S + PROJECT_P90_WALL_S
                + (
                    FUZZY_P90_WALL_S + len(SEEDS) * TRAIN_P90_WALL_S
                    + GEOMETRY_P90_WALL_S
                    if STREAMED_MODES_ADMITTED else 0.0
                )
            ),
        },
        "scientific_contract": {
            "question": (
                "Review-0229-01 blocked '50M and 100M are unblocked inside the "
                "existing device and storage budgets' on a storage argument, and "
                "the 1.23 TB figure that reached this round is the total spill "
                "VOLUME at 100M s = 8, not peak scratch at any instant. R0227's "
                "builder already bounds peak scratch by packing whole clusters "
                "into groups against a budget, but no round has ever MEASURED it: "
                "`peak_scratch_bytes` is computed from `sizes` before a byte is "
                "written. So: what is the measured peak-scratch law, can the "
                "spill be restructured to zero scratch without changing the "
                "graph, and does 100M fit the 280 GB /data has?"
            ),
            "design_note": DESIGN_NOTE,
            "grid_a": [dict(cell) for cell in GRID_A],
            "grid_b": [dict(cell) for cell in GRID_B],
            "identity_families": [list(family) for family in IDENTITY_FAMILIES],
            "arm_cell": ARM_CELL,
            "arm_reference_cell": ARM_REFERENCE_CELL,
            "arm_recall_floors": {
                "tie_aware": ARM_TIE_AWARE_FLOOR, "strict": ARM_STRICT_FLOOR,
            },
            "recall_population": RECALL_POPULATION,
            "recall_population_note": RECALL_POPULATION_NOTE,
            "rows": ROWS,
            "spill_volume_at_100m_s8_bytes": SPILL_VOLUME_100M_S8_BYTES,
            "falsifiable_predictions": {
                "P1": (
                    "at (2M, c = 200, s = 8) with a 4 GiB bound the MEASURED "
                    "filesystem peak is <= 4 GiB + max_cluster_bytes and the "
                    "realised spill_groups equals the packing's group count. If "
                    "not, the packing claim — the whole basis for saying 100M "
                    "does not need 1.23 TB — is wrong and this round says so."
                ),
                "P2": (
                    "every stream-* cell measures a filesystem peak of exactly 0 "
                    "bytes in its scratch directory"
                ),
                "P3": (
                    "at matched (rows, c, s) over a cached partition the sha256 of "
                    "top_ids and top_cos is equal across all three modes; a "
                    "mismatch is quantified in rows and recall, never hidden. Two "
                    "materialising cells at different bounds are the control for "
                    "nn-descent non-determinism."
                ),
                "P4": (
                    "the streamed arm reaches tie-aware >= 0.998 and strict >= "
                    "0.997 over all 2,000,000 rows with 0 zero-degree rows"
                ),
            },
            "measurement_discipline": (
                "measured_peak_scratch_bytes is the peak of st_blocks x 512 summed "
                "over the cell's scratch directory every 50 ms while the build "
                "ran. st_size would report a whole cluster as resident the moment "
                "open_memmap truncated the file. Every prior round published a "
                "modelled number under the name peak_scratch_bytes; both are "
                "published here, side by side."
            ),
            "displacement_test": (
                "R0228's registered statistic, R0228's code imported read-only, "
                "against the same 8-map exact null arm on the same "
                "density-matched row sets; exact permutation over "
                f"{PERMUTATION_LABELLINGS} labellings"
            ),
            "alpha": DISPLACEMENT_ALPHA,
            "smallest_attainable_p": PERMUTATION_RESOLUTION_CEILING,
            "minimum_detectable_displacement_sd": MINIMUM_DETECTABLE_DISPLACEMENT_SD,
            "non_rejection_note": NON_REJECTION_NOTE,
            "projection_discipline": (
                "the projection is emitted BY A NODE and hash-bound, not written "
                "as prose (review-0229-01 defect 3). The device law is refitted at "
                "gd 64 / igd 256, the wall law is a single unpooled stratum with "
                "its fitted range stated, scratch and substrate passes are their "
                "own lines, the I/O rate is measured in this round, and no "
                "projection is divided by another projection."
            ),
            "guard_budget_note": GUARD_BUDGET_NOTE,
            "disk_guard_note": DISK_GUARD_NOTE,
            "round_scratch_budget_bytes": ROUND_SCRATCH_BUDGET_BYTES,
            "disk_free_reserve_bytes": DISK_FREE_RESERVE_BYTES,
            "data_free_bytes_at_prepare": free_now,
            "watchdog_poll_s": WATCHDOG_POLL_S,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "refusal_is_data": (
                "a cell whose predicted device, host-anonymous or disk footprint "
                "exceeds a budget is refused before launch and recorded with its "
                "prediction; a cell whose REALISED largest cluster exceeds the "
                "registered capacity refuses itself after assignment; a cell whose "
                "MEASURED scratch runs past its bound aborts itself "
                "cooperatively. All three are data."
            ),
            "never_sigkill_a_cuda_context": (
                "aborts are cooperative, then SIGTERM with a 180 s grace; SIGKILL "
                "is a recorded last resort and its use is an execution-check "
                "failure"
            ),
            "addendum_1_machine_safety": {
                "streamed_modes_admitted": bool(STREAMED_MODES_ADMITTED),
                "reason": STREAMED_REFUSAL_REASON,
                "cells_refused": [
                    cell["cell"] for cell in (*GRID_A, *GRID_B)
                    if not guards[str(cell["cell"])]["allowed"]
                ],
                "map_arm_runs": bool(STREAMED_MODES_ADMITTED),
                "predictions_unresolved": (
                    ["P2", "P3", "P4"] if not STREAMED_MODES_ADMITTED else []
                ),
                "note": (
                    "these cells are REFUSED and published, not dropped; the "
                    "registered displacement probe is reported as NOT RUN and no "
                    "streamed-arm p is published or inferred"
                ),
            },
            "no_registered_cell_dropped": (
                "every cell of both grids appears in the published tables, "
                "including cells that refuse, abort or complicate the story"
            ),
            "training_performed": True,
            "evaluation_performed": True,
            "gate_registerable_here": False,
            "adoption_claimed": ADOPTION_CLAIMED,
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
        "queue_manifest": prepare_round0232(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
