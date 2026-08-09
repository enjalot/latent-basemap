#!/usr/bin/env python3
"""Prepare, but never launch, the R0229 phase-1 queue.

Three nodes: the nn-descent quality sweep at `c = 16` over one shared partition,
the reachability grid against `(c, s)`, and the registered displacement
inference rule applied to R0228's sealed per-map gaps.

Nothing here trains, registers a gate, or seals a graph for downstream use.
Phase 2 is prepared separately and only if the registered trigger fires.
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
    CANDIDATE,
    CLUSTER_CAPACITY_ROWS,
    GPU_HOURS_CAP,
    GUARD_BUDGET_NOTE,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    SUBSTRATE_2M_PATH,
    TRUTH_COS_PATH,
    TRUTH_IDS_PATH,
    TRUTH_RECEIPT_PATH,
    WATCHDOG_POLL_S,
)
from basemap.round0229_quality_contract import (
    ADOPTION_CLAIMED,
    BASELINE_CELL,
    DIMENSION,
    DISPLACEMENT_ALPHA,
    DISPLACEMENT_TEST_NOTE,
    DECISION_RULE_NOTE,
    GRAPH_K,
    PERMUTATION_LABELLINGS,
    PERMUTATION_RESOLUTION_CEILING,
    PHASE2_CEILING_TRIGGER,
    PHASE2_RECALL_TRIGGER,
    PHASE2_RUNGS,
    PHASE2_TRIGGER_NOTE,
    QUALITY_SWEEP,
    RECALL_POPULATION,
    RECALL_POPULATION_NOTE,
    RESOLUTION_RULE_NOTE,
    RETRO_CAPABILITY,
    ROUND_ID,
    ROWS,
    SPILL_CAPABILITY,
    SPILL_GRID,
    SPILL_IO_NOTE,
    STRUCTURAL_BOUND_NOTE,
    SWEEP_CAPABILITY,
    SWEEP_CLUSTERS,
    SWEEP_SPILL,
    TIE_QUERY_ROWS,
    TIE_QUERY_SEED,
    TREND_ASSIGNMENTS,
    TREND_RESOLUTION_CEILING,
    TREND_TEST_NOTE,
)
from experiments.round0229_nodes import (
    RETRO_ACTION,
    SPILL_ACTION,
    SWEEP_ACTION,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0229"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0229-2026-08-09.md")

#: R0227's sealed reachability artifact — the source of the structural bound.
R0227_REACHABILITY_PATH = (
    "/data/latent-basemap/runs/round-0227/queue/artifacts/low-c-reachability/"
    "reachability-vs-cluster-count.json"
)
#: R0228's sealed geometry artifact — the source of the per-map gaps.
R0228_GEOMETRY_PATH = (
    "/data/latent-basemap/runs/round-0228/queue-correction-1/artifacts/"
    "minilm-mixed-2m-cluster-spill-map-geometry-v1/cluster-spill-map-geometry.json"
)

#: Eight builds at c = 16 over a 318k-row largest cluster, the costliest at
#: igd 256 / 40 iterations, plus eight full-population recall passes.
SWEEP_P90_WALL_S = 3_600.0
#: Eleven k-means fits, eleven full-population coverage scans, eleven 200k-row
#: tie scans, the largest at spill 8.
SPILL_P90_WALL_S = 2_400.0
#: Pure CPU enumeration of 165 relabellings, four times over.
RETRO_P90_WALL_S = 300.0


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor",
         base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0229 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0229 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0229 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0229_quality.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
        timeout=240,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0229 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": "pytest -q tests/test_round0229_quality.py",
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "note": (
            "CUDA-hidden CPU smoke over this round's contract, its registered "
            "test statistics and its grids; preparation validation, not a queue "
            "node and not a scientific result"
        ),
    }


def prepare_round0229(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0229 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    for path in (
        SUBSTRATE_2M_PATH, TRUTH_RECEIPT_PATH, TRUTH_IDS_PATH, TRUTH_COS_PATH,
        R0227_REACHABILITY_PATH, R0228_GEOMETRY_PATH,
    ):
        if not os.path.exists(path):
            raise RuntimeError(f"R0229 input is absent at {path}")

    substrate_signature = expected_input_signature(SUBSTRATE_2M_PATH)
    truth_signature = expected_input_signature(TRUTH_RECEIPT_PATH)
    reachability_signature = expected_input_signature(R0227_REACHABILITY_PATH)
    geometry_signature = expected_input_signature(R0228_GEOMETRY_PATH)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0229 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        substrate_signature,
        truth_signature,
        reachability_signature,
        geometry_signature,
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    cache_root = os.path.join(ROUND_ROOT, "child-cache")
    scratch_root = os.path.join(ROUND_ROOT, "spill-scratch")
    partition_root = os.path.join(ROUND_ROOT, "shared-partition")

    sweep_dir = os.path.join(artifacts, SWEEP_CAPABILITY)
    sweep_path = os.path.join(sweep_dir, "nnd-quality-sweep.json")
    spill_dir = os.path.join(artifacts, SPILL_CAPABILITY)
    spill_path = os.path.join(spill_dir, "spill-reachability.json")
    retro_dir = os.path.join(artifacts, RETRO_CAPABILITY)
    retro_path = os.path.join(retro_dir, "registered-displacement-test.json")

    sweep_node = "sweep_nnd_quality_c16"
    spill_node = "sweep_spill_reachability"
    retro_node = "probe_retrospective_displacement"

    jobs = [
        {
            "id": sweep_node,
            "action": SWEEP_ACTION,
            "handler_module": "experiments.round0229_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [sweep_dir],
            "done_marker": os.path.join(artifacts, f"{sweep_node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": SWEEP_P90_WALL_S,
            "capability": SWEEP_CAPABILITY,
            "artifact_dir": sweep_dir,
            "artifact_path": sweep_path,
            "cuvs_cache_root": cache_root,
            "cache_root": cache_root,
            "scratch_root": scratch_root,
            "partition_root": partition_root,
            "substrate_signature": substrate_signature,
            "truth_signature": truth_signature,
            "r0227_reachability_signature": reachability_signature,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
        {
            "id": spill_node,
            "action": SPILL_ACTION,
            "handler_module": "experiments.round0229_nodes",
            "handler_callable": "run_job",
            "deps": [sweep_node],
            "outputs": [spill_dir],
            "done_marker": os.path.join(artifacts, f"{spill_node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": SPILL_P90_WALL_S,
            "capability": SPILL_CAPABILITY,
            "artifact_dir": spill_dir,
            "artifact_path": spill_path,
            "cuvs_cache_root": cache_root,
            "cache_root": cache_root,
            "substrate_signature": substrate_signature,
            "truth_signature": truth_signature,
            "r0227_reachability_signature": reachability_signature,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
        {
            "id": retro_node,
            "action": RETRO_ACTION,
            "handler_module": "experiments.round0229_nodes",
            "handler_callable": "run_job",
            "deps": [spill_node],
            "outputs": [retro_dir],
            "done_marker": os.path.join(artifacts, f"{retro_node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": RETRO_P90_WALL_S,
            "capability": RETRO_CAPABILITY,
            "artifact_dir": retro_dir,
            "artifact_path": retro_path,
            "r0228_geometry_signature": geometry_signature,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
    ]

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
        "schema": "round0229-nnd-quality-and-spill-reachability-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-benchmark",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-2m-substrate-and-exact-k15-graph-v1",
            "minilm-100m-low-cluster-count-graph-configuration-v1",
            "minilm-mixed-2m-cluster-spill-map-geometry-v1",
        ],
        "capabilities_produced": [
            SWEEP_CAPABILITY, SPILL_CAPABILITY, RETRO_CAPABILITY,
        ],
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            sweep_node: SWEEP_P90_WALL_S,
            spill_node: SPILL_P90_WALL_S,
            retro_node: RETRO_P90_WALL_S,
            "total": SWEEP_P90_WALL_S + SPILL_P90_WALL_S + RETRO_P90_WALL_S,
        },
        "scientific_contract": {
            "question": (
                "R0228 established that maps trained on cluster-spill-nnd "
                "graphs at c = 8 and c = 16 displace the rows that lost edges, "
                "with complete separation from the exact family (exact "
                "permutation p = 1/165, review-0228-01), while c = 4 does not "
                "(p = 0.436). Review-0227-01 recommended raising nn-descent's "
                "quality knobs to rescue high c. Can they? And if not, is there "
                "any (c, s) inside the device budget whose reachability ceiling "
                "reaches the c = 4 level at 50M or 100M?"
            ),
            "candidate": CANDIDATE,
            "structural_bound_note": STRUCTURAL_BOUND_NOTE,
            "structural_bound_is_falsifiable": (
                "a sweep cell above its own partition's measured strict ceiling "
                "means the ceiling instrument is wrong; the sweep measures that "
                "partition's ceiling itself rather than importing one"
            ),
            "recall_population": RECALL_POPULATION,
            "recall_population_note": RECALL_POPULATION_NOTE,
            "sweep": {
                "rows": ROWS,
                "clusters": SWEEP_CLUSTERS,
                "spill": SWEEP_SPILL,
                "cells": [dict(cell) for cell in QUALITY_SWEEP],
                "baseline_cell": BASELINE_CELL,
                "shared_partition_note": (
                    "one k-means partition, written by the first cell and bound "
                    "by every later cell, so a recall difference between two "
                    "nn-descent settings cannot be a difference in what was "
                    "reachable. A cell that cannot bind the cached partition "
                    "refuses."
                ),
                "ladder_axis": (
                    "ascending in nn-descent cost. The ladder stops at its "
                    "first refusal, abort, timeout or failure."
                ),
                "igd_host_law": (
                    "the intermediate graph is host-resident and quantised as "
                    "2 B/row x roundUp32(1.3 x igd); the RMM device relation "
                    "1048 B x max_cluster_rows is calibrated only at "
                    "graph_degree 32 / intermediate 48 and review-0227-01 "
                    "states it will not survive a change to either, so this "
                    "round measures device and host cost per setting and refits"
                ),
            },
            "spill_grid": {
                "rows": ROWS,
                "cells": [dict(cell) for cell in SPILL_GRID],
                "matched_family_note": (
                    "device cost is set by the largest cluster, so (c, s) pairs "
                    "with the same c / s cost the same memory; families hold "
                    "c / s constant so the only thing that varies is how the "
                    "same budget is spent"
                ),
                "controls": (
                    "(16, 2) and (4, 2) reproduce R0227's sealed strict "
                    "ceilings 0.953250 and 0.991562 over all 2,000,000 rows"
                ),
                "tie_query_rows": TIE_QUERY_ROWS,
                "query_seed": TIE_QUERY_SEED,
                "tie_sampling_disclosure": (
                    "the strict ceiling is over all 2,000,000 rows and is the "
                    "figure every decision uses; only the tie-aware ceiling is "
                    "sampled, at 200,000 seeded uniform rows, SE ~2e-4"
                ),
            },
            "displacement_test": {
                "statistic": DISPLACEMENT_TEST_NOTE,
                "decision_rule": DECISION_RULE_NOTE,
                "alpha": DISPLACEMENT_ALPHA,
                "labellings": PERMUTATION_LABELLINGS,
                "smallest_attainable_p": PERMUTATION_RESOLUTION_CEILING,
                "resolution_rule": RESOLUTION_RULE_NOTE,
                "trend_test": TREND_TEST_NOTE,
                "trend_assignments": TREND_ASSIGNMENTS,
                "trend_smallest_attainable_p": TREND_RESOLUTION_CEILING,
                "retrospective_note": (
                    "review-0228-01 already ran this test and got p = 0.43636 / "
                    "0.00606 / 0.00606 at c = 4 / 8 / 16 with complete "
                    "separation at the latter two; this node is third-codebase "
                    "confirmation, labelled as such, and adds only the same "
                    "test on R0223's cuVS arm plus smallest_attainable_p"
                ),
            },
            "phase2_trigger": {
                "note": PHASE2_TRIGGER_NOTE,
                "recall_trigger": PHASE2_RECALL_TRIGGER,
                "ceiling_trigger": PHASE2_CEILING_TRIGGER,
                "prepared_separately": True,
            },
            "per_rung_c": (
                "R0227's sealed MEASURED imbalance and nothing else; c values "
                "outside the measured set are never interpolated or modelled, "
                "which is review-0227-01's correction after R0227 published "
                "c = 22 off review-0226-01's model"
            ),
            "phase2_rungs": list(PHASE2_RUNGS),
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "substrate": SUBSTRATE_2M_PATH,
            "truth_ids": TRUTH_IDS_PATH,
            "truth_cosines": TRUTH_COS_PATH,
            "cluster_capacity_rows": CLUSTER_CAPACITY_ROWS,
            "spill_io_note": SPILL_IO_NOTE,
            "projection_discipline": (
                "every 50M/100M figure is labelled a projection, carries its "
                "fitted range and extrapolation factor, and no projection is "
                "divided by another projection. Spill I/O is its own line, "
                "never folded into a compute fit."
            ),
            "no_registered_cell_dropped": (
                "every cell of both grids appears in the published tables, "
                "including cells that refuse, abort, or complicate the story; "
                "review-0228-01 found R0228 dropped twelve registered cuVS "
                "tests and the occupied_bins instrument from its result"
            ),
            "guard_device_budget_bytes": GUARD_DEVICE_BUDGET_BYTES,
            "guard_host_anon_budget_bytes": GUARD_HOST_ANON_BUDGET_BYTES,
            "guard_swap_growth_abort_bytes": GUARD_SWAP_GROWTH_ABORT_BYTES,
            "guard_budget_note": GUARD_BUDGET_NOTE,
            "watchdog_poll_s": WATCHDOG_POLL_S,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "refusal_is_data": (
                "a cell whose predicted footprint exceeds a budget is refused "
                "before launch and recorded with its prediction; a cell whose "
                "REALISED largest cluster exceeds the registered capacity "
                "refuses itself after assignment and before any per-cluster "
                "build. Both are data: they measure where the configuration "
                "stops being launchable on this box."
            ),
            "never_sigkill_a_cuda_context": (
                "aborts are cooperative, then SIGTERM with a 180 s grace; "
                "SIGKILL is a recorded last resort and its use is an "
                "execution-check failure"
            ),
            "training_performed": False,
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
        "queue_manifest": prepare_round0229(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
