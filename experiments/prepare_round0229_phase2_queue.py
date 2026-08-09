#!/usr/bin/env python3
"""Prepare, but never launch, the R0229 phase-2 queue.

Phase 2 runs only if the registered trigger fired on phase 1's sealed artifacts.
This script evaluates that trigger itself and refuses if it did not — the trigger
is not a judgement call made in prose.
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
from basemap.round0229_phase2_contract import (
    ADOPTION_CLAIMED,
    ARM_NAME,
    GEOMETRY_CAPABILITY,
    GRAPH_CAPABILITY,
    ROWS,
    SEEDS,
    TREATMENT_INVARIANT_SHA256,
    TREND_ARMS,
    map_capability,
    select_arm,
)
from basemap.round0229_quality_contract import (
    DECISION_RULE_NOTE,
    DISPLACEMENT_ALPHA,
    DISPLACEMENT_TEST_NOTE,
    PHASE2_TRIGGER_NOTE,
    RECALL_POPULATION,
    RECALL_POPULATION_NOTE,
    RESOLUTION_RULE_NOTE,
    ROUND_ID,
    TREND_TEST_NOTE,
    guard_for_spill,
    phase2_trigger,
)
from experiments.round0229_phase2_nodes import (
    BUILD_ACTION,
    FUZZY_ACTION,
    GEOMETRY_ACTION,
    TRAIN_ACTION,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0229"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue-phase2")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0229-2026-08-09.md")

PHASE1_ROOT = os.path.join(ROUND_ROOT, "queue-correction-1", "artifacts")
SWEEP_PATH = os.path.join(
    PHASE1_ROOT, "minilm-mixed-2m-nnd-quality-sweep-v1", "nnd-quality-sweep.json"
)
SPILL_PATH = os.path.join(
    PHASE1_ROOT, "minilm-mixed-2m-spill-reachability-v1", "spill-reachability.json"
)
R0228_GEOMETRY_PATH = (
    "/data/latent-basemap/runs/round-0228/queue-correction-1/artifacts/"
    "minilm-mixed-2m-cluster-spill-map-geometry-v1/cluster-spill-map-geometry.json"
)
R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1"
)
R0216_GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")

#: The eight exact-graph maps: R0218 seeds 42-45, R0222 seeds 46-49. These are
#: declared as DIRECT queue inputs, which review-0228-01 asked for after R0228
#: bound them one level indirect through its comparison artifact.
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

BUILD_P90_WALL_S = 1_800.0
FUZZY_P90_WALL_S = 900.0
TRAIN_P90_WALL_S = 1_800.0
GEOMETRY_P90_WALL_S = 900.0


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
        raise RuntimeError("R0229 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0229 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    """The model-producing release's CUDA-hidden CPU smoke.

    It reaches the actual post-fit accounting, the checkpoint publish, the
    full-population reload, the coordinate publication and the receipt seal. It
    is preparation validation, not a queue node and not a scientific result --
    and it is what caught this round's `train_config` cell-list refusal before
    a single GPU second was spent on it.
    """
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
         "tests/test_round0229_phase2_cpu_smoke.py",
         "tests/test_round0229_phase2.py", "tests/test_round0229_quality.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
        timeout=240,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0229 phase-2 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0229_phase2_cpu_smoke.py "
            "tests/test_round0229_phase2.py tests/test_round0229_quality.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "ceil-derived dose", "R0217-template config construction",
            "cross-round treatment-digest equality", "post-fit accounting",
            "checkpoint publish", "full-population reload",
            "coordinate publication", "receipt seal",
        ],
    }


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def prepare_round0229_phase2(
    *, release_sha: str, queue_root: str = QUEUE_ROOT
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0229 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    sweep = _load(SWEEP_PATH)
    spill = _load(SPILL_PATH)

    # The registered trigger, evaluated mechanically. Phase 2 does not run on a
    # judgement call.
    trigger = phase2_trigger(
        sweep_cells=sweep["cells"],
        spill_cells=spill["cells"],
        partition_strict_ceiling=float(
            sweep["shared_partition"]["strict_ceiling_all_rows"]
        ),
    )
    if not trigger["phase2_runs"]:
        raise RuntimeError(
            "R0229 phase-2 trigger did not fire; phase 2 must not run. "
            f"{json.dumps(trigger, indent=2, sort_keys=True)}"
        )
    arm = select_arm(sweep=sweep, spill=spill)
    guard = guard_for_spill(
        rows=ROWS, clusters=int(arm["clusters"]), spill=int(arm["spill"])
    )
    if not guard.get("allowed"):
        raise RuntimeError(f"R0229 phase-2 arm is refused a priori: {guard}")

    for path in (
        SUBSTRATE_2M_PATH, TRUTH_RECEIPT_PATH, R0228_GEOMETRY_PATH,
        R0216_GRAPH_MANIFEST, *EXACT_COORDINATES.values(),
    ):
        if not os.path.exists(path):
            raise RuntimeError(f"R0229 phase-2 input is absent at {path}")

    substrate_signature = expected_input_signature(SUBSTRATE_2M_PATH)
    truth_signature = expected_input_signature(TRUTH_RECEIPT_PATH)
    geometry_signature = expected_input_signature(R0228_GEOMETRY_PATH)
    sweep_signature = expected_input_signature(SWEEP_PATH)
    spill_signature = expected_input_signature(SPILL_PATH)
    r0216_manifest_signature = expected_input_signature(R0216_GRAPH_MANIFEST)
    r0216_graph_signature = dict(_load(R0216_GRAPH_MANIFEST)["graph"])
    exact_signatures = {
        seed: expected_input_signature(path)
        for seed, path in sorted(EXACT_COORDINATES.items())
    }

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0229 phase-2 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    selection_path = os.path.join(preflight, "arm-selection.json")
    atomic_write_new_json(
        selection_path,
        {
            "trigger": trigger, "arm": arm, "guard": guard,
            "sweep": sweep_signature, "spill": spill_signature,
            "note": (
                "the registered trigger and the registered arm-selection rule, "
                "evaluated at prepare time from phase 1's sealed artifacts and "
                "re-evaluated inside every node against the same bytes"
            ),
        },
        immutable=True,
    )

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(selection_path),
        substrate_signature, truth_signature, geometry_signature,
        sweep_signature, spill_signature, r0216_manifest_signature,
        *exact_signatures.values(),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    cache_root = os.path.join(ROUND_ROOT, "child-cache")
    scratch_root = os.path.join(ROUND_ROOT, "spill-scratch")

    build_dir = os.path.join(artifacts, "spill-lifted-build")
    build_path = os.path.join(build_dir, "spill-lifted-build.json")
    graph_dir = os.path.join(artifacts, GRAPH_CAPABILITY)
    graph_manifest = os.path.join(graph_dir, "spill-lifted-graph.json")
    geometry_dir = os.path.join(artifacts, GEOMETRY_CAPABILITY)
    geometry_path = os.path.join(geometry_dir, "spill-lifted-geometry.json")

    common = {
        "arm": arm,
        "sweep_signature": sweep_signature,
        "spill_signature": spill_signature,
        "substrate_signature": substrate_signature,
        "truth_signature": truth_signature,
    }
    intra = lambda path: {"kind": "file", "canonical_path": path}  # noqa: E731

    jobs: list[dict[str, Any]] = [
        {
            "id": "build_spill_lifted", "action": BUILD_ACTION,
            "handler_module": "experiments.round0229_phase2_nodes",
            "handler_callable": "run_job", "deps": [],
            "outputs": [build_dir],
            "done_marker": os.path.join(artifacts, "build_spill_lifted.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": BUILD_P90_WALL_S,
            "artifact_dir": build_dir, "artifact_path": build_path,
            "cache_root": cache_root, "cuvs_cache_root": cache_root,
            "scratch_root": scratch_root, "guard": guard,
            **common,
            "node_policy": {
                "gpu_required": True, "training_performed": False,
                "cpu_heavy": False,
            },
        },
        {
            "id": "fuzzy_spill_lifted", "action": FUZZY_ACTION,
            "handler_module": "experiments.round0229_phase2_nodes",
            "handler_callable": "run_job", "deps": ["build_spill_lifted"],
            "outputs": [graph_dir],
            "done_marker": os.path.join(artifacts, "fuzzy_spill_lifted.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": FUZZY_P90_WALL_S,
            "capability": GRAPH_CAPABILITY,
            "build_reference": intra(build_path),
            **common,
            "node_policy": {
                "gpu_required": True, "training_performed": False,
                "cpu_heavy": False,
            },
        },
    ]
    train_nodes: list[str] = []
    for seed in SEEDS:
        node = f"train_spill_lifted_seed{seed}"
        train_nodes.append(node)
        output = os.path.join(artifacts, map_capability(seed))
        jobs.append({
            "id": node, "action": TRAIN_ACTION,
            "handler_module": "experiments.round0229_phase2_nodes",
            "handler_callable": "run_job",
            "deps": ["fuzzy_spill_lifted"] + train_nodes[:-1],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": TRAIN_P90_WALL_S,
            "capability": map_capability(seed),
            "training_seed": seed,
            "graph_manifest_reference": intra(graph_manifest),
            "r0216_graph_signature": r0216_graph_signature,
            "r0216_graph_manifest_signature": r0216_manifest_signature,
            "treatment_invariant_sha256": TREATMENT_INVARIANT_SHA256,
            **common,
            "node_policy": {
                "gpu_required": True, "training_performed": True,
                "cpu_heavy": False,
            },
        })

    jobs.append({
        "id": "probe_spill_lifted_geometry", "action": GEOMETRY_ACTION,
        "handler_module": "experiments.round0229_phase2_nodes",
        "handler_callable": "run_job", "deps": list(train_nodes),
        "outputs": [geometry_dir],
        "done_marker": os.path.join(
            artifacts, "probe_spill_lifted_geometry.done.json"
        ),
        "expected_inputs": expected_inputs,
        "p90_wall_s": GEOMETRY_P90_WALL_S,
        "capability": GEOMETRY_CAPABILITY,
        "artifact_dir": geometry_dir, "artifact_path": geometry_path,
        "graph_manifest_reference": intra(graph_manifest),
        "r0228_geometry_signature": geometry_signature,
        "candidate_coordinates": [
            {
                "name": f"{ARM_NAME}-seed{seed}",
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

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0229-spill-lifted-map-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-train",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-2m-substrate-and-exact-k15-graph-v1",
            "minilm-mixed-2m-nnd-quality-sweep-v1",
            "minilm-mixed-2m-spill-reachability-v1",
            "minilm-mixed-2m-cluster-spill-map-geometry-v1",
        ],
        "capabilities_produced": [
            GRAPH_CAPABILITY, *[map_capability(seed) for seed in SEEDS],
            GEOMETRY_CAPABILITY,
        ],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "total": BUILD_P90_WALL_S + FUZZY_P90_WALL_S
            + len(SEEDS) * TRAIN_P90_WALL_S + GEOMETRY_P90_WALL_S
        },
        "scientific_contract": {
            "question": (
                "phase 1 fired the registered structural-gain trigger: at "
                "matched device cost the reachability ceiling rises steeply "
                "with spill, and 100M-feasible configurations exist whose "
                "ceiling clears the c = 4 level R0228 found clean. Does a map "
                "trained on one of them behave like c = 4 or like c = 16?"
            ),
            "phase2_trigger": trigger,
            "phase2_trigger_note": PHASE2_TRIGGER_NOTE,
            "arm_selection": arm,
            "arm": ARM_NAME,
            "seeds": list(SEEDS),
            "recall_population": RECALL_POPULATION,
            "recall_population_note": RECALL_POPULATION_NOTE,
            "treatment": (
                "R0217's in every respect except the seed and the graph; each "
                "cell rebuilds its config from R0217's own template and refuses "
                "to train unless the treatment-invariant digest equals the "
                f"cross-round constant {TREATMENT_INVARIANT_SHA256}"
            ),
            "treatment_invariant_sha256": TREATMENT_INVARIANT_SHA256,
            "dose_rule": (
                "ceil(1e6 x active_directed_edges / 603,086,368) applied to this "
                "graph's own sealed edge count; derived, never carried"
            ),
            "displacement_test": DISPLACEMENT_TEST_NOTE,
            "decision_rule": DECISION_RULE_NOTE,
            "trend_test": TREND_TEST_NOTE,
            "trend_arms": list(TREND_ARMS),
            "alpha": DISPLACEMENT_ALPHA,
            "resolution_rule": RESOLUTION_RULE_NOTE,
            "multiplicity": (
                "one new arm means one new test at alpha = 0.05; the registered "
                "Holm-Bonferroni machinery is inert at a single test and is "
                "reported as inert rather than silently omitted"
            ),
            "geometry_code": (
                "R0228's basemap/round0228_geometry.py imported read-only, with "
                "R0228's registered constants, so the numbers are directly "
                "comparable with R0228's c = 4 / 8 / 16"
            ),
            "exact_coordinates_are_direct_queue_inputs": (
                "review-0228-01 noted R0228 bound the coordinate arrays one "
                "level indirect through its comparison artifact; the eight "
                "exact-family arrays are declared queue inputs here"
            ),
            "guard": guard,
            "guard_budget_note": GUARD_BUDGET_NOTE,
            "watchdog_poll_s": WATCHDOG_POLL_S,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "never_sigkill_a_cuda_context": (
                "aborts are cooperative, then SIGTERM with a 180 s grace; "
                "SIGKILL is a recorded last resort and an execution-check failure"
            ),
            "training_performed": True,
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
        "queue_manifest": prepare_round0229_phase2(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
