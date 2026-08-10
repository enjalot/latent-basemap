#!/usr/bin/env python3
"""Prepare, but never launch, the R0241 queue — qualify the 100M k15 graph.

Two nodes, both cheap, both bound entirely to bytes that already exist.

* `tripwire_100000k` runs the R0215 degree-zero tripwire over all
  `100,000,000` rows in BOTH directions. It needs only the `6 GB` neighbour-id
  array. It is FIRST in the graph because R0240 proved that a `1.64 s` check
  can be held hostage for `4.3 h` behind a gather it does not need.
* `recall_100000k` scores R0238's already-registered `500,000`-row uniform
  probe (seed `238000`) against its sealed brute-force truth, states the
  probe's sampling power honestly, and discharges the three arithmetic debts.

Nothing is assembled, built or symmetrised. The fuzzy symmetrisation over all
`100,000,000` rows is a `~4.3 h` PRODUCT step and becomes its own registered
round after this qualification passes; ordering it first is what let a failed
product step destroy a passing science check in R0240.

The node module starts no child process at all. This preparation script starts
two — git and the CPU smoke — and neither carries a signalling wall bound of any
kind, because CPython implements that bound as a SIGKILL on expiry and this
program does not keep a construct that signals children in one place and
forbids it in another.
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
from basemap.round0238_rung5 import (
    DIMENSION,
    GRAPH_K,
    IMBALANCE_REPLICATE_SEEDS,
    PRIMARY_IMBALANCE_SEED,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RECALL_POPULATION,
    SPILL,
    STRUCTURAL_POPULATION,
    TRUTH_METHOD,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
    ZERO_RECALL_FORENSIC_NOTE,
)
from basemap.round0240_rung5 import (
    INHERITED_REACHABILITY_MANIFEST,
    INHERITED_SUBSTRATE_MANIFEST,
    INHERITED_TRUTH_MANIFEST,
    REGISTERED_LADDER_PREFIX_SHA256,
    REGISTERED_REACHABILITY_CEILING_C400,
)
from basemap.round0241_qualify import (
    CPU_ARITHMETIC_NOTE,
    DEBTS_NOTE,
    GPU_HOURS_CAP,
    GPU_HOURS_CAP_NOTE,
    INHERITANCE_NOTE,
    INHERITED_GRAPH_COS,
    INHERITED_GRAPH_IDS,
    INHERITED_LADDER_RECEIPT,
    MAX_ZERO_DEGREE_ROWS,
    NO_SYMMETRISATION_NOTE,
    PROBE_GATHER_BLOCK,
    RECALL_CAPABILITY,
    RECALL_DEADLINE_S,
    RECALL_NOTE,
    RECALL_RESERVE_S,
    REGISTERED_GRAPH_COS_SHA256,
    REGISTERED_GRAPH_IDS_SHA256,
    REGISTERED_LADDER_RECEIPT_SHA256,
    REGISTERED_SELECTED_CELL,
    REGISTERED_SELECTED_CLUSTERS,
    ROUND_ID,
    ROWS,
    SAFETY_NOTE,
    SCOPE_NOTE,
    SYMMETRISED_IDENTITY_NOTE,
    TRIPWIRE_BLOCK,
    TRIPWIRE_CAPABILITY,
    TRIPWIRE_DEADLINE_S,
    TRIPWIRE_NOTE,
    WALL_GUARD_NOTE,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0241_nodes import RECALL_ACTION, TRIPWIRE_ACTION


ROUND_ROOT = "/data/latent-basemap/runs/round-0241"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0241-2026-08-10.md")
HANDLER_MODULE = "experiments.round0241_nodes"

R0236_LADDER = (
    "/data/latent-basemap/runs/round-0236/queue-correction-2/artifacts/"
    "minilm-mixed-25000k-cluster-spill-build-ladder-v1/build-ladder.json"
)
R0237_LADDER = (
    "/data/latent-basemap/runs/round-0237/queue/artifacts/"
    "minilm-mixed-50000k-cluster-spill-build-ladder-v1/build-ladder.json"
)

#: Small, hash-bound scientific inputs. Every one of them is re-hashed by the
#: runner at every node boundary, which is why the two 6 GB graph arrays are
#: bound per job instead: they are verified once inside each node, at their
#: full signature, rather than four times by the boundary check.
SMALL_INPUTS: dict[str, str] = {
    "substrate_manifest": INHERITED_SUBSTRATE_MANIFEST,
    "truth_reference": INHERITED_TRUTH_MANIFEST,
    "reachability_reference": INHERITED_REACHABILITY_MANIFEST,
    "ladder_reference": INHERITED_LADDER_RECEIPT,
    "r0236_ladder": R0236_LADDER,
    "r0237_ladder": R0237_LADDER,
}
BULK_INPUTS: dict[str, str] = {
    "graph_ids": INHERITED_GRAPH_IDS,
    "graph_cos": INHERITED_GRAPH_COS,
}


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor",
         base_commit, release_sha],
        check=False,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0241 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0241 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0241 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0241_contract.py", "tests/test_round0241_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0241 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0241_contract.py "
            "tests/test_round0241_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "the symmetrised-isolation IDENTITY, proved against R0238's own "
            "_fuzzy_symmetrise_blocked on synthetic graphs that DO contain "
            "isolated rows - this is what licenses reporting in-degree without "
            "running the 4.3 h symmetrisation",
            "the degree pass cross-checked field by field against R0220's "
            "reviewed graph_validity through R0238's blocked wrapper",
            "the fail-closed inheritance verification against the LIVE sealed "
            "manifests, including a refusal when a graph hash differs",
            "the probe view's refusal to be indexed by anything but the "
            "registered uniform draw, and _score_probe reaching identical "
            "numbers through it",
            "the StageGuard refusing a stage a priori on its own measured rate, "
            "polling every unit, and raising in-band on its deadline",
            "the sampling-power arithmetic, including the exact hypergeometric "
            "smallest-detectable-region",
            "the tolerance chain reproducing 73.8255% and 51.3984% from sealed "
            "guard inputs",
            "a source-level assertion that no file this round adds contains a "
            "signalling construct - read, not delegated to a detector",
        ],
    }


def prepare_round0241(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    prior_gpu_wall_s: float = 0.0,
) -> str:
    """Build the queue. Never launches anything."""
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0241 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    signatures: dict[str, dict[str, Any]] = {}
    for key, path in {**SMALL_INPUTS, **BULK_INPUTS}.items():
        if not os.path.exists(path):
            raise RuntimeError(f"R0241 inherited input absent: {key} at {path}")
        signatures[key] = expected_input_signature(path)
    for key, digest in (
        ("graph_ids", REGISTERED_GRAPH_IDS_SHA256),
        ("graph_cos", REGISTERED_GRAPH_COS_SHA256),
        ("ladder_reference", REGISTERED_LADDER_RECEIPT_SHA256),
    ):
        if signatures[key].get("sha256") != digest:
            raise RuntimeError(
                f"R0241 STOP: {key} hashes to {signatures[key].get('sha256')}, "
                f"registered {digest}"
            )

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0241 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        *(signatures[key] for key in SMALL_INPUTS),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    tripwire_dir = os.path.join(artifacts, TRIPWIRE_CAPABILITY)
    recall_dir = os.path.join(artifacts, RECALL_CAPABILITY)

    policy = {"gpu_required": True, "training_performed": False, "cpu_heavy": False}
    gpu_budget_remaining_s = float(GPU_HOURS_CAP * 3600.0 - prior_gpu_wall_s)
    shared = {key: signatures[key] for key in ("substrate_manifest",
                                               "truth_reference",
                                               "reachability_reference",
                                               "ladder_reference",
                                               "graph_ids", "graph_cos")}

    jobs: list[dict[str, Any]] = [
        {
            "id": "tripwire_100000k", "action": TRIPWIRE_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": [],
            "outputs": [tripwire_dir],
            "done_marker": os.path.join(artifacts, "tripwire_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 600.0,
            "capability": TRIPWIRE_CAPABILITY,
            **shared,
            "stage_budget_s": TRIPWIRE_DEADLINE_S,
            "gpu_budget_remaining_s": gpu_budget_remaining_s,
            "recall_reserve_s": RECALL_RESERVE_S,
            "node_policy": dict(policy),
        },
        {
            "id": "recall_100000k", "action": RECALL_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["tripwire_100000k"],
            "outputs": [recall_dir],
            "done_marker": os.path.join(artifacts, "recall_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 900.0,
            "capability": RECALL_CAPABILITY,
            **shared,
            "r0236_ladder": signatures["r0236_ladder"],
            "r0237_ladder": signatures["r0237_ladder"],
            "probe_rows": TRUTH_PROBE_ROWS,
            "probe_seed": TRUTH_PROBE_SEED,
            "stage_budget_s": RECALL_DEADLINE_S,
            "node_policy": dict(policy),
        },
    ]

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0241-100000k-graph-qualification-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-graph",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-100000k-nested-substrate-and-reserves-v1",
            "minilm-mixed-100000k-uniform-probe-k15-truth-v1",
            "minilm-mixed-100000k-cluster-spill-c400-reachability-v1",
            "minilm-mixed-100000k-cluster-spill-k15-neighbour-graph-v1",
            "minilm-mixed-cluster-spill-s8-imbalance-100m-c400-five-seed-v2",
            "minilm-mixed-50000k-cluster-spill-build-ladder-v1",
            "minilm-mixed-25000k-cluster-spill-build-ladder-v1",
            "runner-signal-free-node-supervision-v1",
        ],
        "capabilities_produced": [TRIPWIRE_CAPABILITY, RECALL_CAPABILITY],
        "prior_attempt_gpu_wall_s": float(prior_gpu_wall_s),
        "gpu_budget_remaining_s": gpu_budget_remaining_s,
        "substrate_inherited": True,
        "graph_inherited": True,
        "inherited_artifacts": [
            "graph", "ladder", "reachability", "substrate", "truth",
        ],
        "inherited_from": [
            "/data/latent-basemap/runs/round-0238/queue",
            "/data/latent-basemap/runs/round-0240/queue",
        ],
        "inheritance_note": INHERITANCE_NOTE,
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "total": sum(float(job["p90_wall_s"]) for job in jobs)
        },
        "scientific_contract": {
            "question": (
                "does the 100,000,000-row c = 400, s = 8, k = 15 graph R0240 "
                "built carry ANY edgeless row - the v1 failure mode R0215 "
                "traced the clumps to - and does it reproduce "
                "builder-independent brute-force truth on the registered "
                "uniform probe at the registered floors?"
            ),
            "rows": ROWS,
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "spill": SPILL,
            "clusters": REGISTERED_SELECTED_CLUSTERS,
            "cell": REGISTERED_SELECTED_CELL,
            "tripwire_note": TRIPWIRE_NOTE,
            "symmetrised_identity_note": SYMMETRISED_IDENTITY_NOTE,
            "tripwire_block_rows": TRIPWIRE_BLOCK,
            "structural_population": STRUCTURAL_POPULATION,
            "recall_population": RECALL_POPULATION,
            "recall_note": RECALL_NOTE,
            "truth_method": TRUTH_METHOD,
            "truth_probe_rows": TRUTH_PROBE_ROWS,
            "truth_probe_seed": TRUTH_PROBE_SEED,
            "probe_is_the_registered_r0238_draw": True,
            "probe_gather_block_rows": PROBE_GATHER_BLOCK,
            "zero_recall_forensic_note": ZERO_RECALL_FORENSIC_NOTE,
            "inherited_ladder_prefix_sha256": {
                str(rows): digest
                for rows, digest in sorted(
                    REGISTERED_LADDER_PREFIX_SHA256.items()
                )
            },
            "inherited_graph_ids_sha256": REGISTERED_GRAPH_IDS_SHA256,
            "inherited_graph_cos_sha256": REGISTERED_GRAPH_COS_SHA256,
            "inherited_ladder_receipt_sha256": REGISTERED_LADDER_RECEIPT_SHA256,
            "inherited_reachability_ceiling_c400": (
                REGISTERED_REACHABILITY_CEILING_C400
            ),
            "imbalance_replicate_seeds": list(IMBALANCE_REPLICATE_SEEDS),
            "primary_imbalance_seed": PRIMARY_IMBALANCE_SEED,
            "debts_note": DEBTS_NOTE,
            "wall_guard_note": WALL_GUARD_NOTE,
            "cpu_arithmetic_note": CPU_ARITHMETIC_NOTE,
            "no_symmetrisation_note": NO_SYMMETRISATION_NOTE,
            "floors": {
                "tie_aware_mean": RECALL_MEAN_FLOOR,
                "tie_aware_p10": RECALL_P10_FLOOR,
                "symmetrised_isolated_rows": MAX_ZERO_DEGREE_ROWS,
                "out_degree_zero_rows": MAX_ZERO_DEGREE_ROWS,
                "raw_graph_zero_degree_rows": MAX_ZERO_DEGREE_ROWS,
            },
            "gpu_hours_cap": GPU_HOURS_CAP,
            "gpu_hours_cap_note": GPU_HOURS_CAP_NOTE,
            "safety": SAFETY_NOTE,
            "scope": SCOPE_NOTE,
            "no_training": True,
            "no_gate_registered": True,
            "no_adoption_claimed": True,
            "no_map_quality_claimed": True,
            "no_assembly": True,
            "no_build": True,
            "no_symmetrisation": True,
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
        "--prior-gpu-wall-s", type=float, default=0.0,
        help=(
            "GPU wall every earlier attempt of THIS round already charged; "
            "carried so a correction is sized against the round's real "
            "remaining cap (review-0224: charge every attempt)"
        ),
    )
    args = parser.parse_args(argv)
    path = prepare_round0241(
        release_sha=args.release_sha, queue_root=args.queue_root,
        prior_gpu_wall_s=args.prior_gpu_wall_s,
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
