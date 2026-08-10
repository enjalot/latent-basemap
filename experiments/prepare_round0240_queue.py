#!/usr/bin/env python3
"""Prepare, but never launch, the R0240 queue — build and qualify the 100M rung.

Two nodes. No assembly node exists by design: R0238's `153.6 GB` substrate, its
`500,000`-row uniform truth probe and its `c = 400` reachability ceiling are
declared at their FULL sha256 signatures and re-earned by nothing. Every
registered literal about them - all five ladder prefix hashes, the composition,
the union and increment shard coverage, the reserve disjointness, the ceiling -
is asserted inside the node before any CUDA context exists.

Everything else is R0238's queue wiring unchanged, because the nodes are
R0238's `run_ladder` and `run_qualify` reached through `round0240_nodes`. The
three things that are this round's own are its budget (9.0 GPU-h, owner-raised
2026-08-10), its deadlines (a 26,000 s cooperative build flag under a 32,000 s
runner soft deadline, so the runner's relayed flag is a backstop and never the
first thing to fire), and its carried prediction (R0238's MEASURED 2.8204385 at
this rung, not R0237's 2.456543 extrapolated from 50M).
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
    C_BUILD_MIN,
    DIMENSION,
    GRAPH_K,
    GUARD_IMBALANCE_MARGIN,
    IMBALANCE_PROBE_CLUSTERS,
    IMBALANCE_PROBE_ROWS,
    IMBALANCE_REPLICATE_SEEDS,
    MARGIN_NOTE,
    NN_DESCENT_SETTING,
    PRIMARY_IMBALANCE_SEED,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RECALL_POPULATION,
    SELECTION_CANDIDATES,
    SELECTION_LAW,
    SPILL,
    STRUCTURAL_POPULATION,
    TRUTH_METHOD,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
    ZERO_RECALL_FORENSIC_NOTE,
)
from basemap.round0240_rung5 import (
    ADVERSE_CARRY_NOTE,
    ATTEST_CAPABILITY,
    BUILD_TIMEOUT_NOTE,
    BUILD_TIMEOUT_S,
    CARRIED_IMBALANCE_AT_C400,
    CARRIED_IMBALANCE_SOURCE,
    COST_NOTE,
    GPU_HOURS_CAP,
    GPU_HOURS_CAP_NOTE,
    GRAPH_CAPABILITY,
    IMBALANCE_CAPABILITY,
    IMPLEMENTATION_NOTE,
    INHERITANCE_NOTE,
    INHERITED_REACHABILITY_MANIFEST,
    INHERITED_SUBSTRATE_MANIFEST,
    INHERITED_TRUTH_MANIFEST,
    LADDER_CAPABILITY,
    LADDER_P90_WALL_S,
    P90_WALL_NOTE,
    QUALIFICATION_NOTE,
    QUALIFY_P90_WALL_S,
    QUALIFY_RESERVE_S,
    R0238_REALISED_TOLERANCE_AT_C400,
    REGISTERED_LADDER_PREFIX_SHA256,
    REGISTERED_REACHABILITY_CEILING_C400,
    REGISTERED_RESERVE_ROWS,
    ROUND_ID,
    ROWS,
    SAFETY_NOTE,
    SELECTION_NOTE,
    TOLERANCE_PUBLICATION_NOTE,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0240_nodes import LADDER_ACTION_0240, QUALIFY_ACTION_0240


ROUND_ROOT = "/data/latent-basemap/runs/round-0240"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0240-2026-08-10.md")
HANDLER_MODULE = "experiments.round0240_nodes"

#: Inherited, sealed, hash-bound scientific inputs — identical to the set
#: R0238's ladder and qualification nodes read, so no device point, imbalance
#: figure or reachability reference can come from anywhere but a sealed file.
R0237_ROOT = "/data/latent-basemap/runs/round-0237/queue/artifacts"
R0237_LADDER = os.path.join(
    R0237_ROOT, "minilm-mixed-50000k-cluster-spill-build-ladder-v1",
    "build-ladder.json",
)
R0236_LADDER = (
    "/data/latent-basemap/runs/round-0236/queue-correction-2/artifacts/"
    "minilm-mixed-25000k-cluster-spill-build-ladder-v1/build-ladder.json"
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

INHERITED_INPUTS: dict[str, str] = {
    "substrate_manifest": INHERITED_SUBSTRATE_MANIFEST,
    "truth_reference": INHERITED_TRUTH_MANIFEST,
    "reachability_reference": INHERITED_REACHABILITY_MANIFEST,
    "r0237_ladder": R0237_LADDER,
    "r0236_ladder": R0236_LADDER,
    "r0235_ladder": R0235_LADDER,
    "r0233_ladder": R0233_LADDER,
    "r0229_sweep": R0229_SWEEP,
    "r0229_arm": R0229_ARM,
    "r0229_reachability": R0229_REACHABILITY,
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
        raise RuntimeError("R0240 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0240 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0240 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    #: No signalling bound on any child this round launches, anywhere, not even
    #: this CUDA-hidden CPU one. CPython implements that bound as `Popen.kill()`,
    #: and
    #: this program does not keep a construct that signals children in one place
    #: and forbids it in another (review-0238-01/F1). A hung smoke blocks
    #: preparation visibly, which is the correct failure.
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0240_contract.py", "tests/test_round0240_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0240 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0240_contract.py "
            "tests/test_round0240_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "the fail-closed inheritance verification against R0238's LIVE "
            "sealed manifests (must raise on a changed prefix hash, a changed "
            "composition, a coverage below 1.0, a non-disjoint reserve and a "
            "changed reachability ceiling)",
            "the refusal to accept an intra-queue reference for an inherited "
            "artifact (a reference with no sha256 must raise)",
            "the tolerance arithmetic, reproducing review-0238-01/F4's "
            "73.8255 / 51.3984 / 91.9620 percent from the sealed guard inputs",
            "the ladder attestation on a synthetic receipt, including the "
            "carried-prediction comparison and the per-seed movement table",
            "the qualification attestation, including that the R0215 tripwire "
            "reads BOTH in- and out-degree and does not hold when either is "
            "nonzero",
            "the wall-budget arithmetic under a 9.0 GPU-h cap",
            "the signal-safety detector over every file this round adds",
        ],
    }


def prepare_round0240(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    prior_gpu_wall_s: float = 0.0,
) -> str:
    """Build the queue. Never launches anything."""
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0240 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    signatures: dict[str, dict[str, Any]] = {}
    for key, path in INHERITED_INPUTS.items():
        if not os.path.exists(path):
            raise RuntimeError(f"R0240 inherited input absent: {key} at {path}")
        signatures[key] = expected_input_signature(path)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0240 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        *signatures.values(),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    cache_root = os.path.join(ROUND_ROOT, "child-cache")
    scratch_root = os.path.join(ROUND_ROOT, "spill-scratch")
    ladder_dir = os.path.join(artifacts, LADDER_CAPABILITY)
    ladder_manifest = os.path.join(ladder_dir, "build-ladder.json")
    graph_dir = os.path.join(artifacts, GRAPH_CAPABILITY)

    policy = {"gpu_required": True, "training_performed": False, "cpu_heavy": False}
    gpu_budget_remaining_s = float(GPU_HOURS_CAP * 3600.0 - prior_gpu_wall_s)

    jobs: list[dict[str, Any]] = [
        {
            "id": "ladder_100000k", "action": LADDER_ACTION_0240,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": [],
            "outputs": [ladder_dir],
            "done_marker": os.path.join(artifacts, "ladder_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": LADDER_P90_WALL_S,
            "capability": LADDER_CAPABILITY,
            "substrate_manifest": signatures["substrate_manifest"],
            "truth_reference": signatures["truth_reference"],
            "reachability_reference": signatures["reachability_reference"],
            "r0229_sweep": signatures["r0229_sweep"],
            "r0229_arm": signatures["r0229_arm"],
            "r0233_ladder": signatures["r0233_ladder"],
            "r0235_ladder": signatures["r0235_ladder"],
            "r0236_ladder": signatures["r0236_ladder"],
            "r0237_ladder": signatures["r0237_ladder"],
            "scratch_root": scratch_root,
            "cache_root": cache_root,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "gpu_budget_remaining_s": gpu_budget_remaining_s,
            "qualify_reserve_s": QUALIFY_RESERVE_S,
            "node_policy": dict(policy),
        },
        {
            "id": "qualify_100000k", "action": QUALIFY_ACTION_0240,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["ladder_100000k"],
            "outputs": [graph_dir],
            "done_marker": os.path.join(artifacts, "qualify_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": QUALIFY_P90_WALL_S,
            "capability": GRAPH_CAPABILITY,
            "substrate_manifest": signatures["substrate_manifest"],
            "truth_reference": signatures["truth_reference"],
            "reachability_reference": signatures["reachability_reference"],
            "ladder_reference": {
                "kind": "file", "canonical_path": ladder_manifest,
            },
            "r0229_reachability": signatures["r0229_reachability"],
            "r0233_ladder": signatures["r0233_ladder"],
            "r0235_ladder": signatures["r0235_ladder"],
            "r0236_ladder": signatures["r0236_ladder"],
            "r0237_ladder": signatures["r0237_ladder"],
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
        "schema": "round0240-100000k-build-and-qualify-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-graph",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-100000k-nested-substrate-and-reserves-v1",
            "minilm-mixed-100000k-uniform-probe-k15-truth-v1",
            "minilm-mixed-100000k-cluster-spill-c400-reachability-v1",
            "minilm-mixed-cluster-spill-s8-imbalance-100m-c400-five-seed-v1",
            "minilm-mixed-50000k-cluster-spill-build-ladder-v1",
            "minilm-mixed-25000k-cluster-spill-build-ladder-v1",
            "minilm-mixed-cluster-spill-s8-imbalance-replicate-grid-v1",
            "signal-free-gpu-child-supervision-v1",
            "runner-signal-free-node-supervision-v1",
        ],
        "capabilities_produced": [
            LADDER_CAPABILITY, GRAPH_CAPABILITY, IMBALANCE_CAPABILITY,
            ATTEST_CAPABILITY,
        ],
        "prior_attempt_gpu_wall_s": float(prior_gpu_wall_s),
        "gpu_budget_remaining_s": gpu_budget_remaining_s,
        "substrate_inherited": True,
        "substrate_reference": signatures["substrate_manifest"],
        "truth_inherited": True,
        "reachability_inherited": True,
        "inherited_artifacts": ["reachability", "substrate", "truth"],
        "inherited_from": (
            "/data/latent-basemap/runs/round-0238/queue"
        ),
        "inheritance_note": INHERITANCE_NOTE,
        "implementation_note": IMPLEMENTATION_NOTE,
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "total": sum(float(job["p90_wall_s"]) for job in jobs)
        },
        "scientific_contract": {
            "question": (
                "does cluster-spill-nnd at s = 8 and c = 400 - a partition no "
                "graph in this program has ever been built at - reproduce "
                "builder-independent brute-force truth on a registered uniform "
                "probe over 100,000,000 rows with NO edgeless row in either "
                "direction, at what realised cost in device, host anonymous "
                "memory, scratch, wall and physical I/O, and does this round's "
                "OWN five-seed imbalance at c = 400 still re-derive c = 400 "
                "under the unchanged margin after R0238 measured a +14.81% "
                "adverse move with the seed ranking inverted?"
            ),
            "rows": ROWS,
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "spill": SPILL,
            "nn_descent_setting": NN_DESCENT_SETTING,
            "selection_law": SELECTION_LAW,
            "selection_candidates": list(SELECTION_CANDIDATES),
            "selection_note": SELECTION_NOTE,
            "c_build_min": C_BUILD_MIN,
            "imbalance_probe_clusters": list(IMBALANCE_PROBE_CLUSTERS),
            "imbalance_probe_rows": list(IMBALANCE_PROBE_ROWS),
            "imbalance_replicate_seeds": list(IMBALANCE_REPLICATE_SEEDS),
            "primary_imbalance_seed": PRIMARY_IMBALANCE_SEED,
            "imbalance_margin": GUARD_IMBALANCE_MARGIN,
            "imbalance_margin_note": MARGIN_NOTE,
            "imbalance_margin_unchanged_note": (
                "IMBALANCE_GUARD_MARGIN stays at 1.164884. review-0236-01/F3 "
                "found no path from seed spread to OOM, so raising it creates "
                "a spurious-refusal hazard and defends nothing."
            ),
            "adverse_carry_note": ADVERSE_CARRY_NOTE,
            "carried_prediction_imbalance_at_c400": CARRIED_IMBALANCE_AT_C400,
            "carried_prediction_source": CARRIED_IMBALANCE_SOURCE,
            "carried_prediction_realised_tolerance": (
                R0238_REALISED_TOLERANCE_AT_C400
            ),
            "tolerance_publication_note": TOLERANCE_PUBLICATION_NOTE,
            "inherited_ladder_prefix_sha256": {
                str(rows): digest
                for rows, digest in sorted(
                    REGISTERED_LADDER_PREFIX_SHA256.items()
                )
            },
            "inherited_reachability_ceiling_c400": (
                REGISTERED_REACHABILITY_CEILING_C400
            ),
            "inherited_reserve_rows": REGISTERED_RESERVE_ROWS,
            "recall_population": RECALL_POPULATION,
            "structural_population": STRUCTURAL_POPULATION,
            "truth_method": TRUTH_METHOD,
            "truth_probe_rows": TRUTH_PROBE_ROWS,
            "truth_probe_seed": TRUTH_PROBE_SEED,
            "qualification_note": QUALIFICATION_NOTE,
            "zero_recall_forensic_note": ZERO_RECALL_FORENSIC_NOTE,
            "cost_note": COST_NOTE,
            "gpu_hours_cap": GPU_HOURS_CAP,
            "gpu_hours_cap_note": GPU_HOURS_CAP_NOTE,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "build_timeout_note": BUILD_TIMEOUT_NOTE,
            "p90_wall_note": P90_WALL_NOTE,
            "floors": {
                "tie_aware_mean": RECALL_MEAN_FLOOR,
                "tie_aware_p10": RECALL_P10_FLOOR,
                "zero_degree_rows_in_and_out": 0,
            },
            "safety": SAFETY_NOTE,
            "no_training": True,
            "no_gate_registered": True,
            "no_adoption_claimed": True,
            "no_map_quality_claimed": True,
            "no_assembly": True,
            "builds_the_100m_graph": True,
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
            "carried into the wall guard so a correction is sized against the "
            "round's real remaining cap (review-0224: charge every attempt)"
        ),
    )
    args = parser.parse_args(argv)
    path = prepare_round0240(
        release_sha=args.release_sha, queue_root=args.queue_root,
        prior_gpu_wall_s=args.prior_gpu_wall_s,
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
