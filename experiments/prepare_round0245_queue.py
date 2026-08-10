"""Build R0245's queue. Never launches anything.

Three CPU-shaped nodes against bytes that already exist: R0244's sealed
watchdog receipt (differentiated for the allocation slope), R0242's and
R0243's per-row loss vectors, R0238's truth probe and substrate, R0240's graph
ids, and R0243's `10 GB` weight array. Nothing is rebuilt and no edge list is
republished.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0238_rung5 import GRAPH_K, TRUTH_PROBE_ROWS, TRUTH_PROBE_SEED
from basemap.round0240_rung5 import (
    INHERITED_REACHABILITY_MANIFEST,
    INHERITED_SUBSTRATE_MANIFEST,
    INHERITED_TRUTH_MANIFEST,
)
from basemap.round0241_qualify import (
    INHERITED_GRAPH_COS,
    INHERITED_GRAPH_IDS,
    INHERITED_LADDER_RECEIPT,
    REGISTERED_GRAPH_COS_SHA256,
    REGISTERED_GRAPH_IDS_SHA256,
    REGISTERED_LADDER_RECEIPT_SHA256,
    REGISTERED_SELECTED_CLUSTERS,
)
from basemap.round0244_prereq import (
    R0238_PROVENANCE_SHA256,
    R0238_REACHABILITY_VECTOR_SHA256,
    R0242_PROBE_BUILDER_MISSING_SHA256,
    R0243_EDGES_HEADER_SHA256,
    R0243_EDGES_WTS_SHA256,
    R0243_FUZZY_RECEIPT_SHA256,
    R0243_TIE_AWARE_BUILDER_MISSING_SHA256,
)
from basemap.round0245_did import (
    PERMUTATION_COST_NOTE,
    TIE_EFFECTIVE_RULE,
    TIE_SCORING_NOTE,
)
from basemap.round0245_guard import (
    ABORT_FLAG_NOTE,
    GPU_HOURS_CAP,
    POLL_SPACING_NOTE,
    R0244_BUDGET_HEADROOM_BYTES,
    R0244_MEASURED_SLOPE_BYTES_PER_S,
    R0244_STRIPE_LATENCY_S,
    ROUND_ID,
    ROWS,
    r0244_stripe_verdict,
)
from basemap.round0245_sampler import (
    BLIND_SPOT_NOTE,
    DRAW_COUNT_NOTE,
    R0245_SAMPLER_DRAWS,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0245_nodes import (
    DID_ACTION,
    DID_CAPABILITY,
    GUARD_ACTION,
    GUARD_CAPABILITY,
    NODE_ANON_BUDGET_BYTES,
    NODE_HEADROOM_BYTES,
    SAMPLER_ACTION,
    SAMPLER_CAPABILITY,
)

ROUND_ROOT = "/data/latent-basemap/runs/round-0245"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0245-2026-08-10.md")
HANDLER_MODULE = "experiments.round0245_nodes"

R0238_TRUTH_DIR = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-uniform-probe-k15-truth-v1"
)
R0238_REACHABILITY_DIR = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-cluster-spill-c400-reachability-v1"
)
R0242_VECTORS = (
    "/data/latent-basemap/runs/round-0242/queue-correction-1/artifacts/"
    "minilm-mixed-100000k-k15-loss-locality-v1/vectors"
)
R0243_RESIDUAL_VECTORS = (
    "/data/latent-basemap/runs/round-0243/queue/artifacts/"
    "minilm-mixed-100000k-k15-tie-aware-loss-locality-v1/vectors"
)
R0243_FUZZY_DIR = (
    "/data/latent-basemap/runs/round-0243/queue/artifacts/"
    "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1"
)
R0244_WATCHDOG_RECEIPT = (
    "/data/latent-basemap/runs/round-0244/queue/artifacts/"
    "round0244-threaded-host-watchdog-v1/host-watchdog.json"
)

SMALL_INPUTS: dict[str, str] = {
    "substrate_manifest": INHERITED_SUBSTRATE_MANIFEST,
    "truth_reference": INHERITED_TRUTH_MANIFEST,
    "reachability_reference": INHERITED_REACHABILITY_MANIFEST,
    "ladder_reference": INHERITED_LADDER_RECEIPT,
    "r0243_fuzzy_receipt": os.path.join(R0243_FUZZY_DIR, "fuzzy-graph.json"),
    "edges_header": os.path.join(R0243_FUZZY_DIR, "edges-k15-fuzzy-header.npz"),
    "r0244_watchdog_receipt": R0244_WATCHDOG_RECEIPT,
}
VECTOR_INPUTS: dict[str, str] = {
    "r0238_strict_reachability": os.path.join(
        R0238_REACHABILITY_DIR, "strict-c400.f64.npy"
    ),
    "probe_query_rows": os.path.join(R0238_TRUTH_DIR, "probe-query-rows.i64.npy"),
    "truth_ids": os.path.join(R0238_TRUTH_DIR, "truth-k15-ids.i32.npy"),
    "truth_cos": os.path.join(R0238_TRUTH_DIR, "truth-k15-cos.f32.npy"),
    "r0242_probe_builder_missing_edges": os.path.join(
        R0242_VECTORS, "probe-builder-missing-edges.i16.npy"
    ),
    "r0243_probe_tie_aware_builder_missing_edges": os.path.join(
        R0243_RESIDUAL_VECTORS, "probe-tie-aware-builder-missing-edges.i16.npy"
    ),
}
#: Bound at full sha256 here and by size in the node, but kept out of
#: `expected_inputs`: the v2.1 proportionate-verification rule.
BULK_INPUTS: dict[str, str] = {
    "graph_ids": INHERITED_GRAPH_IDS,
    "graph_cos": INHERITED_GRAPH_COS,
    "edges_wts": os.path.join(R0243_FUZZY_DIR, "edges-k15-fuzzy-wts.f32.npy"),
}
MANIFEST_INPUTS = ("substrate_array", "provenance")

REGISTERED_DIGESTS = (
    ("graph_ids", REGISTERED_GRAPH_IDS_SHA256),
    ("graph_cos", REGISTERED_GRAPH_COS_SHA256),
    ("ladder_reference", REGISTERED_LADDER_RECEIPT_SHA256),
    ("r0238_strict_reachability", R0238_REACHABILITY_VECTOR_SHA256),
    ("r0242_probe_builder_missing_edges", R0242_PROBE_BUILDER_MISSING_SHA256),
    (
        "r0243_probe_tie_aware_builder_missing_edges",
        R0243_TIE_AWARE_BUILDER_MISSING_SHA256,
    ),
    ("r0243_fuzzy_receipt", R0243_FUZZY_RECEIPT_SHA256),
    ("edges_header", R0243_EDGES_HEADER_SHA256),
    ("edges_wts", R0243_EDGES_WTS_SHA256),
)


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
        raise RuntimeError("R0245 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0245 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0245 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0245_contract.py", "tests/test_round0245_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0245 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0245_contract.py "
            "tests/test_round0245_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "ALL THREE node entry paths executed end to end through run_job on "
            "tiny inputs - the defect class that killed R0216 (NameError), "
            "R0236 (arity) and R0242 attempt 1 (missing module)",
            "the sampler-thread-death positive control: a non-OSError raised "
            "inside the sampling thread, the death recorded, the receipt "
            "reporting a dead sampler and poll() raising",
            "the missing-abort-flag positive control: ROUNDRUN_ABORT_FLAG "
            "removed and both the node precondition and the watchdog "
            "constructor refusing",
            "the allocation-slope positive control: a synthetic stage faster "
            "than the measured 11,767,996,416 B/s slope, stopped inside its "
            "headroom in the compliant arm and overshooting it in the "
            "breaching arm",
            "the DiD decision map ROUTING an underpowered null to "
            "INDETERMINATE, and refusing impossible inputs",
            "the arm-assignment gate REFUSING R0244's unclamped populations on "
            "a planted tie > strict row and accepting the clamped ones",
            "the sampler draw floor REFUSING a draw count below what the "
            "registered mis-sampler family needs",
            "a source-level assertion that no file this round adds contains a "
            "signalling construct - read, not delegated to a detector",
        ],
    }


def prepare_round0245(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    prior_gpu_wall_s: float = 0.0,
    only: tuple[str, ...] = (),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0245 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    signatures: dict[str, dict[str, Any]] = {}
    for key, path in {**SMALL_INPUTS, **VECTOR_INPUTS, **BULK_INPUTS}.items():
        if not os.path.exists(path):
            raise RuntimeError(f"R0245 inherited input absent: {key} at {path}")
        signatures[key] = expected_input_signature(path)
    for key, digest in REGISTERED_DIGESTS:
        if signatures[key].get("sha256") != digest:
            raise RuntimeError(
                f"R0245 STOP: {key} hashes to {signatures[key].get('sha256')}, "
                f"registered {digest}"
            )
    with open(INHERITED_SUBSTRATE_MANIFEST, encoding="utf-8") as handle:
        substrate_manifest = json.load(handle)
    signatures["substrate_array"] = dict(substrate_manifest["substrate"])
    signatures["provenance"] = dict(substrate_manifest["provenance"])
    if signatures["provenance"].get("sha256") != R0238_PROVENANCE_SHA256:
        raise RuntimeError("R0245 STOP: R0238 provenance digest moved")
    for key in MANIFEST_INPUTS:
        path = str(signatures[key]["canonical_path"])
        if not os.path.exists(path):
            raise RuntimeError(f"R0245 inherited input absent: {key} at {path}")
        if os.path.getsize(path) != int(signatures[key]["bytes"]):
            raise RuntimeError(f"R0245 {key} is not its declared size")

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0245 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )
    stripe = r0244_stripe_verdict()
    atomic_write_new_json(
        os.path.join(preflight, "poll-spacing-requirement.json"),
        {
            "derived_before_the_round_ran": stripe,
            "node_anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "node_headroom_bytes": NODE_HEADROOM_BYTES,
            "node_max_poll_spacing_s": (
                float(NODE_HEADROOM_BYTES)
                / float(R0244_MEASURED_SLOPE_BYTES_PER_S)
            ),
            "note": POLL_SPACING_NOTE,
        },
        immutable=True,
    )

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        *(signatures[key] for key in SMALL_INPUTS),
        *(signatures[key] for key in VECTOR_INPUTS),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    guard_dir = os.path.join(artifacts, GUARD_CAPABILITY)
    did_dir = os.path.join(artifacts, DID_CAPABILITY)
    sampler_dir = os.path.join(artifacts, SAMPLER_CAPABILITY)

    policy = {"gpu_required": True, "training_performed": False, "cpu_heavy": True}
    gpu_budget_remaining_s = float(GPU_HOURS_CAP * 3600.0 - prior_gpu_wall_s)

    jobs: list[dict[str, Any]] = [
        {
            "id": "guard_0245", "action": GUARD_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": [],
            "outputs": [guard_dir],
            "done_marker": os.path.join(artifacts, "guard_0245.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 600.0,
            "capability": GUARD_CAPABILITY,
            "r0244_watchdog_receipt": signatures["r0244_watchdog_receipt"],
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 1_800.0,
            "gpu_budget_remaining_s": gpu_budget_remaining_s,
            "node_policy": dict(policy),
        },
        {
            "id": "did_0245", "action": DID_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["guard_0245"],
            "outputs": [did_dir],
            "done_marker": os.path.join(artifacts, "did_0245.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 1_200.0,
            "capability": DID_CAPABILITY,
            **{key: signatures[key] for key in VECTOR_INPUTS},
            "substrate_array": signatures["substrate_array"],
            "graph_ids": signatures["graph_ids"],
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 3_600.0,
            "node_policy": dict(policy),
        },
        {
            "id": "sampler_0245", "action": SAMPLER_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["did_0245"],
            "outputs": [sampler_dir],
            "done_marker": os.path.join(artifacts, "sampler_0245.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 2_400.0,
            "capability": SAMPLER_CAPABILITY,
            "edges_header": signatures["edges_header"],
            "edges_wts": signatures["edges_wts"],
            "draws": R0245_SAMPLER_DRAWS,
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 5_400.0,
            "node_policy": dict(policy),
        },
    ]

    if only:
        keep = set(only)
        unknown = keep - {str(job["id"]) for job in jobs}
        if unknown:
            raise RuntimeError(f"R0245 has no such node(s): {sorted(unknown)}")
        jobs = [job for job in jobs if str(job["id"]) in keep]
        for job in jobs:
            job["deps"] = [dep for dep in job["deps"] if dep in keep]

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0245-guard-closure-and-did-gate-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-graph",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-100000k-nested-substrate-and-reserves-v1",
            "minilm-mixed-100000k-uniform-probe-k15-truth-v1",
            "minilm-mixed-100000k-cluster-spill-c400-reachability-v1",
            "minilm-mixed-100000k-cluster-spill-k15-neighbour-graph-v1",
            "minilm-mixed-100000k-cluster-spill-build-ladder-v1",
            "minilm-mixed-100000k-k15-loss-locality-v1",
            "minilm-mixed-100000k-k15-tie-aware-loss-locality-v1",
            "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1",
            "minilm-mixed-100000k-displacement-did-registration-v1",
            "minilm-mixed-100000k-k15-fuzzy-edge-sampler-v1",
            "round0244-threaded-host-watchdog-v1",
            "runner-signal-free-node-supervision-v1",
        ],
        "capabilities_produced": [
            GUARD_CAPABILITY, DID_CAPABILITY, SAMPLER_CAPABILITY,
        ],
        "prior_attempt_gpu_wall_s": float(prior_gpu_wall_s),
        "gpu_budget_remaining_s": gpu_budget_remaining_s,
        "substrate_inherited": True,
        "graph_inherited": True,
        "inherited_artifacts": [
            "graph", "ladder", "provenance", "r0242_vectors", "r0243_edges",
            "r0243_vectors", "r0244_watchdog_receipt", "reachability",
            "substrate", "truth",
        ],
        "inherited_from": [
            "/data/latent-basemap/runs/round-0238/queue",
            "/data/latent-basemap/runs/round-0240/queue",
            "/data/latent-basemap/runs/round-0242/queue-correction-1",
            "/data/latent-basemap/runs/round-0243/queue",
            "/data/latent-basemap/runs/round-0244/queue",
        ],
        "inheritance_note": (
            "Every byte this round reads already exists and is bound at its "
            "published digest. The only new bulk read is R0243's "
            "10,044,413,144-byte weight array, streamed once for the block "
            "profile the sampler power analysis needs. Nothing is rebuilt, no "
            "edge list is republished, and no map is trained."
        ),
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "total": sum(float(job["p90_wall_s"]) for job in jobs)
        },
        "scientific_contract": {
            "question": (
                "review-0244-01 accepted R0244 and found seven defects. Three "
                "are guard defects that gate the first multi-hour training "
                "node; two corrupt the displacement DiD before it costs 50 "
                "GPU-h; two more must be documented rather than silently "
                "accepted. This round closes the five and documents the two."
            ),
            "rows": ROWS,
            "k": GRAPH_K,
            "clusters": REGISTERED_SELECTED_CLUSTERS,
            "fix_1_sampler_thread_death": {
                "defect": (
                    "review-0244-01 A1: ThreadedHostWatchdog._loop caught only "
                    "OSError, so any other exception killed the sampling "
                    "thread with nothing raised, nothing logged and no node "
                    "gating on the coverage figure that would reveal it. On a "
                    "10 h node that is a silent reversion to the R0243 defect."
                ),
                "fix": (
                    "catch BaseException in _loop, record the death, raise "
                    "from poll() and from raise_if_thread_died(), publish "
                    "sampling_thread_alive in the receipt, and gate every "
                    "R0245 node on it. Thread death is now a visible failure "
                    "rather than an absence of observations."
                ),
                "positive_control": (
                    "a ValueError planted inside the sampling thread with the "
                    "budget deliberately untouched, so nothing but the death "
                    "itself can make the node stop."
                ),
            },
            "fix_2_enforcement_precondition": {
                "defect": ABORT_FLAG_NOTE,
                "fix": (
                    "require_enforceable_abort_flag() runs first in every "
                    "node and EnforcedHostWatchdog refuses to arm without a "
                    "writable flag path."
                ),
                "positive_control": (
                    "ROUNDRUN_ABORT_FLAG removed from the environment; both "
                    "the precondition and the constructor must refuse, and "
                    "the environment is restored in a finally."
                ),
            },
            "fix_3_poll_spacing": {
                "defect": (
                    "review-0244-01 A2: 0.25 s is the SAMPLING interval, not "
                    "the safety number. R0243's stage climbs at "
                    "11,767,996,416 B/s and its stripe latency is "
                    f"{R0244_STRIPE_LATENCY_S} s, so the guard permits about "
                    "41 GB of growth after it has decided to stop, against "
                    f"{R0244_BUDGET_HEADROOM_BYTES} B of declared headroom."
                ),
                "requirement": "poll_spacing_s <= headroom_bytes / slope_bytes_per_s",
                "derived_max_poll_spacing_s": stripe["max_poll_spacing_s"],
                "r0243_stripe_latency_s": R0244_STRIPE_LATENCY_S,
                "measured_slope_bytes_per_s": R0244_MEASURED_SLOPE_BYTES_PER_S,
                "headroom_bytes": R0244_BUDGET_HEADROOM_BYTES,
                "node_headroom_bytes": NODE_HEADROOM_BYTES,
                "positive_control": (
                    "a synthetic stage allocating written anonymous memory "
                    "faster than the measured slope, in two arms: one whose "
                    "unit meets the requirement and is stopped inside its "
                    "headroom, one whose unit violates it and overshoots. The "
                    "second arm is what makes the requirement non-vacuous."
                ),
                "note": POLL_SPACING_NOTE,
            },
            "fix_4_did_decision_gate": {
                "defect": (
                    "review-0244-01 section C: DID_DECISION_RULE is a string "
                    "constant with no did_decision() a future round must call "
                    "and no test that can plant an underpowered null into it."
                ),
                "fix": (
                    "did_decision() executes the registered rule and returns "
                    "HARMFUL / HARMLESS / INDETERMINATE, with the anti-vacuity "
                    "clause as a branch: a null whose power was not "
                    "demonstrated on the same maps is INDETERMINATE, never "
                    "harmless."
                ),
                "positive_control": (
                    "eight planted cases including three shapes of "
                    "underpowered null, each required to route as registered."
                ),
            },
            "fix_5_arm_assignment": {
                "defect": (
                    "review-0244-01 F1: probe rows 21785 and 453495 carry more "
                    "tie-aware loss than strict loss, and 21785 therefore "
                    "lands in both the treated genuine population and the "
                    "control pool."
                ),
                "diagnosis": TIE_SCORING_NOTE,
                "rule": TIE_EFFECTIVE_RULE,
                "positive_control": (
                    "the same planted row through R0244's unclamped "
                    "definitions must be refused by the disjointness gate and "
                    "through the clamped ones must pass it."
                ),
            },
            "finding_6_permutation_cost": PERMUTATION_COST_NOTE,
            "finding_7_sampler_power": {
                "draw_count": DRAW_COUNT_NOTE,
                "blind_spot": BLIND_SPOT_NOTE,
                "registered_draws": R0245_SAMPLER_DRAWS,
            },
            "truth_probe_rows": TRUTH_PROBE_ROWS,
            "truth_probe_seed": TRUTH_PROBE_SEED,
            "gpu_hours_cap": GPU_HOURS_CAP,
            "gpu_hours_cap_note": (
                "1.0 GPU-h. Every node is CPU-shaped: the round holds the GPU "
                "lease so no second workload can touch the card, and creates "
                "no CUDA context at all."
            ),
            "no_training": True,
            "no_gate_registered": True,
            "no_adoption_claimed": True,
            "no_map_quality_claimed": True,
            "no_displacement_measured": True,
            "no_atlas_claim": True,
            "no_build": True,
            "no_assembly": True,
        },
    })
    queue_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(queue_path, queue, immutable=True)
    return queue_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument("--prior-gpu-wall-s", type=float, default=0.0)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args(argv)
    path = prepare_round0245(
        release_sha=args.release_sha, queue_root=args.queue_root,
        prior_gpu_wall_s=args.prior_gpu_wall_s, only=tuple(args.only),
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
