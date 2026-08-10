"""Build R0247's queue. Never launches anything.

Three CPU-shaped nodes against bytes that already exist: R0238's truth probe and
substrate (for the `float64` cosine recompute and the whole-probe flip-rate
measurement) and R0240's graph ids. The only new bytes are a `60` MB `float64`
truth-cosine array, produced by node 2 and consumed by node 3. Nothing is
rebuilt, no edge list is republished, and no map is trained.
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
from basemap.round0227_low_c_contract import TIE_TOLERANCE
from basemap.round0238_rung5 import GRAPH_K, TRUTH_PROBE_ROWS, TRUTH_PROBE_SEED
from basemap.round0240_rung5 import (
    INHERITED_SUBSTRATE_MANIFEST,
    INHERITED_TRUTH_MANIFEST,
)
from basemap.round0241_qualify import (
    INHERITED_GRAPH_IDS,
    REGISTERED_GRAPH_IDS_SHA256,
    REGISTERED_SELECTED_CLUSTERS,
)
from basemap.round0244_prereq import R0238_PROVENANCE_SHA256
from basemap.round0246_tie import (
    TIE_AGGREGATE_ONLY_RULE,
    TIE_AWARE_CLAIM_LEDGER,
    TIE_CLAIM_MAX_EXPECTED_FLIPS_OVER_MARGIN,
    TIE_PRECISION_ROWS,
    TIE_PRECISION_SEED,
    TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN,
)
from basemap.round0247_guard import R0245_SEALED_BLOCKER_GAP_S
from basemap.round0247_registry import (
    CONSTRUCTION_PATH_NOTE,
    GPU_HOURS_CAP,
    REGISTERED_REGISTRY_SHA256,
    REGISTERED_SAFETY_PARAMETERS,
    ROUND_ID,
    ROWS,
    SAFETY_PARAMETER_CLASS_NOTE,
    registry_fingerprint,
    safety_parameter_inventory,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0247_nodes import (
    NODE_ANON_BUDGET_BYTES,
    PARAMGUARD_ACTION,
    PARAMGUARD_CAPABILITY,
    TIE_ACTION,
    TIE_CAPABILITY,
    TIE_FULL_PROBE_ROWS,
    TRUTHCOS_ACTION,
    TRUTHCOS_CAPABILITY,
    TRUTH_COS_F64_FILE,
)

ROUND_ROOT = "/data/latent-basemap/runs/round-0247"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0247-2026-08-10.md")
HANDLER_MODULE = "experiments.round0247_nodes"

R0238_TRUTH_DIR = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-uniform-probe-k15-truth-v1"
)

SMALL_INPUTS: dict[str, str] = {
    "substrate_manifest": INHERITED_SUBSTRATE_MANIFEST,
    "truth_reference": INHERITED_TRUTH_MANIFEST,
}
VECTOR_INPUTS: dict[str, str] = {
    "probe_query_rows": os.path.join(R0238_TRUTH_DIR, "probe-query-rows.i64.npy"),
    "truth_ids": os.path.join(R0238_TRUTH_DIR, "truth-k15-ids.i32.npy"),
    "truth_cos": os.path.join(R0238_TRUTH_DIR, "truth-k15-cos.f32.npy"),
}
#: Bound at full sha256 here and by size in the node, but kept out of
#: `expected_inputs`: the v2.1 proportionate-verification rule.
BULK_INPUTS: dict[str, str] = {"graph_ids": INHERITED_GRAPH_IDS}
MANIFEST_INPUTS = ("substrate_array", "provenance")

REGISTERED_DIGESTS = (("graph_ids", REGISTERED_GRAPH_IDS_SHA256),)


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
        raise RuntimeError("R0247 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0247 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0247 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0247_contract.py", "tests/test_round0247_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0247 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0247_contract.py "
            "tests/test_round0247_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "ALL THREE node entry paths executed end to end through run_job on "
            "tiny inputs - the defect class that killed R0216 (NameError), "
            "R0236 (arity) and R0242 attempt 1 (missing module)",
            "one positive control per registered safety parameter: the "
            "override attempted through its own construction path, the "
            "registered value used, the attempt recorded, and the gate "
            "refusing it",
            "review-0246-01 C's sixteenth attack in its exact construction - "
            f"R0245's sealed {R0245_SEALED_BLOCKER_GAP_S} s blocker gap with "
            "headroom 1 << 50 and max_poll_spacing_s 1e6 and "
            "training_performed true - REFUSED, with "
            "registered_max_poll_spacing_s reporting the registry",
            "review-0246-01 A's 5.0 s coverage attack refused on three arms "
            "that do not share a denominator",
            "the self-attack battery: a module-global assignment, a write into "
            "the registry, a scripted clock, a no-op abort reader, a replay "
            "sealed as evidence, the unguarded base class, and a fabricated "
            "receipt - with the one that still succeeds published",
            "the float64 truth-cosine recompute from the sealed truth ids, and "
            "the storage-versus-arithmetic decomposition it settles",
            "the Poisson upper bound reducing to the rule of three at zero "
            "events, and the sealed ledger adjudication at both registered "
            "criteria",
            "a source-level assertion that no file this round adds contains a "
            "signalling construct - read, not delegated to a detector",
        ],
    }


def prepare_round0247(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    prior_gpu_wall_s: float = 0.0,
    only: tuple[str, ...] = (),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0247 release SHA must be one full commit")
    if registry_fingerprint() != REGISTERED_REGISTRY_SHA256:
        raise RuntimeError(
            "R0247 STOP: the safety-parameter registry fingerprint "
            f"{registry_fingerprint()} does not match the pinned "
            f"{REGISTERED_REGISTRY_SHA256}"
        )
    round_signature, required_reviews = _issued_round(release_sha)

    signatures: dict[str, dict[str, Any]] = {}
    for key, path in {**SMALL_INPUTS, **VECTOR_INPUTS, **BULK_INPUTS}.items():
        if not os.path.exists(path):
            raise RuntimeError(f"R0247 inherited input absent: {key} at {path}")
        signatures[key] = expected_input_signature(path)
    for key, digest in REGISTERED_DIGESTS:
        if signatures[key].get("sha256") != digest:
            raise RuntimeError(
                f"R0247 STOP: {key} hashes to {signatures[key].get('sha256')}, "
                f"registered {digest}"
            )
    with open(INHERITED_SUBSTRATE_MANIFEST, encoding="utf-8") as handle:
        substrate_manifest = json.load(handle)
    signatures["substrate_array"] = dict(substrate_manifest["substrate"])
    signatures["provenance"] = dict(substrate_manifest["provenance"])
    if signatures["provenance"].get("sha256") != R0238_PROVENANCE_SHA256:
        raise RuntimeError("R0247 STOP: R0238 provenance digest moved")
    for key in MANIFEST_INPUTS:
        path = str(signatures[key]["canonical_path"])
        if not os.path.exists(path):
            raise RuntimeError(f"R0247 inherited input absent: {key} at {path}")
        if os.path.getsize(path) != int(signatures[key]["bytes"]):
            raise RuntimeError(f"R0247 {key} is not its declared size")

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0247 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )
    atomic_write_new_json(
        os.path.join(preflight, "registered-safety-parameters.json"),
        {
            "registered_before_the_round_ran": safety_parameter_inventory(),
            "node_anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "registry_fingerprint": registry_fingerprint(),
            "registered_registry_sha256": REGISTERED_REGISTRY_SHA256,
            "class_note": SAFETY_PARAMETER_CLASS_NOTE,
            "construction_path_note": CONSTRUCTION_PATH_NOTE,
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
    paramguard_dir = os.path.join(artifacts, PARAMGUARD_CAPABILITY)
    truthcos_dir = os.path.join(artifacts, TRUTHCOS_CAPABILITY)
    tie_dir = os.path.join(artifacts, TIE_CAPABILITY)

    policy = {"gpu_required": True, "training_performed": False, "cpu_heavy": True}
    gpu_budget_remaining_s = float(GPU_HOURS_CAP * 3600.0 - prior_gpu_wall_s)

    jobs: list[dict[str, Any]] = [
        {
            "id": "paramguard_0247", "action": PARAMGUARD_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": [],
            "outputs": [paramguard_dir],
            "done_marker": os.path.join(artifacts, "paramguard_0247.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 600.0,
            "capability": PARAMGUARD_CAPABILITY,
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 1_800.0,
            "gpu_budget_remaining_s": gpu_budget_remaining_s,
            "node_policy": dict(policy),
        },
        {
            "id": "truthcos_0247", "action": TRUTHCOS_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["paramguard_0247"],
            "outputs": [truthcos_dir],
            "done_marker": os.path.join(artifacts, "truthcos_0247.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 1_800.0,
            "capability": TRUTHCOS_CAPABILITY,
            **{key: signatures[key] for key in VECTOR_INPUTS},
            "substrate_array": signatures["substrate_array"],
            "truth_rows": TRUTH_PROBE_ROWS,
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 5_400.0,
            "node_policy": dict(policy),
        },
        {
            "id": "tie_0247", "action": TIE_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["truthcos_0247"],
            "outputs": [tie_dir],
            "done_marker": os.path.join(artifacts, "tie_0247.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 3_600.0,
            "capability": TIE_CAPABILITY,
            **{key: signatures[key] for key in VECTOR_INPUTS},
            "substrate_array": signatures["substrate_array"],
            "graph_ids": signatures["graph_ids"],
            #: Produced by `truthcos_0247` inside this queue, so it is bound by
            #: path and checked by the node rather than declared as an input
            #: that must already exist at preflight.
            "truth_cos_f64": {
                "canonical_path": os.path.join(truthcos_dir, TRUTH_COS_F64_FILE),
                "bytes": -1,
                "produced_by": "truthcos_0247",
            },
            "replication_rows": TIE_PRECISION_ROWS,
            "full_probe_rows": TIE_FULL_PROBE_ROWS,
            "tie_seed": TIE_PRECISION_SEED,
            "anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
            "stage_budget_s": 7_200.0,
            "node_policy": dict(policy),
        },
    ]

    if only:
        keep = set(only)
        unknown = keep - {str(job["id"]) for job in jobs}
        if unknown:
            raise RuntimeError(f"R0247 has no such node(s): {sorted(unknown)}")
        jobs = [job for job in jobs if str(job["id"]) in keep]
        for job in jobs:
            job["deps"] = [dep for dep in job["deps"] if dep in keep]

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0247-safety-parameter-registry-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-graph",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-100000k-nested-substrate-and-reserves-v1",
            "minilm-mixed-100000k-uniform-probe-k15-truth-v1",
            "minilm-mixed-100000k-cluster-spill-k15-neighbour-graph-v1",
            "minilm-mixed-100000k-tie-aware-aggregate-only-v1",
            "round0244-threaded-host-watchdog-v1",
            "round0245-guard-closure-v1",
            "round0246-guard-closure-v1",
            "runner-signal-free-node-supervision-v1",
        ],
        "capabilities_produced": [
            PARAMGUARD_CAPABILITY, TRUTHCOS_CAPABILITY, TIE_CAPABILITY,
        ],
        "prior_attempt_gpu_wall_s": float(prior_gpu_wall_s),
        "gpu_budget_remaining_s": gpu_budget_remaining_s,
        "substrate_inherited": True,
        "graph_inherited": True,
        "inherited_artifacts": ["graph", "provenance", "substrate", "truth"],
        "inherited_from": [
            "/data/latent-basemap/runs/round-0238/queue",
            "/data/latent-basemap/runs/round-0240/queue",
        ],
        "inheritance_note": (
            "Every byte this round reads already exists and is bound at its "
            "published digest. The only new bytes are a 60,000,000-byte "
            "float64 truth-cosine array recomputed from R0238's SEALED truth "
            "ids - the operation review-0246-01 F priced at a CPU gather "
            "rather than the 100M-row GPU job result-0246 quoted. Nothing is "
            "rebuilt, no edge list is republished, and no map is trained."
        ),
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "total": sum(float(job["p90_wall_s"]) for job in jobs)
        },
        "scientific_contract": {
            "question": (
                "review-0246-01 defeated R0246 twice in the same shape R0246 "
                "believed it had closed: a registered safety bound reached the "
                "decision as a constructor keyword, and the receipt published "
                "the caller's value under a registered_* key. R0247 fixes the "
                "CLASS rather than the five instances, reconciles the ledger "
                "and its rule text, and runs the precision fix the reviewer "
                "showed was cheap."
            ),
            "rows": ROWS,
            "k": GRAPH_K,
            "clusters": REGISTERED_SELECTED_CLUSTERS,
            "closure_1_the_class": {
                "defect": (
                    "review-0246-01 C: AbortPollGate(max_poll_spacing_s=...) "
                    "overrode the registered 2.5109531834854018 s outright and "
                    "the receipt published the override as "
                    "registered_max_poll_spacing_s: 1000000.0. R0246 had "
                    "already closed the same shape for the worst-case slope "
                    "and the declared headroom and left the third parameter "
                    "unguarded."
                ),
                "closure": (
                    "every parameter across R0244-R0246 that participates in a "
                    "safety decision is enumerated in a single frozen "
                    "registry under a pinned SHA-256 that every gate verifies. "
                    "A caller may be stricter and may not be weaker; a "
                    "weakening request returns the REGISTERED value and is "
                    "recorded as a violation that fails the gate; and every "
                    "registered_* receipt field is produced by "
                    "registered_bounds(), which reads the registry and cannot "
                    "read a caller."
                ),
                "parameters_registered": len(REGISTERED_SAFETY_PARAMETERS),
                "registry_fingerprint": registry_fingerprint(),
                "registered_registry_sha256": REGISTERED_REGISTRY_SHA256,
                "class_note": SAFETY_PARAMETER_CLASS_NOTE,
                "construction_path_note": CONSTRUCTION_PATH_NOTE,
                "positive_control": (
                    "one per registered parameter, through its own "
                    "construction path, plus review-0246-01 C's sixteenth "
                    "attack in its exact construction"
                ),
            },
            "closure_2_the_coverage_denominator": {
                "defect": (
                    "review-0246-01 A: thread coverage is a ratio to a "
                    "self-declared sampling interval. A guard declaring "
                    "interval_s = 5.0 reports coverage 0.9994, passes "
                    "require_live_sampler, and buys an observation gap of "
                    "5.884e10 B - 1.99x the sealed headroom."
                ),
                "closure": (
                    "the coverage denominator is measured wall time over the "
                    "REGISTERED 0.25 s interval, and two further arms are "
                    "measured in seconds and are not ratios to anything: the "
                    "widest and the mean interval between two successful "
                    "thread samples, timestamped by the sampling thread "
                    "itself, against registered ceilings of "
                    "2.5109531834854018 s and 0.5 s."
                ),
                "positive_control": (
                    "review-0246-01 A's exact 10 h / 5.0 s guard, refused on "
                    "each of the three arms independently"
                ),
            },
            "closure_3_the_ledger": {
                "defect": (
                    "review-0246-01 E: the count is EIGHT, not seven; the "
                    "bound adjudication is prose rather than a receipt; and "
                    "TIE_AGGREGATE_ONLY_RULE's '1% of the margin' is 100x "
                    "stricter than the 1.0 the code applied."
                ),
                "closure": (
                    "the count is computed in the receipt instead of written "
                    "in prose, with the already-repaired claim counted and "
                    "separately labelled; the bound adjudication is a sealed "
                    "artifact produced by running the sealed ledger through "
                    "the sealed function; and the two criteria are reconciled "
                    "by registering 1.0 for adjudicating an ALREADY PUBLISHED "
                    "claim and the sealed 1% for ADMITTING a new consumption, "
                    "which is the question each function was always asking."
                ),
                "tie_claim_max_expected_flips_over_margin": (
                    TIE_CLAIM_MAX_EXPECTED_FLIPS_OVER_MARGIN
                ),
                "tie_use_max_expected_flips_over_margin": (
                    TIE_USE_MAX_EXPECTED_FLIPS_OVER_MARGIN
                ),
                "claims_in_the_ledger": len(TIE_AWARE_CLAIM_LEDGER),
                "rule": TIE_AGGREGATE_ONLY_RULE,
            },
            "closure_4_the_precision_fix": {
                "defect": (
                    "review-0246-01 F: result-0246 attributed the float64 "
                    "residual to float32 STORAGE, whose half-ulp is 2.98e-08 "
                    "against a measured p99 of 5.336e-07 - nine times larger. "
                    "The residual is the truth's float32 ARITHMETIC, so "
                    "recomputing the cosines from the sealed ids is a CPU "
                    "gather and not the 100M-row GPU job result-0246 priced."
                ),
                "closure": (
                    "the cosines are recomputed in float64 from the sealed "
                    "truth ids; the storage/arithmetic decomposition is "
                    "measured against a second float64 contraction order; the "
                    "flip rate is re-measured against the new reference over "
                    "the whole 500,000-row probe; and the tolerance the new "
                    "floor supports is STATED and not applied, because "
                    "TIE_TOLERANCE is consumed by published R0241 and R0243 "
                    "figures."
                ),
                "tie_tolerance": float(TIE_TOLERANCE),
                "full_probe_rows": TIE_FULL_PROBE_ROWS,
                "full_probe_decisions": TIE_FULL_PROBE_ROWS * GRAPH_K,
                "replication_rows": TIE_PRECISION_ROWS,
                "tie_precision_seed": TIE_PRECISION_SEED,
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
    path = prepare_round0247(
        release_sha=args.release_sha, queue_root=args.queue_root,
        prior_gpu_wall_s=args.prior_gpu_wall_s, only=tuple(args.only),
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
