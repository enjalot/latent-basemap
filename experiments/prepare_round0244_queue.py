#!/usr/bin/env python3
"""Prepare, but never launch, the R0244 queue.

Four CPU-shaped nodes that discharge review-0243-01 section 8's four
instrument-side prerequisites. Nothing is trained, nothing is rebuilt, and no
byte this round reads is created by it: R0238's substrate, provenance,
reachability vector and truth probe, R0240's graph arrays, R0242's per-row
vectors and R0243's symmetrised edge list all already exist and are bound at
their published digests.

It also performs the correction review-0243-01 section 7 asked for outside any
node: R0238's `strict-c400.f64.npy` is declared a first-class inherited
artifact (it appears in this queue's `expected_inputs`, so `roundreport` puts
it in the Inputs table) and its mode is set to `0444` to match its sibling
receipt.

This preparation script starts two child processes - git and the CPU smoke -
and neither carries a signalling wall bound. The node module starts none.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
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
from basemap.round0243_residual import R0242_PRIMARY_CLUSTER_SHA256
from basemap.round0244_guard import (
    FUZZY_STAGE_ANON_BUDGET_BYTES,
    GPU_HOURS_CAP,
    ROUND_ID,
    ROWS,
    WATCHDOG_NOTE,
    WATCHDOG_SAMPLE_INTERVAL_S,
)
from basemap.round0244_prereq import (
    DID_ALPHA,
    DID_DECISION_RULE,
    DID_MAPS_PER_ARM,
    DID_NULL_ARM,
    DID_PERMUTATION,
    DID_STATISTIC,
    DID_WHAT_IT_CANNOT_SETTLE,
    NEAR_DUPLICATE_CATEGORIES,
    R0238_PROVENANCE_SHA256,
    R0238_REACHABILITY_VECTOR_SHA256,
    R0242_PROBE_BUILDER_MISSING_SHA256,
    R0242_PROBE_CLUSTER_SHA256,
    R0242_PROBE_STRICT_RECALL_SHA256,
    R0243_DIRECTED_EDGES,
    R0243_EDGES_DST_SHA256,
    R0243_EDGES_HEADER_SHA256,
    R0243_EDGES_SRC_SHA256,
    R0243_EDGES_WTS_SHA256,
    R0243_FUZZY_RECEIPT_SHA256,
    R0243_TIE_AWARE_BUILDER_MISSING_SHA256,
    R0243_TOTAL_WEIGHT,
    SAMPLER_BLOCK_EDGES,
    SAMPLER_DRAWS,
    SAMPLER_EPOCHS,
    SAMPLER_MAX_ABS_Z,
    SAMPLER_MAX_ANONYMOUS_BYTES,
    SAMPLER_MIN_CHI_SQUARE_P,
    SAMPLER_NOTE,
    SAMPLER_SEED,
    TEXT_BINDING_COSINE_FLOOR,
    TEXT_NOTE,
    TEXT_SAMPLE_PAIRS,
    TEXT_SAMPLE_SEED,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0244_nodes import (
    CLUSTER_UNDER_TEST,
    DID_ACTION,
    DID_CAPABILITY,
    DID_FILE,
    SAMPLER_ACTION,
    SAMPLER_CAPABILITY,
    TEXT_ACTION,
    TEXT_CAPABILITY,
    WATCHDOG_ACTION,
    WATCHDOG_CAPABILITY,
)

ROUND_ROOT = "/data/latent-basemap/runs/round-0244"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0244-2026-08-10.md")
HANDLER_MODULE = "experiments.round0244_nodes"

R0238_SUBSTRATE_DIR = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1"
)
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

R0238_REACHABILITY_VECTOR = os.path.join(
    R0238_REACHABILITY_DIR, "strict-c400.f64.npy"
)

#: Small enough to rehash cheaply, so `roundreport` puts every one of them in
#: the Inputs table. `strict-c400.f64.npy` is here because review-0243-01
#: section 7 found the round's most load-bearing input invisible there.
SMALL_INPUTS: dict[str, str] = {
    "substrate_manifest": INHERITED_SUBSTRATE_MANIFEST,
    "truth_reference": INHERITED_TRUTH_MANIFEST,
    "reachability_reference": INHERITED_REACHABILITY_MANIFEST,
    "ladder_reference": INHERITED_LADDER_RECEIPT,
    "r0243_fuzzy_receipt": os.path.join(R0243_FUZZY_DIR, "fuzzy-graph.json"),
    "edges_header": os.path.join(
        R0243_FUZZY_DIR, "edges-k15-fuzzy-header.npz"
    ),
}
VECTOR_INPUTS: dict[str, str] = {
    "r0238_strict_reachability": R0238_REACHABILITY_VECTOR,
    "probe_query_rows": os.path.join(R0238_TRUTH_DIR, "probe-query-rows.i64.npy"),
    "truth_ids": os.path.join(R0238_TRUTH_DIR, "truth-k15-ids.i32.npy"),
    "truth_cos": os.path.join(R0238_TRUTH_DIR, "truth-k15-cos.f32.npy"),
    "r0242_probe_cluster": os.path.join(R0242_VECTORS, "probe-cluster-c400.i16.npy"),
    "r0242_probe_strict_recall": os.path.join(
        R0242_VECTORS, "probe-strict-recall.f64.npy"
    ),
    "r0242_probe_builder_missing_edges": os.path.join(
        R0242_VECTORS, "probe-builder-missing-edges.i16.npy"
    ),
    "r0243_probe_tie_aware_builder_missing_edges": os.path.join(
        R0243_RESIDUAL_VECTORS, "probe-tie-aware-builder-missing-edges.i16.npy"
    ),
    "r0242_primary_cluster": os.path.join(
        R0242_VECTORS, "primary-cluster-c400.i16.npy"
    ),
}
#: Bound at full sha256 in this script and by size in the node, but kept out of
#: `expected_inputs`: the v2.1 amendment's proportionate-verification rule, and
#: a `roundreport` that rehashes `196 GB` per invocation helps no reviewer.
BULK_INPUTS: dict[str, str] = {
    "graph_ids": INHERITED_GRAPH_IDS,
    "graph_cos": INHERITED_GRAPH_COS,
    "edges_src": os.path.join(R0243_FUZZY_DIR, "edges-k15-fuzzy-src.i32.npy"),
    "edges_dst": os.path.join(R0243_FUZZY_DIR, "edges-k15-fuzzy-dst.i32.npy"),
    "edges_wts": os.path.join(R0243_FUZZY_DIR, "edges-k15-fuzzy-wts.f32.npy"),
}
#: Declared from R0238's own sealed manifest rather than rehashed: `153.6 GB`
#: and `1.1 GB` whose digests a reviewed round already verified at creation.
MANIFEST_INPUTS = ("substrate_array", "provenance")

REGISTERED_DIGESTS = (
    ("graph_ids", REGISTERED_GRAPH_IDS_SHA256),
    ("graph_cos", REGISTERED_GRAPH_COS_SHA256),
    ("ladder_reference", REGISTERED_LADDER_RECEIPT_SHA256),
    ("r0238_strict_reachability", R0238_REACHABILITY_VECTOR_SHA256),
    ("r0242_probe_cluster", R0242_PROBE_CLUSTER_SHA256),
    ("r0242_probe_strict_recall", R0242_PROBE_STRICT_RECALL_SHA256),
    ("r0242_probe_builder_missing_edges", R0242_PROBE_BUILDER_MISSING_SHA256),
    ("r0242_primary_cluster", R0242_PRIMARY_CLUSTER_SHA256),
    (
        "r0243_probe_tie_aware_builder_missing_edges",
        R0243_TIE_AWARE_BUILDER_MISSING_SHA256,
    ),
    ("r0243_fuzzy_receipt", R0243_FUZZY_RECEIPT_SHA256),
    ("edges_header", R0243_EDGES_HEADER_SHA256),
    ("edges_src", R0243_EDGES_SRC_SHA256),
    ("edges_dst", R0243_EDGES_DST_SHA256),
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
        raise RuntimeError("R0244 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0244 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0244 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0244_contract.py", "tests/test_round0244_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0244 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0244_contract.py "
            "tests/test_round0244_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "ALL FOUR node entry paths executed end to end through run_job on "
            "tiny inputs - the defect class that killed R0216 (NameError), "
            "R0236 (arity) and R0242 attempt 1 (missing module) and that no "
            "static check can see",
            "the host watchdog POSITIVE CONTROL: a real over-budget anonymous "
            "allocation, observed by the sampling thread, the cooperative "
            "abort flag written, runner_abort_reason() reporting it, a "
            "stripe-shaped loop unwinding on it, and poll() raising",
            "the sampling-distribution fidelity check REJECTING a uniform "
            "mis-sampler on the same weights it accepts a correct draw on",
            "the DiD registration REFUSING an arm size whose smallest "
            "attainable p cannot clear its own family correction",
            "the near-duplicate classifier separating identical, near-"
            "identical and distinct text",
            "a source-level assertion that no file this round adds contains a "
            "signalling construct, including subprocess timeout - read, not "
            "delegated to a detector",
        ],
    }


def _chmod_discharge() -> dict[str, Any]:
    """review-0243-01 section 7's second half: the vector's mode."""
    before = stat.S_IMODE(os.stat(R0238_REACHABILITY_VECTOR).st_mode)
    if before != 0o444:
        os.chmod(R0238_REACHABILITY_VECTOR, 0o444)
    after = stat.S_IMODE(os.stat(R0238_REACHABILITY_VECTOR).st_mode)
    return {
        "path": R0238_REACHABILITY_VECTOR,
        "mode_before": f"0{before:o}",
        "mode_after": f"0{after:o}",
        "changed": bool(before != after),
        "why": (
            "review-0243-01 section 7: R0238's per-row reachability vector was "
            "mode 0664 while its sibling receipt was 0444. Every other sealed "
            "artifact in this program is read-only; this one was writable."
        ),
    }


def prepare_round0244(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    prior_gpu_wall_s: float = 0.0,
    only: tuple[str, ...] = (),
) -> str:
    """Build the queue. Never launches anything.

    `only` builds a correction queue carrying a subset of the nodes, with the
    dependencies of the omitted ones dropped. Every earlier attempt's GPU wall
    is carried in `prior_gpu_wall_s` so the correction is sized against the
    round's real remaining cap (review-0224: charge every attempt).
    """
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0244 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    signatures: dict[str, dict[str, Any]] = {}
    for key, path in {**SMALL_INPUTS, **VECTOR_INPUTS, **BULK_INPUTS}.items():
        if not os.path.exists(path):
            raise RuntimeError(f"R0244 inherited input absent: {key} at {path}")
        signatures[key] = expected_input_signature(path)
    for key, digest in REGISTERED_DIGESTS:
        if signatures[key].get("sha256") != digest:
            raise RuntimeError(
                f"R0244 STOP: {key} hashes to {signatures[key].get('sha256')}, "
                f"registered {digest}"
            )
    with open(INHERITED_SUBSTRATE_MANIFEST, encoding="utf-8") as handle:
        substrate_manifest = json.load(handle)
    signatures["substrate_array"] = dict(substrate_manifest["substrate"])
    signatures["provenance"] = dict(substrate_manifest["provenance"])
    if signatures["provenance"].get("sha256") != R0238_PROVENANCE_SHA256:
        raise RuntimeError("R0244 STOP: R0238 provenance digest moved")
    for key in MANIFEST_INPUTS:
        path = str(signatures[key]["canonical_path"])
        if not os.path.exists(path):
            raise RuntimeError(f"R0244 inherited input absent: {key} at {path}")
        if os.path.getsize(path) != int(signatures[key]["bytes"]):
            raise RuntimeError(f"R0244 {key} is not its declared size")

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0244 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    atomic_write_new_json(
        os.path.join(preflight, "reachability-mode-discharge.json"),
        _chmod_discharge(), immutable=True,
    )
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        *(signatures[key] for key in SMALL_INPUTS),
        *(signatures[key] for key in VECTOR_INPUTS),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    did_dir = os.path.join(artifacts, DID_CAPABILITY)
    watchdog_dir = os.path.join(artifacts, WATCHDOG_CAPABILITY)
    sampler_dir = os.path.join(artifacts, SAMPLER_CAPABILITY)
    text_dir = os.path.join(artifacts, TEXT_CAPABILITY)

    policy = {"gpu_required": True, "training_performed": False, "cpu_heavy": True}
    gpu_budget_remaining_s = float(GPU_HOURS_CAP * 3600.0 - prior_gpu_wall_s)
    inheritance = {
        key: signatures[key] for key in (
            "substrate_manifest", "truth_reference", "reachability_reference",
            "ladder_reference", "graph_ids", "graph_cos",
        )
    }

    jobs: list[dict[str, Any]] = [
        {
            "id": "did_registration", "action": DID_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": [],
            "outputs": [did_dir],
            "done_marker": os.path.join(artifacts, "did_registration.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 300.0,
            "capability": DID_CAPABILITY,
            **{key: signatures[key] for key in VECTOR_INPUTS},
            "stage_budget_s": 900.0,
            "gpu_budget_remaining_s": gpu_budget_remaining_s,
            "node_policy": dict(policy),
        },
        {
            "id": "watchdog_100000k", "action": WATCHDOG_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["did_registration"],
            "outputs": [watchdog_dir],
            "done_marker": os.path.join(artifacts, "watchdog_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 1_800.0,
            "capability": WATCHDOG_CAPABILITY,
            **inheritance,
            "anonymous_budget_bytes": FUZZY_STAGE_ANON_BUDGET_BYTES,
            "stage_budget_s": 5_400.0,
            "node_policy": dict(policy),
        },
        {
            "id": "sampler_100000k", "action": SAMPLER_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["watchdog_100000k"],
            "outputs": [sampler_dir],
            "done_marker": os.path.join(artifacts, "sampler_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 1_800.0,
            "capability": SAMPLER_CAPABILITY,
            **{
                key: signatures[key] for key in
                ("edges_header", "edges_src", "edges_dst", "edges_wts")
            },
            "draws": SAMPLER_DRAWS,
            "stage_budget_s": 5_400.0,
            "node_policy": dict(policy),
        },
        {
            "id": "text_100000k", "action": TEXT_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["sampler_100000k"],
            "outputs": [text_dir],
            "done_marker": os.path.join(artifacts, "text_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 1_800.0,
            "capability": TEXT_CAPABILITY,
            **inheritance,
            **{key: signatures[key] for key in VECTOR_INPUTS},
            "substrate_array": signatures["substrate_array"],
            "provenance": signatures["provenance"],
            "stage_budget_s": 5_400.0,
            "node_policy": dict(policy),
        },
    ]

    if only:
        keep = set(only)
        unknown = keep - {str(job["id"]) for job in jobs}
        if unknown:
            raise RuntimeError(f"R0244 has no such node(s): {sorted(unknown)}")
        jobs = [job for job in jobs if str(job["id"]) in keep]
        for job in jobs:
            job["deps"] = [dep for dep in job["deps"] if dep in keep]

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0244-100000k-pre-training-prerequisites-queue-v1",
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
            "minilm-mixed-2m-cluster-spill-map-geometry-v1",
            "runner-signal-free-node-supervision-v1",
        ],
        "capabilities_produced": [
            DID_CAPABILITY, WATCHDOG_CAPABILITY, SAMPLER_CAPABILITY,
            TEXT_CAPABILITY,
        ],
        "prior_attempt_gpu_wall_s": float(prior_gpu_wall_s),
        "gpu_budget_remaining_s": gpu_budget_remaining_s,
        "substrate_inherited": True,
        "graph_inherited": True,
        "inherited_artifacts": [
            "graph", "ladder", "provenance", "r0242_vectors",
            "r0243_edges", "r0243_vectors", "reachability", "substrate",
            "truth",
        ],
        "inherited_from": [
            "/data/latent-basemap/runs/round-0238/queue",
            "/data/latent-basemap/runs/round-0240/queue",
            "/data/latent-basemap/runs/round-0242/queue-correction-1",
            "/data/latent-basemap/runs/round-0243/queue",
        ],
        "inheritance_note": (
            "Every byte this round reads already exists and is bound at its "
            "published digest: R0243's three 10,044,413,144-byte edge arrays "
            "and their header, R0240's two 6,000,000,128-byte graph arrays, "
            "R0238's 153,600,000,128-byte substrate, its 1,100,000,192-byte "
            "provenance, its 500,000-row truth probe and - declared in an "
            "Inputs table for the first time - its 4,000,128-byte per-row "
            "reachability vector. Nothing is rebuilt and no edge list is "
            "republished."
        ),
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "total": sum(float(job["p90_wall_s"]) for job in jobs)
        },
        "scientific_contract": {
            "question": (
                "review-0243-01 accepted R0243 and gave a plain verdict: the "
                "100,000,000-row graph is sound enough to train maps on, and "
                "that is NOT the same as 'the residual is harmless'. It named "
                "four prerequisites before any ladder map is trained. This "
                "round discharges all four and claims nothing about a map."
            ),
            "rows": ROWS,
            "k": GRAPH_K,
            "clusters": REGISTERED_SELECTED_CLUSTERS,
            "prerequisite_a_host_watchdog": {
                "defect": (
                    "review-0243-01 section 6: _HostWatchdog has no thread, "
                    "run_fuzzy polls at six stage boundaries, and the stripe "
                    "loop polls the abort flag but never the memory guard. "
                    "During the ~35 GiB stage the conjunctive guard was NOT "
                    "EVALUATED AT ALL, and the published 12,058,918,912 B is "
                    "the value at the preceding boundary."
                ),
                "fix": WATCHDOG_NOTE,
                "sample_interval_s": WATCHDOG_SAMPLE_INTERVAL_S,
                "anonymous_budget_bytes": FUZZY_STAGE_ANON_BUDGET_BYTES,
                "positive_control_is_required_to_fire": True,
                "re_measurement": (
                    "the node re-runs R0243's descending sort, distance "
                    "transform and fuzzy symmetrisation on the same bound "
                    "bytes under the threaded instrument and publishes the "
                    "stage's true anonymous peak against the receipt's "
                    "12,058,918,912 B. The re-run must reproduce R0243's "
                    "2,511,103,254 directed edges and both UMAP moments "
                    "exactly, or the peak is not R0243's peak and the node "
                    "fails. The edge list is NOT republished."
                ),
            },
            "prerequisite_b_displacement_did": {
                "statistic": DID_STATISTIC,
                "null_arm": DID_NULL_ARM,
                "permutation_design": DID_PERMUTATION,
                "alpha_family_wise": DID_ALPHA,
                "maps_per_arm": DID_MAPS_PER_ARM,
                "decision_rule": DID_DECISION_RULE,
                "what_this_round_cannot_settle": DID_WHAT_IT_CANNOT_SETTLE,
                "registered_before_any_data": True,
            },
            "prerequisite_c_cluster_168_text": {
                "note": TEXT_NOTE,
                "cluster": CLUSTER_UNDER_TEST,
                "sample_pairs": TEXT_SAMPLE_PAIRS,
                "sample_seed": TEXT_SAMPLE_SEED,
                "text_binding_cosine_floor": TEXT_BINDING_COSINE_FLOOR,
                "categories": [name for name, _t in NEAR_DUPLICATE_CATEGORIES],
                "binding_is_verified_not_assumed": (
                    "each chunk is re-embedded with the substrate's own "
                    "all-MiniLM-L6-v2 and must reproduce its substrate row to "
                    "cosine >= 0.999, on the CPU, so the round holds the GPU "
                    "lease and never uses the card."
                ),
            },
            "prerequisite_d_edge_sampler": {
                "note": SAMPLER_NOTE,
                "directed_edges": R0243_DIRECTED_EDGES,
                "total_weight": R0243_TOTAL_WEIGHT,
                "block_edges": SAMPLER_BLOCK_EDGES,
                "draws": SAMPLER_DRAWS,
                "seed": SAMPLER_SEED,
                "epochs": SAMPLER_EPOCHS,
                "max_abs_z": SAMPLER_MAX_ABS_Z,
                "min_chi_square_p": SAMPLER_MIN_CHI_SQUARE_P,
                "max_anonymous_bytes": SAMPLER_MAX_ANONYMOUS_BYTES,
                "mis_sampler_control_must_be_rejected": True,
            },
            "reachability_vector_discharge": (
                "review-0243-01 section 7: R0238's strict-c400.f64.npy is "
                "declared in expected_inputs so it appears in the Inputs "
                "table, its mode is corrected from 0664 to 0444, and the DiD "
                "node CONSUMES it - recomputing the builder-missing vector "
                "through the imported loss_decomposition and requiring bitwise "
                "agreement with R0242's sealed bytes. A declaration nothing "
                "reads is decoration."
            ),
            "truth_probe_rows": TRUTH_PROBE_ROWS,
            "truth_probe_seed": TRUTH_PROBE_SEED,
            "gpu_hours_cap": GPU_HOURS_CAP,
            "gpu_hours_cap_note": (
                "2.0 GPU-h. Every node is CPU-shaped: the round holds the GPU "
                "lease so no second workload can touch the card, and creates "
                "no CUDA context at all. The dominant cost is the "
                "symmetrisation re-run, which R0243 measured at 0.093 GPU-h."
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
    path = prepare_round0244(
        release_sha=args.release_sha, queue_root=args.queue_root,
        prior_gpu_wall_s=args.prior_gpu_wall_s, only=tuple(args.only),
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
