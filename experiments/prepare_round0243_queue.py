#!/usr/bin/env python3
"""Prepare, but never launch, the R0243 queue.

Two nodes:

* `residual_100000k` (Part A) settles the locality question on the TIE-AWARE
  scale. Every per-row vector it needs was sealed by R0242 and is bound here at
  its full sha256 signature; nothing is rebuilt and nothing is re-measured that
  a sealed artifact already carries. It costs the card one re-realisation of
  the registered `c = 400, s = 8, seed 226` partition, which exists only to
  discharge review-0242-01/F9.3 by SEALING a reproduced reachability vector,
  and two instrumented SORTED gathers off the substrate.
* `fuzzy_100000k` (Part B) runs only if Part A's sealed verdict permits it. It
  symmetrises with UMAP's own law, reports symmetrised degree ONCE, and runs
  the R0215 degree-zero tripwire AFTER canonicalization - which is where the v1
  defect arose (R0034: `2,779,481` rows) and where it has never run at this
  rung.

The node module starts no child process. This preparation script starts two -
git and the CPU smoke - and neither carries a signalling wall bound.
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
    GRAPH_K,
    SPILL,
    TRUTH_METHOD,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
)
from basemap.round0240_rung5 import (
    INHERITED_REACHABILITY_MANIFEST,
    INHERITED_SUBSTRATE_MANIFEST,
    INHERITED_TRUTH_MANIFEST,
    REGISTERED_REACHABILITY_CEILING_C400,
)
from basemap.round0241_qualify import (
    INHERITED_GRAPH_COS,
    INHERITED_GRAPH_IDS,
    INHERITED_LADDER_RECEIPT,
    MAX_ZERO_DEGREE_ROWS,
    REGISTERED_GRAPH_COS_SHA256,
    REGISTERED_GRAPH_IDS_SHA256,
    REGISTERED_LADDER_RECEIPT_SHA256,
    REGISTERED_SELECTED_CELL,
    REGISTERED_SELECTED_CLUSTERS,
)
from basemap.round0243_residual import (
    CANONICALIZATION_NOTE,
    CANONICAL_CAPABILITY,
    CLUSTERS,
    CONCENTRATION_TOP_M,
    DIMENSION,
    EXPOSURE_GUARD_NOTE,
    FUZZY_CAPABILITY,
    FUZZY_DEADLINE_S,
    FUZZY_FILE,
    FUZZY_STAGE_BUDGET_S,
    GPU_HOURS_CAP,
    HALT_CELL_TIE_AWARE_BUILDER_RATE,
    HALT_GLOBAL_TIE_AWARE_BUILDER_RATE,
    HALT_P_VALUE,
    HALT_RULE_NOTE,
    HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE,
    HALT_SINGLE_CLUSTER_SHARE,
    HALT_TOP_M_SHARE,
    PARTITION_SEED,
    PERMUTATIONS,
    PERMUTATION_SEED,
    R0242_LOCALITY_SHA256,
    R0242_PRIMARY_CLUSTER_SHA256,
    R0242_TIE_AWARE_VECTOR_SHA256,
    RESIDUAL_CAPABILITY,
    RESIDUAL_DEADLINE_S,
    RESIDUAL_FILE,
    RESIDUAL_STAGE_BUDGET_S,
    ROUND_ID,
    ROWS,
    SAFETY_NOTE,
    SCOPE_NOTE,
    SORTED_GATHER_NOTE,
    SYMMETRISED_DEGREE_ONCE_NOTE,
    TIE_AWARE_NOTE,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0243_nodes import (
    FUZZY_ACTION,
    RESIDUAL_ACTION,
    SORTED_GATHER_ANCHORS,
    SORTED_GATHER_SEED,
)

ROUND_ROOT = "/data/latent-basemap/runs/round-0243"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0243-2026-08-10.md")
HANDLER_MODULE = "experiments.round0243_nodes"

R0242_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0242/queue-correction-1/artifacts/"
    "minilm-mixed-100000k-k15-loss-locality-v1"
)
R0242_LOCALITY = os.path.join(R0242_ARTIFACTS, "loss-locality.json")
R0242_VECTORS = os.path.join(R0242_ARTIFACTS, "vectors")

SMALL_INPUTS: dict[str, str] = {
    "substrate_manifest": INHERITED_SUBSTRATE_MANIFEST,
    "truth_reference": INHERITED_TRUTH_MANIFEST,
    "reachability_reference": INHERITED_REACHABILITY_MANIFEST,
    "ladder_reference": INHERITED_LADDER_RECEIPT,
    "r0242_locality": R0242_LOCALITY,
}
VECTOR_INPUTS: dict[str, str] = {
    "r0242_probe_cluster": os.path.join(R0242_VECTORS, "probe-cluster-c400.i16.npy"),
    "r0242_probe_strict_recall": os.path.join(
        R0242_VECTORS, "probe-strict-recall.f64.npy"
    ),
    "r0242_probe_tie_aware_recall": os.path.join(
        R0242_VECTORS, "probe-tie-aware-recall.f64.npy"
    ),
    "r0242_probe_missing_edges": os.path.join(
        R0242_VECTORS, "probe-missing-edges.i16.npy"
    ),
    "r0242_probe_builder_missing_edges": os.path.join(
        R0242_VECTORS, "probe-builder-missing-edges.i16.npy"
    ),
    "r0242_probe_in_degree": os.path.join(R0242_VECTORS, "probe-in-degree.i32.npy"),
    "r0242_primary_cluster": os.path.join(
        R0242_VECTORS, "primary-cluster-c400.i16.npy"
    ),
}
BULK_INPUTS: dict[str, str] = {
    "graph_ids": INHERITED_GRAPH_IDS,
    "graph_cos": INHERITED_GRAPH_COS,
}

REGISTERED_DIGESTS = (
    ("graph_ids", REGISTERED_GRAPH_IDS_SHA256),
    ("graph_cos", REGISTERED_GRAPH_COS_SHA256),
    ("ladder_reference", REGISTERED_LADDER_RECEIPT_SHA256),
    ("r0242_locality", R0242_LOCALITY_SHA256),
    ("r0242_probe_tie_aware_recall", R0242_TIE_AWARE_VECTOR_SHA256),
    ("r0242_primary_cluster", R0242_PRIMARY_CLUSTER_SHA256),
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
        raise RuntimeError("R0243 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0243 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0243 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0243_contract.py", "tests/test_round0243_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0243 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0243_contract.py "
            "tests/test_round0243_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "BOTH node entry paths executed end to end through run_job with "
            "tiny inputs - the defect class that killed R0216 (NameError), "
            "R0236 (arity) and R0242 attempt 1 (missing module) and that no "
            "static check can see",
            "the post-canonicalization degree-zero tripwire with the v1 defect "
            "PLANTED, proving the guard fires",
            "the re-expressed exposure guard EXCLUDING cells at a realised "
            "size distribution where R0242's absolute 1% guard excludes none - "
            "the vacuity positive control",
            "the magnitude arms H1 and H2 each firing on a planted defect and "
            "each declining to fire on a clean input",
            "the strict reproduction gate H0 detecting a one-unit drift",
            "the tie-aware decomposition recovering a planted tie-forgiveness "
            "rate through the same imported loss_decomposition",
            "a source-level assertion that no file this round adds contains a "
            "signalling construct, including subprocess timeout - read, not "
            "delegated to a detector",
        ],
    }


def prepare_round0243(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    prior_gpu_wall_s: float = 0.0,
) -> str:
    """Build the queue. Never launches anything."""
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0243 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    signatures: dict[str, dict[str, Any]] = {}
    for key, path in {**SMALL_INPUTS, **VECTOR_INPUTS, **BULK_INPUTS}.items():
        if not os.path.exists(path):
            raise RuntimeError(f"R0243 inherited input absent: {key} at {path}")
        signatures[key] = expected_input_signature(path)
    for key, digest in REGISTERED_DIGESTS:
        if signatures[key].get("sha256") != digest:
            raise RuntimeError(
                f"R0243 STOP: {key} hashes to {signatures[key].get('sha256')}, "
                f"registered {digest}"
            )

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0243 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
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
    residual_dir = os.path.join(artifacts, RESIDUAL_CAPABILITY)
    fuzzy_dir = os.path.join(artifacts, FUZZY_CAPABILITY)

    policy = {"gpu_required": True, "training_performed": False, "cpu_heavy": False}
    gpu_budget_remaining_s = float(GPU_HOURS_CAP * 3600.0 - prior_gpu_wall_s)
    shared = {
        key: signatures[key] for key in (
            "substrate_manifest", "truth_reference", "reachability_reference",
            "ladder_reference", "graph_ids", "graph_cos",
        )
    }

    jobs: list[dict[str, Any]] = [
        {
            "id": "residual_100000k", "action": RESIDUAL_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": [],
            "outputs": [residual_dir],
            "done_marker": os.path.join(artifacts, "residual_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 3_000.0,
            "capability": RESIDUAL_CAPABILITY,
            **shared,
            **{key: signatures[key] for key in VECTOR_INPUTS},
            "r0242_locality": signatures["r0242_locality"],
            "probe_rows": TRUTH_PROBE_ROWS,
            "probe_seed": TRUTH_PROBE_SEED,
            "clusters": CLUSTERS,
            "partition_seed": PARTITION_SEED,
            "permutations": PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "sorted_gather_anchors": list(SORTED_GATHER_ANCHORS),
            "sorted_gather_seed": SORTED_GATHER_SEED,
            "stage_budget_s": RESIDUAL_STAGE_BUDGET_S,
            "gpu_budget_remaining_s": gpu_budget_remaining_s,
            "node_policy": dict(policy),
        },
        {
            "id": "fuzzy_100000k", "action": FUZZY_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["residual_100000k"],
            "outputs": [fuzzy_dir],
            "done_marker": os.path.join(artifacts, "fuzzy_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 21_600.0,
            "capability": FUZZY_CAPABILITY,
            **shared,
            "residual_reference": os.path.join(residual_dir, RESIDUAL_FILE),
            "stage_budget_s": FUZZY_STAGE_BUDGET_S,
            "node_policy": dict(policy),
        },
    ]

    queue = _base_manifest(
        round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
        queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu", gpu=True,
    )
    queue.update({
        "schema": "round0243-100000k-tie-aware-locality-and-fuzzy-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-graph",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-100000k-nested-substrate-and-reserves-v1",
            "minilm-mixed-100000k-uniform-probe-k15-truth-v1",
            "minilm-mixed-100000k-cluster-spill-c400-reachability-v1",
            "minilm-mixed-100000k-cluster-spill-k15-neighbour-graph-v1",
            "minilm-mixed-100000k-k15-degree-zero-tripwire-v1",
            "minilm-mixed-100000k-k15-neighbour-graph-qualification-v1",
            "minilm-mixed-100000k-cluster-spill-build-ladder-v1",
            "minilm-mixed-100000k-k15-loss-locality-v1",
            "runner-signal-free-node-supervision-v1",
        ],
        "capabilities_produced": [
            RESIDUAL_CAPABILITY, FUZZY_CAPABILITY, CANONICAL_CAPABILITY,
        ],
        "prior_attempt_gpu_wall_s": float(prior_gpu_wall_s),
        "gpu_budget_remaining_s": gpu_budget_remaining_s,
        "substrate_inherited": True,
        "graph_inherited": True,
        "inherited_artifacts": [
            "graph", "ladder", "r0242_locality", "r0242_vectors",
            "reachability", "substrate", "truth",
        ],
        "inherited_from": [
            "/data/latent-basemap/runs/round-0238/queue",
            "/data/latent-basemap/runs/round-0240/queue",
            "/data/latent-basemap/runs/round-0241/queue",
            "/data/latent-basemap/runs/round-0242/queue-correction-1",
        ],
        "inheritance_note": (
            "Every byte this round reads already exists and is bound at its "
            "full sha256 signature: R0240's two 6,000,000,128-byte graph "
            "arrays and build-ladder receipt, R0238's 153.6 GB substrate, "
            "500,000-row uniform truth probe and c = 400 reachability vector, "
            "and R0242's sealed loss-locality receipt plus its seven per-row "
            "vectors - including probe-tie-aware-recall.f64.npy, the vector "
            "R0242 sealed and never joined. Nothing is rebuilt."
        ),
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "total": sum(float(job["p90_wall_s"]) for job in jobs)
        },
        "scientific_contract": {
            "question": (
                "R0242 measured the 100,000,000-row k15 loss exclusively in "
                "STRICT missing edges and found it spatially concentrated. "
                "review-0242-01 confirmed the concentration against two "
                "independent nulls and then corrected its magnitude: 97.74% of "
                "the worst cell's loss is TIE-FORGIVEN, i.e. the builder "
                "returned a substitute neighbour within the tie threshold of "
                "the true k-th. R0242 sealed the tie-aware vector and never "
                "joined it. This round runs the same decomposition and the "
                "same cluster tests on that vector, beside the strict "
                "figures, and asks the question that is actually open: is the "
                "RESIDUAL - the loss that is not tie-forgiven - large enough "
                "to matter to a map? Only if the registered magnitude rule "
                "says no does the round go on to symmetrise the graph and run "
                "the first post-canonicalization degree-zero tripwire at this "
                "rung."
            ),
            "rows": ROWS,
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "spill": SPILL,
            "clusters": REGISTERED_SELECTED_CLUSTERS,
            "cell": REGISTERED_SELECTED_CELL,
            "tie_aware_note": TIE_AWARE_NOTE,
            "halt_rule": HALT_RULE_NOTE,
            "halt_global_tie_aware_builder_rate": (
                HALT_GLOBAL_TIE_AWARE_BUILDER_RATE
            ),
            "halt_cell_tie_aware_builder_rate": HALT_CELL_TIE_AWARE_BUILDER_RATE,
            "halt_single_cluster_share": HALT_SINGLE_CLUSTER_SHARE,
            "halt_single_cluster_exposure_multiple_of_uniform": (
                HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE
            ),
            "shape_statistics_are_reported_and_do_not_gate": {
                "top_m": CONCENTRATION_TOP_M,
                "top_m_share_threshold_r0242_used": HALT_TOP_M_SHARE,
                "chi_square_p_threshold_r0242_used": HALT_P_VALUE,
                "why": (
                    "R0242's shape halt existed so a spatial-concentration "
                    "finding could not be buried under a product step. "
                    "review-0242-01 released that finding, so there is nothing "
                    "left to bury. The shape statistics are computed on both "
                    "scales with 10,000-permutation nulls and published in "
                    "full; the magnitude arms are the ones that halt."
                ),
            },
            "exposure_guard_note": EXPOSURE_GUARD_NOTE,
            "permutations": PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "partition_seed": PARTITION_SEED,
            "truth_method": TRUTH_METHOD,
            "truth_probe_rows": TRUTH_PROBE_ROWS,
            "truth_probe_seed": TRUTH_PROBE_SEED,
            "probe_is_the_registered_r0238_draw": True,
            "reproduction_gate": (
                "H0: the strict decomposition and the strict per-cluster "
                "dispersion statistics are recomputed here from R0242's sealed "
                "per-row vectors through the same imported loss_decomposition "
                "and _dispersion, and must equal R0242's sealed values "
                "EXACTLY. A tie-aware re-run of a loss vector that is not "
                "R0242's loss vector answers nothing, so a mismatch halts "
                "Part B."
            ),
            "sorted_gather_note": SORTED_GATHER_NOTE,
            "sorted_gather_anchors": list(SORTED_GATHER_ANCHORS),
            "sorted_gather_seed": SORTED_GATHER_SEED,
            "partition_discharge_note": (
                "review-0242-01/F9.3: R0242's reproduced reachability vector "
                "was never sealed, so the one line in that round a reviewer "
                "could not check was its strongest partition claim. This round "
                "re-realises the partition once and SEALS the reproduced "
                "reachability vector and the primary labels it came from. It "
                "gates nothing: the tie-aware analysis is stratified by "
                "R0242's SEALED probe labels and Part B consumes only the two "
                "graph arrays, so a disagreement is published as a finding."
            ),
            "symmetrised_degree_once_note": SYMMETRISED_DEGREE_ONCE_NOTE,
            "canonicalization_note": CANONICALIZATION_NOTE,
            "builder_cosine_substitution_note": (
                "Part B takes the fuzzy weights' distances from the builder's "
                "SEALED cosines (474f14d2...), as R0242 registered and as "
                "R0242's own adversarial agreement check validated to "
                "3.5762786865234375e-07 over uniform, zero-in-degree and "
                "top-1%-in-degree strata. Part B therefore performs NO "
                "substrate gather at all, and the gather term is priced "
                "separately in Part A - as a SORTED gather, whose physical "
                "read is bounded above by the substrate itself."
            ),
            "edge_output_format_note": (
                "the symmetrised edge list is published as three streamed, "
                "memmappable .npy arrays plus a small header .npz, not as one "
                "bulk .npz: zipfile cannot stream a member, so the archive "
                "path materialises about 20 GB of ANONYMOUS memory per 10 GB "
                "member for no benefit and yields an archive no 100M trainer "
                "can memmap."
            ),
            "inherited_graph_ids_sha256": REGISTERED_GRAPH_IDS_SHA256,
            "inherited_graph_cos_sha256": REGISTERED_GRAPH_COS_SHA256,
            "inherited_ladder_receipt_sha256": REGISTERED_LADDER_RECEIPT_SHA256,
            "inherited_r0242_locality_sha256": R0242_LOCALITY_SHA256,
            "inherited_r0242_tie_aware_vector_sha256": (
                R0242_TIE_AWARE_VECTOR_SHA256
            ),
            "inherited_reachability_ceiling_c400": (
                REGISTERED_REACHABILITY_CEILING_C400
            ),
            "floors": {
                "post_canonical_zero_degree_rows": MAX_ZERO_DEGREE_ROWS,
                "fuzzy_weight_min_exclusive": 0.0,
                "fuzzy_weight_max": 1.0,
            },
            "gpu_hours_cap": GPU_HOURS_CAP,
            "gpu_hours_cap_note": (
                "12.0 GPU-h. Part A is priced at well under 0.5 GPU-h - it "
                "joins vectors that already exist, plus two instrumented "
                "sorted gathers and one partition re-realisation. Part B's "
                "symmetrisation is the ~4.5 h product step R0241 deferred and "
                "R0242 halted. Every attempt's GPU time is charged, including "
                "one that produces nothing."
            ),
            "fuzzy_deadline_s": FUZZY_DEADLINE_S,
            "residual_deadline_s": RESIDUAL_DEADLINE_S,
            "fuzzy_receipt_file": FUZZY_FILE,
            "residual_receipt_file": RESIDUAL_FILE,
            "safety": SAFETY_NOTE,
            "scope": SCOPE_NOTE,
            "no_training": True,
            "no_gate_registered": True,
            "no_adoption_claimed": True,
            "no_map_quality_claimed": True,
            "no_assembly": True,
            "no_build": True,
            "no_atlas_claim": True,
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
    path = prepare_round0243(
        release_sha=args.release_sha, queue_root=args.queue_root,
        prior_gpu_wall_s=args.prior_gpu_wall_s,
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
