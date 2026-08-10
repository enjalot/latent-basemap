#!/usr/bin/env python3
"""Prepare, but never launch, the R0242 queue — locality first, product second.

Two nodes, in this order and for this reason:

* `locality_100000k` (Part A) joins evidence that already exists on disk:
  R0241's per-row in-degree over all `100,000,000` rows, R0238's sealed
  `500,000`-row uniform probe with exact `k15` truth, and R0238's sealed
  per-row `c = 400` partition reachability. It asks whether recall loss is
  concentrated in antihubs or hubs, and whether it is concentrated in
  particular clusters of the partition - the spatial question the probe is
  structurally blind to. It costs the card one re-realisation of the
  registered `c = 400, s = 8, seed 226` partition and nothing else.
* `fuzzy_100000k` (Part B) runs ONLY if Part A's sealed verdict permits it,
  and the node re-reads that verdict and refuses itself rather than trusting
  job ordering. It symmetrises with UMAP's own law, reports symmetrised degree
  ONCE, and runs the R0215 degree-zero tripwire AFTER canonicalization - which
  is where the v1 defect arose and where no round in this program has ever run
  it at this rung.

Nothing is assembled and no neighbour graph is built. The `100,000,000`-row
graph arrays and the substrate are bound at their full sha256 signatures and
re-earned by nothing.

The node module starts no child process at all. This preparation script starts
two - git and the CPU smoke - and neither carries a signalling wall bound of any
kind.
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
    PROBE_GATHER_BLOCK,
    REGISTERED_GRAPH_COS_SHA256,
    REGISTERED_GRAPH_IDS_SHA256,
    REGISTERED_LADDER_RECEIPT_SHA256,
    REGISTERED_SELECTED_CELL,
    REGISTERED_SELECTED_CLUSTERS,
)
from basemap.round0242_locality import (
    CANONICALIZATION_NOTE,
    CANONICAL_CAPABILITY,
    CLUSTERS,
    CONCENTRATION_TOP_M,
    DIMENSION,
    FUZZY_CAPABILITY,
    FUZZY_DEADLINE_S,
    FUZZY_FILE,
    FUZZY_STAGE_BUDGET_S,
    GATHER_TERM_NOTE,
    GPU_HOURS_CAP,
    HALT_P_VALUE,
    HALT_RULE_NOTE,
    HALT_SINGLE_CLUSTER_EXPOSURE,
    HALT_SINGLE_CLUSTER_SHARE,
    HALT_TOP_M_SHARE,
    LOCALITY_CAPABILITY,
    LOCALITY_DEADLINE_S,
    LOCALITY_FILE,
    LOCALITY_NOTE,
    LOCALITY_STAGE_BUDGET_S,
    PARTITION_REALISATION_NOTE,
    PARTITION_SEED,
    PERMUTATIONS,
    PERMUTATION_SEED,
    ROUND_ID,
    ROWS,
    SAFETY_NOTE,
    SCOPE_NOTE,
    SYMMETRISED_DEGREE_ONCE_NOTE,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.round0242_nodes import FUZZY_ACTION, LOCALITY_ACTION

ROUND_ROOT = "/data/latent-basemap/runs/round-0242"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0242-2026-08-10.md")
HANDLER_MODULE = "experiments.round0242_nodes"

R0241_TRIPWIRE = (
    "/data/latent-basemap/runs/round-0241/queue/artifacts/"
    "minilm-mixed-100000k-k15-degree-zero-tripwire-v1/degree-zero-tripwire.json"
)
R0241_QUALIFICATION = (
    "/data/latent-basemap/runs/round-0241/queue/artifacts/"
    "minilm-mixed-100000k-k15-neighbour-graph-qualification-v1/"
    "graph-qualification.json"
)

SMALL_INPUTS: dict[str, str] = {
    "substrate_manifest": INHERITED_SUBSTRATE_MANIFEST,
    "truth_reference": INHERITED_TRUTH_MANIFEST,
    "reachability_reference": INHERITED_REACHABILITY_MANIFEST,
    "ladder_reference": INHERITED_LADDER_RECEIPT,
    "r0241_tripwire": R0241_TRIPWIRE,
    "r0241_qualification": R0241_QUALIFICATION,
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
        raise RuntimeError("R0242 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0242 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError(
            f"R0242 release checkout is at {observed}, not {release_sha}"
        )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [os.path.join(RELEASE_ROOT, ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_round0242_contract.py", "tests/test_round0242_cpu_smoke.py"],
        cwd=RELEASE_ROOT, env=environment, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0242 release CPU smoke failed:\n{completed.stdout[-4000:]}"
        )
    return {
        "release_sha": release_sha,
        "cuda_visible_devices": "",
        "command": (
            "pytest -q tests/test_round0242_contract.py "
            "tests/test_round0242_cpu_smoke.py"
        ),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "reaches": [
            "the post-canonicalization degree-zero tripwire with the v1 defect "
            "PLANTED - a row whose every fuzzy weight underflowed to zero, and "
            "a row left holding only a self-loop - proving the guard fires",
            "the canonicalization folding mirrored pairs, dropping self-loops "
            "and counting out-of-range entries rather than crashing",
            "the cluster locality test on a PLANTED concentration (it fires at "
            "the permutation floor) and on uniform loss (it does not fire)",
            "the registered halt rule refusing to halt on diffuse structure, "
            "halting on concentration, and refusing to be halted by the "
            "partition-limited population",
            "the loss decomposition splitting partition-forced from "
            "builder-inside-the-partition missing edges, per EDGE not per row",
            "the two reproduction gates detecting a one-unit drift against "
            "R0241's sealed recall and in-degree distributions",
            "the corrected StageGuard stamping its wall at stage completion "
            "and measuring deadline_reached rather than asserting it",
            "a source-level assertion that no file this round adds contains a "
            "signalling construct, including subprocess timeout - read, not "
            "delegated to a detector",
        ],
    }


def prepare_round0242(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    prior_gpu_wall_s: float = 0.0,
) -> str:
    """Build the queue. Never launches anything."""
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0242 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    signatures: dict[str, dict[str, Any]] = {}
    for key, path in {**SMALL_INPUTS, **BULK_INPUTS}.items():
        if not os.path.exists(path):
            raise RuntimeError(f"R0242 inherited input absent: {key} at {path}")
        signatures[key] = expected_input_signature(path)
    for key, digest in (
        ("graph_ids", REGISTERED_GRAPH_IDS_SHA256),
        ("graph_cos", REGISTERED_GRAPH_COS_SHA256),
        ("ladder_reference", REGISTERED_LADDER_RECEIPT_SHA256),
    ):
        if signatures[key].get("sha256") != digest:
            raise RuntimeError(
                f"R0242 STOP: {key} hashes to {signatures[key].get('sha256')}, "
                f"registered {digest}"
            )

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0242 queue")
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
    locality_dir = os.path.join(artifacts, LOCALITY_CAPABILITY)
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
            "id": "locality_100000k", "action": LOCALITY_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": [],
            "outputs": [locality_dir],
            "done_marker": os.path.join(artifacts, "locality_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 2_400.0,
            "capability": LOCALITY_CAPABILITY,
            **shared,
            "r0241_tripwire": signatures["r0241_tripwire"],
            "r0241_qualification": signatures["r0241_qualification"],
            "probe_rows": TRUTH_PROBE_ROWS,
            "probe_seed": TRUTH_PROBE_SEED,
            "clusters": CLUSTERS,
            "partition_seed": PARTITION_SEED,
            "permutations": PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "stage_budget_s": LOCALITY_STAGE_BUDGET_S,
            "gpu_budget_remaining_s": gpu_budget_remaining_s,
            "node_policy": dict(policy),
        },
        {
            "id": "fuzzy_100000k", "action": FUZZY_ACTION,
            "handler_module": HANDLER_MODULE, "handler_callable": "run_job",
            "deps": ["locality_100000k"],
            "outputs": [fuzzy_dir],
            "done_marker": os.path.join(artifacts, "fuzzy_100000k.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 21_600.0,
            "capability": FUZZY_CAPABILITY,
            **shared,
            "locality_reference": os.path.join(locality_dir, LOCALITY_FILE),
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
        "schema": "round0242-100000k-loss-locality-and-fuzzy-queue-v1",
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
            "runner-signal-free-node-supervision-v1",
        ],
        "capabilities_produced": [
            LOCALITY_CAPABILITY, FUZZY_CAPABILITY, CANONICAL_CAPABILITY,
        ],
        "prior_attempt_gpu_wall_s": float(prior_gpu_wall_s),
        "gpu_budget_remaining_s": gpu_budget_remaining_s,
        "substrate_inherited": True,
        "graph_inherited": True,
        "inherited_artifacts": [
            "graph", "ladder", "r0241_qualification", "r0241_tripwire",
            "reachability", "substrate", "truth",
        ],
        "inherited_from": [
            "/data/latent-basemap/runs/round-0238/queue",
            "/data/latent-basemap/runs/round-0240/queue",
            "/data/latent-basemap/runs/round-0241/queue",
        ],
        "inheritance_note": (
            "Every byte this round reads already exists and is bound at its "
            "full sha256 signature: R0240's two 6,000,000,128-byte graph "
            "arrays and build-ladder receipt, R0238's 153.6 GB substrate, "
            "500,000-row uniform truth probe and c = 400 reachability vector, "
            "and R0241's two sealed qualification artifacts. Nothing is "
            "rebuilt. The only new computation on the card is one "
            "re-realisation of the registered c = 400, s = 8, seed 226 "
            "partition, which no artifact seals per row and which is validated "
            "against R0238's sealed reachability vector before it is used."
        ),
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "total": sum(float(job["p90_wall_s"]) for job in jobs)
        },
        "scientific_contract": {
            "question": (
                "is the measured k15 neighbour-graph loss at 100,000,000 rows "
                "SPATIALLY CONCENTRATED - in low-in-degree rows, in hubs, or "
                "in particular clusters of the c = 400 partition - or is it "
                "spread uniformly? R0228 is the precedent: a uniform average "
                "went null over exactly the population where a structured "
                "effect ran +3.94 sd, so a uniform probe's silence is not "
                "evidence of the absence of structure. Only if the answer is "
                "'not concentrated' does this round go on to symmetrise the "
                "graph and run the first post-canonicalization degree-zero "
                "tripwire at this rung."
            ),
            "rows": ROWS,
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "spill": SPILL,
            "clusters": REGISTERED_SELECTED_CLUSTERS,
            "cell": REGISTERED_SELECTED_CELL,
            "part_a_note": LOCALITY_NOTE,
            "part_a_costs_nothing_note": (
                "Part A runs FIRST and its inputs already exist. R0240 spent "
                "five GPU-hours on a product step and never reached its "
                "science check; that ordering is not repeated."
            ),
            "halt_rule": HALT_RULE_NOTE,
            "halt_p_value": HALT_P_VALUE,
            "halt_top_m_share": HALT_TOP_M_SHARE,
            "halt_top_m": CONCENTRATION_TOP_M,
            "halt_single_cluster_share": HALT_SINGLE_CLUSTER_SHARE,
            "halt_single_cluster_exposure": HALT_SINGLE_CLUSTER_EXPOSURE,
            "permutations": PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "partition_seed": PARTITION_SEED,
            "partition_realisation_note": PARTITION_REALISATION_NOTE,
            "truth_method": TRUTH_METHOD,
            "truth_probe_rows": TRUTH_PROBE_ROWS,
            "truth_probe_seed": TRUTH_PROBE_SEED,
            "probe_is_the_registered_r0238_draw": True,
            "probe_gather_block_rows": PROBE_GATHER_BLOCK,
            "reproduction_gates": (
                "Part A recomputes R0241's recall through the same reviewed "
                "_score_probe / strict_containment_rows / tie_aware_rows and "
                "STOPS unless strict mean, tie-aware mean, "
                "rows_carrying_any_loss, missing_true_edges and "
                "tie_aware_rows_at_zero all reproduce exactly; it accumulates "
                "its own per-row in-degree ARRAY - which no reviewed function "
                "returns - and STOPS unless every aggregate reproduces "
                "R0241's sealed distribution exactly. A locality test on a "
                "loss vector that is not the published loss vector would be "
                "worthless."
            ),
            "symmetrised_degree_once_note": SYMMETRISED_DEGREE_ONCE_NOTE,
            "canonicalization_note": CANONICALIZATION_NOTE,
            "gather_term_note": GATHER_TERM_NOTE,
            "builder_cosine_substitution_note": (
                "Part B takes the fuzzy weights' distances from the builder's "
                "SEALED cosines (474f14d2...) rather than re-gathering the "
                "153.6 GB substrate for all 100,000,000 anchors. That gather "
                "is the ~5 GPU-h term that put R0240 1.675 h over its cap and "
                "which review-0240-01 could only price at 4-9 h +/- 2x, and it "
                "would reproduce numbers that already exist: R0241 measured "
                "the two arithmetics agreeing to 4.172325134277344e-07 on "
                "500,000 rows. This is a registered design decision, not a "
                "silent substitution, and it is paid for with a STRONGER "
                "check than the one it replaces: Part A re-measures the "
                "agreement adversarially on 100,000 rows drawn where a "
                "builder accumulator error would hide - uniform, "
                "zero-in-degree, and the top 1% by in-degree - rather than on "
                "a uniform draw alone. No recall claim anywhere in this "
                "program has ever rested on the builder's cosines and none "
                "does here; the recall in Part A is the independent recompute."
            ),
            "edge_output_format_note": (
                "the symmetrised edge list is published as three streamed, "
                "memmappable .npy arrays plus a small header .npz, not as one "
                "bulk .npz. zipfile cannot stream a member: the archive path "
                "materialises about 20 GB of ANONYMOUS memory per 10 GB member "
                "inside an io.BytesIO for no benefit, and produces an archive "
                "no 100M trainer can memmap. Registered here in advance."
            ),
            "inherited_graph_ids_sha256": REGISTERED_GRAPH_IDS_SHA256,
            "inherited_graph_cos_sha256": REGISTERED_GRAPH_COS_SHA256,
            "inherited_ladder_receipt_sha256": REGISTERED_LADDER_RECEIPT_SHA256,
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
                "12.0 GPU-h, owner-raised on 2026-08-10. Part A is priced at "
                "well under 0.5 GPU-h; Part B's fuzzy symmetrisation is the "
                "~4.3 h product step R0241 deferred. Every attempt's GPU time "
                "is charged, including one that produces nothing."
            ),
            "fuzzy_deadline_s": FUZZY_DEADLINE_S,
            "locality_deadline_s": LOCALITY_DEADLINE_S,
            "fuzzy_receipt_file": FUZZY_FILE,
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
    path = prepare_round0242(
        release_sha=args.release_sha, queue_root=args.queue_root,
        prior_gpu_wall_s=args.prior_gpu_wall_s,
    )
    print(json.dumps({"queue": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
