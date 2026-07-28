#!/usr/bin/env python3
"""Prepare one bounded 150M graph part or the reviewed CPU assembly."""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0088_graph import (
    ASSEMBLY_ROUND_ID,
    CORPUS_BY_ROUND,
    CORPUS_SPECS,
    ROUND_BY_CORPUS,
    Round0088Error,
    projected_corpus_wall_seconds,
    selected_benchmark_seconds_per_query,
    validate_filter_receipt,
    validate_part_receipt,
    validate_qualification,
    validate_staged_substrate,
)
from basemap.round0093_policy import load_decision as load_r0093_decision
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.round0059_nodes import FAISS_WHEEL
from experiments.round0088_nodes import RUNTIME_SPEC


RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
STAGING_ROOT = "/data/latent-basemap/runs/round-0086/queue/artifacts"
SUBSTRATE = os.path.join(
    STAGING_ROOT,
    "balanced-150m-substrate/balanced-150m-substrate-v1.json",
)
FILTERED_INDEX = os.path.join(
    STAGING_ROOT,
    "filtered-index-150m/balanced-150m-retained.ivfpq",
)
FILTER_RECEIPT = os.path.join(
    STAGING_ROOT,
    "filtered-index-150m/filter-receipt.json",
)
QUALIFICATION = os.path.join(
    "/data/latent-basemap/runs/round-0093/queue/artifacts",
    "lower-recall-policy-qualification-150m/"
    "gpu-ivfpq-policy-qualification-v1.json",
)
POLICY_DECISION = os.path.join(
    "/data/latent-basemap/runs/round-0093/queue/artifacts",
    "lower-recall-policy-qualification-150m/"
    "lower-recall-policy-decision.json",
)
PART_ROOTS = {
    corpus: (
        f"/data/latent-basemap/runs/round-{round_id}/queue/artifacts/"
        f"graph-part-{corpus}"
    )
    for corpus, round_id in ROUND_BY_CORPUS.items()
}


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_issued_round(path: str) -> None:
    if _frontmatter_status(path) != "issued":
        raise RuntimeError(f"{path} remains draft; refuse queue materialization")


def _require_review(
    path: str,
    *,
    expected_sha256: str,
    required_text: tuple[str, ...],
) -> dict[str, Any]:
    if _frontmatter_status(path) not in {"accepted", "partial"}:
        raise RuntimeError(f"{path} does not release reviewed evidence")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if any(value not in text for value in required_text):
        raise RuntimeError(f"{path} does not bind required evidence")
    return signature


def _authenticated_staging(
    *,
    substrate_sha256: str,
    filter_receipt_sha256: str,
    filtered_index_sha256: str,
    qualification_sha256: str,
    policy_decision_sha256: str,
) -> dict[str, Any]:
    substrate = validate_staged_substrate(
        SUBSTRATE, expected_sha256=substrate_sha256
    )
    filtered = expected_input_signature(FILTERED_INDEX)
    if filtered["sha256"] != filtered_index_sha256:
        raise Round0088Error("R0086 filtered-index bytes changed")
    filter_receipt = validate_filter_receipt(
        FILTER_RECEIPT,
        expected_sha256=filter_receipt_sha256,
        substrate_signature=substrate["signature"],
        filtered_index_signature=filtered,
    )
    qualification = validate_qualification(
        QUALIFICATION,
        expected_sha256=qualification_sha256,
        substrate_signature=substrate["signature"],
        filtered_index_signature=filtered,
    )
    decision = load_r0093_decision(
        POLICY_DECISION,
        expected_sha256=policy_decision_sha256,
    )
    decision_receipt = decision["receipt"]
    if (
        decision_receipt.get("selected") != qualification["selected"]
        or decision_receipt.get("qualification")
        != qualification["signature"]
        or decision_receipt.get("substrate") != substrate["signature"]
        or decision_receipt.get("filtered_index") != filtered
    ):
        raise Round0088Error(
            "R0093 decision does not bind the selected qualification"
        )
    return {
        "substrate": substrate,
        "filtered_index": filtered,
        "filter_receipt": filter_receipt,
        "qualification": qualification,
        "policy_decision": decision,
    }


def prepare_part_queue(
    *,
    round_id: str,
    release_sha: str,
    r0086_review_path: str,
    r0086_review_sha256: str,
    r0093_review_path: str,
    r0093_review_sha256: str,
    substrate_sha256: str,
    filter_receipt_sha256: str,
    filtered_index_sha256: str,
    qualification_sha256: str,
    policy_decision_sha256: str,
    runtime_spec_sha256: str,
    queue_root: str | None = None,
) -> str:
    if round_id not in CORPUS_BY_ROUND:
        raise ValueError(f"{round_id} is not a 150M graph-part round")
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("graph-part release SHA must be one full commit")
    corpus = CORPUS_BY_ROUND[round_id]
    round_file = os.path.join(
        LAB_ROOT, f"round-{round_id}-2026-07-28.md"
    )
    _require_issued_round(round_file)
    review_0086 = _require_review(
        r0086_review_path,
        expected_sha256=r0086_review_sha256,
        required_text=(
            "capability:minilm-balanced-150m-int8-input-v1",
            substrate_sha256,
            filter_receipt_sha256,
            filtered_index_sha256,
        ),
    )
    review_0093 = _require_review(
        r0093_review_path,
        expected_sha256=r0093_review_sha256,
        required_text=(
            "capability:minilm-graph-recall-operational-floor-0p84-v1",
            (
                "capability:minilm-balanced-150m-gpu-ivfpq-"
                "search-qualified-low-recall-v1"
            ),
            qualification_sha256,
            policy_decision_sha256,
        ),
    )
    staged = _authenticated_staging(
        substrate_sha256=substrate_sha256,
        filter_receipt_sha256=filter_receipt_sha256,
        filtered_index_sha256=filtered_index_sha256,
        qualification_sha256=qualification_sha256,
        policy_decision_sha256=policy_decision_sha256,
    )
    runtime = expected_input_signature(RUNTIME_SPEC)
    if runtime["sha256"] != runtime_spec_sha256:
        raise Round0088Error("graph runtime specification changed")
    selected_search, selected_rerank = selected_benchmark_seconds_per_query(
        staged["qualification"]["selected"]
    )
    projected = projected_corpus_wall_seconds(
        corpus,
        selected_search_seconds_per_query=selected_search,
        selected_rerank_seconds_per_query=selected_rerank,
    )
    if projected > 7.5 * 3600:
        raise Round0088Error(
            f"{corpus} projected p90 {projected / 3600:.2f}h exceeds the "
            "7.5h bounded-part limit; split this corpus before issuance"
        )
    outputs = staged["substrate"]["manifest"]["outputs"]
    inputs = _dedupe([
        expected_input_signature(round_file),
        review_0086,
        review_0093,
        staged["substrate"]["signature"],
        outputs["int8"],
        outputs["scales"],
        outputs["eligibility"],
        staged["filter_receipt"]["signature"],
        staged["filtered_index"],
        staged["qualification"]["signature"],
        staged["policy_decision"]["signature"],
        runtime,
        expected_input_signature(FAISS_WHEEL),
    ])
    root = queue_root or (
        f"/data/latent-basemap/runs/round-{round_id}/queue"
    )
    root = create_fresh_directory(root, label=f"Round {round_id} graph queue")
    artifacts = ensure_data_directory(os.path.join(root, "artifacts"))
    output = os.path.join(artifacts, f"graph-part-{corpus}")
    manifest = _base_manifest(
        round_id=round_id,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=root,
        gpu_hours_cap=8.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0088-split-150m-graph-part-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0086", "0093"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-150m-int8-input-v1",
        "minilm-graph-recall-operational-floor-0p84-v1",
        "minilm-balanced-150m-gpu-ivfpq-search-qualified-low-recall-v1",
    ]
    capability = f"minilm-balanced-150m-graph-part-{corpus}-v1"
    manifest["capabilities_produced"] = [capability]
    manifest["training_performed"] = False
    manifest["reviewed_inputs"] = {
        "review_0086": review_0086,
        "review_0093": review_0093,
        "substrate": staged["substrate"]["signature"],
        "filter_receipt": staged["filter_receipt"]["signature"],
        "filtered_index": staged["filtered_index"],
        "gpu_qualification": staged["qualification"]["signature"],
        "policy_decision": staged["policy_decision"]["signature"],
    }
    selected = staged["qualification"]["selected"]
    spec = CORPUS_SPECS[corpus]
    manifest["scientific_contract"] = {
        "corpus": corpus,
        "global_start": spec["start"],
        "global_stop": spec["stop"],
        "retained_sources": spec["retained_rows"],
        "excluded_sources": spec["excluded_rows"],
        "nprobe": int(selected["nprobe"]),
        "search_width": int(selected["shortlist_width"]),
        "selected_neighbors": 15,
        "mean_recall_floor": staged["qualification"]["mean_recall_floor"],
        "exact_rerank": True,
        "fixed_degree_on_retained_sources": 15,
        "candidate_universe": "all retained balanced-150m representatives",
        "no_training": True,
        "no_scale_decision": True,
        "r0078_corpus_calibrated_p90_seconds": projected,
    }
    node_id = f"build_balanced_150m_graph_part_{corpus}"
    manifest["jobs"] = [{
        "id": node_id,
        "action": "build_150m_graph_part",
        "handler_module": "experiments.round0088_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
        "expected_inputs": inputs,
        "p90_wall_s": projected,
        "corpus": corpus,
        "substrate_manifest": SUBSTRATE,
        "substrate_manifest_sha256": substrate_sha256,
        "filter_receipt": FILTER_RECEIPT,
        "filter_receipt_sha256": filter_receipt_sha256,
        "filtered_index": FILTERED_INDEX,
        "gpu_qualification_receipt": QUALIFICATION,
        "gpu_qualification_receipt_sha256": qualification_sha256,
        "policy_decision": POLICY_DECISION,
        "policy_decision_sha256": policy_decision_sha256,
        "runtime_spec": RUNTIME_SPEC,
        "runtime_spec_sha256": runtime_spec_sha256,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {node_id: projected, "total": projected}
    path = os.path.join(root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def prepare_assembly_queue(
    *,
    release_sha: str,
    review_paths: dict[str, str],
    review_sha256: dict[str, str],
    part_receipt_sha256: dict[str, str],
    substrate_sha256: str,
    part_roots: dict[str, str] | None = None,
    queue_root: str = (
        "/data/latent-basemap/runs/round-0091/queue"
    ),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("assembly release SHA must be one full commit")
    round_file = os.path.join(LAB_ROOT, "round-0091-2026-07-28.md")
    _require_issued_round(round_file)
    substrate = validate_staged_substrate(
        SUBSTRATE, expected_sha256=substrate_sha256
    )
    roots = dict(part_roots or PART_ROOTS)
    if set(roots) != set(CORPUS_SPECS):
        raise ValueError("assembly must bind one root for every graph part")
    parts: dict[str, dict[str, Any]] = {}
    reviews: dict[str, dict[str, Any]] = {}
    shard_receipts: list[dict[str, Any]] = []
    for corpus, round_id in ROUND_BY_CORPUS.items():
        part_path = os.path.join(roots[corpus], "part-receipt.json")
        parts[corpus] = validate_part_receipt(
            part_path,
            expected_sha256=part_receipt_sha256[corpus],
        )
        capability = f"capability:minilm-balanced-150m-graph-part-{corpus}-v1"
        reviews[corpus] = _require_review(
            review_paths[corpus],
            expected_sha256=review_sha256[corpus],
            required_text=(capability, part_receipt_sha256[corpus]),
        )
        expected_count = (
            CORPUS_SPECS[corpus]["stop"]
            - CORPUS_SPECS[corpus]["start"]
        ) // 100_000
        receipt_paths = sorted(glob.glob(
            os.path.join(roots[corpus], "shards", "receipt-*.json")
        ))
        if len(receipt_paths) != expected_count:
            raise Round0088Error(
                f"{corpus} has {len(receipt_paths)} shard receipts, "
                f"expected {expected_count}"
            )
        shard_receipts.extend(
            expected_input_signature(path) for path in receipt_paths
        )
    inputs = _dedupe([
        expected_input_signature(round_file),
        substrate["signature"],
        substrate["manifest"]["outputs"]["eligibility"],
        *reviews.values(),
        *(value["signature"] for value in parts.values()),
        *shard_receipts,
    ])
    root = create_fresh_directory(
        queue_root, label="Round 0091 graph assembly queue"
    )
    artifacts = ensure_data_directory(os.path.join(root, "artifacts"))
    output = os.path.join(artifacts, "canonical-graph-balanced-150m")
    manifest = _base_manifest(
        round_id=ASSEMBLY_ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=root,
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    manifest["schema"] = "round0091-balanced-150m-assembly-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "cpu"
    manifest["required_reviews"] = ["0088", "0089", "0090"]
    manifest["capability_dependencies"] = [
        f"minilm-balanced-150m-graph-part-{corpus}-v1"
        for corpus in CORPUS_SPECS
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-150m-gpu-native-graph-v1"
    ]
    manifest["training_performed"] = False
    manifest["reviewed_inputs"] = {
        **{f"review_{ROUND_BY_CORPUS[key]}": value
           for key, value in reviews.items()},
        **{f"part_{key}": value["signature"]
           for key, value in parts.items()},
        "substrate": substrate["signature"],
    }
    manifest["scientific_contract"] = {
        "rows": 150_000_000,
        "retained_sources": 147_221_757,
        "excluded_sources": 2_778_243,
        "fixed_degree": 15,
        "valid_edges": 2_208_326_355,
        "parts": list(CORPUS_SPECS),
        "assembly_only": True,
        "no_training": True,
        "no_scale_decision": True,
    }
    node_id = "assemble_balanced_150m_canonical_graph"
    manifest["jobs"] = [{
        "id": node_id,
        "action": "assemble_150m_graph",
        "handler_module": "experiments.round0088_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
        "expected_inputs": inputs,
        "p90_wall_s": 1_800.0,
        "substrate_manifest": SUBSTRATE,
        "substrate_manifest_sha256": substrate_sha256,
        "part_roots": roots,
        "part_receipt_sha256": part_receipt_sha256,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {"total": 0.0}
    path = os.path.join(root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    part = sub.add_parser("part")
    part.add_argument("--round-id", required=True)
    part.add_argument("--release-sha", required=True)
    part.add_argument("--r0086-review", required=True)
    part.add_argument("--r0086-review-sha256", required=True)
    part.add_argument("--r0093-review", required=True)
    part.add_argument("--r0093-review-sha256", required=True)
    part.add_argument("--substrate-sha256", required=True)
    part.add_argument("--filter-receipt-sha256", required=True)
    part.add_argument("--filtered-index-sha256", required=True)
    part.add_argument("--qualification-sha256", required=True)
    part.add_argument("--policy-decision-sha256", required=True)
    part.add_argument("--runtime-spec-sha256", required=True)
    part.add_argument("--queue-root")
    assembly = sub.add_parser("assembly")
    assembly.add_argument("--release-sha", required=True)
    assembly.add_argument("--substrate-sha256", required=True)
    for corpus in CORPUS_SPECS:
        assembly.add_argument(f"--{corpus}-review", required=True)
        assembly.add_argument(f"--{corpus}-review-sha256", required=True)
        assembly.add_argument(f"--{corpus}-part-receipt-sha256", required=True)
        assembly.add_argument(
            f"--{corpus}-part-root",
            default=PART_ROOTS[corpus],
        )
    assembly.add_argument(
        "--queue-root",
        default="/data/latent-basemap/runs/round-0091/queue",
    )
    args = parser.parse_args(argv)
    if args.command == "part":
        print(prepare_part_queue(
            round_id=args.round_id,
            release_sha=args.release_sha,
            r0086_review_path=args.r0086_review,
            r0086_review_sha256=args.r0086_review_sha256,
            r0093_review_path=args.r0093_review,
            r0093_review_sha256=args.r0093_review_sha256,
            substrate_sha256=args.substrate_sha256,
            filter_receipt_sha256=args.filter_receipt_sha256,
            filtered_index_sha256=args.filtered_index_sha256,
            qualification_sha256=args.qualification_sha256,
            policy_decision_sha256=args.policy_decision_sha256,
            runtime_spec_sha256=args.runtime_spec_sha256,
            queue_root=args.queue_root,
        ))
        return 0
    if args.command == "assembly":
        print(prepare_assembly_queue(
            release_sha=args.release_sha,
            review_paths={
                corpus: getattr(args, f"{corpus}_review")
                for corpus in CORPUS_SPECS
            },
            review_sha256={
                corpus: getattr(args, f"{corpus}_review_sha256")
                for corpus in CORPUS_SPECS
            },
            part_receipt_sha256={
                corpus: getattr(
                    args, f"{corpus}_part_receipt_sha256"
                )
                for corpus in CORPUS_SPECS
            },
            substrate_sha256=args.substrate_sha256,
            part_roots={
                corpus: getattr(args, f"{corpus}_part_root")
                for corpus in CORPUS_SPECS
            },
            queue_root=args.queue_root,
        ))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
