#!/usr/bin/env python3
"""Prepare the retained diverse-Jina fuzzy-graph queue."""
from __future__ import annotations

import argparse
import glob
import json
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
from basemap.round0105_search import ELIGIBILITY_PATH, ELIGIBILITY_SHA256, K
from basemap.round0106_graph import (
    LOCAL_CONNECTIVITY,
    N_NEIGHBORS,
    PAIR_BUCKETS,
    PARTS,
    RETAINED_ROWS,
    ROUND_ID,
    SEARCH_BATCH_ROWS,
    SHARD_ROWS,
    validate_search_artifacts,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.round0105_nodes import _substrate_arrays


ROUND_ROOT = "/data/latent-basemap/runs/round-0106"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0106-*.md")
R0105_ROOT = (
    "/data/latent-basemap/runs/round-0105/queue-attempt-3/artifacts"
)
INDEX = os.path.join(
    R0105_ROOT,
    "retained-index",
    "jina-diverse-25m-retained.ivfpq",
)
INDEX_RECEIPT = os.path.join(
    R0105_ROOT, "retained-index", "index-receipt.json"
)
QUALIFICATION = os.path.join(
    R0105_ROOT, "search-qualification", "search-qualification.json"
)
DECISION = os.path.join(
    R0105_ROOT, "search-qualification", "search-policy-decision.json"
)

REVIEW_DEFAULTS = {
    "0087": (
        os.path.join(LAB_ROOT, "review-0087-2026-07-28.md"),
        "61ab9268899c2edc47519bdbe4efeea65a54f0c9fda52bd89e7cad0dafd9d483",
        "capability:jina-diverse-25m-inventory-v1",
    ),
    "0103": (
        os.path.join(LAB_ROOT, "review-0103-2026-07-29.md"),
        "c6c4f780c20cc34c7707132581ffaaf8daa8cc7ea9eb1cee3f76e128b6c37a51",
        "capability:jina-diverse-25m-full768-int8-substrate-v1",
    ),
}


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4_096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter_status(path) == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0106 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str,
    *,
    expected_sha256: str,
    capability: str,
) -> dict[str, Any]:
    if _frontmatter_status(path) != "accepted":
        raise RuntimeError(f"{path} is not an accepted review")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if capability not in text:
        raise RuntimeError(f"{path} lacks {capability}")
    return signature


def _part_job(
    *,
    node_id: str,
    part: str,
    output: str,
    inputs: list[dict[str, Any]],
    p90_wall_s: float,
    release_sha: str,
    search: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": node_id,
        "action": "build_part",
        "handler_module": "experiments.round0106_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            os.path.dirname(output), f"{node_id}.done.json"
        ),
        "expected_inputs": inputs,
        "p90_wall_s": p90_wall_s,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
        "part": part,
        "release_sha": release_sha,
        "index": INDEX,
        "index_sha256": search["index"]["sha256"],
        "index_receipt": INDEX_RECEIPT,
        "index_receipt_sha256": search["index_receipt"]["sha256"],
        "qualification": QUALIFICATION,
        "qualification_sha256": search["qualification"]["sha256"],
        "decision": DECISION,
        "decision_sha256": search["decision"]["sha256"],
    }


def prepare_round0106(
    *,
    release_sha: str,
    r0105_review_path: str,
    r0105_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0106 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = {
        round_id: _require_review(
            path, expected_sha256=sha, capability=capability
        )
        for round_id, (path, sha, capability) in REVIEW_DEFAULTS.items()
    }
    reviews["0105"] = _require_review(
        r0105_review_path,
        expected_sha256=r0105_review_sha256,
        capability="capability:jina-diverse-25m-full768-search-qualified-v1",
    )
    substrate, _excluded, _encoded, _scales = _substrate_arrays()
    eligibility = expected_input_signature(ELIGIBILITY_PATH)
    if eligibility["sha256"] != ELIGIBILITY_SHA256:
        raise RuntimeError("R0087 eligibility bytes changed")
    search = validate_search_artifacts(
        index_path=INDEX,
        index_sha256=expected_input_signature(INDEX)["sha256"],
        index_receipt_path=INDEX_RECEIPT,
        index_receipt_sha256=expected_input_signature(INDEX_RECEIPT)["sha256"],
        qualification_path=QUALIFICATION,
        qualification_sha256=expected_input_signature(QUALIFICATION)["sha256"],
        decision_path=DECISION,
        decision_sha256=expected_input_signature(DECISION)["sha256"],
        substrate_signature=substrate["signature"],
    )
    queue_root = create_fresh_directory(
        queue_root, label="R0106 retained fuzzy-graph queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    inputs = _dedupe([
        expected_input_signature(round_file),
        *reviews.values(),
        substrate["signature"],
        substrate["payloads"]["int8"],
        substrate["payloads"]["scales"],
        substrate["payloads"]["labels"],
        eligibility,
        search["index"],
        search["index_receipt"],
        search["qualification"],
        search["decision"],
    ])
    part_outputs = {
        part: os.path.join(artifacts, f"graph-part-{part}")
        for part in PARTS
    }
    part_p90 = {
        "english": 9_600.0,
        "languages-a": 8_400.0,
        "languages-b": 9_000.0,
    }
    jobs = [
        _part_job(
            node_id=f"build_{part.replace('-', '_')}_part",
            part=part,
            output=part_outputs[part],
            inputs=inputs,
            p90_wall_s=part_p90[part],
            release_sha=release_sha,
            search=search,
        )
        for part in PARTS
    ]
    assembly_output = os.path.join(artifacts, "canonical-fuzzy-graph")
    jobs.append({
        "id": "assemble_canonical_fuzzy_graph",
        "action": "assemble_graph",
        "handler_module": "experiments.round0106_nodes",
        "handler_callable": "run_job",
        "deps": [job["id"] for job in jobs],
        "outputs": [assembly_output],
        "done_marker": os.path.join(
            artifacts, "assemble_canonical_fuzzy_graph.done.json"
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 3_600.0,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
        },
        "part_outputs": part_outputs,
    })
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=8.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0106-retained-fuzzy-graph-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0087", "0103", "0105"]
    manifest["capability_dependencies"] = [
        "jina-diverse-25m-inventory-v1",
        "jina-diverse-25m-full768-int8-substrate-v1",
        "jina-diverse-25m-full768-search-qualified-v1",
    ]
    manifest["capabilities_produced"] = [
        "jina-diverse-25m-full768-fuzzy-graph-v1"
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "retained_rows": RETAINED_ROWS,
        "k_real": K,
        "n_neighbors_including_self": N_NEIGHBORS,
        "local_connectivity": LOCAL_CONNECTIVITY,
        "parts": PARTS,
        "shard_rows": SHARD_ROWS,
        "search_batch_rows": SEARCH_BATCH_ROWS,
        "selected_search_policy": search["selected"],
        "candidate_universe": "one-global-physically-retained-R0105-index",
        "exact_rerank": "native-int8-plus-fp16-scale-cosine-fp32-on-gpu",
        "compact_id_universe": [0, RETAINED_ROWS],
        "directed_fuzzy": "umap-smooth-knn-distance",
        "symmetrization": "a+b-a*b-set-op-mix-ratio-1",
        "pair_buckets": PAIR_BUCKETS,
        "corpus_neighbor_quotas": False,
        "mixing_diagnostics": "22x22-source-destination",
        "no_training": True,
        "no_map_decision": True,
    }
    manifest["jobs"] = jobs
    manifest["p90_gpu_seconds"] = {
        f"build_{part.replace('-', '_')}_part": part_p90[part]
        for part in PARTS
    }
    manifest["p90_gpu_seconds"]["total"] = sum(part_p90.values())
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0105-review", required=True)
    parser.add_argument("--r0105-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0106(
        release_sha=args.release_sha,
        r0105_review_path=args.r0105_review,
        r0105_review_sha256=args.r0105_review_sha256,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
