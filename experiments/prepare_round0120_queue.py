#!/usr/bin/env python3
"""Prepare, but never launch, R0120 canonical prompted-Pile production."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0112_prompt_substrate import model_member_signatures
from basemap.round0120_prompted_pile import (
    CAPABILITY,
    CHUNK_ROWS,
    CORPUS_ROWS,
    DATASET,
    EMBED_MINIMUM_ROWS_PER_S,
    EMBED_WARNING_ROWS_PER_S,
    INVENTORY_MANIFEST_PATH,
    PERFORMANCE_GUARD_ROWS,
    PROMPT_PREFIX,
    R0087_PILE_GLOBAL_OFFSET,
    R0087_PILE_GLOBAL_STOP,
    R0114_MANIFEST_PATH,
    ROUND_ID,
    WORK_RANGES,
    canonical_source_layout,
    clip_layout,
    environment_freeze_receipt,
    load_r0114_model_prompt_closure,
    model_contract,
    production_payload_bytes,
    required_free_bytes,
    source_manifest_summary,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0120"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0120-*.md")
OUTPUT_NAMESPACE = "canonical-jina-document-pile-native8192-v1"
R0116_RELEASE_SHA = "5243a994c45c1fdfacdf48b665ad00077d798286"
R0116_TERMINAL_PATH = (
    "/data/latent-basemap/runs/round-0116/queue/runner-terminal.json"
)
R0116_REQUIRED_JOBS = (
    "embed_fineweb_tail",
    "embed_redpajama_00",
    "embed_redpajama_01",
    "embed_redpajama_02",
    "finalize_canonical_prompted_english",
)
REVIEW_DEFAULTS = {
    "0087": (
        "review-0087-2026-07-28.md",
        "61ab9268899c2edc47519bdbe4efeea65a54f0c9fda52bd89e7cad0dafd9d483",
        "capability:jina-diverse-25m-inventory-v1",
    ),
    "0114": (
        "review-0114-2026-07-30.md",
        "610a9abb93f3fb6908a018d855f81feecc1045e261c007a3ca13ad8379eec4b9",
        "capability:jina-fineweb-2m-dual-prompt-native8192-substrate-v2",
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
            f"R0120 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str, *, expected_sha256: str, capability: str
) -> dict[str, Any]:
    if _frontmatter_status(path) != "accepted":
        raise RuntimeError(f"{path} is not an accepted review")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if capability not in text:
        raise RuntimeError(f"{path} does not release {capability}")
    return signature


def _require_successful_r0116_terminal(
    path: str, *, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    if signature["canonical_path"] != R0116_TERMINAL_PATH:
        raise RuntimeError("R0116 terminal receipt is not the canonical receipt")
    if signature["sha256"] != expected_sha256:
        raise RuntimeError("R0116 terminal receipt bytes changed")
    with open(path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    required_jobs = list(R0116_REQUIRED_JOBS)
    checkouts = (
        terminal.get("release_checkout"),
        terminal.get("release_checkout_at_finish"),
    )
    checkout_valid = all(
        isinstance(checkout, dict)
        and checkout.get("repo_root") == RELEASE_ROOT
        and checkout.get("head") == R0116_RELEASE_SHA
        and checkout.get("detached") is True
        and checkout.get("dirty") is False
        for checkout in checkouts
    )
    queue_sha = terminal.get("queue_manifest_sha256")
    queue_sha_at_finish = terminal.get("queue_manifest_sha256_at_finish")
    nodes = terminal.get("nodes")
    nodes_valid = (
        isinstance(nodes, list)
        and [node.get("node") for node in nodes if isinstance(node, dict)]
        == required_jobs
        and len(nodes) == len(required_jobs)
        and all(
            isinstance(node, dict)
            and node.get("returncode") == 0
            and node.get("validation_problems") in (None, [])
            for node in nodes
        )
    )
    if (
        terminal.get("schema") != "slim-runner-terminal-v3"
        or terminal.get("round_id") != "0116"
        or terminal.get("verdict") != "succeeded"
        or terminal.get("stop_reason") is not None
        or terminal.get("required_jobs") != required_jobs
        or terminal.get("completed_jobs") != required_jobs
        or not checkout_valid
        or terminal.get("release_checkout_unchanged") is not True
        or not isinstance(queue_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", queue_sha) is None
        or queue_sha_at_finish != queue_sha
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("boundary_problems") not in (None, [])
        or terminal.get("validation_problems") not in (None, [])
        or not nodes_valid
    ):
        raise RuntimeError("R0116 did not reach a clean terminal evaluation")
    return terminal, signature


def _node_p90(rows: int) -> float:
    # Accepted R0112 receipts measured 252.90--254.15 convention rows/s.
    # Budget at 200 rows/s plus five minutes of closure/source/receipt work.
    return float(rows / 200.0 + 300.0)


def rehearse_round0120_inputs() -> dict[str, Any]:
    """Validate the real immutable inputs, mapping, and disk without writing."""
    reviews = {
        round_id: _require_review(
            os.path.join(LAB_ROOT, name),
            expected_sha256=sha256,
            capability=capability,
        )
        for round_id, (name, sha256, capability) in REVIEW_DEFAULTS.items()
    }
    layout, inventory_signature = canonical_source_layout()
    _, r0114_signature = load_r0114_model_prompt_closure()
    model_members = model_member_signatures()
    disk = shutil.disk_usage("/data")
    required = required_free_bytes()
    work = []
    for node_id, start, stop in WORK_RANGES:
        clipped = clip_layout(layout, start=start, stop=stop)
        work.append(
            {
                "node_id": node_id,
                "dataset_row_range": [start, stop],
                "rows": stop - start,
                "source_slices": len(clipped),
                "p90_wall_s": _node_p90(stop - start),
            }
        )
    if sum(item["rows"] for item in work) != CORPUS_ROWS:
        raise RuntimeError("R0120 rehearsal work ranges do not close")
    return {
        "round_id": ROUND_ID,
        "reviews": reviews,
        "inventory_manifest": inventory_signature,
        "r0114_model_prompt_manifest": r0114_signature,
        "model_member_count": len(model_members),
        "source_manifest": source_manifest_summary(layout),
        "source_layout_first": layout[0],
        "source_layout_last": layout[-1],
        "work_ranges": work,
        "p90_gpu_seconds": sum(item["p90_wall_s"] for item in work),
        "disk": {
            "filesystem": "/data",
            "free_bytes_observed": disk.free,
            "payload_bytes": production_payload_bytes(),
            "required_free_bytes": required,
            "passed": disk.free >= required,
        },
    }


def prepare_round0120(
    *,
    release_sha: str,
    r0116_terminal_path: str,
    r0116_terminal_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0120 release SHA must be one full commit")
    round_file = _require_issued_round()
    rehearsal = rehearse_round0120_inputs()
    if rehearsal["disk"]["passed"] is not True:
        raise RuntimeError(
            "R0120 disk preflight failed: "
            f"{rehearsal['disk']['free_bytes_observed']:,} free bytes < "
            f"{rehearsal['disk']['required_free_bytes']:,} required"
        )
    r0116_terminal, r0116_terminal_signature = (
        _require_successful_r0116_terminal(
            r0116_terminal_path,
            expected_sha256=r0116_terminal_sha256,
        )
    )
    reviews = rehearsal["reviews"]
    layout, inventory_signature = canonical_source_layout()
    _, r0114_signature = load_r0114_model_prompt_closure()
    model_members = model_member_signatures()
    environment_freeze = environment_freeze_receipt()
    model_inputs = [
        {
            key: value
            for key, value in member.items()
            if key != "model_relative_path"
        }
        for member in model_members
    ]

    queue_root = create_fresh_directory(
        queue_root, label="R0120 canonical prompted-Pile production queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    namespace = ensure_data_directory(
        os.path.join(artifacts, OUTPUT_NAMESPACE),
        label="R0120 prompted-Pile output namespace",
    )
    common_inputs = _dedupe(
        [
            expected_input_signature(round_file),
            *reviews.values(),
            r0116_terminal_signature,
            inventory_signature,
            r0114_signature,
            *model_inputs,
        ]
    )
    jobs: list[dict[str, Any]] = []
    receipt_paths: list[str] = []
    for node_id, start, stop in WORK_RANGES:
        node_layout = clip_layout(layout, start=start, stop=stop)
        output = os.path.join(namespace, node_id)
        receipt_path = os.path.join(output, "node-receipt.json")
        receipt_paths.append(receipt_path)
        jobs.append(
            {
                "id": node_id,
                "action": "embed_document_rows",
                "handler_module": "experiments.round0120_nodes",
                "handler_callable": "run_job",
                "deps": [],
                "outputs": [output],
                "done_marker": os.path.join(
                    artifacts, f"{node_id}.done.json"
                ),
                "expected_inputs": _dedupe(
                    [
                        *common_inputs,
                        *[dict(item["text"]) for item in node_layout],
                    ]
                ),
                "p90_wall_s": _node_p90(stop - start),
                "dataset": DATASET,
                "dataset_row_start": start,
                "dataset_row_stop": stop,
                "corpus_global_row_start": start,
                "corpus_global_row_stop": stop,
                "r0087_global_row_start": (
                    R0087_PILE_GLOBAL_OFFSET + start
                ),
                "r0087_global_row_stop": R0087_PILE_GLOBAL_OFFSET + stop,
                "authenticated_source_layout": node_layout,
                "inventory_manifest_path": INVENTORY_MANIFEST_PATH,
                "inventory_manifest_signature": inventory_signature,
                "r0114_manifest_path": R0114_MANIFEST_PATH,
                "r0114_manifest_signature": r0114_signature,
                "environment_freeze": environment_freeze,
                "node_policy": {
                    "gpu_required": True,
                    "training_performed": False,
                },
            }
        )

    final_id = "finalize_canonical_prompted_pile"
    final_output = os.path.join(artifacts, CAPABILITY)
    jobs.append(
        {
            "id": final_id,
            "action": "finalize_canonical_prompted_pile",
            "handler_module": "experiments.round0120_nodes",
            "handler_callable": "run_job",
            "deps": [item[0] for item in WORK_RANGES],
            "outputs": [final_output],
            "done_marker": os.path.join(
                artifacts, f"{final_id}.done.json"
            ),
            "expected_inputs": _dedupe(
                [
                    *common_inputs,
                    *[dict(item["text"]) for item in layout],
                ]
            ),
            "p90_wall_s": 600.0,
            "node_receipts": receipt_paths,
            "canonical_source_layout": layout,
            "inventory_manifest_path": INVENTORY_MANIFEST_PATH,
            "inventory_manifest_signature": inventory_signature,
            "r0114_manifest_path": R0114_MANIFEST_PATH,
            "r0114_manifest_signature": r0114_signature,
            "environment_freeze": environment_freeze,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
        }
    )

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=6.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue["schema"] = "round0120-canonical-prompted-pile-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-data-production-filler"
    queue["required_reviews"] = list(REVIEW_DEFAULTS)
    queue["ordering_dependencies"] = ["0116-clean-terminal"]
    queue["capability_dependencies"] = [
        "jina-diverse-25m-inventory-v1",
        "jina-fineweb-2m-dual-prompt-native8192-substrate-v2",
    ]
    queue["capabilities_produced"] = [CAPABILITY]
    queue["training_performed"] = False
    queue["output_namespace"] = namespace
    queue["scientific_contract"] = {
        "purpose": "canonical prompted Pile embedding tranche only",
        "scheduling_role": (
            "lower priority than R0116; run only after its clean terminal "
            "proves the shared embedding setup"
        ),
        "r0116_ordering_terminal": r0116_terminal_signature,
        "r0116_scientific_result_required": False,
        "r0116_review_required": False,
        "r0116_terminal_summary": {
            key: r0116_terminal.get(key)
            for key in (
                "verdict",
                "completed_jobs",
                "required_jobs",
                "release_checkout_unchanged",
                "queue_manifest_unchanged",
            )
        },
        "dataset": DATASET,
        "corpus_rows": CORPUS_ROWS,
        "corpus_local_row_range": [0, CORPUS_ROWS],
        "r0087_global_row_range": [
            R0087_PILE_GLOBAL_OFFSET,
            R0087_PILE_GLOBAL_STOP,
        ],
        "new_gpu_rows": CORPUS_ROWS,
        "model": model_contract(),
        "atomic_chunk_rows_maximum": CHUNK_ROWS,
        "work_ranges": [
            {
                "node_id": node_id,
                "dataset_row_range": [start, stop],
                "rows": stop - start,
            }
            for node_id, start, stop in WORK_RANGES
        ],
        "source_manifest": source_manifest_summary(layout),
        "inventory_manifest": inventory_signature,
        "r0114_model_prompt_manifest": r0114_signature,
        "accepted_raw_embeddings_are_identity_evidence_not_inputs": True,
        "performance_calibration": {
            "source": "accepted R0112/R0114 embedding receipts",
            "measured_document_rows_per_s_range": [
                252.89684892474966,
                254.15208030721982,
            ],
            "expected_document_rows_per_s": 253.0,
            "expected_encode_gpu_hours": CORPUS_ROWS / 253.0 / 3600.0,
            "expected_gpu_hours_with_fixed_overhead": 4.0,
            "p90_gpu_hours": sum(
                _node_p90(stop - start)
                for _, start, stop in WORK_RANGES
            )
            / 3600.0,
            "p90_budget_document_rows_per_s": 200.0,
            "minimum_document_rows_per_s": EMBED_MINIMUM_ROWS_PER_S,
            "warning_document_rows_per_s": EMBED_WARNING_ROWS_PER_S,
            "guard_evaluated_after_rows": PERFORMANCE_GUARD_ROWS,
        },
        "disk_preflight": rehearsal["disk"],
        "environment_freeze": environment_freeze,
        "prompt_prefix": PROMPT_PREFIX,
        "source_order": [DATASET],
        "no_graph": True,
        "no_map": True,
        "no_training": True,
        "no_quality_claim": True,
        "no_prompt_transfer_claim": True,
        "no_production_promotion": True,
        "no_complete_sae_corpus_claim": True,
    }
    queue["jobs"] = jobs
    gpu_p90 = {
        job["id"]: float(job["p90_wall_s"])
        for job in jobs
        if job["node_policy"]["gpu_required"]
    }
    queue["p90_gpu_seconds"] = {
        **gpu_p90,
        "total": sum(gpu_p90.values()),
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha")
    parser.add_argument("--r0116-terminal")
    parser.add_argument("--r0116-terminal-sha256")
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    parser.add_argument("--rehearse-only", action="store_true")
    args = parser.parse_args(argv)
    if args.rehearse_only:
        value = {"rehearsal": rehearse_round0120_inputs()}
    else:
        missing = [
            name
            for name, value in (
                ("--release-sha", args.release_sha),
                ("--r0116-terminal", args.r0116_terminal),
                (
                    "--r0116-terminal-sha256",
                    args.r0116_terminal_sha256,
                ),
            )
            if not value
        ]
        if missing:
            parser.error("required for queue preparation: " + ", ".join(missing))
        value = {
            "queue_manifest": prepare_round0120(
                release_sha=args.release_sha,
                r0116_terminal_path=args.r0116_terminal,
                r0116_terminal_sha256=args.r0116_terminal_sha256,
                queue_root=args.queue_root,
            )
        }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
