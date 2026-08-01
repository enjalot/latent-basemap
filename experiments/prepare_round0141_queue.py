#!/usr/bin/env python3
"""Rehearse or prepare, but never launch, R0141 multilingual production."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0112_prompt_substrate import model_member_signatures
from basemap.round0141_prompted_multilingual import (
    BATCH_SIZE,
    CAPABILITY,
    CHUNK_ROWS,
    CORPUS_ROWS,
    EMBED_MINIMUM_ROWS_PER_S,
    EMBED_WARNING_ROWS_PER_S,
    EXPECTED_FIXED_SECONDS_PER_NODE,
    EXPECTED_ROWS_PER_S,
    GPU_HOURS_CAP,
    INVENTORY_MANIFEST_PATH,
    LANGUAGE_TRANCHES,
    PERFORMANCE_GUARD_ROWS,
    PROMPT_PREFIX,
    R0087_GLOBAL_START,
    R0087_GLOBAL_STOP,
    R0087_REVIEW_SHA256,
    R0114_MANIFEST_PATH,
    R0114_REVIEW_SHA256,
    ROUND_ID,
    ROWS_PER_LANGUAGE,
    canonical_source_layout,
    environment_freeze_receipt,
    expected_gpu_seconds,
    load_r0114_model_prompt_closure,
    model_contract,
    node_p90_seconds,
    production_payload_bytes,
    required_free_bytes,
    source_for_node,
    source_manifest_summary,
    worst_passing_gpu_seconds,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0141"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
RUN_ENVIRONMENT_PREFIX = os.path.join(RELEASE_ROOT, ".venv")
RUN_PYTHON = os.path.join(RUN_ENVIRONMENT_PREFIX, "bin", "python")
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0141-*.md")
OUTPUT_NAMESPACE = "canonical-jina-document-native8192-multilingual-004-v1"
REVIEW_DEFAULTS = {
    "0087": (
        "review-0087-2026-07-28.md",
        R0087_REVIEW_SHA256,
        "capability:jina-diverse-25m-inventory-v1",
    ),
    "0114": (
        "review-0114-2026-07-30.md",
        R0114_REVIEW_SHA256,
        "capability:jina-fineweb-2m-dual-prompt-native8192-substrate-v2",
    ),
}
TOKEN_LENGTH_CALIBRATION = {
    "sampling": (
        "1000 numpy-linspace rows over local [0,835454), literal "
        "Document prefix, add_special_tokens=true, no truncation"
    ),
    "sample_indices_sha256": (
        "de7c7b93bbbb0510ebd4dab95e8917b3af012c850fb28d242233e9358a2bb2c5"
    ),
    "reference_pile": {"mean": 437, "p95": 572, "maximum": 1328},
    "jpn_Jpan": {"mean": 515.161, "p95": 683.05, "maximum": 731},
    "kor_Hang": {"mean": 413.641, "p95": 545.05, "maximum": 612},
    "nld_Latn": {"mean": 423.832, "p95": 606.0, "maximum": 687},
    "rows_over_native_8192": 0,
}
_SIGNATURE_KEYS = ("kind", "canonical_path", "bytes", "sha256")


def _signature_only(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in _SIGNATURE_KEYS}


def _binding(role: str, signature: dict[str, Any]) -> dict[str, Any]:
    return {"role": role, "signature": _signature_only(signature)}


def _require_dedicated_run_environment() -> None:
    observed_python = os.path.abspath(sys.executable)
    observed_prefix = os.path.abspath(sys.prefix)
    if observed_python != RUN_PYTHON or observed_prefix != RUN_ENVIRONMENT_PREFIX:
        raise RuntimeError(
            "R0141 queue preparation must use the dedicated run environment: "
            f"python={RUN_PYTHON}, prefix={RUN_ENVIRONMENT_PREFIX}; observed "
            f"python={observed_python}, prefix={observed_prefix}"
        )


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
            f"R0141 requires exactly one issued round; found {len(candidates)}"
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


def _authenticate_real_inputs() -> dict[str, Any]:
    started = time.monotonic()
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
    environment_freeze = environment_freeze_receipt()
    disk = shutil.disk_usage("/data")
    required = required_free_bytes()
    return {
        "reviews": reviews,
        "layout": layout,
        "inventory_manifest": inventory_signature,
        "r0114_model_prompt_manifest": r0114_signature,
        "model_members": model_members,
        "environment_freeze": environment_freeze,
        "disk": {
            "filesystem": "/data",
            "free_bytes_observed": disk.free,
            "payload_bytes": production_payload_bytes(),
            "required_free_bytes": required,
            "passed": disk.free >= required,
        },
        "authentication_wall_s": time.monotonic() - started,
    }


def rehearse_round0141_inputs() -> dict[str, Any]:
    """Full read-only rehearsal over the real source and closure bytes."""
    authenticated = _authenticate_real_inputs()
    layout = authenticated["layout"]
    work = [
        {
            "node_id": tranche["node_id"],
            "language": tranche["language"],
            "dataset": tranche["dataset"],
            "dataset_row_range": [0, ROWS_PER_LANGUAGE],
            "corpus_global_row_range": tranche["corpus_global_row_range"],
            "r0087_global_row_range": tranche["r0087_global_row_range"],
            "rows": ROWS_PER_LANGUAGE,
            "atomic_chunks": (ROWS_PER_LANGUAGE + CHUNK_ROWS - 1) // CHUNK_ROWS,
            "p90_wall_s": node_p90_seconds(),
        }
        for tranche in LANGUAGE_TRANCHES
    ]
    return {
        "round_id": ROUND_ID,
        "read_only": True,
        "authentication_wall_s": authenticated["authentication_wall_s"],
        "reviews": authenticated["reviews"],
        "inventory_manifest": authenticated["inventory_manifest"],
        "r0114_model_prompt_manifest": authenticated[
            "r0114_model_prompt_manifest"
        ],
        "model": model_contract(),
        "model_members": authenticated["model_members"],
        "environment_freeze": authenticated["environment_freeze"],
        "source_manifest": source_manifest_summary(layout),
        "source_layout": layout,
        "work": work,
        "performance_budget": {
            "expected_gpu_seconds": expected_gpu_seconds(),
            "expected_gpu_hours": expected_gpu_seconds() / 3_600.0,
            "p90_gpu_seconds": sum(item["p90_wall_s"] for item in work),
            "p90_gpu_hours": sum(item["p90_wall_s"] for item in work) / 3_600.0,
            "hard_cap_gpu_seconds": GPU_HOURS_CAP * 3_600.0,
            "hard_cap_gpu_hours": GPU_HOURS_CAP,
            "worst_passing_gpu_seconds": worst_passing_gpu_seconds(),
            "worst_passing_gpu_hours": worst_passing_gpu_seconds() / 3_600.0,
            "expected_rows_per_s": EXPECTED_ROWS_PER_S,
            "expected_fixed_seconds_per_node": EXPECTED_FIXED_SECONDS_PER_NODE,
            "minimum_rows_per_s": EMBED_MINIMUM_ROWS_PER_S,
            "warning_rows_per_s": EMBED_WARNING_ROWS_PER_S,
            "guard_after_rows": PERFORMANCE_GUARD_ROWS,
        },
        "token_length_calibration": TOKEN_LENGTH_CALIBRATION,
        "disk": authenticated["disk"],
    }


def prepare_round0141(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0141 release SHA must be one full commit")
    _require_dedicated_run_environment()
    round_file = _require_issued_round()
    authenticated = _authenticate_real_inputs()
    if authenticated["disk"]["passed"] is not True:
        raise RuntimeError(
            "R0141 disk preflight failed: "
            f"{authenticated['disk']['free_bytes_observed']:,} free bytes < "
            f"{authenticated['disk']['required_free_bytes']:,} required"
        )
    layout = authenticated["layout"]
    reviews = authenticated["reviews"]
    inventory_signature = authenticated["inventory_manifest"]
    r0114_signature = authenticated["r0114_model_prompt_manifest"]
    environment_freeze = authenticated["environment_freeze"]
    model_inputs = [
        _signature_only(member) for member in authenticated["model_members"]
    ]
    round_signature = expected_input_signature(round_file)

    queue_root = create_fresh_directory(
        queue_root, label="R0141 multilingual prompted production queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    namespace = ensure_data_directory(
        os.path.join(artifacts, OUTPUT_NAMESPACE),
        label="R0141 prompted multilingual output namespace",
    )
    common_bindings = [
        _binding("round", round_signature),
        _binding("review-0087", reviews["0087"]),
        _binding("review-0114", reviews["0114"]),
        _binding("inventory", inventory_signature),
        _binding("model-prompt-manifest", r0114_signature),
        *[_binding("model-member", item) for item in model_inputs],
    ]
    jobs: list[dict[str, Any]] = []
    receipt_paths: list[str] = []
    for tranche in LANGUAGE_TRANCHES:
        node_id = str(tranche["node_id"])
        source = source_for_node(layout, node_id)
        bindings = [
            *common_bindings,
            _binding("source-parquet", source["text"]),
            _binding("raw-embedding", source["accepted_raw_embedding"]),
        ]
        output = os.path.join(namespace, node_id)
        receipt_path = os.path.join(output, "node-receipt.json")
        receipt_paths.append(receipt_path)
        jobs.append(
            {
                "id": node_id,
                "action": "embed_document_rows",
                "handler_module": "experiments.round0141_nodes",
                "handler_callable": "run_job",
                "deps": [],
                "outputs": [output],
                "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
                "expected_inputs": _dedupe(
                    [dict(item["signature"]) for item in bindings]
                ),
                "p90_wall_s": node_p90_seconds(),
                "language": tranche["language"],
                "dataset": tranche["dataset"],
                "dataset_row_range": [0, ROWS_PER_LANGUAGE],
                "corpus_global_row_range": tranche[
                    "corpus_global_row_range"
                ],
                "r0087_global_row_range": tranche["r0087_global_row_range"],
                "authenticated_source": source,
                "authenticated_boundary_inputs": bindings,
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

    final_id = "finalize_prompted_multilingual_tranche"
    final_bindings = [
        *common_bindings,
        *[_binding("source-parquet", item["text"]) for item in layout],
        *[
            _binding("raw-embedding", item["accepted_raw_embedding"])
            for item in layout
        ],
    ]
    final_output = os.path.join(artifacts, CAPABILITY)
    jobs.append(
        {
            "id": final_id,
            "action": final_id,
            "handler_module": "experiments.round0141_nodes",
            "handler_callable": "run_job",
            "deps": [str(item["node_id"]) for item in LANGUAGE_TRANCHES],
            "outputs": [final_output],
            "done_marker": os.path.join(artifacts, f"{final_id}.done.json"),
            "expected_inputs": _dedupe(
                [dict(item["signature"]) for item in final_bindings]
            ),
            "p90_wall_s": 600.0,
            "node_receipts": receipt_paths,
            "canonical_source_layout": layout,
            "authenticated_boundary_inputs": final_bindings,
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
        gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue["schema"] = "round0141-canonical-prompted-multilingual-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-data-production-filler"
    queue["required_reviews"] = list(REVIEW_DEFAULTS)
    queue["ordering_dependencies"] = []
    queue["capability_dependencies"] = [
        "jina-diverse-25m-inventory-v1",
        "jina-fineweb-2m-dual-prompt-native8192-substrate-v2",
    ]
    queue["capabilities_produced"] = [CAPABILITY]
    queue["training_performed"] = False
    queue["output_namespace"] = namespace
    queue["scientific_contract"] = {
        "purpose": "canonical prompted multilingual embedding tranche only",
        "scheduling_role": "P2 GPU data-production filler",
        "languages": [str(item["language"]) for item in LANGUAGE_TRANCHES],
        "corpus_rows": CORPUS_ROWS,
        "rows_per_language": ROWS_PER_LANGUAGE,
        "corpus_global_row_range": [0, CORPUS_ROWS],
        "r0087_global_row_range": [R0087_GLOBAL_START, R0087_GLOBAL_STOP],
        "new_gpu_rows": CORPUS_ROWS,
        "model": model_contract(),
        "batch_size": BATCH_SIZE,
        "adaptive_oom_floor_batch_size": 8,
        "atomic_chunk_rows_maximum": CHUNK_ROWS,
        "work": [
            {
                "node_id": item["node_id"],
                "language": item["language"],
                "dataset": item["dataset"],
                "dataset_row_range": [0, ROWS_PER_LANGUAGE],
                "corpus_global_row_range": item["corpus_global_row_range"],
                "r0087_global_row_range": item["r0087_global_row_range"],
            }
            for item in LANGUAGE_TRANCHES
        ],
        "source_manifest": source_manifest_summary(layout),
        "inventory_manifest": inventory_signature,
        "r0114_model_prompt_manifest": r0114_signature,
        "reviews": reviews,
        "environment_freeze": environment_freeze,
        "accepted_raw_embeddings_are_identity_evidence_not_inputs": True,
        "job_boundary_rehash_includes": [
            "issued-round",
            "accepted-reviews",
            "R0087-inventory",
            "R0114-model-prompt-manifest",
            "all-model-members",
            "consumed-source-parquet",
            "accepted-raw-embedding",
            "environment-freeze",
        ],
        "performance_calibration": {
            "reference": "R0116 accepted and live R0120 native8192 receipts",
            "live_r0120_document_rows_per_s": 266.0,
            "expected_document_rows_per_s": EXPECTED_ROWS_PER_S,
            "expected_gpu_hours": expected_gpu_seconds() / 3_600.0,
            "p90_gpu_hours": (
                len(LANGUAGE_TRANCHES) * node_p90_seconds() / 3_600.0
            ),
            "p90_budget_document_rows_per_s": 190.0,
            "minimum_document_rows_per_s": EMBED_MINIMUM_ROWS_PER_S,
            "warning_document_rows_per_s": EMBED_WARNING_ROWS_PER_S,
            "guard_evaluated_after_rows": PERFORMANCE_GUARD_ROWS,
            "worst_passing_gpu_hours": worst_passing_gpu_seconds() / 3_600.0,
            "hard_cap_gpu_hours": GPU_HOURS_CAP,
            "throughput_floor_fits_cap": (
                worst_passing_gpu_seconds() < GPU_HOURS_CAP * 3_600.0
            ),
            "token_length_calibration": TOKEN_LENGTH_CALIBRATION,
        },
        "disk_preflight": authenticated["disk"],
        "prompt_prefix": PROMPT_PREFIX,
        "source_order": [str(item["language"]) for item in LANGUAGE_TRANCHES],
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
    queue["p90_gpu_seconds"] = {**gpu_p90, "total": sum(gpu_p90.values())}
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha")
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    parser.add_argument("--rehearse-only", action="store_true")
    args = parser.parse_args(argv)
    if args.rehearse_only:
        value = {"rehearsal": rehearse_round0141_inputs()}
    else:
        if not args.release_sha:
            parser.error("required for queue preparation: --release-sha")
        value = {
            "queue_manifest": prepare_round0141(
                release_sha=args.release_sha,
                queue_root=args.queue_root,
            )
        }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
