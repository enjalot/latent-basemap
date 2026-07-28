#!/usr/bin/env python3
"""Prepare the balanced-150M IVF32768 build and qualification queue."""
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
from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0086_program import validate_substrate
from basemap.round0096_larger_nlist import (
    GLOBAL_MEAN_FLOOR,
    NLIST,
    PER_CORPUS_MEAN_FLOOR,
    POLICY_GRID,
    QUALITY_ROWS,
    QUALITY_SAMPLE_SHA256,
    QUALITY_SEED,
    ROUND_ID,
    TRAIN_ROWS,
    TRAIN_SAMPLE_SHA256,
    TRAIN_SEED,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0096"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0096-*.md")
INITIAL_RELEASE_SHA = "6aae68eba1601037b24ce02415d5807fc56919af"


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4_096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_issued_round() -> str:
    candidates = [
        path for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter_status(path) == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0096 requires one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    if _frontmatter_status(path) not in {"accepted", "partial"}:
        raise RuntimeError("R0095 review does not release a successor")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError("R0095 review bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    required = (
        "minilm-balanced-150m-unbiased-search-audit-v1",
        "larger-nlist qualification",
        "FineWeb recall remains below `0.84`",
    )
    if any(value not in text for value in required):
        raise RuntimeError("R0095 review does not support R0096")
    return signature


def _require_r0095_evidence(
    *,
    audit_path: str,
    audit_sha256: str,
    decision_path: str,
    decision_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = expected_input_signature(audit_path)
    decision = expected_input_signature(decision_path)
    if audit["sha256"] != audit_sha256 or decision["sha256"] != decision_sha256:
        raise RuntimeError("R0095 evidence bytes changed")
    with open(audit["canonical_path"], encoding="utf-8") as handle:
        audit_value = json.load(handle)
    with open(decision["canonical_path"], encoding="utf-8") as handle:
        decision_value = json.load(handle)
    if (
        audit_value.get("schema")
        != "round0095-balanced-150m-unbiased-search-audit-v1"
        or audit_value.get("validity_passed") is not True
        or decision_value.get("schema")
        != "round0095-balanced-150m-search-correction-decision-v1"
        or decision_value.get("audit") != audit
        or decision_value.get("larger_nlist_qualification_is_next")
        is not True
        or decision_value.get("graph_build_remains_blocked") is not True
    ):
        raise RuntimeError("R0095 does not support larger-nlist qualification")
    return audit, decision


def _require_prior_attempt(
    prior_queue_root: str,
) -> dict[str, Any]:
    root = os.path.realpath(prior_queue_root)
    expected_root = os.path.join(ROUND_ROOT, "queue")
    if root != expected_root:
        raise RuntimeError(
            f"R0096 retry must reuse exact Attempt 1 root {expected_root}"
        )
    manifest = expected_input_signature(os.path.join(root, "queue.json"))
    terminal = expected_input_signature(
        os.path.join(root, "runner-terminal.json")
    )
    with open(terminal["canonical_path"], encoding="utf-8") as handle:
        terminal_value = json.load(handle)
    expected_completed = [
        "train_larger_index_template",
        "build_larger_index_fineweb",
        "build_larger_index_redpajama",
        "build_larger_index_pile",
        "assemble_larger_index",
    ]
    if (
        terminal_value.get("schema") != "slim-runner-terminal-v3"
        or terminal_value.get("round_id") != ROUND_ID
        or terminal_value.get("verdict") != "failed"
        or terminal_value.get("completed_jobs") != expected_completed
        or terminal_value.get("gpu_wall_accounting_complete") is not True
        or terminal_value.get("queue_manifest_sha256")
        != manifest["sha256"]
        or (
            (terminal_value.get("release_checkout") or {}).get("head")
            != INITIAL_RELEASE_SHA
        )
    ):
        raise RuntimeError("R0096 Attempt 1 terminal evidence changed")
    index_receipt = expected_input_signature(os.path.join(
        root, "artifacts", "larger-index-assembly", "index-receipt.json",
    ))
    with open(index_receipt["canonical_path"], encoding="utf-8") as handle:
        index_receipt_value = json.load(handle)
    body = {
        key: value
        for key, value in index_receipt_value.items()
        if key != "identity_sha256"
    }
    index = expected_input_signature(os.path.join(
        root,
        "artifacts",
        "larger-index-assembly",
        "balanced-150m-retained-ivf32768.ivfpq",
    ))
    if (
        index_receipt_value.get("schema")
        != "round0096-balanced-150m-ivf32768-index-v1"
        or index_receipt_value.get("round_id") != ROUND_ID
        or index_receipt_value.get("release_sha") != INITIAL_RELEASE_SHA
        or index_receipt_value.get("index") != index
        or index_receipt_value.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
    ):
        raise RuntimeError("R0096 Attempt 1 assembled-index evidence changed")
    return {
        "queue_manifest": manifest,
        "terminal": terminal,
        "terminal_value": terminal_value,
        "index_receipt": index_receipt,
        "index": index,
    }


def _job(
    *,
    node_id: str,
    action: str,
    deps: list[str],
    output: str,
    p90_wall_s: float,
    inputs: list[dict[str, Any]],
    gpu: bool,
    **values: Any,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "action": action,
        "handler_module": "experiments.round0096_nodes",
        "handler_callable": "run_job",
        "deps": deps,
        "outputs": [output],
        "done_marker": os.path.join(
            os.path.dirname(output), f"{node_id}.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": p90_wall_s,
        "node_policy": {
            "gpu_required": gpu,
            "training_performed": False,
        },
        **values,
    }


def prepare_round0096(
    *,
    release_sha: str,
    r0095_review_path: str,
    r0095_review_sha256: str,
    r0095_audit_path: str,
    r0095_audit_sha256: str,
    r0095_decision_path: str,
    r0095_decision_sha256: str,
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
    runtime_spec_path: str,
    runtime_spec_sha256: str,
    prior_queue_root: str | None = None,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    round_file = _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0096 release SHA must be one full commit")
    review = _require_review(
        r0095_review_path, expected_sha256=r0095_review_sha256,
    )
    audit, decision = _require_r0095_evidence(
        audit_path=r0095_audit_path,
        audit_sha256=r0095_audit_sha256,
        decision_path=r0095_decision_path,
        decision_sha256=r0095_decision_sha256,
    )
    substrate = validate_substrate(
        substrate_manifest_path,
        expected_sha256=substrate_manifest_sha256,
    )
    runtime = expected_input_signature(runtime_spec_path)
    if runtime["sha256"] != runtime_spec_sha256:
        raise RuntimeError("R0096 runtime specification changed")
    prior = (
        _require_prior_attempt(prior_queue_root)
        if prior_queue_root is not None
        else None
    )

    queue_root = create_fresh_directory(
        queue_root, label="Round 0096 larger-nlist queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    template_output = os.path.join(artifacts, "larger-index-template")
    template_index = os.path.join(
        template_output, "ivf32768-pq48x8-template.ivfpq",
    )
    template_receipt = os.path.join(
        template_output, "template-receipt.json",
    )
    shard_outputs = {
        corpus: os.path.join(artifacts, f"larger-index-{corpus}")
        for corpus in ("fineweb", "redpajama", "pile")
    }
    shard_indexes = {
        corpus: os.path.join(output, f"{corpus}.ivfpq")
        for corpus, output in shard_outputs.items()
    }
    shard_receipts = {
        corpus: os.path.join(output, "shard-receipt.json")
        for corpus, output in shard_outputs.items()
    }
    assembly_output = os.path.join(artifacts, "larger-index-assembly")
    assembled_index = os.path.join(
        assembly_output, "balanced-150m-retained-ivf32768.ivfpq",
    )
    assembled_receipt = os.path.join(
        assembly_output, "index-receipt.json",
    )
    qualification_output = os.path.join(
        artifacts, "larger-index-qualification",
    )
    inputs = _dedupe([
        expected_input_signature(round_file),
        review,
        audit,
        decision,
        substrate["signature"],
        runtime,
        *(
            [
                prior["queue_manifest"],
                prior["terminal"],
                prior["index_receipt"],
                prior["index"],
            ]
            if prior is not None
            else []
        ),
    ])
    external = {
        "substrate_manifest": substrate_manifest_path,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "runtime_spec": runtime_spec_path,
        "runtime_spec_sha256": runtime_spec_sha256,
        "r0095_review": r0095_review_path,
        "r0095_review_sha256": r0095_review_sha256,
    }
    if prior is None:
        jobs = [
            _job(
                node_id="train_larger_index_template",
                action="train_larger_index_template",
                deps=[],
                output=template_output,
                p90_wall_s=10_800.0,
                inputs=inputs,
                gpu=True,
                **external,
            ),
        ]
        for corpus in ("fineweb", "redpajama", "pile"):
            jobs.append(_job(
                node_id=f"build_larger_index_{corpus}",
                action="build_larger_index_shard",
                deps=["train_larger_index_template"],
                output=shard_outputs[corpus],
                p90_wall_s=3_600.0,
                inputs=inputs,
                gpu=True,
                corpus=corpus,
                template_index=template_index,
                template_receipt=template_receipt,
                **external,
            ))
        jobs.append(_job(
            node_id="assemble_larger_index",
            action="assemble_larger_index",
            deps=[
                "build_larger_index_fineweb",
                "build_larger_index_redpajama",
                "build_larger_index_pile",
            ],
            output=assembly_output,
            p90_wall_s=1_800.0,
            inputs=inputs,
            gpu=False,
            template_index=template_index,
            template_receipt=template_receipt,
            **{
                f"{corpus}_{kind}": value
                for corpus in ("fineweb", "redpajama", "pile")
                for kind, value in (
                    ("index", shard_indexes[corpus]),
                    ("receipt", shard_receipts[corpus]),
                )
            },
            **external,
        ))
        qualification_deps = ["assemble_larger_index"]
        qualification_index = assembled_index
        qualification_receipt = assembled_receipt
        prior_fields: dict[str, Any] = {}
    else:
        jobs = []
        qualification_deps = []
        qualification_index = prior["index"]["canonical_path"]
        qualification_receipt = prior["index_receipt"]["canonical_path"]
        prior_fields = {
            "index_release_sha": INITIAL_RELEASE_SHA,
            "prior_terminal": prior["terminal"]["canonical_path"],
            "prior_terminal_sha256": prior["terminal"]["sha256"],
        }
    jobs.append(_job(
        node_id="qualify_larger_index",
        action="qualify_larger_index",
        deps=qualification_deps,
        output=qualification_output,
        p90_wall_s=1_800.0,
        inputs=inputs,
        gpu=True,
        index=qualification_index,
        index_receipt=qualification_receipt,
        r0095_audit=r0095_audit_path,
        r0095_audit_sha256=r0095_audit_sha256,
        r0095_decision=r0095_decision_path,
        r0095_decision_sha256=r0095_decision_sha256,
        **prior_fields,
        **external,
    ))

    prior_gpu_seconds = (
        float(prior["terminal_value"]["gpu_wall_s"])
        if prior is not None
        else 0.0
    )
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=max(0.1, 8.0 - prior_gpu_seconds / 3_600.0),
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = (
        "round0096-balanced-150m-ivf32768-retry-queue-v1"
        if prior is not None
        else "round0096-balanced-150m-ivf32768-queue-v1"
    )
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0095"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-150m-unbiased-search-audit-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-150m-ivf32768-search-qualified-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "index_geometry": {
            "nlist": NLIST,
            "pq_m": 48,
            "pq_bits": 8,
            "metric": "inner-product over unit-normalized int8 directions",
        },
        "index_training": {
            "rows": TRAIN_ROWS,
            "seed": TRAIN_SEED,
            "sample_sha256": TRAIN_SAMPLE_SHA256,
            "semantics": "uniform retained rows; subset before final sort",
        },
        "quality": {
            "rows": QUALITY_ROWS,
            "seed": QUALITY_SEED,
            "sample_sha256": QUALITY_SAMPLE_SHA256,
            "global_mean_recall_floor": GLOBAL_MEAN_FLOOR,
            "every_corpus_mean_recall_floor": PER_CORPUS_MEAN_FLOOR,
            "policy_grid": [
                {"nprobe": nprobe, "shortlist_width": width}
                for nprobe, width in POLICY_GRID
            ],
        },
        "performance_selector": (
            "lowest median three-repeat complete search-plus-exact-rerank "
            "seconds/query among dual-floor passing cells"
        ),
        "performance_benchmark": {
            "rows": 8_192,
            "seed": 97,
            "warmup_rows": 512,
            "repeats": 3,
        },
        "no_graph": True,
        "no_map_training": True,
        "no_scale_decision": True,
        "setup_correction": (
            {
                "prior_release_sha": INITIAL_RELEASE_SHA,
                "prior_queue_manifest": prior["queue_manifest"],
                "prior_terminal": prior["terminal"],
                "prior_gpu_wall_seconds": prior_gpu_seconds,
                "only_change": (
                    "maximum exact-rerank width 2048 -> 2047 so "
                    "width-plus-self request respects FAISS max-k 2048"
                ),
                "reuses_exact_assembled_index": prior["index"],
            }
            if prior is not None
            else None
        ),
    }
    manifest["jobs"] = jobs
    manifest["p90_gpu_seconds"] = (
        {
            "qualify_larger_index": 1_800.0,
            "total": 1_800.0,
        }
        if prior is not None
        else {
            "train_larger_index_template": 10_800.0,
            "build_larger_index_fineweb": 3_600.0,
            "build_larger_index_redpajama": 3_600.0,
            "build_larger_index_pile": 3_600.0,
            "qualify_larger_index": 1_800.0,
            "total": 23_400.0,
        }
    )
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0095-review", required=True)
    parser.add_argument("--r0095-review-sha256", required=True)
    parser.add_argument("--r0095-audit", required=True)
    parser.add_argument("--r0095-audit-sha256", required=True)
    parser.add_argument("--r0095-decision", required=True)
    parser.add_argument("--r0095-decision-sha256", required=True)
    parser.add_argument("--substrate-manifest", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    parser.add_argument("--runtime-spec", required=True)
    parser.add_argument("--runtime-spec-sha256", required=True)
    parser.add_argument("--prior-queue")
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    path = prepare_round0096(
        release_sha=args.release_sha,
        r0095_review_path=args.r0095_review,
        r0095_review_sha256=args.r0095_review_sha256,
        r0095_audit_path=args.r0095_audit,
        r0095_audit_sha256=args.r0095_audit_sha256,
        r0095_decision_path=args.r0095_decision,
        r0095_decision_sha256=args.r0095_decision_sha256,
        substrate_manifest_path=args.substrate_manifest,
        substrate_manifest_sha256=args.substrate_manifest_sha256,
        runtime_spec_path=args.runtime_spec,
        runtime_spec_sha256=args.runtime_spec_sha256,
        prior_queue_root=args.prior_queue,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
