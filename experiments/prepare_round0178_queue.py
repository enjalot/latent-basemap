#!/usr/bin/env python3
"""Prepare, but never launch, the R0178 prompted-universality recovery."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0112_prompt_substrate import model_member_signatures
from basemap.round0116_prompted_corpus import environment_freeze_receipt
from basemap.round0142_jina_universality import PROBE_ORDER
from basemap.round0178_prompted_universality import (
    CAPABILITY,
    PROMPTED_MAP_ORDER,
    ROUND_ID,
)
from experiments import prepare_round0167_queue as prior
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0178"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0178-2026-08-03.md")
CONTROL_TEXT = (
    "/data/chunks/fineweb-edu-sample-10BT-chunked-500/heldout/"
    "data-00090-of-00099.parquet"
)
R0177_ROOT = "/data/latent-basemap/runs/round-0177/queue/artifacts"
TRAINING_TEXT_HASHES = {
    "r0115-r0117-prompted-2m": (
        "/data/latent-basemap/runs/round-0113/queue/artifacts/"
        "compact-arrays/source-text-sha256-sorted.v32.npy"
    ),
    "r0171-prompted-8m": (
        "/data/latent-basemap/runs/round-0165/queue-correction-1/artifacts/"
        "prompted-english-8m-frozen-prefix/"
        "source-text-sha256-sorted.v32.npy"
    ),
}
TRAINING_ARRAYS = {
    "r0115-r0117-prompted-2m": (
        "/data/latent-basemap/runs/round-0113/queue/artifacts/"
        "compact-arrays/document-compact.f16",
        1_993_761,
    ),
    "r0171-prompted-8m": (
        "/data/latent-basemap/runs/round-0165/queue-correction-1/artifacts/"
        "prompted-english-8m-frozen-prefix/document-compact.f16",
        7_952_419,
    ),
}
MAPS = {
    "r0115-prompted-2m-seed42": (
        "/data/latent-basemap/runs/round-0115/queue-attempt-2/artifacts/"
        "document/train/model.pt"
    ),
    "r0117-prompted-2m-seed43": (
        "/data/latent-basemap/runs/round-0117/queue/artifacts/"
        "document/train/model.pt"
    ),
    "r0171-prompted-8m-seed42": (
        "/data/latent-basemap/runs/round-0171/queue/artifacts/"
        "seed42-train/model.pt"
    ),
}
GPU_HOURS_MINIMUM = 0.05
GPU_HOURS_EXPECTED = 0.20
GPU_HOURS_MAXIMUM = 1.50
HANDLER_MODULE = "experiments.round0178_nodes"


def _review_bundle(round_id: str) -> list[dict[str, Any]]:
    review = prior._one_document("review", round_id, status="accepted")
    frontmatter = _frontmatter(review["canonical_path"])
    result = expected_input_signature(
        os.path.join(LAB_ROOT, str(frontmatter.get("result") or ""))
    )
    issued = expected_input_signature(
        os.path.join(LAB_ROOT, str(frontmatter.get("round") or ""))
    )
    if (
        result["sha256"] != frontmatter.get("result_sha256")
        or issued["sha256"] != frontmatter.get("round_sha256")
    ):
        raise RuntimeError(f"Review {round_id} bindings changed")
    return [issued, result, review]


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        [
            "git",
            "-C",
            RELEASE_ROOT,
            "merge-base",
            "--is-ancestor",
            base_commit,
            release_sha,
        ],
        check=False,
        timeout=10,
    ).returncode == 0
    if frontmatter.get("status") != "issued" or not descendant:
        raise RuntimeError("R0178 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def _reused_probe_inputs() -> tuple[
    dict[str, str], list[dict[str, Any]]
]:
    outputs: dict[str, str] = {}
    signatures: list[dict[str, Any]] = []
    for name in PROBE_ORDER:
        output = os.path.join(R0177_ROOT, f"prompted-{name}")
        receipt_path = os.path.join(output, "receipt.json")
        with open(receipt_path, encoding="utf-8") as handle:
            receipt = json.load(handle)
        if (
            receipt.get("round_id") != "0177"
            or receipt.get("probe") != name
            or receipt.get("prompt_applied") is not True
        ):
            raise RuntimeError(f"R0177 {name} probe receipt changed")
        receipt_signature = expected_input_signature(receipt_path)
        payloads = [
            expected_input_signature(receipt[key]["canonical_path"])
            for key in (
                "corpus_embeddings",
                "query_embeddings",
                "corpus_source_rows",
                "query_source_rows",
            )
        ]
        for key, signature in zip(
            (
                "corpus_embeddings",
                "query_embeddings",
                "corpus_source_rows",
                "query_source_rows",
            ),
            payloads,
            strict=True,
        ):
            if signature != receipt[key]:
                raise RuntimeError(f"R0177 {name} {key} changed")
        outputs[name] = output
        signatures.extend((receipt_signature, *payloads))
    return outputs, _dedupe(signatures)


def prepare_round0178(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0178 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews: list[dict[str, Any]] = []
    for round_id in ("0115", "0117", "0142", "0146", "0171", "0177"):
        reviews.extend(_review_bundle(round_id))
    maps = {
        key: expected_input_signature(path) for key, path in MAPS.items()
    }
    if tuple(maps) != PROMPTED_MAP_ORDER:
        raise RuntimeError("R0178 prompted map order changed")
    training_text_hashes = {
        key: expected_input_signature(path)
        for key, path in TRAINING_TEXT_HASHES.items()
    }
    training_sources = {
        key: {
            "signature": expected_input_signature(path),
            "rows": int(rows),
        }
        for key, (path, rows) in TRAINING_ARRAYS.items()
    }
    for label, source in training_sources.items():
        if source["signature"]["bytes"] != source["rows"] * 768 * 2:
            raise RuntimeError(f"R0178 {label} training matrix size changed")
    sources = prior._source_specs()
    model_members = model_member_signatures()
    environment = environment_freeze_receipt()
    canary = prior._canary_inputs()
    raw_table = expected_input_signature(prior.R0142_TABLE)
    raw_predictors = expected_input_signature(prior.R0146_PREDICTORS)
    control_text = expected_input_signature(CONTROL_TEXT)
    probe_coordinates = {
        name: prior._coordinate_signature(name) for name in PROBE_ORDER
    }
    control_coordinates = {
        name: prior._coordinate_signature(name, control=True)
        for name in PROBE_ORDER
    }
    probe_outputs, reused_probe_signatures = _reused_probe_inputs()
    probe_sources = {
        name: {
            "probe": name,
            "r0142_coordinates": probe_coordinates[name],
            **sources[name],
        }
        for name in PROBE_ORDER
    }

    external_inputs = _dedupe([
        round_signature,
        *reviews,
        *maps.values(),
        *training_text_hashes.values(),
        *[source["signature"] for source in training_sources.values()],
        *reused_probe_signatures,
        *model_members,
        canary["text"],
        canary["document"],
        raw_table,
        raw_predictors,
        control_text,
        *probe_coordinates.values(),
        *control_coordinates.values(),
        *[
            signature
            for source in sources.values()
            for signature in source.values()
            if isinstance(signature, Mapping)
        ],
    ])

    queue_root = create_fresh_directory(
        queue_root, label="R0178 prompted-universality recovery queue"
    )
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path,
        prior._release_cpu_smoke(release_sha, maps),
        immutable=True,
    )
    smoke_signature = expected_input_signature(smoke_path)
    external_inputs = _dedupe([*external_inputs, smoke_signature])
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))

    canary_output = os.path.join(artifacts, "prompt-model-canary")
    selector_output = os.path.join(artifacts, "fineweb-control-selector")
    control_output = os.path.join(artifacts, "prompted-fineweb-control")
    masks_output = os.path.join(artifacts, "source-text-sensitivity-masks")
    audit_output = os.path.join(artifacts, "prompted-training-disjoint-audit")
    jobs: list[dict[str, Any]] = [
        {
            "id": "prompt_model_canary",
            "action": "prompt_canary",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [canary_output],
            "done_marker": os.path.join(
                artifacts, "prompt-model-canary.done.json"
            ),
            "expected_inputs": _dedupe([
                round_signature,
                *reviews,
                *model_members,
                canary["text"],
                canary["document"],
                smoke_signature,
            ]),
            "p90_wall_s": 180.0,
            "canary_text": canary["text"],
            "canary_document": canary["document"],
            "canary_positions": canary["positions"],
            "model_members": model_members,
            "environment_freeze": environment,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
        {
            "id": "select_disjoint_fineweb_control",
            "action": "select_disjoint_control",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [selector_output],
            "done_marker": os.path.join(
                artifacts, "fineweb-control-selector.done.json"
            ),
            "expected_inputs": _dedupe([
                round_signature,
                *reviews,
                control_text,
                *training_text_hashes.values(),
            ]),
            "p90_wall_s": 180.0,
            "text_source": control_text,
            "training_text_hashes": training_text_hashes,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
        },
        {
            "id": "embed_disjoint_fineweb_control",
            "action": "embed_disjoint_control",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": [
                "prompt_model_canary",
                "select_disjoint_fineweb_control",
            ],
            "outputs": [control_output],
            "done_marker": os.path.join(
                artifacts, "prompted-fineweb-control.done.json"
            ),
            "expected_inputs": _dedupe([
                round_signature,
                *reviews,
                *model_members,
                control_text,
                smoke_signature,
            ]),
            "p90_wall_s": 600.0,
            "text_source": control_text,
            "selector_output": selector_output,
            "canary_output": canary_output,
            "model_members": model_members,
            "environment_freeze": environment,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
        {
            "id": "seal_source_text_sensitivity_masks",
            "action": "seal_sensitivity_masks",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["embed_disjoint_fineweb_control"],
            "outputs": [masks_output],
            "done_marker": os.path.join(
                artifacts, "source-text-sensitivity-masks.done.json"
            ),
            "expected_inputs": external_inputs,
            "p90_wall_s": 600.0,
            "probe_outputs": probe_outputs,
            "probe_sources": probe_sources,
            "control_output": control_output,
            "control_coordinates": control_coordinates,
            "training_text_hashes": training_text_hashes,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": True,
            },
        },
        {
            "id": "audit_prompted_rows_against_map_training",
            "action": "audit_training_disjoint",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["seal_source_text_sensitivity_masks"],
            "outputs": [audit_output],
            "done_marker": os.path.join(
                artifacts, "prompted-training-disjoint-audit.done.json"
            ),
            "expected_inputs": external_inputs,
            "p90_wall_s": 1_800.0,
            "training_sources": training_sources,
            "probe_outputs": probe_outputs,
            "control_output": control_output,
            "control_coordinates": control_coordinates,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": True,
            },
        },
    ]

    map_outputs: dict[str, str] = {}
    score_ids: list[str] = []
    for map_key in PROMPTED_MAP_ORDER:
        job_id = f"score_{map_key}"
        output = os.path.join(artifacts, map_key)
        map_outputs[map_key] = output
        score_ids.append(job_id)
        jobs.append({
            "id": job_id,
            "action": "score_map",
            "handler_module": HANDLER_MODULE,
            "handler_callable": "run_job",
            "deps": ["audit_prompted_rows_against_map_training"],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{job_id}.done.json"),
            "expected_inputs": external_inputs,
            "p90_wall_s": 900.0,
            "map_key": map_key,
            "model": maps[map_key],
            "probe_outputs": probe_outputs,
            "control_output": control_output,
            "control_coordinates": control_coordinates,
            "sensitivity_masks_output": masks_output,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        })

    final_output = os.path.join(artifacts, CAPABILITY)
    jobs.append({
        "id": "assemble_prompted_universality",
        "action": "assemble",
        "handler_module": HANDLER_MODULE,
        "handler_callable": "run_job",
        "deps": score_ids,
        "outputs": [final_output],
        "done_marker": os.path.join(
            artifacts, "assemble-prompted-universality.done.json"
        ),
        "expected_inputs": external_inputs,
        "p90_wall_s": 1_800.0,
        "map_outputs": map_outputs,
        "probe_outputs": probe_outputs,
        "raw_retention_table": raw_table,
        "raw_predictors": raw_predictors,
        "training_disjoint_audit": audit_output,
        "sensitivity_masks_output": masks_output,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
    })

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0178-prompted-universality-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-evaluation",
        "required_reviews": [
            "0115",
            "0117",
            "0142",
            "0146",
            "0171",
            "0177",
        ],
        "ordering_dependencies": ["0177"],
        "capability_dependencies": [
            "jina-fineweb-2m-prompt-map-contrast-v1",
            "jina-fineweb-2m-prompt-map-seed43-contrast-v1",
            "jina-diverse-universality-panel-v1",
            "jina-diverse-projection-loss-predictors-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            **{
                str(job["id"]): float(job["p90_wall_s"])
                for job in jobs
                if job["node_policy"]["gpu_required"]
            },
            "total": sum(
                float(job["p90_wall_s"])
                for job in jobs
                if job["node_policy"]["gpu_required"]
            ),
        },
        "scientific_contract": {
            "question": (
                "prompted OOD retention and TwoNN association on a control "
                "universe disjoint from every scored map's training rows"
            ),
            "maps": list(PROMPTED_MAP_ORDER),
            "probes": list(PROBE_ORDER),
            "probe_selection": "exact accepted R0142 corpus/query rows",
            "probe_embeddings": (
                "reviewed successful R0177 probe nodes, exact receipt and "
                "payload hashes; no failed control or audit reuse"
            ),
            "control": (
                "first 60000 source-text-unique heldout-shard rows after "
                "rejecting both map-training source-text families"
            ),
            "training_overlap": (
                "source-text audit first; complete stored-fp16 audit second; "
                "query/control overlap blocks"
            ),
            "sensitivity": (
                "full panel primary plus paired union mask for exact source-text "
                "corpus copies; stored-fp16 copy masks diagnostic"
            ),
            "metrics": [
                "probe FFR",
                "control FFR",
                "FFR retention",
                "recall10 retention",
            ],
            "twonn": "R0146 exact 2048-row estimator in prompted geometry",
            "diagnostic_only": True,
            "no_causal_prompt_claim": True,
            "no_universal_map_claim": True,
            "no_quality_gate_change": True,
            "no_training": True,
            "release_cpu_smoke": smoke_signature,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0178(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
