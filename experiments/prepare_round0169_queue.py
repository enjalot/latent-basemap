#!/usr/bin/env python3
"""Prepare, but never launch, the conditional R0169 prompted-diverse U12 rung."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0105_search import GROUPS
from basemap.round0108_evaluation import IN_MIX_LANGUAGES, POLISH
from basemap.round0112_prompt_substrate import model_member_signatures
from basemap.round0116_prompted_corpus import environment_freeze_receipt
from basemap.round0160_prompted_seed_family import CAPABILITY as FAMILY_CAPABILITY
from basemap.round0161_prompted_gate_registration import CAPABILITY as GATE_CAPABILITY
from basemap.round0166_prompted_8m import CAPABILITY as Q2_CAPABILITY
from basemap.round0168_prompted_diverse_staging import (
    CAPABILITY as STAGING_CAPABILITY,
    MANIFEST_SCHEMA as STAGING_SCHEMA,
)
from basemap.round0169_prompted_diverse import (
    CAPABILITY,
    DIMENSION,
    GRAPH_K,
    GRAPH_MEAN_RECALL_FLOOR,
    GRAPH_NLIST,
    GRAPH_NPROBE,
    GRAPH_P10_RECALL_FLOOR,
    GRAPH_VECTOR_STORAGE,
    ROWS,
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    diverse_train_config,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0167_queue import _canary_inputs


ROUND_ROOT = "/data/latent-basemap/runs/round-0169"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0169-2026-08-03.md")
STAGING_MANIFEST = (
    "/data/latent-basemap/runs/round-0168/queue/artifacts/"
    "prompted-diverse-u12/prompted-u12-manifest.json"
)
Q2_EVALUATION = (
    "/data/latent-basemap/runs/round-0166/queue/artifacts/"
    f"{Q2_CAPABILITY}/scale-evaluation.json"
)
FAMILY_PATH = (
    "/data/latent-basemap/runs/round-0160/queue/artifacts/"
    "jina-fineweb-2m-prompted-seed42-45-family-v1/prompted-seed-family.json"
)
GATES_PATH = (
    "/data/latent-basemap/runs/round-0161/queue/artifacts/"
    "jina-prompted-universe-quality-gates-v1/prompted-quality-gates.json"
)
SELECTION_PATH = (
    "/data/latent-basemap/runs/round-0108/queue/inputs/"
    "registered-selections.npz"
)
GROUP_IDS_PATH = (
    "/data/latent-basemap/runs/round-0132/queue/artifacts/half-subset/"
    "compact-group-ids.u8.npy"
)
RAW_R0132_OOD = (
    "/data/latent-basemap/runs/round-0132/queue/artifacts/matched-ood/"
    "matched-ood.json"
)
LANGUAGES = (*IN_MIX_LANGUAGES, POLISH)
GPU_HOURS_MAXIMUM = 8.0


def _read_sealed(signature: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = prompt_contract.verify_signature(dict(signature), label=label)
    return prompt_contract.read_sealed(path, label=label)


def _accepted_bundle(
    round_id: str, *, review_path: str | None = None
) -> list[dict[str, Any]]:
    if review_path is None:
        candidates = [
            path
            for path in sorted(
                glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))
            )
            if _frontmatter(path).get("status") == "accepted"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"R0169 requires one accepted Review {round_id}; found {len(candidates)}"
            )
        review_path = candidates[0]
    review = expected_input_signature(review_path)
    frontmatter = _frontmatter(review_path)
    if (
        frontmatter.get("round_id") != round_id
        or frontmatter.get("status") != "accepted"
    ):
        raise RuntimeError(f"R0169 Review {round_id} is not accepted")
    round_path = os.path.join(LAB_ROOT, str(frontmatter.get("round") or ""))
    result_path = os.path.join(LAB_ROOT, str(frontmatter.get("result") or ""))
    issued = expected_input_signature(round_path)
    result = expected_input_signature(result_path)
    if (
        issued["sha256"] != frontmatter.get("round_sha256")
        or result["sha256"] != frontmatter.get("result_sha256")
    ):
        raise RuntimeError(f"R0169 Review {round_id} document binding changed")
    return [issued, result, review]


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if frontmatter.get("status") != "issued" or not descendant:
        raise RuntimeError("R0169 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def _language_sources(selection_signature: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    sources = {
        language: expected_input_signature(
            f"/data/chunks/fineweb2-{language}-chunked-500/train/000_00000.parquet"
        )
        for language in LANGUAGES
    }
    with np.load(selection_signature["canonical_path"], allow_pickle=False) as archive:
        expected = {
            f"{language}__{suffix}"
            for language in LANGUAGES
            for suffix in (
                "corpus",
                "queries",
                "original_queries",
                "query_replacement_mask",
            )
        }
        if not expected.issubset(set(archive.files)):
            raise RuntimeError("R0169 accepted R0108 language selectors are incomplete")
        for language, signature in sources.items():
            rows = int(pq.ParquetFile(signature["canonical_path"]).metadata.num_rows)
            corpus = np.asarray(archive[f"{language}__corpus"], dtype=np.int64)
            queries = np.asarray(archive[f"{language}__queries"], dtype=np.int64)
            if (
                corpus.shape != (49_500,)
                or queries.shape != (500,)
                or np.any(corpus < 0)
                or np.any(queries < 0)
                or int(max(corpus.max(), queries.max())) >= rows
            ):
                raise RuntimeError(f"R0169 {language} text/selector mapping changed")
    return sources


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0169 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0169_prompted_diverse.py",
        "tests/test_round0166_cpu_smoke.py",
        "tests/test_panel_v2.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0169-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
        "path_exercised": (
            "Q3 config/decision/dispatch and exact scale identity plus reused "
            "train -> seal -> checkpoint reload -> panel path"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0169 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _config_smoke() -> dict[str, Any]:
    config, digest = diverse_train_config(
        graph_signature={"canonical_path": "/future/graph.npz", "sha256": "a" * 64},
        graph_manifest_signature={
            "canonical_path": "/future/graph-manifest.json",
            "sha256": "b" * 64,
        },
        graph_edges=ROWS * 50,
        retained_rows=ROWS,
    )
    stamp = config["execution"]["expected_pipeline_stamp"]
    if (
        config["paired_invariant"]["rows"] != ROWS
        or config["paired_invariant"]["graph_vector_storage"] != GRAPH_VECTOR_STORAGE
        or config["optimizer"]["successful_positive_lr_updates"] != SUCCESSFUL_UPDATES
        or config["graph"]["k"] != GRAPH_K
        or stamp["compact_retained_rows"] != ROWS
    ):
        raise RuntimeError("R0169 config smoke changed")
    return prompt_contract.seal({
        "schema": "round0169-config-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "config_sha256": digest,
        "rows": ROWS,
        "successful_updates": SUCCESSFUL_UPDATES,
        "graph_k": GRAPH_K,
        "graph_vector_storage": GRAPH_VECTOR_STORAGE,
        "expected_pipeline_stamp": stamp,
    })


def prepare_round0169(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0169 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    dependency_inputs: list[dict[str, Any]] = []
    for dependency in ("0108", "0114", "0132", "0160", "0161", "0166"):
        dependency_inputs.extend(_accepted_bundle(dependency))
    dependency_inputs.extend(_accepted_bundle(
        "0168",
        review_path=os.path.join(LAB_ROOT, "review-0168-2026-08-03-01.md"),
    ))

    staging_signature = expected_input_signature(STAGING_MANIFEST)
    staging = _read_sealed(staging_signature, label="accepted R0168 staging")
    if (
        staging.get("schema") != STAGING_SCHEMA
        or staging.get("round_id") != "0168"
        or staging.get("capability") != STAGING_CAPABILITY
        or int(staging.get("rows", -1)) != ROWS
        or int(staging.get("dimension", -1)) != DIMENSION
        or (staging.get("population") or {}).get("polish_held_out") is not True
    ):
        raise RuntimeError("R0169 accepted staging contract changed")
    staging_inputs = [
        staging_signature,
        dict(staging["host_fp16"]),
        dict(staging["population"]["mapping"]),
        dict(staging["duplicate_control"]["arrays"]),
    ]

    q2_signature = expected_input_signature(Q2_EVALUATION)
    q2 = _read_sealed(q2_signature, label="accepted positive R0166 evaluation")
    if (
        q2.get("round_id") != "0166"
        or (q2.get("decision") or {}).get("passed") is not True
        or (q2.get("decision") or {}).get("outcome")
        != "prompted-english-8m-scale-rung-qualified"
        or q2.get("capabilities") != [Q2_CAPABILITY]
    ):
        raise RuntimeError("R0169 is blocked because Q2 did not qualify")

    family_signature = expected_input_signature(FAMILY_PATH)
    family = _read_sealed(family_signature, label="accepted R0160 family")
    gate_signature = expected_input_signature(GATES_PATH)
    gates = _read_sealed(gate_signature, label="accepted R0161 gates")
    if (
        family.get("capability") != FAMILY_CAPABILITY
        or gates.get("capability") != GATE_CAPABILITY
        or gates.get("registered") is not True
        or gates.get("family_evidence") != family_signature
    ):
        raise RuntimeError("R0169 prompted family/gate lineage changed")
    seed42 = family["cells"]["seed42"]
    accepted_score = _read_sealed(seed42["native_score"], label="accepted seed42 score")
    accepted_query = _read_sealed(
        accepted_score["query_reserve"], label="accepted R0113 query reserve"
    )
    accepted_selection = _read_sealed(
        accepted_score["query_selection"], label="accepted seed42 query selection"
    )
    matched_inputs = [
        family_signature,
        gate_signature,
        dict(family["lineage"]["assembly"]),
        dict(family["lineage"]["document_compact"]),
        dict(family["shared_prompted_reference"]),
        *[dict(value) for value in family["centroids"].values()],
        dict(seed42["native_score"]),
        dict(accepted_score["train_receipt"]),
        dict(accepted_score["combined_query_truth"]),
        dict(accepted_score["query_reserve"]),
        dict(accepted_query["outputs"]["document"]),
        dict(accepted_score["query_selection"]),
        dict(accepted_selection["positions"]),
    ]

    selection_signature = expected_input_signature(SELECTION_PATH)
    language_sources = _language_sources(selection_signature)
    group_signature = expected_input_signature(GROUP_IDS_PATH)
    group_ids = np.load(GROUP_IDS_PATH, mmap_mode="r", allow_pickle=False)
    if (
        group_ids.shape != (ROWS,)
        or group_ids.dtype != np.uint8
        or set(np.unique(group_ids).tolist()) != set(range(len(GROUPS)))
    ):
        raise RuntimeError("R0169 accepted R0132 group IDs changed")
    raw_ood_signature = expected_input_signature(RAW_R0132_OOD)
    raw_ood = _read_sealed(raw_ood_signature, label="accepted R0132 OOD")
    if (
        raw_ood.get("schema") != "round0132-matched-ood-scale-panel-v1"
        or raw_ood.get("round_id") != "0132"
        or set(raw_ood.get("control_12p5m") or {})
        != {
            "fineweb_recall_at_50_of_high10",
            "in_mix_median_recall_at_50_of_high10",
            "polish_recall_at_50_of_high10",
        }
    ):
        raise RuntimeError("R0169 accepted R0132 OOD summary changed")

    model_members = model_member_signatures()
    environment = environment_freeze_receipt()
    canary = _canary_inputs()
    queue_root = create_fresh_directory(queue_root, label="R0169 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )
    config_path = os.path.join(preflight, "config-smoke.json")
    atomic_write_new_json(config_path, _config_smoke(), immutable=True)
    smoke_inputs = [
        expected_input_signature(smoke_path),
        expected_input_signature(config_path),
    ]
    protocol_inputs = _dedupe([
        round_signature,
        *dependency_inputs,
        q2_signature,
        *smoke_inputs,
    ])
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    canary_output = os.path.join(artifacts, "prompt-model-canary")
    jobs: list[dict[str, Any]] = [{
        "id": "prompt_model_canary",
        "action": "prompt_canary",
        "handler_module": "experiments.round0169_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [canary_output],
        "done_marker": os.path.join(artifacts, "prompt-model-canary.done.json"),
        "expected_inputs": _dedupe([
            *protocol_inputs,
            *model_members,
            canary["text"],
            canary["document"],
        ]),
        "p90_wall_s": 180.0,
        "canary_text": canary["text"],
        "canary_document": canary["document"],
        "canary_positions": canary["positions"],
        "model_members": model_members,
        "environment_freeze": environment,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }]

    language_outputs: dict[str, str] = {}
    embed_ids: list[str] = []
    for language in LANGUAGES:
        node_id = f"embed_prompted_{language}"
        output = os.path.join(artifacts, f"prompted-{language}")
        embed_ids.append(node_id)
        language_outputs[language] = output
        jobs.append({
            "id": node_id,
            "action": "embed_language_probe",
            "handler_module": "experiments.round0169_nodes",
            "handler_callable": "run_job",
            "deps": ["prompt_model_canary"],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
            "expected_inputs": _dedupe([
                *protocol_inputs,
                *model_members,
                selection_signature,
                language_sources[language],
            ]),
            "p90_wall_s": 300.0,
            "language": language,
            "selection": selection_signature,
            "text_source": language_sources[language],
            "canary_output": canary_output,
            "model_members": model_members,
            "environment_freeze": environment,
            "node_policy": {"gpu_required": True, "training_performed": False},
        })

    audit_output = os.path.join(artifacts, "prompted-ood-training-disjoint")
    jobs.append({
        "id": "audit_prompted_ood_training_disjoint",
        "action": "audit_probe_training_disjoint",
        "handler_module": "experiments.round0169_nodes",
        "handler_callable": "run_job",
        "deps": embed_ids,
        "outputs": [audit_output],
        "done_marker": os.path.join(artifacts, "ood-training-audit.done.json"),
        "expected_inputs": _dedupe([*protocol_inputs, *staging_inputs]),
        "p90_wall_s": 1_800.0,
        "staging_manifest": staging_signature,
        "language_outputs": language_outputs,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
    })

    graph_output = os.path.join(artifacts, "fuzzy-k50-graph-and-reference")
    graph_manifest = os.path.join(graph_output, "graph-manifest.json")
    jobs.append({
        "id": "build_graph_and_reference",
        "action": "build_graph_and_reference",
        "handler_module": "experiments.round0169_nodes",
        "handler_callable": "run_job",
        "deps": ["audit_prompted_ood_training_disjoint"],
        "outputs": [graph_output],
        "done_marker": os.path.join(artifacts, "graph-reference.done.json"),
        "expected_inputs": _dedupe([*protocol_inputs, *staging_inputs]),
        "p90_wall_s": 9_000.0,
        "staging_manifest": staging_signature,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": True,
        },
    })

    train_output = os.path.join(artifacts, "seed42-train")
    jobs.append({
        "id": "train_prompted_diverse_u12",
        "action": "train_prompted_diverse_u12",
        "handler_module": "experiments.round0169_nodes",
        "handler_callable": "run_job",
        "deps": ["build_graph_and_reference"],
        "outputs": [train_output],
        "done_marker": os.path.join(artifacts, "train.done.json"),
        "expected_inputs": _dedupe([*protocol_inputs, *staging_inputs]),
        "p90_wall_s": 6_000.0,
        "staging_manifest": staging_signature,
        "graph_manifest": graph_manifest,
        "node_policy": {"gpu_required": True, "training_performed": True},
    })

    evaluation_output = os.path.join(artifacts, CAPABILITY)
    jobs.append({
        "id": "evaluate_prompted_diverse_u12",
        "action": "evaluate_prompted_diverse_u12",
        "handler_module": "experiments.round0169_nodes",
        "handler_callable": "run_job",
        "deps": ["train_prompted_diverse_u12", *embed_ids],
        "outputs": [evaluation_output],
        "done_marker": os.path.join(artifacts, "evaluation.done.json"),
        "expected_inputs": _dedupe([
            *protocol_inputs,
            *staging_inputs,
            *matched_inputs,
            selection_signature,
            group_signature,
            raw_ood_signature,
            *language_sources.values(),
        ]),
        "p90_wall_s": 6_600.0,
        "staging_manifest": staging_signature,
        "graph_manifest": graph_manifest,
        "train_output": train_output,
        "family_evidence": family_signature,
        "gate_registration": gate_signature,
        "q2_evaluation": q2_signature,
        "group_ids": group_signature,
        "raw_r0132_ood": raw_ood_signature,
        "ood_audit_output": audit_output,
        "language_outputs": language_outputs,
        "node_policy": {"gpu_required": True, "training_performed": False},
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
        "schema": "round0169-prompted-diverse-u12-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0108", "0114", "0132", "0160", "0161", "0166", "0168"],
        "ordering_dependencies": ["0166"],
        "capability_dependencies": [
            Q2_CAPABILITY,
            STAGING_CAPABILITY,
            FAMILY_CAPABILITY,
            GATE_CAPABILITY,
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
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
            "question": "does prompted scale quality survive the R0132 U12 multilingual mixture?",
            "only_q2_to_q3_treatment": "exact population changes from 8M English to R0132 U12 diverse mixture",
            "population_rows": ROWS,
            "population_law": "exact accepted R0132 U12 compact order; Polish held out",
            "embedding_convention": "literal Document: prefix",
            "duplicate_policy": "diagnostic metadata only; do not alter U12",
            "graph": {
                "k": GRAPH_K,
                "nlist": GRAPH_NLIST,
                "nprobe": GRAPH_NPROBE,
                "mean_recall_floor": GRAPH_MEAN_RECALL_FLOOR,
                "p10_recall_floor": GRAPH_P10_RECALL_FLOOR,
                "vector_storage": GRAPH_VECTOR_STORAGE,
                "vector_storage_change_role": "registered capacity representation; fp32 vectors exceed the single 32GiB device",
            },
            "training": {
                "seed": 42,
                "successful_positive_lr_updates": SUCCESSFUL_UPDATES,
                "same_recipe_graph_law_and_fixed_dose_as_q2": True,
            },
            "ood": {
                "selectors": "exact accepted R0108 selectors",
                "prompted_rows": len(LANGUAGES) * 50_000,
                "all_rows_must_be_exact_prompted_family_disjoint_from_training_before_graph": True,
                "post_embedding_replacements": False,
                "polish_role": "held-out headline OOD",
                "projection_ffr": "diagnostic-only",
            },
            "native_prompted_floor_stack": True,
            "matched_2m_minimum_ratio": 0.97,
            "language_ffr_minimum_ratio_to_pooled_english": 0.40,
            "polish_recall50_minimum_ratio_to_in_mix_median": 0.50,
            "prompted_ood_minimum_ratio_to_raw_r0132": 0.97,
            "negative_outcome_releases_no_map_capability": True,
            "release_cpu_smoke": smoke_inputs[0],
            "config_cpu_smoke": smoke_inputs[1],
        },
    })
    p90_total = float(queue["p90_gpu_seconds"]["total"])
    if p90_total > GPU_HOURS_MAXIMUM * 3_600:
        raise RuntimeError("R0169 registered p90 exceeds the eight-GPU-hour cap")
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0169(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
