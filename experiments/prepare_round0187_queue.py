#!/usr/bin/env python3
"""Prepare, but never launch, the R0187 composition-controlled ladder."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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
from basemap.round0187_composition_nested_ladder import (
    CAPABILITY,
    COMPOUND_RETENTION_RATIO,
    FULL_GRAPH_EDGES,
    FULL_SUCCESSFUL_UPDATES,
    HASH_NAMESPACE,
    POPULATION_CAPABILITY,
    PRIMARY_METRICS,
    RETENTION_RATIO,
    ROUND_ID,
    RUNG_COUNTS,
    RUNG_ROWS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    successful_updates_for_edges,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0187"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0187-2026-08-04.md")
HANDLER_MODULE = "experiments.round0187_nodes"
QUEUE_SCHEMA = "round0187-composition-controlled-nested-ladder-queue-v1"
GPU_HOURS_CAP = 7.5

SOURCE_POPULATION = (
    "/data/latent-basemap/runs/round-0165/queue-correction-1/artifacts/"
    "prompted-english-8m-frozen-prefix/frozen-prefix-population.json"
)
PILE_QUERY_RECEIPT = (
    "/data/latent-basemap/runs/round-0171/queue/artifacts/"
    "heldout-query-reserve/query-reserve.json"
)
FULL_TRAIN_RECEIPT = (
    "/data/latent-basemap/runs/round-0180/queue/artifacts/"
    "seed42-dose-matched-train/train-receipt.json"
)
R0180_QUEUE = "/data/latent-basemap/runs/round-0180/queue/queue.json"
R0180_TERMINAL = "/data/latent-basemap/runs/round-0180/queue/runner-terminal.json"

P90 = {
    "stage_nested_populations": 900.0,
    "build_quarter_graph": 900.0,
    "train_quarter": 6_000.0,
    "evaluate_quarter": 600.0,
    "build_half_graph": 1_200.0,
    "train_half": 12_000.0,
    "evaluate_half": 600.0,
    "evaluate_full_endpoint": 600.0,
    "synthesize_nested_ladder": 120.0,
}


def _document(prefix: str, round_id: str, *, status: str) -> dict[str, Any]:
    candidates = []
    for name in sorted(os.listdir(LAB_ROOT)):
        if not re.fullmatch(
            rf"{prefix}-{round_id}-[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}(?:-[0-9]{{2}})?\.md",
            name,
        ):
            continue
        path = os.path.join(LAB_ROOT, name)
        if _frontmatter(path).get("status") == status:
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0187 requires one {status} {prefix} for R{round_id}; "
            f"found {len(candidates)}"
        )
    return expected_input_signature(candidates[0])


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0187 round is not issued for the exact release")
    return expected_input_signature(ROUND_FILE)


def _read_json(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _accepted_lineage() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for round_id in ("0165", "0171", "0180", "0184"):
        documents.append(_document("review", round_id, status="accepted"))
        documents.append(_document("result", round_id, status="complete"))
    source_signature = expected_input_signature(SOURCE_POPULATION)
    source = prompt_contract.read_sealed(
        SOURCE_POPULATION, label="accepted R0165 population"
    )
    query_signature = expected_input_signature(PILE_QUERY_RECEIPT)
    query = prompt_contract.read_sealed(
        PILE_QUERY_RECEIPT, label="accepted R0171 Pile query reserve"
    )
    train_signature = expected_input_signature(FULL_TRAIN_RECEIPT)
    train = prompt_contract.read_sealed(
        FULL_TRAIN_RECEIPT, label="accepted R0180 train receipt"
    )
    queue_signature = expected_input_signature(R0180_QUEUE)
    terminal_signature = expected_input_signature(R0180_TERMINAL)
    queue = _read_json(R0180_QUEUE, label="R0180 queue")
    terminal = _read_json(R0180_TERMINAL, label="R0180 terminal")
    required = [str(job.get("id") or "") for job in queue.get("jobs") or []]
    if (
        source.get("round_id") != "0165"
        or source.get("outcome")
        != "prompted-8m-frozen-prefix-population-qualified"
        or int(source.get("retained_rows", -1)) != RUNG_ROWS["full"]
        or query.get("round_id") != "0171"
        or query.get("candidate_canonical_range") != [8_000_000, 8_004_096]
        or query.get("selected_before_training") is not True
        or (query.get("training_copy_audit") or {}).get(
            "selected_exact_training_identity_disjoint"
        )
        is not True
        or train.get("round_id") != "0180"
        or int(train.get("optimizer_updates", -1)) != FULL_SUCCESSFUL_UPDATES
        or queue.get("round_id") != "0180"
        or terminal.get("round_id") != "0180"
        or terminal.get("verdict") != "succeeded"
        or terminal.get("required_jobs") != required
        or sorted(terminal.get("completed_jobs") or []) != sorted(required)
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("gpu_wall_accounting_complete") is not True
        or terminal.get("boundary_problems") != []
    ):
        raise RuntimeError("R0187 accepted lineage changed")
    direct = [
        *documents,
        source_signature,
        dict(source["mapping"]),
        dict(source["document_compact"]),
        query_signature,
        dict(query["queries"]),
        dict(query["canonical_rows"]),
        dict(query["source_text_hashes"]),
        train_signature,
        dict(train["model"]),
        queue_signature,
        terminal_signature,
    ]
    return _dedupe(direct)


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0187 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0187_composition_nested_ladder.py",
        "tests/test_round0184_prompted_8m_dose_midpoint.py",
        "tests/test_round0166_cpu_smoke.py",
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
        timeout=180,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0187-release-cpu-smoke-v1",
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
            "nested selection -> graph/train dispatch -> explicit dose accounting "
            "-> exact nonempty train checks -> common-core decision branches"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0187 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _config_smoke() -> dict[str, Any]:
    if (
        sum(RUNG_COUNTS["quarter"].values()) != RUNG_ROWS["quarter"]
        or sum(RUNG_COUNTS["half"].values()) != RUNG_ROWS["half"]
        or sum(RUNG_COUNTS["full"].values()) != RUNG_ROWS["full"]
        or successful_updates_for_edges(FULL_GRAPH_EDGES)
        != FULL_SUCCESSFUL_UPDATES
    ):
        raise RuntimeError("R0187 frozen arithmetic changed")
    return prompt_contract.seal({
        "schema": "round0187-config-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "rung_counts": RUNG_COUNTS,
        "rung_rows": RUNG_ROWS,
        "hash_namespace": HASH_NAMESPACE,
        "full_graph_edges": FULL_GRAPH_EDGES,
        "full_successful_updates": FULL_SUCCESSFUL_UPDATES,
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "retention_ratio": RETENTION_RATIO,
        "compound_retention_ratio": COMPOUND_RETENTION_RATIO,
        "primary_metrics": list(PRIMARY_METRICS),
    })


def _job(
    *,
    job_id: str,
    action: str,
    deps: list[str],
    outputs: list[str],
    done_marker: str,
    common: list[dict[str, Any]],
    gpu: bool,
    training: bool,
    cpu_heavy: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "action": action,
        "handler_module": HANDLER_MODULE,
        "handler_callable": "run_job",
        "deps": deps,
        "outputs": outputs,
        "done_marker": done_marker,
        "expected_inputs": common,
        "p90_wall_s": P90[job_id],
        "node_policy": {
            "gpu_required": gpu,
            "training_performed": training,
            "cpu_heavy": cpu_heavy,
        },
        **extra,
    }


def prepare_round0187(
    *, release_sha: str, queue_root: str = QUEUE_ROOT
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0187 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    lineage = _accepted_lineage()
    source_population_signature = expected_input_signature(SOURCE_POPULATION)
    pile_query_signature = expected_input_signature(PILE_QUERY_RECEIPT)
    full_train_signature = expected_input_signature(FULL_TRAIN_RECEIPT)

    queue_root = create_fresh_directory(
        queue_root, label="R0187 composition nested queue"
    )
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    release_smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        release_smoke_path, _release_cpu_smoke(release_sha), immutable=True
    )
    config_smoke_path = os.path.join(preflight, "config-smoke.json")
    atomic_write_new_json(config_smoke_path, _config_smoke(), immutable=True)
    common = _dedupe([
        round_signature,
        *lineage,
        source_population_signature,
        pile_query_signature,
        full_train_signature,
        expected_input_signature(release_smoke_path),
        expected_input_signature(config_smoke_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    population_parent = ensure_data_directory(
        os.path.join(artifacts, "nested-populations")
    )
    population_roots = {
        rung: os.path.join(population_parent, rung) for rung in ("quarter", "half")
    }
    population_receipts = {
        rung: os.path.join(root, "population.json")
        for rung, root in population_roots.items()
    }
    population_summary = os.path.join(artifacts, "nested-population-summary.json")
    graph_outputs = {
        rung: os.path.join(artifacts, f"{rung}-graph")
        for rung in ("quarter", "half")
    }
    graph_manifests = {
        rung: os.path.join(root, "graph-manifest.json")
        for rung, root in graph_outputs.items()
    }
    train_outputs = {
        rung: os.path.join(artifacts, f"{rung}-seed42-train")
        for rung in ("quarter", "half")
    }
    evaluation_outputs = {
        rung: os.path.join(artifacts, f"{rung}-common-core-evaluation")
        for rung in ("quarter", "half", "full")
    }
    synthesis_output = os.path.join(artifacts, "nested-ladder-synthesis")

    jobs = [
        _job(
            job_id="stage_nested_populations",
            action="stage_nested_populations",
            deps=[],
            outputs=list(population_roots.values()),
            done_marker=os.path.join(artifacts, "nested-populations.done.json"),
            common=common,
            gpu=False,
            training=False,
            cpu_heavy=True,
            source_population_receipt=source_population_signature,
            population_roots=population_roots,
            population_summary=population_summary,
        ),
        _job(
            job_id="build_quarter_graph",
            action="build_nested_graph",
            deps=["stage_nested_populations"],
            outputs=[graph_outputs["quarter"]],
            done_marker=os.path.join(artifacts, "quarter-graph.done.json"),
            common=common,
            gpu=True,
            training=False,
            rung="quarter",
            population_receipt_path=population_receipts["quarter"],
        ),
        _job(
            job_id="train_quarter",
            action="train_nested_rung",
            deps=["build_quarter_graph"],
            outputs=[train_outputs["quarter"]],
            done_marker=os.path.join(artifacts, "quarter-train.done.json"),
            common=common,
            gpu=True,
            training=True,
            rung="quarter",
            population_receipt_path=population_receipts["quarter"],
            graph_manifest=graph_manifests["quarter"],
        ),
        _job(
            job_id="evaluate_quarter",
            action="evaluate_nested_rung",
            deps=["train_quarter"],
            outputs=[evaluation_outputs["quarter"]],
            done_marker=os.path.join(artifacts, "quarter-evaluation.done.json"),
            common=common,
            gpu=True,
            training=False,
            rung="quarter",
            population_receipt_path=population_receipts["quarter"],
            graph_manifest=graph_manifests["quarter"],
            train_output=train_outputs["quarter"],
            common_population_receipt_path=population_receipts["quarter"],
            common_graph_manifest=graph_manifests["quarter"],
            pile_query_receipt=PILE_QUERY_RECEIPT,
        ),
        _job(
            job_id="build_half_graph",
            action="build_nested_graph",
            deps=["evaluate_quarter"],
            outputs=[graph_outputs["half"]],
            done_marker=os.path.join(artifacts, "half-graph.done.json"),
            common=common,
            gpu=True,
            training=False,
            rung="half",
            population_receipt_path=population_receipts["half"],
        ),
        _job(
            job_id="train_half",
            action="train_nested_rung",
            deps=["build_half_graph"],
            outputs=[train_outputs["half"]],
            done_marker=os.path.join(artifacts, "half-train.done.json"),
            common=common,
            gpu=True,
            training=True,
            rung="half",
            population_receipt_path=population_receipts["half"],
            graph_manifest=graph_manifests["half"],
        ),
        _job(
            job_id="evaluate_half",
            action="evaluate_nested_rung",
            deps=["train_half", "evaluate_quarter"],
            outputs=[evaluation_outputs["half"]],
            done_marker=os.path.join(artifacts, "half-evaluation.done.json"),
            common=common,
            gpu=True,
            training=False,
            rung="half",
            population_receipt_path=population_receipts["half"],
            graph_manifest=graph_manifests["half"],
            train_output=train_outputs["half"],
            common_population_receipt_path=population_receipts["quarter"],
            common_graph_manifest=graph_manifests["quarter"],
            pile_query_receipt=PILE_QUERY_RECEIPT,
            quarter_evaluation_output=evaluation_outputs["quarter"],
        ),
        _job(
            job_id="evaluate_full_endpoint",
            action="evaluate_full_endpoint",
            deps=["evaluate_half", "evaluate_quarter"],
            outputs=[evaluation_outputs["full"]],
            done_marker=os.path.join(artifacts, "full-evaluation.done.json"),
            common=common,
            gpu=True,
            training=False,
            common_population_receipt_path=population_receipts["quarter"],
            common_graph_manifest=graph_manifests["quarter"],
            pile_query_receipt=PILE_QUERY_RECEIPT,
            quarter_evaluation_output=evaluation_outputs["quarter"],
            full_train_receipt=FULL_TRAIN_RECEIPT,
        ),
        _job(
            job_id="synthesize_nested_ladder",
            action="synthesize_nested_ladder",
            deps=["evaluate_quarter", "evaluate_half", "evaluate_full_endpoint"],
            outputs=[synthesis_output],
            done_marker=os.path.join(artifacts, "nested-ladder-synthesis.done.json"),
            common=common,
            gpu=False,
            training=False,
            evaluation_outputs=evaluation_outputs,
        ),
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    gpu_jobs = [job for job in jobs if job["node_policy"]["gpu_required"]]
    queue.update({
        "schema": QUEUE_SCHEMA,
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0165", "0171", "0180", "0184"],
        "capability_dependencies": [
            "jina-document-english-first8m-frozen-prefix-population-v1",
            "jina-document-english-8m-prompted-map-seed42-dose-matched-v1",
        ],
        "capabilities_produced": [POPULATION_CAPABILITY, CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            **{job["id"]: float(job["p90_wall_s"]) for job in gpu_jobs},
            "total": sum(float(job["p90_wall_s"]) for job in gpu_jobs),
        },
        "scientific_contract": {
            "question": (
                "does prompted-map quality regress with N after corpus composition, "
                "population nesting, model capacity, seed, graph/sampler law and "
                "consumed-positive-draws per edge are controlled?"
            ),
            "population": {
                "source_round": "0165",
                "hash_namespace": HASH_NAMESPACE,
                "rank": "one per-corpus canonical-ID SHA-256 rank",
                "emit_order": "accepted R0165 canonical order",
                "rung_counts": RUNG_COUNTS,
                "nesting": "quarter subset half subset full",
                "reembedding": False,
                "multiplicity_is_metadata": True,
            },
            "training": {
                "new_rungs": ["quarter", "half"],
                "full_endpoint_reused_from_round": "0180",
                "seed": 42,
                "hidden_dimension": 2048,
                "prompt": "Document: ",
                "precision": "host fp16 endpoints; device fp32 conversion; bf16 AMP",
                "graph": "fuzzy k50, IVF8192, nprobe64, seeds 113/114",
                "sampler": "fuzzy-weight proportional with replacement",
                "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
                "horizon": (
                    "ceil(R0180 updates * active directed edges / R0180 directed edges)"
                ),
            },
            "evaluation": {
                "common_mixed_core": "entire R0187 quarter population",
                "common_per_corpus_cores": list(RUNG_COUNTS["quarter"]),
                "true_ood": "R0171 post-8M Pile reserve, disjoint by text and fp16 bytes",
                "fineweb_redpajama_ood": (
                    "unavailable without new embeddings; never relabel in-support "
                    "probes as held-out"
                ),
                "primary_metrics": list(PRIMARY_METRICS),
                "diagnostic_only": ["density", "Pile projection FFR"],
            },
            "decision": {
                "per_step_retention_ratio": RETENTION_RATIO,
                "quarter_to_full_compound_ratio": COMPOUND_RETENTION_RATIO,
                "all_steps_pass": "controlled scale retained; no seed-43 replay",
                "concordant_compound_miss": (
                    "controlled size regression; seed-43 first failing boundary, "
                    "then targeted capacity if confirmed"
                ),
                "otherwise": "boundary/discordant; seed-43 first sub-0.97 boundary",
            },
            "receipt_corrections": {
                "literal_requested_positive_draws_per_edge": True,
                "literal_consumed_positive_draws": True,
                "literal_consumed_positive_draws_per_edge": True,
                "exact_nonempty_train_check_set": True,
            },
            "release_cpu_smoke": expected_input_signature(release_smoke_path),
            "config_cpu_smoke": expected_input_signature(config_smoke_path),
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0187(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
