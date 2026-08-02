#!/usr/bin/env python3
"""Prepare, but never launch, the conditional R0152 scale-rescue queue."""
from __future__ import annotations

import argparse
import glob
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
from basemap.round0104_training import validate_substrate_manifest
from basemap.round0105_search import ELIGIBILITY_PATH
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0140_subsystem_bisection import RESTORATION_FLOORS
from basemap.round0151_scale_census import (
    CAPABILITY as R0151_CAPABILITY,
    EXPECTED_GROUP_IDS_ORDERED_SHA256,
    EXPECTED_MAPPING_ORDERED_SHA256,
)
from basemap.round0152_scale_rescue import (
    CAPABILITY,
    DECISION_SCHEMA,
    DENSITY_FLOOR,
    GRAPH_K,
    OOD_RETENTION,
    RETAINED_ROWS,
    ROUND_ID,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0132_queue import (
    FULL_GRAPH_MANIFEST,
    FULL_MAPPING,
    FULL_TRAIN_OUTPUT,
    R0108_CALIBRATION,
    R0108_QUEUE,
    R0108_SELECTION,
    R0108_TERMINAL,
    R0108_TRANSFORM,
    R0108_TRANSFORM_RECEIPT,
    _accepted_control_signatures,
    _require_clean_r0108,
)
from experiments.prepare_round0138_queue import (
    _accepted_review,
    _embedded_signatures,
    _frontmatter,
)
from experiments.round0152_nodes import GRAPH_PART_NAMES


ROUND_ROOT = "/data/latent-basemap/runs/round-0152"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0152-*.md")
R0151_OUTPUT = os.path.join(
    "/data/latent-basemap/runs/round-0151/queue/artifacts", R0151_CAPABILITY
)
R0151_CENSUS = os.path.join(R0151_OUTPUT, "census.json")
R0151_MAPPING = os.path.join(R0151_OUTPUT, "compact-to-global.i64.npy")
R0151_GROUP_IDS = os.path.join(R0151_OUTPUT, "compact-group-ids.u8.npy")
R0140_QUEUE = "/data/latent-basemap/runs/round-0140/queue-attempt-2/queue.json"
INDEX_FILENAME = "jina-diverse-12p5m.ivfpq"

GPU_HOURS_MINIMUM = 2.1
GPU_HOURS_EXPECTED = 2.55
GPU_HOURS_P90 = 3.45
GPU_HOURS_MAXIMUM = 5.0
P90_GPU_SECONDS = {
    "build_search_index": 240.0,
    "qualify_fixed_search": 300.0,
    "graph_part": 700.0,
    "train_map": 8_000.0,
    "transform_map": 300.0,
    "score_matched_native": 600.0,
    "score_matched_ood": 430.0,
    "score_functional_density": 450.0,
}

REVIEW_CAPABILITIES = {
    "0087": "jina-diverse-25m-inventory-v1",
    "0103": "jina-diverse-25m-full768-int8-substrate-v1",
    "0105": "jina-diverse-25m-full768-search-qualified-v1",
    "0106": "jina-diverse-25m-full768-fuzzy-graph-v1",
    "0107": "jina-diverse-25m-full768-trained-map-seed42-v1",
    "0108": "jina-diverse-25m-map-registry-v1",
    "0119": "jina-density-failure-localization-v1",
    "0132": "jina-diverse-12p5m-25m-scale-policy-geometry-v1",
    "0140": "jina-2m-subsystem-bisection-v1",
    "0150": "jina-2m-drop-only-seed-replication-v1",
    "0151": R0151_CAPABILITY,
}


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _issued_round(release_sha: str) -> tuple[str, dict[str, Any]]:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"R0152 requires exactly one issued round; found {len(candidates)}")
    if _frontmatter(candidates[0]).get("base_commit") != release_sha:
        raise RuntimeError("R0152 issued base commit differs from release")
    return candidates[0], expected_input_signature(candidates[0])


def _accepted_activation() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for round_id, capability in REVIEW_CAPABILITIES.items():
        reviews.extend(_accepted_review(round_id, capability))
    census_signature = expected_input_signature(R0151_CENSUS)
    census = _read_json(R0151_CENSUS)
    validate_seal(census, label="accepted R0151 census")
    mapping = expected_input_signature(R0151_MAPPING)
    group_ids = expected_input_signature(R0151_GROUP_IDS)
    if (
        census.get("round_id") != "0151"
        or census.get("capability") != R0151_CAPABILITY
        or census.get("retained_rows") != RETAINED_ROWS
        or census.get("mapping") != mapping
        or census.get("group_ids") != group_ids
        or census.get("mapping_ordered_sha256") != EXPECTED_MAPPING_ORDERED_SHA256
        or census.get("group_ids_ordered_sha256")
        != EXPECTED_GROUP_IDS_ORDERED_SHA256
        or not all((census.get("checks") or {}).values())
    ):
        raise RuntimeError("R0152 accepted R0151 activation changed")
    return reviews, census_signature, mapping, group_ids


def _functional_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queue_signature = expected_input_signature(R0140_QUEUE)
    queue = _read_json(R0140_QUEUE)
    candidates = [job for job in queue.get("jobs", []) if job.get("action") == "functional_panel"]
    if len(candidates) != 1:
        raise RuntimeError("accepted R0140 functional contract is absent")
    source = candidates[0]
    required = (
        "source",
        "shared_reference_receipt",
        "high_d_reference",
        "query_truth",
        "query_embeddings",
        "centroids",
    )
    contract = {name: source[name] for name in required}
    direct = [queue_signature]
    for name in required:
        value = contract[name]
        if name == "centroids":
            direct.extend(value.values())
        else:
            direct.append(value)
    for signature in direct[1:]:
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise RuntimeError("R0140 functional contract input changed")
    return contract, direct


def _pytest_receipt(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0152 run checkout is not at the requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0152_scale_rescue.py",
        "tests/test_round0151_scale_census.py",
        "tests/test_round0132_scale_bridge.py::test_cpu_train_seal_reload_transform_panel_smoke",
        "tests/test_round0132_scale_bridge.py::test_actual_r0132_train_contract_seals_reloads_and_scores_on_cpu",
        "tests/test_round0107_training.py",
    ]
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
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
    receipt = seal({
        "schema": "round0152-release-pytest-and-cpu-path-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
    })
    if completed.returncode != 0:
        raise RuntimeError(f"R0152 release smoke failed:\n{completed.stdout}\n{completed.stderr}")
    return receipt


def _job(
    *,
    node_id: str,
    action: str,
    deps: list[str],
    output: str,
    inputs: list[dict[str, Any]],
    p90_wall_s: float,
    gpu: bool,
    training: bool = False,
    **values: Any,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "action": action,
        "handler_module": "experiments.round0152_nodes",
        "handler_callable": "run_job",
        "deps": deps,
        "outputs": [output],
        "done_marker": os.path.join(os.path.dirname(output), f"{node_id}.done.json"),
        "expected_inputs": _dedupe(inputs),
        "p90_wall_s": p90_wall_s,
        "node_policy": {"gpu_required": gpu, "training_performed": training},
        **values,
    }


def prepare_round0152(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0152 release SHA must be one full commit")
    round_path, round_signature = _issued_round(release_sha)
    reviews, census, mapping, group_ids = _accepted_activation()
    r0108_queue_signature, r0108_terminal_signature = _require_clean_r0108()
    r0108_queue = _read_json(R0108_QUEUE)
    ood_sources = [job for job in r0108_queue["jobs"] if job.get("action") == "score_ood"]
    if len(ood_sources) != 1:
        raise RuntimeError("accepted R0108 OOD source contract changed")
    ood_source = ood_sources[0]
    functional, functional_inputs = _functional_contract()
    accepted_controls = _accepted_control_signatures()
    full_graph = expected_input_signature(FULL_GRAPH_MANIFEST)
    full_mapping = expected_input_signature(FULL_MAPPING)
    selection = expected_input_signature(R0108_SELECTION)
    full_transform_receipt = next(
        signature
        for signature in accepted_controls
        if os.path.realpath(signature["canonical_path"])
        == os.path.realpath(R0108_TRANSFORM_RECEIPT)
    )
    substrate = validate_substrate_manifest(verify_payloads=False)
    eligibility = expected_input_signature(ELIGIBILITY_PATH)
    calibration = expected_input_signature(R0108_CALIBRATION)
    calibration_value = _read_json(R0108_CALIBRATION)
    calibration_inputs = _embedded_signatures(calibration_value)

    queue_root = create_fresh_directory(queue_root, label="R0152 scale rescue queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-pytest-and-cpu-path-smoke.json")
    atomic_write_new_json(smoke_path, _pytest_receipt(release_sha), immutable=True)
    smoke = expected_input_signature(smoke_path)
    common = _dedupe([
        round_signature,
        *reviews,
        census,
        mapping,
        group_ids,
        r0108_queue_signature,
        r0108_terminal_signature,
        substrate["signature"],
        substrate["payloads"]["int8"],
        substrate["payloads"]["scales"],
        substrate["payloads"]["labels"],
        eligibility,
        smoke,
    ])

    subset_output = os.path.join(artifacts, "prefix-drop-subset")
    index_output = os.path.join(artifacts, "prefix-drop-search-index")
    # The reviewed inherited R0132 builder publishes this exact basename.
    index_path = os.path.join(index_output, INDEX_FILENAME)
    qualification_output = os.path.join(artifacts, "prefix-drop-search-qualification")
    part_outputs = {
        part: os.path.join(artifacts, f"prefix-drop-graph-part-{part}")
        for part in GRAPH_PART_NAMES
    }
    graph_output = os.path.join(artifacts, "prefix-drop-fuzzy-graph")
    graph_manifest = os.path.join(graph_output, "graph-manifest.json")
    train_output = os.path.join(artifacts, "train-prefix-drop-seed42")
    transform_output = os.path.join(artifacts, "prefix-drop-coordinates")
    native_output = os.path.join(artifacts, "matched-native")
    ood_output = os.path.join(artifacts, "matched-ood")
    functional_output = os.path.join(artifacts, "functional-density")
    decision_output = os.path.join(artifacts, CAPABILITY)

    jobs = [_job(
        node_id="materialize_prefix_drop_subset",
        action="materialize_prefix_drop_subset",
        deps=[],
        output=subset_output,
        inputs=common,
        p90_wall_s=180.0,
        gpu=False,
        census=census,
        mapping=mapping,
        group_ids=group_ids,
    )]
    jobs.append(_job(
        node_id="build_search_index",
        action="build_search_index",
        deps=["materialize_prefix_drop_subset"],
        output=index_output,
        inputs=common,
        p90_wall_s=P90_GPU_SECONDS["build_search_index"],
        gpu=True,
        subset_output=subset_output,
    ))
    jobs.append(_job(
        node_id="qualify_fixed_search",
        action="qualify_fixed_search",
        deps=["build_search_index"],
        output=qualification_output,
        inputs=common,
        p90_wall_s=P90_GPU_SECONDS["qualify_fixed_search"],
        gpu=True,
        subset_output=subset_output,
        index_output=index_output,
        index=index_path,
    ))
    for part in GRAPH_PART_NAMES:
        jobs.append(_job(
            node_id=f"build_graph_part_{part}",
            action="build_graph_part",
            deps=["qualify_fixed_search"],
            output=part_outputs[part],
            inputs=common,
            p90_wall_s=P90_GPU_SECONDS["graph_part"],
            gpu=True,
            part=part,
            subset_output=subset_output,
            index_output=index_output,
            index=index_path,
            qualification_output=qualification_output,
        ))
    graph_job_ids = [f"build_graph_part_{part}" for part in GRAPH_PART_NAMES]
    jobs.append(_job(
        node_id="assemble_graph",
        action="assemble_graph",
        deps=graph_job_ids,
        output=graph_output,
        inputs=common,
        p90_wall_s=1_200.0,
        gpu=False,
        subset_output=subset_output,
        part_outputs=part_outputs,
    ))
    jobs.append(_job(
        node_id="train_map",
        action="train_map",
        deps=["assemble_graph"],
        output=train_output,
        inputs=common,
        p90_wall_s=P90_GPU_SECONDS["train_map"],
        gpu=True,
        training=True,
        release_sha=release_sha,
        graph_release_sha=release_sha,
        graph_manifest=graph_manifest,
        graph_manifest_late_bound_from="assemble_graph",
    ))
    jobs.append(_job(
        node_id="transform_map",
        action="transform_map",
        deps=["train_map"],
        output=transform_output,
        inputs=common,
        p90_wall_s=P90_GPU_SECONDS["transform_map"],
        gpu=True,
        train_output=train_output,
        graph_manifest=graph_manifest,
    ))
    jobs.append(_job(
        node_id="score_matched_native",
        action="score_matched_native",
        deps=["transform_map"],
        output=native_output,
        inputs=[*common, *accepted_controls],
        p90_wall_s=P90_GPU_SECONDS["score_matched_native"],
        gpu=True,
        subset_output=subset_output,
        train_output=train_output,
        graph_manifest=graph_manifest,
        transform_output=transform_output,
        full_transform_output=R0108_TRANSFORM,
        full_transform_receipt_sha256=full_transform_receipt["sha256"],
        full_mapping=FULL_MAPPING,
        full_mapping_sha256=full_mapping["sha256"],
        eligibility=ELIGIBILITY_PATH,
        stale_calibration=R0108_CALIBRATION,
    ))
    ood_inputs = _dedupe([
        *common,
        *accepted_controls,
        *ood_source["language_sources"].values(),
        *ood_source["diagnostic_sources"].values(),
    ])
    jobs.append(_job(
        node_id="score_matched_ood",
        action="score_matched_ood",
        deps=["train_map"],
        output=ood_output,
        inputs=ood_inputs,
        p90_wall_s=P90_GPU_SECONDS["score_matched_ood"],
        gpu=True,
        train_output=train_output,
        graph_manifest=graph_manifest,
        full_train_output=FULL_TRAIN_OUTPUT,
        full_graph_manifest=FULL_GRAPH_MANIFEST,
        full_graph_manifest_sha256=full_graph["sha256"],
        selection=R0108_SELECTION,
        selection_sha256=selection["sha256"],
        language_sources=ood_source["language_sources"],
        diagnostic_sources=ood_source["diagnostic_sources"],
    ))
    functional_common = _dedupe([
        *common,
        *functional_inputs,
        calibration,
        *calibration_inputs,
    ])
    jobs.append(_job(
        node_id="score_functional_density",
        action="score_functional_density",
        deps=["train_map"],
        output=functional_output,
        inputs=functional_common,
        p90_wall_s=P90_GPU_SECONDS["score_functional_density"],
        gpu=True,
        train_output=train_output,
        graph_manifest=graph_manifest,
        r0108_calibration=calibration,
        **functional,
    ))
    jobs.append(_job(
        node_id="decide_rescue",
        action="decide_rescue",
        deps=["score_matched_native", "score_matched_ood", "score_functional_density"],
        output=decision_output,
        inputs=common,
        p90_wall_s=90.0,
        gpu=False,
        train_output=train_output,
        graph_manifest=graph_manifest,
        native_output=native_output,
        ood_output=ood_output,
        functional_output=functional_output,
    ))

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_path,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0152-prefix-drop-rescue-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": list(REVIEW_CAPABILITIES),
        "capability_dependencies": list(REVIEW_CAPABILITIES.values()),
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "scientific_contract": {
            "question": "does the replicated 2M prefix/drop-only package transfer to a distinct diverse 12.5M population?",
            "estimand": "population plus induced graph plus coverage-aligned horizon; not a unique duplicate, row-order, cardinality, graph, dose, or pure-N effect",
            "rows": RETAINED_ROWS,
            "mapping_ordered_sha256": EXPECTED_MAPPING_ORDERED_SHA256,
            "graph": {"k_nonself": GRAPH_K, "fresh": True, "resumable_parts": list(GRAPH_PART_NAMES)},
            "training": {"seed": 42, "successful_updates": "ceil(actual directed fuzzy edges / 409)"},
            "selector": {
                "functional_floors": RESTORATION_FLOORS,
                "fixed_density_v2_floor": DENSITY_FLOOR,
                "ood_retention_vs_accepted_25m": OOD_RETENTION,
                "ood_metrics": ["FineWeb recall@50", "Polish recall@50", "19-language median recall@50"],
                "all_axes_required": True,
                "projection_ffr_in_functional_panel": "gating under the frozen R0140 floor",
                "projection_ffr_in_ood_panel": "diagnostic only",
            },
            "native_matched_panel": "diagnostic geometry plus execution validity",
            "positive_branch": "release candidate for a separate 25M prefix/drop round and later reviewed registry promotion",
            "registry_mutation": False,
            "production_or_publishing": False,
            "decision_schema": DECISION_SCHEMA,
            "release_smoke": smoke,
        },
        "gpu_hours": {
            "minimum": GPU_HOURS_MINIMUM,
            "expected": GPU_HOURS_EXPECTED,
            "p90": GPU_HOURS_P90,
            "maximum": GPU_HOURS_MAXIMUM,
        },
        "p90_gpu_seconds": {
            **P90_GPU_SECONDS,
            "graph_parts_total": len(GRAPH_PART_NAMES) * P90_GPU_SECONDS["graph_part"],
            "total": sum(P90_GPU_SECONDS.values()) + 2 * P90_GPU_SECONDS["graph_part"],
        },
        "jobs": jobs,
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(prepare_round0152(release_sha=args.release_sha, queue_root=args.queue_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
