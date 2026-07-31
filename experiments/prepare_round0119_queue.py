#!/usr/bin/env python3
"""Prepare the no-training R0119 matched-density localization queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
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
from basemap.round0108_evaluation import validate_seal
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.round0119_nodes import (
    CALIBRATION_SCHEMA,
    CELL_ORDER,
)


ROUND_ID = "0119"
REQUIRED_REVIEWS = (
    "0037",
    "0038",
    "0107",
    "0108",
    "0109",
    "0115",
    "0117",
)
REQUIRED_CAPABILITIES = {
    "0037": "capability:jina-mrl-seed42-screen-v1",
    "0038": "capability:jina-mrl-two-seed-decision-v1",
    "0107": "capability:jina-diverse-25m-full768-trained-map-seed42-v1",
    # R0108 did not release a calibration capability. Its accepted result and
    # review directly enumerate the exact calibration evidence consumed here.
    "0108": None,
    "0109": "capability:jina-diverse-25m-full768-trained-map-seed43-v1",
    "0115": "capability:jina-fineweb-2m-prompt-map-contrast-v1",
    "0117": "capability:jina-fineweb-2m-prompt-map-seed43-contrast-v1",
}
ROUND_ROOT = "/data/latent-basemap/runs/round-0119"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0119-*.md")
R0037_TRAIN = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/d768_s42/train"
)
R0038_TRAIN = (
    "/data/latent-basemap/runs/round-0038/queue/artifacts/d768_s43/train"
)
R0115_RAW_TRAIN = (
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/"
    "artifacts/raw/train"
)
R0117_RAW_TRAIN = (
    "/data/latent-basemap/runs/round-0117/queue/artifacts/raw/train"
)
R0107_TRAIN = (
    "/data/latent-basemap/runs/round-0107/queue/artifacts/"
    "train-diverse-jina-25m"
)
R0109_TRAIN = (
    "/data/latent-basemap/runs/round-0109/queue/artifacts/"
    "train-diverse-jina-25m-seed43"
)
R0108_CALIBRATION = (
    "/data/latent-basemap/runs/round-0108/queue-attempt-3/artifacts/"
    "jina-density-calibration/jina-density-calibration.json"
)
R0115_QUEUE = (
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/queue.json"
)
R0115_TERMINAL = (
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/"
    "runner-terminal.json"
)
R0117_QUEUE = "/data/latent-basemap/runs/round-0117/queue/queue.json"
R0117_TERMINAL = (
    "/data/latent-basemap/runs/round-0117/queue/runner-terminal.json"
)


def _document(path: str) -> tuple[dict[str, str], str]:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if not text.startswith("---\n"):
        raise RuntimeError(f"missing frontmatter: {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError(f"unterminated frontmatter: {path}")
    values: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values, text


def _frontmatter(path: str) -> dict[str, str]:
    return _document(path)[0]


def _frontmatter_list(
    frontmatter: Mapping[str, str],
    key: str,
    *,
    label: str,
) -> list[str]:
    try:
        value = json.loads(frontmatter.get(key, ""))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} {key} is not a JSON list") from error
    if (
        not isinstance(value, list)
        or not all(isinstance(item, str) for item in value)
    ):
        raise RuntimeError(f"{label} {key} is not a string list")
    return value


def _issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0119 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _accepted_review(
    path: str,
    expected_sha256: str,
    *,
    round_id: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    frontmatter, review_text = _document(path)
    capability = REQUIRED_CAPABILITIES[round_id]
    if (
        signature["sha256"] != expected_sha256
        or frontmatter.get("round_id") != round_id
        or frontmatter.get("status") != "accepted"
        or (
            capability is not None
            and capability
            not in _frontmatter_list(
                frontmatter, "releases", label=f"R{round_id} review"
            )
        )
    ):
        raise RuntimeError(f"R{round_id} review is not exact and accepted")
    result_name = frontmatter.get("result") or ""
    if (
        not result_name
        or os.path.basename(result_name) != result_name
        or not re.fullmatch(
            rf"result-{round_id}-[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\.md",
            result_name,
        )
    ):
        raise RuntimeError(f"R{round_id} review has an invalid result binding")
    result_path = os.path.join(os.path.dirname(path), result_name)
    result_signature = expected_input_signature(result_path)
    result_frontmatter, result_text = _document(result_path)
    expected_result_sha256 = frontmatter.get("result_sha256")
    review_release = frontmatter.get("verified_release_commit")
    if (
        result_signature["sha256"] != expected_result_sha256
        or result_frontmatter.get("round_id") != round_id
        or result_frontmatter.get("status") != "complete"
        or result_frontmatter.get("release_commit") != review_release
        or not re.fullmatch(r"[0-9a-f]{40}", review_release or "")
    ):
        raise RuntimeError(
            f"R{round_id} accepted review does not close to its result/release"
        )
    if capability is not None:
        produced = _frontmatter_list(
            result_frontmatter,
            "capabilities_produced",
            label=f"R{round_id} result",
        )
        if capability.removeprefix("capability:") not in produced:
            raise RuntimeError(
                f"R{round_id} result does not produce its reviewed capability"
            )
    return {
        "review": signature,
        "result": result_signature,
        "release_commit": review_release,
        "capability": capability,
        "evidence_text": review_text + "\n" + result_text,
    }


def _clean_terminal(
    queue_path: str,
    terminal_path: str,
    *,
    round_id: str,
    expected_release_sha: str,
) -> dict[str, dict[str, Any]]:
    queue_signature = expected_input_signature(queue_path)
    terminal_signature = expected_input_signature(terminal_path)
    with open(queue_path, encoding="utf-8") as handle:
        queue = json.load(handle)
    with open(terminal_path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    required_jobs = [job.get("id") for job in queue.get("jobs") or []]
    release_sha = queue.get("release_sha")
    repo_root = os.path.realpath(str(queue.get("repo_root") or ""))
    start = terminal.get("release_checkout") or {}
    finish = terminal.get("release_checkout_at_finish") or {}
    nodes = terminal.get("nodes")
    if (
        queue.get("round_id") != round_id
        or not re.fullmatch(r"[0-9a-f]{40}", str(release_sha or ""))
        or release_sha != expected_release_sha
        or repo_root != os.path.realpath(RELEASE_ROOT)
        or not required_jobs
        or len(required_jobs) != len(set(required_jobs))
        or terminal.get("schema") != "slim-runner-terminal-v3"
        or terminal.get("round_id") != round_id
        or terminal.get("verdict") != "succeeded"
        or terminal.get("required_jobs") != required_jobs
        or terminal.get("completed_jobs") != required_jobs
        or terminal.get("gpu_wall_accounting_complete") is not True
        or terminal.get("queue_manifest_sha256")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("boundary_problems") != []
        or any(
            checkout.get("repo_root") != repo_root
            or checkout.get("head") != release_sha
            or checkout.get("detached") is not True
            or checkout.get("dirty") is not False
            for checkout in (start, finish)
        )
        or not isinstance(nodes, list)
        or [node.get("node") for node in nodes] != required_jobs
        or any(
            node.get("returncode") != 0
            or node.get("validation_problems") != []
            for node in nodes
        )
    ):
        raise RuntimeError(f"R{round_id} terminal is not a clean success")
    return {"queue": queue_signature, "terminal": terminal_signature}


def _calibration_inputs(
    evidence: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    calibration_signature = expected_input_signature(R0108_CALIBRATION)
    with open(R0108_CALIBRATION, encoding="utf-8") as handle:
        calibration = json.load(handle)
    validate_seal(calibration, label="R0108 density calibration")
    if (
        calibration.get("schema") != CALIBRATION_SCHEMA
        or calibration.get("round_id") != "0108"
        or calibration.get("threshold_tuned_after_treatment") is not False
        or (calibration.get("floor_calibration") or {}).get(
            "registered_floor"
        )
        != 0.17589389755990817
    ):
        raise RuntimeError("R0108 calibration identity changed")
    signatures = {
        "calibration": calibration_signature,
        "calibration_arrays": dict(calibration["arrays"]),
        "census": dict(calibration["census"]),
        "census_receipt": dict(calibration["census_receipt"]),
        "representative_reference": dict(
            calibration["representative_reference"]
        ),
    }
    for label, signature in signatures.items():
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise RuntimeError(f"R0108 {label} bytes changed")
    with open(
        signatures["census_receipt"]["canonical_path"], encoding="utf-8"
    ) as handle:
        census_receipt = json.load(handle)
    source_signature = dict(census_receipt["source"])
    if expected_input_signature(
        source_signature["canonical_path"]
    ) != source_signature:
        raise RuntimeError("R0040 FineWeb source bytes changed")
    signatures["source"] = source_signature

    # The accepted review/result pair directly enumerates the calibration,
    # calibration arrays, census receipt and high-D reference. The exact census
    # receipt then transitively binds both census bytes and the source.
    text = str(evidence["evidence_text"])
    for label in (
        "calibration",
        "calibration_arrays",
        "census_receipt",
        "representative_reference",
    ):
        if signatures[label]["sha256"] not in text:
            raise RuntimeError(
                f"R0108 accepted evidence does not bind {label}"
            )
    return signatures


def _bundle_signature(
    *,
    key: str,
    group: str,
    root: str,
    round_id: str,
    seed: int,
    train_schema: str,
    config_receipt_schema: str,
    config_receipt_round_id: str | None,
    config_schema: str,
    training_population: str,
    training_graph: str,
    training_dose: str,
    training_representation: str,
    training_dequantization: str,
    semantic_contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    arm: str | None = None,
    legacy_integer_key_json_roundtrip: bool = False,
) -> dict[str, Any]:
    bundle = {
        "key": key,
        "group": group,
        "round_id": round_id,
        "seed": seed,
        "train_schema": train_schema,
        "config_receipt_schema": config_receipt_schema,
        "config_receipt_round_id": config_receipt_round_id,
        "config_schema": config_schema,
        "training_population": training_population,
        "training_graph": training_graph,
        "training_dose": training_dose,
        "training_representation": training_representation,
        "training_dequantization": training_dequantization,
        "semantic_contract": dict(semantic_contract),
        "train_receipt": expected_input_signature(
            os.path.join(root, "train-receipt.json")
        ),
        "production_config": expected_input_signature(
            os.path.join(root, "production-config.json")
        ),
        "model": expected_input_signature(os.path.join(root, "model.pt")),
        "legacy_integer_key_json_roundtrip": (
            legacy_integer_key_json_roundtrip
        ),
    }
    evidence_text = str(evidence["evidence_text"])
    for field in ("train_receipt", "model"):
        if bundle[field]["sha256"] not in evidence_text:
            raise RuntimeError(
                f"R{round_id} accepted evidence does not bind "
                f"{key} {field}"
            )
    # R0037 predates file-level production-config reporting, but its reviewed
    # train receipt binds the canonical inner config hash. All later evidence
    # directly enumerates the production-config file SHA too.
    if (
        not legacy_integer_key_json_roundtrip
        and bundle["production_config"]["sha256"] not in evidence_text
    ):
        raise RuntimeError(
            f"R{round_id} accepted evidence does not bind "
            f"{key} production_config"
        )
    bundle["accepted_review"] = dict(evidence["review"])
    bundle["accepted_result"] = dict(evidence["result"])
    bundle["reviewed_capability"] = str(evidence["capability"])
    if arm is not None:
        bundle["arm"] = arm
    return bundle


def _model_bundles(
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _bundle_signature(
            key="historical_2m_seed42",
            group="historical_2m",
            root=R0037_TRAIN,
            round_id="0037",
            seed=42,
            train_schema="round0037-train-receipt-v1",
            config_receipt_schema="round0037-production-config-receipt-v1",
            config_receipt_round_id=None,
            config_schema="round0037-d768_s42-production-config-v1",
            training_population="R0037 jina-en-2M-nested exact 2M rows",
            training_graph="R0037 fuzzy k50 graph",
            training_dose="500000 successful positive-LR updates",
            training_representation=(
                "full fp16 source resident on device as fp16"
            ),
            training_dequantization="identity fp16-to-fp32 preprocessing cast",
            semantic_contract={
                "population_rows": 2_000_000,
                "graph_neighbors": 50,
                "successful_updates": 500_000,
                "pipeline": "device",
                "sampler_class": "DeviceEdgeSampler",
                "positive_sampling": "weighted_with_replacement",
                "multiplicity_policy": "row_multiplicity_uncapped",
                "feature_residency": "device_fp16",
                "source_representation": "<f2",
                "dequantization": "identity-fp32-cast",
            },
            evidence=evidence["0037"],
            legacy_integer_key_json_roundtrip=True,
        ),
        _bundle_signature(
            key="historical_2m_seed43",
            group="historical_2m",
            root=R0038_TRAIN,
            round_id="0038",
            seed=43,
            train_schema="round0038-train-receipt-v1",
            config_receipt_schema="round0038-production-config-receipt-v1",
            config_receipt_round_id=None,
            config_schema="round0038-d768_s43-production-config-v1",
            training_population="R0037 jina-en-2M-nested exact 2M rows",
            training_graph="R0037 fuzzy k50 graph",
            training_dose="500000 successful positive-LR updates",
            training_representation=(
                "full fp16 source resident on device as fp16"
            ),
            training_dequantization="identity fp16-to-fp32 preprocessing cast",
            semantic_contract={
                "population_rows": 2_000_000,
                "graph_neighbors": 50,
                "successful_updates": 500_000,
                "pipeline": "device",
                "sampler_class": "DeviceEdgeSampler",
                "positive_sampling": "weighted_with_replacement",
                "multiplicity_policy": "row_multiplicity_uncapped",
                "feature_residency": "device_fp16",
                "source_representation": "<f2",
                "dequantization": "identity-fp32-cast",
            },
            evidence=evidence["0038"],
            legacy_integer_key_json_roundtrip=True,
        ),
        _bundle_signature(
            key="current_2m_seed42",
            group="current_2m",
            root=R0115_RAW_TRAIN,
            round_id="0115",
            seed=42,
            train_schema="round0113-train-receipt-v1",
            config_receipt_schema="round0113-production-config-v1",
            config_receipt_round_id="0115",
            config_schema="round0113-prompt-arm-train-config-v1",
            training_population=(
                "R0113 raw prompt-family-union representatives, 1993761 rows"
            ),
            training_graph="accepted R0115 raw fuzzy k50 graph",
            training_dose="500000 successful positive-LR updates",
            training_representation="raw compact fp16 host memmap",
            training_dequantization="device fp32 conversion from exact fp16",
            semantic_contract={
                "population_rows": 1_993_761,
                "graph_neighbors": 50,
                "successful_updates": 500_000,
                "pipeline": "host_weighted_jina_prompt_contrast",
                "sampler_class": "PromptWeightedJinaSampler",
                "positive_sampling": (
                    "fuzzy_weight_proportional_with_replacement_via_exact_"
                    "uniform_envelope_rejection"
                ),
                "multiplicity_policy": (
                    "shared-source-raw-document-union-representative-only"
                ),
                "feature_residency": (
                    "host-contiguous-compact-fp16-memmap"
                ),
                "source_representation": "raw-fp16",
                "dequantization": "device-fp32-from-exact-fp16",
            },
            evidence=evidence["0115"],
            arm="raw",
        ),
        _bundle_signature(
            key="current_2m_seed43",
            group="current_2m",
            root=R0117_RAW_TRAIN,
            round_id="0117",
            seed=43,
            train_schema="round0113-train-receipt-v1",
            config_receipt_schema="round0113-production-config-v1",
            config_receipt_round_id="0117",
            config_schema="round0113-prompt-arm-train-config-v1",
            training_population=(
                "R0113 raw prompt-family-union representatives, 1993761 rows"
            ),
            training_graph="accepted R0115 raw fuzzy k50 graph reused",
            training_dose="500000 successful positive-LR updates",
            training_representation="raw compact fp16 host memmap",
            training_dequantization="device fp32 conversion from exact fp16",
            semantic_contract={
                "population_rows": 1_993_761,
                "graph_neighbors": 50,
                "successful_updates": 500_000,
                "pipeline": "host_weighted_jina_prompt_contrast",
                "sampler_class": "PromptWeightedJinaSampler",
                "positive_sampling": (
                    "fuzzy_weight_proportional_with_replacement_via_exact_"
                    "uniform_envelope_rejection"
                ),
                "multiplicity_policy": (
                    "shared-source-raw-document-union-representative-only"
                ),
                "feature_residency": (
                    "host-contiguous-compact-fp16-memmap"
                ),
                "source_representation": "raw-fp16",
                "dequantization": "device-fp32-from-exact-fp16",
            },
            evidence=evidence["0117"],
            arm="raw",
        ),
        _bundle_signature(
            key="current_25m_seed42",
            group="current_25m",
            root=R0107_TRAIN,
            round_id="0107",
            seed=42,
            train_schema="round0107-diverse-jina-train-receipt-v1",
            config_receipt_schema="round0107-production-config-v1",
            config_receipt_round_id="0107",
            config_schema="round0107-diverse-jina-train-config-v1",
            training_population=(
                "R0106 diverse Jina exact-family representatives, 24948663 rows"
            ),
            training_graph=(
                "R0106 variable-symmetric fuzzy k15 topology "
                "(n_neighbors=16 including self)"
            ),
            training_dose="1459722 successful positive-LR updates",
            training_representation=(
                "signed int8 plus exact per-row fp16 scale on host"
            ),
            training_dequantization=(
                "device fp32 int8 times exact row fp16 scale"
            ),
            semantic_contract={
                "population_rows": 24_948_663,
                "graph_neighbors": 15,
                "graph_neighbors_including_self": 16,
                "successful_updates": 1_459_722,
                "pipeline": "host_weighted_jina_diverse_25m",
                "sampler_class": "DiverseWeightedJinaSampler",
                "positive_sampling": (
                    "fuzzy_weight_proportional_with_replacement_via_exact_"
                    "uniform_envelope_rejection"
                ),
                "multiplicity_policy": None,
                "feature_residency": (
                    "host-mmap-global-int8-plus-compact-map-and-host-fp16-scale"
                ),
                "source_representation": "int8-treatment",
                "dequantization": (
                    "device-fp32-int8-times-exact-row-fp16-scale"
                ),
            },
            evidence=evidence["0107"],
        ),
        _bundle_signature(
            key="current_25m_seed43",
            group="current_25m",
            root=R0109_TRAIN,
            round_id="0109",
            seed=43,
            train_schema="round0109-diverse-jina-train-receipt-v1",
            config_receipt_schema="round0109-production-config-v1",
            config_receipt_round_id="0109",
            config_schema="round0109-diverse-jina-train-config-v1",
            training_population=(
                "R0106 diverse Jina exact-family representatives, 24948663 rows"
            ),
            training_graph=(
                "R0106 variable-symmetric fuzzy k15 topology "
                "(n_neighbors=16 including self)"
            ),
            training_dose="1459722 successful positive-LR updates",
            training_representation=(
                "signed int8 plus exact per-row fp16 scale on host"
            ),
            training_dequantization=(
                "device fp32 int8 times exact row fp16 scale"
            ),
            semantic_contract={
                "population_rows": 24_948_663,
                "graph_neighbors": 15,
                "graph_neighbors_including_self": 16,
                "successful_updates": 1_459_722,
                "pipeline": "host_weighted_jina_diverse_25m",
                "sampler_class": "DiverseWeightedJinaSampler",
                "positive_sampling": (
                    "fuzzy_weight_proportional_with_replacement_via_exact_"
                    "uniform_envelope_rejection"
                ),
                "multiplicity_policy": None,
                "feature_residency": (
                    "host-mmap-global-int8-plus-compact-map-and-host-fp16-scale"
                ),
                "source_representation": "int8-treatment",
                "dequantization": (
                    "device-fp32-int8-times-exact-row-fp16-scale"
                ),
            },
            evidence=evidence["0109"],
        ),
    ]


def prepare_round0119(
    *,
    release_sha: str,
    reviews: Mapping[str, tuple[str, str]],
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0119 release SHA must be one full commit")
    round_file = _issued_round()
    if set(reviews) != set(REQUIRED_REVIEWS):
        raise RuntimeError("R0119 review set is incomplete")
    review_evidence = {
        round_id: _accepted_review(path, sha256, round_id=round_id)
        for round_id in REQUIRED_REVIEWS
        for path, sha256 in [reviews[round_id]]
    }
    review_inputs = [
        dict(review_evidence[round_id][field])
        for round_id in REQUIRED_REVIEWS
        for field in ("review", "result")
    ]
    runtime_receipts = [
        _clean_terminal(
            R0115_QUEUE,
            R0115_TERMINAL,
            round_id="0115",
            expected_release_sha=review_evidence["0115"]["release_commit"],
        ),
        _clean_terminal(
            R0117_QUEUE,
            R0117_TERMINAL,
            round_id="0117",
            expected_release_sha=review_evidence["0117"]["release_commit"],
        ),
    ]
    runtime_inputs = [
        dict(receipt[field])
        for receipt in runtime_receipts
        for field in ("queue", "terminal")
    ]
    calibration_inputs = _calibration_inputs(review_evidence["0108"])
    calibration_signature = calibration_inputs["calibration"]
    model_bundles = _model_bundles(review_evidence)
    if [bundle["key"] for bundle in model_bundles] != list(CELL_ORDER):
        raise RuntimeError("R0119 model-cell order changed")

    common_inputs = _dedupe([
        expected_input_signature(round_file),
        *review_inputs,
        *runtime_inputs,
        *calibration_inputs.values(),
        *[
            dict(bundle[field])
            for bundle in model_bundles
            for field in ("train_receipt", "production_config", "model")
        ],
    ])
    queue_root = create_fresh_directory(
        queue_root, label="R0119 density localization queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    score_output = os.path.join(artifacts, "density-localization-panel")
    decision_output = os.path.join(
        artifacts, "density-localization-decision"
    )
    jobs = [
        {
            "id": "score_density_localization",
            "action": "score_density_localization",
            "handler_module": "experiments.round0119_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [score_output],
            "done_marker": os.path.join(
                artifacts, "score_density_localization.done.json"
            ),
            "expected_inputs": common_inputs,
            "p90_wall_s": 180.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
            "r0108_calibration": calibration_signature,
            "model_bundles": model_bundles,
        },
        {
            "id": "decide_density_localization",
            "action": "decide_density_localization",
            "handler_module": "experiments.round0119_nodes",
            "handler_callable": "run_job",
            "deps": ["score_density_localization"],
            "outputs": [decision_output],
            "done_marker": os.path.join(
                artifacts, "decide_density_localization.done.json"
            ),
            "expected_inputs": _dedupe([
                expected_input_signature(round_file),
                *review_inputs,
                *runtime_inputs,
                calibration_signature,
            ]),
            "p90_wall_s": 30.0,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
            "score_output": score_output,
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=0.25,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0119-jina-density-localization-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": list(REQUIRED_REVIEWS),
        "capability_dependencies": [
            "jina-mrl-seed42-screen-v1",
            "jina-mrl-two-seed-decision-v1",
            "jina-diverse-25m-full768-trained-map-seed42-v1",
            "jina-diverse-25m-full768-trained-map-seed43-v1",
            "jina-fineweb-2m-prompt-map-contrast-v1",
            "jina-fineweb-2m-prompt-map-seed43-contrast-v1",
        ],
        "capabilities_produced": [
            "jina-density-failure-localization-v1",
        ],
        "training_performed": False,
        "scientific_contract": {
            "universe": (
                "exact R0040 1996279-row FineWeb representative universe "
                "reconstructed from the accepted R0108 calibration lineage"
            ),
            "anchors_and_high_d_radii": (
                "exact R0040 reference arrays bound by accepted R0108"
            ),
            "family_filter": "exact family size <16",
            "density_floor": (
                "unchanged R0108 registered floor 0.17589389755990817"
            ),
            "density_neighbors": 15,
            "transform_batch_rows": 8192,
            "historical_transform_path": (
                "transform all 2000000 source rows, then select the exact "
                "1996279 retained representative global rows"
            ),
            "cells": list(CELL_ORDER),
            "historical_control_requirement": (
                "both R0037/R0038 controls reproduce frozen R0108 arrays "
                "within fixed 1e-6 absolute/relative tolerance and clear "
                "the unchanged floor"
            ),
            "localization_rule": (
                "if controls reproduce, current 2M pair clears, and current "
                "25M pair does not both clear, localize only to the bundled "
                "2M-to-25M population/graph/dose/representation/"
                "dequantization/execution transition"
            ),
            "failure_uniqueness_rule": (
                "if either current 2M seed fails, reject only the claim that "
                "the failure is unique to the 25M tuple; do not exclude an "
                "additional scale contribution"
            ),
            "single_cause_localization": False,
            "matched_cell_can_rescue_native_quality": False,
            "production_or_prompt_transfer": False,
            "map_decision": False,
        },
        "p90_gpu_seconds": {
            "score_density_localization": 180.0,
            "total": 180.0,
        },
        "estimate_basis": {
            "measured_r0110_two_model_matched_density_wall_s": (
                13.835361725185066
            ),
            "six_model_linear_projection_s": 41.5060851755552,
            "expected_gpu_seconds": 45.0,
            "p90_multiplier_over_linear_projection": (
                180.0 / 41.5060851755552
            ),
            "hard_cap_gpu_seconds": 900.0,
        },
        "jobs": jobs,
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    for round_id in REQUIRED_REVIEWS:
        parser.add_argument(f"--r{round_id}-review", required=True)
        parser.add_argument(f"--r{round_id}-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    reviews = {
        round_id: (
            getattr(args, f"r{round_id}_review"),
            getattr(args, f"r{round_id}_review_sha256"),
        )
        for round_id in REQUIRED_REVIEWS
    }
    path = prepare_round0119(
        release_sha=args.release_sha,
        reviews=reviews,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
