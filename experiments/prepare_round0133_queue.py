#!/usr/bin/env python3
"""Prepare, but never launch, the conditional R0133 seed-replay queue."""
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

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0104_training import validate_substrate_manifest
from basemap.round0132_scale_bridge import (
    DECISION_SCHEMA as R0132_DECISION_SCHEMA,
    GRAPH_SCHEMA,
    NATIVE_SCHEMA as R0132_NATIVE_SCHEMA,
    OOD_SCHEMA as R0132_OOD_SCHEMA,
    OUTCOME_INVALID,
    PRODUCTION_CONFIG_SCHEMA as R0132_PRODUCTION_CONFIG_SCHEMA,
    SCALE_POLICY_CAPABILITY,
    TRAIN_RECEIPT_SCHEMA as R0132_TRAIN_RECEIPT_SCHEMA,
    validate_seal,
    validate_train_execution,
)
from basemap.round0133_seed_replay import (
    DECISION_SCHEMA,
    ROUND_ID,
    TWO_SEED_CAPABILITY,
    assert_no_r0110_coordinate_inputs,
    validate_accepted_seed42_decision,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.round0132_nodes import (
    _authenticate_native_selector,
    _authenticate_ood_metrics,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0133"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0133-*.md")

R0103_REVIEW = os.path.join(LAB_ROOT, "review-0103-2026-07-29.md")
R0103_REVIEW_SHA256 = "c6c4f780c20cc34c7707132581ffaaf8daa8cc7ea9eb1cee3f76e128b6c37a51"
R0103_QUEUE = "/data/latent-basemap/runs/round-0103/queue/queue.json"
R0103_TERMINAL = "/data/latent-basemap/runs/round-0103/queue/runner-terminal.json"

R0109_REVIEW = os.path.join(LAB_ROOT, "review-0109-2026-07-30.md")
R0109_REVIEW_SHA256 = "8302f5e7d729f090b8d07baad52100c3ef11dc781f9cfd285d4df8b92be9639e"
R0109_QUEUE = "/data/latent-basemap/runs/round-0109/queue/queue.json"
R0109_TERMINAL = "/data/latent-basemap/runs/round-0109/queue/runner-terminal.json"
R0109_TRAIN_OUTPUT = (
    "/data/latent-basemap/runs/round-0109/queue/artifacts/"
    "train-diverse-jina-25m-seed43"
)
R0109_MODEL = os.path.join(R0109_TRAIN_OUTPUT, "model.pt")
R0109_CONFIG = os.path.join(R0109_TRAIN_OUTPUT, "production-config.json")
R0109_TRAIN = os.path.join(R0109_TRAIN_OUTPUT, "train-receipt.json")
R0109_MODEL_SHA256 = "8da3a59b3e97cc79d05da9fc278df09d26e31e1eee30fffe9c4b448449048ba0"
R0109_CONFIG_SHA256 = "4ecb32d5f3186cbbc4af86298b2e214f508c7f7d65e8a8ceb5466c24cc6cbbc0"
R0109_TRAIN_SHA256 = "5c18140a7d0cb7a946db1adb53c4a568997c511973eb438506b5a1f1b19930de"

R0106_GRAPH = (
    "/data/latent-basemap/runs/round-0106/queue-attempt-3/artifacts/"
    "canonical-fuzzy-graph/graph-manifest.json"
)

R0132_REVIEW_GLOB = os.path.join(LAB_ROOT, "review-0132-*.md")
R0132_QUEUE = "/data/latent-basemap/runs/round-0132/queue/queue.json"
R0132_TERMINAL = "/data/latent-basemap/runs/round-0132/queue/runner-terminal.json"
R0132_ARTIFACTS = "/data/latent-basemap/runs/round-0132/queue/artifacts"
R0132_SUBSET = os.path.join(R0132_ARTIFACTS, "half-subset")
R0132_SUBSET_MANIFEST = os.path.join(R0132_SUBSET, "subset-manifest.json")
R0132_GRAPH_ROOT = os.path.join(R0132_ARTIFACTS, "half-fuzzy-graph")
R0132_GRAPH = os.path.join(R0132_GRAPH_ROOT, "graph-manifest.json")
R0132_TRAIN_OUTPUT = os.path.join(R0132_ARTIFACTS, "train-half-seed42")
R0132_TRAIN = os.path.join(R0132_TRAIN_OUTPUT, "train-receipt.json")
R0132_CONFIG = os.path.join(R0132_TRAIN_OUTPUT, "production-config.json")
R0132_MODEL = os.path.join(R0132_TRAIN_OUTPUT, "model.pt")
R0132_NATIVE_ROOT = os.path.join(R0132_ARTIFACTS, "matched-native")
R0132_NATIVE = os.path.join(R0132_NATIVE_ROOT, "matched-native.json")
R0132_OOD_ROOT = os.path.join(R0132_ARTIFACTS, "matched-ood")
R0132_OOD = os.path.join(R0132_OOD_ROOT, "matched-ood.json")
R0132_DECISION = os.path.join(R0132_ARTIFACTS, "decision", "decision.json")

GPU_HOURS_MINIMUM = 1.6
GPU_HOURS_EXPECTED = 2.0
GPU_HOURS_P90 = 2.59
GPU_HOURS_MAXIMUM = 3.5
P90_NODE_SECONDS = {
    "train_u12_seed43": 8_000.0,
    "transform_seed43_models_on_u12": 300.0,
    "score_matched_native_seed43": 600.0,
    "score_matched_ood_seed43": 430.0,
}
P90_GPU_TOTAL_SECONDS = sum(P90_NODE_SECONDS.values())


def _frontmatter(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if not text.startswith("---\n"):
        raise RuntimeError(f"missing frontmatter: {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError(f"unterminated frontmatter: {path}")
    output: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            output[key.strip()] = value.strip().strip("\"'")
    return output


def _frontmatter_list(
    frontmatter: Mapping[str, str], key: str, *, label: str
) -> list[str]:
    try:
        values = json.loads(frontmatter.get(key) or "[]")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} {key} is malformed") from exc
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise RuntimeError(f"{label} {key} is malformed")
    return values


def _read_json(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0133 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_round_release(round_file: str, release_sha: str) -> None:
    if _frontmatter(round_file).get("base_commit") != release_sha:
        raise RuntimeError(
            "R0133 issued round base_commit must equal the materialized release SHA"
        )


def _require_review_result(
    review_path: str,
    *,
    expected_review_sha256: str | None,
    round_id: str,
    capability: str,
) -> dict[str, Any]:
    review_signature = expected_input_signature(review_path)
    frontmatter = _frontmatter(review_path)
    release_name = f"capability:{capability}"
    if (
        expected_review_sha256 is not None
        and review_signature["sha256"] != expected_review_sha256
    ):
        raise RuntimeError(f"Review {round_id} bytes changed")
    if (
        frontmatter.get("round_id") != round_id
        or frontmatter.get("status") != "accepted"
        or release_name
        not in _frontmatter_list(frontmatter, "releases", label=f"R{round_id} review")
    ):
        raise RuntimeError(f"Review {round_id} does not release {capability}")
    result_name = frontmatter.get("result") or ""
    if (
        os.path.basename(result_name) != result_name
        or not re.fullmatch(
            rf"result-{round_id}-[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}(?:-[0-9]{{2}})?\.md",
            result_name,
        )
    ):
        raise RuntimeError(f"Review {round_id} result binding is malformed")
    result_path = os.path.join(os.path.dirname(review_path), result_name)
    result_signature = expected_input_signature(result_path)
    result_frontmatter = _frontmatter(result_path)
    release = result_frontmatter.get("release_commit") or ""
    if (
        result_signature["sha256"] != frontmatter.get("result_sha256")
        or result_frontmatter.get("round_id") != round_id
        or result_frontmatter.get("status") != "complete"
        or frontmatter.get("verified_release_commit") != release
        or not re.fullmatch(r"[0-9a-f]{40}", release)
        or capability
        not in _frontmatter_list(
            result_frontmatter, "capabilities_produced", label=f"R{round_id} result"
        )
    ):
        raise RuntimeError(f"Review {round_id} does not close its exact result")
    return {
        "review": review_signature,
        "review_text": _read_text(review_path),
        "review_frontmatter": frontmatter,
        "result": result_signature,
        "result_frontmatter": result_frontmatter,
        "result_text": _read_text(result_path),
        "release_sha": release,
    }


def _require_successful_terminal(
    *,
    round_id: str,
    queue_path: str,
    terminal_path: str,
    expected_queue_schema: str,
    release_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    queue_signature = expected_input_signature(queue_path)
    terminal_signature = expected_input_signature(terminal_path)
    queue = _read_json(queue_path, label=f"R{round_id} queue")
    terminal = _read_json(terminal_path, label=f"R{round_id} terminal")
    jobs = queue.get("jobs") or []
    required = [str(job.get("id") or "") for job in jobs]
    nodes = terminal.get("nodes") or []
    if (
        queue.get("schema") != expected_queue_schema
        or queue.get("round_id") != round_id
        or queue.get("release_sha") != release_sha
        or not required
        or any(not value for value in required)
        or len(set(required)) != len(required)
        or terminal.get("schema") != "slim-runner-terminal-v3"
        or terminal.get("round_id") != round_id
        or terminal.get("verdict") != "succeeded"
        or terminal.get("required_jobs") != required
        or sorted(terminal.get("completed_jobs") or []) != sorted(required)
        or sorted(str(node.get("node") or "") for node in nodes) != sorted(required)
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("gpu_wall_accounting_complete") is not True
        or terminal.get("boundary_problems") != []
        or any(node.get("validation_problems") != [] for node in nodes)
        or (terminal.get("release_checkout") or {}).get("head") != release_sha
        or (terminal.get("release_checkout_at_finish") or {}).get("head")
        != release_sha
    ):
        raise RuntimeError(f"R{round_id} successful execution linkage changed")
    return queue_signature, terminal_signature, queue, terminal


def _declared_signature(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    required = {"kind", "canonical_path", "bytes", "sha256"}
    if (
        set(value) != required
        or value.get("kind") != "file"
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256") or ""))
    ):
        raise RuntimeError(f"{label} signature is malformed")
    path = str(value["canonical_path"])
    if (
        not os.path.isfile(path)
        or os.path.realpath(path) != path
        or os.path.getsize(path) != int(value["bytes"])
    ):
        raise RuntimeError(f"{label} declared file is absent or changed size")
    return dict(value)


def _embedded_signatures(value: Any, *, label: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if set(value) == {"kind", "canonical_path", "bytes", "sha256"}:
            output.append(_declared_signature(value, label=label))
        else:
            for key, item in value.items():
                output.extend(_embedded_signatures(item, label=f"{label}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            output.extend(_embedded_signatures(item, label=f"{label}[{index}]"))
    return _dedupe(output)


def _require_r0103() -> dict[str, Any]:
    evidence = _require_review_result(
        R0103_REVIEW,
        expected_review_sha256=R0103_REVIEW_SHA256,
        round_id="0103",
        capability="jina-diverse-25m-full768-int8-substrate-v1",
    )
    queue_path = (evidence["result_frontmatter"].get("queue_manifest") or "").removeprefix(
        "gsv:"
    )
    if os.path.realpath(queue_path) != os.path.realpath(R0103_QUEUE):
        raise RuntimeError("R0103 accepted result points at a different queue")
    queue_sig, terminal_sig, queue, _terminal = _require_successful_terminal(
        round_id="0103",
        queue_path=R0103_QUEUE,
        terminal_path=R0103_TERMINAL,
        expected_queue_schema="round0103-diverse-jina-substrate-queue-v1",
        release_sha=evidence["release_sha"],
    )
    substrate = validate_substrate_manifest(verify_payloads=False)
    if (
        "jina-diverse-25m-full768-int8-substrate-v1"
        not in (queue.get("capabilities_produced") or [])
        or (
            evidence["result_frontmatter"].get("queue_manifest_sha256")
            != queue_sig["sha256"]
            and queue_sig["sha256"] not in evidence["result_text"]
        )
    ):
        raise RuntimeError("R0103 substrate execution/result linkage changed")
    return {
        **evidence,
        "queue": queue_sig,
        "terminal": terminal_sig,
        "substrate": substrate,
        "signatures": _dedupe([
            evidence["review"],
            evidence["result"],
            queue_sig,
            terminal_sig,
            substrate["signature"],
            *substrate["payloads"].values(),
        ]),
    }


def _require_r0109() -> dict[str, Any]:
    evidence = _require_review_result(
        R0109_REVIEW,
        expected_review_sha256=R0109_REVIEW_SHA256,
        round_id="0109",
        capability="jina-diverse-25m-full768-trained-map-seed43-v1",
    )
    queue_path = (evidence["result_frontmatter"].get("queue_manifest") or "").removeprefix(
        "gsv:"
    )
    if os.path.realpath(queue_path) != os.path.realpath(R0109_QUEUE):
        raise RuntimeError("R0109 accepted result points at a different queue")
    queue_sig, terminal_sig, queue, _terminal = _require_successful_terminal(
        round_id="0109",
        queue_path=R0109_QUEUE,
        terminal_path=R0109_TERMINAL,
        expected_queue_schema="round0109-diverse-jina-seed43-training-queue-v1",
        release_sha=evidence["release_sha"],
    )
    exact = {
        "model": (R0109_MODEL, R0109_MODEL_SHA256),
        "config": (R0109_CONFIG, R0109_CONFIG_SHA256),
        "train": (R0109_TRAIN, R0109_TRAIN_SHA256),
    }
    signatures: dict[str, dict[str, Any]] = {}
    for key, (path, digest) in exact.items():
        signature = expected_input_signature(path)
        if signature["sha256"] != digest:
            raise RuntimeError(f"accepted R0109 {key} bytes changed")
        signatures[key] = signature
    train = _read_json(R0109_TRAIN, label="R0109 train receipt")
    config = _read_json(R0109_CONFIG, label="R0109 production config")
    validate_seal(train, label="accepted R0109 train receipt")
    inner_config = config.get("config") or {}
    runtime = train.get("exact_execution_receipt") or {}
    if (
        train.get("round_id") != "0109"
        or train.get("model") != signatures["model"]
        or inner_config.get("optimizer", {}).get("seed") != 43
        or config.get("config_sha256") != sha256_bytes(canonical_json(inner_config))
        or train.get("production_config_sha256") != config.get("config_sha256")
        or train.get("train_accounting", {}).get("optimizer_steps_succeeded")
        != 1_459_722
        or any(
            train.get("train_accounting", {}).get(key) != 0
            for key in (
                "amp_overflow_skips",
                "nonfinite_loss_skips",
                "nonfinite_gradient_skips",
            )
        )
        or runtime.get("pipeline") != "host_weighted_jina_diverse_25m"
        or runtime.get("sampler_class") != "DiverseWeightedJinaSampler"
        or runtime.get("positive_sampling")
        != (
            "fuzzy_weight_proportional_with_replacement_via_exact_"
            "uniform_envelope_rejection"
        )
        or runtime.get("negative_sampling")
        != "uniform-24,948,663-compact-retained-rows-nonself"
        or queue.get("capabilities_produced")
        != ["jina-diverse-25m-full768-trained-map-seed43-v1"]
        or evidence["result_frontmatter"].get("queue_manifest_sha256")
        != queue_sig["sha256"]
        or not all(
            digest in (evidence["result_text"] + evidence["review_text"])
            for digest in (
                queue_sig["sha256"],
                terminal_sig["sha256"],
                R0109_MODEL_SHA256,
                R0109_CONFIG_SHA256,
                R0109_TRAIN_SHA256,
            )
        )
    ):
        raise RuntimeError("R0109 model execution/result linkage changed")
    graph = expected_input_signature(R0106_GRAPH)
    return {
        **evidence,
        **signatures,
        "queue": queue_sig,
        "terminal": terminal_sig,
        "graph": graph,
        "signatures": _dedupe([
            evidence["review"],
            evidence["result"],
            queue_sig,
            terminal_sig,
            *signatures.values(),
            graph,
        ]),
    }


def _discover_accepted_r0132_review() -> str:
    candidates: list[str] = []
    for path in sorted(glob.glob(R0132_REVIEW_GLOB)):
        try:
            frontmatter = _frontmatter(path)
            releases = _frontmatter_list(frontmatter, "releases", label="R0132 review")
        except (OSError, RuntimeError):
            continue
        if (
            frontmatter.get("round_id") == "0132"
            and frontmatter.get("status") == "accepted"
            and f"capability:{SCALE_POLICY_CAPABILITY}" in releases
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            "R0133 requires exactly one accepted capability-bearing R0132 review; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _require_r0132() -> dict[str, Any]:
    review_path = _discover_accepted_r0132_review()
    evidence = _require_review_result(
        review_path,
        expected_review_sha256=None,
        round_id="0132",
        capability=SCALE_POLICY_CAPABILITY,
    )
    result_queue = (evidence["result_frontmatter"].get("queue_manifest") or "").removeprefix(
        "gsv:"
    )
    if os.path.realpath(result_queue) != os.path.realpath(R0132_QUEUE):
        raise RuntimeError("R0132 accepted result points at a different queue")
    queue_sig, terminal_sig, queue, _terminal = _require_successful_terminal(
        round_id="0132",
        queue_path=R0132_QUEUE,
        terminal_path=R0132_TERMINAL,
        expected_queue_schema="round0132-matched-scale-policy-queue-v1",
        release_sha=evidence["release_sha"],
    )
    if (
        queue.get("capabilities_produced") != [SCALE_POLICY_CAPABILITY]
        or evidence["result_frontmatter"].get("queue_manifest_sha256")
        != queue_sig["sha256"]
    ):
        raise RuntimeError("R0132 queue/result capability linkage changed")

    subset = _read_json(R0132_SUBSET_MANIFEST, label="R0132 subset manifest")
    graph = _read_json(R0132_GRAPH, label="R0132 graph manifest")
    train = _read_json(R0132_TRAIN, label="R0132 train receipt")
    config = _read_json(R0132_CONFIG, label="R0132 production config")
    native = _read_json(R0132_NATIVE, label="R0132 native receipt")
    ood = _read_json(R0132_OOD, label="R0132 OOD receipt")
    decision = _read_json(R0132_DECISION, label="R0132 decision")
    for value, label in (
        (subset, "R0132 subset manifest"),
        (graph, "R0132 graph manifest"),
        (train, "R0132 train receipt"),
        (native, "R0132 native receipt"),
        (ood, "R0132 OOD receipt"),
        (decision, "R0132 decision"),
    ):
        validate_seal(value, label=label)
    if (
        graph.get("schema") != GRAPH_SCHEMA
        or graph.get("round_id") != "0132"
        or train.get("schema") != R0132_TRAIN_RECEIPT_SCHEMA
        or config.get("schema") != R0132_PRODUCTION_CONFIG_SCHEMA
        or native.get("schema") != R0132_NATIVE_SCHEMA
        or ood.get("schema") != R0132_OOD_SCHEMA
        or decision.get("schema") != R0132_DECISION_SCHEMA
        or decision.get("round_id") != "0132"
        or decision.get("outcome") == OUTCOME_INVALID
    ):
        raise RuntimeError("R0132 accepted artifact schemas/outcome changed")
    validate_train_execution(train=train, config_receipt=config, graph=graph)
    _authenticate_native_selector(native)
    _authenticate_ood_metrics(ood)
    validate_accepted_seed42_decision(decision)

    primary_paths = [
        R0132_SUBSET_MANIFEST,
        R0132_GRAPH,
        R0132_TRAIN,
        R0132_CONFIG,
        R0132_MODEL,
        R0132_NATIVE,
        str((native.get("arrays") or {}).get("canonical_path") or ""),
        R0132_OOD,
        str((ood.get("arrays") or {}).get("canonical_path") or ""),
        R0132_DECISION,
    ]
    primary = [expected_input_signature(path) for path in primary_paths]
    if (
        train.get("model") != primary[4]
        or decision.get("native_panel") != primary[5]
        or decision.get("ood_panel") != primary[7]
        or not all(
            digest in (evidence["result_text"] + evidence["review_text"])
            for digest in (
                queue_sig["sha256"],
                terminal_sig["sha256"],
                primary[-1]["sha256"],
            )
        )
    ):
        raise RuntimeError("R0132 result does not bind accepted decision artifacts")
    declared = _dedupe([
        *_embedded_signatures(subset, label="R0132 subset"),
        *_embedded_signatures(graph, label="R0132 graph"),
        *_embedded_signatures(train, label="R0132 train"),
        *_embedded_signatures(native, label="R0132 native"),
        *_embedded_signatures(ood, label="R0132 OOD"),
        *_embedded_signatures(decision, label="R0132 decision"),
    ])
    round_name = evidence["review_frontmatter"].get("round") or ""
    if (
        os.path.basename(round_name) != round_name
        or not re.fullmatch(r"round-0132-[0-9]{4}-[0-9]{2}-[0-9]{2}\.md", round_name)
    ):
        raise RuntimeError("R0132 review issued-round binding is malformed")
    round_path = os.path.join(LAB_ROOT, round_name)
    round_signature = expected_input_signature(round_path)
    if round_signature["sha256"] != evidence["review_frontmatter"].get("round_sha256"):
        raise RuntimeError("R0132 review does not bind its issued round")
    return {
        **evidence,
        "review_path": review_path,
        "round": round_signature,
        "queue": queue_sig,
        "terminal": terminal_sig,
        "queue_value": queue,
        "subset": subset,
        "graph": graph,
        "train": train,
        "config": config,
        "native": native,
        "ood": ood,
        "decision": decision,
        "primary": {path: signature for path, signature in zip(primary_paths, primary)},
        "signatures": _dedupe([
            round_signature,
            evidence["review"],
            evidence["result"],
            queue_sig,
            terminal_sig,
            *primary,
            *declared,
        ]),
    }


def _job(
    *,
    node_id: str,
    action: str,
    deps: list[str],
    output: str,
    expected_inputs: list[dict[str, Any]],
    p90_wall_s: float,
    gpu: bool,
    training: bool = False,
    **values: Any,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "action": action,
        "handler_module": "experiments.round0133_nodes",
        "handler_callable": "run_job",
        "deps": deps,
        "outputs": [output],
        "done_marker": os.path.join(os.path.dirname(output), f"{node_id}.done.json"),
        "expected_inputs": _dedupe(expected_inputs),
        "p90_wall_s": p90_wall_s,
        "node_policy": {"gpu_required": gpu, "training_performed": training},
        **values,
    }


def prepare_round0133(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0133 release SHA must be one full commit")
    round_file = _require_issued_round()
    _require_round_release(round_file, release_sha)
    r0103 = _require_r0103()
    r0109 = _require_r0109()
    r0132 = _require_r0132()
    common = _dedupe([
        expected_input_signature(round_file),
        *r0103["signatures"],
        *r0109["signatures"],
        *r0132["signatures"],
    ])
    source_jobs = {
        str(job["action"]): job for job in r0132["queue_value"]["jobs"]
    }
    source_ood = source_jobs.get("score_matched_ood")
    if not isinstance(source_ood, Mapping):
        raise RuntimeError("accepted R0132 OOD source job is missing")
    language_sources = source_ood.get("language_sources")
    diagnostic_sources = source_ood.get("diagnostic_sources")
    if not isinstance(language_sources, Mapping) or not isinstance(
        diagnostic_sources, Mapping
    ):
        raise RuntimeError("accepted R0132 OOD source bindings are missing")
    source_inputs = _dedupe([
        *language_sources.values(), *diagnostic_sources.values()
    ])
    selection = str(source_ood["selection"])
    selection_signature = expected_input_signature(selection)
    if selection_signature["sha256"] != source_ood.get("selection_sha256"):
        raise RuntimeError("accepted R0132 OOD selection bytes changed")

    queue_root = create_fresh_directory(
        queue_root, label="R0133 seed-43 matched scale-policy replay queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    train_output = os.path.join(artifacts, "train-u12-seed43")
    transform_output = os.path.join(artifacts, "seed43-transforms-on-u12")
    native_output = os.path.join(artifacts, "matched-native-seed43")
    ood_output = os.path.join(artifacts, "matched-ood-seed43")
    decision_output = os.path.join(artifacts, "two-seed-decision")
    graph_signature = r0132["primary"][R0132_GRAPH]
    r0132_config_signature = r0132["primary"][R0132_CONFIG]
    r0132_native_signature = r0132["primary"][R0132_NATIVE]
    r0132_ood_signature = r0132["primary"][R0132_OOD]
    r0132_decision_signature = r0132["primary"][R0132_DECISION]

    shared_model_values = {
        "graph_manifest": R0132_GRAPH,
        "graph_manifest_sha256": graph_signature["sha256"],
        "graph_release_sha": r0132["release_sha"],
        "train_output": train_output,
        "full_train_output": R0109_TRAIN_OUTPUT,
        "full_graph_manifest": R0106_GRAPH,
        "full_graph_manifest_sha256": r0109["graph"]["sha256"],
        "full_model": R0109_MODEL,
        "full_model_sha256": R0109_MODEL_SHA256,
        "full_production_config": R0109_CONFIG,
        "full_production_config_sha256": R0109_CONFIG_SHA256,
        "full_train_receipt": R0109_TRAIN,
        "full_train_receipt_sha256": R0109_TRAIN_SHA256,
        "r0132_production_config": R0132_CONFIG,
        "r0132_production_config_sha256": r0132_config_signature["sha256"],
    }
    train_job = _job(
        node_id="train_u12_seed43",
        action="train_u12_seed43",
        deps=[],
        output=train_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["train_u12_seed43"],
        gpu=True,
        training=True,
        release_sha=release_sha,
        graph_manifest=R0132_GRAPH,
        graph_manifest_sha256=graph_signature["sha256"],
        graph_release_sha=r0132["release_sha"],
    )
    transform_job = _job(
        node_id="transform_seed43_models_on_u12",
        action="transform_seed43_models_on_u12",
        deps=["train_u12_seed43"],
        output=transform_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["transform_seed43_models_on_u12"],
        gpu=True,
        **shared_model_values,
    )
    native_job = _job(
        node_id="score_matched_native_seed43",
        action="score_matched_native_seed43",
        deps=["transform_seed43_models_on_u12"],
        output=native_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["score_matched_native_seed43"],
        gpu=True,
        subset_output=R0132_SUBSET,
        transform_output=transform_output,
        r0132_native_receipt=R0132_NATIVE,
        r0132_native_receipt_sha256=r0132_native_signature["sha256"],
    )
    ood_job = _job(
        node_id="score_matched_ood_seed43",
        action="score_matched_ood_seed43",
        deps=["transform_seed43_models_on_u12"],
        output=ood_output,
        expected_inputs=[*common, selection_signature, *source_inputs],
        p90_wall_s=P90_NODE_SECONDS["score_matched_ood_seed43"],
        gpu=True,
        **shared_model_values,
        selection=selection,
        selection_sha256=selection_signature["sha256"],
        language_sources=dict(language_sources),
        diagnostic_sources=dict(diagnostic_sources),
        r0132_ood_receipt=R0132_OOD,
        r0132_ood_receipt_sha256=r0132_ood_signature["sha256"],
    )
    decision_job = _job(
        node_id="decide_two_seed_scale_policy",
        action="decide_two_seed_scale_policy",
        deps=["score_matched_native_seed43", "score_matched_ood_seed43"],
        output=decision_output,
        expected_inputs=common,
        p90_wall_s=60.0,
        gpu=False,
        **shared_model_values,
        transform_output=transform_output,
        native_output=native_output,
        ood_output=ood_output,
        r0132_decision=R0132_DECISION,
        r0132_decision_sha256=r0132_decision_signature["sha256"],
    )
    jobs = [train_job, transform_job, native_job, ood_job, decision_job]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0133-seed43-matched-scale-policy-replay-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0103", "0109", "0132"],
        "capability_dependencies": [
            "jina-diverse-25m-full768-int8-substrate-v1",
            "jina-diverse-25m-full768-trained-map-seed43-v1",
            SCALE_POLICY_CAPABILITY,
        ],
        "capabilities_produced": [TWO_SEED_CAPABILITY],
        "training_performed": True,
        "gpu_hours": {
            "minimum": GPU_HOURS_MINIMUM,
            "expected": GPU_HOURS_EXPECTED,
            "p90": GPU_HOURS_P90,
            "maximum": GPU_HOURS_MAXIMUM,
        },
        "p90_gpu_seconds": {
            **P90_NODE_SECONDS,
            "total": P90_GPU_TOTAL_SECONDS,
            "provisional_until_r0132_reviewed_actuals_are_registered": True,
        },
        "scientific_contract": {
            "treatment": "change only U12 training RNG seed from 42 to 43",
            "u12_subset_graph_and_horizon": "byte-identical reviewed R0132",
            "full25m_seed43_model": "exact accepted R0109 model",
            "both_models_retransformed_on_u12_in_this_release": True,
            "r0110_coordinate_inputs_forbidden": True,
            "native_and_ood_panel_math": "unchanged R0132",
            "seed_level_selector": "unchanged R0132",
            "two_seed_bootstrap_pooling": False,
            "two_seed_anchor_pooling": False,
            "realized_graph_conditioned_draw_pairing": False,
            "population_seed_variance_claim": False,
            "pure_n_or_preferred_rung_claim": False,
        },
        "jobs": jobs,
    })
    assert_no_r0110_coordinate_inputs(queue)
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(prepare_round0133(release_sha=args.release_sha, queue_root=args.queue_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
