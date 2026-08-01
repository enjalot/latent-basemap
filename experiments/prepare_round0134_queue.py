#!/usr/bin/env python3
"""Materialize, but never launch, the R0134 functional-showdown queue."""
from __future__ import annotations

import argparse
import copy
import glob
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
from basemap.round0134_functional_showdown import (
    CAPABILITY,
    CELL_ORDER,
    CURRENT_R0104_SEED42,
    CURRENT_RAW_SEED42,
    CURRENT_RAW_SEED43,
    HISTORICAL_SEED42,
    HISTORICAL_SEED43,
    ROUND_ID,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.round0119_nodes import SOURCE_ROWS


ROUND_ROOT = "/data/latent-basemap/runs/round-0134"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
IMPLEMENTATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0134-*.md")

R0119_QUEUE = "/data/latent-basemap/runs/round-0119/queue/queue.json"
R0122_QUEUE = "/data/latent-basemap/runs/round-0122/queue/queue.json"
SHARED_RECEIPT = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/"
    "shared-reference/receipt.json"
)
SOURCE = "/data/latent-basemap/jina-en-2M-nested/train/data-00000.npy"

REVIEW_CAPABILITIES = {
    "0037": "jina-mrl-seed42-screen-v1",
    "0038": "jina-mrl-two-seed-decision-v1",
    "0104": "jina-full768-host-int8-training-validation-v1",
    "0115": "jina-fineweb-2m-prompt-map-contrast-v1",
    "0117": "jina-fineweb-2m-prompt-map-seed43-contrast-v1",
    "0119": "jina-density-failure-localization-v1",
    "0122": "jina-density-provenance-representation-bridge-v1",
}

FROZEN_COORDINATES = {
    HISTORICAL_SEED42: {
        "coordinates": (
            "/data/latent-basemap/runs/round-0037/queue/artifacts/"
            "d768_s42/transform/coordinates.npy"
        ),
        "query_coordinates": (
            "/data/latent-basemap/runs/round-0037/queue/artifacts/"
            "d768_s42/transform/oos-query-coordinates.npy"
        ),
    },
    HISTORICAL_SEED43: {
        "coordinates": (
            "/data/latent-basemap/runs/round-0038/queue/artifacts/"
            "d768_s43/transform/coordinates.npy"
        ),
        "query_coordinates": (
            "/data/latent-basemap/runs/round-0038/queue/artifacts/"
            "d768_s43/transform/oos-query-coordinates.npy"
        ),
    },
}

GPU_HOURS_MINIMUM = 0.10
GPU_HOURS_EXPECTED = 0.30
GPU_HOURS_P90 = 0.42
GPU_HOURS_MAXIMUM = 0.50


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _frontmatter(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if not text.startswith("---\n"):
        raise RuntimeError(f"frontmatter missing: {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError(f"frontmatter unterminated: {path}")
    output: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            output[key.strip()] = value.strip().strip("\"'")
    return output


def _frontmatter_list(frontmatter: Mapping[str, str], key: str) -> list[str]:
    value = json.loads(frontmatter.get(key) or "[]")
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RuntimeError(f"frontmatter {key} is malformed")
    return value


def _authorize_release(round_path: str, release_sha: str) -> None:
    frontmatter = _frontmatter(round_path)
    original = frontmatter.get("base_commit") or ""
    if original == release_sha:
        return
    with open(round_path, encoding="utf-8") as handle:
        text = handle.read()
    if (
        "pre-execution setup correction addendum" not in text
        or f"`{release_sha}`" not in text
    ):
        raise RuntimeError("R0134 corrected release lacks an exact round addendum")
    ancestor = subprocess.run(
        ["git", "-C", IMPLEMENTATION_ROOT, "merge-base", "--is-ancestor", original, release_sha],
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("R0134 corrected release does not descend from base_commit")
    changed = subprocess.check_output(
        ["git", "-C", IMPLEMENTATION_ROOT, "diff", "--name-only", original, release_sha],
        text=True,
    ).splitlines()
    allowed = {
        "experiments/prepare_round0134_queue.py",
        "tests/test_round0134_functional_showdown.py",
    }
    if not changed or set(changed) - allowed:
        raise RuntimeError("R0134 setup correction changed runtime/scientific code")


def _issued_round(release_sha: str) -> tuple[str, dict[str, Any]]:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"R0134 requires exactly one issued round; found {len(candidates)}")
    _authorize_release(candidates[0], release_sha)
    return candidates[0], expected_input_signature(candidates[0])


def _accepted_review(round_id: str, capability: str) -> list[dict[str, Any]]:
    matches = sorted(glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md")))
    accepted = []
    for review_path in matches:
        frontmatter = _frontmatter(review_path)
        if (
            frontmatter.get("round_id") != round_id
            or frontmatter.get("status") != "accepted"
            or f"capability:{capability}"
            not in _frontmatter_list(frontmatter, "releases")
        ):
            continue
        result_name = frontmatter.get("result") or ""
        if os.path.basename(result_name) != result_name:
            raise RuntimeError(f"Review {round_id} result path is malformed")
        result_path = os.path.join(LAB_ROOT, result_name)
        result_signature = expected_input_signature(result_path)
        if result_signature["sha256"] != frontmatter.get("result_sha256"):
            raise RuntimeError(f"Review {round_id} result binding changed")
        result_frontmatter = _frontmatter(result_path)
        if (
            result_frontmatter.get("round_id") != round_id
            or result_frontmatter.get("status") not in {"complete", "failed"}
            or result_frontmatter.get("release_commit")
            != frontmatter.get("verified_release_commit")
        ):
            raise RuntimeError(f"Review {round_id} result/release binding changed")
        round_name = frontmatter.get("round") or ""
        round_path = os.path.join(LAB_ROOT, round_name)
        round_signature = expected_input_signature(round_path)
        if round_signature["sha256"] != frontmatter.get("round_sha256"):
            raise RuntimeError(f"Review {round_id} round binding changed")
        accepted.append(
            {
                "review": expected_input_signature(review_path),
                "result": result_signature,
                "round": round_signature,
            }
        )
    if len(accepted) != 1:
        raise RuntimeError(
            f"R0134 requires one accepted Review {round_id}; found {len(accepted)}"
        )
    return [accepted[0]["round"], accepted[0]["result"], accepted[0]["review"]]


def _model_bundles() -> list[dict[str, Any]]:
    r0119 = _read_json(R0119_QUEUE)
    r0122 = _read_json(R0122_QUEUE)
    source_specs = {
        spec["key"]: spec for spec in r0119["jobs"][0]["model_bundles"]
    }
    mapping = {
        HISTORICAL_SEED42: "historical_2m_seed42",
        HISTORICAL_SEED43: "historical_2m_seed43",
        CURRENT_RAW_SEED42: "current_2m_seed42",
        CURRENT_RAW_SEED43: "current_2m_seed43",
    }
    bundles: dict[str, dict[str, Any]] = {}
    for key, source_key in mapping.items():
        bundle = copy.deepcopy(source_specs[source_key])
        bundle["key"] = key
        bundles[key] = bundle
    r0104_specs = {
        spec["arm"]: spec for spec in r0122["jobs"][0]["r0104_model_bundles"]
    }
    r0104 = copy.deepcopy(r0104_specs["fp16_control"])
    r0104["key"] = CURRENT_R0104_SEED42
    bundles[CURRENT_R0104_SEED42] = r0104
    return [bundles[key] for key in CELL_ORDER]


def _embedded_signatures(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if set(("canonical_path", "kind", "bytes", "sha256")) <= set(value):
            signature = expected_input_signature(str(value["canonical_path"]))
            if signature != dict(value):
                raise RuntimeError(f"embedded artifact changed: {value['canonical_path']}")
            output.append(signature)
        else:
            for child in value.values():
                output.extend(_embedded_signatures(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(_embedded_signatures(child))
    return output


def prepare_round0134(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0134 release SHA must be one full commit")
    round_path, round_signature = _issued_round(release_sha)
    reviews = _dedupe(
        [
            signature
            for round_id, capability in REVIEW_CAPABILITIES.items()
            for signature in _accepted_review(round_id, capability)
        ]
    )
    shared = _read_json(SHARED_RECEIPT)
    shared_signature = expected_input_signature(SHARED_RECEIPT)
    source_signature = expected_input_signature(SOURCE)
    if shared.get("train") != source_signature:
        raise RuntimeError("R0037 shared reference/source binding changed")
    high_d_reference = expected_input_signature(
        shared["high_d_reference"]["canonical_path"]
    )
    query_truth = expected_input_signature(shared["query_truth"]["canonical_path"])
    query_embeddings = expected_input_signature(
        shared["query_embeddings"]["canonical_path"]
    )
    centroids = {
        key: expected_input_signature(value["canonical_path"])
        for key, value in shared["centroids"].items()
    }
    if (
        high_d_reference != shared["high_d_reference"]
        or query_truth != shared["query_truth"]
        or query_embeddings != shared["query_embeddings"]
    ):
        raise RuntimeError("R0037 shared functional artifacts changed")
    frozen_coordinates = {
        key: {
            role: expected_input_signature(path)
            for role, path in paths.items()
        }
        for key, paths in FROZEN_COORDINATES.items()
    }
    model_bundles = _model_bundles()
    context = [
        expected_input_signature(R0119_QUEUE),
        expected_input_signature(R0122_QUEUE),
    ]
    common = _dedupe(
        [
            round_signature,
            *reviews,
            *context,
            shared_signature,
            source_signature,
            high_d_reference,
            query_truth,
            query_embeddings,
            *centroids.values(),
            *_embedded_signatures(model_bundles),
            *_embedded_signatures(frozen_coordinates),
        ]
    )
    queue_root = create_fresh_directory(
        queue_root, label="R0134 functional showdown queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    panel_output = os.path.join(artifacts, "functional-showdown")
    decision_output = os.path.join(artifacts, "decision")
    jobs = [
        {
            "id": "functional_showdown_panel",
            "action": "functional_showdown_panel",
            "handler_module": "experiments.round0134_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [panel_output],
            "done_marker": os.path.join(artifacts, "functional_showdown_panel.done.json"),
            "expected_inputs": common,
            "p90_wall_s": 1_500.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
            "source": source_signature,
            "shared_reference_receipt": shared_signature,
            "high_d_reference": high_d_reference,
            "query_truth": query_truth,
            "query_embeddings": query_embeddings,
            "centroids": centroids,
            "model_bundles": model_bundles,
            "frozen_coordinates": frozen_coordinates,
        },
        {
            "id": "functional_showdown_decision",
            "action": "functional_showdown_decision",
            "handler_module": "experiments.round0134_nodes",
            "handler_callable": "run_job",
            "deps": ["functional_showdown_panel"],
            "outputs": [decision_output],
            "done_marker": os.path.join(
                artifacts, "functional_showdown_decision.done.json"
            ),
            "expected_inputs": common,
            "p90_wall_s": 30.0,
            "node_policy": {"gpu_required": False, "training_performed": False},
            "panel_output": panel_output,
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_path,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update(
        {
            "schema": "round0134-functional-showdown-queue-v1",
            "repo_root": RELEASE_ROOT,
            "queue_class": "gpu-research",
            "required_reviews": list(REVIEW_CAPABILITIES),
            "capability_dependencies": list(REVIEW_CAPABILITIES.values()),
            "capabilities_produced": [CAPABILITY],
            "training_performed": False,
            "gpu_hours": {
                "minimum": GPU_HOURS_MINIMUM,
                "expected": GPU_HOURS_EXPECTED,
                "p90": GPU_HOURS_P90,
                "maximum": GPU_HOURS_MAXIMUM,
            },
            "scientific_contract": {
                "same_ordered_evaluation_rows": SOURCE_ROWS,
                "historical_cells": [HISTORICAL_SEED42, HISTORICAL_SEED43],
                "current_cells": [
                    CURRENT_R0104_SEED42,
                    CURRENT_RAW_SEED42,
                    CURRENT_RAW_SEED43,
                ],
                "functional_metrics": [
                    "ffr",
                    "purity_fidelity_k256",
                    "purity_fidelity_k1024",
                    "projection_ffr",
                    "ood_recall_at_10",
                ],
                "raw_purity_ratios_preserved": True,
                "purity_ideal": 1.0,
                "decision": "current >= historical on every functional metric",
                "density_is_selector_input": False,
                "threshold_or_floor_changed": False,
                "training_or_graph_build": False,
                "side_by_side_renders": True,
            },
            "jobs": jobs,
        }
    )
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(prepare_round0134(release_sha=args.release_sha, queue_root=args.queue_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
