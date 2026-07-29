#!/usr/bin/env python3
"""Prepare, but never launch, the Round 0039 30M update-budget ladder."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0039_program import (
    ARMS,
    CAP_SHA256,
    GRAPH_EFFECTIVE_EDGES,
    GRAPH_MANIFEST_SHA256,
    GRAPH_PATH,
    GRAPH_RESIDENT_EDGES,
    GRAPH_SHA256,
    R0023_SEED_SPREAD,
    UPDATES_BY_ARM,
    train_config_for_arm,
)
from experiments import prepare_round0030_queue as source
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    R0019_HIGH_D_REFERENCE,
    RUN_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0039"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0039-2026-07-24.md")
R0030_REVIEW_PATH = os.path.join(
    LAB_ROOT, "review-0030-2026-07-21.md")
R0030_CONTROL_TRAIN = (
    "/data/latent-basemap/runs/round-0030/queue/artifacts/"
    "uniform/train/train-receipt.json"
)
R0030_CONTROL_PANEL = (
    "/data/latent-basemap/runs/round-0030/queue/artifacts/"
    "uniform/panel/panel.json"
)
R0036_REVIEW_PATH = os.path.join(
    LAB_ROOT, "review-0036-2026-07-23.md")
R0036_PANEL_PATH = (
    "/data/latent-basemap/runs/round-0036/queue/artifacts/panel/panel.json"
)
FIXED_CONTEXT = {
    R0030_REVIEW_PATH: (
        "c4f2b80fe4055a1205715bc5471698cc8e2db86aa74b90d5209bfa3fffd91609",
        13_398,
    ),
    R0030_CONTROL_TRAIN: (
        "3ff40d57fc76c8af1878818460fb899f595bef32ab373b221cfee9c433192d15",
        16_393,
    ),
    R0030_CONTROL_PANEL: (
        "2606c1abd4fe01d3d49599e8f3430daab4cee0b0779e2d67290d155a59f94170",
        59_037,
    ),
    R0036_REVIEW_PATH: (
        "0e1dd3e4045615050b30fbc67d851cbf496d377583f2590e2053194b0daf4fb5",
        16_524,
    ),
    R0036_PANEL_PATH: (
        "5d30384f1c3af89f952357c4ae52686950a817908bb7e4634740cb5ca9195423",
        7_521,
    ),
}
HANDLER_MODULE = "experiments.round0039_nodes"
GPU_P90_SECONDS = 19_300.0


def _require_issued_round(path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    if not lines or lines[0].strip() != "---":
        raise RuntimeError(f"Round 0039 frontmatter is missing: {path}")
    statuses: list[str] = []
    closed = False
    for line in lines[1:]:
        if line.strip() == "---":
            closed = True
            break
        key, separator, value = line.partition(":")
        if separator and key.strip() == "status":
            statuses.append(value.strip().strip("\"'"))
    if not closed or len(statuses) != 1:
        raise RuntimeError(
            f"Round 0039 frontmatter has ambiguous status: {path}")
    if statuses[0] != "issued":
        raise RuntimeError(
            "Round 0039 queue materialization requires status: issued; "
            f"observed {statuses[0]!r}")


def _transform_templates(
    queue_root: str, release_sha: str
) -> dict[str, str]:
    from basemap.round0014_transform import build_transform_template

    inputs_root = ensure_data_directory(os.path.join(queue_root, "inputs"))
    templates: dict[str, str] = {}
    for arm in ARMS:
        config, digest = train_config_for_arm(arm)
        template = build_transform_template(
            release_root=RUN_ROOT,
            release_sha=release_sha,
            train_output_relative_path=f"artifacts/{arm}/train/model.pt",
            production_config=config,
            production_config_sha256=digest,
        )
        path = os.path.join(
            inputs_root, f"{arm}-transform-spec-template.json")
        atomic_write_new_json(path, template, immutable=True)
        templates[arm] = path
    return templates


def _static_inputs(templates: dict[str, str]) -> list[dict[str, Any]]:
    # Reuse R0030's exact 30M graph/data/evaluator closure and append only the
    # new issued contract plus fixed control/scale-diagnostic evidence.
    inputs = _dedupe([
        *source._static_inputs(templates),
        *_file_inputs([ROUND_FILE, *FIXED_CONTEXT]),
    ])
    by_path = {item["canonical_path"]: item for item in inputs}
    mismatches = {}
    for path, (sha, size) in FIXED_CONTEXT.items():
        observed = by_path.get(os.path.realpath(path), {})
        if observed.get("sha256") != sha or observed.get("bytes") != size:
            mismatches[path] = {
                "expected_sha256": sha,
                "observed_sha256": observed.get("sha256"),
                "expected_bytes": size,
                "observed_bytes": observed.get("bytes"),
            }
    if mismatches:
        raise RuntimeError(f"R0039 fixed context changed: {mismatches}")
    return inputs


def _arm_paths(artifacts: str) -> dict[str, dict[str, str]]:
    return {
        arm: {
            "train": os.path.join(artifacts, arm, "train"),
            "coordinates": os.path.join(artifacts, arm, "coordinates"),
            "panel": os.path.join(artifacts, arm, "panel"),
            "semantic_renders": os.path.join(
                artifacts, arm, "semantic-renders"),
        }
        for arm in ARMS
    }


def _jobs(
    *, artifacts: str, templates: dict[str, str], inputs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    paths = _arm_paths(artifacts)
    canary = os.path.join(artifacts, "sampler-canary")

    def job(
        node_id: str,
        handler: str,
        deps: list[str],
        output: str,
        p90: float,
        *,
        gpu: bool = True,
        training: bool = False,
        standalone: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        value = {
            "id": node_id,
            "handler": handler,
            "deps": deps,
            "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
            "outputs": [output],
            "expected_inputs": inputs,
            "p90_wall_s": float(p90),
            "node_policy": {
                "gpu_required": gpu,
                "training_performed": training,
            },
            **extra,
        }
        if standalone:
            value["handler_module"] = HANDLER_MODULE
            value["handler_callable"] = standalone
        return value

    jobs: list[dict[str, Any]] = [
        job(
            "sampler_canary",
            "round0039_sampler_canary",
            [],
            canary,
            300.0,
            standalone="run_sampler_canary",
        ),
        job(
            "u250k_train_30m",
            "train_seed42_30m",
            ["sampler_canary"],
            paths["u250k"]["train"],
            3_000.0,
            training=True,
            arm="u250k",
            canary_output=canary,
        ),
        job(
            "u1000k_train_30m",
            "train_seed42_30m",
            ["u250k_train_30m"],
            paths["u1000k"]["train"],
            10_000.0,
            training=True,
            arm="u1000k",
            canary_output=canary,
        ),
    ]
    for arm in ARMS:
        jobs.append(job(
            f"{arm}_transform_30m",
            "transform_30m",
            [f"{arm}_train_30m"],
            paths[arm]["coordinates"],
            300.0,
            arm=arm,
            train_output=paths[arm]["train"],
            transform_spec_template=templates[arm],
        ))
    for arm in ARMS:
        jobs.append(job(
            f"{arm}_registered_panel",
            "registered_panel",
            [f"{arm}_transform_30m"],
            paths[arm]["panel"],
            2_700.0,
            arm=arm,
            canary_output=canary,
            train_output=paths[arm]["train"],
            transform_output=paths[arm]["coordinates"],
            reference_output=R0019_HIGH_D_REFERENCE,
        ))
    for arm in ARMS:
        jobs.append(job(
            f"{arm}_semantic_renders",
            "semantic_renders",
            [f"{arm}_registered_panel"],
            paths[arm]["semantic_renders"],
            180.0,
            gpu=False,
            arm=arm,
            transform_output=paths[arm]["coordinates"],
            panel_output=paths[arm]["panel"],
        ))
    jobs.append(job(
        "budget_response",
        "round0039_budget_response",
        [f"{arm}_semantic_renders" for arm in ARMS],
        os.path.join(artifacts, "budget-response"),
        120.0,
        gpu=False,
        standalone="run_budget_response",
        cell_paths=paths,
        control_paths={
            "train": R0030_CONTROL_TRAIN,
            "panel": R0030_CONTROL_PANEL,
        },
        scale_panel_path=R0036_PANEL_PATH,
    ))
    return jobs


def prepare_round0039(release_sha: str) -> str:
    _require_issued_round(ROUND_FILE)
    round_root = ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(
        os.path.join(round_root, "queue"), label="R0039 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    templates = _transform_templates(queue_root, release_sha)
    inputs = _static_inputs(templates)
    manifest = _base_manifest(
        round_id="0039",
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=6.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["required_reviews"] = ["0030", "0036"]
    manifest["capability_dependencies"] = [
        "30m-fuzzy-sampling-decision-v1",
    ]
    manifest["capabilities_produced"] = [
        "30m-update-budget-response-v1",
    ]
    manifest["training_performed"] = True
    manifest["scientific_contract"] = {
        "question": "does 30M density materially respond to update horizon",
        "graph": expected_input_signature(GRAPH_PATH),
        "graph_sha256": GRAPH_SHA256,
        "manifest_sha256": GRAPH_MANIFEST_SHA256,
        "resident_edges": GRAPH_RESIDENT_EDGES,
        "effective_retained_source_edges": GRAPH_EFFECTIVE_EDGES,
        "duplicate_cap_sha256": CAP_SHA256,
        "seed": 42,
        "new_arms": {
            arm: {
                "successful_positive_lr_updates": UPDATES_BY_ARM[arm],
                "production_config_sha256": train_config_for_arm(arm)[1],
            }
            for arm in ARMS
        },
        "reused_500k_control": expected_input_signature(
            R0030_CONTROL_PANEL),
        "r0036_density_context": expected_input_signature(R0036_PANEL_PATH),
        "density_materiality_band": R0023_SEED_SPREAD["density"],
        "diagnostic_only_no_recipe_adoption": True,
        "gpu_p90_seconds": GPU_P90_SECONDS,
    }
    manifest["jobs"] = _jobs(
        artifacts=artifacts, templates=templates, inputs=inputs)
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        {"queue_manifest": prepare_round0039(args.release_sha)},
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
