#!/usr/bin/env python3
"""Prepare, but never launch, the matched 50M fneg-off queue.

The default queue contains all three R0267 seeds.  ``--seeds 43`` prepares the
paired pilot without changing the registered three-seed family recipe.  Every
selected train is followed by one shared panel and a descriptive paired
comparison with the superseding R0267 result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap import baseline_50m_fneg_off as C
from basemap import round0113_prompt_contrast as prompt_contract
from basemap import round0267_int8_treatment as R0267
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0217_minilm_2m_seed_family import successful_updates_for_edges
from basemap.round0218_minilm_2m_panel import CENTROID_KS
from basemap.round0247_registry import registry_fingerprint
from experiments.baseline_50m_fneg_off_nodes import (
    COMPARE_ACTION,
    COMPARE_CAPABILITY,
    PANEL_ACTION,
    PANEL_CAPABILITY,
    TRAIN_ACTION,
)
from experiments.round0267_nodes import GATE_CAPABILITY as R0267_GATE_CAPABILITY
from experiments.prepare_round0020_0022_queues import _base_manifest, _dedupe
from experiments.prepare_round0267_queue import (
    R0237_GRAPH_MANIFEST,
    R0237_RESERVE,
    R0237_RESERVE_QUERY_ROWS,
    R0237_SUBSTRATE_MANIFEST,
    R0267_RESERVE_NEIGHBOUR_TRUTH,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUND_SPEC = os.path.join(REPO_ROOT, "docs", "baselines", "50m-fneg-off.md")
ROUND_ROOT = "/data/latent-basemap/runs/round-0269"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-baselines-run"
R0267_INT8_SLICE_MANIFEST = (
    "/data/latent-basemap/runs/round-0267/queue-correction-5/preflight/"
    "int8-slice-substrate-manifest.json"
)
R0267_TREATED_GATE = (
    "/data/latent-basemap/runs/round-0267/queue-correction-6/artifacts/"
    "minilm-fneg-50m-x2-seedmean-gate-v1/fneg-50m-x2-seedmean-gate.json"
)

TRAIN_P90_WALL_S = 50_400.0
PANEL_P90_WALL_S = 10_800.0
COMPARE_P90_WALL_S = 300.0


def _signature(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"50M fneg-off input absent: {label} at {path}")
    return expected_input_signature(path)


def _queue_deadline(total_p90_s: float) -> str:
    margin_s = 1.15 * float(total_p90_s) + 12 * 3600
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=margin_s)).isoformat(
        timespec="seconds"
    )


def _validate_release(release_sha: str, repo_root: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("release SHA must be one full lowercase commit")
    if os.path.realpath(repo_root) != os.path.realpath(REPO_ROOT):
        raise RuntimeError(
            "prepare the 50M queue from the same detached checkout named by --repo-root"
        )
    head = subprocess.run(
        ["git", "-C", repo_root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", repo_root, "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    attached = subprocess.run(
        ["git", "-C", repo_root, "symbolic-ref", "-q", "HEAD"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if head != release_sha or status.stdout.strip() or attached:
        raise RuntimeError(
            "50M release checkout must be clean, detached, and at the requested release: "
            f"head={head!r}, dirty={bool(status.stdout.strip())}, attached={attached}"
        )


def _closure_seal(release_sha: str) -> dict[str, Any]:
    observed = C.runtime_closure_hashes()
    files: dict[str, Any] = {}
    for name, entry in observed.items():
        files[name] = {
            "module": name,
            "path": os.path.relpath(entry["path"], REPO_ROOT),
            "bytes_at_release": entry["bytes"],
            "sha256_at_release": entry["sha256"],
        }
    return prompt_contract.seal(
        {
            "schema": C.CLOSURE_SCHEMA,
            "round_id": C.ROUND_ID,
            "study_id": C.STUDY_ID,
            "release_sha": release_sha,
            "modules": list(C.TRAIN_CLOSURE_MODULES),
            "files": files,
            "meaning": (
                "R0267's complete treatment closure plus the one-axis fneg-off control "
                "module. The train refuses if any source file differs at runtime."
            ),
        }
    )


def _selected_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds).issubset(C.SEEDS):
        raise ValueError(f"seeds must be a nonempty unique subset of {C.SEEDS}")
    return tuple(sorted(seeds))


def prepare_queue(
    *,
    release_sha: str,
    seeds: Sequence[int] = C.SEEDS,
    queue_root: str = QUEUE_ROOT,
    repo_root: str = RELEASE_ROOT,
) -> str:
    seeds = _selected_seeds(seeds)
    _validate_release(release_sha, repo_root)
    round_spec = _signature(ROUND_SPEC, "50M fneg-off execution specification")
    substrate_manifest_signature = _signature(
        R0237_SUBSTRATE_MANIFEST, "R0237 50M substrate manifest"
    )
    graph_manifest_signature = _signature(
        R0237_GRAPH_MANIFEST, "R0237 50M graph manifest"
    )
    substrate_manifest = prompt_contract.read_sealed(
        substrate_manifest_signature["canonical_path"], label="R0237 50M substrate"
    )
    graph_manifest = prompt_contract.read_sealed(
        graph_manifest_signature["canonical_path"], label="R0237 50M graph"
    )
    if (
        substrate_manifest.get("capability") != R0267.R0237_SUBSTRATE_CAPABILITY
        or int(substrate_manifest.get("rows", -1)) != C.ROWS
        or substrate_manifest.get("ordered_substrate_sha256")
        != R0267.R0237_SUBSTRATE_ORDERED_SHA256
    ):
        raise RuntimeError("bound substrate is not the sealed R0237 50M substrate")
    if (
        graph_manifest.get("capability") != R0267.R0237_GRAPH_CAPABILITY
        or int(graph_manifest.get("rows", -1)) != C.ROWS
        or int(graph_manifest.get("directed_edges", -1)) != C.SEALED_DIRECTED_EDGES
    ):
        raise RuntimeError("bound graph is not the sealed R0237 50M k15 graph")
    substrate_signature = dict(substrate_manifest["substrate"])
    graph_signature = dict(graph_manifest["graph"])
    base_horizon = successful_updates_for_edges(C.SEALED_DIRECTED_EDGES)

    configs: dict[int, dict[str, Any]] = {}
    config_digests: dict[str, str] = {}
    for seed in C.SEEDS:
        config, digest = C.control_train_config(
            seed=seed,
            graph_signature=graph_signature,
            graph_manifest_signature=graph_manifest_signature,
            substrate_signature=substrate_signature,
            graph_edges=C.SEALED_DIRECTED_EDGES,
            rows=C.ROWS,
        )
        C.assert_registered_control(config)
        configs[seed] = config
        config_digests[str(seed)] = digest
    family = C.assert_family_shares_one_recipe(configs)
    refusal_controls = C.recipe_refusal_controls()
    if not (
        refusal_controls["every_planted_defect_was_refused"]
        and refusal_controls["the_honest_control_still_passes"]
    ):
        raise RuntimeError("50M fneg-off recipe refusal controls failed")

    int8_manifest = _signature(R0267_INT8_SLICE_MANIFEST, "R0267 host-int8 slice law")
    heldout_reserve = _signature(R0237_RESERVE, "R0237 held-out reserve")
    reserve_query_rows = _signature(R0237_RESERVE_QUERY_ROWS, "R0237 reserve query rows")
    reserve_truth = _signature(
        R0267_RESERVE_NEIGHBOUR_TRUTH, "R0267 reserve-neighbour truth"
    )
    treated_gate = _signature(R0267_TREATED_GATE, "R0267 superseding treated gate")

    ensure_data_directory(ROUND_ROOT, label="50M fneg-off round root")
    queue_root = create_fresh_directory(queue_root, label="50M fneg-off queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    closure_path = os.path.join(preflight, "treatment-source-closure.json")
    atomic_write_new_json(closure_path, _closure_seal(release_sha), immutable=True)
    closure_signature = expected_input_signature(closure_path)
    identity_path = os.path.join(preflight, "matched-control-identity.json")
    atomic_write_new_json(
        identity_path,
        prompt_contract.seal(
            {
                "schema": "baseline-minilm-50m-fneg-off-identity-v1",
                "round_id": C.ROUND_ID,
                "study_id": C.STUDY_ID,
                "release_sha": release_sha,
                "registered_family_seeds": list(C.SEEDS),
                "selected_queue_seeds": list(seeds),
                "family": family,
                "per_seed_config_sha256": config_digests,
                "recipe": C.assert_registered_control(configs[C.CANONICAL_SEED]),
                "refusal_controls": refusal_controls,
                "rows": C.ROWS,
                "directed_edges": C.SEALED_DIRECTED_EDGES,
                "base_horizon": base_horizon,
                "x2_horizon": C.DOSE_MULTIPLIER * base_horizon,
                "registry_fingerprint": registry_fingerprint(),
            }
        ),
        immutable=True,
    )
    identity_signature = expected_input_signature(identity_path)
    shared_inputs = _dedupe(
        [
            round_spec,
            substrate_manifest_signature,
            graph_manifest_signature,
            substrate_signature,
            graph_signature,
            int8_manifest,
            closure_signature,
            identity_signature,
        ]
    )

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    train_nodes: list[str] = []
    panel_cells: list[dict[str, Any]] = []
    p90: dict[str, float] = {}
    for seed in seeds:
        capability = C.capability_for_seed(seed)
        node = f"{TRAIN_ACTION}_seed{seed}"
        output = os.path.join(artifacts, capability)
        train_nodes.append(node)
        panel_cells.append(
            {
                "seed": seed,
                "capability": capability,
                "train_receipt": {
                    "kind": "file",
                    "canonical_path": os.path.join(output, "train-receipt.json"),
                },
            }
        )
        jobs.append(
            {
                "id": node,
                "action": TRAIN_ACTION,
                "handler_module": "experiments.baseline_50m_fneg_off_nodes",
                "handler_callable": "run_job",
                "deps": [],
                "outputs": [output],
                "done_marker": os.path.join(artifacts, f"{node}.done.json"),
                "expected_inputs": shared_inputs,
                "p90_wall_s": TRAIN_P90_WALL_S,
                "training_seed": seed,
                "capability": capability,
                "graph_manifest_signature": graph_manifest_signature,
                "substrate_manifest_signature": substrate_manifest_signature,
                "int8_substrate_manifest_signature": int8_manifest,
                "cell_seed_invariant_sha256": family["seed_invariant_sha256"],
                "base_horizon": base_horizon,
                "treatment_closure": closure_signature,
                "node_policy": {
                    "gpu_required": True,
                    "training_performed": True,
                    "cpu_heavy": False,
                },
            }
        )
        p90[node] = TRAIN_P90_WALL_S

    panel_output = os.path.join(artifacts, PANEL_CAPABILITY)
    jobs.append(
        {
            "id": PANEL_ACTION,
            "action": PANEL_ACTION,
            "handler_module": "experiments.baseline_50m_fneg_off_nodes",
            "handler_callable": "run_job",
            "deps": train_nodes,
            "outputs": [panel_output],
            "done_marker": os.path.join(artifacts, f"{PANEL_ACTION}.done.json"),
            "expected_inputs": _dedupe(
                [*shared_inputs, heldout_reserve, reserve_query_rows, reserve_truth]
            ),
            "p90_wall_s": PANEL_P90_WALL_S,
            "substrate_manifest_signature": substrate_manifest_signature,
            "centroid_ks": list(CENTROID_KS),
            "heldout_reserve": heldout_reserve,
            "reserve_query_rows": reserve_query_rows,
            "reserve_truth": reserve_truth,
            "cells": panel_cells,
            "gate_registerable_here": False,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        }
    )
    p90[PANEL_ACTION] = PANEL_P90_WALL_S

    compare_output = os.path.join(artifacts, COMPARE_CAPABILITY)
    jobs.append(
        {
            "id": COMPARE_ACTION,
            "action": COMPARE_ACTION,
            "handler_module": "experiments.baseline_50m_fneg_off_nodes",
            "handler_callable": "run_job",
            "deps": [PANEL_ACTION],
            "outputs": [compare_output],
            "done_marker": os.path.join(artifacts, f"{COMPARE_ACTION}.done.json"),
            "expected_inputs": _dedupe([*shared_inputs, treated_gate]),
            "p90_wall_s": COMPARE_P90_WALL_S,
            "control_panel": {
                "kind": "file",
                "canonical_path": os.path.join(panel_output, "fneg-off-50m-x2-panel.json"),
            },
            "treated_gate": treated_gate,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": False,
            },
        }
    )
    p90[COMPARE_ACTION] = COMPARE_P90_WALL_S
    p90["total"] = sum(p90.values())

    gpu_hours_cap = 14.0 * len(seeds) + 4.0
    queue = _base_manifest(
        round_id=C.ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_SPEC,
        queue_root=queue_root,
        gpu_hours_cap=gpu_hours_cap,
        execution_authority="owner-launched-gpu",
        gpu=True,
    )
    queue.update(
        {
            "schema": "baseline-minilm-50m-fneg-off-queue-v1",
            "program": C.STUDY_ID,
            "repo_root": os.path.abspath(repo_root),
            "queue_class": "gpu-training-baseline",
            "training_performed": True,
            "deadline_utc": _queue_deadline(p90["total"]),
            "required_reviews": [],
            "capability_dependencies": [
                R0267.R0237_SUBSTRATE_CAPABILITY,
                R0267.R0237_GRAPH_CAPABILITY,
                R0267.INT8_SLICE_SUBSTRATE_CAPABILITY,
                R0267_GATE_CAPABILITY,
            ],
            "capabilities_produced": [
                *(C.capability_for_seed(seed) for seed in seeds),
                PANEL_CAPABILITY,
                COMPARE_CAPABILITY,
            ],
            "jobs": jobs,
            "p90_wall_s": p90,
            "registered": {
                "study_id": C.STUDY_ID,
                "purpose": "matched 50M ablation of fneg",
                "registered_family_seeds": list(C.SEEDS),
                "selected_queue_seeds": list(seeds),
                "pilot": list(seeds) != list(C.SEEDS),
                "rows": C.ROWS,
                "dose_multiplier": C.DOSE_MULTIPLIER,
                "x_residency": C.X_RESIDENCY,
                "only_treatment_delta": {
                    "path": "optimizer.fneg_weight",
                    "parent": 1.0,
                    "control": 0.0,
                },
                "gate_registerable_here": False,
                "decision_rule": (
                    "descriptive paired comparison only; no outcome changes the promoted "
                    "map or retrospectively registers a gate"
                ),
            },
        }
    )
    manifest_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(manifest_path, queue, immutable=True)
    return manifest_path


def _parse_seed_csv(value: str) -> tuple[int, ...]:
    try:
        return _selected_seeds([int(piece.strip()) for piece in value.split(",") if piece.strip()])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--seeds", type=_parse_seed_csv, default=C.SEEDS)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument("--repo-root", default=RELEASE_ROOT)
    args = parser.parse_args(argv)
    print(
        prepare_queue(
            release_sha=args.release_sha,
            seeds=args.seeds,
            queue_root=args.queue_root,
            repo_root=args.repo_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
