#!/usr/bin/env python3
"""Prepare, but never launch, the pinned upstream ParamRepulsor 2M queue."""
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

from basemap import paramrepulsor_baseline as P
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0216_minilm_2m_substrate import CAPABILITY as R0216_CAPABILITY
from basemap.round0218_minilm_2m_panel import (
    CAPABILITY as R0218_PANEL_CAPABILITY,
    CENTROID_KS,
)
from basemap.round0247_registry import registry_fingerprint
from experiments.paramrepulsor_2m_nodes import (
    COMPARE_ACTION,
    COMPARE_CAPABILITY,
    PANEL_ACTION,
    PANEL_CAPABILITY,
    TRAIN_ACTION,
)
from experiments.round0265_nodes import PANEL_CAPABILITY as R0265_PANEL_CAPABILITY
from experiments.prepare_round0020_0022_queues import _base_manifest, _dedupe
from experiments.prepare_round0265_queue import (
    GRAPH_MANIFEST,
    HELDOUT_PROBES,
    HELDOUT_TRUTH,
    R0218_PANEL,
)
from experiments.prepare_round0266_queue import R0265_PANEL


SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUND_SPEC = os.path.join(SOURCE_ROOT, "docs", "baselines", "paramrepulsor-2m.md")
ENV_LOCK = os.path.join(SOURCE_ROOT, "requirements", "paramrepulsor-cu124.lock.txt")
ROUND_ROOT = "/data/latent-basemap/runs/round-0270"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-paramrepulsor-run"

TRAIN_P90_WALL_S = 108_000.0
PANEL_P90_WALL_S = 7_200.0
COMPARE_P90_WALL_S = 300.0


def _signature(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"ParamRepulsor input absent: {label} at {path}")
    return expected_input_signature(path)


def _queue_deadline(total_p90_s: float) -> str:
    margin_s = 1.15 * float(total_p90_s) + 12 * 3600
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=margin_s)).isoformat(
        timespec="seconds"
    )


def _validate_release(release_sha: str, repo_root: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("release SHA must be one full lowercase commit")
    if os.path.realpath(repo_root) != os.path.realpath(SOURCE_ROOT):
        raise RuntimeError(
            "prepare the ParamRepulsor queue from the same detached checkout named by --repo-root"
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
            "ParamRepulsor release checkout must be clean, detached, and at the requested "
            f"release: head={head!r}, dirty={bool(status.stdout.strip())}, attached={attached}"
        )
    python = os.path.join(repo_root, ".venv", "bin", "python")
    if not os.path.isfile(python):
        raise RuntimeError(
            f"dedicated ParamRepulsor environment absent at {python}; run "
            "experiments/setup_paramrepulsor_env.sh in the release checkout"
        )
    version = subprocess.run(
        [python, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not version.startswith("Python 3.10."):
        raise RuntimeError(
            f"ParamRepulsor release environment is {version}, expected Python 3.10"
        )


def _selected_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds).issubset(P.SEEDS):
        raise ValueError(f"seeds must be a nonempty unique subset of {P.SEEDS}")
    return tuple(sorted(seeds))


def prepare_queue(
    *,
    release_sha: str,
    seeds: Sequence[int] = (P.CANONICAL_SEED,),
    queue_root: str = QUEUE_ROOT,
    repo_root: str = RELEASE_ROOT,
) -> str:
    seeds = _selected_seeds(seeds)
    _validate_release(release_sha, repo_root)
    round_spec = _signature(ROUND_SPEC, "ParamRepulsor execution specification")
    environment_lock = _signature(ENV_LOCK, "ParamRepulsor CUDA 12.4 lock")
    graph_manifest_signature = _signature(GRAPH_MANIFEST, "R0216 substrate receipt")
    graph_manifest = prompt_contract.read_sealed(
        graph_manifest_signature["canonical_path"], label="R0216 substrate receipt"
    )
    if (
        graph_manifest.get("capability") != R0216_CAPABILITY
        or int(graph_manifest.get("rows", -1)) != P.ROWS
        or int(graph_manifest.get("dimension", -1)) != P.DIMENSION
    ):
        raise RuntimeError("ParamRepulsor baseline requires the sealed R0216 2M substrate")
    substrate_signature = dict(graph_manifest["substrate"])
    ordered_substrate_sha256 = str(graph_manifest["ordered_substrate_sha256"])
    r0218_panel = _signature(R0218_PANEL, "R0218 frozen panel")
    heldout_probes = _signature(HELDOUT_PROBES, "R0265 held-out probes")
    heldout_truth = _signature(HELDOUT_TRUTH, "R0265 held-out truth")
    fneg_panel = _signature(R0265_PANEL, "R0265 fneg n=13 panel")

    recipes = {seed: P.recipe(seed) for seed in P.SEEDS}
    invariants = {P.seed_invariant_sha256(value) for value in recipes.values()}
    if len(invariants) != 1:
        raise RuntimeError("ParamRepulsor seed family does not share one recipe")
    invariant = next(iter(invariants))

    ensure_data_directory(ROUND_ROOT, label="ParamRepulsor round root")
    queue_root = create_fresh_directory(queue_root, label="ParamRepulsor queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    identity_path = os.path.join(preflight, "upstream-paramrepulsor-identity.json")
    atomic_write_new_json(
        identity_path,
        prompt_contract.seal(
            {
                "schema": "baseline-upstream-paramrepulsor-2m-identity-v1",
                "round_id": P.ROUND_ID,
                "study_id": P.STUDY_ID,
                "release_sha": release_sha,
                "registered_family_seeds": list(P.SEEDS),
                "selected_queue_seeds": list(seeds),
                "seed_invariant_sha256": invariant,
                "canonical_recipe": recipes[P.CANONICAL_SEED],
                "upstream": {
                    "repository": P.UPSTREAM_REPOSITORY,
                    "commit": P.UPSTREAM_COMMIT,
                    "version": P.UPSTREAM_VERSION,
                    "license": P.UPSTREAM_LICENSE,
                    "source_closure_sha256": dict(P.UPSTREAM_SOURCE_CLOSURE),
                },
                "environment_lock": environment_lock,
                "rows": P.ROWS,
                "dimension": P.DIMENSION,
                "ordered_substrate_sha256": ordered_substrate_sha256,
                "registry_fingerprint": registry_fingerprint(),
            }
        ),
        immutable=True,
    )
    identity_signature = expected_input_signature(identity_path)
    shared_inputs = _dedupe(
        [
            round_spec,
            environment_lock,
            graph_manifest_signature,
            substrate_signature,
            identity_signature,
        ]
    )

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    train_nodes: list[str] = []
    panel_cells: list[dict[str, Any]] = []
    p90: dict[str, float] = {}
    for seed in seeds:
        capability = P.capability_for_seed(seed)
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
                "handler_module": "experiments.paramrepulsor_2m_nodes",
                "handler_callable": "run_job",
                "deps": [],
                "outputs": [output],
                "done_marker": os.path.join(artifacts, f"{node}.done.json"),
                "expected_inputs": shared_inputs,
                "p90_wall_s": TRAIN_P90_WALL_S,
                "training_seed": seed,
                "capability": capability,
                "graph_manifest_signature": graph_manifest_signature,
                "ordered_substrate_sha256": ordered_substrate_sha256,
                "seed_invariant_sha256": invariant,
                "node_policy": {
                    "gpu_required": True,
                    "training_performed": True,
                    "cpu_heavy": True,
                },
            }
        )
        p90[node] = TRAIN_P90_WALL_S

    panel_output = os.path.join(artifacts, PANEL_CAPABILITY)
    jobs.append(
        {
            "id": PANEL_ACTION,
            "action": PANEL_ACTION,
            "handler_module": "experiments.paramrepulsor_2m_nodes",
            "handler_callable": "run_job",
            "deps": train_nodes,
            "outputs": [panel_output],
            "done_marker": os.path.join(artifacts, f"{PANEL_ACTION}.done.json"),
            "expected_inputs": _dedupe(
                [*shared_inputs, r0218_panel, heldout_probes, heldout_truth]
            ),
            "p90_wall_s": PANEL_P90_WALL_S,
            "graph_manifest_signature": graph_manifest_signature,
            "panel_evidence": R0218_PANEL,
            "centroid_ks": list(CENTROID_KS),
            "heldout_probes": heldout_probes,
            "heldout_truth": heldout_truth,
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
            "handler_module": "experiments.paramrepulsor_2m_nodes",
            "handler_callable": "run_job",
            "deps": [PANEL_ACTION],
            "outputs": [compare_output],
            "done_marker": os.path.join(artifacts, f"{COMPARE_ACTION}.done.json"),
            "expected_inputs": _dedupe([*shared_inputs, fneg_panel]),
            "p90_wall_s": COMPARE_P90_WALL_S,
            "upstream_panel": {
                "kind": "file",
                "canonical_path": os.path.join(panel_output, "paramrepulsor-2m-panel.json"),
            },
            "fneg_panel": fneg_panel,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": False,
            },
        }
    )
    p90[COMPARE_ACTION] = COMPARE_P90_WALL_S
    p90["total"] = sum(p90.values())

    gpu_hours_cap = 32.0 * len(seeds) + 3.0
    queue = _base_manifest(
        round_id=P.ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_SPEC,
        queue_root=queue_root,
        gpu_hours_cap=gpu_hours_cap,
        execution_authority="owner-launched-gpu",
        gpu=True,
    )
    queue.update(
        {
            "schema": "baseline-upstream-paramrepulsor-2m-queue-v1",
            "program": P.STUDY_ID,
            "repo_root": os.path.abspath(repo_root),
            "queue_class": "gpu-training-external-baseline",
            "training_performed": True,
            "deadline_utc": _queue_deadline(p90["total"]),
            "required_reviews": [],
            "capability_dependencies": [
                R0216_CAPABILITY,
                R0218_PANEL_CAPABILITY,
                R0265_PANEL_CAPABILITY,
            ],
            "capabilities_produced": [
                *(P.capability_for_seed(seed) for seed in seeds),
                PANEL_CAPABILITY,
                COMPARE_CAPABILITY,
            ],
            "jobs": jobs,
            "p90_wall_s": p90,
            "registered": {
                "study_id": P.STUDY_ID,
                "purpose": "same-substrate external method baseline",
                "method": "upstream ParamRepulsor",
                "upstream_repository": P.UPSTREAM_REPOSITORY,
                "upstream_commit": P.UPSTREAM_COMMIT,
                "upstream_version": P.UPSTREAM_VERSION,
                "registered_family_seeds": list(P.SEEDS),
                "selected_queue_seeds": list(seeds),
                "pilot": list(seeds) != list(P.SEEDS),
                "rows": P.ROWS,
                "dimension": P.DIMENSION,
                "settings": "upstream defaults plus explicit seed and verbose logging",
                "environment": "dedicated Python 3.10 / upstream CUDA 12.4 lock",
                "gate_registerable_here": False,
                "decision_rule": (
                    "descriptive external-method comparison; no post-hoc winner or "
                    "promotion decision is registered"
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
    parser.add_argument("--seeds", type=_parse_seed_csv, default=(P.CANONICAL_SEED,))
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
