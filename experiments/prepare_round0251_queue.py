#!/usr/bin/env python3
"""Prepare, but never launch, the R0251 queue.

Three nodes, in this order:

1. `trainsetup_0251` (GPU) — the trainer's SETUP interval with the new
   `basemap/pumap/` abort-poll hook installed, and the per-batch tail modelled
   from a `10,000`-update rung's whole gap series. review-0250-01 §A.3 and §D.3.
2. `rescore_seed42_0251` (GPU) — R0218's archived seed-42 checkpoint rescored on
   this release against the same frozen panel. review-0250-01 §B.6, which names
   this experiment in one line and blocks the poolability claim until it is run.
3. `estimator_table_0251` (CPU) — the six-candidate joint table at `n = 16` with
   the `c8-seed42` column attached. review-0250-01 §B.4/§B.5, the coupling.

The GPU nodes come first so the card is busy while the CPU table runs last; the
table depends on nothing this queue produces, so it could run anywhere, and it
is placed last only because it is the cheapest to re-run.

**This queue registers nothing.** No floor, no estimator, no capability that any
downstream round consumes as a threshold. Every artifact is a measurement.
"""
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
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0216_minilm_2m_substrate import CAPABILITY as R0216_CAPABILITY
from basemap.round0217_minilm_2m_seed_family import (
    TRAIN_SCHEMA as R0217_TRAIN_SCHEMA,
    capability_for_seed as r0217_capability_for_seed,
)
from basemap.round0247_registry import registered_value, registry_fingerprint
from basemap.round0250_gate_n16 import (
    GATE_CAPABILITY as R0250_GATE_CAPABILITY,
    N_EXACT,
)
from basemap.round0250_panel_n16 import (
    PANEL_CAPABILITY,
    PANEL_CAPABILITY_N16,
    POOLED_SEEDS,
    REFERENCE_KEY,
)
from basemap.round0250_seed_extension_n16 import (
    DIMENSION,
    GRAPH_CAPABILITY,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    GRAPH_K,
    ROWS,
    SEALED_DIRECTED_EDGES,
    TEMPLATE_SEED,
)
from basemap.round0251_estimator_table import (
    COUPLING_CELL_ID,
    COUPLING_METRIC,
    COUPLING_STATEMENT,
    NOT_INDEPENDENT_EVIDENCE,
    PRE_REGISTERED_CRITERIA,
    REGISTERS_NOTHING,
    TABLE_CAPABILITY,
)
from basemap.round0251_rescore import (
    RESCORED_SEED,
    RESCORE_CAPABILITY,
    SCORER_DRIFT_TOLERANCE,
    SCORING_RELEASE_GENERATIONS,
)
from basemap.round0251_trainer_setup import (
    DECLARED_TRAINER_POLL_SITES,
    NODE_SETUP_SITES,
    POT_THRESHOLD_QUANTILE,
    SETUP_CAPABILITY,
    SETUP_RUNG_UPDATES,
    TAIL_RUNG_UPDATES,
    TAIL_TARGET_HOURS,
    declared_sites_match_the_release,
)
from experiments.round0251_nodes import (
    RESCORE_ACTION,
    TABLE_ACTION,
    TRAINSETUP_ACTION,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ID = "0251"
ROUND_ROOT = "/data/latent-basemap/runs/round-0251"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0251-2026-08-11.md")

R0216_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    f"{R0216_CAPABILITY}"
)
GRAPH_MANIFEST = os.path.join(R0216_ARTIFACTS, "substrate-graph.json")
R0217_ARTIFACTS = "/data/latent-basemap/runs/round-0217/queue-correction-1/artifacts"
R0218_PANEL = (
    f"/data/latent-basemap/runs/round-0218/queue/artifacts/{PANEL_CAPABILITY}/"
    "seed-family-panel.json"
)
R0228_COMPARISON = (
    "/data/latent-basemap/runs/round-0228/queue/artifacts/"
    "minilm-mixed-2m-cluster-spill-graph-map-comparison-v1/"
    "cluster-spill-graph-map-comparison.json"
)
R0234_GATE = (
    "/data/latent-basemap/runs/round-0234/queue/artifacts/"
    "minilm-mixed-2m-calibrated-robust-floors-n13-v1/"
    "minilm-calibrated-robust-floors-n13.json"
)
R0250_ARTIFACTS = "/data/latent-basemap/runs/round-0250/queue/artifacts"
R0250_PANEL_N16 = os.path.join(
    R0250_ARTIFACTS, PANEL_CAPABILITY_N16, "seed-family-panel-n16.json"
)
R0250_GATE = os.path.join(
    R0250_ARTIFACTS, R0250_GATE_CAPABILITY, "minilm-calibrated-robust-floors-n16.json"
)

#: R0250 measured 17.44 s for a three-cell panel node and 98.11 s for a gate
#: node that ran `calibrate` twice at 4,000,000 families. This round scores one
#: cell, calibrates the same two, and runs three short fits totalling roughly
#: 600 + 600 + 10,000 updates at ~114 upd/s plus three cold setups. The cap is
#: the round's registered 2.0 GPU-h, deliberately far above the estimate.
GPU_HOURS_CAP = 2.0
TRAINSETUP_P90_WALL_S = 1_800.0
RESCORE_P90_WALL_S = 900.0
TABLE_P90_WALL_S = 1_800.0


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0251 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0251 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _upstream_review_state(required: list[str]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    contingent: list[str] = []
    for round_id in required:
        reviews = []
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))):
            frontmatter = _frontmatter(path)
            reviews.append({
                "file": os.path.basename(path),
                "status": frontmatter.get("status"),
                "sha256": expected_input_signature(path)["sha256"],
            })
        accepted = [item for item in reviews if item["status"] == "accepted"]
        state[round_id] = {
            "reviews_present": reviews,
            "accepted_reviews": len(accepted),
        }
        if not accepted:
            contingent.append(round_id)
    return {
        "required_reviews": list(required),
        "by_round": state,
        "rounds_without_an_accepted_review": contingent,
        "claims_contingent_on": contingent,
    }


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0251 release checkout differs from requested release")
    basetemp = "/data/tmp/pytest-r0251-smoke"
    tmpdir = "/data/tmp/pytest-r0251-smoke-tmp"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--basetemp={basetemp}",
        "tests/test_round0251_contract.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": tmpdir,
    })
    os.makedirs(tmpdir, exist_ok=True)
    started = time.monotonic()
    #: **No `timeout=` anywhere in this file, deliberately.** CPython implements
    #: `subprocess.run(..., timeout=N)` as `Popen.kill()`, i.e. a hidden SIGKILL,
    #: and `plan-minilm-100m-v2.md` makes purging that construct binding before
    #: any further GPU round. A contract test greps this file for it.
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0251 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return prompt_contract.seal({
        "schema": "round0251-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cuda_hidden": True,
        "returncode": completed.returncode,
        "wall_seconds": wall,
        "basetemp": basetemp,
        "tmpdir": tmpdir,
        "stdout_tail": completed.stdout.strip().splitlines()[-5:],
        "poll_sites": declared_sites_match_the_release(),
    })


def _sealed_graph() -> tuple[dict[str, Any], dict[str, Any], int]:
    signature = expected_input_signature(GRAPH_MANIFEST)
    manifest = prompt_contract.read_sealed(
        GRAPH_MANIFEST, label="R0216 sealed substrate+graph receipt"
    )
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != GRAPH_SOURCE_ROUND_ID
        or manifest.get("capability") != GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
    ):
        raise RuntimeError("R0251 sealed R0216 substrate+graph contract changed")
    edges = int(manifest.get("directed_edge_count", 0))
    if edges != SEALED_DIRECTED_EDGES:
        raise RuntimeError(
            f"R0251 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    return signature, manifest, edges


def _sealed_seed42() -> dict[str, Any]:
    """R0217's archived seed-42 receipt and checkpoint, verified at prepare time."""
    capability = r0217_capability_for_seed(RESCORED_SEED)
    path = os.path.join(R0217_ARTIFACTS, capability, "train-receipt.json")
    receipt = prompt_contract.read_sealed(path, label="R0217 seed-42 train receipt")
    if (
        receipt.get("schema") != R0217_TRAIN_SCHEMA
        or receipt.get("round_id") != "0217"
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != RESCORED_SEED
    ):
        raise RuntimeError("R0217 seed-42 train receipt contract changed")
    model = dict(receipt["model"])
    if expected_input_signature(model["canonical_path"]) != model:
        raise RuntimeError("R0217 seed-42 checkpoint bytes changed")
    return {
        "capability": capability,
        "receipt": expected_input_signature(path),
        "model": model,
    }


def _sealed_panels() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """R0218's frozen panel and R0250's sealed sixteen-cell pooled table."""
    inputs: list[dict[str, Any]] = []
    panel = prompt_contract.read_sealed(R0218_PANEL, label="R0218 four-seed panel")
    if str(panel["high_d_reference_key"]) != REFERENCE_KEY:
        raise RuntimeError("R0218's reference identity is not the registered one")
    inputs.append(expected_input_signature(R0218_PANEL))
    reference = dict(panel["shared_high_d_reference"])
    if expected_input_signature(reference["canonical_path"]) != reference:
        raise RuntimeError("R0218's published high-D reference bytes changed")
    inputs.append(reference)
    for entry in dict(panel.get("centroids") or {}).values():
        centroid = dict(entry)
        if expected_input_signature(centroid["canonical_path"]) != centroid:
            raise RuntimeError("R0218 published centroid bytes changed")
        inputs.append(centroid)
    pooled = prompt_contract.read_sealed(
        R0250_PANEL_N16, label="R0250 sealed sixteen-cell panel"
    )
    if int(pooled.get("n", -1)) != len(POOLED_SEEDS) or str(
        pooled.get("high_d_reference_key")
    ) != REFERENCE_KEY:
        raise RuntimeError("R0250's sixteen-cell panel contract changed")
    inputs.append(expected_input_signature(R0250_PANEL_N16))
    return inputs, pooled


def _sealed_table_inputs() -> list[dict[str, Any]]:
    signatures = []
    for path, label in (
        (R0228_COMPARISON, "R0228 sealed cluster-spill comparison"),
        (R0234_GATE, "R0234 sealed n=13 calibrated gate"),
        (R0250_GATE, "R0250 sealed n=16 calibrated gate"),
    ):
        prompt_contract.read_sealed(path, label=label)
        signatures.append(expected_input_signature(path))
    return signatures


def prepare_round0251(
    *,
    release_sha: str,
    queue_root: str = QUEUE_ROOT,
    only_nodes: tuple[str, ...] | None = None,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0251 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    graph_manifest_signature, graph_manifest, edges = _sealed_graph()
    substrate_signature = dict(graph_manifest["substrate"])
    graph_signature = dict(graph_manifest["graph"])
    provenance_signature = dict(graph_manifest["provenance"])
    seed42 = _sealed_seed42()
    panel_inputs, pooled = _sealed_panels()
    table_inputs = _sealed_table_inputs()
    review_state = _upstream_review_state(list(required_reviews))
    sites = declared_sites_match_the_release()

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0251 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)

    shared_inputs = _dedupe([
        round_signature,
        graph_manifest_signature,
        substrate_signature,
        graph_signature,
        provenance_signature,
        expected_input_signature(smoke_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    p90: dict[str, float] = {}

    trainsetup_node = "trainsetup_0251"
    jobs.append({
        "id": trainsetup_node,
        "action": TRAINSETUP_ACTION,
        "handler_module": "experiments.round0251_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, SETUP_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{trainsetup_node}.done.json"),
        "expected_inputs": list(shared_inputs),
        "p90_wall_s": TRAINSETUP_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "template_seed": TEMPLATE_SEED,
        "setup_rung_updates": SETUP_RUNG_UPDATES,
        "tail_rung_updates": TAIL_RUNG_UPDATES,
        "declared_poll_sites": list(DECLARED_TRAINER_POLL_SITES),
        "node_setup_sites": list(NODE_SETUP_SITES),
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": True,
            "training_performed": True,
            "cpu_heavy": False,
        },
    })
    p90[trainsetup_node] = TRAINSETUP_P90_WALL_S

    rescore_node = "rescore_seed42_0251"
    jobs.append({
        "id": rescore_node,
        "action": RESCORE_ACTION,
        "handler_module": "experiments.round0251_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, RESCORE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{rescore_node}.done.json"),
        "expected_inputs": _dedupe([
            *shared_inputs, *panel_inputs, seed42["receipt"], seed42["model"],
        ]),
        "p90_wall_s": RESCORE_P90_WALL_S,
        "graph_manifest": GRAPH_MANIFEST,
        "graph_manifest_signature": graph_manifest_signature,
        "panel_evidence": R0218_PANEL,
        "panel_n16": expected_input_signature(R0250_PANEL_N16),
        "rescored_seed": RESCORED_SEED,
        "rescored_capability": seed42["capability"],
        "scorer_drift_tolerance": SCORER_DRIFT_TOLERANCE,
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": False,
        },
    })
    p90[rescore_node] = RESCORE_P90_WALL_S

    table_node = "estimator_table_0251"
    jobs.append({
        "id": table_node,
        "action": TABLE_ACTION,
        "handler_module": "experiments.round0251_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, TABLE_CAPABILITY)],
        "done_marker": os.path.join(artifacts, f"{table_node}.done.json"),
        "expected_inputs": _dedupe([
            *shared_inputs,
            expected_input_signature(R0250_PANEL_N16),
            *table_inputs,
        ]),
        "p90_wall_s": TABLE_P90_WALL_S,
        "panel_n16": expected_input_signature(R0250_PANEL_N16),
        "r0228_comparison": table_inputs[0],
        "r0234_gate": table_inputs[1],
        "r0250_gate": table_inputs[2],
        "upstream_review_state": review_state,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
    })
    p90[table_node] = TABLE_P90_WALL_S

    if only_nodes is not None:
        wanted = set(only_nodes)
        unknown = wanted - {job["id"] for job in jobs}
        if unknown:
            raise RuntimeError(f"R0251 has no node(s) {sorted(unknown)}")
        jobs = [job for job in jobs if job["id"] in wanted]
        for job in jobs:
            missing = set(job["deps"]) - wanted
            if missing:
                raise RuntimeError(
                    f"R0251 correction node {job['id']} depends on a node the "
                    "correction queue does not carry"
                )
        p90 = {key: value for key, value in p90.items() if key in wanted}
    p90["total"] = sum(value for key, value in p90.items() if key != "total")

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0251-setup-tail-rescore-and-coupling-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            GRAPH_CAPABILITY,
            PANEL_CAPABILITY,
            PANEL_CAPABILITY_N16,
            "minilm-mixed-2m-cluster-spill-graph-map-comparison-v1",
            "minilm-mixed-2m-calibrated-robust-floors-n13-v1",
            R0250_GATE_CAPABILITY,
            seed42["capability"],
        ],
        "capabilities_produced": [
            job_capability
            for job_id, job_capability in (
                (trainsetup_node, SETUP_CAPABILITY),
                (rescore_node, RESCORE_CAPABILITY),
                (table_node, TABLE_CAPABILITY),
            )
            if only_nodes is None or job_id in set(only_nodes)
        ],
        "correction_of": (None if only_nodes is None else QUEUE_ROOT),
        "correction_nodes": (None if only_nodes is None else list(only_nodes)),
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": p90,
        "scientific_contract": {
            "question_a": (
                "with a cooperative-abort read installed in basemap/pumap/ itself "
                "at five declared sites, what is the trainer's widest abort-read "
                "gap across SETUP and steady state separately, against the "
                "registered ceiling of "
                f"{registered_value('r0246_max_poll_spacing_s')} s; is the setup "
                "gap below that ceiling with margin, or by how much is it not; "
                "and what does the whole per-batch gap series say about the "
                f"widest gap over the {TAIL_TARGET_HOURS} h node's ~410x more "
                "batches, under both an extreme-value model and a "
                "distribution-free bound?"
            ),
            "question_b": (
                "does R0218's archived seed-42 checkpoint, rescored on this "
                "release against the same frozen panel, reproduce every value "
                "R0218 published for it -- and therefore are the sixteen pooled "
                "cells one population on the MAP side as well as on the "
                "map-independent reference side, or is the n=16 gate fitted "
                "across a mixed scorer?"
            ),
            "question_c": (
                "fitted at n = 16 on the same sixteen cells, which of the six "
                f"candidate estimators' floors fail {COUPLING_CELL_ID} on "
                f"{COUPLING_METRIC}, and how does that column line up against "
                "each candidate's coverage, false-fail rate, detection power, "
                "invariance depth and n=13 qualification?"
            ),
            "population": "sealed R0216 2,000,000-row mixed MiniLM substrate",
            "sealed_directed_edges": edges,
            "pooled_seed_family": list(POOLED_SEEDS),
            "n_pooled": len(POOLED_SEEDS),
            "n_exact_at_the_gate": N_EXACT,
            "rescored_seed": RESCORED_SEED,
            "scoring_release_generations": list(SCORING_RELEASE_GENERATIONS),
            "scorer_drift_tolerance": SCORER_DRIFT_TOLERANCE,
            "declared_trainer_poll_sites": list(DECLARED_TRAINER_POLL_SITES),
            "node_setup_sites": list(NODE_SETUP_SITES),
            "poll_sites_match_the_release_class": bool(sites["sites_match"]),
            "setup_rung_updates": SETUP_RUNG_UPDATES,
            "tail_rung_updates": TAIL_RUNG_UPDATES,
            "tail_projection_target_hours": TAIL_TARGET_HOURS,
            "pot_threshold_quantile": POT_THRESHOLD_QUANTILE,
            "registered_max_poll_spacing_s": registered_value(
                "r0246_max_poll_spacing_s"
            ),
            "coupling_cell": COUPLING_CELL_ID,
            "coupling_metric": COUPLING_METRIC,
            "coupling_statement": COUPLING_STATEMENT,
            "the_c8_seed42_reproduction_is_not_independent_evidence": (
                NOT_INDEPENDENT_EVIDENCE
            ),
            "pre_registered_dominance_criteria": [
                dict(item) for item in PRE_REGISTERED_CRITERIA
            ],
            "registers_nothing": REGISTERS_NOTHING,
            "gate_registered": False,
            "floors_registered": 0,
            "registry_fingerprint": registry_fingerprint(),
            "registry_mutated": False,
            "guard_modules_edited": False,
            "science_modules_edited": False,
            "trainer_module_edited": (
                "basemap/pumap/parametric_umap/core.py, additive only: one "
                "`abort_poll` attribute defaulting to None, five site constants, "
                "one `_poll_abort` method, and five calls to it. No existing line "
                "is modified and no science path changes when no hook is "
                "installed."
            ),
            "upstream_review_state": review_state,
            "evaluation_performed": True,
            "production_or_publishing": False,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument("--only-nodes", default=None)
    args = parser.parse_args(argv)
    only = (
        tuple(item.strip() for item in args.only_nodes.split(",") if item.strip())
        if args.only_nodes
        else None
    )
    print(json.dumps({
        "queue_manifest": prepare_round0251(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
            only_nodes=only,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
