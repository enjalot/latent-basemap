#!/usr/bin/env python3
"""Materialize, but never launch, the conditional R0147 row-policy queue."""
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
from basemap.jina_historical_selection import (
    HISTORICAL_CORPORA,
    IndexedInventoryFp16Array,
    derive_first_eligible_historical_rows,
    load_historical_provenance,
)
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0027_program import CENTROIDS, SOURCE_4M_PATH, TRAIN_PATH
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST,
    RESTORATION_FLOORS,
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
    metric_view,
)
from basemap.round0142_jina_universality import COMMON_CORPUS_ROWS
from basemap.round0147_row_policy import (
    CAPABILITY,
    ROUND_ID,
    ROWS,
    TREATMENT,
    build_decision,
    treatment_train_config,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0138_queue import (
    _accepted_review,
    _frontmatter,
)
from experiments.round0147_nodes import training_accounting_mismatches


ROUND_ROOT = "/data/latent-basemap/runs/round-0147"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0147-*.md")
INVENTORY_PATH = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/"
    "jina-diverse-25m-inventory/jina-diverse-25m-inventory-v1.json"
)
ELIGIBILITY_PATH = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/"
    "jina-diverse-25m-inventory/jina-diverse-25m-eligibility-v1.npz"
)
HISTORICAL_PROVENANCE = "/data/latent-basemap/jina-en-8m/provenance.npz"
R0037_SHARED = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/"
    "shared-reference/receipt.json"
)
R0140_PANEL = (
    "/data/latent-basemap/runs/round-0140/queue-attempt-2/artifacts/"
    "functional-panel/functional-bisection.json"
)
R0140_DECISION = (
    "/data/latent-basemap/runs/round-0140/queue-attempt-2/artifacts/"
    "decision/decision.json"
)
R0140_CONTROL_TRAIN = (
    "/data/latent-basemap/runs/round-0140/queue/artifacts/"
    "current_graph_current_host/train/train-receipt.json"
)
R0142_ROOT = "/data/latent-basemap/runs/round-0142/queue/artifacts"
CONTROL = (
    "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-"
    "jina-v5-nano-heldout/train/data-00000.npy"
)
DADABASE = "/data/embeddings/dadabase/jina-v5-nano.npy"
DADABASE_TEXTS = "/data/embeddings/dadabase/jokes.parquet"
BEIR = {
    name: {
        "corpus": f"/data/embeddings/beir/{name}-pooled-jina-v5-nano/corpus_vectors.npy",
        "queries": f"/data/embeddings/beir/{name}-pooled-jina-v5-nano/query_vectors.npy",
        "corpus_ids": f"/data/embeddings/beir/{name}-pooled-jina-v5-nano/corpus_ids.json",
        "query_ids": f"/data/embeddings/beir/{name}-pooled-jina-v5-nano/query_ids.json",
    }
    for name in ("scifact", "trec-covid")
}

REVIEW_CAPABILITIES = {
    "0087": "jina-diverse-25m-inventory-v1",
    "0140": "jina-2m-subsystem-bisection-v1",
    "0142": "jina-diverse-universality-panel-v1",
}

GPU_HOURS_MINIMUM = 1.25
GPU_HOURS_EXPECTED = 1.50
GPU_HOURS_P90 = 1.82
GPU_HOURS_MAXIMUM = 2.50


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _read_sealed(path: str, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    value = _read_json(path)
    validate_seal(value, label=label)
    return value, signature


def _issued_round(release_sha: str) -> tuple[str, dict[str, Any]]:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0147 requires exactly one issued round; found {len(candidates)}"
        )
    if _frontmatter(candidates[0]).get("base_commit") != release_sha:
        raise RuntimeError("R0147 issued base_commit differs from release")
    return candidates[0], expected_input_signature(candidates[0])


def _inventory_source_signatures(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    historical_datasets = {dataset for _key, dataset in HISTORICAL_CORPORA}
    ranges = (inventory.get("selection") or {}).get("ranges")
    if not isinstance(ranges, list):
        raise RuntimeError("R0087 inventory ranges are absent")
    signatures: list[dict[str, Any]] = []
    for item in ranges:
        if item.get("dataset") not in historical_datasets:
            continue
        shard = item.get("shard") or {}
        signatures.append({
            "canonical_path": os.path.realpath(str(shard.get("canonical_path") or "")),
            "kind": "file",
            "bytes": int(shard.get("bytes", -1)),
            "sha256": str(shard.get("sha256") or ""),
        })
    signatures = _dedupe(signatures)
    if len(signatures) != 39 or any(
        not os.path.isfile(item["canonical_path"])
        or os.path.getsize(item["canonical_path"]) != item["bytes"]
        or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        for item in signatures
    ):
        raise RuntimeError("R0147 historical inventory shard census changed")
    return signatures


def _inventory_bundle() -> tuple[
    dict[str, Any],
    dict[str, Any],
    np.ndarray,
    list[dict[str, Any]],
]:
    inventory, inventory_signature = _read_sealed(
        INVENTORY_PATH, label="R0087 diverse inventory"
    )
    if (
        inventory.get("round_id") != "0087"
        or inventory.get("capability") != "jina-diverse-25m-inventory-v1"
        or inventory.get("capability_ready") is not True
        or (inventory.get("selection") or {}).get("selected_rows") != 25_000_000
    ):
        raise RuntimeError("R0087 inventory capability changed")
    eligibility_signature = expected_input_signature(ELIGIBILITY_PATH)
    with np.load(ELIGIBILITY_PATH, allow_pickle=False) as archive:
        if set(archive.files) != {
            "duplicate_representative_rows",
            "duplicate_excluded_rows",
            "excluded_rows",
            "family_counts",
            "family_offsets",
            "member_rows",
            "nonfinite_rows",
            "representative_rows",
            "zero_rows",
        }:
            raise RuntimeError("R0087 eligibility members changed")
        excluded = np.asarray(archive["excluded_rows"], dtype=np.int64)
        duplicate_excluded = np.asarray(
            archive["duplicate_excluded_rows"], dtype=np.int64
        )
        zero = np.asarray(archive["zero_rows"], dtype=np.int64)
        nonfinite = np.asarray(archive["nonfinite_rows"], dtype=np.int64)
    if (
        len(excluded) != 51_337
        or not np.array_equal(excluded, duplicate_excluded)
        or len(zero)
        or len(nonfinite)
        or np.any(excluded[1:] <= excluded[:-1])
    ):
        raise RuntimeError("R0087 eligibility semantics changed")
    return (
        inventory,
        inventory_signature,
        excluded,
        [eligibility_signature, *_inventory_source_signatures(inventory)],
    )


def _accepted_activation() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    reviews: list[dict[str, Any]] = []
    for round_id, capability in REVIEW_CAPABILITIES.items():
        reviews.extend(_accepted_review(round_id, capability))

    r0140_inputs = _accepted_review("0140", REVIEW_CAPABILITIES["0140"])
    round_signature, result_signature, review_signature = r0140_inputs
    if (
        os.path.basename(result_signature["canonical_path"])
        != "result-0140-2026-08-01-01.md"
        or os.path.basename(review_signature["canonical_path"])
        != "review-0140-2026-08-01-01.md"
    ):
        raise RuntimeError("R0147 requires the exact accepted R0140 retry pair")

    decision, decision_signature = _read_sealed(
        R0140_DECISION, label="accepted R0140 decision"
    )
    panel, panel_signature = _read_sealed(
        R0140_PANEL, label="accepted R0140 functional panel"
    )
    control = (panel.get("cells") or {}).get(CURRENT_GRAPH_CURRENT_HOST)
    if (
        decision.get("round_id") != "0140"
        or decision.get("outcome")
        != "historical-row-universe-restores-with-current-trainer"
        or decision.get("next_action")
        != "recover-and-test-row-policy-on-current-population"
        or decision.get("panel") != panel_signature
        or not isinstance(control, Mapping)
        or any(
            metric_view(control)[key] < floor
            for key, floor in RESTORATION_FLOORS.items()
        )
    ):
        raise RuntimeError("R0140 accepted evidence does not activate R0147")
    return (
        reviews,
        decision_signature,
        panel_signature,
        dict(control),
    )


def _shared_reference() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shared, signature = _read_sealed(R0037_SHARED, label="R0037 shared reference")
    members = [signature]
    for key in ("high_d_reference", "query_truth", "query_embeddings"):
        actual = expected_input_signature(shared[key]["canonical_path"])
        if actual != shared[key]:
            raise RuntimeError(f"R0037 shared reference member changed: {key}")
        members.append(actual)
    return shared, members


def _r0140_control(
    control: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train, train_signature = _read_sealed(
        R0140_CONTROL_TRAIN, label="R0140 current-host control train"
    )
    model = expected_input_signature(train["model"]["canonical_path"])
    production = expected_input_signature(
        train["production_config"]["canonical_path"]
    )
    coordinates = expected_input_signature(control["coordinates"]["canonical_path"])
    if (
        train.get("round_id") != "0140"
        or train.get("cell") != CURRENT_GRAPH_CURRENT_HOST
        or train.get("exact_execution_receipt", {}).get("pipeline")
        != "host_weighted_jina_paired"
        or model != train["model"]
        or production != train["production_config"]
        or coordinates != control["coordinates"]
        or control.get("training", {}).get("train") != train_signature
        or control.get("training", {}).get("model") != model
    ):
        raise RuntimeError("R0140 restoring control lineage changed")
    return train, [train_signature, model, production, coordinates]


def _universality_inputs() -> tuple[
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, dict[str, Any]]],
    list[dict[str, Any]],
]:
    common_outputs: dict[str, str] = {}
    inputs: list[dict[str, Any]] = []
    for name in COMMON_CORPUS_ROWS:
        output = os.path.join(R0142_ROOT, f"common-{name}-raw-jina")
        receipt_path = os.path.join(output, "receipt.json")
        receipt, receipt_signature = _read_sealed(
            receipt_path, label=f"accepted R0142 {name} embeddings"
        )
        embedding = expected_input_signature(receipt["embedding"]["canonical_path"])
        if (
            receipt.get("round_id") != "0142"
            or receipt.get("probe") != name
            or receipt.get("prompt_applied") is not False
            or embedding != receipt["embedding"]
        ):
            raise RuntimeError(f"R0142 common probe changed: {name}")
        common_outputs[name] = output
        inputs.extend([receipt_signature, embedding])
    control = expected_input_signature(CONTROL)
    dadabase = expected_input_signature(DADABASE)
    dadabase_texts = expected_input_signature(DADABASE_TEXTS)
    beir = {
        name: {key: expected_input_signature(path) for key, path in paths.items()}
        for name, paths in BEIR.items()
    }
    inputs.extend([
        control,
        dadabase,
        dadabase_texts,
        *[
            signature
            for probe in beir.values()
            for signature in probe.values()
        ],
    ])
    return common_outputs, control, dadabase, dadabase_texts, beir, inputs


def _cpu_smoke(
    *,
    inventory: Mapping[str, Any],
    excluded: np.ndarray,
    model_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Exercise real row selection, model reload, accounting closure, and selector."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("R0147 CPU smoke requires CUDA_VISIBLE_DEVICES='' or '-1'")
    started = time.monotonic()
    provenance = load_historical_provenance(HISTORICAL_PROVENANCE)
    selected = derive_first_eligible_historical_rows(
        provenance, inventory, excluded, target_rows=ROWS
    )
    source = IndexedInventoryFp16Array(
        selected["arrays"]["global_rows"], inventory, dimension=768
    )
    sample = np.asarray(source[:32], dtype=np.float32)
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_signature["canonical_path"], device="cpu")
    coordinates = np.asarray(model.transform(sample, batch_size=16), dtype=np.float32)
    if coordinates.shape != (32, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("R0147 CPU smoke model transform failed")

    fake_graph = {
        "canonical_path": "/preflight/not-a-runtime-graph.npz",
        "kind": "file",
        "bytes": 1,
        "sha256": "1" * 64,
    }
    fake_manifest = {
        "canonical_path": "/preflight/not-a-runtime-manifest.json",
        "kind": "file",
        "bytes": 1,
        "sha256": "2" * 64,
    }
    config, config_sha = treatment_train_config(
        graph_signature=fake_graph,
        graph_manifest_signature=fake_manifest,
        graph_edges=123_456,
        source_sha256="3" * 64,
        selection_sha256="4" * 64,
    )
    batch_size = int(config["optimizer"]["batch_size"])
    expected_rows = SUCCESSFUL_UPDATES * batch_size
    runtime = {
        **config["execution"]["expected_pipeline_stamp"],
        "source_rows_gathered": expected_rows,
        "destination_rows_gathered": expected_rows,
        "host_prefetch_producer_batches": SUCCESSFUL_UPDATES + 1,
        "host_prefetch_consumer_batches": SUCCESSFUL_UPDATES,
    }
    accounting = {
        "lr_horizon": SUCCESSFUL_UPDATES,
        "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
        "scheduler_steps": SUCCESSFUL_UPDATES,
        "attempted_batches": SUCCESSFUL_UPDATES,
        "finite_loss_batches": SUCCESSFUL_UPDATES,
        "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
        "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": 123_456,
    }
    mismatches = training_accounting_mismatches(
        accounting=accounting,
        runtime=runtime,
        expected_pipeline=config["execution"]["expected_pipeline_stamp"],
        graph_edges=123_456,
        batch_size=batch_size,
        profiler={"aborted": False},
        rate=TRAIN_MINIMUM_UPDATES_PER_S + 1.0,
    )
    if mismatches:
        raise RuntimeError(f"R0147 CPU smoke accounting failed: {mismatches}")
    prototype_values = {key: value + 0.01 for key, value in RESTORATION_FLOORS.items()}
    prototype = {
        "panel": {
            "ffr": prototype_values["ffr"],
            "purity": {
                "k256": prototype_values["purity_fidelity_k256"],
                "k1024": prototype_values["purity_fidelity_k1024"],
            },
        },
        "projection": {
            "ffr": prototype_values["projection_ffr"],
            "recall_at_10": prototype_values["ood_recall_at_10"],
        },
    }
    decision = build_decision(
        {
            CURRENT_GRAPH_CURRENT_HOST: prototype,
            TREATMENT: prototype,
        },
        selection_summary=selected["summary"],
    )
    smoke = seal({
        "schema": "round0147-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scope": (
            "real selector -> reviewed control reload/transform -> production "
            "config -> post-fit accounting closure -> sealed selector"
        ),
        "model": dict(model_signature),
        "sample_rows": len(sample),
        "coordinates_finite": True,
        "selection_summary": selected["summary"],
        "training_config_sha256": config_sha,
        "accounting_mismatches": mismatches,
        "decision_outcome": decision["outcome"],
        "wall_seconds": time.monotonic() - started,
    })
    validate_seal(smoke, label="R0147 CPU smoke")
    return smoke


def _pytest_smoke(*, release_sha: str) -> dict[str, Any]:
    """Persist the exact release test slice used as preparation evidence."""
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0147 pytest checkout is not at the requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0147_nodes.py",
        "tests/test_round0147_row_policy.py",
        "tests/test_jina_historical_selection.py",
        "tests/test_round0142_jina_universality.py",
        "tests/test_round0140_subsystem_bisection.py",
        "tests/test_round0104_training.py",
        "tests/test_panel_v2.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
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
    receipt = seal({
        "schema": "round0147-release-pytest-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "pythondontwritebytecode": "1",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
    })
    validate_seal(receipt, label="R0147 release pytest")
    if completed.returncode != 0 or "60 passed" not in completed.stdout:
        raise RuntimeError(
            "R0147 release pytest failed:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def prepare_round0147(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0147 release SHA must be one full commit")
    round_path, round_signature = _issued_round(release_sha)
    reviews, decision_signature, panel_signature, control_cell = _accepted_activation()
    inventory, inventory_signature, excluded, inventory_inputs = _inventory_bundle()
    provenance_signature = expected_input_signature(HISTORICAL_PROVENANCE)
    shared, shared_inputs = _shared_reference()
    control_train, control_inputs = _r0140_control(control_cell)
    (
        common_outputs,
        ood_control,
        dadabase,
        dadabase_texts,
        beir,
        universality_inputs,
    ) = _universality_inputs()

    queue_root = create_fresh_directory(queue_root, label="R0147 row-policy queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    pytest_path = os.path.join(preflight, "release-pytest.json")
    atomic_write_new_json(
        pytest_path,
        _pytest_smoke(release_sha=release_sha),
        immutable=True,
    )
    pytest_signature = expected_input_signature(pytest_path)
    smoke_path = os.path.join(preflight, "cpu-smoke.json")
    atomic_write_new_json(
        smoke_path,
        _cpu_smoke(
            inventory=inventory,
            excluded=excluded,
            model_signature=control_train["model"],
        ),
        immutable=True,
    )
    smoke_signature = expected_input_signature(smoke_path)
    external_inputs = _dedupe([
        round_signature,
        *reviews,
        decision_signature,
        panel_signature,
        inventory_signature,
        *inventory_inputs,
        provenance_signature,
        expected_input_signature(TRAIN_PATH),
        expected_input_signature(SOURCE_4M_PATH),
        *shared_inputs,
        *control_inputs,
        *universality_inputs,
        *[
            expected_input_signature(item["path"])
            for item in CENTROIDS.values()
        ],
        pytest_signature,
        smoke_signature,
    ])

    selection_output = os.path.join(artifacts, "historical-eligibility-selection")
    graph_output = os.path.join(artifacts, "current-graph-eligible-historical")
    train_output = os.path.join(artifacts, TREATMENT, "train")
    functional_output = os.path.join(artifacts, "functional-row-policy-panel")
    control_universality = os.path.join(
        artifacts, f"universality-{CURRENT_GRAPH_CURRENT_HOST}"
    )
    treatment_universality = os.path.join(
        artifacts, f"universality-{TREATMENT}"
    )
    decision_output = os.path.join(artifacts, CAPABILITY)
    common_panel = {
        "source": expected_input_signature(TRAIN_PATH),
        "shared_reference_receipt": shared_inputs[0],
        "high_d_reference": dict(shared["high_d_reference"]),
        "query_truth": dict(shared["query_truth"]),
        "query_embeddings": dict(shared["query_embeddings"]),
        "centroids": {
            str(k): expected_input_signature(value["path"])
            for k, value in CENTROIDS.items()
        },
    }
    common_ood = {
        "common_outputs": common_outputs,
        "control_embeddings": ood_control,
        "dadabase": dadabase,
        "dadabase_texts": dadabase_texts,
        "beir": beir,
    }

    jobs: list[dict[str, Any]] = [{
        "id": "materialize_historical_eligibility_selection",
        "action": "materialize_selection",
        "handler_module": "experiments.round0147_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [selection_output],
        "done_marker": os.path.join(artifacts, "materialize-selection.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 120.0,
        "historical_provenance": provenance_signature,
        "inventory": inventory_signature,
        "eligibility": inventory_inputs[0],
        "node_policy": {"gpu_required": False, "training_performed": False},
    }, {
        "id": "score_universality_raw_historical_control",
        "action": "universality_panel",
        "map_key": CURRENT_GRAPH_CURRENT_HOST,
        "model": control_train["model"],
        **common_ood,
        "handler_module": "experiments.round0147_nodes",
        "handler_callable": "run_job",
        "deps": ["materialize_historical_eligibility_selection"],
        "outputs": [control_universality],
        "done_marker": os.path.join(artifacts, "universality-control.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 120.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "build_current_graph_eligible_historical",
        "action": "build_graph",
        "selection_output": selection_output,
        "handler_module": "experiments.round0147_nodes",
        "handler_callable": "run_job",
        "deps": ["score_universality_raw_historical_control"],
        "outputs": [graph_output],
        "done_marker": os.path.join(artifacts, "build-treatment-graph.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 600.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "train_eligible_historical_current_host",
        "action": "train",
        "selection_output": selection_output,
        "graph_output": graph_output,
        "handler_module": "experiments.round0147_nodes",
        "handler_callable": "run_job",
        "deps": ["build_current_graph_eligible_historical"],
        "outputs": [train_output],
        "done_marker": os.path.join(artifacts, "train-treatment.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 5_400.0,
        "node_policy": {"gpu_required": True, "training_performed": True},
    }, {
        "id": "score_functional_row_policy",
        "action": "functional_panel",
        "selection_output": selection_output,
        "train_output": train_output,
        "r0140_panel": panel_signature,
        **common_panel,
        "handler_module": "experiments.round0147_nodes",
        "handler_callable": "run_job",
        "deps": ["train_eligible_historical_current_host"],
        "outputs": [functional_output],
        "done_marker": os.path.join(artifacts, "functional-panel.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 300.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "score_universality_eligible_historical_treatment",
        "action": "universality_panel",
        "map_key": TREATMENT,
        "train_output": train_output,
        **common_ood,
        "handler_module": "experiments.round0147_nodes",
        "handler_callable": "run_job",
        "deps": ["score_functional_row_policy"],
        "outputs": [treatment_universality],
        "done_marker": os.path.join(artifacts, "universality-treatment.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 120.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "decide_historical_row_policy",
        "action": "decide",
        "selection_output": selection_output,
        "functional_output": functional_output,
        "universality_outputs": {
            CURRENT_GRAPH_CURRENT_HOST: control_universality,
            TREATMENT: treatment_universality,
        },
        "handler_module": "experiments.round0147_nodes",
        "handler_callable": "run_job",
        "deps": ["score_universality_eligible_historical_treatment"],
        "outputs": [decision_output],
        "done_marker": os.path.join(artifacts, "row-policy-decision.done.json"),
        "expected_inputs": external_inputs,
        "p90_wall_s": 60.0,
        "node_policy": {"gpu_required": False, "training_performed": False},
    }]

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
        "schema": "round0147-historical-row-policy-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": list(REVIEW_CAPABILITIES),
        "capability_dependencies": list(REVIEW_CAPABILITIES.values()),
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            str(job["id"]): float(job["p90_wall_s"])
            for job in jobs
            if job["node_policy"]["gpu_required"]
        },
        "scientific_contract": {
            "question": (
                "does the R0087 exact-family eligibility policy preserve the "
                "R0140 historical-row functional restoration?"
            ),
            "activation": {
                "r0140_decision": decision_signature,
                "required_outcome": (
                    "historical-row-universe-restores-with-current-trainer"
                ),
            },
            "control": {
                "cell": CURRENT_GRAPH_CURRENT_HOST,
                "population": "raw historical R0037 2M rows",
                "functional_panel": panel_signature,
                "model": control_train["model"],
            },
            "treatment": {
                "cell": TREATMENT,
                "population": (
                    "first 2M R0087-eligible rows in historical R0037 shuffle order"
                ),
                "size_preserving": True,
                "trainer": "current R0104 host weighted pipeline, seed 42",
                "graph": "current R0104 graph rebuilt on treatment population",
                "successful_updates": SUCCESSFUL_UPDATES,
            },
            "causal_scope": (
                "complete row-policy package including its induced graph change; "
                "not a pure duplicate-only intervention"
            ),
            "selector": {
                "metrics": list(RESTORATION_FLOORS),
                "floors": RESTORATION_FLOORS,
                "all_metrics_required": True,
                "density_diagnostic_only": True,
            },
            "universality": {
                "source": "accepted R0142 exact probe artifacts and split policy",
                "maps": [CURRENT_GRAPH_CURRENT_HOST, TREATMENT],
                "role": "paired diagnostic only; never selector input",
            },
            "claims_excluded": [
                "duplicate control caused the historical restoration",
                "diverse 25M transfer",
                "density floor change",
                "map registry or publication state change",
            ],
            "cpu_smoke": smoke_signature,
            "release_pytest": pytest_signature,
        },
    })
    queue["p90_gpu_seconds"]["total"] = sum(
        value for key, value in queue["p90_gpu_seconds"].items() if key != "total"
    )
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0147(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
