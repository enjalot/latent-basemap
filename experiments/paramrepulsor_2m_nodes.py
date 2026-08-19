"""Run the pinned upstream ParamRepulsor comparison on the sealed 2M substrate."""
from __future__ import annotations

import gc
import json
import math
import os
import random
import resource
import statistics
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap import paramrepulsor_baseline as P
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import atomic_save_new_npy, atomic_write_new_json, create_fresh_directory
from basemap.round0238_rung5 import json_safe
from basemap.round0242_locality import json_scrub
from basemap.round0247_registry import registry_fingerprint
from basemap.round0253_stop_hooks import install_stop_hooks
from experiments import round0218_nodes
from experiments.round0265_nodes import (
    K_TRUE,
    N_PROBES,
    _bound_path,
    _intra_queue_signature,
    _load_centroids,
    _start_node,
    score_one_map,
)


TRAIN_ACTION = "train_upstream_paramrepulsor_2m"
PANEL_ACTION = "score_upstream_paramrepulsor_2m_panel"
COMPARE_ACTION = "compare_upstream_paramrepulsor_to_fneg_family"

TRAIN_SCHEMA = "baseline-upstream-paramrepulsor-2m-train-receipt-v1"
PANEL_SCHEMA = "baseline-upstream-paramrepulsor-2m-panel-v1"
COMPARE_SCHEMA = "baseline-upstream-paramrepulsor-vs-fneg-family-v1"
PANEL_CAPABILITY = "minilm-mixed-2m-upstream-paramrepulsor-panel-v1"
COMPARE_CAPABILITY = "minilm-mixed-2m-upstream-paramrepulsor-vs-fneg-family-v1"

SAFETY_NOTE = (
    "the node calls the authors' pinned ParamRepulsor package directly. Upstream fit has "
    "no cooperative abort callback, so the runner's abort flag can be checked only before "
    "and after fit. Upstream also creates one PyTorch DataLoader worker for each inference "
    "call; the receipts count those workers explicitly."
)


class ParamRepulsorNodeError(RuntimeError):
    """The upstream baseline execution contract changed."""


def _receipt_envelope(
    manifest: Mapping[str, Any], *, child_processes_launched: int
) -> dict[str, Any]:
    return {
        "round_id": P.ROUND_ID,
        "study_id": P.STUDY_ID,
        "release_sha": str(manifest["release_sha"]),
        "registry_fingerprint": registry_fingerprint(),
        "safety_note": SAFETY_NOTE,
        "cuvs_calls": 0,
        "child_processes_launched": int(child_processes_launched),
        "child_process_accounting": (
            "one upstream PyTorch DataLoader worker per completed inference call; the "
            "adapter launches no other child and sends no signal directly"
        ),
        "signals_sent_directly_by_adapter": 0,
    }


def _seal(output: str, name: str, body: Mapping[str, Any]) -> None:
    atomic_write_new_json(
        os.path.join(output, name),
        prompt_contract.seal(json_safe(json_scrub(dict(body)))),
        immutable=True,
    )


def _seed(job: Mapping[str, Any]) -> int:
    seed = job.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in P.SEEDS:
        raise ParamRepulsorNodeError(f"invalid ParamRepulsor seed {seed!r}")
    if str(job.get("capability") or "") != P.capability_for_seed(seed):
        raise ParamRepulsorNodeError("ParamRepulsor capability does not match its seed")
    return seed


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="paramrepulsor_2m_nodes.run_train")
    import torch

    if active.get("manifest", {}).get("round_id") != P.ROUND_ID:
        raise ParamRepulsorNodeError("ParamRepulsor train received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise ParamRepulsorNodeError("ParamRepulsor train requires a leased CUDA device")
    seed = _seed(job)
    capability = P.capability_for_seed(seed)
    label = f"upstream ParamRepulsor 2M seed {seed}"
    abort_flag = _start_node(label)
    environment = P.verify_upstream_environment()
    if not torch.cuda.is_available():
        raise ParamRepulsorNodeError("pinned ParamRepulsor environment cannot see CUDA")

    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    recipe = P.recipe(seed)
    P.assert_registered_recipe(recipe)
    invariant = P.seed_invariant_sha256(recipe)
    if invariant != str(job.get("seed_invariant_sha256") or ""):
        raise ParamRepulsorNodeError("ParamRepulsor seed-invariant digest changed")

    output = create_fresh_directory(str(job["outputs"][0]), label=label)
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": P.RECIPE_SCHEMA,
            "round_id": P.ROUND_ID,
            "study_id": P.STUDY_ID,
            "seed": seed,
            "capability": capability,
            "recipe": recipe,
            "recipe_sha256": sha256_recipe(recipe),
            "seed_invariant_sha256": invariant,
            "upstream_environment": environment,
        },
        immutable=True,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    reducer = P.new_reducer(recipe)
    if str(reducer.device) != "cuda":
        raise ParamRepulsorNodeError(f"upstream estimator selected {reducer.device}, not cuda")
    started = time.monotonic()
    coordinates = np.asarray(reducer.fit_transform(source), dtype=np.float32)
    train_wall_s = time.monotonic() - started
    if coordinates.shape != (P.ROWS, 2) or not np.isfinite(coordinates).all():
        raise ParamRepulsorNodeError("upstream ParamRepulsor returned an invalid 2M map")

    coordinates_path = os.path.join(output, "coordinates.npy")
    atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
    coordinates_digest = ordered_array_sha256(coordinates)
    checkpoint_path = os.path.join(output, "paramrepulsor.pt")
    P.save_checkpoint(
        reducer,
        checkpoint_path,
        recipe_value=recipe,
        environment=environment,
    )
    model_parameters = sum(parameter.numel() for parameter in reducer.model.parameters())
    projector_shape = (
        list(reducer._projector.components_.shape)
        if reducer._projector is not None
        else None
    )
    memory = {
        "device_name": torch.cuda.get_device_name("cuda"),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
    }
    del reducer, coordinates
    torch.cuda.empty_cache()
    gc.collect()

    checks = {
        "pinned_upstream_commit_verified": environment["commit"] == P.UPSTREAM_COMMIT,
        "upstream_source_closure_verified": environment["every_source_digest_matches"] is True,
        "upstream_defaults_verified": P.assert_registered_recipe(recipe) == recipe,
        "same_2m_substrate": sealed["ordered_substrate_sha256"]
        == str(job["ordered_substrate_sha256"]),
        "coordinates_are_finite_2d": True,
        "default_pca_was_fit": projector_shape == [100, P.DIMENSION],
        "checkpoint_published": os.path.isfile(checkpoint_path),
    }
    if not all(checks.values()):
        raise ParamRepulsorNodeError(f"ParamRepulsor train checks failed: {checks}")
    body = {
        **_receipt_envelope(active["manifest"], child_processes_launched=1),
        "schema": TRAIN_SCHEMA,
        "capability": capability,
        "training_seed": seed,
        "training_performed": True,
        "evaluation_performed": False,
        "abort_flag_precondition": abort_flag,
        "recipe": recipe,
        "recipe_sha256": sha256_recipe(recipe),
        "seed_invariant_sha256": invariant,
        "production_config": expected_input_signature(config_path),
        "upstream_environment": environment,
        "model": expected_input_signature(checkpoint_path),
        "coordinates": expected_input_signature(coordinates_path),
        "coordinates_ordered_sha256": coordinates_digest,
        "substrate": sealed["substrate_signature"],
        "graph_manifest": sealed["manifest_signature"],
        "ordered_substrate_sha256": sealed["ordered_substrate_sha256"],
        "rows": P.ROWS,
        "dimension": P.DIMENSION,
        "model_parameters": int(model_parameters),
        "pca_components_shape": projector_shape,
        "train_wall_s": train_wall_s,
        "memory": memory,
        "abortability": {
            "checked_before_fit": True,
            "cooperative_poll_inside_upstream_fit": False,
            "why": "the unmodified upstream estimator exposes no fit callback",
        },
        "upstream_process_model": {
            "training_loader": "in-process FastDataloader",
            "inference_calls": 1,
            "workers_per_inference": int(recipe["estimator"]["num_workers"]),
        },
        "train_checks": checks,
        "gate_registerable_here": False,
        "map_decision_made": False,
        "node": str(active.get("node_id") or TRAIN_ACTION),
    }
    _seal(output, "train-receipt.json", body)
    print(json.dumps({"capability": capability, "seed": seed, "wall_s": train_wall_s}))


def sha256_recipe(value: Mapping[str, Any]) -> str:
    from basemap.artifact_identity import canonical_json, sha256_bytes

    return sha256_bytes(canonical_json(value))


def _authenticate_cell(
    cell: Mapping[str, Any], sealed: Mapping[str, Any]
) -> dict[str, Any]:
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in P.SEEDS:
        raise ParamRepulsorNodeError(f"invalid ParamRepulsor panel seed {seed!r}")
    capability = P.capability_for_seed(seed)
    if str(cell.get("capability") or "") != capability:
        raise ParamRepulsorNodeError("ParamRepulsor panel capability changed")
    path, signature = _intra_queue_signature(
        cell["train_receipt"], label=f"ParamRepulsor seed {seed} train receipt"
    )
    receipt = prompt_contract.read_sealed(path, label=f"ParamRepulsor seed {seed} receipt")
    checks = dict(receipt.get("train_checks") or {})
    if (
        receipt.get("schema") != TRAIN_SCHEMA
        or receipt.get("round_id") != P.ROUND_ID
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != seed
        or (receipt.get("upstream_environment") or {}).get("commit") != P.UPSTREAM_COMMIT
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise ParamRepulsorNodeError(f"ParamRepulsor seed {seed} receipt changed")
    if receipt.get("ordered_substrate_sha256") != sealed["ordered_substrate_sha256"]:
        raise ParamRepulsorNodeError(f"ParamRepulsor seed {seed} used another substrate")
    return {
        "seed": seed,
        "capability": capability,
        "receipt": receipt,
        "receipt_signature": signature,
        "model_path": prompt_contract.verify_signature(
            receipt["model"], label=f"ParamRepulsor seed {seed} checkpoint"
        ),
        "coordinates_path": prompt_contract.verify_signature(
            receipt["coordinates"], label=f"ParamRepulsor seed {seed} coordinates"
        ),
    }


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="paramrepulsor_2m_nodes.run_panel")
    import torch
    from basemap.panel_v2 import load_hiD_reference, reset_process_cuda_peak, score_panel

    if active.get("manifest", {}).get("round_id") != P.ROUND_ID:
        raise ParamRepulsorNodeError("ParamRepulsor panel received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise ParamRepulsorNodeError("ParamRepulsor panel requires a leased CUDA device")
    label = "upstream ParamRepulsor 2M panel"
    abort_flag = _start_node(label)
    environment = P.verify_upstream_environment()
    if not torch.cuda.is_available():
        raise ParamRepulsorNodeError("ParamRepulsor panel environment cannot see CUDA")
    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    panel_evidence = prompt_contract.read_sealed(
        str(job["panel_evidence"]), label="R0218 frozen 2M panel"
    )
    centroid_ks = [int(value) for value in job["centroid_ks"]]
    centroids, centroid_signatures = _load_centroids(panel_evidence, centroid_ks)
    reference_signature = dict(panel_evidence["shared_high_d_reference"])
    reference = load_hiD_reference(
        prompt_contract.verify_signature(reference_signature, label="R0218 high-D reference")
    )
    reference_identity = {
        "data_identity": {
            "kind": "ordered_array",
            "shape": [P.ROWS, P.DIMENSION],
            "dtype": np.dtype("<f4").str,
            "sha256": sealed["ordered_substrate_sha256"],
        },
        "convention": dict(round0218_nodes.REFERENCE_CONVENTION),
    }
    probes = np.asarray(
        np.load(
            _bound_path(job, "heldout_probes", label="R0265 held-out probes"),
            allow_pickle=False,
        ),
        dtype=np.float32,
    )
    truth = np.asarray(
        np.load(
            _bound_path(job, "heldout_truth", label="R0265 held-out truth"),
            allow_pickle=False,
        ),
        dtype=np.int64,
    )
    if probes.shape != (N_PROBES, P.DIMENSION) or truth.shape != (N_PROBES, K_TRUE):
        raise ParamRepulsorNodeError("R0265 held-out instrument geometry changed")

    cells_in = job.get("cells")
    if not isinstance(cells_in, list) or not cells_in:
        raise ParamRepulsorNodeError("ParamRepulsor panel has no cells")
    authenticated = [_authenticate_cell(cell, sealed) for cell in cells_in]
    seeds = sorted(entry["seed"] for entry in authenticated)
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset(P.SEEDS):
        raise ParamRepulsorNodeError("ParamRepulsor panel seed set changed")
    invariants = {entry["receipt"]["seed_invariant_sha256"] for entry in authenticated}
    if len(invariants) != 1:
        raise ParamRepulsorNodeError("ParamRepulsor panel mixes recipes")

    output = create_fresh_directory(str(job["outputs"][0]), label=label)
    reset_process_cuda_peak()
    started = time.monotonic()
    cfg = prompt_contract.panel_config()
    cells: dict[str, dict[str, Any]] = {}
    for entry in sorted(authenticated, key=lambda value: value["seed"]):
        seed = entry["seed"]
        coordinates = np.load(entry["coordinates_path"], mmap_mode="r", allow_pickle=False)
        if coordinates.shape != (P.ROWS, 2) or coordinates.dtype != np.float32:
            raise ParamRepulsorNodeError(f"ParamRepulsor seed {seed} coordinates changed")
        if not np.isfinite(coordinates).all():
            raise ParamRepulsorNodeError(f"ParamRepulsor seed {seed} coordinates are nonfinite")
        panel = score_panel(
            source,
            coordinates,
            config=cfg,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=reference_identity,
            provenance={
                "round_id": P.ROUND_ID,
                "study_id": P.STUDY_ID,
                "seed": seed,
                "capability": entry["capability"],
                "method": "upstream ParamRepulsor",
                "upstream_commit": P.UPSTREAM_COMMIT,
            },
        )
        purity = {
            "k256": float(panel["purity"]["k256"]),
            "k1024": float(panel["purity"]["k1024"]),
        }
        reducer, checkpoint_recipe = P.load_checkpoint(entry["model_path"], device="cuda")
        if int(checkpoint_recipe["seed"]) != seed:
            raise ParamRepulsorNodeError(f"ParamRepulsor seed {seed} checkpoint changed")
        placed = np.asarray(reducer.transform(probes), dtype=np.float32)
        metrics = score_one_map(
            coordinates=coordinates,
            probes_placed=placed,
            truth_top10=truth,
            purity_ratios=purity,
        )
        cells[str(seed)] = {
            "seed": seed,
            "capability": entry["capability"],
            "train_receipt": entry["receipt_signature"],
            "model": entry["receipt"]["model"],
            "coordinates": entry["receipt"]["coordinates"],
            "coordinates_ordered_sha256": entry["receipt"][
                "coordinates_ordered_sha256"
            ],
            "metrics": metrics,
            "panel_purity_numerators": panel.get("purity_numerators"),
        }
        del reducer, coordinates, placed
        torch.cuda.empty_cache()
        gc.collect()

    metric_table = {
        str(seed): {
            key: cells[str(seed)]["metrics"][key]
            for key in (
                "heldout_ffr",
                "regressor_ffr",
                "net_minus_regressor",
                "purity_fidelity_k256",
                "purity_fidelity_k1024",
                "collapse",
                "fog",
                "resolution_levels",
                "degenerate",
            )
        }
        for seed in seeds
    }
    checks = {
        "every_requested_seed_scored": set(cells) == {str(seed) for seed in seeds},
        "one_recipe": len(invariants) == 1,
        "pinned_upstream_environment_reverified": environment["commit"]
        == P.UPSTREAM_COMMIT,
        "every_primary_metric_finite": all(
            math.isfinite(float(row[key]))
            for row in metric_table.values()
            for key in (
                "heldout_ffr",
                "purity_fidelity_k256",
                "purity_fidelity_k1024",
                "collapse",
                "fog",
            )
        ),
        "result_is_descriptive_not_a_gate": job.get("gate_registerable_here") is False,
    }
    if not all(checks.values()):
        raise ParamRepulsorNodeError(f"ParamRepulsor panel checks failed: {checks}")
    body = {
        **_receipt_envelope(
            active["manifest"], child_processes_launched=len(seeds)
        ),
        "schema": PANEL_SCHEMA,
        "capability": PANEL_CAPABILITY,
        "node": str(active.get("node_id") or PANEL_ACTION),
        "abort_flag_precondition": abort_flag,
        "n": len(seeds),
        "seeds": seeds,
        "method": "upstream ParamRepulsor",
        "upstream_repository": P.UPSTREAM_REPOSITORY,
        "upstream_commit": P.UPSTREAM_COMMIT,
        "upstream_environment": environment,
        "seed_invariant_sha256": next(iter(invariants)),
        "panel_metric_table": metric_table,
        "cells": cells,
        "centroids": centroid_signatures,
        "shared_high_d_reference": reference_signature,
        "heldout_instrument": {
            "probes": dict(job["heldout_probes"]),
            "truth": dict(job["heldout_truth"]),
            "disc": int(P.ROWS * 0.001),
        },
        "lineage": {
            "substrate": sealed["substrate_signature"],
            "graph_manifest": sealed["manifest_signature"],
            "ordered_substrate_sha256": sealed["ordered_substrate_sha256"],
        },
        "execution_checks": checks,
        "training_performed": False,
        "evaluation_performed": True,
        "gate_registered": False,
        "gate_registerable_here": False,
        "peak_device_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "performance": {"node_wall_s": time.monotonic() - started},
        "upstream_process_model": {
            "inference_calls": len(seeds),
            "workers_per_inference": 1,
        },
    }
    _seal(output, "paramrepulsor-2m-panel.json", body)
    print(json.dumps({"capability": PANEL_CAPABILITY, "seeds": seeds}))


def run_compare(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="paramrepulsor_2m_nodes.run_compare")
    if active.get("manifest", {}).get("round_id") != P.ROUND_ID:
        raise ParamRepulsorNodeError("ParamRepulsor comparison received another queue")
    abort_flag = _start_node("upstream ParamRepulsor comparison")
    upstream = prompt_contract.read_sealed(
        _bound_path(job, "upstream_panel", label="upstream ParamRepulsor panel"),
        label="upstream ParamRepulsor panel",
    )
    family = prompt_contract.read_sealed(
        _bound_path(job, "fneg_panel", label="R0265 fneg family panel"),
        label="R0265 fneg family panel",
    )
    if (
        upstream.get("schema") != PANEL_SCHEMA
        or upstream.get("upstream_commit") != P.UPSTREAM_COMMIT
    ):
        raise ParamRepulsorNodeError("comparison received another upstream panel")
    family_table = dict(family.get("panel_metric_table") or {})
    if int(family.get("n", -1)) != 13 or len(family_table) != 13:
        raise ParamRepulsorNodeError("comparison requires the sealed R0265 n=13 panel")
    upstream_table = dict(upstream["panel_metric_table"])
    metrics = (
        "heldout_ffr",
        "purity_fidelity_k256",
        "purity_fidelity_k1024",
        "collapse",
        "fog",
    )
    family_median = {
        metric: statistics.median(float(row[metric]) for row in family_table.values())
        for metric in metrics
    }
    rows: dict[str, Any] = {}
    for seed, observed in sorted(upstream_table.items(), key=lambda item: int(item[0])):
        paired = family_table.get(seed)
        rows[seed] = {
            metric: {
                "paramrepulsor": float(observed[metric]),
                "fneg_same_seed": float(paired[metric]) if paired is not None else None,
                "fneg_family_median": family_median[metric],
                "paramrepulsor_minus_family_median": (
                    float(observed[metric]) - family_median[metric]
                ),
            }
            for metric in metrics
        }
    output = create_fresh_directory(str(job["outputs"][0]), label="ParamRepulsor comparison")
    _seal(
        output,
        "paramrepulsor-vs-fneg-2m.json",
        {
            **_receipt_envelope(active["manifest"], child_processes_launched=0),
            "schema": COMPARE_SCHEMA,
            "capability": COMPARE_CAPABILITY,
            "abort_flag_precondition": abort_flag,
            "upstream_commit": P.UPSTREAM_COMMIT,
            "seeds": [int(seed) for seed in rows],
            "n": len(rows),
            "fneg_family_n": len(family_table),
            "fneg_family_median": family_median,
            "comparisons": rows,
            "difference_convention": "ParamRepulsor - R0265 fneg family median",
            "interpretation": {
                "heldout_ffr": "higher is better",
                "purity_fidelity": "closer to 1 is better within the registered band",
                "collapse": "lower can indicate collapse; use the registered floor",
                "fog": "lower is better",
            },
            "method_ranking_claim": False,
            "decision_made": False,
            "gate_registered": False,
            "training_performed": False,
            "evaluation_performed": True,
            "upstream_panel": dict(job["upstream_panel"]),
            "fneg_panel": dict(job["fneg_panel"]),
        },
    )
    print(json.dumps({"capability": COMPARE_CAPABILITY, "seeds": list(rows)}))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="paramrepulsor_2m_nodes.run_job")
    action = str(job.get("action") or "")
    if action == TRAIN_ACTION:
        run_train(active, job)
    elif action == PANEL_ACTION:
        run_panel(active, job)
    elif action == COMPARE_ACTION:
        run_compare(active, job)
    else:
        raise ParamRepulsorNodeError(f"unknown ParamRepulsor action {action!r}")


__all__ = [
    "COMPARE_ACTION",
    "COMPARE_CAPABILITY",
    "PANEL_ACTION",
    "PANEL_CAPABILITY",
    "TRAIN_ACTION",
    "run_compare",
    "run_job",
    "run_panel",
    "run_train",
]
