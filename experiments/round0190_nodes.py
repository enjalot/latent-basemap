"""Execute the CPU-only R0190 three-seed boundary synthesis."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0190_three_seed_boundary_synthesis import (
    CAPABILITY,
    GATE_METRICS,
    ROUND_ID,
    SCHEMA,
    Round0190Error,
    synthesize,
)
from basemap import round0113_prompt_contrast as prompt_contract


def _read(path: str, *, label: str) -> dict[str, Any]:
    try:
        return prompt_contract.read_sealed(path, label=label)
    except Exception as error:
        raise Round0190Error(f"{label} is missing or unsealed") from error


def _signature(path: str, *, label: str) -> dict[str, Any]:
    try:
        return expected_input_signature(path)
    except Exception as error:
        raise Round0190Error(f"{label} signature is unavailable") from error


def _gate_view(receipt: Mapping[str, Any]) -> dict[str, float]:
    primary = receipt.get("primary_metrics") or {}
    diagnostic = receipt.get("diagnostic_metrics") or {}
    return {
        "pile_ffr": float(primary["pile_ffr"]),
        "density_v2": float(diagnostic["mixed_density"]),
        "ffr": float(primary["mixed_ffr"]),
        "purity_fidelity_k256": float(primary["mixed_purity_fidelity_k256"]),
        "purity_fidelity_k1024": float(primary["mixed_purity_fidelity_k1024"]),
        "projection_ffr": float(diagnostic["mixed_projection_ffr"]),
        "heldout_recall_at_10": float(primary["pile_ood_recall_at_10"]),
    }


def run_synthesis(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0190Error("handler received another round")
    expected_reviews = job.get("review_signatures") or {}
    if set(expected_reviews) != {"0187", "0188", "0189"}:
        raise Round0190Error("review binding set changed")
    review_signatures = {
        round_id: _signature(str(signature["canonical_path"]), label=f"R{round_id} review")
        for round_id, signature in expected_reviews.items()
    }
    if review_signatures != expected_reviews:
        raise Round0190Error("accepted review bytes changed")

    decisions = {
        round_id: _read(str(path), label=f"R{round_id} decision")
        for round_id, path in (job.get("decision_paths") or {}).items()
    }
    if (
        set(decisions) != {"0187", "0188", "0189"}
        or (decisions["0187"].get("decision") or {}).get("outcome")
        != "composition-controlled-size-regression"
        or (decisions["0188"].get("decision") or {}).get("outcome")
        != "composition-controlled-size-regression-not-replicated"
        or (decisions["0189"].get("decision") or {}).get("outcome")
        != "composition-controlled-size-regression-seed44-positive"
    ):
        raise Round0190Error("accepted three-seed decision lineage changed")

    expected_cells = {
        "seed42_quarter": ("0187", "quarter", None),
        "seed42_half": ("0187", "half", None),
        "seed42_full": ("0187", "full", None),
        "seed43_half": ("0188", "half", 43),
        "seed43_full": ("0188", "full", 43),
        "seed44_half": ("0189", "half", 44),
        "seed44_full": ("0189", "full", 44),
    }
    paths = job.get("evaluation_paths") or {}
    if set(paths) != set(expected_cells):
        raise Round0190Error("evaluation cell set changed")
    views: dict[str, dict[str, float]] = {}
    evaluation_signatures: dict[str, dict[str, Any]] = {}
    for cell, (round_id, rung, seed) in expected_cells.items():
        path = str(paths[cell])
        receipt = _read(path, label=f"{cell} evaluation")
        if (
            receipt.get("round_id") != round_id
            or receipt.get("rung") != rung
            or (seed is not None and int(receipt.get("seed", -1)) != seed)
            or not all((receipt.get("execution_checks") or {}).values())
        ):
            raise Round0190Error(f"{cell} evaluation contract changed")
        views[cell] = _gate_view(receipt)
        evaluation_signatures[cell] = _signature(path, label=f"{cell} evaluation")

    gate_path = str(job["r0161_gate_path"])
    gate = _read(gate_path, label="accepted R0161 prompted gates")
    if (
        gate.get("schema") != "round0161-prompted-universe-quality-gates-v1"
        or gate.get("round_id") != "0161"
        or gate.get("registered") is not True
        or set(gate.get("gates") or {}) != set(GATE_METRICS)
    ):
        raise Round0190Error("R0161 gate contract changed")
    family_path = str(job["r0160_family_path"])
    family = _read(family_path, label="accepted R0160 FineWeb family")
    seed42 = (family.get("cells") or {}).get("seed42") or {}
    fineweb_seed42 = seed42.get("decision_metrics") or {}
    if (
        family.get("schema") != "round0160-prompted-four-seed-family-evidence-v1"
        or family.get("round_id") != "0160"
        or set(fineweb_seed42) != set(GATE_METRICS)
    ):
        raise Round0190Error("R0160 FineWeb seed-42 cell changed")

    cells = {
        f"seed{seed}": {
            rung: views[f"seed{seed}_{rung}"] for rung in ("half", "full")
        }
        for seed in (42, 43, 44)
    }
    decision = synthesize(
        cells=cells,
        quarter_seed42=views["seed42_quarter"],
        fineweb_seed42={metric: float(fineweb_seed42[metric]) for metric in GATE_METRICS},
        gates=gate["gates"],
    )
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0190 three-seed synthesis"
    )
    receipt = prompt_contract.seal({
        "schema": SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY],
        "decision": decision,
        "lineage": {
            "accepted_reviews": review_signatures,
            "decisions": {
                key: _signature(str(path), label=f"R{key} decision")
                for key, path in (job.get("decision_paths") or {}).items()
            },
            "evaluations": evaluation_signatures,
            "r0161_gates": _signature(gate_path, label="R0161 gates"),
            "r0160_fineweb_family": _signature(family_path, label="R0160 family"),
        },
        "training_performed": False,
        "gpu_used": False,
    })
    atomic_write_new_json(
        os.path.join(output, "three-seed-boundary-synthesis.json"),
        receipt,
        immutable=True,
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if job.get("action") != "synthesize_three_seed_boundary":
        raise Round0190Error(f"R0190 does not authorize action {job.get('action')!r}")
    run_synthesis(active, job)


__all__ = ["run_job"]
