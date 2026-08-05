"""Pure owner-facing v0 release proposal synthesis for R0195."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any


ROUND_ID = "0195"
CAPABILITY = "jina-fineweb-2m-v0-release-proposal-v1"
CANDIDATE_ID = "basemap-jina-v5-nano-en-2m-v0"
METRICS = (
    "density_v2",
    "ffr",
    "heldout_recall_at_10",
    "projection_ffr",
    "purity_fidelity_k1024",
    "purity_fidelity_k256",
)


class Round0195Error(RuntimeError):
    """The accepted evidence needed for the proposal changed."""


def build_proposal(
    family: Mapping[str, Any],
    gates: Mapping[str, Any],
    universality: Mapping[str, Any],
    methods: Mapping[str, Any],
    scale: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        family.get("schema") != "round0160-prompted-four-seed-family-evidence-v1"
        or family.get("seeds") != [42, 43, 44, 45]
        or (family.get("population") or {}).get("rows") != 1_993_761
        or (family.get("population") or {}).get("embedding_convention")
        != "Document: "
    ):
        raise Round0195Error("R0160 prompted family contract changed")
    if (
        gates.get("schema") != "round0161-prompted-universe-quality-gates-v1"
        or gates.get("registered") is not True
        or gates.get("seed_family") != [42, 43, 44, 45]
    ):
        raise Round0195Error("R0161 gate contract changed")
    cells = family.get("cells") or {}
    gate_cells = gates.get("gates") or {}
    if set(cells) != {"seed42", "seed43", "seed44", "seed45"}:
        raise Round0195Error("R0160 seed cells changed")
    if set(gate_cells) != set(METRICS):
        raise Round0195Error("R0161 metric set changed")
    qualification: dict[str, Any] = {}
    for name in sorted(cells):
        observed = cells[name].get("decision_metrics") or {}
        metric_cells = {
            metric: {
                "observed": float(observed[metric]),
                "floor": float(gate_cells[metric]["floor"]),
                "pass": float(observed[metric]) >= float(gate_cells[metric]["floor"]),
            }
            for metric in METRICS
        }
        qualification[name] = {
            "seed": int(cells[name]["seed"]),
            "all_six_pass": all(cell["pass"] for cell in metric_cells.values()),
            "metrics": metric_cells,
            "coordinates": dict(cells[name]["coordinates"]),
            "train_receipt": dict(cells[name]["train_receipt"]),
        }
    if not all(cell["all_six_pass"] for cell in qualification.values()):
        raise Round0195Error("the four-seed candidate family does not pass its gates")

    if (
        universality.get("schema")
        != "round0182-universality-readout-packet-v1"
        or universality.get("diagnostic_only") is not True
    ):
        raise Round0195Error("R0182 OOD readout contract changed")
    ood: dict[str, Any] = {}
    for map_name in ("r0115-prompted-2m-seed42", "r0117-prompted-2m-seed43"):
        rows = [row["maps"][map_name] for row in universality["rows"]]
        counts = Counter(str(row["verdict"]) for row in rows)
        ordered = sorted(float(row["ffr_retention"]) for row in rows)
        ood[map_name] = {
            "probe_count": len(rows),
            "verdict_counts": dict(sorted(counts.items())),
            "median_ffr_retention": ordered[len(ordered) // 2],
        }

    if methods.get("schema") != "round0183-heldout-projection-method-table-v1":
        raise Round0195Error("R0183 method table contract changed")
    method_2m = (methods.get("rows") or {}).get("2m") or {}
    parametric = method_2m.get("corrected_parametric_standard_curve_seed42") or {}
    aumap = method_2m.get("aumap_inverse_distance_k15") or {}
    if float(parametric.get("ffr", -1)) != 0.5785 or float(aumap.get("ffr", -1)) != 0.53973:
        raise Round0195Error("R0183 2M FFR context changed")
    scale_decision = scale.get("decision") or {}
    if scale_decision.get("outcome") != "confirmed-2-of-3-seed-sensitive":
        raise Round0195Error("R0190 scale verdict changed")

    return {
        "schema": "round0195-fineweb-2m-v0-release-proposal-v1",
        "round_id": ROUND_ID,
        "capabilities": [CAPABILITY],
        "candidate_id": CANDIDATE_ID,
        "candidate_scope": {
            "rows": 1_993_761,
            "dimension": 768,
            "embedding_convention": "Document: ",
            "corpus": "FineWeb English frozen R0113 population",
            "canonical_seed": 42,
            "canonical_coordinates": qualification["seed42"]["coordinates"],
            "canonical_train_receipt": qualification["seed42"]["train_receipt"],
        },
        "qualification": {
            "family_seeds": [42, 43, 44, 45],
            "all_four_seeds_pass_all_six_commensurate_gates": True,
            "cells": qualification,
        },
        "method_context": {
            "corrected_parametric_2m_ffr": float(parametric["ffr"]),
            "aumap_2m_ffr": float(aumap["ffr"]),
            "ffr_difference": float(parametric["ffr"]) - float(aumap["ffr"]),
            "candidate_specific_contrast": False,
            "interpretation": (
                "R0183 is historical same-scale method context, not evidence that "
                "this prompted candidate itself causally beats aUMAP"
            ),
        },
        "ood_caveats": {
            "coverage": "11 probes on prompted seed42 and seed43 only",
            "maps": ood,
            "universal_quality_claim": False,
            "known_named_failures_must_ship_with_candidate": True,
        },
        "scale_limitations": {
            "mixed_english_boundary_verdict": scale_decision["outcome"],
            "pile_full_rung_regression_seed_count": 2,
            "seed_count": 3,
            "transfer_to_fineweb_only_candidate_claimed": False,
        },
        "proposal": {
            "recommendation": "owner-go-no-go-required",
            "registry_promotion_performed": False,
            "production_or_publishing_performed": False,
            "sae_readiness_claimed": False,
            "if_approved_next_action": (
                "separate registry-promotion round binding this packet and the "
                "canonical seed42 map artifacts"
            ),
        },
        "training_performed": False,
    }


__all__ = ["CAPABILITY", "CANDIDATE_ID", "METRICS", "ROUND_ID", "Round0195Error", "build_proposal"]
