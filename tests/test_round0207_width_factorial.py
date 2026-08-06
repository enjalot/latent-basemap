"""Decision tests for the width-by-N synthesis and U12 memo."""
from __future__ import annotations

import copy

from basemap.round0108_evaluation import validate_seal
from basemap.round0187_composition_nested_ladder import PRIMARY_METRICS, RUNG_ROWS
from basemap.round0202_h4096_nested_dose_ladder import TARGET_POSITIVE_DRAWS_PER_EDGE
from basemap.round0207_width_factorial import (
    FACTORIAL_CAPABILITY,
    MEMO_CAPABILITY,
    build_factorial,
    build_u12_design,
    render_factorial_markdown,
    render_u12_markdown,
)


def _ladder(width: str, pile: tuple[float, float, float]) -> dict:
    round_id = "0203" if width == "h2048" else "0202"
    schema = (
        "round0203-h2048-composition-nested-low-dose-summary-v1"
        if width == "h2048"
        else "round0202-h4096-composition-nested-ladder-summary-v1"
    )
    capability = (
        "jina-document-english-h2048-composition-nested-low-dose-ladder-v1"
        if width == "h2048"
        else "jina-document-english-h4096-composition-nested-dose-ladder-v1"
    )
    cells = {}
    for index, rung in enumerate(("quarter", "half", "full")):
        cells[rung] = {metric: 0.5 for metric in PRIMARY_METRICS}
        cells[rung]["pile_ffr"] = pile[index]
    retentions = {
        metric: {
            "half_over_quarter": cells["half"][metric] / cells["quarter"][metric],
            "full_over_half": cells["full"][metric] / cells["half"][metric],
            "full_over_quarter": cells["full"][metric] / cells["quarter"][metric],
        }
        for metric in PRIMARY_METRICS
    }
    rate = 100.0 if width == "h2048" else 33.0
    wall_multiple = 1.0 if width == "h2048" else 3.0
    economics = {}
    for index, rung in enumerate(("quarter", "half", "full"), start=1):
        economics[rung] = {
            "retained_rows": RUNG_ROWS[rung],
            "directed_edges": 100_000_000 * index,
            "successful_updates": 100_000 * index,
            "achieved_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "updates_per_s": rate,
            "train_wall_s": 1_000.0 * index * wall_multiple,
            "train_receipt": {
                "canonical_path": f"/test/{width}/{rung}.json",
                "kind": "file",
                "bytes": 1,
                "sha256": "a" * 64,
            },
        }
    return {
        "schema": schema,
        "round_id": round_id,
        "capabilities": [capability],
        "summary": {
            "cells": cells,
            "retentions": retentions,
            "registered_metric": "pile_ffr",
            "decision_deferred_to_track_a3": True,
        },
        "training_economics": economics,
        "scientific_scope": {
            "width": int(width.removeprefix("h")),
            "seed": 42,
            "rungs": ["quarter", "half", "full"],
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "primary_registered_metric": "pile_ffr",
            "density_role": "diagnostic-only",
        },
    }


def _diagnostics() -> dict:
    return {
        width: {
            rung: {
                "mixed_density": 0.2,
                "mixed_projection_ffr": 0.4,
                "corpus_density": {
                    "fineweb": 0.2,
                    "redpajama": 0.2,
                    "pile": 0.2,
                },
            }
            for rung in ("quarter", "half", "full")
        }
        for width in ("h2048", "h4096")
    }


def _contexts(ladders: dict) -> tuple[dict, dict, dict]:
    r0190 = {
        "schema": "round0190-three-seed-boundary-synthesis-v1",
        "round_id": "0190",
        "decision": {
            "width_null_noise_scale": {
                "metric": "pile_ffr",
                "source": "three-seed full-rung sample SD (ddof=1)",
                "value": 0.01,
            }
        },
    }
    r0191 = {
        "schema": "round0191-full-h4096-width-decision-v1",
        "round_id": "0191",
        "decision": {
            "registered_metric": "pile_ffr",
            "boundary_recovered": True,
            "r0184_h2048": copy.deepcopy(ladders["h2048"]["summary"]["cells"]["full"]),
            "h4096": copy.deepcopy(ladders["h4096"]["summary"]["cells"]["full"]),
            "full_rung_pile_ffr_delta_h4096_minus_h2048": (
                ladders["h4096"]["summary"]["cells"]["full"]["pile_ffr"]
                - ladders["h2048"]["summary"]["cells"]["full"]["pile_ffr"]
            ),
        },
    }
    r0201 = {
        "schema": "round0201-pile-boundary-loss-localization-v1",
        "round_id": "0201",
        "descriptive_pattern": "diffuse",
        "steering": "prioritize capacity/global-geometry explanations",
        "cluster_localization": {
            "256": {
                "losing_cluster_coverage": 0.98,
                "top_decile_negative_loss_mass_share": 0.24,
            }
        },
        "predictors": {
            "spearman": {
                "hubness_occurrence_vs_mean_delta": 0.0,
                "local_log_r2_r1_vs_mean_delta": 0.0,
                "mixture_centroid_distance_vs_mean_delta": 0.0,
            }
        },
    }
    return r0190, r0191, r0201


def _factorial(
    h2048: tuple[float, float, float], h4096: tuple[float, float, float]
) -> dict:
    ladders = {"h2048": _ladder("h2048", h2048), "h4096": _ladder("h4096", h4096)}
    r0190, r0191, r0201 = _contexts(ladders)
    return build_factorial(
        ladders=ladders,
        diagnostics=_diagnostics(),
        r0190=r0190,
        r0191=r0191,
        r0201=r0201,
    )


def test_width_flattens_branch_and_economics() -> None:
    value = _factorial((0.50, 0.49, 0.46), (0.50, 0.49, 0.48))
    assert value["capability"] == FACTORIAL_CAPABILITY
    assert value["outcome"] == "width-flattens-size-regression"
    assert value["future_scale_readout"]["selected_u12_hidden_dimension"] == 4096
    assert value["future_scale_readout"]["larger_n_green_light"] is True
    assert value["economics"]["full_rung_width_cost_multiple"] == 3.0
    assert "h4096" in render_factorial_markdown(value)
    validate_seal(value, label="test R0207 factorial")


def test_partial_and_no_compensation_branches_are_exhaustive() -> None:
    partial = _factorial((0.50, 0.47, 0.43), (0.50, 0.48, 0.46))
    assert partial["outcome"] == "capacity-partially-compensates"
    assert partial["future_scale_readout"]["selected_u12_hidden_dimension"] == 4096
    assert partial["future_scale_readout"]["larger_n_green_light"] is False

    no_effect = _factorial((0.50, 0.49, 0.47), (0.50, 0.47, 0.44))
    assert no_effect["outcome"] == "no-consistent-capacity-compensation"
    assert no_effect["future_scale_readout"]["selected_u12_hidden_dimension"] is None


def test_both_widths_flat_selects_cheaper_width() -> None:
    value = _factorial((0.50, 0.49, 0.48), (0.50, 0.49, 0.48))
    assert value["outcome"] == "both-widths-flat-at-low-dose"
    assert value["future_scale_readout"]["selected_u12_hidden_dimension"] == 2048


def test_u12_memo_scales_graph_and_splits_h4096_train() -> None:
    factorial = _factorial((0.50, 0.49, 0.46), (0.50, 0.49, 0.48))
    u12 = {
        "schema": "round0168-prompted-diverse-u12-staging-v1",
        "round_id": "0168",
        "rows": 12_474_331,
        "dimension": 768,
        "dtype": "<f2",
        "embedding_convention": "Document: ",
        "graph_built": False,
        "host_fp16": {
            "canonical_path": "/test/u12.npy",
            "kind": "file",
            "bytes": 1,
            "sha256": "b" * 64,
        },
        "duplicate_control": {
            "summary": {"duplicate_copy_rows_excluded": 1048}
        },
    }
    graph = {
        "schema": "round0171-prompted-8m-fuzzy-graph-v1",
        "round_id": "0171",
        "retained_rows": 7_952_419,
        "directed_edge_count": 603_086_368,
        "performance": {"total_wall_s": 932.5969},
    }
    audit = {
        "schema": "round0173-prompted-ood-training-disjoint-v1",
        "round_id": "0173",
        "passed": False,
        "exact_training_family_overlap_count": 5,
    }
    memo = build_u12_design(
        factorial=factorial,
        u12_manifest=u12,
        graph_precedent=graph,
        ood_audit=audit,
    )
    assert memo["capability"] == MEMO_CAPABILITY
    assert memo["selected_hidden_dimension"] == 4096
    assert memo["graph_plan"]["expected_shards"] == 4
    assert memo["train_plan"]["queue_split_required"] is True
    assert len(memo["train_plan"]["stages"]) >= 2
    assert "not GPU launch authority" in render_u12_markdown(memo)
    validate_seal(memo, label="test R0207 U12 memo")
