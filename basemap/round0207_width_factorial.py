"""Frozen decision logic for the width-by-N factorial and U12 design memo."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .round0108_evaluation import seal
from .round0187_composition_nested_ladder import PRIMARY_METRICS, RUNG_ROWS
from .round0202_h4096_nested_dose_ladder import TARGET_POSITIVE_DRAWS_PER_EDGE


ROUND_ID = "0207"
FACTORIAL_CAPABILITY = "jina-width-by-n-factorial-capacity-economics-v1"
MEMO_CAPABILITY = "jina-prompted-diverse-u12-next-rung-design-v1"
FACTORIAL_SCHEMA = "round0207-width-by-n-factorial-v1"
MEMO_SCHEMA = "round0207-prompted-diverse-u12-design-v1"
RETENTION_FLOOR = 0.97
WIDTHS = ("h2048", "h4096")
RUNGS = ("quarter", "half", "full")
STEPS = ("half_over_quarter", "full_over_half")
DIVERSE_ROWS = 12_474_331

LADDER_SCHEMA = {
    "h2048": "round0203-h2048-composition-nested-low-dose-summary-v1",
    "h4096": "round0202-h4096-composition-nested-ladder-summary-v1",
}
LADDER_ROUND = {"h2048": "0203", "h4096": "0202"}
LADDER_CAPABILITY = {
    "h2048": "jina-document-english-h2048-composition-nested-low-dose-ladder-v1",
    "h4096": "jina-document-english-h4096-composition-nested-dose-ladder-v1",
}


class Round0207Error(RuntimeError):
    """The preregistered width-factorial synthesis changed or is incomplete."""


def _positive(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise Round0207Error(f"{label} is not numeric") from error
    if not math.isfinite(number) or number <= 0.0:
        raise Round0207Error(f"{label} is not finite and positive")
    return number


def _validate_ladder(value: Mapping[str, Any], *, width: str) -> dict[str, Any]:
    if width not in WIDTHS:
        raise Round0207Error(f"unknown width {width!r}")
    scope = value.get("scientific_scope") or {}
    summary = value.get("summary") or {}
    cells = summary.get("cells") or {}
    retentions = summary.get("retentions") or {}
    economics = value.get("training_economics") or {}
    if (
        value.get("schema") != LADDER_SCHEMA[width]
        or value.get("round_id") != LADDER_ROUND[width]
        or value.get("capabilities") != [LADDER_CAPABILITY[width]]
        or scope.get("width") != int(width.removeprefix("h"))
        or scope.get("seed") != 42
        or scope.get("rungs") != list(RUNGS)
        or abs(
            float(scope.get("target_positive_draws_per_edge", -1.0))
            - TARGET_POSITIVE_DRAWS_PER_EDGE
        )
        > 1.0e-15
        or scope.get("primary_registered_metric") != "pile_ffr"
        or scope.get("density_role") != "diagnostic-only"
        or summary.get("registered_metric") != "pile_ffr"
        or summary.get("decision_deferred_to_track_a3") is not True
        or set(cells) != set(RUNGS)
        or set(economics) != set(RUNGS)
    ):
        raise Round0207Error(f"{width} ladder identity changed")

    normalized_cells: dict[str, dict[str, float]] = {}
    normalized_economics: dict[str, dict[str, Any]] = {}
    for rung in RUNGS:
        if set(cells[rung]) != set(PRIMARY_METRICS):
            raise Round0207Error(f"{width}/{rung} metric set changed")
        normalized_cells[rung] = {
            metric: _positive(cells[rung][metric], label=f"{width}/{rung}/{metric}")
            for metric in PRIMARY_METRICS
        }
        cell_economics = economics[rung]
        expected_rows = RUNG_ROWS[rung]
        if int(cell_economics.get("retained_rows", -1)) != expected_rows:
            raise Round0207Error(f"{width}/{rung} row count changed")
        normalized_economics[rung] = {
            "retained_rows": expected_rows,
            "directed_edges": int(
                _positive(
                    cell_economics.get("directed_edges"),
                    label=f"{width}/{rung}/directed_edges",
                )
            ),
            "successful_updates": int(
                _positive(
                    cell_economics.get("successful_updates"),
                    label=f"{width}/{rung}/successful_updates",
                )
            ),
            "achieved_positive_draws_per_edge": _positive(
                cell_economics.get("achieved_positive_draws_per_edge"),
                label=f"{width}/{rung}/dose",
            ),
            "updates_per_s": _positive(
                cell_economics.get("updates_per_s"),
                label=f"{width}/{rung}/updates_per_s",
            ),
            "train_wall_s": _positive(
                cell_economics.get("train_wall_s"),
                label=f"{width}/{rung}/train_wall_s",
            ),
            "train_receipt": dict(cell_economics.get("train_receipt") or {}),
        }
        if (
            abs(
                normalized_economics[rung]["achieved_positive_draws_per_edge"]
                - TARGET_POSITIVE_DRAWS_PER_EDGE
            )
            > 1.0e-6
        ):
            raise Round0207Error(f"{width}/{rung} dose changed")

    normalized_retentions: dict[str, dict[str, float]] = {}
    if set(retentions) != set(PRIMARY_METRICS):
        raise Round0207Error(f"{width} retention metric set changed")
    for metric in PRIMARY_METRICS:
        expected = {
            "half_over_quarter": normalized_cells["half"][metric]
            / normalized_cells["quarter"][metric],
            "full_over_half": normalized_cells["full"][metric]
            / normalized_cells["half"][metric],
            "full_over_quarter": normalized_cells["full"][metric]
            / normalized_cells["quarter"][metric],
        }
        observed = retentions[metric]
        if set(observed) != set(expected):
            raise Round0207Error(f"{width}/{metric} retention shape changed")
        for key, expected_value in expected.items():
            if not math.isclose(
                _positive(observed[key], label=f"{width}/{metric}/{key}"),
                expected_value,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise Round0207Error(f"{width}/{metric}/{key} arithmetic changed")
        normalized_retentions[metric] = expected
    return {
        "cells": normalized_cells,
        "retentions": normalized_retentions,
        "economics": normalized_economics,
    }


def _validate_diagnostics(
    value: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, dict[str, dict[str, Any]]]:
    if set(value) != set(WIDTHS):
        raise Round0207Error("diagnostic width set changed")
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for width in WIDTHS:
        if set(value[width]) != set(RUNGS):
            raise Round0207Error(f"{width} diagnostic rung set changed")
        output[width] = {}
        for rung in RUNGS:
            cell = value[width][rung]
            corpus = cell.get("corpus_density") or {}
            if set(corpus) != {"fineweb", "redpajama", "pile"}:
                raise Round0207Error(f"{width}/{rung} corpus density changed")
            output[width][rung] = {
                "mixed_density_v2": _positive(
                    cell.get("mixed_density"), label=f"{width}/{rung}/density"
                ),
                "mixed_projection_ffr": _positive(
                    cell.get("mixed_projection_ffr"),
                    label=f"{width}/{rung}/projection_ffr",
                ),
                "corpus_density_v2": {
                    key: _positive(
                        corpus[key], label=f"{width}/{rung}/{key}_density"
                    )
                    for key in ("fineweb", "redpajama", "pile")
                },
            }
    return output


def build_factorial(
    *,
    ladders: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    r0190: Mapping[str, Any],
    r0191: Mapping[str, Any],
    r0201: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exhaustive preregistered factorial/cost decision."""
    if set(ladders) != set(WIDTHS):
        raise Round0207Error("factorial ladder width set changed")
    normalized = {
        width: _validate_ladder(ladders[width], width=width) for width in WIDTHS
    }
    diagnostic_cells = _validate_diagnostics(diagnostics)

    noise = ((r0190.get("decision") or {}).get("width_null_noise_scale") or {})
    prior_width = r0191.get("decision") or {}
    if (
        r0190.get("schema") != "round0190-three-seed-boundary-synthesis-v1"
        or r0190.get("round_id") != "0190"
        or noise.get("metric") != "pile_ffr"
        or noise.get("source") != "three-seed full-rung sample SD (ddof=1)"
        or r0191.get("schema") != "round0191-full-h4096-width-decision-v1"
        or r0191.get("round_id") != "0191"
        or prior_width.get("registered_metric") != "pile_ffr"
        or prior_width.get("boundary_recovered") is not True
        or r0201.get("schema") != "round0201-pile-boundary-loss-localization-v1"
        or r0201.get("round_id") != "0201"
        or r0201.get("descriptive_pattern") != "diffuse"
        or r0201.get("steering") != "prioritize capacity/global-geometry explanations"
    ):
        raise Round0207Error("accepted capacity context changed")
    seed_noise = _positive(noise.get("value"), label="R0190 seed-noise scale")
    for width, key in (("h2048", "r0184_h2048"), ("h4096", "h4096")):
        if prior_width.get(key) != normalized[width]["cells"]["full"]:
            raise Round0207Error(f"R0191 {width} full endpoint changed")

    pile_steps = {
        width: {
            step: normalized[width]["retentions"]["pile_ffr"][step]
            for step in STEPS
        }
        for width in WIDTHS
    }
    passes = {
        width: {step: value >= RETENTION_FLOOR for step, value in steps.items()}
        for width, steps in pile_steps.items()
    }
    all_pass = {width: all(values.values()) for width, values in passes.items()}
    componentwise_improved = all(
        pile_steps["h4096"][step] >= pile_steps["h2048"][step]
        for step in STEPS
    ) and any(
        pile_steps["h4096"][step] > pile_steps["h2048"][step]
        for step in STEPS
    )
    if all_pass["h4096"] and not all_pass["h2048"]:
        outcome = "width-flattens-size-regression"
        selected_width = 4096
        larger_n_green_light = True
    elif all_pass["h4096"] and all_pass["h2048"]:
        outcome = "both-widths-flat-at-low-dose"
        selected_width = 2048
        larger_n_green_light = True
    elif not all_pass["h4096"] and componentwise_improved:
        outcome = "capacity-partially-compensates"
        selected_width = 4096
        larger_n_green_light = False
    else:
        outcome = "no-consistent-capacity-compensation"
        selected_width = None
        larger_n_green_light = False

    cells: dict[str, dict[str, Any]] = {}
    for width in WIDTHS:
        cells[width] = {}
        for rung in RUNGS:
            economics = normalized[width]["economics"][rung]
            gpu_hours = economics["train_wall_s"] / 3600.0
            cells[width][rung] = {
                "rows": economics["retained_rows"],
                "directed_edges": economics["directed_edges"],
                "successful_updates": economics["successful_updates"],
                "positive_draws_per_edge": economics[
                    "achieved_positive_draws_per_edge"
                ],
                "updates_per_s": economics["updates_per_s"],
                "train_wall_s": economics["train_wall_s"],
                "train_gpu_hours": gpu_hours,
                "primary_metrics": normalized[width]["cells"][rung],
                "diagnostic_metrics": diagnostic_cells[width][rung],
                "primary_metric_per_train_gpu_hour": {
                    metric: normalized[width]["cells"][rung][metric] / gpu_hours
                    for metric in PRIMARY_METRICS
                },
                "train_receipt": economics["train_receipt"],
            }

    contrasts: dict[str, Any] = {}
    for rung in RUNGS:
        h2 = cells["h2048"][rung]
        h4 = cells["h4096"][rung]
        extra_hours = h4["train_gpu_hours"] - h2["train_gpu_hours"]
        contrasts[rung] = {
            "h4096_over_h2048_train_wall": h4["train_wall_s"]
            / h2["train_wall_s"],
            "h4096_over_h2048_update_rate": h4["updates_per_s"]
            / h2["updates_per_s"],
            "extra_train_gpu_hours_h4096": extra_hours,
            "primary_metric_delta_h4096_minus_h2048": {
                metric: h4["primary_metrics"][metric]
                - h2["primary_metrics"][metric]
                for metric in PRIMARY_METRICS
            },
            "pile_ffr_delta_exceeds_r0190_seed_noise": abs(
                h4["primary_metrics"]["pile_ffr"]
                - h2["primary_metrics"]["pile_ffr"]
            )
            > seed_noise,
            "marginal_pile_ffr_per_extra_gpu_hour": (
                (
                    h4["primary_metrics"]["pile_ffr"]
                    - h2["primary_metrics"]["pile_ffr"]
                )
                / extra_hours
                if extra_hours > 0.0
                else None
            ),
        }

    localization = r0201.get("cluster_localization") or {}
    predictors = r0201.get("predictors") or {}
    output = {
        "schema": FACTORIAL_SCHEMA,
        "round_id": ROUND_ID,
        "capability": FACTORIAL_CAPABILITY,
        "outcome": outcome,
        "selector": {
            "registered_metric": "pile_ffr",
            "retention_floor": RETENTION_FLOOR,
            "h4096_flattens_rule": (
                "both h4096 step retentions >=0.97 while at least one h2048 "
                "step retention is <0.97"
            ),
            "partial_rule": (
                "h4096 still misses but improves both Pile FFR step retentions "
                "componentwise, with at least one strict improvement"
            ),
            "pile_step_retentions": pile_steps,
            "pile_step_passes": passes,
            "componentwise_improved": componentwise_improved,
        },
        "cells": cells,
        "retentions": {
            width: normalized[width]["retentions"] for width in WIDTHS
        },
        "width_contrasts": contrasts,
        "economics": {
            "new_quarter_half_train_gpu_hours": {
                width: sum(
                    cells[width][rung]["train_gpu_hours"]
                    for rung in ("quarter", "half")
                )
                for width in WIDTHS
            },
            "all_three_cell_train_gpu_hours": {
                width: sum(
                    cells[width][rung]["train_gpu_hours"] for rung in RUNGS
                )
                for width in WIDTHS
            },
            "full_rung_width_cost_multiple": contrasts["full"][
                "h4096_over_h2048_train_wall"
            ],
            "quality_per_gpu_hour_is_descriptive_not_a_gate": True,
        },
        "capacity_context": {
            "r0190_full_rung_pile_ffr_seed_noise_sd": seed_noise,
            "r0191_full_rung_width_delta": float(
                prior_width["full_rung_pile_ffr_delta_h4096_minus_h2048"]
            ),
            "r0201_loss_pattern": "diffuse",
            "r0201_k256_losing_cluster_coverage": float(
                localization["256"]["losing_cluster_coverage"]
            ),
            "r0201_k256_top_decile_loss_mass_share": float(
                localization["256"]["top_decile_negative_loss_mass_share"]
            ),
            "r0201_predictor_spearman": dict(predictors.get("spearman") or {}),
        },
        "future_scale_readout": {
            "selected_u12_hidden_dimension": selected_width,
            "larger_n_green_light": larger_n_green_light,
            "memo_only": True,
            "gpu_launch_authorized": False,
        },
        "claim_scope": (
            "paired seed-42 composition-controlled 2x3 factorial at fixed dose; "
            "density-v2 and projection FFR are diagnostic"
        ),
        "training_performed": False,
        "production_or_publishing": False,
    }
    return seal(output)


def build_u12_design(
    *,
    factorial: Mapping[str, Any],
    u12_manifest: Mapping[str, Any],
    graph_precedent: Mapping[str, Any],
    ood_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Turn the factorial branch into a bounded, non-launch U12 design."""
    if (
        factorial.get("schema") != FACTORIAL_SCHEMA
        or factorial.get("round_id") != ROUND_ID
        or factorial.get("capability") != FACTORIAL_CAPABILITY
        or u12_manifest.get("schema") != "round0168-prompted-diverse-u12-staging-v1"
        or u12_manifest.get("round_id") != "0168"
        or u12_manifest.get("rows") != DIVERSE_ROWS
        or u12_manifest.get("dimension") != 768
        or u12_manifest.get("dtype") != "<f2"
        or u12_manifest.get("embedding_convention") != "Document: "
        or u12_manifest.get("graph_built") is not False
        or graph_precedent.get("schema") != "round0171-prompted-8m-fuzzy-graph-v1"
        or graph_precedent.get("round_id") != "0171"
        or graph_precedent.get("retained_rows") != 7_952_419
        or graph_precedent.get("directed_edge_count") != 603_086_368
        or ood_audit.get("schema") != "round0173-prompted-ood-training-disjoint-v1"
        or ood_audit.get("round_id") != "0173"
        or ood_audit.get("passed") is not False
        or ood_audit.get("exact_training_family_overlap_count") != 5
    ):
        raise Round0207Error("U12 design inputs changed")
    selected_width = (factorial.get("future_scale_readout") or {}).get(
        "selected_u12_hidden_dimension"
    )
    precedent_rows = int(graph_precedent["retained_rows"])
    precedent_edges = int(graph_precedent["directed_edge_count"])
    estimated_edges = round(precedent_edges * DIVERSE_ROWS / precedent_rows)
    estimated_updates = math.ceil(
        1_000_000 * estimated_edges / precedent_edges
    )
    rate = None
    if selected_width in {2048, 4096}:
        width_key = f"h{selected_width}"
        rate = float(factorial["cells"][width_key]["full"]["updates_per_s"])
    estimated_train_hours = estimated_updates / rate / 3600.0 if rate else None
    max_stage_hours = 6.5
    stages = (
        max(1, math.ceil(estimated_train_hours / max_stage_hours))
        if estimated_train_hours is not None else 0
    )
    stage_targets = []
    for stage in range(stages):
        start = estimated_updates * stage // stages
        stop = estimated_updates * (stage + 1) // stages
        stage_targets.append({
            "stage": stage + 1,
            "start_successful_update": start,
            "stop_successful_update": stop,
            "updates": stop - start,
            "estimated_gpu_hours": (stop - start) / rate / 3600.0,
        })
    graph_seconds_per_row = float(graph_precedent["performance"]["total_wall_s"])
    graph_seconds_per_row /= precedent_rows
    duplicate_summary = (u12_manifest.get("duplicate_control") or {}).get(
        "summary"
    ) or {}
    output = {
        "schema": MEMO_SCHEMA,
        "round_id": ROUND_ID,
        "capability": MEMO_CAPABILITY,
        "status": "design-only",
        "factorial_outcome": factorial["outcome"],
        "selected_hidden_dimension": selected_width,
        "design_ready": selected_width is not None,
        "gpu_rounds_may_be_drafted": selected_width is not None,
        "launch_ready": False,
        "launch_authorized": False,
        "population": {
            "rows": DIVERSE_ROWS,
            "dimension": 768,
            "embedding_convention": "Document: ",
            "ordered_population": "exact accepted R0168/R0132 U12 compact order",
            "host_fp16": dict(u12_manifest["host_fp16"]),
            "exact_duplicate_copy_rows": int(
                duplicate_summary.get("duplicate_copy_rows_excluded", -1)
            ),
            "duplicate_policy": (
                "retain exact U12 population identity; multiplicity is metadata "
                "and never a sampler weight"
            ),
        },
        "graph_plan": {
            "method": (
                "fresh prompted-vector fuzzy k50 graph using one shared fp32 "
                "IVF8192 coarse quantizer, sequential <=4M-row GPU shards, "
                "search every shard, exact global similarity/id top-k merge"
            ),
            "shard_rows_maximum": 4_000_000,
            "expected_shards": math.ceil(DIVERSE_ROWS / 4_000_000),
            "qualification_nprobe_grid": [16, 32, 64, 128, 256],
            "mean_recall_at_49_floor": 0.90,
            "p10_recall_at_49_floor": 0.80,
            "estimated_directed_edges_from_r0171_rate": estimated_edges,
            "estimated_gpu_hours_from_r0171_wall_per_row": (
                graph_seconds_per_row * DIVERSE_ROWS / 3600.0
            ),
            "actual_graph_receipt_controls_train_horizon": True,
        },
        "train_plan": {
            "seed": 42,
            "hidden_dimension": selected_width,
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "estimated_successful_updates": estimated_updates,
            "estimated_updates_per_s_from_full_factorial_cell": rate,
            "estimated_gpu_hours": estimated_train_hours,
            "queue_stage_maximum_gpu_hours": max_stage_hours,
            "queue_split_required": stages > 1,
            "stages": stage_targets,
            "split_state": (
                "explicit model/optimizer/scaler/RNG/global-cosine-schedule state "
                "dict bound by exact hash; no dill/pickle object"
            ),
            "stage_2_can_issue_after_stage_1_result_without_science_review": (
                "only when stage 1 is a mechanically exact checkpoint boundary "
                "and the full multi-stage schedule was preregistered"
            ),
        },
        "evaluation_and_gate_plan": {
            "native": [
                "density-v2 diagnostic",
                "FFR",
                "purity fidelity k256/k1024",
                "per-language FFR",
            ],
            "ood": [
                "held-out pol_Latn headline panel",
                "repaired multilingual prompted probe panel",
            ],
            "ood_repair": (
                "remove the five exact R0173 training-family overlaps from the "
                "probe candidates, refill deterministically, and seal/audit the "
                "final query IDs before graph or model training"
            ),
            "scale_relative_retention": (
                "pre-register against the nearest composition/prompt/width-matched "
                "rung; do not reuse FineWeb-only absolute floors"
            ),
            "commensurate_diverse_gate_family_required": True,
            "gate_calibration": (
                "calibrate seed variation on this prompted diverse population "
                "before any atlas-quality or production claim"
            ),
            "projection_ffr_role": "diagnostic",
        },
        "estimated_gpu_hours": {
            "graph": graph_seconds_per_row * DIVERSE_ROWS / 3600.0,
            "train": estimated_train_hours,
            "evaluation_placeholder": 0.75,
            "total_before_seed_calibration": (
                graph_seconds_per_row * DIVERSE_ROWS / 3600.0
                + (estimated_train_hours or 0.0)
                + 0.75
            ),
        },
        "blocking_reason": (
            None
            if selected_width is not None
            else "factorial found no consistent capacity compensation; replan width"
        ),
        "next_action": (
            "draft separate graph, staged-train, and evaluation rounds"
            if selected_width is not None
            else "do not issue U12 GPU work until width is replanned"
        ),
        "training_performed": False,
        "production_or_publishing": False,
    }
    return seal(output)


def render_factorial_markdown(value: Mapping[str, Any]) -> str:
    lines = [
        "# Width × N factorial and capacity economics",
        "",
        f"Outcome: **{value['outcome']}**.",
        "",
        "Pile FFR is the registered selector; density-v2 and projection FFR are diagnostic.",
        "",
        "| width | rung | rows | Pile FFR | q→h retention | h→f retention | train GPU-h | updates/s | density-v2 | projection FFR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for width in WIDTHS:
        retention = value["retentions"][width]["pile_ffr"]
        for rung in RUNGS:
            cell = value["cells"][width][rung]
            lines.append(
                f"| {width} | {rung} | {cell['rows']:,} | "
                f"{cell['primary_metrics']['pile_ffr']:.6f} | "
                f"{retention['half_over_quarter']:.6f} | "
                f"{retention['full_over_half']:.6f} | "
                f"{cell['train_gpu_hours']:.4f} | {cell['updates_per_s']:.2f} | "
                f"{cell['diagnostic_metrics']['mixed_density_v2']:.6f} | "
                f"{cell['diagnostic_metrics']['mixed_projection_ffr']:.6f} |"
            )
    lines.extend([
        "",
        "The table is a paired seed-42, composition-controlled, fixed-dose factorial. "
        "The per-GPU-hour quantities in the JSON are descriptive economics, not gates.",
        "",
    ])
    return "\n".join(lines)


def render_u12_markdown(value: Mapping[str, Any]) -> str:
    width = value.get("selected_hidden_dimension")
    lines = [
        "# Prompted diverse U12 next-rung design memo",
        "",
        f"Factorial outcome: **{value['factorial_outcome']}**.",
        f"Selected hidden dimension: **{width if width is not None else 'none'}**.",
        "This memo is not GPU launch authority.",
        "",
        "## Planned execution",
        "",
        f"- Population: {value['population']['rows']:,} exact R0168 U12 rows.",
        f"- Graph: {value['graph_plan']['expected_shards']} sequential fp32-IVF shards; estimated {value['graph_plan']['estimated_gpu_hours_from_r0171_wall_per_row']:.2f} GPU-h.",
        f"- Training: {value['train_plan']['estimated_successful_updates']:,} provisional updates; estimated {value['train_plan']['estimated_gpu_hours'] if value['train_plan']['estimated_gpu_hours'] is not None else 'n/a'} GPU-h.",
        f"- Queue split required: {value['train_plan']['queue_split_required']}.",
        "- Final update horizon is recomputed from the sealed prompted graph edge count.",
        "- Repair and seal the multilingual OOD reserve before training; retain Polish as the headline held-out language.",
        "- Calibrate a commensurate prompted-diverse seed family before any atlas-quality claim.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "DIVERSE_ROWS",
    "FACTORIAL_CAPABILITY",
    "FACTORIAL_SCHEMA",
    "MEMO_CAPABILITY",
    "MEMO_SCHEMA",
    "RETENTION_FLOOR",
    "ROUND_ID",
    "RUNGS",
    "Round0207Error",
    "STEPS",
    "WIDTHS",
    "build_factorial",
    "build_u12_design",
    "render_factorial_markdown",
    "render_u12_markdown",
]
