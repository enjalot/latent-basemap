"""Frozen synthesis contract for the R0183 held-out projection table."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .round0108_evaluation import seal
from .round0175_aumap_baseline import SCALES


ROUND_ID = "0183"
CAPABILITY = "jina-heldout-projection-method-table-v1"
NUMAP_CAPABILITY = "jina-numap-grease-fixed-normalization-oos-baseline-v1"


class Round0183Error(RuntimeError):
    """The registered R0183 synthesis contract was violated."""


def _metric(value: Mapping[str, Any], key: str, *, label: str) -> float:
    try:
        observed = float(value[key])
    except (KeyError, TypeError, ValueError) as error:
        raise Round0183Error(f"{label} lacks {key}") from error
    if not math.isfinite(observed) or observed < 0.0 or observed > 1.0:
        raise Round0183Error(f"{label} {key} is invalid")
    return observed


def _pair(value: Mapping[str, Any], *, label: str) -> dict[str, float]:
    return {
        "ffr": _metric(value, "ffr", label=label),
        "recall_at_10": _metric(value, "recall_at_10", label=label),
    }


def _parametric_pair(value: Mapping[str, Any], *, label: str) -> dict[str, float]:
    return {
        "ffr": _metric(value, "projection_ffr", label=label),
        "recall_at_10": _metric(value, "projection_recall_at_10", label=label),
    }


def _delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {key: float(left[key]) - float(right[key]) for key in ("ffr", "recall_at_10")}


def build_table(
    *,
    aumap: Mapping[str, Any],
    numap: Mapping[str, Any] | None,
    numap_terminal_status: str,
) -> dict[str, Any]:
    scales = aumap.get("scales")
    if (
        aumap.get("schema") != "round0175-aumap-oos-synthesis-v1"
        or aumap.get("round_id") != "0175"
        or aumap.get("outcome") != "aumap-oos-baseline-measured"
        or not isinstance(scales, Mapping)
        or set(scales) != set(SCALES)
    ):
        raise Round0183Error("accepted R0175 synthesis identity changed")
    if numap_terminal_status not in {"measured", "terminal-retry-failed"}:
        raise Round0183Error("R0181 terminal branch is invalid")
    if (numap_terminal_status == "measured") != (numap is not None):
        raise Round0183Error("R0181 capability presence disagrees with its branch")

    rows: dict[str, Any] = {}
    for scale in SCALES:
        source = scales.get(scale)
        if not isinstance(source, Mapping):
            raise Round0183Error(f"R0175 {scale} cell is absent")
        aumap_metrics = _pair(
            source.get("aumap_inverse_distance") or {}, label=f"aUMAP {scale}"
        )
        historical = source.get("historical_parametric_context")
        corrected: dict[str, float] | None = None
        legacy: dict[str, float] | None = None
        evidence: Mapping[str, Any] | None = None
        if scale in {"200k", "2m"}:
            if not isinstance(historical, Mapping):
                raise Round0183Error(f"R0175 {scale} parametric context is absent")
            corrected_source = historical.get("standard_curve_seed42")
            legacy_source = historical.get("legacy_a1b1_seed42")
            evidence = historical.get("evidence")
            if not all(
                isinstance(value, Mapping)
                for value in (corrected_source, legacy_source, evidence)
            ):
                raise Round0183Error(f"R0175 {scale} parametric context is incomplete")
            corrected = _parametric_pair(
                corrected_source, label=f"corrected parametric {scale}"
            )
            legacy = _parametric_pair(legacy_source, label=f"legacy parametric {scale}")
        elif historical is not None:
            raise Round0183Error("R0175 unexpectedly acquired a 500k parametric cell")

        rows[scale] = {
            "rows": {"200k": 200_000, "500k": 500_000, "2m": 2_000_000}[scale],
            "held_hash": {"200k": "0e81ac067567", "500k": "7d94f88eb0bc", "2m": "cd1208a56d17"}[scale],
            "aumap_inverse_distance_k15": aumap_metrics,
            "corrected_parametric_standard_curve_seed42": corrected,
            "corrected_parametric_minus_aumap": (
                _delta(corrected, aumap_metrics) if corrected is not None else None
            ),
            "legacy_parametric_a1b1_seed42_context": legacy,
            "parametric_evidence": dict(evidence) if evidence is not None else None,
            "corrected_parametric_status": (
                "measured-historical-context"
                if corrected is not None
                else "not-measured-no-corrected-500k-model"
            ),
            "comparability": (
                "same deterministic held-out source IDs and canonical formulas; "
                "different fitted/transductive maps, so descriptive not paired"
            ),
        }

    numap_cell: dict[str, Any] | None = None
    if numap is not None:
        comparison = numap.get("comparison_to_reviewed_r0175")
        if (
            numap.get("schema")
            != "round0181-numap-fixed-normalization-synthesis-v1"
            or numap.get("round_id") != "0181"
            or numap.get("outcome")
            != "numap-grease-fixed-normalization-baseline-measured"
            or not isinstance(comparison, Mapping)
        ):
            raise Round0183Error("R0181 NUMAP synthesis identity changed")
        numap_metrics = _pair(
            comparison.get("numap_fixed_normalization") or {}, label="NUMAP 200k"
        )
        bound_aumap = _pair(
            comparison.get("aumap_inverse_distance") or {}, label="NUMAP-bound aUMAP"
        )
        if any(
            not math.isclose(bound_aumap[key], rows["200k"]["aumap_inverse_distance_k15"][key], abs_tol=1.0e-15)
            for key in ("ffr", "recall_at_10")
        ):
            raise Round0183Error("R0181 and R0175 aUMAP cells disagree")
        numap_cell = {
            "scale": "200k",
            "metrics": numap_metrics,
            "minus_aumap": _delta(numap_metrics, bound_aumap),
            "comparability": comparison.get("comparability"),
        }

    return seal({
        "schema": "round0183-heldout-projection-method-table-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "rows": rows,
        "numap_grease_fixed_normalization": numap_cell,
        "numap_terminal_status": numap_terminal_status,
        "primary_table_scope": (
            "corrected parametric standard-curve context versus accepted aUMAP "
            "at 200k/500k/2m; 500k parametric absence is explicit"
        ),
        "missing_cell": {
            "method": "corrected parametric standard curve",
            "scale": "500k",
            "reason": (
                "no corrected-kernel 500k model/evaluation exists; legacy or "
                "different-recipe checkpoints are not silently substituted"
            ),
        },
        "selector": "authenticated transcription only; no method-winner selector",
        "training_performed": False,
        "new_metric_computation": False,
        "diagnostic_only": True,
        "production_or_publishing": False,
    })


def render_markdown(table: Mapping[str, Any]) -> str:
    if table.get("schema") != "round0183-heldout-projection-method-table-v1":
        raise Round0183Error("cannot render unknown method table")
    lines = [
        "# Held-out projection method table",
        "",
        "| scale | corrected parametric FFR / R@10 | aUMAP FFR / R@10 | parametric − aUMAP |",
        "| --- | ---: | ---: | ---: |",
    ]
    for scale in SCALES:
        row = table["rows"][scale]
        parametric = row["corrected_parametric_standard_curve_seed42"]
        aumap = row["aumap_inverse_distance_k15"]
        delta = row["corrected_parametric_minus_aumap"]
        if parametric is None:
            ptext, dtext = "not measured", "not measured"
        else:
            ptext = f"{parametric['ffr']:.5f} / {parametric['recall_at_10']:.5f}"
            dtext = f"{delta['ffr']:+.5f} / {delta['recall_at_10']:+.5f}"
        lines.append(
            f"| {scale} | {ptext} | {aumap['ffr']:.5f} / {aumap['recall_at_10']:.5f} | {dtext} |"
        )
    lines.extend(["", "## NUMAP/GrEASE 200k", ""])
    numap = table.get("numap_grease_fixed_normalization")
    if isinstance(numap, Mapping):
        metrics = numap["metrics"]
        lines.append(
            f"Measured after the bounded normalization retry: FFR {metrics['ffr']:.5f}, recall@10 {metrics['recall_at_10']:.5f}."
        )
    else:
        lines.append("Unavailable: the single bounded normalization retry ended without a released capability.")
    lines.extend([
        "",
        "The corrected 500k parametric cell is intentionally shown as unmeasured; no legacy checkpoint is substituted.",
        "Comparisons share held-out IDs and formulas but use different fitted/transductive maps, so they are descriptive rather than paired method-winner tests.",
        "",
    ])
    return "\n".join(lines)


__all__ = [
    "CAPABILITY",
    "NUMAP_CAPABILITY",
    "ROUND_ID",
    "Round0183Error",
    "build_table",
    "render_markdown",
]
