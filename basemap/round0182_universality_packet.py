"""Frozen synthesis contract for the R0182 universality readout packet."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .round0142_jina_universality import PROBE_ORDER
from .round0167_prompted_universality import seal


ROUND_ID = "0182"
CAPABILITY = "jina-prompted-raw-universality-readout-v1"
PROMPTED_MAP_ORDER = (
    "r0115-prompted-2m-seed42",
    "r0117-prompted-2m-seed43",
    "r0171-prompted-8m-seed42",
)
RAW_MAP_ORDER = (
    "r0132-12p5m-seed42",
    "r0107-25m-seed42",
)


class Round0182Error(RuntimeError):
    """The registered R0182 synthesis contract was violated."""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _indexed_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    maps: Sequence[str],
    label: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    expected = {(map_key, probe) for map_key in maps for probe in PROBE_ORDER}
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("map") or ""), str(row.get("probe") or ""))
        if key in indexed or key not in expected:
            raise Round0182Error(f"{label} contains duplicate or unknown cell {key}")
        if not _finite(row.get("ffr_retention")) or not _finite(
            row.get("recall10_retention")
        ):
            raise Round0182Error(f"{label} cell {key} lacks retention metrics")
        indexed[key] = dict(row)
    if set(indexed) != expected:
        raise Round0182Error(f"{label} does not cover its full map/probe product")
    return indexed


def _rho(
    correlations: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    outcome: str = "ffr_retention",
) -> float:
    matches = [
        row
        for row in correlations
        if row.get("scope") == scope
        and row.get("outcome") == outcome
        and row.get("predictor") == "twonn_intrinsic_dimension"
    ]
    if len(matches) != 1 or not _finite(matches[0].get("spearman_rho")):
        raise Round0182Error(f"missing unique TwoNN correlation for {scope}/{outcome}")
    return float(matches[0]["spearman_rho"])


def _verdict(ffr_retention: float) -> str:
    if ffr_retention >= 0.70:
        return "pass"
    if ffr_retention >= 0.50:
        return "amber"
    return "named-failure"


def build_packet(
    *,
    prompted: Mapping[str, Any],
    raw: Mapping[str, Any],
    predictors: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        prompted.get("schema") != "jina-prompted-universality-panel-v1"
        or prompted.get("round_id") != "0178"
        or prompted.get("map_order") != list(PROMPTED_MAP_ORDER)
        or prompted.get("probe_order") != list(PROBE_ORDER)
        or prompted.get("diagnostic_only") is not True
        or prompted.get("no_causal_prompt_claim") is not True
        or raw.get("schema") != "jina-diverse-universality-panel-v1"
        or raw.get("round_id") != "0142"
        or raw.get("probe_order") != list(PROBE_ORDER)
        or predictors.get("schema")
        != "jina-diverse-projection-loss-predictors-v1"
        or predictors.get("round_id") != "0146"
    ):
        raise Round0182Error("universality source identity changed")
    prompted_rows = prompted.get("cells")
    raw_rows = raw.get("rows")
    prompted_correlations = prompted.get("twonn_correlations")
    raw_correlations = predictors.get("correlations")
    prompted_geometry = prompted.get("prompted_geometry")
    if not all(
        isinstance(value, (list, tuple))
        for value in (prompted_rows, raw_rows, prompted_correlations, raw_correlations)
    ) or not isinstance(prompted_geometry, Mapping):
        raise Round0182Error("universality source tables are absent")

    prompted_index = _indexed_cells(
        prompted_rows, maps=PROMPTED_MAP_ORDER, label="prompted panel"
    )
    raw_index = _indexed_cells(raw_rows, maps=RAW_MAP_ORDER, label="raw panel")
    rows: list[dict[str, Any]] = []
    for probe in PROBE_ORDER:
        geometry = prompted_geometry.get(probe)
        twonn = (
            ((geometry or {}).get("geometry") or {}).get("twonn")
            if isinstance(geometry, Mapping)
            else None
        )
        dimension = twonn.get("intrinsic_dimension") if isinstance(twonn, Mapping) else None
        if not _finite(dimension):
            raise Round0182Error(f"prompted TwoNN geometry is absent for {probe}")
        cells: dict[str, Any] = {}
        for substrate, maps, index in (
            ("prompted", PROMPTED_MAP_ORDER, prompted_index),
            ("raw", RAW_MAP_ORDER, raw_index),
        ):
            for map_key in maps:
                cell = index[(map_key, probe)]
                ffr = float(cell["ffr_retention"])
                cells[map_key] = {
                    "substrate": substrate,
                    "ffr_retention": ffr,
                    "recall10_retention": float(cell["recall10_retention"]),
                    "verdict": _verdict(ffr),
                }
        rows.append({
            "probe": probe,
            "prompted_twonn_intrinsic_dimension": float(dimension),
            "maps": cells,
        })

    prompted_rhos = {
        map_key: _rho(prompted_correlations, scope=map_key)
        for map_key in PROMPTED_MAP_ORDER
    }
    prompted_rhos["pooled-descriptive"] = _rho(
        prompted_correlations, scope="pooled-descriptive"
    )
    raw_rhos = {
        map_key: _rho(raw_correlations, scope=map_key) for map_key in RAW_MAP_ORDER
    }
    raw_rhos["pooled-descriptive"] = _rho(
        raw_correlations, scope="pooled-descriptive"
    )
    prompted_pooled = prompted_rhos["pooled-descriptive"]
    raw_pooled = raw_rhos["pooled-descriptive"]
    if (
        not math.isclose(
            float((prompted.get("raw_comparison") or {}).get(
                "prompted_pooled_twonn_ffr_rho"
            )),
            prompted_pooled,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            float((prompted.get("raw_comparison") or {}).get(
                "raw_pooled_twonn_ffr_rho"
            )),
            raw_pooled,
            abs_tol=1.0e-15,
        )
    ):
        raise Round0182Error("R0178 cross-substrate correlation binding changed")

    return seal({
        "schema": "round0182-universality-readout-packet-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "probe_order": list(PROBE_ORDER),
        "map_order": {
            "prompted": list(PROMPTED_MAP_ORDER),
            "raw": list(RAW_MAP_ORDER),
        },
        "rows": rows,
        "twonn_ffr_spearman": {
            "prompted": prompted_rhos,
            "raw": raw_rhos,
            "pooled_prompted_minus_raw": prompted_pooled - raw_pooled,
        },
        "substrate_invariance_observation": (
            "TwoNN intrinsic dimension is negatively associated with FFR "
            "retention in every prompted map, both raw maps, and both pooled "
            "descriptive panels. The direction replicates across embedding "
            "conventions; magnitudes are not a causal prompt contrast because "
            "the raw and prompted panels use different map scales."
        ),
        "interpretation": (
            "owner-facing transcription of accepted R0142/R0146/R0178 evidence; "
            "no metric is recomputed and no new science selector is applied"
        ),
        "training_performed": False,
        "diagnostic_only": True,
        "no_causal_prompt_claim": True,
        "no_universal_map_claim": True,
        "production_or_publishing": False,
    })


def render_markdown(packet: Mapping[str, Any]) -> str:
    if packet.get("schema") != "round0182-universality-readout-packet-v1":
        raise Round0182Error("cannot render an unknown packet")
    headers = ["probe"] + [
        f"{map_key} FFR / R@10" for map_key in (*PROMPTED_MAP_ORDER, *RAW_MAP_ORDER)
    ]
    lines = [
        "# Prompted/raw Jina universality readout",
        "",
        "This is a diagnostic transcription of accepted R0142, R0146, and R0178 evidence.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in packet["rows"]:
        values = [str(row["probe"])]
        for map_key in (*PROMPTED_MAP_ORDER, *RAW_MAP_ORDER):
            cell = row["maps"][map_key]
            values.append(
                f"{cell['ffr_retention']:.4f} / {cell['recall10_retention']:.4f}"
            )
        lines.append("| " + " | ".join(values) + " |")
    correlations = packet["twonn_ffr_spearman"]
    lines.extend([
        "",
        "## TwoNN versus FFR-retention Spearman rho",
        "",
        "| substrate | map/scope | rho |",
        "| --- | --- | ---: |",
    ])
    for substrate in ("prompted", "raw"):
        for scope, value in correlations[substrate].items():
            lines.append(f"| {substrate} | {scope} | {value:.4f} |")
    lines.extend([
        "",
        packet["substrate_invariance_observation"],
        "",
        "This packet does not establish a causal prompt effect, universal-map quality, or publication readiness.",
        "",
    ])
    return "\n".join(lines)


__all__ = [
    "CAPABILITY",
    "PROMPTED_MAP_ORDER",
    "RAW_MAP_ORDER",
    "ROUND_ID",
    "Round0182Error",
    "build_packet",
    "render_markdown",
]
