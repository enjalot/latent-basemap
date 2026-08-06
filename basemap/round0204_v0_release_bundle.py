"""Frozen contract and rendering helpers for the FineWeb-2M v0 bundle."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ROUND_ID = "0204"
CANDIDATE_ID = "basemap-jina-v5-nano-en-2m-v0"
CAPABILITY = "basemap-jina-v5-nano-en-2m-v0-release-bundle-v1"
BUNDLE_SCHEMA = "round0204-basemap-jina-v5-nano-en-2m-v0-release-bundle-v1"
PROPOSAL_SCHEMA = "round0195-fineweb-2m-v0-release-proposal-v1"
PROMPTED_MAPS = (
    "r0115-prompted-2m-seed42",
    "r0117-prompted-2m-seed43",
)
PROBE_ORDER = (
    "cebuano",
    "code",
    "culture",
    "danish",
    "government",
    "latin",
    "science",
    "web",
    "scifact",
    "trec-covid",
    "dadabase",
)


class Round0204Error(RuntimeError):
    """The accepted release evidence or required caveat changed."""


def detailed_ood_rows(packet: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if tuple(packet.get("probe_order") or ()) != PROBE_ORDER:
        raise Round0204Error("R0182 probe order changed")
    rows = packet.get("rows") or []
    if [row.get("probe") for row in rows] != list(PROBE_ORDER):
        raise Round0204Error("R0182 OOD rows changed")
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in PROMPTED_MAPS}
    for row in rows:
        for map_name in PROMPTED_MAPS:
            cell = (row.get("maps") or {}).get(map_name) or {}
            if cell.get("substrate") != "prompted" or cell.get("verdict") not in {
                "pass",
                "amber",
                "named-failure",
            }:
                raise Round0204Error(f"R0182 {map_name} OOD cell changed")
            output[map_name].append({
                "probe": row["probe"],
                "verdict": cell["verdict"],
                "ffr_retention": float(cell["ffr_retention"]),
                "recall10_retention": float(cell["recall10_retention"]),
            })
    expected_failures = {
        "r0115-prompted-2m-seed42": 7,
        "r0117-prompted-2m-seed43": 6,
    }
    for map_name, count in expected_failures.items():
        observed = sum(
            row["verdict"] == "named-failure" for row in output[map_name]
        )
        if observed != count:
            raise Round0204Error(f"R0182 {map_name} named-failure count changed")
    return output


def render_model_card(
    *, proposal: Mapping[str, Any], ood_rows: Mapping[str, list[Mapping[str, Any]]]
) -> str:
    scope = proposal["candidate_scope"]
    qualification = proposal["qualification"]
    method = proposal["method_context"]
    seed42_failures = [
        str(row["probe"])
        for row in ood_rows[PROMPTED_MAPS[0]]
        if row["verdict"] == "named-failure"
    ]
    seed43_failures = [
        str(row["probe"])
        for row in ood_rows[PROMPTED_MAPS[1]]
        if row["verdict"] == "named-failure"
    ]
    gate_lines = []
    for seed_name in ("seed42", "seed43", "seed44", "seed45"):
        cell = qualification["cells"][seed_name]
        metrics = cell["metrics"]
        gate_lines.append(
            f"| {cell['seed']} | "
            + " | ".join(
                f"{float(metrics[key]['observed']):.6g} / "
                f"{float(metrics[key]['floor']):.6g}"
                for key in (
                    "density_v2",
                    "ffr",
                    "purity_fidelity_k256",
                    "purity_fidelity_k1024",
                    "projection_ffr",
                    "heldout_recall_at_10",
                )
            )
            + " | pass |"
        )
    return "\n".join([
        "---",
        "license: apache-2.0",
        "library_name: latent-basemap",
        "tags:",
        "- dimensionality-reduction",
        "- jina-embeddings-v5",
        "- fineweb",
        "---",
        "",
        f"# {CANDIDATE_ID}",
        "",
        "Draft model card. This file is an authenticated local release artifact; ",
        "it does not indicate that a Hugging Face upload has occurred.",
        "",
        "## Model description",
        "",
        f"A seed-{scope['canonical_seed']} parametric 2D basemap trained on ",
        f"{scope['rows']:,} representative rows from {scope['corpus']} using ",
        f"{scope['embedding_convention']!r} Jina-v5 document embeddings ",
        f"({scope['dimension']} dimensions). The local registry artifact contains ",
        "the exact coordinates and authenticated training receipt named in the ",
        "release bundle.",
        "",
        "## Qualification",
        "",
        "All four registered seeds pass all six commensurate R0161 gates.",
        "Each table cell is `observed / floor`.",
        "",
        "| seed | density-v2 | FFR | purity k256 | purity k1024 | projection FFR | held-out R@10 | verdict |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
        *gate_lines,
        "",
        "## Out-of-distribution limitations",
        "",
        "This map is not universally reliable out of distribution. On the 11 named ",
        "R0182 probes, the canonical seed-42 map has **7 of 11 named failures**: ",
        f"{', '.join(seed42_failures)}. Seed 43 has **6 of 11 named failures**: ",
        f"{', '.join(seed43_failures)}. OOD coverage exists for seeds 42 and 43 ",
        "only. Median FFR retention is about 0.454 and 0.460 respectively. Treat ",
        "projections of new corpora as diagnostic and validate them against local ",
        "high-dimensional neighbors before relying on map geometry.",
        "",
        "## Method context",
        "",
        f"Historical same-scale evidence reports corrected-parametric FFR ",
        f"{float(method['corrected_parametric_2m_ffr']):.5f} and aUMAP FFR ",
        f"{float(method['aumap_2m_ffr']):.5f}. This is not a candidate-specific ",
        "paired comparison and does not establish a method winner.",
        "",
        "## Intended use",
        "",
        "Exploratory global layout and projection of semantically similar English ",
        "web text under the exact documented embedding convention. Not suitable for ",
        "distance-sensitive decisions, universal multilingual/code/scientific ",
        "retrieval claims, or SAE readiness claims without separate validation.",
        "",
        "## Release status",
        "",
        "Local registry promotion is separately authorized and review-gated. External ",
        "publication or Hugging Face upload is not authorized by this artifact.",
        "",
    ])


__all__ = [
    "BUNDLE_SCHEMA",
    "CANDIDATE_ID",
    "CAPABILITY",
    "PROBE_ORDER",
    "PROPOSAL_SCHEMA",
    "ROUND_ID",
    "Round0204Error",
    "detailed_ood_rows",
    "render_model_card",
]
