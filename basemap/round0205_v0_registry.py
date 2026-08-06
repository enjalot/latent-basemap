"""Frozen contract for the FineWeb-2M v0 local registry promotion."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ROUND_ID = "0205"
CANDIDATE_ID = "basemap-jina-v5-nano-en-2m-v0"
CAPABILITY = "basemap-jina-v5-nano-en-2m-v0-local-registry-v1"
MAP_DEFINITION_SCHEMA = (
    "round0205-basemap-jina-v5-nano-en-2m-v0-map-definition-v1"
)
PUBLICATION_SCHEMA = (
    "round0205-basemap-jina-v5-nano-en-2m-v0-registry-publication-v1"
)
BUNDLE_SCHEMA = (
    "round0204-basemap-jina-v5-nano-en-2m-v0-release-bundle-v1"
)
EXPECTED_COORDINATES_SHA256 = (
    "ab9766d9d147d51e9e20ff76170a6f1c815ca99642191e9708098f0370fe0f8a"
)
EXPECTED_TRAIN_RECEIPT_SHA256 = (
    "45965c184f5610c7a009169a7f9eb5fe202b6e9aa842924b2e5d98fd633f5d51"
)
ROWS = 1_993_761
INPUT_DIMENSION = 768
OUTPUT_DIMENSION = 2


class Round0205Error(RuntimeError):
    """The reviewed release bundle or local promotion contract changed."""


def canonical_metrics(bundle: Mapping[str, Any]) -> dict[str, float]:
    """Extract the exact seed-42 gate values used by the registry card."""
    try:
        qualification = bundle["qualification"]
        cell = qualification["per_seed_gate_table"]["seed42"]
        metrics = cell["metrics"]
    except (KeyError, TypeError) as error:
        raise Round0205Error("R0204 seed-42 gate table is missing") from error
    if (
        qualification.get("all_four_seeds_pass_all_six_commensurate_gates")
        is not True
        or cell.get("seed") != 42
        or cell.get("all_six_pass") is not True
    ):
        raise Round0205Error("R0204 seed-42 qualification changed")
    keys = {
        "density": "density_v2",
        "ffr": "ffr",
        "purity_k256": "purity_fidelity_k256",
        "purity_k1024": "purity_fidelity_k1024",
        "projection_ffr": "projection_ffr",
        "heldout_recall_at_10": "heldout_recall_at_10",
    }
    output: dict[str, float] = {}
    for output_name, source_name in keys.items():
        source = metrics.get(source_name) or {}
        if source.get("pass") is not True:
            raise Round0205Error(f"R0204 seed-42 {source_name} no longer passes")
        try:
            output[output_name] = float(source["observed"])
        except (KeyError, TypeError, ValueError) as error:
            raise Round0205Error(
                f"R0204 seed-42 {source_name} value is missing"
            ) from error
    return output


def named_ood_failures(bundle: Mapping[str, Any]) -> list[str]:
    """Return and authenticate the canonical seed's plainly named failures."""
    limitations = bundle.get("ood_limitations") or {}
    rows = (limitations.get("maps") or {}).get(
        "r0115-prompted-2m-seed42"
    ) or []
    failures = [
        str(row.get("probe"))
        for row in rows
        if isinstance(row, Mapping) and row.get("verdict") == "named-failure"
    ]
    expected = [
        "code",
        "culture",
        "danish",
        "government",
        "latin",
        "science",
        "trec-covid",
    ]
    if (
        failures != expected
        or limitations.get("seed42_named_failure_count") != len(expected)
        or limitations.get("universal_quality_claim") is not False
    ):
        raise Round0205Error("R0204 canonical OOD limitations changed")
    return failures


__all__ = [
    "BUNDLE_SCHEMA",
    "CANDIDATE_ID",
    "CAPABILITY",
    "EXPECTED_COORDINATES_SHA256",
    "EXPECTED_TRAIN_RECEIPT_SHA256",
    "INPUT_DIMENSION",
    "MAP_DEFINITION_SCHEMA",
    "OUTPUT_DIMENSION",
    "PUBLICATION_SCHEMA",
    "ROUND_ID",
    "ROWS",
    "Round0205Error",
    "canonical_metrics",
    "named_ood_failures",
]
