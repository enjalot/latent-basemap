"""Round 0041: canonical 30M graph and sampler-semantics evidence.

This round turns the reviewed R0020 exact-family census into the same
representative-destination graph policy used at 150M.  It does not train.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from .round0034_pipeline import build_canonical_graph


ROUND_ID = "0041"
ROW_COUNT = 30_000_000
K = 15
ELIGIBILITY_SCHEMA = "minilm-fp16-row-eligibility-v1"
GRAPH_PATH = "/data/checkpoints/pumap/edges_30m_k15.npz"
GRAPH_SHA256 = (
    "2fc30fc27ced442c5b69fde084ab41c054fcc1bf5e7913a5cee9d20f59baadca"
)
CENSUS_PATH = (
    "/data/latent-basemap/runs/round-0020/queue/artifacts/"
    "duplicate-census/global-duplicate-census-v1.npz"
)
CENSUS_SHA256 = (
    "834089fcbd9a722cec4f05be6382ed8430d27280e7e23ca0855785e3f48ea5e2"
)
SELECTOR_PATH = (
    "/data/latent-basemap/runs/round-0040/queue/artifacts/"
    "minilm-reference/representative-selector.npz"
)
SELECTOR_SHA256 = (
    "4f3b8a13649589d4b7ce6e4fb4828cefec606d84fc880d99ac7dd119ad787bde"
)
R0021_TRAIN_RECEIPT = (
    "/data/latent-basemap/runs/round-0021/queue/artifacts/train/"
    "train-receipt.json"
)
R0030_TRAIN_RECEIPT = (
    "/data/latent-basemap/runs/round-0030/queue/artifacts/uniform/train/"
    "train-receipt.json"
)


class Round0041Error(RuntimeError):
    """The registered 30M semantic-alignment contract was violated."""


def _validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0041Error(f"{label} identity seal is invalid")


def _read_metadata(raw: Any) -> dict[str, Any]:
    value = raw.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise Round0041Error("NPZ metadata is not an object")
    return parsed


def load_fp16_eligibility() -> dict[str, Any]:
    """Load R0020 families and produce an explicit copy->representative map."""
    census_signature = expected_input_signature(CENSUS_PATH)
    if census_signature["sha256"] != CENSUS_SHA256:
        raise Round0041Error("R0020 census bytes changed")
    with np.load(CENSUS_PATH, allow_pickle=False) as archive:
        expected_names = {
            "metadata",
            "representative_rows",
            "family_counts",
            "family_offsets",
            "member_rows",
            "family_hash_h0",
            "family_hash_h1",
        }
        if set(archive.files) != expected_names:
            raise Round0041Error("R0020 census members changed")
        metadata = _read_metadata(archive["metadata"])
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "metadata"
        }
    _validate_seal(metadata, label="R0020 duplicate census")
    observed_hashes = {
        name: ordered_array_sha256(value)
        for name, value in arrays.items()
    }
    representatives = np.asarray(arrays["representative_rows"], dtype=np.int64)
    counts = np.asarray(arrays["family_counts"], dtype=np.int64)
    offsets = np.asarray(arrays["family_offsets"], dtype=np.int64)
    members = np.asarray(arrays["member_rows"], dtype=np.int64)
    summary = metadata.get("summary") or {}
    if (
        metadata.get("schema") != "global-duplicate-census-v1"
        or metadata.get("row_count") != ROW_COUNT
        or metadata.get("array_sha256") != observed_hashes
        or representatives.ndim != 1
        or counts.shape != representatives.shape
        or offsets.shape != (len(representatives) + 1,)
        or members.ndim != 1
        or not np.array_equal(offsets[1:] - offsets[:-1], counts)
        or offsets[0] != 0
        or offsets[-1] != len(members)
        or np.any(counts < 2)
        or not np.array_equal(representatives, members[offsets[:-1]])
        or not np.array_equal(representatives, np.sort(representatives))
        or len(np.unique(members)) != len(members)
        or (len(members) and (members[0] < 0 or members.max() >= ROW_COUNT))
        or int(summary.get("exact_family_count", -1)) != len(representatives)
        or int(summary.get("duplicated_copy_rows", -1))
        != len(members) - len(representatives)
    ):
        raise Round0041Error("R0020 census content or geometry changed")

    repeated_representatives = np.repeat(representatives, counts)
    is_copy = np.ones(len(members), dtype=bool)
    is_copy[offsets[:-1]] = False
    duplicate_rows = members[is_copy]
    duplicate_representatives = repeated_representatives[is_copy]
    order = np.argsort(duplicate_rows, kind="stable")
    duplicate_rows = duplicate_rows[order]
    duplicate_representatives = duplicate_representatives[order]
    if (
        not np.array_equal(duplicate_rows, np.unique(duplicate_rows))
        or np.intersect1d(duplicate_rows, representatives).size
        or np.any(duplicate_representatives >= duplicate_rows)
    ):
        raise Round0041Error("copy-to-representative mapping is invalid")

    selector_signature = expected_input_signature(SELECTOR_PATH)
    if selector_signature["sha256"] != SELECTOR_SHA256:
        raise Round0041Error("R0040 representative selector bytes changed")
    with np.load(SELECTOR_PATH, allow_pickle=False) as archive:
        selector_metadata = _read_metadata(archive["metadata"])
        selector_arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "metadata"
        }
    _validate_seal(selector_metadata, label="R0040 representative selector")
    if (
        selector_metadata.get("schema")
        != "round0040-minilm-representative-selector-v1"
        or selector_metadata.get("row_count") != ROW_COUNT
        or selector_metadata.get("zero_rows") != 0
        or selector_metadata.get("nonfinite_rows") != 0
        or selector_metadata.get("array_sha256")
        != {
            name: ordered_array_sha256(value)
            for name, value in selector_arrays.items()
        }
        or not np.array_equal(
            selector_arrays.get("excluded_rows"), duplicate_rows
        )
    ):
        raise Round0041Error(
            "R0040 full invalid scan and R0020 copy exclusions disagree"
        )

    eligibility_metadata = {
        "schema": ELIGIBILITY_SCHEMA,
        "row_count": ROW_COUNT,
        "source_census_identity_sha256": metadata["identity_sha256"],
        "source_selector_identity_sha256": selector_metadata["identity_sha256"],
        "summary": {
            "row_count": ROW_COUNT,
            "zero_row_count": 0,
            "exact_nonzero_family_count": len(representatives),
            "rows_in_exact_nonzero_families": len(members),
            "duplicate_copy_rows_excluded": len(duplicate_rows),
            "excluded_row_count": len(duplicate_rows),
            "retained_row_count": ROW_COUNT - len(duplicate_rows),
        },
    }
    return {
        "metadata": eligibility_metadata,
        # The census is the artifact that contains the exact family membership
        # needed to derive this mapping; the selector independently proves the
        # full corpus has no additional zero/nonfinite exclusions.
        "signature": census_signature,
        "zero_rows": np.empty(0, dtype=np.int64),
        "excluded_rows": duplicate_rows,
        "duplicate_excluded_rows": duplicate_rows,
        "duplicate_representative_rows": duplicate_representatives,
        "retained_row_count": ROW_COUNT - len(duplicate_rows),
        "selector_signature": selector_signature,
    }


def build_graph(output_root: str) -> dict[str, Any]:
    eligibility = load_fp16_eligibility()
    return build_canonical_graph(
        graph_path=GRAPH_PATH,
        expected_graph_sha256=GRAPH_SHA256,
        eligibility=eligibility,
        output_root=output_root,
        row_count=ROW_COUNT,
        k=K,
        round_id=ROUND_ID,
        eligibility_schema=ELIGIBILITY_SCHEMA,
        output_label="Round 0041 canonical graph output",
    )


def read_training_semantics(path: str) -> dict[str, Any]:
    signature = expected_input_signature(path)
    with open(path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    stats = receipt.get("train_stats") or receipt.get("train_accounting")
    if not isinstance(stats, dict) or stats.get("budget_satisfied") is not True:
        raise Round0041Error(f"training receipt is incomplete: {path}")
    prefix = "pipeline_"
    return {
        "receipt": signature,
        "pipeline": stats.get(prefix + "pipeline"),
        "sampler_class": stats.get(prefix + "sampler_class"),
        "positive_sampling": stats.get(prefix + "positive_sampling"),
        "positive_source_sampling": stats.get(
            prefix + "multiplicity_positive_source_sampling"
        ),
        "positive_destinations": stats.get(
            prefix + "multiplicity_positive_destinations"
        ),
        "graph_degree": stats.get(prefix + "multiplicity_graph_degree"),
        "effective_positive_edges": int(stats["n_pos_edges"]),
        "successful_updates": int(stats["positive_lr_optimizer_steps"]),
    }
