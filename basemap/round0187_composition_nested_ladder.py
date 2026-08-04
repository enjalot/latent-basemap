"""Frozen scientific contract for the composition-controlled nested ladder."""
from __future__ import annotations

import copy
import hashlib
import math
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap import round0113_prompt_contrast as r0113


ROUND_ID = "0187"
SEED = 42
DIMENSION = 768
HASH_NAMESPACE = "latent-basemap/r0187/composition-nested-canonical-id-sha256-v1"
POPULATION_SCHEMA = "round0187-composition-nested-population-v1"
GRAPH_SCHEMA_PREFIX = "round0187-composition-nested-fuzzy-graph"
TRAIN_SCHEMA_PREFIX = "round0187-composition-nested-train-receipt"
EVALUATION_SCHEMA = "round0187-composition-nested-common-core-evaluation-v1"
SYNTHESIS_SCHEMA = "round0187-composition-nested-ladder-decision-v1"
CAPABILITY = "jina-document-english-composition-controlled-nested-ladder-v1"
POPULATION_CAPABILITY = "jina-document-english-composition-nested-populations-v1"

CORPORA = (
    ("fineweb", 0, 2_890_362),
    ("redpajama", 2_890_362, 5_727_340),
    ("pile", 5_727_340, 8_000_000),
)
FULL_COUNTS = {
    "fineweb": 2_876_432,
    "redpajama": 2_810_627,
    "pile": 2_265_360,
}
RUNG_COUNTS = {
    "quarter": {
        "fineweb": 719_108,
        "redpajama": 702_656,
        "pile": 566_340,
    },
    "half": {
        "fineweb": 1_438_216,
        "redpajama": 1_405_313,
        "pile": 1_132_680,
    },
    "full": FULL_COUNTS,
}
RUNG_ROWS = {
    rung: sum(counts.values()) for rung, counts in RUNG_COUNTS.items()
}

# R0180's accepted whole-update dose, represented as an exact rational so the
# next horizon cannot drift due to a rounded decimal in prose.
FULL_GRAPH_EDGES = 603_086_368
FULL_SUCCESSFUL_UPDATES = 2_026_478
POSITIVE_ROWS_PER_UPDATE = r0113.POSITIVE_ROWS_PER_UPDATE
TARGET_POSITIVE_DRAWS_PER_EDGE = (
    FULL_SUCCESSFUL_UPDATES * POSITIVE_ROWS_PER_UPDATE / FULL_GRAPH_EDGES
)
RETENTION_RATIO = 0.97
COMPOUND_RETENTION_RATIO = RETENTION_RATIO**2
PRIMARY_METRICS = (
    "mixed_ffr",
    "mixed_purity_fidelity_k256",
    "mixed_purity_fidelity_k1024",
    "pile_ood_recall_at_10",
    "fineweb_ffr",
    "redpajama_ffr",
    "pile_ffr",
)
REQUIRED_TRAIN_CHECKS = {
    "exact_update_closure",
    "zero_numerical_skips",
    "no_pipeline_stamp_drift",
    "endpoint_rows_match_updates",
    "weighted_rejection_accounting_closes",
}
MULTIPLICITY_POLICY = (
    "prompted-source-text-and-document-fp16-union-representative-only"
)


class Round0187Error(RuntimeError):
    """The preregistered R0187 contract changed or failed authentication."""


def canonical_id_digest(corpus: str, canonical_row: int) -> bytes:
    """Return the frozen per-corpus canonical-ID rank digest."""
    if corpus not in FULL_COUNTS or canonical_row < 0:
        raise Round0187Error("invalid canonical-ID hash input")
    prefix = HASH_NAMESPACE.encode("utf-8") + b"\0" + corpus.encode("ascii") + b"\0"
    return hashlib.sha256(prefix + struct.pack(">Q", canonical_row)).digest()


def _select_nested_positions_for_spec(
    mapping: np.ndarray,
    *,
    corpora: tuple[tuple[str, int, int], ...],
    full_counts: Mapping[str, int],
    rung_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, np.ndarray]:
    """Pure implementation shared by the frozen population and tiny smoke."""
    values = np.asarray(mapping)
    if values.ndim != 1 or values.dtype != np.int64:
        raise Round0187Error("R0165 mapping geometry changed")
    expected_full = sum(int(value) for value in full_counts.values())
    if len(values) != expected_full or np.any(values[1:] <= values[:-1]):
        raise Round0187Error("R0165 mapping cardinality/order changed")

    selected: dict[str, list[np.ndarray]] = {"quarter": [], "half": []}
    for corpus, start, stop in corpora:
        positions = np.flatnonzero((values >= start) & (values < stop)).astype(np.int64)
        if len(positions) != int(full_counts[corpus]):
            raise Round0187Error(f"{corpus} full count changed")
        ids = values[positions]
        hashes = np.empty(len(ids), dtype="V32")
        prefix = (
            HASH_NAMESPACE.encode("utf-8")
            + b"\0"
            + corpus.encode("ascii")
            + b"\0"
        )
        for offset, canonical_row in enumerate(ids):
            hashes[offset] = hashlib.sha256(
                prefix + struct.pack(">Q", int(canonical_row))
            ).digest()
        rank = np.argsort(hashes, kind="stable")
        for rung in ("quarter", "half"):
            count = int(rung_counts[rung][corpus])
            chosen = np.sort(positions[rank[:count]])
            selected[rung].append(chosen)

    output = {
        rung: np.concatenate(parts).astype(np.int64, copy=False)
        for rung, parts in selected.items()
    }
    if not np.array_equal(
        output["quarter"],
        np.intersect1d(output["quarter"], output["half"], assume_unique=True),
    ):
        raise Round0187Error("quarter is not an exact subset of half")
    for rung in ("quarter", "half"):
        expected = sum(int(value) for value in rung_counts[rung].values())
        if (
            len(output[rung]) != expected
            or np.any(output[rung][1:] <= output[rung][:-1])
        ):
            raise Round0187Error(f"{rung} selection closure failed")
    return output


def select_nested_positions(mapping: np.ndarray) -> dict[str, np.ndarray]:
    """Select exact quarter/half strata with one hash rank, then restore order."""
    return _select_nested_positions_for_spec(
        mapping,
        corpora=CORPORA,
        full_counts=FULL_COUNTS,
        rung_counts=RUNG_COUNTS,
    )


def successful_updates_for_edges(edge_count: int) -> int:
    """Ceil the exact R0180 consumed-positive-draws/edge rational."""
    if edge_count <= 0:
        raise Round0187Error("edge count must be positive")
    numerator = FULL_SUCCESSFUL_UPDATES * int(edge_count)
    return (numerator + FULL_GRAPH_EDGES - 1) // FULL_GRAPH_EDGES


class NestedScalePromptTrainingInput(r0113.PromptTrainingInput):
    """R0113's exact sampler with an authenticated dynamic nested population."""

    def __init__(
        self,
        dataset: r0113.HostFp16EndpointArray,
        graph: Mapping[str, Any],
        *,
        arm: str = "document",
    ) -> None:
        self.dataset = dataset
        self.graph = dict(graph)
        self.arm = arm
        self.shape = dataset.shape
        self._last_sampler = None
        if (
            arm != "document"
            or self.shape[1:] != (DIMENSION,)
            or self.shape[0] <= 0
            or int(graph.get("n_nodes", -1)) != len(dataset)
        ):
            raise Round0187Error("nested training input geometry changed")

    @staticmethod
    def _patch_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(runtime)
        value["multiplicity_policy"] = MULTIPLICITY_POLICY
        return value

    def prepare_round0034_training(self, **kwargs: Any):
        dataset, sampler, edges, runtime, provenance = super().prepare_round0034_training(
            **kwargs
        )
        return dataset, sampler, edges, self._patch_runtime(runtime), provenance

    def runtime_stamp(self) -> dict[str, Any]:
        return self._patch_runtime(super().runtime_stamp())


def train_config(
    *,
    rung: str,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Clone the accepted recipe and change only N-bound identities/horizon."""
    if rung not in {"quarter", "half"} or retained_rows != RUNG_ROWS[rung]:
        raise Round0187Error("nested train rung/cardinality changed")
    updates = successful_updates_for_edges(graph_edges)
    config, _ = r0113.train_config(
        "document",
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=r0113.RETAINED_ROWS,
        seed=SEED,
    )
    config = copy.deepcopy(config)
    config["schema"] = f"round0187-{rung}-composition-nested-train-config-v1"
    config["paired_invariant"] = {
        "rung": rung,
        "rows": retained_rows,
        "dimension": DIMENSION,
        "seed": SEED,
        "successful_positive_lr_updates": updates,
        "dose_rule": (
            "ceil(R0180_successful_updates * active_edges / "
            "R0180_directed_edges)"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "graph_policy": "fuzzy k50, IVF8192, nprobe64, seeds 113/114",
        "sampler": r0113.SAMPLER_CLASS,
        "hidden_dimension": 2048,
    }
    config["input"].update({
        "rows": retained_rows,
        "representation": "prompted-document-host-fp16",
        "multiplicity_policy": MULTIPLICITY_POLICY,
        "composition": dict(RUNG_COUNTS[rung]),
        "nested_population_schema": POPULATION_SCHEMA,
    })
    expected = config["execution"]["expected_pipeline_stamp"]
    expected["negative_sampling"] = (
        f"uniform-{retained_rows}-compact-representatives-nonself"
    )
    expected["compact_retained_rows"] = retained_rows
    expected["multiplicity_policy"] = MULTIPLICITY_POLICY
    config["optimizer"]["successful_positive_lr_updates"] = updates
    config["execution"].update({
        "scale_change": (
            "composition-preserving nested N and dose-matched horizon only; "
            "h2048, seed, prompt, precision, graph and sampler semantics frozen"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": (
            updates * POSITIVE_ROWS_PER_UPDATE / graph_edges
        ),
        "graph_vector_storage": "gpu-ivfflat-fp32-complete-shard-search",
        "graph_execution": "all-row-shards-shared-quantizer-global-topk",
    })
    config["dose_registration"] = {
        "source_round": "0180",
        "source_graph_edges": FULL_GRAPH_EDGES,
        "source_successful_updates": FULL_SUCCESSFUL_UPDATES,
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "active_graph_edges": graph_edges,
        "successful_updates": updates,
        "rounding": "ceiling to first whole successful update at/above target",
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": (
            updates * POSITIVE_ROWS_PER_UPDATE / graph_edges
        ),
    }
    return config, sha256_bytes(canonical_json(config))


def purity_fidelity(value: Any) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise Round0187Error("purity ratio must be finite and positive")
    return math.exp(-abs(math.log(number)))


def primary_metric_view(
    *,
    mixed_panel: Mapping[str, Any],
    corpus_panels: Mapping[str, Mapping[str, Any]],
    pile_ood: Mapping[str, Any],
) -> dict[str, float]:
    if set(corpus_panels) != set(FULL_COUNTS):
        raise Round0187Error("per-corpus panel set changed")
    purity = mixed_panel.get("purity") or {}
    values = {
        "mixed_ffr": float(mixed_panel["ffr"]),
        "mixed_purity_fidelity_k256": purity_fidelity(purity["k256"]),
        "mixed_purity_fidelity_k1024": purity_fidelity(purity["k1024"]),
        "pile_ood_recall_at_10": float(pile_ood["recall_at_10"]),
        **{
            f"{corpus}_ffr": float(corpus_panels[corpus]["ffr"])
            for corpus in FULL_COUNTS
        },
    }
    if set(values) != set(PRIMARY_METRICS) or not np.isfinite(
        tuple(values.values())
    ).all():
        raise Round0187Error("primary metric vector is incomplete")
    return values


def train_checks_close(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == REQUIRED_TRAIN_CHECKS
        and all(value[key] is True for key in REQUIRED_TRAIN_CHECKS)
    )


def ladder_decision(cells: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Apply the preregistered 3% per-step and compounded trend branches."""
    if set(cells) != {"quarter", "half", "full"}:
        raise Round0187Error("ladder cell set changed")
    normalized = {
        rung: {key: float(value) for key, value in metrics.items()}
        for rung, metrics in cells.items()
    }
    if any(set(metrics) != set(PRIMARY_METRICS) for metrics in normalized.values()):
        raise Round0187Error("ladder metric set changed")
    if not np.isfinite(
        [value for metrics in normalized.values() for value in metrics.values()]
    ).all() or any(
        value <= 0 for metrics in normalized.values() for value in metrics.values()
    ):
        raise Round0187Error("ladder metrics must be finite and positive")

    steps: dict[str, Any] = {}
    for label, left, right in (
        ("quarter_to_half", "quarter", "half"),
        ("half_to_full", "half", "full"),
    ):
        steps[label] = {
            metric: {
                "left": normalized[left][metric],
                "right": normalized[right][metric],
                "retention_ratio": normalized[right][metric] / normalized[left][metric],
                "passed": (
                    normalized[right][metric] / normalized[left][metric]
                    >= RETENTION_RATIO
                ),
            }
            for metric in PRIMARY_METRICS
        }
    compound = {
        metric: {
            "quarter": normalized["quarter"][metric],
            "full": normalized["full"][metric],
            "retention_ratio": (
                normalized["full"][metric] / normalized["quarter"][metric]
            ),
            "material_regression": (
                normalized["full"][metric] / normalized["quarter"][metric]
                < COMPOUND_RETENTION_RATIO
            ),
        }
        for metric in PRIMARY_METRICS
    }
    all_steps_pass = all(
        cell["passed"] for step in steps.values() for cell in step.values()
    )
    concordant = {
        metric: (
            steps["quarter_to_half"][metric]["retention_ratio"] < 1.0
            and steps["half_to_full"][metric]["retention_ratio"] < 1.0
            and compound[metric]["material_regression"]
        )
        for metric in PRIMARY_METRICS
    }
    any_controlled_regression = any(concordant.values())
    if all_steps_pass:
        outcome = "composition-controlled-scale-retained"
        follow_up = "no seed-43 scale replay; composition was the R0180 confound"
    elif any_controlled_regression:
        outcome = "composition-controlled-size-regression"
        follow_up = "seed-43 replay at the first materially failing boundary"
    else:
        outcome = "composition-controlled-boundary-or-discordant"
        follow_up = "seed-43 replay at the first sub-0.97 boundary"
    return {
        "outcome": outcome,
        "retention_ratio": RETENTION_RATIO,
        "compound_retention_ratio": COMPOUND_RETENTION_RATIO,
        "primary_metrics": list(PRIMARY_METRICS),
        "cells": normalized,
        "steps": steps,
        "quarter_to_full": compound,
        "concordant_material_regression": concordant,
        "follow_up": follow_up,
        "capacity_activated": outcome == "composition-controlled-size-regression",
    }


__all__ = [
    "CAPABILITY",
    "COMPOUND_RETENTION_RATIO",
    "CORPORA",
    "DIMENSION",
    "EVALUATION_SCHEMA",
    "FULL_COUNTS",
    "FULL_GRAPH_EDGES",
    "FULL_SUCCESSFUL_UPDATES",
    "GRAPH_SCHEMA_PREFIX",
    "HASH_NAMESPACE",
    "NestedScalePromptTrainingInput",
    "POPULATION_CAPABILITY",
    "POPULATION_SCHEMA",
    "POSITIVE_ROWS_PER_UPDATE",
    "PRIMARY_METRICS",
    "RETENTION_RATIO",
    "ROUND_ID",
    "RUNG_COUNTS",
    "RUNG_ROWS",
    "Round0187Error",
    "SEED",
    "SYNTHESIS_SCHEMA",
    "TARGET_POSITIVE_DRAWS_PER_EDGE",
    "TRAIN_SCHEMA_PREFIX",
    "canonical_id_digest",
    "ladder_decision",
    "primary_metric_view",
    "purity_fidelity",
    "select_nested_positions",
    "successful_updates_for_edges",
    "train_checks_close",
    "train_config",
]
