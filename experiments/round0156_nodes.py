"""Retag the reviewed R0132/R0152 mechanics for Round 0156."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from basemap import round0152_scale_rescue as contract
from basemap.round0156_scale_rescue import (
    CAPABILITY,
    DECISION_SCHEMA,
    FUNCTIONAL_SCHEMA,
    GRAPH_DEGREE,
    GRAPH_K,
    GRAPH_PART_SCHEMA,
    GRAPH_SCHEMA,
    GRAPH_SHARD_SCHEMA,
    INDEX_SCHEMA,
    NATIVE_SCHEMA,
    N_NEIGHBORS,
    OOD_SCHEMA,
    OUTCOME_FAIL,
    OUTCOME_INVALID,
    OUTCOME_PASS,
    PARENT_CAPABILITY,
    PARENT_ROUND_ID,
    PIPELINE,
    PIPELINE_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    PRODUCTION_CONFIG_SCHEMA,
    QUALIFICATION_SCHEMA,
    RETAINED_ROWS,
    ROUND_ID,
    SEED,
    SUBSET_SCHEMA,
    TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA,
    UPDATE_RULE,
)
from experiments import round0152_nodes as base


PATCH = {
    "ROUND_ID": ROUND_ID,
    "CAPABILITY": CAPABILITY,
    "PARENT_ROUND_ID": PARENT_ROUND_ID,
    "PARENT_CAPABILITY": PARENT_CAPABILITY,
    "RETAINED_ROWS": RETAINED_ROWS,
    "GRAPH_K": GRAPH_K,
    "N_NEIGHBORS": N_NEIGHBORS,
    "SEED": SEED,
    "SUBSET_SCHEMA": SUBSET_SCHEMA,
    "INDEX_SCHEMA": INDEX_SCHEMA,
    "QUALIFICATION_SCHEMA": QUALIFICATION_SCHEMA,
    "GRAPH_SHARD_SCHEMA": GRAPH_SHARD_SCHEMA,
    "GRAPH_PART_SCHEMA": GRAPH_PART_SCHEMA,
    "GRAPH_SCHEMA": GRAPH_SCHEMA,
    "TRAIN_CONFIG_SCHEMA": TRAIN_CONFIG_SCHEMA,
    "PRODUCTION_CONFIG_SCHEMA": PRODUCTION_CONFIG_SCHEMA,
    "TRAIN_RECEIPT_SCHEMA": TRAIN_RECEIPT_SCHEMA,
    "NATIVE_SCHEMA": NATIVE_SCHEMA,
    "OOD_SCHEMA": OOD_SCHEMA,
    "FUNCTIONAL_SCHEMA": FUNCTIONAL_SCHEMA,
    "DECISION_SCHEMA": DECISION_SCHEMA,
    "PIPELINE": PIPELINE,
    "PIPELINE_SCHEMA": PIPELINE_SCHEMA,
    "POSITIVE_DESTINATION_POLICY": POSITIVE_DESTINATION_POLICY,
    "UPDATE_RULE": UPDATE_RULE,
    "GRAPH_DEGREE": GRAPH_DEGREE,
    "OUTCOME_PASS": OUTCOME_PASS,
    "OUTCOME_FAIL": OUTCOME_FAIL,
    "OUTCOME_INVALID": OUTCOME_INVALID,
    "FULL_25M_TEST_ON_PASS": False,
}


@contextmanager
def _configured() -> Iterator[None]:
    """Temporarily retag inherited mechanics without contaminating imports."""
    overrides = {
        "PARENT_CENSUS_FIELD": "r0155_census",
        "INHERITED_NODE_OVERRIDES": {
        "SEARCH_POSITIVE_OUTCOME": (
            "qualified-fixed-r0105-policy-on-r0155-historical-prefix-universe"
        ),
        "SEARCH_NEGATIVE_OUTCOME": (
            "fixed-r0105-policy-failed-on-r0155-historical-prefix-universe"
        ),
        "SEARCH_EXACT_UNIVERSE_CHECK": (
            "candidate_universe_is_exact_r0155_historical_prefix_subset"
        ),
        "GRAPH_CANDIDATE_UNIVERSE": "exact R0155 historical-prefix subset",
        "TRANSFORM_MAP_KEY": "r0156-diverse-jina-historical-prefix-12p5m-seed42",
        "TRANSFORM_SCIENTIFIC_UNIVERSE": (
            "R0155 exact 12,485,206-row historical-prefix population"
        ),
        "TRANSFORM_ROW_ORDER": "R0155 historical-prefix compact order",
        "NATIVE_TREATMENT_KEY": "accepted_25m_on_r0155_rows",
        "NATIVE_SHARED_ROWS_CHECK": "same_r0155_candidate_rows",
        "NATIVE_GLOBAL_FFR_ROLE": "diagnostic-only; execution-validity evidence",
        },
    }
    snapshots: list[tuple[object, str, bool, Any]] = []
    try:
        for module in (contract, base):
            for name, value in PATCH.items():
                if hasattr(module, name):
                    snapshots.append((module, name, True, getattr(module, name)))
                    setattr(module, name, value)
        for name, value in overrides.items():
            present = hasattr(base, name)
            snapshots.append((base, name, present, getattr(base, name, None)))
            setattr(base, name, value)
        yield
    finally:
        for module, name, present, value in reversed(snapshots):
            if present:
                setattr(module, name, value)
            else:
                delattr(module, name)


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> Any:
    with _configured():
        return base.run_job(active, job)
