"""Execute the one-train R0149 drop-only historical-row decomposition."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0108_evaluation import seal
from basemap.round0140_subsystem_bisection import CURRENT_GRAPH_CURRENT_HOST, metric_view
from basemap.round0142_jina_universality import PROBE_ORDER
from basemap.round0147_row_policy import TREATMENT as SIZE_PRESERVING_TREATMENT
from basemap.round0149_drop_only import (
    CAPABILITY,
    DIMENSION,
    RAW_PREFIX_EXCLUDED_ROWS,
    RAW_PREFIX_ROWS,
    ROUND_ID,
    ROWS,
    ROW_UNIVERSE,
    TREATMENT,
    Round0149Error,
    build_decision,
    derive_drop_only_selection,
    treatment_preprocessing_stamp,
    treatment_train_config,
)
from experiments import round0147_nodes as base


EVALUATION_ROWS = RAW_PREFIX_ROWS
RENDER_SEED = 14_900
TREATMENT_ROLE = "drop-only-historical-eligibility-treatment"


def _configure_base() -> None:
    """Bind the proven R0147 execution machinery to the R0149 contract."""
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.TREATMENT = TREATMENT
    base.ROWS = ROWS
    base.EVALUATION_ROWS = EVALUATION_ROWS
    base.RENDER_SEED = RENDER_SEED
    base.ARTIFACT_SCHEMA_PREFIX = "round0149"
    base.ROW_UNIVERSE = ROW_UNIVERSE
    base.TREATMENT_ROLE = TREATMENT_ROLE
    base.SOURCE_PROOF_ROW_ORDER = (
        "eligible members of the raw R0037 historical 2M prefix in exact "
        "historical order, with no replacement"
    )
    base.GRAPH_METRIC_INPUT = (
        "exact R0147 staged fp16 prefix for the R0149 drop-only treatment, "
        "normalized in fp32"
    )
    base.GRAPH_SEMANTICS = (
        "R0104 current builder on the R0149 drop-only historical population"
    )
    base.GRAPH_VERIFIED_BY = "round0149-current-graph-drop-only-builder-v1"
    base.StagedTreatmentArray = DropOnlyStagedArray
    base._selection_receipt = _selection_receipt
    base._source_proof = _source_proof
    base.treatment_preprocessing_stamp = treatment_preprocessing_stamp
    base.treatment_train_config = treatment_train_config


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0149Error(f"JSON object required: {path}")
    return value


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0149Error(f"{label} bytes changed")
    return actual


def _read_sealed(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _signature(expected, label=label)
    value = _read_json(signature["canonical_path"])
    base.validate_seal(value, label=label)
    return value


def run_materialize_selection(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0149 drop-only historical selection"
    )
    started = time.monotonic()
    parent_signature = _signature(
        job["r0147_selection_receipt"], label="accepted R0147 selection receipt"
    )
    parent = _read_sealed(parent_signature, label="accepted R0147 selection receipt")
    if (
        parent.get("round_id") != "0147"
        or parent.get("target_rows") != RAW_PREFIX_ROWS
        or parent.get("size_preserving") is not True
        or parent.get("excluded_rows_absent") is not True
    ):
        raise Round0149Error("accepted R0147 selection identity changed")
    parent_arrays_signature = _signature(
        parent["selection_arrays"], label="accepted R0147 selection arrays"
    )
    staged_signature = _signature(
        parent["staged_source"], label="accepted R0147 staged source"
    )
    with np.load(parent_arrays_signature["canonical_path"], allow_pickle=False) as archive:
        arrays, summary = derive_drop_only_selection(
            {key: np.asarray(archive[key]) for key in archive.files},
            parent_summary=parent["selection_summary"],
        )
    selection_path = os.path.join(output, "selection.npz")
    atomic_save_new_npz(selection_path, immutable=True, **arrays)
    selection_signature = expected_input_signature(selection_path)
    receipt = seal({
        "schema": "round0149-drop-only-historical-selection-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "parent_r0147_selection_receipt": parent_signature,
        "parent_r0147_selection_arrays": parent_arrays_signature,
        "selection_arrays": selection_signature,
        "staged_source": staged_signature,
        "source_prefix_rows": [0, ROWS],
        "selection_summary": summary,
        "target_rows": ROWS,
        "dimension": DIMENSION,
        "dtype": "<f2",
        "size_preserving": False,
        "replacement_rows": 0,
        "historical_order_preserved": True,
        "excluded_rows_absent": True,
        "row_policy_includes_induced_graph_change": True,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "selection-receipt.json"), receipt, immutable=True
    )


def _selection_receipt(output: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = os.path.join(output, "selection-receipt.json")
    signature = expected_input_signature(path)
    receipt = _read_sealed(signature, label="R0149 drop-only selection")
    if (
        receipt.get("round_id") != ROUND_ID
        or receipt.get("target_rows") != ROWS
        or receipt.get("size_preserving") is not False
        or receipt.get("replacement_rows") != 0
    ):
        raise Round0149Error("R0149 selection receipt changed")
    return receipt, signature


class DropOnlyStagedArray:
    """Read-only prefix view of the already materialized R0147 eligible stream."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        self.signature = _signature(
            receipt["staged_source"], label="R0147 staged treatment source"
        )
        parent = np.load(
            self.signature["canonical_path"], mmap_mode="r", allow_pickle=False
        )
        if (
            parent.shape != (RAW_PREFIX_ROWS, DIMENSION)
            or parent.dtype != np.dtype("<f2")
            or not parent.flags.c_contiguous
        ):
            raise Round0149Error("R0147 staged source geometry changed")
        self.array = parent[:ROWS]
        self.shape = self.array.shape
        self.dtype = self.array.dtype
        self.selection_signature = dict(receipt["selection_arrays"])
        self.segments = [{
            "global_row_start": 0,
            "global_row_stop": ROWS,
            "dataset": "r0149-drop-only-r0147-staged-prefix",
            "shard": self.signature,
            "shard_rows": RAW_PREFIX_ROWS,
            "shard_row_start": 0,
            "shard_row_stop": ROWS,
            "selection": self.selection_signature,
        }]

    def __len__(self) -> int:
        return ROWS

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.array[key]


def _source_proof(
    selection_output: str,
) -> tuple[DropOnlyStagedArray, dict[str, Any]]:
    receipt, receipt_signature = _selection_receipt(selection_output)
    source = DropOnlyStagedArray(receipt)
    return source, {
        "schema": "round0149-drop-only-historical-source-proof-v1",
        "rows": ROWS,
        "evaluation_rows": EVALUATION_ROWS,
        "dimension": DIMENSION,
        "dtype": "<f2",
        "staged_parent_source": source.signature,
        "source_prefix_rows": [0, ROWS],
        "selection_receipt": receipt_signature,
        "selection_arrays": source.selection_signature,
        "selection_summary": receipt["selection_summary"],
        "segments": source.segments,
        "row_order": (
            "eligible members of the raw historical 2M prefix in exact order; "
            "no rows beyond the prefix and no replacement"
        ),
    }


def _panel_signature(expected: Mapping[str, Any], *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = _signature(expected, label=label)
    return _read_sealed(signature, label=label), signature


def run_decision(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0149 drop-only decomposition decision"
    )
    functional_path = os.path.join(
        str(job["functional_output"]), "functional-panel.json"
    )
    functional_signature = expected_input_signature(functional_path)
    functional = _read_sealed(
        functional_signature, label="R0149 functional panel"
    )
    if functional.get("round_id") != ROUND_ID:
        raise Round0149Error("R0149 functional panel identity changed")

    r0147_decision, r0147_decision_signature = _panel_signature(
        job["r0147_decision"], label="accepted R0147 decision"
    )
    r0147_panel, r0147_panel_signature = _panel_signature(
        job["r0147_functional_panel"], label="accepted R0147 functional panel"
    )
    if (
        r0147_decision.get("round_id") != "0147"
        or r0147_decision.get("capability")
        != "jina-2m-historical-row-policy-duplicate-control-v1"
        or r0147_decision.get("outcome")
        != "eligible-historical-row-policy-does-not-restore"
        or r0147_decision.get("duplicate_control_compatible_with_restoration")
        is not False
        or r0147_decision.get("functional_panel") != r0147_panel_signature
        or r0147_panel.get("round_id") != "0147"
    ):
        raise Round0149Error("accepted R0147 negative evidence changed")

    cells = {
        CURRENT_GRAPH_CURRENT_HOST: functional["cells"][CURRENT_GRAPH_CURRENT_HOST],
        SIZE_PRESERVING_TREATMENT: r0147_panel["cells"][SIZE_PRESERVING_TREATMENT],
        TREATMENT: functional["cells"][TREATMENT],
    }
    if metric_view(cells[CURRENT_GRAPH_CURRENT_HOST]) != metric_view(
        r0147_panel["cells"][CURRENT_GRAPH_CURRENT_HOST]
    ):
        raise Round0149Error("R0140 control differs across R0147/R0149 panels")
    selection, selection_signature = _selection_receipt(str(job["selection_output"]))
    decision = build_decision(
        cells, selection_summary=selection["selection_summary"]
    )

    universality: dict[str, Any] = {}
    for key, expected in job["r0147_universality"].items():
        panel, signature = _panel_signature(
            expected, label=f"accepted R0147 {key} universality"
        )
        if (
            panel.get("round_id") != "0147"
            or panel.get("map_key") != key
            or panel.get("probe_order") != list(PROBE_ORDER)
        ):
            raise Round0149Error("accepted R0147 universality identity changed")
        universality[key] = {
            "panel": signature,
            "metrics": {
                name: panel["probes"][name]["metrics"] for name in PROBE_ORDER
            },
        }
    drop_path = os.path.join(
        str(job["drop_universality_output"]), "universality-panel.json"
    )
    drop_signature = expected_input_signature(drop_path)
    drop = _read_sealed(drop_signature, label="R0149 drop-only universality")
    if (
        drop.get("round_id") != ROUND_ID
        or drop.get("map_key") != TREATMENT
        or drop.get("probe_order") != list(PROBE_ORDER)
    ):
        raise Round0149Error("R0149 universality identity changed")
    universality[TREATMENT] = {
        "panel": drop_signature,
        "metrics": {name: drop["probes"][name]["metrics"] for name in PROBE_ORDER},
    }
    diagnostic_deltas = {
        baseline: {
            name: {
                metric: (
                    float(universality[TREATMENT]["metrics"][name][metric])
                    - float(universality[baseline]["metrics"][name][metric])
                )
                for metric in ("ffr_retention", "recall10_retention")
                if universality[TREATMENT]["metrics"][name].get(metric) is not None
                and universality[baseline]["metrics"][name].get(metric) is not None
            }
            for name in PROBE_ORDER
        }
        for baseline in (CURRENT_GRAPH_CURRENT_HOST, SIZE_PRESERVING_TREATMENT)
    }
    receipt = seal({
        **decision,
        "release_sha": active["manifest"]["release_sha"],
        "functional_panel": functional_signature,
        "selection_receipt": selection_signature,
        "accepted_r0147_decision": r0147_decision_signature,
        "accepted_r0147_functional_panel": r0147_panel_signature,
        "universality_diagnostic": universality,
        "universality_drop_minus_baselines": diagnostic_deltas,
        "universality_used_for_selector": False,
        "row_policy_includes_induced_graph_change": True,
        "unique_causal_factor_claimed": False,
    })
    atomic_write_new_json(os.path.join(output, "decision.json"), receipt, immutable=True)


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> None:
    _configure_base()
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0149Error("R0149 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "materialize_selection":
        return run_materialize_selection(active, job)
    if action == "build_graph":
        return base.run_build_graph(active, job)
    if action == "train":
        return base.run_train(active, job)
    if action == "functional_panel":
        return base.run_functional_panel(active, job)
    if action == "universality_panel":
        return base.run_universality_panel(active, job)
    if action == "decide":
        return run_decision(active, job)
    raise Round0149Error(f"unknown R0149 action: {action!r}")
