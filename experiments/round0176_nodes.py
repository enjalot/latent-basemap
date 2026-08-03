"""Execute the negative-Q2-aware R0176 prompted-universality panel."""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap import round0167_prompted_universality as contract_base
from basemap.round0142_jina_universality import PROBE_ORDER
from basemap.round0176_prompted_universality import (
    CAPABILITY,
    PROMPTED_MAP_ORDER,
    ROUND_ID,
    Round0176Error,
    exact_training_overlap_report,
)
from experiments import round0167_nodes as base


def _configure() -> None:
    contract_bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "PROMPTED_MAP_ORDER": PROMPTED_MAP_ORDER,
        "Round0167Error": Round0176Error,
    }
    for name, value in contract_bindings.items():
        setattr(contract_base, name, value)
    node_bindings = {
        **contract_bindings,
        "CANARY_SCHEMA": "round0176-prompt-model-canary-v1",
        "PROBE_SCHEMA": "round0176-prompted-probe-embeddings-v1",
        "CONTROL_SCHEMA": "round0176-prompted-fineweb-control-v1",
        "MAP_PANEL_SCHEMA": "round0176-prompted-universality-map-panel-v1",
    }
    for name, value in node_bindings.items():
        setattr(base, name, value)


def run_training_disjoint_audit(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0176 prompted training-overlap audit"
    )
    started = time.monotonic()
    entries: list[dict[str, Any]] = []
    probe_receipts: dict[str, Any] = {}
    for name in PROBE_ORDER:
        corpus, queries, corpus_rows, query_rows, inputs = base._load_probe(
            str(job["probe_outputs"][name]), name
        )
        probe_receipts[name] = inputs
        entries.extend((
            {
                "label": name,
                "split": "corpus",
                "values": corpus,
                "source_rows": np.asarray(corpus_rows, dtype=np.int64),
            },
            {
                "label": name,
                "split": "queries",
                "values": queries,
                "source_rows": np.asarray(query_rows, dtype=np.int64),
            },
        ))

    control_receipt_path = os.path.join(
        str(job["control_output"]), "receipt.json"
    )
    control_receipt = base._read_sealed(
        control_receipt_path, label="R0176 prompted FineWeb control receipt"
    )
    control_signature = base._signature(
        control_receipt["embeddings"], label="R0176 prompted FineWeb control"
    )
    control = np.load(
        control_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if (
        control_receipt.get("round_id") != ROUND_ID
        or control_receipt.get("prompt_applied") is not True
        or control.shape != (60_000, 768)
        or control.dtype != np.float16
    ):
        raise Round0176Error("R0176 prompted FineWeb control changed")
    control_rows: list[np.ndarray] = []
    for name in PROBE_ORDER:
        corpus_rows, query_rows, _signature = base._coordinate_rows(
            job["control_coordinates"][name],
            label=f"{name} control",
            control=True,
        )
        control_rows.extend((corpus_rows, query_rows))
    audited_control_rows = np.unique(np.concatenate(control_rows)).astype(
        np.int64, copy=False
    )
    entries.append({
        "label": "fineweb-control",
        "split": "control",
        "values": np.asarray(control[audited_control_rows]),
        "source_rows": audited_control_rows,
    })

    training_arrays: dict[str, np.ndarray] = {}
    training_receipts: dict[str, Any] = {}
    for label, source in job["training_sources"].items():
        signature = base._signature(
            source["signature"], label=f"R0176 {label} training matrix"
        )
        rows = int(source["rows"])
        training_arrays[str(label)] = np.memmap(
            signature["canonical_path"],
            dtype="<f2",
            mode="r",
            shape=(rows, 768),
        )
        training_receipts[str(label)] = {
            "signature": signature,
            "rows": rows,
        }

    report = exact_training_overlap_report(
        entries=entries,
        training_sources=training_arrays,
    )
    receipt = base.seal({
        "schema": "round0176-prompted-training-overlap-audit-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        **report,
        "training_source_receipts": training_receipts,
        "probe_receipts": probe_receipts,
        "control_receipt": expected_input_signature(control_receipt_path),
        "control_rows_audited": int(len(audited_control_rows)),
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "audit.json"), receipt, immutable=True
    )
    if receipt["passed"] is not True:
        raise Round0176Error(
            "R0176 prompted query/control rows overlap map training: "
            f"{receipt['blocking_query_or_control_overlap_count']} exact copies"
        )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any] | None = None) -> None:
    _configure()
    if job is not None and job.get("action") == "audit_training_disjoint":
        return run_training_disjoint_audit(active, job)
    base.run_job(dict(active), dict(job) if job is not None else None)
