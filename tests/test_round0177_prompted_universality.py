"""Contract tests for the duplicate-aware R0177 prompted panel."""
from __future__ import annotations

import numpy as np
import pytest

from basemap import round0167_prompted_universality as contract_base
from basemap.round0177_prompted_universality import (
    CAPABILITY,
    PROMPTED_MAP_ORDER,
    ROUND_ID,
)
from experiments import (
    prepare_round0167_queue,
    prepare_round0177_queue,
    round0167_nodes,
    round0177_nodes,
)
from experiments import round0176_nodes as audit_nodes


def test_r0177_configuration_binds_duplicate_sensitivity() -> None:
    names = (
        "ROUND_ID",
        "CAPABILITY",
        "PROMPTED_MAP_ORDER",
        "ROUND_ROOT",
        "ROUND_FILE",
        "GPU_HOURS_MINIMUM",
        "GPU_HOURS_EXPECTED",
        "GPU_HOURS_MAXIMUM",
        "HANDLER_MODULE",
        "QUEUE_SCHEMA",
        "QUEUE_LABEL",
        "PROBE_FAMILY_POLICY",
    )
    previous = {
        name: getattr(prepare_round0167_queue, name)
        for name in names
    }
    contract_names = (
        "ROUND_ID",
        "CAPABILITY",
        "PROMPTED_MAP_ORDER",
        "Round0167Error",
    )
    contract_previous = {
        name: getattr(contract_base, name)
        for name in contract_names
    }
    try:
        prepare_round0177_queue._configure()
        assert prepare_round0167_queue.ROUND_ID == "0177"
        assert prepare_round0167_queue.CAPABILITY == CAPABILITY
        assert (
            prepare_round0167_queue.PROMPTED_MAP_ORDER
            == PROMPTED_MAP_ORDER
        )
        assert prepare_round0167_queue.GPU_HOURS_EXPECTED == 0.35
        assert prepare_round0167_queue.GPU_HOURS_MAXIMUM == 2.5
        assert "paired sensitivity" in (
            prepare_round0167_queue.PROBE_FAMILY_POLICY
        )
    finally:
        for name, value in previous.items():
            setattr(prepare_round0167_queue, name, value)
        for name, value in contract_previous.items():
            setattr(contract_base, name, value)


def test_r0177_family_audit_reports_cross_split_without_hiding_it() -> None:
    corpus = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float16,
    )
    queries = np.asarray(
        [[1.0, 0.0], [0.5, 0.5]],
        dtype=np.float16,
    )
    previous_dimension = round0167_nodes.DIMENSION
    previous_allow = round0167_nodes.ALLOW_CROSS_SPLIT_FAMILIES
    try:
        round0167_nodes.DIMENSION = 2
        round0167_nodes.ALLOW_CROSS_SPLIT_FAMILIES = False
        with pytest.raises(
            contract_base.Round0167Error,
            match="exact families overlap",
        ):
            round0167_nodes._exact_family_audit(corpus, queries)

        round0167_nodes.ALLOW_CROSS_SPLIT_FAMILIES = True
        report = round0167_nodes._exact_family_audit(corpus, queries)
        assert report["cross_split_family_overlap"] == 1
        assert report["query_rows_with_exact_corpus_copy"] == 1
        assert report["corpus_duplicate_rows"] == 1
        assert report["passed"] is None
    finally:
        round0167_nodes.DIMENSION = previous_dimension
        round0167_nodes.ALLOW_CROSS_SPLIT_FAMILIES = previous_allow


def test_r0177_sensitivity_excludes_union_of_paired_leakage() -> None:
    probe_corpus = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float16
    )
    probe_queries = np.asarray(
        [[1.0, 0.0], [0.5, 0.5], [0.2, 0.8]],
        dtype=np.float16,
    )
    control_corpus = np.asarray(
        [[0.25, 0.75], [0.75, 0.25]], dtype=np.float16
    )
    control_queries = np.asarray(
        [[0.1, 0.9], [0.75, 0.25], [0.2, 0.8]],
        dtype=np.float16,
    )
    keep, audit = round0167_nodes._duplicate_sensitivity_mask(
        probe_corpus=probe_corpus,
        probe_queries=probe_queries,
        control_corpus=control_corpus,
        control_queries=control_queries,
    )
    assert keep.tolist() == [False, False, True]
    assert audit["excluded_query_positions"] == [0, 1]
    assert audit["probe_copy_query_positions"] == [0]
    assert audit["control_copy_query_positions"] == [1]
    assert audit["retained_query_rows"] == 1
    assert (
        audit["probe_copy_audit"][
            "query_rows_with_exact_reference_copy"
        ]
        == 1
    )
    assert (
        audit["control_copy_audit"][
            "query_rows_with_exact_reference_copy"
        ]
        == 1
    )


def test_r0177_dispatch_enables_dual_view(monkeypatch) -> None:
    contract_names = (
        "ROUND_ID",
        "CAPABILITY",
        "PROMPTED_MAP_ORDER",
        "Round0167Error",
    )
    node_names = (
        *contract_names,
        "CANARY_SCHEMA",
        "PROBE_SCHEMA",
        "CONTROL_SCHEMA",
        "MAP_PANEL_SCHEMA",
        "ALLOW_CROSS_SPLIT_FAMILIES",
        "DUPLICATE_SENSITIVITY",
    )
    contract_previous = {
        name: getattr(contract_base, name)
        for name in contract_names
    }
    node_previous = {
        name: getattr(round0167_nodes, name)
        for name in node_names
    }
    audit_names = (
        "ROUND_ID",
        "CAPABILITY",
        "PROMPTED_MAP_ORDER",
        "Round0176Error",
        "exact_training_overlap_report",
    )
    audit_previous = {
        name: getattr(audit_nodes, name)
        for name in audit_names
    }
    observed = {}
    monkeypatch.setattr(
        round0167_nodes,
        "run_job",
        lambda active, job: observed.update({
            "round": round0167_nodes.ROUND_ID,
            "allow": round0167_nodes.ALLOW_CROSS_SPLIT_FAMILIES,
            "sensitivity": round0167_nodes.DUPLICATE_SENSITIVITY,
            "schema": round0167_nodes.MAP_PANEL_SCHEMA,
        }),
    )
    try:
        round0177_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}},
            {"action": "assemble"},
        )
        assert observed == {
            "round": "0177",
            "allow": True,
            "sensitivity": True,
            "schema": "round0177-prompted-universality-map-panel-v1",
        }
    finally:
        for name, value in contract_previous.items():
            setattr(contract_base, name, value)
        for name, value in node_previous.items():
            setattr(round0167_nodes, name, value)
        for name, value in audit_previous.items():
            setattr(audit_nodes, name, value)
