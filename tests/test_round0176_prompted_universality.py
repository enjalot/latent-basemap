"""Contract tests for the negative-Q2-aware prompted universality round."""
from __future__ import annotations

import numpy as np
import pytest

from basemap import round0167_prompted_universality as contract_base
from basemap.round0142_jina_universality import PROBE_ORDER
from basemap.round0176_prompted_universality import (
    PROMPTED_MAP_ORDER,
    ROUND_ID,
    exact_training_overlap_report,
    twonn_correlations,
)
from experiments import (
    prepare_round0167_queue,
    prepare_round0176_queue,
    round0167_nodes,
    round0176_nodes,
)


def test_r0176_correlation_matrix_includes_negative_r0171_artifact() -> None:
    names = ("ROUND_ID", "CAPABILITY", "PROMPTED_MAP_ORDER", "Round0167Error")
    before = {name: getattr(contract_base, name) for name in names}
    cells = [
        {
            "map": map_key,
            "probe": probe,
            "twonn_intrinsic_dimension": float(index + 1),
            "ffr_retention": float(100 - index + map_index),
            "recall10_retention": float(50 - index + map_index),
        }
        for map_index, map_key in enumerate(PROMPTED_MAP_ORDER)
        for index, probe in enumerate(PROBE_ORDER)
    ]
    try:
        assert PROMPTED_MAP_ORDER[-1] == "r0171-prompted-8m-seed42"
        assert len(twonn_correlations(cells)) == 8
    finally:
        for name, value in before.items():
            setattr(contract_base, name, value)


def test_r0176_dispatch_rebinds_schemas_and_map_order(monkeypatch) -> None:
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
    )
    contract_before = {name: getattr(contract_base, name) for name in contract_names}
    node_before = {name: getattr(round0167_nodes, name) for name in node_names}
    observed = {}
    monkeypatch.setattr(
        round0167_nodes,
        "run_job",
        lambda active, job: observed.update({
            "round_id": round0167_nodes.ROUND_ID,
            "maps": round0167_nodes.PROMPTED_MAP_ORDER,
            "schema": round0167_nodes.MAP_PANEL_SCHEMA,
        }),
    )
    try:
        round0176_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}}, {"action": "assemble"}
        )
        assert observed == {
            "round_id": "0176",
            "maps": PROMPTED_MAP_ORDER,
            "schema": "round0176-prompted-universality-map-panel-v1",
        }
    finally:
        for name, value in contract_before.items():
            setattr(contract_base, name, value)
        for name, value in node_before.items():
            setattr(round0167_nodes, name, value)


def test_r0176_binds_accepted_negative_review_without_claiming_capability(
    monkeypatch,
) -> None:
    review = {"canonical_path": "/review-0171.md", "sha256": "review"}
    frontmatter = {
        "round": "round-0171.md",
        "result": "result-0171.md",
        "round_sha256": "round",
        "result_sha256": "result",
        "releases": [],
    }
    monkeypatch.setattr(
        prepare_round0167_queue, "_one_document", lambda *a, **k: review
    )
    monkeypatch.setattr(
        prepare_round0167_queue, "_frontmatter", lambda path: frontmatter
    )
    monkeypatch.setattr(prepare_round0167_queue, "LAB_ROOT", "/labs")

    def signature(path: str):
        digest = "result" if path.endswith("result-0171.md") else "round"
        return {"canonical_path": path, "sha256": digest}

    monkeypatch.setattr(
        prepare_round0167_queue, "expected_input_signature", signature
    )
    base_names = (
        "ROUND_ID",
        "CAPABILITY",
        "PROMPTED_MAP_ORDER",
        "ROUND_ROOT",
        "RELEASE_ROOT",
        "ROUND_FILE",
        "MAPS",
        "Q2_ROUND_ID",
        "Q2_CAPABILITY",
        "Q2_MAP_ROLE",
        "TRAINING_AUDIT_PATHS",
        "TRAINING_AUDIT_POLICY",
        "GPU_HOURS_MINIMUM",
        "GPU_HOURS_EXPECTED",
        "GPU_HOURS_MAXIMUM",
        "HANDLER_MODULE",
        "QUEUE_SCHEMA",
        "QUEUE_LABEL",
    )
    contract_names = (
        "ROUND_ID",
        "CAPABILITY",
        "PROMPTED_MAP_ORDER",
        "Round0167Error",
    )
    previous_base = {
        name: getattr(prepare_round0167_queue, name) for name in base_names
    }
    previous_contract = {
        name: getattr(contract_base, name) for name in contract_names
    }
    try:
        prepare_round0176_queue._configure()
        assert prepare_round0167_queue.Q2_CAPABILITY is None
        assert prepare_round0167_queue._accepted_any_review("0171") == [
            {"canonical_path": "/labs/round-0171.md", "sha256": "round"},
            {"canonical_path": "/labs/result-0171.md", "sha256": "result"},
            review,
        ]
        assert (
            "not a released map capability"
            in prepare_round0167_queue.Q2_MAP_ROLE
        )
        assert prepare_round0167_queue.GPU_HOURS_MAXIMUM == 2.5
        assert set(prepare_round0167_queue.TRAINING_AUDIT_PATHS) == {
            "r0115-r0117-prompted-2m",
            "r0171-prompted-8m",
        }
    finally:
        for name, value in previous_base.items():
            setattr(prepare_round0167_queue, name, value)
        for name, value in previous_contract.items():
            setattr(contract_base, name, value)


def test_positive_q2_rounds_still_require_the_named_capability(monkeypatch) -> None:
    review = {"canonical_path": "/review.md", "sha256": "review"}
    frontmatter = {
        "round": "round.md",
        "result": "result.md",
        "round_sha256": "round",
        "result_sha256": "result",
        "releases": [],
    }
    monkeypatch.setattr(
        prepare_round0167_queue, "_one_document", lambda *a, **k: review
    )
    monkeypatch.setattr(
        prepare_round0167_queue, "_frontmatter", lambda path: frontmatter
    )
    monkeypatch.setattr(prepare_round0167_queue, "LAB_ROOT", "/labs")
    monkeypatch.setattr(
        prepare_round0167_queue,
        "expected_input_signature",
        lambda path: {
            "canonical_path": path,
            "sha256": "result" if path.endswith("result.md") else "round",
        },
    )
    monkeypatch.setattr(prepare_round0167_queue, "Q2_CAPABILITY", "positive-map")
    with pytest.raises(RuntimeError, match="did not release required"):
        prepare_round0167_queue._accepted_any_review("0171")


def test_training_overlap_policy_blocks_queries_but_reports_corpus() -> None:
    corpus = np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float16)
    queries = np.asarray([[0.0, 1.0]], dtype=np.float16)
    control = np.asarray([[0.25, 0.75]], dtype=np.float16)
    entries = [
        {
            "label": "probe",
            "split": "corpus",
            "values": corpus,
            "source_rows": np.asarray([10, 11], dtype=np.int64),
        },
        {
            "label": "probe",
            "split": "queries",
            "values": queries,
            "source_rows": np.asarray([20], dtype=np.int64),
        },
        {
            "label": "fineweb-control",
            "split": "control",
            "values": control,
            "source_rows": np.asarray([30], dtype=np.int64),
        },
    ]
    diagnostic = exact_training_overlap_report(
        entries=entries,
        training_sources={
            "2m": np.asarray([[1.0, 0.0], [0.1, 0.9]], dtype=np.float16)
        },
        block_rows=1,
    )
    assert diagnostic["passed"] is True
    assert diagnostic["all_rows_training_disjoint"] is False
    assert diagnostic["diagnostic_corpus_overlap_count"] == 1
    assert diagnostic["blocking_query_or_control_overlap_count"] == 0

    blocked = exact_training_overlap_report(
        entries=entries,
        training_sources={
            "8m": np.asarray([[0.0, 1.0], [0.25, 0.75]], dtype=np.float16)
        },
        block_rows=1,
    )
    assert blocked["passed"] is False
    assert blocked["blocking_query_or_control_overlap_count"] == 2
    assert {item["split"] for item in blocked["exact_training_family_overlaps"]} == {
        "queries",
        "control",
    }
