"""Contract tests for the R0171-backed prompted universality correction."""
from __future__ import annotations

import pytest

from basemap import round0167_prompted_universality as contract_base
from basemap.round0142_jina_universality import PROBE_ORDER
from basemap.round0172_prompted_universality import (
    PROMPTED_MAP_ORDER,
    ROUND_ID,
    twonn_correlations,
)
from experiments import (
    prepare_round0167_queue,
    round0167_nodes,
    round0172_nodes,
)


def test_r0172_correlation_matrix_names_reviewed_r0171_map() -> None:
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


def test_r0172_dispatch_rebinds_schemas_and_map_order(monkeypatch) -> None:
    contract_names = ("ROUND_ID", "CAPABILITY", "PROMPTED_MAP_ORDER", "Round0167Error")
    node_names = (*contract_names, "CANARY_SCHEMA", "PROBE_SCHEMA", "CONTROL_SCHEMA", "MAP_PANEL_SCHEMA")
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
        round0172_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}}, {"action": "assemble"}
        )
        assert observed == {
            "round_id": "0172",
            "maps": PROMPTED_MAP_ORDER,
            "schema": "round0172-prompted-universality-map-panel-v1",
        }
    finally:
        for name, value in contract_before.items():
            setattr(contract_base, name, value)
        for name, value in node_before.items():
            setattr(round0167_nodes, name, value)


def test_q2_review_must_release_the_positive_map_capability(monkeypatch) -> None:
    review = {"canonical_path": "/review-0171.md", "sha256": "review"}
    frontmatter = {
        "round": "round-0171.md",
        "result": "result-0171.md",
        "round_sha256": "round",
        "result_sha256": "result",
        "releases": [],
    }

    monkeypatch.setattr(
        prepare_round0167_queue,
        "_one_document",
        lambda *args, **kwargs: review,
    )
    monkeypatch.setattr(
        prepare_round0167_queue,
        "_frontmatter",
        lambda path: frontmatter,
    )
    monkeypatch.setattr(prepare_round0167_queue, "LAB_ROOT", "/labs")

    def signature(path: str):
        digest = "result" if path.endswith("result-0171.md") else "round"
        return {"canonical_path": path, "sha256": digest}

    monkeypatch.setattr(
        prepare_round0167_queue, "expected_input_signature", signature
    )
    capability = (
        "jina-document-english-8m-prompted-map-seed42-sharded-fp32-ivf-v1"
    )
    monkeypatch.setattr(prepare_round0167_queue, "Q2_CAPABILITY", capability)

    with pytest.raises(RuntimeError, match="did not release required capability"):
        prepare_round0167_queue._accepted_any_review("0171")

    frontmatter["releases"] = [f"capability:{capability}"]
    assert prepare_round0167_queue._accepted_any_review("0171") == [
        {"canonical_path": "/labs/round-0171.md", "sha256": "round"},
        {"canonical_path": "/labs/result-0171.md", "sha256": "result"},
        review,
    ]
