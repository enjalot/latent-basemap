from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from basemap.round0105_search import GROUPS
from basemap.round0108_evaluation import (
    IN_MIX_LANGUAGES,
    POLISH,
    Round0108Error,
    core_geometry_decision,
    exact_split_duplicate_diagnostics,
    fixed_probe_split,
    headline_ood_decision,
    jina_density_floor,
    map_family_sizes,
    projection_metrics,
)
from experiments.prepare_round0108_queue import GRAPH_MANIFEST, PART_OUTPUTS
from experiments.round0108_nodes import _gather_directed_memberships


def test_graph_inputs_bind_successful_r0106_attempt() -> None:
    expected_root = "/round-0106/queue-attempt-3/artifacts/"
    assert expected_root in GRAPH_MANIFEST
    assert all(expected_root in path for path in PART_OUTPUTS.values())


def test_directed_membership_gather_supports_eliminated_zero_weights() -> None:
    targets, weights, counts = _gather_directed_memberships(
        np.asarray([10, 10, 10, 11, 11], dtype=np.int32),
        np.asarray([1, 2, 3, 4, 5], dtype=np.int32),
        np.asarray([0.9, 0.5, 0.1, 1.0, 0.2], dtype=np.float32),
        np.asarray([11, 10], dtype=np.int64),
    )
    assert counts.tolist() == [2, 3]
    assert targets[0, :3].tolist() == [4, 5, -1]
    assert targets[1, :4].tolist() == [1, 2, 3, -1]
    assert weights[0, :3].tolist() == pytest.approx([1.0, 0.2, 0.0])
    assert weights[1, :4].tolist() == pytest.approx([0.9, 0.5, 0.1, 0.0])


def test_directed_membership_gather_rejects_missing_anchor_source() -> None:
    with pytest.raises(Round0108Error):
        _gather_directed_memberships(
            np.asarray([10], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            np.asarray([0.9], dtype=np.float32),
            np.asarray([11], dtype=np.int64),
        )


def _density_cell(point: float, sd: float, null: float) -> dict:
    return {
        "density_v2": {
            "correlation": point,
            "bootstrap": {"standard_deviation": sd},
            "permuted_radius_null": {
                "absolute_99_9_percentile": null
            },
        }
    }


def test_jina_density_floor_uses_only_two_preregistered_cells() -> None:
    floor = jina_density_floor({
        "seed42": _density_cell(0.20, 0.01, 0.03),
        "seed43": _density_cell(0.18, 0.02, 0.04),
    })
    assert floor["proposed_floor"] == pytest.approx(0.12)
    assert floor["registered_floor"] == pytest.approx(0.12)
    assert floor["gating_floor_registered"] is True

    failed = jina_density_floor({
        "seed42": _density_cell(0.05, 0.02, 0.03),
        "seed43": _density_cell(0.04, 0.01, 0.03),
    })
    assert failed["proposed_floor"] < 0
    assert failed["registered_floor"] is None


def test_fixed_probe_split_is_deterministic_disjoint_and_in_tail() -> None:
    first = fixed_probe_split(
        row_start=835_454, row_stop=2_000_000, seed=108
    )
    second = fixed_probe_split(
        row_start=835_454, row_stop=2_000_000, seed=108
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(first[0]) == 49_500
    assert len(first[1]) == 500
    assert int(first[0].min()) >= 835_454
    assert len(np.intersect1d(*first)) == 0


def test_exact_duplicate_audit_distinguishes_within_and_cross_split_families(
) -> None:
    corpus = np.asarray(
        [[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=np.float16,
    )
    queries = np.asarray(
        [[4, 5, 6], [10, 11, 12]], dtype=np.float16
    )
    report = exact_split_duplicate_diagnostics(corpus, queries)
    assert report["exact_nontrivial_family_count"] == 2
    assert report["rows_in_exact_nontrivial_families"] == 4
    assert report["maximum_exact_family_size"] == 2
    assert report["cross_split_exact_family_count"] == 1
    assert report["query_rows_with_exact_corpus_copy"] == 1
    assert report["corpus_query_exact_family_disjoint"] is False

    clean = exact_split_duplicate_diagnostics(corpus, queries[1:])
    assert clean["exact_nontrivial_family_count"] == 1
    assert clean["cross_split_exact_family_count"] == 0
    assert clean["corpus_query_exact_family_disjoint"] is True

    collision = np.zeros((2, 64), dtype=np.float16)
    sampled = set(np.linspace(0, 63, 32, dtype=np.int64).tolist())
    unsampled = next(index for index in range(64) if index not in sampled)
    collision[1, unsampled] = 1
    split = exact_split_duplicate_diagnostics(
        collision, np.ones((1, 64), dtype=np.float16)
    )
    assert split["candidate_repeated_groups"] == 1
    assert split["candidate_collision_splits"] == 1
    assert split["exact_nontrivial_family_count"] == 0


def test_projection_metrics_keep_ffr_diagnostic_and_recall_ordered() -> None:
    truth = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    low = np.asarray([list(range(1, 51))])
    metrics = projection_metrics(truth, low, fraction_k=50)
    assert metrics == {
        "ffr_diagnostic": 1.0,
        "recall_at_10": 1.0,
        "recall_at_50_of_high10": 1.0,
    }


def test_headline_ood_gate_uses_polish_not_projection_ffr() -> None:
    cells = {
        language: {
            "recall_at_10": 0.10,
            "recall_at_50_of_high10": 0.40,
            "ffr_diagnostic": -100.0,
        }
        for language in IN_MIX_LANGUAGES
    }
    cells[POLISH] = {
        "recall_at_10": 0.15,
        "recall_at_50_of_high10": 0.21,
        "ffr_diagnostic": -100.0,
    }
    decision = headline_ood_decision(cells)
    assert decision["passed"] is True
    assert decision["polish_to_in_mix_median_ratio"] == pytest.approx(0.525)
    assert decision["projection_ffr_used_for_decision"] is False

    cells[POLISH]["recall_at_50_of_high10"] = 0.15
    assert headline_ood_decision(cells)["passed"] is False


def test_core_gate_requires_each_language_relative_to_pooled_english() -> None:
    group_ffr = {group: 0.50 for group in GROUPS}
    passed = core_geometry_decision(
        density_value=0.20,
        density_floor=0.10,
        global_ffr=0.45,
        group_ffr=group_ffr,
        recall_at_10=0.01,
        recall_at_50=0.02,
        finite_noncollapsed=True,
    )
    assert passed["passed"] is True
    assert passed["projection_ffr_used_for_decision"] is False

    group_ffr[IN_MIX_LANGUAGES[0]] = 0.19
    failed = core_geometry_decision(
        density_value=0.20,
        density_floor=0.10,
        global_ffr=0.45,
        group_ffr=group_ffr,
        recall_at_10=0.01,
        recall_at_50=0.02,
        finite_noncollapsed=True,
    )
    assert failed["passed"] is False


def test_family_size_lookup_defaults_singletons_and_maps_representatives() -> None:
    rows = np.asarray([2, 5, 10, 11], dtype=np.int64)
    representatives = np.asarray([5, 10], dtype=np.int64)
    counts = np.asarray([20, 3], dtype=np.int64)
    assert map_family_sizes(rows, representatives, counts).tolist() == [
        1, 20, 3, 1
    ]
    with pytest.raises(Round0108Error):
        map_family_sizes(
            rows,
            np.asarray([10, 5], dtype=np.int64),
            counts,
        )


def test_map_registry_discovers_explicit_round0108_atlas(
    tmp_path: Path,
) -> None:
    from experiments.map_registry import scan_round0108_atlas

    round_dir = tmp_path / "round-0108"
    artifacts = round_dir / "queue" / "artifacts"
    (artifacts / "semantic-renders").mkdir(parents=True)
    (artifacts / "coordinates" / "chunk-00000").mkdir(parents=True)
    (artifacts / "core-geometry").mkdir()
    (artifacts / "decision").mkdir()
    np.save(
        artifacts / "coordinates" / "chunk-00000" / "coordinates.npy",
        np.zeros((2, 2), dtype=np.float32),
    )
    (round_dir / "queue" / "queue.json").write_text(json.dumps({
        "release_sha": "a" * 40,
    }))
    (artifacts / "semantic-renders" / "map-definition.json").write_text(
        json.dumps({
            "schema": "round0108-map-definition-v1",
            "round_id": "0108",
            "map_key": "r0107-diverse-jina-25m-seed42",
            "map_label": "r0107-diverse-jina-25m-seed42",
        })
    )
    (artifacts / "coordinates" / "actual-transform.json").write_text(
        json.dumps({
            "round_id": "0108",
            "map_key": "r0107-diverse-jina-25m-seed42",
            "model": {"sha256": "b" * 64},
            "row_accounting": {
                "all_rows": 24_948_663,
                "retained_representatives": 24_948_663,
            },
        })
    )
    (artifacts / "core-geometry" / "core-geometry.json").write_text(
        json.dumps({
            "schema": "round0108-diverse-jina-core-geometry-v1",
            "metrics": {
                "global": {"ffr": 0.5},
                "density_v2": {"correlation": 0.2},
            },
            "geometry_diagnostics": {},
        })
    )
    (artifacts / "decision" / "atlas-decision.json").write_text(
        json.dumps({
            "schema": "round0108-diverse-jina-atlas-decision-v1",
            "atlas_quality_capability_released": True,
        })
    )
    maps = scan_round0108_atlas(
        round_dir, {"0108": {"round": {"status": "issued"}}}
    )
    assert len(maps) == 1
    assert maps[0]["dims"] == [768, 2]
    assert maps[0]["panel"]["ffr"] == 0.5
    assert maps[0]["capability_candidate"] is True


def test_registry_view_failure_is_sealed_but_does_not_fail_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import map_registry
    from experiments.round0108_nodes import _refresh_registry_best_effort

    definition = tmp_path / "map-definition.json"
    decision = tmp_path / "atlas-decision.json"
    publication = tmp_path / "registry-publication.json"
    definition.write_text(json.dumps({"artifact": "map definition"}))
    decision.write_text(json.dumps({"artifact": "atlas decision"}))

    def fail_scan() -> dict:
        raise RuntimeError("synthetic mutable-view outage")

    monkeypatch.setattr(map_registry, "scan", fail_scan)
    receipt = _refresh_registry_best_effort(
        receipt_path=str(publication),
        map_definition_path=str(definition),
        decision_path=str(decision),
    )

    assert publication.is_file()
    refresh = receipt["mutable_view_refresh"]
    assert refresh["status"] == "deferred-best-effort-view-refresh"
    assert refresh["requires_followup"] is True
    assert refresh["scientific_decision_affected"] is False
    assert refresh["stages"]["scan"]["error_type"] == "builtins.RuntimeError"
    assert refresh["stages"]["write_mutable_registry"]["status"] == "skipped"
    assert refresh["stages"]["publish_site"]["status"] == "skipped"
    assert refresh["stages"]["site_artifacts"]["status"] == "skipped"
    assert receipt["immutable_artifacts"]["map_definition"]["sha256"]
    assert receipt["immutable_artifacts"]["atlas_decision"]["sha256"]


def test_probe_ids_bind_one_unique_disjoint_id_to_each_embedding() -> None:
    from experiments.round0108_nodes import _validated_probe_ids

    corpus, queries, receipt = _validated_probe_ids(
        np.asarray(["c0", "c1"]),
        np.asarray(["q0"]),
        corpus_rows=2,
        query_rows=1,
    )
    assert corpus.tolist() == ["c0", "c1"]
    assert queries.tolist() == ["q0"]
    assert receipt["corpus"]["rows"] == 2
    assert receipt["queries"]["rows"] == 1
    assert receipt["disjoint"] is True

    invalid = (
        (np.asarray(["c0"]), np.asarray(["q0"]), 2, 1),
        (np.asarray(["c0", "c0"]), np.asarray(["q0"]), 2, 1),
        (np.asarray(["c0", "c1"]), np.asarray(["q0", "q0"]), 2, 2),
        (np.asarray(["c0", "c1"]), np.asarray(["c1"]), 2, 1),
        (np.asarray([["c0", "c1"]]), np.asarray(["q0"]), 2, 1),
    )
    for corpus_ids, query_ids, corpus_rows, query_rows in invalid:
        with pytest.raises(Round0108Error):
            _validated_probe_ids(
                corpus_ids,
                query_ids,
                corpus_rows=corpus_rows,
                query_rows=query_rows,
            )


def test_registry_view_success_requires_expected_round_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import map_registry
    from experiments.round0108_nodes import _refresh_registry_best_effort

    definition = tmp_path / "map-definition.json"
    decision = tmp_path / "atlas-decision.json"
    publication = tmp_path / "registry-publication.json"
    registry_path = tmp_path / "maps.json"
    history_path = tmp_path / "maps-history.json"
    definition.write_text(json.dumps({"artifact": "map definition"}))
    decision.write_text(json.dumps({"artifact": "atlas decision"}))
    registry = {
        "maps": [{
            "round_id": "0108",
            "map_id": (
                "round-0108-r0107-diverse-jina-25m-seed42"
            ),
            "kind": "round-map",
        }] + [
            {
                "round_id": "0108",
                "map_id": f"round-0108-{probe}-projection",
                "kind": "projection-map",
                "projection": {"probe": probe},
            }
            for probe in ("dadabase", "fineweb-heldout", "trec-covid")
        ]
    }

    def write_registry(value: dict) -> Path:
        registry_path.write_text(json.dumps(value))
        history_path.write_text(json.dumps(value))
        return history_path

    published: list[dict] = []

    def publish(value: dict) -> None:
        published.append(value)
        base = tmp_path / "site" / "round-0108"
        base.mkdir(parents=True)
        (base / "index.html").write_text("atlas")
        for item in value["maps"]:
            if item.get("kind") != "projection-map":
                continue
            root = (
                tmp_path / "site" / "projections" / item["map_id"]
            )
            root.mkdir(parents=True)
            (root / "index.html").write_text("projection")
            (root / "manifest.json").write_text("{}")

    monkeypatch.setattr(map_registry, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(map_registry, "SITE_DIR", tmp_path / "site")
    monkeypatch.setattr(map_registry, "scan", lambda: registry)
    monkeypatch.setattr(map_registry, "write_registry", write_registry)
    monkeypatch.setattr(map_registry, "publish", publish)
    receipt = _refresh_registry_best_effort(
        receipt_path=str(publication),
        map_definition_path=str(definition),
        decision_path=str(decision),
    )

    refresh = receipt["mutable_view_refresh"]
    assert refresh["status"] == "published"
    assert refresh["requires_followup"] is False
    assert refresh["stages"]["inventory_validation"]["status"] == "completed"
    assert refresh["stages"]["site_artifacts"]["status"] == "completed"
    assert published == [registry]
