"""Unit tests for the slim v2 family scanner and its compare-group ordering.

R0218 onward score a *family* of maps in one node, publishing a per-seed cell
map instead of the ladder rounds' single ``artifacts/train/``. Three behaviours
kept the whole MiniLM 2M family (and every cuVS-graph sibling) out of the
registry, and each is cheap to regress:

  * queue discovery ignoring ``queue-correction-N``, the naming every one of
    those rounds actually used;
  * cell detection keyed to a schema string, which changes every round;
  * compare ordering choosing the treatment as the drift reference, which
    would measure the control against the thing under test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import gallery_v2  # noqa: E402
import map_registry  # noqa: E402


# ------------------------------------------------------- queue discovery ----

def _queue(round_dir: Path, name: str) -> Path:
    (round_dir / name / "artifacts").mkdir(parents=True)
    return round_dir / name


def test_latest_queue_dir_prefers_highest_correction(tmp_path):
    rd = tmp_path / "round-0223"
    _queue(rd, "queue")
    _queue(rd, "queue-correction-1")
    latest = _queue(rd, "queue-correction-3")
    assert map_registry._latest_queue_dir(rd) == latest


def test_latest_queue_dir_ranks_attempts_and_corrections_alike(tmp_path):
    rd = tmp_path / "round-0100"
    _queue(rd, "queue")
    latest = _queue(rd, "queue-attempt-2")
    assert map_registry._latest_queue_dir(rd) == latest


def test_latest_queue_dir_ignores_abandoned_suffixed_attempts(tmp_path):
    """``queue-attempt-1-unrunnable-metadata`` is a preserved failure."""
    rd = tmp_path / "round-0101"
    canonical = _queue(rd, "queue")
    _queue(rd, "queue-attempt-1-unrunnable-metadata")
    _queue(rd, "queue-preflight-dryrun")
    assert map_registry._latest_queue_dir(rd) == canonical


def test_latest_queue_dir_skips_queues_without_artifacts(tmp_path):
    rd = tmp_path / "round-0102"
    canonical = _queue(rd, "queue")
    (rd / "queue-correction-1").mkdir()
    assert map_registry._latest_queue_dir(rd) == canonical


# ---------------------------------------------------------- cell finding ----

def _cell(seed: int) -> dict:
    return {
        "capability": f"map-seed{seed}-v1",
        "coordinates": {"canonical_path": f"/nope/coordinates-seed{seed}.npy"},
        "panel_metrics": {"ffr": 0.33},
        "seed": seed,
    }


def test_slim_cells_matches_structurally_not_by_schema():
    key, cells = map_registry._slim_cells(
        {"schema": "round9999-never-seen-before", "cells": {"42": _cell(42)}}
    )
    assert (key, sorted(cells)) == ("cells", ["42"])


def test_slim_cells_finds_new_cells_key():
    key, cells = map_registry._slim_cells({"new_cells": {"46": _cell(46)}})
    assert (key, sorted(cells)) == ("new_cells", ["46"])


def test_slim_cells_rejects_bare_metric_maps():
    """R0222 also carries ``pooled_panel_metric_cells``: seeds -> metrics, no
    checkpoint and no coordinates. Admitting those would mint mapless entries."""
    doc = {"cells": {"42": {"ffr": 0.33, "density_v2": 0.44}}}
    assert map_registry._slim_cells(doc) == (None, {})


def test_slim_cells_ignores_non_dict_documents():
    assert map_registry._slim_cells([1, 2, 3]) == (None, {})


# ----------------------------------------------------------- scan output ----

def _write_family_round(tmp_path: Path) -> tuple[Path, Path]:
    """A two-treatment round pair on disk: exact seeds 42-43, cuVS seed 42."""
    runs = tmp_path / "runs"
    for rid, cap_prefix, graph_cap in (
        ("0217", "minilm-2m-map", "minilm-2m-exact-k15-graph-v1"),
        ("0223", "minilm-2m-cuvs-map", "minilm-2m-cuvs-igd48-k15-graph-v1"),
    ):
        seeds = (42, 43) if rid == "0217" else (42,)
        art = runs / f"round-{rid}" / "queue" / "artifacts"
        cells = {}
        for seed in seeds:
            cap = f"{cap_prefix}-seed{seed}-v1"
            map_dir = art / cap
            map_dir.mkdir(parents=True)
            (map_dir / "train-receipt.json").write_text(json.dumps({
                "rows": 2_000_000, "dimension": 384, "release_sha": "abc123",
                "graph_capability": graph_cap, "optimizer_updates": 80_163,
                "exact_execution_receipt": {"graph": {"graph": {"sha256": "g" * 8}}},
            }))
            (map_dir / "production-config.json").write_text(json.dumps({
                "config": {"model": {"architecture": "residual_bottleneck",
                                     "hidden_dimension": 2048,
                                     "output_dimension": 2}}}))
            coords = art / f"panel/coordinates-seed{seed}.npy"
            coords.parent.mkdir(parents=True, exist_ok=True)
            coords.write_bytes(b"")
            cells[str(seed)] = {
                "capability": cap,
                "seed": seed,
                "coordinates": {"canonical_path": str(coords)},
                "model": {"canonical_path": str(map_dir / "model.pt"), "sha256": "m" * 8},
                "train_receipt": {"canonical_path": str(map_dir / "train-receipt.json")},
                "panel_metrics": {"ffr": 0.33, "density_v2": 0.44,
                                  "purity_fidelity_k256": 0.99,
                                  "purity_fidelity_k1024": 0.72},
            }
        (art / "panel" / "panel.json").write_text(
            json.dumps({"schema": f"round{rid}-panel-v1", "cells": cells}))
    return runs / "round-0217", runs / "round-0223"


def test_scan_labels_treatment_from_the_receipt_graph(tmp_path):
    exact_round, cuvs_round = _write_family_round(tmp_path)
    entries = (map_registry.scan_slim_panel_round(exact_round, {})
               + map_registry.scan_slim_panel_round(cuvs_round, {}))
    by_id = {e["map_id"]: e for e in entries}
    assert len(by_id) == 3
    assert by_id["round-0217-minilm-2m-map-seed42-v1"]["graph"]["treatment"] == "exact"
    assert by_id["round-0223-minilm-2m-cuvs-map-seed42-v1"]["graph"]["treatment"] == "cuvs"


def test_scan_binds_a_single_coordinates_file_and_page_per_map(tmp_path):
    exact_round, _ = _write_family_round(tmp_path)
    entries = map_registry.scan_slim_panel_round(exact_round, {})
    pages = {e["page"] for e in entries}
    assert len(pages) == len(entries), "one page per map, not one per round"
    for e in entries:
        assert e["coordinates"]["file"].endswith(f"coordinates-seed{e['seed']}.npy")
        assert e["n_rows"] == 2_000_000
        assert e["panel"]["density_semantics"] == "density-v2"


def test_scan_attributes_a_map_to_the_round_that_trained_it(tmp_path):
    """R0218 scores R0217's family; the maps stay R0217's."""
    runs = tmp_path / "runs"
    art = runs / "round-0218" / "queue" / "artifacts"
    trained = runs / "round-0217" / "queue" / "artifacts" / "map-seed42-v1"
    trained.mkdir(parents=True)
    (trained / "train-receipt.json").write_text(json.dumps(
        {"rows": 2_000_000, "graph_capability": "exact-graph-v1"}))
    coords = art / "panel" / "coordinates-seed42.npy"
    coords.parent.mkdir(parents=True)
    coords.write_bytes(b"")
    (art / "panel" / "panel.json").write_text(json.dumps({"cells": {"42": {
        "capability": "map-seed42-v1", "seed": 42,
        "coordinates": {"canonical_path": str(coords)},
        "model": {"canonical_path": str(trained / "model.pt")},
        "train_receipt": {"canonical_path": str(trained / "train-receipt.json")},
        "panel_metrics": {"ffr": 0.33}}}}))

    entry, = map_registry.scan_slim_panel_round(runs / "round-0218", {})
    assert entry["round_id"] == "0217"
    assert entry["scored_in_round"] == "0218"
    assert entry["map_id"] == "round-0217-map-seed42-v1"


# ------------------------------------------------------- group ordering -----

def _m(map_id, treatment, seed):
    return {"map_id": map_id, "graph": {"treatment": treatment}, "seed": seed,
            "evidence_status": "review:accepted"}


def test_order_group_references_the_majority_treatment():
    group = [_m("c43", "cuvs", 43), _m("e45", "exact", 45),
             _m("c42", "cuvs", 42), _m("e42", "exact", 42), _m("e43", "exact", 43)]
    ordered = gallery_v2._order_group(group)
    assert ordered[0]["map_id"] == "e42", "drift must be measured against the control"
    assert [m["map_id"] for m in ordered] == ["e42", "e43", "e45", "c42", "c43"]


def test_order_group_prefers_the_control_when_treatments_tie():
    """A tie must not hand the reference to the arm under test."""
    group = [_m("c42", "cuvs", 42), _m("e42", "exact", 42)]
    assert gallery_v2._order_group(group)[0]["map_id"] == "e42"


def test_order_group_leaves_untreated_groups_alone():
    group = [{"map_id": "b", "evidence_status": "x"}, {"map_id": "a", "evidence_status": "x"}]
    assert gallery_v2._order_group(group) == group
