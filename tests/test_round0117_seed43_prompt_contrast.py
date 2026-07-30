from __future__ import annotations

import json
from pathlib import Path

import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0113_prompt_contrast import (
    NEGATIVE_RNG_SEED_OFFSET,
    RETAINED_ROWS,
    Round0113Error,
    train_config,
)
from experiments.round0113_nodes import (
    _graph_execution_round_id,
    _training_seed,
)
from experiments.prepare_round0117_queue import (
    R0115_DECISION_SHA256,
    R0115_GRAPH_SHA256,
    R0115_QUERY_SELECTION_SHA256,
    TRAINING_SEED,
    prepare_round0117,
)
import experiments.prepare_round0117_queue as queue_prep


def _signature(path: str, byte: str) -> dict[str, object]:
    return {
        "canonical_path": path,
        "kind": "file",
        "bytes": 1,
        "sha256": byte * 64,
    }


def _changed_paths(left, right, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right)
        changed: set[str] = set()
        for key in left:
            child = f"{prefix}.{key}" if prefix else key
            changed.update(_changed_paths(left[key], right[key], child))
        return changed
    return {prefix} if left != right else set()


def test_seed43_changes_only_registered_rng_fields_in_training_config():
    kwargs = {
        "arm": "raw",
        "graph_signature": _signature("/data/raw-graph.npz", "a"),
        "graph_manifest_signature": _signature(
            "/data/raw-graph.json", "b"
        ),
        "graph_edges": 123,
        "retained_rows": RETAINED_ROWS,
    }
    seed42, digest42 = train_config(**kwargs, seed=42)
    seed43, digest43 = train_config(**kwargs, seed=TRAINING_SEED)
    assert _changed_paths(seed42, seed43) == {
        "paired_invariant.seed",
        "optimizer.seed",
        "optimizer.positive_rng_seed",
        "optimizer.negative_rng_seed",
        "execution.expected_pipeline_stamp.positive_rng_seed",
        "execution.expected_pipeline_stamp.negative_rng_seed",
    }
    assert seed43["optimizer"]["negative_rng_seed"] == (
        TRAINING_SEED + NEGATIVE_RNG_SEED_OFFSET
    )
    assert digest42 != digest43


def test_seed43_registry_and_reused_graph_provenance_fail_closed():
    active = {"manifest": {"round_id": "0117"}}
    assert _training_seed(active, {"training_seed": 43}) == 43
    assert (
        _graph_execution_round_id(
            active, {"graph_execution_round_id": "0115"}
        )
        == "0115"
    )
    with pytest.raises(Round0113Error, match="training seed"):
        _training_seed(active, {"training_seed": 42})
    with pytest.raises(Round0113Error, match="graph provenance"):
        _graph_execution_round_id(
            active, {"graph_execution_round_id": "0117"}
        )


def test_existing_seed42_rounds_retain_their_original_registry():
    for round_id in ("0113", "0115"):
        active = {"manifest": {"round_id": round_id}}
        assert _training_seed(active, {}) == 42
        assert _graph_execution_round_id(active, {}) == round_id


def test_r0117_binds_exact_accepted_seed42_artifacts():
    assert R0115_DECISION_SHA256 == (
        "65172c2b445391d3e30799e833479d359196f2593d846369fdb0fc76f6e5b24c"
    )
    assert R0115_GRAPH_SHA256 == {
        "raw": (
            "b39a705bf5f426777c33c5941607738a4e0070969f8892234ef42b94b077973c"
        ),
        "document": (
            "3c617463f308ae756c3256cea83180af70a5d2492cb7c09617f0ee6881917912"
        ),
    }
    assert R0115_QUERY_SELECTION_SHA256 == (
        "7b8eaaa82f2ae1484510f8cb4422d4169c8dc0390387409ea01fc3d13292dbf8"
    )


def test_queue_reuses_graphs_and_queries_and_schedules_only_seed43_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    round_file = tmp_path / "round-0117-test.md"
    round_file.write_text(
        "---\nround_id: \"0117\"\nstatus: issued\n---\n", encoding="utf-8"
    )
    round_root = tmp_path / "run"
    queue_root = round_root / "queue"
    monkeypatch.setattr(queue_prep, "ROUND_FILE_GLOB", str(round_file))
    monkeypatch.setattr(queue_prep, "ROUND_ROOT", str(round_root))
    monkeypatch.setattr(
        queue_prep,
        "ensure_data_directory",
        lambda path: str(Path(path).mkdir(parents=True, exist_ok=True) or path),
    )
    monkeypatch.setattr(
        queue_prep,
        "create_fresh_directory",
        lambda path, **kwargs: str(Path(path).mkdir(parents=True) or path),
    )
    monkeypatch.setattr(
        queue_prep,
        "atomic_write_new_json",
        lambda path, payload, **kwargs: Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        ),
    )
    review = Path(
        "/home/enjalot/code/latent-labs/"
        "basemap-100m/review-0115-2026-07-30.md"
    )
    queue_path = prepare_round0117(
        release_sha="a" * 40,
        r0115_review=str(review),
        r0115_review_sha256=expected_input_signature(str(review))["sha256"],
        queue_root=str(queue_root),
    )
    with open(queue_path, encoding="utf-8") as handle:
        queue = json.load(handle)
    jobs = {job["id"]: job for job in queue["jobs"]}
    assert set(jobs) == {
        "train_raw_map_seed43",
        "train_document_map_seed43",
        "evaluate_raw_map_seed43",
        "evaluate_document_map_seed43",
        "decide_seed43_prompt_contrast",
    }
    assert queue["scientific_contract"]["training"]["seed"] == 43
    assert queue["reuse"]["rebuild_graphs"] is False
    assert queue["reuse"]["reselect_queries"] is False
    for arm in ("raw", "document"):
        train = jobs[f"train_{arm}_map_seed43"]
        evaluate = jobs[f"evaluate_{arm}_map_seed43"]
        assert train["training_seed"] == evaluate["training_seed"] == 43
        assert (
            train["graph_execution_round_id"]
            == evaluate["graph_execution_round_id"]
            == "0115"
        )
        assert (
            train["graph_manifest"] == evaluate["graph_manifest"]
            == (
                "/data/latent-basemap/runs/round-0115/queue-attempt-2/"
                f"artifacts/{arm}/graph/graph-manifest.json"
            )
        )
