from __future__ import annotations

import pytest

from basemap.round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST,
    METRICS,
    RESTORATION_FLOORS,
)
from basemap.round0149_drop_only import CAPABILITY as R0149_CAPABILITY
from basemap.round0149_drop_only import TREATMENT as DROP_ONLY
from basemap.round0150_seed_replay import (
    SEED,
    Round0150Error,
    build_decision,
    drop_seed43_train_config,
    raw_seed43_train_config,
)
from experiments.round0150_nodes import run_job


RAW = CURRENT_GRAPH_CURRENT_HOST
ELIGIBLE = "eligible_historical_current_graph_current_host"


def signature(digit: str) -> dict[str, object]:
    return {
        "canonical_path": f"/tmp/{digit}.npz",
        "kind": "file",
        "bytes": 1,
        "sha256": digit * 64,
    }


def values(*, passed: bool) -> dict[str, float]:
    if passed:
        return {key: float(value) + 0.01 for key, value in RESTORATION_FLOORS.items()}
    result = {key: float(value) + 0.01 for key, value in RESTORATION_FLOORS.items()}
    result["ffr"] = RESTORATION_FLOORS["ffr"] - 0.001
    return result


def cell(*, passed: bool) -> dict[str, object]:
    metric = values(passed=passed)
    return {
        "seed": SEED,
        "panel": {
            "ffr": metric["ffr"],
            "purity": {
                "k256": metric["purity_fidelity_k256"],
                "k1024": metric["purity_fidelity_k1024"],
            },
        },
        "projection": {
            "ffr": metric["projection_ffr"],
            "recall_at_10": metric["ood_recall_at_10"],
        },
    }


def floor_test(metric: dict[str, float]) -> dict[str, object]:
    rows = {
        key: {
            "observed": metric[key],
            "floor": RESTORATION_FLOORS[key],
            "passed": metric[key] >= RESTORATION_FLOORS[key],
        }
        for key in METRICS
    }
    return {"metrics": rows, "passed_all": all(row["passed"] for row in rows.values())}


def r0149(*, drop_passed: bool) -> dict[str, object]:
    raw = values(passed=True)
    eligible = values(passed=False)
    drop = values(passed=drop_passed)
    return {
        "round_id": "0149",
        "capability": R0149_CAPABILITY,
        "outcome": (
            "drop-only-historical-row-policy-restores"
            if drop_passed
            else "drop-only-historical-row-policy-does-not-restore"
        ),
        "metrics": {RAW: raw, ELIGIBLE: eligible, DROP_ONLY: drop},
        "restoration": {
            RAW: floor_test(raw),
            ELIGIBLE: floor_test(eligible),
            DROP_ONLY: floor_test(drop),
        },
    }


def test_configs_retag_only_the_registered_seed() -> None:
    raw, raw_digest = raw_seed43_train_config(
        graph_signature=signature("1"),
        graph_manifest_signature=signature("2"),
        graph_edges=123,
    )
    drop, drop_digest = drop_seed43_train_config(
        graph_signature=signature("3"),
        graph_manifest_signature=signature("4"),
        graph_edges=456,
        source_sha256="5" * 64,
        selection_sha256="6" * 64,
    )
    for config, digest in ((raw, raw_digest), (drop, drop_digest)):
        assert config["paired_invariant"]["seed"] == SEED
        assert config["optimizer"]["seed"] == SEED
        assert config["causal_matrix"]["replication_seed"] == SEED
        assert config["causal_matrix"]["graph_reused_byte_exact"] is True
        assert len(digest) == 64
    assert drop["paired_invariant"]["rows"] == 1_989_633
    assert drop["execution"]["expected_pipeline_stamp"]["negative_sampling"] == (
        "uniform-1989633-row-universe-nonself"
    )


def test_selector_releases_scale_candidate_only_on_two_seed_replication() -> None:
    decision = build_decision(
        r0149(drop_passed=True),
        {RAW: cell(passed=True), DROP_ONLY: cell(passed=True)},
    )
    assert decision["outcome"] == "drop-only-restoration-replicates-across-seeds"
    assert decision["drop_only_scale_candidate_released"] is True


def test_selector_closes_replicated_failure() -> None:
    decision = build_decision(
        r0149(drop_passed=False),
        {RAW: cell(passed=True), DROP_ONLY: cell(passed=False)},
    )
    assert decision["outcome"] == "drop-only-restoration-fails-across-seeds"
    assert decision["drop_only_scale_candidate_released"] is False


@pytest.mark.parametrize(
    ("seed42_drop", "raw43", "drop43"),
    [(False, True, True), (True, True, False), (True, False, True)],
)
def test_selector_keeps_discordant_or_bad_control_replay_inconclusive(
    seed42_drop: bool, raw43: bool, drop43: bool
) -> None:
    decision = build_decision(
        r0149(drop_passed=seed42_drop),
        {RAW: cell(passed=raw43), DROP_ONLY: cell(passed=drop43)},
    )
    assert decision["outcome"] == (
        "drop-only-restoration-is-seed-sensitive-or-control-inconclusive"
    )
    assert decision["drop_only_scale_candidate_released"] is False


def test_selector_rejects_inconsistent_parent_outcome() -> None:
    parent = r0149(drop_passed=False)
    parent["outcome"] = "drop-only-historical-row-policy-restores"
    with pytest.raises(Round0150Error, match="outcome"):
        build_decision(parent, {RAW: cell(passed=True), DROP_ONLY: cell(passed=True)})


def test_handler_rejects_wrong_round_manifest_before_dispatch() -> None:
    with pytest.raises(Round0150Error, match="exact queue manifest"):
        run_job({"manifest": {"round_id": "0149"}}, {"action": "unknown"})
