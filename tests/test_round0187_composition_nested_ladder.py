from __future__ import annotations

import numpy as np
import pytest

from basemap.round0187_composition_nested_ladder import (
    FULL_GRAPH_EDGES,
    FULL_SUCCESSFUL_UPDATES,
    PRIMARY_METRICS,
    RETENTION_RATIO,
    RUNG_ROWS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    _select_nested_positions_for_spec,
    canonical_id_digest,
    ladder_decision,
    successful_updates_for_edges,
    train_checks_close,
    train_config,
)
from experiments import round0166_nodes as q2


def _metrics(value: float) -> dict[str, float]:
    return {metric: value for metric in PRIMARY_METRICS}


def test_canonical_id_hash_is_frozen_and_namespaced() -> None:
    assert canonical_id_digest("fineweb", 17).hex() == (
        "35b9ee79764b1a65c12678c52923e1509ac0e0c17408095590fde4662ae5e178"
    )
    assert canonical_id_digest("fineweb", 17) != canonical_id_digest(
        "redpajama", 17
    )


def test_tiny_composition_selection_is_nested_and_canonical() -> None:
    mapping = np.arange(24, dtype=np.int64)
    corpora = (
        ("fineweb", 0, 8),
        ("redpajama", 8, 16),
        ("pile", 16, 24),
    )
    full = {corpus: 8 for corpus, _start, _stop in corpora}
    counts = {
        "quarter": {corpus: 2 for corpus in full},
        "half": {corpus: 4 for corpus in full},
    }
    selected = _select_nested_positions_for_spec(
        mapping, corpora=corpora, full_counts=full, rung_counts=counts
    )
    assert len(selected["quarter"]) == 6
    assert len(selected["half"]) == 12
    assert np.all(selected["quarter"][1:] > selected["quarter"][:-1])
    assert np.all(selected["half"][1:] > selected["half"][:-1])
    assert set(selected["quarter"]).issubset(set(selected["half"]))
    for rung, per_corpus in counts.items():
        for corpus, start, stop in corpora:
            assert np.count_nonzero(
                (selected[rung] >= start) & (selected[rung] < stop)
            ) == per_corpus[corpus]


def test_dose_horizon_uses_exact_r0180_rational() -> None:
    assert successful_updates_for_edges(FULL_GRAPH_EDGES) == FULL_SUCCESSFUL_UPDATES
    for edges in (150_000_001, 301_000_003):
        updates = successful_updates_for_edges(edges)
        achieved = updates * 409 / edges
        assert achieved >= TARGET_POSITIVE_DRAWS_PER_EDGE
        assert achieved - TARGET_POSITIVE_DRAWS_PER_EDGE < 409 / edges


def test_train_config_freezes_h2048_and_explicit_dose() -> None:
    signature = {
        "canonical_path": "/tmp/graph",
        "kind": "file",
        "bytes": 1,
        "sha256": "0" * 64,
    }
    config, digest = train_config(
        rung="quarter",
        graph_signature=signature,
        graph_manifest_signature={**signature, "canonical_path": "/tmp/manifest"},
        graph_edges=150_000_001,
        retained_rows=RUNG_ROWS["quarter"],
    )
    assert len(digest) == 64
    assert config["model"]["hidden_dimension"] == 2048
    assert config["optimizer"]["seed"] == 42
    assert config["execution"]["target_positive_draws_per_edge"] == pytest.approx(
        TARGET_POSITIVE_DRAWS_PER_EDGE, abs=0
    )
    assert config["input"]["composition"] == {
        "fineweb": 719_108,
        "redpajama": 702_656,
        "pile": 566_340,
    }


def test_train_checks_cannot_pass_empty_or_extra() -> None:
    checks = {
        "exact_update_closure": True,
        "zero_numerical_skips": True,
        "no_pipeline_stamp_drift": True,
        "endpoint_rows_match_updates": True,
        "weighted_rejection_accounting_closes": True,
    }
    assert train_checks_close(checks)
    assert q2._train_checks_close(checks)
    assert not train_checks_close({})
    assert not q2._train_checks_close({})
    assert not train_checks_close({**checks, "unexpected": True})
    assert not q2._train_checks_close({**checks, "unexpected": True})


def test_ladder_decision_branches_are_preregistered() -> None:
    retained = ladder_decision({
        "quarter": _metrics(1.0),
        "half": _metrics(RETENTION_RATIO),
        "full": _metrics(RETENTION_RATIO**2),
    })
    assert retained["outcome"] == "composition-controlled-scale-retained"

    controlled = ladder_decision({
        "quarter": _metrics(1.0),
        "half": _metrics(0.96),
        "full": _metrics(0.90),
    })
    assert controlled["outcome"] == "composition-controlled-size-regression"
    assert controlled["capacity_activated"] is True

    discordant = ladder_decision({
        "quarter": _metrics(1.0),
        "half": _metrics(0.96),
        "full": _metrics(0.98),
    })
    assert discordant["outcome"] == "composition-controlled-boundary-or-discordant"
