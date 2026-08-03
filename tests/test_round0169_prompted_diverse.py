"""Decision-contract tests for the conditional prompted-diverse Q3 rung."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from basemap.round0105_search import GROUPS
from basemap.panel_v2 import _require_score_panel_scale_admission
from basemap.round0113_prompt_contrast import seal
from basemap.round0166_prompted_8m import METRICS
from basemap.round0169_prompted_diverse import (
    GRAPH_VECTOR_STORAGE,
    MULTIPLICITY_POLICY,
    RETENTION_RATIO,
    ROWS,
    Round0169Error,
    diverse_train_config,
    prompted_diverse_decision,
)
from experiments import round0169_nodes


def _inputs():
    baseline = {metric: 1.0 for metric in METRICS}
    return {
        "native": {metric: 1.0 for metric in METRICS},
        "matched_2m": {metric: RETENTION_RATIO for metric in METRICS},
        "baseline_2m_seed42": baseline,
        "prompted_floors": {metric: 0.9 for metric in METRICS},
        "group_ffr": {name: (0.8 if name in GROUPS[:3] else 0.32) for name in GROUPS},
        "prompted_ood": {
            "polish_recall_at_50_of_high10": 0.10,
            "in_mix_median_recall_at_50_of_high10": 0.20,
        },
        "raw_r0132_ood": {
            "polish_recall_at_50_of_high10": 0.10 / RETENTION_RATIO,
            "in_mix_median_recall_at_50_of_high10": 0.20 / RETENTION_RATIO,
        },
    }


def test_all_registered_boundaries_are_inclusive() -> None:
    decision = prompted_diverse_decision(**_inputs())
    assert decision["passed"] is True
    assert decision["outcome"] == "prompted-diverse-u12-rung-qualified"
    assert decision["language_relative_ffr"]["floor"] == pytest.approx(0.32)
    assert decision["polish_ood_gate"]["ratio"] == pytest.approx(0.5)
    assert all(cell["passed"] for cell in decision["raw_r0132_ood_retention_gates"].values())


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ("native", "density_v2"),
        ("matched_2m", "ffr"),
        ("group_ffr", GROUPS[3]),
        ("prompted_ood", "polish_recall_at_50_of_high10"),
        ("prompted_ood", "in_mix_median_recall_at_50_of_high10"),
    ],
)
def test_each_gate_stack_can_fail_the_decision(path: str, key: str) -> None:
    values = _inputs()
    values[path][key] *= 0.8
    decision = prompted_diverse_decision(**values)
    assert decision["passed"] is False
    assert decision["outcome"] == "prompted-diverse-u12-rung-not-qualified"


def test_metric_or_language_omission_is_invalid_not_a_negative() -> None:
    values = _inputs()
    del values["group_ffr"][GROUPS[-1]]
    with pytest.raises(Round0169Error, match="incomplete"):
        prompted_diverse_decision(**values)

    values = _inputs()
    del values["matched_2m"][METRICS[-1]]
    with pytest.raises(Round0169Error, match="metric set changed"):
        prompted_diverse_decision(**values)


def test_q3_config_changes_only_bound_population_and_registered_storage() -> None:
    signature = {
        "kind": "file",
        "canonical_path": "/data/frozen/edges.npz",
        "bytes": 123,
        "sha256": "a" * 64,
    }
    manifest = {
        "kind": "file",
        "canonical_path": "/data/frozen/graph.json",
        "bytes": 456,
        "sha256": "b" * 64,
    }
    config, digest = diverse_train_config(
        graph_signature=signature,
        graph_manifest_signature=manifest,
        graph_edges=ROWS * 60,
        retained_rows=ROWS,
    )
    assert len(digest) == 64
    assert config["input"]["rows"] == ROWS
    assert config["input"]["multiplicity_policy"] == MULTIPLICITY_POLICY
    assert config["execution"]["graph_vector_storage"] == GRAPH_VECTOR_STORAGE
    stamp = config["execution"]["expected_pipeline_stamp"]
    assert stamp["compact_retained_rows"] == ROWS
    assert stamp["multiplicity_policy"] == MULTIPLICITY_POLICY
    assert stamp["negative_sampling"] == f"uniform-{ROWS}-compact-representatives-nonself"


def test_q3_config_rejects_any_other_population_size() -> None:
    signature = {
        "kind": "file",
        "canonical_path": "/data/frozen/edges.npz",
        "bytes": 123,
        "sha256": "a" * 64,
    }
    with pytest.raises(Round0169Error, match="population"):
        diverse_train_config(
            graph_signature=signature,
            graph_manifest_signature=signature,
            graph_edges=10,
            retained_rows=ROWS - 1,
        )


def test_q3_exact_u12_scale_identity_is_admitted_without_legacy_gate(tmp_path) -> None:
    source_path = tmp_path / "prompted-u12.f16.npy"
    source_path.touch()
    identity = seal({
        "schema": "round0169-prompted-diverse-scale-input-v1",
        "round_id": "0169",
        "row_count": ROWS,
        "dimensions": 768,
        "source": {
            "kind": "file",
            "canonical_path": str(source_path),
            "bytes": ROWS * 768 * 2 + 128,
            "sha256": "a" * 64,
        },
        "mapping": {
            "kind": "file",
            "canonical_path": "/frozen/compact-to-global.i64.npy",
            "bytes": ROWS * 8 + 128,
            "sha256": "b" * 64,
        },
        "staging": {
            "kind": "file",
            "canonical_path": "/frozen/prompted-u12-manifest.json",
            "bytes": 123,
            "sha256": "c" * 64,
        },
        "staging_identity_sha256": "d" * 64,
        "population_law": "exact accepted R0132 U12 compact order",
        "duplicate_policy": (
            "exact duplicate families are diagnostic metadata only; "
            "the accepted U12 population is unchanged"
        ),
    })

    class View:
        round0169_prompted_diverse_view = True
        shape = (ROWS, 768)
        source = SimpleNamespace(filename=str(source_path))

        def __len__(self):
            return ROWS

        def scale_admission_identity(self):
            return dict(identity)

    admitted = _require_score_panel_scale_admission(View(), None)
    assert admitted["identity_sha256"] == identity["identity_sha256"]

    with pytest.raises(RuntimeError, match="carries its own exact identity"):
        _require_score_panel_scale_admission(View(), {"legacy": True})

    changed = dict(identity)
    changed["duplicate_policy"] = "deduplicate"
    changed.pop("identity_sha256")
    changed = seal(changed)

    class Changed(View):
        def scale_admission_identity(self):
            return changed

    with pytest.raises(RuntimeError, match="scale identity is invalid"):
        _require_score_panel_scale_admission(Changed(), None)


def test_q3_evaluation_action_is_reachable(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(
        round0169_nodes,
        "run_evaluate",
        lambda active, job: observed.append((active, job)),
    )
    active = {"manifest": {"round_id": "0169"}}
    job = {"action": "evaluate_prompted_diverse_u12"}
    round0169_nodes.run_job(active, job)
    assert observed == [(active, job)]
