from __future__ import annotations

import copy
import json

import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0184_prompted_8m_dose_midpoint import (
    scale_train_config as h2048_train_config,
)
from basemap.round0191_full_width_contrast import (
    HIDDEN_DIMENSION,
    MINIMUM_TRAIN_UPDATES_PER_S,
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    WARNING_TRAIN_UPDATES_PER_S,
    Round0191Error,
    h4096_train_config,
    width_decision,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0191_nodes as nodes
from tests.test_round0166_cpu_smoke import _run_train_seal_reload_panel_cpu_smoke


FULL_GRAPH = (
    "/data/latent-basemap/runs/round-0171/queue/artifacts/"
    "fuzzy-k50-graph-and-reference/graph-manifest.json"
)
R0184_TRAIN = (
    "/data/latent-basemap/runs/round-0184/queue/artifacts/"
    "seed42-1m-update-train/train-receipt.json"
)
R0190_SYNTHESIS = (
    "/data/latent-basemap/runs/round-0190/queue/artifacts/"
    "jina-composition-boundary-three-seed-synthesis-v1/"
    "three-seed-boundary-synthesis.json"
)
METRICS = {
    "mixed_ffr",
    "mixed_purity_fidelity_k256",
    "mixed_purity_fidelity_k1024",
    "pile_ood_recall_at_10",
    "fineweb_ffr",
    "redpajama_ffr",
    "pile_ffr",
}


def _configs() -> tuple[dict, dict]:
    graph = prompt_contract.read_sealed(FULL_GRAPH, label="accepted full graph")
    kwargs = {
        "graph_signature": graph["graph"],
        "graph_manifest_signature": expected_input_signature(FULL_GRAPH),
        "graph_edges": int(graph["directed_edge_count"]),
        "retained_rows": int(graph["retained_rows"]),
    }
    return h2048_train_config(**kwargs)[0], h4096_train_config(**kwargs)[0]


def _changed_paths(left, right, prefix="") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        changed: set[str] = set()
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                changed.add(child)
            else:
                changed |= _changed_paths(left[key], right[key], child)
        return changed
    return {prefix} if left != right else set()


def _metrics(pile_ffr: float) -> dict[str, float]:
    return {
        key: pile_ffr if key == "pile_ffr" else 0.5 + index / 100
        for index, key in enumerate(sorted(METRICS))
    }


def _track_a() -> dict:
    receipt = prompt_contract.read_sealed(R0190_SYNTHESIS, label="R0190 synthesis")
    return copy.deepcopy(receipt["decision"])


def test_h4096_changes_only_registered_width_and_descriptive_fields() -> None:
    h2048, h4096 = _configs()
    assert ROUND_ID == "0191"
    assert SUCCESSFUL_UPDATES == 1_000_000
    assert h2048["model"]["hidden_dimension"] == 2048
    assert h4096["model"]["hidden_dimension"] == HIDDEN_DIMENSION == 4096
    assert h4096["optimizer"] == h2048["optimizer"]
    assert h4096["graph"] == h2048["graph"]
    assert h4096["input"] == h2048["input"]
    assert h4096["execution"]["minimum_train_upd_s"] == MINIMUM_TRAIN_UPDATES_PER_S
    assert h4096["execution"]["warning_train_upd_s"] == WARNING_TRAIN_UPDATES_PER_S
    assert _changed_paths(h2048, h4096) == {
        "schema",
        "model.hidden_dimension",
        "paired_invariant.hidden_dimension",
        "paired_invariant.only_treatment_relative_to_r0184",
        "execution.scale_change",
        "execution.width_contrast_role",
        "execution.minimum_train_upd_s",
        "execution.warning_train_upd_s",
        "dose_registration.role",
    }


def test_accepted_r0184_reference_still_matches_frozen_config_and_accounting() -> None:
    graph = prompt_contract.read_sealed(FULL_GRAPH, label="accepted full graph")
    _config, digest = h2048_train_config(
        graph_signature=graph["graph"],
        graph_manifest_signature=expected_input_signature(FULL_GRAPH),
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=int(graph["retained_rows"]),
    )
    with open(R0184_TRAIN, encoding="utf-8") as handle:
        train = json.load(handle)
    prompt_contract.validate_seal(train, label="accepted R0184 train")
    accounting = train["train_accounting"]
    assert train["production_config_sha256"] == digest
    assert train["optimizer_updates"] == SUCCESSFUL_UPDATES
    assert accounting["positive_lr_optimizer_steps"] == SUCCESSFUL_UPDATES
    assert accounting["pipeline_endpoint_gather_calls"] == SUCCESSFUL_UPDATES
    assert accounting["pipeline_source_rows_gathered"] == (
        SUCCESSFUL_UPDATES * prompt_contract.BATCH_SIZE
    )
    assert all(train["train_checks"].values())


def test_width_selector_distinguishes_recovery_and_seed_noise() -> None:
    track_a = _track_a()
    reference = _metrics(0.4358)
    within_noise = width_decision(
        track_a=track_a,
        h4096_metrics=_metrics(0.4420),
        h2048_metrics=reference,
    )
    assert within_noise["boundary_recovered"] is True
    assert within_noise["within_seed_noise_of_r0184"] is True
    assert within_noise["outcome"] == "boundary-recovered-within-seed-noise"

    material = width_decision(
        track_a=track_a,
        h4096_metrics=_metrics(0.4600),
        h2048_metrics=reference,
    )
    assert material["boundary_recovered"] is True
    assert material["width_effect_detected"] is True
    assert material["outcome"] == "boundary-recovered-with-width-effect"

    null = width_decision(
        track_a=track_a,
        h4096_metrics=_metrics(0.4300),
        h2048_metrics=reference,
    )
    assert null["boundary_recovered"] is False
    assert null["within_seed_noise_of_r0184"] is True
    assert null["outcome"] == "boundary-not-recovered-width-null"


def test_inactive_track_a_and_changed_metric_set_fail_closed() -> None:
    inactive = _track_a()
    inactive["capacity_sibling_activated"] = False
    with pytest.raises(Round0191Error, match="did not activate"):
        width_decision(
            track_a=inactive,
            h4096_metrics=_metrics(0.45),
            h2048_metrics=_metrics(0.44),
        )
    changed = _metrics(0.45)
    changed["density_v2"] = 0.2
    with pytest.raises(Round0191Error, match="metric set"):
        width_decision(
            track_a=_track_a(),
            h4096_metrics=changed,
            h2048_metrics=_metrics(0.44),
        )


def test_q2_configuration_and_train_seal_reload_panel_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    names = (
        "ROUND_ID",
        "CAPABILITY",
        "SEED",
        "SUCCESSFUL_UPDATES",
        "HOST_RSS_LIMIT_GIB",
        "Round0166Error",
        "GRAPH_SCHEMA",
        "TRAIN_SCHEMA",
        "PRODUCTION_CONFIG_SCHEMA",
        "GRAPH_INDEX_DESCRIPTION",
        "GRAPH_REFERENCE_ROW_ORDER",
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE",
        "GRAPH_SOURCE_ROUND_ID",
        "GRAPH_BUILT_IN_ROUND",
        "POPULATION_READER",
        "MIN_SCALE_ROWS_EXCLUSIVE",
        "scale_train_config",
    )
    for name in names:
        monkeypatch.setattr(q2, name, getattr(q2, name))
    nodes._configure_q2("full", {"graph_manifest": FULL_GRAPH})
    assert q2.ROUND_ID == ROUND_ID
    assert q2.SEED == 42
    assert q2.SUCCESSFUL_UPDATES == SUCCESSFUL_UPDATES
    assert q2.GRAPH_SOURCE_ROUND_ID == "0171"
    assert q2.GRAPH_BUILT_IN_ROUND is False
    config, _digest = q2.scale_train_config(
        graph_signature={"kind": "file", "canonical_path": "/g", "bytes": 1, "sha256": "a" * 64},
        graph_manifest_signature={
            "kind": "file",
            "canonical_path": "/m",
            "bytes": 1,
            "sha256": "b" * 64,
        },
        graph_edges=603_086_368,
        retained_rows=7_952_419,
    )
    assert config["model"]["hidden_dimension"] == 4096
    _run_train_seal_reload_panel_cpu_smoke(
        monkeypatch,
        tmp_path,
        config_graph_edges=603_086_368,
        expected_seed=42,
    )


def test_unknown_action_fails_before_execution() -> None:
    with pytest.raises(Round0191Error, match="does not authorize"):
        nodes.run_job({"manifest": {"round_id": ROUND_ID}}, {"action": "ladder"})
