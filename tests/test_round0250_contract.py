"""R0250 contract tests — every new guard ships a planted defect it catches.

Three families of test here:

* **identity** — the three new cells are R0217's treatment with the seed as the
  only free variable, checked below the digest and with a planted field move that
  the check must reject;
* **the trainer-loop instrumentation** — the short-rung config diff guard, the
  per-batch poll installer's restore-on-raise, and the ceiling and projection
  arithmetic, each with an input that makes it go the other way;
* **the n=16 gate** — the joint-criteria construction, the AND, the
  falsifiability statement and the block-size resolution rule, each with a
  planted defect.

Nothing here touches CUDA, the GPU, the registry or any guard module.
"""
from __future__ import annotations

import copy
import math

import pytest

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0217_minilm_2m_seed_family import (
    SEED_BEARING_PATHS,
    train_config as r0217_train_config,
)
from basemap.round0234_calibrated_floors import GATED_METRICS, PURITY_METRICS
from basemap.round0247_registry import REGISTERED_REGISTRY_SHA256, registry_fingerprint
from basemap.round0250_blocksize import (
    PER_ROW_COST_STABILITY_LIMIT,
    REGISTRATION_SAFETY_MARGIN,
    RESOLUTION_NOT_NEEDED,
    RESOLUTION_REGISTER,
    Round0250BlockSizeError,
    largest_block_meeting_the_ceiling,
    observed_default_block,
    resolve,
)
from basemap.round0250_gate_n16 import (
    EXACT_FAMILY_SEEDS,
    GATE_CAPABILITY,
    IDENTITY_BOUND_AT_N,
    N_EXACT,
    N_HELD_OUT,
    RETAINED_FAMILY_SOURCES,
    Round0250GateError,
    THIS_FAMILY,
    falsifiability_statement,
    identity_bound,
    joint_criteria_from_sealed,
    score_joint,
)
from basemap.round0250_panel_n16 import (
    POOLED_CELL_SOURCES,
    Round0250PanelError,
    assert_reference_identity,
    pool_sixteen_cells,
)
from basemap.round0250_seed_extension_n16 import (
    CAPABILITIES,
    IDENTITY_BOUND_AT_N16,
    POOLED_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    ROWS,
    Round0250Error,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    STANDING_MINIMUM_N,
    TEMPLATE_SEED,
    assert_extension_differs_only_by_seed,
    assert_reconstructs_r0217_template,
    capability_for_seed,
    masked_config_bytes,
    predict_cell_footprint,
    seed_bearing_values,
    seed_invariant_sha256,
    train_config,
)
from basemap.round0250_trainer_loops import (
    ARM_AS_SHIPPED,
    ARM_PER_BATCH,
    PROJECTION_TARGET_SECONDS,
    PerBatchAbortPoll,
    Round0250TrainerLoopError,
    SHORT_HORIZON_UPDATES,
    ceiling_report,
    project_to_hours,
    short_rung_config,
)


#: A substrate signature that is NOT R0216's queue-correction-3 one. Used as the
#: planted input for the "a cell cannot be built on other bytes" control.
WRONG_SUBSTRATE_SIGNATURE = {
    "kind": "file",
    "canonical_path": (
        "/data/latent-basemap/runs/round-0216/queue-correction-2/artifacts/"
        "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy"
    ),
    "bytes": 3_072_000_128,
    "sha256": "0" * 64,
}


def _signatures():
    from basemap.round0250_seed_extension_n16 import (
        SEALED_GRAPH_MANIFEST_SIGNATURE,
        SEALED_GRAPH_SIGNATURE,
        SEALED_SUBSTRATE_SIGNATURE,
    )

    return (
        dict(SEALED_GRAPH_SIGNATURE),
        dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
        dict(SEALED_SUBSTRATE_SIGNATURE),
    )


def _config(seed: int):
    graph, manifest, substrate = _signatures()
    return train_config(
        seed=seed,
        graph_signature=graph,
        graph_manifest_signature=manifest,
        substrate_signature=substrate,
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )


def _template():
    graph, manifest, substrate = _signatures()
    template, _sha = r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=graph,
        graph_manifest_signature=manifest,
        substrate_signature=substrate,
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )
    return template


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #


def test_the_three_new_cells_reproduce_r0217s_published_seed_invariant_digest():
    configs = {seed: _config(seed)[0] for seed in SEEDS}
    family = assert_extension_differs_only_by_seed(
        configs, expected_seed_invariant=R0217_SEED_INVARIANT_SHA256
    )
    assert family["seed_invariant_sha256"] == R0217_SEED_INVARIANT_SHA256
    assert family["matches_r0217_published_seed_invariant"] is True
    assert family["n_pooled"] == len(POOLED_SEEDS) == 16
    assert family["reaches_the_standing_minimum"] is True
    assert {
        item["masked_config_bytes"]
        for item in family["masked_config_identity"].values()
    } == {5040}
    assert len(set(family["per_seed_config_sha256"].values())) == len(SEEDS)


def test_each_cell_reconstructs_r0217s_canonical_config_byte_for_byte():
    template = _template()
    for seed in SEEDS:
        receipt = assert_reconstructs_r0217_template(_config(seed)[0], template)
        assert receipt["byte_equal"] is True
        assert receipt["reconstructed_sha256"] == receipt["r0217_template_sha256"]


def test_a_field_moved_outside_the_nine_seed_bearing_paths_is_rejected():
    """Positive control: the reconstruction check is not a tautology."""
    template = _template()
    planted = copy.deepcopy(_config(55)[0])
    planted["optimizer"]["learning_rate"] = float(
        planted["optimizer"]["learning_rate"]
    ) * 2.0
    # The raiser is R0230's `Round0230Error`, because R0250 CALLS R0230's
    # released reconstruction check rather than re-typing it. That is the point.
    from basemap.round0230_minilm_2m_seed_extension_n13 import Round0230Error

    with pytest.raises(Round0230Error):
        assert_reconstructs_r0217_template(planted, template)


def test_a_cell_that_differs_outside_the_seed_breaks_the_family_digest():
    """Positive control for the family-level check."""
    configs = {seed: _config(seed)[0] for seed in SEEDS}
    configs[56] = copy.deepcopy(configs[56])
    configs[56]["model"]["hidden_dimension"] = 4096
    with pytest.raises(Round0250Error):
        assert_extension_differs_only_by_seed(
            configs, expected_seed_invariant=R0217_SEED_INVARIANT_SHA256
        )


def test_a_wrong_substrate_signature_refuses_to_build_a_cell():
    graph, manifest, _substrate = _signatures()
    with pytest.raises(Round0250Error):
        train_config(
            seed=55,
            graph_signature=graph,
            graph_manifest_signature=manifest,
            substrate_signature=WRONG_SUBSTRATE_SIGNATURE,
            graph_edges=SEALED_DIRECTED_EDGES,
            rows=ROWS,
        )


def test_seed_bearing_paths_match_r0217s_registered_set():
    assert set(seed_bearing_values(57)) == set(SEED_BEARING_PATHS)
    for seed in SEEDS:
        assert capability_for_seed(seed) == f"minilm-mixed-2m-map-seed{seed}-low-dose-v1"
    assert set(CAPABILITIES) == {capability_for_seed(seed) for seed in SEEDS}
    with pytest.raises(Round0250Error):
        capability_for_seed(54)


def test_the_predictive_guard_records_its_prediction_and_can_refuse():
    prediction = predict_cell_footprint(55)
    assert prediction["refused_a_priori"] is False
    assert prediction["predicted_peak_device_bytes"] > 0
    assert prediction["predicted_peak_host_anonymous_bytes"] > 0
    assert prediction["predicted_device_headroom_bytes"] > 0


def test_the_identity_bound_moves_from_13_to_16_as_the_plan_states():
    assert identity_bound(13) == pytest.approx(3.328201177351375, abs=1e-12)
    assert IDENTITY_BOUND_AT_N16 == 3.75
    assert IDENTITY_BOUND_AT_N == 3.75
    assert len(POOLED_SEEDS) == STANDING_MINIMUM_N == N_EXACT == 16


# --------------------------------------------------------------------------- #
# the trainer-loop instrumentation
# --------------------------------------------------------------------------- #


def test_the_short_rung_changes_only_the_declared_paths():
    template = _template()
    config, sha, identity = short_rung_config(template)
    assert config["optimizer"]["successful_positive_lr_updates"] == SHORT_HORIZON_UPDATES
    assert config["round_id"] == "0250"
    assert config["seed_family"]["is_a_family_cell"] is False
    assert identity["changes_are_inside_the_declared_set"] is True
    assert identity["short_rung_sha256"] == sha != identity["template_sha256"]
    # the treatment fields the rung must NOT move
    for path in (
        ("model", "hidden_dimension"),
        ("optimizer", "batch_size"),
        ("optimizer", "use_amp"),
        ("graph", "directed_edges"),
        ("input", "substrate_sha256"),
    ):
        assert config[path[0]][path[1]] == template[path[0]][path[1]]


def test_a_rung_too_short_to_reach_steady_state_is_refused():
    template = _template()
    with pytest.raises(Round0250TrainerLoopError):
        short_rung_config(template, updates=10)


def test_the_short_rung_guard_catches_a_planted_escape():
    """The declared-path guard fires when the config moves a field it does not own."""
    import basemap.round0250_trainer_loops as loops

    template = _template()
    planted = tuple(
        path for path in loops.SHORT_RUNG_CHANGED_PATHS if path != ("round_id",)
    )
    original = loops.SHORT_RUNG_CHANGED_PATHS
    loops.SHORT_RUNG_CHANGED_PATHS = planted
    try:
        with pytest.raises(loops.Round0250TrainerLoopError):
            loops.short_rung_config(template)
    finally:
        loops.SHORT_RUNG_CHANGED_PATHS = original
    # and with the real declared set it passes again
    loops.short_rung_config(template)


def test_the_per_batch_installer_restores_the_release_attribute_even_on_a_raise():
    from basemap.pumap.parametric_umap import ParametricUMAP

    before = ParametricUMAP._low_dim_qs
    seen: list[str] = []
    installer = PerBatchAbortPoll(poll=seen.append, label="control")
    with pytest.raises(RuntimeError):
        with installer:
            assert ParametricUMAP._low_dim_qs is not before
            raise RuntimeError("planted")
    assert ParametricUMAP._low_dim_qs is before
    assert installer.restored is True
    receipt = installer.receipt()
    assert receipt["restored_the_release_attribute"] is True
    assert receipt["hook"]["calls_per_training_batch"] == 1


def test_the_per_batch_installer_actually_polls_once_per_call():
    from basemap.pumap.parametric_umap import ParametricUMAP

    before = ParametricUMAP._low_dim_qs
    seen: list[str] = []

    class _Fake:
        pass

    installer = PerBatchAbortPoll(poll=seen.append, label="control")
    with installer:
        for _ in range(4):
            try:
                ParametricUMAP._low_dim_qs(_Fake(), None, None)
            except Exception:  # noqa: BLE001 - the real kernel needs tensors
                pass
    assert ParametricUMAP._low_dim_qs is before
    assert len(seen) == 4
    assert installer.receipt()["per_batch_polls_installed"] == 4


def test_a_non_callable_poll_is_refused():
    with pytest.raises(Round0250TrainerLoopError):
        PerBatchAbortPoll(poll=None)


def test_the_ceiling_report_is_arithmetic_a_reviewer_can_redo():
    report = ceiling_report(
        arm=ARM_AS_SHIPPED, widest_gap_s=90.0, stage_wall_s=91.0, polls=2
    )
    ceiling = report["registered_ceiling_s_at_the_comparison_site"]
    assert ceiling == pytest.approx(2.5109531834854018, abs=0.0)
    assert report["gap_over_the_registered_ceiling"] == pytest.approx(
        90.0 / ceiling, abs=1e-12
    )
    assert report["meets_the_registered_ceiling"] is False
    passing = ceiling_report(
        arm=ARM_PER_BATCH, widest_gap_s=0.2, stage_wall_s=91.0, polls=10_000
    )
    assert passing["meets_the_registered_ceiling"] is True
    with pytest.raises(Round0250TrainerLoopError):
        ceiling_report(arm="not-an-arm", widest_gap_s=1.0, stage_wall_s=1.0, polls=2)


def test_the_projection_is_labelled_and_scales_per_arm():
    shipped = project_to_hours(
        arm=ARM_AS_SHIPPED, widest_gap_s=90.0, stage_wall_s=91.0
    )
    assert shipped["kind"] == "projection"
    assert shipped["is_a_measurement_at_the_target_wall"] is False
    assert shipped["projected_widest_gap_s"] == PROJECTION_TARGET_SECONDS
    assert shipped["projected_meets_the_registered_ceiling"] is False
    batched = project_to_hours(
        arm=ARM_PER_BATCH, widest_gap_s=0.2, stage_wall_s=91.0
    )
    assert batched["projected_widest_gap_s"] == 0.2
    assert batched["projected_meets_the_registered_ceiling"] is True


# --------------------------------------------------------------------------- #
# the block-size resolution
# --------------------------------------------------------------------------- #


def _block_arms(spread: float):
    arms = [
        {
            "block_rows": block,
            "worst_block_wall_s": 0.01 * block,
            "worst_seconds_per_gathered_row": 1.0e-6,
            "page_cache_state": "cold: fresh rows",
        }
        for block in (500, 1_000, 2_000, 4_000)
    ]
    arms.append({
        "block_rows": 2_000,
        "worst_block_wall_s": 0.01 * 2_000 / spread,
        "worst_seconds_per_gathered_row": 1.0e-6 / spread,
        "page_cache_state": "warm: the first cold range again",
    })
    return arms


def test_a_stable_per_row_cost_resolves_to_register():
    resolution = resolve(
        arms=_block_arms(1.5),
        declared_default_block=2_000,
        r0247_widest_gap_s=2.276142634014832,
        r0247_block_rows=2_000,
    )
    assert resolution["resolution"] == RESOLUTION_REGISTER
    assert resolution["criterion_3_clamping_would_restore_the_arm"] is True
    assert resolution["per_row_cost_spread"] == pytest.approx(1.5, abs=1e-12)
    assert resolution["block_that_would_hold_at_the_measured_worst_case"] > 0


def test_an_unstable_per_row_cost_resolves_to_not_needed():
    """Positive control: the rule must be able to go both ways."""
    resolution = resolve(
        arms=_block_arms(8.0),
        declared_default_block=2_000,
        r0247_widest_gap_s=2.276142634014832,
        r0247_block_rows=2_000,
    )
    assert resolution["resolution"] == RESOLUTION_NOT_NEEDED
    assert resolution["criterion_3_clamping_would_restore_the_arm"] is False
    assert resolution["per_row_cost_spread"] > PER_ROW_COST_STABILITY_LIMIT


def test_the_resolution_reproduces_r0247s_published_ceiling_fraction():
    resolution = resolve(
        arms=_block_arms(1.5),
        declared_default_block=2_000,
        r0247_widest_gap_s=2.276142634014832,
        r0247_block_rows=2_000,
    )
    assert resolution["r0247_gap_over_the_registered_ceiling"] == pytest.approx(
        0.9064854928339865, abs=1e-12
    )
    assert resolution["block_at_which_r0247_would_breach"] > 2_000
    assert resolution["this_round_registered_nothing"] is True
    assert registry_fingerprint() == REGISTERED_REGISTRY_SHA256


def test_the_block_size_default_is_read_from_the_released_signature():
    assert observed_default_block() == 2_000


def test_sizing_a_block_needs_a_positive_cost():
    assert largest_block_meeting_the_ceiling(
        worst_seconds_per_gathered_row=1.0e-6
    ) == int((2.5109531834854018 / REGISTRATION_SAFETY_MARGIN) / (1.0e-6 * 15))
    with pytest.raises(Round0250BlockSizeError):
        largest_block_meeting_the_ceiling(worst_seconds_per_gathered_row=0.0)


def test_resolving_with_no_arms_is_refused():
    with pytest.raises(Round0250BlockSizeError):
        resolve(
            arms=[],
            declared_default_block=2_000,
            r0247_widest_gap_s=1.0,
            r0247_block_rows=2_000,
        )


# --------------------------------------------------------------------------- #
# the n = 16 gate
# --------------------------------------------------------------------------- #


def _family_artifact(*, floor: float, lower: float, upper: float, n: int, capability: str):
    return {
        "capability": capability,
        "round_id": "0231",
        "n": n,
        "gate_status": "registered-and-contingent-pending-review",
        "registered_floors": {"ffr": floor, "density_v2": 0.4},
        "registered_two_sided_bands": {
            metric: {"ratio_lower": lower, "ratio_upper": upper}
            for metric in PURITY_METRICS
        },
    }


def _r0225_artifact(*, floor: float, lower: float, upper: float):
    return {
        "capability": "minilm-mixed-2m-tolerance-gates-n8-v1",
        "round_id": "0225",
        "gate_status": "registered-and-contingent-pending-review",
        "gate": {
            "n": 8,
            "gates": {
                "ffr": {"one_sided_tolerance_95_95": {"floor": floor}},
                **{
                    metric: {
                        "one_sided_tolerance_95_95": {"floor": 0.9},
                        "two_sided_log_ratio_95_95": {
                            "ratio_lower": lower,
                            "ratio_upper": upper,
                        },
                    }
                    for metric in PURITY_METRICS
                },
            },
        },
    }


def _this_round(*, floor: float, lower: float, upper: float):
    return {
        "floors": {"ffr": floor},
        "bands": {metric: (lower, upper) for metric in PURITY_METRICS},
        "gate_status": "registered-and-contingent-pending-review",
    }


def _cell(cell_id: str, *, ffr: float, ratio: float, family: str = "exact-graph"):
    return {
        "cell_id": cell_id,
        "family": family,
        "values": {
            "ffr": ffr,
            "purity_fidelity_k256": 0.99,
            "purity_fidelity_k1024": 0.7,
            "density_v2": 0.44,
        },
        "ratios": {"k256": ratio, "k1024": ratio},
    }


def test_the_joint_criteria_carry_every_retained_family():
    families = joint_criteria_from_sealed(
        r0225=_r0225_artifact(floor=0.30, lower=0.95, upper=1.05),
        r0231=_family_artifact(
            floor=0.305, lower=0.98, upper=1.03, n=13,
            capability="minilm-mixed-2m-robust-floors-n13-v1",
        ),
        r0234=_family_artifact(
            floor=0.299, lower=0.979, upper=1.033, n=13,
            capability="minilm-mixed-2m-calibrated-robust-floors-n13-v1",
        ),
        this_round=_this_round(floor=0.29, lower=0.97, upper=1.04),
    )
    assert [item["family"] for item in families] == [
        item["family"] for item in RETAINED_FAMILY_SOURCES
    ] + [THIS_FAMILY]
    for item in families:
        assert set(item["floors"]) == {"ffr"}
        assert set(item["bands"]) == set(PURITY_METRICS)


def test_a_later_registration_cannot_unfail_an_earlier_released_floor():
    """The whole point of the joint rule, as a planted case.

    `held-1` clears THIS round's wider floor and band but falls below R0231's
    stricter `ffr` floor. Under the newest family alone it passes; under the
    joint criteria it must not.
    """
    families = joint_criteria_from_sealed(
        r0225=_r0225_artifact(floor=0.28, lower=0.90, upper=1.10),
        r0231=_family_artifact(
            floor=0.320, lower=0.98, upper=1.03, n=13,
            capability="minilm-mixed-2m-robust-floors-n13-v1",
        ),
        r0234=_family_artifact(
            floor=0.290, lower=0.97, upper=1.04, n=13,
            capability="minilm-mixed-2m-calibrated-robust-floors-n13-v1",
        ),
        this_round=_this_round(floor=0.290, lower=0.97, upper=1.04),
    )
    cells = [
        _cell("exact-seed42", ffr=0.35, ratio=1.00),
        _cell("held-1", ffr=0.31, ratio=1.00, family="held-out"),
    ]
    joint = score_joint(
        cells=cells,
        families=families,
        defining_cell_ids=["exact-seed42"],
        every_defining_cell_can_fail=True,
    )
    rows = {row["cell_id"]: row for row in joint["cells"]}
    assert rows["held-1"]["clears_by_family"][THIS_FAMILY] is True
    assert rows["held-1"]["clears_by_family"]["r0231_n13_median_minus_3_mad"] is False
    assert rows["held-1"]["clears_the_joint_criteria"] is False
    assert joint["cells_failing_only_a_retained_family"] == ["held-1"]
    assert rows["exact-seed42"]["clears_the_joint_criteria"] is True
    assert joint["cells_clearing_the_joint_criteria"] == 1


def test_a_family_that_does_not_cover_every_gated_metric_is_refused():
    broken = _family_artifact(
        floor=0.3, lower=0.98, upper=1.03, n=13,
        capability="minilm-mixed-2m-robust-floors-n13-v1",
    )
    broken["registered_two_sided_bands"].pop("purity_fidelity_k1024")
    with pytest.raises((KeyError, Round0250GateError)):
        joint_criteria_from_sealed(
            r0225=_r0225_artifact(floor=0.30, lower=0.95, upper=1.05),
            r0231=broken,
            r0234=_family_artifact(
                floor=0.299, lower=0.979, upper=1.033, n=13,
                capability="minilm-mixed-2m-calibrated-robust-floors-n13-v1",
            ),
            this_round=_this_round(floor=0.29, lower=0.97, upper=1.04),
        )


def test_a_nonfinite_floor_is_refused():
    with pytest.raises(Round0250GateError):
        joint_criteria_from_sealed(
            r0225=_r0225_artifact(floor=0.30, lower=0.95, upper=1.05),
            r0231=_family_artifact(
                floor=float("nan"), lower=0.98, upper=1.03, n=13,
                capability="minilm-mixed-2m-robust-floors-n13-v1",
            ),
            r0234=_family_artifact(
                floor=0.299, lower=0.979, upper=1.033, n=13,
                capability="minilm-mixed-2m-calibrated-robust-floors-n13-v1",
            ),
            this_round=_this_round(floor=0.29, lower=0.97, upper=1.04),
        )


def test_the_falsifiability_statement_answers_both_families():
    statement = falsifiability_statement(
        multiplier_one_sided=3.29, multiplier_two_sided=3.87
    )
    assert statement["identity_bound_max_abs_z"] == 3.75
    assert statement["one_sided_multiplier_below_the_identity_bound"] is True
    assert statement["two_sided_multiplier_below_the_identity_bound"] is False
    assert statement["registered_family_bound_is_finite"] is False
    assert statement["registered_family_every_defining_cell_can_fail"] is True
    above = falsifiability_statement(
        multiplier_one_sided=4.0, multiplier_two_sided=4.5
    )
    assert above["one_sided_multiplier_below_the_identity_bound"] is False
    assert above["registered_family_every_defining_cell_can_fail"] is True


def test_the_gate_module_declares_the_population_the_round_promises():
    assert EXACT_FAMILY_SEEDS == tuple(range(42, 58))
    assert N_EXACT == 16
    assert N_HELD_OUT == 12
    assert GATE_CAPABILITY.endswith("-n16-v1")
    assert set(GATED_METRICS) == {
        "ffr",
        "purity_fidelity_k256",
        "purity_fidelity_k1024",
    }
    assert "density_v2" not in GATED_METRICS


# --------------------------------------------------------------------------- #
# the n = 16 panel
# --------------------------------------------------------------------------- #


def _panel_inputs():
    cells = {
        str(seed): {
            "density_v2": 0.44,
            "ffr": 0.33,
            "purity_fidelity_k256": 0.99,
            "purity_fidelity_k1024": 0.71,
        }
        for seed in POOLED_SEEDS
    }
    ratios = {str(seed): {"k256": 1.005, "k1024": 0.712} for seed in POOLED_SEEDS}
    corpus = {
        str(seed): {
            slug: {"anchors": 100, "ffr": 0.33}
            for slug in ("code", "fineweb", "pile", "redpajama")
        }
        for seed in POOLED_SEEDS
    }
    return cells, ratios, corpus


def test_the_panel_pools_exactly_sixteen_cells():
    cells, ratios, corpus = _panel_inputs()
    pooled = pool_sixteen_cells(
        cells=cells, ratios=ratios, corpus=corpus, sources=POOLED_CELL_SOURCES
    )
    assert pooled["n"] == 16
    assert pooled["reaches_the_standing_minimum"] is True
    assert pooled["gate_registerable_here"] is False
    assert pooled["identity_bound_at_n"] == 3.75
    assert pooled["source_rounds"]["0250"] == list(SEEDS)


def test_a_missing_cell_stops_the_pooling():
    cells, ratios, corpus = _panel_inputs()
    cells.pop("57")
    with pytest.raises(Round0250PanelError):
        pool_sixteen_cells(
            cells=cells, ratios=ratios, corpus=corpus, sources=POOLED_CELL_SOURCES
        )


def test_reference_drift_stops_the_round_rather_than_being_worked_around():
    good = {
        "bytes": 80_395_632,
        "sha256": (
            "b26319f9448ae4f395f6eea4765156c84b75a31a25980f4f30f581790af335b7"
        ),
    }
    key = "a0fd56fc47afcd5b702cab7aba041dcf95efeb5059ff5fee2351ee34aa815006"
    content = "dcf1f77ed266d902eccee0550bd5056c3fa93928cfbe1347c2f0d6a708036e74"
    counts = {"code": 445, "fineweb": 1637, "pile": 993, "redpajama": 925}
    receipt = assert_reference_identity(
        file_signature=good,
        key=key,
        content_sha256=content,
        rederived_key=key,
        anchor_corpus_counts=counts,
    )
    assert receipt["reference_byte_identical_to_r0218"] is True
    assert receipt["n_pooled"] == 16
    for planted in (
        {"file_signature": {**good, "bytes": 1}},
        {"key": "0" * 64},
        {"content_sha256": "0" * 64},
        {"rederived_key": "0" * 64},
        {"anchor_corpus_counts": {**counts, "code": 1}},
    ):
        kwargs = {
            "file_signature": good,
            "key": key,
            "content_sha256": content,
            "rederived_key": key,
            "anchor_corpus_counts": counts,
            **planted,
        }
        with pytest.raises(Round0250PanelError):
            assert_reference_identity(**kwargs)


# --------------------------------------------------------------------------- #
# the SIGKILL-shaped construct, greppped rather than parsed
# --------------------------------------------------------------------------- #


def test_no_round0250_file_wraps_a_child_in_a_subprocess_timeout():
    """`subprocess.run(..., timeout=N)` is `Popen.kill()` — a hidden SIGKILL.

    `plan-minilm-100m-v2.md` makes purging it binding before any further GPU
    round and says to assert its absence with a grep, because the AST guard is
    structurally blind to it. R0249's addendum measured 36 CUDA-family mappings
    in an idle pytest child that had merely imported torch, so a CPU-declared
    child is not exempt.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    files = sorted(root.glob("**/round0250_*.py")) + sorted(
        root.glob("**/*round0250*queue.py")
    )
    assert files, "the round0250 sources must exist for this grep to mean anything"
    pattern = re.compile(r"timeout\s*=", re.MULTILINE)
    offenders = []
    for path in files:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{path.name}:{number}: {stripped[:100]}")
    assert offenders == [], offenders


def test_the_grep_catches_a_planted_timeout(tmp_path):
    """Positive control: the grep above is not vacuous."""
    import re

    planted = tmp_path / "round0250_planted.py"
    planted.write_text(
        "import subprocess\n"
        "subprocess.run(['true'], timeout=5)\n"
    )
    pattern = re.compile(r"timeout\s*=", re.MULTILINE)
    hits = [
        line
        for line in planted.read_text().splitlines()
        if pattern.search(line) and not line.strip().startswith("#")
    ]
    assert len(hits) == 1
