from __future__ import annotations

import json
import os

import pytest

from basemap.round0213_scaling_story_synthesis import (
    HIGH_DOSE,
    LOW_DOSE,
    OPERATING_RULE,
    RETENTION_FLOOR,
    Round0213Error,
    dose_axis,
    loss_locality,
    operating_rule,
    width_axis,
)

R0190 = ("/data/latent-basemap/runs/round-0190/queue/artifacts/"
         "jina-composition-boundary-three-seed-synthesis-v1/"
         "three-seed-boundary-synthesis.json")
R0207 = ("/data/latent-basemap/runs/round-0207/queue/artifacts/"
         "jina-width-by-n-factorial-capacity-economics-v1/width-factorial.json")


def _live():
    if not (os.path.exists(R0190) and os.path.exists(R0207)):
        pytest.skip("sealed R0190/R0207 artifacts are not present")
    return json.load(open(R0190))["decision"], json.load(open(R0207))


def test_dose_axis_reproduces_the_sealed_seed_sensitivity() -> None:
    r0190, r0207 = _live()
    axis = dose_axis(
        high_dose_retention=r0190["retention_summary"]["values"],
        high_dose_positive_by_seed=r0190["positive_by_seed"],
        low_dose_full_over_half=r0207["retentions"]["h2048"]["pile_ffr"]["full_over_half"],
        seed_noise_sd=r0190["width_null_noise_scale"]["value"],
    )
    assert axis["high_dose"]["seeds"] == 3
    assert axis["high_dose"]["seeds_clearing_floor"] == 2
    assert axis["high_dose"]["clears_floor_on_mean"] is False
    assert axis["high_dose"]["floor_inside_one_sd"] is True
    assert axis["low_dose"]["clears_floor"] is True
    assert axis["low_dose"]["full_over_half"] > 1.0
    assert axis["retention_floor"] == RETENTION_FLOOR == 0.97


def test_every_width_cell_is_at_the_low_dose() -> None:
    """The claim the campaign brief makes about high-dose width has no cell."""
    _r0190, r0207 = _live()
    for width, rungs in r0207["cells"].items():
        for rung, cell in rungs.items():
            assert cell["positive_draws_per_edge"] == pytest.approx(
                LOW_DOSE, abs=1e-6
            ), f"{width}/{rung} is not at the low dose"
    assert "fixed dose" in r0207["claim_scope"]
    assert HIGH_DOSE != LOW_DOSE


def test_width_axis_refuses_the_capacity_absorbs_dose_claim() -> None:
    r0190, r0207 = _live()
    axis = width_axis(
        contrasts=r0207["width_contrasts"],
        seed_noise_sd=r0190["width_null_noise_scale"]["value"],
        low_dose_widths_flat=r0207["outcome"] == "both-widths-flat-at-low-dose",
    )
    assert axis["capacity_absorbs_dose_claim_supported"] is False
    assert axis["widths_measured_at_high_dose"] == [2048]
    assert axis["missing_cell"]["hidden_dimension"] == 4096
    assert axis["missing_cell"]["target_positive_draws_per_edge"] == HIGH_DOSE
    assert axis["cells"]["full"]["train_wall_ratio_h4096_over_h2048"] > 3.0
    # the full-rung width delta is real but barely above seed noise
    assert axis["cells"]["full"]["exceeds_seed_noise_sd"] is True
    assert 1.0 < axis["cells"]["full"]["delta_in_seed_noise_sds"] < 1.5


def test_loss_is_diffuse_not_localised() -> None:
    _r0190, r0207 = _live()
    locality = loss_locality(context=r0207["capacity_context"])
    assert locality["pattern"] == "diffuse"
    assert locality["localised"] is False
    assert locality["strongest_absolute_spearman"] < 0.05
    assert locality["k256_losing_cluster_coverage"] > 0.95


def test_operating_rule_states_both_what_is_and_is_not_supported() -> None:
    r0190, r0207 = _live()
    dose = dose_axis(
        high_dose_retention=r0190["retention_summary"]["values"],
        high_dose_positive_by_seed=r0190["positive_by_seed"],
        low_dose_full_over_half=r0207["retentions"]["h2048"]["pile_ffr"]["full_over_half"],
        seed_noise_sd=r0190["width_null_noise_scale"]["value"],
    )
    width = width_axis(
        contrasts=r0207["width_contrasts"],
        seed_noise_sd=r0190["width_null_noise_scale"]["value"],
        low_dose_widths_flat=True,
    )
    rule = operating_rule(dose=dose, width=width)
    assert rule["rule"] == OPERATING_RULE
    assert rule["hidden_dimension"] == 2048
    assert rule["target_positive_draws_per_edge"] == LOW_DOSE
    assert any("no such cell" in item for item in rule["not_supported"])
    assert any("never probed" in item for item in rule["not_supported"])
    assert any("composition-matched" in item for item in rule["not_supported"])


def test_rule_refuses_itself_if_the_low_dose_stopped_being_flat() -> None:
    r0190, r0207 = _live()
    dose = dose_axis(
        high_dose_retention=r0190["retention_summary"]["values"],
        high_dose_positive_by_seed=r0190["positive_by_seed"],
        low_dose_full_over_half=0.5,  # counterfactual: low dose also regresses
        seed_noise_sd=r0190["width_null_noise_scale"]["value"],
    )
    width = width_axis(
        contrasts=r0207["width_contrasts"],
        seed_noise_sd=r0190["width_null_noise_scale"]["value"],
        low_dose_widths_flat=True,
    )
    with pytest.raises(Round0213Error):
        operating_rule(dose=dose, width=width)


def test_fewer_than_three_seeds_is_not_a_boundary_claim() -> None:
    with pytest.raises(Round0213Error):
        dose_axis(
            high_dose_retention=[0.95, 0.98],
            high_dose_positive_by_seed={"seed42": True, "seed43": False},
            low_dose_full_over_half=1.05,
            seed_noise_sd=0.0116,
        )
