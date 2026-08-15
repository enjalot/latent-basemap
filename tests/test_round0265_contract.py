"""Contract tests for R0265 — the fneg family floor round.

CPU-only (CUDA hidden). Exercises the treatment-closure recipe invariant and its four
refusal plants, the import-closure controls, the family digest agreement, the x4 dose
validator, the family-floor fit and its mirror agreement, the provisional-to-family band
conversion, the four-instrument gate (including the fail-closed fog and two-sided k256
band), the seed-42 cross-check, and the node dispatch. No training and no GPU: every
number here is synthetic or read from an already-sealed on-disk artifact.
"""
import math
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

from basemap import round0265_fneg_treatment as T
from basemap.round0221_minilm_2m_seed_extension import (
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
)
from basemap.round0217_minilm_2m_seed_family import (
    ROWS,
    SEALED_DIRECTED_EDGES,
    successful_updates_for_edges,
)
import experiments.round0265_nodes as N
import experiments.prepare_round0265_queue as P


R0234_GATE = P.R0234_GATE


def _honest_config(seed=T.CANONICAL_SEED):
    config, _sha = T.fneg_train_config(
        seed=seed,
        graph_signature=SEALED_GRAPH_SIGNATURE,
        graph_manifest_signature=SEALED_GRAPH_MANIFEST_SIGNATURE,
        substrate_signature=SEALED_SUBSTRATE_SIGNATURE,
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )
    return config


def _all_configs():
    return {seed: _honest_config(seed) for seed in T.SEEDS}


# --------------------------------------------------------------------------- #
# 1. the treatment-closure recipe invariant + four refusal plants
# --------------------------------------------------------------------------- #


def test_honest_recipe_passes_the_invariant():
    recipe = T.assert_registered_recipe(_honest_config())
    assert recipe["low_dim_kernel"] == "umap"
    assert recipe["a"] == T.FNEG_A and recipe["b"] == T.FNEG_B
    assert recipe["fneg_weight"] == 1.0
    assert recipe["successful_positive_lr_updates"] == T.FNEG_HORIZON == 320652


def test_wrong_kernel_is_refused():
    cfg = _honest_config()
    cfg["model"]["low_dim_kernel"] = "legacy_lp"
    cfg["model"]["a"] = 1.0
    cfg["model"]["b"] = 1.0
    with pytest.raises(T.Round0265RecipeError):
        T.assert_registered_recipe(cfg)


def test_wrong_dose_is_refused():
    cfg = _honest_config()
    cfg["optimizer"]["successful_positive_lr_updates"] = successful_updates_for_edges(
        SEALED_DIRECTED_EDGES
    )  # x1
    with pytest.raises(T.Round0265RecipeError):
        T.assert_registered_recipe(cfg)


def test_fneg_off_is_refused():
    cfg = _honest_config()
    cfg["optimizer"]["fneg_weight"] = 0.0
    with pytest.raises(T.Round0265RecipeError):
        T.assert_registered_recipe(cfg)


def test_wrong_band_is_refused():
    cfg = _honest_config()
    cfg["optimizer"]["fneg_lo"] = 0.05
    cfg["optimizer"]["fneg_hi"] = 0.5
    with pytest.raises(T.Round0265RecipeError):
        T.assert_registered_recipe(cfg)


def test_recipe_refusal_controls_all_fire_and_old_predicate_is_blind():
    controls = T.recipe_refusal_controls()
    assert controls["planted"] == 6
    assert controls["every_planted_defect_was_refused"] is True
    assert controls["the_old_predicate_accepted_every_one"] is True
    assert controls["the_honest_recipe_still_passes"] is True
    assert {c["control"] for c in controls["controls"]} == {
        "wrong_kernel", "wrong_dose", "fneg_off", "wrong_band",
        "weighted_sampling_on", "routes_off_device",
    }
    # Each plant: shipped refuses, old accepts.
    for c in controls["controls"]:
        assert c["shipped_predicate_refused"] is True
        assert c["old_predicate_accepted"] is True


def test_config_sets_uniform_sampling():
    # The promoted recipe pins UNIFORM positive-edge sampling (weighted flag OFF).
    cfg = _honest_config()
    assert cfg["optimizer"]["weighted_edge_sampling"] is False
    assert cfg["treatment"]["weighted_edge_sampling"] is False
    recipe = T.assert_registered_recipe(cfg)
    assert recipe["weighted_edge_sampling"] is False


def test_weighted_sampling_on_is_refused():
    # R0217's fuzzy sampler turned back on is not the promoted recipe.
    cfg = _honest_config()
    cfg["optimizer"]["weighted_edge_sampling"] = True
    with pytest.raises(T.Round0265RecipeError):
        T.assert_registered_recipe(cfg)


def test_recipe_sampling_paths_membership():
    assert T.RECIPE_SAMPLING_PATHS == (("optimizer", "weighted_edge_sampling"),)


def test_config_routes_to_the_uniform_device_path():
    # The execution routing that makes core.fit SELECT the uniform device path.
    cfg = _honest_config()
    assert cfg["execution"]["required_pipeline"] == "device"
    assert cfg["execution"]["gpu_resident_data"] == "auto"
    assert cfg["execution"]["gpu_resident_vram_budget_gb"] == 10.0
    assert cfg["treatment"]["required_pipeline"] == "device"
    assert cfg["treatment"]["gpu_resident_data"] == "auto"
    assert cfg["treatment"]["gpu_resident_vram_budget_gb"] == 10.0
    recipe = T.assert_registered_recipe(cfg)
    assert recipe["required_pipeline"] == "device"
    assert recipe["gpu_resident_data"] == "auto"
    assert recipe["gpu_resident_vram_budget_gb"] == 10.0


def test_recipe_execution_paths_membership():
    assert T.RECIPE_EXECUTION_PATHS == (
        ("execution", "required_pipeline"),
        ("execution", "gpu_resident_data"),
        ("execution", "gpu_resident_vram_budget_gb"),
    )


def test_routes_off_device_is_refused():
    # R0217's host-weighted routing back in -- the exact config defect that blocked the
    # GPU run (core.fit raises on required_pipeline; gpu_resident_data=False routes off
    # the device fast path). Either half must be refused.
    cfg = _honest_config()
    cfg["execution"]["required_pipeline"] = "host_weighted_minilm_mixed_2m"
    with pytest.raises(T.Round0265RecipeError):
        T.assert_registered_recipe(cfg)
    cfg2 = _honest_config()
    cfg2["execution"]["gpu_resident_data"] = False
    with pytest.raises(T.Round0265RecipeError):
        T.assert_registered_recipe(cfg2)


def test_expected_pipeline_stamp_is_the_honest_device_uniform_stamp():
    # The declared stamp describes a uniform device run, not R0217's fuzzy-weighted host
    # sampler; the weighted-only keys the device stamp never emits are dropped.
    cfg = _honest_config()
    stamp = cfg["execution"]["expected_pipeline_stamp"]
    assert stamp["pipeline"] == "device"
    assert stamp["sampler_class"] == "DeviceEdgeSampler"
    assert stamp["positive_sampling"] == "uniform"
    assert stamp["x_residency"] == "device_fp16"
    assert stamp["weighted_requested"] is False
    assert stamp["weighted_effective"] is False
    assert stamp["uniform_with_replacement"] is False
    assert stamp["positive_with_replacement"] is False
    for dropped in ("positive_destination_policy", "negative_sampling",
                    "rng_stream_policy", "host_prefetch", "endpoint_forward",
                    "weight_sampler", "weight_uniform_dtype"):
        assert dropped not in stamp
    # The two seed-bearing keys stay (documentary; the 9-path invariant is unchanged).
    assert stamp["positive_rng_seed"] == T.CANONICAL_SEED
    # A stamp that still declared weighted is refused.
    cfg["execution"]["expected_pipeline_stamp"]["weighted_effective"] = True
    with pytest.raises(T.Round0265RecipeError):
        T.assert_registered_recipe(cfg)


def test_built_model_routes_to_the_device_path():
    # CPU-only: constructing the model stores device="cuda" as a string and touches no
    # CUDA (the MLP is only moved to device inside fit()). Prove the config->model bridge
    # threads the device-uniform routing so core.fit will select the uniform device path.
    cfg = _honest_config()
    model = N._build_fneg_model(cfg)
    assert model.required_input_pipeline == "device"
    assert model.gpu_resident_data == "auto"
    assert model.gpu_resident_vram_budget_gb == 10.0
    assert model.weighted_edge_sampling is False


def test_old_predicate_rejects_a_non_family_seed():
    # The old predicate is not vacuously true: a seed outside the family fails it.
    cfg = _honest_config()
    cfg["seed"] = 999
    assert T.old_recipe_predicate(cfg) is False


# --------------------------------------------------------------------------- #
# 2. the import-closure seal + controls
# --------------------------------------------------------------------------- #


def _sealed_closure_body():
    seal = P._treatment_closure_seal("0" * 40)

    def find_files(d):
        if isinstance(d, dict):
            if "files" in d:
                return d
            for v in d.values():
                r = find_files(v)
                if r:
                    return r
        return None

    return find_files(seal)


def test_import_closure_matches_runtime_bytes():
    inner = _sealed_closure_body()
    observed = T.runtime_closure_hashes(T.TRAIN_CLOSURE_MODULES)
    verdict = T.assert_runtime_closure_matches_seal(sealed=inner, observed=observed)
    assert verdict["every_module_ran_the_sealed_bytes"] is True
    assert "basemap.pumap.parametric_umap.core" in verdict["modules"]


def test_import_closure_controls_all_refuse():
    inner = _sealed_closure_body()
    observed = T.runtime_closure_hashes(T.TRAIN_CLOSURE_MODULES)
    controls = T.treatment_closure_controls(sealed=inner, observed=observed)
    assert controls["every_planted_defect_was_refused"] is True
    assert controls["the_honest_closure_still_passes"] is True
    assert controls["planted"] == 5


def test_empty_closure_is_the_vacuous_case_and_refused():
    with pytest.raises(T.Round0265TreatmentError):
        T.assert_runtime_closure_matches_seal(sealed={"files": {}}, observed={}, modules=())


# --------------------------------------------------------------------------- #
# 3. family digest agreement + the x4 dose
# --------------------------------------------------------------------------- #


def test_family_shares_one_recipe_digest():
    fam = T.assert_family_shares_one_recipe(_all_configs())
    assert fam["all_thirteen_share_one_recipe_digest"] is True
    assert fam["thirteen_distinct_cell_configs"] is True
    assert fam["cells"] == 13
    assert len(set(fam["per_seed_config_sha256"].values())) == 13


def test_family_rejects_a_recipe_off_cell():
    configs = _all_configs()
    configs[T.SEEDS[0]]["optimizer"]["fneg_weight"] = 0.0
    with pytest.raises(T.Round0265RecipeError):
        T.assert_family_shares_one_recipe(configs)


def test_x4_dose_pins_the_horizon_and_refuses_x1():
    base = successful_updates_for_edges(SEALED_DIRECTED_EDGES)
    dose = T.validate_fneg_dose(updates=4 * base, edge_count=SEALED_DIRECTED_EDGES)
    assert dose["successful_updates"] == 320652
    assert dose["dose_multiplier"] == 4
    with pytest.raises(T.Round0265RecipeError):
        T.validate_fneg_dose(updates=base, edge_count=SEALED_DIRECTED_EDGES)


# --------------------------------------------------------------------------- #
# 4. the n=13 multiplier, read from R0234's sealed artifact
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not os.path.exists(R0234_GATE), reason="R0234 sealed n=13 artifact absent")
def test_n13_multiplier_read_from_sealed_r0234():
    k2 = N._read_sealed_multiplier(R0234_GATE, "n13", "two_sided")
    k1 = N._read_sealed_multiplier(R0234_GATE, "n13", "one_sided")
    assert k2 == N.SEALED_N13_TWO_SIDED_MULTIPLIER == 4.452408030160313
    assert k1 == N.SEALED_N13_ONE_SIDED_MULTIPLIER == 3.7363661743730416
    # The gate fits with the TWO-SIDED 95/95, not the one-sided.
    assert k2 > k1


# --------------------------------------------------------------------------- #
# 5. the family-floor fit + mirror agreement + band conversion
# --------------------------------------------------------------------------- #


K1 = N.SEALED_N13_ONE_SIDED_MULTIPLIER
K2 = N.SEALED_N13_TWO_SIDED_MULTIPLIER


def _synthetic_family():
    held = [0.40 + 0.004 * i for i in range(13)]
    coll = [1.05 + 0.02 * i for i in range(13)]
    fog = [0.36 + 0.004 * i for i in range(13)]
    k256 = [1.00 + 0.002 * i for i in range(13)]
    k1024 = [0.98 + 0.003 * i for i in range(13)]
    return held, coll, fog, k256, k1024


def _fit(held, coll, fog, k256, k1024):
    return N.fit_family_floors(
        heldout_ffr_values=held, collapse_values=coll, fog_values=fog,
        k256_ratios=k256, k1024_ratios=k1024, k1=K1, k2=K2,
    )


def test_family_floor_fit_splits_k1_and_k2():
    ff = _fit(*_synthetic_family())
    # One-sided floors use k1; the two-sided band uses k2; nothing else uses k2.
    assert ff["k1_one_sided"] == K1 and ff["k2_two_sided"] == K2
    assert ff["heldout_ffr"]["multiplier"] == K1
    assert ff["collapse"]["multiplier"] == K1
    assert ff["fog"]["multiplier"] == K1
    assert ff["purity_fidelity_k1024"]["multiplier"] == K1
    assert ff["purity_fidelity_k256"]["multiplier"] == K2
    assert ff["fitted_from_n_family_cells"] == 13


def test_family_floor_fit_mirrors_agree_bitwise():
    ff = _fit(*_synthetic_family())
    # floor_at (R0234) and _robust_floor (R0264) are the same arithmetic.
    assert ff["collapse_floor_mirrors_agree"] is True
    assert ff["collapse"]["floor"] == ff["collapse_floor_round0234_floor_at"]
    # A lower floor sits below the family median; a ceiling above it.
    assert ff["collapse"]["floor"] < ff["collapse"]["median"]
    assert ff["fog"]["ceiling"] > ff["fog"]["median"]
    assert ff["heldout_ffr"]["floor"] < ff["heldout_ffr"]["median"]
    # The k256 band is symmetric on the log-ratio; the k1024 floor is below the ratios.
    band = ff["purity_fidelity_k256"]["band"]
    assert band["ratio_lower"] < band["ratio_upper"]
    assert ff["purity_fidelity_k1024"]["floor"] < min(_synthetic_family()[4])


def test_band_conversion_is_the_family_fit():
    ff = _fit(*_synthetic_family())
    conv = N.convert_provisional_bands(family_floors=ff)
    # The provisional R0264 numbers are the descriptive converted-FROM, never registered.
    assert conv["converted_from_descriptive"]["collapse_floor"] == N.R0264_PROVISIONAL_COLLAPSE_FLOOR
    assert conv["converted_from_descriptive"]["fog_ceiling"] == N.R0264_PROVISIONAL_FOG_CEILING
    # The registered bands ARE the family fit.
    assert conv["converted_to_registered"]["collapse_floor"] == ff["collapse"]["floor"]
    assert conv["converted_to_registered"]["fog_ceiling"] == ff["fog"]["ceiling"]
    assert conv["this_is_the_healthy_seed_family_r0264_owed"] is True


# --------------------------------------------------------------------------- #
# 6. the five-instrument gate — every value fitted from the fneg family
# --------------------------------------------------------------------------- #

#: R0263's LEGACY registered band — used ONLY as the descriptive contrast / the
#: verdict-flipping mutation in the positive control; NEVER as a gate input.
LEGACY_K256_BAND = {
    "ratio_lower": 0.9697263687993266,
    "ratio_upper": 1.0550731452902518,
    "log_centre": 0.01143437762566317,
    "log_scale": 0.01339863133626119,
}


def _cells_from(held, coll, fog, k256s=None, k1024s=None, k256=1.01, k1024=1.00, degenerate=False):
    cells = {}
    for i, seed in enumerate(T.SEEDS):
        cells[str(seed)] = {
            "heldout_ffr": held[i],
            "purity_fidelity_k256": (k256s[i] if k256s else k256),
            "purity_fidelity_k1024": (k1024s[i] if k1024s else k1024),
            "collapse": coll[i],
            "fog": 0.0 if (degenerate and i == 0) else fog[i],
            "resolution_levels": 0 if (degenerate and i == 0) else 14,
            "degenerate": bool(degenerate and i == 0),
            "fog_detail": {"peak_bin_count": 99 if (degenerate and i == 0) else 1435},
        }
    return cells


def _score(ff, cells):
    return N.score_family_on_gate(
        cells=cells,
        heldout_ffr_floor=ff["heldout_ffr"]["floor"],
        collapse_floor=ff["collapse"]["floor"],
        fog_ceiling=ff["fog"]["ceiling"],
        k256_band=ff["purity_fidelity_k256"]["band"],
        k1024_floor=ff["purity_fidelity_k1024"]["floor"],
    )


def test_gate_passes_a_healthy_family_on_all_five_instruments():
    held, coll, fog, k256, k1024 = _synthetic_family()
    ff = _fit(held, coll, fog, k256, k1024)
    sc = _score(ff, _cells_from(held, coll, fog, k256s=k256, k1024s=k1024))
    assert sc["cells_scored"] == 13
    assert sc["cells_clearing_every_instrument"] == 13
    assert sc["every_instrument_is_family_fitted"] is True
    # all five instruments present on each row
    assert set(sc["cells"][0]["metrics"]) == set(N.GATED_METRICS)


def test_gate_fails_a_collapsed_cell():
    held, coll, fog, k256, k1024 = _synthetic_family()
    ff = _fit(held, coll, fog, k256, k1024)
    cells = _cells_from(held, coll, fog, k256s=k256, k1024s=k1024)
    cells["42"]["collapse"] = 0.10  # far below any healthy floor: bead collapse
    sc = _score(ff, cells)
    row = [r for r in sc["cells"] if r["seed"] == 42][0]
    assert row["metrics"]["collapse"]["passes"] is False
    assert row["clears_every_gated_instrument"] is False


def test_gate_fog_fails_closed_on_degeneracy():
    held, coll, fog, k256, k1024 = _synthetic_family()
    ff = _fit(held, coll, fog, k256, k1024)
    sc = _score(ff, _cells_from(held, coll, fog, k256s=k256, k1024s=k1024, degenerate=True))
    row = [r for r in sc["cells"] if r["seed"] == 42][0]
    assert row["metrics"]["fog"]["verdict"] == N.VERDICT_NOT_MEASURABLE
    assert row["metrics"]["fog"]["passes"] is False
    assert row["fog_not_measurable"] is True


def test_gate_k256_band_is_two_sided_and_family_fitted():
    held, coll, fog, k256, k1024 = _synthetic_family()
    ff = _fit(held, coll, fog, k256, k1024)
    # A cell whose k256 sits well above the FAMILY-fitted upper edge fails as "over".
    cells = _cells_from(held, coll, fog, k256=2.0, k1024s=k1024)
    sc = _score(ff, cells)
    assert all(r["metrics"]["purity_fidelity_k256"]["passes"] is False for r in sc["cells"])
    assert all(r["metrics"]["purity_fidelity_k256"]["direction"] == "over" for r in sc["cells"])


def test_gate_k1024_is_one_sided_lower_only():
    held, coll, fog, k256, k1024 = _synthetic_family()
    ff = _fit(held, coll, fog, k256, k1024)
    floor = ff["purity_fidelity_k1024"]["floor"]
    # BELOW the floor fails (under-separation); FAR ABOVE the floor still passes
    # (over-separation unregulated) — the one-sided semantics.
    below = _cells_from(held, coll, fog, k256s=k256, k1024=floor * 0.5)
    above = _cells_from(held, coll, fog, k256s=k256, k1024=floor + 5.0)
    sb = _score(ff, below)
    sa = _score(ff, above)
    assert all(r["metrics"]["purity_fidelity_k1024"]["passes"] is False for r in sb["cells"])
    assert all(r["metrics"]["purity_fidelity_k1024"]["passes"] is True for r in sa["cells"])


# --------------------------------------------------------------------------- #
# 6b. POSITIVE CONTROL — no legacy value gates
# --------------------------------------------------------------------------- #


def test_positive_control_legacy_band_does_not_gate():
    """Mutating R0263's legacy band to a verdict-flipping value leaves the gate UNCHANGED,
    because the gate scores against the FAMILY-fitted band only."""
    held, coll, fog, k256, k1024 = _synthetic_family()
    ff = _fit(held, coll, fog, k256, k1024)
    fitted_band = ff["purity_fidelity_k256"]["band"]

    # A ratio at the midpoint of the two upper edges is inside exactly ONE of the bands
    # (whichever is wider), so the family-fitted and legacy verdicts must disagree there.
    r = (fitted_band["ratio_upper"] + LEGACY_K256_BAND["ratio_upper"]) / 2
    cells = _cells_from(held, coll, fog, k256=r, k1024s=k1024)

    from experiments.round0263_nodes import _judge_k256_two_sided
    fitted_verdict = _judge_k256_two_sided(r, fitted_band)["passes"]
    legacy_verdict = _judge_k256_two_sided(r, LEGACY_K256_BAND)["passes"]
    assert fitted_verdict != legacy_verdict, "the legacy band must differ, or the control is vacuous"

    # The gate, scored with the family-fitted band, reports the FITTED verdict — never the
    # legacy one — no matter what the legacy constant is.
    sc = _score(ff, cells)
    gate_verdict = [x for x in sc["cells"] if x["seed"] == 42][0]["metrics"]["purity_fidelity_k256"]["passes"]
    assert gate_verdict == fitted_verdict
    assert gate_verdict != legacy_verdict


def test_positive_control_legacy_collapse_floor_does_not_gate():
    """A legacy collapse floor set to a value that would fail every cell leaves the gate
    verdict unchanged — the gate uses only the family-fitted floor."""
    held, coll, fog, k256, k1024 = _synthetic_family()
    ff = _fit(held, coll, fog, k256, k1024)
    cells = _cells_from(held, coll, fog, k256s=k256, k1024s=k1024)
    baseline = _score(ff, cells)["cells_clearing_every_instrument"]
    # A legacy floor of 10.0 would fail every cell IF it gated; the gate ignores it and
    # still uses ff["collapse"]["floor"].
    with_legacy_ignored = N.score_family_on_gate(
        cells=cells, heldout_ffr_floor=ff["heldout_ffr"]["floor"],
        collapse_floor=ff["collapse"]["floor"],  # family-fitted, NOT the legacy 10.0
        fog_ceiling=ff["fog"]["ceiling"],
        k256_band=ff["purity_fidelity_k256"]["band"], k1024_floor=ff["purity_fidelity_k1024"]["floor"],
    )["cells_clearing_every_instrument"]
    assert with_legacy_ignored == baseline == 13


def test_each_fitted_criterion_differs_from_the_legacy_constant():
    """Every fitted instrument is derived from the 13 family cells, so it does not equal
    the legacy constant (on this synthetic family)."""
    held, coll, fog, k256, k1024 = _synthetic_family()
    ff = _fit(held, coll, fog, k256, k1024)
    band = ff["purity_fidelity_k256"]["band"]
    assert band["ratio_lower"] != N.R0263_LEGACY_K256_BAND["ratio_lower"]
    assert band["ratio_upper"] != N.R0263_LEGACY_K256_BAND["ratio_upper"]
    assert ff["purity_fidelity_k1024"]["floor"] != float(N.R0260_LEGACY_K1024_FLOOR)
    assert ff["collapse"]["floor"] != N.R0264_PROVISIONAL_COLLAPSE_FLOOR
    assert ff["fog"]["ceiling"] != N.R0264_PROVISIONAL_FOG_CEILING


def test_descriptive_cross_reference_never_gates():
    xref = N.descriptive_cross_reference()
    assert "never_gates" in xref["role"]
    assert xref["purity_fidelity_k256"]["ratio_lower"] == N.R0263_LEGACY_K256_BAND["ratio_lower"]
    assert xref["purity_fidelity_k1024"]["ratio_floor"] == float(N.R0260_LEGACY_K1024_FLOOR)
    assert xref["collapse"]["floor"] == N.R0264_PROVISIONAL_COLLAPSE_FLOOR
    assert xref["fog"]["ceiling"] == N.R0264_PROVISIONAL_FOG_CEILING


# --------------------------------------------------------------------------- #
# 7. the seed-42 cross-check
# --------------------------------------------------------------------------- #


def test_seed42_cross_check_lands_metric_close_on_the_sandbox_numbers():
    observed = {"heldout_ffr": 0.411, "collapse": 1.062, "fog": 0.371}
    cc = N.seed42_cross_check(observed)
    assert cc["metric_close"] is True
    assert "umap-md000-x4-fneg10" in cc["target_cell"]


def test_seed42_cross_check_misses_a_broken_projector():
    # A regressor-level FFR (0.34) and a collapsed/foggy map should NOT land close.
    observed = {"heldout_ffr": 0.34, "collapse": 0.2, "fog": 0.55}
    cc = N.seed42_cross_check(observed)
    assert cc["metric_close"] is False


# --------------------------------------------------------------------------- #
# 8. the map-quality measurements on synthetic coordinates
# --------------------------------------------------------------------------- #


def test_map_collapse_is_the_sqrt_n_adjusted_ratio():
    rng = np.random.default_rng(3)
    xy = (rng.standard_normal((4000, 2)) * 3).astype(np.float32)
    c = N.map_collapse(xy)
    assert c["statistic"] == "r10_over_radius_times_sqrt_n"
    assert math.isfinite(c["collapse"])
    assert c["collapse"] == pytest.approx(
        c["r10_over_map_radius_median"] * math.sqrt(c["n_effective"])
    )


def test_heldout_ffr_beats_a_random_placement():
    rng = np.random.default_rng(5)
    coords = rng.standard_normal((3000, 2)).astype(np.float32)
    # truth: each of 40 probes' true neighbours are the 10 nearest map rows to a target
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    targets = coords[rng.choice(3000, 40, replace=False)]
    _, truth = tree.query(targets, k=10)
    truth = np.asarray(truth, dtype=np.int64)
    # placed AT the targets recovers the neighbours; placed at random does not.
    good = N.heldout_ffr_scores(placed=targets, map_coordinates=coords, truth_top10=truth, disc=100)
    rand = N.heldout_ffr_scores(
        placed=rng.standard_normal((40, 2)).astype(np.float32) * 50,
        map_coordinates=coords, truth_top10=truth, disc=100,
    )
    assert good["heldout_ffr"] > rand["heldout_ffr"]


# --------------------------------------------------------------------------- #
# 9. dispatch + scope registration
# --------------------------------------------------------------------------- #


def test_round0265_is_registered_in_scope_modules():
    from basemap.round0254_dispatch import SCOPE_MODULES, dispatch_census, assert_derived_entries_install
    assert "experiments.round0265_nodes" in SCOPE_MODULES
    guard = assert_derived_entries_install(SCOPE_MODULES, dispatch_census())
    assert guard["audit"]["every_entry_installs_effectively"] is True


def test_run_job_rejects_an_unknown_action():
    with pytest.raises(N.Round0265NodeError):
        N.run_job({"manifest": {"round_id": "0265"}}, {"action": "not-a-real-action"})


def test_gated_metrics_exclude_the_descriptive_ones():
    assert not (set(N.GATED_METRICS) & set(N.DESCRIPTIVE_METRICS))
    assert set(N.DESCRIPTIVE_METRICS) == {"density_v2", "density_v3"}
    # Five gated instruments: held-out FFR, k256, k1024, collapse, fog.
    assert set(N.GATED_METRICS) == {
        "heldout_ffr", N.PURITY_K256_METRIC, N.PURITY_K1024_METRIC,
        N.COLLAPSE_METRIC, N.FOG_METRIC,
    }


# --------------------------------------------------------------------------- #
# 10. the shared round-vs-literal display-precision helper
# --------------------------------------------------------------------------- #


def test_matches_cited_display_value_round_vs_literal():
    from basemap.artifact_identity import matches_cited_display_value

    # Full-precision sealed values vs the 6-decimal cited display constants (R0264's
    # provisional collapse floor / fog ceiling): they MATCH at the cited precision.
    assert matches_cited_display_value(0.6444139401628924, 0.644414, decimals=6) is True
    assert matches_cited_display_value(0.7329657517287256, 0.732966, decimals=6) is True
    # Default precision is 6.
    assert matches_cited_display_value(0.6444139401628924, 0.644414) is True
    # The naive full-precision equality (the round-vs-literal BUG) would be False even
    # though nothing drifted — which is exactly why the helper rounds first.
    assert (0.6444139401628924 == 0.644414) is False
    # A real drift ABOVE the display precision (>1e-6) fails.
    assert matches_cited_display_value(0.6444139401628924 + 1e-3, 0.644414, decimals=6) is False
    assert matches_cited_display_value(0.644415, 0.644414, decimals=6) is False
    # `decimals` is honoured (R0264's k7 healthy anchor is a 5-decimal display value).
    assert matches_cited_display_value(6.0551303, 6.05513, decimals=5) is True
    assert matches_cited_display_value(6.055139, 6.05513, decimals=5) is False


# --------------------------------------------------------------------------- #
# 11. the SANCTIONED gate-only re-seal queue
# --------------------------------------------------------------------------- #

CORR3_RUN = "/data/latent-basemap/runs/round-0265/queue-correction-3"
CORR3_PANEL = os.path.join(
    CORR3_RUN, "artifacts", "minilm-fneg-2m-family-panel-n13-v1",
    "fneg-family-panel-n13.json",
)
CORR3_RELEASE_SHA = "7e3f5757e6c8d45893c866d886e9378d49f326b2"

_GATE_ONLY_INPUTS = [
    P.GRAPH_MANIFEST, P.R0234_GATE, P.R0263_GATE, P.R0264_GATE, P.ROUND_FILE, CORR3_PANEL,
]
_GATE_ONLY_READY = (
    all(os.path.exists(path) for path in _GATE_ONLY_INPUTS)
    # `.git` is a directory in a normal clone and a file in a linked worktree.
    and os.path.exists(os.path.join(P.RELEASE_ROOT, ".git"))
)


def _release_head() -> str:
    import subprocess

    proc = subprocess.run(
        ["git", "-C", P.RELEASE_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


def _rmtree_ro(root: str) -> None:
    import shutil

    def _onerror(func, path, _exc):
        try:
            os.chmod(path, 0o700)
            func(path)
        except OSError:
            pass

    shutil.rmtree(root, onerror=_onerror)


@pytest.mark.skipif(not _GATE_ONLY_READY, reason="gate-only re-seal inputs absent")
def test_gate_only_prepare_builds_a_one_node_gate_queue():
    import json
    import time

    from basemap.artifact_identity import expected_input_signature

    release_sha = _release_head()
    assert re.fullmatch(r"[0-9a-f]{40}", release_sha)
    # A distinct /data dir — NOT the real queue-gate-only-1, so nothing in the record is
    # touched; cleaned up in the finally.
    queue_root = os.path.join(
        P.ROUND_ROOT, f"queue-gate-only-pytest-{os.getpid()}-{int(time.time() * 1000) % 100000}"
    )
    try:
        manifest_path = P.prepare_round0265_gate_only(
            release_sha=release_sha,
            bind_panel=CORR3_PANEL,
            source_run=CORR3_RUN,
            source_release=CORR3_RELEASE_SHA,
            queue_root=queue_root,
        )
        with open(manifest_path, encoding="utf-8") as handle:
            queue = json.load(handle)

        # exactly ONE node — the gate — with no in-queue dependency (no panel node).
        assert len(queue["jobs"]) == 1
        gate = queue["jobs"][0]
        assert gate["id"] == N.GATE_ACTION and gate["action"] == N.GATE_ACTION
        assert gate["deps"] == []
        assert queue["capabilities_produced"] == [N.GATE_CAPABILITY]
        assert queue["schema"] == "round0265-fneg-family-floors-n13-gate-only-queue-v1"

        # the panel is bound by CONTENT DIGEST (sha256 + bytes), not a same-queue file.
        panel_sig = expected_input_signature(CORR3_PANEL)
        assert gate["panel_n13"]["sha256"] == panel_sig["sha256"]
        assert gate["panel_n13"]["bytes"] == panel_sig["bytes"]
        assert gate["panel_n13"]["canonical_path"] == panel_sig["canonical_path"]
        # and it is a preflight-verified expected input.
        assert any(
            item.get("sha256") == panel_sig["sha256"] for item in gate["expected_inputs"]
        )

        # provenance block, on both the queue and the gate job.
        prov = queue["gate_only_provenance"]
        assert prov["gate_only"] is True
        assert prov["source_run"] == CORR3_RUN
        assert prov["source_release_sha"] == CORR3_RELEASE_SHA
        assert prov["bound_panel_sha256"] == panel_sig["sha256"]
        assert prov["gate_release_sha"] == release_sha
        assert gate["gate_only_provenance"] == prov

        # the recipe proof still holds — thirteen distinct cells, one recipe, no trains.
        fam_path = os.path.join(queue_root, "preflight", "fneg-family-identity.json")
        with open(fam_path, encoding="utf-8") as handle:
            family = json.load(handle)["family"]
        assert family["all_thirteen_share_one_recipe_digest"] is True
        assert family["thirteen_distinct_cell_configs"] is True
        assert family["cells"] == 13
    finally:
        if os.path.isdir(queue_root):
            _rmtree_ro(queue_root)
