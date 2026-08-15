"""Contract tests for R0266 — the map-level host-int8 fidelity cell of the fneg recipe.

CPU-only (CUDA hidden). Exercises the int8 recipe invariant (delegating the whole R0265
recipe proof, then the single x_residency delta), its refusal plants, the import-closure
seal + controls, the config-delta identity (R0266 == R0265 seed-42 + one field), the CPU
`_build` int8 delta, and the two-criterion map-fidelity gate. The constants-discipline
guard mutates a SEALED artifact on disk and asserts the gate's bands/ranges track the
file — a typed literal would not. No training and no GPU: every number is synthetic or
read from an already-sealed on-disk artifact.
"""
import copy
import json
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

from basemap import round0266_int8_treatment as T
from basemap import round0265_fneg_treatment as R0265
from basemap import round0113_prompt_contrast as prompt_contract
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
import experiments.round0266_nodes as N
import experiments.prepare_round0266_queue as P


FLOORS = P.R0265_FLOORS
PANEL = P.R0265_PANEL

SCRATCH = os.environ.get(
    "R0266_SCRATCH",
    "/tmp/claude-1000/-home-enjalot-code/44761e3d-6bf2-4b87-bc1a-0a2230694374/scratchpad",
)


def _honest_config():
    config, _sha = T.int8_train_config(
        graph_signature=SEALED_GRAPH_SIGNATURE,
        graph_manifest_signature=SEALED_GRAPH_MANIFEST_SIGNATURE,
        substrate_signature=SEALED_SUBSTRATE_SIGNATURE,
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )
    return config


def _r0265_seed42_config():
    config, _sha = R0265.fneg_train_config(
        seed=R0265.CANONICAL_SEED,
        graph_signature=SEALED_GRAPH_SIGNATURE,
        graph_manifest_signature=SEALED_GRAPH_MANIFEST_SIGNATURE,
        substrate_signature=SEALED_SUBSTRATE_SIGNATURE,
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )
    return config


# --------------------------------------------------------------------------- #
# 1. the int8 recipe invariant: R0265 proof delegated + the x_residency delta
# --------------------------------------------------------------------------- #


def test_honest_int8_recipe_passes_and_carries_host_int8():
    cfg = _honest_config()
    recipe = T.assert_registered_int8_recipe(cfg)
    assert recipe["x_residency"] == "host_int8"
    assert recipe["expected_pipeline_stamp_x_residency"] == "host_int8"
    # the whole R0265 recipe is proved by delegation.
    assert recipe["low_dim_kernel"] == "umap"
    assert recipe["fneg_weight"] == 1.0
    assert recipe["successful_positive_lr_updates"] == 320652
    assert cfg["execution"]["x_residency"] == "host_int8"
    assert cfg["execution"]["expected_pipeline_stamp"]["x_residency"] == "host_int8"
    assert cfg["treatment"]["x_residency"] == "host_int8"


def test_int8_is_only_the_residency_and_routing_delta_over_r0265_seed42():
    """R0266's config is R0265's seed-42 config plus exactly the residency + routing
    delta: x_residency (+ its stamp mirror) and required_pipeline (+ its stamp mirror),
    all device -> host_int8. Nothing else differs."""
    int8 = _honest_config()
    fp32 = _r0265_seed42_config()

    # Normalise the identity fields that legitimately differ (round_id / schema /
    # treatment metadata) and the residency+routing delta; everything else identical.
    def _strip(cfg):
        c = copy.deepcopy(cfg)
        c.pop("round_id", None)
        c.pop("schema", None)
        c.pop("treatment", None)
        c["execution"].pop("x_residency", None)
        c["execution"].pop("required_pipeline", None)
        c["execution"]["expected_pipeline_stamp"].pop("x_residency", None)
        c["execution"]["expected_pipeline_stamp"].pop("pipeline", None)
        return c

    assert _strip(int8) == _strip(fp32)
    # The delta is present in int8 and device/absent in fp32.
    assert int8["execution"]["x_residency"] == "host_int8"
    assert int8["execution"]["required_pipeline"] == "host_int8"
    assert int8["execution"]["expected_pipeline_stamp"]["pipeline"] == "host_int8"
    assert "x_residency" not in fp32["execution"]
    assert fp32["execution"]["required_pipeline"] == "device"
    assert fp32["execution"]["expected_pipeline_stamp"]["x_residency"] == "device_fp16"
    assert fp32["execution"]["expected_pipeline_stamp"]["pipeline"] == "device"


def test_x_residency_device_fp16_is_refused():
    cfg = _honest_config()
    cfg["execution"]["x_residency"] = "device_fp16"
    with pytest.raises(T.Round0266RecipeError):
        T.assert_registered_int8_recipe(cfg)


def test_x_residency_auto_is_refused():
    cfg = _honest_config()
    cfg["execution"]["x_residency"] = "auto"
    with pytest.raises(T.Round0266RecipeError):
        T.assert_registered_int8_recipe(cfg)


def test_stamp_x_residency_still_device_fp16_is_refused():
    cfg = _honest_config()
    cfg["execution"]["expected_pipeline_stamp"]["x_residency"] = "device_fp16"
    with pytest.raises(T.Round0266RecipeError):
        T.assert_registered_int8_recipe(cfg)


def test_base_r0265_recipe_defect_is_refused_through_the_int8_guard():
    # fneg turned off is an R0265 recipe defect; the delegated proof must still fire.
    cfg = _honest_config()
    cfg["optimizer"]["fneg_weight"] = 0.0
    with pytest.raises(R0265.Round0265RecipeError):
        T.assert_registered_int8_recipe(cfg)
    # weighted sampling turned back on likewise.
    cfg2 = _honest_config()
    cfg2["optimizer"]["weighted_edge_sampling"] = True
    with pytest.raises(R0265.Round0265RecipeError):
        T.assert_registered_int8_recipe(cfg2)


def test_int8_recipe_refusal_controls_all_fire():
    controls = T.int8_recipe_refusal_controls()
    assert controls["planted"] == 7
    assert controls["every_planted_defect_was_refused"] is True
    assert controls["the_honest_recipe_still_passes"] is True
    assert {c["control"] for c in controls["controls"]} == {
        "x_residency_device_fp16", "x_residency_auto", "stamp_x_residency_device_fp16",
        "required_pipeline_device", "stamp_pipeline_device",
        "base_recipe_fneg_off", "base_recipe_weighted_sampling_on",
    }
    for c in controls["controls"]:
        assert c["shipped_predicate_refused"] is True


def test_int8_residency_paths_membership():
    assert T.INT8_RESIDENCY_PATHS == (
        ("execution", "x_residency"),
        ("execution", "expected_pipeline_stamp", "x_residency"),
        ("execution", "required_pipeline"),
        ("execution", "expected_pipeline_stamp", "pipeline"),
    )


def test_single_seed_only():
    with pytest.raises(T.Round0266RecipeError):
        T.int8_train_config(
            graph_signature=SEALED_GRAPH_SIGNATURE,
            graph_manifest_signature=SEALED_GRAPH_MANIFEST_SIGNATURE,
            substrate_signature=SEALED_SUBSTRATE_SIGNATURE,
            graph_edges=SEALED_DIRECTED_EDGES,
            rows=ROWS,
            seed=43,
        )


# --------------------------------------------------------------------------- #
# 2. the import-closure seal + controls (R0265's closure + the round0266 module)
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


def test_import_closure_includes_round0266_and_matches_runtime():
    inner = _sealed_closure_body()
    observed = T.runtime_closure_hashes(T.TRAIN_CLOSURE_MODULES)
    verdict = T.assert_runtime_closure_matches_seal(sealed=inner, observed=observed)
    assert verdict["every_module_ran_the_sealed_bytes"] is True
    assert "basemap.round0266_int8_treatment" in verdict["modules"]
    assert "basemap.pumap.parametric_umap.core" in verdict["modules"]


def test_import_closure_controls_all_refuse():
    inner = _sealed_closure_body()
    observed = T.runtime_closure_hashes(T.TRAIN_CLOSURE_MODULES)
    controls = T.treatment_closure_controls(sealed=inner, observed=observed)
    assert controls["every_planted_defect_was_refused"] is True
    assert controls["the_honest_closure_still_passes"] is True
    assert controls["planted"] == 5


def test_empty_closure_is_refused():
    with pytest.raises(R0265.Round0265TreatmentError):
        T.assert_runtime_closure_matches_seal(sealed={"files": {}}, observed={}, modules=())


# --------------------------------------------------------------------------- #
# 3. the CPU model build threads the int8 delta
# --------------------------------------------------------------------------- #


def test_built_model_carries_host_int8_and_routes_host_int8():
    cfg = _honest_config()
    model = N._build_int8_model(cfg)
    assert model.x_residency == "host_int8"
    assert model.weighted_edge_sampling is False
    # The host-int8 path stamps pipeline="host_int8", so required_input_pipeline must
    # fail-closed to "host_int8" (R0265's "device" pin would make core.fit refuse it).
    assert model.required_input_pipeline == "host_int8"
    assert model.gpu_resident_data == "auto"


# --------------------------------------------------------------------------- #
# 4. the two-criterion gate scoring (synthetic bands/ranges)
# --------------------------------------------------------------------------- #


_SYNTH_BANDS = {
    "heldout_ffr_floor": 0.39,
    "collapse_floor": 0.80,
    "fog_ceiling": 0.42,
    "k1024_floor": 0.85,
    "k256_band": {
        "ratio_lower": 1.05, "ratio_upper": 1.13,
        "log_centre": 0.083, "log_scale": 0.0073,
    },
}


def _int8(collapse=1.05, fog=0.33, ffr=0.42, k256=1.09, k1024=0.92, degenerate=False):
    return {
        "heldout_ffr": ffr,
        "purity_fidelity_k256": k256,
        "purity_fidelity_k1024": k1024,
        "collapse": collapse,
        "fog": 0.0 if degenerate else fog,
        "resolution_levels": 0 if degenerate else 14,
        "degenerate": bool(degenerate),
        "fog_detail": {"peak_bin_count": 99 if degenerate else 1400},
    }


def test_criterion_a_passes_a_healthy_int8_cell():
    a = N.score_criterion_a(int8_metrics=_int8(), bands=_SYNTH_BANDS)
    assert a["passes"] is True
    assert set(a["by_metric"]) == set(N.GATE_METRICS)


def test_criterion_a_fails_a_collapsed_cell():
    a = N.score_criterion_a(int8_metrics=_int8(collapse=0.10), bands=_SYNTH_BANDS)
    assert a["by_metric"]["collapse"]["passes"] is False
    assert a["passes"] is False


def test_criterion_a_fog_fails_closed_on_degeneracy():
    a = N.score_criterion_a(int8_metrics=_int8(degenerate=True), bands=_SYNTH_BANDS)
    assert a["by_metric"]["fog"]["verdict"] == N.VERDICT_NOT_MEASURABLE
    assert a["by_metric"]["fog"]["passes"] is False
    assert a["any_fog_not_measurable"] is True


def test_criterion_b_passes_within_family_range_and_fails_a_large_delta():
    fp32 = {"heldout_ffr": 0.42, "purity_fidelity_k256": 1.08,
            "purity_fidelity_k1024": 0.92, "collapse": 1.13, "fog": 0.366}
    ranges = {"heldout_ffr": 0.013, "purity_fidelity_k256": 0.024,
              "purity_fidelity_k1024": 0.046, "collapse": 0.247, "fog": 0.10}
    # int8 close to fp32 → within range.
    close = _int8(collapse=1.02, fog=0.318, ffr=0.417, k256=1.078, k1024=0.920)
    b = N.score_criterion_b(int8_metrics=close, fp32_metrics=fp32, ranges=ranges)
    assert b["passes"] is True
    # a big collapse delta (far beyond the family's 0.247 range) fails.
    far = _int8(collapse=1.13 + 5.0)
    bf = N.score_criterion_b(int8_metrics=far, fp32_metrics=fp32, ranges=ranges)
    assert bf["by_metric"]["collapse"]["passes"] is False
    assert bf["passes"] is False


# --------------------------------------------------------------------------- #
# 5. the gate reads bands/ranges from SEALED artifacts — the constants-discipline guard
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not os.path.exists(FLOORS), reason="R0265 sealed floors absent")
def test_criterion_a_bands_are_read_from_the_sealed_floors_artifact():
    got = N.read_family_bands(FLOORS)
    bands = got["bands"]
    # Cross-check against an INDEPENDENT read of the same file.
    raw = json.load(open(FLOORS))
    rc = raw["registered_criteria"]
    assert bands["collapse_floor"] == float(rc["collapse"]["floor"])
    assert bands["fog_ceiling"] == float(rc["fog"]["ceiling"])
    assert bands["heldout_ffr_floor"] == float(rc["heldout_ffr"]["floor"])
    assert bands["k1024_floor"] == float(rc["purity_fidelity_k1024"]["ratio_floor"])
    assert bands["k256_band"]["ratio_lower"] == float(rc["purity_fidelity_k256"]["ratio_lower"])
    assert bands["k256_band"]["ratio_upper"] == float(rc["purity_fidelity_k256"]["ratio_upper"])
    assert got["source_capability"] == N.R0265N.GATE_CAPABILITY


@pytest.mark.skipif(not os.path.exists(FLOORS), reason="R0265 sealed floors absent")
def test_constants_discipline_criterion_a_tracks_a_mutated_artifact():
    """CONSTANTS-DISCIPLINE GUARD: re-seal the floors with a changed collapse floor and
    assert `read_family_bands` returns the CHANGED value. A typed literal would not."""
    original = N.read_family_bands(FLOORS)["bands"]["collapse_floor"]
    mutated_value = original + 0.5

    sealed = prompt_contract.read_sealed(FLOORS, label="floors")
    body = {k: v for k, v in sealed.items() if k != "identity_sha256"}
    body = copy.deepcopy(body)
    body["registered_criteria"]["collapse"]["floor"] = mutated_value
    resealed = prompt_contract.seal(body)

    os.makedirs(SCRATCH, exist_ok=True)
    tmp = os.path.join(SCRATCH, "r0266_floors_mutated.json")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(resealed, handle)
    try:
        got = N.read_family_bands(tmp)["bands"]["collapse_floor"]
        # It tracks the file, not a constant: the mutated value is returned, and it is
        # NOT the original (so a hardcoded literal would have failed this assertion).
        assert got == mutated_value
        assert got != original
    finally:
        os.remove(tmp)


@pytest.mark.skipif(not os.path.exists(PANEL), reason="R0265 sealed panel absent")
def test_criterion_b_ranges_are_computed_from_the_sealed_panel():
    got = N.read_fp32_sibling_and_ranges(PANEL)
    raw = json.load(open(PANEL))
    table = raw["panel_metric_table"]
    seeds = sorted(int(k) for k in table)
    assert got["n_seeds"] == len(seeds) == 13
    # fp32 seed-42 sibling read straight from the table.
    assert got["fp32_seed42"]["collapse"] == float(table["42"]["collapse"])
    # ranges are computed max-min over the 13 seeds.
    for m in N.GATE_METRICS:
        vals = [float(table[str(s)][m]) for s in seeds]
        assert got["family_13seed_range"][m] == pytest.approx(max(vals) - min(vals))


@pytest.mark.skipif(not os.path.exists(PANEL), reason="R0265 sealed panel absent")
def test_constants_discipline_criterion_b_range_tracks_a_mutated_panel():
    """CONSTANTS-DISCIPLINE GUARD: widen the panel's collapse spread and assert the
    computed range grows. A typed range literal would not move."""
    base = N.read_fp32_sibling_and_ranges(PANEL)["family_13seed_range"]["collapse"]

    sealed = prompt_contract.read_sealed(PANEL, label="panel")
    body = {k: v for k, v in sealed.items() if k != "identity_sha256"}
    body = copy.deepcopy(body)
    # push seed-54's collapse far out → the max-min range must grow.
    body["panel_metric_table"]["54"]["collapse"] = 99.0
    resealed = prompt_contract.seal(body)

    os.makedirs(SCRATCH, exist_ok=True)
    tmp = os.path.join(SCRATCH, "r0266_panel_mutated.json")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(resealed, handle)
    try:
        grown = N.read_fp32_sibling_and_ranges(tmp)["family_13seed_range"]["collapse"]
        assert grown > base
        assert grown == pytest.approx(99.0 - min(
            float(body["panel_metric_table"][str(s)]["collapse"])
            for s in sorted(int(k) for k in body["panel_metric_table"])
        ))
    finally:
        os.remove(tmp)


@pytest.mark.skipif(
    not (os.path.exists(FLOORS) and os.path.exists(PANEL)),
    reason="R0265 sealed floors/panel absent",
)
def test_both_criteria_computed_from_sealed_inputs_end_to_end():
    """The sandbox host-int8 arm (collapse 1.0218 / fog 0.3182) scored against the REAL
    sealed R0265 bands and 13-seed range: both criteria pass, entirely from sealed inputs.
    """
    bands = N.read_family_bands(FLOORS)["bands"]
    sib = N.read_fp32_sibling_and_ranges(PANEL)
    int8 = _int8(collapse=1.0218, fog=0.3182, ffr=0.417, k256=1.078, k1024=0.920)
    a = N.score_criterion_a(int8_metrics=int8, bands=bands)
    b = N.score_criterion_b(
        int8_metrics=int8, fp32_metrics=sib["fp32_seed42"], ranges=sib["family_13seed_range"]
    )
    assert a["passes"] is True
    assert b["passes"] is True
    # every band used in (a) equals the sealed floor, not a literal.
    assert a["bands_used"]["collapse_floor"] == bands["collapse_floor"]


# --------------------------------------------------------------------------- #
# 6. dispatch + scope registration, and the action guard
# --------------------------------------------------------------------------- #


def test_round0266_is_registered_in_scope_modules():
    from basemap.round0254_dispatch import (
        SCOPE_MODULES,
        assert_derived_entries_install,
        dispatch_census,
    )
    assert "experiments.round0266_nodes" in SCOPE_MODULES
    guard = assert_derived_entries_install(SCOPE_MODULES, dispatch_census())
    assert guard["audit"]["every_entry_installs_effectively"] is True


def test_run_job_rejects_an_unknown_action():
    with pytest.raises(N.Round0266NodeError):
        N.run_job({"manifest": {"round_id": "0266"}}, {"action": "not-a-real-action"})


def test_gate_metrics_are_the_five_r0265_gated_metrics():
    assert set(N.GATE_METRICS) == {
        "heldout_ffr", "purity_fidelity_k256", "purity_fidelity_k1024", "collapse", "fog",
    }


def test_dose_horizon_is_x4_of_the_2m_base():
    cfg = _honest_config()
    base = successful_updates_for_edges(SEALED_DIRECTED_EDGES)
    assert cfg["optimizer"]["successful_positive_lr_updates"] == 4 * base == 320652
