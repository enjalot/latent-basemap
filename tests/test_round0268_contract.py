"""Contract tests for R0268 — the 100M ×2 host-int8 FLAGSHIP of the fneg recipe.

CPU-only (CUDA hidden). Exercises the 100M ×2 recipe invariant (delegating R0265's whole
recipe proof on a probe, then the ×2 dose + host-int8 delta), its refusal plants (incl. the
×4-not-×2 dose, fp32 residency, weighted-on), the import-closure seal + controls, the dose
×2 derivation (horizon 8,327,508 = 2·successful_updates_for_edges(2,511,103,254)), the
seed-mean gate (criterion-1 band arithmetic, per-seed backstops, no-straddle), the FFR
reserve-projection disc (100,000 = int(ROWS·0.001)), the PRE-SEALED int8 FULL-FILE load, the
DESCRIPTIVE purity + INVERTED lineage check (100M-prefix != R0216-c3), and the ANALYTIC
HOST_RSS limit (115.0) present + emitted into the train receipt. The constants-discipline
guard mutates each SEALED artifact on disk and asserts the gate's band/floor/σ_fam/P1-edge
tracks the file — a typed literal would not. No training and no GPU. FRESH round: no salvage,
no bind, no gate-only re-seal.
"""
import copy
import json
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

from basemap import round0268_int8_treatment as T
from basemap import round0265_fneg_treatment as R0265
from basemap import round0266_int8_treatment as R0266
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import ordered_array_sha256
from basemap.round0217_minilm_2m_seed_family import successful_updates_for_edges
import experiments.round0268_nodes as N
import experiments.prepare_round0268_queue as P


FLOORS = P.R0265_FLOORS
PANEL = P.R0265_PANEL
P1 = P.P1_ASYMPTOTE

SCRATCH = os.environ.get(
    "R0268_SCRATCH",
    "/tmp/claude-1000/-home-enjalot-code/44761e3d-6bf2-4b87-bc1a-0a2230694374/scratchpad",
)


def _honest(seed=T.CANONICAL_SEED):
    return T._honest_100m_config(seed)


# --------------------------------------------------------------------------- #
# 1. the 100M ×2 recipe invariant: R0265 proof delegated + ×2 dose + int8 delta
# --------------------------------------------------------------------------- #


def test_honest_recipe_passes_and_carries_x2_hostint8():
    cfg = _honest()
    recipe = T.assert_registered_100m_int8_recipe(cfg)
    assert recipe["dose_multiplier"] == 2
    assert recipe["rows"] == 100_000_000
    assert recipe["x_residency"] == "host_int8"
    assert recipe["expected_pipeline_stamp_x_residency"] == "host_int8"
    # the whole R0265 recipe is proved by delegation on the probe.
    assert recipe["low_dim_kernel"] == "umap"
    assert recipe["a"] == 1.9328 and recipe["b"] == 0.7905
    assert recipe["fneg_weight"] == 1.0
    assert cfg["execution"]["x_residency"] == "host_int8"
    assert cfg["execution"]["required_pipeline"] == "host_int8"
    assert cfg["execution"]["expected_pipeline_stamp"]["pipeline"] == "host_int8"


def test_dose_is_x2_of_the_100m_base_and_equals_8327508():
    cfg = _honest()
    base = successful_updates_for_edges(T.SEALED_DIRECTED_EDGES)
    assert base == 4_163_754
    horizon = cfg["optimizer"]["successful_positive_lr_updates"]
    assert horizon == 2 * base == 8_327_508
    recipe = T.assert_registered_100m_int8_recipe(cfg)
    assert recipe["successful_positive_lr_updates"] == 8_327_508
    assert recipe["base_horizon"] == 4_163_754
    # published cross-check anchors + the module-level derived constants.
    assert T.BASE_HORIZON == 4_163_754 and T.HORIZON == 8_327_508
    # the 100M edge count is NOT exactly 2x the 50M count (the derivation-pin correction).
    assert T.SEALED_DIRECTED_EDGES != 2 * 1_255_091_326
    assert 2 * successful_updates_for_edges(1_255_091_326) == 4_162_228  # 50M's naive 2x
    assert T.HORIZON == 8_327_508 != 4_162_228


def test_validate_dose_x2_enforces_the_rule():
    d = T.validate_dose_x2(updates=8_327_508, edge_count=T.SEALED_DIRECTED_EDGES)
    assert d["successful_updates"] == 8_327_508
    assert d["base_successful_updates"] == 4_163_754
    assert d["dose_multiplier"] == 2
    with pytest.raises(T.Round0268RecipeError):
        T.validate_dose_x2(updates=8_324_456, edge_count=T.SEALED_DIRECTED_EDGES)  # 50M's x2


def test_wrong_dose_x4_is_refused():
    cfg = _honest()
    x4 = R0265.validate_fneg_dose(
        updates=R0265.DOSE_MULTIPLIER * successful_updates_for_edges(T.SEALED_DIRECTED_EDGES),
        edge_count=T.SEALED_DIRECTED_EDGES,
    )
    cfg["optimizer"]["successful_positive_lr_updates"] = x4["successful_updates"]
    cfg["execution"]["target_positive_draws_per_edge"] = x4["target_positive_draws_per_edge"]
    cfg["dose_registration"] = x4
    with pytest.raises(T.Round0268RecipeError):
        T.assert_registered_100m_int8_recipe(cfg)


def test_x_residency_device_fp16_is_refused():
    cfg = _honest()
    cfg["execution"]["x_residency"] = "device_fp16"
    with pytest.raises(T.Round0268RecipeError):
        T.assert_registered_100m_int8_recipe(cfg)


def test_weighted_sampling_on_is_refused_through_delegated_proof():
    cfg = _honest()
    cfg["optimizer"]["weighted_edge_sampling"] = True
    with pytest.raises(R0265.Round0265RecipeError):
        T.assert_registered_100m_int8_recipe(cfg)


def test_fneg_off_is_refused_through_delegated_proof():
    cfg = _honest()
    cfg["optimizer"]["fneg_weight"] = 0.0
    with pytest.raises(R0265.Round0265RecipeError):
        T.assert_registered_100m_int8_recipe(cfg)


def test_recipe_refusal_controls_all_fire():
    controls = T.recipe_refusal_controls()
    assert controls["planted"] == 6
    assert controls["every_planted_defect_was_refused"] is True
    assert controls["the_honest_recipe_still_passes"] is True
    assert {c["control"] for c in controls["controls"]} == {
        "wrong_dose_x4", "x_residency_device_fp16", "required_pipeline_device",
        "stamp_x_residency_device_fp16", "base_recipe_weighted_sampling_on",
        "base_recipe_fneg_off",
    }
    for c in controls["controls"]:
        assert c["shipped_predicate_refused"] is True


def test_three_seeds_share_one_masked_invariant():
    invs = {T.fneg_seed_invariant_sha256(_honest(s)) for s in T.SEEDS}
    assert len(invs) == 1
    fulls = {json.dumps(_honest(s), sort_keys=True) for s in T.SEEDS}
    assert len(fulls) == 3


def test_only_the_three_registered_seeds():
    with pytest.raises(T.Round0268RecipeError):
        T.int8_100m_train_config(
            graph_signature={"canonical_path": "/x", "sha256": "b" * 64},
            graph_manifest_signature={"canonical_path": "/x", "sha256": "c" * 64},
            substrate_signature={"canonical_path": "/x", "sha256": "a" * 64},
            graph_edges=T.SEALED_DIRECTED_EDGES,
            rows=T.ROWS,
            seed=45,
        )


def test_dose_multiplier_and_seeds_and_scale():
    assert T.DOSE_MULTIPLIER == 2 and N.DOSE_MULTIPLIER == 2
    assert T.SEEDS == (42, 43, 44)
    assert T.ROWS == 100_000_000
    assert T.SEALED_DIRECTED_EDGES == 2_511_103_254


# --------------------------------------------------------------------------- #
# 2. the import-closure seal + controls (R0266's closure + the round0268 module)
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


def test_import_closure_includes_round0268_and_matches_runtime():
    inner = _sealed_closure_body()
    observed = T.runtime_closure_hashes(T.TRAIN_CLOSURE_MODULES)
    verdict = T.assert_runtime_closure_matches_seal(sealed=inner, observed=observed)
    assert verdict["every_module_ran_the_sealed_bytes"] is True
    assert "basemap.round0268_int8_treatment" in verdict["modules"]
    assert "basemap.round0266_int8_treatment" in verdict["modules"]
    assert "basemap.pumap.parametric_umap.core" in verdict["modules"]


def test_import_closure_controls_all_refuse():
    inner = _sealed_closure_body()
    observed = T.runtime_closure_hashes(T.TRAIN_CLOSURE_MODULES)
    controls = T.treatment_closure_controls(sealed=inner, observed=observed)
    assert controls["every_planted_defect_was_refused"] is True
    assert controls["the_honest_closure_still_passes"] is True
    assert controls["planted"] == 5


# --------------------------------------------------------------------------- #
# 3. the seed-mean gate: criterion-1 band arithmetic + per-seed backstops
# --------------------------------------------------------------------------- #


def test_criterion_1_band_arithmetic():
    c1 = N.score_collapse_seed_mean(
        seed_collapse={"42": 0.90, "43": 0.95, "44": 1.00},
        p1_lower=0.930, p1_upper=0.985, sigma_fam_collapse=0.06, z=1.96, n=3,
    )
    assert c1["seed_mean"] == pytest.approx(0.95)
    allowance = 1.96 * 0.06 / math.sqrt(3)
    assert c1["seed_noise_allowance"] == pytest.approx(allowance)
    assert c1["widened_band"][0] == pytest.approx(0.930 - allowance)
    assert c1["widened_band"][1] == pytest.approx(0.985 + allowance)
    assert c1["passes"] is True


def test_criterion_1_fails_a_seed_mean_far_outside():
    c1 = N.score_collapse_seed_mean(
        seed_collapse={"42": 2.0, "43": 2.0, "44": 2.0},
        p1_lower=0.930, p1_upper=0.985, sigma_fam_collapse=0.06,
    )
    assert c1["seed_mean"] == pytest.approx(2.0)
    assert c1["passes"] is False


_SYNTH_BACKSTOPS = {
    "heldout_ffr_floor": 0.39,
    "collapse_floor": 0.81,
    "fog_ceiling": 0.41,
    "k1024_floor": 0.88,
    "k256_band": {"ratio_lower": 1.05, "ratio_upper": 1.13,
                  "log_centre": 0.083, "log_scale": 0.0073},
}


def _seed_metrics(collapse=0.97, fog=0.06, ffr=0.45, k256=1.09, k1024=0.92, degenerate=False):
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


def test_backstops_pass_a_healthy_three_seed_family():
    mt = {str(s): _seed_metrics() for s in T.SEEDS}
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    assert bs["every_seed_clears_every_backstop"] is True
    assert bs["any_gate_straddles"] is False
    assert bs["any_fog_near_ceiling_escalation"] is False
    # the cell ids are the 100M ids, not 50M's.
    assert all("100m" in row["cell_id"] for row in bs["cells"])


def test_backstops_no_straddle_flags_a_split_collapse_gate():
    mt = {str(s): _seed_metrics() for s in T.SEEDS}
    mt["44"] = _seed_metrics(collapse=0.5)  # below the 0.81 floor
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    assert bs["every_seed_clears_every_backstop"] is False
    assert "collapse" in bs["straddled_gates"]
    assert bs["any_gate_straddles"] is True


def test_backstops_fog_near_ceiling_escalates_even_if_under():
    mt = {str(s): _seed_metrics() for s in T.SEEDS}
    mt["43"] = _seed_metrics(fog=0.40)  # under 0.41 but within 1·σ_fam,fog (0.02)
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    assert bs["any_fog_near_ceiling_escalation"] is True


def test_backstops_fog_over_ceiling_fails():
    mt = {str(s): _seed_metrics() for s in T.SEEDS}
    mt["42"] = _seed_metrics(fog=0.9)
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    assert bs["every_seed_clears_every_backstop"] is False
    assert "fog" in bs["straddled_gates"]


def test_gate_metrics_are_the_three_and_purity_is_descriptive():
    assert set(N.GATE_METRICS) == {"heldout_ffr", "collapse", "fog"}
    assert set(N.DESCRIPTIVE_PURITY_METRICS) == {
        "purity_fidelity_k256", "purity_fidelity_k1024",
    }
    assert set(N.GATE_METRICS).isdisjoint(N.DESCRIPTIVE_PURITY_METRICS)


def test_backstops_record_purity_descriptively_and_never_gate_on_it():
    mt = {str(s): _seed_metrics() for s in T.SEEDS}
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    for row in bs["cells"]:
        assert set(row["metrics"]) == {"heldout_ffr", "collapse", "fog"}
        assert row["descriptive_purity"]["gated"] is False
        assert row["descriptive_purity"]["descriptive"] is True
    assert set(bs["straddled_gates"]).issubset(set(N.GATE_METRICS))
    assert bs["descriptive_purity_bands_recorded"]["gated"] is False


def test_gate_verdict_ignores_failing_purity():
    """Healthy collapse/fog/FFR but catastrophic purity -> the go/no-go is unchanged (PASS)."""
    mt = {str(s): _seed_metrics(collapse=0.97, fog=0.06, ffr=0.45, k256=0.01, k1024=0.01)
          for s in T.SEEDS}
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    for row in bs["cells"]:
        assert row["descriptive_purity"]["purity_fidelity_k256"]["passes"] is False
        assert row["descriptive_purity"]["purity_fidelity_k1024"]["passes"] is False
    assert bs["every_seed_clears_every_backstop"] is True
    assert bs["any_gate_straddles"] is False
    c1 = N.score_collapse_seed_mean(
        seed_collapse={s: mt[s]["collapse"] for s in mt},
        p1_lower=0.930, p1_upper=0.985, sigma_fam_collapse=0.06)
    passes = bool(c1["passes"] and bs["every_seed_clears_every_backstop"]
                  and not bs["any_gate_straddles"])
    verdict = ("100M_PASS" if passes and not bs["any_fog_near_ceiling_escalation"]
               else "100M_FAIL_OR_AMBIGUOUS")
    assert passes is True and verdict == "100M_PASS"


# --------------------------------------------------------------------------- #
# 4. constants-discipline: every band/floor/σ_fam/P1-edge is READ from a SEALED input
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not os.path.exists(PANEL), reason="R0265 sealed panel absent")
def test_sigma_fam_recomputed_from_the_sealed_panel():
    got = N.sigma_fam_from_panel(PANEL)
    raw = json.load(open(PANEL))
    table = raw["panel_metric_table"]
    seeds = sorted(int(k) for k in table)
    collapse = [float(table[str(s)]["collapse"]) for s in seeds]
    fog = [float(table[str(s)]["fog"]) for s in seeds]
    assert got["sigma_fam_collapse"] == pytest.approx(N._madn(collapse))
    assert got["sigma_fam_fog"] == pytest.approx(N._madn(fog))
    assert got["n_family_cells"] == 13


@pytest.mark.skipif(not os.path.exists(PANEL), reason="R0265 sealed panel absent")
def test_constants_discipline_sigma_fam_tracks_a_mutated_panel():
    base = N.sigma_fam_from_panel(PANEL)["sigma_fam_collapse"]
    sealed = prompt_contract.read_sealed(PANEL, label="panel")
    body = {k: v for k, v in sealed.items() if k != "identity_sha256"}
    body = copy.deepcopy(body)
    body["panel_metric_table"]["54"]["collapse"] = 99.0
    resealed = prompt_contract.seal(body)
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = os.path.join(SCRATCH, "r0268_panel_mutated.json")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(resealed, handle)
    try:
        grown = N.sigma_fam_from_panel(tmp)["sigma_fam_collapse"]
        assert grown > base
    finally:
        os.remove(tmp)


@pytest.mark.skipif(not os.path.exists(FLOORS), reason="R0265 sealed floors absent")
def test_constants_discipline_backstop_tracks_a_mutated_floors_artifact():
    import experiments.round0266_nodes as R0266N

    original = R0266N.read_family_bands(FLOORS)["bands"]["collapse_floor"]
    mutated_value = original + 0.5
    sealed = prompt_contract.read_sealed(FLOORS, label="floors")
    body = {k: v for k, v in sealed.items() if k != "identity_sha256"}
    body = copy.deepcopy(body)
    body["registered_criteria"]["collapse"]["floor"] = mutated_value
    resealed = prompt_contract.seal(body)
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = os.path.join(SCRATCH, "r0268_floors_mutated.json")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(resealed, handle)
    try:
        got = R0266N.read_family_bands(tmp)["bands"]["collapse_floor"]
        assert got == mutated_value and got != original
    finally:
        os.remove(tmp)


@pytest.mark.skipif(not os.path.exists(P1), reason="P1 analysis-v2 result absent")
def test_p1_band_read_from_the_sealed_analysis_v2_result():
    got = N.read_p1_x2_asymptote_band(P1)
    raw = json.load(open(P1))
    lo, hi = raw["bands"]["yinf_x2"]
    assert got["p1_lower"] == float(lo) and got["p1_upper"] == float(hi)
    assert got["verdict"] == "GO"
    # the 100M band is the SAME as 50M's (λ=37 saturated) -> the same [0.930, 0.985].
    assert round(got["p1_lower"], 3) == 0.930 and round(got["p1_upper"], 3) == 0.985


@pytest.mark.skipif(not os.path.exists(P1), reason="P1 analysis-v2 result absent")
def test_constants_discipline_p1_band_tracks_a_mutated_result():
    original = N.read_p1_x2_asymptote_band(P1)["p1_upper"]
    raw = copy.deepcopy(json.load(open(P1)))
    raw["bands"]["yinf_x2"] = [0.5, 0.6]
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = os.path.join(SCRATCH, "r0268_p1_mutated.json")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(raw, handle)
    try:
        got = N.read_p1_x2_asymptote_band(tmp)
        assert got["p1_lower"] == 0.5 and got["p1_upper"] == 0.6 and got["p1_upper"] != original
    finally:
        os.remove(tmp)


@pytest.mark.skipif(
    not (os.path.exists(FLOORS) and os.path.exists(PANEL) and os.path.exists(P1)),
    reason="sealed inputs absent",
)
def test_gate_end_to_end_from_sealed_inputs():
    import experiments.round0266_nodes as R0266N

    sigma = N.sigma_fam_from_panel(PANEL)
    p1 = N.read_p1_x2_asymptote_band(P1)
    backstops = R0266N.read_family_bands(FLOORS)["bands"]
    c1 = N.score_collapse_seed_mean(
        seed_collapse={"42": 0.96, "43": 0.98, "44": 0.99},
        p1_lower=p1["p1_lower"], p1_upper=p1["p1_upper"],
        sigma_fam_collapse=sigma["sigma_fam_collapse"],
    )
    assert c1["passes"] is True
    mt = {str(s): _seed_metrics(collapse=0.97, fog=0.06, ffr=0.45, k256=1.09, k1024=0.92)
          for s in T.SEEDS}
    bs = N.score_per_seed_backstops(
        metric_table=mt, backstops=backstops, sigma_fam_fog=sigma["sigma_fam_fog"]
    )
    assert bs["every_seed_clears_every_backstop"] is True
    assert bs["any_gate_straddles"] is False
    # σ_fam is the sealed family value, not a literal (universe-independent R0265 panel).
    assert sigma["sigma_fam_collapse"] == pytest.approx(0.057774, abs=1e-6)


# --------------------------------------------------------------------------- #
# 4b. the PRE-SEALED int8 FULL-FILE load: the FULL-FILE LAW + whole-file digest
#     binding, the loader's verification, the core hook selecting a pre-built
#     HostInt8ArrayDataset without re-encoding, and the file-backed / no-large-copy
#     contract. CPU-only, small synthetic int8 mmaps (no GPU). No prefix slice.
# --------------------------------------------------------------------------- #


def _write_synthetic_int8_full(tmp_path, rows=4, dim=384):
    i8 = (((np.arange(rows * dim) % 255) - 127).astype(np.int8)).reshape(rows, dim)
    sc = np.full(rows, 0.5, dtype=np.float16)
    i8_path = str(tmp_path / "substrate.i8")
    sc_path = str(tmp_path / "substrate-scales.f16")
    i8.tofile(i8_path)
    sc.tofile(sc_path)
    digests = T.int8_full_digests(i8_path, sc_path, rows=rows, dimension=dim)
    manifest = {
        "schema": T.INT8_SUBSTRATE_SCHEMA,
        "capability": T.INT8_SUBSTRATE_CAPABILITY,
        "round_id": T.ROUND_ID,
        "rows": rows,
        "dimension": dim,
        "x_residency": T.X_RESIDENCY,
        "full_file_law": {
            "law": "full-file-of-100m-int8-substrate",
            "parent_artifact": T.R0262_INT8_CAPABILITY,
            "parent_round": T.R0262_ROUND_ID,
            "i8_path": i8_path,
            "scales_path": sc_path,
            "rows": rows,
            "offset": 0,
            "dimension": dim,
            "i8_sha256": digests["i8_sha256"],
            "scales_sha256": digests["scales_sha256"],
        },
    }
    return i8, sc, manifest, i8_path, sc_path


def test_full_file_law_records_the_pinned_whole_file_digests():
    law = T.full_file_law_block()
    assert law["rows"] == 100_000_000 and law["offset"] == 0 and law["dimension"] == 384
    assert law["parent_artifact"] == "minilm-mixed-100m-int8-v1"
    assert law["i8_bytes"] == 100_000_000 * 384
    assert law["scales_bytes"] == 100_000_000 * 2
    assert law["i8_sha256"] == T.FULL_I8_SHA256
    assert law["scales_sha256"] == T.FULL_SCALES_SHA256
    body = T.int8_full_substrate_manifest_body(release_sha="0" * 40)
    assert body["schema"] == T.INT8_SUBSTRATE_SCHEMA
    assert body["capability"] == T.INT8_SUBSTRATE_CAPABILITY
    assert body["rows"] == 100_000_000 and body["dimension"] == 384
    assert body["x_residency"] == "host_int8"
    assert body["full_file_law"]["i8_sha256"] == law["i8_sha256"]
    # NO prefix slice — the whole file is the substrate (R0267 sliced; R0268 loads whole).
    assert body["full_file_law"]["law"] == "full-file-of-100m-int8-substrate"


def test_loader_verifies_full_file_digests_and_raises_on_mismatch(tmp_path):
    i8, sc, manifest, i8_path, sc_path = _write_synthetic_int8_full(tmp_path)
    got_i8, got_sc, receipt = N._load_verified_int8_full(manifest)
    assert got_i8.shape == (4, 384) and got_sc.shape == (4,)
    np.testing.assert_array_equal(np.asarray(got_i8), i8)
    np.testing.assert_array_equal(np.asarray(got_sc), sc)
    assert receipt["verified_against_sealed_manifest"] is True
    assert receipt["re_encoded_at_train_time"] is False
    assert receipt["offset"] == 0
    assert receipt["load_mode"] == "pre_sealed_file_backed_full_file"
    bad_i8 = copy.deepcopy(manifest)
    bad_i8["full_file_law"]["i8_sha256"] = "0" * 64
    with pytest.raises(N.Round0268NodeError):
        N._load_verified_int8_full(bad_i8)
    bad_sc = copy.deepcopy(manifest)
    bad_sc["full_file_law"]["scales_sha256"] = "0" * 64
    with pytest.raises(N.Round0268NodeError):
        N._load_verified_int8_full(bad_sc)
    bad_off = copy.deepcopy(manifest)
    bad_off["full_file_law"]["offset"] = 1
    with pytest.raises(N.Round0268NodeError):
        N._load_verified_int8_full(bad_off)


def test_build_hostint8_dataset_from_full_does_not_reencode(tmp_path, monkeypatch):
    from basemap.pumap.parametric_umap.datasets import edge_list_dataset as E

    calls = {"n": 0}
    orig = E.quantize_int8_rows

    def spy(block):
        calls["n"] += 1
        return orig(block)

    monkeypatch.setattr(E, "quantize_int8_rows", spy)
    i8, sc, manifest, i8_path, sc_path = _write_synthetic_int8_full(tmp_path)
    ds, receipt = N.build_hostint8_dataset_from_full(manifest, "cpu")
    assert ds.host_int8_dataset is True
    assert ds.shape == (4, 384)
    assert calls["n"] == 0


def test_core_hook_uses_prebuilt_hostint8_dataset_without_reencode(tmp_path, monkeypatch):
    from basemap.pumap.parametric_umap.datasets import edge_list_dataset as E
    from basemap.pumap.parametric_umap import ParametricUMAP

    calls = {"n": 0}
    orig = E.quantize_int8_rows

    def spy(block):
        calls["n"] += 1
        return orig(block)

    monkeypatch.setattr(E, "quantize_int8_rows", spy)
    i8, sc, manifest, i8_path, sc_path = _write_synthetic_int8_full(tmp_path, rows=4, dim=384)
    ds, _receipt = N.build_hostint8_dataset_from_full(manifest, "cpu")

    n = 4
    src = np.array([0, 1, 2, 3, 0, 1], dtype=np.int32)
    dst = np.array([1, 2, 3, 0, 2, 3], dtype=np.int32)
    npz = str(tmp_path / "edges.npz")
    np.savez(npz, sources=src, targets=dst, n_nodes=np.int64(n))

    model = ParametricUMAP(
        device="cpu",
        x_residency="host_int8",
        require_graph_manifest=False,
        positive_target_mode="binary",
        weighted_edge_sampling=False,
        batch_size=4,
        pos_ratio=0.5,
    )
    dataset, loader, n_pos = model._prepare_edge_list_training(ds, npz, n, False, 0)
    assert dataset is ds
    assert model._X_dev is ds
    assert model._pipeline_info["x_residency"] == "host_int8"
    assert model._pipeline_info["pipeline"] == "host_int8"
    assert calls["n"] == 0


def test_hostint8_dataset_stays_file_backed_no_large_copy(tmp_path):
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        HostInt8ArrayDataset,
    )

    rows, dim = 8, 384
    i8 = (((np.arange(rows * dim) % 255) - 127).astype(np.int8)).reshape(rows, dim)
    sc = np.full(rows, 0.5, dtype=np.float16)
    i8_path = str(tmp_path / "s.i8")
    sc_path = str(tmp_path / "s.f16")
    i8.tofile(i8_path)
    sc.tofile(sc_path)
    mm_i8 = np.memmap(i8_path, dtype=np.int8, mode="r").reshape(-1, dim)
    mm_sc = np.memmap(sc_path, dtype=np.float16, mode="r")
    assert mm_i8.flags["C_CONTIGUOUS"] and mm_sc.flags["C_CONTIGUOUS"]
    ds = HostInt8ArrayDataset(None, "cpu", encoded=mm_i8, scales=mm_sc)
    assert np.shares_memory(ds._i8.numpy(), mm_i8)
    assert np.shares_memory(ds._scales.numpy(), mm_sc)


# --------------------------------------------------------------------------- #
# 5. dispatch + scope registration, and the action guard
# --------------------------------------------------------------------------- #


def test_round0268_is_registered_in_scope_modules():
    from basemap.round0254_dispatch import (
        SCOPE_MODULES,
        assert_derived_entries_install,
        dispatch_census,
    )
    assert "experiments.round0268_nodes" in SCOPE_MODULES
    guard = assert_derived_entries_install(SCOPE_MODULES, dispatch_census())
    assert guard["audit"]["every_entry_installs_effectively"] is True


def test_run_job_rejects_an_unknown_action():
    with pytest.raises(N.Round0268NodeError):
        N.run_job({"manifest": {"round_id": "0268"}}, {"action": "not-a-real-action"})


# --------------------------------------------------------------------------- #
# 6. treatment-vs-execution: the masked-config / treatment invariant digest is
#    INVARIANT to a change in HOST_RSS_LIMIT_GIB (and the other resource constants),
#    so the analytic RSS limit is an EXECUTION-resource field, excluded from the
#    config and the treatment digest.
# --------------------------------------------------------------------------- #


def test_host_rss_limit_is_the_analytic_115():
    assert N.HOST_RSS_LIMIT_GIB == 115.0
    # PANEL RSS limit refined from the throwaway-map dry-run (plan §3; delegate option-A
    # approval 2026-08-17): 120.0 = measured ru_maxrss peak 115.46 GiB + ~4.5 GiB margin,
    # still 3.4 GiB under physical (123.4) so it fires on genuine RAM exhaustion. The train
    # HOST_RSS analytic limit (115) is unchanged; the 64 GiB anon guard is the real OOM tripwire.
    assert N.PANEL_RSS_LIMIT_GIB == 120.0
    # the analytic basis is a number with its derivation, emitted into the receipt.
    basis = N.HOST_RSS_ANALYTIC_BASIS
    assert basis["limit_gib"] == 115.0
    assert basis["r0267_50m_measured_peak_rss_gib"] == 75.66
    assert basis["projected_100m_peak_rss_gib_approx"] == 104.0
    assert basis["margin_gib"] == 11.0
    assert basis["int8_x_bytes_100m"] == 38_400_000_000
    assert "scaled to 100M" in basis["method"]


def test_run_train_emits_the_analytic_rss_basis_into_the_receipt():
    import inspect

    src = inspect.getsource(N.run_train)
    # the receipt carries the limit AND its analytic basis (not "should be fine").
    assert '"host_rss_limit_gib": HOST_RSS_LIMIT_GIB' in src
    assert '"host_rss_limit_basis": HOST_RSS_ANALYTIC_BASIS' in src
    # and the RSS backstop is checked against the analytic limit.
    assert "peak_rss_gib > HOST_RSS_LIMIT_GIB" in src


def test_treatment_digest_excludes_execution_resource_fields(monkeypatch):
    cfg = _honest()
    digest = T.fneg_seed_invariant_sha256(cfg)
    blob = json.dumps(cfg)
    for token in ("115.0", "104.0", str(N.R0268_ANON_BUDGET_BYTES), str(N.DEVICE_BUDGET_BYTES)):
        assert token not in blob
    # changing the module's execution-resource constants leaves the treatment digest unchanged.
    monkeypatch.setattr(N, "HOST_RSS_LIMIT_GIB", 60.0)
    monkeypatch.setattr(N, "R0268_ANON_BUDGET_BYTES", 7 * (1 << 30))
    monkeypatch.setattr(N, "DEVICE_BUDGET_BYTES", 3 * (1 << 30))
    digest_at_60 = T.fneg_seed_invariant_sha256(_honest())
    monkeypatch.setattr(N, "HOST_RSS_LIMIT_GIB", 115.0)
    digest_at_115 = T.fneg_seed_invariant_sha256(_honest())
    assert digest_at_60 == digest_at_115 == digest


# --------------------------------------------------------------------------- #
# 7. purity DESCRIPTIVE + the INVERTED lineage check (100M-prefix != R0216-c3)
# --------------------------------------------------------------------------- #


def test_descriptive_purity_lineage_caveat_states_the_lineage_fact():
    caveat = N.DESCRIPTIVE_PURITY_LINEAGE_CAVEAT
    assert isinstance(caveat, str) and caveat
    low = caveat.lower()
    assert "descriptive" in low and "ungated" in low
    assert "no r0218 dependency" in low or "self-contained" in low
    assert "different build lineage" in low
    assert "r0216-c3" in low or "cb44d0a7" in low
    assert N.PREFIX_ROWS == 2_000_000


def test_lineage_check_is_inverted_non_match_and_purity_descriptive():
    """The 100M-prefix ordered hash != R0216-c3's sealed 2M reference -> descriptive; a
    MATCH (unexpected) raises to escalate to the owner."""
    src = np.arange(8 * 384, dtype=np.float32).reshape(8, 384)
    observed = ordered_array_sha256(src[:4])
    # a non-matching reference: the pre-registered expectation -> recorded, non-match.
    out = N.verify_hundred_m_prefix_lineage(src, "f" * 64, prefix_rows=4)
    assert out["matches_r0216_c3"] is False
    assert out["purity_is_descriptive"] is True
    assert out["expected"] == "non_match"
    assert out["observed_hundred_m_prefix_sha256"] == observed
    # a MATCHING reference (the 100M-prefix equals R0216-c3) is UNEXPECTED -> raises.
    with pytest.raises(N.Round0268NodeError):
        N.verify_hundred_m_prefix_lineage(src, observed, prefix_rows=4)


def test_run_panel_records_lineage_check_and_prefix_inline_descriptive_purity():
    import inspect

    src = inspect.getsource(N.run_panel)
    # collapse / fog / held-out FFR: score_one_map on the FULL 100M coordinates.
    assert "score_one_map(" in src
    assert "coordinates=coordinates," in src
    assert "probes_placed=placed," in src
    # the FLOOR-MATCHED FFR instrument at the N-scaled disc.
    assert "truth_top10=reserve_truth," in src
    assert "proj_model.transform(reserve_embeddings" in src
    assert "disc=reserve_disc," in src
    assert "reserve_disc = int(ROWS * 0.001)" in src
    # exactly ONE score_panel call — the descriptive prefix purity pass.
    assert src.count("score_panel(") == 1
    assert "source[:prefix_rows]" in src
    assert "coordinates[:prefix_rows]" in src
    assert "hiD_reference=None," in src
    assert "_build_prefix_purity_centroids(" in src
    # the lineage check is computed + recorded.
    assert "verify_hundred_m_prefix_lineage(" in src
    assert "_read_r0216_c3_reference(" in src
    assert '"lineage_check": lineage_check' in src
    # no >=8M scale_admission is carried anywhere in run_panel.
    assert "scale_admission=" not in src


def test_run_gate_records_descriptive_purity_and_lineage_and_gates_on_three_only():
    import inspect

    src = inspect.getsource(N.run_gate)
    assert '"descriptive_purity": descriptive_purity' in src
    assert "DESCRIPTIVE_PURITY_LINEAGE_CAVEAT" in src
    assert "purity_is_descriptive_not_gated" in src
    assert "lineage_check_is_non_match_and_descriptive" in src
    # the verdict is the criterion-1 + gated-backstops expression.
    assert "backstop_scoring[\"every_seed_clears_every_backstop\"]" in src
    assert "backstop_scoring[\"any_gate_straddles\"]" in src
    assert "100M_PASS" in src and "100M_FAIL_OR_AMBIGUOUS" in src


# --------------------------------------------------------------------------- #
# 8. the FFR reserve-projection instrument: disc = int(ROWS·0.001) = 100,000
# --------------------------------------------------------------------------- #

import experiments.round0265_nodes as R0265N


def test_run_panel_ffr_disc_is_n_scaled_100000_not_the_fixed_2000():
    """The R0268 held-out-FFR disc is int(ROWS * 0.001) = 100,000 at the 100M rung, NOT the
    fixed 2000 (trip 9) and NOT the 50M rung's 50,000."""
    assert int(T.ROWS * 0.001) == 100_000
    assert int(T.ROWS * 0.001) != 2000
    assert int(T.ROWS * 0.001) != 50_000
    # R0265's own FFR_DISC is the 2M value (0.1%·2M = 2000) — the rule, at its own N.
    assert R0265N.FFR_DISC == int(R0265N.ROWS * 0.001) == 2000


def test_score_one_map_forwards_disc_to_heldout_ffr():
    """score_one_map's `disc` kwarg threads to heldout_ffr_scores; a 100M caller passes its
    own N-scaled disc = 100,000. A cheap real-cKDTree fixture (50 coords)."""
    rng = np.random.RandomState(0)
    coords = rng.randn(50, 2).astype("float32")
    placed = coords[:4] + 1e-4
    truth = np.tile(np.arange(3, dtype=np.int64), (4, 1))
    out_default = R0265N.score_one_map(
        coordinates=coords, probes_placed=placed, truth_top10=truth,
        purity_ratios={"k256": 1.0, "k1024": 1.0},
    )
    assert out_default["heldout_ffr_detail"]["disc"] == R0265N.FFR_DISC == 2000
    out_disc = R0265N.score_one_map(
        coordinates=coords, probes_placed=placed, truth_top10=truth,
        purity_ratios={"k256": 1.0, "k1024": 1.0}, disc=7,
    )
    assert out_disc["heldout_ffr_detail"]["disc"] == 7
    assert out_default["collapse"] == out_disc["collapse"]
    assert out_default["fog"] == out_disc["fog"]


def test_run_panel_ffr_uses_out_of_substrate_reserve_projection():
    import inspect

    src = inspect.getsource(N.run_panel)
    assert "reserve_disc = int(ROWS * 0.001)" in src
    assert "proj_model.transform(reserve_embeddings" in src
    assert "truth_top10=reserve_truth," in src
    assert "disc=reserve_disc," in src
    assert '_bound_path(job, "heldout_reserve"' in src
    assert '_bound_path(job, "reserve_query_rows"' in src
    assert '_bound_path(job, "reserve_truth"' in src
    assert "reserve_all[reserve_query_rows]" in src


# --------------------------------------------------------------------------- #
# 9. prepare: the 100M path constants + the graph binding + the round-file gate
# --------------------------------------------------------------------------- #


def test_prepare_path_constants_target_the_100m_artifacts():
    assert P.R0238_SUBSTRATE_MANIFEST.endswith(
        "minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.json"
    )
    assert P.R0243_GRAPH_MANIFEST.endswith(
        "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1/fuzzy-graph.json"
    )
    assert P.R0238_RESERVE.endswith("reserve.f32.npy")
    assert P.R0238_RESERVE_QUERY_ROWS.endswith("reserve-query-rows.i64.npy")
    assert P.R0268_RESERVE_NEIGHBOUR_TRUTH.endswith(
        "round-0268/ffr/reserve-truth-100m/truth-top10.npy"
    )
    # the int8 full-file parent is R0262's WHOLE 100M substrate.
    assert T.R0262_I8_PATH.endswith("minilm-mixed-100m-int8-v1/substrate.i8")
    assert T.R0262_I8_BYTES == 38_400_000_000 and T.R0262_SCALES_BYTES == 200_000_000


@pytest.mark.skipif(not os.path.exists(P.R0243_GRAPH_MANIFEST), reason="R0243 graph manifest absent")
def test_prepare_graph_binding_reads_the_streamed_members():
    g = P._sealed_graph_binding()
    assert g["directed_edges"] == 2_511_103_254
    assert set(g["member_signatures"]) == {
        "edges_header", "edges_sources", "edges_targets", "edges_weights",
    }
    # the edge PATH is the artifact DIRECTORY (load_edge_arrays claims the streamed members).
    assert g["edges_dir"].endswith("minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1")
    assert g["graph_signature"]["canonical_path"] == g["edges_dir"]


def test_prepare_refuses_until_the_round_file_is_issued():
    # round-0268-*.md must be ISSUED before prepare runs; absent -> EXPECTED refusal.
    assert P.ROUND_FILE.endswith("round-0268-2026-08-17.md")
    if not os.path.exists(P.ROUND_FILE):
        with pytest.raises(RuntimeError):
            P._issued_round("0" * 40)


@pytest.mark.skipif(
    not os.path.exists(R0218_panel_path := P.R0218_PANEL), reason="R0218 panel absent"
)
def test_r0218_panel_carries_the_r0216_c3_2m_reference():
    """The lineage check reads R0216-c3's sealed 2M ordered reference from the R0218 panel's
    lineage.ordered_substrate_sha256 (constants-discipline: not a hardcoded literal)."""
    panel = prompt_contract.read_sealed(P.R0218_PANEL, label="R0218 panel")
    reference = str((panel.get("lineage") or {}).get("ordered_substrate_sha256") or "")
    assert len(reference) == 64
    # the pre-registered R0216-c3 reference the plan names (cb44d0a7…).
    assert reference.startswith("cb44d0a7")
