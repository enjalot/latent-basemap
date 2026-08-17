"""Contract tests for R0267 — the 50M ×2 host-int8 staging rung of the fneg recipe.

CPU-only (CUDA hidden). Exercises the 50M ×2 recipe invariant (delegating R0265's whole
recipe proof on a probe, then the ×2 dose + host-int8 delta), its refusal plants (incl.
the ×4-not-×2 dose, fp32 residency, weighted-on), the import-closure seal + controls, the
dose ×2 derivation, and the seed-mean gate (criterion-1 band arithmetic, the per-seed
backstops, the no-straddle rule). The constants-discipline guard mutates each SEALED
artifact on disk and asserts the gate's band/floor/σ_fam/P1-edge tracks the file — a typed
literal would not. No training and no GPU.
"""
import copy
import json
import math
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

from basemap import round0267_int8_treatment as T
from basemap import round0265_fneg_treatment as R0265
from basemap import round0266_int8_treatment as R0266
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0217_minilm_2m_seed_family import successful_updates_for_edges
import experiments.round0267_nodes as N
import experiments.prepare_round0267_queue as P


FLOORS = P.R0265_FLOORS
PANEL = P.R0265_PANEL
P1 = P.P1_ASYMPTOTE

SCRATCH = os.environ.get(
    "R0267_SCRATCH",
    "/tmp/claude-1000/-home-enjalot-code/44761e3d-6bf2-4b87-bc1a-0a2230694374/scratchpad",
)


def _honest(seed=T.CANONICAL_SEED):
    return T._honest_50m_config(seed)


# --------------------------------------------------------------------------- #
# 1. the 50M ×2 recipe invariant: R0265 proof delegated + ×2 dose + int8 delta
# --------------------------------------------------------------------------- #


def test_honest_recipe_passes_and_carries_x2_hostint8():
    cfg = _honest()
    recipe = T.assert_registered_50m_int8_recipe(cfg)
    assert recipe["dose_multiplier"] == 2
    assert recipe["rows"] == 50_000_000
    assert recipe["x_residency"] == "host_int8"
    assert recipe["expected_pipeline_stamp_x_residency"] == "host_int8"
    # the whole R0265 recipe is proved by delegation on the probe.
    assert recipe["low_dim_kernel"] == "umap"
    assert recipe["a"] == 1.9328 and recipe["b"] == 0.7905
    assert recipe["fneg_weight"] == 1.0
    assert cfg["execution"]["x_residency"] == "host_int8"
    assert cfg["execution"]["required_pipeline"] == "host_int8"
    assert cfg["execution"]["expected_pipeline_stamp"]["pipeline"] == "host_int8"


def test_dose_is_x2_of_the_50m_base_and_equals_4162228():
    cfg = _honest()
    base = successful_updates_for_edges(T.SEALED_DIRECTED_EDGES)
    assert base == 2_081_114
    horizon = cfg["optimizer"]["successful_positive_lr_updates"]
    assert horizon == 2 * base == 4_162_228
    recipe = T.assert_registered_50m_int8_recipe(cfg)
    assert recipe["successful_positive_lr_updates"] == 4_162_228
    assert recipe["base_horizon"] == 2_081_114


def test_wrong_dose_x4_is_refused():
    cfg = _honest()
    x4 = R0265.validate_fneg_dose(
        updates=R0265.DOSE_MULTIPLIER * successful_updates_for_edges(T.SEALED_DIRECTED_EDGES),
        edge_count=T.SEALED_DIRECTED_EDGES,
    )
    cfg["optimizer"]["successful_positive_lr_updates"] = x4["successful_updates"]
    cfg["execution"]["target_positive_draws_per_edge"] = x4["target_positive_draws_per_edge"]
    cfg["dose_registration"] = x4
    with pytest.raises(T.Round0267RecipeError):
        T.assert_registered_50m_int8_recipe(cfg)


def test_x_residency_device_fp16_is_refused():
    cfg = _honest()
    cfg["execution"]["x_residency"] = "device_fp16"
    with pytest.raises(T.Round0267RecipeError):
        T.assert_registered_50m_int8_recipe(cfg)


def test_weighted_sampling_on_is_refused_through_delegated_proof():
    cfg = _honest()
    cfg["optimizer"]["weighted_edge_sampling"] = True
    with pytest.raises(R0265.Round0265RecipeError):
        T.assert_registered_50m_int8_recipe(cfg)


def test_fneg_off_is_refused_through_delegated_proof():
    cfg = _honest()
    cfg["optimizer"]["fneg_weight"] = 0.0
    with pytest.raises(R0265.Round0265RecipeError):
        T.assert_registered_50m_int8_recipe(cfg)


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
    # and the full configs differ (three real cells).
    fulls = {json.dumps(_honest(s), sort_keys=True) for s in T.SEEDS}
    assert len(fulls) == 3


def test_only_the_three_registered_seeds():
    with pytest.raises(T.Round0267RecipeError):
        T.int8_50m_train_config(
            graph_signature={"canonical_path": "/x", "sha256": "b" * 64},
            graph_manifest_signature={"canonical_path": "/x", "sha256": "c" * 64},
            substrate_signature={"canonical_path": "/x", "sha256": "a" * 64},
            graph_edges=T.SEALED_DIRECTED_EDGES,
            rows=T.ROWS,
            seed=45,
        )


# --------------------------------------------------------------------------- #
# 2. the import-closure seal + controls (R0266's closure + the round0267 module)
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


def test_import_closure_includes_round0267_and_matches_runtime():
    inner = _sealed_closure_body()
    observed = T.runtime_closure_hashes(T.TRAIN_CLOSURE_MODULES)
    verdict = T.assert_runtime_closure_matches_seal(sealed=inner, observed=observed)
    assert verdict["every_module_ran_the_sealed_bytes"] is True
    assert "basemap.round0267_int8_treatment" in verdict["modules"]
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


def test_backstops_no_straddle_flags_a_split_collapse_gate():
    mt = {str(s): _seed_metrics() for s in T.SEEDS}
    mt["44"] = _seed_metrics(collapse=0.5)  # below the 0.81 floor
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    assert bs["every_seed_clears_every_backstop"] is False
    assert "collapse" in bs["straddled_gates"]
    assert bs["any_gate_straddles"] is True


def test_backstops_fog_near_ceiling_escalates_even_if_under():
    mt = {str(s): _seed_metrics() for s in T.SEEDS}
    # fog just under the 0.41 ceiling but within 1·σ_fam,fog (0.02) of it: 0.40 > 0.41-0.02.
    mt["43"] = _seed_metrics(fog=0.40)
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    assert bs["any_fog_near_ceiling_escalation"] is True


def test_backstops_fog_over_ceiling_fails():
    mt = {str(s): _seed_metrics() for s in T.SEEDS}
    mt["42"] = _seed_metrics(fog=0.9)
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    assert bs["every_seed_clears_every_backstop"] is False
    assert "fog" in bs["straddled_gates"]


def test_gate_metrics_are_the_three_and_purity_is_descriptive():
    # amendment 2026-08-17: purity is DESCRIPTIVE-only at 50M and is NOT a gated metric.
    assert set(N.GATE_METRICS) == {"heldout_ffr", "collapse", "fog"}
    assert set(N.DESCRIPTIVE_PURITY_METRICS) == {
        "purity_fidelity_k256", "purity_fidelity_k1024",
    }
    # the two sets are disjoint — purity can never enter the go/no-go.
    assert set(N.GATE_METRICS).isdisjoint(N.DESCRIPTIVE_PURITY_METRICS)


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
    """Widen the panel's collapse spread on disk and assert σ_fam grows. A literal would not."""
    base = N.sigma_fam_from_panel(PANEL)["sigma_fam_collapse"]
    sealed = prompt_contract.read_sealed(PANEL, label="panel")
    body = {k: v for k, v in sealed.items() if k != "identity_sha256"}
    body = copy.deepcopy(body)
    body["panel_metric_table"]["54"]["collapse"] = 99.0
    resealed = prompt_contract.seal(body)
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = os.path.join(SCRATCH, "r0267_panel_mutated.json")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(resealed, handle)
    try:
        grown = N.sigma_fam_from_panel(tmp)["sigma_fam_collapse"]
        assert grown > base
    finally:
        os.remove(tmp)


@pytest.mark.skipif(not os.path.exists(FLOORS), reason="R0265 sealed floors absent")
def test_constants_discipline_backstop_tracks_a_mutated_floors_artifact():
    """Re-seal the floors with a changed collapse floor; the gate's backstop must track it."""
    import experiments.round0266_nodes as R0266N

    original = R0266N.read_family_bands(FLOORS)["bands"]["collapse_floor"]
    mutated_value = original + 0.5
    sealed = prompt_contract.read_sealed(FLOORS, label="floors")
    body = {k: v for k, v in sealed.items() if k != "identity_sha256"}
    body = copy.deepcopy(body)
    body["registered_criteria"]["collapse"]["floor"] = mutated_value
    resealed = prompt_contract.seal(body)
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = os.path.join(SCRATCH, "r0267_floors_mutated.json")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(resealed, handle)
    try:
        got = R0266N.read_family_bands(tmp)["bands"]["collapse_floor"]
        assert got == mutated_value
        assert got != original
    finally:
        os.remove(tmp)


@pytest.mark.skipif(not os.path.exists(P1), reason="P1 analysis-v2 result absent")
def test_p1_band_read_from_the_sealed_analysis_v2_result():
    got = N.read_p1_x2_asymptote_band(P1)
    raw = json.load(open(P1))
    lo, hi = raw["bands"]["yinf_x2"]
    assert got["p1_lower"] == float(lo)
    assert got["p1_upper"] == float(hi)
    assert got["verdict"] == "GO"
    assert round(got["p1_lower"], 3) == 0.930 and round(got["p1_upper"], 3) == 0.985


@pytest.mark.skipif(not os.path.exists(P1), reason="P1 analysis-v2 result absent")
def test_constants_discipline_p1_band_tracks_a_mutated_result():
    """Change the P1 band on disk; the gate must return the CHANGED edges, not a literal."""
    original = N.read_p1_x2_asymptote_band(P1)["p1_upper"]
    raw = copy.deepcopy(json.load(open(P1)))
    raw["bands"]["yinf_x2"] = [0.5, 0.6]
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = os.path.join(SCRATCH, "r0267_p1_mutated.json")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(raw, handle)
    try:
        got = N.read_p1_x2_asymptote_band(tmp)
        assert got["p1_lower"] == 0.5 and got["p1_upper"] == 0.6
        assert got["p1_upper"] != original
    finally:
        os.remove(tmp)


@pytest.mark.skipif(
    not (os.path.exists(FLOORS) and os.path.exists(PANEL) and os.path.exists(P1)),
    reason="sealed inputs absent",
)
def test_gate_end_to_end_from_sealed_inputs():
    """A healthy 3-seed family scored against the REAL sealed bands, σ_fam and P1 band:
    criterion 1 and the backstops both pass, entirely from sealed inputs."""
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
    # σ_fam is the sealed family value, not a literal.
    assert sigma["sigma_fam_collapse"] == pytest.approx(0.057774, abs=1e-6)


# --------------------------------------------------------------------------- #
# 4b. the PRE-SEALED int8 substrate LOAD (delegate-approved fix): the SLICE LAW +
#     prefix-digest binding, the loader's verification, the core hook selecting a
#     pre-built HostInt8ArrayDataset without re-encoding, and the file-backed /
#     no-large-copy contract. CPU-only, small synthetic int8 mmaps (no GPU).
# --------------------------------------------------------------------------- #


def _write_synthetic_int8_slice(tmp_path, rows=4, dim=384):
    """A tiny raw int8 substrate + fp16 scales + a sealed-shape slice-law manifest."""
    i8 = (((np.arange(rows * dim) % 255) - 127).astype(np.int8)).reshape(rows, dim)
    sc = np.full(rows, 0.5, dtype=np.float16)
    i8_path = str(tmp_path / "substrate.i8")
    sc_path = str(tmp_path / "substrate-scales.f16")
    i8.tofile(i8_path)
    sc.tofile(sc_path)
    digests = T.int8_slice_prefix_digests(i8_path, sc_path, rows=rows, dimension=dim)
    manifest = {
        "schema": T.INT8_SLICE_SUBSTRATE_SCHEMA,
        "capability": T.INT8_SLICE_SUBSTRATE_CAPABILITY,
        "round_id": T.ROUND_ID,
        "rows": rows,
        "dimension": dim,
        "x_residency": T.X_RESIDENCY,
        "slice_law": {
            "law": "nested-prefix-of-100m-int8-substrate-offset-0",
            "parent_artifact": T.R0262_INT8_CAPABILITY,
            "parent_round": T.R0262_ROUND_ID,
            "parent_i8_path": i8_path,
            "parent_scales_path": sc_path,
            "rows": rows,
            "offset": 0,
            "dimension": dim,
            "prefix_i8_sha256": digests["prefix_i8_sha256"],
            "prefix_scales_sha256": digests["prefix_scales_sha256"],
        },
    }
    return i8, sc, manifest, i8_path, sc_path


def test_slice_law_records_the_pinned_prefix_digests():
    law = T.slice_law_block()
    assert law["rows"] == 50_000_000
    assert law["offset"] == 0
    assert law["dimension"] == 384
    assert law["parent_artifact"] == "minilm-mixed-100m-int8-v1"
    assert law["parent_rows"] == 100_000_000
    assert law["prefix_i8_bytes"] == 50_000_000 * 384
    assert law["prefix_scales_bytes"] == 50_000_000 * 2
    assert law["prefix_i8_sha256"] == (
        "49e41c519487f732ef562af0c508a2c2d93cb2427ac3e1cfe9f6054304780d85"
    )
    assert law["prefix_scales_sha256"] == (
        "4162242e6105ec5083588c2ab3f80c4de406a7ffebfe17300474c270d6c0e96a"
    )
    body = T.int8_slice_substrate_manifest_body(release_sha="0" * 40)
    assert body["schema"] == T.INT8_SLICE_SUBSTRATE_SCHEMA
    assert body["capability"] == T.INT8_SLICE_SUBSTRATE_CAPABILITY
    assert body["rows"] == 50_000_000 and body["dimension"] == 384
    assert body["x_residency"] == "host_int8"
    # the load-bearing content pin is carried into the manifest verbatim.
    assert body["slice_law"]["prefix_i8_sha256"] == law["prefix_i8_sha256"]
    assert body["slice_law"]["prefix_scales_sha256"] == law["prefix_scales_sha256"]


def test_loader_verifies_prefix_digests_and_raises_on_mismatch(tmp_path):
    i8, sc, manifest, i8_path, sc_path = _write_synthetic_int8_slice(tmp_path)
    got_i8, got_sc, receipt = N._load_verified_int8_slice(manifest)
    assert got_i8.shape == (4, 384)
    assert got_sc.shape == (4,)
    np.testing.assert_array_equal(np.asarray(got_i8), i8)
    np.testing.assert_array_equal(np.asarray(got_sc), sc)
    assert receipt["verified_against_sealed_manifest"] is True
    assert receipt["re_encoded_at_train_time"] is False
    assert receipt["offset"] == 0
    # mutate the i8 prefix digest on the manifest -> the loader refuses.
    bad_i8 = copy.deepcopy(manifest)
    bad_i8["slice_law"]["prefix_i8_sha256"] = "0" * 64
    with pytest.raises(N.Round0267NodeError):
        N._load_verified_int8_slice(bad_i8)
    # mutate the scales prefix digest -> the loader refuses.
    bad_sc = copy.deepcopy(manifest)
    bad_sc["slice_law"]["prefix_scales_sha256"] = "0" * 64
    with pytest.raises(N.Round0267NodeError):
        N._load_verified_int8_slice(bad_sc)
    # a non-zero offset (not a true nested prefix) -> refused.
    bad_off = copy.deepcopy(manifest)
    bad_off["slice_law"]["offset"] = 1
    with pytest.raises(N.Round0267NodeError):
        N._load_verified_int8_slice(bad_off)


def test_build_hostint8_dataset_from_slice_does_not_reencode(tmp_path, monkeypatch):
    from basemap.pumap.parametric_umap.datasets import edge_list_dataset as E

    calls = {"n": 0}
    orig = E.quantize_int8_rows

    def spy(block):
        calls["n"] += 1
        return orig(block)

    monkeypatch.setattr(E, "quantize_int8_rows", spy)
    i8, sc, manifest, i8_path, sc_path = _write_synthetic_int8_slice(tmp_path)
    ds, receipt = N.build_hostint8_dataset_from_slice(manifest, "cpu")
    assert ds.host_int8_dataset is True
    assert ds.shape == (4, 384)
    # the pre-sealed load path constructs the dataset from the int8 bytes verbatim.
    assert calls["n"] == 0


def test_core_hook_uses_prebuilt_hostint8_dataset_without_reencode(tmp_path, monkeypatch):
    """x_residency=host_int8 + a pre-built HostInt8ArrayDataset X: core.fit uses it
    directly (the returned dataset IS the one passed in) and never re-encodes."""
    from basemap.pumap.parametric_umap.datasets import edge_list_dataset as E
    from basemap.pumap.parametric_umap import ParametricUMAP

    calls = {"n": 0}
    orig = E.quantize_int8_rows

    def spy(block):
        calls["n"] += 1
        return orig(block)

    monkeypatch.setattr(E, "quantize_int8_rows", spy)
    i8, sc, manifest, i8_path, sc_path = _write_synthetic_int8_slice(tmp_path, rows=4, dim=384)
    ds, _receipt = N.build_hostint8_dataset_from_slice(manifest, "cpu")

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
    # THE HOOK: the pre-built dataset is used directly, not re-wrapped/re-encoded.
    assert dataset is ds
    assert model._X_dev is ds
    assert model._pipeline_info["x_residency"] == "host_int8"
    assert model._pipeline_info["pipeline"] == "host_int8"
    assert calls["n"] == 0


def test_hostint8_dataset_stays_file_backed_no_large_copy(tmp_path):
    """A C-contiguous int8 mmap prefix handed in as encoded= stays file-backed:
    the torch buffer shares memory with the mmap (no 19.2 GB anonymous copy)."""
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
    mm_i8 = np.memmap(i8_path, dtype=np.int8, mode="r").reshape(-1, dim)[:rows]
    mm_sc = np.memmap(sc_path, dtype=np.float16, mode="r")[:rows]
    assert mm_i8.flags["C_CONTIGUOUS"] and mm_sc.flags["C_CONTIGUOUS"]
    ds = HostInt8ArrayDataset(None, "cpu", encoded=mm_i8, scales=mm_sc)
    # no copy was made: the dataset's int8/scale buffers share memory with the mmap
    # prefixes we passed in (a materialising copy would break shares_memory).
    assert np.shares_memory(ds._i8.numpy(), mm_i8)
    assert np.shares_memory(ds._scales.numpy(), mm_sc)


# --------------------------------------------------------------------------- #
# 5. dispatch + scope registration, and the action guard
# --------------------------------------------------------------------------- #


def test_round0267_is_registered_in_scope_modules():
    from basemap.round0254_dispatch import (
        SCOPE_MODULES,
        assert_derived_entries_install,
        dispatch_census,
    )
    assert "experiments.round0267_nodes" in SCOPE_MODULES
    guard = assert_derived_entries_install(SCOPE_MODULES, dispatch_census())
    assert guard["audit"]["every_entry_installs_effectively"] is True


def test_run_job_rejects_an_unknown_action():
    with pytest.raises(N.Round0267NodeError):
        N.run_job({"manifest": {"round_id": "0267"}}, {"action": "not-a-real-action"})


def test_dose_multiplier_is_two():
    assert T.DOSE_MULTIPLIER == 2
    assert N.DOSE_MULTIPLIER == 2
    assert T.SEEDS == (42, 43, 44)


# --------------------------------------------------------------------------- #
# 6. treatment-vs-execution: the masked-config / treatment invariant digest is
#    INVARIANT to a change in HOST_RSS_LIMIT_GIB (and the other resource
#    constants), so seed42 (trained at limit 60) and 43/44 (limit 100) are
#    treatment-identical. The RSS limit is an execution-resource field, excluded
#    from the config and the treatment digest.
# --------------------------------------------------------------------------- #


def test_host_rss_limit_is_raised_to_100():
    assert N.HOST_RSS_LIMIT_GIB == 100.0


def test_treatment_digest_excludes_execution_resource_fields(monkeypatch):
    cfg = _honest()
    digest = T.fneg_seed_invariant_sha256(cfg)

    # The node's post-train RSS backstop (N.HOST_RSS_LIMIT_GIB, raised 60->100) and the
    # anon/device byte budgets are EXECUTION-resource fields living in round0267_nodes, not
    # in the config: neither the raised value nor the old value leaks into the config /
    # treatment. (The config DOES carry a fixed R0217 pipeline-stamp field
    # `host_rss_limit_gib: 32.0`, which is unrelated to the node backstop and is the same
    # 32.0 for every seed, so it does not distinguish the two correction runs.)
    blob = json.dumps(cfg)
    for token in ("100.0", "60.0", str(N.R0267_ANON_BUDGET_BYTES), str(N.DEVICE_BUDGET_BYTES)):
        assert token not in blob

    # THE EXECUTED PROOF: changing the module's execution-resource constants leaves the
    # treatment / masked-config invariant digest unchanged — so seed42 (trained at limit 60)
    # and seeds 43/44 (limit 100) are treatment-identical.
    monkeypatch.setattr(N, "HOST_RSS_LIMIT_GIB", 60.0)
    monkeypatch.setattr(N, "R0267_ANON_BUDGET_BYTES", 7 * (1 << 30))
    monkeypatch.setattr(N, "DEVICE_BUDGET_BYTES", 3 * (1 << 30))
    digest_at_60 = T.fneg_seed_invariant_sha256(_honest())
    monkeypatch.setattr(N, "HOST_RSS_LIMIT_GIB", 100.0)
    digest_at_100 = T.fneg_seed_invariant_sha256(_honest())
    assert digest_at_60 == digest_at_100 == digest


# --------------------------------------------------------------------------- #
# 7. the SALVAGE of seed42: bind the sealed correction-3 artifacts (verify
#    digests, extract the clean-train evidence), the panel reads the bound
#    coordinates (pin-verified) with no train-receipt, and the gate scores all
#    three cells including the salvaged one.
# --------------------------------------------------------------------------- #

SALVAGE_DIR = (
    "/data/latent-basemap/runs/round-0267/queue-correction-3/artifacts/"
    "minilm-mixed-50000k-fneg-x2-md000-hostint8-seed42-r0267-v1"
)
SALVAGE_LOG = (
    "/data/latent-basemap/runs/round-0267/queue-correction-3/logs/"
    "train_minilm_fneg_50m_x2_hostint8_seed42.log"
)
_SALVAGE_PRESENT = os.path.exists(SALVAGE_DIR) and os.path.exists(SALVAGE_LOG)


def test_salvage_pins_match_the_node_and_prepare_constants():
    assert N.SALVAGE_SEED == 42
    assert N.SALVAGE_SOURCE_RUN == "queue-correction-3"
    assert (
        N.SALVAGE_SEED42_COORDINATES_SHA256
        == "aa7bbe678e6206c96ec6eb443aa6b6a2c4d8b589585c8c0db6f1b7eb9fc55284"
    )
    # prepare's coordinates digest is the node's pinned digest (single source of truth).
    assert P.SALVAGE_EXPECTED_DIGESTS["coordinates.npy"] == N.SALVAGE_SEED42_COORDINATES_SHA256


@pytest.mark.skipif(not _SALVAGE_PRESENT, reason="correction-3 seed42 artifacts absent")
def test_salvage_bind_verifies_seed42_digests_and_clean_train():
    cell, salvaged_inv = P.bind_salvaged_seed42_cell(
        artifact_dir=SALVAGE_DIR, log_path=SALVAGE_LOG
    )
    assert cell["seed"] == 42
    assert cell["salvaged"] is True
    assert cell["salvage"]["source_run"] == "queue-correction-3"
    assert cell["salvage"]["reason"] == N.SALVAGE_REASON
    # every bound artifact matches its sealed correction-3 digest.
    assert cell["coordinates"]["sha256"] == N.SALVAGE_SEED42_COORDINATES_SHA256
    for name, expected in P.SALVAGE_EXPECTED_DIGESTS.items():
        assert cell["salvage"]["digests"][name] == expected
    assert cell["train_log"]["sha256"] == P.SALVAGE_EXPECTED_LOG_SHA256
    # the clean full-horizon train evidence (in lieu of a train-receipt).
    step = cell["salvage"]["train_evidence"]["step_accounting_line"]
    assert "succeeded 4162228, AMP-skips 0, nonfinite loss/grad 0/0" in step
    assert salvaged_inv and salvaged_inv == cell["salvage"]["seed_invariant_sha256"]
    # treatment-identity guard: a matching invariant binds, a wrong one raises.
    P.bind_salvaged_seed42_cell(
        artifact_dir=SALVAGE_DIR, log_path=SALVAGE_LOG, expected_cell_invariant=salvaged_inv
    )
    with pytest.raises(RuntimeError):
        P.bind_salvaged_seed42_cell(
            artifact_dir=SALVAGE_DIR, log_path=SALVAGE_LOG, expected_cell_invariant="0" * 64
        )


@pytest.mark.skipif(not _SALVAGE_PRESENT, reason="correction-3 seed42 artifacts absent")
def test_salvage_dose_is_a_derived_checked_equality_not_a_literal():
    """Requirement (b): the log's succeeded count is checked as `== 2 * S_u_f_e(edges)`.

    The bind runs the log's parsed succeeded count through validate_dose_x2 (which derives
    the target from the sealed edge count), so the salvage carries the ×2 dose as a checked
    equality against a derived target — not a match on the literal 4,162,228.
    """
    cell, _inv = P.bind_salvaged_seed42_cell(artifact_dir=SALVAGE_DIR, log_path=SALVAGE_LOG)
    evidence = cell["salvage"]["train_evidence"]
    receipt = evidence["dose_receipt"]
    base = successful_updates_for_edges(T.SEALED_DIRECTED_EDGES)
    expected = T.DOSE_MULTIPLIER * base
    # the log's succeeded count equals the DERIVED ×2 horizon (the checked equality).
    assert evidence["succeeded_updates"] == expected
    assert receipt["successful_updates"] == expected
    assert receipt["base_successful_updates"] == base
    assert receipt["dose_multiplier"] == 2
    assert "validate_dose_x2" in evidence["dose_checked_equality"]
    assert str(expected) in evidence["dose_checked_equality"]
    # published cross-check anchor: the derivation lands on 4,162,228.
    assert expected == 4_162_228 == 2 * 2_081_114


@pytest.mark.skipif(not _SALVAGE_PRESENT, reason="correction-3 seed42 artifacts absent")
def test_salvage_bind_raises_on_a_mutated_digest(monkeypatch):
    bad = dict(P.SALVAGE_EXPECTED_DIGESTS)
    bad["model.pt"] = "0" * 64
    monkeypatch.setattr(P, "SALVAGE_EXPECTED_DIGESTS", bad)
    with pytest.raises(RuntimeError):
        P.bind_salvaged_seed42_cell(artifact_dir=SALVAGE_DIR, log_path=SALVAGE_LOG)


@pytest.mark.skipif(not _SALVAGE_PRESENT, reason="correction-3 seed42 artifacts absent")
def test_salvage_bind_raises_on_a_mutated_log_digest(monkeypatch):
    monkeypatch.setattr(P, "SALVAGE_EXPECTED_LOG_SHA256", "0" * 64)
    with pytest.raises(RuntimeError):
        P.bind_salvaged_seed42_cell(artifact_dir=SALVAGE_DIR, log_path=SALVAGE_LOG)


@pytest.mark.skipif(not _SALVAGE_PRESENT, reason="correction-3 seed42 artifacts absent")
def test_panel_authenticates_and_reads_the_bound_seed42_coords():
    cell, _inv = P.bind_salvaged_seed42_cell(artifact_dir=SALVAGE_DIR, log_path=SALVAGE_LOG)
    entry = N._authenticate_salvaged_50m_map(cell)
    assert entry["salvaged"] is True and entry["seed"] == 42
    # provenance is sourced from the bound production-config, NOT a train-receipt.
    assert entry["seed_invariant_sha256"]
    assert "receipt" not in entry
    # the panel reads the BOUND correction-3 coordinates.
    assert entry["coordinates_path"] == cell["coordinates"]["canonical_path"]
    coords = np.load(entry["coordinates_path"], mmap_mode="r", allow_pickle=False)
    assert coords.shape == (T.ROWS, 2)


@pytest.mark.skipif(not _SALVAGE_PRESENT, reason="correction-3 seed42 artifacts absent")
def test_panel_pin_rejects_a_wrong_seed42_coordinates_digest(monkeypatch):
    cell, _inv = P.bind_salvaged_seed42_cell(artifact_dir=SALVAGE_DIR, log_path=SALVAGE_LOG)
    # the bound bytes are self-consistent (verify_signature passes) but the panel's
    # independent PIN rejects a coordinates map that is not the sealed correction-3 one.
    monkeypatch.setattr(N, "SALVAGE_SEED42_COORDINATES_SHA256", "0" * 64)
    with pytest.raises(N.Round0267NodeError):
        N._authenticate_salvaged_50m_map(cell)


def _panel_cell(seed, *, salvaged=False):
    if salvaged:
        return {
            "seed": seed,
            "salvaged": True,
            "salvage": {
                "salvaged": True,
                "source_run": "queue-correction-3",
                "reason": N.SALVAGE_REASON,
                "train_evidence": {"step_accounting_line": "succeeded 4162228, AMP-skips 0"},
            },
        }
    return {"seed": seed, "salvaged": False, "train_receipt": {"kind": "file"}}


def test_gate_scores_three_cells_including_the_salvaged_one():
    panel = {
        "capability": N.PANEL_CAPABILITY,
        "schema": N.PANEL_SCHEMA,
        "panel_metric_table": {str(s): _seed_metrics() for s in T.SEEDS},
        "cells": {
            "42": _panel_cell(42, salvaged=True),
            "43": _panel_cell(43),
            "44": _panel_cell(44),
        },
    }
    mt = N._metric_table_from_panel(panel)
    assert {int(s) for s in mt} == set(T.SEEDS)
    prov = N._salvage_provenance_from_panel(panel)
    assert set(prov) == {"42"}
    assert prov["42"]["source_run"] == "queue-correction-3"
    assert prov["42"]["salvaged"] is True


def test_gate_salvage_provenance_rejects_a_wrong_source_run():
    panel = {
        "capability": N.PANEL_CAPABILITY,
        "schema": N.PANEL_SCHEMA,
        "panel_metric_table": {str(s): _seed_metrics() for s in T.SEEDS},
        "cells": {"42": _panel_cell(42, salvaged=True), "43": _panel_cell(43), "44": _panel_cell(44)},
    }
    panel["cells"]["42"]["salvage"]["source_run"] = "queue-correction-1"
    with pytest.raises(N.Round0267NodeError):
        N._salvage_provenance_from_panel(panel)


# --------------------------------------------------------------------------- #
# 7b. the COMPLETED-CELL BIND of seeds 43/44 (correction-5): bind their correction-4
#     artifacts as completed cells (compute + seal digests, verify the real train-
#     receipt as first-class provenance, prove treatment-identity), the panel
#     authenticates + scores three BOUND cells from bound coords, and the gate scores
#     all three including the bound-completed ones. Generalises the seed42 salvage while
#     leaving it intact. CPU-only.
# --------------------------------------------------------------------------- #

COMPLETED_DIR = (
    "/data/latent-basemap/runs/round-0267/queue-correction-4/artifacts/"
    "minilm-mixed-50000k-fneg-x2-md000-hostint8-seed{seed}-r0267-v1"
)
_COMPLETED_PRESENT = all(
    os.path.exists(COMPLETED_DIR.format(seed=s)) for s in (43, 44)
)


def test_completed_bind_pins_match_the_node_and_prepare_constants():
    assert N.COMPLETED_BIND_SEEDS == (43, 44)
    assert P.COMPLETED_BIND_SEEDS == (43, 44)
    assert N.COMPLETED_SOURCE_RUN == "queue-correction-4"
    assert N.COMPLETED_REASON == P.COMPLETED_REASON
    assert set(P.COMPLETED_CELL_ARTIFACTS) == {
        "coordinates.npy", "model.pt", "admission.json",
        "production-config.json", "train-receipt.json",
    }


def test_completed_bind_rejects_unknown_and_salvage_seeds():
    # only 43/44 may be completed-bound; 42 (salvage) and 45 (unknown) are refused.
    with pytest.raises(RuntimeError):
        P.bind_completed_seed_cell(seed=42, artifact_dir="/x")
    with pytest.raises(RuntimeError):
        P.bind_completed_seed_cell(seed=45, artifact_dir="/x")


@pytest.mark.skipif(not _COMPLETED_PRESENT, reason="correction-4 seed43/44 artifacts absent")
def test_completed_bind_verifies_43_44_digests_and_receipt():
    cells = {}
    invs = set()
    for seed in (43, 44):
        cell, inv = P.bind_completed_seed_cell(seed=seed, artifact_dir=COMPLETED_DIR.format(seed=seed))
        cells[seed] = cell
        invs.add(inv)
        assert cell["seed"] == seed
        assert cell["bound_completed"] is True and cell["salvaged"] is False
        assert cell["bound"]["source_run"] == "queue-correction-4"
        assert cell["bound"]["reason"] == N.COMPLETED_REASON
        # digests are COMPUTED from the files (5 artifacts) and sealed into the record.
        assert set(cell["bound"]["digests"]) == set(P.COMPLETED_CELL_ARTIFACTS)
        # the per-cell coordinates pin equals the computed coordinates digest.
        assert cell["coordinates_sha256_pin"] == cell["coordinates"]["sha256"]
        assert cell["coordinates_sha256_pin"] == cell["bound"]["digests"]["coordinates.npy"]
        # first-class receipt provenance (NOT a log): tied to the coordinates bytes.
        prov = cell["bound"]["train_receipt_provenance"]
        assert prov["is_first_class_receipt"] is True
        assert prov["coordinates_sha256"] == cell["coordinates_sha256_pin"]
        assert prov["train_receipt_sha256"] == cell["train_receipt"]["sha256"]
    # 43 and 44 share ONE masked recipe invariant (the same treatment).
    assert len(invs) == 1
    # and the two cells' coordinates differ (two real maps).
    assert cells[43]["coordinates_sha256_pin"] != cells[44]["coordinates_sha256_pin"]


@pytest.mark.skipif(not _COMPLETED_PRESENT, reason="correction-4 seed43/44 artifacts absent")
def test_completed_bind_treatment_identity_guard():
    _cell, inv = P.bind_completed_seed_cell(seed=43, artifact_dir=COMPLETED_DIR.format(seed=43))
    # a matching expected invariant binds; a wrong one raises (treatment-identity).
    P.bind_completed_seed_cell(
        seed=43, artifact_dir=COMPLETED_DIR.format(seed=43), expected_cell_invariant=inv
    )
    with pytest.raises(RuntimeError):
        P.bind_completed_seed_cell(
            seed=43, artifact_dir=COMPLETED_DIR.format(seed=43), expected_cell_invariant="0" * 64
        )


@pytest.mark.skipif(not _COMPLETED_PRESENT, reason="correction-4 seed43/44 artifacts absent")
def test_completed_bind_raises_when_a_digest_does_not_describe_the_coords(tmp_path):
    """A tampered coordinates file (its digest no longer matches the sealed receipt) raises."""
    import shutil

    src = COMPLETED_DIR.format(seed=43)
    dst = tmp_path / "seed43"
    dst.mkdir()
    for name in ("admission.json", "production-config.json", "train-receipt.json"):
        shutil.copy(os.path.join(src, name), dst / name)
    # fake coordinates/model: their bytes (hence digests) differ from what the receipt records.
    (dst / "coordinates.npy").write_bytes(b"not the real coordinates")
    (dst / "model.pt").write_bytes(b"not the real model")
    with pytest.raises(RuntimeError):
        P.bind_completed_seed_cell(seed=43, artifact_dir=str(dst))


@pytest.mark.skipif(not _COMPLETED_PRESENT, reason="correction-4 seed43/44 artifacts absent")
def test_panel_authenticates_three_bound_cells_from_bound_coords():
    """seed42 salvaged + seed43/44 completed-bound: all three authenticate, one recipe."""
    salvaged, _ = P.bind_salvaged_seed42_cell(artifact_dir=SALVAGE_DIR, log_path=SALVAGE_LOG)
    completed = {
        s: P.bind_completed_seed_cell(seed=s, artifact_dir=COMPLETED_DIR.format(seed=s))[0]
        for s in (43, 44)
    }
    ordered = completed[43]["bound"]["train_receipt_provenance"]["ordered_substrate_sha256"]
    substrate = {"ordered_substrate_sha256": ordered}
    entries = [
        N._authenticate_salvaged_50m_map(salvaged),
        N._authenticate_bound_completed_50m_map(completed[43], substrate),
        N._authenticate_bound_completed_50m_map(completed[44], substrate),
    ]
    assert {e["seed"] for e in entries} == set(T.SEEDS)
    # one pooled recipe invariant across the three bound cells.
    assert len({e["seed_invariant_sha256"] for e in entries}) == 1
    for e in entries[1:]:
        assert e["bound_completed"] is True
        assert e["coordinates_path"].endswith("coordinates.npy")
        assert "receipt" not in e  # provenance is the bound receipt signature, not a fresh receipt


@pytest.mark.skipif(not _COMPLETED_PRESENT, reason="correction-4 seed43/44 artifacts absent")
def test_authenticate_bound_completed_rejects_a_tampered_pin_or_signature():
    cell, _ = P.bind_completed_seed_cell(seed=43, artifact_dir=COMPLETED_DIR.format(seed=43))
    ordered = cell["bound"]["train_receipt_provenance"]["ordered_substrate_sha256"]
    substrate = {"ordered_substrate_sha256": ordered}
    # honest authenticates.
    N._authenticate_bound_completed_50m_map(cell, substrate)
    # tamper only the per-cell coords pin (signature still valid) -> the pin check raises.
    bad_pin = copy.deepcopy(cell)
    bad_pin["coordinates_sha256_pin"] = "0" * 64
    with pytest.raises(N.Round0267NodeError):
        N._authenticate_bound_completed_50m_map(bad_pin, substrate)
    # tamper the bound coordinates signature digest -> verify_signature raises.
    bad_sig = copy.deepcopy(cell)
    bad_sig["coordinates"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        N._authenticate_bound_completed_50m_map(bad_sig, substrate)
    # a mismatched substrate lineage -> raises.
    with pytest.raises(N.Round0267NodeError):
        N._authenticate_bound_completed_50m_map(cell, {"ordered_substrate_sha256": "0" * 64})


def _completed_panel_cell(seed):
    return {
        "seed": seed,
        "salvaged": False,
        "bound_completed": True,
        "bound": {
            "bound_completed": True,
            "source_run": "queue-correction-4",
            "reason": N.COMPLETED_REASON,
            "train_receipt_provenance": {
                "train_receipt_sha256": "a" * 64, "is_first_class_receipt": True,
            },
        },
        "train_receipt": {"kind": "file"},
    }


def test_gate_scores_three_bound_cells_incl_completed():
    """42 salvaged + 43/44 completed-bound: the gate reads all three metrics and sources
    salvage provenance for 42 and bound-receipt provenance for 43/44."""
    panel = {
        "capability": N.PANEL_CAPABILITY,
        "schema": N.PANEL_SCHEMA,
        "panel_metric_table": {str(s): _seed_metrics() for s in T.SEEDS},
        "cells": {
            "42": _panel_cell(42, salvaged=True),
            "43": _completed_panel_cell(43),
            "44": _completed_panel_cell(44),
        },
    }
    mt = N._metric_table_from_panel(panel)
    assert {int(s) for s in mt} == set(T.SEEDS)
    salvage_prov = N._salvage_provenance_from_panel(panel)
    assert set(salvage_prov) == {"42"}
    bound_prov = N._bound_provenance_from_panel(panel)
    assert set(bound_prov) == {"43", "44"}
    for s in ("43", "44"):
        assert bound_prov[s]["bound_completed"] is True
        assert bound_prov[s]["source_run"] == "queue-correction-4"
        assert "train_receipt_provenance" in bound_prov[s]


def test_gate_bound_provenance_rejects_a_wrong_source_run():
    panel = {
        "capability": N.PANEL_CAPABILITY,
        "schema": N.PANEL_SCHEMA,
        "panel_metric_table": {str(s): _seed_metrics() for s in T.SEEDS},
        "cells": {
            "42": _panel_cell(42, salvaged=True),
            "43": _completed_panel_cell(43),
            "44": _completed_panel_cell(44),
        },
    }
    panel["cells"]["43"]["bound"]["source_run"] = "queue-correction-1"
    with pytest.raises(N.Round0267NodeError):
        N._bound_provenance_from_panel(panel)


def test_prepare_rejects_an_unknown_or_double_completed_bind_seed():
    # unknown completed-bind seed refused (before any round/file work). correction-5:
    # no slim-scale-cert is required or accepted.
    with pytest.raises(ValueError):
        P.prepare_round0267(
            release_sha="0" * 40, completed_binds={45: "/x"}
        )


def test_cli_bind_completed_rejects_bad_specs():
    # unknown seed, double-bind, and a malformed spec all fail at the CLI boundary.
    for argv in (
        ["--release-sha", "0" * 40, "--bind-completed", "45=/x"],
        ["--release-sha", "0" * 40, "--bind-completed", "43=/a", "--bind-completed", "43=/b"],
        ["--release-sha", "0" * 40, "--bind-completed", "43"],
    ):
        with pytest.raises(SystemExit):
            P.main(argv)


# --------------------------------------------------------------------------- #
# 8. the slim >=8M scale-performance admission (design-slim-8m-admission-2026-08-17):
#    the additive second accepted score_panel scale_admission form. All CPU/CUDA-
#    hidden, small (<8M) fixtures + monkeypatch: NO real 50M cert, NO GPU.
# --------------------------------------------------------------------------- #

import types

import basemap.panel_v2 as pv
import basemap.slim_scale_admission as SLIM
import experiments.produce_slim_scale_cert as PSC
from basemap.artifact_identity import expected_input_signature
from experiments.round0005_performance_gate import derive_scale_rows

RELEASE_SHA = "933c31ed18265ba0d8b0dd1df33b921f760cef06"


class _FakeBig:
    """A stand-in matrix reporting a >=8M length so the guard reaches its scale branches."""

    def __len__(self):
        return 8_000_000


def _write_substrate(tmp_path, rows=16, dim=4, name="substrate.f32.npy"):
    array = np.arange(rows * dim, dtype=np.float32).reshape(rows, dim)
    path = str(tmp_path / name)
    np.save(path, array)
    return path, np.load(path, mmap_mode="r", allow_pickle=False)


def _make_production_panel(tmp_path, *, name="measurement-panel.json", wall_s=100.0,
                          rss=50.0, allocated=1000, reserved=2000, with_purpose=True):
    panel = {
        "schema": "panel_v2",
        "provenance": {
            "wall_s": wall_s,
            "peak_rss_gb": rss,
            "cuda_peak": {"available": True, "allocated_bytes": allocated,
                          "reserved_bytes": reserved},
        },
    }
    if with_purpose:
        panel["purpose"] = SLIM.SLIM_CERT_PRODUCTION_PURPOSE
    path = str(tmp_path / name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(panel, handle)
    return path


def _make_json(tmp_path, name, body):
    path = str(tmp_path / name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(body, handle)
    return path


def _build_cert(tmp_path, *, release_sha=RELEASE_SHA, scientific_rows, row_derivation,
                panel_path, regression_path, receipt_path,
                drop_receipt=False, name="slim-scale-cert.json"):
    release_preflight = ({"identity_sha256": "a" * 64} if drop_receipt else
                         {"receipt": expected_input_signature(receipt_path),
                          "identity_sha256": "a" * 64})
    body = {
        "schema": SLIM.SLIM_SCALE_CERT_SCHEMA,
        "kind": SLIM.SLIM_SCALE_CERT_KIND,
        "release_sha": release_sha,
        "scientific_rows": scientific_rows,
        "row_derivation": dict(row_derivation),
        "budgets": dict(SLIM.SLIM_BUDGETS),
        "measurement": {"panel_output": expected_input_signature(panel_path)},
        "regression": expected_input_signature(regression_path),
        "release_preflight": release_preflight,
    }
    cert = prompt_contract.seal(body)
    path = str(tmp_path / name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cert, handle)
    return path, cert


@pytest.fixture
def slim(tmp_path, monkeypatch):
    """Honest slim admission fixture (small rows; passing inner validators)."""
    monkeypatch.setattr(SLIM, "SLIM_SCALE_ROWS", 8)
    monkeypatch.setattr(SLIM, "validate_regression_certificate", lambda report: {"passed": True})
    monkeypatch.setattr(SLIM, "validate_release_preflight_receipt",
                        lambda path, **kwargs: {"release_sha": RELEASE_SHA})
    sub_path, src = _write_substrate(tmp_path)
    rd = derive_scale_rows(sub_path, dimensions=4, loaded_matrix=src)
    return types.SimpleNamespace(
        tmp_path=tmp_path, monkeypatch=monkeypatch, src=src, rd=rd, sub_path=sub_path,
        panel_path=_make_production_panel(tmp_path),
        reg_path=_make_json(tmp_path, "regression.json",
                            {"schema": "round0005_real_scorer_4x_regression.v3"}),
        rec_path=_make_json(tmp_path, "release-preflight.json",
                           {"schema": "round0005_release_preflight.v3",
                            "identity_sha256": "a" * 64, "release_sha": RELEASE_SHA}),
    )


def _admission(cert_path, rd, *, rows=16, release_sha=RELEASE_SHA):
    return SLIM.build_slim_scale_admission(
        release_sha=release_sha, scientific_rows=rows, row_derivation=rd,
        certificate=expected_input_signature(cert_path))


# ---- positive controls ---------------------------------------------------- #


def test_slim_admission_validates_on_a_fixture(slim):
    cert_path, cert = _build_cert(
        slim.tmp_path, scientific_rows=16, row_derivation=slim.rd,
        panel_path=slim.panel_path, regression_path=slim.reg_path, receipt_path=slim.rec_path)
    out = SLIM.validate_slim_scale_admission(
        _admission(cert_path, slim.rd), X=slim.src, scientific_rows=16)
    assert out["identity_sha256"] == cert["identity_sha256"]
    assert out["kind"] == SLIM.SLIM_SCALE_CERT_KIND


def test_v1_admission_still_routed_to_the_v1_validator():
    """A v1-shaped admission (no `kind`) is NOT intercepted by the slim branches: it
    reaches the byte-identical v1 origin check (proving the v1 path is untouched)."""
    v1 = {
        "performance_gate": "/x", "release_sha": "9" * 40,
        "row_derivation": {"embedding_input": {"canonical_path": "/nonexistent"},
                           "scientific_rows": 8_000_000, "dimensions": 4},
        "scale_policy": None,
    }
    with pytest.raises(RuntimeError, match="reopened signed embedding input"):
        pv._require_score_panel_scale_admission(_FakeBig(), v1)


def test_absence_still_hard_raises_at_scale():
    with pytest.raises(RuntimeError, match="requires exact replayable scale admission"):
        pv._require_score_panel_scale_admission(_FakeBig(), None)


def test_guard_routes_slim_cert_kind_at_scale(monkeypatch):
    seen = {}

    def fake(scale_admission, *, X, scientific_rows):
        seen["rows"] = scientific_rows
        return {"ok": True}

    monkeypatch.setattr(SLIM, "validate_slim_scale_admission", fake)
    out = pv._require_score_panel_scale_admission(_FakeBig(), {"kind": "slim-scale-cert"})
    assert out == {"ok": True} and seen["rows"] == 8_000_000


def test_guard_routes_production_marker_at_scale(monkeypatch):
    seen = {}

    def fake(scale_admission, *, X, scientific_rows):
        seen["rows"] = scientific_rows
        return {"purpose": SLIM.SLIM_CERT_PRODUCTION_PURPOSE}

    monkeypatch.setattr(SLIM, "permit_slim_cert_production", fake)
    out = pv._require_score_panel_scale_admission(_FakeBig(), {"kind": "slim-cert-production"})
    assert out["purpose"] == SLIM.SLIM_CERT_PRODUCTION_PURPOSE and seen["rows"] == 8_000_000


def test_permit_production_marker_binds_and_stamps(slim):
    marker = SLIM.slim_cert_production_marker(release_sha=RELEASE_SHA, row_derivation=slim.rd)
    rec = SLIM.permit_slim_cert_production(marker, X=slim.src, scientific_rows=16)
    assert rec["purpose"] == SLIM.SLIM_CERT_PRODUCTION_PURPOSE
    assert SLIM._valid_sha256(rec["identity_sha256"])
    bad = dict(marker)
    bad["budgets"] = {"wall_max_s": 1.0}
    with pytest.raises(SLIM.SlimScaleAdmissionError):
        SLIM.permit_slim_cert_production(bad, X=slim.src, scientific_rows=16)


# ---- the SEVEN refusal plants --------------------------------------------- #


def test_plant_a_mismatched_release_sha(slim):
    cert_path, _ = _build_cert(
        slim.tmp_path, scientific_rows=16, row_derivation=slim.rd,
        panel_path=slim.panel_path, regression_path=slim.reg_path, receipt_path=slim.rec_path)
    admission = _admission(cert_path, slim.rd, release_sha="b" * 40)
    with pytest.raises(SLIM.SlimScaleAdmissionError):
        SLIM.validate_slim_scale_admission(admission, X=slim.src, scientific_rows=16)


def test_plant_b_missing_receipt(slim):
    cert_path, _ = _build_cert(
        slim.tmp_path, scientific_rows=16, row_derivation=slim.rd,
        panel_path=slim.panel_path, regression_path=slim.reg_path,
        receipt_path=slim.rec_path, drop_receipt=True)
    with pytest.raises(SLIM.SlimScaleAdmissionError):
        SLIM.validate_slim_scale_admission(
            _admission(cert_path, slim.rd), X=slim.src, scientific_rows=16)


def test_plant_c_budget_exceeding_measurement(slim):
    bad_panel = _make_production_panel(slim.tmp_path, name="over-budget.json", wall_s=99999.0)
    cert_path, _ = _build_cert(
        slim.tmp_path, scientific_rows=16, row_derivation=slim.rd,
        panel_path=bad_panel, regression_path=slim.reg_path, receipt_path=slim.rec_path)
    with pytest.raises(SLIM.SlimScaleAdmissionError):
        SLIM.validate_slim_scale_admission(
            _admission(cert_path, slim.rd), X=slim.src, scientific_rows=16)


def test_plant_d_altered_regression(slim):
    slim.monkeypatch.setattr(SLIM, "validate_regression_certificate",
                             lambda report: {"passed": False})
    cert_path, _ = _build_cert(
        slim.tmp_path, scientific_rows=16, row_derivation=slim.rd,
        panel_path=slim.panel_path, regression_path=slim.reg_path, receipt_path=slim.rec_path)
    with pytest.raises(SLIM.SlimScaleAdmissionError):
        SLIM.validate_slim_scale_admission(
            _admission(cert_path, slim.rd), X=slim.src, scientific_rows=16)


def test_plant_e_altered_release_preflight(slim):
    def boom(path, **kwargs):
        raise RuntimeError("release preflight changed after publication")

    slim.monkeypatch.setattr(SLIM, "validate_release_preflight_receipt", boom)
    cert_path, _ = _build_cert(
        slim.tmp_path, scientific_rows=16, row_derivation=slim.rd,
        panel_path=slim.panel_path, regression_path=slim.reg_path, receipt_path=slim.rec_path)
    with pytest.raises(SLIM.SlimScaleAdmissionError):
        SLIM.validate_slim_scale_admission(
            _admission(cert_path, slim.rd), X=slim.src, scientific_rows=16)


def test_plant_f_broken_input_digest(slim):
    cert_path, _ = _build_cert(
        slim.tmp_path, scientific_rows=16, row_derivation=slim.rd,
        panel_path=slim.panel_path, regression_path=slim.reg_path, receipt_path=slim.rec_path)
    admission = _admission(cert_path, slim.rd)  # signature captured BEFORE tampering
    with open(cert_path, "a", encoding="utf-8") as handle:
        handle.write("  ")  # mutate the cert bytes -> the digest no longer matches
    with pytest.raises(SLIM.SlimScaleAdmissionError):
        SLIM.validate_slim_scale_admission(admission, X=slim.src, scientific_rows=16)


def test_plant_g_consumption_side_refuses_a_production_panel():
    panel = {
        "capability": N.PANEL_CAPABILITY,
        "schema": N.PANEL_SCHEMA,
        "purpose": SLIM.SLIM_CERT_PRODUCTION_PURPOSE,
        "panel_metric_table": {str(s): _seed_metrics() for s in T.SEEDS},
        "cells": {"42": _panel_cell(42, salvaged=True), "43": _panel_cell(43), "44": _panel_cell(44)},
    }
    with pytest.raises(N.Round0267NodeError):
        N._metric_table_from_panel(panel)
    with pytest.raises(N.Round0267NodeError):
        SLIM.assert_not_slim_cert_production_panel(
            panel, label="gate", error_cls=N.Round0267NodeError)
    # a scientific panel (no production stamp) is NOT refused.
    SLIM.assert_not_slim_cert_production_panel(
        {"schema": "panel_v2"}, label="gate", error_cls=N.Round0267NodeError)


# ---- the stamp + the scoring-invariance guard ----------------------------- #


def _small_panel_inputs(seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(400, 8).astype("float32")
    Z = rng.randn(400, 2).astype("float32")
    cfg = pv.PanelV2Config(frac=0.02, n_anchors=60, corpus_chunk=128)
    return X, Z, cfg


def test_score_panel_stamps_only_the_production_pass(monkeypatch):
    X, Z, cfg = _small_panel_inputs(5)
    monkeypatch.setattr(pv, "_require_score_panel_scale_admission",
                        lambda X_, sa: {"purpose": SLIM.SLIM_CERT_PRODUCTION_PURPOSE,
                                        "identity_sha256": "a" * 64})
    assert pv.score_panel(X, Z, config=cfg, provenance={"t": "s"}).get("purpose") == \
        SLIM.SLIM_CERT_PRODUCTION_PURPOSE
    monkeypatch.setattr(pv, "_require_score_panel_scale_admission",
                        lambda X_, sa: {"identity_sha256": "a" * 64})
    assert "purpose" not in pv.score_panel(X, Z, config=cfg, provenance={"t": "s"})
    monkeypatch.setattr(pv, "_require_score_panel_scale_admission", lambda X_, sa: None)
    assert "purpose" not in pv.score_panel(X, Z, config=cfg, provenance={"t": "s"})


def test_stamp_does_not_change_any_scoring_output(monkeypatch):
    X, Z, cfg = _small_panel_inputs(7)
    monkeypatch.setattr(pv, "_require_score_panel_scale_admission", lambda X_, sa: None)
    base = pv.score_panel(X, Z, config=cfg, provenance={"t": "inv"})
    monkeypatch.setattr(pv, "_require_score_panel_scale_admission",
                        lambda X_, sa: {"purpose": SLIM.SLIM_CERT_PRODUCTION_PURPOSE,
                                        "identity_sha256": "a" * 64})
    stamped = pv.score_panel(X, Z, config=cfg, provenance={"t": "inv"})
    scoring_keys = ["schema", "formula_version", "n", "n_dims_hi", "n_dims_lo", "frac",
                    "k_hit", "k_frac", "k_density", "anchor_seed", "n_anchors", "anchor_hash",
                    "ffr", "recall@k", "n_ffr_anchors", "density", "n_density_anchors",
                    "guards"]
    assert all(base[key] == stamped[key] for key in scoring_keys)
    assert "purpose" not in base
    assert stamped["purpose"] == SLIM.SLIM_CERT_PRODUCTION_PURPOSE


# ---- the cert-production assembly logic (produce_slim_scale_cert) ---------- #


def test_assemble_slim_scale_cert_round_trips(slim):
    slim.monkeypatch.setattr(PSC, "validate_regression_certificate",
                             lambda report: {"passed": True})
    slim.monkeypatch.setattr(PSC, "validate_release_preflight_receipt",
                             lambda path, **kwargs: {"release_sha": RELEASE_SHA})
    out_path = str(slim.tmp_path / "assembled-cert.json")
    _out, cert = PSC.assemble_slim_scale_cert(
        release_sha=RELEASE_SHA, scientific_rows=16, row_derivation=slim.rd,
        panel_output_path=slim.panel_path, regression_path=slim.reg_path,
        release_preflight_receipt_path=slim.rec_path,
        release_preflight_identity_sha256="a" * 64, out_path=out_path)
    assert cert["schema"] == SLIM.SLIM_SCALE_CERT_SCHEMA
    validated = SLIM.validate_slim_scale_admission(
        _admission(out_path, slim.rd), X=slim.src, scientific_rows=16)
    assert validated["identity_sha256"] == cert["identity_sha256"]
    # assemble refuses a budget-exceeding measurement (fails closed at cert-production).
    bad_panel = _make_production_panel(slim.tmp_path, name="bad-panel.json", wall_s=99999.0)
    with pytest.raises(RuntimeError):
        PSC.assemble_slim_scale_cert(
            release_sha=RELEASE_SHA, scientific_rows=16, row_derivation=slim.rd,
            panel_output_path=bad_panel, regression_path=slim.reg_path,
            release_preflight_receipt_path=slim.rec_path,
            release_preflight_identity_sha256="a" * 64,
            out_path=str(slim.tmp_path / "bad-cert.json"))


# --------------------------------------------------------------------------- #
# purity is DESCRIPTIVE-only at 50M (amendment 2026-08-17, OWNER-authorized):
#   * purity k256/k1024 are scored on the R0237 first-PREFIX_ROWS prefix against a
#     reference + centroids built INLINE on that same prefix (hiD_reference=None), so
#     the number is self-contained on the prefix with NO R0218 dependency;
#   * collapse / fog / held-out FFR are the GATED metrics on the full 50M (unchanged
#     score_one_map path);
#   * purity is labelled descriptive/ungated with the lineage caveat and NEVER gates.
# CPU-only structural + unit fixtures (CUDA hidden at module top); no GPU, no real 50M.
#
# RETIRED here (moot per amendment 2026-08-17 — the descriptive reference is built on the
# prefix itself, so there is no cross-lineage identity claim to verify, and purity is no
# longer gated so the comparability-to-a-direct-2M-panel invariant no longer applies):
#   * test_two_m_prefix_reference_identity_reproduces_the_frozen_key
#     (its subject N.two_m_prefix_reference_identity was DROPPED from the round module);
#   * test_nested_prefix_refusal_plant_fires_on_a_mismatched_first_2m
#     (its subject N.verify_nested_prefix_identity was DROPPED from the round module);
#   * test_2m_prefix_purity_equals_a_direct_2m_panel
#     (tied to the superseded GATING semantics — purity no longer meets a 2M band).
# The two dropped helpers now live in experiments.produce_slim_scale_cert (the standing
# >=8M slim-cert producer that still mirrors the pre-amendment two-pass structure).
# --------------------------------------------------------------------------- #


def test_dropped_nested_prefix_helpers_are_gone_from_the_round_module():
    """The R0218-frozen-reference nested-prefix helpers are removed from the round module
    (moot per amendment 2026-08-17) and relocated to the standing slim-cert producer."""
    import experiments.produce_slim_scale_cert as PSC

    assert not hasattr(N, "two_m_prefix_reference_identity")
    assert not hasattr(N, "verify_nested_prefix_identity")
    # they still exist where the standing >=8M slim-cert producer needs them.
    assert hasattr(PSC, "two_m_prefix_reference_identity")
    assert hasattr(PSC, "verify_nested_prefix_identity")


def test_descriptive_purity_lineage_caveat_states_the_lineage_fact():
    caveat = N.DESCRIPTIVE_PURITY_LINEAGE_CAVEAT
    assert isinstance(caveat, str) and caveat
    low = caveat.lower()
    # the caveat names the descriptive/ungated status, the R0237-prefix self-contained
    # construction, and WHY it is not comparable to the R0265 2M family bands.
    assert "descriptive" in low and "ungated" in low
    assert "no r0218 dependency" in low or "self-contained" in low
    assert "different build lineage" in low
    assert "r0216-c3" in low
    assert N.PREFIX_ROWS == 2_000_000


def test_run_panel_descriptive_purity_is_prefix_inline_and_labelled():
    """Structural guard (amendment 2026-08-17): run_panel's ONLY score_panel call is the
    DESCRIPTIVE purity pass on the R0237 prefix — hiD_reference=None + prefix-fit centroids
    (no R0218 reference), carrying NO scale_admission; it is labelled descriptive/ungated
    with the lineage caveat.  collapse / fog / held-out FFR come from score_one_map on the
    FULL 50M coordinates."""
    import inspect

    src = inspect.getsource(N.run_panel)
    # collapse / fog / held-out FFR: score_one_map on the FULL 50M coordinates.
    assert "score_one_map(" in src
    assert "coordinates=coordinates," in src
    assert "probes_placed=placed," in src
    # PIECE A: the FLOOR-MATCHED FFR instrument — out-of-substrate reserve projection +
    # the reserve-neighbour truth + the N-scaled disc.  The old IN-SUBSTRATE probe design
    # (coordinates[probe_rows]) and its fixed-2000 disc are GONE.
    assert "truth_top10=reserve_truth," in src
    assert "proj_model.transform(reserve_embeddings" in src
    assert "disc=reserve_disc," in src
    assert "reserve_disc = int(ROWS * 0.001)" in src
    assert "coordinates[probe_rows]" not in src.replace(
        "# 0.1%·N; trip 10: the IN-SUBSTRATE coordinates[probe_rows] instead of the OUT-OF-", ""
    ).replace(
        '"floor. Supersedes the earlier IN-SUBSTRATE coordinates[probe_rows] + fixed "', ""
    )
    assert "truth_top10=truth_ids," not in src
    # run_panel makes EXACTLY ONE score_panel call -- the descriptive prefix purity pass.
    assert src.count("score_panel(") == 1
    # that pass builds the reference INLINE on the prefix (hiD_reference=None) with
    # prefix-fit centroids -> self-contained, no R0218 dependency.
    assert "source[:prefix_rows]" in src
    assert "coordinates[:prefix_rows]" in src
    assert "hiD_reference=None," in src
    assert "_build_prefix_purity_centroids(" in src
    assert 'purity_panel["purity"]' in src
    # it is labelled descriptive/ungated with the lineage caveat.
    assert '"descriptive": True' in src
    assert '"gated": False' in src
    assert "DESCRIPTIVE_PURITY_LINEAGE_CAVEAT" in src
    # the DROPPED R0218-frozen-reference path does not survive anywhere in run_panel.
    assert "verify_nested_prefix_identity" not in src
    assert "two_m_prefix_reference_identity" not in src
    assert "hiD_reference=reference," not in src
    assert "load_hiD_reference" not in src
    assert "panel_evidence" not in src
    # no >=8M scale_admission is carried anywhere in run_panel.
    assert "scale_admission=" not in src


def test_backstops_record_purity_descriptively_and_never_gate_on_it():
    """The per-seed backstops RECORD purity verdicts under `descriptive_purity` (gated
    False) but purity is NOT in `metrics` (the gated set) and never enters clears/straddle."""
    mt = {str(s): _seed_metrics() for s in T.SEEDS}
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    for row in bs["cells"]:
        # gated metrics are the three; purity lives only under descriptive_purity.
        assert set(row["metrics"]) == {"heldout_ffr", "collapse", "fog"}
        assert row["descriptive_purity"]["gated"] is False
        assert row["descriptive_purity"]["descriptive"] is True
        assert "purity_fidelity_k256" in row["descriptive_purity"]
        assert "purity_fidelity_k1024" in row["descriptive_purity"]
    # the straddle set only ever contains gated metrics.
    assert set(bs["straddled_gates"]).issubset(set(N.GATE_METRICS))
    assert "purity_fidelity_k256" not in bs["straddled_gates"]
    assert "purity_fidelity_k1024" not in bs["straddled_gates"]
    assert bs["descriptive_purity_bands_recorded"]["gated"] is False


def test_gate_verdict_ignores_failing_purity():
    """Coverage (b), amendment 2026-08-17: with collapse/fog/FFR healthy but purity FAR
    below the old floor+band, the go/no-go is unchanged — every seed still clears, no gate
    straddles, and the reconstructed 50M verdict is PASS.  Purity cannot flip the gate."""
    # healthy gated metrics, catastrophic purity (k256/k1024 far below the R0265 band/floor).
    mt = {str(s): _seed_metrics(collapse=0.97, fog=0.06, ffr=0.45, k256=0.01, k1024=0.01)
          for s in T.SEEDS}
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    # purity actually FAILED its (recorded, descriptive) bands — proving it is IGNORED, not
    # merely absent.
    for row in bs["cells"]:
        assert row["descriptive_purity"]["purity_fidelity_k256"]["passes"] is False
        assert row["descriptive_purity"]["purity_fidelity_k1024"]["passes"] is False
    # ...yet the gated go/no-go is untouched.
    assert bs["every_seed_clears_every_backstop"] is True
    assert bs["any_gate_straddles"] is False
    assert bs["any_fog_near_ceiling_escalation"] is False
    # reconstruct run_gate's verdict expression: criterion 1 passing + gated backstops clear.
    c1 = N.score_collapse_seed_mean(
        seed_collapse={s: mt[s]["collapse"] for s in mt},
        p1_lower=0.930, p1_upper=0.985, sigma_fam_collapse=0.06)
    passes = bool(
        c1["passes"]
        and bs["every_seed_clears_every_backstop"]
        and not bs["any_gate_straddles"]
    )
    verdict = ("50M_PASS" if passes and not bs["any_fog_near_ceiling_escalation"]
               else "50M_FAIL_OR_AMBIGUOUS")
    assert passes is True
    assert verdict == "50M_PASS"


def test_run_gate_records_descriptive_purity_and_gates_on_three_only():
    """Structural guard: run_gate records descriptive purity + the lineage caveat, computes
    the verdict from collapse/fog/FFR only, and asserts purity is descriptive-not-gated."""
    import inspect

    src = inspect.getsource(N.run_gate)
    assert '"descriptive_purity": descriptive_purity' in src
    assert "DESCRIPTIVE_PURITY_LINEAGE_CAVEAT" in src
    assert "purity_is_descriptive_not_gated" in src
    # the verdict is still the criterion-1 + gated-backstops expression (byte-identical).
    assert "backstop_scoring[\"every_seed_clears_every_backstop\"]" in src
    assert "backstop_scoring[\"any_gate_straddles\"]" in src


# --------------------------------------------------------------------------- #
# 9. PIECE A — the FLOOR-MATCHED FFR instrument (2026-08-17): run_panel measures
#    held-out FFR with the R0265 floor's own instrument — the OUT-OF-SUBSTRATE
#    reserve projection (model.transform(reserve)) against the sealed reserve-
#    neighbour truth, at the N-scaled disc = int(ROWS * 0.001). CPU-only structural
#    + a cheap real-scipy unit for the score_one_map disc forwarding. No 50M, no GPU.
# --------------------------------------------------------------------------- #

import experiments.round0265_nodes as R0265N
from basemap.artifact_identity import expected_input_signature as _sig


def test_score_one_map_forwards_disc_to_heldout_ffr():
    """score_one_map's new `disc` kwarg threads to heldout_ffr_scores; the default is
    R0265's FFR_DISC (byte-identical for every existing caller), a larger-N caller passes
    its own N-scaled disc.  A cheap real-cKDTree fixture (50 coords)."""
    rng = np.random.RandomState(0)
    coords = rng.randn(50, 2).astype("float32")
    placed = coords[:4] + 1e-4
    truth = np.tile(np.arange(3, dtype=np.int64), (4, 1))  # (4, 3) indices into coords
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
    # the disc is the ONLY thing changed — collapse/fog stay identical.
    assert out_default["collapse"] == out_disc["collapse"]
    assert out_default["fog"] == out_disc["fog"]


def test_run_panel_ffr_disc_is_n_scaled_not_the_fixed_2000():
    """The R0267 held-out-FFR disc is int(ROWS * 0.001) (0.1%·N), which SCALES with the
    substrate: 50000 at the 50M rung, NOT the fixed 2000 (trip 9)."""
    assert int(T.ROWS * 0.001) == 50_000
    assert int(T.ROWS * 0.001) != 2000
    # and it tracks a larger N (the 100M panel is born on the same rule).
    assert int(100_000_000 * 0.001) == 100_000
    # R0265's own FFR_DISC is the 2M value (0.1%·2M = 2000) — the rule, at its own N.
    assert R0265N.FFR_DISC == int(R0265N.ROWS * 0.001) == 2000


def test_run_panel_ffr_uses_out_of_substrate_reserve_projection():
    """Structural (PIECE A): run_panel projects the OUT-OF-SUBSTRATE reserve THROUGH each
    map for the floor-matched FFR, scores against the reserve-neighbour truth at the
    N-scaled disc, and binds the reserve + reserve-truth as sealed inputs.  The old
    IN-SUBSTRATE probe design (coordinates[probe_rows]) + fixed disc are GONE."""
    import inspect

    src = inspect.getsource(N.run_panel)
    # the floor-matched instrument.
    assert "reserve_disc = int(ROWS * 0.001)" in src
    assert "proj_model.transform(reserve_embeddings" in src
    assert "truth_top10=reserve_truth," in src
    assert "disc=reserve_disc," in src
    # reserve + reserve-truth are read as sealed panel inputs.
    assert '_bound_path(job, "heldout_reserve"' in src
    assert '_bound_path(job, "reserve_query_rows"' in src
    assert '_bound_path(job, "reserve_truth"' in src
    assert "reserve_all[reserve_query_rows]" in src
    # the OLD in-substrate probe design does not survive as executable code.
    code_only = "\n".join(
        line for line in src.splitlines()
        if not line.lstrip().startswith("#") and '"' not in line.split("=", 1)[0]
    )
    assert "coordinates[probe_rows]" not in code_only
    assert "truth_top10=truth_ids," not in src


def test_prepare_binds_the_reserve_and_reserve_truth_inputs():
    """PIECE A: prepare exposes the reserve-query-rows + reserve-neighbour-truth path
    constants and binds them into the panel job (structural — no heavy full-queue build)."""
    import inspect

    assert P.R0237_RESERVE_QUERY_ROWS.endswith("reserve-query-rows.i64.npy")
    assert P.R0267_RESERVE_NEIGHBOUR_TRUTH.endswith(
        "round-0267/ffr-correction/reserve-truth-50m/truth-top10.npy"
    )
    src = inspect.getsource(P.prepare_round0267)
    assert '_signature(R0237_RESERVE_QUERY_ROWS' in src
    assert '_signature(R0267_RESERVE_NEIGHBOUR_TRUTH' in src
    assert '"reserve_query_rows": reserve_query_rows,' in src
    assert '"reserve_truth": reserve_truth,' in src


# --------------------------------------------------------------------------- #
# 10. PIECE B — the GATE-ONLY FFR RE-SEAL (queue-correction-6): the gate reads the
#     CORRECTED floor-matched FFR from the bound re-score and SUPERSEDES correction-5's
#     mis-measured FFR, with collapse/fog byte-identical from the correction-5 panel and
#     purity descriptive; full superseding provenance (3 shas + trip-9/10 diagnosis +
#     "correction-5 FAIL stays"). CPU-only fixtures; no 50M, no GPU.
# --------------------------------------------------------------------------- #

_FLOOR = 0.39905871498653556  # the sealed R0265 held-out-FFR floor (unchanged)
#: The correction-5 panel's real collapse/fog + its mis-measured (~0.09) FFR, and the
#: corrected (~0.55) FFR from the sealed re-score.
_C5_COLLAPSE = {"42": 1.0860159998405134, "43": 1.0326237205997602, "44": 0.9233916709564777}
_C5_FOG = {"42": 0.2472015948320957, "43": 0.11650973419621417, "44": 0.16312232184635772}
_C5_OLD_FFR = {"42": 0.0920146, "43": 0.09308386666666667, "44": 0.0884056}
_C6_CORRECTED_FFR = {"42": 0.55935, "43": 0.5539, "44": 0.54945}


def _write_rescore(tmp_path, *, per_seed, disc=None, floor=_FLOOR, name="rescore-results.json"):
    disc = int(T.ROWS * 0.001) if disc is None else disc
    body = {
        "disc": disc, "floor": floor, "B": 1000, "boot_seed": 20260817,
        "per_map": {
            str(s): {"heldout_ffr": float(per_seed[str(s)]), "regressor_ffr": 0.49,
                     "net": 0.06, "verdict": "PASS"}
            for s in T.SEEDS
        },
    }
    path = str(tmp_path / name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(body, handle)
    return path, _sig(path)


def _write_reserve_truth(tmp_path, name="truth-top10.npy"):
    arr = np.zeros((2000, 10), dtype=np.int64)
    path = str(tmp_path / name)
    np.save(path, arr)
    return path, _sig(path)


def _c5_mt(ffr_map):
    return {
        s: _seed_metrics(collapse=_C5_COLLAPSE[s], fog=_C5_FOG[s], ffr=ffr_map[s],
                         k256=1.09, k1024=0.92)
        for s in (str(x) for x in T.SEEDS)
    }


def test_corrected_ffr_from_rescore_reads_bound_values(tmp_path):
    _, rescore_sig = _write_rescore(tmp_path, per_seed=_C6_CORRECTED_FFR)
    _, truth_sig = _write_reserve_truth(tmp_path)
    job = {"corrected_ffr_source": rescore_sig, "reserve_truth": truth_sig}
    out = N._corrected_ffr_from_rescore(job, backstops_floor=_FLOOR)
    assert out is not None
    assert out["gate_only_reseal"] is True
    assert out["corrected_ffr_disc"] == 50_000
    assert out["corrected_ffr_floor"] == pytest.approx(_FLOOR)
    for s in T.SEEDS:
        assert out["per_seed_corrected_ffr"][str(s)] == pytest.approx(_C6_CORRECTED_FFR[str(s)])
    assert out["rescore_results_sha256"] == rescore_sig["sha256"]
    assert out["reserve_neighbour_truth_sha256"] == truth_sig["sha256"]


def test_corrected_ffr_full_queue_path_returns_none():
    # No corrected_ffr_source bound => full-queue gate is byte-identical (helper is a no-op).
    assert N._corrected_ffr_from_rescore({}, backstops_floor=_FLOOR) is None


def test_corrected_ffr_trip9_disc_guard_fires(tmp_path):
    # a re-score at the FIXED 2000 disc (trip 9) is refused.
    _, rescore_sig = _write_rescore(tmp_path, per_seed=_C6_CORRECTED_FFR, disc=2000)
    with pytest.raises(N.Round0267NodeError):
        N._corrected_ffr_from_rescore({"corrected_ffr_source": rescore_sig}, backstops_floor=_FLOOR)


def test_corrected_ffr_floor_guard_fires(tmp_path):
    # a re-score whose floor disagrees with the sealed R0265 floor is refused.
    _, rescore_sig = _write_rescore(tmp_path, per_seed=_C6_CORRECTED_FFR, floor=0.10)
    with pytest.raises(N.Round0267NodeError):
        N._corrected_ffr_from_rescore({"corrected_ffr_source": rescore_sig}, backstops_floor=_FLOOR)


def test_corrected_ffr_requires_the_three_seeds(tmp_path):
    body = {"disc": int(T.ROWS * 0.001), "floor": _FLOOR,
            "per_map": {"42": {"heldout_ffr": 0.55}, "43": {"heldout_ffr": 0.55}}}
    path = str(tmp_path / "partial.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(body, handle)
    with pytest.raises(N.Round0267NodeError):
        N._corrected_ffr_from_rescore({"corrected_ffr_source": _sig(path)}, backstops_floor=_FLOOR)


def test_corrected_ffr_tamper_evident(tmp_path):
    _, rescore_sig = _write_rescore(tmp_path, per_seed=_C6_CORRECTED_FFR)
    with open(rescore_sig["canonical_path"], "a", encoding="utf-8") as handle:
        handle.write("  ")  # mutate the bytes -> the bound digest no longer matches
    with pytest.raises(Exception):
        N._corrected_ffr_from_rescore({"corrected_ffr_source": rescore_sig}, backstops_floor=_FLOOR)


def test_gate_only_verdict_is_pass_with_corrected_ffr_and_fail_with_the_old():
    """The superseding decision: correction-5's collapse/fog are healthy but its
    mis-measured FFR ~0.09 FAILS the floor (verdict FAIL); the corrected ~0.55 FFR CLEARS
    it, so the reconstructed 50M verdict is PASS.  Uses the synthetic bands + real
    correction-5 collapse/fog + real corrected/old FFR."""
    # collapse seed-mean is inside the widened P1 band (collapse is byte-identical either way).
    c1 = N.score_collapse_seed_mean(
        seed_collapse=_C5_COLLAPSE, p1_lower=0.930, p1_upper=0.985, sigma_fam_collapse=0.06)
    assert c1["passes"] is True

    # OLD (mis-measured ~0.09) FFR -> every seed FAILS the FFR floor -> FAIL.
    bs_old = N.score_per_seed_backstops(
        metric_table=_c5_mt(_C5_OLD_FFR), backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    for row in bs_old["cells"]:
        assert row["metrics"]["heldout_ffr"]["passes"] is False
    assert bs_old["every_seed_clears_every_backstop"] is False
    verdict_old = ("50M_PASS" if (c1["passes"]
                   and bs_old["every_seed_clears_every_backstop"]
                   and not bs_old["any_gate_straddles"]
                   and not bs_old["any_fog_near_ceiling_escalation"])
                   else "50M_FAIL_OR_AMBIGUOUS")
    assert verdict_old == "50M_FAIL_OR_AMBIGUOUS"

    # CORRECTED (~0.55) FFR -> every seed CLEARS -> PASS. collapse/fog UNCHANGED.
    bs_new = N.score_per_seed_backstops(
        metric_table=_c5_mt(_C6_CORRECTED_FFR), backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    for row in bs_new["cells"]:
        assert row["metrics"]["heldout_ffr"]["passes"] is True
    assert bs_new["every_seed_clears_every_backstop"] is True
    assert bs_new["any_gate_straddles"] is False
    assert bs_new["any_fog_near_ceiling_escalation"] is False
    verdict_new = ("50M_PASS" if (c1["passes"]
                   and bs_new["every_seed_clears_every_backstop"]
                   and not bs_new["any_gate_straddles"]
                   and not bs_new["any_fog_near_ceiling_escalation"])
                   else "50M_FAIL_OR_AMBIGUOUS")
    assert verdict_new == "50M_PASS"
    # collapse & fog are byte-identical between the two runs (only FFR moved).
    for s in (str(x) for x in T.SEEDS):
        assert _c5_mt(_C5_OLD_FFR)[s]["collapse"] == _c5_mt(_C6_CORRECTED_FFR)[s]["collapse"]
        assert _c5_mt(_C5_OLD_FFR)[s]["fog"] == _c5_mt(_C6_CORRECTED_FFR)[s]["fog"]


def test_gate_only_override_then_backstops_flip_to_pass(tmp_path):
    """End-to-end of the gate's override step: read the corrected FFR from the bound
    re-score, OVERRIDE the panel's mis-measured FFR column, re-score the backstops -> PASS.
    Mirrors run_gate's supersede-then-score sequence without a real panel/GPU."""
    _, rescore_sig = _write_rescore(tmp_path, per_seed=_C6_CORRECTED_FFR, floor=_SYNTH_BACKSTOPS["heldout_ffr_floor"])
    corr = N._corrected_ffr_from_rescore(
        {"corrected_ffr_source": rescore_sig},
        backstops_floor=_SYNTH_BACKSTOPS["heldout_ffr_floor"])
    mt = _c5_mt(_C5_OLD_FFR)
    # the gate's override: collapse/fog untouched, FFR column replaced.
    superseded = {s: mt[s]["heldout_ffr"] for s in mt}
    for s in T.SEEDS:
        mt[str(s)]["heldout_ffr"] = corr["per_seed_corrected_ffr"][str(s)]
    assert all(superseded[str(s)] < 0.1 for s in T.SEEDS)          # was mis-measured
    assert all(mt[str(s)]["heldout_ffr"] > 0.5 for s in T.SEEDS)   # now corrected
    bs = N.score_per_seed_backstops(metric_table=mt, backstops=_SYNTH_BACKSTOPS, sigma_fam_fog=0.02)
    assert bs["every_seed_clears_every_backstop"] is True


def test_trip_9_10_diagnosis_states_both_trips_and_both_fixes():
    d = N.TRIP_9_10_DIAGNOSIS.lower()
    # trip 9 — the discovery-radius axis (fixed 2000 vs N-scaled 0.1%·N).
    assert "trip 9" in d and "disc" in d and "2000" in d and "50000" in d
    # trip 10 — the probe-design axis (in-substrate vs out-of-substrate reserve).
    assert "trip 10" in d and "out-of-" in d and "coordinates[probe_rows]" in d
    assert "reserve" in d
    # and the two fixes: the gate-only supersede + PIECE A born-correct.
    assert "supersed" in d and "piece a" in d
    assert N.SUPERSEDED_SOURCE_RUN == "queue-correction-5"
    assert N.SUPERSEDED_VERDICT == "50M_FAIL_OR_AMBIGUOUS"


def test_run_gate_records_superseding_provenance_and_binds_corrected_ffr():
    """Structural (PIECE B): run_gate reads the corrected FFR from the bound re-score,
    OVERRIDES the panel's FFR column (collapse/fog untouched), and records the superseding
    provenance — the three shas, the trip-9/10 diagnosis, and that correction-5's FAIL
    STAYS.  The criterion-1 + gated-backstop verdict arithmetic is byte-identical."""
    import inspect

    src = inspect.getsource(N.run_gate)
    # the corrected FFR is read from the bound re-score and overrides only the FFR column.
    assert "_corrected_ffr_from_rescore(" in src
    assert "metric_table[str(s)][HELDOUT_FFR_METRIC] = (" in src
    # collapse/fog are NOT recomputed — they come from the bound panel (byte-identical).
    assert "collapse_fog_bound_from_correction_5_panel" in src
    assert "corrected_ffr_sourced_from_bound_rescore" in src
    # the superseding provenance: three shas + diagnosis + FAIL-stays.
    assert "TRIP_9_10_DIAGNOSIS" in src
    assert '"correction_5_panel_sha256"' in src
    assert '"reserve_neighbour_truth_sha256"' in src
    assert '"rescore_results_sha256"' in src
    assert "SUPERSEDED_SOURCE_RUN" in src and "SUPERSEDED_VERDICT" in src
    assert "STAYS" in src
    assert "superseding_provenance_records_three_shas_and_fail_stays" in src
    # the verdict arithmetic (criterion 1 + gated backstops + no-straddle) is unchanged.
    assert "criterion_1[\"passes\"]" in src
    assert "backstop_scoring[\"every_seed_clears_every_backstop\"]" in src
    assert "backstop_scoring[\"any_gate_straddles\"]" in src


# ---- the gate-only re-seal QUEUE build (queue-correction-6) ---------------- #

_GATE_ONLY_B_INPUTS = [
    P.CORRECTION_5_PANEL, P.FFR_RESCORE_RESULTS, P.R0267_RESERVE_NEIGHBOUR_TRUTH,
    P.R0237_SUBSTRATE_MANIFEST, P.R0237_GRAPH_MANIFEST, P.R0265_FLOORS, P.R0265_PANEL,
    P.P1_ASYMPTOTE, P.ROUND_FILE,
]
_GATE_ONLY_B_READY = (
    all(os.path.exists(path) for path in _GATE_ONLY_B_INPUTS)
    and os.path.exists(os.path.join(P.RELEASE_ROOT, ".git"))
)


def _release_head_b() -> str:
    import subprocess

    proc = subprocess.run(
        ["git", "-C", P.RELEASE_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


def _rmtree_ro_b(root: str) -> None:
    import shutil

    def _onerror(func, path, _exc):
        try:
            os.chmod(path, 0o700)
            func(path)
        except OSError:
            pass

    shutil.rmtree(root, onerror=_onerror)


@pytest.mark.skipif(not _GATE_ONLY_B_READY, reason="gate-only re-seal inputs absent")
def test_gate_only_prepare_builds_a_one_node_gate_queue_correction_6():
    import time

    release_sha = _release_head_b()
    assert re.fullmatch(r"[0-9a-f]{40}", release_sha)
    # A distinct /data dir — NOT the real queue-correction-6, so nothing in the record is
    # touched; cleaned up in the finally.
    queue_root = os.path.join(
        P.ROUND_ROOT, f"queue-corr6-pytest-{os.getpid()}-{int(time.time() * 1000) % 100000}"
    )
    try:
        manifest_path = P.prepare_round0267_gate_only(
            release_sha=release_sha,
            source_run="queue-correction-5",
            source_release="0" * 40,
            queue_root=queue_root,
        )
        with open(manifest_path, encoding="utf-8") as handle:
            queue = json.load(handle)

        # exactly ONE node — the gate — with no in-queue dependency (no train/panel node).
        assert len(queue["jobs"]) == 1
        gate = queue["jobs"][0]
        assert gate["id"] == N.GATE_ACTION and gate["action"] == N.GATE_ACTION
        assert gate["deps"] == []
        assert queue["capabilities_produced"] == [N.GATE_CAPABILITY]
        assert queue["schema"] == "round0267-fneg-50m-x2-seedmean-gate-only-queue-v1"

        # the three superseding inputs are bound by CONTENT DIGEST (sha256 + bytes).
        panel_sig = _sig(P.CORRECTION_5_PANEL)
        truth_sig = _sig(P.R0267_RESERVE_NEIGHBOUR_TRUTH)
        rescore_sig = _sig(P.FFR_RESCORE_RESULTS)
        assert gate["panel"]["sha256"] == panel_sig["sha256"]
        assert gate["reserve_truth"]["sha256"] == truth_sig["sha256"]
        assert gate["corrected_ffr_source"]["sha256"] == rescore_sig["sha256"]
        # all three are in expected_inputs (tamper-evident at run).
        expected_shas = {inp["sha256"] for inp in gate["expected_inputs"]}
        assert {panel_sig["sha256"], truth_sig["sha256"], rescore_sig["sha256"]} <= expected_shas

        # the provenance records the three shas, the superseded verdict, and FAIL-stays.
        prov = queue["gate_only_provenance"]
        assert prov["gate_only_reseal"] is True
        assert prov["correction_5_panel_sha256"] == panel_sig["sha256"]
        assert prov["reserve_neighbour_truth_sha256"] == truth_sig["sha256"]
        assert prov["rescore_results_sha256"] == rescore_sig["sha256"]
        assert prov["superseded_verdict"] == "50M_FAIL_OR_AMBIGUOUS"
        assert "STAYS" in prov["note"]
        assert prov["corrected_ffr_disc"] == 50_000
        # the corrected per-seed FFR carried for provenance matches the sealed re-score.
        rescore = json.load(open(P.FFR_RESCORE_RESULTS))
        for s in T.SEEDS:
            assert prov["per_seed_corrected_ffr"][str(s)] == pytest.approx(
                float(rescore["per_map"][str(s)]["heldout_ffr"]))
    finally:
        if os.path.isdir(queue_root):
            _rmtree_ro_b(queue_root)
