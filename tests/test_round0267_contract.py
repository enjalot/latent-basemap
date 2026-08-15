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


def test_gate_metrics_are_the_five():
    assert set(N.GATE_METRICS) == {
        "heldout_ffr", "purity_fidelity_k256", "purity_fidelity_k1024", "collapse", "fog",
    }


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
