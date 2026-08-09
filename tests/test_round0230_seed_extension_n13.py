"""Registered checks for the R0230 n=13 seed extension and panel pooling.

The load-bearing tests here are the ones that would have caught the defects the
program keeps rediscovering:

* the thirteen cells are R0217's treatment, and that is checked **below the
  digest** — by restoring the nine seed-bearing paths and comparing canonical
  JSON bytes — so a digest that happened to agree could not hide a moved field;
* at `n = 13` the identity bound `(n-1)/sqrt(n) = 3.3282` exceeds every `mean-k*s`
  multiplier in play, so a defining cell **can** fail. The test asserts the
  inequality in both directions: it holds at `n = 13` and fails at `n = 4` and
  `n = 8` under `k = 3.187`;
* the high-D reference identity aborts on every one of its five components, not
  just on the file hash;
* the predictive guard refuses a cell whose prediction breaches a budget, and the
  refusal is recorded rather than silently skipped.
"""
from __future__ import annotations

import copy
import math

import pytest

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0217_minilm_2m_seed_family import (
    SEEDS as R0217_SEEDS,
    expected_seed_bearing_values as r0217_seed_bearing_values,
    train_config as r0217_train_config,
)
from basemap.round0221_minilm_2m_seed_extension import SEEDS as R0221_SEEDS
from basemap.round0230_minilm_2m_panel_n13 import (
    ANCHOR_CORPUS_COUNTS,
    HI_D_AGREEMENT,
    PANEL_METRICS,
    POOLED_CELL_SOURCES,
    REFERENCE_CONTENT_SHA256,
    REFERENCE_KEY,
    REFERENCE_SHA256,
    REFERENCE_BYTES,
    Round0230PanelError,
    assert_hi_d_agreement,
    assert_reference_identity,
    pool_thirteen_cells,
    raw_purity_ratios,
)
from basemap.round0230_minilm_2m_seed_extension_n13 import (
    DEVICE_BUDGET_BYTES,
    HOST_ANON_BUDGET_BYTES,
    IDENTITY_BOUND_AT_N13,
    N_TARGET,
    POOLED_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    REGISTERED_ACHIEVED_DRAWS_PER_EDGE,
    REGISTERED_SUCCESSFUL_UPDATES,
    ROWS,
    Round0230Error,
    SEALED_DIRECTED_EDGES,
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
    SEEDS,
    SEED_BEARING_PATHS,
    TEMPLATE_SEED,
    assert_extension_differs_only_by_seed,
    assert_reconstructs_r0217_template,
    capability_for_seed,
    identity_bound,
    masked_config_bytes,
    predict_cell_footprint,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config,
    validate_registered_dose,
)


def _configs() -> dict[int, dict]:
    return {
        seed: train_config(
            seed=seed,
            graph_signature=dict(SEALED_GRAPH_SIGNATURE),
            graph_manifest_signature=dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
            substrate_signature=dict(SEALED_SUBSTRATE_SIGNATURE),
            graph_edges=SEALED_DIRECTED_EDGES,
            rows=ROWS,
        )[0]
        for seed in SEEDS
    }


def _template() -> dict:
    return r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=dict(SEALED_GRAPH_SIGNATURE),
        graph_manifest_signature=dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
        substrate_signature=dict(SEALED_SUBSTRATE_SIGNATURE),
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )[0]


# --------------------------------------------------------------------------- #
# the family
# --------------------------------------------------------------------------- #


def test_the_pooled_family_is_thirteen_distinct_seeds() -> None:
    assert SEEDS == (50, 51, 52, 53, 54)
    assert POOLED_SEEDS == tuple(R0217_SEEDS) + tuple(R0221_SEEDS) + SEEDS
    assert len(POOLED_SEEDS) == len(set(POOLED_SEEDS)) == 13 == N_TARGET
    assert set(POOLED_CELL_SOURCES) == {str(seed) for seed in POOLED_SEEDS}


def test_every_cell_carries_r0217s_published_seed_invariant_digest() -> None:
    family = assert_extension_differs_only_by_seed(
        _configs(), expected_seed_invariant=R0217_SEED_INVARIANT_SHA256
    )
    assert family["seed_invariant_sha256"] == R0217_SEED_INVARIANT_SHA256
    assert family["matches_r0217_published_seed_invariant"] is True
    assert len(set(family["per_seed_config_sha256"].values())) == len(SEEDS)


def test_the_masked_bytes_are_publishable_and_hash_to_the_digest() -> None:
    """The masker is checkable below the digest, as review-0221's reviewer did."""
    configs = _configs()
    blobs = {seed: masked_config_bytes(config) for seed, config in configs.items()}
    assert len({bytes(blob) for blob in blobs.values()}) == 1
    for seed, blob in blobs.items():
        assert sha256_bytes(blob) == seed_invariant_sha256(configs[seed])
        assert sha256_bytes(blob) == R0217_SEED_INVARIANT_SHA256
    family = assert_extension_differs_only_by_seed(configs)
    for seed in SEEDS:
        entry = family["masked_config_identity"][str(seed)]
        assert entry["masked_config_bytes"] == len(blobs[seed])
        assert entry["masked_config_sha256"] == R0217_SEED_INVARIANT_SHA256


def test_each_cell_reconstructs_r0217s_canonical_config_byte_for_byte() -> None:
    template = _template()
    for seed, config in _configs().items():
        proof = assert_reconstructs_r0217_template(config, template)
        assert proof["byte_equal"] is True
        assert proof["reconstructed_sha256"] == proof["r0217_template_sha256"]
        assert proof["bytes"] == len(canonical_json(template))
        assert len(proof["seed_bearing_paths_restored"]) == len(SEED_BEARING_PATHS)


def test_a_field_outside_the_nine_seed_bearing_paths_breaks_reconstruction() -> None:
    """A digest could collide; byte equality of the restored config cannot."""
    template = _template()
    config = copy.deepcopy(_configs()[50])
    config["execution"]["minimum_train_upd_s"] = (
        float(config["execution"]["minimum_train_upd_s"]) + 1.0
    )
    with pytest.raises(Round0230Error):
        assert_reconstructs_r0217_template(config, template)


def test_only_the_nine_registered_paths_differ_from_the_template() -> None:
    template = _template()
    for seed, config in _configs().items():
        differing = []
        for path in SEED_BEARING_PATHS:
            mine = config
            theirs = template
            for key in path:
                mine = mine[key]
                theirs = theirs[key]
            if mine != theirs:
                differing.append(".".join(path))
        # seed 50-54 differ from seed 42 in all nine
        assert len(differing) == len(SEED_BEARING_PATHS)
        for path, want in r0217_seed_bearing_values(TEMPLATE_SEED).items():
            cursor = template
            for key in path:
                cursor = cursor[key]
            assert cursor == want


def test_a_foreign_substrate_signature_is_refused_before_the_config_exists() -> None:
    bad = dict(SEALED_SUBSTRATE_SIGNATURE)
    bad["sha256"] = "0" * 64
    with pytest.raises(Round0230Error):
        train_config(
            seed=50,
            graph_signature=dict(SEALED_GRAPH_SIGNATURE),
            graph_manifest_signature=dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
            substrate_signature=bad,
            graph_edges=SEALED_DIRECTED_EDGES,
            rows=ROWS,
        )


def test_seeds_outside_the_registered_five_are_refused() -> None:
    for seed in (42, 46, 49, 55, 99):
        with pytest.raises(Round0230Error):
            capability_for_seed(seed)


def test_capabilities_follow_r0217s_template() -> None:
    for seed in SEEDS:
        assert capability_for_seed(seed) == (
            f"minilm-mixed-2m-map-seed{seed}-low-dose-v1"
        )


# --------------------------------------------------------------------------- #
# the point of the round: at n = 13 a defining cell CAN fail
# --------------------------------------------------------------------------- #


def test_the_identity_bound_at_n13_exceeds_every_multiplier_in_play() -> None:
    assert identity_bound(13) == pytest.approx(12 / math.sqrt(13), abs=0.0)
    assert IDENTITY_BOUND_AT_N13 == identity_bound(N_TARGET)
    assert IDENTITY_BOUND_AT_N13 == pytest.approx(3.328201177351375, abs=1e-12)
    # mean - 2s, the one-sided 95/95 factor at n=13, and Howe's two-sided factor
    for multiplier in (2.0, 2.670504, 3.100799):
        assert IDENTITY_BOUND_AT_N13 > multiplier


def test_the_same_inequality_fails_at_n4_and_at_n8_under_k_3_187() -> None:
    """Why R0219's '4/4 pass' and R0225's '0 failures' were theorems."""
    assert identity_bound(4) == pytest.approx(1.5, abs=1e-12)
    assert identity_bound(4) < 2.0
    assert identity_bound(8) == pytest.approx(2.4748737341529163, abs=1e-12)
    assert identity_bound(8) > 2.0          # R0222's n=8 gate could be failed
    assert identity_bound(8) < 3.187        # R0225's n=8 tolerance gate could not
    assert identity_bound(6) > 2.0 > identity_bound(5)


def test_a_family_of_thirteen_can_actually_place_a_cell_below_mean_minus_2sd() -> None:
    """Not an argument about bounds: an explicit witness at n = 13."""
    import statistics

    values = [1.0] * 12 + [0.0]
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    assert max(abs(value - mean) for value in values) / sd == pytest.approx(
        IDENTITY_BOUND_AT_N13, rel=1e-12
    )
    assert min(values) < mean - 2.0 * sd
    assert min(values) < mean - 2.670504 * sd
    assert min(values) < mean - 3.100799 * sd


# --------------------------------------------------------------------------- #
# dose
# --------------------------------------------------------------------------- #


def test_the_dose_lands_on_the_registered_ceil_derived_value() -> None:
    updates = successful_updates_for_edges(SEALED_DIRECTED_EDGES)
    assert updates == REGISTERED_SUCCESSFUL_UPDATES
    dose = validate_registered_dose(
        updates=updates, edge_count=SEALED_DIRECTED_EDGES
    )
    assert dose["landed_on_registered_ceil_value"] is True
    assert dose["achieved_positive_draws_per_edge"] == (
        REGISTERED_ACHIEVED_DRAWS_PER_EDGE
    )


def test_a_different_edge_count_is_refused() -> None:
    with pytest.raises(Round0230Error):
        validate_registered_dose(
            updates=REGISTERED_SUCCESSFUL_UPDATES,
            edge_count=SEALED_DIRECTED_EDGES + 1,
        )


# --------------------------------------------------------------------------- #
# the predictive memory guard
# --------------------------------------------------------------------------- #


def test_every_cell_is_predicted_and_none_is_refused_a_priori() -> None:
    for seed in SEEDS:
        prediction = predict_cell_footprint(seed)
        assert prediction["seed"] == seed
        assert prediction["refused_a_priori"] is False
        assert prediction["predicted_peak_device_bytes"] < DEVICE_BUDGET_BYTES
        assert (
            prediction["predicted_peak_host_anonymous_bytes"] < HOST_ANON_BUDGET_BYTES
        )
        assert prediction["predicted_device_headroom_bytes"] > 0
        assert prediction["predicted_host_headroom_bytes"] > 0


def test_the_guard_can_actually_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """A guard that cannot refuse is not a guard."""
    import basemap.round0230_minilm_2m_seed_extension_n13 as contract

    monkeypatch.setattr(contract, "MEASURED_PEAK_DEVICE_BYTES", 20 * 1024 ** 3)
    prediction = contract.predict_cell_footprint(50)
    assert prediction["device_budget_exceeded"] is True
    assert prediction["refused_a_priori"] is True
    monkeypatch.setattr(contract, "MEASURED_PEAK_DEVICE_BYTES", 796_540_416)
    monkeypatch.setattr(contract, "MEASURED_PEAK_HOST_RSS_GIB", 40.0)
    prediction = contract.predict_cell_footprint(51)
    assert prediction["host_budget_exceeded"] is True
    assert prediction["refused_a_priori"] is True


# --------------------------------------------------------------------------- #
# the frozen panel
# --------------------------------------------------------------------------- #


def _reference_kwargs() -> dict:
    return {
        "file_signature": {
            "kind": "file",
            "canonical_path": "/x",
            "bytes": REFERENCE_BYTES,
            "sha256": REFERENCE_SHA256,
        },
        "key": REFERENCE_KEY,
        "content_sha256": REFERENCE_CONTENT_SHA256,
        "rederived_key": REFERENCE_KEY,
        "anchor_corpus_counts": dict(ANCHOR_CORPUS_COUNTS),
    }


def test_the_reference_identity_passes_on_r0218s_published_values() -> None:
    receipt = assert_reference_identity(**_reference_kwargs())
    assert receipt["reference_byte_identical_to_r0218"] is True
    assert receipt["anchors"] == 4_000


@pytest.mark.parametrize(
    "mutate",
    [
        lambda kw: kw["file_signature"].__setitem__("bytes", REFERENCE_BYTES + 1),
        lambda kw: kw["file_signature"].__setitem__("sha256", "0" * 64),
        lambda kw: kw.__setitem__("key", "0" * 64),
        lambda kw: kw.__setitem__("content_sha256", "0" * 64),
        lambda kw: kw.__setitem__("rederived_key", "0" * 64),
        lambda kw: kw["anchor_corpus_counts"].__setitem__("code", 444),
    ],
)
def test_every_component_of_the_reference_identity_can_stop_the_round(mutate) -> None:
    kwargs = _reference_kwargs()
    mutate(kwargs)
    with pytest.raises(Round0230PanelError):
        assert_reference_identity(**kwargs)


def test_the_hi_d_agreement_numerators_must_be_r0218s() -> None:
    good = {key: {"hi_D_agreement": value} for key, value in HI_D_AGREEMENT.items()}
    assert assert_hi_d_agreement(50, good) == dict(HI_D_AGREEMENT)
    bad = {key: {"hi_D_agreement": value} for key, value in HI_D_AGREEMENT.items()}
    bad["k256"] = {"hi_D_agreement": 0.3829}
    with pytest.raises(Round0230PanelError):
        assert_hi_d_agreement(50, bad)


def test_raw_ratios_are_the_unfolded_ones_and_a_zero_denominator_aborts() -> None:
    assert raw_purity_ratios({"purity": {"k256": 1.0370, "k1024": 0.7266}}) == {
        "k256": 1.0370,
        "k1024": 0.7266,
    }
    with pytest.raises(Round0230PanelError):
        raw_purity_ratios({"purity": {"k256": None, "k1024": 0.7}})
    with pytest.raises(Round0230PanelError):
        raw_purity_ratios({"purity": {"k256": 0.0, "k1024": 0.7}})


def _pool_inputs():
    cells = {
        str(seed): {metric: 0.5 for metric in PANEL_METRICS} for seed in POOLED_SEEDS
    }
    ratios = {str(seed): {"k256": 1.01, "k1024": 0.71} for seed in POOLED_SEEDS}
    corpus = {
        str(seed): {
            slug: {"anchors": count, "ffr": 0.33}
            for slug, count in ANCHOR_CORPUS_COUNTS.items()
        }
        for seed in POOLED_SEEDS
    }
    return cells, ratios, corpus


def test_pooling_thirteen_cells_registers_no_floor() -> None:
    cells, ratios, corpus = _pool_inputs()
    pooled = pool_thirteen_cells(
        cells=cells, ratios=ratios, corpus=corpus, sources=POOLED_CELL_SOURCES
    )
    assert pooled["n"] == 13
    assert pooled["gate_registerable_here"] is False
    assert pooled["identity_bound_at_n"] == IDENTITY_BOUND_AT_N13
    assert set(pooled["raw_purity_ratios"]) == {str(s) for s in POOLED_SEEDS}
    assert "DESCRIPTIVE ONLY" in pooled["density_v2_status"]


def test_pooling_refuses_a_missing_or_foreign_cell() -> None:
    cells, ratios, corpus = _pool_inputs()
    del cells["54"]
    with pytest.raises(Round0230PanelError):
        pool_thirteen_cells(
            cells=cells, ratios=ratios, corpus=corpus, sources=POOLED_CELL_SOURCES
        )
    cells, ratios, corpus = _pool_inputs()
    cells["55"] = {metric: 0.5 for metric in PANEL_METRICS}
    with pytest.raises(Round0230PanelError):
        pool_thirteen_cells(
            cells=cells, ratios=ratios, corpus=corpus, sources=POOLED_CELL_SOURCES
        )


def test_pooling_refuses_an_inadmissible_metric() -> None:
    cells, ratios, corpus = _pool_inputs()
    cells["50"]["ffr"] = 1.5
    with pytest.raises(Round0230PanelError):
        pool_thirteen_cells(
            cells=cells, ratios=ratios, corpus=corpus, sources=POOLED_CELL_SOURCES
        )
