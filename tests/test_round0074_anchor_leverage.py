from __future__ import annotations

import inspect

import numpy as np

from experiments import prepare_round0074_queue, round0074_nodes


ELIGIBILITY = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "balanced-30m-int8-substrate/"
    "minilm-balanced-30m-int8-row-eligibility-v1.npz"
)
R0019_REFERENCE = (
    "/data/latent-basemap/runs/round-0019/queue/artifacts/"
    "high-d-reference/reference.npz"
)


def test_anchor_family_mapping_handles_representatives_copies_and_singletons() -> None:
    eligibility = {
        "duplicate_excluded_rows": np.asarray([4, 5, 9]),
        "duplicate_representative_rows": np.asarray([2, 2, 8]),
        "representative_rows": np.asarray([2, 8]),
        "family_counts": np.asarray([3, 2]),
    }
    canonical, counts = round0074_nodes.map_anchor_families(
        np.asarray([1, 2, 4, 5, 8, 9, 10]),
        eligibility,
    )
    assert canonical.tolist() == [1, 2, 2, 2, 8, 8, 10]
    assert counts.tolist() == [1, 3, 3, 3, 2, 2, 1]


def test_density_summary_attributes_full_sample_covariance() -> None:
    high = np.linspace(1.0, 10.0, 10_000)
    low = high * 2.0
    family_count = np.ones(10_000, dtype=np.int64)
    family_count[-20:] = 100
    canonical = np.arange(10_000, dtype=np.int64)
    canonical[-10:] = canonical[-20:-10]
    labels = np.repeat(
        np.asarray(["fineweb", "redpajama", "pile"], dtype="<U10"),
        [3_334, 3_333, 3_333],
    )
    summary = round0074_nodes.density_leverage_summary(
        high,
        low,
        family_count,
        canonical,
        labels,
    )
    assert summary["all"]["correlation"] == 1.0
    assert (
        summary["anchor_population_sensitivity"][
            "exclude_family_ge_16"
        ]["anchors"]
        == 9_980
    )
    assert (
        summary["anchor_population_sensitivity"][
            "one_anchor_per_canonical_family"
        ]["anchors"]
        == 9_990
    )
    assert (
        summary["duplicate_group_attribution"]["family_ge_16"]["anchors"]
        == 20
    )
    assert sum(
        value["anchors"] for value in summary["by_corpus"].values()
    ) == 10_000


def test_registered_classification_requires_both_fixed_model_bridges() -> None:
    def cell(full: float, without: float, fraction: float) -> dict:
        return {
            "all": {"correlation": full},
            "anchor_population_sensitivity": {
                "exclude_family_ge_16": {"correlation": without},
            },
            "duplicate_group_attribution": {
                "family_ge_16": {
                    "full_sample_covariance_numerator_fraction": fraction,
                },
            },
        }

    supported = round0074_nodes.classify_anchor_leverage(
        replay_exact=True,
        legacy=cell(0.78, 0.10, 0.91),
        modern=cell(0.76, 0.11, 0.88),
        representative_anchor_cells={
            "legacy_original": 0.08,
            "modern_original": 0.10,
        },
    )
    assert (
        supported["classification"]
        == "duplicate-heavy-anchor-leverage-supported"
    )
    assert supported["calibrates_density_threshold"] is False
    assert supported["authorizes_larger_training_rung"] is False

    inconclusive = round0074_nodes.classify_anchor_leverage(
        replay_exact=True,
        legacy=cell(0.30, 0.15, 0.80),
        modern=cell(0.31, 0.10, 0.80),
        representative_anchor_cells={
            "legacy_original": 0.10,
            "modern_original": 0.10,
        },
    )
    assert (
        inconclusive["classification"]
        == "duplicate-heavy-anchor-leverage-inconclusive"
    )


def test_registered_r0019_anchor_family_facts_reproduce() -> None:
    with np.load(ELIGIBILITY, allow_pickle=False) as eligibility:
        arrays = {name: np.asarray(eligibility[name]) for name in (
            "duplicate_excluded_rows",
            "duplicate_representative_rows",
            "representative_rows",
            "family_counts",
        )}
    with np.load(R0019_REFERENCE, allow_pickle=False) as reference:
        anchors = np.asarray(reference["anchor_ids"], dtype=np.int64)
        high_radius = np.asarray(reference["r_hd"], dtype=np.float64)
    canonical, family_count = round0074_nodes.map_anchor_families(
        anchors,
        arrays,
    )
    assert np.count_nonzero(family_count > 1) == 124
    assert np.count_nonzero(family_count >= 16) == 20
    assert len(np.unique(canonical)) == 9_983
    assert family_count.max() == 30_088
    assert np.count_nonzero(high_radius == 0) == 20
    assert np.all(high_radius[family_count >= 16] == 0)


def test_queue_is_one_bounded_no_training_gpu_node() -> None:
    source = inspect.getsource(prepare_round0074_queue.prepare_round0074)
    assert source.count('"action": "anchor_leverage"') == 1
    assert "gpu_hours_cap=0.10" in source
    assert '"p90_wall_s": 120.0' in source
    assert '"training_performed": False' in source
    assert '"authorizes_larger_training_rung": False' in source


def test_node_holds_anchor_and_candidate_universe_fixed() -> None:
    source = inspect.getsource(round0074_nodes.run_anchor_leverage)
    assert source.count("_self_knn(") == 1
    assert '("legacy_r0019", legacy_coordinates)' in source
    assert '("modern_r0061", modern_coordinates)' in source
    assert "anchors," in source
    assert "hi_dim=False" in source
    assert "want_dist=True" in source
    assert "R0019 density did not exactly replay" in source


def test_no_training_or_graph_handler_contract() -> None:
    source = inspect.getsource(round0074_nodes)
    assert '"training_performed": False' in source
    assert "optimizer" not in inspect.getsource(
        round0074_nodes.run_anchor_leverage
    )
    assert "ParametricUMAP.load" not in source
    handler = inspect.getsource(round0074_nodes.run_anchor_leverage)
    assert "build_graph" not in handler
    assert "faiss.write_index" not in handler
