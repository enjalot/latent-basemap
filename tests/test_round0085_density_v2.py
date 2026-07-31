from __future__ import annotations

import inspect

import numpy as np

from experiments import prepare_round0085_queue, round0085_nodes


R0074_ROOT = (
    "/data/latent-basemap/runs/round-0074/queue-attempt-2/artifacts/"
    "duplicate-anchor-leverage"
)


def test_density_v2_is_deterministic_and_separates_a_signal_from_null() -> None:
    rng = np.random.RandomState(7)
    high = np.exp(rng.normal(size=500))
    low = high * np.exp(rng.normal(scale=0.2, size=500))
    first, first_bootstrap, first_null = (
        round0085_nodes.density_v2_calibration(
            high,
            low,
            bootstrap_draws=40,
            null_draws=40,
        )
    )
    second, second_bootstrap, second_null = (
        round0085_nodes.density_v2_calibration(
            high,
            low,
            bootstrap_draws=40,
            null_draws=40,
        )
    )
    assert first == second
    assert np.array_equal(first_bootstrap, second_bootstrap)
    assert np.array_equal(first_null, second_null)
    assert first["correlation"] > 0.9
    assert (
        first["correlation"]
        > first["permuted_radius_null"]["absolute_99_9_percentile"]
    )


def _cell(point: float, sd: float, null: float) -> dict:
    return {
        "density_v2": {
            "correlation": point,
            "bootstrap": {"standard_deviation": sd},
            "permuted_radius_null": {
                "absolute_99_9_percentile": null,
            },
        }
    }


def test_registered_floor_is_positive_and_null_guarded() -> None:
    good = {
        key: _cell(0.10 + index * 0.01, 0.005, 0.03)
        for index, key in enumerate(round0085_nodes.MATCHED_CELL_KEYS)
    }
    floor = round0085_nodes.registered_floor(good)
    assert floor["proposed_floor"] == 0.085
    assert floor["gating_floor_registered"] is True
    assert floor["registered_floor"] == 0.085

    null_overlaps = dict(good)
    null_overlaps[round0085_nodes.MATCHED_CELL_KEYS[0]] = _cell(
        0.10, 0.005, 0.09
    )
    rejected = round0085_nodes.registered_floor(null_overlaps)
    assert rejected["gating_floor_registered"] is False
    assert rejected["registered_floor"] is None


def test_r0074_filtered_values_replay_exactly() -> None:
    receipt = (
        f"{R0074_ROOT}/duplicate-anchor-leverage.json"
    )
    radii = f"{R0074_ROOT}/anchor-leverage-radii.npz"
    from basemap.artifact_identity import expected_input_signature

    replay = round0085_nodes.replay_r0074(
        receipt_path=receipt,
        receipt_sha256=expected_input_signature(receipt)["sha256"],
        radii_path=radii,
        radii_sha256=expected_input_signature(radii)["sha256"],
    )
    assert replay["exact"] is True
    assert replay["anchors_after_filter"] == 9_980
    assert replay["replayed_correlations"] == {
        "legacy_r0019": 0.0985,
        "modern_r0061": 0.1125,
    }


def test_queue_is_one_bounded_no_training_gpu_node() -> None:
    source = inspect.getsource(
        prepare_round0085_queue.prepare_round0085
    )
    assert source.count('"action": "density_v2"') == 1
    assert "gpu_hours_cap=0.60" in source
    assert '"p90_wall_s": 1_200.0' in source
    assert '"training_performed": False' in source
    assert '"minilm-density-v2-calibration-v1"' in source


def test_metric_uses_representative_candidate_and_filtered_anchor_universes() -> None:
    source = inspect.getsource(round0085_nodes.run_density_v2)
    assert "RepresentativeArrayView(" in source
    assert "anchors[eligible]" in source
    assert "high_radius = universe[\"high_radius\"][eligible]" in source
    assert "want_dist=True" in source
    assert "exact=True" in source
    assert "registered_floor(cells)" in source


def test_no_training_or_graph_construction_contract() -> None:
    source = inspect.getsource(round0085_nodes.run_density_v2)
    assert "optimizer" not in source
    assert "fit(" not in source
    assert "build_graph" not in source
    assert "write_index" not in source
    assert '"training_performed": False' in source


def test_handler_reads_round_id_from_runner_manifest() -> None:
    source = inspect.getsource(round0085_nodes.run_job)
    assert 'active.get("manifest", {}).get("round_id")' in source
    assert 'active.get("round_id")' not in source
