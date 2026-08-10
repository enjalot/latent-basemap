"""R0240 contract — the registered budget, deadlines, literals and floors.

Everything here is arithmetic or a literal comparison against a sealed R0238
artifact. No GPU, no large file is read: the substrate is bound by the sealed
manifest's hashes, which an independent reviewer already recomputed from its
153.6 GB in one pass.
"""
from __future__ import annotations

import json
import os

import pytest

from basemap.round0238_rung5 import (
    GUARD_IMBALANCE_MARGIN,
    MAX_ZERO_DEGREE_ROWS,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    SELECTION_CANDIDATES,
    SPILL,
)
import basemap.round0240_rung5 as rung
from basemap.round0240_rung5 import (
    Round0240Error,
    tolerance_to_adverse_imbalance,
    verify_inherited_reachability,
    verify_inherited_substrate,
    verify_inherited_truth,
)


def _sealed(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_the_cap_is_the_owner_raised_nine_hours_and_says_why():
    assert rung.GPU_HOURS_CAP == 9.0
    assert "6.0" in rung.GPU_HOURS_CAP_NOTE
    assert "2026-08-10" in rung.GPU_HOURS_CAP_NOTE


def test_the_round_needs_less_than_the_cap_on_r0238s_own_pricing():
    """Grid + predicted build + qualification reserve, against 9.0 h."""
    grid_s = 780.606162478005
    predicted_build_s = 18_237.28974096077
    needed = grid_s + predicted_build_s + rung.QUALIFY_RESERVE_S
    assert needed == pytest.approx(22_017.9, abs=1.0)
    assert needed < rung.GPU_HOURS_CAP * 3600.0
    # and it does NOT fit the cap R0238 was refused under
    assert needed > 6.0 * 3600.0


def test_the_build_deadline_sits_between_the_prediction_and_the_cap():
    """A cooperative flag that can only fire when the cap is genuinely at risk."""
    predicted_build_s = 18_237.28974096077
    grid_s = 780.606162478005
    budget = rung.GPU_HOURS_CAP * 3600.0 - grid_s - rung.QUALIFY_RESERVE_S
    assert predicted_build_s < rung.BUILD_TIMEOUT_S < budget


def test_the_runner_soft_deadline_is_a_backstop_behind_the_cells_own():
    """max(2 x p90, 900) must exceed the node's own bound, or it fires first."""
    ladder_soft = max(2.0 * rung.LADDER_P90_WALL_S, 900.0)
    grid_s = 780.606162478005
    assert ladder_soft > grid_s + rung.BUILD_TIMEOUT_S
    qualify_soft = max(2.0 * rung.QUALIFY_P90_WALL_S, 900.0)
    # R0237 measured 634.0 s for this node at half the rung
    assert qualify_soft > 4.0 * 634.0


def test_the_margin_is_unchanged_and_the_candidate_set_is_c400():
    assert GUARD_IMBALANCE_MARGIN == 1.164884
    assert SELECTION_CANDIDATES == (400,)
    assert SPILL == 8
    assert "not raised" in rung.ADVERSE_CARRY_NOTE.lower() or (
        "NOT raised" in rung.ADVERSE_CARRY_NOTE
    )


def test_the_floors_are_the_registered_ones_and_zero_degree_is_zero():
    assert RECALL_MEAN_FLOOR == 0.9
    assert RECALL_P10_FLOOR == 0.8
    assert MAX_ZERO_DEGREE_ROWS == 0


def test_the_tolerance_definition_reproduces_review_0238s_three_figures():
    """73.8255 / 51.3984 / 91.9620 percent, from the sealed guard inputs."""
    admissible = rung.R0238_ADMISSIBLE_MAX_CLUSTER_ROWS
    margin = GUARD_IMBALANCE_MARGIN
    mean_cluster_rows = 2_000_000.0  # s * N / c = 8 * 1e8 / 400
    for imbalance, expected in (
        (rung.R0237_PREDICTION_IMBALANCE_AT_C400, 73.825529),
        (rung.CARRIED_IMBALANCE_AT_C400, 51.398403),
        (rung.R0238_MEASURED_IMBALANCE_BY_SEED[226], 91.962050),
    ):
        guarded = imbalance * margin * mean_cluster_rows
        observed = tolerance_to_adverse_imbalance(
            admissible_max_cluster_rows=admissible,
            guarded_max_cluster_rows=guarded,
        )
        assert observed * 100.0 == pytest.approx(expected, abs=1e-5)


def test_the_carried_prediction_is_the_measured_100m_value_not_the_50m_one():
    assert rung.CARRIED_IMBALANCE_AT_C400 == 2.8204385
    assert rung.CARRIED_IMBALANCE_AT_C400 == max(
        rung.R0238_MEASURED_IMBALANCE_BY_SEED.values()
    )
    assert rung.R0237_PREDICTION_IMBALANCE_AT_C400 == 2.456543
    excess = (
        rung.CARRIED_IMBALANCE_AT_C400
        / rung.R0237_PREDICTION_IMBALANCE_AT_C400 - 1.0
    )
    assert excess == pytest.approx(0.1481331692545174, abs=1e-12)
    assert rung.R0238_REALISED_TOLERANCE_AT_C400 == pytest.approx(
        0.513984, abs=1e-5
    )


def test_the_guarded_number_reproduces_from_the_carried_imbalance():
    guarded = rung.CARRIED_IMBALANCE_AT_C400 * GUARD_IMBALANCE_MARGIN * 2_000_000.0
    assert guarded == pytest.approx(rung.R0238_GUARDED_MAX_CLUSTER_ROWS, abs=1e-6)


def test_tolerance_refuses_a_nonpositive_denominator():
    with pytest.raises(Round0240Error):
        tolerance_to_adverse_imbalance(
            admissible_max_cluster_rows=1.0, guarded_max_cluster_rows=0.0
        )


# --------------------------------------------------------------------------- #
# the inherited artifacts, verified against the LIVE sealed manifests
# --------------------------------------------------------------------------- #
def test_the_inherited_substrate_reproduces_every_registered_literal():
    sealed = _sealed(rung.INHERITED_SUBSTRATE_MANIFEST)
    record = verify_inherited_substrate(sealed)
    assert record["verified"] is True
    assert record["ordered_substrate_sha256"].startswith("f3f1b4b7")
    assert record["ladder_prefix_ordered_sha256"]["6250000"].startswith("5d976ab6")
    assert record["ladder_prefix_ordered_sha256"]["12500000"].startswith("bd004db8")
    assert record["ladder_prefix_ordered_sha256"]["25000000"].startswith("466ef039")
    assert record["ladder_prefix_ordered_sha256"]["50000000"].startswith("e7ccf848")
    assert record["reserve_rows"] == 200_000
    assert set(record["composition"].values()) == {
        40_000_000, 25_000_000, 10_000_000
    }
    assert sum(record["composition"].values()) == 100_000_000
    for entry in record["shard_coverage"].values():
        assert entry["union"] == 1.0 and entry["increment"] == 1.0


@pytest.mark.parametrize("rows", sorted(rung.REGISTERED_LADDER_PREFIX_SHA256))
def test_a_single_changed_prefix_hash_stops_the_round(rows):
    sealed = _sealed(rung.INHERITED_SUBSTRATE_MANIFEST)
    sealed["nesting"]["ladder_prefix_ordered_sha256"][str(rows)] = "0" * 64
    with pytest.raises(Round0240Error, match="ladder prefix"):
        verify_inherited_substrate(sealed)


def test_a_changed_composition_stops_the_round():
    sealed = _sealed(rung.INHERITED_SUBSTRATE_MANIFEST)
    corpus = "starcoderdata-code-chunked-120-all-MiniLM-L6-v2"
    sealed["composition"][corpus]["rows"] = 9_999_999
    with pytest.raises(Round0240Error, match="composition"):
        verify_inherited_substrate(sealed)


def test_coverage_below_one_stops_the_round():
    sealed = _sealed(rung.INHERITED_SUBSTRATE_MANIFEST)
    corpus = "pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2"
    sealed["selection"]["shard_span"][corpus]["increment"]["coverage"] = 0.9995
    with pytest.raises(Round0240Error, match="coverage"):
        verify_inherited_substrate(sealed)


def test_a_reserve_that_touches_training_stops_the_round():
    sealed = _sealed(rung.INHERITED_SUBSTRATE_MANIFEST)
    sealed["reserve"]["disjointness"]["global_intersection_rows"] = 1
    with pytest.raises(Round0240Error, match="disjoint"):
        verify_inherited_substrate(sealed)


def test_a_changed_substrate_file_hash_stops_the_round():
    sealed = _sealed(rung.INHERITED_SUBSTRATE_MANIFEST)
    sealed["substrate"]["sha256"] = "0" * 64
    with pytest.raises(Round0240Error, match="substrate.f32.npy"):
        verify_inherited_substrate(sealed)


def test_the_inherited_truth_probe_is_the_registered_uniform_draw():
    sealed = _sealed(rung.INHERITED_TRUTH_MANIFEST)
    record = verify_inherited_truth(sealed)
    assert record["probe_rows"] == 500_000
    assert record["probe_seed"] == 238_000
    assert "uniform" in str(record["population"])
    assert "no seed set" in str(record["population"])
    assert "no neighbour union" in str(record["population"])


def test_a_probe_drawn_at_another_seed_stops_the_round():
    sealed = _sealed(rung.INHERITED_TRUTH_MANIFEST)
    sealed["probe_seed"] = 1
    with pytest.raises(Round0240Error, match="registered 500,000 rows"):
        verify_inherited_truth(sealed)


def test_the_inherited_reachability_ceiling_is_r0238s_sealed_one():
    sealed = _sealed(rung.INHERITED_REACHABILITY_MANIFEST)
    record = verify_inherited_reachability(sealed)
    assert record["strict_ceiling_c400"] == 0.9983062666666668
    assert record["rows_with_zero_reachable_c400"] == 1
    assert record["probe_rows"] == 500_000


def test_a_moved_reachability_ceiling_stops_the_round():
    sealed = _sealed(rung.INHERITED_REACHABILITY_MANIFEST)
    sealed["strict_ceiling_by_clusters"]["400"] = 0.99
    with pytest.raises(Round0240Error, match="ceiling"):
        verify_inherited_reachability(sealed)


def test_every_inherited_manifest_exists_where_the_round_binds_it():
    for path in (
        rung.INHERITED_SUBSTRATE_MANIFEST,
        rung.INHERITED_TRUTH_MANIFEST,
        rung.INHERITED_REACHABILITY_MANIFEST,
    ):
        assert os.path.exists(path), path


def test_the_round_re_assembles_nothing():
    """No assemble node, and the substrate is declared inherited."""
    import experiments.prepare_round0240_queue as prepare

    assert "assemble" not in json.dumps(prepare.INHERITED_INPUTS)
    assert prepare.INHERITED_INPUTS["substrate_manifest"] == (
        rung.INHERITED_SUBSTRATE_MANIFEST
    )
