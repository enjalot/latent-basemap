from __future__ import annotations

import numpy as np
import pytest

from basemap.round0208_prompted_ood_repair import (
    CAPABILITY,
    REGISTERED_TRAINING_OVERLAPS,
    REGISTERED_WITHIN_PACK_DUPLICATE_ROWS,
    REGISTERED_WITHIN_PACK_FAMILIES,
    RETAINED_CORPUS_ROWS,
    RETAINED_QUERY_ROWS,
    ROUND_ID,
    Round0208Error,
    SOURCE_CORPUS_ROWS,
    SOURCE_QUERY_ROWS,
    TRAINING_ROWS,
    repair_plan,
    validate_census,
)
from experiments import round0208_nodes as nodes


def _census(**overrides) -> dict:
    census = {
        "training_rows": TRAINING_ROWS,
        "probe_rows": 20 * (SOURCE_CORPUS_ROWS + SOURCE_QUERY_ROWS),
        "exact_training_family_overlaps": [
            {
                "language": language,
                "split": split,
                "ordinal": ordinal,
                "source_row": source_row,
                "training_compact_row": training_row,
            }
            for language, split, ordinal, source_row, training_row
            in REGISTERED_TRAINING_OVERLAPS
        ],
        "within_pack_exact_families": REGISTERED_WITHIN_PACK_FAMILIES,
        "within_pack_duplicate_rows": REGISTERED_WITHIN_PACK_DUPLICATE_ROWS,
        "within_pack_maximum_family": 2,
        "within_pack_cross_split_families": 0,
        "within_pack_cross_language_families": 0,
        "source_row_identity_overlaps": 0,
    }
    census.update(overrides)
    return census


def test_registered_census_passes() -> None:
    validate_census(_census())


def test_a_new_training_overlap_fails_closed() -> None:
    census = _census()
    census["exact_training_family_overlaps"].append({
        "language": "deu_Latn",
        "split": "corpus",
        "ordinal": 7,
        "source_row": 900_000,
        "training_compact_row": 11,
    })
    with pytest.raises(Round0208Error):
        validate_census(census)


def test_a_missing_training_overlap_fails_closed() -> None:
    census = _census()
    census["exact_training_family_overlaps"].pop()
    with pytest.raises(Round0208Error):
        validate_census(census)


def test_source_row_identity_leakage_fails_closed() -> None:
    with pytest.raises(Round0208Error):
        validate_census(_census(source_row_identity_overlaps=1))


def test_changed_within_pack_census_fails_closed() -> None:
    with pytest.raises(Round0208Error):
        validate_census(_census(within_pack_exact_families=17))
    with pytest.raises(Round0208Error):
        validate_census(_census(within_pack_cross_split_families=1))


def test_repair_plan_is_removal_only_and_ordered() -> None:
    retained = repair_plan(
        language="kor_Hang",
        split="corpus",
        excluded_ordinals=[27027, 32023, 32755, 33177, 38410, 39396],
        source_rows=SOURCE_CORPUS_ROWS,
    )
    assert len(retained) == RETAINED_CORPUS_ROWS
    assert retained == sorted(retained)
    assert len(set(retained)) == len(retained)
    assert not set(retained) & {27027, 32023, 32755, 33177, 38410, 39396}
    assert max(retained) < SOURCE_CORPUS_ROWS


def test_repair_plan_equalizes_every_language_to_one_shape() -> None:
    clean = repair_plan(
        language="deu_Latn", split="corpus", excluded_ordinals=[], source_rows=SOURCE_CORPUS_ROWS
    )
    dirty = repair_plan(
        language="arb_Arab",
        split="corpus",
        excluded_ordinals=[1691, 28454, 45612],
        source_rows=SOURCE_CORPUS_ROWS,
    )
    assert len(clean) == len(dirty) == RETAINED_CORPUS_ROWS


def test_queries_are_kept_whole() -> None:
    retained = repair_plan(
        language="pol_Latn", split="queries", excluded_ordinals=[], source_rows=SOURCE_QUERY_ROWS
    )
    assert retained == list(range(RETAINED_QUERY_ROWS))


def test_query_exclusion_would_break_the_registered_shape() -> None:
    with pytest.raises(Round0208Error):
        repair_plan(
            language="pol_Latn",
            split="queries",
            excluded_ordinals=[0],
            source_rows=SOURCE_QUERY_ROWS,
        )


def test_changed_source_shape_fails_closed() -> None:
    with pytest.raises(Round0208Error):
        repair_plan(
            language="deu_Latn", split="corpus", excluded_ordinals=[], source_rows=49_499
        )


def test_out_of_range_exclusion_fails_closed() -> None:
    with pytest.raises(Round0208Error):
        repair_plan(
            language="deu_Latn",
            split="corpus",
            excluded_ordinals=[SOURCE_CORPUS_ROWS],
            source_rows=SOURCE_CORPUS_ROWS,
        )


def test_node_rejects_another_action() -> None:
    with pytest.raises(Round0208Error):
        nodes.run_job({"manifest": {"round_id": ROUND_ID}}, {"action": "train"})


def test_node_rejects_another_queue() -> None:
    with pytest.raises(Round0208Error):
        nodes.run_repair({"manifest": {"round_id": "0169"}}, {"action": "repair_prompted_ood_pack"})


def test_fingerprints_agree_with_row_bytes() -> None:
    rows = np.arange(8 * 768, dtype=np.float16).reshape(8, 768) / 1024.0 + 0.5
    h0, h1, zero, nonfinite = nodes._fingerprints(rows)
    assert not zero.any() and not nonfinite.any()
    assert len(set(zip(h0.tolist(), h1.tolist()))) == 8
    copy = rows.copy()
    copy[3] = rows[5]
    g0, g1, _z, _n = nodes._fingerprints(copy)
    assert (int(g0[3]), int(g1[3])) == (int(h0[5]), int(h1[5]))


def test_capability_is_the_v2_pack() -> None:
    assert CAPABILITY == "jina-prompted-u12-ood-probe-pack-v2"
    assert ROUND_ID == "0208"
