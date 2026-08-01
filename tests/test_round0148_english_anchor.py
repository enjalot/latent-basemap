from __future__ import annotations

import copy

import pytest

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0105_search import GROUPS
from basemap.round0132_scale_bridge import HALF_RETAINED_ROWS, SUBSET_NAMESPACE
from basemap.round0148_english_anchor import (
    DENSITY_V2_FLOOR,
    ENGLISH_GROUPS,
    LANGUAGE_GROUPS,
    OOD_METRICS,
    OUTCOME_DENSITY,
    OUTCOME_FUNCTIONAL,
    OUTCOME_INVALID,
    OUTCOME_OOD,
    OUTCOME_RESCUED,
    Round0148Error,
    build_subset_plan,
    english_anchor_decision,
    english_anchor_quotas,
    ranking_namespace,
)


REAL_COUNTS = {
    "fineweb-edu-sample-10BT-chunked-500-jina-v5-nano": 2_878_533,
    "RedPajama-Data-V2-sample-10B-chunked-500-jina-v5-nano": 2_814_846,
    "pile-uncopyrighted-chunked-500-jina-v5-nano": 3_385_908,
    "arb_Arab": 835_367,
    "ces_Latn": 835_247,
    "cmn_Hani": 835_129,
    "deu_Latn": 835_318,
    "ell_Grek": 835_334,
    "fra_Latn": 835_364,
    "hin_Deva": 834_703,
    "ind_Latn": 835_315,
    "ita_Latn": 835_220,
    "jpn_Jpan": 835_089,
    "kor_Hang": 834_542,
    "nld_Latn": 835_392,
    "por_Latn": 835_398,
    "rus_Cyrl": 835_325,
    "spa_Latn": 835_299,
    "swe_Latn": 835_309,
    "tha_Thai": 835_305,
    "tur_Latn": 835_338,
    "vie_Latn": 835_382,
}

EXPECTED_LANGUAGE_QUOTAS = {
    "arb_Arab": 178_716,
    "ces_Latn": 178_690,
    "cmn_Hani": 178_665,
    "deu_Latn": 178_705,
    "ell_Grek": 178_709,
    "fra_Latn": 178_715,
    "hin_Deva": 178_574,
    "ind_Latn": 178_705,
    "ita_Latn": 178_684,
    "jpn_Jpan": 178_656,
    "kor_Hang": 178_539,
    "nld_Latn": 178_721,
    "por_Latn": 178_722,
    "rus_Cyrl": 178_707,
    "spa_Latn": 178_701,
    "swe_Latn": 178_703,
    "tha_Thai": 178_703,
    "tur_Latn": 178_710,
    "vie_Latn": 178_719,
}


def test_real_candidate_quota_and_nested_intersection_close() -> None:
    quotas = english_anchor_quotas(REAL_COUNTS)
    assert sum(quotas.values()) == HALF_RETAINED_ROWS == 12_474_331
    assert {group: quotas[group] for group in LANGUAGE_GROUPS} == (
        EXPECTED_LANGUAGE_QUOTAS
    )
    assert all(quotas[group] == REAL_COUNTS[group] for group in ENGLISH_GROUPS)

    plan = build_subset_plan(REAL_COUNTS)
    assert plan["common_intersection_rows"] == 7_934_687
    assert plan["english_rows_added_vs_u12"] == 4_539_644
    assert plan["language_rows_removed_vs_u12"] == 4_539_644
    assert plan["identity_sha256"] == (
        "ee4a911a9d458dc8a6bb3107e1cbb0964beb0255cd85957c7270f3a93992f1d6"
    )
    body = {key: value for key, value in plan.items() if key != "identity_sha256"}
    assert plan["identity_sha256"] == sha256_bytes(canonical_json(body))


def test_languages_reuse_exact_r0132_rank_namespace() -> None:
    for group in GROUPS:
        assert ranking_namespace(group) == (
            SUBSET_NAMESPACE + group.encode("utf-8") + b"\0"
        )


def test_population_drift_fails_closed() -> None:
    changed = copy.deepcopy(REAL_COUNTS)
    changed[GROUPS[0]] -= 1
    with pytest.raises(Round0148Error, match="full retained population"):
        english_anchor_quotas(changed)


def test_group_key_drift_fails_closed() -> None:
    changed = copy.deepcopy(REAL_COUNTS)
    changed["invented"] = changed.pop(GROUPS[-1])
    with pytest.raises(Round0148Error, match="group-count keys"):
        english_anchor_quotas(changed)


def test_unknown_namespace_group_fails_closed() -> None:
    with pytest.raises(Round0148Error, match="unknown registered"):
        ranking_namespace("not-a-group")


def _functional(*, ffr: float = 0.60) -> dict:
    return {
        "panel": {
            "ffr": ffr,
            "purity": {
                "k256": 0.95,
                "k1024": 0.96,
            },
        },
        "projection": {"ffr": 0.54, "recall_at_10": 0.0100},
    }


def _ood(value: float = 0.25) -> dict[str, float]:
    return {key: value for key in OOD_METRICS}


def test_rescue_selector_requires_function_density_and_ood_together() -> None:
    result = english_anchor_decision(
        validity_checks={"train": True, "panels": True},
        candidate_functional_cell=_functional(),
        density_v2=DENSITY_V2_FLOOR,
        control_ood=_ood(),
        candidate_ood=_ood(0.25 * 0.97),
    )
    assert result["outcome"] == OUTCOME_RESCUED
    assert result["functional"]["passed"] is True
    assert result["density_v2"]["passed"] is True
    assert result["matched_ood"]["passed"] is True
    assert result["registered_density_floor_changed"] is False
    assert result["twenty_five_million_transfer_claimed"] is False


@pytest.mark.parametrize(
    ("kwargs", "outcome"),
    [
        ({"validity_checks": {"train": False}}, OUTCOME_INVALID),
        ({"candidate_functional_cell": _functional(ffr=0.55)}, OUTCOME_FUNCTIONAL),
        ({"density_v2": DENSITY_V2_FLOOR - 1e-12}, OUTCOME_DENSITY),
        ({"candidate_ood": _ood(0.25 * 0.969)}, OUTCOME_OOD),
    ],
)
def test_rescue_selector_branch_order_is_frozen(kwargs: dict, outcome: str) -> None:
    arguments = {
        "validity_checks": {"train": True},
        "candidate_functional_cell": _functional(),
        "density_v2": DENSITY_V2_FLOOR,
        "control_ood": _ood(),
        "candidate_ood": _ood(),
    }
    arguments.update(kwargs)
    assert english_anchor_decision(**arguments)["outcome"] == outcome


def test_rescue_selector_rejects_nonfinite_or_zero_control() -> None:
    with pytest.raises(Round0148Error, match="finite"):
        english_anchor_decision(
            validity_checks={"train": True},
            candidate_functional_cell=_functional(),
            density_v2=float("nan"),
            control_ood=_ood(),
            candidate_ood=_ood(),
        )
    control = _ood()
    control[OOD_METRICS[0]] = 0.0
    with pytest.raises(Round0148Error, match="positive"):
        english_anchor_decision(
            validity_checks={"train": True},
            candidate_functional_cell=_functional(),
            density_v2=DENSITY_V2_FLOOR,
            control_ood=control,
            candidate_ood=_ood(),
        )
