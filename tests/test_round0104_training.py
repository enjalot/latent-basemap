import numpy as np
import pytest

from basemap.round0104_training import (
    ARMS,
    DECISION_METRICS,
    NONINFERIORITY_RATIO,
    Round0104Error,
    paired_decision,
    preprocessing_stamp,
)
from experiments.round0104_nodes import _recall_rows, _without_self


def _score(multiplier=1.0, gates=True):
    return {
        "metrics": {
            metric: multiplier * (0.5 + index / 10)
            for index, metric in enumerate(DECISION_METRICS)
        },
        "execution_gates": {
            "finite": gates,
            "accounting": gates,
        },
    }


def test_paired_decision_applies_registered_ratio_to_every_metric():
    control = _score()
    treatment = _score(NONINFERIORITY_RATIO)
    decision = paired_decision(control=control, treatment=treatment)
    assert decision["passed"] is True
    assert set(decision["metric_gates"]) == set(DECISION_METRICS)
    assert all(row["passed"] for row in decision["metric_gates"].values())


def test_paired_decision_fails_one_metric_or_execution_gate():
    control = _score()
    treatment = _score()
    treatment["metrics"][DECISION_METRICS[0]] *= 0.969
    assert paired_decision(control=control, treatment=treatment)["passed"] is False
    treatment = _score()
    treatment["execution_gates"]["accounting"] = False
    assert paired_decision(control=control, treatment=treatment)["passed"] is False


def test_preprocessing_stamps_bind_arm_and_only_registered_arms():
    left = preprocessing_stamp(ARMS[0])
    right = preprocessing_stamp(ARMS[1])
    assert left["identity_sha256"] != right["identity_sha256"]
    assert left["source_rows"] == right["source_rows"]
    with pytest.raises(Round0104Error):
        preprocessing_stamp("ambient")


def test_search_helpers_exclude_self_and_measure_recall():
    rows = np.asarray([[0, 2, 3], [1, 3, 2]], dtype=np.int64)
    observed = _without_self(rows, np.asarray([0, 1]), 2)
    assert np.array_equal(observed, np.asarray([[2, 3], [3, 2]]))
    truth = np.asarray([[2, 4], [3, 5]], dtype=np.int64)
    assert np.allclose(_recall_rows(observed, truth), [0.5, 0.5])
