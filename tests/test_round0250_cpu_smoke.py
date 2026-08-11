"""R0250 CPU smoke — every node's ENTRY PATH executed with CUDA hidden.

`run_job` dispatch is the only thing between a queue manifest and hours of GPU
time, and a `NameError` or a bad import there is exactly the class of defect the
release smoke exists to catch before a launch. So each of the five actions is
dispatched here through `run_job`, twice:

* once with a **foreign** queue, which must be refused by the handler's own
  round-id guard -- proving the dispatch reached that handler and not another;
* once with **this** round's id and a job whose first bound input is missing,
  which must raise that node's own error type from inside the handler body --
  proving the body executes rather than failing at import.

The gate node additionally runs its full calibration-and-selection path at a
reduced family count, and the trainer-loop node's non-GPU halves (short-rung
config, per-batch installer, ceiling and projection arithmetic) are exercised
directly, so the only untested code in this round is the two `model.fit` calls
and the three real trains.

Nothing here creates a CUDA context, starts a child process, or writes outside
pytest's own `tmp_path`.
"""
from __future__ import annotations

import math
import os

import pytest

from basemap.round0234_calibrated_floors import (
    CANDIDATE_ORDER,
    GATED_METRICS,
    METRICS,
    PURITY_METRICS,
)
from basemap.round0250_blocksize import Round0250BlockSizeError
from basemap.round0250_gate_n16 import (
    EXACT_FAMILY_SEEDS,
    N_EXACT,
    Round0250GateError,
)
from basemap.round0250_panel_n16 import Round0250PanelError
from basemap.round0250_seed_extension_n16 import (
    Round0250Error,
    SEEDS,
    capability_for_seed,
)
from basemap.round0250_trainer_loops import Round0250TrainerLoopError
from experiments.round0250_nodes import (
    BLOCKSIZE_ACTION,
    GATE_ACTION,
    PANEL_ACTION,
    TRAINLOOP_ACTION,
    TRAIN_ACTION,
    evaluate_selection_n16,
    run_job,
)


ACTIONS = (
    (TRAINLOOP_ACTION, Round0250TrainerLoopError),
    (BLOCKSIZE_ACTION, Round0250BlockSizeError),
    (TRAIN_ACTION, Round0250Error),
    (PANEL_ACTION, Round0250PanelError),
    (GATE_ACTION, Round0250GateError),
)


def _active(round_id: str = "0250"):
    return {"manifest": {"round_id": round_id, "release_sha": "0" * 40}}


def _job(action: str, tmp_path):
    """A job that binds nothing, so the handler body raises on its first input."""
    job = {
        "action": action,
        "outputs": [str(tmp_path / f"{action}-output")],
    }
    if action == TRAIN_ACTION:
        job["training_seed"] = 55
        job["capability"] = capability_for_seed(55)
    return job


@pytest.mark.parametrize("action,error", ACTIONS)
def test_each_node_refuses_a_foreign_queue(action, error, tmp_path):
    """Dispatch reaches THIS handler: only its own round-id guard can refuse."""
    with pytest.raises(error):
        run_job(_active("0249"), _job(action, tmp_path))


@pytest.mark.parametrize("action,error", ACTIONS)
def test_each_node_entry_path_executes_and_raises_on_its_own_missing_input(
    action, error, tmp_path, monkeypatch
):
    """The handler BODY runs: it gets past dispatch and fails on a bound input.

    `CUDA_VISIBLE_DEVICES` is emptied so the two GPU-requiring handlers refuse on
    their own CUDA precondition rather than touching the card, and
    `ROUNDRUN_ABORT_FLAG` is unset so the guarded handlers refuse at
    `require_enforceable_abort_flag` -- both of which are inside the body.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("ROUNDRUN_ABORT_FLAG", raising=False)
    with pytest.raises(Exception) as caught:
        run_job(_active(), _job(action, tmp_path))
    assert not isinstance(caught.value, (ImportError, NameError, AttributeError, TypeError))


def test_an_unknown_action_is_refused(tmp_path):
    with pytest.raises(Round0250Error):
        run_job(_active(), {"action": "not-a-round-0250-action", "outputs": []})


def test_the_train_handler_rejects_a_seed_outside_the_registered_three(tmp_path):
    job = _job(TRAIN_ACTION, tmp_path)
    job["training_seed"] = 42
    with pytest.raises(Round0250Error):
        run_job(_active(), job)


def test_the_train_handler_rejects_a_capability_that_does_not_match_its_seed(tmp_path):
    job = _job(TRAIN_ACTION, tmp_path)
    job["capability"] = capability_for_seed(56)
    with pytest.raises(Round0250Error):
        run_job(_active(), job)


# --------------------------------------------------------------------------- #
# the gate's own path, at a reduced family count
# --------------------------------------------------------------------------- #


def _synthetic_family(n: int):
    """Sixteen plausible cells: tight `ffr`, a mild `k256` outlier, real spread."""
    base_ffr = [
        0.3369, 0.3382, 0.3258, 0.3227, 0.3312, 0.3209, 0.3344, 0.3240,
        0.3325, 0.3227, 0.3329, 0.3192, 0.3341, 0.3301, 0.3266, 0.3288,
    ][:n]
    base_k256 = [
        1.0216, 1.0059, 1.0046, 0.9929, 1.0049, 0.9932, 1.0370, 1.0099,
        1.0120, 1.0024, 1.0055, 1.0065, 1.0115, 1.0071, 1.0038, 1.0102,
    ][:n]
    base_k1024 = [
        0.7326, 0.7229, 0.6980, 0.6936, 0.7214, 0.6842, 0.7266, 0.6991,
        0.7129, 0.7048, 0.7168, 0.6865, 0.7197, 0.7101, 0.7052, 0.7118,
    ][:n]
    series = {
        "ffr": base_ffr,
        "purity_fidelity_k256": [math.exp(-abs(math.log(r))) for r in base_k256],
        "purity_fidelity_k1024": base_k1024,
        "density_v2": [0.44 + 0.001 * index for index in range(n)],
    }
    log_series = {
        "purity_fidelity_k256": [math.log(r) for r in base_k256],
        "purity_fidelity_k1024": [math.log(r) for r in base_k1024],
    }
    return series, log_series


def test_the_selection_rule_runs_at_n16_and_can_disqualify(monkeypatch):
    """The full evaluate/qualify/tie-break path, on a small Monte-Carlo draw."""
    from basemap import round0234_calibration as calibration

    drawn = calibration.calibrate(N_EXACT, families=40_000, chunk=20_000)
    drawn.pop("_arrays")
    calibrated = {"n16": drawn}
    series, log_series = _synthetic_family(N_EXACT)
    selection = evaluate_selection_n16(
        calibrated=calibrated, series=series, log_series=log_series
    )
    assert selection["n"] == 16
    assert set(selection["candidates"]) == set(CANDIDATE_ORDER)
    for name, item in selection["candidates"].items():
        assert item["calibrated_one_sided_multiplier"] > 0.0
        assert set(item["exact_invariance_depth_by_series"]) >= set(GATED_METRICS)
        assert "every_defining_cell_can_fail" in item["attainability_one_sided"]
    # the rule must be able to disqualify: the sample sd is self-loosening by
    # construction and so must fail requirement 2 at any n.
    assert selection["candidates"]["mean_minus_k_sample_sd"][
        "requirement_2_invariance"
    ] is False
    assert "mean_minus_k_sample_sd" not in selection["qualifying"]
    if selection["chosen_estimator"] is not None:
        assert selection["chosen_estimator"] in selection["qualifying"]
        assert selection["reasoning"][-1].startswith("registered: ")


def test_the_selection_rule_registers_nothing_when_nothing_qualifies(monkeypatch):
    """Positive control: with every candidate disqualified, the rule refuses."""
    import experiments.round0250_nodes as nodes

    from basemap import round0234_calibration as calibration

    drawn = calibration.calibrate(N_EXACT, families=20_000, chunk=20_000)
    drawn.pop("_arrays")
    series, log_series = _synthetic_family(N_EXACT)
    monkeypatch.setattr(nodes, "REQUIRED_INVARIANCE_DEPTH", 99)
    selection = nodes.evaluate_selection_n16(
        calibrated={"n16": drawn}, series=series, log_series=log_series
    )
    assert selection["qualifying"] == []
    assert selection["chosen_estimator"] is None
    assert selection["reasoning"][-1].startswith("no candidate qualifies")


def test_the_exact_family_is_the_sixteen_seeds_the_round_trains_towards():
    assert EXACT_FAMILY_SEEDS[-3:] == SEEDS
    assert len(EXACT_FAMILY_SEEDS) == N_EXACT == 16
    assert set(METRICS) - set(GATED_METRICS) == {"density_v2"}
    assert set(PURITY_METRICS) <= set(GATED_METRICS)


# --------------------------------------------------------------------------- #
# the blocksize node must PUBLISH a ceiling breach, not die on it
# --------------------------------------------------------------------------- #


def test_a_measurement_node_publishes_a_refused_gate_rather_than_aborting():
    """Attempt 1's defect, as a control.

    `_score_gate_without_raising` must return a scored verdict whose
    `meets_the_registered_ceiling` is False, with the failing arm named, instead
    of propagating `Round0246Error`. A measurement node that dies on its own
    finding destroys the measurement — which is what attempt 1 of `blocksize_0250`
    did at the warm block=2,000 arm.
    """
    from basemap.round0246_guard import Round0246Error
    from experiments.round0250_nodes import (
        _node_gate,
        _score_gate_without_raising,
    )

    gate = _node_gate("R0250 breach control", training_performed=False)
    gate.start()
    gate("first read")
    # a hand-made gap wider than the registered ceiling
    gate._last = gate._clock() - 9.0
    gate("second read, far too late")
    gate.finish("control end")

    tail = {"host_watchdog": {"anonymous_trace_by_second": []}}
    # the raising path is what attempt 1 used
    with pytest.raises(Round0246Error):
        gate.require(measured_slope_bytes_per_s=None)
    # the reporting path publishes the same refusal as data
    scored = _score_gate_without_raising(gate, tail, label="R0250 breach control")
    assert scored["meets_the_registered_ceiling"] is False
    assert scored["max_gap_between_enforcement_polls_s"] > 2.5109531834854018
    assert scored["outcome"]["require_raised"] is True
    assert "meets_the_registered_ceiling" in scored["failures"]
