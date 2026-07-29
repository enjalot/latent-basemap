from __future__ import annotations

import importlib
import inspect

import pytest

from experiments.round0058_nodes import (
    Round0058Error,
    _select_smallest_passing,
)


def test_selection_uses_smallest_passing_registered_probe() -> None:
    rows = {
        "32": {"mean_recall_at_15_unambiguous": 0.89},
        "40": {"mean_recall_at_15_unambiguous": 0.90},
        "48": {"mean_recall_at_15_unambiguous": 0.92},
        "56": {"mean_recall_at_15_unambiguous": 0.93},
        "64": {"mean_recall_at_15_unambiguous": 0.94},
    }
    assert _select_smallest_passing(rows) == 40


def test_selection_fails_without_the_frozen_floor() -> None:
    with pytest.raises(Round0058Error):
        _select_smallest_passing({
            "32": {"mean_recall_at_15_unambiguous": 0.89},
            "64": {"mean_recall_at_15_unambiguous": 0.899},
        })


def test_r0058_is_no_training_and_reproduces_r0049() -> None:
    from experiments import prepare_round0058_queue, round0058_nodes

    prep = inspect.getsource(prepare_round0058_queue.prepare_round0058)
    node = inspect.getsource(round0058_nodes.run_sweep)
    assert "gpu_hours_cap=0.25" in prep
    assert '"nprobes": list(NPROBES)' in prep
    assert prepare_round0058_queue.RELEASE_ROOT == (
        "/home/enjalot/code/latent-basemap-run"
    )
    assert '"training_performed": False' in node
    assert '"optimizer_updates": 0' in node
    assert "nprobe64_mean_matches" in node
    assert "nprobe64_p10_matches" in node


def test_r0058_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("experiments.round0058_nodes")
    importlib.import_module("experiments.prepare_round0058_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
