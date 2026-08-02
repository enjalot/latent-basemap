"""CUDA-hidden contract tests for R0157 prompted density recovery."""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0108_evaluation import seal
from basemap.round0157_prompted_density import (
    ANCHORS,
    ROWS,
    Round0157Error,
    density_v2_from_radii,
    transcribe_native_prompted_score,
)
from experiments import prepare_round0157_queue


def _signature() -> dict[str, object]:
    return {
        "canonical_path": "/tmp/coordinates.npy",
        "kind": "file",
        "bytes": 10,
        "sha256": "a" * 64,
    }


def _score() -> dict[str, object]:
    coordinates = _signature()
    return seal({
        "arm": "document",
        "coordinates": {"training": coordinates},
        "metrics": {
            "density": 0.22,
            "ffr": 0.63,
            "oos_recall_at_10": 0.011,
            "oos_recall_at_50": 0.048,
            "recall_at_10": 0.015,
        },
        "execution_gates": {"finite_noncollapsed_coordinates": True},
        "ood": {"pol_Latn": {"role": "diagnostic"}},
        "panel": {
            "n": ROWS,
            "n_dims_hi": 768,
            "n_anchors": 4_000,
            "k_density": 15,
            "provenance": {
                "arm": "document",
                "train_receipt": _signature(),
            },
        },
    })


def test_transcription_requires_native_document_arm() -> None:
    value = transcribe_native_prompted_score(
        _score(), seed=42, expected_coordinates=_signature()
    )
    assert value["native_training"] is True
    assert value["training_rows"] == ROWS
    bad = _score()
    bad["arm"] = "raw"
    with pytest.raises(Round0157Error, match="native prompted score changed"):
        transcribe_native_prompted_score(
            bad, seed=42, expected_coordinates=_signature()
        )


def test_density_v2_is_deterministic_and_validated() -> None:
    high = np.linspace(0.01, 1.0, ANCHORS, dtype=np.float64)
    low = high ** 0.8
    first = density_v2_from_radii(high, low)
    second = density_v2_from_radii(high, low)
    assert first[0]["correlation"] == pytest.approx(1.0)
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    with pytest.raises(Round0157Error, match="radii changed"):
        density_v2_from_radii(high[:-1], low[:-1])


def test_queue_is_no_training_and_uses_gpu_run_checkout() -> None:
    assert prepare_round0157_queue.RELEASE_ROOT.endswith("latent-basemap-run")
    assert prepare_round0157_queue.ROUND_ROOT.endswith("round-0157")

