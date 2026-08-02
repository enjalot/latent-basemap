"""CUDA-hidden tests for the reauthorized R0155 census."""
from __future__ import annotations

import pytest

from basemap.round0153_density_forensics import CAPABILITY as R0153_CAPABILITY
from experiments import prepare_round0155_queue as queue_prep


def test_cpu_queue_uses_independent_execution_checkout() -> None:
    assert queue_prep.RELEASE_ROOT.endswith("latent-basemap-cpu-run")
    assert queue_prep.RELEASE_ROOT != "/home/enjalot/code/latent-basemap-run"


def test_activation_accepts_only_positive_density_branch(monkeypatch) -> None:
    monkeypatch.setattr(queue_prep, "_accepted_review", lambda *_args: [])
    monkeypatch.setattr(
        queue_prep,
        "_read_sealed",
        lambda *_args, **_kwargs: (
            {
                "round_id": "0153",
                "capability": R0153_CAPABILITY,
                "decision": {
                    "outcome": "density-restores-with-row-universe",
                    "track_f_activated": True,
                    "floor_changed": False,
                },
            },
            {"canonical_path": "/tmp/evidence"},
        ),
    )
    _reviews, signature = queue_prep._accepted_activation()
    assert signature["canonical_path"] == "/tmp/evidence"


@pytest.mark.parametrize(
    "decision",
    [
        {
            "outcome": "density-does-not-restore",
            "track_f_activated": False,
            "floor_changed": False,
        },
        {
            "outcome": "density-mixed-owner-decision-required",
            "track_f_activated": False,
            "floor_changed": False,
        },
        {
            "outcome": "density-restores-with-row-universe",
            "track_f_activated": True,
            "floor_changed": True,
        },
    ],
)
def test_activation_rejects_every_nonregistered_branch(
    monkeypatch, decision: dict
) -> None:
    monkeypatch.setattr(queue_prep, "_accepted_review", lambda *_args: [])
    monkeypatch.setattr(
        queue_prep,
        "_read_sealed",
        lambda *_args, **_kwargs: (
            {
                "round_id": "0153",
                "capability": R0153_CAPABILITY,
                "decision": decision,
            },
            {},
        ),
    )
    with pytest.raises(RuntimeError, match="positive R0153 activation"):
        queue_prep._accepted_activation()
