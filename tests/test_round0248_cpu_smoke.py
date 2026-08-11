"""R0248 CPU smoke — both node entry paths, end to end, through `run_job`.

The defect class this exists for is R0216's `NameError`, R0236's arity mismatch
and R0242 attempt 1's missing module: a node that imports fine and dies on its
own last line after the work is done. Both R0248 nodes are cheap enough to run
in full, so nothing here is a stub.

`CUDA_VISIBLE_DEVICES=""` throughout, and the external node's allocator children
import no array library, so no CUDA context exists anywhere near a cgroup OOM
kill.
"""
from __future__ import annotations

import json
import os
import uuid

import pytest

from basemap.round0248_external import CONTROL_MEMORY_MAX_BYTES
from experiments.round0248_nodes import (
    EXTERNAL_ACTION,
    EXTERNAL_CAPABILITY,
    EXTERNAL_FILE,
    GAPGUARD_ACTION,
    GAPGUARD_CAPABILITY,
    GAPGUARD_FILE,
    ROUND_ID,
    run_job,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_ROOT = "/data/latent-basemap/tests"
RELEASE = "0" * 40


@pytest.fixture()
def scratch():
    root = os.path.join(SMOKE_ROOT, f"round0248-{uuid.uuid4().hex}")
    os.makedirs(root, exist_ok=True)
    return root


@pytest.fixture()
def armed(monkeypatch, scratch):
    logs = os.path.join(scratch, "logs")
    os.makedirs(logs, exist_ok=True)
    monkeypatch.setenv("ROUNDRUN_ABORT_FLAG", os.path.join(logs, "node.abort"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return logs


def _active(**manifest):
    body = {
        "round_id": ROUND_ID,
        "release_sha": RELEASE,
        "repo_root": REPO,
        "external_memory_limit": {
            "max_bytes": 73_014_444_032, "swap_max_bytes": 0,
            "mode": "user-scope", "required": True,
        },
    }
    body.update(manifest)
    return {"manifest": body}


def _sealed(output: str, name: str) -> dict:
    with open(os.path.join(output, name), encoding="utf-8") as handle:
        return json.load(handle)


def test_gapguard_entry_path_runs_end_to_end(scratch, armed) -> None:
    output = os.path.join(scratch, "gapguard")
    run_job(_active(), {
        "action": GAPGUARD_ACTION,
        "outputs": [output],
    })
    body = _sealed(output, GAPGUARD_FILE)
    assert body["capabilities"] == [GAPGUARD_CAPABILITY]
    assert body["closure"]["holds"] is True
    assert body["closure"]["arms"][
        "the_inventory_is_derived_mechanically_and_is_complete"] is True
    assert body["cuda_context_created"] is False
    assert body["signal_delivered"] is False
    assert body["child_processes_launched"] == 0
    #: the receipt must publish the registry's number, and the poll gate must
    #: have sealed as real enforcement evidence
    assert body["registered_replay"] == 0.0
    assert body["enforcement_poll_spacing"]["enforcement_evidence"]["holds"] is True
    #: and the attacks that still succeed must be named in it
    assert body["closure"]["self_attack_battery"]["attacks_that_still_succeed"]


def test_gapguard_refuses_another_rounds_queue(scratch, armed) -> None:
    with pytest.raises(Exception):
        run_job(_active(round_id="0247"), {
            "action": GAPGUARD_ACTION,
            "outputs": [os.path.join(scratch, "wrong-round")],
        })


def test_gapguard_fails_closed_when_the_inventory_does_not_hold(
    scratch, armed, monkeypatch, tmp_path
) -> None:
    """The node's FIRST act is the derivation; plant a defect and it stops."""
    planted = tmp_path / "basemap"
    planted.mkdir()
    (planted / "round0246_guard.py").write_text(
        "R0246_MAX_POLL_SPACING_S = 2.5\n"
        "def gate(gap):\n"
        "    return gap <= R0246_MAX_POLL_SPACING_S\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="mechanically derived inventory"):
        run_job(_active(repo_root=str(tmp_path)), {
            "action": GAPGUARD_ACTION,
            "outputs": [os.path.join(scratch, "planted")],
        })


def test_external_entry_path_runs_end_to_end(scratch, armed) -> None:
    output = os.path.join(scratch, "external")
    run_job(_active(), {
        "action": EXTERNAL_ACTION,
        "outputs": [output],
        "external_control_max_bytes": CONTROL_MEMORY_MAX_BYTES,
    })
    body = _sealed(output, EXTERNAL_FILE)
    assert body["capabilities"] == [EXTERNAL_CAPABILITY]
    control = body["control"]
    assert control["holds"] is True
    #: the kill was the kernel's, on the external bound, with the in-process
    #: guard disabled — that is the whole claim
    assert control["arms"]["the_over_allocating_child_was_killed"] is True
    assert control["arms"]["the_kernel_oom_killer_did_it"] is True
    assert control["arms"][
        "an_under_allocating_child_survives_the_same_limit"] is True
    assert control["arms"]["the_in_process_guard_was_not_armed"] is True
    assert control["killed_arm"]["child_receipt"][
        "in_process_guard_armed"] is False
    assert control["killed_arm"]["child_receipt"]["cuda_modules_imported"] == []
    assert body["cuda_context_created"] is False
    assert body["signal_delivered"] is False
    assert body["production_limit_declaration"]["swap_max_bytes"] == 0
    #: and the escape battery ran in both modes and published what it can do
    assert control["escape_arm"]["attempts"]
    assert control["the_other_mode_escape_arm"]["attempts"]


def test_external_refuses_an_unknown_action(scratch, armed) -> None:
    with pytest.raises(Exception, match="does not authorize"):
        run_job(_active(), {
            "action": "not_an_r0248_action",
            "outputs": [os.path.join(scratch, "nope")],
        })
