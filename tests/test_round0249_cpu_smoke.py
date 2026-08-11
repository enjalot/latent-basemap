"""R0249 CPU smoke — both node entry paths, end to end, through `run_job`.

The defect class this exists for is R0216's `NameError`, R0236's arity mismatch
and R0242 attempt 1's missing module: a node that imports fine and dies on its
own last line after the work is done. Both R0249 nodes are cheap enough to run
in full, so nothing here is a stub.

`CUDA_VISIBLE_DEVICES=""` throughout, and the external node's allocator children
import no array library. R0249's production bound is `memory.high`, which cannot
OOM-kill at all; the one `memory.max` child in the contrast arm is a plain
`bytearray` loop with no CUDA context anywhere near it.
"""
from __future__ import annotations

import json
import os
import uuid

import pytest

from basemap.round0249_external import CONTROL_MEMORY_HIGH_BYTES
from experiments.round0249_nodes import (
    EXTERNAL_ACTION,
    EXTERNAL_CAPABILITY,
    EXTERNAL_FILE,
    GUARDFIX_ACTION,
    GUARDFIX_CAPABILITY,
    GUARDFIX_FILE,
    ROUND_ID,
    run_job,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_ROOT = "/data/latent-basemap/tests"
RELEASE = "0" * 40


@pytest.fixture()
def scratch():
    root = os.path.join(SMOKE_ROOT, f"round0249-{uuid.uuid4().hex}")
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
            "mode": "root-scope", "required": True,
            "limit_file": "memory.high",
        },
    }
    body.update(manifest)
    return {"manifest": body}


def _sealed(output: str, name: str) -> dict:
    with open(os.path.join(output, name), encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# node 1
# --------------------------------------------------------------------------- #
def test_guardfix_entry_path_runs_end_to_end(scratch, armed) -> None:
    output = os.path.join(scratch, "guardfix")
    run_job(_active(), {"action": GUARDFIX_ACTION, "outputs": [output]})
    body = _sealed(output, GUARDFIX_FILE)
    assert body["capabilities"] == [GUARDFIX_CAPABILITY]
    assert body["closure"]["holds"] is True
    for control in body["closure"]["controls"].values():
        assert control["holds"] is True, control["control"]
    assert body["cuda_context_created"] is False
    assert body["signal_delivered"] is False
    assert body["child_processes_launched"] == 0
    #: the receipt publishes the registry's number and its CORRECTED class
    assert body["registered_replay"] == 0.0
    assert body["registered_replay_enforcement"] == "declared"
    #: all four disclosure fields are read-only properties
    assert all(body["disclosure_fields_are_read_only_properties"].values())
    #: the node's own poll gate still seals as real enforcement evidence
    assert body["enforcement_poll_spacing"]["enforcement_evidence"][
        "holds"] is True
    #: and the risks R0249 deliberately does not chase are named
    assert len(body["closure"]["accepted_residual_risks"]) == 3
    assert body["closure"]["what_is_not_closed"]
    #: the kernel's own account of the limit applied to this process
    assert "external_memory_limit_as_the_kernel_applied_it" in body


def test_guardfix_refuses_another_rounds_queue(scratch, armed) -> None:
    with pytest.raises(Exception):
        run_job(_active(round_id="0248"), {
            "action": GUARDFIX_ACTION,
            "outputs": [os.path.join(scratch, "wrong-round")],
        })


def test_guardfix_fails_closed_when_the_inventory_does_not_hold(
    scratch, armed, tmp_path
) -> None:
    """The node's FIRST act is the derivation; plant a defect and it stops.

    The planted module does not import the registry, so it is outside the
    triage scope — the position `round0242_nodes.py:246` was in when the
    hand-written list hid it. The discovered wide scope must catch it anyway.
    """
    planted = tmp_path / "experiments"
    planted.mkdir()
    (planted / "round0242_planted.py").write_text(
        "WATCHDOG_ANON_BYTES = 60 * (1 << 30)\n"
        "def poll(host):\n"
        "    return int(host['anonymous_bytes']) > WATCHDOG_ANON_BYTES\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="mechanically derived inventory"):
        run_job(_active(repo_root=str(tmp_path)), {
            "action": GUARDFIX_ACTION,
            "outputs": [os.path.join(scratch, "planted")],
        })


# --------------------------------------------------------------------------- #
# node 2
# --------------------------------------------------------------------------- #
def test_external_entry_path_runs_end_to_end(scratch, armed) -> None:
    output = os.path.join(scratch, "external")
    run_job(_active(), {
        "action": EXTERNAL_ACTION,
        "outputs": [output],
        "external_control_high_bytes": CONTROL_MEMORY_HIGH_BYTES,
    })
    body = _sealed(output, EXTERNAL_FILE)
    assert body["capabilities"] == [EXTERNAL_CAPABILITY]

    throttle = body["throttle_control"]
    assert throttle["holds"] is True
    #: THE claim: it was throttled and it SURVIVED
    assert throttle["arms"]["the_over_allocating_child_survived"] is True
    assert throttle["arms"]["the_kernel_throttled_it"] is True
    assert throttle["arms"]["nothing_was_oom_killed_in_the_scope"] is True
    assert throttle["arms"]["no_memory_max_kill_limit_was_set"] is True
    assert throttle["throttled_arm"]["returncode"] == 0
    assert throttle["throttled_arm"]["memory_events_high_delta"] > 0
    #: and the contrast: the identical allocator under R0248's memory.max dies
    assert throttle["memory_max_contrast_arm"]["killed_by_signal"] is True
    #: the child armed no in-process guard and imported no array library
    child = throttle["throttled_arm"]["child_receipt"]
    assert child["in_process_guard_armed"] is False
    assert child["cuda_modules_imported"] == []

    escapes = body["escape_battery"]
    assert escapes["holds"] is True
    assert escapes["default_mode"]["mode"] == "root-scope"
    assert escapes["default_mode"]["attempts_run"] == 5
    assert escapes["default_mode"]["the_node_can_defeat_this_mode"] is False
    #: the LOSING mode is published too, and it must actually have run — the
    #: round's first attempt published `attempts: []` here with
    #: `the_node_can_defeat_this_mode: false` beside it
    other = escapes["the_other_mode"]
    assert other["mode"] == "user-scope"
    assert other["attempts_run"] == 5
    assert other["the_node_can_defeat_this_mode"] is True

    assert body["fail_closed_control"]["holds"] is True
    assert body["production_limit_declaration"]["swap_max_bytes"] == 0
    assert body["production_limit_declaration"]["limit_file"] == "memory.high"
    assert body["production_limit_declaration"][
        "can_oom_kill_the_node"] is False
    assert body["cuda_context_created"] is False
    assert body["signal_delivered"] is False
    #: review-0248-01 G item 7: the inventory runs in THIS node too
    assert body["inventory"]["holds"] is True


def test_external_refuses_an_unknown_action(scratch, armed) -> None:
    with pytest.raises(Exception, match="does not authorize"):
        run_job(_active(), {
            "action": "not_an_r0249_action",
            "outputs": [os.path.join(scratch, "nope")],
        })
