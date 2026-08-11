"""R0252 contract — every guard this round adds ships a planted-defect control.

`AGENT_STARTUP.md`: "Any guard, detector, or tripwire the round adds must ship a
positive control — a test that plants the defect and proves the guard catches
it." R0252 adds three mechanisms — a chunk-level abort read in
`basemap/artifact_identity.py`, ten abort reads in `basemap/panel_v2.py`, and a
stop-latency instrument — and each has a test here that removes or breaks it and
proves the failure is visible.

Nothing here touches CUDA. The suite is the release CPU smoke.
"""
from __future__ import annotations

import ast
import hashlib
import os
import time

import numpy as np
import pytest

from basemap import artifact_identity, panel_v2
from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
    sha256_file,
)
from basemap.round0252_stoppability import (
    FlagFileAbortPoll,
    INTEGRITY_GUARANTEE,
    Round0252CooperativeAbort,
    Round0252Error,
    STOP_CONTROL_UPDATES,
    TAIL_RUNG_UPDATES,
    TARGET_DIMENSION,
    TARGET_ROWS,
    TARGET_SUBSTRATE_BYTES,
    declared_sites_match_the_release,
    gap_reduction,
    gap_report,
    measure_stop_latency,
    prior_rung_from_artifact,
    size_law,
    tail_identification,
)


CHUNK = 8 << 20


@pytest.fixture(autouse=True)
def _no_hook_leaks():
    """Every test starts and ends with both module hooks uninstalled."""
    artifact_identity.set_abort_poll(None)
    panel_v2.set_abort_poll(None)
    yield
    artifact_identity.set_abort_poll(None)
    panel_v2.set_abort_poll(None)


def _write(tmp_path, name, size):
    path = os.path.join(str(tmp_path), name)
    block = bytes(range(256)) * 4096          # 1 MiB, deterministic
    with open(path, "wb") as handle:
        written = 0
        while written < size:
            take = min(len(block), size - written)
            handle.write(block[:take])
            written += take
    return path


# --------------------------------------------------------------------------- #
# the integrity guarantee
# --------------------------------------------------------------------------- #


def test_the_digest_is_identical_with_and_without_the_hook(tmp_path):
    """The whole point: polling changes the timing, never the bytes."""
    path = _write(tmp_path, "s.bin", 3 * CHUNK + 17)
    expected = hashlib.sha256(open(path, "rb").read()).hexdigest()
    plain = sha256_file(path)
    plain_identity = expected_input_signature(path)["sha256"]

    calls = []
    artifact_identity.set_abort_poll(calls.append)
    polled = sha256_file(path)
    polled_identity = expected_input_signature(path)["sha256"]
    artifact_identity.set_abort_poll(None)

    assert plain == polled == plain_identity == polled_identity == expected
    assert calls, "the hook was never called"


def test_the_hook_defaults_to_none_and_costs_nothing_when_uninstalled(tmp_path):
    assert artifact_identity.abort_poll is None
    assert panel_v2.abort_poll is None
    path = _write(tmp_path, "s.bin", CHUNK + 1)
    assert sha256_file(path)  # no hook installed, no exception, real digest


def test_a_poll_is_read_between_every_chunk(tmp_path):
    """POSITIVE CONTROL for the interval bound: count reads against chunks."""
    size = 5 * CHUNK + 3
    path = _write(tmp_path, "s.bin", size)
    seen = []
    artifact_identity.set_abort_poll(lambda where: seen.append(where))
    sha256_file(path)
    artifact_identity.set_abort_poll(None)
    expected_chunks = -(-size // CHUNK)
    assert len(seen) == expected_chunks
    assert set(seen) == {artifact_identity.ABORT_POLL_SITE_FILE_CHUNK}


def test_an_abort_mid_hash_stops_and_yields_no_digest(tmp_path):
    """POSITIVE CONTROL: plant an abort, prove the hash stops and returns nothing."""
    path = _write(tmp_path, "s.bin", 6 * CHUNK)
    state = {"n": 0}

    def poll(_where):
        state["n"] += 1
        if state["n"] == 2:
            raise Round0252CooperativeAbort("planted")

    artifact_identity.set_abort_poll(poll)
    with pytest.raises(Round0252CooperativeAbort):
        expected_input_signature(path)
    artifact_identity.set_abort_poll(None)
    assert state["n"] == 2, "the hash did not stop at the planted abort"


def test_ordered_array_hash_polls_and_keeps_its_digest():
    array = np.arange(200_000, dtype=np.float32).reshape(-1, 2)
    plain = ordered_array_sha256(array, row_chunk=10_000)
    seen = []
    artifact_identity.set_abort_poll(seen.append)
    polled = ordered_array_sha256(array, row_chunk=10_000)
    artifact_identity.set_abort_poll(None)
    assert plain == polled
    assert len(seen) == 10
    assert set(seen) == {artifact_identity.ABORT_POLL_SITE_ARRAY_CHUNK}


def test_the_integrity_statement_names_what_changed_and_what_did_not():
    assert INTEGRITY_GUARANTEE["still_guaranteed"]
    assert INTEGRITY_GUARANTEE["newly_possible"]
    rejected = INTEGRITY_GUARANTEE["rejected_alternatives"]
    assert set(rejected) == {
        "cache_by_path_size_mtime_inode", "fold_into_the_streaming_read"
    }
    for text in rejected.values():
        assert text.startswith("REJECTED.")


def test_a_stat_keyed_cache_would_have_been_unsound(tmp_path):
    """POSITIVE CONTROL for the rejected alternative, not a straw man.

    The round rejects caching a digest by `(path, size, mtime, inode)`. This
    plants exactly the defect that key would miss: an in-place, same-length
    write with the mtime restored. Size, inode and mtime are unchanged; the
    content, and therefore the true digest, is not.
    """
    path = _write(tmp_path, "s.bin", 4096)
    before = os.stat(path)
    first = sha256_file(path)
    with open(path, "r+b") as handle:
        handle.seek(1000)
        handle.write(b"\x00" * 16)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.stat(path)
    assert (after.st_size, after.st_ino, after.st_mtime_ns) == (
        before.st_size, before.st_ino, before.st_mtime_ns
    )
    assert sha256_file(path) != first, (
        "the content changed while every field a stat-keyed cache would have "
        "used stayed identical -- which is why the cache was rejected"
    )


# --------------------------------------------------------------------------- #
# the scorer's hook, checked structurally (its numbers need CUDA)
# --------------------------------------------------------------------------- #


def _call_sites(module):
    source = ast.parse(open(module.__file__, encoding="utf-8").read())
    return sorted({
        node.args[0].id
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_poll_abort"
        and node.args
        and isinstance(node.args[0], ast.Name)
    })


def test_every_declared_site_has_a_call_and_every_call_a_declaration():
    report = declared_sites_match_the_release()
    assert set(report) == {"artifact_identity", "panel_v2", "parametric_umap"}
    for name, module in (("artifact_identity", artifact_identity), ("panel_v2", panel_v2)):
        declared = sorted(
            attribute for attribute in dir(module)
            if attribute.startswith("ABORT_POLL_SITE_")
        )
        assert _call_sites(module) == declared
        assert report[name]["sites_match"] is True


def test_a_removed_call_site_is_caught(tmp_path, monkeypatch):
    """POSITIVE CONTROL: delete a call site from the source the check reads."""
    source = open(panel_v2.__file__, encoding="utf-8").read()
    doctored = source.replace(
        "    _poll_abort(ABORT_POLL_SITE_PANEL_DENSITY)\n", "", 1
    )
    assert doctored != source, "the density call site was not found to remove"
    path = os.path.join(str(tmp_path), "panel_v2_doctored.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(doctored)
    monkeypatch.setattr(panel_v2, "__file__", path)
    with pytest.raises(Round0252Error):
        declared_sites_match_the_release()


def test_the_scorer_hook_rejects_a_non_callable():
    with pytest.raises(TypeError):
        panel_v2.set_abort_poll(object())
    assert panel_v2.abort_poll is None


def test_the_scorer_polls_its_bounded_loops_not_just_its_phases():
    """The gap is inside `_self_knn`, so the loop sites are the load-bearing ones."""
    source = ast.parse(open(panel_v2.__file__, encoding="utf-8").read())
    knn = next(
        node for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == "_self_knn"
    )
    inside = _names_called_in(knn)
    assert {
        "ABORT_POLL_SITE_KNN_CORPUS_CHUNK",
        "ABORT_POLL_SITE_KNN_TILE",
        "ABORT_POLL_SITE_KNN_EMIT_TILE",
    } <= inside
    for site in inside:
        parents = [
            node for node in ast.walk(knn)
            if isinstance(node, (ast.For, ast.While))
            and site in _names_called_in(node)
        ]
        assert parents, f"{site} is not inside a loop"


def _names_called_in(node):
    return {
        child.args[0].id
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "_poll_abort"
        and child.args
        and isinstance(child.args[0], ast.Name)
    }


# --------------------------------------------------------------------------- #
# the stop-latency instrument
# --------------------------------------------------------------------------- #


def test_stop_latency_measures_a_real_stop(tmp_path):
    flag = os.path.join(str(tmp_path), "control.abort")

    def run(poll):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            poll("unit of work")
            time.sleep(0.005)

    outcome = measure_stop_latency(
        label="synthetic", flag_path=flag, delay_s=0.2, run=run
    )
    assert outcome["the_work_stopped_cooperatively"] is True
    assert outcome["the_work_ran_to_completion_instead"] is False
    assert 0.0 <= outcome["stop_latency_s"] < 1.0
    assert outcome["stop_latency_meets_the_registered_ceiling"] is True
    assert not os.path.exists(flag), "the control left its flag behind"


def test_stop_latency_reports_a_failure_to_stop_rather_than_hiding_it(tmp_path):
    """POSITIVE CONTROL: a work loop that never polls must NOT read as stopped."""
    flag = os.path.join(str(tmp_path), "control.abort")

    def run(_poll):
        time.sleep(0.5)          # never calls the poll

    outcome = measure_stop_latency(
        label="unpolled", flag_path=flag, delay_s=0.1, run=run
    )
    assert outcome["the_work_stopped_cooperatively"] is False
    assert outcome["the_work_ran_to_completion_instead"] is True
    assert outcome["stop_latency_s"] is None
    assert outcome["stop_latency_meets_the_registered_ceiling"] is None


def test_the_control_never_writes_the_runner_flag(tmp_path):
    flag = os.path.join(str(tmp_path), "control.abort")
    poll = FlagFileAbortPoll(flag_path=flag)
    poll("no flag yet")
    open(flag, "w").close()
    with pytest.raises(Round0252CooperativeAbort):
        poll("flag present")
    assert poll.observed_site == "flag present"


def test_a_pre_existing_flag_is_refused(tmp_path):
    flag = os.path.join(str(tmp_path), "control.abort")
    open(flag, "w").close()
    with pytest.raises(Round0252Error):
        measure_stop_latency(
            label="stale", flag_path=flag, delay_s=0.1, run=lambda poll: None
        )


# --------------------------------------------------------------------------- #
# reporting: the census keys, the shortfall branch, the size law
# --------------------------------------------------------------------------- #


def test_gap_report_emits_the_keys_the_census_enumerates():
    report = gap_report([("a", 0.1), ("b", 2.0), ("a", 0.3)], arm="x")
    assert report["widest_gap_s"] == 2.0
    assert report["widest_gap_after"] == "b"
    assert "widest_gap_over_the_ceiling" in report
    assert "registered_ceiling_s" in report
    assert report["gaps_by_site"]["a"]["widest_gap_s"] == 0.3
    assert report["gaps_by_site"]["a"]["reads"] == 2


def test_gap_report_refuses_an_empty_series():
    with pytest.raises(Round0252Error):
        gap_report([], arm="x")


def test_the_shortfall_is_reported_in_the_under_ceiling_branch():
    """POSITIVE CONTROL for review-0251-01 §C.4.

    R0251's field was populated only when the CEILING was breached, so the
    margin-missed-but-under-ceiling branch that actually occurred published
    `null` and its control exercised the other branch. Both branches are planted
    here.
    """
    ceiling = gap_report([("a", 1.0)], arm="x")["registered_ceiling_s"]

    under_but_short = gap_reduction(
        before=gap_report([("a", 2.0)], arm="b"),
        after=gap_report([("a", ceiling / 1.5)], arm="a"),
    )
    assert under_but_short["is_below_the_ceiling"] is True
    assert under_but_short["is_below_the_ceiling_with_the_required_margin"] is False
    assert under_but_short["shortfall_over_the_ceiling_if_over"] is None
    assert under_but_short["shortfall_factor_against_the_required_margin_if_under"] == pytest.approx(2.0 / 1.5)

    over = gap_reduction(
        before=gap_report([("a", 10.0)], arm="b"),
        after=gap_report([("a", ceiling * 3.0)], arm="a"),
    )
    assert over["is_below_the_ceiling"] is False
    assert over["shortfall_over_the_ceiling_if_over"] == pytest.approx(3.0)
    assert over["shortfall_factor_against_the_required_margin_if_under"] is None

    clear = gap_reduction(
        before=gap_report([("a", 10.0)], arm="b"),
        after=gap_report([("a", ceiling / 5.0)], arm="a"),
    )
    assert clear["is_below_the_ceiling_with_the_required_margin"] is True
    assert clear["shortfall_over_the_ceiling_if_over"] is None
    assert clear["shortfall_factor_against_the_required_margin_if_under"] is None


def test_size_law_separates_a_flat_gap_from_a_linear_one():
    flat = size_law([
        {"bytes": 3_072_000_128, "widest_gap_s": 0.004},
        {"bytes": 49_152_000_128, "widest_gap_s": 0.0042},
        {"bytes": TARGET_SUBSTRATE_BYTES, "widest_gap_s": 0.0041},
    ])
    linear = size_law([
        {"bytes": 3_072_000_128, "widest_gap_s": 1.3},
        {"bytes": 49_152_000_128, "widest_gap_s": 20.8},
        {"bytes": TARGET_SUBSTRATE_BYTES, "widest_gap_s": 65.0},
    ])
    assert abs(flat["slope_seconds_per_byte"]) < 1e-12
    assert linear["slope_seconds_per_byte"] > 1e-11
    assert linear["gap_at_the_100m_substrate_under_the_linear_fit_over_the_ceiling"] > 10.0


def test_size_law_refuses_a_single_point():
    with pytest.raises(Round0252Error):
        size_law([{"bytes": 1, "widest_gap_s": 1.0}])


# --------------------------------------------------------------------------- #
# the tail
# --------------------------------------------------------------------------- #


def test_the_tail_estimator_is_r0251s_and_still_refuses_an_unidentified_fit():
    """POSITIVE CONTROL: a heavy-tailed series must NOT read as identified."""
    rng = np.random.default_rng(20260811)
    gaps = list(rng.pareto(0.7, size=20_000) * 0.001 + 0.001)
    out = tail_identification(gaps, arm_wall_s=180.0)
    assert out["the_estimator_is_r0251s_unchanged"] is True
    assert out["identification_limit"] == 10.0
    assert out["tail_verdict"]["the_extreme_value_fit_is_identified"] in (True, False)
    assert "plain_statement" in out["tail_verdict"]


def test_the_tail_compares_against_the_prior_rung():
    rng = np.random.default_rng(7)
    gaps = list(rng.exponential(0.008, size=20_000))
    out = tail_identification(
        gaps,
        arm_wall_s=180.0,
        prior_rung={
            "batches": 10_000,
            "threshold_ladder_return_level_spread": 86.69863492594678,
            "identified": False,
        },
    )
    against = out["against_the_prior_rung"]
    assert against["prior_batches"] == 10_000
    assert against["batches_now"] == 20_000
    assert against["batch_multiple"] == pytest.approx(2.0)


def test_prior_rung_reads_r0251s_sealed_shape():
    prior = prior_rung_from_artifact({
        "tail_model": {"peaks_over_threshold": {"batches_observed": 10_000}},
        "tail_verdict": {
            "threshold_ladder_return_level_spread": 86.69863492594678,
            "the_extreme_value_fit_is_identified": False,
        },
    })
    assert prior == {
        "batches": 10_000,
        "threshold_ladder_return_level_spread": 86.69863492594678,
        "identified": False,
    }


# --------------------------------------------------------------------------- #
# the round's own registered constants
# --------------------------------------------------------------------------- #


def test_the_target_size_is_the_100m_substrate_not_a_round_number():
    assert TARGET_SUBSTRATE_BYTES == TARGET_ROWS * TARGET_DIMENSION * 4
    assert TARGET_SUBSTRATE_BYTES == 153_600_000_000


def test_the_rung_is_sixty_times_r0251s():
    assert TAIL_RUNG_UPDATES == 600_000
    assert TAIL_RUNG_UPDATES == 60 * 10_000
    assert STOP_CONTROL_UPDATES == 3_000


def test_no_hidden_sigkill_in_the_prepare_path():
    """`subprocess.run(..., timeout=N)` is a hidden `Popen.kill()`; R0238 shipped one.

    Checked on the AST rather than on the text, so the round's own prose about
    the ban does not trip its own guard -- and so a `timeout` passed under a
    different spelling still does.
    """
    for name in (
        "experiments/prepare_round0252_queue.py",
        "experiments/round0252_nodes.py",
        "basemap/round0252_stoppability.py",
    ):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), name)
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not any(
                    keyword.arg == "timeout" for keyword in node.keywords
                ), f"{name} passes a timeout= (a hidden SIGKILL)"
                if isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {"kill", "terminate"}, (
                        f"{name} signals a process"
                    )
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"SIGKILL", "SIGTERM"}, f"{name} names {node.attr}"


def test_the_nodes_declare_no_registration():
    source = open(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "experiments/round0252_nodes.py"
        ),
        encoding="utf-8",
    ).read()
    assert '"gate_registered": False' in source
    assert '"published_a_map": False' in source
    assert '"is_a_family_cell": False' in source
