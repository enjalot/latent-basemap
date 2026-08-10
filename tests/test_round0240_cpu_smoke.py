"""R0240 CUDA-hidden CPU smoke — reach the paths a GPU queue would.

Preparation validation, not a scientific result: run the real verification,
attestation and seal path on toy inputs so a late NameError, an accounting
shape drift or a serialization failure surfaces before hours of GPU time.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature
from basemap.round0238_rung5 import GUARD_IMBALANCE_MARGIN
import basemap.round0240_rung5 as rung
from basemap.round0240_rung5 import (
    Round0240Error,
    ladder_attestation,
    qualification_attestation,
)
import experiments.round0240_nodes as nodes
import experiments.prepare_round0240_queue as prepare


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(nodes.__file__)))
NEW_FILES = (
    "basemap/round0240_rung5.py",
    "experiments/round0240_nodes.py",
    "experiments/prepare_round0240_queue.py",
    "tests/test_round0240_contract.py",
    "tests/test_round0240_cpu_smoke.py",
)


def _job() -> dict:
    return {
        "substrate_manifest": expected_input_signature(
            rung.INHERITED_SUBSTRATE_MANIFEST
        ),
        "truth_reference": expected_input_signature(rung.INHERITED_TRUTH_MANIFEST),
        "reachability_reference": expected_input_signature(
            rung.INHERITED_REACHABILITY_MANIFEST
        ),
    }


def test_dispatch_refuses_an_unregistered_action():
    with pytest.raises(Round0240Error, match="does not authorize"):
        nodes.run_job({"manifest": {}}, {"action": "train_a_map"})


def test_the_inheritance_verification_closes_on_the_live_artifacts():
    record = nodes.verify_inheritance(_job())
    assert record["substrate"]["verified"] is True
    assert record["truth"]["verified"] is True
    assert record["reachability"]["verified"] is True
    assert record["substrate"]["source"]["sha256"]
    assert record["substrate"]["ordered_substrate_sha256"] == (
        rung.REGISTERED_ORDERED_SUBSTRATE_SHA256
    )


def test_an_intra_queue_reference_is_refused_for_an_inherited_artifact():
    """Inheritance across queues must be bound by sha256, never by path alone."""
    job = _job()
    job["substrate_manifest"] = {
        "kind": "file",
        "canonical_path": rung.INHERITED_SUBSTRATE_MANIFEST,
    }
    with pytest.raises(Round0240Error, match="full sha256 signature"):
        nodes.verify_inheritance(job)


def test_changed_manifest_bytes_are_refused_before_any_gpu_work(tmp_path):
    sealed = json.loads(
        open(rung.INHERITED_SUBSTRATE_MANIFEST, encoding="utf-8").read()
    )
    copy = tmp_path / "substrate.json"
    copy.write_text(json.dumps(sealed), encoding="utf-8")
    job = _job()
    job["substrate_manifest"] = {
        **expected_input_signature(str(copy)), "sha256": "0" * 64,
    }
    with pytest.raises(Exception, match="content changed"):
        nodes.verify_inheritance(job)


# --------------------------------------------------------------------------- #
# the attestations
# --------------------------------------------------------------------------- #
def _ladder_receipt(*, per_seed: dict[int, float], selected: int = 400) -> dict:
    worst = max(per_seed.values())
    mean_cluster_rows = 2_000_000.0
    guarded = worst * GUARD_IMBALANCE_MARGIN * mean_cluster_rows
    return {
        "rows": rung.ROWS,
        "cluster_selection": {
            "selected_clusters": selected,
            "candidates_considered": [
                {
                    "clusters": selected,
                    "admissible": True,
                    "guarded_max_cluster_rows": guarded,
                    "admissible_max_cluster_rows": (
                        rung.R0238_ADMISSIBLE_MAX_CLUSTER_ROWS
                    ),
                },
            ],
        },
        "worst_seed_imbalance_at_this_rung": {selected: worst},
        "primary_seed_imbalance_at_this_rung": {selected: per_seed[226]},
        "measured_imbalance": {
            "cells": [
                {
                    "rows": rung.ROWS, "clusters": selected, "seed": seed,
                    "imbalance_max_over_mean": value,
                }
                for seed, value in sorted(per_seed.items())
            ],
        },
        "wall_budget": {"fits": True, "predicted_build_wall_s": 18_237.0},
        "ladder_stopped_at": None,
        "ladder": [{"clusters": selected, "run": True, "guard": {}}],
    }


def test_the_ladder_attestation_publishes_the_measured_tolerance():
    receipt = _ladder_receipt(
        per_seed=dict(rung.R0238_MEASURED_IMBALANCE_BY_SEED)
    )
    record = ladder_attestation(ladder=receipt, release_sha="a" * 40)
    assert record["round_id"] == "0240"
    assert record["selected_clusters"] == 400
    assert record["selected_differs_from_registered"] is False
    assert record["measured_tolerance_at_selected_c"] == pytest.approx(
        rung.R0238_REALISED_TOLERANCE_AT_C400, abs=1e-9
    )
    assert record["measured_tolerance_percent"] == pytest.approx(51.398403, abs=1e-5)
    carried = record["carried_prediction"]
    assert carried["measured_exceeds_carried_prediction"] is False
    assert carried["excess_over_carried_prediction"] == pytest.approx(0.0, abs=1e-12)
    assert set(carried["per_seed_movement_from_r0238"]) == {
        "226", "236", "1236", "2236", "3236"
    }
    for entry in carried["per_seed_movement_from_r0238"].values():
        assert entry["relative_move"] == pytest.approx(0.0, abs=1e-12)


def test_the_attestation_reports_an_adverse_move_when_one_happens():
    worse = {
        seed: value * 1.1
        for seed, value in rung.R0238_MEASURED_IMBALANCE_BY_SEED.items()
    }
    record = ladder_attestation(
        ladder=_ladder_receipt(per_seed=worse), release_sha="a" * 40
    )
    carried = record["carried_prediction"]
    assert carried["measured_exceeds_carried_prediction"] is True
    assert carried["excess_over_carried_prediction"] == pytest.approx(0.1, abs=1e-9)
    # a worse imbalance must consume tolerance, never create it
    assert (
        record["measured_tolerance_at_selected_c"]
        < rung.R0238_REALISED_TOLERANCE_AT_C400
    )


def test_the_attestation_flags_a_c_that_differs_from_the_registered_one():
    receipt = _ladder_receipt(
        per_seed=dict(rung.R0238_MEASURED_IMBALANCE_BY_SEED), selected=800
    )
    record = ladder_attestation(ladder=receipt, release_sha="a" * 40)
    assert record["selected_clusters"] == 800
    assert record["selected_differs_from_registered"] is True


def _qualified_receipt(*, in_zero: int = 0, out_zero: int = 0) -> dict:
    return {
        "rows": rung.ROWS,
        "k": 15,
        "selected_clusters": 400,
        "selected_cell": "r0238-n100000000-c400-s8",
        "recall_population": "uniform probe",
        "structural_population": "all 100,000,000 rows, both in and out",
        "probe_rows": 500_000,
        "probe_seed": 238_000,
        "selected_graph": {
            "strict": {"mean": 0.997, "p10": 1.0, "min": 0.0, "n": 500_000},
            "tie_aware": {"mean": 0.999, "p10": 1.0, "min": 0.0, "n": 500_000},
            "density_decile_strict": [0.99] * 10,
            "density_decile_tie_aware": [0.999] * 10,
            "rows_carrying_any_loss": 6_907,
            "fraction_carrying_any_loss": 0.013814,
            "missing_true_edges": 7_500,
            "tie_aware_rows_at_zero": 15,
            "zero_recall_forensic": {"verified_duplicate_family_rows": 15},
            "structural": {"zero_degree_rows": 0},
        },
        "degrees": {
            "population": "all 100,000,000 rows, both in-degree and out-degree",
            "in_degree_zero_rows": in_zero,
            "out_degree_zero_rows": out_zero,
            "in_degree_min": 0 if in_zero else 5,
            "out_degree_min": 0 if out_zero else 5,
        },
        "directed_edges": 2_510_182_652,
        "edges_per_row": 25.10182652,
        "fuzzy_weight_range": [1e-9, 1.0],
        "per_rung_derivation": {
            str(rung.ROWS): {"selected_tolerance": {
                "tolerance_to_adverse_imbalance": 0.513984
            }},
        },
        "imbalance_table": {"measured_doubling_50m_to_100m": {}},
        "io_observed_regime": {"regime_observed": "page-cache-thrashing"},
        "graph": {"sha256": "b" * 64},
        "neighbour_ids": {"sha256": "c" * 64},
    }


def test_the_qualification_attestation_carries_the_r0215_tripwire_both_ways():
    record = qualification_attestation(
        qualified=_qualified_receipt(), release_sha="a" * 40
    )
    assert record["round_id"] == "0240"
    assert record["r0215_tripwire"]["holds"] is True
    assert record["r0215_tripwire"]["in_degree_zero_rows"] == 0
    assert record["r0215_tripwire"]["out_degree_zero_rows"] == 0
    assert record["floors"]["zero_degree_rows_in_and_out"] == 0
    assert record["strict_recall_at_15"]["min"] == 0.0
    assert record["tie_aware_rows_at_zero"] == 15
    assert len(record["density_decile_tie_aware"]) == 10
    assert record["measured_tolerance_at_this_rung"][
        "tolerance_to_adverse_imbalance"
    ] == pytest.approx(0.513984)


@pytest.mark.parametrize("in_zero,out_zero", [(1, 0), (0, 1), (3, 7)])
def test_the_tripwire_does_not_hold_when_either_direction_is_nonzero(
    in_zero, out_zero
):
    record = qualification_attestation(
        qualified=_qualified_receipt(in_zero=in_zero, out_zero=out_zero),
        release_sha="a" * 40,
    )
    assert record["r0215_tripwire"]["holds"] is False


def test_seal_and_reload_round_trips_the_attestation(tmp_path):
    from basemap.output_safety import atomic_write_new_json
    from basemap.round0238_rung5 import json_safe

    body = ladder_attestation(
        ladder=_ladder_receipt(
            per_seed=dict(rung.R0238_MEASURED_IMBALANCE_BY_SEED)
        ),
        release_sha="a" * 40,
    )
    receipt = prompt_contract.seal(json_safe(body))
    path = str(tmp_path / rung.LADDER_ATTEST_FILE)
    atomic_write_new_json(path, receipt, immutable=True)
    reloaded = prompt_contract.read_sealed(path, label="R0240 attestation")
    assert reloaded["round_id"] == "0240"
    assert reloaded["measured_tolerance_percent"] == pytest.approx(
        51.398403, abs=1e-5
    )


# --------------------------------------------------------------------------- #
# safety
# --------------------------------------------------------------------------- #
def test_the_signal_safety_detector_is_clean_on_every_file_this_round_adds():
    completed = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "experiments/check_signal_safety.py"),
         *[os.path.join(REPO_ROOT, name) for name in NEW_FILES]],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_no_round0240_file_can_signal_anything():
    """A grep, because the AST guard is structurally blind to the delegated kill.

    `timeout=` is in the forbidden set deliberately and with no waiver: CPython
    implements `subprocess.run(..., timeout=N)` as `Popen.kill()`, and R0238's
    fired against a live cuML child. Not one file this round adds carries it,
    including the CUDA-hidden preparation smoke.
    """
    #: The shell signal binaries are deliberately NOT named here - the detector
    #: in the test above owns that rule, and naming them in a literal would make
    #: this file its own finding.
    forbidden = (
        "os.kill", "os.killpg", "signal.signal", "signal.alarm",
        ".terminate(", ".send_signal(", "SIGKILL", "SIGTERM", "py-spy",
        "ptrace", "timeout=",
    )
    for name in NEW_FILES:
        source = open(os.path.join(REPO_ROOT, name), encoding="utf-8").read()
        for token in forbidden:
            #: this test names the tokens it forbids, so skip its own body
            body = source.split("def test_no_round0240_file_can_signal", 1)[0]
            assert token not in body, f"{name} contains {token!r}"


def test_the_preparation_script_declares_two_gpu_nodes_and_no_assembly():
    source = open(
        os.path.join(REPO_ROOT, "experiments/prepare_round0240_queue.py"),
        encoding="utf-8",
    ).read()
    assert '"id": "ladder_100000k"' in source
    assert '"id": "qualify_100000k"' in source
    assert '"id": "assemble' not in source
    #: The runner never signals a node that declares gpu_required, and the
    #: default when the key is absent is also True (R0239, workshop branch
    #: runner-signal-safety). Both nodes declare it explicitly anyway.
    assert '"gpu_required": True' in source
    assert prepare.HANDLER_MODULE == "experiments.round0240_nodes"


def test_the_round_calls_r0238s_reviewed_functions_rather_than_copying_them():
    """The registered checks must be the reviewed ones, not a re-typing of them."""
    import experiments.round0238_nodes as r0238

    assert nodes._r0238_run_ladder is r0238.run_ladder
    assert nodes._r0238_run_qualify is r0238.run_qualify
    #: and nothing in this round rebinds anything inside them
    source = open(
        os.path.join(REPO_ROOT, "experiments/round0240_nodes.py"),
        encoding="utf-8",
    ).read()
    for token in ("setattr(", "monkeypatch", "r0238.", "round0238_nodes."):
        assert token not in source, token
