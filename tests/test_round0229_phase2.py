"""R0229 phase 2 — pre-launch checks on the arm rule and the trend statistic."""
from __future__ import annotations

import json
import os

import pytest

from basemap import round0229_phase2_contract as phase2
from basemap import round0229_quality_contract as contract


SWEEP_PATH = (
    "/data/latent-basemap/runs/round-0229/queue-correction-1/artifacts/"
    "minilm-mixed-2m-nnd-quality-sweep-v1/nnd-quality-sweep.json"
)
SPILL_PATH = (
    "/data/latent-basemap/runs/round-0229/queue-correction-1/artifacts/"
    "minilm-mixed-2m-spill-reachability-v1/spill-reachability.json"
)


def _load(path: str):
    if not os.path.exists(path):
        pytest.skip(f"phase-1 artifact absent: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_the_trigger_fired_on_structural_gain():
    sweep, spill = _load(SWEEP_PATH), _load(SPILL_PATH)
    trigger = contract.phase2_trigger(
        sweep_cells=sweep["cells"], spill_cells=spill["cells"],
        partition_strict_ceiling=float(
            sweep["shared_partition"]["strict_ceiling_all_rows"]
        ),
    )
    assert trigger["phase2_runs"] is True
    assert trigger["triggers"]["structural_gain"] is True
    # The nn-descent knob did NOT move recall by the registered amount, and the
    # bound was not violated. Both are recorded, not just the one that fired.
    assert trigger["triggers"]["tunable_gain"] is False
    assert trigger["triggers"]["bound_violated"] is False


def test_arm_selection_picks_a_100m_feasible_cell_by_the_registered_rule():
    sweep, spill = _load(SWEEP_PATH), _load(SPILL_PATH)
    arm = phase2.select_arm(sweep=sweep, spill=spill)
    chosen = next(c for c in spill["cells"] if c["cell"] == arm["cell"])
    assert chosen["feasible_at_100m"] is True
    best = max(
        (c for c in spill["cells"]
         if c.get("feasible_at_100m") and c.get("strict_ceiling_all_rows")),
        key=lambda c: c["strict_ceiling_all_rows"],
    )
    assert arm["cell"] == best["cell"]
    assert contract.rung_is_feasible(
        rows=100_000_000, clusters=arm["clusters"], spill=arm["spill"]
    )


def test_arm_nn_descent_setting_is_the_sweep_winner():
    sweep, spill = _load(SWEEP_PATH), _load(SPILL_PATH)
    arm = phase2.select_arm(sweep=sweep, spill=spill)
    best = max(
        (c for c in sweep["cells"] if c.get("scored")),
        key=lambda c: c["tie_aware_recall_all_rows"],
    )
    assert arm["nn_descent"]["cell"] == best["cell"]


def test_select_arm_refuses_when_nothing_is_feasible():
    sweep = _load(SWEEP_PATH)
    with pytest.raises(contract.Round0229Error):
        phase2.select_arm(sweep=sweep, spill={"cells": []})


def test_per_map_did_centres_each_arm_on_its_own_null():
    values = phase2.per_map_did(
        candidate_gaps=[1.0, 2.0, 3.0], exact_gaps=[0.5, 1.5]
    )
    assert values == [0.0, 1.0, 2.0]
    with pytest.raises(contract.Round0229Error):
        phase2.per_map_did(candidate_gaps=[1.0], exact_gaps=[0.5])


def test_trend_arms_are_three_and_ordered_low_to_high_loss():
    assert phase2.TREND_ARMS == ("c4", phase2.ARM_NAME, "c16")
    assert len(phase2.SEEDS) == 3
    assert phase2.TREATMENT_INVARIANT_SHA256.startswith("c28cfd61")


def test_capabilities_are_r0229s_own_names_not_r0228s():
    for seed in phase2.SEEDS:
        name = phase2.map_capability(seed)
        assert "spill-lifted" in name
        assert "cluster-spill-c" not in name
    assert phase2.ADOPTION_CLAIMED is False
    assert phase2.GATE_REGISTERABLE_HERE is False
    assert phase2.EQUIVALENCE_CLAIMED is False


def test_phase1_artifacts_are_hash_bound_not_identity_sealed(tmp_path):
    """Regression: phase 1's artifacts carry no `prompt_contract` identity seal.

    They are written with `atomic_write_new_json` and bound by their
    {path, bytes, sha256} signature. Demanding `read_sealed` of them failed the
    phase-2 build node pre-CUDA; the resolver must verify the signature and not
    require a seal the artifact never claimed.
    """
    from basemap.artifact_identity import expected_input_signature
    from experiments import round0229_phase2_nodes as nodes

    payload = {"cells": [], "note": "not identity sealed"}
    path = tmp_path / "phase1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    signature = expected_input_signature(str(path))
    loaded, echoed = nodes._verified_json(
        {"phase1": signature}, "phase1", label="phase 1"
    )
    assert loaded == payload
    assert echoed["sha256"] == signature["sha256"]


def test_the_real_phase1_artifacts_load_through_the_resolver():
    from basemap.artifact_identity import expected_input_signature
    from experiments import round0229_phase2_nodes as nodes

    for path in (SWEEP_PATH, SPILL_PATH):
        if not os.path.exists(path):
            pytest.skip("phase-1 artifact absent")
        signature = expected_input_signature(path)
        loaded, _ = nodes._verified_json({"a": signature}, "a", label="phase 1")
        assert "cells" in loaded


def test_the_build_artifact_is_identity_sealed_because_fuzzy_reads_it_sealed():
    """Regression: the phase-2 build artifact is consumed by `read_sealed`.

    Writing it with a bare `atomic_write_new_json` failed the fuzzy node
    pre-CUDA after the build had already spent its GPU time. An intra-queue
    artifact a later node reads as a *document* must carry an identity seal.
    """
    import inspect

    from experiments import round0229_phase2_nodes as nodes

    build_source = inspect.getsource(nodes.run_build)
    assert "prompt_contract.seal(" in build_source
    fuzzy_source = inspect.getsource(nodes.run_fuzzy)
    assert "read_sealed" in fuzzy_source
