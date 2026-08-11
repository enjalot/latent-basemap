"""R0257 contract tests — the CUDA-hidden CPU smoke the protocol requires.

Every guard this round adds ships a positive control here that **plants the defect
and proves the guard catches it**. A control that only exercises the honest input
certifies nothing (R0238), and a control that re-implements the guard it is testing
tests itself (review-0253), so every assertion below calls the SHIPPED function.
"""
from __future__ import annotations

import ast
import copy
import json
import os

import pytest

from basemap.round0255_treatment import Round0255FamilyError
from basemap.round0257_judgement import (
    GATED_METRICS,
    ONE_SIDED_POWER,
    PANEL_FALSE_ALARM_RATE,
    REGISTERED_FFR_FLOOR,
    REGISTERED_K1024_BAND,
    REGISTERED_K256_BAND,
    TWO_SIDED_POWER,
    Round0257JudgementError,
    judge_map,
    judge_population,
    judgement_controls,
    validate_gate_artifact,
)
from basemap.round0257_rung_contract import (
    REGISTERED_SUCCESSFUL_UPDATES,
    REGISTERED_UPDATE_BOUND,
    RUNG_BEARING_PATHS,
    RUNG_ROWS,
    Round0257Error,
    Round0257FamilyError,
    SEALED_RUNG_DIRECTED_EDGES,
    SEEDS,
    assert_no_rung_map_in_the_gate_family,
    dose_view,
    r0217_template,
    rung_cell_id,
    rung_cell_ids,
    rung_family_purity_controls,
    rung_invariant_sha256,
    rung_train_config,
)
from basemap.round0257_rung_pipeline import (
    RungMixedTrainingInput,
    predict_rung_footprint,
    validate_full_rung_map,
)

R0256_GATE = (
    "/data/latent-basemap/runs/round-0256/queue-correction-1/artifacts/"
    "minilm-mixed-2m-calibrated-madn-floors-n29-v2/"
    "minilm-calibrated-madn-floors-n29-repaired.json"
)
DEFINING_FAMILY = [f"exact-seed{seed}" for seed in range(42, 71)]

SUB = {"kind": "file", "canonical_path": "/x/rung.npy", "sha256": "d" * 64, "bytes": 1}
GRAPH = {"kind": "file", "canonical_path": "/x/r.npz", "sha256": "e" * 64, "bytes": 2}
GMAN = {"kind": "file", "canonical_path": "/x/rm.json", "sha256": "f" * 64, "bytes": 3}
R2SUB = {"kind": "file", "canonical_path": "/x/s.npy", "sha256": "a" * 64, "bytes": 4}
R2GRAPH = {"kind": "file", "canonical_path": "/x/e.npz", "sha256": "b" * 64, "bytes": 5}
R2GMAN = {"kind": "file", "canonical_path": "/x/m.json", "sha256": "c" * 64, "bytes": 6}


def _config(seed: int = 42, **overrides):
    kwargs = dict(
        seed=seed,
        rows=RUNG_ROWS,
        graph_edges=SEALED_RUNG_DIRECTED_EDGES,
        substrate_signature=SUB,
        graph_signature=GRAPH,
        graph_manifest_signature=GMAN,
        r0217_substrate_signature=R2SUB,
        r0217_graph_signature=R2GRAPH,
        r0217_graph_manifest_signature=R2GMAN,
    )
    kwargs.update(overrides)
    return rung_train_config(**kwargs)


@pytest.fixture(scope="module")
def gate_artifact():
    if not os.path.exists(R0256_GATE):
        pytest.skip("the sealed n=29 gate artifact is not on this machine")
    with open(R0256_GATE, "rb") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# the rung config
# --------------------------------------------------------------------------- #


def test_the_horizon_is_the_registered_ceil_and_within_the_bound():
    view = dose_view(SEALED_RUNG_DIRECTED_EDGES)
    assert view["successful_updates"] == REGISTERED_SUCCESSFUL_UPDATES == 255_142
    assert view["successful_updates"] <= REGISTERED_UPDATE_BOUND
    assert (
        abs(
            view["achieved_positive_draws_per_edge"]
            - view["target_positive_draws_per_edge"]
        )
        <= view["dose_quantum_draws_per_edge"]
    )


def test_all_three_cells_share_one_rung_invariant_and_differ_in_full_digest():
    invariants = set()
    digests = set()
    for seed in SEEDS:
        config, sha, invariant = _config(seed)
        invariants.add(invariant)
        digests.add(sha)
        assert config["optimizer"]["successful_positive_lr_updates"] == 255_142
        assert config["input"]["rows"] == RUNG_ROWS
        assert config["family_invariant"]["rows"] == RUNG_ROWS
        assert config["seed_family"]["cells_required_for_gate"] == 0
    assert len(invariants) == 1
    assert len(digests) == len(SEEDS)


def test_the_rung_invariant_equals_r0217s_own_template_under_the_mask():
    _config_value, _sha, invariant = _config(42)
    template = r0217_template(
        substrate_signature=R2SUB,
        graph_signature=R2GRAPH,
        graph_manifest_signature=R2GMAN,
    )
    assert invariant == rung_invariant_sha256(template)


@pytest.mark.parametrize(
    "path",
    [
        ("model", "hidden_dimension"),
        ("optimizer", "learning_rate"),
        ("optimizer", "batch_size"),
        ("optimizer", "use_amp"),
        ("execution", "minimum_train_upd_s"),
    ],
)
def test_a_treatment_change_breaks_the_rung_invariant(path):
    """POSITIVE CONTROL: perturb a field the mask does NOT cover; the digest must
    move. Without this the invariant could be masking everything."""
    template = r0217_template(
        substrate_signature=R2SUB,
        graph_signature=R2GRAPH,
        graph_manifest_signature=R2GMAN,
    )
    baseline = rung_invariant_sha256(template)
    tampered = copy.deepcopy(template)
    cursor = tampered
    for key in path[:-1]:
        cursor = cursor[key]
    value = cursor[path[-1]]
    cursor[path[-1]] = (value * 2) if isinstance(value, (int, float)) else "tampered"
    assert rung_invariant_sha256(tampered) != baseline


def test_the_mask_does_not_cover_the_pipeline_identity_strings():
    """The four strings left unmasked must genuinely be inside the invariant."""
    dotted = {".".join(path) for path in RUNG_BEARING_PATHS}
    for name in (
        "execution.required_pipeline",
        "execution.expected_pipeline_stamp.pipeline",
        "execution.expected_pipeline_stamp.schema",
        "execution.expected_pipeline_stamp.source_representation",
    ):
        assert name not in dotted


def test_a_wrong_rung_or_a_wrong_edge_count_is_refused():
    with pytest.raises(Round0257Error):
        _config(42, rows=2_000_000)
    with pytest.raises(Round0257Error):
        _config(42, graph_edges=SEALED_RUNG_DIRECTED_EDGES + 1)
    with pytest.raises(Round0257Error):
        _config(99)


# --------------------------------------------------------------------------- #
# the pipeline subclass
# --------------------------------------------------------------------------- #


class _FakeDataset:
    def __init__(self, rows):
        self.shape = (rows, 384)

    def __len__(self):
        return self.shape[0]


def test_the_rung_pipeline_asserts_the_rungs_own_cardinality():
    graph = {"n_nodes": RUNG_ROWS}
    ok = RungMixedTrainingInput(_FakeDataset(RUNG_ROWS), graph, seed=42)
    assert ok.shape == (RUNG_ROWS, 384)
    with pytest.raises(Round0257Error):
        RungMixedTrainingInput(_FakeDataset(2_000_000), graph, seed=42)
    with pytest.raises(Round0257Error):
        RungMixedTrainingInput(
            _FakeDataset(RUNG_ROWS), {"n_nodes": 2_000_000}, seed=42
        )


def test_the_subclass_overrides_only_its_geometry_assertion():
    own = {
        name
        for name, value in vars(RungMixedTrainingInput).items()
        if callable(value) and not name.startswith("__module__")
    }
    assert own == {"__init__"}


def test_the_map_validator_refuses_a_short_or_nonfinite_map():
    import numpy as np

    good = np.random.RandomState(0).normal(size=(64, 2)).astype(np.float32)
    assert validate_full_rung_map(good, rows=64)["full_population_finite"] is True
    with pytest.raises(Round0257Error):
        validate_full_rung_map(good, rows=65)
    bad = good.copy()
    bad[3, 1] = np.nan
    with pytest.raises(Round0257Error):
        validate_full_rung_map(bad, rows=64)


def test_the_footprint_prediction_is_labelled_and_guards_anonymous_memory():
    prediction = predict_rung_footprint(42)
    assert "PREDICTION" in prediction["prediction_basis"]
    assert prediction["guarded_on"].startswith("host ANONYMOUS")


# --------------------------------------------------------------------------- #
# the family rule
# --------------------------------------------------------------------------- #


def test_the_honest_family_passes_and_is_disjoint_from_the_judged_set():
    verdict = assert_no_rung_map_in_the_gate_family(DEFINING_FAMILY)
    assert verdict["judged_and_defining_are_disjoint"] is True
    assert verdict["n_defining"] == 29
    assert set(verdict["judged_cell_ids"]) == set(rung_cell_ids())


@pytest.mark.parametrize("seed", SEEDS)
def test_a_rung_map_in_the_family_is_refused(seed):
    """POSITIVE CONTROL: the v0 defect, planted one map at a time."""
    with pytest.raises((Round0255FamilyError, Round0257FamilyError)):
        assert_no_rung_map_in_the_gate_family(DEFINING_FAMILY + [rung_cell_id(seed)])


def test_the_shipped_family_controls_all_fire():
    controls = rung_family_purity_controls(DEFINING_FAMILY)
    assert controls["every_planted_defect_was_refused"] is True
    assert controls["the_old_predicate_accepted_every_one"] is True
    assert controls["the_honest_family_still_passes"] is True
    assert controls["planted"] == len(SEEDS) + 1


def test_a_held_out_cell_in_the_family_is_still_refused():
    with pytest.raises(Round0255FamilyError):
        assert_no_rung_map_in_the_gate_family(
            DEFINING_FAMILY + ["cluster-spill-c8-seed42"]
        )


# --------------------------------------------------------------------------- #
# the judgement
# --------------------------------------------------------------------------- #


def test_the_sealed_gate_validates_and_carries_the_registered_values(gate_artifact):
    gate = validate_gate_artifact(gate_artifact)
    assert gate["n"] == 29
    criteria = gate["registered_criteria"]
    assert set(criteria) == set(GATED_METRICS)
    assert criteria["ffr"]["floor"] == REGISTERED_FFR_FLOOR
    assert (
        criteria["purity_fidelity_k256"]["ratio_lower"],
        criteria["purity_fidelity_k256"]["ratio_upper"],
    ) == REGISTERED_K256_BAND
    assert (
        criteria["purity_fidelity_k1024"]["ratio_lower"],
        criteria["purity_fidelity_k1024"]["ratio_upper"],
    ) == REGISTERED_K1024_BAND


def test_every_judgement_plant_is_refused_and_every_behavioural_control_holds(
    gate_artifact,
):
    controls = judgement_controls(gate_artifact)
    assert controls["every_planted_defect_was_refused"] is True
    assert controls["every_behavioural_control_held"] is True
    assert controls["the_honest_artifact_still_passes"] is True
    assert controls["the_old_predicate_accepted_every_one"] is True
    assert controls["planted"] == 9


def test_power_is_matched_to_sidedness(gate_artifact):
    gate = validate_gate_artifact(gate_artifact)
    verdict = judge_map(
        cell_id="probe",
        metrics={
            "ffr": REGISTERED_FFR_FLOOR + 0.01,
            "purity_fidelity_k256": 1.0,
            "purity_fidelity_k1024": 1.0,
            "density_v2": 0.44,
        },
        raw_ratios={"k256": 1.0, "k1024": 0.71},
        gate=gate,
    )
    assert verdict["per_metric"]["ffr"]["applicable_power"] == "one_sided"
    assert (
        verdict["per_metric"]["ffr"]["detection_power"]["minus_2_sigma"]
        == ONE_SIDED_POWER["minus_2_sigma"]
    )
    for metric in ("purity_fidelity_k256", "purity_fidelity_k1024"):
        assert verdict["per_metric"][metric]["applicable_power"] == "two_sided"
        assert (
            verdict["per_metric"][metric]["detection_power"]["minus_2_sigma"]
            == TWO_SIDED_POWER["minus_2_sigma"]
        )
    assert verdict["panel_false_alarm_rate"] == PANEL_FALSE_ALARM_RATE


def test_the_purity_criteria_read_the_unfolded_ratio_not_the_folded_fidelity(
    gate_artifact,
):
    """A folded fidelity is always in (0, 1]; the k256 band's upper edge is above
    1.0, so a judge reading the folded value could never fail on over-separation."""
    gate = validate_gate_artifact(gate_artifact)
    verdict = judge_map(
        cell_id="probe",
        metrics={
            "ffr": REGISTERED_FFR_FLOOR + 0.01,
            # a folded fidelity of 1.0 would sit inside the band
            "purity_fidelity_k256": 1.0,
            "purity_fidelity_k1024": 1.0,
            "density_v2": 0.44,
        },
        # the UNFOLDED ratio is far above the band
        raw_ratios={"k256": 1.4, "k1024": 0.71},
        gate=gate,
    )
    assert verdict["verdict"] == "FAIL"
    assert verdict["failing_criteria"] == ["purity_fidelity_k256"]
    assert verdict["per_metric"]["purity_fidelity_k256"]["observed"] == 1.4


def test_a_population_verdict_reports_margins_power_and_the_false_alarm_rate(
    gate_artifact,
):
    gate = validate_gate_artifact(gate_artifact)
    cells = {
        rung_cell_id(42): {
            "panel_metrics": {
                "ffr": REGISTERED_FFR_FLOOR + 0.02,
                "purity_fidelity_k256": 1.0,
                "purity_fidelity_k1024": 1.0,
                "density_v2": 0.44,
            },
            "raw_purity_ratios": {"k256": 1.0, "k1024": 0.71},
        },
        rung_cell_id(43): {
            "panel_metrics": {
                "ffr": REGISTERED_FFR_FLOOR - 0.02,
                "purity_fidelity_k256": 1.0,
                "purity_fidelity_k1024": 1.0,
                "density_v2": 0.44,
            },
            "raw_purity_ratios": {"k256": 1.0, "k1024": 0.71},
        },
    }
    outcome = judge_population(cells=cells, gate=gate)
    assert outcome["maps_passing"] == [rung_cell_id(42)]
    assert outcome["maps_failing"] == [rung_cell_id(43)]
    assert outcome["unanimous"] is False
    assert outcome["panel_false_alarm_rate"] == PANEL_FALSE_ALARM_RATE
    assert "KNOWN LIMITATION" in outcome["independence_limitation"]
    margins = outcome["by_metric"]["ffr"]["margins"]
    assert margins[rung_cell_id(42)] > 0 > margins[rung_cell_id(43)]


def test_a_gate_artifact_with_a_moved_floor_is_refused(gate_artifact):
    tampered = copy.deepcopy(gate_artifact)
    tampered["registered_criteria"]["ffr"]["floor"] = 0.31
    with pytest.raises(Round0257JudgementError):
        validate_gate_artifact(tampered)


# --------------------------------------------------------------------------- #
# safety
# --------------------------------------------------------------------------- #


ROUND_SOURCES = (
    "basemap/round0257_rung_contract.py",
    "basemap/round0257_judgement.py",
    "basemap/round0257_rung_pipeline.py",
    "experiments/round0257_nodes.py",
    "experiments/prepare_round0257_queue.py",
    "tests/test_round0257_contract.py",
)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_subprocess_timeout_anywhere_in_this_rounds_sources():
    """`subprocess.run(..., timeout=N)` is a hidden SIGKILL; the AST guard is blind
    to it, so this is the grep the standing rule asks for."""
    offenders = []
    for relative in ROUND_SOURCES:
        path = os.path.join(_repo_root(), relative)
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = getattr(target, "attr", None) or getattr(target, "id", None)
            if name not in {"run", "call", "check_call", "check_output", "communicate"}:
                continue
            for keyword in node.keywords:
                if keyword.arg == "timeout":
                    offenders.append(f"{relative}:{node.lineno}")
    assert offenders == []


def test_no_signalling_call_anywhere_in_this_rounds_sources():
    banned = {"kill", "terminate", "killpg", "send_signal", "pkill"}
    offenders = []
    for relative in ROUND_SOURCES:
        path = os.path.join(_repo_root(), relative)
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name in banned:
                    offenders.append(f"{relative}:{node.lineno}")
    assert offenders == []


def test_the_nodes_module_dispatches_all_three_registered_actions():
    from experiments import round0257_nodes as nodes

    assert nodes.TRAIN_ACTION.startswith("train_")
    assert nodes.PANEL_ACTION.startswith("score_")
    assert nodes.JUDGE_ACTION.startswith("judge_")
    with pytest.raises(Round0257Error):
        nodes.run_job({"manifest": {"round_id": "0257"}}, {"action": "nope"})
