"""C1 — the closure gate must FLIP to false when any source class is mutated.

Each test loads the REAL pinned evidence, corrupts one field of one source class,
and asserts the corresponding gate check fails. If the evidence is not present
(pre-G0 checkout) the test is skipped rather than giving a false green.
"""
import copy, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
import experiments.gate_summary as gs

REAL = {name: gs._load(path) for name, path in gs.SOURCE_FILES.items()}


def _need(name):
    if REAL.get(name) is None:
        pytest.skip(f"pinned source {name} not present in this checkout")
    return copy.deepcopy(REAL[name])


def _patch(monkeypatch, path_substr, mutated):
    real_load = gs._load
    def fake(path):
        return mutated if path_substr in path else real_load(path)
    monkeypatch.setattr(gs, "_load", fake)


def test_baseline_gate_is_green():
    # sanity: with the real evidence, every check passes (else mutation tests are
    # meaningless). Skips cleanly if evidence is absent.
    for name in ("ablation_2x2", "golden_2m_extended", "kernel_decision_v22"):
        if REAL.get(name) is None:
            pytest.skip("evidence absent")
    assert gs.check_2x2()[0] and gs.check_golden()[0] and gs.check_kernel_v22()[0]


def test_2x2_mutation_drops_a_cell(monkeypatch):
    d = _need("ablation_2x2")
    k = next(iter(d["runs"])); d["runs"].pop(k)          # 11 maps, broken cross-product
    _patch(monkeypatch, "r1_ablation_mn4", d)
    assert gs.check_2x2()[0] is False


def test_golden_mutation_breaks_tolerance(monkeypatch):
    d = _need("golden_2m_extended")
    m = next(iter(d["reference"])); d["streamed"][m] = d["reference"][m] + 999.0
    _patch(monkeypatch, "golden_2m_extended", d)
    assert gs.check_golden()[0] is False


def test_golden_mutation_missing_triplet(monkeypatch):
    d = _need("golden_2m_extended")
    d["streamed"].pop("ffr", None)                       # incomplete triplet must FAIL
    _patch(monkeypatch, "golden_2m_extended", d)
    assert gs.check_golden()[0] is False


def test_a3_mutation_flips_decision(monkeypatch):
    d = _need("a3_rescore")
    d["decision"] = "mn4"                                # != recomputed nomn
    _patch(monkeypatch, "a3_rescore", d)
    assert gs.check_a3()[0] is False


def test_a3_mutation_dirty_tree(monkeypatch):
    d = _need("a3_rescore")
    d.setdefault("provenance", {})["code_dirty"] = True
    _patch(monkeypatch, "a3_rescore", d)
    assert gs.check_a3()[0] is False


def test_kernel_mutation_reference_not_reused(monkeypatch):
    d = _need("kernel_200k_v22")
    first = next(iter(d["runs"]))
    d["runs"][first]["hiD_reference_reused"] = False
    _patch(monkeypatch, "complete_200k_v22", d)
    assert gs.check_kernel_v22()[0] is False


def test_kernel_mutation_dirty_scorer(monkeypatch):
    d = _need("kernel_2m_v22"); d["scorer_dirty"] = True
    _patch(monkeypatch, "complete_2m_v22", d)
    assert gs.check_kernel_v22()[0] is False


def test_kernel_mutation_winner_flips(monkeypatch):
    d = _need("kernel_decision_v22")
    d["primary_axis_winners"]["2m"]["legacy_vs_stdcurve"]["ffr"] = "umap_stdcurve"
    _patch(monkeypatch, "kernel_decision_v22", d)
    assert gs.check_kernel_v22()[0] is False


def test_bridge_mutation_wrong_pipeline(monkeypatch):
    # the bridge check reopens run results, not this summary, so corrupt the
    # canary source it also requires.
    d = _need("canary_pass")
    if d is None:
        pytest.skip("canary evidence absent")
    d["steady_updates_per_s"] = 10.0                     # below floor
    _patch(monkeypatch, os.path.join("r1_8m", "canary_8m.json"), d)
    assert gs.check_bridge()[0] is False
