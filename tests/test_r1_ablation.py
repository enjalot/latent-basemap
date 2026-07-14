"""P0-1 (overnight review): the 2×2 ablation stamps the mid-near dose, requires all
four cells before computing effects, and reports both main effects and the
interaction."""
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.run_r1_ablation import build_cfg, CELLS, compute_effects, MN_ON_SCALE

BASE = os.path.join(os.path.dirname(__file__), '..', 'experiments/configs/jina_en_200k_k50_fuzzy.yaml')


def test_all_four_cells_have_exact_dose_tuples():
    # (midnear_enabled, mn_weight_scale, weighted_edge_sampling) must be exact —
    # the on level is scale 4.0 (A3 mn4), never the default 1.0.
    got = {}
    for mn, ws in CELLS:
        tag, scale, cfg = build_cfg(BASE, mn, ws, 42, 500000)
        got[(mn, ws)] = (cfg.train.midnear_enabled, cfg.train.mn_weight_scale,
                         cfg.train.weighted_edge_sampling)
    assert got[(False, False)] == (False, 1.0, False)
    assert got[(False, True)] == (False, 1.0, True)
    assert got[(True, False)] == (True, MN_ON_SCALE, False)
    assert got[(True, True)] == (True, MN_ON_SCALE, True)
    assert MN_ON_SCALE == 4.0


def test_on_cell_tag_encodes_scale():
    tag, scale, cfg = build_cfg(BASE, True, False, 42, 500000)
    assert "s4" in tag and scale == 4.0, tag
    tag0, _, _ = build_cfg(BASE, False, False, 42, 500000)
    assert "s1" in tag0, tag0


def _mk(mn, ws, ffr, dens, rk=0.1):
    return {"midnear": mn, "weighted": ws, "ffr": ffr, "density": dens, "recall@k": rk}


def test_effects_main_and_interaction():
    # construct a table with a known interaction: mid-near helps density only WITH
    # weighting. c00=.2,c01=.3 (w off/on, mn off); c10=.2,c11=.5 (mn on)
    runs = {"a": _mk(False, False, 0.5, 0.2), "b": _mk(False, True, 0.6, 0.3),
            "c": _mk(True, False, 0.5, 0.2), "d": _mk(True, True, 0.7, 0.5)}
    e = compute_effects(runs)
    # density: weighted_main = ((.3+.5)-(.2+.2))/2 = .2 ; midnear_main = ((.2+.5)-(.2+.3))/2 = .1
    assert e["density"]["weighted_main"] == 0.2
    assert e["density"]["midnear_main"] == 0.1
    # interaction = (c11-c10)-(c01-c00) = (.5-.2)-(.3-.2) = .2
    assert e["density"]["interaction"] == 0.2
    # ffr weighted_main = ((.6+.7)-(.5+.5))/2 = .15 ; midnear_main = ((.5+.7)-(.5+.6))/2 = .05
    assert e["ffr"]["weighted_main"] == 0.15
    assert e["ffr"]["midnear_main"] == 0.05


def test_missing_cell_raises_not_zero():
    runs = {"a": _mk(False, False, 0.5, 0.2), "b": _mk(False, True, 0.6, 0.3),
            "c": _mk(True, False, 0.5, 0.2)}   # missing (True,True)
    with pytest.raises(ValueError, match="missing"):
        compute_effects(runs)


def test_missing_metric_raises():
    runs = {"a": _mk(False, False, 0.5, 0.2), "b": _mk(False, True, 0.6, 0.3),
            "c": _mk(True, False, 0.5, 0.2), "d": {"midnear": True, "weighted": True,
                                                   "ffr": None, "density": 0.5, "recall@k": 0.1}}
    with pytest.raises(ValueError, match="missing metric|missing"):
        compute_effects(runs)


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
