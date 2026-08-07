from __future__ import annotations

import os

import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0105_search import GROUPS
from basemap.round0108_evaluation import IN_MIX_LANGUAGES
from basemap.round0166_prompted_8m import METRICS, NATIVE_ABSOLUTE_METRICS
from basemap.round0169_prompted_diverse import prompted_diverse_decision
from basemap.round0211_prompted_diverse_panel import (
    CAPABILITY,
    PACK_CORPUS_ROWS,
    PACK_QUERY_ROWS,
    PACK_ROWS,
    ROUND_ID,
    Round0211Error,
    diverse_panel_decision,
)
from experiments import round0211_nodes as nodes


HEALTHY = {
    "density_v2": 0.90,
    "ffr": 0.60,
    "purity_fidelity_k256": 0.80,
    "purity_fidelity_k1024": 0.80,
    "projection_ffr": 0.50,
    "heldout_recall_at_10": 0.40,
}
FLOORS = {metric: 0.10 for metric in METRICS}
PACK_PATH = (
    "/data/latent-basemap/runs/round-0208/queue/artifacts/"
    "jina-prompted-u12-ood-probe-pack-v2/probe-pack.json"
)


def _live_pack_signature() -> dict:
    if not os.path.exists(PACK_PATH):
        pytest.skip("R0208 pack v2 is not present on this machine")
    return expected_input_signature(PACK_PATH)


def _inputs(**overrides):
    payload = {
        "native": dict(HEALTHY),
        "matched_2m": dict(HEALTHY),
        "baseline_2m_seed42": dict(HEALTHY),
        "prompted_floors": dict(FLOORS),
        "group_ffr": {name: 0.60 for name in GROUPS},
        "prompted_ood": {
            "polish_recall_at_50_of_high10": 0.50,
            "in_mix_median_recall_at_50_of_high10": 0.60,
        },
        "raw_r0132_ood": {
            "polish_recall_at_50_of_high10": 0.50,
            "in_mix_median_recall_at_50_of_high10": 0.60,
        },
    }
    payload.update(overrides)
    return payload


def test_pack_shape_constants() -> None:
    assert PACK_ROWS == 20 * (PACK_CORPUS_ROWS + PACK_QUERY_ROWS) == 999_880
    assert PACK_CORPUS_ROWS == 49_494 and PACK_QUERY_ROWS == 500


def test_healthy_rung_passes_on_retention() -> None:
    decision = diverse_panel_decision(**_inputs())
    assert decision["passed"] is True
    assert decision["primary_registered_readout"] == "scale-relative retention"
    assert decision["atlas_quality_claim_available"] is False
    assert decision["production_claim_available"] is False


def test_native_absolute_cells_are_descriptive_not_decisive() -> None:
    """A rung below every FineWeb-English floor still passes on retention."""
    high_floors = {metric: 0.99 for metric in METRICS}
    decision = diverse_panel_decision(**_inputs(prompted_floors=high_floors))
    cells = decision["native_absolute_cells"]["cells"]
    assert set(cells) == set(NATIVE_ABSOLUTE_METRICS)
    assert all(cell["role"] == "descriptive" for cell in cells.values())
    assert all(cell["would_have_passed_under_r0169"] is False for cell in cells.values())
    assert all("passed" not in cell for cell in cells.values())
    assert decision["passed"] is True
    # ...and R0169's own structure would have failed it, which is the point.
    assert decision["r0169_style_verdict_for_reference"]["passed"] is False
    assert (
        prompted_diverse_decision(**_inputs(prompted_floors=high_floors))["passed"]
        is False
    )


def test_retention_miss_fails() -> None:
    weak = {**HEALTHY, "ffr": 0.30}
    decision = diverse_panel_decision(**_inputs(matched_2m=weak))
    assert decision["passed"] is False
    assert decision["scale_relative_retention_gates"]["ffr"]["passed"] is False
    assert decision["outcome"].endswith("retention-not-qualified")


def test_per_language_ffr_collapse_fails() -> None:
    groups = {name: 0.60 for name in GROUPS}
    groups[IN_MIX_LANGUAGES[0]] = 0.001
    decision = diverse_panel_decision(**_inputs(group_ffr=groups))
    assert decision["passed"] is False
    assert (
        decision["language_relative_ffr"]["cells"][IN_MIX_LANGUAGES[0]]["passed"]
        is False
    )


def test_polish_ood_collapse_fails() -> None:
    decision = diverse_panel_decision(
        **_inputs(
            prompted_ood={
                "polish_recall_at_50_of_high10": 0.05,
                "in_mix_median_recall_at_50_of_high10": 0.60,
            }
        )
    )
    assert decision["passed"] is False
    assert decision["polish_ood_gate"]["passed"] is False


def test_ood_retention_miss_fails() -> None:
    decision = diverse_panel_decision(
        **_inputs(
            prompted_ood={
                "polish_recall_at_50_of_high10": 0.30,
                "in_mix_median_recall_at_50_of_high10": 0.35,
            }
        )
    )
    assert decision["passed"] is False
    assert any(
        cell["passed"] is False
        for cell in decision["raw_r0132_ood_retention_gates"].values()
    )


def test_density_v2_is_named_diagnostic() -> None:
    decision = diverse_panel_decision(**_inputs())
    assert any("density-v2" in item for item in decision["diagnostic_only"])
    assert any("projection" in item.lower() for item in decision["diagnostic_only"])


def test_node_rejects_another_action_or_queue() -> None:
    with pytest.raises(Round0211Error):
        nodes.run_job({"manifest": {"round_id": ROUND_ID}}, {"action": "train"})
    with pytest.raises(Round0211Error):
        nodes.run_evaluate(
            {"manifest": {"round_id": "0169"}},
            {"action": "evaluate_prompted_diverse_u12_low_dose"},
        )


def test_sealed_pack_binds_the_live_v2_reserve() -> None:
    pack, signature = nodes._sealed_pack({"ood_pack": _live_pack_signature()})
    assert pack["capability"] == "jina-prompted-u12-ood-probe-pack-v2"
    assert pack["shape"]["pack_rows"] == PACK_ROWS
    assert signature["canonical_path"] == PACK_PATH


def test_sealed_pack_rejects_the_blocked_v1_audit() -> None:
    v1_audit = (
        "/data/latent-basemap/runs/round-0173/queue/artifacts/"
        "jina-prompted-u12-ood-probe-pack-v1/audit.json"
    )
    if not os.path.exists(v1_audit):
        pytest.skip("R0173 v1 audit is not present on this machine")
    with pytest.raises(Round0211Error):
        nodes._sealed_pack({"ood_pack": expected_input_signature(v1_audit)})


def test_retained_probe_slices_to_the_registered_shape() -> None:
    pack, _signature = nodes._sealed_pack({"ood_pack": _live_pack_signature()})
    for language in ("kor_Hang", "arb_Arab", "pol_Latn"):
        corpus, queries, corpus_rows, query_rows, _sigs = nodes._retained_probe(
            pack, language
        )
        assert corpus.shape == (PACK_CORPUS_ROWS, 768)
        assert queries.shape == (PACK_QUERY_ROWS, 768)
        assert corpus_rows.shape == (PACK_CORPUS_ROWS,)
        assert query_rows.shape == (PACK_QUERY_ROWS,)
        assert len(set(corpus_rows.tolist())) == PACK_CORPUS_ROWS


def test_removed_overlap_rows_are_absent_from_the_retained_corpus() -> None:
    """The five R0173 training-family source rows must not survive the slice."""
    pack, _signature = nodes._sealed_pack({"ood_pack": _live_pack_signature()})
    removed = {
        "arb_Arab": {875069, 1505153},
        "cmn_Hani": {849744, 856357},
        "tha_Thai": {1788247},
    }
    for language, rows in removed.items():
        _corpus, _queries, corpus_rows, _query_rows, _sigs = nodes._retained_probe(
            pack, language
        )
        assert not rows & set(corpus_rows.tolist())


def test_capability_and_round_identity() -> None:
    assert CAPABILITY == "jina-prompted-diverse-u12-evaluation-panel-v1"
    assert ROUND_ID == "0211"


def test_bound_upstream_paths_are_the_queues_that_actually_sealed() -> None:
    """Guard the 'bound the failed queue' bug class.

    R0209 and R0210 each reached a registered terminal `failed` state on their
    first queue and sealed their artifact in a dated `queue-correction-N`
    relaunch. A prepare script that keeps pointing at `queue/` binds a path that
    does not exist, and the failure surfaces only at launch.
    """
    from experiments.prepare_round0210_queue import GRAPH_MANIFEST
    from experiments.prepare_round0211_queue import OOD_PACK_PATH, TRAIN_OUTPUT

    for path, label in (
        (GRAPH_MANIFEST, "sealed R0209 graph manifest"),
        (os.path.join(TRAIN_OUTPUT, "train-receipt.json"), "sealed R0210 train receipt"),
        (os.path.join(TRAIN_OUTPUT, "model.pt"), "sealed R0210 model"),
        (OOD_PACK_PATH, "sealed R0208 pack v2"),
    ):
        assert os.path.exists(path), f"{label} is not at the bound path {path}"


def test_train_receipt_matches_the_sealed_graph_horizon() -> None:
    """The panel must not score a model trained off a different edge count."""
    from basemap.round0210_prompted_diverse_low_dose import successful_updates_for_edges
    from basemap import round0113_prompt_contrast as prompt_contract
    from experiments.prepare_round0210_queue import GRAPH_MANIFEST
    from experiments.prepare_round0211_queue import TRAIN_OUTPUT

    receipt_path = os.path.join(TRAIN_OUTPUT, "train-receipt.json")
    if not (os.path.exists(GRAPH_MANIFEST) and os.path.exists(receipt_path)):
        pytest.skip("sealed R0209/R0210 artifacts are not present on this machine")
    graph = prompt_contract.read_sealed(GRAPH_MANIFEST, label="graph")
    train = prompt_contract.read_sealed(receipt_path, label="train")
    edges = int(graph["directed_edge_count"])
    assert int(train["train_accounting"]["n_pos_edges"]) == edges
    assert int(train["optimizer_updates"]) == successful_updates_for_edges(edges)
    assert train["graph_manifest"]["canonical_path"] == GRAPH_MANIFEST
