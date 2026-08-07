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


def test_probe_receipt_guard_names_the_embedding_round_not_the_executing_one() -> None:
    """A later round must be able to load the R0173 probe receipts.

    `_load_language_probe` guards the receipt's `round_id`. That receipt names the
    round that embedded the probe (0173), not the round executing the check, so
    comparing it against `ROUND_ID` made the guard pass only inside R0173 itself.
    """
    from experiments import round0169_nodes as diverse

    probe_dir = os.path.join(
        "/data/latent-basemap/runs/round-0173/queue/artifacts", "prompted-arb_Arab"
    )
    if not os.path.exists(os.path.join(probe_dir, "receipt.json")):
        pytest.skip("R0173 probe receipts are not present on this machine")
    assert diverse.LANGUAGE_RECEIPT_ROUND_ID == "0173"
    saved = diverse.ROUND_ID
    try:
        # Any executing round other than 0173 must still load the pack.
        diverse.ROUND_ID = "0211"
        corpus, queries, corpus_rows, query_rows, _sigs = diverse._load_language_probe(
            probe_dir, "arb_Arab"
        )
        assert corpus.shape == (49_500, 768)
        assert queries.shape == (500, 768)
        assert corpus_rows.shape == (49_500,) and query_rows.shape == (500,)
    finally:
        diverse.ROUND_ID = saved


def test_probe_receipt_guard_still_rejects_a_foreign_pack_round() -> None:
    from experiments import round0169_nodes as diverse
    from basemap.round0169_prompted_diverse import Round0169Error

    probe_dir = os.path.join(
        "/data/latent-basemap/runs/round-0173/queue/artifacts", "prompted-arb_Arab"
    )
    if not os.path.exists(os.path.join(probe_dir, "receipt.json")):
        pytest.skip("R0173 probe receipts are not present on this machine")
    saved = diverse.LANGUAGE_RECEIPT_ROUND_ID
    try:
        diverse.LANGUAGE_RECEIPT_ROUND_ID = "9999"
        with pytest.raises(Round0169Error):
            diverse._load_language_probe(probe_dir, "arb_Arab")
    finally:
        diverse.LANGUAGE_RECEIPT_ROUND_ID = saved


def test_matched_2m_reference_convention_matches_the_accepted_reference() -> None:
    """The matched-2M panel's declared convention must reproduce the reference key.

    `_matched_2m_panel` declares the reference identity independently and
    `_resolve_reference` recomputes the key from it, so any drift between the
    declared convention and the accepted R0160 reference's stored convention
    fails closed at score time — after the model transform has already been
    paid for. Catch it on CPU instead.
    """
    import dataclasses
    import json

    from basemap import round0113_prompt_contrast as pc
    from basemap.artifact_identity import expected_input_signature
    from basemap.panel_v2 import load_hiD_reference
    from experiments import round0166_nodes as q2
    from experiments.prepare_round0169_queue import FAMILY_PATH, _read_sealed

    if not os.path.exists(FAMILY_PATH):
        pytest.skip("accepted R0160 family is not present on this machine")
    family = _read_sealed(expected_input_signature(FAMILY_PATH), label="family")
    reference = load_hiD_reference(
        pc.verify_signature(family["shared_prompted_reference"], label="reference")
    )
    stored = reference["key_parts"]
    declared_row_order = (
        "R0113 shared source/raw/document union-representative compact order"
    )
    assert stored["convention"]["row_order"] == declared_row_order
    assert stored["convention"]["anchor_namespace"] == "R0113 compact IDs"
    assert stored["convention"]["embedding_prompt"] == "document"
    assert (
        stored["convention"]["distance"]
        == "cosine via fp32-L2-normalized squared L2"
    )
    assert stored["convention"]["self_exclusion"] is True
    # The panel config and the reference's bound data identity must also agree.
    cfg = pc.panel_config()
    assert stored["formula"] == cfg.formula_version
    assert json.dumps(stored["config"], sort_keys=True) == json.dumps(
        dataclasses.asdict(cfg), sort_keys=True
    )
    assembly = q2._read_sealed(family["lineage"]["assembly"], label="assembly")
    assert json.dumps(
        q2.prompt_nodes._data_identity(assembly, arm="document"), sort_keys=True
    ) == json.dumps(stored["data"], sort_keys=True)


def test_accepted_matched_query_truth_assertions_hold() -> None:
    """Pre-verify the assertions `_matched_2m_panel` makes after the transform."""
    from basemap import round0113_prompt_contrast as pc
    from basemap.artifact_identity import expected_input_signature
    from basemap.panel_v2 import load_query_truth
    from experiments import round0166_nodes as q2
    from experiments.prepare_round0169_queue import FAMILY_PATH, _read_sealed

    if not os.path.exists(FAMILY_PATH):
        pytest.skip("accepted R0160 family is not present on this machine")
    family = _read_sealed(expected_input_signature(FAMILY_PATH), label="family")
    seed42 = family["cells"]["seed42"]
    score = q2._read_sealed(seed42["native_score"], label="score")
    assert score["round_id"] == "0115" and score["arm"] == "document"
    assert score["projections"]["matched"]["truth_row_range"] == [0, q2.QUERY_ROWS]
    truth_path = pc.verify_signature(score["combined_query_truth"], label="truth")
    try:
        truth = load_query_truth(truth_path)
    except ValueError as error:
        # `load_query_truth` binds the exact CUDA backend identity that produced
        # the truth, which cannot reproduce in this CUDA-hidden test process.
        # The production node runs with CUDA present.
        if "backend identity" not in str(error):
            raise
        pytest.skip(f"query truth needs its CUDA backend: {error}")
    assert truth["corpus_cardinality"] == q2.MATCHED_ROWS
