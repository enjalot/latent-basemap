from __future__ import annotations

import json

import pytest

from basemap.round0204_v0_release_bundle import (
    CANDIDATE_ID,
    PROMPTED_MAPS,
    ROUND_ID,
    Round0204Error,
    detailed_ood_rows,
    render_model_card,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0204_nodes as nodes


PROPOSAL = (
    "/data/latent-basemap/runs/round-0195/queue/artifacts/"
    "jina-fineweb-2m-v0-release-proposal-v1/release-proposal.json"
)
PACKET = (
    "/data/latent-basemap/runs/round-0182/queue/artifacts/"
    "jina-prompted-raw-universality-readout-v1/packet.json"
)


def _read(path: str) -> dict:
    return prompt_contract.read_sealed(path, label=path)


def test_ood_failures_are_named_and_counted_exactly() -> None:
    rows = detailed_ood_rows(_read(PACKET))
    assert [
        row["probe"]
        for row in rows[PROMPTED_MAPS[0]]
        if row["verdict"] == "named-failure"
    ] == ["code", "culture", "danish", "government", "latin", "science", "trec-covid"]
    assert [
        row["probe"]
        for row in rows[PROMPTED_MAPS[1]]
        if row["verdict"] == "named-failure"
    ] == ["code", "culture", "danish", "government", "latin", "trec-covid"]


def test_model_card_states_limits_and_no_method_winner() -> None:
    proposal = _read(PROPOSAL)
    card = render_model_card(
        proposal=proposal, ood_rows=detailed_ood_rows(_read(PACKET))
    )
    assert CANDIDATE_ID in card
    assert "7 of 11 named failures" in card
    assert "6 of 11 named failures" in card
    assert "does not establish a method winner" in card
    assert "Hugging Face upload has occurred" in card
    assert "ab9766d9" not in card  # exact identity belongs in the sealed JSON bundle


def test_changed_probe_order_fails_closed() -> None:
    packet = _read(PACKET)
    packet = {**packet, "probe_order": list(reversed(packet["probe_order"]))}
    with pytest.raises(Round0204Error, match="probe order"):
        detailed_ood_rows(packet)


def test_real_bundle_assembly_preserves_canonical_artifacts(tmp_path) -> None:
    output = tmp_path / "bundle"
    nodes.run_job(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
        {
            "action": "assemble_v0_release_bundle",
            "r0195_proposal": PROPOSAL,
            "r0182_packet": PACKET,
            "outputs": [str(output)],
        },
    )
    with (output / "release-bundle.json").open(encoding="utf-8") as handle:
        bundle = json.load(handle)
    prompt_contract.validate_seal(bundle, label="R0204 test bundle")
    assert bundle["candidate_id"] == CANDIDATE_ID
    assert bundle["canonical_artifact"]["coordinates"]["sha256"] == (
        "ab9766d9d147d51e9e20ff76170a6f1c815ca99642191e9708098f0370fe0f8a"
    )
    assert bundle["canonical_artifact"]["train_receipt"]["sha256"] == (
        "45965c184f5610c7a009169a7f9eb5fe202b6e9aa842924b2e5d98fd633f5d51"
    )
    assert bundle["ood_limitations"]["seed42_named_failure_count"] == 7
    assert bundle["method_context"]["method_winner_claim"] is False
    assert bundle["release_actions"]["huggingface_upload_authorized"] is False
    assert (output / "README.md").is_file()
