"""Execute the R0204 FineWeb-2M v0 release-bundle synthesis."""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0204_v0_release_bundle import (
    BUNDLE_SCHEMA,
    CANDIDATE_ID,
    CAPABILITY,
    PROPOSAL_SCHEMA,
    ROUND_ID,
    Round0204Error,
    detailed_ood_rows,
    render_model_card,
)
from basemap import round0113_prompt_contrast as prompt_contract


def _signature(path: str, *, label: str) -> dict[str, Any]:
    try:
        return expected_input_signature(path)
    except Exception as error:
        raise Round0204Error(f"{label} is unavailable or changed") from error


def _read(path: str, *, label: str) -> dict[str, Any]:
    try:
        return prompt_contract.read_sealed(path, label=label)
    except Exception as error:
        raise Round0204Error(f"{label} is missing or unsealed") from error


def run_assemble(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0204Error("release-bundle handler received another queue")
    started = time.monotonic()
    proposal_path = str(job["r0195_proposal"])
    proposal_signature = _signature(proposal_path, label="accepted R0195 proposal")
    proposal = _read(proposal_path, label="accepted R0195 proposal")
    packet_path = str(job["r0182_packet"])
    packet_signature = _signature(packet_path, label="accepted R0182 packet")
    packet = _read(packet_path, label="accepted R0182 packet")
    if (
        proposal.get("schema") != PROPOSAL_SCHEMA
        or proposal.get("round_id") != "0195"
        or proposal.get("candidate_id") != CANDIDATE_ID
        or proposal.get("capabilities")
        != ["jina-fineweb-2m-v0-release-proposal-v1"]
        or (proposal.get("qualification") or {}).get(
            "all_four_seeds_pass_all_six_commensurate_gates"
        )
        is not True
        or (proposal.get("candidate_scope") or {}).get("canonical_seed") != 42
        or proposal["candidate_scope"]["canonical_coordinates"]
        != proposal["qualification"]["cells"]["seed42"]["coordinates"]
        or proposal["candidate_scope"]["canonical_train_receipt"]
        != proposal["qualification"]["cells"]["seed42"]["train_receipt"]
        or packet.get("schema") != "round0182-universality-readout-packet-v1"
        or packet.get("round_id") != "0182"
        or packet.get("no_universal_map_claim") is not True
    ):
        raise Round0204Error("accepted release evidence changed")
    for signature in (
        proposal["candidate_scope"]["canonical_coordinates"],
        proposal["candidate_scope"]["canonical_train_receipt"],
    ):
        prompt_contract.verify_signature(signature, label="canonical seed42 artifact")
    ood_rows = detailed_ood_rows(packet)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0204 v0 release bundle"
    )
    card = render_model_card(proposal=proposal, ood_rows=ood_rows)
    card_path = os.path.join(output, "README.md")

    def write_card(temp_path: str) -> None:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(card)

    atomic_build_new_file(card_path, write_card, immutable=True)
    gate_table = {
        name: {
            "seed": int(cell["seed"]),
            "all_six_pass": cell["all_six_pass"],
            "metrics": cell["metrics"],
        }
        for name, cell in proposal["qualification"]["cells"].items()
    }
    bundle = prompt_contract.seal({
        "schema": BUNDLE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "candidate_id": CANDIDATE_ID,
        "capabilities": [CAPABILITY],
        "canonical_artifact": {
            **proposal["candidate_scope"],
            "coordinates": proposal["candidate_scope"]["canonical_coordinates"],
            "train_receipt": proposal["candidate_scope"][
                "canonical_train_receipt"
            ],
        },
        "qualification": {
            "source_gate_family": proposal["sources"]["gates"],
            "all_four_seeds_pass_all_six_commensurate_gates": True,
            "per_seed_gate_table": gate_table,
        },
        "ood_limitations": {
            "source": packet_signature,
            "coverage": "11 named probes; prompted seed42 and seed43 only",
            "maps": ood_rows,
            "seed42_named_failure_count": 7,
            "seed43_named_failure_count": 6,
            "universal_quality_claim": False,
            "must_ship_with_candidate": True,
        },
        "method_context": {
            **proposal["method_context"],
            "role": "historical same-scale context only",
            "method_winner_claim": False,
        },
        "scale_limitations": proposal["scale_limitations"],
        "model_card": _signature(card_path, label="R0204 draft model card"),
        "sources": {
            "accepted_r0195_proposal": proposal_signature,
            "accepted_r0182_packet": packet_signature,
        },
        "release_actions": {
            "local_registry_round_authorized_by_campaign": True,
            "registry_mutation_performed_here": False,
            "huggingface_upload_authorized": False,
            "huggingface_upload_performed": False,
            "production_or_publication_performed": False,
        },
        "training_performed": False,
        "wall_s": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "release-bundle.json"), bundle, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "assemble_v0_release_bundle":
        raise Round0204Error("R0204 authorizes only release-bundle assembly")
    run_assemble(active, job)


__all__ = ["run_job"]
