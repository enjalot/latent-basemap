"""Execute the R0213 dose x N x width publication synthesis (CPU only)."""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0213_scaling_story_synthesis import (
    CAPABILITY,
    OPERATING_RULE,
    ROUND_ID,
    Round0213Error,
    SYNTHESIS_SCHEMA,
    dose_axis,
    loss_locality,
    operating_rule,
    width_axis,
)
from basemap import round0113_prompt_contrast as prompt_contract


def _bind(value: Any, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = dict(
        expected_input_signature(prompt_contract.verify_signature(value, label=label))
    )
    return signature, prompt_contract.read_sealed(
        signature["canonical_path"], label=label
    )


def run_synthesise(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0213Error("R0213 synthesis handler received another queue")
    started = time.monotonic()
    boundary_signature, boundary = _bind(
        job["r0190_synthesis"], label="accepted R0190 three-seed synthesis"
    )
    factorial_signature, factorial = _bind(
        job["r0207_factorial"], label="accepted R0207 width factorial"
    )
    if (
        boundary.get("round_id") != "0190"
        or (boundary.get("decision") or {}).get("registered_metric") != "pile_ffr"
        or factorial.get("round_id") != "0207"
        or "fixed dose" not in str(factorial.get("claim_scope") or "")
    ):
        raise Round0213Error("R0213 accepted source evidence changed")
    decision = boundary["decision"]
    noise = float(decision["width_null_noise_scale"]["value"])

    dose = dose_axis(
        high_dose_retention=decision["retention_summary"]["values"],
        high_dose_positive_by_seed=decision["positive_by_seed"],
        low_dose_full_over_half=(
            factorial["retentions"]["h2048"]["pile_ffr"]["full_over_half"]
        ),
        seed_noise_sd=noise,
    )
    width = width_axis(
        contrasts=factorial["width_contrasts"],
        seed_noise_sd=noise,
        low_dose_widths_flat=(
            factorial.get("outcome") == "both-widths-flat-at-low-dose"
        ),
    )
    locality = loss_locality(context=factorial["capacity_context"])
    rule = operating_rule(dose=dose, width=width)

    # Fail closed if any width cell is not at the low dose: the whole point of
    # this artifact is that the width axis was never measured at the high dose.
    for label, rungs in (factorial.get("cells") or {}).items():
        for rung, cell in rungs.items():
            if abs(float(cell["positive_draws_per_edge"]) - width[
                "dose_of_every_width_cell"
            ]) > 1.0e-6:
                raise Round0213Error(
                    f"R0213 found width cell {label}/{rung} off the low dose; the "
                    "missing-cell claim would be wrong"
                )

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0213 scaling-story synthesis"
    )
    synthesis = prompt_contract.seal({
        "schema": SYNTHESIS_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "capabilities": [CAPABILITY],
        "headline": (
            "the composition-controlled size regression is a dose effect at "
            "h2048, not a size law; it is seed-sensitive at the high dose and "
            "absent at the low dose, and the width axis was only ever measured "
            "at the low dose"
        ),
        "dose_axis": dose,
        "width_axis": width,
        "loss_locality": locality,
        "operating_rule": rule,
        "operating_rule_text": OPERATING_RULE,
        "sources": {
            "r0190_three_seed_boundary": boundary_signature,
            "r0207_width_factorial": factorial_signature,
            "embedded": [
                dict(value)
                for value in (job.get("embedded_sources") or [])
            ],
        },
        "campaign_brief_correction": {
            "brief_says": (
                "the width recovery at high dose; h4096 absorbs the high dose, "
                "at 3.118x cost"
            ),
            "receipts_say": (
                "every width cell reports the low dose 0.6781781544098838; the "
                "1.3743 dose exists only at h2048 in R0187/R0188/R0189, so the "
                "3.118x cost line is a low-dose measurement and no h4096 "
                "high-dose cell exists"
            ),
            "resolution": "the missing cell is named as a gap, not claimed",
        },
        "training_performed": False,
        "production_or_publishing": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        f"{output}/scaling-story.json", synthesis, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "synthesise_scaling_story":
        raise Round0213Error("R0213 authorizes only the scaling-story synthesis")
    run_synthesise(active, job)


__all__ = ["run_job", "run_synthesise"]
