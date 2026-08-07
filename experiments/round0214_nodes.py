"""Execute the R0214 seed-43 panel readout.

Reuses R0211's evaluation wholesale — same native panel, same matched-2M panel,
same pack-v2 retained ordinals, same probe scorer — and rebinds exactly three
things: the round identity, the sealed cell being scored (R0212's seed-43 model
instead of R0210's seed-42 one), and the decision, which is descriptive here
because no replacement gate has been registered.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.round0209_prompted_diverse_graph import GRAPH_SCHEMA
from basemap.round0210_prompted_diverse_low_dose import successful_updates_for_edges
from basemap.round0212_prompted_diverse_seed43 import (
    CAPABILITY as MODEL_CAPABILITY,
    SEED,
    TRAIN_SCHEMA,
    seed43_train_config,
)
from basemap.round0214_seed43_panel import (
    CAPABILITY,
    EVALUATION_SCHEMA,
    PAIRED_EVALUATION_SCHEMA,
    PAIRED_REFERENCE,
    ROUND_ID,
    Round0214Error,
    descriptive_panel_decision,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0169_nodes as diverse
from experiments import round0211_nodes as panel


def _configure(updates: int) -> None:
    """Bind the accepted kernels to the sealed seed-43 cell."""
    diverse._configure_q2_kernel()
    for name, value in {
        "ROUND_ID": "0212",
        "CAPABILITY": MODEL_CAPABILITY,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "GRAPH_SOURCE_ROUND_ID": "0209",
        "GRAPH_BUILT_IN_ROUND": False,
        "TRAIN_SCHEMA": TRAIN_SCHEMA,
        "SUCCESSFUL_UPDATES": updates,
        "SEED": SEED,
        "scale_train_config": seed43_train_config,
    }.items():
        setattr(q2, name, value)
    if int(q2.SEED) != SEED:
        raise Round0214Error("R0214 seed did not reach the panel kernel")


def _bind_paired_cell(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read the accepted seed-42 readout so the spread can be reported."""
    signature = dict(
        expected_input_signature(
            prompt_contract.verify_signature(
                job["r0211_evaluation"], label="accepted R0211 seed-42 panel"
            )
        )
    )
    evaluation = prompt_contract.read_sealed(
        signature["canonical_path"], label="accepted R0211 seed-42 panel"
    )
    if (
        evaluation.get("schema") != PAIRED_EVALUATION_SCHEMA
        or evaluation.get("round_id") != "0211"
    ):
        raise Round0214Error("R0214 paired seed-42 readout changed")
    metrics = (evaluation.get("native_u12") or {}).get("decision_metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise Round0214Error("R0214 paired seed-42 native metrics are missing")
    PAIRED_REFERENCE.clear()
    PAIRED_REFERENCE.update({
        "signature": signature,
        "native_decision_metrics": {k: float(v) for k, v in metrics.items()},
    })
    return signature


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "evaluate_prompted_diverse_u12_seed43":
        raise Round0214Error("R0214 authorizes only the seed-43 panel readout")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0214Error("R0214 panel handler received another queue")
    _bind_paired_cell(job)
    # Rebind R0211's panel to this round's identity, cell, and decision.
    panel.ROUND_ID = ROUND_ID
    panel.EVALUATION_SCHEMA = EVALUATION_SCHEMA
    panel.CAPABILITY = CAPABILITY
    panel.MODEL_CAPABILITY = MODEL_CAPABILITY
    panel.TRAIN_SCHEMA = TRAIN_SCHEMA
    panel.diverse_panel_decision = descriptive_panel_decision
    panel.successful_updates_for_edges = successful_updates_for_edges
    panel._configure = _configure
    panel.run_evaluate(active, {**job, "action": "evaluate_prompted_diverse_u12_low_dose"})


__all__ = ["run_job"]
