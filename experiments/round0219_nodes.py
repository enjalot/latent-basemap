"""Execute R0219's CPU-only MiniLM mixed-2M quality-gate registration.

Zero GPU. The node reads R0218's sealed four-cell panel receipt, re-derives the
mean - 2 sigma family with the R0161/R0193 estimator over **FFR and the two
purity fidelities only**, transcribes `density_v2` as a diagnostic, carries the
per-corpus FFR spread descriptively, and seals the result. No metric is
recomputed and no map is touched.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0218_minilm_2m_panel import (
    CAPABILITY as PANEL_CAPABILITY,
    EVALUATION_SCHEMA as PANEL_SCHEMA,
)
from basemap.round0219_minilm_2m_gate_registration import (
    CAPABILITY,
    EXCLUDED_METRICS,
    GATE_METRICS,
    ROUND_ID,
    Round0219Error,
    register_minilm_gates,
)
from basemap import round0113_prompt_contrast as prompt_contract


ACTION = "register_minilm_mixed_2m_quality_gates"


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if (
        str(job.get("action") or "") != ACTION
        or active.get("manifest", {}).get("round_id") != ROUND_ID
    ):
        raise Round0219Error("unknown R0219 action or queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0219Error("R0219 is CPU-only")
    panel_path = str(job["panel_evidence"])
    panel_signature = expected_input_signature(panel_path)
    panel = prompt_contract.read_sealed(
        panel_path, label="R0218 MiniLM 2M four-seed panel"
    )
    if (
        panel.get("schema") != PANEL_SCHEMA
        or panel.get("round_id") != "0218"
        or panel.get("capabilities") != [PANEL_CAPABILITY]
        or panel.get("evaluation_performed") is not True
        or panel.get("map_quality_claim_available") is not False
    ):
        raise Round0219Error("R0218 panel receipt contract changed")
    registration = register_minilm_gates(panel)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0219 MiniLM gate registration"
    )
    receipt = prompt_contract.seal({
        **registration,
        "capabilities": [CAPABILITY],
        "release_sha": active["manifest"]["release_sha"],
        "panel_evidence": panel_signature,
        "panel_capability": PANEL_CAPABILITY,
        "panel_release_sha": panel.get("release_sha"),
        "source_panel_seed_invariant_sha256": panel.get("seed_invariant_sha256"),
        "upstream_review_state": dict(job["upstream_review_state"]),
        "decision": {
            "outcome": "minilm-mixed-2m-quality-gates-registered",
            "gated_metrics": list(GATE_METRICS),
            "excluded_metrics": sorted(EXCLUDED_METRICS),
            "applies_to": (
                "future byte-commensurate maps of the R0216 queue-correction-3 "
                "mixed MiniLM 2M universe under the registered R0217 recipe"
            ),
            "does_not_apply_to": (
                "jina universes, differently composed or differently sized "
                "MiniLM universes, PQ-derived graphs, or any map scored on a "
                "different panel configuration"
            ),
            "gpu_used": False,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "minilm-quality-gates.json"), receipt, immutable=True
    )


__all__ = ["ACTION", "run_job"]
