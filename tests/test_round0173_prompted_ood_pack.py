"""Contract tests for the prompted U12 OOD probe pack."""
from __future__ import annotations

from basemap.round0173_prompted_ood_pack import (
    CANARY_SCHEMA,
    CAPABILITY,
    LANGUAGE_PROBE_SCHEMA,
    OOD_AUDIT_SCHEMA,
    ROUND_ID,
)
from experiments import round0169_nodes, round0173_nodes


def test_r0173_dispatch_binds_pack_schemas(monkeypatch) -> None:
    names = (
        "ROUND_ID",
        "Round0169Error",
        "CANARY_SCHEMA",
        "LANGUAGE_PROBE_SCHEMA",
        "OOD_AUDIT_SCHEMA",
        "OOD_PACK_CAPABILITY",
    )
    before = {name: getattr(round0169_nodes, name) for name in names}
    observed = {}
    monkeypatch.setattr(
        round0169_nodes,
        "run_job",
        lambda active, job: observed.update({
            "round_id": round0169_nodes.ROUND_ID,
            "canary": round0169_nodes.CANARY_SCHEMA,
            "probe": round0169_nodes.LANGUAGE_PROBE_SCHEMA,
            "audit": round0169_nodes.OOD_AUDIT_SCHEMA,
            "capability": round0169_nodes.OOD_PACK_CAPABILITY,
        }),
    )
    try:
        round0173_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}}, {"action": "prompt_canary"}
        )
        assert observed == {
            "round_id": "0173",
            "canary": CANARY_SCHEMA,
            "probe": LANGUAGE_PROBE_SCHEMA,
            "audit": OOD_AUDIT_SCHEMA,
            "capability": CAPABILITY,
        }
    finally:
        for name, value in before.items():
            setattr(round0169_nodes, name, value)
