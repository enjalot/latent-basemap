"""Train the seed-43 replicate of the retained 25M diverse-Jina atlas."""
from __future__ import annotations

from typing import Any

from basemap.round0107_training import Round0107Error
from experiments.round0107_nodes import run_train_contract


ROUND_ID = "0109"
SEED = 43
TRAIN_CONFIG_SCHEMA = "round0109-diverse-jina-train-config-v1"
PRODUCTION_CONFIG_SCHEMA = "round0109-production-config-v1"
TRAIN_RECEIPT_SCHEMA = "round0109-diverse-jina-train-receipt-v1"


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0107Error("R0109 handler requires its exact round/job")
    if job.get("action") != "train_diverse_jina_seed43":
        raise Round0107Error(f"unknown R0109 action: {job.get('action')!r}")
    return run_train_contract(
        active,
        job,
        round_id=ROUND_ID,
        seed=SEED,
        train_config_schema=TRAIN_CONFIG_SCHEMA,
        production_config_schema=PRODUCTION_CONFIG_SCHEMA,
        train_receipt_schema=TRAIN_RECEIPT_SCHEMA,
        output_label="R0109 seed-43 diverse-Jina train output",
    )
