"""Round 0024: h1024/h4096 capacity ladder siblings of the R0019 map."""
from __future__ import annotations

import copy
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0019_program import (
    NODES,
    Round0019MaterializedArray,
    TRAIN_CONFIG as _ROUND0019_TRAIN_CONFIG,
)


ROUND_ID = "0024"
Round0024MaterializedArray = Round0019MaterializedArray


CELL_SPECS: dict[str, dict[str, Any]] = {
    "h1024": {
        "label": "h1024",
        "hidden_dimension": 1024,
        "minimum_train_upd_s": 45.0,
        "warn_train_upd_s": 55.0,
        "canary_minimum_upd_s": 100.0,
    },
    "h4096": {
        "label": "h4096",
        "hidden_dimension": 4096,
        "minimum_train_upd_s": 25.0,
        "warn_train_upd_s": 35.0,
        "canary_minimum_upd_s": 40.0,
    },
}


def train_config_for_cell(label: str) -> tuple[dict[str, Any], str]:
    if label not in CELL_SPECS:
        raise ValueError(f"unknown Round 0024 cell: {label!r}")
    spec = CELL_SPECS[label]
    config = copy.deepcopy(_ROUND0019_TRAIN_CONFIG)
    config["schema"] = f"round0024-{label}-production-config-v1"
    config["phrase"] = (
        f"30M MiniLM capacity ladder {label} sibling of R0019 on one GSV RTX 5090"
    )
    config["model"]["hidden_dimension"] = int(spec["hidden_dimension"])
    config["execution"]["round0024_capacity_ladder_cell"] = dict(spec)
    config["execution"]["capacity_ladder_reference"] = {
        "round": "0019",
        "hidden_dimension": 2048,
        "same_except": ["model.hidden_dimension", "schema", "phrase"],
        "panel_sha256": "2abfb6a5fe0ab3d4fbea67709d595cfe7c5d2b437468b2f19a2c6a0373334649",
    }
    return config, sha256_bytes(canonical_json(config))


def cell_from_job(job: dict[str, Any]) -> dict[str, Any]:
    label = str(job.get("cell") or job.get("capacity_cell") or "h1024")
    config, digest = train_config_for_cell(label)
    return {
        **CELL_SPECS[label],
        "train_config": config,
        "train_config_sha256": digest,
    }

