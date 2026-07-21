"""Round 0023: R0019 local-cap siblings for seeds 43 and 44."""
from __future__ import annotations

import copy
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0019_program import (
    NODES,
    Round0019MaterializedArray,
    TRAIN_CONFIG as _ROUND0019_TRAIN_CONFIG,
)


ROUND_ID = "0023"
Round0023MaterializedArray = Round0019MaterializedArray
SEEDS = (43, 44)


R0019_REFERENCE = {
    "round": "0019",
    "release_base_commit": "0deb7aaeb2e92e93534ba1ee94e0b1aa8134b476",
    "local_duplicate_cap_sha256": (
        "cb5617f7ef672801c59a6ecbe87af4c7c65390ec59b1305fbff77ec673aad007"
    ),
    "seed42_model_sha256": (
        "2f5eb27582e26735491b4bed9417cf27992bb213ef942e433a5bcba97d481a32"
    ),
    "seed42_coordinate_receipt_sha256": (
        "8d6d5ab2b16be0e08b636a248f667d6a963217d1ec3a223af0b0730875d491d9"
    ),
    "seed42_panel_sha256": (
        "2abfb6a5fe0ab3d4fbea67709d595cfe7c5d2b437468b2f19a2c6a0373334649"
    ),
    "semantic_sample_ids_file_sha256": (
        "efd5884d338843cd27f0b9dcf12b7d31640a8e013860e05fb2578b9333f3393f"
    ),
    "high_d_reference_receipt_sha256": (
        "af9756ad0e154a3586ee347e9eecec6bafd4d349f9ae6fa0547bf3ab5c65de81"
    ),
    "high_d_reference_npz_sha256": (
        "e477ad605afd5eda142f049d34f15874325806acf58618aa1135935de9df4560"
    ),
    "recall50_truth_sha256": (
        "46fa8364472ee5efe280bbd49146bd22c6a9b17eeeef1fd6296b3d476d73c19b"
    ),
}


def train_config_for_seed(seed: int) -> tuple[dict[str, Any], str]:
    if int(seed) not in SEEDS:
        raise ValueError(f"unknown Round 0023 seed: {seed!r}")
    config = copy.deepcopy(_ROUND0019_TRAIN_CONFIG)
    config["schema"] = f"round0023-seed{int(seed)}-production-config-v1"
    config["phrase"] = (
        f"30M MiniLM R0019 local-cap sibling seed {int(seed)} on one GSV RTX 5090"
    )
    config["optimizer"]["seed"] = int(seed)
    config["execution"]["round0023_seed_replicate"] = {
        "seed": int(seed),
        "same_except": ["schema", "phrase", "optimizer.seed"],
        "reference": dict(R0019_REFERENCE),
    }
    return config, sha256_bytes(canonical_json(config))


def seed_from_job(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job.get("seed") or job.get("training_seed") or 43)
    config, digest = train_config_for_seed(seed)
    return {
        "seed": seed,
        "train_config": config,
        "train_config_sha256": digest,
    }
