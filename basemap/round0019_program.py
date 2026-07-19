"""Round 0019: cap the dominant exact-duplicate island at one row-equivalent."""
from __future__ import annotations

import copy

from .artifact_identity import canonical_json, sha256_bytes
from .round0018_program import (
    NODES,
    Round0018MaterializedArray,
    TRAIN_CONFIG as _ROUND0018_TRAIN_CONFIG,
)


ROUND_ID = "0019"
Round0019MaterializedArray = Round0018MaterializedArray

TRAIN_CONFIG = copy.deepcopy(_ROUND0018_TRAIN_CONFIG)
TRAIN_CONFIG["schema"] = "round0019-production-config-v1"
TRAIN_CONFIG["execution"]["duplicate_multiplicity"] = {
    "policy": "cap-exact-embedding-families-at-one-row-equivalent",
    "scope": "r0018-coordinate-x-below-minus-900-disconnected-component",
    "artifact_path": (
        "/data/latent-basemap/runs/round-0019/input/duplicate-cap.npz"
    ),
    "artifact_sha256": (
        "cb5617f7ef672801c59a6ecbe87af4c7c65390ec59b1305fbff77ec673aad007"
    ),
    "excluded_rows": 10_162,
    "retained_rows": 29_989_838,
    "fixed_edges_per_source": 15,
    "effective_positive_edges": 449_847_570,
    "positive_source_sampling": (
        "uniform-retained-row-then-uniform-neighbor-slot-with-replacement"
    ),
    "negative_sampling": "uniform-retained-rows-nonself",
    "positive_destinations": "original-authenticated-graph-rows",
    "baseline_diagnostic_path": (
        "/data/latent-basemap/runs/round-0019/input/r0018-duplicate-baseline.json"
    ),
    "baseline_diagnostic_sha256": (
        "2bf66abd142065663c3631fcaeb94fdf89b35ad6feba16874cb1be169a686c5a"
    ),
    "baseline_maximum_mahalanobis": 10.837448594760865,
    "required_maximum_mahalanobis": 5.4187242973804325,
}
TRAIN_CONFIG["graph"]["sampling"] = (
    "uniform-over-retained-source-rows-and-fixed-k-slots"
)
TRAIN_CONFIG["graph"]["directed_edges_effective"] = 449_847_570
TRAIN_CONFIG_SHA256 = sha256_bytes(canonical_json(TRAIN_CONFIG))
