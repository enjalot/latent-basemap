"""Round 0021: global exact-family cap-one at 30M."""
from __future__ import annotations

import copy

from .artifact_identity import canonical_json, sha256_bytes
from .round0019_program import (
    NODES,
    Round0019MaterializedArray,
    TRAIN_CONFIG as _ROUND0019_TRAIN_CONFIG,
)


ROUND_ID = "0021"
Round0021MaterializedArray = Round0019MaterializedArray

TRAIN_CONFIG = copy.deepcopy(_ROUND0019_TRAIN_CONFIG)
TRAIN_CONFIG["schema"] = "round0021-production-config-v1"
TRAIN_CONFIG["phrase"] = (
    "30M MiniLM seed42 map with global exact-family cap-one from R0020 census"
)
TRAIN_CONFIG["execution"]["duplicate_multiplicity"] = {
    "policy": "cap-exact-embedding-families-at-one-row-equivalent",
    "scope": "r0020-global-exact-fp16-family-census",
    "artifact_path": (
        "/data/latent-basemap/runs/round-0020/queue/artifacts/"
        "duplicate-census/global-cap-v1.npz"
    ),
    "artifact_sha256": (
        "9511ceca802da603bfbfe9164f8c6ffd7006df82df17b9499d4ed33288fde7cb"
    ),
    "cap_identity_sha256": (
        "8c03230f4b9f6f681c8d759465806c8d65a08125ce1ceecf886c2839f88e63ae"
    ),
    "excluded_rows": 218_242,
    "retained_rows": 29_781_758,
    "represented_exact_families": 138_601,
    "global_family_rows": 356_843,
    "fixed_edges_per_source": 15,
    "effective_positive_edges": 446_726_370,
    "positive_source_sampling": (
        "uniform-retained-row-then-uniform-neighbor-slot-with-replacement"
    ),
    "negative_sampling": "uniform-retained-rows-nonself",
    "positive_destinations": "original-authenticated-graph-rows",
    "baseline_diagnostic_path": (
        "/data/latent-basemap/runs/round-0020/queue/artifacts/"
        "duplicate-census/r0019-global-baseline.json"
    ),
    "baseline_diagnostic_sha256": (
        "0d209b7b18e351559c1c39cc492b1a4b2ada4fe219d0ab07fdadf98977dd1f6e"
    ),
    "baseline_schema": "r0019-global-duplicate-baseline-v1",
    "baseline_method": (
        "fixed-sample-mahalanobis-excluding-union-of-all-census-family-rows"
    ),
    "baseline_top_family_count": 50,
    "baseline_sample_seed": 20260719,
    "baseline_sample_size_requested": 200_000,
    "baseline_sample_size_effective": 197_528,
    "baseline_sample_ids_sha256": (
        "8c39cba2d421f59d5a3453f14dd994d93cd8264b5c91ca3f7f25abbd7bc279fb"
    ),
    "baseline_maximum_mahalanobis": 10.168911145642515,
    "required_maximum_mahalanobis": 5.084455572821257,
    "diagnostic": "global-top50-family-mahalanobis",
}
TRAIN_CONFIG["graph"]["sampling"] = (
    "uniform-over-retained-source-rows-and-fixed-k-slots"
)
TRAIN_CONFIG["graph"]["directed_edges_effective"] = 446_726_370
TRAIN_CONFIG_SHA256 = sha256_bytes(canonical_json(TRAIN_CONFIG))
