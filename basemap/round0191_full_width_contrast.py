"""Frozen contract for the R0191 full-rung h4096 width contrast."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0184_prompted_8m_dose_midpoint import (
    ACHIEVED_POSITIVE_DRAWS_PER_EDGE,
    RETAINED_ROWS,
    SUCCESSFUL_UPDATES,
    TARGET_GRAPH_EDGES,
    scale_train_config as h2048_train_config,
)


ROUND_ID = "0191"
SEED = 42
HIDDEN_DIMENSION = 4096
CAPABILITY = "jina-document-english-8m-h4096-width-contrast-v1"
TRAIN_SCHEMA = "round0191-full-h4096-width-train-receipt-v1"
SYNTHESIS_SCHEMA = "round0191-full-h4096-width-decision-v1"
H4096_EVALUATION_SCHEMA = "round0191-full-h4096-common-core-evaluation-v1"
H2048_EVALUATION_SCHEMA = "round0191-r0184-h2048-common-core-evaluation-v1"
MINIMUM_TRAIN_UPDATES_PER_S = 35.0
WARNING_TRAIN_UPDATES_PER_S = 37.0
# R0024 measured 28.433 GiB for the prior h4096 cell. Preserve a real guard
# without placing an already-observed valid execution above the abort line.
HOST_RSS_LIMIT_GIB = 32.0
RETENTION_FLOOR = 0.97


class Round0191Error(RuntimeError):
    """The R0191 width-only treatment or selector changed."""


def h4096_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """Clone R0184 and change only hidden width plus registered rate floors."""
    config, _ = h2048_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
    )
    config = copy.deepcopy(config)
    config["schema"] = "round0191-full-h4096-width-train-config-v1"
    config["model"]["hidden_dimension"] = HIDDEN_DIMENSION
    config["paired_invariant"].update({
        "hidden_dimension": HIDDEN_DIMENSION,
        "only_treatment_relative_to_r0184": "hidden_dimension 2048 -> 4096",
    })
    config["execution"].update({
        "scale_change": (
            "hidden dimension only relative to accepted R0184; full population, "
            "graph, seed, 1M horizon, cosine schedule, sampler, optimizer, "
            "precision, and evaluation core remain frozen"
        ),
        "width_contrast_role": (
            "single branch-authorized capacity sibling; not a width ladder"
        ),
        "minimum_train_upd_s": MINIMUM_TRAIN_UPDATES_PER_S,
        "warning_train_upd_s": WARNING_TRAIN_UPDATES_PER_S,
    })
    config["dose_registration"]["role"] = (
        "width-only contrast at the accepted R0184 1M-update dose and schedule"
    )
    return config, sha256_bytes(canonical_json(config))


def _finite(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise Round0191Error(f"{label} must be finite and positive")
    return number


def width_decision(
    *,
    track_a: Mapping[str, Any],
    h4096_metrics: Mapping[str, float],
    h2048_metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the preregistered recovery and width-null selectors."""
    if (
        track_a.get("outcome") != "confirmed-2-of-3-seed-sensitive"
        or track_a.get("capacity_sibling_activated") is not True
        or int(track_a.get("positive_seed_count", -1)) != 2
    ):
        raise Round0191Error("R0190 did not activate the width sibling")
    metric_set = {
        "mixed_ffr",
        "mixed_purity_fidelity_k256",
        "mixed_purity_fidelity_k1024",
        "pile_ood_recall_at_10",
        "fineweb_ffr",
        "redpajama_ffr",
        "pile_ffr",
    }
    if set(h4096_metrics) != metric_set or set(h2048_metrics) != metric_set:
        raise Round0191Error("width comparison metric set changed")
    treatment = {
        key: _finite(value, label=f"h4096/{key}")
        for key, value in h4096_metrics.items()
    }
    reference = {
        key: _finite(value, label=f"h2048/{key}")
        for key, value in h2048_metrics.items()
    }
    seed42_half = _finite(
        track_a["cells"]["seed42"]["half"]["pile_ffr"],
        label="R0190 seed42 half Pile FFR",
    )
    noise = _finite(
        track_a["width_null_noise_scale"]["value"],
        label="R0190 full-rung Pile FFR sample SD",
    )
    recovery_threshold = RETENTION_FLOOR * seed42_half
    pile_delta = treatment["pile_ffr"] - reference["pile_ffr"]
    recovers = treatment["pile_ffr"] >= recovery_threshold
    within_noise = abs(pile_delta) <= noise
    if recovers and not within_noise and pile_delta > 0:
        outcome = "boundary-recovered-with-width-effect"
    elif recovers:
        outcome = "boundary-recovered-within-seed-noise"
    elif within_noise:
        outcome = "boundary-not-recovered-width-null"
    else:
        outcome = "boundary-not-recovered-with-width-effect"
    return {
        "outcome": outcome,
        "registered_metric": "pile_ffr",
        "h4096": treatment,
        "r0184_h2048": reference,
        "seed42_half_h2048_pile_ffr": seed42_half,
        "recovery_retention_floor": RETENTION_FLOOR,
        "recovery_threshold": recovery_threshold,
        "boundary_recovered": recovers,
        "full_rung_pile_ffr_delta_h4096_minus_h2048": pile_delta,
        "null_noise_scale": noise,
        "null_noise_source": "R0190 three-seed full-rung Pile FFR sample SD",
        "within_seed_noise_of_r0184": within_noise,
        "width_effect_detected": not within_noise,
        "other_metric_deltas_h4096_minus_h2048": {
            key: treatment[key] - reference[key]
            for key in sorted(metric_set - {"pile_ffr"})
        },
    }


__all__ = [
    "ACHIEVED_POSITIVE_DRAWS_PER_EDGE",
    "CAPABILITY",
    "H2048_EVALUATION_SCHEMA",
    "H4096_EVALUATION_SCHEMA",
    "HIDDEN_DIMENSION",
    "HOST_RSS_LIMIT_GIB",
    "MINIMUM_TRAIN_UPDATES_PER_S",
    "RETAINED_ROWS",
    "ROUND_ID",
    "SEED",
    "SUCCESSFUL_UPDATES",
    "SYNTHESIS_SCHEMA",
    "TARGET_GRAPH_EDGES",
    "TRAIN_SCHEMA",
    "Round0191Error",
    "h4096_train_config",
    "width_decision",
]
