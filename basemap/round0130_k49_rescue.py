"""Fixed contract and selectors for the 25M k49 graph-degree rescue."""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from .round0107_training import train_config
from .round0108_evaluation import pearson_log_radius


ROUND_ID = "0130"
GRAPH_K = 49
N_NEIGHBORS = GRAPH_K + 1
FIXED_SUCCESSFUL_UPDATES = 1_459_722

GRAPH_PART_SCHEMA = "round0130-jina-diverse-25m-k49-graph-part-v1"
GRAPH_SHARD_SCHEMA = "round0130-jina-diverse-25m-k49-graph-shard-v1"
GRAPH_SCHEMA = "round0130-jina-diverse-25m-k49-fuzzy-graph-v1"
TRAIN_CONFIG_SCHEMA = "round0130-diverse-jina-k49-train-config-v1"
PRODUCTION_CONFIG_SCHEMA = "round0130-production-config-v1"
TRAIN_RECEIPT_SCHEMA = "round0130-diverse-jina-k49-train-receipt-v1"
CORE_SCHEMA = "round0130-diverse-jina-k49-core-geometry-v1"
OOD_SCHEMA = "round0130-diverse-jina-k49-ood-evaluation-v1"
MATCHED_DENSITY_SCHEMA = "round0130-k49-matched-fineweb-density-v1"
DECISION_SCHEMA = "round0130-k49-degree-rescue-decision-v1"
DEGREE_RESCUE_CAPABILITY = (
    "jina-diverse-25m-native-k49-degree-rescue-two-seed-v1"
)
ATLAS_QUALITY_CAPABILITY = (
    "jina-diverse-25m-k49-atlas-quality-two-seed-v1"
)

MAP_KEY = "r0130-diverse-jina-25m-k49-seed42"
MAP_LABEL = MAP_KEY
POSITIVE_DESTINATION_POLICY = (
    "R0130-global-retained-fuzzy-tconorm-k49-graph"
)
GRAPH_DEGREE_STAMP = "variable-symmetric-fuzzy-k49-topology"
UPDATE_RULE = "fixed-R0107-dose-1459722-successful-updates"

DENSITY_MATERIAL_DELTA = 0.03
DENSITY_BOOTSTRAP_DRAWS = 1_000
DENSITY_BOOTSTRAP_SEED = 12_801
HEADLINE_KPI_RETENTION = 0.97
R0107_SEED42_INITIAL_STATE_SHA256 = (
    "dda740f51f2f78436fb2195166943e50898bbd1cd35d41ffc02f3be2a52c46d8"
)


class Round0130Error(RuntimeError):
    """The R0130 single-treatment contract was violated."""


def model_state_sha256(model: Any) -> str:
    """Hash ordered tensor names, dtypes, shapes, and values canonically."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        for payload in (
            key.encode(),
            str(array.dtype).encode(),
            repr(tuple(array.shape)).encode(),
            array.tobytes(order="C"),
        ):
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return digest.hexdigest()


def verify_r0107_seed42_initial_state(model: Any) -> dict[str, Any]:
    observed = model_state_sha256(model)
    if observed != R0107_SEED42_INITIAL_STATE_SHA256:
        raise Round0130Error("seed-42 initialized model state changed")
    return {
        "algorithm": (
            "sha256 over sorted state_dict key/dtype/shape/contiguous-bytes"
        ),
        "observed_sha256": observed,
        "expected_r0107_seed42_sha256": R0107_SEED42_INITIAL_STATE_SHA256,
        "captured_before_optimizer_construction_and_update_zero": True,
        "historical_evidence_kind": (
            "deterministic-reconstruction-not-original-reviewed-receipt"
        ),
        "matches_deterministic_r0107_release_reconstruction": True,
    }


def k49_train_config(
    *,
    graph_manifest: Mapping[str, Any],
    graph_signature: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Build the R0107 config with only graph-dependent fields changed."""
    return train_config(
        graph_manifest=graph_manifest,
        graph_signature=graph_signature,
        seed=42,
        schema=TRAIN_CONFIG_SCHEMA,
        n_neighbors_including_self=N_NEIGHBORS,
        successful_updates=FIXED_SUCCESSFUL_UPDATES,
        update_rule=UPDATE_RULE,
        positive_destination_policy=POSITIVE_DESTINATION_POLICY,
        graph_degree=GRAPH_DEGREE_STAMP,
    )


def _signature_payload(value: Mapping[str, Any]) -> tuple[int, str]:
    try:
        return int(value["bytes"]), str(value["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Round0130Error("artifact signature is malformed") from exc


def assert_treatment_isolation(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the config differs only where a k15->k49 graph requires it.

    Artifact paths and hashes necessarily change for the graph itself. The
    compact mapping must remain byte-identical. The successful-update dose,
    seed, representation, model, optimizer, sampler law, negative law, and
    runtime path remain exact.
    """
    left = copy.deepcopy(dict(control))
    right = copy.deepcopy(dict(treatment))
    if (
        int(left["optimizer"]["successful_positive_lr_updates"])
        != FIXED_SUCCESSFUL_UPDATES
        or int(right["optimizer"]["successful_positive_lr_updates"])
        != FIXED_SUCCESSFUL_UPDATES
    ):
        raise Round0130Error("R0107 and R0130 doses are not exact")
    if _signature_payload(left["input"]["compact_mapping"]) != (
        _signature_payload(right["input"]["compact_mapping"])
    ):
        raise Round0130Error("compact retained population changed")
    if int(left["graph"]["n_neighbors_including_self"]) != 16:
        raise Round0130Error("control is not the reviewed k15 graph")
    if int(right["graph"]["n_neighbors_including_self"]) != N_NEIGHBORS:
        raise Round0130Error("treatment is not the registered k49 graph")

    # Normalize exactly the metadata that must follow new graph bytes. Equality
    # after this normalization is the executable treatment-isolation proof.
    right["schema"] = left["schema"]
    right["input"]["compact_mapping"] = left["input"]["compact_mapping"]
    for key in ("manifest", "outputs", "directed_edges"):
        right["graph"][key] = left["graph"][key]
    right["graph"]["n_neighbors_including_self"] = left["graph"][
        "n_neighbors_including_self"
    ]
    right["optimizer"]["update_rule"] = left["optimizer"]["update_rule"]
    right_stamp = right["execution"]["expected_pipeline_stamp"]
    left_stamp = left["execution"]["expected_pipeline_stamp"]
    for key in (
        "positive_destination_policy",
        "graph_degree",
        "valid_canonical_edge_count",
    ):
        right_stamp[key] = left_stamp[key]
    if right != left:
        raise Round0130Error(
            "R0130 config changes more than graph identity/degree metadata"
        )
    return {
        "population_mapping_byte_identical": True,
        "fixed_successful_updates": FIXED_SUCCESSFUL_UPDATES,
        "control_nonself_degree": 15,
        "treatment_nonself_degree": GRAPH_K,
        "unchanged_fields": [
            "input rows/dimension/representation",
            "seed and model initialization",
            "model architecture",
            "optimizer and cosine schedule",
            "successful-update dose",
            "weighted positive sampler law",
            "uniform nonself negative law",
            "host-int8 device-dequantization runtime",
            "bf16 autocast",
        ],
        "changed_fields": [
            "graph artifact identity",
            "directed fuzzy edge cardinality",
            "nonself graph degree 15->49",
            "graph-dependent execution-stamp labels",
        ],
    }


def paired_density_materiality(
    *,
    control_high_radius: np.ndarray,
    control_low_radius: np.ndarray,
    treatment_high_radius: np.ndarray,
    treatment_low_radius: np.ndarray,
    draws: int = DENSITY_BOOTSTRAP_DRAWS,
    seed: int = DENSITY_BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return the paired k49-minus-k15 native density selector."""
    control_high = np.asarray(control_high_radius, dtype=np.float64)
    treatment_high = np.asarray(treatment_high_radius, dtype=np.float64)
    control_low = np.asarray(control_low_radius, dtype=np.float64)
    treatment_low = np.asarray(treatment_low_radius, dtype=np.float64)
    if (
        control_high.ndim != 1
        or control_low.shape != control_high.shape
        or treatment_high.shape != control_high.shape
        or treatment_low.shape != control_high.shape
        or len(control_high) < 100
        or draws != DENSITY_BOOTSTRAP_DRAWS
        or seed != DENSITY_BOOTSTRAP_SEED
        or not np.array_equal(control_high, treatment_high)
    ):
        raise Round0130Error("paired native density arrays changed")
    control_value = pearson_log_radius(control_high, control_low)
    treatment_value = pearson_log_radius(treatment_high, treatment_low)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        rows = rng.integers(0, len(control_high), len(control_high))
        bootstrap[draw] = (
            pearson_log_radius(treatment_high[rows], treatment_low[rows])
            - pearson_log_radius(control_high[rows], control_low[rows])
        )
    lower, upper = np.quantile(bootstrap, [0.005, 0.995]).tolist()
    delta = treatment_value - control_value
    if lower >= DENSITY_MATERIAL_DELTA:
        outcome = "k49-materially-improves-native-density"
    elif upper < DENSITY_MATERIAL_DELTA:
        outcome = "k49-does-not-materially-improve-native-density"
    else:
        outcome = "k49-native-density-effect-inconclusive"
    return ({
        "control_density_v2": control_value,
        "treatment_density_v2": treatment_value,
        "treatment_minus_control": delta,
        "material_delta": DENSITY_MATERIAL_DELTA,
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "interval": "central-99-percent",
            "lower": lower,
            "upper": upper,
        },
        "outcome": outcome,
    }, bootstrap)


def _metric(value: Mapping[str, Any], *keys: str) -> float:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise Round0130Error(f"missing comparison metric {'/'.join(keys)}")
        current = current[key]
    try:
        result = float(current)
    except (TypeError, ValueError) as exc:
        raise Round0130Error(
            f"nonnumeric comparison metric {'/'.join(keys)}"
        ) from exc
    if not np.isfinite(result):
        raise Round0130Error(f"nonfinite comparison metric {'/'.join(keys)}")
    return result


def noninferiority_checks(
    *,
    control_core: Mapping[str, Any],
    treatment_core: Mapping[str, Any],
    control_ood: Mapping[str, Any],
    treatment_ood: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply frozen native non-density and Polish/in-mix margins."""
    control_ffr = _metric(control_core, "metrics", "global", "ffr")
    treatment_ffr = _metric(treatment_core, "metrics", "global", "ffr")
    control_r10 = _metric(
        control_core, "metrics", "global", "recall_at_10"
    )
    treatment_r10 = _metric(
        treatment_core, "metrics", "global", "recall_at_10"
    )
    control_r50 = _metric(
        control_core, "metrics", "global", "recall_at_50_of_high10"
    )
    treatment_r50 = _metric(
        treatment_core, "metrics", "global", "recall_at_50_of_high10"
    )
    treatment_core_checks = (treatment_core.get("decision") or {}).get(
        "checks"
    )
    if not isinstance(treatment_core_checks, Mapping):
        raise Round0130Error("treatment core checks are missing")
    required_non_density = {
        "coordinates_finite_and_noncollapsed",
        "every_language_ffr_at_least_0_40_of_pooled_english",
        "global_ffr_at_least_0_40",
        "global_recall50_strictly_exceeds_recall10",
    }
    absolute_core = all(
        bool(treatment_core_checks.get(key)) for key in required_non_density
    )

    def ood_metric(receipt: Mapping[str, Any], language: str, key: str) -> float:
        return _metric(receipt, "language_cells", language, "probe", key)

    control_polish = ood_metric(
        control_ood, "pol_Latn", "recall_at_50_of_high10"
    )
    treatment_polish = ood_metric(
        treatment_ood, "pol_Latn", "recall_at_50_of_high10"
    )
    control_inmix = _metric(
        control_ood,
        "headline_decision",
        "in_mix_median_recall_at_50_of_high10",
    )
    treatment_inmix = _metric(
        treatment_ood,
        "headline_decision",
        "in_mix_median_recall_at_50_of_high10",
    )
    control_ratio = _metric(
        control_ood,
        "headline_decision",
        "polish_to_in_mix_median_ratio",
    )
    treatment_ratio = _metric(
        treatment_ood,
        "headline_decision",
        "polish_to_in_mix_median_ratio",
    )
    treatment_ood_absolute = bool(
        (treatment_ood.get("headline_decision") or {}).get("passed")
    )
    checks = {
        "native_absolute_non_density_checks_pass": absolute_core,
        "global_ffr_at_least_0p97_control": (
            treatment_ffr >= HEADLINE_KPI_RETENTION * control_ffr
        ),
        "global_recall10_at_least_0p97_control": (
            treatment_r10 >= HEADLINE_KPI_RETENTION * control_r10
        ),
        "global_recall50_at_least_0p97_control": (
            treatment_r50 >= HEADLINE_KPI_RETENTION * control_r50
        ),
        "frozen_polish_absolute_gate_pass": treatment_ood_absolute,
        "polish_recall50_at_least_0p97_control": (
            treatment_polish >= HEADLINE_KPI_RETENTION * control_polish
        ),
        "in_mix_median_recall50_at_least_0p97_control": (
            treatment_inmix >= HEADLINE_KPI_RETENTION * control_inmix
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "global_ffr": {"control": control_ffr, "treatment": treatment_ffr},
            "global_recall_at_10": {
                "control": control_r10,
                "treatment": treatment_r10,
            },
            "global_recall_at_50_of_high10": {
                "control": control_r50,
                "treatment": treatment_r50,
            },
            "polish_recall_at_50_of_high10": {
                "control": control_polish,
                "treatment": treatment_polish,
            },
            "in_mix_median_recall_at_50": {
                "control": control_inmix,
                "treatment": treatment_inmix,
            },
            "polish_to_in_mix_median_ratio": {
                "control": control_ratio,
                "treatment": treatment_ratio,
            },
        },
    }
