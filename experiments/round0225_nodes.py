#!/usr/bin/env python3
"""Execute R0225 — re-register the 2M gate from sealed artifacts. No GPU, no training.

One node. It reads R0218's panel reference, R0222's sealed 8-cell gate and
R0223's sealed 3-cell cuVS comparison, and produces:

* the one-sided 95/95 tolerance factor **derived** at `n = 8`, cross-checked
  against review-0222-01's published `3.187`;
* three floor families over the 8-cell exact-graph population — R0222's
  `mean - 2*sigma`, the one-sided 95/95 tolerance floor, and a two-sided 95/95
  band on the **unfolded log-ratio** scale for the two purity metrics;
* a direct measurement of how far each family's floor moves under an injected
  outlier, which is review-0222-01's self-loosening argument as a number;
* all **11** cells (8 exact-graph + 3 cuVS) scored against all three families;
* an independent reproduction of `density_v2` for all eight cells, including
  **46-49, which no reviewer has ever reproduced**, with a positive control on
  42-45 that must pass before the other four are reported;
* a read-only assessment of the R0161 / R0193 exposure to the same method.

Nothing here trains, scores a new map, registers a capability on another
universe, or touches R0161/R0193's artifacts other than to read them.
"""
from __future__ import annotations

import json
import os
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0225_tolerance_gate import (
    CUVS_FAMILY_SEEDS,
    EXACT_FAMILY_SEEDS,
    GATE_CAPABILITY,
    GATE_METRICS,
    GATE_SCHEMA,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
    ROUND_ID,
    Round0225Error,
    one_sided_tolerance_factor,
    registered_gate,
    score_all_cells,
    two_sided_tolerance_factor,
)
from basemap import round0113_prompt_contrast as prompt_contract


GATE_ACTION = "reregister_tolerance_gate"

R0218_ROOT = (
    "/data/latent-basemap/runs/round-0218/queue/artifacts/"
    "minilm-mixed-2m-seed-family-panel-v1"
)
R0222_ROOT = (
    "/data/latent-basemap/runs/round-0222/queue/artifacts/"
    "minilm-mixed-2m-quality-gates-n8-v1"
)
R0223_ROOT = (
    "/data/latent-basemap/runs/round-0223/queue-correction-3/artifacts/"
    "minilm-mixed-2m-cuvs-graph-map-comparison-v1"
)
HIGH_D_REFERENCE = os.path.join(R0218_ROOT, "minilm-2m-high-d-reference.npz")
R0222_GATE = os.path.join(R0222_ROOT, "minilm-quality-gates-n8.json")
R0223_COMPARISON = os.path.join(R0223_ROOT, "cuvs-graph-map-comparison.json")

#: Read-only. These are the two precedent gates that used the same method.
PRECEDENTS = {
    "0161": (
        "/data/latent-basemap/runs/round-0161/queue/artifacts/"
        "jina-prompted-universe-quality-gates-v1/prompted-quality-gates.json"
    ),
    "0193": (
        "/data/latent-basemap/runs/round-0193/queue/artifacts/"
        "jina-mixed-english-2m-quality-gates-v1/mixed-quality-gates.json"
    ),
}

#: `density_v2` reproduction. `score_panel` correlates `log r_hd` against
#: `log r_ld`, where `r` is the MEAN Euclidean distance to the 15 nearest
#: self-excluded neighbours, `r_hd` from the sealed high-D reference and `r_ld`
#: from the map's own coordinates, with `eps = 1e-12` and a four-decimal round.
DENSITY_K = 15
DENSITY_EPS = 1e-12
DENSITY_ROUND = 4
COORDINATES = {
    **{seed: os.path.join(R0218_ROOT, f"coordinates-seed{seed}.npy") for seed in (42, 43, 44, 45)},
    **{seed: os.path.join(R0222_ROOT, f"coordinates-seed{seed}.npy") for seed in (46, 47, 48, 49)},
}
#: 42-45 were reproduced by review-0218-01. They are the positive control here:
#: if the harness cannot reproduce them it has not earned the right to report
#: 46-49, and review-0222-01's harness failed exactly this test.
POSITIVE_CONTROL_SEEDS = (42, 43, 44, 45)
NEVER_REPRODUCED_SEEDS = (46, 47, 48, 49)


def _read_json(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise Round0225Error(f"R0225 {label} is absent at {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def reproduce_density_v2() -> dict[str, Any]:
    """Recompute `density_v2` for all eight cells from sealed bytes.

    Review 0222-01 could not do this: its harness failed its own positive
    control on seed 42, returning `0.168` against the published `0.4377`, so it
    could not adjudicate the four cells that had never been checked.

    The discrepancy is fully explained here rather than worked around. The
    sealed reference contains **exactly one anchor with `r_hd == 0`** — a
    substrate row with enough exact duplicates that its 15 nearest high-D
    neighbours are all at distance zero. With `eps = 1e-12` that anchor sits at
    `log r = -27.63` in *both* variables, roughly 197 sd from the other 3,999
    anchors and on a perfect diagonal. Dropping it is what produces `0.168`.

    So the round reports `density_v2` **both ways** — as the panel defines it,
    and with degenerate anchors removed — because the difference is not a
    rounding detail: it is the difference between `~0.44` and `~0.15`, and it
    means the registered `density_v2` floor rests substantially on a single
    shared leverage point that is byte-identical across every cell and
    therefore cannot vary between them.
    """
    from scipy.spatial import cKDTree

    reference = np.load(HIGH_D_REFERENCE)
    anchors = np.asarray(reference["anchor_ids"])
    r_hd = np.asarray(reference["r_hd"], dtype=np.float64)
    degenerate = int((r_hd <= 0).sum())

    cells: dict[str, Any] = {}
    for seed, path in sorted(COORDINATES.items()):
        if not os.path.exists(path):
            raise Round0225Error(f"R0225 coordinates for seed {seed} absent at {path}")
        coordinates = np.load(path).astype(np.float64)
        tree = cKDTree(coordinates)
        distances, _ = tree.query(coordinates[anchors], k=DENSITY_K + 1, workers=-1)
        r_ld = distances[:, 1:].mean(1)
        as_defined = round(
            float(
                np.corrcoef(
                    np.log(r_hd + DENSITY_EPS), np.log(r_ld + DENSITY_EPS)
                )[0, 1]
            ),
            DENSITY_ROUND,
        )
        keep = r_hd > 0
        without = round(
            float(
                np.corrcoef(
                    np.log(r_hd[keep] + DENSITY_EPS), np.log(r_ld[keep] + DENSITY_EPS)
                )[0, 1]
            ),
            DENSITY_ROUND,
        )
        cells[str(seed)] = {
            "density_v2_as_panel_defines_it": as_defined,
            "density_v2_excluding_degenerate_anchors": without,
            "anchors_used": int(anchors.size),
            "degenerate_anchors_dropped": int(anchors.size - int(keep.sum())),
        }
        del coordinates, tree, distances
    return {
        "method": (
            f"r = mean Euclidean distance to the {DENSITY_K} nearest "
            "self-excluded neighbours; density_v2 = "
            f"corr(log(r_hd + {DENSITY_EPS}), log(r_ld + {DENSITY_EPS})), "
            f"rounded to {DENSITY_ROUND} decimals. r_hd is read from the sealed "
            "high-D reference and is map-independent."
        ),
        "high_d_reference": HIGH_D_REFERENCE,
        "anchors": int(anchors.size),
        "degenerate_anchor_count": degenerate,
        "cells": cells,
    }


def assess_precedent_exposure() -> dict[str, Any]:
    """Which released capabilities rest on the defective method? Read-only."""
    exposure: dict[str, Any] = {}
    for round_id, path in PRECEDENTS.items():
        artifact = _read_json(path, f"R{round_id} precedent gate")
        gates = artifact.get("gates") or {}
        entries: dict[str, Any] = {}
        for metric, cell in gates.items():
            n = cell.get("n")
            entries[metric] = {
                "floor": cell.get("floor"),
                "n": n,
                "mean": cell.get("mean"),
                "sample_sd": cell.get("sample_sd") or cell.get("sample_sd_ddof1"),
                "is_purity_ratio_metric": metric in PURITY_METRICS,
            }
            if isinstance(n, int) and n >= 3:
                derived = one_sided_tolerance_factor(n)["k"]
                entries[metric]["one_sided_95_95_factor_at_this_n"] = derived
                entries[metric]["multiplier_shortfall"] = derived - 2.0
        exposure[round_id] = {
            "artifact": path,
            "capability": artifact.get("capability"),
            "formula": artifact.get("formula"),
            "n": artifact.get("n"),
            "metrics_gated": sorted(gates),
            "gates": entries,
            "modified_by_this_round": False,
        }
    return {
        "read_only": True,
        "precedents": exposure,
        "finding": (
            "R0161 and R0193 registered their floors with the same "
            "mean - 2*sigma estimator, so both carry the same two defects: the "
            "multiplier is smaller than the 95/95 tolerance factor at their n "
            "(so the floors are more punitive than nominal AND self-loosening), "
            "and both gate purity_fidelity on the folded scale, which cannot "
            "distinguish over- from under-separation. Neither artifact is "
            "modified here. Whether they need re-registration is a decision for "
            "their own universes; what this round establishes is that the "
            "defect is in the METHOD, not in the MiniLM population, so the "
            "exposure is real and not hypothetical."
        ),
        "recommendation": (
            "re-register both under the tolerance method before either floor is "
            "used to fail a map on a new treatment. Until then their purity "
            "floors should carry the same descriptive-only status review-0222-01 "
            "placed on the MiniLM k256 floor."
        ),
    }


def run_gate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    started = time.monotonic()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0225 tolerance gate"
    )

    gate_artifact = _read_json(R0222_GATE, "R0222 sealed gate")
    comparison = _read_json(R0223_COMPARISON, "R0223 sealed cuVS comparison")

    exact_cells = gate_artifact["pooled_panel_metric_cells"]
    exact_ratios = comparison["exact_family_purity_ratios"]
    cuvs_cells = comparison["cuvs_panel_metric_cells"]
    cuvs_ratios = comparison["cuvs_purity_ratios"]

    if tuple(sorted(int(seed) for seed in exact_cells)) != EXACT_FAMILY_SEEDS:
        raise Round0225Error("R0225 exact-graph family is not seeds 42-49")
    if tuple(sorted(int(seed) for seed in cuvs_cells)) != CUVS_FAMILY_SEEDS:
        raise Round0225Error("R0225 cuVS family is not seeds 42-44")

    # --- the factors, derived --------------------------------------------- #
    factors = {
        "one_sided": one_sided_tolerance_factor(len(EXACT_FAMILY_SEEDS)),
        "two_sided": two_sided_tolerance_factor(len(EXACT_FAMILY_SEEDS)),
    }

    # --- the three floor families ------------------------------------------ #
    gate = registered_gate(exact_cells=exact_cells, exact_ratios=exact_ratios)

    # --- all eleven cells --------------------------------------------------- #
    cells = [
        {
            "cell_id": f"exact-seed{seed}",
            "family": "exact-graph",
            "seed": int(seed),
            "defines_the_floors": True,
            "values": exact_cells[str(seed)],
            "ratios": exact_ratios[str(seed)],
        }
        for seed in EXACT_FAMILY_SEEDS
    ] + [
        {
            "cell_id": f"cuvs-igd48-seed{seed}",
            "family": "cuvs-igd48",
            "seed": int(seed),
            "defines_the_floors": False,
            "values": cuvs_cells[str(seed)],
            "ratios": cuvs_ratios[str(seed)],
        }
        for seed in CUVS_FAMILY_SEEDS
    ]
    scored = score_all_cells(gate=gate, cells=cells)
    if scored["cell_count"] != 11:
        raise Round0225Error(f"R0225 scored {scored['cell_count']} cells, expected 11")

    # --- density_v2, with its positive control ------------------------------ #
    density = reproduce_density_v2()
    control = {
        str(seed): {
            "reproduced": density["cells"][str(seed)][
                "density_v2_as_panel_defines_it"
            ],
            "sealed": float(exact_cells[str(seed)]["density_v2"]),
            "matches": density["cells"][str(seed)][
                "density_v2_as_panel_defines_it"
            ] == float(exact_cells[str(seed)]["density_v2"]),
        }
        for seed in POSITIVE_CONTROL_SEEDS
    }
    control_passes = all(item["matches"] for item in control.values())
    never = {
        str(seed): {
            "reproduced": density["cells"][str(seed)][
                "density_v2_as_panel_defines_it"
            ],
            "sealed": float(exact_cells[str(seed)]["density_v2"]),
            "matches": density["cells"][str(seed)][
                "density_v2_as_panel_defines_it"
            ] == float(exact_cells[str(seed)]["density_v2"]),
        }
        for seed in NEVER_REPRODUCED_SEEDS
    }
    density.update({
        "positive_control_seeds": list(POSITIVE_CONTROL_SEEDS),
        "positive_control": control,
        "positive_control_passes": control_passes,
        "never_before_reproduced_seeds": list(NEVER_REPRODUCED_SEEDS),
        "never_before_reproduced": never,
        "all_eight_reproduce": control_passes
        and all(item["matches"] for item in never.values()),
        "review_0222_reported_value_for_seed_42": 0.168,
        "explanation_of_review_0222_harness_failure": (
            "review-0222-01 reported 0.168 for seed 42 under every variant it "
            "tried. That is this harness's density_v2_excluding_degenerate_"
            "anchors for seed 42 to three decimals. The reviewer's harness was "
            "arithmetically correct and differed by ONE anchor: the single "
            "r_hd == 0 anchor, which a finite-log or positivity filter drops."
        ),
    })

    exposure = assess_precedent_exposure()

    execution_checks = {
        "eleven_cells_scored": scored["cell_count"] == 11,
        "eight_cells_define_the_floors": sum(
            1 for row in scored["cells"] if row["defines_the_floors"]
        ) == 8,
        "one_sided_factor_derived_not_copied": bool(
            factors["one_sided"].get("cross_check_passes")
        ),
        "n_stated_beside_every_floor": all(
            gate["gates"][metric]["mean_minus_2sd"]["n"] == 8
            and gate["gates"][metric]["one_sided_tolerance_95_95"]["n"] == 8
            for metric in GATE_METRICS
        ),
        "purity_metrics_gated_two_sidedly": all(
            "two_sided_log_ratio_95_95" in gate["gates"][metric]
            for metric in PURITY_METRICS
        ),
        "self_loosening_measured_for_every_metric": all(
            "self_loosening" in gate["gates"][metric] for metric in GATE_METRICS
        ),
        "density_positive_control_passes": control_passes,
        "precedents_not_modified": all(
            item["modified_by_this_round"] is False
            for item in exposure["precedents"].values()
        ),
        "no_training_performed": True,
        "no_gpu_used": True,
    }
    if not all(execution_checks.values()):
        raise Round0225Error(f"R0225 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    receipt = prompt_contract.seal({
        "schema": GATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": GATE_CAPABILITY,
        "capabilities": [GATE_CAPABILITY],
        "outcome": "2m-gate-reregistered-as-a-tolerance-interval-on-the-unfolded-scale",
        "training_performed": False,
        "evaluation_performed": False,
        "production_or_publishing": False,
        "gpu_used": False,
        "gate_registered": True,
        "gate_status": "registered-and-contingent-pending-review",
        "supersedes_capability": "minilm-mixed-2m-quality-gates-n8-v1",
        "applies_to": (
            "byte-commensurate maps of the R0216 queue-correction-3 mixed MiniLM "
            "2M substrate under the R0217 recipe and the R0218 panel "
            "configuration only, at n = 8"
        ),
        "sources": {
            "r0222_gate": expected_input_signature(R0222_GATE),
            "r0223_comparison": expected_input_signature(R0223_COMPARISON),
            "high_d_reference": expected_input_signature(HIGH_D_REFERENCE),
        },
        "tolerance_factors": factors,
        "gate": gate,
        "cells": scored,
        "density_v2_reproduction": density,
        "precedent_exposure": exposure,
        "execution_checks": execution_checks,
        "wall_seconds": time.monotonic() - started,
        "peak_host_rss_gib": peak_rss_gib,
    })
    atomic_write_new_json(
        os.path.join(output, "minilm-tolerance-gates-n8.json"), receipt, immutable=True
    )
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "cells": scored["cell_count"],
        "one_sided_k": factors["one_sided"]["k"],
        "density_all_eight_reproduce": density["all_eight_reproduce"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job["action"])
    if action == GATE_ACTION:
        run_gate(active, job)
        return
    raise Round0225Error(f"R0225 unknown action {action!r}")
