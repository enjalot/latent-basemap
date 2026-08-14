"""Execute R0265 — thirteen fneg 2M cells, the n=13 family panel, and the gate.

Fifteen nodes in one queue, mirroring R0255's train -> panel -> floor shape:

* thirteen GPU trains (`seeds 42-54`) under the PINNED fneg recipe, the seed the
  only free variable, each config built and proved by `round0265_fneg_treatment`
  (umap kernel a=1.9328/b=0.7905, dose x4, fneg_weight=1.0/lo=0.1/hi=0.4);
* `score_minilm_fneg_2m_panel_n13` (GPU) — the thirteen maps scored on held-out FFR
  (R0233's sealed 20k reserve, truth = exact cosine top-10 in the 2M substrate),
  purity_fidelity_k256/k1024 (R0218's frozen panel reference), COLLAPSE (r10/R*sqrt(N))
  and FOG (low_density_mass_fraction), plus the seed-42 cross-check against the sandbox
  `umap-md000-x4-fneg10@2m` cell;
* `register_fneg_family_floors_n13` (CPU) — pool n=13, read R0234's sealed 95/95 n=13
  multiplier, fit the held-out FFR floor, CONVERT R0264's provisional collapse floor +
  fog ceiling to family-calibrated on this fneg family, and GATE on: held-out FFR floor
  + purity k256 two-sided band (R0263's registered criterion) + collapse floor +
  fail-closed fog ceiling (R0264). density_v2/v3 stay descriptive-only.

Every registered instrument is IMPORTED and called: R0263's `_judge_k256_two_sided`,
R0264's `_judge_collapse` / `_judge_fog` / `_map_fog`, R0234's `floor_at` /
`attainability`. The guard/seal/poll-recorder/`ParametricUMAP.abort_poll` machinery is
R0255's, imported rather than re-typed. Nothing signals any process, hands cuVS
anything, or wraps a child in a timeout.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import resource
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0217_minilm_2m_seed_family import (
    WARMUP_SUCCESSFUL_UPDATES,
    validate_published_map,
)
from basemap.round0242_locality import json_scrub
from basemap.round0238_rung5 import json_safe
from basemap.round0234_calibrated_floors import (
    MAD_CONSISTENCY,
    attainability,
    band_at,
    centre_and_scale,
    floor_at,
    identity_bound,
)
from basemap.round0245_guard import (
    EnforcedHostWatchdog,
    require_enforceable_abort_flag,
)
from basemap.round0246_guard import (
    AbortPollGate,
    Round0246Error,
    measured_slope_from_trace,
    require_abort_flag_landed,
    require_enforcement_evidence,
    require_live_sampler,
)
from basemap.round0247_registry import (
    Round0247Error,
    registered_bounds,
    registry_fingerprint,
    verify_registry,
)
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report
from basemap.round0253_coverage import CoverageLedger
from basemap.round0253_stop_hooks import install_stop_hooks
from basemap.round0265_fneg_treatment import (
    CANONICAL_SEED,
    CAPABILITIES,
    DOSE_MULTIPLIER,
    SEALED_DIRECTED_EDGES,
    FNEG_A,
    FNEG_B,
    FNEG_HI,
    FNEG_HORIZON,
    FNEG_LO,
    FNEG_LOW_DIM_KERNEL,
    FNEG_TRAIN_CONFIG_SCHEMA,
    FNEG_WEIGHT,
    ROUND_ID,
    SEEDS,
    Round0265FamilyError,
    Round0265RecipeError,
    Round0265TreatmentError,
    TRAIN_CLOSURE_MODULES,
    assert_family_shares_one_recipe,
    assert_registered_recipe,
    assert_runtime_closure_matches_seal,
    capability_for_seed,
    exact_cell_id,
    fneg_seed_invariant_sha256,
    fneg_train_config,
    recipe_refusal_controls,
    runtime_closure_hashes,
    treatment_closure_controls,
)

from experiments import round0218_nodes
from experiments.round0230_nodes import (
    CellWatchdog,
    _open_substrate,
    _sealed_graph,
    _weighted_rejection_accounting_mismatch,
)
from experiments.round0113_nodes import _new_model
from experiments.round0238_nodes import _check_runner_abort
from basemap.round0260_one_sided_purity import (
    KIND_RATIO_FLOOR,
    one_sided_ratio_floor,
)
from experiments.round0263_nodes import (
    GATE_CAPABILITY as R0263_GATE_CAPABILITY,
    GATE_SCHEMA as R0263_GATE_SCHEMA,
    RESTORED_METRIC as PURITY_K256_METRIC,
    SEALED_K1024_ONE_SIDED_FLOOR as R0260_LEGACY_K1024_FLOOR,
    UNCHANGED_RATIO_METRIC as PURITY_K1024_METRIC,
    _fit_two_sided_k256_band,
    _judge_k256_two_sided,
    _judge_one_sided_floor,
)
from experiments.round0264_nodes import (
    COLLAPSE_METRIC,
    FOG_METRIC,
    GATE_CAPABILITY as R0264_GATE_CAPABILITY,
    GATE_SCHEMA as R0264_GATE_SCHEMA,
    VERDICT_NOT_MEASURABLE,
    _judge_collapse,
    _judge_fog,
    _map_fog,
    _robust_ceiling,
    _robust_floor,
)


TRAIN_ACTION = "train_minilm_fneg_2m_seed"
PANEL_ACTION = "score_minilm_fneg_2m_panel_n13"
GATE_ACTION = "register_fneg_family_floors_n13"

# --------------------------------------------------------------------------- #
# registered capabilities + schemas
# --------------------------------------------------------------------------- #

TRAIN_SCHEMA = "round0265-minilm-fneg-2m-train-receipt-v1"
PANEL_CAPABILITY = "minilm-fneg-2m-family-panel-n13-v1"
PANEL_SCHEMA = "round0265-minilm-fneg-2m-family-panel-n13-v1"
GATE_CAPABILITY = "minilm-fneg-2m-family-floors-n13-v1"
GATE_SCHEMA = "round0265-minilm-fneg-2m-family-floors-n13-v1"

CAPABILITY_TEMPLATE = "minilm-mixed-2m-fneg-x4-md000-seed{seed}-r0265-v1"

ROWS = 2_000_000
DIMENSION = 384
N_FAMILY = len(SEEDS)

#: The registered estimator the whole program calibrates: median - k*MAD_n.
ESTIMATOR = "median_minus_k_madn"

#: The metrics this round scores. Held-out FFR + purity gate; collapse/fog gate;
#: density_v2/v3 are DESCRIPTIVE-ONLY and never gate a map.
HELDOUT_FFR_METRIC = "heldout_ffr"
GATED_METRICS: tuple[str, ...] = (
    HELDOUT_FFR_METRIC, PURITY_K256_METRIC, PURITY_K1024_METRIC, COLLAPSE_METRIC, FOG_METRIC
)
DESCRIPTIVE_METRICS: tuple[str, ...] = ("density_v2", "density_v3")

#: R0234's sealed n=13 95/95 (two-sided) MAD_n multiplier. READ from the artifact at
#: runtime; this literal is the published cross-check anchor, never used in its place.
R0234_N13_TWO_SIDED_KEY = (
    "calibration.n13.candidates.median_minus_k_madn.two_sided.calibrated_multiplier"
)
R0234_N13_ONE_SIDED_KEY = (
    "calibration.n13.candidates.median_minus_k_madn.one_sided.calibrated_multiplier"
)
SEALED_N13_TWO_SIDED_MULTIPLIER = 4.452408030160313
SEALED_N13_ONE_SIDED_MULTIPLIER = 3.7363661743730416

#: LEGACY, DESCRIPTIVE-ONLY cross-treatment reference values. Per the owner-proxy
#: ruling (2026-08-14): per-family calibration is registered law — EVERY R0265 gate
#: value is FITTED from the fneg family's own thirteen cells. These legacy numbers,
#: fitted on OTHER treatment families, are published beside the fitted criteria as a
#: seed-matched contrast and NEVER enter the gate verdict.
#:   R0264's provisional collapse floor / fog ceiling (n=7 non-seed family):
R0264_PROVISIONAL_COLLAPSE_FLOOR = 0.644414
R0264_PROVISIONAL_FOG_CEILING = 0.732966
#:   R0263's registered two-sided k256 band (n=29 exact-graph legacy family):
R0263_LEGACY_K256_BAND = {"ratio_lower": 0.9697263687993266, "ratio_upper": 1.0550731452902518}
#:   R0260's registered k1024 one-sided ratio floor (imported from R0263's sealed copy):
#:   R0260_LEGACY_K1024_FLOOR == 0.6690913806270514

#: The sandbox `umap-md000-x4-fneg10@2m` seed-42 cell the cross-check compares against.
#: A DIFFERENT core lineage (the sandbox round-0115 core, not the merged round-0208
#: core), so the requirement is metric-close, not byte-equal.
SEED42_CROSS_CHECK_TARGET = {
    "arm": "umap-md000-x4-fneg10",
    "rung": "2m",
    "seed": 42,
    "heldout_ffr": 0.41113,
    "low_density_mass_fraction": 0.3706289999204829,
    "r10_over_map_radius_median": 0.0007509993201379904,
    "collapse": 1.062,
    "source": "/data/latent-basemap/sandbox/2m-knobs/heldout-eval.json results[umap-md000-x4-fneg10]",
}
#: Metric-close tolerances for the cross-check. Wide enough for a different core
#: lineage + a different seeded held-out truth; tight enough that a broken projector
#: (regressor-level FFR ~0.34, or a collapsed/foggy map) would miss.
SEED42_TOLERANCE = {"heldout_ffr": 0.05, "collapse": 0.15, "fog": 0.05}

#: Held-out FFR protocol (R0233 sealed reserve, sandbox `heldout_eval.py`).
N_PROBES = 20_000
K_TRUE = 10
FFR_DISC = int(ROWS * 0.001)  # 2000
COLLAPSE_K = 10
COLLAPSE_RADIUS_PCT = 90
COLLAPSE_SAMPLE = 20_000
COLLAPSE_RNG_SEED = 0
FOG_BINS = 1024

TRANSFORM_CHUNK_ROWS = 100_000
FULL_TRANSFORM_BATCH = 8192
BATCH_SIZE = 8192
POSITIVE_ROWS_PER_UPDATE = 409

DEVICE_BUDGET_BYTES = 30 * (1 << 30)
HOST_ANON_BUDGET_BYTES = 24 * (1 << 30)
HOST_RSS_LIMIT_GIB = 40.0
NODE_ANON_BUDGET_BYTES = 16 * (1 << 30)

SAFETY_NOTE = (
    "no node in this module signals any process, starts a child process, hands cuVS "
    "anything, or wraps a subprocess in a timeout. Every bulk input is a read-only "
    "np.memmap. The per-batch abort read is the release's own ParametricUMAP.abort_poll "
    "attribute, set to this node's recorder and cleared in a finally."
)


class Round0265NodeError(RuntimeError):
    """The R0265 node contract changed."""


# --------------------------------------------------------------------------- #
# shared node scaffolding (imported R0255 patterns)
# --------------------------------------------------------------------------- #


def _bound_path(job: Mapping[str, Any], key: str, *, label: str) -> str:
    reference = job.get(key)
    if not isinstance(reference, Mapping):
        raise Round0265NodeError(f"R0265 job is missing the bound input {key!r} ({label})")
    if reference.get("sha256"):
        return prompt_contract.verify_signature(dict(reference), label=label)
    path = str(reference["canonical_path"])
    if not os.path.exists(path):
        raise Round0265NodeError(f"{label} is absent at {path}")
    return path


def _intra_queue_signature(reference: Mapping[str, Any], *, label: str) -> tuple[str, dict[str, Any]]:
    reference = dict(reference)
    if reference.get("sha256"):
        return prompt_contract.verify_signature(reference, label=label), reference
    path = str(reference["canonical_path"])
    if not os.path.exists(path):
        raise Round0265NodeError(f"{label} is absent at {path}")
    return path, expected_input_signature(path)


def _start_node(label: str) -> dict[str, Any]:
    verify_registry(label=label)
    return require_enforceable_abort_flag(label=label)


def _node_guard(label: str, *, interval_s: float = 0.05) -> EnforcedHostWatchdog:
    return EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        interval_s=float(interval_s),
        label=label,
    )


def _node_gate(label: str, *, training_performed: bool) -> AbortPollGate:
    return AbortPollGate(
        inner=_check_runner_abort,
        headroom_bytes=int(
            registered_bounds(["max_declared_headroom_bytes"])[
                "registered_max_declared_headroom_bytes"
            ]
        ),
        label=label,
        training_performed=bool(training_performed),
    )


def _guard_tail_reported(watchdog: EnforcedHostWatchdog, *, label: str) -> dict[str, Any]:
    receipt = watchdog.receipt()
    tail: dict[str, Any] = {"host_watchdog": receipt}
    for key, gate in (
        ("sampler_liveness", require_live_sampler),
        ("abort_flag_landing", require_abort_flag_landed),
    ):
        try:
            tail[key] = gate(receipt, label=label)
        except (Round0246Error, Round0247Error) as error:
            tail[key] = {
                "holds": False,
                "raised": f"{type(error).__name__}: {error}",
                "reported_rather_than_aborting": True,
            }
    return tail


def _score_gate_without_raising(gate: AbortPollGate, tail: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    slope = measured_slope_from_trace(tail["host_watchdog"]["anonymous_trace_by_second"])
    verdict = gate.verdict(measured_slope_bytes_per_s=slope)
    outcome: dict[str, Any] = {"measured_slope_bytes_per_s": slope, "require_raised": False, "require_error": None}
    try:
        required = gate.require(measured_slope_bytes_per_s=slope)
    except (Round0246Error, Round0247Error) as error:
        outcome["require_raised"] = True
        outcome["require_error"] = f"{type(error).__name__}: {error}"
        required = dict(verdict)
        required["failures"] = [
            arm for arm in (
                "meets_the_registered_ceiling",
                "meets_the_required_poll_count",
                "measured_spacing_is_non_zero",
            ) if verdict.get(arm) is False
        ]
        required["holds"] = not required["failures"]
    try:
        required["enforcement_evidence"] = require_enforcement_evidence(required, label=label)
    except (Round0246Error, Round0247Error) as error:
        required["enforcement_evidence"] = {"holds": False, "error": f"{type(error).__name__}: {error}"}
    required["outcome"] = outcome
    return required


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "registry_fingerprint": registry_fingerprint(),
        "safety_note": SAFETY_NOTE,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": False,
    }


def _seal(output: str, name: str, body: Mapping[str, Any]) -> None:
    atomic_write_new_json(
        os.path.join(output, name),
        prompt_contract.seal(json_safe(json_scrub(dict(body)))),
        immutable=True,
    )


def _closure_evidence(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sealed = prompt_contract.read_sealed(
        _bound_path(job, "treatment_closure", label="R0265 treatment closure seal"),
        label="R0265 treatment closure seal",
    )
    observed = runtime_closure_hashes(TRAIN_CLOSURE_MODULES)
    verdict = assert_runtime_closure_matches_seal(sealed=sealed, observed=observed)
    return sealed, {
        "runtime_closure": observed,
        "verdict": verdict,
        "controls": treatment_closure_controls(sealed=sealed, observed=observed),
    }


def _build_fneg_model(config: Mapping[str, Any]):
    """`_new_model` plus the three fneg kwargs the config→model bridge does not thread.

    `_new_model` builds R0217's constructor call but predates the fneg merge, so it
    passes neither `fneg_weight` nor the band. This wraps it: the base model is built
    from the config, then the fneg fields are set on the instance BEFORE `fit()` reads
    them. The recipe invariant has already proved they are the pinned values.
    """
    model = _new_model(config)
    optimizer = config["optimizer"]
    model.fneg_weight = float(optimizer["fneg_weight"])
    model.fneg_lo = float(optimizer["fneg_lo"])
    model.fneg_hi = float(optimizer["fneg_hi"])
    model.fneg_telemetry = None
    # `_new_model` hardcodes n_epochs=2 (fits R0217's low-dose horizon). The x4
    # dose horizon (320652 updates) needs the loader to supply >= that many
    # batches; at steps_per_epoch = ceil(edges/num_pos) that is 3 epochs. Scale
    # n_epochs up to cover the horizon (the sandbox run_arm rule); the horizon
    # break still ends training at total_steps_estimate, so the dose is unchanged.
    num_pos = max(1, int(model.batch_size * model.pos_ratio))
    steps_per_epoch = math.ceil(SEALED_DIRECTED_EDGES / num_pos)
    needed_epochs = math.ceil(int(model.total_steps_estimate) / steps_per_epoch)
    if needed_epochs > int(model.n_epochs):
        model.n_epochs = needed_epochs
    return model


def _transform_in_chunks(model: Any, source: Any, poll: Any) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, ROWS, TRANSFORM_CHUNK_ROWS):
        stop = min(start + TRANSFORM_CHUNK_ROWS, ROWS)
        block = np.asarray(
            model.transform(source[start:stop], batch_size=FULL_TRANSFORM_BATCH),
            dtype=np.float32,
        )
        parts.append(block)
        poll(f"R0265 transform rows {start}-{stop}")
    return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------- #
# the map-quality metrics: COLLAPSE (ported), FOG (imported), held-out FFR
# --------------------------------------------------------------------------- #

MAX_TREE_ROWS = 256 * (1 << 20) // 16  # 16_777_216, the cKDTree memory rule


def map_collapse(
    xy,
    *,
    sample_size: int = COLLAPSE_SAMPLE,
    rng_seed: int = COLLAPSE_RNG_SEED,
    k_neighbor: int = COLLAPSE_K,
    radius_pct: int = COLLAPSE_RADIUS_PCT,
    workers: int = 8,
    max_tree_rows: int = MAX_TREE_ROWS,
) -> dict[str, Any]:
    """Bead-collapse statistic — a byte-faithful port of the sandbox `collapse_fog.map_collapse`.

    COLLAPSE = (r10 / R) * sqrt(n_effective): median distance to the 10th neighbour
    over the p90 map radius, N-adjusted so it is commensurate across map sizes. The
    gate direction is one-sided from below (too small means the map folded into beads).
    """
    from scipy.spatial import cKDTree

    arr = np.asarray(xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected (N, 2) coordinates, got shape {arr.shape}")
    n_rows = int(arr.shape[0])
    if n_rows <= k_neighbor + 1:
        raise ValueError(f"need more than {k_neighbor + 1} rows, got {n_rows}")
    rng = np.random.default_rng(rng_seed)
    subsampled = n_rows > max_tree_rows
    if subsampled:
        base_idx = np.sort(rng.choice(n_rows, size=max_tree_rows, replace=False))
        base = np.asarray(arr[base_idx], dtype=np.float32)
    else:
        base = np.asarray(arr, dtype=np.float32)
    n_eff = int(base.shape[0])
    n_sample = int(min(sample_size, n_eff))
    sample_idx = np.sort(rng.choice(n_eff, size=n_sample, replace=False))
    sample = base[sample_idx]
    tree = cKDTree(base)
    dists, _ = tree.query(sample, k=k_neighbor + 1, workers=workers)
    r_k = float(np.median(dists[:, k_neighbor]))
    centroid = base.mean(axis=0)
    radius = float(np.percentile(np.linalg.norm(base - centroid, axis=1), radius_pct))
    ratio = r_k / radius if radius > 0 else float("nan")
    return {
        "metric": "collapse",
        "statistic": "r10_over_radius_times_sqrt_n",
        "gate_direction": "one_sided_lower_floor",
        "n_rows": n_rows,
        "n_effective": n_eff,
        "subsampled_for_tree": bool(subsampled),
        "sample_size": n_sample,
        "rng_seed": int(rng_seed),
        "k_neighbor": int(k_neighbor),
        "radius_pct": int(radius_pct),
        "r10_median": r_k,
        "map_radius_p90": radius,
        "r10_over_map_radius_median": ratio,
        "r10_over_radius_times_sqrt_n": ratio * math.sqrt(n_eff),
        "collapse": ratio * math.sqrt(n_eff),
    }


def heldout_ffr_scores(
    *,
    placed: np.ndarray,
    map_coordinates: np.ndarray,
    truth_top10: np.ndarray,
    disc: int = FFR_DISC,
    workers: int = 8,
) -> dict[str, Any]:
    """Held-out FFR@0.1% + the kNN-regressor guard — the sandbox `heldout_eval` protocol.

    `placed` are the 20k held-out probes projected through the trained map; `truth_top10`
    are their exact cosine top-10 neighbours as ROW INDICES into the 2M substrate (== into
    `map_coordinates`). FFR = fraction of a probe's true neighbours found among the `disc`
    map neighbours of its projected point. The regressor places each probe at the mean 2D
    position of its true neighbours instead: if a dumb regressor matches, the map is a
    picture, not a projector.
    """
    from scipy.spatial import cKDTree

    placed = np.asarray(placed, dtype=np.float32)
    coords = np.asarray(map_coordinates, dtype=np.float32)
    truth = np.asarray(truth_top10, dtype=np.int64)
    tree = cKDTree(coords)

    def _ffr(query: np.ndarray) -> float:
        _, near = tree.query(np.asarray(query, dtype=np.float32), k=disc, workers=workers)
        hits = sum(int(np.isin(truth[i], near[i]).sum()) for i in range(len(query)))
        return float(hits / truth.size)

    heldout = _ffr(placed)
    regressor_positions = coords[truth].mean(axis=1)
    regressor = _ffr(regressor_positions)
    return {
        "heldout_ffr": heldout,
        "regressor_ffr": regressor,
        "net_minus_regressor": heldout - regressor,
        "disc": int(disc),
        "n_probes": int(len(placed)),
        "k_true": int(truth.shape[1]),
    }


def score_one_map(
    *,
    coordinates: np.ndarray,
    probes_placed: np.ndarray,
    truth_top10: np.ndarray,
    purity_ratios: Mapping[str, float],
) -> dict[str, Any]:
    """The four registrable numbers for one map, assembled from the three programs.

    `purity_ratios` are `panel_v2`'s raw unfolded k256/k1024 purity ratios (computed by
    the caller against R0218's frozen reference, so they stay commensurate with R0263's
    registered band). Collapse and fog are measured here on the coordinates; held-out FFR
    from the projected probes.
    """
    ffr = heldout_ffr_scores(
        placed=probes_placed, map_coordinates=coordinates, truth_top10=truth_top10
    )
    collapse = map_collapse(coordinates)
    fog = _map_fog(coordinates, bins=FOG_BINS)
    return {
        "heldout_ffr": float(ffr["heldout_ffr"]),
        "regressor_ffr": float(ffr["regressor_ffr"]),
        "net_minus_regressor": float(ffr["net_minus_regressor"]),
        "heldout_ffr_detail": ffr,
        "purity_fidelity_k256": float(purity_ratios["k256"]),
        "purity_fidelity_k1024": float(purity_ratios["k1024"]),
        "collapse": float(collapse["r10_over_radius_times_sqrt_n"]),
        "collapse_detail": collapse,
        "fog": float(fog["fog"]),
        "fog_detail": fog,
        "low_density_mass_fraction": float(fog["low_density_mass_fraction"]),
        "resolution_levels": int(fog["resolution_levels"]),
        "degenerate": bool(fog["degenerate"]),
    }


# --------------------------------------------------------------------------- #
# the train node — one fneg map per seed
# --------------------------------------------------------------------------- #


def _seed(job: Mapping[str, Any]) -> int:
    seed = job.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Round0265NodeError(f"R0265 job seed {seed!r} is not an integer")
    if seed not in SEEDS:
        raise Round0265NodeError(f"R0265 job seed {seed!r} is not a registered fneg cell")
    if str(job.get("capability") or "") != capability_for_seed(seed):
        raise Round0265NodeError("R0265 job capability does not match its seed")
    return int(seed)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0265 round0265_nodes.run_train")
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0265NodeError("R0265 train handler received another queue")
    seed = _seed(job)
    capability = capability_for_seed(seed)
    node_id = str(active.get("node_id") or f"train_seed{seed}")
    label = f"R0265 {capability}"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)

    closure_seal, closure = _closure_evidence(job)
    if not closure["controls"]["every_planted_defect_was_refused"]:
        raise Round0265TreatmentError(
            "R0265 closure guard did not refuse every planted defect: "
            f"{closure['controls']['controls']}"
        )
    if not closure["controls"]["the_honest_closure_still_passes"]:
        raise Round0265TreatmentError("R0265 closure guard rejects the honest closure")

    graph = _sealed_graph(job)
    edges = graph["directed_edges"]
    source, substrate_signature = _open_substrate(graph)
    config, config_sha = fneg_train_config(
        seed=seed,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        substrate_signature=substrate_signature,
        graph_edges=edges,
        rows=ROWS,
    )
    recipe = assert_registered_recipe(config)
    observed_invariant = fneg_seed_invariant_sha256(config)
    declared_invariant = str(job.get("family_seed_invariant_sha256") or "")
    if not declared_invariant or observed_invariant != declared_invariant:
        raise Round0265NodeError(
            "R0265 cell config is not the sealed fneg recipe: "
            f"{observed_invariant} != {declared_invariant}"
        )
    updates = int(config["optimizer"]["successful_positive_lr_updates"])
    if updates != DOSE_MULTIPLIER * int(job.get("base_horizon", -1)):
        raise Round0265NodeError("R0265 horizon does not match the sealed x4 base horizon")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0265 train output")
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": FNEG_TRAIN_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "seed": seed,
            "capability": capability,
            "recipe": recipe,
            "seed_invariant_sha256": observed_invariant,
            "config": config,
            "config_sha256": config_sha,
        },
        immutable=True,
    )

    from basemap.round0217_minilm_2m_pipeline import (
        MiniLMHostFp32EndpointArray,
        MiniLMMixedTrainingInput,
    )

    dataset = MiniLMHostFp32EndpointArray(
        source, source_signature=substrate_signature, buffer_rows=BATCH_SIZE
    )
    wrapper = MiniLMMixedTrainingInput(dataset, graph, seed=seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = _build_fneg_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")

    window = ledger.window(f"R0265 {capability} train stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=True)
    watchdog = CellWatchdog()
    watchdog.start()
    started = time.monotonic()
    try:
        with guard_ctx:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor(f"R0265 {capability} stage entered")
            wrapped = window.wrap(recorder)
            model.abort_poll = wrapped
            try:
                model.fit(
                    wrapper,
                    low_memory=True,
                    verbose=False,
                    n_processes=6,
                    random_state=seed,
                    resample_negatives=False,
                    precomputed_edges_path=graph["signature"]["canonical_path"],
                    use_wandb=False,
                )
            finally:
                model.abort_poll = None
            wall = time.monotonic() - started
            wrapped("R0265 fit() returned")
            accounting = dict(model._train_stats)
            runtime = wrapper.runtime_stamp()
            fneg_telemetry = dict(model.fneg_telemetry) if model.fneg_telemetry else None
            model_path = os.path.join(output, "model.pt")
            from basemap.output_safety import atomic_build_new_file

            atomic_build_new_file(model_path, model.save, immutable=True)
            wrapped("R0265 checkpoint published")
            free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
            memory = {
                "device_total_bytes": int(total_bytes),
                "post_train_free_bytes": int(free_bytes),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            }
            del model, wrapper, dataset
            torch.cuda.empty_cache()
            gc.collect()
            wrapped("R0265 training objects released")

            from basemap.pumap.parametric_umap import ParametricUMAP

            reloaded = ParametricUMAP.load(model_path, device="cuda")
            checkpoint_fneg_roundtrip = (
                float(reloaded.fneg_weight) == FNEG_WEIGHT
                and float(reloaded.fneg_lo) == FNEG_LO
                and float(reloaded.fneg_hi) == FNEG_HI
            )
            if not checkpoint_fneg_roundtrip:
                raise Round0265NodeError("R0265 checkpoint did not round-trip the fneg params")
            wrapped("R0265 checkpoint reloaded")
            coordinates = _transform_in_chunks(reloaded, source, wrapped)
            validate_published_map(coordinates)
            coordinates_path = os.path.join(output, "coordinates.npy")
            atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
            coordinates_ordered_sha256 = ordered_array_sha256(coordinates)
            transform_rows_finite = int(np.isfinite(coordinates).all(axis=1).sum())
            del reloaded, coordinates
            torch.cuda.empty_cache()
            gc.collect()
            gate.finish(f"R0265 {capability} stage end")
        window.close()
        tail = _guard_tail_reported(guard_ctx, label=label)
        scored = _score_gate_without_raising(gate, tail, label=label)
        gaps = gap_report(recorder.records, arm=node_id)
    finally:
        watchdog_state = watchdog.stop()

    if watchdog_state["tripped"]:
        raise Round0265NodeError(
            f"R0265 seed-{seed} watchdog tripped: {watchdog_state['trip_reason']!r}"
        )
    if int(memory["peak_reserved_bytes"]) > DEVICE_BUDGET_BYTES:
        raise Round0265NodeError(f"R0265 seed-{seed} peak reserved bytes exceed the budget")
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0265NodeError(f"R0265 train peak RSS {peak_rss_gib:.2f} GiB exceeds {HOST_RSS_LIMIT_GIB} GiB")
    memory["peak_host_rss_gib"] = peak_rss_gib

    expected_rows = updates * BATCH_SIZE
    producer_delta = (
        int(runtime["host_prefetch_producer_batches"]) - int(runtime["host_prefetch_consumer_batches"])
    )
    weighted = _weighted_rejection_accounting_mismatch(runtime, producer_delta=producer_delta, updates=updates)
    coverage = ledger.receipt()
    receipt_body = {
        "schema": TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "capability": capability,
        "capabilities": [capability],
        "training_seed": seed,
        "is_a_family_cell": True,
        "release_sha": active["manifest"]["release_sha"],
        "abort_flag_precondition": abort_flag,
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "seed_invariant_sha256": observed_invariant,
        "recipe": recipe,
        "treatment_closure": closure["verdict"],
        "treatment_closure_controls": closure["controls"],
        "treatment_closure_seal": expected_input_signature(
            _bound_path(job, "treatment_closure", label="R0265 treatment closure seal")
        ),
        "model": expected_input_signature(model_path),
        "coordinates": expected_input_signature(coordinates_path),
        "coordinates_ordered_sha256": coordinates_ordered_sha256,
        "substrate": substrate_signature,
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "rows": ROWS,
        "dimension": DIMENSION,
        "directed_edges": edges,
        "optimizer_updates": updates,
        "base_horizon": int(job.get("base_horizon", -1)),
        "dose_multiplier": DOSE_MULTIPLIER,
        "consumed_positive_draws_per_edge": float(updates * POSITIVE_ROWS_PER_UPDATE / edges),
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "fneg_telemetry": fneg_telemetry,
        "train_wall_s": wall,
        "memory": memory,
        "memory_watchdog": watchdog_state,
        "gap_report": gaps,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "training_performed": True,
        "gate_registerable_here": False,
        "map_decision_made": False,
        # Every entry is the expression that decided it, not a literal.
        "train_checks": {
            "recipe_is_the_registered_fneg_recipe": (
                observed_invariant == declared_invariant
            ),
            "every_planted_closure_defect_was_refused": bool(
                closure["controls"]["every_planted_defect_was_refused"]
            ),
            "closure_ran_the_sealed_bytes": bool(
                closure["verdict"]["every_module_ran_the_sealed_bytes"]
            ),
            "fneg_reweighting_was_active": fneg_telemetry is not None,
            "checkpoint_round_trips_fneg_params": checkpoint_fneg_roundtrip,
            "all_2m_coordinates_finite": transform_rows_finite == ROWS,
            "weighted_rejection_accounting_closes": weighted is None,
            "watchdog_did_not_trip": not bool(watchdog_state["tripped"]),
            "zero_numerical_skips": (
                int(accounting.get("amp_overflow_skips", 0)) == 0
                and int(accounting.get("nonfinite_loss_skips", 0)) == 0
                and int(accounting.get("nonfinite_gradient_skips", 0)) == 0
            ),
        },
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "node": node_id,
    }
    _seal(output, "train-receipt.json", receipt_body)
    del source, graph
    gc.collect()
    print(json.dumps({
        "capability": capability,
        "seed": seed,
        "fneg_active": fneg_telemetry is not None,
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


# --------------------------------------------------------------------------- #
# the n = 13 panel
# --------------------------------------------------------------------------- #


def _authenticate_map(cell: Mapping[str, Any], sealed: Mapping[str, Any]) -> dict[str, Any]:
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Round0265NodeError(f"R0265 cell seed {seed!r} is not an integer")
    if seed not in SEEDS:
        raise Round0265NodeError(f"R0265 cell seed {seed!r} is not an fneg cell")
    capability = capability_for_seed(seed)
    if str(cell.get("capability") or "") != capability:
        raise Round0265NodeError(f"R0265 seed-{seed} cell capability changed")
    receipt_path, receipt_signature = _intra_queue_signature(
        cell["train_receipt"], label=f"R0265 seed-{seed} train receipt"
    )
    receipt = prompt_contract.read_sealed(receipt_path, label=f"R0265 seed-{seed} train receipt")
    train_checks = receipt.get("train_checks") or {}
    if (
        receipt.get("schema") != TRAIN_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != seed
        or receipt.get("training_performed") is not True
        or not train_checks
        or not all(bool(value) for value in train_checks.values())
    ):
        raise Round0265NodeError(f"R0265 seed-{seed} train receipt contract changed")
    if (
        dict(receipt.get("substrate") or {}) != dict(sealed["substrate_signature"])
        or dict(receipt.get("graph_manifest") or {}) != dict(sealed["manifest_signature"])
    ):
        raise Round0265NodeError(f"R0265 seed-{seed} was not trained on the panel's substrate")
    model_path = prompt_contract.verify_signature(receipt["model"], label=f"R0265 seed-{seed} map")
    return {
        "seed": seed,
        "capability": capability,
        "receipt": receipt,
        "receipt_signature": receipt_signature,
        "model_path": model_path,
    }


def _load_centroids(panel: Mapping[str, Any], centroid_ks: Sequence[int]):
    declared = dict(panel.get("centroids") or {})
    if set(declared) != {str(k) for k in centroid_ks}:
        raise Round0265NodeError("R0218 centroid vocabularies changed")
    centroids: dict[int, np.ndarray] = {}
    signatures: dict[str, Any] = {}
    for k in centroid_ks:
        signature = dict(declared[str(k)])
        path = prompt_contract.verify_signature(signature, label=f"R0218 purity centroids k{k}")
        array = np.load(path, allow_pickle=False)
        if array.shape != (k, DIMENSION) or array.dtype != np.dtype("float32"):
            raise Round0265NodeError(f"R0218 centroids k{k} geometry changed")
        centroids[k] = array
        signatures[str(k)] = signature
    return centroids, signatures


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0265 round0265_nodes.run_panel")
    import torch
    from basemap.panel_v2 import (
        hiD_reference_key,
        load_hiD_reference,
        reset_process_cuda_peak,
        sample_anchors,
        score_panel,
    )
    from basemap.pumap.parametric_umap import ParametricUMAP

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0265NodeError("R0265 panel handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0265NodeError("R0265 panel scoring requires CUDA")
    node_id = str(active.get("node_id") or PANEL_ACTION)
    label = "R0265 n=13 fneg panel"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)

    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    panel_evidence = prompt_contract.read_sealed(
        str(job["panel_evidence"]), label="R0218 MiniLM 2M frozen panel"
    )
    centroid_ks = [int(k) for k in job["centroid_ks"]]
    cfg = prompt_contract.panel_config()

    # The held-out truth (R0233 reserve): the 20k probe embeddings and their exact
    # cosine top-10 substrate row indices, bound by signature.
    probes = np.load(_bound_path(job, "heldout_probes", label="R0265 held-out probes"), allow_pickle=False)
    truth_top10 = np.load(_bound_path(job, "heldout_truth", label="R0265 held-out truth top10"), allow_pickle=False)
    probes = np.asarray(probes, dtype=np.float32)
    truth_top10 = np.asarray(truth_top10, dtype=np.int64)
    if probes.shape != (N_PROBES, DIMENSION) or truth_top10.shape != (N_PROBES, K_TRUE):
        raise Round0265NodeError("R0265 held-out truth geometry changed")

    cells_in = job.get("cells")
    if not isinstance(cells_in, list):
        raise Round0265NodeError("R0265 cell input matrix changed")
    authenticated = [_authenticate_map(cell, sealed) for cell in cells_in]
    if {entry["seed"] for entry in authenticated} != set(SEEDS):
        raise Round0265NodeError("R0265 thirteen-cell input matrix changed")
    invariants = {str(entry["receipt"]["seed_invariant_sha256"]) for entry in authenticated}
    if len(invariants) != 1:
        raise Round0265NodeError("R0265 pooled family is not one fneg recipe")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0265 n=13 panel")
    started = time.monotonic()
    reset_process_cuda_peak()

    window = ledger.window("R0265 n=13 panel scoring stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0265 panel stage entered")
        wrapped = window.wrap(recorder)

        centroids, centroid_signatures = _load_centroids(panel_evidence, centroid_ks)
        reference_signature = dict(panel_evidence["shared_high_d_reference"])
        reference_path = prompt_contract.verify_signature(reference_signature, label="R0218 shared high-D reference")
        reference = load_hiD_reference(reference_path)
        anchors = sample_anchors(ROWS, cfg)
        reference_identity = {
            "data_identity": {
                "kind": "ordered_array",
                "shape": [ROWS, DIMENSION],
                "dtype": np.dtype("<f4").str,
                "sha256": sealed["ordered_substrate_sha256"],
            },
            "convention": dict(round0218_nodes.REFERENCE_CONVENTION),
        }
        wrapped("R0265 frozen reference + centroids loaded")

        cells: dict[str, dict[str, Any]] = {}
        for entry in sorted(authenticated, key=lambda e: e["seed"]):
            seed = entry["seed"]
            model = ParametricUMAP.load(entry["model_path"], device="cuda")
            coordinates = np.asarray(model.transform(source, batch_size=FULL_TRANSFORM_BATCH), dtype=np.float32)
            if coordinates.shape != (ROWS, 2) or not np.isfinite(coordinates).all():
                raise Round0265NodeError(f"R0265 seed-{seed} transform is not a finite 2M map")
            panel = score_panel(
                source,
                coordinates,
                config=cfg,
                centroids_by_k=centroids,
                hiD_reference=reference,
                reference_identity=reference_identity,
                provenance={
                    "round_id": ROUND_ID,
                    "seed": seed,
                    "capability": entry["capability"],
                    "treatment": "fneg-x4-md000",
                },
            )
            purity_ratios = {"k256": float(panel["purity"]["k256"]), "k1024": float(panel["purity"]["k1024"])}
            placed = np.asarray(model.transform(probes, batch_size=FULL_TRANSFORM_BATCH), dtype=np.float32)
            scored_map = score_one_map(
                coordinates=coordinates,
                probes_placed=placed,
                truth_top10=truth_top10,
                purity_ratios=purity_ratios,
            )
            cells[str(seed)] = {
                "seed": seed,
                "capability": entry["capability"],
                "train_receipt": dict(entry["receipt_signature"]),
                "model": dict(entry["receipt"]["model"]),
                "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
                "metrics": scored_map,
                "panel_purity_numerators": panel.get("purity_numerators"),
            }
            del model, coordinates, placed
            torch.cuda.empty_cache()
            gc.collect()
            wrapped(f"R0265 seed-{seed} scored")
        gate.finish("R0265 panel stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    # The seed-42 cross-check: this round's seed-42 cell vs the sandbox fneg10 cell.
    cross_check = seed42_cross_check(cells["42"]["metrics"])

    metric_table = {
        str(seed): {
            "heldout_ffr": cells[str(seed)]["metrics"]["heldout_ffr"],
            "purity_fidelity_k256": cells[str(seed)]["metrics"]["purity_fidelity_k256"],
            "purity_fidelity_k1024": cells[str(seed)]["metrics"]["purity_fidelity_k1024"],
            "collapse": cells[str(seed)]["metrics"]["collapse"],
            "fog": cells[str(seed)]["metrics"]["fog"],
            "resolution_levels": cells[str(seed)]["metrics"]["resolution_levels"],
            "degenerate": cells[str(seed)]["metrics"]["degenerate"],
        }
        for seed in SEEDS
    }
    execution_checks = {
        "all_thirteen_cells_scored": set(cells) == {str(seed) for seed in SEEDS},
        "pooled_family_one_recipe_digest": len(invariants) == 1,
        "every_heldout_ffr_finite": all(
            math.isfinite(row["heldout_ffr"]) for row in metric_table.values()
        ),
        "every_collapse_finite": all(
            math.isfinite(row["collapse"]) for row in metric_table.values()
        ),
        "every_purity_ratio_positive": all(
            row["purity_fidelity_k256"] > 0 and row["purity_fidelity_k1024"] > 0
            for row in metric_table.values()
        ),
        "no_gate_registered_here": not job.get("gate_registerable_here", False),
        "seed42_cross_check_lands_metric_close": bool(cross_check["metric_close"]),
    }
    if not all(value for key, value in execution_checks.items() if key != "seed42_cross_check_lands_metric_close"):
        # The cross-check outcome is a MEASUREMENT reported either way; the rest are hard.
        raise Round0265NodeError(f"R0265 panel execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    coverage = ledger.receipt()
    body = {
        **_receipt_envelope(active["manifest"]),
        "schema": PANEL_SCHEMA,
        "capability": PANEL_CAPABILITY,
        "capabilities": [PANEL_CAPABILITY],
        "node": node_id,
        "abort_flag_precondition": abort_flag,
        "n": len(SEEDS),
        "seeds": list(SEEDS),
        "seed_invariant_sha256": sorted(invariants)[0],
        "panel_metric_table": metric_table,
        "cells": cells,
        "seed42_cross_check": cross_check,
        "reference": reference_signature,
        "centroids": centroid_signatures,
        "gated_metrics": list(GATED_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        "lineage": {
            "graph_manifest": dict(sealed["manifest_signature"]),
            "substrate": dict(sealed["substrate_signature"]),
            "ordered_substrate_sha256": sealed["ordered_substrate_sha256"],
        },
        "gap_report": gaps,
        "enforcement_poll_spacing": scored_gate,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gate_registered": False,
        "gate_registerable_here": False,
        "peak_host_rss_gib": peak_rss_gib,
        "performance": {"node_wall_s": time.monotonic() - started},
    }
    _seal(output, "fneg-family-panel-n13.json", body)
    print(json.dumps({
        "capability": PANEL_CAPABILITY,
        "n": len(SEEDS),
        "seed42_heldffr": cells["42"]["metrics"]["heldout_ffr"],
        "seed42_cross_check_metric_close": cross_check["metric_close"],
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))
    del source, reference, centroids
    gc.collect()


# --------------------------------------------------------------------------- #
# the seed-42 cross-check
# --------------------------------------------------------------------------- #


def seed42_cross_check(observed: Mapping[str, Any]) -> dict[str, Any]:
    """Compare this round's seed-42 cell against the sandbox fneg10 cell, metric by metric.

    NOT byte-equal: the sandbox cell trained under the round-0115 sandbox core and a
    different seeded held-out truth. The requirement is metric-close on held-out FFR,
    collapse and fog; the fog/collapse numbers should land near the sandbox's
    (heldFFR 0.411, collapse 1.062, fog 0.371). Reported either way — a near miss is a
    finding, not a round failure.
    """
    target = SEED42_CROSS_CHECK_TARGET
    comparisons: dict[str, Any] = {}
    for metric, obs_key, tgt_key in (
        ("heldout_ffr", "heldout_ffr", "heldout_ffr"),
        ("collapse", "collapse", "collapse"),
        ("fog", "fog", "low_density_mass_fraction"),
    ):
        obs = float(observed[obs_key])
        tgt = float(target[tgt_key])
        tol = float(SEED42_TOLERANCE[metric])
        delta = abs(obs - tgt)
        comparisons[metric] = {
            "observed": obs,
            "sandbox_target": tgt,
            "abs_delta": delta,
            "tolerance": tol,
            "within_tolerance": delta <= tol,
        }
    return {
        "target_cell": f"{target['arm']}@{target['rung']} (seed {target['seed']})",
        "source": target["source"],
        "comparisons": comparisons,
        "metric_close": all(item["within_tolerance"] for item in comparisons.values()),
        "why_not_byte_equal": (
            "the sandbox cell trained under the round-0115 sandbox core, not the "
            "merged round-0208 core, and against a different seeded 20k held-out truth. "
            "Metric-closeness validates the merged recipe reproduces the sandbox's map "
            "quality; byte-equality is neither expected nor required."
        ),
    }


# --------------------------------------------------------------------------- #
# the n = 13 gate: fit family floors, convert bands, gate on the instruments
# --------------------------------------------------------------------------- #


def _read_sealed_multiplier(path: str, n_key: str, side: str) -> float:
    """Read a calibrated MAD_n multiplier from R0234's plain-JSON calibration artifact."""
    with open(path, "r", encoding="utf-8") as handle:
        d = json.load(handle)
    return float(
        d["calibration"][n_key]["candidates"][ESTIMATOR][side]["calibrated_multiplier"]
    )


def fit_family_floors(
    *,
    heldout_ffr_values: Sequence[float],
    collapse_values: Sequence[float],
    fog_values: Sequence[float],
    k256_ratios: Sequence[float],
    k1024_ratios: Sequence[float],
    k1: float,
    k2: float,
) -> dict[str, Any]:
    """Fit EVERY registered instrument from the fneg family's OWN thirteen cells.

    Per-family calibration is registered law (owner-proxy 2026-08-14): no gate value is
    imported from a legacy family; the legacy families contribute SEMANTICS only.

    * held-out FFR / collapse: one-sided lower floor `median - k1*MAD_n` on the raw
      family values (`floor_at` / R0264's `_robust_floor` mirror).
    * fog: one-sided upper ceiling `median + k1*MAD_n` (`_robust_ceiling`).
    * k256: two-sided band on the UNFOLDED log-ratio with `k2` (R0263 semantics,
      `_fit_two_sided_k256_band` fed the family's OWN k256 log-ratios).
    * k1024: one-sided lower ratio floor with `k1` (R0260 semantics,
      `one_sided_ratio_floor` fed the family's OWN k1024 log-ratios) — fail on
      under-separation only; over-separation stays unregulated.

    `k1` is the one-sided 95/95 n=13 multiplier and drives every one-sided floor/ceiling;
    `k2` is the two-sided 95/95 n=13 multiplier and is used by NOTHING but the k256 band.
    """
    k1 = float(k1)
    k2 = float(k2)
    ffr = list(map(float, heldout_ffr_values))
    collapse = list(map(float, collapse_values))
    fog = list(map(float, fog_values))
    k256 = list(map(float, k256_ratios))
    k1024 = list(map(float, k1024_ratios))
    log_k256 = [math.log(r) for r in k256]
    log_k1024 = [math.log(r) for r in k1024]

    ffr_floor = floor_at(ESTIMATOR, ffr, k1)
    collapse_floor_r0234 = floor_at(ESTIMATOR, collapse, k1)
    collapse_floor_r0264 = _robust_floor(collapse, k1)
    fog_ceiling = _robust_ceiling(fog, k1)
    k256_band = _fit_two_sided_k256_band(ESTIMATOR, log_k256, k2)
    k1024_fit = one_sided_ratio_floor(ESTIMATOR, log_k1024, k1)
    k1024_floor = float(k1024_fit["ratio_floor"])
    return {
        "k1_one_sided": k1,
        "k2_two_sided": k2,
        "estimator": ESTIMATOR,
        "fitted_from_n_family_cells": len(ffr),
        "heldout_ffr": {
            "direction": "one_sided_lower_floor",
            "multiplier": k1,
            "median": _median(ffr),
            "madn": _madn(ffr),
            "floor": ffr_floor,
            "min": min(ffr),
            "max": max(ffr),
        },
        "collapse": {
            "direction": "one_sided_lower_floor",
            "multiplier": k1,
            "median": _median(collapse),
            "madn": _madn(collapse),
            "floor": collapse_floor_r0264,
            "min": min(collapse),
            "max": max(collapse),
        },
        "fog": {
            "direction": "one_sided_upper_ceiling_fail_closed",
            "multiplier": k1,
            "median": _median(fog),
            "madn": _madn(fog),
            "ceiling": fog_ceiling,
            "min": min(fog),
            "max": max(fog),
        },
        "purity_fidelity_k256": {
            "direction": "two_sided_ratio_band_on_unfolded_log_ratio",
            "multiplier": k2,
            "band": k256_band,
        },
        "purity_fidelity_k1024": {
            "direction": KIND_RATIO_FLOOR,
            "multiplier": k1,
            "floor": k1024_floor,
            "log_centre": float(k1024_fit["log_centre"]),
            "log_scale": float(k1024_fit["log_scale"]),
            "log_floor": float(k1024_fit["log_floor"]),
            "fails_only": "under_separation (over-separation unregulated)",
        },
        "collapse_floor_mirrors_agree": collapse_floor_r0234 == collapse_floor_r0264,
        "collapse_floor_round0234_floor_at": collapse_floor_r0234,
    }


def convert_provisional_bands(*, family_floors: Mapping[str, Any]) -> dict[str, Any]:
    """The collapse/fog per-family fit, published as the conversion of R0264's provisional.

    R0264 fitted its collapse floor / fog ceiling to a seven-arm NON-seed family and
    marked them provisional, owing a real one-recipe seed family (n >= 13). This IS that
    family, and the family-calibrated collapse floor / fog ceiling in `family_floors` ARE
    the conversion: fitted from these thirteen cells with the one-sided multiplier. R0264's
    provisional values are shown as the descriptive converted-FROM; they never gate.
    """
    calibrated_collapse = float(family_floors["collapse"]["floor"])
    calibrated_fog = float(family_floors["fog"]["ceiling"])
    return {
        "converted_from_descriptive": {
            "status": "PROVISIONAL (R0264, n=7 non-seed family) — descriptive-only",
            "collapse_floor": R0264_PROVISIONAL_COLLAPSE_FLOOR,
            "fog_ceiling": R0264_PROVISIONAL_FOG_CEILING,
        },
        "converted_to_registered": {
            "status": "family-calibrated (R0265, n=13 fneg seed family)",
            "multiplier": float(family_floors["collapse"]["multiplier"]),
            "collapse_floor": calibrated_collapse,
            "fog_ceiling": calibrated_fog,
        },
        "collapse_floor_delta": calibrated_collapse - R0264_PROVISIONAL_COLLAPSE_FLOOR,
        "fog_ceiling_delta": calibrated_fog - R0264_PROVISIONAL_FOG_CEILING,
        "this_is_the_healthy_seed_family_r0264_owed": True,
        "policy": (
            "the conversion IS the per-family fit; the provisional-to-family movement "
            "is a finding about the fneg treatment, published as a descriptive contrast, "
            "never entered into any verdict"
        ),
    }


def descriptive_cross_reference() -> dict[str, Any]:
    """The legacy cross-treatment values, published beside the fitted criteria.

    NEVER a gate input. These are the seed-matched contrasts the owner-proxy asked to be
    shown for the reviewer's eye: R0263's k256 band, R0260's k1024 floor, R0264's
    provisional collapse/fog bands — each fitted on a DIFFERENT treatment family.
    """
    return {
        "role": "descriptive_cross_treatment_reference_never_gates",
        "purity_fidelity_k256": {
            "source": "R0263 registered two-sided k256 band (n=29 exact-graph legacy)",
            "ratio_lower": R0263_LEGACY_K256_BAND["ratio_lower"],
            "ratio_upper": R0263_LEGACY_K256_BAND["ratio_upper"],
        },
        "purity_fidelity_k1024": {
            "source": "R0260 registered k1024 one-sided ratio floor (legacy)",
            "ratio_floor": float(R0260_LEGACY_K1024_FLOOR),
        },
        "collapse": {
            "source": "R0264 provisional collapse floor (n=7 non-seed family)",
            "floor": R0264_PROVISIONAL_COLLAPSE_FLOOR,
        },
        "fog": {
            "source": "R0264 provisional fog ceiling (n=7 non-seed family)",
            "ceiling": R0264_PROVISIONAL_FOG_CEILING,
        },
    }


def _median(values: Sequence[float]) -> float:
    import statistics

    return float(statistics.median(list(values)))


def _madn(values: Sequence[float]) -> float:
    import statistics

    numbers = list(map(float, values))
    centre = statistics.median(numbers)
    return float(MAD_CONSISTENCY * statistics.median([abs(v - centre) for v in numbers]))


def score_family_on_gate(
    *,
    cells: Mapping[str, Mapping[str, Any]],
    heldout_ffr_floor: float,
    collapse_floor: float,
    fog_ceiling: float,
    k256_band: Mapping[str, float],
    k1024_floor: float,
) -> dict[str, Any]:
    """Score every family cell on the FIVE family-fitted instruments and AND the verdicts.

    Every band/floor passed here is FITTED FROM THE FNEG FAMILY (per-family calibration is
    registered law); the legacy R0263/R0260/R0264 numbers contribute only the JUDGE
    mechanics (`_judge_k256_two_sided`, `_judge_one_sided_floor`, `_judge_collapse`,
    `_judge_fog`) and are never a value here.
    """
    rows: list[dict[str, Any]] = []
    for seed_key, metrics in sorted(cells.items(), key=lambda kv: int(kv[0])):
        ffr = float(metrics["heldout_ffr"])
        ffr_pass = ffr >= float(heldout_ffr_floor)
        collapse_verdict = _judge_collapse(float(metrics["collapse"]), float(collapse_floor))
        fog_result = {
            "fog": float(metrics["fog"]),
            "resolution_levels": int(metrics["resolution_levels"]),
            "degenerate": bool(metrics["degenerate"]),
            "peak_bin_count": metrics.get("fog_detail", {}).get("peak_bin_count") if isinstance(metrics.get("fog_detail"), Mapping) else None,
        }
        fog_verdict = _judge_fog(fog_result, float(fog_ceiling))
        # k256: two-sided band (R0263 judge mechanics on the FAMILY-fitted band).
        k256_verdict = _judge_k256_two_sided(float(metrics["purity_fidelity_k256"]), k256_band)
        # k1024: one-sided lower ratio floor (R0260 judge mechanics on the FAMILY-fitted
        # floor) — fail on under-separation only.
        k1024_verdict = _judge_one_sided_floor(float(metrics["purity_fidelity_k1024"]), float(k1024_floor))
        by_metric = {
            HELDOUT_FFR_METRIC: {"verdict": "PASS" if ffr_pass else "FAIL", "passes": ffr_pass, "value": ffr, "floor": float(heldout_ffr_floor), "margin": ffr - float(heldout_ffr_floor)},
            COLLAPSE_METRIC: collapse_verdict,
            FOG_METRIC: fog_verdict,
            PURITY_K256_METRIC: k256_verdict,
            PURITY_K1024_METRIC: k1024_verdict,
        }
        clears = all(bool(v["passes"]) for v in by_metric.values())
        rows.append({
            "cell_id": exact_cell_id(int(seed_key)),
            "seed": int(seed_key),
            "metrics": by_metric,
            "clears_every_gated_instrument": clears,
            "fog_not_measurable": fog_verdict["verdict"] == VERDICT_NOT_MEASURABLE,
        })
    cleared = [r for r in rows if r["clears_every_gated_instrument"]]
    return {
        "cells": rows,
        "cells_scored": len(rows),
        "cells_clearing_every_instrument": len(cleared),
        "instruments": {
            "heldout_ffr_floor": float(heldout_ffr_floor),
            "collapse_floor": float(collapse_floor),
            "fog_ceiling": float(fog_ceiling),
            "purity_k256_band": {"ratio_lower": float(k256_band["ratio_lower"]), "ratio_upper": float(k256_band["ratio_upper"])},
            "purity_k1024_floor": float(k1024_floor),
        },
        "every_instrument_is_family_fitted": True,
        "any_cell_fog_not_measurable": any(r["fog_not_measurable"] for r in rows),
    }


def _read_legacy_semantics(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read R0263/R0264's sealed registrations for their SEMANTICS and DESCRIPTIVE values.

    Per the owner-proxy ruling, no number in these registrations gates an R0265 map; they
    are read (a) to confirm the sidedness/direction/estimator this round reuses, and (b)
    to populate the descriptive cross-reference. The values read are cross-checked against
    the published legacy constants so a drifted artifact is a finding, not a silent gate.
    """
    r0263 = prompt_contract.read_sealed(
        _bound_path(job, "r0263_gate", label="R0263 two-sided k256 band"),
        label="R0263 two-sided k256 band",
    )
    if (
        r0263.get("capability") != R0263_GATE_CAPABILITY
        or r0263.get("schema") != R0263_GATE_SCHEMA
        or r0263.get("gate_registered") is not True
    ):
        raise Round0265NodeError("R0263 registered band contract changed")
    r0263_band = dict(r0263["two_sided_k256_band"])
    r0263_criterion = dict(r0263["registered_criteria"][PURITY_K256_METRIC])

    r0264 = prompt_contract.read_sealed(
        _bound_path(job, "r0264_gate", label="R0264 provisional collapse/fog bands"),
        label="R0264 provisional collapse/fog bands",
    )
    if (
        r0264.get("capability") != R0264_GATE_CAPABILITY
        or r0264.get("schema") != R0264_GATE_SCHEMA
        or r0264.get("gate_registered") is not True
    ):
        raise Round0265NodeError("R0264 registered bands contract changed")
    r0264_criteria = dict(r0264["registered_criteria"])
    return {
        # SEMANTICS this round reuses (direction/sidedness/estimator), not numbers.
        "k256_sidedness": str(r0263_criterion.get("kind")),
        "collapse_direction": str(r0264_criteria[COLLAPSE_METRIC].get("kind")),
        "fog_direction": str(r0264_criteria[FOG_METRIC].get("kind")),
        "fog_fails_closed_on_degeneracy": bool(r0264_criteria[FOG_METRIC]["fails_closed_on_degeneracy"]),
        # DESCRIPTIVE legacy values (display only) + drift cross-check.
        "legacy_k256_band": {
            "ratio_lower": float(r0263_band["ratio_lower"]),
            "ratio_upper": float(r0263_band["ratio_upper"]),
        },
        "legacy_collapse_floor": float(r0264_criteria[COLLAPSE_METRIC]["floor"]),
        "legacy_fog_ceiling": float(r0264_criteria[FOG_METRIC]["ceiling"]),
        "legacy_values_match_published_constants": (
            float(r0263_band["ratio_lower"]) == R0263_LEGACY_K256_BAND["ratio_lower"]
            and float(r0263_band["ratio_upper"]) == R0263_LEGACY_K256_BAND["ratio_upper"]
            and float(r0264_criteria[COLLAPSE_METRIC]["floor"]) == R0264_PROVISIONAL_COLLAPSE_FLOOR
            and float(r0264_criteria[FOG_METRIC]["ceiling"]) == R0264_PROVISIONAL_FOG_CEILING
        ),
    }


def run_gate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0265 round0265_nodes.run_gate")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0265NodeError("R0265 gate handler received another queue")
    node_id = str(active.get("node_id") or GATE_ACTION)
    label = "R0265 n=13 fneg family floor registration"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0265 gate")

    window = ledger.window("R0265 gate registration stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0265 gate stage entered")
        wrapped = window.wrap(recorder)

        # 1. the multiplier, read from R0234's sealed n=13 artifact (not hardcoded).
        r0234_path = _bound_path(job, "r0234_n13", label="R0234 sealed n=13 calibration")
        k_two_sided = _read_sealed_multiplier(r0234_path, "n13", "two_sided")
        k_one_sided = _read_sealed_multiplier(r0234_path, "n13", "one_sided")
        multiplier_checks = {
            "two_sided_reproduces_the_anchor": k_two_sided == SEALED_N13_TWO_SIDED_MULTIPLIER,
            "one_sided_reproduces_the_anchor": k_one_sided == SEALED_N13_ONE_SIDED_MULTIPLIER,
        }
        wrapped("R0265 n=13 multiplier read")

        # 2. the family panel.
        panel = prompt_contract.read_sealed(
            _bound_path(job, "panel_n13", label="R0265 sealed n=13 panel"),
            label="R0265 sealed n=13 panel",
        )
        if panel.get("capability") != PANEL_CAPABILITY or int(panel.get("n", -1)) != N_FAMILY:
            raise Round0265NodeError("R0265 panel contract changed")
        table = dict(panel["panel_metric_table"])
        if {int(seed) for seed in table} != set(SEEDS):
            raise Round0265NodeError("R0265 panel is not the thirteen fneg cells")
        heldout = [float(table[str(seed)]["heldout_ffr"]) for seed in SEEDS]
        collapse = [float(table[str(seed)]["collapse"]) for seed in SEEDS]
        fog = [float(table[str(seed)]["fog"]) for seed in SEEDS]
        k256_ratios = [float(table[str(seed)]["purity_fidelity_k256"]) for seed in SEEDS]
        k1024_ratios = [float(table[str(seed)]["purity_fidelity_k1024"]) for seed in SEEDS]
        wrapped("R0265 family panel read")

        # 3. FIT every instrument from the fneg family's OWN thirteen cells. k1 drives the
        # one-sided floors/ceiling; k2 drives NOTHING but the two-sided k256 band.
        family_floors = fit_family_floors(
            heldout_ffr_values=heldout,
            collapse_values=collapse,
            fog_values=fog,
            k256_ratios=k256_ratios,
            k1024_ratios=k1024_ratios,
            k1=k_one_sided,
            k2=k_two_sided,
        )
        conversion = convert_provisional_bands(family_floors=family_floors)
        wrapped("R0265 every instrument fitted from the fneg family")

        heldout_floor = float(family_floors["heldout_ffr"]["floor"])
        collapse_floor = float(family_floors["collapse"]["floor"])
        fog_ceiling = float(family_floors["fog"]["ceiling"])
        k256_band = dict(family_floors["purity_fidelity_k256"]["band"])
        k1024_floor = float(family_floors["purity_fidelity_k1024"]["floor"])

        # 4. the legacy semantics + descriptive cross-reference (READ ONLY, NEVER GATES).
        legacy = _read_legacy_semantics(job)
        cross_reference = descriptive_cross_reference()
        wrapped("R0265 legacy semantics read; descriptive cross-reference built")

        # 5. attainability beside every count. For MAD_n every defining cell can fail; the
        # one-sided floors use k1, the two-sided band uses k2.
        power = {
            metric: attainability(ESTIMATOR, n=N_FAMILY, multiplier=k_one_sided)
            for metric in (HELDOUT_FFR_METRIC, COLLAPSE_METRIC, FOG_METRIC, PURITY_K1024_METRIC)
        }
        power[PURITY_K256_METRIC] = attainability(ESTIMATOR, n=N_FAMILY, multiplier=k_two_sided)

        # 6. score the thirteen cells on the five FAMILY-FITTED instruments.
        scoring = score_family_on_gate(
            cells={str(seed): table[str(seed)] for seed in SEEDS},
            heldout_ffr_floor=heldout_floor,
            collapse_floor=collapse_floor,
            fog_ceiling=fog_ceiling,
            k256_band=k256_band,
            k1024_floor=k1024_floor,
        )
        wrapped("R0265 thirteen cells scored on the family-fitted gate")

        cross_check = dict(panel.get("seed42_cross_check") or {})
        gate.finish("R0265 gate stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    # No legacy value is a gate input: prove the fitted instruments differ from the
    # legacy constants (they are fitted on a different family) so a reviewer can see the
    # gate is not silently reading a legacy number.
    fitted_differs_from_legacy = {
        "k256_band_lower": k256_band["ratio_lower"] != R0263_LEGACY_K256_BAND["ratio_lower"],
        "k256_band_upper": k256_band["ratio_upper"] != R0263_LEGACY_K256_BAND["ratio_upper"],
        "k1024_floor": k1024_floor != float(R0260_LEGACY_K1024_FLOOR),
        "collapse_floor": collapse_floor != R0264_PROVISIONAL_COLLAPSE_FLOOR,
        "fog_ceiling": fog_ceiling != R0264_PROVISIONAL_FOG_CEILING,
    }
    execution_checks = {
        "multiplier_read_from_sealed_r0234": bool(multiplier_checks["two_sided_reproduces_the_anchor"]),
        "one_sided_multiplier_read_from_sealed_r0234": bool(multiplier_checks["one_sided_reproduces_the_anchor"]),
        "collapse_floor_mirrors_agree_bitwise": bool(family_floors["collapse_floor_mirrors_agree"]),
        "k256_band_is_two_sided": (k256_band["ratio_lower"] < k256_band["ratio_upper"]),
        "k256_band_fitted_from_family_not_legacy": bool(
            fitted_differs_from_legacy["k256_band_lower"]
            or fitted_differs_from_legacy["k256_band_upper"]
        ),
        "k1024_floor_fitted_from_family_not_legacy": bool(fitted_differs_from_legacy["k1024_floor"]),
        "collapse_fog_fitted_from_family_not_provisional": (
            fitted_differs_from_legacy["collapse_floor"]
            and fitted_differs_from_legacy["fog_ceiling"]
        ),
        "fog_gate_fails_closed": bool(legacy["fog_fails_closed_on_degeneracy"]),
        "legacy_artifacts_match_published_constants": bool(legacy["legacy_values_match_published_constants"]),
        "every_defining_cell_can_fail": bool(
            power[HELDOUT_FFR_METRIC]["every_defining_cell_can_fail"]
        ),
        "count_is_attainable": bool(
            power[COLLAPSE_METRIC]["every_defining_cell_can_fail"]
        ),
        "five_instruments_all_family_fitted": bool(scoring["every_instrument_is_family_fitted"]),
        "density_v2_v3_are_not_gated": not (set(DESCRIPTIVE_METRICS) & set(GATED_METRICS)),
        "thirteen_cells_scored": scoring["cells_scored"] == N_FAMILY,
        "conversion_registers_family_calibrated_bands": bool(
            conversion["this_is_the_healthy_seed_family_r0264_owed"]
        ),
    }
    if not all(execution_checks.values()):
        raise Round0265NodeError(f"R0265 gate execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    coverage = ledger.receipt()
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": GATE_SCHEMA,
        "capability": GATE_CAPABILITY,
        "capabilities": [GATE_CAPABILITY],
        "node": node_id,
        "abort_flag_precondition": abort_flag,
        "n": N_FAMILY,
        "estimator": ESTIMATOR,
        "identity_bound_at_n": identity_bound(N_FAMILY),
        "multiplier": {
            "k1_one_sided_95_95": k_one_sided,
            "k2_two_sided_95_95": k_two_sided,
            "one_sided_source": R0234_N13_ONE_SIDED_KEY,
            "two_sided_source": R0234_N13_TWO_SIDED_KEY,
            "reproduction": multiplier_checks,
            "note": (
                f"k1 ({k_one_sided}) drives every one-sided floor/ceiling (held-out FFR, "
                f"collapse, fog, k1024); k2 ({k_two_sided}) drives NOTHING but the "
                "two-sided k256 band. Both read from R0234's sealed n=13 artifact."
            ),
        },
        "family_floors": family_floors,
        "band_conversion": conversion,
        "registered_criteria": {
            HELDOUT_FFR_METRIC: {
                "kind": "one_sided_lower_floor",
                "floor": heldout_floor,
                "multiplier": k_one_sided,
                "statistic": "heldout_ffr_at_0.1pct",
                "gates_a_map": True,
                "fitted_from": "the fneg family's own 13 held-out FFR values",
            },
            COLLAPSE_METRIC: {
                "kind": "one_sided_lower_floor",
                "floor": collapse_floor,
                "multiplier": k_one_sided,
                "statistic": "r10_over_radius_times_sqrt_n",
                "gates_a_map": True,
                "fitted_from": "the fneg family's own 13 collapse values",
            },
            FOG_METRIC: {
                "kind": "one_sided_upper_ceiling_fail_closed",
                "ceiling": fog_ceiling,
                "multiplier": k_one_sided,
                "statistic": "low_density_mass_fraction",
                "fails_closed_on_degeneracy": True,
                "gates_a_map": True,
                "fitted_from": "the fneg family's own 13 fog values",
            },
            PURITY_K256_METRIC: {
                "kind": "two_sided_ratio_band",
                "ratio_lower": k256_band["ratio_lower"],
                "ratio_upper": k256_band["ratio_upper"],
                "log_centre": k256_band["log_centre"],
                "log_scale": k256_band["log_scale"],
                "multiplier": k_two_sided,
                "gates_a_map": True,
                "semantics_from": R0263_GATE_CAPABILITY,
                "fitted_from": "the fneg family's own 13 k256 log-ratios",
            },
            PURITY_K1024_METRIC: {
                "kind": KIND_RATIO_FLOOR,
                "ratio_floor": k1024_floor,
                "multiplier": k_one_sided,
                "gates_a_map": True,
                "fails_only": "under_separation (over-separation unregulated)",
                "semantics_from": "R0260 one-sided ratio floor",
                "fitted_from": "the fneg family's own 13 k1024 log-ratios",
            },
        },
        "descriptive_cross_reference_never_gates": cross_reference,
        "legacy_semantics_read_only": legacy,
        "fitted_differs_from_legacy": fitted_differs_from_legacy,
        "attainability_and_power": power,
        "gate_scoring": scoring,
        "seed42_cross_check": cross_check,
        "instrument_provenance": {
            "every_gate_value_is_fitted_from_the_fneg_family": True,
            "legacy_families_contribute_semantics_only": {
                "k256_two_sided_sidedness": R0263_GATE_CAPABILITY,
                "k1024_one_sided_ratio_floor_sidedness": "R0260",
                "collapse_fog_directions_and_fail_closed": R0264_GATE_CAPABILITY,
                "estimator_and_multiplier": R0234_N13_TWO_SIDED_KEY,
            },
        },
        "gated_metrics": list(GATED_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        "gate_status": "registered-and-contingent-pending-review",
        "gate_registered": True,
        "evaluation_performed": True,
        "training_performed": False,
        "gpu_used": False,
        "upstream_review_state": dict(job.get("upstream_review_state") or {}),
        "execution_checks": execution_checks,
        "gap_report": gaps,
        "enforcement_poll_spacing": scored_gate,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "peak_host_rss_gib": peak_rss_gib,
        "wall_seconds": time.monotonic() - started,
    })
    _seal(output, "minilm-fneg-family-floors-n13.json", body)
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "n": N_FAMILY,
        "k1_one_sided": k_one_sided,
        "k2_two_sided": k_two_sided,
        "heldout_ffr_floor": heldout_floor,
        "collapse_floor": collapse_floor,
        "fog_ceiling": fog_ceiling,
        "k1024_floor": k1024_floor,
        "cells_clearing_every_instrument": scoring["cells_clearing_every_instrument"],
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0265 round0265_nodes.run_job")
    action = str(job.get("action") or "")
    if action == TRAIN_ACTION:
        run_train(active, job)
        return
    if action == PANEL_ACTION:
        run_panel(active, job)
        return
    if action == GATE_ACTION:
        run_gate(active, job)
        return
    raise Round0265NodeError(f"R0265 unknown action {action!r}")


__all__ = [
    "GATE_ACTION",
    "GATE_CAPABILITY",
    "GATE_SCHEMA",
    "PANEL_ACTION",
    "PANEL_CAPABILITY",
    "PANEL_SCHEMA",
    "TRAIN_ACTION",
    "TRAIN_SCHEMA",
    "convert_provisional_bands",
    "descriptive_cross_reference",
    "fit_family_floors",
    "heldout_ffr_scores",
    "map_collapse",
    "run_gate",
    "run_job",
    "run_panel",
    "run_train",
    "score_family_on_gate",
    "score_one_map",
    "seed42_cross_check",
]
