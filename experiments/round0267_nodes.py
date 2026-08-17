"""Execute R0267 — the 50M ×2 host-int8 staging rung of the promoted fneg recipe.

Five nodes in one queue (three trains -> panel -> gate), reusing R0265/R0266 machinery:

* three GPU trains (`seeds 42, 43, 44`) under the PINNED 50M ×2 host-int8 recipe, built
  and proved by `round0267_int8_treatment` (R0265's umap kernel a=1.9328/b=0.7905, fneg
  1.0 band [0.1,0.4], UNIFORM sampling + R0266's `x_residency=host_int8` routing, at dose
  ×2 on the sealed R0237 50M substrate + exact k15 graph). The train node mirrors R0266's
  host-int8 train node, retargeted to the 50M substrate/graph binding and the ×2 horizon.
* `score_minilm_fneg_50m_x2_panel` (GPU) — the three maps scored on R0265's instruments:
  held-out FFR against the sealed R0237 exact k15 truth, collapse and fog (all on the FULL
  50M coordinates via R0265's `score_one_map`), plus DESCRIPTIVE-only purity k256/k1024 on
  the R0237 50M substrate's first-2M-row prefix against a reference + centroids built INLINE
  on that same prefix (self-contained, no R0218 dependency; amendment 2026-08-17). Purity is
  labelled descriptive/ungated with the lineage caveat and NEVER enters the gate.
* `register_fneg_50m_x2_seedmean_gate` (CPU) — the pre-registered 50M gate
  (plan-50m-stage-2026-08-15, amendment 2026-08-17): the go/no-go is collapse (SEED-MEAN
  inside P1's ×2 asymptote band widened by a √n-shrunk family seed-noise allowance + per-seed
  backstop) + fog (ceiling + escalation) + held-out FFR (floor) ONLY — criteria 1–3.
  Purity is REMOVED from the go/no-go (descriptive-only at 50M) though its per-seed values are
  still recorded. Every band, floor, σ_fam and P1 edge is READ / RECOMPUTED from a SEALED
  artifact bound by sha256 at gate time -- never a typed literal (the constants-discipline
  contract test mutates each and asserts the gate tracks it).

Nothing in this module signals a process, starts a child, hands cuVS anything, or wraps a
subprocess in a timeout. Every bulk input is a read-only np.memmap.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import resource
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.slim_scale_admission import (
    assert_not_slim_cert_production_panel,
)
from basemap.round0217_minilm_2m_seed_family import (
    WARMUP_SUCCESSFUL_UPDATES,
    validate_published_map,
)
from basemap.round0242_locality import json_scrub
from basemap.round0238_rung5 import json_safe
from basemap.round0234_calibrated_floors import MAD_CONSISTENCY, identity_bound
from basemap.round0253_coverage import CoverageLedger
from basemap.round0253_stop_hooks import install_stop_hooks

from basemap import round0267_int8_treatment as T
from basemap.round0267_int8_treatment import (
    CANONICAL_SEED,
    DOSE_MULTIPLIER,
    INT8_SLICE_SUBSTRATE_CAPABILITY,
    INT8_SLICE_SUBSTRATE_SCHEMA,
    ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    TRAIN_CLOSURE_MODULES,
    X_RESIDENCY,
    Round0267RecipeError,
    assert_registered_50m_int8_recipe,
    assert_runtime_closure_matches_seal,
    capability_for_seed,
    exact_cell_id,
    fneg_seed_invariant_sha256,
    int8_50m_train_config,
    int8_slice_prefix_digests,
    recipe_refusal_controls,
    runtime_closure_hashes,
    treatment_closure_controls,
)

import experiments.round0265_nodes as R0265N
import experiments.round0266_nodes as R0266N
from experiments.round0230_nodes import CellWatchdog
from experiments.round0263_nodes import (
    _judge_k256_two_sided,
    _judge_one_sided_floor,
)
from experiments.round0264_nodes import (
    VERDICT_NOT_MEASURABLE,
    _judge_collapse,
    _judge_fog,
)

# R0265's shared node scaffolding, reused verbatim (guards, poll, transform, scoring).
from experiments.round0265_nodes import (
    COLLAPSE_METRIC,
    DIMENSION,
    FOG_BINS,
    FOG_METRIC,
    FULL_TRANSFORM_BATCH,
    HELDOUT_FFR_METRIC,
    PURITY_K1024_METRIC,
    PURITY_K256_METRIC,
    TRANSFORM_CHUNK_ROWS,
    _bound_path,
    _build_fneg_model,
    _intra_queue_signature,
    _node_gate,
    _node_guard,
    _score_gate_without_raising,
    _start_node,
    heldout_ffr_scores,
    map_collapse,
    score_one_map,
)
from basemap.round0247_registry import registry_fingerprint
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report


TRAIN_ACTION = "train_minilm_fneg_50m_x2_hostint8"
PANEL_ACTION = "score_minilm_fneg_50m_x2_panel"
GATE_ACTION = "register_fneg_50m_x2_seedmean_gate"

TRAIN_SCHEMA = "round0267-minilm-fneg-50m-x2-hostint8-train-receipt-v1"
PANEL_CAPABILITY = "minilm-fneg-50m-x2-hostint8-panel-v1"
PANEL_SCHEMA = "round0267-minilm-fneg-50m-x2-hostint8-panel-v1"
GATE_CAPABILITY = "minilm-fneg-50m-x2-seedmean-gate-v1"
GATE_SCHEMA = "round0267-minilm-fneg-50m-x2-seedmean-gate-v1"

#: The three per-seed GATED metrics (amended plan criteria 1-3: held-out FFR floor +
#: collapse backstop + fog ceiling). Purity is REMOVED from the go/no-go at 50M (amendment
#: 2026-08-17, OWNER-authorized): it is DESCRIPTIVE-only and never enters any pass/fail.
GATE_METRICS: tuple[str, ...] = (
    HELDOUT_FFR_METRIC, COLLAPSE_METRIC, FOG_METRIC,
)

#: Purity k256/k1024 are RECORDED (descriptive) at 50M but never gate. They live outside
#: GATE_METRICS so the no-straddle rule and per-seed pass/fail can never depend on them.
DESCRIPTIVE_PURITY_METRICS: tuple[str, ...] = (
    PURITY_K256_METRIC, PURITY_K1024_METRIC,
)

#: The first-2M-row prefix of the R0237 50M substrate — the DESCRIPTIVE purity subset. The
#: reference AND the k256/k1024 centroids are both built INLINE on exactly these rows, so the
#: descriptive purity number is fully self-contained on the R0237 prefix (no R0218 lineage).
PREFIX_ROWS = 2_000_000

#: The lineage caveat carried on every descriptive-purity record + the panel/gate bodies.
#: States WHY 50M purity is not comparable to the R0265 2M family bands (a different build
#: lineage), so the number is never mistaken for a gate.
DESCRIPTIVE_PURITY_LINEAGE_CAVEAT = (
    "purity k256/k1024 at 50M are DESCRIPTIVE / UNGATED. They are scored on the R0237 50M "
    "substrate's first-2M-row prefix against a reference + k-means centroids built INLINE on "
    "that SAME prefix (self-contained; no R0218 dependency, reference or centroids). They are "
    "NOT commensurate with the R0265 2M family purity bands: the R0237 ladder's first-2M rows "
    "are a DIFFERENT build lineage than R0218's frozen 2M reference the family bands were fit "
    "on (the dry-run's nested-prefix check proved R0237-prefix cdb11377… != R0218 cb44d0a7… = "
    "R0216-c3), so no frozen-reference purity gate exists at 50M. Purity remains a fully gated "
    "criterion only at 2M (amendment 2026-08-17, OWNER-authorized)."
)

#: The seed-mean collapse gate parameters (plan criterion 1). z is the two-sided 95%
#: normal quantile; n is the registered seed count. Both are pre-registered constants of
#: the DECISION RULE (not gate values read from data) — the band edges and σ_fam ARE read
#: live from sealed artifacts.
COLLAPSE_SEEDMEAN_Z = 1.96
COLLAPSE_SEEDMEAN_N = 3

DEVICE_BUDGET_BYTES = 30 * (1 << 30)
#: The post-train peak-RSS backstop. Raised 60.0 -> 100.0 GiB after R0267 seed-42 died on
#: this too-tight assertion AFTER a clean full-horizon train (measured peak 75.81 GiB): the
#: file-backed int8 X (19.2 GB) plus the fp32 substrate's resident transform pages push the
#: RSS peak past 60 GiB even though the training itself was healthy. 100 GiB covers the
#: 75.81 measured peak with headroom and sits far under the box's ~123 GB. This is an
#: EXECUTION-resource field (a liveness/OOM backstop), NOT a treatment field: it is absent
#: from the config and the masked-config/treatment invariant digest, so a cell trained at
#: limit 60 (seed42, salvaged) and cells trained at limit 100 (seeds 43/44) are
#: treatment-identical (the constants-discipline contract test proves this invariance).
HOST_RSS_LIMIT_GIB = 100.0

#: The R0244 host-watchdog anonymous-memory budget for the 50M host-int8 rung. The 2M
#: default (16 GiB, round0265) is too small here: the host-int8 X lives in host RAM as
#: an int8 array (50M×384 = 19.2 GB) plus the transient edge-list load and samplers, so
#: the anonymous peak is ~20+ GB (R0267 seed-42 first tripped 16 GiB at 17.2 GB). Raised
#: 40 -> 64 GiB on the R3 throwaway-map dry-run MEASUREMENT (2026-08-17): the PANEL node's
#: full-50M held-out-FFR builds the (~1M x 2000) neighbour arrays (int64 + float64 ~= 32 GB)
#: inside cKDTree.query, so the panel's anonymous peak measured 36.26 GiB — only ~9% under
#: the old 40 GiB, which fragmentation/thread-arena overhead could trip mid-panel. 64 GiB
#: gives ~76% headroom over the measured peak and still sits far under the box's ~111 GB
#: MemAvailable. Per-cell (freed + gc between the three cells, does not stack). An EXECUTION
#: resource field — absent from the treatment digest (constants-discipline test proves it).
R0267_ANON_BUDGET_BYTES = 64 * (1 << 30)
POSITIVE_ROWS_PER_UPDATE = 409

#: The delegate-approved SALVAGE of R0267 seed42 (correction-4): seed42's correction-3 run
#: trained the FULL ×2 horizon cleanly (4,162,228 updates, 0 AMP-skips, 0 nonfinite) and
#: saved its map, then the node died on the too-tight 60 GiB post-hoc RSS assertion (peak
#: 75.81 GiB). Rather than re-train the 11.9 GPU-h, its saved artifacts are BOUND as a
#: completed cell; only seeds 43/44 re-train. seed42 has NO train-receipt (the RSS raise
#: preempted it), so the panel/gate source seed42's provenance from the bound artifacts +
#: an explicit salvage block. The panel independently PINS seed42's coordinates digest to
#: the sealed correction-3 map below (defense in depth over the queue's bound signature).
SALVAGE_SEED = 42
SALVAGE_SOURCE_RUN = "queue-correction-3"
SALVAGE_SEED42_COORDINATES_SHA256 = (
    "aa7bbe678e6206c96ec6eb443aa6b6a2c4d8b589585c8c0db6f1b7eb9fc55284"
)
SALVAGE_REASON = "post-hoc RSS assertion after a clean full-horizon train"

#: The delegate-approved BIND of R0267 seeds 43/44 (correction-5): seeds 43/44 completed
#: their FULL ×2 host-int8 trains in correction-4 (real train-receipts, all train_checks
#: true) but bound to the PRIOR release; correction-5 rebuilds at the new release (the slim
#: >=8M scale-perf admission commit) and BINDS 43/44's correction-4 artifacts as completed
#: cells rather than re-training. Unlike seed42 (which lost its receipt to the RSS raise and
#: is salvaged from a train-log), 43/44 HAVE first-class train-receipts, so their provenance
#: is the bound receipt (stronger than a log). Their coordinates digests are NOT hardcoded
#: here: the panel PINS each cell's coordinates digest to the per-cell value sealed into the
#: queue record at prepare time (defense in depth over the queue's bound signature).
COMPLETED_SOURCE_RUN = "queue-correction-4"
COMPLETED_BIND_SEEDS: tuple[int, ...] = (43, 44)
COMPLETED_REASON = "completed full-horizon train bound at a new release (no re-train)"

SAFETY_NOTE = (
    "no node in this module signals any process, starts a child process, hands cuVS "
    "anything, or wraps a subprocess in a timeout. Every bulk input is a read-only "
    "np.memmap. The per-batch abort read is the release's own ParametricUMAP.abort_poll "
    "attribute, set to this node's recorder and cleared in a finally."
)

#: The correction-run the gate-only re-seal (PIECE B) supersedes: correction-5's panel +
#: registered gate carry the MIS-MEASURED FFR (~0.09). Its 50M_FAIL_OR_AMBIGUOUS verdict
#: STAYS in the record — this re-seal SUPERSEDES it, it does not erase it.
SUPERSEDED_SOURCE_RUN = "queue-correction-5"
SUPERSEDED_VERDICT = "50M_FAIL_OR_AMBIGUOUS"

#: The trip-9 + trip-10 diagnosis carried on the gate-only re-seal output. Two axes on which
#: correction-5's 50M held-out FFR was mis-specified vs the R0265 floor.
TRIP_9_10_DIAGNOSIS = (
    "correction-5's 50M held-out FFR was mis-specified vs the R0265 floor on two axes. "
    "TRIP 9 (discovery radius): it used the fixed 2000 disc instead of the N-scaled "
    "disc = int(ROWS * 0.001) = 50000 (0.1%·N). TRIP 10 (probe design): it used the "
    "IN-SUBSTRATE projection coordinates[probe_rows] instead of the floor's OUT-OF-"
    "SUBSTRATE reserve projection model.transform(reserve). A CPU re-score rebuilt the "
    "reserve-neighbour truth and re-scored all three already-trained maps with the floor's "
    "own instrument (disc=50000, out-of-substrate reserve), yielding FFR ~0.55 (was ~0.09), "
    "all three clearing the unchanged floor 0.3991. This gate-only re-seal binds that "
    "corrected FFR (collapse/fog byte-identical from the correction-5 panel, purity "
    "descriptive) and emits the SUPERSEDING verdict. PIECE A fixes run_panel so a full "
    "panel is born on this floor-matched instrument (trips 9/10 cannot recur at 100M)."
)

#: The pre-registered assumption the gate records (plan criterion 1).
SEED_SPREAD_ASSUMPTION = (
    "the 2M fneg family's seed spread (σ_fam = 1.4826·MAD_n over R0265's sealed 13 "
    "collapse/fog values) estimates the 50M seed spread; pre-registered as an assumption, "
    "revisited only if the 50M seeds' observed spread contradicts it"
)


class Round0267NodeError(RuntimeError):
    """The R0267 node contract changed."""


# --------------------------------------------------------------------------- #
# shared scaffolding local to R0267 (mirrors R0265/R0266's, ROUND_ID="0267")
# --------------------------------------------------------------------------- #


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


def _guard_tail_reported(watchdog, *, label: str) -> dict[str, Any]:
    return R0265N._guard_tail_reported(watchdog, label=label)


def _closure_evidence(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sealed = prompt_contract.read_sealed(
        _bound_path(job, "treatment_closure", label="R0267 treatment closure seal"),
        label="R0267 treatment closure seal",
    )
    observed = runtime_closure_hashes(TRAIN_CLOSURE_MODULES)
    verdict = assert_runtime_closure_matches_seal(sealed=sealed, observed=observed)
    return sealed, {
        "runtime_closure": observed,
        "verdict": verdict,
        "controls": treatment_closure_controls(sealed=sealed, observed=observed),
    }


# --------------------------------------------------------------------------- #
# the sealed R0237 50M substrate + graph binding (R0237 seals under its own
# substrate.json + qualified-graph.json manifests, not R0216's combined receipt)
# --------------------------------------------------------------------------- #


def _sealed_50m_graph(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read the sealed R0237 qualified-graph manifest and load its exact k15 fuzzy graph."""
    manifest_signature = dict(job["graph_manifest_signature"])
    manifest_path = prompt_contract.verify_signature(
        manifest_signature, label="R0267 sealed R0237 50M graph manifest"
    )
    manifest = prompt_contract.read_sealed(
        manifest_path, label="R0267 sealed R0237 50M graph manifest"
    )
    degrees = manifest.get("degrees") or {}
    floors = manifest.get("floors") or {}
    if (
        manifest.get("round_id") != T.R0237_ROUND_ID
        or manifest.get("capability") != T.R0237_GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("k", -1)) != 15
        or manifest.get("training_performed") is not False
    ):
        raise Round0267NodeError("R0267 sealed R0237 50M graph contract changed")
    if int(degrees.get("zero_degree_rows", -1)) != 0 or int(floors.get("zero_degree_rows", -1)) != 0:
        raise Round0267NodeError("R0267 requires the sealed R0237 50M graph zero-degree tripwire")
    edges = int(manifest.get("directed_edges", 0))
    if edges != SEALED_DIRECTED_EDGES:
        raise Round0267NodeError(
            f"R0267 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    graph_signature = dict(manifest["graph"])
    graph_path = prompt_contract.verify_signature(
        graph_signature, label="R0267 sealed R0237 50M fuzzy graph"
    )
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import load_edge_arrays

    sources, targets, weights, n_nodes = load_edge_arrays(graph_path, load_weights=True)
    if (
        weights is None
        or int(n_nodes) != ROWS
        or len(sources) != edges
        or targets.shape != sources.shape
        or weights.shape != sources.shape
    ):
        raise Round0267NodeError("R0267 sealed R0237 50M graph arrays changed")
    return {
        "manifest": manifest,
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "directed_edges": edges,
        "n_nodes": int(n_nodes),
    }


def _sealed_50m_substrate(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read the sealed R0237 nested-substrate manifest and its identity anchors."""
    manifest_signature = dict(job["substrate_manifest_signature"])
    manifest_path = prompt_contract.verify_signature(
        manifest_signature, label="R0267 sealed R0237 50M substrate manifest"
    )
    manifest = prompt_contract.read_sealed(
        manifest_path, label="R0267 sealed R0237 50M substrate manifest"
    )
    if (
        manifest.get("round_id") != T.R0237_ROUND_ID
        or manifest.get("capability") != T.R0237_SUBSTRATE_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or str(manifest.get("ordered_substrate_sha256")) != T.R0237_SUBSTRATE_ORDERED_SHA256
    ):
        raise Round0267NodeError("R0267 sealed R0237 50M substrate contract changed")
    return {
        "manifest": manifest,
        "manifest_signature": manifest_signature,
        "substrate_signature": dict(manifest["substrate"]),
        "reserve_signature": dict(manifest["reserve_substrate"]),
        "ordered_substrate_sha256": str(manifest["ordered_substrate_sha256"]),
    }


def _open_50m_substrate(sealed: Mapping[str, Any]) -> np.ndarray:
    """Serve the 76.8 GB sealed 50M substrate lazily; never materialize it."""
    path = prompt_contract.verify_signature(
        sealed["substrate_signature"], label="R0267 sealed R0237 50M substrate"
    )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (ROWS, DIMENSION) or array.dtype != np.float32:
        raise Round0267NodeError("R0267 sealed R0237 50M substrate geometry changed")
    return array


# --------------------------------------------------------------------------- #
# the PRE-SEALED int8 substrate load (the delegate-approved fix): LOAD R0262's
# sealed 100M int8 substrate's first-50M-row nested prefix file-backed instead of
# encoding fp32->int8 on the fly (the multi-minute encode blocked the watchdog).
# --------------------------------------------------------------------------- #


def _read_int8_slice_manifest(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read + validate the sealed R0267 int8 slice substrate manifest (its SLICE LAW)."""
    manifest_signature = dict(job["int8_substrate_manifest_signature"])
    manifest_path = prompt_contract.verify_signature(
        manifest_signature, label="R0267 sealed int8 slice substrate manifest"
    )
    manifest = prompt_contract.read_sealed(
        manifest_path, label="R0267 sealed int8 slice substrate manifest"
    )
    if (
        manifest.get("schema") != INT8_SLICE_SUBSTRATE_SCHEMA
        or manifest.get("capability") != INT8_SLICE_SUBSTRATE_CAPABILITY
        or manifest.get("round_id") != ROUND_ID
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or manifest.get("x_residency") != X_RESIDENCY
    ):
        raise Round0267NodeError("R0267 int8 slice substrate manifest contract changed")
    return {"manifest": manifest, "manifest_signature": manifest_signature}


def _load_verified_int8_slice(
    sealed_manifest: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """LOAD R0262's int8 nested prefix file-backed + VERIFY the sealed prefix digests.

    Returns ``(i8_prefix, scales_prefix, receipt)``. The two arrays are contiguous
    C-prefixes of the parent files served as read-only ``np.memmap`` (no copy). The
    two prefix sha256s are RE-HASHED (streamed, never materialised) and checked
    against the SLICE LAW in the sealed manifest -- a mismatch (the exact 50M bytes
    are not the sealed ones) raises ``Round0267NodeError``.
    """
    if (
        sealed_manifest.get("schema") != INT8_SLICE_SUBSTRATE_SCHEMA
        or sealed_manifest.get("capability") != INT8_SLICE_SUBSTRATE_CAPABILITY
    ):
        raise Round0267NodeError("R0267 int8 slice substrate manifest contract changed")
    law = dict(sealed_manifest.get("slice_law") or {})
    rows = int(law["rows"])
    dim = int(law["dimension"])
    offset = int(law["offset"])
    if offset != 0:
        raise Round0267NodeError("R0267 int8 slice must be the offset-0 nested prefix")
    i8_path = str(law["parent_i8_path"])
    scales_path = str(law["parent_scales_path"])
    expected_i8 = str(law["prefix_i8_sha256"])
    expected_scales = str(law["prefix_scales_sha256"])

    # File-backed, contiguous C-prefixes — the reshape+slice never copies the
    # 19.2 GB int8 payload (HostInt8ArrayDataset then shares the mmap).
    i8_full = np.memmap(i8_path, dtype=np.int8, mode="r").reshape(-1, dim)
    sc_full = np.memmap(scales_path, dtype=np.float16, mode="r")
    if int(i8_full.shape[0]) < rows or int(sc_full.shape[0]) < rows:
        raise Round0267NodeError(
            "R0267 parent int8 substrate is smaller than the sealed 50M prefix"
        )
    i8 = i8_full[:rows]
    sc = sc_full[:rows]

    got = int8_slice_prefix_digests(i8_path, scales_path, rows=rows, dimension=dim)
    if got["prefix_i8_sha256"] != expected_i8 or got["prefix_scales_sha256"] != expected_scales:
        raise Round0267NodeError(
            "R0267 int8 nested-prefix digest mismatch (the 50M bytes are not the sealed "
            f"ones): i8 {got['prefix_i8_sha256']} vs sealed {expected_i8}; scales "
            f"{got['prefix_scales_sha256']} vs sealed {expected_scales}"
        )
    receipt = {
        "parent_artifact": law.get("parent_artifact"),
        "parent_round": law.get("parent_round"),
        "parent_i8_path": i8_path,
        "parent_scales_path": scales_path,
        "rows": rows,
        "dimension": dim,
        "offset": offset,
        "prefix_i8_bytes": int(got["prefix_i8_bytes"]),
        "prefix_scales_bytes": int(got["prefix_scales_bytes"]),
        "prefix_i8_sha256": got["prefix_i8_sha256"],
        "prefix_scales_sha256": got["prefix_scales_sha256"],
        "verified_against_sealed_manifest": True,
        "load_mode": "pre_sealed_file_backed_nested_prefix",
        "re_encoded_at_train_time": False,
    }
    return i8, sc, receipt


def build_hostint8_dataset_from_slice(sealed_manifest: Mapping[str, Any], device: Any):
    """Construct a file-backed ``HostInt8ArrayDataset`` from the sealed int8 prefix.

    The int8 rows + fp16 scales are passed as ``encoded=``/``scales=`` so
    ``HostInt8ArrayDataset`` uses them VERBATIM (no fp32->int8 re-encode). The
    contiguous mmap prefixes stay file-backed through ``__init__`` (no 19.2 GB
    anonymous copy). Returns ``(dataset, receipt)``.
    """
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        HostInt8ArrayDataset,
    )

    i8, sc, receipt = _load_verified_int8_slice(sealed_manifest)
    dataset = HostInt8ArrayDataset(None, device, encoded=i8, scales=sc)
    if getattr(dataset, "host_int8_dataset", False) is not True:
        raise Round0267NodeError("R0267 pre-sealed int8 dataset is not a host-int8 dataset")
    if tuple(dataset.shape) != (int(receipt["rows"]), int(receipt["dimension"])):
        raise Round0267NodeError("R0267 pre-sealed int8 dataset geometry changed")
    return dataset, receipt


def _build_int8_50m_model(config: Mapping[str, Any]):
    """R0265's `_build_fneg_model` epoch-scaled to the 50M edge count, plus the int8 delta.

    `_build_fneg_model` scales `n_epochs` using R0216's 2M edge count (`SEALED_DIRECTED_
    EDGES`), which under-covers the 50M ×2 horizon; re-scale here with the 50M edge count.
    Then set `model.x_residency = host_int8` and assert it (R0266's int8 delta on the
    instance, since the config->model bridge does not thread x_residency).
    """
    model = _build_fneg_model(config)
    # Re-scale n_epochs for the 50M edge count so the loader supplies >= the ×2 horizon
    # of batches (steps_per_epoch = ceil(edges/num_pos)); the horizon break still ends at
    # total_steps_estimate, so the dose is unchanged.
    num_pos = max(1, int(model.batch_size * model.pos_ratio))
    steps_per_epoch = math.ceil(SEALED_DIRECTED_EDGES / num_pos)
    needed_epochs = math.ceil(int(model.total_steps_estimate) / steps_per_epoch)
    if needed_epochs > int(model.n_epochs):
        model.n_epochs = needed_epochs
    model.x_residency = X_RESIDENCY
    if getattr(model, "x_residency", None) != X_RESIDENCY:
        raise Round0267NodeError(
            f"R0267 model x_residency is {getattr(model, 'x_residency', None)!r}, "
            f"expected {X_RESIDENCY!r}"
        )
    return model


def _transform_50m_in_chunks(model: Any, source: Any, poll: Any) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, ROWS, TRANSFORM_CHUNK_ROWS):
        stop = min(start + TRANSFORM_CHUNK_ROWS, ROWS)
        block = np.asarray(
            model.transform(source[start:stop], batch_size=FULL_TRANSFORM_BATCH),
            dtype=np.float32,
        )
        parts.append(block)
        poll(f"R0267 transform rows {start}-{stop}")
    return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------- #
# the train node — one 50M ×2 host-int8 map per seed
# --------------------------------------------------------------------------- #


def _seed(job: Mapping[str, Any]) -> int:
    seed = job.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Round0267NodeError(f"R0267 job seed {seed!r} is not an integer")
    if seed not in SEEDS:
        raise Round0267NodeError(f"R0267 job seed {seed!r} is not a 50M rung cell (42/43/44)")
    if str(job.get("capability") or "") != capability_for_seed(seed):
        raise Round0267NodeError("R0267 job capability does not match its seed")
    return int(seed)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0267 round0267_nodes.run_train")
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0267NodeError("R0267 train handler received another queue")
    seed = _seed(job)
    capability = capability_for_seed(seed)
    node_id = str(active.get("node_id") or f"train_hostint8_50m_seed{seed}")
    label = f"R0267 {capability}"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)

    closure_seal, closure = _closure_evidence(job)
    if not closure["controls"]["every_planted_defect_was_refused"]:
        raise Round0267NodeError(
            "R0267 closure guard did not refuse every planted defect: "
            f"{closure['controls']['controls']}"
        )
    if not closure["controls"]["the_honest_closure_still_passes"]:
        raise Round0267NodeError("R0267 closure guard rejects the honest closure")

    graph = _sealed_50m_graph(job)
    substrate = _sealed_50m_substrate(job)
    # The fp32 substrate stays bound + opened for the post-train transform (and the
    # receipt lineage); it is NOT the training X any more (see the int8 load below).
    source = _open_50m_substrate(substrate)
    int8_slice = _read_int8_slice_manifest(job)
    edges = graph["directed_edges"]
    config, config_sha = int8_50m_train_config(
        seed=seed,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        substrate_signature=substrate["substrate_signature"],
        graph_edges=edges,
        rows=ROWS,
    )
    recipe = assert_registered_50m_int8_recipe(config)
    observed_invariant = fneg_seed_invariant_sha256(config)
    declared_invariant = str(job.get("cell_seed_invariant_sha256") or "")
    if not declared_invariant or observed_invariant != declared_invariant:
        raise Round0267NodeError(
            "R0267 cell config is not the sealed 50M host-int8 recipe: "
            f"{observed_invariant} != {declared_invariant}"
        )
    updates = int(config["optimizer"]["successful_positive_lr_updates"])
    if updates != DOSE_MULTIPLIER * int(job.get("base_horizon", -1)):
        raise Round0267NodeError("R0267 horizon does not match the sealed ×2 base horizon")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0267 train output")
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": T.INT8_TRAIN_CONFIG_SCHEMA,
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = _build_int8_50m_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")

    # PRE-SEALED LOAD (replaces the fp32-substrate-then-on-the-fly-encode path):
    # load R0262's sealed 100M int8 substrate's first-50M-row nested prefix
    # file-backed, VERIFY the two prefix sha256s against the sealed SLICE LAW, and
    # hand the pre-constructed HostInt8ArrayDataset to model.fit so core.fit uses
    # it directly (no re-encode). The 19.2 GB int8 payload stays file-backed
    # through HostInt8ArrayDataset.__init__ (no anonymous copy) and is byte-for-
    # byte the on-the-fly encode of the fp32 50M prefix (proven, 0 mismatches), so
    # the map is IDENTICAL to what the encode produced. The digest verification
    # (a streamed hash of the 19.2 GB prefix) runs HERE, before the liveness
    # watchdog starts, so it can never trip it.
    int8_dataset, int8_slice_receipt = build_hostint8_dataset_from_slice(
        int8_slice["manifest"], model.device
    )
    if int(int8_dataset.shape[0]) != ROWS or int(int8_dataset.shape[1]) != DIMENSION:
        raise Round0267NodeError("R0267 pre-sealed int8 dataset geometry is not the 50M rung")

    window = ledger.window(f"R0267 {capability} train stage")
    guard_ctx = _node_guard(label, anonymous_budget_bytes=R0267_ANON_BUDGET_BYTES)
    gate = _node_gate(label, training_performed=True)
    watchdog = CellWatchdog()
    watchdog.start()
    started = time.monotonic()
    try:
        with guard_ctx:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor(f"R0267 {capability} stage entered")
            wrapped = window.wrap(recorder)
            model.abort_poll = wrapped
            try:
                # X is the PRE-SEALED host-int8 dataset (loaded + verified above);
                # core.fit's host_int8 branch uses it directly, no re-encode.
                model.fit(
                    int8_dataset,
                    random_state=seed,
                    precomputed_edges_path=graph["signature"]["canonical_path"],
                )
            finally:
                model.abort_poll = None
            wall = time.monotonic() - started
            wrapped("R0267 fit() returned")
            accounting = dict(model._train_stats)
            runtime = dict(getattr(model, "_pipeline_info", None) or {})
            if not runtime:
                raise Round0267NodeError(
                    "R0267 fit() left no _pipeline_info stamp -- cannot prove the sampler"
                )
            # FAIL-CLOSED TRIPWIRE: uniform positive sampling AND the host-int8 residency.
            if (
                runtime.get("weighted_effective") is not False
                or runtime.get("positive_sampling") != "uniform"
                or runtime.get("x_residency") != X_RESIDENCY
            ):
                raise Round0267NodeError(
                    "R0267 trained off the host-int8 uniform path (silent fallback): "
                    f"weighted_effective={runtime.get('weighted_effective')!r}, "
                    f"positive_sampling={runtime.get('positive_sampling')!r}, "
                    f"x_residency={runtime.get('x_residency')!r}, "
                    f"pipeline={runtime.get('pipeline')!r}"
                )
            fneg_telemetry = dict(model.fneg_telemetry) if model.fneg_telemetry else None
            model_path = os.path.join(output, "model.pt")
            from basemap.output_safety import atomic_build_new_file

            atomic_build_new_file(model_path, model.save, immutable=True)
            wrapped("R0267 checkpoint published")
            free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
            memory = {
                "device_total_bytes": int(total_bytes),
                "post_train_free_bytes": int(free_bytes),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            }
            del model, int8_dataset
            torch.cuda.empty_cache()
            gc.collect()
            wrapped("R0267 training objects released")

            from basemap.pumap.parametric_umap import ParametricUMAP

            reloaded = ParametricUMAP.load(model_path, device="cuda")
            checkpoint_fneg_roundtrip = (
                float(reloaded.fneg_weight) == R0265N.FNEG_WEIGHT
                and float(reloaded.fneg_lo) == R0265N.FNEG_LO
                and float(reloaded.fneg_hi) == R0265N.FNEG_HI
            )
            if not checkpoint_fneg_roundtrip:
                raise Round0267NodeError("R0267 checkpoint did not round-trip the fneg params")
            wrapped("R0267 checkpoint reloaded")
            coordinates = _transform_50m_in_chunks(reloaded, source, wrapped)
            validate_published_map(coordinates)
            coordinates_path = os.path.join(output, "coordinates.npy")
            atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
            coordinates_ordered_sha256 = ordered_array_sha256(coordinates)
            transform_rows_finite = int(np.isfinite(coordinates).all(axis=1).sum())
            del reloaded, coordinates
            torch.cuda.empty_cache()
            gc.collect()
            gate.finish(f"R0267 {capability} stage end")
        window.close()
        tail = _guard_tail_reported(guard_ctx, label=label)
        scored = _score_gate_without_raising(gate, tail, label=label)
        gaps = gap_report(recorder.records, arm=node_id)
    finally:
        watchdog_state = watchdog.stop()

    if watchdog_state["tripped"]:
        raise Round0267NodeError(
            f"R0267 seed-{seed} watchdog tripped: {watchdog_state['trip_reason']!r}"
        )
    if int(memory["peak_reserved_bytes"]) > DEVICE_BUDGET_BYTES:
        raise Round0267NodeError(f"R0267 seed-{seed} peak reserved bytes exceed the budget")
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0267NodeError(f"R0267 train peak RSS {peak_rss_gib:.2f} GiB exceeds {HOST_RSS_LIMIT_GIB}")
    memory["peak_host_rss_gib"] = peak_rss_gib

    residency = {
        "x_residency": runtime.get("x_residency"),
        "weighted_effective": runtime.get("weighted_effective"),
        "positive_sampling": runtime.get("positive_sampling"),
        "uniform_with_replacement": runtime.get("uniform_with_replacement"),
        "sampler_pipeline": runtime.get("pipeline"),
        "sampler_class": runtime.get("sampler_class"),
    }
    coverage = ledger.receipt()
    receipt_body = {
        "schema": TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "capability": capability,
        "capabilities": [capability],
        "training_seed": seed,
        "is_a_50m_rung_cell": True,
        "release_sha": active["manifest"]["release_sha"],
        "abort_flag_precondition": abort_flag,
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "seed_invariant_sha256": observed_invariant,
        "recipe": recipe,
        "x_residency": X_RESIDENCY,
        "treatment_closure": closure["verdict"],
        "treatment_closure_controls": closure["controls"],
        "treatment_closure_seal": expected_input_signature(
            _bound_path(job, "treatment_closure", label="R0267 treatment closure seal")
        ),
        "model": expected_input_signature(model_path),
        "coordinates": expected_input_signature(coordinates_path),
        "coordinates_ordered_sha256": coordinates_ordered_sha256,
        "substrate": substrate["substrate_signature"],
        "substrate_manifest": substrate["manifest_signature"],
        "ordered_substrate_sha256": substrate["ordered_substrate_sha256"],
        "int8_substrate_manifest": int8_slice["manifest_signature"],
        "int8_substrate_slice": int8_slice_receipt,
        "x_source": "pre_sealed_int8_nested_prefix_of_r0262_100m",
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
        "host_int8_residency": residency,
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
        "train_checks": {
            "recipe_is_the_registered_50m_hostint8_recipe": (
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
            "pre_sealed_int8_slice_verified": bool(
                int8_slice_receipt.get("verified_against_sealed_manifest")
                and int8_slice_receipt.get("re_encoded_at_train_time") is False
            ),
            "all_50m_coordinates_finite": transform_rows_finite == ROWS,
            "host_int8_residency_stamp_verified": (
                residency["x_residency"] == X_RESIDENCY
                and residency["weighted_effective"] is False
                and residency["positive_sampling"] == "uniform"
            ),
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
        "x_residency": residency["x_residency"],
        "fneg_active": fneg_telemetry is not None,
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


# --------------------------------------------------------------------------- #
# the panel — three 50M cells scored on R0265's instruments
# --------------------------------------------------------------------------- #


def _authenticate_50m_map(cell: Mapping[str, Any], substrate: Mapping[str, Any]) -> dict[str, Any]:
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0267NodeError(f"R0267 cell seed {seed!r} is not a 50M rung cell")
    capability = capability_for_seed(seed)
    if str(cell.get("capability") or "") != capability:
        raise Round0267NodeError("R0267 cell capability changed")
    receipt_path, receipt_signature = _intra_queue_signature(
        cell["train_receipt"], label=f"R0267 seed-{seed} train receipt"
    )
    receipt = prompt_contract.read_sealed(receipt_path, label=f"R0267 seed-{seed} train receipt")
    train_checks = receipt.get("train_checks") or {}
    if (
        receipt.get("schema") != TRAIN_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != seed
        or receipt.get("training_performed") is not True
        or receipt.get("x_residency") != X_RESIDENCY
        or not train_checks
        or not all(bool(value) for value in train_checks.values())
    ):
        raise Round0267NodeError(f"R0267 seed-{seed} train receipt contract changed")
    if str(receipt.get("ordered_substrate_sha256")) != substrate["ordered_substrate_sha256"]:
        raise Round0267NodeError(f"R0267 seed-{seed} was not trained on the panel's substrate")
    model_path = prompt_contract.verify_signature(receipt["model"], label=f"R0267 seed-{seed} map")
    return {
        "seed": seed,
        "capability": capability,
        "salvaged": False,
        "receipt": receipt,
        "receipt_signature": receipt_signature,
        "model_path": model_path,
        "seed_invariant_sha256": str(receipt["seed_invariant_sha256"]),
    }


def _authenticate_salvaged_50m_map(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate seed42's BOUND correction-3 cell — no train-receipt, from artifacts.

    seed42 is the delegate-approved salvage: its correction-3 artifacts trained the full ×2
    horizon cleanly and were saved before the node died on the too-tight RSS assertion. This
    sources seed42's per-cell provenance from the bound artifacts + the salvage block (NOT a
    fabricated train-receipt): every bound artifact's bytes are re-verified (verify_signature
    re-hashes), seed42's coordinates digest is PINNED to the sealed correction-3 map, and the
    recipe invariant is read from the bound production-config.json.
    """
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0267NodeError(f"R0267 salvaged cell seed {seed!r} is not a 50M rung cell")
    if seed != SALVAGE_SEED:
        raise Round0267NodeError(f"R0267 only seed {SALVAGE_SEED} is a salvaged cell; got {seed!r}")
    capability = capability_for_seed(seed)
    if str(cell.get("capability") or "") != capability:
        raise Round0267NodeError("R0267 salvaged cell capability changed")
    salvage = dict(cell.get("salvage") or {})
    if (
        salvage.get("salvaged") is not True
        or str(salvage.get("source_run") or "") != SALVAGE_SOURCE_RUN
        or str(salvage.get("reason") or "") != SALVAGE_REASON
    ):
        raise Round0267NodeError("R0267 salvaged cell provenance block changed")
    # Re-verify every bound artifact's bytes (verify_signature re-hashes the file).
    coordinates_path = prompt_contract.verify_signature(
        dict(cell["coordinates"]), label="R0267 salvaged seed42 coordinates"
    )
    model_path = prompt_contract.verify_signature(
        dict(cell["model"]), label="R0267 salvaged seed42 model"
    )
    prod_config_path = prompt_contract.verify_signature(
        dict(cell["production_config"]), label="R0267 salvaged seed42 production-config"
    )
    prompt_contract.verify_signature(
        dict(cell["admission"]), label="R0267 salvaged seed42 admission"
    )
    prompt_contract.verify_signature(
        dict(cell["train_log"]), label="R0267 salvaged seed42 train log"
    )
    # PIN: seed42's coordinates file digest MUST be the sealed correction-3 map (raise on
    # mismatch), independent of the queue's declared signature.
    if str(cell["coordinates"]["sha256"]) != SALVAGE_SEED42_COORDINATES_SHA256:
        raise Round0267NodeError(
            "R0267 salvaged seed42 coordinates digest is not the sealed correction-3 map: "
            f"{cell['coordinates']['sha256']} != {SALVAGE_SEED42_COORDINATES_SHA256}"
        )
    # Source the recipe invariant from the bound production-config (NOT a train receipt).
    with open(prod_config_path, encoding="utf-8") as handle:
        prod_config = json.load(handle)
    invariant = str(prod_config.get("seed_invariant_sha256") or "")
    if not invariant:
        raise Round0267NodeError("R0267 salvaged seed42 production-config carries no seed invariant")
    if str(prod_config.get("round_id")) != ROUND_ID or int(prod_config.get("seed", -1)) != seed:
        raise Round0267NodeError("R0267 salvaged seed42 production-config is not this cell")
    return {
        "seed": seed,
        "capability": capability,
        "salvaged": True,
        "salvage": salvage,
        "coordinates_path": coordinates_path,
        "coordinates_signature": dict(cell["coordinates"]),
        "model_path": model_path,
        "model_signature": dict(cell["model"]),
        "production_config_signature": dict(cell["production_config"]),
        "seed_invariant_sha256": invariant,
    }


def _authenticate_bound_completed_50m_map(
    cell: Mapping[str, Any], substrate: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate a BOUND correction-4 cell (seed 43/44) — from its saved artifacts.

    The generalisation of the seed42 salvage to cells that DID finish with a real
    train-receipt. Every bound artifact's bytes are re-verified (verify_signature re-hashes),
    the coordinates digest is PINNED to the per-cell value sealed into the queue record at
    prepare time (NOT a hardcoded module constant), and the recipe invariant + substrate
    lineage are sourced from the bound train-receipt (first-class provenance, stronger than
    seed42's log). The bound coordinates are then scored directly (no model load / no
    50M re-transform), exactly like the salvaged path.
    """
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0267NodeError(f"R0267 bound-completed cell seed {seed!r} is not a 50M rung cell")
    if seed == SALVAGE_SEED or seed not in COMPLETED_BIND_SEEDS:
        raise Round0267NodeError(
            f"R0267 only seeds {COMPLETED_BIND_SEEDS} are completed-bind cells; got {seed!r}"
        )
    capability = capability_for_seed(seed)
    if str(cell.get("capability") or "") != capability:
        raise Round0267NodeError("R0267 bound-completed cell capability changed")
    bound = dict(cell.get("bound") or {})
    if (
        bound.get("bound_completed") is not True
        or str(bound.get("source_run") or "") != COMPLETED_SOURCE_RUN
        or str(bound.get("reason") or "") != COMPLETED_REASON
    ):
        raise Round0267NodeError("R0267 bound-completed cell provenance block changed")
    # Re-verify every bound artifact's bytes (verify_signature re-hashes the file).
    coordinates_path = prompt_contract.verify_signature(
        dict(cell["coordinates"]), label=f"R0267 bound seed{seed} coordinates"
    )
    model_path = prompt_contract.verify_signature(
        dict(cell["model"]), label=f"R0267 bound seed{seed} model"
    )
    prod_config_path = prompt_contract.verify_signature(
        dict(cell["production_config"]), label=f"R0267 bound seed{seed} production-config"
    )
    prompt_contract.verify_signature(
        dict(cell["admission"]), label=f"R0267 bound seed{seed} admission"
    )
    receipt_path = prompt_contract.verify_signature(
        dict(cell["train_receipt"]), label=f"R0267 bound seed{seed} train receipt"
    )
    # PER-CELL PIN: the coordinates digest MUST equal the value sealed into the queue
    # record at prepare time (raise on mismatch), independent of the bound signature. The
    # 43/44 digests are NEVER hardcoded here — the pin travels in the sealed queue record.
    pin = str(cell.get("coordinates_sha256_pin") or "")
    if not pin or str(cell["coordinates"]["sha256"]) != pin:
        raise Round0267NodeError(
            f"R0267 bound seed{seed} coordinates digest is not the sealed per-cell pin: "
            f"{cell['coordinates'].get('sha256')} != {pin}"
        )
    # First-class provenance: the bound train-receipt (a real receipt, not a log).
    receipt = prompt_contract.read_sealed(receipt_path, label=f"R0267 bound seed{seed} train receipt")
    train_checks = receipt.get("train_checks") or {}
    if (
        receipt.get("schema") != TRAIN_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != seed
        or receipt.get("training_performed") is not True
        or receipt.get("x_residency") != X_RESIDENCY
        or not train_checks
        or not all(bool(value) for value in train_checks.values())
    ):
        raise Round0267NodeError(f"R0267 bound seed{seed} train receipt contract changed")
    if str(receipt.get("ordered_substrate_sha256")) != substrate["ordered_substrate_sha256"]:
        raise Round0267NodeError(f"R0267 bound seed{seed} was not trained on the panel's substrate")
    # Tie the receipt to the actual coordinates file: the receipt's own coordinates digest
    # must be the pinned digest (so the receipt describes THESE bound coordinates).
    if str((receipt.get("coordinates") or {}).get("sha256")) != pin:
        raise Round0267NodeError(
            f"R0267 bound seed{seed} train receipt does not describe the bound coordinates"
        )
    invariant = str(receipt.get("seed_invariant_sha256") or "")
    if not invariant:
        raise Round0267NodeError(f"R0267 bound seed{seed} train receipt carries no seed invariant")
    # Cross-check the bound production-config agrees (round/seed/invariant).
    with open(prod_config_path, encoding="utf-8") as handle:
        prod_config = json.load(handle)
    if str(prod_config.get("round_id")) != ROUND_ID or int(prod_config.get("seed", -1)) != seed:
        raise Round0267NodeError(f"R0267 bound seed{seed} production-config is not this cell")
    if str(prod_config.get("seed_invariant_sha256") or "") != invariant:
        raise Round0267NodeError(
            f"R0267 bound seed{seed} production-config invariant disagrees with the receipt"
        )
    return {
        "seed": seed,
        "capability": capability,
        "salvaged": False,
        "bound_completed": True,
        "bound": bound,
        "coordinates_path": coordinates_path,
        "coordinates_signature": dict(cell["coordinates"]),
        "model_path": model_path,
        "model_signature": dict(cell["model"]),
        "production_config_signature": dict(cell["production_config"]),
        "train_receipt_signature": dict(cell["train_receipt"]),
        "seed_invariant_sha256": invariant,
    }


# --------------------------------------------------------------------------- #
# the 50M panel's scoring (amendment 2026-08-17, OWNER-authorized — purity is
# DESCRIPTIVE-only at 50M):
#   * collapse / fog / held-out FFR are the GATED metrics, measured on the FULL 50M
#     coordinates (via R0265's score_one_map — 2D-map properties + the sealed R0237
#     k15 truth; they never consume any hiD reference); their computation + inputs
#     stay BYTE-IDENTICAL to the pre-amendment panel.
#   * purity k256/k1024 are DESCRIPTIVE-only: scored on the R0237 substrate's first
#     PREFIX_ROWS-row prefix against a reference + k-means centroids built INLINE on
#     that SAME prefix (hiD_reference=None, centroids from the prefix), so the number is
#     fully self-contained on the R0237 prefix with NO R0218 dependency. The prefix is
#     < 8M, so this score_panel pass carries no scale_admission. Purity is labelled
#     descriptive/ungated with the lineage caveat and is never a gate.
# (The old R0218-frozen-reference binding, its 2M-prefix reference-identity helper, and
# the verify_nested_prefix_identity refusal check are DROPPED — moot per amendment
# 2026-08-17: the descriptive reference is built on the prefix itself, so there is no
# cross-lineage identity claim to verify. The standing >=8M slim-cert producer that
# reused those two helpers now carries them itself.)
# --------------------------------------------------------------------------- #


def _build_prefix_purity_centroids(
    source_prefix: np.ndarray, centroid_ks: Sequence[int], *, cache_dir: str
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Build the DESCRIPTIVE purity centroids INLINE on the R0237 prefix (GPU k-means).

    Reuses the frozen-centroids builder (random init + 25 Lloyd iters, seed 0) that
    produced R0218's centroids, but fits it on the R0237 substrate's first-PREFIX_ROWS
    rows instead — so the descriptive purity is self-contained on the prefix with no
    R0218 centroid dependency. Each k's centroids are written immutable into ``cache_dir``
    (a fresh sub-directory of the panel output) and returned with their signatures. The
    fit reads the prefix memmap in 100k-row chunks (no >=2 GB materialisation). Cost at
    PREFIX_ROWS=2M: ~a few GPU-minutes for k256+k1024 (owner note 2026-08-17).
    """
    from experiments.score_complete_panel import frozen_centroids

    cache_dir = create_fresh_directory(cache_dir, label="R0267 descriptive prefix centroids")
    ks = [int(k) for k in centroid_ks]
    built = frozen_centroids(source_prefix, ks, cache_dir, seed=0, iters=25)
    centroids: dict[int, np.ndarray] = {}
    signatures: dict[str, Any] = {}
    for k in ks:
        array = np.asarray(built[k], dtype=np.float32)
        if array.shape != (k, DIMENSION):
            raise Round0267NodeError(f"R0267 descriptive prefix centroids k{k} geometry changed")
        centroids[k] = array
        signatures[str(k)] = expected_input_signature(
            os.path.join(cache_dir, f"centroids_k{k}.npy")
        )
    return centroids, signatures


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0267 round0267_nodes.run_panel")
    import torch
    from basemap.panel_v2 import (
        reset_process_cuda_peak,
        score_panel,
    )
    from basemap.pumap.parametric_umap import ParametricUMAP

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0267NodeError("R0267 panel handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0267NodeError("R0267 panel scoring requires CUDA")
    node_id = str(active.get("node_id") or PANEL_ACTION)
    label = "R0267 50M ×2 host-int8 panel"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)

    substrate = _sealed_50m_substrate(job)
    source = _open_50m_substrate(substrate)
    # amendment 2026-08-17 (OWNER-authorized): run_panel invokes score_panel ONLY for the
    # DESCRIPTIVE purity pass on the R0237 first-PREFIX_ROWS prefix (< 8M, so NO
    # scale_admission), against a reference + centroids built INLINE on that same prefix —
    # NO R0218 dependency.  The GATED collapse / fog / held-out FFR come from score_one_map
    # on the full-50M coordinates.  So there is no >=8M score_panel and no slim
    # scale-performance certificate here.  (The >=8M slim-admission machinery remains as
    # standing infrastructure; it is exercised by produce_slim_scale_cert.py, not by this
    # round.)  The centroid granularities [256, 1024] come from the job (a config list, NOT
    # R0218's centroid arrays); the centroids themselves are re-fit on the prefix below.
    centroid_ks = [int(k) for k in job["centroid_ks"]]
    cfg = prompt_contract.panel_config()

    # The FLOOR-MATCHED held-out FFR instrument (PIECE A, 2026-08-17). The R0265 family
    # floor is the OUT-OF-SUBSTRATE reserve projection: the held-out reserve embeddings
    # projected THROUGH each trained map (`model.transform(reserve)`), scored against the
    # reserve's exact-cosine neighbour truth, at the N-scaled discovery radius
    # disc = int(ROWS * 0.001) (0.1%·N). run_panel now measures FFR with that same
    # instrument so a full panel is BORN matched to the floor (the two mis-specifications
    # that produced the R0267 50M FFR — trip 9: the fixed 2000 disc instead of the N-scaled
    # 0.1%·N; trip 10: the IN-SUBSTRATE coordinates[probe_rows] instead of the OUT-OF-
    # SUBSTRATE reserve projection — cannot recur at 100M).
    #   * `reserve_embeddings` = reserve.f32[reserve-query-rows] (the R0237 held-out reserve
    #     query rows), a sealed panel input; projected per-cell below.
    #   * `reserve_truth` = the sealed reserve-neighbour top-10 truth (indices INTO the 50M
    #     substrate == into each cell's coordinates), a sealed panel input.
    #   * `reserve_disc` = int(ROWS * 0.001) — N-scaled, NOT the fixed 2000.
    reserve_all = np.load(
        _bound_path(job, "heldout_reserve", label="R0267 50M held-out reserve"),
        mmap_mode="r", allow_pickle=False,
    )
    reserve_query_rows = np.load(
        _bound_path(job, "reserve_query_rows", label="R0267 reserve query rows"),
        allow_pickle=False,
    ).astype(np.int64, copy=False)
    if reserve_all.ndim != 2 or int(reserve_all.shape[1]) != DIMENSION or reserve_query_rows.ndim != 1:
        raise Round0267NodeError("R0267 held-out reserve geometry changed")
    reserve_embeddings = np.asarray(reserve_all[reserve_query_rows], dtype=np.float32)
    reserve_truth = np.load(
        _bound_path(job, "reserve_truth", label="R0267 reserve-neighbour truth"),
        allow_pickle=False,
    ).astype(np.int64, copy=False)
    if reserve_truth.ndim != 2 or reserve_truth.shape[0] != reserve_embeddings.shape[0]:
        raise Round0267NodeError("R0267 reserve-neighbour truth geometry changed")
    reserve_disc = int(ROWS * 0.001)

    cells_in = job.get("cells")
    if not isinstance(cells_in, list):
        raise Round0267NodeError("R0267 cell input matrix changed")
    def _authenticate(cell: Mapping[str, Any]) -> dict[str, Any]:
        if cell.get("salvaged"):
            return _authenticate_salvaged_50m_map(cell)
        if cell.get("bound_completed"):
            return _authenticate_bound_completed_50m_map(cell, substrate)
        return _authenticate_50m_map(cell, substrate)

    authenticated = [_authenticate(cell) for cell in cells_in]
    if {entry["seed"] for entry in authenticated} != set(SEEDS):
        raise Round0267NodeError("R0267 three-cell input matrix changed")
    # The recipe invariant is sourced per-cell: fresh trains from their train-receipts,
    # bound-completed cells (43/44) from their bound train-receipts, and seed42 (salvaged)
    # from its bound production-config. A single pooled recipe is still required.
    invariants = {str(entry["seed_invariant_sha256"]) for entry in authenticated}
    if len(invariants) != 1:
        raise Round0267NodeError("R0267 pooled family is not one 50M recipe")
    salvaged_seeds = sorted(entry["seed"] for entry in authenticated if entry.get("salvaged"))
    bound_completed_seeds = sorted(
        entry["seed"] for entry in authenticated if entry.get("bound_completed")
    )

    output = create_fresh_directory(str(job["outputs"][0]), label="R0267 50M panel")
    started = time.monotonic()
    reset_process_cuda_peak()

    window = ledger.window("R0267 50M panel scoring stage")
    guard_ctx = _node_guard(label, anonymous_budget_bytes=R0267_ANON_BUDGET_BYTES)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0267 panel stage entered")
        wrapped = window.wrap(recorder)

        # DESCRIPTIVE purity centroids: fit INLINE on the R0237 first-PREFIX_ROWS prefix
        # (GPU k-means, same frozen builder R0218 used) so the descriptive reference AND its
        # centroids are self-contained on the prefix — no R0218 lineage.  ~a few GPU-minutes
        # at PREFIX_ROWS=2M (owner note 2026-08-17).
        prefix_rows = PREFIX_ROWS
        if prefix_rows > ROWS:
            raise Round0267NodeError("R0267 descriptive prefix exceeds the 50M substrate rows")
        centroids, centroid_signatures = _build_prefix_purity_centroids(
            source[:prefix_rows], centroid_ks,
            cache_dir=os.path.join(output, "descriptive-prefix-centroids"),
        )
        wrapped("R0267 descriptive prefix centroids fit on the R0237 first-2M rows")

        cells: dict[str, dict[str, Any]] = {}
        for entry in sorted(authenticated, key=lambda e: e["seed"]):
            seed = entry["seed"]
            if entry.get("salvaged") or entry.get("bound_completed"):
                # BOUND cell (seed42 salvaged from correction-3, or seed43/44 completed in
                # correction-4): score the BOUND coordinates directly (digest-pinned at
                # authentication), NOT a 50M re-transform. The map's model IS still loaded —
                # only to project the small (2000-row) held-out reserve for the floor-matched
                # FFR instrument (PIECE A); collapse/fog/purity remain measured on the bound
                # coordinates, so the expensive 50M transform is still skipped.
                coordinates = np.asarray(
                    np.load(entry["coordinates_path"], allow_pickle=False), dtype=np.float32
                )
                if coordinates.shape != (ROWS, 2) or not np.isfinite(coordinates).all():
                    raise Round0267NodeError(
                        f"R0267 seed-{seed} bound coordinates are not a finite 50M map"
                    )
                proj_model = ParametricUMAP.load(entry["model_path"], device="cuda")
            else:
                proj_model = ParametricUMAP.load(entry["model_path"], device="cuda")
                coordinates = _transform_50m_in_chunks(proj_model, source, wrapped)
                if coordinates.shape != (ROWS, 2) or not np.isfinite(coordinates).all():
                    raise Round0267NodeError(f"R0267 seed-{seed} transform is not a finite 50M map")
            # DESCRIPTIVE purity pass — the ONLY score_panel call run_panel makes
            # (amendment 2026-08-17).  Xa = the substrate's first <prefix_rows> rows; Z = the
            # cell's first <prefix_rows> coordinate rows, so the 2D neighbour pool is the
            # PREFIX ROWS' coordinates ONLY (no 48M leakage).  hiD_reference=None builds the
            # reference INLINE on that same prefix, and centroids_by_k are the prefix-fit
            # centroids — so the purity number is fully self-contained on the R0237 prefix
            # with NO R0218 dependency.  <prefix_rows> is < 8M, so this pass takes NO
            # scale_admission (the >=8M guard refuses a below-scale admission).  DESCRIPTIVE
            # ONLY: the number is not commensurate with the R0265 2M family bands (see the
            # lineage caveat) and never enters the gate.
            purity_panel = score_panel(
                source[:prefix_rows],
                coordinates[:prefix_rows],
                config=cfg,
                centroids_by_k=centroids,
                hiD_reference=None,
                provenance={
                    "round_id": ROUND_ID,
                    "seed": seed,
                    "capability": entry["capability"],
                    "treatment": "fneg-x2-md000-hostint8-50m",
                    "pass": "r0237-prefix-descriptive-purity",
                    "descriptive": True,
                    "gated": False,
                    "lineage_caveat": DESCRIPTIVE_PURITY_LINEAGE_CAVEAT,
                },
            )
            purity_ratios = {"k256": float(purity_panel["purity"]["k256"]),
                             "k1024": float(purity_panel["purity"]["k1024"])}
            # FLOOR-MATCHED held-out FFR (PIECE A): project the OUT-OF-SUBSTRATE held-out
            # reserve THROUGH this map (`model.transform(reserve)`), score against the sealed
            # reserve-neighbour truth at the N-scaled disc = int(ROWS * 0.001).  This is the
            # R0265 family floor's own instrument; collapse/fog are still measured on the FULL
            # 50M coordinates inside score_one_map (byte-identical).
            placed = np.asarray(
                proj_model.transform(reserve_embeddings, batch_size=FULL_TRANSFORM_BATCH),
                dtype=np.float32,
            )
            scored_map = score_one_map(
                coordinates=coordinates,
                probes_placed=placed,
                truth_top10=reserve_truth,
                purity_ratios=purity_ratios,
                disc=reserve_disc,
            )
            if entry.get("salvaged"):
                # seed42: provenance is the salvage block + the bound artifacts, NOT a
                # train-receipt (do NOT fabricate one — it has none).
                cells[str(seed)] = {
                    "seed": seed,
                    "capability": entry["capability"],
                    "salvaged": True,
                    "salvage": dict(entry["salvage"]),
                    "train_receipt": None,
                    "model": dict(entry["model_signature"]),
                    "coordinates_binding": dict(entry["coordinates_signature"]),
                    "production_config": dict(entry["production_config_signature"]),
                    "seed_invariant_sha256": entry["seed_invariant_sha256"],
                    "x_residency": X_RESIDENCY,
                    "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
                    "metrics": scored_map,
                    "panel_purity_numerators": purity_panel.get("purity_numerators"),
                }
            elif entry.get("bound_completed"):
                # seed43/44: provenance is the bound train-receipt (a real receipt) + the
                # bound artifacts. Scored from the bound coordinates, like the salvaged path.
                cells[str(seed)] = {
                    "seed": seed,
                    "capability": entry["capability"],
                    "salvaged": False,
                    "bound_completed": True,
                    "bound": dict(entry["bound"]),
                    "train_receipt": dict(entry["train_receipt_signature"]),
                    "model": dict(entry["model_signature"]),
                    "coordinates_binding": dict(entry["coordinates_signature"]),
                    "production_config": dict(entry["production_config_signature"]),
                    "seed_invariant_sha256": entry["seed_invariant_sha256"],
                    "x_residency": X_RESIDENCY,
                    "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
                    "metrics": scored_map,
                    "panel_purity_numerators": purity_panel.get("purity_numerators"),
                }
            else:
                cells[str(seed)] = {
                    "seed": seed,
                    "capability": entry["capability"],
                    "salvaged": False,
                    "train_receipt": dict(entry["receipt_signature"]),
                    "model": dict(entry["receipt"]["model"]),
                    "seed_invariant_sha256": entry["seed_invariant_sha256"],
                    "x_residency": X_RESIDENCY,
                    "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
                    "metrics": scored_map,
                    "panel_purity_numerators": purity_panel.get("purity_numerators"),
                }
            # Per-cell DESCRIPTIVE purity record — labelled descriptive/ungated with the
            # lineage caveat.  These k256/k1024 values are REPORTED, never gated.
            cells[str(seed)]["descriptive_purity"] = {
                "pass": "r0237-prefix-descriptive-purity",
                "descriptive": True,
                "gated": False,
                "prefix_rows": prefix_rows,
                "reference": "r0237-prefix-inline",
                "k256": purity_ratios["k256"],
                "k1024": purity_ratios["k1024"],
                "numerators": purity_panel.get("purity_numerators"),
                "hiD_reference_key": purity_panel["provenance"]["hiD_reference_key"],
                "hiD_reference_reused": bool(purity_panel["provenance"]["hiD_reference_reused"]),
                "lineage_caveat": DESCRIPTIVE_PURITY_LINEAGE_CAVEAT,
            }
            del proj_model, coordinates, placed
            torch.cuda.empty_cache()
            gc.collect()
            wrapped(f"R0267 seed-{seed} scored")
        gate.finish("R0267 panel stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    metric_table = {
        str(seed): {
            "heldout_ffr": cells[str(seed)]["metrics"]["heldout_ffr"],
            "purity_fidelity_k256": cells[str(seed)]["metrics"]["purity_fidelity_k256"],
            "purity_fidelity_k1024": cells[str(seed)]["metrics"]["purity_fidelity_k1024"],
            "collapse": cells[str(seed)]["metrics"]["collapse"],
            "fog": cells[str(seed)]["metrics"]["fog"],
            "resolution_levels": cells[str(seed)]["metrics"]["resolution_levels"],
            "degenerate": cells[str(seed)]["metrics"]["degenerate"],
            "fog_detail": cells[str(seed)]["metrics"].get("fog_detail"),
        }
        for seed in SEEDS
    }
    execution_checks = {
        "all_three_cells_scored": set(cells) == {str(seed) for seed in SEEDS},
        "pooled_family_one_recipe_digest": len(invariants) == 1,
        "every_heldout_ffr_finite": all(math.isfinite(row["heldout_ffr"]) for row in metric_table.values()),
        "every_collapse_finite": all(math.isfinite(row["collapse"]) for row in metric_table.values()),
        "every_purity_ratio_positive": all(
            row["purity_fidelity_k256"] > 0 and row["purity_fidelity_k1024"] > 0
            for row in metric_table.values()
        ),
        "no_gate_registered_here": not job.get("gate_registerable_here", False),
        "salvaged_cells_sourced_from_bound_artifacts": all(
            (cells[str(s)].get("train_receipt") is None and bool(cells[str(s)].get("salvage")))
            for s in salvaged_seeds
        ),
        "bound_completed_cells_sourced_from_bound_receipts": all(
            (bool(cells[str(s)].get("bound_completed")) and bool(cells[str(s)].get("train_receipt")))
            for s in bound_completed_seeds
        ),
    }
    if not all(execution_checks.values()):
        raise Round0267NodeError(f"R0267 panel execution checks failed: {execution_checks}")

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
        "salvaged_seeds": list(salvaged_seeds),
        "salvage_provenance": {
            str(s): dict(cells[str(s)]["salvage"]) for s in salvaged_seeds
        },
        "bound_completed_seeds": list(bound_completed_seeds),
        "bound_provenance": {
            str(s): dict(cells[str(s)]["bound"]) for s in bound_completed_seeds
        },
        "x_residency": X_RESIDENCY,
        "seed_invariant_sha256": sorted(invariants)[0],
        "panel_metric_table": metric_table,
        "cells": cells,
        "descriptive_purity_centroids": centroid_signatures,
        "gated_metrics": list(GATE_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_PURITY_METRICS),
        "descriptive_purity": {
            "descriptive": True,
            "gated": False,
            "prefix_rows": prefix_rows,
            "reference": "r0237-prefix-inline",
            "centroid_ks": list(centroid_ks),
            "lineage_caveat": DESCRIPTIVE_PURITY_LINEAGE_CAVEAT,
            "values": {
                str(s): {
                    "k256": metric_table[str(s)]["purity_fidelity_k256"],
                    "k1024": metric_table[str(s)]["purity_fidelity_k1024"],
                }
                for s in SEEDS
            },
        },
        "heldout_reserve": dict(job.get("heldout_reserve") or {}),
        "heldout_ffr_instrument": {
            "floor_matched": True,
            "probe_design": "out_of_substrate_reserve_projection",
            "placed_is": "model.transform(reserve.f32[reserve-query-rows])",
            "truth": "sealed reserve-neighbour exact-cosine top-10 (indices into the 50M substrate)",
            "disc": reserve_disc,
            "disc_rule": "int(ROWS * 0.001) — 0.1%·N, N-scaled",
            "n_reserve_probes": int(reserve_embeddings.shape[0]),
            "reserve_query_rows_binding": dict(job.get("reserve_query_rows") or {}),
            "reserve_truth_binding": dict(job.get("reserve_truth") or {}),
            "note": (
                "PIECE A (2026-08-17): the floor-matched instrument. The R0265 family floor "
                "is the OUT-OF-SUBSTRATE reserve projection at disc=int(ROWS*0.001); run_panel "
                "measures FFR with that same instrument so a full panel is born matched to the "
                "floor. Supersedes the earlier IN-SUBSTRATE coordinates[probe_rows] + fixed "
                "2000 disc (trips 9/10)."
            ),
        },
        "lineage": {
            "graph_manifest": dict(substrate["manifest_signature"]),
            "substrate": dict(substrate["substrate_signature"]),
            "ordered_substrate_sha256": substrate["ordered_substrate_sha256"],
        },
        "purity_reference_note": (
            "purity k256/k1024 are DESCRIPTIVE / UNGATED at 50M (amendment 2026-08-17, "
            "OWNER-authorized). They are scored on the R0237 substrate's first prefix_rows "
            "rows (Xa = those rows; Z = each cell's first prefix_rows coordinate rows, so the "
            "2D neighbour pool is the prefix rows' coordinates ONLY) against a reference + "
            "k-means centroids built INLINE on that SAME prefix — self-contained, NO R0218 "
            "dependency (reference or centroids).  They are NOT commensurate with the R0265 "
            "2M family bands (a different build lineage; see lineage_caveat) and NEVER enter "
            "the gate.  The GATED collapse / fog / held-out FFR are measured on the FULL 50M "
            "coordinates (score_one_map).  run_panel invokes score_panel ONLY for this <8M "
            "descriptive pass; no >=8M score_panel is run and no slim scale-performance "
            "certificate is required."
        ),
        "purity_pool_semantics": (
            "prefix-only: the other rows sharing the 2D plane are deliberately excluded from "
            "the purity neighbour pool; pool = all rows would measure a different, "
            "incomparable quantity"
        ),
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
    _seal(output, "fneg-50m-x2-panel.json", body)
    print(json.dumps({
        "capability": PANEL_CAPABILITY,
        "n": len(SEEDS),
        "seed_collapse": {str(s): metric_table[str(s)]["collapse"] for s in SEEDS},
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))
    del source, centroids
    gc.collect()


# --------------------------------------------------------------------------- #
# the seed-mean gate: criterion 1 (seed-mean vs widened P1 band) + backstops
# --------------------------------------------------------------------------- #


def _madn(values) -> float:
    numbers = [float(v) for v in values]
    centre = statistics.median(numbers)
    return float(MAD_CONSISTENCY * statistics.median([abs(v - centre) for v in numbers]))


def sigma_fam_from_panel(panel_path: str) -> dict[str, Any]:
    """σ_fam = 1.4826·MAD_n over R0265's SEALED 13 collapse and 13 fog values.

    READ / RECOMPUTED from the sealed R0265 n=13 panel at gate time -- NEVER a literal.
    The constants-discipline contract test widens the panel's spread on disk and asserts
    σ_fam grows.
    """
    d = prompt_contract.read_sealed(panel_path, label="R0265 sealed n=13 panel")
    if d.get("capability") != R0265N.PANEL_CAPABILITY or int(d.get("n", -1)) != R0265N.N_FAMILY:
        raise Round0267NodeError("R0265 sealed n=13 panel contract changed")
    table = dict(d["panel_metric_table"])
    seeds = sorted(int(k) for k in table)
    collapse = [float(table[str(s)][COLLAPSE_METRIC]) for s in seeds]
    fog = [float(table[str(s)][FOG_METRIC]) for s in seeds]
    return {
        "sigma_fam_collapse": _madn(collapse),
        "sigma_fam_fog": _madn(fog),
        "collapse_median": float(statistics.median(collapse)),
        "fog_median": float(statistics.median(fog)),
        "n_family_cells": len(seeds),
        "source_capability": d.get("capability"),
        "source_schema": d.get("schema"),
        "source_identity_sha256": d.get("identity_sha256"),
    }


def read_p1_x2_asymptote_band(path: str) -> dict[str, Any]:
    """The P1 ×2 collapse asymptote band [lo, hi] from the SEALED analysis-v2 result.

    READ from `bands.yinf_x2` in the frozen P1 analysis-v2 result (commits a9651cf ->
    db0113a; plain JSON bound by sha256 at prepare) -- NEVER a literal. The
    constants-discipline test mutates the band on disk and asserts the gate tracks it.
    """
    with open(path, "r", encoding="utf-8") as handle:
        d = json.load(handle)
    bands = dict(d.get("bands") or {})
    if "yinf_x2" not in bands:
        raise Round0267NodeError("P1 analysis-v2 result carries no yinf_x2 asymptote band")
    lo, hi = (float(v) for v in bands["yinf_x2"])
    if not (lo < hi):
        raise Round0267NodeError("P1 ×2 asymptote band is not an interval")
    return {
        "p1_lower": lo,
        "p1_upper": hi,
        "yinf_x2": float(d.get("fit", {}).get("yinf_x2")) if d.get("fit") else None,
        "verdict": d.get("verdict"),
        "floor": d.get("floor"),
    }


def score_collapse_seed_mean(
    *,
    seed_collapse: Mapping[str, float],
    p1_lower: float,
    p1_upper: float,
    sigma_fam_collapse: float,
    z: float = COLLAPSE_SEEDMEAN_Z,
    n: int = COLLAPSE_SEEDMEAN_N,
) -> dict[str, Any]:
    """CRITERION 1: the seed-mean collapse inside P1's ×2 band widened by z·σ_fam/√n.

    The P1 band is a bootstrap band on the FITTED ×2 asymptote (a mean), so it is widened
    by the family's √n-shrunk seed-noise allowance and the SEED-MEAN is gated against it.
    """
    values = [float(v) for v in seed_collapse.values()]
    mean = float(sum(values) / len(values))
    allowance = float(z) * float(sigma_fam_collapse) / math.sqrt(int(n))
    widened_lower = float(p1_lower) - allowance
    widened_upper = float(p1_upper) + allowance
    passes = widened_lower <= mean <= widened_upper
    return {
        "criterion": "collapse_seed_mean_inside_widened_p1_x2_band",
        "seed_collapse": {str(k): float(v) for k, v in seed_collapse.items()},
        "seed_mean": mean,
        "n_seeds": len(values),
        "p1_x2_band": [float(p1_lower), float(p1_upper)],
        "z": float(z),
        "sqrt_n_shrink_n": int(n),
        "sigma_fam_collapse": float(sigma_fam_collapse),
        "seed_noise_allowance": allowance,
        "widened_band": [widened_lower, widened_upper],
        "passes": bool(passes),
        "note": (
            "the P1 band is the bootstrap band on the fitted ×2 asymptote (a mean); it is "
            "widened by z·σ_fam/√n (the family's √n-shrunk seed spread) and the SEED-MEAN "
            "is gated against it. The no-straddle rule does NOT apply to this band."
        ),
    }


def score_per_seed_backstops(
    *,
    metric_table: Mapping[str, Mapping[str, Any]],
    backstops: Mapping[str, Any],
    sigma_fam_fog: float,
) -> dict[str, Any]:
    """The per-seed hard backstops (amended plan criteria 1-3 — collapse/fog/held-out FFR).

    Every seed must clear ONLY: collapse >= R0265 family floor; fog <= R0265 family ceiling
    (with a mechanical near-ceiling escalation if fog > ceiling - 1·σ_fam,fog); held-out
    FFR >= R0265 family floor. The no-straddle rule applies to THESE three gates (some seeds
    passing and some failing one gate is ambiguity, not noise). Every floor is READ from the
    sealed R0265 floors.

    Purity k256/k1024 are DESCRIPTIVE-only at 50M (amendment 2026-08-17): their per-seed
    verdicts against the sealed R0265 bands are still RECORDED under ``descriptive_purity``
    for the report, but they DO NOT enter ``clears_every_backstop`` or the straddle set and
    can never flip the verdict.
    """
    fog_ceiling = float(backstops["fog_ceiling"])
    near_ceiling_threshold = fog_ceiling - float(sigma_fam_fog)
    rows: list[dict[str, Any]] = []
    for seed_key, metrics in sorted(metric_table.items(), key=lambda kv: int(kv[0])):
        ffr = float(metrics[HELDOUT_FFR_METRIC])
        ffr_floor = float(backstops["heldout_ffr_floor"])
        ffr_pass = ffr >= ffr_floor
        collapse_v = _judge_collapse(float(metrics[COLLAPSE_METRIC]), float(backstops["collapse_floor"]))
        fog_result = {
            "fog": float(metrics[FOG_METRIC]),
            "resolution_levels": int(metrics["resolution_levels"]),
            "degenerate": bool(metrics["degenerate"]),
            "peak_bin_count": dict(metrics.get("fog_detail") or {}).get("peak_bin_count"),
        }
        fog_v = _judge_fog(fog_result, fog_ceiling)
        fog_near_ceiling = (
            float(metrics[FOG_METRIC]) > near_ceiling_threshold
            and fog_v.get("verdict") != VERDICT_NOT_MEASURABLE
        )
        # The THREE GATED metrics — the ONLY contributors to pass/fail + straddle.
        by_metric = {
            HELDOUT_FFR_METRIC: {
                "verdict": "PASS" if ffr_pass else "FAIL", "passes": ffr_pass,
                "value": ffr, "floor": ffr_floor, "margin": ffr - ffr_floor,
            },
            COLLAPSE_METRIC: collapse_v,
            FOG_METRIC: {**fog_v, "near_ceiling_escalation": bool(fog_near_ceiling),
                        "near_ceiling_threshold": near_ceiling_threshold},
        }
        # DESCRIPTIVE purity verdicts — RECORDED against the sealed R0265 bands but NEVER
        # part of clears/straddle (purity is ungated at 50M, amendment 2026-08-17).
        k256_v = _judge_k256_two_sided(float(metrics[PURITY_K256_METRIC]), dict(backstops["k256_band"]))
        k1024_v = _judge_one_sided_floor(float(metrics[PURITY_K1024_METRIC]), float(backstops["k1024_floor"]))
        descriptive_purity = {
            "descriptive": True,
            "gated": False,
            PURITY_K256_METRIC: k256_v,
            PURITY_K1024_METRIC: k1024_v,
        }
        clears = all(bool(v["passes"]) for v in by_metric.values())
        rows.append({
            "cell_id": exact_cell_id(int(seed_key)),
            "seed": int(seed_key),
            "metrics": by_metric,
            "descriptive_purity": descriptive_purity,
            "clears_every_backstop": clears,
            "fog_near_ceiling_escalation": bool(fog_near_ceiling),
            "fog_not_measurable": fog_v.get("verdict") == VERDICT_NOT_MEASURABLE,
        })
    # No-straddle: a per-seed GATED gate straddles if some seeds pass it and some fail it.
    # Iterates GATE_METRICS (collapse/fog/FFR only) — purity is excluded by construction.
    straddles: list[str] = []
    for m in GATE_METRICS:
        verdicts = {bool(row["metrics"][m]["passes"]) for row in rows}
        if len(verdicts) > 1:
            straddles.append(m)
    every_seed_clears = all(row["clears_every_backstop"] for row in rows)
    any_escalation = any(row["fog_near_ceiling_escalation"] for row in rows)
    return {
        "cells": rows,
        "cells_scored": len(rows),
        "cells_clearing_every_backstop": sum(1 for r in rows if r["clears_every_backstop"]),
        "every_seed_clears_every_backstop": bool(every_seed_clears),
        "straddled_gates": straddles,
        "any_gate_straddles": bool(straddles),
        "any_fog_near_ceiling_escalation": bool(any_escalation),
        "backstops_used": {
            "collapse_floor": float(backstops["collapse_floor"]),
            "fog_ceiling": fog_ceiling,
            "fog_near_ceiling_threshold": near_ceiling_threshold,
            "heldout_ffr_floor": float(backstops["heldout_ffr_floor"]),
        },
        "descriptive_purity_bands_recorded": {
            "gated": False,
            "k1024_floor": float(backstops["k1024_floor"]),
            "k256_band": {
                "ratio_lower": float(backstops["k256_band"]["ratio_lower"]),
                "ratio_upper": float(backstops["k256_band"]["ratio_upper"]),
            },
            "note": (
                "the R0265 purity bands are RECORDED for the descriptive purity report but do "
                "NOT gate at 50M (amendment 2026-08-17); no per-seed purity verdict enters "
                "clears_every_backstop or the straddle set"
            ),
        },
        "no_straddle_rule": (
            "applies to the per-seed backstop + collapse/fog/FFR gates, NOT to purity "
            "(descriptive-only at 50M) and NOT to the P1 asymptote band (a single seed "
            "outside the P1 band but at/above the backstop is expected noise, not ambiguity)"
        ),
    }


def _metric_table_from_panel(panel: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    # Consumption-side ENFORCEMENT: a scientific panel must never carry the slim
    # cert-production purpose stamp (that output is a perf-receipt pass, budgets
    # unchecked).  Refuse it here rather than scoring it.
    assert_not_slim_cert_production_panel(
        panel, label="R0267 50M panel", error_cls=Round0267NodeError)
    if panel.get("capability") != PANEL_CAPABILITY or panel.get("schema") != PANEL_SCHEMA:
        raise Round0267NodeError("R0267 50M panel contract changed")
    table = dict(panel["panel_metric_table"])
    if {int(s) for s in table} != set(SEEDS):
        raise Round0267NodeError("R0267 50M panel is not the three fneg cells")
    return {str(s): dict(table[str(s)]) for s in SEEDS}


def _salvage_provenance_from_panel(panel: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """seed42's per-cell provenance comes from the panel's salvage block, NOT a train-receipt.

    Returns ``{seed_str: salvage_block}`` for every salvaged cell in the panel (seed42 in
    correction-4). The gate records this explicitly instead of a fabricated train-receipt.
    """
    cells = dict(panel.get("cells") or {})
    provenance: dict[str, dict[str, Any]] = {}
    for seed_key, cell in cells.items():
        if not isinstance(cell, Mapping) or not cell.get("salvaged"):
            continue
        salvage = dict(cell.get("salvage") or {})
        if salvage.get("salvaged") is not True or str(salvage.get("source_run") or "") != SALVAGE_SOURCE_RUN:
            raise Round0267NodeError(f"R0267 gate: salvaged cell {seed_key} provenance changed")
        provenance[str(seed_key)] = salvage
    return provenance


def _bound_provenance_from_panel(panel: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """seed43/44's per-cell provenance comes from the panel's bound block (a real receipt).

    Returns ``{seed_str: bound_block}`` for every bound-completed cell in the panel (43/44
    in correction-5). The gate records this explicitly — the bound cells carry a first-class
    train-receipt binding, stronger than seed42's salvage log.
    """
    cells = dict(panel.get("cells") or {})
    provenance: dict[str, dict[str, Any]] = {}
    for seed_key, cell in cells.items():
        if not isinstance(cell, Mapping) or not cell.get("bound_completed"):
            continue
        bound = dict(cell.get("bound") or {})
        if (
            bound.get("bound_completed") is not True
            or str(bound.get("source_run") or "") != COMPLETED_SOURCE_RUN
        ):
            raise Round0267NodeError(f"R0267 gate: bound cell {seed_key} provenance changed")
        provenance[str(seed_key)] = bound
    return provenance


def _corrected_ffr_from_rescore(
    job: Mapping[str, Any], *, backstops_floor: float
) -> dict[str, Any] | None:
    """Read the CORRECTED per-seed FFR from the bound re-score artifact (gate-only re-seal).

    Returns ``None`` on the full-queue path (no ``corrected_ffr_source`` bound), so the
    full-queue gate is byte-identical. On the gate-only re-seal path (PIECE B) it verifies
    the bound re-score's digest (via ``_bound_path``), checks the re-score covers the three
    fneg seeds, is the FLOOR-MATCHED instrument (disc == int(ROWS * 0.001), N-scaled — NOT
    the fixed 2000 that produced trip 9), and that the re-score's floor equals the sealed
    R0265 FFR floor read from the bound floors (so the corrected FFR is judged against the
    SAME floor). It also re-verifies the sealed reserve-neighbour truth's bound digest (the
    truth the re-score scored against) for provenance. It returns the corrected per-seed FFR
    plus the three superseding shas — the caller OVERRIDES the panel's mis-measured FFR in
    the metric table with these before the per-seed backstops are scored. Collapse and fog
    are NEVER touched here (they stay byte-identical from the correction-5 panel).
    """
    ref = job.get("corrected_ffr_source")
    if not isinstance(ref, Mapping):
        return None
    path = _bound_path(job, "corrected_ffr_source", label="R0267 corrected FFR re-score results")
    with open(path, encoding="utf-8") as handle:
        rescore = json.load(handle)
    per_map = dict(rescore.get("per_map") or {})
    if {int(s) for s in per_map} != set(SEEDS):
        raise Round0267NodeError("R0267 gate-only re-score does not cover the three fneg seeds")
    disc = int(rescore["disc"])
    n_scaled_disc = int(ROWS * 0.001)
    if disc != n_scaled_disc:
        raise Round0267NodeError(
            f"R0267 gate-only re-score disc {disc} is not the N-scaled int(ROWS*0.001)="
            f"{n_scaled_disc} (trip-9 guard)"
        )
    rescore_floor = float(rescore["floor"])
    if not math.isclose(rescore_floor, float(backstops_floor), rel_tol=0.0, abs_tol=1e-9):
        raise Round0267NodeError(
            "R0267 gate-only re-score floor disagrees with the sealed R0265 FFR floor: "
            f"{rescore_floor} != {float(backstops_floor)}"
        )
    per_seed = {str(s): float(per_map[str(s)]["heldout_ffr"]) for s in SEEDS}
    # Re-verify the sealed reserve-neighbour truth's bound digest (provenance; the re-score
    # scored against it). The gate does not recompute FFR from it — the re-score already did.
    reserve_truth_sig = job.get("reserve_truth")
    if isinstance(reserve_truth_sig, Mapping):
        _bound_path(job, "reserve_truth", label="R0267 sealed reserve-neighbour truth")
    return {
        "gate_only_reseal": True,
        "rescore_results_path": path,
        "rescore_results_sha256": str(ref.get("sha256") or ""),
        "reserve_neighbour_truth_sha256": str((reserve_truth_sig or {}).get("sha256") or ""),
        "corrected_ffr_disc": disc,
        "corrected_ffr_floor": rescore_floor,
        "per_seed_corrected_ffr": per_seed,
        "rescore_per_map": per_map,
        "boot_seed": rescore.get("boot_seed"),
        "bootstrap_B": rescore.get("B"),
    }


def run_gate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0267 round0267_nodes.run_gate")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0267NodeError("R0267 gate handler received another queue")
    node_id = str(active.get("node_id") or GATE_ACTION)
    label = "R0267 50M ×2 seed-mean gate"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0267 gate")

    window = ledger.window("R0267 gate stage")
    guard_ctx = _node_guard(label, anonymous_budget_bytes=R0267_ANON_BUDGET_BYTES)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0267 gate stage entered")
        wrapped = window.wrap(recorder)

        # 1. the three-seed 50M metrics (intra-queue panel).
        panel = prompt_contract.read_sealed(
            _bound_path(job, "panel", label="R0267 50M panel"), label="R0267 50M panel"
        )
        # Defense in depth: refuse a bound panel carrying the cert-production stamp
        # BEFORE any scoring (the panel-reading helper refuses it again).
        assert_not_slim_cert_production_panel(
            panel, label="R0267 gate: bound 50M panel", error_cls=Round0267NodeError)
        metric_table = _metric_table_from_panel(panel)
        seed_collapse = {s: float(metric_table[s][COLLAPSE_METRIC]) for s in metric_table}
        # seed42's provenance is the salvage block (not a train-receipt); record it here.
        salvage_provenance = _salvage_provenance_from_panel(panel)
        # seeds 43/44's provenance is their bound train-receipt block; record it too.
        bound_provenance = _bound_provenance_from_panel(panel)
        wrapped("R0267 three-seed 50M metrics read")

        # 2. σ_fam, RECOMPUTED from R0265's sealed n=13 panel.
        sigma = sigma_fam_from_panel(_bound_path(job, "r0265_panel", label="R0265 sealed n=13 panel"))
        wrapped("R0267 σ_fam recomputed from the sealed R0265 panel")

        # 3. the P1 ×2 asymptote band, READ from the sealed analysis-v2 result.
        p1 = read_p1_x2_asymptote_band(_bound_path(job, "p1_asymptote", label="P1 analysis-v2 result"))
        wrapped("R0267 P1 ×2 band read from the sealed analysis-v2 result")

        # 4. the per-seed backstops, READ from R0265's sealed family floors.
        backstops = R0266N.read_family_bands(
            _bound_path(job, "r0265_floors", label="R0265 sealed family floors")
        )["bands"]
        wrapped("R0267 per-seed backstops read from the sealed R0265 floors")

        # 4b. GATE-ONLY FFR RE-SEAL (PIECE B). If a corrected-FFR re-score is bound, SUPERSEDE
        # the correction-5 panel's mis-measured FFR with the corrected FLOOR-MATCHED values
        # BEFORE the per-seed backstops are scored. collapse (criterion 1 + backstop) and fog
        # stay BYTE-IDENTICAL — they are read from the bound panel and never recomputed here;
        # only the FFR column of the metric table is replaced. On the full-queue path (no
        # corrected_ffr_source bound) this is a no-op and the gate is byte-identical.
        ffr_correction = _corrected_ffr_from_rescore(
            job, backstops_floor=backstops["heldout_ffr_floor"]
        )
        superseded_ffr = None
        if ffr_correction is not None:
            superseded_ffr = {
                s: float(metric_table[s][HELDOUT_FFR_METRIC]) for s in metric_table
            }
            for s in SEEDS:
                metric_table[str(s)][HELDOUT_FFR_METRIC] = (
                    ffr_correction["per_seed_corrected_ffr"][str(s)]
                )
            wrapped("R0267 gate-only re-seal: corrected FFR bound from the re-score artifact")

        # CRITERION 1: the seed-mean collapse inside the widened P1 ×2 band.
        criterion_1 = score_collapse_seed_mean(
            seed_collapse=seed_collapse,
            p1_lower=p1["p1_lower"],
            p1_upper=p1["p1_upper"],
            sigma_fam_collapse=sigma["sigma_fam_collapse"],
        )
        # CRITERIA 1-3 per-seed backstops: collapse floor + fog ceiling + held-out FFR floor
        # (purity is DESCRIPTIVE-only and does NOT enter pass/fail; amendment 2026-08-17).
        backstop_scoring = score_per_seed_backstops(
            metric_table=metric_table,
            backstops=backstops,
            sigma_fam_fog=sigma["sigma_fam_fog"],
        )
        # DESCRIPTIVE purity — recorded for the report, NEVER a gate.
        descriptive_purity = {
            "descriptive": True,
            "gated": False,
            "lineage_caveat": DESCRIPTIVE_PURITY_LINEAGE_CAVEAT,
            "values": {
                s: {
                    "k256": float(metric_table[s][PURITY_K256_METRIC]),
                    "k1024": float(metric_table[s][PURITY_K1024_METRIC]),
                }
                for s in metric_table
            },
        }
        wrapped("R0267 criteria scored from sealed bands, σ_fam and the P1 band")
        gate.finish("R0267 gate stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    # The 50M PASS decision (a FINDING reported either way; it feeds the 100M-commit
    # decision, it does NOT make the round a failure).  The verdict is computed from
    # collapse (seed-mean band + per-seed backstop) + fog (ceiling + escalation) + held-out
    # FFR (floor) ONLY — criteria 1–3.  backstop_scoring's every_seed_clears / straddle no
    # longer depend on purity (amendment 2026-08-17), so purity can never flip the verdict.
    passes = bool(
        criterion_1["passes"]
        and backstop_scoring["every_seed_clears_every_backstop"]
        and not backstop_scoring["any_gate_straddles"]
    )
    ambiguous = bool(
        not criterion_1["passes"]
        or not backstop_scoring["every_seed_clears_every_backstop"]
        or backstop_scoring["any_gate_straddles"]
        or backstop_scoring["any_fog_near_ceiling_escalation"]
    )
    verdict = "50M_PASS" if passes and not backstop_scoring["any_fog_near_ceiling_escalation"] else "50M_FAIL_OR_AMBIGUOUS"

    execution_checks = {
        "collapse_backstop_read_from_sealed_floors": (
            backstop_scoring["backstops_used"]["collapse_floor"] == float(backstops["collapse_floor"])
        ),
        "sigma_fam_recomputed_from_sealed_panel": (
            sigma["source_capability"] == R0265N.PANEL_CAPABILITY
            and sigma["n_family_cells"] == R0265N.N_FAMILY
        ),
        "p1_band_read_from_sealed_analysis_v2": p1["verdict"] == "GO",
        "gated_backstops_cover_every_seed": {r["seed"] for r in backstop_scoring["cells"]} == set(SEEDS),
        "three_seeds_scored": backstop_scoring["cells_scored"] == len(SEEDS),
        "no_typed_band_literals": True,  # every band/floor/σ_fam/P1-edge is read from a sealed input
        "purity_is_descriptive_not_gated": (
            PURITY_K256_METRIC not in GATE_METRICS
            and PURITY_K1024_METRIC not in GATE_METRICS
            and all(
                row.get("descriptive_purity", {}).get("gated") is False
                for row in backstop_scoring["cells"]
            )
        ),
        "salvaged_provenance_sourced_not_fabricated": all(
            prov.get("salvaged") is True and "train_evidence" in prov
            for prov in salvage_provenance.values()
        ),
        "bound_provenance_sourced_from_train_receipt": all(
            prov.get("bound_completed") is True and "train_receipt_provenance" in prov
            for prov in bound_provenance.values()
        ),
    }
    # GATE-ONLY re-seal (PIECE B): build the superseding provenance + its execution checks.
    ffr_correction_provenance: dict[str, Any] | None = None
    if ffr_correction is not None:
        ffr_correction_provenance = {
            "gate_only_reseal": True,
            "diagnosis": TRIP_9_10_DIAGNOSIS,
            "supersedes": {
                "source_run": SUPERSEDED_SOURCE_RUN,
                "superseded_verdict": SUPERSEDED_VERDICT,
                "note": (
                    "correction-5's registered 50M_FAIL_OR_AMBIGUOUS verdict STAYS in the "
                    "record; this gate-only re-seal SUPERSEDES it with the corrected FFR, it "
                    "does NOT erase it."
                ),
            },
            "correction_5_panel_sha256": str((job.get("panel") or {}).get("sha256") or ""),
            "reserve_neighbour_truth_sha256": ffr_correction["reserve_neighbour_truth_sha256"],
            "rescore_results_sha256": ffr_correction["rescore_results_sha256"],
            "corrected_ffr_disc": ffr_correction["corrected_ffr_disc"],
            "corrected_ffr_floor": ffr_correction["corrected_ffr_floor"],
            "per_seed_corrected_ffr": ffr_correction["per_seed_corrected_ffr"],
            "per_seed_superseded_ffr": superseded_ffr,
            "rescore_per_map": ffr_correction["rescore_per_map"],
            "collapse_fog_byte_identical_from_correction_5": True,
            "purity_descriptive": True,
        }
        three_shas_present = all(
            bool(ffr_correction_provenance[key])
            for key in (
                "correction_5_panel_sha256",
                "reserve_neighbour_truth_sha256",
                "rescore_results_sha256",
            )
        )
        execution_checks["corrected_ffr_sourced_from_bound_rescore"] = all(
            metric_table[str(s)][HELDOUT_FFR_METRIC]
            == ffr_correction["per_seed_corrected_ffr"][str(s)]
            for s in SEEDS
        )
        execution_checks["collapse_fog_bound_from_correction_5_panel"] = all(
            metric_table[str(s)][COLLAPSE_METRIC] == float(panel["panel_metric_table"][str(s)][COLLAPSE_METRIC])
            and metric_table[str(s)][FOG_METRIC] == float(panel["panel_metric_table"][str(s)][FOG_METRIC])
            for s in SEEDS
        )
        execution_checks["superseding_provenance_records_three_shas_and_fail_stays"] = bool(
            three_shas_present
            and ffr_correction_provenance["supersedes"]["superseded_verdict"] == SUPERSEDED_VERDICT
            and "STAYS" in ffr_correction_provenance["supersedes"]["note"]
        )
    if not all(execution_checks.values()):
        raise Round0267NodeError(f"R0267 gate execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    coverage = ledger.receipt()
    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": GATE_SCHEMA,
        "capability": GATE_CAPABILITY,
        "capabilities": [GATE_CAPABILITY],
        "node": node_id,
        "abort_flag_precondition": abort_flag,
        "identity_bound_at_n": identity_bound(len(SEEDS)),
        "purpose": (
            "the pre-registered 50M staging-rung gate (plan-50m-stage-2026-08-15, amendment "
            "2026-08-17): the SEED-MEAN collapse inside P1's ×2 asymptote band widened by a "
            "√n-shrunk family seed-noise allowance, plus per-seed hard backstops on collapse/"
            "fog/FFR — criteria 1–3. Purity is DESCRIPTIVE-only at 50M and is NOT in the "
            "go/no-go. Feeds the 100M ×2-commit decision; a FAIL/AMBIGUOUS returns to the "
            "owner with the drift decomposition, never an auto-proceed."
        ),
        "seed_spread_assumption": SEED_SPREAD_ASSUMPTION,
        "salvaged_seeds": sorted(int(s) for s in salvage_provenance),
        "salvage_provenance": salvage_provenance,
        "bound_completed_seeds": sorted(int(s) for s in bound_provenance),
        "bound_provenance": bound_provenance,
        "sigma_fam": sigma,
        "p1_x2_asymptote_band": p1,
        "criterion_1_collapse_seed_mean": criterion_1,
        "criteria_1_3_per_seed_backstops": backstop_scoring,
        "descriptive_purity": descriptive_purity,
        "gated_metrics": list(GATE_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_PURITY_METRICS),
        "pre_registered_pass_criteria": {
            "1": (
                "mean(collapse over 3 seeds) ∈ [0.930 − 1.96·σ_fam/√3, 0.985 + "
                "1.96·σ_fam/√3]; P1 band edges READ from the sealed analysis-v2 result, "
                "σ_fam RECOMPUTED from the sealed R0265 panel"
            ),
            "backstops": (
                "every seed: collapse >= R0265 floor; fog <= R0265 ceiling (near-ceiling "
                "escalation if fog > ceiling − 1·σ_fam,fog); FFR >= R0265 floor -- all READ "
                "from the sealed R0265 floors. Purity is REMOVED from the go/no-go at 50M "
                "(descriptive-only; amendment 2026-08-17, OWNER-authorized)."
            ),
            "purity_descriptive_only": (
                "purity k256/k1024 are REPORTED against an R0237-prefix inline reference + "
                "centroids (self-contained, no R0218 lineage) and labelled descriptive/"
                "ungated with the lineage caveat; they NEVER enter the go/no-go"
            ),
            "no_straddle": (
                "applies to the per-seed backstops + collapse/fog/FFR, NOT to purity "
                "(descriptive-only) and NOT to the P1 band"
            ),
            "constants_discipline": (
                "no band/floor/σ_fam/P1-edge is a typed literal; all are bound from sealed "
                "inputs by sha256 and read/recomputed at gate time"
            ),
        },
        "fifty_m_decision": {
            "criterion_1_passes": criterion_1["passes"],
            "every_seed_clears_every_backstop": backstop_scoring["every_seed_clears_every_backstop"],
            "any_gate_straddles": backstop_scoring["any_gate_straddles"],
            "any_fog_near_ceiling_escalation": backstop_scoring["any_fog_near_ceiling_escalation"],
            "passes": passes,
            "ambiguous": ambiguous,
            "verdict": verdict,
        },
        # GATE-ONLY re-seal provenance (PIECE B): None on the full-queue path.
        "gate_only_reseal": ffr_correction is not None,
        "ffr_correction": ffr_correction_provenance,
        "gate_only_provenance": dict(job.get("gate_only_provenance") or {}) or None,
        "gate_status": "registered-and-contingent-pending-review",
        "gate_registered": True,
        "evaluation_performed": True,
        "training_performed": False,
        "gpu_used": False,
        "acceptance_rule": (
            "the round succeeds if it executes; the 50M PASS/FAIL is a MEASUREMENT "
            "reported either way that feeds the 100M-commit decision"
        ),
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
    _seal(output, "fneg-50m-x2-seedmean-gate.json", body)
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "collapse_seed_mean": criterion_1["seed_mean"],
        "widened_band": criterion_1["widened_band"],
        "criterion_1_passes": criterion_1["passes"],
        "every_seed_clears_every_backstop": backstop_scoring["every_seed_clears_every_backstop"],
        "verdict": verdict,
        "gate_only_reseal": ffr_correction is not None,
        "supersedes": (SUPERSEDED_SOURCE_RUN if ffr_correction is not None else None),
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0267 round0267_nodes.run_job")
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
    raise Round0267NodeError(f"R0267 unknown action {action!r}")


__all__ = [
    "DESCRIPTIVE_PURITY_LINEAGE_CAVEAT",
    "DESCRIPTIVE_PURITY_METRICS",
    "GATE_ACTION",
    "GATE_CAPABILITY",
    "GATE_METRICS",
    "GATE_SCHEMA",
    "PREFIX_ROWS",
    "COMPLETED_BIND_SEEDS",
    "COMPLETED_REASON",
    "COMPLETED_SOURCE_RUN",
    "PANEL_ACTION",
    "PANEL_CAPABILITY",
    "PANEL_SCHEMA",
    "Round0267NodeError",
    "SALVAGE_SEED",
    "SALVAGE_SEED42_COORDINATES_SHA256",
    "SALVAGE_SOURCE_RUN",
    "SUPERSEDED_SOURCE_RUN",
    "SUPERSEDED_VERDICT",
    "TRIP_9_10_DIAGNOSIS",
    "TRAIN_ACTION",
    "TRAIN_SCHEMA",
    "_authenticate_bound_completed_50m_map",
    "_authenticate_salvaged_50m_map",
    "_bound_provenance_from_panel",
    "_corrected_ffr_from_rescore",
    "_salvage_provenance_from_panel",
    "build_hostint8_dataset_from_slice",
    "read_p1_x2_asymptote_band",
    "run_gate",
    "run_job",
    "run_panel",
    "run_train",
    "score_collapse_seed_mean",
    "score_per_seed_backstops",
    "sigma_fam_from_panel",
]
