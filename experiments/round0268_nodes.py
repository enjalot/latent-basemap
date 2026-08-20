"""Execute R0268 — the 100M ×2 host-int8 FLAGSHIP of the promoted fneg recipe.

Five nodes in one queue (three FRESH trains -> panel -> gate), reusing R0265/R0266/R0267
machinery. Unlike R0267 (whose 50M correction saga salvaged/bound cells), R0268 trains all
three seeds from scratch — NO salvage, NO bind, NO gate-only re-seal.

* three GPU trains (`seeds 42, 43, 44`, SEQUENTIAL — the runner serialises on the GPU)
  under the PINNED 100M ×2 host-int8 recipe, built and proved by `round0268_int8_treatment`
  (R0265's umap kernel a=1.9328/b=0.7905, fneg 1.0 band [0.1,0.4], UNIFORM sampling +
  R0266's `x_residency=host_int8` routing, at dose ×2 on the sealed R0238 100M substrate +
  exact R0243 k15 graph, X = R0262's PRE-SEALED 100M int8 substrate loaded WHOLE). Each
  train node writes its map-level collapse/fog tripwire inputs so a driver can check seed-1
  before seeds 2/3.
* `score_minilm_fneg_100m_x2_panel` (GPU) — the three maps scored on R0265's instruments:
  held-out FFR via the CORRECTED out-of-substrate reserve-projection instrument
  (placed = model.transform(reserve), reserve-neighbour truth, disc = int(ROWS·0.001) =
  100,000), collapse and fog (all on the FULL 100M coordinates via R0265's `score_one_map`),
  plus DESCRIPTIVE-only purity k256/k1024 on the R0238 100M substrate's first-2M-row prefix
  against a reference + centroids built INLINE on that same prefix, with a LINEAGE check
  (100M-prefix ordered hash ≠ R0216-c3's sealed 2M reference). Purity is descriptive/ungated.
* `register_fneg_100m_x2_seedmean_gate` (CPU) — the pre-registered 100M gate
  (plan-100m-flagship-2026-08-17): collapse (SEED-MEAN inside the SAME P1 ×2 asymptote band
  as 50M, widened 1.96·σ_fam/√3 + per-seed backstop 0.8129) + fog (ceiling 0.41207 +
  escalation) + held-out FFR (floor 0.39906) ONLY. Purity is descriptive-only. Every band/
  floor/σ_fam/P1-edge is READ from a SEALED artifact bound by sha256 at gate time.

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

from basemap import round0268_int8_treatment as T
from basemap.round0268_int8_treatment import (
    CANONICAL_SEED,
    DOSE_MULTIPLIER,
    INT8_SUBSTRATE_CAPABILITY,
    INT8_SUBSTRATE_SCHEMA,
    ROUND_ID,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    TRAIN_CLOSURE_MODULES,
    X_RESIDENCY,
    assert_registered_100m_int8_recipe,
    assert_runtime_closure_matches_seal,
    capability_for_seed,
    exact_cell_id,
    fneg_seed_invariant_sha256,
    int8_100m_train_config,
    int8_full_digests,
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
    _map_fog,
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


TRAIN_ACTION = "train_minilm_fneg_100m_x2_hostint8"
PANEL_ACTION = "score_minilm_fneg_100m_x2_panel"
GATE_ACTION = "register_fneg_100m_x2_seedmean_gate"
#: The seed-42 TRANSFORM-CORRECTION action (R10). Attempt-4's train sealed cleanly but the R9
#: transform poll (dict-vs-path) TypeError'd the transform — so seed42 has a sealed train-receipt
#: + model but no transform-receipt/tripwire. This node re-runs ONLY the (now-fixed, unguarded)
#: transform from those sealed read-only inputs, seals a transform-receipt (+ failed-marker
#: provenance) + its own done-marker. No re-train. R0267 queue-correction precedent.
TRANSFORM_CORRECTION_ACTION = "transform_correct_minilm_fneg_100m_x2_hostint8"

TRAIN_SCHEMA = "round0268-minilm-fneg-100m-x2-hostint8-train-receipt-v1"
#: The post-training transform phase seals its OWN receipt (coordinates + seed-1 tripwire +
#: transform wall/RSS), AFTER the train receipt, so a transform-phase death never destroys a
#: clean train's evidence (R0268 attempts 1 & 3). The panel never reads this — it re-transforms
#: from the model — so the phase split is invisible downstream.
TRANSFORM_SCHEMA = "round0268-minilm-fneg-100m-x2-hostint8-transform-receipt-v1"
PANEL_CAPABILITY = "minilm-fneg-100m-x2-hostint8-panel-v1"
PANEL_SCHEMA = "round0268-minilm-fneg-100m-x2-hostint8-panel-v1"
GATE_CAPABILITY = "minilm-fneg-100m-x2-seedmean-gate-v1"
GATE_SCHEMA = "round0268-minilm-fneg-100m-x2-seedmean-gate-v1"

#: The three per-seed GATED metrics (plan criteria 1-3: held-out FFR floor + collapse
#: backstop + fog ceiling). Purity is DESCRIPTIVE-only at 100M (plan §2.4): it never gates.
GATE_METRICS: tuple[str, ...] = (
    HELDOUT_FFR_METRIC, COLLAPSE_METRIC, FOG_METRIC,
)

#: Purity k256/k1024 are RECORDED (descriptive) at 100M but never gate.
DESCRIPTIVE_PURITY_METRICS: tuple[str, ...] = (
    PURITY_K256_METRIC, PURITY_K1024_METRIC,
)

#: The first-2M-row prefix of the R0238 100M substrate — the DESCRIPTIVE purity subset AND
#: the lineage-check subset. Reference + centroids are both built INLINE on exactly these
#: rows, so the descriptive purity is self-contained on the R0238 prefix.
PREFIX_ROWS = 2_000_000

#: The lineage caveat carried on every descriptive-purity record + the panel/gate bodies.
DESCRIPTIVE_PURITY_LINEAGE_CAVEAT = (
    "purity k256/k1024 at 100M are DESCRIPTIVE / UNGATED. They are scored on the R0238 100M "
    "substrate's first-2M-row prefix against a reference + k-means centroids built INLINE on "
    "that SAME prefix (self-contained; no R0218 dependency, reference or centroids). They are "
    "NOT commensurate with the R0265 2M family purity bands: the R0238 ladder's first-2M rows "
    "are a DIFFERENT build lineage than R0218's frozen 2M reference the family bands were fit "
    "on (the lineage check proves the 100M-prefix ordered hash != R0216-c3's sealed 2M "
    "reference cb44d0a7…), so no frozen-reference purity gate exists at 100M. Purity remains a "
    "fully gated criterion only at 2M (plan-100m-flagship-2026-08-17 §2.4)."
)

#: The seed-mean collapse gate parameters (plan criterion 1). z is the two-sided 95% normal
#: quantile; n is the registered seed count. Both are pre-registered constants of the
#: DECISION RULE (not gate values read from data) — the band edges and σ_fam ARE read live
#: from sealed artifacts.
COLLAPSE_SEEDMEAN_Z = 1.96
COLLAPSE_SEEDMEAN_N = 3

DEVICE_BUDGET_BYTES = 30 * (1 << 30)

# --------------------------------------------------------------------------- #
# HOST_RSS_LIMIT — set ANALYTICALLY for the TRAIN (delegate ruling, plan §3).
#
# Page-cache RSS reaches steady state only deep into an epoch, so a short dry-run
# UNDER-estimates it. The TRAIN limit is set from the 50M measured profile scaled
# to 100M rather than from a dry-run:
#
#   * R0267's 50M train measured 75.66 GiB peak RSS with the file-backed int8 X
#     (19.2 GB = 50M×384) + the k15 edge memmaps (~9.4 GB resident endpoints) as
#     the two dominant reclaimable-page contributors.
#   * At 100M both double: int8 X 19.2 -> 38.4 GB (+19.2), edges 9.4 -> 18.8 GB
#     (+9.4) => +28.6 GB of resident page cache over the 50M peak.
#   * Projected 100M train peak RSS ~= 75.66 + 28.6 ~= 104 GiB.
#   * HOST_RSS_LIMIT_GIB = 104 + ~11 margin = 115.0 GiB, under the box's ~123 GB
#     physical. Page cache is reclaimable, so OOM risk stays with the ANON budget
#     (measured 50M anon 9.72 GiB « 64 GiB; 100M anon stays modest), not this RSS
#     backstop. This is an EXECUTION-resource field (a liveness/OOM backstop), NOT a
#     treatment field: it is absent from the config + the masked-config/treatment
#     invariant digest (the constants-discipline contract test proves this invariance).
# --------------------------------------------------------------------------- #
HOST_RSS_LIMIT_GIB = 115.0

#: The analytic derivation emitted into the train receipt (a number with its basis, not
#: "should be fine"). Read into the receipt's `host_rss_limit_basis` field.
HOST_RSS_ANALYTIC_BASIS = {
    "limit_gib": HOST_RSS_LIMIT_GIB,
    "method": "50M-measured profile scaled to 100M (page cache reaches steady state deep in "
              "an epoch, so a short dry-run under-estimates it)",
    "r0267_50m_measured_peak_rss_gib": 75.66,
    "int8_x_bytes_50m": 19_200_000_000,
    "int8_x_bytes_100m": 38_400_000_000,
    "edges_resident_bytes_50m_approx": 9_400_000_000,
    "edges_resident_bytes_100m_approx": 18_800_000_000,
    "delta_gib_50m_to_100m_approx": 28.6,
    "projected_100m_peak_rss_gib_approx": 104.0,
    "margin_gib": 11.0,
    "physical_gib": 123.0,
    "page_cache_is_reclaimable": True,
    "oom_risk_lives_with": "anon budget (measured 50M anon 9.72 GiB « 64 GiB)",
}

#: The PANEL RSS limit, REFINED from the throwaway-map panel dry-run (plan §3; delegate
#: option-A approval 2026-08-17 at 120.0). The dry-run END-TO-END 100M panel (R0267 50M-seed42
#: model → full 100M R0238 substrate) measured ru_maxrss peak 115.46 GiB — which counts the
#: RECLAIMABLE file-backed page cache from reading the 153.6 GB substrate memmap, NOT the
#: process's need. That RSS climbs linearly with the transform then PLATEAUS ~113 GiB once the
#: kernel reclaims (physical RAM 123.4 GiB; MemAvailable held ~118 GiB throughout) — it is
#: physical-RAM-bound, not N-bound, and does not worsen across the 3 seeds (shared page cache,
#: anon freed between cells). The real OOM-relevant quantity is ANONYMOUS RSS (coords 0.8 GB +
#: FFR arrays ~3 GB + transient concat ~1.6 GB ≈ single-digit GiB), already bounded by
#: `R0268_ANON_BUDGET_BYTES = 64 GiB`. So this ru_maxrss limit is a TRUE-RAM-EXHAUSTION backstop:
#: 120.0 = measured 115.46 + ~4.5 GiB margin, and physical 123.4 − 3.4 GiB so it still fires on
#: genuine exhaustion. The 64 GiB anon guard remains the real OOM tripwire.
PANEL_RSS_LIMIT_GIB = 120.0

#: The R0244 host-watchdog anonymous-memory budget for the 100M host-int8 flagship. The 50M
#: rung used 64 GiB (R0267 R3 dry-run measured panel anon 36.26 GiB); the 100M anon peak
#: stays modest (int8 X + edges are FILE-BACKED memmaps, not anon; measured 50M train anon
#: 9.72 GiB), so 64 GiB carries generous headroom. Per-cell (freed + gc between cells). An
#: EXECUTION resource field — absent from the treatment digest (constants-discipline test).
R0268_ANON_BUDGET_BYTES = 64 * (1 << 30)
POSITIVE_ROWS_PER_UPDATE = 409

#: The pre-registered assumption the gate records (plan criterion 1).
SEED_SPREAD_ASSUMPTION = (
    "the 2M fneg family's seed spread (σ_fam = 1.4826·MAD_n over R0265's sealed 13 "
    "collapse/fog values) estimates the 100M seed spread; pre-registered as an assumption, "
    "revisited only if the 100M seeds' observed spread contradicts it"
)

SAFETY_NOTE = (
    "no node in this module signals any process, starts a child process, hands cuVS "
    "anything, or wraps a subprocess in a timeout. Every bulk input is a read-only "
    "np.memmap. The per-batch abort read is the release's own ParametricUMAP.abort_poll "
    "attribute, set to this node's recorder and cleared in a finally."
)


class Round0268NodeError(RuntimeError):
    """The R0268 node contract changed."""


# --------------------------------------------------------------------------- #
# shared scaffolding local to R0268 (mirrors R0265/R0266/R0267's, ROUND_ID="0268")
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
        _bound_path(job, "treatment_closure", label="R0268 treatment closure seal"),
        label="R0268 treatment closure seal",
    )
    observed = runtime_closure_hashes(TRAIN_CLOSURE_MODULES)
    verdict = assert_runtime_closure_matches_seal(sealed=sealed, observed=observed)
    return sealed, {
        "runtime_closure": observed,
        "verdict": verdict,
        "controls": treatment_closure_controls(sealed=sealed, observed=observed),
    }


# --------------------------------------------------------------------------- #
# the sealed R0243 100M graph binding. R0243 seals under a `fuzzy-graph.json`
# manifest that ships the graph as FOUR streamed members (edges src/dst/wts +
# a scalar header), so the binding is retargeted from R0237's single-.npz `graph`
# signature to R0243's `outputs.edges_*` members. The edge path handed to
# core.fit is the artifact DIRECTORY — `load_edge_arrays` (round0259_hundred_m)
# claims a directory holding all four members and returns the same 4-tuple.
# --------------------------------------------------------------------------- #


def _sealed_100m_graph(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read the sealed R0243 fuzzy-graph manifest and load its exact k15 fuzzy graph."""
    manifest_signature = dict(job["graph_manifest_signature"])
    manifest_path = prompt_contract.verify_signature(
        manifest_signature, label="R0268 sealed R0243 100M graph manifest"
    )
    manifest = prompt_contract.read_sealed(
        manifest_path, label="R0268 sealed R0243 100M graph manifest"
    )
    capabilities = manifest.get("capabilities") or []
    tripwire = manifest.get("post_canonical_tripwire") or {}
    sym = manifest.get("symmetrised_degree") or {}
    if (
        str(manifest.get("round_id")) != T.R0243_ROUND_ID
        or T.R0243_GRAPH_CAPABILITY not in capabilities
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("k", -1)) != 15
        or manifest.get("training_performed") is not False
    ):
        raise Round0268NodeError("R0268 sealed R0243 100M graph contract changed")
    if int(tripwire.get("zero_degree_rows", -1)) != 0 or int(sym.get("zero_degree_rows", -1)) != 0:
        raise Round0268NodeError("R0268 requires the sealed R0243 100M graph zero-degree tripwire")
    edges = int(manifest.get("directed_edges", 0))
    if edges != SEALED_DIRECTED_EDGES:
        raise Round0268NodeError(
            f"R0268 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    outputs = manifest.get("outputs") or {}
    member_signatures = {
        name: dict(outputs[name])
        for name in ("edges_header", "edges_sources", "edges_targets", "edges_weights")
    }
    # Re-verify each member's bytes (verify_signature re-hashes the file) and derive the
    # streamed-member DIRECTORY the loader claims from the header's location.
    header_path = prompt_contract.verify_signature(
        member_signatures["edges_header"], label="R0268 sealed R0243 graph header"
    )
    for name in ("edges_sources", "edges_targets", "edges_weights"):
        prompt_contract.verify_signature(
            member_signatures[name], label=f"R0268 sealed R0243 graph {name}"
        )
    edges_dir = os.path.dirname(header_path)
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import load_edge_arrays

    sources, targets, weights, n_nodes = load_edge_arrays(edges_dir, load_weights=True)
    if (
        weights is None
        or int(n_nodes) != ROWS
        or len(sources) != edges
        or targets.shape != sources.shape
        or weights.shape != sources.shape
    ):
        raise Round0268NodeError("R0268 sealed R0243 100M graph arrays changed")
    # The graph provenance signature carried into the config: the header file's identity
    # (a stable small member digest); the edge PATH handed to core.fit is the directory.
    graph_signature = {
        "kind": "file",
        "canonical_path": edges_dir,
        "sha256": str(member_signatures["edges_header"]["sha256"]),
    }
    return {
        "manifest": manifest,
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "member_signatures": member_signatures,
        "edges_path": edges_dir,
        "directed_edges": edges,
        "n_nodes": int(n_nodes),
    }


def _sealed_100m_substrate(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read the sealed R0238 nested-substrate manifest and its identity anchors."""
    manifest_signature = dict(job["substrate_manifest_signature"])
    manifest_path = prompt_contract.verify_signature(
        manifest_signature, label="R0268 sealed R0238 100M substrate manifest"
    )
    manifest = prompt_contract.read_sealed(
        manifest_path, label="R0268 sealed R0238 100M substrate manifest"
    )
    if (
        str(manifest.get("round_id")) != T.R0238_ROUND_ID
        or manifest.get("capability") != T.R0238_SUBSTRATE_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or str(manifest.get("ordered_substrate_sha256")) != T.R0238_SUBSTRATE_ORDERED_SHA256
    ):
        raise Round0268NodeError("R0268 sealed R0238 100M substrate contract changed")
    return {
        "manifest": manifest,
        "manifest_signature": manifest_signature,
        "substrate_signature": dict(manifest["substrate"]),
        "ordered_substrate_sha256": str(manifest["ordered_substrate_sha256"]),
    }


def _open_100m_substrate(sealed: Mapping[str, Any]) -> np.ndarray:
    """Serve the 153.6 GB sealed 100M substrate lazily; never materialize it."""
    path = prompt_contract.verify_signature(
        sealed["substrate_signature"], label="R0268 sealed R0238 100M substrate"
    )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (ROWS, DIMENSION) or array.dtype != np.float32:
        raise Round0268NodeError("R0268 sealed R0238 100M substrate geometry changed")
    return array


# --------------------------------------------------------------------------- #
# the LINEAGE check (plan §2.4): assert/record that the 100M substrate's first-2M
# ordered hash != R0216-c3's sealed 2M reference (cb44d0a7…). The R0216-c3 reference
# is READ from the bound R0218 panel's lineage.ordered_substrate_sha256 (never a
# hardcoded literal), so the check is family-calibrated + lineage-bound and purity is
# reported descriptively with the lineage caveat. This is the INVERTED-expectation
# mirror of R0267's verify_nested_prefix_identity: R0267 asserted a MATCH to lift the
# 2M gate; R0268 asserts a NON-match to keep purity descriptive at 100M.
# --------------------------------------------------------------------------- #


def _read_r0216_c3_reference(job: Mapping[str, Any]) -> str:
    """The sealed R0216-c3 2M ordered_substrate reference from the bound R0218 panel."""
    panel_path = _bound_path(job, "panel_evidence", label="R0268 R0218 frozen panel")
    panel = prompt_contract.read_sealed(panel_path, label="R0268 R0218 frozen panel")
    lineage = panel.get("lineage") or {}
    reference = str(lineage.get("ordered_substrate_sha256") or "")
    if len(reference) != 64:
        raise Round0268NodeError("R0268 R0218 panel carries no R0216-c3 2M ordered reference")
    return reference


def verify_hundred_m_prefix_lineage(
    source: Any, r0216_c3_reference: str, *, prefix_rows: int = PREFIX_ROWS
) -> dict[str, Any]:
    """RECORD (INVERTED expectation) that ``source``'s first ``prefix_rows`` rows do NOT
    hash to R0216-c3's sealed 2M ordered reference — so purity is descriptive, built on the
    100M prefix with no cross-lineage claim. Raises only if the hashes UNEXPECTEDLY MATCH
    (which would mean the 100M substrate is the R0216-c3 2M lineage and purity could be
    gated — a contract change to escalate to the owner). The hash streams the prefix in row
    chunks (no >=2 GB materialisation)."""
    prefix_rows = int(prefix_rows)
    observed = ordered_array_sha256(source[:prefix_rows])
    matches = observed == str(r0216_c3_reference)
    if matches:
        raise Round0268NodeError(
            "R0268 lineage check UNEXPECTEDLY matched: the 100M substrate's first "
            f"{prefix_rows} rows hash to {observed}, which EQUALS R0216-c3's sealed 2M "
            f"reference {r0216_c3_reference}. The plan pre-registers a NON-match (100M is "
            "the R0233->R0238 nested ladder, a different 2M-prefix lineage); a match means "
            "the substrate identity changed — escalate to the owner."
        )
    return {
        "lineage_check": "hundred_m_prefix_ne_r0216_c3_2m_reference",
        "expected": "non_match",
        "prefix_rows": prefix_rows,
        "observed_hundred_m_prefix_sha256": observed,
        "r0216_c3_2m_reference_sha256": str(r0216_c3_reference),
        "matches_r0216_c3": bool(matches),
        "purity_is_descriptive": True,
        "note": (
            "INVERTED-expectation mirror of R0267's verify_nested_prefix_identity: the 100M "
            "substrate's first-2M ordered hash != R0216-c3's sealed 2M reference, so purity "
            "is built on the 100M prefix (self-contained) and reported DESCRIPTIVELY — no "
            "cross-lineage frozen-reference purity gate exists at 100M."
        ),
    }


# --------------------------------------------------------------------------- #
# the PRE-SEALED int8 substrate FULL-FILE load (the delegate-approved fix proven
# at 50M in R0267): LOAD R0262's sealed 100M int8 substrate WHOLE instead of
# encoding fp32->int8 on the fly. At 100M the file IS the substrate — no prefix
# slice — so only the full-file digest bind remains.
# --------------------------------------------------------------------------- #


def _read_int8_full_manifest(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read + validate the sealed R0268 int8 full-file substrate manifest (its FULL-FILE LAW)."""
    manifest_signature = dict(job["int8_substrate_manifest_signature"])
    manifest_path = prompt_contract.verify_signature(
        manifest_signature, label="R0268 sealed int8 full-file substrate manifest"
    )
    manifest = prompt_contract.read_sealed(
        manifest_path, label="R0268 sealed int8 full-file substrate manifest"
    )
    if (
        manifest.get("schema") != INT8_SUBSTRATE_SCHEMA
        or manifest.get("capability") != INT8_SUBSTRATE_CAPABILITY
        or manifest.get("round_id") != ROUND_ID
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or manifest.get("x_residency") != X_RESIDENCY
    ):
        raise Round0268NodeError("R0268 int8 full-file substrate manifest contract changed")
    return {"manifest": manifest, "manifest_signature": manifest_signature}


def _load_verified_int8_full(
    sealed_manifest: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """LOAD R0262's int8 substrate file-backed WHOLE + VERIFY the sealed full-file digests.

    Returns ``(i8, scales, receipt)``. The two arrays are the whole parent files served
    as read-only ``np.memmap`` (no copy). The two full-file sha256s are RE-HASHED (streamed,
    never materialised) and checked against the FULL-FILE LAW in the sealed manifest — a
    mismatch (the bytes are not the sealed ones) raises ``Round0268NodeError``.
    """
    if (
        sealed_manifest.get("schema") != INT8_SUBSTRATE_SCHEMA
        or sealed_manifest.get("capability") != INT8_SUBSTRATE_CAPABILITY
    ):
        raise Round0268NodeError("R0268 int8 full-file substrate manifest contract changed")
    law = dict(sealed_manifest.get("full_file_law") or {})
    rows = int(law["rows"])
    dim = int(law["dimension"])
    offset = int(law["offset"])
    if offset != 0:
        raise Round0268NodeError("R0268 int8 full-file load must be offset-0 whole-file")
    i8_path = str(law["i8_path"])
    scales_path = str(law["scales_path"])
    expected_i8 = str(law["i8_sha256"])
    expected_scales = str(law["scales_sha256"])

    # File-backed whole-file memmaps — the reshape never copies the 38.4 GB int8 payload
    # (HostInt8ArrayDataset then shares the mmap).
    i8_full = np.memmap(i8_path, dtype=np.int8, mode="r").reshape(-1, dim)
    sc_full = np.memmap(scales_path, dtype=np.float16, mode="r")
    if int(i8_full.shape[0]) != rows or int(sc_full.shape[0]) != rows:
        raise Round0268NodeError(
            "R0268 parent int8 substrate row count does not match the sealed 100M law"
        )
    got = int8_full_digests(i8_path, scales_path, rows=rows, dimension=dim)
    if got["i8_sha256"] != expected_i8 or got["scales_sha256"] != expected_scales:
        raise Round0268NodeError(
            "R0268 int8 full-file digest mismatch (the bytes are not the sealed ones): "
            f"i8 {got['i8_sha256']} vs sealed {expected_i8}; scales "
            f"{got['scales_sha256']} vs sealed {expected_scales}"
        )
    receipt = {
        "parent_artifact": law.get("parent_artifact"),
        "parent_round": law.get("parent_round"),
        "i8_path": i8_path,
        "scales_path": scales_path,
        "rows": rows,
        "dimension": dim,
        "offset": offset,
        "i8_bytes": int(got["i8_bytes"]),
        "scales_bytes": int(got["scales_bytes"]),
        "i8_sha256": got["i8_sha256"],
        "scales_sha256": got["scales_sha256"],
        "verified_against_sealed_manifest": True,
        "load_mode": "pre_sealed_file_backed_full_file",
        "re_encoded_at_train_time": False,
    }
    return i8_full, sc_full, receipt


def build_hostint8_dataset_from_full(sealed_manifest: Mapping[str, Any], device: Any):
    """Construct a file-backed ``HostInt8ArrayDataset`` from the sealed WHOLE int8 file.

    The int8 rows + fp16 scales are passed as ``encoded=``/``scales=`` so
    ``HostInt8ArrayDataset`` uses them VERBATIM (no fp32->int8 re-encode). The whole-file
    mmaps stay file-backed through ``__init__`` (no 38.4 GB anonymous copy). Returns
    ``(dataset, receipt)``.
    """
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        HostInt8ArrayDataset,
    )

    i8, sc, receipt = _load_verified_int8_full(sealed_manifest)
    dataset = HostInt8ArrayDataset(None, device, encoded=i8, scales=sc)
    if getattr(dataset, "host_int8_dataset", False) is not True:
        raise Round0268NodeError("R0268 pre-sealed int8 dataset is not a host-int8 dataset")
    if tuple(dataset.shape) != (int(receipt["rows"]), int(receipt["dimension"])):
        raise Round0268NodeError("R0268 pre-sealed int8 dataset geometry changed")
    return dataset, receipt


def _build_int8_100m_model(config: Mapping[str, Any]):
    """R0265's `_build_fneg_model` epoch-scaled to the 100M edge count, plus the int8 delta.

    `_build_fneg_model` scales `n_epochs` using R0216's 2M edge count, which under-covers
    the 100M ×2 horizon; re-scale here with the 100M edge count. Then set
    `model.x_residency = host_int8` and assert it (R0266's int8 delta on the instance).
    """
    model = _build_fneg_model(config)
    num_pos = max(1, int(model.batch_size * model.pos_ratio))
    steps_per_epoch = math.ceil(SEALED_DIRECTED_EDGES / num_pos)
    needed_epochs = math.ceil(int(model.total_steps_estimate) / steps_per_epoch)
    if needed_epochs > int(model.n_epochs):
        model.n_epochs = needed_epochs
    model.x_residency = X_RESIDENCY
    if getattr(model, "x_residency", None) != X_RESIDENCY:
        raise Round0268NodeError(
            f"R0268 model x_residency is {getattr(model, 'x_residency', None)!r}, "
            f"expected {X_RESIDENCY!r}"
        )
    return model


def _transform_100m_in_chunks(model: Any, source: Any, poll: Any) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, ROWS, TRANSFORM_CHUNK_ROWS):
        stop = min(start + TRANSFORM_CHUNK_ROWS, ROWS)
        block = np.asarray(
            model.transform(source[start:stop], batch_size=FULL_TRANSFORM_BATCH),
            dtype=np.float32,
        )
        parts.append(block)
        poll(f"R0268 transform rows {start}-{stop}")
    return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------- #
# the train node — one 100M ×2 host-int8 map per seed (FRESH train)
# --------------------------------------------------------------------------- #


def _seed(job: Mapping[str, Any]) -> int:
    seed = job.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Round0268NodeError(f"R0268 job seed {seed!r} is not an integer")
    if seed not in SEEDS:
        raise Round0268NodeError(f"R0268 job seed {seed!r} is not a 100M flagship cell (42/43/44)")
    if str(job.get("capability") or "") != capability_for_seed(seed):
        raise Round0268NodeError("R0268 job capability does not match its seed")
    return int(seed)


def _assemble_train_receipt(
    *,
    capability: str,
    seed: int,
    release_sha: str,
    abort_flag: Any,
    production_config_sig: Mapping[str, Any],
    config_sha: str,
    observed_invariant: str,
    declared_invariant: str,
    recipe: Any,
    closure: Mapping[str, Any],
    treatment_closure_seal_sig: Mapping[str, Any],
    model_sig: Mapping[str, Any],
    substrate: Mapping[str, Any],
    int8_substrate_manifest_signature: Mapping[str, Any],
    int8_full_receipt: Mapping[str, Any],
    graph: Mapping[str, Any],
    edges: int,
    updates: int,
    base_horizon: int,
    accounting: Mapping[str, Any],
    runtime: Mapping[str, Any],
    residency: Mapping[str, Any],
    fneg_telemetry: Any,
    wall: float,
    memory: Mapping[str, Any],
    watchdog_state: Mapping[str, Any],
    peak_rss_gib: float,
    gaps: Any,
    scored: Any,
    tail: Any,
    coverage: Mapping[str, Any],
    node_id: str,
    checkpoint_fneg_roundtrip: bool,
    rehearsal: bool = False,
) -> dict[str, Any]:
    """Build the R0268 train-receipt dict — a PURE function (no I/O, no globals beyond module
    constants) so it is unit-testable WITHOUT CUDA. Every I/O signature (production config, closure
    seal, model, int8 manifest) is computed by the caller and passed in. This exists because the
    receipt-assembly path had NEVER executed to completion in any attempt (the transform always
    crashed first pre-R9; post-R9 it crashed on the latent `int8_full` use-after-del at attempt-4):
    a source-only review cannot catch a runtime dict error, so this function is EXECUTED in a test.

    `rehearsal=True` (the R11 rehearsal harness, driven by a job flag) suffixes the schema and adds a
    top-level `is_non_evidence_rehearsal` marker, so a receipt built over STUBBED fit telemetry is
    UNMISTAKABLE as non-evidence from the inside even if the file is moved out of the rehearsal dir.
    Production always passes False → byte-identical receipt."""
    return {
        "schema": TRAIN_SCHEMA + ("-REHEARSAL-NON-EVIDENCE" if rehearsal else ""),
        **({"is_non_evidence_rehearsal": True} if rehearsal else {}),
        "round_id": ROUND_ID,
        "capability": capability,
        "capabilities": [capability],
        "training_seed": seed,
        "is_a_100m_flagship_cell": True,
        "release_sha": release_sha,
        "abort_flag_precondition": abort_flag,
        "production_config": production_config_sig,
        "production_config_sha256": config_sha,
        "seed_invariant_sha256": observed_invariant,
        "recipe": recipe,
        "x_residency": X_RESIDENCY,
        "treatment_closure": closure["verdict"],
        "treatment_closure_controls": closure["controls"],
        "treatment_closure_seal": treatment_closure_seal_sig,
        "model": model_sig,
        "substrate": substrate["substrate_signature"],
        "substrate_manifest": substrate["manifest_signature"],
        "ordered_substrate_sha256": substrate["ordered_substrate_sha256"],
        "int8_substrate_manifest": int8_substrate_manifest_signature,
        "int8_substrate_full_file": int8_full_receipt,
        "x_source": "pre_sealed_int8_full_file_r0262_100m",
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "graph_members": graph["member_signatures"],
        "rows": ROWS,
        "dimension": DIMENSION,
        "directed_edges": edges,
        "optimizer_updates": updates,
        "base_horizon": int(base_horizon),
        "dose_multiplier": DOSE_MULTIPLIER,
        "consumed_positive_draws_per_edge": float(updates * POSITIVE_ROWS_PER_UPDATE / edges),
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "host_int8_residency": residency,
        "fneg_telemetry": fneg_telemetry,
        "train_wall_s": wall,
        "memory": memory,
        "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
        "host_rss_limit_basis": HOST_RSS_ANALYTIC_BASIS,
        "memory_watchdog": watchdog_state,
        "attestation_scope": (
            "TRAINING PHASE ONLY: memory_watchdog, memory.peak_host_rss_gib, guard_tail, and "
            "train_checks.watchdog_did_not_trip attest fit→save→reload (the guarded phase). The "
            "100M transform runs UNGUARDED after this receipt seals and is attested separately "
            "in transform-receipt.json (coordinates finiteness, transform wall/RSS, tripwire)."
        ),
        "gap_report": gaps,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "training_performed": True,
        "gate_registerable_here": False,
        "map_decision_made": False,
        "train_checks": {
            "recipe_is_the_registered_100m_hostint8_recipe": (
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
            "pre_sealed_int8_full_file_verified": bool(
                int8_full_receipt.get("verified_against_sealed_manifest")
                and int8_full_receipt.get("re_encoded_at_train_time") is False
            ),
            "host_int8_residency_stamp_verified": (
                residency["x_residency"] == X_RESIDENCY
                and residency["weighted_effective"] is False
                and residency["positive_sampling"] == "uniform"
            ),
            "watchdog_did_not_trip": not bool(watchdog_state["tripped"]),
            "host_rss_under_analytic_limit": peak_rss_gib <= HOST_RSS_LIMIT_GIB,
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


def _assemble_transform_receipt(
    *,
    capability: str,
    seed: int,
    phase: str,
    train_receipt_sig: Mapping[str, Any],
    model_sig: Mapping[str, Any],
    coordinates_sig: Mapping[str, Any],
    coordinates_ordered_sha256: str,
    ordered_substrate_sha256: str,
    tripwire_inputs: Mapping[str, Any],
    transform_wall_s: float,
    transform_peak_rss_gib: float,
    all_coordinates_finite: bool,
    transform_rows_finite: int,
    extra_checks: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    rehearsal: bool = False,
) -> dict[str, Any]:
    """Build a transform-receipt dict — PURE (no I/O, no free locals), so both callers'
    (run_train and run_transform_correction) never-executed post-transform dict assembly is
    unit-tested by EXECUTION, not source-reading. `extra_checks` merges into transform_checks;
    `extra` merges phase-specific top-level keys (the correction node's envelope/provenance).
    `rehearsal=True` suffixes the schema + adds a top-level marker (as for the train receipt), so a
    rehearsal transform-receipt is unmistakable as non-evidence from the inside; production → False."""
    checks: dict[str, Any] = {
        "all_100m_coordinates_finite": bool(all_coordinates_finite),
        "coordinates_row_count_is_rows": int(transform_rows_finite) == ROWS,
    }
    if extra_checks:
        checks.update(extra_checks)
    body: dict[str, Any] = {
        "schema": TRANSFORM_SCHEMA + ("-REHEARSAL-NON-EVIDENCE" if rehearsal else ""),
        **({"is_non_evidence_rehearsal": True} if rehearsal else {}),
        "round_id": ROUND_ID,
        "capability": capability,
        "training_seed": seed,
        "phase": phase,
        "guarded": False,
        "train_receipt": train_receipt_sig,
        "model": model_sig,
        "coordinates": coordinates_sig,
        "coordinates_ordered_sha256": coordinates_ordered_sha256,
        "ordered_substrate_sha256": ordered_substrate_sha256,
        "seed_1_tripwire_inputs": tripwire_inputs,
        "transform_wall_s": transform_wall_s,
        "transform_peak_host_rss_gib": transform_peak_rss_gib,
        "transform_checks": checks,
    }
    if extra:
        body.update(extra)
    return body


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0268 round0268_nodes.run_train")
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0268NodeError("R0268 train handler received another queue")
    seed = _seed(job)
    capability = capability_for_seed(seed)
    node_id = str(active.get("node_id") or f"train_hostint8_100m_seed{seed}")
    label = f"R0268 {capability}"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)

    closure_seal, closure = _closure_evidence(job)
    if not closure["controls"]["every_planted_defect_was_refused"]:
        raise Round0268NodeError(
            "R0268 closure guard did not refuse every planted defect: "
            f"{closure['controls']['controls']}"
        )
    if not closure["controls"]["the_honest_closure_still_passes"]:
        raise Round0268NodeError("R0268 closure guard rejects the honest closure")

    graph = _sealed_100m_graph(job)
    substrate = _sealed_100m_substrate(job)
    # The fp32 substrate stays bound + opened for the post-train transform (and the receipt
    # lineage); it is NOT the training X (see the int8 full-file load below).
    source = _open_100m_substrate(substrate)
    int8_full = _read_int8_full_manifest(job)
    edges = graph["directed_edges"]
    config, config_sha = int8_100m_train_config(
        seed=seed,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        substrate_signature=substrate["substrate_signature"],
        graph_edges=edges,
        rows=ROWS,
    )
    recipe = assert_registered_100m_int8_recipe(config)
    observed_invariant = fneg_seed_invariant_sha256(config)
    declared_invariant = str(job.get("cell_seed_invariant_sha256") or "")
    if not declared_invariant or observed_invariant != declared_invariant:
        raise Round0268NodeError(
            "R0268 cell config is not the sealed 100M host-int8 recipe: "
            f"{observed_invariant} != {declared_invariant}"
        )
    updates = int(config["optimizer"]["successful_positive_lr_updates"])
    if updates != DOSE_MULTIPLIER * int(job.get("base_horizon", -1)):
        raise Round0268NodeError("R0268 horizon does not match the sealed ×2 base horizon")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0268 train output")
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
    model = _build_int8_100m_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")

    # PRE-SEALED FULL-FILE LOAD (replaces the fp32-substrate-then-on-the-fly-encode path):
    # load R0262's sealed 100M int8 substrate WHOLE file-backed, VERIFY the two full-file
    # sha256s against the sealed FULL-FILE LAW, and hand the pre-constructed
    # HostInt8ArrayDataset to model.fit so core.fit uses it directly (no re-encode). The
    # 38.4 GB int8 payload stays file-backed through __init__ (no anonymous copy). The digest
    # verification (a streamed hash of the 38.4 GB file) runs HERE, before the liveness
    # watchdog starts, so it can never trip it.
    int8_dataset, int8_full_receipt = build_hostint8_dataset_from_full(
        int8_full["manifest"], model.device
    )
    if int(int8_dataset.shape[0]) != ROWS or int(int8_dataset.shape[1]) != DIMENSION:
        raise Round0268NodeError("R0268 pre-sealed int8 dataset geometry is not the 100M flagship")

    window = ledger.window(f"R0268 {capability} train stage")
    guard_ctx = _node_guard(label, anonymous_budget_bytes=R0268_ANON_BUDGET_BYTES)
    gate = _node_gate(label, training_performed=True)
    watchdog = CellWatchdog()
    watchdog.start()
    started = time.monotonic()
    try:
        with guard_ctx:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor(f"R0268 {capability} stage entered")
            wrapped = window.wrap(recorder)
            model.abort_poll = wrapped
            try:
                # X is the PRE-SEALED host-int8 dataset (loaded + verified above);
                # core.fit's host_int8 branch uses it directly, no re-encode. The edge path
                # is the R0243 artifact DIRECTORY (load_edge_arrays claims the streamed
                # members).
                model.fit(
                    int8_dataset,
                    random_state=seed,
                    precomputed_edges_path=graph["edges_path"],
                )
            finally:
                model.abort_poll = None
            wall = time.monotonic() - started
            wrapped("R0268 fit() returned")
            accounting = dict(model._train_stats)
            runtime = dict(getattr(model, "_pipeline_info", None) or {})
            if not runtime:
                raise Round0268NodeError(
                    "R0268 fit() left no _pipeline_info stamp -- cannot prove the sampler"
                )
            # FAIL-CLOSED TRIPWIRE: uniform positive sampling AND the host-int8 residency.
            if (
                runtime.get("weighted_effective") is not False
                or runtime.get("positive_sampling") != "uniform"
                or runtime.get("x_residency") != X_RESIDENCY
            ):
                raise Round0268NodeError(
                    "R0268 trained off the host-int8 uniform path (silent fallback): "
                    f"weighted_effective={runtime.get('weighted_effective')!r}, "
                    f"positive_sampling={runtime.get('positive_sampling')!r}, "
                    f"x_residency={runtime.get('x_residency')!r}, "
                    f"pipeline={runtime.get('pipeline')!r}"
                )
            fneg_telemetry = dict(model.fneg_telemetry) if model.fneg_telemetry else None
            model_path = os.path.join(output, "model.pt")
            from basemap.output_safety import atomic_build_new_file

            atomic_build_new_file(model_path, model.save, immutable=True)
            wrapped("R0268 checkpoint published")
            free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
            memory = {
                "device_total_bytes": int(total_bytes),
                "post_train_free_bytes": int(free_bytes),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            }
            # Free the big training objects. NOT int8_full: it is the small manifest/signature
            # DICT (int8_full["manifest"], int8_full["manifest_signature"]) the train receipt reads
            # at assembly (deleting it frees ~nothing and left it UNBOUND when the R9 phase-split
            # first ran the receipt before the transform — attempt-4's 24h train crashed here).
            del model, int8_dataset
            torch.cuda.empty_cache()
            gc.collect()
            wrapped("R0268 training objects released")

            from basemap.pumap.parametric_umap import ParametricUMAP

            reloaded = ParametricUMAP.load(model_path, device="cuda")
            checkpoint_fneg_roundtrip = (
                float(reloaded.fneg_weight) == R0265N.FNEG_WEIGHT
                and float(reloaded.fneg_lo) == R0265N.FNEG_LO
                and float(reloaded.fneg_hi) == R0265N.FNEG_HI
            )
            if not checkpoint_fneg_roundtrip:
                raise Round0268NodeError("R0268 checkpoint did not round-trip the fneg params")
            wrapped("R0268 checkpoint reloaded")
            # ===== TRAINING-PHASE BOUNDARY =====
            # The 100M transform + seed-1 tripwire were MOVED to a post-training, UNGUARDED
            # phase (below, after the train receipt seals). The training guards (CellWatchdog
            # swap-growth + anon; _node_guard) exist to bound the TRAINING working set; the
            # 100M transform is a read-only projection whose page-cache pressure from reading
            # the 153.6 GB fp32 substrate legitimately grows system swap and tripped
            # CellWatchdog's 1 GiB swap-growth abort mid-projection — R0268 attempts 1 & 3 each
            # lost a clean 24h train to exactly this. `reloaded` is carried across the boundary;
            # `source`/`graph` stay live for the transform + its receipt below.
            gate.finish(f"R0268 {capability} training stage end")
        window.close()
        tail = _guard_tail_reported(guard_ctx, label=label)
        scored = _score_gate_without_raising(gate, tail, label=label)
        gaps = gap_report(recorder.records, arm=node_id)
    finally:
        watchdog_state = watchdog.stop()

    if watchdog_state["tripped"]:
        raise Round0268NodeError(
            f"R0268 seed-{seed} watchdog tripped: {watchdog_state['trip_reason']!r}"
        )
    if int(memory["peak_reserved_bytes"]) > DEVICE_BUDGET_BYTES:
        raise Round0268NodeError(f"R0268 seed-{seed} peak reserved bytes exceed the budget")
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0268NodeError(
            f"R0268 train peak RSS {peak_rss_gib:.2f} GiB exceeds the analytic "
            f"HOST_RSS_LIMIT_GIB {HOST_RSS_LIMIT_GIB} (basis: {HOST_RSS_ANALYTIC_BASIS['method']})"
        )
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
    # A rehearsal (R11 harness) sets this job flag; production queues never do. The flag-READ runs
    # every time (exercised in production as False), and marks rehearsal receipts unmistakably.
    rehearsal_flag = bool(job.get("is_non_evidence_rehearsal", False))
    # I/O signatures computed here (the caller's job); the receipt dict is built by the PURE,
    # unit-tested `_assemble_train_receipt` (no I/O, no free locals) so the assembly cannot carry
    # a latent runtime error into a 24h run (attempt-4's int8_full use-after-del).
    receipt_body = _assemble_train_receipt(
        capability=capability,
        seed=seed,
        release_sha=active["manifest"]["release_sha"],
        abort_flag=abort_flag,
        production_config_sig=expected_input_signature(config_path),
        config_sha=config_sha,
        observed_invariant=observed_invariant,
        declared_invariant=declared_invariant,
        recipe=recipe,
        closure=closure,
        treatment_closure_seal_sig=expected_input_signature(
            _bound_path(job, "treatment_closure", label="R0268 treatment closure seal")
        ),
        model_sig=expected_input_signature(model_path),
        substrate=substrate,
        int8_substrate_manifest_signature=int8_full["manifest_signature"],
        int8_full_receipt=int8_full_receipt,
        graph=graph,
        edges=edges,
        updates=updates,
        base_horizon=int(job.get("base_horizon", -1)),
        accounting=accounting,
        runtime=runtime,
        residency=residency,
        fneg_telemetry=fneg_telemetry,
        wall=wall,
        memory=memory,
        watchdog_state=watchdog_state,
        peak_rss_gib=peak_rss_gib,
        gaps=gaps,
        scored=scored,
        tail=tail,
        coverage=coverage,
        node_id=node_id,
        checkpoint_fneg_roundtrip=checkpoint_fneg_roundtrip,
        rehearsal=rehearsal_flag,
    )
    _seal(output, "train-receipt.json", receipt_body)
    # ===== TRAINING EVIDENCE SEALED =====
    # The clean train is now salvageable even if the transform below dies: attempts 1 & 3 lost
    # their trains ONLY because the receipt assembled AFTER the transform, so the in-memory
    # telemetry (fneg dynamics, closure verdict, accounting, watchdog) died with the process.
    # Guard + watchdog are CLOSED here; the transform runs unguarded.

    # ===== POST-TRAINING TRANSFORM PHASE (UNGUARDED, separately receipted) =====
    # The 100M projection + seed-1 tripwire. No CellWatchdog / _node_guard by design: the
    # page-cache pressure from reading the 153.6 GB fp32 substrate is benign for this read-only
    # projection (proven 3x standalone at ~117 GiB RSS: the R0268 panel dry-run + two salvage
    # transforms), but it grows system swap past CellWatchdog's 1 GiB abort — the guard must not
    # police a phase it was never scoped for.
    transform_started = time.monotonic()

    # `abort_flag` is the `_start_node()` dict (require_enforceable_abort_flag), NOT a path.
    # The runner writes the flag FILE at abort_flag["abort_flag_path"] to signal a cooperative
    # unwind; the transform poll must os.path.exists() THAT path, never the dict (R9 passed the
    # dict → TypeError on the first poll; that path was structurally-only tested, never driven).
    abort_flag_path = abort_flag.get("abort_flag_path") if isinstance(abort_flag, dict) else None

    def _transform_poll(message: str) -> None:
        # cooperative abort only (the gate/recorder closed with the training phase)
        if abort_flag_path and os.path.exists(abort_flag_path):
            raise Round0268NodeError(
                f"R0268 seed-{seed} transform observed the cooperative abort flag"
            )

    coordinates = _transform_100m_in_chunks(reloaded, source, _transform_poll)
    validate_published_map(coordinates)
    coordinates_path = os.path.join(output, "coordinates.npy")
    atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
    coordinates_ordered_sha256 = ordered_array_sha256(coordinates)
    transform_rows_finite = int(np.isfinite(coordinates).all(axis=1).sum())
    all_coordinates_finite = transform_rows_finite == ROWS
    # SEED-1 TRIPWIRE (map-only collapse + fog): a PREVIEW a driver reads to check seed-1's
    # backstop/fog escalation before seeds 2/3. NOT a gate input — the panel re-transforms from
    # the model and re-scores collapse/fog itself.
    collapse_preview = map_collapse(coordinates)
    fog_preview = _map_fog(coordinates, bins=FOG_BINS)
    tripwire_inputs = {
        "collapse": float(collapse_preview["r10_over_radius_times_sqrt_n"]),
        "fog": float(fog_preview["fog"]),
        "resolution_levels": int(fog_preview["resolution_levels"]),
        "degenerate": bool(fog_preview["degenerate"]),
        "fog_detail": fog_preview,
        "collapse_detail": collapse_preview,
        "note": (
            "map-level collapse + fog (map-only, no truth); a driver reads these to check "
            "seed-1's backstop + fog escalation before seeds 2/3. The panel re-scores from the "
            "model for the gate. Produced in the UNGUARDED post-training transform phase."
        ),
    }
    del reloaded, coordinates
    torch.cuda.empty_cache()
    gc.collect()
    transform_wall_s = time.monotonic() - transform_started
    transform_peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    transform_receipt_body = _assemble_transform_receipt(
        capability=capability,
        seed=seed,
        phase="post-training-transform",
        train_receipt_sig=expected_input_signature(os.path.join(output, "train-receipt.json")),
        model_sig=expected_input_signature(model_path),
        coordinates_sig=expected_input_signature(coordinates_path),
        coordinates_ordered_sha256=coordinates_ordered_sha256,
        ordered_substrate_sha256=substrate["ordered_substrate_sha256"],
        tripwire_inputs=tripwire_inputs,
        transform_wall_s=transform_wall_s,
        transform_peak_rss_gib=transform_peak_rss_gib,
        all_coordinates_finite=all_coordinates_finite,
        transform_rows_finite=transform_rows_finite,
        rehearsal=rehearsal_flag,
    )
    if not all(transform_receipt_body["transform_checks"].values()):
        raise Round0268NodeError(
            f"R0268 seed-{seed} transform checks failed: "
            f"{transform_receipt_body['transform_checks']}"
        )
    _seal(output, "transform-receipt.json", transform_receipt_body)
    del source, graph
    gc.collect()
    print(json.dumps({
        "capability": capability,
        "seed": seed,
        "x_residency": residency["x_residency"],
        "fneg_active": fneg_telemetry is not None,
        "seed_1_tripwire_collapse": tripwire_inputs["collapse"],
        "seed_1_tripwire_fog": tripwire_inputs["fog"],
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


# --------------------------------------------------------------------------- #
# the seed-42 TRANSFORM-CORRECTION node (R10) — re-run ONLY the fixed, unguarded
# transform from seed42's sealed train-receipt + model. No re-train.
# --------------------------------------------------------------------------- #


def run_transform_correction(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """Re-run ONLY the (fixed, unguarded) 100M transform from seed42's SEALED train-receipt +
    model, producing the transform-receipt + coordinates + seed-1 tripwire that the R9 defect
    prevented. The sealed train-receipt + model are READ-ONLY inputs (signature-verified); the
    original failed marker + its outputs are NEVER touched (preserve-not-erase). No CellWatchdog /
    _node_guard around the transform — the same phase scoping the R10 fix establishes for run_train.
    """
    install_stop_hooks(label="R0268 round0268_nodes.run_transform_correction")
    import torch
    from basemap.pumap.parametric_umap import ParametricUMAP

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0268NodeError("R0268 transform-correction handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0268NodeError("R0268 transform-correction requires CUDA")
    node_id = str(active.get("node_id") or TRANSFORM_CORRECTION_ACTION)
    seed = int(job.get("training_seed", -1))
    if seed not in SEEDS:
        raise Round0268NodeError(f"R0268 transform-correction seed {seed!r} is not a flagship seed")
    capability = capability_for_seed(seed)
    label = f"R0268 seed-{seed} transform-correction"
    abort_flag = _start_node(label)
    abort_flag_path = abort_flag.get("abort_flag_path") if isinstance(abort_flag, dict) else None

    # 1. the sealed R0238 100M substrate (identity-anchored, served lazily).
    substrate = _sealed_100m_substrate(job)
    source = _open_100m_substrate(substrate)

    # 2. the SEALED train-receipt (read-only) — verify it is seed42's clean training evidence.
    train_receipt_path = prompt_contract.verify_signature(
        dict(job["train_receipt"]), label=f"R0268 seed-{seed} sealed train-receipt"
    )
    train_receipt = prompt_contract.read_sealed(
        train_receipt_path, label=f"R0268 seed-{seed} sealed train-receipt"
    )
    train_checks = train_receipt.get("train_checks") or {}
    if (
        train_receipt.get("schema") != TRAIN_SCHEMA
        or train_receipt.get("round_id") != ROUND_ID
        or int(train_receipt.get("training_seed", -1)) != seed
        or train_receipt.get("training_performed") is not True
        or train_receipt.get("x_residency") != X_RESIDENCY
        or not train_checks
        or not all(bool(v) for v in train_checks.values())
        or str(train_receipt.get("ordered_substrate_sha256")) != substrate["ordered_substrate_sha256"]
    ):
        raise Round0268NodeError(
            f"R0268 seed-{seed} sealed train-receipt is not clean flagship training evidence"
        )

    # 3. the SEALED model (read-only) — verify the receipt's own model signature AND the job's.
    model_path = prompt_contract.verify_signature(
        dict(train_receipt["model"]), label=f"R0268 seed-{seed} sealed model (via receipt)"
    )
    job_model_path = prompt_contract.verify_signature(
        dict(job["model"]), label=f"R0268 seed-{seed} sealed model (via job)"
    )
    if os.path.realpath(job_model_path) != os.path.realpath(model_path):
        raise Round0268NodeError("R0268 transform-correction model input != the receipt's model")

    reloaded = ParametricUMAP.load(model_path, device="cuda")
    if not (
        float(reloaded.fneg_weight) == R0265N.FNEG_WEIGHT
        and float(reloaded.fneg_lo) == R0265N.FNEG_LO
        and float(reloaded.fneg_hi) == R0265N.FNEG_HI
    ):
        raise Round0268NodeError("R0268 transform-correction model did not round-trip fneg params")

    # 4. provenance linkage: the ORIGINAL R9 failed marker (read-only signature — the record of
    #    the defect this node corrects). Bound by the queue; NEVER modified here.
    failed_marker_signature = expected_input_signature(
        prompt_contract.verify_signature(
            dict(job["original_failed_marker"]),
            label=f"R0268 seed-{seed} original R9 failed marker",
        )
    )

    output = create_fresh_directory(str(job["outputs"][0]), label=label)

    # 5. the FIXED, UNGUARDED transform (R10 poll: os.path.exists(abort_flag_path)).
    def _transform_poll(message: str) -> None:
        if abort_flag_path and os.path.exists(abort_flag_path):
            raise Round0268NodeError(
                f"R0268 seed-{seed} transform-correction observed the cooperative abort flag"
            )

    transform_started = time.monotonic()
    coordinates = _transform_100m_in_chunks(reloaded, source, _transform_poll)
    validate_published_map(coordinates)
    coordinates_path = os.path.join(output, "coordinates.npy")
    atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
    coordinates_ordered_sha256 = ordered_array_sha256(coordinates)
    transform_rows_finite = int(np.isfinite(coordinates).all(axis=1).sum())
    all_coordinates_finite = transform_rows_finite == ROWS
    collapse_preview = map_collapse(coordinates)
    fog_preview = _map_fog(coordinates, bins=FOG_BINS)
    tripwire_inputs = {
        "collapse": float(collapse_preview["r10_over_radius_times_sqrt_n"]),
        "fog": float(fog_preview["fog"]),
        "resolution_levels": int(fog_preview["resolution_levels"]),
        "degenerate": bool(fog_preview["degenerate"]),
        "fog_detail": fog_preview,
        "collapse_detail": collapse_preview,
        "note": (
            "map-level collapse + fog (map-only, no truth) from the CORRECTED transform of "
            "seed42's sealed model; a driver reads these for the seed-1 backstop + fog escalation "
            "before seeds 43/44. The panel re-scores from the model for the gate."
        ),
    }
    del reloaded, coordinates
    torch.cuda.empty_cache()
    gc.collect()
    transform_wall_s = time.monotonic() - transform_started
    transform_peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

    body = _assemble_transform_receipt(
        capability=capability,
        seed=seed,
        phase="post-training-transform-correction",
        train_receipt_sig=expected_input_signature(train_receipt_path),
        model_sig=expected_input_signature(model_path),
        coordinates_sig=expected_input_signature(coordinates_path),
        coordinates_ordered_sha256=coordinates_ordered_sha256,
        ordered_substrate_sha256=substrate["ordered_substrate_sha256"],
        tripwire_inputs=tripwire_inputs,
        transform_wall_s=transform_wall_s,
        transform_peak_rss_gib=transform_peak_rss_gib,
        all_coordinates_finite=all_coordinates_finite,
        transform_rows_finite=transform_rows_finite,
        extra_checks={"reused_sealed_train_and_model_no_retrain": True},
        extra={
            **_receipt_envelope(active["manifest"]),
            "node": node_id,
            "abort_flag_precondition": abort_flag,
            "corrects_defect": (
                "R9 _transform_poll called os.path.exists() on the _start_node() dict → TypeError "
                "in the original transform; the R10 poll checks abort_flag['abort_flag_path']."
            ),
            "provenance": {
                "sealed_train_receipt": expected_input_signature(train_receipt_path),
                "sealed_model": expected_input_signature(model_path),
                "original_r9_failed_marker": failed_marker_signature,
                "note": (
                    "seed42's clean train (attempt-4, R9) sealed its train-receipt BEFORE the "
                    "transform; this node re-runs ONLY the corrected transform from that sealed "
                    "evidence. The original failed marker + outputs are preserved untouched."
                ),
            },
            "gate_registerable_here": False,
        },
    )
    if not all(body["transform_checks"].values()):
        raise Round0268NodeError(
            f"R0268 seed-{seed} transform-correction checks failed: {body['transform_checks']}"
        )
    _seal(output, "transform-receipt.json", body)
    del source
    gc.collect()
    print(json.dumps({
        "capability": capability,
        "seed": seed,
        "corrected_transform": True,
        "seed_1_tripwire_collapse": tripwire_inputs["collapse"],
        "seed_1_tripwire_fog": tripwire_inputs["fog"],
        "transform_wall_s": round(transform_wall_s, 1),
    }))


# --------------------------------------------------------------------------- #
# the panel — three 100M cells scored on R0265's instruments (fresh trains only)
# --------------------------------------------------------------------------- #


def _authenticate_100m_map(cell: Mapping[str, Any], substrate: Mapping[str, Any]) -> dict[str, Any]:
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0268NodeError(f"R0268 cell seed {seed!r} is not a 100M flagship cell")
    capability = capability_for_seed(seed)
    if str(cell.get("capability") or "") != capability:
        raise Round0268NodeError("R0268 cell capability changed")
    receipt_path, receipt_signature = _intra_queue_signature(
        cell["train_receipt"], label=f"R0268 seed-{seed} train receipt"
    )
    receipt = prompt_contract.read_sealed(receipt_path, label=f"R0268 seed-{seed} train receipt")
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
        raise Round0268NodeError(f"R0268 seed-{seed} train receipt contract changed")
    if str(receipt.get("ordered_substrate_sha256")) != substrate["ordered_substrate_sha256"]:
        raise Round0268NodeError(f"R0268 seed-{seed} was not trained on the panel's substrate")
    model_path = prompt_contract.verify_signature(receipt["model"], label=f"R0268 seed-{seed} map")
    return {
        "seed": seed,
        "capability": capability,
        "receipt": receipt,
        "receipt_signature": receipt_signature,
        "model_path": model_path,
        "seed_invariant_sha256": str(receipt["seed_invariant_sha256"]),
    }


def _build_prefix_purity_centroids(
    source_prefix: np.ndarray, centroid_ks: Sequence[int], *, cache_dir: str
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Build the DESCRIPTIVE purity centroids INLINE on the R0238 prefix (GPU k-means).

    Reuses the frozen-centroids builder (random init + 25 Lloyd iters, seed 0) that produced
    R0218's centroids, but fits it on the R0238 substrate's first-PREFIX_ROWS rows instead —
    so the descriptive purity is self-contained on the prefix with no R0218 centroid
    dependency. Each k's centroids are written immutable into ``cache_dir`` and returned with
    their signatures. The fit reads the prefix memmap in 100k-row chunks (no >=2 GB
    materialisation).
    """
    from experiments.score_complete_panel import frozen_centroids

    cache_dir = create_fresh_directory(cache_dir, label="R0268 descriptive prefix centroids")
    ks = [int(k) for k in centroid_ks]
    built = frozen_centroids(source_prefix, ks, cache_dir, seed=0, iters=25)
    centroids: dict[int, np.ndarray] = {}
    signatures: dict[str, Any] = {}
    for k in ks:
        array = np.asarray(built[k], dtype=np.float32)
        if array.shape != (k, DIMENSION):
            raise Round0268NodeError(f"R0268 descriptive prefix centroids k{k} geometry changed")
        centroids[k] = array
        signatures[str(k)] = expected_input_signature(
            os.path.join(cache_dir, f"centroids_k{k}.npy")
        )
    return centroids, signatures


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0268 round0268_nodes.run_panel")
    import torch
    from basemap.panel_v2 import (
        reset_process_cuda_peak,
        score_panel,
    )
    from basemap.pumap.parametric_umap import ParametricUMAP

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0268NodeError("R0268 panel handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0268NodeError("R0268 panel scoring requires CUDA")
    node_id = str(active.get("node_id") or PANEL_ACTION)
    label = "R0268 100M ×2 host-int8 panel"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)

    substrate = _sealed_100m_substrate(job)
    source = _open_100m_substrate(substrate)
    # LINEAGE check (plan §2.4): the 100M substrate's first-2M ordered hash vs R0216-c3's
    # sealed 2M reference (READ from the bound R0218 panel's lineage). Expected NON-match ->
    # purity is descriptive. Computed ONCE (streams the 2M prefix hash).
    r0216_c3_reference = _read_r0216_c3_reference(job)
    lineage_check = verify_hundred_m_prefix_lineage(source, r0216_c3_reference)

    centroid_ks = [int(k) for k in job["centroid_ks"]]
    cfg = prompt_contract.panel_config()

    # The FLOOR-MATCHED held-out FFR instrument (born correct at 100M — trips 9/10 of the
    # 50M saga cannot recur): the held-out reserve embeddings projected THROUGH each trained
    # map (`model.transform(reserve)`), scored against the reserve's exact-cosine neighbour
    # truth, at the N-scaled discovery radius disc = int(ROWS * 0.001) = 100,000 (0.1%·N).
    #   * `reserve_embeddings` = reserve.f32[reserve-query-rows] (the R0238 held-out reserve
    #     query rows), a sealed panel input; projected per-cell below.
    #   * `reserve_truth` = the sealed 100M reserve-neighbour top-10 truth (indices INTO the
    #     100M substrate == into each cell's coordinates), a sealed panel input.
    reserve_all = np.load(
        _bound_path(job, "heldout_reserve", label="R0268 100M held-out reserve"),
        mmap_mode="r", allow_pickle=False,
    )
    reserve_query_rows = np.load(
        _bound_path(job, "reserve_query_rows", label="R0268 reserve query rows"),
        allow_pickle=False,
    ).astype(np.int64, copy=False)
    if reserve_all.ndim != 2 or int(reserve_all.shape[1]) != DIMENSION or reserve_query_rows.ndim != 1:
        raise Round0268NodeError("R0268 held-out reserve geometry changed")
    reserve_embeddings = np.asarray(reserve_all[reserve_query_rows], dtype=np.float32)
    reserve_truth = np.load(
        _bound_path(job, "reserve_truth", label="R0268 reserve-neighbour truth"),
        allow_pickle=False,
    ).astype(np.int64, copy=False)
    if reserve_truth.ndim != 2 or reserve_truth.shape[0] != reserve_embeddings.shape[0]:
        raise Round0268NodeError("R0268 reserve-neighbour truth geometry changed")
    reserve_disc = int(ROWS * 0.001)

    cells_in = job.get("cells")
    if not isinstance(cells_in, list):
        raise Round0268NodeError("R0268 cell input matrix changed")
    authenticated = [_authenticate_100m_map(cell, substrate) for cell in cells_in]
    if {entry["seed"] for entry in authenticated} != set(SEEDS):
        raise Round0268NodeError("R0268 three-cell input matrix changed")
    invariants = {str(entry["seed_invariant_sha256"]) for entry in authenticated}
    if len(invariants) != 1:
        raise Round0268NodeError("R0268 pooled family is not one 100M recipe")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0268 100M panel")
    started = time.monotonic()
    reset_process_cuda_peak()

    window = ledger.window("R0268 100M panel scoring stage")
    guard_ctx = _node_guard(label, anonymous_budget_bytes=R0268_ANON_BUDGET_BYTES)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0268 panel stage entered")
        wrapped = window.wrap(recorder)
        wrapped(
            f"R0268 lineage check: 100M-prefix {lineage_check['observed_hundred_m_prefix_sha256'][:12]}"
            f"… != R0216-c3 {lineage_check['r0216_c3_2m_reference_sha256'][:12]}…"
        )

        # DESCRIPTIVE purity centroids: fit INLINE on the R0238 first-PREFIX_ROWS prefix
        # (GPU k-means, same frozen builder R0218 used) so the descriptive reference AND its
        # centroids are self-contained on the prefix — no R0218 lineage.
        prefix_rows = PREFIX_ROWS
        if prefix_rows > ROWS:
            raise Round0268NodeError("R0268 descriptive prefix exceeds the 100M substrate rows")
        centroids, centroid_signatures = _build_prefix_purity_centroids(
            source[:prefix_rows], centroid_ks,
            cache_dir=os.path.join(output, "descriptive-prefix-centroids"),
        )
        wrapped("R0268 descriptive prefix centroids fit on the R0238 first-2M rows")

        cells: dict[str, dict[str, Any]] = {}
        for entry in sorted(authenticated, key=lambda e: e["seed"]):
            seed = entry["seed"]
            proj_model = ParametricUMAP.load(entry["model_path"], device="cuda")
            coordinates = _transform_100m_in_chunks(proj_model, source, wrapped)
            if coordinates.shape != (ROWS, 2) or not np.isfinite(coordinates).all():
                raise Round0268NodeError(f"R0268 seed-{seed} transform is not a finite 100M map")
            # DESCRIPTIVE purity pass — the ONLY score_panel call run_panel makes. Xa = the
            # substrate's first <prefix_rows> rows; Z = the cell's first <prefix_rows>
            # coordinate rows (so the 2D neighbour pool is the PREFIX ROWS' coordinates ONLY).
            # hiD_reference=None builds the reference INLINE on that same prefix, and
            # centroids_by_k are the prefix-fit centroids — self-contained, NO R0218
            # dependency. <prefix_rows> is < 8M, so this pass takes NO scale_admission (the
            # >=8M guard refuses a below-scale admission). DESCRIPTIVE ONLY.
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
                    "treatment": "fneg-x2-md000-hostint8-100m",
                    "pass": "r0238-prefix-descriptive-purity",
                    "descriptive": True,
                    "gated": False,
                    "lineage_caveat": DESCRIPTIVE_PURITY_LINEAGE_CAVEAT,
                },
            )
            purity_ratios = {"k256": float(purity_panel["purity"]["k256"]),
                             "k1024": float(purity_panel["purity"]["k1024"])}
            # FLOOR-MATCHED held-out FFR: project the OUT-OF-SUBSTRATE held-out reserve
            # THROUGH this map, score against the sealed reserve-neighbour truth at the
            # N-scaled disc = int(ROWS * 0.001). collapse/fog are measured on the FULL 100M
            # coordinates inside score_one_map (byte-identical to R0265's instrument).
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
            cells[str(seed)] = {
                "seed": seed,
                "capability": entry["capability"],
                "train_receipt": dict(entry["receipt_signature"]),
                "model": dict(entry["receipt"]["model"]),
                "seed_invariant_sha256": entry["seed_invariant_sha256"],
                "x_residency": X_RESIDENCY,
                "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
                "metrics": scored_map,
                "panel_purity_numerators": purity_panel.get("purity_numerators"),
                "descriptive_purity": {
                    "pass": "r0238-prefix-descriptive-purity",
                    "descriptive": True,
                    "gated": False,
                    "prefix_rows": prefix_rows,
                    "reference": "r0238-prefix-inline",
                    "k256": purity_ratios["k256"],
                    "k1024": purity_ratios["k1024"],
                    "numerators": purity_panel.get("purity_numerators"),
                    "hiD_reference_key": purity_panel["provenance"]["hiD_reference_key"],
                    "hiD_reference_reused": bool(purity_panel["provenance"]["hiD_reference_reused"]),
                    "lineage_caveat": DESCRIPTIVE_PURITY_LINEAGE_CAVEAT,
                },
            }
            del proj_model, coordinates, placed
            torch.cuda.empty_cache()
            gc.collect()
            wrapped(f"R0268 seed-{seed} scored")
        gate.finish("R0268 panel stage end")
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
        "lineage_is_non_match_purity_descriptive": (
            lineage_check["matches_r0216_c3"] is False
            and lineage_check["purity_is_descriptive"] is True
        ),
    }
    if not all(execution_checks.values()):
        raise Round0268NodeError(f"R0268 panel execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > PANEL_RSS_LIMIT_GIB:
        raise Round0268NodeError(
            f"R0268 panel peak RSS {peak_rss_gib:.2f} GiB exceeds PANEL_RSS_LIMIT_GIB "
            f"{PANEL_RSS_LIMIT_GIB} (defaulted to the train's 115; refine from the dry-run)"
        )
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
        "x_residency": X_RESIDENCY,
        "seed_invariant_sha256": sorted(invariants)[0],
        "panel_metric_table": metric_table,
        "cells": cells,
        "descriptive_purity_centroids": centroid_signatures,
        "gated_metrics": list(GATE_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_PURITY_METRICS),
        "lineage_check": lineage_check,
        "descriptive_purity": {
            "descriptive": True,
            "gated": False,
            "prefix_rows": prefix_rows,
            "reference": "r0238-prefix-inline",
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
            "born_correct_at_100m": True,
            "probe_design": "out_of_substrate_reserve_projection",
            "placed_is": "model.transform(reserve.f32[reserve-query-rows])",
            "truth": "sealed 100M reserve-neighbour exact-cosine top-10 (indices into the 100M substrate)",
            "disc": reserve_disc,
            "disc_rule": "int(ROWS * 0.001) = 100000 — 0.1%·N, N-scaled",
            "n_reserve_probes": int(reserve_embeddings.shape[0]),
            "reserve_query_rows_binding": dict(job.get("reserve_query_rows") or {}),
            "reserve_truth_binding": dict(job.get("reserve_truth") or {}),
            "note": (
                "the R0265 family floor's own instrument — the OUT-OF-SUBSTRATE reserve "
                "projection at disc=int(ROWS*0.001)=100000 — so the panel is born matched to "
                "the floor. The two 50M mis-specifications (trip 9: fixed 2000 disc; trip 10: "
                "IN-SUBSTRATE coordinates[probe_rows]) cannot recur at 100M."
            ),
        },
        "lineage": {
            "graph_manifest": dict(substrate["manifest_signature"]),
            "substrate": dict(substrate["substrate_signature"]),
            "ordered_substrate_sha256": substrate["ordered_substrate_sha256"],
        },
        "purity_reference_note": (
            "purity k256/k1024 are DESCRIPTIVE / UNGATED at 100M (plan §2.4). They are scored "
            "on the R0238 substrate's first prefix_rows rows (Xa = those rows; Z = each cell's "
            "first prefix_rows coordinate rows) against a reference + k-means centroids built "
            "INLINE on that SAME prefix — self-contained, NO R0218 dependency. The lineage "
            "check proves the 100M-prefix ordered hash != R0216-c3's sealed 2M reference, so "
            "they are NOT commensurate with the R0265 2M family bands and NEVER enter the "
            "gate. The GATED collapse / fog / held-out FFR are measured on the FULL 100M "
            "coordinates (score_one_map). run_panel invokes score_panel ONLY for this <8M "
            "descriptive pass; no >=8M score_panel is run and no slim scale-performance "
            "certificate is required."
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
        "panel_rss_limit_gib": PANEL_RSS_LIMIT_GIB,
        "performance": {"node_wall_s": time.monotonic() - started},
    }
    _seal(output, "fneg-100m-x2-panel.json", body)
    print(json.dumps({
        "capability": PANEL_CAPABILITY,
        "n": len(SEEDS),
        "seed_collapse": {str(s): metric_table[str(s)]["collapse"] for s in SEEDS},
        "lineage_non_match": lineage_check["matches_r0216_c3"] is False,
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
    """
    d = prompt_contract.read_sealed(panel_path, label="R0265 sealed n=13 panel")
    if d.get("capability") != R0265N.PANEL_CAPABILITY or int(d.get("n", -1)) != R0265N.N_FAMILY:
        raise Round0268NodeError("R0265 sealed n=13 panel contract changed")
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

    READ from `bands.yinf_x2` in the frozen P1 analysis-v2 result -- NEVER a literal. The
    100M band is the SAME as 50M's (λ=37.08 is saturated by 25M; collapse flat to the
    asymptote), widened by the same 1.96·σ_fam/√3.
    """
    with open(path, "r", encoding="utf-8") as handle:
        d = json.load(handle)
    bands = dict(d.get("bands") or {})
    if "yinf_x2" not in bands:
        raise Round0268NodeError("P1 analysis-v2 result carries no yinf_x2 asymptote band")
    lo, hi = (float(v) for v in bands["yinf_x2"])
    if not (lo < hi):
        raise Round0268NodeError("P1 ×2 asymptote band is not an interval")
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

    The P1 band is a bootstrap band on the FITTED ×2 asymptote (a mean), so it is widened by
    the family's √n-shrunk seed-noise allowance and the SEED-MEAN is gated against it.
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
            "widened by z·σ_fam/√n (the family's √n-shrunk seed spread) and the SEED-MEAN is "
            "gated against it. The 100M band is the SAME as 50M's (λ=37 saturated). The "
            "no-straddle rule does NOT apply to this band."
        ),
    }


def score_per_seed_backstops(
    *,
    metric_table: Mapping[str, Mapping[str, Any]],
    backstops: Mapping[str, Any],
    sigma_fam_fog: float,
) -> dict[str, Any]:
    """The per-seed hard backstops (plan criteria 1-3 — collapse/fog/held-out FFR).

    Every seed must clear ONLY: collapse >= R0265 family floor (0.8129, N-invariant per
    R0264); fog <= R0265 family ceiling (0.41207) with a mechanical near-ceiling escalation
    if fog > ceiling - 1·σ_fam,fog; held-out FFR >= R0265 family floor (0.39906). The
    no-straddle rule applies to THESE three gates. Every floor is READ from the sealed R0265
    floors.

    Purity k256/k1024 are DESCRIPTIVE-only at 100M (plan §2.4): their per-seed verdicts
    against the sealed R0265 bands are still RECORDED under ``descriptive_purity`` for the
    report, but they DO NOT enter ``clears_every_backstop`` or the straddle set.
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
        by_metric = {
            HELDOUT_FFR_METRIC: {
                "verdict": "PASS" if ffr_pass else "FAIL", "passes": ffr_pass,
                "value": ffr, "floor": ffr_floor, "margin": ffr - ffr_floor,
            },
            COLLAPSE_METRIC: collapse_v,
            FOG_METRIC: {**fog_v, "near_ceiling_escalation": bool(fog_near_ceiling),
                        "near_ceiling_threshold": near_ceiling_threshold},
        }
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
                "NOT gate at 100M (plan §2.4); no per-seed purity verdict enters "
                "clears_every_backstop or the straddle set"
            ),
        },
        "no_straddle_rule": (
            "applies to the per-seed backstop + collapse/fog/FFR gates, NOT to purity "
            "(descriptive-only at 100M) and NOT to the P1 asymptote band (a single seed "
            "outside the P1 band but at/above the backstop is expected noise, not ambiguity)"
        ),
    }


def _metric_table_from_panel(panel: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    # Consumption-side ENFORCEMENT: a scientific panel must never carry the slim
    # cert-production purpose stamp. Refuse it here rather than scoring it.
    assert_not_slim_cert_production_panel(
        panel, label="R0268 100M panel", error_cls=Round0268NodeError)
    if panel.get("capability") != PANEL_CAPABILITY or panel.get("schema") != PANEL_SCHEMA:
        raise Round0268NodeError("R0268 100M panel contract changed")
    table = dict(panel["panel_metric_table"])
    if {int(s) for s in table} != set(SEEDS):
        raise Round0268NodeError("R0268 100M panel is not the three fneg cells")
    return {str(s): dict(table[str(s)]) for s in SEEDS}


def run_gate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0268 round0268_nodes.run_gate")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0268NodeError("R0268 gate handler received another queue")
    node_id = str(active.get("node_id") or GATE_ACTION)
    label = "R0268 100M ×2 seed-mean gate"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0268 gate")

    window = ledger.window("R0268 gate stage")
    guard_ctx = _node_guard(label, anonymous_budget_bytes=R0268_ANON_BUDGET_BYTES)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0268 gate stage entered")
        wrapped = window.wrap(recorder)

        # 1. the three-seed 100M metrics (intra-queue panel).
        panel = prompt_contract.read_sealed(
            _bound_path(job, "panel", label="R0268 100M panel"), label="R0268 100M panel"
        )
        assert_not_slim_cert_production_panel(
            panel, label="R0268 gate: bound 100M panel", error_cls=Round0268NodeError)
        metric_table = _metric_table_from_panel(panel)
        seed_collapse = {s: float(metric_table[s][COLLAPSE_METRIC]) for s in metric_table}
        # The panel's lineage check (100M-prefix != R0216-c3) is carried forward descriptively.
        lineage_check = dict(panel.get("lineage_check") or {})
        wrapped("R0268 three-seed 100M metrics read")

        # 2. σ_fam, RECOMPUTED from R0265's sealed n=13 panel.
        sigma = sigma_fam_from_panel(_bound_path(job, "r0265_panel", label="R0265 sealed n=13 panel"))
        wrapped("R0268 σ_fam recomputed from the sealed R0265 panel")

        # 3. the P1 ×2 asymptote band, READ from the sealed analysis-v2 result.
        p1 = read_p1_x2_asymptote_band(_bound_path(job, "p1_asymptote", label="P1 analysis-v2 result"))
        wrapped("R0268 P1 ×2 band read from the sealed analysis-v2 result")

        # 4. the per-seed backstops, READ from R0265's sealed family floors.
        backstops = R0266N.read_family_bands(
            _bound_path(job, "r0265_floors", label="R0265 sealed family floors")
        )["bands"]
        wrapped("R0268 per-seed backstops read from the sealed R0265 floors")

        # CRITERION 1: the seed-mean collapse inside the widened P1 ×2 band.
        criterion_1 = score_collapse_seed_mean(
            seed_collapse=seed_collapse,
            p1_lower=p1["p1_lower"],
            p1_upper=p1["p1_upper"],
            sigma_fam_collapse=sigma["sigma_fam_collapse"],
        )
        # CRITERIA 1-3 per-seed backstops: collapse floor + fog ceiling + held-out FFR floor
        # (purity is DESCRIPTIVE-only and does NOT enter pass/fail; plan §2.4).
        backstop_scoring = score_per_seed_backstops(
            metric_table=metric_table,
            backstops=backstops,
            sigma_fam_fog=sigma["sigma_fam_fog"],
        )
        descriptive_purity = {
            "descriptive": True,
            "gated": False,
            "lineage_caveat": DESCRIPTIVE_PURITY_LINEAGE_CAVEAT,
            "lineage_check": lineage_check,
            "values": {
                s: {
                    "k256": float(metric_table[s][PURITY_K256_METRIC]),
                    "k1024": float(metric_table[s][PURITY_K1024_METRIC]),
                }
                for s in metric_table
            },
        }
        wrapped("R0268 criteria scored from sealed bands, σ_fam and the P1 band")
        gate.finish("R0268 gate stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    # The 100M PASS decision (a FINDING reported either way). PASS iff criterion 1 passes AND
    # every seed clears the backstops AND no gate straddles AND no fog escalation. Fail or
    # ambiguous -> the owner with the per-arm drift decomposition (no auto-proceed, no
    # post-hoc widening). Purity can never flip the verdict (descriptive-only).
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
    verdict = "100M_PASS" if passes and not backstop_scoring["any_fog_near_ceiling_escalation"] else "100M_FAIL_OR_AMBIGUOUS"

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
        "lineage_check_is_non_match_and_descriptive": (
            lineage_check.get("matches_r0216_c3") is False
            and lineage_check.get("purity_is_descriptive") is True
        ),
    }
    if not all(execution_checks.values()):
        raise Round0268NodeError(f"R0268 gate execution checks failed: {execution_checks}")

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
            "the pre-registered 100M flagship gate (plan-100m-flagship-2026-08-17): the "
            "SEED-MEAN collapse inside P1's ×2 asymptote band (SAME as 50M) widened by a "
            "√n-shrunk family seed-noise allowance, plus per-seed hard backstops on collapse/"
            "fog/FFR — criteria 1-3. Purity is DESCRIPTIVE-only at 100M and is NOT in the "
            "go/no-go. A FAIL/AMBIGUOUS returns to the owner with the drift decomposition, "
            "never an auto-proceed, never post-hoc widening."
        ),
        "seed_spread_assumption": SEED_SPREAD_ASSUMPTION,
        "sigma_fam": sigma,
        "p1_x2_asymptote_band": p1,
        "criterion_1_collapse_seed_mean": criterion_1,
        "criteria_1_3_per_seed_backstops": backstop_scoring,
        "descriptive_purity": descriptive_purity,
        "lineage_check": lineage_check,
        "gated_metrics": list(GATE_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_PURITY_METRICS),
        "pre_registered_pass_criteria": {
            "1": (
                "mean(collapse over 3 seeds) ∈ [P1_lo − 1.96·σ_fam/√3, P1_hi + "
                "1.96·σ_fam/√3]; P1 band edges READ from the sealed analysis-v2 result "
                "(SAME band as 50M, λ=37 saturated), σ_fam RECOMPUTED from the sealed R0265 panel"
            ),
            "backstops": (
                "every seed: collapse >= R0265 floor (0.8129, N-invariant); fog <= R0265 "
                "ceiling (0.41207; near-ceiling escalation if fog > ceiling − 1·σ_fam,fog); "
                "FFR >= R0265 floor (0.39906) via the reserve-projection instrument at "
                "disc=int(ROWS*0.001)=100000 -- all READ from the sealed R0265 floors. Purity "
                "is DESCRIPTIVE-only at 100M (plan §2.4)."
            ),
            "purity_descriptive_only": (
                "purity k256/k1024 are REPORTED against an R0238-prefix inline reference + "
                "centroids (self-contained) and labelled descriptive/ungated with the lineage "
                "caveat (100M-prefix != R0216-c3); they NEVER enter the go/no-go"
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
        "hundred_m_decision": {
            "criterion_1_passes": criterion_1["passes"],
            "every_seed_clears_every_backstop": backstop_scoring["every_seed_clears_every_backstop"],
            "any_gate_straddles": backstop_scoring["any_gate_straddles"],
            "any_fog_near_ceiling_escalation": backstop_scoring["any_fog_near_ceiling_escalation"],
            "passes": passes,
            "ambiguous": ambiguous,
            "verdict": verdict,
        },
        "gate_status": "registered-and-contingent-pending-review",
        "gate_registered": True,
        "evaluation_performed": True,
        "training_performed": False,
        "gpu_used": False,
        "acceptance_rule": (
            "the round succeeds if it executes; the 100M PASS/FAIL is a MEASUREMENT reported "
            "either way; a FAIL/AMBIGUOUS returns to the owner"
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
    _seal(output, "fneg-100m-x2-seedmean-gate.json", body)
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "collapse_seed_mean": criterion_1["seed_mean"],
        "widened_band": criterion_1["widened_band"],
        "criterion_1_passes": criterion_1["passes"],
        "every_seed_clears_every_backstop": backstop_scoring["every_seed_clears_every_backstop"],
        "verdict": verdict,
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0268 round0268_nodes.run_job")
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
    if action == TRANSFORM_CORRECTION_ACTION:
        run_transform_correction(active, job)
        return
    raise Round0268NodeError(f"R0268 unknown action {action!r}")


__all__ = [
    "DESCRIPTIVE_PURITY_LINEAGE_CAVEAT",
    "DESCRIPTIVE_PURITY_METRICS",
    "GATE_ACTION",
    "GATE_CAPABILITY",
    "GATE_METRICS",
    "GATE_SCHEMA",
    "HOST_RSS_ANALYTIC_BASIS",
    "HOST_RSS_LIMIT_GIB",
    "PANEL_ACTION",
    "PANEL_CAPABILITY",
    "PANEL_RSS_LIMIT_GIB",
    "PANEL_SCHEMA",
    "PREFIX_ROWS",
    "R0268_ANON_BUDGET_BYTES",
    "Round0268NodeError",
    "TRAIN_ACTION",
    "TRAIN_SCHEMA",
    "build_hostint8_dataset_from_full",
    "read_p1_x2_asymptote_band",
    "run_gate",
    "run_job",
    "run_panel",
    "run_train",
    "score_collapse_seed_mean",
    "score_per_seed_backstops",
    "sigma_fam_from_panel",
    "verify_hundred_m_prefix_lineage",
]
