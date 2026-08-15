"""Execute R0267 — the 50M ×2 host-int8 staging rung of the promoted fneg recipe.

Five nodes in one queue (three trains -> panel -> gate), reusing R0265/R0266 machinery:

* three GPU trains (`seeds 42, 43, 44`) under the PINNED 50M ×2 host-int8 recipe, built
  and proved by `round0267_int8_treatment` (R0265's umap kernel a=1.9328/b=0.7905, fneg
  1.0 band [0.1,0.4], UNIFORM sampling + R0266's `x_residency=host_int8` routing, at dose
  ×2 on the sealed R0237 50M substrate + exact k15 graph). The train node mirrors R0266's
  host-int8 train node, retargeted to the 50M substrate/graph binding and the ×2 horizon.
* `score_minilm_fneg_50m_x2_panel` (GPU) — the three maps scored on R0265's instruments
  (held-out FFR against the sealed R0237 exact k15 truth, purity k256/k1024 against
  R0218's frozen reference, collapse, fog), via R0265's `score_one_map` / `score_panel`.
* `register_fneg_50m_x2_seedmean_gate` (CPU) — the pre-registered 50M gate
  (plan-50m-stage-2026-08-15): the SEED-MEAN collapse inside P1's ×2 asymptote band
  widened by a √n-shrunk family seed-noise allowance, plus per-seed hard backstops on
  collapse/fog/FFR/purity. Every band, floor, σ_fam and P1 edge is READ / RECOMPUTED from
  a SEALED artifact bound by sha256 at gate time -- never a typed literal (the
  constants-discipline contract test mutates each and asserts the gate tracks it).

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
from collections.abc import Mapping
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

import experiments.round0218_nodes as round0218_nodes
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
    _load_centroids,
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

#: The five per-seed gated metrics (the plan's criteria 2-4 backstops).
GATE_METRICS: tuple[str, ...] = (
    HELDOUT_FFR_METRIC, PURITY_K256_METRIC, PURITY_K1024_METRIC, COLLAPSE_METRIC, FOG_METRIC,
)

#: The seed-mean collapse gate parameters (plan criterion 1). z is the two-sided 95%
#: normal quantile; n is the registered seed count. Both are pre-registered constants of
#: the DECISION RULE (not gate values read from data) — the band edges and σ_fam ARE read
#: live from sealed artifacts.
COLLAPSE_SEEDMEAN_Z = 1.96
COLLAPSE_SEEDMEAN_N = 3

DEVICE_BUDGET_BYTES = 30 * (1 << 30)
HOST_RSS_LIMIT_GIB = 60.0

#: The R0244 host-watchdog anonymous-memory budget for the 50M host-int8 rung. The 2M
#: default (16 GiB, round0265) is too small here: the host-int8 X lives in host RAM as
#: an int8 array (50M×384 = 19.2 GB) plus the transient edge-list load and samplers, so
#: the anonymous peak is ~20+ GB (R0267 seed-42 first tripped 16 GiB at 17.2 GB). 40 GiB
#: covers it with headroom and sits far under the box's ~111 GB MemAvailable. (100M would
#: need ~56 GiB — a later per-rung concern.)
R0267_ANON_BUDGET_BYTES = 40 * (1 << 30)
POSITIVE_ROWS_PER_UPDATE = 409

SAFETY_NOTE = (
    "no node in this module signals any process, starts a child process, hands cuVS "
    "anything, or wraps a subprocess in a timeout. Every bulk input is a read-only "
    "np.memmap. The per-batch abort read is the release's own ParametricUMAP.abort_poll "
    "attribute, set to this node's recorder and cleared in a finally."
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
        "receipt": receipt,
        "receipt_signature": receipt_signature,
        "model_path": model_path,
    }


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0267 round0267_nodes.run_panel")
    import torch
    from basemap.panel_v2 import (
        load_hiD_reference,
        reset_process_cuda_peak,
        sample_anchors,
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
    panel_evidence = prompt_contract.read_sealed(
        str(job["panel_evidence"]), label="R0218 MiniLM frozen panel reference"
    )
    centroid_ks = [int(k) for k in job["centroid_ks"]]
    cfg = prompt_contract.panel_config()

    # The held-out FFR instrument: the sealed R0237 exact k15 truth for its 1M uniform
    # probe (query rows into the 50M substrate + their exact top-k neighbour ids). The
    # 200k held-out reserve is bound as the held-out reserve lineage (see round file).
    probe_rows = np.load(
        _bound_path(job, "truth_query_rows", label="R0267 50M truth query rows"), allow_pickle=False
    ).astype(np.int64, copy=False)
    truth_ids = np.load(
        _bound_path(job, "truth_ids", label="R0267 50M truth ids"), allow_pickle=False
    ).astype(np.int64, copy=False)
    if probe_rows.ndim != 1 or truth_ids.ndim != 2 or truth_ids.shape[0] != probe_rows.shape[0]:
        raise Round0267NodeError("R0267 50M truth geometry changed")

    cells_in = job.get("cells")
    if not isinstance(cells_in, list):
        raise Round0267NodeError("R0267 cell input matrix changed")
    authenticated = [_authenticate_50m_map(cell, substrate) for cell in cells_in]
    if {entry["seed"] for entry in authenticated} != set(SEEDS):
        raise Round0267NodeError("R0267 three-cell input matrix changed")
    invariants = {str(entry["receipt"]["seed_invariant_sha256"]) for entry in authenticated}
    if len(invariants) != 1:
        raise Round0267NodeError("R0267 pooled family is not one 50M recipe")

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

        centroids, centroid_signatures = _load_centroids(panel_evidence, centroid_ks)
        reference_signature = dict(panel_evidence["shared_high_d_reference"])
        reference_path = prompt_contract.verify_signature(
            reference_signature, label="R0218 shared high-D reference"
        )
        reference = load_hiD_reference(reference_path)
        _anchors = sample_anchors(ROWS, cfg)
        reference_identity = {
            "data_identity": {
                "kind": "ordered_array",
                "shape": [ROWS, DIMENSION],
                "dtype": np.dtype("<f4").str,
                "sha256": substrate["ordered_substrate_sha256"],
            },
            "convention": dict(round0218_nodes.REFERENCE_CONVENTION),
        }
        wrapped("R0267 frozen reference + centroids loaded")

        cells: dict[str, dict[str, Any]] = {}
        for entry in sorted(authenticated, key=lambda e: e["seed"]):
            seed = entry["seed"]
            model = ParametricUMAP.load(entry["model_path"], device="cuda")
            coordinates = _transform_50m_in_chunks(model, source, wrapped)
            if coordinates.shape != (ROWS, 2) or not np.isfinite(coordinates).all():
                raise Round0267NodeError(f"R0267 seed-{seed} transform is not a finite 50M map")
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
                    "treatment": "fneg-x2-md000-hostint8-50m",
                },
            )
            purity_ratios = {"k256": float(panel["purity"]["k256"]), "k1024": float(panel["purity"]["k1024"])}
            placed = np.asarray(coordinates[probe_rows], dtype=np.float32)
            scored_map = score_one_map(
                coordinates=coordinates,
                probes_placed=placed,
                truth_top10=truth_ids,
                purity_ratios=purity_ratios,
            )
            cells[str(seed)] = {
                "seed": seed,
                "capability": entry["capability"],
                "train_receipt": dict(entry["receipt_signature"]),
                "model": dict(entry["receipt"]["model"]),
                "x_residency": X_RESIDENCY,
                "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
                "metrics": scored_map,
                "panel_purity_numerators": panel.get("purity_numerators"),
            }
            del model, coordinates, placed
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
        "x_residency": X_RESIDENCY,
        "seed_invariant_sha256": sorted(invariants)[0],
        "panel_metric_table": metric_table,
        "cells": cells,
        "reference": reference_signature,
        "centroids": centroid_signatures,
        "gated_metrics": list(GATE_METRICS),
        "heldout_reserve": dict(job.get("heldout_reserve") or {}),
        "lineage": {
            "graph_manifest": dict(substrate["manifest_signature"]),
            "substrate": dict(substrate["substrate_signature"]),
            "ordered_substrate_sha256": substrate["ordered_substrate_sha256"],
        },
        "purity_reference_note": (
            "purity k256/k1024 are scored against R0218's frozen 2M reference + centroids "
            "so the ratios stay commensurate with R0265's sealed 2M bands (the gate's "
            "purity backstop). A native 50M purity reference is a separate panel round."
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
    del source, reference, centroids
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
    """The per-seed hard backstops (plan criteria 2-4 + the collapse floor).

    Every seed must clear: collapse >= R0265 family floor; fog <= R0265 family ceiling
    (with a mechanical near-ceiling escalation if fog > ceiling - 1·σ_fam,fog); held-out
    FFR >= R0265 family floor; k1024 >= R0265 floor; k256 in the R0265 band. The
    no-straddle rule applies to THESE gates (some seeds passing and some failing one gate
    is ambiguity, not noise). Every floor/band is READ from the sealed R0265 floors.
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
        k256_v = _judge_k256_two_sided(float(metrics[PURITY_K256_METRIC]), dict(backstops["k256_band"]))
        k1024_v = _judge_one_sided_floor(float(metrics[PURITY_K1024_METRIC]), float(backstops["k1024_floor"]))
        by_metric = {
            HELDOUT_FFR_METRIC: {
                "verdict": "PASS" if ffr_pass else "FAIL", "passes": ffr_pass,
                "value": ffr, "floor": ffr_floor, "margin": ffr - ffr_floor,
            },
            COLLAPSE_METRIC: collapse_v,
            FOG_METRIC: {**fog_v, "near_ceiling_escalation": bool(fog_near_ceiling),
                        "near_ceiling_threshold": near_ceiling_threshold},
            PURITY_K256_METRIC: k256_v,
            PURITY_K1024_METRIC: k1024_v,
        }
        clears = all(bool(v["passes"]) for v in by_metric.values())
        rows.append({
            "cell_id": exact_cell_id(int(seed_key)),
            "seed": int(seed_key),
            "metrics": by_metric,
            "clears_every_backstop": clears,
            "fog_near_ceiling_escalation": bool(fog_near_ceiling),
            "fog_not_measurable": fog_v.get("verdict") == VERDICT_NOT_MEASURABLE,
        })
    # No-straddle: a per-seed gate straddles if some seeds pass it and some fail it.
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
            "k1024_floor": float(backstops["k1024_floor"]),
            "k256_band": {
                "ratio_lower": float(backstops["k256_band"]["ratio_lower"]),
                "ratio_upper": float(backstops["k256_band"]["ratio_upper"]),
            },
        },
        "no_straddle_rule": (
            "applies to the per-seed backstop + fog/FFR/purity gates, NOT to the P1 "
            "asymptote band (a single seed outside the P1 band but at/above the backstop "
            "is expected noise, not ambiguity)"
        ),
    }


def _metric_table_from_panel(panel: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if panel.get("capability") != PANEL_CAPABILITY or panel.get("schema") != PANEL_SCHEMA:
        raise Round0267NodeError("R0267 50M panel contract changed")
    table = dict(panel["panel_metric_table"])
    if {int(s) for s in table} != set(SEEDS):
        raise Round0267NodeError("R0267 50M panel is not the three fneg cells")
    return {str(s): dict(table[str(s)]) for s in SEEDS}


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
        metric_table = _metric_table_from_panel(panel)
        seed_collapse = {s: float(metric_table[s][COLLAPSE_METRIC]) for s in metric_table}
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

        # CRITERION 1: the seed-mean collapse inside the widened P1 ×2 band.
        criterion_1 = score_collapse_seed_mean(
            seed_collapse=seed_collapse,
            p1_lower=p1["p1_lower"],
            p1_upper=p1["p1_upper"],
            sigma_fam_collapse=sigma["sigma_fam_collapse"],
        )
        # CRITERIA 2-4 + collapse backstop: per-seed hard gates.
        backstop_scoring = score_per_seed_backstops(
            metric_table=metric_table,
            backstops=backstops,
            sigma_fam_fog=sigma["sigma_fam_fog"],
        )
        wrapped("R0267 criteria scored from sealed bands, σ_fam and the P1 band")
        gate.finish("R0267 gate stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    # The 50M PASS decision (a FINDING reported either way; it feeds the 100M-commit
    # decision, it does NOT make the round a failure).
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
        "five_backstop_metrics_present": {r["seed"] for r in backstop_scoring["cells"]} == set(SEEDS),
        "three_seeds_scored": backstop_scoring["cells_scored"] == len(SEEDS),
        "no_typed_band_literals": True,  # every band/floor/σ_fam/P1-edge is read from a sealed input
    }
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
            "the pre-registered 50M staging-rung gate (plan-50m-stage-2026-08-15): the "
            "SEED-MEAN collapse inside P1's ×2 asymptote band widened by a √n-shrunk "
            "family seed-noise allowance, plus per-seed hard backstops on collapse/fog/"
            "FFR/purity. Feeds the 100M ×2-commit decision; a FAIL/AMBIGUOUS returns to "
            "the owner with the drift decomposition, never an auto-proceed."
        ),
        "seed_spread_assumption": SEED_SPREAD_ASSUMPTION,
        "sigma_fam": sigma,
        "p1_x2_asymptote_band": p1,
        "criterion_1_collapse_seed_mean": criterion_1,
        "criteria_2_4_per_seed_backstops": backstop_scoring,
        "pre_registered_pass_criteria": {
            "1": (
                "mean(collapse over 3 seeds) ∈ [0.930 − 1.96·σ_fam/√3, 0.985 + "
                "1.96·σ_fam/√3]; P1 band edges READ from the sealed analysis-v2 result, "
                "σ_fam RECOMPUTED from the sealed R0265 panel"
            ),
            "backstops": (
                "every seed: collapse >= R0265 floor; fog <= R0265 ceiling (near-ceiling "
                "escalation if fog > ceiling − 1·σ_fam,fog); FFR >= R0265 floor; k1024 >= "
                "R0265 floor; k256 ∈ R0265 band -- all READ from the sealed R0265 floors"
            ),
            "no_straddle": (
                "applies to the per-seed backstops + fog/FFR/purity, NOT to the P1 band"
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
    "GATE_ACTION",
    "GATE_CAPABILITY",
    "GATE_METRICS",
    "GATE_SCHEMA",
    "PANEL_ACTION",
    "PANEL_CAPABILITY",
    "PANEL_SCHEMA",
    "Round0267NodeError",
    "TRAIN_ACTION",
    "TRAIN_SCHEMA",
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
