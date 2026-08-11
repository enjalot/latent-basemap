"""Execute R0257 — the Phase 2 ladder's `6250k` rung maps, judged by the `n = 29`
`MAD_n` gate that was fitted on the 2M universe and never on these.

Five nodes:

* `train_seed42`, `train_seed43`, `train_seed44` (GPU) — one rung map each, on
  R0233's sealed `6,250,000`-row substrate and its sealed `cluster-spill-nnd` k15
  fuzzy graph, under R0217's treatment with only the rung, the graph and the seed
  moved. `rung_invariant_sha256` asserts that byte-for-byte against R0217's own
  template.
* `panel_6250k` (GPU) — the rung universe's own purity centroids and high-D
  reference (R0218's recipe, unchanged), then all three maps scored with the
  accepted `panel_v2` config.
* `judge_6250k` (CPU) — reads the SEALED `n = 29` floors artifact and applies its
  `registered_criteria` verbatim. It fits nothing, writes no floor, and adds no map
  to any family: `assert_no_rung_map_in_the_gate_family` calls the shipped
  `round0255_treatment.assert_family_is_2m_only` on the gate's defining family and
  asserts disjointness from the judged set, with a positive control per judged map.

**Coverage is earned by the layout (R0254).** Every node constructs its poll gate
and its coverage window BEFORE its expensive work; the train nodes hand the wrapped
poll to `ParametricUMAP.abort_poll` so `fit()` is inside the window and polls once
per batch, and the full-population transform is walked in chunks. **Every node
seals an artifact whose FILENAME CONTAINS ITS NODE ID and whose top level carries
`observed_span_s`** — the attribution R0255 lost by naming its artifacts
`train-receipt.json`.

Nothing here signals any process, starts a child process, hands cuVS anything, or
wraps a subprocess in a timeout. Every bulk input is a read-only `np.memmap`.
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
from basemap.round0217_minilm_2m_pipeline import MiniLMHostFp32EndpointArray
from basemap.round0217_minilm_2m_seed_family import (
    WARMUP_SUCCESSFUL_UPDATES,
    successful_updates_for_edges,
    validate_dose,
)
from basemap.round0218_minilm_2m_panel import (
    CENTROID_ITERS,
    CENTROID_KS,
    CENTROID_SEED,
    panel_execution_ok,
    panel_metric_view,
)
from basemap.round0233_substrate import COMPOSITION
from basemap.round0238_rung5 import json_safe
from basemap.round0242_locality import json_scrub
from basemap.round0247_registry import registry_fingerprint
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report
from basemap.round0253_coverage import CoverageLedger
from basemap.round0253_stop_hooks import install_stop_hooks
from basemap.round0255_seed_extension_n29 import (
    BATCH_SIZE,
    DIMENSION,
    FULL_TRANSFORM_BATCH,
    POSITIVE_ROWS_PER_UPDATE,
)
from basemap.round0255_treatment import (
    Round0255FamilyError,
    assert_family_is_2m_only,
)
from basemap.round0257_judgement import (
    DESCRIPTIVE_METRICS,
    GATED_METRICS,
    GATE_CAPABILITY,
    INDEPENDENCE_LIMITATION,
    PANEL_FALSE_ALARM_RATE,
    POWER_MATERIALITY,
    REGISTERED_FINGERPRINT,
    REGISTERED_N,
    Round0257JudgementError,
    judge_population,
    judgement_controls,
    validate_gate_artifact,
)
from basemap.round0257_rung_contract import (
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    GRAPH_SCHEMA,
    PANEL_CAPABILITY,
    PANEL_SCHEMA,
    PIPELINE_IDENTITY_NOTE,
    PRODUCTION_CONFIG_SCHEMA,
    REGISTERED_SUCCESSFUL_UPDATES,
    REGISTERED_UPDATE_BOUND,
    ROUND_ID,
    RUNG_ROWS,
    RUNG_SLUG,
    RUNG_SOURCE_ROUND_ID,
    Round0257Error,
    Round0257FamilyError,
    SEALED_RUNG_DIRECTED_EDGES,
    SEALED_TIE_AWARE_RECALL,
    SEALED_ZERO_DEGREE_ROWS,
    SEEDS,
    SUBSTRATE_CAPABILITY,
    SUBSTRATE_SCHEMA,
    TRAIN_SCHEMA,
    VERDICT_CAPABILITY,
    VERDICT_SCHEMA,
    assert_no_rung_map_in_the_gate_family,
    dose_view,
    map_capability,
    rung_cell_id,
    rung_cell_ids,
    rung_family_purity_controls,
    rung_invariant_sha256,
    rung_train_config,
)
from basemap.round0257_rung_pipeline import (
    DEVICE_BUDGET_BYTES,
    DEVICE_PEAK_PREDICTION_NOTE,
    HOST_ANON_BUDGET_BYTES,
    HOST_RSS_LIMIT_GIB,
    RungMixedTrainingInput,
    predict_rung_footprint,
    validate_full_rung_map,
)
from experiments.round0113_nodes import _new_model
from experiments.round0230_nodes import (
    CellWatchdog,
    _weighted_rejection_accounting_mismatch,
)
from experiments.round0255_nodes import (
    _guard_tail_reported,
    _node_gate,
    _node_guard,
    _score_gate_without_raising,
    _start_node,
)

TRAIN_ACTION = f"train_minilm_mixed_{RUNG_SLUG}_ladder_map"
PANEL_ACTION = f"score_minilm_mixed_{RUNG_SLUG}_ladder_panel"
JUDGE_ACTION = f"judge_minilm_mixed_{RUNG_SLUG}_against_the_n29_gate"

#: Rows per full-population transform chunk. Numerically inert (the projection is
#: row-wise); it exists so the transform is inside the coverage window.
TRANSFORM_CHUNK_ROWS = 100_000

#: R0233's registered corpus order. Index i in the provenance array is
#: `COMPOSITION[i]`, asserted against the sealed per-corpus row counts.
CORPUS_SLUGS: tuple[str, ...] = tuple(name for name, _rows in COMPOSITION)
CORPUS_ROWS: dict[str, int] = {name: int(rows) for name, rows in COMPOSITION}

SAFETY_NOTE = (
    "no node in this module signals any process, starts a child process, hands "
    "cuVS anything, or wraps a subprocess in a timeout. Every bulk input is a "
    "read-only np.memmap. The per-batch abort read is the release's own "
    "`ParametricUMAP.abort_poll` attribute, set to this node's recorder and "
    "cleared in a finally. Host memory is guarded on ANONYMOUS bytes, never RSS."
)


# --------------------------------------------------------------------------- #
# bound inputs
# --------------------------------------------------------------------------- #


def _bound_path(job: Mapping[str, Any], key: str, *, label: str) -> str:
    signature = job.get(key)
    if not isinstance(signature, Mapping):
        raise Round0257Error(f"{label} is not bound into the job as {key!r}")
    return prompt_contract.verify_signature(dict(signature), label=label)


def _intra_queue_signature(
    reference: Mapping[str, Any], *, label: str
) -> tuple[str, dict[str, Any]]:
    """Resolve a reference to an artifact THIS queue produces.

    A node downstream of a producer cannot be handed a hash at prepare time,
    because the bytes do not exist yet. R0255 established the shape: if the
    reference carries a `sha256` it is verified in full; otherwise the path is
    resolved and its signature computed at read time. Using `verify_signature`
    on a hashless intra-queue reference is what failed this round's first panel
    attempt.
    """
    reference = dict(reference)
    if reference.get("sha256"):
        return prompt_contract.verify_signature(reference, label=label), reference
    path = str(reference.get("canonical_path") or "")
    if not path or not os.path.exists(path):
        raise Round0257Error(f"{label} is absent at {path!r}")
    return path, expected_input_signature(path)


def _read_json(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise Round0257Error(f"{label} is absent at {path}")
    with open(path, "rb") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise Round0257Error(f"{label} is not a JSON object")
    return payload


def _sealed_rung_substrate(job: Mapping[str, Any]) -> dict[str, Any]:
    """R0233's sealed rung substrate receipt, re-asserted field by field."""
    path = _bound_path(
        job, "substrate_manifest_signature", label="R0233 sealed rung substrate receipt"
    )
    manifest = _read_json(path, "R0233 sealed rung substrate receipt")
    if (
        str(manifest.get("schema")) != SUBSTRATE_SCHEMA
        or str(manifest.get("round_id")) != RUNG_SOURCE_ROUND_ID
        or str(manifest.get("capability")) != SUBSTRATE_CAPABILITY
        or int(manifest.get("rows", -1)) != RUNG_ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or manifest.get("training_performed") is not False
    ):
        raise Round0257Error("R0257 sealed rung substrate contract changed")
    composition = manifest.get("composition") or {}
    for slug, rows in CORPUS_ROWS.items():
        entry = composition.get(slug) or {}
        if int(entry.get("rows", -1)) != rows:
            raise Round0257Error(
                f"R0257 rung composition changed: {slug} has "
                f"{entry.get('rows')!r} rows, registered {rows}"
            )
    return {
        "manifest": manifest,
        "manifest_signature": expected_input_signature(path),
        "substrate_signature": dict(manifest["substrate"]),
        "provenance_signature": dict(manifest["provenance"]),
        "ordered_substrate_sha256": str(manifest["ordered_substrate_sha256"]),
    }


def _open_rung_substrate(sealed: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Serve the 9.6 GB sealed rung substrate lazily; never materialize it."""
    signature = dict(sealed["substrate_signature"])
    path = prompt_contract.verify_signature(
        signature, label="R0233 sealed rung substrate"
    )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (RUNG_ROWS, DIMENSION) or array.dtype != np.float32:
        raise Round0257Error("R0257 sealed rung substrate geometry changed")
    return array, signature


def _corpus_of_row(sealed: Mapping[str, Any]) -> np.ndarray:
    path = prompt_contract.verify_signature(
        dict(sealed["provenance_signature"]), label="R0233 sealed rung provenance"
    )
    provenance = np.load(path, mmap_mode="r", allow_pickle=False)
    if provenance.shape != (RUNG_ROWS,) or provenance.dtype.names != (
        "corpus",
        "shard",
        "row",
    ):
        raise Round0257Error("R0257 sealed rung provenance layout changed")
    corpus = np.asarray(provenance["corpus"], dtype=np.int64)
    counts = np.bincount(corpus, minlength=len(CORPUS_SLUGS))
    if len(counts) != len(CORPUS_SLUGS):
        raise Round0257Error("R0257 rung provenance carries an unregistered corpus id")
    for index, slug in enumerate(CORPUS_SLUGS):
        if int(counts[index]) != CORPUS_ROWS[slug]:
            raise Round0257Error(
                f"R0257 rung corpus {slug} has {int(counts[index])} rows, "
                f"registered {CORPUS_ROWS[slug]}"
            )
    del provenance
    return corpus


def _sealed_rung_graph(job: Mapping[str, Any]) -> dict[str, Any]:
    """R0233's sealed rung graph receipt and its qualification, re-asserted."""
    path = _bound_path(
        job, "graph_manifest_signature", label="R0233 sealed rung graph receipt"
    )
    manifest = _read_json(path, "R0233 sealed rung graph receipt")
    if (
        str(manifest.get("schema")) != GRAPH_SCHEMA
        or str(manifest.get("round_id")) != RUNG_SOURCE_ROUND_ID
        or str(manifest.get("capability")) != GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != RUNG_ROWS
        or int(manifest.get("k", -1)) != 15
        or manifest.get("training_performed") is not False
    ):
        raise Round0257Error("R0257 sealed rung graph contract changed")
    edges = int(manifest.get("directed_edges", -1))
    if edges != SEALED_RUNG_DIRECTED_EDGES:
        raise Round0257Error(
            f"R0257 sealed rung graph reports {edges} directed edges, registered "
            f"{SEALED_RUNG_DIRECTED_EDGES}"
        )
    degrees = manifest.get("degrees") or {}
    selected = manifest.get("selected_graph") or {}
    zero_degree = int(degrees.get("zero_degree_rows", -1))
    if zero_degree != SEALED_ZERO_DEGREE_ROWS:
        raise Round0257Error(
            f"R0257 R0215 degree-zero tripwire: sealed rung graph reports "
            f"{zero_degree} edgeless rows"
        )
    tie_aware_block = selected.get("tie_aware")
    if not isinstance(tie_aware_block, Mapping):
        raise Round0257Error(
            "R0257 sealed rung graph carries no tie-aware recall block"
        )
    tie_aware = float(tie_aware_block["mean"])
    tie_aware_p10 = float(tie_aware_block["p10"])
    if int(tie_aware_block.get("n", -1)) != RUNG_ROWS:
        raise Round0257Error(
            "R0257 sealed rung graph recall was not measured over all rung rows"
        )
    floors = manifest.get("floors") or {}
    floor = float(floors.get("tie_aware_mean", 0.9))
    p10_floor = float(floors.get("tie_aware_p10", 0.8))
    if tie_aware < floor or tie_aware_p10 < p10_floor:
        raise Round0257Error(
            f"R0257 sealed rung graph tie-aware recall mean {tie_aware} / p10 "
            f"{tie_aware_p10} is below its registered floors {floor} / {p10_floor}"
        )
    if abs(tie_aware - SEALED_TIE_AWARE_RECALL) > 1e-6:
        raise Round0257Error(
            f"R0257 sealed rung graph tie-aware recall {tie_aware} is not the "
            f"registered {SEALED_TIE_AWARE_RECALL}"
        )
    graph_signature = dict(manifest["graph"])
    graph_path = prompt_contract.verify_signature(
        graph_signature, label="R0233 sealed rung fuzzy graph"
    )
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )

    sources, targets, weights, n_nodes = load_edge_arrays(graph_path, load_weights=True)
    if (
        weights is None
        or int(n_nodes) != RUNG_ROWS
        or len(sources) != edges
        or targets.shape != sources.shape
        or weights.shape != sources.shape
        or sources.dtype != np.int32
        or targets.dtype != np.int32
        or weights.dtype != np.float32
    ):
        raise Round0257Error("R0257 sealed rung graph arrays changed")
    return {
        "manifest": manifest,
        "manifest_signature": expected_input_signature(path),
        "signature": graph_signature,
        "directed_edges": edges,
        "zero_degree_rows": zero_degree,
        "tie_aware_recall_at_k": tie_aware,
        "tie_aware_p10": tie_aware_p10,
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": int(n_nodes),
    }


# --------------------------------------------------------------------------- #
# receipts
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


def _seal(output: str, name: str, body: Mapping[str, Any]) -> str:
    path = os.path.join(output, name)
    atomic_write_new_json(
        path, prompt_contract.seal(json_safe(json_scrub(dict(body)))), immutable=True
    )
    return path


def _node_id(active: Mapping[str, Any], fallback: str) -> str:
    return str(active.get("node_id") or fallback)


def _coverage(ledger: CoverageLedger, node_wall_s: float) -> dict[str, Any]:
    """R0253's shipped coverage receipt, against this node's own wall.

    The denominator is the runner-visible node wall, which includes interpreter
    start and imports — the smaller, honest fraction R0256 chose to quote rather
    than the stage-only one.
    """
    return {
        **ledger.receipt(node_wall_s=float(node_wall_s)),
        "what_covered_wall_time_does_not_mean": (
            "Covered is not bounded. A window's span is the interval over which "
            "reads happened; the widest gap inside it is a separate number and is "
            "published beside it in `enforcement_poll_spacing` and `gap_report`."
        ),
    }


# --------------------------------------------------------------------------- #
# node 1-3 — one rung map per seed
# --------------------------------------------------------------------------- #


def _seed(job: Mapping[str, Any]) -> int:
    seed = job.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Round0257Error(f"R0257 job seed {seed!r} is not an integer")
    if seed not in SEEDS:
        raise Round0257Error(f"R0257 job seed {seed!r} is not a registered rung cell")
    if str(job.get("capability") or "") != map_capability(seed):
        raise Round0257Error("R0257 job capability does not match its seed")
    return int(seed)


def _transform_in_chunks(model: Any, source: Any, poll: Any) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, RUNG_ROWS, TRANSFORM_CHUNK_ROWS):
        stop = min(start + TRANSFORM_CHUNK_ROWS, RUNG_ROWS)
        parts.append(
            np.asarray(
                model.transform(source[start:stop], batch_size=FULL_TRANSFORM_BATCH),
                dtype=np.float32,
            )
        )
        poll(f"R0257 transform rows {start}-{stop}")
    return np.concatenate(parts, axis=0)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0257 round0257_nodes.run_train")
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0257Error("R0257 train handler received another queue")
    seed = _seed(job)
    capability = map_capability(seed)
    cell_id = rung_cell_id(seed)
    node_id = _node_id(active, f"train_seed{seed}")
    label = f"R0257 {capability}"
    ledger = CoverageLedger(node=node_id)
    node_started = time.monotonic()
    abort_flag = _start_node(label)

    prediction = predict_rung_footprint(seed)
    declared = job.get("memory_prediction")
    if declared is not None and dict(declared) != prediction:
        raise Round0257Error(
            "R0257 cell prediction differs from the one sealed at prepare time"
        )

    sealed_substrate = _sealed_rung_substrate(job)
    graph = _sealed_rung_graph(job)
    edges = graph["directed_edges"]
    updates = successful_updates_for_edges(edges)
    if updates != REGISTERED_SUCCESSFUL_UPDATES or updates > REGISTERED_UPDATE_BOUND:
        raise Round0257Error(
            f"R0257 derived horizon {updates} is not the registered "
            f"{REGISTERED_SUCCESSFUL_UPDATES} within the bound "
            f"{REGISTERED_UPDATE_BOUND}"
        )
    dose = validate_dose(updates=updates, edge_count=edges)
    source, substrate_signature = _open_rung_substrate(sealed_substrate)

    r0217 = job.get("r0217_template_signatures")
    if not isinstance(r0217, Mapping):
        raise Round0257Error("R0257 R0217 template signatures are not bound")
    config, config_sha, invariant = rung_train_config(
        seed=seed,
        rows=RUNG_ROWS,
        graph_edges=edges,
        substrate_signature=substrate_signature,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        r0217_substrate_signature=dict(r0217["substrate"]),
        r0217_graph_signature=dict(r0217["graph"]),
        r0217_graph_manifest_signature=dict(r0217["graph_manifest"]),
    )
    declared_invariant = str(job.get("rung_invariant_sha256") or "")
    if not declared_invariant or invariant != declared_invariant:
        raise Round0257Error(
            "R0257 rung cell config is not the treatment sealed at prepare time: "
            f"{invariant} != {declared_invariant}"
        )

    output = create_fresh_directory(str(job["outputs"][0]), label="R0257 train output")
    atomic_write_new_json(
        os.path.join(output, f"{node_id}-production-config.json"),
        {
            "schema": PRODUCTION_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "treatment_config_round_id": "0217",
            "rung": RUNG_SLUG,
            "rows": RUNG_ROWS,
            "seed": seed,
            "capability": capability,
            "cell_id": cell_id,
            "is_a_gate_family_cell": False,
            "rung_invariant_sha256": invariant,
            "pipeline_identity_note": PIPELINE_IDENTITY_NOTE,
            "config": config,
            "config_sha256": config_sha,
        },
        immutable=True,
    )

    dataset = MiniLMHostFp32EndpointArray(
        source, source_signature=substrate_signature, buffer_rows=BATCH_SIZE
    )
    wrapper = RungMixedTrainingInput(dataset, graph, seed=seed, rows=RUNG_ROWS)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = _new_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = config["execution"]["performance_windows"]
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, f"{node_id}-admission.json")

    window = ledger.window(f"R0257 {capability} train stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=True)
    watchdog = CellWatchdog()
    watchdog.start()
    started = time.monotonic()
    try:
        with guard_ctx:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor(f"R0257 {capability} stage entered")
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
            wrapped("R0257 fit() returned")
            accounting = dict(model._train_stats)
            runtime = wrapper.runtime_stamp()
            wrapped("R0257 train accounting read")
            model_path = os.path.join(output, f"{node_id}-model.pt")
            from basemap.output_safety import atomic_build_new_file

            atomic_build_new_file(model_path, model.save, immutable=True)
            wrapped("R0257 checkpoint published")
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
            wrapped("R0257 training objects released")

            from basemap.pumap.parametric_umap import ParametricUMAP

            reloaded = ParametricUMAP.load(model_path, device="cuda")
            wrapped("R0257 checkpoint reloaded")
            coordinates = _transform_in_chunks(reloaded, source, wrapped)
            published = validate_full_rung_map(coordinates)
            published["model"] = expected_input_signature(model_path)
            coordinates_ordered_sha256 = ordered_array_sha256(coordinates)
            del reloaded, coordinates
            torch.cuda.empty_cache()
            gc.collect()
            gate.finish(f"R0257 {capability} stage end")
        window.close()
        tail = _guard_tail_reported(guard_ctx, label=label)
        scored = _score_gate_without_raising(gate, tail, label=label)
        gaps = gap_report(recorder.records, arm=node_id)
    finally:
        watchdog_state = watchdog.stop()

    if watchdog_state["tripped"]:
        raise Round0257Error(
            f"R0257 seed-{seed} watchdog tripped: {watchdog_state['trip_reason']!r}"
        )

    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_stamp.items()
        if runtime.get(key) != value
    }
    exact = {
        "lr_horizon": updates,
        "positive_lr_optimizer_steps": updates,
        "scheduler_steps": updates,
        "attempted_batches": updates,
        "finite_loss_batches": updates,
        "optimizer_steps_attempted": updates,
        "optimizer_steps_succeeded": updates,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": edges,
    }
    mismatches.update({
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    })
    expected_rows = updates * BATCH_SIZE
    producer_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        int(runtime["source_rows_gathered"]) != expected_rows
        or int(runtime["destination_rows_gathered"]) != expected_rows
        or int(runtime["host_prefetch_consumer_batches"]) != updates
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows,
            "runtime": runtime,
        }
    weighted = _weighted_rejection_accounting_mismatch(
        runtime, producer_delta=producer_delta, updates=updates
    )
    if weighted is not None:
        mismatches["weighted_rejection_accounting"] = weighted
    if mismatches:
        raise Round0257Error(f"R0257 train accounting failed: {mismatches}")
    prompt_contract.synchronize_runtime_counters(accounting, runtime)
    accounting["pipeline_runtime"] = dict(runtime)

    if int(memory["peak_reserved_bytes"]) > DEVICE_BUDGET_BYTES:
        raise Round0257Error(
            f"R0257 seed-{seed} peak reserved device bytes "
            f"{memory['peak_reserved_bytes']} exceed the {DEVICE_BUDGET_BYTES} budget"
        )
    if int(watchdog_state["peak_anonymous_bytes"]) > HOST_ANON_BUDGET_BYTES:
        raise Round0257Error(
            f"R0257 seed-{seed} peak anonymous bytes "
            f"{watchdog_state['peak_anonymous_bytes']} exceed the "
            f"{HOST_ANON_BUDGET_BYTES} budget"
        )
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0257Error(
            f"R0257 train peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    memory["peak_host_rss_gib"] = peak_rss_gib
    memory["peak_host_anonymous_bytes"] = int(watchdog_state["peak_anonymous_bytes"])

    achieved_upd_s = updates / wall if wall > 0 else 0.0
    floor = float(config["execution"]["minimum_train_upd_s"])
    if achieved_upd_s < floor:
        raise Round0257Error(
            f"R0257 seed-{seed} sustained {achieved_upd_s:.3f} upd/s against the "
            f"registered {floor} floor"
        )

    node_wall = time.monotonic() - node_started
    receipt = {
        **_receipt_envelope(active["manifest"]),
        "schema": TRAIN_SCHEMA,
        "capability": capability,
        "capabilities": [capability],
        "cell_id": cell_id,
        "node": node_id,
        "rung": RUNG_SLUG,
        "rows": RUNG_ROWS,
        "seed": seed,
        "is_a_gate_family_cell": False,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "this_map_is_judged_not_fitted": (
            "This cell is scored AGAINST the registered n = 29 MAD_n criteria and "
            "is never added to the family they are fitted on."
        ),
        "production_config_sha256": config_sha,
        "rung_invariant_sha256": invariant,
        "pipeline_identity_note": PIPELINE_IDENTITY_NOTE,
        "directed_edges": edges,
        "dose": dict(dose),
        "dose_view": dose_view(edges),
        "successful_updates": updates,
        "train_accounting": accounting,
        "memory": memory,
        "memory_prediction": prediction,
        "device_peak_prediction_note": DEVICE_PEAK_PREDICTION_NOTE,
        "device_peak_prediction_holds": (
            int(memory["peak_reserved_bytes"]) <= DEVICE_BUDGET_BYTES
        ),
        "performance": {
            "train_wall_s": wall,
            "achieved_upd_s": achieved_upd_s,
            "registered_minimum_upd_s": floor,
            "measured_at_this_rung": True,
        },
        "model": dict(published["model"]),
        "published_map": {
            key: value for key, value in published.items() if key != "model"
        },
        "coordinates_ordered_sha256": coordinates_ordered_sha256,
        "substrate": dict(substrate_signature),
        "substrate_manifest": dict(sealed_substrate["manifest_signature"]),
        "ordered_substrate_sha256": sealed_substrate["ordered_substrate_sha256"],
        "graph": dict(graph["signature"]),
        "graph_manifest": dict(graph["manifest_signature"]),
        "zero_degree_rows": graph["zero_degree_rows"],
        "graph_tie_aware_recall_at_k": graph["tie_aware_recall_at_k"],
        "watchdog": watchdog_state,
        "guard_tail": tail,
        "enforcement_poll_spacing": scored,
        "gap_report": gaps,
        "abort_flag_precondition": abort_flag,
        "poll_coverage": _coverage(ledger, node_wall),
        "observed_span_s": float(ledger.observed_span_s()),
        "node_wall_s": node_wall,
        "training_performed": True,
        "evaluation_performed": False,
        "production_or_publishing": False,
    }
    _seal(output, f"{node_id}-train-receipt.json", receipt)
    del source
    gc.collect()


# --------------------------------------------------------------------------- #
# node 4 — the rung panel
# --------------------------------------------------------------------------- #


REFERENCE_CONVENTION = {
    "row_order": f"R{RUNG_SOURCE_ROUND_ID} {RUNG_SLUG} substrate row order",
    "distance": "fp32 cosine on L2-normalized rows",
    "anchor_namespace": f"R{RUNG_SOURCE_ROUND_ID} {RUNG_SLUG} substrate row IDs",
}


def _authenticate_rung_map(
    cell: Mapping[str, Any], sealed: Mapping[str, Any]
) -> tuple[int, dict[str, Any], dict[str, Any], str]:
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0257Error(f"R0257 panel cell seed {seed!r} is not registered")
    receipt_path, receipt_signature = _intra_queue_signature(
        dict(cell["train_receipt"]), label=f"R0257 seed-{seed} rung train receipt"
    )
    receipt = prompt_contract.read_sealed(
        receipt_path, label=f"R0257 seed-{seed} rung train receipt"
    )
    if (
        str(receipt.get("round_id")) != ROUND_ID
        or str(receipt.get("schema")) != TRAIN_SCHEMA
        or int(receipt.get("seed", -1)) != seed
        or int(receipt.get("rows", -1)) != RUNG_ROWS
        or int(receipt.get("directed_edges", -1)) != SEALED_RUNG_DIRECTED_EDGES
        or receipt.get("is_a_gate_family_cell") is not False
    ):
        raise Round0257Error(f"R0257 seed-{seed} train receipt contract changed")
    if dict(receipt.get("substrate") or {}) != dict(sealed["substrate_signature"]):
        raise Round0257Error(
            f"R0257 seed-{seed} was not trained on the substrate this panel scores"
        )
    model_path = prompt_contract.verify_signature(
        dict(receipt["model"]), label=f"R0257 seed-{seed} rung checkpoint"
    )
    return seed, receipt, receipt_signature, model_path


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0257 round0257_nodes.run_panel")
    import torch
    from basemap.panel_v2 import (
        build_hiD_reference,
        reset_process_cuda_peak,
        sample_anchors,
        save_hiD_reference,
        score_panel,
    )
    from basemap.pumap.parametric_umap import ParametricUMAP
    from experiments.score_complete_panel import frozen_centroids

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0257Error("R0257 panel handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0257Error("R0257 panel requires CUDA")

    node_id = _node_id(active, f"panel_{RUNG_SLUG}")
    label = f"R0257 {RUNG_SLUG} ladder panel"
    ledger = CoverageLedger(node=node_id)
    node_started = time.monotonic()
    abort_flag = _start_node(label)

    sealed = _sealed_rung_substrate(job)
    source, substrate_signature = _open_rung_substrate(sealed)
    corpus_of_row = _corpus_of_row(sealed)

    cells_in = job.get("cells")
    if not isinstance(cells_in, list) or {
        int(cell.get("seed", -1)) for cell in cells_in
    } != set(SEEDS):
        raise Round0257Error("R0257 panel input matrix changed")
    authenticated = {}
    for cell in cells_in:
        seed, receipt, receipt_signature, model_path = _authenticate_rung_map(
            cell, sealed
        )
        authenticated[seed] = {
            "receipt": receipt,
            "receipt_signature": receipt_signature,
            "model_path": model_path,
        }
    invariants = {
        str(authenticated[seed]["receipt"]["rung_invariant_sha256"]) for seed in SEEDS
    }
    if len(invariants) != 1:
        raise Round0257Error(
            "R0257 scored rung maps are not commensurate: "
            f"{len(invariants)} rung-invariant config digests"
        )
    model_hashes = {
        str(authenticated[seed]["receipt"]["model"]["sha256"]) for seed in SEEDS
    }
    if len(model_hashes) != len(SEEDS):
        raise Round0257Error("R0257 scored rung maps contain a duplicated checkpoint")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0257 rung panel")

    window = ledger.window(f"R0257 {RUNG_SLUG} panel stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    started = time.monotonic()
    reset_process_cuda_peak()
    cells: dict[int, dict[str, Any]] = {}
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor(f"R0257 {RUNG_SLUG} panel stage entered")
        wrapped = window.wrap(recorder)

        cfg = prompt_contract.panel_config()
        centroid_root = create_fresh_directory(
            os.path.join(output, "centroids"), label="R0257 rung purity centroids"
        )
        centroids = frozen_centroids(
            source, CENTROID_KS, centroid_root, seed=CENTROID_SEED, iters=CENTROID_ITERS
        )
        wrapped("R0257 rung purity centroids built")
        centroid_signatures = {
            str(k): expected_input_signature(
                os.path.join(centroid_root, f"centroids_k{k}.npy")
            )
            for k in CENTROID_KS
        }
        reference_identity = {
            "data_identity": {
                "kind": "ordered_array",
                "shape": [RUNG_ROWS, DIMENSION],
                "dtype": np.dtype("<f4").str,
                "sha256": sealed["ordered_substrate_sha256"],
            },
            "convention": dict(REFERENCE_CONVENTION),
        }
        anchors = sample_anchors(RUNG_ROWS, cfg)
        reference = build_hiD_reference(
            source, anchors, cfg, centroids, **reference_identity
        )
        wrapped("R0257 rung high-D reference built")
        reference_path = os.path.join(
            output, f"{node_id}-{RUNG_SLUG}-high-d-reference.npz"
        )
        save_hiD_reference(reference, reference_path)
        reference_signature = expected_input_signature(reference_path)
        anchor_labels = np.asarray(
            [CORPUS_SLUGS[int(value)] for value in corpus_of_row[anchors]], dtype="U48"
        )
        anchor_corpus_counts = {
            slug: int((anchor_labels == slug).sum()) for slug in CORPUS_SLUGS
        }
        if any(count <= 0 for count in anchor_corpus_counts.values()):
            raise Round0257Error(
                f"R0257 anchor sample misses a corpus: {anchor_corpus_counts}"
            )

        for seed in SEEDS:
            entry = authenticated[seed]
            model = ParametricUMAP.load(entry["model_path"], device="cuda")
            coordinates = _transform_in_chunks(model, source, wrapped)
            validate_full_rung_map(coordinates)
            coordinates_path = os.path.join(
                output, f"{node_id}-coordinates-seed{seed}.npy"
            )
            atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
            coordinates_signature = expected_input_signature(coordinates_path)
            panel = score_panel(
                source,
                coordinates,
                config=cfg,
                centroids_by_k=centroids,
                hiD_reference=reference,
                reference_identity=reference_identity,
                ffr_group_labels=anchor_labels,
                scale_admission=None,
                provenance={
                    "round_id": ROUND_ID,
                    "seed": seed,
                    "capability": map_capability(seed),
                    "cell_id": rung_cell_id(seed),
                    "universe": f"R{RUNG_SOURCE_ROUND_ID}-minilm-mixed-{RUNG_SLUG}",
                    "substrate": dict(substrate_signature),
                    "train_receipt": dict(entry["receipt_signature"]),
                    "coordinates": coordinates_signature,
                    "shared_high_d_reference": reference_signature,
                },
            )
            if not panel_execution_ok(panel):
                raise Round0257Error(f"R0257 seed-{seed} panel is collapsed/nonfinite")
            purity = panel["purity"]
            raw_ratios = {
                "k256": float(purity["k256"]),
                "k1024": float(purity["k1024"]),
            }
            cells[seed] = {
                "seed": seed,
                "cell_id": rung_cell_id(seed),
                "capability": map_capability(seed),
                "is_a_gate_family_cell": False,
                "train_receipt": dict(entry["receipt_signature"]),
                "model": dict(entry["receipt"]["model"]),
                "coordinates": coordinates_signature,
                "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
                "panel": panel,
                "panel_metrics": panel_metric_view(panel),
                "raw_purity_ratios": raw_ratios,
                "corpus_ffr": {
                    slug: {
                        "anchors": int(
                            (panel["ffr_by_group"][slug] or {}).get("anchors", 0)
                        ),
                        "ffr": float((panel["ffr_by_group"][slug] or {}).get("ffr")),
                    }
                    for slug in CORPUS_SLUGS
                },
            }
            wrapped(f"R0257 seed-{seed} rung map scored")
            del model, coordinates
            torch.cuda.empty_cache()
            gc.collect()
        gate.finish(f"R0257 {RUNG_SLUG} panel stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    execution_checks = {
        "all_three_rung_maps_scored": set(cells) == set(SEEDS),
        "every_metric_finite": all(
            math.isfinite(float(value))
            for cell in cells.values()
            for value in (
                *cell["panel_metrics"].values(),
                *cell["raw_purity_ratios"].values(),
                *(slice_["ffr"] for slice_ in cell["corpus_ffr"].values()),
            )
        ),
        "per_corpus_ffr_slices_complete": all(
            set(cell["corpus_ffr"]) == set(CORPUS_SLUGS) for cell in cells.values()
        ),
        "code_slice_present": all(
            cell["corpus_ffr"][CORPUS_SLUGS[3]]["anchors"] > 0
            for cell in cells.values()
        ),
        "rung_maps_commensurate_one_invariant_digest": len(invariants) == 1,
        "three_distinct_checkpoints": len(model_hashes) == len(SEEDS),
        "shared_reference_reused_by_content_key": all(
            bool(cell["panel"]["provenance"]["hiD_reference_reused"])
            for cell in cells.values()
        ),
        "no_cell_here_is_a_gate_family_cell": all(
            cell["is_a_gate_family_cell"] is False for cell in cells.values()
        ),
    }
    if not all(execution_checks.values()):
        raise Round0257Error(f"R0257 panel execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0257Error(
            f"R0257 panel peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    node_wall = time.monotonic() - node_started
    receipt = {
        **_receipt_envelope(active["manifest"]),
        "schema": PANEL_SCHEMA,
        "capability": PANEL_CAPABILITY,
        "capabilities": [PANEL_CAPABILITY],
        "node": node_id,
        "rung": RUNG_SLUG,
        "rows": RUNG_ROWS,
        "cells": {str(seed): cells[seed] for seed in SEEDS},
        "cell_ids": list(rung_cell_ids()),
        "these_cells_define_nothing": (
            "The rung maps scored here are judged by the registered n = 29 MAD_n "
            "criteria fitted on the 2M universe. They are not pooled, not fitted, "
            "and never enter a gate family."
        ),
        "panel_configuration": {
            "source": "accepted R0113 panel_config()",
            "frac": cfg.frac,
            "k_hit": cfg.k_hit,
            "k_density": cfg.k_density,
            "k_frac_effective": int(reference["kf"]),
            "n_anchors": int(len(anchors)),
            "anchor_seed": cfg.anchor_seed,
            "formula_version": cfg.formula_version,
            "centroid_ks": list(CENTROID_KS),
            "centroid_recipe": (
                f"GPU Lloyd k-means over all {RUNG_ROWS} rows; seed "
                f"{CENTROID_SEED}; {CENTROID_ITERS} iterations (R0218's recipe)"
            ),
            "reference_is_built_on_this_rungs_own_rows": True,
            "why": (
                "the CRITERIA come from the 2M universe; the MEASUREMENT is made on "
                "the map's own population, exactly as every 2M cell's was made on "
                "its own. A high-D reference from another universe would not be a "
                "measurement of this map."
            ),
        },
        "shared_high_d_reference": reference_signature,
        "high_d_reference_key": str(reference["key"]),
        "high_d_reference_content_sha256": str(reference["content_sha256"]),
        "reference_convention": dict(REFERENCE_CONVENTION),
        "centroids": centroid_signatures,
        "anchor_corpus_counts": anchor_corpus_counts,
        "corpus_rows": dict(CORPUS_ROWS),
        "rung_invariant_sha256": sorted(invariants)[0],
        "lineage": {
            "substrate": dict(substrate_signature),
            "substrate_manifest": dict(sealed["manifest_signature"]),
            "ordered_substrate_sha256": sealed["ordered_substrate_sha256"],
        },
        "execution_checks": execution_checks,
        "guard_tail": tail,
        "enforcement_poll_spacing": scored,
        "gap_report": gaps,
        "abort_flag_precondition": abort_flag,
        "poll_coverage": _coverage(ledger, node_wall),
        "observed_span_s": float(ledger.observed_span_s()),
        "node_wall_s": node_wall,
        "performance": {
            "panel_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
        "training_performed": False,
        "evaluation_performed": True,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "production_or_publishing": False,
    }
    _seal(output, f"{node_id}-ladder-panel.json", receipt)
    del source, corpus_of_row, reference, centroids
    gc.collect()


# --------------------------------------------------------------------------- #
# node 5 — the judgement
# --------------------------------------------------------------------------- #


def _defining_family_from_the_sealed_gate(artifact: Mapping[str, Any]) -> list[str]:
    """The cells the n = 29 floors were fitted on, read out of the gate itself."""
    seeds = artifact.get("exact_family_seeds")
    if not isinstance(seeds, Sequence) or not seeds:
        raise Round0257JudgementError(
            "R0257 the sealed gate artifact does not name the cells it was fitted on"
        )
    return [f"exact-seed{int(seed)}" for seed in seeds]


def run_judge(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0257 round0257_nodes.run_judge")

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0257Error("R0257 judge handler received another queue")
    node_id = _node_id(active, f"judge_{RUNG_SLUG}")
    label = f"R0257 {RUNG_SLUG} rung judged by the n=29 gate"
    ledger = CoverageLedger(node=node_id)
    node_started = time.monotonic()
    abort_flag = _start_node(label)

    output = create_fresh_directory(str(job["outputs"][0]), label="R0257 rung verdict")

    window = ledger.window(f"R0257 {RUNG_SLUG} judgement stage")
    guard_ctx = _node_guard(label)
    gate_guard = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate_guard.start()
        recorder = PollRecorder(gate=gate_guard, clock=time.monotonic)
        recorder.anchor(f"R0257 {RUNG_SLUG} judgement stage entered")
        wrapped = window.wrap(recorder)

        gate_path = _bound_path(
            job, "gate_artifact_signature", label="R0256 sealed n=29 gate artifact"
        )
        gate_artifact = _read_json(gate_path, "R0256 sealed n=29 gate artifact")
        wrapped("R0257 sealed gate artifact read")
        gate = validate_gate_artifact(gate_artifact)
        wrapped("R0257 sealed gate artifact validated")

        defining = _defining_family_from_the_sealed_gate(gate_artifact)
        family_verdict = assert_no_rung_map_in_the_gate_family(defining)
        family_controls = rung_family_purity_controls(defining)
        if not family_controls["every_planted_defect_was_refused"]:
            raise Round0257FamilyError(
                "R0257 the shipped family guard did not refuse every planted rung "
                f"map: {family_controls['controls']}"
            )
        if not family_controls["the_honest_family_still_passes"]:
            raise Round0257FamilyError(
                "R0257 the shipped family guard rejects the honest family, so its "
                "refusals prove nothing"
            )
        wrapped("R0257 family purity guard and its controls ran")

        controls = judgement_controls(gate_artifact)
        if not controls["every_planted_defect_was_refused"]:
            raise Round0257JudgementError(
                f"R0257 judgement guard did not refuse every plant: {controls}"
            )
        if not controls["every_behavioural_control_held"]:
            raise Round0257JudgementError(
                f"R0257 behavioural controls did not hold: {controls}"
            )
        wrapped("R0257 judgement controls ran")

        panel_path, panel_signature = _intra_queue_signature(
            dict(job["panel_signature"]), label="R0257 sealed rung panel"
        )
        panel = prompt_contract.read_sealed(panel_path, label="R0257 sealed rung panel")
        if (
            str(panel.get("schema")) != PANEL_SCHEMA
            or str(panel.get("round_id")) != ROUND_ID
            or int(panel.get("rows", -1)) != RUNG_ROWS
        ):
            raise Round0257Error("R0257 sealed rung panel contract changed")
        cells = {
            str(cell["cell_id"]): {
                "panel_metrics": dict(cell["panel_metrics"]),
                "raw_purity_ratios": dict(cell["raw_purity_ratios"]),
            }
            for cell in dict(panel["cells"]).values()
        }
        if set(cells) != set(rung_cell_ids()):
            raise Round0257Error(
                f"R0257 judged set {sorted(cells)} is not the registered rung set"
            )
        wrapped("R0257 rung panel read")

        judgement = judge_population(cells=cells, gate=gate)
        wrapped("R0257 verdicts computed")

        descriptive = {
            cell_id: dict(payload["panel_metrics"]) for cell_id, payload in cells.items()
        }
        gate_guard.finish(f"R0257 {RUNG_SLUG} judgement stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored = _score_gate_without_raising(gate_guard, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)

    #: Re-read the sealed gate off disk AFTER the judgement, and compare the digest
    #: to the one bound into the job. Equality is the measurement that this node
    #: consumed the registered artifact and left it untouched -- the honest form of
    #: "no floor was written here", which as a literal would be an assertion
    #: `vacuouscheck` correctly refuses.
    gate_signature_after = expected_input_signature(gate_path)
    gate_signature_bound = dict(job["gate_artifact_signature"])

    execution_checks = {
        "the_gate_artifact_is_the_registered_n29_one": (
            int(gate["n"]) == REGISTERED_N
            and str(gate["registry_fingerprint"]) == REGISTERED_FINGERPRINT
        ),
        "the_sealed_gate_artifact_is_byte_identical_after_the_judgement": (
            str(gate_signature_after["sha256"])
            == str(gate_signature_bound["sha256"])
            and int(gate_signature_after["bytes"]) == int(gate_signature_bound["bytes"])
        ),
        "no_rung_map_is_in_the_fitting_family": bool(
            family_verdict["judged_and_defining_are_disjoint"]
        ),
        "every_planted_rung_map_was_refused": bool(
            family_controls["every_planted_defect_was_refused"]
        ),
        "every_judgement_plant_was_refused": bool(
            controls["every_planted_defect_was_refused"]
        ),
        "every_behavioural_control_held": bool(
            controls["every_behavioural_control_held"]
        ),
        "density_v2_contributes_to_no_verdict": all(
            item["descriptive_only"][metric]["contributes_to_verdict"] is False
            for item in judgement["verdicts"].values()
            for metric in DESCRIPTIVE_METRICS
        ),
        "every_verdict_carries_its_false_alarm_rate": all(
            float(item["panel_false_alarm_rate"]) == PANEL_FALSE_ALARM_RATE
            for item in judgement["verdicts"].values()
        ),
        "power_is_matched_to_sidedness": all(
            item["per_metric"][metric]["applicable_power"]
            == ("one_sided" if metric == "ffr" else "two_sided")
            for item in judgement["verdicts"].values()
            for metric in GATED_METRICS
        ),
    }
    if not all(execution_checks.values()):
        raise Round0257JudgementError(
            f"R0257 judgement execution checks failed: {execution_checks}"
        )

    node_wall = time.monotonic() - node_started
    receipt = {
        **_receipt_envelope(active["manifest"]),
        "schema": VERDICT_SCHEMA,
        "capability": VERDICT_CAPABILITY,
        "capabilities": [VERDICT_CAPABILITY],
        "node": node_id,
        "rung": RUNG_SLUG,
        "rows": RUNG_ROWS,
        "gate": gate,
        "gate_capability_consumed": GATE_CAPABILITY,
        "gate_artifact": gate_signature_after,
        "gate_artifact_as_bound": gate_signature_bound,
        "no_floor_is_written_by_this_round": (
            "DESIGN INTENT, stated as prose rather than as a check: this module "
            "has no code path that fits, refits or registers a floor. The "
            "corresponding MEASUREMENT is "
            "`the_sealed_gate_artifact_is_byte_identical_after_the_judgement`."
        ),
        "panel_artifact": panel_signature,
        "judgement": judgement,
        "descriptive_metrics_by_cell": descriptive,
        "family_verdict": family_verdict,
        "family_purity_controls": family_controls,
        "judgement_controls": controls,
        "power_materiality": POWER_MATERIALITY,
        "independence_limitation": INDEPENDENCE_LIMITATION,
        "execution_checks": execution_checks,
        "guard_tail": tail,
        "enforcement_poll_spacing": scored,
        "gap_report": gaps,
        "abort_flag_precondition": abort_flag,
        "poll_coverage": _coverage(ledger, node_wall),
        "observed_span_s": float(ledger.observed_span_s()),
        "node_wall_s": node_wall,
        "gpu_used": False,
        "training_performed": False,
        "evaluation_performed": True,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "production_or_publishing": False,
    }
    _seal(output, f"{node_id}-rung-verdict.json", receipt)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == TRAIN_ACTION:
        run_train(active, job)
    elif action == PANEL_ACTION:
        run_panel(active, job)
    elif action == JUDGE_ACTION:
        run_judge(active, job)
    else:
        raise Round0257Error(f"R0257 does not implement action {action!r}")


__all__ = [
    "CORPUS_ROWS",
    "CORPUS_SLUGS",
    "JUDGE_ACTION",
    "PANEL_ACTION",
    "SAFETY_NOTE",
    "TRAIN_ACTION",
    "TRANSFORM_CHUNK_ROWS",
    "run_job",
    "run_judge",
    "run_panel",
    "run_train",
]
