"""Execute R0230: five more cells (seeds 50-54), then pool thirteen on one panel.

Six nodes. Five GPU trains and one GPU panel-scoring node that depends on them.

**The trains** are R0221's, structurally unchanged: same sealed R0216
`queue-correction-3` substrate and exact k15 graph, same derived horizon, same
sampler, same residency, same accounting closure, same full-population finiteness
check, and a config built by overwriting only the nine paths R0217 registered as
seed-bearing. Two things are added, and both are safety rather than science:

1. a **predictive guard** that runs before the cell touches the GPU, records its
   prediction as data whether or not the cell runs, and refuses a cell whose
   predicted footprint breaches the `24 GiB` device or `60 GiB` host-anonymous
   budget (`refused_a_priori`);
2. a **watchdog** that samples the training process and trips on system swap
   **growth over a pre-launch baseline** — never on an absolute swap level, since
   this box holds swap at idle. A trip is a cooperative `interrupt_main()` so the
   interpreter unwinds and the driver tears the CUDA context down through its
   normal path. Nothing here ever sends `SIGKILL` — or `SIGTERM` — to a process
   holding a CUDA context. That is what wedged this machine once already.

**The panel node** does not rebuild the high-D reference. It loads R0218's
published `minilm-2m-high-d-reference.npz`, checks its file signature, content key
and `content_sha256` against the values R0218 sealed and R0222/R0223 re-verified,
re-derives the key from this substrate and these centroids, checks the anchor ids
and per-corpus counts, and requires every cell's `hi_D_agreement` numerators to be
literally R0218's `0.3828` / `0.2385`. If any of that disagrees the node raises
`REFERENCE_MISMATCH_MESSAGE` and the round stops: the thirteen cells would not be
poolable and that is the finding, not an obstacle.

The eight existing cells are **not rescored**. They are read from R0218's and
R0222's sealed receipts and R0223's sealed raw ratios, all bound by hash. This
round registers no floor; R0231 fits them, on a robust scale, at n = 13.
"""
from __future__ import annotations

import _thread
import gc
import json
import math
import os
import random
import resource
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0217_minilm_2m_pipeline import (
    MiniLMHostFp32EndpointArray,
    MiniLMMixedTrainingInput,
)
from basemap.round0217_minilm_2m_seed_family import (
    WARMUP_SUCCESSFUL_UPDATES,
)
from basemap.round0230_minilm_2m_seed_extension_n13 import (
    BATCH_SIZE,
    CAPABILITY_TEMPLATE,
    DEVICE_BUDGET_BYTES,
    DIMENSION,
    FULL_TRANSFORM_BATCH,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    HOST_ANON_BUDGET_BYTES,
    HOST_RSS_LIMIT_GIB,
    MEMORY_POLICY,
    POOLED_SEEDS,
    POSITIVE_ROWS_PER_UPDATE,
    PRODUCTION_CONFIG_SCHEMA,
    R0217_SEED_INVARIANT_SHA256,
    ROUND_ID,
    ROWS,
    Round0230Error,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    SWAP_GROWTH_ABORT_BYTES,
    TEMPLATE_SEED,
    TRAIN_SCHEMA,
    WATCHDOG_POLL_S,
    capability_for_seed,
    masked_config_bytes,
    performance_windows,
    predict_cell_footprint,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config,
    validate_full_population_map,
    validate_registered_dose,
)
from basemap.round0230_minilm_2m_panel_n13 import (
    CENTROID_KS,
    CORPUS_SLUGS,
    DENSITY_V2_STATUS,
    PANEL_CAPABILITY,
    PANEL_CAPABILITY_N13,
    PANEL_METRICS,
    PANEL_SCHEMA,
    PANEL_SCHEMA_N13,
    POOLED_CELL_SOURCES,
    REFERENCE_MISMATCH_MESSAGE,
    R0218_SEEDS,
    R0221_SEEDS,
    Round0230PanelError,
    assert_hi_d_agreement,
    assert_reference_identity,
    corpus_ffr_view,
    descriptive_family_summary,
    panel_execution_ok,
    panel_metric_view,
    pool_thirteen_cells,
    raw_purity_ratios,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0113_nodes as prompt_nodes
from experiments import round0218_nodes
from basemap.round0253_stop_hooks import install_stop_hooks


TRAIN_ACTION = "train_minilm_mixed_2m_seed_extension_n13"
PANEL_ACTION = "score_minilm_mixed_2m_panel_n13"


# --------------------------------------------------------------------------- #
# the memory guard
# --------------------------------------------------------------------------- #


def _proc_memory_bytes(pid: int) -> tuple[int, int]:
    """`(VmRSS, RssAnon)`. Anonymous bytes are the swappable ones."""
    rss = 0
    anon = 0
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("RssAnon:"):
                    anon = int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return rss, anon


def _swap_used_bytes() -> int:
    total = 0
    free = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("SwapFree:"):
                    free = int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return max(0, total - free)


class CellWatchdog(threading.Thread):
    """Sample this process and abort **cooperatively** if a budget is breached.

    Trips on system swap GROWTH over the pre-launch baseline, never on an
    absolute level — this box holds swap at idle, so an absolute threshold would
    refuse every cell before it started. It also trips on the host ANONYMOUS
    budget; file-backed pages are clean page cache, evicted rather than swapped
    (Review 0224), so they are measured and never charged.

    The abort is `_thread.interrupt_main()`: a `KeyboardInterrupt` raised inside
    the interpreter, which unwinds Python and lets the CUDA driver tear its
    context down through the normal path. No signal is sent to any process. A
    `SIGKILL` delivered to a process holding a CUDA context is what deadlocked
    RCU and forced a reboot on this machine, and this round will not repeat it.
    """

    def __init__(
        self,
        *,
        poll_s: float = WATCHDOG_POLL_S,
        host_anon_budget_bytes: int = HOST_ANON_BUDGET_BYTES,
        swap_growth_abort_bytes: int = SWAP_GROWTH_ABORT_BYTES,
    ) -> None:
        super().__init__(daemon=True)
        self._pid = os.getpid()
        self._poll = float(poll_s)
        self._anon_budget = int(host_anon_budget_bytes)
        self._swap_abort = int(swap_growth_abort_bytes)
        self._halt = threading.Event()
        self.swap_baseline_bytes = _swap_used_bytes()
        self.swap_peak_bytes = self.swap_baseline_bytes
        self.rss_peak_bytes = 0
        self.anon_peak_bytes = 0
        self.samples = 0
        self.tripped = False
        self.trip_reason: str | None = None

    def _trip(self, reason: str) -> None:
        if self.tripped:
            return
        self.tripped = True
        self.trip_reason = reason
        _thread.interrupt_main()

    def run(self) -> None:
        while not self._halt.is_set():
            self.samples += 1
            rss, anon = _proc_memory_bytes(self._pid)
            self.rss_peak_bytes = max(self.rss_peak_bytes, rss)
            self.anon_peak_bytes = max(self.anon_peak_bytes, anon)
            swap = _swap_used_bytes()
            self.swap_peak_bytes = max(self.swap_peak_bytes, swap)
            growth = swap - self.swap_baseline_bytes
            if growth > self._swap_abort:
                self._trip(
                    f"system swap grew {growth / 1024 ** 3:.2f} GiB over its "
                    f"{self.swap_baseline_bytes / 1024 ** 3:.2f} GiB pre-launch "
                    f"baseline, exceeding the "
                    f"{self._swap_abort / 1024 ** 3:.2f} GiB growth threshold"
                )
            elif anon > self._anon_budget:
                self._trip(
                    f"process anonymous memory {anon / 1024 ** 3:.2f} GiB exceeds "
                    f"the {self._anon_budget / 1024 ** 3:.2f} GiB budget"
                )
            self._halt.wait(self._poll)

    def stop(self) -> dict[str, Any]:
        self._halt.set()
        self.join(timeout=self._poll * 2 + 5.0)
        return {
            "poll_s": self._poll,
            "samples": self.samples,
            "swap_baseline_bytes": self.swap_baseline_bytes,
            "swap_peak_bytes": self.swap_peak_bytes,
            "swap_growth_bytes": self.swap_peak_bytes - self.swap_baseline_bytes,
            "swap_growth_abort_bytes": self._swap_abort,
            "peak_rss_bytes": self.rss_peak_bytes,
            "peak_anonymous_bytes": self.anon_peak_bytes,
            "host_anonymous_budget_bytes": self._anon_budget,
            "tripped": self.tripped,
            "trip_reason": self.trip_reason,
            "abort_mechanism": (
                "cooperative interrupt_main(); no signal is sent to any process "
                "and SIGKILL is never used against a CUDA context"
            ),
        }


# --------------------------------------------------------------------------- #
# train nodes
# --------------------------------------------------------------------------- #


def _seed(job: Mapping[str, Any]) -> int:
    seed = job.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0230Error(f"R0230 job seed {seed!r} is not a registered cell")
    if str(job.get("capability") or "") != capability_for_seed(seed):
        raise Round0230Error("R0230 job capability does not match its seed")
    return int(seed)


def _sealed_graph(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read the sealed R0216 receipt and load its exact k15 fuzzy graph."""
    manifest_signature = dict(job["graph_manifest_signature"])
    manifest_path = prompt_contract.verify_signature(
        manifest_signature, label="R0230 sealed R0216 substrate+graph receipt"
    )
    manifest = prompt_contract.read_sealed(
        manifest_path, label="R0230 sealed R0216 substrate+graph receipt"
    )
    checks = manifest.get("graph_checks") or {}
    degrees = manifest.get("degrees") or {}
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != GRAPH_SOURCE_ROUND_ID
        or manifest.get("capability") != GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or manifest.get("training_performed") is not False
    ):
        raise Round0230Error("R0230 sealed R0216 substrate+graph contract changed")
    if (
        int(checks.get("zero_degree_rows", -1)) != 0
        or int(degrees.get("zero_degree_rows", -1)) != 0
        or float(checks.get("mean_recall_at_k", 0.0))
        < float(checks.get("mean_recall_floor", 1.0))
        or float(checks.get("p10_recall_at_k", 0.0))
        < float(checks.get("p10_recall_floor", 1.0))
    ):
        raise Round0230Error(
            "R0230 requires the sealed R0216 graph to have passed its exactness "
            "and zero-degree checks"
        )
    edges = int(manifest.get("directed_edge_count", 0))
    if edges <= 0:
        edges = int(checks.get("directed_edges", 0))
    if edges != SEALED_DIRECTED_EDGES:
        raise Round0230Error(
            f"R0230 sealed graph reports {edges} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    graph_signature = dict(manifest["graph"])
    graph_path = prompt_contract.verify_signature(
        graph_signature, label="R0230 sealed R0216 fuzzy graph"
    )
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )

    sources, targets, weights, n_nodes = load_edge_arrays(graph_path, load_weights=True)
    if (
        weights is None
        or int(n_nodes) != ROWS
        or len(sources) != edges
        or targets.shape != sources.shape
        or weights.shape != sources.shape
        or sources.dtype != np.int32
        or targets.dtype != np.int32
        or weights.dtype != np.float32
    ):
        raise Round0230Error("R0230 sealed R0216 graph arrays changed")
    return {
        "manifest": manifest,
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": int(n_nodes),
        "directed_edges": edges,
    }


def _open_substrate(graph: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Serve the 3.07 GB sealed substrate lazily; never materialize it."""
    signature = dict(graph["manifest"]["substrate"])
    path = prompt_contract.verify_signature(
        signature, label="R0230 sealed R0216 substrate"
    )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (ROWS, DIMENSION) or array.dtype != np.float32:
        raise Round0230Error("R0230 sealed R0216 substrate geometry changed")
    return array, signature


def _weighted_rejection_accounting_mismatch(
    runtime: Mapping[str, Any], *, producer_delta: int, updates: int
) -> dict[str, Any] | None:
    expected_emitted = (updates + producer_delta) * POSITIVE_ROWS_PER_UPDATE
    if (
        int(runtime["weight_emitted_draws"]) != expected_emitted
        or int(runtime["weight_acceptances"])
        != int(runtime["weight_emitted_draws"]) + int(runtime["weight_buffered_draws"])
        or int(runtime["weight_proposals"]) < int(runtime["weight_acceptances"])
        or not 0 < float(runtime["weight_acceptance_rate"]) <= 1
    ):
        return {
            "expected_emitted_positive_draws": expected_emitted,
            "expected_consumed_positive_draws": updates * POSITIVE_ROWS_PER_UPDATE,
            "producer_delta": producer_delta,
            "runtime": runtime,
        }
    return None


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0253 round0230_nodes.run_train")
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0230Error("R0230 train handler received another queue")
    seed = _seed(job)

    # --- the predictive guard, before anything touches the GPU -------------- #
    prediction = predict_cell_footprint(seed)
    declared = job.get("memory_prediction")
    if declared is not None and dict(declared) != prediction:
        raise Round0230Error(
            "R0230 cell prediction differs from the one sealed at prepare time"
        )
    if prediction["refused_a_priori"]:
        raise Round0230Error(
            f"R0230 seed-{seed} refused a priori: predicted "
            f"{prediction['predicted_peak_device_bytes']} device bytes and "
            f"{prediction['predicted_peak_host_anonymous_bytes']} host anonymous "
            f"bytes against budgets {DEVICE_BUDGET_BYTES} / "
            f"{HOST_ANON_BUDGET_BYTES}"
        )

    graph = _sealed_graph(job)
    edges = graph["directed_edges"]
    updates = successful_updates_for_edges(edges)
    dose = validate_registered_dose(updates=updates, edge_count=edges)
    declared_bound = job.get("registered_dose_bound")
    if declared_bound is not None and updates > int(declared_bound):
        raise Round0230Error(
            "R0230 derived update horizon exceeds the registered round bound"
        )
    source, substrate_signature = _open_substrate(graph)
    config, config_sha = train_config(
        seed=seed,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        substrate_signature=substrate_signature,
        graph_edges=edges,
        rows=ROWS,
    )
    declared_invariant = str(job.get("family_seed_invariant_sha256") or "")
    observed_invariant = seed_invariant_sha256(config)
    masked = masked_config_bytes(config)
    if (
        not declared_invariant
        or observed_invariant != declared_invariant
        or observed_invariant != R0217_SEED_INVARIANT_SHA256
    ):
        raise Round0230Error(
            "R0230 cell config is not R0217's treatment outside the seed: "
            f"{observed_invariant} != {declared_invariant} / "
            f"{R0217_SEED_INVARIANT_SHA256}"
        )
    if int(config["optimizer"]["successful_positive_lr_updates"]) != updates:
        raise Round0230Error("R0230 horizon did not reach the train config")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0230 train output")
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": PRODUCTION_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "treatment_config_round_id": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "seed": seed,
            "capability": capability_for_seed(seed),
            "seed_invariant_sha256": observed_invariant,
            "masked_config_bytes": len(masked),
            "config": config,
            "config_sha256": config_sha,
        },
        immutable=True,
    )

    dataset = MiniLMHostFp32EndpointArray(
        source,
        source_signature=substrate_signature,
        buffer_rows=BATCH_SIZE,
    )
    wrapper = MiniLMMixedTrainingInput(dataset, graph, seed=seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = prompt_nodes._new_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = performance_windows(updates)
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")

    watchdog = CellWatchdog()
    watchdog.start()
    started = time.monotonic()
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
    except KeyboardInterrupt as error:
        watchdog_state = watchdog.stop()
        raise Round0230Error(
            f"R0230 seed-{seed} aborted by the memory watchdog: "
            f"{watchdog_state.get('trip_reason')!r}"
        ) from error
    wall = time.monotonic() - started

    accounting = dict(model._train_stats)
    runtime = wrapper.runtime_stamp()
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
        watchdog.stop()
        raise Round0230Error(f"R0230 train accounting failed: {mismatches}")
    prompt_contract.synchronize_runtime_counters(accounting, runtime)
    accounting["pipeline_runtime"] = dict(runtime)

    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (updates - WARMUP_SUCCESSFUL_UPDATES) / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    if profiler.get("aborted") is not False or rate < config["execution"][
        "minimum_train_upd_s"
    ]:
        watchdog.stop()
        raise Round0230Error("R0230 train performance admission failed")

    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
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

    # The published checkpoint must reload from disk and project every row.
    from basemap.pumap.parametric_umap import ParametricUMAP

    reloaded = ParametricUMAP.load(model_path, device="cuda")
    coordinates = np.asarray(
        reloaded.transform(source, batch_size=FULL_TRANSFORM_BATCH), dtype=np.float32
    )
    published = validate_full_population_map(coordinates)
    published["model"] = expected_input_signature(model_path)
    del reloaded, coordinates
    torch.cuda.empty_cache()
    gc.collect()

    watchdog_state = watchdog.stop()
    if watchdog_state["tripped"]:
        raise Round0230Error(
            f"R0230 seed-{seed} watchdog tripped: {watchdog_state['trip_reason']!r}"
        )
    if int(memory["peak_reserved_bytes"]) > DEVICE_BUDGET_BYTES:
        raise Round0230Error(
            f"R0230 seed-{seed} peak reserved device bytes "
            f"{memory['peak_reserved_bytes']} exceed the {DEVICE_BUDGET_BYTES} "
            "budget"
        )
    if int(watchdog_state["peak_anonymous_bytes"]) > HOST_ANON_BUDGET_BYTES:
        raise Round0230Error(
            f"R0230 seed-{seed} peak anonymous bytes "
            f"{watchdog_state['peak_anonymous_bytes']} exceed the "
            f"{HOST_ANON_BUDGET_BYTES} budget"
        )
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0230Error(
            f"R0230 train peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    memory["peak_host_rss_gib"] = peak_rss_gib
    memory["peak_host_anonymous_bytes"] = int(watchdog_state["peak_anonymous_bytes"])

    receipt = prompt_contract.seal({
        "schema": TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "treatment_config_round_id": "0217",
        "treatment_template_seed": TEMPLATE_SEED,
        "pooled_seed_family": list(POOLED_SEEDS),
        "capability": capability_for_seed(seed),
        "capabilities": [capability_for_seed(seed)],
        "training_seed": seed,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "seed_invariant_sha256": observed_invariant,
        "masked_config_bytes": len(masked),
        "r0217_published_seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "model": published["model"],
        "substrate": substrate_signature,
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "graph_capability": GRAPH_CAPABILITY,
        "rows": ROWS,
        "dimension": DIMENSION,
        "directed_edges": edges,
        "dose_registration": dose,
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "requested_positive_draws_per_edge": float(
            config["execution"]["target_positive_draws_per_edge"]
        ),
        "consumed_positive_draws": int(updates * POSITIVE_ROWS_PER_UPDATE),
        "consumed_positive_draws_per_edge": float(
            updates * POSITIVE_ROWS_PER_UPDATE / edges
        ),
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_s": wall,
        "published_map_check": published,
        "memory_prediction": prediction,
        "memory_watchdog": watchdog_state,
        "memory_policy": MEMORY_POLICY,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "endpoint_rows_match_updates": True,
            "weighted_rejection_accounting_closes": True,
            "dose_derived_from_sealed_edge_count": True,
            "dose_landed_on_registered_ceil_value": True,
            "treatment_identical_to_r0217_except_seed": True,
            "reconstructs_r0217_template_byte_for_byte": True,
            "published_checkpoint_reloads_finite_and_uncollapsed": True,
            "all_2m_coordinates_finite": True,
            "predicted_before_launch": True,
            "not_refused_a_priori": True,
            "watchdog_did_not_trip": True,
        },
        "memory": memory,
        "training_performed": True,
        "optimizer_updates": updates,
        "map_decision_made": False,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    })
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), receipt, immutable=True
    )
    del source, graph
    gc.collect()


# --------------------------------------------------------------------------- #
# the n = 13 panel node
# --------------------------------------------------------------------------- #


def _intra_queue_signature(
    reference: Mapping[str, Any], *, label: str
) -> tuple[str, dict[str, Any]]:
    """Resolve a reference to an artifact produced earlier in THIS queue.

    R0228's geometry node died on `verify_signature` of an intra-queue reference
    that carries a path and no hash at prepare time, because the bytes did not
    exist when the queue was written. This is R0229's fix, reused verbatim.
    """
    reference = dict(reference)
    if reference.get("sha256"):
        return prompt_contract.verify_signature(reference, label=label), reference
    path = str(reference["canonical_path"])
    if not os.path.exists(path):
        raise Round0230PanelError(f"{label} is absent at {path}")
    return path, expected_input_signature(path)


def _read_json(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise Round0230PanelError(f"R0230 {label} is absent at {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sealed_panel(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind R0218's sealed four-cell panel: the reference and four of the cells."""
    panel_path = str(job["panel_evidence"])
    signature = expected_input_signature(panel_path)
    panel = prompt_contract.read_sealed(
        panel_path, label="R0218 MiniLM 2M four-seed panel"
    )
    checks = panel.get("execution_checks") or {}
    if (
        panel.get("schema") != PANEL_SCHEMA
        or panel.get("round_id") != "0218"
        or panel.get("capabilities") != [PANEL_CAPABILITY]
        or panel.get("seeds") != list(R0218_SEEDS)
        or int(panel.get("n", -1)) != len(R0218_SEEDS)
        or panel.get("evaluation_performed") is not True
        or panel.get("gate_registered") is not False
        or panel.get("metrics") != list(PANEL_METRICS)
        or panel.get("seed_invariant_sha256") != R0217_SEED_INVARIANT_SHA256
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise Round0230PanelError("R0218 panel receipt contract changed")
    return panel, signature


def _load_centroids(panel: Mapping[str, Any]):
    """R0218's published purity vocabularies, loaded rather than recomputed."""
    declared = dict(panel.get("centroids") or {})
    if set(declared) != {str(k) for k in CENTROID_KS}:
        raise Round0230PanelError("R0218 centroid vocabularies changed")
    centroids: dict[int, np.ndarray] = {}
    signatures: dict[str, Any] = {}
    for k in CENTROID_KS:
        signature = dict(declared[str(k)])
        path = prompt_contract.verify_signature(
            signature, label=f"R0218 purity centroids k{k}"
        )
        array = np.load(path, allow_pickle=False)
        if array.shape != (k, DIMENSION) or array.dtype != np.dtype("float32"):
            raise Round0230PanelError(f"R0218 centroids k{k} geometry changed")
        centroids[k] = array
        signatures[str(k)] = signature
    return centroids, signatures


def _authenticate_r0230_map(
    cell: Mapping[str, Any], sealed: Mapping[str, Any]
) -> tuple[int, dict[str, Any], dict[str, Any], str]:
    """Bind one R0230 map to the exact substrate R0218's panel scored."""
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0230PanelError(f"R0230 cell seed {seed!r} is not an R0230 cell")
    capability = capability_for_seed(seed)
    if str(cell.get("capability") or "") != capability:
        raise Round0230PanelError(f"R0230 seed-{seed} cell capability changed")
    receipt_path, receipt_signature = _intra_queue_signature(
        cell["train_receipt"], label=f"R0230 seed-{seed} train receipt"
    )
    receipt = prompt_contract.read_sealed(
        receipt_path, label=f"R0230 seed-{seed} train receipt"
    )
    train_checks = receipt.get("train_checks") or {}
    if (
        receipt.get("schema") != TRAIN_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("treatment_config_round_id") != "0217"
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != seed
        or receipt.get("training_performed") is not True
        or receipt.get("gate_registerable_here") is not False
        or receipt.get("map_decision_made") is not False
        or int(receipt.get("rows", -1)) != ROWS
        or int(receipt.get("dimension", -1)) != DIMENSION
        or int(receipt.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
        or receipt.get("graph_capability") != GRAPH_CAPABILITY
        or str(receipt.get("seed_invariant_sha256") or "")
        != R0217_SEED_INVARIANT_SHA256
        or not train_checks
        or not all(bool(value) for value in train_checks.values())
    ):
        raise Round0230PanelError(f"R0230 seed-{seed} train receipt contract changed")
    if (
        dict(receipt.get("substrate") or {}) != dict(sealed["substrate_signature"])
        or dict(receipt.get("graph_manifest") or {})
        != dict(sealed["manifest_signature"])
    ):
        raise Round0230PanelError(
            f"R0230 seed-{seed} was not trained on the substrate this panel scores"
        )
    model_path = prompt_contract.verify_signature(
        receipt["model"], label=f"R0230 seed-{seed} published map"
    )
    return seed, receipt, receipt_signature, model_path


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0253 round0230_nodes.run_panel")
    import torch
    from basemap.panel_v2 import reset_process_cuda_peak, sample_anchors, score_panel
    from basemap.pumap.parametric_umap import ParametricUMAP

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0230PanelError("R0230 panel handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0230PanelError("R0230 panel scoring requires CUDA")

    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    corpus_of_row = round0218_nodes._corpus_of_row(sealed)
    panel_evidence, panel_signature = _sealed_panel(job)

    r0222_path, r0222_signature = _intra_queue_signature(
        job["r0222_gate_signature"], label="R0222 sealed n=8 gate"
    )
    r0222 = prompt_contract.read_sealed(r0222_path, label="R0222 sealed n=8 gate")
    r0223_path, r0223_signature = _intra_queue_signature(
        job["r0223_comparison_signature"], label="R0223 sealed cuVS comparison"
    )
    r0223 = prompt_contract.read_sealed(
        r0223_path, label="R0223 sealed cuVS comparison"
    )

    prior_cells = dict(r0222["pooled_panel_metric_cells"])
    prior_corpus = dict(r0222["pooled_corpus_ffr_cells"])
    prior_ratios = dict(r0223["exact_family_purity_ratios"])
    prior_seeds = {str(seed) for seed in R0218_SEEDS} | {
        str(seed) for seed in R0221_SEEDS
    }
    if (
        set(prior_cells) != prior_seeds
        or set(prior_corpus) != prior_seeds
        or set(prior_ratios) != prior_seeds
    ):
        raise Round0230PanelError(
            "R0230 requires exactly the eight already-scored cells 42-49 from "
            "R0222's and R0223's sealed artifacts"
        )
    for seed in R0218_SEEDS:
        mine = {
            key: float(value)
            for key, value in panel_evidence["panel_metric_cells"][str(seed)].items()
        }
        theirs = {key: float(value) for key, value in prior_cells[str(seed)].items()}
        if mine != theirs:
            raise Round0230PanelError(
                f"R0222's pooled cell for seed {seed} disagrees with R0218's own "
                "sealed panel; the eight prior cells are not one table"
            )

    cells_in = job.get("cells")
    if not isinstance(cells_in, list) or {
        int(cell.get("seed", -1)) for cell in cells_in
    } != set(SEEDS):
        raise Round0230PanelError("R0230 five-cell input matrix changed")
    authenticated: dict[int, dict[str, Any]] = {}
    for cell in cells_in:
        seed, receipt, receipt_signature, model_path = _authenticate_r0230_map(
            cell, sealed
        )
        authenticated[seed] = {
            "receipt": receipt,
            "receipt_signature": receipt_signature,
            "model_path": model_path,
        }
    invariants = {
        str(authenticated[seed]["receipt"]["seed_invariant_sha256"]) for seed in SEEDS
    } | {R0217_SEED_INVARIANT_SHA256}
    if len(invariants) != 1:
        raise Round0230PanelError(
            "R0230 pooled family is not commensurate: the new cells do not carry "
            "R0217's seed-invariant config digest"
        )
    model_hashes = {
        str(authenticated[seed]["receipt"]["model"]["sha256"]) for seed in SEEDS
    }
    model_hashes |= {
        str(panel_evidence["cells"][str(seed)]["model"]["sha256"])
        for seed in R0218_SEEDS
    }
    model_hashes |= {
        str(r0222["new_cells"][str(seed)]["model"]["sha256"]) for seed in R0221_SEEDS
    }
    if len(model_hashes) != len(POOLED_SEEDS):
        raise Round0230PanelError(
            "R0230 pooled family contains a duplicated checkpoint"
        )

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0230 n=13 panel"
    )
    started = time.monotonic()
    reset_process_cuda_peak()

    cfg = prompt_contract.panel_config()
    centroids, centroid_signatures = _load_centroids(panel_evidence)
    data_identity = {
        "kind": "ordered_array",
        "shape": [ROWS, DIMENSION],
        "dtype": np.dtype("<f4").str,
        "sha256": sealed["ordered_substrate_sha256"],
    }
    reference_identity = {
        "data_identity": data_identity,
        "convention": dict(round0218_nodes.REFERENCE_CONVENTION),
    }
    from basemap.panel_v2 import hiD_reference_key, load_hiD_reference

    reference_signature = dict(panel_evidence["shared_high_d_reference"])
    reference_path = prompt_contract.verify_signature(
        reference_signature, label="R0218 shared high-D reference"
    )
    observed_signature = expected_input_signature(reference_path)
    if observed_signature != reference_signature:
        raise Round0230PanelError(f"{REFERENCE_MISMATCH_MESSAGE} file signature drift")
    reference = load_hiD_reference(reference_path)
    anchors = sample_anchors(ROWS, cfg)
    if not np.array_equal(
        np.asarray(anchors, dtype=np.int64),
        np.asarray(reference["anchor_ids"], dtype=np.int64),
    ):
        raise Round0230PanelError(f"{REFERENCE_MISMATCH_MESSAGE} anchor drift")
    rederived_key, _parts = hiD_reference_key(
        source, anchors, cfg, centroids, kf=int(reference["kf"]), **reference_identity
    )
    anchor_labels = round0218_nodes._anchor_corpus_labels(corpus_of_row, anchors)
    anchor_corpus_counts = {
        slug: int((anchor_labels == slug).sum()) for slug in CORPUS_SLUGS
    }
    reference_identity_receipt = assert_reference_identity(
        file_signature=observed_signature,
        key=str(reference["key"]),
        content_sha256=str(reference["content_sha256"]),
        rederived_key=str(rederived_key),
        anchor_corpus_counts=anchor_corpus_counts,
    )
    if (
        str(reference["key"]) != str(panel_evidence["high_d_reference_key"])
        or str(reference["content_sha256"])
        != str(panel_evidence["high_d_reference_content_sha256"])
        or str(reference["key"]) != str(r0222["high_d_reference_key"])
        or str(reference["key"]) != str(r0223["high_d_reference_key"])
    ):
        raise Round0230PanelError(
            f"{REFERENCE_MISMATCH_MESSAGE} the reference R0218, R0222 and R0223 "
            "used is not the one loaded here"
        )

    cells: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        entry = authenticated[seed]
        model = ParametricUMAP.load(entry["model_path"], device="cuda")
        coordinates = np.asarray(
            model.transform(source, batch_size=8192), dtype=np.float32
        )
        if coordinates.shape != (ROWS, 2):
            raise Round0230PanelError(
                f"R0230 seed-{seed} transform produced {coordinates.shape}, "
                f"expected ({ROWS}, 2)"
            )
        if not np.isfinite(coordinates).all():
            raise Round0230PanelError(
                f"R0230 seed-{seed} transform over {ROWS} rows is not finite"
            )
        coordinates_path = os.path.join(output, f"coordinates-seed{seed}.npy")
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
                "capability": capability_for_seed(seed),
                "universe": "R0216-queue-correction-3-minilm-mixed-2m",
                "substrate": dict(sealed["substrate_signature"]),
                "provenance_array": dict(sealed["provenance_signature"]),
                "train_receipt": dict(entry["receipt_signature"]),
                "coordinates": coordinates_signature,
                "shared_high_d_reference": reference_signature,
                "reference_source_round": "0218",
            },
        )
        if not panel_execution_ok(panel):
            raise Round0230PanelError(
                f"R0230 seed-{seed} panel is collapsed or nonfinite"
            )
        if not bool(panel["provenance"]["hiD_reference_reused"]):
            raise Round0230PanelError(
                f"{REFERENCE_MISMATCH_MESSAGE} seed-{seed} recomputed"
            )
        agreement = assert_hi_d_agreement(seed, panel["purity_numerators"])
        cells[seed] = {
            "seed": seed,
            "capability": capability_for_seed(seed),
            "train_receipt": dict(entry["receipt_signature"]),
            "model": dict(entry["receipt"]["model"]),
            "coordinates": coordinates_signature,
            "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
            "panel": panel,
            "panel_metrics": panel_metric_view(panel),
            "raw_purity_ratios": raw_purity_ratios(panel),
            "hi_d_agreement": agreement,
            "corpus_ffr": corpus_ffr_view(panel),
            "panel_finite_noncollapsed": True,
            "transform_rows_finite": ROWS,
        }
        del model, coordinates
        torch.cuda.empty_cache()
        gc.collect()

    pooled_cells: dict[str, dict[str, float]] = {}
    pooled_ratios: dict[str, dict[str, float]] = {}
    pooled_corpus: dict[str, dict[str, dict[str, float]]] = {}
    for seed in list(R0218_SEEDS) + list(R0221_SEEDS):
        key = str(seed)
        pooled_cells[key] = {
            name: float(value) for name, value in prior_cells[key].items()
        }
        pooled_ratios[key] = {
            name: float(value) for name, value in prior_ratios[key].items()
        }
        pooled_corpus[key] = {
            slug: {
                "anchors": int(prior_corpus[key][slug]["anchors"]),
                "ffr": float(prior_corpus[key][slug]["ffr"]),
            }
            for slug in CORPUS_SLUGS
        }
    for seed in SEEDS:
        key = str(seed)
        pooled_cells[key] = dict(cells[seed]["panel_metrics"])
        pooled_ratios[key] = dict(cells[seed]["raw_purity_ratios"])
        pooled_corpus[key] = dict(cells[seed]["corpus_ffr"])

    pooled = pool_thirteen_cells(
        cells=pooled_cells,
        ratios=pooled_ratios,
        corpus=pooled_corpus,
        sources=POOLED_CELL_SOURCES,
    )
    summary = descriptive_family_summary({
        metric: [pooled_cells[str(seed)][metric] for seed in POOLED_SEEDS]
        for metric in PANEL_METRICS
    })

    execution_checks = {
        "all_five_new_cells_scored": set(cells) == set(SEEDS),
        "thirteen_pooled_cells": len(pooled_cells) == len(POOLED_SEEDS),
        "every_metric_finite": all(
            math.isfinite(float(value))
            for cell in pooled_cells.values()
            for value in cell.values()
        ),
        "every_raw_ratio_positive_and_finite": all(
            math.isfinite(float(value)) and float(value) > 0.0
            for cell in pooled_ratios.values()
            for value in cell.values()
        ),
        "no_collapsed_panel": all(
            bool(cell["panel_finite_noncollapsed"]) for cell in cells.values()
        ),
        "map_transform_finite_over_all_rows": all(
            int(cell["transform_rows_finite"]) == ROWS for cell in cells.values()
        ),
        "per_corpus_ffr_slices_complete": all(
            set(pooled_corpus[str(seed)]) == set(CORPUS_SLUGS)
            for seed in POOLED_SEEDS
        ),
        "pooled_family_one_seed_invariant_digest": len(invariants) == 1,
        "thirteen_distinct_checkpoints": len(model_hashes) == len(POOLED_SEEDS),
        "reference_byte_identical_to_r0218": bool(
            reference_identity_receipt["reference_byte_identical_to_r0218"]
        ),
        "hi_d_agreement_identical_in_every_new_cell": all(
            cell["hi_d_agreement"] == {"k256": 0.3828, "k1024": 0.2385}
            for cell in cells.values()
        ),
        "shared_reference_reused_by_content_key": all(
            bool(cell["panel"]["provenance"]["hiD_reference_reused"])
            for cell in cells.values()
        ),
        "prior_eight_cells_not_rescored": True,
        "no_floor_registered_here": not pooled["gate_registerable_here"],
    }
    if not all(execution_checks.values()):
        raise Round0230PanelError(f"R0230 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0230PanelError(
            f"R0230 panel peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )

    receipt = prompt_contract.seal({
        **pooled,
        "schema": PANEL_SCHEMA_N13,
        "round_id": ROUND_ID,
        "capability": PANEL_CAPABILITY_N13,
        "capabilities": [PANEL_CAPABILITY_N13],
        "release_sha": active["manifest"]["release_sha"],
        "panel_evidence": panel_signature,
        "panel_capability": PANEL_CAPABILITY,
        "panel_release_sha": panel_evidence.get("release_sha"),
        "r0222_gate_artifact": r0222_signature,
        "r0223_comparison_artifact": r0223_signature,
        "lineage": {
            "graph_manifest": dict(sealed["manifest_signature"]),
            "substrate": dict(sealed["substrate_signature"]),
            "provenance": dict(sealed["provenance_signature"]),
            "ordered_substrate_sha256": sealed["ordered_substrate_sha256"],
        },
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
            "centroid_source": "R0218 published frozen centroid arrays, loaded",
        },
        "shared_high_d_reference": reference_signature,
        "high_d_reference_key": str(reference["key"]),
        "high_d_reference_content_sha256": str(reference["content_sha256"]),
        "reference_convention": dict(round0218_nodes.REFERENCE_CONVENTION),
        "reference_identity": reference_identity_receipt,
        "centroids": centroid_signatures,
        "anchor_corpus_counts": anchor_corpus_counts,
        "new_cells": {str(seed): cells[seed] for seed in SEEDS},
        "descriptive_family_summary": summary,
        "density_v2_status": DENSITY_V2_STATUS,
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gate_registered": False,
        "gate_registerable_here": False,
        "map_decision_made": False,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "outcome": "minilm-mixed-2m-seed-family-pooled-at-n13-on-r0218s-frozen-panel",
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "seed-family-panel-n13.json"), receipt, immutable=True
    )
    print(json.dumps({
        "capability": PANEL_CAPABILITY_N13,
        "n": len(POOLED_SEEDS),
        "reference_byte_identical_to_r0218": True,
    }))
    del source, corpus_of_row, reference, centroids
    gc.collect()


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == TRAIN_ACTION:
        run_train(active, job)
        return
    if action == PANEL_ACTION:
        run_panel(active, job)
        return
    raise Round0230Error(f"R0230 unknown action {action!r}")


__all__ = [
    "CellWatchdog",
    "PANEL_ACTION",
    "TRAIN_ACTION",
    "run_job",
    "run_panel",
    "run_train",
]
