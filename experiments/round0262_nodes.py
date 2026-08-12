"""Execute R0262 — the host-int8 X ↔ weighted fuzzy sampler adapter.

review-0259-01 §D corrected R0259's headline and left exactly one blocker
standing for the 100M rung:

    "no host-resident-X path is wired to the weighted fuzzy edge sampler"

Two nodes.

`quantise0262` (CPU-bound, holds the lease)
    Re-proves numpy's pairwise split rule **on this entry** (R0259 recorded the
    pin and never ran the check in its 100M node), then streams the R0238 fp32
    substrate to an int8 + exact-fp16-scale artifact in polled chunks, and
    accumulates the quantisation error over **every one of the 100,000,000
    rows** rather than a sample. Also runs the backing and accounting controls:
    a planted `ascontiguousarray`-of-a-memmap that the isinstance reader rule
    calls host-resident, and a `MAP_SHARED` allocation that no anonymous guard
    sees.

`wired0262` (GPU)
    Opens R0243's real 100M graph through the shipped `load_edge_arrays`, loads
    the int8 substrate into anonymous host RAM, and builds the loader through
    the **shipped** `ParametricUMAP._prepare_edge_list_training`. It asserts the
    selected pipeline is `host_int8_hybrid` with `weighted_effective=True`, then
    measures the per-update interval over a registered bound of updates at the
    registered batch shape — a *full* update of `2 x batch_size` feature rows,
    not half of one. It also measures the training signal three ways (same
    input recomputed / int8 of the same rows / an independent minibatch) so the
    int8 perturbation is reported against the noise the recipe already runs on.

**No fit runs and none is claimed.** The probe is bounded by
`PROBE_UPDATES` and refuses an unbounded run.

Safety: every bulk input is opened read-only; no process is signalled; no child
process is started; cuVS is handed nothing. The host guard gates on ANONYMOUS
bytes, never RSS.
"""
from __future__ import annotations

import gc
import mmap
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.output_safety import create_fresh_directory
from basemap.round0238_rung5 import json_safe
from basemap.round0242_locality import host_anonymous_bytes, io_counters, json_scrub
from basemap.round0245_guard import EnforcedHostWatchdog
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report
from basemap.round0253_coverage import CoverageLedger
from basemap.round0253_stop_hooks import (
    NOT_A_FAMILY_CELL,
    THE_INSTRUMENT_IS_DEFEATABLE,
    install_stop_hooks,
    over_the_ceiling,
    registered_ceiling_s,
)
from basemap.round0254_writeback import interval_summary
from basemap.round0259_hundred_m import (
    RUNG_100M,
    SUBSTRATE_100M_PATH,
    SUBSTRATE_100M_SHAPE,
    assert_pairwise_rule,
    assert_rung,
    assert_substrate_dimension,
)
from basemap.round0262_host_int8_adapter import (
    INT8_100M_PATH,
    SCALES_100M_PATH,
    Round0262BackingError,
    Round0262WiringError,
    WIRING,
    assert_buffer_admits_batch,
    build_host_int8_source,
    classify_host_backing,
    dequantise_block,
    host_memory_accounting,
    quantise_block_int8,
    quantise_substrate_to_int8,
)
from experiments.round0251_nodes import (
    _guard_tail_reported,
    _node_gate,
    _receipt_envelope as _r0251_envelope,
    _score_gate_without_raising,
    _seal,
    _start_node,
)


ROUND_ID = "0262"

QUANTISE_ACTION = "round0262_quantise_and_fidelity"
WIRED_ACTION = "round0262_wired_host_int8_interval"

QUANTISE_CAPABILITY = "round0262-hundred-m-int8-substrate-and-fidelity-v1"
WIRED_CAPABILITY = "round0262-host-int8-weighted-sampler-wiring-and-intervals-v1"

QUANTISE_SCHEMA = QUANTISE_CAPABILITY
WIRED_SCHEMA = WIRED_CAPABILITY

#: Anonymous budget for the wired node, as arithmetic. The host-int8 X is
#: `100,000,000 x 384 = 38,400,000,000` B plus `200,000,000` B of fp16 scales;
#: `HostStreamEdgeSampler`'s float64 fuzzy CDF over `2,511,103,254` edges is
#: `20,088,826,032` B. That is `58,688,826,032` B = `54.66` GiB.
#:
#: The two int32 endpoint arrays are NOT in this figure and that is a measured
#: fact, not an omission: `HostStreamEdgeSampler` builds them with
#: `np.ascontiguousarray(np.asarray(sources), dtype=np.int32)`, which copies
#: NOTHING when `sources` is already a C-contiguous int32 memmap. They stay
#: file-backed page cache. `classify_host_backing` seals the base chain that
#: proves it. If that no-op ever became a real copy the budget would need
#: another `20,088,826,032` B and would exceed the registry's 60 GiB ceiling.
NODE_ANON_BUDGET_BYTES = 60 * (1 << 30)

#: Refuse to start the wired node without this much MemAvailable: the 54.66 GiB
#: of anonymous peak plus the ~28 GiB of page cache the graph members occupy
#: plus headroom for the concurrently-running round.
MIN_MEM_AVAILABLE_BYTES = 88 * (1 << 30)

#: Refuse to start the quantise node without this much: it holds one
#: `rows_per_chunk` block plus its float32 fidelity temporaries.
MIN_MEM_AVAILABLE_QUANTISE_BYTES = 24 * (1 << 30)

#: The registered batch shape. `round0113_prompt_contrast.py:74-76` ->
#: `batch_size = 8192`, `pos_ratio = 0.05`, so `int(8192 * 0.05) = 409`
#: positives and `8192 - 409 = 7783` negatives per update. A fit gathers
#: `2 x batch_size = 16,384` feature rows per update
#: (`edge_list_dataset.py:776-777`), which is what this node times.
BATCH_SIZE = 8192
POS_RATIO = 0.05
POSITIVE_ROWS_PER_UPDATE = int(BATCH_SIZE * POS_RATIO)
FEATURE_ROWS_PER_UPDATE = 2 * BATCH_SIZE

#: The registered horizon (review-0259-01 §C.2):
#: `ceil(1_000_000 x 2_511_103_254 / 603_086_368)`.
LR_HORIZON_100M = 4_163_754

#: The plan's per-update budget at ~110 upd/s: `1 / 110 = 9.09` ms.
UPDATE_BUDGET_S = 1.0 / 110.0

#: Bounded probe. This round is not a fit and must not become one by omission.
PROBE_UPDATES = 400
PROBE_WARMUP_UPDATES = 20
MAX_PROBE_UPDATES = 2000

#: Gradient-fidelity arms.
FIDELITY_BATCHES = 24
FIDELITY_WARMUP_STEPS = 600
FIDELITY_SAMPLE_BLOCKS = 12
FIDELITY_ROWS_PER_BLOCK = 50_000

#: Chunking for the quantiser. 2M rows is 3.07 GB of fp32 in flight.
QUANTISE_ROWS_PER_CHUNK = 2_000_000

ROWS_100M = 100_000_000
DIMENSION = 384
DIRECTED_EDGES_100M = 2_511_103_254

R0243_FUZZY_DIR = (
    "/data/latent-basemap/runs/round-0243/queue/artifacts/"
    "minilm-mixed-100000k-cluster-spill-k15-fuzzy-graph-v1"
)
R0243_FUZZY_MANIFEST = os.path.join(R0243_FUZZY_DIR, "fuzzy-graph.json")

SAFETY_NOTE = (
    "R0262 allocates 38,600,000,000 B of anonymous host int8 X and a "
    "20,088,826,032 B float64 CDF on a 119 GiB host that is running another "
    "round concurrently. Every allocation is preceded by a MemAvailable check "
    "that refuses rather than swaps. No process is signalled, no child process "
    "is started, cuVS is handed nothing, and every bulk input is opened "
    "read-only."
)

WHAT_THIS_ENTRY_IS = (
    "The 100M rung's feature path, wired. `_prepare_edge_list_training` selects "
    "`host_int8_hybrid` and stamps `weighted_effective: true`; the per-update "
    "interval is measured at a FULL update of 2 x batch_size = 16,384 feature "
    "rows against the plan's 9.09 ms budget and the registered 4,163,754-update "
    "horizon. No fit runs."
)


class Round0262NodeError(RuntimeError):
    """R0262 fails closed."""


def _require(condition: Any, message: str) -> bool:
    """Assert a control's verdict as a call, and return what it proved.

    Written as a function rather than a bare `if`/`raise` so that a control's
    verdict is *passed* to something -- `vacuouscheck`'s `reported-not-tested`
    rule exists because R0255 built a perturbation, counted it into the
    artifact, and then ran the arm on the unperturbed input. A verdict that is
    only ever subscripted inside a dict literal has the same shape as that bug.
    """
    if not condition:
        raise Round0262NodeError(message)
    return True


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(_r0251_envelope(manifest))
    body["round_id"] = ROUND_ID
    return body


def _node_guard(label: str, *, interval_s: float = 0.05) -> EnforcedHostWatchdog:
    return EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        interval_s=float(interval_s),
        label=label,
    )


def _mem_available_bytes() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise Round0262NodeError("R0262 could not read MemAvailable")


def _require_headroom(*, label: str, required: int) -> dict[str, Any]:
    available = _mem_available_bytes()
    if available < required:
        raise Round0262NodeError(
            f"R0262 refuses to start {label}: MemAvailable {available} B is "
            f"below the required {required} B."
        )
    return {"mem_available_bytes": available, "required_bytes": int(required)}


def _interval_block(values: list[float], *, label: str) -> dict[str, Any]:
    ceiling = registered_ceiling_s()
    ordered = sorted(values)
    n = len(ordered)
    return {
        "label": label,
        "updates": n,
        "max_s": ordered[-1] if n else None,
        "max_over_the_ceiling": (ordered[-1] / ceiling) if n else None,
        "p50_s": ordered[n // 2] if n else None,
        "p99_s": ordered[min(n - 1, int(round(0.99 * (n - 1))))] if n else None,
        "min_s": ordered[0] if n else None,
        "mean_s": (sum(ordered) / n) if n else None,
        "updates_over_the_ceiling": sum(1 for value in ordered if value > ceiling),
        "interval_summary": interval_summary(ordered, ceiling_s=ceiling),
    }


# --------------------------------------------------------------------------- #
# backing and accounting controls -- positive controls that PLANT the defect
# --------------------------------------------------------------------------- #

def backing_and_accounting_controls(*, scratch: str) -> dict[str, Any]:
    """Plant each accounting blind spot and prove the shipped rule sees it.

    Three plants, each a defect a reader rule in this repo actually has:

    1. `ascontiguousarray` of a C-contiguous int32 memmap. This is verbatim what
       `HostStreamEdgeSampler.__init__` does to `sources`. It copies nothing.
       The `isinstance(x, np.memmap)` rule must call it host-resident (the
       defect) and the base-chain rule must call it file-backed (the fix).
       **If `classify_host_backing` stopped walking `.base` this control fails.**
    2. A genuinely anonymous array, which both rules must agree is anonymous —
       so plant 1's disagreement is not an artefact of a rule that always says
       "file-backed".
    3. A `MAP_SHARED` mapping. `RssAnon` must NOT move and `RssShmem` must, which
       is why an anonymous-bytes guard is blind to `load_array_polled`'s staging.
    """
    os.makedirs(scratch, exist_ok=True)
    checks: dict[str, Any] = {}

    member = os.path.join(scratch, "planted-endpoints.i32.npy")
    np.save(member, np.arange(4_000_000, dtype=np.int32))
    backing_file = np.load(member, mmap_mode="r")
    planted = np.ascontiguousarray(np.asarray(backing_file), dtype=np.int32)
    planted_verdict = classify_host_backing(planted)
    _require(
        planted_verdict["file_backed"],
        "R0262 positive control failed: the base-chain rule did not see through "
        "an ascontiguousarray view of a memmap",
    )
    _require(
        not planted_verdict["isinstance_np_memmap"],
        "R0262 positive control is vacuous: isinstance already saw the memmap, "
        "so the two rules do not disagree here",
    )
    _require(
        np.shares_memory(planted, backing_file),
        "R0262 positive control is vacuous: ascontiguousarray did copy, so "
        "nothing was planted",
    )
    checks["the_ascontiguousarray_copy_is_a_no_op"] = {
        "what_was_planted": (
            "np.ascontiguousarray(np.asarray(memmap), dtype=np.int32) -- verbatim "
            "HostStreamEdgeSampler.__init__ on an int32 C-contiguous member"
        ),
        "shares_memory_with_the_file_mapping": bool(
            np.shares_memory(planted, backing_file)
        ),
        "base_chain": planted_verdict["base_chain"],
        "base_chain_rule_says": planted_verdict["backing"],
        "isinstance_rule_says": (
            "anonymous_host_resident" if not planted_verdict["isinstance_np_memmap"]
            else "file_backed"
        ),
        "the_two_rules_agree": planted_verdict["isinstance_rule_agrees_with_base_chain"],
        "the_guard_catches_it": bool(planted_verdict["file_backed"]),
    }

    honest = np.empty(4_000_000, dtype=np.int32)
    honest_verdict = classify_host_backing(honest)
    _require(
        not honest_verdict["file_backed"],
        "R0262 control failed: the base-chain rule called an anonymous np.empty "
        "file-backed, so it is not a rule, it is a constant",
    )
    checks["an_honest_anonymous_array"] = {
        "base_chain": honest_verdict["base_chain"],
        "base_chain_rule_says": honest_verdict["backing"],
        "the_two_rules_agree": honest_verdict["isinstance_rule_agrees_with_base_chain"],
    }

    shared_bytes = 512 * (1 << 20)
    before = host_memory_accounting()
    shared = mmap.mmap(-1, shared_bytes, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
    view = np.frombuffer(shared, dtype=np.uint8)
    view[::4096] = 1  # fault every page
    anon_probe = host_anonymous_bytes()
    during = host_memory_accounting()
    del view
    shared.close()
    after = host_memory_accounting()
    anon_delta = during["RssAnon"] - before["RssAnon"]
    shmem_delta = during["RssShmem"] - before["RssShmem"]
    _require(
        shmem_delta >= shared_bytes // 2,
        f"R0262 control is vacuous: RssShmem moved {shmem_delta} B for a "
        f"{shared_bytes} B MAP_SHARED mapping, so nothing was planted",
    )
    _require(
        anon_delta <= shared_bytes // 2,
        f"R0262 control failed: RssAnon moved {anon_delta} B, so MAP_SHARED is "
        f"NOT invisible to an anonymous guard and the finding is wrong",
    )
    checks["map_shared_is_invisible_to_the_anonymous_guard"] = {
        "what_was_planted": f"{shared_bytes} B of MAP_SHARED|MAP_ANONYMOUS, fully faulted",
        "bytes": int(shared_bytes),
        "rss_anon_delta_bytes": anon_delta,
        "rss_shmem_delta_bytes": shmem_delta,
        "rss_file_delta_bytes": during["RssFile"] - before["RssFile"],
        "host_anonymous_bytes_during": anon_probe,
        "the_anonymous_guard_sees_it": bool(anon_delta > shared_bytes // 2),
        "rss_shmem_sees_it": bool(shmem_delta > shared_bytes // 2),
        "released": bool(after["RssShmem"] <= during["RssShmem"]),
        "why_it_matters": (
            "load_array_polled stages through a MAP_SHARED mapping, so a node "
            "that gates only on anonymous bytes cannot observe that staging."
        ),
    }
    # Computed from the checks, not asserted at them. Every arm above already
    # raised through `_require` if it failed, so this is a re-derivation from
    # the sealed fields rather than a literal standing in for one.
    held = [
        checks["the_ascontiguousarray_copy_is_a_no_op"]["the_guard_catches_it"],
        not checks["the_ascontiguousarray_copy_is_a_no_op"]["the_two_rules_agree"],
        checks["an_honest_anonymous_array"]["the_two_rules_agree"],
        not checks["map_shared_is_invisible_to_the_anonymous_guard"][
            "the_anonymous_guard_sees_it"],
        checks["map_shared_is_invisible_to_the_anonymous_guard"]["rss_shmem_sees_it"],
    ]
    return {
        "schema": "round0262-backing-and-accounting-controls-v1",
        "checks": checks,
        "controls_that_held": int(sum(1 for value in held if value)),
        "controls_evaluated": len(held),
        "every_control_held": bool(all(held)),
    }


# --------------------------------------------------------------------------- #
# node A -- quantise the substrate and account for the error over every row
# --------------------------------------------------------------------------- #

def run_quantise(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0262 round0262_nodes.run_quantise")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0262NodeError("R0262 quantise handler received another queue")
    node_id = str(active.get("node_id") or "quantise0262")
    label = "R0262 int8 substrate and full-substrate fidelity"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label=label)
    node_started = time.monotonic()
    headroom = _require_headroom(
        label=label, required=MIN_MEM_AVAILABLE_QUANTISE_BYTES)
    io_before = io_counters()

    ledger = CoverageLedger(node=node_id)
    window = ledger.window("R0262 quantise: pin, controls and the streamed encode")
    guard = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    payload: dict[str, Any] = {}
    try:
        with guard:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor("R0262 quantise entered")
            wrapped = window.wrap(recorder)

            # The numpy pin, CLOSED on this entry. R0259 recorded 2.4.4 and
            # never ran assert_pairwise_rule in its 100M node, so the bitwise
            # sum identity was unprotected against an upgrade on the path that
            # matters. This runs it here, before anything relies on it.
            wrapped("R0262 numpy pairwise rule")
            pin_started = time.monotonic()
            payload["numpy_pairwise_rule_selfcheck"] = assert_pairwise_rule()
            payload["numpy_pairwise_rule_wall_s"] = time.monotonic() - pin_started
            payload["numpy_version_observed"] = str(np.__version__)

            wrapped("R0262 backing and accounting controls")
            payload["backing_and_accounting_controls"] = (
                backing_and_accounting_controls(
                    scratch=os.path.join(output, "controls-scratch"))
            )

            wrapped("R0262 substrate dimension")
            payload["substrate"] = assert_substrate_dimension()

            wrapped("R0262 streamed int8 encode")
            payload["quantise"] = quantise_substrate_to_int8(
                source_path=SUBSTRATE_100M_PATH,
                expected_shape=SUBSTRATE_100M_SHAPE,
                int8_path=INT8_100M_PATH,
                scales_path=SCALES_100M_PATH,
                rows_per_chunk=QUANTISE_ROWS_PER_CHUNK,
                poll=wrapped,
            )
            gate.finish("R0262 quantise end")
    finally:
        gc.collect()

    window.close()
    node_wall = time.monotonic() - node_started
    tail = _guard_tail_reported(guard, label=label)
    scored = _score_gate_without_raising(gate, tail, label=label)
    report = gap_report(recorder.records, arm="quantise_and_fidelity")
    coverage = ledger.receipt(node_wall_s=node_wall)

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": QUANTISE_SCHEMA,
        "capability": QUANTISE_CAPABILITY,
        "node_id": node_id,
        "label": label,
        "what_this_entry_is": WHAT_THIS_ENTRY_IS,
        "safety_note": SAFETY_NOTE,
        "not_a_family_cell": NOT_A_FAMILY_CELL,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "is_a_family_cell": False,
        "gate_registered": False,
        "training_performed": False,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": False,
        "abort_flag_precondition": abort_flag,
        "headroom": headroom,
        "node_wall_s": node_wall,
        "io_counters": {"before": io_before, "after": io_counters()},
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "gap_report": report,
        "coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "host_memory_accounting_at_finish": host_memory_accounting(),
    })
    body.update(payload)
    _seal(output, f"{node_id}-int8-substrate.json", json_scrub(json_safe(body)))


# --------------------------------------------------------------------------- #
# node B -- the wired path, its interval, and the training-signal fidelity
# --------------------------------------------------------------------------- #

def _training_signal_fidelity(*, poll, rng) -> dict[str, Any]:
    """Three arms of the same measurement, at the registered batch shape.

    A: the same fp32 rows recomputed — the determinism control. If this is not
       exactly zero the other two arms are noise and must not be read.
    B: int8 of the same rows — the quantity under test.
    C: an independent minibatch — the stochastic gradient noise the registered
       recipe already trains through.

    B is only meaningful against C. A gradient perturbation is "small" or
    "large" relative to the noise the optimizer already tolerates, not against
    zero, and no prior round in this program has reported it that way.
    """
    import torch
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    substrate = np.load(SUBSTRATE_100M_PATH, mmap_mode="r", allow_pickle=False)
    stride = (ROWS_100M - FIDELITY_ROWS_PER_BLOCK) / (FIDELITY_SAMPLE_BLOCKS - 1)
    blocks = []
    for index in range(FIDELITY_SAMPLE_BLOCKS):
        poll(f"R0262 fidelity sample block {index}")
        start = int(index * stride)
        blocks.append(np.array(
            substrate[start:start + FIDELITY_ROWS_PER_BLOCK],
            dtype=np.float32, copy=True))
    rows = np.concatenate(blocks)
    del blocks
    encoded, scales = quantise_block_int8(rows)
    dequantised = dequantise_block(encoded, scales)

    difference = dequantised - rows
    error = np.linalg.norm(difference, axis=1)
    reference = np.linalg.norm(rows, axis=1)
    relative = error / reference
    cosine = (np.einsum("ij,ij->i", rows, dequantised)
              / (reference * np.linalg.norm(dequantised, axis=1)))

    left = rng.integers(0, len(rows), 200_000)
    right = rng.integers(0, len(rows), 200_000)
    pair_fp32 = np.linalg.norm(rows[left] - rows[right], axis=1)
    pair_int8 = np.linalg.norm(dequantised[left] - dequantised[right], axis=1)

    poll("R0262 fidelity model warmup")
    fp32_pool = torch.from_numpy(rows).cuda()
    int8_pool = torch.from_numpy(dequantised).cuda()
    estimator = ParametricUMAP(
        n_components=2, batch_size=BATCH_SIZE, pos_ratio=POS_RATIO,
        weighted_edge_sampling=True, device="cuda", n_epochs=1)
    estimator._init_model(DIMENSION)
    model = estimator.model
    loss_fn = torch.nn.BCELoss()
    targets = torch.zeros(BATCH_SIZE, device="cuda")
    targets[:POSITIVE_ROWS_PER_UPDATE] = 1.0

    def draw():
        source = torch.from_numpy(
            rng.integers(0, len(rows), BATCH_SIZE)).cuda()
        destination = torch.from_numpy(
            rng.integers(0, len(rows), BATCH_SIZE)).cuda()
        # The positives must be genuinely more similar than the negatives or the
        # warmup has no signal to learn; a small index offset stands in for the
        # graph's neighbours, which this node deliberately does not open.
        offset = torch.randint(
            1, 64, (POSITIVE_ROWS_PER_UPDATE,), device="cuda")
        destination[:POSITIVE_ROWS_PER_UPDATE] = (
            source[:POSITIVE_ROWS_PER_UPDATE] + offset) % len(rows)
        return source, destination

    def signal(source, destination, pool):
        model.zero_grad(set_to_none=True)
        left_embedding = model(pool.index_select(0, source).float())
        right_embedding = model(pool.index_select(0, destination).float())
        qs, _ = estimator._low_dim_qs(left_embedding, right_embedding)
        qs = torch.clamp(torch.nan_to_num(qs, nan=1e-7), 1e-7, 1 - 1e-7)
        loss = loss_fn(qs.float(), targets)
        loss.backward()
        gradient = torch.cat(
            [p.grad.reshape(-1) for p in model.parameters()]).detach().clone()
        return float(loss.detach()), gradient

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    warmup_losses = []
    for step in range(FIDELITY_WARMUP_STEPS):
        if step % 100 == 0:
            poll(f"R0262 fidelity warmup {step}")
        source, destination = draw()
        loss_value, _ = signal(source, destination, fp32_pool)
        optimizer.step()
        warmup_losses.append(loss_value)

    def relative_error(a, b):
        return float((b - a).norm() / a.norm())

    def cosine_of(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    arms: dict[str, dict[str, list[float]]] = {
        name: {"relative_l2": [], "cosine": []}
        for name in ("a_same_fp32_input_recomputed", "b_int8_of_the_same_rows",
                     "c_an_independent_minibatch")
    }
    loss_relative = []
    gradient_norms = []
    for index in range(FIDELITY_BATCHES):
        poll(f"R0262 fidelity batch {index}")
        source, destination = draw()
        other_source, other_destination = draw()
        loss_fp32, gradient_fp32 = signal(source, destination, fp32_pool)
        _, gradient_again = signal(source, destination, fp32_pool)
        loss_int8, gradient_int8 = signal(source, destination, int8_pool)
        _, gradient_other = signal(other_source, other_destination, fp32_pool)
        gradient_norms.append(float(gradient_fp32.norm()))
        for name, candidate in (
            ("a_same_fp32_input_recomputed", gradient_again),
            ("b_int8_of_the_same_rows", gradient_int8),
            ("c_an_independent_minibatch", gradient_other),
        ):
            arms[name]["relative_l2"].append(
                relative_error(gradient_fp32, candidate))
            arms[name]["cosine"].append(cosine_of(gradient_fp32, candidate))
        loss_relative.append(abs(loss_int8 - loss_fp32) / abs(loss_fp32))

    def summarise(values):
        ordered = sorted(values)
        return {
            "mean": float(np.mean(ordered)),
            "median": float(np.median(ordered)),
            "min": float(ordered[0]),
            "max": float(ordered[-1]),
        }

    arm_report = {
        name: {"relative_l2": summarise(values["relative_l2"]),
               "cosine": summarise(values["cosine"])}
        for name, values in arms.items()
    }
    determinism = arm_report["a_same_fp32_input_recomputed"]["relative_l2"]["max"]
    if determinism != 0.0:
        raise Round0262NodeError(
            "R0262: the determinism arm is not exactly zero "
            f"({determinism}); the int8 and SGD arms are then not comparable"
        )
    int8_mean = arm_report["b_int8_of_the_same_rows"]["relative_l2"]["mean"]
    sgd_mean = arm_report["c_an_independent_minibatch"]["relative_l2"]["mean"]
    int8_median = arm_report["b_int8_of_the_same_rows"]["relative_l2"]["median"]
    sgd_median = arm_report["c_an_independent_minibatch"]["relative_l2"]["median"]

    del fp32_pool, int8_pool, rows, dequantised, encoded, difference
    torch.cuda.empty_cache()

    return {
        "schema": "round0262-int8-training-signal-fidelity-v1",
        "rows_sampled": int(len(rows)),
        "sample_blocks": FIDELITY_SAMPLE_BLOCKS,
        "sampled_from": SUBSTRATE_100M_PATH,
        "encoding": "per-row symmetric max-abs int8 with an exact fp16 row scale",
        "representation_error": {
            "relative_l2_mean": float(relative.mean()),
            "relative_l2_p99": float(np.percentile(relative, 99)),
            "relative_l2_max": float(relative.max()),
            "aggregate_relative_l2": float(
                np.sqrt((error ** 2).sum() / (reference ** 2).sum())),
            "cosine_mean": float(cosine.mean()),
            "cosine_min": float(cosine.min()),
            "components_at_the_clip": int((np.abs(encoded) >= 127).sum()),
            "components_total": int(encoded.size),
        },
        "pairwise_distance": {
            "pairs": int(len(pair_fp32)),
            "relative_error_mean": float(
                np.abs(pair_int8 - pair_fp32).mean() / pair_fp32.mean()),
            "pearson_r": float(np.corrcoef(pair_fp32, pair_int8)[0, 1]),
        },
        "warmup_updates": FIDELITY_WARMUP_STEPS,
        "warmup_loss_first": warmup_losses[0],
        "warmup_loss_last": warmup_losses[-1],
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "batches": FIDELITY_BATCHES,
        "arms": arm_report,
        "bce_loss_relative_difference": summarise(loss_relative),
        "int8_perturbation_over_sgd_noise_mean": int8_mean / sgd_mean,
        "int8_perturbation_over_sgd_noise_median": int8_median / sgd_median,
        "the_determinism_arm_is_exactly_zero": True,
    }


def _wired_body(*, job, poll, rng) -> dict[str, Any]:
    import torch
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )
    import json as _json

    payload: dict[str, Any] = {"wiring": WIRING}

    poll("R0262 numpy pairwise rule")
    payload["numpy_pairwise_rule_selfcheck"] = assert_pairwise_rule()
    payload["numpy_version_observed"] = str(np.__version__)

    poll("R0262 rung")
    with open(R0243_FUZZY_MANIFEST, "r", encoding="utf-8") as handle:
        manifest = _json.load(handle)
    payload["rung"] = {
        "contract": assert_rung(
            manifest, expected=RUNG_100M,
            entry="experiments.round0262_nodes.run_wired"),
        "substrate": assert_substrate_dimension(),
    }

    # R0262 addendum 2026-08-12: the fidelity arms run FIRST, before the
    # 38,600,000,000 B host-int8 X exists. The first attempt ran them last and
    # tripped the node's own anonymous-budget watchdog at 64,433,184,768 B
    # against 64,424,509,440 B -- an overshoot of 8,675,328 B (0.013%), because
    # `estimator._X_dev` still referenced X while the fidelity pools were live.
    # Ordering them this way means the two never coexist; nothing scientific
    # changes, because the fidelity arms sample the fp32 substrate directly and
    # never touch the int8 artifact or the graph.
    poll("R0262 training signal fidelity")
    payload["training_signal_fidelity"] = _training_signal_fidelity(
        poll=poll, rng=rng)
    gc.collect()
    torch.cuda.empty_cache()
    payload["host_memory_accounting_after_fidelity"] = host_memory_accounting()

    poll("R0262 open the 100M graph")
    started = time.monotonic()
    sources, targets, weights, n_nodes = load_edge_arrays(
        R0243_FUZZY_DIR, load_weights=True)
    payload["graph"] = {
        "path": R0243_FUZZY_DIR,
        "open_wall_s": time.monotonic() - started,
        "directed_edges": int(len(sources)),
        "n_nodes": int(n_nodes),
        "dtypes": [str(sources.dtype), str(targets.dtype), str(weights.dtype)],
        "opened_through": "the shipped load_edge_arrays",
        # The mandate's third defect, measured on the real members rather than
        # on a plant: what backs the arrays the sampler will call "host".
        "sources_backing": classify_host_backing(sources),
        "targets_backing": classify_host_backing(targets),
        "weights_backing": classify_host_backing(weights),
    }
    if int(len(sources)) != DIRECTED_EDGES_100M:
        raise Round0262NodeError(
            f"R0262: the 100M graph holds {len(sources)} edges, expected "
            f"{DIRECTED_EDGES_100M}")

    poll("R0262 load the host-int8 X")
    host_x, load_receipt = build_host_int8_source(
        int8_path=INT8_100M_PATH, scales_path=SCALES_100M_PATH,
        row_count=ROWS_100M, dimension=DIMENSION, device="cuda",
        buffer_rows=BATCH_SIZE, poll=poll, rows_per_chunk=2_000_000)
    payload["host_int8_x"] = load_receipt
    payload["host_int8_x"]["buffer_admits_batch"] = assert_buffer_admits_batch(
        host_x, BATCH_SIZE)

    poll("R0262 build the loader through the shipped entry")
    estimator = ParametricUMAP(
        n_components=2, batch_size=BATCH_SIZE, pos_ratio=POS_RATIO,
        weighted_edge_sampling=True, positive_target_mode="binary",
        device="cuda", n_epochs=1)
    estimator._allow_model_before_admission = True
    started = time.monotonic()
    dataset, loader, n_pos_edges = estimator._prepare_edge_list_training(
        host_x, R0243_FUZZY_DIR, ROWS_100M, low_memory=False, random_state=0)
    build_wall = time.monotonic() - started
    pipeline = dict(estimator._pipeline_info)

    # The claim, checked rather than described.
    if pipeline.get("pipeline") != "host_int8_hybrid":
        raise Round0262NodeError(
            f"R0262: the shipped entry selected {pipeline.get('pipeline')!r}, "
            f"not 'host_int8_hybrid'")
    if not pipeline.get("weighted_effective"):
        raise Round0262NodeError(
            "R0262: the host-int8 pipeline did not stamp weighted_effective")
    if pipeline.get("positive_sampling") != "weighted_with_replacement":
        raise Round0262NodeError(
            f"R0262: positive_sampling is {pipeline.get('positive_sampling')!r}")
    if type(loader).__name__ != "HostStreamEdgeSampler":
        raise Round0262NodeError(
            f"R0262: the loader is {type(loader).__name__}")

    payload["pipeline"] = {
        "selected": pipeline,
        "build_wall_s": build_wall,
        "n_pos_edges": int(n_pos_edges),
        "loader_class": type(loader).__name__,
        "dataset_is_the_host_int8_x": bool(dataset is host_x),
        "fast_device_path": bool(estimator._fast_device_path),
        "num_pos_per_update": int(loader.num_pos),
        "num_neg_per_update": int(loader.num_neg),
        "feature_rows_per_update": FEATURE_ROWS_PER_UPDATE,
        # What the sampler's "host-resident" endpoint copy actually produced.
        "sampler_src_backing": classify_host_backing(loader._src_h),
        "sampler_dst_backing": classify_host_backing(loader._dst_h),
        "sampler_cdf_backing": classify_host_backing(loader._cdf_h),
        "sampler_cdf_bytes": int(loader._cdf_h.nbytes),
    }
    payload["host_memory_accounting_with_everything_resident"] = (
        host_memory_accounting())
    payload["host_anonymous_bytes_with_everything_resident"] = (
        host_anonymous_bytes())

    poll("R0262 per-update interval probe")
    if PROBE_UPDATES > MAX_PROBE_UPDATES:
        raise Round0262NodeError(
            f"R0262 refuses an unbounded probe: {PROBE_UPDATES} updates")
    iterator = iter(loader)
    intervals: list[float] = []
    endpoint_rows = 0
    for index in range(PROBE_WARMUP_UPDATES + PROBE_UPDATES):
        if index % 50 == 0:
            poll(f"R0262 update probe {index}")
        started = time.monotonic()
        src_feats, dst_feats, batch_targets = next(iterator)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if index >= PROBE_WARMUP_UPDATES:
            intervals.append(elapsed)
            endpoint_rows += int(src_feats.shape[0]) + int(dst_feats.shape[0])
    if int(src_feats.shape[0]) + int(dst_feats.shape[0]) != FEATURE_ROWS_PER_UPDATE:
        raise Round0262NodeError(
            f"R0262: an update gathered {src_feats.shape[0]}+{dst_feats.shape[0]} "
            f"rows, expected {FEATURE_ROWS_PER_UPDATE}")

    block = _interval_block(intervals, label="wired host-int8 per update")
    p50 = block["p50_s"]
    payload["per_update_interval"] = {
        **block,
        "warmup_updates": PROBE_WARMUP_UPDATES,
        "feature_rows_per_update": FEATURE_ROWS_PER_UPDATE,
        "feature_rows_gathered": endpoint_rows,
        "src_dtype": str(src_feats.dtype),
        "src_device": str(src_feats.device),
        "src_shape": list(src_feats.shape),
        "targets_shape": list(batch_targets.shape),
        "this_is_a_full_update_not_half_of_one": True,
        "update_budget_s": UPDATE_BUDGET_S,
        "p50_over_the_update_budget": p50 / UPDATE_BUDGET_S,
        "lr_horizon": LR_HORIZON_100M,
        "horizon_feature_path_hours_at_p50": (
            p50 * LR_HORIZON_100M / 3600.0),
        "what_this_interval_contains": (
            "one HostStreamEdgeSampler.__next__: the queue pull of a "
            "producer-drawn positive batch, the on-device negative draw, the "
            "index D2H, the fused host int8 gather of both endpoints, the H2D "
            "of both endpoints and their scales, the on-device dequantisation, "
            "and a torch.cuda.synchronize. It does NOT contain the model "
            "forward/backward or the optimizer step."
        ),
    }

    loader.close()
    payload["closed_the_sampler"] = True
    # `estimator._X_dev` and `dataset` both alias the host-int8 X; dropping only
    # the local name frees nothing. That is what tripped the first attempt.
    estimator._X_dev = None
    del loader, dataset, host_x, sources, targets, weights, estimator
    gc.collect()
    torch.cuda.empty_cache()
    payload["host_memory_accounting_after_release"] = host_memory_accounting()
    return payload


def run_wired(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0262 round0262_nodes.run_wired")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0262NodeError("R0262 wired handler received another queue")
    node_id = str(active.get("node_id") or "wired0262")
    label = "R0262 wired host-int8 pipeline and per-update intervals"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label=label)
    node_started = time.monotonic()
    headroom = _require_headroom(label=label, required=MIN_MEM_AVAILABLE_BYTES)
    io_before = io_counters()
    rng = np.random.default_rng(20260812)

    ledger = CoverageLedger(node=node_id)
    window = ledger.window("R0262 wired: graph, X, pipeline, intervals, fidelity")
    guard = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    payload: dict[str, Any] = {}
    try:
        with guard:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor("R0262 wired entry entered")
            wrapped = window.wrap(recorder)
            payload = _wired_body(job=job, poll=wrapped, rng=rng)
            gate.finish("R0262 wired end")
    finally:
        gc.collect()

    window.close()
    node_wall = time.monotonic() - node_started
    tail = _guard_tail_reported(guard, label=label)
    scored = _score_gate_without_raising(gate, tail, label=label)
    report = gap_report(recorder.records, arm="wired_host_int8_entry")
    coverage = ledger.receipt(node_wall_s=node_wall)

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": WIRED_SCHEMA,
        "capability": WIRED_CAPABILITY,
        "node_id": node_id,
        "label": label,
        "what_this_entry_is": WHAT_THIS_ENTRY_IS,
        "safety_note": SAFETY_NOTE,
        "not_a_family_cell": NOT_A_FAMILY_CELL,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "is_a_family_cell": False,
        "gate_registered": False,
        "training_performed": False,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": False,
        "abort_flag_precondition": abort_flag,
        "headroom": headroom,
        "declared_anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
        "node_wall_s": node_wall,
        "io_counters": {"before": io_before, "after": io_counters()},
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "gap_report": report,
        "coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "host_memory_accounting_at_finish": host_memory_accounting(),
    })
    body.update(payload)
    _seal(output, f"{node_id}-wired-host-int8.json", json_scrub(json_safe(body)))


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0262 round0262_nodes.run_job")
    action = str(job.get("action") or "")
    if action == QUANTISE_ACTION:
        run_quantise(active, job)
        return
    if action == WIRED_ACTION:
        run_wired(active, job)
        return
    raise Round0262NodeError(f"R0262 does not serve action {action!r}")
