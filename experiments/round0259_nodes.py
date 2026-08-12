"""Execute R0259 — the 100M rung's loader, its entry, and one bounded probe.

Two nodes.

`controls_0259` (CPU)
    Ten positive controls, each planting a defect that the **shipped** path
    accepts. Five structural: four sources that pass R0258's `chunk_loop_polls`
    (review-0258-01 §D.3 wrote them and reported them passing) plus the runtime
    no-op poll that no static guard can see. Five numeric: chunked sums that are
    not numpy's pairwise reduction, all refused bitwise and all accepted by
    `np.allclose`. Then the container controls (five wrong-shape streamed graphs
    that must fail loudly, and one bulk `.npz` that must take the old path
    unchanged), the rung applicability controls in both directions, and the
    numpy pairwise self-check.

`train100m_0259` (GPU)
    The 100M rung's registered entry, `run_train_100m`, exercised at the real
    rung on a node holding a live CUDA context. It resolves the rung, checks the
    substrate's dimension, opens R0243's three streamed members, runs the polled
    bounds / validity / contiguous / CDF stages R0258 proved bitwise identical,
    removes R0258's named unpollable residue with the chunked pairwise sum,
    materialises both endpoint arrays into genuinely resident memory, and then
    measures the per-batch producer interval at 100M over a **registered bound**
    of batches — resident and file-backed, side by side.

**No fit runs and none is claimed.** `run_train_100m` stops where the shipped
fast path stops being constructible at this rung: `DeviceArrayDataset` would
have to hold `100,000,000 x 384` fp16 = `76,800,000,000` B on a `33,679,736,832`
B card. That is measured and reported, not asserted, and it is the item this
round hands forward.

Safety: every bulk input is opened read-only; no process is signalled; no child
process is started; cuVS is handed nothing; no subprocess is wrapped in a
timeout. The host guard gates on ANONYMOUS bytes, never RSS.
"""
from __future__ import annotations

import gc
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
from basemap.round0253_coverage import CoverageLedger, coverage_summary
from basemap.round0253_stop_hooks import (
    NOT_A_FAMILY_CELL,
    THE_INSTRUMENT_IS_DEFEATABLE,
    install_stop_hooks,
    over_the_ceiling,
    registered_ceiling_s,
)
from basemap.round0254_writeback import interval_summary
from basemap.round0258_graph_load import (
    SCAN_CHUNK_ELEMENTS,
    endpoint_bounds_polled,
    weight_validity_polled,
)
from basemap import round0259_hundred_m as rung
from basemap.round0259_hundred_m import (
    RUNG_100M,
    SUBSTRATE_100M_PATH,
    SUBSTRATE_100M_SHAPE,
    assert_chunk_loops_poll_v2,
    assert_container_defect_controls,
    assert_pairwise_rule,
    assert_resident,
    assert_rung,
    assert_rung_applicability_controls,
    assert_structural_defect_controls_v2,
    assert_substrate_dimension,
    bulk_npz_is_not_claimed,
    endpoint_residency,
    materialise_endpoints,
    weight_cdf_fully_polled,
)
from experiments.round0253_nodes import _cuda_maps_evidence, force_remove_tree
from experiments.round0251_nodes import (
    _guard_tail_reported,
    _node_gate,
    _receipt_envelope as _r0251_envelope,
    _score_gate_without_raising,
    _seal,
    _start_node,
)


ROUND_ID = "0259"

CONTROLS_ACTION = "round0259_hundred_m_controls"
TRAIN_ACTION = "round0259_100m_entry_and_batch_probe"

CONTROLS_CAPABILITY = "round0259-hundred-m-loader-and-guard-controls-v1"
TRAIN_CAPABILITY = "round0259-100m-entry-and-per-batch-intervals-v1"

CONTROLS_SCHEMA = "round0259-hundred-m-loader-and-guard-controls-v1"
TRAIN_SCHEMA = "round0259-100m-entry-and-per-batch-intervals-v1"

#: Anonymous budget, as arithmetic. Two resident int32 endpoint arrays at
#: `2,511,103,254 x 4 = 10,044,413,016` B each, plus the float64 CDF at
#: `20,088,826,032` B, is `40,177,652,064` B = `37.42` GiB. `48` GiB is above it
#: and below the registered `max_declared_anonymous_budget_bytes` of `60` GiB.
NODE_ANON_BUDGET_BYTES = 48 * (1 << 30)

#: Refuse to start without this much MemAvailable: the `37.42` GiB peak plus the
#: `28` GiB of page cache the three members occupy plus headroom.
MIN_MEM_AVAILABLE_BYTES = 72 * (1 << 30)

#: The registered per-batch bounds. `run_train_100m` refuses an unbounded probe:
#: this round is not a fit and must not become one by omission.
PROBE_BATCHES_RESIDENT = 1000
PROBE_BATCHES_FILE_BACKED = 120
PROBE_BATCHES_SUBSTRATE = 60
MAX_PROBE_BATCHES = 5000

#: `HostStreamEdgeSampler`'s own numbers, verbatim: `batch_size = 4096`,
#: `pos_ratio = 0.2` -> `num_pos = int(4096 * 0.2) = 819` positive draws and
#: `4096 - 819 = 3277` negatives per batch.
BATCH_SIZE = 4096
POS_RATIO = 0.2

#: The three leaf sizes review-0258-01 §E tabulated. Reproduced at the real rung.
SUM_PROOF_LEAVES = (1 << 24, 1 << 20, 1 << 14)

SAFETY_NOTE = (
    "no node in this module signals any process, starts a child process, hands "
    "cuVS anything, or wraps a subprocess in a timeout. The three edge members "
    "and the substrate are opened as read-only np.memmaps and their residency is "
    "classified by walking the .base chain, not by isinstance -- review-0258-01 "
    "§H.3 showed R0230's isinstance rule mis-classifies exactly the object that "
    "matters. The probe node holds a live CUDA context for its whole wall on "
    "purpose: /proc/<pid>/maps matches libcuda, so this process must never be "
    "signalled, which is the only case whose stop latency matters."
)

WHAT_THIS_ENTRY_IS = (
    "run_train_100m is the 100M rung's registered entry. review-0258-01 §H.1 "
    "established that round0113_nodes.run_train is not one: it is a ~2M, k=50, "
    "768-d, R0113-schema trainer, and three manifest checks fire before the "
    "container mismatch ever reaches load_edge_arrays. This entry accepts "
    "100,000,000 rows, 384 dimensions and R0243's streamed edge schema, and "
    "refuses everything else through assert_rung on its first line. It performs "
    "every stage between 'the graph file exists' and 'the first batch is ready' "
    "and then measures per-batch producer intervals under a registered bound. It "
    "contains NO fit loop and this round claims no train."
)

WHY_THERE_IS_NO_FIT_HERE = (
    "the shipped fast path gathers features with DeviceArrayDataset.index_select "
    "from an X held resident on the device. At this rung X is 100,000,000 x 384; "
    "at fp16 that is 76,800,000,000 B against a 33,679,736,832 B card. The "
    "feature gather of a 100M fit therefore cannot be constructed on the shipped "
    "path at all, and this node measures the host-side alternative -- a per-batch "
    "gather of 4,096 rows from the 153,600,000,128 B substrate memmap -- rather "
    "than asserting a number for a path that does not exist."
)


class Round0259NodeError(RuntimeError):
    """R0259 fails closed."""


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
    raise Round0259NodeError("R0259 could not read MemAvailable")


def _require_headroom(*, label: str) -> dict[str, Any]:
    available = _mem_available_bytes()
    if available < MIN_MEM_AVAILABLE_BYTES:
        raise Round0259NodeError(
            f"R0259 refuses to start {label}: MemAvailable {available} B is below "
            f"the required {MIN_MEM_AVAILABLE_BYTES} B."
        )
    return {"mem_available_bytes": available,
            "required_bytes": MIN_MEM_AVAILABLE_BYTES}


def _interval_block(values: list[float], *, label: str) -> dict[str, Any]:
    ceiling = registered_ceiling_s()
    ordered = sorted(values)
    n = len(ordered)
    return {
        "label": label,
        "batches": n,
        "max_s": ordered[-1] if n else None,
        "max_over_the_ceiling": (ordered[-1] / ceiling) if n else None,
        "p50_s": ordered[n // 2] if n else None,
        "p99_s": ordered[min(n - 1, int(round(0.99 * (n - 1))))] if n else None,
        "min_s": ordered[0] if n else None,
        "mean_s": (sum(ordered) / n) if n else None,
        "batches_over_the_ceiling": sum(1 for value in ordered if value > ceiling),
        "interval_summary": interval_summary(ordered, ceiling_s=ceiling),
    }


# --------------------------------------------------------------------------- #
# node A -- the controls
# --------------------------------------------------------------------------- #


def run_controls(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0259 round0259_nodes.run_controls")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0259NodeError("R0259 controls handler received another queue")
    node_id = str(active.get("node_id") or "controls_0259")
    ledger = CoverageLedger(node=node_id)
    label = "R0259 loader, rung and guard controls"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0259 controls")
    scratch = str(job.get("scratch") or "/data/tmp/round0259-controls")
    force_remove_tree(scratch)

    window = ledger.window("R0259 controls stage")
    guard = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    try:
        with guard:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor("R0259 controls stage entered")
            wrapped = window.wrap(recorder)

            structural = assert_structural_defect_controls_v2()
            wrapped("R0259 structural plants refused by the strengthened guard")
            pairwise = assert_pairwise_rule()
            wrapped("R0259 numpy pairwise split rule re-proved bitwise")
            audit = assert_chunk_loops_poll_v2()
            wrapped("R0259 chunk-loop audit v2 complete")
            containers = assert_container_defect_controls(scratch)
            wrapped("R0259 wrong-shape containers refused")
            bulk = bulk_npz_is_not_claimed(scratch)
            if not bulk["the_bulk_path_is_unchanged"]:
                raise Round0259NodeError(
                    "R0259: the loader branch changed the bulk .npz path"
                )
            wrapped("R0259 bulk .npz path proved unchanged")
            applicability = assert_rung_applicability_controls()
            wrapped("R0259 rung applicability proved in both directions")
            incompatibilities = _incompatibility_census()
            wrapped("R0259 incompatibility census complete")
            gate.finish("R0259 controls stage end")
    finally:
        force_remove_tree(scratch)
    window.close()
    tail = _guard_tail_reported(guard, label=label)
    scored = _score_gate_without_raising(gate, tail, label=label)
    report = gap_report(recorder.records, arm="controls")
    coverage = ledger.receipt()

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": CONTROLS_SCHEMA,
        "capability": CONTROLS_CAPABILITY,
        "capabilities": [CONTROLS_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "node": node_id,
        "arm": "controls",
        "gate_registered": False,
        "is_a_family_cell": False,
        "not_a_family_cell": NOT_A_FAMILY_CELL,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "safety_note": SAFETY_NOTE,
        "registered_ceiling_s": registered_ceiling_s(),
        "structural_defect_controls_v2": structural,
        "pairwise_rule_selfcheck": pairwise,
        "chunk_loop_audit_v2": audit,
        "container_defect_controls": containers,
        "bulk_npz_untouched": bulk,
        "rung_applicability_controls": applicability,
        "the_five_incompatibilities": incompatibilities,
        "plants_total": (
            int(structural["defects_planted"])
            + int(pairwise["numeric_controls"]["defects_planted"])
        ),
        "training_performed": False,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": False,
        "gap_report": report,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
    })
    _seal(output, f"{node_id}-hundred-m-controls.json", json_scrub(json_safe(body)))


def _incompatibility_census() -> dict[str, Any]:
    """The five blocking mismatches, each evaluated rather than described.

    review-0258-01 §H.1 says *"the container mismatch is the last of four
    incompatibilities"* while its own table lists four MANIFEST checks. Counted
    from `round0113_prompt_contrast.load_graph` there are five, and this census
    evaluates every one against R0243's real sealed manifest rather than quoting
    the review.
    """
    from basemap import round0113_prompt_contrast as r0113
    from basemap.artifact_identity import expected_input_signature
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )

    manifest_path = rung.R0243_FUZZY_MANIFEST
    manifest = r0113.read_sealed(manifest_path, label="R0243 sealed fuzzy graph")
    directory = os.path.dirname(manifest_path)
    header = os.path.join(directory, rung.MEMBER_FILENAMES["header"])
    member = os.path.join(directory, rung.MEMBER_FILENAMES["sources"])

    checks = {
        "I1_schema": {
            "r0113_requires": r0113.GRAPH_SCHEMA,
            "r0243_has": manifest.get("schema"),
            "blocks": manifest.get("schema") != r0113.GRAPH_SCHEMA,
        },
        "I2_rows": {
            "r0113_requires": int(r0113.RETAINED_ROWS),
            "r0243_has": int(manifest.get("rows", -1)),
            "r0113_reads_key": "retained_rows",
            "r0243_seals_key": "rows",
            "blocks": int(manifest.get("rows", -1)) != int(r0113.RETAINED_ROWS),
        },
        "I3_k": {
            "r0113_requires": int(r0113.GRAPH_K),
            "r0243_has": int(manifest.get("k", -1)),
            "blocks": int(manifest.get("k", -1)) != int(r0113.GRAPH_K),
        },
        "I4_dimension": {
            "r0113_requires": int(r0113.DIMENSION),
            "r0243_has": int(SUBSTRATE_100M_SHAPE[1]),
            "r0243_manifest_carries_a_dimension_key": "dimension" in manifest,
            "checked_against": SUBSTRATE_100M_PATH,
            "blocks": int(SUBSTRATE_100M_SHAPE[1]) != int(r0113.DIMENSION),
        },
    }

    # I5, evaluated three ways against the shipped loader BEFORE the branch, by
    # calling the pre-branch body directly.
    def _shipped_load_edge_arrays(path):
        npz = np.load(path, mmap_mode="r")
        return npz["sources"]

    container: dict[str, Any] = {}
    for name, argument in (
        ("the_header_npz", header),
        ("a_bulk_member_npy", member),
        ("the_artifact_directory", directory),
    ):
        try:
            _shipped_load_edge_arrays(argument)
            container[name] = {"raised": None, "opened": True}
        except Exception as error:  # noqa: BLE001 -- the failure IS the datum
            container[name] = {"raised": type(error).__name__, "opened": False}
    checks["I5_container"] = {
        "r0113_requires": "one bulk .npz carrying a `sources` member",
        "r0243_has": "three streamed .npy members plus a scalar header .npz",
        "the_pre_branch_loader_body_on_each_100m_path": container,
        "blocks": all(not entry["opened"] for entry in container.values()),
    }

    # And the branch opens all three of them.
    branch: dict[str, Any] = {}
    for name, argument in (
        ("the_header_npz", header), ("the_artifact_directory", directory)
    ):
        started = time.monotonic()
        sources, targets, weights, n_nodes = load_edge_arrays(argument)
        branch[name] = {
            "open_s": time.monotonic() - started,
            "edges": int(sources.shape[0]),
            "n_nodes": int(n_nodes),
            "dtypes": [str(sources.dtype), str(targets.dtype), str(weights.dtype)],
            "file_backed": rung.file_backed(sources),
        }
        del sources, targets, weights

    # R0113's own entry must still refuse the 100M manifest, loudly.
    signature = expected_input_signature(manifest_path)
    try:
        r0113.load_graph(
            manifest_path,
            expected_sha256=signature["sha256"],
            arm=sorted(r0113.ARMS)[0],
        )
        r0113_refusal = None
    except Exception as error:  # noqa: BLE001
        r0113_refusal = {"error": type(error).__name__, "message": str(error)}

    return {
        "schema": "round0259-incompatibility-census-v1",
        "checks": checks,
        "incompatibilities_that_block": sorted(
            name for name, entry in checks.items() if entry["blocks"]
        ),
        "the_r0259_branch_opens_the_100m_graph": branch,
        "the_2m_entry_still_refuses_the_100m_manifest": r0113_refusal,
        "note": (
            "review-0258-01 §H.1 counted four and its own table listed four "
            "MANIFEST checks; counted from load_graph's source there are five "
            "distinct blocking mismatches and the container is the fifth. Each "
            "row above is evaluated against R0243's sealed manifest, not quoted."
        ),
    }


# --------------------------------------------------------------------------- #
# node B -- the 100M rung's registered entry, exercised
# --------------------------------------------------------------------------- #


def run_train_100m(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """The 100M rung's entry. See `WHAT_THIS_ENTRY_IS`."""
    install_stop_hooks(label="R0259 round0259_nodes.run_train_100m")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0259NodeError("R0259 entry received another queue")
    node_id = str(active.get("node_id") or "train100m_0259")
    label = "R0259 100M entry and per-batch probe"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0259 entry")
    node_started = time.monotonic()

    resident_batches = int(job.get("probe_batches_resident", PROBE_BATCHES_RESIDENT))
    file_backed_batches = int(
        job.get("probe_batches_file_backed", PROBE_BATCHES_FILE_BACKED)
    )
    substrate_batches = int(
        job.get("probe_batches_substrate", PROBE_BATCHES_SUBSTRATE)
    )
    for name, value in (
        ("probe_batches_resident", resident_batches),
        ("probe_batches_file_backed", file_backed_batches),
        ("probe_batches_substrate", substrate_batches),
    ):
        if value <= 0 or value > MAX_PROBE_BATCHES:
            raise Round0259NodeError(
                f"R0259 {name} = {value} is outside (0, {MAX_PROBE_BATCHES}]. This "
                "entry refuses an unbounded probe: the round is not a fit."
            )

    import torch

    if not torch.cuda.is_available():
        raise Round0259NodeError("R0259 probe requires a live CUDA device")
    held = torch.zeros(1024, 1024, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    device_properties = torch.cuda.get_device_properties(0)
    cuda_context = {
        "device": torch.cuda.get_device_name(0),
        "total_device_memory_bytes": int(device_properties.total_memory),
        "resident_tensor_bytes": int(held.numel() * held.element_size()),
        "torch_cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "maps": _cuda_maps_evidence(),
        "held_for": (
            "the whole node. Every interval below is measured in the state a "
            "training node is in -- the state that cannot be signalled."
        ),
    }
    if not cuda_context["maps"]["this_process_maps_a_cuda_library"]:
        raise Round0259NodeError("R0259 probe node holds no CUDA mapping")

    io_before = io_counters()
    headroom = _require_headroom(label=label)
    ledger = CoverageLedger(node=node_id)
    window = ledger.window("R0259 100M entry: rung, load, prepare and probe")
    guard = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    stages: dict[str, Any] = {}
    anonymous: dict[str, int] = {}
    try:
        with guard:
            gate.start()
            recorder = PollRecorder(gate=gate, clock=time.monotonic)
            recorder.anchor("R0259 100M entry entered")
            wrapped = window.wrap(recorder)
            anonymous["before"] = host_anonymous_bytes()["anonymous_bytes"]

            payload = _entry_body(
                job=job,
                poll=wrapped,
                torch=torch,
                stages=stages,
                anonymous=anonymous,
                resident_batches=resident_batches,
                file_backed_batches=file_backed_batches,
                substrate_batches=substrate_batches,
            )
            gate.finish("R0259 100M entry end")
    finally:
        del held
        torch.cuda.empty_cache()
        gc.collect()
    window.close()
    node_wall = time.monotonic() - node_started
    tail = _guard_tail_reported(guard, label=label)
    scored = _score_gate_without_raising(gate, tail, label=label)
    report = gap_report(recorder.records, arm="hundred_m_entry")
    io_after = io_counters()
    coverage = ledger.receipt(node_wall_s=node_wall)

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": TRAIN_SCHEMA,
        "capability": TRAIN_CAPABILITY,
        "capabilities": [TRAIN_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "node": node_id,
        "arm": "hundred_m_entry",
        "gate_registered": False,
        "is_a_family_cell": False,
        "not_a_family_cell": NOT_A_FAMILY_CELL,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "safety_note": SAFETY_NOTE,
        "what_this_entry_is": WHAT_THIS_ENTRY_IS,
        "why_there_is_no_fit_here": WHY_THERE_IS_NO_FIT_HERE,
        "registered_ceiling_s": registered_ceiling_s(),
        "cuda_context": cuda_context,
        "mem_available_at_start": headroom,
        "declared_anonymous_budget_bytes": NODE_ANON_BUDGET_BYTES,
        "scan_chunk_elements": SCAN_CHUNK_ELEMENTS,
        "batch_size": BATCH_SIZE,
        "pos_ratio": POS_RATIO,
        "probe_batches": {
            "resident": resident_batches,
            "file_backed": file_backed_batches,
            "substrate": substrate_batches,
            "max_allowed": MAX_PROBE_BATCHES,
        },
        "stages": stages,
        "host_anonymous_bytes_by_stage": dict(sorted(anonymous.items())),
        "peak_host_anonymous_bytes_observed": max(anonymous.values()),
        "io_delta": {
            key: int(io_after.get(key, 0) - io_before.get(key, 0))
            for key in sorted(set(io_before) | set(io_after))
        },
        "training_performed": False,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": False,
        "gap_report": report,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "coverage_summary": coverage_summary([coverage]),
    })
    body.update(payload)
    _seal(output, f"{node_id}-100m-entry.json", json_scrub(json_safe(body)))


def _entry_body(
    *,
    job: Mapping[str, Any],
    poll: Any,
    torch: Any,
    stages: dict[str, Any],
    anonymous: dict[str, int],
    resident_batches: int,
    file_backed_batches: int,
    substrate_batches: int,
) -> dict[str, Any]:
    """Everything between 'the graph file exists' and 'the first batch is ready'."""
    from basemap import round0113_prompt_contrast as r0113
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )

    manifest_path = str(job.get("graph_manifest") or rung.R0243_FUZZY_MANIFEST)

    # ---- S0: applicability, checked before anything is opened --------------- #
    started = time.monotonic()
    manifest = r0113.read_sealed(manifest_path, label="R0243 sealed fuzzy graph")
    contract = assert_rung(manifest, expected=RUNG_100M, entry="run_train_100m")
    substrate = assert_substrate_dimension()
    directed_edges = int(manifest["directed_edges"])
    stages["rung"] = {
        "wall_s": time.monotonic() - started,
        "manifest": manifest_path,
        "resolved_rung": contract["rung"],
        "rows": int(manifest["rows"]),
        "k": int(manifest["k"]),
        "directed_edges": directed_edges,
        "substrate": substrate,
        "contract": contract,
    }
    poll("R0259 rung resolved and the substrate dimension checked")

    # ---- S1: the loader branch --------------------------------------------- #
    graph_directory = os.path.dirname(manifest_path)
    started = time.monotonic()
    sources, targets, weights, n_nodes = load_edge_arrays(graph_directory)
    open_wall = time.monotonic() - started
    if int(n_nodes) != contract["rows"] or int(sources.shape[0]) != directed_edges:
        raise Round0259NodeError(
            f"R0259: the branch returned n_nodes={n_nodes}, edges="
            f"{int(sources.shape[0])}; the manifest says {contract['rows']} and "
            f"{directed_edges}"
        )
    stages["load"] = {
        "wall_s": open_wall,
        "over_the_ceiling": over_the_ceiling(open_wall),
        "signature": rung.streamed_member_signature(graph_directory),
        "residency": {
            name: endpoint_residency(array, label=name)
            for name, array in (
                ("sources", sources), ("targets", targets), ("weights", weights)
            )
        },
        "note": (
            "the branch memmaps; it materialises nothing. R0258's polled resident "
            "load is the other registered route and is exercised below on the "
            "targets member."
        ),
    }
    anonymous["after_load"] = host_anonymous_bytes()["anonymous_bytes"]
    poll("R0259 streamed members opened")

    # ---- S2: polled endpoint bounds ---------------------------------------- #
    bounds: dict[str, Any] = {}
    bounds_walls: dict[str, float] = {}
    for name, array in (("sources", sources), ("targets", targets)):
        started = time.monotonic()
        low, high = endpoint_bounds_polled(
            array, poll=poll, chunk=SCAN_CHUNK_ELEMENTS
        )
        bounds_walls[name] = time.monotonic() - started
        bounds[name] = {"min": int(low), "max": int(high)}
        if low < 0 or high >= contract["rows"]:
            raise Round0259NodeError(
                f"R0259: {name} endpoints [{low}, {high}] leave [0, "
                f"{contract['rows']})"
            )
    stages["bounds"] = {
        "wall_s": float(sum(bounds_walls.values())),
        "per_array_wall_s": dict(sorted(bounds_walls.items())),
        "widest_single_array_s": max(bounds_walls.values()),
        "widest_over_the_ceiling": over_the_ceiling(max(bounds_walls.values())),
        "observed": bounds,
    }
    poll("R0259 endpoint bounds complete")

    # ---- S3: polled weight validity ---------------------------------------- #
    started = time.monotonic()
    finite, non_positive, above_one = weight_validity_polled(
        weights, poll=poll, chunk=SCAN_CHUNK_ELEMENTS
    )
    validity_wall = time.monotonic() - started
    if not finite or non_positive:
        raise Round0259NodeError(
            "R0259: the weights are not samplable (finite="
            f"{finite}, any<=0={non_positive})"
        )
    stages["validity"] = {
        "wall_s": validity_wall,
        "over_the_ceiling": over_the_ceiling(validity_wall),
        "all_finite": bool(finite),
        "any_non_positive": bool(non_positive),
        "any_above_one": bool(above_one),
    }
    poll("R0259 weight validity complete")

    # ---- S4: materialise the endpoints ------------------------------------- #
    layout = rung.streamed_member_layout(graph_directory)
    started = time.monotonic()
    resident_sources, resident_targets, materialisation = materialise_endpoints(
        sources, targets,
        poll=poll,
        chunk=SCAN_CHUNK_ELEMENTS,
        source_route="contiguous_polled",
        target_route="load_polled",
        target_path=layout["members"]["targets"],
    )
    stages["materialise"] = dict(materialisation)
    stages["materialise"]["wall_s"] = time.monotonic() - started
    stages["materialise"]["over_the_ceiling"] = over_the_ceiling(
        stages["materialise"]["wall_s"]
    )
    anonymous["after_materialise"] = host_anonymous_bytes()["anonymous_bytes"]
    poll("R0259 endpoints materialised")

    # ---- S5: the fully polled CDF, residue removed -------------------------- #
    started = time.monotonic()
    cdf, total, cdf_timings = weight_cdf_fully_polled(
        weights,
        poll=poll,
        chunk=SCAN_CHUNK_ELEMENTS,
        leaf=1 << 20,
        sum_proof_leaves=SUM_PROOF_LEAVES,
        sum_proof_controls=True,
    )
    cdf_wall = time.monotonic() - started
    proof = cdf_timings.get("sum_proof") or {}
    if not proof.get("every_leaf_bitwise_identical"):
        raise Round0259NodeError(
            "R0259: the chunked pairwise sum is not bitwise identical at the real "
            "rung; the residue is not removable on this numpy"
        )
    stages["cdf"] = {
        "wall_s": cdf_wall,
        "over_the_ceiling": over_the_ceiling(cdf_wall),
        "timings": cdf_timings,
        "float64_bytes": int(cdf.nbytes),
        "weight_total": float(total),
        "cdf_first": float(cdf[0]),
        "cdf_last": float(cdf[-1]),
        "the_named_unpollable_residue_is_gone": {
            "what_r0258_sealed": (
                "np.sum uses pairwise summation ... it is therefore left as ONE "
                "call ... This is the only stage interval in this module that "
                "polling cannot reduce."
            ),
            "one_np_sum_s": proof.get("one_np_sum_s"),
            "chunked_sum_s": cdf_timings.get("sum_s"),
            "chunked_sum_poll_sites": cdf_timings.get("sum_leaves"),
            "by_leaf": proof.get("by_leaf"),
            "controls_at_the_real_rung": proof.get("controls_at_the_real_rung"),
        },
    }
    anonymous["after_cdf"] = host_anonymous_bytes()["anonymous_bytes"]
    poll("R0259 CDF complete, residue chunked")

    # ---- S6: the per-batch probe ------------------------------------------- #
    probe = _per_batch_probe(
        poll=poll,
        torch=torch,
        cdf=cdf,
        resident_sources=resident_sources,
        resident_targets=resident_targets,
        member_sources=sources,
        member_targets=targets,
        n_nodes=int(n_nodes),
        directed_edges=directed_edges,
        resident_batches=resident_batches,
        file_backed_batches=file_backed_batches,
        substrate_batches=substrate_batches,
    )
    stages["probe"] = probe
    anonymous["after_probe"] = host_anonymous_bytes()["anonymous_bytes"]
    poll("R0259 per-batch probe complete")

    device_resident_x_bytes = int(contract["rows"]) * int(contract["dimension"]) * 2
    card_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    return {
        "rows": int(contract["rows"]),
        "k": int(contract["k"]),
        "dimension": int(contract["dimension"]),
        "directed_edges": directed_edges,
        "graph_manifest": manifest_path,
        "the_feature_gather_blocker": {
            "schema": "round0259-device-resident-x-blocker-v1",
            "rows": int(contract["rows"]),
            "dimension": int(contract["dimension"]),
            "fp16_bytes_required": device_resident_x_bytes,
            "device_total_memory_bytes": card_bytes,
            "fits": device_resident_x_bytes <= card_bytes,
            "shortfall_bytes": device_resident_x_bytes - card_bytes,
            "ratio": device_resident_x_bytes / card_bytes,
            "note": WHY_THERE_IS_NO_FIT_HERE,
        },
    }


def _per_batch_probe(
    *,
    poll: Any,
    torch: Any,
    cdf: np.ndarray,
    resident_sources: np.ndarray,
    resident_targets: np.ndarray,
    member_sources: Any,
    member_targets: Any,
    n_nodes: int,
    directed_edges: int,
    resident_batches: int,
    file_backed_batches: int,
    substrate_batches: int,
) -> dict[str, Any]:
    """`HostStreamEdgeSampler`'s producer, batch by batch, at the real rung.

    Replicates `_draw_positive_indices` (weighted inverse-CDF `searchsorted`) and
    `_produce`'s gather / cast / pin, then the consumer half `__next__` performs
    on device. The one component NOT measured is the feature gather, for the
    reason in `WHY_THERE_IS_NO_FIT_HERE`.
    """
    num_pos = max(1, int(BATCH_SIZE * POS_RATIO))
    num_neg = BATCH_SIZE - num_pos
    rng = np.random.default_rng(20260812)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260812)

    def _draw(count: int) -> np.ndarray:
        proposal = np.searchsorted(cdf, rng.random(count), side="right")
        np.clip(proposal, 0, directed_edges - 1, out=proposal)
        return proposal

    def _producer_batch(src: Any, dst: Any) -> tuple[float, dict[str, float]]:
        started = time.monotonic()
        draw_started = time.monotonic()
        idx = _draw(num_pos)
        draw_s = time.monotonic() - draw_started
        gather_started = time.monotonic()
        ps = torch.from_numpy(src[idx].astype(np.int64))
        pd = torch.from_numpy(dst[idx].astype(np.int64))
        gather_s = time.monotonic() - gather_started
        pin_started = time.monotonic()
        ps = ps.pin_memory()
        pd = pd.pin_memory()
        pin_s = time.monotonic() - pin_started
        return time.monotonic() - started, {
            "draw_s": draw_s, "gather_s": gather_s, "pin_s": pin_s,
            "_tensors": (ps, pd),
        }

    def _consumer_batch(ps: Any, pd: Any) -> float:
        started = time.monotonic()
        p_src = ps.to("cuda", non_blocking=True)
        p_dst = pd.to("cuda", non_blocking=True)
        neg_src = torch.randint(0, n_nodes, (num_neg,), generator=generator,
                                device="cuda")
        offset = torch.randint(1, n_nodes, (num_neg,), generator=generator,
                               device="cuda")
        neg_dst = (neg_src + offset) % n_nodes
        all_src = torch.cat([p_src, neg_src])
        all_dst = torch.cat([p_dst, neg_dst])
        torch.cuda.synchronize()
        wall = time.monotonic() - started
        del p_src, p_dst, neg_src, neg_dst, offset, all_src, all_dst
        return wall

    # --- resident arm ------------------------------------------------------- #
    assert_resident(resident_sources, label="resident sources")
    assert_resident(resident_targets, label="resident targets")
    resident_producer: list[float] = []
    resident_draw: list[float] = []
    resident_gather: list[float] = []
    consumer: list[float] = []
    end_to_end: list[float] = []
    for index in range(resident_batches):
        wall, parts = _producer_batch(resident_sources, resident_targets)
        resident_producer.append(wall)
        resident_draw.append(parts["draw_s"])
        resident_gather.append(parts["gather_s"])
        ps, pd = parts["_tensors"]
        consumer_wall = _consumer_batch(ps, pd)
        consumer.append(consumer_wall)
        end_to_end.append(wall + consumer_wall)
        del ps, pd
        if index % 100 == 0:
            poll(f"R0259 resident producer batch {index}")
    poll("R0259 resident arm complete")

    # --- file-backed contrast (review-0258-01 §H.2's condition) -------------- #
    file_producer: list[float] = []
    file_gather: list[float] = []
    file_residency = {
        "sources": endpoint_residency(member_sources, label="member sources"),
        "targets": endpoint_residency(member_targets, label="member targets"),
    }
    for index in range(file_backed_batches):
        wall, parts = _producer_batch(member_sources, member_targets)
        file_producer.append(wall)
        file_gather.append(parts["gather_s"])
        del parts
        if index % 20 == 0:
            poll(f"R0259 file-backed producer batch {index}")
    poll("R0259 file-backed arm complete")

    # --- the host-side feature gather --------------------------------------- #
    substrate = np.load(SUBSTRATE_100M_PATH, mmap_mode="r", allow_pickle=False)
    if not isinstance(substrate, np.memmap) or substrate.flags.writeable:
        raise Round0259NodeError("R0259: the substrate is not a read-only memmap")
    substrate_gather: list[float] = []
    for index in range(substrate_batches):
        rows = rng.integers(0, n_nodes, size=BATCH_SIZE)
        started = time.monotonic()
        block = np.asarray(substrate[rows], dtype=np.float32)
        substrate_gather.append(time.monotonic() - started)
        if int(block.shape[0]) != BATCH_SIZE:
            raise Round0259NodeError("R0259: the substrate gather returned a short block")
        del block
        if index % 20 == 0:
            poll(f"R0259 substrate gather batch {index}")
    del substrate
    poll("R0259 substrate gather arm complete")

    resident_gather_p50 = sorted(resident_gather)[len(resident_gather) // 2]
    file_gather_p50 = sorted(file_gather)[len(file_gather) // 2]
    return {
        "schema": "round0259-per-batch-probe-v1",
        "num_pos_per_batch": num_pos,
        "num_neg_per_batch": num_neg,
        "resident": {
            "producer_total": _interval_block(resident_producer, label="producer"),
            "cdf_searchsorted": _interval_block(resident_draw, label="draw"),
            "endpoint_gather": _interval_block(resident_gather, label="gather"),
            "consumer_h2d_and_negatives": _interval_block(
                consumer, label="consumer"),
            "end_to_end_minus_the_feature_gather": _interval_block(
                end_to_end, label="end_to_end"),
        },
        "file_backed": {
            "producer_total": _interval_block(file_producer, label="producer"),
            "endpoint_gather": _interval_block(file_gather, label="gather"),
            "residency": file_residency,
        },
        "host_substrate_feature_gather": _interval_block(
            substrate_gather, label="substrate_gather"),
        "file_backed_over_resident_gather_p50": (
            file_gather_p50 / resident_gather_p50 if resident_gather_p50 else None
        ),
        "what_is_not_measured": (
            "the device-side feature gather. DeviceArrayDataset.index_select "
            "requires X resident on the card and X does not fit at this rung; "
            "the host-side substrate gather above is the alternative a "
            "host-streamed X path would pay, measured, and it is NOT the same "
            "operation."
        ),
    }


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0259 round0259_nodes.run_job")
    action = str(job.get("action") or "")
    if action == CONTROLS_ACTION:
        run_controls(active, job)
    elif action == TRAIN_ACTION:
        run_train_100m(active, job)
    else:
        raise Round0259NodeError(f"R0259 does not know action {action!r}")
