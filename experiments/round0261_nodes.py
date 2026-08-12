"""Execute R0261 — seal the price prediction, then build the 4M exact k15 graph.

Two nodes.

`predict_0261` (CPU, no CUDA)
    Seals the pre-registered price of the 4M exact search as four named models
    over two sealed prior walls, re-read from R0216's and R0233's artifacts on
    disk rather than trusted as literals, plus the round's other falsifiable
    predictions (substrate bytes, directed edges, npz bytes, peak device bytes,
    fuzzy wall, node wall). Then runs every positive control: nineteen planted
    defects across six shipped guards, including BOTH refusal branches of the
    ordering guard that review-0260-01 §K asked for by name.

`build_0261` (GPU)
    Refuses to start until that seal exists and predates it, then assembles the
    4,000,000-row substrate on R0216's span-sampling law, builds the exact
    brute-force fp32 cosine k15 graph at R0216's block sizes with the GPU search
    and the per-row postprocess timed SEPARATELY, scores it against an
    INDEPENDENT plain-NumPy CPU probe, symmetrises it with
    `umap.fuzzy_simplicial_set`, runs the R0215 degree-zero tripwire, and then
    fits `t = a*N^2 + b*N` from its own decomposition and back-checks that law
    against R0216's sealed 2M wall.

Safety. Every corpus shard is opened read-only (`np.load(mmap_mode="r")` or a
read-only `np.memmap`). No process is signalled, no child process is started,
cuVS is handed nothing, no subprocess is wrapped in a timeout. The host guard
gates on ANONYMOUS bytes, never RSS; the build node allocates no `MAP_SHARED`
region, so the R0259 blind spot does not apply here and the receipt says so.
The build node holds a live CUDA context for most of its wall on purpose:
`/proc/<pid>/maps` matches `libcuda`, so it must never be signalled.

The `AbortPollGate` class is under an owner stop and this module does not import
or touch it. Stop-latency evidence is `PollRecorder` + `gap_report` +
`CoverageLedger`, and both nodes publish `observed_span_s` at their top level.
"""
from __future__ import annotations

import glob
import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy, atomic_save_new_npz, atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0238_rung5 import json_safe
from basemap.round0242_locality import host_anonymous_bytes, json_scrub
from basemap.round0245_guard import EnforcedHostWatchdog
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report
from basemap.round0253_coverage import CoverageLedger, coverage_summary
from basemap.round0253_stop_hooks import install_stop_hooks, registered_ceiling_s
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0261_four_m_graph import (
    CAPABILITY, COMPOSITION, CPU_PROBE_BLOCK, CPU_PROBE_ROWS, CPU_PROBE_SEED,
    DEVICE_BUDGET_BYTES, DIMENSION, EXCLUDED_SHARDS, GPU_PROBE_IS_INDEPENDENT,
    GPU_PROBE_ROWS, GPU_PROBE_SEED, GRAPH_K, GRAPH_SCHEMA,
    MIN_MEM_AVAILABLE_BYTES, NESTED_IN_R0216, NODE_ANON_BUDGET_BYTES,
    PREDICTION_SCHEMA, PRICE_CAPABILITY, PRICE_SCHEMA, QUERY_BLOCK,
    R0216_EXACT_SEARCH_S, R0216_ROWS, R0233_EXACT_SEARCH_S, R0233_ROWS,
    RAW_FORMAT, ROUND_ID, ROWS, ROW_POLICY, SEARCH_BLOCK, SELECTION_SEED,
    SUBSTRATE_SCHEMA, TRAILING_FRAGMENT_POLICY, ZERO_ROW_POLICY,
    Round0261Error, all_controls, assert_prediction_precedes_build,
    assert_shard_span, back_check_at_2m, cost_prediction, cpu_exact_topk,
    degree_census, gating_recall_block, other_predictions, price_other_rungs,
    resolve_shard_rows, score_cpu_probe, score_prediction, validate_composition,
    validate_exact_graph,
)
from experiments.round0238_nodes import _check_runner_abort
from experiments.round0253_nodes import _cuda_maps_evidence, force_remove_tree
from experiments.round0251_nodes import _start_node

EMB = "/data/embeddings"

PREDICT_ACTION = "round0261_seal_price_prediction_and_controls"
BUILD_ACTION = "round0261_build_4m_substrate_and_exact_graph"

PREDICT_CAPABILITY = "round0261-four-m-exact-graph-price-prediction-and-controls-v1"
PREDICT_SCHEMA = "round0261-four-m-price-prediction-and-controls-v1"

#: Refuse to start the build without this much free space on /data: the
#: predicted `6,144,000,128 + 44,000,192 + ~1,160,000,000` B artifact plus room
#: for the `.tmp` copy `atomic_save_new_npz` writes before renaming.
MIN_DATA_FREE_BYTES = 40 * (1 << 30)

SAFETY_NOTE = (
    "no node in this module signals any process, starts a child process, hands "
    "cuVS anything, or wraps a subprocess in a timeout. Every corpus shard is "
    "opened read-only. The host guard gates on ANONYMOUS bytes, never RSS. This "
    "module allocates no MAP_SHARED region, so the R0259 blind spot (shared "
    "mappings are invisible to an anonymous-bytes guard) does not apply; the "
    "build node's whole footprint is anonymous or read-only file-backed. The "
    "build node maps libcuda and must never be signalled."
)


class Round0261NodeError(RuntimeError):
    """R0261 fails closed."""


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "safety_note": SAFETY_NOTE,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": False,
        "map_shared_allocations": 0,
        "registered_ceiling_s": registered_ceiling_s(),
    }


def _node_guard(label: str, *, interval_s: float = 0.05) -> EnforcedHostWatchdog:
    return EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        interval_s=float(interval_s), label=label,
    )


def _seal(output: str, name: str, body: Mapping[str, Any]) -> str:
    path = os.path.join(output, name)
    atomic_write_new_json(
        path, prompt_contract.seal(json_scrub(json_safe(dict(body)))), immutable=True)
    return path


def _mem_available_bytes() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise Round0261NodeError("R0261 could not read MemAvailable")


def _data_free_bytes(path: str = "/data") -> int:
    stat = os.statvfs(path)
    return int(stat.f_bavail) * int(stat.f_frsize)


def _require_headroom(*, label: str) -> dict[str, Any]:
    available = _mem_available_bytes()
    free = _data_free_bytes()
    if available < MIN_MEM_AVAILABLE_BYTES:
        raise Round0261NodeError(
            f"R0261 refuses to start {label}: MemAvailable {available} B is below "
            f"the required {MIN_MEM_AVAILABLE_BYTES} B."
        )
    if free < MIN_DATA_FREE_BYTES:
        raise Round0261NodeError(
            f"R0261 refuses to start {label}: /data has {free} B free, below the "
            f"required {MIN_DATA_FREE_BYTES} B."
        )
    return {"mem_available_bytes": available,
            "mem_available_required_bytes": MIN_MEM_AVAILABLE_BYTES,
            "data_free_bytes": free,
            "data_free_required_bytes": MIN_DATA_FREE_BYTES,
            "host_memory_at_entry": host_anonymous_bytes()}


def _read_sealed_number(signature: Any, *, keys: tuple[str, ...], label: str) -> float:
    """Re-read a sealed prior wall from its artifact instead of trusting a literal."""
    path = prompt_contract.verify_signature(signature, label=label)
    body = prompt_contract.read_sealed(path, label=label)
    node: Any = body
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            raise Round0261NodeError(f"{label}: {'.'.join(keys)} missing from {path}")
        node = node[key]
    return float(node)


# --------------------------------------------------------------------------- #
# node A -- the prediction and the controls
# --------------------------------------------------------------------------- #


def run_predict(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0261 round0261_nodes.run_predict")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0261NodeError("R0261 predict handler received another queue")
    node_id = str(active.get("node_id") or "predict_0261")
    label = "R0261 price prediction and positive controls"
    started = time.monotonic()
    abort_flag = _start_node(label)
    ledger = CoverageLedger(node=node_id)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0261 prediction")

    window = ledger.window("R0261 prediction stage")
    guard = _node_guard(label)
    with guard:
        recorder = PollRecorder(gate=_check_runner_abort, clock=time.monotonic)
        recorder.anchor("R0261 prediction stage entered")
        wrapped = window.wrap(recorder)

        # The two prior walls, re-read from their sealed artifacts. The module
        # holds them as literals so the arithmetic is readable; this proves the
        # literals are the artifacts.
        r0216_wall = _read_sealed_number(
            job["r0216_graph_receipt"], keys=("performance", "exact_search_s"),
            label="R0261 R0216 2M graph receipt")
        wrapped("R0261 re-read R0216's sealed 2M exact_search_s")
        r0233_wall = _read_sealed_number(
            job["r0233_truth_receipt"], keys=("performance", "exact_search_s"),
            label="R0261 R0233 6250k exact truth receipt")
        wrapped("R0261 re-read R0233's sealed 6250k exact_search_s")

        literals_match = {
            "r0216_exact_search_s_on_disk": r0216_wall,
            "r0216_exact_search_s_literal": R0216_EXACT_SEARCH_S,
            "r0216_identical": r0216_wall == R0216_EXACT_SEARCH_S,
            "r0233_exact_search_s_on_disk": r0233_wall,
            "r0233_exact_search_s_literal": R0233_EXACT_SEARCH_S,
            "r0233_identical": r0233_wall == R0233_EXACT_SEARCH_S,
        }
        if not (literals_match["r0216_identical"] and literals_match["r0233_identical"]):
            raise Round0261NodeError(
                f"R0261 prior walls moved under the module literals: {literals_match}"
            )
        wrapped("R0261 both prior walls match their module literals bitwise")

        prediction = cost_prediction()
        wrapped("R0261 four cost models evaluated")
        others = other_predictions()
        wrapped("R0261 non-wall predictions evaluated")
        controls = all_controls()
        wrapped("R0261 nineteen planted defects all refused by the shipped guards")

    window.close()
    coverage = ledger.receipt(node_wall_s=time.monotonic() - started)
    report = gap_report(recorder.records, arm="predict")

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": PREDICT_SCHEMA,
        "capability": PREDICT_CAPABILITY,
        "capabilities": [PREDICT_CAPABILITY],
        "node": node_id,
        "abort_flag_precondition": abort_flag,
        "sealed_at_unix": time.time(),
        "rows": ROWS,
        "cost_prediction": prediction,
        "other_predictions": others,
        "prior_walls_reread_from_disk": literals_match,
        "positive_controls": controls,
        "gpu_used": False,
        "cuda_presence": _cuda_maps_evidence(),
        "training_performed": False,
        "gap_report": report,
        "host_watchdog": guard.receipt(),
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "what_this_node_does_not_do": (
            "it reads no 4,000,000-row measurement, because none exists when it "
            "runs. Its only inputs are two sealed prior walls and the round "
            "file. The build node refuses to start unless this artifact's "
            "sealed_at_unix strictly precedes its own start."
        ),
    })
    _seal(output, f"{node_id}-price-prediction.json", body)


# --------------------------------------------------------------------------- #
# node B -- the build
# --------------------------------------------------------------------------- #


def _verify_sizes(job: Mapping[str, Any]) -> dict[str, Any]:
    signature = job.get("source_size_manifest")
    if signature is None:
        raise Round0261NodeError("R0261 requires the bound source size manifest")
    path = prompt_contract.verify_signature(signature, label="R0261 source manifest")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    drift = []
    checked = 0
    for corpus, entry in (manifest.get("corpora") or {}).items():
        for rel, want in (entry.get("shard_sizes") or {}).items():
            actual = os.path.getsize(os.path.join(EMB, rel))
            checked += 1
            if actual != int(want):
                drift.append(f"{rel}: {actual} != {want}")
    if drift:
        raise Round0261NodeError(
            f"R0261 source shards changed size since preparation: {drift[:4]}")
    return {"shards_size_checked": checked, "manifest": path}


def _shards(corpus: str) -> list[tuple[str, int, bool]]:
    out = []
    for path in sorted(glob.glob(os.path.join(EMB, corpus, "train", "*.npy"))):
        if path.endswith(".tmp.npy"):
            continue
        if os.path.relpath(path, EMB) in EXCLUDED_SHARDS:
            continue
        with open(path, "rb") as handle:
            real_npy = handle.read(6) == b"\x93NUMPY"
        if real_npy:
            rows = int(np.load(path, mmap_mode="r").shape[0])
        else:
            rows = resolve_shard_rows(relative_path=os.path.relpath(path, EMB),
                                      size_bytes=os.path.getsize(path))
        out.append((path, rows, real_npy))
    if not out:
        raise Round0261NodeError(f"R0261 found no shards for {corpus}")
    return out


def _open(path: str, rows: int, real_npy: bool) -> np.ndarray:
    if real_npy:
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype="<f4", mode="r", shape=(rows, DIMENSION))


def _assemble(poll: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """R0216's span-sampling law at 4,000,000 rows. Numerically unchanged."""
    X = np.empty((ROWS, DIMENSION), dtype=np.float32)
    provenance = np.empty(ROWS, dtype=np.dtype([
        ("corpus", "u1"), ("shard", "u2"), ("row", "i8")]))
    counts: dict[str, int] = {}
    rejects: dict[str, int] = {}
    spans: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    at = 0
    for ci, (corpus, want) in enumerate(COMPOSITION):
        shards = _shards(corpus)
        total = sum(r for _p, r, _n in shards)
        if total < want:
            raise Round0261NodeError(f"{corpus}: need {want} rows, corpus has {total}")
        rng = np.random.RandomState(SELECTION_SEED + ci)
        offs_all = np.concatenate([[0], np.cumsum([r for _p, r, _n in shards])])
        chosen: list[np.ndarray] = []
        chosen_vecs: list[np.ndarray] = []
        picked = np.zeros(total, dtype=bool)
        need = want
        dropped = 0
        rounds = 0
        while need > 0:
            rounds += 1
            if rounds > 8:
                raise Round0261NodeError(
                    f"{corpus}: replacement did not converge after 8 rounds")
            free = np.flatnonzero(~picked)
            if free.size < need:
                raise Round0261NodeError(f"{corpus}: exhausted usable rows")
            draw = np.sort(rng.choice(free, need, replace=False)).astype(np.int64)
            picked[draw] = True
            shard_of = np.searchsorted(offs_all, draw, side="right") - 1
            for si, (path, rows, real_npy) in enumerate(shards):
                local = draw[shard_of == si] - offs_all[si]
                if local.size == 0:
                    continue
                arr = _open(path, rows, real_npy)
                block = np.asarray(arr[local], dtype=np.float32)
                norm = np.linalg.norm(block, axis=1)
                ok = np.isfinite(block).all(axis=1) & (norm > 0)
                dropped += int((~ok).sum())
                if ok.any():
                    chosen.append(draw[shard_of == si][ok])
                    chosen_vecs.append(block[ok])
                del arr, block
                poll(f"R0261 assembly {corpus} shard {si}")
            got = sum(len(c) for c in chosen)
            need = want - got
        order = np.argsort(np.concatenate(chosen))
        sel = np.concatenate(chosen)[order]
        vecs = np.concatenate(chosen_vecs, axis=0)[order]
        shard_of = np.searchsorted(offs_all, sel, side="right") - 1
        X[at:at + want] = vecs
        provenance["corpus"][at:at + want] = ci
        provenance["shard"][at:at + want] = shard_of
        provenance["row"][at:at + want] = sel - offs_all[shard_of]
        at += want
        span = assert_shard_span(corpus=corpus,
                                 shards_touched=int(len(np.unique(shard_of))),
                                 shards_total=len(shards))
        span["replacement_rounds"] = rounds
        del chosen, chosen_vecs, vecs, picked, free, draw
        counts[corpus] = want
        rejects[corpus] = dropped
        spans[corpus] = span
        sources[corpus] = {
            "shards": len(shards), "corpus_rows": int(total),
            "selected_rows": int(want),
            "format": "npy" if shards[0][2] else RAW_FORMAT,
            "first_shard": expected_input_signature(shards[0][0]),
        }
        poll(f"R0261 assembled {corpus}")
    if at != ROWS:
        raise Round0261NodeError(f"assembled {at} rows, expected {ROWS}")
    composition = validate_composition(counts)
    norms = np.linalg.norm(X, axis=1)
    if not np.isfinite(X).all() or float(norms.min()) <= 0:
        raise Round0261NodeError("substrate contains nonfinite or zero rows")
    X /= norms[:, None]
    meta = {"composition": composition, "degenerate_rows_dropped": rejects,
            "shard_span": spans, "sources": sources}
    return X, provenance, meta


def _exact_search(X: np.ndarray, torch: Any, device: Any, poll: Any
                  ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """R0216's builder, unchanged in arithmetic, with the two costs timed apart.

    `gpu_search_s` is the blocked GEMM + top-k merge over every (query,
    candidate) pair -- the quadratic term. `postprocess_s` is the per-row Python
    loop that drops the self-match and fills `nbr`/`dist` -- the linear term.
    The split point is after `.cpu().numpy()`, which is where the device work
    for a query block is forced to complete, so no GPU time is charged to the
    postprocess and none of the postprocess is charged to the GPU.
    """
    Xt = torch.from_numpy(X).to(device)
    nbr = np.empty((ROWS, GRAPH_K), dtype=np.int32)
    dist = np.empty((ROWS, GRAPH_K), dtype=np.float32)
    gpu_s = 0.0
    post_s = 0.0
    for qs in range(0, ROWS, QUERY_BLOCK):
        qe = min(qs + QUERY_BLOCK, ROWS)
        t0 = time.monotonic()
        q = Xt[qs:qe]
        best_s = torch.full((qe - qs, GRAPH_K + 1), -float("inf"), device=device)
        best_i = torch.full((qe - qs, GRAPH_K + 1), -1, device=device, dtype=torch.int64)
        for cs in range(0, ROWS, SEARCH_BLOCK):
            ce = min(cs + SEARCH_BLOCK, ROWS)
            sims = q @ Xt[cs:ce].T
            k = min(GRAPH_K + 1, ce - cs)
            ts, ti = torch.topk(sims, k, dim=1)
            cs_ = torch.cat([best_s, ts], 1)
            ci_ = torch.cat([best_i, ti.to(torch.int64) + cs], 1)
            order = torch.argsort(cs_, dim=1, descending=True)[:, : GRAPH_K + 1]
            best_s = torch.gather(cs_, 1, order)
            best_i = torch.gather(ci_, 1, order)
            del sims, ts, ti, cs_, ci_, order
            poll(f"R0261 exact search candidate block {qs}:{cs}")
        ids = best_i.cpu().numpy()
        sims_np = best_s.cpu().numpy()
        t1 = time.monotonic()
        gpu_s += t1 - t0
        for n in range(qe - qs):
            row = qs + n
            keep = [(int(i), float(s)) for i, s in zip(ids[n], sims_np[n])
                    if int(i) != row][:GRAPH_K]
            if len(keep) < GRAPH_K:
                raise Round0261NodeError(f"row {row} got {len(keep)} neighbours")
            nbr[row] = [i for i, _ in keep]
            dist[row] = [1.0 - s for _, s in keep]
        post_s += time.monotonic() - t1
        poll(f"R0261 exact search query block {qs}")
    del Xt
    torch.cuda.empty_cache()
    return nbr, dist, {"gpu_search_s": float(gpu_s), "postprocess_s": float(post_s),
                       "exact_search_s": float(gpu_s + post_s)}


def _gpu_probe(X: np.ndarray, nbr: np.ndarray, torch: Any, device: Any, poll: Any
               ) -> dict[str, Any]:
    """R0216's own probe, kept for continuity. It never gates -- see the module docstring."""
    rng = np.random.RandomState(GPU_PROBE_SEED)
    probe = np.sort(rng.choice(ROWS, GPU_PROBE_ROWS, replace=False))
    Xt = torch.from_numpy(X).to(device)
    qp = Xt[torch.from_numpy(probe).to(device)]
    truth_s = torch.full((len(probe), GRAPH_K + 1), -float("inf"), device=device)
    truth_i = torch.full((len(probe), GRAPH_K + 1), -1, device=device, dtype=torch.int64)
    for cs in range(0, ROWS, SEARCH_BLOCK):
        ce = min(cs + SEARCH_BLOCK, ROWS)
        sims = qp @ Xt[cs:ce].T
        k = min(GRAPH_K + 1, ce - cs)
        ts, ti = torch.topk(sims, k, dim=1)
        cs_ = torch.cat([truth_s, ts], 1)
        ci_ = torch.cat([truth_i, ti.to(torch.int64) + cs], 1)
        order = torch.argsort(cs_, dim=1, descending=True)[:, : GRAPH_K + 1]
        truth_s = torch.gather(cs_, 1, order)
        truth_i = torch.gather(ci_, 1, order)
        del sims, ts, ti, cs_, ci_, order
        poll(f"R0261 GPU probe block {cs}")
    ti_np = truth_i.cpu().numpy()
    recalls = []
    for n, row in enumerate(probe):
        truth = [int(i) for i in ti_np[n] if int(i) != int(row)][:GRAPH_K]
        recalls.append(len(set(truth) & set(nbr[row].tolist())) / GRAPH_K)
    del Xt, qp, truth_s, truth_i
    torch.cuda.empty_cache()
    values = np.asarray(recalls, dtype=np.float64)
    return {
        "probe_rows": int(values.size),
        "seed": GPU_PROBE_SEED,
        "mean_recall_at_k": float(values.mean()),
        "p10_recall_at_k": float(np.percentile(values, 10)),
        "min_recall_at_k": float(values.min()),
        "is_independent": GPU_PROBE_IS_INDEPENDENT,
        "why_it_does_not_gate": (
            "it re-runs the builder's own kernel, block geometry and merge "
            "order over the same device array, so it shares the builder's "
            "accumulator. review-0216-01 established that such a probe cannot "
            "establish exactness; the standing rule requires a separate pass, "
            "ideally a plain CPU one, computed by someone other than the builder."
        ),
    }


def run_build(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    install_stop_hooks(label="R0261 round0261_nodes.run_build")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0261NodeError("R0261 build handler received another queue")
    node_id = str(active.get("node_id") or "build_0261")
    label = "R0261 4M substrate and exact k15 graph"
    started_monotonic = time.monotonic()
    started_unix = time.time()
    abort_flag = _start_node(label)
    headroom = _require_headroom(label=label)
    ledger = CoverageLedger(node=node_id)

    # Produced by `predict_0261` inside this same queue, so it cannot carry a
    # prepare-time signature. `read_sealed` validates its identity digest, and
    # `assert_prediction_precedes_build` below is what makes it a prediction.
    prediction_path = str(job["price_prediction_path"])
    prediction_body = prompt_contract.read_sealed(
        prediction_path, label="R0261 price prediction")
    if str(prediction_body.get("schema")) != PREDICT_SCHEMA:
        raise Round0261NodeError(
            f"R0261 build was handed {prediction_body.get('schema')!r}, not the "
            f"price prediction {PREDICT_SCHEMA!r}")
    ordering = assert_prediction_precedes_build(
        prediction=prediction_body, build_started_unix=started_unix)
    registered = prediction_body["cost_prediction"]
    if str(registered.get("schema")) != PREDICTION_SCHEMA:
        raise Round0261NodeError("R0261 price prediction carries no cost model")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0261 4M graph")
    window = ledger.window("R0261 build stage")
    guard = _node_guard(label)
    sealed = False
    try:
        with guard:
            recorder = PollRecorder(gate=_check_runner_abort, clock=time.monotonic)
            recorder.anchor("R0261 build stage entered")
            poll = window.wrap(recorder)

            sizes = _verify_sizes(job)
            poll("R0261 source shard sizes re-verified")

            assemble_t0 = time.monotonic()
            X, provenance, meta = _assemble(poll)
            assemble_s = time.monotonic() - assemble_t0
            poll("R0261 4,000,000-row substrate assembled and normalised")

            write_t0 = time.monotonic()
            sub_path = atomic_save_new_npy(
                os.path.join(output, "substrate.f32.npy"), X, immutable=True)
            prov_path = atomic_save_new_npy(
                os.path.join(output, "provenance.npy"), provenance, immutable=True)
            substrate_write_s = time.monotonic() - write_t0
            poll("R0261 substrate and provenance written")

            device = torch.device("cuda")
            nbr, dist, timing = _exact_search(X, torch, device, poll)
            poll("R0261 exact brute-force k15 search complete")

            gpu_probe_t0 = time.monotonic()
            gpu_probe = _gpu_probe(X, nbr, torch, device, poll)
            gpu_probe_s = time.monotonic() - gpu_probe_t0
            poll("R0261 builder GPU probe complete (non-gating)")

            cpu_t0 = time.monotonic()
            rng = np.random.RandomState(CPU_PROBE_SEED)
            cpu_rows = np.sort(rng.choice(ROWS, CPU_PROBE_ROWS, replace=False))
            truth_ids, truth_cos = cpu_exact_topk(
                X, cpu_rows, k=GRAPH_K, block=CPU_PROBE_BLOCK, poll=poll)
            builder_ids = nbr[cpu_rows].astype(np.int64)
            builder_cos = (1.0 - dist[cpu_rows]).astype(np.float64)
            cpu_probe = score_cpu_probe(truth_ids=truth_ids, truth_cos=truth_cos,
                                        builder_ids=builder_ids,
                                        builder_cos=builder_cos, k=GRAPH_K)
            cpu_probe["seed"] = CPU_PROBE_SEED
            cpu_probe["block"] = CPU_PROBE_BLOCK
            cpu_probe["overlap_rows_with_the_gpu_probe"] = int(
                np.intersect1d(cpu_rows, np.sort(np.random.RandomState(
                    GPU_PROBE_SEED).choice(ROWS, GPU_PROBE_ROWS, replace=False))).size)
            cpu_probe_s = time.monotonic() - cpu_t0
            poll("R0261 independent CPU probe complete (gating)")

            import umap.umap_ as umap_api
            fuzzy_t0 = time.monotonic()
            graph, _sigmas, _rhos = umap_api.fuzzy_simplicial_set(
                X, n_neighbors=GRAPH_K, random_state=np.random.RandomState(42),
                metric="cosine", knn_indices=nbr, knn_dists=dist,
            )
            coo = graph.tocoo()
            src = np.asarray(coo.row, dtype=np.int32)
            dst = np.asarray(coo.col, dtype=np.int32)
            wts = np.asarray(coo.data, dtype=np.float32)
            fuzzy_s = time.monotonic() - fuzzy_t0
            poll("R0261 fuzzy simplicial set built")

            if not np.isfinite(wts).all() or wts.min() <= 0 or wts.max() > 1:
                raise Round0261NodeError("fuzzy weights are invalid")
            degrees = degree_census(src, rows=ROWS)
            checks = validate_exact_graph(
                degrees=degrees, gating_recall=gating_recall_block(cpu_probe),
                builder_recall={"mean_recall_at_k": gpu_probe["mean_recall_at_k"],
                                "p10_recall_at_k": gpu_probe["p10_recall_at_k"]},
                edges=int(src.size))
            poll("R0261 degree-zero tripwire and recall floors asserted")

            edge_t0 = time.monotonic()
            graph_path = atomic_save_new_npz(
                os.path.join(output, "edges-k15-fuzzy.npz"), immutable=True,
                compressed=False, sources=src, targets=dst, weights=wts,
                n_nodes=np.asarray(ROWS, dtype=np.int64),
                k=np.asarray(GRAPH_K, dtype=np.int64))
            edge_write_s = time.monotonic() - edge_t0
            poll("R0261 fuzzy edge set written")

            substrate_sha = ordered_array_sha256(X)
            peak_bytes = int(torch.cuda.max_memory_allocated("cuda"))
            del X
            poll("R0261 substrate digest computed")

        window.close()
        node_wall_s = time.monotonic() - started_monotonic
        coverage = ledger.receipt(node_wall_s=node_wall_s)
        report = gap_report(recorder.records, arm="build")

        # ---- the measured price, and what it licenses ---- #
        a_measured = timing["gpu_search_s"] / (float(ROWS) ** 2)
        b_measured = timing["postprocess_s"] / float(ROWS)
        law = {"quadratic_s_per_pair": float(a_measured),
               "linear_s_per_row": float(b_measured),
               "fitted_from": "this node's own decomposition at 4,000,000 rows",
               "quadratic_share_of_exact_search": (
                   timing["gpu_search_s"] / timing["exact_search_s"]),
               "arithmetic": (
                   f"a = gpu_search_s / N^2 = {timing['gpu_search_s']} / "
                   f"{ROWS}^2 ; b = postprocess_s / N = "
                   f"{timing['postprocess_s']} / {ROWS}"),
               }
        back_check = back_check_at_2m(quadratic_s_per_pair=a_measured,
                                      linear_s_per_row=b_measured)
        scored = score_prediction(prediction=registered,
                                  measured_s=timing["exact_search_s"])
        rungs = price_other_rungs(quadratic_s_per_pair=a_measured,
                                  linear_s_per_row=b_measured)

        artifact_bytes = {
            "substrate.f32.npy": os.path.getsize(sub_path),
            "provenance.npy": os.path.getsize(prov_path),
            "edges-k15-fuzzy.npz": os.path.getsize(graph_path),
        }
        artifact_bytes["total"] = int(sum(artifact_bytes.values()))
        disk = {
            "artifact_bytes": artifact_bytes,
            "data_free_bytes_after": _data_free_bytes(),
            "data_free_bytes_before": headroom["data_free_bytes"],
            "artifact_as_fraction_of_free_before": (
                artifact_bytes["total"] / float(headroom["data_free_bytes"])),
        }

        body = dict(_receipt_envelope(active["manifest"]))
        body.update({
            "schema": GRAPH_SCHEMA,
            "price_schema": PRICE_SCHEMA,
            "substrate_schema": SUBSTRATE_SCHEMA,
            "capability": CAPABILITY,
            "capabilities": [CAPABILITY, PRICE_CAPABILITY],
            "node": node_id,
            "abort_flag_precondition": abort_flag,
            "headroom": headroom,
            "ordering": ordering,
            "price_prediction_artifact": expected_input_signature(prediction_path),
            "rows": ROWS, "dimension": DIMENSION, "k": GRAPH_K,
            "nested_in_r0216_2m": NESTED_IN_R0216,
            "composition": meta["composition"],
            "sources": meta["sources"],
            "source_size_check": sizes,
            "loading_contract": {"raw_format": RAW_FORMAT, "row_policy": ROW_POLICY,
                                 "trailing_fragment_policy": TRAILING_FRAGMENT_POLICY},
            "selection": {
                "seed": SELECTION_SEED,
                "law": ("per-corpus uniform over ALL complete rows of "
                        "non-excluded shards; rejected rows replaced by fresh "
                        "uniform draws from the unpicked complement until quota "
                        "is met; never a prefix"),
                "zero_row_policy": ZERO_ROW_POLICY,
                "degenerate_rows_dropped": meta["degenerate_rows_dropped"],
                "shard_span": meta["shard_span"],
                "excluded_shards": {k: v["reason"] for k, v in EXCLUDED_SHARDS.items()},
            },
            "substrate": expected_input_signature(sub_path),
            "provenance": expected_input_signature(prov_path),
            "ordered_substrate_sha256": substrate_sha,
            "graph": expected_input_signature(graph_path),
            "graph_checks": checks,
            "degrees": degrees,
            "gating_cpu_probe": cpu_probe,
            "builder_gpu_probe": gpu_probe,
            "performance": {
                "assemble_s": float(assemble_s),
                "substrate_write_s": float(substrate_write_s),
                "gpu_search_s": timing["gpu_search_s"],
                "postprocess_s": timing["postprocess_s"],
                "exact_search_s": timing["exact_search_s"],
                "gpu_probe_s": float(gpu_probe_s),
                "cpu_probe_s": float(cpu_probe_s),
                "fuzzy_s": float(fuzzy_s),
                "edge_write_s": float(edge_write_s),
                "node_wall_s": float(node_wall_s),
                "peak_allocated_bytes": peak_bytes,
                "device_budget_bytes": DEVICE_BUDGET_BYTES,
                "query_block": QUERY_BLOCK, "search_block": SEARCH_BLOCK,
            },
            "measured_price": {
                "schema": PRICE_SCHEMA,
                "label": "measurement",
                "exact_search_s": timing["exact_search_s"],
                "node_wall_s": float(node_wall_s),
                "gpu_hours": float(node_wall_s) / 3600.0,
                "scored_against_the_registered_prediction": scored,
                "two_term_law": law,
                "back_check_at_2m": back_check,
                "other_rungs": rungs,
                "what_the_law_licenses": (
                    "only the rung actually measured here is a measurement. "
                    "Every other entry in other_rungs carries label 'prediction' "
                    "and is licensed only while back_check_at_2m holds."
                ),
            },
            "disk": disk,
            "training_performed": False,
            "gpu_used": True,
            "cuda_presence": _cuda_maps_evidence(),
            "gap_report": report,
            "host_watchdog": guard.receipt(),
            "poll_coverage": coverage,
            "observed_span_s": coverage["observed_span_s"],
            "node_wall_s": float(node_wall_s),
            "coverage_summary": coverage_summary([coverage]),
        })
        _seal(output, "substrate-graph.json", body)
        sealed = True
    finally:
        if not sealed:
            force_remove_tree(output)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == PREDICT_ACTION:
        run_predict(active, job)
    elif action == BUILD_ACTION:
        run_build(active, job)
    else:
        raise Round0261NodeError(
            f"R0261 authorizes only {PREDICT_ACTION!r} and {BUILD_ACTION!r}, "
            f"not {action!r}")


__all__ = ["BUILD_ACTION", "PREDICT_ACTION", "PREDICT_CAPABILITY",
           "PREDICT_SCHEMA", "Round0261NodeError", "run_build", "run_job",
           "run_predict"]
