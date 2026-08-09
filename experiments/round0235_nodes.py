"""Execute R0235 — the nested 12.5M rung, its graph, and the imbalance sweep.

Four nodes, one queue, deliberately narrow (no map is trained here):

* `assemble_12500k` (GPU lease, CPU work) assembles the 12,500,000-row mixed
  substrate so that R0233's 6,250,000 training rows are its **positional
  prefix**, draws the increment uniformly from the complement of rung 1's
  training AND reserve rows, inherits rung 1's reserve verbatim, asserts nesting
  by hash and by provenance-key containment, asserts reserve disjointness, and
  asserts `>= 99.9%` shard coverage per corpus on the union *and* on the
  increment alone.
* `truth_12500k` (GPU) computes exact brute-force k15 truth over **all**
  12,500,000 rows. A sampled truth would not support a uniform-population claim.
* `ladder_12500k` (GPU) measures cluster imbalance at `c in (16, 32, 64, 128,
  200, 400)` at `s = 8` on this substrate, fits the device law on points read
  from **sealed** artifacts and filtered on the registered `(gd, igd, it)`
  triple, selects `c` under a guard that applies the imbalance margin, and builds
  the registered control cell plus the selected cell under a SIGNAL-FREE
  watchdog.
* `qualify_12500k` (GPU) scores every emitted graph over all 12,500,000 rows
  against the exact truth, applies the R0215 degree-zero tripwire, symmetrises
  the selected graph through R0216's identical fuzzy law, refits the device law
  with this round's own cells added, publishes the 2M / 6.25M / 12.5M imbalance
  drift table, and re-derives every remaining Phase-2 rung WITH the margin.

Nothing here registers a gate, trains a map, or claims an adoption.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0220_cuvs_qualification import (
    graph_validity,
    strict_containment_rows,
    summarize,
    tie_aware_rows,
)
from basemap.round0235_rung2 import (
    CANDIDATE,
    COMPOSITION,
    CONTROL_CLUSTERS,
    DATA_READ_CONTIGUOUS_BYTES_PER_S,
    DATA_READ_FRAGMENTED_BYTES_PER_S,
    DENSITY_DECILES,
    DETERMINISM_NOTE,
    DIMENSION,
    EXCLUDED_SHARDS,
    FUZZY_RANDOM_STATE_SEED,
    GRAPH_CAPABILITY,
    GRAPH_DEGREE,
    GRAPH_K,
    GRAPH_SCHEMA,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_IMBALANCE_MARGIN,
    GUARD_NOTE,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    IMBALANCE_CAPABILITY,
    IMBALANCE_PROBE_CLUSTERS,
    INCREMENT_BY_CORPUS,
    INTERMEDIATE_GRAPH_DEGREE,
    LADDER_CAPABILITY,
    LADDER_RULE,
    LADDER_SCHEMA,
    LAW_GRAPH_DEGREE,
    LAW_HOMOGENEITY_NOTE,
    LAW_INTERMEDIATE_GRAPH_DEGREE,
    LAW_MAX_ITERATIONS,
    LAW_RESIDUAL_MARGIN,
    LAW_SCHEMA,
    MAX_ITERATIONS,
    MAX_ZERO_DEGREE_ROWS,
    NESTING_NOTE,
    NN_DESCENT_SETTING,
    PARENT_COMPOSITION,
    PARENT_ROUND_ID,
    PARENT_ROWS,
    PHASE2_RUNGS,
    RAW_FORMAT,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RECALL_POPULATION,
    RESERVE_NOTE,
    RESERVE_QUERY_ROWS,
    RESERVE_ROWS,
    RESERVE_ROWS_PER_CORPUS,
    ROUND_ID,
    ROWS,
    ROW_POLICY,
    Round0235Error,
    SAMPLE_INTERVAL_S,
    SELECTION_CANDIDATES,
    SELECTION_LAW,
    SELECTION_SEED,
    SPILL,
    SUBSTRATE_CAPABILITY,
    SUBSTRATE_SCHEMA,
    TRAILING_FRAGMENT_POLICY,
    TRUTH_CAPABILITY,
    TRUTH_METHOD,
    TRUTH_SCHEMA,
    ZERO_ROW_POLICY,
    admissible_max_cluster_rows,
    assert_nesting,
    assert_no_signal_policy,
    assert_reserve_disjoint,
    fit_device_law,
    guard_decision,
    imbalance_drift,
    io_projection,
    provenance_keys,
    rung_derivation,
    select_clusters,
    validate_composition,
    validate_shard_span,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.gpu_child_supervision import (
    emit_child_abort_preamble,
    run_gpu_child_cooperative,
)
from experiments.round0226_nodes import (
    _child_environment,
    _nvidia_smi_device_bytes,
    _swap_used_bytes,
)
from experiments.round0227_nodes import CUML_LAUNCHER
from experiments.round0233_nodes import (
    FlagWatchdog,
    _cold_read_rate,
    _draw,
    _open,
    _shards,
)

EMB = "/data/embeddings"
BUILD_SCRIPT = "basemap/round0235_build.py"

ASSEMBLE_ACTION = "assemble_12500k"
TRUTH_ACTION = "truth_12500k"
LADDER_ACTION = "ladder_12500k"
QUALIFY_ACTION = "qualify_12500k"

#: Exact-truth blocking. The resident substrate is 19.2 GB at this rung, so the
#: similarity block is halved against R0233's: `8,192 x 65,536 x 4 B = 2.0 GiB`
#: of scratch on top of `17.9 GiB` of data keeps the node inside the 24 GiB
#: budget with the card's 31.37 GiB well clear.
TRUTH_QUERY_BLOCK = 8_192
TRUTH_SEARCH_BLOCK = 65_536
COSINE_BLOCK = 32_768
COPY_BLOCK = 250_000


# --------------------------------------------------------------------------- #
# signal-free child supervision — R0233's, with the capacity passed in
# --------------------------------------------------------------------------- #
def _run_child(
    *,
    config: Mapping[str, Any],
    out_dir: str,
    cache_root: str,
    repo_root: str,
    guard: Mapping[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """Guard, launch, watch, record. Every exit path yields a measurement.

    No signal is delivered to the child at any point, including on timeout: the
    parent sets the cooperative flag and then waits for the child to unwind its
    own CUDA context. `FlagWatchdog` has no `os.kill`/`terminate()`/`kill()`
    path at all.
    """
    identity = {
        "setting_id": str(config["setting_id"]),
        "candidate": CANDIDATE,
        "rows": int(config["rows"]),
        "clusters": int(config["clusters"]),
        "spill": int(config["spill"]),
        "config": dict(config),
    }
    ensure_data_directory(out_dir)
    config_path = os.path.join(out_dir, "config.json")
    atomic_write_new_json(config_path, dict(config), immutable=True)
    if not guard.get("allowed"):
        receipt = {
            **identity,
            "fit": False,
            "oom": False,
            "timed_out": False,
            "cooperatively_aborted": False,
            "refused_a_priori": True,
            "error_type": "RefusedAPriori",
            "guard": dict(guard),
            "refusal_reasons": list(guard.get("refusal_reasons") or []),
            "builder_seconds": None,
            "device_wide_peak_bytes": None,
            "device_wide_peak_over_baseline_bytes": None,
            "nvidia_smi_per_process_peak_bytes": None,
            "child_device_peak_sampled_bytes": None,
            "rmm_peak_bytes": None,
            "host_rss_peak_bytes": None,
            "host_anon_peak_bytes": None,
            "host_vmhwm_bytes": None,
            "system_swap_growth_bytes": None,
            "peak_scratch_bytes": None,
            "spill_groups": None,
            "substrate_passes": None,
            "watchdog_escalations": [],
        }
        atomic_write_new_json(
            os.path.join(out_dir, "build-receipt.json"), receipt, immutable=True
        )
        return receipt

    flag_path = os.path.join(out_dir, "abort.flag")
    device_baseline = _nvidia_smi_device_bytes()
    swap_baseline = _swap_used_bytes()
    started = time.perf_counter()
    process = subprocess.Popen(
        [
            CUML_LAUNCHER,
            os.path.join(repo_root, BUILD_SCRIPT),
            "--config", config_path,
            "--out", out_dir,
        ],
        cwd=repo_root,
        env=_child_environment(cache_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    watchdog = FlagWatchdog(
        flag_path=flag_path,
        pid=process.pid,
        poll_s=0.25,
        host_anon_budget_bytes=int(guard["host_anon_budget_bytes"]),
        swap_growth_abort_bytes=GUARD_SWAP_GROWTH_ABORT_BYTES,
        device_baseline_bytes=device_baseline,
        swap_baseline_bytes=swap_baseline,
    )
    watchdog.start()
    timed_out = False
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            watchdog._trip(  # noqa: SLF001 - the registered in-band abort path
                f"cell exceeded its {timeout_s:.0f} s deadline; cooperative flag "
                "set, no signal sent"
            )
            stdout, stderr = process.communicate()
    finally:
        watchdog.halt()
    readings = watchdog.readings()
    assert_no_signal_policy(readings.get("watchdog_escalations") or [])
    subprocess_seconds = time.perf_counter() - started

    receipt_path = os.path.join(out_dir, "build-receipt.json")
    if os.path.exists(receipt_path):
        with open(receipt_path, encoding="utf-8") as handle:
            child = json.load(handle)
    else:
        raise Round0235Error(
            f"R0235 child {config['setting_id']} produced no receipt "
            f"(rc={process.returncode}, timed_out={timed_out}):\n"
            f"{stdout[-2000:]}\n{stderr[-2000:]}"
        )
    return {
        **identity,
        **{
            key: value for key, value in child.items()
            if key not in ("config", "setting_id")
        },
        **readings,
        "subprocess_seconds": subprocess_seconds,
        "refused_a_priori": False,
        "timed_out": bool(timed_out),
        "guard": dict(guard),
        "stderr_tail": stderr[-2000:],
    }


# --------------------------------------------------------------------------- #
# node 1 — the nested substrate and the inherited reserve
# --------------------------------------------------------------------------- #
def _verify_source_sizes(job: Mapping[str, Any]) -> str:
    """Re-verify every base-corpus shard byte size against the bound manifest."""
    signature = job.get("source_size_manifest")
    if signature is None:
        raise Round0235Error("R0235 requires the bound source size manifest")
    path = prompt_contract.verify_signature(
        dict(signature), label="R0235 source size manifest"
    )
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    drift: list[str] = []
    for _corpus, entry in (manifest.get("corpora") or {}).items():
        for relative, want in (entry.get("shard_sizes") or {}).items():
            actual = os.path.getsize(os.path.join(EMB, relative))
            if actual != int(want):
                drift.append(f"{relative}: {actual} != {want}")
    if drift:
        raise Round0235Error(
            f"R0235 source shards changed size since preparation: {drift[:4]}"
        )
    return path


def _parent_substrate(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read and fully verify R0233's sealed rung-1 substrate manifest."""
    path = prompt_contract.verify_signature(
        dict(job["parent_substrate_manifest"]), label="R0233 substrate manifest"
    )
    sealed = prompt_contract.read_sealed(path, label="R0233 substrate manifest")
    if str(sealed.get("round_id")) != PARENT_ROUND_ID or int(
        sealed.get("rows", -1)
    ) != PARENT_ROWS:
        raise Round0235Error("R0235 parent manifest is not R0233's 6.25M rung")
    resolved = {
        key: prompt_contract.verify_signature(
            dict(sealed[key]), label=f"R0233 {key}"
        )
        for key in (
            "substrate", "provenance", "reserve_substrate", "reserve_provenance",
            "reserve_query_rows",
        )
    }
    return {"manifest_path": path, "sealed": sealed, "paths": resolved}


def _corpus_offsets(shards: Sequence[tuple[str, int, bool]]) -> np.ndarray:
    return np.concatenate(
        [[0], np.cumsum([rows for _p, rows, _n in shards])]
    ).astype(np.int64)


def _global_rows(records: np.ndarray, index: int, offsets: np.ndarray) -> np.ndarray:
    """Global corpus row ids for one corpus's slice of a provenance array."""
    mask = np.asarray(records["corpus"], dtype=np.int64) == int(index)
    shard = np.asarray(records["shard"], dtype=np.int64)[mask]
    row = np.asarray(records["row"], dtype=np.int64)[mask]
    return offsets[shard] + row


def run_assemble(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0235Error("R0235 handler received another queue")
    size_manifest_path = _verify_source_sizes(job)
    parent = _parent_substrate(job)
    parent_sealed = parent["sealed"]
    output = create_fresh_directory(str(job["outputs"][0]), label="R0235 substrate")
    started = time.monotonic()

    parent_prov = np.load(parent["paths"]["provenance"], allow_pickle=False)
    parent_reserve_prov = np.load(
        parent["paths"]["reserve_provenance"], allow_pickle=False
    )
    if parent_prov.shape[0] != PARENT_ROWS or parent_reserve_prov.shape[0] != (
        RESERVE_ROWS
    ):
        raise Round0235Error("R0235 parent provenance has the wrong row count")

    provenance = np.empty(ROWS, dtype=parent_prov.dtype)
    provenance[:PARENT_ROWS] = parent_prov
    counts: dict[str, int] = {}
    rejects: dict[str, int] = {}
    spans: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    per_shard_counts: dict[str, Any] = {}
    increments: dict[str, np.ndarray] = {}
    substrate_path = os.path.join(output, "substrate.f32.npy")

    # ---- draw the increment, per corpus, from the complement of rung 1 ----
    at = PARENT_ROWS
    increment_vectors: dict[str, str] = {}
    staging = ensure_data_directory(os.path.join(output, ".staging"))
    for index, (corpus, want_total) in enumerate(COMPOSITION):
        shards = _shards(corpus)
        total = int(sum(rows for _p, rows, _n in shards))
        sealed_source = (parent_sealed.get("sources") or {}).get(corpus) or {}
        if int(sealed_source.get("corpus_rows", -1)) != total or int(
            sealed_source.get("shards", -1)
        ) != len(shards):
            raise Round0235Error(
                f"{corpus}: shard enumeration differs from R0233's sealed record "
                f"({len(shards)} shards / {total} rows against "
                f"{sealed_source.get('shards')} / {sealed_source.get('corpus_rows')}); "
                "row ids would not mean the same thing and nesting would be a lie"
            )
        offsets = _corpus_offsets(shards)
        parent_train = _global_rows(parent_prov, index, offsets)
        parent_reserve = _global_rows(parent_reserve_prov, index, offsets)
        if parent_train.size != dict(PARENT_COMPOSITION)[corpus]:
            raise Round0235Error(f"{corpus}: parent training rows do not close")
        if parent_reserve.size != RESERVE_ROWS_PER_CORPUS:
            raise Round0235Error(f"{corpus}: parent reserve rows do not close")

        picked = np.zeros(total, dtype=bool)
        picked[parent_train] = True
        picked[parent_reserve] = True
        want = int(INCREMENT_BY_CORPUS[corpus])
        rng = np.random.RandomState(SELECTION_SEED + index)
        selected, vectors, dropped, rounds = _draw(
            shards=shards, offsets=offsets, picked=picked, rng=rng,
            want=want, corpus=f"{corpus}[increment]",
        )
        if np.intersect1d(selected, parent_train).size != 0:
            raise Round0235Error(f"{corpus}: increment overlaps rung 1's training")
        if np.intersect1d(selected, parent_reserve).size != 0:
            raise Round0235Error(f"{corpus}: increment overlaps rung 1's reserve")

        shard_of = np.searchsorted(offsets, selected, side="right") - 1
        provenance["corpus"][at:at + want] = index
        provenance["shard"][at:at + want] = shard_of
        provenance["row"][at:at + want] = selected - offsets[shard_of]

        union = np.concatenate([parent_train, selected])
        union_shard = np.searchsorted(offsets, union, side="right") - 1
        spans[corpus] = {
            "union": validate_shard_span(
                corpus=corpus, shards_touched=int(np.unique(union_shard).size),
                shards_total=len(shards), label="union",
            ),
            "increment": validate_shard_span(
                corpus=corpus, shards_touched=int(np.unique(shard_of).size),
                shards_total=len(shards), label="increment",
            ),
            "replacement_rounds": int(rounds),
        }
        per_shard_counts[corpus] = {
            "increment": np.bincount(
                shard_of, minlength=len(shards)
            ).astype(int).tolist(),
            "union": np.bincount(
                union_shard, minlength=len(shards)
            ).astype(int).tolist(),
            "shard_rows": [int(rows) for _p, rows, _n in shards],
        }
        stage_path = os.path.join(staging, f"increment-{index}.npy")
        np.save(stage_path, vectors.astype(np.float32, copy=False))
        increment_vectors[corpus] = stage_path
        increments[corpus] = selected

        counts[corpus] = int(want_total)
        rejects[corpus] = int(dropped)
        sources[corpus] = {
            "shards": len(shards),
            "corpus_rows": total,
            "selected_rows": int(want_total),
            "inherited_rows": dict(PARENT_COMPOSITION)[corpus],
            "increment_rows": want,
            "reserve_rows": RESERVE_ROWS_PER_CORPUS,
            "format": "npy" if shards[0][2] else RAW_FORMAT,
            "first_shard": expected_input_signature(shards[0][0]),
        }
        at += want
        del vectors, shard_of, union, union_shard, picked, parent_train, parent_reserve
        gc.collect()
    if at != ROWS:
        raise Round0235Error(f"assembled {at} rows, expected {ROWS}")

    # ---- write: rung 1's bytes verbatim, then the normalised increment ----
    parent_substrate_path = parent["paths"]["substrate"]

    def _write(tmp: str) -> None:
        train = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float32, shape=(ROWS, DIMENSION)
        )
        source = np.load(parent_substrate_path, mmap_mode="r", allow_pickle=False)
        if source.shape != (PARENT_ROWS, DIMENSION):
            raise Round0235Error("R0233 substrate geometry is not (6250000, 384)")
        for begin in range(0, PARENT_ROWS, COPY_BLOCK):
            end = min(begin + COPY_BLOCK, PARENT_ROWS)
            block = np.asarray(source[begin:end])
            norms = np.linalg.norm(block, axis=1)
            if not np.isfinite(block).all() or float(norms.min()) <= 0:
                raise Round0235Error("R0233 substrate prefix is degenerate")
            if float(np.abs(norms - 1.0).max()) > 1e-4:
                raise Round0235Error(
                    "R0233 substrate prefix is not unit-normalised; renormalising "
                    "it would break the byte-identical nesting this rung asserts"
                )
            # Copied verbatim: renormalising already-normalised rows changes bits.
            train[begin:end] = block
            del block, norms
        cursor = PARENT_ROWS
        for corpus, _want_total in COMPOSITION:
            want = int(INCREMENT_BY_CORPUS[corpus])
            staged = np.load(increment_vectors[corpus], mmap_mode="r")
            if staged.shape != (want, DIMENSION):
                raise Round0235Error(f"{corpus}: staged increment has {staged.shape}")
            for begin in range(0, want, COPY_BLOCK):
                end = min(begin + COPY_BLOCK, want)
                block = np.asarray(staged[begin:end], dtype=np.float32)
                norms = np.linalg.norm(block, axis=1)
                if not np.isfinite(block).all() or float(norms.min()) <= 0:
                    raise Round0235Error(f"{corpus}: increment holds a degenerate row")
                train[cursor + begin:cursor + end] = block / norms[:, None]
                del block, norms
            cursor += want
            del staged
        if cursor != ROWS:
            raise Round0235Error(f"wrote {cursor} rows, expected {ROWS}")
        train.flush()
        del train

    atomic_build_new_file(substrate_path, _write, immutable=True)
    shutil.rmtree(staging, ignore_errors=True)

    written = np.load(substrate_path, mmap_mode="r", allow_pickle=False)
    prefix_sha = ordered_array_sha256(written[:PARENT_ROWS])
    parent_ordered = str(parent_sealed["ordered_substrate_sha256"])
    if prefix_sha != parent_ordered:
        raise Round0235Error(
            "R0235 substrate prefix is not R0233's bytes: "
            f"{prefix_sha} != {parent_ordered}. The registered nesting is "
            "positional and byte-identical."
        )
    ordered = ordered_array_sha256(written)
    del written

    composition = validate_composition(counts)
    nesting = assert_nesting(parent=parent_prov, child=provenance)
    nesting.update({
        "prefix_ordered_sha256": prefix_sha,
        "parent_ordered_sha256": parent_ordered,
        "byte_identical_prefix": True,
        "parent_substrate": dict(parent_sealed["substrate"]),
    })

    prov_path = atomic_save_new_npy(
        os.path.join(output, "provenance.npy"), provenance, immutable=True
    )

    # ---- the reserve, inherited verbatim and copied out of the parent tree ----
    reserve_copies: dict[str, Any] = {}
    for key, name in (
        ("reserve_substrate", "reserve.f32.npy"),
        ("reserve_provenance", "reserve-provenance.npy"),
        ("reserve_query_rows", "reserve-query-rows.i64.npy"),
    ):
        source_path = parent["paths"][key]
        destination = os.path.join(output, name)

        def _copy(tmp: str, _source: str = source_path) -> None:
            shutil.copyfile(_source, tmp)

        atomic_build_new_file(destination, _copy, immutable=True)
        signature = expected_input_signature(destination)
        if signature["sha256"] != dict(parent_sealed[key])["sha256"]:
            raise Round0235Error(f"R0235 inherited {key} copy is not byte-identical")
        reserve_copies[key] = signature

    reserve_prov_copy = np.load(
        reserve_copies["reserve_provenance"]["canonical_path"], allow_pickle=False
    )
    disjoint = assert_reserve_disjoint(training=provenance, reserve=reserve_prov_copy)
    reserve_index: dict[str, Any] = {}
    for index, (corpus, _want) in enumerate(COMPOSITION):
        reserve_index[corpus] = {
            "block_start": int(index * RESERVE_ROWS_PER_CORPUS),
            "block_rows": RESERVE_ROWS_PER_CORPUS,
            "heldout_corpus_rows": RESERVE_ROWS_PER_CORPUS - RESERVE_QUERY_ROWS,
            "heldout_query_rows": RESERVE_QUERY_ROWS,
        }
    reserve_span: dict[str, Any] = {}
    for index, (corpus, _want) in enumerate(COMPOSITION):
        mask = np.asarray(reserve_prov_copy["corpus"], dtype=np.int64) == index
        shard_total = int(sources[corpus]["shards"])
        touched = int(np.unique(np.asarray(reserve_prov_copy["shard"])[mask]).size)
        reserve_span[corpus] = {
            "shards_touched": touched,
            "shards_total": shard_total,
            "coverage": touched / float(shard_total),
        }

    read_rate = _cold_read_rate(substrate_path, limit_bytes=4 * 1024 ** 3)
    receipt = prompt_contract.seal({
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": SUBSTRATE_CAPABILITY,
        "capabilities": [SUBSTRATE_CAPABILITY],
        "rows": ROWS,
        "dimension": DIMENSION,
        "reserve_rows": RESERVE_ROWS,
        "parent": {
            "round_id": PARENT_ROUND_ID,
            "rows": PARENT_ROWS,
            "manifest": expected_input_signature(parent["manifest_path"]),
            "note": (
                "rung 1's substrate lives under round-0233/queue/, the tree of "
                "the queue that terminated failed (review-0233-01 D7). This "
                "round's own substrate carries those 6,250,000 rows "
                "byte-identically in its prefix, in a queue tree of its own, so "
                "rung 1's bytes now exist outside a failed-queue path."
            ),
        },
        "composition": composition,
        "sources": sources,
        "loading_contract": {
            "raw_format": RAW_FORMAT,
            "row_policy": ROW_POLICY,
            "trailing_fragment_policy": TRAILING_FRAGMENT_POLICY,
        },
        "selection": {
            "seed": SELECTION_SEED,
            "law": SELECTION_LAW,
            "zero_row_policy": ZERO_ROW_POLICY,
            "degenerate_rows_dropped": rejects,
            "shard_span": spans,
            "per_shard_selection_counts": per_shard_counts,
            "excluded_shards": {
                key: value["reason"] for key, value in EXCLUDED_SHARDS.items()
            },
        },
        "nesting": nesting,
        "reserve": {
            "note": RESERVE_NOTE,
            "inherited_from_round": PARENT_ROUND_ID,
            "rows_per_corpus": RESERVE_ROWS_PER_CORPUS,
            "query_rows_per_corpus": RESERVE_QUERY_ROWS,
            "index": reserve_index,
            "shard_span": reserve_span,
            "disjointness": disjoint,
            "enables": ["heldout_recall_at_10", "projection_ffr"],
        },
        "source_size_manifest": expected_input_signature(size_manifest_path),
        "substrate": expected_input_signature(substrate_path),
        "provenance": expected_input_signature(prov_path),
        "reserve_substrate": reserve_copies["reserve_substrate"],
        "reserve_provenance": reserve_copies["reserve_provenance"],
        "reserve_query_rows": reserve_copies["reserve_query_rows"],
        "ordered_substrate_sha256": ordered,
        "substrate_read_measurement": read_rate,
        "read_rate_reference": {
            "fragmented_bytes_per_s": DATA_READ_FRAGMENTED_BYTES_PER_S,
            "contiguous_bytes_per_s": DATA_READ_CONTIGUOUS_BYTES_PER_S,
            "source": "review-0232-2026-08-09-01",
            "caveat": (
                "review-0233-01 D6 refuted extent COUNT as the discriminating "
                "variable; the fragmented rate is kept as a conservative floor, "
                "not as an extent-count model"
            ),
        },
        "performance": {"total_wall_s": time.monotonic() - started},
        "training_performed": False,
    })
    atomic_write_new_json(
        os.path.join(output, "substrate.json"), receipt, immutable=True
    )


# --------------------------------------------------------------------------- #
# node 2 — exact k15 truth over every row
# --------------------------------------------------------------------------- #
def _resident_substrate(path: str, torch: Any, device: Any) -> Any:
    host = np.load(path, mmap_mode="r", allow_pickle=False)
    if host.shape != (ROWS, DIMENSION) or host.dtype != np.float32:
        raise Round0235Error(f"R0235 substrate geometry is {host.shape}/{host.dtype}")
    tensor = torch.empty((ROWS, DIMENSION), dtype=torch.float32, device=device)
    for begin in range(0, ROWS, COPY_BLOCK):
        end = min(begin + COPY_BLOCK, ROWS)
        tensor[begin:end] = torch.from_numpy(
            np.ascontiguousarray(host[begin:end])
        ).to(device)
    return tensor


def _substrate_from_manifest(job: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    path, _observed = _intra(job, "substrate_manifest", label="R0235 substrate manifest")
    sealed = prompt_contract.read_sealed(path, label="R0235 substrate manifest")
    if str(sealed.get("round_id")) != ROUND_ID or int(sealed.get("rows", -1)) != ROWS:
        raise Round0235Error("R0235 substrate manifest is not this round's rung")
    return prompt_contract.verify_signature(
        dict(sealed["substrate"]), label="R0235 substrate"
    ), sealed


def run_truth(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest = active["manifest"]
    substrate_path, _sealed = _substrate_from_manifest(job)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0235 exact truth")
    started = time.monotonic()

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    tensor = _resident_substrate(substrate_path, torch, device)

    ids = np.empty((ROWS, GRAPH_K), dtype=np.int32)
    cosines = np.empty((ROWS, GRAPH_K), dtype=np.float32)
    width = GRAPH_K + 1
    search_started = time.monotonic()
    for qs in range(0, ROWS, TRUTH_QUERY_BLOCK):
        qe = min(qs + TRUTH_QUERY_BLOCK, ROWS)
        query = tensor[qs:qe]
        best_s = torch.full((qe - qs, width), -float("inf"), device=device)
        best_i = torch.full((qe - qs, width), -1, device=device, dtype=torch.int64)
        for cs in range(0, ROWS, TRUTH_SEARCH_BLOCK):
            ce = min(cs + TRUTH_SEARCH_BLOCK, ROWS)
            sims = query @ tensor[cs:ce].T
            take = min(width, ce - cs)
            top_s, top_i = torch.topk(sims, take, dim=1)
            merged_s = torch.cat([best_s, top_s], 1)
            merged_i = torch.cat([best_i, top_i.to(torch.int64) + cs], 1)
            order = torch.argsort(merged_s, dim=1, descending=True)[:, :width]
            best_s = torch.gather(merged_s, 1, order)
            best_i = torch.gather(merged_i, 1, order)
            del sims, top_s, top_i, merged_s, merged_i, order
        block_ids = best_i.cpu().numpy()
        block_cos = best_s.cpu().numpy()
        self_ids = np.arange(qs, qe, dtype=np.int64)[:, None]
        is_self = block_ids == self_ids
        keep = np.argsort(is_self, axis=1, kind="stable")
        ids[qs:qe] = np.take_along_axis(block_ids, keep, axis=1)[:, :GRAPH_K]
        cosines[qs:qe] = np.take_along_axis(block_cos, keep, axis=1)[:, :GRAPH_K]
        del best_s, best_i, block_ids, block_cos, is_self, keep
    search_s = time.monotonic() - search_started
    del tensor
    torch.cuda.empty_cache()
    gc.collect()

    self_rows = np.arange(ROWS, dtype=np.int64)[:, None]
    if int((ids.astype(np.int64) == self_rows).sum()) != 0:
        raise Round0235Error("R0235 exact truth retained a self edge")
    if int(ids.min()) < 0 or int(ids.max()) >= ROWS:
        raise Round0235Error("R0235 exact truth holds an out-of-range id")
    if not np.isfinite(cosines).all():
        raise Round0235Error("R0235 exact truth cosines are not finite")
    if int((np.diff(cosines, axis=1) > 1e-5).sum()) != 0:
        raise Round0235Error("R0235 exact truth is not descending in cosine")

    ids_path = atomic_save_new_npy(
        os.path.join(output, "truth-k15-ids.i32.npy"), ids, immutable=True
    )
    cos_path = atomic_save_new_npy(
        os.path.join(output, "truth-k15-cos.f32.npy"), cosines, immutable=True
    )
    receipt = prompt_contract.seal({
        "schema": TRUTH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": TRUTH_CAPABILITY,
        "capabilities": [TRUTH_CAPABILITY],
        "rows": ROWS,
        "k": GRAPH_K,
        "method": TRUTH_METHOD,
        "population": RECALL_POPULATION,
        "query_block": TRUTH_QUERY_BLOCK,
        "search_block": TRUTH_SEARCH_BLOCK,
        "outputs": {
            "ids": expected_input_signature(ids_path),
            "cosines": expected_input_signature(cos_path),
        },
        "kth_cosine": {
            "mean": float(cosines[:, GRAPH_K - 1].mean()),
            "min": float(cosines[:, GRAPH_K - 1].min()),
            "max": float(cosines[:, GRAPH_K - 1].max()),
        },
        "performance": {
            "exact_search_s": search_s,
            "total_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "training_performed": False,
    })
    atomic_write_new_json(
        os.path.join(output, "exact-k15-truth.json"), receipt, immutable=True
    )


# --------------------------------------------------------------------------- #
# node 3 — the imbalance sweep, the law, c selection, the ladder
# --------------------------------------------------------------------------- #
def _measure_imbalance(
    *,
    substrate_path: str,
    clusters: Sequence[int],
    output: str,
    repo_root: str,
    cache_root: str,
) -> dict[str, Any]:
    """Realised `max/mean` at every probed `c`, on THIS substrate at THIS N.

    Runs as its own short child under the RAPIDS env, because k-means and the
    spill assignment live there. It builds no graph and hands cuVS nothing.
    """
    probe_dir = ensure_data_directory(os.path.join(output, "imbalance"))
    script = os.path.join(probe_dir, "probe.py")
    flag_path = os.path.join(probe_dir, "abort.flag")
    body = f'''
import json, os, sys
import numpy as np
sys.path.insert(0, {repo_root!r})
import cupy as cp
import cupyx
from basemap.round0226_cluster_spill_build import _assign, _kmeans
from basemap.round0226_graph_builders import A_SEED
{emit_child_abort_preamble(flag_path)}
rows = {ROWS}
spill = {SPILL}
dataset = np.load({substrate_path!r}, mmap_mode="r")[:rows]
assert isinstance(dataset, np.memmap) and not dataset.flags.writeable
cells = {{}}
aborted = False
try:
  for clusters in {tuple(int(value) for value in clusters)!r}:
    _check_abort()
    centroids = _kmeans(cp, cupyx, dataset, clusters=clusters, seed=A_SEED)
    assignment = _assign(cp, dataset, centroids, rows=rows, spill=spill)
    sizes = np.bincount(assignment.ravel(), minlength=clusters).astype(np.int64)
    cells[str(clusters)] = {{
        "clusters": int(clusters),
        "spill": spill,
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "empty_clusters": int((sizes == 0).sum()),
        "imbalance_max_over_mean": float(sizes.max() / sizes.mean()),
    }}
    del centroids, assignment, sizes
    cp.get_default_memory_pool().free_all_blocks()
except _CooperativeAbort:
    aborted = True
with open(os.path.join({probe_dir!r}, "imbalance.json"), "w") as handle:
    json.dump({{"rows": rows, "spill": spill, "cells": cells,
               "aborted_cooperatively": aborted}}, handle,
              indent=2, sort_keys=True)
'''
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(body)
    completed = run_gpu_child_cooperative(
        [CUML_LAUNCHER, script], cwd=repo_root,
        env=_child_environment(cache_root),
        flag_path=flag_path, deadline_s=7_200,
    )
    result_path = os.path.join(probe_dir, "imbalance.json")
    if completed.returncode != 0 or not os.path.exists(result_path):
        raise Round0235Error(
            f"R0235 imbalance probe failed ({completed.returncode}):\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    with open(result_path, encoding="utf-8") as handle:
        measured = json.load(handle)
    if measured.get("aborted_cooperatively"):
        raise Round0235Error(
            f"R0235 imbalance probe hit its {completed.deadline_s:.0f} s "
            "cooperative deadline and unwound its own CUDA context; no signal "
            "was delivered. Partial cells are sealed in imbalance.json."
        )
    measured["method"] = (
        "R0226's _kmeans and _assign, imported unmodified, at seed 226 with 25 "
        "Lloyd iterations on a 1,000,000-row subsample; imbalance is "
        "max(bincount(assignment)) / (spill * N / c)"
    )
    measured["probed_clusters"] = [int(value) for value in clusters]
    return measured


def _read_bound(
    job: Mapping[str, Any], key: str, *, label: str, sealed: bool
) -> tuple[str, dict[str, Any]]:
    """Read one hash-bound inherited artifact.

    The queue binds every inherited input at its full signature, so provenance
    is enforced by `verify_signature` regardless. `sealed` says whether the
    artifact additionally carries a `prompt_contract` identity seal: R0233's
    build ladder and R0229's adopted-arm build receipt do; R0229's nn-descent
    quality sweep and spill-reachability artifacts predate that convention and
    carry none, so demanding one would refuse a perfectly intact file.
    """
    path = prompt_contract.verify_signature(dict(job[key]), label=label)
    if sealed:
        return path, prompt_contract.read_sealed(path, label=label)
    with open(path, encoding="utf-8") as handle:
        return path, json.load(handle)


def _inherited_law_points(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every sealed `gd 64 / igd 256 / it 40` device point that exists.

    Read from sealed artifacts and tagged with the setting the receipt itself
    records, so `admit_law_point` can enforce homogeneity rather than trust a
    label. Review-0233-01 D3: R0233 carried one point from a review's PROSE, in
    rounded GiB, and it was the wrong sweep cell.
    """
    points: list[dict[str, Any]] = []

    _sweep_path, sweep = _read_bound(
        job, "r0229_sweep", label="R0229 nnd quality sweep", sealed=False
    )
    for cell in sweep.get("cells") or []:
        memory = cell.get("memory") or {}
        points.append({
            "source": f"R0229 sweep {cell.get('cell')}",
            "cell": cell.get("cell"),
            "rows": sweep.get("rows"),
            "clusters": cell.get("clusters"),
            "spill": cell.get("spill"),
            "graph_degree": cell.get("graph_degree"),
            "intermediate_graph_degree": cell.get("intermediate_graph_degree"),
            "max_iterations": cell.get("max_iterations"),
            "max_cluster_rows": (cell.get("cluster_sizes") or {}).get("max"),
            "device_bytes": memory.get("device_wide_peak_over_baseline_bytes"),
        })

    _arm_path, arm = _read_bound(
        job, "r0229_arm", label="R0229 adopted arm build", sealed=True
    )
    build = arm.get("build") or {}
    points.append({
        "source": "R0229 adopted arm (c=200, s=8) @ 2M",
        "cell": "spill-lifted",
        "rows": build.get("rows"),
        "clusters": build.get("clusters"),
        "spill": build.get("spill"),
        "graph_degree": build.get("graph_degree"),
        "intermediate_graph_degree": build.get("intermediate_graph_degree"),
        "max_iterations": build.get("max_iterations"),
        "max_cluster_rows": (build.get("cluster_sizes") or {}).get("max"),
        "device_bytes": build.get("device_wide_peak_over_baseline_bytes"),
    })

    _ladder_path, ladder = _read_bound(
        job, "r0233_ladder", label="R0233 build ladder", sealed=True
    )
    setting = ladder.get("nn_descent") or {}
    for entry in ladder.get("ladder") or []:
        cell_build = entry.get("build") or {}
        if not cell_build.get("fit"):
            continue
        points.append({
            "source": f"R0233 ladder {entry.get('cell')}",
            "cell": entry.get("cell"),
            "rows": ladder.get("rows"),
            "clusters": entry.get("clusters"),
            "spill": entry.get("spill"),
            "graph_degree": setting.get("graph_degree"),
            "intermediate_graph_degree": setting.get("intermediate_graph_degree"),
            "max_iterations": setting.get("max_iterations"),
            "max_cluster_rows": (cell_build.get("cluster_sizes") or {}).get("max"),
            "device_bytes": cell_build.get(
                "device_wide_peak_over_baseline_bytes"
            ),
        })
    return points


def _inherited_imbalance(job: Mapping[str, Any]) -> dict[str, Any]:
    """Sealed `s = 8` imbalance at 2M (R0229) and 6.25M (R0233)."""
    reach_path, reach = _read_bound(
        job, "r0229_reachability", label="R0229 spill reachability", sealed=False
    )
    at_2m = {
        int(cell["clusters"]): float(cell["realised_imbalance"])
        for cell in reach.get("cells") or []
        if int(cell.get("spill") or 0) == SPILL
        and cell.get("realised_imbalance") is not None
    }
    ladder_path, ladder = _read_bound(
        job, "r0233_ladder", label="R0233 build ladder", sealed=True
    )
    cells = (ladder.get("measured_imbalance") or {}).get("cells") or {}
    at_6250k = {
        int(key): float(value["imbalance_max_over_mean"])
        for key, value in cells.items()
        if int(value.get("spill") or 0) == SPILL
    }
    return {
        int(reach["rows"]): at_2m,
        int(ladder["rows"]): at_6250k,
        "sources": {
            str(int(reach["rows"])): expected_input_signature(reach_path),
            str(int(ladder["rows"])): expected_input_signature(ladder_path),
        },
    }


def run_ladder(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    repo_root = str(manifest["repo_root"])
    substrate_path, _sealed = _substrate_from_manifest(job)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0235 build ladder")
    scratch_root = ensure_data_directory(str(job["scratch_root"]))
    cache_root = str(job["cache_root"])
    started = time.monotonic()

    inherited_points = _inherited_law_points(job)
    inherited_law = fit_device_law(inherited_points, label="inherited-gd64-igd256-it40")
    laws = [inherited_law]

    imbalance = _measure_imbalance(
        substrate_path=substrate_path, clusters=IMBALANCE_PROBE_CLUSTERS,
        output=output, repo_root=repo_root, cache_root=cache_root,
    )
    measured = {
        int(key): float(value["imbalance_max_over_mean"])
        for key, value in imbalance["cells"].items()
    }
    selectable = {
        c: value for c, value in measured.items() if c in set(SELECTION_CANDIDATES)
    }
    selection = select_clusters(
        rows=ROWS, measured_imbalance=selectable, laws=laws,
    )
    selected_clusters = int(selection["selected_clusters"])
    capacity_rows = int(admissible_max_cluster_rows(laws))

    statvfs = os.statvfs("/data")
    disk_free = int(statvfs.f_bavail) * int(statvfs.f_frsize)

    plan = sorted(
        {int(value) for value in CONTROL_CLUSTERS} | {selected_clusters},
        reverse=True,
    )  # descending c == ascending predicted max-cluster rows
    cells: list[dict[str, Any]] = []
    stopped: str | None = None
    for clusters in plan:
        cell_id = f"r0235-n{ROWS}-c{clusters}-s{SPILL}"
        guard = guard_decision(
            rows=ROWS, clusters=clusters, imbalance=measured[clusters],
            imbalance_source="measured on this substrate at this N",
            laws=laws, disk_free_bytes=disk_free,
        )
        role = (
            "selected" if clusters == selected_clusters
            else "control (matched max-cluster N-dependence probe)"
        )
        if stopped is not None:
            cells.append({
                "cell": cell_id, "clusters": int(clusters), "spill": SPILL,
                "role": role, "guard": guard, "run": False,
                "not_run_reason": f"ladder stopped at {stopped}",
            })
            continue
        config = {
            "setting_id": cell_id,
            "cell": cell_id,
            "candidate": CANDIDATE,
            "rows": ROWS,
            "clusters": int(clusters),
            "spill": SPILL,
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "substrate": substrate_path,
            "emit_graph": True,
            "scratch_root": scratch_root,
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "graph_degree": GRAPH_DEGREE,
            "intermediate_graph_degree": INTERMEDIATE_GRAPH_DEGREE,
            "max_iterations": MAX_ITERATIONS,
            "cluster_capacity_rows": capacity_rows,
        }
        cell_dir = os.path.join(output, "builds", cell_id)
        config["abort_flag"] = os.path.join(cell_dir, "abort.flag")
        record = _run_child(
            config=config, out_dir=cell_dir, cache_root=cache_root,
            repo_root=repo_root, guard=guard,
            timeout_s=float(job["build_timeout_s"]),
        )
        ids_path = os.path.join(cell_dir, "graph-k15-ids.i32.npy")
        cells.append({
            "cell": cell_id, "clusters": int(clusters), "spill": SPILL,
            "role": role, "guard": guard, "run": True, "build": record,
            "graph_ids": (
                expected_input_signature(ids_path)
                if os.path.exists(ids_path) else None
            ),
        })
        if not record.get("fit"):
            stopped = cell_id
    disk_free_after = os.statvfs("/data")
    receipt = prompt_contract.seal({
        "schema": LADDER_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": LADDER_CAPABILITY,
        "capabilities": [LADDER_CAPABILITY, IMBALANCE_CAPABILITY],
        "rows": ROWS,
        "spill": SPILL,
        "nn_descent_setting": NN_DESCENT_SETTING,
        "nn_descent": {
            "graph_degree": GRAPH_DEGREE,
            "intermediate_graph_degree": INTERMEDIATE_GRAPH_DEGREE,
            "max_iterations": MAX_ITERATIONS,
        },
        "measured_imbalance": imbalance,
        "inherited_device_law": inherited_law,
        "cluster_capacity_rows": capacity_rows,
        "cluster_capacity_note": (
            "derived from the inherited law plus the registered margins and "
            "passed to the build child in its config.json, where it is "
            "hash-bound; R0233's static 5,204,724 would have refused this rung's "
            "own selected cell a priori (review-0233-01 D1)"
        ),
        "guard_note": GUARD_NOTE,
        "cluster_selection": selection,
        "ladder_rule": LADDER_RULE,
        "ladder": cells,
        "ladder_stopped_at": stopped,
        "disk_free_bytes_before": disk_free,
        "disk_free_bytes_after": int(disk_free_after.f_bavail)
        * int(disk_free_after.f_frsize),
        "abort_policy": (
            "in-band cooperative flag only; no SIGTERM, no SIGKILL, no ptrace"
        ),
        "determinism_note": DETERMINISM_NOTE,
        "performance": {"total_wall_s": time.monotonic() - started},
        "training_performed": False,
    })
    atomic_write_new_json(
        os.path.join(output, "build-ladder.json"), receipt, immutable=True
    )


# --------------------------------------------------------------------------- #
# node 4 — qualification, the fuzzy graph, the drift table, the re-derivation
# --------------------------------------------------------------------------- #
def _score_graph(
    *,
    ids: np.ndarray,
    tensor: Any,
    torch: Any,
    truth_ids: np.ndarray,
    kth: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Strict and tie-aware containment over ALL rows, cosines recomputed here."""
    candidate_cos = np.empty(ids.shape, dtype=np.float32)
    for begin in range(0, ROWS, COSINE_BLOCK):
        end = min(begin + COSINE_BLOCK, ROWS)
        anchor = tensor[begin:end]
        neighbours = tensor[
            torch.from_numpy(ids[begin:end].astype(np.int64)).to(anchor.device)
        ]
        candidate_cos[begin:end] = (
            torch.einsum("bd,bkd->bk", anchor, neighbours).cpu().numpy()
        )
        del anchor, neighbours
    strict = strict_containment_rows(ids, truth_ids)
    tie = tie_aware_rows(candidate_cos.astype(np.float64), ids, kth)
    structural = graph_validity(ids, rows=ROWS)
    order = np.argsort(kth, kind="stable")
    decile_tie = [
        float(tie[order[index * ROWS // DENSITY_DECILES:
                        (index + 1) * ROWS // DENSITY_DECILES]].mean())
        for index in range(DENSITY_DECILES)
    ]
    decile_strict = [
        float(strict[order[index * ROWS // DENSITY_DECILES:
                           (index + 1) * ROWS // DENSITY_DECILES]].mean())
        for index in range(DENSITY_DECILES)
    ]
    lost = np.rint((1.0 - strict) * GRAPH_K).astype(np.int16)
    summary = {
        "recall_population": RECALL_POPULATION,
        "rows_measured": int(ROWS),
        "strict": summarize(strict, label="R0235 strict recall@15"),
        "tie_aware": summarize(tie, label="R0235 tie-aware recall@15"),
        "density_decile_tie_aware": decile_tie,
        "density_decile_strict": decile_strict,
        "density_decile_definition": (
            "deciles of the row's own exact 15th-best cosine; decile 0 sparsest"
        ),
        "rows_carrying_any_loss": int((lost > 0).sum()),
        "fraction_carrying_any_loss": float((lost > 0).mean()),
        "missing_true_edges": int(lost.sum()),
        "tie_aware_rows_below_one": int((tie < 1.0).sum()),
        "structural": structural,
        "zero_degree_rows": int(structural["zero_degree_rows"]),
    }
    del strict, tie, order
    return summary, candidate_cos


def run_qualify(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest = active["manifest"]
    substrate_path, substrate_sealed = _substrate_from_manifest(job)
    truth_manifest_path, _ = _intra(job, "truth_reference", label="R0235 truth")
    truth = prompt_contract.read_sealed(truth_manifest_path, label="R0235 truth")
    truth_ids_path, _ = _intra_signature(
        dict(truth["outputs"]["ids"]), label="R0235 truth ids"
    )
    truth_cos_path, _ = _intra_signature(
        dict(truth["outputs"]["cosines"]), label="R0235 truth cosines"
    )
    ladder_path, _ = _intra(job, "ladder_reference", label="R0235 build ladder")
    ladder = prompt_contract.read_sealed(ladder_path, label="R0235 build ladder")

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0235 qualification"
    )
    started = time.monotonic()

    truth_ids = np.load(truth_ids_path, allow_pickle=False)
    truth_cos = np.load(truth_cos_path, allow_pickle=False)
    if truth_ids.shape != (ROWS, GRAPH_K) or truth_cos.shape != (ROWS, GRAPH_K):
        raise Round0235Error("R0235 truth arrays have the wrong shape")
    kth = truth_cos[:, GRAPH_K - 1].astype(np.float64)
    del truth_cos

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    tensor = _resident_substrate(substrate_path, torch, device)

    selected_clusters = int(ladder["cluster_selection"]["selected_clusters"])
    scored: dict[str, Any] = {}
    chosen_ids: np.ndarray | None = None
    chosen_cos: np.ndarray | None = None
    chosen_cell: str | None = None
    for entry in ladder["ladder"]:
        if not entry.get("run") or not entry.get("graph_ids"):
            continue
        ids_path = prompt_contract.verify_signature(
            dict(entry["graph_ids"]), label=f"R0235 {entry['cell']} graph"
        )
        ids = np.ascontiguousarray(
            np.load(ids_path, allow_pickle=False).astype(np.int32)
        )
        if ids.shape != (ROWS, GRAPH_K):
            raise Round0235Error(f"R0235 {entry['cell']} graph is {ids.shape}")
        summary, candidate_cos = _score_graph(
            ids=ids, tensor=tensor, torch=torch, truth_ids=truth_ids, kth=kth
        )
        summary["clusters"] = int(entry["clusters"])
        summary["spill"] = SPILL
        summary["role"] = entry.get("role")
        scored[str(entry["cell"])] = summary
        if int(entry["clusters"]) == selected_clusters:
            chosen_ids, chosen_cos, chosen_cell = ids, candidate_cos, str(entry["cell"])
        else:
            del ids, candidate_cos
        gc.collect()

    if chosen_ids is None or chosen_cos is None or chosen_cell is None:
        raise Round0235Error(
            f"R0235 selected c = {selected_clusters} produced no scored graph"
        )
    selected = scored[chosen_cell]
    if float(selected["tie_aware"]["mean"]) < RECALL_MEAN_FLOOR:
        raise Round0235Error(
            f"R0235 tie-aware recall {selected['tie_aware']['mean']} is below "
            f"the registered {RECALL_MEAN_FLOOR} floor"
        )
    if float(selected["tie_aware"]["p10"]) < RECALL_P10_FLOOR:
        raise Round0235Error(
            f"R0235 tie-aware p10 {selected['tie_aware']['p10']} is below the "
            f"registered {RECALL_P10_FLOOR} floor"
        )
    del tensor, truth_ids
    torch.cuda.empty_cache()
    gc.collect()

    # ---- R0216's fuzzy law, unchanged, on the selected graph ----
    sort_order = np.argsort(-chosen_cos, axis=1, kind="stable")
    ids_sorted = np.take_along_axis(chosen_ids, sort_order, axis=1).astype(np.int32)
    cos_sorted = np.take_along_axis(chosen_cos, sort_order, axis=1)
    del sort_order, chosen_ids, chosen_cos
    dists = np.maximum((1.0 - cos_sorted).astype(np.float32), 0.0)
    if not np.isfinite(dists).all():
        raise Round0235Error("R0235 candidate distances are not finite")
    del cos_sorted
    gc.collect()

    host = np.load(substrate_path, mmap_mode="r", allow_pickle=False)
    import umap.umap_ as umap_api

    fuzzy_started = time.monotonic()
    graph, _sigmas, _rhos = umap_api.fuzzy_simplicial_set(
        host, n_neighbors=GRAPH_K,
        random_state=np.random.RandomState(FUZZY_RANDOM_STATE_SEED),
        metric="cosine", knn_indices=ids_sorted, knn_dists=dists,
    )
    coo = graph.tocoo()
    src = np.asarray(coo.row, dtype=np.int32)
    dst = np.asarray(coo.col, dtype=np.int32)
    wts = np.asarray(coo.data, dtype=np.float32)
    fuzzy_s = time.monotonic() - fuzzy_started
    del graph, coo, dists
    gc.collect()

    if not np.isfinite(wts).all() or wts.min() <= 0 or wts.max() > 1:
        raise Round0235Error("R0235 fuzzy weights are invalid")
    degree_counts = np.bincount(src, minlength=ROWS)
    degrees = {
        "zero_degree_rows": int((degree_counts == 0).sum()),
        "min": int(degree_counts.min()),
        "median": float(np.median(degree_counts)),
        "mean": float(degree_counts.mean()),
        "max": int(degree_counts.max()),
    }
    if degrees["zero_degree_rows"] > MAX_ZERO_DEGREE_ROWS:
        raise Round0235Error(
            f"R0235 R0215 tripwire: {degrees['zero_degree_rows']} zero-degree rows"
        )
    ids_out = atomic_save_new_npy(
        os.path.join(output, "graph-k15-ids.i32.npy"), ids_sorted, immutable=True
    )
    edges_path = atomic_save_new_npz(
        os.path.join(output, "edges-k15-fuzzy.npz"), immutable=True,
        compressed=False, sources=src, targets=dst, weights=wts,
        n_nodes=np.asarray(ROWS, dtype=np.int64),
        k=np.asarray(GRAPH_K, dtype=np.int64),
    )
    directed_edges = int(len(src))
    weight_min = float(wts.min())
    weight_max = float(wts.max())
    del src, dst, wts, ids_sorted, degree_counts
    gc.collect()

    # ---- the law, refitted with this round's own cells added ----
    inherited_law = ladder["inherited_device_law"]
    own_points: list[dict[str, Any]] = []
    instruments: list[dict[str, Any]] = []
    setting = ladder.get("nn_descent") or {}
    for entry in ladder["ladder"]:
        build = entry.get("build") or {}
        largest = int((build.get("cluster_sizes") or {}).get("max") or 0)
        measured_bytes = build.get("device_wide_peak_over_baseline_bytes")
        instruments.append({
            "cell": entry["cell"],
            "clusters": int(entry["clusters"]),
            "role": entry.get("role"),
            "run": bool(entry.get("run")),
            "fit": build.get("fit"),
            "refused_a_priori": build.get("refused_a_priori"),
            "refusal_reasons": build.get("refusal_reasons") or [],
            "max_cluster_rows": largest or None,
            "imbalance_max_over_mean": (
                build.get("cluster_sizes") or {}
            ).get("imbalance_max_over_mean"),
            "device_wide_peak_bytes": build.get("device_wide_peak_bytes"),
            "device_wide_peak_over_baseline_bytes": measured_bytes,
            "nvidia_smi_per_process_peak_bytes": build.get(
                "nvidia_smi_per_process_peak_bytes"
            ),
            "child_device_peak_sampled_bytes": build.get(
                "child_device_peak_sampled_bytes"
            ),
            "rmm_peak_bytes": build.get("rmm_peak_bytes"),
            "rmm_bytes_per_max_cluster_row": (
                float(build["rmm_peak_bytes"]) / largest
                if largest and build.get("rmm_peak_bytes") else None
            ),
            "host_rss_peak_bytes": build.get("host_rss_peak_bytes"),
            "host_anon_peak_bytes": build.get("host_anon_peak_bytes"),
            "host_vmhwm_bytes": build.get("host_vmhwm_bytes"),
            "system_swap_growth_bytes": build.get("system_swap_growth_bytes"),
            "peak_scratch_bytes": build.get("peak_scratch_bytes"),
            "spill_groups": build.get("spill_groups"),
            "substrate_passes": build.get("substrate_passes"),
            "builder_seconds": build.get("builder_seconds"),
            "watchdog_escalations": build.get("watchdog_escalations"),
            "signal_handler_installed": build.get("signal_handler_installed"),
            "cuvs_inputs_asserted_memmap": build.get("cuvs_inputs_asserted_memmap"),
            "cluster_capacity_rows": build.get("cluster_capacity_rows"),
            "phases": build.get("phases"),
        })
        if not build.get("fit") or largest <= 0 or not measured_bytes:
            continue
        own_points.append({
            "source": f"R0235 ladder {entry['cell']}",
            "cell": entry["cell"],
            "rows": ROWS,
            "clusters": int(entry["clusters"]),
            "spill": SPILL,
            "graph_degree": setting.get("graph_degree"),
            "intermediate_graph_degree": setting.get("intermediate_graph_degree"),
            "max_iterations": setting.get("max_iterations"),
            "max_cluster_rows": largest,
            "device_bytes": float(measured_bytes),
        })

    combined_law = fit_device_law(
        list(inherited_law["points"]) + own_points, label="all-sealed-gd64-igd256-it40"
    )
    own_law = (
        fit_device_law(own_points, label="r0235-own-cells")
        if len(own_points) >= 2 else None
    )
    laws = [combined_law] + ([own_law] if own_law else [])

    # ---- the N-dependence control ----
    control = None
    inherited_by_cluster = {
        int(point["max_cluster_rows"]): point for point in inherited_law["points"]
    }
    for point in own_points:
        for reference_rows, reference in inherited_by_cluster.items():
            if abs(reference_rows - int(point["max_cluster_rows"])) <= 5_000:
                control = {
                    "matched_on": "max_cluster_rows within 5,000",
                    "reference": reference,
                    "this_round": point,
                    "max_cluster_rows_ratio": (
                        float(point["max_cluster_rows"]) / float(reference_rows)
                    ),
                    "rows_ratio": float(point["rows"]) / float(reference["rows"]),
                    "device_bytes_delta": (
                        float(point["device_bytes"]) - float(reference["device_bytes"])
                    ),
                    "device_bytes_relative_delta": (
                        float(point["device_bytes"]) / float(reference["device_bytes"])
                        - 1.0
                    ),
                    "reading": (
                        "a device law that is a function of max-cluster rows "
                        "ALONE predicts a relative delta of 0 here; a positive "
                        "delta is the N-dependence review-0233-01 asked about "
                        "before 50M is priced"
                    ),
                }
                break

    # ---- the drift table and the per-rung re-derivation ----
    inherited_imbalance = _inherited_imbalance(job)
    measured_now = {
        int(key): float(value["imbalance_max_over_mean"])
        for key, value in ladder["measured_imbalance"]["cells"].items()
    }
    series = {
        key: value for key, value in inherited_imbalance.items()
        if isinstance(key, int)
    }
    series[ROWS] = measured_now
    drift = imbalance_drift(series)

    rungs = {
        str(rung): {
            "with_margin": rung_derivation(
                rung=int(rung), imbalance_by_c=measured_now,
                imbalance_source=(
                    f"measured at N = {ROWS:,}, s = 8, on this round's substrate"
                ),
                laws=laws, apply_margin=True,
            ),
            "point_estimate_no_margin": rung_derivation(
                rung=int(rung), imbalance_by_c=measured_now,
                imbalance_source=(
                    f"measured at N = {ROWS:,}, s = 8, on this round's substrate"
                ),
                laws=laws, apply_margin=False,
            ),
        }
        for rung in PHASE2_RUNGS
    }

    selected_build = next(
        entry["build"] for entry in ladder["ladder"]
        if entry.get("run") and int(entry["clusters"]) == selected_clusters
    )
    io_term = io_projection(
        rows=ROWS, substrate_passes=int(selected_build.get("substrate_passes") or 1)
    )
    io_contiguous = io_projection(
        rows=ROWS, substrate_passes=int(selected_build.get("substrate_passes") or 1),
        read_bytes_per_s=DATA_READ_CONTIGUOUS_BYTES_PER_S,
    )

    receipt = prompt_contract.seal({
        "schema": GRAPH_SCHEMA,
        "law_schema": LAW_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": GRAPH_CAPABILITY,
        "capabilities": [GRAPH_CAPABILITY, IMBALANCE_CAPABILITY],
        "rows": ROWS,
        "k": GRAPH_K,
        "spill": SPILL,
        "selected_clusters": selected_clusters,
        "selected_cell": chosen_cell,
        "cluster_selection": ladder["cluster_selection"],
        "recall_population": RECALL_POPULATION,
        "truth_method": TRUTH_METHOD,
        "scored_graphs": scored,
        "selected_graph": selected,
        "floors": {
            "tie_aware_mean": RECALL_MEAN_FLOOR,
            "tie_aware_p10": RECALL_P10_FLOOR,
            "zero_degree_rows": MAX_ZERO_DEGREE_ROWS,
        },
        "degrees": degrees,
        "directed_edges": directed_edges,
        "edges_per_row": directed_edges / float(ROWS),
        "fuzzy_weight_range": [weight_min, weight_max],
        "graph": expected_input_signature(edges_path),
        "neighbour_ids": expected_input_signature(ids_out),
        "nesting": substrate_sealed.get("nesting"),
        "device_law_inherited": inherited_law,
        "device_law_combined": combined_law,
        "device_law_own_cells": own_law,
        "device_law_homogeneity_note": LAW_HOMOGENEITY_NOTE,
        "device_law_setting": {
            "graph_degree": LAW_GRAPH_DEGREE,
            "intermediate_graph_degree": LAW_INTERMEDIATE_GRAPH_DEGREE,
            "max_iterations": LAW_MAX_ITERATIONS,
        },
        "n_dependence_control": control,
        "guard": {
            "note": GUARD_NOTE,
            "imbalance_margin": GUARD_IMBALANCE_MARGIN,
            "law_residual_margin": LAW_RESIDUAL_MARGIN,
            "device_budget_bytes": int(GUARD_DEVICE_BUDGET_BYTES),
            "admissible_max_cluster_rows": float(admissible_max_cluster_rows(laws)),
            "cluster_capacity_rows_used": ladder.get("cluster_capacity_rows"),
        },
        "imbalance_drift": drift,
        "imbalance_sources": inherited_imbalance.get("sources"),
        "per_rung_derivation": rungs,
        "io_term_fragmented": io_term,
        "io_term_contiguous": io_contiguous,
        "substrate_read_measurement": substrate_sealed.get(
            "substrate_read_measurement"
        ),
        "per_cell_instruments": instruments,
        "determinism_note": DETERMINISM_NOTE,
        "performance": {
            "fuzzy_s": fuzzy_s,
            "total_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": False,
    })
    atomic_write_new_json(
        os.path.join(output, "qualified-graph.json"), receipt, immutable=True
    )


# --------------------------------------------------------------------------- #
# intra-queue references
# --------------------------------------------------------------------------- #
def _intra_signature(
    signature: Mapping[str, Any], *, label: str
) -> tuple[str, dict[str, Any]]:
    path = str(signature.get("canonical_path") or "")
    if not path or not os.path.exists(path):
        raise Round0235Error(f"{label} is absent at {path!r}")
    observed = expected_input_signature(path)
    if signature.get("sha256") and observed.get("sha256") != signature.get("sha256"):
        raise Round0235Error(f"{label} bytes changed")
    return path, observed


def _intra(
    job: Mapping[str, Any], key: str, *, label: str
) -> tuple[str, dict[str, Any]]:
    reference = dict(job[key])
    return _intra_signature(reference, label=label)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == ASSEMBLE_ACTION:
        run_assemble(active, job)
    elif action == TRUTH_ACTION:
        run_truth(active, job)
    elif action == LADDER_ACTION:
        run_ladder(active, job)
    elif action == QUALIFY_ACTION:
        run_qualify(active, job)
    else:
        raise Round0235Error(f"R0235 does not authorize action {action!r}")


__all__ = [
    "ASSEMBLE_ACTION",
    "LADDER_ACTION",
    "QUALIFY_ACTION",
    "TRUTH_ACTION",
    "run_job",
]
