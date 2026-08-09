"""Execute R0233 — assemble the 6.25M rung, build its graph, qualify it.

Four nodes, one queue, deliberately narrow (no map is trained here):

* `assemble_6250k` (GPU lease, CPU work) assembles the 6,250,000-row mixed
  substrate at the owner-confirmed 40/25/25/10 shares with full row provenance,
  carves a held-out reserve per training corpus, asserts `>= 99.9%` shard
  coverage per corpus with a real `raise`, and measures the written substrate's
  extent count and cold read rate so the rung's I/O term is budgeted from a
  measurement rather than from a 3 GB file's throughput.
* `truth_6250k` (GPU) computes exact brute-force k15 truth over **all**
  6,250,000 rows. There is no sealed truth at this N and a sampled one would not
  support a uniform-population recall claim.
* `ladder_6250k` (GPU) measures cluster imbalance at every candidate `c` on this
  substrate, selects `c` from the MEASURED imbalance, and builds the registered
  ladder in ascending max-cluster order under a guard and a SIGNAL-FREE
  watchdog. Every cell emits its graph.
* `qualify_6250k` (GPU) scores every emitted graph over all 6,250,000 rows
  against the exact truth, applies the R0215 degree-zero tripwire, symmetrises
  the selected graph through R0216's identical fuzzy law, refits the device
  memory law on this round's own cells, and re-derives `c` for every remaining
  Phase-2 rung.

Nothing here registers a gate, trains a map, or claims an adoption.
"""
from __future__ import annotations

import gc
import glob
import json
import os
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
from basemap.round0233_substrate import (
    CANDIDATE,
    COMPOSITION,
    DATA_READ_CONTIGUOUS_BYTES_PER_S,
    DATA_READ_FRAGMENTED_BYTES_PER_S,
    DENSITY_DECILES,
    DETERMINISM_NOTE,
    DIMENSION,
    EXCLUDED_SHARDS,
    FUZZY_RANDOM_STATE_SEED,
    GD64_PRIOR_NOTE,
    GD64_PRIOR_POINTS,
    GRAPH_CAPABILITY,
    GRAPH_DEGREE,
    GRAPH_K,
    GRAPH_SCHEMA,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    INTERMEDIATE_GRAPH_DEGREE,
    LADDER_CAPABILITY,
    LADDER_CLUSTERS,
    LADDER_SCHEMA,
    LAW_SCHEMA,
    MAX_ITERATIONS,
    MAX_REPLACEMENT_ROUNDS,
    MAX_ZERO_DEGREE_ROWS,
    NN_DESCENT_SETTING,
    PHASE2_RUNGS,
    RAW_FORMAT,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    RECALL_POPULATION,
    RESERVE_QUERY_ROWS,
    RESERVE_ROWS,
    RESERVE_ROWS_PER_CORPUS,
    RESERVE_NOTE,
    ROUND_ID,
    ROWS,
    ROW_POLICY,
    Round0233Error,
    SAMPLE_INTERVAL_S,
    SELECTION_SEED,
    SPILL,
    SUBSTRATE_CAPABILITY,
    SUBSTRATE_SCHEMA,
    TRAILING_FRAGMENT_POLICY,
    TRUTH_CAPABILITY,
    TRUTH_METHOD,
    TRUTH_SCHEMA,
    ZERO_ROW_POLICY,
    assert_no_signal_policy,
    guard_decision,
    io_projection,
    refit_device_law,
    reserve_split,
    resolve_shard_rows,
    rung_derivation,
    select_clusters,
    validate_composition,
    validate_shard_span,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments.round0226_nodes import (
    BuildWatchdog,
    _child_environment,
    _nvidia_smi_device_bytes,
    _swap_used_bytes,
)
from experiments.round0227_nodes import CUML_LAUNCHER

EMB = "/data/embeddings"
BUILD_SCRIPT = "basemap/round0233_build.py"

ASSEMBLE_ACTION = "assemble_6250k"
TRUTH_ACTION = "truth_6250k"
LADDER_ACTION = "ladder_6250k"
QUALIFY_ACTION = "qualify_6250k"

#: Exact-truth blocking. Sized so the resident substrate (9.6 GB) plus one
#: similarity block stays well inside the 24 GiB device budget.
TRUTH_QUERY_BLOCK = 8_192
TRUTH_SEARCH_BLOCK = 131_072
COSINE_BLOCK = 65_536


# --------------------------------------------------------------------------- #
# signal-free child supervision
# --------------------------------------------------------------------------- #
class FlagWatchdog(BuildWatchdog):
    """R0226's watchdog with the signal removed.

    R0226 tripped by sending `SIGTERM`. Review-0232 located R0232's UVM deadlock
    in the signal-triggered exit path, so that escalation is itself a machine
    hazard. This subclass writes a flag file the child polls between clusters and
    never touches `os.kill`, `terminate()` or `kill()`.
    """

    def __init__(self, *, flag_path: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._flag_path = str(flag_path)

    def _trip(self, reason: str) -> None:  # noqa: D401 - overrides a signal path
        if self.aborted:
            return
        self.aborted = True
        self.abort_reason = reason
        try:
            with open(self._flag_path, "w", encoding="utf-8") as handle:
                handle.write(reason)
            self.escalations.append("cooperative-flag")
        except OSError as error:
            self.escalations.append(f"cooperative-flag-failed:{error}")


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
    own CUDA context.
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
    elif process.returncode != 0 or timed_out:
        raise Round0233Error(
            f"R0233 child {config['setting_id']} produced no receipt "
            f"(rc={process.returncode}, timed_out={timed_out}):\n"
            f"{stdout[-2000:]}\n{stderr[-2000:]}"
        )
    else:
        raise Round0233Error(f"R0233 child {config['setting_id']} wrote no receipt")
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
# node 1 — the substrate and its reserves
# --------------------------------------------------------------------------- #
def _shards(corpus: str) -> list[tuple[str, int, bool]]:
    """(path, complete_rows, is_real_npy) honouring R0025's loading contract."""
    out: list[tuple[str, int, bool]] = []
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
            rel = os.path.relpath(path, EMB)
            rows = resolve_shard_rows(
                relative_path=rel, size_bytes=os.path.getsize(path)
            )
        out.append((path, rows, real_npy))
    if not out:
        raise Round0233Error(f"no shards for {corpus}")
    return out


def _open(path: str, rows: int, real_npy: bool) -> np.ndarray:
    if real_npy:
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype="<f4", mode="r", shape=(rows, DIMENSION))


def _draw(
    *,
    shards: Sequence[tuple[str, int, bool]],
    offsets: np.ndarray,
    picked: np.ndarray,
    rng: np.random.RandomState,
    want: int,
    corpus: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Uniform over the WHOLE corpus, replacing rejects from the complement.

    This is R0216's corrected law. The earlier oversample-then-stop-at-quota
    approach walked shards in order and silently produced a leading PREFIX; the
    replacement rounds here keep the sample uniform over non-degenerate rows.
    """
    chosen: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    need = int(want)
    dropped = 0
    rounds = 0
    while need > 0:
        rounds += 1
        if rounds > MAX_REPLACEMENT_ROUNDS:
            raise Round0233Error(
                f"{corpus}: replacement did not converge after "
                f"{MAX_REPLACEMENT_ROUNDS} rounds"
            )
        free = np.flatnonzero(~picked)
        if free.size < need:
            raise Round0233Error(f"{corpus}: exhausted usable rows")
        draw = np.sort(rng.choice(free, need, replace=False)).astype(np.int64)
        del free
        picked[draw] = True
        shard_of = np.searchsorted(offsets, draw, side="right") - 1
        for index, (path, rows, real_npy) in enumerate(shards):
            local = draw[shard_of == index] - offsets[index]
            if local.size == 0:
                continue
            array = _open(path, rows, real_npy)
            block = np.asarray(array[local], dtype=np.float32)
            norm = np.linalg.norm(block, axis=1)
            ok = np.isfinite(block).all(axis=1) & (norm > 0)
            dropped += int((~ok).sum())
            if ok.any():
                chosen.append(draw[shard_of == index][ok])
                vectors.append(block[ok])
            del array, block, norm, ok
        need = int(want) - sum(len(item) for item in chosen)
        del draw, shard_of
    selected = np.concatenate(chosen)
    order = np.argsort(selected)
    return (
        selected[order],
        np.concatenate(vectors, axis=0)[order],
        dropped,
        rounds,
    )


def _extent_count(path: str) -> int | None:
    try:
        completed = subprocess.run(
            ["filefrag", path], check=False, capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = completed.stdout.strip()
    if "extent" not in text:
        return None
    for token in text.split():
        if token.isdigit():
            return int(token)
    return None


def _cold_read_rate(path: str, *, limit_bytes: int) -> dict[str, Any]:
    """Cold sequential read after `posix_fadvise(DONTNEED)`, on the real file.

    Review-0232-01 named this the highest-value cheap measurement in the program:
    `/data` delivers 5.90 GB/s on a 4-extent file and 1.24 GB/s on a 395-extent
    one, and the difference moves the 100M I/O term by ~4x.
    """
    size = os.path.getsize(path)
    take = int(min(limit_bytes, size))
    handle = os.open(path, os.O_RDONLY)
    try:
        try:
            os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
        except (OSError, AttributeError):
            pass
        started = time.perf_counter()
        read = 0
        chunk = 8 * 1024 * 1024
        while read < take:
            data = os.read(handle, min(chunk, take - read))
            if not data:
                break
            read += len(data)
        elapsed = time.perf_counter() - started
    finally:
        os.close(handle)
    return {
        "file_bytes": int(size),
        "bytes_read": int(read),
        "seconds": float(elapsed),
        "bytes_per_s": float(read / elapsed) if elapsed > 0 else None,
        "extents": _extent_count(path),
        "method": "posix_fadvise(DONTNEED) then sequential os.read",
    }


def _verify_source_sizes(job: Mapping[str, Any]) -> str:
    """Re-verify every base-corpus shard byte size against the bound manifest.

    The base corpora are 581 GB and cannot be rehashed per prepare, so the queue
    binds their sizes and the node re-checks them: a shard that changed under us
    would shift row ids silently.
    """
    signature = job.get("source_size_manifest")
    if signature is None:
        raise Round0233Error("R0233 requires the bound source size manifest")
    path = prompt_contract.verify_signature(
        dict(signature), label="R0233 source size manifest"
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
        raise Round0233Error(
            f"R0233 source shards changed size since preparation: {drift[:4]}"
        )
    return path


def run_assemble(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    if str(manifest.get("round_id")) != ROUND_ID:
        raise Round0233Error("R0233 handler received another queue")
    size_manifest_path = _verify_source_sizes(job)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0233 substrate")
    started = time.monotonic()

    provenance = np.empty(ROWS, dtype=np.dtype([
        ("corpus", "u1"), ("shard", "u2"), ("row", "i8")]))
    reserve_provenance = np.empty(RESERVE_ROWS, dtype=provenance.dtype)
    counts: dict[str, int] = {}
    rejects: dict[str, int] = {}
    spans: dict[str, Any] = {}
    reserve_spans: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    substrate_path = os.path.join(output, "substrate.f32.npy")
    reserve_path = os.path.join(output, "reserve.f32.npy")

    def _write_substrate(tmp_train: str, tmp_reserve: str) -> None:
        train = np.lib.format.open_memmap(
            tmp_train, mode="w+", dtype=np.float32, shape=(ROWS, DIMENSION)
        )
        reserve = np.lib.format.open_memmap(
            tmp_reserve, mode="w+", dtype=np.float32, shape=(RESERVE_ROWS, DIMENSION)
        )
        at = 0
        reserve_at = 0
        for index, (corpus, want) in enumerate(COMPOSITION):
            shards = _shards(corpus)
            total = int(sum(rows for _p, rows, _n in shards))
            if total < want + RESERVE_ROWS_PER_CORPUS:
                raise Round0233Error(
                    f"{corpus}: need {want + RESERVE_ROWS_PER_CORPUS} rows, "
                    f"corpus has {total}"
                )
            offsets = np.concatenate(
                [[0], np.cumsum([rows for _p, rows, _n in shards])]
            ).astype(np.int64)
            picked = np.zeros(total, dtype=bool)
            rng = np.random.RandomState(SELECTION_SEED + index)

            selected, vectors, dropped, rounds = _draw(
                shards=shards, offsets=offsets, picked=picked, rng=rng,
                want=int(want), corpus=corpus,
            )
            shard_of = np.searchsorted(offsets, selected, side="right") - 1
            train[at:at + want] = vectors
            provenance["corpus"][at:at + want] = index
            provenance["shard"][at:at + want] = shard_of
            provenance["row"][at:at + want] = selected - offsets[shard_of]
            spans[corpus] = {
                **validate_shard_span(
                    corpus=corpus,
                    shards_touched=int(np.unique(shard_of).size),
                    shards_total=len(shards),
                ),
                "replacement_rounds": int(rounds),
            }
            train_selected = selected.copy()
            del vectors, shard_of, selected

            # The reserve is drawn from the complement of the training picks, so
            # it is disjoint by construction; the disjointness is asserted below
            # on provenance as well.
            reserve_rng = np.random.RandomState(SELECTION_SEED + 500 + index)
            r_selected, r_vectors, r_dropped, r_rounds = _draw(
                shards=shards, offsets=offsets, picked=picked, rng=reserve_rng,
                want=RESERVE_ROWS_PER_CORPUS, corpus=f"{corpus}[reserve]",
            )
            if np.intersect1d(train_selected, r_selected).size != 0:
                raise Round0233Error(
                    f"{corpus}: reserve overlaps the training selection"
                )
            r_shard_of = np.searchsorted(offsets, r_selected, side="right") - 1
            reserve[reserve_at:reserve_at + RESERVE_ROWS_PER_CORPUS] = r_vectors
            reserve_provenance["corpus"][
                reserve_at:reserve_at + RESERVE_ROWS_PER_CORPUS] = index
            reserve_provenance["shard"][
                reserve_at:reserve_at + RESERVE_ROWS_PER_CORPUS] = r_shard_of
            reserve_provenance["row"][
                reserve_at:reserve_at + RESERVE_ROWS_PER_CORPUS] = (
                    r_selected - offsets[r_shard_of])
            reserve_spans[corpus] = {
                "shards_touched": int(np.unique(r_shard_of).size),
                "shards_total": len(shards),
                "coverage": float(np.unique(r_shard_of).size / len(shards)),
                "replacement_rounds": int(r_rounds),
                "degenerate_rows_dropped": int(r_dropped),
            }
            del r_vectors, r_shard_of, r_selected, train_selected, picked

            at += int(want)
            reserve_at += RESERVE_ROWS_PER_CORPUS
            counts[corpus] = int(want)
            rejects[corpus] = int(dropped)
            sources[corpus] = {
                "shards": len(shards),
                "corpus_rows": total,
                "selected_rows": int(want),
                "reserve_rows": RESERVE_ROWS_PER_CORPUS,
                "format": "npy" if shards[0][2] else RAW_FORMAT,
                "first_shard": expected_input_signature(shards[0][0]),
            }
        if at != ROWS or reserve_at != RESERVE_ROWS:
            raise Round0233Error(
                f"assembled {at} train / {reserve_at} reserve rows, expected "
                f"{ROWS} / {RESERVE_ROWS}"
            )
        for array in (train, reserve):
            block = 500_000
            for begin in range(0, array.shape[0], block):
                end = min(begin + block, array.shape[0])
                piece = np.asarray(array[begin:end])
                norms = np.linalg.norm(piece, axis=1)
                if not np.isfinite(piece).all() or float(norms.min()) <= 0:
                    raise Round0233Error(
                        "substrate contains nonfinite or zero rows"
                    )
                array[begin:end] = piece / norms[:, None]
                del piece, norms
            array.flush()
        del train, reserve

    tmp_reserve = os.path.join(output, ".reserve.building.npy")

    def _writer(tmp: str) -> None:
        _write_substrate(tmp, tmp_reserve)

    atomic_build_new_file(substrate_path, _writer, immutable=True)
    os.replace(tmp_reserve, reserve_path)
    os.chmod(reserve_path, 0o444)

    composition = validate_composition(counts)
    prov_path = atomic_save_new_npy(
        os.path.join(output, "provenance.npy"), provenance, immutable=True
    )
    reserve_prov_path = atomic_save_new_npy(
        os.path.join(output, "reserve-provenance.npy"), reserve_provenance,
        immutable=True,
    )

    reserve_index: dict[str, Any] = {}
    query_positions: list[np.ndarray] = []
    for index, (corpus, _want) in enumerate(COMPOSITION):
        corpus_side, query_side = reserve_split(index)
        base = index * RESERVE_ROWS_PER_CORPUS
        query_positions.append(query_side + base)
        reserve_index[corpus] = {
            "block_start": int(base),
            "block_rows": RESERVE_ROWS_PER_CORPUS,
            "heldout_corpus_rows": int(corpus_side.size),
            "heldout_query_rows": int(query_side.size),
        }
    query_path = atomic_save_new_npy(
        os.path.join(output, "reserve-query-rows.i64.npy"),
        np.sort(np.concatenate(query_positions)).astype(np.int64), immutable=True,
    )

    read_rate = _cold_read_rate(substrate_path, limit_bytes=4 * 1024 ** 3)
    substrate_signature = expected_input_signature(substrate_path)
    ordered = ordered_array_sha256(np.load(substrate_path, mmap_mode="r"))

    receipt = prompt_contract.seal({
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": SUBSTRATE_CAPABILITY,
        "capabilities": [SUBSTRATE_CAPABILITY],
        "rows": ROWS,
        "dimension": DIMENSION,
        "reserve_rows": RESERVE_ROWS,
        "composition": composition,
        "sources": sources,
        "loading_contract": {
            "raw_format": RAW_FORMAT,
            "row_policy": ROW_POLICY,
            "trailing_fragment_policy": TRAILING_FRAGMENT_POLICY,
        },
        "selection": {
            "seed": SELECTION_SEED,
            "law": (
                "per-corpus uniform over ALL complete rows of non-excluded "
                "shards; rejected rows replaced by fresh uniform draws from the "
                "unpicked complement until quota is met; never a prefix"
            ),
            "zero_row_policy": ZERO_ROW_POLICY,
            "degenerate_rows_dropped": rejects,
            "shard_span": spans,
            "excluded_shards": {
                key: value["reason"] for key, value in EXCLUDED_SHARDS.items()
            },
        },
        "reserve": {
            "note": RESERVE_NOTE,
            "rows_per_corpus": RESERVE_ROWS_PER_CORPUS,
            "query_rows_per_corpus": RESERVE_QUERY_ROWS,
            "index": reserve_index,
            "shard_span": reserve_spans,
            "disjoint_from_training": True,
            "enables": ["heldout_recall_at_10", "projection_ffr"],
        },
        "source_size_manifest": expected_input_signature(size_manifest_path),
        "substrate": substrate_signature,
        "provenance": expected_input_signature(prov_path),
        "reserve_substrate": expected_input_signature(reserve_path),
        "reserve_provenance": expected_input_signature(reserve_prov_path),
        "reserve_query_rows": expected_input_signature(query_path),
        "ordered_substrate_sha256": ordered,
        "substrate_read_measurement": read_rate,
        "read_rate_reference": {
            "fragmented_bytes_per_s": DATA_READ_FRAGMENTED_BYTES_PER_S,
            "contiguous_bytes_per_s": DATA_READ_CONTIGUOUS_BYTES_PER_S,
            "source": "review-0232-2026-08-09-01",
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
        raise Round0233Error(f"R0233 substrate geometry is {host.shape}/{host.dtype}")
    tensor = torch.empty((ROWS, DIMENSION), dtype=torch.float32, device=device)
    block = 500_000
    for begin in range(0, ROWS, block):
        end = min(begin + block, ROWS)
        tensor[begin:end] = torch.from_numpy(
            np.ascontiguousarray(host[begin:end])
        ).to(device)
    return tensor


def run_truth(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest = active["manifest"]
    substrate = prompt_contract.read_sealed(
        prompt_contract.verify_signature(
            dict(job["substrate_manifest"]), label="R0233 substrate manifest"
        ),
        label="R0233 substrate manifest",
    )
    substrate_path = prompt_contract.verify_signature(
        dict(substrate["substrate"]), label="R0233 substrate"
    )
    output = create_fresh_directory(str(job["outputs"][0]), label="R0233 exact truth")
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
        raise Round0233Error("R0233 exact truth retained a self edge")
    if int(ids.min()) < 0 or int(ids.max()) >= ROWS:
        raise Round0233Error("R0233 exact truth holds an out-of-range id")
    if not np.isfinite(cosines).all():
        raise Round0233Error("R0233 exact truth cosines are not finite")
    if int((np.diff(cosines, axis=1) > 1e-5).sum()) != 0:
        raise Round0233Error("R0233 exact truth is not descending in cosine")

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
# node 3 — measured imbalance, c selection, the build ladder
# --------------------------------------------------------------------------- #
def run_ladder(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    manifest = active["manifest"]
    repo_root = str(manifest["repo_root"])
    substrate = prompt_contract.read_sealed(
        prompt_contract.verify_signature(
            dict(job["substrate_manifest"]), label="R0233 substrate manifest"
        ),
        label="R0233 substrate manifest",
    )
    substrate_path = prompt_contract.verify_signature(
        dict(substrate["substrate"]), label="R0233 substrate"
    )
    output = create_fresh_directory(str(job["outputs"][0]), label="R0233 build ladder")
    scratch_root = ensure_data_directory(str(job["scratch_root"]))
    cache_root = str(job["cache_root"])
    started = time.monotonic()

    imbalance = _measure_imbalance(
        substrate_path=substrate_path, clusters=LADDER_CLUSTERS,
        output=output, repo_root=repo_root, cache_root=cache_root,
    )
    measured = {
        int(key): float(value["imbalance_max_over_mean"])
        for key, value in imbalance["cells"].items()
    }
    selection = select_clusters(rows=ROWS, measured_imbalance=measured)

    statvfs = os.statvfs("/data")
    disk_free = int(statvfs.f_bavail) * int(statvfs.f_frsize)

    cells: list[dict[str, Any]] = []
    stopped: str | None = None
    for clusters in LADDER_CLUSTERS:
        cell_id = f"r0233-n{ROWS}-c{clusters}-s{SPILL}"
        guard = guard_decision(
            rows=ROWS, clusters=clusters, imbalance=measured[clusters],
            imbalance_source="measured on this substrate at this N",
            disk_free_bytes=disk_free,
        )
        if stopped is not None:
            cells.append({
                "cell": cell_id, "clusters": int(clusters), "spill": SPILL,
                "guard": guard, "run": False,
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
        }
        cell_dir = os.path.join(output, "builds", cell_id)
        config["abort_flag"] = os.path.join(cell_dir, "abort.flag")
        record = _run_child(
            config=config, out_dir=cell_dir, cache_root=cache_root,
            repo_root=repo_root, guard=guard,
            timeout_s=float(job["build_timeout_s"]),
        )
        ids_path = os.path.join(cell_dir, "graph-k15-ids.i32.npy")
        entry = {
            "cell": cell_id, "clusters": int(clusters), "spill": SPILL,
            "guard": guard, "run": True, "build": record,
            "graph_ids": (
                expected_input_signature(ids_path)
                if os.path.exists(ids_path) else None
            ),
        }
        cells.append(entry)
        if not record.get("fit"):
            stopped = cell_id
    disk_free_after = os.statvfs("/data")
    receipt = prompt_contract.seal({
        "schema": LADDER_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": manifest["release_sha"],
        "capability": LADDER_CAPABILITY,
        "capabilities": [LADDER_CAPABILITY],
        "rows": ROWS,
        "spill": SPILL,
        "nn_descent_setting": NN_DESCENT_SETTING,
        "nn_descent": {
            "graph_degree": GRAPH_DEGREE,
            "intermediate_graph_degree": INTERMEDIATE_GRAPH_DEGREE,
            "max_iterations": MAX_ITERATIONS,
        },
        "measured_imbalance": imbalance,
        "cluster_selection": selection,
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


def _measure_imbalance(
    *,
    substrate_path: str,
    clusters: Sequence[int],
    output: str,
    repo_root: str,
    cache_root: str,
) -> dict[str, Any]:
    """Realised `max/mean` at every candidate `c`, on THIS substrate at THIS N.

    Runs as its own short child under the RAPIDS env, because k-means and the
    spill assignment live there. It builds no graph and hands cuVS nothing.
    """
    probe_dir = ensure_data_directory(os.path.join(output, "imbalance"))
    script = os.path.join(probe_dir, "probe.py")
    body = f'''
import json, os, sys
import numpy as np
sys.path.insert(0, {repo_root!r})
import cupy as cp
import cupyx
from basemap.round0226_cluster_spill_build import _assign, _kmeans
from basemap.round0226_graph_builders import A_SEED

rows = {ROWS}
spill = {SPILL}
dataset = np.load({substrate_path!r}, mmap_mode="r")[:rows]
assert isinstance(dataset, np.memmap) and not dataset.flags.writeable
cells = {{}}
for clusters in {tuple(int(value) for value in clusters)!r}:
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
with open(os.path.join({probe_dir!r}, "imbalance.json"), "w") as handle:
    json.dump({{"rows": rows, "spill": spill, "cells": cells}}, handle,
              indent=2, sort_keys=True)
'''
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(body)
    completed = subprocess.run(
        [CUML_LAUNCHER, script], cwd=repo_root,
        env=_child_environment(cache_root), capture_output=True, text=True,
        timeout=3_600,
    )
    result_path = os.path.join(probe_dir, "imbalance.json")
    if completed.returncode != 0 or not os.path.exists(result_path):
        raise Round0233Error(
            f"R0233 imbalance probe failed ({completed.returncode}):\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    with open(result_path, encoding="utf-8") as handle:
        measured = json.load(handle)
    measured["method"] = (
        "R0226's _kmeans and _assign, imported unmodified, at seed 226 with 25 "
        "Lloyd iterations on a 1,000,000-row subsample; imbalance is "
        "max(bincount(assignment)) / (spill * N / c)"
    )
    return measured


# --------------------------------------------------------------------------- #
# node 4 — qualification, the fuzzy graph, the refit, the per-rung derivation
# --------------------------------------------------------------------------- #
def _score_graph(
    *,
    ids: np.ndarray,
    tensor: Any,
    torch: Any,
    truth_ids: np.ndarray,
    kth: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Strict and tie-aware containment over ALL rows, cosines recomputed here.

    The candidate cosines are recomputed from the sealed substrate rather than
    read from the builder's own accumulator: review-0216-01 established that an
    in-node probe sharing the builder's accumulator is not independent.
    """
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
        "strict": summarize(strict, label="R0233 strict recall@15"),
        "tie_aware": summarize(tie, label="R0233 tie-aware recall@15"),
        "density_decile_tie_aware": decile_tie,
        "density_decile_strict": decile_strict,
        "density_decile_definition": (
            "deciles of the row's own exact 15th-best cosine; decile 0 sparsest"
        ),
        "rows_carrying_any_loss": int((lost > 0).sum()),
        "fraction_carrying_any_loss": float((lost > 0).mean()),
        "missing_true_edges": int(lost.sum()),
        "structural": structural,
        "zero_degree_rows": int(structural["zero_degree_rows"]),
    }
    del strict
    return summary, candidate_cos, tie


def run_qualify(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest = active["manifest"]
    substrate = prompt_contract.read_sealed(
        prompt_contract.verify_signature(
            dict(job["substrate_manifest"]), label="R0233 substrate manifest"
        ),
        label="R0233 substrate manifest",
    )
    substrate_path = prompt_contract.verify_signature(
        dict(substrate["substrate"]), label="R0233 substrate"
    )
    truth_manifest_path, _ = _intra(job, "truth_reference", label="R0233 truth")
    truth = prompt_contract.read_sealed(truth_manifest_path, label="R0233 truth")
    truth_ids_path, _ = _intra_signature(
        dict(truth["outputs"]["ids"]), label="R0233 truth ids"
    )
    truth_cos_path, _ = _intra_signature(
        dict(truth["outputs"]["cosines"]), label="R0233 truth cosines"
    )
    ladder_path, _ = _intra(job, "ladder_reference", label="R0233 build ladder")
    ladder = prompt_contract.read_sealed(ladder_path, label="R0233 build ladder")

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0233 qualification"
    )
    started = time.monotonic()

    truth_ids = np.load(truth_ids_path, allow_pickle=False)
    truth_cos = np.load(truth_cos_path, allow_pickle=False)
    if truth_ids.shape != (ROWS, GRAPH_K) or truth_cos.shape != (ROWS, GRAPH_K):
        raise Round0233Error("R0233 truth arrays have the wrong shape")
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
            dict(entry["graph_ids"]), label=f"R0233 {entry['cell']} graph"
        )
        ids = np.ascontiguousarray(
            np.load(ids_path, allow_pickle=False).astype(np.int32)
        )
        if ids.shape != (ROWS, GRAPH_K):
            raise Round0233Error(f"R0233 {entry['cell']} graph is {ids.shape}")
        summary, candidate_cos, _tie = _score_graph(
            ids=ids, tensor=tensor, torch=torch, truth_ids=truth_ids, kth=kth
        )
        summary["clusters"] = int(entry["clusters"])
        summary["spill"] = SPILL
        scored[str(entry["cell"])] = summary
        if int(entry["clusters"]) == selected_clusters:
            chosen_ids, chosen_cos, chosen_cell = ids, candidate_cos, str(entry["cell"])
        else:
            del ids, candidate_cos
        gc.collect()

    if chosen_ids is None or chosen_cos is None or chosen_cell is None:
        raise Round0233Error(
            f"R0233 selected c = {selected_clusters} produced no scored graph"
        )
    selected = scored[chosen_cell]
    if float(selected["tie_aware"]["mean"]) < RECALL_MEAN_FLOOR:
        raise Round0233Error(
            f"R0233 tie-aware recall {selected['tie_aware']['mean']} is below "
            f"the registered {RECALL_MEAN_FLOOR} floor"
        )
    if float(selected["tie_aware"]["p10"]) < RECALL_P10_FLOOR:
        raise Round0233Error(
            f"R0233 tie-aware p10 {selected['tie_aware']['p10']} is below the "
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
        raise Round0233Error("R0233 candidate distances are not finite")
    del cos_sorted

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
        raise Round0233Error("R0233 fuzzy weights are invalid")
    degree_counts = np.bincount(src, minlength=ROWS)
    degrees = {
        "zero_degree_rows": int((degree_counts == 0).sum()),
        "min": int(degree_counts.min()),
        "median": float(np.median(degree_counts)),
        "mean": float(degree_counts.mean()),
        "max": int(degree_counts.max()),
    }
    if degrees["zero_degree_rows"] > MAX_ZERO_DEGREE_ROWS:
        raise Round0233Error(
            f"R0233 R0215 tripwire: {degrees['zero_degree_rows']} zero-degree rows"
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
    del src, dst, wts, ids_sorted

    # ---- the refit R0229 registered and did not do ----
    points: list[tuple[int, float]] = list(GD64_PRIOR_POINTS)
    own: list[dict[str, Any]] = []
    for entry in ladder["ladder"]:
        build = entry.get("build") or {}
        if not build.get("fit"):
            continue
        largest = int((build.get("cluster_sizes") or {}).get("max") or 0)
        measured_bytes = build.get("device_wide_peak_over_baseline_bytes")
        if largest <= 0 or not measured_bytes:
            continue
        points.append((largest, float(measured_bytes)))
        own.append({
            "cell": entry["cell"],
            "clusters": int(entry["clusters"]),
            "max_cluster_rows": largest,
            "imbalance_max_over_mean": float(
                (build.get("cluster_sizes") or {}).get("imbalance_max_over_mean") or 0.0
            ),
            "device_wide_peak_bytes": int(build.get("device_wide_peak_bytes") or 0),
            "device_wide_peak_over_baseline_bytes": int(measured_bytes),
            "child_device_peak_sampled_bytes": int(
                build.get("child_device_peak_sampled_bytes") or 0
            ),
            "rmm_peak_bytes": int(build.get("rmm_peak_bytes") or 0),
            "host_anon_peak_bytes": int(build.get("host_anon_peak_bytes") or 0),
            "host_rss_peak_bytes": int(build.get("host_rss_peak_bytes") or 0),
            "host_vmhwm_bytes": int(build.get("host_vmhwm_bytes") or 0),
            "system_swap_growth_bytes": int(
                build.get("system_swap_growth_bytes") or 0
            ),
            "peak_scratch_bytes": int(build.get("peak_scratch_bytes") or 0),
            "spill_groups": int(build.get("spill_groups") or 0),
            "substrate_passes": int(build.get("substrate_passes") or 0),
            "builder_seconds": float(build.get("builder_seconds") or 0.0),
            "phases": build.get("phases"),
        })
    points.sort()
    law = refit_device_law(
        max_cluster_rows=[value for value, _b in points],
        device_bytes=[value for _a, value in points],
    )
    law["points"] = [
        {"max_cluster_rows": int(a), "device_bytes": float(b)} for a, b in points
    ]
    law["prior_gd64_points_included"] = [
        {"max_cluster_rows": int(a), "device_bytes": float(b)}
        for a, b in GD64_PRIOR_POINTS
    ]
    law["prior_gd64_note"] = GD64_PRIOR_NOTE

    measured_imbalance = {
        int(key): float(value["imbalance_max_over_mean"])
        for key, value in ladder["measured_imbalance"]["cells"].items()
    }
    slope = float(law["refit"]["slope"])
    intercept = float(law["refit"]["intercept"])
    rungs = {
        str(rung): rung_derivation(
            rung=int(rung), imbalance_by_c=measured_imbalance,
            imbalance_source=(
                "measured at N = 6,250,000, s = 8, on this round's substrate"
            ),
            law_intercept_bytes=intercept, law_bytes_per_row=slope,
        )
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
        "capabilities": [GRAPH_CAPABILITY],
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
        "graph": expected_input_signature(edges_path),
        "neighbour_ids": expected_input_signature(ids_out),
        "device_law_refit": law,
        "per_rung_derivation": rungs,
        "io_term_fragmented": io_term,
        "io_term_contiguous": io_contiguous,
        "substrate_read_measurement": substrate.get("substrate_read_measurement"),
        "per_cell_instruments": own,
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
        raise Round0233Error(f"{label} is absent at {path!r}")
    observed = expected_input_signature(path)
    if signature.get("sha256") and observed.get("sha256") != signature.get("sha256"):
        raise Round0233Error(f"{label} bytes changed")
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
        raise Round0233Error(f"R0233 does not authorize action {action!r}")


__all__ = [
    "ASSEMBLE_ACTION",
    "LADDER_ACTION",
    "QUALIFY_ACTION",
    "TRUTH_ACTION",
    "run_job",
]
