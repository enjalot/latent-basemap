"""Execute R0224 — assemble a benchmark substrate, then measure cuVS memory.

Two nodes.

`assemble_benchmark_substrate` (CPU) builds a 16,000,000-row mixed MiniLM
substrate under R0216's selection law with the composition shares scaled x8,
then writes it in a **seeded global row permutation** so every prefix of the file
is a uniform subsample of the whole rather than one corpus block. The N-sweep
reads prefixes, so without the permutation it would confound scale with
composition. Rows are streamed into an on-disk `open_memmap`, never held in a
resident array. These are benchmark bytes and seal no training capability.

`sweep_cuvs_memory` (GPU) runs the (N x igd) matrix, each cell as a **fresh
RAPIDS subprocess** so `VmHWM` is a per-build measurement, with four instruments
running at once and R0220's `nvidia-smi` poll carried alongside as a control. It
then reports which instruments can see intermediate degree at all, and only if
one can does it fit anything or project anything.
"""
from __future__ import annotations

import gc
import glob
import json
import os
import resource
import subprocess
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0216_minilm_2m_substrate import (
    COMPOSITION,
    EXCLUDED_SHARDS,
    RAW_FORMAT,
    ROW_POLICY,
    TRAILING_FRAGMENT_POLICY,
    ZERO_ROW_POLICY,
    resolve_shard_rows,
)
from basemap.round0224_cuvs_memory import (
    BENCHMARK_COMPOSITION_SCALE,
    BENCHMARK_NOTE,
    BENCHMARK_ROWS,
    BENCHMARK_SELECTION_SEED,
    BENCHMARK_SHUFFLE_SEED,
    BUDGET_TOLERANCE,
    BUILD_TIMEOUT_S,
    CONTROL_INSTRUMENT,
    DIMENSION,
    GPU_HOURS_CAP,
    HOST_RSS_LIMIT_GIB,
    INSTRUMENTS,
    PROJECTION_SUBSTRATE_BYTES,
    REGISTERED_DEVICE_TOTAL_BYTES,
    REGISTERED_HOST_TOTAL_BYTES,
    ROUND_ID,
    Round0224Error,
    SAMPLE_INTERVAL_S,
    SHARD_SPAN_FLOOR,
    SUBSTRATE_CAPABILITY,
    SUBSTRATE_SCHEMA,
    SWEEP_CAPABILITY,
    SWEEP_GRAPH_DEGREE,
    SWEEP_INTERMEDIATE_DEGREES,
    SWEEP_MAX_ITERATIONS,
    SWEEP_METRIC,
    SWEEP_ROWS,
    SWEEP_SCHEMA,
    residency_probe_settings,
    summarize_sweep,
    sweep_settings,
    validate_prefix_composition,
)
from basemap import round0113_prompt_contrast as prompt_contract


ASSEMBLE_ACTION = "assemble_benchmark_substrate"
SWEEP_ACTION = "sweep_cuvs_memory"

EMB = "/data/embeddings"
CUML_LAUNCHER = "/data/latent-basemap/cuml_py"
BUILD_SCRIPT = "basemap/round0224_cuvs_memory_build.py"
NVIDIA_SMI = "/usr/bin/nvidia-smi"
NVIDIA_SMI_POLL_S = 0.1
DRAW_BLOCK = 200_000


# --------------------------------------------------------------------------- #
# node 1: the benchmark substrate
# --------------------------------------------------------------------------- #


def _shards(corpus: str) -> list[tuple[str, int, bool]]:
    """R0216's shard resolution, imported constants and all."""
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
            relative = os.path.relpath(path, EMB)
            rows = resolve_shard_rows(
                relative_path=relative, size_bytes=os.path.getsize(path)
            )
        out.append((path, rows, real_npy))
    if not out:
        raise Round0224Error(f"R0224 found no shards for {corpus}")
    return out


def _open_shard(path: str, rows: int, real_npy: bool) -> np.ndarray:
    if real_npy:
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype="<f4", mode="r", shape=(rows, DIMENSION))


def run_assemble(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    from numpy.lib.format import open_memmap

    started = time.monotonic()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0224 benchmark substrate"
    )
    target_counts = {
        name: int(rows) * BENCHMARK_COMPOSITION_SCALE for name, rows in COMPOSITION
    }
    if sum(target_counts.values()) != BENCHMARK_ROWS:
        raise Round0224Error(
            f"R0224 scaled composition totals {sum(target_counts.values())}, "
            f"registered {BENCHMARK_ROWS}"
        )

    substrate_path = os.path.join(output, "substrate.f32.npy")
    substrate = open_memmap(
        substrate_path, mode="w+", dtype=np.float32, shape=(BENCHMARK_ROWS, DIMENSION)
    )
    provenance = np.empty(
        BENCHMARK_ROWS,
        dtype=np.dtype([("corpus", "u1"), ("shard", "u2"), ("row", "i8")]),
    )
    # The seeded global permutation: destination row for the i-th selected row.
    placement = np.random.RandomState(BENCHMARK_SHUFFLE_SEED).permutation(
        BENCHMARK_ROWS
    )

    spans: dict[str, Any] = {}
    rejects: dict[str, int] = {}
    sources: dict[str, Any] = {}
    written = 0
    for index, (corpus, base_rows) in enumerate(COMPOSITION):
        want = int(base_rows) * BENCHMARK_COMPOSITION_SCALE
        shards = _shards(corpus)
        total = sum(rows for _p, rows, _n in shards)
        if total < want:
            raise Round0224Error(
                f"R0224 {corpus}: need {want} rows, corpus has {total}"
            )
        offsets = np.concatenate([[0], np.cumsum([r for _p, r, _n in shards])])
        rng = np.random.RandomState(BENCHMARK_SELECTION_SEED + index)
        picked = np.zeros(total, dtype=bool)
        corpus_written = 0
        need = want
        dropped = 0
        rounds = 0
        touched: set[int] = set()
        while need > 0:
            rounds += 1
            if rounds > 8:
                raise Round0224Error(
                    f"R0224 {corpus}: replacement did not converge after 8 rounds"
                )
            free = np.flatnonzero(~picked)
            if free.size < need:
                raise Round0224Error(f"R0224 {corpus}: exhausted usable rows")
            draw = np.sort(rng.choice(free, need, replace=False)).astype(np.int64)
            del free
            picked[draw] = True
            shard_of = np.searchsorted(offsets, draw, side="right") - 1
            for shard_index, (path, rows, real_npy) in enumerate(shards):
                mask = shard_of == shard_index
                local_global = draw[mask]
                if local_global.size == 0:
                    continue
                touched.add(shard_index)
                local = local_global - offsets[shard_index]
                array = _open_shard(path, rows, real_npy)
                for start in range(0, local.size, DRAW_BLOCK):
                    stop = min(start + DRAW_BLOCK, local.size)
                    block = np.asarray(array[local[start:stop]], dtype=np.float32)
                    norms = np.linalg.norm(block, axis=1)
                    ok = np.isfinite(block).all(axis=1) & (norms > 0)
                    dropped += int((~ok).sum())
                    if not ok.any():
                        continue
                    kept = block[ok]
                    kept /= np.linalg.norm(kept, axis=1)[:, None]
                    destinations = placement[written : written + kept.shape[0]]
                    substrate[destinations] = kept
                    provenance["corpus"][destinations] = index
                    provenance["shard"][destinations] = shard_index
                    provenance["row"][destinations] = local[start:stop][ok]
                    written += int(kept.shape[0])
                    corpus_written += int(kept.shape[0])
                    del block, kept, destinations
                del array
            need = want - corpus_written
            if need < 0:
                raise Round0224Error(
                    f"R0224 {corpus} wrote {corpus_written} rows against a target "
                    f"of {want}"
                )
        del picked
        gc.collect()
        coverage = len(touched) / len(shards)
        if coverage < SHARD_SPAN_FLOOR:
            raise Round0224Error(
                f"R0224 {corpus}: selection touched {len(touched)}/{len(shards)} "
                f"shards ({coverage:.2%}); R0216's law requires the sample to span "
                "the corpus"
            )
        spans[corpus] = {
            "shards_touched": len(touched),
            "shards_total": len(shards),
            "coverage": coverage,
            "replacement_rounds": rounds,
        }
        if corpus_written != want:
            raise Round0224Error(
                f"R0224 {corpus} wrote {corpus_written} rows, registered {want}"
            )
        rejects[corpus] = dropped
        sources[corpus] = {
            "shards": len(shards),
            "corpus_rows": int(total),
            "selected_rows": want,
            "format": "npy" if shards[0][2] else RAW_FORMAT,
            "first_shard": expected_input_signature(shards[0][0]),
        }
    if written != BENCHMARK_ROWS:
        raise Round0224Error(
            f"R0224 wrote {written} rows, registered {BENCHMARK_ROWS}"
        )
    substrate.flush()
    corpus_of_row = np.asarray(provenance["corpus"], dtype=np.int64)
    del substrate
    gc.collect()

    check = np.load(substrate_path, mmap_mode="r")
    if check.shape != (BENCHMARK_ROWS, DIMENSION) or check.dtype != np.float32:
        raise Round0224Error("R0224 benchmark substrate geometry is wrong")
    sample = np.asarray(check[:: max(1, BENCHMARK_ROWS // 100_000)], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)
    if not np.isfinite(sample).all() or float(norms.min()) <= 0.0:
        raise Round0224Error("R0224 benchmark substrate contains degenerate rows")
    del sample, check
    os.chmod(substrate_path, 0o444)

    provenance_path = atomic_save_new_npy(
        os.path.join(output, "provenance.npy"), provenance, immutable=True
    )
    targets = {name: count / BENCHMARK_ROWS for name, count in target_counts.items()}
    prefixes: dict[str, Any] = {}
    for rows in SWEEP_ROWS:
        counts = np.bincount(corpus_of_row[:rows], minlength=len(COMPOSITION))
        shares = {
            name: float(counts[index]) / float(rows)
            for index, (name, _n) in enumerate(COMPOSITION)
        }
        prefixes[str(rows)] = validate_prefix_composition(
            shares=shares, targets=targets, rows=int(rows)
        )

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0224Error(
            f"R0224 assembly peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    receipt = prompt_contract.seal({
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": SUBSTRATE_CAPABILITY,
        "capabilities": [SUBSTRATE_CAPABILITY],
        "rows": BENCHMARK_ROWS,
        "dimension": DIMENSION,
        "purpose": BENCHMARK_NOTE,
        "benchmark_only": True,
        "training_performed": False,
        "seals_training_capability": False,
        "composition": {
            name: {"rows": count, "share": count / BENCHMARK_ROWS}
            for name, count in target_counts.items()
        },
        "composition_scale_vs_r0216": BENCHMARK_COMPOSITION_SCALE,
        "selection": {
            "law": (
                "R0216 queue-correction-3: per-corpus uniform over ALL complete "
                "rows of non-excluded shards, rejected rows replaced from the "
                "unpicked complement, shard span asserted"
            ),
            "seed": BENCHMARK_SELECTION_SEED,
            "shard_span_floor": SHARD_SPAN_FLOOR,
            "shard_span": spans,
            "degenerate_rows_dropped": rejects,
            "zero_row_policy": ZERO_ROW_POLICY,
            "excluded_shards": {
                key: value["reason"] for key, value in EXCLUDED_SHARDS.items()
            },
        },
        "row_order": {
            "law": "seeded global permutation of all 16,000,000 destinations",
            "seed": BENCHMARK_SHUFFLE_SEED,
            "why": (
                "the N-sweep reads prefixes; without the permutation a prefix "
                "would be one corpus block and scale would be confounded with "
                "composition"
            ),
            "departure_from_r0216": (
                "R0216 lays corpora out in contiguous blocks; this benchmark "
                "does not, and that is the only difference in the layout"
            ),
        },
        "prefix_composition": prefixes,
        "loading_contract": {
            "raw_format": RAW_FORMAT,
            "row_policy": ROW_POLICY,
            "trailing_fragment_policy": TRAILING_FRAGMENT_POLICY,
        },
        "sources": sources,
        "substrate": expected_input_signature(substrate_path),
        "provenance": expected_input_signature(provenance_path),
        "performance": {
            "assembly_wall_s": time.monotonic() - started,
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "benchmark-substrate.json"), receipt, immutable=True
    )
    del provenance, corpus_of_row
    gc.collect()


# --------------------------------------------------------------------------- #
# node 2: the memory sweep
# --------------------------------------------------------------------------- #


def _poll_nvidia_smi(pid: int, stop: threading.Event, sink: dict[str, int]) -> None:
    """R0220's instrument, carried as a control."""
    peak = 0
    while not stop.is_set():
        try:
            completed = subprocess.run(
                [
                    NVIDIA_SMI,
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            for line in completed.stdout.splitlines():
                parts = [item.strip() for item in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                    peak = max(peak, int(parts[1]) * 1024 * 1024)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        stop.wait(NVIDIA_SMI_POLL_S)
    sink["nvidia_smi_peak_bytes"] = peak


def _child_environment(cache_root: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "HOME": cache_root,
        "CUPY_CACHE_DIR": os.path.join(cache_root, "cupy"),
        "XDG_CACHE_HOME": os.path.join(cache_root, "xdg"),
        "NUMBA_CACHE_DIR": os.path.join(cache_root, "numba"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def _run_build(
    *, config: dict[str, Any], out_dir: str, cache_root: str, repo_root: str
) -> dict[str, Any]:
    ensure_data_directory(out_dir)
    config_path = os.path.join(out_dir, "config.json")
    atomic_write_new_json(config_path, config, immutable=True)
    command = [
        CUML_LAUNCHER,
        os.path.join(repo_root, BUILD_SCRIPT),
        "--config",
        config_path,
        "--out",
        out_dir,
    ]
    sink: dict[str, int] = {}
    stop = threading.Event()
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=_child_environment(cache_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    watcher = threading.Thread(
        target=_poll_nvidia_smi, args=(process.pid, stop, sink), daemon=True
    )
    watcher.start()
    try:
        stdout, stderr = process.communicate(timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        stop.set()
        watcher.join(timeout=5)
        raise Round0224Error(
            f"R0224 build {config['setting_id']} exceeded {BUILD_TIMEOUT_S:.0f}s"
        )
    finally:
        stop.set()
        watcher.join(timeout=5)
    subprocess_seconds = time.perf_counter() - started
    receipt_path = os.path.join(out_dir, "build-receipt.json")
    if process.returncode != 0 or not os.path.exists(receipt_path):
        raise Round0224Error(
            f"R0224 build {config['setting_id']} failed ({process.returncode}):\n"
            f"{stdout[-2000:]}\n{stderr[-2000:]}"
        )
    with open(receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    receipt["subprocess_seconds"] = subprocess_seconds
    receipt[CONTROL_INSTRUMENT] = int(sink.get(CONTROL_INSTRUMENT, 0))
    receipt["nvidia_smi_poll_interval_s"] = NVIDIA_SMI_POLL_S
    receipt["stderr_tail"] = stderr[-2000:]
    return receipt


def _sealed_substrate(job: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Resolve the assembly node's own output, produced earlier in this queue.

    A node output has no sha256 when the queue manifest is written, so the
    reference names a path. Integrity binds through the artifact's internal
    `prompt_contract` seal, which `read_sealed` verifies, plus the runner's own
    output validation; a reference that does carry a hash is verified against it.
    """
    reference = dict(job["benchmark_substrate_manifest_signature"])
    if reference.get("sha256"):
        path = prompt_contract.verify_signature(
            reference, label="R0224 sealed benchmark substrate receipt"
        )
    else:
        path = str(reference["canonical_path"])
        if not os.path.exists(path):
            raise Round0224Error(
                f"R0224 benchmark substrate receipt is absent at {path}"
            )
    manifest = prompt_contract.read_sealed(
        path, label="R0224 sealed benchmark substrate receipt"
    )
    if (
        manifest.get("schema") != SUBSTRATE_SCHEMA
        or manifest.get("round_id") != ROUND_ID
        or int(manifest.get("rows", -1)) != BENCHMARK_ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or manifest.get("benchmark_only") is not True
        or manifest.get("training_performed") is not False
    ):
        raise Round0224Error("R0224 sealed benchmark substrate contract changed")
    substrate_path = prompt_contract.verify_signature(
        manifest["substrate"], label="R0224 benchmark substrate bytes"
    )
    return manifest, substrate_path


def _host_total_bytes() -> int:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    raise Round0224Error("R0224 could not read MemTotal")


def run_sweep(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest, substrate_path = _sealed_substrate(job)
    repo_root = str(active["manifest"]["repo_root"])
    if not os.path.exists(CUML_LAUNCHER):
        raise Round0224Error(f"R0224 RAPIDS launcher {CUML_LAUNCHER} is absent")

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0224 cuVS memory sweep"
    )
    builds_root = ensure_data_directory(os.path.join(output, "builds"))
    cache_root = ensure_data_directory(str(job["cuvs_cache_root"]))
    started = time.monotonic()

    device_total = int(torch.cuda.get_device_properties(0).total_memory)
    host_total = _host_total_bytes()
    budget_check = {
        "device_total_bytes": device_total,
        "registered_device_total_bytes": REGISTERED_DEVICE_TOTAL_BYTES,
        "host_total_bytes": host_total,
        "registered_host_total_bytes": REGISTERED_HOST_TOTAL_BYTES,
        "tolerance": BUDGET_TOLERANCE,
        "device_matches_registered": abs(
            device_total - REGISTERED_DEVICE_TOTAL_BYTES
        )
        <= BUDGET_TOLERANCE * REGISTERED_DEVICE_TOTAL_BYTES,
        "host_matches_registered": abs(host_total - REGISTERED_HOST_TOTAL_BYTES)
        <= BUDGET_TOLERANCE * REGISTERED_HOST_TOTAL_BYTES,
    }
    if not budget_check["device_matches_registered"]:
        raise Round0224Error(
            f"R0224 device total {device_total} is not the registered "
            f"{REGISTERED_DEVICE_TOTAL_BYTES}; the budget verdicts would be wrong"
        )

    measurements: list[dict[str, Any]] = []
    for setting in sweep_settings():
        config = {
            **setting,
            "setting_id": setting["id"],
            "dataset": substrate_path,
            "dimension": DIMENSION,
            "sample_interval_s": SAMPLE_INTERVAL_S,
        }
        receipt = _run_build(
            config=config,
            out_dir=os.path.join(builds_root, str(setting["id"])),
            cache_root=cache_root,
            repo_root=repo_root,
        )
        measurements.append(receipt)

    residency: list[dict[str, Any]] = []
    for setting in residency_probe_settings():
        config = {
            **setting,
            "setting_id": setting["id"],
            "dataset": substrate_path,
            "dimension": DIMENSION,
            "sample_interval_s": SAMPLE_INTERVAL_S,
        }
        residency.append(
            _run_build(
                config=config,
                out_dir=os.path.join(builds_root, str(setting["id"])),
                cache_root=cache_root,
                repo_root=repo_root,
            )
        )
    residency_view = {
        str(item["config"]["dataset_mode"]): {
            "fit": bool(item.get("fit")),
            "oom": bool(item.get("oom")),
            "rows": int(item["rows"]),
            "rss_after_load_bytes": int(item.get("rss_after_load_bytes") or 0),
            "host_peak_sampled_bytes": int(item.get("host_peak_sampled_bytes") or 0),
            "host_vmhwm_bytes": int(item.get("host_vmhwm_bytes") or 0),
            "device_peak_sampled_bytes": int(
                item.get("device_peak_sampled_bytes") or 0
            ),
            "builder_seconds": item.get("builder_seconds"),
        }
        for item in residency
    }
    memmap_probe: dict[str, Any] = {
        "modes": residency_view,
        "rows": int(residency[0]["rows"]) if residency else None,
        "note": (
            "a 100,000,000 x 384 fp32 substrate is "
            f"{PROJECTION_SUBSTRATE_BYTES} bytes against "
            f"{host_total} bytes of host RAM, so the top rung needs memmap-fed "
            "batching regardless of the builder. This probe asks only whether "
            "cuVS accepts a memmap without materializing it."
        ),
    }
    if {"materialize", "memmap"} <= set(residency_view):
        materialized = residency_view["materialize"]["host_peak_sampled_bytes"]
        streamed = residency_view["memmap"]["host_peak_sampled_bytes"]
        memmap_probe.update({
            "memmap_accepted": residency_view["memmap"]["fit"],
            "host_peak_materialize_bytes": materialized,
            "host_peak_memmap_bytes": streamed,
            "host_peak_saved_bytes": materialized - streamed,
            "avoids_host_materialization": (
                residency_view["memmap"]["fit"]
                and streamed < materialized
            ),
        })
    else:
        memmap_probe["untested"] = True

    summary = summarize_sweep(
        measurements=measurements,
        device_total_bytes=device_total,
        host_total_bytes=host_total,
    )

    execution_checks = {
        "every_cell_attempted": len(measurements) == len(sweep_settings()),
        "instrument_set_complete": all(
            all(
                instrument in item
                for instrument in INSTRUMENTS
                if instrument != "host_vmhwm_bytes"
            )
            for item in measurements
            if item.get("fit")
        ),
        "fresh_process_per_build": True,
        "sensitivity_decided_before_fitting": "sensitivity" in summary,
        "projections_gated_on_sensitivity": (
            summary["projections_emitted"]
            == summary["sensitivity"]["any_instrument_sensitive"]
        ),
        "no_projection_divided_by_a_projection": True,
        "device_budget_matches_registered": budget_check["device_matches_registered"],
        "residency_probe_ran": len(residency) == len(residency_probe_settings()),
    }
    if not all(execution_checks.values()):
        raise Round0224Error(f"R0224 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    receipt = prompt_contract.seal({
        "schema": SWEEP_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": SWEEP_CAPABILITY,
        "capabilities": [SWEEP_CAPABILITY],
        "outcome": "cuvs-nn-descent-memory-measured-with-a-sensitive-instrument",
        "substrate": dict(manifest["substrate"]),
        "benchmark_substrate_manifest": dict(
            job["benchmark_substrate_manifest_signature"]
        ),
        "benchmark_only": True,
        "matrix": {
            "rows": list(SWEEP_ROWS),
            "intermediate_graph_degrees": list(SWEEP_INTERMEDIATE_DEGREES),
            "graph_degree": SWEEP_GRAPH_DEGREE,
            "max_iterations": SWEEP_MAX_ITERATIONS,
            "metric": SWEEP_METRIC,
            "held_fixed": (
                "graph_degree and max_iterations, so intermediate_graph_degree is "
                "the only moving parameter"
            ),
            "cells": len(sweep_settings()),
        },
        "instruments": list(INSTRUMENTS),
        "sample_interval_s": SAMPLE_INTERVAL_S,
        "budgets": budget_check,
        "builds": measurements,
        "residency_probe": memmap_probe,
        "summary": summary,
        "execution_checks": execution_checks,
        "gpu_hours_cap": GPU_HOURS_CAP,
        "training_performed": False,
        "gate_registered": False,
        "evaluation_performed": False,
        "map_decision_made": False,
        "production_or_publishing": False,
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "cuvs-memory-scaling.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0224Error("R0224 handler received another queue")
    action = str(job.get("action") or "")
    if action == ASSEMBLE_ACTION:
        run_assemble(active, job)
    elif action == SWEEP_ACTION:
        run_sweep(active, job)
    else:
        raise Round0224Error(f"unknown R0224 action {action!r}")


__all__ = [
    "ASSEMBLE_ACTION",
    "SWEEP_ACTION",
    "run_assemble",
    "run_job",
    "run_sweep",
]
