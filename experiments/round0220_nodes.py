"""Execute R0220 — qualify cuVS as a k15 all-neighbours graph builder.

Two GPU nodes.

`rebuild_exact_k15_truth` recomputes the exact fp32 cosine top-15 over all
2,000,000 rows of R0216's sealed `queue-correction-3` substrate under R0216's
identical blocked law, because R0216 persisted only the symmetrised fuzzy edge
file and not its `nbr`/`dist` arrays. Before the truth is usable it must
reproduce the sealed adjacency on a pre-registered probe (seed 220, 65,536
rows) under the tie-aware measure review-0216-02 used; otherwise the node
aborts and no cuVS number is produced.

`qualify_cuvs_k15_builder` then builds cuVS graphs in a separate RAPIDS-env
subprocess (five settings plus a size-scaling series), measures recall@15
against that truth over the **full** 2,000,000 rows, reports structural
validity including the R0215 zero-degree tripwire, and fits a measured scaling
exponent that a clearly-labelled 100M projection is derived from.

No training, no map, no gate, no capability consumption beyond R0216.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0220_cuvs_qualification import (
    CAPABILITY,
    CUVS_METRIC,
    DIMENSION,
    EDGES_SHA256,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    METRIC_EQUIVALENCE,
    OUT_OF_CORE_MODULE,
    PROJECTION_ROWS,
    QUALIFICATION_SCHEMA,
    QUERY_BLOCK,
    REQUIRED_CUVS_MODULES,
    ROUND_ID,
    ROWS,
    Round0220Error,
    SCALING_ROWS,
    SCALING_SETTING_ID,
    SEALED_DIRECTED_EDGES,
    SEARCH_BLOCK,
    SUBSTRATE_SHA256,
    SWEEP,
    TIE_TOLERANCE,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
    TRUTH_SCHEMA,
    graph_validity,
    power_law,
    project_cost,
    setting,
    strict_containment_rows,
    summarize,
    tie_aware_rows,
    validate_truth_probe,
)


TRUTH_ACTION = "rebuild_exact_k15_truth"
QUALIFY_ACTION = "qualify_cuvs_k15_builder"

#: The RAPIDS launcher documented in `/data/latent-basemap/cuml-env/SETUP.md`.
CUML_LAUNCHER = "/data/latent-basemap/cuml_py"
BUILD_SCRIPT = "basemap/round0220_cuvs_build.py"
BUILD_TIMEOUT_S = 5_400.0
NVIDIA_SMI = "/usr/bin/nvidia-smi"
MEMORY_POLL_INTERVAL_S = 0.1
EVAL_BLOCK = 16_384


def _sealed_graph_manifest(job: Mapping[str, Any]) -> dict[str, Any]:
    path = prompt_contract.verify_signature(
        job["graph_manifest_signature"], label="R0216 sealed substrate+graph receipt"
    )
    manifest = prompt_contract.read_sealed(path, label="R0216 sealed substrate+graph receipt")
    checks = manifest.get("graph_checks") or {}
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != GRAPH_SOURCE_ROUND_ID
        or manifest.get("capability") != GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or int(checks.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
        or int(checks.get("zero_degree_rows", -1)) != 0
    ):
        raise Round0220Error("R0216 sealed substrate+graph contract changed")
    substrate = dict(manifest["substrate"])
    graph = dict(manifest["graph"])
    if substrate.get("sha256") != SUBSTRATE_SHA256:
        raise Round0220Error("R0216 substrate hash is not the sealed one")
    if graph.get("sha256") != EDGES_SHA256:
        raise Round0220Error("R0216 edge-file hash is not the sealed one")
    prompt_contract.verify_signature(substrate, label="R0216 substrate bytes")
    prompt_contract.verify_signature(graph, label="R0216 exact k15 fuzzy edges")
    return manifest


def _exact_top_k(torch: Any, tensor: Any, *, rows: int, k: int) -> tuple[Any, Any]:
    """R0216's blocked fp32 cosine search, self-excluded, over `rows` queries."""
    ids = torch.empty((rows, k), dtype=torch.int32, device="cpu")
    cosines = torch.empty((rows, k), dtype=torch.float32, device="cpu")
    width = k + 1
    for qs in range(0, rows, QUERY_BLOCK):
        qe = min(qs + QUERY_BLOCK, rows)
        queries = tensor[qs:qe]
        best_s = torch.full((qe - qs, width), -float("inf"), device=tensor.device)
        best_i = torch.full((qe - qs, width), -1, device=tensor.device, dtype=torch.int64)
        for cs in range(0, rows, SEARCH_BLOCK):
            ce = min(cs + SEARCH_BLOCK, rows)
            sims = queries @ tensor[cs:ce].T
            take = min(width, ce - cs)
            top_s, top_i = torch.topk(sims, take, dim=1)
            merged_s = torch.cat([best_s, top_s], 1)
            merged_i = torch.cat([best_i, top_i.to(torch.int64) + cs], 1)
            order = torch.argsort(merged_s, dim=1, descending=True)[:, :width]
            best_s = torch.gather(merged_s, 1, order)
            best_i = torch.gather(merged_i, 1, order)
        self_ids = torch.arange(qs, qe, device=tensor.device, dtype=torch.int64)[:, None]
        keep = best_i != self_ids
        # Guarantee exactly `k` survivors: drop the last column when no self hit.
        cumulative = torch.cumsum(keep.to(torch.int32), dim=1)
        keep &= cumulative <= k
        ids[qs:qe] = best_i[keep].view(qe - qs, k).to(torch.int32).cpu()
        cosines[qs:qe] = best_s[keep].view(qe - qs, k).cpu()
    return ids, cosines


def _csr_bounds(sources: np.ndarray, rows: int) -> np.ndarray:
    if np.any(np.diff(sources) < 0):
        raise Round0220Error("R0216 edge sources are not sorted; CSR view is invalid")
    return np.searchsorted(sources, np.arange(rows + 1, dtype=sources.dtype))


def run_truth(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest = _sealed_graph_manifest(job)
    substrate_path = str(manifest["substrate"]["canonical_path"])
    edges_path = str(manifest["graph"]["canonical_path"])

    started = time.perf_counter()
    host = np.load(substrate_path, mmap_mode="r")
    if host.shape != (ROWS, DIMENSION):
        raise Round0220Error(f"substrate shape {host.shape} is not ({ROWS}, {DIMENSION})")
    device = torch.device("cuda")
    tensor = torch.from_numpy(np.array(host, dtype=np.float32, order="C", copy=True)).to(device)
    load_seconds = time.perf_counter() - started

    search_started = time.perf_counter()
    ids, cosines = _exact_top_k(torch, tensor, rows=ROWS, k=GRAPH_K)
    search_seconds = time.perf_counter() - search_started
    ids_np = ids.numpy()
    cos_np = cosines.numpy()
    if not np.all(np.isfinite(cos_np)):
        raise Round0220Error("recomputed truth contains non-finite cosines")
    if int(ids_np.min()) < 0 or int(ids_np.max()) >= ROWS:
        raise Round0220Error("recomputed truth contains out-of-range ids")

    # --- registered probe: does the recomputation reproduce R0216's graph? ---
    rng = np.random.RandomState(TRUTH_PROBE_SEED)
    probe = np.sort(rng.choice(ROWS, TRUTH_PROBE_ROWS, replace=False)).astype(np.int64)
    with np.load(edges_path) as bundle:
        sources = np.asarray(bundle["sources"])
        targets = np.asarray(bundle["targets"])
    bounds = _csr_bounds(sources, ROWS)
    kth = cos_np[probe, GRAPH_K - 1].astype(np.float64)

    tie_values = np.empty(probe.size, dtype=np.float64)
    strict_values = np.empty(probe.size, dtype=np.float64)
    probe_started = time.perf_counter()
    for start in range(0, probe.size, 2_048):
        stop = min(start + 2_048, probe.size)
        rows_here = probe[start:stop]
        members = [
            np.unique(targets[bounds[row] : bounds[row + 1]].astype(np.int64))
            for row in rows_here
        ]
        counts = np.array([item.size for item in members], dtype=np.int64)
        flat = np.concatenate(members) if counts.sum() else np.zeros(0, dtype=np.int64)
        owner = np.repeat(np.arange(rows_here.size, dtype=np.int64), counts)
        if flat.size:
            flat_t = torch.from_numpy(flat).to(device)
            owner_rows = torch.from_numpy(rows_here[owner]).to(device)
            member_cos = (
                (tensor[flat_t] * tensor[owner_rows]).sum(dim=1).to(torch.float64)
            )
            member_cos_np = member_cos.cpu().numpy()
        else:
            member_cos_np = np.zeros(0, dtype=np.float64)
        thresholds = kth[start:stop][owner] - TIE_TOLERANCE
        valid = np.zeros(rows_here.size, dtype=np.int64)
        np.add.at(valid, owner, (member_cos_np >= thresholds).astype(np.int64))
        tie_values[start:stop] = np.minimum(valid, GRAPH_K) / float(GRAPH_K)
        for offset, member_set in enumerate(members):
            truth_row = ids_np[rows_here[offset]].astype(np.int64)
            hits = int(np.isin(truth_row, member_set, assume_unique=True).sum())
            strict_values[start + offset] = hits / float(GRAPH_K)
    probe_seconds = time.perf_counter() - probe_started

    tie_summary = summarize(tie_values, label="truth probe tie-aware validity")
    strict_summary = summarize(strict_values, label="truth probe strict containment")
    probe_checks = validate_truth_probe(tie_aware=tie_summary, strict=strict_summary)

    output = create_fresh_directory(str(job["outputs"][0]), label="R0220 exact k15 truth")
    ids_path = os.path.join(output, "truth-k15-ids.i32.npy")
    cos_path = os.path.join(output, "truth-k15-cos.f32.npy")
    np.save(ids_path, ids_np)
    np.save(cos_path, cos_np)
    # Fail closed in the producing node: R0220's first queue shipped a probe
    # variable that shadowed `cos_np`, so the cosine file held one probe block
    # instead of the truth. The consumer caught it six GPU-minutes later.
    for path, expected_dtype in ((ids_path, np.int32), (cos_path, np.float32)):
        written = np.load(path, mmap_mode="r")
        if written.shape != (ROWS, GRAPH_K) or written.dtype != expected_dtype:
            raise Round0220Error(
                f"{os.path.basename(path)} is {written.shape}/{written.dtype}, "
                f"not ({ROWS}, {GRAPH_K})/{np.dtype(expected_dtype).name}"
            )
    receipt = prompt_contract.seal({
        "schema": TRUTH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "rows": ROWS,
        "dimension": DIMENSION,
        "k": GRAPH_K,
        "search_law": {
            "kernel": "blocked fp32 cosine top-(k+1), self-excluded",
            "search_block": SEARCH_BLOCK,
            "query_block": QUERY_BLOCK,
            "identical_to_round": GRAPH_SOURCE_ROUND_ID,
            "why_recomputed": (
                "R0216 persisted only the symmetrised fuzzy edge file; its "
                "nbr/dist arrays were never saved (review-0216-02, correction 3)"
            ),
        },
        "source": {
            "graph_manifest": dict(job["graph_manifest_signature"]),
            "substrate": dict(manifest["substrate"]),
            "edges": dict(manifest["graph"]),
        },
        "probe": {
            **probe_checks,
            "tie_aware": tie_summary,
            "strict": strict_summary,
            "tie_tolerance": TIE_TOLERANCE,
            "note": (
                "tie-aware validity of R0216's sealed adjacency against this "
                "recomputed truth, the measure review-0216-02 used; the "
                "adjacency is symmetrised and carries reverse edges, so the "
                "per-row count is capped at k"
            ),
        },
        "outputs": {
            "ids": expected_input_signature(ids_path),
            "cosines": expected_input_signature(cos_path),
        },
        "performance": {
            "load_seconds": load_seconds,
            "exact_search_seconds": search_seconds,
            "probe_seconds": probe_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "training_performed": False,
        "gate_registered": False,
        "map_quality_claim_available": False,
    })
    atomic_write_new_json(os.path.join(output, "truth-rebuild.json"), receipt, immutable=True)


def _poll_peak_memory(pid: int, stop: threading.Event, sink: dict[str, int]) -> None:
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
        stop.wait(MEMORY_POLL_INTERVAL_S)
    sink["peak_gpu_bytes"] = peak


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


def _run_cuvs_build(
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
        target=_poll_peak_memory, args=(process.pid, stop, sink), daemon=True
    )
    watcher.start()
    try:
        stdout, stderr = process.communicate(timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        stop.set()
        watcher.join(timeout=5)
        raise Round0220Error(f"cuVS build {config['setting_id']} exceeded its timeout")
    finally:
        stop.set()
        watcher.join(timeout=5)
    subprocess_seconds = time.perf_counter() - started
    if process.returncode != 0:
        raise Round0220Error(
            f"cuVS build {config['setting_id']} failed ({process.returncode}):\n"
            f"{stdout}\n{stderr}"
        )
    receipt_path = os.path.join(out_dir, "build-receipt.json")
    with open(receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    receipt["subprocess_seconds"] = subprocess_seconds
    receipt["peak_gpu_bytes_sampled"] = int(sink.get("peak_gpu_bytes", 0))
    receipt["peak_gpu_sampling"] = {
        "source": "nvidia-smi --query-compute-apps=used_gpu_memory against the child pid",
        "interval_seconds": MEMORY_POLL_INTERVAL_S,
        "includes_cuda_context": True,
        "note": "a sampled maximum can miss a spike shorter than the interval",
    }
    receipt["stderr_tail"] = stderr[-2000:]
    return receipt


def _evaluate(
    torch: Any,
    tensor: Any,
    *,
    graph: np.ndarray,
    truth_ids: np.ndarray,
    truth_cos: np.ndarray,
) -> dict[str, Any]:
    device = tensor.device
    leading = np.ascontiguousarray(graph[:, :GRAPH_K])
    kth = truth_cos[:, GRAPH_K - 1].astype(np.float64)
    candidate_cos = np.empty((ROWS, GRAPH_K), dtype=np.float64)
    for start in range(0, ROWS, EVAL_BLOCK):
        stop = min(start + EVAL_BLOCK, ROWS)
        block = torch.from_numpy(leading[start:stop].astype(np.int64)).to(device)
        gathered = tensor[block.reshape(-1)].reshape(stop - start, GRAPH_K, DIMENSION)
        queries = tensor[start:stop].unsqueeze(2)
        candidate_cos[start:stop] = (
            torch.bmm(gathered, queries).squeeze(2).to(torch.float64).cpu().numpy()
        )
    strict = strict_containment_rows(leading, truth_ids)
    tie = tie_aware_rows(candidate_cos, leading, kth)
    full = strict_containment_rows(graph, truth_ids)
    return {
        "strict_recall_at_15": summarize(strict, label="strict recall@15"),
        "tie_aware_recall_at_15": summarize(tie, label="tie-aware recall@15"),
        "strict_containment_full_width": summarize(
            full, label="strict containment in the full returned width"
        ),
        "validity": graph_validity(leading, rows=ROWS),
        "graph_width": int(graph.shape[1]),
        "leading_columns_used": GRAPH_K,
        "ordering_assumption": (
            "cuVS returns each row sorted by increasing distance, so the "
            "leading 15 columns are the builder's k15 graph"
        ),
    }


def run_qualify(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    manifest = _sealed_graph_manifest(job)
    substrate_path = str(manifest["substrate"]["canonical_path"])
    truth_receipt_path = str(job["truth_receipt"])
    truth = prompt_contract.read_sealed(truth_receipt_path, label="R0220 exact k15 truth")
    if truth.get("schema") != TRUTH_SCHEMA or truth.get("round_id") != ROUND_ID:
        raise Round0220Error("R0220 truth receipt contract changed")
    if not truth["probe"]["passed"]:
        raise Round0220Error("R0220 truth receipt did not pass its registered probe")
    ids_path = prompt_contract.verify_signature(
        truth["outputs"]["ids"], label="R0220 truth ids"
    )
    cos_path = prompt_contract.verify_signature(
        truth["outputs"]["cosines"], label="R0220 truth cosines"
    )

    output = create_fresh_directory(str(job["outputs"][0]), label="R0220 cuVS qualification")
    builds_root = ensure_data_directory(os.path.join(output, "builds"))
    cache_root = ensure_data_directory(str(job["cuvs_cache_root"]))
    repo_root = str(active["manifest"]["repo_root"])
    if not os.path.exists(CUML_LAUNCHER):
        raise Round0220Error(f"RAPIDS launcher {CUML_LAUNCHER} is absent")

    # --- phase 1: every cuVS build, in the RAPIDS env, torch not yet loaded ---
    sweep_receipts: list[dict[str, Any]] = []
    for item in SWEEP:
        config = {
            **item,
            "setting_id": item["id"],
            "dataset": substrate_path,
            "rows": ROWS,
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "metric": CUVS_METRIC,
            "save_graph": True,
        }
        sweep_receipts.append(
            _run_cuvs_build(
                config=config,
                out_dir=os.path.join(builds_root, str(item["id"])),
                cache_root=cache_root,
                repo_root=repo_root,
            )
        )

    scaling_receipts: list[dict[str, Any]] = []
    base = setting(SCALING_SETTING_ID)
    for rows in SCALING_ROWS:
        if rows == ROWS:
            match = next(r for r in sweep_receipts if r["setting_id"] == SCALING_SETTING_ID)
            scaling_receipts.append({
                "rows": ROWS,
                "reused_from_sweep": True,
                "builder_seconds": float(match["builder_seconds"]),
                "peak_gpu_bytes_sampled": int(match["peak_gpu_bytes_sampled"]),
            })
            continue
        config = {
            **base,
            "setting_id": f"{SCALING_SETTING_ID}-n{rows}",
            "dataset": substrate_path,
            "rows": int(rows),
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "metric": CUVS_METRIC,
            "save_graph": False,
        }
        receipt = _run_cuvs_build(
            config=config,
            out_dir=os.path.join(builds_root, f"scaling-n{rows}"),
            cache_root=cache_root,
            repo_root=repo_root,
        )
        scaling_receipts.append({
            "rows": int(rows),
            "reused_from_sweep": False,
            "builder_seconds": float(receipt["builder_seconds"]),
            "peak_gpu_bytes_sampled": int(receipt["peak_gpu_bytes_sampled"]),
        })

    modules = set(sweep_receipts[0].get("cuvs_neighbors_modules") or [])
    missing = [name for name in REQUIRED_CUVS_MODULES if name not in modules]
    if missing:
        raise Round0220Error(f"cuVS is missing required modules: {missing}")

    # --- phase 2: recall against exact truth, over the full population ---
    device = torch.device("cuda")
    host = np.load(substrate_path, mmap_mode="r")
    tensor = torch.from_numpy(np.array(host, dtype=np.float32, order="C", copy=True)).to(device)
    truth_ids = np.load(ids_path)
    truth_cos = np.load(cos_path)
    if truth_ids.shape != (ROWS, GRAPH_K) or truth_cos.shape != (ROWS, GRAPH_K):
        raise Round0220Error("R0220 truth arrays have the wrong shape")

    results: list[dict[str, Any]] = []
    for item, receipt in zip(SWEEP, sweep_receipts):
        graph = np.load(str(receipt["graph_path"]))
        measurement = _evaluate(
            torch, tensor, graph=graph, truth_ids=truth_ids, truth_cos=truth_cos
        )
        results.append({
            "setting": dict(item),
            "build": {
                key: receipt[key]
                for key in (
                    "algo",
                    "rows",
                    "load_seconds",
                    "warmup_seconds",
                    "build_seconds",
                    "search_seconds",
                    "extract_seconds",
                    "builder_seconds",
                    "subprocess_seconds",
                    "peak_gpu_bytes_sampled",
                    "peak_gpu_sampling",
                    "graph_shape",
                    "cuvs_version",
                )
            },
            "measurement": measurement,
        })

    fit = power_law(
        [item["rows"] for item in scaling_receipts],
        [item["builder_seconds"] for item in scaling_receipts],
    )
    memory_fit = power_law(
        [item["rows"] for item in scaling_receipts],
        [max(item["peak_gpu_bytes_sampled"], 1) for item in scaling_receipts],
    )
    projection = project_cost(fit, rows=PROJECTION_ROWS)

    receipt = prompt_contract.seal({
        "schema": QUALIFICATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY],
        "rows_evaluated": ROWS,
        "k": GRAPH_K,
        "metric": CUVS_METRIC,
        "metric_equivalence": METRIC_EQUIVALENCE,
        "truth": {
            "receipt": expected_input_signature(truth_receipt_path),
            "identity_sha256": truth["identity_sha256"],
            "probe": truth["probe"],
        },
        "source": {
            "graph_manifest": dict(job["graph_manifest_signature"]),
            "substrate": dict(manifest["substrate"]),
            "edges": dict(manifest["graph"]),
        },
        "cuvs": {
            "version": sweep_receipts[0].get("cuvs_version"),
            "neighbors_modules": sorted(modules),
            "out_of_core_module": OUT_OF_CORE_MODULE,
            "out_of_core_module_available": OUT_OF_CORE_MODULE in modules,
            "launcher": CUML_LAUNCHER,
        },
        "sweep": results,
        "scaling": {
            "setting_id": SCALING_SETTING_ID,
            "points": scaling_receipts,
            "wall_fit": fit,
            "peak_gpu_fit": memory_fit,
        },
        "projection_100m": projection,
        "gate_registered": False,
        "map_quality_claim_available": False,
        "training_performed": False,
        "evaluation_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    })
    atomic_write_new_json(
        os.path.join(output, "cuvs-qualification.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0220Error("unknown R0220 queue")
    action = str(job.get("action") or "")
    if action == TRUTH_ACTION:
        run_truth(active, job)
    elif action == QUALIFY_ACTION:
        run_qualify(active, job)
    else:
        raise Round0220Error(f"unknown R0220 action {action!r}")


__all__ = [
    "BUILD_SCRIPT",
    "CUML_LAUNCHER",
    "QUALIFY_ACTION",
    "TRUTH_ACTION",
    "run_job",
    "run_qualify",
    "run_truth",
]


if __name__ == "__main__":
    raise SystemExit(f"{sys.argv[0]} is a queue handler module, not a script")
