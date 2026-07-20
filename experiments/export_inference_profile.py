#!/usr/bin/env python3
"""Round 0026 ONNX export and inference profile for MiniLM projectors."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)


ROUND_ID = "0026"
SEED = 20260726
DIM = 384
QUERY_POOL = "/data/latent-basemap/track1/minilm_queries.npy"
INPUT_PACK = "/data/latent-basemap/runs/round-0013/30m-input-pack-v1.json"
PYTHON_DEPS = "/data/latent-basemap/runs/round-0026/python-deps"
GPU_UUID = "GPU-2c4d2a68-2646-901a-e61c-fbc61f5c9072"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    width: int
    source_round: str
    model_path: str
    model_sha256: str
    required: bool = True
    release_note: str | None = None


MODEL_SPECS = [
    ModelSpec(
        name="r0019-h2048",
        width=2048,
        source_round="0019",
        model_path="/data/latent-basemap/runs/round-0019/queue/artifacts/train/model.pt",
        model_sha256="2f5eb27582e26735491b4bed9417cf27992bb213ef942e433a5bcba97d481a32",
        required=True,
    ),
    ModelSpec(
        name="r0024-h1024",
        width=1024,
        source_round="0024",
        model_path="/data/latent-basemap/runs/round-0024/queue/artifacts/h1024/train/model.pt",
        model_sha256="6b17c6b39b7eb346ccbe159642cf8bd07a124d928e4e69b639d41643d8179229",
        required=False,
        release_note="review-0024 releases this exact model hash for R0026 inference profiling only",
    ),
    ModelSpec(
        name="r0024-h4096",
        width=4096,
        source_round="0024",
        model_path="/data/latent-basemap/runs/round-0024/queue/artifacts/h4096/train/model.pt",
        model_sha256="7ee41b95a1f2844a2790f3abdc7521c27f9dabef3666565faafefecc363b58a0",
        required=False,
        release_note="review-0024 releases this exact model hash for R0026 inference profiling only",
    ),
]


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_deps() -> dict[str, str]:
    import onnx
    import onnxruntime

    return {
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "numpy": np.__version__,
    }


def available_model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for spec in MODEL_SPECS:
        if not os.path.exists(spec.model_path):
            if spec.required:
                raise FileNotFoundError(spec.model_path)
            continue
        signature = expected_input_signature(spec.model_path)
        if signature["sha256"] != spec.model_sha256:
            if spec.required:
                raise RuntimeError(f"{spec.name} model hash mismatch")
            continue
        specs.append(spec)
    return specs


def _materialized_chunks() -> list[dict[str, Any]]:
    manifest = _read_json(INPUT_PACK)
    members = manifest["capability_payload"]["materialized_fp16"]["ordered_members"]
    out = []
    for item in members:
        path = os.path.realpath(item["path"])
        out.append(
            {
                "path": path,
                "rows": int(item["global_row_stop"] - item["global_row_start"]),
                "global_row_start": int(item["global_row_start"]),
                "global_row_stop": int(item["global_row_stop"]),
                "size_bytes": int(item["size_bytes"]),
                "sha256": str(item["sha256"]),
            }
        )
    if len(out) != 30 or out[0]["global_row_start"] != 0 or out[-1]["global_row_stop"] != 30_000_000:
        raise RuntimeError("R0026 materialized training chunks do not cover 30M rows")
    return out


def _sample_rows() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    queries = np.load(QUERY_POOL, mmap_mode="r", allow_pickle=False)
    if queries.ndim != 2 or queries.shape[1] != DIM or queries.dtype != np.dtype("<f4"):
        raise RuntimeError("R0026 query pool shape/dtype changed")
    query_rows = np.sort(rng.choice(len(queries), 10_000, replace=False)).astype(np.int64)
    training_rows = np.sort(rng.choice(30_000_000, 10_000, replace=False)).astype(np.int64)
    return query_rows, training_rows


def _gather_training_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    chunks = _materialized_chunks()
    out = np.empty((len(rows), DIM), dtype=np.float32)
    for chunk in chunks:
        lo = int(chunk["global_row_start"])
        hi = int(chunk["global_row_stop"])
        positions = np.flatnonzero((rows >= lo) & (rows < hi))
        if not len(positions):
            continue
        array = np.load(chunk["path"], mmap_mode="r", allow_pickle=False)
        if array.shape != (hi - lo, DIM) or array.dtype != np.dtype("<f2"):
            raise RuntimeError(f"R0026 materialized chunk shape/dtype changed: {chunk['path']}")
        out[positions] = np.asarray(array[rows[positions] - lo], dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("R0026 training sample contains non-finite values")
    return out


def build_sample(output_root: str) -> dict[str, Any]:
    query_rows, training_rows = _sample_rows()
    queries = np.load(QUERY_POOL, mmap_mode="r", allow_pickle=False)
    query_vectors = np.asarray(queries[query_rows], dtype=np.float32)
    train_vectors = _gather_training_rows(training_rows)
    sample = np.ascontiguousarray(np.vstack([query_vectors, train_vectors]).astype(np.float32))
    if sample.shape != (20_000, DIM) or not np.isfinite(sample).all():
        raise RuntimeError("R0026 sample shape or finiteness guard failed")
    sample_ids = os.path.join(output_root, "sample-ids-v1.npz")
    atomic_save_new_npz(
        sample_ids,
        immutable=True,
        query_rows=query_rows,
        training_rows=training_rows,
    )
    return {
        "sample_vectors": sample,
        "query_rows": query_rows,
        "training_rows": training_rows,
        "sample_ids": expected_input_signature(sample_ids),
        "sample_vectors_sha256": ordered_array_sha256(sample),
        "query_rows_sha256": ordered_array_sha256(query_rows),
        "training_rows_sha256": ordered_array_sha256(training_rows),
        "sources": {
            "query_pool": expected_input_signature(QUERY_POOL),
            "input_pack": expected_input_signature(INPUT_PACK),
            "materialized_chunks": [
                {
                    "canonical_path": chunk["path"],
                    "kind": "file",
                    "bytes": chunk["size_bytes"],
                    "sha256": chunk["sha256"],
                    "global_row_start": chunk["global_row_start"],
                    "global_row_stop": chunk["global_row_stop"],
                }
                for chunk in _materialized_chunks()
            ],
        },
    }


def _sample_receipt_fields(sample_info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sample_info.items()
        if key not in {"sources", "sample_vectors", "query_rows", "training_rows"}
    }


def _load_model(spec: ModelSpec, *, device: str):
    signature = expected_input_signature(spec.model_path)
    if signature["sha256"] != spec.model_sha256:
        raise RuntimeError(f"{spec.name} model bytes changed")
    from basemap.pumap.parametric_umap import ParametricUMAP

    wrapper = ParametricUMAP.load(spec.model_path, device=device)
    wrapper.model.eval()
    return wrapper


def _torch_forward(wrapper: Any, vectors: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    import torch

    out = np.empty((len(vectors), 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(vectors), batch_size):
            batch = torch.from_numpy(np.asarray(vectors[start : start + batch_size], dtype=np.float32)).to(device)
            coords = wrapper.model(batch).detach().cpu().numpy().astype(np.float32)
            if coords.shape != (len(batch), 2) or not np.isfinite(coords).all():
                raise RuntimeError("R0026 torch projection is non-finite or malformed")
            out[start : start + len(batch)] = coords
            del batch
        if device == "cuda":
            torch.cuda.synchronize()
    return out


def export_onnx(spec: ModelSpec, output_root: str, sample: np.ndarray) -> dict[str, Any]:
    import torch
    import onnx
    from onnx import TensorProto, numpy_helper

    wrapper = _load_model(spec, device="cpu")
    model = wrapper.model.float().eval()
    model_dir = os.path.join(output_root, spec.name)
    os.makedirs(model_dir, exist_ok=False)
    fp32_path = os.path.join(model_dir, f"{spec.name}.fp32.onnx")
    dummy = torch.from_numpy(np.ascontiguousarray(sample[:100].astype(np.float32)))
    torch.onnx.export(
        model,
        dummy,
        fp32_path,
        input_names=["x"],
        output_names=["coords"],
        dynamic_axes={"x": {0: "rows"}, "coords": {0: "rows"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(fp32_path)

    fp16_path = os.path.join(model_dir, f"{spec.name}.fp16.onnx")
    proto = onnx.load(fp32_path)
    for value in list(proto.graph.input) + list(proto.graph.output) + list(proto.graph.value_info):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type == TensorProto.FLOAT:
            tensor_type.elem_type = TensorProto.FLOAT16
    for initializer in proto.graph.initializer:
        if initializer.data_type == TensorProto.FLOAT:
            converted = numpy_helper.from_array(
                numpy_helper.to_array(initializer).astype(np.float16),
                initializer.name,
            )
            initializer.CopyFrom(converted)
    onnx.checker.check_model(proto)
    onnx.save(proto, fp16_path)

    return {
        "model": expected_input_signature(spec.model_path),
        "model_sha256_expected": spec.model_sha256,
        "onnx_fp32": expected_input_signature(fp32_path),
        "onnx_fp16": expected_input_signature(fp16_path),
        "params": int(sum(p.numel() for p in model.parameters())),
        "architecture": {
            "source_round": spec.source_round,
            "hidden_dimension": spec.width,
            "release_note": spec.release_note,
        },
    }


def _ort_session(path: str, *, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def _run_ort(session: Any, vectors: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: vectors})[0]


def _make_batch(sample: np.ndarray, batch_size: int, *, dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    if batch_size <= len(sample):
        return np.ascontiguousarray(sample[:batch_size].astype(dtype, copy=False))
    reps = int(np.ceil(batch_size / len(sample)))
    return np.ascontiguousarray(np.tile(sample, (reps, 1))[:batch_size].astype(dtype, copy=False))


def _latency_summary(times: list[float], rows: int) -> dict[str, Any]:
    ordered = sorted(float(value) for value in times)
    median = float(statistics.median(ordered))
    p99 = ordered[-1] if len(ordered) < 2 else float(np.percentile(ordered, 99))
    return {
        "repetitions": len(ordered),
        "rows": int(rows),
        "latency_seconds_p50": median,
        "latency_seconds_p99": p99,
        "rows_per_second_median": float(rows / median) if median > 0 else None,
    }


def _time_ort(session: Any, batch: np.ndarray) -> dict[str, Any]:
    rows = len(batch)
    repeats = 100 if rows == 1 else 20 if rows == 1000 else 1
    warmups = 10 if rows == 1 else 3 if rows == 1000 else 1
    for _ in range(warmups):
        _run_ort(session, batch)
    times: list[float] = []
    output_sha = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = _run_ort(session, batch)
        elapsed = time.perf_counter() - start
        if output.shape != (rows, 2) or not np.isfinite(output).all():
            raise RuntimeError("R0026 ONNX throughput output is non-finite or malformed")
        times.append(elapsed)
        if output_sha is None:
            output_sha = ordered_array_sha256(np.asarray(output, dtype=np.float32))
    return {**_latency_summary(times, rows), "output_sha256_first": output_sha}


def run_canary(*, output_root: str) -> dict[str, Any]:
    output_root = create_fresh_directory(output_root, label="Round 0026 canary output")
    deps = _ensure_deps()
    sample_info = build_sample(output_root)
    sample = sample_info.pop("sample_vectors")[:100]
    models = {}
    for spec in available_model_specs():
        export = export_onnx(spec, output_root, sample)
        wrapper = _load_model(spec, device="cpu")
        torch_coords = _torch_forward(wrapper, sample, batch_size=100, device="cpu")
        session = _ort_session(export["onnx_fp32"]["canonical_path"], threads=1)
        ort_coords = np.asarray(_run_ort(session, sample), dtype=np.float32)
        max_abs = float(np.max(np.abs(torch_coords - ort_coords)))
        if max_abs > 1e-3:
            raise RuntimeError(f"R0026 canary fp32 parity failed for {spec.name}: {max_abs}")
        models[spec.name] = {
            **export,
            "canary_rows": 100,
            "torch_coords_sha256": ordered_array_sha256(torch_coords),
            "onnx_coords_sha256": ordered_array_sha256(ort_coords),
            "fp32_parity_max_abs": max_abs,
        }
    body = {
        "schema": "round0026-export-canary-v1",
        "round_id": ROUND_ID,
        "dependencies": deps,
        "python_deps_path": os.path.realpath(PYTHON_DEPS),
        "sample": _sample_receipt_fields(sample_info),
        "sample_sources": sample_info["sources"],
        "models": models,
        "available_model_count": len(models),
    }
    receipt = _seal(body)
    path = os.path.join(output_root, "canary.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "canary": expected_input_signature(path)}


def run_cpu_profile(*, canary_path: str, output_root: str) -> dict[str, Any]:
    output_root = create_fresh_directory(output_root, label="Round 0026 CPU profile output")
    canary = _read_json(canary_path)
    sample_info = build_sample(output_root)
    sample = sample_info.pop("sample_vectors")
    models = {}
    for spec in available_model_specs():
        export = export_onnx(spec, output_root, sample)
        wrapper = _load_model(spec, device="cpu")
        torch_coords = _torch_forward(wrapper, sample, batch_size=4096, device="cpu")
        session = _ort_session(export["onnx_fp32"]["canonical_path"], threads=8)
        ort_coords = np.asarray(_run_ort(session, sample), dtype=np.float32)
        fp32_parity = float(np.max(np.abs(torch_coords - ort_coords)))
        if fp32_parity > 1e-3:
            raise RuntimeError(f"R0026 fp32 parity failed for {spec.name}: {fp32_parity}")
        fp16_status: dict[str, Any]
        try:
            fp16_session = _ort_session(export["onnx_fp16"]["canonical_path"], threads=8)
            fp16_coords = np.asarray(_run_ort(fp16_session, sample.astype(np.float16)), dtype=np.float32)
            fp16_status = {
                "status": "available",
                "parity_max_abs": float(np.max(np.abs(torch_coords - fp16_coords))),
                "coords_sha256": ordered_array_sha256(fp16_coords),
                "tolerance_reported": 5e-3,
            }
        except Exception as exc:
            fp16_status = {"status": "unavailable", "reason": repr(exc), "tolerance_reported": 5e-3}
        cpu: dict[str, Any] = {}
        for threads in (1, 8):
            session = _ort_session(export["onnx_fp32"]["canonical_path"], threads=threads)
            per_batch = {}
            for batch_size in (1, 1000, 100000):
                per_batch[str(batch_size)] = _time_ort(session, _make_batch(sample, batch_size))
            cpu[f"threads_{threads}"] = per_batch
        models[spec.name] = {
            **export,
            "torch_fp32_coords_sha256": ordered_array_sha256(torch_coords),
            "onnx_fp32_coords_sha256": ordered_array_sha256(ort_coords),
            "fp32_parity_max_abs": fp32_parity,
            "fp16_export": fp16_status,
            "cpu_onnxruntime": cpu,
            "runtime_inference_panel_column_partial": {
                "params": export["params"],
                "bytes_onnx": export["onnx_fp32"]["bytes"],
                "cpu1_rows_s": cpu["threads_1"]["100000"]["rows_per_second_median"],
                "cpu8_rows_s": cpu["threads_8"]["100000"]["rows_per_second_median"],
                "parity_max_abs": fp32_parity,
            },
        }
    body = {
        "schema": "round0026-cpu-inference-profile-v1",
        "round_id": ROUND_ID,
        "canary": expected_input_signature(canary_path),
        "canary_identity_sha256": canary.get("identity_sha256"),
        "sample": _sample_receipt_fields(sample_info),
        "sample_sources": sample_info["sources"],
        "models": models,
        "batch_sizes": [1, 1000, 100000],
        "cpu_thread_counts": [1, 8],
    }
    receipt = _seal(body)
    path = os.path.join(output_root, "cpu-profile.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "cpu_profile": expected_input_signature(path)}


def _time_torch_gpu(spec: ModelSpec, batch: np.ndarray) -> dict[str, Any]:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("BASEMAP_ROUND0026_GPU_UUID", GPU_UUID)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("R0026 GPU profile requires CUDA after GPU lease acquisition")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    wrapper = _load_model(spec, device="cuda")
    values = torch.from_numpy(batch.astype(np.float32, copy=False)).to("cuda")
    with torch.no_grad():
        for _ in range(2):
            out = wrapper.model(values)
        torch.cuda.synchronize()
        times = []
        output_sha = None
        for _ in range(5):
            start = time.perf_counter()
            out = wrapper.model(values)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            output = out.detach().cpu().numpy().astype(np.float32)
            if output.shape != (len(batch), 2) or not np.isfinite(output).all():
                raise RuntimeError("R0026 GPU torch throughput output is non-finite or malformed")
            if output_sha is None:
                output_sha = ordered_array_sha256(output)
            times.append(elapsed)
    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    del values, out, wrapper
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {
        **_latency_summary(times, len(batch)),
        "output_sha256_first": output_sha,
        "device": "cuda",
        "input_residency": "device_fp32",
        "peak_memory_allocated_bytes": peak_alloc,
        "peak_memory_reserved_bytes": peak_reserved,
    }


def run_gpu_merge(*, cpu_profile_path: str, output_root: str) -> dict[str, Any]:
    output_root = create_fresh_directory(output_root, label="Round 0026 GPU merge output")
    cpu_profile = _read_json(cpu_profile_path)
    sample_info = build_sample(output_root)
    sample = sample_info.pop("sample_vectors")
    batch = _make_batch(sample, 100000)
    gpu = {}
    for spec in available_model_specs():
        gpu[spec.name] = _time_torch_gpu(spec, batch)
    models = {}
    for name, cpu_model in cpu_profile["models"].items():
        models[name] = {
            **cpu_model,
            "gpu_torch_batch100k": gpu[name],
            "runtime_inference_panel_column": {
                **cpu_model["runtime_inference_panel_column_partial"],
                "gpu_rows_s": gpu[name]["rows_per_second_median"],
            },
        }
    body = {
        "schema": "projector-inference-profile-v1",
        "round_id": ROUND_ID,
        "cpu_profile": expected_input_signature(cpu_profile_path),
        "sample": _sample_receipt_fields(sample_info),
        "sample_sources": sample_info["sources"],
        "models": models,
        "acceptance": {
            "all_fp32_parity_at_most_1e_3": all(
                float(item["fp32_parity_max_abs"]) <= 1e-3 for item in models.values()
            ),
            "all_registered_measurements_present": True,
        },
    }
    receipt = _seal(body)
    path = os.path.join(output_root, "inference-profile.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "inference_profile": expected_input_signature(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    canary = sub.add_parser("canary")
    canary.add_argument("--out", required=True)
    cpu = sub.add_parser("cpu-profile")
    cpu.add_argument("--canary", required=True)
    cpu.add_argument("--out", required=True)
    gpu = sub.add_parser("gpu-merge")
    gpu.add_argument("--cpu-profile", required=True)
    gpu.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "canary":
        result = run_canary(output_root=os.path.realpath(args.out))
    elif args.command == "cpu-profile":
        result = run_cpu_profile(
            canary_path=os.path.realpath(args.canary),
            output_root=os.path.realpath(args.out),
        )
    else:
        result = run_gpu_merge(
            cpu_profile_path=os.path.realpath(args.cpu_profile),
            output_root=os.path.realpath(args.out),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
