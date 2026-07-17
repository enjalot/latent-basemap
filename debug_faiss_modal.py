"""
Diagnostic script to debug FAISS GPU issues on Modal A10G.
Tests faiss-gpu-cu12 and faiss-cpu packages with different index types.

Usage:
    modal run debug_faiss_modal.py

Previous error with faiss-gpu (1.7.2):
    Faiss assertion 'err__ == cudaSuccess' failed
    CUDA error 9 invalid configuration argument
    Also hangs on IndexFlatL2 GPU transfer.
"""

from basemap.round0005_retirement import refuse_retired_launcher

refuse_retired_launcher("debug_faiss_modal.py")

import modal

app = modal.App("debug-faiss")

# Image with faiss-gpu-cu12 on CUDA base image
faiss_gpu_cu12_cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .pip_install(
        "numpy==1.26.3",
        "faiss-gpu-cu12",
    )
)

# Image with faiss-gpu-cu12 on debian slim (no CUDA toolkit)
faiss_gpu_cu12_slim_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.3",
        "faiss-gpu-cu12",
    )
)

# Image with faiss-cpu only (baseline)
faiss_cpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.3",
        "faiss-cpu",
    )
)


def _run_faiss_tests(label: str):
    """Run FAISS diagnostics. Called inside the Modal container."""
    import time
    import signal
    import traceback
    import numpy as np
    import os
    import subprocess

    results = {"label": label, "tests": {}}

    # ---- Environment info ----
    try:
        import faiss
        results["faiss_version"] = faiss.__version__ if hasattr(faiss, '__version__') else "unknown"
        results["faiss_module_file"] = faiss.__file__
    except Exception as e:
        results["faiss_import_error"] = str(e)
        return results

    # GPU count
    try:
        ngpu = faiss.get_num_gpus()
        results["faiss_num_gpus"] = ngpu
    except Exception as e:
        results["faiss_num_gpus"] = f"error: {e}"

    # Check for GPU support attributes
    gpu_attrs = []
    for attr in ["StandardGpuResources", "GpuIndexFlatL2", "GpuIndexIVFFlat", "index_cpu_to_gpu"]:
        gpu_attrs.append(f"{attr}={'YES' if hasattr(faiss, attr) else 'NO'}")
    results["gpu_support"] = ", ".join(gpu_attrs)

    # CUDA / GPU info
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"], text=True)
        results["nvidia_smi"] = out.strip()
    except Exception as e:
        results["nvidia_smi"] = f"error: {e}"

    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.STDOUT)
        for line in out.strip().split("\n"):
            if "release" in line.lower():
                results["nvcc_version"] = line.strip()
    except Exception as e:
        results["nvcc_version"] = f"not installed: {e}"

    # Check CUDA_HOME and LD_LIBRARY_PATH
    results["CUDA_HOME"] = os.environ.get("CUDA_HOME", "not set")
    results["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "not set")

    # ---- Timeout helper using threading (CUDA doesn't survive fork) ----
    import threading

    def run_with_timeout(fn, timeout_sec=60):
        """Run fn in a thread with timeout. Returns (status, result_or_error, elapsed)."""
        result_holder = [None, None]  # [status, msg]

        def wrapper():
            try:
                res = fn()
                result_holder[0] = "PASS"
                result_holder[1] = res
            except Exception as e:
                result_holder[0] = "ERROR"
                result_holder[1] = f"{e}\n{traceback.format_exc()}"

        t0 = time.time()
        thread = threading.Thread(target=wrapper)
        thread.start()
        thread.join(timeout=timeout_sec)
        elapsed = time.time() - t0

        if thread.is_alive():
            # Thread is stuck (likely CUDA hang) - we can't kill it, but we report
            return "TIMEOUT/HANG", f"Still running after {elapsed:.1f}s (thread cannot be killed)", elapsed
        elif result_holder[0] is not None:
            return result_holder[0], result_holder[1], elapsed
        else:
            return "ERROR", "Thread exited without result", elapsed

    # ---- Test data ----
    np.random.seed(42)
    n = 1000
    d = 384
    data = np.random.random((n, d)).astype("float32")
    queries = np.random.random((5, d)).astype("float32")

    # ---- Test functions ----
    def test_flat_cpu():
        idx = faiss.IndexFlatL2(d)
        idx.add(data)
        D, I = idx.search(queries, 10)
        return f"OK, neighbors shape {I.shape}"

    def test_flat_gpu():
        res = faiss.StandardGpuResources()
        cpu_idx = faiss.IndexFlatL2(d)
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
        gpu_idx.add(data)
        D, I = gpu_idx.search(queries, 10)
        return f"OK, neighbors shape {I.shape}"

    def test_flat_gpu_direct():
        res = faiss.StandardGpuResources()
        gpu_idx = faiss.GpuIndexFlatL2(res, d)
        gpu_idx.add(data)
        D, I = gpu_idx.search(queries, 10)
        return f"OK, neighbors shape {I.shape}"

    def test_ivfflat_cpu():
        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFFlat(quantizer, d, 10)
        idx.train(data)
        idx.add(data)
        idx.nprobe = 5
        D, I = idx.search(queries, 10)
        return f"OK, neighbors shape {I.shape}"

    def test_ivfflat_gpu():
        res = faiss.StandardGpuResources()
        quantizer = faiss.IndexFlatL2(d)
        cpu_idx = faiss.IndexIVFFlat(quantizer, d, 10)
        cpu_idx.train(data)
        cpu_idx.add(data)
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
        gpu_idx.nprobe = 5
        D, I = gpu_idx.search(queries, 10)
        return f"OK, neighbors shape {I.shape}"

    def test_hnsw_cpu():
        idx = faiss.IndexHNSWFlat(d, 16)
        idx.add(data)
        D, I = idx.search(queries, 10)
        return f"OK, neighbors shape {I.shape}"

    tests = [
        ("flat_cpu", test_flat_cpu),
        ("flat_gpu_via_transfer", test_flat_gpu),
        ("flat_gpu_direct", test_flat_gpu_direct),
        ("ivfflat_cpu", test_ivfflat_cpu),
        ("ivfflat_gpu_transfer", test_ivfflat_gpu),
        ("hnsw_cpu", test_hnsw_cpu),
    ]

    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")

        # Skip GPU tests if no GPU support
        if "gpu" in name and results.get("faiss_num_gpus") == 0:
            results["tests"][name] = {"status": "SKIP", "result": "No GPU support in this FAISS build"}
            print(f"  SKIP: No GPU support in this FAISS build")
            continue

        if "gpu" in name and not hasattr(faiss, "StandardGpuResources"):
            results["tests"][name] = {"status": "SKIP", "result": "StandardGpuResources not available"}
            print(f"  SKIP: StandardGpuResources not available")
            continue

        status, msg, elapsed = run_with_timeout(fn, timeout_sec=60)
        results["tests"][name] = {"status": status, "result": msg, "time": f"{elapsed:.2f}s"}
        print(f"  {status} ({elapsed:.2f}s): {msg}")

    return results


@app.function(
    image=faiss_gpu_cu12_cuda_image,
    gpu="A10G",
    timeout=300,
)
def test_faiss_gpu_cu12_cuda():
    """Test with faiss-gpu-cu12 on CUDA base image."""
    return _run_faiss_tests("faiss-gpu-cu12 + CUDA 12.4 base")


@app.function(
    image=faiss_gpu_cu12_slim_image,
    gpu="A10G",
    timeout=300,
)
def test_faiss_gpu_cu12_slim():
    """Test with faiss-gpu-cu12 on debian slim (no CUDA toolkit)."""
    return _run_faiss_tests("faiss-gpu-cu12 + debian slim")


@app.function(
    image=faiss_cpu_image,
    gpu="A10G",
    timeout=300,
)
def test_faiss_cpu():
    """Test with faiss-cpu pip package (GPU hardware available but using CPU FAISS)."""
    return _run_faiss_tests("faiss-cpu pip package")


@app.local_entrypoint()
def main():
    import json

    print("\n" + "=" * 70)
    print("FAISS GPU DIAGNOSTIC - Modal A10G")
    print("=" * 70)

    # Run all in parallel
    cu12_cuda_future = test_faiss_gpu_cu12_cuda.spawn()
    cu12_slim_future = test_faiss_gpu_cu12_slim.spawn()
    cpu_future = test_faiss_cpu.spawn()

    all_results = []
    for label, future in [
        ("faiss-gpu-cu12 + CUDA base", cu12_cuda_future),
        ("faiss-gpu-cu12 + slim", cu12_slim_future),
        ("faiss-cpu", cpu_future),
    ]:
        print(f"\n>>> Waiting for {label} results...")
        try:
            res = future.get()
            print(json.dumps(res, indent=2))
            all_results.append((label, res))
        except Exception as e:
            print(f"{label} function FAILED: {e}")
            all_results.append((label, {"error": str(e)}))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for label, res in all_results:
        print(f"\n--- {label} ---")
        if "error" in res:
            print(f"  FUNCTION ERROR: {res['error']}")
            continue
        print(f"  FAISS version: {res.get('faiss_version', '?')}")
        print(f"  GPUs detected by FAISS: {res.get('faiss_num_gpus', '?')}")
        print(f"  GPU support: {res.get('gpu_support', '?')}")
        print(f"  nvidia-smi: {res.get('nvidia_smi', '?')}")
        print(f"  nvcc: {res.get('nvcc_version', '?')}")
        for tname, tresult in res.get("tests", {}).items():
            status = tresult["status"]
            t = tresult.get("time", "?")
            extra = tresult.get("result", tresult.get("error", ""))
            print(f"  {tname}: {status} ({t}) {extra}")
