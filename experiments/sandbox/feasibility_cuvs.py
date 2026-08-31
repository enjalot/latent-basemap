#!/usr/bin/env python3
"""cuVS feasibility check for the evolution benchmark's arm B (plan §2, owner 2026-08-31). cuml-env.

Arm B reruns FULL transductive UMAP at every snapshot up to final=2xT0, so a scale is only usable if a
single rerun at the final size runs within a wall budget AND fits VRAM. Tests both the IN-CORE
(build_algo=nn_descent, data on GPU) and OUT-OF-CORE (nnd_data_on_host) paths.

HARDENED after the first run's 8M out-of-core DEADLOCK (STAT=D, 0% CPU/GPU for 2h40m, killed): each
(scale, path) runs in a TIMEOUT-guarded SUBPROCESS (SIGKILL on hang), and peak VRAM is sampled
externally via nvidia-smi (the cupy-pool read was broken — cuml uses RMM). One config's hang never
blocks the rest.

Runner: `feasibility_cuvs.py`. Worker (internal): `feasibility_cuvs.py --worker <N> <incore|outofcore>`.
Output: /data/latent-basemap/sandbox/cuvs-feasibility.json
"""
import json, os, subprocess, sys, time, threading
from pathlib import Path

DIM = 384
N_EPOCHS = int(os.environ.get("CUVS_EPOCHS", "200"))
TIMEOUT_S = int(os.environ.get("CUVS_TIMEOUT", "600"))     # per (scale,path); >this = not comfortable
WALL_BUDGET = int(os.environ.get("CUVS_WALL_BUDGET", "600"))
VRAM_CAP_GB = float(os.environ.get("CUVS_VRAM_CAP", "29"))
OUT = Path("/data/latent-basemap/sandbox/cuvs-feasibility.json")
# in-core is VRAM-bounded (~data 15GB@10M + graph/embed); out-of-core deadlocks >4M in cuml 25.02 (test 1 pt)
CONFIGS = [(n, "incore") for n in (4_000_000, 6_000_000, 8_000_000, 10_000_000)] + [(8_000_000, "outofcore")]


def _worker(N, path):
    import numpy as np
    from cuml.manifold import UMAP
    X = np.random.default_rng(0).standard_normal((N, DIM), dtype=np.float32)
    kw = dict(build_algo="nn_descent")
    if path == "outofcore":
        kw["build_kwds"] = {"nnd_data_on_host": True}
    t0 = time.time()
    um = UMAP(n_neighbors=15, n_components=2, n_epochs=N_EPOCHS, min_dist=0.1, random_state=42, **kw)
    emb = um.fit_transform(X)
    print(json.dumps({"wall_s": round(time.time() - t0, 1), "emb_rows": int(emb.shape[0])}), flush=True)


def _peak_vram_sampler(stop, peak):
    while not stop.is_set():
        try:
            u = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                         "--format=csv,noheader,nounits"], timeout=5)
            peak[0] = max(peak[0], int(u.decode().splitlines()[0]) / 1024.0)
        except Exception:
            pass
        time.sleep(1.0)


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        _worker(int(sys.argv[2]), sys.argv[3]); return 0
    results = []
    for N, path in CONFIGS:
        rec = {"N": N, "path": path, "n_epochs": N_EPOCHS, "timeout_s": TIMEOUT_S}
        stop = threading.Event(); peak = [0.0]
        sampler = threading.Thread(target=_peak_vram_sampler, args=(stop, peak), daemon=True); sampler.start()
        p = subprocess.Popen([sys.executable, __file__, "--worker", str(N), path],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            out, _ = p.communicate(timeout=TIMEOUT_S)
            txt = out.decode()[-2000:]
            line = [l for l in txt.splitlines() if l.strip().startswith("{")]
            if p.returncode == 0 and line:
                w = json.loads(line[-1])
                rec.update({"status": "OK", "wall_s": w["wall_s"], "peak_vram_gb": round(peak[0], 2)})
            else:
                rec.update({"status": f"FAIL(rc={p.returncode})", "peak_vram_gb": round(peak[0], 2),
                            "tail": txt[-400:]})
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
            rec.update({"status": f"TIMEOUT(>{TIMEOUT_S}s — hang/too-slow)", "peak_vram_gb": round(peak[0], 2)})
        finally:
            stop.set(); sampler.join(timeout=2)
        print(f"N={N/1e6:.0f}M {path}: {rec['status']} wall={rec.get('wall_s')} vram={rec.get('peak_vram_gb')}GB",
              flush=True)
        results.append(rec)

    comfortable = [r for r in results if r.get("status") == "OK"
                   and r.get("wall_s", 1e9) <= WALL_BUDGET and r.get("peak_vram_gb", 99) <= VRAM_CAP_GB]
    largest = max((r["N"] for r in comfortable), default=0)
    decision = {"largest_comfortable_final": largest, "recommended_T0": largest // 2,
                "wall_budget_s": WALL_BUDGET, "vram_cap_gb": VRAM_CAP_GB,
                "note": ("comfortable = OK AND single-rerun wall<=budget AND peak_vram<=cap (arm B reruns "
                         "5x, so the per-rerun wall must not eat the benchmark budget). out-of-core path "
                         "deadlocks >4M in cuml 25.02 — in-core (VRAM-bounded) is the usable route.")}
    OUT.write_text(json.dumps({"schema": "cuvs-feasibility-2026-08-31-v2", "configs": results,
                               "scale_decision": decision}, indent=1))
    print(f"\n=== largest_comfortable_final={largest/1e6:.0f}M -> recommended_T0={decision['recommended_T0']/1e6:.0f}M ===",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
