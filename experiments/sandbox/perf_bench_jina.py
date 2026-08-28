#!/usr/bin/env python3
"""Jina-shape THROUGHPUT bench (external review #12, gate 2).

Measures int8 vs fp16-RESIDENT training throughput (updates/s) at the TRUE jina
D768 shape, encoder widths h2048 + h3072, using within-run delta-method segments
so per-process setup (model init, substrate load, int8 quantization) cancels in
the diff. Gate 2 (C1_port_proposal.md §5b) PASSES when, at each width,

    int8 updates/s  >=  0.85 x  fp16-resident updates/s.

This bench does NOT measure quality: the int8 tax (gate 3) is settled by the FULL
`jina-multi-2m champion-bs16k` vs `champion-bs16k-hostint8` parity ARM, because a
truncated-horizon quality readout is a proxy and proxies don't measure the real
thing. Here we only time the transport+forward+optimizer at production geometry.

Reusable segment harness: SEGMENTS is a list of (label, kwarg-overrides); each is
timed by the same `delta_updates_per_s`. The C1 B2 throughput levers (BF16,
combined-gather once C1 lands, cached labels, strided fneg/clip) plug in later as
additional segments on the same machinery — do NOT fork a second bench for them.

Excluded on purpose (workspace don't-revisit list): torch.compile, TF32,
batch=32768, quiet-logging micro-opts. Not throughput levers worth re-testing.

CPU-launch, single GPU, run in an idle window (systemd, after Phase A). The
champion recipe stack (dose4, fneg1.0, tanh4, pos0.10, rankneg 500K, bs16k) is
held fixed across segments; only width and x_residency vary.

Output: /data/latent-basemap/sandbox/perf-bench-jina/results.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("PERF_JINA_OUT",
           "/data/latent-basemap/sandbox/perf-bench-jina"))
DS = "jina-multi-2m"
EDGES = Path(f"/data/latent-basemap/sandbox/{DS}/edges-k15-fuzzy.npz")
# delta method: (short, long) horizons; the long-minus-short diff is pure
# steady-state training (setup, warmup, and int8 quantization all cancel).
DELTA_UPDATES = (400, 6_400)
GATE2_RATIO = 0.85

# (label, kwarg overrides on top of the champion stack). Width + residency only;
# B2 levers append here later (same machinery).
SEGMENTS = [
    ("h2048-resident", {"hidden_dim": 2048, "x_residency": "auto"}),
    ("h2048-int8",     {"hidden_dim": 2048, "x_residency": "host_int8"}),
    ("h3072-resident", {"hidden_dim": 3072, "x_residency": "auto"}),
    ("h3072-int8",     {"hidden_dim": 3072, "x_residency": "host_int8"}),
]

# --batchscan (owner 2026-08-28): does batch amortize the fixed per-batch int8 transport
# overhead the way width did (0.639→0.769)? bs16/32/64K × h2048/h3072 × int8/resident,
# reporting samples/s (ups×batch) + int8:resident ratio per batch. SCOPED EXCEPTION to the
# don't-revisit "bs32K no benefit" — that result was fp16-RESIDENT MiniLM D384 on the OLD
# partially-invalidated bench; it does NOT cover int8-transport jina-D768 batch-scaling.
BATCH_SCAN_BATCHES = (16384, 32768, 65536)
BATCH_SCAN_WIDTHS = (2048, 3072)
BATCHSCAN_SEGMENTS = [
    (f"h{w}-bs{b // 1024}k-{'int8' if r == 'host_int8' else 'resident'}",
     {"hidden_dim": w, "x_residency": r, "batch_size": b})
    for w in BATCH_SCAN_WIDTHS for b in BATCH_SCAN_BATCHES
    for r in ("auto", "host_int8")
]
BATCHSCAN_PROVENANCE = (
    "SCOPED EXCEPTION to don't-revisit 'bs32K no benefit': that was fp16-resident MiniLM "
    "D384 on the OLD partially-invalidated perf bench; this measures int8-TRANSPORT jina "
    "D768 batch-scaling, a different question. Owner-authorized 2026-08-28. Not editing the "
    "don't-revisit list/config — provenance recorded here."
)


def _ensure_paths():
    # repo root (for `basemap`), experiments, and sandbox — delta_updates_per_s
    # imports `basemap...` so the ROOT (parents[2]) MUST be on the path, not just
    # experiments/sandbox (the original bug: all segments ModuleNotFoundError).
    for p in (Path(__file__).resolve().parents[2],
              Path(__file__).resolve().parents[1],
              Path(__file__).resolve().parent):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_jina():
    _ensure_paths()
    from image_map_pipeline import DATASETS, _norm
    x = _norm(DATASETS[DS]["load"]())
    e = int(len(np.load(EDGES, mmap_mode="r")["sources"]))
    return np.asarray(x, dtype=np.float32), e


def _champion_kwargs(base_kwargs: dict, overrides: dict, horizon: int,
                     e: int) -> dict:
    """Champion stack at a given horizon + segment overrides."""
    from knobs_2m import MD
    batch = overrides.get("batch_size", 16384)
    pos_ratio = overrides.get("pos_ratio", 0.10)
    n_epochs = max(1, math.ceil(horizon * batch * pos_ratio / e) + 1)
    kwargs = dict(base_kwargs)
    kwargs.update({
        "low_dim_kernel": "umap", **MD["000"],
        "fneg_weight": 1.0, "neg_tanh_gamma": 4.0, "pos_ratio": pos_ratio,
        "rankneg_window": 500_000, "batch_size": batch,
        "total_steps_estimate": horizon, "n_epochs": n_epochs,
    })
    kwargs.update(overrides)
    return kwargs


def delta_updates_per_s(label: str, overrides: dict, X, e: int) -> dict:
    """Run short + long horizons in this process; return steady-state updates/s
    (setup/quantization cancel in the long-short diff)."""
    import torch

    from basemap.pumap.parametric_umap import core as C
    from knobs_2m import BASE_KWARGS

    walls, ups_seen, vram = [], [], 0.0
    torch.cuda.reset_peak_memory_stats()
    for target in DELTA_UPDATES:
        kwargs = _champion_kwargs(BASE_KWARGS, overrides, target, e)
        q = C.ParametricUMAP(**kwargs)
        t0 = time.time()
        q.fit(X, precomputed_edges_path=str(EDGES), random_state=42)
        walls.append(time.time() - t0)
        st = q._train_stats or {}
        ups_seen.append(int(st.get("positive_lr_optimizer_steps")
                            or st.get("optimizer_steps_succeeded")
                            or target))
        vram = max(vram, torch.cuda.max_memory_allocated() / 1e9)
        del q
        torch.cuda.empty_cache()
    d_up = ups_seen[1] - ups_seen[0]
    d_wall = walls[1] - walls[0]
    ups = d_up / max(d_wall, 1e-9)
    batch = overrides.get("batch_size", 16384)
    return {"segment": label, "overrides": overrides,
            "updates": ups_seen, "walls_s": [round(w, 1) for w in walls],
            "updates_per_s": round(ups, 1),
            "edges_per_s": round(ups * batch),
            "peak_vram_gb": round(vram, 2)}


def main() -> int:
    args = sys.argv[1:]
    batchscan = "--batchscan" in args
    which = next((a for a in args if not a.startswith("--")), None)
    segments = BATCHSCAN_SEGMENTS if batchscan else SEGMENTS
    out_dir = (OUT.parent / "perf-bench-jina-batchscan") if batchscan else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    X, e = _load_jina()
    print(f"jina substrate {X.shape} dtype={X.dtype}; edges={e}"
          + (" [BATCHSCAN]" if batchscan else ""), flush=True)

    rows = {}
    for label, ov in segments:
        if which and label != which:
            continue
        try:
            r = delta_updates_per_s(label, ov, X, e)
        except Exception as ex:  # noqa: BLE001 — a segment may OOM/err; record it
            r = {"segment": label, "overrides": ov,
                 "error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        rows[label] = r
        print(json.dumps(r), flush=True)
        (out_dir / f"{label}.json").write_text(json.dumps(r, indent=1))

    if batchscan:
        # per-(width,batch): int8/resident ratio + samples/s (ups×batch). Does a bigger
        # batch amortize the fixed int8 transport overhead enough to clear 0.85?
        grid = {}
        for w in BATCH_SCAN_WIDTHS:
            for b in BATCH_SCAN_BATCHES:
                bk = b // 1024
                ru = rows.get(f"h{w}-bs{bk}k-resident", {}).get("updates_per_s")
                iu = rows.get(f"h{w}-bs{bk}k-int8", {}).get("updates_per_s")
                grid[f"h{w}-bs{bk}k"] = {
                    "resident_ups": ru, "int8_ups": iu,
                    "resident_samples_s": (round(ru * b) if ru else None),
                    "int8_samples_s": (round(iu * b) if iu else None),
                    "ratio": (round(iu / ru, 3) if ru and iu else None),
                    "clears_0.85": bool(ru and iu and iu / ru >= GATE2_RATIO)}
        if not which:
            (out_dir / "batchscan-results.json").write_text(json.dumps({
                "schema": "perf-bench-jina-batchscan-2026-08-28",
                "shape": {"dataset": DS, "dim": int(X.shape[1]), "rows": int(X.shape[0])},
                "gate2_ratio_threshold": GATE2_RATIO, "provenance": BATCHSCAN_PROVENANCE,
                "grid": grid, "segments": rows}, indent=1))
            print(f"\nbatchscan grid: {json.dumps(grid)}", flush=True)
    else:
        # gate-2 ratios per width (int8 / resident); PASS iff >= 0.85 at every width.
        gate2 = {}
        for w in ("h2048", "h3072"):
            res = rows.get(f"{w}-resident", {}).get("updates_per_s")
            i8 = rows.get(f"{w}-int8", {}).get("updates_per_s")
            ratio = (round(i8 / res, 3) if res and i8 else None)
            gate2[w] = {"resident_ups": res, "int8_ups": i8, "ratio": ratio,
                        "pass": (ratio is not None and ratio >= GATE2_RATIO)}
        all_measured = all(v["ratio"] is not None for v in gate2.values())
        verdict = ("PASS" if all_measured and all(v["pass"] for v in gate2.values())
                   else ("FAIL" if all_measured else "INCOMPLETE"))
        if not which:
            (OUT / "results.json").write_text(json.dumps({
                "schema": "perf-bench-jina-gate2-2026-08-27",
                "shape": {"dataset": DS, "dim": int(X.shape[1]),
                          "rows": int(X.shape[0]), "edges": e},
                "delta_updates": list(DELTA_UPDATES), "gate2_ratio_threshold": GATE2_RATIO,
                "segments": rows, "gate2": gate2, "gate2_verdict": verdict,
                "note": "gate 2 (throughput) ONLY; int8 / fp16-resident updates/s at jina "
                        "D768, within-run delta segments. int8 TAX (gate 3) is the "
                        "champion-bs16k-hostint8 parity ARM. --batchscan for bs16/32/64K.",
            }, indent=1))
            print(f"\ngate2: {json.dumps(gate2)}\nverdict: {verdict}", flush=True)

    # FAIL LOUDLY (non-zero) when a segment errored — a fail-closed orchestrator must not
    # read null throughput as success.
    errored = [k for k, v in rows.items() if isinstance(v, dict) and v.get("error")]
    if errored:
        print(f"ERROR: {len(errored)} segment(s) failed: {errored} — non-zero exit", flush=True)
        return 1
    if not which and not batchscan and verdict == "INCOMPLETE":
        print("ERROR: gate2 INCOMPLETE (unmeasured) — non-zero exit", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
