#!/usr/bin/env python3
"""Export evolution-benchmark timelines for the evolution viz page.

Per arm x snapshot -> ~/.agent/basemap-maps/evolution/data/<arm>/S<k>.bin
(int16 xy, quantized in ONE FIXED FRAME per arm — the union 0.5/99.5 box over
all its snapshots — so on-screen motion is real motion, never rescaling).

Per transition Sk->Sk+1 (shared rows only):
  <arm>/disp-S<k>.bin   f16 displacement magnitude per shared row (radius-
                        normalized, same normalization as the churn metric)
  <arm>/trails-S<k>.bin int16 quads (x0,y0,x1,y1) for a seeded ~120K sample
                        of shared rows — GL_LINES trails.

cohorts.json: tranche row-boundaries (snapshots are concatenations
T0|T1|...|Tk by construction) + arm list + per-arm frame + churn stats.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SB = Path("/data/latent-basemap/sandbox")
OUT = Path.home() / ".agent/basemap-maps/evolution/data"
T0_N = 4_000_000
TRANCHE = 800_000
SNAP_N = [T0_N + k * TRANCHE for k in range(6)]
TRAIL_SAMPLE = 120_000
RNG = np.random.default_rng(7)

ARMS = {
    "armA-frozen": SB / "evolbench-armA-frozen",
    "armA-triggered": SB / "evolbench-armA-triggered",
    "armB-cuvs": SB / "evolbench-armB",
    "comp-umap-frozen": SB / "evolbench-competitor-umap-frozen_transform",
    "comp-umap-full": SB / "evolbench-competitor-umap-full_timeline",
}


def load_snap(d: Path, k: int) -> np.ndarray | None:
    for name in (f"coords-S{k}.npy", f"raw-S{k}.npy"):
        f = d / name
        if f.exists():
            xy = np.load(f, mmap_mode="r")
            if xy.ndim == 2 and xy.shape[1] == 2:
                return np.asarray(xy, dtype=np.float32)
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"snapshots": SNAP_N, "t0": T0_N, "tranche": TRANCHE, "arms": {}}
    for arm, d in ARMS.items():
        snaps = {k: load_snap(d, k) for k in range(6)}
        snaps = {k: v for k, v in snaps.items() if v is not None}
        if not snaps:
            print(f"{arm}: no snapshots, skip")
            continue
        ad = OUT / arm
        ad.mkdir(exist_ok=True)
        # one fixed frame per arm: union percentile box over all snapshots
        los, his = [], []
        for xy in snaps.values():
            los.append(np.percentile(xy, 0.5, axis=0))
            his.append(np.percentile(xy, 99.5, axis=0))
        lo = np.min(los, axis=0).astype(np.float64)
        hi = np.max(his, axis=0).astype(np.float64)
        span = np.maximum(hi - lo, 1e-9)
        arm_meta = {"snaps": sorted(snaps), "lo": list(lo),
                    "span": list(span), "churn": {}}
        # radius normalization consistent with the churn metric (p90 of S0)
        s0 = snaps.get(0)
        R = float(np.percentile(
            np.linalg.norm(s0 - np.median(s0, axis=0), axis=1), 90)) \
            if s0 is not None else 1.0
        arm_meta["radius"] = R

        def quant(xy):
            q = np.clip((xy - lo) / span, 0, 1) * 65535.0 - 32768.0
            return q.astype("<i2")

        for k in sorted(snaps):
            quant(snaps[k]).tofile(ad / f"S{k}.bin")
        for k in sorted(snaps):
            if k + 1 not in snaps:
                continue
            a, b = snaps[k], snaps[k + 1]
            n = min(len(a), len(b))
            disp = np.linalg.norm(b[:n] - a[:n], axis=1) / max(R, 1e-9)
            disp.astype(np.float16).tofile(ad / f"disp-S{k}.bin")
            arm_meta["churn"][str(k)] = {
                "mean": round(float(disp.mean()), 5),
                "p95": round(float(np.percentile(disp, 95)), 5),
                "max": round(float(disp.max()), 4), "n": int(n)}
            idx = RNG.choice(n, size=min(TRAIL_SAMPLE, n), replace=False)
            idx.sort()
            quad = np.concatenate([quant(a[idx]), quant(b[idx])], axis=1)
            quad.astype("<i2").tofile(ad / f"trails-S{k}.bin")
            (ad / f"trailidx-S{k}.bin").write_bytes(
                idx.astype("<u4").tobytes())
        meta["arms"][arm] = arm_meta
        print(f"{arm}: {len(snaps)} snapshots exported "
              f"(churn means: {[v['mean'] for v in arm_meta['churn'].values()]})")
    (OUT / "cohorts.json").write_text(json.dumps(meta))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
