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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sandbox"))
import frame  # the ONE rigid gauge (review-item B) shared with the scorers

SB = Path("/data/latent-basemap/sandbox")
TRAIL_SAMPLE = 120_000
RNG = np.random.default_rng(7)

# λ-frontier pseudo-arm: "snapshots" are the anchor-weight ladder at the T3
# (reddit-injection) scenario, frozen → loosening anchors → full retrain, so
# the morph slider walks the frontier instead of time. Endpoints per
# pplan_lambda_sweep.sh: w=inf = armA-frozen S3, w=0 = armA-triggered S3.
# 3ep budget variants excluded to keep the path monotone in w.
LAMBDA_WS = ["100", "50", "20", "10", "5", "2", "1",
             "0.5", "0.25", "0.1", "0.05", "0.02"]


def _ladder(frozen: Path, lam_dir: Path, retrain: Path):
    return ([("w=∞ frozen", frozen)]
            + [(f"w={t}", lam_dir / f"coords-w{t}.npy") for t in LAMBDA_WS]
            + [("w=0 retrain", retrain)])


# One export per embedding space; each gets its own out dir + cohorts.json
# (d768 = jina timeline at half scale: T0=2M + 5x400K).
SPACES = {
    "minilm": {
        "out": Path.home() / ".agent/basemap-maps/evolution/data",
        "t0": 4_000_000, "tranche": 800_000,
        "arms": {
            "armA-frozen": SB / "evolbench-armA-frozen",
            "armA-triggered": SB / "evolbench-armA-triggered",
            "armB-cuvs": SB / "evolbench-armB",
            "comp-umap-frozen":
                SB / "evolbench-competitor-umap-frozen_transform",
            "comp-umap-full": SB / "evolbench-competitor-umap-full_timeline",
        },
        "ladder": _ladder(SB / "evolbench-armA-frozen/coords-S3.npy",
                          SB / "lambda",
                          SB / "evolbench-armA-triggered/coords-S3.npy"),
        "frontier": SB / "evolbench-lambda-frontier.json",
    },
    "d768": {
        "out": Path.home() / ".agent/basemap-maps/evolution/data-d768",
        "t0": 2_000_000, "tranche": 400_000,
        "arms": {
            "armA-frozen": SB / "evolbench-armA-d768-frozen-v2",
            "armA-triggered": SB / "evolbench-armA-d768-triggered-v2",
            "armB-cuvs": SB / "evolbench-d768-armB",
        },
        "ladder": _ladder(
            SB / "evolbench-armA-d768-frozen-v2/coords-S3.npy",
            SB / "lambda-d768",
            SB / "evolbench-armA-d768-triggered-v2/coords-S3.npy"),
        "frontier": SB / "evolbench-lambda-d768-frontier.json",
    },
}


def load_snap(d: Path, k: int) -> np.ndarray | None:
    for name in (f"coords-S{k}.npy", f"raw-S{k}.npy"):
        f = d / name
        if f.exists():
            xy = np.load(f, mmap_mode="r")
            if xy.ndim == 2 and xy.shape[1] == 2:
                return np.asarray(xy, dtype=np.float32)
    return None


def lambda_snaps(ladder) -> tuple[dict[int, np.ndarray], list[str]]:
    snaps, labels = {}, []
    for label, f in ladder:
        if not f.exists():
            continue  # ladder cells land incrementally; export what's there
        xy = np.load(f, mmap_mode="r")
        if xy.ndim == 2 and xy.shape[1] == 2:
            snaps[len(labels)] = np.asarray(xy, dtype=np.float32)
            labels.append(label)
    return snaps, labels


def export_space(name: str, cfg: dict) -> None:
    OUT = cfg["out"]
    OUT.mkdir(parents=True, exist_ok=True)
    t0, tranche = cfg["t0"], cfg["tranche"]
    meta = {"space": name, "snapshots": [t0 + k * tranche for k in range(6)],
            "t0": t0, "tranche": tranche, "arms": {}}
    jobs = [(arm, {k: load_snap(d, k) for k in range(6)}, None)
            for arm, d in cfg["arms"].items()]
    lam, lam_labels = lambda_snaps(cfg["ladder"])
    if lam:
        jobs.append(("lambda-frontier", lam, lam_labels))
    for arm, snaps, step_labels in jobs:
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
        if step_labels:
            arm_meta["step_labels"] = step_labels
        # canonical radius from the shared gauge (frame.py), stored for the
        # lambda page's dvf normalization
        s0 = snaps.get(0)
        R = frame.frame_radius(s0) if s0 is not None else 1.0
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
            # SAME churn as the scorers: rigid-aligned, prev-radius-normalized
            disp, _ = frame.churn(b, a[:n])
            disp = np.asarray(disp, np.float32)
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
        print(f"[{name}] {arm}: {len(snaps)} snapshots exported "
              f"(churn means: {[v['mean'] for v in arm_meta['churn'].values()]})")
    # attach frontier metrics to the lambda arm (matched by w; 3ep excluded)
    front = cfg["frontier"]
    lam_meta = meta["arms"].get("lambda-frontier")
    if lam_meta and front.exists():
        byw = {}
        for r in json.loads(front.read_text())["frontier"]:
            if "-3ep" in r.get("label", ""):
                continue
            byw[str(float(r["w"]))] = {
                k: r.get(k) for k in ("reddit_ffr", "ood_gain", "churn_mean",
                                      "overall_ffr", "cost_gpu_min")}
        def _wkey(lbl: str) -> str:
            t = lbl.split()[0].split("=")[1]
            return "inf" if t == "∞" else str(float(t))
        lam_meta["metrics"] = [byw.get(_wkey(l))
                               for l in lam_meta["step_labels"]]
    (OUT / "cohorts.json").write_text(json.dumps(meta))
    print(f"-> {OUT}")


def main() -> int:
    for name, cfg in SPACES.items():
        export_space(name, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
