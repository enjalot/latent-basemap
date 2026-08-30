#!/usr/bin/env python3
"""Substrate-draw universality scorer (owner 2026-08-30). GPU.

Three same-seed heads on disjoint composition-matched 2M draws (A/B/C). VALIDITY GATE: their three
init_state_sha256 MUST be equal (same seed -> bit-identical init) — else the experiment is void and we
abort. Readouts:
  (1) FFR SPREAD on a shared eval (a1-common-neutral, proven 0-overlap): quality draw-variance. The
      P1.5 baseline heads @42/@43 are scored on the SAME a1 eval -> SEED-variance, apples-to-apples.
  (2) ROTATION cross/self FFR: head H on each slice X (X's own k15 truth). self = head X on X (member,
      in-sample); cross = head Y!=X on X (unseen). cross-vs-self member advantage per slice.
  (3) PROCRUSTES-aligned per-point deviation + spread-ratio on the a1 projections = geometric agreement.
  (4) CONTEXT TABLE: draw-variance (A/B/C on a1) vs seed-variance (baseline@42/@43 on a1).
Output: /data/latent-basemap/sandbox/draw-univ-score.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

SB = Path("/data/latent-basemap/sandbox")
SUB = Path("/data/latent-basemap/substrates")
SLICES = ("A", "B", "C")


def _norm_load(path):
    x = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return x / n


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from knobs_2m import quick_ffr_v2
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    from scipy.spatial import procrustes

    heads = {s: SB / f"draw-univ-{s}/champion-bs16k/model.pt" for s in SLICES}
    baseline = {"s42": SB / "minilm-mix-2m/p15-baseline-s42/model.pt",
                "s43": SB / "minilm-mix-2m/p15-baseline-s43/model.pt"}
    for k, p in {**heads, **baseline}.items():
        if not Path(p).exists():
            raise SystemExit(f"missing model {k}: {p}")

    # --- VALIDITY GATE: the three same-seed head inits must be bit-identical ---
    init_sha = {}
    for s in SLICES:
        summ = json.loads((SB / f"draw-univ-{s}/champion-bs16k/summary.json").read_text())
        init_sha[s] = summ.get("init_state_sha256")
    gate_ok = len(set(init_sha.values())) == 1 and None not in init_sha.values()
    print(f"[validity gate] init_state_sha256 {init_sha} -> {'PASS' if gate_ok else 'FAIL'}", flush=True)

    a1_x = _norm_load(SUB / "a1-common-neutral/substrate.f32.npy")
    a1_truth = SB / "a1-common-neutral/edges-k15-fuzzy.npz"
    a1_n = int(a1_x.shape[0])

    # --- (1)/(4) shared-eval FFR + a1 projections (for procrustes) ---
    a1_ffr, a1_xy = {}, {}
    for name, mp in {**{f"head-{s}": heads[s] for s in SLICES},
                     **{f"base-{k}": baseline[k] for k in baseline}}.items():
        m = ParametricUMAP.load(str(mp), device="cuda")
        xy = np.asarray(m.transform(a1_x, batch_size=8192), dtype=np.float32)
        a1_xy[name] = xy
        a1_ffr[name] = float(quick_ffr_v2(xy, a1_truth, a1_n))
        print(f"  a1 FFR [{name}] = {a1_ffr[name]:.4f}", flush=True)
        del m

    draw_vals = [a1_ffr[f"head-{s}"] for s in SLICES]
    seed_vals = [a1_ffr["base-s42"], a1_ffr["base-s43"]]
    context = {
        "draw_variance": {"heads": {s: a1_ffr[f"head-{s}"] for s in SLICES},
                          "range": round(max(draw_vals) - min(draw_vals), 4),
                          "std": round(float(np.std(draw_vals)), 4)},
        "seed_variance": {"baseline@42": seed_vals[0], "baseline@43": seed_vals[1],
                          "range": round(abs(seed_vals[0] - seed_vals[1]), 4)},
    }

    # --- (3) procrustes-aligned per-point deviation + spread-ratio on a1 ---
    def _proc(u, v):
        # scipy.procrustes standardizes both to unit Frobenius norm, aligns v to u.
        m1, m2, disparity = procrustes(u, v)
        dev = float(np.sqrt(((m1 - m2) ** 2).sum(axis=1)).mean())   # mean per-point deviation
        # spread-ratio: ratio of the two clouds' RMS radius (pre-standardization scale ratio)
        ru = float(np.sqrt((((u - u.mean(0)) ** 2).sum(1)).mean()))
        rv = float(np.sqrt((((v - v.mean(0)) ** 2).sum(1)).mean()))
        return {"disparity": round(disparity, 5), "mean_pointdev": round(dev, 5),
                "spread_ratio": round(max(ru, rv) / max(min(ru, rv), 1e-9), 4)}
    geom = {}
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        geom[f"head-{a} vs head-{b}"] = _proc(a1_xy[f"head-{a}"], a1_xy[f"head-{b}"])
    geom["base-s42 vs base-s43"] = _proc(a1_xy["base-s42"], a1_xy["base-s43"])

    # --- (2) rotation cross/self FFR on each slice's own truth ---
    rotation = {}
    member_adv = {}
    for X in SLICES:  # eval slice X
        xt = SB / f"draw-univ-{X}/edges-k15-fuzzy.npz"
        xs = _norm_load(SUB / f"draw-univ-{X}/substrate.f32.npy")
        xn = int(xs.shape[0])
        per_head = {}
        for H in SLICES:  # head H projecting X
            m = ParametricUMAP.load(str(heads[H]), device="cuda")
            xy = np.asarray(m.transform(xs, batch_size=8192), dtype=np.float32)
            per_head[H] = float(quick_ffr_v2(xy, xt, xn))
            del m
        self_ffr = per_head[X]
        cross = [per_head[H] for H in SLICES if H != X]
        rotation[f"eval-{X}"] = {"self(head-%s,member)" % X: self_ffr,
                                 "cross(unseen heads)": {H: per_head[H] for H in SLICES if H != X},
                                 "cross_mean": round(float(np.mean(cross)), 4)}
        member_adv[X] = round(self_ffr - float(np.mean(cross)), 4)
        print(f"  rotation eval-{X}: self {self_ffr:.4f} cross_mean {np.mean(cross):.4f} "
              f"member_adv {member_adv[X]:+.4f}", flush=True)

    out = SB / "draw-univ-score.json"
    out.write_text(json.dumps({
        "schema": "draw-univ-score-2026-08-30",
        "validity_gate": {"init_state_sha256": init_sha, "all_equal": gate_ok,
                          "note": "same seed 42 -> inits MUST be bit-identical; any difference below is DATA DRAW."},
        "shared_eval_a1_ffr": a1_ffr,
        "context_table": context,
        "geometry_agreement": geom,
        "rotation": rotation,
        "cross_vs_self_member_advantage": member_adv,
        "note": ("A/B/C are disjoint composition-matched 2M draws, same champion recipe + seed 42. "
                 "Draw-variance (range/std of A,B,C FFR on a1) sits next to seed-variance (baseline "
                 "@42 vs @43 on a1) in context_table. member_advantage = self(in-sample) - cross(unseen) "
                 "per eval slice. NOTE if validity_gate.all_equal is False the FFR differences conflate "
                 "init variance and the experiment is void."),
    }, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"\n[context] draw-variance range {context['draw_variance']['range']} vs "
          f"seed-variance range {context['seed_variance']['range']}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if gate_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
