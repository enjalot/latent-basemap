#!/usr/bin/env python3
"""Image-space universality scorer (sisap-CLIP, owner 2026-08-30). GPU.

Mirrors p_drawuniv_score for CLIP768. Three same-seed heads A/B/C on disjoint 2M draws + a shared
neutral eval slice D (disjoint from A/B/C). VALIDITY GATE: the three head init_state_sha256 must be
equal (D768 → different hash from MiniLM; equality-across-three is the gate). Readouts:
  (1) FFR SPREAD on the shared eval D = quality draw-variance + collapse/fog/occupancy per head.
  (2) ROTATION cross/self FFR: head H on each slice X's own truth; self=head X on X (member),
      cross=head Y!=X (unseen); cross-vs-self member advantage.
  (3) PROCRUSTES-aligned per-point deviation + spread-ratio on the D projections = geometric agreement.
Output: /data/latent-basemap/sandbox/img-univ-score.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

SB = Path("/data/latent-basemap/sandbox")
SUB = Path("/data/latent-basemap/substrates")
HEADS = ("A", "B", "C")


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
    try:
        from analysis_v2 import map_quality as _mq
    except Exception:
        _mq = None

    models = {h: SB / f"img-univ-{h}/champion-bs16k/model.pt" for h in HEADS}
    for h, p in models.items():
        if not Path(p).exists():
            raise SystemExit(f"missing head {h}: {p}")

    init_sha = {h: json.loads((SB / f"img-univ-{h}/champion-bs16k/summary.json").read_text())
                .get("init_state_sha256") for h in HEADS}
    gate_ok = len(set(init_sha.values())) == 1 and None not in init_sha.values()
    print(f"[validity gate] init {init_sha} -> {'PASS' if gate_ok else 'FAIL'}", flush=True)

    # (1) shared eval D
    D_x = _norm_load(SUB / "img-univ-D/substrate.f32.npy")
    D_truth = SB / "img-univ-D/edges-k15-fuzzy.npz"
    D_n = int(D_x.shape[0])
    d_ffr, d_xy, d_quality = {}, {}, {}
    for h in HEADS:
        m = ParametricUMAP.load(str(models[h]), device="cuda")
        xy = np.asarray(m.transform(D_x, batch_size=8192), dtype=np.float32)
        d_xy[h] = xy
        d_ffr[h] = float(quick_ffr_v2(xy, D_truth, D_n))
        if _mq is not None:
            try:
                q = _mq(xy)
                d_quality[h] = {"collapse": float(q["collapse"]["r10_over_radius_times_sqrt_n"]),
                                "fog": float(q["fog"]["fog"]),
                                "occupancy": float(q["fog"]["occupied_bin_fraction"])}
            except Exception as e:
                d_quality[h] = {"error": f"{type(e).__name__}"}
        print(f"  D FFR [{h}] = {d_ffr[h]:.4f} q={d_quality.get(h)}", flush=True)
        del m
    dv = list(d_ffr.values())
    draw_variance = {"heads": d_ffr, "range": round(max(dv) - min(dv), 4),
                     "std": round(float(np.std(dv)), 4)}

    # (3) procrustes on D projections
    def _proc(u, v):
        m1, m2, disparity = procrustes(u, v)
        dev = float(np.sqrt(((m1 - m2) ** 2).sum(axis=1)).mean())
        ru = float(np.sqrt((((u - u.mean(0)) ** 2).sum(1)).mean()))
        rv = float(np.sqrt((((v - v.mean(0)) ** 2).sum(1)).mean()))
        return {"disparity": round(disparity, 5), "mean_pointdev": round(dev, 5),
                "spread_ratio": round(max(ru, rv) / max(min(ru, rv), 1e-9), 4)}
    geom = {f"{a} vs {b}": _proc(d_xy[a], d_xy[b])
            for i, a in enumerate(HEADS) for b in HEADS[i + 1:]}

    # (2) rotation on each slice's own truth
    rotation, member_adv = {}, {}
    for X in HEADS:
        xs = _norm_load(SUB / f"img-univ-{X}/substrate.f32.npy")
        xt = SB / f"img-univ-{X}/edges-k15-fuzzy.npz"
        xn = int(xs.shape[0])
        per = {}
        for H in HEADS:
            m = ParametricUMAP.load(str(models[H]), device="cuda")
            xy = np.asarray(m.transform(xs, batch_size=8192), dtype=np.float32)
            per[H] = float(quick_ffr_v2(xy, xt, xn)); del m
        cross = [per[H] for H in HEADS if H != X]
        rotation[f"eval-{X}"] = {"self_member": per[X],
                                 "cross_unseen": {H: per[H] for H in HEADS if H != X},
                                 "cross_mean": round(float(np.mean(cross)), 4)}
        member_adv[X] = round(per[X] - float(np.mean(cross)), 4)
        print(f"  rotation eval-{X}: self {per[X]:.4f} cross {np.mean(cross):.4f} "
              f"member_adv {member_adv[X]:+.4f}", flush=True)

    out = SB / "img-univ-score.json"
    out.write_text(json.dumps({
        "schema": "img-univ-score-2026-08-30", "space": "sisap-CLIP768",
        "validity_gate": {"init_state_sha256": init_sha, "all_equal": gate_ok},
        "shared_eval_D_ffr": d_ffr, "shared_eval_D_quality": d_quality,
        "draw_variance": draw_variance, "geometry_agreement": geom,
        "rotation": rotation, "cross_vs_self_member_advantage": member_adv,
        "note": ("image-space replication of the MiniLM draw-universality. A/B/C disjoint 2M CLIP draws, "
                 "same seed 42. Draw-variance = range/std of A,B,C FFR on the shared eval D. member_adv "
                 "= self(in-sample) - cross(unseen). If validity_gate.all_equal is False the experiment "
                 "is void. No seed-variance pair on the image side (that's the MiniLM result); this is "
                 "the draw-interchangeability check for CLIP space."),
    }, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"\n[draw-variance] D-FFR range {draw_variance['range']} std {draw_variance['std']}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if gate_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
