"""Image-evolution scoring (owner item, 2026-09-05): does the anchored tier ABSORB the BL sink-collapse the
frozen 4M head produced? The image-space answer to the text-chain result. CPU (faiss/KDTree/frame) + one CPU
projection of BL through the frozen 4M head.

Preregistered metrics (frozen-4M baseline vs the anchored update model-wimgevo.pt):
1. reload-assert the first IMAGE MapState (model-wimgevo reload reproduces coords-wimgevo <=1e-4).
2. BL own-truth FFR (per segment) — predict gain >= +0.05: BL high-D 15NN among BL vs BL 2D 0.1%-disc, frozen vs updated.
3. Sink dispersal (per segment) — BL 2D top-1%-cell mass + occupancy entropy, frozen vs updated (was 98.4% top1% frozen 2M-2d).
4. Member churn — 4M members' displacement frozen(gated-4M coords)->updated (frame.py rigid gauge); predict <=0.02 of frame radius.
5. Retention — 4M member own-truth FFR (exact-k15 truth) frozen vs updated; predict >= -0.02.

Frozen BL coords = BL projected through the gated 4M head (CPU). Updated coords = coords-wimgevo (members [0,4M), BL [4M,5.08M)).
Output: image-evolution-4m-bl-20260905/score.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); IEV = SB / "image-evolution-4m-bl-20260905"
BL = Path("/data2/monet/bl-clip"); FOURM = Path("/data2/monet/random-clip-4m")
GATED = SB / "monet-random-clip-4m/champion-bs16k"
N4M = 4_000_000; K = 15; SEG = ["covers", "medium", "embellishments", "plates"]
HERE = Path(__file__).resolve().parent


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _occ(xy, ext, bins=256):
    (x0, x1), (y0, y1) = ext
    hx = np.clip(((xy[:, 0]-x0)/(x1-x0+1e-9)*bins).astype(int), 0, bins-1)
    hy = np.clip(((xy[:, 1]-y0)/(y1-y0+1e-9)*bins).astype(int), 0, bins-1)
    c = np.bincount(hx*bins+hy, minlength=bins*bins).astype(np.float64); p = c/c.sum(); nz = p[p > 0]
    return {"entropy_norm": round(float(-(nz*np.log(nz)).sum()/np.log(bins*bins)), 4),
            "top1pct_cell_mass": round(float(np.sort(p)[::-1][:max(1, bins*bins//100)].sum()), 4)}


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    import faiss
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(HERE)); import frame
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    rng = np.random.default_rng(0)

    bl = _norm(np.asarray(np.load(BL/"bl-clip.f32.npy", mmap_mode="r"), np.float32)); nbl = bl.shape[0]
    seg = np.load(BL/"bl-segment.npy")
    updated = np.load(IEV/"coords-wimgevo.npy")                 # (5.08M,2): members[0:4M], BL[4M:]
    up_mem = updated[:N4M].astype(np.float64); up_bl = updated[N4M:].astype(np.float64)

    # reload-assert (first image MapState)
    pu = ParametricUMAP.load(str(IEV/"model-wimgevo.pt"), device="cpu")
    smp = np.sort(rng.choice(updated.shape[0], 20000, replace=False))
    X0 = _norm(np.asarray(np.load(FOURM/"clip-substrate.f32.npy", mmap_mode="r")[smp[smp < N4M]], np.float32))
    re = np.asarray(pu.transform(X0, batch_size=8192), np.float32)
    reload_max = float(np.abs(re - updated[smp[smp < N4M]]).max())

    # frozen-4M BL coords (project BL through the gated head, CPU) + frozen members = gated coords
    pf = ParametricUMAP.load(str(GATED/"model.pt"), device="cpu")
    fr_bl = np.asarray(pf.transform(bl, batch_size=8192), np.float64)
    fr_mem = np.load(GATED/"coordinates.npy").astype(np.float64)

    # BL own-truth FFR per segment (frozen vs updated); truth = BL high-D 15NN among BL
    idx = faiss.IndexFlatIP(512); idx.add(bl)
    def ffr(coords2d, qpos):
        q = np.sort(rng.choice(qpos, min(nq, qpos.size), replace=False))
        _, hd = idx.search(bl[q], K+1)
        tree = cKDTree(coords2d); disc = max(int(round(nbl*0.001)), K)
        _, d2 = tree.query(coords2d[q], k=disc, workers=-1)
        r = np.empty(q.size)
        for i in range(q.size):
            ds = set(int(x) for x in d2[i]); r[i] = sum(int(x) in ds for x in hd[i] if int(x) != q[i]) / K
        return round(float(r.mean()), 4)
    seg_ffr = {}
    for si, s in enumerate(SEG):
        pos = np.where(seg == si)[0]
        f, u = ffr(fr_bl, pos), ffr(up_bl, pos)
        seg_ffr[s] = {"frozen": f, "updated": u, "gain": round(u-f, 4)}
        print(f"[imgevo] {s}: BL FFR frozen {f} -> updated {u} (gain {u-f:+.4f})", flush=True)
    all_f, all_u = ffr(fr_bl, np.arange(nbl)), ffr(up_bl, np.arange(nbl))

    # sink dispersal per segment (frozen vs updated), on the frozen member-map extent
    ext = ((float(fr_mem[:,0].min()), float(fr_mem[:,0].max())), (float(fr_mem[:,1].min()), float(fr_mem[:,1].max())))
    sink = {"all": {"frozen": _occ(fr_bl, ext), "updated": _occ(up_bl, ext)}}
    for si, s in enumerate(SEG):
        m = seg == si; sink[s] = {"frozen": _occ(fr_bl[m], ext), "updated": _occ(up_bl[m], ext)}

    # member churn (frozen gated coords -> updated members), frame.py rigid gauge
    disp, info = frame.churn(up_mem, fr_mem)
    churn = {"mean": round(float(disp.mean()), 5), "p95": round(float(np.percentile(disp, 95)), 5), "learned_scale": info.get("learned_scale")}

    # retention: 4M member own-truth FFR (exact-k15) frozen vs updated
    mknn = np.load(SB/"monet-random-clip-4m"/"knn_indices.npy", mmap_mode="r")
    def mem_ffr(coords2d):
        q = np.sort(rng.choice(N4M, nq, replace=False)); tree = cKDTree(coords2d); disc = max(int(round(N4M*0.001)), K)
        _, d2 = tree.query(coords2d[q], k=disc, workers=-1); r = np.empty(q.size)
        for i in range(q.size):
            ds = set(int(x) for x in d2[i]); tr = [int(t) for t in mknn[q[i]][:K] if t != q[i]]
            r[i] = sum(t in ds for t in tr)/len(tr) if tr else 0
        return round(float(r.mean()), 4)
    ret_f, ret_u = mem_ffr(fr_mem), mem_ffr(up_mem)

    out = {"schema": "image-evolution-score-2026-09-05", "first_image_mapstate": True,
           "reload_assert_max_coord_diff": reload_max, "reload_pass": bool(reload_max <= 1e-4),
           "bl_own_truth_ffr": {"all": {"frozen": all_f, "updated": all_u, "gain": round(all_u-all_f, 4)}, "by_segment": seg_ffr,
                                "gate_gain_ge_0.05": bool((all_u-all_f) >= 0.05)},
           "sink_dispersal": sink,
           "member_churn": churn, "member_churn_gate_le_0.02": bool(churn["mean"] <= 0.02),
           "retention_member_ffr": {"frozen": ret_f, "updated": ret_u, "delta": round(ret_u-ret_f, 4), "gate_ge_-0.02": bool(ret_u-ret_f >= -0.02)},
           "note": "frozen=gated 4M head; updated=anchored BL fine-tune (w=0.02). BL thumb-derived vs 4M original-derived (~0.08 offset rides along, flagged)."}
    (IEV/"score.json").write_text(json.dumps(out, indent=1))
    print(f"[imgevo] reload {reload_max:.2e} | BL FFR {all_f}->{all_u} ({all_u-all_f:+.4f}) | "
          f"sink top1% {sink['all']['frozen']['top1pct_cell_mass']}->{sink['all']['updated']['top1pct_cell_mass']} | "
          f"member churn {churn['mean']} | retention {ret_f}->{ret_u}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
