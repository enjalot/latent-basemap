"""MiniLM mix-pilot per-register scorer (owner gate 2026-09-06). Scores ONE 2M map (minilm-mix-2m or
minilm-base-2m): overall v2 FFR + PER-REGISTER reception (held-out val projected through the champion head) +
social-SINK concentration (top-1%-cell mass + entropy per register). The driver runs this on both maps; the
mix-vs-base diff gives base-displacement (base reception drop) + social-sink.

Usage: minilm_mix_score.py <ds>   (minilm-mix-2m | minilm-base-2m) -> <ds>/register-score.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); DIM = 384; K = 15; GRID = 256


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    ds = sys.argv[1]; arm = SB / ds; ckpt = arm / "champion-bs16k"
    D = Path(f"/data2/monet/{ds}")                              # draw dir (substrate, register, held_out)
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from knobs_2m import quick_ffr_v2
    from minilm_mix_draw import Corpus, REGISTERS, BASE
    import faiss, torch
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    coords = np.asarray(np.load(ckpt / "coordinates.npy"), np.float32); n = coords.shape[0]
    reg = np.load(D / "register.npy")
    held = json.loads((D / "held_out.json").read_text())
    ffr_v2 = float(quick_ffr_v2(coords, arm / "edges-k15-fuzzy.npz", n))

    sub = np.load(D / "substrate.f32.npy", mmap_mode="r")
    t0 = time.time(); hdx = faiss.IndexFlatIP(DIM)
    for i in range(0, n, 500_000):
        hdx.add(np.ascontiguousarray(sub[i:i + 500_000]))
    d2x = faiss.IndexFlatL2(2); d2x.add(np.ascontiguousarray(coords)); disc = max(int(round(n * 0.001)), K)
    print(f"[mix-score {ds}] v2FFR {ffr_v2:.4f} | faiss built {time.time()-t0:.0f}s disc {disc}", flush=True)

    pumap = ParametricUMAP.load(str(ckpt / "model.pt"), device="cuda"); pumap.model.eval()

    # per-register reception: gather that register's held-out val, project through the head, recall@15 into THIS map
    per_reg = {}
    for rname in REGISTERS:
        vidx = np.array(held["val_idx"][rname], np.int64)
        if vidx.size == 0:
            continue
        vhd = _norm(Corpus(rname, rname in BASE).gather(np.sort(vidx)))
        with torch.no_grad():
            v2d = pumap.model(torch.from_numpy(vhd).cuda()).cpu().numpy().astype(np.float32)
        _, hd = hdx.search(vhd, K); _, dd = d2x.search(np.ascontiguousarray(v2d), disc)
        rec = np.mean([len(set(int(x) for x in hd[i]) & set(int(x) for x in dd[i])) / K for i in range(vhd.shape[0])])
        # sink: this register's TRAINING rows' 2D concentration (top-1%-cell mass + entropy on a GRID x GRID hist)
        rc = coords[reg == REGISTERS.index(rname)]
        lo, hi = coords.min(0), coords.max(0); span = np.where(hi > lo, hi - lo, 1.0)
        b = np.clip(((rc - lo) / span * (GRID - 1)).astype(int), 0, GRID - 1)
        flat = b[:, 0] * GRID + b[:, 1]; counts = np.bincount(flat, minlength=GRID * GRID).astype(np.float64)
        p = counts[counts > 0] / counts.sum(); ent = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
        top1 = float(np.sort(counts)[::-1][:max(1, len(p) // 100)].sum() / counts.sum())
        per_reg[rname] = {"reception@15": round(float(rec), 4), "n_val": int(vhd.shape[0]),
                          "top1pct_cell_mass": round(top1, 4), "entropy_norm": round(ent, 4), "n_rows": int((reg == REGISTERS.index(rname)).sum())}
        print(f"[mix-score {ds}] {rname}: recep {rec:.4f} top1% {top1:.4f} ent {ent:.4f}", flush=True)

    out = {"schema": "minilm-mix-pilot-score-2026-09-06", "ds": ds, "n_rows": int(n),
           "v2_ffr_own_truth": round(ffr_v2, 5), "per_register": per_reg,
           "note": "per-register reception = held-out val through the champion head, recall@15 into this map. sink = "
                   "training-row 2D concentration (top-1%-cell mass high + entropy low = collapsed). Compare mix vs base."}
    (arm / "register-score.json").write_text(json.dumps(out, indent=1))
    print(f"[mix-score {ds}] DONE -> {arm/'register-score.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
