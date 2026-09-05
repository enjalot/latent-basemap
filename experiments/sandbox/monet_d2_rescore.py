"""D2: uniform quick_ffr_v2 rescore of the cross-encoder/MONET maps (owner D-block, 2026-09-05). CPU.

The published summaries carry quick_ffr_at_0.1pct (v1: ID-ordered fuzzy-graph truth). D2 recomputes the
CORRECTED quick_ffr_v2 (exact-k15 knn truth, external review 2026-08-27) on the SAME coordinates so every
cross-encoder/MONET FFR number is on one uniform instrument before the articles. Auto-discovers champion dirs
that have coordinates.npy + summary.json[quick_ffr_at_0.1pct] AND a co-located edges-k15-fuzzy.npz +
knn_indices.npy (in the dataset dir = the champion dir's parent). Tabulates old vs v2 + delta.

Output: /data/latent-basemap/sandbox/d2-rescore-20260905.json + a printed table.
Usage: monet_d2_rescore.py [FILTER=monet,siglip,clip,jina,sisap,neomme,bl] [NQ=20000].
"""
import json, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
DEFAULT_FILTER = ("monet", "siglip", "clip", "jina", "sisap", "neomme", "bl")


def main():
    filt = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else DEFAULT_FILTER
    nq = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from knobs_2m import quick_ffr_v2

    targets = []
    for summ in sorted(SB.glob("*/*/summary.json")):
        arm_dir = summ.parent; ds_dir = arm_dir.parent
        ds = ds_dir.name
        if not any(f in ds for f in filt):
            continue
        try:
            s = json.loads(summ.read_text())
        except Exception:
            continue
        old = s.get("quick_ffr_at_0.1pct")
        coords = arm_dir / "coordinates.npy"
        edges = ds_dir / "edges-k15-fuzzy.npz"; knn = ds_dir / "knn_indices.npy"
        if old is None or not coords.exists() or not edges.exists():
            continue
        targets.append((ds, arm_dir.name, coords, edges, knn if knn.exists() else None, float(old)))

    print(f"[d2] {len(targets)} maps to rescore (filter={filt})", flush=True)
    rows = []
    for ds, arm, coords, edges, knn, old in targets:
        t0 = time.time()
        xy = np.load(coords)
        try:
            v2 = float(quick_ffr_v2(xy, edges, xy.shape[0], n_queries=nq,
                                    knn_indices_path=str(knn) if knn else None))
        except Exception as e:
            print(f"[d2] {ds}/{arm}: FAILED {e}", flush=True); continue
        rows.append({"dataset": ds, "arm": arm, "n": int(xy.shape[0]), "quick_ffr_at_0.1pct_v1": round(old, 4),
                     "quick_ffr_v2": round(v2, 4), "delta_v2_minus_v1": round(v2 - old, 4),
                     "has_exact_knn_truth": knn is not None})
        print(f"[d2] {ds}/{arm}: v1 {old:.4f} -> v2 {v2:.4f} ({v2-old:+.4f}) "
              f"{'exact-knn' if knn else 'fuzzy-truth'} ({time.time()-t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: r["dataset"])
    out = {"schema": "d2-quick-ffr-v2-rescore-2026-09-05", "n_queries": nq, "n_maps": len(rows),
           "note": "v1=quick_ffr_at_0.1pct (ID-ordered fuzzy truth); v2=quick_ffr_v2 (exact-k15 knn truth). "
                   "Uniform instrument across cross-encoder/MONET maps for the articles.", "maps": rows}
    (SB / "d2-rescore-20260905.json").write_text(json.dumps(out, indent=1))
    print(f"[d2] DONE {len(rows)} maps -> {SB/'d2-rescore-20260905.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
