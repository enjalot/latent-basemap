"""exp-2d dilution-matched within-image FFR (owner ratio sweep 2026-09-06). The joint map's 0.1%-disc is ~46%
text-diluted, so a fair within-image quality guard measures recall of the within-image-15 truth among the SAME
number of IMAGE neighbors (250) the image-only baseline used — not each map's own geometric 0.1% disc.

For each sampled image row: expand its 2D neighborhood until 250 IMAGE points (modality==0), then recall of its
within-image-15 truth among those. Reports cost = baseline_image_ffr (0.6371) - this, guard ≤0.10.

Usage: exp2d_dilution_ffr.py <ds_dir>   (reads <ds_dir>/champion-bs16k/coordinates.npy + <ds_dir>/knn_indices.npy)
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

N = 250000; K = 15; DISC_IMG = 250; BASELINE_IMG_FFR = 0.6371
MOD = "/data2/monet/exp2d-siglip-500k/modality.npy"


def main():
    d = Path(sys.argv[1])
    xy = np.asarray(np.load(d / "champion-bs16k" / "coordinates.npy"), dtype=np.float32)
    mod = np.load(MOD); knn = np.load(d / "knn_indices.npy", mmap_mode="r")   # within-image-15 truth
    rng = np.random.default_rng(0); q = rng.choice(N, min(10000, N), replace=False)   # image rows only
    tree = cKDTree(xy)
    _, nbr = tree.query(xy[q], k=1500, workers=-1)      # 1500 >> 250/(1-0.46) so ≥250 image survive dilution
    hit = tot = 0
    for r, qi in enumerate(q):
        cand = set(int(x) for x in nbr[r] if x != qi and mod[x] == 0)   # image neighbors
        # first DISC_IMG image neighbors in geometric order
        img_order = [int(x) for x in nbr[r] if x != qi and mod[x] == 0][:DISC_IMG]
        cs = set(img_order); truth = [int(t) for t in knn[qi][:K] if t != qi]
        if truth:
            hit += sum(t in cs for t in truth); tot += len(truth)
    ffr = round(hit / tot, 4) if tot else None
    cost = round(BASELINE_IMG_FFR - ffr, 4)
    res = {"dilution_matched_image_ffr": ffr, "baseline_image_ffr": BASELINE_IMG_FFR, "cost": cost,
           "n_image_neighbors": DISC_IMG, "guard_passed": bool(cost <= 0.10),
           "note": "recall of within-image-15 truth among 250 IMAGE neighbors (dilution-matched to the image-only baseline); guard cost ≤0.10"}
    (d / "dilution_ffr.json").write_text(json.dumps(res, indent=1))
    print(f"[dilution-ffr] {d.name}: matched image FFR {ffr} | baseline {BASELINE_IMG_FFR} | cost {cost} | guard {'PASS' if res['guard_passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
