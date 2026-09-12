"""Card013 per-row radii (per card013-prereg.md). r_i = sqrt(mean_j d_HD(i,j)^2) over the 15 directed
fixed15 neighbors (frozen card010 knn_dist, cols 1..15; self col0 excluded), divided by training p95;
numerical floor 1e-6 AFTER normalization. STOP (exit 3) if >0.1% of rows need the floor. Also builds the
shuffled-radii control (same per-SOURCE distribution permuted within each source). CPU-only; touches NO
core/runtime files. Persists r_actual, r_shuffled, and a bound identity (sha). Usage: build_card013_radii.py
"""
import os, sys, json, hashlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np

SUBD = Path("/data/latent-basemap/substrates/card010-adaptive")
OUTD = Path("/data/latent-basemap/sandbox/card013-radii"); OUTD.mkdir(parents=True, exist_ok=True)
OC = Path("/data/latent-basemap/sandbox/overseer-codex")
SOURCES = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
K = 15; FLOOR = 1e-6; SEED = 13013


def _sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    knn_dist = np.load(SUBD / "knn_dist.npy")                      # (300K,61) L2; col0=self dist0
    dsrc = np.load(SUBD / "draw_source.npy", allow_pickle=True).astype(str)
    n = knn_dist.shape[0]
    d15 = knn_dist[:, 1:K + 1].astype(np.float64)                  # 15 nearest real neighbors
    assert np.isfinite(d15).all() and (d15 >= 0).all(), "knn_dist not finite/nonneg"
    r_raw = np.sqrt((d15 ** 2).mean(axis=1))                       # RMS high-D radius
    assert np.isfinite(r_raw).all() and (r_raw > 0).all(), "r_raw not finite/positive"
    p95 = float(np.percentile(r_raw, 95))
    r = r_raw / p95
    need_floor = int((r < FLOOR).sum()); floor_frac = need_floor / n
    r = np.maximum(r, FLOOR).astype(np.float32)
    # shuffled: permute WITHIN each source (same per-source distribution, location shuffled)
    rng = np.random.default_rng(SEED); r_shuf = r.copy()
    for s in SOURCES:
        idx = np.where(dsrc == s)[0]
        r_shuf[idx] = r[idx][rng.permutation(idx.size)]
    np.save(OUTD / "r_actual.npy", r); np.save(OUTD / "r_shuffled.npy", r_shuf)
    result = {"schema": "card013-radii-2026-09-11", "n": int(n), "k": K, "train_p95": round(p95, 6),
              "floor": FLOOR, "need_floor_count": need_floor, "need_floor_frac": round(floor_frac, 6),
              "floor_ok_le_0.1pct": bool(floor_frac <= 0.001),
              "r_actual_sha": _sha(r), "r_shuffled_sha": _sha(r_shuf),
              "r_deciles": [round(float(x), 5) for x in np.percentile(r, np.arange(10, 100, 10))],
              "r_min": round(float(r.min()), 6), "r_max": round(float(r.max()), 6),
              "per_source_mean_r": {s: round(float(r[dsrc == s].mean()), 5) for s in SOURCES},
              "VIABLE": bool(floor_frac <= 0.001 and np.isfinite(r).all() and (r > 0).all())}
    (OC / "card013-radii.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
    return 0 if result["VIABLE"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
