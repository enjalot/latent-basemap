"""Evolution benchmark — trade-surface scorer v2 (5th review #4-5). .venv (scipy cKDTree).

Corrects the v1 instrument and is efficient at 8M (ONE cKDTree per snapshot, single query pass bucketed
into all cohorts — v1 rebuilt the tree per cohort call, ~48 builds, unusable at scale):
  DYNAMIC MEMBERSHIP: member iff query index < the ACTIVE head's training size at snapshot k (from the
    arm manifest's active_head_k), not a static n0.
  ARRIVAL COHORTS (ranges from coords shapes, no config): T0-retention [0:n_0]; per-tranche arrival
    [n_{k-1}:n_k] at snapshot k; reddit-as-its-own-cohort (OOD tranche REDDIT_K) at every snapshot from
    arrival on (pre/post-retrain reception); post-update = overall FFR after the last head switch.
  SERVICE latency/cost from the manifest (append-only armA v2 fills place_wall_s/update_wall_s/head_gpu_h;
    falls back to transform_wall_s for the diagnostic re-score of existing full-transform coords).
Truth = exact k15 knn_indices.npy per snapshot. Churn = shared first-n_{k-1} vs previous, radius-norm, cum.
ARMS_JSON {label:{dir,kind}}; HEAD_GPU_H {label:hours}; env EVOLBENCH_K(5), REDDIT_K(3), N_QUERIES(60000).
Output: evolbench-tradesurface-v2.json.
"""
import json, os, sys
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
K = int(os.environ.get("EVOLBENCH_K", "5"))
REDDIT_K = int(os.environ.get("REDDIT_K", "3"))
NQ = int(os.environ.get("N_QUERIES", "60000"))
KT = 15


def _radius(xy):
    return float(np.percentile(np.linalg.norm(xy - xy.mean(0), axis=1), 90))


def _snapshot_ffr(xy, knn_idx, n, cohorts):
    """ONE tree build + one query pass. cohorts: {name:(lo,hi)}. Returns {name: ffr or None}.
    A query q is scored into every cohort whose [lo,hi) contains q; 'overall' spans [0,n)."""
    from scipy.spatial import cKDTree
    disc = max(int(round(n * 0.001)), 1)
    rng = np.random.default_rng(0)
    q = rng.choice(n, size=min(NQ, n), replace=False)
    tree = cKDTree(xy)
    _, near = tree.query(xy[q], k=disc + 1, workers=8)
    hits = {c: 0.0 for c in cohorts}; tot = {c: 0 for c in cohorts}
    for qi in range(len(q)):
        qq = int(q[qi])
        truth = np.asarray(knn_idx[qq][:KT]); truth = truth[truth != qq]
        if truth.size == 0:
            continue
        disc_set = set(int(x) for x in near[qi]); disc_set.discard(qq)
        rec = len(set(int(t) for t in truth) & disc_set) / truth.size
        for c, (lo, hi) in cohorts.items():
            if lo <= qq < hi:
                hits[c] += rec; tot[c] += 1
    return {c: (round(hits[c] / tot[c], 4) if tot[c] else None) for c in cohorts}


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    arms = json.loads(os.environ.get("ARMS_JSON", "{}"))
    if not arms:
        raise SystemExit("ARMS_JSON required: {label:{dir,kind}}")
    head_gpu_h = json.loads(os.environ.get("HEAD_GPU_H", "{}"))
    out = {"schema": "evolbench-tradesurface-v2-2026-09-01", "reddit_k": REDDIT_K, "n_queries": NQ, "arms": {}}
    for label, cfg in arms.items():
        d = Path(cfg["dir"])
        mp = next((d / nm for nm in ("manifest.json", "armB-manifest.json") if (d / nm).exists()), None)
        man = json.loads(mp.read_text()) if mp else {"snapshots": []}
        snaps = {s["k"]: s for s in man.get("snapshots", [])}
        ns = {k: int(np.load(d / f"coords-S{k}.npy", mmap_mode="r").shape[0])
              for k in range(K + 1) if (d / f"coords-S{k}.npy").is_file()}
        n0 = ns.get(0, 0)
        rows = []; cum_churn = 0.0; prev = None; prev_n = 0; cum_cost = 0.0; last_switch = 0
        for k in sorted(ns):
            xy = np.asarray(np.load(d / f"coords-S{k}.npy"), dtype=np.float32); n = ns[k]
            knn = np.load(SB / f"evolbench-S{k}" / "knn_indices.npy", mmap_mode="r")
            churn_mean = churn_p95 = 0.0
            if prev is not None:
                disp = np.linalg.norm(xy[:prev_n] - prev, axis=1) / max(_radius(prev), 1e-9)
                churn_mean = float(disp.mean()); churn_p95 = float(np.percentile(disp, 95))
                cum_churn += churn_mean
            active_head_k = snaps.get(k, {}).get("active_head_k", 0 if cfg["kind"] == "A" else k)
            if snaps.get(k, {}).get("head_switch"):
                last_switch = k
            member_cut = ns.get(active_head_k, n0) if cfg["kind"] == "A" else n
            # build cohort ranges
            cohorts = {"overall": (0, n), "member": (0, member_cut), "unseen": (member_cut, n),
                       "T0_retention": (0, n0)}
            if k >= 1 and prev_n > 0:
                cohorts["arrival"] = (prev_n, n)
            if k >= REDDIT_K and (REDDIT_K - 1) in ns and REDDIT_K in ns:
                cohorts["reddit"] = (ns[REDDIT_K - 1], ns[REDDIT_K])
            ff = _snapshot_ffr(xy, knn, n, cohorts)
            s = snaps.get(k, {})
            place = s.get("place_wall_s"); update = s.get("update_wall_s")
            step_cost = update or place or s.get("transform_wall_s") or s.get("wall_s") or 0.0
            cum_cost += float(step_cost or 0.0)
            rows.append({"k": k, "n": n, "churn_mean": round(churn_mean, 5), "churn_p95": round(churn_p95, 5),
                         "cum_churn": round(cum_churn, 5),
                         "quality": {"ffr": ff["overall"], "ffr_member": ff["member"],
                                     "ffr_unseen": ff["unseen"], "member_cutoff": int(member_cut),
                                     "active_head_k": active_head_k},
                         "cohorts": {c: ff[c] for c in ("T0_retention", "arrival", "reddit") if c in ff},
                         "latency": {"place_wall_s": place, "update_wall_s": update,
                                     "fallback_transform_wall_s": s.get("transform_wall_s", s.get("wall_s"))},
                         "cum_gpu_s": round(cum_cost, 1)})
            print(f"[{label}] S{k}: churn {churn_mean:.4f}(cum {cum_churn:.4f}) FFR {ff['overall']} "
                  f"[mem {ff['member']} cut {member_cut:,}] T0ret {ff['T0_retention']} "
                  f"arr {ff.get('arrival')} reddit {ff.get('reddit')}", flush=True)
            prev = xy[:n].copy(); prev_n = n
        cost_h = round(cum_cost / 3600.0 + float(head_gpu_h.get(label, 0.0)), 3)
        out["arms"][label] = {"kind": cfg["kind"], "trajectory": rows, "last_switch_k": last_switch,
                              "head_gpu_h": head_gpu_h.get(label, 0.0), "total_gpu_h_incl_heads": cost_h,
                              "post_update_ffr": rows[-1]["quality"]["ffr"] if rows else None}
    OUT = SB / "evolbench-tradesurface-v2.json"
    OUT.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("\n=== TRADE SURFACE v2 ===", flush=True)
    for label, a in out["arms"].items():
        t = a["trajectory"][-1] if a["trajectory"] else {}
        print(f"  {label}: final FFR {t.get('quality',{}).get('ffr')} cum churn {t.get('cum_churn')} "
              f"GPU-h(incl heads) {a['total_gpu_h_incl_heads']}", flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
