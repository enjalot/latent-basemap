"""Card-005 4-way comparison + movement (CPU, off-flock). DINO image-source arrival. Reception on eval-common-v2
(DINO-1536 truth, per-SOURCE query means — the reference is inherited from v1, NOT balanced; gates use fixed
per-source means, not a new truth instrument). Heads: frozen T0 / anchored / unanchored / fresh-final. Arriving =
3 synthetic sources; old = 5 real; diffusion-aesthetic-4k EXCLUDED from gate averages (diagnostic). Movement of old
anchor-holdout rows T0->update on the FIXED trained-T0 centroid-p90 radius (rigid fit on ACTIVE anchors, eval
HOLDOUT). Card-004 lessons: fresh-gap point-estimates, direct anchored-vs-unanchored (no universal-dominance),
cross-source recovery all 4 heads (conditional counts, per-query + edge-weighted). Overall gate NOT relaxed.
"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_jina_pair as EP
import frame as F
from card005_gates import arrival_gate

SEAL = Path("/data2/monet/eval-common-v2"); SB = Path("/data/latent-basemap/sandbox")
T0D = SB / "dino-arrival-t0"; UPD = T0D / "updates"
T0SUB = Path("/data/latent-basemap/substrates/dino-arrival-t0"); FINALSUB = Path("/data/latent-basemap/substrates/dino-arrival-final")
POOL_SRC = "/data2/monet/pool-20m/source.npy"
ARRIVING = ["synthetic-flux-klein", "synthetic-flux-schnell", "synthetic-z-image"]
OLD = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
EXCLUDE = ["diffusion-aesthetic-4k"]
HEADS = {"frozen_t0": T0D / "champion-bs16k/model.pt", "anchored": UPD / "model-anchored.pt",
         "unanchored": UPD / "model-unanchored.pt", "fresh_final": SB / "dino-arrival-final/champion-bs16k/model.pt"}


def main():
    ref_hd = np.asarray(np.load(SEAL / "ref_hd.f16.npy"), np.float32); val_hd = np.asarray(np.load(SEAL / "val_hd.f16.npy"), np.float32)
    truth = np.load(SEAL / "truth_val.npy"); src = np.load(SEAL / "val_source.npy", allow_pickle=True).astype(str)
    val_idx = np.load(SEAL / "val_idx.npy"); ref_idx = np.load(SEAL / "ref_idx.npy")
    pool_src = np.load(POOL_SRC, allow_pickle=True).astype(str); ref_src = pool_src[ref_idx]
    rng = np.random.default_rng(0)
    fixed_radius = json.load(open(T0D / "anchor.meta.json"))["fixed_radius_trained_T0_centroid_p90"]

    pq = {n: EP.per_query(ck, ref_hd, val_hd, truth) for n, ck in HEADS.items() if ck.exists()}
    np.savez(SB / "overseer-codex/card005-perq.npz", val_idx=val_idx, val_source=src,
             **{f"{h}_B{B}": pq[h][B] for h in pq for B in EP.BUDGETS})

    def m(a, c): return float(a[src == c].mean())
    rep = {h: {f"B{B}": {c: round(m(pq[h][B], c), 4) for c in sorted(set(src.tolist()))} for B in EP.BUDGETS} for h in pq}
    # natural corpus weights per source (pool counts)
    import collections
    natc = collections.Counter(pool_src.tolist())

    gains = {}
    for B in EP.BUDGETS:
        gains[f"B{B}"] = {}
        for c in ARRIVING:
            fz = m(pq["frozen_t0"][B], c); an = m(pq["anchored"][B], c); un = m(pq["unanchored"][B], c); fr = m(pq["fresh_final"][B], c)
            gap = fr - fz
            gains[f"B{B}"][c] = {"frozen": round(fz, 4), "anchored": round(an, 4), "unanchored": round(un, 4), "fresh": round(fr, 4),
                                 "anchored_gain": round(an - fz, 4), "anchored_frac_of_fresh_gap": round((an - fz) / gap, 3) if abs(gap) > 1e-9 else None}
        # aggregate arriving (source-mean over the 3 synthetic)
        da = np.concatenate([(pq["anchored"][B] - pq["frozen_t0"][B])[src == c] for c in ARRIVING])
        bs = np.array([da[rng.integers(0, da.size, da.size)].mean() for _ in range(2000)])
        gains[f"B{B}"]["arriving_agg"] = {"anchored_gain": round(float(da.mean()), 4),
                                          "ci95": (round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4))}

    # Gates: aggregate arriving gain with each arm's own CI; real retention.
    gate = {}
    for arm in ("anchored", "unanchored"):
        g = {"arriving_B250_gain_per_source": {c: round(m(pq[arm][250], c) - m(pq["frozen_t0"][250], c), 4) for c in ARRIVING}}
        old_d = {B: {c: m(pq[arm][B], c) - m(pq["frozen_t0"][B], c) for c in OLD} for B in EP.BUDGETS}
        g["old_mean"] = {f"B{B}": round(float(np.mean(list(old_d[B].values()))), 4) for B in EP.BUDGETS}
        g["old_worst"] = {f"B{B}": round(float(min(old_d[B].values())), 4) for B in EP.BUDGETS}
        g.update(arrival_gate(pq[arm][250], pq["frozen_t0"][250], src, ARRIVING))
        g["gate_old_stable"] = bool(all(np.mean(list(old_d[B].values())) >= -0.005 and min(old_d[B].values()) >= -0.01 for B in EP.BUDGETS))
        gate[arm] = g

    # cross-source true-neighbor recovery for ALL 4 heads (arriving queries' cross-source true-NN in the B250 disc)
    import faiss, torch; faiss.omp_set_num_threads(4)
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    arr_q = np.where(np.isin(src, ARRIVING))[0]
    xsrc = {}
    for arm in HEADS:
        if not HEADS[arm].exists():
            continue
        mo = ParametricUMAP.load(str(HEADS[arm]), device="cpu"); mo.model.eval()
        def proj(X):
            o = []
            with torch.no_grad():
                for i in range(0, X.shape[0], 50000):
                    o.append(mo.model(torch.from_numpy(EP._norm(X[i:i + 50000]))).numpy().astype(np.float32))
            return np.concatenate(o)
        rc = proj(ref_hd); vc = proj(val_hd[arr_q]); d2 = faiss.IndexFlatL2(rc.shape[1]); d2.add(np.ascontiguousarray(rc))
        _, nn = d2.search(np.ascontiguousarray(vc), 250)
        perq_rec, tot_cross, hit_cross, nqc = [], 0, 0, 0
        for j, qi in enumerate(arr_q):
            tset = set(int(x) for x in truth[qi]); cross = [t for t in tset if ref_src[t] not in ARRIVING]
            if cross:
                disc = set(int(x) for x in nn[j]); h = sum(t in disc for t in cross)
                perq_rec.append(h / len(cross)); tot_cross += len(cross); hit_cross += h; nqc += 1
        xsrc[arm] = {"per_query_macro": round(float(np.mean(perq_rec)), 4), "edge_weighted": round(hit_cross / tot_cross, 4) if tot_cross else None,
                     "n_arriving_q_with_cross": nqc, "n_cross_edges": tot_cross}

    # movement: fixed trained-T0 radius, anchor-holdout rows both arms
    t0_coords = np.asarray(np.load(T0D / "champion-bs16k/coordinates.npy"), np.float64)
    t0_idx = np.load(T0SUB / "draw_idx.npy"); final_idx = np.load(FINALSUB / "draw_idx.npy")
    common, t0_local, final_local = np.intersect1d(t0_idx, final_idx, assume_unique=True, return_indices=True)
    hold = np.load(UPD / "anchor_holdout_ids-anchored.npy"); hold_mask = np.isin(final_local, hold)
    active = np.load(UPD / "anchor_active_ids-anchored.npy"); active_mask = np.isin(final_local, active)
    mv = {"fixed_T0_centroid_p90_radius": fixed_radius, "holdout_n": int(hold_mask.sum())}
    for arm in ("anchored", "unanchored"):
        upd = np.asarray(np.load(UPD / f"coords-{arm}.npy"), np.float64)
        s0o, updo = t0_coords[t0_local], upd[final_local]
        # rigid (rotation+translation) fit on ACTIVE anchors, then apply to ALL old rows; evaluate holdout separately
        al_active, info = F.rigid_align(updo[active_mask], s0o[active_mask])   # fit transform on ACTIVE anchors
        R, tvec = np.asarray(info["R"], np.float64), np.asarray(info["t"], np.float64)
        aligned = updo @ R.T + tvec                                           # frame convention: aligned = src @ R.T + t
        disp = np.linalg.norm(aligned - s0o, axis=1) / fixed_radius           # evaluate on holdout (below), active, all
        def st(msk): d = disp[msk]; return {"n": int(msk.sum()), "mean": round(float(d.mean()), 6), "p95": round(float(np.percentile(d, 95)), 6), "p99": round(float(np.percentile(d, 99)), 6)}
        mv[arm] = {"anchor_holdout": st(hold_mask), "active": st(active_mask), "all_retained": st(np.ones(disp.shape[0], bool))}
        mv[arm]["gate_movement_holdout"] = bool(mv[arm]["anchor_holdout"]["mean"] <= 0.01 and mv[arm]["anchor_holdout"]["p99"] <= 0.05)

    overall = bool(gate["anchored"]["gate_new_source"] and gate["anchored"]["gate_old_stable"] and mv["anchored"]["gate_movement_holdout"])
    out = {"schema": "card005-compare-2026-09-10", "reception_per_source": rep, "arriving_gains": gains,
           "reception_gates": gate, "cross_source_recovery_B250": xsrc, "movement_fixed_T0_radius": mv,
           "OVERALL_PREREG_GATE_PASS": overall,
           "note": "eval-common-v2 reference inherited from v1 (NOT balanced); per-source query means. Arriving=3 synthetic, "
                   "old=5 real, diffusion-aesthetic-4k excluded from gate averages. Movement on FIXED trained-T0 centroid-p90 "
                   "radius; rigid rotation+translation, no scale/radius redefinition. Single update/seed = operational example."}
    (SB / "overseer-codex/card005-compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"overall": overall, "arriving_gains_B250": {c: gains["B250"][c]["anchored_gain"] for c in ARRIVING},
                      "arriving_agg": gains["B250"]["arriving_agg"], "gates": gate,
                      "movement_holdout": {a: mv[a]["anchor_holdout"] for a in ("anchored", "unanchored")},
                      "crosssource": xsrc}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
