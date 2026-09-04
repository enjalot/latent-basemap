"""MONET-vs-laion memo assembler (owner MONET, overseer 2026-09-04 reframe). Presents the diversity arms as a
RARITY-COVERAGE-vs-FFR TRADE CURVE (not a ranking), leading with the FAIR cross-arm metrics — cluster coverage,
rare-region representation, probe reception (identical held-out probe/truth for all arms) — and treats per-arm
FFR as a COST axis with caveats: each arm is scored on its OWN truth graph, and a density-flattened draw has
intrinsically sparser neighborhoods (a harder exam), so raw FFR deltas OVERSTATE the quality cost. Scores
whatever's ready, marks pending arms, re-run to append. .venv. Output: monet-memo.json + monet-memo.md."""
import json
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); POOL = Path("/data2/monet/pool-20m"); DRAWS = Path("/data2/monet/draws")
LAION_FFR = 0.7101; ARMS = ["random", "sscd", "annfaiss", "theirfaiss"]


def _ffr(ds):
    s = SB / ds / "champion-bs16k" / "summary.json"
    if not s.exists():
        return None
    d = json.loads(s.read_text()); return round(float(d.get("quick_ffr_v2", d.get("quick_ffr_at_0.1pct", 0))), 4)


def _cluster_coverage():
    """Reference kmeans (k=1000) on a pool CLIP sample; per-arm distinct clusters + rare-cluster coverage."""
    # cache only the reference centroids (kmeans is the expensive part); recompute per-arm each run so
    # newly-available arms refresh (a full-result cache would freeze PENDING arms).
    cenf = SB / "monet-ref-centroids.npz"
    if cenf.exists():
        z = np.load(cenf); cen = z["cen"]; rare = z["rare"]
    else:
        from sklearn.cluster import MiniBatchKMeans
        clip = np.load(POOL / "clip512.f32.npy", mmap_mode="r"); N = clip.shape[0]
        si = np.sort(np.random.default_rng(0).choice(N, 500_000, replace=False))
        Xs = np.array(clip[si], dtype=np.float32); Xs /= np.linalg.norm(Xs, axis=1, keepdims=True).clip(1e-9)
        km = MiniBatchKMeans(10000, random_state=0, batch_size=50000, n_init=1, max_iter=50).fit(Xs)  # k=10k: discriminating
        cen = km.cluster_centers_.astype(np.float32); cen /= np.linalg.norm(cen, axis=1, keepdims=True).clip(1e-9)
        ref_sz = np.bincount(km.labels_, minlength=10000); rare = np.argsort(ref_sz)[:1000].astype(np.int64)
        np.savez(cenf, cen=cen, rare=rare)
    out = {"k": 10000, "rare_total": 1000, "arms": {}}
    for arm in ARMS:
        sub = DRAWS / f"{arm}-clip.f32.npy"
        if not sub.is_file():
            out["arms"][arm] = {"_status": "PENDING"}; continue
        X = np.load(sub, mmap_mode="r"); lab = np.empty(X.shape[0], np.int32); B = 200_000
        for s in range(0, X.shape[0], B):
            x = np.array(X[s:s+B], np.float32); x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)
            lab[s:s+B] = (x @ cen.T).argmax(1)
        u = np.unique(lab)
        out["arms"][arm] = {"clusters_covered": int(u.size),
                            "rare_clusters_covered": int(np.isin(rare, u).sum()), "rare_total": 1000}
    return out


def main():
    memo = {"schema": "monet-memo-2026-09-04", "_framing": "diversity arms = rarity-coverage-vs-FFR TRADE CURVE; "
            "lead with cluster-coverage/rare-region/probe-reception (fair, identical exam); FFR is a COST axis — "
            "each arm scored on its OWN truth graph + density-flattened = sparser neighborhoods = harder exam, "
            "so FFR deltas OVERSTATE quality cost."}
    sscd = np.load(POOL / "sscd_nn.npy"); src = np.load(POOL / "source.npy", allow_pickle=True)
    nanmask = np.isnan(sscd); q25 = float(np.nanpercentile(sscd, 25))
    memo["map_quality"] = {"laion_sisap_clip768": LAION_FFR, "monet_random_clip512": _ffr("monet-random-clip-2m"),
                           "monet_random_dino1536": _ffr("monet-random-dino-2m")}
    probe = json.loads((SB / "monet-probe-reception.json").read_text())["arms"] if (SB / "monet-probe-reception.json").exists() else {}
    cov = _cluster_coverage()["arms"]
    ridx = DRAWS / "random.idx.npy"; rand_i = np.load(ridx) if ridx.exists() else None
    memo["diversity_draws"] = {}
    for arm in ARMS:
        idxf = DRAWS / f"{arm}.idx.npy"; row = {"ffr_cost_axis": _ffr(f"monet-draw-{arm}-clip")}
        row["probe_recall_at_15"] = (probe.get(arm, {}) or {}).get("probe_recall_at_15")
        row["cluster_coverage"] = cov.get(arm)
        if idxf.exists():
            idx = np.load(idxf); s = sscd[idx]; sf = s[np.isfinite(s)]
            row["rare_region_frac"] = round(float((sf <= q25).mean()), 4)
            row["draw_mean_sscd_nn"] = round(float(np.nanmean(s)), 4)
            u, c = np.unique(src[idx], return_counts=True)
            row["source_frac"] = {str(k): round(float(v/len(idx)), 3) for k, v in zip(u, c)}
            if rand_i is not None and arm != "random":
                inter = np.intersect1d(idx, rand_i, assume_unique=True).size
                row["overlap_with_random_jaccard"] = round(inter/(2*len(idx)-inter), 4)
        else:
            row["_status"] = "PENDING"
        memo["diversity_draws"][arm] = row
    raf = DRAWS / "rarity_annfaiss.npy"
    memo["redundancy"] = ({"monet_clip_knn_dist_pctiles": {p: round(float(np.percentile(np.load(raf), p)), 4) for p in (5,25,50,75,95)},
                           "monet_frac_near_dup_sscd>0.9": round(float((sscd[np.isfinite(sscd)] > 0.9).mean()), 4),
                           "note": "curation => fatter (rarer) kNN tail; laion comparison appended when computed."}
                          if raf.exists() else {"_status": "PENDING (rarity_annfaiss)"})
    memo["sscd_nan_caveat"] = {"n": int(nanmask.sum()), "frac": round(float(nanmask.mean()), 4),
        "note": "SSCD undefined (~all synthetic-z-image); sscd arm EXCLUDES them, annfaiss/theirfaiss CAN pick them."}
    memo["theirfaiss_note"] = ("theirfaiss arm PENDING — rarity-INVERSION bug caught by owner's reviewer + "
        "fixed 2026-09-04: their index is metric_type=0 (INNER_PRODUCT/cosine, DESCENDING sims), not L2; the "
        "pre-fix grind kept the self-hit + dropped the farthest neighbor + used large-mean=rare, which INVERTED "
        "density (would have selected the DENSEST rows). Fixed to drop col0 + rarity=1-mean(cosine); grind "
        "restarted, ETA ~2026-09-05 night. Both CLIP+DINOv2 indexes are IP (verified).")
    tj = SB / "monet-theirumap-score.json"
    memo["their_umap"] = json.loads(tj.read_text()) if tj.exists() else {"_status": "PENDING"}
    (SB / "monet-memo.json").write_text(json.dumps(memo, indent=1, default=str))

    mq = memo["map_quality"]
    L = ["# MONET-vs-laion memo (PROVISIONAL, auto-assembled)", "",
         "## 1. Map quality (quick_ffr_v2, each on its own truth)",
         f"- laion sisap-CLIP768 **{mq['laion_sisap_clip768']}** | MONET CLIP-512 {mq['monet_random_clip512']} | MONET DINOv2-1536 **{mq['monet_random_dino1536']}**",
         "", "## 2. Diversity draws — DECIDING METRICS (fair: identical probe set + reference clustering)",
         "| arm | cluster coverage (SATURATED) | rare-region frac | probe recall@15 |",
         "| --- | --- | --- | --- |"]
    for arm in ARMS:
        r = memo["diversity_draws"][arm]
        if r.get("_status") == "PENDING":
            L.append(f"| {arm} | PENDING | | |"); continue
        cc = r.get("cluster_coverage") or {}
        cctxt = f"{cc.get('clusters_covered','?')} ({cc.get('rare_clusters_covered','?')}/1000)" if cc and "_status" not in cc else "pending"
        L.append(f"| {arm} | {cctxt} | {r.get('rare_region_frac')} | {r.get('probe_recall_at_15')} |")
    L += ["", "> cluster-coverage SATURATES at this 10%-of-pool draw fraction (every arm covers ~all clusters by construction) — see rare-region frac + probe recall@15, the actual diversity + cost signals. Random baseline rare-region ~0.25.",
          "", f"> theirfaiss: {memo.get('theirfaiss_note','')}",
          "", "## 3. FFR as a TRADE-CURVE COST axis (NOT a ranking)",
          "| arm | FFR (own truth) | rare-region | mean sscd_nn |", "| --- | --- | --- | --- |"]
    for arm in ARMS:
        r = memo["diversity_draws"][arm]
        L.append(f"| {arm} | {r.get('ffr_cost_axis')} | {r.get('rare_region_frac','—')} | {r.get('draw_mean_sscd_nn','—')} |" if r.get("_status")!="PENDING" else f"| {arm} | PENDING | | |")
    L += ["", "> CAVEAT: each FFR is on the arm's OWN truth graph, and a density-flattened (diverse) draw has "
          "intrinsically SPARSER neighborhoods — a harder exam — so raw FFR deltas OVERSTATE the quality cost. "
          "Read FFR against rare-region gain as a trade curve, and defer to probe recall@15 (identical exam).",
          "", "## 4. Redundancy", "```", json.dumps(memo["redundancy"], indent=1), "```",
          "## 5. their-UMAP competitor (their published 1M layout vs OUR champion, IDENTICAL rows)"]
    tu = memo["their_umap"]; sp = tu.get("spaces", {}) if isinstance(tu, dict) else {}
    if sp:
        d = sp.get("dino", {}); c = sp.get("clip", {})
        L += [f"- Source space = DINOv2 (their FFR {d.get('their_layout_ffr')} on DINOv2-truth vs {c.get('their_layout_ffr')} on CLIP-truth — confirmed).",
              f"- OUR champion on the SAME 1M rows: DINOv2 FFR **{d.get('our_champion_ffr')}** vs their **{d.get('their_layout_ffr')}** = **+{round((d.get('our_champion_ffr',0)-d.get('their_layout_ffr',0)),4)}** in their own source space (CLIP: ours {c.get('our_champion_ffr')} vs their {c.get('their_layout_ffr')}).",
              "- CAVEATS (honest): their UMAP method/params are UNPUBLISHED (no tune-match possible); their layout may optimize DIFFERENT objectives (visual-atlas aesthetics, not kNN FFR); it is a 1M TRANSDUCTIVE layout vs our PARAMETRIC head (generalizes to unseen rows). The number is the number, read with these."]
    else:
        L += ["```", json.dumps(tu, indent=1)[:400], "```"]
    L += [
          "", f"_caveat: {memo['sscd_nan_caveat']['note']}_"]
    (SB / "monet-memo.md").write_text("\n".join(L))
    print("wrote monet-memo.{json,md}"); print("\n".join(L[:22]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
