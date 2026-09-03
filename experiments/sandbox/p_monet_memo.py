"""MONET-vs-laion memo assembler (owner MONET, overseer 2026-09-03). Scores whatever's ready, marks pending
arms (e.g. theirfaiss), re-run to append. .venv. Sections:
 1. map quality per space: monet-random-{clip,dino}-2m FFR vs laion sisap-clip-2m (0.7101, reuse).
 2. diversity draws: per arm FFR + rare-region (draw's sscd_nn vs pool = common yardstick) + source composition
    + pairwise overlap vs random. Does an index-driven draw beat random?
 3. redundancy: MONET pool CLIP kNN-dist (rarity_annfaiss) vs laion sample — curation => fatter kNN distances.
 4. their-UMAP competitor: monet-theirumap-score.json (their layout FFR + our champion + source-space verdict).
Output: monet-memo.json + monet-memo.md."""
import json
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); POOL = Path("/data2/monet/pool-20m"); DRAWS = Path("/data2/monet/draws")
LAION_FFR = 0.7101   # sisap-clip-2m champion quick_ffr_v2 (existing, reuse)
ARMS = ["random", "sscd", "annfaiss", "theirfaiss"]


def _ffr(ds):
    s = SB / ds / "champion-bs16k" / "summary.json"
    if not s.exists():
        return None
    d = json.loads(s.read_text())
    return round(float(d.get("quick_ffr_v2", d.get("quick_ffr_at_0.1pct", 0))), 4)


def main():
    memo = {"schema": "monet-memo-2026-09-03", "_status": "auto-assembled; pending arms marked"}
    sscd = np.load(POOL / "sscd_nn.npy"); src = np.load(POOL / "source.npy", allow_pickle=True)
    nanmask = np.isnan(sscd)                # 0.40% NaN: SSCD undefined, ~all synthetic-z-image
    q25 = float(np.nanpercentile(sscd, 25))   # rarest quartile = LOW sscd_nn (high=near-dup); NaN-safe
    memo_nan = {"n_nan_sscd_nn": int(nanmask.sum()), "frac": round(float(nanmask.mean()), 4),
                "by_source": {str(k): int(v) for k, v in zip(*np.unique(src[nanmask], return_counts=True))},
                "note": "SSCD near-neighbor undefined for these (mostly synthetic-z-image). The sscd draw arm "
                        "EXCLUDES them (rarity undefined); annfaiss/theirfaiss CAN select them (CLIP-based) — inter-arm caveat."}

    # 1. map quality
    memo["map_quality"] = {"laion_sisap_clip768": LAION_FFR,
                           "monet_random_clip512": _ffr("monet-random-clip-2m"),
                           "monet_random_dino1536": _ffr("monet-random-dino-2m")}

    # 2. diversity draws
    memo["diversity_draws"] = {}
    ridx = DRAWS / "random.idx.npy"
    rand_i = np.load(ridx) if ridx.exists() else None
    for arm in ARMS:
        idxf = DRAWS / f"{arm}.idx.npy"
        row = {"ffr": _ffr(f"monet-draw-{arm}-clip")}
        if idxf.exists():
            idx = np.load(idxf); s = sscd[idx]; sf = s[np.isfinite(s)]
            row["draw_mean_sscd_nn"] = round(float(np.nanmean(s)), 4)       # lower = rarer (finite only)
            row["frac_in_rarest_quartile"] = round(float((sf <= q25).mean()), 4)  # random~0.25, among finite
            row["nan_sscd_in_draw"] = int(np.isnan(s).sum())
            u, c = np.unique(src[idx], return_counts=True)
            row["source_frac"] = {str(k): round(float(v / len(idx)), 3) for k, v in zip(u, c)}
            if rand_i is not None and arm != "random":
                inter = np.intersect1d(idx, rand_i, assume_unique=True).size
                row["overlap_with_random_jaccard"] = round(inter / (2 * len(idx) - inter), 4)
        else:
            row["_status"] = "PENDING"
        memo["diversity_draws"][arm] = row

    # 3. redundancy (MONET CLIP kNN-dist vs laion) — MONET side needs rarity_annfaiss
    raf = DRAWS / "rarity_annfaiss.npy"
    if raf.exists():
        r = np.load(raf)
        memo["redundancy"] = {"monet_clip_knn_dist_pctiles": {p: round(float(np.percentile(r, p)), 4) for p in (5, 25, 50, 75, 95)},
                              "note": "MONET pool CLIP-512 mean-kNN-dist (cuVS int8). Curation => fatter (rarer) tail. "
                                      "laion kNN-dist comparison appended when computed; sscd_nn (shipped) also gates near-dups."}
        memo["redundancy"]["monet_sscd_nn_pctiles"] = {p: round(float(np.percentile(sscd, p)), 4) for p in (50, 90, 95, 99)}
        memo["redundancy"]["monet_frac_near_dup_sscd>0.9"] = round(float((sscd > 0.9).mean()), 4)
    else:
        memo["redundancy"] = {"_status": "PENDING (rarity_annfaiss density)"}

    memo["sscd_nan_caveat"] = memo_nan
    # 4. their-UMAP competitor
    tj = SB / "monet-theirumap-score.json"
    memo["their_umap"] = json.loads(tj.read_text()) if tj.exists() else {"_status": "PENDING"}

    (SB / "monet-memo.json").write_text(json.dumps(memo, indent=1, default=str))
    # markdown
    L = ["# MONET-vs-laion memo (auto-assembled, PROVISIONAL)", ""]
    mq = memo["map_quality"]
    L += ["## 1. Map quality (quick_ffr_v2)",
          f"- laion sisap-CLIP768: {mq['laion_sisap_clip768']}",
          f"- MONET random CLIP-512: {mq['monet_random_clip512']}",
          f"- MONET random DINOv2-1536: {mq['monet_random_dino1536']}", ""]
    L += ["## 2. Diversity draws (does an index-driven draw beat random?)",
          "| arm | FFR | mean sscd_nn (lower=rarer) | rarest-quartile frac | overlap-w-random |",
          "| --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        r = memo["diversity_draws"][arm]
        if r.get("_status") == "PENDING":
            L.append(f"| {arm} | PENDING | | | |")
        else:
            L.append(f"| {arm} | {r['ffr']} | {r.get('draw_mean_sscd_nn')} | {r.get('frac_in_rarest_quartile')} | {r.get('overlap_with_random_jaccard','—')} |")
    L += ["", "## 3. Redundancy", "```", json.dumps(memo["redundancy"], indent=1), "```",
          "", "## 4. their-UMAP competitor", "```", json.dumps(memo["their_umap"], indent=1)[:800], "```"]
    (SB / "monet-memo.md").write_text("\n".join(L))
    print("wrote monet-memo.json + monet-memo.md")
    print("\n".join(L[:24]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
