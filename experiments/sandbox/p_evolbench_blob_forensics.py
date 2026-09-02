"""Reddit-blob forensics (owner investigation program, overseer 2026-09-02). CPU. In an anchored λ map the
owner sees two large reddit "sink" blobs that pull nearby base data in, while the rest of reddit spreads.
This characterizes them: (a) grid-density blob detection on the reddit cohort's final 2D positions -> top-2
dense components + diffuse remainder; (b) per blob vs diffuse vs base baseline: n, 2D area/density, frozen
spread, embedding geometry (norm, within-cosine, dist-to-nearest-base = true OOD-ness); (c) "sucked-in" base
rows terminating near a blob centroid (count, source datasets, distance travelled); (d) semantic spot-check
pulling REAL text (row->prov->parquet chunk_text) for top-moved + stationary-adjacent rows.

Provenance is REAL (substrate provenance.npy -> reddit/base chunk parquets, row-aligned 500k shards). If a
corpus's chunk parquets are missing we FAIL-CLOSED for that corpus's text (annotate; never NN-reconstruct).
Usage: p_evolbench_blob_forensics.py <wtag>   (e.g. 0.02, 0.1). Output: evolbench-blob-forensics-w<tag>.json"""
import json, sys, glob
from pathlib import Path
import numpy as np

import os
SB = Path("/data/latent-basemap/sandbox")
# env-parameterized so the OOD battery can point this at an ood substrate/frozen/lambda dir + cohort code.
SUBROOT = Path(os.environ.get("EVOLBENCH_FOR_SUBROOT", "/data/latent-basemap/substrates/evolbench"))
LAMBDA_DIR = SB / os.environ.get("EVOLBENCH_FOR_LAMBDADIR", "lambda")
FROZEN_DIR = SB / os.environ.get("EVOLBENCH_FOR_FROZEN", "evolbench-armA-frozen")
OOD_CODE = int(os.environ.get("EVOLBENCH_FOR_OODCODE", "4"))   # injected-cohort corpus code (reddit=4)
E = "/data/embeddings"
N0 = 4_000_000; N2 = 5_600_000; N = 6_400_000
SHARD = 500_000
CORPUS_NAME = {0: "fineweb", 1: "redpajama", 2: "pile", 3: "starcoder", 4: "reddit", 5: "ca", 7: "bluesky"}
CHUNK_DIR = {  # row-aligned chunk parquets (chunk_text col); missing dirs fail-closed (no NN reconstruction)
    "reddit": f"{E.replace('/embeddings','/chunks')}/reddit-tldr17-chunked-120",
    "fineweb": "/data/chunks/fineweb-edu-sample-10BT-chunked-120",
    "redpajama": "/data/chunks/RedPajama-Data-V2-sample-10B-chunked-120",
    "pile": "/data/chunks/pile-uncopyrighted-chunked-120",
    "starcoder": "/data/chunks/starcoderdata-code-chunked-120",
    "ca": "/data/chunks/communityarchive-tweets", "bluesky": "/data/chunks/bluesky-5m-chunked-120"}


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _procrustes_fit(src, ref):
    mu_s = src.mean(0); mu_r = ref.mean(0)
    U, S, Vt = np.linalg.svd((src - mu_s).T @ (ref - mu_r)); R = U @ Vt
    sc = S.sum() / max(((src - mu_s) ** 2).sum(), 1e-9)
    return mu_s, (sc * R), mu_r


def _apply(xy, fit):
    mu_s, scR, mu_r = fit
    return (xy - mu_s) @ scR + mu_r


def _load_prov():
    return np.concatenate([np.load(SUBROOT / t / "provenance.npy", allow_pickle=False)
                           for t in ("T0", "T1", "T2", "T3")])


def _grid_blobs(pts, bins=512, dens_pct=99.0):
    """Grid-density blob detection: fine 2D histogram, threshold high-density cells, connected-component.
    Returns labels (0=diffuse, 1..=blobs ranked by size) over pts, + component stats."""
    from scipy import ndimage
    lo = pts.min(0); hi = pts.max(0); span = np.maximum(hi - lo, 1e-9)
    ix = np.clip(((pts - lo) / span * (bins - 1)).astype(np.int32), 0, bins - 1)
    H = np.zeros((bins, bins), np.int64)
    np.add.at(H, (ix[:, 0], ix[:, 1]), 1)
    thr = np.percentile(H[H > 0], dens_pct)
    mask = H >= max(thr, 2)
    lab, ncomp = ndimage.label(mask)  # 4-connectivity
    comp_of_cell = lab[ix[:, 0], ix[:, 1]]              # component id per point (0 if below-threshold cell)
    sizes = np.array([(comp_of_cell == c).sum() for c in range(1, ncomp + 1)])
    order = np.argsort(-sizes)[:2] + 1 if ncomp else np.array([], int)  # top-2 components by point count
    labels = np.zeros(len(pts), np.int8)
    for rank, c in enumerate(order, start=1):
        labels[comp_of_cell == c] = rank
    return labels, {"n_components": int(ncomp), "top_sizes": [int(sizes[c - 1]) for c in order],
                    "grid_bins": bins, "dens_pct": dens_pct}


def _bbox_area(pts):
    if len(pts) < 3:
        return 0.0
    lo = pts.min(0); hi = pts.max(0)
    return float((hi[0] - lo[0]) * (hi[1] - lo[1]))


def _emb_geom(vecs, base_sample):
    """Embedding geometry for a group: norm stats, within-cosine (sample), min cosine-dist to a base sample."""
    v = np.asarray(vecs, np.float32)
    norms = np.linalg.norm(v, axis=1)
    vn = _norm(v)
    si = np.random.default_rng(0).choice(len(vn), min(2000, len(vn)), replace=False)
    sub = vn[si]
    # within-group mean pairwise cosine (on the sample)
    G = sub @ sub.T
    iu = np.triu_indices(len(sub), 1)
    within_cos = float(G[iu].mean()) if len(iu[0]) else None
    # mean nearest-base cosine distance (true OOD-ness): for sampled group vecs, max dot over base sample
    nb = (sub @ base_sample.T).max(1)            # cosine similarity to nearest base
    nearest_base_cosdist = float((1.0 - nb).mean())
    return {"n": int(len(v)), "norm_mean": round(float(norms.mean()), 4),
            "norm_std": round(float(norms.std()), 4), "within_cosine_mean": round(within_cos, 4),
            "nearest_base_cosine_dist_mean": round(nearest_base_cosdist, 4)}


def _resolve(pr, corpus, k):
    """Resolve up to k provenance records (corpus, shard, row) to REAL text. The pipeline invariant is
    shard == sorted-chunk-file index and row == LOCAL row in that file (chunk-file-i produced embedding-
    shard-i). FAIL-CLOSED: if chunks are missing, or any (shard,row) is out of the parquet's bounds (the
    corpus's chunk files do not align with its provenance — e.g. pile), we SKIP that row and annotate the
    count; we NEVER reconstruct text by embedding-NN. Reads each needed parquet once (grouped by shard)."""
    import pyarrow.parquet as pq
    name = CORPUS_NAME.get(corpus); cdir = CHUNK_DIR.get(name)
    files = sorted(glob.glob(f"{cdir}/train/*.parquet")) if cdir else []
    if not files:
        return {"_fail_closed": f"no chunk parquets for '{name}' at {cdir} — text unavailable, NOT reconstructed"}
    counts = [pq.read_metadata(f).num_rows for f in files]
    pr = pr[:k]; cols = ["chunk_text", "subreddit", "author"] if name == "reddit" else ["chunk_text"]
    out = []; oob = 0
    shards = pr["shard"].astype(int); locrows = pr["row"].astype(int)
    for s in np.unique(shards):
        if s >= len(files):
            oob += int((shards == s).sum()); continue
        want = locrows[shards == s]
        valid = want[want < counts[s]]; oob += int((want >= counts[s]).sum())
        if len(valid) == 0:
            continue
        t = pq.read_table(files[s], columns=cols).to_pandas()
        for r in valid:
            rec = {"shard": int(s), "row": int(r), "text": str(t.iloc[int(r)]["chunk_text"])[:600]}
            if name == "reddit":
                rec["subreddit"] = str(t.iloc[int(r)]["subreddit"]); rec["author"] = str(t.iloc[int(r)]["author"])
            out.append(rec)
    res = {"corpus": name, "n_text": len(out), "rows": out}
    if oob:
        res["_fail_closed_oob"] = f"{oob} rows skipped: chunk parquets misaligned with provenance for '{name}' (no NN fallback)"
    return res


def main():
    wtag = sys.argv[1] if len(sys.argv) > 1 else "0.02"
    suffix = os.environ.get("EVOLBENCH_FOR_TAG", "")   # e.g. "-ca" so ood outputs don't collide with reddit
    cw = LAMBDA_DIR / f"coords-w{wtag}.npy"
    if not cw.is_file():
        raise SystemExit(f"λ map absent: {cw}")
    xy_w = np.asarray(np.load(cw), np.float32)
    xy_f = np.asarray(np.load(FROZEN_DIR / "coords-S3.npy"), np.float32)
    # align w -> frozen on the ANCHORED base rows [0:N2] (the pinned frame the service uses)
    fit = _procrustes_fit(xy_w[:N2].astype(np.float64), xy_f[:N2].astype(np.float64))
    xy_wa = _apply(xy_w.astype(np.float64), fit).astype(np.float32)
    disp = np.linalg.norm(xy_wa - xy_f, axis=1)                    # per-row displacement vs frozen

    prov = _load_prov()
    reddit = np.arange(N2, N)
    rpts = xy_wa[reddit]
    labels, ginfo = _grid_blobs(rpts)

    # base embedding sample (for OOD-ness), normalized
    base_vecs = _norm(np.asarray(np.load(SUBROOT / "T0" / "substrate.f32.npy", mmap_mode="r")[:200000], np.float32))
    bsi = np.random.default_rng(1).choice(len(base_vecs), 50000, replace=False)
    base_sample = base_vecs[bsi]
    T3vecs = np.asarray(np.load(SUBROOT / "T3" / "substrate.f32.npy", mmap_mode="r"), np.float32)

    out = {"schema": "evolbench-blob-forensics-2026-09-02", "w": wtag, "grid": ginfo,
           "row_layout": {"base": [0, N2], "reddit": [N2, N]}, "groups": {}, "sucked_in": {}, "spot_check": {}}

    # (a)/(b) per group: blob1, blob2, diffuse, + base baseline
    groups = {"blob1": reddit[labels == 1], "blob2": reddit[labels == 2],
              "diffuse_reddit": reddit[labels == 0]}
    base_rows = np.arange(0, N2)
    centroids = {}
    for gname, rows in groups.items():
        if len(rows) == 0:
            out["groups"][gname] = {"n": 0}; continue
        p2 = xy_wa[rows]; cen = p2.mean(0); centroids[gname] = cen
        gvecs = T3vecs[rows - N2]                                  # T3-local index
        geom = _emb_geom(gvecs, base_sample)
        out["groups"][gname] = {"n": int(len(rows)),
            "area_bbox": round(_bbox_area(p2), 3), "density_per_area": round(len(rows) / max(_bbox_area(p2), 1e-9), 3),
            "centroid_2d": [round(float(cen[0]), 3), round(float(cen[1]), 3)],
            "frozen_spread_std": round(float(xy_f[rows].std(0).mean()), 4),
            "current_spread_std": round(float(p2.std(0).mean()), 4),
            "embedding_geom": geom}
    # base baseline geometry (sample)
    bb = base_vecs[np.random.default_rng(2).choice(len(base_vecs), 3000, replace=False)]
    out["groups"]["base_baseline"] = {"n_sampled": len(bb),
        "embedding_geom": _emb_geom(bb, base_sample)}

    # (c) sucked-in base rows: large displacement AND terminating near a blob centroid
    for bname in ("blob1", "blob2"):
        if bname not in centroids:
            continue
        cen = centroids[bname]; rows = groups[bname]
        rad = np.percentile(np.linalg.norm(xy_wa[rows] - cen, axis=1), 90)   # blob characteristic radius
        end_near = np.linalg.norm(xy_wa[base_rows] - cen, axis=1) < rad      # base rows ending inside blob
        moved = disp[base_rows] > np.percentile(disp[base_rows], 99)         # top-1% displaced base rows
        sucked = base_rows[end_near & moved]
        src = {CORPUS_NAME.get(int(c), str(c)): int((prov[sucked]["corpus"] == c).sum())
               for c in np.unique(prov[sucked]["corpus"])} if len(sucked) else {}
        out["sucked_in"][bname] = {"n": int(len(sucked)), "blob_radius": round(float(rad), 3),
            "source_datasets": src,
            "mean_displacement": round(float(disp[sucked].mean()), 3) if len(sucked) else None,
            "max_displacement": round(float(disp[sucked].max()), 3) if len(sucked) else None}

    # (d) semantic spot-check: per blob, reddit member texts (most central 100) + stationary-adjacent base (50)
    for bname in ("blob1", "blob2"):
        if bname not in centroids:
            continue
        cen = centroids[bname]; rows = groups[bname]
        d2c = np.linalg.norm(xy_wa[rows] - cen, axis=1)
        central = rows[np.argsort(d2c)[:100]]                      # 100 most-central reddit members
        rtext = _resolve(prov[central], OOD_CODE, 100)
        # stationary-adjacent base rows: base rows inside the blob radius with SMALL displacement
        rad = np.percentile(d2c, 90)
        base_near = base_rows[np.linalg.norm(xy_wa[base_rows] - cen, axis=1) < rad]
        stationary = base_near[disp[base_near] < np.percentile(disp[base_rows], 50)][:50]
        # group stationary base rows by corpus, resolve text per corpus (fail-closed each; shard=file idx)
        stext = [_resolve(prov[stationary[prov[stationary]["corpus"] == c]], int(c), 25)
                 for c in np.unique(prov[stationary]["corpus"])]
        out["spot_check"][bname] = {"reddit_members": rtext, "stationary_adjacent_base": stext}

    outp = SB / f"evolbench-blob-forensics{suffix}-w{wtag}.json"
    outp.write_text(json.dumps(out, indent=1, default=str))
    # console summary
    print(f"=== BLOB FORENSICS w={wtag} ===", flush=True)
    print("grid:", ginfo, flush=True)
    for g, v in out["groups"].items():
        print(f"  {g:>16}: {v}", flush=True)
    for b, v in out["sucked_in"].items():
        print(f"  sucked_in[{b}]: {v}", flush=True)
    print(f"wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
