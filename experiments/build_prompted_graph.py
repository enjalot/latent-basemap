"""O1 — build the prompted-200k analog of the unprompted testbed's training
graph: exact GPU k=50 kNN -> UMAP fuzzy simplicial set (mirrors
``edges_k50_fuzzy.npz`` at /data/latent-basemap/jina-en-200k, reconstructed
from its content: sources/targets int32, weights float32, scalars n_nodes/k,
first-neighbor weight 1.0, mean out-degree ~1.4x k after set-union
symmetrization — the textbook signature of ``umap.umap_.fuzzy_simplicial_set``
with ``set_op_mix_ratio=1.0`` applied to a precomputed k=50 cosine kNN). The
ORIGINAL builder script for that graph is not present in this checkout (only
its manifest survived, written post-hoc by ``backfill_graph_manifests.py``),
so this script re-derives the construction method from the artifact's content
rather than from source — flagged here for the record.

Also computes:
  - frozen k=256 / k=1024 centroids (reusing
    ``experiments/score_complete_panel.py::frozen_centroids`` verbatim, same
    GPU k-means recipe used for the unprompted testbed's purity centroids);
  - held-out query embeddings: N text rows drawn from the FULL source corpus
    that are provably OUTSIDE ``sample_indices`` (never trained on), embedded
    through the SAME prompted pipeline as the training rows. NOTE: the
    unprompted testbed's own held-out query construction script is likewise
    not present in this checkout (only its eval outputs, e.g.
    jina-en-200k/eval/projection_fidelity_mn4s42.json, survived); this
    "reserve N ids disjoint from sample_indices, same corpus, same
    pipeline" construction is a documented, defensible analog — not a
    byte-for-byte reproduction of whatever produced the original eval set;
  - a prompted-vs-unprompted PRE-TRAINING shift report: high-D kNN neighbour
    overlap (Jaccard@k over anchor rows, same row ids in both spaces) and
    centroid-cluster agreement (ARI of nearest-centroid assignment between
    the two spaces), so the orchestrator can see how much the prompt moved
    the space before spending GPU hours training either map.

Usage (orchestrator, under a held/inherited GPU lease, AFTER
experiments/embed_prompted_200k.py has produced the prompted 200k):

  python experiments/build_prompted_graph.py \
      --prompted /data/latent-basemap/jina-en-200k-prompted \
      --unprompted /data/latent-basemap/jina-en-200k \
      --text-dir /data/chunks/fineweb-edu-sample-10BT-chunked-500/train \
      --embed-dir /data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train \
      --k 50 --n-holdout 5000 --n-anchors 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.graph_validation import graph_manifest_v2, edge_endpoint_cosine_check, write_manifest
from experiments.embed_prompted_200k import (
    apply_prompt, assert_row_identity, load_model, embed_texts, sha_file,
    verify_shard_alignment, fetch_texts_for_indices, build_shard_offsets,
    PROMPT_PREFIX, MODEL_ID,
)
from experiments.score_complete_panel import frozen_centroids


# --------------------------------------------------------------------------
# GPU kNN (CPU-testable in shape/signature only; actual execution needs torch)
# --------------------------------------------------------------------------

def topk_neighbors(Xq, Xc, k, device="cuda", chunk=4096, exclude_self_ids=None):
    """Exact top-k cosine neighbours of each row of ``Xq`` within ``Xc``
    (both assumed L2-normalized; cosine == dot product). Returns
    (nbr_idx int32 (nq,k), nbr_dist float32 (nq,k) = 1 - cosine).

    ``exclude_self_ids``, if given, is a length-``nq`` array of row ids in
    ``Xc`` to drop from each row's own neighbour list (used when Xq is a
    subset of Xc, e.g. the training-graph case Xq is Xc). When Xq and Xc are
    disjoint corpora (held-out queries), pass ``None``.
    """
    import torch
    nq = Xq.shape[0]
    kmax = k + 1 if exclude_self_ids is not None else k
    Xqt = torch.from_numpy(np.ascontiguousarray(Xq)).to(device).float()
    Xct = torch.from_numpy(np.ascontiguousarray(Xc)).to(device).float()
    nbr_idx = np.empty((nq, k), dtype=np.int32)
    nbr_sim = np.empty((nq, k), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, nq, chunk):
            sims = Xqt[i:i + chunk] @ Xct.T
            vals, idx = torch.topk(sims, kmax, dim=1)
            idx_np = idx.cpu().numpy()
            val_np = vals.cpu().numpy()
            if exclude_self_ids is not None:
                self_ids = exclude_self_ids[i:i + idx_np.shape[0]]
                for r in range(idx_np.shape[0]):
                    keep = idx_np[r] != self_ids[r]
                    row_idx = idx_np[r][keep][:k]
                    row_val = val_np[r][keep][:k]
                    nbr_idx[i + r] = row_idx
                    nbr_sim[i + r] = row_val
            else:
                nbr_idx[i:i + idx_np.shape[0]] = idx_np[:, :k]
                nbr_sim[i:i + idx_np.shape[0]] = val_np[:, :k]
    nbr_dist = (1.0 - nbr_sim).astype(np.float32)
    return nbr_idx, nbr_dist


def build_fuzzy_graph(X, k=50, seed=42, device="cuda", chunk=4096):
    """Exact k=50 cosine kNN -> umap.umap_.fuzzy_simplicial_set (set-union
    symmetrization, set_op_mix_ratio=1.0 default) -> COO edge list. Mirrors
    the construction the existing edges_k50_fuzzy.npz manifests attest to."""
    import umap.umap_ as uu
    n = X.shape[0]
    ids = np.arange(n, dtype=np.int64)
    nbr_idx, nbr_dist = topk_neighbors(X, X, k, device=device, chunk=chunk, exclude_self_ids=ids)
    graph, sigmas, rhos = uu.fuzzy_simplicial_set(
        X, n_neighbors=k, random_state=np.random.RandomState(seed), metric="cosine",
        knn_indices=nbr_idx, knn_dists=nbr_dist,
    )
    coo = graph.tocoo()
    sources = coo.row.astype(np.int32)
    targets = coo.col.astype(np.int32)
    weights = coo.data.astype(np.float32)
    return sources, targets, weights, {"n_neighbors": k, "n_nodes": n,
                                       "n_edges": int(len(sources)),
                                       "mean_out_degree": float(len(sources) / n)}


# --------------------------------------------------------------------------
# prompted-vs-unprompted shift report
# --------------------------------------------------------------------------

def neighbor_overlap_report(X_a, X_b, anchor_pos, k=50, device="cuda", chunk=4096):
    """Jaccard@k neighbour overlap between two spaces (X_a, X_b MUST be the
    SAME row-id space — row i in X_a and row i in X_b are the same source
    text). Returns per-anchor overlap fraction + summary stats."""
    Xq_a = X_a[anchor_pos]
    Xq_b = X_b[anchor_pos]
    ids = anchor_pos.astype(np.int64)
    nbr_a, _ = topk_neighbors(Xq_a, X_a, k, device=device, chunk=chunk, exclude_self_ids=ids)
    nbr_b, _ = topk_neighbors(Xq_b, X_b, k, device=device, chunk=chunk, exclude_self_ids=ids)
    overlaps = np.empty(len(anchor_pos), dtype=np.float32)
    for i in range(len(anchor_pos)):
        sa, sb = set(nbr_a[i].tolist()), set(nbr_b[i].tolist())
        overlaps[i] = len(sa & sb) / float(k)
    return {
        "k": int(k), "n_anchors": int(len(anchor_pos)),
        "mean_overlap": float(overlaps.mean()), "std_overlap": float(overlaps.std()),
        "min_overlap": float(overlaps.min()), "max_overlap": float(overlaps.max()),
    }


def centroid_agreement_report(X_a, X_b, cents_a, cents_b, sample_pos, device="cuda"):
    """Nearest-centroid cluster assignment agreement (Adjusted Rand Index)
    between two spaces sharing the same row-id sample. A low ARI means the
    prompt substantially reshuffled which rows co-cluster, independent of
    any specific centroid identity (two independent k-means runs have no
    inherent centroid-to-centroid correspondence, so ARI over the induced
    partitions is the meaningful cross-space comparison, not centroid
    distance)."""
    import torch
    from sklearn.metrics import adjusted_rand_score
    dev = device
    def assign(X, C):
        Xt = torch.from_numpy(np.ascontiguousarray(X[sample_pos])).to(dev).float()
        Ct = torch.from_numpy(np.ascontiguousarray(C)).to(dev).float()
        # cosine-nearest centroid (both assumed normalized-ish; use cdist for safety)
        lab = torch.cdist(Xt, Ct).argmin(1).cpu().numpy()
        return lab
    out = {}
    for kname, (ca, cb) in {"k256": (cents_a.get(256), cents_b.get(256)),
                            "k1024": (cents_a.get(1024), cents_b.get(1024))}.items():
        if ca is None or cb is None:
            continue
        lab_a = assign(X_a, ca)
        lab_b = assign(X_b, cb)
        out[kname] = {"ari": float(adjusted_rand_score(lab_a, lab_b)),
                     "n_clusters": int(ca.shape[0])}
    out["n_sample"] = int(len(sample_pos))
    return out


# --------------------------------------------------------------------------
# held-out queries
# --------------------------------------------------------------------------

def select_holdout_ids(total_n, sample_indices, n_holdout, seed=123):
    """Deterministically pick ``n_holdout`` global row ids from
    ``[0, total_n)`` that are PROVABLY disjoint from ``sample_indices`` (the
    200k training rows) — never trained on, in either the prompted or
    unprompted map."""
    in_train = np.zeros(total_n, dtype=bool)
    in_train[sample_indices] = True
    available = np.nonzero(~in_train)[0]
    rng = np.random.RandomState(seed)
    n_holdout = min(n_holdout, len(available))
    chosen = np.sort(rng.choice(available, size=n_holdout, replace=False))
    assert not in_train[chosen].any(), "holdout selection leaked a training row (bug)"
    return chosen


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompted", default="/data/latent-basemap/jina-en-200k-prompted")
    ap.add_argument("--unprompted", default="/data/latent-basemap/jina-en-200k")
    ap.add_argument("--text-dir", default="/data/chunks/fineweb-edu-sample-10BT-chunked-500/train")
    ap.add_argument("--embed-dir",
                     default="/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-holdout", type=int, default=5000)
    ap.add_argument("--n-anchors", type=int, default=2000)
    ap.add_argument("--n-cluster-sample", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        from basemap.run_controller import require_active_lease
        require_active_lease()

    # ---- load + cross-check the two testbeds share the SAME row-id space ----
    Xp = np.load(os.path.join(args.prompted, "train", "data-00000.npy"), mmap_mode="r")
    Xu = np.load(os.path.join(args.unprompted, "train", "data-00000.npy"), mmap_mode="r")
    sid_p = np.load(os.path.join(args.prompted, "sample_indices.npy"))
    sid_u = np.load(os.path.join(args.unprompted, "sample_indices.npy"))
    assert_row_identity(sid_p, sid_u,
                        context="prompted vs unprompted sample_indices (must be identical "
                                "and in the same order for row-to-row comparison)")
    if Xp.shape != Xu.shape:
        raise ValueError(f"prompted X {Xp.shape} != unprompted X {Xu.shape} — not comparable.")
    n = Xp.shape[0]
    print(f"[build_prompted_graph] {n:,} rows, dim={Xp.shape[1]}, row-id spaces verified identical",
          flush=True)
    Xp = np.asarray(Xp, dtype=np.float32)
    Xu = np.asarray(Xu, dtype=np.float32)

    # ---- 1. fuzzy k=50 graph over the PROMPTED embeddings ----
    t0 = time.time()
    sources, targets, weights, ginfo = build_fuzzy_graph(Xp, k=args.k, seed=args.seed, device=device)
    print(f"[build_prompted_graph] fuzzy graph: {ginfo} in {time.time() - t0:.1f}s", flush=True)
    graph_path = os.path.join(args.prompted, f"edges_k{args.k}_fuzzy.npz")
    np.savez(graph_path, sources=sources, targets=targets, weights=weights,
            n_nodes=n, k=args.k)
    probe = edge_endpoint_cosine_check(sources, targets, Xp, n_probe=20000, seed=0, min_margin=0.15)
    print(f"[build_prompted_graph] endpoint-cosine probe: {probe}", flush=True)
    man = graph_manifest_v2(
        sources, targets, n, X=Xp, graph_path=graph_path,
        data_paths=[os.path.join(args.prompted, "train", "data-00000.npy")],
        sample_indices_path=os.path.join(args.prompted, "sample_indices.npy"),
        k=args.k, metric="cosine", directed=True,
        weight_semantics=f"fuzzy_simplicial_set(k{args.k})",
        builder_commit=_git_commit(), cosine_probe=probe,
        extra={"builder": "build_prompted_graph.py", "prompt_prefix": PROMPT_PREFIX,
              "mean_out_degree": ginfo["mean_out_degree"]},
    )
    write_manifest(graph_path + ".manifest.json", man)
    print(f"[build_prompted_graph] wrote {graph_path} + manifest", flush=True)

    # ---- 2. frozen centroids k256/k1024 over the PROMPTED embeddings ----
    cents_p = frozen_centroids(Xp, (256, 1024), args.prompted, seed=0)
    print(f"[build_prompted_graph] prompted centroids: "
          f"{ {k: v.shape for k, v in cents_p.items()} }", flush=True)

    # ---- 3. held-out queries (provably outside sample_indices), PROMPTED ----
    text_shards, embed_sizes, offsets, dim = verify_shard_alignment(args.embed_dir, args.text_dir)
    holdout_ids = select_holdout_ids(int(offsets[-1]), sid_p, args.n_holdout, seed=123)
    print(f"[build_prompted_graph] {len(holdout_ids):,} held-out query ids "
          f"(disjoint from the {n:,} training rows)", flush=True)
    ho_texts = fetch_texts_for_indices(holdout_ids, args.text_dir, text_shards, offsets)
    ho_prompted = apply_prompt(ho_texts, PROMPT_PREFIX)
    model, commit = load_model(device=device, dtype=args.dtype)
    ho_emb = embed_texts(model, ho_prompted, batch_size=args.batch_size, show_progress=True)
    np.save(os.path.join(args.prompted, "holdout_query_embeddings.npy"), ho_emb)
    np.save(os.path.join(args.prompted, "holdout_query_ids.npy"), holdout_ids)
    ho_manifest = {
        "schema": "prompted_holdout_manifest.v1", "n_holdout": int(len(holdout_ids)),
        "model_id": MODEL_ID, "model_commit": commit, "prompt_prefix": PROMPT_PREFIX,
        "disjoint_from": os.path.abspath(os.path.join(args.prompted, "sample_indices.npy")),
        "note": "IDs drawn from the same source corpus as the 200k training rows but "
               "excluded from sample_indices; NOT a reproduction of the unprompted "
               "testbed's original held-out eval sets (that builder script is not "
               "present in this checkout) — a documented analog construction.",
    }
    json.dump(ho_manifest, open(os.path.join(args.prompted, "holdout_query_embeddings.npy.manifest.json"), "w"),
              indent=1)
    print(f"[build_prompted_graph] wrote holdout_query_embeddings.npy "
          f"({ho_emb.shape}) + ids + manifest", flush=True)

    # ---- 4. prompted-vs-unprompted pre-training shift report ----
    rng = np.random.RandomState(7)
    anchor_pos = np.sort(rng.choice(n, min(args.n_anchors, n), replace=False))
    nbr_report = neighbor_overlap_report(Xu, Xp, anchor_pos, k=args.k, device=device)
    print(f"[build_prompted_graph] kNN overlap (unprompted vs prompted, k={args.k}): "
          f"mean={nbr_report['mean_overlap']:.4f} std={nbr_report['std_overlap']:.4f}", flush=True)

    cents_u_path_ok = all(os.path.exists(os.path.join(args.unprompted, f"centroids_k{k}.npy"))
                          for k in (256, 1024))
    shift_report = {"neighbor_overlap": nbr_report}
    if cents_u_path_ok:
        cents_u = {256: np.load(os.path.join(args.unprompted, "centroids_k256.npy")),
                  1024: np.load(os.path.join(args.unprompted, "centroids_k1024.npy"))}
        cl_sample = np.sort(rng.choice(n, min(args.n_cluster_sample, n), replace=False))
        cl_report = centroid_agreement_report(Xu, Xp, cents_u, cents_p, cl_sample, device=device)
        print(f"[build_prompted_graph] centroid-cluster ARI: "
              f"{ {k: v for k, v in cl_report.items() if k != 'n_sample'} }", flush=True)
        shift_report["centroid_agreement"] = cl_report
    else:
        print("[build_prompted_graph] unprompted centroids not found — skipping "
              "centroid_agreement (neighbor_overlap still computed)", flush=True)
    shift_report["seed"] = 7
    shift_report["created_utc"] = _now()
    out_path = os.path.join(args.prompted, "prompt_shift_report.json")
    json.dump(shift_report, open(out_path, "w"), indent=1)
    print(f"[build_prompted_graph] wrote {out_path}", flush=True)


def _git_commit():
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                       text=True).strip()[:12]
    except Exception:
        return None


def _now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
