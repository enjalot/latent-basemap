"""P0-2 (review) — post-hoc identity verification + v2 manifest backfill for the
R1 fuzzy graphs that were built before manifests were mandatory.

For each (graph, testbed) it: hashes the graph/data/sample-index files, validates
every endpoint is in [0, n_nodes), runs an endpoint-cosine probe (edges must
connect near-neighbours), and writes `graph_manifest.v2` beside the graph plus a
verification report. If the checks pass and assets are unchanged, the existing
R1 evidence can be annotated 'post-hoc identity verified' WITHOUT retraining.

CPU by default (BASEMAP_PROBE_CPU=1) so it never contends with a running GPU job.

Usage:
  BASEMAP_PROBE_CPU=1 python experiments/backfill_graph_manifests.py
"""
from __future__ import annotations
import os, sys, json, subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.graph_validation import (validate_edge_bounds, graph_manifest_v2,
                                       edge_endpoint_cosine_check, write_manifest)
from basemap.panel_v2 import load_embeddings

ASSETS = [
    dict(name="jina-en-200k",
         graph="/data/latent-basemap/jina-en-200k/edges_k50_fuzzy.npz",
         train="/data/latent-basemap/jina-en-200k/train",
         sample_idx="/data/latent-basemap/jina-en-200k/sample_indices.npy", dim=768),
    dict(name="jina-en-2m",
         graph="/data/latent-basemap/jina-en-2m/edges_k50_fuzzy.npz",
         train="/data/latent-basemap/jina-en-2m/train",
         sample_idx="/data/latent-basemap/jina-en-2m/sample_indices.npy", dim=768),
]


def _git():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        c = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()[:12]
        d = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip())
        return c, d
    except Exception:
        return None, None


def main():
    import glob
    if os.environ.get("BASEMAP_PROBE_CPU") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = {"assets": []}
    commit, dirty = _git()
    for a in ASSETS:
        z = np.load(a["graph"])
        s, t = z["sources"], z["targets"]
        n_nodes = int(z["n_nodes"]); k = int(z["k"]) if "k" in z else None
        validate_edge_bounds(s, t, n_nodes)                       # endpoint bounds
        shards = sorted(glob.glob(os.path.join(a["train"], "*.npy")))
        X = load_embeddings(a["train"], dim=a["dim"])
        # small CPU endpoint-cosine probe (edges connect near-neighbours)
        probe = edge_endpoint_cosine_check(s, t, X, n_probe=5000, seed=0, min_margin=0.1)
        man = graph_manifest_v2(
            s, t, n_nodes, X=X, graph_path=a["graph"], data_paths=shards,
            sample_indices_path=a["sample_idx"], k=k, metric="cosine", directed=True,
            weight_semantics="fuzzy_simplicial_set(k50)", builder_commit="pre-manifest(backfill)",
            cosine_probe=probe, extra={"verified_by": "backfill_graph_manifests.py",
                                       "verifier_commit": commit, "verifier_dirty": dirty,
                                       "post_hoc_identity_verified": True})
        mpath = a["graph"] + ".manifest.json"
        write_manifest(mpath, man)
        rec = {"name": a["name"], "graph": a["graph"], "n_nodes": n_nodes, "n_edges": int(len(s)),
               "endpoint_cosine": probe, "manifest": mpath, "graph_sha": man["graph_sha"],
               "data_fingerprint": man.get("data_fingerprint")}
        report["assets"].append(rec)
        print(f"[backfill] {a['name']}: edges={len(s):,} cos={probe} -> {mpath}", flush=True)
    out = os.path.join(os.path.dirname(__file__), "evidence", "r1_graph_verification.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(f"[backfill] report -> {out}", flush=True)


if __name__ == "__main__":
    main()
