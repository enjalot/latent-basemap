"""O2 — backfill graph_manifest.v2 for the 2M->4M growth-step graph.

Mirrors ``experiments/backfill_graph_manifests.py``'s approach (P0-2 manifest
backfill) for a single additional asset: the 4M-row nested graph used by the
O2 sparse-anchor self-growth pilot (the ~2M old points from the 2M->4M growth
step become the sparse ``anchor_ids`` landmark pool that
``core.py:_load_sparse_anchor_landmarks`` holds fixed). This script does NOT
touch the existing 8M/200k/2m asset list in ``backfill_graph_manifests.py``.

The manifest-building logic is factored into ``build_manifest_for_asset`` so
it is a pure function of an asset dict — testable against a tiny synthetic
graph/train-dir (see ``tests/test_o2_sparse_anchors.py``) without touching the
real 4M data on ``/data``. Running ``main()`` DOES touch the real 4M graph
(3.6 GB) + train shard (6 GB) on disk — CPU only, no GPU lease required.

CPU by default (BASEMAP_PROBE_CPU=1) so it never contends with a running GPU
training job.

Usage:
  BASEMAP_PROBE_CPU=1 .venv/bin/python experiments/backfill_4m_manifest.py
"""
from __future__ import annotations
import os, sys, json, glob, subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.graph_validation import (validate_edge_bounds, graph_manifest_v2,
                                       edge_endpoint_cosine_check, write_manifest)
from basemap.panel_v2 import load_embeddings

ASSET = dict(
    name="jina-en-4M-nested",
    graph="/data/latent-basemap/jina-en-4M-nested/edges_k50_fuzzy.npz",
    train="/data/latent-basemap/jina-en-4M-nested/train",
    sample_idx="/data/latent-basemap/jina-en-4M-nested/sample_indices.npy",
    dim=768,
)


def _git():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        c = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()[:12]
        d = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip())
        return c, d
    except Exception:
        return None, None


def build_manifest_for_asset(asset: dict, *, n_probe: int = 5000, min_margin: float = 0.1,
                             cosine_probe: bool = True) -> dict:
    """Build + write a ``graph_manifest.v2`` for one ``(graph, train-dir)`` asset.

    A pure function of ``asset`` (keys: ``name``, ``graph``, ``train``,
    ``sample_idx``, ``dim``) so it is exercisable against synthetic tiny data
    in a unit test, not just the real 4M asset. Writes ``<graph>.manifest.json``
    (graph_sha + ordered ``data_shards`` + ``data_shard_sha``, per S0/P0-2) and
    returns the manifest dict. ``cosine_probe=False`` skips the endpoint-cosine
    sanity check (useful for synthetic graphs whose edges are not real kNN).
    """
    z = np.load(asset["graph"])
    s, t = z["sources"], z["targets"]
    n_nodes = int(z["n_nodes"]); k = int(z["k"]) if "k" in z else None
    validate_edge_bounds(s, t, n_nodes)                       # endpoint bounds
    shards = sorted(glob.glob(os.path.join(asset["train"], "*.npy")))
    if not shards:
        raise ValueError(f"no .npy shards found under {asset['train']}")
    X = load_embeddings(asset["train"], dim=asset.get("dim"))
    probe = None
    if cosine_probe:
        # small CPU endpoint-cosine probe (edges connect near-neighbours)
        probe = edge_endpoint_cosine_check(s, t, X, n_probe=n_probe, seed=0, min_margin=min_margin)
    commit, dirty = _git()
    sample_idx = asset.get("sample_idx")
    man = graph_manifest_v2(
        s, t, n_nodes, X=X, graph_path=asset["graph"], data_paths=shards,
        sample_indices_path=sample_idx, k=k, metric="cosine", directed=True,
        weight_semantics="fuzzy_simplicial_set(k50)", builder_commit="pre-manifest(backfill_4m)",
        cosine_probe=probe, extra={"verified_by": "backfill_4m_manifest.py",
                                   "verifier_commit": commit, "verifier_dirty": dirty,
                                   "post_hoc_identity_verified": True,
                                   "o2_growth_step": "2m->4m"})
    mpath = asset["graph"] + ".manifest.json"
    write_manifest(mpath, man)
    return man


def main():
    if os.environ.get("BASEMAP_PROBE_CPU") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    man = build_manifest_for_asset(ASSET)
    rec = {"name": ASSET["name"], "graph": ASSET["graph"], "n_nodes": man["n_nodes"],
           "n_edges": man["n_edges"], "endpoint_cosine": man.get("endpoint_cosine"),
           "manifest": ASSET["graph"] + ".manifest.json", "graph_sha": man["graph_sha"],
           "data_fingerprint": man.get("data_fingerprint")}
    out = os.path.join(os.path.dirname(__file__), "evidence", "o2_4m_graph_verification.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"assets": [rec]}, open(out, "w"), indent=1)
    print(f"[backfill_4m] {ASSET['name']}: edges={man['n_edges']:,} cos={man.get('endpoint_cosine')} "
          f"-> {ASSET['graph']}.manifest.json", flush=True)
    print(f"[backfill_4m] report -> {out}", flush=True)


if __name__ == "__main__":
    main()
