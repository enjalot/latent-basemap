"""S2.5 — build the shared, content-addressed high-D reference ONCE for a corpus.

Downstream scoring of N maps then reuses this single verified reference (the
scorer recomputes the content key and fails closed on any drift). Runs on GPU
under a held/inherited lease.

Usage (under a held lease / controller):
  python experiments/build_reference.py --train /data/latent-basemap/jina-en-8M-nested/train \
      --labels /data/latent-basemap/jina-en-8m/corpus_labels.npy \
      --c256 .../audit_centroids_k256_s0.npy --c1024 .../centroids_fineweb_k1024.npy \
      --out /data/latent-basemap/closure/hiD_reference_8m.npz
"""
from __future__ import annotations
import argparse, os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.panel_v2 import (PanelV2Config, load_embeddings, sample_anchors,
                              build_hiD_reference, save_hiD_reference, FORMULA_VERSION)
from basemap.round0005_retirement import refuse_retired_launcher


def main():
    refuse_retired_launcher("experiments/build_reference.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/data/latent-basemap/jina-en-8M-nested/train")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--c256", default="/data/latent-basemap/track1/audit_centroids_k256_s0.npy")
    ap.add_argument("--c1024", default="/data/latent-basemap/track1/centroids_fineweb_k1024.npy")
    ap.add_argument("--n-anchors", type=int, default=2000)
    ap.add_argument("--out", default="/data/latent-basemap/closure/hiD_reference_8m.npz")
    args = ap.parse_args()
    # S1: GPU work runs under a held/inherited lease.
    import torch as _t
    if _t.cuda.is_available():
        from basemap.run_controller import require_active_lease
        require_active_lease()
    X = load_embeddings(args.train, dim=args.dim); n = len(X)
    cents = {256: np.load(args.c256), 1024: np.load(args.c1024)}
    cfg = PanelV2Config(n_anchors=args.n_anchors, corpus_chunk=500_000, k_clust=(256, 1024))
    aidx = sample_anchors(n, cfg)
    t0 = time.time()
    ref = build_hiD_reference(X, aidx, cfg, cents)
    path = save_hiD_reference(ref, args.out)
    meta = {"gate": "hiD_reference", "n": int(n), "key": ref["key"], "kf": ref["kf"],
            "formula_version": FORMULA_VERSION, "n_anchors": args.n_anchors,
            "build_wall_s": round(time.time() - t0, 1), "path": path,
            "key_parts": ref["key_parts"]}
    json.dump(meta, open(os.path.splitext(args.out)[0] + ".meta.json", "w"), indent=1)
    print(f"[reference] key={ref['key']} kf={ref['kf']} built in {meta['build_wall_s']}s -> {path}",
          flush=True)


if __name__ == "__main__":
    main()
