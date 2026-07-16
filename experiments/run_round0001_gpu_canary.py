"""Short, no-training CUDA canary for Round 0001 scorer admission."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.artifact_identity import git_checkout_state, path_signature
from basemap.panel_v2 import (PanelV2Config, build_hiD_reference,
                              hiD_reference_key, sample_anchors, score_panel)
from basemap.run_controller import require_active_lease


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args(argv)
    out = os.path.realpath(args.out)
    if not out.startswith("/data/"):
        raise ValueError("canary output must be under /data")

    require_active_lease()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Round 0001 GPU canary requires visible CUDA")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    rng = np.random.RandomState(args.seed)
    X = rng.standard_normal((args.rows, args.dimensions)).astype(np.float32)
    Z1 = rng.standard_normal((args.rows, 2)).astype(np.float32)
    Z2 = (Z1 @ np.array([[0.8, -0.2], [0.2, 1.1]], dtype=np.float32) +
          rng.standard_normal(Z1.shape).astype(np.float32) * 0.01)
    centroids = {16: rng.standard_normal((16, args.dimensions)).astype(np.float32)}
    cfg = PanelV2Config(frac=0.01, n_anchors=128, anchor_seed=args.seed,
                        corpus_chunk=1024, block_elems=2_000_000)
    anchors = sample_anchors(len(X), cfg)
    reference = build_hiD_reference(X, anchors, cfg, centroids)
    scores = [score_panel(X, coords, config=cfg, centroids_by_k=centroids,
                          hiD_reference=reference,
                          provenance={"round": "0001", "map": label})
              for label, coords in (("independent", Z1), ("affine", Z2))]

    unsampled = next(index for index in range(len(X)) if index not in set(anchors.tolist()))
    changed = X.copy()
    changed[unsampled, -1] += np.float32(1e-7)
    changed_key, _ = hiD_reference_key(changed, anchors, cfg, centroids,
                                       kf=reference["kf"])
    torch.cuda.synchronize()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report = {
        "schema": "round0001_gpu_scorer_canary.v1",
        "passed": (all(score["provenance"]["device"] == "cuda" and
                       score["provenance"]["hiD_reference_reused"]
                       for score in scores) and changed_key != reference["key"]),
        "training_performed": False,
        "seed": args.seed,
        "rows": args.rows,
        "dimensions": args.dimensions,
        "config": cfg.__dict__,
        "reference_key": reference["key"],
        "unsampled_mutation": {"row": unsampled, "changed_key": changed_key,
                               "detected": changed_key != reference["key"]},
        "scores": scores,
        "cuda_device": torch.cuda.get_device_name(0),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4),
        "wall_s": round(time.time() - started, 3),
        "checkout": git_checkout_state(root),
        "canary_source": path_signature(__file__),
        "scorer_source": path_signature(os.path.join(root, "basemap", "panel_v2.py")),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"passed": report["passed"], "wall_s": report["wall_s"],
                      "peak_gpu_gb": report["peak_gpu_gb"]}, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
