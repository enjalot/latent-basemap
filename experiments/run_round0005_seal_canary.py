"""Final short no-training CUDA seal canary for Round 0005."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import git_checkout_state, path_signature
from basemap.output_safety import atomic_write_new_json, refuse_existing
from basemap.panel_v2 import (PanelV2Config, build_hiD_reference,
                              hiD_reference_key, sample_anchors, score_panel)
from basemap.run_controller import (require_active_lease,
                                    require_round0005_child_admission)
from experiments.compare_panel_cache import load_fixture, run_actual_scorer


def main(argv=None) -> int:
    require_round0005_child_admission("experiments/run_round0005_seal_canary.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--scorer-fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args(argv)
    if not os.path.realpath(args.out).startswith("/data/"):
        raise ValueError("seal output must live under /data")
    refuse_existing(args.out, label="Round 0005 seal canary output")
    fixture = json.load(open(args.fixture, encoding="utf-8"))
    if fixture.get("passed") is not True:
        raise RuntimeError("Round 0005 expected-signature/all-node fixture is not passing")
    require_active_lease()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Round 0005 seal canary requires CUDA")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.time()
    rng = np.random.RandomState(args.seed)
    X = rng.standard_normal((args.rows, args.dimensions)).astype(np.float32)
    Z = rng.standard_normal((args.rows, 2)).astype(np.float32)
    C = {16: rng.standard_normal((16, args.dimensions)).astype(np.float32)}
    cfg = PanelV2Config(frac=0.01, n_anchors=128, anchor_seed=args.seed,
                        corpus_chunk=1024, block_elems=2_000_000)
    anchors = sample_anchors(len(X), cfg)
    reference = build_hiD_reference(X, anchors, cfg, C)
    score = score_panel(X, Z, config=cfg, centroids_by_k=C,
                        hiD_reference=reference,
                        provenance={"round": "0005", "node": "seal-canary"})
    unsampled = next(index for index in range(len(X))
                     if index not in set(anchors.tolist()))
    changed = X.copy(); changed[unsampled, -1] += np.float32(1e-7)
    changed_key, _ = hiD_reference_key(changed, anchors, cfg, C, kf=reference["kf"])
    scorer_fixture = load_fixture(args.scorer_fixture)
    independent_query_score = run_actual_scorer(
        scorer_fixture, cache_enabled=False, cache_dir=None,
        label="independent-no-training-seal")
    torch.cuda.synchronize()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    passed = (score["provenance"]["device"] == "cuda" and
              score["provenance"]["hiD_reference_reused"] is True and
              changed_key != reference["key"] and
              independent_query_score["query_truth_cache"]["build_count"] == 1)
    report = {
        "schema": "round0005_no_training_seal_canary.v1",
        "passed": bool(passed),
        "training_performed": False,
        "fixture": path_signature(args.fixture),
        "scorer_fixture": path_signature(args.scorer_fixture),
        "post_gate_reports_consumed": False,
        "config": cfg.__dict__,
        "reference_key": reference["key"],
        "unsampled_mutation": {"row": unsampled, "changed_key": changed_key,
                               "detected": changed_key != reference["key"]},
        "score": score,
        "independent_query_score": independent_query_score,
        "cuda_device": torch.cuda.get_device_name(0),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4),
        "wall_s": round(time.time() - started, 3),
        "checkout": git_checkout_state(root),
        "source": path_signature(__file__),
    }
    atomic_write_new_json(args.out, report, immutable=True)
    print(json.dumps({"passed": report["passed"], "wall_s": report["wall_s"],
                      "peak_gpu_gb": report["peak_gpu_gb"]}, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
