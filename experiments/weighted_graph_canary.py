#!/usr/bin/env python3
"""No-update production-path canary for a content-bound 30M weighted graph.

This intentionally calls the trainer's real admission and sampler-selection
path. It uploads the accepted 30M feature pack, constructs the host-streamed
weighted sampler, pulls one production batch, and exits without allocating a
model or taking an optimizer step.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import sha256_file
from basemap.output_safety import atomic_write_new_json, refuse_existing


def run_canary(args) -> dict:
    refuse_existing(args.json_out, label="weighted graph canary receipt")
    manifest_path = args.artifact + ".manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != args.expected_manifest_sha256:
        raise RuntimeError(
            f"manifest SHA-256 {manifest_sha256} != expected "
            f"{args.expected_manifest_sha256}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if (manifest.get("graph_sha256") != args.expected_graph_sha256
            or manifest.get("production_trainer_ready") is not True
            or manifest.get("builder_dirty") is not False):
        raise RuntimeError(
            "weighted graph manifest is not a clean, production-ready binding")

    import torch
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    from basemap.round0014_program import Round0014MaterializedArray

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("weighted graph canary requires exactly one visible CUDA GPU")
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    X = Round0014MaterializedArray()
    trainer = ParametricUMAP(
        a=1.0, b=1.0, architecture="residual_bottleneck",
        hidden_dim=2048, n_layers=3, n_components=2,
        correlation_weight=0.0, batch_size=args.batch_size,
        pos_ratio=args.pos_ratio, device="cuda", use_amp=True,
        positive_target_mode="binary", weighted_edge_sampling=True,
        gpu_resident_data="auto", gpu_resident_vram_budget_gb=31.0,
        required_input_pipeline="hybrid", require_graph_manifest=True,
        require_full_budget=False,
    )
    dataset = loader = batch = None
    try:
        dataset, loader, effective_edges = trainer._prepare_edge_list_training(
            X, args.artifact, len(X), low_memory=True,
            random_state=args.seed)
        pipeline = dict(trainer._pipeline_info)
        expected = {
            "pipeline": "hybrid",
            "sampler_class": "HostStreamEdgeSampler",
            "positive_sampling": "weighted_with_replacement",
            "x_residency": "device_fp16",
            "weighted_requested": True,
            "weighted_effective": True,
            "positive_with_replacement": True,
        }
        mismatches = {key: {"expected": value, "observed": pipeline.get(key)}
                      for key, value in expected.items()
                      if pipeline.get(key) != value}
        if mismatches:
            raise RuntimeError(f"production pipeline stamp mismatch: {mismatches}")
        trusted_graph_sha256 = trainer._pipeline_verified_hashes.get("graph_sha256")
        if trusted_graph_sha256 != args.expected_graph_sha256:
            raise RuntimeError(
                f"admitted graph SHA-256 {trusted_graph_sha256} != expected "
                f"{args.expected_graph_sha256}")
        batch = next(iter(loader))
        torch.cuda.synchronize()
        src, dst, labels = batch
        if (tuple(src.shape) != tuple(dst.shape)
                or src.shape != (args.batch_size, X.shape[1])
                or labels.shape != (args.batch_size,)
                or src.dtype != torch.float32 or dst.dtype != torch.float32
                or labels.dtype != torch.float32
                or not bool(torch.isfinite(src).all())
                or not bool(torch.isfinite(dst).all())
                or not bool(torch.isfinite(labels).all())):
            raise RuntimeError("production batch shape/dtype/finite contract failed")
        positive_labels = int(torch.count_nonzero(labels == 1.0).item())
        negative_labels = int(torch.count_nonzero(labels == 0.0).item())
        expected_positive = max(1, int(args.batch_size * args.pos_ratio))
        if (positive_labels != expected_positive
                or positive_labels + negative_labels != args.batch_size):
            raise RuntimeError(
                f"batch label counts {positive_labels}/{negative_labels} do not match "
                f"expected {expected_positive}/{args.batch_size - expected_positive}")
        receipt = {
            "schema": "weighted-graph-production-canary-v1",
            "training_performed": False,
            "optimizer_updates": 0,
            "artifact": os.path.realpath(args.artifact),
            "graph_sha256": trusted_graph_sha256,
            "manifest_sha256": manifest_sha256,
            "build_contract_sha256": manifest["build_contract_sha256"],
            "pipeline": pipeline,
            "verified_hashes": trainer._pipeline_verified_hashes,
            "effective_edges": int(effective_edges),
            "batch": {
                "rows": int(args.batch_size),
                "features": int(X.shape[1]),
                "feature_dtype": "float32",
                "positive_labels": positive_labels,
                "negative_labels": negative_labels,
                "finite": True,
            },
            "wall_seconds": time.monotonic() - start,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                  * 1024),
            "cuda_device_name": torch.cuda.get_device_name(0),
        }
        atomic_write_new_json(args.json_out, receipt)
        return receipt
    finally:
        if loader is not None and hasattr(loader, "close"):
            loader.close()
        del batch, loader, dataset, trainer, X
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--expected-graph-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--pos-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    receipt = run_canary(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
