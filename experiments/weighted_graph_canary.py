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
    duplicate_cap = getattr(args, "duplicate_cap", None)
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
    if duplicate_cap:
        trainer.duplicate_multiplicity_cap_path = duplicate_cap
        trainer.duplicate_multiplicity_cap_sha256 = (
            args.expected_duplicate_cap_sha256)
        trainer.duplicate_multiplicity_fixed_k = args.fixed_edges_per_source
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
            "uniform_with_replacement": False,
            "positive_with_replacement": True,
        }
        if duplicate_cap:
            expected.update({
                "multiplicity_policy": "exact_duplicate_cap_one",
                "multiplicity_cap_artifact_sha256": (
                    args.expected_duplicate_cap_sha256),
                "multiplicity_excluded_source_rows": args.expected_excluded_rows,
                "multiplicity_retained_rows": args.expected_retained_rows,
                "multiplicity_positive_edges_effective": int(effective_edges),
                "multiplicity_positive_source_sampling": (
                    "fuzzy_weight_proportional_over_retained_source_edges_with_replacement"
                ),
                "multiplicity_negative_sampling": "uniform_retained_rows_nonself",
                "multiplicity_positive_destinations": "original_graph_rows",
                "multiplicity_graph_degree": "variable_or_weighted_edge_universe",
            })
        else:
            expected["multiplicity_policy"] = "row_multiplicity_uncapped"
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
        if duplicate_cap:
            retained = getattr(loader, "_retained_node_rows_t", None)
            if (retained is None
                    or len(retained) != args.expected_retained_rows
                    or int(loader.n_pos) != int(effective_edges)
                    or int(loader.source_n_pos - loader.excluded_positive_edges)
                    != int(effective_edges)):
                raise RuntimeError(
                    "production duplicate-cap sampler universe does not match admission")
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
            "duplicate_cap": (
                {
                    "path": os.path.realpath(duplicate_cap),
                    "sha256": args.expected_duplicate_cap_sha256,
                    "fixed_edges_per_source": args.fixed_edges_per_source,
                    "excluded_rows": args.expected_excluded_rows,
                    "retained_rows": args.expected_retained_rows,
                    "excluded_positive_edges": int(loader.excluded_positive_edges),
                }
                if duplicate_cap else None
            ),
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
    parser.add_argument("--duplicate-cap")
    parser.add_argument("--expected-duplicate-cap-sha256")
    parser.add_argument("--fixed-edges-per-source", type=int, default=15)
    parser.add_argument("--expected-excluded-rows", type=int)
    parser.add_argument("--expected-retained-rows", type=int)
    args = parser.parse_args()
    cap_fields = (
        args.expected_duplicate_cap_sha256,
        args.expected_excluded_rows,
        args.expected_retained_rows,
    )
    if (bool(args.duplicate_cap) != all(value is not None for value in cap_fields)
            or (any(value is not None for value in cap_fields)
                and not all(value is not None for value in cap_fields))):
        parser.error(
            "--duplicate-cap requires its expected SHA, excluded rows, and retained rows")
    receipt = run_canary(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
