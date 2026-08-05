#!/usr/bin/env python3
"""Run one R0197 scale with the accepted R0196 fixed-chunk inference patch."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from basemap.artifact_identity import expected_input_signature
from basemap.round0197_grease_baseline import SELECTED_PATCHES
from experiments import round0181_numap_reference as base
from experiments.round0196_grease_batch_stable_reference import (
    BatchStableInferenceGrEASE,
)
from basemap.round0196_grease_batch_stable import fixed_chunks


def _install_patch(selected_patch: str) -> None:
    if selected_patch not in SELECTED_PATCHES:
        raise RuntimeError("R0197 selected inference patch is not registered")
    # R0181's installer reads this module global, so replace only the GrEASE
    # implementation that NUMAP constructs. Training remains the R0181 path;
    # every inference call uses the reviewed fixed-geometry network adapter.
    base.StoredTrainNormalizationGrEASE = BatchStableInferenceGrEASE
    if selected_patch == "fixed-256-row-grease-and-pumap-networks":
        from numap.umap_pytorch import PUMAP

        original = PUMAP.transform

        def fixed_transform(self: Any, values: torch.Tensor):
            return fixed_chunks(values, lambda chunk: original(self, chunk))

        PUMAP.transform = fixed_transform


def run(
    *, matrix: str, queries: str, rows: int, scale: str, selected_patch: str,
    output: str, smoke: bool = False,
) -> dict[str, Any]:
    if os.path.exists(output):
        if not os.path.isdir(output) or os.listdir(output):
            raise RuntimeError("R0197 reference output must be a fresh empty directory")
    else:
        os.makedirs(output)
    _install_patch(selected_patch)
    started = time.monotonic()
    base_output = os.path.join(output, "base")
    base_execution = base.run_fit(
        matrix=matrix,
        queries=queries,
        output=base_output,
        rows=rows,
        smoke=smoke,
    )
    checkpoint = base_execution.get("checkpoint") or {}
    for key in ("reload_full_max_abs_error", "reload_batch_max_abs_error"):
        if float(checkpoint.get(key, 1.0)) > 1.0e-4:
            raise RuntimeError(f"R0197 patched reload guard failed at {key}")
    receipt = {
        "schema": "round0197-grease-batch-stable-reference-execution-v1",
        "mode": "smoke" if smoke else "real",
        "scale": scale,
        "train_rows": int(rows),
        "query_rows": int(base_execution["query_rows"]),
        "dimension": int(base_execution["dimension"]),
        "cuda_available": bool(base_execution["cuda_available"]),
        "cuda_device": base_execution.get("cuda_device"),
        "inference_patch": {
            "source_round": "0196",
            "source_capability": "jina-grease-batch-stable-inference-patch-v1",
            "selected_patch": selected_patch,
            "chunk_rows": 256,
            "normalization": "stored R0181 train mean/std",
            "orthonormalization_weights_updated_at_inference": False,
            "vendored_patch": True,
        },
        "base_execution": base_execution,
        "base_execution_receipt": expected_input_signature(
            os.path.join(base_output, "execution.json")
        ),
        "base_output": "base",
        "total_seconds": time.monotonic() - started,
    }
    with open(os.path.join(output, "execution.json"), "x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix")
    parser.add_argument("--queries")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--selected-patch", required=True, choices=sorted(SELECTED_PATCHES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        os.makedirs(args.output, exist_ok=False)
        matrix, queries = base._write_smoke_inputs(args.output)
        receipt = run(
            matrix=matrix,
            queries=queries,
            rows=256,
            scale="smoke",
            selected_patch=args.selected_patch,
            output=os.path.join(args.output, "fit"),
            smoke=True,
        )
    else:
        if not args.matrix or not args.queries or args.rows is None:
            parser.error("real execution requires matrix, queries, and rows")
        receipt = run(**vars(args))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
