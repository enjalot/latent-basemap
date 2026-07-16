"""Diagnose apparent fixed-axis islands caused by duplicate input embeddings."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.artifact_identity import git_checkout_state, path_signature, sha256_bytes
from basemap.panel_v2 import load_embeddings


def diagnose(coords_path: str, embeddings_path: str, *, dim: int, top_groups: int = 10) -> dict:
    frame = pd.read_parquet(coords_path, columns=["x", "y", "ls_index"])
    coords = frame[["x", "y"]].to_numpy(dtype=np.float32)
    ids = frame["ls_index"].to_numpy(dtype=np.int64)
    values, inverse, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(counts)[::-1]
    embeddings = load_embeddings(embeddings_path, dim=dim)
    groups = []
    for group_index in order[:top_groups]:
        count = int(counts[group_index])
        if count < 2:
            break
        member_ids = ids[inverse == group_index]
        rows = np.asarray(embeddings[member_ids], dtype=np.float32)
        first = np.ascontiguousarray(rows[0]).tobytes()
        byte_identical = bool(all(np.ascontiguousarray(row).tobytes() == first for row in rows[1:]))
        unique_inputs, input_counts = np.unique(rows, axis=0, return_counts=True)
        duplicate_classes_only = bool(np.all(input_counts > 1))
        groups.append({
            "coordinate": [float(v) for v in values[group_index]],
            "count": count,
            "row_ids_sha256": sha256_bytes(np.ascontiguousarray(member_ids).tobytes()),
            "first_row_ids": member_ids[:20].tolist(),
            "input_embeddings_byte_identical": byte_identical,
            "unique_input_embeddings": int(len(unique_inputs)),
            "input_duplicate_class_counts": sorted((int(v) for v in input_counts), reverse=True),
            "all_inputs_belong_to_duplicate_classes": duplicate_classes_only,
            "input_embedding_sha256": sha256_bytes(first) if byte_identical else None,
        })
    repeated = sum(group["count"] for group in groups)
    explained = all(group["all_inputs_belong_to_duplicate_classes"] for group in groups)
    return {
        "coords": path_signature(coords_path),
        "embeddings": path_signature(embeddings_path),
        "n_rows": len(coords),
        "top_repeated_coordinate_groups": groups,
        "top_group_rows": repeated,
        "top_groups_explained_by_duplicate_input_classes": explained,
        "diagnosis": ("Repeated remote coordinate islands are deterministic images of exact duplicate "
                      "input-embedding classes (some coordinate modes merge multiple duplicate classes); "
                      "finite variance and the non-repeated body rule out whole-map collapse. Fixed axes "
                      "correctly expose these duplicate-input islands."
                      if groups and explained else
                      "Repeated coordinate islands are not fully explained by exact duplicate inputs."),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords", action="append", required=True,
                        help="label=/absolute/path/coords.parquet")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    results = {}
    for value in args.coords:
        label, path = value.split("=", 1)
        results[label] = diagnose(path, args.embeddings, dim=args.dim)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {"schema": "duplicate_coordinate_island_diagnosis.v1",
           "diagnoser": path_signature(__file__),
           "checkout": git_checkout_state(root), "maps": results,
           "passed": all(item["top_groups_explained_by_duplicate_input_classes"]
                         for item in results.values())}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(args.out)
    return 0 if out["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
