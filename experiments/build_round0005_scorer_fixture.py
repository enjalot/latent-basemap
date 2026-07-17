"""Build a deterministic signed-input fixture for real Round 0005 scorer paths."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import canonical_json, ordered_array_sha256, sha256_bytes
from basemap.output_safety import atomic_save_new_npz, refuse_existing
from basemap.panel_v2 import PanelV2Config
from experiments.compare_panel_cache import FIXTURE_SCHEMA


def build_fixture(path: str, *, rows: int = 512, query_rows: int = 32,
                  dimensions: int = 12, seed: int = 20260716) -> dict:
    path = os.path.abspath(path)
    if not path.startswith("/data/"):
        raise ValueError("Round 0005 scorer fixture must live under /data")
    refuse_existing(path, label="scorer fixture")
    if rows <= 32 or query_rows <= 0 or dimensions <= 1:
        raise ValueError("scorer fixture dimensions are too small")
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((rows, dimensions)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xq = rng.standard_normal((query_rows, dimensions)).astype(np.float32)
    Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
    Z = rng.standard_normal((rows, 2)).astype(np.float32)
    Zq = rng.standard_normal((query_rows, 2)).astype(np.float32)
    cfg = PanelV2Config(frac=0.03, k_hit=10, corpus_chunk=128,
                        block_elems=1_000_000, rerank_byte_cap=16_000_000,
                        peak_byte_cap=64_000_000)
    meta = {"seed": int(seed), "panel_config": cfg.__dict__}
    payload_sha = sha256_bytes(canonical_json({
        "meta": meta,
        "arrays": {name: ordered_array_sha256(value)
                   for name, value in {"X": X, "Z": Z, "Xq": Xq, "Zq": Zq}.items()},
    }))
    atomic_save_new_npz(
        path, immutable=True, schema=np.array(FIXTURE_SCHEMA), X=X, Z=Z, Xq=Xq, Zq=Zq,
        meta=np.array(json.dumps(meta, sort_keys=True, separators=(",", ":"))),
        payload_sha256=np.array(payload_sha))
    return {"path": path, "payload_sha256": payload_sha,
            "shape": {"X": list(X.shape), "Xq": list(Xq.shape)}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--query-rows", type=int, default=32)
    parser.add_argument("--dimensions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args(argv)
    print(json.dumps(build_fixture(
        args.out, rows=args.rows, query_rows=args.query_rows,
        dimensions=args.dimensions, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
