from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


SAMPLER_VERSION = "reject-positive-v1"


@dataclass(frozen=True)
class GraphCacheSpec:
    cache_dir: str
    key: str
    manifest: Dict[str, Any]

    @property
    def artifact_dir(self) -> Path:
        return Path(self.cache_dir).expanduser() / self.key

    @property
    def p_sym_path(self) -> str:
        return str(self.artifact_dir / "p_sym.pkl")

    @property
    def negatives_path(self) -> str:
        return str(self.artifact_dir / "negatives.pkl")

    @property
    def manifest_path(self) -> str:
        return str(self.artifact_dir / "manifest.json")

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "artifact_dir": str(self.artifact_dir),
            "p_sym_path": self.p_sym_path,
            "negatives_path": self.negatives_path,
            "manifest_path": self.manifest_path,
            "p_sym_exists": Path(self.p_sym_path).exists(),
            "negatives_exists": Path(self.negatives_path).exists(),
            "manifest_exists": Path(self.manifest_path).exists(),
        }


def hash_array_exact(x: np.ndarray, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    """Return an exact content hash for an array without making avoidable copies."""
    arr = np.ascontiguousarray(x)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    view = memoryview(arr).cast("B")
    for start in range(0, len(view), chunk_bytes):
        h.update(view[start:start + chunk_bytes])
    return h.hexdigest()


def stable_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_graph_cache_spec(cfg, x_train: np.ndarray, eval_mode: str) -> Optional[GraphCacheSpec]:
    dc = cfg.data
    cache_dir = getattr(dc, "graph_cache_dir", "") or ""
    if not getattr(dc, "use_graph_cache", False):
        return None
    if not cache_dir:
        raise ValueError("data.use_graph_cache=true requires data.graph_cache_dir")
    if eval_mode != "holdout_rows":
        raise ValueError("Graph cache currently supports holdout_rows mode only")
    if cfg.train.resample_negatives:
        raise ValueError("Graph cache does not support train.resample_negatives=true")

    data_fingerprint = hash_array_exact(x_train)
    manifest = {
        "created_by": "experiments.artifact_cache",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data": {
            "source": dc.source,
            "h5_path": dc.h5_path,
            "h5_dataset": dc.h5_dataset,
            "memmap_dirs": [os.path.expanduser(p) for p in dc.memmap_dirs],
            "lancedb_path": dc.lancedb_path,
            "lancedb_table": dc.lancedb_table,
            "lancedb_columns": dc.lancedb_columns,
            "input_dim": dc.input_dim,
            "n_samples": dc.n_samples,
            "random_seed": dc.random_seed,
            "train_shape": list(x_train.shape),
            "train_dtype": str(x_train.dtype),
            "train_content_hash": data_fingerprint,
        },
        "graph": {
            "n_neighbors": dc.n_neighbors,
            "metric": "l2",
            "p_sym_format": "pickle",
        },
        "negatives": {
            "sampler_version": SAMPLER_VERSION,
            "random_state": dc.random_seed,
            "negatives_per_positive": 5,
        },
        "train": {
            "positive_target_mode": cfg.train.positive_target_mode,
            "pos_ratio": cfg.train.pos_ratio,
        },
    }
    key = stable_hash({
        "data": manifest["data"],
        "graph": manifest["graph"],
        "negatives": manifest["negatives"],
    })
    return GraphCacheSpec(cache_dir=cache_dir, key=key, manifest=manifest)


def write_manifest_if_missing(spec: GraphCacheSpec) -> None:
    artifact_dir = spec.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(spec.manifest_path)
    if not manifest_path.exists():
        with manifest_path.open("w") as f:
            json.dump(spec.manifest, f, indent=2)
