"""CUDA-hidden CPU smoke for x_residency="host_int8" + fneg (Track 4C).

Mirrors smoke_density_nn.py. Constructs a tiny synthetic X + edge npz, trains
1-2 epochs with x_residency="host_int8" and fneg_weight>0, and asserts:
  * the transform output is finite;
  * the int8 dataset was ACTUALLY used (HostInt8ArrayDataset on _X_dev, and the
    pipeline receipt is stamped x_residency=host_int8) — not the fp32 path;
  * the saved checkpoint records x_residency="host_int8";
  * the fneg telemetry is populated (the fog-targeted-negative term ran through
    the int8 gathers).

CPU-only; launches no GPU job.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/home/enjalot/code/latent-basemap")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import torch  # noqa: E402
from basemap.pumap.parametric_umap.core import ParametricUMAP  # noqa: E402
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (  # noqa: E402
    HostInt8ArrayDataset,
)

N, D = 400, 384
rng = np.random.default_rng(0)
X = rng.standard_normal((N, D)).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)

srcs, dsts, wts = [], [], []
for i in range(N):
    for j in rng.choice(N, size=4, replace=False):
        if j == i:
            continue
        srcs.append(i); dsts.append(int(j)); wts.append(1.0)
edges = Path(tempfile.mkdtemp()) / "edges.npz"
np.savez(edges, sources=np.asarray(srcs, np.int64),
         targets=np.asarray(dsts, np.int64), weights=np.asarray(wts, np.float32))

BASE = dict(n_components=2, hidden_dim=64, n_layers=2, n_neighbors=4,
            a=1.9328, b=0.7905, low_dim_kernel="umap", kernel_alpha=1.0,
            learning_rate=1e-3, n_epochs=2, batch_size=128, pos_ratio=0.05,
            architecture="residual_bottleneck", positive_target_mode="binary",
            use_amp=False, use_batchnorm=False, use_dropout=False,
            correlation_weight=0.0, require_full_budget=False, device="cpu")

ok = True

torch.manual_seed(42)
m = ParametricUMAP(**{**BASE, "x_residency": "host_int8", "fneg_weight": 1.0})
m.fit(X, precomputed_edges_path=str(edges), random_state=42)
xy = m.transform(X, batch_size=128)

finite = bool(np.isfinite(xy).all())
print(f"transform finite={finite} shape={xy.shape}")
ok &= finite

used_int8 = (isinstance(m._X_dev, HostInt8ArrayDataset)
             and m._X_dev.storage_dtype == torch.int8)
print(f"host-int8 dataset used: {used_int8} (_X_dev={type(m._X_dev).__name__})")
ok &= used_int8

stamped = (m._pipeline_info.get("x_residency") == "host_int8"
           and m._pipeline_info.get("pipeline") == "host_int8")
print(f"pipeline stamped host_int8: {stamped} ({m._pipeline_info.get('pipeline')},"
      f" {m._pipeline_info.get('sampler_class')})")
ok &= stamped

tel = getattr(m, "fneg_telemetry", None)
tel_ok = isinstance(tel, dict) and "per_epoch" in tel
print(f"fneg telemetry populated: {tel_ok}")
ok &= tel_ok

sp = Path(tempfile.mkdtemp()) / "model.pt"
m.save(str(sp))
ck = torch.load(str(sp), map_location="cpu", weights_only=False)
receipt_ok = ck.get("x_residency") == "host_int8"
print(f"checkpoint records x_residency=host_int8: {receipt_ok} "
      f"(={ck.get('x_residency')!r})")
ok &= receipt_ok

print("SMOKE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
