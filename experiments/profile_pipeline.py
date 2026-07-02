#!/usr/bin/env python3
"""Phase-resolved profiler for the edge-list input pipeline.

Breaks per-batch wall time into: (1) sampler/loader batch construction,
(2) feature gather + host->device transfer, (3) model fwd/bwd/step. Compares the
legacy list-of-tuples path against the device-resident fast path. GPU phases are
timed with cuda.synchronize() around each phase.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def sync(device):
    if "cuda" in str(device):
        torch.cuda.synchronize()


def build(n_rows, dim, k, seed, device):
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((n_rows, dim)).astype(np.float32)
    sources = np.repeat(np.arange(n_rows, dtype=np.int32), k)
    targets = rng.randint(0, n_rows, size=n_rows * k).astype(np.int32)
    weights = rng.uniform(0.01, 1.0, size=n_rows * k).astype(np.float32)
    return X, sources, targets, weights


def make_model(dim, device):
    from basemap.pumap.parametric_umap.models.mlp import ResidualBottleneckMLP
    m = ResidualBottleneckMLP(input_dim=dim, hidden_dim=1024, output_dim=2,
                              num_layers=3).to(device)
    return m


def step(model, opt, src, dst, targets, device, use_amp):
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
        zs = model(src); zd = model(dst)
        dists = torch.norm(zs - zd, dim=1, p=2)
        qs = torch.pow(1 + dists, -1)
        qs = torch.clamp(qs, 1e-7, 1 - 1e-7)
    loss = torch.nn.functional.binary_cross_entropy(qs.float(), targets.float())
    loss.backward()
    opt.step()


def profile_legacy(X, sources, targets, weights, device, batch, pr, steps, use_amp):
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        EdgeListBalancedIterator)
    from basemap.pumap.parametric_umap.datasets.covariates_datasets import (
        VariableDataset)
    from basemap.pumap.parametric_umap.utils.data_prefetcher import DataPrefetcher

    dataset = VariableDataset(X).to(device)
    n = X.shape[0]
    loader = EdgeListBalancedIterator(sources, targets, weights, n_nodes=n,
                                      pos_ratio=pr, batch_size=batch, shuffle=True,
                                      random_state=0, positive_target_mode="binary")
    model = make_model(X.shape[1], device); opt = torch.optim.AdamW(model.parameters(), 1e-3)
    it = iter(loader)
    t_sample = t_gather = t_step = 0.0
    done = 0
    for _ in range(steps + 3):
        sync(device); a = time.perf_counter()
        edge_batch, labels = next(it)
        sync(device); b = time.perf_counter()
        src_idx = [i for i, j in edge_batch]; dst_idx = [j for i, j in edge_batch]
        src = dataset[src_idx]; dst = dataset[dst_idx]
        tgt = torch.as_tensor(labels, dtype=torch.float32).to(device)
        if not src.is_cuda and "cuda" in str(device):
            src = src.to(device); dst = dst.to(device)
        sync(device); c = time.perf_counter()
        step(model, opt, src, dst, tgt, device, use_amp)
        sync(device); d = time.perf_counter()
        if done >= 3:  # warmup skip
            t_sample += b - a; t_gather += c - b; t_step += d - c
        done += 1
    return t_sample, t_gather, t_step, steps


def profile_fast(X, sources, targets, weights, device, batch, pr, steps, use_amp):
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        DeviceArrayDataset, DeviceEdgeSampler)
    n = X.shape[0]
    dd = DeviceArrayDataset(X, device)
    loader = DeviceEdgeSampler(dd, sources, targets, weights, n_nodes=n,
                               pos_ratio=pr, batch_size=batch, shuffle=True,
                               random_state=0, positive_target_mode="binary",
                               device=device)
    model = make_model(X.shape[1], device); opt = torch.optim.AdamW(model.parameters(), 1e-3)
    it = iter(loader)
    t_sample = t_step = 0.0
    done = 0
    for _ in range(steps + 3):
        sync(device); a = time.perf_counter()
        src, dst, tgt = next(it)          # gather is fused into the sampler
        sync(device); b = time.perf_counter()
        step(model, opt, src, dst, tgt, device, use_amp)
        sync(device); c = time.perf_counter()
        if done >= 3:
            t_sample += b - a; t_step += c - b
        done += 1
    return t_sample, 0.0, t_step, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-rows", type=int, default=100000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--pos-ratio", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()
    dev = args.device
    use_amp = "cuda" in str(dev)
    X, s, t, w = build(args.n_rows, args.dim, args.k, 42, dev)
    b = args.batch_size

    print(f"# device={dev} X=({args.n_rows},{args.dim}) batch={b} steps={args.steps}")
    for name, fn in [("legacy", profile_legacy), ("fast  ", profile_fast)]:
        ts, tg, tp, st = fn(X, s, t, w, dev, b, args.pos_ratio, args.steps, use_amp)
        tot = ts + tg + tp
        sps = st * b / tot
        print(f"\n[{name}] {sps:,.0f} samples/s  ({1000*tot/st:.1f} ms/step total)")
        print(f"   sample/build : {1000*ts/st:7.2f} ms/step ({100*ts/tot:4.1f}%)")
        print(f"   gather+xfer  : {1000*tg/st:7.2f} ms/step ({100*tg/tot:4.1f}%)")
        print(f"   fwd/bwd/step : {1000*tp/st:7.2f} ms/step ({100*tp/tot:4.1f}%)")


if __name__ == "__main__":
    main()
