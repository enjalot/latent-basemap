#!/usr/bin/env python3
"""Per-space capacity mini-grid (Phase 2, owner-approved 2026-08-24).

Distill each space's OWN upstream teacher (already on disk from the ceiling
runs) into encoders of widths {1024, 2048, 3072}; regression FFR vs the
space's truth isolates representational demand per embedding space (768-d /
1152-d inputs may shift the width curve vs MiniLM's 384-d).

Usage: space_capacity_probe.py <dataset>
Outputs: sandbox/<dataset>/capacity-h<width>/summary.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")
WIDTHS = (1024, 2048, 3072)
UPDATES = 20_000
BATCH = 8192
SEED = 42


def main() -> int:
    ds = sys.argv[1]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from image_map_pipeline import DATASETS, _norm
    from knobs_2m import BASE_KWARGS, quick_ffr

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    X = _norm(DATASETS[ds]["load"]())
    t = np.load(SANDBOX / ds / "upstream-06dev/coordinates.npy").astype(np.float32)
    t = t - t.mean(axis=0)
    t *= 5.0 / max(float(np.sqrt((t ** 2).sum(axis=1).mean())), 1e-9)
    edges = SANDBOX / ds / "edges-k15-fuzzy.npz"
    n = X.shape[0]
    Xg = torch.from_numpy(X).cuda().half()
    Tg = torch.from_numpy(t).cuda()
    for width in WIDTHS:
        d = SANDBOX / ds / f"capacity-h{width}"
        if (d / "summary.json").exists():
            print(f"{ds} h{width}: done, skip")
            continue
        d.mkdir(parents=True, exist_ok=True)
        p = ParametricUMAP(**{**BASE_KWARGS, "hidden_dim": width, "n_epochs": 1})
        p._init_model(X.shape[1])
        net = p.model
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda u: min(1.0, (u + 1) / 200) * 0.5 * (
                1 + math.cos(math.pi * min(u / UPDATES, 1.0))))
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        torch.manual_seed(SEED)
        t0 = time.time()
        net.train()
        for u in range(UPDATES):
            idx = torch.randint(0, n, (BATCH,), generator=gen, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(Xg.index_select(0, idx).float())
            loss = torch.nn.functional.huber_loss(
                pred.float(), Tg.index_select(0, idx), delta=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            xy = np.concatenate([net(Xg[i:i + 65536].float()).cpu().numpy()
                                 for i in range(0, n, 65536)]).astype(np.float32)
        ffr = quick_ffr(xy, edges, n)
        (d / "summary.json").write_text(json.dumps({
            "probe": "space-capacity", "dataset": ds, "width": width,
            "updates": UPDATES, "wall_s": round(time.time() - t0, 1),
            "quick_ffr_at_0.1pct": float(ffr),
        }, indent=1))
        print(f"{ds} h{width}: distill-FFR {ffr:.4f}", flush=True)
        del p, net, opt
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
