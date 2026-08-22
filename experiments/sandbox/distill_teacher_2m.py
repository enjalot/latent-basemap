#!/usr/bin/env python3
"""T1+T2 of plan-teacher-distillation.md: distill the 0.6dev 2M layout into
our encoder, then probe generalization. Sandbox, seed 42, no sealed claims.

T1 (train): encoder (promoted architecture via ParametricUMAP) regressed onto
the teacher's coordinates with Huber loss; 10% of rows HELD OUT of training
(the teacher saw them; the encoder must place them from the embedding alone).
Arms: distill-huber (20K updates) and distill-huber-2x (40K).

T2 (probes), per arm:
  - full-map quick-FFR vs the 2M truth (vs teacher 0.4798 / tanh4 0.3690)
  - heldout-FFR: same instrument, queries drawn ONLY from the held-out 10%
  - teacher agreement: median 2D offset in teacher-RMS-radius units
  - wikipedia projection (42M rows) rendered in the map's frame

Outputs: /data/latent-basemap/sandbox/2m-knobs/<arm>/ (review-page cards) —
T1/T2 gates read per plan (T2 pass gate: full FFR >= 0.42).
"""
from __future__ import annotations

import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SUB = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
           "minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
TEACHER = Path("/data/latent-basemap/sandbox/2m-knobs/upstream-06dev-2m/"
               "coordinates.npy")
OUT_ROOT = Path("/data/latent-basemap/sandbox/2m-knobs")
WIKI = sorted(glob.glob(
    "/data/embeddings/wikipedia-en-chunked-120-all-MiniLM-L6-v2/train/*.npy"))
SEED = 42
ROWS = 2_000_000
HOLDOUT_FRAC = 0.10
BATCH = 8192
ARMS = {"distill-huber": 20_000, "distill-huber-2x": 40_000}


def _quick_ffr_queries(xy, edges_path, rows, queries, k_true=15):
    """knobs_2m.quick_ffr with an explicit query set (held-out probes)."""
    from scipy.spatial import cKDTree
    with np.load(edges_path) as z:
        sources, dests = z["sources"], z["targets"]
    order = np.argsort(sources, kind="stable")
    sources, dests = sources[order], dests[order]
    starts = np.searchsorted(sources, np.arange(rows))
    ends = np.searchsorted(sources, np.arange(rows), side="right")
    disc = max(int(rows * 0.001), 1)
    tree = cKDTree(xy)
    _, near = tree.query(xy[queries], k=disc, workers=8)
    hits = total = 0
    for qi, q in enumerate(queries):
        truth = dests[starts[q]:ends[q]][:k_true]
        if len(truth) == 0:
            continue
        hits += np.isin(truth, near[qi]).sum()
        total += len(truth)
    return hits / max(total, 1)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from knobs_2m import BASE_KWARGS, quick_ffr
    from map_renders import binned_counts, render_png, robust_extent

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    X = np.array(np.load(SUB / "substrate.f32.npy", mmap_mode="r"))
    teacher = np.load(TEACHER).astype(np.float32)
    # normalize teacher scale to RMS radius 5 (anchor convention); FFR is
    # scale-invariant, this just conditions the regression.
    t = teacher - teacher.mean(axis=0)
    rms = float(np.sqrt((t ** 2).sum(axis=1).mean()))
    t *= 5.0 / rms

    rng = np.random.default_rng(SEED)
    held = rng.choice(ROWS, size=int(ROWS * HOLDOUT_FRAC), replace=False)
    held_mask = np.zeros(ROWS, dtype=bool)
    held_mask[held] = True
    train_rows = np.nonzero(~held_mask)[0]

    dev = "cuda"
    Xg = torch.from_numpy(X).to(dev).half()
    Tg = torch.from_numpy(t).to(dev)
    tr = torch.from_numpy(train_rows).to(dev)

    for arm, updates in ARMS.items():
        d = OUT_ROOT / arm
        if (d / "summary.json").exists():
            print(f"{arm}: done, skip")
            continue
        d.mkdir(parents=True, exist_ok=True)
        p = ParametricUMAP(**{**BASE_KWARGS, "n_epochs": 1})
        p._init_model(X.shape[1])
        net = p.model
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda u: min(1.0, (u + 1) / 200) * 0.5 * (
                1 + math.cos(math.pi * min(u / updates, 1.0))))
        gen = torch.Generator(device=dev).manual_seed(SEED)
        torch.manual_seed(SEED)
        t0 = time.time()
        net.train()
        for u in range(updates):
            idx = tr[torch.randint(0, len(tr), (BATCH,), generator=gen,
                                   device=dev)]
            xb = Xg.index_select(0, idx).float()
            yb = Tg.index_select(0, idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(xb)
            loss = torch.nn.functional.huber_loss(pred.float(), yb, delta=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if (u + 1) % 5000 == 0:
                print(f"{arm}: {u+1}/{updates} loss {loss.item():.5f}", flush=True)
        wall = time.time() - t0

        p.is_fitted = True
        net.eval()
        with torch.no_grad():
            parts = [net(Xg[i:i + 65536].float()).cpu().numpy()
                     for i in range(0, ROWS, 65536)]
        xy = np.concatenate(parts).astype(np.float32)
        np.save(d / "coordinates.npy", xy)
        p.save(str(d / "model.pt"))
        frame = robust_extent(xy)
        render_png(binned_counts(xy, frame), d / "density.png")

        full_ffr = quick_ffr(xy, SUB / "edges-k15-fuzzy.npz", ROWS)
        held_q = rng.choice(held, size=2000, replace=False)
        held_ffr = _quick_ffr_queries(xy, SUB / "edges-k15-fuzzy.npz", ROWS,
                                      held_q)
        agree = float(np.median(np.linalg.norm(xy - t, axis=1)) / 5.0)

        # wikipedia OOD projection (42M rows through the distilled encoder)
        wiki_dir = Path("/data/latent-basemap/projections/wiki") / arm
        wiki_dir.mkdir(parents=True, exist_ok=True)
        wparts = []
        with torch.no_grad():
            for f in WIKI:
                s = np.load(f, mmap_mode="r")
                for i in range(0, s.shape[0], 65536):
                    b = torch.from_numpy(
                        np.asarray(s[i:i + 65536], dtype=np.float32)).to(dev)
                    wparts.append(net(b).cpu().numpy())
        wxy = np.concatenate(wparts).astype(np.float32)
        np.save(wiki_dir / "coordinates.npy", wxy)
        render_png(binned_counts(wxy, frame), wiki_dir / "density.png")
        wiki_finite = float(np.isfinite(wxy).all(axis=1).mean())
        del wparts, wxy

        (d / "summary.json").write_text(json.dumps({
            "arm": arm, "rung": "2m",
            "overrides": {"teacher": "upstream-06dev-2m", "loss": "huber",
                          "updates": updates, "holdout_frac": HOLDOUT_FRAC},
            "seed": SEED, "wall_s": wall,
            "quick_ffr_at_0.1pct": float(full_ffr),
            "heldout_ffr": float(held_ffr),
            "teacher_agreement_median_r": agree,
            "wiki_projection": str(wiki_dir),
            "wiki_finite_frac": wiki_finite,
            "substrate": str(SUB / "substrate.f32.npy"),
            "edges": str(SUB / "edges-k15-fuzzy.npz"),
            "note": "T1 teacher distillation (plan-teacher-distillation.md); "
                    "10% rows held out of encoder training; sandbox only.",
        }, indent=1))
        print(f"{arm}: full FFR {full_ffr:.4f} | heldout {held_ffr:.4f} | "
              f"agree {agree:.3f}R | wiki finite {wiki_finite:.4f} | "
              f"{wall/60:.1f} min")
        del p, net, opt
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
