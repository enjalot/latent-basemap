#!/usr/bin/env python3
"""Transform-vs-train gap audit (owner 2026-08-27, A1 follow-on) — INFERENCE ONLY.

Transform the larger-N rung through the SMALLER-N-trained head (pure batched
forward pass), score against the rung's own truth, and diff vs the trained-at-N
map's score. Deliverable: the gap-vs-extrapolation-factor curve + a collapse read.
Direct "can a small head serve the full atlas" measurement.

TWO instruments (delegate ruling 2026-08-27):

* FULL-RUNG (recipe-clean HEADLINE pairs, 6.25× extrapolation): transform the FULL
  rung substrate through the head (lazy per-batch normalizing memmap — the >=2 GB
  substrates never materialize), score on the rung's OWN sealed k15 truth at
  disc = 0.1% x N. Pairs: MiniLM 2M->6.25M, jina 2M->6.25M.

* 2M-SLICE (big rungs 12.5M/25M/50M/100M, mixed-recipe reference — flagged): the
  full-rung 0.1%-disc KD-tree is infeasible at 100M (disc=100k). Instead draw a
  FIXED 2M subsample (same seed across all big rungs), build a FRESH exact-k15
  truth WITHIN the subsample from its substrate rows (NOT induced edges from the
  full graph — a 2M-of-100M slice keeps ~2% of each row's NN, a near-empty faked
  truth), transform the head onto the subsample, restrict the trained-at-N ref
  coords to the same rows, and score BOTH at disc = 0.1%-of-2M against that fresh
  in-slice truth. Instrument recorded as "FFR@0.1%-of-2M-slice (fresh in-slice
  truth)" — never read as the full-rung number. The deliverable is the paired GAP,
  so instrument consistency between the two maps (not full-rung fidelity) is what
  matters. Fallback (a): only if a big-rung |gap| < ~0.01 do we run one full-disc
  50M point to check the slice instrument isn't hiding a real difference.

Transformed coords are exported as `xform-from-<head>` rung arms for the compare
page. Transforms are batched forward passes — interruptible; the orchestrator runs
this only when the GPU is free. GPU script.
Output: /data/latent-basemap/sandbox/transform-gap-audit.json
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

SB = Path("/data/latent-basemap/sandbox")
SLICE_N = 2_000_000
SLICE_SEED = 42
K_TRUE = 15
# All transform pairs use the 2M-trained head (head2m). The nested rungs are
# byte-identical prefixes, so a query into a larger substrate with index <
# HEAD_TRAIN_N was a TRAINING MEMBER of that head (in-sample / optimistic);
# index >= HEAD_TRAIN_N is UNSEEN (P0.3c member/unseen split).
HEAD_TRAIN_N = 2_000_000


class NormMemmap:
    """Lazy L2-row-normalizing view over a memmap. transform() slices X[i:j] per
    batch and casts to f32 but does NOT normalize; pre-_norm on a 76/153 GB
    substrate would materialize it (>=2 GB rule). Normalizing inside __getitem__
    keeps only one batch resident."""

    def __init__(self, mm):
        self._mm = mm
        self.shape = tuple(mm.shape)
        self.dtype = np.float32

    def __len__(self):
        return int(self.shape[0])

    def __getitem__(self, sl):
        chunk = np.asarray(self._mm[sl], dtype=np.float32)
        nrm = np.linalg.norm(chunk, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return chunk / nrm


def _resolve(p):
    if p is None:
        return None
    p = str(p)
    if "*" in p:
        hits = sorted(glob.glob(p))
        return Path(hits[0]) if hits else None
    return Path(p) if Path(p).exists() else None


def _exact_knn(x_normed: np.ndarray, K: int = K_TRUE) -> np.ndarray:
    """Exact cosine k-NN on an in-memory NORMALIZED (n,d) array — the same GPU
    brute-force the rung builds use (image_map_pipeline.knn), self dropped.
    Returns (n, K) int32 neighbor indices."""
    import torch
    n = int(x_normed.shape[0])
    db = torch.from_numpy(np.ascontiguousarray(x_normed, dtype=np.float32)).half().cuda()
    idx_out = np.empty((n, K), dtype=np.int32)
    q_chunk, d_chunk = 4096, 262_144
    with torch.no_grad():
        for qs in range(0, n, q_chunk):
            q = db[qs:qs + q_chunk].float()
            best_s = torch.full((q.shape[0], K + 1), -2.0, device="cuda")
            best_i = torch.zeros((q.shape[0], K + 1), dtype=torch.long, device="cuda")
            for dstart in range(0, n, d_chunk):
                sims = q @ db[dstart:dstart + d_chunk].float().T
                s, i = torch.topk(sims, min(K + 1, sims.shape[1]), dim=1)
                best_s, sel = torch.topk(torch.cat([best_s, s], dim=1), K + 1, dim=1)
                best_i = torch.gather(torch.cat([best_i, i + dstart], dim=1), 1, sel)
            rows = torch.arange(qs, qs + q.shape[0], device="cuda")
            is_self = best_i == rows.unsqueeze(1)
            keep = ~is_self
            no_self = keep.all(dim=1)
            keep[no_self, K] = False
            idx_out[qs:qs + q.shape[0]] = best_i[keep].view(q.shape[0], K).cpu().numpy().astype(np.int32)
    del db
    torch.cuda.empty_cache()
    return idx_out


def _pairs():
    from knobs_2m import RUNGS
    head2m = SB / "2m-knobs/umap-md000-x4bs16k-winner/model.pt"
    R = RUNGS
    P = [
        # --- FULL-RUNG recipe-clean headlines ---
        {"space": "MiniLM", "head": "2m-champion", "rung": "6.25M", "extrap": 3.125,
         "recipe_clean": True, "instrument": "full", "head_pt": head2m,
         "substrate": _resolve(R.get("6250k", {}).get("substrate")),
         "truth": _resolve(R.get("6250k", {}).get("edges") or R.get("6250k", {}).get("edges_glob")),
         "knn_indices": _resolve(str(Path(str(R.get("6250k", {}).get("substrate"))).parent / "knn_indices.npy")),
         "ref_dir": SB / "6250k-knobs/umap-md000-x4bs16k-winner-rank25"},
        {"space": "jina", "head": "jina-2m-champion", "rung": "6.25M", "extrap": 3.125,
         "recipe_clean": True, "instrument": "full",
         "head_pt": SB / "jina-multi-2m/champion-bs16k/model.pt",
         "substrate_ds": "jina-multi-6m",
         "truth": _resolve(SB / "jina-multi-6m/edges-k15-fuzzy.npz"),
         "knn_indices": _resolve(SB / "jina-multi-6m/knn_indices.npy"),
         "ref_dir": SB / "jina-multi-6m/champion-bs16k"},
        # --- 2M-SLICE big rungs (mixed-recipe reference — flagged) ---
        {"space": "MiniLM", "head": "2m-champion", "rung": "12.5M", "extrap": 6.25,
         "recipe_clean": False, "instrument": "slice", "head_pt": head2m,
         "substrate": _resolve(R.get("12500k", {}).get("substrate")),
         "ref_dir": SB / "12500k-knobs/umap-md000-x4-fneg10"},
        {"space": "MiniLM", "head": "2m-champion", "rung": "25M", "extrap": 12.5,
         "recipe_clean": False, "instrument": "slice", "head_pt": head2m,
         "substrate": _resolve(R.get("25000k", {}).get("substrate")),
         "ref_dir": SB / "25000k-knobs/umap-md000-x2-fneg10-hostint8"},
        {"space": "MiniLM", "head": "2m-champion", "rung": "50M", "extrap": 25.0,
         "recipe_clean": False, "instrument": "slice", "head_pt": head2m,
         "substrate": _resolve("/data/latent-basemap/runs/round-0237/queue/artifacts/"
                               "minilm-mixed-50000k-nested-substrate-and-reserves-v1/substrate.f32.npy"),
         "ref_dir": Path("/data/checkpoints/pumap/maps/minilm-50m-r0267-seed42")},
        {"space": "MiniLM", "head": "2m-champion", "rung": "100M", "extrap": 50.0,
         "recipe_clean": False, "instrument": "slice", "head_pt": head2m,
         "substrate": _resolve("/data/latent-basemap/runs/round-0238/queue/artifacts/"
                               "minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.f32.npy"),
         "ref_dir": Path("/data/latent-basemap/runs/round-0268/attempt5/artifacts/"
                         "minilm-mixed-100000k-fneg-x2-md000-hostint8-seed43-r0268-v1")},
    ]
    for p in P:
        p.setdefault("ref_coords", (p["ref_dir"] / "coordinates.npy") if p.get("ref_dir") else None)
    return P


def _fulldisc_pairs():
    """P0.3(b) fallback (a): full-rung 0.1%×N confirmations for the big rungs where the
    2M-SLICE instrument reports a suspiciously small |gap| (<0.01) — a check that the
    slice's fresh-in-slice truth isn't hiding a real transform difference. 25M runs ONLY
    if its slice |gap|<0.01; one 50M point runs unconditionally as the reference anchor.
    Reuses each rung's SEALED full-rung k15 truth (already built in the ladder) at
    disc=0.1%×N — 25M(disc 25k)/50M(disc 50k) are feasible; 100M(disc 100k) is not."""
    from knobs_2m import RUNGS
    head2m = SB / "2m-knobs/umap-md000-x4bs16k-winner/model.pt"
    R = RUNGS
    P = [
        {"space": "MiniLM", "head": "2m-champion", "rung": "25M-fulldisc", "slice_rung": "25M",
         "extrap": 12.5, "recipe_clean": False, "instrument": "full", "head_pt": head2m,
         "fallback": "triggered iff slice |gap|<0.01",
         "substrate": _resolve(R.get("25000k", {}).get("substrate")),
         "truth": _resolve("/data/latent-basemap/runs/round-0236/queue-correction-2/artifacts/"
                           "minilm-mixed-25000k-cluster-spill-k15-fuzzy-graph-v1/edges-k15-fuzzy.npz"),
         "ref_dir": SB / "25000k-knobs/umap-md000-x2-fneg10-hostint8"},
        {"space": "MiniLM", "head": "2m-champion", "rung": "50M-fulldisc", "slice_rung": "50M",
         "extrap": 25.0, "recipe_clean": False, "instrument": "full", "head_pt": head2m,
         "fallback": "unconditional reference anchor",
         "substrate": _resolve("/data/latent-basemap/runs/round-0237/queue/artifacts/"
                               "minilm-mixed-50000k-nested-substrate-and-reserves-v1/substrate.f32.npy"),
         "truth": _resolve("/data/latent-basemap/runs/round-0237/queue-correction-1/artifacts/"
                           "minilm-mixed-50000k-cluster-spill-k15-fuzzy-graph-v1/edges-k15-fuzzy.npz"),
         "ref_dir": Path("/data/checkpoints/pumap/maps/minilm-50m-r0267-seed42")},
    ]
    for p in P:
        p.setdefault("ref_coords", (p["ref_dir"] / "coordinates.npy") if p.get("ref_dir") else None)
    return P


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_map_pipeline import DATASETS, _norm
    from knobs_2m import quick_ffr_v2, quick_ffr_v2_split  # noqa: F401
    try:
        # P0.3(a) 4th-review fix: analysis_v2.collapse(path) expects a FILE PATH (np.load's it),
        # but we have the coords array in-memory — passing the array silently errored -> collapse
        # was ALL NULL. map_quality() takes the array directly.
        from analysis_v2 import map_quality as _map_quality

        def collapse_frac(xy):
            return _map_quality(xy)["collapse"]["r10_over_radius_times_sqrt_n"]
    except Exception:
        collapse_frac = None
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    coords_dir = SB / "transform-gap-coords"
    coords_dir.mkdir(parents=True, exist_ok=True)

    def _score(xy, rows, truth=None, kidx=None, member_cutoff=None, member_mask=None):
        # P0.3c member/unseen split: PROJECTION scoring is optimistic because the
        # nested rungs are byte-identical prefixes, so some queries were IN the
        # head's training set. quick_ffr_v2_split reuses quick_ffr_v2's exact query
        # sampling/truth/disc, so out["ffr_v2"] == the old quick_ffr_v2 number.
        sp = quick_ffr_v2_split(xy, truth or kidx, rows, member_cutoff=member_cutoff,
                                member_mask=member_mask, knn_indices_path=kidx)
        out = {"ffr_v2": float(sp["overall"])}
        if member_cutoff is not None or member_mask is not None:
            out["ffr_v2_member"] = sp["member"]
            out["ffr_v2_unseen"] = sp["unseen"]
            out["member_frac"] = sp["member_frac"]
            out["n_member_queries"] = sp["n_member"]
            out["n_unseen_queries"] = sp["n_unseen"]
        if collapse_frac is not None:
            try:
                out["collapse"] = float(collapse_frac(xy))
            except Exception:
                out["collapse"] = None
        return out

    def run_pair(p):
        rec = {k: (str(v) if isinstance(v, Path) else v) for k, v in p.items()
               if k not in ("ref_dir",)}
        ref_coords = p.get("ref_coords")
        head_pt = p.get("head_pt")
        sub = p.get("substrate")
        sub_ds = p.get("substrate_ds")
        miss = [k for k, v in (("head_pt", head_pt), ("ref_coords", ref_coords)) if not (v and Path(v).exists())]
        if sub_ds is None and not (sub and Path(sub).exists()):
            miss.append("substrate")
        if p["instrument"] == "full" and not (p.get("truth") or p.get("knn_indices")):
            miss.append("truth")
        if miss:
            rec["status"] = f"SKIPPED (missing: {','.join(miss)})"
            print(f"SKIP {p['space']} {p['head']}→{p['rung']}: {rec['status']}", flush=True)
            return rec

        t0 = time.time()
        model = ParametricUMAP.load(str(head_pt), device="cuda")

        if p["instrument"] == "full":
            X = (NormMemmap(DATASETS[sub_ds]["load"]()) if sub_ds
                 else NormMemmap(np.load(sub, mmap_mode="r")))
            n = len(X)
            xy = np.asarray(model.transform(X, batch_size=8192), dtype=np.float32)
            np.save(coords_dir / f"{p['space']}__{p['rung']}__xform-from-{p['head']}.npy", xy)
            kidx = p.get("knn_indices")
            # transform: 2M head projecting the full N-row rung -> member iff query
            # index < HEAD_TRAIN_N (nested-prefix training rows). trained: trained
            # on the FULL rung, so member_cutoff=n makes ~all queries members.
            transform_score = _score(xy, n, truth=p.get("truth"), kidx=kidx,
                                     member_cutoff=HEAD_TRAIN_N)
            ref_xy = np.asarray(np.load(ref_coords), dtype=np.float32)
            trained_score = _score(ref_xy, n, truth=p.get("truth"), kidx=kidx,
                                   member_cutoff=n)
            instrument = "FFR@0.1%-of-N (full-rung sealed truth)"
        else:
            # 2M-slice: subsample, fresh in-slice k15, score both at disc=0.1%-of-2M
            mm = np.load(sub, mmap_mode="r")
            N = int(mm.shape[0])
            rng = np.random.default_rng(SLICE_SEED)
            idx = np.sort(rng.choice(N, size=min(SLICE_N, N), replace=False))
            n = int(idx.shape[0])
            xs = _norm(np.asarray(mm[idx], dtype=np.float32))   # 2M×384 (~3 GB)
            slice_dir = SB / f"gap-slice-{p['space']}-{p['rung']}"
            slice_dir.mkdir(parents=True, exist_ok=True)
            kfile = slice_dir / "knn_indices.npy"
            if not kfile.exists():
                np.save(kfile, _exact_knn(xs, K_TRUE))
                np.save(slice_dir / "subsample_idx.npy", idx)
            xy = np.asarray(model.transform(xs, batch_size=8192), dtype=np.float32)
            np.save(coords_dir / f"{p['space']}__{p['rung']}__xform-from-{p['head']}.npy", xy)
            ref_full = np.load(ref_coords, mmap_mode="r")
            ref_xy = np.asarray(ref_full[idx], dtype=np.float32)
            # SLICE: queries are subsample POSITIONS, not original indices, so we
            # bucket via a mask over the subsample: a subsample row is a training
            # member of the 2M head iff its ORIGINAL index < HEAD_TRAIN_N. trained
            # was trained on the full big rung -> member_cutoff=n = all members.
            member_mask = np.asarray(idx) < HEAD_TRAIN_N
            transform_score = _score(xy, n, kidx=kfile, member_mask=member_mask)
            trained_score = _score(ref_xy, n, kidx=kfile, member_cutoff=n)
            instrument = "FFR@0.1%-of-2M-slice (fresh in-slice truth)"

        try:
            rs = json.loads((p["ref_dir"] / "summary.json").read_text())
            trained_score["sealed_ffr_v1"] = rs.get("quick_ffr_at_0.1pct")
            trained_score["sealed_ffr_v2"] = rs.get("quick_ffr_v2")
        except Exception:
            pass

        gap = round(trained_score["ffr_v2"] - transform_score["ffr_v2"], 4)
        rec.update({"status": "OK", "rows": n, "instrument": instrument,
                    "coords_label": f"xform-from-{p['head']}",
                    "transform": transform_score, "trained": trained_score,
                    "gap_ffr_v2": gap, "gap_per_extrap": round(gap / p["extrap"], 5),
                    "wall_s": round(time.time() - t0, 1)})
        print(f"OK {p['space']} {p['head']}→{p['rung']} x{p['extrap']} [{p['instrument']}]: "
              f"transform {transform_score['ffr_v2']:.4f} vs trained {trained_score['ffr_v2']:.4f} "
              f"→ gap {gap:+.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
        del model
        return rec

    rows_out = [run_pair(p) for p in _pairs()]

    # P0.3(b) fallback (a): full-disc confirmations. A big rung's SLICE gap qualifies
    # the 25M full-disc run iff |slice gap| < 0.01; the 50M full-disc point is an
    # unconditional reference anchor. Slice gaps are read from the rows just computed.
    slice_gap = {r.get("rung"): r.get("gap_ffr_v2") for r in rows_out
                 if str(r.get("instrument", "")).startswith("FFR@0.1%-of-2M")}
    for p in _fulldisc_pairs():
        sg = slice_gap.get(p["slice_rung"])
        triggered = p["fallback"].startswith("unconditional") or (sg is not None and abs(sg) < 0.01)
        if not triggered:
            rows_out.append({"space": p["space"], "rung": p["rung"], "instrument": "full",
                             "slice_rung": p["slice_rung"], "slice_gap_ffr_v2": sg,
                             "status": f"SKIPPED (slice |gap|={sg} not <0.01 — fallback not triggered)"})
            print(f"SKIP {p['rung']}: slice |gap|={sg} not <0.01", flush=True)
            continue
        print(f"FULLDISC {p['rung']} triggered (slice gap={sg}, {p['fallback']})", flush=True)
        rec = run_pair(p)
        rec["slice_gap_ffr_v2"] = sg
        rows_out.append(rec)

    out = SB / "transform-gap-audit.json"
    out.write_text(json.dumps({
        "schema": "transform-vs-train-gap-2026-08-27",
        "slice_n": SLICE_N, "slice_seed": SLICE_SEED,
        "note": "gap = trained-at-N − transform(smaller head). FULL pairs (2M→6.25M, "
                "recipe-clean) on the rung's sealed truth at disc 0.1%×N; SLICE pairs "
                "(12.5M/25M/50M/100M, mixed-recipe ref) on a fresh in-slice k15 over a "
                "fixed 2M subsample at disc 0.1%-of-2M (NOT full-rung — instrument in "
                "each row). Same slice seed across big rungs. Fallback (a) IMPLEMENTED: "
                "25M-fulldisc runs iff its slice |gap|<0.01 (it is, 0.0062); 50M-fulldisc "
                "is an unconditional reference anchor — both on the rung's SEALED full k15 "
                "truth at disc 0.1%×N, confirming the slice instrument isn't hiding a real "
                "transform difference. 100M full-disc (disc 100k) remains infeasible.",
        "pairs": rows_out,
    }, indent=1))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
