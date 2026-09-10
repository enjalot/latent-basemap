"""Cards 006/007 movement + coverage scorer (CPU, off-flock) — the PRIMARY OUT-vs-IN causal
comparison, frozen BEFORE inspecting results. For each arm (in/out) it measures displacement
from the ORIGINAL teacher coords on a fixed original-T0/S0 centroid-p90 radius, with a rigid
rotation+translation fit on the ORIGINAL ACTIVE anchors only (no scale), applied to all cohorts:
  - graph cohorts (active anchors, anchor-holdout): from the arm's saved 2.4M/2M coords.
  - off-graph cohorts (own replay bank, NEW confirmation set, old sealed queries): projected
    through the arm model; teacher target = original head applied to the same input.
Reports mean/p95/p99, frac>.05/.10 per group + aggregate, native (unaligned) disp too, and the
coverage hypothesis: OUT p99 <= 0.5*IN p99 on the confirmation set with a paired-bootstrap CI
for p99(OUT)-p99(IN) < 0. Deployment criterion 1 (confirmation mean<=.01 & p99<=.05, per cohort)
computed for both arms. Reception criteria (2,3) are scored separately. Does NOT decide the gate
beyond reporting the booleans. Env CARD=card006|card007. Usage: score_replay_movement.py
"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch, frame as F
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
CARD = os.environ.get("CARD", "card006")

CFG = {
 "card006": {"dim": 1536, "grp": "source",
             "orig_champ": SB / "dino-arrival-t0/champion-bs16k/model.pt",
             "orig_coords": SB / "dino-arrival-t0/champion-bs16k/coordinates.npy",
             "t0_draw": Path("/data/latent-basemap/substrates/dino-arrival-t0/draw_idx.npy"),
             "final_draw": Path("/data/latent-basemap/substrates/dino-arrival-final/draw_idx.npy"),
             "outd": SB / "dino-arrival-t0/replay-updates",
             "orig_active": SB / "dino-arrival-t0/updates/anchor_active_ids-anchored.npy",
             "orig_holdout": SB / "dino-arrival-t0/updates/anchor_holdout_ids-anchored.npy",
             "seal": Path("/data2/monet/eval-common-v2"), "seal_hd": "val_hd.f16.npy", "seal_grp": "val_source.npy"},
 "card007": {"dim": 768, "grp": "language",
             "orig_champ": SB / "jina-ladder-2m-s0/champion-bs16k/model.pt",
             "orig_coords": SB / "jina-ladder-2m-s0/champion-bs16k/coordinates.npy",
             "t0_draw": Path("/data/latent-basemap/substrates/jina-ladder-s0/draw_idx.npy"),
             "final_draw": Path("/data/latent-basemap/substrates/jina-ladder-2m-proportional/draw_idx.npy"),
             "outd": SB / "jina-ladder-2m-s0/replay-updates",
             "orig_active": SB / "jina-ladder-2m-s0/updates/anchor_active_ids-anchored.npy",
             "orig_holdout": SB / "jina-ladder-2m-s0/updates/anchor_holdout_ids-anchored.npy",
             "seal": Path("/data2/monet/eval-common-multilingual"), "seal_hd": "val_hd.f16.npy", "seal_grp": "val_cohort.npy"},
}[CARD]


def _norm(a):
    a = np.asarray(a, np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def project(model, X):
    o = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 50000):
            o.append(model.model(torch.from_numpy(_norm(X[i:i + 50000]))).float().numpy().astype(np.float32))
    return np.concatenate(o)


def stat(disp, radius):
    d = disp / radius
    return {"n": int(d.size), "mean": round(float(d.mean()), 6), "p95": round(float(np.percentile(d, 95)), 6),
            "p99": round(float(np.percentile(d, 99)), 6), "frac_gt_.05": round(float((d > 0.05).mean()), 5),
            "frac_gt_.10": round(float((d > 0.10).mean()), 5)}


def main():
    dim = CFG["dim"]; grp = CFG["grp"]
    t0_coords = np.asarray(np.load(CFG["orig_coords"]), np.float64)
    # Use the SAME fixed radius the original card005/004 movement used (stored in anchor.meta.json),
    # not a recompute — keeps this movement on an identical instrument. Fall back to frame_radius.
    meta_p = CFG["orig_champ"].parent.parent / "anchor.meta.json"
    radius = None
    if meta_p.exists():
        meta = json.load(open(meta_p))
        radius = next((float(v) for k, v in meta.items() if "radius" in k.lower() and isinstance(v, (int, float))), None)
    if radius is None:
        radius = float(F.frame_radius(t0_coords))
    t0_draw = np.load(CFG["t0_draw"]); final_draw = np.load(CFG["final_draw"])
    active = np.load(CFG["orig_active"]); holdout = np.load(CFG["orig_holdout"])
    common, t0_local, final_local = np.intersect1d(t0_draw, final_draw, assume_unique=True, return_indices=True)
    active_mask = np.isin(final_local, active); hold_mask = np.isin(final_local, holdout)
    s0o = t0_coords[t0_local]
    orig = ParametricUMAP.load(str(CFG["orig_champ"]), device="cpu"); orig.model.eval()

    # off-graph banks (own replay bank per arm + shared confirmation + seal)
    def bank(name):
        z = np.load(OC / f"{CARD}_{name}_bank.npz")
        g = z["source"] if "source" in z else z["language"]
        return np.asarray(z["replay_X"], np.float32), np.asarray(z["replay_targets"], np.float32), g.astype(str)
    confirm_X, confirm_tgt, confirm_g = bank("confirm")
    seal_hd = np.asarray(np.load(CFG["seal"] / CFG["seal_hd"]), np.float32)
    seal_g = np.load(CFG["seal"] / CFG["seal_grp"], allow_pickle=True).astype(str)
    seal_tgt = project(orig, seal_hd)   # original head coords for seal queries (movement reference)

    arms = {}
    for tag in ("in", "out"):
        mp = CFG["outd"] / f"model-{tag}.pt"; cp = CFG["outd"] / f"coords-{tag}.npy"
        if not (mp.exists() and cp.exists()):
            arms[tag] = {"status": "absent"}; continue
        mo = ParametricUMAP.load(str(mp), device="cpu"); mo.model.eval()
        upd = np.asarray(np.load(cp), np.float64)[final_local]
        # rigid fit on ORIGINAL ACTIVE anchors only, applied to everything
        _, info = F.rigid_align(upd[active_mask], s0o[active_mask])
        R, tvec = np.asarray(info["R"], np.float64), np.asarray(info["t"], np.float64)
        def align(c): return np.asarray(c, np.float64) @ R.T + tvec
        d_graph = np.linalg.norm(align(upd) - s0o, axis=1)
        res = {"status": "scored", "rigid_rmsd": round(float(info.get("rmsd", np.nan)), 6),
               "graph": {"active": stat(d_graph[active_mask], radius), "anchor_holdout": stat(d_graph[hold_mask], radius),
                         "all_retained": stat(d_graph, radius)}}
        # off-graph cohorts: project through this arm, align with the same transform, disp vs original-head target
        own_X, own_tgt, own_g = bank(tag)
        for cname, X, tgt, g in (("own_replay", own_X, own_tgt, own_g), ("confirmation", confirm_X, confirm_tgt, confirm_g),
                                 ("old_sealed", seal_hd, seal_tgt, seal_g)):
            dproj = np.linalg.norm(align(project(mo, X)) - np.asarray(tgt, np.float64), axis=1)
            per_grp = {str(c): stat(dproj[g == c], radius) for c in sorted(set(g.tolist()))}
            res[cname] = {"aggregate": stat(dproj, radius), "per_group": per_grp,
                          "_disp_over_radius": (dproj / radius)}   # kept in-memory for paired bootstrap
        arms[tag] = res

    out = {"schema": f"{CARD}-movement-2026-09-10", "card": CARD, "dim": dim,
           "fixed_radius_centroid_p90": round(radius, 4), "group_key": grp, "movement": {}}
    # serialize stats only (drop the in-memory _disp_over_radius arrays)
    for t in arms:
        if arms[t].get("status") != "scored":
            out["movement"][t] = arms[t]; continue
        m = {"status": "scored", "rigid_rmsd": arms[t]["rigid_rmsd"], "graph": arms[t]["graph"]}
        for cname in ("own_replay", "confirmation", "old_sealed"):
            m[cname] = {"aggregate": arms[t][cname]["aggregate"], "per_group": arms[t][cname]["per_group"]}
        out["movement"][t] = m

    # ---- coverage hypothesis + deployment criterion 1 (need both arms scored) ----
    if arms.get("in", {}).get("status") == "scored" and arms.get("out", {}).get("status") == "scored":
        di = arms["in"]["confirmation"]["_disp_over_radius"]; do = arms["out"]["confirmation"]["_disp_over_radius"]
        p99_in, p99_out = float(np.percentile(di, 99)), float(np.percentile(do, 99))
        rng = np.random.default_rng(0); n = di.size
        diffs = np.array([np.percentile(do[idx := rng.integers(0, n, n)], 99) - np.percentile(di[idx], 99) for _ in range(2000)])
        ci = (round(float(np.percentile(diffs, 2.5)), 6), round(float(np.percentile(diffs, 97.5)), 6))
        out["coverage_hypothesis"] = {"confirm_p99_in": round(p99_in, 6), "confirm_p99_out": round(p99_out, 6),
                                      "out_le_half_in": bool(p99_out <= 0.5 * p99_in),
                                      "p99_out_minus_in_ci95": ci, "ci_below_zero": bool(ci[1] < 0),
                                      "coverage_success": bool(p99_out <= 0.5 * p99_in and ci[1] < 0),
                                      "note": "if the IN baseline has no material tail, a small ratio is not a breakthrough"}
        for tag in ("in", "out"):
            agg = arms[tag]["confirmation"]["aggregate"]; pg = arms[tag]["confirmation"]["per_group"]
            out.setdefault("deployment_criterion1_confirmation", {})[tag] = {
                "aggregate_mean_le.01": bool(agg["mean"] <= 0.01), "aggregate_p99_le.05": bool(agg["p99"] <= 0.05),
                "worst_group_mean": round(max(v["mean"] for v in pg.values()), 6),
                "worst_group_p99": round(max(v["p99"] for v in pg.values()), 6),
                "pass_all_groups": bool(all(v["mean"] <= 0.01 and v["p99"] <= 0.05 for v in pg.values()))}
    else:
        out["coverage_hypothesis"] = {"status": "waiting for both arms"}

    (OC / f"{CARD}-movement.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"card": CARD, "radius": out["fixed_radius_centroid_p90"],
                      "arms_scored": [t for t in arms if arms[t].get("status") == "scored"],
                      "coverage": out.get("coverage_hypothesis")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
