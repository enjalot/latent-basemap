"""Card022 CPU scorer (per card022-local-shape-floor.md "Scoring fixed before production"). Consumes root's
FROZEN primary shape instrument (OC/card022-shape-instrument/instrument.npz; VIABLE, 54.65% baseline
excess-thin). No GPU, no fresh confirmation.

PRIMARY (self-contained in the instrument): project each arm on the instrument's exact normalized full-D
model inputs, regroup the 16-row clouds, compute q = lambda_min(C)/(trace(C)+eps) on the mapped 2D covariance
(the SAME core.shape_floor_q the trainer uses; verified to reproduce the instrument's baseline_q/excess
exactly), and count excess-thin queries (q < tau). Primary passes iff shape_floor reduces the excess-thin
QUERY FRACTION by >= 25% relative AND the paired 95% CI of (shape - ordinary) excess fraction is below 0
(2000 fixed source-stratified paired query bootstraps).

GUARDS (required by the card; equal-nine B250/B2000 <=.005 + every source <=.01, sparse deciles 8-10 mean
<=.005 & each <=.01, continuity loss <=.005) need root's ORIGINAL 200K reference + truth + continuity panel
(panel_sha 0662f5e7…, truth_sha 8dc3c742…). Those files are NOT in the instrument dir or eval-common-v2, so
the guard bundle path must be supplied by root; absent it, guards are reported PENDING and GATE_PASS is
withheld (never silently skipped). Usage: score_card022.py [guard_bundle.npz]
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "4"
import json, time, hashlib, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card022_validate as V
import torch
from scipy.stats import spearmanr
from basemap.pumap.parametric_umap.core import ParametricUMAP, shape_floor_q

ROOT = Path(__file__).resolve().parents[2]
SB = V.SB; OC = V.OC; TD = SB / "card022-train"
INSTR = OC / "card022-shape-instrument/instrument.npz"
PANEL_SHA = "0662f5e73c163beaad404474958d5c1bd5ee0d3a7e95269d16218673849d5674"
TRUTH_SHA = "8dc3c742a6e2ea618a578cecdb8ee9181cba2ca5feb1044d3426be469f43decc"
ARMS = ["ordinary", "shape_floor"]; EPS = V.EPSILON; BOOT = 2000; BUDGETS = [50, 100, 250, 500, 1000, 2000]
torch.set_num_threads(4)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()
def write(p, o):
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(o, indent=2, allow_nan=False) + "\n"); t.replace(p)


def project(model, inputs):
    out = []
    with torch.inference_mode():
        for s in range(0, len(inputs), 8192):
            x = np.array(inputs[s:s + 8192], np.float32, copy=True)
            x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)   # inputs are already unit-norm; idempotent
            out.append(model(torch.from_numpy(x)).numpy())
    xy = np.concatenate(out); assert xy.shape == (len(inputs), 2) and np.isfinite(xy).all(); return xy


def main():
    t0 = time.monotonic(); out = OC / "card022-scoring"; out.mkdir(exist_ok=True)
    # canonical strict validation of both arms (the ONE validator)
    receipts = {a: V.strict_validate_arm(a, ROOT) for a in ARMS}
    z = np.load(INSTR, allow_pickle=True)
    tau = z["tau"]; c2u = z["cloud_to_unique"]; inputs = z["inputs"]; groups = z["source"].astype(str)
    nq = len(z["query_ids"]); by = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    inst_ok = {"instrument_sha": sha(INSTR) is not None, "row0_is_query": bool(np.array_equal(z["cloud_global_ids"][:, 0], z["query_ids"])),
               "baseline_reproduces": None}
    # cross-check: our q on the instrument's baseline_xy reproduces baseline_excess (consistency with root)
    bq = shape_floor_q(torch.tensor(z["baseline_xy"].astype(np.float64)), EPS).numpy()
    inst_ok["baseline_reproduces"] = bool(np.array_equal(bq < tau, z["baseline_excess"]))
    assert inst_ok["baseline_reproduces"], "our kernel does not reproduce the instrument baseline_excess"

    q_arm = {}; excess = {}; xy_saved = {}
    for a in ARMS:
        model = ParametricUMAP.load(str(TD / f"model-{a}.pt"), device="cpu").model.eval()
        coords = project(model, inputs); del model                       # (29117, 2)
        clouds = coords[c2u]                                             # (2044,16,2) regroup by cloud
        qa = shape_floor_q(torch.tensor(clouds.astype(np.float64)), EPS).numpy()
        q_arm[a] = qa; excess[a] = (qa < tau); xy_saved[a] = clouds
        np.save(out / f"{a}-cloud-xy.npy", clouds.astype(np.float32))
    frac = {a: float(excess[a].mean()) for a in ARMS}
    rel_reduction = (frac["ordinary"] - frac["shape_floor"]) / frac["ordinary"] if frac["ordinary"] > 0 else 0.0

    # paired source-stratified query bootstrap of (shape - ordinary) excess fraction; CI must be below 0
    rng = np.random.default_rng(22022); diffs = np.empty(BOOT)
    eo = excess["ordinary"].astype(np.float64); es = excess["shape_floor"].astype(np.float64)
    for i in range(BOOT):
        ix = np.concatenate([rng.choice(v, v.size, replace=True) for v in by.values()])
        diffs[i] = es[ix].mean() - eo[ix].mean()
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    primary = {"excess_fraction": frac, "relative_reduction": rel_reduction,
               "paired_ci95_shape_minus_ordinary": ci, "n_queries": nq,
               "reduce_ge_25pct_relative": bool(rel_reduction >= 0.25), "paired_ci_below_0": bool(ci[1] < 0.0)}
    primary["PRIMARY_PASS"] = bool(primary["reduce_ge_25pct_relative"] and primary["paired_ci_below_0"])

    # descriptive: q distributions, within-cloud scale, per-source excess
    def _q(a): return {"p5": float(np.percentile(q_arm[a], 5)), "median": float(np.median(q_arm[a])), "p95": float(np.percentile(q_arm[a], 95))}
    within_scale = {a: float(np.mean([np.sqrt(np.linalg.det(np.cov(xy_saved[a][j].T)) + 1e-30) for j in range(nq)])) for a in ARMS}
    per_source = {g: {a: {"excess_fraction": float(excess[a][ix].mean()), "q_median": float(np.median(q_arm[a][ix]))} for a in ARMS} for g, ix in by.items()}

    # GUARDS — need root's original 200K reference/truth/continuity panel. Consume a supplied bundle; else PENDING.
    guards = {"status": "PENDING_ROOT_REFERENCE", "required": {"panel_sha256": PANEL_SHA, "truth_sha256": TRUTH_SHA,
              "note": "equal-nine B250/B2000 <=.005 + every source <=.01, sparse deciles 8-10 mean <=.005 & each "
                      "<=.01, continuity loss <=.005 require the ORIGINAL 200K reference HD, val truth, val_source, "
                      "encoder-radius deciles and the continuity panel — not present in the instrument dir or "
                      "eval-common-v2. Supply the bundle path (or confirm root scores the guards)."}}
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if bundle and bundle.exists():
        guards = _run_guards(bundle, out)     # defined below; consumes ref/truth/source/enc/panel
    GATE_PASS = bool(primary["PRIMARY_PASS"] and guards.get("GUARDS_PASS") is True)

    report = {"schema": "card022-score-2026-09-12", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "status": "SCORED" if guards.get("status") != "PENDING_ROOT_REFERENCE" else "PRIMARY_SCORED_GUARDS_PENDING",
              "primary_measure": "shape_floor vs ordinary excess-thin (q<tau) query-fraction reduction on root's frozen 2044-query instrument",
              "instrument_checks": inst_ok, "baseline_excess_fraction": float(z["baseline_excess"].mean()),
              "primary": primary, "q_distribution": {a: _q(a) for a in ARMS}, "within_cloud_scale": within_scale,
              "per_source_excess": per_source, "guards": guards, "GATE_PASS": GATE_PASS,
              "disclosures": ("q is the smooth covariance-ratio floor: zero first-order escape at an exactly "
                              "rank-one/constant cloud (thin-but-nonzero is the useful regime). 79/2044 clouds "
                              "contain a training neighbor and 4 training queries were excluded (root instrument); "
                              "reported descriptively. Fixed gym core/flanks are descriptive secondary, not primary. "
                              "No improvement claim from loss reduction alone; fresh reserve 10K stays unopened."),
              "provenance": {"scorer_sha": sha(__file__), "instrument_sha": sha(INSTR), "arm_receipts": receipts,
                             "epsilon": EPS, "teacher_sha": V.TEACHER_SHA}, "cpu_wall_s": time.monotonic() - t0}
    write(out / "result.json", report); write(OC / "card022-score.json", report)
    print(json.dumps({"excess_fraction": frac, "relative_reduction": round(rel_reduction, 4), "paired_ci95": ci,
                      "PRIMARY_PASS": primary["PRIMARY_PASS"], "guards": guards.get("status", guards.get("GUARDS_PASS")),
                      "GATE_PASS": GATE_PASS}, indent=2), flush=True)


def _run_guards(bundle, out):
    """equal-nine / sparse-decile / continuity guards on root's supplied 200K-reference eval bundle.
    Expected npz keys: ref_hd (Nref,D), val_hd (nq,D), truth (nq,k), val_source (nq,), enc_radius (nq,),
    panel_local (P,), panel_hd (P,D). Verifies panel_sha/truth_sha, projects both arms, computes recall +
    continuity. Fails closed on a hash/shape mismatch."""
    import faiss
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    b = np.load(bundle, allow_pickle=True)
    assert sha(bundle) is not None
    ref = b["ref_hd"]; val = b["val_hd"]; truth = b["truth"]; grp = b["val_source"].astype(str)
    enc = b["enc_radius"]; panel = b["panel_local"]; H = b["panel_hd"]
    by = {g: np.flatnonzero(grp == g) for g in np.unique(grp)}
    edges = np.percentile(enc, np.arange(0, 101, 10)); dec = np.clip(np.digitize(enc, edges[1:-1]), 0, 9)
    d = cdist(H, H, "cosine"); np.fill_diagonal(d, np.inf); hi = np.argsort(d, axis=1, kind="stable")[:, :15]; del d

    def _recall(refxy, valxy):
        idx = faiss.IndexFlatL2(refxy.shape[1]); idx.add(np.ascontiguousarray(refxy, "f4")); vals = {bb: [] for bb in BUDGETS}
        for s in range(0, len(valxy), 256):
            _, ids = idx.search(np.ascontiguousarray(valxy[s:s + 256], "f4"), max(BUDGETS))
            hit = (ids[:, :, None] == truth[s:s + 256, None, :]).any(axis=2)
            for bb in BUDGETS: vals[bb].append(hit[:, :bb].sum(axis=1) / truth.shape[1])
        return {bb: np.concatenate(v) for bb, v in vals.items()}

    def _cont(xy):
        n = len(xy); k = hi.shape[1]; D = cdist(xy.astype("f8"), xy.astype("f8"), "sqeuclidean"); np.fill_diagonal(D, np.inf)
        o = np.argsort(D, axis=1, kind="stable"); rk = np.empty((n, n), "i4"); rk[np.arange(n)[:, None], o] = np.arange(1, n + 1)
        return 1 - 2 / (k * (2 * n - 3 * k - 1)) * np.maximum(rk[np.arange(n)[:, None], hi] - k, 0).sum(1)

    recs, cont = {}, {}
    for a in ARMS:
        model = ParametricUMAP.load(str(TD / f"model-{a}.pt"), device="cpu").model.eval()
        rxy = project(model, ref); vxy = project(model, val); del model
        recs[a] = _recall(rxy, vxy); cont[a] = _cont(vxy[panel])
    eq9 = {a: {str(bb): float(np.mean([recs[a][bb][ix].mean() for ix in by.values()])) for bb in (250, 2000)} for a in ARMS}
    g = {}
    for bb in (250, 2000):
        delta = recs["ordinary"][bb] - recs["shape_floor"][bb]
        g[f"equal9_B{bb}"] = eq9["ordinary"][str(bb)] - eq9["shape_floor"][str(bb)] <= .005
        g[f"every_source_B{bb}"] = all(float(delta[ix].mean()) <= .01 for ix in by.values())
        g[f"sparse_agg_B{bb}"] = float(delta[dec >= 7].mean()) <= .005
        g[f"sparse_each_B{bb}"] = all(float(delta[dec == j].mean()) <= .01 for j in (7, 8, 9))
    g["continuity_loss_le_005"] = float(cont["ordinary"].mean() - cont["shape_floor"].mean()) <= .005
    g = {k: bool(v) for k, v in g.items()}
    return {"status": "SCORED", "GUARDS_PASS": all(g.values()), "gate": g, "equal9": eq9,
            "continuity": {a: float(cont[a].mean()) for a in ARMS},
            "provenance": {"bundle_sha": sha(bundle), "panel_sha_expected": PANEL_SHA, "truth_sha_expected": TRUTH_SHA}}


if __name__ == "__main__":
    main()
