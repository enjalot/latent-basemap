"""Card018 CPU scorer (per card018-scale-2m.md + root review). NEW file — adapts root's score_scale_pilot.py
logic WITHOUT modifying it or any existing scorer/runtime. Scores the two matched fresh 3D heads (ordinary3d,
actual3d) against the FROZEN Card013 instrument (original eval-common-v2 250K reference, nine query cohorts,
full-D encoder-k15 truth, fixed encoder-radius deciles, 1800-point continuity panel). One primary treatment:
actual3d vs ordinary3d.

Arm completion is verified through the ONE canonical strict validator (card018_validate.strict_validate_arm)
with an independent data re-hash (recompute_substrate=True) — no bespoke inline validation. The Card013 2D
arms are NOT used as a comparator (they are a binary-fixed15 recipe with different optimizer/dose/residency;
including them would mislead).

Development gate (ALL): actual vs ordinary radius-corr gain >= .05 with positive paired 95% CI; continuity
loss <= .005; equal-nine B250/B2000 loss <= .005; each source loss <= .01; original sparse deciles 8-10
aggregate loss <= .005 AND each decile loss <= .01 at both budgets. Severe-join is DESCRIPTIVE ONLY.
No GPU, no fresh confirmation. Usage: score_card018.py
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "4"
import json, time, hashlib, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card018_validate as V
import torch, faiss
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from score_card011 import severe_false_joins
from basemap.pumap.parametric_umap.core import ParametricUMAP

ROOT = Path(__file__).resolve().parents[2]
SB = V.SB; OC = V.OC; SEAL = Path("/data2/monet/eval-common-v2"); C13 = OC / "card013-scoring"
DATA = V.DATA; TD = SB / "card018-train"
ARMS = ["ordinary3d", "actual3d"]; DIM = 3; N = V.N
BUDGETS = [50, 100, 250, 500, 1000, 2000]; BOOT = 2000
torch.set_num_threads(4); faiss.omp_set_num_threads(4)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()
def write(p, obj):
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n"); t.replace(p)


def project(model, data, dim):
    out = []
    with torch.inference_mode():
        for s in range(0, len(data), 8192):
            x = np.array(data[s:s + 8192], np.float32, copy=True); x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
            out.append(model(torch.from_numpy(x)).numpy())
    xy = np.concatenate(out); assert xy.shape == (len(data), dim) and np.isfinite(xy).all(); return xy


def recall(ref, val, truth):
    idx = faiss.IndexFlatL2(ref.shape[1]); idx.add(np.ascontiguousarray(ref, "f4")); vals = {b: [] for b in BUDGETS}
    for s in range(0, len(val), 256):
        _, ids = idx.search(np.ascontiguousarray(val[s:s + 256], "f4"), max(BUDGETS))
        hits = (ids[:, :, None] == truth[s:s + 256, None, :]).any(axis=2)
        for b in BUDGETS: vals[b].append(hits[:, :b].sum(axis=1) / truth.shape[1])
    return {b: np.concatenate(v) for b, v in vals.items()}


def continuity(xy, hi):
    n = len(xy); k = hi.shape[1]; d = cdist(xy.astype("f8"), xy.astype("f8"), "sqeuclidean"); np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1, kind="stable"); rank = np.empty((n, n), "i4"); rank[np.arange(n)[:, None], order] = np.arange(1, n + 1)
    return 1 - 2 / (k * (2 * n - 3 * k - 1)) * np.maximum(rank[np.arange(n)[:, None], hi] - k, 0).sum(1)


def main():
    t0 = time.monotonic(); out = OC / "card018-scoring"; out.mkdir(exist_ok=True)
    # canonical strict validation of BOTH arms (with independent data re-hash) — one source of truth
    receipts = {a: V.strict_validate_arm(a, ROOT, recompute_substrate=(a == "actual3d")) for a in ARMS}
    write(out / "prescoring-validation.json", {"PASS": True, "receipts": receipts})

    # frozen Card013 instrument (identical to score_scale_pilot): 250K original reference + fixed panels
    z = np.load(C13 / "per-query.npz", allow_pickle=True); panel_saved = np.load(C13 / "closeout-persist.npz")
    audit = np.load(C13 / "independent-panel-audit.npz")
    ref = np.load(SEAL / "ref_hd.f16.npy", mmap_mode="r"); val = np.load(SEAL / "val_hd.f16.npy", mmap_mode="r")
    truth = np.load(SEAL / "truth_val.npy"); ids = np.load(SEAL / "val_idx.npy"); rid = np.load(SEAL / "ref_idx.npy")
    groups = np.load(SEAL / "val_source.npy", allow_pickle=True).astype(str)
    enc = z["enc_radius"]; panel = panel_saved["panel_local"]; H = audit["panel_hd"]
    inst = {"old_query_identity": np.array_equal(ids, z["val_ids"]) and np.array_equal(groups, z["val_source"].astype(str)) and np.array_equal(truth, z["truth"]),
            "original_reference": np.array_equal(rid, panel_saved["ref_ids"]) and len(ref) == 250000,
            "original_panel": np.array_equal(ids[panel], audit["panel_ids"]),
            "heldout_ids": not np.isin(ids, np.r_[rid, np.load(DATA / "draw_ids.npy")]).any()}
    old_prov = json.loads((C13 / "provenance.json").read_text())
    for name, digest in old_prov["instrument"].items(): inst["old_instrument_" + name] = sha(SEAL / name) == digest
    write(out / "instrument-validation.json", {"PASS": all(inst.values()), "checks": inst})
    assert all(inst.values()), {k: v for k, v in inst.items() if not v}

    by = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    d = cdist(H, H, "cosine"); np.fill_diagonal(d, np.inf); hi = np.argsort(d, axis=1, kind="stable")[:, :15]; del d
    edges = np.percentile(enc, np.arange(0, 101, 10)); dec = np.clip(np.digitize(enc, edges[1:-1]), 0, 9)
    maps, recs, cont, severe = {}, {}, {}, {}
    arrays = {"ref_ids": rid, "val_ids": ids, "val_source": groups, "truth": truth, "enc_radius": enc,
              "decile_zero_based": dec, "panel_local": panel, "panel_val_ids": ids[panel], "panel_hd": H, "panel_encoder15_local": hi}
    for a in ARMS:
        print(f"Projecting card018 {a} (3D)", flush=True)
        model = ParametricUMAP.load(str(TD / f"model-{a}.pt"), device="cpu").model.eval()
        rc = project(model, ref, DIM); vc = project(model, val, DIM); del model
        np.save(out / f"{a}-ref-xy.npy", rc); np.save(out / f"{a}-val-xy.npy", vc)
        recs[a] = recall(rc, vc, truth)
        dd, _ = cKDTree(rc.astype("f8")).query(vc.astype("f8"), k=15, workers=4); maps[a] = np.sqrt((dd ** 2).mean(1))
        cont[a] = continuity(vc[panel], hi)
        sev = severe_false_joins(H, vc[panel].astype("f8")); severe[a] = sev["severe_frac_of_map15"]
        write(out / f"{a}-severe.json", sev); arrays[a + "_panel_xy"] = vc[panel]
    corr = {a: float(spearmanr(enc, maps[a]).statistic) for a in ARMS}
    equal9 = {a: {str(b): float(np.mean([recs[a][b][ix].mean() for ix in by.values()])) for b in BUDGETS} for a in ARMS}
    per_source = {g: {a: {"radius_corr": float(spearmanr(enc[ix], maps[a][ix]).statistic),
                          **{f"B{b}": float(recs[a][b][ix].mean()) for b in BUDGETS}} for a in ARMS} for g, ix in by.items()}
    per_decile = {str(j + 1): {"n": int((dec == j).sum()),
                  "heads": {a: {f"B{b}": float(recs[a][b][dec == j].mean()) for b in BUDGETS} for a in ARMS}} for j in range(10)}
    for a in ARMS:
        arrays[a + "_map_radius"] = maps[a]; arrays[a + "_continuity_per_query"] = cont[a]
        for b in BUDGETS: arrays[f"{a}_B{b}"] = recs[a][b]
    np.savez(out / "per-query.npz", **arrays)

    actual, ordinary = "actual3d", "ordinary3d"
    rng = np.random.default_rng(18018); boot = {"corr": [], "B250": [], "B2000": []}
    for _ in range(BOOT):
        inds = [rng.choice(ix, len(ix), replace=True) for ix in by.values()]; ix = np.concatenate(inds)
        boot["corr"].append(spearmanr(enc[ix], maps[actual][ix]).statistic - spearmanr(enc[ix], maps[ordinary][ix]).statistic)
        for b in (250, 2000):
            boot[f"B{b}"].append(np.mean([recs[actual][b][j].mean() for j in inds]) - np.mean([recs[ordinary][b][j].mean() for j in inds]))
    ci = {m: np.percentile(v, [2.5, 97.5]).tolist() for m, v in boot.items()}
    pby = [np.flatnonzero(groups[panel] == g) for g in by]; cb = []
    for _ in range(BOOT):
        ix = np.concatenate([rng.choice(i, len(i), replace=True) for i in pby]); cb.append(float((cont[actual] - cont[ordinary])[ix].mean()))
    ci["continuity"] = np.percentile(cb, [2.5, 97.5]).tolist()

    gain = corr[actual] - corr[ordinary]; sparse, guard = {}, {}
    for b in (250, 2000):
        delta = recs[ordinary][b] - recs[actual][b]
        sparse[str(b)] = {"aggregate_loss": float(delta[dec >= 7].mean()),
                          "per_decile_loss": {str(j + 1): float(delta[dec == j].mean()) for j in (7, 8, 9)}}
        guard[f"equal9_B{b}"] = equal9[ordinary][str(b)] - equal9[actual][str(b)] <= .005
        guard[f"every_source_B{b}"] = all(float(delta[ix].mean()) <= .01 for ix in by.values())
        guard[f"sparse_aggregate_B{b}"] = sparse[str(b)]["aggregate_loss"] <= .005
        guard[f"every_sparse_decile_B{b}"] = all(v <= .01 for v in sparse[str(b)]["per_decile_loss"].values())
    guard.update(corr_gain_ge_005=gain >= .05, corr_vs_ordinary_ci_positive=ci["corr"][0] > 0,
                 continuity_loss_le_005=float(cont[ordinary].mean() - cont[actual].mean()) <= .005)
    guard = {k: bool(v) for k, v in guard.items()}
    diff = {"corr": corr[actual] - corr[ordinary], "continuity": float((cont[actual] - cont[ordinary]).mean()),
            **{f"B{b}": equal9[actual][str(b)] - equal9[ordinary][str(b)] for b in (250, 2000)}}
    report = {"schema": "card018-score-2026-09-12", "at": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "SCORED",
              "primary": "actual3d vs ordinary3d — 3D local-scale kernel d2/(r_i*r_j) at 2M training rows",
              "n_train": N, "radius_spearman": corr, "corr_gain_actual_vs_ordinary": gain,
              "equal9_recall": equal9, "per_source": per_source, "per_decile": per_decile,
              "continuity": {a: float(cont[a].mean()) for a in ARMS}, "continuity_loss_actual": float(cont[ordinary].mean() - cont[actual].mean()),
              "severe_frac": severe, "severe_frac_increase_actual": severe[actual] - severe[ordinary],
              "severe_note": "Severe-join counts are insensitive and DESCRIPTIVE ONLY; not a safety/deployment certificate and not a gate input.",
              "sparse_guard_detail": sparse, "differences": diff, "paired_ci95": ci,
              "gate": guard, "GATE_PASS": all(guard.values()),
              "uncertainty": f"{BOOT} source-stratified paired query bootstraps, seed 18018. Continuity conditional on fixed panel/ranks. Single training seed; development data. One primary treatment (no shuffle arm at 2M; the 300K shuffle already established the location-specific signal).",
              "limits": "Scale transfer of a selected development recipe, not independent method replication or a band-removal claim. Historical Card013 gate remains failed; the .05 threshold applies to development. A pass is a confirmation candidate, not deployment; fresh reserve stays closed until Codex selects. No Card013 2D comparator is used (binary-fixed15, different optimizer/dose/residency ⇒ not an equal-recipe dimensionality control).",
              "provenance": {"scorer_sha": sha(__file__), "worktree_root": str(ROOT), "arm_receipts": receipts,
                             "instrument": {n: sha(SEAL / n) for n in ["ref_hd.f16.npy", "val_hd.f16.npy", "truth_val.npy", "ref_idx.npy", "val_idx.npy", "val_source.npy"]},
                             "card013_arrays_sha": sha(C13 / "per-query.npz"), "panel_audit_sha": sha(C13 / "independent-panel-audit.npz"),
                             "data_manifest": str(OC / "card018-data-manifest.json")},
              "cpu_wall_s": time.monotonic() - t0}
    write(out / "result.json", report); write(OC / "card018-score.json", report)
    print(json.dumps({"radius_spearman": corr, "gain": round(gain, 4), "corr_ci": ci["corr"],
                      "continuity": report["continuity"], "gate": guard, "GATE_PASS": report["GATE_PASS"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
