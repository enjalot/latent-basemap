"""Bounded CPU card009 evaluation. Frozen identities, full anchor gauge, unrounded gates.

All confirmation inputs are exactly stored fp16, cast to fp32 for BOTH teacher and
student. Retrieval uses the established L2-normalized encoder inputs and original
k15 truth. Exposed evaluation sets are development data. No training-seed inference.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from _paths import ensure_paths
ensure_paths()
import torch
import faiss
import frame
from basemap.pumap.parametric_umap.core import ParametricUMAP

torch.set_num_threads(4)
faiss.omp_set_num_threads(4)
SB = Path("/data/latent-basemap/sandbox")
OC = SB / "overseer-codex"
TD = SB / "dino-arrival-t0"
SEAL = Path("/data2/monet/eval-common-v2")
SUB = Path("/data/latent-basemap/substrates")
OUT = OC / "card009-scoring"
R0 = 33.6717
BUDGETS = (50, 100, 250, 500, 1000, 2000)
OLD = ("laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m")
ARR = ("synthetic-flux-klein", "synthetic-flux-schnell", "synthetic-z-image")
HEADS = {
    "frozen_t0": (TD / "champion-bs16k/model.pt", TD / "champion-bs16k/coordinates.npy"),
    "anchored": (TD / "updates/model-anchored.pt", TD / "updates/coords-anchored.npy"),
    "historical_out": (TD / "replay-updates/model-out.pt", TD / "replay-updates/coords-out.npy"),
    **{h: (TD / f"card009/model-{h}.pt", TD / f"card009/coords-{h}.npy")
       for h in ("ordinary_out", "stronger_pointwise", "derivative")},
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def project(model, x, normalize=False):
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), 8192):
            xx = np.array(x[start:start + 8192], dtype=np.float32, copy=True)
            if normalize:
                xx /= np.linalg.norm(xx, axis=1, keepdims=True).clip(1e-12)
            chunks.append(model(torch.from_numpy(xx)).numpy())
    result = np.concatenate(chunks)
    assert result.shape == (len(x), 2) and np.isfinite(result).all()
    return result


def recall(rc, vc, truth):
    index = faiss.IndexFlatL2(2)
    index.add(np.ascontiguousarray(rc))
    result = {b: [] for b in BUDGETS}
    for start in range(0, len(vc), 256):
        _, nn = index.search(np.ascontiguousarray(vc[start:start + 256]), 2000)
        hits = (nn[:, :, None] == truth[start:start + 256, None, :]).any(axis=2)
        for b in BUDGETS:
            result[b].append(hits[:, :b].sum(axis=1) / truth.shape[1])
    return {b: np.concatenate(v) for b, v in result.items()}


def stats(d):
    assert len(d) and np.isfinite(d).all()
    return {"mean": float(d.mean()), "p95": float(np.percentile(d, 95)),
            "p99": float(np.percentile(d, 99)),
            "fraction_gt_005": float((d > .05).mean()),
            "fraction_gt_010": float((d > .10).mean())}


def paired(a, b, fn=np.mean, strata=None, seed=9009):
    """Paired, source-stratified bootstrap; retain cohort counts and full precision."""
    assert a.shape == b.shape and len(a)
    rng = np.random.default_rng(seed)
    groups = [np.arange(len(a))] if strata is None else [np.flatnonzero(strata == g) for g in np.unique(strata)]
    draws = np.empty(2000)
    for j in range(len(draws)):
        ix = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        draws[j] = fn(a[ix]) - fn(b[ix])
    return {"difference": float(fn(a) - fn(b)),
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist()}


def main():
    started = time.time()
    OUT.mkdir(exist_ok=True)
    inputs = {p.name: p for p in [SEAL / n for n in
        ("ref_hd.f16.npy", "val_hd.f16.npy", "truth_val.npy", "ref_idx.npy", "val_idx.npy", "val_source.npy")]}
    inputs.update({"confirmation": OC / "card006_confirm_bank.npz",
                   "t0_draw": SUB / "dino-arrival-t0/draw_idx.npy",
                   "final_draw": SUB / "dino-arrival-final/draw_idx.npy",
                   "active": TD / "updates/anchor_active_ids-anchored.npy",
                   "holdout": TD / "updates/anchor_holdout_ids-anchored.npy"})
    provenance = {"scorer_sha256": sha(__file__), "inputs": {k: {"path": str(p), "sha256": sha(p)} for k, p in inputs.items()},
                  "heads": {h: {"model": str(mp), "model_sha256": sha(mp), "coordinates": str(cp), "coordinates_sha256": sha(cp)}
                            for h, (mp, cp) in HEADS.items()}}
    write_json(OUT / "provenance.json", provenance)
    ref = np.load(inputs["ref_hd.f16.npy"], mmap_mode="r")
    val = np.load(inputs["val_hd.f16.npy"], mmap_mode="r")
    truth = np.load(inputs["truth_val.npy"])
    groups = np.load(inputs["val_source.npy"], allow_pickle=True).astype(str)
    ref_ids = np.load(inputs["ref_idx.npy"])
    val_ids = np.load(inputs["val_idx.npy"])
    assert truth.shape == (len(val), 15) and groups.shape == val_ids.shape == (len(val),)
    assert (truth >= 0).all() and (truth < len(ref)).all()
    assert all(len(np.unique(row)) == 15 for row in truth)
    for g in OLD + ARR:
        assert (groups == g).sum() == 1200, (g, (groups == g).sum())
    tdraw = np.load(inputs["t0_draw"])
    fdraw = np.load(inputs["final_draw"])
    common, ti, fi = np.intersect1d(tdraw, fdraw, assume_unique=True, return_indices=True)
    active = np.load(inputs["active"])
    held = np.load(inputs["holdout"])
    am = np.isin(fi, active)
    hm = np.isin(fi, held)
    assert am.sum() == len(active) and hm.sum() == len(held) and not (am & hm).any()
    base = np.asarray(np.load(HEADS["frozen_t0"][1])[ti], np.float64)
    bank = np.load(inputs["confirmation"])
    cx, ci, cs = bank["replay_X"], bank["replay_ids"], bank["source"].astype(str)
    assert cx.dtype == np.float16 and len(np.unique(ci)) == len(ci)
    assert len(cs) == len(cx) == len(ci) and all((cs == g).sum() > 0 for g in OLD)
    assert not np.isin(ci, np.concatenate([fdraw, ref_ids, val_ids])).any()
    arrays = {"val_ids": val_ids, "ref_ids": ref_ids, "val_groups": groups, "truth": truth,
              "confirm_ids": ci, "confirm_sources": cs, "common_graph_ids": common,
              "active_graph_ids": common[am], "holdout_graph_ids": common[hm]}
    heads = {}
    target = None
    for h, (mp, cp) in HEADS.items():
        print(f"Scoring {h}", flush=True)
        model = ParametricUMAP.load(str(mp), device="cpu").model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        rc, vc = project(model, ref, True), project(model, val, True)
        pq = recall(rc, vc, truth)
        cc = project(model, cx).astype(np.float64)
        # Check deployed forward path does not depend on inference chunk size.
        check = project(model, cx[:37]).astype(np.float64)
        assert np.max(np.abs(check - cc[:37])) < 1e-4
        if h == "frozen_t0":
            target = cc.copy()
            updated = base
            R, t = np.eye(2), np.zeros(2)
            info = {"R": R.tolist(), "t": t.tolist(), "rmsd": 0., "learned_scale": 1.}
        else:
            coords = np.load(cp, mmap_mode="r")
            assert coords.shape == (len(fdraw), 2) and np.isfinite(coords).all()
            updated = np.asarray(coords[fi], np.float64)
            _, info = frame.rigid_align(updated[am], base[am])
            R, t = np.array(info["R"]), np.array(info["t"])
        native = np.linalg.norm(cc - target, axis=1) / R0
        aligned = np.linalg.norm(cc @ R.T + t - target, axis=1) / R0
        hn = np.linalg.norm(updated[hm] - base[hm], axis=1) / R0
        ha = np.linalg.norm(updated[hm] @ R.T + t - base[hm], axis=1) / R0
        for b, values in pq.items():
            assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
            arrays[f"{h}_B{b}"] = values
        for name, values in {"confirmation_xy": cc, "disp_native": native, "disp_aligned": aligned,
                             "holdout_native": hn, "holdout_aligned": ha, "R": R, "t": t}.items():
            arrays[f"{h}_{name}"] = values
        heads[h] = {"frame": info, "frame_rows": int(am.sum()), "frame_method": "identity" if h == "frozen_t0" else "full_original_active_saved_coords",
                    "confirmation_native": stats(native), "confirmation_aligned": stats(aligned),
                    "confirmation_by_source": {g: {"native": stats(native[cs == g]), "aligned": stats(aligned[cs == g])} for g in OLD},
                    "graph_holdout_native": stats(hn), "graph_holdout_aligned": stats(ha),
                    "recall": {str(b): {g: float(values[groups == g].mean()) for g in np.unique(groups)} for b, values in pq.items()},
                    "arrival": {str(b): float(values[np.isin(groups, ARR)].mean()) for b, values in pq.items()},
                    "old": {str(b): float(values[np.isin(groups, OLD)].mean()) for b, values in pq.items()}}
        # Durable raw results before any downstream gate/report failure.
        np.savez(OUT / f"{h}.npz", **{k: v for k, v in arrays.items() if k.startswith(h + "_")})
        write_json(OUT / f"{h}.json", heads[h])
        print(json.dumps({"head": h, "arrival250": heads[h]["arrival"]["250"], "movement": stats(aligned)}), flush=True)
        del model, rc, vc
    arrays["confirmation_teacher_xy"] = target
    np.savez(OUT / "per-query.npz", **arrays)
    contrasts = {}
    pairs = [(h, "ordinary_out") for h in ("stronger_pointwise", "derivative", "historical_out")]
    pairs += [("derivative", "stronger_pointwise")]
    arrmask = np.isin(groups, ARR)
    for a, b in pairs:
        da, db = arrays[f"{a}_disp_aligned"], arrays[f"{b}_disp_aligned"]
        contrasts[f"{a}-minus-{b}"] = {
            "p99": paired(da, db, lambda x: np.percentile(x, 99), cs),
            "fraction_gt_005": paired((da > .05).astype(float), (db > .05).astype(float), strata=cs),
            "arrival": {str(bud): paired(arrays[f"{a}_B{bud}"][arrmask], arrays[f"{b}_B{bud}"][arrmask], strata=groups[arrmask]) for bud in (250, 2000)}}
    guards, promotions, deployment = {}, {}, {}
    for h in ("ordinary_out", "stronger_pointwise", "derivative"):
        losses = {str(b): {g: heads["frozen_t0"]["recall"][str(b)][g] - heads[h]["recall"][str(b)][g] for g in OLD} for b in (250, 2000)}
        old_ok = all(np.mean(list(ls.values())) <= .005 and max(ls.values()) <= .01 for ls in losses.values())
        arr_ok = all(heads[h]["arrival"][str(b)] >= heads["ordinary_out"]["arrival"][str(b)] - .005 for b in (250, 2000))
        guards[h] = {"old_losses_vs_frozen": losses, "old_retention_both_budgets": bool(old_ok), "arrival_vs_ordinary_both_budgets": bool(arr_ok)}
        fresh_gain = paired(arrays[f"{h}_B250"][arrmask], arrays["frozen_t0_B250"][arrmask], strata=groups[arrmask])
        anch_ok = all(heads[h]["arrival"][str(b)] >= heads["anchored"]["arrival"][str(b)] - .005 for b in (250, 2000))
        dc = {"confirmation_mean_le_001": heads[h]["confirmation_aligned"]["mean"] <= .01,
              "confirmation_p99_le_005": heads[h]["confirmation_aligned"]["p99"] <= .05,
              "old_retention": bool(old_ok), "arrival_within_005_of_anchored": bool(anch_ok),
              "arrival_gain_ci_positive": fresh_gain["ci95"][0] > 0}
        deployment[h] = {"checks": dc, "PASS": all(dc.values()), "arrival_B250_vs_frozen": fresh_gain,
                         "note": "Development eligibility only; untouched confirmation required before deployment."}
        if h != "ordinary_out":
            comparison = contrasts[f"{h}-minus-ordinary_out"]
            ratio = heads[h]["confirmation_aligned"]["p99"] / heads["ordinary_out"]["confirmation_aligned"]["p99"]
            gc = {"p99_reduction_ge_30pct": ratio <= .70,
                  "paired_p99_ci_below_zero": comparison["p99"]["ci95"][1] < 0,
                  "fraction_gt_005_not_increased": heads[h]["confirmation_aligned"]["fraction_gt_005"] <= heads["ordinary_out"]["confirmation_aligned"]["fraction_gt_005"],
                  "old_quality_preserved": bool(old_ok), "arrival_preserved": bool(arr_ok)}
            promotions[h] = {"checks": gc, "p99_ratio": ratio, "PROMOTE": all(gc.values()),
                             "role": "primary" if h == "derivative" else "matched pointwise control, diagnostic screen"}
    pool_source_path = Path("/data2/monet/pool-20m/source.npy")
    source = np.load(pool_source_path, allow_pickle=True).astype(str)
    names, counts = np.unique(source, return_counts=True)
    weights = {str(n): int(c) for n, c in zip(names, counts) if n in OLD + ARR}
    assert set(weights) == set(OLD + ARR)
    natural = {h: {str(b): sum(weights[g] * heads[h]["recall"][str(b)][g] for g in weights) / sum(weights.values()) for b in BUDGETS} for h in heads}
    report = {"schema": "card009-codex-score-v1", "status": "COMPLETE", "R0": R0, "heads": heads,
              "contrasts": contrasts, "guards": guards, "promotion": promotions, "deployment": deployment,
              "natural_pool_weights": weights, "natural_query_weighted": natural, "pool_source_sha256": sha(pool_source_path),
              "quality_definition": "Recall of original encoder k15 truth within map B-neighbor inspection budget; reference unchanged.",
              "uncertainty": "2000 paired source-stratified bootstrap draws, seed9009. Query uncertainty in one trajectory, not training-seed reproducibility.",
              "gates": "Unrounded float64 arrays/metrics. Original qualitative guards operationalized in OPERATING.md before reading quality. Deployment gates are separate from primary promotion.",
              "caveats": ["All evaluated cohorts are development data, not an untouched deployment seal.",
                          "Historical OUT is context; newly trained ordinary OUT is the matched control.",
                          "Rigid transforms fitted only to full original active anchors; no fitted scale. Deployment would need the same transform.",
                          "Graph anchor holdout and wholly unseen confirmation are distinct populations."],
              "training_audit": json.loads((OC / "card009-completion-audit.json").read_text()),
              "provenance": str(OUT / "provenance.json"), "per_query": str(OUT / "per-query.npz"), "cpu_wall_seconds": time.time() - started}
    write_json(OUT / "result.json", report)
    write_json(OC / "card009-result.json", report)
    print(json.dumps({"promotion": promotions, "deployment": deployment, "contrasts": contrasts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
