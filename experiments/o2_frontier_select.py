"""O2 phase 2 — sparse-hold frontier selection (content-bound CPU/GPU decision).

For each hold weight w, compare the sparse-hold 4M map to the unanchored control:
  - old-point STABILITY: low-D kNN overlap of a landmark sample between the
    sparse map and the control (mean Jaccard@k). Select gate: >= 0.5.
  - old-point DRIFT: mean landmark displacement between the sparse map and the
    teacher (control) coords, normalized by control RMS radius (diagnostic).
  - QUALITY: panel KPIs (ffr / purity_k1024 / density) of the sparse map must be
    >= 90% of the control's (from the pre-scored frontier + control panels).

Selects a weight only if overlap >= 0.5 AND every KPI ratio >= 0.90.
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.panel_v2 import load_coords, PanelV2Config, cross_knn, _ids_hash


def _lowd_knn(coords, query_idx, k, cfg):
    Q = np.asarray(coords[query_idx], dtype=np.float32)
    return cross_knn(Q, np.asarray(coords, dtype=np.float32), k, cfg, hi_dim=False, q_tile=4096)


def _jaccard_overlap(nb_a, nb_b):
    ov = []
    for a, b in zip(nb_a, nb_b):
        sa, sb = set(a.tolist()), set(b.tolist())
        ov.append(len(sa & sb) / max(1, len(sa | sb)))
    return float(np.mean(ov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="control_s42 run dir")
    ap.add_argument("--anchors", required=True, help="anchors npz")
    ap.add_argument("--frontier-scores", required=True, help="complete_4m_frontier_v22.json")
    ap.add_argument("--control-scores", required=True, help="complete_4m_controls_v22.json")
    ap.add_argument("--weights", default="2,10,50")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--n-sample", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PanelV2Config(corpus_chunk=500_000)
    Zc, zc = load_coords(os.path.join(args.control, "coords.parquet"))
    if zc is not None and not np.array_equal(np.asarray(zc), np.arange(len(Zc))):
        Zc = Zc[np.argsort(np.asarray(zc))]
    anch = np.load(args.anchors)
    ids = np.asarray(anch["anchor_ids"], np.int64)
    rng = np.random.RandomState(0)
    samp = np.sort(rng.choice(ids, size=min(args.n_sample, len(ids)), replace=False))
    nb_control = _lowd_knn(Zc, samp, args.k, cfg)
    rms_c = float(np.sqrt((Zc ** 2).sum(1).mean()))

    fscores = json.load(open(args.frontier_scores)).get("runs", {})
    cscores = json.load(open(args.control_scores)).get("runs", {})
    # control KPI = mean over the 3 control seeds
    def _mean(runs, metric):
        vals = [r[metric] for r in runs.values() if r.get(metric) is not None]
        return float(np.mean(vals)) if vals else None
    ctrl_kpi = {m: _mean(cscores, m) for m in ("ffr", "purity_k1024", "density")}

    results = {}
    selected = None
    for w in [int(x) for x in args.weights.split(",")]:
        rd = os.path.join(os.path.dirname(args.control), f"sparse_w{w}_s42")
        Zf, zf = load_coords(os.path.join(rd, "coords.parquet"))
        if zf is not None and not np.array_equal(np.asarray(zf), np.arange(len(Zf))):
            Zf = Zf[np.argsort(np.asarray(zf))]
        nb_f = _lowd_knn(Zf, samp, args.k, cfg)
        overlap = _jaccard_overlap(nb_f, nb_control)
        drift = float(np.linalg.norm(np.asarray(Zf[ids]) - np.asarray(Zc[ids]), axis=1).mean() / rms_c)
        fr = fscores.get(f"sparse_w{w}") or {}
        kpi = {m: fr.get(m) for m in ("ffr", "purity_k1024", "density")}
        ratios = {m: (kpi[m] / ctrl_kpi[m] if kpi.get(m) and ctrl_kpi.get(m) else None)
                  for m in kpi}
        kpi_ok = all((r is not None and r >= 0.90) for r in ratios.values())
        passed = bool(overlap >= 0.5 and kpi_ok)
        results[f"w{w}"] = {"landmark_overlap_jaccard": round(overlap, 4),
                            "old_point_drift_norm": round(drift, 4),
                            "kpi": kpi, "kpi_ratio_vs_control": {m: (round(r, 4) if r else None)
                                                                for m, r in ratios.items()},
                            "overlap_ge_0.5": bool(overlap >= 0.5), "kpi_ge_90pct": kpi_ok,
                            "passed": passed}
        if passed and selected is None:
            selected = w
    out = {"gate": "o2_sparse_hold_frontier", "control": os.path.abspath(args.control),
           "anchor_id_hash": _ids_hash(ids), "control_kpi_mean": ctrl_kpi,
           "n_landmark_sample": int(len(samp)), "k": args.k,
           "results": results, "selected_weight": selected,
           "rule": "select w iff landmark low-D kNN overlap >= 0.5 AND every panel KPI "
                   ">= 90% of the unanchored control. drift is diagnostic.",
           "surrogate_note": "landmarks = deterministic random 4M subset held to the seed-42 "
                             "control's coords (no real 2M->4M provenance). FLAGGED for review."}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[o2_frontier] selected_weight={selected} | " +
          " ".join(f"w{w}:ov={results[f'w{w}']['landmark_overlap_jaccard']}"
                   f",pass={results[f'w{w}']['passed']}" for w in [int(x) for x in args.weights.split(',')]),
          flush=True)


if __name__ == "__main__":
    main()
