"""Evolution benchmark — trade-surface scorer (plan §2 metrics). GPU (.venv, quick_ffr_v2).

Consumes each arm's per-snapshot coords + manifest and the snapshot truths, emits the preregistered
(churn, quality, latency, cost) trajectory table + one-line verdict per arm. No single-scalar winner —
the deliverable is the trade surface (the product argument rests on churn being visibly catastrophic
for the cuVS baseline, not on winning quality).

Metrics per snapshot k, per arm:
  CHURN   = displacement of the SHARED unchanged points (first n_{k-1} rows) between coords_{k-1} and
            coords_k, radius-normalized; mean + p95; CUMULATIVE = Σ per-step [overseer: vs the PREVIOUS
            snapshot, so reshuffles accumulate]. Arm A frozen = 0 between retrains by construction.
  QUALITY = FFR@0.1% on Sk's own exact truth (quick_ffr_v2); member/unseen split (member = the active
            head's training rows; frozen arm A → first n_0 = T0; cuVS arm B → all, it refits on Sk).
  LATENCY = per-snapshot placement wall (from the arm manifest: transform_wall_s / UMAP wall_s).
  COST    = cumulative GPU wall.
Arm dirs via ARMS_JSON env: {label: {"dir": <coords dir>, "kind": "A"|"B", "member_n0": <int or 0>}}.
Output: /data/latent-basemap/sandbox/evolbench-tradesurface.json
"""
import json, os, sys
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
SUB = Path("/data/latent-basemap/substrates/evolbench")
K = int(os.environ.get("EVOLBENCH_K", "5"))


def _radius(xy):
    return float(np.percentile(np.linalg.norm(xy - xy.mean(0), axis=1), 90))


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()  # repo root on sys.path (parents[2], not [1])
    from knobs_2m import quick_ffr_v2, quick_ffr_v2_split
    arms = json.loads(os.environ.get("ARMS_JSON", "{}"))
    if not arms:
        raise SystemExit("ARMS_JSON required: {label:{dir,kind,member_n0}}")

    out = {"schema": "evolbench-tradesurface-2026-08-31", "arms": {}}
    for label, cfg in arms.items():
        d = Path(cfg["dir"]); man = json.loads((d / "manifest.json").read_text())
        wall_key = "transform_wall_s" if cfg["kind"] == "A" else "wall_s"
        snaps = {s["k"]: s for s in man["snapshots"]}
        rows = []; cum_churn = 0.0; prev = None; prev_n = 0
        for k in range(K + 1):
            xy = np.asarray(np.load(d / f"coords-S{k}.npy"), dtype=np.float32); n = int(xy.shape[0])
            truth = SB / f"evolbench-S{k}" / "edges-k15-fuzzy.npz"
            # churn vs previous snapshot on the shared first-prev_n points
            churn_mean = churn_p95 = 0.0
            if prev is not None:
                disp = np.linalg.norm(xy[:prev_n] - prev, axis=1) / max(_radius(prev), 1e-9)
                churn_mean = float(disp.mean()); churn_p95 = float(np.percentile(disp, 95))
                cum_churn += churn_mean
            # quality FFR (member/unseen where the head has a finite training set)
            mn0 = int(cfg.get("member_n0", 0))
            if mn0 and mn0 < n:
                mask = np.zeros(n, bool); mask[:mn0] = True
                sp = quick_ffr_v2_split(xy, str(truth), n, member_mask=mask)
                q = {"ffr": round(float(sp["overall"]), 4), "ffr_member": sp["member"],
                     "ffr_unseen": sp["unseen"]}
            else:
                q = {"ffr": round(float(quick_ffr_v2(xy, str(truth), n)), 4)}
            lat = snaps.get(k, {}).get(wall_key)
            cost = snaps.get(k, {}).get("cum_gpu_s")
            rows.append({"k": k, "n": n, "churn_mean": round(churn_mean, 5), "churn_p95": round(churn_p95, 5),
                         "cum_churn": round(cum_churn, 5), "quality": q, "latency_s": lat, "cum_cost_s": cost})
            print(f"[{label}] S{k}: churn {churn_mean:.4f} (cum {cum_churn:.4f}) FFR {q['ffr']} "
                  f"lat {lat}s", flush=True)
            prev = xy[:n].copy(); prev_n = n
        final_ffr = rows[-1]["quality"]["ffr"]
        verdict = (f"cum churn {rows[-1]['cum_churn']:.3f}, final FFR {final_ffr}, "
                   f"final latency {rows[-1]['latency_s']}s")
        out["arms"][label] = {"kind": cfg["kind"], "trajectory": rows, "one_line": verdict}

    OUT = SB / "evolbench-tradesurface.json"
    OUT.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("\n=== TRADE SURFACE ===", flush=True)
    for label, a in out["arms"].items():
        print(f"  {label}: {a['one_line']}", flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
