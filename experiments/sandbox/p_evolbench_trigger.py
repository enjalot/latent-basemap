"""Evolution benchmark — §4 drift trigger + its preregistered correctness check. GPU (.venv/torch).

Per-tranche NOVELTY AGGREGATE = mean over the tranche's rows of the 1-NN COSINE DISTANCE to a fixed T0
reference sample (input space). In-distribution tranches (T1/T2/T4/T5, same 40/25/25/10 mix) sit near
T0 → low novelty; the OOD reddit tranche (T3) sits far → high novelty. Aggregate, not per-point (the
plan's validated calibration level). Band CALIBRATED at T0+noise: a HELD-OUT T0 slice's novelty
(mean+3σ over sub-batches) is the no-drift baseline; a tranche FIRES if its aggregate exceeds it.

PREREGISTERED CORRECTNESS CHECK: T1,T2,T4,T5 must NOT fire; T3 MUST. That is itself a result — it
validates the drift rule the benchmark's arm A retrains on.

Output: /data/latent-basemap/sandbox/evolbench-trigger.json
"""
import json, os
from pathlib import Path
import numpy as np

SUB = Path("/data/latent-basemap/substrates/evolbench")
OUT = Path("/data/latent-basemap/sandbox/evolbench-trigger.json")
REF_N = int(os.environ.get("TRIGGER_REF_N", "500000"))
TRANCHES = ("T1", "T2", "T3", "T4", "T5")


def _norm_t(x, torch):
    x = torch.from_numpy(np.ascontiguousarray(x, np.float32)).cuda()
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _mean_1nn_cosdist(Q, ref, torch, qchunk=16384, dchunk=200000):
    """mean over rows of Q of (1 - max cosine-sim to ref). Both L2-normalized on GPU."""
    tot = 0.0; nq = int(Q.shape[0])
    for qs in range(0, nq, qchunk):
        q = Q[qs:qs+qchunk]
        best = torch.full((q.shape[0],), -2.0, device="cuda")
        for ds in range(0, int(ref.shape[0]), dchunk):
            best = torch.maximum(best, (q @ ref[ds:ds+dchunk].T).max(dim=1).values)
        tot += float((1.0 - best).sum().cpu())
    return tot / nq


def main():
    import torch
    T0 = np.load(SUB / "T0" / "substrate.f32.npy", mmap_mode="r")
    rng = np.random.default_rng(42)
    idx = rng.permutation(int(T0.shape[0]))
    ref = _norm_t(np.asarray(T0[np.sort(idx[:REF_N])], np.float32), torch)               # reference
    cal = np.sort(idx[REF_N:REF_N + 800_000])                                            # held-out T0 slice
    cal_x = _norm_t(np.asarray(T0[cal], np.float32), torch)
    # baseline band: mean+3σ over sub-batches of the held-out T0 slice's novelty (no-drift noise)
    sub = 100_000
    batch_nov = [_mean_1nn_cosdist(cal_x[i:i+sub], ref, torch) for i in range(0, cal_x.shape[0], sub)]
    base_mean = float(np.mean(batch_nov)); base_std = float(np.std(batch_nov))
    band = base_mean + 3 * base_std
    print(f"T0 baseline novelty {base_mean:.4f} ± {base_std:.4f} → band {band:.4f}", flush=True)

    res = {}
    for t in TRANCHES:
        x = _norm_t(np.asarray(np.load(SUB / t / "substrate.f32.npy", mmap_mode="r"), np.float32), torch)
        nov = _mean_1nn_cosdist(x, ref, torch)
        fires = bool(nov > band)
        res[t] = {"novelty": round(nov, 4), "fires": fires, "over_band": round(nov - band, 4)}
        print(f"{t}: novelty {nov:.4f} {'FIRES' if fires else 'quiet'} (band {band:.4f})", flush=True)
        del x
    check = (not res["T1"]["fires"] and not res["T2"]["fires"] and not res["T4"]["fires"]
             and not res["T5"]["fires"] and res["T3"]["fires"])
    out = {"schema": "evolbench-trigger-2026-08-31", "ref_n": REF_N,
           "baseline_mean": round(base_mean, 4), "baseline_std": round(base_std, 4), "band": round(band, 4),
           "tranches": res,
           "preregistered_check_passes": bool(check),
           "check": "T1/T2/T4/T5 quiet AND T3 fires (OOD injection is the positive control)"}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n=== PREREGISTERED CHECK {'PASS' if check else 'FAIL'} ===\nwrote {OUT}", flush=True)
    return 0 if check else 6


if __name__ == "__main__":
    raise SystemExit(main())
