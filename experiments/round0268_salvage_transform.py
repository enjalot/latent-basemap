#!/usr/bin/env python3
"""R0268 seed42 SALVAGE: the post-training transform + tripwire from the SAVED model.

seed42 trained the FULL dose (23.83h, model.pt saved + immutable + ParametricUMAP.load
verified) but the node's post-training 100M transform was killed by an EXTERNAL SIGINT
(returncode -2, KeyboardInterrupt during model.transform; no cgroup limit / no OOM) — most
likely a ~24h background-task lifetime reap. No training is lost.

This reproduces exactly what the node would have computed after training: the full 100M
coordinates (via the same node helper _transform_100m_in_chunks) and the seed-1 tripwire
preview (map_collapse + _map_fog on the full 100M coordinates). It writes coordinates to a
SALVAGE dir (never touches the immutable artifact) and prints the tripwire numbers so the
seed-1 checkpoint can be reported. Values here ARE evidence-grade for the tripwire (same
model, same substrate, same instruments) — but the runner-valid completed cell (receipt +
done-marker) is produced separately by the resume path.

Runs under the SAME conditions as the successful dry-run (fp32 substrate memmap only, no int8
resident), so the memory-pressure hypothesis for the SIGINT cannot recur here.
"""
import os, sys, time, json, resource

os.environ.setdefault("HF_HOME", "/data/hf")
import numpy as np

MODEL = ("/data/latent-basemap/runs/round-0268/queue/artifacts/"
         "minilm-mixed-100000k-fneg-x2-md000-hostint8-seed42-r0268-v1/model.pt")
SUBSTRATE = ("/data/latent-basemap/runs/round-0238/queue/artifacts/"
             "minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.f32.npy")
SALVAGE_DIR = "/data/latent-basemap/runs/round-0268/salvage/seed42"
OUT_JSON = os.path.join(SALVAGE_DIR, "seed42-tripwire.json")

# tripwire thresholds (plan §2, N-invariant / sealed R0265 family)
COLLAPSE_BACKSTOP = 0.8129   # per-seed hard floor
FOG_CEILING = 0.41207
FOG_ESCALATION = 0.3919      # ceiling - 1*sigma_fam,fog


def main():
    os.makedirs(SALVAGE_DIR, exist_ok=True)
    import torch
    from experiments.round0268_nodes import (
        ROWS, DIMENSION, _transform_100m_in_chunks,
    )
    from experiments.round0265_nodes import map_collapse, _map_fog, FOG_BINS
    from basemap.pumap.parametric_umap import ParametricUMAP

    t0 = time.monotonic()
    source = np.load(SUBSTRATE, mmap_mode="r", allow_pickle=False)
    assert source.shape == (ROWS, DIMENSION) and source.dtype == np.float32, source.shape

    model = ParametricUMAP.load(MODEL, device="cuda")
    print(f"[salvage] model loaded; transforming {ROWS:,} rows", flush=True)

    def poll(msg):
        if not hasattr(poll, "n"):
            poll.n = 0
        poll.n += 1
        if poll.n % 25 == 0:
            print(f"  {msg}", flush=True)

    coordinates = _transform_100m_in_chunks(model, source, poll)
    assert coordinates.shape == (ROWS, 2) and np.isfinite(coordinates).all(), coordinates.shape
    t_transform = time.monotonic() - t0

    # the seed-1 tripwire FIRST (the critical output) — map_collapse + _map_fog on the FULL
    # 100M coords (node's instruments), so a downstream save bug can never lose it.
    collapse = map_collapse(coordinates)
    fog = _map_fog(coordinates, bins=FOG_BINS)
    collapse_val = float(collapse["r10_over_radius_times_sqrt_n"])
    fog_val = float(fog["fog"])

    backstop_ok = collapse_val >= COLLAPSE_BACKSTOP
    fog_ceiling_ok = fog_val <= FOG_CEILING
    fog_escalation = fog_val >= FOG_ESCALATION  # True => near-ceiling, escalate
    tripwire_pass = backstop_ok and fog_ceiling_ok and not fog_escalation

    coord_path = os.path.join(SALVAGE_DIR, "coordinates.npy")
    report = {
        "role": "R0268 seed42 salvage — post-training transform + seed-1 tripwire from saved model",
        "model": MODEL,
        "rows": ROWS,
        "coordinates_saved": coord_path,
        "coordinates_ordered_sha256_note": "computed by the resume path; here descriptive",
        "transform_wall_s": round(t_transform, 1),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2), 2),
        "tripwire": {
            "collapse": collapse_val,
            "collapse_backstop": COLLAPSE_BACKSTOP,
            "collapse_backstop_ok": bool(backstop_ok),
            "fog": fog_val,
            "fog_ceiling": FOG_CEILING,
            "fog_ceiling_ok": bool(fog_ceiling_ok),
            "fog_escalation_threshold": FOG_ESCALATION,
            "fog_escalation_triggered": bool(fog_escalation),
            "resolution_levels": int(fog["resolution_levels"]),
            "degenerate": bool(fog["degenerate"]),
            "PASS": bool(tripwire_pass),
        },
        "collapse_detail": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in collapse.items()},
    }
    json.dump(report, open(OUT_JSON, "w"), indent=2, default=str)
    print(f"\n[salvage] transform {report['transform_wall_s']}s peak_rss {report['peak_rss_gib']}GiB",
          flush=True)
    print(f"[salvage] TRIPWIRE collapse={collapse_val:.4f} (backstop {COLLAPSE_BACKSTOP}, "
          f"ok={backstop_ok}) | fog={fog_val:.4f} (ceiling {FOG_CEILING} ok={fog_ceiling_ok}, "
          f"escalation>={FOG_ESCALATION} triggered={fog_escalation})", flush=True)
    print(f"[salvage] SEED-1 TRIPWIRE {'PASS' if tripwire_pass else 'FAIL/ESCALATE'}", flush=True)
    print(f"[salvage] report -> {OUT_JSON}", flush=True)

    # best-effort coordinate save LAST (tripwire already persisted above). np.save writes
    # exactly `coord_path` since it ends in .npy; write to a .npy-suffixed tmp then rename.
    try:
        tmp = coord_path + ".partial.npy"
        np.save(tmp, coordinates)
        os.replace(tmp, coord_path)
        print(f"[salvage] coordinates -> {coord_path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[salvage] WARN coordinate save failed (tripwire already saved): {exc}", flush=True)


if __name__ == "__main__":
    main()
