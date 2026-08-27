#!/usr/bin/env python3
"""Re-score EVERY saved map's FFR with the corrected instrument (quick_ffr_v2).

External review 2026-08-27, P1 #7: v1 ``quick_ffr`` built high-D "truth" as an
ID-ordered slice of the fuzzy graph and let the query sit in its own retrieval
budget.  ``knobs_2m.quick_ffr_v2`` fixes both (exact knn_indices.npy where
present, else top-k BY FUZZY WEIGHT; query excluded from both sides).

This walks every ``coordinates.npy`` under /data/latent-basemap/sandbox, resolves
that map's truth graph, computes ``quick_ffr_v2`` over the SAVED coordinates (NO
retraining, CPU only — KD-tree over saved 2D coords), and writes it back into the
map dir's ``summary.json`` as ``quick_ffr_v2`` (+ ``quick_ffr_v2_truth_mode``),
leaving the archived v1 ``quick_ffr_at_0.1pct`` untouched.

CPU only.  Bounded: maps are processed sequentially; each KD-tree query uses a
small fixed thread pool, so the running GPU pipeline's CPU side is not starved.

Usage:
    rescore_ffr_v2.py            # dry-run: resolve + report, write nothing
    rescore_ffr_v2.py --run      # compute + write quick_ffr_v2 into summaries
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knobs_2m import quick_ffr_v2  # noqa: E402

SANDBOX = Path("/data/latent-basemap/sandbox")

# cuml-ref/<sub> reuses the knobs rung's truth graph (same sealed substrate rows).
CUML_ALIAS = {"2m": "2m-knobs", "6250k": "6250k-knobs",
              "12500k": "12500k-knobs", "25000k": "25000k-knobs"}


def build_rung_edges() -> dict[str, str]:
    """top-level dir name -> its canonical fuzzy-edge npz (modal 'edges' value of
    the summaries under it that carry one).  Used to resolve the few map dirs
    whose own summary omits an 'edges' field (knobs-rung ablations, cuml-ref)."""
    from collections import Counter
    per_top: dict[str, Counter] = {}
    for sj in SANDBOX.glob("*/**/summary.json"):
        try:
            e = json.loads(sj.read_text()).get("edges")
        except Exception:
            continue
        if e and Path(e).is_file():
            top = sj.relative_to(SANDBOX).parts[0]
            per_top.setdefault(top, Counter())[e] += 1
    return {top: c.most_common(1)[0][0] for top, c in per_top.items()}


def resolve_truth(map_dir: Path, rung_edges: dict[str, str]):
    """Return (edges_path or None, note).  Preference:
       1. summary['edges'] if the file exists;
       2. edges-k15-fuzzy.npz in this dir or its parent;
       3. the top-level rung's canonical edges (knobs ablations / cuml-ref)."""
    sj = map_dir / "summary.json"
    if sj.exists():
        try:
            e = json.loads(sj.read_text()).get("edges")
        except Exception:
            e = None
        if e and Path(e).is_file():
            return Path(e), "summary.edges"
    for cand in (map_dir / "edges-k15-fuzzy.npz",
                 map_dir.parent / "edges-k15-fuzzy.npz"):
        if cand.is_file():
            return cand, "sibling-npz"
    parts = map_dir.relative_to(SANDBOX).parts
    top = parts[0]
    if top in rung_edges:
        return Path(rung_edges[top]), "rung-fallback"
    if top == "cuml-ref" and len(parts) >= 2 and parts[1] in CUML_ALIAS:
        alias = CUML_ALIAS[parts[1]]
        if alias in rung_edges:
            return Path(rung_edges[alias]), "cuml-alias"
    return None, "UNRESOLVED"


def main(argv: list[str]) -> int:
    run = "--run" in argv
    rung_edges = build_rung_edges()
    coord_files = sorted(SANDBOX.glob("**/coordinates.npy"))
    print(f"{len(coord_files)} coordinate files; run={run}", flush=True)

    done = skipped = 0
    t_all = time.time()
    for cp in coord_files:
        d = cp.parent
        rel = d.relative_to(SANDBOX)
        edges, how = resolve_truth(d, rung_edges)
        if edges is None:
            print(f"SKIP {rel}  ({how})", flush=True)
            skipped += 1
            continue
        # row-count guard: coords rows must match the truth graph's node count.
        xy = np.load(cp, mmap_mode="r")
        rows = int(xy.shape[0])
        with np.load(edges) as z:
            n_nodes = int(z["n_nodes"]) if "n_nodes" in z.files else None
        if n_nodes is not None and n_nodes != rows:
            print(f"SKIP {rel}  (row mismatch coords={rows} truth={n_nodes})",
                  flush=True)
            skipped += 1
            continue
        if not run:
            knn = edges.parent / "knn_indices.npy"
            mode = "exact" if knn.is_file() else "weight"
            print(f"OK   {rel}  <- {edges.parent.name}  [{mode}, {how}]",
                  flush=True)
            done += 1
            continue

        xy = np.asarray(xy, dtype=np.float32)
        t0 = time.time()
        ffr = float(quick_ffr_v2(xy, edges, rows))
        mode = quick_ffr_v2.last_truth_mode
        sj = d / "summary.json"
        summary = json.loads(sj.read_text()) if sj.exists() else {}
        summary["quick_ffr_v2"] = ffr
        summary["quick_ffr_v2_truth_mode"] = mode
        summary["quick_ffr_v2_edges"] = str(edges)
        sj.write_text(json.dumps(summary, indent=1))
        v1 = summary.get("quick_ffr_at_0.1pct")
        v1s = f"{v1:.4f}" if isinstance(v1, (int, float)) else "n/a"
        print(f"OK   {rel}  v1={v1s} v2={ffr:.4f} [{mode},{how}] "
              f"{time.time()-t0:.1f}s", flush=True)
        done += 1

    print(f"\n{'wrote' if run else 'resolved'} {done}, skipped {skipped}, "
          f"total {(time.time()-t_all)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
