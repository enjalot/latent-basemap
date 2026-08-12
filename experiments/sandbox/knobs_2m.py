"""One-variable visual-quality arms at 2M on the sealed R0216 substrate/graph.

Owner-directed sandbox (see PLAN.md). Not a round: no capability, no gate, no
sealed claim. Each arm re-runs R0217's treatment with exactly one knob changed
and renders the result for visual comparison against the sealed baseline.

Safety: refuses to start when any compute process holds the GPU (the round
runner keeps priority), and refuses when the output directory already exists
(arms are write-once; delete the arm dir to re-run).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

R0216 = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts"
             "/minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
SUBSTRATE = R0216 / "substrate.f32.npy"
EDGES = R0216 / "edges-k15-fuzzy.npz"
R0217_RECEIPT = Path("/data/latent-basemap/runs/round-0217/queue-correction-1/artifacts"
                     "/minilm-mixed-2m-map-seed42-low-dose-v1/train-receipt.json")
OUT_ROOT = Path("/data/latent-basemap/sandbox/2m-knobs")
SITE_DIR = Path.home() / ".agent/basemap-maps/sandbox/2m-knobs"

ROWS = 2_000_000
DIRECTED_EDGES = 48_344_648
BATCH = 8192
POS_RATIO = 0.05
POS_PER_UPDATE = int(BATCH * POS_RATIO)          # 409
BASE_HORIZON = 80_163                            # R0217: 0.6782 draws/edge
SEED = 42

# R0217's constructor treatment (from the sealed checkpoint + config).
BASE_KWARGS = dict(
    n_components=2, hidden_dim=2048, n_layers=3, n_neighbors=15,
    a=1.0, b=1.0, low_dim_kernel="legacy_lp", correlation_weight=0.0,
    learning_rate=1e-3, n_epochs=1, batch_size=BATCH, pos_ratio=POS_RATIO,
    architecture="residual_bottleneck", positive_target_mode="binary",
    lr_schedule="cosine", use_amp=True, use_batchnorm=False, use_dropout=False,
    total_steps_estimate=BASE_HORIZON,
    require_full_budget=False,   # exploratory: the horizon break ends the run
    device="cuda",
)

ARMS: dict[str, dict] = {
    "replay-baseline": {},
    "dose-x2": {"total_steps_estimate": 2 * BASE_HORIZON, "n_epochs": 2},
    "kernel-a4": {"a": 4.0},
    "kernel-b2": {"b": 2.0},
    "umap-kernel": {"low_dim_kernel": "umap", "a": 1.577, "b": 0.895},
}


def gpu_is_free() -> bool:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        raise SystemExit("nvidia-smi failed; do not touch a possibly-sick driver")
    return not out.stdout.strip()


def receipt_diff(kwargs: dict) -> list[str]:
    """Fields where our constructor departs from R0217's sealed checkpoint.

    The sealed train receipt + checkpoint are the authority on the treatment;
    an arm must depart only on its own knob.
    """
    import torch
    ck_path = R0217_RECEIPT.parent / "model.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    mismatches = []
    for key in ("a", "b", "low_dim_kernel", "correlation_weight", "learning_rate",
                "pos_ratio", "positive_target_mode", "use_batchnorm", "use_dropout",
                "architecture"):
        ours, sealed = kwargs.get(key), ck.get(key)
        if key == "architecture":
            sealed = ck.get("architecture")
        if ours != sealed:
            mismatches.append(f"{key}: ours={ours!r} sealed={sealed!r}")
    return mismatches


def quick_ffr(xy: np.ndarray, n_queries: int = 20_000, k_true: int = 15) -> float:
    """FFR@0.1% with the sealed exact graph's edges as high-D truth.

    The k15 graph IS brute-force truth on this substrate (R0216), so each
    query's edge destinations are its true neighbors. Not the sealed panel;
    a fast guard against an arm that tidies the picture by scrambling it.
    """
    from scipy.spatial import cKDTree

    with np.load(EDGES) as z:
        names = z.files
        src_name = next(n for n in ("sources", "src", "rows") if n in names)
        dst_name = next(n for n in ("destinations", "dst", "cols", "targets") if n in names)
        sources = z[src_name]
        dests = z[dst_name]
    order = np.argsort(sources, kind="stable")
    sources, dests = sources[order], dests[order]
    starts = np.searchsorted(sources, np.arange(ROWS))
    ends = np.searchsorted(sources, np.arange(ROWS), side="right")

    rng = np.random.default_rng(0)
    queries = rng.choice(ROWS, size=n_queries, replace=False)
    disc = max(int(ROWS * 0.001), 1)
    tree = cKDTree(xy)
    _, near = tree.query(xy[queries], k=disc, workers=8)
    hits = 0
    total = 0
    for qi, q in enumerate(queries):
        truth = dests[starts[q]:ends[q]][:k_true]
        if len(truth) == 0:
            continue
        hits += np.isin(truth, near[qi]).sum()
        total += len(truth)
    return hits / max(total, 1)


def run_arm(arm: str, dry_run: bool) -> int:
    overrides = ARMS[arm]
    kwargs = {**BASE_KWARGS, **overrides}
    horizon = kwargs["total_steps_estimate"]
    dose = horizon * POS_PER_UPDATE / DIRECTED_EDGES

    for path in (SUBSTRATE, EDGES, R0217_RECEIPT):
        if not path.is_file():
            raise SystemExit(f"missing sealed input: {path}")

    expected_knobs = set(overrides)
    departures = [d for d in receipt_diff(kwargs)
                  if d.split(":")[0] not in expected_knobs]
    print(f"arm={arm}  horizon={horizon} updates  dose={dose:.4f} draws/edge")
    print("constructor:", json.dumps({k: v for k, v in kwargs.items()}, default=str))
    if departures:
        raise SystemExit("treatment departs from R0217 outside the arm's knob:\n  "
                         + "\n  ".join(departures))
    print("receipt check: all non-arm fields match R0217's sealed checkpoint")
    if dry_run:
        return 0

    out_dir = OUT_ROOT / arm
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; arms are write-once (delete to re-run)")
    if not gpu_is_free():
        raise SystemExit("GPU busy (round runner has priority); try again later")
    out_dir.mkdir(parents=True)
    log_path = out_dir / "fit.log"
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(log_path),
                                  logging.StreamHandler()])

    import torch
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    torch.manual_seed(SEED)
    started = datetime.datetime.now(datetime.timezone.utc)
    X = np.load(SUBSTRATE, mmap_mode="r")
    model = ParametricUMAP(**kwargs)
    model.fit(X, precomputed_edges_path=str(EDGES), random_state=SEED)
    model.save(str(out_dir / "model.pt"))
    xy = model.transform(X, batch_size=8192).astype(np.float32)
    np.save(out_dir / "coordinates.npy", xy)
    wall_s = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds()

    ffr = quick_ffr(xy)
    from map_renders import robust_extent, binned_counts, render_png
    extent = robust_extent(xy)
    render_png(binned_counts(xy, extent), out_dir / "density.png")

    summary = {
        "arm": arm, "overrides": overrides, "seed": SEED,
        "horizon_updates": horizon, "draws_per_edge": dose,
        "wall_s": wall_s, "quick_ffr_at_0.1pct": ffr,
        "substrate": str(SUBSTRATE), "edges": str(EDGES),
        "started_utc": started.isoformat(),
        "note": "sandbox artifact; not a round, no sealed claim",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    build_page()
    return 0


def build_page() -> None:
    """Comparison page over every completed arm plus the sealed baseline render."""
    import html as html_mod
    import shutil

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    cards = []
    sealed = Path("/data/latent-basemap/render-cache"
                  "/round-0217-minilm-mixed-2m-map-seed42-low-dose-v1/density.png")
    if sealed.is_file():
        shutil.copy2(sealed, SITE_DIR / "sealed-r0217-seed42.png")
        cards.append(("sealed R0217 seed 42", "sealed-r0217-seed42.png",
                      "the registered low-dose baseline (ffr 0.3369)"))
    for arm_dir in sorted(OUT_ROOT.iterdir()) if OUT_ROOT.is_dir() else []:
        s_path = arm_dir / "summary.json"
        png = arm_dir / "density.png"
        if not (s_path.is_file() and png.is_file()):
            continue
        s = json.loads(s_path.read_text())
        name = f"{s['arm']}.png"
        shutil.copy2(png, SITE_DIR / name)
        cards.append((s["arm"], name,
                      f"quick-ffr {s['quick_ffr_at_0.1pct']:.4f} · "
                      f"{s['horizon_updates']:,} updates · "
                      f"{s['draws_per_edge']:.3f} draws/edge"))
    figs = "".join(
        f'<figure style="margin:0"><img src="{f}" style="width:100%;border:1px solid #ddd;border-radius:6px">'
        f'<figcaption><b>{html_mod.escape(t)}</b><br><small>{html_mod.escape(c)}</small></figcaption></figure>'
        for t, f, c in cards)
    (SITE_DIR / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>2M knob sandbox</title>"
        '<body style="font-family:system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem">'
        "<h1>2M visual-quality knobs (sandbox)</h1>"
        "<p>One knob per arm on the sealed R0216 substrate/graph, seed 42. "
        "Not rounds; quick-ffr uses sealed graph edges as truth. "
        "See experiments/sandbox/PLAN.md.</p>"
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px">{figs}</div>')
    print(f"page: {SITE_DIR}/index.html  (http://gsv.local:8800/basemap-maps/sandbox/2m-knobs/)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=sorted(ARMS), required=False)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild-page", action="store_true")
    args = ap.parse_args()
    if args.rebuild_page:
        build_page()
        return 0
    if not args.arm:
        raise SystemExit("pass --arm (or --rebuild-page)")
    return run_arm(args.arm, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
