#!/usr/bin/env python3
"""Export sandbox map coordinates for the compare page.

Every sandbox arm with coordinates.npy becomes data/<rung>/<arm>.bin:
int16 x,y interleaved (little-endian), quantized to the arm's 0.1/99.9
percentile box (matches the render clipping convention). catalog.json lists
arms recent-first with the dequant transform, FFR, params, and rung row
counts so the UI can group same-substrate maps (only same-rung maps are
comparable point-for-point).

Idempotent: bins are rewritten only when coordinates.npy is newer. Run after
new arms seal (the review watcher can call it).
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")
OUT = Path.home() / ".agent/basemap-maps/compare"
DATA = OUT / "data"

#: rung dir -> (label, model, corpus). Anything not listed still exports with
#: generic labels; this is presentation only.
RUNG_META = {
    "2m-knobs": ("MiniLM 2M", "MiniLM", "mixed-4"),
    "6250k-knobs": ("MiniLM 6.25M", "MiniLM", "mixed-4"),
    "500k-crosscheck": ("MiniLM 500K", "MiniLM", "mixed-4"),
    "minilm-mix-1m": ("MiniLM 1M", "MiniLM", "mixed-4"),
    "minilm-mix-500k": ("MiniLM 500K", "MiniLM", "mixed-4"),
    "bl-siglip-1m": ("BL 1.08M", "SigLIP2", "BL books"),
    "sisap-clip-2m": ("LAION 2M", "CLIP ViT-L/14", "LAION"),
    "sisap-clip-2m-dedup": ("LAION 2M dedup", "CLIP ViT-L/14", "LAION dedup"),
    "jina-en-2m": ("jina EN 2M", "jina-v5-nano", "mixed-3 EN"),
    "jina-multi-2m": ("jina multi 2M", "jina-v5-nano", "EN+20 langs"),
    "jina-multi-6m": ("jina multi 6.25M", "jina-v5-nano", "EN+20 langs"),
    "minilm-redditmix-2m": ("redditmix 2M", "MiniLM", "mixed-4+reddit"),
    "reddit-2m": ("reddit 2M", "MiniLM", "reddit"),
    "communityarchive-2m": ("CA 2M", "MiniLM", "tweets"),
    "minilm-curated-2m": ("curated 2M", "MiniLM", "mixed-4 curated"),
    "minilm-random-2m": ("random 2M", "MiniLM", "mixed-4 random"),
}

#: rungs whose content is images (table shows thumbnails, not text)
IMAGE_RUNGS = {"bl-siglip-1m", "sisap-clip-2m", "sisap-clip-2m-dedup"}

#: bump to force re-export of all bins (e.g. alignment changes)
EXPORT_VERSION = 2  # v2: procrustes-align arms to the rung's md000 upstream


def rung_reference(rung_dir: Path) -> Path | None:
    """The md000 0.6dev upstream arm of a rung (alignment reference)."""
    for name in ("upstream-06dev", "upstream-06dev-2m"):
        d = rung_dir / name / "coordinates.npy"
        if d.exists():
            return d
    return None


def procrustes(xy: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Similarity-align xy onto ref (rotation+reflection+scale, all rows)."""
    mx, mr = xy.mean(axis=0), ref.mean(axis=0)
    Xc, Rc = xy - mx, ref - mr
    u, s, vt = np.linalg.svd(Xc.T @ Rc)
    rot = u @ vt
    scale = s.sum() / max((Xc ** 2).sum(), 1e-12)
    return scale * (Xc @ rot) + mr


def export_arm(rung: str, arm_dir: Path,
               ref: np.ndarray | None) -> dict | None:
    cf = arm_dir / "coordinates.npy"
    sf = arm_dir / "summary.json"
    if not cf.exists() or not sf.exists():
        return None
    s = json.loads(sf.read_text())
    if s.get("n_components", 2) != 2:
        return None
    bin_path = DATA / rung / f"{arm_dir.name}.bin"
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    xy = np.asarray(np.load(cf, mmap_mode="r"), dtype=np.float64)
    n = int(xy.shape[0])
    aligned = False
    if ref is not None and ref.shape[0] == n:
        xy = procrustes(xy, ref)
        aligned = True
    ver = bin_path.with_suffix(".v")
    stale = (not bin_path.exists()
             or bin_path.stat().st_mtime < cf.stat().st_mtime
             or not ver.exists()
             or ver.read_text() != f"{EXPORT_VERSION}:{int(aligned)}")
    lo = np.percentile(xy, 0.1, axis=0).astype(np.float64)
    hi = np.percentile(xy, 99.9, axis=0).astype(np.float64)
    span = np.maximum(hi - lo, 1e-9)
    if stale:
        q = (xy - lo) / span
        q = np.clip(q, 0.0, 1.0) * 65535.0 - 32768.0
        q.astype("<i2").tofile(bin_path)
        ver.write_text(f"{EXPORT_VERSION}:{int(aligned)}")

    ts = None
    if s.get("started_utc"):
        try:
            ts = datetime.datetime.fromisoformat(s["started_utc"]).timestamp()
        except ValueError:
            pass
    if ts is None:
        ts = sf.stat().st_mtime
    label, model, corpus = RUNG_META.get(rung, (rung, "?", "?"))
    return {
        "rung": rung, "arm": arm_dir.name, "n": n,
        "bin": f"data/{rung}/{arm_dir.name}.bin",
        "lo": list(lo), "span": list(span),
        "ffr": s.get("quick_ffr_at_0.1pct"),
        "ts": ts,
        "date_et": datetime.datetime.fromtimestamp(ts).astimezone(
            datetime.timezone(datetime.timedelta(hours=-4))
        ).strftime("%m-%d %H:%M"),
        "upstream": "upstream" in arm_dir.name,
        "rung_label": label, "model": model, "corpus": corpus,
        "content": "image" if rung in IMAGE_RUNGS else "text",
        "aligned": aligned,
        "overrides": s.get("overrides") or {},
        "dose": s.get("dose_multiplier"),
    }


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    entries = []
    for rung_dir in sorted(SANDBOX.iterdir()):
        if not rung_dir.is_dir() or rung_dir.name in ("logs", "distill-grid"):
            continue
        ref_path = rung_reference(rung_dir)
        ref = None
        if ref_path is not None:
            r = np.load(ref_path, mmap_mode="r")
            if r.ndim == 2 and r.shape[1] == 2:
                ref = np.asarray(r, dtype=np.float64)
        for arm_dir in sorted(rung_dir.iterdir()):
            if not arm_dir.is_dir():
                continue
            try:
                e = export_arm(rung_dir.name, arm_dir, ref)
            except Exception as ex:  # never let one arm kill the export
                print(f"SKIP {rung_dir.name}/{arm_dir.name}: {ex}")
                continue
            if e:
                entries.append(e)
    entries.sort(key=lambda e: -e["ts"])
    (DATA / "catalog.json").write_text(json.dumps({
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "maps": entries,
    }))
    n2m = sum(1 for e in entries if 1_900_000 <= e["n"] <= 2_200_000)
    print(f"{len(entries)} maps exported ({n2m} at ~2M) -> {DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
