#!/usr/bin/env python3
"""mirror_pack.py — publish a real map pack into the viewer's packs directory.

The pack itself stays on /data (packs are big and are another tool's output —
we never write into them). What lands in the destination is:

  <dest>/<map_id>/density -> symlink        (tiles are fetched whole, no ranges)
  <dest>/<map_id>/bins    -> symlink
  <dest>/<map_id>/model   -> symlink
  <dest>/<map_id>/points/ real directory: symlinks to the .bin/.u64 files PLUS
                          generated `<name>.part{N}` slices
  <dest>/<map_id>/text    -> symlink to the substrate sidecar (only with --sidecar)
  <dest>/<map_id>/manifest.json  rewritten copy carrying a `chunking` block

The parts exist because the LAN static server (python http.server on
gsv.local:8800) ignores HTTP Range: it answers 200 with the whole body. The
viewer detects that and reads fixed-size parts instead. On a Range-capable host
(gh-pages, GCS) the same pack works without ever touching the parts.

Usage:
  mirror_pack.py --src /data/latent-basemap/mappacks/<id> --dest ~/.agent/basemap-maps/packs
  mirror_pack.py --src ... --dest ... --sidecar        # also link the text sidecar
  mirror_pack.py --src ... --dest ... --max-split-bytes 268435456
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

PART_BYTES = 4 * 1024 * 1024
RANGE_FILES = ["points/xy_id.bin", "points/lod.bin", "points/tile_index.u64"]
SIDECAR_FILES = ["text/offsets.u64", "text/blob.utf8"]


def link(dst: Path, src: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src)


def split(src: Path, out_dir: Path, part_bytes: int) -> int:
    """Write `<out_dir>/<src.name>.part{N}`; skip parts already the right size."""
    total = src.stat().st_size
    n = max(1, math.ceil(total / part_bytes))
    with src.open("rb") as fh:
        for i in range(n):
            part = out_dir / f"{src.name}.part{i}"
            expect = min(part_bytes, total - i * part_bytes)
            if part.exists() and part.stat().st_size == expect:
                continue
            fh.seek(i * part_bytes)
            part.write_bytes(fh.read(expect))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--part-bytes", type=int, default=PART_BYTES)
    ap.add_argument("--max-split-bytes", type=int, default=256 * 1024 * 1024,
                    help="files larger than this are linked but not split")
    ap.add_argument("--sidecar", action="store_true",
                    help="link + split the per-substrate text sidecar (big!)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    manifest = json.loads((src / "manifest.json").read_text())
    map_id = manifest.get("map_id", src.name)
    dest = Path(os.path.expanduser(args.dest)).resolve() / map_id
    dest.mkdir(parents=True, exist_ok=True)

    # whole-file tiers: plain symlinks
    for sub in ("density", "bins", "model"):
        if (src / sub).is_dir():
            link(dest / sub, src / sub)

    # In-browser projection: the map head is exported by a separate step and is
    # NOT listed in the pack's `files` block, so declare it in the manifest.
    # The viewer refuses to probe for it (a 404 is an error in the smoke check).
    head = src / "model" / "map_head.onnx"
    models_json = src / "model" / "models.json"
    if head.is_file() and models_json.is_file():
        encoder = None
        try:
            encoder = (json.loads(models_json.read_text()).get("encoder") or {}).get("name")
        except Exception:
            pass
        manifest["model"] = {
            "models_json": "model/models.json",
            "map_head": "model/map_head.onnx",
            "map_head_bytes": head.stat().st_size,
            "encoder": encoder,
        }
        print(f"    model: map_head.onnx {head.stat().st_size/1e6:.1f} MB")
    else:
        manifest.pop("model", None)

    chunking: dict[str, dict[str, int]] = {}

    (dest / "points").mkdir(exist_ok=True)
    for rel in RANGE_FILES:
        f = src / rel
        if not f.is_file():
            continue
        link(dest / rel, f)
        size = f.stat().st_size
        if size <= args.max_split_bytes:
            parts = split(f, dest / "points", args.part_bytes)
            chunking[rel] = {"bytes": size, "parts": parts}
            print(f"    {rel}: {size/1e6:.1f} MB -> {parts} part(s)")
        else:
            print(f"    {rel}: {size/1e6:.1f} MB — too big to split, "
                  f"Range-capable hosts only")

    # text sidecar (per substrate, lives outside the pack)
    side = ((manifest.get("text") or {}).get("sidecar") or {}).get("dir")
    if args.sidecar and side and Path(side).is_dir():
        (dest / "text").mkdir(exist_ok=True)
        for rel in SIDECAR_FILES:
            f = Path(side) / Path(rel).name
            if not f.is_file():
                continue
            link(dest / rel, f)
            size = f.stat().st_size
            if size <= args.max_split_bytes:
                parts = split(f, dest / "text", args.part_bytes)
                chunking[rel] = {"bytes": size, "parts": parts}
                print(f"    {rel}: {size/1e6:.1f} MB -> {parts} part(s)")
            else:
                print(f"    {rel}: {size/1e6:.1f} MB — linked, not split")
    else:
        # no sidecar published here: tell the viewer so it doesn't offer text
        manifest.setdefault("text", {})["text_available"] = False
        if side:
            manifest["text"]["sidecar_note"] = (
                f"not published with this pack; source: {side}")

    manifest["chunking"] = {
        "part_bytes": args.part_bytes,
        "suffix": ".part{n}",
        "files": chunking,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"    manifest -> {dest/'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
