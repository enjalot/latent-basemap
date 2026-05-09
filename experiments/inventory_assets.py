#!/usr/bin/env python3
"""
Inventory local and Modal assets without loading tensors.

This is intentionally dependency-light so it can run before the ML environment
is installed. It records paths, sizes, and small NPZ metadata for planning local
experiments and deciding what still needs to be pulled from Modal.

Examples:
    python3 -m experiments.inventory_assets --local /data --output experiments/assets.json
    python3 -m experiments.inventory_assets --modal-volume pumap-data --modal-volume checkpoints
"""

import argparse
import json
import os
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path


INTERESTING_SUFFIXES = {
    ".index", ".npz", ".pkl", ".pt", ".pth", ".h5", ".parquet", ".npy", ".json", ".yaml"
}


def _run(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT, text=True)
        return {"ok": True, "stdout": out.strip()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _npz_members(path):
    try:
        with zipfile.ZipFile(path) as zf:
            return [
                {
                    "name": info.filename,
                    "size_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                }
                for info in zf.infolist()
            ]
    except Exception as exc:
        return [{"error": str(exc)}]


def inventory_local(root):
    root = Path(os.path.expanduser(root))
    entries = []
    if not root.exists():
        return {"root": str(root), "exists": False, "entries": entries}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in INTERESTING_SUFFIXES:
            continue
        stat = path.stat()
        entry = {
            "path": str(path),
            "suffix": path.suffix,
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
        if path.suffix == ".npz":
            entry["npz_members"] = _npz_members(path)
        entries.append(entry)

    return {
        "root": str(root),
        "exists": True,
        "entries": entries,
        "total_size_bytes": sum(e["size_bytes"] for e in entries),
    }


def inventory_modal_volume(volume, paths):
    listings = {}
    for path in paths:
        result = _run(["modal", "volume", "ls", volume, path])
        listings[path] = result
    return {"volume": volume, "listings": listings}


def main():
    parser = argparse.ArgumentParser(description="Inventory latent-basemap assets")
    parser.add_argument("--local", action="append", default=None,
                        help="Local root to scan; may be provided more than once")
    parser.add_argument("--modal-volume", action="append", default=[],
                        help="Modal volume to list")
    parser.add_argument("--modal-path", action="append", default=["/"],
                        help="Modal path to list for every --modal-volume")
    parser.add_argument("--output", default="experiments/assets_inventory.json",
                        help="JSON output path")
    args = parser.parse_args()
    local_roots = args.local or ["/data/checkpoints/pumap"]

    report = {
        "created_at": datetime.now().isoformat(),
        "local": [inventory_local(path) for path in local_roots],
        "modal": [inventory_modal_volume(vol, args.modal_path) for vol in args.modal_volume],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {out_path}")
    for local in report["local"]:
        print(f"local {local['root']}: {len(local.get('entries', []))} files")
    for modal in report["modal"]:
        print(f"modal {modal['volume']}: {len(modal['listings'])} listing(s)")


if __name__ == "__main__":
    main()
