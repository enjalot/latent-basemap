#!/usr/bin/env python3
"""publish_pack.py — prepare a map pack for PUBLIC hosting on GCS.

The public architecture (PLAN6, owner-approved 2026-08-19):
  * gh-pages hosts the viewer site + packs/index.json ONLY (density planes
    alone are ~800 MB per big pack — far past gh-pages comfort).
  * each published pack lives whole under a GCS prefix; the index entry's
    `url` points at it. GCS serves HTTP ranges natively, so deep points and
    text sidecars work in the viewer's preferred range mode.

What this script does per pack (NO uploads itself — it emits the commands):
  1. writes a CLEANED manifest to a small staging dir: internal absolute
     paths (substrate_dir, source_coordinates, sidecar dir, build timings)
     are stripped; everything the viewer reads is kept byte-identical.
  2. emits `upload.sh` — gsutil commands that upload the original pack
     files (density/, points/, bins/, model/ if present), the staged
     manifest, and optionally the text sidecar (offsets/blob from the
     substrate sidecar dir) under the GCS prefix. No 48 GB copies on disk.
  3. prints the packs/index.json entry to add on the gh-pages side.

Usage:
  publish_pack.py --pack /data/latent-basemap/mappacks/<id> \
      --gcs gs://fun-data/latent-basemap/packs/<id> \
      [--title "..."] [--with-text] [--staging DIR]
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

# manifest keys that leak internal filesystem layout or build context;
# everything else is preserved byte-for-byte.
STRIP_KEYS = {"substrate_dir", "source_coordinates", "timings_s", "build_wall_s"}
GCS_HTTP = "https://storage.googleapis.com/"


def clean_manifest(man: dict) -> dict:
    out = {k: v for k, v in man.items() if k not in STRIP_KEYS}
    prov = out.get("provenance")
    if isinstance(prov, str) and prov.startswith("/"):
        out["provenance"] = Path(prov).name
    text = out.get("text")
    if isinstance(text, dict):
        sidecar = text.get("sidecar")
        if isinstance(sidecar, dict):
            sidecar.pop("dir", None)
    sub_man = out.get("substrate_manifest")
    if isinstance(sub_man, dict) and isinstance(sub_man.get("canonical_path"), str):
        sub_man["canonical_path"] = Path(sub_man["canonical_path"]).name
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--gcs", required=True,
                    help="gs://bucket/prefix for this pack (no trailing slash)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--with-text", action="store_true",
                    help="also upload the substrate text sidecar as text/")
    ap.add_argument("--sidecar-root", type=Path,
                    default=Path("/data/latent-basemap/textsidecar"))
    ap.add_argument("--staging", type=Path,
                    default=Path("/data/latent-basemap/publish"))
    args = ap.parse_args()

    pack: Path = args.pack
    gcs = args.gcs.rstrip("/")
    if not gcs.startswith("gs://"):
        raise SystemExit("--gcs must be a gs:// prefix")
    man_path = pack / "manifest.json"
    man = json.loads(man_path.read_text())
    map_id = man.get("map_id", pack.name)

    stage = args.staging / map_id
    stage.mkdir(parents=True, exist_ok=True)

    sidecar_dir = None
    if args.with_text:
        text = man.get("text") or {}
        side = (text.get("sidecar") or {}).get("dir")
        if side is None:
            # mirror-style pack: text/ may already be symlinks into the sidecar
            side = pack / "text"
        sidecar_dir = Path(side)
        for f in ("offsets.u64", "blob.utf8"):
            if not (sidecar_dir / f).exists():
                raise SystemExit(f"--with-text: missing {sidecar_dir / f}")
    else:
        man = dict(man)
        if isinstance(man.get("text"), dict):
            man["text"] = {**man["text"], "text_available": False}

    cleaned = clean_manifest(man)
    (stage / "manifest.json").write_text(json.dumps(cleaned, indent=1))

    cmds = ["#!/usr/bin/env bash", "set -euo pipefail",
            'GSUTIL="${GSUTIL:-gsutil -m}"', ""]
    for sub in ("density", "points", "bins", "model"):
        if (pack / sub).exists():
            cmds.append(f"$GSUTIL rsync -r {shlex.quote(str(pack / sub))} "
                        f"{shlex.quote(f'{gcs}/{sub}')}")
    if sidecar_dir is not None:
        for f in ("offsets.u64", "blob.utf8"):
            src = (sidecar_dir / f).resolve()
            cmds.append(f"$GSUTIL cp {shlex.quote(str(src))} "
                        f"{shlex.quote(f'{gcs}/text/{f}')}")
    # cache header set AT upload (creator-only IAM has no objects.update, so
    # setmeta is unavailable); the manifest is the mutable entry point
    cmds.append(f"$GSUTIL -h 'Cache-Control:public, max-age=300' cp "
                f"{shlex.quote(str(stage / 'manifest.json'))} "
                f"{shlex.quote(f'{gcs}/manifest.json')}")
    upload = stage / "upload.sh"
    upload.write_text("\n".join(cmds) + "\n")
    upload.chmod(0o755)

    http_base = GCS_HTTP + gcs[len("gs://"):] + "/"
    entry = {
        "map_id": map_id,
        "title": args.title or map_id,
        "url": http_base,
        "N": man.get("n_points"),
        "zmax": (man.get("tiles") or {}).get("max_zoom"),
        "synthetic": False,
    }
    (stage / "index-entry.json").write_text(json.dumps(entry, indent=1))
    print(f"staged: {stage}")
    print(f"upload: {upload}")
    print(f"entry : {json.dumps(entry)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
