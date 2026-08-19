#!/usr/bin/env bash
# deploy.sh — build the viewer and publish it under ~/.agent/basemap-maps/,
# which the LAN server (moonshine-web.service) exposes at
#   http://gsv.local:8800/basemap-maps/viewer/
#
# Layout produced:
#   ~/.agent/basemap-maps/viewer/          the built static site
#   ~/.agent/basemap-maps/viewer/packs ->  ../packs   (symlink)
#   ~/.agent/basemap-maps/packs/           map packs (fixtures and/or real)
#   ~/.agent/basemap-maps/packs/index.json merged pack index
#
# Real packs: if $REAL_PACKS (default /data/latent-basemap/mappacks) exists,
# every pack directory in it is symlinked into packs/ and merged into the
# index. Fixtures are only linked when $WITH_FIXTURES=1 (default 1).
#
# In-browser projection (viewer v2): the vendored ONNX runtime, the MiniLM
# encoder and the tokenizer live in projection-poc/{vendor,models} and are
# gitignored. They are NOT part of the vite build (122 MB); this script rsyncs
# them next to the site so `src/projection.ts` finds them at ./vendor/ and
# ./models/. Set PROJECTION=0 to publish the viewer without them.
#
# Usage:
#   scripts/deploy.sh
#   WITH_FIXTURES=0 scripts/deploy.sh
#   PROJECTION=0 scripts/deploy.sh
#   REAL_PACKS=/data/latent-basemap/mappacks scripts/deploy.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$HOME/.agent/basemap-maps}"
REAL_PACKS="${REAL_PACKS:-/data/latent-basemap/mappacks}"
WITH_FIXTURES="${WITH_FIXTURES:-1}"
PROJECTION="${PROJECTION:-1}"

# shellcheck disable=SC1090
[ -s "$HOME/.nvm/nvm.sh" ] && . "$HOME/.nvm/nvm.sh"

cd "$HERE"
echo "==> building"
npm run build

mkdir -p "$DEST/viewer" "$DEST/packs"

echo "==> publishing site -> $DEST/viewer"
# --delete only prunes what this script owns. `packs` is a symlink we manage,
# `vendor`/`models` are rsynced separately below, and other tools publish
# sibling directories (round-*/ projection pages) into the same viewer root.
rsync -a --delete \
  --exclude packs --exclude vendor --exclude models --exclude 'round-*' \
  dist/ "$DEST/viewer/"
ln -sfn ../packs "$DEST/viewer/packs"

if [ "$PROJECTION" = "1" ] && [ -d "$HERE/projection-poc/vendor" ]; then
  echo "==> publishing projection runtime -> $DEST/viewer/{vendor,models}"
  rsync -aL --delete "$HERE/projection-poc/vendor/" "$DEST/viewer/vendor/"
  rsync -aL --delete "$HERE/projection-poc/models/" "$DEST/viewer/models/"
  du -sh "$DEST/viewer/vendor" "$DEST/viewer/models" | sed 's/^/    /'
else
  echo "==> skipping projection runtime (PROJECTION=$PROJECTION)"
fi

link_pack() {  # $1 = source pack dir
  local src="$1" name
  name="$(basename "$src")"
  [ -f "$src/manifest.json" ] || return 0
  ln -sfn "$src" "$DEST/packs/$name"
  echo "    linked $name -> $src"
}

SIDECAR_FLAG=""
[ "${WITH_SIDECAR:-0}" = "1" ] && SIDECAR_FLAG="--sidecar"

if [ -d "$REAL_PACKS" ]; then
  echo "==> mirroring real packs from $REAL_PACKS"
  for d in "$REAL_PACKS"/*/; do
    [ -f "${d}manifest.json" ] || continue
    echo "  $(basename "${d%/}")"
    python3 "$HERE/scripts/mirror_pack.py" --src "${d%/}" --dest "$DEST/packs" \
      $SIDECAR_FLAG
  done
else
  echo "==> no real packs at $REAL_PACKS (skipping)"
fi

if [ "$WITH_FIXTURES" = "1" ] && [ -d "$HERE/packs" ]; then
  echo "==> linking fixture packs from $HERE/packs"
  for d in "$HERE"/packs/*/; do [ -d "$d" ] && link_pack "${d%/}"; done
fi

echo "==> rebuilding $DEST/packs/index.json"
python3 - "$DEST/packs" <<'PY'
import json, os, sys
root = sys.argv[1]
packs = []
for name in sorted(os.listdir(root)):
    mf = os.path.join(root, name, "manifest.json")
    if not os.path.isfile(mf):
        continue
    try:
        m = json.load(open(mf))
    except Exception as e:
        print(f"    skip {name}: {e}")
        continue
    # keys differ between the fixture schema and the real builder's manifest
    packs.append({
        "map_id": m.get("map_id", name),
        "title": m.get("title", name),
        "path": name,
        "N": m.get("N") or m.get("n_points"),
        "zmax": ((m.get("zoom") or {}).get("max")
                 if m.get("zoom") else (m.get("tiles") or {}).get("max_zoom")),
        "synthetic": bool(m.get("synthetic")),
    })
json.dump({"schema": "basemap-pack-index-v1", "packs": packs},
          open(os.path.join(root, "index.json"), "w"), indent=1)
print(f"    {len(packs)} pack(s): " + ", ".join(p["map_id"] for p in packs))
PY

echo
echo "done:  http://gsv.local:8800/basemap-maps/viewer/"
