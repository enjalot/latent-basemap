#!/usr/bin/env bash
# Build the React map-viewer app and deploy it to the static site under
# ~/.agent/basemap-maps/app/. The app dir is additive (it does not touch the
# python-generated viewer/, maps-index.json, or legacy pages).
#
# Usage: web/deploy.sh
# Node 24 via nvm (system node is too old, engine-strict fails).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$HOME/.agent/basemap-maps/app}"

# shellcheck disable=SC1090
source "$HOME/.nvm/nvm.sh"
nvm use 24 >/dev/null

cd "$HERE"

# Install deps if node_modules is missing or package.json is newer.
if [ ! -d node_modules ] || [ package.json -nt node_modules ]; then
  echo "[deploy] installing deps…"
  if [ -f package-lock.json ]; then npm ci; else npm install; fi
fi

echo "[deploy] building (vite, base './')…"
npm run build

echo "[deploy] rsync dist/ -> $DEST"
mkdir -p "$DEST"
rsync -a --delete dist/ "$DEST/"

echo "[deploy] done. Open: http://gsv.local:8800/basemap-maps/app/index.html#/"
