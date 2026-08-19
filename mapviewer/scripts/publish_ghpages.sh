#!/usr/bin/env bash
# publish_ghpages.sh — build the viewer + assemble the PUBLIC site and push it
# to the gh-pages branch of enjalot/latent-basemap.
#
# Layout published:
#   /                    landing redirect -> viewer/
#   /viewer/             the built viewer (relative base)
#   /viewer/packs/index.json   entries whose `url` points at GCS pack prefixes
#   /gallery/            the scale-ladder gallery (build_ladder_gallery.py)
#
# Heavy pack bytes are NOT here — they live under gs://fun-data/latent-basemap/
# (see experiments/mappack/publish_pack.py). The in-browser projection runtime
# (vendor/ + models/, ~100 MB) is included when present: every file is far
# under GitHub's 100 MB object limit.
#
# Usage:  scripts/publish_ghpages.sh [--dry-run]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SITE="$(mktemp -d /tmp/basemap-ghpages.XXXX)"
INDEX_SRC="${INDEX_SRC:-$REPO/publish/packs-index.json}"
GALLERY="${GALLERY:-$HOME/.agent/basemap-maps/gallery}"
DRY="${1:-}"

# shellcheck disable=SC1090
[ -s "$HOME/.nvm/nvm.sh" ] && . "$HOME/.nvm/nvm.sh"

cd "$HERE"
echo "==> building viewer"
npm run build >/dev/null

mkdir -p "$SITE/viewer/packs"
rsync -a --exclude packs dist/ "$SITE/viewer/"

if [ -d "$HERE/projection-poc/vendor" ]; then
  echo "==> including projection runtime"
  rsync -aL "$HERE/projection-poc/vendor/" "$SITE/viewer/vendor/"
  rsync -aL "$HERE/projection-poc/models/" "$SITE/viewer/models/"
fi

if [ -f "$INDEX_SRC" ]; then
  cp "$INDEX_SRC" "$SITE/viewer/packs/index.json"
  echo "==> pack index: $(python3 -c "import json;print(', '.join(p['map_id'] for p in json.load(open('$INDEX_SRC'))['packs']))")"
else
  echo "ERROR: no pack index at $INDEX_SRC" >&2; exit 1
fi

if [ -d "$GALLERY" ]; then
  echo "==> including gallery"
  rsync -a "$GALLERY/" "$SITE/gallery/"
fi

cat > "$SITE/index.html" <<'EOF'
<!doctype html><meta charset="utf-8"><title>latent-basemap</title>
<meta http-equiv="refresh" content="0; url=viewer/">
<a href="viewer/">viewer</a> · <a href="gallery/">gallery</a>
EOF
touch "$SITE/.nojekyll"

du -sh "$SITE" | sed 's/^/==> site size: /'
[ "$DRY" = "--dry-run" ] && { echo "dry run — site left at $SITE"; exit 0; }

WT="$REPO/.ghpages-worktree"
if [ ! -d "$WT" ]; then
  git -C "$REPO" worktree add "$WT" gh-pages 2>/dev/null || {
    git -C "$REPO" worktree add --detach "$WT"
    git -C "$WT" checkout --orphan gh-pages
    git -C "$WT" rm -rf --quiet . 2>/dev/null || true
  }
fi
rsync -a --delete --exclude .git "$SITE/" "$WT/"
cd "$WT"
git add -A
git commit -q -m "publish $(date -u +%Y-%m-%dT%H:%MZ)" || echo "==> nothing changed"
git push -q origin gh-pages
rm -rf "$SITE"
echo "done: https://enjalot.github.io/latent-basemap/"
