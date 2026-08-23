#!/usr/bin/env bash
# Dedup round (owner 2026-08-23): sisap dedup substrate -> graph -> the same
# two recipes as the original suite, for the satellite/render comparison.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
UPY=/data/latent-basemap/umap06dev-env/bin/python
LOG=/data/latent-basemap/sandbox/logs/dedup-round.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
run() { local label=$1; shift
  "$@" >>/data/latent-basemap/sandbox/logs/dedup-round-work.log 2>&1 \
    && log "$label DONE" || log "$label FAILED (continuing)"; }
while systemctl --user is-active --quiet night12-winners.service; do sleep 120; done
sleep 30
log "dedup round starting"
if [ ! -f /data/latent-basemap/sandbox/sisap-clip-2m-dedup/edges-k15-fuzzy.npz ]; then
  run "dedup knn" $PY experiments/sandbox/image_map_pipeline.py sisap-clip-2m-dedup knn
  run "dedup fuzzy" $UPY experiments/sandbox/image_map_pipeline.py sisap-clip-2m-dedup fuzzy
fi
run "dedup train" $PY experiments/sandbox/image_map_pipeline.py sisap-clip-2m-dedup train
$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "dedup round finished — GPU idle"
