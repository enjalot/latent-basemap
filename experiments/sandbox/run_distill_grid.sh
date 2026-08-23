#!/usr/bin/env bash
# Capacity-vs-scale distill grid — after night16, BEFORE the jina30m GPU window.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
LOG=/data/latent-basemap/sandbox/logs/distill-grid.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night16-efficiency.service; do sleep 120; done
sleep 30
log "distill grid starting"
/data/latent-basemap/umap06dev-env/bin/python experiments/sandbox/distill_grid.py prep \
  >>/data/latent-basemap/sandbox/logs/distill-grid-work.log 2>&1 \
  && log "prep DONE" || log "prep FAILED"
$LB/.venv/bin/python experiments/sandbox/distill_grid.py grid \
  >>/data/latent-basemap/sandbox/logs/distill-grid-work.log 2>&1 \
  && log "grid DONE" || log "grid FAILED"
log "distill grid finished — GPU idle"
