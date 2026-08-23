#!/usr/bin/env bash
# Night driver #13 (owner 2026-08-23): winner@md010 on sisap + both jina 2Ms.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
LOG=/data/latent-basemap/sandbox/logs/night13-md010.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night12-winners.service; do sleep 120; done
sleep 30
log "night13 md010 driver starting"
for ds in sisap-clip-2m jina-en-2m jina-multi-2m; do
  log "=== STAGE $ds composed-x8-md010 start"
  $LB/.venv/bin/python experiments/sandbox/image_map_pipeline.py "$ds" train \
    >>"/data/latent-basemap/sandbox/logs/$ds.log" 2>&1 \
    && log "$ds DONE" || log "$ds FAILED (continuing)"
done
$LB/.venv/bin/python experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night13 driver finished — GPU idle"
