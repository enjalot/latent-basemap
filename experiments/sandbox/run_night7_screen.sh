#!/usr/bin/env bash
# Composition screen: x2 composed core + each rejected lever (4 arms, ~1.6h).
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
LOG=/data/latent-basemap/sandbox/logs/night7-screen.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night6-final.service; do sleep 120; done
sleep 60
log "night7 composition screen starting"
for arm in umap-md000-x2-core-rankneg500k umap-md000-x2-core-wes \
           umap-md000-x2-core-anneal25 umap-md005-x2-core; do
  [ -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ] && { log "arm $arm done, skip"; continue; }
  $LB/.venv/bin/python experiments/sandbox/knobs_2m.py --arm "$arm" \
    >>/data/latent-basemap/sandbox/logs/arm-$arm.log 2>&1 \
    && log "arm $arm DONE" || log "arm $arm FAILED (continuing)"
done
$LB/.venv/bin/python experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night7 screen finished — GPU idle"
