#!/usr/bin/env bash
# Night driver #14: architecture sweep under the x2 composed core.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
LOG=/data/latent-basemap/sandbox/logs/night14-arch.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night13-md010.service; do sleep 120; done
sleep 30
log "night14 arch sweep starting"
for arm in core-h512 core-h1024 core-h3072 core-h4096 core-L2 core-L4 core-L5 core-mlp; do
  [ -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ] && { log "arm $arm done, skip"; continue; }
  $LB/.venv/bin/python experiments/sandbox/knobs_2m.py --arm "$arm" \
    >>"/data/latent-basemap/sandbox/logs/arm-$arm.log" 2>&1 \
    && log "arm $arm DONE" || log "arm $arm FAILED (continuing)"
done
$LB/.venv/bin/python experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night14 driver finished — GPU idle"
