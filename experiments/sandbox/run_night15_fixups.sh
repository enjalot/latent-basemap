#!/usr/bin/env bash
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
LOG=/data/latent-basemap/sandbox/logs/night15-fixups.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night14-arch.service; do sleep 120; done
sleep 30
log "night15 fixups starting"
while IFS= read -r arm; do
  [ -z "$arm" ] && continue
  [ -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ] && continue
  $LB/.venv/bin/python experiments/sandbox/knobs_2m.py --arm "$arm" \
    >>"/data/latent-basemap/sandbox/logs/arm-$arm.log" 2>&1 \
    && log "arm $arm DONE" || log "arm $arm FAILED"
done </data/latent-basemap/sandbox/next-arms.txt
$LB/.venv/bin/python experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night15 finished — GPU idle"
