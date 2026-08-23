#!/usr/bin/env bash
# Night driver #12: runs the arm list written at candidate-selection time
# (/data/latent-basemap/sandbox/next-arms.txt — the crowned recipe's md005/
# md010 variants + rankneg composition, decided from tonight's results).
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
LIST=/data/latent-basemap/sandbox/next-arms.txt
LOG=/data/latent-basemap/sandbox/logs/night12-winners.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet perf-round2.service; do sleep 120; done
sleep 30
log "night12 winners driver starting"
if [ ! -f "$LIST" ]; then log "no next-arms.txt — nothing to do"; exit 0; fi
while IFS= read -r arm; do
  [ -z "$arm" ] && continue
  [ -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ] && { log "arm $arm done, skip"; continue; }
  $LB/.venv/bin/python experiments/sandbox/knobs_2m.py --arm "$arm" \
    >>"/data/latent-basemap/sandbox/logs/arm-$arm.log" 2>&1 \
    && log "arm $arm DONE" || log "arm $arm FAILED (continuing)"
done <"$LIST"
$LB/.venv/bin/python experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night12 driver finished — GPU idle"
