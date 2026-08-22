#!/usr/bin/env bash
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
LOGDIR=/data/latent-basemap/sandbox/logs
LOG=$LOGDIR/cheap-anticollapse.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night7-screen.service; do sleep 60; done
sleep 20
log "cheap anti-collapse starting"
for arm in umap-md005-x2-fneg10-tanh4 umap-md010-x2-fneg10-tanh4; do
  [ -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ] && continue
  $PY experiments/sandbox/knobs_2m.py --arm "$arm" \
    >>"$LOGDIR/arm-$arm.log" 2>&1 && log "arm $arm DONE" || log "arm $arm FAILED"
done
$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "cheap anti-collapse finished"
