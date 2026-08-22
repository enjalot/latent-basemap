#!/usr/bin/env bash
# Night driver #10 (owner 2026-08-22): the anti-collapse ladder — looser
# kernels x best candidates on MiniLM 2M, + md005+tanh4 on both jina suites.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
LOGDIR=/data/latent-basemap/sandbox/logs
LOG=$LOGDIR/night10-anticollapse.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night8-registers.service; do sleep 120; done
sleep 30
log "night10 anti-collapse driver starting"
# x8 looser-kernel arms DEFERRED (owner 2026-08-22): the composed-winner
# choice waits for the factorial; x2 looks run early via cheap-anticollapse.
for arm in umap-md005-x2-fneg10-tanh4 umap-md010-x2-fneg10-tanh4; do
  [ -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ] && { log "arm $arm done, skip"; continue; }
  $PY experiments/sandbox/knobs_2m.py --arm "$arm" \
    >>"$LOGDIR/arm-$arm.log" 2>&1 && log "arm $arm DONE" || log "arm $arm FAILED (continuing)"
done
for ds in jina-en-2m jina-multi-2m; do
  log "=== STAGE $ds md005+tanh start"
  $PY experiments/sandbox/image_map_pipeline.py "$ds" train \
    >>"$LOGDIR/$ds.log" 2>&1 && log "$ds new arm DONE" || log "$ds new arm FAILED"
done
$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night10 driver finished — GPU idle"
