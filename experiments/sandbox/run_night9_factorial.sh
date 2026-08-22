#!/usr/bin/env bash
# Night driver #9 (external-review plan 2026-08-22): dose/composition
# factorial controls + distill-init composed-core. Waits for night8.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
LOGDIR=/data/latent-basemap/sandbox/logs
LOG=$LOGDIR/night9-factorial.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night8-registers.service; do sleep 120; done
sleep 30
log "night9 factorial driver starting"
for arm in umap-md000-x2-fneg10-pos10 umap-md000-x4-fneg10-tanh2 \
           umap-md000-x4-fneg10-tanh4-pos10 umap-md000-x8-fneg10-tanh4; do
  [ -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ] && { log "arm $arm done, skip"; continue; }
  $PY experiments/sandbox/knobs_2m.py --arm "$arm" \
    >>"$LOGDIR/arm-$arm.log" 2>&1 && log "arm $arm DONE" || log "arm $arm FAILED (continuing)"
done
log "=== STAGE distillinit-core start"
$PY experiments/sandbox/distill_init_core.py \
  >>"$LOGDIR/distillinit-core.log" 2>&1 && log "distillinit-core DONE" || log "distillinit-core FAILED"
$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night9 driver finished — GPU idle"
