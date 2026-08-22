#!/usr/bin/env bash
# Night driver #5: waits for night4, runs the T3 distill-init hybrid arms.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
LOG=/data/latent-basemap/sandbox/logs/night5-hybrid.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night4-fundamentals.service; do sleep 120; done
sleep 60
log "night5 hybrid driver starting"
log "=== STAGE distill-init-finetune start"
$LB/.venv/bin/python experiments/sandbox/distill_init_finetune.py \
  >>/data/latent-basemap/sandbox/logs/distill-init.log 2>&1 \
  && log "distill-init arms DONE" || log "distill-init arms FAILED"
$LB/.venv/bin/python experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night5 driver finished — GPU idle"
