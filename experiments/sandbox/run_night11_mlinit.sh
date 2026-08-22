#!/usr/bin/env bash
# Night driver #11: the faithful multilevel-init port arms. Waits for night10.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
LOG=/data/latent-basemap/sandbox/logs/night11-mlinit.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night10-anticollapse.service; do sleep 120; done
sleep 30
log "night11 mlinit driver starting"
$LB/.venv/bin/python experiments/sandbox/multilevel_train.py \
  >>/data/latent-basemap/sandbox/logs/mlinit.log 2>&1 \
  && log "mlinit arms DONE" || log "mlinit arms FAILED"
$LB/.venv/bin/python experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night11 driver finished — GPU idle"
