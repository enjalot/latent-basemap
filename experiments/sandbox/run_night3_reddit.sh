#!/usr/bin/env bash
# Night driver #3 (owner order 2026-08-22): waits for night2, then the reddit
# register work — embed 10M tldr-17 chunks -> 2M-sample graph -> OOD probe
# through the frozen canonical maps (+ tanh4 sandbox map).
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
UPY=/data/latent-basemap/umap06dev-env/bin/python
LOGDIR=/data/latent-basemap/sandbox/logs
LOG=$LOGDIR/night3-reddit.log
mkdir -p "$LOGDIR"; cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
run() { local label=$1 lf=$2; shift 2
  "$@" >>"$LOGDIR/$lf" 2>&1 && log "$label DONE" || log "$label FAILED (continuing)"; }

while systemctl --user is-active --quiet night2-20260822.service; do sleep 120; done
sleep 60
log "night3 reddit driver starting"

log "=== STAGE reddit-embed start"
run "reddit embed" reddit-embed.log $PY /home/enjalot/code/latent-data-modal/embed_reddit_local.py

log "=== STAGE reddit-graph start"
if [ ! -f /data/latent-basemap/sandbox/reddit-2m/edges-k15-fuzzy.npz ]; then
  run "reddit knn" reddit-graph.log $PY experiments/sandbox/image_map_pipeline.py reddit-2m knn
  run "reddit fuzzy" reddit-graph.log $UPY experiments/sandbox/image_map_pipeline.py reddit-2m fuzzy
fi

log "=== STAGE reddit-ood-probe start"
run "reddit ood probe" reddit-probe.log $PY experiments/sandbox/reddit_ood_probe.py

log "night3 reddit driver finished — GPU idle"
