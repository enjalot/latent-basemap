#!/usr/bin/env bash
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
UPY=/data/latent-basemap/umap06dev-env/bin/python
LOG=/data/latent-basemap/sandbox/logs/night17b.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
run() { local label=$1 lf=$2; shift 2
  "$@" >>"/data/latent-basemap/sandbox/logs/$lf" 2>&1 && log "$label DONE" || log "$label FAILED"; }
log "night17b starting"
run "6250k teacher (cpu)" distill-grid-work.log $UPY experiments/sandbox/distill_grid.py prep625
run "distill grid 6.25M" distill-grid-work.log $PY experiments/sandbox/distill_grid.py grid
run "jina-6m train (budget fix)" jina-6m.log $PY experiments/sandbox/image_map_pipeline.py jina-multi-6m train
$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night17b finished — GPU idle"
