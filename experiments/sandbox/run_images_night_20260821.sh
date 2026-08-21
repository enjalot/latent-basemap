#!/usr/bin/env bash
# Overnight driver 2026-08-21 (owner orders): BL SigLIP 1.08M suite -> SISAP
# CLIP 2M suite -> gamma mini-sweep + tanh4 seed replicates -> review page.
# Best-3 recipes per image dataset: promoted-fneg10 / fneg10-tanh4 / md005-fneg10.
set -uo pipefail

LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
UPY=/data/latent-basemap/umap06dev-env/bin/python
LOGDIR=/data/latent-basemap/sandbox/logs
LOG=$LOGDIR/images-night-20260821.log
mkdir -p "$LOGDIR"
cd "$LB"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
stage() { log "=== STAGE $1 start"; }

run() { # run <label> <logfile> <cmd...>
  local label=$1 lf=$2; shift 2
  "$@" >>"$LOGDIR/$lf" 2>&1 && log "$label DONE" || log "$label FAILED (continuing)"
}

log "images-night driver starting"

for ds in bl-siglip-1m sisap-clip-2m; do
  stage "$ds"
  if [ ! -f "/data/latent-basemap/sandbox/$ds/edges-k15-fuzzy.npz" ]; then
    run "$ds knn" "$ds.log" $PY experiments/sandbox/image_map_pipeline.py "$ds" knn
    run "$ds fuzzy" "$ds.log" $UPY experiments/sandbox/image_map_pipeline.py "$ds" fuzzy
  else
    log "$ds graph exists, skip"
  fi
  run "$ds train" "$ds.log" $PY experiments/sandbox/image_map_pipeline.py "$ds" train
done

stage gamma-sweep
for arm in umap-md000-x2-fneg10-tanh2 umap-md000-x2-fneg10-tanh8; do
  [ -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ] && { log "arm $arm done, skip"; continue; }
  run "arm $arm" "arm-$arm.log" $PY experiments/sandbox/knobs_2m.py --arm "$arm"
done
for seed in 43 44; do
  [ -f "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x2-fneg10-tanh4-seed$seed/summary.json" ] && { log "tanh4 seed$seed done, skip"; continue; }
  run "tanh4 seed$seed" "arm-tanh4-seed$seed.log" \
    $PY experiments/sandbox/knobs_2m.py --arm umap-md000-x2-fneg10-tanh4 --seed "$seed"
done

stage review-page
$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "images-night driver finished — GPU idle"
