#!/usr/bin/env bash
# Night driver #6: hybrid + composed-dose fundamentals, then jina last.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
UPY=/data/latent-basemap/umap06dev-env/bin/python
LOGDIR=/data/latent-basemap/sandbox/logs
LOG=$LOGDIR/night6-final.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
run() { local label=$1 lf=$2; shift 2
  "$@" >>"$LOGDIR/$lf" 2>&1 && log "$label DONE" || log "$label FAILED (continuing)"; }

log "night6 driver starting"
log "=== STAGE hybrid start"
run "distill-init arms" distill-init.log $PY experiments/sandbox/distill_init_finetune.py

log "=== STAGE composed-dose start"
if [ ! -f /data/latent-basemap/sandbox/2m-knobs/umap-md000-x8-fneg10-tanh4-pos10/summary.json ]; then
  run "arm x8-tanh4-pos10" arm-x8-composed.log \
    $PY experiments/sandbox/knobs_2m.py --arm umap-md000-x8-fneg10-tanh4-pos10
fi

log "=== STAGE jina-prompted-embed start"
run "jina prompted embed" jina-embed.log \
  $PY /home/enjalot/code/latent-data-modal/embed_jina_prompted_subsets.py
if [ -f /data/latent-basemap/substrates/jina-prompted/manifest.json ]; then
  for ds in jina-en-2m jina-multi-2m; do
    log "=== STAGE $ds start"
    if [ ! -f "/data/latent-basemap/sandbox/$ds/edges-k15-fuzzy.npz" ]; then
      run "$ds knn" "$ds.log" $PY experiments/sandbox/image_map_pipeline.py "$ds" knn
      run "$ds fuzzy" "$ds.log" $UPY experiments/sandbox/image_map_pipeline.py "$ds" fuzzy
    fi
    run "$ds train" "$ds.log" $PY experiments/sandbox/image_map_pipeline.py "$ds" train
  done
else
  log "jina embed incomplete -> jina suites SKIPPED"
fi
log "=== STAGE review-page start"
$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night6 driver finished — GPU idle"
