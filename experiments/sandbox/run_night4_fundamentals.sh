#!/usr/bin/env bash
# Night driver #4 (owner reprioritization 2026-08-22): fundamentals first.
# gap-closure sweep -> reddit register probe -> jina (embed + suites) last.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
UPY=/data/latent-basemap/umap06dev-env/bin/python
LOGDIR=/data/latent-basemap/sandbox/logs
LOG=$LOGDIR/night4-fundamentals.log
mkdir -p "$LOGDIR"; cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
run() { local label=$1 lf=$2; shift 2
  "$@" >>"$LOGDIR/$lf" 2>&1 && log "$label DONE" || log "$label FAILED (continuing)"; }

log "night4 fundamentals driver starting"

log "=== STAGE gap-closure start"
for arm in umap-md000-x2-fneg10-wes umap-md000-x2-fneg10-probt \
           umap-md000-x2-rankneg500k-fneg10 umap-md000-x2-fneg10-tanh4-pos02 \
           umap-md000-x2-fneg10-tanh4-pos10 umap-md000-x4-fneg10-tanh4 \
           umap-md000-x8-fneg10; do
  [ -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ] && { log "arm $arm done, skip"; continue; }
  run "arm $arm" "arm-$arm.log" $PY experiments/sandbox/knobs_2m.py --arm "$arm"
done

log "=== STAGE reddit start"
run "reddit embed" reddit-embed.log $PY /home/enjalot/code/latent-data-modal/embed_reddit_local.py
if [ ! -f /data/latent-basemap/sandbox/reddit-2m/edges-k15-fuzzy.npz ]; then
  run "reddit knn" reddit-graph.log $PY experiments/sandbox/image_map_pipeline.py reddit-2m knn
  run "reddit fuzzy" reddit-graph.log $UPY experiments/sandbox/image_map_pipeline.py reddit-2m fuzzy
fi
run "reddit ood probe" reddit-probe.log $PY experiments/sandbox/reddit_ood_probe.py

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
  log "jina embed incomplete -> jina suites SKIPPED (never train raw)"
fi

log "=== STAGE review-page start"
$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night4 driver finished — GPU idle"
