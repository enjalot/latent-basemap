#!/usr/bin/env bash
# Night driver #2 (owner orders 2026-08-22): waits for images-night to finish,
# then T1+T2 teacher distillation -> jina-en-2m suite -> jina-multi-2m suite
# -> review page. Same best-3 recipes for the jina suites.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
UPY=/data/latent-basemap/umap06dev-env/bin/python
LOGDIR=/data/latent-basemap/sandbox/logs
LOG=$LOGDIR/night2-20260822.log
mkdir -p "$LOGDIR"; cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
run() { local label=$1 lf=$2; shift 2
  "$@" >>"$LOGDIR/$lf" 2>&1 && log "$label DONE" || log "$label FAILED (continuing)"; }

while systemctl --user is-active --quiet images-night-20260821.service; do sleep 120; done
sleep 60
log "night2 driver starting (images-night finished)"

log "=== STAGE distill-T1T2 start"
run "distill T1+T2" distill.log $PY experiments/sandbox/distill_teacher_2m.py

for ds in jina-en-2m jina-multi-2m; do
  log "=== STAGE $ds start"
  if [ ! -f "/data/latent-basemap/sandbox/$ds/edges-k15-fuzzy.npz" ]; then
    run "$ds knn" "$ds.log" $PY experiments/sandbox/image_map_pipeline.py "$ds" knn
    run "$ds fuzzy" "$ds.log" $UPY experiments/sandbox/image_map_pipeline.py "$ds" fuzzy
  fi
  run "$ds train" "$ds.log" $PY experiments/sandbox/image_map_pipeline.py "$ds" train
done

log "=== STAGE review-page start"
$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night2 driver finished — GPU idle"
