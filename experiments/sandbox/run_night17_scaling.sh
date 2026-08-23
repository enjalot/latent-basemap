#!/usr/bin/env bash
# Scaling night (owner 2026-08-23): 6.25M for MiniLM + jina-multi.
# GPU-serial: minilm champion@6.25M -> distill-grid 6.25M anchor -> jina
# prompted top-up embed -> jina 6.25M graph -> jina champion -> jina
# efficiency -> review. 6.25M teacher preps on CPU in parallel at start.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
UPY=/data/latent-basemap/umap06dev-env/bin/python
LOGDIR=/data/latent-basemap/sandbox/logs
LOG=$LOGDIR/night17-scaling.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
run() { local label=$1 lf=$2; shift 2
  "$@" >>"$LOGDIR/$lf" 2>&1 && log "$label DONE" || log "$label FAILED (continuing)"; }

while systemctl --user is-active --quiet decomp-after-grid.service; do sleep 60; done
systemctl --user stop jina30m-gpu2.service 2>/dev/null || true
sleep 10
log "night17 scaling driver starting"

# CPU teacher for the 6.25M distill anchor, in parallel with GPU work
systemd-run --user --unit=distill625-teacher -p Nice=10 \
  "$UPY" experiments/sandbox/distill_grid.py prep625 2>/dev/null || true

log "=== STAGE minilm-6250k champion start"
if [ ! -f /data/latent-basemap/sandbox/6250k-knobs/umap-md000-x4bs16k-winner/summary.json ]; then
  run "minilm 6.25M champion-bs16k" arm-6250k-champion.log \
    $PY experiments/sandbox/knobs_2m.py --rung 6250k --arm umap-md000-x4bs16k-winner
fi

log "=== STAGE distill-grid-625 start"
for i in $(seq 1 60); do
  [ -f /data/latent-basemap/sandbox/distill-grid/subset-6250k/teacher.npy ] && break
  sleep 60
done
run "distill grid 6.25M cells" distill-grid-work.log \
  $PY experiments/sandbox/distill_grid.py grid

log "=== STAGE jina-625-topup start"
run "jina prompted top-up" jina625-embed.log \
  $PY /home/enjalot/code/latent-data-modal/embed_jina_625_topup.py

if [ -f /data/latent-basemap/substrates/jina-prompted/substrate-6250k.f16.npy ]; then
  log "=== STAGE jina-multi-6m graph start"
  if [ ! -f /data/latent-basemap/sandbox/jina-multi-6m/edges-k15-fuzzy.npz ]; then
    run "jina-6m knn" jina-6m.log $PY experiments/sandbox/image_map_pipeline.py jina-multi-6m knn
    run "jina-6m fuzzy" jina-6m.log $UPY experiments/sandbox/image_map_pipeline.py jina-multi-6m fuzzy
  fi
  log "=== STAGE jina-multi-6m train start"
  run "jina-6m train (champion + efficiency)" jina-6m.log \
    $PY experiments/sandbox/image_map_pipeline.py jina-multi-6m train
else
  log "jina 6.25M substrate missing -> jina stages SKIPPED"
fi

$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night17 finished — GPU idle (jina30m embed may resume on owner call)"
