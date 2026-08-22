#!/usr/bin/env bash
# Night driver #8 (owner priorities 2026-08-22, post-B2-cancellation):
# community-archive embed+probe -> redditmix 2M (promoted + composed-x8) ->
# dose x16 composed -> review page. Waits for night7 (composition screen).
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
UPY=/data/latent-basemap/umap06dev-env/bin/python
LOGDIR=/data/latent-basemap/sandbox/logs
LOG=$LOGDIR/night8-registers.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
run() { local label=$1 lf=$2; shift 2
  "$@" >>"$LOGDIR/$lf" 2>&1 && log "$label DONE" || log "$label FAILED (continuing)"; }

while systemctl --user is-active --quiet cheap-anticollapse.service; do sleep 120; done
sleep 30
log "night8 registers driver starting"

log "=== STAGE community-archive start"
for i in $(seq 1 240); do
  [ -f /data/chunks/communityarchive-tweets/manifest.json ] && break
  sleep 60
done
if [ -f /data/chunks/communityarchive-tweets/manifest.json ]; then
  run "ca embed" ca-embed.log $PY /home/enjalot/code/latent-data-modal/embed_ca_local.py
  if [ ! -f /data/latent-basemap/sandbox/communityarchive-2m/edges-k15-fuzzy.npz ]; then
    run "ca knn" ca-graph.log $PY experiments/sandbox/image_map_pipeline.py communityarchive-2m knn
    run "ca fuzzy" ca-graph.log $UPY experiments/sandbox/image_map_pipeline.py communityarchive-2m fuzzy
  fi
  run "ca ood probe" ca-probe.log $PY experiments/sandbox/register_ood_probe.py communityarchive-2m
else
  log "ca pull never finished -> CA stage SKIPPED"
fi

log "=== STAGE redditmix start"
run "redditmix substrate" redditmix.log $PY experiments/sandbox/build_redditmix_substrate.py
if [ ! -f /data/latent-basemap/sandbox/minilm-redditmix-2m/edges-k15-fuzzy.npz ]; then
  run "redditmix knn" redditmix.log $PY experiments/sandbox/image_map_pipeline.py minilm-redditmix-2m knn
  run "redditmix fuzzy" redditmix.log $UPY experiments/sandbox/image_map_pipeline.py minilm-redditmix-2m fuzzy
fi
run "redditmix train" redditmix.log $PY experiments/sandbox/image_map_pipeline.py minilm-redditmix-2m train
run "redditmix reddit-probe" redditmix.log $PY experiments/sandbox/register_ood_probe.py reddit-2m
[ -f /data/latent-basemap/sandbox/communityarchive-2m/edges-k15-fuzzy.npz ] && \
  run "redditmix ca-probe" redditmix.log $PY experiments/sandbox/register_ood_probe.py communityarchive-2m

log "=== STAGE dose-x16 start"
if [ ! -f /data/latent-basemap/sandbox/2m-knobs/umap-md000-x16-fneg10-tanh4-pos10/summary.json ]; then
  run "arm x16 composed" arm-x16.log \
    $PY experiments/sandbox/knobs_2m.py --arm umap-md000-x16-fneg10-tanh4-pos10
fi

log "=== STAGE review-page start"
$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night8 driver finished — GPU idle"
