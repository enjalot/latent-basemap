#!/usr/bin/env bash
# Night driver #16: the best-efficiency guess on MiniLM 2M + jina-en +
# jina-multi + sisap. Waits for night15.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
LOG=/data/latent-basemap/sandbox/logs/night16-efficiency.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night15-fixups.service; do sleep 120; done
sleep 30
log "night16 efficiency driver starting"
arm=umap-md010-h1024L2mlp-bs16k-x4-winner
if [ ! -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ]; then
  $LB/.venv/bin/python experiments/sandbox/knobs_2m.py --arm "$arm" \
    >>"/data/latent-basemap/sandbox/logs/arm-$arm.log" 2>&1 \
    && log "arm $arm DONE" || log "arm $arm FAILED"
fi
for ds in jina-en-2m jina-multi-2m sisap-clip-2m; do
  log "=== STAGE $ds efficiency-x4 start"
  $LB/.venv/bin/python experiments/sandbox/image_map_pipeline.py "$ds" train \
    >>"/data/latent-basemap/sandbox/logs/$ds.log" 2>&1 \
    && log "$ds DONE" || log "$ds FAILED (continuing)"
done
$LB/.venv/bin/python experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true
log "night16 finished — GPU idle"
