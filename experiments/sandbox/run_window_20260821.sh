#!/usr/bin/env bash
# GPU-window driver 2026-08-21 (owner-ordered sequence): wiki embed -> 5
# aesthetic-cross arms -> ParamRepulsor 2M -> 0.6dev sweep -> review page.
# BL SigLIP image work is deliberately NOT here (owner: "then we will move to
# images" — interactive, after this sequence).
#
# Launched automatically by window-waiter when round0268-seed43.service goes
# down (B1 seal). Stages continue past individual failures; everything logs to
# $LOG. Idempotent-ish: finished wiki shards skip; finished arms are re-run
# only if their summary.json is missing.
set -uo pipefail

LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
PRPY=/data/latent-basemap/paramrepulsor-env/bin/python
LOGDIR=/data/latent-basemap/sandbox/logs
LOG=$LOGDIR/window-20260821.log
mkdir -p "$LOGDIR"
cd "$LB"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
stage() { log "=== STAGE $1 start"; }

log "window driver starting (seed43 unit down; GPU assumed free)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >>"$LOG" 2>&1

# ---- 1. wikipedia MiniLM embed (resumable per shard) ------------------------
stage wiki-embed
$PY /home/enjalot/code/latent-data-modal/embed_wikipedia_local.py \
  >>"$LOGDIR/wiki-embed.log" 2>&1 \
  && log "wiki embed DONE" || log "wiki embed FAILED (continuing)"

# ---- 2. aesthetic x fneg cross (owner picks) --------------------------------
stage aesthetic-arms
for arm in umap-md005-x2-fneg10 umap-md020-x2-fneg10 \
           gc-a2-md000-x2-fneg10 gc-a2-md005-x2-fneg10 gc-a2-md020-x2-fneg10; do
  if [ -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ]; then
    log "arm $arm already done, skip"; continue
  fi
  $PY experiments/sandbox/knobs_2m.py --arm "$arm" \
    >>"$LOGDIR/arm-$arm.log" 2>&1 \
    && log "arm $arm DONE" || log "arm $arm FAILED (continuing)"
done

# ---- 3. ParamRepulsor 2M external baseline ----------------------------------
stage paramrepulsor
if [ -f /data/latent-basemap/sandbox/2m-knobs/paramrepulsor-upstream/summary.json ]; then
  log "paramrepulsor already done, skip"
else
  $PRPY experiments/sandbox/paramrepulsor_2m_sandbox.py fit \
    >>"$LOGDIR/paramrepulsor.log" 2>&1 \
    && $PY experiments/sandbox/paramrepulsor_2m_sandbox.py score \
      >>"$LOGDIR/paramrepulsor.log" 2>&1 \
    && log "paramrepulsor DONE" || log "paramrepulsor FAILED (continuing)"
fi

# ---- 4. umap-0.6dev sweep ---------------------------------------------------
stage 06dev-sweep
for arm in umap-md000-x2-rankneg100k umap-md000-x2-rankneg200k \
           umap-md000-x2-rankneg500k umap-md000-x2-rankneg200k-xn \
           umap-md000-x2-rankneg200k-fneg10 umap-md000-x2-fneg10-tanh4 \
           umap-md000-x2-fneg10-anneal25; do
  if [ -f "/data/latent-basemap/sandbox/2m-knobs/$arm/summary.json" ]; then
    log "arm $arm already done, skip"; continue
  fi
  $PY experiments/sandbox/knobs_2m.py --arm "$arm" \
    >>"$LOGDIR/arm-$arm.log" 2>&1 \
    && log "arm $arm DONE" || log "arm $arm FAILED (continuing)"
done

# ---- 5. refresh the review page --------------------------------------------
stage review-page
$PY experiments/build_sandbox_review.py >>"$LOG" 2>&1 || true

log "window driver finished — GPU idle; BL SigLIP phase + B2 release are owner calls"
