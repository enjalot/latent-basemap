#!/usr/bin/env bash
# Perf bench v2 (delta-method) — waits for night13.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
LOG=/data/latent-basemap/sandbox/logs/perf-round2.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
log "perf round 2 starting"
mkdir -p /data/latent-basemap/sandbox/perf-bench2
for v in v00-baseline v01-quiet v02-fused-adamw v03-tf32 v04-compile \
         v05-compile-max v06-combo-safe v07-combo-compile v08-batch16k v09-batch32k; do
  [ -f "/data/latent-basemap/sandbox/perf-bench2/$v.json" ] && continue
  PERF_OUT=/data/latent-basemap/sandbox/perf-bench2 \
  $LB/.venv/bin/python experiments/sandbox/perf_bench.py "$v" \
    >>/data/latent-basemap/sandbox/logs/perf-bench2.log 2>&1 \
    && log "$v DONE" || log "$v FAILED (continuing)"
done
log "perf round 2 finished"
