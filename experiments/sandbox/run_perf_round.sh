#!/usr/bin/env bash
# Perf round (owner 2026-08-23): 10 throughput experiments (~40 min, each
# variant its own process so compile/tf32/patches can't leak) + the 2M
# reproducibility verification (~25 min). Slots between night11 and night12.
set -uo pipefail
LB=/home/enjalot/code/latent-basemap
PY=$LB/.venv/bin/python
LOG=/data/latent-basemap/sandbox/logs/perf-round.log
cd "$LB"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
while systemctl --user is-active --quiet night11-mlinit.service; do sleep 60; done
sleep 20
log "perf round starting"
for v in v00-baseline v01-quiet v02-fused-adamw v03-tf32 v04-compile \
         v05-compile-max v06-combo-safe v07-combo-compile v08-batch16k v09-batch32k; do
  [ -f "/data/latent-basemap/sandbox/perf-bench/$v.json" ] && { log "$v done, skip"; continue; }
  $PY experiments/sandbox/perf_bench.py "$v" \
    >>/data/latent-basemap/sandbox/logs/perf-bench.log 2>&1 \
    && log "$v DONE" || log "$v FAILED (continuing)"
done
log "=== verify-repro start"
$PY experiments/sandbox/verify_repro.py \
  >>/data/latent-basemap/sandbox/logs/perf-verify.log 2>&1 \
  && log "verify-repro DONE" || log "verify-repro FAILED"
log "perf round finished — GPU idle"
