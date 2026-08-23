#!/usr/bin/env bash
set -uo pipefail
LOG=/data/latent-basemap/sandbox/logs/distill-grid.log
cd /home/enjalot/code/latent-basemap
printf '%s grid rerun starting\n' "$(date -u +%FT%TZ)" >>"$LOG"
.venv/bin/python experiments/sandbox/distill_grid.py grid \
  >>/data/latent-basemap/sandbox/logs/distill-grid-work.log 2>&1 \
  && printf '%s grid DONE\n' "$(date -u +%FT%TZ)" >>"$LOG" \
  || printf '%s grid FAILED\n' "$(date -u +%FT%TZ)" >>"$LOG"
systemctl --user stop jina30m-embed.service 2>/dev/null || true
DEVICE=cuda BATCH=256 TORCH_THREADS=8 \
  .venv/bin/python /home/enjalot/code/latent-data-modal/embed_jina30m_cpu.py
