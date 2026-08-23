#!/usr/bin/env bash
# After night15, give idle GPU time to the 30M jina embed (DEVICE=cuda picks
# up the same resumable units ~50x faster than CPU). Stop anytime with
# `systemctl --user stop jina30m-gpu` — at most one 100K unit is lost.
set -uo pipefail
while systemctl --user is-active --quiet night15-fixups.service; do sleep 120; done
sleep 30
systemctl --user stop jina30m-embed.service 2>/dev/null || true
DEVICE=cuda BATCH=256 TORCH_THREADS=8 \
  /home/enjalot/code/latent-basemap/.venv/bin/python \
  /home/enjalot/code/latent-data-modal/embed_jina30m_cpu.py
