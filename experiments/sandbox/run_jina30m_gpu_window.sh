#!/usr/bin/env bash
set -u
while systemctl --user is-active --quiet night16-efficiency.service; do sleep 120; done
sleep 30
systemctl --user stop jina30m-embed.service 2>/dev/null || true
DEVICE=cuda BATCH=256 TORCH_THREADS=8 \
  /home/enjalot/code/latent-basemap/.venv/bin/python \
  /home/enjalot/code/latent-data-modal/embed_jina30m_cpu.py
