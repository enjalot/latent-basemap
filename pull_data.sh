#!/usr/bin/env bash
# Pull latent-basemap data from Modal volumes onto /data for local training.
#
# Pulls enough shards per dataset to cover N_ROWS_PER_DATASET (default 50M),
# using `modal volume ls --json` to discover actual shard names — the three
# datasets have different totals:
#   fineweb:   99 shards (of-00099), ~94M rows total
#   RedPajama: 150 shards (of-00150), ~84M rows total
#   pile:      177 present shards (named of-01987), rows vary
#
# Shared layout on disk (embeddings are consumed by both latent-basemap and
# latent-sae, so they live at /data/embeddings, not under any project dir):
#   /data/embeddings/<dataset>/train/*.npy
#   /data/checkpoints/pumap/faiss_ivf_pq_150m.index
#   /data/checkpoints/pumap/edges_150m_k15.npz
#
# Idempotent: skips shards already present on disk with nonzero size.
# Re-run to resume after an interruption.

set -uo pipefail

EMB_DIR="${EMB_DIR:-/data/embeddings}"
CKPT_ROOT="${CKPT_ROOT:-/data/checkpoints}"
N_ROWS_PER_DATASET="${N_ROWS_PER_DATASET:-50000000}"
DIM=384                          # all-MiniLM-L6-v2 output dim
BYTES_PER_ROW=$((DIM * 4))       # float32
SKIP_EMBEDDINGS="${SKIP_EMBEDDINGS:-0}"
SKIP_CHECKPOINTS="${SKIP_CHECKPOINTS:-0}"

DATASETS=(
  "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2"
  "RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2"
  "pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2"
)

CKPT_DIR="$CKPT_ROOT/pumap"
mkdir -p "$EMB_DIR" "$CKPT_DIR"

TARGET_BYTES=$((N_ROWS_PER_DATASET * BYTES_PER_ROW))
echo "== latent-basemap data pull =="
echo "EMB_DIR             = $EMB_DIR"
echo "CKPT_DIR            = $CKPT_DIR"
echo "N_ROWS_PER_DATASET  = $N_ROWS_PER_DATASET  (~$(python3 -c "print(f'{$TARGET_BYTES/1024**3:.0f}')") GiB/dataset)"
echo "datasets            = ${#DATASETS[@]}"
echo

# For a dataset, emit the list of shard filenames (one per line) — in
# ascending index order — whose cumulative size reaches TARGET_BYTES.
select_shards() {
  local ds="$1"
  modal volume ls --json embeddings "${ds}/train" 2>/dev/null | python3 -c "
import json, sys, re
data = json.load(sys.stdin)
def to_bytes(s):
    if not isinstance(s, str): return 0
    v, u = s.split()
    v = float(v)
    return int(v * {'B':1,'KiB':1024,'MiB':1024**2,'GiB':1024**3,'TiB':1024**4}[u])
def shard_idx(fn):
    m = re.search(r'data-(\d+)-of-', fn)
    return int(m.group(1)) if m else 10**9
items = [(shard_idx(e['Filename']), e['Filename'], to_bytes(e.get('Size','0 B')))
         for e in data if e.get('Filename','').endswith('.npy')]
items.sort()
target = $TARGET_BYTES
acc = 0
for _, fn, sz in items:
    if acc >= target: break
    print(fn)
    acc += sz
"
}

pull_shard() {
  local remote="$1"
  local local_path="$EMB_DIR/$remote"
  mkdir -p "$(dirname "$local_path")"
  if [[ -s "$local_path" ]]; then
    return 0
  fi
  echo "  get $remote"
  modal volume get embeddings "$remote" "$local_path"
}

if [[ "$SKIP_EMBEDDINGS" != "1" ]]; then
  for ds in "${DATASETS[@]}"; do
    echo "[$ds]"
    shards=$(select_shards "$ds")
    n=$(echo "$shards" | grep -c . || true)
    echo "  selecting $n shards for ~${N_ROWS_PER_DATASET} rows"
    while IFS= read -r fn; do
      [[ -z "$fn" ]] && continue
      pull_shard "$fn" || { echo "  FAILED: $fn"; exit 1; }
    done <<< "$shards"
    du -sh "$EMB_DIR/$ds" 2>/dev/null || true
    echo
  done
fi

if [[ "$SKIP_CHECKPOINTS" != "1" ]]; then
  echo "[checkpoints/pumap]"
  for fname in "faiss_ivf_pq_150m.index" "edges_150m_k15.npz"; do
    local_path="$CKPT_DIR/$fname"
    if [[ -s "$local_path" ]]; then
      echo "  skip $fname (exists)"
      continue
    fi
    echo "  get pumap/$fname"
    modal volume get checkpoints "pumap/$fname" "$local_path" || {
      echo "  FAILED: pumap/$fname"; exit 1; }
  done
  du -sh "$CKPT_DIR" 2>/dev/null || true
  echo
fi

echo "== done =="
du -sh "$EMB_DIR" "$CKPT_ROOT" 2>/dev/null || true
