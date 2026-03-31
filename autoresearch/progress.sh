#!/bin/bash
# Monitor autoresearch progress across all agents
# Usage: bash autoresearch/progress.sh
#        watch -n 30 bash autoresearch/progress.sh

echo "═══════════════════════════════════════════════════════════"
echo "  AUTORESEARCH PROGRESS — $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════"

for dir in /Users/enjalot/code/latent-basemap /Users/enjalot/code/latent-basemap-ar2 /Users/enjalot/code/latent-basemap-ar3; do
  tsv="$dir/autoresearch/results.tsv"
  if [ -f "$tsv" ]; then
    branch=$(cd "$dir" && git branch --show-current 2>/dev/null || echo "?")
    total=$(tail -n +2 "$tsv" | wc -l | tr -d ' ')
    kept=$(grep -c "keep" "$tsv" 2>/dev/null || echo 0)
    discarded=$(grep -c "discard" "$tsv" 2>/dev/null || echo 0)
    crashed=$(grep -c "crash" "$tsv" 2>/dev/null || echo 0)
    best=$(tail -n +2 "$tsv" | grep "keep" | awk -F'\t' '{print $2}' | sort -rn | head -1)
    latest=$(tail -1 "$tsv")
    echo ""
    echo "  Branch: $branch"
    echo "  Runs: $total total ($kept kept, $discarded discarded, $crashed crashed)"
    echo "  Best knn_10: ${best:-n/a}"
    echo "  Latest: $latest"
  fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"

# Show all results combined, sorted by knn_10
echo ""
echo "  TOP 10 ACROSS ALL AGENTS (by knn_10):"
echo "  ─────────────────────────────────────────────────────────"
(for dir in /Users/enjalot/code/latent-basemap /Users/enjalot/code/latent-basemap-ar2 /Users/enjalot/code/latent-basemap-ar3; do
  tsv="$dir/autoresearch/results.tsv"
  if [ -f "$tsv" ]; then
    tail -n +2 "$tsv" | grep "keep"
  fi
done) | sort -t$'\t' -k2 -rn | head -10 | while IFS=$'\t' read commit knn10 knn25 knn50 status desc; do
  printf "  %-8s knn_10=%-10s %s\n" "$commit" "$knn10" "$desc"
done

echo ""
