# Map pack v1 — build report

2026-08-14 · GSV · CPU only · saved by the session agent from the build agent's inline report.

## Built

map_pack.py (build / sidecar / validate / validate-sidecar), verify_text_alignment.py, tests/ (22 pass).
Packs: /data/latent-basemap/mappacks/sandbox-2m-umap-md000-x4-fneg10 (148 MB, Z=3) and
sandbox-12500k-umap-md000-x2-fneg10 (381 MB, Z=4). Sidecars (per substrate):
2M 923 MB, 12.5M 5.7 GB. Publishable preview+LOD subsets: ~73 MB / ~230 MB.
Wall: pack geometry 2.7 s / 15.3 s; sidecars 251 s / 301 s (one-off per substrate).

## Text coverage: 100%, verified end-to-end

All four corpora's 120-token chunk text is local under /data/chunks (445/445 shards
row-matched vs embedding shards). Resolution chain: provenance (corpus,shard,row) ->
corpus codes DERIVED by (rows, shard-count) bijection vs substrate.json (refuses to
guess); shard index = position in the NON-excluded shard list (fineweb shard 37
exclusion established byte-identically); parquet<->npy stem mapping per the original
embed pipeline. POSITIVE CONTROL: re-embedding 256 sampled sidecar texts with MiniLM
reproduces the sealed substrate vectors at cosine 1.0000 min/median across all
corpora — sidecar text for row i is exactly what produced vector i.

Caveat for viewers: chunk_text is the detokenized (bert-uncased, lowercased) string
that was embedded — not the verbatim source doc. Parquets carry url/doc_id for a
future "view source" sidecar.

## Contract deviations/additions (recorded in each manifest)

Squared extent about the trimmed core; floor-quantization q=floor(u*65536) so bin
at level z = q >> (8-z) (validator re-derives tiles from points alone); zoom rule
(256*2^z)^2 >= N cap 5; sort = row-major finest tile then in-tile Morton; EMPTY
TILES/PLANES OMITTED with density/z{z}/index.json inventory per level (saves
~250 MB of zeros at 12.5M — viewers fetch index.json, never 404-probe);
priority-based deterministic reservoir K=4; LOD stratified at deepest fitting grid
with min_zoom byte offsets in the manifest. Not built (deliberate): bins TF-IDF
terms (PLAN6 item 4), deeper bin levels, source-metadata sidecar, published-profile
packer.

## Validation

validate --full re-derives every invariant from files alone (inventory/hashes,
tile_index delimits exactly the right points, sort order, corpus-bit round-trip,
density planes sum to N at every level + bin-for-bin tile reconstruction, LOD
consistency, samples land in their bins). Planted-defect controls: wrong-order
sort, shifted index, corrupted plane, tampered hash, missing file, mismatched
corpus counts — all caught. Both packs and both sidecars: PASS.

/data at 89% (395 GB free) after sidecars.
