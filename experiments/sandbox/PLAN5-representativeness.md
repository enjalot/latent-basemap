# Sandbox phase 5: representativeness audit + register corpora (owner-approved 2026-08-14)

**Priority: BACKGROUND.** Never preempts the evidence chain, rounds, or any
GPU work; the GPU pieces here are minutes-scale and run in idle windows.
Network/CPU pieces (downloads, chunking) can run anytime.

## Why

all-MiniLM-L6-v2's geometry was shaped by ~1.17B contrastive pairs
(model card): **~62% Reddit conversational**, ~13% QA-format
(WikiAnswers/PAQ/GOOAQ/Yahoo), ~10% S2ORC scientific, ~6% Stack Exchange,
~1% MS MARCO, ~0.1% code. Our 100M mix (FineWeb-edu/RPJ/Pile/starcoder
40/25/25/10) covers formal prose and scientific registers well but has
**near-zero conversational and QA-format text** — the majority register of
the embedding model's own training — while over-weighting code ~100× and
including registers MiniLM never saw (legal/patent). The audit quantifies
this; the corpora it lands become available slices for future maps.

## Deliverable 1 — register corpora on /data (network+CPU, then ~1 GPU-h total)

Land three register datasets through the STANDARD pipeline (120-token chunks
— matching MiniLM's 128-token training truncation — then MiniLM embeddings),
in the standing shared layout so `datacat` sees them and future substrate
assembly can consume them directly:

```
/data/chunks/<name>-chunked-120/train/*.parquet
/data/embeddings/<name>-chunked-120-all-MiniLM-L6-v2/train/*.npy
```

| register | candidate sources (executor verifies availability/license on HF) | target size |
| --- | --- | --- |
| Reddit conversational | `webis/tldr-17`, `HuggingFaceGECLM/REDDIT_comments`, `sentence-transformers/reddit-title-body` | ~10M chunks (~15 GB fp32) |
| QA-format | PAQ, GOOAQ, `yahoo_answers_topics`, WikiAnswers (the embedding-training-data mirrors) | ~5M chunks |
| MS MARCO | `microsoft/ms_marco` — BOTH passages (~8.8M) AND queries (as their own short-chunk set) | passages ~8.8M + queries ~1M |

MiniLM embedding throughput on the 5090 makes the GPU cost trivial
(~10-20k chunks/s; the whole program is under ~1 GPU-h, run in idle
windows). Re-run `datacat scan` after landing. Total disk ~40-60 GB.

## Deliverable 2 — the audit (CPU + minutes of GPU)

Embedding-space first, so "mix doesn't cover it" separates from "model
doesn't represent it" before any map is involved:

1. **Coverage CDFs**: for a 20k sample per register, exact cosine
   nearest-neighbor distance into the sealed 100M substrate (chunked GPU
   brute, the heldout-truth pattern; minutes per register). Baseline: the
   substrate's own NN-distance distribution (free from the sealed k15
   graph). Report per-register CDFs + a summary quantile table. A register
   whose NN distances sit far right of the baseline is territory the mix
   never covered.
2. **TwoNN intrinsic dimension** per register (the program's only measured
   OOD-loss predictor) beside the coverage numbers.
3. **Map-level retention** (once the promoted map exists): project each
   register through the checkpoint, standard retention panel + collapse/fog
   on the projected cloud, cards on the kernels page aligned in the usual
   frame.

## Deliverable 3 — MS MARCO as a standing map eval (owner's design point)

MS MARCO is IN the embedding model's training (~1% of pairs) but NOT in our
corpus — so the model represents it well while the map never saw it. That
isolates exactly the quantity the projector product promises:
**map-coverage generalization** with model-competence held constant — the
MiniLM analog of held-out Polish on the U12 map. Registered as a standing
eval probe for every future MiniLM map: passages as the in-model/out-of-
corpus retention probe, queries as a second, harsher short-text register.
(Contrast probes that confound the two: Latin fails MiniLM itself.)

## Boundaries

Sandbox; no rounds, no capabilities; nothing here blocks or delays the
evidence chain, R0264, or the go/no-go. Licenses checked before landing
data (Reddit sources especially — record the license in the dataset dir).
If a source is unavailable or license-encumbered, substitute and document,
never scrape ad hoc. Log the landed assets in the data topic log +
`datacat` per standing convention.
