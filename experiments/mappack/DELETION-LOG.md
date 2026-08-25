# Deletion execution log — `gsv:/data` disk reclaim

Executes **T1 + T2-A + T2-B only** of `DELETION-PLAN.md` (2026-08-14).
**T2-C is HELD** by owner instruction — not touched.
**T3 is the keep-set** — not touched.

Execution date: **2026-08-14**. Machine: `gsv`. No `sudo` used.

Rules applied (from the plan and the execution brief):

- Every item's recorded verification command re-run **before** deletion.
- Any mismatch / unexpected external grep hit / missing verification ⇒ item
  **SKIPPED-BLOCKED**, never deleted, always logged.
- T2 items: `RECOVERY-*.md` note written **before** any byte removed.
- **Never deleted:** `*.json` receipts, manifests, `*.log`, `*.md`. Bulk payloads
  only (`.npy` / `.npz` / `.pt` / `.index` / `.i8` / `.bin` / `.pkl` / `.parquet`).
- Every path confirmed under `/data/latent-basemap/` or `/data/checkpoints/`
  before `rm`.

---

## df snapshot — START

```
Filesystem                       Size  Used Avail Use% Mounted on
/dev/mapper/ubuntu--vg-data--lv  3.4T  2.9T  394G  89% /data
```

Avail at start: **422,991,106,048 B = 393.9 GiB**.

> Note: the plan measured 401.2 GiB free on 2026-08-14. The volume has since
> lost ~7.3 GiB to an in-flight GPU round. Deltas below are computed from this
> 393.9 GiB baseline, not the plan's.

---

## Tier 1 — failed-queue bulk payloads

### Verification pass 1 — zero-citation greps (plan §3, second block)

Re-ran the plan's grep block verbatim. **It did not return "no hits" as the plan
predicted.** Every one of the six patterns returned at least one hit. Each hit
was then characterised:

| pattern | hits | character of hit |
| --- | --- | --- |
| `round-0224/queue-correction-1` | own `queue.json` | **tautological self-match** — the queue manifest that lives *inside* the directory being pruned, declaring its own output paths. The plan explicitly retains this file. |
| `round-0224/queue/artifacts/minilm-mixed-16m` | own `queue.json` | tautological self-match |
| `round-0237/queue/artifacts/minilm-mixed-50000k-cluster-spill-k15-fuzzy` | own `queue.json` | tautological self-match |
| `round-0216/queue-correction-1` | own `queue.json` **+ `latent-labs/basemap-100m/round-0216-2026-08-08.md:133`** | **genuine external document hit** |
| `round-0223/queue-correction-1` | own `queue.json` **+ `latent-labs/basemap-100m/result-0223-2026-08-08.md`** | **genuine external hit — sealed hash table naming the exact files** |
| `round-0223/queue-correction-2` | own `queue.json` **+ `result-0223-2026-08-08.md`** | **genuine external hit — sealed hash table naming the exact files** |

A self-match is structurally guaranteed for *every* queue directory on the
volume (each `queue.json` declares its own artifact paths), so it carries no
citation information; the plan's own §2 method states citation is resolved at
**file granularity, never directory granularity** (finding F2). Genuine external
hits were treated as admission failures.

### Verification pass 2 — file-granularity citation counts

Exact bulk filenames grepped across `latent-labs/**/*.md`, `maps.json`,
`registry-history/`, `sandbox/`, `release/`, `canonical/`, and all
`runs/*/queue*/queue.json`:

| file | external citations |
| --- | ---: |
| `round-0224/queue-correction-1/…/substrate.f32.npy` | **0** |
| `round-0224/queue/…/substrate.f32.npy` | **0** |
| `round-0237/queue/…/edges-k15-fuzzy.npz` | **0** |
| `round-0237/queue/…/graph-k15-ids.i32.npy` | **0** |
| `round-0216/queue-correction-1/…/substrate.f32.npy` | **0** |
| `round-0216/queue-correction-1/…/edges-k15-fuzzy.npz` | **0** |
| `round-0216/queue-correction-1/…/provenance.npy` | **0** |
| `round-0223/queue-correction-1/…/edges-k15-fuzzy.npz` | **1** |
| `round-0223/queue-correction-1/…/cuvs-k15-ids.i32.npy` | **1** |
| `round-0223/queue-correction-2/…/edges-k15-fuzzy.npz` | **1** |
| `round-0223/queue-correction-2/…/cuvs-k15-ids.i32.npy` | **1** |

External `queue.json` references from any *other* round: **none for any item.**

### Verification pass 3 — `cmp` against retained siblings (plan §3, first block)

Ran the plan's `cmp` block verbatim, 2026-08-14 15:50–15:51 UTC (11:50 EDT):

```
FAIL-1   round-0224 correction-2 vs correction-1 → differ: byte 3201, line 7
OK-2     round-0224 correction-2 vs queue        → identical
KEEP-sha256: a904e3741e48cb8f2d9199da90327001412ce027c7d109b454c3d536a8d37c00
OK-3     round-0237 correction-1 vs queue, edges-k15-fuzzy.npz   → identical
OK-4     round-0237 correction-1 vs queue, graph-k15-ids.i32.npy → identical
DIFF-216-substrate.f32.npy      → differ: byte 29313
DIFF-216-edges-k15-fuzzy.npz    → differ: byte 15 (sizes differ: 579,640,252 vs 580,136,932)
DIFF-216-provenance.npy         → differ: byte 405
OK-c1    round-0223 correction-3 vs correction-1 → identical
OK-c2    round-0223 correction-3 vs correction-2 → identical
```

The retained R0224 `correction-2` copy re-hashes to
`a904e3741e48cb8f…`, **matching the sealed hash** the plan recorded and the
`expected_inputs` of R0226 / R0227 / R0232.

### T1 item disposition

| # | item | GiB | action | reason |
| --: | --- | ---: | --- | --- |
| 1 | `round-0224/queue-correction-1/…/substrate.f32.npy` | 22.89 | **SKIPPED-BLOCKED** | **`cmp` FAILED** — differs from `correction-2` at byte 3201. Plan §3/§7: items 1–4 must not be deleted if their `cmp` fails, because their justification rests on a retained byte-identical twin. This copy is **unique unreceipted content**, not a duplicate. |
| 2 | `round-0224/queue/…/substrate.f32.npy` | 22.89 | **DELETED** | `cmp` OK; sealed hash survives in retained twin; 0 external file-level citations. |
| 3 | `round-0237/queue/…/edges-k15-fuzzy.npz` | 14.03 | **DELETED** | `cmp` OK; retained twin holds the `qualified-graph.json` receipt; 0 external citations. |
| 4 | `round-0237/queue/…/graph-k15-ids.i32.npy` | 2.79 | **DELETED** | `cmp` OK; survives in two retained copies; 0 external citations. |
| 5 | `round-0216/queue-correction-1/…/substrate.f32.npy` | 2.86 | **SKIPPED-BLOCKED** | Plan's zero-citation grep returned a **genuine external hit**: `latent-labs/basemap-100m/round-0216-2026-08-08.md:133` names the directory. Plan predicted no hits. `cmp` also differs (expected). Blocked under the "any unexpected grep hit ⇒ skip" rule. |
| 6 | `round-0216/queue-correction-1/…/edges-k15-fuzzy.npz` | 0.54 | **SKIPPED-BLOCKED** | same external hit |
| 7 | `round-0216/queue-correction-1/…/provenance.npy` | 0.02 | **SKIPPED-BLOCKED** | same external hit |
| 8 | `round-0223/queue-correction-1/…/{edges-k15-fuzzy.npz,cuvs-k15-ids.i32.npy}` | 0.65 | **SKIPPED-BLOCKED** | `result-0223-2026-08-08.md` lines 747–749 carry a **sealed hash table naming these exact files by full path, with byte counts and sha256**. That is a strategy-1 *and* strategy-2 citation at file granularity, so the item fails the plan's own T1 admission standard. `cmp` passed, but citation gates first. |
| 9 | `round-0223/queue-correction-2/…/{edges-k15-fuzzy.npz,cuvs-k15-ids.i32.npy}` | 0.65 | **SKIPPED-BLOCKED** | same — `result-0223` lines 599–601 |

### T1 deletions executed

```
deleted 24576000128  round-0224/queue/artifacts/minilm-mixed-16m-benchmark-substrate-v1/substrate.f32.npy
deleted 15061097304  round-0237/queue/artifacts/minilm-mixed-50000k-cluster-spill-k15-fuzzy-graph-v1/edges-k15-fuzzy.npz
deleted  3000000128  round-0237/queue/artifacts/minilm-mixed-50000k-cluster-spill-k15-fuzzy-graph-v1/graph-k15-ids.i32.npy
```

**T1 bytes freed: 42,637,097,560 B = 39.71 GiB** (plan estimate 67.32 GiB;
shortfall 27.61 GiB is the six blocked items).

Recovery notes written **before** deletion:
`runs/round-0224/RECOVERY-2026-08-14-failed-queue-bulk.md`,
`runs/round-0237/RECOVERY-2026-08-14-failed-queue-bulk.md`.

No `.json`, `.md` or `.log` was removed. `provenance.npy` and
`benchmark-substrate.json` in `round-0224/queue/…` were retained (not in the
plan's item spec).

## df snapshot — after T1

```
/dev/mapper/ubuntu--vg-data--lv  3.4T  2.8T  434G  87% /data
```

Avail: **465,693,347,840 B = 433.7 GiB** (+39.8 GiB).

---

## Tier 2-A — Modal-era `checkpoints/pumap` artifacts

### Verification

| check | result |
| --- | --- |
| refs in `maps.json` + all `registry-history/*.json` | **0** for every candidate |
| `queue.json` refs, `faiss_ivf_pq_150m.index` | rounds 0036, 0049, 0050, 0053, 0058, 0059, 0060, 0066, 0072, 0077, 0086 — **all pre-R0200** ✓ |
| `queue.json` refs, `edges_150m_k15.npz` | **round-0215** ⚠️ |
| `queue.json` refs, 15m/3m/1M indices + edges | **none** ✓ |
| `sandbox`/`release`/`canonical` refs | only Modal-era scripts inside the `release/round-0262/` code snapshot: `train_modal.py`, `psym_modal.py`, `edges_modal.py`, `scope_modal.py`, `edges_local.py`, `sweep_{global,structure,v3}_modal.py`, `bench_query_a100.py`, `pull_data.sh`, `docs/LOCAL_EXPERIMENTS.md`, `basemap/round00{36,49}_*.py` — exactly the citation classes the plan recorded ✓ |

### BLOCKED — `edges_150m_k15.npz`

The recovery-note draft asserts: *"No round ≥ R0200 declares any removed file as
an `expected_inputs` — checked against all 611 `queue.json` under `runs/`."*
**This is false for `edges_150m_k15.npz`.**
`runs/round-0215/queue/queue.json` declares it in `expected_inputs` of job
`forensic_v1_150m`:

```json
{"bytes": 7407631763,
 "canonical_path": "/data/checkpoints/pumap/edges_150m_k15.npz",
 "kind": "file",
 "sha256": "4cf448a05bfdc230f3a538dffaa641a1ab4969b075c7b0628a41fc2ee80d990a"}
```

R0215 ≥ R0200, and strategy 3 (DAG consumption) is the plan's own "load-bearing"
citation test. The item therefore belongs in T3, not T2-A.
**SKIPPED-BLOCKED — 7,407,631,763 B = 6.90 GiB retained.**

`faiss_ivf_pq_150m.index` was re-checked independently and is consumed **only**
by pre-R0200 queues, so it passed and was removed as approved.

### The 150M index seal turned out to be verifiable

The plan warned that deleting `faiss_ivf_pq_150m.index` would leave `result-0086`
"asserting a hash that can never again be checked" (it was flagged
"> rehash limit", declared-only). It was **rehashed at execution** and matches
exactly: `7ed8ba062baf148b9b076f84c0089849ddb42610f0566a7c197f4c80852893c1`.
`result-0086`'s seal is now confirmed *verified*, recorded permanently in the
recovery note. The evidentiary loss is smaller than the plan feared.

### T2-A executed

Recovery note written **before** deletion:
`gsv:/data/checkpoints/pumap/RECOVERY-2026-08-14-modal-era-pumap-checkpoints.md`
(carries all 19 sha256 values, computed pre-deletion).

19 files removed. **T2-A bytes freed: 14,674,289,006 B = 13.67 GiB**
(plan estimate 20.57 GiB; shortfall 6.90 GiB = the blocked `edges_150m_k15.npz`).

Retained in `checkpoints/pumap/`: 29 × `model_*.pt`,
`edges_30m_k15_fuzzy.npz.manifest.json`, `_wg_30m_build.log`,
`edges_30m_k15.npz`, `faiss_ivf_pq_30m.index`, `faiss_ivf_pq_3m.index`,
`edges_30m_k15_fuzzy.npz`, `edges_150m_k15.npz`. No `.json`/`.log` removed.

## df snapshot — after T2-A

Avail: **480,367,693,824 B = 447.4 GiB**.

---

## Tier 2-B — superseded jina / MiniLM testbed corpora

### Verification

| check | result |
| --- | --- |
| refs in `maps.json` + all `registry-history/*.json` | **0** for every directory |
| `queue.json` refs — `minilm-15m`, `jina-en-8M-nested`, `jina-en-6M-nested`, `jina-en-1M-nested`, `jina-en-200k-prompted`, `jina-babel-200k`, `jina-en-16m` | **none** ✓ |
| `queue.json` refs — `jina-en-8m` | round-0147 only — **pre-R0200** ✓ |
| `queue.json` refs — `jina-en-2m`, `jina-en-500k`, `jina-en-200k` | **R0206, R0175, R0179, R0181** ⚠️ |
| `jina-en-4M-nested`, `jina-en-2M-nested` excluded and retained | ✓ (31 / 37 `queue.json`) |

### BLOCKED — `jina-en-2m`, `jina-en-500k`, `jina-en-200k`

The plan's draft note asserts: *"No `queue.json` from any round ≥ R0200 names any
of them. The newest consuming round is R0147."* **False for three directories.**
`round-0206/queue.json` declares them in `expected_inputs` of jobs
`fit_and_score_fresh_grease_{200k,500k,2m}` and `synthesize_fresh_grease_baseline`,
naming the exact bulk files this tier targeted:

```
/data/latent-basemap/jina-en-{200k,500k,2m}/train/data-00000.npy
/data/latent-basemap/jina-en-{200k,500k,2m}/sample_indices.npy
/data/latent-basemap/jina-en-{200k,500k,2m}/ceiling_umaplearn_k50.parquet
```

R0206 ≥ R0200; strategy-3 DAG citation. **SKIPPED-BLOCKED — 12,582,823,218 B =
11.72 GiB retained, all three directories fully intact.**

### T2-B executed

Recovery note written **before** deletion:
`gsv:/data/latent-basemap/RECOVERY-2026-08-14-jina-minilm-testbeds.md`
(carries all 15 sha256 values, computed pre-deletion).

15 files removed from 7 directories, plus the empty `jina-en-16m/` stub
(0 files, 0 bytes — contained only an empty `train/` subdir, `rmdir`'d).

**T2-B bytes freed: 76,722,797,523 B = 71.45 GiB** (plan estimate 83.33 GiB;
shortfall 11.88 GiB = 11.72 GiB blocked + ~0.16 GiB of `.npy` files outside the
plan's `train/*.npy` / `edges_*.npz` / `*.parquet` deletion pattern, retained).

Only `train/*.npy`, `edges_*.npz` and `*.parquet` were removed. Retained:
every `.json`, `.md`, `.log`, manifest, `.shacache.json`, both `provenance.npz`
files (incl. the sealed `jina-en-8m/provenance.npz`), all `corpus_labels.npy`,
`lang_labels.npy`, `sample_indices.npy`, `centroids_k*.npy`, `holdout_query_*.npy`.

**New finding:** `jina-en-8m/train/data-00000.npy` and
`jina-en-8M-nested/train/data-00000.npy` both hash to
`5de2d63c105bb6f5b2290f97e4ed495894091322117fa00e2f59865d236855e8` — byte-identical
duplicates, 11.44 GiB stored twice. Not detected by the plan's §2.1 duplicate
table, which only scanned inside `runs/`.

---

## Keep-set integrity check (post-deletion)

All verified present:

```
OK  jina-en-8m/provenance.npz
OK  checkpoints/pumap/{edges_150m_k15.npz,edges_30m_k15.npz,faiss_ivf_pq_30m.index,
                       faiss_ivf_pq_3m.index,edges_30m_k15_fuzzy.npz}
OK  jina-en-{2m,500k,200k}/train/data-00000.npy
OK  round-0224/queue-correction-{1,2}/…/substrate.f32.npy
OK  round-0237/queue-correction-1/…/edges-k15-fuzzy.npz
```

**T2-C untouched, at its exact recorded sizes:**
`round-0029/staging/weighted-graph-v2` = 19,647,715,226 B ✓ ;
`round-0018/queue/artifacts/high-d-reference/reference.npz` = 2,640,987,528 B ✓.

**T3 untouched.** Nothing outside `/data/latent-basemap/` and
`/data/checkpoints/` was read for deletion or removed. No `sudo`. No wildcard
`rm` — every deletion was an explicit file path from a pre-built, sanity-checked
list (out-of-scope path, receipt-extension and existence guards on every entry).

---

## Catalog refresh

```
cd ~/code/latent-data && uv run datacat scan
→ wrote /data/catalog.json: 32 corpora, 35 chunkings, 48 embedding datasets
```

---

## df snapshot — END

```
Filesystem                       Size  Used Avail Use% Mounted on
/dev/mapper/ubuntu--vg-data--lv  3.4T  2.8T  519G  85% /data
```

Avail at end: **557,090,598,912 B = 518.8 GiB** (89% → **85%** used).

---

## Totals

| tier | plan estimate | actually freed | blocked |
| --- | ---: | ---: | ---: |
| T1 | 67.32 GiB | **39.71 GiB** | 27.61 GiB (items 1, 5–7, 8–9) |
| T2-A | 20.57 GiB | **13.67 GiB** | 6.90 GiB (`edges_150m_k15.npz`) |
| T2-B | 83.33 GiB | **71.45 GiB** | 11.72 GiB (3 jina dirs) + 0.16 GiB retained |
| **total** | **171.22 GiB** | **124.83 GiB** | **46.23 GiB** |

Measured `df` delta: **124.89 GiB** (393.9 → 518.8 GiB free), matching the
124.83 GiB of deleted bytes to within rounding and concurrent-round churn.

### Blocked items summary — all still on disk, none modified

| item | GiB | reason |
| --- | ---: | --- |
| T1 #1 `round-0224/queue-correction-1/…/substrate.f32.npy` | 22.89 | **`cmp` failed** (differs at byte 3201) — unique unreceipted content, not a duplicate |
| T1 #5–7 `round-0216/queue-correction-1/…` | 3.42 | external citation in `round-0216-2026-08-08.md:133`; plan predicted zero hits |
| T1 #8–9 `round-0223/queue-correction-{1,2}/…` | 1.30 | `result-0223` seals these exact files by full path + sha256 |
| T2-A `edges_150m_k15.npz` | 6.90 | `expected_inputs` of **R0215** (≥ R0200) |
| T2-B `jina-en-2m` / `jina-en-500k` / `jina-en-200k` | 11.72 | `expected_inputs` of **R0206** (≥ R0200) |
| **total held for owner ruling** | **46.23** | |

### Corrections the plan needs

1. **T1 item 1 is not a duplicate.** `cmp` proves R0224 `queue-correction-1`'s
   substrate differs from `correction-2`. §2.1 and surprise S3 both count it as
   one of three copies of an identical 22.89 GiB substrate; there are only **two**
   identical copies plus one distinct one. S3's "45.78 GiB reclaimable" is really
   22.89 GiB.
2. **Both T2 recovery-note drafts contain a false "no round ≥ R0200" claim**, in
   T2-A (`edges_150m_k15.npz` ← R0215) and T2-B (three jina dirs ← R0206). The
   executed notes state the correction explicitly.
3. **The plan's zero-citation grep block never returns "no hits"** — each pattern
   tautologically matches the `queue.json` inside the directory it names. A future
   revision should exclude the item's own queue directory and grep at file
   granularity, per the plan's own §2 method.
4. **§5.5 remains unfixed** — the two dangling registry panel pointers
   (`round-0042`, `round-0046`) were out of this execution's scope and were not
   touched.

