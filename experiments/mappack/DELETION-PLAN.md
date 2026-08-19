# Disk-reclaim audit and tiered deletion plan — `gsv:/data`

> **STATUS: PLAN ONLY. Nothing has been deleted, moved, or modified.**
> This document is read-only analysis plus a proposal. Every deletion below is
> gated on (a) the citation evidence recorded inline and (b) a `RECOVERY-*.md`
> note being written **before** any byte is removed.

**Written 2026-08-14.** Scope: basemap artifacts on `gsv:/data`. Top-level `/data`
numbers are context only.

**Binding rules** — `latent-labs/basemap-100m/OWNER-DECISIONS-PENDING.md` §2:
nothing is deletable without proof it is redundant/reproducible **or**
explicitly superseded-and-never-cited, plus a drafted `RECOVERY-*.md` carrying
sealed hashes and regeneration info. **R0236/R0237 precedent:** delete bulk
payloads, **always keep every `.json` receipt, manifest, and log.**

---

## 0. Headline, and a correction to the standing owner-decision item

**OWNER-DECISIONS-PENDING §2 is stale.** It states `runs/` holds 1.2 TiB with
705 GiB below R0200 and `/data` at 92% / 285 GiB free. A reclamation was
**already executed on 2026-08-11** (`RECLAMATION-PLAN-pre-r0200.md` §11), freeing
**174.57 GiB**. Current measured state:

| | then (§2) | now (measured 2026-08-14) |
| --- | ---: | ---: |
| `/data` free | 285 GiB | **401.2 GiB** (88% used) |
| `runs/` total | 1.2 TiB | **999 GiB** |
| rounds below R0200 | 705 GiB | **526 GiB** |
| rounds ≥ R0200 | — | **474 GiB** |

The cheap pre-R0200 wins are **already spent**. The remaining pre-R0200 tree is
dominated by GPU-built, non-bit-reproducible artifacts that the prior plan
deliberately retained, and that judgement still holds.

**Where the value has moved.** The prior plan's own closing finding (§11, "New
finding") was that duplicate content **above** R0200 now exceeds anything left
below it. This audit confirms and extends that:

| tier | GiB | what it is |
| --- | ---: | --- |
| **T1 — safe now** | **67.32** | failed-queue bulk payloads with zero citations; receipts preserved |
| **T2 — after recovery note** | **124.66** | Modal-era + superseded-testbed artifacts, cited only by pre-program docs |
| **T3 — KEEP** | rest | everything cited by an active round, the registry, the ladder, or the sandbox |
| **T1 + T2 ceiling** | **191.98** | |

**Two scopes were never audited before and carry most of T2:**
`/data/latent-basemap/` **non-runs** directories (~145 GiB of legacy jina/MiniLM
testbeds) and `/data/checkpoints/pumap/` (28 GiB of Modal-era indices). The
2026-08-11 plan was scoped to `runs/` only.

---

## 1. Inventory

### 1.1 `/data` top level (context only — not deletion scope)

| dir | size |
| --- | ---: |
| `latent-basemap` | **1.2 T** |
| `embeddings` | 947 G |
| `chunks` | 400 G |
| `latent-sae` | 124 G |
| `cache` | 81 G |
| `hf` | 41 G |
| `archive` | 34 G |
| `gnm-map` | 33 G |
| `checkpoints` | 29 G |
| `ollama` | 23 G |
| `latent-scope` | 20 G |
| `moments` / `orbital` / `latent-taxonomy` | 7.2 / 6.7 / 5.2 G |
| `tmp` | 3.5 G |
| `logs` | 1.7 G |
| `latent-scope-1.0` | 1.4 G |
| everything else | < 1 G each |

`df`: **3,722,216,030,208 B total / 430,766,354,432 B free (401.2 GiB, 88% used).**

### 1.2 `runs/` by round — the large end

`runs/` = 999 GiB across 200 round directories. Rounds ≥ 1 GiB, with the latest
content mtime:

| round | size | last modified | round | size | last modified |
| --- | ---: | --- | --- | ---: | --- |
| round-0238 | **145 G** | 2026-08-09 | round-0106 | 22 G | 2026-07-29 |
| round-0224 | **68 G** | 2026-08-08 | round-0029 | 23 G | 2026-07-21 |
| round-0025 | 55 G | 2026-07-20 | round-0010 | 25 G | 2026-07-21 |
| round-0236 | 48 G | 2026-08-09 | round-0103 | 18 G | 2026-07-29 |
| round-0237 | 41 G | 2026-08-09 | round-0156 | 18 G | 2026-08-11 |
| round-0262 | 36 G | 2026-08-12 | round-0168 | 18 G | 2026-08-03 |
| round-0243 | 34 G | 2026-08-10 | round-0187 | 16 G | 2026-08-04 |
| round-0235 | 27 G | 2026-08-09 | round-0096 | 16 G | 2026-07-28 |
| round-0233 | 15 G | 2026-08-09 | round-0036/0102 | 15 G | 2026-07-23/29 |
| round-0240 | 12 G | 2026-08-10 | round-0078 | 14 G | 2026-07-27 |
| round-0216 | 11 G | 2026-08-08 | round-0132 | 13 G | 2026-08-01 |
| round-0209 | 12 G | 2026-08-07 | round-0163/4/5 | 12 G ea | 2026-08-03/11 |

Rounds below R0200 total **526.0 GiB**; rounds ≥ R0200 total **474.0 GiB**.

### 1.3 `/data/checkpoints/pumap/` — Modal-era legacy (28 GiB)

| artifact | bytes | mtime | citation status |
| --- | ---: | --- | --- |
| `faiss_ivf_pq_150m.index` | 8,413,041,844 | 2026-04-20 | 6 labs md, 14 `queue.json`; hash sealed in `result-0086` |
| `edges_150m_k15.npz` | 7,407,631,763 | 2026-04-20 | 11 labs md; hash sealed in `result-0215` (forensic) |
| `edges_30m_k15_fuzzy.npz` | 4,586,950,577 | 2026-07-20 | 5 labs md; hash discussed in `review-0029` |
| `faiss_ivf_pq_30m.index` | 1,688,849,884 | 2026-07-11 | 1 labs md, **15 `queue.json`** |
| `edges_30m_k15.npz` | 1,333,493,118 | 2026-07-11 | 8 labs md, **16 `queue.json`** |
| `faiss_ivf_pq_15m.index` | 846,371,764 | 2026-07-11 | **only** `assets_inventory.json` + Modal sweeps |
| `edges_15m_k15.npz` | 637,336,873 | 2026-07-11 | **only** `sweep_{global,structure,v3}_modal.py` |
| `faiss_ivf_sq_1000000.index` | 393,547,256 | 2026-07-11 | **only** `assets_inventory.json` |
| `faiss_ivf_pq_3m.index` | 171,067,604 | 2026-07-11 | 8 labs md, 4 `queue.json`; hash sealed in `result-0044` |
| `edges_3m_k15.npz` | 116,074,421 | 2026-07-11 | **only** `assets_inventory.json` |
| `faiss_ivf_pq_1000000.index` | 57,937,396 | 2026-07-11 | **only** `bench_query_a100.py` + inventory |
| `wikipedia-en-chunked-500-…/` | 2,977,703,639 | 2026-07-11 | `precomputed_psym.pkl`; only Modal `train_modal.py`/`psym_modal.py` |
| `wikipedia-en-chunked-120-…/` | 596,616,462 | 2026-07-11 | Modal-era demo checkpoints |
| `ls-fineweb-edu-100k/`, `ls-dataisplural/` | 635,659,351 | 2026-07-11 | Modal-era latent-scope demos |
| 33 × `model_*.pt` | ~130 MB total | 2026-07-11 | small; **retained by default** |

### 1.4 `/data/latent-basemap/` non-runs directories (~145 GiB, never audited)

| dir | bytes | mtime | name-pass citation result |
| --- | ---: | --- | --- |
| `minilm-15m` | 25,423,078,768 | 2026-07-13 | 1 process log (`2026-07-13_phaseB-minilm-ladder.md`) |
| `jina-en-8M-nested` | 19,550,214,399 | 2026-07-07 | **brace-form only** — `jina-en-{1,2,4,8}M-nested` |
| `jina-en-6M-nested` | 14,655,823,040 | 2026-07-08 | 1 process log |
| `toolchains` | ~15 G | 2026-08-03 | 18 labs md, **22 `queue.json`** — live |
| `canonical` | ~14 G | — | 407 labs md, **561 maps.json**, 329 `queue.json` — live |
| `jina-en-8m` | 13,306,923,139 | 2026-07-07 | `result-0147` seals `provenance.npz` only |
| `jina-en-4M-nested` | 9,798,412,411 | 2026-07-07 | 15 labs md, **31 `queue.json`** |
| `jina-en-2m` | 9,240,469,039 | 2026-07-06 | 5 labs md, 2 `queue.json` |
| `cuml-env` | ~7.3 G | 2026-07-02 | live RAPIDS env (`cuml_py` launcher) |
| `sandbox` | ~6.0 G | 2026-08-12 | **actively read by the ladder** |
| `jina-en-2M-nested` | 4,860,740,404 | — | 31 labs md, **37 `queue.json`** |
| `jina-en-1M-nested` | 2,417,713,288 | 2026-07-07 | **brace-form only** |
| `jina-en-500k` | 2,297,093,959 | — | 4 labs md, 2 `queue.json` |
| `closure` | ~1.8 G | — | 263 labs md, 22 `queue.json` |
| `jina-en-200k` | 1,045,260,220 | — | 10 labs md, 5 `queue.json` |
| `jina-en-200k-prompted` | 804,147,030 | 2026-07-16 | 7 refs |
| `jina-babel-200k` | 738,646,525 | 2026-07-05 | 2 process-log refs |
| `envs`, `tmp`, `track1`, `ls-squad`, `precomputed`, `tests`, `release`, `worktrees`, `registry-history`, `render-cache`, `renders` | < 1.2 G each | — | mixed; **all retained** |
| `jina-en-16m` | 0 (empty stub) | — | 0 refs |

---

## 2. Method — what "cited" was searched for

Citation was resolved at **file** granularity, never directory granularity
(the prior plan's finding F2: directory-level analysis would have over-protected
121 GiB while licensing deletion of the 100 MB file that was actually
load-bearing).

**Indices built (all read-only), reproducible from the commands in §6:**

| source | extracted |
| --- | --- |
| `~/code/latent-labs/**` (excl. `.git`), 1,338 files | 9,243 SHA-256 literals, 9,170 `round-NNNN/...` path refs |
| `maps.json` + 24 `registry-history/*.json` | 573 hashes, 748 paths |
| all 611 `queue.json` under `runs/` | 31,071 declared path refs, 12,509 declared hashes (the DAG) |
| 44,803 JSON receipts under `runs/` | path→hash index: 616,160 bindings, 37,988 paths, 30,265 hashes |
| `sandbox/`, `canonical/`, `release/`, `closure/`, `campaigns/`, `audits/`, `precomputed/`, `renders/`, and the `latent-basemap` repo | 2,291 hashes, 1,936 paths |

**Four independent match strategies** — an artifact is `cited` if **any** hits:

1. **By path** — its own path in a labs doc, registry entry, or `queue.json`.
2. **By SHA-256** — its declared hash appearing anywhere, with no path nearby.
3. **By consumption (DAG)** — declared as `expected_inputs` by any round. This is
   the load-bearing one and is invisible to prose search.
4. **By name** — a bare directory/artifact name, including **brace and glob
   forms**. See §8, surprise S6: this pass alone rescued two directories.

**Anything not positively established as uncited is classed `cited` and lands in
T3.** T1 admits **only** items with zero hits across all four strategies, or
whose every hit is a SHA-256 that survives byte-identically in a retained copy.

### 2.1 Duplicate detection, and a false-positive trap

A first pass keyed hashes to files by **basename**, and produced nonsense such as
`round-0025/…/embeddings.i8` sharing a hash with `round-0102/…/reference.npz`.
The corrected detector requires **(declared SHA-256 match) AND (identical byte
size) AND (both files exist)**. Result — **35.51 GiB** of exact-duplicate content
in rounds ≥ R0200, independently reproducing the prior plan's 36.6 GiB estimate:

| size | copies | artifact |
| ---: | :---: | --- |
| 22.89 GiB | ×2 | `round-0224/queue{,-correction-2}/…/16m-benchmark-substrate-v1/substrate.f32.npy` |
| 5.59 GiB | ×2 | `round-0243/queue/…` ↔ `round-0240/queue/…/builds/r0238-n100000000-c400-s8/graph-k15-ids.i32.npy` |
| 2.79 GiB | ×2 | `round-0237/queue-correction-1/…` ↔ `round-0237/queue/…/builds/…/graph-k15-ids.i32.npy` |
| 2.46 GiB | ×2 | `round-0019/…` ↔ `round-0018/…/high-d-reference/reference.npz` |
| 0.70 GiB | ×2 | `round-0235` build-ladder ↔ k15-fuzzy-graph `graph-k15-ids.i32.npy` |
| 0.54 GiB | ×3 | `round-0223/queue-correction-{1,2,3}/…/cuvs-igd48-k15-fuzzy-graph-v1/edges-k15-fuzzy.npz` |

Not all of these are reclaimable — see T3 for the R0243/R0240 and R0235 pairs,
where **both** copies are declared `expected_inputs` by later rounds.

### 2.2 Queue-outcome map (the basis for tier 1)

`runner-terminal.json` verdicts for every multi-queue round carrying bulk:

| round | queue verdicts (size) |
| --- | --- |
| R0216 | `queue` failed (156K) · `correction-1` **failed (3.5G)** · `correction-2` succeeded (3.5G) · `correction-3` succeeded (3.5G) |
| R0223 | `queue` failed · `correction-1` **failed (670M)** · `correction-2` **failed (805M)** · `correction-3` succeeded (851M) |
| R0224 | `queue` **no terminal (24G)** · `correction-1` **no terminal (22G)** · `correction-2` succeeded (24G) |
| R0229 | phase2 `correction-3` succeeded; four earlier failed (small) |
| R0233 | `queue` **failed (9.3G)** · `correction-1` succeeded (5.6G) |
| R0236 | `queue`/`correction-1` failed (small) · `correction-2` succeeded (48G) |
| R0237 | `queue` **failed (24G)** · `correction-1` succeeded (17G) |
| R0238 | `queue` **failed (145G)** · `correction-1` succeeded (176K) |
| R0240 | `queue` **failed (12G)** — no correction |
| R0262 | `queue`/`correction-1` failed · `correction-2` succeeded (176K); bulk sits **outside** any queue |

**A failed queue is not by itself a deletion licence.** R0238, R0233 and R0240
all hold the *only* copy of an artifact that later rounds consume — see T3.

---

## 3. T1 — safe now (67.32 GiB)

Admission standard: **zero citations across all four strategies**, or every
citation is a SHA-256 that survives byte-identically in a retained sibling. All
belong to **failed / superseded queues**. Per the R0236/R0237 precedent, only
bulk payloads are listed — **every `.json`, manifest, log, `queue.json`,
`runner.log`, `preflight/` and `cache/` entry in these directories is retained.**

| # | path (under `gsv:/data/latent-basemap/runs/`) | bytes | GiB | why safe |
| --: | --- | ---: | ---: | --- |
| 1 | `round-0224/queue-correction-1/artifacts/minilm-mixed-16m-benchmark-substrate-v1/substrate.f32.npy` | 24,576,000,128 | 22.89 | **Zero citations of any kind. No declared hash anywhere. No receipt in the directory** — a bare payload from a queue with no terminal verdict. Byte-size identical to the succeeded `correction-2` copy. |
| 2 | `round-0224/queue/artifacts/minilm-mixed-16m-benchmark-substrate-v1/substrate.f32.npy` | 24,576,000,128 | 22.89 | Zero **path** citations. Its 3 citations are all the SHA-256 `a904e3741e48…`, which is **also** the declared hash of the retained `correction-2` copy (8 citations, incl. `expected_inputs` of R0226/R0227/R0232). The sealed hash survives. |
| 3 | `round-0237/queue/artifacts/minilm-mixed-50000k-cluster-spill-k15-fuzzy-graph-v1/edges-k15-fuzzy.npz` | 15,061,097,304 | 14.03 | Zero citations, no declared hash. The **failed** `queue` copy; the succeeded `correction-1` copy is identical in size, carries hash `b0c030663fb1…`, is sealed by `result-0237`, and **has the `qualified-graph.json` receipt this directory lacks**. |
| 4 | `round-0237/queue/artifacts/minilm-mixed-50000k-cluster-spill-k15-fuzzy-graph-v1/graph-k15-ids.i32.npy` | 3,000,000,128 | 2.79 | Zero citations, no declared hash. Hash `c728d01e6758…` survives in **two** retained copies (`correction-1` and this round's own `build-ladder`). |
| 5 | `round-0216/queue-correction-1/artifacts/minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy` | 3,072,000,128 | 2.86 | Zero citations, no declared hash, **failed** queue, no receipt in directory. The 136-citation ladder substrate is `correction-3`'s copy — retained (T3). |
| 6 | `round-0216/queue-correction-1/…/edges-k15-fuzzy.npz` | 579,640,252 | 0.54 | Same failed queue; zero citations. |
| 7 | `round-0216/queue-correction-1/…/provenance.npy` | 22,000,192 | 0.02 | Same failed queue; zero citations. |
| 8 | `round-0223/queue-correction-1/…/cuvs-igd48-k15-fuzzy-graph-v1/{edges-k15-fuzzy.npz,cuvs-k15-ids.i32.npy}` | 700,326,948 | 0.65 | **Failed** queue. Hash `7eb7dbebe0df…` identical across correction-1/2/3; `correction-3` succeeded and is retained. `cuvs-graph.json` retained. |
| 9 | `round-0223/queue-correction-2/…/cuvs-igd48-k15-fuzzy-graph-v1/{edges-k15-fuzzy.npz,cuvs-k15-ids.i32.npy}` | 700,326,948 | 0.65 | As above. |
| | **T1 total** | **72,287,392,156** | **67.32** | |

### T1 verification commands — run these AT EXECUTION, before each `rm`

Items 1 and 3–9 rest on size + receipt evidence, not on a full-file hash. **Every
one is byte-verifiable against a retained sibling in a single `cmp`.** Run the
matching command; a nonzero exit means **do not delete**.

```bash
R=/data/latent-basemap/runs

# 1 + 2 — R0224: both drop-copies against the retained succeeded correction-2
KEEP=$R/round-0224/queue-correction-2/artifacts/minilm-mixed-16m-benchmark-substrate-v1/substrate.f32.npy
cmp "$KEEP" $R/round-0224/queue-correction-1/artifacts/minilm-mixed-16m-benchmark-substrate-v1/substrate.f32.npy && echo OK-1
cmp "$KEEP" $R/round-0224/queue/artifacts/minilm-mixed-16m-benchmark-substrate-v1/substrate.f32.npy && echo OK-2
# and confirm the retained copy still carries its sealed hash:
sha256sum "$KEEP"   # must equal a904e3741e48cb8f… (full value in the receipt)

# 3 + 4 — R0237: failed queue vs succeeded correction-1
A=$R/round-0237/queue/artifacts/minilm-mixed-50000k-cluster-spill-k15-fuzzy-graph-v1
B=$R/round-0237/queue-correction-1/artifacts/minilm-mixed-50000k-cluster-spill-k15-fuzzy-graph-v1
cmp "$B/edges-k15-fuzzy.npz"   "$A/edges-k15-fuzzy.npz"   && echo OK-3
cmp "$B/graph-k15-ids.i32.npy" "$A/graph-k15-ids.i32.npy" && echo OK-4

# 5-7 — R0216: failed correction-1 vs the 136-citation correction-3
C=$R/round-0216/queue-correction-1/artifacts/minilm-mixed-2m-substrate-and-exact-k15-graph-v1
D=$R/round-0216/queue-correction-3/artifacts/minilm-mixed-2m-substrate-and-exact-k15-graph-v1
for f in substrate.f32.npy edges-k15-fuzzy.npz provenance.npy; do cmp "$D/$f" "$C/$f" && echo "OK-$f"; done

# 8 + 9 — R0223: failed correction-1/2 vs succeeded correction-3
E=$R/round-0223/queue-correction-3/artifacts/minilm-mixed-2m-cuvs-igd48-k15-fuzzy-graph-v1
for q in 1 2; do
  F=$R/round-0223/queue-correction-$q/artifacts/minilm-mixed-2m-cuvs-igd48-k15-fuzzy-graph-v1
  cmp "$E/edges-k15-fuzzy.npz" "$F/edges-k15-fuzzy.npz" && cmp "$E/cuvs-k15-ids.i32.npy" "$F/cuvs-k15-ids.i32.npy" && echo "OK-c$q"
done
```

**Note on item 5.** R0216 `correction-1` is expected to differ from `correction-3`
— it is the *failed* attempt, and R0216 went through two more corrections for a
reason. If `cmp` fails, that is the expected outcome and item 5–7 still stand on
**zero citations + no declared hash + no receipt**; the `cmp` is offered as a
bonus proof, not a precondition. Items 1–4 and 8–9 **do** require their `cmp` to
pass, since their justification rests on a retained byte-identical twin.

Zero-citation re-verification, to be re-run at execution (must return no hits):

```bash
cd ~/code/latent-labs
for p in round-0224/queue-correction-1 round-0224/queue/artifacts/minilm-mixed-16m \
         round-0237/queue/artifacts/minilm-mixed-50000k-cluster-spill-k15-fuzzy \
         round-0216/queue-correction-1 round-0223/queue-correction-1 round-0223/queue-correction-2; do
  echo "== $p"
  grep -rl -- "$p" --include='*.md' .
  grep -l  -- "$p" /data/latent-basemap/maps.json /data/latent-basemap/registry-history/*.json
  grep -rl -- "$p" /data/latent-basemap/runs/*/queue*/queue.json
  grep -rl -- "$p" /data/latent-basemap/sandbox /data/latent-basemap/release /data/latent-basemap/canonical
done
```

**Reclaim: 67.32 GiB. Projected `/data` free after T1: 468.5 GiB (88% → 86%).**

---

## 4. T2 — after a recovery note (124.66 GiB)

Cited only by **superseded or pre-program** documents and Modal-era code, or
unreproducible-but-program-complete. Each sub-tier needs its `RECOVERY-*.md`
written and committed **before** deletion. Drafts are inline below.

### T2-A — Modal-era `checkpoints/pumap` artifacts (20.57 GiB)

| artifact | bytes | GiB | citation class |
| --- | ---: | ---: | --- |
| `faiss_ivf_pq_150m.index` | 8,413,041,844 | 7.83 | `result-0086` size-verified + **declared** hash `7ed8ba06…` (explicitly "> rehash limit"); consumed only by pre-R0200 `queue.json` |
| `edges_150m_k15.npz` | 7,407,631,763 | 6.90 | `result-0215` (forensic) seals **declared** hash `4cf448a0…`, same "> rehash limit" caveat |
| `wikipedia-en-chunked-500-…/` (incl. `precomputed_psym.pkl`) | 2,977,703,639 | 2.77 | only `train_modal.py`, `psym_modal.py`, `edges_modal.py` — **Modal wound down 2026-07-11** |
| `faiss_ivf_pq_15m.index` | 846,371,764 | 0.79 | only `assets_inventory.json` + Modal sweeps |
| `edges_15m_k15.npz` | 637,336,873 | 0.59 | only `sweep_{global,structure,v3}_modal.py` |
| `ls-fineweb-edu-100k/` | 630,741,105 | 0.59 | Modal-era latent-scope demo |
| `wikipedia-en-chunked-120-…/` | 596,616,462 | 0.56 | Modal-era demo |
| `faiss_ivf_sq_1000000.index` | 393,547,256 | 0.37 | only `assets_inventory.json` |
| `edges_3m_k15.npz` | 116,074,421 | 0.11 | only `assets_inventory.json` |
| `faiss_ivf_pq_1000000.index` | 57,937,396 | 0.05 | only `bench_query_a100.py` |
| `ls-dataisplural/` | 4,918,246 | 0.005 | Modal-era demo |
| **subtotal** | **22,081,920,769** | **20.57** | |

> **Honest statement of the cost.** The 150M index and edges are the two items in
> this whole plan whose deletion is **irreversible in the strict sense**. Both
> were built on Modal (now wound down), and FAISS/GPU builds are not
> bit-reproducible across driver and library versions, so regeneration would
> **not** restore the sealed hash. `result-0086` and `result-0215` will keep
> asserting a hash that can never again be checked. That is a real evidentiary
> loss, not a documentation-availability cost. It is proposed as T2 rather than
> T3 only because the program is complete on this substrate — the 100M ladder
> reads `runs/round-0238` and `runs/round-0243`, never these files — and no round
> ≥ R0200 declares either as an input. **If the owner prefers to keep the hash
> checkable, move both to T3; the cost is 14.73 GiB.**

#### `RECOVERY-2026-08-14-modal-era-pumap-checkpoints.md` — DRAFT

> # RECOVERY-2026-08-14-modal-era-pumap-checkpoints.md
>
> **Status: DRAFT — not executed. No bytes have been removed.**
>
> ## What would be removed
>
> Modal-era FAISS indices, kNN edge files and latent-scope demo checkpoints under
> `gsv:/data/checkpoints/pumap/`. **20.57 GiB.** All 33 `model_*.pt` files, the
> `*.manifest.json` sidecars, `_wg_30m_build.log`, and every artifact still
> declared as an `expected_inputs` by any round are **retained in place** —
> specifically `edges_30m_k15.npz` (16 `queue.json`), `faiss_ivf_pq_30m.index`
> (15 `queue.json`), `faiss_ivf_pq_3m.index` (hash sealed in `result-0044`), and
> `edges_30m_k15_fuzzy.npz` (hash discussed in `review-0029`).
>
> ## Why this was judged safe
>
> These artifacts were produced on Modal volumes `embeddings` / `checkpoints`
> during the pre-program 150M/30M/15M era. The Modal wind-down completed
> 2026-07-11 (`latent-labs/logs/process/2026-07-11_modal-winddown.md`), and the
> MiniLM-100M v2 program builds its own substrates and graphs under
> `gsv:/data/latent-basemap/runs/`. Verified before removal:
>
> - **No round ≥ R0200 declares any removed file as an `expected_inputs`** —
>   checked against all 611 `queue.json` under `runs/`.
> - **`maps.json` and all 24 `registry-history` snapshots reference none of them**
>   (0 hits for every removed filename).
> - The only references to `faiss_ivf_pq_15m.index`, `edges_15m_k15.npz`,
>   `edges_3m_k15.npz`, `faiss_ivf_sq_1000000.index` and
>   `faiss_ivf_pq_1000000.index` are `experiments/assets_inventory.json` and the
>   retired Modal scripts `sweep_{global,structure,v3}_modal.py`,
>   `bench_query_a100.py` — none of which can run since the wind-down.
>
> ## Sealed hashes (as declared; record before removal)
>
> | file | bytes | sha256 | source of seal |
> | --- | ---: | --- | --- |
> | `faiss_ivf_pq_150m.index` | 8413041844 | `7ed8ba062baf148b9b076f84c0089849ddb42610f0566a7c197f4c80852893c1` | `result-0086-2026-07-28.md:313` (declared) |
> | `edges_150m_k15.npz` | 7407631763 | `4cf448a05bfdc230f3a538dffaa641a1ab4969b075c7b0628a41fc2ee80d990a` | `result-0215-2026-08-08.md:186` (declared) |
> | remaining 9 items | see table above | **compute and paste `sha256sum` output here before deleting** | — |
>
> **Mandatory pre-deletion step:** run `sha256sum` over every file in the removal
> set and paste the output into this note. Several currently carry **no** sealed
> hash anywhere; deleting them without first recording one destroys the only
> chance to ever identify what was removed.
>
> ## Regeneration
>
> **Not byte-reproducible.** The two 150M artifacts were built on Modal A100/A10G
> FAISS and cannot be regenerated to their sealed hashes on this box. A
> functionally equivalent index is rebuildable from
> `gsv:/data/embeddings/{fineweb-edu-sample-10BT,RedPajama-Data-V2-sample-10B,pile-uncopyrighted}-chunked-120-all-MiniLM-L6-v2/train/`
> (all present) via `archive/early-prototypes/build_150m_index_modal.py` ported
> to local cuVS/FAISS —
> IVF_PQ, nlist ≈ 12k, PQ48x8 — at an estimated multi-hour GPU cost. The
> resulting index will **not** match `7ed8ba06…`.
>
> The smaller 15M/3M/1M indices and the wikipedia/latent-scope demo checkpoints
> are rebuildable the same way at minutes-to-hours cost, likewise not byte-exact.
>
> ## Rollback
>
> None. This deletion is one-way. Do not execute without an explicit owner ruling
> acknowledging the irreversibility statement in the plan's T2-A note.

### T2-B — superseded jina / MiniLM testbed corpora (83.33 GiB)

Pre-program testbeds from the jina era, superseded by the MiniLM-100M v2 program.
Cited only by `logs/process/` and `logs/topics/` narrative entries and by
`result-*` documents from the jina campaigns — never by a round ≥ R0200, never by
the registry.

| dir | bytes | GiB | citations |
| --- | ---: | ---: | --- |
| `minilm-15m` | 25,423,078,768 | 23.68 | 1 process log (Phase B testbed note) |
| `jina-en-8M-nested` | 19,550,214,399 | 18.21 | brace-form in 1 process log |
| `jina-en-6M-nested` | 14,655,823,040 | 13.65 | 1 process log |
| `jina-en-8m` | 13,306,923,139 | 12.39 | `result-0147` + `round-0147/queue.json` |
| `jina-en-2m` | 9,240,469,039 | 8.61 | 5 labs md, 2 `queue.json` (all ≤ R0147) |
| `jina-en-1M-nested` | 2,417,713,288 | 2.25 | brace-form in 1 process log |
| `jina-en-500k` | 2,297,093,959 | 2.14 | 4 labs md, 2 `queue.json` |
| `jina-en-200k` | 1,045,260,220 | 0.97 | 10 labs md, 5 `queue.json` |
| `jina-en-200k-prompted` | 804,147,030 | 0.75 | 7 refs, all pre-program |
| `jina-babel-200k` | 738,646,525 | 0.69 | 2 process-log refs |
| `jina-en-16m` | 0 (empty stub) | 0 | 0 refs — remove the empty dir |
| **subtotal** | **89,479,369,407** | **83.33** | |

**Explicitly NOT in this tier — actively DAG-cited, see T3:**
`jina-en-4M-nested` (9.80 GiB, **31** `queue.json`) and `jina-en-2M-nested`
(4.86 GiB, **37** `queue.json`).

**Retention rule inside every T2-B directory** (R0236/R0237 precedent): delete
only `train/*.npy`, `edges_*.npz`, `*.parquet` bulk. **Retain every `.json`,
`.md`, `.log`, manifest, and — named explicitly — `jina-en-8m/provenance.npz`
(128,001,258 B, sha256 `5b398f0f12c00ff13ed219392031c3116aa939acdf05319cdd723d2b26ad8076`, rehashed and matched in `result-0147-2026-08-01.md:391`).**

#### `RECOVERY-2026-08-14-jina-minilm-testbeds.md` — DRAFT

> # RECOVERY-2026-08-14-jina-minilm-testbeds.md
>
> **Status: DRAFT — not executed. No bytes have been removed.**
>
> ## What would be removed
>
> Bulk embedding and edge payloads from ten pre-program testbed directories at
> `gsv:/data/latent-basemap/` top level. **83.33 GiB.** Every `.json`, `.md`,
> `.log`, manifest and `provenance.npz` in those directories is **retained in
> place**, so each testbed remains identifiable and its receipts still verify.
>
> ## Why this was judged safe
>
> These are the jina-era and Phase-B MiniLM testbeds. The active program
> (`guides/plan-minilm-100m-v2.md`, Phase 3) builds its substrates under
> `runs/round-02*/` and reads none of them. Verified before removal:
>
> - **Zero references in `maps.json` and all 24 `registry-history` snapshots.**
> - **No `queue.json` from any round ≥ R0200 names any of them.** The newest
>   consuming round is R0147 (`jina-en-8m`), superseded by the prompted-substrate
>   pivot (`campaign-2026-08-03-prompted-substrate-pivot.md`).
> - A **name pass including brace and glob forms** was run — this is what
>   established that `jina-en-8M-nested` and `jina-en-1M-nested` are referenced
>   as `jina-en-{1,2,4,8}M-nested` in
>   `logs/process/2026-07-07_fidelity-scaling-law.md:6`. Both are in this tier
>   *because* that citation is a narrative log entry, not a sealed hash or a DAG
>   input. **`jina-en-4M-nested` and `jina-en-2M-nested`, matched by the same
>   brace form, were excluded and retained** — they carry 31 and 37 `queue.json`
>   citations respectively.
>
> ## Sealed hashes
>
> **Mandatory pre-deletion step:** these directories are largely unreceipted.
> Run `sha256sum` over every file in the removal set and paste the output here
> before deleting. Known sealed hash in the retained set:
>
> | file | bytes | sha256 | status |
> | --- | ---: | --- | --- |
> | `jina-en-8m/provenance.npz` | 128001258 | `5b398f0f12c00ff13ed219392031c3116aa939acdf05319cdd723d2b26ad8076` | **RETAINED**, sealed in `result-0147` |
>
> ## Regeneration
>
> Re-embed from `gsv:/data/chunks/` with the jina-v5 / MiniLM pipelines. **Not
> byte-exact** — GPU embedding is not bit-reproducible across driver versions,
> and the source chunk sets for the 200k/500k samples were random draws whose
> seeds are not all recorded. Treat these testbeds as **not recoverable** in
> practice; they are proposed for deletion because nothing reads them, not
> because they can be rebuilt.
>
> ## Rollback
>
> None. One-way.

### T2-C — owner-flagged items, previously withheld (20.76 GiB)

**Neither may be executed on this plan's authority.** Both were surfaced by the
2026-08-11 plan and consciously **not** released; they are listed only so the
accounting is complete and the owner can rule once.

| item | bytes | GiB | standing status |
| --- | ---: | ---: | --- |
| `runs/round-0029/staging/weighted-graph-v2/{out_parts,parts}/…` | 19,647,715,226 | 18.30 | Prior plan's C3. **Owner explicitly withheld.** Still on disk, untouched. Pre-merge shards; the merged product they feed is cited and retained. Recomputable by GPU re-run, **not** byte-exact. |
| `runs/round-0018/queue/artifacts/high-d-reference/reference.npz` | 2,640,987,528 | 2.46 | Byte-identical (hash `e477ad605afd5eda…`) to R0019's copy, which carries **115** citations. But `round-0018-seed42` is a **registered map** in `maps.json`. The prior plan asked for an explicit owner nod; **that nod is still not on record.** |
| **subtotal** | **22,288,702,754** | **20.76** | |

**Reclaim T2 (A+B+C): 124.66 GiB. Projected `/data` free after T1+T2: 593.2 GiB (88% → 82%).**
If T2-C is not released: 103.90 GiB, free 572.4 GiB (83%).

---

## 5. T3 — KEEP (explicit, auditable keep-set)

Everything below is **retained**. Listed explicitly so the keep-set is auditable
and so no future pass has to re-derive it.

### 5.1 The current ladder — substrates, graphs, reserves

| artifact | GiB | why kept |
| --- | ---: | --- |
| `round-0238/queue/artifacts/minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.f32.npy` | **143.05** | **19 citations.** `expected_inputs` of R0244, R0245, R0246, R0247, R0250, R0259; hash `c0b85842236d…` sealed in `result-0238`, `result-0250`, `review-0238`; read by `release/round-0262/basemap/round0240_rung5.py`. **Its queue is marked `failed` and `correction-1` is 176 K — this is the only copy of the 100M substrate.** |
| `round-0236/queue-correction-2/…/25000k-nested-substrate-and-reserves-v1/substrate.f32.npy` | 35.76 | succeeded queue; sealed in `result-0236`; read by `sandbox/25000k-knobs/` |
| `round-0262/artifacts/minilm-mixed-100m-int8-v1/substrate.i8` (+ scales) | 35.76 | newest 100M int8 substrate; declared by `round-0262/queue.json` |
| `round-0224/queue-correction-2/…/16m-benchmark-substrate-v1/*` | 22.89 | the succeeded copy; `expected_inputs` of R0226/R0227/R0232 |
| `round-0235/queue/…/12500k-nested-substrate-and-reserves-v1/substrate.f32.npy` | 17.88 | succeeded queue; sealed in `result-0235`; read by `sandbox/12500k-knobs/`, `sandbox/cuml-ref/` |
| `round-0237/queue-correction-1/…/50000k-cluster-spill-k15-fuzzy-graph-v1/*` | 16.82 | succeeded queue; sealed in `result-0237`; holds the `qualified-graph.json` receipt |
| `round-0243/queue/…/100000k-cluster-spill-k15-fuzzy-graph-v1/{src,dst,wts,ids}` | ~33 | succeeded; `expected_inputs` of R0244/R0246/R0258/R0259/R0262; sealed in `result-0243`, `result-0258`, `result-0259`, `review-0258` |
| `round-0240/queue/…/build-ladder-v1/builds/r0238-n100000000-c400-s8/*` | ~12 | **failed queue, but 23 citations** — `expected_inputs` of R0241–R0246; sealed in `result-0240`, `review-0240`, `review-0241`. **Do not delete despite the duplicate with R0243** — both copies are path-declared inputs. |
| `round-0233/queue/…/6250k-substrate-and-reserves-v1/substrate.f32.npy` | 8.94 | **failed queue, but 11 citations** — `expected_inputs` of R0257; the correction has no substrate, so this is the only copy |
| `round-0233/queue-correction-1/…` (graph, truth, ladder) | ~5.6 | succeeded; `expected_inputs` of R0257 |
| `round-0216/queue-correction-3/…/2m-substrate-and-exact-k15-graph-v1/*` | 3.5 | **136 citations** — the ladder's founding 2M substrate; `expected_inputs` of R0217–R0223; 66 sandbox reads |
| `round-0216/queue-correction-2/…/*` | 3.5 | succeeded; 5 citations incl. `expected_inputs` of R0217 and `review-0216` |
| `round-0235` `graph-k15-ids.i32.npy` duplicate pair | 0.70 | both copies inside a **succeeded** queue and both cited; not a failed-queue payload |
| `round-0223/queue-correction-3/…` | 0.85 | the succeeded cuVS graph |
| `round-0229/queue-{correction-1,phase2-correction-3}` | ~2.9 | succeeded queues |

### 5.2 Registry-bound rounds

`maps.json` (148 entries) actively references **45 round directories**. All are
retained at least in the files the registry binds:

`round-0001, 0018, 0019, 0021, 0022, 0028, 0030, 0033, 0034, 0035, 0036, 0042,
0046, 0049, 0061, 0063, 0064, 0065, 0068, 0069, 0071, 0075, 0076, 0079, 0080,
0101, 0102, 0106, 0107, 0108, 0109, 0110, 0111, 0118, 0217, 0218, 0221, 0222,
0223, 0228, 0229, 0230, 0250, 0255, 0257`

This includes the **round-0018/0019 renders** and **round-0034 coordinates** the
brief flags as actively read. `round-0106` alone is referenced 254 times.

Registry path-existence check: **612 paths, 610 resolve, 2 missing** — see §5.5.

### 5.3 Pre-R0200 bulk retained on non-reproducibility grounds

Unchanged from the 2026-08-11 judgement, and re-confirmed here:

| pool | ~GiB | reason |
| --- | ---: | --- |
| `high-d-reference*.npz` (R0036/0064/0069/0076/0080/0102) | ~50 | GPU kNN builds are not bit-reproducible; regen breaks hash verifiability |
| `native-graph-*` / `canonical-graph-*` | ~35 | same |
| `*.ivfpq` indices (R0059/0072/0077/0086/0096) | ~25 | same; R0096's `larger-index-*` additionally covered by the prior plan's Erratum 2 |
| `round-0025/…/int8-shards/minilm-int8-150m/` | 53.6 | **136 citations**; `expected_inputs` of R0033–R0071, R0215, R0216; 23 registry hash bindings; parent of the whole substrate chain |
| `round-0103/…/jina-diverse-25m-full768-int8-substrate/` | 17.9 | 71 citations, `expected_inputs` of R0104–R0109 |
| `round-0168/…/prompted-diverse-u12/` | 17.8 | 44 citations, `expected_inputs` of R0173, R0208–R0212 |
| `round-0165/queue-correction-1/…/document-compact.f16` | 11.4 | 81 citations, `expected_inputs` of R0166–R0178 |
| `round-0209/queue-correction-1/…/edges-k50-fuzzy.npz` | 10.7 | 27 citations, `expected_inputs` of R0210–R0214 |
| `round-0036/queue.attempt-1362-…` | 1.12 | prior plan's Erratum 3 — cited under "Preserved earlier attempts" |
| everything < 1 GiB under `runs/` | ~66 | not individually adjudicated; retained by default |

### 5.4 Live tooling and working directories

| dir | ~GiB | why |
| --- | ---: | --- |
| `sandbox/` | 6.0 | **actively read by the ladder**; `sandbox/logs/` holds the signed §5-A addendum draft cited by `OWNER-DECISIONS-PENDING` |
| `canonical/` | 14 | 561 `maps.json` refs, 407 labs md, 329 `queue.json` |
| `toolchains/` | 15 | 22 `queue.json` — pinned interpreters rounds are bound to |
| `cuml-env/` | 7.3 | live RAPIDS/cuVS env behind `cuml_py`; the teacher/benchmark path |
| `jina-en-4M-nested/` | 9.8 | 31 `queue.json` |
| `jina-en-2M-nested/` | 4.9 | 37 `queue.json` |
| `release/`, `registry-history/`, `closure/`, `precomputed/`, `track1/`, `tests/`, `renders/`, `render-cache/`, `worktrees/`, `envs/`, `campaigns/`, `audits/`, `queues/` | < 25 total | receipts, published pages, sealed reserves |
| `checkpoints/pumap/edges_30m_k15.npz`, `faiss_ivf_pq_30m.index`, `faiss_ivf_pq_3m.index`, `edges_30m_k15_fuzzy.npz`, all `model_*.pt` | ~7.5 | still DAG-cited or hash-sealed by a published review |

### 5.5 Correctness bug still open (free to fix, independent of deletion)

The prior plan's **F1** is **unfixed**. Of 612 registry paths, exactly 2 do not
resolve:

```
/data/latent-basemap/runs/round-0042/queue/artifacts/panel/panel.json   MISSING
/data/latent-basemap/runs/round-0046/queue/artifacts/panel/panel.json   MISSING
```

Both `round-0042-seed42` and `round-0046-seed42` carry
`evidence_status: review:accepted` with every panel metric `null`. **An accepted
map should not have an unresolvable panel pointer.** Fix or annotate before any
deletion — it costs nothing and it is the only registry integrity defect found.

---

## 6. Reclaim totals and disk projection

| tier | bytes | GiB | `/data` free after | use% |
| --- | ---: | ---: | ---: | ---: |
| *(current)* | — | — | 401.2 | 88% |
| **T1** safe now | 72,287,392,156 | **67.32** | **468.5** | 86% |
| **T2-A** Modal-era pumap | 22,081,920,769 | 20.57 | 489.1 | 86% |
| **T2-B** jina/MiniLM testbeds | 89,479,369,407 | 83.33 | 572.4 | 83% |
| **T2-C** owner-flagged (needs ruling) | 22,288,702,754 | 20.76 | **593.2** | **82%** |
| **T1 + T2 ceiling** | 206,137,385,086 | **191.98** | **593.2** | **82%** |

### Disk pressure forecast

Upcoming known consumers:

| consumer | size |
| --- | ---: |
| PLAN5 corpora | ~50 GB (46.6 GiB) |
| 100M text sidecar | ~48 GB (44.7 GiB) |
| flagship working space | ~40 GB (37.3 GiB) |
| **total** | **~138 GB (128.5 GiB)** |

| scenario | free after consumers |
| --- | ---: |
| do nothing | **272.7 GiB** |
| T1 only | 340.0 GiB |
| T1 + T2-A/B | 443.9 GiB |
| T1 + all T2 | **464.7 GiB** |

**Doing nothing is survivable but tight.** The binding constraint is not the
steady-state total — it is transient staging. The prior plan's **F8** recorded
R0252 holding ~141 GB in `/data/tmp` at once, and R0224 alone materialised three
22.89 GiB copies of one substrate inside a single round. At 272.7 GiB free, a
single round behaving like R0252 or R0224 consumes half the headroom. **T1 alone
(67.32 GiB, zero-citation, byte-verifiable) restores enough margin to run Phase 3
concurrently with a large build**, and it is the tier that needs no ruling beyond
the `RECOVERY` note.

### Suggested order

1. **§5.5 first — it is free.** Fix or annotate the two dangling registry pointers.
2. **T1 (67.32 GiB).** Run every `cmp` in §3. Write
   `RECOVERY-2026-08-14-failed-queue-bulk.md` into each affected round directory.
3. **T2-B (83.33 GiB).** Largest tier, lowest evidentiary weight — nothing
   ≥ R0200 and nothing in the registry touches it. `sha256sum` the removal set
   into the recovery note first.
4. **T2-A (20.57 GiB).** Only after an explicit owner acknowledgement of the
   irreversibility statement, since `result-0086` and `result-0215` seal hashes
   that will become permanently uncheckable.
5. **T2-C (20.76 GiB).** Owner ruling only. Do not act on this plan's authority.

### How this plan makes the cardinal sin impossible by construction

T1 admits an item only if **all four** citation strategies return zero hits, or
if its every hit is a SHA-256 that provably survives in a retained copy. The
greps that establish this are recorded **in §3** and are written to be re-run at
execution time — a T1 item whose grep returns a hit at execution has failed its
admission test and must be promoted, not deleted. No item enters T1 on age, on
directory name, or on a queue's `failed` verdict alone: **R0238, R0233 and R0240
are all failed queues and all are T3**, precisely because the citation index
says so. When the evidence was mixed — R0243/R0240, R0235, R0018, R0029 — the
item went to T3 or to an owner-ruling tier.

---

## 7. What could not be established

- **Byte-identity for T1 items is asserted from size + receipt structure, not
  from full-file hashes.** No file was read in full for this plan. Every T1 item
  ships a `cmp` command in §3 to close that gap **at execution**; items 1–4 and
  8–9 must not be deleted if their `cmp` fails.
- **"Cited" cannot be disproven, only not-found.** Prose can name an artifact by
  description ("the 16M benchmark substrate") that no path or hash search
  matches. Four strategies were used; the brace-form pass alone moved two
  directories out of T1 (§8, S6). Anything found by any strategy is `cited`.
- **65.9 GiB in pre-R0200 units under 1 GiB remains un-adjudicated**, unchanged
  from the prior plan. Retained by default; untouched here.
- **GPU bit-reproducibility was not tested.** The ~110 GiB "recomputable" pool is
  retained on the assumption that FAISS/cuVS rebuilds do not restore sealed
  hashes. If someone demonstrates deterministic rebuilds on pinned versions, that
  pool becomes the largest remaining candidate by a wide margin.
- **Whether a superseded document's citation still binds** is a policy question,
  not a technical one. T2 exists precisely because this plan does not presume the
  answer.

---

## 8. Surprises worth recording regardless of disk

**S1 — The standing owner-decision item is working from stale numbers.**
`OWNER-DECISIONS-PENDING.md` §2 quotes 705 GiB below R0200 and 285 GiB free. A
reclamation already ran on 2026-08-11 and freed 174.57 GiB. The real figures are
526 GiB and 401.2 GiB. §2 should be annotated, or a reader will plan against a
pool that no longer exists.

**S2 — The largest object on the volume lives in a `failed` queue, and must be
kept.** `round-0238/queue/` is 145 GiB with verdict `failed`, and its
`correction-1` is 176 K. The 143.05 GiB `substrate.f32.npy` inside it is the
**only** copy of the 100M substrate, declared as `expected_inputs` by R0244,
R0245, R0246, R0247, R0250 and R0259. Any rule of the form "reclaim failed
queues" destroys the Phase 3 ladder on its first application. The same trap
applies to R0233 (9.3 GiB substrate, only copy, consumed by R0257) and R0240
(12 GiB, 23 citations).

**S3 — R0224 holds three copies of one 22.89 GiB substrate, not two.** The prior
plan's closing finding counted a single 22.89 GiB pair. There are three:
`queue`, `queue-correction-1`, `queue-correction-2` — 68.67 GiB for one 16M
benchmark substrate, of which **45.78 GiB is reclaimable**. This is the single
largest safe win in the plan and it accounts for 68% of T1.

**S4 — One of those copies is completely unreceipted.**
`round-0224/queue-correction-1/artifacts/minilm-mixed-16m-benchmark-substrate-v1/`
contains exactly one file: a 24.5 GB `substrate.f32.npy`. No manifest, no
receipt, no declared SHA-256 anywhere on the volume, and neither `queue` nor
`queue-correction-1` ever wrote a `runner-terminal.json`. This is the prior
plan's **F5** (unreceipted bulk as a fail-open gap in the seal chain) recurring
above R0200, and **F3** (corrections copy forward instead of referencing) still
live in the runner. R0262's 35.76 GiB `substrate.i8` shows a third variant —
bulk written to `round-0262/artifacts/` **outside any queue directory**, also
with no declared hash.

**S5 — `/data/latent-basemap`'s non-runs directories were never audited.** The
2026-08-11 plan was scoped to `runs/`. The top level holds ~145 GiB of legacy
jina/MiniLM testbeds, and `checkpoints/pumap` a further 28 GiB of Modal-era
indices. Together they are **86% of T2** — more reclaimable material than
anything left inside `runs/` below R0200.

**S6 — Brace-notation citations defeat literal path search.**
`jina-en-8M-nested` and `jina-en-1M-nested` return **zero** hits on a literal
grep across all of `latent-labs`. They are cited as
`gsv:/data/latent-basemap/jina-en-{1,2,4,8}M-nested/` in
`logs/process/2026-07-07_fidelity-scaling-law.md:6`. A path-and-hash audit alone
would have placed 20.5 GiB of cited data in T1. The same form appears three more
times (`jina-en-{500k,2m}`, `jina-en-{200k,500k,2m}`, `jina-en-{1,2,4}M-nested`).
**Any future audit must run a brace/glob-aware name pass.** This is the second
independent occasion (after the prior plan's C4 demotions) on which the name pass
changed the answer.

**S7 — Naïve hash indexing produces dangerous false duplicates.** Keying declared
hashes to files by **basename** made `round-0025/…/embeddings.i8` (53.6 GiB)
appear byte-identical to `round-0102/…/reference.npz` and to
`round-0080/…/reference.npz`. Acting on that would have deleted 22 GiB of
registry-bound, 136-citation substrate. Duplicate claims must require
**hash match AND identical byte size**, and even then be `cmp`-verified before
deletion. The corrected detector agrees with the prior plan to within 1.1 GiB
(35.51 vs 36.6 GiB), which is the cross-check that the strict method is right.

**S8 — The registry's two dangling pointers are still dangling.** F1 was raised
on 2026-08-11 and remains unfixed; see §5.5. It is the only registry integrity
defect in 612 paths, it costs nothing to fix, and it is independent of every
deletion here.
