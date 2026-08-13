# Metric Option C — `density_v3`, a repaired density metric

**Date:** 2026-08-13 · **Machine:** gsv · **Compute:** CPU only
(`CUDA_VISIBLE_DEVICES=""`, 8 BLAS threads) · **Status:** sandbox work product,
nothing sealed, nothing registered, no commit.

## Verdict

The degenerate-anchor defect that benched `density_v2` is **demonstrably
fixed**. On the real 2M corpus a single anchor moved `density_v2` by
**59.5%–91.4%**; the same maps scored with `density_v3` move by at most
**0.73%** under a full leave-one-anchor-out sweep — a 100x reduction, and well
inside the 2% bound the mandate asked for.

The honest counterpart: **the repaired metric does not track the fog/collapse
visual judgement.** It ranks the visually-collapsed legacy family *highest*
(rho ~ 0.20–0.26), the foggy-but-healthy UMAP family in the middle
(rho ~ 0.13–0.20), and the visually-clean cuML reference *lowest* (rho = 0.054).
That ordering survives a commensurate-universe control, so it is a property of
the quantity being measured, not an artefact. `density_v3` is a sound and stable
*measurement of local density agreement*; it is not a proxy for map quality, and
on this evidence it should not be turned into a quality gate. See §6.

Artefacts (all paths `gsv:`):

| what | path |
| --- | --- |
| metric | `/home/enjalot/code/latent-basemap/experiments/metrics/density_v3.py` |
| validation harness | `/home/enjalot/code/latent-basemap/experiments/metrics/density_v3_validation.py` |
| results | `/home/enjalot/code/latent-basemap/experiments/metrics/density_v3_results.json` |
| tests | `/home/enjalot/code/latent-basemap/experiments/metrics/tests/test_density_v3.py` |
| radii cache | `/tmp/claude-1000/-home-enjalot-code/af9bdebc-c677-41a4-8156-9eab383e4bbc/scratchpad/density_v3_hd_*.npz` |

---

## 1. What the metric measures, and where v2 broke

For each anchor row `a`:

```
r_hd(a) = mean Euclidean distance to the k=15 nearest self-excluded
          neighbours in the 384-d unit-norm substrate
r_2d(a) = the same quantity on the 2-D map
density_v2 = Pearson( log(r_hd + 1e-12), log(r_2d + 1e-12) )
```

over 4,000 anchors drawn by `np.random.RandomState(123).choice(2_000_000, 4000)`
(`basemap/panel_v2.py:1781-1786`, anchor draw at `panel_v2.py:561-563`;
the sealed radii live in
`gsv:/data/latent-basemap/runs/round-0218/queue/artifacts/minilm-mixed-2m-seed-family-panel-v1/minilm-2m-high-d-reference.npz`).

Exactly one of those 4,000 anchors — substrate row **1449227**, which has 1,377
exact duplicate rows — has `r_hd == 0`. Its duplicates also collapse onto a
single map coordinate, so `r_2d == 0` as well. With `eps = 1e-12` it lands at
`(-27.63, -27.63)`: 27 log-units from the other 3,999 points, on the identity
diagonal. It is a textbook leverage point, and it is byte-identical in every
map, so it acts as a large additive constant that compresses between-map
variation — which is exactly why `density_v2` looked like the *tightest* of the
four panel metrics and got proposed as a gate.

### 1.1 Defect reproduced, from raw bytes

`density_v3_validation.py` recomputes the whole thing on CPU (cKDTree low-D
radii, sealed `r_hd`, no `panel_v2` import). Sealed values reproduce **exactly**:

| map | v2 as defined | sealed panel value | v2 with row 1449227 dropped | **max single-anchor LOO shift** |
| --- | ---: | ---: | ---: | ---: |
| round-0217 seed 42 | `0.4377` | `0.4377` ok | `0.1681` | **61.6%** (abs 0.2697) |
| round-0217 seed 43 | `0.4406` | `0.4406` ok | `0.1784` | **59.5%** (abs 0.2622) |
| round-0217 seed 44 | `0.4387` | `0.4387` ok | `0.1169` | **73.3%** (abs 0.3218) |
| umap-md000-x2 | `0.5986` | — | `0.1272` | **78.7%** (abs 0.4714) |
| umap-dose-x2 | `0.6663` | — | `0.0914` | **86.3%** (abs 0.5749) |
| umap-kernel | `0.6623` | — | `0.0567` | **91.4%** (abs 0.6056) |

Seeds 42/43/44 and their dropped-anchor values match review-0225's table
digit-for-digit. In every one of the six maps the worst leave-one-out anchor is
the same row, **1449227**. On the sandbox UMAP maps the defect is *worse* than on
the maps it was measured on: one anchor in four thousand supplies **91%** of
`umap-kernel`'s reported density.

---

## 2. The v3 repair

Three changes; the quantity being correlated is unchanged.

**(1) Degenerate anchors excluded at source.** Eligible iff `r_hd > EPS_HD`
(default `1e-3`). This is not a knife-edge threshold — the `r_hd` distribution
has an empty band three orders of magnitude wide:

| universe | sorted smallest `r_hd` in a 10,000-row pool |
| --- | --- |
| `minilm-mixed-2m` | `0 x6`, `2.9e-5`, `2.3e-4`, `2.3e-4`, then **`0.188`** |
| `cuml-1m` subsample | `0 x5`, `5.5e-4`, then **`0.051`** |

`1e-3` sits inside that gap in both. Sweeping `eps_hd` over `1e-3 … 1e-1`
changes the score by **0.0000** (§5). The exclusion depends only on the source
space, so the anchor set is identical for every map of a substrate.

**(2) A larger, deterministic anchor set.** `n_anchors = 8000` (2x v2). Rule:

> pool = `sorted(Generator(PCG64(anchor_seed)).choice(N, ceil(n_anchors*1.25), replace=False))`;
> anchors = the first `n_anchors` pool rows, in ascending order, with `r_hd > eps_hd`.

The 1.25x oversample exists so the exclusion still leaves a full-size set; a
shortfall is reported, never silently topped up.

**(3) A rank statistic is primary.** `spearman` is the reported `value`. It is
invariant to any monotone transform of either radius, so a single anchor can
move it only through its *ordinal* position, bounded by `1/n`. `pearson_log` is
kept for v2 continuity, but the `1e-12` floor is gone and both log-radius
vectors are winsorized at the `[0.1%, 99.9%]` quantiles first. That second guard
is needed because degeneracy also appears on the *map* side: 2–3 anchors per 2M
map are non-degenerate in high-D but land on an exactly duplicated 2-D
coordinate. Without winsorization v3's `pearson_log` still moved 6–17% under
leave-one-out; with it, 1.2–4.3%.

### 2.1 High-D radii on CPU — cheaper than budgeted

Mode (b) computes anchor radii by chunked brute force against the memmapped
substrate: BLAS matmul selects `k + 8 + 1 = 24` candidates per corpus chunk
(exact ordering for unit-norm rows), then the candidate vectors are gathered and
distances recomputed by **per-dimension fp32 accumulation**. That rerank is not
optional — the `2 - 2s` matmul shortcut cancels catastrophically for duplicate
rows and reports ~1e-4 where the truth is 0, which is precisely the degeneracy
this metric must see. (`torch.cdist` at its default `compute_mode` has the same
failure; it produced 1,209 spurious zero radii for one reviewer.)

Measured on gsv (Ryzen 9 9950X, 8 threads, sgemm at ~1.37 TFLOP/s):

| stage | cost |
| --- | --- |
| 10,000 anchors x 2,000,000 x 384 fp32 | **57 s**, ~1.3 GB peak |
| 10,000 anchors x 1,000,000 x 384 fp32 | **28 s** |
| per map: cKDTree over 2M 2-D rows + 8,000 queries + both LOO sweeps | **6.5 s** |

The mandate budgeted 10–20 min for this; it is a minute. **No reduction in
anchor count or row count was necessary** — everything below is at the full
8,000 anchors against the full 2,000,000-row substrate.

**Correctness of mode (b):** run against the sealed 4,000-anchor population, the
CPU radii match the sealed GPU reference to `max |delta| = 5.6e-8` (median `8.0e-9`),
and both report exactly one zero, at the same anchor.

---

## 3. Results across the three map families

`anchor_seed = 0`, `n_anchors = 8000`, `eps_hd = 1e-3`, `winsor_q = 0.001`.
9 of 10,000 pool rows excluded as degenerate in the 2M universe, 6 of 10,000 in
the 1M universe.

| map | family | **v3 rho (Spearman)** | v3 `pearson_log` | v2 on the same anchors + degenerates |
| --- | --- | ---: | ---: | ---: |
| round-0217 seed 42 | legacy-2m | **0.2635** | 0.2211 | 0.6309 |
| round-0217 seed 43 | legacy-2m | **0.2632** | 0.2175 | 0.6227 |
| round-0217 seed 44 | legacy-2m | **0.1964** | 0.1666 | 0.6320 |
| umap-md000-x2 | umap-2m | **0.1954** | 0.1777 | 0.7804 |
| umap-dose-x2 | umap-2m | **0.1714** | 0.1392 | 0.8230 |
| umap-kernel | umap-2m | **0.1323** | 0.1035 | 0.8201 |
| cuml-1m | cuml-1m | **0.0536** | 0.0412 | -0.3496 |

The last column is the same statistic v2 used, over the *same* anchor rows plus
the 9 degenerate ones v3 dropped. Nine anchors in eight thousand — **0.1% of the
population** — move it from 0.22 to 0.63, and on the cuML map they flip its
sign, from `+0.041` to `-0.350`. The v2 statistic is not salvageable by anchor
count alone.

### 3.1 Commensurate-universe control

`cuml-1m` lives in its own 1,000,000-row subsample, so its number above is not
directly comparable. Restricting two 2M maps to the same rows
(`gsv:/data/latent-basemap/sandbox/cuml-1m/rows.npy`) and rescoring against the
1M-universe radii:

| map (restricted to the cuML 1M rows) | v3 rho | v3 `pearson_log` |
| --- | ---: | ---: |
| round-0217 seed 42 | 0.2632 | 0.2345 |
| umap-md000-x2 | 0.1831 | 0.1850 |
| cuml-1m | 0.0536 | 0.0412 |

Same ordering. The family ranking is not a universe artefact.

---

## 4. Stability — the defect is gone

Full leave-one-anchor-out sweep over all 8,000 anchors of each map:

| map | **v3 rho: max LOO shift** | (absolute) | v3 `pearson_log`: max LOO shift | v2-style on the same pool |
| --- | ---: | ---: | ---: | ---: |
| round-0217 seed 42 | **0.17%** | 0.00045 | 1.15% | 4.26% |
| round-0217 seed 43 | **0.17%** | 0.00045 | 1.96% | 4.36% |
| round-0217 seed 44 | **0.23%** | 0.00044 | 2.24% | 4.38% |
| umap-md000-x2 | **0.23%** | 0.00045 | 2.93% | 2.94% |
| umap-dose-x2 | **0.25%** | 0.00042 | 2.05% | 2.49% |
| umap-kernel | **0.31%** | 0.00042 | 4.28% | 2.54% |
| cuml-1m | **0.73%** | 0.00039 | 9.89% | 11.18% |

- **Primary statistic: 0.17%–0.73%, everywhere under the 2% bound.** Against
  59.5%–91.4% for `density_v2` on the released population. The maximum absolute
  shift is `4.5e-4` on every map — no anchor is anywhere near load-bearing.
- `pearson_log` clears 2% on 3 of 7 maps and reaches 9.89% on `cuml-1m`. Note the
  denominator: its absolute LOO shift is `<= 0.0052` everywhere (`0.0041` on
  cuml-1m), i.e. ~100x smaller than v2's `0.27–0.61`. The large percentage is an
  artefact of dividing by a near-zero value (0.041). **`pearson_log` is a
  continuity diagnostic, not the value** — do not gate on it.

### 4.1 Synthetic positive control (the tests)

`tests/test_density_v3.py` builds a 12,000-row corpus in the same weak-correlation
regime as the real one (rho ~ 0.22) with exactly **one** degenerate anchor of 2,000
planted — the real corpus's mechanism, scaled. Measured:

| | value |
| --- | ---: |
| v2 statistic, degenerate anchor in | `0.5412` |
| v2 statistic, degenerate anchor out | `0.2167` |
| **v2 shift from one anchor in 2,000** | **60.0%** (LOO: 59.9%) |
| v3, clean corpus | `0.2192` |
| v3, same corpus with the degeneracy planted | `0.2197` |
| **v3 shift** | **0.24%** |
| v3 max LOO shift (planted / clean) | 0.80% / 0.81% |
| v3 with the exclusion policy disabled entirely | 0.66% shift (winsorization + ranks alone) |

Other controls: a map with density structure preserved scores `rho = 1.0000`
(and `1.0000` after a monotone `x^1.7` transform of the high-D radii, confirming
transform-invariance); a shuffled map scores `rho = 0.0125`, `pearson_log = 0.0161`.

---

## 5. Knob sensitivity

Probe maps: `round-0217-seed42` / `umap-md000-x2`, rho reported.

**`eps_hd`** — flat across two decades, because it lives in the measured gap:

| `eps_hd` | 1e-6 | 1e-4 | **1e-3** | 1e-2 | 1e-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed42 | 0.2642 | 0.2640 | **0.2635** | 0.2635 | 0.2635 |
| umap-md000-x2 | 0.1963 | 0.1961 | **0.1954** | 0.1954 | 0.1954 |
| n excluded | 6 | 7 | **9** | 9 | 9 |

**`anchor_seed`** — this is the metric's real uncertainty:

| seed | 0 | 1 | 2 | spread |
| --- | ---: | ---: | ---: | ---: |
| seed42 | 0.2635 | 0.2479 | 0.2586 | 0.0156 |
| umap-md000-x2 | 0.1954 | 0.1778 | 0.1755 | 0.0199 |
| n excluded | 9 | 16 | 13 | — |

The *ordering* (legacy > umap) holds at all three seeds, but the anchor-draw
spread is +/-0.01–0.02 — the same size as the gap between the two families' ranges.

**`winsor_q`** (affects `pearson_log` only):

| `winsor_q` | 0 | 0.0005 | **0.001** | 0.005 | 0.01 |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed42 `pearson_log` | 0.2162 | 0.2182 | **0.2211** | 0.2455 | 0.2530 |
| seed42 LOO | 1.50% | 1.29% | **1.15%** | 0.51% | 0.53% |
| umap-md000 `pearson_log` | 0.1742 | 0.1753 | **0.1777** | 0.1926 | 0.1967 |
| umap-md000 LOO | 4.14% | 3.53% | **2.93%** | 1.15% | 0.86% |

Robustness and value drift together — another reason `pearson_log` is a
diagnostic and Spearman (untouched by this knob) is the value.

---

## 6. Caveats — read before using this for anything

1. **v3 does not agree with the visual judgement, and I did not make it.**
   The mandate's expectation was legacy = collapsed, umap = healthy-but-foggy,
   cuML = clean+healthy. `density_v3` orders them **legacy (0.20–0.26) > umap
   (0.13–0.20) > cuML (0.054)** — the inverse. The commensurate control (§3.1)
   rules out a universe artefact, and the ordering is stable across three anchor
   seeds. The most likely reading is that the measurement is correct and the
   *interpretation* of density agreement as quality is wrong: UMAP-family
   objectives deliberately equalise local density (that is what `min_dist` and
   the cross-entropy repulsion do, and it is why densMAP exists as a separate
   method), so a cleaner UMAP layout should be expected to score *lower* on
   local-density agreement. **Do not read a high `density_v3` as a good map.**
   The only defensible use on this evidence is descriptive: "how much local
   crowding structure did this projection retain", reported next to FFR and the
   purity fidelities, not folded into them.

2. **Family separation is weak.** legacy = `[0.196, 0.264]`, umap =
   `[0.132, 0.195]`. Disjoint, but by 0.0008 — and the anchor-draw spread alone
   is 0.016–0.020, larger than the gap. With n = 3 per family, `density_v3`
   **cannot** be said to separate these families. Distinguishing them would need
   either many more anchors (the sampling SE at n = 8,000 is ~0.011, so the
   family gap needs n in the 10^5 range) or many more maps per family.

3. **`pearson_log` is not gate-worthy** (§4). It depends on `winsor_q`, and its
   relative LOO shift reaches 9.9% where the value is near zero. Report the
   Spearman.

4. **Registration was not attempted and would need more.** A floor from these
   numbers would repeat two mistakes review-0225 documented: it would be
   estimated from the same cells it gates, and `mean - k*sigma` is self-loosening.
   If this ever becomes a criterion it needs (a) a fresh family of conforming
   cells that did not contribute to the definition, (b) a robust scale estimator
   (`median - 3*MAD`, or a trimmed `mean - 2s`), and (c) a decision about
   claim 1 above, since a floor implies "higher is better" and that is not
   established.

5. **Cross-universe comparison needs the restriction, not just the number.**
   Radii are absolute distances in a fixed-size sample; a 1M subsample has
   systematically larger `r_hd` than the 2M superset. Spearman is scale-free
   *within* a universe but the neighbour *sets* differ across universes. Use the
   §3.1 procedure (restrict the map to the shared rows, rescore against that
   universe's radii) whenever comparing across row counts.

6. **Nothing here is sealed or registered.** Sealed artefacts were read
   read-only. The sandbox maps under `gsv:/data/latent-basemap/sandbox/2m-knobs`
   carry `"note": "sandbox artifact; not a round, no sealed claim"` and their
   density numbers inherit that status. No commit was made.

7. **The high-D radii cache lives in scratch**, not on `/data`. Recomputing it
   costs 57 s, so it was not worth a durable artefact; if v3 is adopted, the
   pool radii should be sealed alongside the substrate the way
   `minilm-2m-high-d-reference.npz` is.

8. **Cross-map registration was never in scope.** `density_v3` compares each
   anchor's own neighbourhood in two spaces and needs no correspondence between
   maps. A cross-map version (does *this* map's density field match *that* one's)
   would need Procrustes or an equivalent registration step, which none of these
   artefacts carry.

---

## 7. Reproducing

```bash
cd /home/enjalot/code/latent-basemap
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  .venv/bin/python experiments/metrics/density_v3_validation.py
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
  experiments/metrics/tests/test_density_v3.py -q
```

Total: ~4 min for the full validation (including three anchor-seed radii
recomputations for §5), ~4 s for the 18 tests. Inputs:

- substrate `gsv:/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate.f32.npy`
- sealed high-D reference + legacy coordinates `gsv:/data/latent-basemap/runs/round-0218/queue/artifacts/minilm-mixed-2m-seed-family-panel-v1/`
- sandbox UMAP maps `gsv:/data/latent-basemap/sandbox/2m-knobs/{umap-md000-x2,umap-dose-x2,umap-kernel}/coordinates.npy`
- cuML reference `gsv:/data/latent-basemap/sandbox/cuml-1m/{cuml-xy.npy,emb.f32.npy,rows.npy}`

API:

```python
from density_v3 import density_v3
result = density_v3(xy, substrate_or_radii, anchor_seed=0, n_anchors=8000)
result["spearman"]                   # the value
result["pearson_log"]                # v2-continuity diagnostic
result["n_excluded_degenerate_hd"]   # anchors dropped by the policy
result["leave_one_out"]["spearman"]["max_relative_shift"]
```

`substrate_or_radii` accepts a `(N,)` per-row radius array, an `(anchor_ids,
radii)` pair, an `(N, D>2)` substrate array/memmap, or a path to a `.npy`
substrate (opened with `mmap_mode='r'`; only bounded chunks are ever read).
