# COLLAPSE and FOG — production metrics and their calibration

Metric Option A. Two CPU-only map-quality metrics that the program's
registered gates cannot see, hardened out of the sandbox, plus their
calibration on the sealed n = 29 2M seed family.

- Metrics: `experiments/metrics/collapse_fog.py`
- Null calibrator: `experiments/metrics/null_calibration.py`
- Driver: `experiments/metrics/calibrate_collapse_fog.py`
- Tests: `experiments/metrics/tests/test_collapse_fog.py`
- Machine-readable results:
  `experiments/metrics/results/collapse-fog-measurements.json`,
  `experiments/metrics/results/collapse-fog-calibration.json`

**Nothing here is a registered gate and nothing sealed was written.** Every
sealed number quoted below carries its file path and JSON key.

## 1. What the two metrics measure

**COLLAPSE** — `r10 / R · sqrt(N)`, where `r10` is the median 2D distance to
the 10th nearest neighbour over a seeded 20k sample and `R` is the p90 radius
about the centroid. Hardened from `tissue_metrics()` in
`experiments/sandbox/heldout_eval.py` and the inline copy in
`experiments/sandbox/knobs_2m.py::run_arm`, both of which report the raw
ratio `r10 / R`.

The `sqrt(N)` factor is the new part. In 2D at fixed occupied area the median
distance to the k-th neighbour scales as `1/sqrt(N)`, so the raw ratio is not
comparable across map sizes. Multiplying by `sqrt(N)` removes that scaling —
see §4, where 3 umap recipes measured at both 2M and 6.25M rows agree to
within 11% on the adjusted statistic while their raw ratios differ by
1.9-2.0x.
Gated **one-sided from below**: too small means the map has folded into
point-like beads. FFR *rewards* that failure, which is why no registered
metric sees it.

**FOG** — fraction of total binned point mass sitting in *occupied* bins
whose count is below 1% of the peak bin count, on a 1024x1024 histogram over
the (0.1, 99.9) percentile extent with 2% pad and edge clipping. The binning
is byte-for-byte `experiments/map_renders.py::robust_extent` /
`binned_counts`; the mass definition is `low_density_mass_fraction` from
`heldout_eval.tissue_metrics`. Gated **one-sided from above**: too large
means diffuse haze between the clusters. The implementation reproduces the
sandbox's published cuML reference value (0.040) exactly: measured here at
**0.0397**.

The two are different failure directions, not two views of one axis. A
collapsed map has *low* fog (all mass in a handful of dense bins) and a low
collapse statistic. A hazy map has *high* fog and a normal collapse
statistic. A gate needs both — and the tests assert exactly that cross
property.

## 2. Calibrated multiplier

The floor/ceiling estimator is the program's established robust form,
`median -/+ k · MAD_n`, consistency constant 1.4826. `k` is calibrated on a
Gaussian null exactly as R0234 did: simulate 4,000,000 null families of the
given size and take the smallest `k` whose floor `median - k · MAD_n` is
cleared by a fresh conforming draw 95% of the time with 95% confidence
(one-sided 95/95 tolerance bound). Because `1 - Phi(L) >= 0.95` iff
`L <= z_0.05`, this inverts per family and `k` is simply the 95th percentile
of `(median + 1.6448536) / MAD_n` — no bisection, no search noise.

**Validation gate, run before trusting anything at n = 29.** R0234's
published one-sided n = 13 `median_minus_k_madn` multiplier, read from
`/data/latent-basemap/runs/round-0234/queue/artifacts/minilm-mixed-2m-calibrated-robust-floors-n13-v1/minilm-calibrated-robust-floors-n13.json`
key `calibration.n13.candidates.median_minus_k_madn.one_sided.calibrated_multiplier`:

| | value |
| --- | --- |
| sealed R0234 (n = 13) | **3.7363661744** |
| reproduced here | **3.7363661744** |
| relative error | **0.0000%** (tolerance 2%) |
| verdict | **PASS** |

The agreement is bit-exact, not merely within tolerance: with the same seed
(20260809), the same 4,000,000 families and the same generator, the
closed-form quantile lands on the identical double.

**Derived n = 29 multiplier:**

| | value |
| --- | --- |
| **k1 (one-sided, n = 29)** | **2.6934393368** |
| null families simulated | 4,000,000 (seed 20260809) |
| new-cell false-fail rate | 0.012233 |
| detection power at -1s / -2s / -3s | 0.0835 / 0.3006 / 0.6284 |

Cross-check against the sealed R0255 value at the same n, read from
`/data/latent-basemap/runs/round-0255/queue/artifacts/minilm-mixed-2m-calibrated-madn-floors-n29-v1/minilm-calibrated-madn-floors-n29.json`
key `calibration.n29.candidates.median_minus_k_madn.one_sided.calibrated_multiplier`
= **2.6934393368**; ours differs by **0.0000%** (also bit-exact).
R0255 calibrated that multiplier for FFR and purity; it is the same estimator
at the same n, so agreement is the expected result and is reported as a
consistency check, not a new claim.

## 3. Legacy fingerprint — the sealed n = 29 2M seed family

All 29 cells, seeds 42–70, resolved through `/data/latent-basemap/maps.json`
(`coordinates.file`, `gsv:` prefix stripped). The authoritative seed list is
R0255's own `exact_family_seeds` in the n = 29 artifact cited in §2 — note it
spans five rounds (R0217/0218, R0221/0222, R0230, R0250, R0255), not the
four one might infer from the round-0255 queue alone.

Every one of the 29 is a `legacy_lp`-kernel map. Every one is bead-collapsed.

| seed | round | N | r10/R (raw) | **r10/R x sqrt(N)** | **fog** | peak bin | occupied bins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0217 | 2,000,000 | 1.17e-04 | 0.1660 | 0.0901 | 4,990 | 0.0388 |
| 43 | 0217 | 2,000,000 | 1.19e-04 | 0.1687 | 0.1078 | 5,766 | 0.0409 |
| 44 | 0217 | 2,000,000 | 1.05e-04 | 0.1488 | 0.0913 | 4,841 | 0.0364 |
| 45 | 0217 | 2,000,000 | 1.35e-04 | 0.1904 | 0.0745 | 2,761 | 0.0449 |
| 46 | 0221 | 2,000,000 | 1.03e-04 | 0.1457 | 0.0808 | 5,460 | 0.0359 |
| 47 | 0221 | 2,000,000 | 1.23e-04 | 0.1741 | 0.0920 | 4,211 | 0.0391 |
| 48 | 0221 | 2,000,000 | 1.20e-04 | 0.1694 | 0.0811 | 4,040 | 0.0397 |
| 49 | 0221 | 2,000,000 | 1.31e-04 | 0.1846 | 0.0793 | 2,834 | 0.0461 |
| 50 | 0230 | 2,000,000 | 1.32e-04 | 0.1860 | 0.0789 | 2,748 | 0.0506 |
| 51 | 0230 | 2,000,000 | 1.07e-04 | 0.1512 | 0.0900 | 6,092 | 0.0353 |
| 52 | 0230 | 2,000,000 | 1.27e-04 | 0.1793 | 0.0948 | 5,191 | 0.0397 |
| 53 | 0230 | 2,000,000 | 1.01e-04 | 0.1422 | 0.0791 | 5,425 | 0.0326 |
| 54 | 0230 | 2,000,000 | 1.08e-04 | 0.1527 | 0.0800 | 3,610 | 0.0425 |
| 55 | 0250 | 2,000,000 | 1.23e-04 | 0.1737 | 0.0784 | 3,197 | 0.0435 |
| 56 | 0250 | 2,000,000 | 1.17e-04 | 0.1650 | 0.0907 | 5,346 | 0.0386 |
| 57 | 0250 | 2,000,000 | 1.25e-04 | 0.1767 | 0.0949 | 5,217 | 0.0416 |
| 58 | 0255 | 2,000,000 | 1.09e-04 | 0.1537 | 0.0829 | 4,856 | 0.0383 |
| 59 | 0255 | 2,000,000 | 1.15e-04 | 0.1631 | 0.0758 | 4,115 | 0.0391 |
| 60 | 0255 | 2,000,000 | 1.05e-04 | 0.1488 | 0.0969 | 6,294 | 0.0375 |
| 61 | 0255 | 2,000,000 | 1.31e-04 | 0.1857 | 0.0944 | 5,219 | 0.0397 |
| 62 | 0255 | 2,000,000 | 1.08e-04 | 0.1530 | 0.0842 | 4,561 | 0.0368 |
| 63 | 0255 | 2,000,000 | 9.93e-05 | 0.1404 | 0.0672 | 3,081 | 0.0404 |
| 64 | 0255 | 2,000,000 | 1.19e-04 | 0.1689 | 0.0981 | 5,906 | 0.0355 |
| 65 | 0255 | 2,000,000 | 1.18e-04 | 0.1663 | 0.1031 | 5,417 | 0.0382 |
| 66 | 0255 | 2,000,000 | 1.24e-04 | 0.1756 | 0.0903 | 4,441 | 0.0426 |
| 67 | 0255 | 2,000,000 | 1.06e-04 | 0.1504 | 0.0830 | 5,088 | 0.0364 |
| 68 | 0255 | 2,000,000 | 1.03e-04 | 0.1462 | 0.0708 | 3,146 | 0.0431 |
| 69 | 0255 | 2,000,000 | 9.59e-05 | 0.1356 | 0.0935 | 7,203 | 0.0357 |
| 70 | 0255 | 2,000,000 | 1.18e-04 | 0.1667 | 0.1585 | 7,967 | 0.0459 |

| statistic | median | MAD_n | min | max |
| --- | --- | --- | --- | --- |
| collapse `r10/R·sqrt(N)` | 0.1660 | 0.0197 | 0.1356 | 0.1904 |
| fog | 0.0900 | 0.0121 | 0.0672 | 0.1585 |

Applying the calibrated n = 29 estimator to this family gives a collapse
floor of **0.1129** and a fog ceiling of **0.1226**.

**Neither is a usable gate and neither should be registered as one.** The
family is 29 replicates of one broken recipe, so a floor fitted to it
certifies that a new map is *as collapsed as the collapsed ones*. Any healthy
map (collapse 1.0–2.8) clears the collapse floor of 0.1129
trivially, and a fog ceiling of 0.1226 would **reject every healthy
map measured in §4** — real tissue carries far more low-density mass than a
bead field does, so the collapsed family is *better* on fog than the maps we
want.

What the family is genuinely good for is a **fingerprint**: it pins the
collapsed mode at collapse = 0.1660 +/- 0.0197 (MAD_n)
across 29 independent seeds and five rounds, with a total range of
0.1356-0.1904. The mode is reproducible and tight, and §5 shows
it is unmoved by dose, by the a/b fit, or by graph choice, which is what makes
the separation in §6 meaningful. (All 29 cells are 2M rows, so the family
itself carries no evidence about N; that evidence is in §4.)

## 4. Healthy family — sandbox umap-kernel arms + the cuML reference

Coordinates under `/data/latent-basemap/sandbox/2m-knobs/*/coordinates.npy`,
`/data/latent-basemap/sandbox/6250k-knobs/*/coordinates.npy` and
`/data/latent-basemap/sandbox/cuml-1m/cuml-xy.npy`. `a` is the UMAP (a, b)
fit; `a = 1.9328` is `min_dist = 0.00` and `a` falls as `min_dist` rises.
Arms marked **(core)** are the `min_dist = 0.00` umap recipe — they differ
only in seed, dose multiplier and rung, which is the closest thing to a
conforming population the sandbox offers.

| arm | rung | N | a (min_dist fit) | dose | seed | r10/R (raw) | **r10/R x sqrt(N)** | **fog** | peak bin | occupied bins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| umap-dose-x2 | 2m | 2,000,000 | 1.5770 | - | 42 | 1.01e-03 | 1.4342 | 0.7060 | 1,433 | 0.2698 |
| umap-kernel | 2m | 2,000,000 | 1.5770 | - | 42 | 1.10e-03 | 1.5508 | 0.8071 | 1,437 | 0.3292 |
| umap-md000-x2 **(core)** | 2m | 2,000,000 | 1.9328 | 2 | 42 | 8.05e-04 | 1.1379 | 0.5013 | 1,435 | 0.2275 |
| umap-md000-x2-seed43 **(core)** | 2m | 2,000,000 | 1.9328 | 2 | 43 | 8.71e-04 | 1.2317 | 0.5123 | 1,435 | 0.2373 |
| umap-md000-x4 **(core)** | 2m | 2,000,000 | 1.9328 | 4 | 42 | 8.37e-04 | 1.1834 | 0.4587 | 1,432 | 0.2136 |
| umap-md0025-x2 | 2m | 2,000,000 | 1.8404 | 2 | 42 | 1.01e-03 | 1.4231 | 0.6148 | 1,434 | 0.2705 |
| umap-md005-x2 | 2m | 2,000,000 | 1.7502 | 2 | 42 | 1.03e-03 | 1.4501 | 0.6815 | 1,431 | 0.2866 |
| umap-md020-x2 | 2m | 2,000,000 | 1.2621 | 2 | 42 | 1.62e-03 | 2.2971 | 0.9516 | 1,436 | 0.4293 |
| umap-md035-x2 | 2m | 2,000,000 | 0.8741 | 2 | 42 | 1.86e-03 | 2.6340 | 0.0000 | 99 | 0.5136 |
| umap-md050-x2 | 2m | 2,000,000 | 0.5830 | 2 | 42 | 1.97e-03 | 2.7872 | 0.9981 | 1,433 | 0.5466 |
| umap-mind0 **(core)** | 2m | 2,000,000 | 1.9290 | - | 42 | 8.43e-04 | 1.1929 | 0.5396 | 1,435 | 0.2373 |
| umap-mind05 | 2m | 2,000,000 | 0.5830 | - | 42 | 2.00e-03 | 2.8289 | 0.9975 | 1,433 | 0.5283 |
| umap-pos02-x2 | 2m | 2,000,000 | 1.5769 | 2 | 42 | 1.25e-03 | 1.7617 | 0.8675 | 1,435 | 0.3845 |
| umap-pos15-x2 | 2m | 2,000,000 | 1.5769 | 2 | 42 | 1.29e-03 | 1.8235 | 0.7899 | 1,437 | 0.3097 |
| umap-md000-x2 **(core)** | 6250k | 6,250,000 | 1.9328 | 2 | 42 | 4.22e-04 | 1.0551 | 0.4509 | 4,544 | 0.2395 |
| umap-md000-x4 **(core)** | 6250k | 6,250,000 | 1.9328 | 4 | 42 | 4.19e-04 | 1.0476 | 0.4456 | 4,541 | 0.2416 |
| umap-mind0 **(core)** | 6250k | 6,250,000 | 1.9328 | 1 | 42 | 4.48e-04 | 1.1196 | 0.4792 | 4,545 | 0.2552 |
| cuml-1m-reference | 1m | 1,000,000 | - | - | - | 1.67e-03 | 1.6724 | 0.0397 | 198 | 0.1928 |

The N-invariance check lives in this table: 3 arm names appear at both
2M and 6.25M rows with the same recipe, i.e. one map spec at 3.1x the rows.
Between the matched rungs the raw `r10/R` differs by 1.88-2.00x
against the sqrt(6.25/2) = 1.77 that pure density scaling predicts, and the
residual is exactly the 6.1-11.5% by which the adjusted
statistic drifts. So the sqrt(N) factor removes the bulk of an ~1.9x effect
and leaves ~10%. That is the whole case for it, and it is measured, not
assumed.

## 5. Legacy-kernel and other sandbox arms, for contrast

| arm | rung | N | a (min_dist fit) | dose | seed | r10/R (raw) | **r10/R x sqrt(N)** | **fog** | peak bin | occupied bins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dose-x2 | 2m | 2,000,000 | - | - | 42 | 9.13e-05 | 0.1291 | 0.0843 | 5,330 | 0.0327 |
| gc-a05-md000-x2 | 2m | 2,000,000 | 1.9328 | 2 | 42 | 7.92e-05 | 0.1121 | 0.0544 | 5,428 | 0.0255 |
| gc-a05-md010-x2 | 2m | 2,000,000 | 1.5769 | 2 | 42 | 1.25e-04 | 0.1766 | 0.0492 | 1,584 | 0.0446 |
| gc-a2-md000-x2 | 2m | 2,000,000 | 1.9328 | 2 | 42 | 1.75e-03 | 2.4814 | 0.9846 | 1,440 | 0.5224 |
| gc-a2-md010-x2 | 2m | 2,000,000 | 1.5769 | 2 | 42 | 1.90e-03 | 2.6932 | 0.9941 | 1,432 | 0.5191 |
| kernel-a4 | 2m | 2,000,000 | 4.0000 | - | 42 | 9.74e-05 | 0.1378 | 0.0984 | 5,720 | 0.0374 |
| kernel-b2 | 2m | 2,000,000 | - | - | 42 | 9.39e-05 | 0.1327 | 0.1100 | 5,156 | 0.0436 |
| replay-baseline | 2m | 2,000,000 | - | - | 42 | 1.09e-04 | 0.1545 | 0.1042 | 6,388 | 0.0380 |

Two things worth flagging. First, `gc-a2-md000-x2` and `gc-a2-md010-x2` (the
generalised-Cauchy kernel at a = 2) are **not** collapsed — collapse 2.48 and
2.69, fog 0.98 and 0.99 — so the collapse failure is a property of the
`legacy_lp` kernel specifically, and of the generalised-Cauchy kernel only at
a = 0.5. Second, the four `legacy_lp` arms (`dose-x2`, `kernel-a4`,
`kernel-b2`, `replay-baseline`) plus the two `gc-a05-*` arms all land in
0.11-0.18, i.e. inside or beside the sealed family's fingerprint of
0.1356-0.1904. Doubling the dose, setting a = 4, setting
b = 2, and replaying the graph all leave the collapse statistic where it was.
Only the low-dimensional kernel moves it.

## 6. Proposed provisional bands

Fitted to the **core umap family** of §4 (n = 7: the `min_dist = 0.00`
arms at both rungs) with the same estimator and `k` calibrated at that family
size — `k = 6.05513` from 1,000,000 Gaussian null families:

| metric | direction | median | MAD_n | min | max | **proposed bound** |
| --- | --- | --- | --- | --- | --- | --- |
| collapse `r10/R·sqrt(N)` | lower floor | 1.1379 | 0.0815 | 1.0476 | 1.2317 | **>= 0.6444** |
| fog | upper ceiling | 0.4792 | 0.0419 | 0.4456 | 0.5396 | **<= 0.7330** |

Against those bounds:

- all 29 sealed family cells **fail** the collapse floor (max
  0.1904 vs floor 0.6444) — which is the point;
- the cuML reference (N = 1,000,000) **passes both**: collapse
  1.6724, fog 0.0397;
- the over-separated umap arms at `min_dist >= 0.20` **fail** the fog ceiling
  at 0.95-1.00, so the ceiling behaves as a haze detector on real maps and not
  only on synthetic ones -- **except** `umap-md035-x2`, which reports fog
  0.0000 and would PASS despite being one of the haziest maps in the sandbox.
  That is the degeneracy in item 5 below, and it is why a fog gate must consult
  `degenerate` before it consults the value.

### Why the bands are not fitted to all 18 umap+cuML arms

Pooling the whole knob sweep (n = 18, `k = 3.13447`) gives a collapse
floor of **-0.0072** and a fog ceiling of **1.1823**: a
negative floor is vacuous and a ceiling above 1.0 is unattainable, because the
sweep deliberately spans `min_dist` 0.00 → 0.50 and is therefore a *designed
contrast*, not a sample from one population (median 1.4422, MAD_n
0.4624 for collapse; 0.5772 / 0.1930 for fog). That
degenerate result is reported rather than hidden — it is the clearest
available demonstration that the 95/95 machinery gives nonsense when its
exchangeability assumption is violated.

### Evidentiary status — read before quoting any number above

1. **Provisional reference bands, not registered gates.** No round registered
   them; this work registered nothing.
2. **The core family is not a seed family.** n = 7 arms sharing one
   (a, b) but differing in seed (2 values), dose (3 levels) and rung (2M and
   6.25M). The 95/95 tolerance machinery assumes n exchangeable draws from
   one conforming population. That is not met. `k` is applied for consistency
   with the program's convention, not because the sample earns it.
3. **The collapse band is the stronger of the two.** The two populations do
   not overlap and are not close: the sealed family's maximum is
   0.1904 and the core family's minimum is 1.0476, a factor of
   5.5, with the proposed floor 0.6444 sitting between
   them. Its N-invariance is measured directly across two rungs (§4).
4. **The fog band is the weaker of the two, and is a ceiling on regression
   rather than a quality bar.** The core arms sit at 0.4792 while the
   cuML reference is an order of magnitude below at 0.0397. Treat
   cuML as the aspiration and 0.7330 as the line past which a map has
   become visibly hazier than the current umap recipe.
5. **Fog has a hard degeneracy that a gate must handle.** Bin counts are
   integers and the cutoff is 1% of the peak bin, so a map whose peak bin
   holds fewer than 100 points reports fog **exactly 0.0000** no matter how
   hazy it is. `map_fog` returns `resolution_levels` and `degenerate` for
   this reason. Degenerate in this measurement set: `umap-md035-x2` (2m, peak bin 99) — an arm that
   reports fog 0.0000 while sitting between two arms that report 0.95 and 1.00.
   The cuML reference is not degenerate but is one step from it, at peak bin
   198 —
   **one** usable level — so its celebrated 0.040 sits a single integer count
   above the degeneracy. Any fog gate must refuse a degenerate measurement
   instead of passing it.
6. **The fog cutoff is partly a corpus property.** On the 2M mixed MiniLM
   substrate the peak bin of every umap-kernel arm is the same exact-duplicate
   group of 1377 identical rows, so the 1%-of-peak cutoff is pinned by
   duplicate text rather than by the map's densest tissue. Changing the
   corpus changes the cutoff.
7. **What would make these registrable:** a real seed family of *healthy*
   maps — one recipe, n >= 13 seeds, no knob variation — measured under this
   protocol. Until then the honest claim is: the collapse statistic separates
   the two modes cleanly and is ready to gate once a healthy family exists;
   the legacy family cannot define a healthy floor for either metric.

## 7. Positive controls

`experiments/metrics/tests/test_collapse_fog.py` carries the mandatory
failing inputs — a guard whose suite contains no failing input is untested at
its only job:

- **(a)** a synthetic bead-collapsed map (40 clusters of near-identical
  points) falls below the collapse floor;
- **(b)** a synthetic haze map (clusters buried in 55% uniform noise) exceeds
  the fog ceiling;
- **(c)** a synthetic healthy map (Gaussian blobs of finite radius, 0.2%
  background) clears both;
- **cross controls**: the collapsed map does *not* trip fog and the hazy map
  does *not* trip collapse — the two are independent directions;
- **(d)** N-invariance: 4x subsampling a healthy map moves `r10/R·sqrt(N)` by
  < 10%, while the raw ratio moves by > 1.5x (asserted as a control on the
  control);
- **(e)** determinism: same seed gives identical dicts; fog has no rng at all;
- **(f)** memmap: the memmap and in-memory paths give identical dicts, and
  the subsampled-tree path (forced with a small `max_tree_rows`) agrees with
  the full-tree path to within 10%;
- plus the fog-degeneracy flag, shape rejection, the exact `sqrt(N)`
  algebra, and a 1M-family smoke reproduction of R0234's sealed multiplier.

A note on the literal requirement "a synthetic uniform-noise map must fail
the fog ceiling": **it cannot, by arithmetic**, and the suite documents that
rather than papering over it. Fog counts mass in bins below 1% of the *peak*
bin, and bin counts are integers. A structureless uniform map has no peak —
at 2M points over 1024² bins the peak bin holds ~13 counts, 1% of that is
0.13, the absolute floor of 1 takes over, and no occupied bin can hold less
than 1 point. Fog is exactly 0.0000. The failure fog exists to catch is
*clusters plus haze*, which is what control (b) uses. Pure structureless
noise is separated cleanly by the companion `occupied_bin_fraction` that
`map_fog` already returns — 0.81 for uniform noise against 0.04–0.55 for
every real map measured here — but that companion is reported, not gated, and
is not part of this proposal.

## 8. Reproduction

```bash
cd /home/enjalot/code/latent-basemap

# both metrics on all 29 sealed family cells + every sandbox arm + cuML (~45 s)
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    experiments/metrics/calibrate_collapse_fog.py measure

# Gaussian-null calibration: R0234 n=13 validation gate, then n=29 (~10 s)
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    experiments/metrics/calibrate_collapse_fog.py calibrate

# regenerate this report from the two JSONs
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    experiments/metrics/calibrate_collapse_fog.py report

# or all three
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    experiments/metrics/calibrate_collapse_fog.py all

# tests, including the mandatory positive controls
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
    experiments/metrics/tests/ -q -p no:cacheprovider
```

Everything is CPU-only: `CUDA_VISIBLE_DEVICES=""` is set in every command,
the driver warns if it is not, and no module in this proposal imports torch,
cuml, or faiss.
