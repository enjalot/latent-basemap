# Metric Option B — the restored TWO-SIDED `purity_fidelity_k256` criterion

**Status: PROPOSAL. This registers nothing.** No round file, queue manifest,
gate artifact or capability was created or modified to produce it. Every sealed
artifact was opened read-only; nothing under `latent-labs/basemap-100m/` was
touched. All work is CPU-only, `CUDA_VISIBLE_DEVICES=""`, `0.0` GPU-h.

Written 2026-08-13 on `gsv`. Implementation:
`gsv:~/code/latent-basemap/experiments/metrics/k256_two_sided.py`.

---

## 0. Why this exists

`purity_fidelity_k256` = `map_agreement / hi_D_agreement` at `k = 256` centroid
labels. **Ideal is `1.0`**: the map's 2D neighbourhoods agree with the `k = 256`
labelling exactly as often as the high-D reference's do.

| ratio | reading |
| --- | --- |
| `> 1` | map is **purer than reality** — manufactured separation |
| `= 1` | perfectly faithful |
| `< 1` | map has **lost** structure |

R0255 registered a two-sided band on the unfolded log-ratio. R0257 judged the
first three 6.25M rung maps with it and **all three failed, from above**. The
owner's 2026-08-12 ruling (`OWNER-DECISIONS-PENDING.md` §5) adopted option 2 —
re-register both purity criteria as one-sided floors that fail under-separation
only — and R0260 executed it. The over-separation side became a descriptive
diagnostic that, in R0260's own words, *"Nothing reads."*

The 2026-08-12/13 sandbox finding reframes that: **over-separation at `k256` is
the bead-collapse signature.** The R0260 ruling therefore silenced the only
registered instrument that was detecting collapse. This artifact rebuilds that
instrument and computes what it would say, so the owner can decide on numbers
rather than on argument.

---

## 1. The criterion, exactly as the program defines it

Two-sided, on the **UNFOLDED** log-ratio. Never fold a two-sided ratio about
`1.0` — review-0222-01 measured that the fold reflects about `r = 1`, which is
*not* the family centre (`r_geo = 1.0115` at `n = 29`), puts all deviation on
one side of a hard ceiling, and roughly **triples** the defining-cell failure
rate. R0255's own artifact publishes the purity entries of `registered_floors`
as `null` for this reason:

> the gated purity criterion is the two-sided band on the UNFOLDED ratio below.
> A consumer reading a folded one-sided purity floor would fail cells this gate
> passes — the trap review-0231-01 found in R0231.

```
centre = median( log r_i )                                  over the family
scale  = MAD_n( log r_i ) = 1.4826 · median| log r_i − centre |
band   = [ exp(centre − k2·scale) , exp(centre + k2·scale) ]     inclusive
```

API (`k256_two_sided.py`):

- `fit_band(family_ratios, k2=None) -> {center, lower, upper, k2, n, log_center, log_scale, log_lower, log_upper, …}`
- `judge(ratio, band) -> {verdict, direction, separation, margin, z_madn, …}`
  where `direction ∈ {"over", "under", "inside"}` distinguishes over- from
  under-separation, and `separation` reports the side of the ideal `1.0`
  independently of pass/fail (the sealed-artifact convention).
- `judge_one_sided(ratio)` — the R0260 criterion currently in force.
- `judge_folded_floor(ratio)` — the folded descriptive floor, present only so
  the folding regression can be written down. It is never used to gate.

---

## 2. The recalibrated `k2` at `n = 29`

Method: R0234 §A, reimplemented independently. Draw `families` samples of size
`n` from `N(0,1)`; per family take `c = median`, `s = MAD_n`; find the minimal
`k` with `Φ(c + k·s) − Φ(c − k·s) ≥ content` by vectorised bisection; `k2` is
the `confidence`-quantile of those minimal `k`. Content `0.95`, confidence
`0.95`. The null reads no cell of this program.

**`k2` at `n = 29` = `3.147763220677517`** (4,000,000 families, seed
`20260809`).

| quantity | this recalibration | sealed value | rel. Δ | sealed source |
| --- | ---: | ---: | ---: | --- |
| `k2` at `n = 13` (**validation gate**) | `4.452408030159920` | `4.452408030160313` | `−8.8e-14` | R0234 `calibration.n13.candidates.median_minus_k_madn.two_sided.multiplier` |
| `k1` at `n = 13` | `3.7363661743730416` | `3.7363661743730416` | `0.0` (bitwise) | same |
| **`k2` at `n = 29`** | **`3.147763220677517`** | `3.147763220676551` | `+3.1e-13` | R0255 `calibration.n29.candidates.median_minus_k_madn.two_sided.multiplier` |
| `k1` at `n = 29` | `2.6934393367626033` | `2.6934393367626024` | `+3.3e-16` | same |

The gate the task set was 2 %; the simulator lands at `1e-13`, i.e. it
reproduces the sealed protocol to bisection tolerance, not merely
statistically. The reference implementation it agrees with is
`gsv:~/code/latent-basemap-run/basemap/round0234_calibration.py` (read-only,
unmodified, and not consulted before the first run).

Because the recalibrated value differs from the sealed one only at `1e-13`, the
band below is fitted with the **sealed** `k2 = 3.147763220676551`, so that every
published digit is comparable bitwise. `--recalibrate` swaps in the simulated
value and moves no verdict.

---

## 3. The band on the sealed `n = 29` family

Family: the 29 sealed defining cells (`exact-seed42 … exact-seed70`), raw
unfolded `k256` ratios from
`gsv:/data/latent-basemap/runs/round-0255/queue/artifacts/minilm-mixed-2m-seed-family-panel-n29-v1/seed-family-panel-n29.json`
(`sha256 76590920ff212113d037676ac7dbd2331d6d3a366228d031607d95e58f88bead`),
field `raw_purity_ratios.<seed>.k256`. Range `0.9885 – 1.0417`.

| quantity | value |
| --- | ---: |
| `log` centre (median) | `0.01143437762566317` |
| `log` scale (`MAD_n`) | `0.01339863133626119` |
| `k2` | `3.147763220676551` |
| **band lower** | **`0.9697263687993266`** |
| **band upper** | **`1.0550731452902518`** |
| geometric centre | `1.0115` |

This reproduces R0255's `registered_two_sided_bands.purity_fidelity_k256`
**bitwise** — the restored criterion is not a new gate, it is the one the
program already registered and R0260 retired. The band straddles the ideal
`1.0`, and `1.0` passes it (a fact the retired `k1024` band could not claim; its
upper edge `0.7719564268121961` failed a perfectly faithful map by `+0.228`,
which is what made option 2 attractive at `k1024`).

Quantisation: `panel_v2` rounds every purity ratio to four decimals inside the
scorer, so the band inherits `±5e-5` in `r`. No verdict below is within `5e-5`
of an edge.

For contrast, the criterion currently in force (R0260,
`minilm-mixed-2m-one-sided-purity-floors-n29-v1`) is the same centre and scale
with the one-sided multiplier: `ratio_floor = exp(centre − k1·scale) =
0.9756474051324426`, **no upper edge at any magnitude**.

---

## 4. Would-be verdicts

45 maps with a published `k256` value, located read-only:

- 29 sealed defining cells + 12 held-out cells (cuvs-igd48 seeds 42–44,
  cluster-spill c4/c8/c16 seeds 42–44), from
  `.../round-0255/queue/artifacts/minilm-mixed-2m-calibrated-madn-floors-n29-v1/minilm-calibrated-madn-floors-n29.json`,
  field `scoring_by_candidate.median_minus_k_madn.all_cells.cells[].metrics.purity_fidelity_k256.raw_ratio`;
- the R0255 replay control (`replay_control_cell.raw_purity_ratios.k256`, not a
  family cell);
- the three 6.25M rung maps, from
  `.../round-0260/queue/artifacts/minilm-mixed-6250k-rung-readjudication-one-sided-v1/readjudication-0257-one-sided.json`,
  field `re_adjudication[].per_metric.purity_fidelity_k256.observed`, which that
  artifact certifies bitwise identical to R0257's published values.

| map | N | ratio | one-sided (in force) | restored two-sided | direction | z (MAD_n) |
| --- | ---: | ---: | --- | --- | --- | ---: |
| exact-seed42 | 2,000,000 | 1.0216 | PASS | PASS | inside | +0.742 |
| exact-seed43 | 2,000,000 | 1.0059 | PASS | PASS | inside | -0.414 |
| exact-seed44 | 2,000,000 | 1.0046 | PASS | PASS | inside | -0.511 |
| exact-seed45 | 2,000,000 | 0.9929 | PASS | PASS | inside | -1.385 |
| exact-seed46 | 2,000,000 | 1.0049 | PASS | PASS | inside | -0.489 |
| exact-seed47 | 2,000,000 | 0.9932 | PASS | PASS | inside | -1.363 |
| exact-seed48 | 2,000,000 | 1.0370 | PASS | PASS | inside | +1.858 |
| exact-seed49 | 2,000,000 | 1.0099 | PASS | PASS | inside | -0.118 |
| exact-seed50 | 2,000,000 | 1.0120 | PASS | PASS | inside | +0.037 |
| exact-seed51 | 2,000,000 | 1.0024 | PASS | PASS | inside | -0.674 |
| exact-seed52 | 2,000,000 | 1.0055 | PASS | PASS | inside | -0.444 |
| exact-seed53 | 2,000,000 | 1.0065 | PASS | PASS | inside | -0.370 |
| exact-seed54 | 2,000,000 | 1.0115 | PASS | PASS | inside | +0.000 |
| exact-seed55 | 2,000,000 | 1.0293 | PASS | PASS | inside | +1.302 |
| exact-seed56 | 2,000,000 | 1.0232 | PASS | PASS | inside | +0.858 |
| exact-seed57 | 2,000,000 | 1.0259 | PASS | PASS | inside | +1.055 |
| exact-seed58 | 2,000,000 | 1.0313 | PASS | PASS | inside | +1.447 |
| exact-seed59 | 2,000,000 | 1.0417 | PASS | PASS | inside | +2.196 |
| exact-seed60 | 2,000,000 | 1.0171 | PASS | PASS | inside | +0.412 |
| exact-seed61 | 2,000,000 | 1.0187 | PASS | PASS | inside | +0.529 |
| exact-seed62 | 2,000,000 | 0.9988 | PASS | PASS | inside | -0.943 |
| exact-seed63 | 2,000,000 | 0.9885 | PASS | PASS | inside | -1.717 |
| exact-seed64 | 2,000,000 | 1.0152 | PASS | PASS | inside | +0.273 |
| exact-seed65 | 2,000,000 | 1.0208 | PASS | PASS | inside | +0.683 |
| exact-seed66 | 2,000,000 | 1.0096 | PASS | PASS | inside | -0.140 |
| exact-seed67 | 2,000,000 | 1.0180 | PASS | PASS | inside | +0.478 |
| exact-seed68 | 2,000,000 | 0.9979 | PASS | PASS | inside | -1.010 |
| exact-seed69 | 2,000,000 | 1.0239 | PASS | PASS | inside | +0.909 |
| exact-seed70 | 2,000,000 | 1.0063 | PASS | PASS | inside | -0.385 |
| cuvs-igd48-seed42 | 2,000,000 | 1.0080 | PASS | PASS | inside | -0.259 |
| cuvs-igd48-seed43 | 2,000,000 | 1.0191 | PASS | PASS | inside | +0.559 |
| cuvs-igd48-seed44 | 2,000,000 | 1.0345 | PASS | PASS | inside | +1.678 |
| cluster-spill-c4-seed42 | 2,000,000 | 0.9946 | PASS | PASS | inside | -1.258 |
| cluster-spill-c4-seed43 | 2,000,000 | 0.9932 | PASS | PASS | inside | -1.363 |
| cluster-spill-c4-seed44 | 2,000,000 | 1.0009 | PASS | PASS | inside | -0.786 |
| cluster-spill-c8-seed42 | 2,000,000 | 0.9893 | PASS | PASS | inside | -1.656 |
| **cluster-spill-c8-seed43** | 2,000,000 | 0.9738 | FAIL | PASS **← FLIP** | inside | -2.835 |
| cluster-spill-c8-seed44 | 2,000,000 | 1.0271 | PASS | PASS | inside | +1.142 |
| cluster-spill-c16-seed42 | 2,000,000 | 0.9989 | PASS | PASS | inside | -0.936 |
| cluster-spill-c16-seed43 | 2,000,000 | 1.0063 | PASS | PASS | inside | -0.385 |
| cluster-spill-c16-seed44 | 2,000,000 | 0.9916 | PASS | PASS | inside | -1.483 |
| replay-control-seed42 | 2,000,000 | 1.0216 | PASS | PASS | inside | +0.742 |
| **ladder-6250k-h2048-seed42** | 6,250,000 | 1.1012 | PASS | FAIL **← FLIP** | **over** | +6.341 |
| **ladder-6250k-h2048-seed43** | 6,250,000 | 1.1022 | PASS | FAIL **← FLIP** | **over** | +6.409 |
| **ladder-6250k-h2048-seed44** | 6,250,000 | 1.0982 | PASS | FAIL **← FLIP** | **over** | +6.138 |

Machine-readable: `experiments/metrics/k256-two-sided-would-be-verdicts.json`.

### 4.1 The four disagreements

**Three flips in the collapse direction — PASS → FAIL, `direction = over`.**
The three 6.25M rung maps sit `+6.14` to `+6.41` family `MAD_n` above the log
centre, `+3.0` to `+3.3` `MAD_n` *outside* the upper edge, and above the entire
observed 29-cell range (`0.9885 – 1.0417`). Under the criterion in force they
clear the floor by `+0.1226` to `+0.1266` and read PASS. Under the restored
criterion they fail by `−0.0431` to `−0.0471`, in the manufactured-separation
direction. These are the rows the R0260 ruling excused — and, per the sandbox
finding, exactly the signal the collapse produces.

The margins that make them PASS are not evidence about the maps: review-0260-01
recorded that the three PASSes were *arithmetically determined* before R0260's
queue started, since a one-sided lower floor is structurally incapable of
failing maps that sit above the entire family range on all three gated metrics.
Both R0260 claims that rest on those PASSes are **already blocked** by
review-0260-01 (`blocks:` list, §5.1 below).

**One flip the other way — FAIL → PASS, `cluster-spill-c8-seed43` at `0.9738`.**
This is a genuine, previously unrecorded consequence of the ruling and is
recorded here in the interest of the whole picture, not just the convenient
half. Because `k1 = 2.6934 < k2 = 3.1478`, R0260's one-sided floor is **tighter
on the under-separation side than the retired band was**. This held-out 2M cell
passed R0255's two-sided gate (`passes: true` in the sealed scoring) and fails
R0260's floor by `−0.001847`. R0260's registration artifact checked only the 29
*defining* cells against the new floor (`family_cells_failing_each_criterion`
is empty on all three metrics) and did not re-score the 12 held-out cells, so
this disagreement is not in any published artifact. Review-0260-01 already noted
the mechanism in the aggregate — the panel false-alarm rate *rose* from
`0.034072` to `0.036217` because a side was dropped — but not this cell.

Restoring the two-sided band un-fails it, which is the correct outcome under the
program's own calibration: `z = −2.835` is inside a `95/95` two-sided tolerance
band at `n = 29`.

### 4.2 Supplementary — not maps, not gated

Review-0257-01's matched-2M transfer diagnostic re-scored the same three maps at
`N = 2M` with centroids refit at the registered `k` and got `k256`
`1.0939 / 1.0962 / 1.0916` (review-0257-01 §D table). Those are `+5.69` to
`+6.00` `MAD_n`, still far outside the band. The over-separation is therefore
not a scoring-`N` artifact; the FAIL survives the transfer control. These rows
are a reviewer diagnostic, not registered map observations, and are excluded
from the table and the flip count.

---

## 5. What re-registration would actually require

**This proposal cannot register anything.** Registration is a round action, and
the criterion in force was set by an owner ruling. Restoring the two-sided
`k256` criterion needs, in order:

1. **An owner ruling that partially supersedes the 2026-08-12 §5 ruling.** The
   ruling's clause 1 ("One-sided, under-separation only … drops the direction
   that cannot [indicate a defect]") is precisely the premise the sandbox
   finding contradicts: at `k256`, over-separation *is* a defect signature. The
   new evidence is same-universe and post-dates the ruling, which is the only
   basis on which the standing rules let a criterion be redefined. Clause 3
   ("registered before any further rung map is judged") binds the restoration
   too: it must land before the next rung map is judged, or it is retro-fitting.
2. **A round that registers it**, fitting the band from the same sealed `n = 29`
   family (clause 2's provenance requirement is satisfied unchanged — the fit
   here is bitwise R0255's), with the ordering guard given the positive control
   review-0260-01 found missing in R0260.
3. **An independent review** of that round.

### 5.1 Exactly which claims a restoration would supersede

Scoped to `purity_fidelity_k256` only. Nothing below touches `ffr` or
`purity_fidelity_k1024`.

| target | what happens |
| --- | --- |
| `capability:minilm-mixed-2m-one-sided-purity-floors-n29-v1` — `registered_criteria.purity_fidelity_k256` and `registered_ratio_floors.purity_fidelity_k256` (`0.9756474051324426`), and the clause *"There is no upper edge: a ratio above the floor passes at any magnitude"* | **superseded for `k256`**. Its `ffr` floor and its `k1024` criterion stand. |
| That artifact's `supersedes_purity_criteria_in: minilm-mixed-2m-calibrated-madn-floors-n29-v2` | **partially reversed for `k256`**: R0255/R0256's two-sided `k256` band returns to force. R0255's numbers were never wrong and are not rewritten. |
| `capability:minilm-mixed-6250k-rung-readjudication-one-sided-v1` — the `k256` component of `re_adjudication[].one_sided_verdict` = PASS for all three maps | **superseded for `k256`** by a new dated addendum. Under the ruling's clause 5 the addendum adds a record; it never rewrites R0260, exactly as R0260 never rewrote R0257. |
| `claim:r0260-the-three-6250k-maps-pass-the-registered-phase-3-purity-criteria` | **already blocked** by review-0260-01. A restoration converts it from blocked to superseded. |
| `claim:r0260-the-registered-acceptance-rule-did-not-have-to-be-exercised` | **already blocked** by review-0260-01. Unaffected by this proposal; listed so the record is complete. |
| R0257's three published FAILs | **stand, unchanged.** They were true statements about the criterion in force when they were made, and under a restored criterion they are true again. This proposal does not rewrite them and does not need to. |
| Owner ruling §5 clause 4 ("the discarded side is still published") | **satisfied and strengthened.** Under a restored two-sided criterion the one-sided floor value should keep being published beside every verdict as the descriptive diagnostic, so no measurement is dropped in either direction. |
| Design-0260 (option 3, reference-anchored quality criterion) | **unaffected and still the target.** Restoring the band is a stopgap that makes the collapse direction gateable now; it does not supply the multi-`N` acceptable-deviation calibration option 3 requires. The two should not be traded against each other. |

### 5.2 What this proposal does not establish

- **That the three 6.25M maps are bad maps.** It establishes that they are
  `+6.1` to `+6.4` family `MAD_n` above the 2M family's log centre at `k256` and
  outside a `95/95` two-sided tolerance band. Nothing here connects a `1.10`
  `k256` ratio to a downstream use, and review-0257-01 made the same
  reservation.
- **Anything at 12.5M, 25M, 50M or 100M.** Three cells at one rung.
- **That two-sided is the right long-run answer.** Review-0257-01's structural
  objection — the band measures conformity to the 2M family, not quality — is
  untouched by this work and is why option 3 exists. At `k256`, and only at
  `k256`, the band's upper edge and the reference anchor happen to point the
  same way (`1.0` is inside the band, `1.0550` is only `+5.5 %` past faithful),
  which is what makes restoration a defensible stopgap here and *not* at
  `k1024`, where the upper edge `0.7719` fails a perfectly faithful map.

---

## 6. Tests

`experiments/metrics/tests/test_k256_two_sided.py` — **25 passed**, `6.3 s`,
CPU only.

| group | what is planted / checked |
| --- | --- |
| band identity | fit reproduces R0255's sealed band bitwise; band straddles `1.0`; R0260's floor re-derives from the same centre/scale with `k1` |
| **(a) over** | planted cell `1.05×` above the upper edge → FAIL, `direction == "over"`; the three real rung maps → FAIL/over while the in-force criterion passes them; a `5.0` ratio (R0260's own sidedness control, which it PASSED) → FAIL/over |
| **(b) under** | planted cell `0.95×` below the lower edge → FAIL, `direction == "under"`; one part in a million below the edge → FAIL/under |
| **(c) conforming** | band centre passes; ratio `1.0` passes; all 29 sealed cells pass; both edges inclusive |
| **(d) simulator gate** | 1,000,000-family Gaussian null at `n = 13` reproduces R0234's `k2 = 4.452408030160313` and `k1 = 3.7363661743730416` within 2 % (actual: `1e-13`); same at `n = 29`; `k2 > k1`; degenerate `n` refused |
| **(e) folding regression** | the asked case — a cell in `[0.9620488…, 0.9697263…)` **passes** the folded descriptive floor and **fails** the unfolded band as under-separation; the converse review-0222 case — `exact-seed48` at `r = 1.037`, folded fidelity `0.96432 <` the `n = 8` folded floor `0.9660625420699066`, so the fold FAILS it while the unfolded band PASSES it; and a cell and its reciprocal fold identically but judge `over` vs `under` |
| hygiene | non-positive / non-finite ratios refused; degenerate families (`MAD_n = 0`, `n = 1`, negative ratios) refused |
| table | the flip set is exactly the four maps named in §4.1 |

The folding regression is the load-bearing one: the folded floor and the
two-sided band are **different criteria that disagree in both directions**,
because the sealed family does not centre on `1.0`. A folded criterion is
structurally unable to report `direction`, which is the whole point of restoring
this one.

---

## 7. Reproduction

```bash
cd /home/enjalot/code/latent-basemap

# recalibrate k1/k2 on the Gaussian null at n = 13 and n = 29 (~21 s, CPU)
CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/metrics/k256_two_sided.py calibrate

# the full would-be-verdict table (uses the sealed k2; add --recalibrate to simulate it)
CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/metrics/k256_two_sided.py verdicts

# tests
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
    experiments/metrics/tests/test_k256_two_sided.py -q
```

Outputs (written by this proposal, under `experiments/metrics/` only):
`k256-two-sided-calibration.json`, `k256-two-sided-would-be-verdicts.json`.

### Sealed inputs, all read-only

| role | path |
| --- | --- |
| 29 sealed `k256` ratios | `gsv:/data/latent-basemap/runs/round-0255/queue/artifacts/minilm-mixed-2m-seed-family-panel-n29-v1/seed-family-panel-n29.json` |
| sealed band, `n = 29` calibration, 41-cell scoring, folded floor | `gsv:/data/latent-basemap/runs/round-0255/queue/artifacts/minilm-mixed-2m-calibrated-madn-floors-n29-v1/minilm-calibrated-madn-floors-n29.json` |
| `n = 13` calibrated multipliers (validation target) | `gsv:/data/latent-basemap/runs/round-0234/queue/artifacts/minilm-mixed-2m-calibrated-robust-floors-n13-v1/minilm-calibrated-robust-floors-n13.json` |
| one-sided criterion in force | `gsv:/data/latent-basemap/runs/round-0260/queue/artifacts/minilm-mixed-2m-one-sided-purity-floors-n29-v1/minilm-one-sided-purity-floors-n29.json` |
| the three rung maps' ratios + both verdicts | `gsv:/data/latent-basemap/runs/round-0260/queue/artifacts/minilm-mixed-6250k-rung-readjudication-one-sided-v1/readjudication-0257-one-sided.json` |
| R0257's published FAILs | `gsv:/data/latent-basemap/runs/round-0257/queue-correction-2/artifacts/minilm-mixed-6250k-rung-gate-verdict-v1/judge_6250k-rung-verdict.json` |
| reference calibration implementation (read, not modified) | `gsv:~/code/latent-basemap-run/basemap/round0234_calibration.py` |

Documents read (read-only, unmodified):
`gsv:~/code/latent-labs/basemap-100m/{OWNER-DECISIONS-PENDING.md §5,
review-0222-2026-08-08-01.md, result-0234-2026-08-09.md,
result-0255-2026-08-11.md, review-0257-2026-08-11-01.md,
result-0260-2026-08-12.md, review-0260-2026-08-12-01.md,
design-0260-reference-anchored-quality-criterion.md}`.
