# Final 100M result checklist

The manuscript is written as though R0268 completed successfully. Do not replace
the tokens from a progress log, a salvaged preview, or an unsealed checkpoint.
Use the final R0268 result and its bound gate artifact.

## Values to fill

| token | required value | source field or calculation |
|---|---|---|
| `{{100M_COLLAPSE_MEAN}}` | three-seed mean, four decimals | final R0268 gate summary |
| `{{100M_COLLAPSE_VALUES}}` | seeds 42/43/44 in that order | final per-seed panel cells |
| `{{100M_FOG_RANGE}}` | minimum to maximum, four decimals | final per-seed panel cells |
| `{{100M_FFR_RANGE}}` | minimum to maximum, four decimals | corrected reserve-projection FFR cells |
| `{{100M_PURITY_SUMMARY}}` | short descriptive range or "reported in Table X" | final descriptive purity block |
| `{{100M_TRAIN_HOURS}}` | measured per-seed range or mean | terminal train receipts, excluding failed attempts |
| `{{100M_TOTAL_HOURS}}` | sum for the three evidence-producing trains | terminal train receipts, excluding failed attempts |

The infrastructure incident may be reported separately as development cost, but
it must not be folded into the per-seed training-time claim.

## Decision rule

The assumed-success prose is valid only if the sealed result says `PASS` and all
of the following hold:

- three-seed mean normalized spacing is in `[0.8650, 1.0505]`;
- every seed has normalized spacing at least `0.8129`;
- every seed has fog at most `0.41207`;
- every seed has held-out FFR at least `0.39906` and clears the registered
  two-standard-error ambiguity rule;
- the metric instruments and reference identities match the preregistration.

If R0268 is `FAIL` or `AMBIGUOUS`, rewrite the abstract and Section 5.3. Do not
preserve the pass narrative by filling only the numbers.

## Release checks

```bash
rg -n '\{\{100M_[A-Z0-9_]+\}\}' paper.md
pandoc paper.md --citeproc --pdf-engine=tectonic -o /tmp/latent-basemap-paper.pdf
```

The first command must return no matches for a release candidate. Also confirm:

- the abstract, Section 5.3, and Section 5.6 use the same sealed values;
- R0268 is linked from `REVIEWER_GUIDE.md` once its result file exists;
- figures use the same coordinate artifacts as the final panel;
- all author, affiliation, venue, license, model, and demo placeholders are
  resolved;
- the AI-assistance disclosure matches the target venue policy.
