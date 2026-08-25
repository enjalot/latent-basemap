# Manuscript source

`paper.md` is the source for the paper PDF and the later Moonshine article. The
Markdown uses Pandoc citations from `references.bib`.

## Build a review PDF

From this directory:

```bash
pandoc paper.md --citeproc --pdf-engine=tectonic -o /tmp/latent-basemap-paper.pdf
```

The output goes to `/tmp` so generated PDFs do not enter the repository. Pandoc
and Tectonic are the only build requirements.

## Before external review

1. Read [`RESULTS_CHECKLIST.md`](RESULTS_CHECKLIST.md).
2. List unresolved 100M values:

   ```bash
   rg -n '\{\{100M_[A-Z0-9_]+\}\}' paper.md
   ```

3. Build the PDF and treat every missing-citation warning as an error.
4. Check that the result tables match the sealed records linked from
   [`../REVIEWER_GUIDE.md`](../REVIEWER_GUIDE.md).

The manuscript intentionally distinguishes settled results, exploratory
ablations, prototypes, and unfinished release work. Keep those labels when
adapting the source for Moonshine.
