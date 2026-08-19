# Early prototypes

This directory preserves the repository's original Modal launchers, local
experiments, data helpers, and exploratory notebooks. They were introduced
between February 2025 and July 2026 and are not part of the current training or
evaluation workflow.

Do not use these files to infer the published method or reproduce the current
50M/100M runs. Many GPU launchers are explicitly retired by
[`basemap/round0005_retirement.py`](../../basemap/round0005_retirement.py) and
fail before they can launch work. The remaining files target old Modal volumes,
demo datasets, or pre-round experiment layouts and are retained only for
provenance.

Current reader paths are:

- [`README.md`](../../README.md) for the repository map and current result.
- [`REVIEWER_GUIDE.md`](../../REVIEWER_GUIDE.md) for the short scientific route.
- [`paper/paper.md`](../../paper/paper.md) for the method and evaluation.
- [`basemap/round*`](../../basemap/) and [`experiments/round*`](../../experiments/)
  for the preregistered execution code.
- [`experiments/sandbox/`](../../experiments/sandbox/) for treatment-selection
  experiments.

## Contents

**Retired Modal launchers and benchmarks:** `bench_*_modal.py`,
`build_*_modal.py`, `debug_faiss_modal.py`, `sweep_*_modal.py`,
`train_*_modal.py`, and `train_modal.py`.

**Early graph and data helpers:** `edges_local.py`, `edges_modal.py`,
`precompute_local.py`, `psym_modal.py`, `pull_data.sh`, `scope_modal.py`, and
`upload_data_modal.py`.

**Local training, projection, and validation prototypes:** `project_local.py`,
`scale_experiment.py`, `train_local.py`, and `validate_umap.py`.

**Exploratory notebooks:** `test_inference.ipynb` and `test_lance.ipynb`.

Commands embedded in these files reflect their original location and
environment. Use Git history when exact historical behavior matters.
