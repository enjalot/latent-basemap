# Card057 inference preparation — not a GPU release

The isolated worktree starts from4339ea4 (Card040 runtime). It adds no core
training change. `card057_common.py` loads the fixed full1536 three-dimensional
heads, verifies checkpoint tensors and loaded-module locations, and runs one
backbone per batch of256 before a confined output correction.

The branch-free kernel implements the already fixed scalar operator. It copies
active FP32 outputs through `torch.where`, including signed zero; FP64 offset
arithmetic and output transfer are included in the future deployment timing.
Synthetic CPU tests in the root receipt pass28 checks against the frozen
reference operator, including FP32 equality, masks, zero-boundary identity and
norm caps. This is not device equivalence or a real-data quality result.

Before any real treatment output: complete/freeze the GPU driver, canary,
benchmark IDs, scorer, independent audit, runtime manifest and budgeted launch
release. The device canary must compare this implementation with the reference
operator and same-input teacher forward. Follow the root Card057 protocol,
measurement freeze and input manifests; do not infer admission from this file.
