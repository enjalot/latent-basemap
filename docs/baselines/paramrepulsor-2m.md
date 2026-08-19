# Upstream ParamRepulsor baseline at 2M

Round ID: `0270`

## Question

How does the authors' released ParamRepulsor implementation score on the same
2M MiniLM substrate and held-out evaluation used for the fneg family?

## Implementation

The adapter installs and calls `parampacmap` from:

```text
https://github.com/hyhuang00/ParamRepulsor
commit be8df72b1ac9041be3aae3d99f16f0d392b492dc
package version 0.1.1rc0
license Apache-2.0
```

The run uses the upstream defaults: 10 near pairs, 20 far pairs, 5 mid-near
pairs, Euclidean distance, PCA to 100 dimensions, a 100-100-100 SiLU network,
batch size 1024, Adam at 0.001, and 450 epochs with the ParamRepulsor weight and
constant schedules. The pilot adds seed 42 and verbose logging. Seeds 43 and 44
are registered follow-ups. These are supported constructor arguments and do not
change an algorithmic default.

The environment follows the upstream CUDA 12.4 lock. At runtime the adapter
checks Python 3.10, package versions, the installed Git commit, and hashes of
the installed estimator, pair builder, dataloader, network, and loss files.

## Comparability

This is an external-method comparison on the same data and evaluator. It is not
a one-axis ablation. ParamRepulsor builds its own Annoy neighbor graph and pair
sets, applies its default PCA, uses a smaller network, and trains for its default
450 epochs. Those differences belong to the method and remain in the receipt.

## Cost

One seed runs `ceil(2,000,000 / 1,024) * 450 = 879,300` upstream training
batches. Before the first batch, the same `fit` call computes PCA, builds a
20-tree Annoy index, and constructs near, far, and mid-near pairs for 2M rows.
The upstream API has no progress callback during that preprocessing and no
cooperative abort callback during `fit`.

The training loader runs in the node process. The default inference loader uses
one PyTorch worker process; the train and panel receipts count each worker launch.
The runner's process group contains the worker if the node exits abnormally.

The first seed is therefore a measured-cost pilot. The queue sets a 30-hour p90
and a 35 GPU-hour cap rather than claiming a precise runtime from small public
benchmarks. The GPU lease remains occupied while the upstream call performs its
CPU preprocessing. Seeds 43 and 44 are implemented but should be queued only
after seed 42 establishes wall time and map quality.

## Evaluation

The panel uses the R0218 frozen high-dimensional reference and centroids plus
the R0265 held-out probes and exact top-10 truth. It reports held-out FFR,
purity fidelity, normalized spacing, and fog. A final CPU node compares each
ParamRepulsor value with the same-seed fneg map and the 13-seed fneg median.
The output does not rank methods or register a gate.

## Outputs

```text
/data/latent-basemap/runs/round-0270/queue/artifacts/
  minilm-mixed-2m-upstream-paramrepulsor-seed*/
    production-config.json
    paramrepulsor.pt
    coordinates.npy
    train-receipt.json
  minilm-mixed-2m-upstream-paramrepulsor-panel-v1/
    paramrepulsor-2m-panel.json
  minilm-mixed-2m-upstream-paramrepulsor-vs-fneg-family-v1/
    paramrepulsor-vs-fneg-2m.json
```
