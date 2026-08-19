# Matched 50M fneg-off control

Round ID: `0269`

## Question

Does fneg account for the 50M map's fog reduction, or would the same result
appear under the parent loss?

## Fixed design

The control starts from the R0267 config and changes one training field:

```text
optimizer.fneg_weight: 1.0 -> 0.0
```

Its round, schema, capability, and treatment labels identify it as a control;
those provenance-only changes do not alter training. The predicate normalizes
those labels and the registered loss change, then requires the complete config
to equal the R0267 parent.

The predicate in `basemap/baseline_50m_fneg_off.py` rebuilds the R0267 parent,
rebuilds the expected control, and compares the complete config. It refuses a
change to the graph, row order, model, sampler, dose, optimizer, host-int8 path,
or any other field.

The registered family uses seeds 42, 43, and 44. A seed-43-only queue is an
early paired measurement. The paper should use the three-seed panel unless the
pilot shows that another full family would not be worth its cost.

## Inputs

- R0237 sealed 50M MiniLM substrate and exact k15 fuzzy graph
- R0262 host-int8 substrate, restricted to the verified first-50M prefix
- R0237 held-out reserve rows
- corrected exact-cosine reserve truth used by the superseding R0267 gate
- the R0267 treated result for paired deltas

## Training

Each seed receives 4,162,228 successful positive-LR updates, the same x2 dose
as R0267. Batch size, positive ratio, negative sampling, network, optimizer,
kernel, and transform code are unchanged. With `fneg_weight=0`, the existing
core takes its unweighted binary-cross-entropy branch and emits no fneg
telemetry.

R0267 took about 11.9 GPU-hours per seed. The queue registers a 14-hour p90 per
control seed and reserves 4 GPU-hours for the panel. The full family should
occupy roughly 36 GPU-hours if throughput matches R0267. The seed-43 pilot
should take about 12 GPU-hours plus panel time.

## Evaluation

The train node saves the model and all 50M coordinates. The panel reuses those
coordinates, projects the held-out reserve through the saved model, and scores:

- held-out FFR at `disc = int(50_000_000 * 0.001) = 50,000`
- normalized spacing (`collapse`) on the full map
- fog on the full map
- descriptive k256 and k1024 purity on the first 2M rows

The comparison output reports `fneg_off - fneg_on` for each shared seed and the
mean across the selected seeds. It makes no pass/fail decision.

## Outputs

```text
/data/latent-basemap/runs/round-0269/queue/artifacts/
  minilm-mixed-50000k-fneg-off-x2-md000-hostint8-seed*/
    production-config.json
    model.pt
    coordinates.npy
    train-receipt.json
  minilm-fneg-off-50m-x2-hostint8-panel-v1/
    fneg-off-50m-x2-panel.json
  minilm-50m-fneg-off-vs-fneg-comparison-v1/
    fneg-off-vs-fneg-50m.json
```
