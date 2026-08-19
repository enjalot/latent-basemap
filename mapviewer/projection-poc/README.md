# projection-poc — "where would my text land?" in the browser

A single static page (no build step, no CDN at runtime, no backend, no GPU):

    text → MiniLM (ONNX, onnxruntime-web WASM) → 384-d unit vector
         → map head (ONNX fp32, dynamic batch) → (x, y)
         → frame transform from the pack manifest → dot on the map's density PNG

The map head is exported by `experiments/mappack/onnx/export_map_head.py`; the
`map/` entries here are symlinks into
`/data/latent-basemap/mappacks/sandbox-2m-umap-md000-x4-fneg10/`.

## Run it

```bash
python3 -m http.server 8971          # must be HTTP; file:// cannot load modules/wasm
# then open http://localhost:8971/index.html
```

Click **Load models** (fp32 encoder ≈ 143 MB total, int8 ≈ 78 MB), type text,
click **Project →**.

## Verify it headless

```bash
source ~/.nvm/nvm.sh
npm i -D playwright-core                       # once
LD_LIBRARY_PATH=/tmp/libs/extracted/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
  node verify_headless.mjs [--encoder onnx/model_quantized.onnx]

cd reference && CUDA_VISIBLE_DEVICES="" HF_HOME=/data/hf \
  /data/latent-basemap/envs/mappack-onnx/bin/python \
  reference_projection.py --compare browser_results.json
```

`verify_headless.mjs` starts its own static server, asserts a finite numeric
(x, y) per test string and zero console/page/request errors, and writes
`reference/browser_results.json` + `reference/poc.png`.
`reference_projection.py` re-embeds the same strings with
sentence-transformers + the torch map head and gates cosine > 0.999 and
|Δxy| < 0.5 % of the map's extent diagonal.
`reference/neighbor_sanity.py` checks that a projected text lands near its
nearest substrate rows' sealed coordinates.

## Layout

```
index.html                       the whole POC
vendor/ort/                      onnxruntime-web 1.22.0, WASM EP only (11.3 MB)
vendor/transformers/             transformers.js 3.8.1 — AutoTokenizer only
models/Xenova/all-MiniLM-L6-v2/  tokenizer + onnx/model.onnx (fp32) + model_quantized.onnx
map/                             symlinks: map_head.onnx, models.json, density.png
reference/                       python reference + generated verification output
```

## Gotchas worth keeping

- `ort.env.wasm.wasmPaths` must be an **absolute** URL — ORT resolves it
  against its own module URL, so `'./vendor/ort/'` becomes
  `/vendor/ort/vendor/ort/…` and session creation fails with "no available
  backend".
- `ort.env.wasm.numThreads = 1`: threads need COOP/COEP (cross-origin
  isolation), which a plain static host (gh-pages) cannot send.
- Pin transformers.js to **3.x**: 4.x moves tokenization into the
  `@huggingface/tokenizers` WASM package. Here it is used only for
  `AutoTokenizer`, with `env.allowRemoteModels = false` and
  `env.localModelPath = './models/'`, so exactly one WASM runtime loads.
- Embedding semantics must match sentence-transformers: mean-pool over the
  attention mask, then L2 normalize (implemented explicitly in `index.html`).
- The int8 encoder is 4× smaller and shifts the landing by ≤ 0.14 % of the
  extent diagonal (~1.4 px at 1024²), but only fp32 reaches cosine > 0.999
  against sentence-transformers.
- The frame transform (extent, 180°-rotated PNG placement, u16 quantization)
  comes from `map/models.json` — never hardcode it.
- Static hosts that don't follow symlinks need real copies of `map/*`.
