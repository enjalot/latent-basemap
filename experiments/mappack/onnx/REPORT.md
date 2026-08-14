# ONNX map-head export + in-browser projection POC (PLAN6)

2026-08-14 · GSV · CPU only · saved by the session agent from the build agent's inline report.

## Summary

Inference is a plain ResidualBottleneckMLP forward (fneg/kernel fields shape training loss only, carried in models.json as provenance). Export: fp32, opset 17, dynamic batch, ops {Gemm, Relu, Add}.

## Parity (4/4 tests pass)

torch vs onnxruntime, 10k random unit-norm rows (gate < 1e-4):
- sandbox-2m-umap-md000-x4-fneg10: max 8.583e-5, p99 2.03e-5 — pass
- sandbox-12500k-umap-md000-x2-fneg10: max 9.799e-5, p99 2.83e-5 — pass
(An order of magnitude tighter than the R0026 standard under a 10x stricter gate.)

vs sealed coordinates, 1k real substrate rows: max abs 1.5e-5 (2m) / 5.0e-5 (12.5m) = ~2e-7..6e-7 of extent diagonal (~2e-4 px at 1024^2) — reproduces the sealed maps exactly for practical purposes. fp16 NOT exported (R0026 precedent); web-safe graph asserted by test.

## Exports

/data/latent-basemap/mappacks/<map_id>/model/map_head.onnx (47,239,092 B each; 11,809,282 params) + models.json (schema mappack-models-v1) carrying the frame contract derived from map_renders.py: robust_extent values, u/v mapping, the 180-degree render rotation (verified empirically against density.png), u16 quantization = round(u*65535).

## Browser POC — PASS

Zero console/page errors, zero failed requests; screenshot mapviewer/projection-poc/reference/poc.png. Browser vs python reference (sentence-transformers + torch head): cosine ~1.0, delta-xy <= 1.4e-8 of extent diag on all 5 test strings (fp32 encoder). int8 encoder: cosine 0.988-0.994 (fails parity gate) but landing shift 0.009-0.139% of diag (<= ~1.4 px) — visually indistinguishable; shipped behind a selector, fp32 default.

Semantic sanity (500k-row cosine stream): in-distribution text lands on coherent neighbors (0.19-0.29% median neighbor distance); OOD text (German, code-vs-prose) has none — natural hook for the deferred TwoNN confidence overlay.

Timing (WASM, 1 thread): model load 0.3-0.4 s; embed 5-25 ms/string; map head ~2 ms.

## Asset inventory / size decision

fp32 profile ~143 MB one-time; int8 profile ~78 MB (ORT wasm 11.2 MB + tokenizer 0.7 MB + transformers.js 0.9 MB + encoder 23 MB int8 / 90 MB fp32 + map head 45 MB). PLAN6 budgeted 35-50 MB -> published-profile default is a product decision (int8 recommended; fp16 head storage is the next lever but must be parity-measured first, R0026's fp16 miss remembered).

## Integration caveats

- ort.env.wasm.wasmPaths must be an ABSOLUTE URL (new URL('./vendor/ort/', document.baseURI).href) — the one real integration bug.
- numThreads=1 (gh-pages cannot send COOP/COEP).
- Pin transformers.js 3.x (tokenizer only, allowRemoteModels=false).
- Serve over HTTP, never file://.
- Symlinks in map/ point into /data; gh-pages needs real copies.
- Scratch venv at /data/latent-basemap/envs/mappack-onnx; repo .venv untouched.
