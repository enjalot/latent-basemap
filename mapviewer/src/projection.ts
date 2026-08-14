/**
 * projection.ts — "where would my text land?" in the browser.
 *
 *   text → MiniLM (ONNX, onnxruntime-web WASM) → 384-d unit vector
 *        → the pack's map head (ONNX fp32, dynamic batch) → (x, y)
 *        → the pack frame → (u, v) in the viewer's world space → a dot on the
 *          OVERLAY layer
 *
 * Lifted from `mapviewer/projection-poc/` (see its README). The gotchas that
 * cost time there are all load-bearing and are re-stated inline below:
 * absolute `wasmPaths`, one thread, transformers.js pinned to 3.x for the
 * tokenizer only, and mean-pool-then-L2 to match sentence-transformers.
 *
 * FRAME CONTRACT — the one thing that is *not* copied from the POC. The POC
 * drew onto `density.png`, whose placement is rotated 180° (`col = (1-u)*w`,
 * `row = v*h`, from `models.json.frame.png_placement`). The viewer does not
 * draw that PNG: its world space is the pack's own quantization grid,
 *
 *     u = (x - xmin) / (xmax - xmin)
 *     v = (ymax - y) / (ymax - ymin)          (y measured downward)
 *
 * which is exactly `map_pack.quantize()`. So the dot is placed with the PACK
 * manifest extent and NO x-flip; applying the PNG rotation here would mirror
 * the dot across the map. `models.json.frame.extent` is the render frame of
 * density.png and differs from the pack's squared extent — it is deliberately
 * not used for placement, only reported for provenance.
 */

import type { Extent, Manifest } from "./types";
import type { ProgressFn } from "./net";

export type Precision = "int8" | "fp32";

export interface ProjectionResult {
  text: string;
  /** map-head output in the substrate's coordinate space */
  x: number;
  y: number;
  /** viewer world space (unit square, v down) */
  u: number;
  v: number;
  /** u16 quantized, the same integers the point tiers store */
  qx: number;
  qy: number;
  tokens: number;
  embedMs: number;
  headMs: number;
}

/** Byte sizes of the vendored runtime, measured from projection-poc/. */
export const ASSET_BYTES = {
  ortWasm: 11_210_254,
  ortJs: 48_259 + 20_856,
  transformers: 888_173,
  tokenizer: 711_661 + 650 + 366 + 125,
  encoder: { int8: 22_972_370, fp32: 90_387_606 } as Record<Precision, number>,
};

export function runtimeBytes(precision: Precision, headBytes: number): number {
  return (
    ASSET_BYTES.ortWasm +
    ASSET_BYTES.ortJs +
    ASSET_BYTES.transformers +
    ASSET_BYTES.tokenizer +
    ASSET_BYTES.encoder[precision] +
    headBytes
  );
}

const ENCODER_ID = "Xenova/all-MiniLM-L6-v2";
const ENCODER_FILE: Record<Precision, string> = {
  int8: "onnx/model_quantized.onnx",
  fp32: "onnx/model.onnx",
};
const MAX_LENGTH = 256; // all-MiniLM-L6-v2 max_seq_length

/** Where the vendored runtime lives, relative to the deployed page. */
export function assetBase(): string {
  return new URL("./", document.baseURI).href;
}

interface OrtTensorCtor {
  new (type: string, data: unknown, dims: readonly number[]): {
    data: ArrayLike<number>;
    dims: number[];
  };
}
interface OrtModule {
  env: {
    wasm: { wasmPaths: string; numThreads: number; proxy: boolean };
    logLevel: string;
  };
  Tensor: OrtTensorCtor;
  InferenceSession: {
    create(
      bytes: Uint8Array,
      opts: { executionProviders: string[]; graphOptimizationLevel: string },
    ): Promise<OrtSession>;
  };
}
interface OrtSession {
  inputNames: string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: ArrayLike<number>; dims: number[] }>>;
}

/** What `models.json` tells us about the head's input/output names. */
interface ModelsJson {
  map_head: {
    input: { name: string };
    output: { name: string };
    params?: number;
    precision?: string;
  };
  frame?: { extent?: number[]; bins?: number };
  encoder?: { name?: string };
}

async function fetchWithProgress(
  url: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${url}`);
  const total = Number(res.headers.get("content-length") || 0);
  if (!res.body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress?.(buf.byteLength, buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out;
}

export class Projector {
  readonly packBase: string;
  readonly extent: Extent;
  readonly headPath: string;
  readonly modelsJsonPath: string;
  readonly headBytes: number;

  precision: Precision = "int8";
  loaded = false;
  loading = false;
  models: ModelsJson | null = null;
  bytesLoaded = 0;

  private ort: OrtModule | null = null;
  private tokenizer: ((text: string, opts: Record<string, unknown>) => Promise<Record<string, { data: ArrayLike<number> | BigInt64Array; dims: number[] }>>) | null = null;
  private encoder: OrtSession | null = null;
  private head: OrtSession | null = null;

  constructor(opts: {
    packBase: string;
    manifest: Manifest;
    headPath: string;
    modelsJsonPath: string;
    headBytes: number;
  }) {
    this.packBase = opts.packBase;
    this.extent = opts.manifest.extent;
    this.headPath = opts.headPath;
    this.modelsJsonPath = opts.modelsJsonPath;
    this.headBytes = opts.headBytes;
  }

  /** map-head output (x, y) -> viewer world space. See the frame note above. */
  toWorld(x: number, y: number): { u: number; v: number; qx: number; qy: number } {
    const e = this.extent;
    const u = (x - e.xmin) / (e.xmax - e.xmin);
    const v = (e.ymax - y) / (e.ymax - e.ymin);
    const clamp = (t: number) => Math.min(1, Math.max(0, t));
    return {
      u,
      v,
      qx: Math.round(clamp(u) * 65535),
      qy: Math.round(clamp(v) * 65535),
    };
  }

  async load(precision: Precision, onProgress?: ProgressFn): Promise<void> {
    if (this.loading) throw new Error("already loading");
    if (this.loaded && precision === this.precision) return;
    this.loading = true;
    try {
      const base = assetBase();
      const encoderBytes = ASSET_BYTES.encoder[precision];

      // ORT resolves wasmPaths against ITS OWN module URL, so a relative
      // './vendor/ort/' becomes '/vendor/ort/vendor/ort/…' and session creation
      // dies with "no available backend". It must be an absolute URL.
      const ort = (await import(
        /* @vite-ignore */ `${base}vendor/ort/ort.wasm.min.mjs`
      )) as unknown as OrtModule;
      ort.env.wasm.wasmPaths = new URL("vendor/ort/", base).href;
      ort.env.wasm.numThreads = 1; // threads need COOP/COEP; a static host can't
      ort.env.wasm.proxy = false;
      ort.env.logLevel = "error";
      this.ort = ort;

      // transformers.js 3.x, tokenizer only — 4.x moves tokenization into a
      // second WASM package, which would mean two runtimes on the page.
      const tf = (await import(
        /* @vite-ignore */ `${base}vendor/transformers/transformers.min.js`
      )) as unknown as {
        env: Record<string, unknown>;
        AutoTokenizer: { from_pretrained(id: string): Promise<never> };
      };
      tf.env.allowRemoteModels = false;
      tf.env.allowLocalModels = true;
      tf.env.localModelPath = new URL("models/", base).href;
      tf.env.useBrowserCache = false;

      this.models = (await (await fetch(this.modelsJsonPath)).json()) as ModelsJson;

      let done = 0;
      const total = encoderBytes + this.headBytes + ASSET_BYTES.tokenizer;
      const step = (loaded: number) => onProgress?.(done + loaded, total);

      this.tokenizer = (await tf.AutoTokenizer.from_pretrained(
        ENCODER_ID,
      )) as unknown as Projector["tokenizer"];
      done += ASSET_BYTES.tokenizer;
      step(0);

      const encBytes = await fetchWithProgress(
        `${base}models/${ENCODER_ID}/${ENCODER_FILE[precision]}`,
        (l) => step(l),
      );
      this.encoder = await ort.InferenceSession.create(encBytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      done += encBytes.byteLength;

      const headBytes = await fetchWithProgress(this.headPath, (l) => step(l));
      this.head = await ort.InferenceSession.create(headBytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      done += headBytes.byteLength;

      this.bytesLoaded = done;
      this.precision = precision;
      this.loaded = true;
      onProgress?.(total, total);
    } finally {
      this.loading = false;
    }
  }

  /** Mean pooling over the attention mask, then L2 — matches sentence-transformers. */
  private async embed(text: string): Promise<{ vec: Float32Array; tokens: number }> {
    const ort = this.ort!;
    const enc = await this.tokenizer!(text, {
      padding: true,
      truncation: true,
      max_length: MAX_LENGTH,
    });
    const feeds: Record<string, unknown> = {};
    for (const name of this.encoder!.inputNames) {
      const t = enc[name] ?? enc["token_type_ids"];
      if (!t) continue;
      feeds[name] = new ort.Tensor("int64", t.data, t.dims);
    }
    if (this.encoder!.inputNames.includes("token_type_ids") && !enc.token_type_ids) {
      const ids = enc["input_ids"];
      const n = ids.data.length;
      feeds["token_type_ids"] = new ort.Tensor("int64", new BigInt64Array(n), ids.dims);
    }
    const out = await this.encoder!.run(feeds);
    const hidden = out["last_hidden_state"] ?? out[Object.keys(out)[0]];
    const [, seq, dim] = hidden.dims;
    const mask = enc.attention_mask.data as unknown as BigInt64Array;
    const vec = new Float32Array(dim);
    let denom = 0;
    for (let s = 0; s < seq; s++) {
      const m = Number(mask[s]);
      if (!m) continue;
      denom += m;
      const b = s * dim;
      for (let d = 0; d < dim; d++) vec[d] += hidden.data[b + d] * m;
    }
    for (let d = 0; d < dim; d++) vec[d] /= denom || 1;
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += vec[d] * vec[d];
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < dim; d++) vec[d] /= norm;
    return { vec, tokens: seq };
  }

  async project(text: string): Promise<ProjectionResult> {
    if (!this.loaded) throw new Error("projection models not loaded");
    const ort = this.ort!;
    const t0 = performance.now();
    const { vec, tokens } = await this.embed(text);
    const t1 = performance.now();
    const spec = this.models!.map_head;
    const res = await this.head!.run({
      [spec.input.name]: new ort.Tensor("float32", vec, [1, vec.length]),
    });
    const xy = res[spec.output.name].data;
    const t2 = performance.now();
    const x = Number(xy[0]);
    const y = Number(xy[1]);
    return {
      text,
      x,
      y,
      ...this.toWorld(x, y),
      tokens,
      embedMs: t1 - t0,
      headMs: t2 - t1,
    };
  }
}
