/**
 * density.ts — the base raster.
 *
 * Fetches per-corpus u32 count planes (`density/z{z}/{x}_{y}.{corpus}.u32`),
 * keeps them on the main thread (hover needs the raw counts for the composition
 * readout), and hands copies to the compose worker to turn into RGBA. The PNG
 * render in the pack is a convenience/fallback only — client-side recomposition
 * from the planes is what makes corpus toggles free.
 */

import type { Manifest } from "./types";
import { fetchBufferOptional } from "./net";
import { hexToRgb, CORPUS_FALLBACK } from "./palette";
import ComposeWorker from "./worker/compose.worker?worker";
import type { ComposeRequest, ComposeResponse } from "./worker/compose.worker";

export type ComposeMode = "combined" | "dominant";

interface TileEntry {
  key: string;
  z: number;
  x: number;
  y: number;
  state: "loading" | "ready" | "missing";
  planes: Map<number, Uint32Array>;
  total?: Uint32Array;
  bitmap?: ImageBitmap;
  bitmapKey?: string;
  composing?: boolean;
}

const MAX_TILES = 96;

export class DensityStore {
  private base: string;
  private manifest: Manifest;
  private tiles = new Map<string, TileEntry>();
  private worker: Worker;
  private pending = new Map<string, TileEntry>();
  private colors: Record<number, [number, number, number]> = {};
  readonly size: number;
  enabled: number[];
  mode: ComposeMode = "combined";
  onChange: () => void = () => {};

  constructor(base: string, manifest: Manifest) {
    this.base = base;
    this.manifest = manifest;
    this.size = manifest.tile_size ?? 256;
    this.enabled = manifest.corpora.map((c) => c.code);
    manifest.corpora.forEach((c, i) => {
      this.colors[c.code] = hexToRgb(
        c.color ?? CORPUS_FALLBACK[i % CORPUS_FALLBACK.length],
      );
    });
    this.worker = new ComposeWorker();
    this.worker.onmessage = (ev: MessageEvent<ComposeResponse>) => {
      this.onComposed(ev.data);
    };
  }

  get renderKey(): string {
    return `${this.mode}:${[...this.enabled].sort((a, b) => a - b).join(",")}`;
  }

  setEnabled(codes: number[]) {
    this.enabled = codes;
    this.invalidateBitmaps();
  }

  setMode(mode: ComposeMode) {
    this.mode = mode;
    this.invalidateBitmaps();
  }

  private invalidateBitmaps() {
    for (const t of this.tiles.values()) {
      if (t.bitmapKey !== this.renderKey && t.state === "ready") {
        t.composing = false;
      }
    }
    this.onChange();
  }

  private levelInfo(z: number) {
    return this.manifest.density?.levels.find((l) => l.z === z);
  }

  levelNorm(z: number): number {
    return this.levelInfo(z)?.max_count ?? 0;
  }

  /** Does the pack claim this tile exists? (sparse packs omit empty tiles) */
  tileExists(z: number, x: number, y: number): boolean {
    const lvl = this.levelInfo(z);
    if (!lvl) return false;
    const n = 1 << z;
    if (x < 0 || y < 0 || x >= n || y >= n) return false;
    if (!lvl.tile_list) return true;
    return lvl.tile_list.includes(`${x}_${y}`);
  }

  private corporaFor(z: number, x: number, y: number): number[] {
    const lvl = this.levelInfo(z);
    const listed = lvl?.tile_corpora?.[`${x}_${y}`];
    return listed ?? this.manifest.corpora.map((c) => c.code);
  }

  private key(z: number, x: number, y: number) {
    return `${z}/${x}/${y}`;
  }

  /** Composed bitmap for a tile, kicking off load/compose if needed. */
  getBitmap(z: number, x: number, y: number): ImageBitmap | undefined {
    const t = this.ensure(z, x, y);
    if (!t || t.state !== "ready") return undefined;
    if (t.bitmapKey !== this.renderKey && !t.composing) this.requestCompose(t);
    return t.bitmap;
  }

  /** Raw planes for hover readout. */
  getTile(z: number, x: number, y: number): TileEntry | undefined {
    const t = this.tiles.get(this.key(z, x, y));
    return t && t.state === "ready" ? t : undefined;
  }

  ensure(z: number, x: number, y: number): TileEntry | undefined {
    if (!this.tileExists(z, x, y)) return undefined;
    const k = this.key(z, x, y);
    let t = this.tiles.get(k);
    if (t) {
      this.tiles.delete(k);
      this.tiles.set(k, t); // LRU bump
      return t;
    }
    t = { key: k, z, x, y, state: "loading", planes: new Map() };
    this.tiles.set(k, t);
    this.evict();
    void this.load(t);
    return t;
  }

  private evict() {
    while (this.tiles.size > MAX_TILES) {
      const oldestKey = this.tiles.keys().next().value as string | undefined;
      if (oldestKey === undefined) break;
      const victim = this.tiles.get(oldestKey);
      if (victim?.composing) break; // don't evict something mid-flight
      victim?.bitmap?.close?.();
      this.tiles.delete(oldestKey);
    }
  }

  private async load(t: TileEntry) {
    const codes = this.corporaFor(t.z, t.x, t.y);
    const results = await Promise.all(
      codes.map(async (code) => {
        const url = `${this.base}density/z${t.z}/${t.x}_${t.y}.${code}.u32`;
        const buf = await fetchBufferOptional(url).catch(() => null);
        return [code, buf] as const;
      }),
    );
    const npx = this.size * this.size;
    const total = new Uint32Array(npx);
    for (const [code, buf] of results) {
      if (!buf) continue;
      const arr = new Uint32Array(buf, 0, Math.min(npx, buf.byteLength >> 2));
      t.planes.set(code, arr);
      for (let i = 0; i < arr.length; i++) total[i] += arr[i];
    }
    t.total = total;
    t.state = t.planes.size > 0 ? "ready" : "missing";
    if (t.state === "ready") this.requestCompose(t);
    this.onChange();
  }

  private requestCompose(t: TileEntry) {
    if (t.composing || t.state !== "ready") return;
    t.composing = true;
    const planes: Record<number, ArrayBuffer> = {};
    const transfer: ArrayBuffer[] = [];
    for (const code of this.enabled) {
      const p = t.planes.get(code);
      if (!p) continue;
      // copy: the worker transfers its input, and we keep the originals
      const copy = p.slice();
      planes[code] = copy.buffer;
      transfer.push(copy.buffer);
    }
    const req: ComposeRequest = {
      key: `${t.key}|${this.renderKey}`,
      planes,
      enabled: this.enabled,
      mode: this.mode,
      norm: this.levelNorm(t.z),
      size: this.size,
      colors: this.colors,
    };
    this.pending.set(req.key, t);
    this.worker.postMessage(req, transfer);
  }

  private async onComposed(res: ComposeResponse) {
    const t = this.pending.get(res.key);
    this.pending.delete(res.key);
    if (!t) return;
    t.composing = false;
    const renderKey = res.key.split("|")[1];
    const img = new ImageData(
      new Uint8ClampedArray(res.rgba),
      res.size,
      res.size,
    );
    const bmp = await createImageBitmap(img);
    t.bitmap?.close?.();
    t.bitmap = bmp;
    t.bitmapKey = renderKey;
    this.onChange();
    // the toggle may have moved on while we were composing
    if (renderKey !== this.renderKey) this.requestCompose(t);
  }

  /** Per-corpus counts for one bin, for the hover panel. */
  binCounts(
    z: number,
    tx: number,
    ty: number,
    bx: number,
    by: number,
  ): { code: number; count: number }[] | undefined {
    const t = this.getTile(z, tx, ty);
    if (!t) return undefined;
    const i = by * this.size + bx;
    const out: { code: number; count: number }[] = [];
    for (const c of this.manifest.corpora) {
      const p = t.planes.get(c.code);
      out.push({ code: c.code, count: p ? p[i] : 0 });
    }
    return out;
  }

  dispose() {
    this.worker.terminate();
    for (const t of this.tiles.values()) t.bitmap?.close?.();
    this.tiles.clear();
  }
}
