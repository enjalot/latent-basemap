/**
 * points.ts — LOD point tier and deep (per-viewport range) point tier.
 *
 * Record layouts (little-endian):
 *   lod.bin     9 B: u16 qx, u16 qy, u32 packed, u8 min_zoom   (min-zoom, then tile order)
 *   xy_id.bin   8 B: u16 qx, u16 qy, u32 packed                (tile order at the finest zoom)
 *   packed = (corpus << 28) | (row_id & 0x0FFFFFFF)
 *   qx/qy quantise u,v in [0,1] over the full extent to 0..65535 (v is y-down)
 */

import type { Manifest } from "./types";
import { RangeReader, type ProgressFn } from "./net";

export interface PointBatch {
  n: number;
  u: Float32Array;
  v: Float32Array;
  corpus: Uint8Array;
  id: Uint32Array;
  minz?: Uint8Array;
}

const Q = 65535;

function decode(
  buf: ArrayBuffer,
  stride: number,
  withMinZoom: boolean,
): PointBatch {
  const bytes = new Uint8Array(buf);
  const n = Math.floor(bytes.byteLength / stride);
  const dv = new DataView(buf);
  const u = new Float32Array(n);
  const v = new Float32Array(n);
  const corpus = new Uint8Array(n);
  const id = new Uint32Array(n);
  const minz = withMinZoom ? new Uint8Array(n) : undefined;
  for (let i = 0; i < n; i++) {
    const o = i * stride;
    u[i] = dv.getUint16(o, true) / Q;
    v[i] = dv.getUint16(o + 2, true) / Q;
    const p = dv.getUint32(o + 4, true);
    corpus[i] = p >>> 28;
    id[i] = p & 0x0fffffff;
    if (minz) minz[i] = dv.getUint8(o + 8);
  }
  return { n, u, v, corpus, id, minz };
}

/**
 * A spatial index over `lod.bin` so a draw only submits the vertices that can
 * be on screen.
 *
 * The file is ordered "min_zoom, then finest tile id" (map_pack.build_lod), so
 * within one min-zoom band the records for a row-major run of tiles are
 * contiguous. `tileStart[band]` holds absolute record offsets for every finest
 * tile plus a terminator, which turns "draw the visible tiles" into a handful
 * of `drawArrays` ranges instead of one 1.8M-vertex draw.
 *
 * The order is verified rather than trusted: if any band is not sorted by tile
 * id, the index is refused and the caller falls back to the prefix draw.
 */
export interface LodIndex {
  z: number;
  /** per band: record offsets, length (1<<z)^2 + 1 */
  tileStart: Int32Array[];
}

export function buildLodIndex(
  b: PointBatch,
  z: number,
  zoomOffsets: number[] | undefined,
): LodIndex | null {
  if (!b.n) return null;
  const n = 1 << z;
  const nTiles = n * n;
  // band boundaries in records; without min_zoom_offsets treat it as one band
  const bounds: [number, number][] = [];
  if (zoomOffsets && zoomOffsets.length >= 2) {
    for (let i = 0; i + 1 < zoomOffsets.length; i++) {
      const lo = Math.min(zoomOffsets[i], b.n);
      const hi = Math.min(zoomOffsets[i + 1], b.n);
      if (hi > lo) bounds.push([lo, hi]);
      else bounds.push([lo, lo]);
    }
  } else {
    bounds.push([0, b.n]);
  }

  const tileStart: Int32Array[] = [];
  for (const [lo, hi] of bounds) {
    const starts = new Int32Array(nTiles + 1);
    let prev = -1;
    for (let i = lo; i < hi; i++) {
      const tx = Math.min(n - 1, Math.max(0, Math.floor(b.u[i] * n)));
      const ty = Math.min(n - 1, Math.max(0, Math.floor(b.v[i] * n)));
      const t = ty * n + tx;
      if (t < prev) return null; // not sorted by tile id — refuse the index
      if (t !== prev) {
        for (let k = prev + 1; k <= t; k++) starts[k] = i;
        prev = t;
      }
    }
    for (let k = prev + 1; k <= nTiles; k++) starts[k] = hi;
    tileStart.push(starts);
  }
  return { z, tileStart };
}

export class PointStore {
  private manifest: Manifest;
  private reader: RangeReader;

  lod: PointBatch | null = null;
  lodLoading = false;

  deepIndex: BigUint64Array | null = null;
  deepIndexLoading = false;
  private deepTiles = new Map<string, PointBatch | "loading">();
  private deepInflight = 0;

  onChange: () => void = () => {};
  /** a deep tile left the cache — the renderer should free its point buffer */
  onDropTile: (key: string) => void = () => {};

  constructor(manifest: Manifest, reader: RangeReader) {
    this.manifest = manifest;
    this.reader = reader;
  }

  get lodBytes(): number {
    const l = this.manifest.points?.lod;
    return l?.bytes ?? (l?.count ?? 0) * (l?.record_bytes ?? 9);
  }

  get hasLod(): boolean {
    return !!this.manifest.points?.lod?.path;
  }

  get hasDeep(): boolean {
    const d = this.manifest.points?.deep;
    return !!(d?.path && d.tile_index?.path && this.reader.supports(d.path));
  }

  get deepIndexBytes(): number {
    const ti = this.manifest.points?.deep?.tile_index;
    if (!ti) return 0;
    const known = this.reader.fileBytes(ti.path);
    if (known) return known;
    const n = ti.entries ?? (1 << ti.z) * (1 << ti.z) + 1;
    return n * 8;
  }

  get deepZ(): number {
    return this.manifest.points?.deep?.tile_index?.z ?? this.manifest.zoom.max;
  }

  async loadLod(onProgress?: ProgressFn): Promise<void> {
    const l = this.manifest.points?.lod;
    if (!l || this.lod || this.lodLoading) return;
    this.lodLoading = true;
    try {
      const buf = await this.reader.readAll(l.path, this.lodBytes, onProgress);
      this.lod = decode(buf, l.record_bytes ?? 9, true);
    } finally {
      this.lodLoading = false;
      this.onChange();
    }
  }

  async loadDeepIndex(onProgress?: ProgressFn): Promise<void> {
    const ti = this.manifest.points?.deep?.tile_index;
    if (!ti || this.deepIndex || this.deepIndexLoading) return;
    this.deepIndexLoading = true;
    try {
      const buf = await this.reader.readAll(
        ti.path,
        this.deepIndexBytes,
        onProgress,
      );
      this.deepIndex = new BigUint64Array(buf);
    } finally {
      this.deepIndexLoading = false;
      this.onChange();
    }
  }

  /**
   * Byte range [start,end) for a finest-level tile. The index stores either
   * byte offsets (real packs) or record offsets (fixtures) — `unit` says which.
   */
  private tileRange(tx: number, ty: number): [number, number] | null {
    const idx = this.deepIndex;
    const d = this.manifest.points?.deep;
    if (!idx || !d) return null;
    const stride = d.record_bytes ?? 8;
    const scale = d.tile_index?.unit === "bytes" ? 1 : stride;
    const n = 1 << this.deepZ;
    if (tx < 0 || ty < 0 || tx >= n || ty >= n) return null;
    const t = ty * n + tx;
    if (t >= idx.length) return null;
    const start = Number(idx[t]) * scale;
    const end =
      t + 1 < idx.length
        ? Number(idx[t + 1]) * scale
        : (d.count ?? 0) * stride;
    return [start, end];
  }

  /** Bytes a deep fetch of these tiles would cost right now. */
  deepCost(tiles: [number, number][]): number {
    const d = this.manifest.points?.deep;
    if (!d) return 0;
    let cost = 0;
    for (const [tx, ty] of tiles) {
      if (this.deepTiles.has(`${tx}_${ty}`)) continue;
      const r = this.tileRange(tx, ty);
      if (!r || r[1] <= r[0]) continue;
      cost += this.reader.costOf(d.path, r[0], r[1] - r[0]);
    }
    return cost;
  }

  deepTile(tx: number, ty: number): PointBatch | null {
    const v = this.deepTiles.get(`${tx}_${ty}`);
    return v && v !== "loading" ? v : null;
  }

  /** Kick off fetches for any visible finest-level tiles we don't have yet. */
  ensureDeep(tiles: [number, number][], maxConcurrent = 4): void {
    const d = this.manifest.points?.deep;
    if (!d || !this.deepIndex) return;
    const stride = d.record_bytes ?? 8;
    for (const [tx, ty] of tiles) {
      if (this.deepInflight >= maxConcurrent) return;
      const key = `${tx}_${ty}`;
      if (this.deepTiles.has(key)) continue;
      const r = this.tileRange(tx, ty);
      if (!r) continue;
      if (r[1] <= r[0]) {
        this.deepTiles.set(key, {
          n: 0,
          u: new Float32Array(0),
          v: new Float32Array(0),
          corpus: new Uint8Array(0),
          id: new Uint32Array(0),
        });
        continue;
      }
      this.deepTiles.set(key, "loading");
      this.deepInflight++;
      void this.reader
        .read(d.path, r[0], r[1] - r[0])
        .then((buf) => this.deepTiles.set(key, decode(buf, stride, false)))
        .catch(() => this.deepTiles.delete(key))
        .finally(() => {
          this.deepInflight--;
          this.onChange();
        });
    }
    // bound memory: drop the least-recently-inserted tiles
    while (this.deepTiles.size > 64) {
      const k = this.deepTiles.keys().next().value as string | undefined;
      if (k === undefined) break;
      this.deepTiles.delete(k);
      this.onDropTile(k);
    }
  }
}
