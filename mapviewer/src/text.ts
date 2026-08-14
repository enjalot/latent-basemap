/**
 * text.ts — click-through to a row's full chunk text.
 *
 * Two ranged reads per lookup, both tiny:
 *   1. 16 B at offsets.u64[row]  -> [start, end) byte range in the blob
 *   2. [start, end) of blob.utf8 -> the text
 * In chunked mode each read costs one part instead (parts are cached), which is
 * why the panel shows what a lookup actually cost.
 */

import type { Manifest } from "./types";
import { RangeReader } from "./net";

export interface TextLookup {
  row: number;
  text: string;
  bytes: number;
}

export class TextSidecar {
  private manifest: Manifest;
  private reader: RangeReader;
  private cache = new Map<number, string>();

  constructor(manifest: Manifest, reader: RangeReader) {
    this.manifest = manifest;
    this.reader = reader;
  }

  get available(): boolean {
    const t = this.manifest.text;
    return !!(t?.offsets && t?.blob && this.reader.supports(t.blob));
  }

  async lookup(row: number): Promise<TextLookup> {
    const t = this.manifest.text;
    if (!t) throw new Error("pack has no text sidecar");
    const cached = this.cache.get(row);
    if (cached !== undefined) return { row, text: cached, bytes: 0 };

    const before = this.reader.costOf(t.offsets, row * 8, 16);
    const offBuf = await this.reader.read(t.offsets, row * 8, 16);
    if (offBuf.byteLength < 16) throw new Error(`row ${row} out of range`);
    const dv = new DataView(offBuf);
    const start = Number(dv.getBigUint64(0, true));
    const end = Number(dv.getBigUint64(8, true));
    const len = end - start;
    if (len < 0 || len > 8 << 20) throw new Error(`bad text extent for row ${row}`);

    const blobCost = this.reader.costOf(t.blob, start, len);
    const buf = await this.reader.read(t.blob, start, len);
    const text = new TextDecoder(t.encoding ?? "utf-8").decode(buf);
    this.cache.set(row, text);
    if (this.cache.size > 200) {
      const k = this.cache.keys().next().value as number | undefined;
      if (k !== undefined) this.cache.delete(k);
    }
    return { row, text, bytes: before + blobCost };
  }
}
