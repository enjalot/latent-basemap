/**
 * net.ts — every byte the viewer pulls goes through here.
 *
 * Two jobs:
 *   1. Session byte accounting (the status line requirement).
 *   2. Byte-range reads in one of two modes, decided per data source:
 *        "http-range"  — a real `Range:` request (gh-pages, GCS, nginx, S3)
 *        "chunked"     — the file was ALSO published as fixed-size
 *                        `<path>.part{N}` files; we fetch the covering parts
 *                        and splice. This is the fallback for static hosts
 *                        that ignore Range (python http.server on
 *                        gsv.local:8800 does exactly that: 200 + full body).
 */

import type { Manifest, RangeMode } from "./types";

// ---------------------------------------------------------------------------
// session byte accounting
// ---------------------------------------------------------------------------

type BytesListener = (bytes: number, requests: number) => void;

let totalBytes = 0;
let totalRequests = 0;
const listeners = new Set<BytesListener>();

export function onBytes(fn: BytesListener): () => void {
  listeners.add(fn);
  fn(totalBytes, totalRequests);
  return () => listeners.delete(fn);
}

function account(n: number) {
  totalBytes += n;
  totalRequests += 1;
  for (const fn of listeners) fn(totalBytes, totalRequests);
}

export function sessionBytes() {
  return { bytes: totalBytes, requests: totalRequests };
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

// ---------------------------------------------------------------------------
// plain fetches
// ---------------------------------------------------------------------------

export async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${url}`);
  const buf = await res.arrayBuffer();
  account(buf.byteLength);
  return JSON.parse(new TextDecoder().decode(buf)) as T;
}

export async function fetchBuffer(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${url}`);
  const buf = await res.arrayBuffer();
  account(buf.byteLength);
  return buf;
}

/** Returns null on 404 instead of throwing (sparse packs skip empty tiles). */
export async function fetchBufferOptional(url: string): Promise<ArrayBuffer | null> {
  const res = await fetch(url);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${url}`);
  const buf = await res.arrayBuffer();
  account(buf.byteLength);
  return buf;
}

export type ProgressFn = (loaded: number, total: number) => void;

/** Streamed fetch with real progress from content-length. */
export async function fetchBufferProgress(
  url: string,
  onProgress?: ProgressFn,
): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${url}`);
  const total = Number(res.headers.get("content-length") || 0);
  if (!res.body || !onProgress) {
    const buf = await res.arrayBuffer();
    account(buf.byteLength);
    onProgress?.(buf.byteLength, buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.byteLength;
  }
  account(loaded);
  return out.buffer;
}

// ---------------------------------------------------------------------------
// range-mode probe
// ---------------------------------------------------------------------------

/**
 * Ask the server for the first 8 bytes of `probeUrl`. A 206 with a short body
 * means real Range support. Anything else (200 + full body is what python
 * http.server does) means we must use the chunked fallback.
 */
export async function probeRangeMode(probeUrl: string): Promise<RangeMode> {
  // A server that ignores Range answers 200 with the WHOLE file — which for a
  // deep-points blob is hundreds of MB. Abort the moment the headers say it
  // isn't a 206 so the probe never costs more than a few packets.
  const ctrl = new AbortController();
  try {
    const res = await fetch(probeUrl, {
      headers: { Range: "bytes=0-7" },
      signal: ctrl.signal,
    });
    if (res.status !== 206) {
      // Ignored Range => this is the whole file. Probe targets are chosen to be
      // tiny, so drain it cleanly; only bail out hard if it really is huge.
      const len = Number(res.headers.get("content-length") || 0);
      if (len && len > 4 * 1024 * 1024) {
        ctrl.abort();
      } else {
        const buf = await res.arrayBuffer().catch(() => new ArrayBuffer(0));
        account(buf.byteLength);
      }
      return "chunked";
    }
    const buf = await res.arrayBuffer();
    account(buf.byteLength);
    if (buf.byteLength <= 8) return "http-range";
  } catch {
    /* fall through */
  }
  return "chunked";
}

// ---------------------------------------------------------------------------
// RangeReader
// ---------------------------------------------------------------------------

const PART_CACHE_MAX = 12;

/**
 * Reads byte ranges out of pack files in whichever mode the source supports.
 * `base` must end with "/".
 */
export class RangeReader {
  readonly base: string;
  readonly mode: RangeMode;
  private partBytes: number;
  private chunkFiles: Record<string, { bytes: number; parts: number }>;
  private partCache = new Map<string, ArrayBuffer>();
  private inflight = new Map<string, Promise<ArrayBuffer>>();

  constructor(base: string, mode: RangeMode, manifest: Manifest) {
    this.base = base;
    this.partBytes = manifest.chunking?.part_bytes ?? 4 * 1024 * 1024;
    this.chunkFiles = manifest.chunking?.files ?? {};
    // A source with no Range support and no published parts can't do ranges.
    this.mode =
      mode === "http-range"
        ? "http-range"
        : Object.keys(this.chunkFiles).length > 0
          ? "chunked"
          : "unavailable";
  }

  /** Is `path` reachable by ranged reads in the active mode? */
  supports(path: string): boolean {
    if (this.mode === "http-range") return true;
    if (this.mode === "chunked") return path in this.chunkFiles;
    return false;
  }

  fileBytes(path: string): number | undefined {
    return this.chunkFiles[path]?.bytes;
  }

  /** Read [start, start+length) of `path`. */
  async read(path: string, start: number, length: number): Promise<ArrayBuffer> {
    if (length <= 0) return new ArrayBuffer(0);
    if (this.mode === "http-range") {
      const end = start + length - 1;
      const res = await fetch(this.base + path, {
        headers: { Range: `bytes=${start}-${end}` },
      });
      if (!res.ok) throw new Error(`${res.status} range read — ${path}`);
      const buf = await res.arrayBuffer();
      account(buf.byteLength);
      // A server that answered 200 handed us the whole file; slice it.
      if (res.status === 200 && buf.byteLength > length) {
        return buf.slice(start, start + length);
      }
      return buf;
    }
    if (this.mode !== "chunked") {
      throw new Error(`no ranged-read mode available for ${path}`);
    }
    const first = Math.floor(start / this.partBytes);
    const last = Math.floor((start + length - 1) / this.partBytes);
    const out = new Uint8Array(length);
    for (let p = first; p <= last; p++) {
      const part = new Uint8Array(await this.part(path, p));
      const partStart = p * this.partBytes;
      const from = Math.max(0, start - partStart);
      const to = Math.min(part.byteLength, start + length - partStart);
      if (to <= from) continue;
      out.set(part.subarray(from, to), partStart + from - start);
    }
    return out.buffer;
  }

  /** How many bytes this ranged read will actually cost on the wire. */
  costOf(path: string, start: number, length: number): number {
    if (this.mode === "http-range") return length;
    if (this.mode !== "chunked") return 0;
    const first = Math.floor(start / this.partBytes);
    const last = Math.floor((start + length - 1) / this.partBytes);
    let cost = 0;
    for (let p = first; p <= last; p++) {
      if (!this.partCache.has(`${path}#${p}`)) cost += this.partBytes;
    }
    return cost;
  }

  private part(path: string, index: number): Promise<ArrayBuffer> {
    const key = `${path}#${index}`;
    const hit = this.partCache.get(key);
    if (hit) {
      this.partCache.delete(key);
      this.partCache.set(key, hit); // LRU bump
      return Promise.resolve(hit);
    }
    const running = this.inflight.get(key);
    if (running) return running;
    const p = fetchBuffer(`${this.base}${path}.part${index}`).then((buf) => {
      this.partCache.set(key, buf);
      while (this.partCache.size > PART_CACHE_MAX) {
        const oldest = this.partCache.keys().next().value as string | undefined;
        if (oldest === undefined) break;
        this.partCache.delete(oldest);
      }
      this.inflight.delete(key);
      return buf;
    });
    this.inflight.set(key, p);
    return p;
  }

  /**
   * Read a whole file with progress. In chunked mode this walks the parts so
   * the progress bar still moves on hosts without Range.
   */
  async readAll(
    path: string,
    totalBytesHint: number,
    onProgress?: ProgressFn,
  ): Promise<ArrayBuffer> {
    if (this.mode !== "chunked") {
      return fetchBufferProgress(this.base + path, onProgress);
    }
    const info = this.chunkFiles[path];
    const total = info?.bytes ?? totalBytesHint;
    const parts = info?.parts ?? Math.ceil(total / this.partBytes);
    const out = new Uint8Array(total);
    let loaded = 0;
    for (let p = 0; p < parts; p++) {
      const buf = await fetchBuffer(`${this.base}${path}.part${p}`);
      out.set(new Uint8Array(buf), p * this.partBytes);
      loaded += buf.byteLength;
      onProgress?.(loaded, total);
    }
    return out.buffer;
  }
}
