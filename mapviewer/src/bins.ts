/**
 * bins.ts — per-bin sampled row ids and text snippets, precomputed per zoom
 * level. Keys are "{z}_{global_bin_x}_{global_bin_y}"; a tolerant resolver also
 * accepts tile-keyed nesting ("{z}_{tile_x}_{tile_y}" -> {"bx_by": [...]}) in
 * case the real pack builder keys them that way.
 */

import type { Manifest } from "./types";
import { fetchJSON } from "./net";

type BinJson = Record<string, unknown>;

export class BinSummaries {
  private base: string;
  private manifest: Manifest;
  private snippets = new Map<number, BinJson | null>();
  private samples = new Map<number, BinJson | null>();
  private loading = new Set<string>();
  onChange: () => void = () => {};

  constructor(base: string, manifest: Manifest) {
    this.base = base;
    this.manifest = manifest;
  }

  private levels(kind: "snippets" | "samples"): number[] {
    return this.manifest.bins?.[kind]?.levels ?? [];
  }

  /** The deepest available preview level at or below `z`. */
  previewLevel(z: number): number | null {
    const ls = this.levels("snippets");
    if (!ls.length) return null;
    const at = ls.filter((l) => l <= z);
    return at.length ? Math.max(...at) : Math.min(...ls);
  }

  private pattern(kind: "snippets" | "samples", z: number): string {
    const p = this.manifest.bins?.[kind]?.pattern ?? `bins/${kind}_z{z}.json`;
    return p.replace("{z}", String(z));
  }

  private ensure(kind: "snippets" | "samples", z: number) {
    const store = kind === "snippets" ? this.snippets : this.samples;
    if (store.has(z)) return;
    const tag = `${kind}:${z}`;
    if (this.loading.has(tag)) return;
    if (!this.levels(kind).includes(z)) {
      store.set(z, null);
      return;
    }
    this.loading.add(tag);
    void fetchJSON<BinJson>(this.base + this.pattern(kind, z))
      .then((j) => store.set(z, j))
      .catch(() => store.set(z, null))
      .finally(() => {
        this.loading.delete(tag);
        this.onChange();
      });
  }

  private resolve<T>(
    store: Map<number, BinJson | null>,
    z: number,
    gbx: number,
    gby: number,
    tileSize: number,
  ): T | undefined {
    const j = store.get(z);
    if (!j) return undefined;
    // fixture keys are "{z}_{bx}_{by}"; the real builder keys "{bx}_{by}"
    // (each file is already per-level, so the z prefix is redundant)
    const direct = j[`${z}_${gbx}_${gby}`] ?? j[`${gbx}_${gby}`];
    if (direct !== undefined) return direct as T;
    // tolerant: tile-keyed nesting
    const tx = Math.floor(gbx / tileSize);
    const ty = Math.floor(gby / tileSize);
    const nested = j[`${z}_${tx}_${ty}`];
    if (nested && typeof nested === "object" && !Array.isArray(nested)) {
      const bx = gbx - tx * tileSize;
      const by = gby - ty * tileSize;
      const rec = nested as Record<string, unknown>;
      const v = rec[`${bx}_${by}`] ?? rec[String(by * tileSize + bx)];
      if (v !== undefined) return v as T;
    }
    return undefined;
  }

  snippetsFor(z: number, gbx: number, gby: number): string[] | undefined {
    this.ensure("snippets", z);
    return this.resolve<string[]>(
      this.snippets,
      z,
      gbx,
      gby,
      this.manifest.tile_size ?? 256,
    );
  }

  samplesFor(z: number, gbx: number, gby: number): number[] | undefined {
    this.ensure("samples", z);
    return this.resolve<number[]>(
      this.samples,
      z,
      gbx,
      gby,
      this.manifest.tile_size ?? 256,
    );
  }

  ready(kind: "snippets" | "samples", z: number): boolean {
    const store = kind === "snippets" ? this.snippets : this.samples;
    return store.has(z);
  }

  /**
   * Preview for the bin under the cursor, walking *up* the pyramid until a
   * level has a summary for the covering bin. Packs sample snippets densely at
   * coarse levels and sparsely (top-count bins only) at fine ones, so the
   * coarse level is the answer for most of the map — this is the "hover shows
   * the coarse preview" behaviour from PLAN6.
   */
  previewAt(
    z: number,
    gbx: number,
    gby: number,
  ): { z: number; snippets: string[]; rows: number[] | undefined; pending: boolean } | null {
    const levels = (this.manifest.bins?.snippets?.levels ?? [])
      .filter((l) => l <= z)
      .sort((a, b) => b - a);
    if (!levels.length) return null;
    let pending = false;
    for (const lz of levels) {
      const shift = z - lz;
      const bx = gbx >> shift;
      const by = gby >> shift;
      const s = this.snippetsFor(lz, bx, by);
      if (!this.ready("snippets", lz)) pending = true;
      if (s && s.length) {
        return { z: lz, snippets: s, rows: this.samplesFor(lz, bx, by), pending: false };
      }
    }
    return { z: levels[0], snippets: [], rows: undefined, pending };
  }
}
