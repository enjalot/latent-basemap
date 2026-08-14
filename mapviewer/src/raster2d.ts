/**
 * raster2d.ts — canvas-2D fallback for the base layer.
 *
 * Used only when WebGL2 is unavailable. It honours the same "upload once"
 * contract as the GL renderer (each composed tile becomes a canvas exactly once
 * per recompose) and implements the same `draw(cam, scene)` interface, so the
 * layering — base redraws only on camera/data change, overlay on hover — is
 * identical. Point drawing is capped, because this path is CPU-bound.
 */

import { stats } from "./render";
import type { Camera, PointData, PointRef, Scene, TileRef } from "./gl";

const MAX_FALLBACK_POINTS = 120_000;

interface Tile2D {
  canvas: HTMLCanvasElement;
  version: string;
  size: number;
}

export class Raster2DRenderer {
  readonly renderer = "canvas-2d fallback (no WebGL2)";
  private ctx: CanvasRenderingContext2D;
  private tiles = new Map<string, Tile2D>();
  private points = new Map<string, PointData>();
  private colors: Record<number, string> = {};
  private dpr = 1;

  constructor(readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("2d context unavailable");
    this.ctx = ctx;
  }

  setCorpusColors(colors: Record<number, [number, number, number]>) {
    for (const [code, rgb] of Object.entries(colors))
      this.colors[Number(code)] = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }

  resize(width: number, height: number, dpr: number) {
    this.dpr = dpr;
    this.canvas.width = Math.max(1, Math.round(width * dpr));
    this.canvas.height = Math.max(1, Math.round(height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  tileVersion(key: string): string | undefined {
    return this.tiles.get(key)?.version;
  }

  uploadTile(key: string, version: string, rgba: Uint8ClampedArray, size: number) {
    let t = this.tiles.get(key);
    if (!t) {
      const c = document.createElement("canvas");
      c.width = size;
      c.height = size;
      t = { canvas: c, version: "", size };
      this.tiles.set(key, t);
    }
    const cx = t.canvas.getContext("2d");
    if (!cx) return;
    // copy through a fresh ArrayBuffer-backed view: ImageData rejects
    // SharedArrayBuffer-backed arrays in the DOM typings
    const img = cx.createImageData(size, size);
    img.data.set(rgba);
    cx.putImageData(img, 0, 0);
    t.version = version;
    stats.tileUploads++;
  }

  dropTile(key: string) {
    this.tiles.delete(key);
  }

  hasPoints(name: string): boolean {
    return this.points.has(name);
  }

  uploadPoints(name: string, b: PointData) {
    this.points.set(name, b);
    stats.pointUploads++;
  }

  dropPoints(name: string) {
    this.points.delete(name);
  }

  retainPoints(keep: Set<string>) {
    for (const k of [...this.points.keys()]) if (!keep.has(k)) this.points.delete(k);
  }

  draw(cam: Camera, scene: Scene) {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = `rgb(${scene.bg[0]},${scene.bg[1]},${scene.bg[2]})`;
    ctx.fillRect(0, 0, cam.width, cam.height);
    ctx.imageSmoothingEnabled = false;
    for (const t of scene.tiles) this.drawTile(cam, t);
    let budget = MAX_FALLBACK_POINTS;
    for (const p of scene.points) budget = this.drawPoints(cam, p, budget);
  }

  private drawTile(cam: Camera, t: TileRef) {
    const e = this.tiles.get(t.key);
    if (!e) return;
    const sx = (t.u0 - cam.cx) * cam.scale + cam.width / 2;
    const sy = (t.v0 - cam.cy) * cam.scale + cam.height / 2;
    this.ctx.drawImage(
      e.canvas,
      t.su * e.size,
      t.sv * e.size,
      t.sw * e.size,
      t.sh * e.size,
      sx,
      sy,
      t.du * cam.scale + 0.5,
      t.dv * cam.scale + 0.5,
    );
  }

  private drawPoints(cam: Camera, p: PointRef, budget: number): number {
    const b = this.points.get(p.name);
    if (!b || !b.n || budget <= 0) return budget;
    const ctx = this.ctx;
    const size = Math.max(1, p.size / this.dpr);
    ctx.globalAlpha = p.alpha;
    let last = -1;
    for (const [first, count] of p.ranges) {
      const lo = Math.max(0, Math.min(first, b.n));
      const hi = Math.min(b.n, lo + count);
      for (let i = lo; i < hi && budget > 0; i++) {
        const c = b.corpus[i];
        if (((p.mask >> c) & 1) === 0) continue;
        if (b.minz && b.minz[i] > p.maxMinZ) continue;
        const x = (b.u[i] - cam.cx) * cam.scale + cam.width / 2;
        const y = (b.v[i] - cam.cy) * cam.scale + cam.height / 2;
        if (x < -4 || y < -4 || x > cam.width + 4 || y > cam.height + 4) continue;
        if (c !== last) {
          ctx.fillStyle = this.colors[c] ?? "#888";
          last = c;
        }
        ctx.fillRect(x - size / 2, y - size / 2, size, size);
        budget--;
      }
      if (budget <= 0) break;
    }
    ctx.globalAlpha = 1;
    return budget;
  }

  sampleDistinctColours(step = 4): number {
    const c = this.canvas;
    const d = this.ctx.getImageData(0, 0, c.width, c.height).data;
    const seen = new Set<string>();
    for (let i = 0; i < d.length; i += 4 * step)
      seen.add(`${d[i]},${d[i + 1]},${d[i + 2]}`);
    return seen.size;
  }

  dispose() {
    this.tiles.clear();
    this.points.clear();
  }
}
