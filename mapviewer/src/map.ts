/**
 * map.ts — camera, canvas renderer and pointer interaction.
 *
 * World space is the unit square: u = (x - xmin)/(xmax - xmin),
 * v = (ymax - y)/(ymax - ymin)  (v is Y-DOWN, matching tile and bin order).
 * The camera is (cx, cy, scale) with scale in device-independent px per world
 * unit; the zoom level picked for tiles is round(log2(scale / tile_size)).
 */

import type { Manifest } from "./types";
import { DensityStore } from "./density";
import { PointStore, type PointBatch } from "./points";
import { CORPUS_FALLBACK } from "./palette";

export interface HoverInfo {
  /** preview zoom level the bin was resolved at */
  z: number;
  tx: number;
  ty: number;
  bx: number;
  by: number;
  gbx: number;
  gby: number;
  counts: { code: number; count: number }[] | undefined;
  total: number;
  /** world coords of the bin's top-left / size, for the highlight */
  u0: number;
  v0: number;
  size: number;
}

export interface PickedPoint {
  id: number;
  corpus: number;
  u: number;
  v: number;
  tier: "lod" | "deep";
}

export type PointMode = "off" | "lod" | "deep";

export class MapView {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private manifest: Manifest;
  private density: DensityStore;
  private points: PointStore;
  private tileSize: number;
  private zmax: number;
  private colors: string[] = [];

  cx = 0.5;
  cy = 0.5;
  scale = 800;
  pointMode: PointMode = "off";
  enabled = new Set<number>();
  hovered: HoverInfo | null = null;
  pinned: HoverInfo | null = null;
  selected: PickedPoint | null = null;
  previewLevelCap = Infinity;

  onHover: (h: HoverInfo | null) => void = () => {};
  onPick: (p: PickedPoint | null) => void = () => {};
  onViewChange: () => void = () => {};

  private raf = 0;
  private dragging = false;
  private lastPointer: { x: number; y: number } | null = null;
  private moved = 0;

  constructor(
    canvas: HTMLCanvasElement,
    manifest: Manifest,
    density: DensityStore,
    points: PointStore,
  ) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("2d canvas context unavailable");
    this.ctx = ctx;
    this.manifest = manifest;
    this.density = density;
    this.points = points;
    this.tileSize = manifest.tile_size ?? 256;
    this.zmax = manifest.zoom.max;
    manifest.corpora.forEach((c, i) => {
      this.colors[c.code] = c.color ?? CORPUS_FALLBACK[i % CORPUS_FALLBACK.length];
      this.enabled.add(c.code);
    });
    this.attach();
  }

  // -- camera ---------------------------------------------------------------

  get width() {
    return this.canvas.clientWidth || 1;
  }
  get height() {
    return this.canvas.clientHeight || 1;
  }

  /**
   * Tile level for the current scale: floor, so a tile is never drawn much
   * below 1:1 — magnifying a coarse tile reads better than a near-empty fine
   * one, and it keeps the bin grid legible.
   */
  get z(): number {
    return Math.max(
      this.manifest.zoom.min,
      Math.min(this.zmax, Math.floor(Math.log2(this.scale / this.tileSize))),
    );
  }

  fit() {
    this.cx = 0.5;
    this.cy = 0.5;
    this.scale = Math.min(this.width, this.height) * 0.92;
    this.requestDraw();
    this.onViewChange();
  }

  zoomBy(factor: number, atX?: number, atY?: number) {
    const px = atX ?? this.width / 2;
    const py = atY ?? this.height / 2;
    const [u, v] = this.worldAt(px, py);
    this.scale = Math.max(64, Math.min(this.scale * factor, this.tileSize * (1 << this.zmax) * 24));
    this.cx = u - (px - this.width / 2) / this.scale;
    this.cy = v - (py - this.height / 2) / this.scale;
    this.clampCamera();
    this.requestDraw();
    this.onViewChange();
  }

  /** Keep the unit square reachable: the camera may drift a quarter-map out. */
  private clampCamera() {
    const m = 0.25;
    this.cx = Math.max(-m, Math.min(this.cx, 1 + m));
    this.cy = Math.max(-m, Math.min(this.cy, 1 + m));
  }

  worldAt(px: number, py: number): [number, number] {
    return [
      (px - this.width / 2) / this.scale + this.cx,
      (py - this.height / 2) / this.scale + this.cy,
    ];
  }

  screenAt(u: number, v: number): [number, number] {
    return [
      (u - this.cx) * this.scale + this.width / 2,
      (v - this.cy) * this.scale + this.height / 2,
    ];
  }

  /** Visible tile range at a level, clamped to the grid. */
  visibleTiles(z: number): { x0: number; x1: number; y0: number; y1: number } {
    const n = 1 << z;
    const [u0, v0] = this.worldAt(0, 0);
    const [u1, v1] = this.worldAt(this.width, this.height);
    return {
      x0: Math.max(0, Math.floor(u0 * n)),
      x1: Math.min(n - 1, Math.floor(u1 * n)),
      y0: Math.max(0, Math.floor(v0 * n)),
      y1: Math.min(n - 1, Math.floor(v1 * n)),
    };
  }

  visibleTileList(z: number): [number, number][] {
    const r = this.visibleTiles(z);
    const out: [number, number][] = [];
    for (let y = r.y0; y <= r.y1; y++)
      for (let x = r.x0; x <= r.x1; x++) out.push([x, y]);
    return out;
  }

  // -- drawing --------------------------------------------------------------

  requestDraw() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.draw();
    });
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.requestDraw();
  }

  private bgColor(): string {
    return getComputedStyle(this.canvas).getPropertyValue("--map-bg").trim() || "#0f1113";
  }

  draw() {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = this.bgColor();
    ctx.fillRect(0, 0, this.width, this.height);

    const z = this.z;
    ctx.imageSmoothingEnabled = false;
    const r = this.visibleTiles(z);
    const n = 1 << z;
    const tw = this.scale / n;
    for (let ty = r.y0; ty <= r.y1; ty++) {
      for (let tx = r.x0; tx <= r.x1; tx++) {
        const [sx, sy] = this.screenAt(tx / n, ty / n);
        const bmp = this.density.getBitmap(z, tx, ty);
        if (bmp) {
          ctx.drawImage(bmp, sx, sy, tw + 0.5, tw + 0.5);
        } else {
          this.drawAncestor(z, tx, ty, sx, sy, tw);
        }
      }
    }

    if (this.pointMode !== "off") this.drawPoints(z);
    this.drawHighlight();
    this.drawSelected();
    ctx.restore();
  }

  /** While a tile loads, upscale the matching sub-rect of a coarser tile. */
  private drawAncestor(
    z: number,
    tx: number,
    ty: number,
    sx: number,
    sy: number,
    tw: number,
  ) {
    for (let pz = z - 1; pz >= this.manifest.zoom.min; pz--) {
      const d = z - pz;
      const px = tx >> d;
      const py = ty >> d;
      const bmp = this.density.getBitmap(pz, px, py);
      if (!bmp) continue;
      const k = 1 << d;
      const sub = this.tileSize / k;
      this.ctx.drawImage(
        bmp,
        (tx - px * k) * sub,
        (ty - py * k) * sub,
        sub,
        sub,
        sx,
        sy,
        tw + 0.5,
        tw + 0.5,
      );
      return;
    }
  }

  private drawPoints(z: number) {
    const ctx = this.ctx;
    const size = Math.max(1, Math.min(3.5, this.scale / 900));
    ctx.globalAlpha = 0.9;
    const batches: PointBatch[] = [];
    let filterMinZoom = false;
    if (this.pointMode === "lod" && this.points.lod) {
      batches.push(this.points.lod);
      filterMinZoom = true;
    } else if (this.pointMode === "deep") {
      const dz = this.points.deepZ;
      for (const [tx, ty] of this.visibleTileList(dz)) {
        const b = this.points.deepTile(tx, ty);
        if (b && b.n) batches.push(b);
      }
      if (!batches.length && this.points.lod) {
        batches.push(this.points.lod);
        filterMinZoom = true;
      }
    }
    if (!batches.length) {
      ctx.globalAlpha = 1;
      return;
    }
    const [uL, vT] = this.worldAt(-8, -8);
    const [uR, vB] = this.worldAt(this.width + 8, this.height + 8);
    for (const code of this.enabled) {
      ctx.fillStyle = this.colors[code] ?? "#888";
      for (const b of batches) {
        for (let i = 0; i < b.n; i++) {
          if (b.corpus[i] !== code) continue;
          if (filterMinZoom && b.minz && b.minz[i] > z) continue;
          const u = b.u[i];
          const v = b.v[i];
          if (u < uL || u > uR || v < vT || v > vB) continue;
          ctx.fillRect(
            (u - this.cx) * this.scale + this.width / 2 - size / 2,
            (v - this.cy) * this.scale + this.height / 2 - size / 2,
            size,
            size,
          );
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  private drawHighlight() {
    const h = this.pinned ?? this.hovered;
    if (!h) return;
    const [sx, sy] = this.screenAt(h.u0, h.v0);
    const w = h.size * this.scale;
    const ctx = this.ctx;
    ctx.save();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = this.pinned ? "#eb6834" : "#ffffff";
    ctx.strokeRect(
      Math.round(sx) - 0.5,
      Math.round(sy) - 0.5,
      Math.max(3, w) + 1,
      Math.max(3, w) + 1,
    );
    if (w < 8) {
      ctx.beginPath();
      ctx.arc(sx + w / 2, sy + w / 2, 9, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawSelected() {
    const p = this.selected;
    if (!p) return;
    const [sx, sy] = this.screenAt(p.u, p.v);
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = "#eb6834";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // -- hit testing ----------------------------------------------------------

  /**
   * Bin under the cursor. Starts at the current level and walks *up* the
   * pyramid while the bin is empty (or its tile hasn't loaded), so a hover deep
   * in a sparse map still reports the coarse region it sits in rather than "0".
   */
  binAt(px: number, py: number, levelCap: number): HoverInfo | null {
    const start = Math.max(
      this.manifest.zoom.min,
      Math.min(this.z, levelCap, this.zmax),
    );
    let fallback: HoverInfo | null = null;
    for (let z = start; z >= this.manifest.zoom.min; z--) {
      const info = this.binAtLevel(px, py, z);
      if (!info) return null;
      if (info.counts && info.total > 0) return info;
      if (!fallback) fallback = info;
    }
    return fallback;
  }

  private binAtLevel(px: number, py: number, z: number): HoverInfo | null {
    const [u, v] = this.worldAt(px, py);
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return null;
    const n = 1 << z;
    const tx = Math.min(n - 1, Math.floor(u * n));
    const ty = Math.min(n - 1, Math.floor(v * n));
    const nb = n * this.tileSize;
    const gbx = Math.min(nb - 1, Math.floor(u * nb));
    const gby = Math.min(nb - 1, Math.floor(v * nb));
    const bx = gbx - tx * this.tileSize;
    const by = gby - ty * this.tileSize;
    const counts = this.density.binCounts(z, tx, ty, bx, by);
    const total = counts ? counts.reduce((a, c) => a + c.count, 0) : 0;
    return {
      z,
      tx,
      ty,
      bx,
      by,
      gbx,
      gby,
      counts,
      total,
      u0: gbx / nb,
      v0: gby / nb,
      size: 1 / nb,
    };
  }

  pickPoint(px: number, py: number, radius = 7): PickedPoint | null {
    if (this.pointMode === "off") return null;
    const z = this.z;
    let best: PickedPoint | null = null;
    let bestD = radius * radius;
    const consider = (b: PointBatch, tier: "lod" | "deep", useMinZoom: boolean) => {
      for (let i = 0; i < b.n; i++) {
        if (!this.enabled.has(b.corpus[i])) continue;
        if (useMinZoom && b.minz && b.minz[i] > z) continue;
        const [sx, sy] = this.screenAt(b.u[i], b.v[i]);
        const d = (sx - px) * (sx - px) + (sy - py) * (sy - py);
        if (d < bestD) {
          bestD = d;
          best = { id: b.id[i], corpus: b.corpus[i], u: b.u[i], v: b.v[i], tier };
        }
      }
    };
    if (this.pointMode === "deep") {
      for (const [tx, ty] of this.visibleTileList(this.points.deepZ)) {
        const b = this.points.deepTile(tx, ty);
        if (b && b.n) consider(b, "deep", false);
      }
    }
    if (!best && this.points.lod) consider(this.points.lod, "lod", true);
    return best;
  }

  // -- events ---------------------------------------------------------------

  private attach() {
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => {
      c.setPointerCapture(e.pointerId);
      this.dragging = true;
      this.moved = 0;
      this.lastPointer = { x: e.offsetX, y: e.offsetY };
    });
    c.addEventListener("pointermove", (e) => {
      if (this.dragging && this.lastPointer) {
        const dx = e.offsetX - this.lastPointer.x;
        const dy = e.offsetY - this.lastPointer.y;
        this.moved += Math.abs(dx) + Math.abs(dy);
        this.cx -= dx / this.scale;
        this.cy -= dy / this.scale;
        this.lastPointer = { x: e.offsetX, y: e.offsetY };
        this.clampCamera();
        this.requestDraw();
        this.onViewChange();
        return;
      }
      const h = this.binAt(e.offsetX, e.offsetY, this.previewLevelCap);
      const changed =
        (h === null) !== (this.hovered === null) ||
        (h &&
          this.hovered &&
          (h.gbx !== this.hovered.gbx ||
            h.gby !== this.hovered.gby ||
            h.z !== this.hovered.z));
      this.hovered = h;
      if (changed) {
        this.onHover(h);
        this.requestDraw();
      }
    });
    const end = (e: PointerEvent) => {
      if (!this.dragging) return;
      this.dragging = false;
      this.lastPointer = null;
      if (this.moved < 4) this.handleClick(e);
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", () => {
      this.dragging = false;
      this.lastPointer = null;
    });
    c.addEventListener("pointerleave", () => {
      if (this.hovered) {
        this.hovered = null;
        this.onHover(null);
        this.requestDraw();
      }
    });
    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const f = Math.pow(2, -e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0022));
        this.zoomBy(f, e.offsetX, e.offsetY);
      },
      { passive: false },
    );
    c.addEventListener("dblclick", (e) => this.zoomBy(2, e.offsetX, e.offsetY));
  }

  private handleClick(e: PointerEvent) {
    const p = this.pickPoint(e.offsetX, e.offsetY);
    if (p) {
      this.selected = p;
      this.onPick(p);
      this.requestDraw();
      return;
    }
    const h = this.binAt(e.offsetX, e.offsetY, this.previewLevelCap);
    this.pinned = h;
    this.selected = null;
    this.onPick(null);
    this.onHover(h);
    this.requestDraw();
  }
}
