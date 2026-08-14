/**
 * map.ts — camera, layered renderer and pointer interaction.
 *
 * World space is the unit square: u = (x - xmin)/(xmax - xmin),
 * v = (ymax - y)/(ymax - ymin)  (v is Y-DOWN, matching tile and bin order).
 * The camera is (cx, cy, scale) with scale in device-independent px per world
 * unit; the zoom level picked for tiles is floor(log2(scale / tile_size)).
 *
 * v2 layering (the fix for the v1 flicker + point lag):
 *
 *   base layer     WebGL2 (`gl.ts`), or the canvas-2D fallback. Density tiles
 *                  are textures uploaded once per recompose; points are VBOs
 *                  uploaded once per data change. Redrawn ONLY on camera or
 *                  data change.
 *   overlay layer  a second canvas (`overlay.ts`) carrying the hover highlight,
 *                  the selection ring and the projection markers. This is the
 *                  only thing a mousemove touches.
 *
 * Both layers are driven by one rAF via `RenderLoop`, so pointer/wheel/tile
 * events coalesce into at most one draw per frame.
 */

import type { Manifest } from "./types";
import { DensityStore } from "./density";
import { PointStore, buildLodIndex, type LodIndex, type PointBatch } from "./points";
import { CORPUS_FALLBACK, hexToRgb } from "./palette";
import { GLRenderer, type Camera, type PointRef, type Scene, type TileRef } from "./gl";
import { Raster2DRenderer } from "./raster2d";
import { OverlayRenderer, type MarkerItem } from "./overlay";
import { RenderLoop } from "./render";

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

type BaseRenderer = GLRenderer | Raster2DRenderer;

const LOD_BUFFER = "lod";
const deepName = (tx: number, ty: number) => `deep:${tx}_${ty}`;

export class MapView {
  readonly canvas: HTMLCanvasElement;
  readonly overlayCanvas: HTMLCanvasElement;
  readonly base: BaseRenderer;
  readonly overlay: OverlayRenderer;
  readonly loop: RenderLoop;
  readonly backend: string;

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
  markers: MarkerItem[] = [];
  previewLevelCap = Infinity;

  onHover: (h: HoverInfo | null) => void = () => {};
  onPick: (p: PickedPoint | null) => void = () => {};
  onViewChange: () => void = () => {};

  private dragging = false;
  private lastPointer: { x: number; y: number } | null = null;
  private moved = 0;
  private detach: (() => void)[] = [];
  /** spatial index over the LOD buffer; null when the file isn't tile-sorted */
  private lodIndex: LodIndex | null = null;

  constructor(
    canvas: HTMLCanvasElement,
    overlayCanvas: HTMLCanvasElement,
    manifest: Manifest,
    density: DensityStore,
    points: PointStore,
  ) {
    this.canvas = canvas;
    this.overlayCanvas = overlayCanvas;
    this.manifest = manifest;
    this.density = density;
    this.points = points;
    this.tileSize = manifest.tile_size ?? 256;
    this.zmax = manifest.zoom.max;

    let base: BaseRenderer;
    try {
      base = new GLRenderer(canvas);
    } catch {
      base = new Raster2DRenderer(canvas);
    }
    this.base = base;
    this.backend = base instanceof GLRenderer ? `webgl2 — ${base.renderer}` : base.renderer;
    this.overlay = new OverlayRenderer(overlayCanvas);
    this.loop = new RenderLoop(
      () => this.drawBase(),
      () => this.drawOverlay(),
    );

    const rgb: Record<number, [number, number, number]> = {};
    manifest.corpora.forEach((c, i) => {
      const hex = c.color ?? CORPUS_FALLBACK[i % CORPUS_FALLBACK.length];
      this.colors[c.code] = hex;
      rgb[c.code] = hexToRgb(hex);
      this.enabled.add(c.code);
    });
    this.base.setCorpusColors(rgb);

    // a tile leaving the LRU must free its GPU texture
    this.density.onEvict = (key) => this.base.dropTile(key);
    this.points.onDropTile = (key) => this.base.dropPoints(`deep:${key}`);

    this.attach();
  }

  // -- camera ---------------------------------------------------------------

  get width() {
    return this.overlayCanvas.clientWidth || 1;
  }
  get height() {
    return this.overlayCanvas.clientHeight || 1;
  }

  get camera(): Camera {
    return { cx: this.cx, cy: this.cy, scale: this.scale, width: this.width, height: this.height };
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
    this.loop.markView();
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
    this.loop.markView();
    this.onViewChange();
  }

  /** Centre the camera on a world point, optionally zooming in. */
  flyTo(u: number, v: number, scale?: number) {
    this.cx = u;
    this.cy = v;
    if (scale) this.scale = Math.max(64, scale);
    this.clampCamera();
    this.loop.markView();
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

  /** Data changed (tile composed, points loaded): base only. */
  requestDraw() {
    this.loop.markView();
  }

  /** Overlay-only invalidation — hover, selection, markers. */
  requestOverlay() {
    this.loop.markOverlay();
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.base.resize(this.width, this.height, dpr);
    this.overlay.resize(this.width, this.height, dpr);
    this.loop.markView();
  }

  private cssVar(name: string, fallback: string): string {
    const v = getComputedStyle(this.overlayCanvas).getPropertyValue(name).trim();
    return v || fallback;
  }

  private bgColor(): [number, number, number] {
    return hexToRgb(this.cssVar("--map-bg", "#0f1113"));
  }

  private drawBase() {
    this.base.draw(this.camera, this.buildScene());
  }

  private drawOverlay() {
    const h = this.pinned ?? this.hovered;
    this.overlay.draw(this.camera, {
      highlight: h
        ? { u0: h.u0, v0: h.v0, size: h.size, pinned: this.pinned !== null }
        : null,
      selected: this.selected ? { u: this.selected.u, v: this.selected.v } : null,
      markers: this.markers,
      accent: this.cssVar("--accent", "#eb6834"),
      marker: this.cssVar("--marker", "#c2185b"),
      ink: this.cssVar("--overlay-ink", "#ffffff"),
      paper: this.cssVar("--bg", "#ffffff"),
    });
  }

  /**
   * The per-frame scene description. Cheap: it only walks the visible tile
   * grid and pushes rects — no pixel work, no allocation per point. Texture and
   * buffer uploads happen here at most once per data change.
   */
  private buildScene(): Scene {
    const z = this.z;
    const n = 1 << z;
    const r = this.visibleTiles(z);
    const tiles: TileRef[] = [];
    for (let ty = r.y0; ty <= r.y1; ty++) {
      for (let tx = r.x0; tx <= r.x1; tx++) {
        const ref = this.tileRef(z, tx, ty, n);
        if (ref) tiles.push(ref);
      }
    }
    return {
      bg: this.bgColor(),
      tiles,
      points: this.pointRefs(z),
    };
  }

  private tileRef(z: number, tx: number, ty: number, n: number): TileRef | null {
    const dest = { u0: tx / n, v0: ty / n, du: 1 / n, dv: 1 / n };
    const t = this.density.getRender(z, tx, ty);
    if (t && t.rgba && t.rgbaKey) {
      if (this.base.tileVersion(t.key) !== t.rgbaKey)
        this.base.uploadTile(t.key, t.rgbaKey, t.rgba, this.tileSize);
      return { key: t.key, ...dest, su: 0, sv: 0, sw: 1, sh: 1 };
    }
    // while a tile loads, magnify the matching sub-rect of a coarser ancestor
    for (let pz = z - 1; pz >= this.manifest.zoom.min; pz--) {
      const d = z - pz;
      const px = tx >> d;
      const py = ty >> d;
      const a = this.density.getRender(pz, px, py);
      if (!a || !a.rgba || !a.rgbaKey) continue;
      if (this.base.tileVersion(a.key) !== a.rgbaKey)
        this.base.uploadTile(a.key, a.rgbaKey, a.rgba, this.tileSize);
      const k = 1 << d;
      return {
        key: a.key,
        ...dest,
        su: (tx - px * k) / k,
        sv: (ty - py * k) / k,
        sw: 1 / k,
        sh: 1 / k,
      };
    }
    return null;
  }

  private get corpusMask(): number {
    let m = 0;
    for (const c of this.enabled) if (c >= 0 && c < 16) m |= 1 << c;
    return m;
  }

  /**
   * How many LOD records to draw at this zoom. `lod.bin` is sorted by min_zoom,
   * so the visible set is a prefix — no per-vertex filtering needed, and z0
   * costs 25k points instead of 485k.
   */
  private lodCount(z: number): number {
    const b = this.points.lod;
    if (!b) return 0;
    const offs = this.manifest.points?.lod?.zoom_offsets;
    if (!offs || offs.length < 2) return b.n;
    const i = Math.min(Math.max(z, 0) + 1, offs.length - 1);
    return Math.min(b.n, offs[i] ?? b.n);
  }

  /**
   * Draw ranges for the LOD buffer: the min-zoom bands up to `z`, each cut down
   * to the visible tile rows via the spatial index. Without the index this is
   * the min-zoom prefix, which is still correct, just heavier.
   */
  private lodRanges(z: number): [number, number][] {
    const b = this.points.lod;
    if (!b || !b.n) return [];
    const idx = this.lodIndex;
    const offs = this.manifest.points?.lod?.zoom_offsets;
    if (!idx || !offs) return [[0, this.lodCount(z)]];

    const n = 1 << idx.z;
    const r = this.visibleTiles(idx.z);
    // one tile of slack: point sprites overhang, and the tile a record is
    // bucketed into is re-derived from its u16 coordinate
    const x0 = Math.max(0, r.x0 - 1);
    const x1 = Math.min(n - 1, r.x1 + 1);
    const y0 = Math.max(0, r.y0 - 1);
    const y1 = Math.min(n - 1, r.y1 + 1);

    const bands = Math.min(idx.tileStart.length - 1, Math.max(0, z));
    const out: [number, number][] = [];
    for (let band = 0; band <= bands; band++) {
      const starts = idx.tileStart[band];
      if (!starts) continue;
      for (let ty = y0; ty <= y1; ty++) {
        const first = starts[ty * n + x0];
        const end = starts[ty * n + x1 + 1];
        if (end > first) out.push([first, end - first]);
      }
    }
    return out;
  }

  private pointRefs(z: number): PointRef[] {
    if (this.pointMode === "off") return [];
    const size = Math.max(1, Math.min(3.5, this.scale / 900)) * this.dprNow();
    const mask = this.corpusMask;
    const out: PointRef[] = [];

    if (this.pointMode === "deep") {
      for (const [tx, ty] of this.visibleTileList(this.points.deepZ)) {
        const b = this.points.deepTile(tx, ty);
        if (!b || !b.n) continue;
        const name = deepName(tx, ty);
        if (!this.base.hasPoints(name)) this.base.uploadPoints(name, b);
        out.push({ name, ranges: [[0, b.n]], size, alpha: 0.9, mask, maxMinZ: 255 });
      }
      if (out.length) return out;
    }

    const lod = this.points.lod;
    if (lod && lod.n) {
      if (!this.base.hasPoints(LOD_BUFFER)) {
        this.base.uploadPoints(LOD_BUFFER, lod);
        this.lodIndex = buildLodIndex(
          lod,
          this.manifest.points?.deep?.tile_index?.z ?? this.zmax,
          this.manifest.points?.lod?.zoom_offsets,
        );
      }
      out.push({
        name: LOD_BUFFER,
        ranges: this.lodRanges(z),
        size,
        alpha: 0.9,
        mask,
        maxMinZ: z,
      });
    }
    return out;
  }

  private dprNow(): number {
    return Math.min(window.devicePixelRatio || 1, 2);
  }

  /** Drop GPU point buffers when the tier is turned off. */
  releasePoints(which: "lod" | "deep" | "all") {
    if (which === "lod" || which === "all") {
      this.base.dropPoints(LOD_BUFFER);
      this.lodIndex = null;
    }
    if (which === "deep" || which === "all")
      this.base.retainPoints(new Set([LOD_BUFFER]));
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
    const consider = (b: PointBatch, tier: "lod" | "deep", limit: number) => {
      const n = Math.min(limit || b.n, b.n);
      for (let i = 0; i < n; i++) {
        if (!this.enabled.has(b.corpus[i])) continue;
        if (b.minz && b.minz[i] > z) continue;
        const sx = (b.u[i] - this.cx) * this.scale + this.width / 2;
        const sy = (b.v[i] - this.cy) * this.scale + this.height / 2;
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
        if (b && b.n) consider(b, "deep", b.n);
      }
    }
    if (!best && this.points.lod) consider(this.points.lod, "lod", this.lodCount(z));
    return best;
  }

  // -- events ---------------------------------------------------------------

  private attach() {
    // The overlay canvas is the topmost layer, so it owns the pointer.
    const c = this.overlayCanvas;
    const on = <K extends keyof HTMLElementEventMap>(
      type: K,
      fn: (e: HTMLElementEventMap[K]) => void,
      opts?: AddEventListenerOptions,
    ) => {
      c.addEventListener(type, fn as EventListener, opts);
      this.detach.push(() => c.removeEventListener(type, fn as EventListener, opts));
    };

    on("pointerdown", (e) => {
      c.setPointerCapture(e.pointerId);
      this.dragging = true;
      this.moved = 0;
      this.lastPointer = { x: e.offsetX, y: e.offsetY };
    });

    on("pointermove", (e) => {
      if (this.dragging && this.lastPointer) {
        const dx = e.offsetX - this.lastPointer.x;
        const dy = e.offsetY - this.lastPointer.y;
        this.moved += Math.abs(dx) + Math.abs(dy);
        this.cx -= dx / this.scale;
        this.cy -= dy / this.scale;
        this.lastPointer = { x: e.offsetX, y: e.offsetY };
        this.clampCamera();
        this.loop.markView();
        this.onViewChange();
        return;
      }
      // HOVER PATH: overlay only. Nothing here may dirty the base layer.
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
        this.loop.markOverlay();
      }
    });

    const end = (e: PointerEvent) => {
      if (!this.dragging) return;
      this.dragging = false;
      this.lastPointer = null;
      if (this.moved < 4) this.handleClick(e);
    };
    on("pointerup", end);
    on("pointercancel", () => {
      this.dragging = false;
      this.lastPointer = null;
    });
    on("pointerleave", () => {
      if (this.hovered) {
        this.hovered = null;
        this.onHover(null);
        this.loop.markOverlay();
      }
    });
    on(
      "wheel",
      (e) => {
        e.preventDefault();
        const f = Math.pow(2, -e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0022));
        this.zoomBy(f, e.offsetX, e.offsetY);
      },
      { passive: false } as AddEventListenerOptions,
    );
    on("dblclick", (e) => this.zoomBy(2, e.offsetX, e.offsetY));
  }

  private handleClick(e: PointerEvent) {
    const p = this.pickPoint(e.offsetX, e.offsetY);
    if (p) {
      this.selected = p;
      this.onPick(p);
      this.loop.markOverlay();
      return;
    }
    const h = this.binAt(e.offsetX, e.offsetY, this.previewLevelCap);
    this.pinned = h;
    this.selected = null;
    this.onPick(null);
    this.onHover(h);
    this.loop.markOverlay();
  }

  dispose() {
    for (const off of this.detach) off();
    this.detach = [];
    this.loop.dispose();
    this.base.dispose();
  }
}
