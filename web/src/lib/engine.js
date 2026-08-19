// ViewerEngine — the canvas core, ported from experiments/viewer_assets/viewer.js.
// Owns the 2D canvas, pan/zoom transform, LOD level pick, grid/points/anchor/query
// drawing, hover -> supertile sample fetch + tooltip DOM. React components drive it
// via setters (setMode, setGridOverlay, setPointOverlays, setProbe, setQuery, ...)
// and read back the live density max via the onLegend callback.

import { parseGrid, parsePoints, parseAnchors } from "./parsers.js";
import { densityRamp, accentRamp, anchorRamp, rampSample } from "./ramps.js";
import { fmt } from "./format.js";
import { cellAt, containingSampleCell, cellDataBounds } from "./hover.js";
import { tiledLevelMap, tilesForViewport, pickLevelFrom, tileCacheKey } from "./tiles.js";
import { wheelFactor, BUTTON_IN, BUTTON_OUT, DBLCLICK_IN } from "./zoom.js";

const MIN_CELL_PX = 7; // finest level whose cells still reach this many px
const TWO_PI = 6.2832;

export class ViewerEngine {
  constructor({ canvas, tooltip, dataDir, manifest, onLegend, isDark }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.tooltip = tooltip;
    this.dataDir = String(dataDir).replace(/\/$/, "");
    this.manifest = manifest;
    this.onLegend = onLegend || (() => {});
    this.isDark = isDark || (() => false);

    const ex = manifest.extent;
    this.extent = { x0: ex[0], y0: ex[1], x1: ex[2], y1: ex[3], w: ex[2] - ex[0] || 1, h: ex[3] - ex[1] || 1 };
    // Base layer is ONLY ever a grid layer. Point-only (projection) manifests
    // have none — the engine must then never fetch grid-*.bin or samples-*.json.
    this.baseLayer = (manifest.layers || []).find((l) => l.kind === "grid") || null;
    if (this.baseLayer && (!this.baseLayer.levels || !this.baseLayer.levels.length))
      this.baseLayer.levels =
        manifest.levels && manifest.levels.length ? manifest.levels : [256];
    // Deep-zoom tiled fine levels (v3). Map(level -> split); empty when the base
    // layer declares no tiled_levels (older manifests / all projection maps) —
    // in which case no fine level is ever picked and no tile is ever fetched.
    this.tiledLevels = tiledLevelMap(this.baseLayer && this.baseLayer.tiled_levels);
    // Candidate LOD levels for the base layer = plain levels ∪ tiled levels.
    this.baseLevels = this.baseLayer
      ? Array.from(new Set([...(this.baseLayer.levels || []), ...this.tiledLevels.keys()])).sort((a, b) => a - b)
      : [];

    this.view = null; // [x0,x1,y0,y1] data space (screen y flips)
    this.cssW = 0; this.cssH = 0; this.dpr = 1;
    this.mode = "map"; // "map" | "metrics"
    this.metricMode = "anchors"; // "anchors" | "queries"
    this.gridOverlay = null; // subset grid layer key or null
    this.pointOverlays = []; // [{ key, label, group, accent }]
    this.probe = null; this.query = null;
    this.anchors = null; this.queriesDoc = null;

    this.gridCache = new Map(); this.gridInflight = new Set();
    this.tileCache = new Map(); this.tileInflight = new Map(); // deep-zoom tiles
    this.pointsCache = new Map();
    this.sampleCache = new Map(); this.sampleInflight = new Map();
    this.count256 = null;
    // Currently-rendered base representation, for hover alignment: the grids
    // actually drawn (plain: one grid; tiled: the loaded viewport tiles) and the
    // level they are at. hoverCell = { cx, cy, level, idx } at that rendered level.
    this.renderedGrids = [];
    this.renderLevel = 0;
    this.hoverCell = null;
    this.raf = 0;
    this._lastLegend = "";
    this._bound = [];
  }

  url(name) { return `${this.dataDir}/${name}`; }
  async fetchBuf(name, signal) {
    const r = await fetch(this.url(name), signal ? { signal } : undefined);
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return r.arrayBuffer();
  }
  layerByKey(key) { return (this.manifest.layers || []).find((l) => l.key === key); }

  // ---- lifecycle ---------------------------------------------------------
  async boot() {
    this.resetView();
    this.bindEvents();
    // preload sample-level grid for hover counts + first paint (grid maps only —
    // point-only maps have no grid or samples files to fetch)
    if (this.baseLayer) {
      const g = await this.getGrid(this.baseLayer.key, this.manifest.sample_level || this.baseLayer.levels[0]);
      if (g) {
        this.count256 = new Map();
        for (let i = 0; i < g.cells.length; i++) this.count256.set(g.cells[i], g.counts[i]);
      }
    }
    this.requestDraw();
  }
  destroy() {
    for (const [t, ev, fn, opt] of this._bound) t.removeEventListener(ev, fn, opt);
    this._bound = [];
    for (const [, ctrl] of this.tileInflight) ctrl.abort();
    this.tileInflight.clear();
    if (this.raf) cancelAnimationFrame(this.raf);
  }
  on(target, ev, fn, opt) { target.addEventListener(ev, fn, opt); this._bound.push([target, ev, fn, opt]); }

  // ---- setters (React -> engine) ----------------------------------------
  setMode(mode) {
    this.mode = mode; this.hideTip();
    if (mode === "metrics") {
      const m = this.manifest.metrics || {};
      if (m.anchors) { this.metricMode = "anchors"; this.loadAnchors(); }
      else if (m.probes && m.probes.length) { this.metricMode = "queries"; this.loadQueriesDoc(); }
    }
    this.requestDraw();
  }
  setMetricMode(mm) {
    this.metricMode = mm;
    if (mm === "anchors") this.loadAnchors();
    else this.loadQueriesDoc();
    this.requestDraw();
  }
  setGridOverlay(key) { this.gridOverlay = key || null; this.requestDraw(); }
  setPointOverlays(list) {
    this.pointOverlays = (list || []).map((o) => (typeof o === "string" ? { key: o } : o));
    for (const o of this.pointOverlays) this.loadPoints(o.key);
    this.requestDraw();
  }
  setProbe(p) { this.probe = p; this.query = null; if (p) this.fitProbe(p); this.requestDraw(); }
  setQuery(q) { this.query = q; if (q) this.fitQuery(q); this.requestDraw(); }
  setTheme() { this.requestDraw(); }

  // ---- data loaders ------------------------------------------------------
  async getGrid(key, level) {
    const ck = `${key}-${level}`;
    if (this.gridCache.has(ck)) return this.gridCache.get(ck);
    if (this.gridInflight.has(ck)) return null;
    this.gridInflight.add(ck);
    try {
      const g = parseGrid(await this.fetchBuf(`grid-${key}-${level}.bin`));
      this.gridCache.set(ck, g);
      return g;
    } catch (e) {
      this.gridCache.set(ck, null); // negative cache
      console.warn(`grid ${ck} unavailable:`, e.message);
      return null;
    } finally {
      this.gridInflight.delete(ck);
      this.requestDraw();
    }
  }
  bestGrid(key, level, levels) {
    const asc = levels.slice().sort((a, b) => a - b);
    let best = null;
    for (const L of asc) { const g = this.gridCache.get(`${key}-${L}`); if (g && L <= level) best = g; }
    if (!best) for (const L of asc) { const g = this.gridCache.get(`${key}-${L}`); if (g) { best = g; break; } }
    return best;
  }
  // ---- deep-zoom tiles ---------------------------------------------------
  // Cached count map (cellIdx -> count) per parsed grid, built once. Shared by
  // hover lookup and by tiled combined-max computation.
  gridCountMap(g) {
    if (!g._cmap) {
      const m = new Map();
      for (let i = 0; i < g.cells.length; i++) m.set(g.cells[i], g.counts[i]);
      g._cmap = m;
    }
    return g._cmap;
  }
  async getTile(key, level, tx, ty) {
    const ck = tileCacheKey(key, level, tx, ty);
    if (this.tileCache.has(ck)) return this.tileCache.get(ck);
    if (this.tileInflight.has(ck)) return null;
    const ctrl = new AbortController();
    this.tileInflight.set(ck, ctrl);
    try {
      const g = parseGrid(await this.fetchBuf(`grid-${key}-${level}-${tx}_${ty}.bin`, ctrl.signal));
      this.tileCache.set(ck, g);
      return g;
    } catch (e) {
      if (e.name !== "AbortError") { this.tileCache.set(ck, null); console.warn(`tile ${ck} unavailable:`, e.message); }
      return null;
    } finally {
      this.tileInflight.delete(ck);
      this.requestDraw();
    }
  }
  // Ensure the viewport tiles for a tiled level are loading; abort any in-flight
  // tile that is no longer needed (stale after pan/zoom). Returns the currently
  // loaded grids for the needed tiles.
  ensureTiles(key, level, split) {
    const need = tilesForViewport([this.view[0], this.view[1], this.view[2], this.view[3]], this.extent, level, split);
    const needKeys = new Set(need.map(({ tx, ty }) => tileCacheKey(key, level, tx, ty)));
    for (const [ck, ctrl] of this.tileInflight) {
      if (!needKeys.has(ck)) { ctrl.abort(); this.tileInflight.delete(ck); }
    }
    const loaded = [];
    for (const { tx, ty } of need) {
      const ck = tileCacheKey(key, level, tx, ty);
      const g = this.tileCache.get(ck);
      if (g) loaded.push(g);
      else if (!this.tileInflight.has(ck)) this.getTile(key, level, tx, ty);
    }
    return loaded;
  }
  async loadPoints(key) {
    if (this.pointsCache.has(key)) return this.pointsCache.get(key);
    try {
      const xy = parsePoints(await this.fetchBuf(`points-${key}.bin`));
      this.pointsCache.set(key, xy); this.requestDraw(); return xy;
    } catch (e) {
      console.warn("points", key, e.message);
      this.pointsCache.set(key, new Float32Array(0)); return null;
    }
  }
  async loadAnchors() {
    if (this.anchors) return this.anchors;
    // Never fetch when the manifest declares no anchor metrics (projection
    // maps), and never retry a failed fetch on every draw.
    const meta = this.manifest.metrics && this.manifest.metrics.anchors;
    if (!meta || this._anchorsFailed || this._anchorsInflight) return null;
    this._anchorsInflight = true;
    try {
      this.anchors = parseAnchors(await this.fetchBuf(meta.file || "metrics-anchors.bin"));
      this.requestDraw();
    } catch (e) {
      this._anchorsFailed = true;
      console.warn("anchors", e.message);
    } finally { this._anchorsInflight = false; }
    return this.anchors;
  }
  async loadQueriesDoc() {
    if (this.queriesDoc) return this.queriesDoc;
    try {
      this.queriesDoc = await (await fetch(this.url("metrics-queries.json"))).json();
    } catch (e) { console.warn("queries", e.message); this.queriesDoc = { probes: [] }; }
    this.requestDraw();
    return this.queriesDoc;
  }

  // ---- geometry / transform ---------------------------------------------
  sx(x) { return (x - this.view[0]) * this.cssW / (this.view[1] - this.view[0]); }
  sy(y) { return this.cssH - (y - this.view[2]) * this.cssH / (this.view[3] - this.view[2]); }
  dataX(px) { return this.view[0] + px * (this.view[1] - this.view[0]) / this.cssW; }
  dataY(py) { return this.view[2] + (this.cssH - py) * (this.view[3] - this.view[2]) / this.cssH; }

  resetView() {
    const e = this.extent, padx = e.w * 0.03, pady = e.h * 0.03;
    this.view = [e.x0 - padx, e.x1 + padx, e.y0 - pady, e.y1 + pady];
    this.requestDraw();
  }
  zoomAt(fx, fy, factor) {
    const cx = this.view[0] + fx * (this.view[1] - this.view[0]);
    const cy = this.view[2] + fy * (this.view[3] - this.view[2]);
    this.view = [cx + (this.view[0] - cx) * factor, cx + (this.view[1] - cx) * factor,
                 cy + (this.view[2] - cy) * factor, cy + (this.view[3] - cy) * factor];
    this.requestDraw();
  }
  pickLevel(levels) {
    return pickLevelFrom(levels, this.extent.w, this.view[1] - this.view[0], this.cssW, MIN_CELL_PX);
  }
  fitProbe(p) {
    const qs = p.queries || []; if (!qs.length) return;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const q of qs) {
      x0 = Math.min(x0, q.xy[0]); x1 = Math.max(x1, q.xy[0]); y0 = Math.min(y0, q.xy[1]); y1 = Math.max(y1, q.xy[1]);
      for (const nb of q.neighbors) { x0 = Math.min(x0, nb[0]); x1 = Math.max(x1, nb[0]); y0 = Math.min(y0, nb[1]); y1 = Math.max(y1, nb[1]); }
    }
    this.padView(x0, x1, y0, y1, 0.15);
  }
  fitQuery(q) {
    let x0 = q.xy[0], x1 = q.xy[0], y0 = q.xy[1], y1 = q.xy[1];
    for (const nb of q.neighbors) { x0 = Math.min(x0, nb[0]); x1 = Math.max(x1, nb[0]); y0 = Math.min(y0, nb[1]); y1 = Math.max(y1, nb[1]); }
    this.padView(x0, x1, y0, y1, 0.35);
  }
  padView(x0, x1, y0, y1, frac) {
    const w = (x1 - x0) || 1, h = (y1 - y0) || 1, px = w * frac + 0.001, py = h * frac + 0.001;
    this.view = [x0 - px, x1 + px, y0 - py, y1 + py];
  }

  // ---- css var access ----------------------------------------------------
  css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888"; }
  accentColor(slot) { return this.css(slot === "a2" ? "--accent-2" : "--accent"); }

  // ---- drawing -----------------------------------------------------------
  requestDraw() { if (this.raf) return; this.raf = requestAnimationFrame(() => { this.raf = 0; this.draw(); }); }
  sizeCanvas() {
    const r = this.canvas.parentElement.getBoundingClientRect();
    this.cssW = Math.max(1, r.width); this.cssH = Math.max(1, r.height);
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.cssW * this.dpr);
    this.canvas.height = Math.round(this.cssH * this.dpr);
  }
  draw() {
    if (!this.view) return;
    this.sizeCanvas();
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    if (this.mode === "map") this.drawMap();
    else if (this.metricMode === "anchors") this.drawAnchors();
    else { this.drawMap(); this.drawQueries(); }
  }
  emitLegend(info) {
    const key = JSON.stringify(info);
    if (key === this._lastLegend) return;
    this._lastLegend = key;
    this.onLegend(info);
  }
  drawGridLayer(g, muted, stops, maxOverride) {
    if (!g) return;
    const ctx = this.ctx, L = g.level, e = this.extent;
    const cw = (e.w / L) * this.cssW / (this.view[1] - this.view[0]);
    const ch = (e.h / L) * this.cssH / (this.view[3] - this.view[2]);
    const denom = Math.log((maxOverride || g.max || 1) + 1);
    ctx.globalAlpha = muted ? 0.32 : 1;
    const w = Math.max(1, Math.ceil(cw)), h = Math.max(1, Math.ceil(ch));
    for (let i = 0; i < g.cells.length; i++) {
      const idx = g.cells[i], cx = idx % L, cy = (idx - cx) / L;
      const xData = e.x0 + (cx + 0.5) * (e.w / L);
      const yData = e.y0 + (cy + 0.5) * (e.h / L);
      const px = this.sx(xData), py = this.sy(yData);
      if (px < -w || px > this.cssW + w || py < -h || py > this.cssH + h) continue;
      const t = Math.log(g.counts[i] + 1) / denom;
      ctx.fillStyle = rampSample(stops, t);
      ctx.fillRect(px - cw / 2, py - ch / 2, w, h);
    }
    ctx.globalAlpha = 1;
  }
  drawPointsLayer(xy, color, size, quiet) {
    const ctx = this.ctx; ctx.globalAlpha = quiet ? 0.5 : 0.85;
    const r = size, ring = 1.5, surf = this.css("--surface");
    for (let i = 0; i < xy.length; i += 2) {
      const px = this.sx(xy[i]), py = this.sy(xy[i + 1]);
      if (px < -4 || px > this.cssW + 4 || py < -4 || py > this.cssH + 4) continue;
      if (!quiet) { ctx.beginPath(); ctx.arc(px, py, r + ring, 0, TWO_PI); ctx.fillStyle = surf; ctx.fill(); }
      ctx.beginPath(); ctx.arc(px, py, r, 0, TWO_PI); ctx.fillStyle = color; ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
  drawMap() {
    const base = this.baseLayer;
    let legendMax = 1, legendOverlay = null;
    this.renderedGrids = []; this.renderLevel = 0;
    if (base) {
      const level = this.pickLevel(this.baseLevels);
      const split = this.tiledLevels.get(level); // defined only for tiled fine levels
      if (split) {
        // Deep zoom: fetch only viewport-intersecting tiles. Draw a coarse
        // fallback underneath so the view is never blank while tiles stream in.
        const tiles = this.ensureTiles(base.key, level, split);
        // Coarse fallback so the view is never blank while tiles stream in.
        const plainMax = (base.levels || []).slice().sort((a, b) => a - b).pop() || 256;
        this.getGrid(base.key, plainMax);
        const fallback = this.bestGrid(base.key, plainMax, base.levels);
        if (fallback && !tiles.length) this.drawGridLayer(fallback, !!this.gridOverlay, this.ramp());
        if (tiles.length) {
          let tmax = 1;
          for (const t of tiles) if (t.max > tmax) tmax = t.max;
          for (const t of tiles) this.drawGridLayer(t, !!this.gridOverlay, this.ramp(), tmax);
          this.renderedGrids = tiles; this.renderLevel = level; legendMax = tmax;
        } else if (fallback) {
          this.renderedGrids = [fallback]; this.renderLevel = fallback.level; legendMax = fallback.max;
        }
      } else {
        this.getGrid(base.key, level);
        const g = this.bestGrid(base.key, level, base.levels);
        this.drawGridLayer(g, !!this.gridOverlay, this.ramp());
        if (g) { this.renderedGrids = [g]; this.renderLevel = g.level; legendMax = g.max; }
      }
      if (this.gridOverlay) {
        const o = this.layerByKey(this.gridOverlay);
        if (o) {
          const oLevel = this.pickLevel(o.levels || base.levels);
          this.getGrid(o.key, oLevel);
          const og = this.bestGrid(o.key, oLevel, o.levels || base.levels);
          this.drawGridLayer(og, false, this.accent());
          legendMax = og ? og.max : 1; legendOverlay = { key: o.key, label: o.label };
        }
      }
    }
    // Point overlays: context layers first (muted gray background scatter, no
    // surface ring so they stay quiet), then accent layers on top.
    const ordered = [...this.pointOverlays].sort((a, b) => (a.context ? 0 : 1) - (b.context ? 0 : 1));
    for (const o of ordered) {
      const xy = this.pointsCache.get(o.key);
      if (!xy) { this.loadPoints(o.key); continue; }
      const color = o.context ? this.css("--ink-muted") : this.accentColor(o.accent);
      this.drawPointsLayer(xy, color, o.size || 2.6, o.context);
    }
    if (base) this.drawHoverHighlight();
    if (base) this.emitLegend({ kind: "density", maxCount: legendMax, overlay: legendOverlay });
    else this.emitLegend({ kind: "points" });
  }
  // Outline the EXACT rendered-level cell under the cursor so the highlight, the
  // drawn bins, and the tooltip count all agree at any zoom depth.
  drawHoverHighlight() {
    const hc = this.hoverCell;
    if (!hc || hc.level !== this.renderLevel) return;
    const [x0, x1, y0, y1] = cellDataBounds(hc.cx, hc.cy, this.extent, hc.level);
    const px0 = this.sx(x0), px1 = this.sx(x1), py0 = this.sy(y0), py1 = this.sy(y1);
    const left = Math.min(px0, px1), top = Math.min(py0, py1);
    const w = Math.abs(px1 - px0), h = Math.abs(py1 - py0);
    const ctx = this.ctx;
    ctx.save();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = this.css("--accent");
    ctx.strokeRect(left + 0.5, top + 0.5, Math.max(1, w - 1), Math.max(1, h - 1));
    ctx.restore();
  }
  drawAnchors() {
    const a = this.anchors; if (!a) { this.loadAnchors(); this.emitLegend({ kind: "anchor" }); return; }
    const ctx = this.ctx, stops = this.anchorRampStops(), surf = this.css("--surface");
    for (let i = 0; i < a.n; i++) {
      const px = this.sx(a.xy[2 * i]), py = this.sy(a.xy[2 * i + 1]);
      if (px < -6 || px > this.cssW + 6 || py < -6 || py > this.cssH + 6) continue;
      ctx.beginPath(); ctx.arc(px, py, 5, 0, TWO_PI); ctx.fillStyle = surf; ctx.fill();
      ctx.beginPath(); ctx.arc(px, py, 3.6, 0, TWO_PI); ctx.fillStyle = rampSample(stops, a.score[i]); ctx.fill();
    }
    this.emitLegend({ kind: "anchor" });
  }
  drawQueries() {
    if (!this.probe) return;
    const ctx = this.ctx, qs = this.probe.queries || [];
    ctx.globalAlpha = 0.5; ctx.fillStyle = this.css("--accent");
    for (const q of qs) {
      const px = this.sx(q.xy[0]), py = this.sy(q.xy[1]);
      if (px < -4 || px > this.cssW + 4 || py < -4 || py > this.cssH + 4) continue;
      ctx.beginPath(); ctx.arc(px, py, 2.4, 0, TWO_PI); ctx.fill();
    }
    ctx.globalAlpha = 1;
    const q = this.query; if (!q) return;
    const qx = this.sx(q.xy[0]), qy = this.sy(q.xy[1]);
    ctx.strokeStyle = this.css("--ink-muted"); ctx.lineWidth = 1; ctx.globalAlpha = 0.55;
    for (const nb of q.neighbors) { ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(this.sx(nb[0]), this.sy(nb[1])); ctx.stroke(); }
    ctx.globalAlpha = 1;
    const good = this.css("--status-good"), serious = this.css("--status-serious"), surf = this.css("--surface");
    for (let i = 0; i < q.neighbors.length; i++) {
      const nx = this.sx(q.neighbors[i][0]), ny = this.sy(q.neighbors[i][1]);
      if (q.hits[i]) {
        ctx.beginPath(); ctx.arc(nx, ny, 6, 0, TWO_PI); ctx.fillStyle = surf; ctx.fill();
        ctx.beginPath(); ctx.arc(nx, ny, 4.5, 0, TWO_PI); ctx.fillStyle = good; ctx.fill();
      } else {
        this.drawDiamond(nx, ny, 6.5, surf, true);
        this.drawDiamond(nx, ny, 5, serious, false, 2);
      }
    }
    ctx.beginPath(); ctx.arc(qx, qy, 7, 0, TWO_PI); ctx.fillStyle = surf; ctx.fill();
    ctx.beginPath(); ctx.arc(qx, qy, 5, 0, TWO_PI); ctx.fillStyle = this.css("--accent"); ctx.fill();
  }
  drawDiamond(x, y, r, color, fill, lw) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath();
    if (fill) { ctx.fillStyle = color; ctx.fill(); }
    else { ctx.strokeStyle = color; ctx.lineWidth = lw || 1.5; ctx.stroke(); }
  }
  ramp() { return densityRamp(this.isDark()); }
  accent() { return accentRamp(this.isDark()); }
  anchorRampStops() { return anchorRamp(this.isDark()); }

  // ---- supertile samples (hover) ----------------------------------------
  supertileKey(cx, cy) {
    const L = this.manifest.sample_level, st = this.manifest.super_tile, per = L / st;
    return `${Math.floor(cx / per)}_${Math.floor(cy / per)}`;
  }
  async getSamples(cx, cy) {
    const key = this.supertileKey(cx, cy);
    if (this.sampleCache.has(key)) return this.sampleCache.get(key);
    for (const [k, ctrl] of this.sampleInflight) { if (k !== key) { ctrl.abort(); this.sampleInflight.delete(k); } }
    if (this.sampleInflight.has(key)) return null;
    const ctrl = new AbortController(); this.sampleInflight.set(key, ctrl);
    try {
      const doc = await (await fetch(this.url(`samples-${this.baseLayer.key}-${key}.json`), { signal: ctrl.signal })).json();
      this.sampleCache.set(key, doc); return doc;
    } catch (e) {
      if (e.name !== "AbortError") this.sampleCache.set(key, { cells: {} });
      return null;
    } finally { this.sampleInflight.delete(key); }
  }

  // ---- tooltip -----------------------------------------------------------
  hideTip() { if (this.tooltip) this.tooltip.hidden = true; }
  positionTip(clientX, clientY) {
    const t = this.tooltip, pad = 14;
    let x = clientX + pad, y = clientY + pad;
    const r = t.getBoundingClientRect();
    if (x + r.width > innerWidth - 6) x = clientX - r.width - pad;
    if (y + r.height > innerHeight - 6) y = clientY - r.height - pad;
    t.style.left = Math.max(6, x) + "px"; t.style.top = Math.max(6, y) + "px";
  }
  // Count in a rendered-level cell (searches the loaded grids/tiles). Cell
  // indices are GLOBAL row-major so a single idx lookup works across tiles.
  renderedCount(idx) {
    for (const g of this.renderedGrids) { const m = this.gridCountMap(g); if (m.has(idx)) return m.get(idx); }
    return 0;
  }
  async hoverMap(clientX, clientY, hc) {
    // hc is the rendered-level cell { cx, cy, level, idx } resolved by mousemove.
    const dx = this.dataX(hc._lx), dy = this.dataY(hc._ly);
    const count = this.renderedCount(hc.idx);
    if (!count) { this.hideTip(); return; }
    const t = this.tooltip; t.innerHTML = "";
    const head = document.createElement("div");
    const c = document.createElement("div"); c.className = "tt-count";
    c.textContent = `${fmt(count)} row${count === 1 ? "" : "s"} in this bin`;
    const co = document.createElement("div"); co.className = "tt-coords";
    co.textContent = `x ${dx.toFixed(3)}, y ${dy.toFixed(3)}`;
    head.appendChild(c); head.appendChild(co); t.appendChild(head);
    t.hidden = false; this.positionTip(clientX, clientY);
    // Text samples come from the CONTAINING sample_level (256) cell — labeled so
    // the reader never conflates the (finer) rendered-bin count with the samples.
    const sampleLevel = this.manifest.sample_level;
    const sc = containingSampleCell(hc.cx, hc.cy, hc.level, sampleLevel);
    if (!sc) return;
    const doc = await this.getSamples(sc.cx, sc.cy);
    if (doc && this.hoverCell && this.hoverCell.idx === hc.idx) {
      const samples = (doc.cells && doc.cells[String(sc.idx)]) || [];
      const cap = document.createElement("div"); cap.className = "tt-caption"; cap.textContent = "sample texts from this area";
      t.appendChild(cap);
      if (!samples.length) {
        const em = document.createElement("div"); em.className = "tt-empty"; em.textContent = "no text sample nearby"; t.appendChild(em);
      } else {
        for (const s of samples.slice(0, 3)) {
          const box = document.createElement("div"); box.className = "tt-sample";
          if (s.g) { const g = document.createElement("div"); g.className = "tt-group"; g.textContent = s.g; box.appendChild(g); }
          const txt = document.createElement("div"); txt.className = "tt-text"; txt.textContent = s.t || ""; box.appendChild(txt);
          t.appendChild(box);
        }
      }
      this.positionTip(clientX, clientY);
    }
  }
  hoverPoints(o, clientX, clientY, localX, localY) {
    const xy = this.pointsCache.get(o.key);
    if (!xy || !xy.length) return false;
    let best = -1, bd = 12 * 12;
    for (let i = 0; i < xy.length; i += 2) {
      const px = this.sx(xy[i]) - localX, py = this.sy(xy[i + 1]) - localY;
      const d = px * px + py * py; if (d < bd) { bd = d; best = i; }
    }
    if (best < 0) return false;
    const t = this.tooltip; t.innerHTML = "";
    const head = document.createElement("div");
    const c = document.createElement("div"); c.className = "tt-count"; c.textContent = o.label || o.key; head.appendChild(c);
    if (o.group) { const g = document.createElement("div"); g.className = "tt-group"; g.textContent = o.group; head.appendChild(g); }
    const co = document.createElement("div"); co.className = "tt-coords";
    co.textContent = `x ${xy[best].toFixed(3)}, y ${xy[best + 1].toFixed(3)}`;
    head.appendChild(co); t.appendChild(head);
    t.hidden = false; this.positionTip(clientX, clientY);
    return true;
  }
  hoverAnchors(clientX, clientY, localX, localY) {
    const a = this.anchors; if (!a) { this.hideTip(); return; }
    let best = -1, bd = 12 * 12;
    for (let i = 0; i < a.n; i++) {
      const px = this.sx(a.xy[2 * i]) - localX, py = this.sy(a.xy[2 * i + 1]) - localY;
      const d = px * px + py * py; if (d < bd) { bd = d; best = i; }
    }
    if (best < 0) { this.hideTip(); return; }
    const t = this.tooltip; t.innerHTML = "";
    const label = (this.manifest.metrics && this.manifest.metrics.anchors && this.manifest.metrics.anchors.score) || "score";
    const row = document.createElement("div");
    const lab = document.createElement("span"); lab.className = "tt-count"; lab.textContent = `${label}: `;
    const val = document.createElement("span"); val.className = "tt-score"; val.textContent = a.score[best].toFixed(3);
    row.appendChild(lab); row.appendChild(val); t.appendChild(row);
    t.hidden = false; this.positionTip(clientX, clientY);
  }

  // ---- events ------------------------------------------------------------
  bindEvents() {
    const c = this.canvas; let drag = null;
    this.on(c, "wheel", (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const fx = (e.clientX - r.left) / r.width, fy = 1 - (e.clientY - r.top) / r.height;
      // Gentle exponential zoom (~1.12x per notch), smooth for trackpad deltas,
      // anchored at the cursor. See lib/zoom.js.
      this.zoomAt(fx, fy, wheelFactor(e.deltaY, e.deltaMode, r.height));
    }, { passive: false });
    this.on(c, "dblclick", (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const fx = (e.clientX - r.left) / r.width, fy = 1 - (e.clientY - r.top) / r.height;
      this.zoomAt(fx, fy, DBLCLICK_IN); // zoom in centered on the cursor
    });
    this.on(c, "mousedown", (e) => { drag = [e.clientX, e.clientY, this.view.slice()]; c.classList.add("dragging"); });
    this.on(window, "mouseup", () => { drag = null; c.classList.remove("dragging"); });
    this.on(window, "mousemove", (e) => {
      const r = c.getBoundingClientRect();
      if (drag) {
        const dx = (e.clientX - drag[0]) * (drag[2][1] - drag[2][0]) / r.width;
        const dy = (e.clientY - drag[1]) * (drag[2][3] - drag[2][2]) / r.height;
        this.view = [drag[2][0] - dx, drag[2][1] - dx, drag[2][2] + dy, drag[2][3] + dy];
        this.requestDraw(); return;
      }
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) { this.hideTip(); return; }
      const lx = e.clientX - r.left, ly = e.clientY - r.top;
      if (this.mode === "map") {
        // Accent point layers take hover priority; muted context points don't
        // grab the tooltip (they'd shadow everything on projection maps).
        for (const o of this.pointOverlays) {
          if (o.context) continue;
          if (this.hoverPoints(o, e.clientX, e.clientY, lx, ly)) { this.hoverCell = null; return; }
        }
        if (!this.baseLayer) { this.hideTip(); return; } // no bins to hover on point-only maps
        // Resolve the cell at the CURRENTLY RENDERED level (v3 hover alignment).
        // renderLevel is set by the last draw; fall back to sample_level pre-paint.
        const L = this.renderLevel || this.manifest.sample_level;
        const prev = this.hoverCell;
        const hc = cellAt(this.dataX(lx), this.dataY(ly), this.extent, L);
        this.hoverCell = hc;
        if (!hc) { this.hideTip(); if (prev) this.requestDraw(); return; }
        hc._lx = lx; hc._ly = ly;
        if (!prev || prev.idx !== hc.idx || prev.level !== hc.level) this.requestDraw();
        this.hoverMap(e.clientX, e.clientY, hc);
      } else if (this.mode === "metrics" && this.metricMode === "anchors") {
        this.hoverAnchors(e.clientX, e.clientY, lx, ly);
      } else { this.hideTip(); }
    });
    this.on(c, "mouseleave", () => { this.hideTip(); if (this.hoverCell) { this.hoverCell = null; this.requestDraw(); } });
    this.on(window, "keydown", (e) => {
      const tag = e.target && e.target.tagName;
      if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === "+" || e.key === "=") this.zoomAt(0.5, 0.5, BUTTON_IN);
      else if (e.key === "-" || e.key === "_") this.zoomAt(0.5, 0.5, BUTTON_OUT);
      else if (e.key === "0") this.resetView();
    });
    this.on(window, "resize", () => this.requestDraw());
  }
}
