/* ============================================================================
 * Basemap viewer — vanilla JS, 2D canvas.
 *
 * Reads a data/ directory matching the frozen `basemap-viewer-manifest-v1`
 * contract (see experiments design doc). The page template injects
 *   window.VIEWER_CONFIG = { dataDir: "data", back: "../../index.html" }
 * and everything else is discovered from manifest.json at runtime.
 *
 * Byte formats parsed here (all little-endian):
 *   grid-<layer>-<L>.bin : u32 magic 0x42494E31, u32 level, u32 ncells, u32 rsvd,
 *                          then u32[ncells] cellIdx (cy*L+cx, y in DATA space),
 *                          then u32[ncells] counts.
 *   points-<layer>.bin   : u32 magic 0x50545331, u32 npoints, then f32 x,y pairs.
 *   metrics-anchors.bin  : u32 magic 0x414E4331, u32 count, then f32 (x,y,score) triples.
 *
 * Pan/zoom transform ported from experiments/projection_gallery.py::_viewer_html.
 * LOD band (6–24 on-screen px per cell) ported from latent-scope atlasLod.js.
 * ========================================================================== */
"use strict";
(function () {
  const CFG = window.VIEWER_CONFIG || { dataDir: "data", back: "../../index.html" };
  const DATA = CFG.dataDir.replace(/\/$/, "");
  const url = (name) => `${DATA}/${name}`;

  const MAGIC = { GRID: 0x42494e31, PTS: 0x50545331, ANC: 0x414e4331 };
  const MIN_CELL_PX = 7;   // finest level whose cells still reach this many px
  const MAX_CELL_PX = 26;  // above this the level looks chunky; informs LOD choice

  // Density sequential ramp (blue, palette.md). Light: index0 = low count (near
  // surface, light) -> high count (dark). Dark mode reverses (low near dark
  // surface -> high bright). "flip anchor in dark", per color-formula.md.
  const RAMP_LIGHT = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7",
                      "#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"];
  const RAMP_DARK = RAMP_LIGHT.slice().reverse();
  // Diverging anchors ramp: blue (below median) <-> gray <-> red (above median).
  const ANCHOR_LIGHT = ["#256abf","#6da7ec","#f0efec","#ec835a","#d03b3b"];
  const ANCHOR_DARK  = ["#3987e5","#6da7ec","#383835","#ec835a","#d03b3b"];

  // ---- tiny color utils ----------------------------------------------------
  function hexRgb(h) {
    const n = parseInt(h.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function rampSample(stops, t) {
    t = Math.max(0, Math.min(1, t));
    const seg = t * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(seg));
    const f = seg - i;
    const a = hexRgb(stops[i]), b = hexRgb(stops[i + 1]);
    return `rgb(${Math.round(a[0] + (b[0]-a[0])*f)},${Math.round(a[1] + (b[1]-a[1])*f)},${Math.round(a[2] + (b[2]-a[2])*f)})`;
  }
  const fmt = (n) => Number(n).toLocaleString("en-US");

  // ---- state ---------------------------------------------------------------
  const S = {
    manifest: null,
    extent: null,           // {x0,y0,x1,y1,w,h}
    view: null,             // [x0,x1,y0,y1] data space (screen y flips)
    canvas: null, ctx: null, cssW: 0, cssH: 0, dpr: 1,
    mode: "map",            // "map" | "metrics"
    metricMode: "anchors",  // "anchors" | "queries"
    baseLayer: null,        // manifest layer obj for "all"
    overlay: null,          // {key,label,kind,levels,rows,group} or null
    gridCache: new Map(),   // `${key}-${L}` -> {level,cells,counts,max}
    gridInflight: new Set(),
    pointsCache: new Map(),  // key -> Float32Array
    sampleCache: new Map(),  // `${sx}_${sy}` -> {cells:{...}}
    sampleInflight: new Map(), // key -> AbortController
    count256: null,          // Map cellIdx->count for base layer @ sample_level
    anchors: null,           // {n, xy:Float32Array, score:Float32Array}
    queriesDoc: null,        // metrics-queries.json
    probe: null, query: null,
    theme: null,             // null=system, "light", "dark"
    hoverCell: null,
    raf: 0,
  };

  const $ = (id) => document.getElementById(id);

  // ---- data loaders --------------------------------------------------------
  async function fetchBuf(name, signal) {
    const r = await fetch(url(name), signal ? { signal } : undefined);
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return r.arrayBuffer();
  }
  function parseGrid(buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== MAGIC.GRID) throw new Error("bad grid magic");
    const level = dv.getUint32(4, true);
    const ncells = dv.getUint32(8, true);
    const cells = new Uint32Array(buf, 16, ncells);
    const counts = new Uint32Array(buf, 16 + ncells * 4, ncells);
    let max = 0;
    for (let i = 0; i < ncells; i++) if (counts[i] > max) max = counts[i];
    return { level, cells, counts, max };
  }
  function parsePoints(buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== MAGIC.PTS) throw new Error("bad points magic");
    const n = dv.getUint32(4, true);
    return new Float32Array(buf, 8, n * 2);
  }
  function parseAnchors(buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== MAGIC.ANC) throw new Error("bad anchors magic");
    const n = dv.getUint32(4, true);
    const f = new Float32Array(buf, 8, n * 3);
    const xy = new Float32Array(n * 2), score = new Float32Array(n);
    for (let i = 0; i < n; i++) { xy[2*i] = f[3*i]; xy[2*i+1] = f[3*i+1]; score[i] = f[3*i+2]; }
    return { n, xy, score };
  }

  async function getGrid(key, level) {
    const ck = `${key}-${level}`;
    if (S.gridCache.has(ck)) return S.gridCache.get(ck);
    if (S.gridInflight.has(ck)) return null;
    S.gridInflight.add(ck);
    try {
      const g = parseGrid(await fetchBuf(`grid-${key}-${level}.bin`));
      S.gridCache.set(ck, g);
      return g;
    } catch (e) {
      S.gridCache.set(ck, null); // negative cache — don't refetch a missing file
      console.warn(`grid ${ck} unavailable:`, e.message);
      return null;
    } finally {
      S.gridInflight.delete(ck);
      requestDraw();
    }
  }
  // best available grid at or coarser than `level` (for instant paint while
  // the target level streams in)
  function bestGrid(key, level, levels) {
    const asc = levels.slice().sort((a,b)=>a-b);
    let best = null;
    for (const L of asc) {
      const g = S.gridCache.get(`${key}-${L}`);
      if (g && L <= level) best = g;
    }
    if (!best) for (const L of asc) { const g = S.gridCache.get(`${key}-${L}`); if (g) { best = g; break; } }
    return best;
  }

  // ---- geometry / transform (ported from projection_gallery viewer) --------
  function sx(x) { return (x - S.view[0]) * S.cssW / (S.view[1] - S.view[0]); }
  function sy(y) { return S.cssH - (y - S.view[2]) * S.cssH / (S.view[3] - S.view[2]); }
  function dataX(px) { return S.view[0] + px * (S.view[1] - S.view[0]) / S.cssW; }
  function dataY(py) { return S.view[2] + (S.cssH - py) * (S.view[3] - S.view[2]) / S.cssH; }

  function resetView() {
    const e = S.extent, padx = e.w * 0.03, pady = e.h * 0.03;
    S.view = [e.x0 - padx, e.x1 + padx, e.y0 - pady, e.y1 + pady];
  }
  function zoomAt(fx, fy, factor) {
    // fx,fy in [0,1] fractional data-space anchor
    const cx = S.view[0] + fx * (S.view[1] - S.view[0]);
    const cy = S.view[2] + fy * (S.view[3] - S.view[2]);
    S.view = [cx + (S.view[0]-cx)*factor, cx + (S.view[1]-cx)*factor,
              cy + (S.view[2]-cy)*factor, cy + (S.view[3]-cy)*factor];
    requestDraw();
  }

  function pickLevel(levels) {
    const asc = levels.slice().sort((a,b)=>a-b);
    const viewW = S.view[1] - S.view[0];
    let chosen = asc[0];
    for (const L of asc) {
      const cellPx = S.cssW * (S.extent.w / L) / viewW;
      if (cellPx >= MIN_CELL_PX) chosen = L; // keep finest that still clears the floor
    }
    return chosen;
  }

  // ---- drawing -------------------------------------------------------------
  const ramp = () => (isDark() ? RAMP_DARK : RAMP_LIGHT);
  const anchorRamp = () => (isDark() ? ANCHOR_DARK : ANCHOR_LIGHT);
  function isDark() {
    if (S.theme === "dark") return true;
    if (S.theme === "light") return false;
    return matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function requestDraw() {
    if (S.raf) return;
    S.raf = requestAnimationFrame(() => { S.raf = 0; draw(); });
  }

  function sizeCanvas() {
    const r = S.canvas.parentElement.getBoundingClientRect();
    S.cssW = Math.max(1, r.width); S.cssH = Math.max(1, r.height);
    S.dpr = window.devicePixelRatio || 1;
    S.canvas.width = Math.round(S.cssW * S.dpr);
    S.canvas.height = Math.round(S.cssH * S.dpr);
  }

  function drawGridLayer(g, key, muted) {
    if (!g) return;
    const ctx = S.ctx, L = g.level, e = S.extent;
    const cw = (e.w / L) * S.cssW / (S.view[1] - S.view[0]);
    const ch = (e.h / L) * S.cssH / (S.view[3] - S.view[2]);
    const denom = Math.log((g.max || 1) + 1);
    ctx.globalAlpha = muted ? 0.32 : 1;
    const w = Math.max(1, Math.ceil(cw)), h = Math.max(1, Math.ceil(ch));
    for (let i = 0; i < g.cells.length; i++) {
      const idx = g.cells[i], cx = idx % L, cy = (idx - cx) / L;
      const xData = e.x0 + (cx + 0.5) * (e.w / L);
      const yData = e.y0 + (cy + 0.5) * (e.h / L);
      const px = sx(xData), py = sy(yData);
      if (px < -w || px > S.cssW + w || py < -h || py > S.cssH + h) continue;
      const t = Math.log(g.counts[i] + 1) / denom;
      ctx.fillStyle = rampSample(ramp(), t);
      ctx.fillRect(px - cw/2, py - ch/2, w, h);
    }
    ctx.globalAlpha = 1;
  }

  function drawPointsLayer(key, xy, color, size) {
    const ctx = S.ctx; ctx.fillStyle = color; ctx.globalAlpha = 0.85;
    const r = size, ring = 1.5;
    for (let i = 0; i < xy.length; i += 2) {
      const px = sx(xy[i]), py = sy(xy[i+1]);
      if (px < -4 || px > S.cssW + 4 || py < -4 || py > S.cssH + 4) continue;
      // 2px surface ring so points stay legible where they overlap
      ctx.beginPath(); ctx.arc(px, py, r + ring, 0, 6.2832);
      ctx.fillStyle = css("--surface"); ctx.fill();
      ctx.beginPath(); ctx.arc(px, py, r, 0, 6.2832);
      ctx.fillStyle = color; ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function css(varName) {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim() || "#888";
  }

  function drawMap() {
    const base = S.baseLayer;
    const level = pickLevel(base.levels);
    getGrid(base.key, level); // kick off load (async; redraws on arrival)
    const g = bestGrid(base.key, level, base.levels);
    drawGridLayer(g, base.key, !!S.overlay);

    if (S.overlay) {
      const o = S.overlay;
      if (o.kind === "grid") {
        const oLevel = pickLevel(o.levels || base.levels);
        getGrid(o.key, oLevel);
        drawGridLayer(bestGrid(o.key, oLevel, o.levels || base.levels), o.key, false);
      } else {
        const xy = S.pointsCache.get(o.key);
        if (xy) drawPointsLayer(o.key, xy, css("--accent"), 2.6);
        else loadPoints(o.key);
      }
    }
    updateLegend("density", g ? g.max : 1);
  }

  function drawAnchors() {
    const a = S.anchors;
    if (!a) return;
    const ctx = S.ctx, stops = anchorRamp();
    for (let i = 0; i < a.n; i++) {
      const px = sx(a.xy[2*i]), py = sy(a.xy[2*i+1]);
      if (px < -6 || px > S.cssW + 6 || py < -6 || py > S.cssH + 6) continue;
      ctx.beginPath(); ctx.arc(px, py, 5, 0, 6.2832);
      ctx.fillStyle = css("--surface"); ctx.fill();
      ctx.beginPath(); ctx.arc(px, py, 3.6, 0, 6.2832);
      ctx.fillStyle = rampSample(stops, a.score[i]); ctx.fill();
    }
    updateLegend("anchor", 1);
  }

  function drawQueries() {
    if (!S.probe) return;
    const ctx = S.ctx;
    const qs = S.probe.queries || [];
    // all query anchor points as faint accent markers
    ctx.globalAlpha = 0.5; ctx.fillStyle = css("--accent");
    for (const q of qs) {
      const px = sx(q.xy[0]), py = sy(q.xy[1]);
      if (px < -4 || px > S.cssW + 4 || py < -4 || py > S.cssH + 4) continue;
      ctx.beginPath(); ctx.arc(px, py, 2.4, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;

    const q = S.query;
    if (!q) return;
    const qx = sx(q.xy[0]), qy = sy(q.xy[1]);
    // connectors
    ctx.strokeStyle = css("--ink-muted"); ctx.lineWidth = 1; ctx.globalAlpha = 0.55;
    for (const nb of q.neighbors) {
      ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(sx(nb[0]), sy(nb[1])); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // neighbor markers: hit = filled circle (good), miss = hollow diamond (serious)
    const good = css("--status-good"), serious = css("--status-serious"), surf = css("--surface");
    for (let i = 0; i < q.neighbors.length; i++) {
      const nx = sx(q.neighbors[i][0]), ny = sy(q.neighbors[i][1]);
      if (q.hits[i]) {
        ctx.beginPath(); ctx.arc(nx, ny, 6, 0, 6.2832); ctx.fillStyle = surf; ctx.fill();
        ctx.beginPath(); ctx.arc(nx, ny, 4.5, 0, 6.2832); ctx.fillStyle = good; ctx.fill();
      } else {
        drawDiamond(ctx, nx, ny, 6.5, surf, true);
        drawDiamond(ctx, nx, ny, 5, serious, false, 2);
      }
    }
    // query marker on top: bold accent ring
    ctx.beginPath(); ctx.arc(qx, qy, 7, 0, 6.2832); ctx.fillStyle = surf; ctx.fill();
    ctx.beginPath(); ctx.arc(qx, qy, 5, 0, 6.2832); ctx.fillStyle = css("--accent"); ctx.fill();
  }

  function drawDiamond(ctx, x, y, r, color, fill, lw) {
    ctx.beginPath();
    ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath();
    if (fill) { ctx.fillStyle = color; ctx.fill(); }
    else { ctx.strokeStyle = color; ctx.lineWidth = lw || 1.5; ctx.stroke(); }
  }

  function draw() {
    if (!S.view) return;
    sizeCanvas();
    const ctx = S.ctx;
    ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
    ctx.clearRect(0, 0, S.cssW, S.cssH);
    if (S.mode === "map") { drawMap(); }
    else if (S.metricMode === "anchors") { drawAnchors(); }
    else { drawMap(); drawQueries(); } // queries drawn over a muted base for context
  }

  // ---- legend --------------------------------------------------------------
  function updateLegend(kind, maxCount) {
    const el = $("legend");
    if (kind === "density") {
      const ticks = [];
      const cap = Math.max(1, maxCount);
      for (let v = 1; v <= cap; v *= 10) {
        const pos = Math.log(v + 1) / Math.log(cap + 1);
        ticks.push(`<span style="left:${(pos*100).toFixed(1)}%">${fmt(v)}</span>`);
      }
      ticks.push(`<span style="left:100%">${fmt(cap)}</span>`);
      el.innerHTML =
        `<div class="legend-title">rows per bin (log scale)</div>` +
        `<div class="ramp-bar" style="background:var(--ramp-legend)"></div>` +
        `<div class="ramp-ticks">${ticks.join("")}</div>`;
    } else {
      const label = (S.manifest.metrics && S.manifest.metrics.anchors && S.manifest.metrics.anchors.score) || "score";
      el.innerHTML =
        `<div class="legend-title">${escText(label)}</div>` +
        `<div class="ramp-bar" style="background:var(--anchor-legend)"></div>` +
        `<div class="ramp-ticks"><span style="left:0%">low</span><span style="left:50%">median</span><span style="left:100%">high</span></div>`;
    }
    el.hidden = false;
  }

  function escText(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

  // ---- points / metrics loaders -------------------------------------------
  async function loadPoints(key) {
    if (S.pointsCache.has(key)) return S.pointsCache.get(key);
    try {
      const xy = parsePoints(await fetchBuf(`points-${key}.bin`));
      S.pointsCache.set(key, xy); requestDraw(); return xy;
    } catch (e) { console.warn("points", key, e.message); S.pointsCache.set(key, new Float32Array(0)); return null; }
  }
  async function loadAnchors() {
    if (S.anchors) return;
    setStatus("loading", "Loading anchors…");
    try {
      S.anchors = parseAnchors(await fetchBuf(S.manifest.metrics.anchors.file || "metrics-anchors.bin"));
      hideStatus(); requestDraw();
    } catch (e) { setError("Could not load anchors", e.message); }
  }
  async function loadQueriesDoc() {
    if (S.queriesDoc) return S.queriesDoc;
    S.queriesDoc = await (await fetch(url("metrics-queries.json"))).json();
    return S.queriesDoc;
  }

  // ---- supertile samples (hover) ------------------------------------------
  function supertileKey(cx, cy) {
    const L = S.manifest.sample_level, st = S.manifest.super_tile;
    const per = L / st;
    return `${Math.floor(cx / per)}_${Math.floor(cy / per)}`;
  }
  async function getSamples(cx, cy) {
    const key = supertileKey(cx, cy);
    if (S.sampleCache.has(key)) return S.sampleCache.get(key);
    // abort any other in-flight supertile fetch — only the current hover matters
    for (const [k, ctrl] of S.sampleInflight) { if (k !== key) { ctrl.abort(); S.sampleInflight.delete(k); } }
    if (S.sampleInflight.has(key)) return null;
    const ctrl = new AbortController();
    S.sampleInflight.set(key, ctrl);
    try {
      const doc = await (await fetch(url(`samples-${S.baseLayer.key}-${key}.json`), { signal: ctrl.signal })).json();
      S.sampleCache.set(key, doc); return doc;
    } catch (e) {
      if (e.name !== "AbortError") S.sampleCache.set(key, { cells: {} });
      return null;
    } finally { S.sampleInflight.delete(key); }
  }

  // ---- tooltip -------------------------------------------------------------
  const tip = () => $("tooltip");
  function hideTip() { tip().hidden = true; }
  function positionTip(clientX, clientY) {
    const t = tip(), pad = 14;
    let x = clientX + pad, y = clientY + pad;
    const r = t.getBoundingClientRect();
    if (x + r.width > innerWidth - 6) x = clientX - r.width - pad;
    if (y + r.height > innerHeight - 6) y = clientY - r.height - pad;
    t.style.left = Math.max(6, x) + "px"; t.style.top = Math.max(6, y) + "px";
  }

  async function hoverMap(clientX, clientY, localX, localY) {
    const L = S.manifest.sample_level;
    const e = S.extent;
    const dx = dataX(localX), dy = dataY(localY);
    let cx = Math.floor((dx - e.x0) / (e.w / L));
    let cy = Math.floor((dy - e.y0) / (e.h / L));
    if (cx < 0 || cy < 0 || cx >= L || cy >= L) { hideTip(); return; }
    const cellIdx = cy * L + cx;
    const count = S.count256 ? (S.count256.get(cellIdx) || 0) : 0;
    if (!count) { hideTip(); return; }

    const t = tip();
    t.innerHTML = "";
    const head = document.createElement("div");
    const c = document.createElement("div"); c.className = "tt-count";
    c.textContent = `${fmt(count)} row${count === 1 ? "" : "s"} in this bin`;
    const co = document.createElement("div"); co.className = "tt-coords";
    co.textContent = `x ${dx.toFixed(3)}, y ${dy.toFixed(3)}`;
    head.appendChild(c); head.appendChild(co); t.appendChild(head);
    t.hidden = false; positionTip(clientX, clientY);

    const doc = await getSamples(cx, cy);
    if (!doc || S.hoverCell !== cellIdx) { /* stale hover */ }
    if (doc && S.hoverCell === cellIdx) {
      const samples = (doc.cells && doc.cells[String(cellIdx)]) || [];
      if (!samples.length) {
        const em = document.createElement("div"); em.className = "tt-empty";
        em.textContent = "no text sample for this bin"; t.appendChild(em);
      } else {
        for (const s of samples.slice(0, 3)) {
          const box = document.createElement("div"); box.className = "tt-sample";
          if (s.g) { const g = document.createElement("div"); g.className = "tt-group"; g.textContent = s.g; box.appendChild(g); }
          const txt = document.createElement("div"); txt.className = "tt-text"; txt.textContent = s.t || ""; box.appendChild(txt);
          t.appendChild(box);
        }
      }
      positionTip(clientX, clientY);
    }
  }

  function hoverAnchors(clientX, clientY, localX, localY) {
    const a = S.anchors; if (!a) { hideTip(); return; }
    let best = -1, bd = 12 * 12;
    for (let i = 0; i < a.n; i++) {
      const px = sx(a.xy[2*i]) - localX, py = sy(a.xy[2*i+1]) - localY;
      const d = px*px + py*py; if (d < bd) { bd = d; best = i; }
    }
    if (best < 0) { hideTip(); return; }
    const t = tip(); t.innerHTML = "";
    const label = (S.manifest.metrics && S.manifest.metrics.anchors && S.manifest.metrics.anchors.score) || "score";
    const row = document.createElement("div");
    const lab = document.createElement("span"); lab.className = "tt-count"; lab.textContent = `${escLabel(label)}: `;
    const val = document.createElement("span"); val.className = "tt-score"; val.textContent = a.score[best].toFixed(3);
    row.appendChild(lab); row.appendChild(val); t.appendChild(row);
    t.hidden = false; positionTip(clientX, clientY);
  }
  function escLabel(s) { return s; }

  // ---- header / panel rendering -------------------------------------------
  function renderHeader() {
    const m = S.manifest;
    document.title = m.title || "Basemap viewer";
    $("title").textContent = m.title || "Basemap map";
    $("backLink").href = CFG.back || "../../index.html";
    const rows = $("rows");
    rows.innerHTML = "";
    const rt = document.createElement("span"); rt.className = "rows-total"; rt.textContent = fmt(m.rows_total);
    const note = document.createElement("span"); note.className = "rows-note";
    note.textContent = " rows" + (m.rows_note ? " — " + m.rows_note : "");
    rows.appendChild(rt); rows.appendChild(note);

    const prov = $("prov"); prov.innerHTML = "";
    const p = m.provenance || {};
    const chips = [];
    if (p.training_round) chips.push(["training round", p.training_round, ""]);
    if (p.eval_round) chips.push(["eval round", p.eval_round, ""]);
    if (m.round_id) chips.push(["round", m.round_id, ""]);
    if (p.evidence_status) {
      const st = String(p.evidence_status).toLowerCase();
      const cls = /accept/.test(st) ? "ok" : /reject|fail/.test(st) ? "bad" : "warn";
      chips.push(["evidence", p.evidence_status, cls]);
    }
    if (m.metrics && m.metrics.anchors && m.metrics.anchors.summary) {
      for (const [k, v] of Object.entries(m.metrics.anchors.summary)) chips.push([k, v, ""]);
    }
    for (const [k, v, cls] of chips) {
      const c = document.createElement("span"); c.className = "chip" + (cls ? " " + cls : "");
      const b = document.createElement("b"); b.textContent = String(v);
      c.appendChild(document.createTextNode(k + " ")); c.appendChild(b);
      prov.appendChild(c);
    }
  }

  function renderPanel() {
    const panel = $("panel");
    panel.innerHTML = "";
    if (S.mode === "map") renderLayersPanel(panel);
    else renderMetricsPanel(panel);
  }

  function groupLabel(g) { return g || "corpus & language subsets"; }

  function renderLayersPanel(panel) {
    const sec = document.createElement("div"); sec.className = "section";
    const h = document.createElement("h2"); h.textContent = "Layers"; sec.appendChild(h);

    const baseRow = document.createElement("label"); baseRow.className = "row";
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = true; cb.disabled = true;
    baseRow.appendChild(cb); baseRow.appendChild(document.createTextNode(" Base density (always on)"));
    sec.appendChild(baseRow);

    const subs = (S.manifest.layers || []).filter((l) => l.key !== S.baseLayer.key);
    if (subs.length) {
      const lab = document.createElement("div"); lab.className = "hint"; lab.textContent = "Accent overlay (one at a time):";
      sec.appendChild(lab);
      const sel = document.createElement("select");
      const none = document.createElement("option"); none.value = ""; none.textContent = "None"; sel.appendChild(none);
      const groups = new Map();
      for (const l of subs) { const g = groupLabel(l.group); if (!groups.has(g)) groups.set(g, []); groups.get(g).push(l); }
      for (const [g, arr] of groups) {
        const og = document.createElement("optgroup"); og.label = g;
        for (const l of arr) {
          const o = document.createElement("option"); o.value = l.key;
          o.textContent = `${l.label} (${fmt(l.rows)}${l.kind === "points" ? " pts" : ""})`;
          og.appendChild(o);
        }
        sel.appendChild(og);
      }
      sel.value = S.overlay ? S.overlay.key : "";
      sel.onchange = () => {
        S.overlay = sel.value ? subs.find((l) => l.key === sel.value) : null;
        requestDraw();
      };
      sec.appendChild(sel);
      const h2 = document.createElement("div"); h2.className = "hint";
      h2.textContent = "Grid subsets recolor by count on the same ramp; point subsets draw as accent markers.";
      sec.appendChild(h2);
    } else {
      const em = document.createElement("div"); em.className = "hint"; em.textContent = "No subset layers in this map.";
      sec.appendChild(em);
    }
    panel.appendChild(sec);

    const help = document.createElement("div"); help.className = "section hint";
    help.innerHTML = "Scroll / +− to zoom, drag to pan, 0 to reset. Hover a bin for row count and text samples.";
    panel.appendChild(help);
  }

  function renderMetricsPanel(panel) {
    const m = S.manifest.metrics || {};
    // sub-mode selector
    const seg = document.createElement("div"); seg.className = "seg";
    const bA = document.createElement("button"); bA.textContent = "Anchors";
    const bQ = document.createElement("button"); bQ.textContent = "Held-out queries";
    bA.className = S.metricMode === "anchors" ? "active" : "";
    bQ.className = S.metricMode === "queries" ? "active" : "";
    const hasAnchors = !!(m.anchors);
    const hasProbes = !!(m.probes && m.probes.length);
    if (!hasAnchors) bA.disabled = true;
    if (!hasProbes) bQ.disabled = true;
    bA.onclick = () => { S.metricMode = "anchors"; loadAnchors(); renderPanel(); requestDraw(); };
    bQ.onclick = () => { S.metricMode = "queries"; setupQueries(); renderPanel(); requestDraw(); };
    seg.appendChild(bA); seg.appendChild(bQ); panel.appendChild(seg);

    if (S.metricMode === "anchors") {
      const sec = document.createElement("div"); sec.className = "section";
      const h = document.createElement("h2"); h.textContent = "Anchors"; sec.appendChild(h);
      if (!hasAnchors) { const e = document.createElement("div"); e.className = "hint"; e.textContent = "No anchor metrics for this map."; sec.appendChild(e); panel.appendChild(sec); return; }
      const d = document.createElement("div"); d.className = "hint";
      const label = m.anchors.score || "score";
      d.textContent = `${m.anchors.count ? fmt(m.anchors.count) + " anchors" : "anchors"} colored by ${label}. Hover a point for its value.`;
      sec.appendChild(d);
      if (m.anchors.summary) {
        for (const [k, v] of Object.entries(m.anchors.summary)) {
          const row = document.createElement("div"); row.className = "hint";
          row.textContent = `${k}: ${v}`; sec.appendChild(row);
        }
      }
      panel.appendChild(sec);
    } else {
      renderQueryPanel(panel, m);
    }
  }

  async function setupQueries() {
    if (!S.queriesDoc) { try { await loadQueriesDoc(); } catch (e) { console.warn(e); } renderPanel(); }
  }

  function renderQueryPanel(panel, m) {
    const sec = document.createElement("div"); sec.className = "section";
    const h = document.createElement("h2"); h.textContent = "Held-out probes"; sec.appendChild(h);
    const doc = S.queriesDoc;
    if (!doc) { const e = document.createElement("div"); e.className = "hint"; e.textContent = "Loading probes…"; sec.appendChild(e); panel.appendChild(sec); return; }
    const probes = doc.probes || [];
    if (!probes.length) { const e = document.createElement("div"); e.className = "hint"; e.textContent = "No query probes for this map."; sec.appendChild(e); panel.appendChild(sec); return; }

    const plist = document.createElement("div"); plist.className = "plist";
    for (const p of probes) {
      const b = document.createElement("button");
      const nm = document.createElement("span"); nm.textContent = p.label || p.key;
      const r = document.createElement("span"); r.className = "r";
      if (p.recall50 != null) r.textContent = "R@50 " + Number(p.recall50).toFixed(3);
      b.appendChild(nm); b.appendChild(r);
      if (S.probe && S.probe.key === p.key) b.className = "active";
      b.onclick = () => { S.probe = p; S.query = null; fitProbe(p); renderPanel(); requestDraw(); };
      plist.appendChild(b);
    }
    sec.appendChild(plist);

    // legend for hit/miss shapes (shape channel — not color alone)
    const lg = document.createElement("div");
    lg.innerHTML =
      `<div class="legend-inline"><span class="swatch circle" style="background:var(--status-good)"></span> hit (in retrieved top-50)</div>` +
      `<div class="legend-inline"><span class="swatch diamond" style="background:var(--status-serious)"></span> miss</div>`;
    sec.appendChild(lg);
    panel.appendChild(sec);

    if (S.probe) {
      const qsec = document.createElement("div"); qsec.className = "section";
      const qh = document.createElement("h2"); qh.textContent = "Queries"; qsec.appendChild(qh);
      const qhint = document.createElement("div"); qhint.className = "hint";
      qhint.textContent = "Click a query to trace its 10 true neighbors.";
      qsec.appendChild(qhint);
      const ql = document.createElement("div"); ql.className = "plist";
      const qs = (S.probe.queries || []).slice(0, 200);
      qs.forEach((q, i) => {
        const b = document.createElement("button");
        const nm = document.createElement("span");
        nm.textContent = q.text ? q.text.slice(0, 42) : `query ${i + 1}`;
        nm.style.overflow = "hidden"; nm.style.textOverflow = "ellipsis"; nm.style.whiteSpace = "nowrap";
        const r = document.createElement("span"); r.className = "r"; r.textContent = (q.recall != null ? q.recall.toFixed(2) : "");
        b.appendChild(nm); b.appendChild(r);
        if (S.query === q) b.className = "active";
        b.onclick = () => { S.query = q; fitQuery(q); renderPanel(); requestDraw(); };
        ql.appendChild(b);
      });
      qsec.appendChild(ql);
      panel.appendChild(qsec);
    }

    if (S.query) panel.appendChild(queryCard(S.query));
  }

  function queryCard(q) {
    const card = document.createElement("div"); card.className = "qcard";
    const rc = document.createElement("div"); rc.className = "qrecall";
    rc.textContent = q.recall != null ? Math.round(q.recall * 100) + "%" : "—";
    const rs = document.createElement("small"); rs.textContent = " recall (hits in top-50)"; rc.appendChild(rs);
    card.appendChild(rc);
    if (q.text) { const t = document.createElement("div"); t.className = "qtext"; t.textContent = q.text; card.appendChild(t); }
    const nl = document.createElement("div"); nl.className = "nlist";
    q.hits.forEach((hit, i) => {
      const row = document.createElement("div"); row.className = "nrow " + (hit ? "hit" : "miss");
      const mk = document.createElement("span"); mk.className = "mark"; mk.textContent = hit ? "●" : "◇";
      const tx = document.createElement("span"); tx.className = "ntext";
      const nb = q.neighbors[i];
      tx.textContent = (q.neighbor_texts && q.neighbor_texts[i]) ? q.neighbor_texts[i] : `neighbor ${i + 1}  (x ${nb[0].toFixed(2)}, y ${nb[1].toFixed(2)})`;
      row.appendChild(mk); row.appendChild(tx); nl.appendChild(row);
    });
    card.appendChild(nl);
    return card;
  }

  function fitProbe(p) {
    const qs = p.queries || []; if (!qs.length) return;
    let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
    for (const q of qs) {
      x0=Math.min(x0,q.xy[0]); x1=Math.max(x1,q.xy[0]); y0=Math.min(y0,q.xy[1]); y1=Math.max(y1,q.xy[1]);
      for (const nb of q.neighbors) { x0=Math.min(x0,nb[0]); x1=Math.max(x1,nb[0]); y0=Math.min(y0,nb[1]); y1=Math.max(y1,nb[1]); }
    }
    padView(x0,x1,y0,y1,0.15);
  }
  function fitQuery(q) {
    let x0=q.xy[0],x1=q.xy[0],y0=q.xy[1],y1=q.xy[1];
    for (const nb of q.neighbors) { x0=Math.min(x0,nb[0]); x1=Math.max(x1,nb[0]); y0=Math.min(y0,nb[1]); y1=Math.max(y1,nb[1]); }
    padView(x0,x1,y0,y1,0.35);
  }
  function padView(x0,x1,y0,y1,frac) {
    const w = (x1-x0)||1, h=(y1-y0)||1, px=w*frac+0.001, py=h*frac+0.001;
    S.view = [x0-px, x1+px, y0-py, y1+py];
  }

  // ---- status overlay ------------------------------------------------------
  function setStatus(kind, msg, detail) {
    const s = $("status"); s.hidden = false; s.className = "status" + (kind === "error" ? " error" : "");
    s.querySelector(".spinner").style.display = kind === "error" ? "none" : "";
    $("statusMsg").textContent = msg || "";
    $("statusDetail").textContent = detail || "";
  }
  function hideStatus() { $("status").hidden = true; }
  function setError(msg, detail) { setStatus("error", msg, detail); console.error(msg, detail); }

  // ---- events --------------------------------------------------------------
  function setMode(mode) {
    S.mode = mode;
    for (const b of $("tabs").children) b.classList.toggle("active", b.dataset.tab === mode);
    hideTip();
    if (mode === "metrics") {
      const m = S.manifest.metrics || {};
      if (m.anchors) { S.metricMode = "anchors"; loadAnchors(); }
      else if (m.probes && m.probes.length) { S.metricMode = "queries"; setupQueries(); }
    }
    renderPanel(); requestDraw();
  }

  function bindEvents() {
    const c = S.canvas;
    let drag = null;
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const fx = (e.clientX - r.left) / r.width, fy = 1 - (e.clientY - r.top) / r.height;
      zoomAt(fx, fy, e.deltaY > 0 ? 1.18 : 0.84);
    }, { passive: false });
    c.addEventListener("mousedown", (e) => { drag = [e.clientX, e.clientY, S.view.slice()]; c.classList.add("dragging"); });
    addEventListener("mouseup", () => { drag = null; c.classList.remove("dragging"); });
    addEventListener("mousemove", (e) => {
      const r = c.getBoundingClientRect();
      if (drag) {
        const dx = (e.clientX - drag[0]) * (drag[2][1] - drag[2][0]) / r.width;
        const dy = (e.clientY - drag[1]) * (drag[2][3] - drag[2][2]) / r.height;
        S.view = [drag[2][0] - dx, drag[2][1] - dx, drag[2][2] + dy, drag[2][3] + dy];
        requestDraw(); return;
      }
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) { hideTip(); return; }
      const lx = e.clientX - r.left, ly = e.clientY - r.top;
      if (S.mode === "map" || (S.mode === "metrics" && S.metricMode === "queries")) {
        if (S.mode === "map") {
          const L = S.manifest.sample_level, e2 = S.extent;
          const cx = Math.floor((dataX(lx) - e2.x0) / (e2.w / L)), cy = Math.floor((dataY(ly) - e2.y0) / (e2.h / L));
          S.hoverCell = (cx>=0&&cy>=0&&cx<L&&cy<L) ? cy*L+cx : null;
          hoverMap(e.clientX, e.clientY, lx, ly);
        } else hideTip();
      } else if (S.mode === "metrics" && S.metricMode === "anchors") {
        hoverAnchors(e.clientX, e.clientY, lx, ly);
      }
    });
    c.addEventListener("mouseleave", hideTip);

    $("zoomIn").onclick = () => zoomAt(0.5, 0.5, 0.8);
    $("zoomOut").onclick = () => zoomAt(0.5, 0.5, 1.25);
    $("zoomReset").onclick = () => { resetView(); requestDraw(); };
    addEventListener("keydown", (e) => {
      if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
      if (e.key === "+" || e.key === "=") zoomAt(0.5, 0.5, 0.8);
      else if (e.key === "-" || e.key === "_") zoomAt(0.5, 0.5, 1.25);
      else if (e.key === "0") { resetView(); requestDraw(); }
    });
    for (const b of $("tabs").children) b.onclick = () => setMode(b.dataset.tab);
    $("themeToggle").onclick = () => {
      S.theme = isDark() ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", S.theme);
      requestDraw();
    };
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!S.theme) requestDraw(); });
    addEventListener("resize", requestDraw);
  }

  // ---- boot ----------------------------------------------------------------
  async function boot() {
    S.canvas = $("plot"); S.ctx = S.canvas.getContext("2d");
    try {
      setStatus("loading", "Loading map…");
      S.manifest = await (await fetch(url("manifest.json"))).json();
      if (S.manifest.schema && !/basemap-viewer-manifest/.test(S.manifest.schema))
        console.warn("unexpected manifest schema:", S.manifest.schema);
      const ex = S.manifest.extent;
      S.extent = { x0: ex[0], y0: ex[1], x1: ex[2], y1: ex[3], w: (ex[2]-ex[0])||1, h: (ex[3]-ex[1])||1 };
      S.baseLayer = (S.manifest.layers || []).find((l) => l.kind === "grid") || (S.manifest.layers || [])[0];
      if (!S.baseLayer) throw new Error("manifest has no layers");
      if (!S.baseLayer.levels || !S.baseLayer.levels.length) S.baseLayer.levels = S.manifest.levels || [256];

      renderHeader();
      resetView();
      bindEvents();

      // preload sample-level grid for hover counts + initial paint
      const g256 = await getGrid(S.baseLayer.key, S.manifest.sample_level || S.baseLayer.levels[0]);
      if (g256) { S.count256 = new Map(); for (let i = 0; i < g256.cells.length; i++) S.count256.set(g256.cells[i], g256.counts[i]); }

      renderPanel();
      hideStatus();
      requestDraw();
    } catch (e) {
      setError("Could not load this map", e.message);
    }
  }

  if (document.readyState === "loading") addEventListener("DOMContentLoaded", boot);
  else boot();
})();
