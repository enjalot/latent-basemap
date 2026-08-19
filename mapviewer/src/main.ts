/**
 * main.ts — wiring: pack discovery, range-mode detection, controls, side panel.
 *
 * Loading policy (PLAN6 owner decision, 2026-08-14):
 *   instant        manifest + visible density tiles
 *   on-interaction bin preview JSONs (hover)
 *   explicit ask   LOD points, deep-point index, projection models — labelled
 *                  with real sizes and driven by streamed progress bars
 *   free           per-click text lookups (KB-scale ranged reads)
 *
 * v2: the side panel is built once per pack into stable sections, and hover
 * only rebuilds the `bin` section, coalesced onto one rAF. v1 re-rendered the
 * whole panel (and redrew the whole map) on every pointer move, which is half
 * of what the flicker was.
 */

import "./style.css";
import type { Manifest, PackIndex, PackIndexEntry, RangeMode } from "./types";
import {
  RangeReader,
  fetchJSON,
  fmtBytes,
  onBytes,
  probeRangeMode,
  sessionBytes,
} from "./net";
import { DensityStore } from "./density";
import { PointStore } from "./points";
import { TextSidecar } from "./text";
import { BinSummaries } from "./bins";
import { MapView, type HoverInfo, type PickedPoint } from "./map";
import { loadManifest } from "./adapt";
import { CORPUS_FALLBACK, rampCss } from "./palette";
import {
  Projector,
  runtimeBytes,
  type Precision,
  type ProjectionResult,
} from "./projection";
import { stats as renderStats } from "./render";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const el = {
  boot: $("boot"),
  stage: $("stage"),
  canvas: $<HTMLCanvasElement>("map"),
  overlay: $<HTMLCanvasElement>("overlay"),
  controls: $("controls"),
  side: $("side"),
  status: $("status"),
  hud: $("hud"),
  select: $<HTMLSelectElement>("mapselect"),
};

// ---------------------------------------------------------------------------
// viewer config — including the per-source range mode we detected
// ---------------------------------------------------------------------------

const params = new URLSearchParams(location.search);

function normBase(s: string): string {
  return s.endsWith("/") ? s : s + "/";
}

const CONFIG = {
  /** where packs/index.json lives, relative to this page unless absolute */
  packsBase: normBase(params.get("packs") ?? "packs/"),
  /** data source -> active range mode; the caveat that shaped the design */
  sources: {} as Record<string, { rangeMode: RangeMode; probedAt: string }>,
  /** which base-layer renderer we got */
  backend: "",
};
(window as unknown as Record<string, unknown>).mapviewerConfig = CONFIG;

// ---------------------------------------------------------------------------
// small DOM helpers
// ---------------------------------------------------------------------------

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  ...kids: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else n.setAttribute(k, v);
  }
  for (const kid of kids) n.append(kid);
  return n;
}

function clear(n: HTMLElement) {
  while (n.firstChild) n.removeChild(n.firstChild);
}

function btn(label: string, fn: () => void): HTMLButtonElement {
  const b = h("button", {}, label) as HTMLButtonElement;
  b.addEventListener("click", fn);
  return b;
}

// ---------------------------------------------------------------------------
// session status line
// ---------------------------------------------------------------------------

onBytes((bytes, reqs) => {
  el.status.textContent = `${fmtBytes(bytes)} fetched · ${reqs} requests`;
});

// ---------------------------------------------------------------------------
// app state
// ---------------------------------------------------------------------------

interface App {
  entry: PackIndexEntry;
  base: string;
  manifest: Manifest;
  reader: RangeReader;
  density: DensityStore;
  points: PointStore;
  text: TextSidecar;
  bins: BinSummaries;
  view: MapView;
  projector: Projector | null;
}

let app: App | null = null;
let panelState: { hover: HoverInfo | null; point: PickedPoint | null } = {
  hover: null,
  point: null,
};
let pointFullText: { row: number; text: string; bytes: number } | null = null;
let projections: ProjectionResult[] = [];
let projectionNote = "";

/** Stable side-panel sections; hover only touches `bin`. */
const sec = {
  source: h("div", { class: "section", id: "sec-source" }),
  projection: h("div", { class: "section", id: "sec-projection" }),
  point: h("div", { class: "section", id: "sec-point" }),
  bin: h("div", { class: "section", id: "sec-bin" }),
};

function corpusColor(m: Manifest, code: number): string {
  const i = m.corpora.findIndex((c) => c.code === code);
  return m.corpora[i]?.color ?? CORPUS_FALLBACK[(i < 0 ? code : i) % CORPUS_FALLBACK.length];
}

function corpusLabel(m: Manifest, code: number): string {
  const c = m.corpora.find((x) => x.code === code);
  return c?.label ?? c?.name ?? `corpus ${code}`;
}

// ---------------------------------------------------------------------------
// panel scheduling — hover must never do more than one small DOM rebuild/frame
// ---------------------------------------------------------------------------

const dirty = new Set<keyof typeof sec>();
let panelRaf = 0;

function schedulePanel(...parts: (keyof typeof sec)[]) {
  for (const p of parts) dirty.add(p);
  if (panelRaf) return;
  panelRaf = requestAnimationFrame(() => {
    panelRaf = 0;
    const todo = [...dirty];
    dirty.clear();
    for (const p of todo) {
      if (p === "source") renderSource();
      else if (p === "projection") renderProjection();
      else if (p === "point") renderPoint();
      else renderBin();
    }
  });
}

// ---------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------

function buildControls(a: App) {
  clear(el.controls);
  askSyncs.length = 0;

  // -- corpora ---------------------------------------------------------------
  const g1 = h("div", { class: "group" }, h("h3", {}, "corpora"));
  for (const c of a.manifest.corpora) {
    const cb = h("input", { type: "checkbox" }) as HTMLInputElement;
    cb.checked = a.view.enabled.has(c.code);
    cb.addEventListener("change", () => {
      if (cb.checked) a.view.enabled.add(c.code);
      else a.view.enabled.delete(c.code);
      a.density.setEnabled([...a.view.enabled]);
      a.view.requestDraw();
      schedulePanel("bin");
    });
    g1.append(
      h(
        "label",
        { class: "check" },
        cb,
        h("span", {
          class: "swatch",
          style: `background:${corpusColor(a.manifest, c.code)}`,
        }),
        h("span", {}, corpusLabel(a.manifest, c.code)),
      ),
    );
  }
  el.controls.append(g1);

  // -- colour mode -----------------------------------------------------------
  const g2 = h("div", { class: "group" }, h("h3", {}, "colour"));
  for (const [mode, label] of [
    ["combined", "combined density (YlGnBu)"],
    ["dominant", "dominant corpus"],
  ] as const) {
    const rb = h("input", { type: "radio", name: "mode" }) as HTMLInputElement;
    rb.checked = a.density.mode === mode;
    rb.addEventListener("change", () => {
      if (!rb.checked) return;
      a.density.setMode(mode);
      a.view.requestDraw();
      renderRamp();
    });
    g2.append(h("label", { class: "check" }, rb, h("span", {}, label)));
  }
  const ramp = h("div", { class: "ramp", id: "ramp" });
  const rampLabels = h(
    "div",
    { class: "ramp-labels", id: "ramplabels" },
    h("span", {}, "low"),
    h("span", {}, "high"),
  );
  g2.append(ramp, rampLabels);
  el.controls.append(g2);

  // -- points (explicit asks) ------------------------------------------------
  const g3 = h("div", { class: "group" }, h("h3", {}, "points"));

  if (a.points.hasLod) {
    g3.append(
      askBlock({
        id: "ask-lod",
        title: "Enable point mode",
        why: `LOD sample: ${fmtBytes(a.points.lodBytes)} once, ${(
          a.manifest.points?.lod?.count ?? 0
        ).toLocaleString()} points, drawn past the zoom cutover.`,
        button: "Load LOD points",
        run: async (progress) => {
          await a.points.loadLod(progress);
          a.view.pointMode = "lod";
          a.view.requestDraw();
        },
        offButton: "Turn off points",
        off: () => {
          a.view.pointMode = "off";
          a.view.selected = null;
          a.view.releasePoints("all");
          pointFullText = null;
          a.view.requestDraw();
          schedulePanel("point");
        },
        isOn: () => a.view.pointMode === "lod",
      }),
    );
  }

  if (a.points.hasDeep) {
    const perViewport = a.manifest.points?.deep?.count
      ? Math.round(
          ((a.manifest.points.deep.count * (a.manifest.points.deep.record_bytes ?? 8)) /
            Math.max(1, 1 << (2 * a.points.deepZ))) *
            9,
        )
      : 0;
    g3.append(
      askBlock({
        id: "ask-deep",
        title: "Deep point mode",
        why:
          `Full ${(a.manifest.points?.deep?.count ?? 0).toLocaleString()} points, fetched ` +
          `per viewport. Index: ${fmtBytes(a.points.deepIndexBytes)} up front, ` +
          `~${fmtBytes(perViewport)} per view as you pan (${a.reader.mode}).`,
        button: "Load deep-point index",
        run: async (progress) => {
          await a.points.loadDeepIndex(progress);
          a.view.pointMode = "deep";
          refreshDeep();
          a.view.requestDraw();
        },
        offButton: "Turn off deep points",
        off: () => {
          a.view.pointMode = a.points.lod ? "lod" : "off";
          a.view.releasePoints("deep");
          a.view.requestDraw();
          schedulePanel("point");
        },
        isOn: () => a.view.pointMode === "deep",
      }),
    );
  }

  if (!a.points.hasLod && !a.points.hasDeep) {
    g3.append(h("div", { class: "empty" }, "this pack ships no point tiers"));
  }
  el.controls.append(g3);

  // -- view ------------------------------------------------------------------
  const g4 = h(
    "div",
    { class: "group" },
    h("h3", {}, "view"),
    h(
      "div",
      { class: "row" },
      btn("−", () => a.view.zoomBy(1 / 1.8)),
      btn("+", () => a.view.zoomBy(1.8)),
      btn("reset", () => a.view.fit()),
    ),
  );
  el.controls.append(g4);

  renderRamp();
}

interface AskSpec {
  id: string;
  title: string;
  why: string;
  button: string;
  run: (progress: (loaded: number, total: number) => void) => Promise<void>;
  offButton: string;
  off: () => void;
  isOn: () => boolean;
}

/** Every ask block registers a sync fn so one mode change updates them all. */
const askSyncs: (() => void)[] = [];

function syncUI() {
  for (const fn of askSyncs) fn();
  renderHud();
  renderRamp();
}

function askBlock(spec: AskSpec): HTMLElement {
  const box = h("div", { class: "ask", id: spec.id });
  const title = h("div", { class: "title" }, spec.title);
  const why = h("div", { class: "why" }, spec.why);
  const bar = h("div", { class: "bar" }, h("i", {}));
  const fill = bar.firstElementChild as HTMLElement;
  const b = h("button", {}, spec.button) as HTMLButtonElement;

  const sync = () => {
    const on = spec.isOn();
    box.classList.toggle("on", on);
    b.textContent = on ? spec.offButton : spec.button;
  };

  askSyncs.push(sync);

  b.addEventListener("click", async () => {
    if (spec.isOn()) {
      spec.off();
      syncUI();
      return;
    }
    b.disabled = true;
    b.textContent = "loading…";
    bar.classList.add("on");
    try {
      await spec.run((loaded, total) => {
        fill.style.width = total ? `${(loaded / total) * 100}%` : "50%";
      });
    } catch (err) {
      b.textContent = `failed: ${(err as Error).message}`;
      bar.classList.remove("on");
      b.disabled = false;
      return;
    }
    bar.classList.remove("on");
    fill.style.width = "0";
    b.disabled = false;
    syncUI();
    schedulePanel("point");
  });

  box.append(title, why, b, bar);
  sync();
  return box;
}

function renderRamp() {
  const a = app;
  if (!a) return;
  const ramp = document.getElementById("ramp");
  const labels = document.getElementById("ramplabels");
  if (!ramp || !labels) return;
  if (a.density.mode === "combined") {
    ramp.style.background = rampCss();
    clear(labels);
    labels.append(
      h("span", {}, "1"),
      h("span", {}, `${a.density.levelNorm(a.view.z) || "max"} / bin`),
    );
  } else {
    ramp.style.background = `linear-gradient(to right, ${a.manifest.corpora
      .map((c) => corpusColor(a.manifest, c.code))
      .join(", ")})`;
    clear(labels);
    labels.append(h("span", {}, "argmax corpus per bin"), h("span", {}, ""));
  }
}

// ---------------------------------------------------------------------------
// side panel — built once per pack, then updated section by section
// ---------------------------------------------------------------------------

function buildPanel(a: App) {
  clear(el.side);
  const m = a.manifest;
  el.side.append(h("h2", {}, m.title ?? m.map_id));
  el.side.append(
    h(
      "div",
      { class: "sub" },
      `${m.N.toLocaleString()} rows · z0–z${m.zoom.max} · ${m.tile_size ?? 256}² bins`,
    ),
  );
  el.side.append(sec.source, sec.projection, sec.point, sec.bin);
  renderSource();
  renderProjection();
  renderPoint();
  renderBin();
}

function renderSource() {
  const a = app;
  if (!a) return;
  clear(sec.source);
  const modePill =
    a.reader.mode === "http-range"
      ? h("span", { class: "pill ok" }, "HTTP Range")
      : a.reader.mode === "chunked"
        ? h("span", { class: "pill warn" }, "chunked parts")
        : h("span", { class: "pill warn" }, "no ranged reads");
  sec.source.append(
    h("h3", {}, "source"),
    h("div", { class: "kv" }, h("span", {}, "range mode"), modePill),
    h("div", { class: "kv" }, h("span", {}, "renderer"), h("b", {}, CONFIG.backend.split(" — ")[0])),
    h(
      "div",
      { class: "kv" },
      h("span", {}, "session"),
      h("b", {}, fmtBytes(sessionBytes().bytes)),
    ),
  );
  if (a.manifest.synthetic)
    sec.source.append(
      h("div", { class: "empty" }, "synthetic fixture pack — not real embeddings"),
    );
}

function renderPoint() {
  const a = app;
  if (!a) return;
  clear(sec.point);
  const p = panelState.point;
  if (!p) return;
  const m = a.manifest;
  sec.point.append(
    h("h3", {}, "selected point"),
    h("div", { class: "kv" }, h("span", {}, "row id"), h("b", {}, String(p.id))),
    h(
      "div",
      { class: "kv" },
      h("span", {}, "corpus"),
      h(
        "b",
        {},
        h("span", {
          class: "swatch",
          style: `display:inline-block;background:${corpusColor(m, p.corpus)}`,
        }),
        " " + corpusLabel(m, p.corpus),
      ),
    ),
    h("div", { class: "kv" }, h("span", {}, "tier"), h("b", {}, p.tier)),
  );
  if (pointFullText && pointFullText.row === p.id) {
    sec.point.append(h("div", { class: "fulltext" }, pointFullText.text));
    sec.point.append(
      h(
        "div",
        { class: "kv" },
        h("span", { class: "muted" }, "fetched"),
        h("b", {}, fmtBytes(pointFullText.bytes)),
      ),
    );
  } else if (a.text.available) {
    sec.point.append(h("div", { class: "empty" }, "fetching text…"));
  } else {
    sec.point.append(
      h("div", { class: "empty" }, "no text sidecar reachable from this source"),
    );
  }
}

function renderBin() {
  const a = app;
  if (!a) return;
  clear(sec.bin);
  const m = a.manifest;
  sec.bin.append(h("h3", {}, "bin"));
  const hv = panelState.hover;
  if (!hv) {
    sec.bin.append(h("div", { class: "empty" }, "hover the map to inspect a bin"));
    return;
  }
  sec.bin.append(
    h(
      "div",
      { class: "kv" },
      h("span", {}, "level / bin"),
      h("b", {}, `z${hv.z} · ${hv.gbx},${hv.gby}`),
    ),
    h(
      "div",
      { class: "kv" },
      h("span", {}, "rows in bin"),
      h("b", {}, hv.total.toLocaleString()),
    ),
  );

  if (hv.counts && hv.total > 0) {
    const comp = h("div", { class: "comp" });
    for (const c of [...hv.counts].sort((x, y) => y.count - x.count)) {
      if (!c.count) continue;
      const pct = (c.count / hv.total) * 100;
      comp.append(
        h(
          "div",
          { class: "lab" },
          h("span", { class: "swatch", style: `background:${corpusColor(m, c.code)}` }),
          h("span", {}, corpusLabel(m, c.code)),
          h("span", { class: "pct" }, `${pct.toFixed(0)}% · ${c.count}`),
        ),
        h(
          "div",
          { class: "track" },
          h("i", { style: `width:${pct}%;background:${corpusColor(m, c.code)}` }),
        ),
      );
    }
    sec.bin.append(comp);
  } else if (!hv.counts) {
    sec.bin.append(h("div", { class: "empty" }, "tile still loading…"));
  }

  const preview = a.bins.previewAt(hv.z, hv.gbx, hv.gby);
  const snips = preview?.snippets;
  const rows = preview?.rows;
  if (snips && snips.length) {
    sec.bin.append(
      h(
        "h3",
        { style: "margin-top:12px" },
        preview && preview.z !== hv.z
          ? `sampled snippets (z${preview.z} preview bin)`
          : "sampled snippets",
      ),
    );
    snips.forEach((s, i) => {
      const row = rows?.[i];
      const box = h(
        "div",
        { class: "snip" },
        h("span", { class: "meta" }, row !== undefined ? `row ${row}` : "sample"),
        h("span", {}, s),
      );
      if (row !== undefined && a.text.available) {
        box.append(
          btn("full text →", () => {
            panelState.point = { id: row, corpus: -1, u: 0, v: 0, tier: "lod" };
            schedulePanel("point");
            void loadText(row);
          }),
        );
      }
      sec.bin.append(box);
    });
  } else if (preview?.pending) {
    sec.bin.append(h("div", { class: "empty" }, `loading bin previews for z${hv.z}…`));
  } else {
    sec.bin.append(h("div", { class: "empty" }, "no snippets sampled for this bin"));
  }
}

async function loadText(row: number) {
  const a = app;
  if (!a || !a.text.available) return;
  try {
    pointFullText = await a.text.lookup(row);
  } catch (err) {
    pointFullText = { row, text: `lookup failed: ${(err as Error).message}`, bytes: 0 };
  }
  schedulePanel("point");
}

// ---------------------------------------------------------------------------
// projection panel — text -> MiniLM -> map head -> a dot on the overlay
// ---------------------------------------------------------------------------

const DEFAULT_TEXT = "Photosynthesis converts light energy into chemical energy in plants.";
let projectionPrecision: Precision = "int8";

function syncMarkers() {
  const a = app;
  if (!a) return;
  a.view.markers = projections.map((p, i) => ({
    u: p.u,
    v: p.v,
    label: p.text,
    active: i === projections.length - 1,
  }));
  a.view.requestOverlay();
}

function renderProjection() {
  const a = app;
  if (!a) return;
  clear(sec.projection);
  sec.projection.append(h("h3", {}, "project text onto this map"));

  const model = a.manifest.model;
  if (!model || !a.projector) {
    sec.projection.append(
      h("div", { class: "empty" }, "this pack ships no map head (model/map_head.onnx)"),
    );
    return;
  }
  const proj = a.projector;

  if (!proj.loaded) {
    const box = h("div", { class: "ask", id: "ask-projection" });
    const bar = h("div", { class: "bar" }, h("i", {}));
    const fill = bar.firstElementChild as HTMLElement;
    const size = runtimeBytes(projectionPrecision, proj.headBytes);
    const b = h(
      "button",
      {},
      `Load projection (~${fmtBytes(size)})`,
    ) as HTMLButtonElement;

    const sel = h("select", { id: "projection-precision" }) as HTMLSelectElement;
    for (const [v, label] of [
      ["int8", `int8 encoder — ${fmtBytes(runtimeBytes("int8", proj.headBytes))} (default)`],
      ["fp32", `fp32 encoder — ${fmtBytes(runtimeBytes("fp32", proj.headBytes))} (high precision)`],
    ] as const) {
      const o = h("option", { value: v }, label) as HTMLOptionElement;
      o.selected = v === projectionPrecision;
      sel.append(o);
    }
    sel.addEventListener("change", () => {
      projectionPrecision = sel.value as Precision;
      schedulePanel("projection");
    });

    const details = h(
      "details",
      { class: "detail" },
      h("summary", {}, "encoder precision"),
      sel,
      h(
        "div",
        { class: "why" },
        "int8 shifts the landing by ≤0.14% of the extent diagonal (~1.4 px at 1024²); " +
          "only fp32 reaches cosine > 0.999 against sentence-transformers.",
      ),
    );

    b.addEventListener("click", async () => {
      b.disabled = true;
      b.textContent = "loading…";
      bar.classList.add("on");
      try {
        await proj.load(projectionPrecision, (loaded, total) => {
          fill.style.width = total ? `${(loaded / total) * 100}%` : "50%";
          b.textContent = `loading… ${fmtBytes(loaded)}`;
        });
        projectionNote = `${proj.precision} encoder · ${fmtBytes(proj.bytesLoaded)} of weights`;
      } catch (err) {
        projectionNote = "";
        b.textContent = `failed: ${(err as Error).message}`;
        b.disabled = false;
        bar.classList.remove("on");
        return;
      }
      schedulePanel("projection");
    });

    box.append(
      h(
        "div",
        { class: "why" },
        "MiniLM + this pack's map head, run in WASM on this page. One-time download; nothing leaves the browser.",
      ),
      b,
      bar,
      details,
    );
    sec.projection.append(box);
    return;
  }

  // -- loaded: the actual projection UI --------------------------------------
  const ta = h("textarea", { id: "projection-text", rows: "3" }) as HTMLTextAreaElement;
  ta.value = DEFAULT_TEXT;
  const out = h("div", { class: "empty", id: "projection-out" }, projectionNote);
  const go = h("button", { id: "projection-run" }, "Project →") as HTMLButtonElement;

  go.addEventListener("click", () => void runProjection(ta.value, out, go));
  ta.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") void runProjection(ta.value, out, go);
  });

  sec.projection.append(
    ta,
    h(
      "div",
      { class: "row" },
      go,
      btn("clear", () => {
        projections = [];
        syncMarkers();
        schedulePanel("projection");
      }),
    ),
    out,
  );

  if (projections.length) {
    const list = h("div", { class: "markers" });
    projections.forEach((p, i) => {
      const row = h(
        "div",
        { class: i === projections.length - 1 ? "marker on" : "marker" },
        h("span", { class: "dot" }),
        h("span", { class: "txt" }, p.text),
        h("span", { class: "xy" }, `${p.x.toFixed(2)}, ${p.y.toFixed(2)}`),
      );
      row.addEventListener("click", () => {
        app?.view.flyTo(p.u, p.v);
      });
      list.append(row);
    });
    sec.projection.append(list);
  }
}

async function runProjection(text: string, out: HTMLElement, go: HTMLButtonElement) {
  const a = app;
  if (!a?.projector?.loaded || !text.trim()) return;
  go.disabled = true;
  out.textContent = "projecting…";
  try {
    const r = await a.projector.project(text.trim());
    projections.push(r);
    (window as unknown as Record<string, unknown>).__lastProjection = r;
    syncMarkers();
    // centre on the new dot only if it landed off-screen
    const [sx, sy] = a.view.screenAt(r.u, r.v);
    if (sx < 0 || sy < 0 || sx > a.view.width || sy > a.view.height)
      a.view.flyTo(r.u, r.v);
    projectionNote =
      `x=${r.x.toFixed(4)} y=${r.y.toFixed(4)} · u,v=${r.u.toFixed(5)},${r.v.toFixed(5)} ` +
      `(u16 ${r.qx},${r.qy}) · ${r.tokens} tokens · embed ${r.embedMs.toFixed(0)} ms · ` +
      `head ${r.headMs.toFixed(1)} ms`;
  } catch (err) {
    projectionNote = `projection failed: ${(err as Error).message}`;
  }
  go.disabled = false;
  schedulePanel("projection");
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

function renderHud() {
  const a = app;
  if (!a) return;
  clear(el.hud);
  el.hud.append(
    h("span", {}, `z${a.view.z}`),
    h("span", {}, "·"),
    h("span", {}, `${(a.view.scale / (a.manifest.tile_size ?? 256)).toFixed(1)}× tiles`),
    h("span", {}, "·"),
    h("span", {}, a.view.pointMode === "off" ? "bins" : `points:${a.view.pointMode}`),
  );
}

// ---------------------------------------------------------------------------
// deep-point refresh on view change
// ---------------------------------------------------------------------------

let deepTimer = 0;
function refreshDeep() {
  const a = app;
  if (!a || a.view.pointMode !== "deep" || !a.points.deepIndex) return;
  window.clearTimeout(deepTimer);
  deepTimer = window.setTimeout(() => {
    a.points.ensureDeep(a.view.visibleTileList(a.points.deepZ));
  }, 90);
}

/** HUD text changes on zoom; coalesce it so panning doesn't thrash the DOM. */
let hudRaf = 0;
function scheduleHud() {
  if (hudRaf) return;
  hudRaf = requestAnimationFrame(() => {
    hudRaf = 0;
    renderHud();
    renderRamp();
  });
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

function bootMessage(msg: string, isError = false) {
  el.boot.className = isError ? "boot error" : "boot";
  el.boot.textContent = msg;
}

/**
 * Fresh canvases per pack. A WebGL context is bound to its canvas for life, so
 * swapping the element is the clean way to hand the old context back.
 */
function freshCanvases() {
  const map = h("canvas", { id: "map", class: "layer" });
  const overlay = h("canvas", { id: "overlay", class: "layer" });
  el.canvas.replaceWith(map);
  el.overlay.replaceWith(overlay);
  el.canvas = map;
  el.overlay = overlay;
}

async function openPack(entry: PackIndexEntry) {
  app?.view.dispose();
  app?.density.dispose();
  app = null;
  pointFullText = null;
  projections = [];
  projectionNote = "";
  panelState = { hover: null, point: null };
  bootMessage(`loading ${entry.map_id}…`);

  const base = entry.url
    ? normBase(entry.url)
    : normBase(CONFIG.packsBase + (entry.path ?? entry.map_id));
  const manifest = await loadManifest(base);

  // Range-mode probe against a real binary in the pack. This is the caveat the
  // whole fallback exists for: python http.server answers 200 + full body — so
  // probe the SMALLEST binary available, never the deep-points blob.
  const probeTarget =
    manifest.points?.deep?.tile_index?.path ??
    manifest.points?.lod?.path ??
    manifest.text?.offsets ??
    "manifest.json";
  const mode = await probeRangeMode(base + probeTarget);
  const reader = new RangeReader(base, mode, manifest);
  CONFIG.sources[base] = { rangeMode: reader.mode, probedAt: new Date().toISOString() };

  freshCanvases();

  const density = new DensityStore(base, manifest);
  const points = new PointStore(manifest, reader);
  const text = new TextSidecar(manifest, reader);
  const bins = new BinSummaries(base, manifest);
  const view = new MapView(el.canvas, el.overlay, manifest, density, points);
  CONFIG.backend = view.backend;

  const projector = manifest.model
    ? new Projector({
        packBase: base,
        manifest,
        headPath: base + manifest.model.map_head,
        modelsJsonPath: base + manifest.model.models_json,
        headBytes: manifest.model.map_head_bytes ?? 47_239_092,
      })
    : null;

  const a: App = { entry, base, manifest, reader, density, points, text, bins, view, projector };
  app = a;

  density.onChange = () => view.requestDraw();
  points.onChange = () => view.requestDraw();
  bins.onChange = () => schedulePanel("bin");

  // Hover resolves the bin at the current density level and falls back up the
  // pyramid; snippets fall back independently (packs may only sample z0).
  view.previewLevelCap = Infinity;

  view.onHover = (hv) => {
    panelState.hover = view.pinned ?? hv;
    schedulePanel("bin");
  };
  view.onPick = (p) => {
    panelState.point = p;
    pointFullText = null;
    schedulePanel("point");
    if (p) void loadText(p.id);
  };
  view.onViewChange = () => {
    scheduleHud();
    refreshDeep();
  };

  buildControls(a);
  buildPanel(a);
  view.resize();
  view.fit();
  renderHud();
  el.boot.classList.add("hidden");
}

async function main() {
  let index: PackIndex;
  try {
    index = await fetchJSON<PackIndex>(CONFIG.packsBase + "index.json");
  } catch (err) {
    bootMessage(
      `No pack index at ${CONFIG.packsBase}index.json — ${(err as Error).message}. ` +
        `Build fixtures with \`npm run fixtures\`, or point the viewer at a pack ` +
        `directory with ?packs=<url>.`,
      true,
    );
    return;
  }
  if (!index.packs?.length) {
    bootMessage("pack index is empty", true);
    return;
  }

  clear(el.select);
  for (const p of index.packs) {
    el.select.append(
      h("option", { value: p.map_id }, `${p.title ?? p.map_id}${p.synthetic ? " (fixture)" : ""}`),
    );
  }
  const wanted = params.get("map");
  const initial = index.packs.find((p) => p.map_id === wanted) ?? index.packs[0];
  el.select.value = initial.map_id;
  el.select.addEventListener("change", () => {
    const p = index.packs.find((x) => x.map_id === el.select.value);
    if (p) void openPack(p).catch((e) => bootMessage(String(e), true));
  });

  await openPack(initial);
}

const ro = new ResizeObserver(() => {
  app?.view.resize();
});
ro.observe(el.stage);
window.addEventListener("resize", () => app?.view.resize());

// ---------------------------------------------------------------------------
// verification surface (scripts/smoke.mjs)
// ---------------------------------------------------------------------------

(window as unknown as Record<string, unknown>).__mapviewer = {
  stats: renderStats,
  get backend() {
    return CONFIG.backend;
  },
  /** draw synchronously, then read the framebuffer back in the same task */
  samplePixels: () => {
    if (!app) return 0;
    app.view.loop.flush();
    return app.view.base.sampleDistinctColours();
  },
  mapId: () => app?.manifest.map_id ?? "",
  camera: () => app?.view.camera,
  flyTo: (u: number, v: number, scale?: number) => app?.view.flyTo(u, v, scale),
  /** the density bin the viewer resolves at a world point (frame sanity check) */
  binAtWorld: (u: number, v: number) => {
    const a = app;
    if (!a) return null;
    const [sx, sy] = a.view.screenAt(u, v);
    return a.view.binAt(sx, sy, Infinity);
  },
  projection: {
    available: () => !!app?.projector,
    loaded: () => !!app?.projector?.loaded,
    setPrecision: (p: Precision) => {
      projectionPrecision = p;
      schedulePanel("projection");
    },
    load: (p: Precision = projectionPrecision) => app?.projector?.load(p),
    project: async (t: string) => {
      const r = await app?.projector?.project(t);
      if (r) {
        projections.push(r);
        syncMarkers();
        schedulePanel("projection");
      }
      return r;
    },
    markers: () => projections,
    extent: () => app?.manifest.extent,
    bytes: () => app?.projector?.bytesLoaded ?? 0,
    precision: () => app?.projector?.precision,
  },
};

main().catch((err) => bootMessage(String(err), true));
