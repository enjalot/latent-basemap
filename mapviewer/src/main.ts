/**
 * main.ts — wiring: pack discovery, range-mode detection, controls, side panel.
 *
 * Loading policy (PLAN6 owner decision, 2026-08-14):
 *   instant        manifest + visible density tiles
 *   on-interaction bin preview JSONs (hover)
 *   explicit ask   LOD points, deep-point index — labelled with real sizes and
 *                  driven by streamed progress bars
 *   free           per-click text lookups (KB-scale ranged reads)
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

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const el = {
  boot: $("boot"),
  stage: $("stage"),
  canvas: $<HTMLCanvasElement>("map"),
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
}

let app: App | null = null;
let panelState: { hover: HoverInfo | null; point: PickedPoint | null } = {
  hover: null,
  point: null,
};
let pointFullText: { row: number; text: string; bytes: number } | null = null;

function corpusColor(m: Manifest, code: number): string {
  const i = m.corpora.findIndex((c) => c.code === code);
  return m.corpora[i]?.color ?? CORPUS_FALLBACK[(i < 0 ? code : i) % CORPUS_FALLBACK.length];
}

function corpusLabel(m: Manifest, code: number): string {
  const c = m.corpora.find((x) => x.code === code);
  return c?.label ?? c?.name ?? `corpus ${code}`;
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
      renderPanel();
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
          pointFullText = null;
          a.view.requestDraw();
          renderPanel();
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
          a.view.requestDraw();
          renderPanel();
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

function btn(label: string, fn: () => void): HTMLButtonElement {
  const b = h("button", {}, label) as HTMLButtonElement;
  b.addEventListener("click", fn);
  return b;
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
    renderPanel();
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
    labels.innerHTML = "";
    labels.append(
      h("span", {}, "1"),
      h("span", {}, `${a.density.levelNorm(a.view.z) || "max"} / bin`),
    );
  } else {
    ramp.style.background = `linear-gradient(to right, ${a.manifest.corpora
      .map((c) => corpusColor(a.manifest, c.code))
      .join(", ")})`;
    labels.innerHTML = "";
    labels.append(h("span", {}, "argmax corpus per bin"), h("span", {}, ""));
  }
}

// ---------------------------------------------------------------------------
// side panel
// ---------------------------------------------------------------------------

function renderPanel() {
  const a = app;
  if (!a) return;
  clear(el.side);

  const m = a.manifest;
  el.side.append(h("h2", {}, m.title ?? m.map_id));
  el.side.append(
    h(
      "div",
      { class: "sub" },
      `${m.N.toLocaleString()} rows · z0–z${m.zoom.max} · ${
        m.tile_size ?? 256
      }² bins`,
    ),
  );

  // source / mode
  const modePill =
    a.reader.mode === "http-range"
      ? h("span", { class: "pill ok" }, "HTTP Range")
      : a.reader.mode === "chunked"
        ? h("span", { class: "pill warn" }, "chunked parts")
        : h("span", { class: "pill warn" }, "no ranged reads");
  el.side.append(
    h(
      "div",
      { class: "section" },
      h("h3", {}, "source"),
      h("div", { class: "kv" }, h("span", {}, "range mode"), modePill),
      h(
        "div",
        { class: "kv" },
        h("span", {}, "session"),
        h("b", {}, fmtBytes(sessionBytes().bytes)),
      ),
      m.synthetic
        ? h("div", { class: "empty" }, "synthetic fixture pack — not real embeddings")
        : h("span", {}),
    ),
  );

  // selected point
  if (panelState.point) {
    const p = panelState.point;
    const sec = h(
      "div",
      { class: "section" },
      h("h3", {}, "selected point"),
      h(
        "div",
        { class: "kv" },
        h("span", {}, "row id"),
        h("b", {}, String(p.id)),
      ),
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
      sec.append(h("div", { class: "fulltext" }, pointFullText.text));
      sec.append(
        h(
          "div",
          { class: "kv" },
          h("span", { class: "muted" }, "fetched"),
          h("b", {}, fmtBytes(pointFullText.bytes)),
        ),
      );
    } else if (a.text.available) {
      sec.append(h("div", { class: "empty" }, "fetching text…"));
    } else {
      sec.append(
        h("div", { class: "empty" }, "no text sidecar reachable from this source"),
      );
    }
    el.side.append(sec);
  }

  // hovered / pinned bin
  const hv = panelState.hover;
  const sec = h("div", { class: "section" }, h("h3", {}, "bin"));
  if (!hv) {
    sec.append(h("div", { class: "empty" }, "hover the map to inspect a bin"));
    el.side.append(sec);
    return;
  }
  sec.append(
    h(
      "div",
      { class: "kv" },
      h("span", {}, "level / bin"),
      h("b", {}, `z${hv.z} · ${hv.gbx},${hv.gby}`),
    ),
  );
  sec.append(
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
          h("span", {
            class: "swatch",
            style: `background:${corpusColor(m, c.code)}`,
          }),
          h("span", {}, corpusLabel(m, c.code)),
          h("span", { class: "pct" }, `${pct.toFixed(0)}% · ${c.count}`),
        ),
        h(
          "div",
          { class: "track" },
          h("i", {
            style: `width:${pct}%;background:${corpusColor(m, c.code)}`,
          }),
        ),
      );
    }
    sec.append(comp);
  } else if (!hv.counts) {
    sec.append(h("div", { class: "empty" }, "tile still loading…"));
  }

  const preview = a.bins.previewAt(hv.z, hv.gbx, hv.gby);
  const snips = preview?.snippets;
  const rows = preview?.rows;
  if (snips && snips.length) {
    sec.append(
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
            void loadText(row);
          }),
        );
      }
      sec.append(box);
    });
  } else if (preview?.pending) {
    sec.append(h("div", { class: "empty" }, `loading bin previews for z${hv.z}…`));
  } else {
    sec.append(h("div", { class: "empty" }, "no snippets sampled for this bin"));
  }

  el.side.append(sec);
}

async function loadText(row: number) {
  const a = app;
  if (!a || !a.text.available) return;
  try {
    pointFullText = await a.text.lookup(row);
  } catch (err) {
    pointFullText = { row, text: `lookup failed: ${(err as Error).message}`, bytes: 0 };
  }
  renderPanel();
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

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

function bootMessage(msg: string, isError = false) {
  el.boot.className = isError ? "boot error" : "boot";
  el.boot.textContent = msg;
}

async function openPack(entry: PackIndexEntry) {
  app?.density.dispose();
  app = null;
  pointFullText = null;
  panelState = { hover: null, point: null };
  bootMessage(`loading ${entry.map_id}…`);

  const base = normBase(CONFIG.packsBase + (entry.path ?? entry.map_id));
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

  const density = new DensityStore(base, manifest);
  const points = new PointStore(manifest, reader);
  const text = new TextSidecar(manifest, reader);
  const bins = new BinSummaries(base, manifest);
  const view = new MapView(el.canvas, manifest, density, points);

  const a: App = { entry, base, manifest, reader, density, points, text, bins, view };
  app = a;

  density.onChange = () => view.requestDraw();
  points.onChange = () => view.requestDraw();
  bins.onChange = () => renderPanel();

  // Hover resolves the bin at the current density level and falls back up the
  // pyramid; snippets fall back independently (packs may only sample z0).
  view.previewLevelCap = Infinity;

  view.onHover = (hv) => {
    panelState.hover = view.pinned ?? hv;
    renderPanel();
  };
  view.onPick = (p) => {
    panelState.point = p;
    pointFullText = null;
    renderPanel();
    if (p) void loadText(p.id);
  };
  view.onViewChange = () => {
    renderHud();
    renderRamp();
    refreshDeep();
  };

  buildControls(a);
  view.resize();
  view.fit();
  renderHud();
  renderPanel();
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

main().catch((err) => bootMessage(String(err), true));
