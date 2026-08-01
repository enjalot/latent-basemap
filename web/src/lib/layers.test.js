import { describe, it, expect } from "vitest";
import {
  findBaseLayer, sectionOf, groupLayers, emptyOverlayState, defaultOverlayState,
  toggleGrid, togglePoint, toggleLayer, isActive, pointAccent, accentFor,
  isContextLayer, markerSize, MAX_POINT_OVERLAYS,
} from "./layers.js";
import { computeTicks } from "./legend.js";
import { rowsPhrase } from "./format.js";

const gridLayer = (key, extra = {}) => ({ key, kind: "grid", rows: 1000, ...extra });
const pointLayer = (key, extra = {}) => ({ key, kind: "points", rows: 500, ...extra });

describe("findBaseLayer", () => {
  it("prefers the 'all' grid", () => {
    const base = findBaseLayer([gridLayer("corpus-x"), gridLayer("all"), pointLayer("probe-y")]);
    expect(base.key).toBe("all");
  });
  it("falls back to the first grid layer", () => {
    const base = findBaseLayer([pointLayer("probe-y"), gridLayer("corpus-x")]);
    expect(base.key).toBe("corpus-x");
  });
  it("returns null for point-only (projection) manifests — points are never base", () => {
    const base = findBaseLayer([
      pointLayer("base-context", { group: "context" }),
      pointLayer("corpus", { group: "probe" }),
      pointLayer("queries", { group: "probe-queries" }),
    ]);
    expect(base).toBe(null);
  });
});

describe("sectionOf", () => {
  it("buckets by key prefix and group", () => {
    expect(sectionOf(gridLayer("corpus-fineweb"))).toBe("Corpora");
    expect(sectionOf(gridLayer("lang-deu"))).toBe("Languages");
    expect(sectionOf(pointLayer("probe-pol_Latn", { group: "held-out" }))).toBe("Held-out & OOD");
  });
  it("maps the REAL projection-map group keys (regression: context/probe/probe-queries)", () => {
    // exact layers from round-0108-...-pol-latn-projection/data/manifest.json
    expect(sectionOf(pointLayer("base-context", { group: "context", accent: "a2", sampled_of: 24948663 }))).toBe("Context");
    expect(sectionOf(pointLayer("corpus", { group: "probe", accent: "a1", sampled_of: 49500 }))).toBe("Held-out & OOD");
    expect(sectionOf(pointLayer("queries", { group: "probe-queries", accent: "a2" }))).toBe("Held-out & OOD");
  });
  it("sends unknown groups to Other layers, never dropping them", () => {
    expect(sectionOf(pointLayer("mystery", { group: "someday-new-group" }))).toBe("Other layers");
    expect(sectionOf(pointLayer("bare"))).toBe("Other layers");
  });
});

describe("groupLayers", () => {
  it("groups and orders sections, excluding the base layer", () => {
    const layers = [
      gridLayer("all"), gridLayer("corpus-fineweb"), gridLayer("lang-deu"),
      pointLayer("probe-pol", { group: "held-out" }),
    ];
    const groups = groupLayers(layers, "all");
    expect(groups.map((g) => g.section)).toEqual(["Corpora", "Languages", "Held-out & OOD"]);
    expect(groups[0].layers[0].key).toBe("corpus-fineweb");
  });
  it("includes EVERY layer when there is no base (point-only manifest)", () => {
    const layers = [
      pointLayer("base-context", { group: "context" }),
      pointLayer("corpus", { group: "probe" }),
      pointLayer("queries", { group: "probe-queries" }),
    ];
    const groups = groupLayers(layers, null);
    const keys = groups.flatMap((g) => g.layers.map((l) => l.key));
    expect(keys.sort()).toEqual(["base-context", "corpus", "queries"]);
    expect(groups.map((g) => g.section)).toEqual(["Held-out & OOD", "Context"]);
  });
});

describe("defaultOverlayState (point-only projection maps)", () => {
  const projLayers = [
    pointLayer("base-context", { group: "context", accent: "a2", sampled_of: 24948663 }),
    pointLayer("corpus", { group: "probe", accent: "a1", sampled_of: 49500 }),
    pointLayer("queries", { group: "probe-queries", accent: "a2" }),
  ];
  it("defaults ALL point layers on: context + corpus + queries", () => {
    const s = defaultOverlayState(projLayers);
    expect(s.context).toEqual(["base-context"]);
    expect(s.points).toEqual(["corpus", "queries"]);
    for (const l of projLayers) expect(isActive(s, l)).toBe(true);
  });
  it("keeps grid maps starting with base density only", () => {
    const s = defaultOverlayState([gridLayer("all"), gridLayer("corpus-fineweb"), pointLayer("probe-x", { group: "held-out" })]);
    expect(s).toEqual(emptyOverlayState());
  });
  it("context layers do not count against the 2-accent point stack", () => {
    let s = defaultOverlayState(projLayers);
    expect(s.points.length).toBe(MAX_POINT_OVERLAYS); // corpus + queries, context separate
    // toggling context off/on never disturbs the accent stack
    s = toggleLayer(s, projLayers[0]);
    expect(s.context).toEqual([]);
    expect(s.points).toEqual(["corpus", "queries"]);
    s = toggleLayer(s, projLayers[0]);
    expect(s.context).toEqual(["base-context"]);
  });
  it("accentFor honors the manifest accent hint; context gets none", () => {
    const s = defaultOverlayState(projLayers);
    expect(accentFor(s, projLayers[1])).toBe("a1"); // corpus
    expect(accentFor(s, projLayers[2])).toBe("a2"); // queries
    expect(accentFor(s, projLayers[0])).toBe(null); // context
    expect(isContextLayer(projLayers[0])).toBe(true);
  });
  it("markerSize: context small, corpus medium, queries larger", () => {
    expect(markerSize(projLayers[0])).toBeLessThan(markerSize(projLayers[1]));
    expect(markerSize(projLayers[1])).toBeLessThan(markerSize(projLayers[2]));
  });
});

describe("grid overlay is radio", () => {
  it("selects, replaces, and clears", () => {
    let s = emptyOverlayState();
    s = toggleGrid(s, "corpus-a");
    expect(s.gridOverlay).toBe("corpus-a");
    s = toggleGrid(s, "corpus-b");
    expect(s.gridOverlay).toBe("corpus-b"); // replaced, not stacked
    s = toggleGrid(s, "corpus-b");
    expect(s.gridOverlay).toBe(null); // toggled off
  });
});

describe("point overlays stack up to 2", () => {
  it("adds, caps at MAX, and removes", () => {
    let s = emptyOverlayState();
    s = togglePoint(s, "p1");
    s = togglePoint(s, "p2");
    expect(s.points).toEqual(["p1", "p2"]);
    expect(pointAccent(s, "p1")).toBe("a1");
    expect(pointAccent(s, "p2")).toBe("a2");
    const before = s.points.slice();
    s = togglePoint(s, "p3"); // over cap -> no-op
    expect(s.points).toEqual(before);
    expect(MAX_POINT_OVERLAYS).toBe(2);
    s = togglePoint(s, "p1"); // remove
    expect(s.points).toEqual(["p2"]);
    expect(pointAccent(s, "p2")).toBe("a1"); // re-slotted to first accent
  });
});

describe("toggleLayer + isActive dispatch by kind", () => {
  it("routes grid vs points and reports active state", () => {
    let s = emptyOverlayState();
    const g = gridLayer("corpus-a"), p = pointLayer("probe-b");
    s = toggleLayer(s, g);
    expect(isActive(s, g)).toBe(true);
    s = toggleLayer(s, p);
    expect(isActive(s, p)).toBe(true);
    expect(isActive(s, g)).toBe(true); // grid + point coexist
  });
});

describe("computeTicks", () => {
  it("always ends at the cap and stays within [0,1]", () => {
    const ticks = computeTicks(12438, 220);
    const last = ticks[ticks.length - 1];
    expect(last.v).toBe(12438);
    expect(last.pos).toBe(1);
    for (const t of ticks) { expect(t.pos).toBeGreaterThanOrEqual(0); expect(t.pos).toBeLessThanOrEqual(1); }
  });
  it("drops decade ticks that would collide with the cap label on a narrow bar", () => {
    const ticks = computeTicks(12, 20); // narrow bar: the '1' tick collides with the cap label
    expect(ticks.length).toBe(1);
    expect(ticks[0].v).toBe(12);
  });
  it("keeps well-separated decades on a wide bar", () => {
    const ticks = computeTicks(1000000, 300).map((t) => t.v);
    expect(ticks).toContain(1);
    expect(ticks[ticks.length - 1]).toBe(1000000);
  });
});

describe("rowsPhrase (sampled_of)", () => {
  it("renders the sampled phrasing", () => {
    expect(rowsPhrase(30000, 24948663)).toBe("30,000 of 24,948,663 rows (sampled)");
  });
  it("renders a plain count without sampled_of", () => {
    expect(rowsPhrase(2878533)).toBe("2,878,533 rows");
  });
});
