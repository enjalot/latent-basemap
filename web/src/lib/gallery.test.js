import { describe, it, expect } from "vitest";
import { mapTags, allTags, mapTime, filterAndSort, DEFAULT_QUERY } from "./gallery.js";

const maps = [
  {
    map_id: "atlas-a", title: "Diverse Jina 25M", kind: "atlas",
    date: "2026-08-01", rows_total: 24948663,
    metrics: { ffr: 0.6386, density_v2: 0.1577 },
    tags: ["dadabase", "pol_Latn", "fineweb-heldout"],
  },
  {
    map_id: "round-b", title: "Balanced 150M", kind: "round-map",
    date: "2026-07-15", rows_total: 147221757,
    metrics: { ffr: 0.5011, density_v2: 0.1031 },
    // no tags -> falls back to probes
    probes: [{ key: "code" }, { key: "dadabase" }, { label: "TREC-COVID" }],
  },
  {
    map_id: "proj-c", title: "Dadabase jokes on r0019", kind: "projection-map",
    // no date, no density_v2 -> must sink in those sorts, never crash
    rows_total: 52066, metrics: { ffr: 0.3624 },
    probes: [{ key: "dadabase", label: "Dadabase jokes" }],
  },
];

describe("mapTags (graceful when tags/probes missing)", () => {
  it("prefers explicit tags, lowercased + deduped", () => {
    expect(mapTags(maps[0])).toEqual(["dadabase", "pol_latn", "fineweb-heldout"]);
  });
  it("falls back to probe key/label when tags absent", () => {
    expect(mapTags(maps[1]).sort()).toEqual(["code", "dadabase", "trec-covid"]);
  });
  it("returns [] for a map with neither", () => {
    expect(mapTags({ map_id: "x" })).toEqual([]);
  });
});

describe("allTags", () => {
  it("unions and sorts tags across all maps", () => {
    expect(allTags(maps)).toEqual(
      ["code", "dadabase", "fineweb-heldout", "pol_latn", "trec-covid"]
    );
  });
});

describe("mapTime", () => {
  it("parses ISO dates and returns null for missing/invalid", () => {
    expect(mapTime(maps[0])).toBe(Date.parse("2026-08-01"));
    expect(mapTime(maps[2])).toBe(null);
    expect(mapTime({ date: "not-a-date" })).toBe(null);
  });
});

describe("filterAndSort", () => {
  it("default = Latest: newest date first, undated maps sink to the bottom", () => {
    const out = filterAndSort(maps, DEFAULT_QUERY).map((m) => m.map_id);
    expect(out).toEqual(["atlas-a", "round-b", "proj-c"]); // proj-c undated -> last
  });
  it("Best by FFR sorts descending", () => {
    const out = filterAndSort(maps, { sort: "ffr" }).map((m) => m.map_id);
    expect(out).toEqual(["atlas-a", "round-b", "proj-c"]);
  });
  it("Best by density_v2 pushes maps missing the metric to the end", () => {
    const out = filterAndSort(maps, { sort: "density_v2" }).map((m) => m.map_id);
    expect(out[0]).toBe("atlas-a");
    expect(out[out.length - 1]).toBe("proj-c"); // no density_v2 -> last
  });
  it("Rows sorts by rows_total descending", () => {
    const out = filterAndSort(maps, { sort: "rows" }).map((m) => m.map_id);
    expect(out).toEqual(["round-b", "atlas-a", "proj-c"]);
  });
  it("kind chip filters, all = no filter", () => {
    expect(filterAndSort(maps, { kind: "projection-map" }).map((m) => m.map_id)).toEqual(["proj-c"]);
    expect(filterAndSort(maps, { kind: "all" }).length).toBe(3);
  });
  it("tag filter keeps only maps carrying that tag (dadabase across all kinds)", () => {
    const out = filterAndSort(maps, { tag: "dadabase" }).map((m) => m.map_id).sort();
    expect(out).toEqual(["atlas-a", "proj-c", "round-b"]);
    expect(filterAndSort(maps, { tag: "pol_latn" }).map((m) => m.map_id)).toEqual(["atlas-a"]);
  });
  it("free-text search matches title or map_id case-insensitively", () => {
    expect(filterAndSort(maps, { q: "balanced" }).map((m) => m.map_id)).toEqual(["round-b"]);
    expect(filterAndSort(maps, { q: "PROJ-C" }).map((m) => m.map_id)).toEqual(["proj-c"]);
  });
  it("combines filters (kind + tag + search) and returns [] when nothing matches", () => {
    expect(filterAndSort(maps, { kind: "atlas", tag: "code" })).toEqual([]);
  });
  it("never mutates the input array", () => {
    const before = maps.map((m) => m.map_id);
    filterAndSort(maps, { sort: "rows" });
    expect(maps.map((m) => m.map_id)).toEqual(before);
  });
});
