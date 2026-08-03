import { describe, it, expect } from "vitest";
import { tiledLevelMap, tilesForViewport, pickLevelFrom, tileCacheKey } from "./tiles.js";

const ext = { x0: 0, y0: 0, x1: 100, y1: 100, w: 100, h: 100 };

describe("tiledLevelMap", () => {
  it("parses valid entries into level->split", () => {
    const m = tiledLevelMap([{ level: 2048, split: 4 }, { level: 4096, split: 4 }]);
    expect(m.get(2048)).toBe(4);
    expect(m.get(4096)).toBe(4);
  });
  it("degrades gracefully: undefined / empty / malformed -> empty map (no fine levels)", () => {
    expect(tiledLevelMap(undefined).size).toBe(0);
    expect(tiledLevelMap([]).size).toBe(0);
    const m = tiledLevelMap([{ level: 0, split: 4 }, { level: 2048 }, { split: 4 }, null]);
    expect(m.size).toBe(0);
  });
});

describe("tilesForViewport", () => {
  it("selects ONLY the tiles intersecting the viewport box", () => {
    // level 2048 split 4 -> 4x4 tiles, each covers 25 data units (100/4).
    // viewport covering data x[10,40], y[10,15] spans tile columns 0..1, row 0.
    const tiles = tilesForViewport([10, 40, 10, 15], ext, 2048, 4);
    const keys = tiles.map((t) => `${t.tx}_${t.ty}`).sort();
    expect(keys).toEqual(["0_0", "1_0"]);
  });
  it("a tight viewport inside one tile fetches exactly one tile", () => {
    const tiles = tilesForViewport([60, 70, 60, 70], ext, 2048, 4); // tile (2,2)
    expect(tiles).toEqual([{ tx: 2, ty: 2 }]);
  });
  it("clamps an over-wide viewport to the full tile grid", () => {
    const tiles = tilesForViewport([-50, 500, -50, 500], ext, 4096, 4);
    expect(tiles.length).toBe(16); // all 4x4 tiles, none out of range
    expect(tiles.every((t) => t.tx >= 0 && t.tx < 4 && t.ty >= 0 && t.ty < 4)).toBe(true);
  });
  it("uses GLOBAL cell math: tx = cx // (L/split)", () => {
    // cell width at L=2048 is 100/2048; per-tile = 2048/4 = 512 cells = 25 data.
    // a point at data x=30 -> cx=floor(30/(100/2048))=614 -> tx=floor(614/512)=1
    const tiles = tilesForViewport([30, 30.01, 30, 30.01], ext, 2048, 4);
    expect(tiles[0].tx).toBe(1);
  });
});

describe("pickLevelFrom (LOD across plain + tiled levels)", () => {
  const levels = [64, 128, 256, 512, 1024, 2048, 4096];
  it("stays <= plain max when zoomed out (no fine level chosen)", () => {
    // whole extent in view: cells must be >= 7px. cssW=800, viewW=100.
    const L = pickLevelFrom(levels, ext.w, 100, 800, 7);
    expect(L).toBeLessThanOrEqual(1024);
  });
  it("selects a fine tiled level (>1024) when zoomed deep", () => {
    // viewW very small (0.5 data units across 800px) -> tiny cells need fine grid
    const L = pickLevelFrom(levels, ext.w, 0.5, 800, 7);
    expect(L).toBeGreaterThan(1024);
  });
  it("never exceeds the finest level offered even at extreme zoom", () => {
    const L = pickLevelFrom(levels, ext.w, 0.001, 800, 7);
    expect(L).toBe(4096);
  });
  it("without tiled levels offered, caps at the plain max (graceful, zero fine fetches)", () => {
    const L = pickLevelFrom([64, 128, 256, 512, 1024], ext.w, 0.001, 800, 7);
    expect(L).toBe(1024);
  });
});

describe("tileCacheKey", () => {
  it("matches the grid-<layer>-<L>-<tx>_<ty> file naming", () => {
    expect(tileCacheKey("all", 2048, 1, 3)).toBe("all-2048-1_3");
  });
});
