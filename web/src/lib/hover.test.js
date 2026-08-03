import { describe, it, expect } from "vitest";
import { cellAt, containingSampleCell, cellDataBounds } from "./hover.js";

// extent used across tests: 100x100 data box anchored at (10, 20).
const ext = { x0: 10, y0: 20, x1: 110, y1: 120, w: 100, h: 100 };

describe("cellAt (rendered-level hover cell)", () => {
  it("resolves the exact cell + GLOBAL row-major idx at the rendered level", () => {
    // level 256: cell width = 100/256. A point just inside cell (cx=5, cy=3).
    const cw = 100 / 256;
    const p = cellAt(ext.x0 + 5 * cw + cw / 2, ext.y0 + 3 * cw + cw / 2, ext, 256);
    expect(p).toEqual({ cx: 5, cy: 3, level: 256, idx: 3 * 256 + 5 });
  });
  it("gives a DIFFERENT, finer cell at a finer level for the same point (the bug)", () => {
    const x = ext.x0 + 40.3, y = ext.y0 + 55.7;
    const at256 = cellAt(x, y, ext, 256);
    const at1024 = cellAt(x, y, ext, 1024);
    // the finer level must not report the coarse cell's index
    expect(at1024.level).toBe(1024);
    expect(Math.floor(at1024.cx / 4)).toBe(at256.cx); // 1024 = 4*256 nesting
    expect(Math.floor(at1024.cy / 4)).toBe(at256.cy);
    expect(at1024.idx).not.toBe(at256.idx);
  });
  it("returns null outside the grid on either axis", () => {
    expect(cellAt(ext.x0 - 1, ext.y0 + 1, ext, 64)).toBe(null);
    expect(cellAt(ext.x0 + 1, ext.y1 + 1, ext, 64)).toBe(null);
    expect(cellAt(ext.x1 + 0.001, ext.y0 + 1, ext, 64)).toBe(null); // exactly at max edge is out
  });
  it("bins the top-right corner into the last cell, not out of range", () => {
    const eps = 1e-6;
    const p = cellAt(ext.x1 - eps, ext.y1 - eps, ext, 8);
    expect(p).toEqual({ cx: 7, cy: 7, level: 8, idx: 7 * 8 + 7 });
  });
});

describe("containingSampleCell (text samples come from the 256 cell)", () => {
  it("maps a fine rendered cell back to its containing sample_level cell", () => {
    // rendered level 1024, sample 256 -> ratio 4. cell (cx=40, cy=13) -> (10, 3)
    const sc = containingSampleCell(40, 13, 1024, 256);
    expect(sc).toEqual({ cx: 10, cy: 3, level: 256, idx: 3 * 256 + 10 });
  });
  it("is identity when rendered level equals sample level", () => {
    const sc = containingSampleCell(7, 9, 256, 256);
    expect(sc).toEqual({ cx: 7, cy: 9, level: 256, idx: 9 * 256 + 7 });
  });
  it("handles a coarser rendered level defensively (ratio < 1)", () => {
    // rendered 64, sample 256 -> ratio 0.25; cell (3,2) -> (12, 8)
    const sc = containingSampleCell(3, 2, 64, 256);
    expect(sc).toEqual({ cx: 12, cy: 8, level: 256, idx: 8 * 256 + 12 });
  });
});

describe("cellDataBounds (highlight rect)", () => {
  it("returns the data-space bbox of a rendered cell", () => {
    const [x0, x1, y0, y1] = cellDataBounds(5, 3, ext, 256);
    const cw = 100 / 256;
    expect(x0).toBeCloseTo(ext.x0 + 5 * cw);
    expect(x1).toBeCloseTo(ext.x0 + 6 * cw);
    expect(y0).toBeCloseTo(ext.y0 + 3 * cw);
    expect(y1).toBeCloseTo(ext.y0 + 4 * cw);
  });
});
