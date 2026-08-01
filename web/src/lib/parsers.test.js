import { describe, it, expect } from "vitest";
import { parseGrid, parsePoints, parseAnchors, MAGIC } from "./parsers.js";

// Fabricate buffers that match the frozen little-endian byte formats.

function buildGrid(level, cells, counts) {
  const n = cells.length;
  const buf = new ArrayBuffer(16 + n * 4 + n * 4);
  const dv = new DataView(buf);
  dv.setUint32(0, MAGIC.GRID, true);
  dv.setUint32(4, level, true);
  dv.setUint32(8, n, true);
  dv.setUint32(12, 0, true); // reserved
  let o = 16;
  for (const c of cells) { dv.setUint32(o, c, true); o += 4; }
  for (const c of counts) { dv.setUint32(o, c, true); o += 4; }
  return buf;
}
function buildPoints(pairs) {
  const n = pairs.length;
  const buf = new ArrayBuffer(8 + n * 8);
  const dv = new DataView(buf);
  dv.setUint32(0, MAGIC.PTS, true);
  dv.setUint32(4, n, true);
  let o = 8;
  for (const [x, y] of pairs) { dv.setFloat32(o, x, true); dv.setFloat32(o + 4, y, true); o += 8; }
  return buf;
}
function buildAnchors(triples) {
  const n = triples.length;
  const buf = new ArrayBuffer(8 + n * 12);
  const dv = new DataView(buf);
  dv.setUint32(0, MAGIC.ANC, true);
  dv.setUint32(4, n, true);
  let o = 8;
  for (const [x, y, s] of triples) {
    dv.setFloat32(o, x, true); dv.setFloat32(o + 4, y, true); dv.setFloat32(o + 8, s, true); o += 12;
  }
  return buf;
}

describe("parseGrid", () => {
  it("reads level, cells, counts, and max", () => {
    const g = parseGrid(buildGrid(256, [0, 5, 130], [3, 100, 42]));
    expect(g.level).toBe(256);
    expect(Array.from(g.cells)).toEqual([0, 5, 130]);
    expect(Array.from(g.counts)).toEqual([3, 100, 42]);
    expect(g.max).toBe(100);
  });
  it("handles an empty grid (max defaults to 0)", () => {
    const g = parseGrid(buildGrid(64, [], []));
    expect(g.cells.length).toBe(0);
    expect(g.max).toBe(0);
  });
  it("rejects a bad magic", () => {
    const buf = new ArrayBuffer(16);
    new DataView(buf).setUint32(0, 0xdeadbeef, true);
    expect(() => parseGrid(buf)).toThrow(/magic/);
  });
});

describe("parsePoints", () => {
  it("reads flat xy pairs", () => {
    const xy = parsePoints(buildPoints([[1, 2], [-3.5, 4.25]]));
    expect(xy.length).toBe(4);
    expect(xy[0]).toBeCloseTo(1); expect(xy[1]).toBeCloseTo(2);
    expect(xy[2]).toBeCloseTo(-3.5); expect(xy[3]).toBeCloseTo(4.25);
  });
  it("rejects a bad magic", () => {
    const buf = new ArrayBuffer(8);
    new DataView(buf).setUint32(0, 1, true);
    expect(() => parsePoints(buf)).toThrow(/magic/);
  });
});

describe("parseAnchors", () => {
  it("splits triples into xy + score", () => {
    const a = parseAnchors(buildAnchors([[10, 20, 0.5], [30, 40, 0.9]]));
    expect(a.n).toBe(2);
    expect(a.xy[0]).toBeCloseTo(10); expect(a.xy[1]).toBeCloseTo(20);
    expect(a.xy[2]).toBeCloseTo(30); expect(a.xy[3]).toBeCloseTo(40);
    expect(a.score[0]).toBeCloseTo(0.5); expect(a.score[1]).toBeCloseTo(0.9);
  });
  it("rejects a bad magic", () => {
    const buf = new ArrayBuffer(8);
    new DataView(buf).setUint32(0, 0, true);
    expect(() => parseAnchors(buf)).toThrow(/magic/);
  });
});
