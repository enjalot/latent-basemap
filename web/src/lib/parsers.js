// Binary parsers for the frozen basemap-viewer data contract. All little-endian.
// Ported byte-for-byte from experiments/viewer_assets/viewer.js so the React app
// reads the exact same files the python builder emits.
//
//   grid-<layer>-<L>.bin : u32 magic 0x42494E31, u32 level, u32 ncells, u32 rsvd,
//                          then u32[ncells] cellIdx (cy*L+cx, y in DATA space),
//                          then u32[ncells] counts.
//   points-<layer>.bin   : u32 magic 0x50545331, u32 npoints, then f32 x,y pairs.
//   metrics-anchors.bin  : u32 magic 0x414E4331, u32 count, then f32 (x,y,score) triples.

export const MAGIC = { GRID: 0x42494e31, PTS: 0x50545331, ANC: 0x414e4331 };

export function parseGrid(buf) {
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

export function parsePoints(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC.PTS) throw new Error("bad points magic");
  const n = dv.getUint32(4, true);
  return new Float32Array(buf, 8, n * 2);
}

export function parseAnchors(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC.ANC) throw new Error("bad anchors magic");
  const n = dv.getUint32(4, true);
  const f = new Float32Array(buf, 8, n * 3);
  const xy = new Float32Array(n * 2);
  const score = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    xy[2 * i] = f[3 * i];
    xy[2 * i + 1] = f[3 * i + 1];
    score[i] = f[3 * i + 2];
  }
  return { n, xy, score };
}
