// Deep-zoom tiled fine levels (v3 fix 3). Pure viewport->tile selection so the
// "fetch ONLY the tiles intersecting the viewport" logic is unit-tested without
// canvas/network.
//
// Contract (design Addendum v3): a grid layer MAY declare
//   "tiled_levels": [{"level": 2048, "split": 4}, {"level": 4096, "split": 4}]
// Tile file: grid-<layer>-<L>-<tx>_<ty>.bin (same BIN1 format; cell indices stay
// GLOBAL row-major for the full LxL grid). A cell belongs to tile
//   tx = cx // (L/split), ty = cy // (L/split).
// Everything degrades gracefully when tiled_levels is absent: no tiled level is
// ever selected, so no tile fetches (zero 404s) are attempted.

// Normalize a manifest's tiled_levels into a Map(level -> split). Missing /
// malformed entries are skipped, so older manifests just yield an empty map.
export function tiledLevelMap(tiledLevels) {
  const m = new Map();
  for (const t of tiledLevels || []) {
    const level = Number(t && t.level);
    const split = Number(t && t.split);
    if (Number.isFinite(level) && level > 0 && Number.isFinite(split) && split > 0) {
      m.set(level, split);
    }
  }
  return m;
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// Tile indices {tx, ty} whose cells intersect the given DATA-space viewport box
// at grid `level` split into `split` tiles per axis. viewBox = [x0, x1, y0, y1]
// (x0<x1, y0<y1). Returns a de-duped list clamped to [0, split). The cell index
// math mirrors the BIN1 contract: cellsPerTile = level / split.
export function tilesForViewport(viewBox, extent, level, split) {
  if (!split || split < 1) return [];
  const per = level / split; // cells per tile along one axis
  const cw = extent.w / level;
  const ch = extent.h / level;
  const [vx0, vx1, vy0, vy1] = viewBox;
  const cx0 = clamp(Math.floor((Math.min(vx0, vx1) - extent.x0) / cw), 0, level - 1);
  const cx1 = clamp(Math.floor((Math.max(vx0, vx1) - extent.x0) / cw), 0, level - 1);
  const cy0 = clamp(Math.floor((Math.min(vy0, vy1) - extent.y0) / ch), 0, level - 1);
  const cy1 = clamp(Math.floor((Math.max(vy0, vy1) - extent.y0) / ch), 0, level - 1);
  const tx0 = clamp(Math.floor(cx0 / per), 0, split - 1);
  const tx1 = clamp(Math.floor(cx1 / per), 0, split - 1);
  const ty0 = clamp(Math.floor(cy0 / per), 0, split - 1);
  const ty1 = clamp(Math.floor(cy1 / per), 0, split - 1);
  const out = [];
  for (let ty = ty0; ty <= ty1; ty++)
    for (let tx = tx0; tx <= tx1; tx++) out.push({ tx, ty });
  return out;
}

// The finest declared level whose cells still render at least MIN_CELL_PX wide,
// given a full candidate ordering. Kept out of the engine so LOD level choice
// (plain + tiled levels merged) is testable. `levels` may mix plain and tiled.
export function pickLevelFrom(levels, extentW, viewW, cssW, minCellPx) {
  const asc = levels.slice().sort((a, b) => a - b);
  let chosen = asc[0];
  for (const L of asc) {
    const cellPx = (cssW * (extentW / L)) / viewW;
    if (cellPx >= minCellPx) chosen = L;
  }
  return chosen;
}

export const tileCacheKey = (layerKey, level, tx, ty) => `${layerKey}-${level}-${tx}_${ty}`;
