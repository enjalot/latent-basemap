// Hover-cell math (v3 fix 1: hover alignment). Pure functions so the
// "which cell is under the cursor at the CURRENTLY RENDERED level" logic is
// unit-tested independently of canvas/DOM.
//
// The bug it fixes: the drawn grid uses the LOD-picked level (up to 1024, or a
// tiled fine level), but the old hover reported counts + a highlight for the
// coarse sample_level (256) cell. Counts and the drawn bins therefore did not
// line up. These helpers resolve the exact rendered-level cell, while text
// samples still come from the containing sample_level cell.

// The extent object shape used across the engine: { x0, y0, x1, y1, w, h }.

// Cell (cx, cy, idx) at `level` containing data-space point (dataX, dataY).
// Returns null when the point is outside the [0, level) grid on either axis.
// idx is the GLOBAL row-major cell index cy*level + cx (matches BIN1 files).
export function cellAt(dataX, dataY, extent, level) {
  if (!level || level < 1) return null;
  const cw = extent.w / level;
  const ch = extent.h / level;
  const cx = Math.floor((dataX - extent.x0) / cw);
  const cy = Math.floor((dataY - extent.y0) / ch);
  if (cx < 0 || cy < 0 || cx >= level || cy >= level) return null;
  return { cx, cy, level, idx: cy * level + cx };
}

// The sample_level (256) cell that CONTAINS a finer rendered-level cell — used
// to fetch text samples for "sample texts from this area". Works whether the
// rendered level is finer, equal, or (defensively) coarser than sampleLevel.
export function containingSampleCell(cx, cy, renderLevel, sampleLevel) {
  if (!renderLevel || !sampleLevel) return null;
  const ratio = renderLevel / sampleLevel;
  const scx = Math.floor(cx / ratio);
  const scy = Math.floor(cy / ratio);
  return { cx: scx, cy: scy, level: sampleLevel, idx: scy * sampleLevel + scx };
}

// Data-space bounding box [x0, x1, y0, y1] of a rendered-level cell, for drawing
// the highlight rect. The engine maps this to screen with sx/sy (which flips y).
export function cellDataBounds(cx, cy, extent, level) {
  const cw = extent.w / level;
  const ch = extent.h / level;
  const x0 = extent.x0 + cx * cw;
  const y0 = extent.y0 + cy * ch;
  return [x0, x0 + cw, y0, y0 + ch];
}
