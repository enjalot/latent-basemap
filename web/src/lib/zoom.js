// Wheel/zoom feel (v3 fix 2). Pure so the "~4x gentler, notch ≈ 1.12x, smooth
// trackpad" tuning is testable and documented in one place.
//
// factor is applied to the view SPAN: factor > 1 zooms OUT (span grows),
// factor < 1 zooms IN. Cursor anchoring happens in the engine (zoomAt).

// Per-pixel exponential constant. exp(K_WHEEL * 100) ≈ 1.12 so one mouse notch
// (~100px of deltaY) is a gentle 1.12x — roughly 4x softer than the old fixed
// 1.18/0.84 step, and naturally smooth for high-frequency trackpad deltas.
export const K_WHEEL = Math.log(1.12) / 100; // ≈ 0.001133

// Moderate discrete step for +/- buttons and keyboard: ~1.6x per press (about
// two-thirds of a full LOD level, which is 2x).
export const BUTTON_IN = 1 / 1.6; // ≈ 0.625 (zoom in)
export const BUTTON_OUT = 1.6; // zoom out

// Double-click zooms in ~1.8x, centered on the cursor.
export const DBLCLICK_IN = 1 / 1.8; // ≈ 0.556

// Normalize a wheel event's deltaY to pixels across deltaMode variants
// (0=pixel, 1=line, 2=page), then map to a multiplicative zoom factor. Clamped
// per event so a single momentum spike can't teleport the view.
export function wheelFactor(deltaY, deltaMode, viewportPx) {
  let px = deltaY;
  if (deltaMode === 1) px = deltaY * 16; // lines -> ~16px each
  else if (deltaMode === 2) px = deltaY * (viewportPx || 800); // pages
  const factor = Math.exp(K_WHEEL * px);
  return Math.max(0.5, Math.min(2, factor));
}
