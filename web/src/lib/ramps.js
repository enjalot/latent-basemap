// Color ramps + samplers. Palette values taken VERBATIM from the validated
// experiments/viewer_assets/viewer.css / viewer.js (see the palette-validation
// header comment in viewer.css). Do not re-tune without re-running the dataviz
// validator.

// Density sequential ramp (blue). Light: low count near-surface (light) -> high
// (dark). Dark mode reverses so high reads bright against the dark surface.
export const RAMP_LIGHT = [
  "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
  "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
];
export const RAMP_DARK = RAMP_LIGHT.slice().reverse();

// Accent (orange) sequential ramp — an ACTIVE grid subset overlay so it reads as
// distinct from the muted blue base density.
export const RAMP_ACCENT_LIGHT = [
  "#fbe4d6", "#f7ccb0", "#f2ae86", "#ec8f5e", "#e5713b", "#d1541f", "#ad4216", "#823110",
];
export const RAMP_ACCENT_DARK = RAMP_ACCENT_LIGHT.slice().reverse();

// Diverging anchors ramp: blue (below median) <-> gray <-> red (above median).
export const ANCHOR_LIGHT = ["#256abf", "#6da7ec", "#f0efec", "#ec835a", "#d03b3b"];
export const ANCHOR_DARK = ["#3987e5", "#6da7ec", "#383835", "#ec835a", "#d03b3b"];

export function hexRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function rampSample(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const seg = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const f = seg - i;
  const a = hexRgb(stops[i]);
  const b = hexRgb(stops[i + 1]);
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(
    a[1] + (b[1] - a[1]) * f
  )},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}

export const rampGradient = (stops) => `linear-gradient(to right, ${stops.join(",")})`;

export const densityRamp = (dark) => (dark ? RAMP_DARK : RAMP_LIGHT);
export const accentRamp = (dark) => (dark ? RAMP_ACCENT_DARK : RAMP_ACCENT_LIGHT);
export const anchorRamp = (dark) => (dark ? ANCHOR_DARK : ANCHOR_LIGHT);
