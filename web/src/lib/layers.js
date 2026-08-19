// Layer classification + the legend-as-control state machine. Pure functions so
// the toggle rules (grid = radio, points = stack up to 2, context = free-form
// background) are unit-tested independently of React/canvas.
//
// Two manifest families share this code:
//  - grid maps (atlas / round-map): base "all" grid + grid subsets + probe points
//  - projection maps (point-only): layers like base-context (group "context"),
//    corpus (group "probe"), queries (group "probe-queries") — NO grid layers.

export const MAX_POINT_OVERLAYS = 2;
export const POINT_ACCENTS = ["a1", "a2"]; // distinct accent slots for stacked point layers

export function isGridLayer(layer) {
  return layer.kind === "grid";
}

// Context layers (group "context") are background scatter — rendered as muted
// gray small points and NOT counted against the 2-accent point stack.
export function isContextLayer(layer) {
  return String(layer.group || "").toLowerCase() === "context";
}

// The base density layer: the "all" grid, else the first grid layer. Point-only
// (projection) manifests have NO base layer — every layer is a legend row.
export function findBaseLayer(layers) {
  if (!layers || !layers.length) return null;
  return (
    layers.find((l) => l.key === "all" && isGridLayer(l)) ||
    layers.find((l) => isGridLayer(l)) ||
    null
  );
}

// Section bucket for the grouped legend list. Real group keys seen in manifests:
//   "context" (projection base-context), "probe" (projection corpus),
//   "probe-queries" (projection queries), "held-out" (atlas probe points),
//   long corpus/language dataset names (atlas grid subsets — bucketed by key).
// Unknown groups fall through to "Other layers"; nothing is ever dropped.
export function sectionOf(layer) {
  const k = String(layer.key || "").toLowerCase();
  const g = String(layer.group || "").toLowerCase();
  if (g === "context") return "Context";
  if (k.startsWith("corpus-")) return "Corpora";
  if (k.startsWith("lang-")) return "Languages";
  if (
    k.startsWith("probe-") ||
    g === "held-out" || g === "probe" || g === "probe-queries" ||
    g.includes("ood") || g.includes("held-out")
  )
    return "Held-out & OOD";
  return "Other layers";
}

const SECTION_ORDER = ["Corpora", "Languages", "Held-out & OOD", "Context", "Other layers"];

// Group the non-base layers into ordered [{ section, layers:[] }] for rendering.
// EVERY manifest layer except the base grid appears exactly once.
export function groupLayers(layers, baseKey) {
  const subs = (layers || []).filter((l) => baseKey == null || l.key !== baseKey);
  const byS = new Map();
  for (const l of subs) {
    const s = sectionOf(l);
    if (!byS.has(s)) byS.set(s, []);
    byS.get(s).push(l);
  }
  const ordered = [];
  for (const s of SECTION_ORDER) if (byS.has(s)) ordered.push({ section: s, layers: byS.get(s) });
  for (const [s, arr] of byS) if (!SECTION_ORDER.includes(s)) ordered.push({ section: s, layers: arr });
  return ordered;
}

// ---- overlay state -------------------------------------------------------
// state = { gridOverlay: <key|null>, points: [<key> up to 2], context: [<key>...] }
export const emptyOverlayState = () => ({ gridOverlay: null, points: [], context: [] });

// Initial state for a manifest. Grid maps start with base density only (all
// overlays off). Point-only maps start with EVERYTHING visible: context layers
// plus up to MAX_POINT_OVERLAYS accent point layers (corpus + queries).
export function defaultOverlayState(layers) {
  const s = emptyOverlayState();
  const hasGrid = (layers || []).some(isGridLayer);
  if (hasGrid) return s;
  for (const l of layers || []) {
    if (isContextLayer(l)) s.context.push(l.key);
    else if (s.points.length < MAX_POINT_OVERLAYS) s.points.push(l.key);
  }
  return s;
}

// Grid overlays are radio: toggling the active one clears it, any other replaces.
export function toggleGrid(state, key) {
  return { ...state, gridOverlay: state.gridOverlay === key ? null : key };
}

// Point overlays stack up to MAX_POINT_OVERLAYS. Toggling an active one removes
// it; adding when full is a no-op (user must deselect first).
export function togglePoint(state, key) {
  const has = state.points.includes(key);
  if (has) return { ...state, points: state.points.filter((k) => k !== key) };
  if (state.points.length >= MAX_POINT_OVERLAYS) return state;
  return { ...state, points: [...state.points, key] };
}

// Context layers toggle freely and never count against the accent stack.
export function toggleContext(state, key) {
  const has = state.context.includes(key);
  return {
    ...state,
    context: has ? state.context.filter((k) => k !== key) : [...state.context, key],
  };
}

// Toggle any layer by kind/group. Returns a new state object.
export function toggleLayer(state, layer) {
  if (isGridLayer(layer)) return toggleGrid(state, layer.key);
  if (isContextLayer(layer)) return toggleContext(state, layer.key);
  return togglePoint(state, layer.key);
}

export function isActive(state, layer) {
  if (isGridLayer(layer)) return state.gridOverlay === layer.key;
  if (isContextLayer(layer)) return state.context.includes(layer.key);
  return state.points.includes(layer.key);
}

// Accent slot ("a1"/"a2") for an active point layer: the manifest's accent hint
// wins when present, else assigned by stack order. Context layers get none.
export function accentFor(state, layer) {
  if (isContextLayer(layer)) return null;
  if (layer.accent === "a1" || layer.accent === "a2") return layer.accent;
  const i = state.points.indexOf(layer.key);
  return i < 0 ? null : POINT_ACCENTS[i] || null;
}

// Back-compat slot-order accent (used when only a key is known).
export function pointAccent(state, key) {
  const i = state.points.indexOf(key);
  return i < 0 ? null : POINT_ACCENTS[i] || null;
}

// Marker size per layer role: context = small background dots; query layers get
// slightly larger markers so they read above the corpus scatter.
export function markerSize(layer) {
  if (isContextLayer(layer)) return 1.7;
  const g = String(layer.group || "").toLowerCase();
  if (g === "probe-queries" || layer.key === "queries") return 3.4;
  return 2.6;
}
