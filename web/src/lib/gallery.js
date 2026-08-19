// Gallery sort/filter (v3 fix 4). Pure functions over maps-index.json entries so
// the sort/filter behavior — including graceful handling of missing date/tags —
// is unit-tested without React. State lives in the hash query so links share.

export const KINDS = ["all", "atlas", "round-map", "projection-map"];

export const SORTS = [
  { key: "latest", label: "Latest" },
  { key: "ffr", label: "Best · FFR" },
  { key: "density_v2", label: "Best · density_v2" },
  { key: "rows", label: "Most rows" },
];

export const DEFAULT_QUERY = { sort: "latest", kind: "all", tag: "", q: "" };

// Tags associated with a map, for the dataset/probe filter. Prefer the explicit
// v3 "tags" list; fall back to probe keys/labels so older index entries (no
// "tags" field) still filter by e.g. "dadabase". Always returns a de-duped,
// lowercased array; never throws on missing fields.
export function mapTags(map) {
  const out = new Set();
  const push = (v) => {
    const s = String(v == null ? "" : v).trim().toLowerCase();
    if (s) out.add(s);
  };
  if (Array.isArray(map && map.tags)) for (const t of map.tags) push(t);
  else if (Array.isArray(map && map.probes)) for (const p of map.probes) push(p && (p.key || p.label));
  return [...out];
}

// Union of all tags across the maps, sorted — powers the dataset/probe select.
export function allTags(maps) {
  const s = new Set();
  for (const m of maps || []) for (const t of mapTags(m)) s.add(t);
  return [...s].sort();
}

// Millisecond timestamp for the "Latest" sort. Missing/invalid date -> null so
// those maps sink to the bottom (never crash Date parsing).
export function mapTime(map) {
  const d = map && map.date;
  if (!d) return null;
  const t = Date.parse(d);
  return Number.isFinite(t) ? t : null;
}

function metricVal(map, key) {
  const v = map && map.metrics && map.metrics[key];
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

// Filter then sort. Filters: kind chip (all = no filter), tag (exact match vs
// mapTags), free-text q (case-insensitive substring of title/map_id). Sorts:
// latest (date desc, undated last), ffr / density_v2 (metric desc, missing
// last), rows (rows_total desc). Stable: ties keep index order.
export function filterAndSort(maps, query) {
  const q = { ...DEFAULT_QUERY, ...(query || {}) };
  const needle = String(q.q || "").trim().toLowerCase();
  const tag = String(q.tag || "").trim().toLowerCase();

  let list = (maps || []).map((m, i) => ({ m, i }));

  if (q.kind && q.kind !== "all") list = list.filter(({ m }) => (m.kind || "") === q.kind);
  if (tag) list = list.filter(({ m }) => mapTags(m).includes(tag));
  if (needle)
    list = list.filter(({ m }) => {
      const hay = `${m.title || ""} ${m.map_id || ""}`.toLowerCase();
      return hay.includes(needle);
    });

  const cmp =
    q.sort === "rows"
      ? (a, b) => (Number(b.m.rows_total) || 0) - (Number(a.m.rows_total) || 0)
      : q.sort === "ffr" || q.sort === "density_v2"
        ? (a, b) => nullsLast(metricVal(a.m, q.sort), metricVal(b.m, q.sort))
        : /* latest */ (a, b) => nullsLast(mapTime(a.m), mapTime(b.m));

  list.sort((a, b) => {
    const c = cmp(a, b);
    return c !== 0 ? c : a.i - b.i; // stable tiebreak
  });
  return list.map(({ m }) => m);
}

// Descending compare that pushes null/undefined values to the end regardless of
// direction (undated maps / maps without the metric always sink).
function nullsLast(a, b) {
  const an = a == null, bn = b == null;
  if (an && bn) return 0;
  if (an) return 1;
  if (bn) return -1;
  return b - a;
}
