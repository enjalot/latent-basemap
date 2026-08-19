import { useEffect, useState } from "react";

// Hash router: "#/" -> gallery; "#/map/<id>" -> viewer. Static-server safe
// (no history rewrites). The gallery carries shareable sort/filter state in the
// hash query (#/?sort=&kind=&tag=&q=). Returns { name, mapId, query }.
export function useHashRoute() {
  const parse = () => {
    const h = (window.location.hash || "#/").replace(/^#/, "");
    const qi = h.indexOf("?");
    const path = qi >= 0 ? h.slice(0, qi) : h;
    const query = {};
    if (qi >= 0) {
      const sp = new URLSearchParams(h.slice(qi + 1));
      for (const [k, v] of sp) query[k] = v;
    }
    const m = path.match(/^\/map\/(.+)$/);
    if (m) return { name: "map", mapId: decodeURIComponent(m[1]), query };
    return { name: "gallery", mapId: null, query };
  };
  const [route, setRoute] = useState(parse);
  useEffect(() => {
    const on = () => setRoute(parse());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

// Serialize a gallery query object into a shareable hash ("#/" or "#/?..."),
// omitting default values so clean states stay clean.
export function galleryHash(query, defaults) {
  const sp = new URLSearchParams();
  for (const k of ["sort", "kind", "tag", "q"]) {
    const v = query[k];
    if (v != null && v !== "" && (!defaults || v !== defaults[k])) sp.set(k, v);
  }
  const s = sp.toString();
  return s ? `#/?${s}` : "#/";
}

// Theme: null=system, "light", "dark". Persists to localStorage and stamps
// data-theme on <html> (explicit override wins over prefers-color-scheme).
export function useTheme() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("bm-theme") || null; } catch { return null; }
  });
  useEffect(() => {
    const el = document.documentElement;
    if (theme) el.setAttribute("data-theme", theme);
    else el.removeAttribute("data-theme");
    try { theme ? localStorage.setItem("bm-theme", theme) : localStorage.removeItem("bm-theme"); } catch {}
  }, [theme]);
  const isDark = () => {
    if (theme === "dark") return true;
    if (theme === "light") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  };
  const toggle = () => setTheme(isDark() ? "light" : "dark");
  return { theme, isDark, toggle };
}
