import { useEffect, useState } from "react";

// Hash router: "#/" -> gallery; "#/map/<id>" -> viewer. Static-server safe
// (no history rewrites). Returns { name, mapId }.
export function useHashRoute() {
  const parse = () => {
    const h = (window.location.hash || "#/").replace(/^#/, "");
    const m = h.match(/^\/map\/(.+)$/);
    if (m) return { name: "map", mapId: decodeURIComponent(m[1]) };
    return { name: "gallery", mapId: null };
  };
  const [route, setRoute] = useState(parse);
  useEffect(() => {
    const on = () => setRoute(parse());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
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
