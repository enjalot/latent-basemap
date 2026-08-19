import React from "react";
import { fmt, sig4 } from "../lib/format.js";
import ThemeToggle from "./ThemeToggle.jsx";

// Header: back link, title, exact rows_total + rows_note, provenance chips
// (curated, sig-4 rounded), withheld note, and Map/Metrics tabs + theme toggle.
export default function HeaderBar({ manifest: m, mode, onMode, theme, hasMetrics }) {
  const p = m.provenance || {};
  const panel = p.panel || {};
  const chips = [];
  if (p.training_round) chips.push(["training round", p.training_round, ""]);
  if (p.eval_round) chips.push(["eval round", p.eval_round, ""]);
  if (m.round_id) chips.push(["round", m.round_id, ""]);
  if (p.evidence_status) {
    const st = String(p.evidence_status).toLowerCase();
    const cls = /accept/.test(st) ? "ok" : /reject|fail/.test(st) ? "bad" : "warn";
    chips.push(["evidence", p.evidence_status, cls]);
  }
  if (panel.ffr != null) chips.push(["FFR", sig4(panel.ffr), ""]);
  if (panel.density != null) chips.push(["density", sig4(panel.density), ""]);
  if (panel.purity_k1024 != null) chips.push(["purity@1024", sig4(panel.purity_k1024), ""]);

  const withheld = Array.isArray(m.skipped) && m.skipped.length ? m.skipped.join(" · ") : null;

  return (
    <div className="vh">
      <div>
        <a className="back" href="#/">← all maps</a>
        <h1>{m.title || "Basemap map"}</h1>
        <div className="rows">
          <span className="rows-total">{fmt(m.rows_total)}</span>
          <span className="rows-note"> rows{m.rows_note ? ` — ${m.rows_note}` : ""}</span>
        </div>
        <div className="chips">
          {chips.map(([k, v, cls], i) => (
            <span key={i} className={"chip" + (cls ? " " + cls : "")}>
              {k} <b>{String(v)}</b>
            </span>
          ))}
          {withheld && <div className="withheld">withheld: {withheld}</div>}
        </div>
      </div>
      <div className="vh-side">
        <div className="tabs">
          <button className={mode === "map" ? "active" : ""} onClick={() => onMode("map")}>Map</button>
          <button className={mode === "metrics" ? "active" : ""} disabled={!hasMetrics} onClick={() => onMode("metrics")}>Metrics</button>
        </div>
        <ThemeToggle theme={theme} />
      </div>
    </div>
  );
}
