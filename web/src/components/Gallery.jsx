import React, { useEffect, useState } from "react";
import { fmt, sig4 } from "../lib/format.js";
import ThemeToggle from "./ThemeToggle.jsx";

// Gallery: reads ../maps-index.json (relative to the deployed app dir) and shows
// a card per map. Designed empty/error states — maps-index.json may 404 until the
// python side lands it, in which case we say so and stay usable.
export default function Gallery({ theme }) {
  const [state, setState] = useState({ status: "loading", maps: [], error: null });

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await fetch("../maps-index.json");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const doc = await r.json();
        if (alive) setState({ status: "ok", maps: doc.maps || [], error: null });
      } catch (e) {
        if (alive) setState({ status: "empty", maps: [], error: e.message });
      }
    })();
    return () => { alive = false; };
  }, []);

  return (
    <div className="gallery">
      <header className="gh">
        <div>
          <h1>Basemap atlas</h1>
          <p className="sub">Interactive density maps of the latent-verse embedding spaces.</p>
        </div>
        <ThemeToggle theme={theme} />
      </header>

      {state.status === "loading" && <div className="g-note">Loading map index…</div>}

      {state.status === "empty" && (
        <div className="g-empty">
          <h2>No map index yet</h2>
          <p>
            <code>maps-index.json</code> is not published at the site root yet
            {state.error ? ` (${state.error})` : ""}. Once the registry publishes it,
            map cards appear here. You can still open a specific viewer directly:
          </p>
          <p className="mono">#/map/&lt;map_id&gt;</p>
        </div>
      )}

      {state.status === "ok" && state.maps.length === 0 && (
        <div className="g-empty"><h2>Index is empty</h2><p>No maps are registered yet.</p></div>
      )}

      {state.status === "ok" && state.maps.length > 0 && (
        <div className="cards">
          {state.maps.map((mp) => (
            <MapCard key={mp.map_id} mp={mp} />
          ))}
        </div>
      )}
    </div>
  );
}

function evidenceClass(s) {
  const t = String(s || "").toLowerCase();
  if (/accept/.test(t)) return "ok";
  if (/reject|fail/.test(t)) return "bad";
  return "warn";
}

function MapCard({ mp }) {
  const metrics = mp.metrics || {};
  const thumb = mp.thumbnail ? `../${mp.thumbnail}` : null;
  return (
    <a className="card" href={`#/map/${encodeURIComponent(mp.map_id)}`}>
      <div className="card-thumb">
        {thumb ? <img src={thumb} alt="" loading="lazy" /> : <div className="thumb-ph">density map</div>}
      </div>
      <div className="card-body">
        <div className="card-kind">{mp.kind || "map"}</div>
        <h3>{mp.title || mp.map_id}</h3>
        <div className="card-rows">
          <b>{fmt(mp.rows_total)}</b> rows
          {mp.rows_note ? <span className="muted"> — {mp.rows_note}</span> : null}
        </div>
        <div className="card-chips">
          {mp.evidence_status && (
            <span className={`chip ${evidenceClass(mp.evidence_status)}`}>
              {mp.evidence_status}
            </span>
          )}
          {metrics.ffr != null && <span className="chip">FFR <b>{sig4(metrics.ffr)}</b></span>}
          {metrics.density_v2 != null && <span className="chip">density <b>{sig4(metrics.density_v2)}</b></span>}
        </div>
      </div>
    </a>
  );
}
