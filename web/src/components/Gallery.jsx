import React, { useEffect, useMemo, useState } from "react";
import { fmt, sig4 } from "../lib/format.js";
import { galleryHash } from "../hooks.js";
import {
  KINDS, SORTS, DEFAULT_QUERY, allTags, filterAndSort, mapTags,
} from "../lib/gallery.js";
import ThemeToggle from "./ThemeToggle.jsx";

// Gallery: reads ../maps-index.json (relative to the deployed app dir) and shows
// a card per map. Sort/filter controls (v3) drive off the hash query so links are
// shareable. Designed empty/error states — maps-index.json may 404 until the
// python side lands it, in which case we say so and stay usable.
export default function Gallery({ theme, query }) {
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

  // Merge the hash query over defaults so missing params fall back sensibly.
  const q = { ...DEFAULT_QUERY, ...query };
  const setQuery = (patch) => {
    window.location.hash = galleryHash({ ...q, ...patch }, DEFAULT_QUERY);
  };

  const tags = useMemo(() => allTags(state.maps), [state.maps]);
  const shown = useMemo(() => filterAndSort(state.maps, q), [state.maps, q.sort, q.kind, q.tag, q.q]);

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
        <>
          <Controls q={q} tags={tags} setQuery={setQuery} shown={shown.length} total={state.maps.length} />
          {shown.length === 0 ? (
            <div className="g-empty"><h2>No maps match</h2><p>Adjust the filters or clear the search.</p></div>
          ) : (
            <div className="cards">
              {shown.map((mp) => <MapCard key={mp.map_id} mp={mp} />)}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Controls({ q, tags, setQuery, shown, total }) {
  return (
    <div className="g-controls">
      <div className="gc-row">
        <label className="gc-field">
          <span>Sort</span>
          <select value={q.sort} onChange={(e) => setQuery({ sort: e.target.value })}>
            {SORTS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
          </select>
        </label>

        <div className="gc-chips" role="group" aria-label="Filter by kind">
          {KINDS.map((k) => (
            <button
              key={k}
              type="button"
              className={"gc-chip" + (q.kind === k ? " on" : "")}
              aria-pressed={q.kind === k}
              onClick={() => setQuery({ kind: k })}
            >
              {k === "all" ? "All kinds" : k}
            </button>
          ))}
        </div>

        {tags.length > 0 && (
          <label className="gc-field">
            <span>Dataset / probe</span>
            <select value={q.tag} onChange={(e) => setQuery({ tag: e.target.value })}>
              <option value="">All</option>
              {tags.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
        )}

        <label className="gc-field gc-search">
          <span>Search</span>
          <input
            type="search"
            placeholder="title…"
            value={q.q}
            onChange={(e) => setQuery({ q: e.target.value })}
          />
        </label>
      </div>
      <div className="gc-count">{shown} of {total} maps</div>
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
  const tags = mapTags(mp);
  return (
    <a className="card" href={`#/map/${encodeURIComponent(mp.map_id)}`}>
      <div className="card-thumb">
        {thumb ? <img src={thumb} alt="" loading="lazy" /> : <div className="thumb-ph">density map</div>}
      </div>
      <div className="card-body">
        <div className="card-kind">
          {mp.kind || "map"}
          {mp.date ? <span className="card-date"> · {String(mp.date).slice(0, 10)}</span> : null}
        </div>
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
        {tags.length > 0 && (
          <div className="card-tags">
            {tags.slice(0, 5).map((t) => <span className="tagchip" key={t}>{t}</span>)}
          </div>
        )}
      </div>
    </a>
  );
}
