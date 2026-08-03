import React, { useEffect, useRef, useState, useCallback } from "react";
import { ViewerEngine } from "../lib/engine.js";
import {
  findBaseLayer, emptyOverlayState, defaultOverlayState, toggleLayer,
  accentFor, markerSize,
} from "../lib/layers.js";
import { BUTTON_IN, BUTTON_OUT } from "../lib/zoom.js";
import HeaderBar from "./HeaderBar.jsx";
import LegendPanel from "./LegendPanel.jsx";
import MetricsPanel from "./MetricsPanel.jsx";

// Fetch the manifest relative to the deployed app dir.
const dataDirFor = (mapId) => `../viewer/${mapId}/data`;

export default function Viewer({ mapId, theme }) {
  const [load, setLoad] = useState({ status: "loading", manifest: null, error: null });
  const [mode, setMode] = useState("map"); // "map" | "metrics"
  const [overlay, setOverlay] = useState(emptyOverlayState());
  const [legend, setLegend] = useState({ kind: "density", maxCount: 1, overlay: null });
  const [metricMode, setMetricMode] = useState("anchors");
  const [probeKey, setProbeKey] = useState(null);
  const [query, setQuery] = useState(null);

  const canvasRef = useRef(null);
  const tooltipRef = useRef(null);
  const engineRef = useRef(null);

  // Load manifest on mapId change.
  useEffect(() => {
    let alive = true;
    setLoad({ status: "loading", manifest: null, error: null });
    (async () => {
      try {
        const r = await fetch(`${dataDirFor(mapId)}/manifest.json`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const m = await r.json();
        if (alive) {
          // Point-only (projection) maps start with all point layers visible;
          // grid maps start with base density only.
          setOverlay(defaultOverlayState(m.layers));
          // Metrics sub-mode must match what the manifest actually has —
          // projection maps have probes but no anchors.
          setMetricMode(m.metrics && m.metrics.anchors ? "anchors" : "queries");
          // Reset per-map UI state so tab/probe/query never carry over when
          // navigating between maps (panel and engine must agree).
          setMode("map");
          setProbeKey(null);
          setQuery(null);
          // Tag the manifest with its mapId so the engine effect never boots
          // with a stale manifest against a new dataDir (grid map -> point map
          // transition would otherwise fetch grid-*.bin from the wrong dir = 404).
          setLoad({ status: "ok", manifest: m, mapId, error: null });
        }
      } catch (e) {
        if (alive) setLoad({ status: "error", manifest: null, error: e.message });
      }
    })();
    return () => { alive = false; };
  }, [mapId]);

  // Create engine once the manifest + canvas are ready. Guard on load.mapId ===
  // mapId so a stale manifest (mid hash-navigation) never boots against the new
  // dataDir. Without this, a grid->point transition 404s on grid-*.bin.
  useEffect(() => {
    if (load.status !== "ok" || load.mapId !== mapId || !canvasRef.current) return;
    const eng = new ViewerEngine({
      canvas: canvasRef.current,
      tooltip: tooltipRef.current,
      dataDir: dataDirFor(mapId),
      manifest: load.manifest,
      onLegend: (info) => setLegend(info),
      isDark: theme.isDark,
    });
    engineRef.current = eng;
    eng.boot();
    return () => { eng.destroy(); engineRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load.status, mapId]);

  // Push theme changes to the engine (redraw with new css vars).
  useEffect(() => { engineRef.current && engineRef.current.setTheme(); }, [theme.theme]);

  // Push overlay state -> engine. Context layers ride along as muted background
  // scatter; accent point layers carry their accent slot + marker size.
  useEffect(() => {
    const eng = engineRef.current; if (!eng || load.status !== "ok") return;
    eng.setGridOverlay(overlay.gridOverlay);
    const byKey = (key) => (load.manifest.layers || []).find((x) => x.key === key) || { key };
    const ctx = overlay.context.map((key) => {
      const l = byKey(key);
      return { key, label: l.label, group: l.group, context: true, size: markerSize(l) };
    });
    const pts = overlay.points.map((key) => {
      const l = byKey(key);
      return { key, label: l.label, group: l.group, accent: accentFor(overlay, l), size: markerSize(l) };
    });
    eng.setPointOverlays([...ctx, ...pts]);
  }, [overlay, load.status, load.manifest]);

  // Push mode/metric selections -> engine.
  useEffect(() => { engineRef.current && engineRef.current.setMode(mode); }, [mode]);
  useEffect(() => { mode === "metrics" && engineRef.current && engineRef.current.setMetricMode(metricMode); }, [metricMode, mode]);

  const onToggleLayer = useCallback((layer) => setOverlay((s) => toggleLayer(s, layer)), []);
  const onPickProbe = useCallback((probe) => {
    setProbeKey(probe ? probe.key : null); setQuery(null);
    engineRef.current && engineRef.current.setProbe(probe);
  }, []);
  const onPickQuery = useCallback((q) => {
    setQuery(q);
    engineRef.current && engineRef.current.setQuery(q);
  }, []);

  if (load.status === "loading")
    return <ViewerShell><div className="status"><div className="spinner" /><div>Loading map…</div></div></ViewerShell>;
  if (load.status === "error")
    return (
      <ViewerShell>
        <div className="status error">
          <div>Could not load this map</div>
          <div className="status-detail">{mapId}: {load.error}</div>
          <a className="back" href="#/">← back to gallery</a>
        </div>
      </ViewerShell>
    );

  const m = load.manifest;
  const base = findBaseLayer(m.layers);

  return (
    <div id="app">
      <HeaderBar
        manifest={m}
        mode={mode}
        onMode={setMode}
        theme={theme}
        hasMetrics={!!(m.metrics && (m.metrics.anchors || (m.metrics.probes && m.metrics.probes.length)))}
      />
      <div className="stage">
        <canvas id="plot" ref={canvasRef} />
        <div className="tooltip" ref={tooltipRef} hidden />
        <div className="zoomctl">
          <button onClick={() => engineRef.current && engineRef.current.zoomAt(0.5, 0.5, BUTTON_IN)} aria-label="Zoom in">+</button>
          <button onClick={() => engineRef.current && engineRef.current.zoomAt(0.5, 0.5, BUTTON_OUT)} aria-label="Zoom out">−</button>
          <button onClick={() => engineRef.current && engineRef.current.resetView()} aria-label="Reset view">⤾</button>
        </div>
      </div>
      <div className="panel">
        {mode === "map" ? (
          <LegendPanel
            manifest={m}
            baseLayer={base}
            overlay={overlay}
            legend={legend}
            isDark={theme.isDark}
            onToggleLayer={onToggleLayer}
          />
        ) : (
          <MetricsPanel
            manifest={m}
            metricMode={metricMode}
            onMetricMode={setMetricMode}
            probeKey={probeKey}
            query={query}
            engineRef={engineRef}
            onPickProbe={onPickProbe}
            onPickQuery={onPickQuery}
            isDark={theme.isDark}
          />
        )}
      </div>
    </div>
  );
}

function ViewerShell({ children }) {
  return (
    <div id="app">
      <div className="vh"><div><a className="back" href="#/">← back to gallery</a><h1>Basemap viewer</h1></div></div>
      <div className="stage">{children}</div>
      <div className="panel" />
    </div>
  );
}
