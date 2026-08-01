import React, { useLayoutEffect, useRef, useState } from "react";
import { fmt, rowsPhrase } from "../lib/format.js";
import { computeTicks } from "../lib/legend.js";
import {
  groupLayers, isGridLayer, isContextLayer, isActive, accentFor,
} from "../lib/layers.js";
import { densityRamp, accentRamp, rampGradient } from "../lib/ramps.js";

// THE control surface: (a) density ramp block (grid maps) or a point-mark legend
// (point-only projection maps); (b) grouped layer list where every row = color
// swatch + label + formatted row count + toggle. EVERY manifest layer gets a row.
export default function LegendPanel({ manifest: m, baseLayer, overlay, legend, isDark, onToggleLayer }) {
  const hasGrid = (m.layers || []).some(isGridLayer);
  const groups = groupLayers(m.layers, baseLayer ? baseLayer.key : null);
  const dark = isDark();
  const gridActive = !!overlay.gridOverlay;

  return (
    <div className="legend-panel">
      {hasGrid ? (
        <DensityBlock legend={legend} dark={dark} gridActive={gridActive} />
      ) : (
        <PointLegendBlock manifest={m} overlay={overlay} />
      )}

      {groups.map(({ section, layers }) => (
        <div className="section" key={section}>
          <h2>{section}</h2>
          <div className="layer-rows">
            {layers.map((l) => (
              <LayerRow
                key={l.key}
                layer={l}
                active={isActive(overlay, l)}
                accentSlot={accentFor(overlay, l)}
                dark={dark}
                onToggle={() => onToggleLayer(l)}
              />
            ))}
          </div>
        </div>
      ))}

      <div className="section hint">
        {hasGrid
          ? "Grid subsets recolor on the orange accent ramp (one at a time; base density mutes). Point layers stack up to two with distinct accents."
          : "Context points render as muted gray; probe layers use accent colors (up to two at once)."}{" "}
        Scroll / +− to zoom, drag to pan, 0 to reset.
        {hasGrid ? " Hover a bin for row count and text samples." : " Hover a probe point for its layer."}
      </div>
    </div>
  );
}

function DensityBlock({ legend, dark, gridActive }) {
  const barRef = useRef(null);
  const [barW, setBarW] = useState(220);
  useLayoutEffect(() => {
    if (!barRef.current) return;
    const measure = () => setBarW(barRef.current.getBoundingClientRect().width || 220);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(barRef.current);
    return () => ro.disconnect();
  }, []);

  const isDensity = legend.kind !== "anchor";
  const stops = gridActive ? accentRamp(dark) : densityRamp(dark);
  const cap = Math.max(1, legend.maxCount || 1);
  const ticks = computeTicks(cap, barW);
  const title = gridActive
    ? `${legend.overlay ? legend.overlay.label : "subset"} — rows per bin (log)`
    : "Density — rows per bin (log scale)";

  return (
    <div className="section density-block">
      <h2>{isDensity ? "Density" : "Anchor score"}</h2>
      <div className="dl-title">{isDensity ? title : "local expansion (blue low ↔ red high)"}</div>
      <div
        className="ramp-bar"
        ref={barRef}
        style={{ background: isDensity ? rampGradient(stops) : "var(--anchor-legend)" }}
      />
      {isDensity ? (
        <div className="ramp-ticks">
          {ticks.map((t, i) => (
            <span key={i} style={{ left: `${(t.pos * 100).toFixed(1)}%` }}>{fmt(t.v)}</span>
          ))}
        </div>
      ) : (
        <div className="ramp-ticks">
          <span style={{ left: "0%" }}>low</span>
          <span style={{ left: "50%" }}>median</span>
          <span style={{ left: "100%" }}>high</span>
        </div>
      )}
    </div>
  );
}

// Point-only maps have no count ramp — show what each visible mark means instead.
function PointLegendBlock({ manifest: m, overlay }) {
  const visible = (m.layers || []).filter((l) => isActive(overlay, l));
  return (
    <div className="section density-block">
      <h2>Point layers</h2>
      <div className="dl-title">every mark is one row (deterministic sample where noted)</div>
      <div className="pt-legend">
        {visible.length === 0 && <div className="hint">No layers visible — toggle one below.</div>}
        {visible.map((l) => {
          const ctx = isContextLayer(l);
          const slot = accentFor(overlay, l);
          const color = ctx
            ? "var(--ink-muted)"
            : `var(${slot === "a2" ? "--accent-2" : "--accent"})`;
          return (
            <div className="pt-row" key={l.key} title={l.label}>
              <span className={"pt-dot" + (ctx ? " ctx" : "")} style={{ background: color }} />
              <span className="pt-label">{l.label || l.key}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function LayerRow({ layer, active, accentSlot, dark, onToggle }) {
  const grid = isGridLayer(layer);
  const ctx = isContextLayer(layer);
  // Swatch: grid subset = accent ramp chip; context = muted gray dot;
  // point layer = accent dot (a1/a2).
  const swatchStyle = grid
    ? { background: rampGradient(accentRamp(dark)) }
    : ctx
      ? { background: "var(--ink-muted)", borderRadius: "50%" }
      : { background: `var(${accentSlot === "a2" ? "--accent-2" : "--accent"})`, borderRadius: "50%" };
  const label = layer.label || layer.key;
  const count = rowsPhrase(layer.rows, layer.sampled_of);
  return (
    <button
      type="button"
      className={"layer-row" + (active ? " active" : "")}
      onClick={onToggle}
      aria-pressed={active}
      title={`${label} — ${count}`}
    >
      <span className={"lr-swatch" + (grid ? " ramp" : " dot")} style={active ? swatchStyle : undefined} />
      <span className="lr-main">
        <span className="lr-label">{label}</span>
        <span className="lr-count">{count}</span>
      </span>
      <span className={"lr-toggle" + (active ? " on" : "")} aria-hidden="true" />
    </button>
  );
}
