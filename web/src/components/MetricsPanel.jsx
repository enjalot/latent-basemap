import React, { useEffect, useState } from "react";
import { fmt, sig4 } from "../lib/format.js";

const METRIC_LABELS = {
  median_radius_ratio: "median radius ratio",
  log2_ratio_min: "log2 ratio min",
  log2_ratio_p10: "log2 ratio p10",
  log2_ratio_median: "log2 ratio median",
  log2_ratio_p90: "log2 ratio p90",
  log2_ratio_max: "log2 ratio max",
  expanded_frac: "expanded fraction",
  ffr: "FFR",
  n_anchors: "anchors",
  recall_at_10: "recall@10 (sparse, secondary)",
  recall_at_50_of_high10: "recall@50-of-high10 (sparse, secondary)",
};
const metricLabel = (k) => METRIC_LABELS[k] || k;

export default function MetricsPanel({ manifest: m, metricMode, onMetricMode, probeKey, query, engineRef, onPickProbe, onPickQuery }) {
  const metrics = m.metrics || {};
  const hasAnchors = !!metrics.anchors;
  const hasProbes = !!(metrics.probes && metrics.probes.length);

  return (
    <div className="metrics-panel">
      <div className="seg">
        <button className={metricMode === "anchors" ? "active" : ""} disabled={!hasAnchors} onClick={() => onMetricMode("anchors")}>Anchors</button>
        <button className={metricMode === "queries" ? "active" : ""} disabled={!hasProbes} onClick={() => onMetricMode("queries")}>Held-out queries</button>
      </div>
      {metricMode === "anchors"
        ? <AnchorsSection anchors={metrics.anchors} />
        : <QueriesSection engineRef={engineRef} probeKey={probeKey} query={query} onPickProbe={onPickProbe} onPickQuery={onPickQuery} />}
    </div>
  );
}

function AnchorsSection({ anchors }) {
  if (!anchors) return <div className="section hint">No anchor metrics for this map.</div>;
  const label = anchors.score || "score";
  const summary = anchors.summary || {};
  return (
    <div className="section">
      <h2>Anchors</h2>
      <div className="hint">
        {anchors.count ? `${fmt(anchors.count)} anchors` : "anchors"} colored by {label}. Hover a point for its value.
      </div>
      <div className="metric-list">
        {Object.entries(summary).map(([k, v]) => {
          if (k === "score_label") return null;
          const val = typeof v === "number" ? sig4(v) : v;
          return <div className="metric-row" key={k}><span>{metricLabel(k)}</span><b>{val}</b></div>;
        })}
      </div>
    </div>
  );
}

function QueriesSection({ engineRef, probeKey, query, onPickProbe, onPickQuery }) {
  const [doc, setDoc] = useState(null);
  useEffect(() => {
    let alive = true;
    const eng = engineRef.current;
    if (!eng) return;
    eng.loadQueriesDoc().then((d) => { if (alive) setDoc(d); });
    return () => { alive = false; };
  }, [engineRef]);

  if (!doc) return <div className="section hint">Loading probes…</div>;
  const probes = doc.probes || [];
  if (!probes.length) return <div className="section hint">No query probes for this map.</div>;
  const probe = probes.find((p) => p.key === probeKey) || null;

  return (
    <>
      <div className="section">
        <h2>Held-out probes</h2>
        <div className="plist">
          {probes.map((p) => (
            <button key={p.key} className={probeKey === p.key ? "active" : ""} onClick={() => onPickProbe(p)}>
              <span>{p.label || p.key}</span>
              <span className="r">{p.recall50 != null ? "R@50 " + Number(p.recall50).toFixed(3) : ""}</span>
            </button>
          ))}
        </div>
        <div className="legend-inline"><span className="swatch circle" style={{ background: "var(--status-good)" }} /> hit (in retrieved top-50)</div>
        <div className="legend-inline"><span className="swatch diamond" style={{ background: "var(--status-serious)" }} /> miss</div>
      </div>

      {probe && (
        <div className="section">
          <h2>Queries</h2>
          <div className="hint">Click a query to trace its 10 true neighbors.</div>
          {query && <QueryCard q={query} />}
          <div className="plist qscroll">
            {(probe.queries || []).slice(0, 200).map((q, i) => (
              <button key={i} className={query === q ? "active" : ""} onClick={() => onPickQuery(q)}>
                <span className="qname">{q.text ? q.text.slice(0, 42) : `query ${i + 1}`}</span>
                <span className="r">{q.recall != null ? q.recall.toFixed(2) : ""}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

function QueryCard({ q }) {
  return (
    <div className="qcard">
      <div className="qrecall">
        {q.recall != null ? Math.round(q.recall * 100) + "%" : "—"}
        <small> recall (hits in top-50)</small>
      </div>
      {q.text && <div className="qtext">{q.text}</div>}
      <div className="nlist">
        {q.hits.map((hit, i) => {
          const nb = q.neighbors[i];
          const txt = (q.neighbor_texts && q.neighbor_texts[i]) || `neighbor ${i + 1}  (x ${nb[0].toFixed(2)}, y ${nb[1].toFixed(2)})`;
          return (
            <div className={"nrow " + (hit ? "hit" : "miss")} key={i}>
              <span className="mark">{hit ? "●" : "◇"}</span>
              <span className="ntext">{txt}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
