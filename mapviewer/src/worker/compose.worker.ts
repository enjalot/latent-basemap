/**
 * compose.worker.ts — recomposes a density tile's RGBA from the per-corpus u32
 * count planes. Runs off the main thread so corpus toggles stay interactive.
 *
 * Two modes:
 *   combined — log1p(sum of enabled planes) on the YlGnBu ramp (default look)
 *   dominant — per-bin argmax over enabled planes -> corpus colour, with
 *              log1p(total) driving alpha so density still reads
 */

import { YLGNBU } from "../palette";

export interface ComposeRequest {
  key: string;
  /** corpus code -> plane buffer (256*256 u32, row-major, y-down) */
  planes: Record<number, ArrayBuffer>;
  enabled: number[];
  mode: "combined" | "dominant";
  /** normalisation count for the log ramp at this level */
  norm: number;
  size: number;
  /** corpus code -> [r,g,b] */
  colors: Record<number, [number, number, number]>;
}

export interface ComposeResponse {
  key: string;
  rgba: ArrayBuffer;
  size: number;
  /** max combined count in this tile, for the legend */
  max: number;
}

function compose(req: ComposeRequest): ComposeResponse {
  const { size, mode, colors } = req;
  const npx = size * size;
  const enabled = req.enabled.filter((c) => req.planes[c] !== undefined);
  const planes = enabled.map((c) => new Uint32Array(req.planes[c]));

  const total = new Uint32Array(npx);
  for (const p of planes) {
    const n = Math.min(npx, p.length);
    for (let i = 0; i < n; i++) total[i] += p[i];
  }

  let max = 0;
  for (let i = 0; i < npx; i++) if (total[i] > max) max = total[i];

  const norm = Math.log1p(Math.max(req.norm, max, 1));
  const out = new Uint8ClampedArray(npx * 4);

  if (mode === "combined") {
    const last = YLGNBU.length - 1;
    for (let i = 0; i < npx; i++) {
      const c = total[i];
      if (c === 0) continue;
      const t = Math.min(1, Math.log1p(c) / norm);
      const idx = t * last;
      const lo = Math.floor(idx);
      const hi = Math.min(lo + 1, last);
      const f = idx - lo;
      const a = YLGNBU[lo];
      const b = YLGNBU[hi];
      const o = i * 4;
      out[o] = a[0] + (b[0] - a[0]) * f;
      out[o + 1] = a[1] + (b[1] - a[1]) * f;
      out[o + 2] = a[2] + (b[2] - a[2]) * f;
      out[o + 3] = 255;
    }
  } else {
    for (let i = 0; i < npx; i++) {
      const c = total[i];
      if (c === 0) continue;
      let best = -1;
      let bestV = 0;
      for (let k = 0; k < planes.length; k++) {
        const v = planes[k][i];
        if (v > bestV) {
          bestV = v;
          best = enabled[k];
        }
      }
      const col = colors[best] ?? [136, 136, 136];
      // purity: how dominant the winner is (0.5 .. 1 for a 2-way split)
      const purity = c > 0 ? bestV / c : 1;
      const t = Math.min(1, Math.log1p(c) / norm);
      const o = i * 4;
      // wash toward white as purity drops, so mixed bins read as mixed
      const w = 1 - Math.min(1, Math.max(0, (purity - 0.34) / 0.66));
      out[o] = col[0] + (245 - col[0]) * w * 0.7;
      out[o + 1] = col[1] + (245 - col[1]) * w * 0.7;
      out[o + 2] = col[2] + (245 - col[2]) * w * 0.7;
      out[o + 3] = 60 + 195 * t;
    }
  }

  return { key: req.key, rgba: out.buffer, size, max };
}

self.onmessage = (ev: MessageEvent<ComposeRequest>) => {
  const res = compose(ev.data);
  (self as unknown as Worker).postMessage(res, [res.rgba]);
};
