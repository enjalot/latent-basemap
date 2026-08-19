/**
 * overlay.ts — the 2D overlay layer.
 *
 * A transparent canvas stacked above the WebGL base layer. Everything that
 * follows the cursor or the selection lives here: the hover bin highlight, the
 * pinned bin, the selected point ring, and the projection markers.
 *
 * This is the ONLY layer a mousemove is allowed to touch. It costs a clearRect
 * plus a handful of strokes, so it can run at pointer rate without the density
 * raster or the point buffers being involved at all.
 */

import type { Camera } from "./gl";

export interface HighlightItem {
  /** world-space top-left of the bin and its size */
  u0: number;
  v0: number;
  size: number;
  pinned: boolean;
}

export interface PointItem {
  u: number;
  v: number;
}

export interface MarkerItem {
  u: number;
  v: number;
  label: string;
  active: boolean;
}

export interface OverlayScene {
  highlight: HighlightItem | null;
  selected: PointItem | null;
  markers: MarkerItem[];
  accent: string;
  ink: string;
  paper: string;
  /** projection markers — deliberately NOT the accent, which collides with the
   *  RedPajama corpus colour when points are on */
  marker: string;
}

export class OverlayRenderer {
  private ctx: CanvasRenderingContext2D;
  private dpr = 1;

  constructor(readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) throw new Error("2d overlay context unavailable");
    this.ctx = ctx;
  }

  resize(width: number, height: number, dpr: number) {
    this.dpr = dpr;
    const w = Math.max(1, Math.round(width * dpr));
    const h = Math.max(1, Math.round(height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private screen(cam: Camera, u: number, v: number): [number, number] {
    return [
      (u - cam.cx) * cam.scale + cam.width / 2,
      (v - cam.cy) * cam.scale + cam.height / 2,
    ];
  }

  draw(cam: Camera, scene: OverlayScene) {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, cam.width, cam.height);

    const h = scene.highlight;
    if (h) {
      const [sx, sy] = this.screen(cam, h.u0, h.v0);
      const w = h.size * cam.scale;
      ctx.save();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = h.pinned ? scene.accent : scene.ink;
      ctx.strokeRect(
        Math.round(sx) - 0.5,
        Math.round(sy) - 0.5,
        Math.max(3, w) + 1,
        Math.max(3, w) + 1,
      );
      if (w < 8) {
        ctx.beginPath();
        ctx.arc(sx + w / 2, sy + w / 2, 9, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();
    }

    const p = scene.selected;
    if (p) {
      const [sx, sy] = this.screen(cam, p.u, p.v);
      ctx.save();
      ctx.strokeStyle = scene.accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx, sy, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    for (const m of scene.markers) this.drawMarker(cam, m, scene);
  }

  /**
   * The projection dot: a halo ring in the page background colour, then the
   * marker ring, then a crosshair for the most recent one. The halo is what
   * makes it legible over dense points, which can be any corpus colour.
   */
  private drawMarker(cam: Camera, m: MarkerItem, scene: OverlayScene) {
    const ctx = this.ctx;
    const [px, py] = this.screen(cam, m.u, m.v);
    if (px < -80 || py < -80 || px > cam.width + 80 || py > cam.height + 80) return;
    const r = m.active ? 11 : 8;
    ctx.save();

    const ring = (radius: number, colour: string, width: number) => {
      ctx.lineWidth = width;
      ctx.strokeStyle = colour;
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.stroke();
    };
    ring(r, scene.paper, m.active ? 5.5 : 4);
    ring(r, scene.marker, m.active ? 2.5 : 1.5);

    if (m.active) {
      for (const [colour, width] of [
        [scene.paper, 4],
        [scene.marker, 2],
      ] as const) {
        ctx.lineWidth = width;
        ctx.strokeStyle = colour;
        ctx.beginPath();
        ctx.moveTo(px - r - 9, py);
        ctx.lineTo(px - r - 3, py);
        ctx.moveTo(px + r + 3, py);
        ctx.lineTo(px + r + 9, py);
        ctx.moveTo(px, py - r - 9);
        ctx.lineTo(px, py - r - 3);
        ctx.moveTo(px, py + r + 3);
        ctx.lineTo(px, py + r + 9);
        ctx.stroke();
      }
    }
    ctx.fillStyle = scene.paper;
    ctx.beginPath();
    ctx.arc(px, py, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = scene.marker;
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();

    if (m.label) {
      ctx.font =
        "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      const text = m.label.length > 34 ? m.label.slice(0, 33) + "…" : m.label;
      const w = ctx.measureText(text).width;
      const bx = px + r + 6;
      const by = py - 9;
      ctx.fillStyle = scene.paper;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(bx - 3, by, w + 6, 16);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = scene.marker;
      ctx.lineWidth = 1;
      ctx.strokeRect(bx - 3.5, by - 0.5, w + 7, 17);
      ctx.fillStyle = scene.ink;
      ctx.textBaseline = "middle";
      ctx.fillText(text, bx, by + 8);
    }
    ctx.restore();
  }
}
