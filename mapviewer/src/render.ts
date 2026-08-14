/**
 * render.ts — the single rAF loop and the dev render counter.
 *
 * Viewer v2 splits rendering into two layers with very different costs:
 *
 *   base    WebGL2 — density tile textures + point sprite buffers. Expensive to
 *           *upload*, cheap to *draw*. Redrawn only when the camera moves or the
 *           data changes.
 *   overlay 2D canvas — hover highlight, selection ring, projection markers.
 *           Redrawn on every pointer move; costs a handful of strokes.
 *
 * Everything funnels through one requestAnimationFrame so a burst of events
 * (mousemove + wheel + tile arrivals) collapses into at most one draw per frame
 * instead of a synchronous re-render storm.
 *
 * `window.__renderStats` is the verification surface: mousing across the map
 * must increment `overlayDraws` and leave `baseDraws`, `tileUploads` and
 * `pointUploads` untouched.
 */

export interface RenderStats {
  /** WebGL base-layer draws (density + points) */
  baseDraws: number;
  /** 2D overlay draws (hover, selection, projection markers) */
  overlayDraws: number;
  /** density tile texture uploads (once per tile recompose) */
  tileUploads: number;
  /** point VBO uploads (once per point batch load) */
  pointUploads: number;
  /** rAF callbacks that did any work */
  frames: number;
  /** ms spent in the last base draw / worst base draw since reset */
  lastBaseMs: number;
  maxBaseMs: number;
  /** ms spent in the last overlay draw / worst overlay draw since reset */
  lastOverlayMs: number;
  maxOverlayMs: number;
  /** performance.now() of the last base draw, for "did hovering redraw?" */
  lastBaseAt: number;
  /** wall clock of the last reset */
  since: number;
}

function blank(): RenderStats {
  return {
    baseDraws: 0,
    overlayDraws: 0,
    tileUploads: 0,
    pointUploads: 0,
    frames: 0,
    lastBaseMs: 0,
    maxBaseMs: 0,
    lastOverlayMs: 0,
    maxOverlayMs: 0,
    lastBaseAt: 0,
    since: performance.now(),
  };
}

export const stats: RenderStats = blank();

export function resetStats(): RenderStats {
  Object.assign(stats, blank());
  return stats;
}

const w = window as unknown as Record<string, unknown>;
w.__renderStats = stats;
w.__resetRenderStats = resetStats;

/**
 * Coalescing draw scheduler. `markBase` is for data changes (a tile composed, a
 * point buffer loaded); `markView` is for camera changes, which move both
 * layers; `markOverlay` is for pointer-driven overlay-only work.
 */
export class RenderLoop {
  private baseDirty = false;
  private overlayDirty = false;
  private raf = 0;

  constructor(
    private readonly drawBase: () => void,
    private readonly drawOverlay: () => void,
  ) {}

  markBase() {
    this.baseDirty = true;
    this.schedule();
  }

  markOverlay() {
    this.overlayDirty = true;
    this.schedule();
  }

  /** Camera moved: both layers are stale. */
  markView() {
    this.baseDirty = true;
    this.overlayDirty = true;
    this.schedule();
  }

  /** Draw now, synchronously — only for resize/teardown paths. */
  flush() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.baseDirty = true;
    this.overlayDirty = true;
    this.tick();
  }

  private schedule() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.tick();
    });
  }

  private tick() {
    stats.frames++;
    if (this.baseDirty) {
      this.baseDirty = false;
      const t0 = performance.now();
      this.drawBase();
      const dt = performance.now() - t0;
      stats.baseDraws++;
      stats.lastBaseMs = dt;
      stats.lastBaseAt = t0;
      if (dt > stats.maxBaseMs) stats.maxBaseMs = dt;
    }
    if (this.overlayDirty) {
      this.overlayDirty = false;
      const t0 = performance.now();
      this.drawOverlay();
      const dt = performance.now() - t0;
      stats.overlayDraws++;
      stats.lastOverlayMs = dt;
      if (dt > stats.maxOverlayMs) stats.maxOverlayMs = dt;
    }
  }

  dispose() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
  }
}
