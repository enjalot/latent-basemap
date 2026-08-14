/**
 * gl.ts — the WebGL2 base layer.
 *
 * Two programs, no framework:
 *
 *   tiles   one textured quad per visible density tile. The RGBA that the
 *           compose worker produced is uploaded to a texture ONCE per recompose
 *           (corpus toggle / colour mode change), keyed by `version`. Panning
 *           and zooming only change two uniforms.
 *   points  one `gl.POINTS` draw per point batch. The interleaved position and
 *           (corpus, min_zoom) attribute buffers are uploaded ONCE per data
 *           change; the camera arrives as uniforms, corpus visibility as a
 *           bitmask, and the LOD cutover as `uMaxMinZ`.
 *
 * World space is the unit square (u right, v down). The camera maps it to clip
 * space with `clip = (world - uCenter) * uScale`, which is the only per-frame
 * state the GPU needs.
 */

import { stats } from "./render";

export interface Camera {
  cx: number;
  cy: number;
  scale: number;
  /** CSS px */
  width: number;
  height: number;
}

/** One textured quad: a destination rect in world space + a source sub-rect. */
export interface TileRef {
  key: string;
  u0: number;
  v0: number;
  du: number;
  dv: number;
  su: number;
  sv: number;
  sw: number;
  sh: number;
}

export interface PointRef {
  name: string;
  /**
   * [firstVertex, count] runs to submit. The caller derives these from the LOD
   * spatial index so off-screen vertices are never shaded — the difference
   * between 1.8M and ~40k vertices per frame on a deep zoom.
   */
  ranges: [number, number][];
  /** device px */
  size: number;
  alpha: number;
  /** bit i set => corpus code i is visible */
  mask: number;
  /** draw only records with min_zoom <= this */
  maxMinZ: number;
}

export interface Scene {
  bg: [number, number, number];
  tiles: TileRef[];
  points: PointRef[];
}

export interface PointData {
  n: number;
  u: Float32Array;
  v: Float32Array;
  corpus: Uint8Array;
  minz?: Uint8Array;
}

const TILE_VS = `#version 300 es
in vec2 aCorner;
uniform vec2 uCenter;
uniform vec2 uScale;
uniform vec4 uRect;   // u0, v0, du, dv   (world)
uniform vec4 uTex;    // su, sv, sw, sh   (0..1)
out vec2 vTex;
void main() {
  vec2 world = uRect.xy + aCorner * uRect.zw;
  gl_Position = vec4((world - uCenter) * uScale, 0.0, 1.0);
  vTex = uTex.xy + aCorner * uTex.zw;
}`;

const TILE_FS = `#version 300 es
precision mediump float;
in vec2 vTex;
uniform sampler2D uTexture;
out vec4 frag;
void main() { frag = texture(uTexture, vTex); }`;

const POINT_VS = `#version 300 es
in vec2 aPos;
in vec2 aMeta;        // x = corpus code, y = min_zoom
uniform vec2 uCenter;
uniform vec2 uScale;
uniform float uPointSize;
uniform float uMaxMinZ;
uniform float uAlpha;
uniform int uMask;
uniform vec3 uColors[16];
out vec4 vColor;
void main() {
  int c = clamp(int(aMeta.x + 0.5), 0, 15);
  if (((uMask >> c) & 1) == 0 || aMeta.y > uMaxMinZ) {
    // cull: outside the clip volume, zero-sized
    gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    vColor = vec4(0.0);
    return;
  }
  gl_Position = vec4((aPos - uCenter) * uScale, 0.0, 1.0);
  gl_PointSize = uPointSize;
  vColor = vec4(uColors[c], uAlpha);
}`;

const POINT_FS = `#version 300 es
precision mediump float;
in vec4 vColor;
out vec4 frag;
void main() { frag = vColor; }`;

interface TexEntry {
  tex: WebGLTexture;
  version: string;
}

interface BufEntry {
  vao: WebGLVertexArrayObject;
  pos: WebGLBuffer;
  meta: WebGLBuffer;
  n: number;
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error("createShader failed");
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`shader compile failed: ${log}`);
  }
  return sh;
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram();
  if (!p) throw new Error("createProgram failed");
  const v = compile(gl, gl.VERTEX_SHADER, vs);
  const f = compile(gl, gl.FRAGMENT_SHADER, fs);
  gl.attachShader(p, v);
  gl.attachShader(p, f);
  gl.linkProgram(p);
  gl.deleteShader(v);
  gl.deleteShader(f);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error(`program link failed: ${log}`);
  }
  return p;
}

export class GLRenderer {
  readonly gl: WebGL2RenderingContext;
  readonly renderer: string;

  private tileProg: WebGLProgram;
  private pointProg: WebGLProgram;
  private quadVao: WebGLVertexArrayObject;
  private tileU: Record<string, WebGLUniformLocation | null> = {};
  private pointU: Record<string, WebGLUniformLocation | null> = {};

  private textures = new Map<string, TexEntry>();
  private buffers = new Map<string, BufEntry>();
  private colors = new Float32Array(16 * 3);

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: "high-performance",
    });
    if (!gl) throw new Error("WebGL2 unavailable");
    this.gl = gl;
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    this.renderer = String(
      dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    );

    this.tileProg = link(gl, TILE_VS, TILE_FS);
    this.pointProg = link(gl, POINT_VS, POINT_FS);

    for (const n of ["uCenter", "uScale", "uRect", "uTex", "uTexture"])
      this.tileU[n] = gl.getUniformLocation(this.tileProg, n);
    for (const n of ["uCenter", "uScale", "uPointSize", "uMaxMinZ", "uAlpha", "uMask", "uColors"])
      this.pointU[n] = gl.getUniformLocation(this.pointProg, n);

    // one unit quad, reused by every tile draw
    const vao = gl.createVertexArray();
    if (!vao) throw new Error("createVertexArray failed");
    this.quadVao = vao;
    gl.bindVertexArray(vao);
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]),
      gl.STATIC_DRAW,
    );
    const loc = gl.getAttribLocation(this.tileProg, "aCorner");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(
      gl.SRC_ALPHA,
      gl.ONE_MINUS_SRC_ALPHA,
      gl.ONE,
      gl.ONE_MINUS_SRC_ALPHA,
    );
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  }

  setCorpusColors(colors: Record<number, [number, number, number]>) {
    this.colors.fill(0.53);
    for (const [code, rgb] of Object.entries(colors)) {
      const i = Number(code);
      if (i < 0 || i > 15) continue;
      this.colors[i * 3] = rgb[0] / 255;
      this.colors[i * 3 + 1] = rgb[1] / 255;
      this.colors[i * 3 + 2] = rgb[2] / 255;
    }
  }

  resize(width: number, height: number, dpr: number) {
    const c = this.gl.canvas as HTMLCanvasElement;
    const w = Math.max(1, Math.round(width * dpr));
    const h = Math.max(1, Math.round(height * dpr));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    this.gl.viewport(0, 0, w, h);
  }

  // -- texture cache ---------------------------------------------------------

  tileVersion(key: string): string | undefined {
    return this.textures.get(key)?.version;
  }

  /** Upload a composed tile. Called once per recompose, never per frame. */
  uploadTile(key: string, version: string, rgba: Uint8ClampedArray, size: number) {
    const gl = this.gl;
    let entry = this.textures.get(key);
    if (!entry) {
      const tex = gl.createTexture();
      if (!tex) return;
      entry = { tex, version: "" };
      this.textures.set(key, entry);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      // NEAREST magnification keeps the bin grid crisp (v1 drew with
      // imageSmoothingEnabled=false); mipmapped minification stops the raster
      // shimmering when a level is drawn below 1:1.
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    } else {
      gl.bindTexture(gl.TEXTURE_2D, entry.tex);
    }
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      size,
      size,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength),
    );
    gl.generateMipmap(gl.TEXTURE_2D);
    entry.version = version;
    stats.tileUploads++;
  }

  dropTile(key: string) {
    const e = this.textures.get(key);
    if (!e) return;
    this.gl.deleteTexture(e.tex);
    this.textures.delete(key);
  }

  // -- point buffers ---------------------------------------------------------

  hasPoints(name: string): boolean {
    return this.buffers.has(name);
  }

  /** Upload a point batch. Called once per data change, never per frame. */
  uploadPoints(name: string, b: PointData) {
    const gl = this.gl;
    this.dropPoints(name);
    const pos = new Float32Array(b.n * 2);
    const meta = new Uint8Array(b.n * 2);
    for (let i = 0; i < b.n; i++) {
      pos[i * 2] = b.u[i];
      pos[i * 2 + 1] = b.v[i];
      meta[i * 2] = b.corpus[i];
      meta[i * 2 + 1] = b.minz ? b.minz[i] : 0;
    }
    const vao = gl.createVertexArray();
    const pbuf = gl.createBuffer();
    const mbuf = gl.createBuffer();
    if (!vao || !pbuf || !mbuf) return;
    gl.bindVertexArray(vao);
    const aPos = gl.getAttribLocation(this.pointProg, "aPos");
    const aMeta = gl.getAttribLocation(this.pointProg, "aMeta");
    gl.bindBuffer(gl.ARRAY_BUFFER, pbuf);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, mbuf);
    gl.bufferData(gl.ARRAY_BUFFER, meta, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(aMeta);
    gl.vertexAttribPointer(aMeta, 2, gl.UNSIGNED_BYTE, false, 0, 0);
    gl.bindVertexArray(null);
    this.buffers.set(name, { vao, pos: pbuf, meta: mbuf, n: b.n });
    stats.pointUploads++;
  }

  dropPoints(name: string) {
    const e = this.buffers.get(name);
    if (!e) return;
    this.gl.deleteVertexArray(e.vao);
    this.gl.deleteBuffer(e.pos);
    this.gl.deleteBuffer(e.meta);
    this.buffers.delete(name);
  }

  /** Drop every point buffer whose name is not in `keep`. */
  retainPoints(keep: Set<string>) {
    for (const name of [...this.buffers.keys()])
      if (!keep.has(name)) this.dropPoints(name);
  }

  // -- draw ------------------------------------------------------------------

  draw(cam: Camera, scene: Scene) {
    const gl = this.gl;
    const sx = (2 * cam.scale) / Math.max(1, cam.width);
    const sy = (-2 * cam.scale) / Math.max(1, cam.height);

    gl.clearColor(scene.bg[0] / 255, scene.bg[1] / 255, scene.bg[2] / 255, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (scene.tiles.length) {
      gl.useProgram(this.tileProg);
      gl.bindVertexArray(this.quadVao);
      gl.uniform2f(this.tileU.uCenter!, cam.cx, cam.cy);
      gl.uniform2f(this.tileU.uScale!, sx, sy);
      gl.uniform1i(this.tileU.uTexture!, 0);
      gl.activeTexture(gl.TEXTURE0);
      for (const t of scene.tiles) {
        const e = this.textures.get(t.key);
        if (!e) continue;
        gl.bindTexture(gl.TEXTURE_2D, e.tex);
        gl.uniform4f(this.tileU.uRect!, t.u0, t.v0, t.du, t.dv);
        gl.uniform4f(this.tileU.uTex!, t.su, t.sv, t.sw, t.sh);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
      gl.bindVertexArray(null);
    }

    if (scene.points.length) {
      gl.useProgram(this.pointProg);
      gl.uniform2f(this.pointU.uCenter!, cam.cx, cam.cy);
      gl.uniform2f(this.pointU.uScale!, sx, sy);
      gl.uniform3fv(this.pointU.uColors!, this.colors);
      for (const p of scene.points) {
        const e = this.buffers.get(p.name);
        if (!e || !e.n) continue;
        gl.bindVertexArray(e.vao);
        gl.uniform1f(this.pointU.uPointSize!, p.size);
        gl.uniform1f(this.pointU.uMaxMinZ!, p.maxMinZ);
        gl.uniform1f(this.pointU.uAlpha!, p.alpha);
        gl.uniform1i(this.pointU.uMask!, p.mask);
        for (const [first, count] of p.ranges) {
          const f = Math.max(0, Math.min(first, e.n));
          const n = Math.max(0, Math.min(count, e.n - f));
          if (n > 0) gl.drawArrays(gl.POINTS, f, n);
        }
      }
      gl.bindVertexArray(null);
    }
  }

  /**
   * How many distinct colours are on screen — the "did it paint?" check.
   * Must be called in the SAME task as a draw: the drawing buffer is not
   * preserved (preserveDrawingBuffer costs a full copy on every swap, which is
   * exactly the kind of per-frame tax this renderer exists to avoid).
   */
  sampleDistinctColours(step = 4): number {
    const gl = this.gl;
    const c = gl.canvas as HTMLCanvasElement;
    const px = new Uint8Array(c.width * c.height * 4);
    gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const seen = new Set<string>();
    for (let i = 0; i < px.length; i += 4 * step)
      seen.add(`${px[i]},${px[i + 1]},${px[i + 2]}`);
    return seen.size;
  }

  dispose() {
    for (const k of [...this.textures.keys()]) this.dropTile(k);
    for (const k of [...this.buffers.keys()]) this.dropPoints(k);
    this.gl.deleteProgram(this.tileProg);
    this.gl.deleteProgram(this.pointProg);
    this.gl.deleteVertexArray(this.quadVao);
    const lose = this.gl.getExtension("WEBGL_lose_context");
    lose?.loseContext();
  }
}
