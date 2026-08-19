#!/usr/bin/env node
/**
 * smoke.mjs — headless smoke check for the built viewer.
 *
 * Serves `dist/` (with `packs/`, `vendor/` and `models/` mounted alongside)
 * over a plain static server that can be configured to honour or ignore HTTP
 * Range, drives the page with playwright-core + the cached
 * chrome-headless-shell, and asserts:
 *   - zero console errors / page errors / failed requests
 *   - the density raster actually painted (WebGL framebuffer is not uniform)
 *   - hover produces a bin readout with corpus composition
 *   - v2 LAYERING: a 2 s scripted mousemove sweep produces ZERO base-layer
 *     draws / texture uploads / point uploads and > 0 overlay draws
 *   - v2 POINT PERF: panning with LOD points on triggers no long task > 50 ms
 *   - point mode loads and a point click resolves text through the sidecar
 *   - projection: text -> MiniLM -> map head lands within tolerance of the
 *     POC's python reference, and the dot is drawn on the overlay canvas
 *
 * Usage:
 *   node scripts/smoke.mjs                 # chunked mode (Range ignored, like gsv:8800)
 *   node scripts/smoke.mjs --range         # http-range mode
 *   node scripts/smoke.mjs --out shots/    # where screenshots land
 *   node scripts/smoke.mjs --no-projection # skip the ~83 MB model download
 *   node scripts/smoke.mjs --projection-precision fp32
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

const args = process.argv.slice(2);
const HONOR_RANGE = args.includes("--range");
const OUT = (() => {
  const i = args.indexOf("--out");
  return path.resolve(ROOT, i >= 0 ? args[i + 1] : "smoke-shots");
})();
/** --url <base>  drive an already-served viewer instead of the built-in server */
const EXT_URL = (() => {
  const i = args.indexOf("--url");
  return i >= 0 ? args[i + 1] : null;
})();
/** --map <id>  pick a specific pack from the index */
const MAP_ID = (() => {
  const i = args.indexOf("--map");
  return i >= 0 ? args[i + 1] : null;
})();
/** --no-projection  skip the model download + projection assertions */
const WITH_PROJECTION = !args.includes("--no-projection");
const PROJECTION_PRECISION = (() => {
  const i = args.indexOf("--projection-precision");
  return i >= 0 ? args[i + 1] : "int8";
})();

/**
 * The POC's python reference (sentence-transformers + the torch map head).
 *
 * `text` are the exact strings from projection-poc/verify_headless.mjs (NOT the
 * 44-char display truncations in comparison.json), `xy` are the matching
 * `xy_reference` values from projection-poc/reference/comparison.json. The
 * browser must reproduce these within the POC's own gate: 0.5 % of the extent
 * diagonal — the int8 encoder measured 0.009–0.139 % there.
 */
const PROJECTION_REFERENCE = {
  /** the reference was computed against THIS pack's map head, not any other */
  mapId: "sandbox-2m-umap-md000-x4-fneg10",
  extentDiagonal: 67.19756717339834,
  gateFrac: 0.005,
  rows: [
    {
      text: "the quick brown fox jumps over the lazy dog",
      xy: [-8.992168426513672, 10.351940155029297],
    },
    {
      text: "Photosynthesis converts light energy into chemical energy in plants.",
      xy: [-12.99052619934082, 9.653836250305176],
    },
    {
      text: "def quicksort(a):\n    if len(a) <= 1: return a",
      xy: [-3.2541537284851074, -29.240671157836914],
    },
    {
      text: "Interest rates rose sharply after the central bank meeting.",
      xy: [4.360757827758789, -5.89774751663208],
    },
    {
      text: "Ich habe gestern ein sehr gutes Buch gelesen.",
      xy: [21.435216903686523, 2.3670802116394043],
    },
  ],
};

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".u32": "application/octet-stream",
  ".bin": "application/octet-stream",
  ".u64": "application/octet-stream",
  ".utf8": "text/plain; charset=utf-8",
};

function resolveFile(urlPath) {
  const clean = decodeURIComponent(urlPath.split("?")[0]);
  let rel = clean.replace(/^\/+/, "");
  if (rel === "" || rel.endsWith("/")) rel += "index.html";
  if (rel.startsWith("packs/")) return path.join(ROOT, rel);
  // the projection runtime is vendored in projection-poc/ and rsynced next to
  // the site by deploy.sh; mirror that layout for the local server
  if (rel.startsWith("vendor/") || rel.startsWith("models/"))
    return path.join(ROOT, "projection-poc", rel);
  return path.join(ROOT, "dist", rel);
}

/**
 * `honorRange:false` reproduces python http.server: the Range header is
 * ignored and the whole body comes back with 200.
 */
function startServer(honorRange) {
  const server = http.createServer((req, res) => {
    const file = resolveFile(req.url);
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404).end("not found");
      return;
    }
    const stat = fs.statSync(file);
    const type = MIME[path.extname(file)] ?? "application/octet-stream";
    const range = req.headers.range;
    if (honorRange && range) {
      const m = /bytes=(\d+)-(\d*)/.exec(range);
      if (m) {
        const start = Number(m[1]);
        const end = m[2] ? Number(m[2]) : stat.size - 1;
        res.writeHead(206, {
          "Content-Type": type,
          "Content-Range": `bytes ${start}-${end}/${stat.size}`,
          "Content-Length": end - start + 1,
          "Accept-Ranges": "bytes",
        });
        fs.createReadStream(file, { start, end }).pipe(res);
        return;
      }
    }
    res.writeHead(200, { "Content-Type": type, "Content-Length": stat.size });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () =>
      resolve({ server, port: server.address().port }),
    );
  });
}

function findChromeShell() {
  const base = path.join(process.env.HOME, ".cache", "ms-playwright");
  const cands = fs
    .readdirSync(base)
    .filter((d) => d.startsWith("chromium_headless_shell") || d.startsWith("chromium-"))
    .map((d) =>
      d.startsWith("chromium_headless_shell")
        ? path.join(base, d, "chrome-headless-shell-linux64", "chrome-headless-shell")
        : path.join(base, d, "chrome-linux", "chrome"),
    )
    .filter((p) => fs.existsSync(p));
  if (!cands.length) throw new Error("no cached chromium found under ~/.cache/ms-playwright");
  return cands[0];
}

function loadPlaywright() {
  const require = createRequire(import.meta.url);
  const candidates = [
    path.join(ROOT, "node_modules", "playwright-core"),
    "playwright-core",
    path.join(process.env.HOME, "code", "latent-scope-frontend", "node_modules", "playwright-core"),
  ];
  for (const c of candidates) {
    try {
      return require(c);
    } catch {
      /* next */
    }
  }
  throw new Error("playwright-core not found");
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  let server = null;
  let base = EXT_URL;
  if (!base) {
    const started = await startServer(HONOR_RANGE);
    server = started.server;
    base = `http://127.0.0.1:${started.port}/`;
  }
  if (!base.endsWith("/")) base += "/";
  if (MAP_ID) base += `?map=${encodeURIComponent(MAP_ID)}`;
  console.log(`driving ${base}`);
  const { chromium } = loadPlaywright();

  // libasound lives in a user-local extract; the shell dlopens it at startup
  const ldPaths = [path.join(process.env.HOME, ".cache", "lib"), process.env.LD_LIBRARY_PATH]
    .filter(Boolean)
    .join(":");

  const browser = await chromium.launch({
    executablePath: findChromeShell(),
    env: { ...process.env, LD_LIBRARY_PATH: ldPaths },
    args: ["--no-sandbox", "--disable-gpu", "--force-color-profile=srgb"],
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  // long-task recorder: the frame-jank evidence for the point layers
  await page.addInitScript(() => {
    window.__longTasks = [];
    try {
      new PerformanceObserver((list) => {
        for (const e of list.getEntries())
          window.__longTasks.push({ start: e.startTime, dur: e.duration });
      }).observe({ entryTypes: ["longtask"] });
    } catch {
      /* longtask unsupported */
    }
    window.__resetLongTasks = () => {
      window.__longTasks.length = 0;
    };
  });

  const errors = [];
  const warnings = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(`console.error: ${msg.text()}`);
    else if (msg.type() === "warning") warnings.push(`console.warn: ${msg.text()}`);
  });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("requestfailed", (r) => {
    const why = r.failure()?.errorText ?? "";
    // a deliberately cancelled range probe is not an error
    const line = `requestfailed: ${r.url()} — ${why}`;
    if (why.includes("ERR_ABORTED")) warnings.push(line);
    else errors.push(line);
  });
  page.on("response", (r) => {
    if (r.status() >= 400) errors.push(`http ${r.status()}: ${r.url()}`);
  });

  const checks = [];
  const check = (name, ok, detail = "") => {
    checks.push({ name, ok, detail });
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
  };

  await page.goto(base, { waitUntil: "networkidle" });
  await page.waitForFunction(
    () => document.getElementById("boot")?.classList.contains("hidden"),
    { timeout: 20000 },
  );
  await sleep(900);

  const mode = await page.evaluate(
    () => Object.values(window.mapviewerConfig.sources)[0]?.rangeMode,
  );
  check("range mode detected", !!mode, `${mode} (server honorRange=${HONOR_RANGE})`);
  if (!EXT_URL) {
    check(
      "range mode matches server",
      HONOR_RANGE ? mode === "http-range" : mode === "chunked",
      mode,
    );
  }

  const backend = await page.evaluate(() => window.__mapviewer.backend);
  check("base layer backend", /webgl2/.test(backend) || /canvas-2d/.test(backend), backend);

  // density actually painted? (reads the base framebuffer, GL or 2D)
  const painted = await page.evaluate(() => window.__mapviewer.samplePixels());
  check("density raster painted", painted > 20, `${painted} distinct colours`);

  await page.screenshot({ path: path.join(OUT, "01-initial.png") });

  // hover: walk a coarse grid until we land on a non-empty bin
  const box = await page.locator("#map").boundingBox();
  let sideText = "";
  let hoverAt = null;
  scan: for (const fy of [0.35, 0.5, 0.28, 0.65, 0.45]) {
    for (const fx of [0.35, 0.5, 0.28, 0.62, 0.45]) {
      await page.mouse.move(box.x + box.width * fx, box.y + box.height * fy);
      await sleep(450);
      sideText = await page.locator("#side").innerText();
      const m = /rows in bin\s*\n?\s*([\d,]+)/i.exec(sideText);
      if (m && Number(m[1].replace(/,/g, "")) > 0) {
        hoverAt = [fx, fy];
        break scan;
      }
    }
  }
  await sleep(900);
  sideText = await page.locator("#side").innerText();
  check("hover bin readout", /rows in bin/i.test(sideText), `at ${hoverAt}`);
  check("bin composition percentages", /\d+%/.test(sideText));
  check("bin snippets", /sampled snippets/i.test(sideText));
  await page.screenshot({ path: path.join(OUT, "02-hover.png") });

  // corpus toggle
  const firstCheck = page.locator("#controls .group input[type=checkbox]").first();
  await firstCheck.uncheck();
  await sleep(700);
  await page.screenshot({ path: path.join(OUT, "03-corpus-off.png") });
  await firstCheck.check();
  await sleep(500);

  // dominant-corpus mode
  await page.locator("#controls input[type=radio]").nth(1).check();
  await sleep(900);
  await page.screenshot({ path: path.join(OUT, "04-dominant.png") });
  await page.locator("#controls input[type=radio]").nth(0).check();
  await sleep(600);

  // point mode
  const lodBtn = page.locator("#ask-lod button");
  if (await lodBtn.count()) {
    await lodBtn.click();
    await page.waitForFunction(
      () => document.querySelector("#ask-lod")?.classList.contains("on"),
      { timeout: 20000 },
    );
    await sleep(600);
    check("LOD point mode enabled", true);
    // zoom in so points are picky-able
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    for (let i = 0; i < 6; i++) {
      await page.mouse.wheel(0, -240);
      await sleep(160);
    }
    await sleep(900);
    await page.screenshot({ path: path.join(OUT, "05-points.png") });
  } else {
    check("LOD point mode enabled", false, "no ask-lod control");
  }

  // -------------------------------------------------------------------------
  // v2 check 1 — hovering must not touch the base layer
  //
  // This is the flicker fix, stated as an assertion: 2 s of mousemove across
  // the map, with the LOD point layer live, must produce zero base-layer draws,
  // zero texture uploads and zero point-buffer uploads.
  // -------------------------------------------------------------------------
  await sleep(1200); // let any in-flight tiles settle first
  await page.evaluate(() => window.__resetRenderStats());
  const sweepStart = Date.now();
  let sweeps = 0;
  while (Date.now() - sweepStart < 2000) {
    const t = ((Date.now() - sweepStart) / 2000) * Math.PI * 2;
    await page.mouse.move(
      box.x + box.width * (0.5 + 0.32 * Math.cos(t)),
      box.y + box.height * (0.5 + 0.28 * Math.sin(t * 1.7)),
      { steps: 6 },
    );
    sweeps++;
    await sleep(25);
  }
  const sweep = await page.evaluate(() => ({ ...window.__renderStats }));
  check(
    "mousemove: zero base-layer redraws",
    sweep.baseDraws === 0 && sweep.tileUploads === 0 && sweep.pointUploads === 0,
    `baseDraws=${sweep.baseDraws} tileUploads=${sweep.tileUploads} ` +
      `pointUploads=${sweep.pointUploads} overlayDraws=${sweep.overlayDraws} ` +
      `(${sweeps} moves / 2 s)`,
  );
  check(
    "mousemove: overlay layer did redraw",
    sweep.overlayDraws > 0,
    `${sweep.overlayDraws} overlay draws, max ${sweep.maxOverlayMs.toFixed(2)} ms`,
  );

  // -------------------------------------------------------------------------
  // v2 check 2 — panning with the point layer live must not drop frames
  // -------------------------------------------------------------------------
  await page.evaluate(() => {
    window.__resetRenderStats();
    window.__resetLongTasks();
  });
  const cx0 = box.x + box.width / 2;
  const cy0 = box.y + box.height / 2;
  await page.mouse.move(cx0, cy0);
  await page.mouse.down();
  for (let i = 0; i < 40; i++) {
    await page.mouse.move(cx0 + Math.sin(i / 5) * 160, cy0 + Math.cos(i / 7) * 110, {
      steps: 3,
    });
    await sleep(20);
  }
  await page.mouse.up();
  await sleep(400);
  const pan = await page.evaluate(() => ({
    stats: { ...window.__renderStats },
    long: window.__longTasks.filter((t) => t.dur > 50),
    worst: window.__longTasks.reduce((m, t) => Math.max(m, t.dur), 0),
  }));
  check(
    "pan with LOD points: no long task > 50 ms",
    pan.long.length === 0,
    `${pan.long.length} long tasks (worst ${pan.worst.toFixed(0)} ms), ` +
      `${pan.stats.baseDraws} base draws, max base draw ` +
      `${pan.stats.maxBaseMs.toFixed(1)} ms, ${pan.stats.pointUploads} point uploads`,
  );
  check(
    "pan re-used the uploaded point buffer",
    pan.stats.pointUploads === 0,
    `${pan.stats.pointUploads} uploads during pan`,
  );
  await page.screenshot({ path: path.join(OUT, "05b-after-pan.png") });

  // deep points
  const deepBtn = page.locator("#ask-deep button");
  if (await deepBtn.count()) {
    await deepBtn.click();
    await page.waitForFunction(
      () => document.querySelector("#ask-deep")?.classList.contains("on"),
      { timeout: 20000 },
    );
    await sleep(1500);
    check("deep point mode enabled", true);
    await page.screenshot({ path: path.join(OUT, "06-deep-points.png") });
  } else {
    check("deep point mode enabled", false, "no ask-deep control");
  }

  // click a point -> full text via sidecar (skipped when the pack ships none)
  let gotText = false;
  let sidecarMissing = false;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  outer: for (let dy = -60; dy <= 60 && !gotText; dy += 12) {
    for (let dx = -60; dx <= 60; dx += 12) {
      await page.mouse.click(cx + dx, cy + dy);
      await sleep(230);
      const t = await page.locator("#side").innerText();
      if (/selected point/i.test(t) && /row id/i.test(t)) {
        if (/no text sidecar reachable/i.test(t)) {
          sidecarMissing = true;
          break outer;
        }
        gotText = /fetched/i.test(t);
        if (gotText) break outer;
      }
    }
  }
  if (sidecarMissing) {
    console.log("SKIP  click point -> sidecar text — pack publishes no text sidecar");
    check("click point -> point selected", true, "text sidecar not published");
  } else {
    check("click point -> sidecar text", gotText);
  }
  await page.screenshot({ path: path.join(OUT, "07-point-text.png") });

  // -------------------------------------------------------------------------
  // v2 check 3 — in-browser projection
  //
  // Loads the vendored MiniLM + this pack's map head, projects the POC's five
  // test strings, and gates each landing against the python reference. Also
  // asserts the dot is painted on the OVERLAY canvas (not the base layer).
  // -------------------------------------------------------------------------
  const hasProjection = await page.evaluate(() => window.__mapviewer.projection.available());
  if (WITH_PROJECTION && hasProjection) {
    console.log(`loading projection models (${PROJECTION_PRECISION}) …`);
    await page.evaluate((p) => window.__mapviewer.projection.setPrecision(p), PROJECTION_PRECISION);
    await sleep(150);
    await page.screenshot({ path: path.join(OUT, "08b-projection-ask.png") });
    const t0 = Date.now();
    // drive the real explicit-ask button, not the test hook
    let loadErr = null;
    try {
      await page.locator("#ask-projection button").click();
      await page.waitForFunction(() => window.__mapviewer.projection.loaded(), {
        timeout: 180000,
      });
    } catch (e) {
      loadErr = String(e);
    }
    const loadedBytes = await page.evaluate(() => window.__mapviewer.projection.bytes());
    check(
      "projection models loaded",
      loadErr === null && loadedBytes > 60e6,
      loadErr ??
        `${PROJECTION_PRECISION}, ${(loadedBytes / 1048576).toFixed(1)} MB of weights ` +
          `in ${((Date.now() - t0) / 1000).toFixed(1)}s`,
    );

    if (!loadErr) {
      const gate = PROJECTION_REFERENCE.gateFrac * PROJECTION_REFERENCE.extentDiagonal;

      // first string through the real UI: type it, press the button
      await page.locator("#projection-text").fill(PROJECTION_REFERENCE.rows[0].text);
      await page.locator("#projection-run").click();
      await page.waitForFunction(() => window.__mapviewer.projection.markers().length >= 1, {
        timeout: 60000,
      });
      const uiOut = await page.locator("#projection-out").innerText();
      check(
        "projection via the panel UI (type + Project)",
        /u,v=/.test(uiOut) && /tokens/.test(uiOut),
        uiOut.replace(/\s+/g, " ").slice(0, 120),
      );

      // the rest through the hook, so the reference sweep stays quick
      const results = await page.evaluate(async (rows) => {
        const done = window.__mapviewer.projection.markers();
        const out = [...done];
        for (const r of rows.slice(done.length))
          out.push(await window.__mapviewer.projection.project(r.text));
        return out;
      }, PROJECTION_REFERENCE.rows);

      let worst = 0;
      let worstText = "";
      let frameOk = true;
      const extent = await page.evaluate(() => window.__mapviewer.projection.extent());
      const mapId = await page.evaluate(() => window.__mapviewer.mapId());
      const refApplies = mapId === PROJECTION_REFERENCE.mapId;
      for (let i = 0; i < results.length; i++) {
        const ref = PROJECTION_REFERENCE.rows[i].xy;
        const got = results[i];
        const d = Math.hypot(got.x - ref[0], got.y - ref[1]);
        if (d > worst) {
          worst = d;
          worstText = PROJECTION_REFERENCE.rows[i].text.slice(0, 28);
        }
        // frame contract: the viewer places the dot with the PACK extent,
        // u right / v DOWN, and no 180-degree PNG rotation
        const u = (got.x - extent.xmin) / (extent.xmax - extent.xmin);
        const v = (extent.ymax - got.y) / (extent.ymax - extent.ymin);
        if (Math.abs(u - got.u) > 1e-9 || Math.abs(v - got.v) > 1e-9) frameOk = false;
      }
      if (refApplies) {
        check(
          "projection matches python reference",
          worst <= gate,
          `worst Δxy = ${worst.toFixed(4)} (${(
            (worst / PROJECTION_REFERENCE.extentDiagonal) *
            100
          ).toFixed(3)}% of extent diag, gate ${gate.toFixed(3)}) on "${worstText}"`,
        );
      } else {
        console.log(
          `SKIP  projection vs python reference — reference is for ` +
            `${PROJECTION_REFERENCE.mapId}, this pack is ${mapId} ` +
            `(different map head / coordinate space)`,
        );
      }
      check("projection frame contract (u right, v down, pack extent)", frameOk);

      // the dot must be on the overlay canvas at the projected position
      const last = results[results.length - 1];
      await page.evaluate((t) => window.__mapviewer.flyTo(t.u, t.v), last);
      await sleep(300);
      const dot = await page.evaluate(async (target) => {
        const mv = window.__mapviewer;
        const cam = mv.camera();
        const sx = (target.u - cam.cx) * cam.scale + cam.width / 2;
        const sy = (target.v - cam.cy) * cam.scale + cam.height / 2;
        const c = document.getElementById("overlay");
        const dpr = c.width / cam.width;
        const ctx = c.getContext("2d");
        const r = 16;
        const x0 = Math.max(0, Math.round((sx - r) * dpr));
        const y0 = Math.max(0, Math.round((sy - r) * dpr));
        const w = Math.min(c.width - x0, Math.round(2 * r * dpr));
        const hgt = Math.min(c.height - y0, Math.round(2 * r * dpr));
        if (w <= 0 || hgt <= 0) return { onScreen: false, opaque: 0, sx, sy };
        const d = ctx.getImageData(x0, y0, w, hgt).data;
        let opaque = 0;
        for (let i = 3; i < d.length; i += 4) if (d[i] > 40) opaque++;
        return { onScreen: true, opaque, sx, sy };
      }, last);
      check(
        "projection dot painted on the overlay layer",
        dot.onScreen && dot.opaque > 20,
        `${dot.opaque} opaque px within 16 px of (${dot.sx.toFixed(0)}, ${dot.sy.toFixed(0)})`,
      );

      const markers = await page.evaluate(() => window.__mapviewer.projection.markers().length);
      check("projection markers persist", markers === PROJECTION_REFERENCE.rows.length, `${markers} markers`);
      await page.screenshot({ path: path.join(OUT, "09-projection.png") });

      // Empirical frame check: a mirrored or flipped frame would drop these
      // in-distribution strings into empty space. Fly to each landing and read
      // the density bin the viewer itself resolves there.
      const landed = [];
      for (const r of results) {
        await page.evaluate((t) => window.__mapviewer.flyTo(t.u, t.v), r);
        await sleep(700);
        landed.push(
          await page.evaluate((t) => window.__mapviewer.binAtWorld(t.u, t.v)?.total ?? 0, r),
        );
      }
      const nonEmpty = landed.filter((n) => n > 0).length;
      check(
        "projections land in non-empty density bins",
        nonEmpty >= results.length - 1,
        `${nonEmpty}/${results.length} — rows in bin: ${landed.join(", ")}`,
      );
    }
  } else {
    console.log(
      `SKIP  projection — ${hasProjection ? "--no-projection" : "pack ships no map head"}`,
    );
  }

  // map switcher
  const opts = await page.locator("#mapselect option").count();
  if (opts > 1) {
    await page.selectOption("#mapselect", { index: 1 });
    await page.waitForFunction(
      () => document.getElementById("boot")?.classList.contains("hidden"),
      { timeout: 20000 },
    );
    await sleep(1200);
    check("map switcher", true, `${opts} packs`);
    await page.screenshot({ path: path.join(OUT, "08-second-pack.png") });
  } else {
    check("map switcher", opts === 1, `${opts} pack(s) in index`);
  }

  const bytes = await page.locator("#status").innerText();
  console.log(`\nsession status line: ${bytes}`);

  check("zero console errors", errors.length === 0, errors.slice(0, 5).join(" ;; "));

  await browser.close();
  server?.close();

  const failed = checks.filter((c) => !c.ok);
  console.log(`\n${checks.length - failed.length}/${checks.length} checks passed`);
  if (warnings.length) console.log(`warnings (non-fatal): ${warnings.length}`);
  console.log(`screenshots: ${OUT}`);
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
