#!/usr/bin/env node
/**
 * smoke.mjs — headless smoke check for the built viewer.
 *
 * Serves `dist/` (with `packs/` mounted alongside) over a plain static server
 * that can be configured to honour or ignore HTTP Range, drives the page with
 * playwright-core + the cached chrome-headless-shell, and asserts:
 *   - zero console errors / page errors / failed requests
 *   - the density raster actually painted (canvas is not uniform background)
 *   - hover produces a bin readout with corpus composition
 *   - point mode loads and a point click resolves text through the sidecar
 *
 * Usage:
 *   node scripts/smoke.mjs                 # chunked mode (Range ignored, like gsv:8800)
 *   node scripts/smoke.mjs --range         # http-range mode
 *   node scripts/smoke.mjs --out shots/    # where screenshots land
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

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
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

  // density actually painted? (sample every 4th pixel, count distinct colours)
  const painted = await page.evaluate(() => {
    const c = document.getElementById("map");
    const ctx = c.getContext("2d");
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    const seen = new Set();
    for (let i = 0; i < d.length; i += 16)
      seen.add(`${d[i]},${d[i + 1]},${d[i + 2]}`);
    return seen.size;
  });
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
