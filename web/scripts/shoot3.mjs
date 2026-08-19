// v3 playwright verification: gallery sort/filter, hover-cell alignment at deep
// zoom, and the deep-zoom path (tiled if live, else the graceful 1024 cap).
// Tracks console/page errors + every 404 on both an atlas and a projection route.
import { chromium } from "playwright-core";

const EXE = process.env.HOME +
  "/.cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell";
const BASE = "http://gsv.local:8800/basemap-maps/app/index.html";
// round-0102 (150M) has LIVE tiled_levels (2048/4096 tiles) — exercises real deep zoom.
const ATLAS = "round-0102-r0101-balanced-150m-seed42";
const PROJ = "round-0102-r0101-balanced-150m-seed42-dadabase-projection";
const OUT = "/tmp/claude-1000/mapviz-shots";

const errors = [];
const notFound = [];
const gridReqs = new Set();
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({ executablePath: EXE, args: ["--no-sandbox"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
let tag = "boot";
page.on("console", (m) => { if (m.type() === "error") errors.push(`[${tag}] console.error: ${m.text()}`); });
page.on("pageerror", (e) => errors.push(`[${tag}] pageerror: ${e.message}`));
page.on("response", (r) => { if (r.status() === 404) notFound.push(`[${tag}] 404 ${r.url()}`); });
page.on("request", (r) => { const m = r.url().match(/grid-all-(\d+(?:-\d+_\d+)?)\.bin/); if (m) gridReqs.add(m[1]); });

// (a) Gallery: Best-FFR sort + dadabase tag filter, driven from the hash query.
tag = "gallery";
await page.goto(`${BASE}#/?sort=ffr&tag=dadabase`, { waitUntil: "load" });
await wait(2000);
await page.screenshot({ path: `${OUT}/v3-a-gallery-ffr-dadabase.png` });
const galleryCount = await page.locator(".gc-count").innerText().catch(() => "n/a");
const firstCard = await page.locator(".card h3").first().innerText().catch(() => "n/a");
console.log("gallery filtered:", galleryCount, "| first card:", firstCard);

// (b) Atlas hover tooltip + visible cell highlight, verified at 3 zoom depths.
tag = "atlas";
await page.goto(`${BASE}#/map/${ATLAS}`, { waitUntil: "load" });
await page.waitForSelector("#plot", { timeout: 15000 });
await wait(3500);
const box = await page.locator("#plot").boundingBox();
const cx = box.x + box.width / 2, cy = box.y + box.height / 2;

// Scan a grid of canvas points, pick the DENSEST bin that shows a tooltip,
// hover it, and screenshot. Returns the chosen [px,py] so the next (deeper)
// zoom can anchor on a populated region.
async function densestPoint(coarse) {
  const cols = coarse ? 13 : 9, rows = coarse ? 9 : 7;
  let best = null, bestCount = -1;
  for (let r = 1; r < rows; r++)
    for (let c = 1; c < cols; c++) {
      const px = box.x + (box.width * c) / cols, py = box.y + (box.height * r) / rows;
      await page.mouse.move(px, py);
      await wait(70);
      const el = page.locator(".tooltip:not([hidden]) .tt-count");
      if (await el.count()) {
        const n = parseInt((await el.innerText()).replace(/[^0-9]/g, ""), 10) || 1;
        if (n > bestCount) { bestCount = n; best = [px, py]; }
      }
    }
  return best;
}
async function hoverShoot(name, pt) {
  if (pt) { await page.mouse.move(pt[0], pt[1]); await wait(500); }
  await page.screenshot({ path: `${OUT}/${name}.png` });
  const tt = await page.locator(".tooltip .tt-count").innerText().catch(() => "NONE");
  const cap = await page.locator(".tooltip .tt-caption").count();
  console.log(`${name}: tooltip="${pt ? tt : "NONE"}" caption=${cap} at=${pt ? pt.map(Math.round) : "-"}`);
}

// depth 1: shallow (renders ~256/512 — finer than the old fixed-256 hover).
await page.mouse.move(cx, cy);
for (let i = 0; i < 14; i++) { await page.mouse.wheel(0, -120); await wait(20); }
await wait(1000);
let pt = await densestPoint(true);
await hoverShoot("v3-b1-hover-shallow", pt);

// depth 2: medium (renders ~1024). Anchor deeper zoom on the densest point.
if (pt) await page.mouse.move(pt[0], pt[1]);
for (let i = 0; i < 16; i++) { await page.mouse.wheel(0, -120); await wait(20); }
await wait(1000);
pt = await densestPoint(false);
await hoverShoot("v3-b2-hover-medium", pt);

// depth 3: deep (renders tiled 2048/4096). Anchor on the densest medium cell so
// the deep viewport lands on populated data, not the sparse map center.
if (pt) await page.mouse.move(pt[0], pt[1]);
for (let i = 0; i < 22; i++) { await page.mouse.wheel(0, -120); await wait(20); }
await wait(1200);
pt = await densestPoint(false);
await hoverShoot("v3-b3-hover-deep", pt);

// alias the deep one as the required (b) name; the (c) deep-zoom view is (b3).
await page.screenshot({ path: `${OUT}/v3-b-atlas-hover-deepzoom.png` });
await page.screenshot({ path: `${OUT}/v3-c-atlas-deepest.png` });

// Projection route error/404 sweep.
tag = "projection";
await page.goto(`${BASE}#/map/${PROJ}`, { waitUntil: "load" });
await page.waitForSelector("#plot", { timeout: 15000 });
await wait(3000);
await page.screenshot({ path: `${OUT}/v3-d-projection.png` });

await browser.close();
console.log("SCREENSHOTS DONE");
console.log("grid/tile files fetched:", [...gridReqs].sort().join(", "));
console.log("errors:", errors.length);
for (const e of errors) console.log("  " + e);
console.log("404s:", notFound.length);
for (const u of notFound) console.log("  " + u);
