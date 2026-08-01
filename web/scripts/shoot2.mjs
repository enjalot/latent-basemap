// Regression smoke after the projection-map fixes: gallery (index now exists),
// the projection route (point-only manifest), and the atlas route (grid manifest).
// Tracks per-route console/page errors AND every 404'd request URL.
import { chromium } from "playwright-core";

const EXE = process.env.HOME +
  "/.cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell";
const BASE = "http://gsv.local:8800/basemap-maps/app/index.html";
const ATLAS = "round-0108-r0107-diverse-jina-25m-seed42";
const PROJ = "round-0108-r0107-diverse-jina-25m-seed42-pol-latn-projection";
const OUT = "/tmp/claude-1000/mapviz-shots";

const errors = [];
const notFound = [];
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({ executablePath: EXE, args: ["--no-sandbox"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
let tag = "boot";
page.on("console", (m) => { if (m.type() === "error") errors.push(`[${tag}] console.error: ${m.text()}`); });
page.on("pageerror", (e) => errors.push(`[${tag}] pageerror: ${e.message}`));
page.on("response", (r) => { if (r.status() === 404) notFound.push(`[${tag}] 404 ${r.url()}`); });

// 1) Gallery — maps-index.json exists now (60 maps)
tag = "gallery";
await page.goto(`${BASE}#/`, { waitUntil: "load" });
await wait(2500);
await page.screenshot({ path: `${OUT}/react-fix-1-gallery.png` });

// 2) Projection route — point-only manifest (the bug report)
tag = "projection";
await page.goto(`${BASE}#/map/${PROJ}`, { waitUntil: "load" });
await page.waitForSelector("#plot", { timeout: 15000 });
await wait(3000); // points-*.bin fetches
// hover an accent point region (canvas center-right) to exercise point hover
const box = await page.locator("#plot").boundingBox();
await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.45);
await wait(800);
await page.screenshot({ path: `${OUT}/react-fix-2-projection.png` });

// 2b) projection metrics (queries only — no anchors)
await page.locator(".tabs button", { hasText: "Metrics" }).click();
await wait(1500);
const probeBtns = page.locator(".plist button");
if (await probeBtns.count()) {
  await probeBtns.first().click();
  await wait(1200);
  const qBtns = page.locator(".plist.qscroll button");
  if (await qBtns.count()) { await qBtns.first().click(); await wait(1000); }
}
await page.screenshot({ path: `${OUT}/react-fix-3-projection-metrics.png` });

// 3) Atlas route still works
tag = "atlas";
await page.goto(`${BASE}#/map/${ATLAS}`, { waitUntil: "load" });
await page.waitForSelector("#plot", { timeout: 15000 });
await wait(3000);
const box2 = await page.locator("#plot").boundingBox();
await page.mouse.move(box2.x + box2.width / 2, box2.y + box2.height / 2);
await wait(1200);
await page.screenshot({ path: `${OUT}/react-fix-4-atlas.png` });

await browser.close();
console.log("SCREENSHOTS DONE");
console.log("errors:", errors.length);
for (const e of errors) console.log("  " + e);
console.log("404s:", notFound.length);
for (const u of notFound) console.log("  " + u);
