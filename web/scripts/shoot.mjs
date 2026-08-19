// Playwright smoke: load the DEPLOYED app off the real static site and screenshot
// four states. Prints any console errors. Run via scripts run below (needs the
// libasound LD_LIBRARY_PATH workaround + headless-shell, see the gsv memory).
import { chromium } from "playwright-core";

const EXE = process.env.HOME +
  "/.cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell";
const BASE = "http://gsv.local:8800/basemap-maps/app/index.html";
const MAP = "round-0108-r0107-diverse-jina-25m-seed42";
const OUT = "/tmp/claude-1000/mapviz-shots";

const errors = [];
function watch(page, tag) {
  page.on("console", (m) => { if (m.type() === "error") errors.push(`[${tag}] console.error: ${m.text()}`); });
  page.on("pageerror", (e) => errors.push(`[${tag}] pageerror: ${e.message}`));
}
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({ executablePath: EXE, args: ["--no-sandbox"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
watch(page, "all");

// 1) Gallery (maps-index.json 404s -> designed empty state)
await page.goto(`${BASE}#/`, { waitUntil: "load" });
await wait(800);
await page.screenshot({ path: `${OUT}/react-1-gallery.png` });

// 2) Viewer, default map mode (density canvas + legend panel). Hover for tooltip.
await page.goto(`${BASE}#/map/${MAP}`, { waitUntil: "load" });
await page.waitForSelector("#plot", { timeout: 15000 });
await wait(2500); // let grid-256 fetch + first paint settle
// hover the canvas center to trigger a bin tooltip
const box = await page.locator("#plot").boundingBox();
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
await wait(1200);
await page.screenshot({ path: `${OUT}/react-2-viewer-map.png` });

// 3) Legend as control: toggle a corpus grid overlay + a held-out point layer.
const rows = page.locator(".layer-row");
const nRows = await rows.count();
let clicked = 0;
for (let i = 0; i < nRows && clicked < 1; i++) {
  const label = (await rows.nth(i).locator(".lr-label").innerText()).toLowerCase();
  if (label.includes("fineweb") || label.includes("english")) { await rows.nth(i).click(); clicked++; }
}
// toggle a probe/point row (Held-out & OOD section)
for (let i = 0; i < nRows; i++) {
  const cls = await rows.nth(i).locator(".lr-swatch").getAttribute("class");
  if (cls && cls.includes("dot")) { await rows.nth(i).click(); break; }
}
await wait(1500);
await page.screenshot({ path: `${OUT}/react-3-overlay.png` });

// 4) Metrics tab: anchors, then query explorer.
await page.locator(".tabs button", { hasText: "Metrics" }).click();
await wait(2000);
await page.screenshot({ path: `${OUT}/react-4-metrics-anchors.png` });

// query explorer: switch to Held-out queries, pick a probe, pick a query
await page.locator(".seg button", { hasText: "Held-out queries" }).click();
await wait(1500);
const probeBtns = page.locator(".plist button");
if (await probeBtns.count()) {
  await probeBtns.first().click();
  await wait(1500);
  const qBtns = page.locator(".plist.qscroll button");
  if (await qBtns.count()) { await qBtns.first().click(); await wait(1200); }
}
await page.screenshot({ path: `${OUT}/react-5-metrics-queries.png` });

await browser.close();
console.log("SCREENSHOTS DONE");
console.log("console/page errors:", errors.length);
for (const e of errors) console.log("  " + e);
