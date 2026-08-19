/**
 * Headless verification of the in-browser projection POC.
 *
 * Serves this directory over http (python3 -m http.server), drives it with
 * playwright-core + the cached chrome-headless-shell, and asserts:
 *   - zero console errors / page errors / failed requests
 *   - every test string yields a finite numeric (x, y) and in-frame pixel
 * Writes reference/browser_results.json (embeddings + xy) and reference/poc.png,
 * which reference/reference_projection.py --compare then checks against
 * sentence-transformers + the torch map head.
 *
 * Run:
 *   source ~/.nvm/nvm.sh
 *   LD_LIBRARY_PATH=/tmp/libs/extracted/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
 *     node verify_headless.mjs [--encoder onnx/model_quantized.onnx]
 */
import { chromium } from 'playwright-core';
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.POC_PORT || 8971);
const SHELL = join(process.env.HOME, '.cache/ms-playwright/chromium_headless_shell-1223',
                   'chrome-headless-shell-linux64/chrome-headless-shell');
const argEncoder = process.argv.indexOf('--encoder');
const ENCODER = argEncoder > -1 ? process.argv[argEncoder + 1] : 'onnx/model.onnx';
const OUT = process.env.POC_OUT || join(DIR, 'reference/browser_results.json');

const TEXTS = [
  'the quick brown fox jumps over the lazy dog',
  'Photosynthesis converts light energy into chemical energy in plants.',
  'def quicksort(a):\n    if len(a) <= 1: return a',
  'Interest rates rose sharply after the central bank meeting.',
  'Ich habe gestern ein sehr gutes Buch gelesen.',
];

const server = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'],
                     { cwd: DIR, stdio: 'ignore' });
const shutdown = () => { try { server.kill('SIGTERM'); } catch {} };
process.on('exit', shutdown);

const failures = [];
let exitCode = 0;
try {
  await new Promise((r) => setTimeout(r, 800));
  const browser = await chromium.launch({ executablePath: SHELL, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });

  const consoleErrors = [], pageErrors = [], badRequests = [];
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', (e) => pageErrors.push(String(e)));
  page.on('requestfailed', (r) => badRequests.push(`${r.url()} ${r.failure()?.errorText}`));
  page.on('response', (r) => { if (r.status() >= 400) badRequests.push(`${r.url()} HTTP ${r.status()}`); });

  await page.goto(`http://127.0.0.1:${PORT}/index.html`, { waitUntil: 'load', timeout: 60000 });
  await page.waitForFunction('window.poc !== undefined', null, { timeout: 30000 });
  await page.evaluate('window.poc.ready');

  await page.evaluate((e) => window.poc.setEncoder(e), ENCODER);
  const t0 = Date.now();
  await page.evaluate('window.poc.load()', null, { timeout: 300000 });
  const loadSeconds = (Date.now() - t0) / 1000;

  const results = [];
  for (const text of TEXTS) {
    const r = await page.evaluate((t) => window.poc.project(t), text);
    if (!Number.isFinite(r.x) || !Number.isFinite(r.y)) failures.push(`non-numeric xy for ${text}`);
    if (!(r.px >= 0 && r.px <= 1024 && r.py >= 0 && r.py <= 1024))
      failures.push(`pixel outside the frame for ${text}: ${r.px},${r.py}`);
    if (r.embedding.length !== 384) failures.push(`embedding dim ${r.embedding.length}`);
    results.push({ text, xy: [r.x, r.y], pixel: [r.px, r.py], uv: [r.u, r.v],
                   quantized: [r.qx, r.qy], tokens: r.tokens,
                   embed_ms: r.embed_ms, head_ms: r.head_ms, embedding: r.embedding });
    console.log(`  (${r.x.toFixed(4)}, ${r.y.toFixed(4)})  px=(${r.px.toFixed(0)},${r.py.toFixed(0)})` +
                `  embed ${r.embed_ms.toFixed(0)}ms head ${r.head_ms.toFixed(1)}ms  ${JSON.stringify(text.slice(0, 40))}`);
  }

  // draw the last point and capture the page
  await page.click('#project');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: join(DIR, 'reference/poc.png'), fullPage: true });

  const inPageErrors = await page.evaluate('window.poc.errors');
  for (const [label, arr] of [['console', consoleErrors], ['page', pageErrors],
                              ['request', badRequests], ['in-page', inPageErrors]]) {
    if (arr.length) failures.push(`${label} errors: ${JSON.stringify(arr)}`);
  }

  writeFileSync(OUT, JSON.stringify({
    encoder_variant: ENCODER, load_seconds: loadSeconds,
    console_errors: consoleErrors, page_errors: pageErrors, request_errors: badRequests,
    results,
  }, null, 1) + '\n');

  console.log(`\nmodels loaded in ${loadSeconds.toFixed(1)}s; wrote ${OUT}`);
  await browser.close();
} catch (e) {
  failures.push(String(e));
}

if (failures.length) { console.error('FAIL\n' + failures.join('\n')); exitCode = 1; }
else console.log('PASS — numeric (x, y) for every string, zero console errors');
shutdown();
process.exit(exitCode);
