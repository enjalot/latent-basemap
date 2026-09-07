import assert from 'node:assert/strict';
import {chromium} from '/home/enjalot/code/latent-basemap/web/node_modules/playwright-core/index.mjs';
const base='http://gsv.local:8800/basemap-maps/';
const browser=await chromium.launch({executablePath:'/home/enjalot/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome',headless:true,args:['--no-sandbox']});
try {
const page=await browser.newPage({viewport:{width:1600,height:1200}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
page.on('response',r=>{if(r.status()>=400)errors.push(r.status()+' '+r.url())});
await page.goto(base+'latest-images/',{waitUntil:'networkidle'});
const catalog=await (await page.request.get(base+'latest-images/catalog.json')).json();
assert.equal(catalog.length,7);
assert(catalog.some(m=>m.key==='dino-6m-104m'));
assert(catalog.some(m=>m.key==='dino-6m-pca768-104m'));
await page.getByRole('button',{name:'103.8M · 6M full vs PCA',exact:true}).click();
assert.equal(await page.locator('#left').inputValue(),'dino-6m-104m');
assert.equal(await page.locator('#right').inputValue(),'dino-6m-pca768-104m');
for(const side of ['left','right'])await page.frameLocator('#'+side+'-frame').locator('#status').waitFor({state:'hidden'});
await page.screenshot({path:'/tmp/image-comparison-completed-full.png',fullPage:true});
await page.locator('#ladder').click();await page.waitForLoadState('networkidle');
assert.equal(await page.locator('#left').inputValue(),'dino-6m');
assert.equal(await page.locator('#right').inputValue(),'dino-6m-pca768');
for(const side of ['left','right'])await page.frameLocator('#'+side+'-frame').locator('#status').waitFor({state:'hidden'});
await page.screenshot({path:'/tmp/image-comparison-completed-ladder.png',fullPage:true});
for(const m of catalog){
  const r=await page.goto(new URL(m.url,base+'latest-images/').href,{waitUntil:'networkidle'});assert(r.ok());
  await page.locator('#status').waitFor({state:'hidden'});
  assert((await page.locator('#rows').innerText()).includes(m.rows.toLocaleString('en-US')));
  await page.locator('#zoomIn').click();await page.locator('#zoomIn').click();await page.waitForLoadState('networkidle');
  await page.locator('#zoomReset').click();
  const canvas=page.locator('#plot');const box=await canvas.boundingBox();
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+box.width/2+60,box.y+box.height/2+30,{steps:6});await page.mouse.up();
}
await page.setViewportSize({width:390,height:850});
await page.goto(base+'latest-images/?left=dino-6m&right=dino-6m-pca768',{waitUntil:'networkidle'});
assert.equal(await page.locator('#left').inputValue(),'dino-6m');
assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
await page.screenshot({path:'/tmp/image-comparison-completed-mobile.png',fullPage:true});
assert.deepEqual(errors,[]);
console.log('PASS: seven maps, full-corpus and ladder pairs, deep links, zoom/pan, mobile layout, no browser/HTTP errors');
}finally{await browser.close()}
