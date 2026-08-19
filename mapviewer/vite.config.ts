import { defineConfig } from "vite";

// base "./" keeps every emitted asset path relative, so the same build works
// from gh-pages (/<repo>/mapviewer/), from http://gsv.local:8800/basemap-maps/
// viewer/, and from file:// preview alike.
export default defineConfig({
  base: "./",
  // public/ holds symlinks to projection-poc/{vendor,models} (122 MB of ONNX +
  // WASM) so `npm run dev` serves them at /vendor and /models — the paths
  // src/projection.ts resolves against document.baseURI. They are NOT copied
  // into dist/: scripts/deploy.sh rsyncs them into the published site instead,
  // which keeps `vite build` from copying 122 MB on every rebuild.
  publicDir: "public",
  build: {
    target: "es2022",
    outDir: "dist",
    assetsDir: "assets",
    sourcemap: false,
    copyPublicDir: false,
  },
  server: {
    host: true,
    port: 5195,
  },
  worker: {
    format: "es",
  },
});
