import { defineConfig } from "vite";

// base "./" keeps every emitted asset path relative, so the same build works
// from gh-pages (/<repo>/mapviewer/), from http://gsv.local:8800/basemap-maps/
// viewer/, and from file:// preview alike.
export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    outDir: "dist",
    assetsDir: "assets",
    sourcemap: false,
  },
  server: {
    host: true,
    port: 5195,
  },
  worker: {
    format: "es",
  },
});
