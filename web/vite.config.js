import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base './' so the built app works when deployed under /basemap-maps/app/ on the
// plain python static server (no path rewrites, no CDN).
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { port: 5196, strictPort: true },
  test: {
    environment: "node",
    include: ["src/**/*.test.{js,jsx}"],
  },
});
