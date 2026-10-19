import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.FLOW_BACKEND || "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 900 },
  server: {
    proxy: { "/api": backend, "/shared": backend, "/vendor": backend, "/studio": backend, "/health": backend },
  },
});
