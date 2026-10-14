import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";

export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:5173",
    viewport: { width: 1440, height: 960 },
    launchOptions: process.env.CHROMIUM_EXECUTABLE_PATH
      ? {
          executablePath: process.env.CHROMIUM_EXECUTABLE_PATH,
          args: [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--single-process",
            "--no-zygote",
            "--in-process-gpu",
            "--font-render-hinting=none",
          ],
        }
      : {},
  },
  webServer: [
    {
      command:
        "python -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000",
      cwd: resolve(import.meta.dirname, ".."),
      url: "http://127.0.0.1:8000/health",
      env: {
        RAG_MODE: "demo",
        RAG_DB_PATH: resolve(import.meta.dirname, "../runtime/e2e.sqlite3"),
        RAG_RERANKER: "none",
      },
      reuseExistingServer: false,
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: false,
    },
  ],
});
