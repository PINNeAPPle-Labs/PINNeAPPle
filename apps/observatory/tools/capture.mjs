// Screenshots of the Observatory for the docs (docs/assets/apps/14-observatory-*.jpg), in English and Portuguese.
//   cd apps/observatory && sh tools/serve_test.sh 8797 &      (offline server: local TLE, no warm-up)
//   node tools/capture.mjs [http://localhost:8797] [name ...]
// Each shot is a URL (fixed time and place, frozen clock) plus an optional action; the images are 1600 x 900 JPEGs.
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";

const BASE = process.argv[2] && process.argv[2].startsWith("http") ? process.argv[2] : "http://localhost:8797";
const ONLY = process.argv.slice(2).filter((a) => !a.startsWith("http"));
const OUT = new URL("../../../docs/assets/apps/", import.meta.url).pathname;
const SP = "lat=-23.5505&lon=-46.6333&elev=760&tz=America/Sao_Paulo&name=saopaulo&freeze=1";

const SHOTS = [
  { name: "sky", url: `/?${SP}&t=2026-10-13T21:49:59Z&az=250&alt=11&fov=62&panel=0` },
  { name: "star-hr", url: `/?${SP}&t=2026-10-14T00:30:00Z&az=40&alt=40&fov=80&select=star:2061`, wait: "#oHR svg" },
  { name: "system", url: `/?${SP}&t=2026-10-13T23:30:00Z&view=system&tab=engine&focusDelay=99999`, wait: "#eOut .card", settle: 9000, timeout: 600000 },
  { name: "engine", url: `/?${SP}&t=2026-10-13T23:30:00Z&tab=engine&az=90&alt=30`, wait: "#eVal table", timeout: 600000 },
  { name: "learn", url: `/?${SP}&t=2026-10-13T23:30:00Z&learn=saturn&az=60&alt=35&fov=70`, wait: ".learn-card .lc-quiz" },
  { name: "observe", url: `/?${SP}&t=2026-10-14T02:30:00Z&observe=M45&eye=plossl32&az=40&alt=40&fov=60`, wait: "#obChart svg", click: "#obTrack", after: "#epCanvas", settle: 4000 },
  { name: "journey", url: `/?${SP}&t=2026-10-14T01:30:00Z&journey=saturn&jrate=1.6`, wait: "#jSkip", click: "#jSkip", after: "#tab-learn .lc-title", settle: 5000 },
  { name: "lensing", url: `/?${SP}&t=2026-10-13T23:30:00Z&tab=analyses&az=225&alt=45&fov=60`, click: "#aLens", wait: "#aLensOut img", scroll: "#aLensOut", timeout: 600000 },
];
const LANGS = { en: "", pt: "-pt" };

mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
for (const s of SHOTS) {
  for (const [lang, suffix] of Object.entries(LANGS)) {
    const name = `14-observatory-${s.name}${suffix}`;
    if (ONLY.length && !ONLY.includes(s.name) && !ONLY.includes(name)) continue;
    const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, locale: lang === "pt" ? "pt-BR" : "en-GB", timezoneId: "America/Sao_Paulo" });
    page.on("pageerror", (e) => console.error(name, e.message));
    await page.goto(`${BASE}${s.url}&capture=1#lang=${lang}`);
    await page.waitForFunction(() => window.__ready === true, null, { timeout: 120000 });
    if (s.wait && s.after) await page.locator(s.wait).first().waitFor({ timeout: s.timeout || 60000 });
    if (s.click) await page.click(s.click);
    if (s.wait) await page.locator(s.after || s.wait).first().waitFor({ timeout: s.timeout || 60000 });
    if (s.scroll) await page.locator(s.scroll).scrollIntoViewIfNeeded();
    await page.waitForTimeout(s.settle || 2500);
    await page.screenshot({ path: `${OUT}${name}.jpg`, type: "jpeg", quality: 86 });
    console.log("wrote", name);
    await page.close();
  }
}
await browser.close();
