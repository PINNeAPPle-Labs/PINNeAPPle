// Observing session (#722): plan, telescope, slew, a 2-hour tracking session at 600x with the target centred, and the
// "real sky" weather with a forced overcast forecast hiding the view and going into the session log.
import { expect, test } from "@playwright/test";

const SP = "lat=-23.5505&lon=-46.6333&elev=760&tz=America/Sao_Paulo";

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (e) => { throw e; });
});

async function ready(page) {
  await page.waitForFunction(() => window.__ready === true);
  await expect(page.locator("#obChart svg")).toBeVisible({ timeout: 60_000 });
}

test("a 2-hour session at 600x keeps the target centred while tracking", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto(`/?${SP}&t=2026-10-13T23:00:00Z&observe=saturn#lang=en`);
  await ready(page);
  await expect(page.locator("#obOptics")).toContainText("48×");                     // 200 mm f/6 + 25 mm eyepiece
  await expect(page.locator("#obPlanOut")).toContainText("Best window");
  await page.click("#obSlew");
  await expect(page.locator("#obSlewOut")).toContainText("On target", { timeout: 30_000 });
  await page.click("#obTrack");
  await expect(page.locator("#epCanvas")).toBeVisible();
  const jd0 = await page.evaluate(() => window.__obs.S.simJD);
  await page.click("#eyepiece [data-speed='600']");
  await page.waitForFunction((j) => window.__obs.S.simJD - j > 2 / 24, jd0, { timeout: 120_000, polling: 250 });
  const st = await page.evaluate(() => { const o = window.__obs.observe; return { ...o.state, w: document.querySelector("#epCanvas").clientWidth }; });
  expect(st.frames).toBeGreaterThan(20);
  expect(st.max_offset_px).toBeLessThan(0.02 * st.w);                               // within 2 % of the field
  expect(Math.abs(st.rotation_deg)).toBeGreaterThan(5);                             // alt-az: the field turned
  await expect(page.locator("#epHud")).toContainText("field rotation");
  await expect(page.locator("#obLog")).toContainText("Tracking Saturn");
});

test("an equatorial mount has no field rotation; Portuguese text", async ({ page }) => {
  await page.goto(`/?${SP}&t=2026-10-14T01:00:00Z&observe=M42#lang=pt`);
  await ready(page);
  await page.selectOption("#obMount", "equatorial");
  await page.selectOption("#obEye", "plossl10");
  await expect(page.locator("#obOptics")).toContainText("120×");
  await page.click("#obTrack");
  await expect(page.locator("#epHud")).toContainText("montagem equatorial: sem rotação de campo");
  await expect(page.locator("#epCredit")).toContainText("Catálogo de Estrelas Brilhantes");    // offline: catalogue field
});

test("real sky with an overcast forecast hides the view and logs it; perfect sky shows it", async ({ page }) => {
  await page.goto(`/?${SP}&t=2026-10-14T01:00:00Z&observe=M42&wx=cloudy#lang=en`);
  await ready(page);
  await page.check("input[name=obWx][value=real]");
  await expect(page.locator("#obWx")).toContainText("100% cloud cover");
  await page.click("#obTrack");
  await page.waitForFunction(() => window.__obs.observe.state.hidden === true);
  await expect(page.locator("#obLog")).toContainText("Clouds hide the view");
  await page.check("input[name=obWx][value=perfect]");
  await page.waitForFunction(() => window.__obs.observe.state.hidden === false);
  await expect(page.locator("#obLog")).toContainText("The view is clear");
});

test("a target that never rises is flagged; a learn card links to the plan", async ({ page }) => {
  await page.goto(`/?${SP}&t=2026-10-14T01:00:00Z&learn=star:424#lang=en`);              // Polaris from São Paulo
  await page.waitForFunction(() => window.__ready === true);
  await page.click("#tab-learn [data-act=observe]");
  await expect(page.locator("#obPlanOut")).toContainText("never rises", { timeout: 60_000 });
});
