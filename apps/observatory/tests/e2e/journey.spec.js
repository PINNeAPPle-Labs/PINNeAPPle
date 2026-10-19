// Space journey (#723): fly to Mars and to Sirius, check the counters and the arrival card; the Sun as a star.
import { expect, test } from "@playwright/test";

const SP = "lat=-23.5505&lon=-46.6333&elev=760&tz=America/Sao_Paulo&t=2026-10-14T01:30:00Z";

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (e) => { throw e; });
});

async function fly(page, target, lang) {
  await page.goto(`/?${SP}&journey=${target}&jrate=1.6#lang=${lang}`);
  await page.waitForFunction(() => window.__ready === true);
  await page.waitForFunction(() => window.__obs.journey.plan && window.__obs.journey.state === "flying");
  return page.evaluate(() => window.__obs.journey.plan);
}

test("fly to Mars: distance from the ephemeris, counters, arrival card", async ({ page, request }) => {
  const plan = await fly(page, "mars", "en");
  const ref = await (await request.get(`/api/journey/plan?target=mars&${SP}`)).json();
  expect(Math.abs(plan.distance_km - ref.distance_km) / ref.distance_km).toBeLessThan(1e-6);   // same trip, seconds apart
  await expect(page.locator("#jHud")).toContainText("The whole trip");
  await expect(page.locator("#jHud")).toContainText("Voyager 1");
  await page.waitForFunction(() => window.__obs.journey.state === "arrived", null, { timeout: 90_000 });
  const c = await page.evaluate(() => window.__obs.journey.counters());
  const total = await page.evaluate(() => window.__obs.journey.total);                  // up 2000 km, then straight on
  expect(c.travelled).toBe(total);
  expect(Math.abs(c.left - 4 * plan.target.radius_km)).toBeLessThan(1);
  await expect(page.locator("#jHud")).toContainText("arrived");
  await expect(page.locator("#tab-learn .lc-title")).toContainText("Mars");             // the info card (#724)
});

test("fly to Sirius in Portuguese: the Sun behind you as a star", async ({ page }) => {
  const plan = await fly(page, "star:2491", "pt");
  expect(plan.distance_ly).toBeGreaterThan(8.5);
  expect(plan.distance_ly).toBeLessThan(8.7);
  await page.click("#jSkip");
  await page.waitForFunction(() => window.__obs.journey.state === "arrived");
  await expect(page.locator("#jHud")).toContainText("chegamos");
  await expect(page.locator("#jHud")).toContainText("o Sol é uma estrela de magnitude");
  const m = plan.sun_seen_from_target_mag;
  expect(m).toBeGreaterThan(1.85);
  expect(m).toBeLessThan(1.95);                                                         // 4.81 + 5 log10(2.637 pc / 10 pc)
  await expect(page.locator("#tab-learn .lc-title")).toContainText("Sirius");
  await page.click("#jBack");                                                          // turn around: the Sun behind you
  await expect(page.locator("#labels .lbl", { hasText: "Sol · mag 1,9" })).toBeVisible();
});
