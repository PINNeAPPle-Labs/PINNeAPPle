// Learn mode (#724): the cards of Saturn, Betelgeuse, M31 and the ISS in English and Portuguese show the dataset's
// numbers with a source each, curiosities with sources, history and a quiz; search works in both languages.
import { expect, test } from "@playwright/test";

const SP = "lat=-23.5505&lon=-46.6333&elev=760&tz=America/Sao_Paulo&t=2026-10-13T23:30:00Z&freeze=1";
const CARDS = [["saturn", "Saturn", "Saturno"], ["star:2061", "Betelgeuse", "Betelgeuse"], ["M31", "Andromeda", "Andrômeda"],
  ["sat:25544", "International Space Station", "Estação Espacial Internacional"]];

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (e) => { throw e; });
});

for (const lang of ["en", "pt"]) {
  for (const [id, en, pt] of CARDS) {
    test(`${id} card in ${lang}`, async ({ page, request }) => {
      await page.goto(`/?${SP}&learn=${encodeURIComponent(id)}#lang=${lang}`);
      await page.waitForFunction(() => window.__ready === true);
      const card = page.locator("#tab-learn .learn-card");
      await expect(card).toBeVisible();
      await expect(card.locator(".lc-title")).toContainText(lang === "en" ? en : pt);
      // the numbers on screen are the dataset's, each with a source mark
      const api = await (await request.get(`/api/facts/card?id=${encodeURIComponent(id)}&lang=${lang}`)).json();
      const rows = card.locator(".lc-numbers tr");
      await expect(rows).toHaveCount(api.numbers.length);
      for (let i = 0; i < api.numbers.length; i++) {
        const r = api.numbers[i];
        await expect(rows.nth(i)).toHaveAttribute("data-key", r.key);
        expect(await rows.nth(i).locator("b").getAttribute("data-raw")).toBe(String(r.value));
        await expect(rows.nth(i).locator("sup.src")).toHaveCount(1);
      }
      const curios = card.locator(".lc-curios li");
      expect(await curios.count()).toBeGreaterThanOrEqual(3);
      for (let i = 0; i < await curios.count(); i++) await expect(curios.nth(i).locator("sup.src")).toHaveCount(1);
      await expect(card.locator(".lc-sources li")).toHaveCount(Object.keys(api.sources).length);
      await expect(card.locator("h4").first()).toHaveText(lang === "en" ? /Tonight from/ : /Hoje à noite, de/);
      // quiz: a click marks the right answer
      const q = card.locator(".lc-quiz .q").first();
      await q.locator("button[data-opt]").first().click();
      await expect(q.locator("button.right")).toHaveCount(1);
      await expect(q.locator(".qres")).not.toBeEmpty();
    });
  }
}

test("search in both languages, browse by kind, the tip", async ({ page }) => {
  await page.goto(`/?${SP}&tab=learn#lang=pt`);
  await page.waitForFunction(() => window.__ready === true);
  await expect(page.locator("#tab-learn .card.tip")).toBeVisible();
  await page.fill("#lSearch", "cruzeiro");
  await page.locator("#lResults [data-id='con-Cru']").click();
  await expect(page.locator("#tab-learn .lc-title")).toContainText("Cruzeiro do Sul");
  await page.fill("#lSearch", "andromeda galaxy");
  await expect(page.locator("#lResults [data-id='M31']")).toBeVisible();
  await page.locator("[data-group=messier]").click();
  await expect(page.locator("#lList [data-id]")).toHaveCount(110);
  await page.locator("#langSel button[data-lang=en]").click();
  await expect(page.locator(".ptabs button[data-tab=learn]")).toHaveText("Learn");
});

test("clicking a planet shows the start of its card in the object tab", async ({ page }) => {
  await page.goto(`/?${SP}&select=body:saturn#lang=pt`);
  await page.waitForFunction(() => window.__ready === true);
  await expect(page.locator("#oLearn .learn-teaser")).toContainText("Saturno");
  await page.locator("#oLearnGo").dispatchEvent("click");          // the object tab is redrawn as the sky data refreshes
  await expect(page.locator("#tab-learn .lc-title")).toContainText("Saturno");
});
