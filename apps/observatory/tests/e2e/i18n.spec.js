// Interface language (#721): switch to Portuguese, check that the main panels have no English left, switch back.
import { expect, test } from "@playwright/test";

const SP = "lat=-23.5505&lon=-46.6333&elev=760&tz=America/Sao_Paulo&t=2026-10-13T23:30:00Z&freeze=1";

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (e) => { throw e; });
});

// Words that occur only in the English catalogue: if one of them is on screen in Portuguese, it is untranslated.
async function englishOnlyWords(page) {
  return page.evaluate(async () => {
    const [en, pt] = await Promise.all(["en", "pt"].map((l) => fetch(`/static/i18n/${l}.json`).then((r) => r.json())));
    const words = (cat) => new Set(Object.values(cat).join(" ").replace(/<[^>]*>|\{\w+\}/g, " ").match(/[A-Za-zÀ-ÿ]{3,}/g));
    const p = words(pt), out = [];
    for (const w of words(en)) if (!p.has(w)) out.push(w);
    return out;
  });
}
async function visibleWords(page, selector) {
  const text = await page.locator(selector).innerText();
  return new Set(text.match(/[A-Za-zÀ-ÿ]{3,}/g) || []);
}
async function ready(page) {
  await page.waitForFunction(() => window.__ready === true);
  await expect(page.locator("#tab-tonight h3").first()).toBeAttached();
}

test("Portuguese: the main panels have no English left, and switching back restores English", async ({ page }) => {
  await page.goto(`/?${SP}&select=body:saturn#lang=en`);
  await ready(page);
  await expect(page.locator(".ptabs button[data-tab=tonight]")).toHaveText("Tonight");
  await expect(page.locator("#tab-object h3").first()).toContainText("Saturn");
  const enOnly = new Set(await englishOnlyWords(page));
  expect(enOnly.has("Tonight")).toBe(true);

  await page.click("#langSel button[data-lang=pt]");
  await expect(page.locator("html")).toHaveAttribute("lang", "pt-BR");
  expect(page.url()).toContain("lang=pt");
  expect(await page.evaluate(() => localStorage.getItem("obs.lang"))).toBe("pt");
  await expect(page.locator(".ptabs button[data-tab=tonight]")).toHaveText("Hoje à noite");
  await expect(page.locator("#clockSub")).toContainText("TSL");             // the clock is redrawn every frame
  await expect(page.locator("#tab-object h3").first()).toContainText("Saturno");

  const check = async (selector) => {
    const left = [...(await visibleWords(page, selector))].filter((w) => enOnly.has(w));
    expect(left, `English words left in ${selector}`).toEqual([]);
  };
  await check("header.obs-top");
  await check("#hud");
  await check(".ptabs");
  await check("#tab-object");                                   // Saturn's object card
  await page.click(".ptabs button[data-tab=tonight]");
  await check("#tab-tonight");
  await page.click(".ptabs button[data-tab=sats]");
  await expect(page.locator("#tab-sats h3").first()).toBeVisible();
  await expect(page.locator("#tab-sats #dRun")).toBeVisible();
  await check("#tab-sats");
  // a star card (Betelgeuse) and the date format of the clock
  await page.goto(`/?${SP}&select=star:2061#lang=pt`);
  await ready(page);
  await expect(page.locator("#tab-object h3").first()).toContainText("Betelgeuse");
  await expect(page.locator("#tab-object")).toContainText("Temperatura efetiva");
  await check("#tab-object");
  await expect(page.locator("#clock")).toContainText("2026");
  const lst = await page.locator("#tab-tonight .kv").innerText();
  expect(lst).toMatch(/\d,\d{3} h/);                           // decimal comma in Portuguese

  await page.click("#langSel button[data-lang=en]");
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await expect(page.locator(".ptabs button[data-tab=tonight]")).toHaveText("Tonight");
  await expect(page.locator("#tab-object")).toContainText("Effective temperature");
  expect(await page.evaluate(() => localStorage.getItem("obs.lang"))).toBe("en");
});

test("the language comes from the browser when nothing is chosen", async ({ browser }) => {
  const ctx = await browser.newContext({ locale: "pt-BR", viewport: { width: 1280, height: 800 } });
  const page = await ctx.newPage();
  await page.goto(`/?${SP}`);
  await ready(page);
  await expect(page.locator(".ptabs button[data-tab=analyses]")).toHaveText("Análises");
  await ctx.close();
});

test("constellations use their Portuguese names", async ({ page }) => {
  await page.goto(`/?${SP}&az=180&alt=40&fov=90#lang=pt`);
  await ready(page);
  await page.waitForTimeout(500);
  const names = await page.locator("#labels .lbl.con").allInnerTexts();
  expect(names.length).toBeGreaterThan(3);
  expect(names.join(" ")).not.toMatch(/\b(Scorpius|Sagittarius|Crux|Centaurus|Capricornus|Aquarius|Pisces)\b/i);
  const cat = await page.evaluate(() => window.__obs.sky.conLabels.map((c) => c.names.pt));
  expect(cat).toContain("Cruzeiro do Sul");
  expect(cat).toContain("Escorpião");
});
