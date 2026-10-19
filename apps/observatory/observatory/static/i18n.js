// Interface language (English / Brazilian Portuguese). Every string the user sees comes from the message catalogues
// in static/i18n/<lang>.json through t(); numbers and dates are formatted with Intl for the active locale.
// The language comes from #lang=pt|en in the URL, then local storage, then the browser language.
export const LANGS = ["en", "pt"];
const LOCALES = { en: "en-GB", pt: "pt-BR" };
const cats = {};

function hashLang() {
  const m = /(?:^|[#&])lang=([a-z]{2})/.exec(location.hash);
  return m && LANGS.includes(m[1]) ? m[1] : null;
}
function storedLang() {
  try { const v = localStorage.getItem("obs.lang"); return LANGS.includes(v) ? v : null; } catch (e) { return null; }
}
function browserLang() {
  const l = (navigator.languages && navigator.languages[0]) || navigator.language || "en";
  return l.toLowerCase().startsWith("pt") ? "pt" : "en";
}

export let lang = hashLang() || storedLang() || browserLang();

async function load(l) {
  if (!cats[l]) cats[l] = await (await fetch(`/static/i18n/${l}.json`)).json();
  return cats[l];
}
await load("en");
if (lang !== "en") await load(lang);

export const locale = () => LOCALES[lang];

// t(key, {name: value}): the message in the active language, {name} placeholders filled in. A missing key falls
// back to English, then to the key itself (the catalogue test keeps both catalogues complete).
export function t(key, vars) {
  let s = cats[lang]?.[key] ?? cats.en[key] ?? key;
  if (vars) s = s.replace(/\{(\w+)\}/g, (m, k) => (vars[k] === undefined ? m : vars[k]));
  return s;
}
export const has = (key) => key in cats.en;
// t() with a fallback text for keys built from server values (check names, sky states, ...)
export const tOr = (key, fallback, vars) => (has(key) ? t(key, vars) : fallback);

const nf = new Map();
export function num(v, d = 2) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const k = lang + d;
  if (!nf.has(k)) nf.set(k, new Intl.NumberFormat(locale(), { minimumFractionDigits: d, maximumFractionDigits: d }));
  return nf.get(k).format(+v);
}
// a number with n significant digits, exponential outside [1e-3, 1e5)
export function sig(v, n = 3) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)) {
    const [m, e] = (+v).toExponential(n - 1).split("e");
    return `${num(+m, n - 1)}×10${superscript(+e)}`;
  }
  return new Intl.NumberFormat(locale(), { maximumSignificantDigits: n }).format(+v);
}
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
function superscript(e) { return String(e).split("").map((c) => SUP[c] ?? c).join(""); }

export function fmtDate(d, opts) {
  try { return new Intl.DateTimeFormat(locale(), opts).format(d); } catch (e) { return new Intl.DateTimeFormat(locale(), { ...opts, timeZone: undefined }).format(d); }
}

// fill the static page: data-i18n (text), data-i18n-title, data-i18n-placeholder, data-i18n-aria
export function applyStatic(root = document) {
  document.documentElement.lang = lang === "pt" ? "pt-BR" : "en";
  for (const el of root.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of root.querySelectorAll("[data-i18n-title]")) el.title = t(el.dataset.i18nTitle);
  for (const el of root.querySelectorAll("[data-i18n-placeholder]")) el.placeholder = t(el.dataset.i18nPlaceholder);
  for (const el of root.querySelectorAll("[data-i18n-aria]")) el.setAttribute("aria-label", t(el.dataset.i18nAria));
  const d = document.querySelector('meta[name="description"]');
  if (d) d.content = t("meta.description");
  document.title = t("meta.title");
  document.dispatchEvent(new Event("app:lang"));            // shared widgets (/shared/author.js) follow
}

const listeners = [];
export const onLangChange = (f) => listeners.push(f);
export async function setLang(l) {
  if (!LANGS.includes(l)) return;
  await load(l);
  lang = l;
  try { localStorage.setItem("obs.lang", l); } catch (e) { /* storage blocked */ }
  const rest = location.hash.replace(/^#/, "").split("&").filter((kv) => kv && !kv.startsWith("lang="));
  history.replaceState(null, "", location.pathname + location.search + "#" + ["lang=" + l, ...rest].join("&"));
  applyStatic();
  for (const f of listeners) f(l);
}
