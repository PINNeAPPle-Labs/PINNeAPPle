// Learn mode (#724): an info card for every object, from pinneapple_physics.astro.facts through /api/facts/*.
// Key numbers with everyday comparisons and a source each, curiosities with sources, history, missions, how to see
// it tonight from the observer's place, and a short quiz. Search in English or Portuguese, a "did you know?" tip.
import { fmtDate, lang, num, t, tOr } from "./i18n.js";

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const GROUPS = ["planet", "dwarf_planet", "moon", "star", "constellation", "messier", "satellite"];

// a number of the dataset in the interface language, with its unit
export function fmtValue(row, ctx) {
  const v = row.value, u = row.unit;
  if (v === null || v === undefined) return "—";
  if (u === "date") return fmtDate(new Date(v + "T12:00:00Z"), { day: "numeric", month: "long", year: "numeric" });
  if (u === "month") return fmtDate(new Date(Date.UTC(2001, v - 1, 15)), { month: "long" });
  if (row.key === "constellation") return esc(ctx.conName(v));
  if (u === "arcmin") return String(v).split("x").map((x) => num(+x, +x < 10 ? 1 : 0) + "′").join(" × ");
  if (typeof v !== "number") return esc(v);
  const a = Math.abs(v);
  let s;
  if (u === "kg" || u === "W") s = sciFmt(v);
  else if (u === "km" && a >= 1e7) return num(v / 1e6, a >= 1e9 ? 0 : 1) + " " + t("unit.millionKm");
  else s = num(v, a >= 100 || Number.isInteger(v) ? 0 : a >= 10 ? 1 : 2);
  if (!u || u === "count" || u === "mag") return s;
  return `${s} ${tOr("unit.f." + u, u)}`;
}
function sciFmt(v) {
  const e = Math.floor(Math.log10(Math.abs(v))), m = v / 10 ** e;
  const sup = String(e).split("").map((c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+c] ?? (c === "-" ? "⁻" : c)).join("");
  return `${num(m, 2)} × 10${sup}`;
}

function tonightHtml(c, ctx) {
  const tn = c.tonight;
  if (c.passes) {
    if (!c.passes.length) return `<p class="muted">${t("learn.noPasses")}</p>`;
    return `<p>${t("learn.passesLine")}</p><ul class="small">${c.passes.map((p) => `<li>${ctx.dfmt(p.rise_utc)}: ${t("learn.maxAlt", { alt: num(p.max_alt, 0) })}${p.visible ? " · " + t("learn.visible") : ""}</li>`).join("")}</ul>`;
  }
  if (!tn) return "";
  if (tn.never_rises) return `<p class="warn-text">${t("learn.neverRises", { place: esc(ctx.place()) })}</p>`;
  let h = `<p>${tn.above_now ? t("learn.upNow", { alt: num(tn.alt_now, 0) }) : t("learn.downNow")}`;
  if (tn.never_sets) h += " " + t("learn.neverSets");
  h += "</p><div class=\"kv\">";
  if (tn.rise) h += `<span>${t("learn.rises")}</span><span>${ctx.dfmt(tn.rise)}</span>`;
  h += `<span>${t("learn.highest")}</span><span>${ctx.dfmt(tn.culmination)} · ${num(tn.max_alt, 0)}°</span>`;
  if (tn.set) h += `<span>${t("learn.sets")}</span><span>${ctx.dfmt(tn.set)}</span>`;
  h += "</div>";
  h += tn.dark_best ? `<p class="small">${t("learn.darkBest", { when: ctx.dfmt(tn.dark_best.time), alt: num(tn.dark_best.alt, 0) })}</p>`
    : `<p class="small muted">${t("learn.notDark")}</p>`;
  return h;
}

export function cardHtml(c, ctx) {
  const srcIds = Object.keys(c.sources);
  const ref = (s) => `<sup class="src" title="${esc(c.sources[s]?.title)}">[${srcIds.indexOf(s) + 1}]</sup>`;
  let h = `<div class="learn-card" data-card="${esc(c.id)}"><h3 class="lc-title">${esc(c.name)} <span class="badge neutral">${esc(c.kind_label)}</span></h3>
    <p class="lc-summary">${esc(c.summary)}</p>`;
  if (c.tonight !== undefined || c.passes) h += `<h4>${t("learn.tonightTitle", { place: esc(ctx.place()) })}</h4>${tonightHtml(c, ctx)}`;
  const canObserve = c.observe || c.ra_deg !== undefined || c.hr;
  h += `<div class="row">${ctx.canLook(c) ? `<button class="primary" data-act="look">${t("object.look")}</button>` : ""}${canObserve && ctx.observe ? `<button class="ghost" data-act="observe">${t("learn.observeIt")}</button>` : ""}${((c.kind === "planet" && c.id !== "earth") || c.hr || c.id === "sun" || c.id === "moon") && ctx.journey ? `<button class="ghost" data-act="journey">${t("learn.flyThere")}</button>` : ""}</div>`;
  h += `<h4>${t("learn.numbers")}</h4><table class="lc-numbers">${c.numbers.map((r) => `<tr data-key="${r.key}"><td>${esc(r.label)}</td><td class="r"><b data-raw="${esc(r.value)}">${fmtValue(r, ctx)}</b>${ref(r.source)}${r.comparison ? `<br><span class="small muted">${esc(r.comparison)}</span>` : ""}</td></tr>`).join("")}</table>`;
  h += `<h4>${t("learn.curiosities")}</h4><ul class="lc-curios">${c.curiosities.map((x) => `<li>${esc(x.text)}${ref(x.source)}</li>`).join("")}</ul>`;
  h += `<h4>${t("learn.history")}</h4><p>${esc(c.history)}</p>`;
  if (c.missions && c.missions.length) h += `<p class="small"><b>${t("learn.missions")}</b> ${c.missions.map(esc).join(" · ")}</p>`;
  if (c.quiz && c.quiz.length) {
    h += `<h4>${t("learn.quiz")}</h4><div class="lc-quiz">${c.quiz.map((q, i) => `<div class="q" data-q="${i}" data-answer="${q.answer}"><p>${esc(q.question)}</p><div class="row">${q.options.map((o, j) => `<button class="ghost" data-opt="${j}">${esc(o)}</button>`).join("")}</div><p class="small qres"></p></div>`).join("")}</div>`;
  }
  h += `<details class="lc-sources"><summary>${t("learn.sources")}</summary><ol class="small">${srcIds.map((s) => `<li><a href="${esc(c.sources[s].url)}" target="_blank" rel="noopener">${esc(c.sources[s].title)}</a> · ${esc(c.sources[s].publisher)}</li>`).join("")}</ol></details></div>`;
  return h;
}

function wireCard(el, c, ctx) {
  const look = el.querySelector("[data-act=look]");
  if (look) look.onclick = () => ctx.look(c);
  const obs = el.querySelector("[data-act=observe]");
  if (obs) obs.onclick = () => ctx.observe(c);
  const fly = el.querySelector("[data-act=journey]");
  if (fly) fly.onclick = () => ctx.journey(c);
  for (const q of el.querySelectorAll(".lc-quiz .q")) {
    for (const b of q.querySelectorAll("button[data-opt]")) b.onclick = () => {
      const ok = +b.dataset.opt === +q.dataset.answer;
      for (const x of q.querySelectorAll("button[data-opt]")) { x.disabled = true; if (+x.dataset.opt === +q.dataset.answer) x.classList.add("right"); }
      if (!ok) b.classList.add("wrong");
      q.querySelector(".qres").textContent = ok ? t("learn.right") : t("learn.wrong");
    };
  }
}

export async function showCard(el, id, ctx) {
  el.innerHTML = `<p class="muted">${t("common.loading")}</p>`;
  try {
    const c = await ctx.api(`/api/facts/card?id=${encodeURIComponent(id)}&lang=${lang}&t=${ctx.iso()}&${ctx.qs()}`);
    el.innerHTML = cardHtml(c, ctx);
    wireCard(el, c, ctx);
    return c;
  } catch (e) { el.innerHTML = `<p>${esc(e.message)}</p>`; return null; }
}

// the Learn tab: search, a tip, browse by kind, the selected card
export async function renderLearn(el, ctx, id) {
  el.innerHTML = `<div class="row"><input id="lSearch" type="text" placeholder="${esc(t("learn.searchPh"))}" autocomplete="off" style="flex:1"></div>
    <div id="lResults" class="lresults"></div>
    <div class="row lgroups">${GROUPS.map((g) => `<button class="ghost" data-group="${g}">${t("learn.group." + g)}</button>`).join("")}</div>
    <div id="lList" class="lresults"></div>
    <div id="lTip"></div><div id="lCard"></div>`;
  const results = el.querySelector("#lResults"), list = el.querySelector("#lList"), cardEl = el.querySelector("#lCard");
  const open = (cid) => { results.innerHTML = ""; list.innerHTML = ""; el.querySelector("#lTip").innerHTML = ""; showCard(cardEl, cid, ctx); };
  const chips = (rows) => rows.map((r) => `<button class="chip" data-id="${esc(r.id)}">${esc(r.name)}</button>`).join("");
  const wire = (box) => { for (const b of box.querySelectorAll("[data-id]")) b.onclick = () => open(b.dataset.id); };
  let timer = null;
  el.querySelector("#lSearch").oninput = (e) => {
    clearTimeout(timer);
    const q = e.target.value.trim();
    if (!q) { results.innerHTML = ""; return; }
    timer = setTimeout(async () => {
      const r = await ctx.api(`/api/facts/search?q=${encodeURIComponent(q)}&lang=${lang}`).catch(() => ({ results: [] }));
      results.innerHTML = r.results.length ? chips(r.results) : `<p class="small muted">${t("learn.noResults")}</p>`;
      wire(results);
    }, 180);
  };
  for (const b of el.querySelectorAll("[data-group]")) b.onclick = async () => {
    const r = await ctx.api(`/api/facts/list?kind=${b.dataset.group}&lang=${lang}`);
    list.innerHTML = chips(r.cards); wire(list);
  };
  if (id) open(id);
  else {
    const tip = await ctx.api(`/api/facts/tip?lang=${lang}`).catch(() => null);
    if (tip) {
      el.querySelector("#lTip").innerHTML = `<div class="card tip"><b>${t("learn.didYouKnow")}</b> ${esc(tip.text)} <span class="small muted">(${esc(tip.source_title)})</span><div class="row"><button class="ghost" data-id="${esc(tip.id)}">${t("learn.more", { name: esc(tip.name) })}</button></div></div>`;
      wire(el.querySelector("#lTip"));
    }
  }
}

// the start-up tip (a small card over the sky)
export async function startupTip(box, ctx) {
  const tip = await ctx.api(`/api/facts/tip?lang=${lang}`).catch(() => null);
  if (!tip) return;
  box.innerHTML = `<button class="x" aria-label="${esc(t("learn.close"))}">×</button><b>${t("learn.didYouKnow")}</b> ${esc(tip.text)}<div class="row"><button class="ghost" data-id="${esc(tip.id)}">${t("learn.more", { name: esc(tip.name) })}</button></div>`;
  box.hidden = false;
  box.querySelector(".x").onclick = () => { box.hidden = true; };
  box.querySelector("[data-id]").onclick = () => { box.hidden = true; ctx.openLearn(tip.id); };
  setTimeout(() => { box.hidden = true; }, 20000);
}
