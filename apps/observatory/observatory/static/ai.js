// PINNeAPPle models in the observatory: orbit decay PINN, small bodies and the learned propagator, space weather,
// observing conditions, stellar ML, the black-hole panel. Each block shows a trust card: verdict, checks against an
// independent reference, and "how this was computed, how much to trust it".
import { num, sig, t, tOr } from "./i18n.js";

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// a check from the library ({id, args, name, ...}) in the interface language; the English name if no message exists
export const checkName = (c) => (c.id ? tOr("check." + c.id, c.name, c.args) : c.name);
export function verdictBadge(verdict) {
  const v = verdict?.status || verdict || "unknown";
  return `<span class="badge ${esc(v)}">${esc(tOr("verdict." + v, String(v).replace(/_/g, " ")))}</span>`;
}
const unit = (u) => (u && u !== "-" ? " " + esc(tOr("unit." + u, u)) : "");

export function trustCard({ title, verdict, checks = [], how, trust, extra = "" }) {
  return `<div class="card trust"><div class="row"><b>${esc(title)}</b>${verdictBadge(verdict)}</div>
    ${extra}
    ${checks.length ? `<table><tr><th>${t("trust.check")}</th><th class="r">${t("trust.value")}</th><th class="r">${t("trust.limit")}</th><th></th></tr>${checks.map((c) =>
      `<tr><td>${esc(checkName(c))}</td><td class="r">${sig(c.value)}${unit(c.unit)}</td><td class="r">${c.higher_is_better ? "≥ " : "≤ "}${sig(c.threshold)}</td><td><span class="badge ${esc(c.status)}">${esc(tOr("status." + c.status, c.status))}</span></td></tr>`).join("")}</table>` : ""}
    <details><summary>${t("trust.how")}</summary><p class="small">${how}</p>${trust ? `<p class="small"><b>${t("trust.trust")}</b> ${trust}</p>` : ""}</details></div>`;
}

// ---------------------------------------------------------------------------------------- orbit decay
export async function renderDecay(el, ctx) {
  el.innerHTML = `<h3>${t("decay.title")}</h3>
    <div class="row small"><label>${t("decay.days")} <input id="dDays" type="number" value="60" min="5" max="365" style="width:64px"></label>
    <label>F10.7 <input id="dF" type="number" value="150" min="60" max="300" style="width:64px"></label>
    <label>Ap <input id="dAp" type="number" value="15" min="0" max="200" style="width:56px"></label>
    <label>B = C<sub>D</sub>A/m <input id="dB" type="number" value="0.005" step="0.001" style="width:72px"></label>
    <button class="primary" id="dRun">${t("decay.run")}</button></div><div id="dOut"></div>`;
  const run = async () => {
    const out = el.querySelector("#dOut");
    out.innerHTML = `<p class="muted">${t("decay.training")}</p>`;
    try {
      const q = `days=${el.querySelector("#dDays").value}&f107=${el.querySelector("#dF").value}&ap=${el.querySelector("#dAp").value}&B=${el.querySelector("#dB").value}&t=${ctx.iso()}`;
      const r = await ctx.api(`/api/sat/decay?norad=25544&${q}`);
      const x = r.t_days.map((d) => +d.toFixed(1));
      const drop = r.h_pinn_km[0] - r.h_pinn_km.at(-1);
      out.innerHTML = trustCard({ title: t("decay.cardTitle", { sat: r.satellite ? r.satellite.name : t("decay.issLike"), h0: num(r.h0_km, 1), h1: num(r.h_pinn_km.at(-1), 1), days: r.days }),
        verdict: r.verdict, checks: r.checks,
        extra: `<p class="small">${t("decay.line", { rate: num(r.decay_rate_m_per_day_t0, 0), drop: num(drop, 2), lo: num(r.h0_km - r.env_p95_km.at(-1), 2), hi: num(r.h0_km - r.env_p05_km.at(-1), 2) })} · ${t("decay.density")} NRLMSIS 2.0</p><div id="dChart"></div>
          ${r.sgp4 ? `<p class="small">${t("decay.sgp4", { a: num(r.sgp4.rate_sgp4_m_day, 1), b: num(r.sgp4.rate_model_m_day, 1), ratio: num(r.sgp4.ratio, 2) })}</p>` : `<p class="small muted">${t("decay.noSgp4")}</p>`}`,
        how: t("decay.how"), trust: t("decay.trust") });
      PChart.mount(out.querySelector("#dChart"), { x, height: 200, yLabel: "km", series: [
        { name: "PINN", y: r.h_pinn_km, color: "#7fb4ff", width: 2 },
        { name: t("decay.s.rk"), y: r.h_ref_km, color: "#ffcf6e", dash: "4 3" },
        { name: t("decay.s.envelope"), y: r.env_median_km, lo: r.env_p05_km, hi: r.env_p95_km, color: "#3ccf8e", dash: "2 2" }] });
    } catch (e) { out.textContent = e.message; }
  };
  el.querySelector("#dRun").onclick = run;
}

// ------------------------------------------------------------------------------------- space weather
export async function renderSpaceWeather(el, ctx) {
  el.innerHTML = `<p class="muted">${t("sw.loading")}</p>`;
  try {
    const r = await ctx.api(`/api/spaceweather?${ctx.qs()}`);
    const f = (h) => r.forecasts.find((x) => x.horizon_h === h);
    const row = (x) => `<tr><td>+${x.horizon_h} h</td><td class="r">${num(x.median, 1)}</td><td class="r">${num(x.lo, 1)}–${num(x.hi, 1)}</td><td class="r">${num(x.aurora.p_overhead * 100, 0)} %</td><td class="r">${num(x.aurora.p_horizon * 100, 0)} %</td><td>${verdictBadge(x.verdict)}</td></tr>`;
    el.innerHTML = `<div class="kv"><span>${t("sw.kpNow")}</span><span>${num(r.kp_now, 1)}</span><span>${t("sw.maglat")}</span><span>${num(r.maglat, 1)}°</span>
      <span>${t("sw.boundary")}</span><span>${num(r.boundary_now, 1)}° ${t("sw.geomag")}</span><span>${t("sw.kpNeeded")}</span><span>${r.forecasts[0] ? num(r.forecasts[0].aurora.kp_needed_overhead, 1) + " / " + num(r.forecasts[0].aurora.kp_needed_horizon, 1) : "—"}</span></div>
      <table><tr><th>${t("sw.lead")}</th><th class="r">Kp</th><th class="r">80 %</th><th class="r">${t("sw.overhead")}</th><th class="r">${t("sw.horizon")}</th><th></th></tr>${r.forecasts.map(row).join("")}</table>
      <p class="small muted">${t(r.live ? "sw.sourceLive" : "sw.sourceOffline")}</p>` +
      trustCard({ title: t("sw.cardTitle"), verdict: f(24)?.verdict, checks: [...(f(3)?.checks || []), ...(f(24)?.checks || [])],
        how: t("sw.how"), trust: t("sw.trust") });
  } catch (e) { el.textContent = e.message; }
}

// --------------------------------------------------------------------------------------- conditions
export async function renderConditions(el, ctx) {
  el.innerHTML = `<p class="muted">${t("wx.fetching")}</p>`;
  const r = await ctx.api(`/api/conditions?${ctx.qs()}`).catch((e) => ({ error: e.message }));
  if (r.error) {
    el.innerHTML = trustCard({ title: t("wx.cardTitleOM"), verdict: "not_available",
      extra: `<p class="small">${esc(r.error)}</p>`, how: t("wx.howLong"), trust: t("wx.trustLong") });
    return;
  }
  const x = r.time.map((s) => s.replace("T", " ").slice(5, 16));
  el.innerHTML = `<div id="cChart"></div><p class="small">${t("wx.members", { n: r.n_members })} · Open-Meteo (CC BY 4.0)</p>` +
    trustCard({ title: t("wx.cardTitle"), verdict: "use_with_caution", how: t("wx.how"), trust: t("wx.trust") });
  PChart.mount(el.querySelector("#cChart"), { x, height: 180, yLabel: "%", yMin: 0, yMax: 100, series: [
    { name: t("wx.s.pclear"), y: r.p_clear.map((p) => (p === null ? null : p * 100)), color: "#3ccf8e", width: 2 },
    { name: t("wx.s.cloud"), y: r.cloud_q50, lo: r.cloud_q10, hi: r.cloud_q90, color: "#8d9bb3" }] });
}

// ------------------------------------------------------------------------------------------ small bodies
export async function renderSmallBodies(el, ctx) {
  const list = await ctx.api("/api/smallbodies");
  el.innerHTML = `<div class="row"><select id="sbBody">${list.bodies.map((b) => `<option value="${b.key}">${esc(b.name)}</option>`).join("")}</select>
    <select id="sbYears"><option value="5">${t("sb.years", { n: 5 })}</option><option value="10" selected>${t("sb.years", { n: 10 })}</option><option value="23">${t("sb.to2049")}</option></select>
    <button class="primary" id="sbRun">${t("sb.approaches")}</button><button class="ghost" id="sbShow">${t("sb.show")}</button></div><div id="sbOut"></div><div id="sbVal"><p class="muted">${t("sb.loadingVal")}</p></div>`;
  const out = el.querySelector("#sbOut");
  el.querySelector("#sbRun").onclick = async () => {
    out.innerHTML = `<p class="muted">${t("sb.integrating")}</p>`;
    try {
      const b = el.querySelector("#sbBody").value;
      const r = await ctx.api(`/api/smallbodies/approaches?body=${b}&start=${ctx.iso()}&years=${el.querySelector("#sbYears").value}&max_au=0.2`);
      out.innerHTML = r.approaches.length ? `<table><tr><th>${t("sb.date")}</th><th class="r">${t("sb.distance")}</th><th class="r">v<sub>rel</sub></th><th class="r">JPL</th></tr>${r.approaches.map((c) =>
        `<tr${c.dist_km < 1e6 ? ' style="color:#ffcf6e"' : ""}><td>${c.utc.slice(0, 16).replace("T", " ")}</td><td class="r">${c.dist_km < 2e6 ? num(c.dist_km / 1000, 1) + " Mm (" + num(c.lunar_distances, 2) + " " + t("unit.ld") + ")" : num(c.dist_au, 4) + " " + t("unit.au")}</td><td class="r">${num(c.v_rel_km_s, 2)} km/s</td><td class="r">${c.jpl ? num(c.jpl.dist_au, 6) + " " + t("unit.au") : "—"}</td></tr>`).join("")}</table><p class="small muted">${t("method.smallbodies")}</p>`
        : `<p class="muted">${t("sb.none")}</p>`;
    } catch (e) { out.textContent = e.message; }
  };
  el.querySelector("#sbShow").onclick = () => ctx.showSmallBody(el.querySelector("#sbBody").value);
  const v = await ctx.api("/api/smallbodies/validation").catch((e) => ({ error: e.message }));
  if (v.error) { el.querySelector("#sbVal").textContent = v.error; return; }
  const a = v.apophis_2029;
  let html = trustCard({ title: t("sb.cardTitle"), verdict: v.verdict, checks: v.checks,
    extra: `<p class="small">${t("sb.apophis", { date: esc(a.jpl_cd), d: num(a.model_dist_km, 0), jd: num(a.jpl_dist_km, 0), s3: num(a.jpl_3sigma_km, 0), v: num(a.model_v_km_s, 2), jv: num(a.jpl_v_km_s, 2) })}</p>`,
    how: t("sb.how"), trust: t("sb.trust") });
  html += `<div id="sbPinn"><button class="ghost" id="sbPinnGo">${t("sb.trainPinn")}</button></div>`;
  const nr = v.neural;
  if (nr) html += trustCard({ title: t("sb.neuralTitle"), verdict: nr.verdict, checks: nr.checks,
    extra: `<p class="small">${t("sb.neuralLine", { ntr: nr.info.n_train, nte: nr.info.n_test })}</p>`,
    how: t("sb.neuralHow"), trust: t("sb.neuralTrust") });
  el.querySelector("#sbVal").innerHTML = html;
  el.querySelector("#sbPinnGo").onclick = async () => {
    const box = el.querySelector("#sbPinn"), b = el.querySelector("#sbBody").value;
    box.innerHTML = `<p class="muted">${t("sb.pinnTraining", { body: esc(b) })}</p>`;
    try {
      const r = await ctx.api(`/api/smallbodies/pinn?body=${b}&t=${ctx.iso()}&days=365`);
      box.innerHTML = trustCard({ title: t("sb.pinnTitle", { body: r.body }), verdict: r.verdict, checks: r.checks,
        extra: `<p class="small">${t("sb.pinnLine", { km: num(r.summary.max_error_km_test, 0), arcsec: num(r.summary.max_angle_arcsec_test, 3), kep: num(r.summary.kepler_max_error_km, 0), s: num(r.train_seconds, 0) })}</p><div id="sbPinnChart"></div>`,
        how: t("sb.pinnHow"), trust: t("sb.pinnTrust") });
      const x = r.t_days.map((d) => Math.round(d));
      PChart.mount(box.querySelector("#sbPinnChart"), { x, height: 160, yLog: true, yLabel: "km", series: [
        { name: t("eng.s.pinn"), y: r.error_km.map((e) => Math.max(e, 1)), color: "#7fb4ff", width: 2 },
        { name: t("eng.s.kepler"), y: r.kepler_error_km.map((e) => Math.max(e, 1)), color: "#ff6b6b", dash: "4 3" },
        { name: t("eng.s.band"), y: r.band_km.map((e) => Math.max(e, 1)), color: "#3ccf8e", dash: "2 2" }] });
    } catch (e) { box.textContent = e.message; }
  };
}

// ---------------------------------------------------------------------------------------------- stellar ML
export async function renderStarsML(el, ctx) {
  el.innerHTML = `<p class="muted">${t("ml.clustering")}</p>`;
  const r = await ctx.api("/api/stars/ml");
  const cols = ["#7fb4ff", "#ff9a6a", "#ffcf6e"];
  const W = el.clientWidth || 360, H = 260, L = 34, B = 24;
  const X = (bv) => L + (bv + 0.4) / 2.4 * (W - L - 6), Y = (m) => 6 + (m + 8) / 24 * (H - 6 - B);
  const up = ctx.aboveHorizon();
  let dots = "";
  for (let k = 0; k < r.hr.length; k++) {
    const isUp = up.has(r.hr[k]);
    dots += `<circle cx="${X(r.bv[k]).toFixed(1)}" cy="${Y(r.Mv[k]).toFixed(1)}" r="${isUp ? 2.2 : 1.2}" fill="${cols[r.cluster[k]]}" opacity="${isUp ? 1 : 0.45}"/>`;
  }
  const group = (n) => tOr("ml.group." + String(n).toLowerCase().replace(/[^a-z0-9]+/g, "_"), n);
  const legend = r.cluster_names.map((n, j) => `<span><i style="background:${cols[j]}"></i>${esc(group(n))}</span>`).join("");
  const lumRec = r.lum_cv.recall.map((x, i) => `${esc(group(r.lum_groups[i]))}: ${x === null ? "—" : Math.round(x * 100) + "%"}`).join(" · ");
  el.innerHTML = `<svg width="${W}" height="${H}" style="font:10px system-ui;fill:#8d9bb3">${dots}
    ${[-0.4, 0, 0.5, 1, 1.5, 2].map((v) => `<text x="${X(v)}" y="${H - 6}" text-anchor="middle">${num(v, v % 1 ? 1 : 0)}</text>`).join("")}${[-6, -3, 0, 3, 6, 9, 12, 15].map((m) => `<text x="${L - 4}" y="${Y(m) + 3}" text-anchor="end">${m}</text>`).join("")}
    <text x="${W / 2}" y="${H}" text-anchor="middle">B − V</text></svg><div class="pchart-legend">${legend}</div><p class="small muted">${t("ml.bright")}</p>` +
    trustCard({ title: t("ml.cardTitle"), verdict: r.verdict, checks: r.checks,
      extra: `<p class="small">${t("ml.line", { n: r.n_stars_hr, nl: r.n_labelled, acc: Math.round(r.lum_cv.accuracy * 100), base: Math.round(r.lum_cv.baseline * 100), rec: lumRec, sacc: Math.round(r.spec_cv.accuracy * 100), sbase: Math.round(r.spec_cv.baseline * 100) })}</p>`,
      how: t("ml.how"), trust: t("ml.trust") });
}

// ------------------------------------------------------------------------------------------- black hole
export async function renderBlackHole(el, ctx) {
  el.innerHTML = `<div class="row"><label class="small"><input type="checkbox" id="bhDop" checked> ${t("bh.doppler")}</label>
    <label class="small">${t("bh.incl")} <input id="bhInc" type="range" min="60" max="89" value="86"></label>
    <button class="primary" id="bhRun">${t("an.renderHere")}</button></div><div id="bhOut"></div><div id="bhZoo"></div>`;
  el.querySelector("#bhRun").onclick = () => {
    const [ra, dec] = ctx.viewRaDec();
    const out = el.querySelector("#bhOut");
    out.innerHTML = `<p class="muted">${t("bh.tracing")}</p>`;
    const img = new Image(); img.className = "lens";
    img.src = `/api/analysis/accretion.png?ra=${ra.toFixed(3)}&dec=${dec.toFixed(3)}&doppler=${el.querySelector("#bhDop").checked}&incl=${el.querySelector("#bhInc").value}`;
    img.onload = () => { out.innerHTML = ""; out.appendChild(img); out.insertAdjacentHTML("beforeend", `<p class="small muted">${t("bh.caption", { ra: num(ra, 1), dec: num(dec, 1) })}</p>`); };
    img.onerror = () => { out.textContent = t("err.render"); };
  };
  const z = await ctx.api("/api/blackhole/status").catch(() => ({ available: false, note: null }));
  el.querySelector("#bhZoo").innerHTML = z.available
    ? `<div class="card"><b>${t("bh.zooTitle")}</b> ${t("bh.zooFrom")} <code>${esc(z.model)}</code> ${t("bh.zooPassed")}</div>`
    : `<div class="card"><b>${t("bh.zooTitle")}</b> <span class="badge neutral">${t("bh.placeholder")}</span><p class="small">${t("bh.zooNote")}</p></div>`;
}
