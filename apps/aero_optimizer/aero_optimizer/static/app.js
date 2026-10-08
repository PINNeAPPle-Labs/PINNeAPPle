"use strict";
// Aircraft Design Optimizer UI: search, the cloud of designs, a design's numbers / polar / flow field, the engine.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const f = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "—" : (+v).toFixed(d));
const KT = 0.514444;
let META = null, RES = null, SEL = null, DETAIL = null, FLOW = { view: "p", case: 0 };

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) { $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n)); $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`)); }

const PRESETS = [
  { name: "Cessna 172 class (default)", note: "1100 kg, 120 kW, 107 kt cruise, CS-23 stall limit 61 kt", ac: {}, req: {} },
  { name: "Short-field bush plane", note: "1100 kg, 135 kW, stall ≤ 48 kt for short strips",
    ac: { power_kw: 135, cruise_speed: 50 }, req: { max_stall_speed_kt: 48 } },
  { name: "Efficient tourer", note: "1100 kg, 120 kW, 120 kt cruise at 3000 m, span ≤ 12 m",
    ac: { cruise_speed: 62, cruise_altitude: 3000 }, req: { max_span: 12.0 } },
];
const AC_KEYS = ["mass", "power_kw", "cruise_altitude", "wing_area", "aspect_ratio", "cd0_rest", "oswald", "prop_efficiency", "bsfc"];
function fillForm(ac, req) {
  for (const k of AC_KEYS) $("#" + k).value = ac[k];
  $("#cruise_kt").value = (ac.cruise_speed / KT).toFixed(0);
  $("#max_stall_speed_kt").value = req.max_stall_speed_kt;
  $("#min_thickness_pct").value = (100 * req.min_thickness).toFixed(1);
  $("#sm_min").value = (100 * req.min_static_margin).toFixed(0);
  $("#sm_max").value = (100 * req.max_static_margin).toFixed(0);
  $("#stall_station").value = (100 * req.max_stall_station).toFixed(0);
  $("#max_span").value = req.max_span;
  $("#max_tail_incidence").value = req.max_tail_incidence;
  $("#max_cruise_cl_ratio").value = req.max_cruise_cl_ratio;
}
function readForm() {
  const ac = {};
  for (const k of AC_KEYS) ac[k] = +$("#" + k).value;
  ac.cruise_speed = +$("#cruise_kt").value * KT;
  return {
    aircraft: ac,
    requirements: { max_stall_speed_kt: +$("#max_stall_speed_kt").value, min_thickness: +$("#min_thickness_pct").value / 100,
      min_static_margin: +$("#sm_min").value / 100, max_static_margin: +$("#sm_max").value / 100,
      max_stall_station: +$("#stall_station").value / 100, max_span: +$("#max_span").value,
      max_tail_incidence: +$("#max_tail_incidence").value, max_cruise_cl_ratio: +$("#max_cruise_cl_ratio").value },
    population: +$("#population").value, generations: +$("#generations").value, seed: +$("#seed").value,
  };
}
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
  return j;
}
$("#run").addEventListener("click", async (e) => {
  const btn = e.currentTarget, old = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Searching…`; $("#status").textContent = "";
  try {
    const form = readForm();
    let r = await post("/api/optimize3d", form);
    const jobId = r.job;
    if (r.status !== "done") {
      $("#status").innerHTML = `<div class="progress"><div></div></div><div class="meta" id="prog-t">flying the first generation…</div>`;
      while (true) {
        await new Promise((ok) => setTimeout(ok, 1000));
        r = await (await fetch(`/api/job/${jobId}`)).json();
        if (r.status === "error") throw new Error(r.error || "search failed");
        if (r.status === "done") break;
        const pc = r.total ? (100 * r.done) / r.total : 0;
        $("#status .progress div").style.width = pc + "%";
        $("#prog-t").textContent = `generation ${r.done} of ${r.total} · ${(r.done * form.population).toLocaleString("en-US")} aircraft flown`;
      }
    }
    RES = r.result; RES.form = form; SEL = RES.picks.balanced ?? null;
    renderResult(); showTab("result"); window.scrollTo(0, 0); $("#status").textContent = "";
    if (SEL !== null) selectDesign(SEL);
  } catch (err) { $("#status").innerHTML = `<span class="err">${esc(err.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; }
});

// ------------------------------------------------------------------ result
const delta = (v, b, d = 1, unit = "", better = "up") => {
  if (!isFinite(v) || !isFinite(b)) return "";
  const x = v - b, good = better === "up" ? x > 0 : x < 0;
  return `<span class="${Math.abs(x) < 1e-9 ? "" : good ? "up" : "down"}">${x > 0 ? "+" : ""}${x.toFixed(d)}${unit}</span>`;
};
const pct = (v, b) => (isFinite(v) && isFinite(b) ? (100 * (v - b)) / b : NaN);

function renderResult() {
  const R = RES, B = R.baseline, D = R.details;
  const pk = R.picks, names = { fastest: "Fastest", greenest: "Lowest CO₂", safest: "Lowest stall speed", balanced: "Balanced" };
  const card = (k) => {
    if (pk[k] === undefined) return "";
    const d = D[pk[k]];
    return `<button class="pick" data-i="${pk[k]}"><div class="k">${names[k]}</div>
      <div class="big">${f(d.vmax_kt)} kt · ${f(d.co2_100km, 2)} kg CO₂/100 km</div>
      <div class="d">${delta(d.vmax_kt, B.vmax_kt, 1, " kt")} top speed · ${delta(pct(d.co2_100km, B.co2_100km), 0, 1, " %", "down")} CO₂ · stall ${f(d.v_stall_kt)} kt (${delta(d.v_stall_kt, B.v_stall_kt, 1, " kt", "down")}) vs NACA 2412<br>
      wing ${f(d.plan.wing_area, 1)} m² · AR ${f(d.plan.aspect_ratio, 1)} · span ${f(d.span, 1)} m · <span class="trust ${d.trust}">${d.trust} trust</span></div></button>`;
  };
  const total = R.designs.length, nf = R.feasible;
  const reasons = Object.entries(R.reasons).sort((a, b) => b[1] - a[1]);
  const maxr = Math.max(1, ...reasons.map((r) => r[1]));
  $("#result").innerHTML = `
    <div class="panel">
      <div class="toolbar"><h2 style="margin:0">${total.toLocaleString("en-US")} aircraft flown in ${f(R.seconds, 0)} s</h2>
        <div class="meta">${nf.toLocaleString("en-US")} flyable · ${R.pareto.length} on the front · baseline (Cessna 172 class, NACA 2412): ${f(B.vmax_kt)} kt, ${f(B.co2_100km, 2)} kg CO₂/100 km, stall ${f(B.v_stall_kt)} kt</div></div>
      ${R.pareto.length ? `<div class="picks">${card("balanced")}${card("fastest")}${card("greenest")}${card("safest")}</div>` : `<div class="banner bad">No design meets every requirement. Relax a requirement (stall speed, thickness) or widen the wing area range.</div>`}
      <div class="grid2" style="grid-template-columns: 2fr 1fr">
        <div><div class="scatter" id="scatter"></div>
          <div class="legend" style="margin-top:6px"><span>flyable, colour = CO₂/100 km <i style="width:90px;background:linear-gradient(90deg,#16a34a,#facc15,#dc2626)"></i> <span id="co2r"></span></span><span><i style="background:#fff;border:1.5px solid #f59e0b"></i>surrogate unsure</span><span><i style="background:#cbd5e1"></i>misses a requirement</span><span><i style="background:#fff;border:2px solid #111"></i>Pareto-optimal</span><span>★ NACA 2412</span></div></div>
        <div><h3 style="margin-top:0">Why designs were rejected</h3>
          <div class="reasons">${reasons.map(([k, v]) => `<span>${esc(k)}</span><span class="meta" style="text-align:right">${v.toLocaleString("en-US")}</span><div class="track"><div class="fill" style="width:${(100 * v) / maxr}%"></div></div>`).join("") || "<span class='meta'>none</span>"}</div>
          <p class="hint">A design can miss several requirements. “Surrogate unsure”: the five networks disagree, so the design is kept off the front until verified.</p></div>
      </div>
    </div>
    <div class="panel" id="detail"><div class="empty"><p>Click a design.</p></div></div>
    <div class="panel">${window.renderScope ? renderScope(META.scope) : ""}</div>`;
  $$(".pick").forEach((b) => b.addEventListener("click", () => selectDesign(+b.dataset.i)));
  drawScatter();
}

function co2col(t) { t = Math.min(1, Math.max(0, t)); const a = [22, 163, 74], b = [250, 204, 21], c = [220, 38, 38]; const [p, q, u] = t < 0.5 ? [a, b, t * 2] : [b, c, (t - 0.5) * 2]; return `rgb(${p.map((v, k) => Math.round(v + u * (q[k] - v)))})`; }
function drawScatter() {
  const R = RES, el = $("#scatter");
  const W = Math.max(400, el.clientWidth || 700), H = 400, L = 58, Rm = 14, T = 12, Bm = 40;
  const pts = R.designs.map((d, i) => ({ ...d, i })).filter((d) => isFinite(d.vmax_kt) && isFinite(d.v_stall_kt));
  const base = R.baseline, lim = R.form.requirements.max_stall_speed_kt;
  const q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(p * (s.length - 1))]; };
  const xs = pts.map((p) => p.v_stall_kt), ys = pts.map((p) => p.vmax_kt);
  let x0 = q(xs, 0.01), x1 = q(xs, 0.99), y0 = q(ys, 0.01), y1 = q(ys, 0.995);
  const front = R.pareto.map((i) => R.designs[i]);
  for (const d of front.concat([base])) { x0 = Math.min(x0, d.v_stall_kt); x1 = Math.max(x1, d.v_stall_kt); y0 = Math.min(y0, d.vmax_kt); y1 = Math.max(y1, d.vmax_kt); }
  x1 = Math.max(x1, lim + 1);
  const px = (x1 - x0) * 0.05, py = (y1 - y0) * 0.06; x0 -= px; x1 += px; y0 -= py; y1 += py;
  const feas = pts.filter((p) => p.feasible && p.trust !== "low");
  const cvals = (feas.length ? feas : pts).map((p) => p.co2_100km), c0 = q(cvals, 0.02), c1 = q(cvals, 0.98);
  $("#co2r") && ($("#co2r").textContent = `${f(c0, 1)} → ${f(c1, 1)} kg`);
  const sx = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - Rm), sy = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - Bm);
  const ticks = (lo, hi, n) => { const st = Math.pow(10, Math.floor(Math.log10((hi - lo) / n))); const m = [1, 2, 5, 10].find((k) => (hi - lo) / (k * st) <= n) * st; const o = []; for (let v = Math.ceil(lo / m) * m; v <= hi; v += m) o.push(+v.toFixed(6)); return o; };
  let g = "";
  for (const t of ticks(x0, x1, 8)) g += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${T}" y2="${H - Bm}" stroke="#eef1f5"/><text x="${sx(t)}" y="${H - Bm + 15}" text-anchor="middle">${t}</text>`;
  for (const t of ticks(y0, y1, 6)) g += `<line x1="${L}" x2="${W - Rm}" y1="${sy(t)}" y2="${sy(t)}" stroke="#eef1f5"/><text x="${L - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`;
  g += `<text x="${(L + W - Rm) / 2}" y="${H - 6}" text-anchor="middle" style="font-weight:600">stall speed, clean wing (kt)  ← shorter runway, gentler landing</text>`;
  g += `<text transform="translate(14 ${(T + H - Bm) / 2}) rotate(-90)" text-anchor="middle" style="font-weight:600">top speed (kt)  → faster</text>`;
  if (lim < x1) g += `<rect x="${sx(lim)}" y="${T}" width="${W - Rm - sx(lim)}" height="${H - T - Bm}" fill="#fde8e7" fill-opacity=".55"/><line x1="${sx(lim)}" x2="${sx(lim)}" y1="${T}" y2="${H - Bm}" stroke="#b42318" stroke-dasharray="5 4"/><text x="${sx(lim) + 5}" y="${T + 14}" style="fill:#b42318;font-weight:700">stall limit ${lim} kt</text>`;
  const inb = (p) => p.v_stall_kt >= x0 && p.v_stall_kt <= x1 && p.vmax_kt >= y0 && p.vmax_kt <= y1;
  for (const p of pts.filter((p) => !p.feasible && inb(p))) g += `<circle cx="${sx(p.v_stall_kt)}" cy="${sy(p.vmax_kt)}" r="2.2" fill="#cbd5e1" data-i="${p.i}"/>`;
  for (const p of pts.filter((p) => p.feasible && p.trust === "low" && inb(p))) g += `<circle cx="${sx(p.v_stall_kt)}" cy="${sy(p.vmax_kt)}" r="2.6" fill="#fff" stroke="#f59e0b" stroke-width="1.2" data-i="${p.i}"/>`;
  for (const p of feas.filter(inb)) g += `<circle cx="${sx(p.v_stall_kt)}" cy="${sy(p.vmax_kt)}" r="2.8" fill="${co2col((p.co2_100km - c0) / (c1 - c0 || 1))}" fill-opacity=".8" data-i="${p.i}"/>`;
  if (front.length) {
    const env = [...front].sort((a, b) => a.v_stall_kt - b.v_stall_kt); let best = -1e9; const line = [];
    for (const d of env) if (d.vmax_kt > best) { best = d.vmax_kt; line.push(d); }
    g += `<polyline fill="none" stroke="#111" stroke-width="1.6" stroke-opacity=".6" points="${line.map((d) => `${sx(d.v_stall_kt)},${sy(d.vmax_kt)}`).join(" ")}"/>`;
    for (const i of R.pareto) { const d = R.designs[i]; g += `<circle cx="${sx(d.v_stall_kt)}" cy="${sy(d.vmax_kt)}" r="4.2" fill="${co2col((d.co2_100km - c0) / (c1 - c0 || 1))}" stroke="#111" stroke-width="1.4" data-i="${i}"/>`; }
  }
  const labs = {};
  for (const [k, lab] of [["fastest", "fastest"], ["greenest", "lowest CO₂"], ["balanced", "balanced"], ["safest", "lowest stall"]]) {
    const i = R.picks[k]; if (i !== undefined) (labs[i] = labs[i] || []).push(lab);
  }
  const placed = [];
  for (const [i, ls] of Object.entries(labs)) { const d = R.designs[i];
    let x = sx(d.v_stall_kt) - 8, y = sy(d.vmax_kt) - 9;
    while (placed.some(([px, py]) => Math.abs(px - x) < 110 && Math.abs(py - y) < 14)) y -= 15;
    placed.push([x, y]);
    g += `<text x="${x}" y="${y}" text-anchor="end" style="font-weight:700;fill:#111;paint-order:stroke;stroke:#fff;stroke-width:3px">${ls.join(" · ")}</text>`; }
  const bx = sx(base.v_stall_kt), by = sy(base.vmax_kt);
  g += `<text x="${bx}" y="${by + 6}" text-anchor="middle" style="font-size:18px;fill:#111">★</text><text x="${bx + 10}" y="${by + 16}" style="font-weight:700;fill:#111">NACA 2412</text>`;
  if (SEL !== null && R.designs[SEL] && isFinite(R.designs[SEL].vmax_kt)) { const d = R.designs[SEL]; g += `<circle cx="${sx(d.v_stall_kt)}" cy="${sy(d.vmax_kt)}" r="9" fill="none" stroke="#0369a1" stroke-width="2.5"/>`; }
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}">${g}<line x1="${L}" x2="${L}" y1="${T}" y2="${H - Bm}" stroke="#c9d1dd"/><line x1="${L}" x2="${W - Rm}" y1="${H - Bm}" y2="${H - Bm}" stroke="#c9d1dd"/></svg><div class="tip"></div>`;
  const svg = $("svg", el), tip = $(".tip", el);
  const near = (ev) => {
    const r = svg.getBoundingClientRect(), k = W / r.width, mx = (ev.clientX - r.left) * k, my = (ev.clientY - r.top) * k;
    let best = null, bd = 64;
    for (const c of svg.querySelectorAll("circle[data-i]")) { const dx = +c.getAttribute("cx") - mx, dy = +c.getAttribute("cy") - my, dd = dx * dx + dy * dy; if (dd <= bd) { bd = dd; best = c; } }
    return best ? +best.dataset.i : null;
  };
  svg.addEventListener("mousemove", (ev) => {
    const i = near(ev); if (i === null) { tip.style.display = "none"; return; }
    const d = R.designs[i];
    tip.innerHTML = `${f(d.vmax_kt)} kt · stall ${f(d.v_stall_kt)} kt · ${f(d.co2_100km, 2)} kg CO₂/100 km · span ${f(d.span, 1)} m · SM ${f(100 * d.static_margin, 0)} %<br>${d.feasible ? (d.trust === "low" ? "flyable, surrogate unsure" : "flyable") : "misses: " + esc((d.violations || []).join(", "))}`;
    const r = el.getBoundingClientRect(); tip.style.display = "block"; tip.style.left = Math.min(r.width - 300, ev.clientX - r.left + 12) + "px"; tip.style.top = ev.clientY - r.top + 12 + "px";
  });
  svg.addEventListener("mouseleave", () => (tip.style.display = "none"));
  svg.addEventListener("click", (ev) => { const i = near(ev); if (i !== null) selectDesign(i); });
}

let VIEWER = null, SECTION = null;
async function selectDesign(i) {
  SEL = i; drawScatter();
  $$(".pick").forEach((b) => b.classList.toggle("sel", +b.dataset.i === i));
  const det = $("#detail");
  if (!$("#viewer", det)) det.innerHTML = `<div class="meta"><span class="spinner" style="border-color:#0369a155;border-top-color:#0369a1"></span> building the aircraft…</div>`;
  try {
    DETAIL = await post("/api/aircraft", { x: RES.designs[i].x, aircraft: RES.form.aircraft, requirements: RES.form.requirements });
    renderDetail();
  } catch (e) { det.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}

function foilSVG(d, base) {
  const W = 520, H = 120, s = W / 1.08, ox = 0.04 * W, oy = H / 2 + 6;
  const path = (o) => "M" + o.x.map((x, k) => `${ox + x * s},${oy - o.upper[k] * s}`).join("L") + "L" + o.x.map((x, k) => `${ox + x * s},${oy - o.lower[k] * s}`).reverse().join("L") + "Z";
  return `<svg viewBox="0 0 ${W} ${H}"><path d="${path(base)}" fill="none" stroke="#111" stroke-width="1.2" stroke-dasharray="4 3"/>
    <path d="${path(d)}" fill="#0369a122" stroke="#0369a1" stroke-width="2"/><text x="${W - 4}" y="${H - 4}" text-anchor="end" style="font-size:11px;fill:#5d6b82">— wing section   - - NACA 2412</text></svg>`;
}

function planSVG(d, b) {
  // top view of both wings (and the baseline dashed), from the planform numbers
  const W = 520, H = 130, sc = W / 14.5, cx = W / 2, top = 10;
  const wing = (sm, p, dash) => {
    const b2 = sm.span / 2, cr = sm.root_chord, ct = sm.tip_chord, sw = Math.tan((p.sweep_le * Math.PI) / 180) * b2, x0 = p.wing_x - 1.7;
    const pts = [[-b2, x0 + sw], [0, x0], [b2, x0 + sw], [b2, x0 + sw + ct], [0, x0 + cr], [-b2, x0 + sw + ct]];
    return `<polygon points="${pts.map(([y, x]) => `${cx + y * sc},${top + x * sc}`).join(" ")}" fill="${dash ? "none" : "#0369a122"}" stroke="${dash ? "#111" : "#0369a1"}" stroke-width="${dash ? 1.2 : 2}" ${dash ? 'stroke-dasharray="4 3"' : ""}/>`;
  };
  return `<svg viewBox="0 0 ${W} ${H}">${wing(b.summary, b.plan, true)}${wing(d.summary, d.plan, false)}
    <text x="${W - 4}" y="${H - 4}" text-anchor="end" style="font-size:11px;fill:#5d6b82">— this wing   - - Cessna 172 class (top view)</text></svg>`;
}

function renderDetail() {
  const d = DETAIL, B = RES.baseline, el = $("#detail");
  if (!d.valid) { el.innerHTML = `<div class="banner bad">Invalid geometry: ${esc((d.violations || []).join(", "))}</div>`; return; }
  const fmtv = (c) => c.unit === "m/s" ? f(c.value / KT, 1) + " kt" : c.unit === "η" ? f(100 * c.value, 0) + " % span" : c.unit === "MAC" ? f(100 * c.value, 1) + " % MAC"
    : c.unit === "deg" ? f(c.value, 1) + "°" : c.unit === "m" ? f(c.value, 2) + " m" : c.name.startsWith("Thickness") ? f(100 * c.value, 1) + " %" : f(c.value, 2);
  const fmtl = (c) => c.kind === "range" ? `${f(100 * c.limit, 0)} – ${f(100 * c.limit_hi, 0)} %` : (c.kind === "min" ? "≥ " : "≤ ") +
    (c.unit === "m/s" ? f(c.limit / KT, 1) + " kt" : c.unit === "η" ? f(100 * c.limit, 0) + " %" : c.unit === "deg" ? f(c.limit, 1) + "°" : c.unit === "m" ? f(c.limit, 1) + " m" : c.name.startsWith("Thickness") ? f(100 * c.limit, 1) + " %" : f(c.limit, 2));
  const chk = d.checks.map((c) => `<tr><td>${c.ok ? "✅" : "❌"} ${esc(c.name)}</td><td class="num">${fmtv(c)}</td><td class="num">${fmtl(c)}</td><td class="meta">${esc(c.why)}</td></tr>`).join("");
  const kv = [["Top speed", `${f(d.vmax_kt)} kt`, delta(d.vmax_kt, B.vmax_kt, 1, " kt")],
    ["CO₂ per 100 km", `${f(d.co2_100km, 2)} kg`, delta(pct(d.co2_100km, B.co2_100km), 0, 1, " %", "down")],
    ["Fuel per 100 km", `${f(d.fuel_l_100km, 1)} L`, delta(d.fuel_l_100km, B.fuel_l_100km, 2, " L", "down")],
    ["Stall speed (clean)", `${f(d.v_stall_kt)} kt`, delta(d.v_stall_kt, B.v_stall_kt, 1, " kt", "down")],
    ["Cruise L/D (aircraft)", f(d.cruise_ld, 1), delta(d.cruise_ld, B.cruise_ld, 1)],
    ["Mass", `${f(d.mass, 0)} kg`, delta(d.mass, B.mass, 0, " kg", "down")]];
  const p = d.plan, sm = d.summary;
  const geo = [["Wing area / span", `${f(p.wing_area, 2)} m² · ${f(d.span, 2)} m`], ["Aspect ratio · taper", `${f(p.aspect_ratio, 2)} · ${f(p.taper, 2)}`],
    ["Sweep (LE) · tip twist", `${f(p.sweep_le, 1)}° · ${f(p.twist, 1)}°`], ["Wing position (root LE)", `${f(p.wing_x, 2)} m from the nose`],
    ["Chords root / tip / MAC", `${f(sm.root_chord, 2)} / ${f(sm.tip_chord, 2)} / ${f(sm.mac, 2)} m`],
    ["Horizontal / vertical tail", `${f(sm.htail_area, 2)} m² · ${f(sm.vtail_area, 2)} m²`],
    ["CG · neutral point", `${f(d.x_cg, 2)} m · ${f(d.x_np, 2)} m (margin ${f(100 * d.static_margin, 1)} % MAC)`],
    ["Cruise α · tail incidence", `${f(d.cruise_alpha, 1)}° · ${f(d.tail_incidence, 1)}°`], ["Span efficiency (Oswald, inviscid)", f(d.oswald, 3)],
    ["Section t/c · camber", `${f(100 * d.props.thickness, 1)} % · ${f(100 * d.props.camber, 1)} %`],
    ["Wing / tail mass", `${f(d.weights.wing, 0)} kg · ${f(d.weights.htail + d.weights.vtail, 0)} kg`]];
  const db = d.drag_breakdown, tot = Object.values(db).reduce((a, b) => a + b, 0);
  const dcol = { induced: "#0369a1", wing_profile: "#38bdf8", tail_profile: "#a78bfa", fuselage_vtail_misc: "#94a3b8" };
  const dlab = { induced: "induced", wing_profile: "wing profile", tail_profile: "tail profile", fuselage_vtail_misc: "fuselage, fin, gear, misc." };
  const xs = d.x.map((v) => (+v).toFixed(5)).join(",");
  const ver = d.openfoam;
  const firstTime = !$("#viewer", el);
  el.innerHTML = `
    <div class="toolbar"><h2 style="margin:0">Aircraft ${SEL} ${d.feasible ? `<span class="badge pass">flyable</span>` : `<span class="badge fail">not flyable</span>`} <span class="trust ${d.trust}">${d.trust} trust</span> ${ver ? `<span class="badge pass">run in OpenFOAM 3D</span>` : ""}</h2>
      <div class="dl"><a href="/api/aircraft.glb?x=${xs}${ver ? "&cp=1" : ""}" download="aircraft.glb">.glb <small>Blender · Unreal · Unity</small></a><a href="/api/aircraft.usda?x=${xs}">.usda <small>Omniverse</small></a><a href="/api/aircraft.stl?x=${xs}">.stl <small>CFD</small></a><a href="#" id="dl-json">.json</a></div></div>
    <div id="viewer"></div>
    ${ver ? `<div class="banner ok" style="margin-top:10px">OpenFOAM 3D (${(ver.cells / 1e3).toFixed(0)}k cells, k-ω SST, α ${f(ver.alpha, 1)}°): CL ${f(ver.CL, 3)} vs ${f(ver.model.CL, 3)} from the vortex lattice, CD ${f(ver.CD, 4)} vs ${f(ver.model.CD, 4)}. Pressure on the skin and streamlines in the viewer come from that run.</div>` : ""}
    <div class="dpanel" style="margin-top:14px">
      <div>
        <table class="t"><tr><th></th><th>this aircraft</th><th>vs Cessna 172 class</th></tr>${kv.map(([k, v, dd]) => `<tr><td>${k}</td><td class="num"><b>${v}</b></td><td class="num">${dd}</td></tr>`).join("")}</table>
        <h3>Requirements</h3><table class="t">${chk}</table>
        <h3>Configuration</h3><table class="t">${geo.map(([k, v]) => `<tr><td>${k}</td><td class="num">${v}</td></tr>`).join("")}</table>
        <h3>Drag in cruise (CD ${f(tot, 4)})</h3>
        <div class="dbar">${Object.entries(db).map(([k, v]) => `<div title="${dlab[k]} ${f(v, 5)}" style="width:${(100 * v) / tot}%;background:${dcol[k]}"></div>`).join("")}</div>
        <div class="legend">${Object.entries(db).map(([k, v]) => `<span><i style="background:${dcol[k]}"></i>${dlab[k]} ${f((100 * v) / tot, 0)} %</span>`).join("")}</div>
      </div>
      <div>
        <h3 style="margin-top:0">Top view and wing section</h3>
        <div class="foil">${planSVG(d, B)}</div><div class="foil">${foilSVG(d.outline, B.outline)}</div>
        <h3>Where the wing stalls first</h3><div id="ch-load"></div>
        <p class="hint">Local cl / cl max along the half-span when the first section reaches its maximum (vortex lattice, trimmed). Stall must start inboard of ${f(100 * RES.form.requirements.max_stall_station, 0)} % of the half-span so the ailerons keep working.</p>
        <h3>Section polar (surrogate, band = network spread)</h3><div id="ch-cl"></div><div id="ch-cd"></div>
        <details class="codebox" id="secbox"><summary><b>Wing section flow field (graph network)</b></summary><div id="secflow"><div class="meta">loading…</div></div></details>
      </div>
    </div>`;
  const P = d.polar, Bp = B.polar;
  PChart.mount($("#ch-cl"), { height: 150, x: P.alpha, yLabel: "cl vs α (°)", series: [
    { name: "this section", y: P.cl, color: "#0369a1", lo: P.cl.map((v, k) => v - 2 * P.cl_sd[k]), hi: P.cl.map((v, k) => v + 2 * P.cl_sd[k]) },
    { name: "NACA 2412", y: Bp.cl, color: "#111", dash: "4 3" }] });
  PChart.mount($("#ch-cd"), { height: 150, x: P.alpha, yLabel: "cd vs α (°)", series: [
    { name: "this section", y: P.cd, color: "#0369a1", lo: P.cd.map((v, k) => v - 2 * P.cd_sd[k]), hi: P.cd.map((v, k) => v + 2 * P.cd_sd[k]) },
    { name: "NACA 2412", y: Bp.cd, color: "#111", dash: "4 3" }] });
  const L = d.loading, LB = B.loading;
  const half = (o) => { const idx = o.eta.map((e, k) => k).filter((k) => true); const pairs = idx.map((k) => [o.eta[k], o.stall_ratio[k]]).sort((a, b) => a[0] - b[0]); const seen = {}; return pairs.filter(([e]) => (seen[e.toFixed(4)] ? false : (seen[e.toFixed(4)] = true))); };
  const hd = half(L), hb = half(LB);
  const etas = hd.map((p) => +p[0].toFixed(3));
  PChart.mount($("#ch-load"), { height: 160, x: etas, yLabel: "cl / cl max  vs  η (0 root → 1 tip)", yMin: 0, yMax: 1.05,
    hlines: [{ y: 1, color: "#b42318", label: "stall", dash: "4 3" }],
    series: [{ name: "this wing", y: hd.map((p) => p[1]), color: "#0369a1" }, { name: "Cessna 172 class", y: etas.map((e) => interp1(hb.map((p) => p[0]), hb.map((p) => p[1]), e)), color: "#111", dash: "4 3" }] });
  $("#dl-json").onclick = (ev) => { ev.preventDefault(); const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify({ ...d, lines_vlm: undefined, lines_cfd: undefined, aircraft: RES.form.aircraft, requirements: RES.form.requirements }, null, 1)], { type: "application/json" })); a.download = `aircraft_${SEL}.json`; a.click(); };
  $("#secbox").addEventListener("toggle", async (ev) => { if (ev.target.open && !SECTION) loadSection(d); });
  const mk = () => {
    if (!VIEWER || !$("#viewer .av")) VIEWER = new window.ADO_AircraftViewer($("#viewer"));
    VIEWER.load({ glb: `/api/aircraft.glb?x=${xs}${ver ? "&cp=1" : ""}`, ground_z: d.ground_z, span: d.span,
      loading: { eta: hd.map((p) => p[0]), ratio: hd.map((p) => p[1]) }, lines_vlm: d.lines_vlm, lines_cfd: d.lines_cfd || [],
      cp_available: !!ver, title: `${f(d.vmax_kt)} kt · ${f(d.co2_100km, 1)} kg CO₂/100 km · stall ${f(d.v_stall_kt)} kt`,
      subtitle: `span ${f(d.span, 1)} m · AR ${f(p.aspect_ratio, 1)} · taper ${f(p.taper, 2)} · ${d.feasible ? "flyable" : "not flyable: " + esc(d.violations.join(", "))}` });
  };
  VIEWER = null; SECTION = null;
  if (window.ADO_AircraftViewer) mk(); else window.addEventListener("ado-viewer-ready", mk, { once: true });
}
function interp1(xs, ys, x) { if (x <= xs[0]) return ys[0]; for (let i = 1; i < xs.length; i++) if (x <= xs[i]) return ys[i - 1] + ((ys[i] - ys[i - 1]) * (x - xs[i - 1])) / (xs[i] - xs[i - 1]); return ys[ys.length - 1]; }

async function loadSection(d) {
  const box = $("#secflow");
  try {
    const sec = await post("/api/design", { x: d.x.slice(0, 6).concat([d.plan.wing_area]) });
    SECTION = sec; const keep = DETAIL; DETAIL = sec; FLOW.case = 0;
    box.innerHTML = `<div class="flowbar"><div class="seg" id="seg-case">${sec.flow.cases.map((c, k) => `<button data-k="${k}" class="${k === FLOW.case ? "on" : ""}">${esc(c.label)} · α ${f(c.alpha, 1)}°</button>`).join("")}</div>
      <div class="seg" id="seg-view"><button data-v="p" class="on">pressure Cp</button><button data-v="speed">speed</button><button data-v="nut">turbulence</button></div></div>
      <div class="flowbox"><canvas id="flow" width="900" height="420"></canvas></div><div id="fbar"></div><div id="ch-cp"></div>
      <p class="hint">MeshGraphNet trained on the 2D OpenFOAM runs: the flow around the wing section at the section's own cruise and near-stall angles.</p>`;
    const redraw = () => { const k = DETAIL; DETAIL = SECTION; drawFlow(); drawCp(); DETAIL = k; };
    $$("#seg-case button").forEach((b) => (b.onclick = () => { FLOW.case = +b.dataset.k; $$("#seg-case button").forEach((x) => x.classList.toggle("on", x === b)); redraw(); }));
    $$("#seg-view button").forEach((b) => (b.onclick = () => { FLOW.view = b.dataset.v; $$("#seg-view button").forEach((x) => x.classList.toggle("on", x === b)); redraw(); }));
    DETAIL = keep; redraw();
  } catch (e) { box.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}

// ------------------------------------------------------------------ flow field on the graph lattice
const CMAP = {
  p: [[0, [33, 102, 172]], [0.5, [247, 247, 247]], [1, [178, 24, 43]]],
  speed: [[0, [48, 18, 59]], [0.25, [70, 107, 227]], [0.5, [40, 188, 235]], [0.7, [164, 252, 60]], [0.85, [251, 185, 56]], [1, [122, 4, 3]]],
  nut: [[0, [255, 255, 255]], [0.4, [253, 208, 162]], [0.75, [241, 105, 19]], [1, [127, 39, 4]]],
};
function cmap(name, t) {
  const s = CMAP[name]; t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < s.length; i++) if (t <= s[i][0]) { const [a, ca] = s[i - 1], [b, cb] = s[i], u = (t - a) / (b - a); return ca.map((c, k) => Math.round(c + u * (cb[k] - c))); }
  return s[s.length - 1][1];
}
function drawFlow() {
  const fl = DETAIL.flow, c = fl.cases[FLOW.case], cv = $("#flow"), g = cv.getContext("2d");
  const W = cv.width, H = cv.height, x0 = -0.45, x1 = 1.75, s = W / (x1 - x0), y0 = (H / s) / 2;
  const X = (x) => (x - x0) * s, Y = (y) => (y0 - y) * s;
  const v = FLOW.view === "p" ? c.p.map((p) => p / 0.5) : FLOW.view === "speed" ? c.speed : c.nut;
  let lo, hi;
  if (FLOW.view === "p") { lo = -2.2; hi = 1; } else if (FLOW.view === "speed") { lo = 0; hi = 1.6; } else { lo = 0; hi = Math.max(...v) || 1; }
  g.fillStyle = "#fff"; g.fillRect(0, 0, W, H);
  const { ni, nj } = fl, px = fl.x, py = fl.y;
  const tnorm = (val) => FLOW.view === "p" ? (val < 0 ? 0.5 * (1 - val / lo) : 0.5 + 0.5 * val / hi) : (val - lo) / (hi - lo);
  for (let j = 0; j < nj - 1; j++) for (let i = 0; i < ni; i++) {
    const a = j * ni + i, b = j * ni + ((i + 1) % ni), cc = (j + 1) * ni + ((i + 1) % ni), d = (j + 1) * ni + i;
    if (px[a] < x0 - 1 && px[d] < x0 - 1) continue;
    const val = (v[a] + v[b] + v[cc] + v[d]) / 4, col = cmap(FLOW.view, tnorm(val));
    g.fillStyle = `rgb(${col})`; g.strokeStyle = g.fillStyle; g.lineWidth = 0.6;
    g.beginPath(); g.moveTo(X(px[a]), Y(py[a])); g.lineTo(X(px[b]), Y(py[b])); g.lineTo(X(px[cc]), Y(py[cc])); g.lineTo(X(px[d]), Y(py[d])); g.closePath(); g.fill(); g.stroke();
  }
  const o = DETAIL.outline;
  g.beginPath(); o.x.forEach((x, k) => (k ? g.lineTo(X(x), Y(o.upper[k])) : g.moveTo(X(x), Y(o.upper[k]))));
  [...o.x].reverse().forEach((x, k) => g.lineTo(X(x), Y(o.lower[o.x.length - 1 - k]))); g.closePath();
  g.fillStyle = "#334155"; g.fill();
  const a = (c.alpha * Math.PI) / 180;
  g.strokeStyle = "#111"; g.fillStyle = "#111"; g.lineWidth = 2; const ax = 30, ay = H - 30;
  g.beginPath(); g.moveTo(ax, ay); g.lineTo(ax + 50 * Math.cos(a), ay - 50 * Math.sin(a)); g.stroke();
  g.font = "13px system-ui"; g.fillText(`free stream, α ${c.alpha.toFixed(1)}°`, ax + 60, ay + 4);
  const lab = { p: "pressure coefficient Cp", speed: "speed |U| / U∞", nut: "turbulent viscosity log(1 + νt/ν)" }[FLOW.view];
  const stops = Array.from({ length: 11 }, (_, k) => `rgb(${cmap(FLOW.view, k / 10)}) ${k * 10}%`).join(",");
  $("#fbar").innerHTML = `<div class="cbarh" style="background:linear-gradient(90deg,${stops})"></div><div class="cbarl"><span>${f(lo, 2)}</span><span>${lab}</span><span>${f(hi, 2)}</span></div>`;
}
function drawCp() {
  const fl = DETAIL.flow, c = fl.cases[FLOW.case], ni = fl.ni;
  const xw = fl.x.slice(0, ni), cp = c.cp_wall;
  const up = [], lo = [];
  for (let i = 0; i < ni; i++) { if (fl.side[i] === "t") continue; (fl.side[i] === "u" ? up : lo).push([xw[i], cp[i]]); }
  up.sort((a, b) => a[0] - b[0]); lo.sort((a, b) => a[0] - b[0]);
  const grid = Array.from({ length: 41 }, (_, k) => 0.5 * (1 - Math.cos((Math.PI * k) / 40)));
  const it = (arr) => grid.map((x) => { let k = arr.findIndex((p) => p[0] >= x); if (k <= 0) return arr[Math.max(0, k)]?.[1] ?? null; const [xa, ya] = arr[k - 1], [xb, yb] = arr[k]; return ya + ((yb - ya) * (x - xa)) / (xb - xa + 1e-12); });
  PChart.mount($("#ch-cp"), { height: 170, x: grid.map((v) => +v.toFixed(3)), yLabel: "−Cp along the chord (x/c)", series: [
    { name: "upper surface", y: it(up).map((v) => (v === null ? null : -v)), color: "#0369a1" }, { name: "lower surface", y: it(lo).map((v) => (v === null ? null : -v)), color: "#ea580c" }] });
}

// ------------------------------------------------------------------ engine tab
function parity(points, k, label, log) {
  const W = 300, H = 260, L = 48, T = 10, Bm = 34, R = 10;
  const t = points.map((p) => p.true[k]), q = points.map((p) => p.pred[k]);
  let lo = Math.min(...t, ...q), hi = Math.max(...t, ...q);
  const tr = (v) => (log ? Math.log10(v) : v); lo = tr(lo); hi = tr(hi); const pd = (hi - lo) * 0.05; lo -= pd; hi += pd;
  const sx = (v) => L + ((tr(v) - lo) / (hi - lo)) * (W - L - R), sy = (v) => T + (1 - (tr(v) - lo) / (hi - lo)) * (H - T - Bm);
  let g = `<line x1="${sx(log ? 10 ** lo : lo)}" y1="${sy(log ? 10 ** lo : lo)}" x2="${sx(log ? 10 ** hi : hi)}" y2="${sy(log ? 10 ** hi : hi)}" stroke="#94a3b8" stroke-dasharray="4 3"/>`;
  points.forEach((p, i) => { g += `<circle cx="${sx(t[i])}" cy="${sy(q[i])}" r="2.6" fill="${p.id.startsWith("NACA") ? "#ea580c" : "#0369a1"}" fill-opacity=".7"><title>${esc(p.id)} α ${f(p.alpha, 1)}°: OpenFOAM ${f(t[i], 4)}, predicted ${f(q[i], 4)}</title></circle>`; });
  return `<svg viewBox="0 0 ${W} ${H}"><rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - Bm}" fill="none" stroke="#e2e8f0"/>${g}
    <text x="${(L + W) / 2}" y="${H - 8}" text-anchor="middle">OpenFOAM ${label}</text><text transform="translate(12 ${(H - Bm) / 2}) rotate(-90)" text-anchor="middle">predicted ${label}</text></svg>`;
}
async function renderEngine() {
  const M = META, m = M.metrics || {}, ds = M.dataset || {}, V = M.verification || {};
  const S = await (await fetch("/api/surrogate")).json().catch(() => ({}));
  const row = (name, mm) => mm ? `<tr><td>${name}</td><td class="num">${f(mm.Cl.mae, 3)}</td><td class="num">${f(mm.Cl.r2, 3)}</td><td class="num">${f(mm.Cd.mape, 1)} %</td><td class="num">${f(mm.Cd.p90, 1)} %</td><td class="num">${f(mm.Cm.mae, 4)}</td></tr>` : "";
  const vr = (V.rounds || []).flatMap((r) => r.designs.map((d) => ({ ...d, round: r.round })));
  $("#engine").innerHTML = `
    <div class="panel"><h2 style="margin-top:0">How the engine works</h2><div class="pipeline" id="pipe-big"></div>
      <div class="grid2"><div><h3>CFD: validated before use</h3><table class="t"><tr><th>Case (NASA turbulence-model resource)</th><th>OpenFOAM here</th><th>Reference</th></tr>
        ${(ds.cfd_validation || []).map((r) => `<tr><td>${esc(r.case)}</td><td class="num">${esc(r.here)}</td><td class="num">${esc(r.ref)}</td></tr>`).join("")}</table>
        <p class="hint">${esc(ds.cfd_note || "")}</p></div>
      <div><h3>Training data</h3><div class="kpis">${(ds.kpis || []).map((k) => `<div class="kpi"><div class="l">${esc(k.l)}</div><div class="v">${esc(k.v)}</div><div class="s">${esc(k.s)}</div></div>`).join("")}</div>
        <div class="foil" id="ds-foils" style="margin-top:10px"></div></div></div></div>
    <div class="panel"><h2 style="margin-top:0">Surrogates on airfoils they never saw</h2>
      <table class="t"><tr><th>Model · test set</th><th>Cl MAE</th><th>Cl R²</th><th>Cd error (mean)</th><th>Cd error (90 %)</th><th>Cm MAE</th></tr>
        ${row("MLP ensemble · held-out LHS airfoils", m.mlp_test)}${row("MLP ensemble · NACA 2412/4412/0012/4415", m.mlp_reference)}
        ${row("MeshGraphNet · held-out LHS airfoils", m.gnn_test)}${row("MeshGraphNet · NACA airfoils", m.gnn_reference)}</table>
      <div class="parity" style="margin-top:12px">${S.mlp_test_points ? parity(S.mlp_test_points.concat(S.mlp_reference_points), 0, "cl") + parity(S.mlp_test_points.concat(S.mlp_reference_points), 1, "cd", true) : ""}
        ${S.gnn_test_points ? parity(S.gnn_test_points.concat(S.gnn_reference_points), 0, "cl") + parity(S.gnn_test_points.concat(S.gnn_reference_points), 1, "cd", true) : ""}</div>
      <p class="hint">Left pair: MLP ensemble; right pair: MeshGraphNet. Blue: held-out airfoils from the design space; orange: NACA airfoils (never in training). Split by airfoil, not by run: a test airfoil is unseen at every angle.</p></div>
    <div class="panel"><h2 style="margin-top:0">Optimizer picks checked with OpenFOAM</h2>
      ${vr.length ? `<table class="t vtable"><tr><th>Design</th><th>wing</th><th>top speed: surrogate / OpenFOAM</th><th>CO₂ /100 km</th><th>stall speed</th><th>flyable</th></tr>
        ${vr.map((d) => `<tr><td>${esc(d.id)}</td><td class="num">${f(d.wing_area, 1)} m²</td><td class="num">${f(d.surrogate.vmax_kt)} / <b>${f(d.openfoam.vmax_kt)}</b> kt</td><td class="num">${f(d.surrogate.co2_100km, 2)} / <b>${f(d.openfoam.co2_100km, 2)}</b></td><td class="num">${f(d.surrogate.v_stall_kt)} / <b>${f(d.openfoam.v_stall_kt)}</b> kt</td><td class="${d.openfoam.feasible ? "ok" : "bad"}">${d.surrogate.feasible ? "yes" : "no"} / ${d.openfoam.feasible ? "yes" : "no: " + esc((d.openfoam.violations || []).join(", "))}</td></tr>`).join("")}</table>
        ${(V.rounds || []).map((r) => r.summary ? `<p class="hint">${esc(r.summary)}</p>` : "").join("")}` : "<p class='meta'>No verification yet.</p>"}</div>
    <div class="panel" id="eng3d"></div>
    <div class="panel">${window.renderScope ? renderScope(M.scope) : ""}</div>`;
  pipeline($("#pipe-big"));
  render3dEngine();
  if (ds.outlines) {
    const W = 520, H = 130, s = W / 1.08, ox = 0.04 * W, oy = H / 2 + 4;
    const p = (o) => "M" + o.x.map((x, k) => `${ox + x * s},${oy - o.upper[k] * s}`).join("L") + "L" + o.x.map((x, k) => `${ox + x * s},${oy - o.lower[k] * s}`).reverse().join("L") + "Z";
    $("#ds-foils").innerHTML = `<svg viewBox="0 0 ${W} ${H}">${ds.outlines.map((o) => `<path d="${p(o)}" fill="none" stroke="#0369a1" stroke-opacity=".18"/>`).join("")}</svg><div class="meta">the ${ds.outlines.length} training airfoils, overlaid</div>`;
  }
}
function render3dEngine() {
  const m3 = META.m3 || {}, V = m3.verification || {}, el = $("#eng3d");
  const rows = (V.designs || []).map((d) => `<tr><td>${esc(d.label)}</td><td class="num">${(d.cells / 1e3).toFixed(0)}k</td><td class="num">${f(d.alpha, 1)}°</td>
    <td class="num">${f(d.model.CL, 3)} / <b>${f(d.CL, 3)}</b></td><td class="num">${f(d.model.CD, 4)} / <b>${f(d.CD, 4)}</b></td><td class="num">${f(d.CD_pressure, 4)} + ${f(d.CD_friction, 4)}</td></tr>`).join("");
  const renders = (m3.renders || []).filter((r) => r.endsWith(".jpg") || r.endsWith(".png"));
  const cap = (r) => (V.captions || {})[r] || r.replace(/[_-]/g, " ").replace(/\.(png|jpg)$/, "");
  el.innerHTML = `<h2 style="margin-top:0">The whole aircraft in 3D</h2>
    <div class="grid2"><div><h3>Vortex lattice, checked against theory</h3><table class="t"><tr><th>Wing (flat plate)</th><th>here</th><th>theory</th></tr>
      <tr><td>Elliptic loading, Trefftz-plane drag</td><td class="num">e = 1.0000</td><td class="num">1</td></tr>
      <tr><td>Rectangular, AR 6: lift slope · Oswald</td><td class="num">4.21 /rad · 0.984</td><td class="num">≈4.2 /rad · ≈0.98</td></tr>
      <tr><td>Taper 0.4, AR 8: Oswald</td><td class="num">0.995</td><td class="num">≈0.99</td></tr></table>
      <p class="hint">Wing and tail as horseshoe vortices (the tail sits in the wing's downwash), trimmed with the tail at every flight condition; cosine spacing with θ-midpoint control points makes the induced drag exact for elliptic loading.</p></div>
    <div><h3>The Cessna 172 class, as a check</h3><table class="t"><tr><th></th><th>model</th><th>published</th></tr>
      <tr><td>Zero-lift drag CD0</td><td class="num">0.030</td><td class="num">≈0.031</td></tr>
      <tr><td>Top speed (full power, 2000 m)</td><td class="num">124 kt</td><td class="num">≈123 kt (C172N)</td></tr>
      <tr><td>Fuel at 107 kt</td><td class="num">14.4 L/100 km</td><td class="num">≈15</td></tr>
      <tr><td>Wing mass (Raymer, general aviation)</td><td class="num">143 kg</td><td class="num">≈100–150 kg</td></tr>
      <tr><td>Stall starts at (3° washout)</td><td class="num">≈20 % of the half-span</td><td class="num">at the root, by design</td></tr></table>
      <p class="hint">Miscellaneous drag (gear, struts, cooling, interference) is calibrated once on this aircraft and kept for every design.</p></div></div>
    <h3>Checked with OpenFOAM in 3D</h3>
    ${rows ? `<table class="t"><tr><th>Aircraft</th><th>cells</th><th>α</th><th>CL: model / OpenFOAM</th><th>CD: model / OpenFOAM</th><th>OpenFOAM CD: pressure + friction</th></tr>${rows}</table>
      <p class="hint">${esc(V.note || "")}</p>` : "<p class='meta'>No 3D run stored.</p>"}
    ${renders.length ? `<h3>Rendered in Blender (Cycles)</h3><div class="renders">${renders.map((r) => `<figure><a href="/renders/${r}" target="_blank"><img src="/renders/${r}" loading="lazy"></a><figcaption>${esc(cap(r))}</figcaption></figure>`).join("")}</div>` : ""}`;
}
function pipeline(el) {
  const ds = (META.dataset || {}), m = META.metrics || {};
  const st = [["Aircraft", "airfoil (6) + area, AR, taper, sweep, twist, position", "12"], ["Mesh", "same 20k-cell O-grid for every section", "20k"],
    ["OpenFOAM", `simpleFoam, k-ω SST, y⁺<1, Re 4·10⁶`, ds.runs ? String(ds.runs) : "—"],
    ["Graph network", "MeshGraphNet on the grid: flow field + cl, cd, cm", "GNN"], ["Ensemble", "5 MLPs vote: section polars + uncertainty", "×5"],
    ["3D aircraft", "vortex lattice, trimmed; drag build-up; weights; stability", "3D"],
    ["NSGA-II", "faster, less CO₂, slower stall, under 8 requirements", "∞"], ["Verify", "OpenFOAM 2D polars and 3D aircraft runs", "✓"]];
  el.innerHTML = st.map(([b, s, n], k) => `${k ? '<div class="ar">→</div>' : ""}<div class="st"><span class="n">${esc(n)}</span><b>${esc(b)}</b>${esc(s)}</div>`).join("");
}

(async function init() {
  META = await (await fetch("/api/meta")).json();
  META.m3 = await (await fetch("/api/meta3d")).json();
  fillForm(META.aircraft, META.m3.requirements);
  $("#presets").innerHTML = PRESETS.map((p, k) => `<button type="button" data-k="${k}"><b>${esc(p.name)}</b>${esc(p.note)}</button>`).join("");
  $$("#presets button").forEach((b) => b.addEventListener("click", () => {
    const p = PRESETS[+b.dataset.k];
    fillForm({ ...META.aircraft, ...p.ac }, { ...META.m3.requirements, ...p.req });
    $("#run").click();
  }));
  pipeline($("#pipe-small"));
  renderEngine();
  const host = location.origin;
  $("#c1").textContent = `# search: returns every design (flyable or not, why), the Pareto front and three picks
curl -s ${host}/api/optimize -H 'Content-Type: application/json' -d '{"aircraft": {"mass": 1100, "power_kw": 120}, "requirements": {"max_stall_speed_kt": 61}, "population": 80, "generations": 60}' | jq '.picks, .feasible, .reasons'

# one design (6 shape weights + wing area): performance, checks, polar, flow field from the graph network
curl -s ${host}/api/design -H 'Content-Type: application/json' -d '{"x": [0.17, 0.145, 0.131, 0.025, 0.071, 0.059, 16.2]}' | jq '.vmax_kt, .co2_100km, .checks'

# a ready OpenFOAM case for it (mesh, setup, a coefficients script)
curl -s -o case.zip "${host}/api/openfoam-case?x=0.17,0.145,0.131,0.025,0.071,0.059,16.2&alpha=2"`;
  $("#c2").textContent = `from pinneapple_design.aero.geometry import naca4, properties
from pinneapple_design.aero.case import run_case              # needs OpenFOAM
from pinneapple_design.aero.aircraft import Aircraft, Requirements, Polar, evaluate
from pinneapple_design.aero.optimize import Engine

shape = naca4("2412")                                         # 6 CST weights
r = run_case("runs/naca2412_a4", shape, alpha_deg=4.0)        # simpleFoam k-omega SST, ~1 min
print(r["Cl"], r["Cd"], r["Cm"], r["status"])

engine = Engine.load("apps/aero_optimizer/model/surrogate.pt")
res = engine.search(Aircraft(mass=1100, power_kw=120), Requirements(), population=80, generations=60)
best = [res["designs"][i] for i in res["pareto"]]
print(len(res["designs"]), "designs,", len(best), "on the front")`;
})();
