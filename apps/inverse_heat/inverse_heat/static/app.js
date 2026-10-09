// Inverse Heat Lab: 1D fin (live, your readings), 2D plate (full run + live), 3D block (full run), code, API.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const f1 = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "—" : Number(v).toFixed(d));
const C = { pinn: "#c2410c", ref: "#172033", lsq: "#0b6bcb", truth: "#1a7f4b", read: "#172033" };
let META = null;

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : j.detail ? j.detail.map((d) => `${d.loc.slice(-1)[0]}: ${d.msg}`).join("; ") : `HTTP ${r.status}`);
  return j;
}

// ------------------------------------------------------------------ tabs
const loaded = {};
function openTab(name) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + name));
  if (!loaded[name]) { loaded[name] = true; ({ plate: loadPlate, block: loadBlock }[name] || (() => {}))(); }
  history.replaceState(null, "", "#" + name);
}
$$("#tabs button").forEach((b) => b.addEventListener("click", () => openTab(b.dataset.tab)));

// ------------------------------------------------------------------ job polling
async function poll(id, onTick) {
  let from = 0, hist = [];
  for (;;) {
    const j = await api(`/api/jobs/${id}?history_from=${from}`);
    hist = hist.concat(j.history); from = hist.length;
    onTick(j, hist);
    if (j.status !== "running") return { job: j, hist };
    await new Promise((r) => setTimeout(r, 700));
  }
}
function liveBlock(label) {
  return `<div class="live"><span class="spinner"></span><span id="live-txt">${esc(label)}</span><div class="progress"><div id="live-bar"></div></div></div>
    <div class="fig" style="margin-top:12px"><div class="cap">h while the network trains (it starts from your guess)</div><div id="live-chart"></div></div>`;
}
function tickLive(j, hist, hTrue) {
  const t = $("#live-txt"), b = $("#live-bar");
  if (t) t.textContent = `step ${j.step} / ${j.total} · h = ${f1(j.h, 2)} W/m²K`;
  if (b) b.style.width = `${(100 * j.step) / Math.max(1, j.total)}%`;
  const el = $("#live-chart");
  if (el && hist.length > 1) hChart(el, hist, hTrue, 200);
}
function hChart(el, hist, hTrue, height = 220, extra = []) {
  PChart.mount(el, {
    height, x: hist.map((p) => p[0]), yLabel: "h (W/m²K)", legend: true,
    series: [{ name: "h learned by the network", y: hist.map((p) => p[1]), color: C.pinn, width: 2.2 }],
    hlines: [...(hTrue ? [{ y: hTrue, color: C.truth, label: [`true h = ${f1(hTrue, 0)}`, ...extra.map((e) => e.label)].join(" · ") }] : []),
      ...extra.map((e) => ({ ...e, label: hTrue ? "" : e.label }))],
  });
}

// ------------------------------------------------------------------ 1D fin
let finMode = "demo";
const DEF_TC = [[10, 65.1], [20, 54.9], [30, 47.7], [40, 43.3], [50, 42.9]];
function tcRow(x = "", t = "") {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td><input type="number" step="any" class="tc-x" value="${x}"></td><td class="mine-only"><input type="number" step="any" class="tc-t" value="${t}"></td><td><button type="button" title="remove">×</button></td>`;
  tr.querySelector("button").onclick = () => tr.remove();
  $("#f-tc").appendChild(tr);
}
DEF_TC.forEach(([x, t]) => tcRow(x, t));
$("#f-add").onclick = () => { const xs = $$(".tc-x").map((i) => +i.value || 0); tcRow(Math.min(+$("#f-l").value, (Math.max(0, ...xs) || 0) + 10), ""); };
function setMode(m) {
  finMode = m;
  $$("#fin-mode button").forEach((b) => b.classList.toggle("on", b.dataset.m === m));
  const f = $("#fin-form");
  f.classList.toggle("hide-demo", m !== "demo"); f.classList.toggle("hide-mine", m !== "mine");
}
$$("#fin-mode button").forEach((b) => (b.onclick = () => setMode(b.dataset.m)));
setMode("demo");
$("#f-mat").onchange = () => {
  const v = $("#f-mat").value;
  $("#f-k-row").hidden = v !== "custom";
  if (v !== "custom") $("#f-k").value = v;
};

$("#f-run").onclick = async () => {
  const st = $("#f-status"), btn = $("#f-run");
  const rows = $$("#f-tc tr").map((tr) => [tr.querySelector(".tc-x").value, tr.querySelector(".tc-t").value]);
  const body = {
    k: +$("#f-k").value, d_mm: +$("#f-d").value, length_mm: +$("#f-l").value, t_base: +$("#f-tb").value,
    t_air: +$("#f-ta").value, h_guess: +$("#f-hg").value, sensors_mm: rows.map((r) => +r[0]),
  };
  if (finMode === "mine") {
    if (rows.some((r) => r[1] === "")) { st.innerHTML = `<span class="err">Fill in every reading (or remove the row).</span>`; return; }
    body.readings = rows.map((r) => +r[1]);
  } else { body.h_true = +$("#f-htrue").value; body.noise = +$("#f-noise").value; }
  st.textContent = ""; btn.disabled = true;
  try {
    const { id } = await api("/api/fin/run", body);
    $("#fin-out").innerHTML = `<h3 style="margin-top:0">Training…</h3>` + liveBlock("starting");
    const { job, hist } = await poll(id, (j, h) => tickLive(j, h, body.h_true));
    if (job.status === "error") throw new Error(job.error);
    renderFin(job.result, hist, job.seconds, body);
    refreshCode("fin", job.result.readings, "with the readings of this run");
  } catch (e) {
    st.innerHTML = `<span class="err">${esc(e.message)}</span>`;
  } finally { btn.disabled = false; }
};

function profileSvg(r) {
  const W = 480, H = 250, L = 46, R = 14, T = 12, B = 34;
  const xmax = r.x_mm[r.x_mm.length - 1];
  const all = [...r.T_pinn, ...r.readings, ...(r.T_true || []), ...r.T_least_squares];
  let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  const X = (x) => L + (x / xmax) * (W - L - R), Y = (t) => T + (1 - (t - lo) / (hi - lo)) * (H - T - B);
  const path = (ys) => ys.map((t, i) => `${i ? "L" : "M"}${X(r.x_mm[i]).toFixed(1)},${Y(t).toFixed(1)}`).join("");
  let g = "";
  const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 4))); const sk = [1, 2, 5, 10].map((m) => m * step).find((s) => (hi - lo) / s <= 6);
  for (let v = Math.ceil(lo / sk) * sk; v <= hi; v += sk) g += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#eef1f5"/><text x="${L - 6}" y="${Y(v) + 3.5}" text-anchor="end">${+v.toFixed(2)}</text>`;
  for (let k = 0; k <= 5; k++) { const x = (k * xmax) / 5; g += `<text x="${X(x)}" y="${H - B + 15}" text-anchor="middle">${+x.toFixed(1)}</text>`; }
  g += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">distance from the wall (mm)</text>`;
  g += `<text x="12" y="${(T + H - B) / 2}" transform="rotate(-90 12 ${(T + H - B) / 2})" text-anchor="middle" font-weight="600">T (°C)</text>`;
  g += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="#c9d1dd"/><line x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" stroke="#c9d1dd"/>`;
  if (r.T_true) g += `<path d="${path(r.T_true)}" fill="none" stroke="${C.truth}" stroke-width="5" stroke-opacity=".25"/>`;
  g += `<path d="${path(r.T_least_squares)}" fill="none" stroke="${C.lsq}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
  g += `<path d="${path(r.T_pinn)}" fill="none" stroke="${C.pinn}" stroke-width="2.2"/>`;
  r.sensors_mm.forEach((x, i) => { g += `<circle cx="${X(x)}" cy="${Y(r.readings[i])}" r="4.5" fill="#fff" stroke="${C.read}" stroke-width="1.8"><title>${x} mm: ${r.readings[i]} °C</title></circle>`; });
  const leg = [[C.pinn, "network (PINN)", ""], [C.lsq, "least squares on the analytic profile", "5 4"], ...(r.T_true ? [[C.truth + "66", "true profile (hidden)", ""]] : [])];
  return `<div class="profile"><svg viewBox="0 0 ${W} ${H}">${g}</svg></div>
    <div class="pchart-legend">${leg.map(([c, n]) => `<span><i style="background:${c}"></i>${n}</span>`).join("")}<span>○ thermocouple readings</span></div>`;
}

function renderFin(r, hist, secs, req) {
  const hasTrue = r.h_true !== undefined;
  const err = hasTrue ? ` · ${f1((100 * (r.h_pinn - r.h_true)) / r.h_true, 1)} % vs true` : "";
  const biotBad = r.biot_cross_section > 0.1;
  $("#fin-out").innerHTML = `
    <h3 style="margin-top:0">Result <span class="meta">${secs} s · ${hist.length ? hist[hist.length - 1][0] : 0} steps</span></h3>
    <div class="kpis">
      <div class="kpi hero"><div class="l">h, learned by the network</div><div class="v">${f1(r.h_pinn, 1)} <small>W/m²K</small></div><div class="s">started at ${f1(req.h_guess, 0)}${err}</div></div>
      <div class="kpi"><div class="l">h, least squares (analytic)</div><div class="v">${f1(r.h_least_squares, 1)}</div><div class="s">the classical cross-check</div></div>
      ${hasTrue ? `<div class="kpi"><div class="l">true h (hidden)</div><div class="v">${f1(r.h_true, 1)}</div><div class="s">demo mode</div></div>` : ""}
      <div class="kpi"><div class="l">heat removed by the fin</div><div class="v">${f1(r.q_pinn_W, 3)} <small>W</small></div><div class="s">least squares ${f1(r.q_least_squares_W, 3)}${hasTrue ? ` · true ${f1(r.q_true_W, 3)}` : ""}</div></div>
      <div class="kpi"><div class="l">Biot number hD/2k</div><div class="v">${r.biot_cross_section.toExponential(1)}</div><div class="s">${biotBad ? "above 0.1: 1D model doubtful" : "below 0.1: 1D model holds"}</div></div>
    </div>
    ${Math.abs(r.h_pinn - r.h_least_squares) > 0.1 * r.h_least_squares ? `<div class="banner warn">The network and the least-squares fit disagree by
      ${f1((100 * Math.abs(r.h_pinn - r.h_least_squares)) / r.h_least_squares, 0)} %: the readings do not fit a fin with these properties well.
      Check the conductivity, the positions, the wall temperature and the thermocouple contact; a reading off the curve below is the usual suspect.</div>` : ""}
    ${biotBad ? `<div class="banner warn">The fin is thick for its conductivity (Biot ${f1(r.biot_cross_section, 2)}): temperature varies across it and the 1D model is an approximation.</div>` : ""}
    <div class="grid2" style="margin-top:14px">
      <div class="fig"><div class="cap">Temperature along the fin</div>${profileSvg(r)}</div>
      <div class="fig"><div class="cap">h during training</div><div id="fin-h"></div></div>
    </div>
    <p class="hint">The network got the fin equation, the wall temperature, the convective tip and your ${r.sensors_mm.length} readings, nothing else.
      The least-squares line fits the textbook solution to the same readings: two independent routes, the same answer when the physics holds.</p>
    ${window.renderScope && META ? renderScope(META.scope) : ""}`;
  hChart($("#fin-h"), hist.length ? hist : r.history, r.h_true, 240, [{ y: r.h_least_squares, color: C.lsq, label: `least squares ${f1(r.h_least_squares, 1)}`, dash: "2 3" }]);
}

// ------------------------------------------------------------------ heat maps
const INFERNO = [[0, [0, 0, 4]], [0.13, [31, 12, 72]], [0.25, [85, 15, 109]], [0.38, [136, 34, 106]], [0.5, [186, 54, 85]],
  [0.63, [227, 89, 51]], [0.75, [249, 140, 10]], [0.88, [249, 201, 50]], [1, [252, 255, 164]]];
const DIVERGE = [[0, [33, 102, 172]], [0.5, [247, 247, 247]], [1, [178, 24, 43]]];
function cmap(stops, t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], f = (t - a) / (b - a);
    return ca.map((c, k) => Math.round(c + f * (cb[k] - c)));
  }
  return stops[stops.length - 1][1];
}
window.IHL_CMAP = (t) => cmap(INFERNO, t);
function cbarCss(stops) { return `linear-gradient(to top, ${stops.map(([t, c]) => `rgb(${c}) ${t * 100}%`).join(", ")})`; }

// field: rows (y) of columns (x), y increasing upward on screen
function heatmap(el, { field, x, y, lo, hi, diverging, unit = "°C", sensors = [], readings = [], marks = [], device, scale = 6 }) {
  const ny = field.length, nx = field[0].length, stops = diverging ? DIVERGE : INFERNO;
  const cv = document.createElement("canvas"); cv.className = "map"; cv.width = nx * scale; cv.height = ny * scale;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(nx, ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const c = cmap(stops, (field[j][i] - lo) / (hi - lo)), p = ((ny - 1 - j) * nx + i) * 4;
    img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255;
  }
  const off = document.createElement("canvas"); off.width = nx; off.height = ny; off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, cv.width, cv.height);
  const xmax = x[nx - 1] + (x[1] - x[0]) / 2, ymax = y[ny - 1] + (y[1] - y[0]) / 2;
  const PX = (v) => (v / xmax) * cv.width, PY = (v) => cv.height - (v / ymax) * cv.height;
  if (device) {
    const [xc, yc, h] = device; ctx.setLineDash([6, 4]); ctx.strokeStyle = "#ffffffcc"; ctx.lineWidth = 2;
    ctx.strokeRect(PX(xc - h), PY(yc + h), PX(2 * h), PY(yc - h) - PY(yc + h)); ctx.setLineDash([]);
  }
  ctx.font = `${Math.round(scale * 2.1)}px system-ui`; ctx.textAlign = "center";
  sensors.forEach(([sx, sy], k) => {
    ctx.beginPath(); ctx.arc(PX(sx), PY(sy), scale * 1.1, 0, 7); ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#172033"; ctx.stroke();
    if (readings[k] !== undefined) { ctx.fillStyle = "#fff"; ctx.strokeStyle = "#000a"; ctx.lineWidth = 3; const s = `${f1(readings[k], 1)}`; ctx.strokeText(s, PX(sx), PY(sy) - scale * 1.8); ctx.fillText(s, PX(sx), PY(sy) - scale * 1.8); }
  });
  marks.forEach(([mx, my, label]) => {
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.5; const a = PX(mx), b = PY(my), s = scale * 1.4;
    ctx.beginPath(); ctx.moveTo(a - s, b); ctx.lineTo(a + s, b); ctx.moveTo(a, b - s); ctx.lineTo(a, b + s); ctx.stroke();
    if (label) { ctx.textAlign = "left"; ctx.fillStyle = "#fff"; ctx.strokeStyle = "#000a"; ctx.lineWidth = 3; ctx.strokeText(label, a + s + 4, b + 4); ctx.fillText(label, a + s + 4, b + 4); ctx.textAlign = "center"; }
  });
  el.innerHTML = `<div class="mapwrap"><div style="position:relative;flex:1"></div><div class="cbar" style="background:${cbarCss(stops)}"></div>
    <div class="cbar-l"><span>${f1(hi, diverging ? 2 : 1)} ${unit}</span><span>${f1((lo + hi) / 2, diverging ? 2 : 1)}</span><span>${f1(lo, diverging ? 2 : 1)} ${unit}</span></div></div>`;
  const host = el.querySelector(".mapwrap > div"); host.appendChild(cv);
  const probe = document.createElement("div"); probe.className = "probe"; host.appendChild(probe);
  cv.addEventListener("mousemove", (e) => {
    const r = cv.getBoundingClientRect(), u = (e.clientX - r.left) / r.width, v = 1 - (e.clientY - r.top) / r.height;
    const i = Math.min(nx - 1, Math.max(0, Math.floor(u * nx))), j = Math.min(ny - 1, Math.max(0, Math.floor(v * ny)));
    probe.textContent = `x ${f1(x[i], 1)} mm, y ${f1(y[j], 1)} mm: ${f1(field[j][i], 2)} ${unit}`;
    probe.style.display = "block"; probe.style.left = Math.min(r.width - 170, e.clientX - r.left + 12) + "px"; probe.style.top = e.clientY - r.top + 12 + "px";
  });
  cv.addEventListener("mouseleave", () => (probe.style.display = "none"));
}
function argmax2(field) { let b = -Infinity, at = [0, 0]; field.forEach((row, j) => row.forEach((v, i) => { if (v > b) { b = v; at = [i, j]; } })); return at; }
function minmax(...fields) { let lo = Infinity, hi = -Infinity; for (const f of fields) for (const row of f) for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; }

// ------------------------------------------------------------------ 2D plate
let PLATE_PRE = null;
async function loadPlate() {
  try { PLATE_PRE = await api("/api/plate/precomputed"); showPlatePre(); }
  catch (e) { $("#plate-out").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
function showPlatePre() {
  const p = PLATE_PRE, s = p.summary, st = s.setup;
  $("#p-pre-note").hidden = false; $("#p-reset").hidden = true;
  renderPlate({
    title: "Full run: 5 noise draws, 3000 steps each", field_pinn: p.field_pinn, field_reference: p.field_reference, x_mm: p.x_mm, y_mm: p.y_mm,
    sensors_mm: st.sensors_mm, readings: p.run.readings_C, device_mm: st.device.slice(0, 3).map((v) => v * 1000), history: p.run.history,
    h_true: st.h_true_W_m2K, h_guess: st.h_initial_guess,
    kpis: [
      ["h, learned by the network", `${f1(s.h_pinn.mean, 1)} ± ${f1(s.h_pinn.std, 1)}`, `true ${f1(st.h_true_W_m2K, 0)} · started at ${f1(st.h_initial_guess, 0)}`, true],
      ["hot spot (never measured)", `${f1(s.hot_spot_pinn_C.mean, 1)} °C`, `finite volumes ${f1(s.reference.hot_spot_C, 1)} °C`],
      ["worst error on the map", `${f1(s.max_field_error_C, 2)} °C`, "over the whole plate, worst draw"],
      ["if you trusted the guess", `${f1(s.hot_spot_if_h_guess_C, 1)} °C`, `hot spot with h = ${f1(st.h_initial_guess, 0)}: ${f1(s.reference.hot_spot_C - s.hot_spot_if_h_guess_C, 0)} °C optimistic`],
    ],
    note: `Classical alternative: re-run the finite-volume model until it matches the readings, h = ${f1(s.h_fv_fit.mean, 2)} ± ${f1(s.h_fv_fit.std, 2)} after ${s.h_fv_fit.solves_per_fit} solves per fit. It needs a simulation model; the network needs only the equation. Reference grid-converged to ${f1(s.reference.grid_convergence_K, 3)} K.`,
    figures: ["2d/plate_result_card.png", "2d/plate_h_convergence.png"],
  });
}
let plateView = "pinn";
function renderPlate(d) {
  const diff = d.field_pinn.map((row, j) => row.map((v, i) => v - d.field_reference[j][i]));
  const [lo, hi] = minmax(d.field_pinn, d.field_reference);
  const dm = Math.max(...diff.flat().map(Math.abs)) || 1;
  const [hi_i, hi_j] = argmax2(d.field_pinn), [ri, rj] = argmax2(d.field_reference);
  $("#plate-out").innerHTML = `
    <h3 style="margin-top:0">${esc(d.title)}</h3>
    <div class="kpis">${d.kpis.map(([l, v, s, hero]) => `<div class="kpi${hero ? " hero" : ""}"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("")}</div>
    <div class="toolbar" style="margin:14px 0 8px"><div class="seg" id="p-view" style="margin:0">
      <button data-v="pinn">Network (PINN)</button><button data-v="ref">Finite volumes (check)</button><button data-v="diff">Difference</button></div>
      <span class="meta">○ thermocouples with their readings (°C) · + hot spot · dashed: device</span></div>
    <div class="heat" id="p-map"></div>
    <div class="grid2" style="margin-top:14px">
      <div class="fig"><div class="cap">h during training</div><div id="p-h"></div></div>
      <div><p class="hint" style="margin-top:0">${esc(d.note)}</p></div>
    </div>
    ${d.figures ? `<div class="figs">${d.figures.map((f) => `<img loading="lazy" src="/figures/${f}" alt="">`).join("")}</div>` : ""}`;
  const draw = () => {
    $$("#p-view button").forEach((b) => b.classList.toggle("on", b.dataset.v === plateView));
    const common = { x: d.x_mm, y: d.y_mm, sensors: d.sensors_mm, device: d.device_mm };
    if (plateView === "diff") heatmap($("#p-map"), { ...common, field: diff, lo: -dm, hi: dm, diverging: true, unit: "K" });
    else {
      const ref = plateView === "ref", F = ref ? d.field_reference : d.field_pinn, [i, j] = ref ? [ri, rj] : [hi_i, hi_j];
      heatmap($("#p-map"), { ...common, field: F, lo, hi, readings: d.readings, marks: [[d.x_mm[i], d.y_mm[j], `${f1(F[j][i], 1)} °C`]] });
    }
  };
  $$("#p-view button").forEach((b) => (b.onclick = () => { plateView = b.dataset.v; draw(); }));
  draw();
  hChart($("#p-h"), d.history, d.h_true, 220);
}
$("#p-reset").onclick = showPlatePre;
$("#p-run").onclick = async () => {
  const st = $("#p-status"), btn = $("#p-run");
  const body = { power: +$("#p-power").value, h_true: +$("#p-htrue").value, noise: +$("#p-noise").value, h_guess: +$("#p-hg").value };
  st.textContent = ""; btn.disabled = true;
  try {
    const { id } = await api("/api/plate/run", body);
    $("#p-pre-note").hidden = true;
    $("#plate-out").innerHTML = `<h3 style="margin-top:0">Training on your plate…</h3>` + liveBlock("starting");
    const { job, hist } = await poll(id, (j, h) => tickLive(j, h, body.h_true));
    if (job.status === "error") throw new Error(job.error);
    const r = job.result;
    renderPlate({
      title: `Your run: ${f1(body.power, 1)} W, true h ${f1(body.h_true, 0)}, ${r.steps} steps (${job.seconds} s)`, ...r, history: hist,
      kpis: [
        ["h, learned by the network", f1(r.h_pinn, 1), `true ${f1(r.h_true, 1)} · ${f1((100 * (r.h_pinn - r.h_true)) / r.h_true, 1)} %`, true],
        ["hot spot (never measured)", `${f1(r.hot_spot_pinn, 1)} °C`, `finite volumes ${f1(r.hot_spot_reference, 1)} °C`],
        ["worst error on the map", `${f1(r.max_field_error, 2)} °C`, "over the whole plate"],
        ["if you trusted the guess", `${f1(r.hot_spot_if_guess, 1)} °C`, `hot spot with h = ${f1(body.h_guess, 0)}`],
      ],
      note: "A shorter training than the full run (1500 steps instead of 3000): the temperature map converges first, h keeps creeping toward its value for longer, so expect h within a few percent here and about 1 % in the full run. The finite-volume map is computed independently with your true h; the network never sees it.",
    });
    $("#p-reset").hidden = false;
    refreshCode("plate", r.readings, "with the readings of this run");
  } catch (e) { st.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; }
};

// ------------------------------------------------------------------ 3D block
async function loadBlock() {
  let b;
  try { b = await api("/api/block/precomputed"); } catch (e) { $("#block-out").innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  const s = b.summary, st = s.setup;
  $("#block-out").innerHTML = `
    <h2 style="margin-top:0">Chip under a cooled block</h2>
    <p class="hint">Steel block 40 × 40 × 10 mm. A 10 W chip (10 × 10 mm) heats the bottom, a fan cools the top with an unknown h, nine
      thermocouples sit just under the top face. The question: how hot is the chip, which nobody can measure? Full run computed offline
      (3 noise draws, about 30 minutes on a CPU); the Code tab has the script.</p>
    <div class="kpis">
      <div class="kpi hero"><div class="l">chip temperature (never measured)</div><div class="v">${f1(s.device_max_pinn_C.mean, 1)} °C</div><div class="s">finite volumes ${f1(s.reference.device_max_C, 1)} °C</div></div>
      <div class="kpi"><div class="l">worst error in the block</div><div class="v">${f1(s.max_field_error_C, 2)} °C</div><div class="s">all ${st.draws} draws, every cell</div></div>
      <div class="kpi"><div class="l">h from the energy balance</div><div class="v">${f1(s.h_energy_balance.mean, 1)}</div><div class="s">true ${f1(st.h_true_W_m2K, 0)} · on the learned top face</div></div>
      <div class="kpi"><div class="l">h parameter of the network</div><div class="v">${f1(s.h_pinn.mean, 1)}</div><div class="s">${f1((100 * (s.h_pinn.mean - st.h_true_W_m2K)) / st.h_true_W_m2K, 0)} %: the field is right, the parameter settles low</div></div>
      <div class="kpi"><div class="l">if you trusted the guess</div><div class="v">${f1(s.device_max_if_h_guess_C, 0)} °C</div><div class="s">chip with h = ${f1(st.h_initial_guess, 0)}</div></div>
    </div>
    <div class="grid2" style="margin-top:14px;grid-template-columns:1.4fr 1fr">
      <div id="b-3d"></div>
      <div>
        <div class="slice-row"><label>Layer <input type="range" id="b-z" min="0" max="${b.shape[0] - 1}" value="0"></label><span id="b-zl"></span>
          <label><select id="b-src"><option value="pinn">Network</option><option value="ref">Finite volumes</option><option value="diff">Difference</option></select></label></div>
        <div class="heat" id="b-slice" style="margin-top:8px"></div>
        <div class="fig" style="margin-top:12px"><div class="cap">h parameter during training</div><div id="b-h"></div></div>
      </div>
    </div>
    <p class="hint">Why two values of h: the readings pin down the temperature field very well, and the energy balance on that field
      (all the chip's power leaves through the top) gives h within 0.5 %. The network's own h parameter is weakly constrained in this
      thick, conductive block and settles 7 % low; the full script reports both, as is.</p>
    <div class="figs">${["3d/block_result_card.png"].map((f) => `<img loading="lazy" src="/figures/${f}" alt="">`).join("")}</div>`;
  const [nz, ny, nx] = b.shape, [lx, ly, lz] = st.L_m.map((v) => v * 1000);
  const P = b.field_pinn, R = b.field_reference;
  const layer = (arr, k) => Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => arr[(k * ny + j) * nx + i]));
  const xs = Array.from({ length: nx }, (_, i) => ((i + 0.5) * lx) / nx), ys = Array.from({ length: ny }, (_, j) => ((j + 0.5) * ly) / ny);
  let lo = Infinity, hi = -Infinity; for (const v of P) { lo = Math.min(lo, v); hi = Math.max(hi, v); } for (const v of R) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  let dm = 0; for (let i = 0; i < P.length; i++) dm = Math.max(dm, Math.abs(P[i] - R[i]));
  const drawSlice = () => {
    const k = +$("#b-z").value, src = $("#b-src").value;
    $("#b-zl").textContent = `z = ${f1(((k + 0.5) * lz) / nz, 2)} mm ${k === 0 ? "(chip side)" : k === nz - 1 ? "(fan side)" : ""}`;
    const F = src === "diff" ? layer(P, k).map((row, j) => row.map((v, i) => v - R[(k * ny + j) * nx + i])) : layer(src === "ref" ? R : P, k);
    heatmap($("#b-slice"), src === "diff" ? { field: F, x: xs, y: ys, lo: -dm, hi: dm, diverging: true, unit: "K", scale: 8 }
      : { field: F, x: xs, y: ys, lo, hi, scale: 8, sensors: k === nz - 1 ? st.sensors_mm : [], readings: k === nz - 1 ? b.run.readings_C : [], device: k === 0 ? st.device.slice(0, 3).map((v) => v * 1000) : null });
  };
  $("#b-z").oninput = drawSlice; $("#b-src").onchange = drawSlice; drawSlice();
  hChart($("#b-h"), b.run.history, st.h_true_W_m2K, 200, [{ y: s.h_energy_balance.mean, color: C.lsq, label: `energy balance ${f1(s.h_energy_balance.mean, 1)}`, dash: "2 3" }]);
  const mount = () => window.IHL_BlockViewer && new window.IHL_BlockViewer($("#b-3d"), { shape: b.shape, size: [lx, ly, lz], pinn: P, ref: R, lo, hi, sensors: st.sensors_mm, device: st.device.map((v) => v * 1000) });
  if (window.IHL_BlockViewer) mount(); else window.addEventListener("ihl-viewer-ready", mount, { once: true });
}

// ------------------------------------------------------------------ code (complete scripts, filled with the inputs)
const py = (v) => (v === null || v === undefined ? "None" : Array.isArray(v) ? `[${v.map(py).join(", ")}]` : typeof v === "number" ? String(+v.toPrecision(6)) : String(v));
function fillScript(code, params) {
  // replaces the value of `NAME = value  # comment` lines in the script's parameter block, keeping the comment
  for (const [k, v] of Object.entries(params)) {
    code = code.replace(new RegExp(`^(${k} = ).*?(\\s{2,}#.*)?$`, "m"), (_, a, c) => {
      const val = a + py(v);
      return c ? val + " ".repeat(Math.max(2, 39 - val.length)) + c.trim() : val;
    });
  }
  return code;
}
const CODE_PARAMS = { fin: null, plate: null, block: {} };
function codeBox(caseId, el, params, note) {
  const sn = META.snippets[caseId], code = fillScript(sn.code, params || {});
  el.innerHTML = `<div class="snippet"><div class="hd"><span class="meta">${esc(sn.file)} · ${esc(sn.runtime)}${note ? " · " + esc(note) : ""}</span>
      <span><button class="copy" data-a="copy">Copy</button> <button class="copy" data-a="dl">Download .py</button></span></div>
    <pre class="code"><code>pip install pinneapple\npython ${esc(sn.file)}</code></pre>
    <pre class="code"><code>${esc(code)}</code></pre></div>`;
  el.querySelector("[data-a=copy]").onclick = async (e) => {
    try { await navigator.clipboard.writeText(code); e.target.textContent = "Copied"; } catch { e.target.textContent = "Select and copy"; }
    setTimeout(() => (e.target.textContent = "Copy"), 1500);
  };
  el.querySelector("[data-a=dl]").onclick = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([code], { type: "text/x-python" })); a.download = sn.file; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
}
function finParams(readings) {
  const rows = $$("#f-tc tr").map((tr) => [tr.querySelector(".tc-x").value, tr.querySelector(".tc-t").value]);
  const mine = finMode === "mine" && rows.every((r) => r[1] !== "");
  return {
    K: +$("#f-k").value, D_MM: +$("#f-d").value, L_MM: +$("#f-l").value, T_WALL: +$("#f-tb").value, T_AIR: +$("#f-ta").value,
    X_MM: rows.map((r) => +r[0]), T_READ: readings || (mine ? rows.map((r) => +r[1]) : null),
    H_TRUE: +$("#f-htrue").value, NOISE: +$("#f-noise").value, H_GUESS: +$("#f-hg").value,
  };
}
function plateParams(readings) {
  return { POWER: +$("#p-power").value, H_TRUE: +$("#p-htrue").value, NOISE: +$("#p-noise").value, H_GUESS: +$("#p-hg").value,
           T_READ: readings || null };
}
function refreshCode(caseId, readings, note) {
  if (!META) return;
  const el = $(`#code-${caseId} .code-host`);
  if (caseId === "fin") codeBox("fin", el, finParams(readings), note);
  else if (caseId === "plate") codeBox("plate", el, plateParams(readings), note);
  else codeBox("block", el, {}, note);
}
let finCodeT = null;
$("#fin-form").addEventListener("input", () => { clearTimeout(finCodeT); finCodeT = setTimeout(() => refreshCode("fin"), 250); });
$("#fin-form").addEventListener("click", () => setTimeout(() => refreshCode("fin"), 50));
$("#tab-plate form").addEventListener("input", () => refreshCode("plate"));

function renderCode() {
  $("#snippets").innerHTML = Object.entries(META.snippets).map(([k, s]) => `
    <div class="snippet"><h3>${esc(s.title)}</h3><div id="all-${k}"></div></div>`).join("");
  for (const k of Object.keys(META.snippets)) codeBox(k, $(`#all-${k}`), {}, "demo values");
  ["fin", "plate", "block"].forEach((c) => refreshCode(c));
  const host = location.origin;
  $("#api-example").textContent = `# start a training with your readings (add -u user:password if the page asks for a login)
curl -s ${host}/api/fin/run -H 'Content-Type: application/json' -d '{
  "k": 16, "d_mm": 5, "length_mm": 50, "t_base": 80, "t_air": 25,
  "sensors_mm": [10, 20, 30, 40, 50], "readings": [65.1, 54.9, 47.7, 43.3, 42.9]
}'
# -> {"id": "3f9c..."}; poll until "status" is "done"
curl -s ${host}/api/jobs/3f9c... | python -m json.tool     # result.h_pinn, result.h_least_squares, result.q_pinn_W
# the complete script of a case: ${host}/api/code/fin  (also plate, block)`;
}

(async function init() {
  try { META = await api("/api/meta"); renderCode(); } catch (e) { console.error(e); }
  const t = location.hash.slice(1);
  openTab(["fin", "plate", "block", "code", "api"].includes(t) ? t : "fin");
})();
