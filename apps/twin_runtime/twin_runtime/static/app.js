"use strict";
// Digital Twin Runtime UI: live twin over MQTT, replay of recorded data, the series flow, API examples.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 4) => (v === null || v === undefined || !isFinite(v) ? "—" : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(d) + "");
const ASSET = "tank-1";
let META = null, POLL = null, UPLOAD = null;

$$("#tabs button").forEach((b) => b.addEventListener("click", () => {
  $$("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${b.dataset.tab}`));
}));

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail || j));
  return j;
}
async function waitJob(job, line) {
  for (;;) {
    const j = await api(`/api/jobs/${job.id}`);
    line.textContent = `${j.status} · ${Math.round(100 * (j.progress || 0))} % ${j.message ? "· " + j.message : ""}`;
    if (j.status === "done") return j;
    if (j.status === "failed" || j.status === "cancelled") throw new Error(j.error || j.status);
    await new Promise((r) => setTimeout(r, 400));
  }
}

// ---------------------------------------------------------------- charts
function stateChart(el, h, fc, key, opts) {
  const n = h.t.length, m = fc ? fc.t.length : 0, nul = (k) => Array(k).fill(null);
  const x = [...h.t, ...(fc ? fc.t : [])].map((t) => +(t / 60).toFixed(1));
  const sd = h[`${key}_sd`] || nul(n), est = h[key];
  const series = [];
  if (h[`obs_${key}`]) series.push({ name: "measured", y: [...h[`obs_${key}`], ...nul(m)], color: "#94a3b8", width: 1 });
  series.push({ name: "estimate ± 2σ", y: [...est, ...nul(m)], lo: [...est.map((v, i) => v - 2 * sd[i]), ...nul(m)],
    hi: [...est.map((v, i) => v + 2 * sd[i]), ...nul(m)], color: "#0f766e", fillOpacity: 0.18 });
  if (fc) series.push({ name: "forecast 5-95 %", y: [...nul(n), ...fc.mean[key]], lo: [...nul(n), ...fc.p05[key]], hi: [...nul(n), ...fc.p95[key]], color: "#7c3aed", fillOpacity: 0.15, dash: "4 3" });
  if (opts.truth) series.push({ name: "true value (simulation)", y: [...opts.truth.map((v) => v * (opts.scale || 1)), ...nul(m)], color: "#b42318", width: 1.2, dash: "2 2" });
  if (opts.scale) for (const s of series) for (const k of ["y", "lo", "hi"]) if (s[k] && s.name !== "true value (simulation)") s[k] = s[k].map((v) => (v === null ? null : v * opts.scale));
  PChart.mount(el, { x, series, height: opts.height || 200, yLabel: opts.label, hlines: opts.hlines || [] });
}
function limitLines(limits, key) {
  const l = (limits || {})[key];
  if (!l) return [];
  return l.filter((v) => v !== null).map((v) => ({ y: v, color: "#b42318", label: `limit ${v}` }));
}
function alertsHtml(alerts) {
  if (!alerts.length) return `<p class="hint">No alert.</p>`;
  return `<div class="alerts">${alerts.slice().reverse().map((a) => `<div class="a${a.cleared ? " cleared" : ""}">
    <span>${fmt(a.t / 60, 4)} min</span><span><span class="pill ${a.cleared ? "ok" : a.severity}">${esc(a.cleared ? "cleared" : a.kind)}</span></span>
    <span>${esc(a.message)}</span></div>`).join("")}</div>`;
}
function validityHtml(d, reval) {
  let s = "";
  if (d) s += `<div><b>Last validity check</b> (t = ${fmt(d.t / 60)} min, ${d.n} one-step predictions): <span class="pill ${d.decision === "ok" ? "ok" : d.decision === "review" ? "warning" : "critical"}">${esc(d.decision)}</span>
    ${d.reasons.length ? `<span class="hint">${esc(d.reasons.join("; "))}</span>` : ""}
    ${d.coverage ? `<div class="hint">coverage of the 95 % predictive band: ${Object.entries(d.coverage).map(([k, v]) => `${k} ${(100 * v).toFixed(0)} %`).join(", ")}${d.out_of_bounds !== null && d.out_of_bounds !== undefined ? ` · inputs outside the model's training box: ${(100 * d.out_of_bounds).toFixed(0)} %` : ""}</div>` : ""}</div>`;
  else s += `<div class="hint">No validity check yet (every 30 readings, on the last 90).</div>`;
  if (reval) s += `<div style="margin-top:6px"><b>Revalidation</b> (${esc(reval.method)}, ${reval.n_samples} samples): <span class="pill ${reval.passed ? "ok" : "critical"}">${reval.passed ? "passed" : "failed"}</span>
    <table style="margin-top:4px"><tr><th>state</th><th class="num">RMSE</th><th class="num">limit</th><th></th></tr>${reval.checks.map((c) => `<tr><td>${esc(c.state)}</td><td class="num">${fmt(c.rmse)}</td><td class="num">${fmt(c.limit)}</td><td>${c.passed ? "✓" : "✗"}</td></tr>`).join("")}</table></div>`;
  return s;
}
function kpis(st, sd, status) {
  const k = (l, v, s) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s || ""}</div></div>`;
  let h = `<div class="kpis">` + k("Status", `<span class="status-pill ${esc(status)}">${esc(status.replace("_", " "))}</span>`) +
    k("Level h", `${fmt(st.h)} m`, `± ${fmt(2 * sd.h, 2)} m (2σ)`) + k("Temperature T", `${fmt(st.T)} °C`, `± ${fmt(2 * sd.T, 2)} °C (2σ)`);
  if (st.q_leak !== undefined) h += k("Unaccounted outflow", `${fmt(1000 * st.q_leak, 3)} L/s`, `± ${fmt(2000 * sd.q_leak, 2)} L/s`);
  if (st.eta !== undefined) h += k("Heater efficiency", `${fmt(100 * st.eta, 3)} %`, `± ${fmt(200 * sd.eta, 2)} %`);
  return h + `</div>`;
}
function twinView(el, d, opts = {}) {
  const h = d.history, st = (d.snapshot || d).state, sd = (d.snapshot || d).state_sd, status = (d.snapshot || d).status;
  el.innerHTML = `<div class="panel">${kpis(st, sd, status)}
    <div class="charts2">
      <div class="chartbox"><div class="cap">Level h [m]: measured, estimated, forecast</div><div id="ch-h"></div></div>
      <div class="chartbox"><div class="cap">Temperature T [°C]</div><div id="ch-T"></div></div>
      ${h.q_leak ? `<div class="chartbox"><div class="cap">Estimated unaccounted outflow [L/s]</div><div id="ch-q"></div></div>` : ""}
      <div class="chartbox"><div class="cap">Filter consistency: normalised innovation (NIS)</div><div id="ch-nis"></div></div>
    </div>
    <div class="grid2" style="margin-top:14px">
      <div><h3>Alerts</h3>${alertsHtml(d.alerts || [])}</div>
      <div><h3>Model validity</h3>${validityHtml((d.snapshot || d).last_validity_check || d.validity, opts.reval || (d.snapshot || d).last_revalidation)}</div>
    </div>${opts.extra || ""}<div class="meta" style="margin-top:8px">time axis in minutes · ${esc(opts.note || "")}</div></div>`;
  const fc = d.forecast;
  stateChart($("#ch-h", el), h, fc, "h", { label: "h [m]", hlines: limitLines(d.limits, "h") });
  stateChart($("#ch-T", el), h, fc, "T", { label: "T [°C]", hlines: limitLines(d.limits, "T") });
  if (h.q_leak) stateChart($("#ch-q", el), h, null, "q_leak", { label: "L/s", scale: 1000, truth: opts.truthLeak, hlines: [{ y: 0.3, color: "#9a6200", label: "alert 0.3 L/s" }] });
  PChart.mount($("#ch-nis", el), { x: h.t.map((t) => +(t / 60).toFixed(1)), series: [{ name: "NIS", y: h.nis, color: "#334155", width: 1 }], height: 160, yLabel: "NIS", hlines: [{ y: d.nis_threshold || 13.8, color: "#b42318", label: "99.9 % χ²" }] });
}

// ---------------------------------------------------------------- live
function liveButtons(running) {
  $("#start").disabled = running; $("#stop").disabled = !running; $("#reval").disabled = !running;
  $$(".faults button").forEach((b) => (b.disabled = !running || $("#source").value !== "simulated"));
}
$("#start").addEventListener("click", async () => {
  try {
    $("#lstatus").textContent = "starting…";
    const r = await api("/api/live/start", { method: "POST", body: JSON.stringify({ asset: ASSET, model: $("#model").value, source: $("#source").value, speed: +$("#speed").value }) });
    $("#lstatus").textContent = `running: ${r.model} · topic ${r.topic}${r.broker_port ? ` · embedded broker port ${r.broker_port}` : ""}`;
    liveButtons(true); poll();
  } catch (e) { $("#lstatus").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
});
$("#stop").addEventListener("click", async () => {
  clearTimeout(POLL);
  try { const r = await api(`/api/live/${ASSET}/stop`, { method: "POST" }); $("#lstatus").textContent = `stopped after ${r.n_updates} updates`; } catch (e) { $("#lstatus").textContent = e.message; }
  liveButtons(false);
});
$$(".faults button[data-fault]").forEach((b) => b.addEventListener("click", async () => {
  try { const f = await api(`/api/live/${ASSET}/fault`, { method: "POST", body: b.dataset.fault }); $("#lstatus").textContent = `fault ${f.kind} injected at t = ${fmt(f.start / 60)} min`; } catch (e) { $("#lstatus").textContent = e.message; }
}));
$("#clear").addEventListener("click", async () => { try { await api(`/api/live/${ASSET}/clear`, { method: "POST" }); $("#lstatus").textContent = "faults cleared"; } catch (e) { $("#lstatus").textContent = e.message; } });
$("#reval").addEventListener("click", async () => {
  try { const r = await api(`/api/live/${ASSET}/revalidate`, { method: "POST" }); $("#lstatus").textContent = `revalidation ${r.passed ? "passed" : "failed"}`; } catch (e) { $("#lstatus").textContent = e.message; }
});
async function poll() {
  clearTimeout(POLL);
  try {
    const d = await api(`/api/assets/${ASSET}/live?n=360&forecast_steps=60`);
    if (d.history && d.history.t && d.history.t.length > 1) {
      twinView($("#live-out"), d, { note: `${d.n_updates} readings assimilated · ${d.published || 0} published on MQTT · assimilation ${fmt(d.assimilation_ms, 2)} ms/reading${d.faults && d.faults.length ? ` · active faults: ${d.faults.map((f) => f.kind).join(", ")}` : ""}` });
    }
    if (d.running) POLL = setTimeout(poll, 1000);
  } catch (e) { liveButtons(false); }
}
$("#source").addEventListener("change", () => liveButtons($("#start").disabled));

// ---------------------------------------------------------------- recorded data
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
$("#file").addEventListener("change", (e) => upload(e.target.files[0]));
async function upload(f) {
  if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  $("#fileinfo").textContent = `uploading ${f.name}…`;
  const r = await fetch("/api/uploads", { method: "POST", body: fd });
  const j = await r.json();
  if (!r.ok) { $("#fileinfo").innerHTML = `<span class="err">${esc(j.detail)}</span>`; return; }
  UPLOAD = j; $("#fileinfo").textContent = `${j.name || f.name} · ${(f.size / 1024).toFixed(0)} kB`; $("#rrun").disabled = false;
}
async function runReplay(body) {
  const line = $("#rstatus");
  try {
    $("#rrun").disabled = $("#rexample").disabled = true;
    const job = await api("/api/replay", { method: "POST", body: JSON.stringify({ model: $("#rmodel").value, ...body }) });
    const j = await waitJob(job, line);
    const res = j.result;
    line.textContent = `done: ${res.n} readings, model ${res.model}`;
    const checks = res.study.checks.map((c) => `<tr><td>${esc(c.name)}</td><td><span class="badge ${c.status === "PASS" ? "pass" : c.status === "WARN" ? "warn" : c.status === "FAIL" ? "fail" : "na"}">${esc(c.status)}</span></td><td class="num">${esc(typeof c.value === "number" ? fmt(c.value) : c.value ?? "")}</td><td class="hint">${esc(c.evidence)}</td></tr>`).join("");
    const extra = `<h3>Checks recorded in the StudyResult</h3><table><tr><th>check</th><th>status</th><th class="num">value</th><th>evidence</th></tr>${checks}</table>
      <div class="toolbar" style="margin-top:10px"><span class="meta">Downloads</span><span>
      <a class="link" href="/api/jobs/${j.id}/files/twin.study.json">StudyResult (JSON)</a> ·
      <a class="link" href="#" id="dl-csv">estimates (CSV)</a></span></div>`;
    twinView($("#record-out"), res, { reval: res.revalidation, extra, truthLeak: res.truth && res.truth.true_q_leak, note: "the example's true leak (red, dashed) is known because the record is simulated" });
    $("#dl-csv").addEventListener("click", (e) => { e.preventDefault(); const h = res.history, keys = Object.keys(h).filter((k) => Array.isArray(h[k])); const rows = [keys.join(",")]; for (let i = 0; i < h.t.length; i++) rows.push(keys.map((k) => h[k][i] ?? "").join(",")); const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv" })); a.download = "twin_estimates.csv"; a.click(); });
  } catch (e) { line.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  $("#rrun").disabled = !UPLOAD; $("#rexample").disabled = false;
}
$("#rrun").addEventListener("click", () => runReplay({ upload_id: UPLOAD.id }));
$("#rexample").addEventListener("click", () => runReplay({ example: true }));

// ---------------------------------------------------------------- series flow
$("#srun").addEventListener("click", async () => {
  const line = $("#sstatus");
  $("#srun").disabled = true;
  try {
    const job = await api("/api/series", { method: "POST", body: "{}" });
    const j = await waitJob(job, line);
    const r = j.result;
    line.textContent = "done";
    $("#series-out").innerHTML = `<div class="steps">${r.steps.map((s, i) => `${i ? `<div class="arrow">↓ ${esc(r.steps[i - 1].contract)}</div>` : ""}
      <div class="step"><div class="day">${s.day}</div><div><b>${esc(s.app)}</b> <span class="ran">· ${esc(s.ran_as)}</span>
      <div style="margin:4px 0">${esc(s.summary)}</div>
      <div class="files">${s.files.map((f) => `<a class="link" href="/api/jobs/${j.id}/files/${encodeURIComponent(f)}">${esc(f)}</a>`).join("")}</div></div></div>`).join("")}</div>
      <p class="hint" style="margin-top:12px">Every number above was computed in this run: held-out errors on runs the surrogate never saw, coverage on a
      further unseen run, the optimum checked against the closed-form steady state, and the twin's revalidation when the inlet water leaves the training range.</p>`;
  } catch (e) { line.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  $("#srun").disabled = false;
});

// ---------------------------------------------------------------- API tab and meta
const origin = location.origin;
$("#c1").textContent = `# start a twin fed by REST pushes, push readings, read its state
curl -X POST ${origin}/api/live/start -H 'Content-Type: application/json' -d '{"asset": "pump-7", "source": "rest"}'
curl -X POST ${origin}/api/assets/pump-7/readings -H 'Content-Type: application/json' \\
     -d '{"t": 0, "h": 1.2, "T": 25.0, "q_in": 0.0066, "Q": 277000, "T_in": 15, "T_amb": 20}'
curl ${origin}/api/assets/pump-7/state            # estimate, spread, status, active alerts, validity
curl ${origin}/api/assets/pump-7/history?table=alerts
curl ${origin}/api/assets/pump-7/study -o study.json   # StudyResult contract`;
$("#c2").textContent = `mosquitto_pub -h <server> -p 1883 -t pinneapple/tank-1/sensors \\
  -m '{"timestamp": 120, "h": 1.21, "T": 25.1, "q_in": 0.0066, "Q": 277000, "T_in": 15, "T_amb": 20}'`;
$("#c3").textContent = `from pinneapple_systems.operations.live import make_twin, replay
from pinneapple_systems.operations.twin import LumpedTankModel
from pinneapple_data.contracts import read_table

df, _ = read_table("tank_day_with_leak.csv")
twin, result = replay(df, LumpedTankModel(dt=10.0))
print(twin.snapshot()["state"], [a["message"] for a in result["alerts"]])`;
api("/api/meta").then((m) => { META = m; $("#scope").innerHTML = renderScope(m.scope); }).catch(() => {});
api("/api/assets").then((a) => { if (a.some((x) => x.asset === ASSET && x.running)) { liveButtons(true); poll(); } else liveButtons(false); }).catch(() => liveButtons(false));
