// Model Calibration Lab page: problem editor (shared), calibration job, report.
const $ = (s) => document.querySelector(s);
const { api, pollJob, xy, heat, fmt, esc } = window.InvUI;
let META = null, ED = null;
const C = { meas: "#0e7490", hold: "#c2410c", before: "#98a2b3", after: "#0e7490", bad: "#b42318" };
const CAP = { autodiff: "predict_torch", adjoint: "adjoint_gradient", pinn: "pinn" };

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
}));

async function init() {
  META = await api("/api/meta");
  ED = InvUI.ProblemEditor($("#editor"), { meta: META, values: "required", validation: true, title: "Measurements (fitted)" });
  $("#preset").innerHTML += META.presets.map((p) => `<option value="${p.id}">${esc(p.title)}</option>`).join("");
  $("#preset").onchange = () => loadPreset($("#preset").value);
  $("#h-mode").onchange = holdUI;
  $("#editor").addEventListener("change", refreshMethods);
  $("#run").onclick = run;
  $("#api-code").textContent = apiExample();
  const first = new URLSearchParams(location.search).get("preset") || "cooling_h";
  $("#preset").value = first;
  await loadPreset(first);
}

function holdUI() {
  const m = $("#h-mode").value;
  $("#h-opts").hidden = !(m === "last" || m === "random");
  $("#h-by").parentElement.hidden = m !== "last";
  $("#h-by").innerHTML = ED.state.inputs.map((k) => `<option>${esc(k)}</option>`).join("");
}

function refreshMethods() {
  const t = ED.state.model.type, caps = (META.models[t] || { capabilities: [] }).capabilities;
  const free = ED.state.parameters.filter((p) => !p.fixed).map((p) => p.name);
  const ok = (m) => !CAP[m] || (caps.includes(CAP[m]) && (m !== "pinn" || (free.length === 1 && free[0] === "h")));
  const prev = new Set([...document.querySelectorAll("#o-cmp input:checked")].map((x) => x.value));
  $("#o-cmp").innerHTML = Object.entries(META.methods).filter(([m]) => m !== "least_squares").map(([m, l]) =>
    `<label class="${ok(m) ? "" : "off"}" title="${ok(m) ? esc(l) : "not available for this model"}"><input type="checkbox" value="${m}" ${ok(m) ? "" : "disabled"} ${ok(m) && prev.has(m) ? "checked" : ""}> ${esc(l.split(" (")[0])}</label>`).join("");
  $("#o-cmp").querySelectorAll("input").forEach((x) => x.addEventListener("change", () => { $("#pinn-row").hidden = !$('#o-cmp input[value="pinn"]')?.checked; }));
  $("#pinn-row").hidden = !$('#o-cmp input[value="pinn"]')?.checked;
}

async function loadPreset(id) {
  if (!id) return;
  const p = await api(`/api/presets/${id}`);
  ED.set(p);
  const meta = META.presets.find((x) => x.id === id);
  $("#preset-note").innerHTML = `${esc(p.summary)}<br><i>${esc(meta?.lesson || "")}</i>`;
  $("#o-prior").value = p.prior_std || 1;
  $("#o-boot").value = (META.models[p.model_spec.type] || {}).cost === "moderate" ? 30 : 100;
  $("#h-mode").value = p.validation ? "validation" : "last";
  holdUI(); refreshMethods();
  document.querySelectorAll("#o-cmp input:not(:disabled)").forEach((x) => { x.checked = x.value === "bayesian"; });
}

async function run() {
  let body;
  try { body = ED.get(); } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  if (!body.observations) { $("#status").innerHTML = `<span class="err">add measurements</span>`; return; }
  const mode = $("#h-mode").value;
  if (mode !== "validation") delete body.validation;
  if (mode === "validation" && !body.validation) { $("#status").innerHTML = `<span class="err">no held-out measurements: add some or choose another validation</span>`; return; }
  body.holdout = { mode, fraction: +$("#h-frac").value, by: $("#h-by").value };
  body.options = { loss: $("#o-loss").value, bootstrap: +$("#o-boot").value, absolute_sigma: $("#o-abs").checked,
    compare: [...document.querySelectorAll("#o-cmp input:checked")].map((x) => x.value), pinn_steps: +$("#o-pinn").value,
    prior_std: +$("#o-prior").value };
  $("#run").disabled = true;
  $("#status").innerHTML = `<div>queued…</div><div class="progress"><div></div></div>`;
  try {
    const job = await api("/api/calibrate", { json: body });
    const done = await pollJob(job.id, (j) => {
      $("#status").innerHTML = `<div>${esc(j.status)} ${esc(j.message || "")}</div><div class="progress"><div style="width:${Math.round(100 * (j.progress || 0))}%"></div></div>`;
    });
    if (done.status !== "done") throw new Error(done.error || done.status);
    $("#status").innerHTML = `done in ${fmt(done.result.seconds, 2)} s`;
    render(done.result, done.id);
  } catch (e) {
    $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`;
  } finally { $("#run").disabled = false; }
}

function ci(v) { return v ? `${fmt(v[0])} … ${fmt(v[1])}` : "—"; }

function render(r, jid) {
  const P = r.parameters, mf = r.metrics.fit, mh = r.metrics.holdout;
  const ne = P.filter((p) => !p.ci_jacobian);
  const holdBad = r.warnings.some((w) => w.startsWith("held-out RMSE"));
  const cls = ne.length === P.length ? "bad" : ne.length || holdBad || !mh ? "warn" : "ok";
  let h = `<div class="verdict ${cls}"><b>${P.length - ne.length} of ${P.length} parameters calibrated with an interval${ne.length ? `; not estimable: ${ne.map((p) => esc(p.name)).join(", ")}` : ""}.</b>
    Fit RMSE ${fmt(mf.before.rmse, 3)} → <b>${fmt(mf.after.rmse, 3)}</b>${mh ? `; held-out RMSE ${fmt(mh.before.rmse, 3)} → <b>${fmt(mh.after.rmse, 3)}</b> on ${mh.after.n} measurements never fitted` : "; not validated (no held-out measurements)"}.</div>`;
  h += `<div class="kpis">
    <div class="kpi"><div class="l">Fit RMSE before → after</div><div class="v">${fmt(mf.before.rmse, 3)} → ${fmt(mf.after.rmse, 3)}</div><div class="s">${mf.after.n} measurements</div></div>
    <div class="kpi"><div class="l">Held-out RMSE before → after</div><div class="v">${mh ? `${fmt(mh.before.rmse, 3)} → ${fmt(mh.after.rmse, 3)}` : "—"}</div><div class="s">${mh ? `${mh.after.n} held out (${esc(r.holdout_mode)})` : "no hold-out"}</div></div>
    <div class="kpi"><div class="l">Reduced χ²</div><div class="v">${fmt(r.fit.reduced_chi2, 3)}</div><div class="s">about 1 when the noise is right</div></div>
    <div class="kpi"><div class="l">Model runs</div><div class="v">${r.fit.nfev}</div><div class="s">${esc(r.fit.derivatives)}</div></div></div>`;
  const truth = P.some((p) => p.truth !== null && p.truth !== undefined);
  h += `<h3 class="sec">Calibrated parameters</h3><table class="t"><thead><tr><th>Parameter</th><th>Start</th><th>Calibrated</th><th>95 % (Jacobian)</th><th>95 % (bootstrap)</th>${truth ? "<th>Truth (example)</th>" : ""}<th>Identifiability</th></tr></thead><tbody>` +
    P.map((p) => `<tr><td><b>${esc(p.name)}</b> ${esc(p.unit)}</td><td>${fmt(p.initial)}</td><td><b>${fmt(p.calibrated)}</b>${p.at_bound ? ' <span class="st warn">on bound</span>' : ""}</td>
      ${p.ci_jacobian ? `<td>${ci(p.ci_jacobian)}</td><td>${ci(p.ci_bootstrap)}</td>` : `<td colspan="2" class="ne"><b>not estimable</b>: ${esc(p.not_estimable || "")}</td>`}
      ${truth ? `<td>${fmt(p.truth)}</td>` : ""}<td><span class="st ${p.identifiability === "identifiable" ? "ok" : "bad"}">${esc(p.identifiability)}</span></td></tr>`).join("") + `</tbody></table>`;
  if (ne.length) h += `<p class="hint">Find which measurement would make them estimable in the <a class="link" href="${esc(META.identifiability_url)}" target="_blank" rel="noopener">Parameter Identifiability Lab</a> (same core, D-optimal design).</p>`;
  if (r.warnings.length) h += `<h3 class="sec">Warnings</h3><ul class="warnings list-plain">${r.warnings.map((w) => `<li>⚠ ${esc(w)}</li>`).join("")}</ul>`;
  h += `<h3 class="sec">Before / after</h3><div id="ba"></div><div class="grid2"><div><h3 class="sec">Residuals after calibration</h3><div id="res"></div></div><div><h3 class="sec">Measured vs model</h3><div id="par"></div></div></div>`;
  const f = r.fields && r.fields.after;
  if (f) h += `<h3 class="sec">Reconstructed field (calibrated)</h3><div id="fld"></div>`;
  if (r.comparison) h += `<h3 class="sec">Same data, other estimators (ranked by ${esc(r.comparison.ranked_by)})</h3><div id="cmp"></div>`;
  if (P.length > 1 && r.fit.correlation) h += `<div class="grid2"><div><h3 class="sec">Parameter correlations (fit)</h3><div id="corr"></div></div><div></div></div>`;
  h += `<div class="dl"><a href="/api/jobs/${jid}/download/report.json">Report (JSON)</a><a href="/api/jobs/${jid}/download/study.json">StudyResult (contract)</a>
    ${r.study.models && r.study.models.length ? `<a href="/api/jobs/${jid}/download/model_ref.json">ModelRef of the calibrated model</a>` : ""}
    <a href="/api/jobs/${jid}/download/parameters.csv">Parameters (CSV)</a><a href="/api/jobs/${jid}/download/residuals.csv">Residuals (CSV)</a></div>`;
  h += `<div class="hint" style="margin-top:6px">Loss ${esc(r.fit.loss)} · bootstrap ${r.bootstrap.n_done} refits · ${fmt(r.seconds, 2)} s on the server.</div>`;
  h += window.renderScope ? renderScope(META.scope) : "";
  $("#out").innerHTML = h;

  const rows = r.rows, n = rows.set.length, key = Object.keys(rows.inputs)[0];
  const isFit = rows.set.map((s) => s === "fit");
  const main = r.curve ? r.curve.observable : rows.observable[0];
  const sel = (fit) => rows.set.map((_, i) => i).filter((i) => isFit[i] === fit && rows.observable[i] === main);
  if (r.curve) {
    const pts = (idx, name, col) => ({ name, mode: "points", x: idx.map((i) => rows.inputs[key][i]), y: idx.map((i) => rows.observed[i]), err: idx.map((i) => rows.sigma[i]), color: col });
    xy($("#ba"), { height: 280, xLabel: r.curve.input, yLabel: main, series: [
      { name: "model, start values", x: r.curve.x, y: r.curve.before, color: C.before, dash: "5 4" },
      { name: "model, calibrated", x: r.curve.x, y: r.curve.after, color: C.after, width: 2.2 },
      pts(sel(true), "fitted measurements (±σ)", C.meas), pts(sel(false), "held-out measurements", C.hold)] });
  } else {
    const idx = rows.set.map((_, i) => i);
    xy($("#ba"), { height: 260, xLabel: "measurement #", yLabel: "value", series: [
      { name: "measured (±σ)", mode: "points", x: idx.map((i) => i + 1), y: rows.observed, err: rows.sigma, color: C.meas },
      { name: "start values", mode: "points", x: idx.map((i) => i + 1), y: rows.before, color: C.before, fill: C.before, r: 3 },
      { name: "calibrated", mode: "points", x: idx.map((i) => i + 1), y: rows.after, color: C.after, fill: C.after, r: 3 }] });
  }
  const resid = rows.observed.map((v, i) => v - rows.after[i]);
  const xr = key && r.curve ? rows.inputs[key] : rows.set.map((_, i) => i + 1);
  const sig = Math.max(...rows.sigma);
  xy($("#res"), { height: 220, xLabel: key && r.curve ? key : "measurement #", yLabel: "measured − model",
    series: [{ name: "fitted", mode: "points", x: xr.filter((_, i) => isFit[i]), y: resid.filter((_, i) => isFit[i]), err: rows.sigma.filter((_, i) => isFit[i]), color: C.meas },
      { name: "held out", mode: "points", x: xr.filter((_, i) => !isFit[i]), y: resid.filter((_, i) => !isFit[i]), err: rows.sigma.filter((_, i) => !isFit[i]), color: C.hold }],
    hlines: [{ y: 0, color: "#98a2b3" }, { y: 2 * sig, label: "+2σ", color: "#c9d1dd" }, { y: -2 * sig, label: "−2σ", color: "#c9d1dd" }] });
  const lo = Math.min(...rows.observed, ...rows.after), hi = Math.max(...rows.observed, ...rows.after);
  xy($("#par"), { height: 220, xLabel: "measured", yLabel: "model", series: [
    { name: "y = x", x: [lo, hi], y: [lo, hi], color: "#c9d1dd" },
    { name: "start values", mode: "points", x: rows.observed, y: rows.before, color: C.before, fill: C.before, r: 3 },
    { name: "calibrated", mode: "points", x: rows.observed, y: rows.after, color: C.after, fill: C.after, r: 3.4 }] });
  if (f) renderField($("#fld"), r.fields);
  if (r.comparison) renderComparison($("#cmp"), r.comparison);
  if (P.length > 1 && r.fit.correlation) heat($("#corr"), r.fit.correlation, r.fit.names);
}

function renderField(el, F) {
  const names = Object.keys(F.after);
  let h = "";
  names.forEach((n, k) => {
    const a = F.after[n], b = F.before[n];
    if (Array.isArray(a.values[0])) h += `<div><canvas class="field" id="fc-${k}" width="${a.x.length}" height="${a.y.length}"></canvas><div class="fbar"><span>${fmt(Math.min(...a.values.flat()), 4)}</span><i></i><span>${fmt(Math.max(...a.values.flat()), 4)} ${esc(a.unit)}</span></div><div class="fcap">${esc(n)} (${esc(a.unit)}) over the plate (x right, y up), calibrated: min ${fmt(Math.min(...a.values.flat()))}, max ${fmt(Math.max(...a.values.flat()))} (start values: max ${fmt(Math.max(...b.values.flat()))})</div></div>`;
    else h += `<div id="fl-${k}"></div>`;
  });
  el.innerHTML = `<div class="grid2">${h}</div>`;
  names.forEach((n, k) => {
    const a = F.after[n], b = F.before[n];
    if (Array.isArray(a.values[0])) drawField(document.getElementById(`fc-${k}`), a.values);
    else xy(document.getElementById(`fl-${k}`), { height: 220, xLabel: a.axis, yLabel: `${n} (${a.unit})`, series: [
      { name: "start values", x: b.x, y: b.values, color: C.before, dash: "5 4" }, { name: "calibrated", x: a.x, y: a.values, color: C.after, width: 2.2 }] });
  });
}

function drawField(cv, v) {
  const ny = v.length, nx = v[0].length, flat = v.flat(), lo = Math.min(...flat), hi = Math.max(...flat);
  const ctx = cv.getContext("2d"), img = ctx.createImageData(nx, ny);
  const stops = [[0, [49, 54, 149]], [0.25, [69, 117, 180]], [0.5, [254, 224, 144]], [0.75, [244, 109, 67]], [1, [165, 0, 38]]];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const t = (v[j][i] - lo) / (hi - lo || 1), s = stops.findIndex((x) => x[0] >= t), a = stops[Math.max(0, s - 1)], b = stops[Math.max(0, s)];
    const u = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]), p = 4 * ((ny - 1 - j) * nx + i);
    for (let c = 0; c < 3; c++) img.data[p + c] = a[1][c] + u * (b[1][c] - a[1][c]);
    img.data[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function renderComparison(el, c) {
  const done = c.solutions.filter((s) => s.status === "done"), names = done.length ? done[0].parameters.map((p) => p.name) : [];
  el.innerHTML = `<table class="t"><thead><tr><th>Method</th>${names.map((n) => `<th>${esc(n)} (95 %)</th>`).join("")}<th>Fit RMSE</th><th>Held-out RMSE</th><th>Model runs</th><th>Time</th>${done.some((s) => s.truth_recovered !== undefined) ? "<th>Truth recovered</th>" : ""}</tr></thead><tbody>` +
    c.solutions.map((s) => s.status !== "done" ? `<tr><td>${esc(s.label)}</td><td colspan="${names.length + 5}" class="hint">${esc(s.status)}: ${esc(s.reason || "")}</td></tr>` :
      `<tr class="${s.method === c.best ? "best" : ""}"><td><b>${esc(s.label)}</b>${s.method === c.best ? ' <span class="st ok">best held-out</span>' : ""}${s.converged === false ? ' <span class="st warn">not converged</span>' : ""}</td>
      ${s.parameters.map((p) => `<td>${fmt(p.estimate)}<div class="hint">${p.interval95 ? `${fmt(p.interval95[0])} … ${fmt(p.interval95[1])}` : esc(p.note ? "no interval" : "—")}</div></td>`).join("")}
      <td>${fmt(s.fit.rmse, 3)}</td><td>${s.validation ? fmt(s.validation.rmse, 3) : "—"}</td><td>${s.method === "pinn" ? "network training" : s.forward_evaluations}</td><td>${fmt(s.seconds, 2)} s</td>
      ${s.truth_recovered !== undefined ? `<td>${s.truth_recovered === null ? "—" : s.truth_recovered ? '<span class="st ok">yes</span>' : '<span class="st bad">no</span>'}</td>` : ""}</tr>`).join("") +
    `</tbody></table><div class="hint">Every method saw the same fitted measurements; held-out RMSE is on the measurements none of them saw. ${c.solutions.some((s) => s.method === "pinn") ? "A single PINN run gives no interval; its h is evaluated in the reference model." : ""}</div>`;
}

function apiExample() {
  return `curl -X POST ${location.origin}/api/calibrate -H 'Content-Type: application/json' -d '{
  "model": {"type": "lumped_cooling", "options": {"area_m2": 0.012}},
  "parameters": [{"name": "h", "value": 30, "lower": 0.5, "upper": 500, "unit": "W/m²K"},
                 {"name": "C", "value": 450, "fixed": true},
                 {"name": "T0", "value": 80, "lower": -50, "upper": 1000, "log": false},
                 {"name": "T_air", "value": 30, "lower": -50, "upper": 500, "log": false}],
  "observations": {"inputs": {"t_s": [0, 300, 600, 900, 1200, 1500, 1800]}, "sigma": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                   "values": [<your readings>]},
  "holdout": {"mode": "last", "fraction": 0.25, "by": "t_s"},
  "options": {"bootstrap": 100, "compare": ["bayesian"]}}'
curl ${location.origin}/api/jobs/<job>                            # status, progress, result
curl -O ${location.origin}/api/jobs/<job>/download/study.json     # StudyResult (calibration)
curl -O ${location.origin}/api/jobs/<job>/download/model_ref.json # ModelRef with held-out metrics`;
}

init().catch((e) => { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; });
