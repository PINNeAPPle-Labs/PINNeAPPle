// Inverse Model Studio page: problem editor (shared), one job per problem, comparison of methods.
const $ = (s) => document.querySelector(s);
const { api, pollJob, xy, fmt, esc } = window.InvUI;
let META = null, ED = null;
const MC = { least_squares: "#0b6bcb", adjoint: "#b45309", autodiff: "#7c3aed", bayesian: "#0f766e", pinn: "#c2410c" };
const CAP = { autodiff: "predict_torch", adjoint: "adjoint_gradient", pinn: "pinn" };
const WHY = { autodiff: "needs a PyTorch model", adjoint: "needs a model with an adjoint solver (finite-volume fin)", pinn: "needs a fin or plate model with only h unknown" };

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
}));

async function init() {
  META = await api("/api/meta");
  $("#ihl").href = META.heat_lab_url;
  ED = InvUI.ProblemEditor($("#editor"), { meta: META, values: "required", validation: true, title: "Measurements (fitted)" });
  $("#preset").innerHTML += META.presets.map((p) => `<option value="${p.id}">${esc(p.title)}</option>`).join("");
  $("#preset").onchange = () => loadPreset($("#preset").value);
  $("#editor").addEventListener("change", () => refreshMethods());
  $("#run").onclick = run;
  $("#api-code").textContent = apiExample();
  const first = new URLSearchParams(location.search).get("preset") || "rod_h_field";
  $("#preset").value = first;
  await loadPreset(first);
}

function refreshMethods(select) {
  const t = ED.state.model.type, caps = (META.models[t] || { capabilities: [] }).capabilities;
  const free = ED.state.parameters.filter((p) => !p.fixed && p.fixed !== "true").map((p) => p.name);
  const ok = (m) => !CAP[m] || (caps.includes(CAP[m]) && (m !== "pinn" || (free.length === 1 && free[0] === "h")));
  const prev = select || new Set([...document.querySelectorAll("#methods input:checked")].map((x) => x.value));
  $("#methods").innerHTML = Object.entries(META.methods).map(([m, l]) => `<label class="${ok(m) ? "" : "off"}"><input type="checkbox" value="${m}" ${ok(m) ? "" : "disabled"} ${ok(m) && prev.has(m) ? "checked" : ""}>
    <span><span style="color:${MC[m]}">●</span> ${esc(l)}${ok(m) ? "" : `<span class="why">not available: ${esc(WHY[m] || "")}</span>`}</span></label>`).join("");
  const pinnOn = () => { $("#pinn-row").hidden = !$('#methods input[value="pinn"]')?.checked; };
  $("#methods").querySelectorAll("input").forEach((x) => x.addEventListener("change", pinnOn)); pinnOn();
}

async function loadPreset(id) {
  if (!id) return;
  const p = await api(`/api/presets/${id}`);
  ED.set(p);
  const meta = META.presets.find((x) => x.id === id);
  $("#preset-note").innerHTML = `${esc(p.summary)}<br><i>${esc(meta?.lesson || "")}</i>`;
  $("#o-prior").value = p.prior_std || 1;
  $("#o-smooth").value = p.smoothness || 0;
  refreshMethods(new Set(p.methods.filter((m) => m !== "pinn")));
}

async function run() {
  let body;
  try { body = ED.get(); } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  if (!body.observations) { $("#status").innerHTML = `<span class="err">add measurements</span>`; return; }
  body.methods = [...document.querySelectorAll("#methods input:checked")].map((x) => x.value);
  if (!body.methods.length) { $("#status").innerHTML = `<span class="err">pick at least one method</span>`; return; }
  body.options = { prior_std: +$("#o-prior").value, smoothness: +$("#o-smooth").value, pinn_steps: +$("#o-pinn").value };
  body.name = $("#preset").selectedOptions[0]?.textContent || "inverse problem";
  $("#run").disabled = true;
  $("#status").innerHTML = `<div>queued…</div><div class="progress"><div></div></div>`;
  try {
    const job = await api("/api/solve", { json: body });
    const done = await pollJob(job.id, (j) => {
      $("#status").innerHTML = `<div>${esc(j.status)} ${esc(j.message || "")}</div><div class="progress"><div style="width:${Math.round(100 * (j.progress || 0))}%"></div></div>`;
    });
    if (done.status !== "done") throw new Error(done.error || done.status);
    $("#status").innerHTML = `done in ${fmt(done.result.seconds, 2)} s`;
    render(done.result, done.id, body);
  } catch (e) {
    $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`;
  } finally { $("#run").disabled = false; }
}

function render(r, jid, body) {
  const done = r.solutions.filter((s) => s.status === "done"), best = done.find((s) => s.method === r.best);
  const names = done.length ? done[0].parameters.map((p) => p.name) : [];
  const truth = done.some((s) => s.truth_recovered !== undefined);
  const rec = done.filter((s) => s.truth_recovered === true).length, failed = done.filter((s) => s.truth_recovered === false).length;
  const id = r.identifiability, nOk = id ? id.parameters.filter((p) => p.status === "identifiable").length : names.length;
  const cls = nOk < names.length ? "warn" : failed ? "warn" : "ok";
  let h = `<div class="verdict ${cls}"><b>${done.length} method${done.length > 1 ? "s" : ""} solved the same problem; best on held-out data: ${esc(best ? best.label : "—")}.</b>
    ${truth ? ` Known answer recovered by ${rec} of ${done.length}${failed ? ` (${failed} missed it)` : ""}.` : ""} ${nOk < names.length ? `${names.length - nOk} unknown${names.length - nOk > 1 ? "s" : ""} not identifiable from these measurements.` : ""}</div>`;
  if (id) h += `<div class="ident"><b>Identifiability:</b> ${esc(id.summary)}</div>`;
  h += `<h3 class="sec">Estimates (95 % interval) by method</h3><table class="t"><thead><tr><th>Method</th>${names.map((n) => `<th>${esc(n)}</th>`).join("")}<th>Fit RMSE</th><th>Held-out RMSE</th><th>Model runs</th><th>Time</th>${truth ? "<th>Truth recovered</th>" : ""}</tr></thead><tbody>`;
  if (truth && done.length) h += `<tr><td class="hint">Truth (example)</td>${done[0].parameters.map((p) => `<td class="hint">${fmt(p.truth)}</td>`).join("")}<td colspan="${truth ? 5 : 4}"></td></tr>`;
  h += r.solutions.map((s) => s.status !== "done" ? `<tr><td>${esc(s.label)}</td><td colspan="${names.length + 5}" class="hint"><span class="st na">${esc(s.status)}</span> ${esc(s.reason || "")}</td></tr>` :
    `<tr class="${s.method === r.best ? "best" : ""}"><td><span style="color:${MC[s.method]}">●</span> <b>${esc(s.label)}</b>${s.method === r.best ? ' <span class="st ok">best held-out</span>' : ""}${s.converged === false ? ' <span class="st warn">not converged</span>' : ""}<div class="hint">${esc(s.derivatives)}</div></td>
      ${s.parameters.map((p) => `<td><b>${fmt(p.estimate)}</b><div class="hint">${p.interval95 ? `${fmt(p.interval95[0])} … ${fmt(p.interval95[1])}` : "no interval"}</div>${p.note ? `<div class="hint" style="color:var(--warn)">${esc(p.note)}</div>` : ""}</td>`).join("")}
      <td>${fmt(s.fit.rmse, 3)}</td><td>${s.validation ? fmt(s.validation.rmse, 3) : "—"}</td><td>${s.method === "pinn" ? "network training" : s.forward_evaluations}</td><td>${fmt(s.seconds, 2)} s</td>
      ${truth ? `<td>${s.truth_recovered === null || s.truth_recovered === undefined ? "—" : s.truth_recovered ? '<span class="st ok">yes</span>' : '<span class="st bad">no</span>'}</td>` : ""}</tr>`).join("") + "</tbody></table>";
  if (truth) h += `<div class="hint">${esc(done[0].truth_check || "")}. Model runs: forward solves of the model (for the Bayesian method, most are the MCMC chain).</div>`;
  const notes = done.flatMap((s) => (s.notes || []).map((n) => `${s.label}: ${n}`));
  if (notes.length) h += `<ul class="list-plain hint">${notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>`;
  const f0 = best && best.fields;
  if (f0) h += `<h3 class="sec">Reconstructed fields</h3><div class="grid2" id="fields"></div>`;
  h += `<div class="grid2"><div><h3 class="sec">Residuals at the measurements</h3><div id="res"></div></div><div><h3 class="sec">Convergence</h3><div id="conv"></div></div></div>`;
  h += `<div class="dl"><a href="/api/jobs/${jid}/download/report.json">Report (JSON)</a><a href="/api/jobs/${jid}/download/study.json">StudyResult (contract)</a>
    <a href="/api/jobs/${jid}/download/estimates.csv">Estimates (CSV)</a>${f0 && Object.values(f0).some((f) => !Array.isArray(f.values[0])) ? `<a href="/api/jobs/${jid}/download/fields.csv">Fields (CSV)</a>` : ""}</div>`;
  h += window.renderScope ? renderScope(META.scope) : "";
  $("#out").innerHTML = h;
  if (f0) renderFields(r, done, best);
  const obs = body.observations, key = Object.keys(obs.inputs)[0], xs = obs.inputs[key];
  const oneD = xs.every((x) => x !== null && isFinite(x)) && new Set(xs).size === xs.length && Object.keys(obs.inputs).length === 1;
  xy($("#res"), { height: 230, xLabel: oneD ? key : "measurement #", yLabel: "model − measured",
    series: done.map((s) => ({ name: s.label.split(" (")[0], mode: "points", x: oneD ? xs : xs.map((_, i) => i + 1), y: s.residuals, color: MC[s.method], fill: MC[s.method], r: 3.2 })),
    hlines: [{ y: 0, color: "#98a2b3" }] });
  const hist = done.filter((s) => s.method !== "pinn" && s.history && s.history.length > 1);
  const pinn = done.find((s) => s.method === "pinn");
  if (hist.length) xy($("#conv"), { height: 230, xLabel: "iteration", yLabel: "χ² (+ prior)", yLog: true,
    series: hist.map((s) => ({ name: s.label.split(" (")[0], x: s.history.map((_, i) => i + 1), y: s.history, color: MC[s.method] })) });
  if (pinn && pinn.history && pinn.history.length) {
    const d = document.createElement("div"); $("#conv").appendChild(d);
    xy(d, { height: 180, xLabel: "PINN training step", yLabel: "h learned by the PINN", series: [{ name: "PINN", x: pinn.history.map((p) => p[0]), y: pinn.history.map((p) => p[1]), color: MC.pinn }],
      hlines: pinn.parameters[0].truth ? [{ y: pinn.parameters[0].truth, label: "truth", color: "#1a7f4b" }] : [] });
  }
}

function renderFields(r, done, best) {
  const host = $("#fields"), names = Object.keys(best.fields), T = r.truth_fields || {};
  names.forEach((n) => {
    const f = best.fields[n], d = document.createElement("div"); host.appendChild(d);
    if (Array.isArray(f.values[0])) {                                     // 2D map: best method, PINN and truth
      const maps = [[`${best.label.split(" (")[0]}`, f.values]];
      const pinn = done.find((s) => s.method === "pinn" && s.pinn_field && Array.isArray(s.pinn_field.values[0]));
      if (pinn) maps.push(["PINN (its own field)", pinn.pinn_field.values]);
      if (T[n]) maps.push(["truth (example)", T[n].values]);
      const all = maps.flatMap((m) => m[1].flat()), lo = Math.min(...all), hi = Math.max(...all);
      d.innerHTML = maps.map((m, i) => `<div style="margin-bottom:8px"><canvas class="field" id="fm-${n}-${i}" width="${m[1][0].length}" height="${m[1].length}"></canvas><div class="fcap">${esc(n)}: ${esc(m[0])}, max ${fmt(Math.max(...m[1].flat()), 4)} ${esc(f.unit)}</div></div>`).join("") +
        `<div class="fbar"><span>${fmt(lo, 4)}</span><i></i><span>${fmt(hi, 4)} ${esc(f.unit)}</span></div><div class="fcap">common colour scale, x right, y up</div>`;
      maps.forEach((m, i) => drawField(document.getElementById(`fm-${n}-${i}`), m[1], lo, hi));
    } else {
      const series = done.filter((s) => s.fields && s.fields[n]).map((s) => ({ name: s.label.split(" (")[0], x: s.fields[n].x, y: s.fields[n].values, color: MC[s.method], width: 1.8 }));
      const pinn = done.find((s) => s.method === "pinn" && s.pinn_field && n === "T");
      if (pinn) series.push({ name: "PINN (its own field)", x: pinn.pinn_field.x, y: pinn.pinn_field.values, color: MC.pinn, dash: "2 3" });
      if (T[n]) series.unshift({ name: "truth (example)", x: T[n].x, y: T[n].values, color: "#c9d1dd", width: 7 });
      if (n === "T" && r.problem && r.problem.observations) {
        const o = r.problem.observations, k = Object.keys(o.inputs)[0];
        if (k === (f.axis || "x_mm")) series.push({ name: "measured (±σ)", mode: "points", x: o.inputs[k], y: o.values, err: o.sigma, color: "#172033" });
      }
      xy(d, { height: 240, xLabel: f.axis || "x", yLabel: `${n} (${f.unit})`, series });
      const err = best.field_error && best.field_error[n];
      if (err) d.insertAdjacentHTML("beforeend", `<div class="fcap">${esc(best.label.split(" (")[0])} vs truth: RMSE ${fmt(err.rmse, 3)}, max ${fmt(err.max_abs, 3)} ${esc(err.unit)}</div>`);
    }
  });
}

function drawField(cv, v, lo, hi) {
  const ny = v.length, nx = v[0].length, ctx = cv.getContext("2d"), img = ctx.createImageData(nx, ny);
  const stops = [[0, [49, 54, 149]], [0.25, [69, 117, 180]], [0.5, [254, 224, 144]], [0.75, [244, 109, 67]], [1, [165, 0, 38]]];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const t = Math.max(0, Math.min(1, (v[j][i] - lo) / (hi - lo || 1))), s = Math.max(1, stops.findIndex((x) => x[0] >= t)), a = stops[s - 1], b = stops[s];
    const u = (t - a[0]) / (b[0] - a[0]), p = 4 * ((ny - 1 - j) * nx + i);
    for (let c = 0; c < 3; c++) img.data[p + c] = a[1][c] + u * (b[1][c] - a[1][c]);
    img.data[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function apiExample() {
  return `curl -X POST ${location.origin}/api/solve -H 'Content-Type: application/json' -d '{
  "model": {"type": "rod_fv", "options": {"d_mm": 6, "length_mm": 120, "n_zones": 4}},
  "parameters": [{"name": "h1", "value": 40, "lower": 2, "upper": 1000}, ..., {"name": "k", "value": 200, "fixed": true}, ...],
  "observations": {"inputs": {"x_mm": [...]}, "sigma": [...], "values": [...]},
  "validation":   {"inputs": {"x_mm": [...]}, "sigma": [...], "values": [...]},
  "methods": ["least_squares", "adjoint", "autodiff", "bayesian"], "options": {"smoothness": 0.5}}'
curl ${location.origin}/api/jobs/<job>                        # status, progress, the comparison
curl -O ${location.origin}/api/jobs/<job>/download/study.json # StudyResult (inverse)
# A complete, runnable body: GET ${location.origin}/api/presets/rod_h_field (model_spec, parameters, observations, validation)`;
}

init().catch((e) => { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; });
