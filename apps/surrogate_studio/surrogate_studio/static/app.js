"use strict";
// Surrogate Studio UI: pick data, run a study (a job), read the comparison and limitations, query the hosted model.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 4) => (v === null || v === undefined || !isFinite(v)) ? "—" : (Math.abs(v) !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) ? v.toExponential(2) : (+v.toPrecision(d)).toString());
const pct = (v) => (v === null || v === undefined || !isFinite(v)) ? "—" : (100 * v).toPrecision(3) + " %";
const sec = (v) => (v === null || v === undefined || !isFinite(v)) ? "—" : v < 1e-3 ? (v * 1e6).toPrecision(3) + " µs" : v < 1 ? (v * 1e3).toPrecision(3) + " ms" : v.toPrecision(3) + " s";
let META = null, INSPECT = null, RES = null, JOB = null, MODELS = [];

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) { $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n)); $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`)); if (n === "predict") loadModels(); }
async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : j.message || JSON.stringify(j.detail || j));
  return j;
}

// ------------------------------------------------------------------ setup
(async function init() {
  META = await api("/api/meta");
  $("#examples").innerHTML = META.examples.map((e) => `<button type="button" data-ex="${e.name}"><b>${esc(e.title)}</b><span>${esc(e.description)}</span></button>`).join("");
  $$("#examples button").forEach((b) => b.onclick = () => runStudy({ example: b.dataset.ex }, b));
  $("#pde").innerHTML += Object.entries(META.pde_types).map(([k, v]) => `<option value="${k}">${k} (coords ${v.coords.join(", ")}${v.coefficients.length ? "; " + v.coefficients.join(", ") : ""})</option>`).join("");
  $("#budget").max = META.limits.max_budget_s; $("#seeds").max = META.limits.max_seeds;
  apiSnippets();
  listJobs();
})();

const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && inspect(e.dataTransfer.files[0]));
$("#file").addEventListener("change", (e) => { if (e.target.files[0]) inspect(e.target.files[0]); e.target.value = ""; });

async function inspect(file) {
  $("#upstatus").textContent = `Reading ${file.name}…`;
  const fd = new FormData(); fd.append("file", file, file.name);
  try {
    INSPECT = await api("/api/inspect", { method: "POST", body: fd });
  } catch (e) { $("#upstatus").innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  const I = INSPECT, sug = I.suggested;
  $("#upstatus").textContent = `${I.filename}: ${I.n_rows} rows, ${I.numeric.length} numeric columns` + (Object.keys(I.fields).length ? `, field${Object.keys(I.fields).length > 1 ? "s" : ""} ${Object.keys(I.fields).join(", ")}` : "") + (I.roles_from_contract ? " · roles from its DatasetSpec" : "");
  $("#colhint").textContent = I.roles_from_contract ? "Roles from the file's DatasetSpec." : "Suggested roles: check them. Inputs are the parameters you set; outputs the quantities the solver computes.";
  const rows = [...I.numeric.map((c) => ({ name: c.name, info: `${fmt(c.min, 3)} .. ${fmt(c.max, 3)}` })),
    ...Object.entries(I.fields).map(([k, v]) => ({ name: k, info: `field, ${v.columns} columns`, field: true }))];
  $("#cols").innerHTML = `<div class="h">column</div><div class="h">input</div><div class="h">output</div>` + rows.map((r) =>
    `<div class="n" title="${esc(r.name)}">${esc(r.name)} <small>${esc(r.info)}</small></div>
     <input type="checkbox" data-in="${esc(r.name)}" ${sug.inputs.includes(r.name) ? "checked" : ""} ${r.field ? "disabled" : ""}>
     <input type="checkbox" data-out="${esc(r.name)}" ${sug.outputs.includes(r.name) ? "checked" : ""}>`).join("");
  $("#group").innerHTML = `<option value="">none</option>` + I.text.map((t) => `<option ${t.name === sug.group ? "selected" : ""}>${esc(t.name)}</option>`).join("");
  $$("#cols input").forEach((x) => x.onchange = () => {          // a column is an input or an output, not both
    const other = x.dataset.in !== undefined ? `#cols input[data-out="${CSS.escape(x.dataset.in)}"]` : `#cols input[data-in="${CSS.escape(x.dataset.out)}"]`;
    if (x.checked && $(other)) $(other).checked = false;
    syncField();
  });
  $("#config").hidden = false;
  syncField();
}
function syncField() {
  const outs = $$("#cols input[data-out]:checked").map((x) => x.dataset.out);
  const f = outs.find((o) => INSPECT.fields[o]);
  $("#fieldbox").hidden = !f;
  if (f && !$("#gridshape").value) $("#gridshape").value = INSPECT.fields[f].columns;
}
$("#pde").onchange = () => {
  const t = META.pde_types[$("#pde").value];
  $("#pdecoef").innerHTML = t && t.coefficients.length ? "Coefficients: " + t.coefficients.map((c) => `${c} = <input data-coef="${c}" style="display:inline;width:110px" placeholder="column or number">`).join(" ") : "";
};
$("#run").onclick = (e) => {
  const inputs = $$("#cols input[data-in]:checked").map((x) => x.dataset.in);
  const outputs = $$("#cols input[data-out]:checked").map((x) => x.dataset.out);
  const body = { upload_id: INSPECT.upload_id, inputs, outputs, group: $("#group").value || null,
    budget_s: +$("#budget").value, seeds: +$("#seeds").value, shift: $("#shift").value,
    model_name: $("#mname").value || null, ood_policy: $("#oodpolicy").value,
    constraints: $("#constraints").value.split("\n").map((s) => s.trim()).filter(Boolean) };
  if (!$("#fieldbox").hidden) {
    body.grid_shape = $("#gridshape").value.split(/[,x ]+/).filter(Boolean).map(Number);
    const names = $("#coordnames").value.split(",").map((s) => s.trim()).filter(Boolean);
    if (names.length) body.coord_names = names;
    const rng = $("#coordranges").value.split(",").map((s) => s.trim()).filter(Boolean).map((s) => s.split(":").map(Number));
    if (rng.length) body.coord_ranges = rng;
  }
  if ($("#pde").value) {
    const t = META.pde_types[$("#pde").value];
    const names = body.coord_names || [];
    body.pde = { type: $("#pde").value, coords: Object.fromEntries(t.coords.map((c, i) => [c, names[i] || c])),
      coefficients: Object.fromEntries($$("#pdecoef input").map((x) => [x.dataset.coef, isNaN(+x.value) || x.value === "" ? x.value : +x.value])) };
  }
  runStudy(body, e.currentTarget);
};

// ------------------------------------------------------------------ jobs
async function runStudy(body, btn) {
  const old = btn.innerHTML; btn.disabled = true;
  $("#status").innerHTML = `<div class="progress"><div style="width:2%"></div></div><div id="jobmsg">Queued…</div><div class="logbox" id="joblog"></div>`;
  try {
    const job = await api("/api/studies", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    await follow(job.id);
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; listJobs(); }
}
async function follow(id) {
  let since = 0;
  for (;;) {
    const j = await api(`/api/jobs/${id}`);
    const logs = await api(`/api/jobs/${id}/logs?since=${since}`);
    for (const l of logs) { since = Math.max(since, l.seq || since + 1); const box = $("#joblog"); if (box) { box.textContent += l.line + "\n"; box.scrollTop = box.scrollHeight; } }
    const bar = $("#status .progress div"); if (bar) bar.style.width = `${Math.round(100 * (j.progress || 0))}%`;
    if ($("#jobmsg")) $("#jobmsg").textContent = `${j.status}${j.message ? ": " + j.message : ""}`;
    if (j.status === "done") { JOB = j; RES = j.result; renderResults(); renderPredict(RES.model.name); showTab("results"); window.scrollTo(0, 0); return; }
    if (j.status === "failed" || j.status === "cancelled") throw new Error(j.error || j.status);
    await new Promise((r) => setTimeout(r, 700));
  }
}
async function listJobs() {
  try {
    const jobs = (await api("/api/jobs?kind=study&limit=8")).filter((j) => j.status === "done");
    $("#jobs").innerHTML = jobs.length ? `<div class="joblist"><b>Your recent studies</b>${jobs.map((j) => `<div><span>${esc(j.id)}</span><button data-j="${j.id}">open</button></div>`).join("")}</div>` : "";
    $$("#jobs button").forEach((b) => b.onclick = async () => { JOB = await api(`/api/jobs/${b.dataset.j}`); RES = JOB.result; renderResults(); renderPredict(RES.model.name); showTab("results"); });
  } catch (e) { /* runtime not reachable: no list */ }
}

// ------------------------------------------------------------------ results
function renderResults() {
  const S = RES.study, M = S.methods, best = M.find((m) => m.id === S.best), sb = best.summary, T = S.trust, L = S.limitations;
  const metric = S.metric === "rel_l2" ? "relative L2" : "nRMSE";
  const ok = M.filter((m) => m.applicable && m.summary && m.summary.n_ok);
  const fail = T.checks.filter((c) => c.status === "contradicts").map((c) => c.name);
  let h = `<div class="verdict ${T.decision}"><div class="big">${T.decision}</div><div class="sub"><b>${esc(best.title)}</b> chosen by validation error out of ${ok.length} methods trained.
     Held-out ${metric} ${pct(sb.test_error_mean)}, distribution shift ${pct(sb.shift_error_mean)}.${fail.length ? " Failed: " + esc(fail.join("; ")) + "." : ""}</div></div>`;
  h += `<div class="kpis">
    <div class="kpi"><div class="l">Held-out ${metric}</div><div class="v">${pct(sb.test_error_mean)}</div><div class="s">${S.splits.n_test} runs never trained on</div></div>
    <div class="kpi"><div class="l">Shift ${metric}</div><div class="v">${pct(sb.shift_error_mean)}</div><div class="s">${esc(S.splits.shift_description || "no shift split")}</div></div>
    <div class="kpi"><div class="l">Inference</div><div class="v">${sec(sb.infer_s_per_query)}</div><div class="s">per query (1 row)${S.dataset.meta && S.dataset.meta.solver_s_per_sample ? `; data ${sec(S.dataset.meta.solver_s_per_sample)} per run` : ""}</div></div>
    <div class="kpi"><div class="l">Training</div><div class="v">${sec(sb.train_s)}</div><div class="s">budget ${fmt(S.budget.wall_s)} s per method and seed</div></div>
    <div class="kpi"><div class="l">Endpoint</div><div class="v" style="font-size:14px;word-break:break-all">${esc(RES.model.name)}</div><div class="s">${esc(RES.model.version)} · ${esc(RES.model.stage)} · OOD ${esc(RES.model.ood_policy)}</div></div>
  </div>`;
  h += `<div class="panel" style="margin-top:14px"><h3 style="margin-top:0">Equal-budget comparison</h3>
    <table><thead><tr><th>method</th><th>family</th><th class="num">validation</th><th class="num">held-out test</th><th class="num">shift</th><th class="num">train</th><th class="num">inference / query</th><th class="num">parameters</th></tr></thead><tbody>
    ${M.map((m) => { const s = m.summary || {}; if (!m.applicable) return `<tr class="na"><td>${esc(m.title)}</td><td>${esc(m.family)}</td><td colspan="6">not run: ${esc(m.reason)}</td></tr>`;
      if (!s.n_ok) return `<tr class="na"><td>${esc(m.title)}</td><td>${esc(m.family)}</td><td colspan="6">failed: ${esc((m.seeds[0] || {}).error)}</td></tr>`;
      return `<tr class="${m.id === S.best ? "best" : ""}"><td>${esc(m.title)}${m.id === S.best ? " ✓" : ""}</td><td>${esc(m.family)}</td><td class="num">${pct(s.val_error_mean)}</td>
        <td class="num">${pct(s.test_error_mean)}${S.seeds.length > 1 ? ` <span class="spread">± ${pct(s.test_error_std)}</span>` : ""}</td><td class="num">${pct(s.shift_error_mean)}</td>
        <td class="num">${sec(s.train_s)}</td><td class="num">${sec(s.infer_s_per_query)}</td><td class="num">${fmt(s.n_params, 3)}</td></tr>`; }).join("")}
    </tbody></table><p class="meta">${esc(metric)} in physical units, mean over ${S.seeds.length} seed${S.seeds.length > 1 ? "s" : ""}. ${S.notes.map(esc).join(" ")}</p></div>`;
  h += `<div class="grid2" style="margin-top:14px"><div class="panel"><h3 style="margin-top:0">Held-out predictions of ${esc(best.title)}</h3><div id="parity"></div></div>
    <div class="panel"><h3 style="margin-top:0">Trust report</h3>${T.checks.map((c) => `<div class="chk"><span class="ic ${c.status}">${c.status === "supports" ? "✓" : c.status === "contradicts" ? "✗" : "–"}</span>
      <div>${esc(c.name)}${c.critical ? ' <span class="pill bad">critical</span>' : ""}<div class="why">${esc(c.reason || c.source || "")}</div></div><div class="val">${c.value === null ? "not run" : fmt(c.value)}${c.limit !== null && c.limit !== undefined ? " / " + fmt(c.limit) : ""}</div></div>`).join("")}
      <p class="meta">${esc(T.rules)}</p></div></div>`;
  h += `<div class="grid2" style="margin-top:14px"><div class="panel"><h3 style="margin-top:0">Where the errors concentrate</h3><p>${esc(L.concentration)}.</p><div id="bins"></div>
      ${L.extrapolation_ratio !== undefined ? `<p class="meta">Shift error / held-out error: <b>${fmt(L.extrapolation_ratio, 3)}</b>.</p>` : ""}
      ${L.pde_residual ? `<p class="meta">PDE residual of the predictions: ${fmt(L.pde_residual.ratio, 3)} × that of the reference data (finite differences).</p>` : ""}</div>
    <div class="panel"><h3 style="margin-top:0">Validity range and OOD gate</h3><table><thead><tr><th>input</th><th class="num">trained from</th><th class="num">to</th></tr></thead><tbody>
      ${Object.entries(L.validity_bounds).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${fmt(v[0])}</td><td class="num">${fmt(v[1])}</td></tr>`).join("")}</tbody></table>
      <p class="meta">${esc(L.ood_gate.rule)}. Policy of this endpoint: <b>${esc(RES.model.ood_policy)}</b>.</p>
      <p class="meta">Held-out runs flagged: ${L.gate_effect.n_flagged} (mean error ${pct(L.gate_effect.mean_error_flagged)}) vs ${L.gate_effect.n_inside} inside (${pct(L.gate_effect.mean_error_inside)}).</p>
      <h3>Worst held-out runs</h3><table><tbody>${L.worst_samples.map((w) => `<tr><td>${Object.entries(w.inputs).map(([k, v]) => `${esc(k)}=${fmt(v, 3)}`).join(", ")}</td><td class="num">${pct(w.error)}</td><td>${w.ood ? '<span class="pill warn">OOD</span>' : ""}</td></tr>`).join("")}</tbody></table></div></div>`;
  if (RES.heldout3d) h += held3dHtml(RES.heldout3d);
  h += `<div class="panel" style="margin-top:14px"><h3 style="margin-top:0">Downloads</h3><div class="files">${RES.files.map((f) => `<a href="/api/studies/${JOB.id}/files/${f}" download>${esc(f)}</a>`).join("")}</div>
    <p class="meta">study_result.json, model_ref.json and dataset_spec.json follow pinneapple_data.contracts, so the next app (uncertainty, sensitivity, optimisation, the Model Gateway) reads them directly.</p>
    ${limitsHtml(S)}</div>`;
  $("#results").innerHTML = h;
  drawParity(); drawBins();
  if (RES.heldout3d) { $$("#v3pick select").forEach((x) => x.onchange = () => view3d()); view3d(); }
}

// ------------------------------------------------------------------ held-out designs on the mesh (3-D)
function held3dHtml(H) {
  const T = H.test, Sh = H.shift, fields = Object.keys(T.fields);
  const zones = Object.keys(T.fields[fields[0]].regions);
  const row = (n) => `<tr><td><b>${esc(n)}</b> <span class="meta">${esc(T.fields[n].unit)}</span></td><td class="num">${pct(T.fields[n].rel_l2)}</td>${zones.map((z) => `<td class="num">${pct(T.fields[n].regions[z])}</td>`).join("")}<td class="num">${Sh.n ? pct(Sh.fields[n].rel_l2) : "—"}</td></tr>`;
  const q = (x, y) => `<tr><td>${esc(x.name)} <span class="meta">${esc(x.unit)}</span></td><td class="num">${pct(x.rel_error_mean)}</td><td class="num">${pct(x.rel_error_max)}</td><td class="num">${y ? pct(y.rel_error_mean) : "—"}</td><td class="num">${y ? pct(y.rel_error_max) : "—"}</td></tr>`;
  const opts = (rp, sp) => rp.samples.map((d, k) => `<option value="${sp}:${k}">${sp} #${k + 1}: ${Object.entries(d.params ? Object.fromEntries(RES.study.dataset.inputs.map((n, i) => [n, d.params[i]])) : {}).map(([n, v]) => `${esc(n)} ${fmt(v, 3)}`).join(", ")} · ${fields.map((n) => `${esc(n)} ${pct(d[n])}`).join(", ")}${d.gate ? ` · gate: ${esc(d.gate.verdict)}` : ""}</option>`).join("");
  const gates = (rp) => rp.gate ? Object.entries(rp.gate).filter(([, c]) => c).map(([v, c]) => `${c} ${esc(v)}`).join(", ") : "—";
  const nonneg = fields.filter((n) => T.fields[n].nonnegative);
  return `<div class="panel" style="margin-top:14px"><h3 style="margin-top:0">Held-out designs on the mesh (3-D)</h3>
    <div class="kpis">
      <div class="kpi"><div class="l">FEM solve</div><div class="v">${sec(H.fem_s)}</div><div class="s">per design, measured here (${H.fem_timed_on} designs solved again)</div></div>
      <div class="kpi"><div class="l">Surrogate</div><div class="v">${sec(H.infer_s)}</div><div class="s">per prediction (median of the study's timing)</div></div>
      <div class="kpi"><div class="l">Speed-up</div><div class="v">${H.speedup ? fmt(H.speedup, 3) + "×" : "—"}</div><div class="s">FEM time / surrogate time, both measured</div></div>
      <div class="kpi"><div class="l">Held-out designs</div><div class="v">${T.n} + ${Sh.n}</div><div class="s">test + shift (never trained on)</div></div></div>
    <div class="grid2" style="margin-top:10px"><div><h4>Relative L2 error, pooled over the held-out designs</h4>
      <table><thead><tr><th>field</th><th class="num">test</th>${zones.map((z) => `<th class="num">${esc(z)}</th>`).join("")}<th class="num">shift</th></tr></thead><tbody>${fields.map(row).join("")}</tbody></table>
      <p class="meta">Zones: the dataset's clamp and loaded-tip node sets grown by 10 % of the body length; the zone columns are test designs.</p>
      ${H.pooling_note ? `<p class="meta"><b>Why the field error is larger than the tip error:</b> ${esc(H.pooling_note)}</p>` : ""}
      ${nonneg.map((n) => `<p class="meta"><b>${esc(n)} ≥ 0:</b> trained on its square root and predicted as its square, so a negative (physically invalid) value cannot occur. Negative values predicted on the held-out designs: <b>${T.fields[n].negative_values + (Sh.n ? Sh.fields[n].negative_values : 0)}</b>.</p>`).join("")}</div>
      <div><h4>Quantities read off the fields</h4><table><thead><tr><th>quantity</th><th class="num">test mean</th><th class="num">test max</th><th class="num">shift mean</th><th class="num">shift max</th></tr></thead>
      <tbody>${T.qois.map((x, i) => q(x, Sh.qois[i])).join("")}</tbody></table><p class="meta">Relative error of the surrogate's value against the FEM's, per design.</p></div></div>
    <p class="meta">OOD gate on the held-out designs (the rule and policy of <code>/v1/models/${esc(RES.model.name)}/predict</code>, policy <b>${esc(H.ood_policy)}</b>): test ${gates(T)}; shift ${gates(Sh)}.</p>
    <div id="v3pick" class="toolbar" style="margin-top:10px"><label>Held-out design <select>${opts(T, "test")}${Sh.n ? opts(Sh, "shift") : ""}</select></label></div>
    <div id="v3gate"></div>
    <div class="v3wrap"><div id="v3d"></div><div class="v3side" id="v3sum"></div></div></div>`;
}
let V3 = null;
const whenViewer = () => new Promise((res) => window.FieldView3D ? res() : window.addEventListener("fieldview3d-ready", res, { once: true }));
async function view3d() {
  const [split, k] = $("#v3pick select").value.split(":");
  $("#v3sum").innerHTML = '<p class="meta">computing the view…</p>';
  let j;
  try { j = await api(`/api/studies/${JOB.id}/view3d?split=${split}&k=${k}`); } catch (e) { $("#v3sum").innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  $("#v3gate").innerHTML = gateHtml(j, split);
  await whenViewer();
  const prev = V3 && V3.field ? V3.field.name : null;
  const start = prev && j.view.fields.some((f) => f.name === prev) ? prev : (j.view.fields.find((f) => f.kind === "error" && /mises/.test(f.name)) || j.view.fields[0]).name;
  V3 = new FieldView3D($("#v3d"), { meta: j.view, glb: j.glb, field: start, height: 460, onField: (f) => sum3d(j, f) });
}
// the hosted endpoint's verdict on this design: what /v1/models/{name}/predict does with it under the published policy
function gateHtml(j, split) {
  const g = j.gate;
  if (!g) return "";
  const what = { "in range": "inside the validity range: the hosted endpoint returns this prediction.",
    warn: "out of distribution: the hosted endpoint returns this prediction flagged as an extrapolation. Do not use it without checking against the FEM.",
    refuse: "out of distribution: the hosted endpoint refuses this design and returns no prediction. The view below is shown only to compare with the FEM." }[g.verdict];
  const cls = g.verdict === "in range" ? "ok" : g.verdict === "warn" ? "warn" : "bad";
  return `<div class="gate ${cls}"><b>OOD gate: ${esc(g.verdict)}</b> (policy ${esc(j.ood_policy || "warn")}) · ${esc(split)} design — ${esc(what)}
    ${g.reasons.length ? `<br><span class="meta">${g.reasons.map(esc).join("; ")}</span>` : `<br><span class="meta">nearest training design at ${fmt(g.knn_distance, 2)} × the typical spacing</span>`}</div>`;
}
function sum3d(j, f) {
  const n = f.name.split(":")[0], e = j.errors[n];
  if (!e) return;
  $("#v3sum").innerHTML = `<h4>Where the error is: ${esc(n)}</h4>${e.summary.map((t) => `<p>${esc(t)}</p>`).join("")}
    <table><thead><tr><th>zone</th><th class="num">share of squared error</th><th class="num">nodes</th><th class="num">RMS / overall</th></tr></thead><tbody>
    ${e.regions.map((g) => `<tr><td>${esc(g.name)}</td><td class="num">${pct(g.share_error)}</td><td class="num">${pct(g.share_points)}</td><td class="num">${g.rms_ratio.toFixed(1)}×</td></tr>`).join("")}</tbody></table>
    ${j.negative && n in j.negative ? `<p class="meta">Negative (invalid) ${esc(n)} values in this prediction: <b>${j.negative[n]}</b>.</p>` : ""}
    <p class="meta">Design ${Object.entries(j.params).map(([k, v]) => `${esc(k)} = ${fmt(v, 3)}`).join(", ")}; this prediction took ${sec(j.predict_s)} (one call, including Python overhead).</p>`;
}
function limitsHtml(S) {
  const items = [
    ["Coverage", "Errors are measured on held-out runs of this campaign; another solver setting, mesh or physics regime is not covered. Keep the OOD policy on refuse for critical use."],
    ["Hyper-parameters", "Each method uses its library defaults (no per-method search); a tuned method may do better. Compare further in the Benchmark Arena."],
    ["OOD gate", "The gate sees the inputs only (training range, nearest-neighbour distance); describe every varied parameter as an input column."],
    ["Physics", "Only the declared PDE residual and constraints are checked; conservation and boundary conditions of the outputs are not (use pinneapple check)."],
    ["Protocol", `Splits fixed before training (seed ${S.splits.seed}); ${S.seeds.length} seed${S.seeds.length > 1 ? "s" : ""} per method; budget ${fmt(S.budget.wall_s)} s; ${S.hardware.cpu || "?"}, ${S.hardware.logical_cpus} logical CPUs; ${S.created}.`]];
  return `<details class="limits"><summary>Limits of this study</summary><ul class="list-plain">${items.map(([k, v]) => `<li><b>${k}.</b> ${esc(v)}</li>`).join("")}</ul></details>`;
}
function svgScatter(series, w, h) {
  const all = series.flatMap((s) => [...s.t, ...s.p]).filter(isFinite);
  let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.05 || 1; lo -= pad; hi += pad;
  const X = (v) => 40 + (v - lo) / (hi - lo) * (w - 50), Y = (v) => h - 24 - (v - lo) / (hi - lo) * (h - 34);
  let g = `<line x1="${X(lo)}" y1="${Y(lo)}" x2="${X(hi)}" y2="${Y(hi)}" stroke="#c9d1dd" stroke-dasharray="4 3"/>`;
  g += `<text x="${X(lo)}" y="${h - 6}">${fmt(lo, 3)}</text><text x="${X(hi) - 30}" y="${h - 6}">${fmt(hi, 3)}</text><text x="2" y="${Y(hi) + 22}">${fmt(hi, 3)}</text><text x="2" y="${Y(lo)}">${fmt(lo, 3)}</text>`;
  for (const s of series) for (let i = 0; i < s.t.length; i++) g += `<circle cx="${X(s.t[i]).toFixed(1)}" cy="${Y(s.p[i]).toFixed(1)}" r="2.6" fill="${s.c}" fill-opacity=".75"/>`;
  return `<svg viewBox="0 0 ${w} ${h}">${g}<text x="${w / 2 - 20}" y="${h - 6}">simulation</text><text x="2" y="12">model</text></svg>`;
}
function drawParity() {
  const P = RES.parity, el = $("#parity");
  if (RES.study.dataset.kind === "scalar") {
    const outs = Object.keys(P.test);
    el.innerHTML = `<div class="legend"><span><i style="background:#0f766e"></i>held-out test</span><span><i style="background:#d97706"></i>distribution shift</span></div>` +
      `<div class="grid2">${outs.map((o) => `<div class="parity"><b>${esc(o)}</b>${svgScatter([{ t: P.test[o].true, p: P.test[o].pred, c: "#0f766e" }, ...(P.shift ? [{ t: P.shift[o].true, p: P.shift[o].pred, c: "#d97706" }] : [])], 360, 260)}</div>`).join("")}</div>`;
  } else if (P.coords.shape.length > 2) {
    el.innerHTML = `<p class="meta">This field lives on a 3-D mesh (grid ${P.coords.shape.join(" × ")} with the channels ${esc((RES.study.dataset.meta.channels || []).join(", "))}): the held-out designs are shown on the body below.</p>`;
  } else {
    const W = (P.shift || P.test).worst, C = P.coords;
    el.innerHTML = `<p class="meta">Worst ${P.shift ? "shift" : "test"} run: ${Object.entries(W.inputs).map(([k, v]) => `${esc(k)} = ${fmt(v, 3)}`).join(", ")} (relative L2 ${pct(W.rel_l2)}). Axes: ${esc(C.names[0])} down, ${esc(C.names[1] || "")} across.</p>
      <div class="heat"><figure><figcaption>simulation</figcaption><canvas id="h1"></canvas></figure><figure><figcaption>model</figcaption><canvas id="h2"></canvas></figure><figure><figcaption>|error|</figcaption><canvas id="h3"></canvas></figure></div>`;
    const flat = (a) => a.flat(Infinity);
    const t = flat(W.true), p = flat(W.pred), e = t.map((v, i) => Math.abs(v - p[i]));
    const lo = Math.min(...t, ...p), hi = Math.max(...t, ...p);
    heat($("#h1"), t, C.shape, lo, hi); heat($("#h2"), p, C.shape, lo, hi); heat($("#h3"), e, C.shape, 0, Math.max(...e), true);
  }
}
function heat(cv, vals, shape, lo, hi, mono) {
  const [n0, n1] = shape.length > 1 ? shape : [shape[0], 1];
  cv.width = n1; cv.height = n0;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(n1, n0);
  for (let i = 0; i < n0; i++) for (let j = 0; j < n1; j++) {
    const v = (vals[i * n1 + j] - lo) / ((hi - lo) || 1), k = 4 * (i * n1 + j);
    const c = mono ? [255, Math.round(255 * (1 - v * 0.75)), Math.round(255 * (1 - v))] : [Math.round(255 * Math.min(1, 2 * v)), Math.round(255 * (1 - Math.abs(2 * v - 1)) * 0.8 + 40), Math.round(255 * Math.min(1, 2 - 2 * v))];
    img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
function drawBins() {
  const E = RES.study.limitations.error_by_input.slice(0, 3);
  $("#bins").innerHTML = E.map((c) => { const mx = Math.max(...c.bins.map((b) => b.mean_error));
    return `<div style="margin-top:8px"><b>${esc(c.input)}</b> <span class="meta">(worst / best quartile ${fmt(c.max_over_min, 3)}×)</span>${c.bins.map((b) => `<div class="binrow"><span>${esc(b.bin)} <small class="meta">n=${b.n}</small></span><div class="track"><div class="fill ${/training\)/.test(b.bin) ? "out" : ""}" style="width:${100 * b.mean_error / mx}%"></div></div><span class="num">${pct(b.mean_error)}</span></div>`).join("")}</div>`; }).join("");
}

// ------------------------------------------------------------------ predict
async function loadModels() {
  try { MODELS = (await api("/v1/models")).models; } catch (e) { MODELS = []; }
  if (!$("#mpick")) renderPredict(RES ? RES.model.name : (MODELS[0] || {}).name);
}
async function renderPredict(name) {
  if (!MODELS.length) { try { MODELS = (await api("/v1/models")).models; } catch (e) { return; } }
  if (!name) return;
  const d = await api(`/v1/models/${encodeURIComponent(name)}`);
  const b = d.validity_bounds || {};
  $("#predict").innerHTML = `<div class="layout"><div class="panel form"><h2>Query a hosted model</h2>
    <label>Model<select id="mpick">${MODELS.map((m) => `<option ${m.name === name ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</select></label>
    <p class="meta">${esc(d.method)} · ${esc(d.version)} · ${esc(d.stage)} · trust ${esc(d.trust || "—")} · OOD policy ${esc(d.ood_policy)}</p>
    <div class="predform">${d.inputs.map((k) => `<label>${esc(k)} <span class="u">${b[k] ? fmt(b[k][0], 3) + " .. " + fmt(b[k][1], 3) : ""}</span><input data-k="${esc(k)}" type="number" step="any" value="${b[k] ? +((b[k][0] + b[k][1]) / 2).toPrecision(4) : 0}"></label>`).join("")}</div>
    <button class="primary" id="go">Predict</button><div class="status-line" id="pstatus"></div></div>
    <div class="panel" id="pout"><div class="empty"><h3>One call, one answer, its validity</h3><p>Change an input past its training range to see the out-of-distribution gate respond.</p></div></div></div>`;
  $("#mpick").onchange = () => renderPredict($("#mpick").value);
  $("#go").onclick = async () => {
    const inputs = Object.fromEntries($$(".predform input").map((x) => [x.dataset.k, +x.value]));
    try {
      const r = await api(`/v1/models/${encodeURIComponent(name)}/predict`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ inputs }) });
      const o = r.ood[0];
      let h = `<div class="kpis">${Object.entries(r.outputs).map(([k, v]) => Array.isArray(v[0]) ? "" : `<div class="kpi"><div class="l">${esc(k)}</div><div class="v">${v[0] === null ? "refused" : fmt(v[0])}</div></div>`).join("")}<div class="kpi"><div class="l">Latency</div><div class="v">${fmt(r.latency_ms, 3)} ms</div><div class="s">server side, incl. the gate</div></div></div>`;
      const field = Object.entries(r.outputs).find(([, v]) => Array.isArray(v[0]));
      if (field && d.grid_shape.length > 2) h += `<p class="meta">${esc(field[0])}: ${field[1][0].flat(Infinity).length} values on a ${d.grid_shape.join(" × ")} grid (returned in the response).</p>`;
      else if (field) h += `<div class="heat" style="margin-top:12px"><figure><figcaption>${esc(field[0])}</figcaption><canvas id="hp" style="width:420px;height:240px"></canvas></figure></div>`;
      h += o.ood ? `<div class="ood"><b>Out of distribution${r.ood_policy === "refuse" ? " — refused" : " — an extrapolation"}</b>${o.reasons.map((x) => `<div>${esc(x)}</div>`).join("")}</div>` : `<p class="meta">Inside the training distribution (nearest training run at ${fmt(o.knn_distance, 3)} × the typical spacing).</p>`;
      h += `<p class="meta">Request ${esc(r.request_id)} · logged in the run log (<a class="link" href="/v1/models/${encodeURIComponent(name)}/runs" target="_blank">runs</a>, <a class="link" href="/v1/metrics" target="_blank">metrics</a>).</p>`;
      $("#pout").innerHTML = h;
      if (field && field[1][0] && d.grid_shape.length <= 2) { const f = field[1][0].flat(Infinity); heat($("#hp"), f, d.grid_shape, Math.min(...f), Math.max(...f)); }
    } catch (e) { $("#pstatus").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  };
}

function apiSnippets() {
  const o = location.origin;
  $("#c1").textContent = `# train: an example, or your table (upload first)
curl -s -X POST ${o}/api/uploads -F file=@runs.csv                      # -> {"id": "..."}
curl -s -X POST ${o}/api/studies -H 'content-type: application/json' \\
  -d '{"upload_id": "<id>", "inputs": ["T0","alpha_deg"], "outputs": ["Cl"], "budget_s": 5, "seeds": 2}'
curl -s ${o}/api/jobs/<job id>                                          # status, progress, result

# predict with the hosted model (validity range enforced)
curl -s -X POST ${o}/v1/models/example-airfoil-cl-cd/predict -H 'content-type: application/json' \\
  -d '{"inputs": {"T0": 0.15, "T1": 0.12, "T2": 0.13, "M0": 0.02, "M1": 0.05, "M2": 0.04, "alpha_deg": [2, 4, 30]}}'`;
  $("#c2").textContent = `import requests
r = requests.post("${o}/v1/models/example-airfoil-cl-cd/predict",
                  json={"inputs": [{"T0": 0.15, "T1": 0.12, "T2": 0.13, "M0": 0.02, "M1": 0.05, "M2": 0.04, "alpha_deg": 4.0}]})
print(r.json()["outputs"], r.json()["ood"])

# the same study in the library
from pinneapple_analysis import surrogate as S
study = S.run_study(S.examples.airfoil_rans(), budget=5.0, seeds=2)
print(study.best, study.trust["decision"], study.limitations["concentration"])`;
}
