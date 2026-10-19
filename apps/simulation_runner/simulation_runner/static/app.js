"use strict";
// Simulation Workflow Runner UI: pick a template and cases, start a run, follow its steps, download the dataset.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => (v === null || v === undefined || v === "" ? "—" : typeof v !== "number" ? esc(v) : !isFinite(v) ? "—" :
  Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(3) : +v.toPrecision(5) + "");
const STATES = ["done", "skipped", "running", "failed", "cancelled", "pending"];
let META = null, TPL = null, EXEC = "local", SRC = "example", UPLOADED = null, RUN = null, POLL = null, SELSTEP = null;

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`));
  if (n === "runs") loadRuns();
  if (n === "dataset") renderDataset();
}

async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j));
  return j;
}

async function init() {
  META = await api("/api/meta");
  TPL = META.templates.find((t) => t.available) || META.templates[0];
  renderTemplates(); renderExecutors(); renderTemplateDetail(); renderApi();
  $("#mode-note").textContent = META.self_hosted
    ? "Self-hosted runner: custom workflows, commands and Docker are enabled (see /docs)."
    : `Public server: only the built-in templates run here, up to ${META.max_cases} cases per run. Your own workflows, scripts and Docker steps run on a self-hosted runner.`;
  loadRuns();
}

function renderTemplates() {
  $("#templates").innerHTML = META.templates.map((t) => `<button type="button" class="tpl ${t.id === TPL.id ? "on" : ""}" data-id="${t.id}">
    <div class="row"><b>${esc(t.title)}</b>${t.available ? '<span class="pill ok">runs here</span>' : '<span class="pill needs">needs solver</span>'}</div>
    <span class="muted">${esc(t.steps.filter((s) => !s.final).map((s) => s.kind).join(" → "))}${t.available ? "" : " · " + esc(t.reason)}</span></button>`).join("");
  $$("#templates .tpl").forEach((b) => b.onclick = () => { TPL = META.templates.find((t) => t.id === b.dataset.id); UPLOADED = null; renderTemplates(); renderTemplateDetail(); renderCasesInfo(); renderExecutors(); });
}

function renderExecutors() {
  $("#executors").innerHTML = META.executors.map((e) => {
    const usable = e.available;
    const label = { local: "This server", docker: "Docker", slurm: "Slurm" }[e.name] || e.name;
    const note = e.name === "local" ? (TPL.available ? "runs the steps now" : "this template needs a solver here") : (e.note || e.reason || "");
    return `<button type="button" class="exe ${e.name === EXEC ? "on" : ""} ${usable ? "" : "off"}" data-n="${e.name}" ${usable ? "" : "disabled"}>
      <div class="row"><b>${esc(label)}</b><span class="mono">${esc(e.name)}</span></div><span class="muted">${esc(note)}</span></button>`;
  }).join("");
  $$("#executors .exe").forEach((b) => b.onclick = () => { if (!b.disabled) { EXEC = b.dataset.n; renderExecutors(); } });
}

function dagSvg(t) {
  const caseSteps = t.steps.filter((s) => !s.final), finals = t.steps.filter((s) => s.final);
  const cases = ["c01", "c02", "c03"], w = 150, h = 34, gx = 40, gy = 16, x0 = 12, y0 = 12;
  const W = x0 * 2 + (caseSteps.length + (finals.length ? 1 : 0)) * (w + gx) - gx, H = y0 * 2 + cases.length * (h + gy) - gy;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Workflow DAG">`;
  const box = (x, y, label, sub, fill) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${fill}" stroke="#0f766e55"/><text x="${x + 10}" y="${y + 15}" font-weight="600">${esc(label)}</text><text x="${x + 10}" y="${y + 28}" fill="#5d6b82" font-size="11">${esc(sub)}</text>`;
  cases.forEach((c, i) => {
    const y = y0 + i * (h + gy);
    caseSteps.forEach((st, j) => {
      const x = x0 + j * (w + gx);
      if (j) s += `<path d="M${x - gx} ${y + h / 2} L${x - 4} ${y + h / 2}" stroke="#0f766e" stroke-width="1.4" marker-end="url(#a)"/>`;
      s += box(x, y, `${c}.${st.id}`, st.kind, "#f0fdfa");
    });
  });
  if (finals.length) {
    const x = x0 + caseSteps.length * (w + gx), y = y0 + (cases.length - 1) * (h + gy) / 2;
    cases.forEach((c, i) => { const yy = y0 + i * (h + gy) + h / 2; s += `<path d="M${x - gx} ${yy} L${x - 4} ${y + h / 2}" stroke="#0f766e" stroke-width="1.2" marker-end="url(#a)"/>`; });
    s += box(x, y, finals[0].id, finals[0].kind + " · all cases", "#ecfeff");
  }
  return s + `<defs><marker id="a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" fill="#0f766e"/></marker></defs></svg>`;
}

function renderTemplateDetail() {
  const t = TPL;
  const rows = t.parameters.map((p) => `<tr><td><b>${esc(p.name)}</b></td><td>${esc(p.kind)}</td><td class="num">${p.kind === "constant" ? fmt(p.value) : fmt(p.low)}</td><td class="num">${p.kind === "constant" ? "" : fmt(p.high)}${p.log ? " (log)" : ""}</td><td>${esc(p.unit || "")}</td><td>${esc(p.group)}</td></tr>`).join("");
  const outs = Object.entries(t.outputs).map(([k, u]) => `<code>${esc(k)}</code>${u ? ` [${esc(u)}]` : ""}`).join(", ");
  $("#tpl-detail").innerHTML = `<h2 style="margin-top:0">${esc(t.title)} ${t.available ? '<span class="pill ok">runs here</span>' : `<span class="pill needs">needs solver: ${esc(t.reason)}</span>`}</h2>
    <p>${esc(t.description)}</p>
    <div class="dag">${dagSvg(t)}</div><p class="muted" style="margin:2px 0 0">… one row per case (3 shown); each box is a step with its own state, log, retries and content hash.</p>
    <div class="grid2r" style="margin-top:12px">
      <div><h3>Parameters (cases must stay inside these ranges)</h3><div class="scroll"><table><tr><th>name</th><th>kind</th><th class="num">low / value</th><th class="num">high</th><th>unit</th><th>group</th></tr>${rows}</table></div>
        <p class="muted">Dataset outputs (step <code>${esc(t.output_step)}</code>): ${outs}. <a class="link" href="/api/templates/${t.id}/example.csv">example case table (CSV)</a></p></div>
      <div class="cards3" style="grid-template-columns:1fr">
        <div><b>Traceable</b><span>every step keeps its state, attempts, time, outputs and log; runs are stored and can be reopened</span></div>
        <div><b>Failure handling</b><span>retries per step; a failure stops only the steps that need it; resume continues an interrupted run</span></div>
        <div><b>Skip-done</b><span>a step with the same content hash (parameters, inputs, upstream results) is reused, not recomputed</span></div>
        <div><b>Dataset</b><span>finished runs become a table + DatasetSpec (CSV, Parquet, HDF5, JSON) for the Surrogate Studio</span></div>
      </div>
    </div>
    ${window.renderScope ? renderScope(META.scope) : ""}`;
  renderCasesInfo();
}

$$("#case-src button").forEach((b) => b.onclick = () => {
  SRC = b.dataset.v; $$("#case-src button").forEach((x) => x.classList.toggle("on", x === b));
  $("#drop").hidden = SRC !== "upload"; renderCasesInfo();
});
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
$("#file").addEventListener("change", (e) => { if (e.target.files[0]) upload(e.target.files[0]); e.target.value = ""; });
async function upload(f) {
  const fd = new FormData(); fd.append("file", f, f.name);
  $("#cases-info").innerHTML = '<span class="spinner"></span>Reading…';
  try { UPLOADED = await api(`/api/cases/parse?template=${TPL.id}`, { method: "POST", body: fd }); }
  catch (e) { UPLOADED = null; $("#cases-info").innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  renderCasesInfo();
}
function renderCasesInfo() {
  const el = $("#cases-info");
  if (SRC === "example") { el.innerHTML = `${TPL.example_cases.length} cases from a Latin hypercube of the template's ranges.`; return; }
  if (!UPLOADED) { el.innerHTML = "Columns must include the template's parameters; a <code>case_id</code> column names the cases."; return; }
  const p = UPLOADED.problems.length ? `<span class="err">${esc(UPLOADED.problems.join("; "))}</span>` : `<span class="ok">ready</span>`;
  el.innerHTML = `<b>${esc(UPLOADED.name)}</b>: ${UPLOADED.rows} cases, id column ${esc(UPLOADED.id_column || "(generated)")}${UPLOADED.contract ? `, contract ${esc(UPLOADED.contract)}` : ""} · ${p}`;
}

async function start(body, btn) {
  const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>Starting…'; $("#status").textContent = "";
  try {
    const r = await api("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    showTab("runs"); openRun(r.run_id);
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; }
}
$("#start").onclick = (e) => {
  const body = { template: TPL.id, executor: EXEC, max_parallel: +$("#par").value || 2 };
  if (SRC === "upload") {
    if (!UPLOADED) { $("#status").innerHTML = '<span class="err">Upload a case table first.</span>'; return; }
    body.cases = UPLOADED.cases; if (UPLOADED.id_column) body.id_column = UPLOADED.id_column;
  }
  start(body, e.currentTarget);
};
$("#try").onclick = (e) => start({ template: "slab_heat", executor: "local", max_parallel: 2 }, e.currentTarget);

async function loadRuns() {
  let runs = [];
  try { runs = await api("/api/runs"); } catch (e) { return; }
  $("#run-list").innerHTML = runs.length ? runs.map((r) => `<div class="run-item ${RUN && RUN.id === r.id ? "on" : ""}" data-id="${r.id}">
    <div class="row"><b>${esc(r.template || r.name)}</b><span class="pill ${r.status}">${esc(r.status)}</span></div>
    <div class="row"><span class="id">${esc(r.id)}</span><span class="muted">${r.n_cases ?? "?"} cases · ${esc(r.executor || "")}</span></div></div>`).join("")
    : '<div class="muted">No runs yet.</div>';
  $$(".run-item").forEach((el) => el.onclick = () => openRun(el.dataset.id));
}

async function openRun(id) {
  clearTimeout(POLL);
  try { RUN = await api(`/api/runs/${id}`); } catch (e) { $("#run-detail").innerHTML = `<div class="panel err">${esc(e.message)}</div>`; return; }
  if (!SELSTEP || !RUN.steps.some((s) => s.step_id === SELSTEP)) SELSTEP = (RUN.steps.find((s) => s.status === "failed") || RUN.steps[0] || {}).step_id;
  renderRun(); loadRuns();
  const live = ["pending", "running"].includes(RUN.status) || (RUN.job && ["queued", "running"].includes(RUN.job.status));
  if (live) POLL = setTimeout(() => openRun(id), 900);
}

function renderRun() {
  const r = RUN, c = r.counts, total = Object.values(c).reduce((a, b) => a + b, 0) || 1;
  const bar = STATES.filter((s) => s !== "pending").map((s) => c[s] ? `<div class="p-${s}" style="width:${100 * c[s] / total}%" title="${s}: ${c[s]}"></div>` : "").join("");
  const os = r.metadata.output_step, main = r.steps.filter((s) => os && s.step_id.endsWith("." + os) && s.outputs);
  const outKeys = [...new Set((main.length ? main : r.steps).flatMap((s) => Object.keys(s.outputs || {}).filter((k) => typeof s.outputs[k] === "number")))].slice(0, 3);
  const rows = r.steps.map((s) => `<tr data-s="${esc(s.step_id)}" class="${s.step_id === SELSTEP ? "sel" : ""}"><td class="mono">${esc(s.step_id)}</td>
    <td><span class="pill ${s.status}">${s.status === "skipped" ? "reused" : esc(s.status)}</span></td><td class="num">${s.attempts || 0}</td>
    <td class="num">${s.seconds == null ? "—" : s.seconds.toFixed(3)}</td>${outKeys.map((k) => `<td class="num">${fmt(s.outputs?.[k])}</td>`).join("")}
    <td class="mono">${esc(s.hash)}</td></tr>`).join("");
  const job = r.job ? `${esc(r.job.status)}${r.job.message ? " · " + esc(r.job.message) : ""}${r.job.error ? ` · <span class="err">${esc(r.job.error)}</span>` : ""}` : "—";
  const live = r.job && ["queued", "running"].includes(r.job.status);
  $("#run-detail").innerHTML = `<div class="panel">
    <div class="row" style="display:flex;justify-content:space-between;gap:10px;align-items:center"><h2 style="margin:0">${esc(r.metadata.template || r.name)} <span class="pill ${r.status}">${esc(r.status)}</span></h2><span class="mono">${esc(r.id)}</span></div>
    <div class="kpis">
      <div class="kpi"><div class="l">Steps</div><div class="v">${total}</div><div class="s">${r.metadata.n_cases ?? "?"} cases · ${esc(r.metadata.executor)}</div></div>
      <div class="kpi"><div class="l">Ran</div><div class="v">${c.done}</div><div class="s">${c.failed} failed · ${c.cancelled} cancelled</div></div>
      <div class="kpi"><div class="l">Reused (skip-done)</div><div class="v">${c.skipped}</div><div class="s">same content hash</div></div>
      <div class="kpi"><div class="l">Compute</div><div class="v">${r.compute_seconds.toFixed(2)} s</div><div class="s">${r.storage_mb.toFixed(2)} MB stored</div></div>
    </div>
    <div class="progress">${bar}</div><div class="muted">Queue job: ${job}</div>
    <div class="actions">
      <button id="a-cancel" ${live ? "" : "disabled"}>Cancel</button>
      <button id="a-resume" ${live || r.status === "done" ? "disabled" : ""}>Resume</button>
      <button id="a-rerun" ${r.metadata.template ? "" : "disabled"}>Run again (reuses unchanged steps)</button>
      <a href="/api/runs/${r.id}/workflow" target="_blank">workflow.json</a>
      <a href="#" id="a-data">Dataset →</a>
    </div></div>
  <div class="panel grid2r">
    <div><h3 style="margin-top:0">Steps</h3><div class="scroll"><table class="steps"><tr><th>step</th><th>state</th><th class="num">tries</th><th class="num">s</th>${outKeys.map((k) => `<th class="num">${esc(k)}</th>`).join("")}<th>hash</th></tr>${rows}</table></div></div>
    <div><h3 style="margin-top:0">Log <span class="mono">${esc(SELSTEP || "")}</span></h3><div class="logbox" id="logbox">…</div><div id="stepinfo" class="muted" style="margin-top:6px"></div></div>
  </div>`;
  $$("table.steps tr[data-s]").forEach((tr) => tr.onclick = () => { SELSTEP = tr.dataset.s; renderRun(); });
  $("#a-cancel").onclick = () => api(`/api/runs/${r.id}/cancel`, { method: "POST" }).then(() => openRun(r.id)).catch((e) => alert(e.message));
  $("#a-resume").onclick = () => api(`/api/runs/${r.id}/resume`, { method: "POST" }).then(() => openRun(r.id)).catch((e) => alert(e.message));
  $("#a-rerun").onclick = () => api(`/api/runs/${r.id}/rerun`, { method: "POST" }).then((x) => openRun(x.run_id)).catch((e) => alert(e.message));
  $("#a-data").onclick = (e) => { e.preventDefault(); showTab("dataset"); };
  const step = r.steps.find((s) => s.step_id === SELSTEP);
  if (step) {
    fetch(`/api/runs/${r.id}/steps/${encodeURIComponent(step.step_id)}/log`).then((x) => x.text()).then((t) => { const b = $("#logbox"); if (b) { b.textContent = t || "(no log yet)"; b.scrollTop = b.scrollHeight; } });
    const outs = step.outputs ? Object.entries(step.outputs).map(([k, v]) => `${esc(k)} = ${typeof v === "object" ? esc(JSON.stringify(v)) : fmt(v)}`).join(" · ") : "";
    $("#stepinfo").innerHTML = `${step.error ? `<span class="err">${esc(step.error)}</span><br>` : ""}${outs}`
      + (step.viewable ? `<div style="margin-top:8px"><button class="primary" id="a-view">View result in 3-D</button> <span class="muted">mesh, fields, ${step.kind === "calculix" ? "deformed shape, " : ""}boundary conditions and loads: a check before the numbers are used</span></div>` : "");
    if (step.viewable) $("#a-view").onclick = () => viewStep(r.id, step);
  }
  if (VIEWED && (VIEWED.run !== r.id || VIEWED.step !== SELSTEP)) { $("#v3panel").hidden = true; $("#v3panel").innerHTML = ""; VIEWED = null; }
}

// ------------------------------------------------------------------ 3-D view of a step's result (shared viewer)
let VIEWED = null, V3 = null;
const whenViewer = () => new Promise((res) => window.FieldView3D ? res() : window.addEventListener("fieldview3d-ready", res, { once: true }));
async function viewStep(runId, step) {
  const el = $("#v3panel");
  el.hidden = false;
  el.innerHTML = '<span class="spinner"></span> Reading the result…';
  VIEWED = { run: runId, step: step.step_id };
  let j;
  try { j = await api(`/api/runs/${runId}/steps/${encodeURIComponent(step.step_id)}/view3d`); } catch (e) { el.innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  const i = j.info, v = j.view;
  const conds = i.conditions.length ? `<table class="conds"><tr><th>Region</th><th>as</th><th>from the case</th><th class="num">${i.kind === "calculix" ? "nodes" : "faces"}</th></tr>${i.conditions.map((c) => `<tr><td><b>${esc(c.name)}</b></td><td><span class="mk ${c.kind}">${c.kind === "bc" ? "boundary condition" : "load"}</span>${c.role ? `<div class="muted">${esc(c.role)}</div>` : ""}</td><td>${esc(c.keyword ? `${c.keyword}, line ${c.line}: ${c.detail}` : `${c.type} patch: ${c.detail}`)}</td><td class="num">${c.nodes ?? c.faces}</td></tr>`).join("")}</table>`
    : `<p class="muted">The case defines no boundary condition or load that could be placed on the mesh${i.deck === null ? " (no input deck next to the result)" : ""}.</p>`;
  el.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px"><h3 style="margin:0">Result of <span class="mono">${esc(step.step_id)}</span> in 3-D</h3>
      <span class="muted">${esc(i.solver)} · ${esc(i.file)}${i.time != null ? ` · t = ${fmt(i.time)}` : ""} · ${i.points.toLocaleString("en-US")} points, ${i.cells.toLocaleString("en-US")} cells${v.decimated ? ` · shown with ${v.faces.toLocaleString("en-US")} of ${v.faces_original.toLocaleString("en-US")} triangles` : ""}</span></div>
    <div class="v3wrap"><div id="v3d"></div><div><h4 style="margin:4px 0 6px">Where the case acts</h4>${conds}
      ${(v.notes || []).map((t) => `<p class="muted">${esc(t)}</p>`).join("")}
      <p class="muted">${i.kind === "calculix"
        ? "Check that supports and loads sit where they should, the deformed shape bends the expected way and the fields have plausible ranges, before the step's numbers go into the dataset."
        : "Check that each patch carries the condition you meant (walls, the driving wall or inlet, outlets), that the flow moves where the driving condition pushes it, and that the fields have plausible ranges, before the step's numbers go into the dataset."} Units are those of the case${i.length_unit === "model units" ? " (the deck's own)" : ""}.</p></div></div>`;
  await whenViewer();
  const start = v.fields.find((f) => f.name === "STRESS") || v.fields.find((f) => /^U$/.test(f.name)) || v.fields[0];
  V3 = new FieldView3D($("#v3d"), { meta: v, glb: j.glb, field: start.name, height: 460 });
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function renderDataset() {
  const el = $("#dataset");
  if (!RUN) return;
  if (!RUN.metadata.template) { el.innerHTML = `<div class="panel empty"><h3>Custom workflow</h3><p>Datasets are assembled from template runs.</p></div>`; return; }
  el.innerHTML = '<div class="panel"><span class="spinner"></span>Assembling the dataset…</div>';
  let d;
  try { d = await api(`/api/runs/${RUN.id}/dataset-spec?format=json`); } catch (e) { el.innerHTML = `<div class="panel err">${esc(e.message)}</div>`; return; }
  const s = d.spec, cols = s.columns;
  const head = cols.map((c) => `<th class="${c.role === "output" ? "num" : ""}">${esc(c.name)}<br><span class="role ${c.role}">${c.role}</span>${c.unit ? ` <span class="muted">[${esc(c.unit)}]</span>` : ""}</th>`).join("");
  const body = d.preview.map((r) => `<tr>${cols.map((c) => `<td class="${c.name === "run_step" ? "nowrap" : typeof r[c.name] === "number" ? "num" : ""}">${c.name === "status" ? `<span class="pill ${r[c.name]}">${esc(r[c.name])}</span>` : c.name === "run_step" ? `<span title="${esc(r[c.name])}">…/${esc(String(r[c.name]).split("/").pop())}</span>` : fmt(r[c.name])}</td>`).join("")}</tr>`).join("");
  const dl = ["csv", "parquet", "h5", "json"].map((f) => `<a href="/api/runs/${RUN.id}/dataset?format=${f}">${f.toUpperCase()}</a>`).join("");
  el.innerHTML = `<div class="panel">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px"><h2 style="margin:0">${esc(s.name)} <span class="pill ${s.status === "complete" ? "done" : "running"}">${esc(s.status)}</span></h2><span class="mono">DatasetSpec ${esc(s.schema_version)}</span></div>
    <div class="kv" style="margin:10px 0"><span>Rows</span><span>${s.n_rows} cases (${d.preview.filter((r) => r.status === "done" || r.status === "skipped").length} with outputs)</span>
      <span>Inputs</span><span>${cols.filter((c) => c.role === "input").map((c) => esc(c.name)).join(", ")}</span>
      <span>Outputs</span><span>${cols.filter((c) => c.role === "output").map((c) => esc(c.name)).join(", ")}</span>
      <span>Provenance</span><span>run <span class="mono">${esc(s.provenance.study_id)}</span>, one run step per row (column <code>run_step</code>)</span>
      <span>File</span><span class="mono">${esc(s.files[0].path)} · sha256 ${esc(s.files[0].sha256.slice(0, 16))}…</span></div>
    <div class="actions">${dl}<a href="#" id="dl-spec">DatasetSpec JSON</a></div>
    <div class="scroll"><table>${"<tr>" + head + "</tr>"}${body}</table></div>
    <p class="muted">Every value comes from this run's steps. Load it in the Surrogate Studio, or in Python with
      <code>pinneapple_data.contracts.read_table</code>; the sidecar <code>.contract.json</code> carries the DatasetSpec.</p></div>`;
  $("#dl-spec").onclick = (e) => {
    e.preventDefault();
    const blob = new Blob([JSON.stringify(s, null, 1)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${s.name}.dataset.json`; a.click();
  };
}

function renderApi() {
  const o = location.origin;
  $("#c1").textContent = `# start a run of a built-in template with your own cases (CSV columns = template parameters)
curl -s -X POST ${o}/api/runs -H 'Content-Type: application/json' \\
  -d '{"template": "slab_heat", "max_parallel": 2, "cases": [
        {"case_id": "a", "L": 0.05, "alpha": 1e-5, "T_left": 400, "t_end": 60},
        {"case_id": "b", "L": 0.02, "alpha": 4e-6, "T_left": 350, "t_end": 120}]}'
# -> {"run_id": "run-...", "job_id": "run-...", "steps": 3}

curl -s ${o}/api/runs/<run_id>                         # states, attempts, outputs, compute time
curl -s ${o}/api/runs/<run_id>/steps/a.solve/log       # one step's log
curl -s -X POST ${o}/api/runs/<run_id>/resume          # continue an interrupted or failed run
curl -s -o slab.parquet "${o}/api/runs/<run_id>/dataset?format=parquet"
curl -s ${o}/api/usage                                 # runs, compute seconds and storage this month`;
  $("#c2").textContent = `from pinneapple_core.workflows import Step, Workflow, WorkflowRunner, TEMPLATES, dataset_from_run

runner = WorkflowRunner("work/runner")
wf = Workflow("beam", [
    Step("mesh", "command", {"argv": ["gmsh", "-3", "beam.geo"]}, inputs=["beam.geo"]),
    Step("solve", "calculix", {"inp": "beam.inp"}, needs=["mesh"], retries=1, executor="slurm",
         executor_options={"partition": "cpu", "ntasks": 8, "dry_run": False}),
])
run = runner.run(wf)                      # again with run_id=run["id"] to resume

t = TEMPLATES["slab_heat"]; cases = t.example_cases(20)
run = runner.run(t.workflow(cases))
spec, path = dataset_from_run(runner, run["id"], cases, output_step="solve", path="slab.parquet")`;
  $("#c3").textContent = `from pinneapple_core.workflows import Adapter, Executor, ExecResult, register_adapter, register_executor

class GuiStep(Adapter):                   # e.g. the Engineering GUI Agent (guiagent) drives a CAE GUI
    kind = "gui"
    def run(self, ctx):                   # ctx.workdir, ctx.params, ctx.upstream, ctx.execute(argv), ctx.log
        ...
        return {"exported": "mesh.inp"}   # JSON outputs: stored, hashed for skip-done, shown here

register_adapter(GuiStep())               # steps with kind="gui" now run in any workflow
register_executor("mycluster", MyExecutor)  # subclass Executor.run(argv, cwd=..., timeout=..., log=...)`;
}

init().catch((e) => { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; });
