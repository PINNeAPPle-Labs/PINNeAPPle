// Engineering Design Optimizer: form -> POST /api/runs (a job of the shared runtime) -> poll -> render.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function fmt(v, p = 4) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
  return +v.toPrecision(p) + "";
}
const LABELS = { mass_g: "mass [g]", cost: "unit cost", pressure_drop_pa: "pressure drop [Pa]", t_source_c: "source temperature [°C]",
  r_base_k_w: "base-to-air resistance [K/W]", fin_stress_mpa: "fin stress [MPa]", fin_gap_mm: "fin gap [mm]", fin_aspect: "fin height / gap" };
const lab = (k) => LABELS[k] || k;
let META = null, ZOO = [], SRC = "preset", TABLE = null, JOB = null, LAST = null, POLL = null;

// ---------------------------------------------------------------------------------------------- tabs
document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => showTab(b.dataset.tab));
function showTab(t) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + t));
}

// ---------------------------------------------------------------------------------------------- init
async function init() {
  META = await (await fetch("/api/meta")).json();
  const hs = META.heatsink;
  $("#hs-objs").innerHTML = hs.objectives.map((o) => `<label><input type="checkbox" value="${o}" ${["mass_g", "cost"].includes(o) ? "checked" : ""}> ${esc(lab(o))}</label>`).join("");
  fillPreset();
  $("#hs-bounds").innerHTML = "<tr><td></td><td class='meta'>lower</td><td class='meta'>upper</td></tr>" + hs.variables.map((v) =>
    `<tr><td>${esc(v.description || v.name)} <span class="u">${esc(v.unit)}</span></td>
     <td><input id="lo-${v.name}" type="number" step="any" value="${v.lower}"></td><td><input id="hi-${v.name}" type="number" step="any" value="${v.upper}"></td></tr>`).join("");
  $("#method").innerHTML = Object.entries(META.methods).map(([m, v]) =>
    `<option value="${m}" ${v.available ? "" : "disabled"}>${esc({ nsga2: "NSGA-II (PINNeAPPle)", "pymoo-nsga2": "NSGA-II (pymoo)", "scipy-de": "Differential evolution (SciPy)", bayesian: "Bayesian optimisation (GP)", random: "Latin hypercube (baseline)" }[m] || m)}${v.available ? "" : " — " + esc(v.reason)}</option>`).join("");
  $("#method").value = "nsga2";
  if (window.renderScope) $("#scope").innerHTML = renderScope(META.scope);
  setSource("preset");
  snippets();
  fetch("/api/zoo").then((r) => r.json()).then((z) => { ZOO = z; $("#zoo-model").innerHTML = z.map((m) => `<option value="${m.name}">${esc(m.name)} — ${esc(m.task)}</option>`).join(""); renderZoo(); });
}
function fillPreset() {
  const hs = META.heatsink;
  for (const [k, v] of Object.entries(hs.operating)) if ($("#op-" + k)) $("#op-" + k).value = v;
  for (const [k, v] of Object.entries(hs.limits)) if ($("#lim-" + k)) $("#lim-" + k).value = v;
  for (const [k, v] of Object.entries(hs.cost)) if ($("#cost-" + k)) $("#cost-" + k).value = v;
}

document.querySelectorAll("#source button").forEach((b) => b.onclick = () => setSource(b.dataset.v));
function setSource(s) {
  SRC = s;
  document.querySelectorAll("#source button").forEach((b) => b.classList.toggle("on", b.dataset.v === s));
  for (const k of ["preset", "table", "zoo"]) $("#src-" + k).hidden = k !== s;
  const opts = [];
  if (s === "preset") opts.push(["library", "FVM heat-sink engine (3-D base conduction)"]);
  for (const k of Object.keys(META.solvers || {})) opts.push([k, "Solver: " + k]);
  opts.push(["none", s === "preset" ? "None" : "None (verify later with your solver results)"]);
  $("#verify").innerHTML = opts.map(([v, t]) => `<option value="${v}">${esc(t)}</option>`).join("");
  $("#mode").disabled = opts[0][0] === "none";
  if ($("#mode").disabled) $("#mode").value = "direct";
  renderOC();
}

// ---------------------------------------------------------------------------------------------- objectives / constraints
let OC = { objectives: [], constraints: [] };
function outputsOf() {
  if (SRC === "table" && TABLE) return TABLE.columns.filter((c) => TABLE.role[c.name] === "output").map((c) => c.name);
  if (SRC === "zoo") { const m = ZOO.find((z) => z.name === $("#zoo-model").value); return m ? m.outputs : []; }
  return [];
}
function renderOC() {
  const box = $("#oc-builder");
  if (SRC === "preset") { box.innerHTML = ""; return; }
  const outs = outputsOf();
  if (!outs.length) { box.innerHTML = ""; return; }
  OC.objectives = OC.objectives.filter((o) => outs.includes(o.name));
  OC.constraints = OC.constraints.filter((o) => outs.includes(o.name));
  if (!OC.objectives.length) OC.objectives = outs.slice(0, Math.min(2, outs.length)).map((n) => ({ name: n, sense: "min" }));
  const sel = (v) => outs.map((o) => `<option ${o === v ? "selected" : ""}>${esc(o)}</option>`).join("");
  box.innerHTML = `<fieldset><legend>Objectives</legend>${OC.objectives.map((o, i) => `<div class="oc"><select data-k="o" data-i="${i}" data-f="name">${sel(o.name)}</select>
      <select data-k="o" data-i="${i}" data-f="sense"><option value="min" ${o.sense === "min" ? "selected" : ""}>min</option><option value="max" ${o.sense === "max" ? "selected" : ""}>max</option></select><span></span>
      <button data-del="o" data-i="${i}" title="remove">×</button></div>`).join("")}
    <button class="ghost" type="button" id="add-o">+ objective</button></fieldset>
    <fieldset><legend>Constraints</legend>${OC.constraints.map((c, i) => `<div class="oc"><select data-k="c" data-i="${i}" data-f="name">${sel(c.name)}</select>
      <select data-k="c" data-i="${i}" data-f="op"><option value="upper" ${c.op === "upper" ? "selected" : ""}>≤</option><option value="lower" ${c.op === "lower" ? "selected" : ""}>≥</option></select>
      <input data-k="c" data-i="${i}" data-f="value" type="number" step="any" value="${c.value ?? ""}"><button data-del="c" data-i="${i}" title="remove">×</button></div>`).join("")}
    <button class="ghost" type="button" id="add-c">+ constraint</button></fieldset>`;
  box.querySelectorAll("[data-k]").forEach((el) => el.onchange = () => {
    const arr = el.dataset.k === "o" ? OC.objectives : OC.constraints;
    arr[+el.dataset.i][el.dataset.f] = el.dataset.f === "value" ? parseFloat(el.value) : el.value;
  });
  box.querySelectorAll("[data-del]").forEach((b) => b.onclick = () => { (b.dataset.del === "o" ? OC.objectives : OC.constraints).splice(+b.dataset.i, 1); renderOC(); });
  $("#add-o").onclick = () => { OC.objectives.push({ name: outs[0], sense: "min" }); renderOC(); };
  $("#add-c").onclick = () => { OC.constraints.push({ name: outs[outs.length - 1], op: "upper", value: null }); renderOC(); };
}

// ---------------------------------------------------------------------------------------------- your data
const drop = $("#drop");
["dragover", "dragenter"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, () => drop.classList.remove("over")));
drop.addEventListener("drop", (ev) => { ev.preventDefault(); if (ev.dataTransfer.files[0]) upload(ev.dataTransfer.files[0]); });
$("#file").onchange = (e) => e.target.files[0] && upload(e.target.files[0]);
async function upload(f) {
  $("#fileinfo").textContent = "Uploading " + f.name + "…";
  const fd = new FormData(); fd.append("file", f);
  const r = await fetch("/api/uploads", { method: "POST", body: fd });
  const up = await r.json();
  if (!r.ok) { $("#fileinfo").innerHTML = `<span class="err">${esc(up.detail || JSON.stringify(up))}</span>`; return; }
  const ins = await (await fetch("/api/table/inspect", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ upload_id: up.id }) })).json();
  if (!ins.columns) { $("#fileinfo").innerHTML = `<span class="err">${esc(ins.detail)}</span>`; return; }
  const num = ins.columns.filter((c) => c.numeric);
  const role = {};
  const hint = ins.contract_hint || {};
  num.forEach((c, i) => role[c.name] = hint.inputs ? (hint.inputs.includes(c.name) ? "input" : hint.outputs?.includes(c.name) ? "output" : "ignore")
    : (i < Math.ceil(num.length / 2) ? "input" : "output"));
  TABLE = { id: up.id, name: f.name, rows: ins.rows, columns: num, role };
  $("#fileinfo").textContent = `${f.name}: ${ins.rows} rows, ${num.length} numeric columns`;
  renderCols();
}
function renderCols() {
  $("#cols").innerHTML = `<fieldset><legend>Columns</legend>${TABLE.columns.map((c) => `<div class="colrow"><span>${esc(c.name)} <span class="rng">${fmt(c.min, 3)} – ${fmt(c.max, 3)}</span></span>
    <select data-col="${esc(c.name)}"><option value="input" ${TABLE.role[c.name] === "input" ? "selected" : ""}>design variable</option><option value="output" ${TABLE.role[c.name] === "output" ? "selected" : ""}>output</option><option value="ignore" ${TABLE.role[c.name] === "ignore" ? "selected" : ""}>ignore</option></select></div>`).join("")}</fieldset>`;
  $("#cols").querySelectorAll("select").forEach((s) => s.onchange = () => { TABLE.role[s.dataset.col] = s.value; renderOC(); });
  renderOC();
}

// ---------------------------------------------------------------------------------------------- zoo
$("#zoo-model").onchange = () => { OC = { objectives: [], constraints: [] }; renderZoo(); };
function renderZoo() {
  const m = ZOO.find((z) => z.name === $("#zoo-model").value);
  if (!m) { $("#zoo-inputs").innerHTML = ""; return; }
  $("#zoo-inputs").innerHTML = `<fieldset><legend>Inputs</legend><p class="hint">Variables are bounded by the model's validity range; fix an input to optimise at that value.</p>${m.inputs.map((i) => `<div class="colrow"><span>${esc(i.name)} <span class="rng">${i.range ? fmt(i.range[0], 3) + " – " + fmt(i.range[1], 3) : "no range"}</span></span>
    <input data-zin="${esc(i.name)}" placeholder="${i.range ? "variable" : "fixed value"}" title="leave empty to optimise; a number fixes it"></div>`).join("")}</fieldset>`;
  renderOC();
}

// ---------------------------------------------------------------------------------------------- spec + run
const num = (id) => parseFloat($(id).value);
function buildSpec() {
  if (SRC === "preset") {
    const hs = META.heatsink;
    const pick = (pre, obj) => Object.fromEntries(Object.keys(obj).filter((k) => $(`#${pre}-${k}`)).map((k) => [k, num(`#${pre}-${k}`)]));
    const objectives = [...document.querySelectorAll("#hs-objs input:checked")].map((i) => ({ name: i.value }));
    return { source: "preset", preset: "heatsink", fidelity: $("#fidelity").value, operating: pick("op", hs.operating),
      limits: pick("lim", hs.limits), cost: pick("cost", Object.fromEntries(Object.entries(hs.cost).filter(([k]) => k !== "currency"))),
      variables: hs.variables.map((v) => ({ name: v.name, lower: num("#lo-" + v.name), upper: num("#hi-" + v.name) })), objectives };
  }
  const objectives = OC.objectives.map((o) => ({ name: o.name, sense: o.sense }));
  const constraints = OC.constraints.filter((c) => isFinite(c.value)).map((c) => ({ name: c.name, [c.op]: c.value }));
  if (SRC === "table") {
    if (!TABLE) throw new Error("upload a table first");
    const inputs = TABLE.columns.filter((c) => TABLE.role[c.name] === "input");
    const integer = inputs.filter((c) => c.unique && c.unique <= 60 && Number.isInteger(c.min) && Number.isInteger(c.max) && c.max - c.min >= 1 && c.unique === c.max - c.min + 1 || (c.unique && c.unique < 40 && /(^n_|count|number)/i.test(c.name))).map((c) => c.name);
    return { source: "table", inputs: inputs.map((c) => c.name), outputs: outputsOf(), integer, objectives, constraints };
  }
  const fixed = {};
  document.querySelectorAll("[data-zin]").forEach((i) => { if (i.value.trim() !== "") fixed[i.dataset.zin] = parseFloat(i.value); });
  return { source: "zoo", model: $("#zoo-model").value, fixed, objectives, constraints };
}

$("#example").onclick = () => { setSource("preset"); fillPreset(); $("#method").value = "nsga2"; $("#budget").value = 600; $("#n_verify").value = 5; $("#mode").value = "direct"; $("#verify").value = "library"; start(); };
$("#run").onclick = () => start();
$("#cancel").onclick = async () => { if (JOB) await fetch(`/api/jobs/${JOB}/cancel`, { method: "POST" }); };

async function start() {
  let spec;
  try { spec = buildSpec(); } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  const body = { spec, method: $("#method").value, budget: parseInt($("#budget").value), seed: parseInt($("#seed").value) || 0,
    n_verify: parseInt($("#n_verify").value), verify_with: $("#verify").value, mode: $("#mode").value, upload_id: TABLE && SRC === "table" ? TABLE.id : null };
  $("#example").disabled = $("#run").disabled = true;
  $("#status").textContent = "Queued…"; $("#prog").hidden = false; $("#prog div").style.width = "2%";
  const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) { done(); $("#status").innerHTML = `<span class="err">${esc(typeof j.detail === "string" ? j.detail : j.message || JSON.stringify(j.detail || j))}</span>`; return; }
  JOB = j.id; $("#cancel").hidden = false;
  poll();
}
function done() { $("#example").disabled = $("#run").disabled = false; $("#prog").hidden = true; $("#cancel").hidden = true; }
async function poll() {
  const j = await (await fetch(`/api/jobs/${JOB}`)).json();
  $("#prog div").style.width = Math.round(100 * (j.progress || 0)) + "%";
  $("#status").textContent = `${j.status}${j.message ? " · " + j.message : ""}`;
  if (j.status === "done") { done(); $("#status").textContent = `Done in ${j.result.seconds} s`; LAST = j.result; showTab("result"); renderResult(j.result, j.id); window.scrollTo(0, 0); return; }
  if (j.status === "failed" || j.status === "cancelled") { done(); $("#status").innerHTML = `<span class="err">${esc(j.error || j.status)}</span>`; return; }
  POLL = setTimeout(poll, 400);
}

// ---------------------------------------------------------------------------------------------- results
const VCLS = { verified: "verified", "rejected by reference": "rejected", "reference failed": "failed", "only surrogate": "surrogate", reference: "verified" };
const VCOL = { verified: "#1a7f4b", rejected: "#b42318", failed: "#9a6200", surrogate: "#7a8599" };
const vbadge = (s) => `<span class="vstat ${VCLS[s] || "surrogate"}">${esc(s === "reference" ? "reference run" : s)}</span>`;

function renderResult(res, jid) {
  const P = res.problem, objs = P.objectives, cons = P.constraints;
  const vars = P.space.variables.map((v) => v.name);
  const cands = res.candidates;
  const counts = {};
  cands.forEach((c) => counts[c.verification] = (counts[c.verification] || 0) + 1);
  const nVer = counts.verified || 0, nRej = counts["rejected by reference"] || 0;
  let banner;
  if (res.mode === "active_learning") banner = `<div class="banner ok">Every design on this front was evaluated with ${esc(res.reference?.name)} (${res.rounds.length} active-learning rounds).</div>`;
  else if (res.verification) banner = `<div class="banner ${nRej ? "warn" : "ok"}">${nVer} of ${cands.length} candidates verified by ${esc(res.verification.reference.name)}${nRej ? `; ${nRej} rejected by it` : ""}.</div>`;
  else banner = `<div class="banner warn">Candidates are surrogate predictions only: no reference was run on them.</div>`;
  const best = objs.map((o) => {
    const feas = res.points.filter((p) => p.f && !p.r && p[o.name] !== null);
    const v = feas.length ? (o.sense === "min" ? Math.min(...feas.map((p) => p[o.name])) : Math.max(...feas.map((p) => p[o.name]))) : null;
    let delta = "";
    if (res.baseline && v !== null && res.baseline[o.name]) {
      const d = 100 * (v - res.baseline[o.name]) / Math.abs(res.baseline[o.name]);
      const good = o.sense === "min" ? d < 0 : d > 0;
      delta = `<div class="s"><span class="${good ? "better" : "worse"}">${d > 0 ? "+" : ""}${d.toFixed(0)} %</span> vs baseline ${fmt(res.baseline[o.name])}</div>`;
    }
    return `<div class="kpi"><div class="l">best ${esc(o.label || lab(o.name))} (${o.sense})</div><div class="v">${fmt(v)}</div>${delta}</div>`;
  }).join("");
  const err = res.verification ? Object.entries(res.verification.errors).filter(([k]) => objs.some((o) => o.name === k) || cons.some((c) => c.name === k)) : [];
  const maxErr = err.length ? Math.max(...err.map(([, e]) => e.max_rel)) : null;
  const s = res.summary;
  let html = `<div class="panel">${banner}
    <div class="kpis"><div class="kpi"><div class="l">evaluations</div><div class="v">${s.n_evals}</div><div class="s">${esc(s.method)} · ${esc(P.evaluator.name)}</div></div>
      <div class="kpi"><div class="l">feasible designs</div><div class="v">${s.n_feasible}</div><div class="s">${s.n_pareto} on the Pareto front</div></div>
      ${best}
      <div class="kpi"><div class="l">max surrogate error at candidates</div><div class="v">${maxErr === null ? "—" : (100 * maxErr).toFixed(2) + " %"}</div><div class="s">${res.verification ? "vs " + esc(res.verification.reference.name) : "no reference run"}</div></div>
      <div class="kpi"><div class="l">run time</div><div class="v">${res.seconds} s</div><div class="s">search ${fmt(s.wall_s, 3)} s${res.verification ? ", reference " + fmt(res.verification.wall_s, 3) + " s" : ""}</div></div></div>
    ${res.fit ? fitBlock(res.fit) : ""}
    <div class="grid2" style="margin-top:14px">
      <div><div class="toolbar"><h3 style="margin:0">Trade-off</h3><div class="axsel">x <select id="ax-x"></select> y <select id="ax-y"></select></div></div><div id="scatter" class="scatter"></div>
        <div class="legend"><span><i style="background:#c9d1dd"></i>infeasible</span><span><i style="background:#93c5fd"></i>feasible</span><span><i style="background:#0b6bcb"></i>Pareto front</span><span><i style="background:#1a7f4b"></i>verified</span><span><i style="background:#b42318"></i>rejected by reference</span>${res.baseline ? '<span><i style="background:#172033"></i>baseline</span>' : ""}</div></div>
      <div><h3 style="margin-top:0">Convergence</h3><div id="conv"></div><p class="hint">Hypervolume of the feasible designs found so far (objectives normalised over this run's feasible designs). A flat end means more evaluations bring little.</p>
        ${res.rounds ? roundsBlock(res.rounds) : ""}</div></div>
    <h3>Candidates</h3>
    <div style="overflow-x:auto"><table><thead><tr><th>#</th>${vars.map((v) => `<th class="num">${esc(v)}</th>`).join("")}${objs.map((o) => `<th class="num">${esc(lab(o.name))}</th>`).join("")}<th>status</th></tr></thead>
      <tbody>${cands.map((c, k) => `<tr><td>${k + 1}</td>${vars.map((v) => `<td class="num">${fmt(c[v])}</td>`).join("")}${objs.map((o) => `<td class="num">${fmt(c[o.name])}</td>`).join("")}<td>${vbadge(c.verification)}</td></tr>`).join("")}</tbody></table></div>
    <h3>Feasibility and high-fidelity confirmation</h3>
    <div id="cards" class="cards"></div>
    ${res.fit ? verifyUploadBlock() : ""}
    <h3>Downloads</h3><div class="files">${res.files.map((f) => `<a class="link" href="/api/runs/${jid}/files/${f}">${esc(f)}</a>`).join("")}</div>
    <p class="hint">study.json is a StudyResult (checks: ${res.study.checks.map((c) => `${esc(c.name)} ${esc(c.status)}`).join(", ")}); parameter_space.json a ParameterSpace; both open in the other PINNeAPPle apps.</p>
    ${P.notes.length ? `<div class="warnings">${P.notes.map((n) => `<div>${esc(n)}</div>`).join("")}</div>` : ""}</div>`;
  $("#result").innerHTML = html;
  // axes: objectives first, then constrained outputs
  const keys = [...new Set([...objs.map((o) => o.name), ...cons.map((c) => c.name)])].filter((k) => k in res.points[0]);
  $("#ax-x").innerHTML = keys.map((k) => `<option value="${k}">${esc(lab(k))}</option>`).join("");
  $("#ax-y").innerHTML = $("#ax-x").innerHTML;
  $("#ax-x").value = keys[0]; $("#ax-y").value = keys[1] || keys[0];
  $("#ax-x").onchange = $("#ax-y").onchange = () => scatter(res);
  scatter(res);
  if (res.convergence.length > 1) PChart.mount($("#conv"), { height: 220, x: res.convergence.map((p) => p[0]), yLabel: "hypervolume", series: [{ name: "hypervolume", y: res.convergence.map((p) => p[1]), color: "#0f766e", width: 2 }] });
  else $("#conv").innerHTML = '<p class="hint">Not enough feasible designs for a convergence curve.</p>';
  $("#cards").innerHTML = cands.map((c, k) => candCard(res, c, k)).join("");
  if (res.fit) wireVerifyUpload(jid);
}

function fitBlock(fit) {
  return `<h3>Surrogate trained on your data</h3><p class="hint">${fit.n_train} rows to train, ${fit.n_test} held out; errors below are on the held-out rows. The model is then refitted on all rows.</p>
    <table><thead><tr><th>output</th><th class="num">R²</th><th class="num">MAE</th><th class="num">mean rel. error</th><th class="num">max rel. error</th></tr></thead><tbody>${Object.entries(fit.held_out).map(([k, m]) =>
    `<tr><td>${esc(k)}</td><td class="num">${fmt(m.r2, 3)}</td><td class="num">${fmt(m.mae)}</td><td class="num">${m.mean_rel === null ? "—" : (100 * m.mean_rel).toFixed(2) + " %"}</td><td class="num">${m.max_rel === null ? "—" : (100 * m.max_rel).toFixed(1) + " %"}</td></tr>`).join("")}</tbody></table>`;
}
function roundsBlock(rounds) {
  return `<h3>Active learning</h3><table><thead><tr><th>round</th><th class="num">reference runs used</th><th class="num">verified</th><th class="num">confirmed feasible</th><th>held-out error (objectives)</th></tr></thead><tbody>${rounds.map((r) =>
    `<tr><td>${r.round + 1}</td><td class="num">${r.n_train}</td><td class="num">${r.verified}</td><td class="num">${r.confirmed_feasible}</td><td>${Object.entries(r.held_out_error).filter(([k]) => LAST.problem.objectives.some((o) => o.name === k) || LAST.problem.constraints.some((c) => c.name === k)).map(([k, e]) => `${esc(k)} ${(100 * e.mean_rel).toFixed(1)} %`).join(", ")}</td></tr>`).join("")}</tbody></table>`;
}
function verifyUploadBlock() {
  return `<div class="cand-card"><h4>Verify with your solver</h4><p class="hint">Run <b>candidates_to_run.csv</b> in your solver, add the output columns, and upload the table: each candidate is matched by its design variables and compared.</p>
    <label class="drop" id="vdrop"><input type="file" id="vfile" accept=".csv,.parquet,.json"><b>Drop solver results</b></label><div id="vout"></div></div>`;
}
function wireVerifyUpload(jid) {
  $("#vfile").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    const up = await (await fetch("/api/uploads", { method: "POST", body: fd })).json();
    const r = await fetch(`/api/runs/${jid}/verify`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ upload_id: up.id }) });
    const v = await r.json();
    if (!r.ok) { $("#vout").innerHTML = `<p class="err">${esc(v.detail)}</p>`; return; }
    $("#vout").innerHTML = `<table><thead><tr><th>#</th><th>status</th>${Object.keys(v.errors).map((k) => `<th class="num">${esc(k)} rel. error</th>`).join("")}</tr></thead><tbody>${v.rows.map((r, k) =>
      `<tr><td>${k + 1}</td><td>${vbadge(r.status)}</td>${Object.keys(v.errors).map((o) => `<td class="num">${r.rel_error && r.rel_error[o] !== undefined ? (100 * r.rel_error[o]).toFixed(2) + " %" : "—"}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  };
}

function candCard(res, c, k) {
  const fr = res.feasibility.find((f) => f.index === c.index);
  const vr = res.verification ? res.verification.rows.find((r) => r.index === c.index) : null;
  const rows = (fr ? fr.constraints : []).map((r) => {
    // utilisation of the limit: value / bound for an upper bound, bound / value for a lower one
    const lower = r.rule.includes(">=");
    const frac = r.bound && r.value ? Math.max(0, Math.min(1, lower ? r.bound / r.value : r.value / r.bound)) : 0;
    const col = r.status === "violated" ? "#b42318" : r.status === "active" ? "#9a6200" : "#0f766e";
    return `<span>${esc(r.label)}</span><div class="track" title="${esc(r.rule)}"><div class="fill" style="width:${(100 * frac).toFixed(1)}%;background:${col}"></div></div>
      <span>${fmt(r.value)} / ${fmt(r.bound)} ${esc(r.unit)} <span class="pill ${r.status === "violated" ? "bad" : r.status === "active" ? "warn" : "ok"}">${esc(r.status)}</span></span>`;
  }).join("");
  let conf = "";
  if (vr && vr.reference && Object.keys(vr.reference).length) {
    const watched = new Set([...res.problem.objectives.map((o) => o.name), ...res.problem.constraints.map((c) => c.name)]);
    const keys = Object.keys(vr.rel_error).filter((x) => x in vr.surrogate && watched.has(x));
    conf = `<table style="margin-top:8px"><thead><tr><th>output</th><th class="num">search model</th><th class="num">reference</th><th class="num">rel. error</th></tr></thead><tbody>${keys.map((x) =>
      `<tr><td>${esc(lab(x))}</td><td class="num">${fmt(vr.surrogate[x])}</td><td class="num">${fmt(vr.reference[x])}</td><td class="num">${(100 * vr.rel_error[x]).toFixed(2)} %</td></tr>`).join("")}</tbody></table>`;
  } else if (vr && vr.message) conf = `<p class="err">${esc(vr.message)}</p>`;
  return `<div class="cand-card"><h4>Candidate ${k + 1} ${vbadge(c.verification)} ${fr && fr.active.length ? `<span class="meta">active: ${fr.active.map(esc).join(", ")}</span>` : ""}</h4>
    <div class="margin">${rows}</div>${conf}</div>`;
}

function scatter(res) {
  const el = $("#scatter"), kx = $("#ax-x").value, ky = $("#ax-y").value;
  const W = Math.max(300, el.clientWidth || 560), H = 300, L = 58, R = 12, T = 10, B = 34;
  const pts = res.points.filter((p) => p[kx] !== null && p[ky] !== null && !p.r);
  const cmap = {};
  res.candidates.forEach((c) => cmap[c.index] = VCLS[c.verification] || "surrogate");
  // axes span the feasible designs (and the baseline): infeasible ones far away would squash the front
  const span = pts.filter((p) => p.f).length >= 3 ? pts.filter((p) => p.f) : pts;
  const xs = span.map((p) => p[kx]).concat(res.baseline && res.baseline[kx] !== undefined ? [res.baseline[kx]] : []);
  const ys = span.map((p) => p[ky]).concat(res.baseline && res.baseline[ky] !== undefined ? [res.baseline[ky]] : []);
  const lim = (k) => res.problem.constraints.filter((c) => c.name === k).map((c) => c.upper ?? c.lower);
  let x0 = Math.min(...xs, ...lim(kx)), x1 = Math.max(...xs, ...lim(kx)), y0 = Math.min(...ys, ...lim(ky)), y1 = Math.max(...ys, ...lim(ky));
  const px = (x1 - x0) * 0.04 || 1, py = (y1 - y0) * 0.04 || 1; x0 -= px; x1 += px; y0 -= py; y1 += py;
  const X = (v) => L + (v - x0) / (x1 - x0) * (W - L - R), Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const ticks = (a, b) => { const st = Math.pow(10, Math.floor(Math.log10((b - a) / 4))); const m = [1, 2, 5, 10].find((q) => (b - a) / (q * st) <= 6) * st; const o = []; for (let v = Math.ceil(a / m) * m; v <= b; v += m) o.push(+v.toPrecision(10)); return o; };
  let g = "";
  for (const t of ticks(y0, y1)) g += `<line x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}" stroke="#eef1f5"/><text x="${L - 6}" y="${Y(t) + 3.5}" text-anchor="end">${fmt(t)}</text>`;
  for (const t of ticks(x0, x1)) g += `<text x="${X(t)}" y="${H - B + 14}" text-anchor="middle">${fmt(t)}</text>`;
  for (const v of lim(kx)) g += `<line x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${H - B}" stroke="#b42318" stroke-dasharray="4 3"/><text x="${X(v) + 3}" y="${T + 10}" style="fill:#b42318">limit</text>`;
  for (const v of lim(ky)) g += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#b42318" stroke-dasharray="4 3"/><text x="${W - R}" y="${Y(v) - 4}" text-anchor="end" style="fill:#b42318">limit</text>`;
  const order = [...pts].sort((a, b) => (a.f - b.f) || (a.p - b.p));
  const inside = (p) => p[kx] >= x0 && p[kx] <= x1 && p[ky] >= y0 && p[ky] <= y1;
  for (const p of order.filter(inside)) {
    const col = p.p ? "#0b6bcb" : p.f ? "#93c5fd" : "#c9d1dd";
    g += `<circle cx="${X(p[kx]).toFixed(1)}" cy="${Y(p[ky]).toFixed(1)}" r="${p.p ? 3.6 : 2.4}" fill="${col}"/>`;
  }
  for (const p of pts.filter((p) => cmap[p.i] && inside(p))) g += `<circle cx="${X(p[kx])}" cy="${Y(p[ky])}" r="7" fill="none" stroke="${VCOL[cmap[p.i]]}" stroke-width="2.2"/>`;
  if (res.baseline && res.baseline[kx] !== undefined && res.baseline[ky] !== undefined) {
    const bx = X(res.baseline[kx]), by = Y(res.baseline[ky]);
    g += `<path d="M${bx} ${by - 7}L${bx + 7} ${by}L${bx} ${by + 7}L${bx - 7} ${by}Z" fill="#172033"/><text x="${bx + 9}" y="${by + 4}" style="fill:#172033;font-weight:600">baseline</text>`;
  }
  g += `<line x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" stroke="#c9d1dd"/><line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="#c9d1dd"/>`;
  g += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle" style="font-weight:600">${esc(lab(kx))}</text><text transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle" style="font-weight:600">${esc(lab(ky))}</text>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}">${g}</svg><div class="tip"></div>`;
  const tip = el.querySelector(".tip"), svg = el.querySelector("svg");
  svg.onmousemove = (e) => {
    const r = svg.getBoundingClientRect(), mx = (e.clientX - r.left) * W / r.width, my = (e.clientY - r.top) * H / r.height;
    let best = null, bd = 144;
    for (const p of pts) { const d = (X(p[kx]) - mx) ** 2 + (Y(p[ky]) - my) ** 2; if (d < bd) { bd = d; best = p; } }
    if (!best) { tip.style.display = "none"; return; }
    tip.style.display = "block"; tip.style.left = (X(best[kx]) * r.width / W + 12) + "px"; tip.style.top = (Y(best[ky]) * r.height / H - 10) + "px";
    tip.innerHTML = `design #${best.i} · ${best.p ? "Pareto" : best.f ? "feasible" : "infeasible"}${cmap[best.i] ? " · " + cmap[best.i] : ""}<br>${esc(lab(kx))}: ${fmt(best[kx])}<br>${esc(lab(ky))}: ${fmt(best[ky])}`;
  };
  svg.onmouseleave = () => tip.style.display = "none";
}

function snippets() {
  const host = location.origin;
  $("#c1").textContent = `# queue the heat-sink example and wait for it
curl -s -X POST ${host}/api/runs -H 'Content-Type: application/json' \\
  -d '{"spec": {"source": "preset", "preset": "heatsink", "objectives": [{"name": "mass_g"}, {"name": "cost"}]},
       "method": "nsga2", "budget": 600, "n_verify": 5, "verify_with": "library"}'
curl -s ${host}/api/jobs/<id>                       # status, progress, result (candidates, verification)
curl -s -O ${host}/api/runs/<id>/files/study.json   # StudyResult contract

# your data: upload a table of evaluated designs, then source "table"
curl -s -F file=@doe.csv ${host}/api/uploads
curl -s -X POST ${host}/api/runs -H 'Content-Type: application/json' -d '{"upload_id": "<upload id>",
  "spec": {"source": "table", "inputs": ["w", "h"], "outputs": ["mass", "T"],
           "objectives": [{"name": "mass"}], "constraints": [{"name": "T", "upper": 85}]}}'`;
  $("#c2").textContent = `from pinneapple_design.design_optimizer import presets, run, select_candidates, verify_candidates

problem = presets.heatsink_problem("surrogate")          # zoo MLP of the FVM engine
res = run(problem, "nsga2", budget=600, initial=problem.space.from_dicts([presets.heatsink_baseline()]))
chosen = select_candidates(res.evaluations, res.pareto(), 5)
report = verify_candidates(problem, res.evaluations, chosen, presets.HeatsinkEvaluator("fvm"))
print([r["status"] for r in report.rows], report.errors["t_source_c"])`;
}

init();
