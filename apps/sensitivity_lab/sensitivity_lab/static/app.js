"use strict";
// Engineering Sensitivity Lab UI: configure a study (model, runs elsewhere, dataset), run it, show the indices.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 3) => (v === null || v === undefined || !isFinite(v)) ? "–" : (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? Number(v).toExponential(2) : Number(v).toPrecision(d));
const DISTS = { uniform: ["low", "high"], normal: ["mean", "std"], triangular: ["low", "mode", "high"], lognormal: ["median", "gsd"], truncnormal: ["mean", "std", "low", "high"] };
let META = null, INFO = null, MODE = "model", R = null, OUT = 0;

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) { $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n)); $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`)); }
$$("#mode button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.v)));
function setMode(m) {
  MODE = m; $$("#mode button").forEach((b) => b.classList.toggle("on", b.dataset.v === m));
  $("#m-model").hidden = m !== "model"; $("#m-runs").hidden = m !== "runs"; $("#m-data").hidden = m !== "data";
  $("#paramset").hidden = m === "data"; $("#params").hidden = m === "runs"; $("#runs-extra").hidden = m !== "runs";
  $("#outset").hidden = m !== "model"; $("#budget").hidden = m === "data";
  for (const id of ["rmorris", "effects", "morris"]) $(`#${id}`).closest("label").hidden = m === "runs";
  $("#run").textContent = m === "runs" ? "Compute the indices from the results" : "Run the study";
  updateCost();
}
for (const id of ["onnx", "runsfile", "datafile"]) $(`#${id}`).addEventListener("change", (e) => { const l = e.target.closest(".drop"); l.classList.toggle("has", !!e.target.files.length); if (e.target.files.length) l.querySelector("b").textContent = e.target.files[0].name; });

async function api(url, opts) {
  const r = await fetch(url, opts);
  const ct = r.headers.get("content-type") || "";
  if (!ct.includes("json")) { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); }
  const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
  return j;
}

async function init() {
  META = await api("/api/meta");
  $("#model").innerHTML = META.builtin.map((b) => `<option value="builtin/${b}">built-in · ${b} (exact indices known)</option>`).join("")
    + META.zoo.map((z) => `<option value="zoo/${z.name}">pp.zoo · ${z.name}</option>`).join("") + `<option value="onnx">Upload an ONNX graph</option>`;
  $("#examples").innerHTML = META.examples.map((e) => `<button type="button" data-ex="${e.name}"><b>Try with example: ${esc(e.title)}</b>${esc(e.description)}</button>`).join("");
  $$("#examples button").forEach((b) => b.addEventListener("click", () => runExample(b.dataset.ex, b)));
  $("#model").addEventListener("change", () => selectModel($("#model").value));
  $("#onnxparams").addEventListener("click", () => {
    const ins = $("#onnx_in").value.split(",").map((s) => s.trim()).filter(Boolean), outs = $("#onnx_out").value.split(",").map((s) => s.trim()).filter(Boolean);
    INFO = { inputs: ins, outputs: outs.length ? outs : ["y"], space: ins.map((n) => ({ name: n, dist: "uniform", low: 0, high: 1 })), fixed: {} };
    renderParams(); renderOutputs();
  });
  for (const id of ["nsobol", "rmorris", "so", "effects", "morris"]) $(`#${id}`).addEventListener("change", updateCost);
  $("#runparams").value = "x1,uniform,-3.14159,3.14159\nx2,uniform,-3.14159,3.14159\nx3,uniform,-3.14159,3.14159";
  $("#runparams").addEventListener("input", updateCost);
  $("#design").addEventListener("click", downloadDesign);
  await selectModel("zoo/heatsink_forced_mlp");
  await fillFromConfig(await api("/api/example/heatsink/config"));
  apiSnippets();
}

async function selectModel(v) {
  $("#onnxbox").hidden = v !== "onnx";
  if (v === "onnx") { INFO = null; $("#params").innerHTML = `<div class="meta">Give the input and output names, then build the parameter table.</div>`; $("#outputs").innerHTML = ""; return; }
  INFO = await api(`/api/model/${v}`);
  renderParams(); renderOutputs();
}

function renderParams(cfg) {
  const space = {}; (cfg ? cfg.parameters : INFO.space).forEach((p) => (space[p.name] = p));
  const fixed = cfg ? (cfg.fixed || {}) : (INFO.fixed || {});
  let h = `<div class="prow phead"><span></span><span>parameter</span><span>distribution</span><span colspan="3">parameters / fixed value</span></div>`;
  for (const n of INFO.inputs) {
    const p = space[n], on = !!p, d = p ? (p.dist || "uniform") : "uniform";
    h += `<div class="prow ${on ? "" : "off"}" data-n="${esc(n)}"><input type="checkbox" class="p-on" ${on ? "checked" : ""}><span class="nm" title="${esc(n)}">${esc(n)}${INFO.units && INFO.units[n] && INFO.units[n] !== "-" ? ` <span class="u">[${esc(INFO.units[n])}]</span>` : ""}</span>
      <select class="p-dist">${Object.keys(DISTS).map((k) => `<option ${k === d ? "selected" : ""}>${k}</option>`).join("")}</select><span class="p-vals" style="display:contents"></span></div>`;
  }
  $("#params").innerHTML = h;
  $$("#params .prow[data-n]").forEach((row) => {
    const n = row.dataset.n, p = space[n];
    const draw = () => {
      const on = $(".p-on", row).checked; row.classList.toggle("off", !on); $(".p-dist", row).disabled = !on;
      if (on) { const keys = DISTS[$(".p-dist", row).value].slice(0, 3); $(".p-vals", row).innerHTML = keys.map((k) => `<input class="p-v" data-k="${k}" type="number" step="any" placeholder="${k}" title="${k}" value="${p && p[k] !== undefined ? p[k] : ""}">`).join("") + "<span></span>".repeat(3 - keys.length); }
      else { const fv = fixed[n] ?? (p ? (p.mean ?? (p.low + p.high) / 2) : ""); $(".p-vals", row).innerHTML = `<input class="p-fix" type="number" step="any" placeholder="fixed" value="${fv ?? ""}"><span></span><span></span>`; }
      updateCost();
    };
    $(".p-on", row).addEventListener("change", draw); $(".p-dist", row).addEventListener("change", draw); draw();
  });
}
function renderOutputs(sel) { $("#outputs").innerHTML = INFO.outputs.map((o) => `<label><input type="checkbox" value="${esc(o)}" ${!sel || sel.includes(o) ? "checked" : ""}> ${esc(o)}</label>`).join(""); }

function readParams() {
  const parameters = [], fixed = {};
  $$("#params .prow[data-n]").forEach((row) => {
    const n = row.dataset.n;
    if ($(".p-on", row).checked) { const p = { name: n, dist: $(".p-dist", row).value }; $$(".p-v", row).forEach((i) => { if (i.value !== "") p[i.dataset.k] = Number(i.value); }); parameters.push(p); }
    else { const v = $(".p-fix", row).value; if (v !== "") fixed[n] = Number(v); }
  });
  return { parameters, fixed };
}
function readRunParams() {
  return $("#runparams").value.trim().split(/\r?\n/).filter((l) => l.trim()).map((l) => {
    const [name, dist, ...v] = l.split(/[,;\t]/).map((s) => s.trim()); const keys = DISTS[dist] || DISTS.uniform; const p = { name, dist: DISTS[dist] ? dist : "uniform" };
    keys.forEach((k, i) => { if (v[i] !== undefined && v[i] !== "") p[k] = Number(v[i]); }); return p;
  });
}
function readConfig() {
  const budget = { n_sobol: Number($("#nsobol").value), second_order: $("#so").checked, r_morris: Number($("#rmorris").value), threshold: Number($("#threshold").value), effects: $("#effects").checked, morris: $("#morris").checked, n_grid: 15, n_background: 48 };
  if (MODE === "runs") return { parameters: readRunParams(), outputs: $("#runouts").value.split(",").map((s) => s.trim()).filter(Boolean), ...budget };
  const v = $("#model").value, { parameters, fixed } = readParams();
  const model = v === "onnx" ? { source: "onnx", input_names: INFO ? INFO.inputs : [], output_names: INFO ? INFO.outputs : [] } : { source: v.split("/")[0], name: v.split("/")[1] };
  return { model, parameters, fixed, outputs: $$("#outputs input:checked").map((i) => i.value), ...budget };
}

let costTimer = null;
function updateCost() {
  clearTimeout(costTimer);
  costTimer = setTimeout(async () => {
    if (MODE === "data") { $("#cost").textContent = ""; return; }
    try {
      const cfg = readConfig(); if (!cfg.parameters.length) { $("#cost").textContent = ""; return; }
      const c = await api("/api/cost", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(cfg) });
      $("#cost").innerHTML = MODE === "runs" ? `<b>${c.parts.sobol.toLocaleString()}</b> solver runs (N = ${c.n_base}, ${cfg.parameters.length} parameters${cfg.second_order ? ", with second order" : ""})`
        : `<b>${c.total.toLocaleString()}</b> model evaluations: Sobol ${c.parts.sobol.toLocaleString()} (N = ${c.n_base}) · Morris ${c.parts.morris.toLocaleString()} · effects ${c.parts.effects.toLocaleString()} · freeze check up to ${c.parts.verify_freeze.toLocaleString()}`;
    } catch (e) { $("#cost").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  }, 150);
}

async function fillFromConfig(cfg) {
  if (cfg.mode === "given_data") { setMode("data"); $("#dataouts").value = (cfg.outputs || []).join(", "); return; }
  setMode("model");
  const v = `${cfg.model.source}/${cfg.model.name}`; $("#model").value = v; INFO = await api(`/api/model/${v}`);
  renderParams(cfg); renderOutputs(cfg.outputs);
  $("#nsobol").value = String(cfg.n_sobol); $("#so").checked = !!cfg.second_order; $("#rmorris").value = cfg.r_morris; $("#threshold").value = cfg.threshold;
  updateCost();
}

async function busy(btn, fn) {
  const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Running…`; $("#status").textContent = "";
  try { await fn(); } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; }
}
$("#run").addEventListener("click", (e) => busy(e.currentTarget, async () => {
  const fd = new FormData();
  if (MODE === "model") {
    fd.append("config", JSON.stringify(readConfig()));
    if ($("#model").value === "onnx" && $("#onnx").files[0]) fd.append("model", $("#onnx").files[0]);
    show(await runJob("/api/study", fd));
  } else if (MODE === "runs") {
    if (!$("#runsfile").files[0]) throw new Error("Upload the results table first.");
    fd.append("spec", $("#spec").value.trim().replace(/^#\s*pinneapple\.sensitivity\.design\/1\s*/, ""));
    fd.append("runs", $("#runsfile").files[0]);
    show(await runJob("/api/runs", fd));
  } else {
    if (!$("#datafile").files[0]) throw new Error("Upload a data table first.");
    const ins = $("#datains").value.split(",").map((s) => s.trim()).filter(Boolean);
    fd.append("config", JSON.stringify({ outputs: $("#dataouts").value.split(",").map((s) => s.trim()).filter(Boolean), ...(ins.length ? { inputs: ins } : {}) }));
    fd.append("data", $("#datafile").files[0]);
    show(await runJob("/api/given-data", fd));
  }
}));
// Studies run in the shared job queue: submit, then follow /api/jobs/{id} until it is done.
async function runJob(url, body) {
  let job = await api(url, { method: "POST", body });
  const t0 = Date.now();
  while (!["done", "failed", "cancelled"].includes(job.status)) {
    $("#status").textContent = `${job.status === "queued" ? "Queued" : "Running"}${job.message ? `: ${job.message}` : ""} (${Math.round((job.progress || 0) * 100)} %, ${Math.round((Date.now() - t0) / 1000)} s)`;
    await new Promise((r) => setTimeout(r, 700));
    job = await api(`/api/jobs/${job.id}`);
  }
  $("#status").textContent = "";
  if (job.status !== "done") throw new Error(job.error ? String(job.error).split("\n").filter(Boolean).pop() : `job ${job.status}`);
  return job.result;
}
async function downloadDesign(e) {
  await busy(e.currentTarget, async () => {
    const text = await api("/api/design", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(readConfig()) });
    $("#spec").value = text.split("\n")[0];
    download("sobol_design.csv", text, "text/csv");
  });
}
async function runExample(name, btn) { await busy(btn, async () => { await fillFromConfig(await api(`/api/example/${name}/config`)); show(await api(`/api/example/${name}`)); }); }

// ------------------------------------------------------------------ results
function show(j) { R = j; OUT = 0; render(); showTab("results"); window.scrollTo(0, 0); }

function render() {
  if (R.mode === "given_data") return renderGiven();
  const s = R.sobol, outs = R.outputs, o = outs[OUT], tbl = s.tables[o];
  const frozen = (R.freeze.rows || []).filter((r) => r.decision === "freeze").map((r) => r.name);
  const incon = (R.freeze.rows || []).filter((r) => r.decision === "inconclusive").map((r) => r.name);
  const sumS1 = s.S1[OUT].reduce((a, b) => a + Math.max(0, b), 0);
  let h = `<div class="panel">${R.example ? `<div class="meta">Example · ${esc(R.example.title)}</div>` : ""}
    <div class="toolbar"><h2 style="margin:0">${R.mode === "runs" ? "Runs done elsewhere" : esc(R.model.name)}: ${R.sobol.names.length} parameters, ${outs.length} response${outs.length > 1 ? "s" : ""}</h2>
      <div class="dl"><button class="ghost" id="dlsr" title="pinneapple_data.contracts StudyResult">StudyResult</button><button class="ghost" id="dljson">Full results JSON</button><button class="ghost" id="dlcsv">Indices CSV</button></div></div>
    ${outs.length > 1 ? `<div class="outsel">${outs.map((x, i) => `<button data-i="${i}" class="${i === OUT ? "on" : ""}">${esc(x)}</button>`).join("")}</div>` : ""}
    <div class="kpis" style="margin-top:10px">
      <div class="kpi"><div class="l">Controls ${esc(o)}</div><div class="v">${esc(tbl[0].name)}</div><div class="s">total index ${fmt(tbl[0].ST)} (${fmt(tbl[0].ST_ci[0])}–${fmt(tbl[0].ST_ci[1])})</div></div>
      <div class="kpi"><div class="l">Interactions</div><div class="v">${(100 * Math.max(0, 1 - sumS1)).toFixed(0)} %</div><div class="s">of the variance: 1 − Σ first-order</div></div>
      <div class="kpi"><div class="l">Can be frozen</div><div class="v">${frozen.length} / ${s.names.length}</div><div class="s">${esc(frozen.join(", ") || "none")}${incon.length ? ` · ${incon.length} inconclusive` : ""}</div></div>
      <div class="kpi"><div class="l">${R.mode === "runs" ? "Solver runs" : "Model evaluations"}</div><div class="v">${R.n_evals.total.toLocaleString()}</div><div class="s">Sobol N = ${s.n_base}${R.usage ? ` · ${R.usage.seconds} s` : ""}</div></div>
    </div></div>`;

  if (R.exact) {
    h += `<div class="panel" style="margin-top:14px"><h3 style="margin-top:0">Check against the exact answer</h3>
      <div class="banner ${R.exact.S1_in_ci.every(Boolean) && R.exact.ST_in_ci.every(Boolean) ? "ok" : "warn"}">Largest error: first-order ${fmt(R.exact.max_abs_error_S1, 2)}, total ${fmt(R.exact.max_abs_error_ST, 2)}; exact value inside the ${Math.round(100 * s.level)} % interval for ${R.exact.S1_in_ci.filter(Boolean).length + R.exact.ST_in_ci.filter(Boolean).length} of ${2 * s.names.length} indices.</div>
      <table><tr><th>Parameter</th><th class="num">S1 estimated</th><th class="num">S1 exact</th><th class="num">ST estimated</th><th class="num">ST exact</th></tr>
      ${s.names.map((n, i) => `<tr><td>${esc(n)}</td><td class="num">${fmt(s.S1[0][i])} <span class="why">(${fmt(s.S1_ci[0][0][i])}–${fmt(s.S1_ci[1][0][i])})</span></td><td class="num">${fmt(R.exact.S1[i])}</td><td class="num">${fmt(s.ST[0][i])} <span class="why">(${fmt(s.ST_ci[0][0][i])}–${fmt(s.ST_ci[1][0][i])})</span></td><td class="num">${fmt(R.exact.ST[i])}</td></tr>`).join("")}</table></div>`;
  }

  h += `<div class="panel" style="margin-top:14px"><div class="sec"><h3>Importance ranking: ${esc(o)}</h3>
      <div class="meta">Sobol indices, ${Math.round(100 * s.level)} % bootstrap intervals (${s.n_boot} resamples). First order: the parameter alone. Total: with every interaction.</div>
      <div class="legend" style="margin:6px 0"><span><i class="s1"></i>first order S1</span><span><i class="st"></i>total ST</span><span>— threshold ${R.threshold}</span></div>
      <div class="bars">${tbl.map((r) => `<span class="nm" title="${esc(r.name)}">${esc(r.name)}${frozen.includes(r.name) ? `<span class="frz">freeze</span>` : ""}</span>${barSvg(r)}<span class="num">${fmt(r.S1, 2)} / ${fmt(r.ST, 2)}</span>`).join("")}</div></div></div>
    <div class="panel" style="margin-top:14px"><div class="sec">${s.S2 ? heatmap(s) : `<h3>Interactions</h3><div class="meta">Total minus first order per parameter (second-order indices not requested).</div>
      <table>${tbl.map((r) => `<tr><td>${esc(r.name)}</td><td class="num">${fmt(r.interaction, 2)}</td></tr>`).join("")}</table>`}</div></div>`;

  if (R.morris) h += `<div class="panel" style="margin-top:14px"><div class="sec"><h3>Morris screening: ${esc(o)}</h3><div class="meta">${R.morris.r} trajectories, ${R.morris.n_evals.toLocaleString()} evaluations. μ*: mean absolute elementary effect (importance), σ: its spread (nonlinearity or interactions). Effects per unit of each parameter's normalised range.</div><div class="scat" id="scat"></div></div></div>`;

  if (R.effects) h += `<div class="panel" style="margin-top:14px"><div class="sec"><h3>Effect curves: ${esc(o)}</h3><div class="meta">Thick: main effect (partial dependence, averaged over ${R.effects.n_background} samples of the other parameters). Thin: individual curves (ICE); when they are not parallel the parameter interacts with others.</div>
    <div class="charts">${R.effects.curves.map((c, i) => `<div class="ec"><b>${esc(c.name)}</b> <span class="why">range of effect ${fmt(c.pdp_range[OUT])} · ICE spread ${fmt(c.ice_spread[OUT])}</span><div id="ec${i}"></div></div>`).join("")}</div></div></div>`;

  if (R.derivatives) {
    const d = R.derivatives, names = s.names;
    h += `<div class="panel" style="margin-top:14px"><div class="sec"><h3>Local derivatives at the nominal point</h3><div class="meta">${esc(d.method)}. Scaled: derivative × range width (linearised change across the range). Elasticity: % change of ${esc(o)} per % change of the parameter.</div>
      <div class="tblwrap"><table><tr><th>Parameter</th><th class="num">nominal</th><th class="num">∂${esc(o)}/∂x</th><th class="num">scaled</th><th class="num">elasticity</th></tr>
      ${names.map((n, i) => `<tr><td>${esc(n)}</td><td class="num">${fmt(d.x0[n], 4)}</td><td class="num">${fmt(d.jacobian[OUT][i])}</td><td class="num">${fmt(d.scaled[OUT][i])}</td><td class="num">${fmt(d.elasticity[OUT][i])}</td></tr>`).join("")}</table></div>
      <div class="meta">At the nominal point: ${esc(o)} = ${fmt(d.y0[o], 4)}.</div></div></div>`;
  }

  const fr = R.freeze, ck = fr.check;
  h += `<div class="panel" style="margin-top:14px"><div class="sec"><h3>Parameters you can freeze</h3>
    <div class="meta">Rule: freeze only when the upper bound of the total index is below ${fr.threshold} for every response. Below the threshold with a wider interval: inconclusive (more samples), never freeze.</div>
    ${ck ? `<div class="banner ${ck.max_change_sd.every((v) => v < 0.25) ? "ok" : "warn"}" style="font-weight:500">Checked by re-running the model on ${ck.n} samples with ${esc(ck.frozen.join(", "))} fixed at ${esc(Object.entries(ck.nominal).map(([k, v]) => `${k} = ${fmt(v, 4)}`).join(", "))}: output variance ratio ${ck.variance_ratio.map((v) => v.toFixed(3)).join(" / ")}, largest change ${ck.max_change_sd.map((v) => v.toFixed(3)).join(" / ")} standard deviations (95th percentile ${ck.p95_change_sd.map((v) => v.toFixed(3)).join(" / ")}).</div>` : fr.note ? `<div class="warnings"><div>${esc(fr.note)}</div></div>` : ""}
    <div class="tblwrap"><table><tr><th>Parameter</th><th>decision</th><th class="num">largest ST</th><th class="num">largest upper bound</th><th>evidence per response</th></tr>
    ${fr.rows.map((r) => `<tr><td>${esc(r.name)}</td><td><span class="dec ${r.decision}">${r.decision}</span></td><td class="num">${fmt(r.max_ST)}</td><td class="num">${fmt(r.max_ST_upper)}</td><td class="why">${r.evidence.map((e) => `${esc(e.output)}: ST ${fmt(e.ST)} (${fmt(e.ST_ci[0])}–${fmt(e.ST_ci[1])})`).join(" · ")}${r.morris_mu_star ? ` · Morris μ* ${r.morris_mu_star.map((v) => fmt(v)).join(" / ")}` : ""}</td></tr>`).join("")}</table></div></div></div>`;
  h += `<div class="panel" style="margin-top:14px">${window.renderScope ? renderScope(R.scope) : ""}</div>`;
  $("#results").innerHTML = h;

  $$(".outsel button").forEach((b) => b.addEventListener("click", () => { OUT = +b.dataset.i; render(); }));
  if (R.morris) scatter($("#scat"), R.morris, OUT);
  if (R.effects) R.effects.curves.forEach((c, i) => effect($(`#ec${i}`), c, OUT));
  $("#dljson").onclick = () => download(`sensitivity_${R.model ? R.model.name : "runs"}.json`, JSON.stringify(R, null, 1), "application/json");
  $("#dlsr").onclick = () => download(`sensitivity_${R.model ? R.model.name : "runs"}.study.json`, JSON.stringify(R.study_result, null, 1), "application/json");
  $("#dlcsv").onclick = () => {
    const lines = ["output,parameter,S1,S1_lo,S1_hi,ST,ST_lo,ST_hi,interaction,decision"];
    const dec = Object.fromEntries(R.freeze.rows.map((r) => [r.name, r.decision]));
    R.outputs.forEach((out) => R.sobol.tables[out].forEach((r) => lines.push([out, r.name, r.S1, r.S1_ci[0], r.S1_ci[1], r.ST, r.ST_ci[0], r.ST_ci[1], r.interaction, dec[r.name]].join(","))));
    download("sobol_indices.csv", lines.join("\n"), "text/csv");
  };
}

function barSvg(r) {
  const X = (v) => 2 + Math.max(0, Math.min(1, v)) * 96, th = Math.max(0, Math.min(1, R.threshold));
  return `<svg viewBox="0 0 100 26" preserveAspectRatio="none"><rect x="0" y="0" width="100" height="26" fill="#f8fafc"/>
    <rect x="2" y="4" width="${X(r.S1) - 2}" height="7" fill="var(--s1)"/><rect x="2" y="14" width="${X(r.ST) - 2}" height="7" fill="var(--st)"/>
    <line x1="${X(r.S1_ci[0])}" x2="${X(r.S1_ci[1])}" y1="7.5" y2="7.5" stroke="#1f2937" stroke-width="0.8"/><line x1="${X(r.ST_ci[0])}" x2="${X(r.ST_ci[1])}" y1="17.5" y2="17.5" stroke="#1f2937" stroke-width="0.8"/>
    ${R.threshold ? `<line x1="${X(th)}" x2="${X(th)}" y1="0" y2="26" stroke="#1a7f4b" stroke-dasharray="1.5 1.5" stroke-width="0.6"/>` : ""}</svg>`;
}

function heatmap(s) {
  const n = s.names, M = s.S2[OUT], mx = Math.max(0.01, ...M.flat().map((v) => Math.abs(v)));
  return `<h3>Second-order interactions</h3><div class="meta">S<sub>ij</sub>: share of variance due to the pair alone (Saltelli 2002); small negative values are estimation noise.</div>
    <div class="tblwrap"><table class="heat"><tr><th></th>${n.map((x) => `<th title="${esc(x)}">${esc(x.length > 9 ? x.slice(0, 8) + "…" : x)}</th>`).join("")}</tr>
    ${n.map((a, i) => `<tr><th style="text-align:left">${esc(a)}</th>${n.map((b, j) => { if (i === j) return `<td style="background:#f1f5f9">–</td>`; const v = M[i][j], t = Math.max(0, v) / mx; return `<td style="background:rgba(180,83,9,${(0.08 + 0.85 * t).toFixed(2)});color:${t > 0.55 ? "#fff" : "#1f2937"}" title="${esc(a)} × ${esc(b)}: ${fmt(v)} (${fmt(s.S2_ci[0][OUT][i][j])}–${fmt(s.S2_ci[1][OUT][i][j])})">${fmt(v, 2)}</td>`; }).join("")}</tr>`).join("")}</table></div>`;
}

function scatter(el, m, k) {
  const W = Math.max(320, el.clientWidth || 700), H = 300, L = 54, Rr = 20, T = 14, B = 34;
  const xs = m.mu_star[k], ys = m.sigma[k], mx = Math.max(...xs) * 1.1 || 1, my = Math.max(...ys) * 1.1 || 1;
  const X = (v) => L + v / mx * (W - L - Rr), Y = (v) => T + (1 - v / my) * (H - T - B);
  let g = `<line x1="${L}" x2="${W - Rr}" y1="${H - B}" y2="${H - B}" stroke="#c9d1dd"/><line x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" stroke="#c9d1dd"/>`;
  g += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(Math.min(mx, my))}" y2="${Y(Math.min(mx, my))}" stroke="#cbd5e1" stroke-dasharray="4 4"/>`;
  for (let i = 0; i <= 4; i++) { g += `<text x="${X(mx * i / 4)}" y="${H - B + 15}" text-anchor="middle">${fmt(mx * i / 4, 2)}</text><text x="${L - 6}" y="${Y(my * i / 4) + 3}" text-anchor="end">${fmt(my * i / 4, 2)}</text>`; }
  g += `<text x="${(L + W - Rr) / 2}" y="${H - 4}" text-anchor="middle">μ* (importance)</text><text x="12" y="${(T + H - B) / 2}" transform="rotate(-90 12 ${(T + H - B) / 2})" text-anchor="middle">σ (nonlinear / interacting)</text>`;
  m.names.forEach((n, i) => { const lo = m.mu_star_ci[0][k][i], hi = m.mu_star_ci[1][k][i];
    g += `<line x1="${X(lo)}" x2="${X(hi)}" y1="${Y(ys[i])}" y2="${Y(ys[i])}" stroke="#d97706" stroke-width="1.2"/><circle cx="${X(xs[i])}" cy="${Y(ys[i])}" r="4.5" fill="#b45309"><title>${esc(n)}: μ* ${fmt(xs[i])} (${fmt(lo)}–${fmt(hi)}), σ ${fmt(ys[i])}</title></circle><text x="${X(xs[i]) + 7}" y="${Y(ys[i]) - 5}" style="fill:#1f2937">${esc(n)}</text>`; });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}">${g}</svg>`;
}

function effect(el, c, k) {
  const W = Math.max(240, el.clientWidth || 280), H = 150, L = 40, Rr = 8, T = 8, B = 22, gx = c.grid, pdp = c.pdp[k], ice = c.ice[k];
  const all = ice.flat().concat(pdp), lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
  const X = (v) => L + (v - gx[0]) / ((gx[gx.length - 1] - gx[0]) || 1) * (W - L - Rr), Y = (v) => T + (1 - (v - lo) / span) * (H - T - B);
  const path = (ys) => ys.map((y, i) => `${i ? "L" : "M"}${X(gx[i]).toFixed(1)},${Y(y).toFixed(1)}`).join("");
  let g = `<line x1="${L}" x2="${W - Rr}" y1="${H - B}" y2="${H - B}" stroke="#c9d1dd"/>`;
  g += ice.map((ys) => `<path d="${path(ys)}" fill="none" stroke="#f59e0b" stroke-opacity=".35" stroke-width="1"/>`).join("");
  g += `<path d="${path(pdp)}" fill="none" stroke="#7c2d12" stroke-width="2.4"/>`;
  g += `<text x="${L - 4}" y="${T + 8}" text-anchor="end">${fmt(hi)}</text><text x="${L - 4}" y="${H - B}" text-anchor="end">${fmt(lo)}</text>`;
  g += `<text x="${L}" y="${H - 6}">${fmt(gx[0])}</text><text x="${W - Rr}" y="${H - 6}" text-anchor="end">${fmt(gx[gx.length - 1])}</text>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}">${g}</svg>`;
}

function renderGiven() {
  R.threshold = null;
  const outs = R.outputs, o = outs[OUT], g = R.given_data[o], mc = R.max_input_correlation;
  const rows = g.names.map((n, i) => ({ n, s: g.S1[i], lo: g.S1_ci[0][i], hi: g.S1_ci[1][i] })).sort((a, b) => b.s - a.s);
  const warn = Math.abs(mc ? mc.r : 0) > 0.3 || g.sum_S1 > 1.05;
  let h = `<div class="panel">${R.example ? `<div class="meta">Example · ${esc(R.example.title)}</div>` : ""}
    <div class="toolbar"><h2 style="margin:0">From data: ${esc(R.source)}</h2><div class="dl"><button class="ghost" id="dlsr" title="pinneapple_data.contracts StudyResult">StudyResult</button><button class="ghost" id="dljson">Full results JSON</button></div></div>
    ${outs.length > 1 ? `<div class="outsel">${outs.map((x, i) => `<button data-i="${i}" class="${i === OUT ? "on" : ""}">${esc(x)}</button>`).join("")}</div>` : ""}
    <div class="kpis" style="margin-top:10px">
      <div class="kpi"><div class="l">Rows</div><div class="v">${R.n_rows.toLocaleString()}</div><div class="s">${g.n_bins} bins per input${g.dropped_rows ? ` · ${g.dropped_rows} rows dropped (NaN)` : ""}</div></div>
      <div class="kpi"><div class="l">Strongest input</div><div class="v">${esc(rows[0].n)}</div><div class="s">S1 ${fmt(rows[0].s)} (${fmt(rows[0].lo)}–${fmt(rows[0].hi)})</div></div>
      <div class="kpi"><div class="l">Σ first order</div><div class="v">${fmt(g.sum_S1)}</div><div class="s">${g.sum_S1 > 1.05 ? "> 1: inputs are not independent" : "≤ 1: the rest is interactions"}</div></div>
      <div class="kpi"><div class="l">Largest input correlation</div><div class="v">${mc ? fmt(mc.r, 2) : "–"}</div><div class="s">${mc ? esc(mc.pair.join(" × ")) : ""}</div></div>
    </div>
    ${warn ? `<div class="banner warn" style="margin-top:12px">The inputs of this dataset are correlated (largest |r| = ${fmt(Math.abs(mc.r), 2)}, ${esc(mc.pair.join(" × "))}); the first-order indices add up to ${fmt(g.sum_S1, 3)}. Each index still measures how much of the variance follows that input, but part of it may belong to the inputs it is correlated with. For a variance decomposition, run a designed study on a model or in your solver.</div>` : ""}
  </div>
  <div class="panel" style="margin-top:14px"><div class="sec"><h3>First-order indices from the data: ${esc(o)}</h3><div class="meta">${esc(g.method)}; ${Math.round(100 * g.level)} % bootstrap intervals. Totals and interactions need a designed experiment.</div>
    <div class="bars" style="margin-top:8px">${rows.map((r) => `<span class="nm">${esc(r.n)}</span>${barSvg({ S1: r.s, ST: 0, S1_ci: [r.lo, r.hi], ST_ci: [0, 0] })}<span class="num">${fmt(r.s, 2)}</span>`).join("")}</div></div></div>
  <div class="panel" style="margin-top:14px">${window.renderScope ? renderScope(R.scope) : ""}</div>`;
  $("#results").innerHTML = h;
  $$(".outsel button").forEach((b) => b.addEventListener("click", () => { OUT = +b.dataset.i; render(); }));
  $("#dljson").onclick = () => download("sensitivity_given_data.json", JSON.stringify(R, null, 1), "application/json");
  $("#dlsr").onclick = () => download("sensitivity_given_data.study.json", JSON.stringify(R.study_result, null, 1), "application/json");
}

function download(name, text, type) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }

function apiSnippets() {
  const host = location.origin;
  $("#c1").textContent = `# a model study (the heat-sink example's configuration)
curl -s ${host}/api/example/heatsink/config > study.json
curl -s -X POST "${host}/api/study?wait=300" -F "config=<study.json" > job.json   # a job; .result when done
curl -s ${host}/api/jobs/<job id>                                     # status, progress, result (StudyResult in .result.study_result)

# runs elsewhere: get the design, run it in OpenFOAM / CalculiX, upload run_id + responses
curl -s -X POST ${host}/api/design -H "content-type: application/json" \\
  -d '{"parameters": [{"name": "load", "dist": "normal", "mean": 1000, "std": 50},
                      {"name": "thickness", "dist": "uniform", "low": 4, "high": 6}], "n_sobol": 256}' > design.csv
head -1 design.csv | cut -c3- | sed 's/^pinneapple.sensitivity.design\\/1 //' > spec.json
curl -s -X POST "${host}/api/runs?wait=300" -F "spec=<spec.json" -F "runs=@results.csv"

# a dataset (first-order indices from data)
curl -s -X POST "${host}/api/given-data?wait=300" -F 'config={"outputs": ["y"]}' -F "data=@experiments.csv"

# shared contracts: a ParameterSpace and a ModelRef in, a StudyResult out
curl -s -X POST "${host}/api/study?wait=300" -F 'config={"model_ref": {"kind": "ModelRef", "name": "hs", "uri": "zoo:heatsink_forced_mlp"}, "parameter_space": <ParameterSpace JSON>}'`;
  $("#c2").textContent = `import pinneapple as pp
from pinneapple_analysis.study_models import StudyModel

model = StudyModel.from_zoo("heatsink_forced_mlp")
space = [{"name": "air_velocity_m_s", "dist": "normal", "mean": 2.5, "std": 0.3},
         {"name": "fin_height_mm", "dist": "uniform", "low": 30, "high": 40}]
fixed = {n: v for n, v in zip(model.input_names, model.reference("test")[0][0]) if n not in ("air_velocity_m_s", "fin_height_mm")}
out = pp.sensitivity.sensitivity_study(model, space, fixed=fixed, n_sobol=1024)
print(out["sobol"]["ranking"], [(r["name"], r["decision"]) for r in out["freeze"]["rows"]])`;
}

init().catch((e) => { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; });
