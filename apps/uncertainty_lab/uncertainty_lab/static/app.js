"use strict";
// Engineering Uncertainty Lab UI: configure a study, run it, show intervals, coverage, routing and propagation.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 4) => (v === null || v === undefined || !isFinite(v)) ? "–" : (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? Number(v).toExponential(2) : Number(v).toPrecision(d));
const pct = (v, d = 1) => (v === null || v === undefined) ? "–" : `${(100 * v).toFixed(d)} %`;
const DISTS = { normal: ["mean", "std"], uniform: ["low", "high"], triangular: ["low", "mode", "high"], lognormal: ["median", "gsd"], truncnormal: ["mean", "std", "low", "high"] };
let META = null, MODEL = null, R = null, SCALE = "relative";

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) { $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n)); $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`)); }
$$("#scale button").forEach((b) => b.addEventListener("click", () => setScale(b.dataset.v)));
function setScale(v) { SCALE = v; $$("#scale button").forEach((b) => b.classList.toggle("on", b.dataset.v === v)); }
for (const id of ["onnx", "reference", "casefile"]) $(`#${id}`).addEventListener("change", (e) => { const l = e.target.closest(".drop"); l.classList.toggle("has", !!e.target.files.length); if (e.target.files.length) l.querySelector("b").textContent = e.target.files[0].name; });

async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
  return j;
}

async function init() {
  META = await api("/api/meta");
  $("#model").innerHTML = META.zoo.map((z) => `<option value="${z.name}">pp.zoo · ${z.name}</option>`).join("") + `<option value="__onnx">Upload an ONNX graph</option>`;
  $("#examples").innerHTML = META.examples.map((e) => `<button type="button" data-ex="${e.name}"><b>Try with example: ${esc(e.title)}</b>${esc(e.description)}</button>`).join("");
  $$("#examples button").forEach((b) => b.addEventListener("click", () => runExample(b.dataset.ex, b)));
  $("#model").addEventListener("change", () => selectModel($("#model").value));
  $("#addunc").addEventListener("click", () => addUnc());
  await selectModel(META.zoo[0].name);
  await fillFromConfig(await api("/api/example/heatsink/config"));
  apiSnippets();
}

async function selectModel(name) {
  $("#onnxbox").hidden = name !== "__onnx";
  if (name === "__onnx") { MODEL = null; $("#output").innerHTML = `<option value="">(from the graph)</option>`; $("#modelcard").innerHTML = ""; $("#hfnote").textContent = "No high-fidelity solver for uploaded models: the routed cases are listed for your solver."; return; }
  MODEL = await api(`/api/model/${name}`);
  $("#output").innerHTML = MODEL.outputs.map((o) => `<option>${o}</option>`).join("");
  $("#hfnote").textContent = MODEL.high_fidelity ? `Solver: ${MODEL.high_fidelity}.` : "No high-fidelity solver in the library for this model: routed cases are listed for your solver.";
  $("#hf").disabled = !MODEL.high_fidelity; if (!MODEL.high_fidelity) $("#hf").value = "none";
  if (!$("#cases").value.trim() || $("#cases").dataset.model !== name) { $("#cases").value = toCsv([{ name: "nominal", ...MODEL.nominal }], MODEL.inputs); $("#cases").dataset.model = name; }
  const vm = MODEL.metadata.validation_metrics || {};
  $("#modelcard").innerHTML = `<div class="mc"><h3>${esc(MODEL.name)} <span class="meta">v${esc(MODEL.metadata.version || "")}</span></h3>
    <div class="meta">${esc(MODEL.description)}</div>
    <dl><dt>Inputs</dt><dd>${MODEL.inputs.map((n) => `${esc(n)}${MODEL.units[n] ? ` [${esc(MODEL.units[n])}]` : ""}`).join(", ")}</dd>
    <dt>Outputs</dt><dd>${MODEL.outputs.map(esc).join(", ")}</dd>
    <dt>Reference data</dt><dd>${esc(MODEL.reference_dataset)}</dd>
    <dt>Card metrics</dt><dd>${Object.entries(vm).slice(0, 6).map(([k, v]) => `${esc(k)} ${fmt(v, 3)}`).join(" · ") || "–"}</dd>
    <dt>Known limits</dt><dd>${(MODEL.metadata.known_limitations || []).map(esc).join(" ") || "–"}</dd></dl></div>`;
}

// CSV with quoted fields (case names can hold commas), separated by commas, semicolons or tabs
const csvCell = (v) => { const t = String(v ?? ""); return /[",;\t\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t; };
function toCsv(rows, cols) { const c = ["name", ...cols]; return [c.join(","), ...rows.map((r) => c.map((k) => csvCell(r[k])).join(","))].join("\n"); }
function splitCsv(line) {
  const out = []; let cur = "", q = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (q) { if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; } else if (ch === '"') q = false; else cur += ch; }
    else if (ch === '"') q = true;
    else if (ch === "," || ch === ";" || ch === "\t") { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur); return out.map((s) => s.trim());
}
function fromCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter((l) => l.trim()); if (lines.length < 2) return [];
  const head = splitCsv(lines[0]);
  return lines.slice(1).map((l, k) => {
    const v = splitCsv(l), o = {};
    head.forEach((h, i) => { const s = v[i] ?? ""; o[h] = h === "name" ? s : (s === "" ? NaN : Number(s)); });
    const bad = head.filter((h) => h !== "name" && !Number.isFinite(o[h]));
    if (bad.length) throw new Error(`Case ${k + 1}${o.name ? ` (${o.name})` : ""}: no number for ${bad.join(", ")}`);
    return o;
  });
}

function addUnc(u = {}) {
  const row = document.createElement("div"); row.className = "unc-row";
  const inputs = MODEL ? MODEL.inputs : [];
  row.innerHTML = `<label class="ph">input<select class="u-name">${inputs.map((n) => `<option ${n === u.name ? "selected" : ""}>${n}</option>`).join("")}</select></label>
    <label class="ph">distribution<select class="u-dist">${Object.keys(DISTS).map((d) => `<option ${d === (u.dist || "normal") ? "selected" : ""}>${d}</option>`).join("")}</select></label>
    <span class="u-params" style="display:contents"></span><button type="button" title="remove">×</button>`;
  const params = () => {
    const d = $(".u-dist", row).value, keys = DISTS[d].slice(0, 3);
    $(".u-params", row).innerHTML = keys.map((k) => `<label class="ph">${k}<input class="u-p" data-k="${k}" type="number" step="any" value="${u[k] ?? ""}"></label>`).join("") + (keys.length < 3 ? "<span></span>".repeat(3 - keys.length) : "");
  };
  $(".u-dist", row).addEventListener("change", params); params();
  $("button", row).addEventListener("click", () => row.remove());
  $("#unc").appendChild(row);
}
function readUnc() {
  return $$("#unc .unc-row").map((r) => { const o = { name: $(".u-name", r).value, dist: $(".u-dist", r).value }; $$(".u-p", r).forEach((i) => { if (i.value !== "") o[i.dataset.k] = Number(i.value); }); return o; });
}

async function fillFromConfig(cfg) {
  if (cfg.model.source === "zoo") { $("#model").value = cfg.model.name; await selectModel(cfg.model.name); }
  $("#output").value = cfg.output; $("#alpha").value = String(cfg.alpha); setScale(cfg.scale);
  $("#lim_lo").value = cfg.limits?.[0] ?? ""; $("#lim_hi").value = cfg.limits?.[1] ?? "";
  $("#tol_pct").value = cfg.tolerance_pct ?? ""; $("#hf").value = cfg.high_fidelity || "none";
  $("#cases").value = toCsv(cfg.cases || [], MODEL ? MODEL.inputs : []); $("#cases").dataset.model = cfg.model.name;
  $("#unc").innerHTML = ""; (cfg.uncertain || []).forEach(addUnc);
  $("#nprop").value = cfg.n_propagation || 4096; $("#pmethod").value = cfg.propagation_method || "lhs";
  $("#cases").dataset.nominal = JSON.stringify(cfg.nominal || {});
}

function readConfig() {
  const onnx = $("#model").value === "__onnx";
  const split = (s) => s.split(",").map((x) => x.trim()).filter(Boolean);
  const cases = fromCsv($("#cases").value);
  let nominal = {};
  try { nominal = JSON.parse($("#cases").dataset.nominal || "{}"); } catch (e) { nominal = {}; }
  // the nominal point of the propagation: a case named "nominal", else the example's, else the data median
  const nomCase = cases.find((c) => /^nominal/i.test(c.name || ""));
  if (MODEL) { if (!Object.keys(nominal).length) nominal = { ...MODEL.nominal }; if (nomCase) for (const k of MODEL.inputs) if (isFinite(nomCase[k])) nominal[k] = nomCase[k]; }
  const lo = $("#lim_lo").value, hi = $("#lim_hi").value;
  return {
    model: onnx ? { source: "onnx", input_names: split($("#onnx_in").value), output_names: split($("#onnx_out").value) } : { source: "zoo", name: $("#model").value },
    output: $("#output").value || undefined, alpha: Number($("#alpha").value), scale: SCALE,
    limits: [lo === "" ? null : Number(lo), hi === "" ? null : Number(hi)],
    tolerance_pct: $("#tol_pct").value === "" ? null : Number($("#tol_pct").value),
    high_fidelity: $("#hf").value, cases, nominal, uncertain: readUnc(),
    n_propagation: Number($("#nprop").value), propagation_method: $("#pmethod").value,
  };
}

async function busy(btn, fn) {
  const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Running…`; $("#status").textContent = "";
  try { await fn(); } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; }
}
$("#run").addEventListener("click", (e) => busy(e.currentTarget, async () => {
  const fd = new FormData();
  fd.append("config", JSON.stringify(readConfig()));
  if ($("#model").value === "__onnx" && $("#onnx").files[0]) fd.append("model", $("#onnx").files[0]);
  if ($("#reference").files[0]) fd.append("reference", $("#reference").files[0]);
  if ($("#casefile").files[0]) fd.append("cases", $("#casefile").files[0]);
  show(await runJob("/api/study", fd));
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
async function runExample(name, btn) {
  await busy(btn, async () => { await fillFromConfig(await api(`/api/example/${name}/config`)); show(await api(`/api/example/${name}`)); });
}

// ------------------------------------------------------------------ results
function show(j) { R = j; render(); showTab("results"); window.scrollTo(0, 0); }

function render() {
  const unit = R.unit && R.unit !== "-" ? R.unit : "";
  const c = R.conformal, u = unit ? ` ${unit}` : "", cs = R.cases, p = R.propagation;
  const ok = c.meets_target;
  const verdict = ok ? `Measured coverage ${pct(c.coverage)} (95 % CI ${pct(c.coverage_ci[0])}–${pct(c.coverage_ci[1])}) is consistent with the ${pct(c.target_coverage, 0)} target on ${c.n_test} held-out cases.`
    : `Measured coverage ${pct(c.coverage)} (95 % CI ${pct(c.coverage_ci[0])}–${pct(c.coverage_ci[1])}) is below the ${pct(c.target_coverage, 0)} target: the calibration data do not represent the test data.`;
  const width = c.normalized && R.scale === "relative" ? `± ${(100 * c.q).toFixed(2)} % of the prediction` : c.normalized ? `± ${fmt(c.q, 3)} × ensemble std` : `± ${fmt(c.q)}${u}`;
  let h = `<div class="panel">
    ${R.example ? `<div class="meta">Example · ${esc(R.example.title)}</div>` : ""}
    <div class="toolbar"><h2 style="margin:0">${esc(R.model.name)} → ${esc(R.output)}</h2>
      <div class="dl"><button class="ghost" id="dlsr" title="pinneapple_data.contracts StudyResult">StudyResult</button><button class="ghost" id="dljson">Full results JSON</button>${cs ? `<button class="ghost" id="dlcsv">Cases CSV</button>` : ""}</div></div>
    <div class="banner ${ok ? "ok" : "bad"}">${esc(verdict)}</div>
    <div class="kpis">
      <div class="kpi"><div class="l">Interval</div><div class="v">${pct(c.target_coverage, 0)}</div><div class="s">${esc(width)}</div></div>
      <div class="kpi"><div class="l">Measured coverage</div><div class="v">${pct(c.coverage)}</div><div class="s">${c.n_test} held-out · CI ${pct(c.coverage_ci[0])}–${pct(c.coverage_ci[1])}</div></div>
      <div class="kpi"><div class="l">Mean interval width</div><div class="v">${fmt(c.mean_width, 3)}</div><div class="s">${esc(unit || "output units")}</div></div>
      <div class="kpi"><div class="l">Held-out error</div><div class="v">${R.accuracy.mean_abs_pct !== undefined ? `${R.accuracy.mean_abs_pct.toFixed(2)} %` : fmt(R.accuracy.mae)}</div><div class="s">${R.accuracy.mean_abs_pct !== undefined ? `mean |error| · p95 ${R.accuracy.p95_abs_pct.toFixed(2)} %` : "mean |error|"}</div></div>
      ${R.ensemble ? `<div class="kpi"><div class="l">Raw ensemble interval</div><div class="v">${pct(R.ensemble.coverage)}</div><div class="s">coverage at ${pct(R.ensemble.nominal, 0)} nominal</div></div>` : ""}
      ${cs ? `<div class="kpi"><div class="l">Cases to the solver</div><div class="v">${cs.summary.high_fidelity} / ${cs.summary.n}</div><div class="s">${esc(Object.entries(cs.summary.reasons).map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`).join(" · ") || "none needed")}</div></div>` : ""}
    </div>
    <div class="meta" style="margin-top:8px">${esc(R.reference_note)} · ${R.usage.model_evaluations.toLocaleString()} model evaluations${R.usage.high_fidelity_runs ? `, ${R.usage.high_fidelity_runs} high-fidelity runs` : ""} in ${R.usage.seconds} s</div>
  </div>`;

  h += `<div class="grid2" style="margin-top:14px"><div class="panel"><div class="sec" style="margin-top:0"><h3>Reliability: nominal vs measured coverage</h3>
      <div class="meta">Same calibration data, every level; the band is the 95 % confidence interval on the held-out measurement.</div><div id="relchart"></div></div></div>
    <div class="panel"><div class="sec" style="margin-top:0"><h3>Where the interval holds</h3>
      <div class="meta">Coverage of the held-out cases by predicted ${esc(R.output)} (equal-count bins) and by applicability gate.</div>
      <table><tr><th>Group</th><th class="num">n</th><th class="num">coverage</th><th class="num">95 % CI</th><th class="num">mean width</th></tr>
      ${c.groups.map((g) => `<tr><td>${esc(g.group)}</td><td class="num">${g.n}</td><td class="num">${pct(g.coverage)}</td><td class="num">${pct(g.coverage_ci[0], 0)}–${pct(g.coverage_ci[1], 0)}</td><td class="num">${fmt(g.mean_width, 3)}</td></tr>`).join("")}
      ${R.test_coverage_by_gate.map((g) => `<tr><td><span class="st ${g.status}">${g.status}</span></td><td class="num">${g.n}</td><td class="num">${pct(g.coverage)}</td><td class="num">${pct(g.coverage_ci[0], 0)}–${pct(g.coverage_ci[1], 0)}</td><td></td></tr>`).join("")}
      </table>${R.ensemble ? `<div class="warnings"><div>The model's own ensemble interval (${esc(R.ensemble.interval)}) covers ${pct(R.ensemble.coverage)} of the held-out cases (CI ${pct(R.ensemble.coverage_ci[0])}–${pct(R.ensemble.coverage_ci[1])}) for a ${pct(R.ensemble.nominal, 0)} nominal level; the conformal interval uses the same spread, rescaled on calibration data.</div></div>` : ""}</div></div></div>`;

  if (cs) {
    const hf = cs.high_fidelity;
    h += `<div class="panel" style="margin-top:14px"><div class="sec" style="margin-top:0"><h3>Cases: surrogate or high-fidelity solver?</h3>
      <div class="meta">Rule: to the solver when outside the validity range, where reference data are sparse (5-NN distance above the 99th percentile of the data's own), when the interval is wider than the tolerance${cs.tolerance_pct ? ` (${cs.tolerance_pct} % of the prediction)` : cs.tolerance ? ` (${fmt(cs.tolerance)}${u})` : ""}, or when it contains a limit${cs.limits ? ` (${cs.limits.map((v, i) => v === null ? "" : `${i ? "≤" : "≥"} ${v}`).filter(Boolean).join(", ")})` : ""}.</div>
      ${hf ? `<div class="banner ${hf.by_route.surrogate && hf.by_route.surrogate.in_interval < 1 ? "warn" : "ok"}" style="font-weight:500">${hf.n_run} case${hf.n_run === 1 ? "" : "s"} also solved by the high-fidelity solver (${hf.scope === "all" ? "all cases" : "routed cases"}). ${Object.entries(hf.by_route).map(([k, g]) => `${k === "surrogate" ? "Kept on the surrogate" : "Routed to the solver"}: ${g.n}, mean |error| ${fmt(g.mean_abs_error, 3)}${u}, max ${fmt(g.max_abs_error, 3)}, solver value inside the interval ${pct(g.in_interval, 0)}`).join(". ")}.</div>` : ""}
      <div class="tblwrap"><table><tr><th>Case</th><th class="num">prediction</th><th class="num">interval</th><th>applicability</th><th>route</th><th>why</th>${hf ? `<th class="num">solver</th><th>in interval</th>` : ""}<th>verdict</th></tr>
      ${cs.rows.map((r) => `<tr><td><b>${esc(r.name)}</b></td><td class="num">${fmt(r.prediction)}</td><td class="num nw">${fmt(r.lower)} – ${fmt(r.upper)}</td>
        <td><span class="st ${r.status}">${r.status}</span>${r.out_of_range.length ? `<div class="why">${esc(r.out_of_range.join(", "))}</div>` : `<div class="why">density ${fmt(r.density_ratio, 2)} · ${r.neighbours_within} neighbours</div>`}</td>
        <td><span class="route ${r.route}">${r.route === "surrogate" ? "surrogate" : "solver"}</span></td><td class="why">${esc(r.reasons.join("; ")) || "–"}</td>
        ${hf ? `<td class="num">${r.high_fidelity === undefined ? "–" : fmt(r.high_fidelity)}</td><td>${r.hf_in_interval === undefined ? "–" : r.hf_in_interval ? `<span class="hfok">yes</span>` : `<span class="hfbad">no</span>`}</td>` : ""}
        <td>${r.verdict ? `<span class="badge ${r.verdict === "meets limits" ? "pass" : "fail"}">${esc(r.verdict)}</span>` : r.route === "high_fidelity" ? `<span class="badge na">ask the solver</span>` : "–"}</td></tr>`).join("")}
      </table></div></div></div>`;
  }

  if (p) {
    h += `<div class="grid2" style="margin-top:14px"><div class="panel"><div class="sec" style="margin-top:0"><h3>Propagation around the nominal point</h3>
      <div class="meta">${p.n.toLocaleString()} samples (${esc({ lhs: "Latin hypercube", mc: "Monte Carlo", sobol: "Sobol QMC" }[p.method] || p.method)}) of ${p.uncertain.map((x) => esc(x.name)).join(", ")}; other inputs at their nominal values.</div>
      <div class="hist" id="hist"></div>
      <div class="kpis" style="margin-top:8px">
        <div class="kpi"><div class="l">Mean ± std</div><div class="v">${fmt(p.mean)}</div><div class="s">± ${fmt(p.std, 3)}${u}</div></div>
        <div class="kpi"><div class="l">5 % – 95 %</div><div class="v">${fmt(p.percentiles.p5, 3)}–${fmt(p.percentiles.p95, 3)}</div><div class="s">${esc(unit)}</div></div>
        ${p.p_within !== undefined ? `<div class="kpi"><div class="l">P(meets limits)</div><div class="v">${pct(p.p_within)}</div><div class="s">prediction only · CI ${pct(p.p_within_ci[0])}–${pct(p.p_within_ci[1])}</div></div>
        <div class="kpi"><div class="l">… with model error</div><div class="v">${pct(p.p_within_with_model_error)}</div><div class="s">whole interval within limits</div></div>` : ""}
        <div class="kpi"><div class="l">Samples outside / sparse</div><div class="v">${pct(p.share_outside)} / ${pct(p.share_sparse)}</div><div class="s">propagation leaving the checked region</div></div>
      </div></div></div>
      <div class="panel"><div class="sec" style="margin-top:0"><h3>Which uncertain input drives the spread</h3>
      ${p.pce ? `<div class="meta">Polynomial chaos, degree ${p.pce.degree}, ${p.pce.n_evals} evaluations: mean ${fmt(p.pce.mean)}, std ${fmt(p.pce.std, 3)} (Monte Carlo: ${fmt(p.mean)}, ${fmt(p.std, 3)}); leave-one-out Q² = ${p.pce.loo_q2.toFixed(3)}${p.pce.loo_q2 < 0.9 ? " — low: read the indices with caution" : ""}.</div>
        <div class="idx" style="margin-top:8px">${Object.entries(p.pce.ST).sort((a, b) => b[1] - a[1]).map(([k, v]) => `<span>${esc(k)}</span><div class="track"><div class="fill" style="width:${Math.max(0, Math.min(1, v)) * 100}%"></div></div><span class="num">${v.toFixed(3)}</span>`).join("")}</div>
        <div class="meta" style="margin-top:6px">Total Sobol index from the chaos coefficients (share of the propagated variance, interactions included). For a full sensitivity study use the Engineering Sensitivity Lab.</div>` : `<div class="meta">Polynomial chaos skipped (more than 8 uncertain inputs or too many terms).</div>`}
      <h3>Nominal values</h3><div class="meta">${Object.entries(p.nominal).map(([k, v]) => `${esc(k)} = ${fmt(v)}`).join(" · ")}</div></div></div></div>`;
  }
  h += `<div class="panel" style="margin-top:14px">${window.renderScope ? renderScope(R.scope) : ""}</div>`;
  $("#results").innerHTML = h;

  const rel = c.reliability;
  PChart.mount($("#relchart"), { height: 230, x: rel.map((r) => `${Math.round(100 * r.nominal)} %`), yMin: Math.min(0.4, ...rel.map((r) => r.coverage_ci[0])), yMax: 1, yLabel: "measured coverage",
    series: [{ name: "ideal", y: rel.map((r) => r.nominal), color: "#94a3b8", dash: "4 4" },
             { name: "measured (95 % CI)", y: rel.map((r) => r.coverage), lo: rel.map((r) => r.coverage_ci[0]), hi: rel.map((r) => r.coverage_ci[1]), color: "#0f766e", width: 2.2 }] });
  if (p) hist($("#hist"), p, R.cases ? R.cases.limits : null);
  $("#dljson").onclick = () => download(`uncertainty_${R.model.name}_${R.output}.json`, JSON.stringify(R, null, 1), "application/json");
  $("#dlsr").onclick = () => download(`uncertainty_${R.model.name}_${R.output}.study.json`, JSON.stringify(R.study_result, null, 1), "application/json");
  if (cs) $("#dlcsv").onclick = () => {
    const ins = Object.keys(cs.rows[0].inputs);
    const cols = ["name", ...ins, "prediction", "lower", "upper", "status", "route", "reasons", "high_fidelity", "hf_in_interval", "verdict"];
    const lines = [cols.join(",")].concat(cs.rows.map((r) => cols.map((k) => { const v = ins.includes(k) ? r.inputs[k] : k === "reasons" ? r.reasons.join("; ") : r[k]; return v === undefined || v === null ? "" : `"${String(v).replace(/"/g, '""')}"`; }).join(",")));
    download(`cases_${R.model.name}.csv`, lines.join("\n"), "text/csv");
  };
}

function hist(el, p, limits) {
  const W = Math.max(320, el.clientWidth || 600), H = 220, L = 46, Rr = 12, T = 12, B = 30;
  const e = p.histogram.edges, cnt = p.histogram.counts, mx = Math.max(...cnt);
  const lo = e[0], hi = e[e.length - 1], X = (v) => L + (v - lo) / (hi - lo || 1) * (W - L - Rr), Y = (v) => T + (1 - v / mx) * (H - T - B);
  let g = `<line x1="${L}" x2="${W - Rr}" y1="${H - B}" y2="${H - B}" stroke="#c9d1dd"/>`;
  cnt.forEach((n, i) => { g += `<rect x="${X(e[i]) + 0.5}" y="${Y(n)}" width="${Math.max(0.5, X(e[i + 1]) - X(e[i]) - 1)}" height="${H - B - Y(n)}" fill="#0f766e" opacity=".75"><title>${fmt(e[i])}–${fmt(e[i + 1])}: ${n}</title></rect>`; });
  for (const [k, lab] of [["p5", "5 %"], ["p50", "median"], ["p95", "95 %"]]) { const x = X(p.percentiles[k]); g += `<line x1="${x}" x2="${x}" y1="${T}" y2="${H - B}" stroke="#334155" stroke-dasharray="3 3"/><text x="${x + 3}" y="${T + 10}">${lab}</text>`; }
  (limits || []).forEach((v, i) => { if (v === null || v < lo || v > hi) return; const x = X(v); const right = x < W * 0.7; g += `<line x1="${x}" x2="${x}" y1="${T}" y2="${H - B}" stroke="#b42318" stroke-width="2"/><text x="${right ? x + 5 : x - 5}" y="${T + 40}" text-anchor="${right ? "start" : "end"}" style="fill:#b42318;font-weight:700">${i ? "upper" : "lower"} limit ${fmt(v)}</text>`; });
  for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; g += `<text x="${X(v)}" y="${H - B + 16}" text-anchor="${k === 0 ? "start" : k === 4 ? "end" : "middle"}">${fmt(v, 3)}</text>`; }
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="propagated output histogram">${g}</svg>`;
}

function download(name, text, type) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }

function apiSnippets() {
  const host = location.origin;
  $("#c1").textContent = `# the heat-sink example's configuration, then run it with your own cases
curl -s ${host}/api/example/heatsink/config > study.json
curl -s -X POST "${host}/api/study?wait=300" -F "config=<study.json" -F "cases=@my_designs.csv" > job.json   # a job; .result when done\ncurl -s ${host}/api/jobs/<job id>                     # status, progress, result (StudyResult in .result.study_result)

# an ONNX model with your reference data (inputs + true output, optional split column cal/test/train)
curl -s -X POST "${host}/api/study?wait=300" \\
  -F 'config={"model": {"source": "onnx", "input_names": ["x1","x2"], "output_names": ["y"]}, "output": "y", "alpha": 0.1}' \\
  -F "model=@surrogate.onnx" -F "reference=@validation_runs.parquet"`;
  $("#c2").textContent = `import pinneapple as pp
from pinneapple_analysis.study_models import StudyModel

model = StudyModel.from_zoo("heatsink_forced_mlp")
X_cal, Y_cal = model.reference("cal"); X_test, Y_test = model.reference("test")
out = pp.uncertainty.uncertainty_study(model, "r_base_k_w", X_cal, Y_cal[:, 0], X_test, Y_test[:, 0],
                                       alpha=0.1, scale="relative", X_ref=model.reference("train")[0],
                                       bounds=model.bounds, cases=X_test[:5], limits=(None, 0.4))
print(out["conformal"]["coverage"], out["cases"]["summary"])`;
}

init().catch((e) => { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; });
