"use strict";
// Simulation Campaign Planner UI: parameter space editor, design options, job progress, coverage and case files.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => (v === null || v === undefined || (typeof v === "number" && !isFinite(v)) ? "—" : typeof v !== "number" ? esc(v) : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(4) + "");
const pct = (v) => (v === null || v === undefined ? "—" : (100 * v).toFixed(0) + "%");
let META = null, RES = null, JOB = null, UPLOADS = { existing: null, base: null, inp: null }, AX = [0, 1];

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`));
}

async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j));
  return j;
}
const postJSON = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
async function upload(file) {
  const fd = new FormData(); fd.append("file", file, file.name);
  return api("/api/uploads", { method: "POST", body: fd });
}

// ------------------------------------------------------------------ parameter table
const KINDS = ["continuous", "integer", "categorical", "constant"];
function paramRow(p = { name: "", kind: "continuous" }) {
  const tr = document.createElement("tr");
  const vals = p.kind === "categorical" ? (p.values || []).join("|") : p.kind === "constant" ? (p.value ?? "") : "";
  tr.innerHTML = `<td class="n"><input data-k="name" value="${esc(p.name)}" placeholder="name"></td>
    <td class="k"><select data-k="kind">${KINDS.map((k) => `<option ${k === p.kind ? "selected" : ""}>${k}</option>`).join("")}</select></td>
    <td class="v"><input data-k="low" value="${p.low ?? ""}" inputmode="decimal"></td>
    <td class="v"><input data-k="high" value="${p.high ?? ""}" inputmode="decimal"></td>
    <td class="vals"><input data-k="values" value="${esc(vals)}" placeholder="a|b|c or value"></td>
    <td class="v"><input data-k="unit" value="${esc(p.unit || "")}"></td>
    <td class="lg"><input type="checkbox" data-k="log" ${p.log ? "checked" : ""}></td>
    <td><button type="button" title="remove">×</button></td>`;
  tr.dataset.group = p.group || "other"; tr.dataset.description = p.description || "";
  tr.querySelector("button").onclick = () => tr.remove();
  const sync = () => { const k = tr.querySelector("[data-k=kind]").value; const num = k === "continuous" || k === "integer";
    ["low", "high"].forEach((f) => tr.querySelector(`[data-k=${f}]`).disabled = !num);
    tr.querySelector("[data-k=values]").disabled = num; tr.querySelector("[data-k=log]").disabled = k !== "continuous"; };
  tr.querySelector("[data-k=kind]").onchange = sync; sync();
  return tr;
}
function setSpace(space) {
  const tb = $("#ptable tbody"); tb.innerHTML = "";
  space.parameters.forEach((p) => tb.appendChild(paramRow(p)));
  $("#constraints").value = (space.constraints || []).join("\n");
  if (space.budget) $("#n").value = space.budget;
  $("#ptable").dataset.name = space.name || "my_campaign";
}
const num = (s) => (s === "" || s === undefined ? undefined : Number(s));
function readSpace() {
  const params = $$("#ptable tbody tr").map((tr) => {
    const g = (k) => tr.querySelector(`[data-k=${k}]`);
    const p = { name: g("name").value.trim(), kind: g("kind").value, unit: g("unit").value.trim(), group: tr.dataset.group };
    if (tr.dataset.description) p.description = tr.dataset.description;
    if (p.kind === "continuous" || p.kind === "integer") { p.low = num(g("low").value); p.high = num(g("high").value); }
    if (p.kind === "continuous") p.log = g("log").checked;
    if (p.kind === "categorical") p.values = g("values").value.split("|").map((s) => s.trim()).filter(Boolean).map((s) => (isFinite(+s) && s !== "" ? +s : s));
    if (p.kind === "constant") { const v = g("values").value.trim(); p.value = isFinite(+v) && v !== "" ? +v : v; }
    return p;
  });
  return { kind: "ParameterSpace", name: $("#ptable").dataset.name || "my_campaign", parameters: params,
    constraints: $("#constraints").value.split("\n").map((s) => s.trim()).filter(Boolean), budget: +$("#n").value || undefined };
}
$("#add-param").onclick = () => $("#ptable tbody").appendChild(paramRow());
$("#ptable-file").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try { const up = await upload(f); setSpace(await postJSON("/api/space/from-table", { upload: up.id, name: f.name.replace(/\.[^.]+$/, "") })); status(`Loaded ${f.name}`); }
  catch (err) { status(err.message, true); }
  e.target.value = "";
};

// ------------------------------------------------------------------ presets
function applyPreset(key) {
  const custom = key === "custom", p = META.presets[key];
  $("#custom-solvers").hidden = !custom;
  $("#use-preset-existing").closest("label").hidden = !(p && p.existing);
  $("#use-preset-existing").checked = !!(p && p.existing);
  $("#preset-existing-label").textContent = p && p.existing ? `Use the ${p.existing}` : "";
  $("#x-openfoam-l").textContent = custom ? "OpenFOAM cases from my base case" : p.openfoam ? "OpenFOAM simpleFoam cases (airfoil builder)" : "OpenFOAM cases (not for this preset)";
  $("#x-calculix-l").textContent = custom ? "CalculiX decks from my template" : p.calculix ? "CalculiX decks (cantilever builder)" : "CalculiX decks (not for this preset)";
  $("#x-openfoam").disabled = !custom && !p.openfoam; $("#x-calculix").disabled = !custom && !p.calculix;
  $("#x-openfoam").checked = !custom && !!p.openfoam; $("#x-calculix").checked = !custom && !!p.calculix;
  if (p) { setSpace(p.space); $("#method").value = p.method; $("#outputs").value = p.outputs.join(", "); $("#preset-note").textContent = p.note; }
  else { $("#preset-note").textContent = "Describe your parameters, or load a table with one row per parameter (name, kind, low, high, values, unit, log)."; $("#method").value = "maximin"; }
}
$("#preset").onchange = () => applyPreset($("#preset").value);

// ------------------------------------------------------------------ uploads of existing runs / solver files
const dropE = $("#drop-existing");
["dragenter", "dragover"].forEach((ev) => dropE.addEventListener(ev, (e) => { e.preventDefault(); dropE.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dropE.addEventListener(ev, (e) => { e.preventDefault(); dropE.classList.remove("over"); }));
dropE.addEventListener("drop", (e) => e.dataTransfer.files[0] && useExisting(e.dataTransfer.files[0]));
$("#existing-file").onchange = (e) => e.target.files[0] && useExisting(e.target.files[0]);
async function useExisting(f) {
  try {
    const up = await upload(f), info = await postJSON("/api/existing/inspect", { upload: up.id });
    UPLOADS.existing = up.id; $("#use-preset-existing").checked = false;
    $("#existing-lab").textContent = `${f.name}: ${info.rows} runs, ${info.columns.length} columns`;
    $("#id-col").innerHTML = `<option value="">(row number)</option>` + info.columns.map((c) => `<option>${esc(c)}</option>`).join("");
    const guess = info.columns.find((c) => /(^|_)(run_?)?id$/i.test(c)); if (guess) $("#id-col").value = guess;
    const names = new Set(readSpace().parameters.map((p) => p.name));
    $("#out-cols").innerHTML = info.numeric.filter((c) => !names.has(c)).map((c) => `<option>${esc(c)}</option>`).join("");
    $("#existing-cols").hidden = false;
  } catch (err) { status(err.message, true); }
}
for (const [id, key, lab] of [["#base-file", "base", "#base-lab"], ["#inp-file", "inp", "#inp-lab"]]) {
  $(id).onchange = async (e) => { const f = e.target.files[0]; if (!f) return;
    try { UPLOADS[key] = (await upload(f)).id; $(lab).textContent = `${f.name} uploaded`; } catch (err) { status(err.message, true); } };
}

// ------------------------------------------------------------------ submit + poll
function status(msg, err = false) { $("#status").innerHTML = err ? `<span class="err">${esc(msg)}</span>` : esc(msg); }
function request() {
  const preset = $("#preset").value, custom = preset === "custom";
  const body = { preset, space: readSpace(), n: +$("#n").value || null, method: $("#method").value, seed: +$("#seed").value || 0,
    levels: +$("#levels").value || 3, drop_repeats: $("#drop-repeats").checked,
    use_preset_existing: $("#use-preset-existing").checked && !UPLOADS.existing,
    outputs: [...new Set([...$("#outputs").value.split(",").map((s) => s.trim()).filter(Boolean), ...$$("#out-cols option:checked").map((o) => o.value)])],
    case_limit: 50 };
  if ($("#tol").value) body.tol = +$("#tol").value;
  if (UPLOADS.existing) { body.existing_upload = UPLOADS.existing; body.existing_id_column = $("#id-col").value || null; }
  if ($("#x-openfoam").checked) {
    body.openfoam = custom ? "base" : "airfoil";
    if (custom) { body.openfoam_base_upload = UPLOADS.base;
      body.openfoam_edits = $("#edits").value.split("\n").filter((l) => l.includes("|")).map((l) => { const [file, key, ...v] = l.split("|"); return { file: file.trim(), key: key.trim(), value: v.join("|").trim() }; }); }
  }
  if ($("#x-calculix").checked) { body.calculix = custom ? "template" : "cantilever"; if (custom) body.calculix_template_upload = UPLOADS.inp; }
  return body;
}
async function start(promise, btn) {
  const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Planning…`; status("");
  try { JOB = await promise; await poll(JOB.id); }
  catch (e) { status(e.message, true); }
  finally { btn.disabled = false; btn.innerHTML = old; $("#progress").hidden = true; }
}
async function poll(id) {
  $("#progress").hidden = false;
  for (;;) {
    const j = await api(`/api/jobs/${id}`);
    $("#pfill").style.width = `${Math.round(100 * (j.progress || 0))}%`; $("#pmsg").textContent = j.message || j.status;
    if (j.status === "done") { RES = j.result; JOB = j; AX = [0, Math.min(1, RES.view.names.length - 1)]; render(); showTab("result"); window.scrollTo(0, 0); return; }
    if (j.status === "failed" || j.status === "cancelled") throw new Error(j.error || j.status);
    await new Promise((r) => setTimeout(r, 400));
  }
}
$("#run").onclick = (e) => start(postJSON("/api/campaigns", request()), e.currentTarget);

// ------------------------------------------------------------------ result
function kpi(l, v, s = "") { return `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`; }
function render() {
  const r = RES, s = r.summary, cp = s.coverage.planned, ce = s.coverage.existing, cc = s.coverage.combined, red = s.redundancy;
  const nfail = r.checks.filter((c) => c.status === "FAIL").length, nwarn = r.checks.filter((c) => c.status === "WARN").length;
  const banner = nfail ? `<div class="banner bad">${nfail} check(s) failed: read them before running.</div>`
    : `<div class="banner ${nwarn ? "warn" : "ok"}">${r.n_cases} cases planned (${esc(s.method.replace("_", " "))})${s.n_dropped_as_repeats ? `, ${s.n_dropped_as_repeats} repeat(s) of existing runs dropped` : ""}${nwarn ? ` · ${nwarn} warning(s)` : ""}</div>`;
  const kp = [kpi("Cases to run", r.n_cases, s.n_generated !== r.n_cases ? `${s.n_generated} generated` : "in priority order"),
    kpi("Repeats dropped", s.n_dropped_as_repeats, `tolerance ${fmt(red.tol)}`),
    kpi("Discrepancy (CD)", fmt(cp.discrepancy), "lower is more uniform"),
    kpi("Min distance", fmt(cp.min_distance), `ideal spacing ${fmt(cp.ideal_spacing)}`)];
  if (ce) {
    kp.push(kpi("Existing runs", red.n_existing, `${esc(s.existing_source || "")}${red.n_existing_outside_space ? ` · ${red.n_existing_outside_space} outside the space` : ""}`));
    kp.push(kpi("Biggest hole", `${fmt(ce.fill_distance)} → ${fmt(cc.fill_distance)}`, `fill distance, ${s.fill_distance_reduction != null ? (100 * s.fill_distance_reduction).toFixed(0) + " % smaller" : ""}`));
    kp.push(kpi("Duplicates already run", red.existing_duplicate_pairs, "near-identical pairs in your runs"));
  } else kp.push(kpi("Biggest hole", fmt(cp.fill_distance), "fill distance of the plan"));
  const checks = r.checks.map((c) => `<div class="item"><span class="badge ${c.status === "PASS" ? "pass" : c.status === "FAIL" ? "fail" : "warn"}">${c.status}</span><div><b>${esc(c.name)}</b> <span class="mono">${fmt(c.value)}</span>${c.threshold !== null && c.threshold !== undefined ? ` <span class="d">(threshold ${fmt(c.threshold)})</span>` : ""}<div class="d">${esc(c.detail)}</div></div></div>`).join("");
  const base = `/api/campaigns/${JOB.id}/download/`;
  const dls = [["campaign.zip", `Everything (.zip, ${r.files.archive_mb} MB)`, "main"], ["cases.csv", "cases.csv"], ["cases.parquet", "cases.parquet"], ["study.json", "StudyResult"],
    ["dataset.skeleton.json", "DatasetSpec skeleton"], ["space.json", "ParameterSpace"], ["run_cases.py", "Python runner"]]
    .concat(r.files.openfoam_plan ? [["plan.json", "OpenFOAM campaign plan"]] : []).concat(s.n_dropped_as_repeats ? [["dropped_repeats.csv", "dropped repeats"]] : [])
    .map(([f, l, c]) => `<a class="${c || ""}" href="${base}${f}">${esc(l)}</a>`).join("");
  const solverNote = r.files.case_dirs ? `${r.files.case_dirs} ${r.files.openfoam ? "OpenFOAM case directories" : "CalculiX decks"} in the archive (first cases in priority order; the tables hold all ${r.n_cases}).` : "No solver case files requested: tables and the Python runner only.";
  $("#result").innerHTML = `
    <div class="panel"><div class="toolbar"><h2 style="margin:0">${esc(s.name)}</h2><span class="meta">${esc(s.id)}</span></div>
      ${banner}<div class="kpis">${kp.join("")}</div></div>
    <div class="grid-res" style="margin-top:16px">
      <div class="panel"><div class="toolbar"><h3 style="margin:0">2-D projection</h3><div class="axsel" id="axsel"></div></div>
        <div class="scatter" id="scatter"></div>
        <div class="legend" style="margin-top:6px"><span><i style="background:#94a3b8"></i>existing runs</span><span><i style="background:var(--brand)"></i>planned (darker = earlier)</span><span><i style="background:#dc2626"></i>dropped: repeats a run</span></div></div>
      <div class="panel"><h3 style="margin-top:0">Checks</h3><div class="checklist">${checks}</div>
        <h3>1-D coverage of the plan <span class="meta">(fraction of n equal bins, or of the k values of a discrete parameter, holding a case)</span></h3><div class="cov1">${cov1(cp)}</div>
        ${cp.projection_2d_worst ? `<p class="meta">Worst-covered pair: <b>${esc(cp.projection_2d_worst.pair.replace("|", " × "))}</b>, ${pct(cp.projection_2d_worst.filled)} of ${cp.projection_2d_bins}×${cp.projection_2d_bins} cells.</p>` : ""}
        ${s.design_info && s.design_info.generators ? `<p class="meta">Generators ${esc(Object.entries(s.design_info.generators).map(([k, v]) => `${k} = ${v}`).join(", "))}; resolution ${s.design_info.resolution}.</p>` : ""}
        ${s.design_info && s.design_info.note ? `<p class="meta">${esc(s.design_info.note)}</p>` : ""}</div>
    </div>
    <div class="panel" style="margin-top:16px"><div class="toolbar"><h3 style="margin:0">Cases, in the order to run them</h3><span class="meta">${r.n_cases > 500 ? "first 500 shown" : ""}</span></div>
      <div class="casetab">${caseTable(r)}</div>
      <h3>Downloads</h3><div class="dl">${dls}</div><p class="meta">${esc(solverNote)}</p>
      ${red.existing_outside_space && red.existing_outside_space.length ? `<p class="meta">Existing runs outside the parameter space (not counted in coverage): ${esc(red.existing_outside_space.slice(0, 12).join(", "))}${red.existing_outside_space.length > 12 ? ", …" : ""}</p>` : ""}
      ${red.existing_redundant && red.existing_redundant.length ? `<p class="meta">Near-duplicate runs in your data: ${esc(red.existing_redundant.slice(0, 12).join(", "))}</p>` : ""}
    </div>
    <div class="panel" style="margin-top:16px">${window.renderScope ? renderScope(r.scope) : ""}</div>`;
  axisSelect(); scatter();
}
function cov1(cp) {
  return Object.keys(cp.projection_1d || {}).map((k) => `<span>${esc(k)}</span><div class="track"><div class="fill" style="width:${100 * (cp.projection_1d[k] || 0)}%;background:var(--brand)"></div></div><span class="mono">${pct(cp.projection_1d[k])}</span>`).join("");
}
function caseTable(r) {
  const hide = new Set(["repeats_planned"].concat(r.view.existing ? [] : ["nearest_existing", "distance_to_existing", "repeats_existing"]));
  const cols = r.columns.filter((c) => !hide.has(c));
  const head = cols.map((c) => `<th class="${c === "case_id" || c === "nearest_existing" ? "" : "num"}">${esc(c.replace(/_/g, " "))}</th>`).join("");
  const body = r.cases.map((row) => `<tr>${cols.map((c) => `<td class="${c === "case_id" || c === "nearest_existing" ? "" : "num"}">${c === "repeats_existing" ? (row[c] ? `<span class="flag">yes</span>` : "no") : c === "information" && row[c] == null ? "first" : fmt(row[c])}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}
function axisSelect() {
  const n = RES.view.names, opt = (i) => n.map((x, k) => `<option value="${k}" ${k === AX[i] ? "selected" : ""}>${esc(RES.view.labels[x])}</option>`).join("");
  $("#axsel").innerHTML = `x <select id="ax0">${opt(0)}</select> y <select id="ax1">${opt(1)}</select>`;
  $("#ax0").onchange = (e) => { AX[0] = +e.target.value; scatter(); }; $("#ax1").onchange = (e) => { AX[1] = +e.target.value; scatter(); };
}
function axisTicks(name) {
  const v = RES.view, b = v.bounds[name];
  if (v.kinds[name] === "categorical") { const vals = v.values[name]; return vals.map((x, i) => [(i + 0.5) / vals.length, String(x)]); }
  if (!b) return [];
  const [lo, hi, log] = b;
  return [0, 0.25, 0.5, 0.75, 1].map((u) => [u, fmt(log ? Math.exp(Math.log(lo) + u * (Math.log(hi) - Math.log(lo))) : lo + u * (hi - lo))]);
}
function scatter() {
  const v = RES.view, [i, j] = AX, W = 620, H = 420, L = 58, R = 12, T = 10, B = 44, pw = W - L - R, ph = H - T - B;
  const X = (u) => L + u * pw, Y = (u) => T + (1 - u) * ph, n = v.planned.length;
  let s = `<svg viewBox="0 0 ${W} ${H}"><rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="#fbfcfe" stroke="#e1e6ee"/>`;
  for (const [u, lab] of axisTicks(v.names[i])) s += `<line class="ax" x1="${X(u)}" x2="${X(u)}" y1="${T + ph}" y2="${T + ph + 4}"/><text x="${X(u)}" y="${T + ph + 16}" text-anchor="middle">${esc(lab)}</text>`;
  for (const [u, lab] of axisTicks(v.names[j])) s += `<line class="ax" x1="${L - 4}" x2="${L}" y1="${Y(u)}" y2="${Y(u)}"/><text x="${L - 6}" y="${Y(u) + 3}" text-anchor="end">${esc(lab)}</text>`;
  s += `<text x="${L + pw / 2}" y="${H - 6}" text-anchor="middle" class="yl">${esc(v.labels[v.names[i]])}</text><text transform="translate(12 ${T + ph / 2}) rotate(-90)" text-anchor="middle">${esc(v.labels[v.names[j]])}</text>`;
  for (const p of v.existing || []) s += `<circle cx="${X(p[i])}" cy="${Y(p[j])}" r="3" fill="#94a3b8" fill-opacity=".55"/>`;
  for (const p of v.dropped || []) s += `<path d="M${X(p[i]) - 4} ${Y(p[j]) - 4}l8 8m0 -8l-8 8" stroke="#dc2626" stroke-width="2"/>`;
  v.planned.forEach((p, k) => { const t = n > 1 ? k / (n - 1) : 0; const c = `hsl(174 ${80 - 30 * t}% ${22 + 38 * t}%)`;
    s += `<circle cx="${X(p[i])}" cy="${Y(p[j])}" r="5" fill="${c}" stroke="#fff" stroke-width="1"><title>${esc(RES.cases[k] ? RES.cases[k].case_id : "")} · priority ${k + 1}</title></circle>`; });
  $("#scatter").innerHTML = s + `</svg>`;
}

// ------------------------------------------------------------------ boot
(async function boot() {
  META = await api("/api/meta");
  $("#examples").innerHTML = Object.entries(META.presets).map(([k, p]) => `<button type="button" data-ex="${k}"><b>${esc(p.title)}</b>${esc(p.note)}</button>`).join("");
  $$("#examples button").forEach((b) => b.onclick = () => { $("#preset").value = b.dataset.ex; applyPreset(b.dataset.ex); start(api(`/api/example/${b.dataset.ex}`, { method: "POST" }), b); });
  applyPreset("airfoil");
  const host = location.origin;
  $("#c1").textContent = `# start a campaign (a job), then poll it and download the files
curl -s -X POST ${host}/api/campaigns -H 'Content-Type: application/json' \\
  -d '{"preset": "cantilever", "method": "maximin", "n": 30, "calculix": "cantilever"}'
curl -s ${host}/api/jobs/<job id>                              # status, progress, result
curl -s -o campaign.zip ${host}/api/campaigns/<job id>/download/campaign.zip
curl -s ${host}/api/usage                                      # your use this month and the free limits`;
  $("#c2").textContent = `from pinneapple_data.contracts import Parameter, ParameterSpace
from pinneapple_data.doe import plan_campaign
from pinneapple_simulation.external_solvers.campaign_cases import write_campaign

space = ParameterSpace(name="fin", budget=30, parameters=[
    Parameter(name="height", low=0.01, high=0.05, unit="m", group="geometry"),
    Parameter(name="material", kind="categorical", values=["Al6061", "Cu"], group="material")],
    constraints=["height <= 0.04 or material == 'Al6061'"])
plan = plan_campaign(space, method="maximin", existing=runs_done, existing_id_column="run_id")
plan.summary()                                  # coverage before / after, repeats dropped
write_campaign(plan, "campaign/", outputs=["Rth"])   # tables, run_cases.py, study.json, dataset.skeleton.json`;
})().catch((e) => status(e.message, true));
