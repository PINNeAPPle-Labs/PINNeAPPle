"use strict";
// Physics AI Benchmark Arena UI: public rankings from stored runs, private benchmarks as jobs, one result view.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 3) => (v === null || v === undefined || !isFinite(v)) ? "—" : (Math.abs(v) !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) ? v.toExponential(2) : (+v.toPrecision(d)).toString());
const pct = (v) => (v === null || v === undefined || !isFinite(v)) ? "—" : (100 * v).toPrecision(3) + " %";
const sec = (v) => (v === null || v === undefined || !isFinite(v)) ? "—" : v < 1e-3 ? (v * 1e6).toPrecision(3) + " µs" : v < 1 ? (v * 1e3).toPrecision(3) + " ms" : v.toPrecision(3) + " s";
const bytes = (v) => !isFinite(v) ? "—" : v < 1e3 ? v + " B" : v < 1e6 ? (v / 1e3).toPrecision(3) + " kB" : (v / 1e6).toPrecision(3) + " MB";
const COL = { ridge: "#64748b", gp: "#0891b2", gbt: "#65a30d", mlp: "#b45309", fno: "#7c3aed", deeponet: "#db2777", pinn: "#dc2626" };
let META = null, INSPECT = null;

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) { $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n)); $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`)); }
async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : j.message || JSON.stringify(j.detail || j));
  return j;
}

(async function init() {
  META = await api("/api/meta");
  const P = META.public;
  $("#benchlist").innerHTML = P.map((b, i) => `<button data-b="${b.id}" class="${i === 0 ? "on" : ""}"><b>${esc(b.title)}</b>
    <span>${b.n_samples} runs · ${b.kind} · ${b.n_methods} methods · ${b.seeds.length} seeds · ${fmt(b.budget.wall_s)} s per training · winner <b style="display:inline">${esc(b.winner)}</b> · run ${esc(b.created.slice(0, 10))}</span></button>`).join("") || `<div class="panel empty">No public results on this server yet (run run_public_benchmarks.py).</div>`;
  $("#benchlist").innerHTML += (META.external || []).map((e) => `<button data-e="${e.id}"><b>${esc(e.title)}</b><span>published run · ${esc(e.hardware.cpu || "")} · commit ${esc(e.commit.slice(0, 7))} (${esc(e.commit_date)}) · pinneapple-benchmark</span></button>`).join("");
  $$("#benchlist button").forEach((b) => b.onclick = () => { $$("#benchlist button").forEach((x) => x.classList.toggle("on", x === b)); b.dataset.e ? openExternal(b.dataset.e) : openPublic(b.dataset.b); });
  if (P.length) openPublic(P[0].id);
  $("#regimes").innerHTML = Object.entries(META.regimes).map(([k, v]) => `<label title="${esc(v)}"><input type="checkbox" data-reg="${k}" checked> ${k.replace("_", " ")}</label>`).join("");
  $("#methods").innerHTML = META.methods.map((m) => `<label><input type="checkbox" data-m="${m.id}" checked> ${esc(m.title)}</label>`).join("");
  $("#pde").innerHTML += Object.entries(META.pde_types).map(([k, v]) => `<option value="${k}">${k} (${v.coords.join(", ")}${v.coefficients.length ? "; " + v.coefficients.join(", ") : ""})</option>`).join("");
  $("#budget").max = META.limits.max_budget_s; $("#seeds").max = META.limits.max_seeds;
  protocol(); snippets(); myJobs();
})();

async function openPublic(id) {
  const r = await api(`/api/public/${id}`);
  render(r, $("#view"), [[`/api/public/${id}/report.md`, "report.md"], [`/api/public/${id}/study_result.json`, "study_result.json"], [`/api/public/${id}`, "result.json"]]);
}

async function openExternal(id) {
  const e = await api(`/api/external/${id}`), el = $("#view");
  const H = e.hardware, probs = [...new Set(e.rows.map((r) => r.problem))];
  let h = `<p class="prov">Published in <a class="link" href="${esc(e.repository)}/tree/${esc(e.commit)}/${esc(e.path)}" target="_blank">${esc(e.repository.replace("https://github.com/", ""))}/${esc(e.path)}</a> at commit <code>${esc(e.commit.slice(0, 10))}</code> (${esc(e.commit_date)}), ${esc(e.license)}. Run on <b>${esc(H.cpu || "?")}</b>, ${H.logical_cpus} logical CPUs, torch ${esc(H.torch || "?")}, ${H.torch_threads} thread per run, no GPU.</p>
    <div class="panel"><p style="margin-top:0">${esc(e.protocol)}</p><p class="meta">${esc(e.note)} Methods marked INVALID failed the run's own training gates.</p></div>`;
  for (const p of probs) {
    const rows = e.rows.filter((r) => r.problem === p), meths = [...new Set(rows.map((r) => r.method))];
    const cols = [...new Set(rows.filter((r) => r.kind === "field" || r.kind === "integral").map((r) => r.column))];
    const get = (m, c) => rows.find((r) => r.method === m && r.column === c);
    const best = Object.fromEntries(cols.map((c) => [c, Math.min(...meths.map((m) => (get(m, c) || {}).mean).filter((v) => v !== undefined))]));
    h += `<div class="panel" style="margin-top:14px"><h3 style="margin-top:0">${esc(p.replace("_", " "))}</h3><div class="hscroll"><table class="matrix ext"><thead><tr><th>method</th>${cols.map((c) => `<th class="reg">${esc(c)}</th>`).join("")}<th class="num">train s</th><th class="num">time per instance</th><th>status</th></tr></thead><tbody>
      ${meths.map((m) => { const st = (rows.find((r) => r.method === m) || {}).status || ""; return `<tr><td>${esc(m)}</td>${cols.map((c) => { const x = get(m, c); return x ? `<td class="cell ${x.mean === best[c] ? "r1" : ""}">${fmt(x.mean)} <span class="sd">± ${fmt(x.std)}</span></td>` : `<td class="na">—</td>`; }).join("")}
        <td class="num">${fmt((get(m, "train_s") || {}).mean)}</td><td class="num">${sec((get(m, "time_per_instance_s") || {}).mean)}</td><td>${st === "INVALID" ? '<span class="badge fail">invalid</span>' : `<span class="badge pass">${esc(st)}</span>`}</td></tr>`; }).join("")}
      </tbody></table></div><p class="meta">Mean ± std over ${(rows.find((r) => r.n_seeds) || {}).n_seeds || "?"} seeds; lower is better; best per column highlighted. Columns as published (test and OOD instance sets of the experiment).</p></div>`;
  }
  h += `<div class="panel" style="margin-top:14px"><div class="files"><a href="/api/external/${id}" download="${id}.json">${id}.json</a></div></div>`;
  el.innerHTML = h;
}

// ------------------------------------------------------------------ result view
function render(r, el, files) {
  const regs = Object.keys(r.rankings), ok = r.methods.filter((m) => m.applicable), mr = Object.fromEntries(r.overall.map((o) => [o.method, o.mean_rank]));
  const P = r.protocol, H = r.hardware;
  ok.sort((a, b) => (mr[a.id] ?? 1e9) - (mr[b.id] ?? 1e9));
  let h = `<p class="prov">Run <b>${esc(r.created.replace("T", " ").slice(0, 16))} UTC</b> on <b>${esc(H.cpu || "?")}</b> (${H.logical_cpus} logical CPUs, torch ${esc(H.torch || "?")}, ${P.threads} thread per run), ${fmt(r.wall_s, 3)} s in total.
    Budget <b>${fmt(P.budget.wall_s)} s</b> wall clock per training · seeds <b>${P.seeds.join(", ")}</b> · ${r.problem.n_samples} samples · dataset sha256 <code>${esc(P.dataset_sha256.slice(0, 16))}</code></p>`;
  h += `<div class="kpis">${regs.map((g) => { const w = r.rankings[g][0]; return `<div class="kpi"><div class="l">${esc(g.replace("_", " "))}</div><div class="v" style="color:${COL[w.method] || "inherit"}">${esc(w.title)}</div><div class="s">${pct(w.error_mean)} ± ${pct(w.error_std)}</div></div>`; }).join("")}
    <div class="kpi"><div class="l">Overall (mean rank)</div><div class="v">${esc(r.overall[0].title)}</div><div class="s">mean rank ${(+r.overall[0].mean_rank).toFixed(1)} over ${r.overall[0].regimes_ranked} regimes</div></div></div>`;
  h += `<div class="panel" style="margin-top:14px"><h3 style="margin-top:0">Error per regime <span class="meta">(${esc(P.metric === "rel_l2" ? "relative L2" : "nRMSE")}, mean ± std over seeds, rank; lower is better)</span></h3>
    <table class="matrix"><thead><tr><th>method</th><th>family</th>${regs.map((g) => `<th class="reg">${esc(g.replace("_", " "))}</th>`).join("")}<th class="num">mean rank</th></tr></thead><tbody>
    ${ok.map((m) => `<tr><td><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${COL[m.id]}"></i> ${esc(m.title)}${m.failures.length ? ` <span class="failed">${m.failures.length} run(s) failed</span>` : ""}</td><td>${esc(m.family)}</td>${regs.map((g) => { const x = r.rankings[g].find((y) => y.method === m.id); return x ? `<td class="cell ${x.rank === 1 ? "r1" : ""}">${pct(x.error_mean)} <span class="sd">± ${pct(x.error_std)}</span><span class="rk">${x.rank}</span></td>` : `<td class="na">—</td>`; }).join("")}<td class="num">${fmt(mr[m.id])}</td></tr>`).join("")}
    </tbody></table>
    <p class="meta">${esc(P.metric_definition)}. ${esc(P.equal_data)}. Stopping: ${esc(P.stopping)}. Hyper-parameters: ${esc(P.hyperparameters)}.</p></div>`;
  h += `<div class="panel" style="margin-top:14px"><h3 style="margin-top:0">Seed spread</h3><p class="meta">Each dot is one seed (log scale); a wide spread means the ranking between close methods is not settled.</p><div class="grid2">${regs.map((g) => `<div class="strip"><b>${esc(g.replace("_", " "))}</b>${strip(r, g, ok)}</div>`).join("")}</div></div>`;
  h += `<div class="grid2" style="margin-top:14px"><div class="panel"><h3 style="margin-top:0">Cost and physics</h3><table class="costtab"><thead><tr><th>method</th><th class="num">train</th><th class="num">inference / query</th><th class="num">model size</th><th class="num">parameters</th><th>violations</th>${ok.some((m) => m.pde_residual_ratio !== null) ? '<th class="num">PDE residual / data</th>' : ""}</tr></thead><tbody>
    ${ok.map((m) => `<tr><td>${esc(m.title)}</td><td class="num">${sec(m.cost.train_s)}</td><td class="num">${sec(m.cost.infer_s_per_query)}</td><td class="num">${bytes(m.cost.size_bytes)}</td><td class="num">${fmt(m.cost.n_params)}</td><td>${Object.entries(m.violations).map(([k, v]) => `<span class="pill ${v > 0 ? "warn" : "ok"}" title="${esc(k)}">${pct(v)}</span>`).join(" ") || "—"}</td>${m.pde_residual_ratio !== null ? `<td class="num">${fmt(m.pde_residual_ratio)}</td>` : (ok.some((x) => x.pde_residual_ratio !== null) ? "<td></td>" : "")}</tr>`).join("")}
    ${r.baselines.map((b) => `<tr style="background:#f8fafc"><td><b>solver baseline</b> <span class="meta">${esc(b.id)}</span><div class="meta">${esc(b.description || "")}; error ${pct(b.error_mean)} on the interpolation set</div></td><td class="num">—</td><td class="num">${sec(b.s_per_query)}</td><td></td><td></td><td>${Object.values(b.violations || {}).map((v) => `<span class="pill ${v > 0 ? "warn" : "ok"}">${pct(v)}</span>`).join(" ")}</td>${b.pde_residual_ratio !== undefined ? `<td class="num">${fmt(b.pde_residual_ratio)}</td>` : ""}</tr>`).join("")}
    </tbody></table><p class="meta">Violations: share of held-out predictions breaking ${(r.problem.constraints || []).map((c) => `<code>${esc(c)}</code>`).join(", ") || "no declared constraint"}. Train time is the mean over seeds on the base training set; one-shot fits report the time they needed.</p></div>
    <div class="panel"><h3 style="margin-top:0">Regimes</h3>${r.regimes.map((g) => `<div class="chk"><span>${g.available ? '<span class="pill ok">run</span>' : '<span class="pill">not run</span>'}</span><div><b>${esc(g.title)}</b> — ${esc(g.available ? g.description : g.reason)}${g.available ? ` <span class="meta">(${g.n_train} train, ${g.n_eval} evaluated)</span>` : ""}</div></div>`).join("")}
      <h3>Not compared</h3><ul class="list-plain">${r.methods.filter((m) => !m.applicable).map((m) => `<li><b>${esc(m.title)}</b>: ${esc(m.reason)}</li>`).join("")}${Object.entries(r.not_run || {}).map(([k, v]) => `<li><b>${esc(k.toUpperCase().replace("_", "-"))}</b>: ${esc(v)}</li>`).join("")}</ul>
      <h3>Independent dataset audit: <span class="badge ${r.audit.decision}">${esc(r.audit.decision)}</span></h3><p class="meta">${esc(r.audit.tool || "")}</p>
      ${(r.audit.checks || []).map((c) => `<div class="chk"><span class="badge ${c.status === "pass" ? "pass" : c.status === "fail" ? "fail" : c.status === "warn" ? "warn" : "na"}">${esc(c.status)}</span><div>${esc(c.id)}: ${esc(c.summary)}</div></div>`).join("")}</div></div>`;
  h += `<div class="panel" style="margin-top:14px"><h3 style="margin-top:0">Downloads</h3><div class="files">${files.map(([u, n]) => `<a href="${u}" download="${n}">${esc(n)}</a>`).join("")}</div>
    <p class="meta">study_result.json is a pinneapple_data.contracts StudyResult (study_type "benchmark"): the table of rankings, the audit checks and the limits, readable by the other apps.</p></div>`;
  el.innerHTML = h;
}
function strip(r, g, ok) {
  const rows = ok.map((m) => ({ m, x: (m.results[g] || {}).errors || [] })).filter((o) => o.x.length);
  const all = rows.flatMap((o) => o.x).filter((v) => v > 0);
  if (!all.length) return "";
  const lo = Math.log10(Math.min(...all)) - 0.1, hi = Math.log10(Math.max(...all)) + 0.1, W = 520, rowH = 18, H = rows.length * rowH + 26;
  const X = (v) => 90 + (Math.log10(v) - lo) / (hi - lo) * (W - 100);
  let s = "";
  rows.forEach((o, i) => { const y = 12 + i * rowH; s += `<text x="0" y="${y + 4}">${esc(o.m.title)}</text><line x1="90" x2="${W - 10}" y1="${y}" y2="${y}" stroke="#eef1f5"/>` + o.x.map((v) => `<circle cx="${X(v).toFixed(1)}" cy="${y}" r="4" fill="${COL[o.m.id]}" fill-opacity=".8"><title>${pct(v)}</title></circle>`).join(""); });
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) s += `<line x1="${X(10 ** e)}" x2="${X(10 ** e)}" y1="4" y2="${H - 18}" stroke="#c9d1dd" stroke-dasharray="2 3"/><text x="${X(10 ** e) - 12}" y="${H - 4}">${pct(10 ** e)}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" style="height:${H}px">${s}</svg>`;
}

// ------------------------------------------------------------------ private benchmark
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && inspect(e.dataTransfer.files[0]));
$("#file").addEventListener("change", (e) => { if (e.target.files[0]) inspect(e.target.files[0]); e.target.value = ""; });
async function inspect(file) {
  $("#upstatus").textContent = `Reading ${file.name}…`;
  const fd = new FormData(); fd.append("file", file, file.name);
  try { INSPECT = await api("/api/inspect", { method: "POST", body: fd }); } catch (e) { $("#upstatus").innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  const I = INSPECT, sug = I.suggested;
  $("#upstatus").textContent = `${I.filename}: ${I.n_rows} rows` + (I.roles_from_contract ? " · roles from its DatasetSpec" : " · check the suggested roles");
  const rows = [...I.numeric.map((c) => ({ name: c.name, info: `${fmt(c.min)} .. ${fmt(c.max)}` })), ...Object.entries(I.fields).map(([k, v]) => ({ name: k, info: `field, ${v.columns} columns`, field: true }))];
  $("#cols").innerHTML = `<div class="h">column</div><div class="h">input</div><div class="h">output</div>` + rows.map((r) => `<div class="n">${esc(r.name)} <small>${esc(r.info)}</small></div><input type="checkbox" data-in="${esc(r.name)}" ${sug.inputs.includes(r.name) ? "checked" : ""} ${r.field ? "disabled" : ""}><input type="checkbox" data-out="${esc(r.name)}" ${sug.outputs.includes(r.name) ? "checked" : ""}>`).join("");
  $("#group").innerHTML = `<option value="">none</option>` + I.text.map((t) => `<option ${t.name === sug.group ? "selected" : ""}>${esc(t.name)}</option>`).join("");
  $$("#cols input").forEach((x) => x.onchange = () => {
    const other = x.dataset.in !== undefined ? `#cols input[data-out="${CSS.escape(x.dataset.in)}"]` : `#cols input[data-in="${CSS.escape(x.dataset.out)}"]`;
    if (x.checked && $(other)) $(other).checked = false; sync();
  });
  $("#config").hidden = false; sync();
}
function sync() {
  const ins = $$("#cols input[data-in]:checked").map((x) => x.dataset.in), outs = $$("#cols input[data-out]:checked").map((x) => x.dataset.out);
  $("#extcol").innerHTML = `<option value="">edge of the input space</option>` + ins.map((c) => `<option value="${esc(c)}|high">${esc(c)}, high end</option><option value="${esc(c)}|low">${esc(c)}, low end</option>`).join("");
  const f = outs.find((o) => INSPECT.fields[o]); $("#fieldbox").hidden = !f;
  if (f && !$("#gridshape").value) $("#gridshape").value = INSPECT.fields[f].columns;
}
$("#pde").onchange = () => { const t = META.pde_types[$("#pde").value]; $("#pdecoef").innerHTML = t && t.coefficients.length ? "Coefficients: " + t.coefficients.map((c) => `${c} = <input data-coef="${c}" style="display:inline;width:110px" placeholder="column or number">`).join(" ") : ""; };
$("#run").onclick = async (e) => {
  const [ec, es] = ($("#extcol").value || "|").split("|");
  const body = { upload_id: INSPECT.upload_id, inputs: $$("#cols input[data-in]:checked").map((x) => x.dataset.in), outputs: $$("#cols input[data-out]:checked").map((x) => x.dataset.out),
    group: $("#group").value || null, methods: $$("#methods input:checked").map((x) => x.dataset.m), regimes: $$("#regimes input:checked").map((x) => x.dataset.reg),
    budget_s: +$("#budget").value, seeds: +$("#seeds").value, few_data_fraction: +$("#few").value, extrapolation_column: ec || null, extrapolation_side: es || "high",
    constraints: $("#constraints").value.split("\n").map((s) => s.trim()).filter(Boolean) };
  if (!$("#fieldbox").hidden) {
    body.grid_shape = $("#gridshape").value.split(/[,x ]+/).filter(Boolean).map(Number);
    const n = $("#coordnames").value.split(",").map((s) => s.trim()).filter(Boolean); if (n.length) body.coord_names = n;
    const rg = $("#coordranges").value.split(",").map((s) => s.trim()).filter(Boolean).map((s) => s.split(":").map(Number)); if (rg.length) body.coord_ranges = rg;
  }
  if ($("#pde").value) { const t = META.pde_types[$("#pde").value]; body.pde = { type: $("#pde").value, coords: Object.fromEntries(t.coords.map((c, i) => [c, (body.coord_names || [])[i] || c])), coefficients: Object.fromEntries($$("#pdecoef input").map((x) => [x.dataset.coef, x.value === "" || isNaN(+x.value) ? x.value : +x.value])) }; }
  const btn = e.currentTarget; btn.disabled = true;
  $("#status").innerHTML = `<div class="progress"><div style="width:2%"></div></div><div id="jobmsg">Queued…</div><div class="logbox" id="joblog"></div>`;
  try { const j = await api("/api/benchmarks", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }); await follow(j.id); }
  catch (err) { $("#status").innerHTML = `<span class="err">${esc(err.message)}</span>`; } finally { btn.disabled = false; myJobs(); }
};
async function follow(id) {
  let since = 0;
  for (;;) {
    const j = await api(`/api/jobs/${id}`), logs = await api(`/api/jobs/${id}/logs?since=${since}`);
    for (const l of logs) { since = Math.max(since, l.seq); const b = $("#joblog"); if (b) { b.textContent += l.line + "\n"; b.scrollTop = b.scrollHeight; } }
    const bar = $("#status .progress div"); if (bar) bar.style.width = `${Math.round(100 * (j.progress || 0))}%`;
    if ($("#jobmsg")) $("#jobmsg").textContent = `${j.status}${j.message ? ": " + j.message : ""}`;
    if (j.status === "done") { showJob(j); return; }
    if (j.status === "failed" || j.status === "cancelled") throw new Error(j.error || j.status);
    await new Promise((r) => setTimeout(r, 900));
  }
}
function showJob(j) { render(j.result, $("#runview"), j.result.files.map((f) => [`/api/benchmarks/${j.id}/files/${f}`, f])); }
async function myJobs() {
  try {
    const jobs = (await api("/api/jobs?kind=arena&limit=8")).filter((j) => j.status === "done");
    if ($("#myjobs")) $("#myjobs").innerHTML = jobs.length ? `<div class="joblist"><b>Your benchmarks</b>${jobs.map((j) => `<div><span>${esc(j.id)}</span><button data-j="${j.id}">open</button></div>`).join("")}</div>` : "";
    $$("#myjobs button").forEach((b) => b.onclick = async () => showJob(await api(`/api/jobs/${b.dataset.j}`)));
  } catch (e) { /* no runtime */ }
}

function protocol() {
  $("#protocol").innerHTML = `<h2 style="margin-top:0">The protocol</h2>
    <p>Rigor is the product: a ranking is only worth what its rules are. Every public and private benchmark here follows the same ones (pinneapple_arena.regimes).</p>
    <ul class="list-plain">
      <li><b>Equal data.</b> One split, fixed before training, shared by every method and seed. The model seed varies; the data does not.</li>
      <li><b>Equal compute.</b> Every iterative method (MLP, FNO, DeepONet, PINN) trains until the same wall-clock ComputeBudget is spent (pinneapple_analysis.cost.equal_budget). One-shot fits (ridge, Gaussian process, gradient-boosted trees) are fitted once and report the time they took — never more than the budget.</li>
      <li><b>Same stopping rule.</b> Budget spent, best of ten validation checkpoints; evaluation time is not charged.</li>
      <li><b>Several seeds.</b> Every number has its spread; the seed-spread plot shows whether close rankings are settled.</li>
      <li><b>Unseen data only.</b> Errors are measured on samples no method trained on or was chosen with.</li>
      <li><b>Regimes.</b> ${Object.entries(META.regimes).map(([k, v]) => `<i>${k.replace("_", " ")}</i>: ${esc(v)}`).join("; ")}. A regime the data cannot support is reported as not run, with the reason.</li>
      <li><b>Physics.</b> Share of predictions breaking the dataset's declared constraints, and for fields with a declared PDE the finite-difference residual of the predictions relative to the reference data's own.</li>
      <li><b>Independent validation.</b> The splits are audited by pinneapple_data.audit — the dataset checks of <code>pinneapple check</code> (PC451–PC453: finite values, duplicates, train/test leakage, declared constraints).</li>
      <li><b>Provenance.</b> Each result records its date, hardware (CPU, cores, torch version, threads) and the dataset's sha256; the public results are files written by run_public_benchmarks.py, rerun on the deployment hardware.</li>
      <li><b>Hyper-parameters.</b> Library defaults for every method, no per-method search: a tuned method may do better. pp.benchmark (pinneapple_analysis.benchmark) adds equal-trial searches and budget curves for field-operator problems.</li>
    </ul>
    <h3>Not compared, and why</h3><ul class="list-plain">${Object.entries(META.not_in_library).map(([k, v]) => `<li><b>${esc(k.toUpperCase().replace("_", "-"))}</b>: ${esc(v)}</li>`).join("")}</ul>`;
}
function snippets() {
  const o = location.origin;
  $("#c1").textContent = `curl -s ${o}/api/public                                   # list
curl -s ${o}/api/public/airfoil_rans | jq .rankings        # full result
curl -s ${o}/api/public/airfoil_rans/report.md            # comparative report

# a private benchmark on your table
curl -s -X POST ${o}/api/uploads -F file=@runs.csv         # -> {"id": ...}
curl -s -X POST ${o}/api/benchmarks -H 'content-type: application/json' \\
  -d '{"upload_id": "<id>", "inputs": ["a","b"], "outputs": ["y"], "budget_s": 3, "seeds": 3}'
curl -s ${o}/api/jobs/<job id>

# the same from Python
from pinneapple_arena.regimes import run_regime_arena
from pinneapple_analysis.surrogate import examples
print(run_regime_arena(examples.airfoil_rans(), budget=3.0, seeds=3).to_markdown())`;
}
