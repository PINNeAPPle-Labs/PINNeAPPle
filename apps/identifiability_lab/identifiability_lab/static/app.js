// Parameter Identifiability Lab page: problem editor (shared), analysis job, report.
const $ = (s) => document.querySelector(s);
const { api, pollJob, xy, heat, fmt, esc } = window.InvUI;
let META = null, ED = null, LAST = null;
const COLORS = ["#6d28d9", "#0b6bcb", "#c2410c", "#1a7f4b", "#b42318", "#0f766e", "#9a6200", "#475569"];

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
}));

function stClass(s) { return s === "identifiable" ? "ok" : s.startsWith("practically") ? "practical" : "structural"; }

async function init() {
  META = await api("/api/meta");
  ED = InvUI.ProblemEditor($("#editor"), { meta: META, values: "optional", title: "Measurement plan" });
  $("#preset").innerHTML += META.presets.map((p) => `<option value="${p.id}">${esc(p.title)}</option>`).join("");
  $("#preset").onchange = () => loadPreset($("#preset").value);
  $("#run").onclick = run;
  $("#api-code").textContent = apiExample();
  const first = new URLSearchParams(location.search).get("preset") || "fin_h_k";
  $("#preset").value = first;
  await loadPreset(first);
}

async function loadPreset(id) {
  if (!id) return;
  const p = await api(`/api/presets/${id}`);
  ED.set(p);
  const meta = META.presets.find((x) => x.id === id);
  $("#preset-note").innerHTML = `${esc(p.summary)}<br><i>${esc(meta?.lesson || "")}</i>`;
}

async function run() {
  const btn = $("#run");
  let body;
  try { body = ED.get(); } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  if (!body.observations) { $("#status").innerHTML = `<span class="err">add at least one measurement</span>`; return; }
  body.options = { n_suggest: +$("#o-n").value, practical_threshold: +$("#o-thr").value,
    profile: { auto: "auto", true: true, false: false }[$("#o-prof").value], posterior: { auto: "auto", true: true, false: false }[$("#o-post").value] };
  btn.disabled = true;
  $("#status").innerHTML = `<div>queued…</div><div class="progress"><div></div></div>`;
  try {
    const job = await api("/api/analyze", { json: body });
    const done = await pollJob(job.id, (j) => {
      $("#status").innerHTML = `<div>${esc(j.status)} ${esc(j.message || "")}</div><div class="progress"><div style="width:${Math.round(100 * (j.progress || 0))}%"></div></div>`;
    });
    if (done.status !== "done") throw new Error(done.error || done.status);
    LAST = { id: done.id, r: done.result };
    $("#status").innerHTML = `done in ${fmt(done.result.seconds, 2)} s`;
    render(done.result, done.id);
  } catch (e) {
    $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`;
  } finally { btn.disabled = false; }
}

function render(r, jid) {
  const P = r.parameters, names = P.map((p) => p.name);
  const nOk = P.filter((p) => p.status === "identifiable").length;
  const cls = nOk === P.length ? "ok" : nOk === 0 ? "bad" : "warn";
  let h = `<div class="verdict ${cls}"><b>${nOk} of ${P.length} parameters identifiable.</b> ${esc(r.summary.split(". ").slice(1).join(". "))}</div>`;
  h += `<div class="kpis">
    <div class="kpi"><div class="l">Fisher rank</div><div class="v">${r.rank} / ${r.n_parameters}</div></div>
    <div class="kpi"><div class="l">Condition number</div><div class="v">${r.condition_number === null ? "∞" : fmt(r.condition_number, 3)}</div></div>
    <div class="kpi"><div class="l">Measurements</div><div class="v">${r.n_observations}</div></div>
    <div class="kpi"><div class="l">Data</div><div class="v" style="font-size:13px">${esc(r.data)}</div></div></div>`;
  h += `<h3 class="sec">Parameters</h3><div class="cards">${P.map((p) => `<div class="card">
      <div class="nm"><span>${esc(p.name)} <span class="meta">${esc(p.unit)}</span></span><span class="st ${stClass(p.status)}">${esc(p.status)}</span></div>
      <div class="row">at <b>${fmt(p.value)}</b> ${esc(p.unit)} · range ${fmt(p.lower)} – ${fmt(p.upper)}</div>
      <div class="row">Cramér-Rao bound: <b>${p.crlb_std === null ? "unbounded" : "± " + fmt(p.crlb_std, 3) + " " + esc(p.unit) + (p.crlb_relative !== null ? ` (±${fmt(100 * p.crlb_relative, 3)} %)` : "")}</b></div>
      <div class="row">Profile 95 %: <b>${p.profile_ci95 ? `${p.profile_ci95[0] === null ? "open" : fmt(p.profile_ci95[0])} … ${p.profile_ci95[1] === null ? "open" : fmt(p.profile_ci95[1])}` : "not run"}</b></div>
      <div class="row">Sensitivity share ${fmt(100 * p.sensitivity_share, 3)} %</div><div class="sbar"><div style="width:${Math.round(100 * p.sensitivity_share)}%"></div></div>
      <div class="row" style="font-size:12px">${esc(p.reason)}</div></div>`).join("")}</div>`;
  if (r.ambiguities.length) {
    h += `<h3 class="sec">What the data cannot tell apart</h3>` + r.ambiguities.map((a) => `<div class="amb ${a.kind}"><b>${a.kind === "structural" ? "Structural" : "Practical"}:</b> ${esc(a.text)}.
      <span class="meta">eigenvalue ${fmt(a.eigenvalue, 3)}${a.std_along ? `, σ along it ${fmt(a.std_along, 3)} (scaled units)` : ""}</span></div>`).join("");
  }
  if (r.suggestions.length) {
    h += `<h3 class="sec">Measurements that would help most (D-optimal)</h3><table class="t"><thead><tr><th>#</th><th>Measure</th><th>σ</th><th>log det F gain</th><th>Rank after</th>${names.map((n) => `<th>±σ ${esc(n)} after</th>`).join("")}</tr></thead><tbody>` +
      r.suggestions.map((s, i) => `<tr><td>${i + 1}</td><td>${esc(s.label)}</td><td>${fmt(s.sigma, 3)}</td><td>${fmt(s.gain_logdet, 3)}</td><td>${s.rank_after} / ${r.n_parameters}</td>${names.map((n) => `<td>${s.std_after[n] === null ? "∞" : fmt(s.std_after[n], 3)}</td>`).join("")}</tr>`).join("") +
      `</tbody></table><div class="hint">σ after in scaled units (ln units for log parameters: 0.1 ≈ ±10 %); each row assumes the rows above it are measured too.</div>`;
  }
  h += `<div class="grid2"><div><h3 class="sec">Expected correlations (Fisher)</h3><div id="corr"></div></div>
    <div><h3 class="sec">Posterior correlations (MCMC)</h3><div id="pcorr">${r.posterior ? "" : '<div class="hint">not run (expensive model, unbounded parameter or skipped)</div>'}</div>
    ${r.posterior ? `<div class="hint">${r.posterior.n_samples} samples, acceptance ${fmt(r.posterior.acceptance, 2)}; 95 % intervals: ${names.map((n, i) => `${esc(n)} ${fmt(r.posterior.lo[i])} … ${fmt(r.posterior.hi[i])}`).join("; ")}</div>` : ""}</div></div>`;
  const prof = Object.keys(r.profiles || {});
  if (prof.length) h += `<h3 class="sec">Profile likelihood</h3><div class="grid2">${prof.map((n, i) => `<div><div id="prof-${i}"></div></div>`).join("")}</div>`;
  h += `<h3 class="sec">Sensitivity of each measurement</h3><div id="sens"></div>`;
  h += `<h3 class="sec">Value of each measurement (leverage)</h3><div id="lev"></div><div class="hint">${esc(r.observations.note)}</div>`;
  if (r.notes.length) h += `<h3 class="sec">Notes</h3><ul class="list-plain">${r.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>`;
  h += `<div class="dl"><a href="/api/jobs/${jid}/download/report.json">Report (JSON)</a><a href="/api/jobs/${jid}/download/study.json">StudyResult (contract)</a>
    <a href="/api/jobs/${jid}/download/parameters.csv">Parameters (CSV)</a><a href="/api/jobs/${jid}/download/suggestions.csv">Suggestions (CSV)</a></div>`;
  h += `<div class="hint" style="margin-top:6px">Derivatives: ${esc(r.derivatives)} · ${fmt(r.seconds, 2)} s on the server.</div>`;
  h += window.renderScope ? renderScope(META.scope) : "";
  $("#out").innerHTML = h;
  heat($("#corr"), r.correlation, names);
  if (r.posterior) heat($("#pcorr"), r.posterior.correlation, names);
  prof.forEach((n, i) => {
    const pr = r.profiles[n], p = P.find((x) => x.name === n);
    xy($(`#prof-${i}`), { height: 200, xLabel: `${n} (${p.unit})`, yLabel: "Δχ²", xLog: p.log,
      series: [{ name: "profile", x: pr.grid, y: pr.delta_chi2, color: COLORS[i % COLORS.length] }],
      hlines: [{ y: pr.threshold, label: "95 % (3.84)", color: "#b42318" }], vlines: [{ x: p.value, label: "analysis point" }] });
  });
  const rows = r.observations.rows, key = rows.length ? Object.keys(rows[0].inputs)[0] : null;
  const xs = rows.map((row) => row.inputs[key]);
  const useIdx = !key || xs.some((x) => x === null || !isFinite(x)) || new Set(xs).size < rows.length / 2;
  xy($("#sens"), { height: 220, xLabel: useIdx ? "measurement #" : key, yLabel: "∂y / ∂(scaled θ)",
    series: names.map((n, k) => ({ name: n, mode: "points", x: rows.map((row, i) => useIdx ? i + 1 : xs[i]), y: rows.map((row) => row.sensitivity[n]), color: COLORS[k % COLORS.length], fill: COLORS[k % COLORS.length], r: 3.2 })) });
  const top = r.observations.ranking.slice(0, 14);
  $("#lev").innerHTML = top.map((i) => { const row = rows[i]; return `<div class="lev"><span>${esc(row.label)}</span><div class="b"><div style="width:${Math.min(100, 100 * row.leverage)}%"></div></div><span>${fmt(row.leverage, 2)}</span></div>`; }).join("");
}

function apiExample() {
  return `curl -X POST ${location.origin}/api/analyze -H 'Content-Type: application/json' -d '{
  "model": {"type": "pin_fin", "options": {"d_mm": 5, "length_mm": 50}},
  "parameters": [{"name": "h", "value": 50, "lower": 1, "upper": 2000, "unit": "W/m²K"},
                 {"name": "k", "value": 16, "lower": 1, "upper": 500, "unit": "W/m·K"},
                 {"name": "T_base", "value": 80, "fixed": true}, {"name": "T_air", "value": 25, "fixed": true}],
  "observations": {"inputs": {"x_mm": [10, 20, 30, 40, 50]}, "sigma": [0.5, 0.5, 0.5, 0.5, 0.5],
                   "observable": ["T", "T", "T", "T", "T"]}}'
# -> {"id": "<job>", "status": "queued", ...}
curl ${location.origin}/api/jobs/<job>                       # status, progress, result (the report)
curl -O ${location.origin}/api/jobs/<job>/download/study.json  # StudyResult for the other apps`;
}

init().catch((e) => { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; });
