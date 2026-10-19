// Physics Design Optimizer: form -> POST /api/studies (a job) -> poll -> comparison, fronts, verified candidates.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function fmt(v, p = 4) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
  return +v.toPrecision(p) + "";
}
const LABELS = { mass_g: "mass [g]", cost: "unit cost", pressure_drop_pa: "pressure drop [Pa]", t_source_c: "source temperature [°C]",
  r_base_k_w: "resistance [K/W]", fin_stress_mpa: "fin stress [MPa]", fin_gap_mm: "fin gap [mm]", Cd: "drag coefficient Cd",
  spar_depth: "spar depth [chord]", Cl: "lift coefficient Cl", Cm: "moment coefficient Cm", aft_thickness: "thickness at 90 % [chord]" };
const lab = (k) => LABELS[k] || k;
const MCOL = { gradient: "#7c3aed", "scipy-slsqp": "#db2777", nsga2: "#0b6bcb", bayesian: "#0f766e", "scipy-de": "#ca8a04", random: "#7a8599" };
let META = null, PRESET = "heatsink", JOB = null;

document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => showTab(b.dataset.tab));
function showTab(t) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + t));
}

async function init() {
  META = await (await fetch("/api/meta")).json();
  const hs = META.presets.heatsink;
  $("#hs-objs").innerHTML = hs.objectives.map((o) => `<label><input type="checkbox" value="${o}" ${["mass_g", "pressure_drop_pa"].includes(o) ? "checked" : ""}> ${esc(lab(o))}</label>`).join("");
  fill();
  renderMethods();
  if (window.renderScope) $("#scope").innerHTML = renderScope(META.scope);
  snippets();
}
function fill() {
  const hs = META.presets.heatsink, af = META.presets.airfoil;
  for (const [k, v] of Object.entries(hs.operating)) if ($("#op-" + k)) $("#op-" + k).value = v;
  for (const [k, v] of Object.entries(hs.limits)) if ($("#lim-" + k)) $("#lim-" + k).value = v;
  for (const [k, v] of Object.entries(af.limits)) $("#af-" + k).value = v;
}
function renderMethods(checked = ["gradient", "scipy-slsqp", "nsga2", "bayesian", "random"]) {
  const avail = META.methods[PRESET];
  $("#methods").innerHTML = Object.entries(avail).map(([m, v]) => `<label><input type="checkbox" value="${m}" ${checked.includes(m) && v.available ? "checked" : ""} ${v.available ? "" : "disabled"}>
    <span><span class="swatch" style="background:${MCOL[m]}"></span>${esc(META.labels[m] || m)}<span class="fam">${esc(v.family)}${v.available ? "" : " — " + esc(v.reason)}</span></span></label>`).join("");
}
document.querySelectorAll("#preset button").forEach((b) => b.onclick = () => setPreset(b.dataset.v));
function setPreset(p) {
  PRESET = p;
  document.querySelectorAll("#preset button").forEach((b) => b.classList.toggle("on", b.dataset.v === p));
  $("#p-heatsink").hidden = p !== "heatsink"; $("#p-airfoil").hidden = p !== "airfoil";
  renderMethods([...document.querySelectorAll("#methods input:checked")].map((i) => i.value));
}

const num = (id) => parseFloat($(id).value);
function body() {
  let spec;
  if (PRESET === "heatsink") {
    const hs = META.presets.heatsink;
    const pick = (pre, obj) => Object.fromEntries(Object.keys(obj).filter((k) => $(`#${pre}-${k}`)).map((k) => [k, num(`#${pre}-${k}`)]));
    spec = { preset: "heatsink", operating: pick("op", hs.operating), limits: pick("lim", hs.limits),
      objectives: [...document.querySelectorAll("#hs-objs input:checked")].map((i) => ({ name: i.value })) };
  } else spec = { preset: "airfoil", limits: { cl_min: num("#af-cl_min"), cm_min: num("#af-cm_min"), aft_min: num("#af-aft_min") } };
  return { spec, methods: [...document.querySelectorAll("#methods input:checked")].map((i) => i.value), budget: parseInt($("#budget").value),
    seeds: $("#seeds").value.split(/[ ,]+/).filter(Boolean).map((s) => parseInt(s)), n_verify: parseInt($("#n_verify").value) };
}
$("#example").onclick = () => { setPreset("heatsink"); fill(); renderMethods(); $("#budget").value = 80; $("#seeds").value = "0"; $("#n_verify").value = 5;
  document.querySelectorAll("#hs-objs input").forEach((i) => i.checked = ["mass_g", "pressure_drop_pa"].includes(i.value)); start(); };
$("#run").onclick = () => start();
$("#cancel").onclick = async () => { if (JOB) await fetch(`/api/jobs/${JOB}/cancel`, { method: "POST" }); };
async function start() {
  $("#example").disabled = $("#run").disabled = true; $("#status").textContent = "Queued…"; $("#prog").hidden = false;
  const r = await fetch("/api/studies", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()) });
  const j = await r.json();
  if (!r.ok) { done(); $("#status").innerHTML = `<span class="err">${esc(typeof j.detail === "string" ? j.detail : j.message || JSON.stringify(j.detail || j))}</span>`; return; }
  JOB = j.id; $("#cancel").hidden = false; poll();
}
function done() { $("#example").disabled = $("#run").disabled = false; $("#prog").hidden = true; $("#cancel").hidden = true; }
async function poll() {
  const j = await (await fetch(`/api/jobs/${JOB}`)).json();
  $("#prog div").style.width = Math.round(100 * (j.progress || 0)) + "%";
  $("#status").textContent = `${j.status}${j.message ? " · " + j.message : ""}`;
  if (j.status === "done") { done(); $("#status").textContent = `Done in ${j.result.seconds} s`; showTab("result"); render(j.result, j.id); window.scrollTo(0, 0); return; }
  if (j.status === "failed" || j.status === "cancelled") { done(); $("#status").innerHTML = `<span class="err">${esc(j.error || j.status)}</span>`; return; }
  setTimeout(poll, 400);
}

// ---------------------------------------------------------------------------------------------- results
const VCLS = { verified: "verified", "rejected by reference": "rejected", "reference failed": "failed", "only surrogate": "surrogate" };
const vbadge = (s) => `<span class="vstat ${VCLS[s] || "surrogate"}">${esc(s)}</span>`;

function render(res, jid) {
  const P = res.problem, objs = P.objectives;
  const nVer = res.candidates.filter((c) => c.verification === "verified").length;
  const nRej = res.candidates.filter((c) => c.verification === "rejected by reference").length;
  const banner = res.verification
    ? `<div class="banner ${nRej ? "warn" : "ok"}">${nVer} of ${res.candidates.length} candidates verified by ${esc(res.reference.name)}${nRej ? `; ${nRej} rejected by it` : ""}. Best front: ${esc(META.labels[res.winner] || res.winner)}.</div>`
    : `<div class="banner warn">No high-fidelity model ran on these candidates: they are surrogate predictions only. Best front: ${esc(META.labels[res.winner] || res.winner)}.</div>`;
  const rows = res.table.map((r) => {
    const best = objs.map((o) => `<td class="num">${fmt(r.best ? r.best[o.name] : null)}</td>`).join("");
    const hv = r.hv_mean === null ? "—" : fmt(r.hv_mean, 3) + (r.seeds > 1 ? ` <span class="meta">(${fmt(r.hv_min, 3)}–${fmt(r.hv_max, 3)})</span>` : "");
    return `<tr class="${r.method === res.winner ? "win" : ""}"><td><span class="swatch" style="background:${MCOL[r.method]}"></span>${esc(META.labels[r.method] || r.method)}<div class="meta">${esc(r.family)}</div></td>
      <td class="num">${r.n_evals}</td><td class="num">${fmt(r.wall_s, 3)} s</td><td class="num">${hv}</td><td class="num">${r.evals_to_95 ?? "not reached"}</td>
      <td>${r.status !== "ok" ? `<span class="err">${esc(r.status)}: ${esc(r.message)}</span>` : r.converged ? '<span class="conv-yes">converged</span>' : '<span class="conv-no">still improving</span>'}</td>${best}</tr>`;
  }).join("");
  const vars = P.space.variables.map((v) => v.name);
  const watch = [...new Set([...objs.map((o) => o.name), ...P.constraints.map((c) => c.name)])];
  $("#result").innerHTML = `<div class="panel">${banner}
    <h3 style="margin-top:0">Methods on the same budget (${res.budget} evaluations per run, seeds ${res.seeds.join(", ")})</h3>
    <div style="overflow-x:auto"><table><thead><tr><th>method</th><th class="num">evaluations</th><th class="num">wall time</th><th class="num">hypervolume</th><th class="num">evals to 95 % of best</th><th>convergence</th>${objs.map((o) => `<th class="num">best ${esc(lab(o.name))}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>
    <p class="hint">Hypervolume of each run's feasible designs, objectives normalised over all runs (higher is better). "Converged": it grew less than 1 % over the run's last quarter. An autograd step counts as one evaluation, a finite-difference gradient as d+1.</p>
    <div class="grid2">
      <div><h3>Pareto fronts found</h3><div id="fronts" class="scatter"></div><div class="legend">${Object.keys(res.fronts).map((m) => `<span><i style="background:${MCOL[m]}"></i>${esc(m)}</span>`).join("")}<span><i style="background:#dbe2ea"></i>feasible designs (all runs)</span><span><i style="background:#1a7f4b"></i>verified candidate</span><span><i style="background:#7a8599"></i>candidate, only surrogate</span></div></div>
      <div><h3>Convergence</h3><div id="conv"></div><p class="hint">Hypervolume of the feasible designs found after k evaluations (first seed).</p></div>
    </div>
    <h3>Candidates (best front over all methods)</h3>
    <div style="overflow-x:auto"><table><thead><tr><th>#</th><th>found by</th>${vars.map((v) => `<th class="num">${esc(v)}</th>`).join("")}${watch.map((k) => `<th class="num">${esc(lab(k))}</th>`).join("")}<th>status</th><th>files</th></tr></thead><tbody>${res.candidates.map((c, k) =>
      `<tr><td>${k + 1}</td><td><span class="swatch" style="background:${MCOL[c.found_by]}"></span>${esc(c.found_by)}</td>${vars.map((v) => `<td class="num">${fmt(c[v])}</td>`).join("")}${watch.map((w) => `<td class="num">${fmt(c[w])}</td>`).join("")}<td>${vbadge(c.verification)}</td>
       <td><a class="link" href="/api/studies/${jid}/design/${k + 1}.stl">STL</a> · <a class="link" href="/api/studies/${jid}/design/${k + 1}.step">STEP</a></td></tr>`).join("")}</tbody></table></div>
    <div class="cards" id="cards"></div>
    <h3>Downloads</h3><div class="files">${res.files.map((f) => `<a class="link" href="/api/studies/${jid}/files/${f}">${esc(f)}</a>`).join("")}</div>
    <p class="hint">candidates.csv holds every candidate's parameters, outputs, verification status and the method that found it; study.json is a StudyResult, parameter_space.json a ParameterSpace.</p>
    ${P.notes.length ? `<div class="warnings">${P.notes.map((n) => `<div>${esc(n)}</div>`).join("")}</div>` : ""}</div>`;
  fronts(res);
  conv(res);
  $("#cards").innerHTML = res.candidates.map((c, k) => card(res, c, k)).join("");
}

function conv(res) {
  const first = Object.entries(res.curves).filter(([k, v]) => v.length && k.endsWith("#" + res.seeds[0]));
  if (!first.length) { $("#conv").innerHTML = '<p class="hint">No feasible design yet.</p>'; return; }
  const xs = [...new Set(first.flatMap(([, v]) => v.map((p) => p[0])))].sort((a, b) => a - b);
  const series = first.map(([k, v]) => {
    const m = k.split("#")[0];
    let j = 0, last = null;
    const y = xs.map((x) => { while (j < v.length && v[j][0] <= x) last = v[j++][1]; return last; });
    return { name: m, y, color: MCOL[m], width: 2 };
  });
  PChart.mount($("#conv"), { height: 250, x: xs, yLabel: "hypervolume", series });
}

function fronts(res) {
  const el = $("#fronts"), objs = res.problem.objectives;
  if (objs.length < 2) { el.innerHTML = '<p class="hint">One objective: see the table.</p>'; return; }
  const [kx, ky] = [objs[0].name, objs[1].name];
  const W = Math.max(300, el.clientWidth || 560), H = 280, L = 58, R = 12, T = 10, B = 34;
  const all = res.points.concat(...Object.values(res.fronts).map((f) => f.map((p) => ({ [kx]: p[0], [ky]: p[1] }))));
  const xs = all.map((p) => p[kx]), ys = all.map((p) => p[ky]);
  // zoom on the fronts: the 98th percentile keeps far, poor designs from squashing them
  const q = (a, f) => { const s = [...a].sort((u, v) => u - v); return s[Math.min(s.length - 1, Math.floor(f * (s.length - 1)))]; };
  let x0 = Math.min(...xs), x1 = q(xs, 0.98), y0 = Math.min(...ys), y1 = q(ys, 0.98);
  const px = (x1 - x0) * 0.05 || 1, py = (y1 - y0) * 0.05 || 1; x0 -= px; x1 += px; y0 -= py; y1 += py;
  const X = (v) => L + (v - x0) / (x1 - x0) * (W - L - R), Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const inside = (x, y) => x >= x0 && x <= x1 && y >= y0 && y <= y1;
  const ticks = (a, b) => { const st = Math.pow(10, Math.floor(Math.log10((b - a) / 4))); const m = [1, 2, 5, 10].find((z) => (b - a) / (z * st) <= 6) * st; const o = []; for (let v = Math.ceil(a / m) * m; v <= b; v += m) o.push(+v.toPrecision(10)); return o; };
  let g = "";
  for (const t of ticks(y0, y1)) g += `<line x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}" stroke="#eef1f5"/><text x="${L - 6}" y="${Y(t) + 3.5}" text-anchor="end">${fmt(t)}</text>`;
  for (const t of ticks(x0, x1)) g += `<text x="${X(t)}" y="${H - B + 14}" text-anchor="middle">${fmt(t)}</text>`;
  for (const p of res.points) if (inside(p[kx], p[ky])) g += `<circle cx="${X(p[kx]).toFixed(1)}" cy="${Y(p[ky]).toFixed(1)}" r="1.8" fill="#dbe2ea"/>`;
  for (const [m, f] of Object.entries(res.fronts)) {
    const s = f.filter((p) => inside(p[0], p[1])).sort((a, b) => a[0] - b[0]);
    if (s.length > 1) g += `<polyline points="${s.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")}" fill="none" stroke="${MCOL[m]}" stroke-width="1.4" opacity=".7"/>`;
    for (const p of s) g += `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="3.2" fill="${MCOL[m]}"/>`;
  }
  for (const c of res.candidates) if (inside(c[kx], c[ky])) g += `<circle cx="${X(c[kx])}" cy="${Y(c[ky])}" r="7.5" fill="none" stroke="${c.verification === "verified" ? "#1a7f4b" : c.verification === "only surrogate" ? "#7a8599" : "#b42318"}" stroke-width="2.2"/>`;
  g += `<line x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" stroke="#c9d1dd"/><line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="#c9d1dd"/>`;
  g += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle" style="font-weight:600">${esc(lab(kx))} (${objs[0].sense})</text><text transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle" style="font-weight:600">${esc(lab(ky))} (${objs[1].sense})</text>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}">${g}</svg>`;
}

function geometry(res, c) {
  if (c.outline) {
    const o = c.outline, W = 200, H = 90, sx = (x) => 8 + x * (W - 16), sy = (y) => H / 2 - y * (W - 16);
    const up = o.x.map((x, i) => `${sx(x).toFixed(1)},${sy(o.upper[i]).toFixed(1)}`), lo = o.x.map((x, i) => `${sx(x).toFixed(1)},${sy(o.lower[i]).toFixed(1)}`).reverse();
    return `<svg class="geo" viewBox="0 0 ${W} ${H}"><polygon points="${up.concat(lo).join(" ")}" fill="#ddd6fe" stroke="#7c3aed" stroke-width="1.2"/><text x="6" y="${H - 6}" style="font-size:9px;fill:#5d6b82">α = ${fmt(c.alpha_deg, 3)}°</text></svg>`;
  }
  // heat sink, seen from the flow direction (to scale)
  const Wd = c.base_width_mm, tb = c.base_thickness_mm, H = c.fin_height_mm, n = Math.round(c.n_fins), t = c.fin_thickness_mm;
  const S = 180 / Math.max(Wd, tb + H), w = 200, h = 20 + (tb + H) * S;
  let g = `<rect x="${10}" y="${h - 10 - tb * S}" width="${Wd * S}" height="${tb * S}" fill="#94a3b8"/>`;
  const pitch = (Wd - t) / Math.max(n - 1, 1);
  for (let i = 0; i < n; i++) g += `<rect x="${10 + i * pitch * S}" y="${h - 10 - (tb + H) * S}" width="${Math.max(t * S, 0.8)}" height="${H * S}" fill="#64748b"/>`;
  return `<svg class="geo" viewBox="0 0 ${w} ${h}">${g}<text x="6" y="12" style="font-size:9px;fill:#5d6b82">${fmt(Wd, 3)} × ${fmt(c.base_depth_mm, 3)} mm, ${n} fins</text></svg>`;
}
function card(res, c, k) {
  const fr = c.feasibility;
  const vr = res.verification ? res.verification.rows.find((r) => r.index === c.index) : null;
  const rows = fr.constraints.map((r) => {
    const lower = r.rule.includes(">=");
    const frac = r.bound && r.value ? Math.max(0, Math.min(1, lower ? r.bound / r.value : r.value / r.bound)) : 0;
    const col = r.status === "violated" ? "#b42318" : r.status === "active" ? "#9a6200" : "#7c3aed";
    return `<span>${esc(r.label)}</span><div class="track" title="${esc(r.rule)}"><div class="fill" style="width:${(100 * frac).toFixed(1)}%;background:${col}"></div></div>
      <span>${fmt(r.value)} / ${fmt(r.bound)} ${esc(r.unit)} <span class="pill ${r.status === "violated" ? "bad" : r.status === "active" ? "warn" : "ok"}">${esc(r.status)}</span></span>`;
  }).join("");
  let conf = "";
  if (vr && vr.reference && Object.keys(vr.reference).length) {
    const watched = new Set([...res.problem.objectives.map((o) => o.name), ...res.problem.constraints.map((x) => x.name)]);
    const keys = Object.keys(vr.rel_error).filter((x) => watched.has(x) && Math.abs(vr.rel_error[x]) > 0);
    conf = `<table style="margin-top:6px"><thead><tr><th>output</th><th class="num">surrogate</th><th class="num">high fidelity</th><th class="num">rel. error</th></tr></thead><tbody>${keys.map((x) =>
      `<tr><td>${esc(lab(x))}</td><td class="num">${fmt(vr.surrogate[x])}</td><td class="num">${fmt(vr.reference[x])}</td><td class="num">${(100 * vr.rel_error[x]).toFixed(2)} %</td></tr>`).join("")}</tbody></table>`;
  }
  return `<div class="cand-card"><h4>Candidate ${k + 1} ${vbadge(c.verification)} <span class="meta">found by ${esc(c.found_by)}${fr.active.length ? " · active: " + fr.active.map(esc).join(", ") : ""}</span></h4>
    <div class="cgrid">${geometry(res, c)}<div><div class="margin">${rows}</div>${conf}</div></div></div>`;
}

function snippets() {
  const host = location.origin;
  $("#c1").textContent = `curl -s -X POST ${host}/api/studies -H 'Content-Type: application/json' \\
  -d '{"spec": {"preset": "heatsink", "objectives": [{"name": "mass_g"}, {"name": "pressure_drop_pa"}]},
       "methods": ["gradient", "nsga2", "bayesian", "random"], "budget": 80, "seeds": [0, 1]}'
curl -s ${host}/api/jobs/<id>                                # progress; result: table, fronts, candidates
curl -s -o design.step ${host}/api/studies/<id>/design/1.step  # candidate 1 as STEP (also .stl)`;
  $("#c2").textContent = `from pinneapple_design.design_optimizer import compare_methods, presets

problem = presets.heatsink_problem("surrogate", objectives=("mass_g", "pressure_drop_pa"))
cmp = compare_methods(problem, ["gradient", "nsga2", "bayesian", "random"], budget=80, seeds=(0, 1))
for row in cmp.table():
    print(row["method"], row["hv_mean"], row["evals_to_95"], row["converged"], row["wall_s"])`;
}

init();
