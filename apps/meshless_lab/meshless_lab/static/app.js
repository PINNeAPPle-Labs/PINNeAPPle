"use strict";
// Meshless Method Lab UI: problem + domain form, node preview, queued refinement study, winners, convergence, maps.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => (v === null || v === undefined || !isFinite(v) ? "—" : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(4) + "");
const tsec = (s) => (s === null || s === undefined || !isFinite(s) ? "—" : s < 1e-3 ? (s * 1e6).toFixed(0) + " µs" : s < 1 ? (s * 1e3).toFixed(1) + " ms" : s.toFixed(2) + " s");
const COLORS = { kansa: "#c2410c", rbf_fd: "#6d28d9", pinn: "#be185d", fd: "#0369a1", fem: "#15803d" };
let META = null, JOB = null, RES = null, DMODE = "preset", CLOUD = null, MAPSEL = "error";

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`));
}

// ------------------------------------------------------------------ form
function syncForm() {
  const k = $("#kind").value, m = $("#mode").value;
  $$(".tonly").forEach((e) => (e.hidden = k !== "heat"));
  $$(".honly").forEach((e) => (e.hidden = k !== "helmholtz"));
  $(".tvar").hidden = k !== "heat";
  $("#l-exact").hidden = m !== "exact"; $("#l-data").hidden = m !== "data";
  if (k === "heat" && m === "data") $$("#l-data .tonly").forEach((e) => (e.hidden = false));
}
["kind", "mode"].forEach((id) => $(`#${id}`).addEventListener("change", () => { syncForm(); checkExpr(); }));
$$("#dseg button").forEach((b) => b.addEventListener("click", () => {
  DMODE = b.dataset.d; $$("#dseg button").forEach((x) => x.classList.toggle("on", x === b));
  ["preset", "polygon", "cloud"].forEach((d) => ($(`#d-${d}`).hidden = d !== DMODE)); preview();
}));
$("#preset").addEventListener("change", preview);
["poly", "holes", "hs"].forEach((id) => $(`#${id}`).addEventListener("change", preview));
$("#cloud").addEventListener("change", (e) => { CLOUD = e.target.files[0]; $("#cname").textContent = CLOUD ? CLOUD.name : "CSV with x, y, boundary"; preview(); });
["exact", "source", "boundary", "k"].forEach((id) => $(`#${id}`).addEventListener("change", checkExpr));

function problem() {
  const p = { kind: $("#kind").value, k: +$("#k").value, alpha: +$("#alpha").value, t_end: +$("#tend").value, n_steps: +$("#nsteps").value };
  if ($("#mode").value === "exact") p.exact = $("#exact").value;
  else { p.source = $("#source").value; p.boundary = $("#boundary").value; if (p.kind === "heat") p.initial = $("#initial").value; }
  return p;
}
function domainSpec() {
  if (DMODE === "preset") return { preset: $("#preset").value };
  if (DMODE === "polygon") {
    const rows = (t) => t.split(/\n/).map((l) => l.trim()).filter(Boolean).map((l) => l.split(/[\s,;]+/).map(Number));
    return { polygon: rows($("#poly").value), holes: rows($("#holes").value) };
  }
  return {};
}
function hs() { return $("#hs").value.split(/[,\s]+/).filter(Boolean).map(Number).filter((x) => x > 0); }
function spec() {
  const s = { problem: problem(), domain: domainSpec(), methods: $$("#methods input:checked").map((x) => x.value),
    pinn_budget_s: +$("#pinnb").value, rbf_fd: { degree: +$("#deg").value }, tolerance: +$("#tol").value };
  if (hs().length) s.h = hs();
  if ($("#eps").value) s.kansa = { eps: +$("#eps").value };
  return s;
}
async function checkExpr() {
  try {
    const r = await fetch("/api/expression", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(problem()) });
    const j = await r.json();
    $("#exprmsg").innerHTML = r.ok ? `<span class="meta">${esc(j.equation)} · ${esc(j.reference)}</span>` : `<span class="err">${esc(j.detail)}</span>`;
  } catch (e) { /* offline */ }
}

// ------------------------------------------------------------------ domain preview
async function preview() {
  const fd = new FormData();
  const h = hs();
  fd.append("spec", JSON.stringify({ domain: domainSpec(), h: h.length ? Math.min(...h) : undefined }));
  if (DMODE === "cloud") { if (!CLOUD) return; fd.append("cloud", CLOUD); }
  const r = await fetch("/api/domain", { method: "POST", body: fd });
  const j = await r.json();
  if (!r.ok) { $("#dinfo").innerHTML = `<span class="err">${esc(j.detail)}</span>`; return; }
  $("#dinfo").textContent = `${j.description || j.name} · ${j.n.toLocaleString("en-US")} nodes at h = ${fmt(j.h)} (${j.boundary.filter(Boolean).length} on the boundary)`;
  drawNodes($("#dview"), j);
}
function frame(cv, pts) {
  const w = cv.clientWidth || 600, h = cv.clientHeight || 360;
  cv.width = w * 2; cv.height = h * 2;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const s = Math.min((w * 2 - 40) / (x1 - x0 || 1), (h * 2 - 40) / (y1 - y0 || 1));
  const ox = (w * 2 - s * (x1 - x0)) / 2, oy = (h * 2 - s * (y1 - y0)) / 2;
  return { ctx: cv.getContext("2d"), X: (x) => ox + (x - x0) * s, Y: (y) => h * 2 - oy - (y - y0) * s, s };
}
function drawNodes(cv, d) {
  const f = frame(cv, d.nodes), ctx = f.ctx;
  for (const line of d.outline || []) { ctx.beginPath(); line.forEach((p, i) => (i ? ctx.lineTo(f.X(p[0]), f.Y(p[1])) : ctx.moveTo(f.X(p[0]), f.Y(p[1])))); ctx.strokeStyle = "#172033"; ctx.lineWidth = 2.5; ctx.stroke(); }
  const r = Math.max(1.5, Math.min(5, 0.28 * d.h * f.s));
  d.nodes.forEach((p, i) => { ctx.fillStyle = d.boundary[i] ? "#be185d" : "#6d28d9"; ctx.beginPath(); ctx.arc(f.X(p[0]), f.Y(p[1]), r, 0, 7); ctx.fill(); });
}

// ------------------------------------------------------------------ jobs
async function submit(url, opts, btn) {
  const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Queued…`;
  $("#status").innerHTML = `<div class="progress"><div id="pbar"></div></div><div id="pmsg" class="meta"></div>`;
  try {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : (j.message || JSON.stringify(j.detail || j)));
    JOB = j.id;
    while (true) {
      await new Promise((res) => setTimeout(res, 600));
      const s = await (await fetch(`/api/jobs/${JOB}`)).json();
      $("#pbar").style.width = `${Math.round(100 * (s.progress || 0))}%`;
      $("#pmsg").textContent = s.message || s.status;
      btn.innerHTML = `<span class="spinner"></span>${s.status === "queued" ? "Queued…" : "Solving…"}`;
      if (s.status === "done") { RES = s.result; render(); showTab("result"); window.scrollTo(0, 0); $("#status").textContent = ""; break; }
      if (s.status === "failed" || s.status === "cancelled") throw new Error(s.error || s.status);
    }
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; }
}
$("#run").addEventListener("click", (e) => {
  const fd = new FormData(); fd.append("spec", JSON.stringify(spec()));
  if (DMODE === "cloud" && CLOUD) fd.append("cloud", CLOUD);
  submit("/api/studies", { method: "POST", body: fd }, e.currentTarget);
});

// ------------------------------------------------------------------ plots
function axes(W, H, m, xr, yr, xlog, ylog, xl, yl) {
  const tx = xlog ? Math.log10 : (v) => v, ty = ylog ? Math.log10 : (v) => v;
  const [x0, x1] = xr.map(tx), [y0, y1] = yr.map(ty);
  const X = (v) => m.l + (tx(v) - x0) / (x1 - x0 || 1) * (W - m.l - m.r), Y = (v) => m.t + (1 - (ty(v) - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const ticks = (a, b, log) => {
    if (log) { const o = []; for (let k = Math.ceil(a); k <= Math.floor(b); k++) o.push(Math.pow(10, k)); return o; }
    const raw = (b - a) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw || 1))), st = [1, 2, 2.5, 5, 10].map((q) => q * mag).find((q) => q >= raw) || raw, o = [];
    for (let v = Math.ceil(a / st) * st; v <= b + st * 1e-9; v += st) o.push(+v.toPrecision(10)); return o;
  };
  let g = "";
  for (const v of ticks(y0, y1, ylog)) g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 5}" y="${Y(v) + 3.5}" text-anchor="end">${ylog ? "1e" + Math.round(Math.log10(v)) : fmt(v)}</text>`;
  for (const v of ticks(x0, x1, xlog)) g += `<text x="${X(v)}" y="${H - m.b + 14}" text-anchor="middle">${xlog ? "1e" + Math.round(Math.log10(v)) : fmt(v)}</text>`;
  g += `<text x="${(m.l + W - m.r) / 2}" y="${H - 3}" text-anchor="middle">${esc(xl)}</text><text x="11" y="${(m.t + H - m.b) / 2}" transform="rotate(-90 11 ${(m.t + H - m.b) / 2})" text-anchor="middle">${esc(yl)}</text>`;
  return { X, Y, g };
}
const lrange = (v) => [Math.min(...v) / 1.8, Math.max(...v) * 1.8];
function legend(ms) { return `<div class="plegend">${ms.map((m) => `<span><i style="background:${COLORS[m]}"></i>${esc(META.methods[m].label)}</span>`).join("")}</div>`; }
function linesSvg(rows, xk, yk, xl, yl, opts = {}) {
  const ok = rows.filter((r) => r.stable && isFinite(r[yk]) && r[yk] > 0 && r[xk] > 0);
  if (!ok.length) return "<p class='meta'>Nothing to plot.</p>";
  const W = 560, H = 290, m = { l: 52, r: 12, t: 10, b: 34 };
  const a = axes(W, H, m, lrange(ok.map((r) => r[xk])), lrange(ok.map((r) => r[yk])), true, true, xl, yl);
  let g = a.g;
  if (opts.tol) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${a.Y(opts.tol)}" y2="${a.Y(opts.tol)}" stroke="#b42318" stroke-dasharray="4 3"/><text x="${W - m.r}" y="${a.Y(opts.tol) - 4}" text-anchor="end" style="fill:#b42318">tolerance</text>`;
  const ms = [...new Set(ok.map((r) => r.method))];
  for (const mt of ms) {
    const s = ok.filter((r) => r.method === mt).sort((p, q) => q.h - p.h);              // refinement order
    if (s.length > 1) g += `<path d="${s.map((r, i) => `${i ? "L" : "M"}${a.X(r[xk])},${a.Y(r[yk])}`).join("")}" fill="none" stroke="${COLORS[mt]}" stroke-width="2"/>`;
    for (const r of s) g += `<circle cx="${a.X(r[xk])}" cy="${a.Y(r[yk])}" r="4" fill="${COLORS[mt]}"><title>${esc(r.label)}, h = ${fmt(r.h)}: ${fmt(r[yk])}</title></circle>`;
    if (opts.slopes && RES.study.convergence[mt] && RES.study.convergence[mt].order !== null && s.length > 1) {
      const e = s[s.length - 1]; g += `<text x="${a.X(e[xk]) + 6}" y="${a.Y(e[yk]) + 4}" style="fill:${COLORS[mt]};font-weight:700">${RES.study.convergence[mt].order.toFixed(2)}</text>`;
    }
  }
  return `<svg class="plot" viewBox="0 0 ${W} ${H}">${g}</svg>${legend(ms)}`;
}
function sweepSvg(sw) {
  const ok = sw.filter((s) => isFinite(s.error) && s.cond);
  if (!ok.length) return "<p class='meta'>No sweep (heat problems, or more than 2,500 nodes at the second spacing).</p>";
  const W = 560, H = 290, m = { l: 52, r: 52, t: 10, b: 34 };
  const a = axes(W, H, m, lrange(ok.map((s) => s.eps)), lrange(ok.map((s) => s.error)), true, true, "shape parameter ε", "error");
  const c0 = Math.log10(Math.min(...ok.map((s) => s.cond))), c1 = Math.log10(Math.max(...ok.map((s) => s.cond))) || c0 + 1;
  const Yc = (v) => m.t + (1 - (Math.log10(v) - c0 + 0.5) / (c1 - c0 + 1)) * (H - m.t - m.b);
  let g = a.g + `<path d="${ok.map((s, i) => `${i ? "L" : "M"}${a.X(s.eps)},${a.Y(s.error)}`).join("")}" fill="none" stroke="${COLORS.kansa}" stroke-width="2"/>`;
  g += `<path d="${ok.map((s, i) => `${i ? "L" : "M"}${a.X(s.eps)},${Yc(s.cond)}`).join("")}" fill="none" stroke="#475569" stroke-width="1.6" stroke-dasharray="4 3"/>`;
  ok.forEach((s) => { g += `<circle cx="${a.X(s.eps)}" cy="${a.Y(s.error)}" r="4" fill="${COLORS.kansa}"/><text x="${a.X(s.eps)}" y="${Yc(s.cond) - 6}" text-anchor="middle" style="font-size:10px">${s.cond.toExponential(0)}</text>`; });
  return `<svg class="plot" viewBox="0 0 ${W} ${H}">${g}</svg><div class="plegend"><span><i style="background:${COLORS.kansa}"></i>error (left)</span><span><i style="background:#475569"></i>condition number (labels)</span></div>`;
}

// ------------------------------------------------------------------ result
function render() {
  const d = RES.study, p = d.problem;
  const wins = `<div class="wins">${d.wins.map((w) => `<div class="win"><div class="c">${esc(w.criterion)}</div>
      <div class="m" style="color:${w.method ? COLORS[w.method] : "var(--muted)"}">${w.method ? esc(d.methods[w.method].label) : "none"}</div>
      <div class="v">${w.value !== null && w.value !== undefined ? `${w.unit === "s" ? tsec(w.value) : fmt(w.value)} <span class="meta">${esc(w.unit === "s" ? "" : w.unit)}</span>` : ""} ${w.detail ? `<span class="meta">${esc(w.detail)}</span>` : ""}</div>
      ${w.runner_up ? `<div class="r">then ${esc(d.methods[w.runner_up].label)}: ${w.unit === "s" ? tsec(w.runner_up_value) : fmt(w.runner_up_value)}</div>` : ""}</div>`).join("")}</div>`;
  const rows = d.rows.map((r) => ({ ...r, total_s: r.total_s ?? null }));
  const table = `<table><thead><tr><th>Method</th><th class="num">h</th><th class="num">Nodes</th><th class="num">Error</th><th class="num">Max</th><th class="num">At nodes</th><th class="num">Setup</th><th class="num">Solve</th><th class="num">Condition</th><th class="num">Noise ampl.</th><th class="num">Non-zeros</th></tr></thead><tbody>
    ${rows.map((r) => `<tr><td><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${COLORS[r.method]};margin-right:6px"></i>${esc(r.label)}${r.stable ? "" : ' <span class="badge fail">unstable</span>'}</td>
      <td class="num">${fmt(r.h)}</td><td class="num">${(r.n_nodes || 0).toLocaleString("en-US")}</td><td class="num">${fmt(r.error)}</td><td class="num">${fmt(r.error_max)}</td><td class="num">${fmt(r.error_nodes)}</td>
      <td class="num">${tsec(r.setup_s)}</td><td class="num">${tsec(r.solve_s)}</td><td class="num">${r.cond ? r.cond.toExponential(1) : "—"}</td><td class="num">${fmt(r.noise_amplification)}</td><td class="num">${r.nnz ? r.nnz.toLocaleString("en-US") : "—"}</td></tr>`).join("")}</tbody></table>
    ${d.skipped.length ? `<div class="skips">Not run: ${d.skipped.map((s) => `${esc(s.method)}${s.h ? ` h=${fmt(s.h)}` : ""} (${esc(s.reason)})`).join("; ")}</div>` : ""}`;
  const notes = `<ul class="notes">${Object.entries(d.notes).map(([m, ns]) => ns.map((n) => `<li><b style="color:${COLORS[m]}">${esc(d.methods[m].label)}</b>: ${esc(n)}</li>`).join("")).join("")}</ul>`;
  const checks = `<div class="checklist">${RES.checks.map((c) => `<div><span class="badge ${c.status === "PASS" ? "pass" : c.status === "FAIL" ? "fail" : "warn"}">${c.status}</span><span>${esc(c.name)}: <b>${fmt(+c.value)}</b> <span class="meta">${esc(c.detail)}</span></span></div>`).join("")}</div>`;
  const dl = (n, l) => `<a href="/api/jobs/${JOB}/files/${n}" download>${l}</a>`;
  const ref = d.reference;
  $("#result").innerHTML = `<div class="panel">
    <div class="toolbar"><h2 style="margin:0">${esc(p.equation)} on the ${esc(d.domain.description || d.domain.name)}</h2><span class="dl">${dl("study_result.json", "StudyResult (JSON)")}${dl("solutions.csv", "Solutions (CSV)")}${dl("study.json", "Study (JSON)")}</span></div>
    <div class="eq">Reference: ${ref.kind === "manufactured solution" ? `u = <code>${esc(ref.expression)}</code>` : `${esc(ref.kind)}, h = ${fmt(ref.h)}, ${ref.n_nodes} nodes, estimated error ${fmt(ref.estimated_error)} (${esc(ref.note)})`}. Errors: relative L2 at ${d.fields.points.length} independent points${p.kind === "heat" ? ` at t = ${p.t_end}` : ""}. Study: ${tsec(RES.wall_s)}.</div>
    <h3 style="margin-top:4px">Where each wins</h3>${wins}
    <div class="grid2c">
      <div class="card"><h4>Convergence</h4><div class="sub">Error against node spacing; the number is the observed order</div>${linesSvg(rows, "h", "error", "node spacing h", "relative L2 error", { slopes: true })}</div>
      <div class="card"><h4>Error against cost</h4><div class="sub">Setup + solve time (PINN: training budget); left and low is better</div>${linesSvg(rows.map((r) => ({ ...r, total: r.total_s || (r.setup_s + r.solve_s) })), "total", "error", "wall time [s]", "relative L2 error", { tol: d.tolerance })}</div>
      <div class="card"><h4>Conditioning</h4><div class="sub">1-norm condition number of the system against the number of unknowns</div>${linesSvg(rows.filter((r) => r.cond), "n_nodes", "cond", "unknowns", "condition number")}</div>
      <div class="card"><h4>Kansa shape parameter</h4><div class="sub">Accuracy against conditioning at one spacing: the trade-off principle</div>${sweepSvg(d.eps_sweep)}</div>
    </div>
    <h3>Error maps <span class="meta">(finest spacing of each method, at the evaluation points)</span>
      <span class="seg" style="display:inline-flex;width:auto;margin-left:10px"><button type="button" data-v="error" class="${MAPSEL === "error" ? "on" : ""}">error</button><button type="button" data-v="values" class="${MAPSEL === "values" ? "on" : ""}">solution</button></span></h3>
    <div class="maps" id="maps"></div>
    <h3>All runs</h3>${table}
    <h3>Notes</h3>${notes}
    <h3>Checks</h3>${checks}
    <div>${window.renderScope ? renderScope(META.scope) : ""}</div>
    <h3>Limits</h3><ul class="meta">${RES.limits.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></div>`;
  $$("#result .seg button").forEach((b) => b.addEventListener("click", () => { MAPSEL = b.dataset.v; render(); }));
  drawMaps();
}
const VIR = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
const DIV = [[33, 102, 172], [146, 197, 222], [247, 247, 247], [244, 165, 130], [178, 24, 43]];
function cmap(st, t) { t = Math.min(1, Math.max(0, t)); const k = Math.min(st.length - 2, Math.floor(t * (st.length - 1))), f = t * (st.length - 1) - k; return st[k].map((c, i) => Math.round(c + f * (st[k + 1][i] - c))); }
function drawMaps() {
  const F = RES.study.fields, P = F.points, ms = Object.keys(F.methods);
  $("#maps").innerHTML = ms.map((m) => { const v = F.methods[m][MAPSEL]; const e = Math.max(...v.map(Math.abs)); return `<div><div class="cap"><span style="color:${COLORS[m]}">${esc(RES.study.methods[m].label)}</span><span class="meta">h = ${fmt(F.methods[m].h)} · ${MAPSEL === "error" ? `max |e| ${fmt(e)}` : ""}</span></div><canvas id="mp-${m}"></canvas></div>`; }).join("");
  const all = ms.flatMap((m) => F.methods[m].values).concat(F.truth), lo = Math.min(...F.truth), hi = Math.max(...F.truth);
  for (const m of ms) {
    const cv = $(`#mp-${m}`); cv.style.height = "220px";
    const f = frame(cv, P), ctx = f.ctx, v = F.methods[m][MAPSEL], em = Math.max(...v.map(Math.abs)) || 1;
    const r = Math.max(3, Math.min(9, 1.4 * Math.sqrt((cv.width * cv.height) / P.length) / 2));
    P.forEach((p, i) => {
      const c = MAPSEL === "error" ? cmap(DIV, 0.5 + v[i] / (2 * em)) : cmap(VIR, (v[i] - lo) / (hi - lo || 1));
      ctx.fillStyle = `rgb(${c})`; ctx.beginPath(); ctx.arc(f.X(p[0]), f.Y(p[1]), r, 0, 7); ctx.fill();
    });
    void all;
  }
}

// ------------------------------------------------------------------ boot
fetch("/api/meta").then((r) => r.json()).then((m) => {
  META = m;
  $("#preset").innerHTML = Object.entries(m.domains).map(([k, d]) => `<option value="${k}" ${k === "lshape" ? "selected" : ""}>${esc(d.description || k)}</option>`).join("");
  $("#methods").innerHTML = Object.entries(m.methods).map(([k, d]) => `<label><input type="checkbox" value="${k}" checked> <span style="color:${COLORS[k]}">●</span> ${esc(d.label)}</label>`).join("");
  $("#examples").innerHTML = m.examples.map((e) => `<button type="button" data-id="${e.id}"><b>${esc(e.title)}</b>${esc(e.note)}</button>`).join("");
  $$("#examples button").forEach((b) => b.addEventListener("click", () => submit(`/api/examples/${b.dataset.id}`, { method: "POST" }, b)));
  syncForm(); preview(); checkExpr();
  const o = location.origin;
  $("#c1").textContent = `curl -F 'spec={"problem":{"kind":"poisson","exact":"sin(pi*x)*sin(pi*y)"},"domain":{"preset":"lshape"},"h":[0.1,0.05,0.025]}' ${o}/api/studies
curl ${o}/api/jobs/<id>                                   # progress, then the study
curl -O ${o}/api/jobs/<id>/files/study_result.json        # StudyResult contract
curl -F 'spec={"problem":{"kind":"poisson","exact":"x*y"}}' -F cloud=@nodes.csv ${o}/api/studies   # x, y, boundary`;
  $("#c2").textContent = `from pinneapple_simulation.meshless import Problem, domain, meshless_study

res = meshless_study(Problem("poisson", exact="sin(pi*x)*cos(2*y) + x*y"), domain("lshape"), h=[0.1, 0.05, 0.025])
for w in res.wins:
    print(w["criterion"], w["method"], w["value"])`;
});
