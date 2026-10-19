"use strict";
// Reduced Order Model Lab UI: upload, queue a study, poll the job, show trade-offs, reconstructions and exports.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => (v === null || v === undefined || !isFinite(v) ? "—" : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(4) + "");
const pct = (v) => (v === null || v === undefined || !isFinite(v) ? "—" : (100 * v < 0.01 && v > 0 ? (100 * v).toExponential(1) : (100 * v).toFixed(100 * v < 10 ? 2 : 1)) + " %");
const tsec = (s) => (s === null || s === undefined || !isFinite(s) ? "—" : s < 1e-3 ? (s * 1e6).toFixed(s < 1e-5 ? 1 : 0) + " µs" : s < 1 ? (s * 1e3).toFixed(1) + " ms" : s.toFixed(s < 10 ? 2 : 1) + " s");
const xfac = (v) => (v === null || v === undefined || !isFinite(v) ? "—" : v >= 1e4 ? Math.round(v).toLocaleString("en-US") + "×" : (+v.toPrecision(3)) + "×");
const COLORS = { persistence: "#94a3b8", nearest: "#94a3b8", pod_rbf: "#0f766e", pod_gpr: "#2563eb", dmd: "#c2410c", opinf: "#7c3aed", neural: "#be185d" };
let FILES = [], META = null, JOB = null, RES = null, SEL = null, KIND = null;

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`));
}

// ------------------------------------------------------------------ inputs
async function filesFromDrop(dt) {
  const out = [];
  const walk = (entry, path) => new Promise((res) => {
    if (entry.isFile) entry.file((f) => { f.relPath = path + f.name; out.push(f); res(); });
    else if (entry.isDirectory) {
      const rd = entry.createReader(), all = [];
      const read = () => rd.readEntries(async (ents) => { if (!ents.length) { for (const en of all) await walk(en, path + entry.name + "/"); res(); } else { all.push(...ents); read(); } });
      read();
    } else res();
  });
  const items = [...(dt.items || [])].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  if (!items.length) return [...dt.files];
  for (const en of items) await walk(en, "");
  return out;
}
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async (e) => addFiles(await filesFromDrop(e.dataTransfer)));
$("#files").addEventListener("change", (e) => { addFiles([...e.target.files]); e.target.value = ""; });
function addFiles(list) {
  for (const f of list) { const n = f.relPath || f.webkitRelativePath || f.name; if (!FILES.some((x) => x.n === n)) FILES.push({ f, n }); }
  renderFiles(); inspect();
}
function renderFiles() {
  $("#filelist").innerHTML = FILES.slice(0, 4).map((x, i) => `<div><span>${esc(x.n)}</span><button data-i="${i}">×</button></div>`).join("") +
    (FILES.length > 4 ? `<div><span>+ ${FILES.length - 4} files</span><button data-i="all">clear</button></div>` : "");
  $$("#filelist button").forEach((b) => b.onclick = () => { if (b.dataset.i === "all") FILES = []; else FILES.splice(+b.dataset.i, 1); renderFiles(); inspect(); });
  $("#run").disabled = !FILES.length;
}
function formData() {
  const fd = new FormData();
  FILES.forEach((x) => fd.append("files", x.f, x.n));
  fd.append("param_columns", $("#pcols").value);
  return fd;
}
async function inspect() {
  $("#detected").hidden = true;
  if (!FILES.length) return;
  try {
    const r = await fetch("/api/inspect", { method: "POST", body: formData() });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail);
    KIND = j.kind;
    const f = j.fields.map((x) => `${esc(x.name)}${x.components > 1 ? ` (${x.components} comp.)` : ""}`).join(", ");
    $("#detected").innerHTML = `<b>${j.n_snapshots} snapshots</b> × ${j.n_values.toLocaleString("en-US")} values · ${j.kind === "time" ? `times ${fmt(j.times[0])} … ${fmt(j.times[1])}` : `parameters ${esc(j.param_names.join(", "))}`}<br>fields: ${f}`;
    $("#detected").hidden = false;
    renderMethods(j.kind);
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}
$("#pcols").addEventListener("change", inspect);
function renderMethods(kind) {
  const ms = META.methods_by_data[kind || "time"];
  $("#methods").innerHTML = ms.map((m) => `<label><input type="checkbox" value="${m}" checked> ${esc(META.methods[m])}</label>`).join("");
}
function options() {
  const ranks = $("#ranks").value.split(/[,\s]+/).filter(Boolean).map(Number).filter((x) => x > 0);
  const o = { methods: $$("#methods input:checked").map((x) => x.value), split: $("#split").value, test_fraction: +$("#tfrac").value,
    budget_s: +$("#budget").value, tolerance: (+$("#tol").value || 0) / 100, log_params: $("#logp").value.split(/[,\s]+/).filter(Boolean),
    amplitude_split: $("#split_amp").checked };
  if (ranks.length) o.ranks = ranks;
  if ($("#hf").value) o.hf_seconds = +$("#hf").value;
  return o;
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
      btn.innerHTML = `<span class="spinner"></span>${s.status === "queued" ? "Queued…" : "Building ROMs…"}`;
      if (s.status === "done") { RES = s.result; SEL = RES.best_key; render(); showTab("result"); window.scrollTo(0, 0); $("#status").textContent = ""; break; }
      if (s.status === "failed" || s.status === "cancelled") throw new Error(s.error || s.status);
    }
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; if (btn.id === "run") $("#run").disabled = !FILES.length; }
}
$("#run").addEventListener("click", (e) => {
  const fd = formData(); fd.append("options", JSON.stringify(options()));
  submit("/api/studies", { method: "POST", body: fd }, e.currentTarget);
});

// ------------------------------------------------------------------ plots (small SVG)
function axes(W, H, m, xr, yr, xlog, ylog, xl, yl) {
  const tx = xlog ? Math.log10 : (v) => v, ty = ylog ? Math.log10 : (v) => v;
  const [x0, x1] = xr.map(tx), [y0, y1] = yr.map(ty);
  const X = (v) => m.l + (tx(v) - x0) / (x1 - x0 || 1) * (W - m.l - m.r), Y = (v) => m.t + (1 - (ty(v) - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  let g = "";
  const ticks = (a, b, log) => {
    if (log) { const o = []; for (let k = Math.floor(a); k <= Math.ceil(b); k++) o.push(Math.pow(10, k)); return o.filter((v) => Math.log10(v) >= a - 1e-9 && Math.log10(v) <= b + 1e-9); }
    const raw = (b - a) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw || 1))), st = [1, 2, 2.5, 5, 10].map((q) => q * mag).find((q) => q >= raw) || raw, o = [];
    for (let v = Math.ceil(a / st) * st; v <= b + st * 1e-9; v += st) o.push(+v.toPrecision(10)); return o;
  };
  for (const v of ticks(y0, y1, ylog)) g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 5}" y="${Y(v) + 3.5}" text-anchor="end">${ylog ? "1e" + Math.round(Math.log10(v)) : fmt(v)}</text>`;
  for (const v of ticks(x0, x1, xlog)) g += `<text x="${X(v)}" y="${H - m.b + 14}" text-anchor="middle">${xlog ? "1e" + Math.round(Math.log10(v)) : fmt(v)}</text>`;
  g += `<text x="${(m.l + W - m.r) / 2}" y="${H - 3}" text-anchor="middle">${esc(xl)}</text><text x="11" y="${(m.t + H - m.b) / 2}" transform="rotate(-90 11 ${(m.t + H - m.b) / 2})" text-anchor="middle">${esc(yl)}</text>`;
  return { X, Y, g };
}
const range = (vs, log) => { let a = Math.min(...vs), b = Math.max(...vs); if (log) { a /= 1.6; b *= 1.6; } else { const p = (b - a) * 0.06 || 1; a -= p; b += p; } return [a, b]; };
function legend(items) { return `<div class="plegend">${items.map(([n, c, line]) => `<span><i class="${line ? "line" : ""}" style="background:${c}"></i>${esc(n)}</span>`).join("")}</div>`; }

function tradeoffSvg(d) {
  const rows = d.rows.filter((r) => isFinite(r.error_mean) && r.error_mean > 0);
  if (!rows.length) return "<p class='meta'>No finite errors.</p>";
  const W = 560, H = 300, m = { l: 52, r: 12, t: 10, b: 34 };
  const a = axes(W, H, m, range(rows.map((r) => r.infer_s), true), range(rows.map((r) => r.error_mean), true), true, true, "inference time per snapshot [s]", "held-out error");
  let g = a.g;
  const front = d.pareto.map((p) => rows.find((r) => r.method === p.method && r.rank === p.rank)).filter(Boolean);
  if (front.length > 1) g += `<path d="${front.map((r, i) => `${i ? "L" : "M"}${a.X(r.infer_s)},${a.Y(r.error_mean)}`).join("")}" fill="none" stroke="#0f766e55" stroke-width="6" stroke-linejoin="round"/>`;
  if (d.tolerance) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${a.Y(d.tolerance)}" y2="${a.Y(d.tolerance)}" stroke="#b42318" stroke-dasharray="4 3"/><text x="${W - m.r}" y="${a.Y(d.tolerance) - 4}" text-anchor="end" style="fill:#b42318">tolerance ${pct(d.tolerance)}</text>`;
  for (const r of rows) {
    const k = r.key === SEL;
    g += `<circle cx="${a.X(r.infer_s)}" cy="${a.Y(r.error_mean)}" r="${k ? 6.5 : 4.5}" fill="${COLORS[r.method] || "#333"}" stroke="${k ? "#111" : "#fff"}" stroke-width="${k ? 2 : 1}" data-key="${r.key}" style="cursor:pointer"><title>${esc(r.label)}${r.rank ? `, rank ${r.rank}` : ""}: ${pct(r.error_mean)}, ${tsec(r.infer_s)} per snapshot</title></circle>`;
    if (r.rank) g += `<text x="${a.X(r.infer_s) + 7}" y="${a.Y(r.error_mean) + 3}" style="font-size:10px">${r.rank}</text>`;
  }
  const ms = [...new Set(rows.map((r) => r.method))];
  return `<svg class="plot" viewBox="0 0 ${W} ${H}" id="trade">${g}</svg>${legend(ms.map((x) => [META.methods[x], COLORS[x]]).concat([["Pareto front", "#0f766e55", true]]))}`;
}
function rankSvg(d) {
  const rk = d.rows.filter((r) => r.rank && isFinite(r.error_mean) && r.error_mean > 0);
  const pr = d.projection.filter((p) => p.error_mean > 0);
  const flat = d.rows.filter((x) => !x.rank && isFinite(x.error_mean) && x.error_mean > 0);
  const all = rk.map((r) => r.error_mean).concat(pr.map((p) => p.error_mean), flat.map((r) => r.error_mean));
  if (!all.length) return "";
  const ranks = rk.map((r) => r.rank).concat(pr.map((p) => p.rank));
  const W = 560, H = 300, m = { l: 52, r: 12, t: 16, b: 34 };
  const a = axes(W, H, m, [Math.max(0.8, Math.min(...ranks) / 1.3), Math.max(...ranks) * 1.3], range(all, true), true, true, "rank (POD modes)", "held-out error");
  let g = a.g;
  g += `<path d="${pr.map((p, i) => `${i ? "L" : "M"}${a.X(p.rank)},${a.Y(p.error_mean)}`).join("")}" fill="none" stroke="#111" stroke-dasharray="5 4" stroke-width="1.5"/>`;
  const ms = [...new Set(rk.map((r) => r.method))];
  for (const mth of ms) {
    const s = rk.filter((r) => r.method === mth).sort((x, y) => x.rank - y.rank);
    g += `<path d="${s.map((r, i) => `${i ? "L" : "M"}${a.X(r.rank)},${a.Y(r.error_mean)}`).join("")}" fill="none" stroke="${COLORS[mth]}" stroke-width="2"/>`;
    for (const r of s) g += `<circle cx="${a.X(r.rank)}" cy="${a.Y(r.error_mean)}" r="3.5" fill="${COLORS[mth]}"><title>${esc(r.label)} rank ${r.rank}: ${pct(r.error_mean)}</title></circle>`;
  }
  for (const r of d.rows.filter((x) => !x.rank && isFinite(x.error_mean) && x.error_mean > 0))
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${a.Y(r.error_mean)}" y2="${a.Y(r.error_mean)}" stroke="${COLORS[r.method]}" stroke-dasharray="2 3"/><text x="${m.l + 4}" y="${a.Y(r.error_mean) - 4}" style="fill:${COLORS[r.method]}">${esc(r.label)}</text>`;
  return `<svg class="plot" viewBox="0 0 ${W} ${H}">${g}</svg>${legend(ms.map((x) => [META.methods[x], COLORS[x], true]).concat([["POD projection (the floor at that rank)", "#111", true]]))}`;
}
function energySvg(d) {
  const sv = d.singular_values.filter((v) => v > 1e-10 * d.singular_values[0]);
  if (!sv.length) return "";
  const W = 560, H = 260, m = { l: 52, r: 46, t: 10, b: 34 };
  const n = sv.length, rel = sv.map((v) => v / sv[0]);
  const a = axes(W, H, m, [0.8, n * 1.2], range(rel, true), true, true, "mode", "σ / σ₁");
  let g = a.g + `<path d="${rel.map((v, i) => `${i ? "L" : "M"}${a.X(i + 1)},${a.Y(v)}`).join("")}" fill="none" stroke="#0f766e" stroke-width="2"/>`;
  const E = d.energy, Ye = (v) => m.t + (1 - v) * (H - m.t - m.b);
  g += `<path d="${E.slice(0, n).map((v, i) => `${i ? "L" : "M"}${a.X(i + 1)},${Ye(v)}`).join("")}" fill="none" stroke="#c2410c" stroke-width="1.6" stroke-dasharray="4 3"/>`;
  for (const v of [0, 0.5, 1]) g += `<text x="${W - m.r + 5}" y="${Ye(v) + 3}" style="fill:#c2410c">${Math.round(v * 100)}%</text>`;
  for (const lvl of [0.99, 0.999]) { const k = E.findIndex((v) => v >= lvl); if (k >= 0) g += `<line x1="${a.X(k + 1)}" x2="${a.X(k + 1)}" y1="${m.t}" y2="${H - m.b}" stroke="#c2410c44"/><text x="${a.X(k + 1) + 3}" y="${m.t + 10 + (lvl > 0.995 ? 12 : 0)}" style="fill:#c2410c;font-size:10px">${lvl * 100}% at ${k + 1}</text>`; }
  return `<svg class="plot" viewBox="0 0 ${W} ${H}">${g}</svg>${legend([["singular values (log, left)", "#0f766e", true], ["cumulative energy (right)", "#c2410c", true]])}`;
}
function perSnapshotSvg(d) {
  const best = d.rows.find((r) => r.key === RES.best_key), sel = d.rows.find((r) => r.key === SEL);
  const base = d.rows.find((r) => r.method === "persistence" || r.method === "nearest");
  const series = [best, sel, base].filter((r, i, arr) => r && arr.indexOf(r) === i && r.errors.every((v) => isFinite(v)));
  if (!series.length) return "";
  const x = d.mode === "time" ? RES.test_inputs : RES.test_inputs.map((_, i) => i + 1);
  const vals = series.flatMap((r) => r.errors).filter((v) => v > 0);
  const W = 560, H = 260, m = { l: 52, r: 12, t: 10, b: 34 };
  const a = axes(W, H, m, range(x, false), range(vals, true), false, true, d.mode === "time" ? "time of the held-out snapshot" : "held-out design", "error");
  let g = a.g;
  for (const r of series) {
    const c = COLORS[r.method];
    g += `<path d="${r.errors.map((v, i) => `${i ? "L" : "M"}${a.X(x[i])},${a.Y(Math.max(v, 1e-12))}`).join("")}" fill="none" stroke="${c}" stroke-width="2" ${r === base ? 'stroke-dasharray="4 3"' : ""}/>`;
  }
  return `<svg class="plot" viewBox="0 0 ${W} ${H}">${g}</svg>${legend(series.map((r) => [`${r.label}${r.rank ? ` r = ${r.rank}` : ""}`, COLORS[r.method], true]))}`;
}

// ------------------------------------------------------------------ result
const dlLink = (path, label) => `<a href="/api/jobs/${JOB}/files/${path}" download>${label}</a>`;
function render() {
  const d = RES.study, best = d.rows.find((r) => r.key === RES.best_key);
  const tol = d.tolerance, ok = best && tol && best.error_mean <= tol;
  const hero = best ? `<div class="hero ${tol && !ok ? "fail" : ""}"><div class="big">${pct(best.error_mean)}</div><div class="txt">
      <b>${esc(best.label)}${best.rank ? `, rank ${best.rank}` : ""}</b> ${tol ? (ok ? `is the fastest model under the ${pct(tol)} tolerance` : `has the lowest error, above the ${pct(tol)} tolerance`) : "has the lowest error"}:
      ${tsec(best.infer_s)} per snapshot${best.speedup ? `, <b>${xfac(best.speedup)}</b> faster than the full-order model (${tsec(d.hf_seconds)} per snapshot)` : ""}.
      <div class="meta">Held-out: ${d.split.n_test} of ${d.split.n_train + d.split.n_test} snapshots (${esc(d.split.how)}); error = ${esc(d.metric)}.</div></div></div>` : `<div class="hero fail"><div class="txt">No model produced a finite prediction.</div></div>`;
  const kp = (l, v, s) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s || ""}</div></div>`;
  const kpis = `<div class="kpis">${kp("Snapshots", `${d.split.n_train} + ${d.split.n_test}`, "train + held-out")}${kp("Values per snapshot", RES.snapshots.n_values.toLocaleString("en-US"), RES.snapshots.fields.map((f) => f.name).join(", "))}
    ${kp("Best held-out error", best ? pct(best.error_mean) : "—", best ? `max ${pct(best.error_max)}` : "")}${kp("Inference", best ? tsec(best.infer_s) : "—", "per snapshot, this server")}
    ${kp("Speed-up", best && best.speedup ? xfac(best.speedup) : "—", d.hf_seconds ? `vs ${tsec(d.hf_seconds)} full-order` : "full-order time not given")}${kp("Study", tsec(RES.wall_s), `${d.rows.length} models, neural budget ${d.budget_s} s`)}</div>`;
  const fields = Object.keys((best || d.rows[0] || {}).field_errors || {});
  const table = `<table class="rows"><thead><tr><th>Method</th><th class="num">Rank</th><th class="num">Mean error</th><th class="num">Max</th>${fields.length > 1 ? fields.map((f) => `<th class="num">${esc(f)}</th>`).join("") : ""}<th class="num">Floor</th><th class="num">Fit</th><th class="num">Inference</th><th class="num">Speed-up</th><th>Export</th></tr></thead><tbody>
    ${d.rows.map((r) => { const fl = d.projection.find((p) => p.rank === r.rank); return `<tr data-key="${r.key}" class="${r.key === SEL ? "sel" : ""}">
      <td class="${r.key === RES.best_key ? "best" : ""}"><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${COLORS[r.method]};margin-right:6px"></i>${esc(r.label)}${r.diverged ? ' <span class="badge fail">diverged</span>' : ""}</td>
      <td class="num">${r.rank ?? "—"}</td><td class="num">${pct(r.error_mean)}</td><td class="num">${pct(r.error_max)}</td>
      ${fields.length > 1 ? fields.map((f) => `<td class="num">${pct(r.field_errors[f])}</td>`).join("") : ""}
      <td class="num">${fl ? pct(fl.error_mean) : "—"}</td><td class="num">${tsec(r.fit_s)}</td><td class="num">${tsec(r.infer_s)}</td><td class="num">${xfac(r.speedup)}</td>
      <td><span class="dl">${dlLink(`models/${r.key}.rom.npz`, ".rom.npz")}${r.onnx ? `<a href="/api/jobs/${JOB}/onnx/${r.key}" download>ONNX</a>` : "<span title='no ONNX graph for this method'>ONNX</span>"}${dlLink(`models/${r.key}.modelref.json`, "ModelRef")}</span></td></tr>`; }).join("")}</tbody></table>
    ${d.skipped.length ? `<div class="skips">Not run: ${d.skipped.map((s) => `${esc(META.methods[s.method] || s.method)}${s.rank ? ` r=${s.rank}` : ""} (${esc(s.reason)})`).join("; ")}</div>` : ""}
    ${d.rows.filter((r) => r.budget).map((r) => `<div class="skips">Neural surrogate: ${r.budget.steps} Adam steps in ${tsec(r.budget.used_s)} (budget ${r.budget.wall_s} s, stopped by ${esc(r.budget.stopped_by)}).</div>`).join("")}`;
  const checks = `<div class="checklist">${RES.checks.map((c) => `<div><span class="badge ${c.status === "PASS" ? "pass" : c.status === "FAIL" ? "fail" : "warn"}">${c.status}</span><span>${esc(c.name)}: <b>${fmt(+c.value)}</b>${c.threshold !== null ? ` (threshold ${fmt(+c.threshold)})` : ""} <span class="meta">${esc(c.detail)}</span></span></div>`).join("")}</div>`;
  $("#result").innerHTML = `<div class="panel">
    <div class="toolbar"><h2 style="margin:0">${esc(RES.name)}</h2><span class="dl">${dlLink("study_result.json", "StudyResult (JSON)")}</span></div>
    ${hero}${kpis}
    <div class="grid2c" style="margin-top:16px">
      <div class="card"><h4>Error vs speed</h4><div class="sub">Every model on held-out snapshots; click a point to inspect it</div>${tradeoffSvg(d)}</div>
      <div class="card"><h4>Error vs rank</h4><div class="sub">Dashed: the best any model in that POD basis can do</div>${rankSvg(d)}</div>
      <div class="card"><h4>Energy of the snapshots</h4><div class="sub">Singular values of the training snapshots (centred)</div>${energySvg(d)}</div>
      <div class="card"><h4>Error per held-out snapshot</h4><div class="sub">Best, selected and baseline model</div>${perSnapshotSvg(d)}</div>
    </div>
    <h3>Models</h3>${table}
    <h3>Reconstruction of a held-out snapshot</h3>
    <div class="toolbar"><div>Model <select id="rmodel">${d.rows.filter((r) => !r.diverged).map((r) => `<option value="${r.key}" ${r.key === SEL ? "selected" : ""}>${esc(r.label)}${r.rank ? ` r = ${r.rank}` : ""}</option>`).join("")}</select>
      snapshot <select id="rsnap">${RES.test_inputs.map((q, i) => `<option value="${i}">${d.mode === "time" ? `t = ${fmt(q)}` : `#${i + 1}: ${q.map(fmt).join(", ")}`}</option>`).join("")}</select>
      field <select id="rfield">${RES.snapshots.fields.map((f) => `<option>${esc(f.name)}</option>`).join("")}</select></div><span class="meta" id="rinfo"></span></div>
    <div class="recon" id="recon"></div>
    ${RES.snapshots.mesh ? `<h3>The held-out ${d.mode === "time" ? "snapshot" : "design"} on its body (3-D)</h3>
    <p class="meta">Same model and snapshot as above. ${esc(RES.snapshots.meta.full_order || "Full-order")}: the stored result; ROM: the prediction at its inputs; the error is |ROM − ${esc(RES.snapshots.meta.full_order || "full-order")}| at each node. Every number below is computed from these two arrays.</p>
    <div class="v3wrap"><div id="v3d"></div><div class="v3side" id="v3sum"></div></div>` : ""}
    <h3>Checks</h3>${checks}
    <div>${window.renderScope ? renderScope(META.scope) : ""}</div>
    <h3>Limits</h3><ul class="meta">${RES.limits.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></div>`;
  $$("#trade circle").forEach((c) => c.addEventListener("click", () => { SEL = c.dataset.key; render(); }));
  $$("table.rows tbody tr").forEach((tr) => tr.addEventListener("click", (e) => { if (e.target.closest("a")) return; SEL = tr.dataset.key; render(); }));
  ["rmodel", "rsnap", "rfield"].forEach((id) => $(`#${id}`).addEventListener("change", () => { if (id === "rmodel") SEL = $("#rmodel").value; recon(); if (id !== "rfield") view3d(); }));
  recon(); view3d();
}

// ------------------------------------------------------------------ 3-D view of the held-out snapshot
let V3 = null;
const whenViewer = () => new Promise((res) => window.FieldView3D ? res() : window.addEventListener("fieldview3d-ready", res, { once: true }));
async function view3d() {
  if (!$("#v3d")) return;
  const key = $("#rmodel").value, i = $("#rsnap").value;
  $("#v3sum").innerHTML = `<p class="meta">computing the view…</p>`;
  const r = await fetch(`/api/jobs/${JOB}/view3d?model=${encodeURIComponent(key)}&snapshot=${i}`);
  const j = await r.json();
  if (!r.ok) { $("#v3d").innerHTML = ""; $("#v3sum").innerHTML = `<span class="err">${esc(j.detail)}</span>`; return; }
  await whenViewer();
  const prev = V3 && V3.field ? V3.field.name : null;
  const start = prev && j.view.fields.some((f) => f.name === prev) ? prev : (j.view.fields.find((f) => f.kind === "error" && /mises|stress/i.test(f.name)) || j.view.fields.find((f) => f.kind === "error")).name;
  V3 = new FieldView3D($("#v3d"), { meta: j.view, glb: j.glb, field: start, height: 470, onField: (f) => summary3d(j, f) });
}
function summary3d(j, f) {
  const base = f.name.split(":")[0], e = j.errors[base];
  if (!e) return;
  const u = e.unit ? ` ${esc(e.unit)}` : "";
  $("#v3sum").innerHTML = `<h4>Where the error is: ${esc(base)}</h4>${e.summary.map((t) => `<p>${esc(t)}</p>`).join("")}
    ${e.regions.length ? `<table><thead><tr><th>Region</th><th class="num">nodes</th><th class="num">share of squared error</th><th class="num">RMS / overall</th><th class="num">max${u ? ` [${esc(e.unit)}]` : ""}</th></tr></thead><tbody>
      ${e.regions.map((g) => `<tr><td>${esc(g.name)}</td><td class="num">${g.n}</td><td class="num">${pct(g.share_error)}</td><td class="num">${g.rms_ratio.toFixed(1)}×</td><td class="num">${fmt(g.max)}</td></tr>`).join("")}</tbody></table>` : ""}
    ${e.hotspots.length ? `<h4>Hotspots E1, E2… (top 10 % of |error|, connected; click to zoom)</h4><ol class="hot">${e.hotspots.map((h, k) => `<li data-k="${k}"><b>${fmt(h.max)}${u}</b> at (${h.at.map(fmt).join(", ")}): ${h.n} nodes, ${pct(h.share)} of the squared error</li>`).join("")}</ol>` : ""}
    <p class="meta">${esc(j.label)}${j.rank ? ` rank ${j.rank}` : ""} at ${j.input.params ? j.input.params.map((v, q) => `${esc(RES.snapshots.param_names[q] || "p" + q)} = ${fmt(v)}`).join(", ") : `t = ${fmt(j.input.times[0])}`}. Zones: the dataset's node sets grown by 10 % of the body length.</p>`;
  $$("#v3sum .hot li").forEach((li) => li.addEventListener("click", () => { const h = e.hotspots[+li.dataset.k]; V3.focus(h.at, Math.max(...h.bbox_max.map((x, q) => x - h.bbox_min[q]))); }));
}
const VIR = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
const DIV = [[33, 102, 172], [146, 197, 222], [247, 247, 247], [244, 165, 130], [178, 24, 43]];
function cmap(st, t) { t = Math.min(1, Math.max(0, t)); const k = Math.min(st.length - 2, Math.floor(t * (st.length - 1))), f = t * (st.length - 1) - k; return st[k].map((c, i) => Math.round(c + f * (st[k + 1][i] - c))); }
async function recon() {
  const key = $("#rmodel").value, i = $("#rsnap").value, f = $("#rfield").value;
  const r = await fetch(`/api/jobs/${JOB}/reconstruction?model=${encodeURIComponent(key)}&snapshot=${i}&field=${encodeURIComponent(f)}`);
  const j = await r.json();
  if (!r.ok) { $("#recon").innerHTML = `<span class="err">${esc(j.detail)}</span>`; return; }
  $("#rinfo").textContent = `relative L2 of this snapshot: ${pct(j.rel_l2)}${j.magnitude ? " (magnitude of the vector shown)" : ""}`;
  if (!j.xy) { $("#recon").innerHTML = `<p class="meta">No coordinates in this dataset: ${j.truth.length} values.</p>`; return; }
  const lo = Math.min(...j.truth, ...j.prediction), hi = Math.max(...j.truth, ...j.prediction), em = Math.max(...j.error.map(Math.abs)) || 1;
  const panels = [["Full-order (held out)", j.truth, (v) => cmap(VIR, (v - lo) / (hi - lo || 1)), [fmt(lo), fmt(hi)], VIR], ["ROM", j.prediction, (v) => cmap(VIR, (v - lo) / (hi - lo || 1)), [fmt(lo), fmt(hi)], VIR], ["ROM − full-order", j.error, (v) => cmap(DIV, 0.5 + v / (2 * em)), [fmt(-em), fmt(em)], DIV]];
  $("#recon").innerHTML = panels.map((p, k) => `<div><div class="cap">${p[0]}${j.unit ? ` [${esc(j.unit)}]` : ""}</div><canvas id="cv${k}"></canvas><div class="cbarrow"><span>${p[3][0]}</span><div class="g" style="background:linear-gradient(90deg,${p[4].map((c) => `rgb(${c})`).join(",")})"></div><span>${p[3][1]}</span></div></div>`).join("");
  const xs = j.xy.map((q) => q[0]), ys = j.xy.map((q) => q[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  panels.forEach((p, k) => {
    const cv = $(`#cv${k}`), w = cv.clientWidth || 380, h = Math.max(120, Math.min(420, w * (y1 - y0) / (x1 - x0 || 1) + 10));
    cv.width = w * 2; cv.height = h * 2; cv.style.height = h + "px";
    const ctx = cv.getContext("2d"), n = xs.length, s = Math.max(2.2, 1.15 * Math.sqrt((w * h) / n)) * 2;
    for (let q = 0; q < n; q++) {
      const c = p[2](p[1][q]); ctx.fillStyle = `rgb(${c})`;
      const px = 8 + (xs[q] - x0) / (x1 - x0 || 1) * (w * 2 - 16), py = h * 2 - 8 - (ys[q] - y0) / (y1 - y0 || 1) * (h * 2 - 16);
      ctx.fillRect(px - s / 2, py - s / 2, s, s);
    }
  });
}

// ------------------------------------------------------------------ run a ROM
let MFILE = null;
$("#mfile").addEventListener("change", (e) => { MFILE = e.target.files[0]; $("#mname").textContent = MFILE ? MFILE.name : "exported from a study"; $("#runm").disabled = $("#runcsv").disabled = !MFILE; });
async function runModel(fmtOut) {
  const fd = new FormData(); fd.append("model", MFILE); fd.append("inputs", $("#queries").value || "{}"); fd.append("fmt", fmtOut);
  $("#rstatus").textContent = "";
  const r = await fetch("/api/run", { method: "POST", body: fd });
  if (!r.ok) { const j = await r.json().catch(() => ({})); $("#rstatus").innerHTML = `<span class="err">${esc(j.detail || r.status)}</span>`; return; }
  if (fmtOut === "csv") { const b = await r.blob(), a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "rom_predictions.csv"; a.click(); return; }
  const j = await r.json();
  $("#runout").innerHTML = `<h3 style="margin-top:0">${esc(META.methods[j.method] || j.method)}${j.rank ? `, rank ${j.rank}` : ""}: ${j.n} snapshot(s) in ${tsec(j.seconds)}</h3>
    <table><thead><tr><th>#</th><th>Inside the training range</th><th class="num">Values</th><th class="num">Min</th><th class="num">Max</th><th>First values</th></tr></thead><tbody>
    ${j.snapshots.map((s, i) => `<tr><td>${i + 1}</td><td>${j.in_range[i] ? '<span class="badge pass">yes</span>' : '<span class="badge warn">no: extrapolation</span>'}</td><td class="num">${s.length}</td><td class="num">${fmt(Math.min(...s))}</td><td class="num">${fmt(Math.max(...s))}</td><td class="meta">${s.slice(0, 6).map(fmt).join(", ")} …</td></tr>`).join("")}</tbody></table>
    <p class="meta">Fields in order: ${esc(j.fields.join(", "))}. Download the CSV for every value.</p>`;
}
$("#runm").addEventListener("click", () => runModel("json"));
$("#runcsv").addEventListener("click", () => runModel("csv"));

// ------------------------------------------------------------------ boot
fetch("/api/meta").then((r) => r.json()).then((m) => {
  META = m; renderMethods("time");
  $("#examples").innerHTML = m.examples.map((e) => `<button type="button" data-id="${e.id}"><b>${esc(e.title)}</b>${esc(e.note)} <a href="/api/examples/${e.id}/file" onclick="event.stopPropagation()">download data</a></button>`).join("");
  $$("#examples button").forEach((b) => b.addEventListener("click", () => submit(`/api/examples/${b.dataset.id}`, { method: "POST" }, b)));
  const o = location.origin;
  $("#c1").textContent = `curl -F files=@wake.npz -F 'options={"ranks":[4,8,16],"budget_s":5,"split":"forecast"}' ${o}/api/studies
curl ${o}/api/jobs/<id>                          # status, progress, result
curl -O ${o}/api/jobs/<id>/files/models/dmd_r16.rom.npz
curl -o dmd_r16.onnx ${o}/api/jobs/<id>/onnx/dmd_r16
curl -F model=@dmd_r16.rom.npz -F 'inputs={"times":[80,81]}' ${o}/api/run`;
  $("#c2").textContent = `from pinneapple_data.snapshots import load_snapshots
from pinneapple_neural.architectures.rom import rom_study, ReducedModel

snaps = load_snapshots([("wake.npz", open("wake.npz", "rb").read())])
res = rom_study(snaps, ranks=[4, 8, 16], budget_s=5)        # held-out: the last 20 % of the times
print(res.to_dict(tolerance=0.05)["best"])
m = ReducedModel.from_bytes(open("dmd_r16.rom.npz", "rb").read())
X = m.predict(times=[80.0, 81.0])                          # (2, n_values)`;
  $("#c3").textContent = `import json, urllib.request
from pinneapple_registry import ModelGateway                 # the model gateway (days 22 and 31)
spec = json.load(urllib.request.urlopen("${o}/api/jobs/<id>/gateway/dmd_r16"))
gw = ModelGateway("models")
gw.publish("wake-rom", "dmd_r16.onnx", kind="onnx", spec=spec)
gw.predict("wake-rom", {"t": [80.0]})`;
});
