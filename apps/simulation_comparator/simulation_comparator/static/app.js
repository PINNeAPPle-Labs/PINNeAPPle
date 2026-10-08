"use strict";
// Simulation Comparator UI: two uploads, the comparison, error maps, parity and distribution plots.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => (v === null || v === undefined || !isFinite(v) ? "—" : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(4) + "");
const pct = (v) => (v === null || v === undefined ? "—" : (100 * v < 0.01 && v > 0 ? (100 * v).toExponential(1) : (100 * v).toFixed(100 * v < 10 ? 2 : 1)) + "%");
let FILES = { ref: [], cand: [] }, RES = null, META = null, SEL = null, SHOW = "error";

const SEQ_ERR = [[0, [255, 247, 243]], [0.25, [252, 187, 161]], [0.5, [251, 106, 74]], [0.75, [203, 24, 29]], [1, [103, 0, 13]]];
const SEQ_VAL = [[0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]], [0.75, [94, 201, 98]], [1, [253, 231, 37]]];
function cmap(stops, t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], f = (t - a) / (b - a);
    return ca.map((c, k) => Math.round(c + f * (cb[k] - c)));
  }
  return stops[stops.length - 1][1];
}
window.CMP_CMAP = { cmap, SEQ_ERR, SEQ_VAL };

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`));
  if (n === "result" && window.CMP_VIEW) window.CMP_VIEW.resize();
}

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
for (const side of ["ref", "cand"]) {
  const d = $(`#drop-${side}`);
  ["dragenter", "dragover"].forEach((ev) => d.addEventListener(ev, (e) => { e.preventDefault(); d.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => d.addEventListener(ev, (e) => { e.preventDefault(); d.classList.remove("over"); }));
  d.addEventListener("drop", async (e) => add(side, await filesFromDrop(e.dataTransfer)));
  $(`#files-${side}`).addEventListener("change", (e) => { add(side, [...e.target.files]); e.target.value = ""; });
}
function add(side, list) {
  for (const f of list) { const n = f.relPath || f.webkitRelativePath || f.name; if (!FILES[side].some((x) => x.n === n)) FILES[side].push({ f, n }); }
  renderFiles();
}
function renderFiles() {
  for (const side of ["ref", "cand"]) {
    const L = FILES[side];
    $(`#list-${side}`).innerHTML = L.slice(0, 4).map((x, i) => `<div><span>${esc(x.n)}</span><button data-s="${side}" data-i="${i}">×</button></div>`).join("") + (L.length > 4 ? `<div><span>+ ${L.length - 4} files</span><button data-s="${side}" data-i="all">clear</button></div>` : "");
  }
  $$(".filelist button").forEach((b) => b.onclick = () => { if (b.dataset.i === "all") FILES[b.dataset.s] = []; else FILES[b.dataset.s].splice(+b.dataset.i, 1); renderFiles(); });
  $("#run").disabled = !(FILES.ref.length && FILES.cand.length);
}
const LAB = { "sim-sim": ["trusted result: fine mesh, validated model", "the result to assess"], "sim-exp": ["measurements: CSV with x, y, z and the measured quantities", "the simulation"],
  "sim-ai": ["the simulation", "the AI prediction (VTU, CSV, NPZ)"], "ai-exp": ["measurements: CSV with x, y, z", "the AI prediction"], "sim-ref": ["analytical or benchmark values (CSV with x, y, z)", "the simulation"] };
$("#mode").onchange = () => { const l = LAB[$("#mode").value]; $("#lab-ref").textContent = l[0]; $("#lab-cand").textContent = l[1]; };

async function load(url, opts, btn) {
  const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Comparing…`; $("#status").textContent = "";
  try {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
    RES = j; SEL = j.fields[0].reference; SHOW = "error"; render(); showTab("result"); window.scrollTo(0, 0);
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; if (btn.id === "run") renderFiles(); }
}
$("#run").addEventListener("click", (e) => {
  const fd = new FormData();
  FILES.ref.forEach((x) => fd.append("reference", x.f, x.n)); FILES.cand.forEach((x) => fd.append("candidate", x.f, x.n));
  fd.append("mode", $("#mode").value); fd.append("mapping", $("#mapping").value); fd.append("method", $("#method").value);
  fd.append("tolerance", $("#tol").value || "0"); fd.append("reference_time", $("#t-ref").value); fd.append("candidate_time", $("#t-cand").value);
  load("/api/compare", { method: "POST", body: fd }, e.currentTarget);
});

function board(r) {
  const g = r.global, line = "─".repeat(28);
  let s = `<span class="head">GLOBAL ERROR</span>  <span class="dim">(normalised by each field's range)</span>\n<span class="dim">${line}</span>\n`;
  s += `${"MAE".padEnd(10)}${fmt(g.mae)}\n${"RMSE".padEnd(10)}${fmt(g.rmse)}\n${"Max error".padEnd(10)}${fmt(g.max)}\n\n`;
  s += `<span class="head">FIELD ERROR</span>  <span class="dim">(relative L2)</span>\n<span class="dim">${line}</span>\n`;
  for (const f of r.fields) {
    const name = (f.reference === f.candidate.replace(/\[.\]$/, "") ? f.reference : `${f.reference} ← ${f.candidate}`);
    const cls = f.verdict === "fail" ? "bad" : f.verdict === "pass" ? "ok" : "";
    s += `${name.padEnd(14)}<span class="${cls}">${pct(f.summary.rel_l2).padStart(8)}</span>${f.verdict ? `  <span class="${cls}">${f.verdict.toUpperCase()}</span>` : ""}\n`;
  }
  return s;
}

function lineSvg(v, unit) {
  const W = 640, H = 300, L = 54, R = 14, T = 12, B = 38;
  const s = v.s, ys = [...v.ref, ...v.cand].filter((x) => x !== null);
  let lo = Math.min(...ys), hi = Math.max(...ys); const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  const x0 = Math.min(...s), x1 = Math.max(...s);
  const X = (x) => L + (x - x0) / (x1 - x0 || 1) * (W - L - R), Y = (y) => T + (1 - (y - lo) / (hi - lo)) * (H - T - B);
  let g = "";
  for (let k = 0; k <= 4; k++) { const y = lo + k * (hi - lo) / 4; g += `<line x1="${L}" x2="${W - R}" y1="${Y(y)}" y2="${Y(y)}" stroke="#eef1f5"/><text x="${L - 6}" y="${Y(y) + 4}" text-anchor="end" fill="#5d6b82">${fmt(y)}</text>`; }
  for (let k = 0; k <= 5; k++) { const x = x0 + k * (x1 - x0) / 5; g += `<text x="${X(x)}" y="${H - B + 15}" text-anchor="middle" fill="#5d6b82">${fmt(x)}</text>`; }
  g += `<text x="${(L + W - R) / 2}" y="${H - 5}" text-anchor="middle" fill="#5d6b82">${v.axis}</text>`;
  g += `<path d="${v.cand.map((y, i) => `${i ? "L" : "M"}${X(s[i]).toFixed(1)},${Y(y).toFixed(1)}`).join("")}" fill="none" stroke="#be123c" stroke-width="2.2"/>`;
  v.ref.forEach((y, i) => { g += `<circle cx="${X(s[i])}" cy="${Y(y)}" r="4" fill="#fff" stroke="#172033" stroke-width="1.7"><title>${v.axis} ${fmt(s[i])}: reference ${fmt(y)}, candidate ${fmt(v.cand[i])}</title></circle>`; });
  return `<div class="plot"><svg viewBox="0 0 ${W} ${H}">${g}</svg><div class="pchart-legend"><span><i style="background:#be123c"></i>candidate (interpolated at the reference points)</span><span>○ reference${unit ? ` (${esc(unit)})` : ""}</span></div></div>`;
}

function scatterCanvas(el, v) {
  const xy = v.xy, n = v.error.length;
  let mn = [Infinity, Infinity], mx = [-Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let k = 0; k < 2; k++) { mn[k] = Math.min(mn[k], xy[2 * i + k]); mx[k] = Math.max(mx[k], xy[2 * i + k]); }
  const w = el.clientWidth || 700, ar = (mx[1] - mn[1]) / (mx[0] - mn[0] || 1), h = Math.max(220, Math.min(520, w * ar));
  el.innerHTML = `<canvas class="scatter" width="${w * 2}" height="${h * 2}" style="height:${h}px"></canvas><div class="legendbar"><span id="lg-lo"></span><div class="grad" id="lg-grad"></div><span id="lg-hi"></span></div>`;
  const cv = el.querySelector("canvas"), ctx = cv.getContext("2d");
  const vals = v[SHOW], fin = vals.filter((x) => x !== null);
  let lo = SHOW === "error" ? 0 : Math.min(...fin), hi = SHOW === "error" ? p99(fin) : Math.max(...fin);
  if (SHOW !== "error") { const o = v[SHOW === "ref" ? "cand" : "ref"].filter((x) => x !== null); lo = Math.min(lo, ...o); hi = Math.max(hi, ...o); }
  const stops = SHOW === "error" ? SEQ_ERR : SEQ_VAL;
  const sx = (cv.width - 20) / (mx[0] - mn[0] || 1), sy = (cv.height - 20) / (mx[1] - mn[1] || 1), s = Math.min(sx, sy);
  const cell = Math.max(3, Math.sqrt((cv.width * cv.height) / n) * 0.95);
  for (let i = 0; i < n; i++) {
    if (vals[i] === null) continue;
    const c = cmap(stops, (vals[i] - lo) / (hi - lo || 1));
    ctx.fillStyle = `rgb(${c})`;
    ctx.fillRect(10 + (xy[2 * i] - mn[0]) * s - cell / 2, cv.height - 10 - (xy[2 * i + 1] - mn[1]) * s - cell / 2, cell, cell);
  }
  $("#lg-lo", el).textContent = fmt(lo); $("#lg-hi", el).textContent = (SHOW === "error" && hi < Math.max(...fin) ? "≥ " : "") + fmt(hi) + (SHOW === "error" && hi < Math.max(...fin) ? " (p99)" : "");
  $("#lg-grad", el).style.background = `linear-gradient(to right, ${stops.map(([t, c]) => `rgb(${c}) ${t * 100}%`).join(",")})`;
}

function p99(a) {
  const s = Float64Array.from(a).sort();
  return s.length ? s[Math.min(s.length - 1, Math.floor(0.99 * s.length))] || s[s.length - 1] : 0;
}
window.CMP_P99 = p99;

function paritySvg(p, unit) {
  const W = 440, H = 330, L = 52, B = 36, T = 10, R = 12;
  const all = [...p.ref, ...p.cand].filter((x) => x !== null);
  const lo = Math.min(...all), hi = Math.max(...all);
  const X = (x) => L + (x - lo) / (hi - lo || 1) * (W - L - R), Y = (y) => T + (1 - (y - lo) / (hi - lo || 1)) * (H - T - B);
  let g = `<line x1="${X(lo)}" y1="${Y(lo)}" x2="${X(hi)}" y2="${Y(hi)}" stroke="#94a3b8" stroke-dasharray="4 3"/>`;
  p.ref.forEach((r, i) => { if (r !== null && p.cand[i] !== null) g += `<circle cx="${X(r).toFixed(1)}" cy="${Y(p.cand[i]).toFixed(1)}" r="1.8" fill="#be123c" fill-opacity=".45"/>`; });
  g += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="#c9d1dd"/><line x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" stroke="#c9d1dd"/>`;
  g += `<text x="${L}" y="${H - B + 14}" fill="#5d6b82">${fmt(lo)}</text><text x="${W - R}" y="${H - B + 14}" text-anchor="end" fill="#5d6b82">${fmt(hi)}</text>`;
  g += `<text x="${(L + W) / 2}" y="${H - 4}" text-anchor="middle" fill="#5d6b82">reference${unit ? ` (${esc(unit)})` : ""}</text><text x="12" y="${(T + H - B) / 2}" transform="rotate(-90 12 ${(T + H - B) / 2})" text-anchor="middle" fill="#5d6b82">candidate</text>`;
  return `<div class="plot"><div class="cap">Parity (${p.ref.length.toLocaleString("en-US")} points)</div><svg viewBox="0 0 ${W} ${H}">${g}</svg></div>`;
}
function histSvg(h, unit) {
  const W = 440, H = 190, L = 40, B = 30, T = 8, R = 10, n = h.counts.length;
  if (!n) return "";
  const max = Math.max(...h.counts, 1), lo = h.edges[0], hi = h.edges[n];
  let g = "";
  h.counts.forEach((c, i) => { const x0 = L + i / n * (W - L - R), w = (W - L - R) / n; const y = T + (1 - Math.sqrt(c / max)) * (H - T - B); if (c) g += `<rect x="${x0 + 0.5}" y="${y}" width="${w - 1}" height="${H - B - y}" fill="#be123c" opacity=".8"><title>${fmt(h.edges[i])}–${fmt(h.edges[i + 1])}: ${c}</title></rect>`; });
  g += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="#c9d1dd"/><text x="${L}" y="${H - 10}" fill="#5d6b82">${fmt(lo)}</text><text x="${W - R}" y="${H - 10}" text-anchor="end" fill="#5d6b82">${fmt(hi)}${unit ? " " + esc(unit) : ""}</text>`;
  return `<div class="plot"><div class="cap">Error distribution <span class="meta">bar height √count</span></div><svg viewBox="0 0 ${W} ${H}">${g}</svg></div>`;
}

function render() {
  const r = RES, st = r.status || "NONE";
  const rows = r.fields.map((f) => {
    const s = f.summary;
    return `<tr data-f="${esc(f.reference)}" class="${f.reference === SEL ? "sel" : ""}"><td><b>${esc(f.reference)}</b>${f.reference !== f.candidate ? ` <span class="meta">← ${esc(f.candidate)}</span>` : ""}<br><span class="meta">${esc(f.type)} · ${esc(f.where)} · ${esc(f.unit || "")}${f.unit_mismatch ? ` <span class="err">≠ ${esc(f.unit_candidate)}</span>` : ""}</span></td>
      <td class="num"><b>${pct(s.rel_l2)}</b></td><td class="num">${fmt(s.mae)}</td><td class="num">${fmt(s.rmse)}</td><td class="num">${fmt(s.max_abs)}</td><td class="num">${fmt(s.p99_abs)}</td><td class="num">${fmt(s.bias)}</td><td class="num">${pct(s.nrmse)}</td><td class="num">${s.r2 === null ? "—" : s.r2.toFixed(4)}</td>
      <td class="meta">${esc(f.interpolation)}${f.outside_candidate ? ` · ${f.outside_candidate} outside` : ""}</td><td>${f.verdict ? `<span class="pill ${f.verdict}">${f.verdict.toUpperCase()}</span>` : ""}</td></tr>`;
  }).join("");
  const side = (lab, m) => `<div><b>${lab}</b>: ${esc(m.file || "")} · ${esc(m.format || "")} · ${m.cells ? `${m.cells.toLocaleString("en-US")} cells · ` : ""}${m.points.toLocaleString("en-US")} points${m.time !== null && m.time !== undefined ? ` · t = ${fmt(m.time)}` : ""}<br><span class="meta">fields: ${Object.entries(m.fields).map(([k, v]) => `${esc(k)}${v.unit ? ` [${esc(v.unit)}]` : ""}`).join(", ")}</span></div>`;
  $("#result").innerHTML = `
    <div class="panel">
      <div class="toolbar"><h2 style="margin:0">${esc(r.example ? r.example.title : r.mode_label)}</h2>
        <div class="noprint" style="display:flex;gap:6px"><button class="ghost" id="dl-json">JSON</button><button class="ghost" id="dl-csv">Metrics CSV</button><button class="ghost" onclick="print()">Print / PDF</button></div></div>
      <div class="verdict ${st}"><div class="big">${st === "NONE" ? r.mode_label : st}</div><div>${st !== "NONE" ? `${r.fields.filter((f) => f.verdict === "pass").length} of ${r.fields.length} fields within ${r.tolerance_pct}% (relative L2) · ` : ""}${esc(r.mode_label)}${r.example ? `<br><span style="font-size:13px">${esc(r.example.note)}</span>` : ""}</div></div>
      <div class="grid2b"><div class="board">${board(r)}</div><div style="font-size:13px;display:grid;gap:8px">${side("Reference", r.reference)}${side("Candidate", r.candidate)}
        <p class="meta">${esc(r.global.note)}.</p></div></div>
    </div>
    <div class="panel">
      <h3 style="margin-top:0">Per field <span class="meta">click a row for its maps</span></h3>
      <div style="overflow-x:auto"><table class="ftab"><thead><tr><th>Field</th><th class="num">Rel. L2</th><th class="num">MAE</th><th class="num">RMSE</th><th class="num">Max</th><th class="num">p99</th><th class="num">Bias</th><th class="num">NRMSE</th><th class="num">R²</th><th>Interpolation</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
    </div>
    <div class="panel" id="fieldpanel"></div>
    <div class="panel">${window.renderScope ? renderScope(r.scope) : ""}</div>`;
  $$("table.ftab tr[data-f]").forEach((tr) => tr.onclick = () => { SEL = tr.dataset.f; render(); $("#fieldpanel").scrollIntoView({ behavior: "smooth" }); });
  $("#dl-json").onclick = () => save("comparison.json", JSON.stringify(Object.fromEntries(Object.entries(r).filter(([k]) => k !== "view")), null, 1), "application/json");
  $("#dl-csv").onclick = () => {
    const rows = [["field", "candidate", "unit", "rel_l2", "mae", "rmse", "max_abs", "max_at", "bias", "nrmse", "r2", "verdict"],
      ...r.fields.map((f) => [f.reference, f.candidate, f.unit, f.summary.rel_l2, f.summary.mae, f.summary.rmse, f.summary.max_abs, f.max_at.join(" "), f.summary.bias, f.summary.nrmse, f.summary.r2, f.verdict])];
    save("comparison-metrics.csv", rows.map((x) => x.map((v) => `"${String(v ?? "").replace(/"/g, '""')}"`).join(",")).join("\n"), "text/csv");
  };
  renderField();
}

function renderField() {
  const r = RES, f = r.fields.find((x) => x.reference === SEL), v = r.view.fields[SEL];
  const unit = f.unit || "";
  const showBar = v.kind !== "line" ? `<div class="mapbar"><b>${esc(SEL)}</b>${[["error", "Error"], ["ref", "Reference"], ["cand", "Candidate"]].map(([k, l]) => `<button data-s="${k}" class="${SHOW === k ? "on" : ""}">${l}</button>`).join("")}<span class="meta">${v.kind.startsWith("surface") ? "on the reference geometry · drag to rotate" : `${v.error.length.toLocaleString("en-US")} reference points`}</span></div>` : "";
  $("#fieldpanel").innerHTML = `<h3 style="margin-top:0">${esc(SEL)}: where the candidate differs <span class="meta">max ${fmt(f.summary.max_abs)} ${esc(unit)} at (${f.max_at.map(fmt).join(", ")})</span></h3>
    <div class="grid3"><div>${v.kind === "line" ? lineSvg(v, unit) : `<div class="mapbox">${showBar}<div id="map"></div></div>`}</div>
      <div style="display:grid;gap:12px">${paritySvg(v.parity, unit)}${histSvg(v.hist, unit)}</div></div>
    <h4>Largest differences</h4>
    <table><thead><tr><th>Location</th><th class="num">Reference</th><th class="num">Candidate</th><th class="num">|error|</th></tr></thead><tbody>${v.worst.slice(0, 8).map((w) => `<tr><td class="mono">${w.at.map(fmt).join(", ")}</td><td class="num">${fmt(w.ref)}</td><td class="num">${fmt(w.cand)}</td><td class="num"><b>${fmt(w.error)}</b></td></tr>`).join("")}</tbody></table>
    ${Object.keys(f.components).length > 1 ? `<h4>Components</h4><table><thead><tr><th>Component</th><th class="num">Rel. L2</th><th class="num">MAE</th><th class="num">RMSE</th><th class="num">Max</th><th class="num">R²</th></tr></thead><tbody>${Object.entries(f.components).map(([k, c]) => `<tr><td>${esc(k)}</td><td class="num">${pct(c.rel_l2)}</td><td class="num">${fmt(c.mae)}</td><td class="num">${fmt(c.rmse)}</td><td class="num">${fmt(c.max_abs)}</td><td class="num">${c.r2 === null ? "—" : c.r2.toFixed(4)}</td></tr>`).join("")}</tbody></table>` : ""}`;
  $$(".mapbar button").forEach((b) => b.onclick = () => { SHOW = b.dataset.s; renderField(); });
  if (v.kind === "scatter2d") scatterCanvas($("#map"), v);
  else if (v.kind.startsWith("surface") || v.kind === "points3d") {
    const mount = () => { window.CMP_VIEW = new window.FieldViewer($("#map"), r.view, SEL, SHOW, unit); };
    if (window.FieldViewer) mount(); else window.addEventListener("cmp-viewer-ready", mount, { once: true });
  }
}

function save(name, text, type) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

(async function init() {
  try {
    META = await (await fetch("/api/meta")).json();
    $("#examples").innerHTML = META.examples.map((e) => `<button type="button" data-id="${e.id}"><b>${esc(e.title)}</b><span class="meta">${esc(META.modes[e.mode])}</span></button>`).join("");
    $$("#examples button").forEach((b) => b.addEventListener("click", () => load(`/api/example/${b.dataset.id}?tolerance=${$("#tol").value || 5}`, {}, b)));
  } catch (e) { console.error(e); }
  const host = location.origin;
  $("#c1").textContent = `curl -s -F reference=@fine_case.zip -F candidate=@coarse_case.zip -F tolerance=5 ${host}/api/compare \\
  | jq '.global, (.fields[] | {field: .reference, rel_l2: .summary.rel_l2, max: .summary.max_abs, verdict})'

# sensors (CSV with x,y,z and the measured fields) against a simulation
curl -s -F mode=sim-exp -F reference=@sensors.csv -F candidate=@case.zip ${host}/api/compare | jq .fields
# an AI prediction against the solver (field names mapped explicitly)
curl -s -F mode=sim-ai -F reference=@cfd.vtu -F candidate=@fno_prediction.npz -F 'mapping={"p":"pressure"}' ${host}/api/compare`;
  $("#c2").textContent = `from pinneapple_data.cae import read_any
from pinneapple_data.cae.compare import compare
from pinneapple_data.cae.upload import expand

ref = read_any(expand([("fine.zip", open("fine.zip", "rb").read())]))
cand = read_any(expand([("coarse.zip", open("coarse.zip", "rb").read())]))
res = compare(ref, cand)                 # candidate interpolated onto the reference cells
for f in res["fields"]:
    print(f["reference"], f"{100 * f['summary']['rel_l2']:.2f} %", f["summary"]["max_abs"], f["max_at"])`;
})();
