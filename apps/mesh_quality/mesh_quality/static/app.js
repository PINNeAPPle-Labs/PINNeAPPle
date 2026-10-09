"use strict";
// Mesh Quality UI: upload, examples, the health report, histograms, problem regions and the 3D view.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => (v === null || v === undefined || !isFinite(v) ? "—" : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(4) + "");
const int = (v) => (typeof v === "number" ? v.toLocaleString("en-US") : "—");
let FILES = [], REP = null, META = null;

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`));
  if (n === "report" && REP && window.MQA_VIEW) window.MQA_VIEW.resize();
}

const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async (e) => addFiles(await filesFromDrop(e.dataTransfer)));
$("#files").addEventListener("change", (e) => { addFiles([...e.target.files]); e.target.value = ""; });

// folders dropped from the desktop keep their relative paths (an OpenFOAM case is a folder)
async function filesFromDrop(dt) {
  const out = [];
  const walk = (entry, path) => new Promise((res) => {
    if (entry.isFile) entry.file((f) => { f.relPath = path + f.name; out.push(f); res(); });
    else if (entry.isDirectory) {
      const rd = entry.createReader(), all = [];
      const read = () => rd.readEntries(async (ents) => {
        if (!ents.length) { for (const en of all) await walk(en, path + entry.name + "/"); res(); }
        else { all.push(...ents); read(); }
      });
      read();
    } else res();
  });
  const items = [...(dt.items || [])].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  if (!items.length) return [...dt.files];
  for (const en of items) await walk(en, "");
  return out;
}
function addFiles(list) {
  for (const f of list) { const n = f.relPath || f.webkitRelativePath || f.name; if (!FILES.some((x) => x.n === n)) FILES.push({ f, n }); }
  renderFiles();
}
function renderFiles() {
  const shown = FILES.slice(0, 8);
  $("#filelist").innerHTML = shown.map((x, i) => `<div><span><b>${esc(x.n)}</b> · ${(x.f.size / 1024).toFixed(0)} KB</span><button data-i="${i}">×</button></div>`).join("") +
    (FILES.length > 8 ? `<div><span>+ ${FILES.length - 8} more files</span><button id="clear">clear</button></div>` : "");
  $$("#filelist button[data-i]").forEach((b) => b.addEventListener("click", () => { FILES.splice(+b.dataset.i, 1); renderFiles(); }));
  if ($("#clear")) $("#clear").onclick = () => { FILES = []; renderFiles(); };
  $("#run").disabled = !FILES.length;
}

async function load(url, opts, btn) {
  const old = btn.innerHTML; btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Checking…`;
  $("#status").textContent = "";
  try {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
    REP = j; render(); showTab("report");
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; if (btn.id === "run") btn.disabled = !FILES.length; }
}
$("#run").addEventListener("click", (e) => {
  const fd = new FormData(); FILES.forEach((x) => fd.append("files", x.f, x.n)); fd.append("target", $("#target").value);
  load("/api/check", { method: "POST", body: fd }, e.currentTarget);
});

const ICON = { pass: "✓", warn: "!", fail: "✗" };
const LABEL = (k) => (REP.metric_info[k] || {}).label || k;

function healthText(r) {
  const s = r.summary, m = r.metrics;
  const line = "─".repeat(34);
  const row = (k, v) => `${(k + ":").padEnd(20)}${v}`;
  const cls = (st) => (st === "fail" ? "bad" : st === "warn" ? "warn" : "ok");
  let out = `<span class="head">MESH HEALTH</span>\n<span class="dim">${line}</span>\n`;
  out += row("Elements", int(s.cells)) + `  <span class="dim">${Object.entries(s.element_types).map(([k, v]) => `${int(v)} ${k}`).join(", ")}</span>\n`;
  if (s.regions !== undefined) out += row("Connected regions", `<span class="${s.regions > 1 ? "warn" : "ok"}">${s.regions}</span>`) + "\n";
  const order = ["aspect_ratio", "skewness", "non_orthogonality", "scaled_jacobian", "element_skewness", "edge_ratio"];
  for (const k of order) {
    const v = m[k];
    if (!v || !v.graded) continue;
    out += `\n${LABEL(k)}\n`;
    out += `  ${"Worst:".padEnd(18)}<span class="${cls(v.status)}">${fmt(v.worst)}</span>${v.bad ? `  <span class="${cls(v.status)}">(${int(v.bad)} ${v.per}s beyond limit)</span>` : ""}\n`;
    out += `  ${"Mean:".padEnd(18)}${fmt(v.mean)}\n`;
  }
  if (r.surface) {
    out += `\n${row("Open edges", `<span class="${r.surface.open_edges ? "bad" : "ok"}">${r.surface.open_edges}</span>`)}\n${row("Watertight", r.surface.watertight ? '<span class="ok">yes</span>' : '<span class="bad">no</span>')}\n`;
  }
  const regs = r.regions.filter((g) => g.centre);
  out += `\nProblem regions:${regs.length ? "" : '  <span class="ok">none</span>'}\n`;
  for (const g of r.regions) out += `  <span class="${g.status === "fail" ? "bad" : "warn"}">${esc(g.name)}</span>  <span class="dim">${int(g.cells)} cells${g.nearest_patch ? ` · near ${esc(g.nearest_patch)}` : ""}</span>\n`;
  out += `\n<span class="dim">${line}</span>\nOverall: <span class="${r.status === "FAIL" ? "bad" : r.status === "WARNING" ? "warn" : "ok"}">${r.status}</span>`;
  return out;
}

function histSvg(k, v) {
  const h = v.hist, info = REP.metric_info[k];
  if (!h || !h.counts.length) return "";
  const W = 300, H = 120, L = 34, B = 22, T = 6, R = 6;
  const max = Math.max(...h.counts, 1), n = h.counts.length;
  const lo = h.edges[0], hi = h.edges[n];
  const X = (x) => L + (h.log ? (Math.log10(x) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (x - lo) / (hi - lo)) * (W - L - R);
  const Y = (c) => T + (1 - Math.sqrt(c / max)) * (H - T - B);
  let g = "";
  h.counts.forEach((c, i) => {
    const x0 = X(h.edges[i]), x1 = X(h.edges[i + 1]);
    const mid = (h.edges[i] + h.edges[i + 1]) / 2;
    const bad = info.better === "low" ? (info.fail !== null && mid > info.fail ? "fail" : info.warn !== null && mid > info.warn ? "warn" : "ok")
      : (info.fail !== null && mid <= info.fail ? "fail" : info.warn !== null && mid < info.warn ? "warn" : "ok");
    const col = bad === "fail" ? "#b42318" : bad === "warn" ? "#d97706" : "#0f766e";
    if (c) g += `<rect x="${x0 + 0.5}" y="${Y(c)}" width="${Math.max(1, x1 - x0 - 1)}" height="${H - B - Y(c)}" fill="${col}" opacity=".85"><title>${fmt(h.edges[i])} – ${fmt(h.edges[i + 1])}: ${int(c)}</title></rect>`;
  });
  for (const [t, c, lab] of [[info.warn, "#d97706", "warn"], [info.fail, "#b42318", "fail"]]) {
    if (t === null || t < lo || t > hi) continue;
    g += `<line x1="${X(t)}" x2="${X(t)}" y1="${T}" y2="${H - B}" stroke="${c}" stroke-dasharray="3 2"/><text x="${X(t) + 3}" y="${T + 9}" fill="${c}">${lab} ${fmt(t)}</text>`;
  }
  g += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="#c9d1dd"/>`;
  g += `<text x="${L}" y="${H - 6}" fill="#5d6b82">${fmt(lo)}</text><text x="${W - R}" y="${H - 6}" text-anchor="end" fill="#5d6b82">${fmt(hi)}${info.unit && !info.unit.startsWith("m3") ? " " + info.unit : ""}</text>`;
  g += `<text x="${L - 4}" y="${T + 8}" text-anchor="end" fill="#5d6b82">${int(max)}</text><text x="${L - 4}" y="${H - B}" text-anchor="end" fill="#5d6b82">0</text>`;
  return `<div class="hist"><div class="cap">${esc(info.label)} <span class="meta">per ${v.per}${h.log ? " · log scale" : ""} · bar height √count</span></div><svg viewBox="0 0 ${W} ${H}">${g}</svg></div>`;
}

function render() {
  const r = REP, s = r.summary;
  const fails = r.checks.filter((c) => c.status === "fail").length, warns = r.checks.filter((c) => c.status === "warn").length;
  const bb = s.bounding_box.size.map(fmt).join(" × ");
  const checks = r.checks.slice().sort((a, b) => ({ fail: 0, warn: 1, pass: 2 }[a.status] - { fail: 0, warn: 1, pass: 2 }[b.status]));
  const metricRows = Object.entries(r.metrics).map(([k, v]) => `<tr><td><b>${esc(LABEL(k))}</b>${v.graded ? "" : ' <span class="meta">(info)</span>'}</td>
      <td class="num">${fmt(v.worst)}</td><td class="num">${fmt(v.mean)}</td><td class="num">${fmt(v.p99)}</td>
      <td class="num">${int(v.bad)} / ${int(v.of)}</td><td><span class="pill ${v.graded ? v.status : ""}">${v.graded ? v.status : "—"}</span></td>
      <td class="meta" style="max-width:340px">${esc(r.metric_info[k].source)}</td></tr>`).join("");
  const regRows = r.regions.map((g, i) => `<tr data-i="${i}"><td><span class="pill ${g.status}">${esc(g.name)}</span></td><td class="num">${int(g.cells)}</td>
      <td>${Object.entries(g.worst).map(([k, w]) => `${esc(LABEL(k))} <b>${fmt(w.worst)}</b> <span class="meta">at ${esc(w.at)}</span>`).join("<br>")}</td>
      <td class="mono">${g.centre ? g.centre.map(fmt).join(", ") : ""}</td><td>${esc(g.nearest_patch || "")}</td></tr>`).join("");
  const worst = {};
  for (const w of r.worst_cells) (worst[w.metric] = worst[w.metric] || []).push(w);
  const fieldOpts = Object.keys(r.view.fields).map((k) => `<option value="${k}">${esc(LABEL(k))}</option>`).join("");
  $("#report").innerHTML = `
    <div class="panel">
      <div class="toolbar"><h2 style="margin:0">${esc(r.example ? r.example.description : r.files.map((f) => f.name).join(", "))}</h2>
        <div class="noprint" style="display:flex;gap:6px"><button class="ghost" id="dl-json">Report JSON</button><button class="ghost" id="dl-csv">Checklist CSV</button><button class="ghost" onclick="print()">Print / PDF</button></div></div>
      <div class="verdict ${r.status}"><div class="big">${r.status}</div><div>${fails} failed, ${warns} warnings, ${r.checks.length - fails - warns} passed ·
        ${esc(s.format)} · ${int(s.cells)} cells · ${int(s.points)} points · ${bb} (model units) · checked for <b>${esc({ cfd: "CFD (finite volume)", fea: "FEA (finite element)", surface: "surface meshing" }[s.target] || s.target)}</b></div></div>
      <div class="grid2b">
        <div class="health">${healthText(r)}</div>
        <div>${checks.map((c) => `<div class="chk"><div class="ic ${c.status}">${ICON[c.status]}</div><div><div>${esc(c.message)}</div>
          ${c.status !== "pass" && c.why ? `<div class="why">${esc(c.why)}</div>` : ""}${c.action ? `<div class="act"><b>Fix:</b> ${esc(c.action)}</div>` : ""}</div></div>`).join("")}
          ${r.notes.length ? `<p class="notes">${r.notes.map(esc).join("<br>")}</p>` : ""}</div>
      </div>
    </div>
    <div class="panel">
      <div class="toolbar"><h3 style="margin:0">Where</h3><span class="meta">${r.view.decimated ? "surface sampled for display · " : ""}boundary faces coloured by the metric of their cell · labels mark the problem regions</span></div>
      <div class="grid2b" style="margin-top:8px">
        <div id="viewer" data-fields='${esc(fieldOpts)}'></div>
        <div>${r.regions.length ? `<table class="regions"><thead><tr><th>Region</th><th class="num">Cells</th><th>Worst</th><th>Centre</th><th>Near</th></tr></thead><tbody>${regRows}</tbody></table>
          <p class="meta">Click a region to fly to it.</p>` : `<div class="banner ok">No problem regions: no cell beyond a threshold.</div>`}</div>
      </div>
    </div>
    <div class="panel">
      <h3 style="margin-top:0">Metrics</h3>
      <div style="overflow-x:auto"><table><thead><tr><th>Metric</th><th class="num">Worst</th><th class="num">Mean</th><th class="num">p99</th><th class="num">Beyond limit</th><th>Status</th><th>Threshold source</th></tr></thead><tbody>${metricRows}</tbody></table></div>
      <div class="hists" style="margin-top:12px">${Object.entries(r.metrics).filter(([, v]) => v.hist && v.hist.counts.length).map(([k, v]) => histSvg(k, v)).join("")}</div>
    </div>
    <div class="panel">
      <h3 style="margin-top:0">Worst elements</h3>
      <div class="hists">${Object.entries(worst).filter(([k]) => r.metrics[k] && r.metrics[k].graded).map(([k, ws]) => `<div><div class="cap" style="font-weight:600;font-size:12.5px">${esc(LABEL(k))}</div>
        <table><thead><tr><th>Element</th><th class="num">Value</th><th>Centre</th></tr></thead><tbody>${ws.slice(0, 8).map((w) => `<tr><td>${esc(w.label)}</td><td class="num">${fmt(w.value)}</td><td class="mono">${w.centre ? w.centre.map(fmt).join(", ") : ""}</td></tr>`).join("")}</tbody></table></div>`).join("")}</div>
      ${s.patches ? `<h3>Boundary patches</h3><p>${s.patches.map((p) => `<span class="pill">${esc(p.name)} · ${esc(p.type)} · ${int(p.faces)} faces</span>`).join(" ")}</p>` : ""}
      ${window.renderScope ? renderScope(r.scope) : ""}
    </div>`;
  $("#dl-json").onclick = () => save(`mesh-health.json`, JSON.stringify(Object.fromEntries(Object.entries(r).filter(([k]) => k !== "view")), null, 1), "application/json");
  $("#dl-csv").onclick = () => {
    const rows = [["status", "check", "message", "why", "action"], ...r.checks.map((c) => [c.status, c.key, c.message, c.why, c.action]),
      [], ["region", "cells", "centre", "nearest patch", "worst"], ...r.regions.map((g) => [g.name, g.cells, (g.centre || []).join(" "), g.nearest_patch || "", Object.entries(g.worst).map(([k, w]) => `${k}=${w.worst} at ${w.at}`).join("; ")])];
    save("mesh-checklist.csv", rows.map((r) => r.map((x) => `"${String(x ?? "").replace(/"/g, '""')}"`).join(",")).join("\n"), "text/csv");
  };
  $$("table.regions tr[data-i]").forEach((tr) => tr.addEventListener("click", () => window.MQA_VIEW && window.MQA_VIEW.focus(r.regions[+tr.dataset.i])));
  const mount = () => { window.MQA_VIEW = new window.MeshViewer($("#viewer"), r); };
  if (window.MeshViewer) mount(); else window.addEventListener("mqa-viewer-ready", mount, { once: true });
}

function save(name, text, type) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

(async function init() {
  try {
    META = await (await fetch("/api/meta")).json();
    $("#examples").innerHTML = META.examples.map((e) => `<button type="button" data-n="${esc(e.name)}"><b>${esc(e.name)}</b>${esc(e.description)}</button>`).join("");
    $$("#examples button").forEach((b) => b.addEventListener("click", () => load(`/api/example/${encodeURIComponent(b.dataset.n)}`, {}, b)));
  } catch (e) { console.error(e); }
  const host = location.origin;
  $("#c1").textContent = `curl -s -F files=@mesh.msh -F target=fea ${host}/api/check | python -m json.tool | head -40
# OpenFOAM: zip the case (or constant/polyMesh) and send the zip
curl -s -F files=@case.zip ${host}/api/check | jq '.status, .checks[] | select(.status != "pass") | .message'`;
  $("#c2").textContent = `from pinneapple_data.cae import read_any, mesh_report
from pinneapple_data.cae.upload import expand

mesh = read_any(expand([("bracket.msh", open("bracket.msh", "rb").read())]))
rep = mesh_report(mesh, target="fea")          # or "cfd": checkMesh-equivalent metrics
print(rep["status"])
for c in rep["checks"]:
    print(c["status"], c["message"], "->", c["action"])`;
})();
