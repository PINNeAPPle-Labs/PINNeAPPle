"use strict";
// Simulation Preflight UI: upload a case, run the rules, show the pre-flight report with locations and fixes.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let FILES = [], REP = null, META = null, FILTER = "problems";

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`));
}

const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async (e) => addFiles(await filesFromDrop(e.dataTransfer)));
$("#files").addEventListener("change", (e) => { addFiles([...e.target.files]); e.target.value = ""; });
async function filesFromDrop(dt) {
  const out = [];
  const walk = (entry, path) => new Promise((res) => {
    if (entry.isFile) entry.file((f) => { f.relPath = path + f.name; out.push(f); res(); });
    else if (entry.isDirectory) {
      const rd = entry.createReader(), all = [];
      const read = () => rd.readEntries(async (ents) => {
        if (!ents.length) { for (const en of all) await walk(en, path + entry.name + "/"); res(); } else { all.push(...ents); read(); }
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
  $("#filelist").innerHTML = FILES.slice(0, 8).map((x, i) => `<div><span><b>${esc(x.n)}</b> · ${(x.f.size / 1024).toFixed(0)} KB</span><button data-i="${i}">×</button></div>`).join("") +
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
    REP = j; FILTER = "problems"; render(); showTab("report"); window.scrollTo(0, 0);
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; if (btn.id === "run") btn.disabled = !FILES.length; }
}
$("#run").addEventListener("click", (e) => {
  const fd = new FormData(); FILES.forEach((x) => fd.append("files", x.f, x.n));
  load("/api/preflight", { method: "POST", body: fd }, e.currentTarget);
});

const MARK = { pass: "✓", info: "·", warn: "⚠", fail: "✗" };
const CLS = { pass: "ok", info: "dim", warn: "warn", fail: "bad" };

function preflightText(r) {
  const line = "─".repeat(30);
  let out = `<span class="head">SIMULATION PRE-FLIGHT</span>  <span class="dim">${esc(r.solver)}</span>\n<span class="dim">${line}</span>\n\n`;
  // one line per section that passed, then every problem
  const sec = Object.fromEntries(r.sections.map((s) => [s.key, s]));
  const okLine = { mesh: "Mesh found", materials: "Material properties", boundary_conditions: "Boundary conditions", initial_conditions: "Initial conditions", solver: "Solver settings" };
  for (const s of r.sections) if (s.status === "pass") out += `<span class="ok">✓</span> ${okLine[s.key]}\n`;
  const probs = r.findings.filter((f) => f.status === "warn" || f.status === "fail");
  if (probs.length) out += "\n";
  for (const f of probs.filter((f) => f.status === "warn")) out += `<span class="warn">⚠</span> ${esc(f.title)}\n`;
  for (const f of probs.filter((f) => f.status === "fail")) out += `<span class="bad">✗</span> ${esc(f.title)}\n`;
  out += `\n<span class="dim">${line}</span>\nOverall: <span class="${r.status === "FAIL" ? "bad" : r.status === "WARNING" ? "warn" : "ok"}">${r.status}</span>`;
  return out;
}

function findingHtml(f) {
  const loc = [f.file, f.line ? `line ${f.line}` : "", f.entity ? `· ${f.entity}` : ""].filter(Boolean).join(" ");
  const snip = f.snippet ? `<div class="snip">${f.snippet.lines.map((l, i) => {
    const n = f.snippet.start + i;
    return `<div class="${n === f.snippet.at ? "at" : ""}"><span class="n">${n}</span>${esc(l)}</div>`;
  }).join("")}</div>` : "";
  return `<div class="find ${f.status}"><div class="hd"><span class="pill ${f.status}">${{ fail: "FAIL", warn: "WARNING", info: "INFO", pass: "PASS" }[f.status]}</span>
      <span class="t">${esc(f.title)}</span><span class="loc">${esc(loc)}</span></div>
    ${f.detail ? `<div class="why">${esc(f.detail)}</div>` : ""}
    ${f.fix ? `<div class="fix"><b>Fix:</b> ${esc(f.fix)}</div>` : ""}
    ${f.evidence ? `<div class="ev"><b>What the solver does with it</b> (our test run): ${esc(f.evidence)}</div>` : ""}${snip}</div>`;
}

function render() {
  const r = REP;
  const c = r.counts;
  const secLabel = { pass: "OK", warn: "warning", fail: "fail", na: "not applicable" };
  const filt = { problems: (f) => f.status === "fail" || f.status === "warn", all: () => true, pass: (f) => f.status === "pass" || f.status === "info" };
  const shown = r.findings.filter(filt[FILTER]);
  const info = r.info || {};
  const facts = [
    info.application ? `application <b>${esc(info.application)}</b>` : "",
    info.turbulence ? `turbulence <b>${esc(info.turbulence)}</b>` : "",
    info.required_fields ? `fields ${esc(info.required_fields.join(", "))}` : "",
    info.mesh ? `mesh <b>${(info.mesh.cells || info.mesh.elements || 0).toLocaleString("en-US")}</b> ${info.mesh.cells ? "cells" : "elements"}` : "",
    info.steps && Array.isArray(info.steps) ? `steps: ${esc(info.steps.map((s) => s.procedure || "?").join(", ").toLowerCase())}` : info.steps ? `${info.steps.toLocaleString("en-US")} time steps` : "",
    info.courant_estimate ? `Courant ≲ ${info.courant_estimate.max.toPrecision(3)}` : "",
  ].filter(Boolean).join(" · ");
  $("#report").innerHTML = `
    <div class="panel">
      <div class="toolbar"><h2 style="margin:0">${esc(r.example ? r.example.description : r.files.slice(0, 3).map((f) => f.name).join(", ") + (r.files.length > 3 ? ` +${r.files.length - 3} files` : ""))}</h2>
        <div class="noprint" style="display:flex;gap:6px"><button class="ghost" id="dl-md">Checklist (Markdown)</button><button class="ghost" id="dl-csv">CSV</button><button class="ghost" id="dl-json">JSON</button><button class="ghost" onclick="print()">Print / PDF</button></div></div>
      <div class="verdict ${r.status}"><div class="big">${r.status}</div><div>${c.fail} failed · ${c.warn} warnings · ${c.pass} passed${c.info ? ` · ${c.info} notes` : ""}<br><span style="font-size:13px">${facts}</span></div></div>
      ${r.example ? `<div class="solverres"><b>What the real solver did with this case:</b> ${esc(r.example.solver_result)} · <a class="link" href="/api/example/${encodeURIComponent(r.example.name)}/log" target="_blank">solver log</a> · <a class="link" href="/api/example/${encodeURIComponent(r.example.name)}/file">download the case</a></div>` : ""}
      <div class="grid2b">
        <div class="preflight">${preflightText(r)}</div>
        <div><div class="secs">${r.sections.map((s) => `<div class="sec ${s.status}"><span><b>${esc(s.label)}</b> <span class="meta">${s.checks} check${s.checks === 1 ? "" : "s"}</span></span><span class="pill ${s.status}">${s.problems ? `${s.problems} problem${s.problems > 1 ? "s" : ""}` : secLabel[s.status]}</span></div>`).join("")}</div>
          ${info.mesh_problem_regions ? `<p class="meta" style="margin-top:10px">Mesh problem regions: ${info.mesh_problem_regions.map((g) => `${esc(g.name)} (${g.cells} cells${g.near ? `, near ${esc(g.near)}` : ""})`).join("; ")}. Open the case in Mesh Quality for the 3D view.</p>` : ""}
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="toolbar"><h3 style="margin:0">Findings</h3>
        <div class="filters">${[["problems", `Problems (${c.fail + c.warn})`], ["all", `All (${r.findings.length})`], ["pass", `Passed (${c.pass + c.info})`]].map(([k, l]) => `<button data-f="${k}" class="${FILTER === k ? "on" : ""}">${l}</button>`).join("")}</div></div>
      ${shown.length ? shown.map(findingHtml).join("") : `<div class="banner ok" style="margin-top:10px">No problems found.</div>`}
      ${window.renderScope ? renderScope(r.scope) : ""}
    </div>`;
  $$(".filters button").forEach((b) => b.onclick = () => { FILTER = b.dataset.f; render(); });
  $("#dl-json").onclick = () => save("preflight.json", JSON.stringify(r, null, 1), "application/json");
  $("#dl-csv").onclick = () => {
    const rows = [["status", "section", "title", "file", "line", "entity", "detail", "fix", "evidence"],
      ...r.findings.map((f) => [f.status, f.section, f.title, f.file, f.line, f.entity, f.detail, f.fix, f.evidence])];
    save("preflight-checklist.csv", rows.map((x) => x.map((v) => `"${String(v ?? "").replace(/"/g, '""')}"`).join(",")).join("\n"), "text/csv");
  };
  $("#dl-md").onclick = () => {
    const box = { pass: "[x]", info: "[x]", warn: "[ ]", fail: "[ ]" };
    let md = `# Simulation pre-flight: ${r.status}\n\n${r.solver}\n\n`;
    for (const s of r.sections) {
      const fs = r.findings.filter((f) => f.section === s.key);
      if (!fs.length) continue;
      md += `## ${s.label}\n\n` + fs.map((f) => `- ${box[f.status]} **${f.status.toUpperCase()}** ${f.title}${f.file ? ` (\`${f.file}${f.line ? `:${f.line}` : ""}\`)` : ""}${f.fix && f.status !== "pass" ? `\n  - Fix: ${f.fix}` : ""}`).join("\n") + "\n\n";
    }
    save("preflight-checklist.md", md, "text/markdown");
  };
}

function save(name, text, type) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

(async function init() {
  try {
    META = await (await fetch("/api/meta")).json();
    const groups = {};
    for (const e of META.examples) (groups[e.group] = groups[e.group] || []).push(e);
    $("#examples").innerHTML = Object.entries(groups).map(([g, es]) => `<h4>${esc(g)}</h4>` + es.map((e) => `<button type="button" data-n="${esc(e.name)}"><b>${esc(e.description)}</b><span class="meta">solver: ${esc(e.solver_result)}</span></button>`).join("")).join("");
    $$("#examples button").forEach((b) => b.addEventListener("click", () => load(`/api/example/${encodeURIComponent(b.dataset.n)}`, {}, b)));
  } catch (e) { console.error(e); }
  const host = location.origin;
  $("#c1").textContent = `# OpenFOAM: zip the case without result time directories
zip -r case.zip 0 constant system
curl -s -F files=@case.zip ${host}/api/preflight | jq '.status, (.findings[] | select(.status=="fail") | .title)'

# CalculiX: the deck (and its *INCLUDE files)
curl -s -F files=@model.inp ${host}/api/preflight | jq .status

# gate a pipeline: do not start the solver on a FAIL
[ "$(curl -s -F files=@case.zip ${host}/api/preflight | jq -r .status)" != "FAIL" ] && ./Allrun`;
  $("#c2").textContent = `from pinneapple_data.cae.upload import expand
from pinneapple_data.preflight import preflight

rep = preflight(expand([("case.zip", open("case.zip", "rb").read())]))
print(rep["status"])
for f in rep["findings"]:
    if f["status"] in ("fail", "warn"):
        print(f["status"], f["title"], f.get("file"), f.get("line"), "->", f.get("fix"))`;
})();
