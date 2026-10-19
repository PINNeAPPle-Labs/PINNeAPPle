"use strict";
// Engineering Evidence Report UI: sources + metadata -> report (HTML preview, PDF, evidence package).
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let FILES = [], RES = null, LAST = null;          // LAST: how the current report was made (to download PDF / zip)

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`));
}
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => add([...e.dataTransfer.files]));
$("#files").addEventListener("change", (e) => { add([...e.target.files]); e.target.value = ""; });
function add(list) { for (const f of list) if (!FILES.some((x) => x.name === f.name)) FILES.push(f); renderFiles(); }
function renderFiles() {
  $("#filelist").innerHTML = FILES.map((f, i) => `<div><span>${esc(f.name)} <span class="u">${(f.size / 1024).toFixed(0)} kB</span></span><button data-i="${i}" type="button">×</button></div>`).join("");
  $$("#filelist button").forEach((b) => (b.onclick = () => { FILES.splice(+b.dataset.i, 1); renderFiles(); }));
  $("#run").disabled = !FILES.length;
}
const lines = (s) => s.split("\n").map((x) => x.trim()).filter(Boolean);
const pairs = (s) => Object.fromEntries(lines(s).map((l) => l.split(/\s*=\s*/)).filter((p) => p.length >= 2).map(([k, ...v]) => [k, v.join("=")]));
function meta() {
  const m = { title: $("#m-title").value, project: $("#m-project").value, author: $("#m-author").value, purpose: $("#m-purpose").value,
    methodology: $("#m-method").value, baseline: $("#m-base").value, candidate: $("#m-cand").value,
    conditions: pairs($("#m-cond").value), limitations: lines($("#m-lim").value), references: lines($("#m-refs").value) };
  return Object.fromEntries(Object.entries(m).filter(([, v]) => (Array.isArray(v) ? v.length : typeof v === "object" ? Object.keys(v).length : v)));
}
function form() {
  const fd = new FormData();
  FILES.forEach((f) => fd.append("files", f, f.name));
  fd.append("meta", JSON.stringify(meta()));
  fd.append("urls", JSON.stringify(pairs($("#m-urls").value)));
  return fd;
}

async function call(url, opts, btn, label) {
  const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>${label}`; $("#status").textContent = "";
  try {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : j.message || JSON.stringify(j.detail || j));
    RES = j; render(); showTab("result"); window.scrollTo(0, 0);
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; if (btn.id === "run") renderFiles(); }
}
$("#run").addEventListener("click", (e) => { LAST = "upload"; call("/api/report", { method: "POST", body: form() }, e.currentTarget, "Building…"); });
$("#example").addEventListener("click", (e) => { LAST = "example"; call("/api/example", {}, e.currentTarget, "Running the study…"); });

async function download(what, btn) {
  const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>${old}`;
  try {
    const r = LAST === "example" ? await fetch(`/api/example/${what === "pdf" ? "pdf" : "package"}`)
      : await fetch(what === "pdf" ? "/api/report.pdf" : "/api/package", { method: "POST", body: form() });
    if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(j.message || j.detail || `HTTP ${r.status}`); }
    save(await r.blob(), what === "pdf" ? "evidence-report.pdf" : "evidence-package.zip");
  } catch (e) { alert(e.message); } finally { btn.disabled = false; btn.innerHTML = old; }
}
function save(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function render() {
  const r = RES.report, c = r.test_counts;
  const ex = RES.example ? `<p class="hint"><b>Example:</b> a study run on this server just now (${RES.example.files.map((f) => esc(f.name)).join(", ")}).
    <a class="link" href="/api/example/sources">Download its source files</a> to rebuild the report yourself.</p>` : "";
  const arts = r.artifacts.map((a) => `<tr><td>${esc(a.name)}</td><td><span class="kind">${esc(a.kind)}</span></td><td class="num">${a.bytes.toLocaleString()}</td><td><code>${esc(a.sha256)}</code></td></tr>`).join("");
  const unrun = r.tests.filter((t) => t.status === "not run");
  $("#result").innerHTML = `
  <div class="panel">
    ${ex}
    <div class="tiles">
      <div class="tile passed"><div class="v">${c.passed}</div><div class="l">tests passed, with a record that they ran</div></div>
      <div class="tile failed"><div class="v">${c.failed}</div><div class="l">tests failed, with a record</div></div>
      <div class="tile notrun"><div class="v">${c["not run"]}</div><div class="l">not run: no intact record</div></div>
      <div class="tile"><div class="v">${r.artifacts.length}</div><div class="l">original artifacts, by SHA-256</div></div>
    </div>
    ${unrun.length ? `<div class="banner warn"><b>Printed as not run:</b> ${unrun.slice(0, 6).map((t) => `${esc(t.name)} (${esc(t.source)}${t.source_claim ? `, which claims ${esc(t.source_claim)}` : ""}: ${esc(t.why)})`).join("; ")}${unrun.length > 6 ? ` and ${unrun.length - 6} more` : ""}.</div>` : ""}
    <div class="toolbar">
      <button class="primary" id="dl-pdf" type="button">Download PDF</button>
      <button class="secondary" id="dl-zip" type="button">Download evidence package (zip)</button>
      <button class="secondary" id="dl-html" type="button">Download HTML</button>
      <button class="secondary" id="dl-json" type="button">Download report JSON</button>
      <button class="secondary" id="dl-sr" type="button">Download StudyResult</button>
    </div>
    <p class="hint">Report SHA-256 <code>${esc(r.report_sha256)}</code> · generated ${esc(r.generated_at)}</p>
  </div>
  <div class="panel"><iframe class="report" id="frame" title="Report preview"></iframe></div>
  <div class="panel"><h2 style="margin-top:0">How each source was read</h2>
    <table><tr><th>Artifact</th><th>Read as</th><th class="num">Bytes</th><th>SHA-256</th></tr>${arts}</table>
    ${window.renderScope ? renderScope(RES.scope) : ""}</div>`;
  $("#frame").srcdoc = RES.html;
  $("#dl-pdf").onclick = (e) => download("pdf", e.currentTarget);
  $("#dl-zip").onclick = (e) => download("zip", e.currentTarget);
  $("#dl-html").onclick = () => save(new Blob([RES.html], { type: "text/html" }), "evidence-report.html");
  $("#dl-json").onclick = () => save(new Blob([JSON.stringify(r, null, 1)], { type: "application/json" }), "evidence-report.json");
  $("#dl-sr").onclick = () => save(new Blob([JSON.stringify(RES.study_result, null, 1)], { type: "application/json" }), "study_result.json");
}

(async function init() {
  try {
    const m = await (await fetch("/api/meta")).json();
    $("#principle").textContent = m.principle;
    $("#kinds").innerHTML = `<h3>Read automatically</h3><table>${Object.entries(m.kinds).map(([k, v]) => `<tr><td><span class="kind">${esc(k)}</span></td><td>${esc(v)}</td></tr>`).join("")}</table>`;
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  const host = location.origin;
  $("#c1").textContent = `# report JSON + HTML
curl -s -F files=@validation.json -F files=@metrics.csv -F files=@study_result.json \\
  -F 'meta={"title": "Heat sink v2", "baseline": "CFD", "candidate": "surrogate"}' \\
  -F 'urls={"case.zip": "s3://runs/42/case.zip"}' ${host}/api/report | jq '.report.test_counts'

# the PDF and the evidence package (report, ledger, figures, original files, SHA-256 manifest)
curl -s -F files=@validation.json -F files=@metrics.csv -F 'meta={"title": "Heat sink v2"}' ${host}/api/report.pdf -o report.pdf
curl -s -F files=@validation.json -F files=@metrics.csv -F 'meta={"title": "Heat sink v2"}' ${host}/api/package -o evidence.zip`;
  $("#c2").textContent = `from pinneapple_veriphysics.evidence_package import build_package, build_report, load_source, verify_package
from pinneapple_veriphysics.ledger import EvidenceLedger, sha256_bytes

ledger = EvidenceLedger()                       # record each test where it runs
with ledger.run("energy balance", inputs_sha256=sha256_bytes(field_bytes), tool="my-check") as rec:
    rec.outputs = {"imbalance": imbalance}
    rec.passed = imbalance < 1e-2

srcs = [load_source(name, data) for name, data in files]          # + the ledger as a JSON file
report = build_report({"title": "Heat sink v2", "baseline": "CFD", "candidate": "surrogate"}, srcs)
zip_bytes = build_package(report, srcs)
assert verify_package(zip_bytes) == []`;
})();
