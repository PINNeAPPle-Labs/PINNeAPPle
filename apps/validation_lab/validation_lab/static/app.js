"use strict";
// Physics Validation Lab UI: inputs (form, configuration or zoo model), the report in three groups, downloads.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => (v === null || v === undefined ? "" : typeof v === "number" ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(4) + "")
  : Array.isArray(v) ? `[${v.map(fmt).join(", ")}]` : typeof v === "object" ? Object.entries(v).map(([k, x]) => `${k}: ${fmt(x)}`).join(", ") : String(v));
const pl = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
let FILES = [], MODE = "form", RES = null, META = null;
const STCLS = { passed: "passed", failed: "failed", warning: "warning" };

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`));
}
$$("#mode button").forEach((b) => b.addEventListener("click", () => {
  MODE = b.dataset.m;
  $$("#mode button").forEach((x) => x.classList.toggle("on", x === b));
  $$("[data-pane]").forEach((p) => (p.hidden = p.dataset.pane !== MODE));
}));

const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => add([...e.dataTransfer.files]));
$("#files").addEventListener("change", (e) => { add([...e.target.files]); e.target.value = ""; });
function add(list) {
  for (const f of list) if (!FILES.some((x) => x.name === f.name)) FILES.push(f);
  if (FILES.some((f) => /pinneapple-check\.toml$/.test(f.name))) $$("#mode button")[1].click();
  renderFiles();
}
function renderFiles() {
  $("#filelist").innerHTML = FILES.map((f, i) => `<div><span>${esc(f.name)} <span class="u">${(f.size / 1024).toFixed(0)} kB</span></span><button data-i="${i}" type="button">×</button></div>`).join("");
  $$("#filelist button").forEach((b) => (b.onclick = () => { FILES.splice(+b.dataset.i, 1); renderFiles(); }));
  const data = FILES.filter((f) => !/\.toml$/.test(f.name)).map((f) => f.name);
  const opts = (sel, none) => (none ? `<option value="">none</option>` : "") + data.map((n) => `<option ${n === sel ? "selected" : ""}>${esc(n)}</option>`).join("");
  $("#f-path").innerHTML = opts($("#f-path").value, false);
  $("#f-ref").innerHTML = opts($("#f-ref").value, true);
  $("#z-queries").innerHTML = opts($("#z-queries").value, true);
}

// "k = 10 W/(m*K), q = 1000" -> {k: "10 W/(m*K)", q: 1000}
function kv(s) {
  const out = {};
  for (const part of s.split(/,(?![^(]*\))/)) {
    const m = part.match(/^\s*([^=]+?)\s*=\s*(.+?)\s*$/);
    if (m) out[m[1]] = isFinite(+m[2]) ? +m[2] : m[2];
  }
  return out;
}
function bounds(s) {
  const out = {};
  for (const part of s.split(",")) {
    const m = part.match(/^\s*([^=]+?)\s*=\s*([^.]*?)\s*\.\.\s*(.*?)\s*$/);
    if (m) out[m[1]] = [m[2] === "" ? "-inf" : +m[2], m[3] === "" ? "inf" : +m[3]];
  }
  return out;
}
function bcs(s) {
  const out = {};
  for (const line of s.split("\n")) {
    const m = line.match(/^\s*([^:]+?)\s*:\s*([^=]+?)\s*=\s*(.+?)\s*$/);
    if (!m) continue;
    let v = m[3];
    if (/^\[.*\]$/.test(v)) v = JSON.parse(v); else if (isFinite(+v)) v = +v;
    (out[m[1]] ||= {})[m[2]] = v;
  }
  return out;
}
function spec() {
  if (MODE === "zoo") {
    const m = { zoo: $("#z-model").value };
    if ($("#z-queries").value) m.queries = $("#z-queries").value;
    return { models: [m] };
  }
  const r = { path: $("#f-path").value };
  if (!r.path) throw new Error("Upload the predictions or result file first.");
  if ($("#f-physics").value) r.physics = $("#f-physics").value;
  const p = kv($("#f-params").value), u = kv($("#f-units").value), b = bounds($("#f-bounds").value), bc = bcs($("#f-bc").value);
  if (Object.keys(p).length) r.parameters = p;
  if (Object.keys(u).length) r.units = Object.fromEntries(Object.entries(u).map(([k, v]) => [k, String(v)]));
  if (Object.keys(b).length) r.bounds = b;
  if (Object.keys(bc).length) r.bc = bc;
  if ($("#f-ref").value) { r.reference = { path: $("#f-ref").value }; if ($("#f-rell2").value) r.reference["rel-l2"] = +$("#f-rell2").value; }
  return { results: [r] };
}

async function call(url, opts, btn) {
  const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Validating…`; $("#status").textContent = "";
  try {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
    RES = j; render(); showTab("result"); window.scrollTo(0, 0);
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; }
}
$("#run").addEventListener("click", (e) => {
  const fd = new FormData();
  FILES.forEach((f) => fd.append("files", f, f.name));
  try {
    if (MODE === "config") fd.append("config", $("#config").value);
    else fd.append("spec", JSON.stringify(spec()));
  } catch (err) { $("#status").innerHTML = `<span class="err">${esc(err.message)}</span>`; return; }
  call("/api/validate", { method: "POST", body: fd }, e.currentTarget);
});

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
const report = ({ views3d, ...rest }) => rest;            // the 3-D views are not part of the hashed report
async function downloadHtml() {
  const r = await fetch("/api/report.html", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(report(RES)) });
  download("validation.html", await r.text(), "text/html");
}

function catCard(key, c) {
  const rows = Object.entries(c.targets).map(([t, s]) => `<tr><td>${esc(t)}</td><td><span class="st ${STCLS[s] || "notrun"}">${esc(s)}</span></td></tr>`).join("");
  return `<div class="cat ${key}"><h3>${esc(c.title)}</h3><div class="q">${esc(c.question)}</div>
    <div class="counts"><b>${c.passed}</b> of ${c.checks} checks passed · <span class="${c.errors ? "err" : ""}">${pl(c.errors, "error")}</span> · ${pl(c.warnings, "warning")}${c.not_run.length ? ` · <i>not run for ${c.not_run.length} target(s)</i>` : ""}</div>
    <table><tr><th>Target</th><th>Status</th></tr>${rows}</table></div>`;
}

function checksTable() {
  const cat = $("#flt-cat")?.value || "", st = $("#flt-st")?.value || "";
  const rows = RES.outcomes.filter((o) => o.rule !== "PC001" && o.category && (!cat || o.category === cat) &&
    (!st || (st === "bad" ? !o.passed : o.passed)));
  return `<table class="checks"><tr><th>Group</th><th>Rule</th><th>Target · subject</th><th>Status</th><th>Message</th><th class="num">Value</th><th class="num">Tolerance</th></tr>` +
    rows.map((o) => {
      const s = o.passed ? "passed" : o.severity === "error" ? "failed" : "warning";
      return `<tr><td>${esc(RES.categories[o.category].title)}</td><td title="${esc(o.title)}"><b>${esc(o.rule)}</b> ${esc(o.name)}</td><td>${esc(o.target)} · ${esc(o.subject)}</td>
        <td><span class="st ${s}">${s}</span></td><td>${esc(o.message)}${!o.passed && o.fix ? `<div class="hint">Fix: ${esc(o.fix)}</div>` : ""}</td><td class="num">${esc(fmt(o.value))}</td><td class="num">${esc(fmt(o.threshold))}</td></tr>`;
    }).join("") + `</table>${rows.length ? "" : "<p class='hint'>No check matches the filter.</p>"}`;
}

function render() {
  const r = RES, s = r.summary, p = r.provenance;
  const label = { pass: "PASS", warn: "WARN", fail: "FAIL" }[r.decision];
  const ex = r.example ? `<p class="hint"><b>Example:</b> ${esc(r.example.title)}. ${esc(r.example.note)}</p>` : "";
  const notes = r.notes.length ? `<div class="banner warn"><b>Could not be checked:</b><ul>${r.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></div>` : "";
  $("#result").innerHTML = `
  <div class="panel">
    ${ex}
    <div class="verdict ${r.decision}"><span class="big">${label}</span><span>${pl(s.checks, "check")} on ${pl(s.targets, "target")}: ${s.passed} passed, ${pl(s.errors, "error")}, ${pl(s.warnings, "warning")}.
      Ran ${esc(p.started_at)} → ${esc(p.finished_at)} with pinneapple check ${esc(p.versions.pinneapple)}.</span></div>
    <p class="note">${esc(r.disclaimer)}</p>
    <div class="toolbar">
      <button class="secondary" id="dl-json" type="button">Download report (JSON)</button>
      <button class="secondary" id="dl-html" type="button">Download report (HTML)</button>
      <button class="secondary" id="dl-toml" type="button">Download configuration</button>
      <button class="secondary" id="dl-ledger" type="button">Download evidence ledger</button>
    </div>
    ${notes}
    <div class="cols">${Object.entries(r.categories).map(([k, c]) => catCard(k, c)).join("")}</div>
  </div>
  ${(r.views3d || []).length ? `<div class="panel">
    <h2 style="margin-top:0">On the mesh: prediction, reference and error (3-D)</h2>
    ${r.example && r.example.id === "cantilever3d" ? `<p class="hint"><b>Example:</b> ${esc(r.example.note)}</p>` : ""}
    <p class="hint">The values compared by the reference check (PC701), at the same points: switch between the prediction, the reference and |prediction − reference|. Declared boundary conditions and the regions with the largest error are marked.</p>
    ${r.views3d.length > 1 ? `<label class="v3pick">Result <select id="v3t">${r.views3d.map((v, i) => `<option value="${i}">${esc(v.target)}</option>`).join("")}</select></label>` : ""}
    <div class="v3wrap"><div id="v3d"></div><div class="v3side" id="v3sum"></div></div></div>` : ""}
  <div class="panel">
    <h2 style="margin-top:0">Every check</h2>
    <div class="filters">Group <select id="flt-cat"><option value="">all</option>${Object.entries(r.categories).map(([k, c]) => `<option value="${k}">${esc(c.title)}</option>`).join("")}</select>
      Status <select id="flt-st"><option value="">all</option><option value="bad">failed or warning</option><option value="ok">passed</option></select></div>
    <div id="checks">${checksTable()}</div>
  </div>
  <div class="panel prov">
    <h2 style="margin-top:0">Provenance and reproduction</h2>
    <table>
      <tr><td>Inputs (manifest) SHA-256</td><td><code>${esc(p.inputs_sha256)}</code> · ${p.inputs.length} files</td></tr>
      <tr><td>Configuration SHA-256</td><td><code>${esc(p.config_sha256)}</code></td></tr>
      <tr><td>Report SHA-256</td><td><code>${esc(r.report_sha256)}</code></td></tr>
      <tr><td>Evidence records</td><td>${r.ledger.entries.length} (one per check: inputs SHA-256, outputs, start and finish time)</td></tr>
      <tr><td>Versions</td><td>${Object.entries(p.versions).map(([k, v]) => `${esc(k)} ${esc(v)}`).join(" · ")}</td></tr>
    </table>
    <pre class="code"><code>${esc(r.reproduce)}</code></pre>
    <details><summary>Configuration (${esc(r.config.split("\n").length)} lines)</summary><pre class="code"><code>${esc(r.config)}</code></pre></details>
    <details><summary>Input files and their SHA-256</summary><table>${p.inputs.map((f) => `<tr><td>${esc(f.path)}</td><td><code>${esc(f.sha256)}</code></td><td class="num">${f.bytes}</td></tr>`).join("")}</table></details>
    ${window.renderScope ? renderScope(r.scope) : ""}
  </div>`;
  $("#flt-cat").onchange = $("#flt-st").onchange = () => ($("#checks").innerHTML = checksTable());
  $("#dl-json").onclick = () => download("validation.json", JSON.stringify(report(r), null, 1), "application/json");
  $("#dl-html").onclick = downloadHtml;
  if ((r.views3d || []).length) { if ($("#v3t")) $("#v3t").onchange = (e) => view3d(+e.target.value); view3d(0); }
  $("#dl-toml").onclick = () => download("pinneapple-check.toml", r.config, "application/toml");
  $("#dl-ledger").onclick = () => download("evidence-ledger.json", JSON.stringify(r.ledger, null, 1), "application/json");
}

// ------------------------------------------------------------------ 3-D view (shared viewer /shared/fieldview3d.js)
let V3 = null;
const whenViewer = () => new Promise((res) => window.FieldView3D ? res() : window.addEventListener("fieldview3d-ready", res, { once: true }));
async function view3d(i) {
  const v = RES.views3d[i];
  await whenViewer();
  const err = v.meta.fields.find((f) => f.kind === "error" && /mises|stress/i.test(f.name)) || v.meta.fields.find((f) => f.kind === "error");
  V3 = new FieldView3D($("#v3d"), { meta: v.meta, glb: v.glb, field: err.name, height: 480, onField: (f) => regions3d(v, f) });
}
function regions3d(v, f) {
  const n = f.name.replace(/:(pred|ref|error)$/, ""), e = v.errors[n];
  if (!e) return;
  const u = e.unit ? ` ${esc(e.unit)}` : "";
  $("#v3sum").innerHTML = `<h4>${esc(v.target)} · ${esc(n)} against ${esc(v.reference)}</h4>${e.summary.map((t) => `<p>${esc(t)}</p>`).join("")}
    ${v.notes.map((t) => `<p class="hint">${esc(t)}</p>`).join("")}
    <h4>Regions with the largest error</h4>
    ${e.hotspots.length ? `<table class="regions"><tr><th>#</th><th class="num">max${u ? ` [${esc(e.unit)}]` : ""}</th><th class="num">mean</th><th class="num">points</th><th class="num">share of squared error</th><th>where</th></tr>
      ${e.hotspots.map((h, k) => `<tr data-k="${k}"><td><b>${esc(h.name)}</b></td><td class="num">${fmt(h.max)}</td><td class="num">${fmt(h.mean)}</td><td class="num">${h.n}</td><td class="num">${(100 * h.share).toFixed(1)} %</td>
        <td>(${h.at.map((x) => +x.toPrecision(3)).join(", ")})${h.touches.length ? `<div class="hint">touches ${esc(h.touches.join(", "))}</div>` : ""}</td></tr>`).join("")}</table>
      <p class="hint">Connected points in the top 10 % of |error|, largest first; click a row to zoom to it.</p>` : `<p class="hint">The error is zero.</p>`}`;
  $$("#v3sum tr[data-k]").forEach((tr) => tr.addEventListener("click", () => { const h = e.hotspots[+tr.dataset.k]; V3.focus(h.at, Math.max(...h.bbox_max.map((x, q) => x - h.bbox_min[q]))); }));
}

(async function init() {
  try {
    META = await (await fetch("/api/meta")).json();
    $("#disclaimer").textContent = META.disclaimer;
    $("#examples").innerHTML = META.examples.map((x) => `<button type="button" data-id="${x.id}"><b>${esc(x.title)}</b><span>${esc(x.note)}</span></button>`).join("");
    $$("#examples button").forEach((b) => b.addEventListener("click", () => call(`/api/example/${b.dataset.id}`, {}, b)));
    $("#z-model").innerHTML = META.zoo.map((m) => `<option>${esc(m)}</option>`).join("");
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  renderFiles();
  const host = location.origin;
  $("#c1").textContent = `# predictions + a pinneapple-check.toml (zips are expanded)
curl -s -F files=@prediction.npz -F files=@reference.npz -F files=@pinneapple-check.toml ${host}/api/validate | jq '.decision, .categories'

# or describe the checks instead of a configuration
curl -s -F files=@T.npz -F 'spec={"results": [{"path": "T.npz", "physics": "heat", "parameters": {"k": "10 W/(m*K)"},
  "units": {"T": "K"}, "bounds": {"T": [250, 400]}}]}' ${host}/api/validate > report.json

# a zoo model (ModelRef) at your operating points
curl -s -F files=@queries.csv -F 'spec={"models": [{"zoo": "heatsink_forced_mlp", "queries": "queries.csv"}]}' ${host}/api/validate`;
  $("#c2").textContent = `# .github/workflows/physics-check.yml
name: Physics check
on: [pull_request]
permissions: { contents: read, security-events: write }
jobs:
  physics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: PINNeAPPle-Labs/PINNeAPPle@main      # pin a release tag or commit SHA
        with:
          config: pinneapple-check.toml
          fail-on: error
      # optional: keep the lab's report with its evidence records for every model version
      - run: |
          zip -r inputs.zip . -x '.git/*'
          curl -sf -u "$VAL_USER:$VAL_PASSWORD" -F files=@inputs.zip ${host}/api/validate > validation.json
        env: { VAL_USER: \${{ secrets.VAL_USER }}, VAL_PASSWORD: \${{ secrets.VAL_PASSWORD }} }`;
  $("#c3").textContent = `from pinneapple_veriphysics.validation_lab import config_from_spec, render_html, run_validation

(work / "pinneapple-check.toml").write_text(config_from_spec({"results": [{"path": "T.npz", "physics": "heat"}]}))
report = run_validation(work).to_dict()          # groups, outcomes, provenance, evidence ledger
print(report["decision"], {k: c["not_run"] for k, c in report["categories"].items()})`;
})();
