"use strict";
// Interoperability Hub UI: upload, normalise, show the neutral dataset, preview, export.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => (v === null || v === undefined || !isFinite(v) ? "—" : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(3) : +v.toPrecision(5) + "");
let FILES = [], R = null, META = null;

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`));
  if (n === "dataset" && window.IOP_VIEW) window.IOP_VIEW.resize();
}
async function filesFromDrop(dt) {
  const out = [];
  const walk = (entry, path) => new Promise((res) => {
    if (entry.isFile) entry.file((f) => { f.relPath = path + f.name; out.push(f); res(); });
    else if (entry.isDirectory) { const rd = entry.createReader(), all = []; const read = () => rd.readEntries(async (ents) => { if (!ents.length) { for (const en of all) await walk(en, path + entry.name + "/"); res(); } else { all.push(...ents); read(); } }); read(); }
    else res();
  });
  const items = [...(dt.items || [])].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  if (!items.length) return [...dt.files];
  for (const en of items) await walk(en, "");
  return out;
}
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async (e) => add(await filesFromDrop(e.dataTransfer)));
$("#files").addEventListener("change", (e) => { add([...e.target.files]); e.target.value = ""; });
function add(list) { for (const f of list) { const n = f.relPath || f.webkitRelativePath || f.name; if (!FILES.some((x) => x.n === n)) FILES.push({ f, n }); } renderFiles(); }
function renderFiles() {
  $("#filelist").innerHTML = FILES.slice(0, 6).map((x, i) => `<div><span><b>${esc(x.n)}</b> · ${(x.f.size / 1024).toFixed(0)} KB</span><button data-i="${i}">×</button></div>`).join("") + (FILES.length > 6 ? `<div><span>+ ${FILES.length - 6} files</span><button data-i="all">clear</button></div>` : "");
  $$("#filelist button").forEach((b) => b.onclick = () => { if (b.dataset.i === "all") FILES = []; else FILES.splice(+b.dataset.i, 1); renderFiles(); });
  $("#run").disabled = !FILES.length;
}

async function load(url, opts, btn) {
  const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Normalising…`; $("#status").textContent = "";
  try {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
    R = j; render(); showTab("dataset"); window.scrollTo(0, 0);
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; if (btn.id === "run") renderFiles(); }
}
$("#run").addEventListener("click", (e) => {
  const fd = new FormData(); FILES.forEach((x) => fd.append("files", x.f, x.n));
  fd.append("time", $("#time").value); fd.append("unit_system", $("#units").value); fd.append("location", $("#location").value);
  fd.append("to_si", $("#tosi").checked ? "true" : "false"); if ($("#rho").value) fd.append("rho", $("#rho").value);
  load("/api/inspect", { method: "POST", body: fd }, e.currentTarget);
});

// the dataset as JSON (what a user of the neutral format sees), values summarised
function jsonView(d) {
  const view = {
    schema: d.schema,
    geometry: { dimension: d.geometry.dimension, length_unit: d.geometry.length_unit, bounding_box: d.geometry.bounding_box },
    mesh: d.mesh,
    coordinates: { location: d.coordinates.location, count: d.coordinates.count, unit: d.coordinates.unit, values: `<${d.coordinates.count} × 3>` },
    fields: Object.fromEntries(Object.entries(d.fields).map(([k, f]) => [k, { quantity: f.quantity, unit: f.unit, location: f.location, components: f.component_names || f.components, values: `<${f.location === d.coordinates.location ? d.coordinates.count : "…"} values>`, min: f.min, max: f.max }])),
    metadata: Object.fromEntries(Object.entries(d.metadata).filter(([k]) => k !== "files" && k !== "times_available")),
  };
  const s = JSON.stringify(view, (k, v) => (typeof v === "number" ? +fmt(v) || v : v), 2);
  return esc(s).replace(/&quot;([^&]+)&quot;:/g, '<span class="k">"$1"</span>:').replace(/: &quot;(&lt;[^&]*&gt;)&quot;/g, ': <span class="c">$1</span>')
    .replace(/: &quot;([^&]*)&quot;/g, ': <span class="s">"$1"</span>').replace(/: (-?[\d.eE+-]+)(,?)$/gm, ': <span class="n">$1</span>$2');
}

function render() {
  const d = R.dataset, md = d.metadata;
  const rows = Object.entries(d.fields).map(([k, f]) => `<tr><td><b>${esc(k)}</b></td><td>${esc(f.quantity || "—")}</td><td>${esc(f.unit || "—")}</td><td>${esc(f.location)}</td><td>${f.component_names ? esc(f.component_names.join(", ")) : f.components}</td>${statCells(f)}</tr>`).join("");
  const exp = Object.entries(R.formats).map(([k, v]) => `<div class="exp"><b>${esc(v.split(":")[0])}</b><span>${esc(v.split(":").slice(1).join(":"))}</span><a class="link" href="/api/export/${R.id}/${k}" download>Download</a></div>`).join("");
  const PF = (R.preview && R.preview.fields) || {};
  const fieldOpts = Object.keys(PF).map((k) => `<option value="${esc(k)}">${esc(k)}${PF[k].kind && PF[k].kind !== "value" ? ` (${esc(PF[k].kind)})` : ""}</option>`).join("");
  const hasMesh = !!(R.preview && R.preview.points && R.preview.points.length);
  $("#dataset").innerHTML = `
    <div class="panel">
      <div class="toolbar"><h2 style="margin:0">${esc(R.example ? R.example.description : md.case || "Dataset")}</h2><div class="noprint"><button class="ghost" onclick="print()">Print / PDF</button></div></div>
      <p class="meta">${esc(md.solver || md.source_format)} → <b>${esc(d.schema)}</b> · ${d.mesh.cells.toLocaleString("en-US")} cells · ${d.mesh.points.toLocaleString("en-US")} points${md.time !== null && md.time !== undefined ? ` · t = ${fmt(md.time)}` : ""} · ${Object.keys(d.fields).length} fields · units: ${esc(md.unit_system)}${md.converted_to_si ? " → SI" : ""}</p>
      ${md.notes && md.notes.length ? `<div class="notes">${md.notes.map(esc).join("<br>")}</div>` : ""}
      <div class="grid2b"><div class="json">${jsonView(d)}</div>
        <div>${hasMesh ? `<div class="mapbar"><b>Preview</b>${fieldOpts ? `<select id="pfield">${fieldOpts}</select><span class="meta">on the boundary surface</span>` : `<span class="meta">mesh only (no fields): the boundary surface with its edges</span>`}</div><div id="viewer"></div>` : `<div class="banner">No surface to preview.</div>`}</div></div>
    </div>
    <div class="panel"><h3 style="margin-top:0">Fields</h3>
      <div style="overflow-x:auto"><table><thead><tr><th>Field</th><th>Quantity</th><th>Unit</th><th>Location</th><th>Components</th><th class="num">Min</th><th class="num">Mean</th><th class="num">Max</th><th>Of</th></tr></thead><tbody>${rows || `<tr><td colspan="9" class="meta">No fields: geometry and mesh only.</td></tr>`}</tbody></table></div></div>
    <div class="panel"><h3 style="margin-top:0">Export</h3><div class="exports">${exp}</div>
      ${md.files ? `<p class="meta" style="margin-top:12px">Source files (sha256 prefix): ${md.files.slice(0, 8).map((f) => `${esc(f.name)} ${esc(f.sha256)}`).join(" · ")}${md.files.length > 8 ? ` · +${md.files.length - 8}` : ""}</p>` : ""}
      ${window.renderScope ? renderScope(R.scope) : ""}</div>`;
  if (hasMesh) {
    const mount = () => { window.IOP_VIEW = new window.PreviewViewer($("#viewer"), R.preview, fieldOpts ? $("#pfield").value : null); };
    if (window.PreviewViewer) mount(); else window.addEventListener("iop-viewer-ready", mount, { once: true });
    if (fieldOpts) $("#pfield").onchange = () => { window.IOP_VIEW && window.IOP_VIEW.paint($("#pfield").value); };
  }
}
// min / mean / max: of the magnitude for vectors, of von Mises for stresses (component ranges in the tooltip)
function statCells(f) {
  const s = f.scalar;
  if (!s) return `<td class="num">${fmt(f.min)}</td><td class="num">${fmt(f.mean)}</td><td class="num">${fmt(f.max)}</td><td class="meta">value</td>`;
  const tip = `components range from ${fmt(f.min)} to ${fmt(f.max)}`;
  return `<td class="num" title="${tip}">${fmt(s.min)}</td><td class="num" title="${tip}">${fmt(s.mean)}</td><td class="num" title="${tip}">${fmt(s.max)}</td><td class="meta" title="${tip}">${esc(s.kind)}</td>`;
}

(async function init() {
  try {
    META = await (await fetch("/api/meta")).json();
    $("#examples").innerHTML = META.examples.map((e) => `<button type="button" data-n="${esc(e.name)}"><b>${esc(e.name)}</b>${esc(e.description)}</button>`).join("");
    $$("#examples button").forEach((b) => b.addEventListener("click", () => load(`/api/example/${encodeURIComponent(b.dataset.n)}`, {}, b)));
  } catch (e) { console.error(e); }
  const host = location.origin;
  $("#c1").textContent = `# one call: an OpenFOAM case to VTU (latest time), p converted to Pa with rho
curl -s -F files=@case.zip -F rho=998 -o case.vtu ${host}/api/convert/vtu
# CalculiX results to SI Parquet
curl -s -F files=@beam.frd -F unit_system=N-mm-t-s -F to_si=true -o beam.parquet ${host}/api/convert/parquet

# or inspect once, then export several formats
ID=$(curl -s -F files=@case.zip ${host}/api/inspect | jq -r .id)
curl -s -o case.h5 ${host}/api/export/$ID/hdf5
curl -s -o case.zarr.zip ${host}/api/export/$ID/pinneapple`;
  $("#c2").textContent = `import pyvista as pv;  grid = pv.read("case.vtu")                      # ParaView / PyVista
import pandas as pd;   df = pd.read_parquet("case.parquet")                # one row per cell
import h5py;           h = h5py.File("case.h5"); h["fields/U"].attrs["unit"]

# PINNeAPPle dataset: a PhysicalSample in the library's Zarr store
import zipfile
from pinneapple_data.serialization import load_zarr
zipfile.ZipFile("case.zarr.zip").extractall(".")
sample = load_zarr("dataset.zarr")[0]
sample.state["U"], sample.provenance["fields"]["U"]["unit"]               # (n, 3) array, "m/s"`;
})();
