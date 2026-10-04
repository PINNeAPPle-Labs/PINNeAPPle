"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let META = null, DOCS = [], MODE = null, RESULT = null, FORM = null, FILTER = "attention";
const TEMPLATE = () => $("#template").value;
const FORMAT = () => META.formats.find((f) => f.id === TEMPLATE());
const SPEC = () => (RESULT && RESULT.spec) || META.spec;
const OFFICIAL = () => FORMAT().output === "official form";
const OVERRIDES = {};   // key -> value typed or chosen by the engineer

// ── tabs ─────────────────────────────────────────────────────────────────────
$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`));
}

async function call(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
  }
  return r;
}
function busy(btn, text) {
  const old = btn.textContent; btn.disabled = true;
  btn.innerHTML = `<span class="spinner" style="border-color:#a14a1255;border-top-color:#a14a12"></span>${text}`;
  return () => { btn.disabled = false; btn.textContent = old; };
}

// ── document format ──────────────────────────────────────────────────────────
function renderFormat() {
  const f = FORMAT();
  $("#fmtcard").innerHTML = `
    <div class="fmt-std">${esc(f.standard)}</div>
    <div class="fmt-sum">${esc(f.summary)}</div>
    <div class="fmt-meta"><span><b>${f.items}</b> items · <b>${f.required}</b> required${f.tables.length ? ` · ${esc(f.tables.join(", ").toLowerCase())}` : ""}</span>
      <span class="badge ${f.output === "official form" ? "pass" : "na"}">${f.output === "official form" ? "fills the official form" : "generates the datasheet (PDF)"}</span></div>
    <div class="fmt-in">Data usually in: ${f.inputs.filter((r) => r !== "other").map(esc).join(" · ")}</div>`;
  const ex = f.example;
  $("#example").hidden = !ex;
  if (ex) {
    $("#example").textContent = `Try the example: ${ex.label}, ${ex.files.length} documents`;
    $("#exlinks").innerHTML = ex.files.map((x) => `<a class="link" href="/api/example/files/${f.id}/${encodeURIComponent(x.name)}">${esc(x.name)}</a>`).join(" · ");
  }
  if (RESULT && RESULT.spec.id !== f.id) { RESULT = null; MODE = null; resetViews(); }
  DOCS.forEach((d) => { if (!f.inputs.includes(d.role)) d.role = guessRole(d.file.name); });
  renderDocs(); renderOutput();
}
$("#template").addEventListener("change", renderFormat);
function resetViews() {
  $("#review").innerHTML = '<div class="empty"><h3>Nothing compiled yet</h3><p>Add documents or load the example first.</p></div>';
  $("#docs-view").innerHTML = $("#docs-intro").innerHTML;
  for (const k of Object.keys(OVERRIDES)) delete OVERRIDES[k];
}

// ── documents (order = priority) ─────────────────────────────────────────────
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => addDocs([...e.dataTransfer.files]));
$("#files").addEventListener("change", (e) => { addDocs([...e.target.files]); e.target.value = ""; });

function guessRole(name) {
  const n = name.toLowerCase().replace(/[_\-.]+/g, " ");
  const roles = FORMAT().inputs;
  const hit = roles.find((r) => r !== "other" && r.split(" ").every((w) => w.length < 4 || n.includes(w.slice(0, 5))));
  if (hit) return hit;
  const loose = roles.find((r) => r !== "other" && n.includes(r.split(" ")[0].slice(0, 5)));
  return loose || (/spec|standard|requirement/.test(n) && roles.find((r) => r.includes("specification"))) || "other";
}
function addDocs(list) {
  for (const f of list) if (!DOCS.some((d) => d.file.name === f.name)) DOCS.push({ file: f, role: guessRole(f.name) });
  DOCS = DOCS.slice(0, META ? META.max_files : 8);
  renderDocs();
}
function renderDocs() {
  const roles = META ? FORMAT().inputs : [];
  $("#filelist").innerHTML = DOCS.map((d, i) => `
    <div class="docrow"><span class="prio">${i + 1}</span>
      <span class="dn"><b>${esc(d.file.name)}</b><select data-i="${i}">${roles.map((r) => `<option ${r === d.role ? "selected" : ""}>${r}</option>`).join("")}</select></span>
      <span class="ops"><button data-up="${i}" title="Higher priority" ${i ? "" : "disabled"}>↑</button><button data-x="${i}" title="Remove">×</button></span></div>`).join("");
  $$("#filelist select").forEach((s) => s.addEventListener("change", () => { DOCS[+s.dataset.i].role = s.value; }));
  $$("#filelist [data-up]").forEach((b) => b.addEventListener("click", () => { const i = +b.dataset.up; [DOCS[i - 1], DOCS[i]] = [DOCS[i], DOCS[i - 1]]; renderDocs(); }));
  $$("#filelist [data-x]").forEach((b) => b.addEventListener("click", () => { DOCS.splice(+b.dataset.x, 1); renderDocs(); }));
  $("#compile").disabled = !DOCS.length;
}

function formData(withForm) {
  const fd = new FormData();
  DOCS.forEach((d) => fd.append("files", d.file));
  fd.append("roles", JSON.stringify(DOCS.map((d) => d.role)));
  fd.append("use_llm", $("#llm").checked ? "true" : "false");
  fd.append("overrides", JSON.stringify(OVERRIDES));
  fd.append("template", TEMPLATE());
  if (withForm && FORM) fd.append("form", FORM);
  return fd;
}
const exampleUrl = () => `/api/example/compile?template=${encodeURIComponent(TEMPLATE())}&overrides=${encodeURIComponent(JSON.stringify(OVERRIDES))}`;

async function compile(mode) {
  const btn = mode === "example" ? $("#example") : $("#compile");
  const done = busy(btn, $("#llm").checked && mode !== "example" ? "Reading (local LLM: may take minutes)…" : "Reading documents…");
  $("#status").textContent = "";
  try {
    if (mode !== MODE) for (const k of Object.keys(OVERRIDES)) delete OVERRIDES[k];
    MODE = mode;
    const r = mode === "example" ? await call(exampleUrl()) : await call("/api/compile", { method: "POST", body: formData(false) });
    RESULT = await r.json();
    renderDocsView(); renderReview(); renderOutput();
    showTab("review");
  } catch (e) {
    $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`;
  } finally { done(); }
}
$("#compile").addEventListener("click", () => compile("upload"));
$("#example").addEventListener("click", () => compile("example"));

// Re-run with the engineer's answers (overrides).
async function recompile() {
  if (!MODE) return;
  try {
    const r = MODE === "example" ? await call(exampleUrl()) : await call("/api/compile", { method: "POST", body: formData(false) });
    RESULT = await r.json();
    renderReview(); renderOutput();
  } catch (e) { $("#review-status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}

// ── documents panel ──────────────────────────────────────────────────────────
function renderDocsView() {
  const docs = RESULT.documents.map((d) => `<tr><td class="num">${d.priority}</td><td><b>${esc(d.name)}</b></td><td>${esc(d.role || "—")}</td>
    <td class="num">${d.pages}</td><td>${d.has_text ? '<span class="badge pass">text layer</span>' : '<span class="badge fail">scanned: no text</span>'}</td></tr>`).join("");
  $("#docs-view").innerHTML = `<h2 style="margin-top:0">Documents read</h2>
    <table><thead><tr><th class="num">Priority</th><th>Document</th><th>Role</th><th class="num">Pages</th><th>Text</th></tr></thead><tbody>${docs}</tbody></table>
    <p class="hint" style="margin-top:8px">Format: <b>${esc(RESULT.spec.title)}</b> (${esc(RESULT.spec.standard)}) · ${RESULT.spec.fields.length} items</p>
    ${RESULT.rejected && RESULT.rejected.length ? `<div class="warnings" style="margin-top:12px"><b>Local LLM answers not used (${RESULT.rejected.length})</b>
      ${RESULT.rejected.slice(0, 12).map((r) => `<div>${esc(r.doc || "")}: ${esc(r.key || "")} ${r.value ? `“${esc(r.value)}”` : ""} — ${esc(r.reason)}</div>`).join("")}</div>` : ""}`;
}

// ── review ───────────────────────────────────────────────────────────────────
const shown = (c) => (c.unit && typeof c.value === "number" ? `${c.value} ${c.unit}` : Array.isArray(c.value) ? c.value.join(", ") : String(c.value));
const STATUS = { filled: ["pass", "found"], conflict: ["warn", "conflict"], missing: ["fail", "missing"], optional: ["na", "not stated"] };

function sourceCell(c) {
  if (!c) return "";
  return `<div class="src"><span class="docp">${esc(c.doc)} · p.${c.page}${c.method === "llm" ? " · local LLM" : ""}</span><span class="snip">${esc(c.snippet)}</span></div>`;
}
function inputFor(f, d) {
  const cur = OVERRIDES[f.key] ?? "";
  if (f.kind === "bool" || f.kind === "choice") {
    return `<select data-k="${f.key}"><option value="">${d.value != null ? "keep: " + esc(d.display) : "—"}</option>${f.options.map((o) => `<option ${o === cur ? "selected" : ""}>${o}</option>`).join("")}</select>`;
  }
  return `<input data-k="${f.key}" value="${esc(cur)}" placeholder="${d.value != null ? "keep: " + esc(d.display) : f.kind === "quantity" ? "value and unit" : "value"}">`;
}

function renderReview() {
  const s = RESULT.summary;
  const spec = SPEC();
  const byKey = Object.fromEntries(spec.fields.map((f) => [f.key, f]));
  const gaps = RESULT.gaps.map((k) => byKey[k]);
  const sections = spec.sections.map((sec) => {
    const rows = spec.fields.filter((f) => f.section === sec).map((f) => {
      const d = RESULT.decisions[f.key];
      const show = FILTER === "all" || (FILTER === "found" && d.value != null) ||
        (FILTER === "attention" && (d.status === "conflict" || (d.status === "missing" && f.required)));
      if (!show) return "";
      const [cls, txt] = STATUS[d.status];
      const others = d.candidates.slice(1).filter((c) => d.status === "conflict" && shown(c) !== shown(d.chosen || {}));
      const choose = d.status === "conflict" ? `<div class="alts">${[d.candidates[0], ...others].map((c, i) =>
        `<label><input type="radio" name="alt-${f.key}" data-k="${f.key}" data-v="${esc(shown(c))}" ${i === 0 && !(f.key in OVERRIDES) ? "checked" : ""}> <b>${esc(shown(c))}</b> <span class="docp">${esc(c.doc)} p.${c.page}: “${esc(c.snippet)}”</span></label>`).join("")}</div>` : "";
      return `<tr class="st-${d.status}"><td>${esc(f.label)}${f.required ? ' <span class="req" title="needed before design can start">required</span>' : ""}</td>
        <td><b>${d.value != null ? esc(d.display) : "—"}</b>${d.note ? `<div class="note">${esc(d.note)}</div>` : ""}${choose}</td>
        <td><span class="badge ${cls}">${txt}</span></td><td>${sourceCell(d.chosen)}</td><td class="edit">${inputFor(f, d)}</td></tr>`;
    }).join("");
    return rows ? `<h3>${esc(sec)}</h3><table class="rev"><thead><tr><th style="width:24%">Item</th><th style="width:22%">Value</th><th style="width:8%">Status</th><th>Source (document · page · text)</th><th style="width:16%">Your answer</th></tr></thead><tbody>${rows}</tbody></table>` : "";
  }).join("");
  const tables = spec.tables.map((t) => {
    const rows = (RESULT.tables || {})[t.key] || [];
    if (!rows.length) return `<h3>${esc(t.label)}</h3><p class="hint">No ${esc(t.label.toLowerCase())} table found in the documents.</p>`;
    return `<h3>${esc(t.label)} (${rows.length} rows, from ${esc(rows[0].doc)})</h3>
      <table><thead><tr>${t.columns.map((c) => `<th>${esc(c.replace(/_/g, " "))}</th>`).join("")}<th>Source</th></tr></thead><tbody>
      ${rows.map((n) => `<tr>${t.columns.map((c) => `<td>${esc(n.value[c])}</td>`).join("")}<td><span class="docp">${esc(n.doc)} p.${n.page}${n.method === "llm" ? " · local LLM" : ""}</span></td></tr>`).join("")}</tbody></table>
      ${t.rows && rows.length > t.rows ? `<p class="hint">The form has ${t.rows} rows; the rest go on a continuation sheet.</p>` : ""}`;
  }).join("");
  const banner = !s.required ? `<div class="banner warn">This form marks no item as required: ${s.filled} of ${s.fields} items found. Check the gaps that matter to you in "All items".</div>` : s.gaps ? `<div class="banner warn">${s.required_filled} of ${s.required} required items found. ${s.gaps} missing and ${s.conflicts} in conflict: resolve them before issuing.</div>`
    : s.conflicts ? `<div class="banner warn">All ${s.required} required items found; ${s.conflicts} conflict(s) between documents to confirm.</div>`
      : `<div class="banner ok">All ${s.required} required items found, no conflicts between documents.</div>`;
  $("#review").innerHTML = `
    <div class="toolbar"><h2 style="margin:0">Review</h2>
      <div class="seg small" id="filter">${[["attention", "Needs attention"], ["found", "Found"], ["all", `All ${spec.fields.length} items`]].map(([k, l]) => `<button data-f="${k}" class="${FILTER === k ? "on" : ""}">${l}</button>`).join("")}</div></div>
    ${banner}
    <div class="kpis">
      <div class="kpi"><div class="l">Items found</div><div class="v">${s.filled}</div><div class="s">of ${s.fields} in this format</div></div>
      <div class="kpi"><div class="l">Required found</div><div class="v">${s.required_filled}/${s.required}</div><div class="s">needed to start the design</div></div>
      <div class="kpi"><div class="l">Conflicts</div><div class="v">${s.conflicts}</div><div class="s">documents disagree</div></div>
      <div class="kpi"><div class="l">Gaps</div><div class="v">${s.gaps}</div><div class="s">required, stated nowhere</div></div>
      ${spec.tables.length ? `<div class="kpi"><div class="l">${esc(spec.tables[0].label)}</div><div class="v">${s.table_rows}</div><div class="s">rows</div></div>` : ""}
    </div>
    ${gaps.length ? `<div class="gaps"><b>Ask for these before issuing:</b><ul>${gaps.map((f) => `<li>${esc(f.label)}${f.hint ? ` <span class="docp">usually in the ${esc(f.hint)}</span>` : ""}</li>`).join("")}</ul></div>` : ""}
    <div class="toolbar" style="margin-top:12px"><span class="hint">Type an answer or pick a source to override; it is used in the form and marked as entered by the engineer.</span>
      <button class="ghost" id="apply">Apply my answers</button></div><div id="review-status"></div>
    ${sections || '<p class="hint" style="margin-top:14px">Nothing in this view. Choose another filter.</p>'}
    ${tables}
    ${window.renderScope ? renderScope(RESULT.scope) : ""}`;
  $$("#filter button").forEach((b) => b.addEventListener("click", () => { FILTER = b.dataset.f; renderReview(); }));
  $$("#review [data-k]").forEach((el) => el.addEventListener(el.type === "radio" ? "change" : "change", () => {
    const k = el.dataset.k, v = el.type === "radio" ? el.dataset.v : el.value.trim();
    if (v) OVERRIDES[k] = v; else delete OVERRIDES[k];
  }));
  $("#apply").addEventListener("click", async () => { const done = busy($("#apply"), "Applying…"); await recompile(); done(); });
}

// ── output ───────────────────────────────────────────────────────────────────
$("#formfile").addEventListener("change", (e) => { FORM = e.target.files[0] || null; $("#formname").textContent = FORM ? FORM.name : "not chosen"; renderOutput(); });
function renderOutput() {
  const f = FORMAT(), official = OFFICIAL();
  $("#outtitle").textContent = f.title;
  $("#official").hidden = !official;
  $("#formintro").innerHTML = `The form is ASME's document, so it is not bundled: upload your copy of the fillable <b>Form U-DR-1 (User's Design Requirements for Single-Chamber Pressure Vessels)</b>, 07/25 revision, from asme.org. ${META.form_on_server ? "A copy is already configured on this server; uploading one is optional." : ""}`;
  $("#dsintro").textContent = official
    ? "Also available: the compiled data as a datasheet, with the source of every value and the open items first."
    : `The output is a compiled datasheet following ${f.standard}: every item with its value and source, open items first, marked as a draft for engineering review.`;
  $("#fill").disabled = !(RESULT && (FORM || META.form_on_server));
  for (const id of ["datasheet", "checklist", "record"]) $(`#${id}`).disabled = !RESULT;
  $("#specdl").href = `/api/spec?template=${encodeURIComponent(f.id)}`;
  if (!RESULT) { $("#formsummary").innerHTML = '<p class="hint">Compile documents (or load the example) first.</p>'; return; }
  const s = RESULT.summary, fp = RESULT.form_preview;
  $("#formsummary").innerHTML = `<div class="kpis">
      <div class="kpi"><div class="l">Items found</div><div class="v">${s.filled}</div><div class="s">of ${s.fields}</div></div>
      <div class="kpi"><div class="l">Required found</div><div class="v">${s.required_filled}/${s.required}</div></div>
      ${official ? `<div class="kpi"><div class="l">Fields to fill</div><div class="v">${fp.text_fields}</div><div class="s">+ ${fp.check_boxes} check boxes</div></div>` : ""}
      <div class="kpi"><div class="l">Open items</div><div class="v">${s.gaps + s.conflicts}</div><div class="s">gaps + conflicts</div></div></div>
    ${s.gaps + s.conflicts ? '<p class="hint">Open items are listed at the top of the datasheet; resolve them in the Review tab to remove them.</p>' : ""}`;
}
async function getPdf(btn, statusEl, request, fallback, okText) {
  const done = busy(btn, "Preparing…");
  statusEl.textContent = "";
  try {
    const r = await request();
    const name = (r.headers.get("content-disposition") || "").match(/filename="([^"]+)"/)?.[1] || fallback;
    download(await r.blob(), name);
    statusEl.innerHTML = `<span class="ok">Downloaded ${esc(name)}. ${okText}</span>`;
  } catch (e) { statusEl.innerHTML = `<span class="err">${esc(e.message)}</span>`; } finally { done(); }
}
$("#fill").addEventListener("click", () => getPdf($("#fill"), $("#fillstatus"), () => {
  if (MODE === "example") {
    const fd = new FormData(); if (FORM) fd.append("form", FORM); fd.append("overrides", JSON.stringify(OVERRIDES));
    return call("/api/example/fill", { method: "POST", body: fd });
  }
  return call("/api/fill", { method: "POST", body: formData(true) });
}, "filled_form.pdf", "Review it, then date and sign."));
$("#datasheet").addEventListener("click", () => getPdf($("#datasheet"), $("#dsstatus"), () => {
  if (MODE === "example") {
    const fd = new FormData(); fd.append("template", TEMPLATE()); fd.append("overrides", JSON.stringify(OVERRIDES));
    return call("/api/example/datasheet", { method: "POST", body: fd });
  }
  return call("/api/datasheet", { method: "POST", body: formData(false) });
}, "datasheet.pdf", "Check the open items, then approve."));
$("#checklist").addEventListener("click", () => {
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = [["section", "item", "required", "status", "value", "document", "page", "source text", "note"].join(",")];
  const spec = SPEC();
  for (const f of spec.fields) {
    const d = RESULT.decisions[f.key], c = d.chosen || {};
    lines.push([f.section, f.label, f.required ? "yes" : "", d.status, d.value != null ? d.display : "", c.doc, c.page, c.snippet, d.note].map(q).join(","));
  }
  for (const t of spec.tables) ((RESULT.tables || {})[t.key] || []).forEach((n, i) => lines.push([t.label, `Row ${i + 1}`, "", "filled",
    t.columns.map((c) => n.value[c]).join(" | "), n.doc, n.page, n.snippet, ""].map(q).join(",")));
  download(new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv" }), `${spec.id}_data_checklist.csv`);
});
$("#record").addEventListener("click", () => {
  const rec = { form: SPEC().id, standard: SPEC().standard, documents: RESULT.documents, data: RESULT.record };
  download(new Blob([JSON.stringify(rec, null, 1)], { type: "application/json" }), `${SPEC().id}_data_record.json`);
});
function download(blob, name) { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click(); }

// ── API snippets + init ──────────────────────────────────────────────────────
function snippets() {
  const base = location.origin;
  $("#c1").textContent = `curl -s ${base}/api/compile -F template=psv \\\n  -F files=@relief_load_summary.pdf -F files=@psv_sizing.pdf -F files=@valve_specification.pdf \\\n  -F 'roles=["relief load summary","PSV sizing calculation","valve specification"]' | jq '.summary, .gaps, .conflicts'`;
  $("#c5").textContent = `curl -s ${base}/api/datasheet -o PSV-101_datasheet.pdf -F template=psv \\\n  -F files=@relief_load_summary.pdf -F files=@psv_sizing.pdf -F 'overrides={"valve_type":"conventional"}'`;
  $("#c2").textContent = `curl -s ${base}/api/fill -o U-DR-1.pdf -F template=asme_u-dr-1 \\\n  -F files=@owner_specification.pdf -F files=@process_datasheet.pdf -F files=@mechanical_datasheet.pdf \\\n  -F form=@bpvc_viii_1_u-dr-1-2025.pdf -F 'overrides={"cyclic_service":"no"}'`;
  $("#c3").textContent = `from pinneapple_data import formfill as ff\n\nprint([s["id"] for s in ff.list_specs()])     # asme_u-dr-1, psv, shell_tube, tank, pump\nspec = ff.get_spec("shell_tube")\ndocs = [ff.read_document(p) for p in ("process_ds.pdf", "mechanical_ds.pdf")]   # priority order\ncands = [c for d in docs for c in ff.extract_rules(d, spec)]\n# optional, local LLM: cands += ff.extract_with_llm(doc, spec, ff.OllamaClient(model="..."))[0]\ncomp = ff.compile(docs, cands, spec)\nprint(comp.summary(), [d.label for d in comp.gaps()], [d.label for d in comp.conflicts()])\nopen("E-101_datasheet.pdf", "wb").write(ff.render_datasheet(comp))\ndata = comp.record()          # values, units, SI values and sources for other forms and calculations`;
}
(async function init() {
  META = await (await call("/api/meta")).json();
  $("#maxfiles").textContent = META.max_files;
  $("#llm").disabled = !META.llm_available;
  $("#llm-note").textContent = META.llm_available ? `(Ollama, ${META.llm_model}; every value is checked against the document text)` : "(no Ollama model configured on this server)";
  $("#template").innerHTML = META.formats.map((t) => `<option value="${esc(t.id)}">${esc(t.title)}</option>`).join("");
  $("#template").value = META.default_template;
  $("#docs-intro").innerHTML = $("#docs-view").innerHTML;
  snippets(); renderFormat();
})();
