"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let META = null, DOCS = [], MODE = null, RESULT = null, FORM = null, SPECFILE = null, FILTER = "attention";
const SPEC = () => (RESULT && RESULT.spec) || META.spec;
const TEMPLATE = () => $("#template").value;
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

// ── target form ──────────────────────────────────────────────────────────────
function renderTemplate() {
  const t = TEMPLATE();
  $("#tplform").hidden = t !== "auto" && t !== "custom";
  $("#tplspec").hidden = t !== "custom";
  $("#tplhint").textContent = t === "auto"
    ? "Every field of the PDF becomes an item, named after its tooltip or field name. Fields with a readable name are found by label; turn on the local LLM for the rest."
    : t === "custom" ? "A JSON spec: download one below or from the Form tab, edit labels, patterns, options and required items, upload it here."
      : "Built-in: 129 items with the phrasings datasheets and specifications use, the nozzle schedule, required items and check boxes.";
  $("#specdl").href = t === "auto" || t === "custom" ? "/api/spec" : `/api/spec?template=${encodeURIComponent(t)}`;
  if (MODE && RESULT && RESULT.spec.id !== t && t !== "auto" && t !== "custom") { RESULT = null; MODE = null; }
  renderDocs(); renderFormIntro(); renderForm();
}
$("#template").addEventListener("change", renderTemplate);
$("#tplformfile").addEventListener("change", (e) => {
  FORM = e.target.files[0] || null;
  $("#tplformname").textContent = FORM ? FORM.name : "not chosen"; $("#formname").textContent = FORM ? FORM.name : "not chosen";
  renderDocs(); renderForm();
});
$("#tplspecfile").addEventListener("change", (e) => { SPECFILE = e.target.files[0] || null; $("#tplspecname").textContent = SPECFILE ? SPECFILE.name : "not chosen"; renderDocs(); });

// ── documents (order = priority) ─────────────────────────────────────────────
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => addDocs([...e.dataTransfer.files]));
$("#files").addEventListener("change", (e) => { addDocs([...e.target.files]); e.target.value = ""; });

function guessRole(name) {
  const n = name.toLowerCase();
  if (/spec|standard|requirement/.test(n)) return "client specification";
  if (/process|pds/.test(n)) return "process datasheet";
  if (/mech|mds|vessel/.test(n)) return "mechanical datasheet";
  return "other";
}
function addDocs(list) {
  for (const f of list) if (!DOCS.some((d) => d.file.name === f.name)) DOCS.push({ file: f, role: guessRole(f.name) });
  DOCS = DOCS.slice(0, META ? META.max_files : 8);
  renderDocs();
}
function renderDocs() {
  const roles = META ? META.roles : [];
  $("#filelist").innerHTML = DOCS.map((d, i) => `
    <div class="docrow"><span class="prio">${i + 1}</span>
      <span class="dn"><b>${esc(d.file.name)}</b><select data-i="${i}">${roles.map((r) => `<option ${r === d.role ? "selected" : ""}>${r}</option>`).join("")}</select></span>
      <span class="ops"><button data-up="${i}" title="Higher priority" ${i ? "" : "disabled"}>↑</button><button data-x="${i}" title="Remove">×</button></span></div>`).join("");
  $$("#filelist select").forEach((s) => s.addEventListener("change", () => { DOCS[+s.dataset.i].role = s.value; }));
  $$("#filelist [data-up]").forEach((b) => b.addEventListener("click", () => { const i = +b.dataset.up; [DOCS[i - 1], DOCS[i]] = [DOCS[i], DOCS[i - 1]]; renderDocs(); }));
  $$("#filelist [data-x]").forEach((b) => b.addEventListener("click", () => { DOCS.splice(+b.dataset.x, 1); renderDocs(); }));
  const t = META ? TEMPLATE() : "";
  $("#compile").disabled = !DOCS.length || (t === "auto" && !FORM) || (t === "custom" && !SPECFILE);
}

function formData(withForm) {
  const fd = new FormData();
  DOCS.forEach((d) => fd.append("files", d.file));
  fd.append("roles", JSON.stringify(DOCS.map((d) => d.role)));
  fd.append("use_llm", $("#llm").checked ? "true" : "false");
  fd.append("overrides", JSON.stringify(OVERRIDES));
  fd.append("template", TEMPLATE());
  if (SPECFILE && TEMPLATE() === "custom") fd.append("spec", SPECFILE);
  if (FORM && (withForm || TEMPLATE() === "auto")) fd.append("form", FORM);
  return fd;
}

async function compile(mode) {
  const btn = mode === "example" ? $("#example") : $("#compile");
  const done = busy(btn, $("#llm").checked && mode !== "example" ? "Reading (local LLM: may take minutes)…" : "Reading documents…");
  $("#status").textContent = "";
  try {
    if (mode !== MODE) for (const k of Object.keys(OVERRIDES)) delete OVERRIDES[k];
    MODE = mode;
    if (mode === "example" && TEMPLATE() !== META.default_template) { $("#template").value = META.default_template; renderTemplate(); }
    const r = mode === "example" ? await call("/api/example/compile") : await call("/api/compile", { method: "POST", body: formData(false) });
    RESULT = await r.json();
    renderDocsView(); renderReview(); renderFormIntro(); renderForm();
    showTab("review");
  } catch (e) {
    $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`;
  } finally { done(); }
}
$("#compile").addEventListener("click", () => compile("upload"));
$("#example").addEventListener("click", () => compile("example"));

// Re-run with the engineer's answers (overrides) without re-uploading for the example; uploads are re-sent.
async function recompile() {
  if (!MODE) return;
  try {
    const r = MODE === "example"
      ? await call("/api/example/compile")
      : await call("/api/compile", { method: "POST", body: formData(false) });
    RESULT = await r.json();
    applyOverridesLocally();
    renderReview(); renderForm();
  } catch (e) { $("#review-status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}
function applyOverridesLocally() {
  for (const [k, v] of Object.entries(OVERRIDES)) {
    const d = RESULT.decisions[k];
    if (!d) continue;
    d.value = v; d.display = String(v); d.status = "filled"; d.note = "entered by engineer";
  }
  RESULT.gaps = RESULT.gaps.filter((k) => !(k in OVERRIDES));
  RESULT.conflicts = RESULT.conflicts.filter((k) => !(k in OVERRIDES));
}

// ── documents panel ──────────────────────────────────────────────────────────
function renderDocsView() {
  const docs = RESULT.documents.map((d) => `<tr><td class="num">${d.priority}</td><td><b>${esc(d.name)}</b></td><td>${esc(d.role || "—")}</td>
    <td class="num">${d.pages}</td><td>${d.has_text ? '<span class="badge pass">text layer</span>' : '<span class="badge fail">scanned: no text</span>'}</td></tr>`).join("");
  $("#docs-view").innerHTML = `<h2 style="margin-top:0">Documents read</h2>
    <table><thead><tr><th class="num">Priority</th><th>Document</th><th>Role</th><th class="num">Pages</th><th>Text</th></tr></thead><tbody>${docs}</tbody></table>
    <p class="hint" style="margin-top:8px">Form: <b>${esc(RESULT.spec.title)}</b> · ${RESULT.spec.fields.length} items</p>
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
  const banner = !s.required ? `<div class="banner warn">This form marks no item as required: ${s.filled} of ${s.fields} items found. Check the gaps that matter to you in "All items".</div>` : s.gaps ? `<div class="banner warn">${s.required_filled} of ${s.required} required items found. ${s.gaps} missing and ${s.conflicts} in conflict: resolve them before issuing the form.</div>`
    : s.conflicts ? `<div class="banner warn">All ${s.required} required items found; ${s.conflicts} conflict(s) between documents to confirm.</div>`
      : `<div class="banner ok">All ${s.required} required items found, no conflicts between documents.</div>`;
  $("#review").innerHTML = `
    <div class="toolbar"><h2 style="margin:0">Review</h2>
      <div class="seg small" id="filter">${[["attention", "Needs attention"], ["found", "Found"], ["all", `All ${spec.fields.length} items`]].map(([k, l]) => `<button data-f="${k}" class="${FILTER === k ? "on" : ""}">${l}</button>`).join("")}</div></div>
    ${banner}
    <div class="kpis">
      <div class="kpi"><div class="l">Items found</div><div class="v">${s.filled}</div><div class="s">of ${s.fields} on the form</div></div>
      <div class="kpi"><div class="l">Required found</div><div class="v">${s.required_filled}/${s.required}</div><div class="s">needed to start the design</div></div>
      <div class="kpi"><div class="l">Conflicts</div><div class="v">${s.conflicts}</div><div class="s">documents disagree</div></div>
      <div class="kpi"><div class="l">Gaps</div><div class="v">${s.gaps}</div><div class="s">required, stated nowhere</div></div>
      ${spec.tables.length ? `<div class="kpi"><div class="l">${esc(spec.tables[0].label)}</div><div class="v">${s.table_rows}</div><div class="s">rows</div></div>` : ""}
    </div>
    ${gaps.length ? `<div class="gaps"><b>Ask for these before issuing the form:</b><ul>${gaps.map((f) => `<li>${esc(f.label)}${f.hint ? ` <span class="docp">usually in the ${esc(f.hint)}</span>` : ""}</li>`).join("")}</ul></div>` : ""}
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

// ── form ─────────────────────────────────────────────────────────────────────
$("#formfile").addEventListener("change", (e) => {
  FORM = e.target.files[0] || null; $("#formname").textContent = FORM ? FORM.name : "not chosen";
  $("#tplformname").textContent = FORM ? FORM.name : "not chosen"; renderDocs(); renderForm();
});
function renderFormIntro() {
  const spec = SPEC(), udr = spec.id === META.default_template;
  $("#formtitle").textContent = `Fill: ${spec.title}`;
  $("#formintro").innerHTML = udr
    ? `The form is ASME's document, so it is not bundled: upload your copy of the fillable <b>Form U-DR-1 (User's Design Requirements for Single-Chamber Pressure Vessels)</b>, 07/25 revision, from asme.org. ${META.form_on_server ? "A copy is already configured on this server; uploading one is optional." : ""}`
    : "Upload the blank fillable PDF this spec describes (for “Any fillable PDF”, the one chosen on the Documents tab is used).";
}
function renderForm() {
  const udr = SPEC().id === META.default_template;
  const ready = RESULT && (FORM || (udr && META.form_on_server));
  $("#fill").disabled = !ready;
  $("#checklist").disabled = !RESULT;
  $("#record").disabled = !RESULT;
  if (!RESULT) return;
  const s = RESULT.summary, fp = RESULT.form_preview;
  $("#formsummary").innerHTML = `<div class="kpis">
      <div class="kpi"><div class="l">Text fields to fill</div><div class="v">${fp.text_fields}</div></div>
      <div class="kpi"><div class="l">Check boxes to tick</div><div class="v">${fp.check_boxes}</div></div>
      <div class="kpi"><div class="l">Open items</div><div class="v">${s.gaps + s.conflicts}</div><div class="s">gaps + conflicts</div></div></div>
    <p class="hint">${(SPEC().left_blank || []).length ? `${SPEC().left_blank.join(", ")} and the signature are left blank on purpose. ` : ""}${SPEC().notes_field ? "Values too long for their box are written in the notes field." : "Long values are written in a smaller font."}</p>`;
}
$("#fill").addEventListener("click", async () => {
  const done = busy($("#fill"), "Filling…");
  $("#fillstatus").textContent = "";
  try {
    let r;
    if (MODE === "example") {
      const fd = new FormData(); if (FORM) fd.append("form", FORM); fd.append("overrides", JSON.stringify(OVERRIDES));
      r = await call("/api/example/fill", { method: "POST", body: fd });
    } else r = await call("/api/fill", { method: "POST", body: formData(true) });
    const blob = await r.blob();
    const name = (r.headers.get("content-disposition") || "").match(/filename="([^"]+)"/)?.[1] || "filled_form.pdf";
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click();
    $("#fillstatus").innerHTML = `<span class="ok">Downloaded ${esc(name)}. Review it, then date and sign.</span>`;
  } catch (e) { $("#fillstatus").innerHTML = `<span class="err">${esc(e.message)}</span>`; } finally { done(); }
});
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
  download(new Blob(["﻿" + lines.join("\n")], { type: "text/csv" }), `${spec.id}_data_checklist.csv`);
});
$("#record").addEventListener("click", () => {
  const rec = { form: SPEC().id, documents: RESULT.documents, data: RESULT.record };
  download(new Blob([JSON.stringify(rec, null, 1)], { type: "application/json" }), `${SPEC().id}_data_record.json`);
});
function download(blob, name) { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click(); }

// ── API snippets + init ──────────────────────────────────────────────────────
function snippets() {
  const base = location.origin;
  $("#c1").textContent = `curl -s ${base}/api/compile \\\n  -F files=@owner_specification.pdf -F files=@process_datasheet.pdf -F files=@mechanical_datasheet.pdf \\\n  -F 'roles=["client specification","process datasheet","mechanical datasheet"]' | jq '.summary, .gaps, .conflicts'`;
  $("#c2").textContent = `curl -s ${base}/api/fill -o U-DR-1.pdf \\\n  -F files=@owner_specification.pdf -F files=@process_datasheet.pdf -F files=@mechanical_datasheet.pdf \\\n  -F form=@bpvc_viii_1_u-dr-1-2025.pdf -F 'overrides={"cyclic_service":"no"}'`;
  $("#c4").textContent = `# fields of any fillable PDF become the items; add -F use_llm=true to read with the local LLM\ncurl -s ${base}/api/fill -o filled.pdf -F template=auto -F form=@blank_form.pdf -F files=@datasheet.pdf\n# or start a custom spec from that PDF, edit it, and use it with -F template=custom -F spec=@my_form.spec.json\ncurl -s ${base}/api/spec/from-pdf -F form=@blank_form.pdf -o my_form.spec.json`;
  $("#c3").textContent = `from pinneapple_data import formfill as ff\n\nspec = ff.get_spec("asme_u-dr-1")        # or ff.FormSpec.from_pdf(blank_bytes) / ff.FormSpec.load("my.spec.json")\ndocs = [ff.read_document(p) for p in ("spec.pdf", "process.pdf", "mech.pdf")]   # priority order\ncands = [c for d in docs for c in ff.extract_rules(d, spec)]\n# optional, local LLM: cands += ff.extract_with_llm(doc, spec, ff.OllamaClient(model="..."))[0]\ncomp = ff.compile(docs, cands, spec)\nprint(comp.summary(), [d.label for d in comp.gaps()], [d.label for d in comp.conflicts()])\nopen("filled.pdf", "wb").write(ff.fill_pdf(open("blank.pdf", "rb").read(), comp))\ndata = comp.record()                       # values, units, SI values and sources for other forms and calculations`;
}
(async function init() {
  META = await (await call("/api/meta")).json();
  $("#maxfiles").textContent = META.max_files;
  $("#exlinks").innerHTML = META.examples.map((n) => `<a class="link" href="/api/example/files/${n}">${n}</a>`).join(" · ");
  $("#llm").disabled = !META.llm_available;
  $("#llm-note").textContent = META.llm_available ? `(Ollama, ${META.llm_model}; every value is checked against the document text)` : "(no Ollama model configured on this server)";
  $("#template").innerHTML = META.templates.map((t) => `<option value="${esc(t.id)}">${esc(t.title)}</option>`).join("");
  $("#template").value = META.default_template;
  snippets(); renderTemplate();
})();
