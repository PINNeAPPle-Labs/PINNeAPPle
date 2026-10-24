// "Create your own": compose a tool from existing blocks (Flow Studio node types), optionally starting from a
// bundled flow and loading a library geometry. The server composes (pinneapple_hub.library.compose): it wires the
// blocks by port type, checks every parameter and returns the flow JSON and the Python; validation runs that
// Python in the sandbox as a job. URL: /create?blocks=type1,type2&name=...&start=<flow>&geometry=<name>, or
// /create?flow=<base64url flow JSON> (Flow Studio's "Publish to Hub"): the whole flow is the start, and the parameters
// ticked under "Parameters of this tool" become variables of the generated Python. /create?mode=canvas embeds the
// Flow Studio canvas instead of the block list: it posts its flow here on every change (window.postMessage, origin
// checked both ways), and the code, validation and My Hub work on that flow.
import { api, codeBlock, copyText, emptyState, esc, fileUrl, go, itemUrl, num, state, toast } from "./app.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
let catalog = null;

const S = { mode: "blocks", name: "My tool", description: "", start: "", geometry: "", blocks: [], flow: null, expose: [], out: null, tab: "py", job: null, run: null };

function body() {
  return { name: S.name || "My tool", description: S.description, start: S.flow ? null : S.start || null, geometry: S.geometry || null,
           blocks: S.blocks.map((b) => ({ type: b.type, params: b.params })), flow: S.flow, expose: S.expose };
}
export const b64 = (o) => btoa(unescape(encodeURIComponent(JSON.stringify(o)))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const unb64 = (t) => JSON.parse(decodeURIComponent(escape(atob(t.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (t.length % 4)) % 4)))));

export async function createView(view, sp) {
  catalog ??= await api("/api/compose/blocks");
  const types = new Map(catalog.blocks.map((b) => [b.type, b]));
  if (sp.get("mode") === "canvas" || sp.get("mode") === "blocks") S.mode = sp.get("mode");
  if (sp.get("flow")) {
    try {
      S.flow = unb64(sp.get("flow")); S.blocks = []; S.expose = []; S.start = ""; S.run = null; S.job = null;
      S.name = S.flow.name || S.name; S.description = S.flow.description || "";
      toast("Flow from Flow Studio loaded: tick the parameters to expose, validate, save");
    } catch { toast("Could not read the flow in the link"); }
    history.replaceState({}, "", "/create");
  } else if (sp.get("blocks") || sp.get("start") || sp.get("name")) {
    S.blocks = (sp.get("blocks") || "").split(",").filter((t) => types.has(t)).map((t) => ({ type: t, params: sp.get("params") ? unb64(sp.get("params")) : {} }));
    S.name = sp.get("name") || S.name; S.start = sp.get("start") || ""; S.geometry = sp.get("geometry") || "";
    S.flow = null; S.expose = []; S.run = null; S.job = null;
    history.replaceState({}, "", "/create");
  }
  const sections = [];
  for (const b of catalog.blocks) {
    const id = b.auto ? `auto:${b.group}` : b.section;
    let s = sections.find((x) => x.id === id);
    if (!s) sections.push((s = { id, title: b.auto ? b.group : b.section_title, auto: b.auto, items: [] }));
    s.items.push(b);
  }
  const btn = (b) => `<button type="button" data-add="${esc(b.type)}" ${b.available ? "" : "disabled"} title="${esc(b.install || b.summary)}" data-text="${esc((b.title + " " + b.type + " " + b.summary).toLowerCase())}">${esc(b.title)}<small>${esc(b.available ? b.summary : b.install || "not installed here")}</small></button>`;
  const secHTML = (s) => `<div data-sec><h4>${esc(s.title)}</h4>${s.items.map(btn).join("")}</div>`;
  const nAuto = catalog.blocks.filter((b) => b.auto).length;
  const fsBase = (state.config.flow_studio || "").replace(/\/$/, "");
  const canvas = S.mode === "canvas";
  const blocksHTML = `<aside class="panel"><h2>1 · Blocks</h2>
        <label>Name<input id="c-name" maxlength="80" value="${esc(S.name)}"></label>
        <label>Description<input id="c-desc" maxlength="500" value="${esc(S.description)}" placeholder="what the tool does"></label>
        <label>Start from a flow <select id="c-start"><option value="">— empty —</option>${catalog.flows.map((f) => `<option value="${esc(f.name)}" ${S.start === f.name ? "selected" : ""}>${esc(f.title)}</option>`).join("")}</select></label>
        <label>Library geometry <select id="c-geo"><option value="">— none —</option>${catalog.geometries.map((g) => `<option value="${esc(g.name)}" ${S.geometry === g.name ? "selected" : ""}>${esc(g.title)}</option>`).join("")}</select></label>
        <label>Add a block<input id="c-filter" type="search" placeholder="filter ${catalog.blocks.length} blocks…" aria-controls="c-blocks"></label>
        <div class="blocklist" id="c-blocks">${sections.filter((s) => !s.auto).map(secHTML).join("")}
          <details id="c-auto"><summary>Library calls <span class="badge">${nAuto}</span></summary>${sections.filter((s) => s.auto).map(secHTML).join("")}</details></div>
      </aside>
      <section class="panel" aria-label="Your tool"><h2>2 · Your tool <span class="badge" id="c-count"></span></h2>
        <div class="chain" id="c-chain"></div>
        <ul class="issues" id="c-issues" aria-live="polite"></ul>
      </section>`;
  const canvasHTML = `<section class="panel canvaspanel" aria-label="Flow Studio canvas"><h2>1 · Build it on the Flow Studio canvas <span class="badge" id="c-count"></span></h2>
        <div class="row2"><label>Name<input id="c-name" maxlength="80" value="${esc(S.name)}"></label>
          <label>Description<input id="c-desc" maxlength="500" value="${esc(S.description)}" placeholder="what the tool does"></label></div>
        <div class="fsframe"><iframe id="c-frame" title="Flow Studio canvas" src="${esc(fsBase)}/?embed=hub&amp;origin=${encodeURIComponent(location.origin)}"></iframe></div>
        <p class="hint" id="c-frame-note">Drag nodes from the palette, connect them, set their parameters: the flow comes here as you edit it.</p>
        <div class="chain" id="c-chain"></div>
        <ul class="issues" id="c-issues" aria-live="polite"></ul>
      </section>`;
  view.innerHTML = `<main><section class="block"><h2>Create your own tool <small>compose existing blocks or build it on the Flow Studio canvas; get the Python and the flow JSON, validated in a sandbox</small></h2>
    <div class="seg2" role="tablist" aria-label="How to build it" style="margin-bottom:12px"><button type="button" role="tab" data-mode="blocks" aria-selected="${!canvas}" class="${canvas ? "" : "on"}">Blocks</button><button type="button" role="tab" data-mode="canvas" aria-selected="${canvas}" class="${canvas ? "on" : ""}">Flow Studio canvas</button></div>
    <div class="create${canvas ? " canvas-mode" : ""}">
      ${canvas ? canvasHTML : blocksHTML}
      <section class="panel out" aria-label="Generated code"><div class="outtabs"><h2 style="margin:0">${canvas ? 2 : 3} · Code</h2>
        <div class="seg2" role="tablist"><button type="button" role="tab" data-tab="py">Python</button><button type="button" role="tab" data-tab="json">Flow JSON</button></div></div>
        <div id="c-code"></div>
        <div class="row" style="margin-top:10px">
          <button class="btn primary" id="c-validate" type="button">Validate in a sandbox</button>
          <a class="btn" id="c-dl-py" href="#">Download .py</a><a class="btn" id="c-dl-json" href="#">Download flow JSON</a>
          <a class="btn" id="c-fs" href="#" target="_blank" rel="noopener">Open in Flow Studio</a>
          <button class="btn" id="c-save" type="button">Save to My Hub</button></div>
        <div id="c-run" aria-live="polite"></div>
      </section></div></section></main>`;
  $$("[data-mode]").forEach((b) => (b.onclick = () => { if (b.dataset.mode !== S.mode) { S.mode = b.dataset.mode; S.run = null; createView(view, new URLSearchParams()); } }));
  $("#c-name").oninput = (e) => { S.name = e.target.value; refresh(); };
  $("#c-desc").oninput = (e) => { S.description = e.target.value; refresh(); };
  $$("[data-tab]").forEach((b) => (b.onclick = () => { S.tab = b.dataset.tab; renderCode(); }));
  $("#c-validate").onclick = validate;
  $("#c-save").onclick = save;
  if (canvas) { wireCanvas(types, fsBase); renderChain(types); refresh(true); return; }
  $("#c-start").onchange = (e) => { S.start = e.target.value; S.flow = null; S.expose = []; renderChain(types); refresh(true); };
  $("#c-geo").onchange = (e) => { S.geometry = e.target.value; refresh(); };
  $("#c-filter").oninput = (e) => {
    const q = e.target.value.trim().toLowerCase();
    $$("[data-add]").forEach((b) => (b.hidden = q && !b.dataset.text.includes(q)));
    $$("[data-sec]").forEach((s) => (s.hidden = !$$("[data-add]", s).some((b) => !b.hidden)));
    $("#c-auto").open = !!q && $$("#c-auto [data-add]").some((b) => !b.hidden);
  };
  $$("[data-add]").forEach((b) => (b.onclick = () => { S.blocks.push({ type: b.dataset.add, params: {} }); renderChain(types); refresh(true); toast(`Added ${types.get(b.dataset.add).title}`); }));
  renderChain(types);
  refresh(true);
}

let onCanvasMessage = null;
/** The embedded Flow Studio: accept messages only from its origin and its frame; send it the current flow when it
 *  is ready; take every flow it posts as this tool's flow (blocks are not mixed with a canvas flow). */
function wireCanvas(types, fsBase) {
  const frame = $("#c-frame"), fsOrigin = new URL(fsBase, location.href).origin;
  let ready = false;
  if (onCanvasMessage) window.removeEventListener("message", onCanvasMessage);
  onCanvasMessage = (e) => {
    if (e.origin !== fsOrigin || e.source !== frame?.contentWindow) return;
    if (e.data?.type === "pinneapple-flow/ready") {
      ready = true;
      $("#c-frame-note").textContent = "Drag nodes from the palette, connect them, set their parameters: the flow comes here as you edit it.";
      if (S.flow?.nodes?.length) frame.contentWindow.postMessage({ type: "pinneapple-flow/load", flow: S.flow }, fsOrigin);
    } else if (e.data?.type === "pinneapple-flow/changed" && Array.isArray(e.data.flow?.nodes)) {
      const ids = new Set(e.data.flow.nodes.map((n) => n.id));
      S.flow = e.data.flow.nodes.length ? e.data.flow : null; S.blocks = []; S.start = "";
      S.expose = S.expose.filter((x) => ids.has(x.node)); S.run = null;
      renderChain(types); refresh();
    }
  };
  window.addEventListener("message", onCanvasMessage);
  setTimeout(() => {
    if (!ready && $("#c-frame-note")) $("#c-frame-note").innerHTML = `The canvas did not answer. Is Flow Studio running at <code>${esc(fsBase)}</code>, and does it allow this Hub (<code>FLOW_FRAME_ANCESTORS=${esc(location.origin)}</code>)? Or <a href="${esc(fsBase)}" target="_blank" rel="noopener">open it in a tab</a> and use “Publish to Hub”.`;
  }, 10000);
}

export function field(type, key, schema, value) {
  const id = `p-${type}-${key}-${Math.random().toString(36).slice(2, 6)}`;
  const lab = `${esc(schema.title || key)}${schema.minimum != null || schema.maximum != null ? ` <span class="u">${schema.minimum ?? ""}…${schema.maximum ?? ""}</span>` : ""}`;
  const tip = esc(schema.description || "");
  if (schema.enum) return `<label for="${id}" title="${tip}">${lab}<select id="${id}" data-k="${esc(key)}" data-t="enum">${schema.enum.map((c) => `<option value='${esc(JSON.stringify(c))}' ${JSON.stringify(c) === JSON.stringify(value) ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></label>`;
  if (schema.type === "boolean") return `<label for="${id}" title="${tip}" style="display:flex;gap:6px;align-items:center;margin-top:22px"><input id="${id}" style="width:auto;margin:0" type="checkbox" data-k="${esc(key)}" data-t="bool" ${value ? "checked" : ""}>${lab}</label>`;
  if (schema.type === "integer" || schema.type === "number") return `<label for="${id}" title="${tip}">${lab}<input id="${id}" type="number" step="${schema.type === "integer" ? 1 : "any"}" data-k="${esc(key)}" data-t="${schema.type}" value="${esc(value ?? "")}"></label>`;
  if (schema.type === "array" || schema.type === "object") return `<label for="${id}" class="wide" title="${tip}">${lab} <span class="u">JSON</span><textarea id="${id}" rows="1" data-k="${esc(key)}" data-t="json">${esc(JSON.stringify(value ?? (schema.type === "array" ? [] : {})))}</textarea></label>`;
  return `<label for="${id}" class="${(schema["x-kind"] === "text" || schema["x-kind"] === "expression") ? "wide" : ""}" title="${tip}">${lab}<input id="${id}" data-k="${esc(key)}" data-t="str" value="${esc(value ?? "")}"></label>`;
}

/** The value of a parameter input made by field(): {ok, value}; marks the input invalid when it does not parse. */
export function readInput(inp) {
  const k = inp.dataset.k, t = inp.dataset.t;
  try {
    const v = t === "bool" ? inp.checked : t === "integer" ? parseInt(inp.value, 10) : t === "number" ? parseFloat(inp.value)
      : t === "enum" ? JSON.parse(inp.value) : t === "json" ? JSON.parse(inp.value || "null") : inp.value;
    if (typeof v === "number" && !Number.isFinite(v)) throw new Error("not a number");
    inp.removeAttribute("aria-invalid");
    return { ok: true, value: v };
  } catch {
    inp.setAttribute("aria-invalid", "true"); toast(`${k}: not valid ${t === "json" ? "JSON" : "input"}`);
    return { ok: false };
  }
}

/** The flow published from Flow Studio: its nodes, and a tick box per parameter to expose. */
function flowCard(types) {
  const f = S.flow, on = (n, p) => S.expose.some((e) => e.node === n && e.param === p);
  return `<article class="blk"><header><span class="num">⇢</span><b>${S.mode === "canvas" ? "On the canvas" : "From Flow Studio"}: ${esc(f.name || "flow")}</b>${S.mode === "canvas" ? "" : '<button type="button" data-drop aria-label="Drop the Flow Studio flow">✕</button>'}</header>
    <div class="ports">${f.nodes.length} nodes, ${f.edges.length} connections · tick the parameters that this tool takes</div>
    <div class="params" style="display:block">${f.nodes.map((n) => {
      const t = types.get(n.type), props = t?.params_schema?.properties || {};
      return `<details ${S.expose.some((e) => e.node === n.id) ? "open" : ""}><summary>${esc(n.label || t?.title || n.type)} <code>${esc(n.id)}</code></summary>${Object.entries(props).map(([k, sch]) => `<label style="display:flex;gap:6px;align-items:center"><input type="checkbox" style="width:auto;margin:0" data-expose="${esc(n.id)}|${esc(k)}" ${on(n.id, k) ? "checked" : ""}> ${esc(sch.title || k)} <span class="u">= ${esc(JSON.stringify((n.params || {})[k] ?? sch.default ?? null))}</span></label>`).join("") || '<span class="hint">no parameters</span>'}</details>`;
    }).join("")}</div></article>`;
}

function renderChain(types) {
  const host = $("#c-chain");
  if (S.mode === "canvas" && !S.flow) {
    $("#c-count").textContent = "empty";
    host.innerHTML = '<p class="hint">The canvas is empty: add nodes there. Its parameters appear here to tick the ones this tool takes.</p>';
    return;
  }
  if (S.flow) {
    host.innerHTML = flowCard(types) + (S.blocks.length || S.mode === "canvas" ? "" : '<p class="hint">Blocks you add are appended to this flow.</p>');
    if ($("[data-drop]", host)) $("[data-drop]", host).onclick = () => { S.flow = null; S.expose = []; renderChain(types); refresh(true); };
    $$("[data-expose]", host).forEach((cb) => (cb.onchange = () => {
      const [node, param] = cb.dataset.expose.split("|");
      S.expose = S.expose.filter((e) => !(e.node === node && e.param === param));
      if (cb.checked) S.expose.push({ node, param });
      refresh();
    }));
    if (!S.blocks.length) { $("#c-count").textContent = "flow"; return; }
    host.insertAdjacentHTML("beforeend", '<div id="c-chain-blocks" class="chain"></div>');
  }
  $("#c-count").textContent = `${S.blocks.length} block${S.blocks.length === 1 ? "" : "s"}${S.start || S.flow ? " + flow" : ""}`;
  const target = S.flow ? $("#c-chain-blocks", host) : host;
  if (!S.blocks.length) {
    host.innerHTML = emptyState(S.start ? "Starting from a flow" : "No blocks yet", S.start ? "Blocks you add are appended to the flow and wired by port type." : "Pick blocks on the left: each new block's inputs are wired from the previous ones by port type. Try Domain → Mesh / grid, or start from a flow.");
    return;
  }
  target.innerHTML = S.blocks.map((b, i) => {
    const t = types.get(b.type), props = t.params_schema.properties || {};
    const basic = Object.entries(props).filter(([, s]) => !s["x-advanced"]), adv = Object.entries(props).filter(([, s]) => s["x-advanced"]);
    const val = (k) => (k in b.params ? b.params[k] : t.defaults[k]);
    return `<article class="blk" data-i="${i}"><header><span class="num">${i + 1}</span><b>${esc(t.title)}</b>
      <button type="button" data-up aria-label="Move up" ${i ? "" : "disabled"}>↑</button><button type="button" data-down aria-label="Move down" ${i < S.blocks.length - 1 ? "" : "disabled"}>↓</button><button type="button" data-rm aria-label="Remove ${esc(t.title)}">✕</button></header>
      <div class="ports">${esc(t.type)} · in: ${t.inputs.map((p) => esc(`${p.name}:${p.type}`)).join(", ") || "—"} · out: ${t.outputs.map((p) => esc(`${p.name}:${p.type}`)).join(", ") || "—"}</div>
      <div class="params">${basic.map(([k, s]) => field(b.type, k, s, val(k))).join("")}</div>
      ${adv.length ? `<details><summary>${adv.length} advanced parameter${adv.length > 1 ? "s" : ""}</summary><div class="params">${adv.map(([k, s]) => field(b.type, k, s, val(k))).join("")}</div></details>` : ""}</article>`;
  }).join("");
  $$(".blk[data-i]", host).forEach((el) => {
    const i = +el.dataset.i;
    el.querySelector("[data-rm]").onclick = () => { S.blocks.splice(i, 1); renderChain(types); refresh(true); };
    el.querySelector("[data-up]").onclick = () => { [S.blocks[i - 1], S.blocks[i]] = [S.blocks[i], S.blocks[i - 1]]; renderChain(types); refresh(true); };
    el.querySelector("[data-down]").onclick = () => { [S.blocks[i + 1], S.blocks[i]] = [S.blocks[i], S.blocks[i + 1]]; renderChain(types); refresh(true); };
    $$("[data-k]", el).forEach((inp) => (inp.onchange = () => {
      const r = readInput(inp);
      if (!r.ok) return;
      S.blocks[i].params[inp.dataset.k] = r.value;
      refresh();
    }));
  });
}

let refreshT = 0;
function refresh(now = false) {
  clearTimeout(refreshT);
  refreshT = setTimeout(async () => {
    try { S.out = await api("/api/compose", { method: "POST", body: JSON.stringify(body()) }); }
    catch (e) { S.out = { error: e.message, issues: [{ level: "error", message: e.message }] }; }
    S.run = null; renderCode(); renderIssues();
  }, now ? 0 : 280);
}

function renderIssues() {
  const ul = $("#c-issues"), is = S.out?.issues || [];
  ul.innerHTML = is.length ? is.map((i) => `<li class="${esc(i.level)}">${esc(i.level)}: ${esc(i.message)}</li>`).join("")
    : S.blocks.length || S.start || S.flow ? '<li class="hint" style="padding:5px 0">✓ the flow validates: every block is a registered node and every required input is wired</li>' : "";
}

function renderCode() {
  $$("[data-tab]").forEach((b) => { b.classList.toggle("on", b.dataset.tab === S.tab); b.setAttribute("aria-selected", b.dataset.tab === S.tab); });
  const o = S.out;
  const empty = !o || o.error || (!S.blocks.length && !S.start && !S.flow);
  $("#c-code").innerHTML = !o ? '<p class="hint">composing…</p>' : o.error ? emptyState("Cannot compose", esc(o.error), true)
    : S.tab === "py" ? codeBlock(o.code, { label: "Copy code" }) : codeBlock(JSON.stringify(o.flow, null, 1), { lang: "json", label: "Copy JSON", what: "Flow JSON" });
  const spec = b64(body());
  $("#c-dl-py").href = `/api/compose/download/py?spec=${spec}`;
  $("#c-dl-json").href = `/api/compose/download/json?spec=${spec}`;
  const fsBase = (state.config.flow_studio || "").replace(/\/$/, "");
  $("#c-fs").href = o?.flow ? `${fsBase}/?flow=${b64(o.flow)}` : fsBase;
  for (const id of ["c-validate", "c-save", "c-dl-py", "c-dl-json", "c-fs"]) {
    const el = $(`#${id}`); const off = empty || (id !== "c-dl-json" && id !== "c-fs" && id !== "c-dl-py" && !o.ok);
    el.toggleAttribute("disabled", !!off); el.setAttribute("aria-disabled", String(!!off)); if (el.tagName === "A") el.style.pointerEvents = off ? "none" : "";
  }
  renderRun();
}

function renderRun() {
  const host = $("#c-run"), r = S.run;
  if (!r) { host.innerHTML = ""; return; }
  if (r.status && r.status !== "done") {
    host.innerHTML = `<div class="runbox" style="margin-top:12px"><div class="runhead">${esc(r.status)} · ${esc(r.message || "")}</div><div class="progress" style="margin:8px 12px 12px"><div style="width:${Math.round((r.progress || 0) * 100)}%"></div></div></div>`;
    return;
  }
  const x = r.result || {};
  host.innerHTML = `<div class="runbox" style="margin-top:12px"><div class="runhead">${x.ok ? '<span class="badge ok">ran without errors</span>' : `<span class="badge bad">${x.timed_out ? "time limit reached" : "failed"}</span>`}<span>${num(x.seconds)} s in the sandbox</span></div>
    ${x.stdout ? `<pre>${esc(x.stdout)}</pre>` : ""}${x.stderr && !x.ok ? `<pre style="color:var(--bad)">${esc(x.stderr)}</pre>` : ""}
    ${(x.figures_png_b64 || []).length ? `<div class="figs" style="padding:0 10px 10px">${x.figures_png_b64.map((f) => `<figure><img src="data:image/png;base64,${f}" alt="figure from the validation run" data-zoom></figure>`).join("")}</div>` : ""}</div>`;
}

async function validate() {
  const btn = $("#c-validate"); btn.disabled = true;
  try {
    const job = await api("/api/compose/validate", { method: "POST", body: JSON.stringify(body()) });
    S.job = job.id; S.run = job; renderRun();
    for (;;) {
      await new Promise((r) => setTimeout(r, 700));
      const j = await api(`/api/jobs/${job.id}`);
      S.run = j; renderRun();
      if (["done", "failed", "cancelled"].includes(j.status)) {
        if (j.status !== "done") { $("#c-run").innerHTML = emptyState("Validation did not finish", esc(j.error || j.status), true); }
        else toast(j.result?.ok ? "Validated: the code ran" : "The code failed in the sandbox");
        break;
      }
    }
  } catch (e) {
    $("#c-run").innerHTML = emptyState(e.status === 429 ? "Limit reached" : "Validation failed to start", esc(e.message), true);
  } finally { btn.disabled = false; }
}

async function save() {
  try {
    const it = await api("/api/mine", { method: "POST", body: JSON.stringify({ ...body(), job_id: S.run?.status === "done" ? S.job : null }) });
    toast("Saved to My Hub");
    go(itemUrl(it.id));
  } catch (e) { toast(`Not saved: ${e.message}`); }
}
