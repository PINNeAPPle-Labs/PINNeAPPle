// PINNeAPPle Hub UI. Everything shown comes from the API (apps/hub/hub/api.py), which reads the generated index
// of pinneapple_hub.library; this file only renders it. Routes: /  (home, or results with ?q= & filters),
// /item/<id>, /create (create.js), /mine.
import { b64, createView, field, readInput } from "./create.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const num = (v) => (typeof v !== "number" || !Number.isFinite(v) ? esc(v) : Number.isInteger(v) ? String(v) : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(3) : String(+v.toPrecision(5)));
const view = $("#view");
const KIND_LABEL = { tool: "Tool", model: "Trained model", geometry: "Geometry", dataset: "Dataset", flow: "Flow", example: "Example" };
const GLYPH = { tool: "f(x)", model: "NN", geometry: "3-D", dataset: "data", flow: "flow", example: "run" };
const ICON = {
  copy: '<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="6" y="6" width="10" height="11" rx="2" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M4 13V4a1 1 0 0 1 1-1h8" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>',
  down: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3v10m-4-4 4 4 4-4M4 16h12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  flow: '<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="2" y="3" width="6" height="5" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.7"/><rect x="12" y="12" width="6" height="5" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M8 5.5h3a2 2 0 0 1 2 2V12" fill="none" stroke="currentColor" stroke-width="1.7"/></svg>',
  ext: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M11 3h6v6M17 3l-8 8M14 12v4a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h4" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>',
  empty: '<svg viewBox="0 0 48 48" aria-hidden="true"><circle cx="21" cy="21" r="12" fill="none" stroke="currentColor" stroke-width="3"/><path d="M30 30l10 10" stroke="currentColor" stroke-width="3" stroke-linecap="round"/><path d="M16 21h10" stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>',
  err: '<svg viewBox="0 0 48 48" aria-hidden="true"><path d="M24 6 44 40H4z" fill="none" stroke="currentColor" stroke-width="3" stroke-linejoin="round"/><path d="M24 19v10M24 34v1" stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>',
};
export const state = { config: null };

export async function api(url, opts = {}) {
  const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
  let body = null;
  try { body = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) {
    const d = body?.detail;
    const msg = typeof d === "string" ? d : d?.message || (Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `HTTP ${r.status}`);
    const e = new Error(msg); e.status = r.status; e.body = body; throw e;
  }
  return body;
}
export const fileUrl = (p) => (p.startsWith("http") ? p : `/files/${p.split("/").map(encodeURIComponent).join("/")}`);
export const itemUrl = (id) => `/item/${encodeURIComponent(id)}`;

let toastT = 0;
export function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("on");
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("on"), 1900);
}

export async function copyText(text, what = "Code") {
  try { await navigator.clipboard.writeText(text); }
  catch {
    const ta = document.createElement("textarea"); ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.append(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  toast(`${what} copied to the clipboard`);
}

// ---------------------------------------------------------------- python highlighting (small, safe: escapes first)
const KW = new Set("import from as def return for in if else elif while with try except finally class lambda not and or is None True False print raise assert yield pass break continue global nonlocal".split(" "));
export function highlight(code) {
  const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|f?"(?:\\.|[^"\\\n])*"|f?'(?:\\.|[^'\\\n])*')|\b(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\b|([A-Za-z_][A-Za-z0-9_]*)(?=\s*\()|([A-Za-z_][A-Za-z0-9_]*)/g;
  let out = "", last = 0, m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    const [t, c, s, n, f, w] = m;
    out += c ? `<span class="tok-c">${esc(t)}</span>` : s ? `<span class="tok-s">${esc(t)}</span>` : n ? `<span class="tok-n">${esc(t)}</span>`
      : f ? (KW.has(f) ? `<span class="tok-k">${esc(t)}</span>` : `<span class="tok-f">${esc(t)}</span>`) : KW.has(w) ? `<span class="tok-k">${esc(t)}</span>` : esc(t);
    last = m.index + t.length;
  }
  return out + esc(code.slice(last));
}
export function codeBlock(code, { label = "Copy code", lang = "python", what = "Code" } = {}) {
  const id = `c${Math.random().toString(36).slice(2, 8)}`;
  setTimeout(() => { const b = document.getElementById(id); if (b) b.onclick = () => copyText(code, what); });
  return `<div class="code"><button class="copy" id="${id}" type="button" aria-label="${esc(label)}">${esc(label)}</button><pre><code class="lang-${lang}">${lang === "python" ? highlight(code) : esc(code)}</code></pre></div>`;
}

// ---------------------------------------------------------------- pieces
export function sealHTML(letter) {
  const t = state.config?.tiers?.[letter];
  if (!t) return "";
  return `<span class="seal" style="background:${t.colour};color:${t.ink}" title="Lab seal ${letter}: ${esc(t.en)}" aria-label="Lab seal ${letter}">${letter}</span>`;
}
function qualityBadge(q) {
  if (!q || q === "not graded") return "";
  if (state.config?.tiers?.[q]) return "";
  const cls = q === "approved" ? "ok" : q === "reject" ? "bad" : "";
  return `<span class="badge ${cls}">${esc(q === "approved" ? "zoo: approved" : `zoo: ${q}`)}</span>`;
}
const kindVar = (k) => `--k:var(--k-${k})`;
export function cardHTML(c) {
  const img = c.thumbnail && /\.(png|jpe?g|gif|webp)$/i.test(c.thumbnail) ? `<img src="${fileUrl(c.thumbnail)}" alt="" loading="lazy">` : `<span class="glyph" aria-hidden="true">${GLYPH[c.kind] || ""}<small>${esc(c.subkind || KIND_LABEL[c.kind])}</small></span>`;
  return `<a class="card" href="${itemUrl(c.id)}" data-link style="${kindVar(c.kind)}">
    ${sealHTML(c.quality)}<div class="thumb">${img}</div>
    <div class="body"><h3>${esc(c.title)}</h3><p>${esc(c.summary)}</p>
    <div class="meta"><span class="badge kind"><span class="dot"></span>${esc(c.subkind || KIND_LABEL[c.kind])}</span>${c.domain && c.domain !== "general" ? `<span class="badge">${esc(c.domain)}</span>` : ""}${qualityBadge(c.quality)}${c.run_ok ? '<span class="badge ok" title="the snippet has a passing recorded run">runs ✓</span>' : ""}</div></div></a>`;
}
export function emptyState(title, text, err = false) {
  return `<div class="state${err ? " err" : ""}" role="${err ? "alert" : "status"}">${err ? ICON.err : ICON.empty}<h3>${esc(title)}</h3><p>${text}</p></div>`;
}
function lightbox(src) {
  const d = document.createElement("div"); d.className = "lightbox"; d.tabIndex = -1;
  d.innerHTML = `<img src="${src}" alt="enlarged figure">`;
  d.onclick = () => d.remove(); d.onkeydown = (e) => e.key === "Escape" && d.remove();
  document.body.append(d); d.focus();
}
function figuresHTML(paths, caption = true) {
  if (!paths?.length) return "";
  return `<div class="figs">${paths.map((p) => /\.(mp4|webm)$/i.test(p)
    ? `<figure><video src="${fileUrl(p)}" controls muted preload="metadata"></video>${caption ? `<figcaption>${esc(p.split("/").pop())}</figcaption>` : ""}</figure>`
    : `<figure><img src="${fileUrl(p)}" alt="figure ${esc(p.split("/").pop())}" loading="lazy" data-zoom>${caption ? `<figcaption>${esc(p.split("/").pop())}</figcaption>` : ""}</figure>`).join("")}</div>`;
}
const bytes = (b) => (b == null ? "" : b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : b > 1e3 ? `${(b / 1e3).toFixed(0)} kB` : `${b} B`);

// ---------------------------------------------------------------- routing
function setNav(name) { $$(".hubnav a").forEach((a) => a.classList.toggle("on", a.dataset.nav === name)); }
export function go(url, { replace = false } = {}) {
  history[replace ? "replaceState" : "pushState"]({}, "", url);
  route();
}
async function route() {
  const u = new URL(location.href), p = decodeURIComponent(u.pathname);
  document.querySelectorAll(".lightbox").forEach((d) => d.remove());
  try {
    if (p.startsWith("/item/")) { setNav("explore"); await itemView(p.slice(6)); }
    else if (p === "/create") { setNav("create"); document.title = "Create your own · PINNeAPPle Hub"; await createView(view, u.searchParams); }
    else if (p === "/mine") { setNav("mine"); await mineView(); }
    else if (p === "/" && [...u.searchParams.keys()].some((k) => ["q", "kind", "domain", "quality", "subkind"].includes(k))) { setNav("explore"); await browseView(u.searchParams); }
    else if (p === "/") { setNav("explore"); await homeView(); }
    else view.innerHTML = `<main>${emptyState("Page not found", `Nothing lives at <code>${esc(p)}</code>. <a href="/" data-link>Back to the Hub</a>.`, true)}</main>`;
  } catch (e) {
    view.innerHTML = `<main>${emptyState("Something went wrong", `${esc(e.message)}. <a href="${esc(location.pathname + location.search)}">Reload</a> or <a href="/" data-link>go home</a>.`, true)}</main>`;
  }
  if (!p.startsWith("/create")) window.scrollTo(0, 0);
  view.focus({ preventScroll: true });
}

// ---------------------------------------------------------------- home
async function homeView() {
  document.title = "PINNeAPPle Hub";
  const [h, cfg] = [await api("/api/home"), state.config];
  const kinds = Object.entries(cfg.kinds);
  view.innerHTML = `<main>
    <section class="hero">
      <h1>Everything PINNeAPPle ships, ready to use</h1>
      <p>Trained and validated models, parametric geometries, datasets, flows, worked examples with their Lab seal, and the library's tools — each with code you can copy, the run it produced, and the files to download.</p>
      <form id="heroform" role="search"><input id="heroq" type="search" placeholder="Try “airfoil”, “heat sink”, “trust card”, “wind turbine”…" aria-label="Search the Hub"><button class="primary" type="submit">Search</button></form>
      <div class="kindbar">${kinds.map(([k, t]) => `<a class="kindchip" href="/?kind=${k}" data-link style="${kindVar(k)}"><span class="dot"></span>${esc(t)} <b>${cfg.counts[k]}</b></a>`).join("")}
        <a class="kindchip" href="/create" data-link>＋ Create your own</a></div>
    </section>
    <section class="block"><h2>Featured <small>approved zoo models, A-sealed examples, geometries and flows, each with a passing run</small></h2>
      <div class="grid">${h.featured.map(cardHTML).join("")}</div></section>
    <section class="block"><h2>Browse by category</h2><div class="cats">${h.categories.map((c) => `<div class="cat" style="${kindVar(c.kind)}">
      <h3><span class="dot"></span>${esc(c.title)} <span class="badge">${c.count}</span></h3><p>${esc(CAT_TEXT[c.kind] || "")}</p>
      <ul>${c.sample.map((s) => `<li><a href="${itemUrl(s.id)}" data-link>${esc(s.title)}</a></li>`).join("")}</ul>
      <a class="btn" href="/?kind=${c.kind}" data-link>See all ${c.count} →</a></div>`).join("")}</div></section>
    <section class="block"><h2>By physical domain</h2><div class="domains">${Object.entries(h.domains).map(([d, n]) => `<button type="button" data-domain="${esc(d)}">${esc(d)}<b>${n}</b></button>`).join("")}</div></section>
  </main>`;
  $("#heroform").onsubmit = (e) => { e.preventDefault(); go(`/?q=${encodeURIComponent($("#heroq").value.trim())}`); };
  $$("[data-domain]").forEach((b) => (b.onclick = () => go(`/?domain=${encodeURIComponent(b.dataset.domain)}`)));
}
const CAT_TEXT = {
  tool: "Public functions, Flow Studio nodes, recipes that run and the MCP server's agent tools.",
  model: "Pre-trained zoo models with held-out errors, physics checks and a validity range; run them here.",
  geometry: "Realistic parametric 3-D geometries in metres: preview in 3-D, download STL / GLB / USD.",
  dataset: "Physics datasets with their verification and splits, bundled or on the Hugging Face Hub.",
  flow: "Ready engineering flows for Flow Studio: open, edit and run them, or run them headless.",
  example: "Worked studies with measured metrics, figures and the Lab's A–F seal and six-stage trust card.",
};

// ---------------------------------------------------------------- results
async function browseView(sp) {
  const f = { q: sp.get("q") || "", kind: sp.get("kind") || "", domain: sp.get("domain") || "", quality: sp.get("quality") || "", subkind: sp.get("subkind") || "" };
  $("#q").value = f.q;
  document.title = `${f.q ? `“${f.q}”` : "Browse"} · PINNeAPPle Hub`;
  const qs = new URLSearchParams(Object.entries(f).filter(([, v]) => v)); qs.set("limit", "500");
  const res = await api(`/api/search?${qs}`);
  const fc = state.config.facets, kinds = state.config.kinds;
  const group = (name, title, entries, label = (k) => k) => `<div class="fgroup" role="group" aria-label="${title}"><h3>${title}</h3>${entries.map(([k, n]) => `<label><input type="radio" name="${name}" value="${esc(k)}" ${f[name] === k ? "checked" : ""}> ${label(k)} <span class="n">${n}</span></label>`).join("")}<label><input type="radio" name="${name}" value="" ${!f[name] ? "checked" : ""}> any</label></div>`;
  const qual = Object.entries(fc.quality).sort(([a], [b]) => (a.length === 1 ? a : "Z" + a).localeCompare(b.length === 1 ? b : "Z" + b));
  const chips = Object.entries(f).filter(([, v]) => v).map(([k, v]) => `<button type="button" data-clear="${k}" aria-label="Remove filter ${k} ${esc(v)}">${esc(k === "q" ? `“${v}”` : v)} ✕</button>`).join("");
  view.innerHTML = `<main><div class="browse">
    <aside class="filters"><details open id="fdet"><summary><h2>Filters</h2><span aria-hidden="true">▾</span></summary><div class="fgroups">
      ${group("kind", "Type", Object.entries(fc.kind), (k) => esc(kinds[k]))}
      ${f.kind === "tool" ? group("subkind", "Tool type", Object.entries(fc.subkind)) : ""}
      ${group("quality", "Quality", qual, (k) => (state.config.tiers[k] ? `${sealHTML(k)} ${esc(state.config.tiers[k].en.toLowerCase())}` : esc(k)))}
      ${group("domain", "Domain", Object.entries(fc.domain), esc)}</div></details></aside>
    <section aria-label="Results"><div class="resultbar"><div class="activefilters">${chips}</div><span class="count">${res.total} result${res.total === 1 ? "" : "s"}</span></div>
      ${res.total ? `<div class="grid">${res.items.map(cardHTML).join("")}</div>` : emptyState("No matches", `Nothing in the Hub matches these filters. Try fewer words, ${chips ? '<a href="/" data-link>clear the filters</a>, ' : ""}or <a href="/create" data-link>create your own tool</a>.`)}
    </section></div></main>`;
  if (matchMedia("(max-width: 860px)").matches) $("#fdet").open = false;
  $$(".filters input").forEach((i) => (i.onchange = () => { const u = new URLSearchParams(location.search); i.value ? u.set(i.name, i.value) : u.delete(i.name); if (i.name === "kind") u.delete("subkind"); go(`/?${u}`); }));
  $$("[data-clear]").forEach((b) => (b.onclick = () => { const u = new URLSearchParams(location.search); u.delete(b.dataset.clear); go([...u.keys()].length ? `/?${u}` : "/?q="); }));
}

// ---------------------------------------------------------------- item page
function flowStudioUrl(it) {
  const base = (state.config.flow_studio || "").replace(/\/$/, "");
  const l = it.links?.flow_studio;
  if (!l) return "";
  if (l.startsWith("?node=")) return it.id.startsWith("node:") ? "" : itemUrl(`node:${l.slice(6)}`);
  return `${base}/${l}`;
}
async function itemView(id) {
  const it = await api(`/api/items/${encodeURIComponent(id)}`);
  document.title = `${it.title} · PINNeAPPle Hub`;
  const q = it.quality || {}, r = it.run || {}, res = it.results || {}, ex = it.extra || {};
  const gh = state.config.github;
  const kind = it.subkind || KIND_LABEL[it.kind];
  const fs = flowStudioUrl(it);
  const parts = [];
  parts.push(`<div class="panel"><h2>Use it</h2>${codeBlock(it.code)}
    ${it.run_cmd ? `<p class="hint" style="margin:10px 0 0">Run the full study: <code>${esc(it.run_cmd)}</code></p>` : ""}
    <div class="runbox"><div class="runhead">${r.status === "ok" ? '<span class="badge ok">recorded run passed</span>' : r.status === "failed" ? '<span class="badge bad">recorded run failed</span>' : '<span class="badge">not run yet</span>'}
      ${r.seconds != null ? `<span>${num(r.seconds)} s</span>` : ""}${it.subkind === "library call" && r.status === "ok" ? '<span title="the node form\'s default values, not a physical case">with the form\'s default values</span>' : ""}${r.recorded ? `<span>recorded ${esc(r.recorded)}</span>` : ""}${r.source ? `<span>by <code>${esc(r.source)}</code></span>` : ""}</div>
      ${r.stdout ? `<pre aria-label="Output of the recorded run">${esc(r.stdout)}</pre>` : r.status === "ok" ? '<pre class="hint">(no printed output)</pre>' : ""}${figuresHTML(r.figures, false)}</div></div>`);
  if (it.kind === "geometry") parts.push(`<div class="panel"><h2>3-D preview <span class="badge">${num(res.triangles)} triangles · light level of detail</span></h2><div class="viewer3d" id="v3d"><div class="vmsg">loading the mesh…</div></div></div>`);
  if (it.kind === "model") parts.push(tryPanel(it));
  if (it.id.startsWith("node:")) parts.push(nodePanel(it));
  if (it.description && it.description !== it.summary) parts.push(`<div class="panel"><h2>Description</h2><p class="desc">${esc(it.description)}</p></div>`);
  if (ex.functions?.length) parts.push(`<div class="panel"><h2>Functions</h2><table><tbody>${ex.functions.map((f) => `<tr><td><code>${esc(f.name)}${esc(f.signature)}</code><div class="hint">${esc(f.summary)}</div></td></tr>`).join("")}</tbody></table></div>`);
  if (ex.reference_run?.figures?.length && r.source !== "scripts/reference/run_examples.py") parts.push(`<div class="panel"><h2>In the module reference</h2>${figuresHTML(ex.reference_run.figures)}</div>`);
  if (res.flow_runs?.length) parts.push(`<div class="panel"><h2>In the bundled flows <span class="badge">already run</span></h2>
    <p class="hint">What this node measured when each flow that uses it ran headless (<code>python -m pinneapple_hub.library flowruns</code>).</p>
    ${res.flow_runs.map((f) => `<h3><a href="${itemUrl(`flow:${f.flow}`)}" data-link>${esc(f.flow)}</a> · node <code>${esc(f.node)}</code> <span class="badge ${f.status === "done" ? "ok" : "bad"}">${esc(f.status || f.flow_status)}</span></h3>${f.error ? `<p class="hint">${esc(f.error)}</p>` : ""}${f.summary ? `<p class="hint">${esc(f.summary)}</p>` : ""}${Object.keys(f.metrics).length ? `<table class="kv"><tbody>${Object.entries(f.metrics).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${num(v)}</td></tr>`).join("")}</tbody></table>` : ""}`).join("")}</div>`);
  const fr = res.flow_run;
  if (fr) parts.push(`<div class="panel"><h2>Recorded run <span class="badge ${fr.status === "done" ? "ok" : "bad"}">${esc(fr.status)}</span> <span class="badge">${num(fr.seconds)} s</span></h2>
    ${fr.error ? `<pre class="hint">${esc(fr.error)}</pre>` : ""}<table><thead><tr><th>Node</th><th>Status</th><th>Measured</th></tr></thead><tbody>${Object.entries(fr.nodes || {}).map(([id, n]) => `<tr><td><a href="${itemUrl(`node:${n.type}`)}" data-link><code>${esc(id)}</code></a><div class="hint">${esc(n.type)}</div></td><td><span class="badge ${n.status === "done" ? "ok" : ""}">${esc(n.status)}</span></td><td class="hint">${Object.entries(n.metrics || {}).map(([k, v]) => `${esc(k)} = ${num(v)}`).join(" · ")}</td></tr>`).join("")}</tbody></table>
    <p class="hint">recorded ${esc(fr.recorded)} by <code>${esc(fr.source)}</code></p></div>`);
  const metrics = res.metrics || [];
  if (metrics.length || res.figures?.length || res.videos?.length || res.reference_row) {
    parts.push(`<div class="panel"><h2>Results already run</h2>
      ${metrics.length ? `<table class="metrics"><thead><tr><th>Measured</th><th class="num">Value</th><th>Unit</th><th>How</th></tr></thead><tbody>${metrics.map((m) => `<tr><td>${esc(m.name)}</td><td class="num">${num(m.value)}</td><td>${esc(m.unit || "")}</td><td class="hint">${esc(m.how || "")}</td></tr>`).join("")}</tbody></table>` : ""}
      ${res.reference_row ? `<h3>Example row of the held-out <code>${esc(res.reference_row.split)}</code> split</h3><table class="kv"><tbody>${Object.entries(res.reference_row.outputs).map(([k, v]) => `<tr><td>${esc(k)} (stored reference)</td><td class="num">${num(v)}</td></tr>`).join("")}</tbody></table><p class="hint">The snippet above predicts this row; compare its printed output with the reference.</p>` : ""}
      ${figuresHTML([...(res.figures || []), ...(res.videos || [])])}</div>`);
  }
  if (it.kind === "geometry" && res.sources?.length) parts.push(`<div class="panel"><h2>Where the dimensions come from</h2><table><tbody>${res.sources.map((s) => `<tr><td>${esc(s.what || s.quantity || "")}</td><td>${esc(s.value ?? "")}</td><td class="hint">${esc(s.source || s.citation || "")}${s.estimate ? ' <span class="badge">estimate</span>' : ""}</td></tr>`).join("")}</tbody></table></div>`);
  if (res.splits) parts.push(`<div class="panel"><h2>Splits and columns</h2><table class="kv"><tbody>${Object.entries(res.splits).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${num(v)} rows</td></tr>`).join("")}</tbody></table>${res.split_rule ? `<p class="hint">${esc(res.split_rule)}</p>` : ""}
    ${Object.keys(res.columns || {}).length ? `<details><summary>${Object.keys(res.columns).length} columns</summary><table><tbody>${Object.entries(res.columns).map(([k, c]) => `<tr><td><code>${esc(k)}</code></td><td>${esc(c.unit || "")}</td><td class="hint">${esc(c.description || "")}</td></tr>`).join("")}</tbody></table></details>` : ""}</div>`);
  if (ex.params?.length) parts.push(`<div class="panel"><h2>Ports and parameters</h2><p class="hint">inputs: ${ex.inputs.map((p) => `<code>${esc(p.name)}: ${esc(p.type)}</code>${p.required ? "" : " (optional)"}`).join(", ") || "none"} · outputs: ${ex.outputs.map((p) => `<code>${esc(p.name)}: ${esc(p.type)}</code>`).join(", ") || "none"}</p><table><thead><tr><th>Parameter</th><th>Default</th><th>Meaning</th></tr></thead><tbody>${ex.params.map((p) => `<tr><td><code>${esc(p.name)}</code></td><td><code>${esc(JSON.stringify(p.default))}</code></td><td class="hint">${esc(p.help)}</td></tr>`).join("")}</tbody></table></div>`);
  if (ex.parameters?.length) parts.push(`<div class="panel"><h2>Parameters (regenerate with your own)</h2><table><tbody>${ex.parameters.map((p) => `<tr><td><code>${esc(p.name)}</code></td><td class="num">${num(p.default)} ${esc(p.unit || "")}</td><td class="hint">${esc(p.description || "")}</td></tr>`).join("")}</tbody></table>${codeBlock(ex.regenerate)}</div>`);

  const side = [];
  side.push(`<div class="panel"><h2>About</h2><table class="kv"><tbody>
    <tr><td>Type</td><td><span class="badge kind" style="${kindVar(it.kind)}"><span class="dot"></span>${esc(kind)}</span></td></tr>
    <tr><td>Domain</td><td>${esc(it.domain)}</td></tr>
    ${q.letter ? `<tr><td>Lab seal</td><td>${sealHTML(q.letter)} ${esc(state.config.tiers[q.letter]?.en || "")}</td></tr>` : ""}
    ${q.decision ? `<tr><td>Zoo validation</td><td><span class="badge ${q.decision === "APPROVED" ? "ok" : "bad"}">${esc(q.decision)}</span> score ${num(q.score)}</td></tr>` : ""}
    ${!q.letter && !q.decision && q.label ? `<tr><td>Quality</td><td>${esc(q.label)}</td></tr>` : ""}
    ${ex.version ? `<tr><td>Version</td><td>${esc(ex.version)}</td></tr>` : ""}${ex.license ? `<tr><td>Licence</td><td>${esc(ex.license)}</td></tr>` : ""}
    <tr><td>Id</td><td><code>${esc(it.id)}</code></td></tr></tbody></table>
    ${it.tags?.length ? `<div class="taglist" style="margin-top:8px">${it.tags.map((t) => `<a class="badge" href="/?q=${encodeURIComponent(t)}" data-link>${esc(t)}</a>`).join("")}</div>` : ""}</div>`);
  if (q.stages && Object.keys(q.stages).length) side.push(`<div class="panel"><h2>Trust card <span class="badge">six questions</span></h2>${q.reason ? `<p class="hint">${esc(q.reason)}</p>` : ""}<div class="stages">${Object.values(q.stages).map((s) => `<div class="stage"><b>${esc(s.title)}</b><span class="st ${esc(s.status)}">${esc(s.status)}</span><p>${esc(s.question)}</p>${s.evidence?.length ? `<ul>${s.evidence.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>` : ""}${s.next ? `<p>next: ${esc(s.next)}</p>` : ""}</div>`).join("")}</div></div>`);
  if (q.checks?.length) side.push(`<div class="panel"><h2>Validation checks</h2><table><tbody>${q.checks.map((c) => `<tr><td>${esc(c.name)}</td><td><span class="badge ${c.status === "supports" ? "ok" : "bad"}">${esc(c.status)}</span></td></tr>`).join("")}</tbody></table></div>`);
  if (q.how && !q.letter) side.push(`<div class="panel"><h2>Not curated yet</h2><p class="hint">Curate it with</p>${codeBlock(q.how, { lang: "sh", label: "Copy" })}</div>`);
  if (it.files?.length) side.push(`<div class="panel files"><h2>Download</h2>${it.files.map((f) => `<a class="btn" href="${esc(f.path ? fileUrl(f.path) : f.url)}" ${f.path ? "download" : 'rel="noopener" target="_blank"'}>${f.path ? ICON.down : ICON.ext}<span>${esc(f.label)}</span><small>${bytes(f.bytes)}${!f.path && f.url ? " · Hub" : ""}</small></a>`).join("")}</div>`);
  const links = [];
  if (fs) links.push(`<a href="${esc(fs)}" ${fs.startsWith("/") ? "data-link" : 'target="_blank" rel="noopener"'}>${ICON.flow} ${it.links.flow_studio.startsWith("?node") ? "Its Flow Studio node" : "Open in Flow Studio"}</a>`);
  for (const [k, v] of Object.entries(it.links || {})) {
    if (!v || k === "flow_studio") continue;
    const href = v.startsWith("http") ? v : `${gh}/${v}`;
    links.push(`<a href="${esc(href)}" target="_blank" rel="noopener">${ICON.ext} ${esc({ guide: "Guide", reference: "Module reference", source: "Source code", hub: "Hugging Face", lab: "Lab catalogue" }[k] || k)}</a>`);
  }
  if (links.length) side.push(`<div class="panel links"><h2>Links</h2>${links.join("")}</div>`);
  if (it.related?.length) side.push(`<div class="panel rel"><h2>Related</h2>${it.related.slice(0, 10).map((c) => `<a href="${itemUrl(c.id)}" data-link style="${kindVar(c.kind)}"><span class="dot"></span><span>${esc(c.title)}</span>${c.quality.length === 1 ? sealHTML(c.quality) : ""}<small>${esc(c.subkind || KIND_LABEL[c.kind])}</small></a>`).join("")}</div>`);

  const crumbKind = it.id.startsWith("mine:") ? `<a href="/mine" data-link>My Hub</a>` : `<a href="/?kind=${it.kind}" data-link>${esc(state.config.kinds[it.kind])}</a>`;
  view.innerHTML = `<main><nav class="crumbs" aria-label="Breadcrumb"><a href="/" data-link>Hub</a> / ${crumbKind} / <span>${esc(it.title)}</span></nav>
    <div class="ihead"><div><span class="badge kind" style="${kindVar(it.kind)}"><span class="dot"></span>${esc(kind)}</span> ${q.letter ? sealHTML(q.letter) : ""}
      <h1>${esc(it.title)}</h1><p class="lead">${esc(it.summary)}</p></div>
      <div class="actions"><button class="btn primary" id="copymain" type="button">${ICON.copy} Copy code</button>${fs ? `<a class="btn" href="${esc(fs)}" ${fs.startsWith("/") ? "data-link" : 'target="_blank" rel="noopener"'}>${ICON.flow} ${it.links.flow_studio.startsWith("?node") ? "Flow Studio node" : "Open in Flow Studio"}</a>` : ""}${it.id.startsWith("node:") ? `<button class="btn" id="node-open-top" type="button">${ICON.flow} Open in Flow Studio</button>` : ""}${it.kind === "flow" || it.flow ? `<button class="btn" id="dlflow" type="button">${ICON.down} Flow JSON</button>` : ""}</div></div>
    <div class="cols"><div>${parts.join("")}</div><aside class="side">${side.join("")}</aside></div></main>`;
  $("#copymain").onclick = () => copyText(it.code);
  if ($("#dlflow")) $("#dlflow").onclick = async () => {
    const doc = it.flow || await api(`/files/pinneapple_flow/flows/${encodeURIComponent(it.name)}.json`);
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 1)], { type: "application/json" })); a.download = `${it.name}.flow.json`; a.click();
  };
  if (it.kind === "geometry") import("./viewer.js").then((m) => m.showGeometry($("#v3d"), fileUrl(ex.glb), res.dimensions_m)).catch((e) => { $("#v3d .vmsg").textContent = `3-D preview unavailable: ${e.message}`; });
  if (it.kind === "model") wireTry(it);
  if (it.id.startsWith("node:")) wireNode(it);
}

// ---------------------------------------------------------------- zoo inference
function tryPanel(it) {
  const ins = it.extra.inputs || [];
  return `<div class="panel" id="try"><h2>Try it <span class="badge">inference</span></h2>
    <p class="hint">Inputs start at the example row of the model's own held-out split. ${it.extra.onnx ? "Run it in your browser (the model's ONNX graph and its query checks, no server) or on the server (PyTorch, <code>pp.zoo</code>)." : "Runs on the server (PyTorch, <code>pp.zoo</code>)."}</p>
    <form class="tryform" id="tryform">${ins.map((i) => `<label>${esc(i.name)} <span class="u">${esc(i.unit && i.unit !== "-" ? i.unit : "")}${Array.isArray(i.range) ? ` · ${num(i.range[0])}–${num(i.range[1])}` : ""}</span><input name="${esc(i.name)}" type="number" step="any" value="${i.example != null ? +i.example.toPrecision(6) : Array.isArray(i.range) ? (i.range[0] + i.range[1]) / 2 : ""}" required title="${esc(i.description || "")}"></label>`).join("")}</form>
    <div class="row" style="margin-top:12px">${it.extra.onnx ? '<button class="btn primary" id="run-browser" type="button">Run in the browser</button>' : ""}<button class="btn ${it.extra.onnx ? "" : "primary"}" id="run-server" type="button">Run on the server</button></div>
    <div class="tryout" id="tryout" aria-live="polite"></div></div>`;
}
function wireTry(it) {
  const read = () => Object.fromEntries($$("#tryform input").map((i) => [i.name, parseFloat(i.value)]));
  const out = $("#tryout");
  const show = (o, where, ms) => {
    const v = o.verdict || {};
    out.innerHTML = `<div class="verdict ${esc(v.status || "")}">${esc(v.title || v.status || "")}${v.text ? ` — <span style="font-weight:500">${esc(v.text)}</span>` : ""}</div>
      <table class="metrics"><thead><tr><th>Output</th><th class="num">Value</th><th>Unit</th></tr></thead><tbody>${Object.entries(o.outputs).map(([k, x]) => `<tr><td>${esc(k)}</td><td class="num">${num(x)}</td><td>${esc((it.extra.outputs.find((u) => u.name === k) || {}).unit || "")}</td></tr>`).join("")}</tbody></table>
      ${o.checks?.length ? `<details><summary>${o.checks.length} query checks</summary><table><tbody>${o.checks.map((c) => `<tr><td><span class="badge ${c.status === "pass" ? "ok" : c.status === "fail" ? "bad" : ""}">${esc(c.status)}</span></td><td class="hint">${esc(c.detail_text || c.name || c.id || "")}</td></tr>`).join("")}</tbody></table></details>` : ""}
      <p class="hint">${esc(where)}${ms != null ? ` · ${num(ms)} ms` : ""}</p>`;
  };
  $("#run-server").onclick = async (e) => {
    e.target.disabled = true; out.innerHTML = '<p class="hint">running on the server…</p>';
    try { const t0 = performance.now(); const o = await api(`/api/models/${it.name}/predict`, { method: "POST", body: JSON.stringify({ inputs: read() }) }); show(o, "server · PyTorch · pp.zoo", performance.now() - t0); }
    catch (err) { out.innerHTML = emptyState("Inference failed", esc(err.message), true); }
    finally { e.target.disabled = false; }
  };
  if ($("#run-browser")) $("#run-browser").onclick = async (e) => {
    e.target.disabled = true; out.innerHTML = '<p class="hint">loading the ONNX graph…</p>';
    try {
      const zoo = await import("/zoo-runtime/zoo.js");
      state.models ??= {};
      state.models[it.name] ??= await zoo.loadModel("/files/data/web/zoo", it.name);
      const m = state.models[it.name], x = read();
      const row = m.inputs.map((k) => x[k]);
      const r = await m.run([row], row);
      show({ outputs: r.rows[0], verdict: r.query.verdict, checks: r.query.checks }, "in your browser · ONNX (onnx.js)", m.lastMs);
    } catch (err) { out.innerHTML = emptyState("Browser inference failed", `${esc(err.message)} — try the server.`, true); }
    finally { e.target.disabled = false; }
  };
}

// ---------------------------------------------------------------- a Flow Studio node: run it, open it as a flow, add it to a flow
function nodePanel(it) {
  const ex = it.extra, props = ex.params_schema?.properties || {};
  const basic = Object.entries(props).filter(([, s]) => !s["x-advanced"]), adv = Object.entries(props).filter(([, s]) => s["x-advanced"]);
  const needs = (ex.inputs || []).filter((p) => p.required);
  return `<div class="panel" id="node"><h2>Run it <span class="badge">${esc(it.subkind)}</span></h2>
    <p class="hint">${needs.length ? `Needs ${needs.map((p) => `<code>${esc(p.name)}: ${esc(p.type)}</code>`).join(", ")} from other nodes: “Open in Flow Studio” builds the flow with inputs → this node → a report.` : "No inputs needed: run it here with these parameters (a sandboxed job), or open it in Flow Studio as inputs → node → report."}${ex.available ? "" : ` <b>Not runnable on this server:</b> ${esc(ex.install)}`}</p>
    ${Object.keys(props).length ? `<form class="params" id="node-form" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:0 12px" onsubmit="return false">${basic.map(([k, sc]) => field(it.name, k, sc, sc.default)).join("")}</form>
      ${adv.length ? `<details><summary class="hint">${adv.length} advanced parameter${adv.length > 1 ? "s" : ""}</summary><form class="params" id="node-form-adv" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:0 12px" onsubmit="return false">${adv.map(([k, sc]) => field(it.name, k, sc, sc.default)).join("")}</form></details>` : ""}` : '<p class="hint">No parameters.</p>'}
    <div class="row" style="margin-top:12px">${!needs.length && ex.available ? '<button class="btn primary" id="node-run" type="button">Run in a sandbox</button>' : ""}
      <button class="btn ${needs.length || !ex.available ? "primary" : ""}" id="node-open" type="button">${ICON.flow} Open in Flow Studio</button>
      <button class="btn" id="node-add" type="button">Add to a flow</button>
      <a class="btn" id="node-compose" href="#" data-link>Compose with it</a></div>
    <div id="node-out" aria-live="polite"></div></div>`;
}
function wireNode(it) {
  const out = $("#node-out");
  const params = () => {
    const p = {}, defaults = Object.fromEntries((it.extra.params || []).map((x) => [x.name, x.default]));
    for (const inp of $$("#node [data-k]")) {
      const r = readInput(inp);
      if (!r.ok) return null;
      if (JSON.stringify(r.value) !== JSON.stringify(defaults[inp.dataset.k])) p[inp.dataset.k] = r.value;
    }
    return p;
  };
  const composeHref = () => { const p = params() || {}; return `/create?blocks=${encodeURIComponent(it.name)}&name=${encodeURIComponent(it.title)}${Object.keys(p).length ? `&params=${b64(p)}` : ""}`; };
  $("#node-compose").onclick = (e) => { e.preventDefault(); go(composeHref()); };
  const link = async () => { const p = params(); if (!p) return null; return api("/api/flowlink/node", { method: "POST", body: JSON.stringify({ type: it.name, params: p }) }); };
  const open = async () => {
    try {
      const r = await link(); if (!r) return;
      window.open(r.open, "_blank", "noopener");
      out.innerHTML = `<p class="hint">Opened in Flow Studio: ${esc(r.source)} · ${r.flow.nodes.length} nodes${r.issues.length ? ` · ${r.issues.length} issue(s): ${esc(r.issues.map((i) => i.message).join("; "))}` : ""}</p>`;
    } catch (err) { out.innerHTML = emptyState("Could not build the flow", esc(err.message), true); }
  };
  $("#node-open").onclick = open;
  if ($("#node-open-top")) $("#node-open-top").onclick = open;
  $("#node-add").onclick = async () => {
    try { const r = await link(); if (!r) return; window.open(r.add, "_blank", "noopener"); out.innerHTML = '<p class="hint">Flow Studio is asking which flow to add it to.</p>'; }
    catch (err) { out.innerHTML = emptyState("Could not add it", esc(err.message), true); }
  };
  if ($("#node-run")) $("#node-run").onclick = async (e) => {
    const p = params(); if (!p) return;
    e.target.disabled = true;
    try {
      const job = await api("/api/compose/validate", { method: "POST", body: JSON.stringify({ name: it.title, blocks: [{ type: it.name, params: p }] }) });
      for (;;) {
        const j = await api(`/api/jobs/${job.id}`);
        out.innerHTML = `<div class="runbox"><div class="runhead">${esc(j.status)} · ${esc(j.message || "")}</div><div class="progress" style="margin:8px 12px 12px"><div style="width:${Math.round((j.progress || 0) * 100)}%"></div></div></div>`;
        if (["done", "failed", "cancelled"].includes(j.status)) {
          const x = j.result || {};
          out.innerHTML = j.status !== "done" ? emptyState("The run did not finish", esc(j.error || j.status), true)
            : `<div class="runbox"><div class="runhead">${x.ok ? '<span class="badge ok">ran</span>' : `<span class="badge bad">${x.timed_out ? "time limit reached" : "failed"}</span>`}<span>${num(x.seconds)} s in the sandbox</span></div>${x.stdout ? `<pre>${esc(x.stdout)}</pre>` : ""}${x.stderr && !x.ok ? `<pre style="color:var(--bad)">${esc(x.stderr)}</pre>` : ""}${(x.figures_png_b64 || []).map((f) => `<img src="data:image/png;base64,${f}" alt="figure from the run" data-zoom style="max-width:100%">`).join("")}</div>`;
          break;
        }
        await new Promise((r) => setTimeout(r, 700));
      }
    } catch (err) { out.innerHTML = emptyState(err.status === 429 ? "Limit reached" : "Could not run it", esc(err.message), true); }
    finally { e.target.disabled = false; }
  };
}

// ---------------------------------------------------------------- my hub
async function mineView() {
  document.title = "My Hub · PINNeAPPle Hub";
  const r = await api("/api/mine");
  view.innerHTML = `<main><section class="block"><h2>My Hub <small>tools you composed and saved on this server (kept per login, API key or address)</small></h2>
    ${r.items.length ? `<div class="grid">${r.items.map((c) => `<div style="position:relative">${cardHTML(c)}<button class="btn" style="position:absolute;bottom:10px;right:10px;padding:4px 9px" data-del="${esc(c.id)}" aria-label="Delete ${esc(c.title)}">Delete</button></div>`).join("")}</div>`
      : emptyState("Nothing saved yet", 'Compose a tool from existing blocks in <a href="/create" data-link>Create your own</a>, validate it, then “Save to My Hub”.')}</section></main>`;
  $$("[data-del]").forEach((b) => (b.onclick = async (e) => {
    e.preventDefault();
    if (!confirm("Delete this tool from My Hub?")) return;
    await api(`/api/mine/${encodeURIComponent(b.dataset.del)}`, { method: "DELETE" }); toast("Deleted"); mineView();
  }));
}

// ---------------------------------------------------------------- quick search (Ctrl+K) and shortcuts
const pal = $("#palette"), pq = $("#pq"), pres = $("#pres");
let palSel = 0, palRows = [], palT = 0;
function openPalette() { pal.hidden = false; pq.value = ""; pres.innerHTML = '<li class="hint">Type to search every item</li>'; pq.focus(); }
function closePalette() { pal.hidden = true; }
async function palSearch() {
  const q = pq.value.trim();
  if (!q) { pres.innerHTML = ""; palRows = []; return; }
  const r = await api(`/api/search?q=${encodeURIComponent(q)}&limit=12`);
  palRows = r.items; palSel = 0;
  pres.innerHTML = palRows.length ? palRows.map((c, i) => `<li role="option" id="po${i}" aria-selected="${i === 0}" data-i="${i}" style="${kindVar(c.kind)}"><span class="dot"></span>${esc(c.title)}<small>${esc(c.subkind || KIND_LABEL[c.kind])}</small></li>`).join("") : '<li class="hint">No matches</li>';
  $$("li[data-i]", pres).forEach((li) => (li.onclick = () => { closePalette(); go(itemUrl(palRows[+li.dataset.i].id)); }));
}
pq.oninput = () => { clearTimeout(palT); palT = setTimeout(palSearch, 120); };
pq.onkeydown = (e) => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault(); if (!palRows.length) return;
    palSel = (palSel + (e.key === "ArrowDown" ? 1 : palRows.length - 1)) % palRows.length;
    $$("li[data-i]", pres).forEach((li, i) => li.setAttribute("aria-selected", i === palSel));
    pq.setAttribute("aria-activedescendant", `po${palSel}`);
  } else if (e.key === "Enter") {
    e.preventDefault(); closePalette();
    palRows[palSel] ? go(itemUrl(palRows[palSel].id)) : go(`/?q=${encodeURIComponent(pq.value.trim())}`);
  }
};
pal.onclick = (e) => { if (e.target === pal) closePalette(); };
document.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); pal.hidden ? openPalette() : closePalette(); }
  else if (e.key === "Escape" && !pal.hidden) closePalette();
  else if (e.key === "/" && !typing) { e.preventDefault(); $("#q").focus(); $("#q").select(); }
});
$("#topsearch").onsubmit = (e) => { e.preventDefault(); const q = $("#q").value.trim(); go(q ? `/?q=${encodeURIComponent(q)}` : "/"); };

// ---------------------------------------------------------------- theme, links, start
const root = document.documentElement;
function applyTheme(t) { if (t) root.dataset.theme = t; else delete root.dataset.theme; }
try { applyTheme(localStorage.getItem("hub-theme")); } catch { /* storage blocked */ }
$("#theme").onclick = () => {
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  const t = dark ? "light" : "dark"; applyTheme(t);
  try { localStorage.setItem("hub-theme", t); } catch { /* storage blocked */ }
  toast(`${t === "dark" ? "Dark" : "Light"} theme`);
};
document.addEventListener("click", (e) => {
  const z = e.target.closest("img[data-zoom]");
  if (z) { e.preventDefault(); lightbox(z.src); return; }
  const a = e.target.closest("a[data-link]");
  if (a && !e.ctrlKey && !e.metaKey && !e.shiftKey && e.button === 0) { e.preventDefault(); go(a.getAttribute("href")); }
});
window.addEventListener("popstate", route);
$("#apidocs").href = "/docs";
try { state.config = await api("/api/config"); route(); }
catch (e) { view.innerHTML = `<main>${emptyState("The Hub is not reachable", `${esc(e.message)}. Check that the server is running, then reload.`, true)}</main>`; }
