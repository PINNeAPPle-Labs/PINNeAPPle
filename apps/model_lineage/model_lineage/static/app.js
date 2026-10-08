"use strict";
// Engineering Model Lineage UI: build the lineage, draw the digital thread, answer questions, export.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const KIND_COL = { geometry: "#2563eb", mesh: "#0891b2", simulation: "#059669", result: "#65a30d", experiment: "#ca8a04", "post-processing": "#d97706",
  dataset: "#7e22ce", model: "#db2777", prediction: "#e11d48", report: "#475569", code: "#64748b", file: "#94a3b8" };
let FILES = [], G = null, SEL = null, HI = null, NODE = {}, POS = {}, VIEW = null;

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) { $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n)); $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`)); }
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
  $("#filelist").innerHTML = FILES.slice(0, 6).map((x, i) => `<div><span><b>${esc(x.n)}</b></span><button data-i="${i}">×</button></div>`).join("") + (FILES.length > 6 ? `<div><span>+ ${FILES.length - 6} files</span><button data-i="all">clear</button></div>` : "");
  $$("#filelist button").forEach((b) => b.onclick = () => { if (b.dataset.i === "all") FILES = []; else FILES.splice(+b.dataset.i, 1); renderFiles(); });
  $("#run").disabled = !FILES.length;
}
async function load(url, opts, btn) {
  const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Building…`; $("#status").textContent = "";
  try {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
    G = j; SEL = null; HI = null; render(); showTab("thread"); window.scrollTo(0, 0);
  } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.innerHTML = old; if (btn.id === "run") renderFiles(); }
}
$("#run").addEventListener("click", (e) => { const fd = new FormData(); FILES.forEach((x) => fd.append("files", x.f, x.n)); load("/api/lineage", { method: "POST", body: fd }, e.currentTarget); });

// ------------------------------------------------------------------ graph queries (client side)
const parents = (id) => (NODE[id] ? NODE[id].inputs || [] : []).filter((i) => NODE[i]);
const kids = (id) => (NODE[id] ? NODE[id].outputs || [] : []);
function walk(id, next) { const out = [], seen = new Set([id]), q = [...next(id)]; while (q.length) { const n = q.shift(); if (seen.has(n)) continue; seen.add(n); out.push(n); q.push(...next(n)); } return out; }
const upstream = (id) => walk(id, parents), downstream = (id) => walk(id, kids);

// ------------------------------------------------------------------ layout: columns by layer, rows by barycentre
const NW = 220, NH = 56, GX = 70, GY = 18;
function layout() {
  const cols = {};
  for (const n of G.nodes) (cols[n.layer] = cols[n.layer] || []).push(n.id);
  const L = Object.keys(cols).map(Number).sort((a, b) => a - b);
  const rank = {};
  for (const l of L) cols[l].sort((a, b) => NODE[a].kind.localeCompare(NODE[b].kind) || a.localeCompare(b)).forEach((id, i) => (rank[id] = i));
  for (let it = 0; it < 4; it++) for (const l of L.slice(1)) {
    const bc = (id) => { const p = parents(id); return p.length ? p.reduce((s, x) => s + rank[x], 0) / p.length : rank[id]; };
    cols[l].sort((a, b) => bc(a) - bc(b)).forEach((id, i) => (rank[id] = i));
  }
  const maxRows = Math.max(...L.map((l) => cols[l].length));
  POS = {};
  for (const l of L) { const off = (maxRows - cols[l].length) * (NH + GY) / 2; cols[l].forEach((id, i) => (POS[id] = { x: 20 + l * (NW + GX), y: 20 + off + i * (NH + GY) })); }
  return { w: 40 + (L.length) * (NW + GX) - GX, h: 40 + maxRows * (NH + GY) - GY };
}

function drawGraph() {
  const { w, h } = layout();
  const hiSet = HI ? new Set([HI.root, ...HI.nodes]) : null;
  let g = `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="#a3a3c2"/></marker>
    <marker id="arrh" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="#7e22ce"/></marker></defs>`;
  for (const [a, b] of G.edges) {
    const p = POS[a], q = POS[b]; if (!p || !q) continue;
    const x1 = p.x + NW, y1 = p.y + NH / 2, x2 = q.x - 3, y2 = q.y + NH / 2, mx = (x1 + x2) / 2;
    const on = hiSet && hiSet.has(a) && hiSet.has(b);
    g += `<path class="edge ${hiSet ? (on ? "hi" : "dim") : ""}" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" marker-end="url(#${on ? "arrh" : "arr"})"/>`;
  }
  for (const n of G.nodes) {
    const p = POS[n.id], col = KIND_COL[n.kind] || "#94a3b8";
    const label = n.id.length > 25 ? "…" + n.id.slice(-24) : n.id;
    const sub = n.software ? `${n.software}${n.version ? " " + n.version : ""}` : (n.name && n.name !== n.id ? n.name : "");
    const badge = n.flag === "fail" ? `<circle cx="${p.x + NW - 12}" cy="${p.y + 12}" r="8" fill="#b42318"/><text x="${p.x + NW - 12}" y="${p.y + 16}" text-anchor="middle" style="fill:#fff;font-weight:800">✗</text>`
      : n.flag === "warn" ? `<circle cx="${p.x + NW - 12}" cy="${p.y + 12}" r="8" fill="#d97706"/><text x="${p.x + NW - 12}" y="${p.y + 16}" text-anchor="middle" style="fill:#fff;font-weight:800">!</text>`
      : n.verified ? `<text x="${p.x + NW - 12}" y="${p.y + 17}" text-anchor="middle" style="fill:#1a7f4b;font-weight:800;font-size:13px">✓</text>` : "";
    g += `<g class="node ${hiSet && !hiSet.has(n.id) ? "dim" : ""} ${SEL === n.id ? "sel" : ""}" data-id="${esc(n.id)}">
      <rect x="${p.x}" y="${p.y}" width="${NW}" height="${NH}" rx="9" fill="#fff" stroke="${col}"/><rect x="${p.x}" y="${p.y}" width="6" height="${NH}" rx="3" fill="${col}"/>
      <text class="k" x="${p.x + 14}" y="${p.y + 15}">${esc(n.kind)}</text><text x="${p.x + 14}" y="${p.y + 32}" style="font-weight:700">${esc(label)}</text>
      <text x="${p.x + 14}" y="${p.y + 47}" style="fill:#5d6b82;font-size:11px">${esc(sub.length > 30 ? sub.slice(0, 29) + "…" : sub)}</text>${badge}</g>`;
  }
  const svg = $("#graph");
  if (!VIEW) {
    // fit the content but never shrink text below ~60%: wide threads start at the left and pan
    const cw = svg.clientWidth || 760, k = Math.min(1.65, Math.max(1, w / cw)), ch = Math.min(620, Math.max(260, h / k + 20));
    svg.style.height = ch + "px";
    const kk = Math.max(k, h / ch);
    VIEW = { w: cw * kk, h: ch * kk }; VIEW.x = Math.min(0, (w - VIEW.w) / 2); VIEW.y = (h - VIEW.h) / 2;
  }
  svg.setAttribute("viewBox", `${VIEW.x} ${VIEW.y} ${VIEW.w} ${VIEW.h}`);
  svg.innerHTML = g;
  $$(".node", svg).forEach((el) => el.addEventListener("click", (e) => { e.stopPropagation(); select(el.dataset.id); }));
}

function select(id) { SEL = id; HI = null; drawGraph(); card(); }
function card() {
  const n = NODE[SEL];
  if (!n) { $("#card").innerHTML = `<p class="meta">Click an artifact to see where it came from and what depends on it.</p>`; return; }
  const col = KIND_COL[n.kind] || "#94a3b8";
  const row = (k, v) => (v !== undefined && v !== null && v !== "" ? `<dt>${k}</dt><dd>${v}</dd>` : "");
  const params = n.parameters ? Object.entries(n.parameters).map(([k, v]) => `${esc(k)} = <b>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</b>`).join("<br>") : "";
  // the thread: follow first inputs upstream
  const chain = []; let cur = SEL, guard = 0;
  while (cur && guard++ < 30) { chain.unshift(cur); const p = parents(cur); cur = p[0]; }
  const thread = chain.map((id) => `${id}${NODE[id].version ? `  (${NODE[id].version})` : NODE[id].parameters && NODE[id].parameters.nCells ? `  (${NODE[id].parameters.nCells.toLocaleString("en-US")} cells)` : ""}`).join("\n   ↓\n");
  const hashLine = n.hash ? `<span class="mono">${esc(n.hash.slice(0, 16))}…</span>${n.actual_hash ? (n.verified ? ` <span class="pill ok">matches the file</span>` : ` <span class="pill critical">file is ${esc(n.actual_hash.slice(0, 12))}…</span>`) : ""}` : "";
  $("#card").innerHTML = `<div class="card"><span class="kind" style="background:${col}">${esc(n.kind)}</span>
    <h3 style="margin-top:6px">${esc(n.id)}</h3>${n.name && n.name !== n.id ? `<div class="meta">${esc(n.name)}</div>` : ""}
    <dl>${row("file", esc(n.file))}${row("version", esc(n.version))}${row("software", esc(n.software))}${row("timestamp", esc(n.timestamp))}${row("sha256", hashLine)}${row("owner", esc(n.owner))}${row("origin", esc(n.origin))}${row("item", esc(n.item))}${row("parameters", params)}${row("notes", esc(n.notes))}${row("found by", esc(n.detected || "declared"))}</dl>
    <div class="meta">provenance completeness</div><div class="bar"><div style="width:${100 * n.completeness}%"></div></div>
    <div class="actions"><button id="b-up" class="${HI && HI.dir === "up" ? "on" : ""}">How was this produced? (${upstream(SEL).length})</button><button id="b-down" class="${HI && HI.dir === "down" ? "on" : ""}">What does it affect? (${downstream(SEL).length})</button><button id="b-clear">Clear</button></div>
    ${chain.length > 1 ? `<div class="meta">The thread</div><div class="thread">${esc(thread)}</div>` : ""}</div>`;
  $("#b-up").onclick = () => { HI = { root: SEL, nodes: upstream(SEL), dir: "up" }; drawGraph(); card(); };
  $("#b-down").onclick = () => { HI = { root: SEL, nodes: downstream(SEL), dir: "down" }; drawGraph(); card(); };
  $("#b-clear").onclick = () => { HI = null; drawGraph(); card(); };
}

function questions() {
  const ids = G.nodes.map((n) => n.id);
  const derived = G.nodes.filter((n) => ["dataset", "model", "prediction", "report", "result", "simulation"].includes(n.kind)).map((n) => n.id);
  const opt = (list, pick) => list.map((i) => `<option ${i === pick ? "selected" : ""}>${esc(i)}</option>`).join("");
  const last = derived[derived.length - 1] || ids[ids.length - 1];
  $("#questions").innerHTML = `<div class="q">
    <label>Which geometry revision produced <select id="q1">${opt(derived.length ? derived : ids, last)}</select></label><div class="ans" id="a1"></div>
    <label>Which simulations is it built on, with which solver? <select id="q2">${opt(derived.length ? derived : ids, last)}</select></label><div class="ans" id="a2"></div>
    <label>If this changes, what has to be redone? <select id="q3">${opt(ids, ids[0])}</select></label><div class="ans" id="a3"></div></div>`;
  const ans = () => {
    const g1 = upstream($("#q1").value).filter((i) => NODE[i].kind === "geometry");
    $("#a1").innerHTML = g1.length ? g1.map((i) => `<b>${esc(i)}</b>${NODE[i].version ? ` (version ${esc(NODE[i].version)})` : ""}${NODE[i].file ? ` <span class="meta">${esc(NODE[i].file)}</span>` : ""}`).join("<br>") : "No geometry upstream.";
    const s2 = [$("#q2").value, ...upstream($("#q2").value)].filter((i) => NODE[i].kind === "simulation");
    $("#a2").innerHTML = s2.length ? s2.map((i) => `<b>${esc(i)}</b>: ${esc(NODE[i].software || "?")} ${esc(NODE[i].version || "")}${NODE[i].timestamp ? ` <span class="meta">${esc(NODE[i].timestamp)}</span>` : ""}`).join("<br>") : "No simulation upstream.";
    const d3 = downstream($("#q3").value);
    $("#a3").innerHTML = d3.length ? `${d3.length} artifact${d3.length > 1 ? "s" : ""}: ` + d3.map((i) => `<b>${esc(i)}</b>`).join(", ") : "Nothing depends on it.";
  };
  $$("#questions select").forEach((s) => s.onchange = ans); ans();
}

function render() {
  NODE = Object.fromEntries(G.nodes.map((n) => [n.id, n])); VIEW = null;
  const s = G.summary;
  const probs = G.checks.filter((c) => c.status === "fail" || c.status === "warn");
  const kinds = Object.entries(s.by_kind).map(([k, v]) => `<span><i style="background:${KIND_COL[k] || "#94a3b8"}"></i>${v} ${esc(k)}</span>`).join("");
  $("#thread").innerHTML = `
    <div class="panel">
      <div class="toolbar"><h2 style="margin:0">${esc(G.project || (G.example ? G.example.description : "Lineage"))}</h2>
        <div class="noprint" style="display:flex;gap:6px"><button class="ghost" id="dl-json">Lineage JSON</button><button class="ghost" id="dl-prov">W3C PROV-JSON</button><button class="ghost" id="dl-md">Markdown</button></div></div>
      <div class="verdict ${G.status}"><div class="big">${G.status}</div><div>${s.artifacts} artifacts · ${s.links} links · ${s.verified} verified against their file · provenance ${Math.round(100 * s.completeness)}% complete · ${G.mode === "detected" ? "detected from the files" : "declared"}${G.example ? `<br><span style="font-size:13px">${esc(G.example.description)}</span>` : ""}</div></div>
      <div class="graphbox"><div class="gbar"><button id="z-in">+</button><button id="z-out">−</button><button id="z-fit">fit</button></div><svg id="graph"></svg></div>
      <div class="legend">${kinds}<span>✓ hash verified · <b style="color:#b42318">✗</b> failed check · <b style="color:#d97706">!</b> warning · drag to pan, scroll to zoom</span></div>
      <div class="grid2b" style="margin-top:14px"><div id="card"></div><div><h3 style="margin-top:0">Questions</h3><div id="questions"></div></div></div>
    </div>
    <div class="panel"><h3 style="margin-top:0">Checks</h3>${(probs.length ? probs : G.checks.filter((c) => c.status === "pass")).map((c) => `<div class="chk"><div class="ic ${c.status}">${{ fail: "✗", warn: "!", pass: "✓", info: "i" }[c.status]}</div><div><div><b>${esc(c.title)}</b></div>${c.detail ? `<div class="why">${esc(c.detail)}</div>` : ""}${c.fix ? `<div class="fix"><b>Fix:</b> ${esc(c.fix)}</div>` : ""}</div></div>`).join("")}
      ${G.checks.filter((c) => c.status === "info").length ? `<details style="margin-top:8px"><summary class="meta">${G.checks.filter((c) => c.status === "info").length} notes on incomplete provenance</summary>${G.checks.filter((c) => c.status === "info").map((c) => `<div class="meta">${esc(c.title)}</div>`).join("")}</details>` : ""}</div>
    <div class="panel">${window.renderScope ? renderScope(G.scope) : ""}</div>`;
  drawGraph(); card(); questions();
  const svg = $("#graph"); let drag = null;
  svg.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY, vx: VIEW.x, vy: VIEW.y }; svg.style.cursor = "grabbing"; });
  window.addEventListener("mouseup", () => { drag = null; svg.style.cursor = "grab"; });
  svg.addEventListener("mousemove", (e) => { if (!drag) return; const k = VIEW.w / svg.clientWidth; VIEW.x = drag.vx - (e.clientX - drag.x) * k; VIEW.y = drag.vy - (e.clientY - drag.y) * k; svg.setAttribute("viewBox", `${VIEW.x} ${VIEW.y} ${VIEW.w} ${VIEW.h}`); });
  const zoom = (f) => { const cx = VIEW.x + VIEW.w / 2, cy = VIEW.y + VIEW.h / 2; VIEW.w *= f; VIEW.h *= f; VIEW.x = cx - VIEW.w / 2; VIEW.y = cy - VIEW.h / 2; svg.setAttribute("viewBox", `${VIEW.x} ${VIEW.y} ${VIEW.w} ${VIEW.h}`); };
  svg.addEventListener("wheel", (e) => { e.preventDefault(); zoom(e.deltaY > 0 ? 1.12 : 0.89); }, { passive: false });
  $("#z-in").onclick = () => zoom(0.8); $("#z-out").onclick = () => zoom(1.25); $("#z-fit").onclick = () => { VIEW = null; drawGraph(); };
  svg.addEventListener("click", () => { SEL = null; HI = null; drawGraph(); card(); });
  $("#dl-json").onclick = () => save("lineage.json", JSON.stringify(G.lineage, null, 1), "application/json");
  $("#dl-prov").onclick = () => save("lineage.prov.json", JSON.stringify(G.prov, null, 1), "application/json");
  $("#dl-md").onclick = () => save("lineage.md", G.markdown, "text/markdown");
  const firstFlag = G.nodes.find((n) => n.flag === "fail") || G.nodes.find((n) => n.flag === "warn");
  if (firstFlag) select(firstFlag.id);
}
function save(name, text, type) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }

(async function init() {
  try {
    const M = await (await fetch("/api/meta")).json();
    $("#examples").innerHTML = M.examples.map((e) => `<button type="button" data-n="${esc(e.name)}"><b>${esc(e.name)}</b>${esc(e.description)}</button>`).join("");
    $$("#examples button").forEach((b) => b.addEventListener("click", () => load(`/api/example/${encodeURIComponent(b.dataset.n)}`, {}, b)));
  } catch (e) { console.error(e); }
  const host = location.origin;
  $("#c1").textContent = `# auto-detect from a project folder (zip it), verify a lineage.json inside it
curl -s -F files=@project.zip ${host}/api/lineage | jq '.status, (.checks[] | select(.status != "info") | .title)'
# or send a lineage JSON directly
curl -s -F files=@lineage.json ${host}/api/lineage | jq .prov > lineage.prov.json      # W3C PROV-JSON`;
  $("#c2").textContent = JSON.stringify({ project: "bracket", artifacts: [
    { id: "CAD-001", kind: "geometry", version: "A", item: "bracket CAD", file: "bracket_revA.step", hash: "<sha256>", owner: "design", timestamp: "2026-03-02T10:00:00Z" },
    { id: "MESH-014", kind: "mesh", inputs: ["CAD-001"], software: "snappyHexMesh", version: "v2312", parameters: { cells: "1.2M" } },
    { id: "SIM-382", kind: "simulation", inputs: ["MESH-014"], software: "OpenFOAM simpleFoam", version: "v2312" },
    { id: "MODEL-07", kind: "model", inputs: ["SIM-382"], software: "PINNeAPPle FNO" }] }, null, 1) +
    `\n\n# or the minimal chain\n{"project": "...", "geometry": "CAD-001", "mesh": "MESH-014", "solver": "OpenFOAM v2312", "simulation": "SIM-382", "model": "MODEL-07", "result": "PRED-883"}`;
})();
