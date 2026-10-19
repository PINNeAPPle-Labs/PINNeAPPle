"use strict";
// Physics AI Model Gateway UI: endpoints, detail with versions / validity / metrics / run log, try it, publish.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 4) => (v === null || v === undefined || !isFinite(v)) ? "—" : (Math.abs(v) !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) ? v.toExponential(2) : (+v.toPrecision(d)).toString());
let META = null, MODELS = [], CUR = null;

$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(n) { $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === n)); $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${n}`)); if (n === "metrics") ops(); }
function creds() {
  const h = {};
  try { const t = sessionStorage.getItem("mdl_tok"), k = sessionStorage.getItem("mdl_key"); if (t) h["X-Admin-Token"] = t; if (k) h["X-API-Key"] = k; } catch (e) { /* storage off */ }
  return h;
}
async function api(url, opts = {}) {
  const r = await fetch(url, { ...opts, headers: { ...(opts.headers || {}), ...creds() } });
  const j = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : j.message || JSON.stringify(j.detail || j));
  return j;
}

(async function init() {
  META = await api("/api/meta");
  $("#examples").innerHTML = META.examples.map((e, i) => `<button type="button" data-i="${i}"><b>${esc(e.title)}</b><span>${esc(e.model)}</span></button>`).join("");
  $$("#examples button").forEach((b) => b.onclick = async () => { const e = META.examples[+b.dataset.i]; await open(e.model); fill(e.inputs); $("#go").click(); });
  who(); snippets(); await list(); if (MODELS.length) open(MODELS[0].name);
})();
async function list() {
  MODELS = (await api("/v1/models")).models;
  $("#mlist").innerHTML = MODELS.map((m) => `<button data-n="${esc(m.name)}" class="${CUR === m.name ? "on" : ""}"><b>${esc(m.name)}</b><span>${esc(m.kind)} · ${esc(m.method || "")} · ${m.versions.length} version${m.versions.length > 1 ? "s" : ""} · ${esc(m.stage)} · OOD ${esc(m.ood_policy)}${m.trust ? " · trust " + esc(m.trust) : ""}</span></button>`).join("");
  $$("#mlist button").forEach((b) => b.onclick = () => open(b.dataset.n));
}
async function open(name, version) {
  CUR = name; $$("#mlist button").forEach((b) => b.classList.toggle("on", b.dataset.n === name));
  const d = await api(`/v1/models/${encodeURIComponent(name)}` + (version ? `/versions/${encodeURIComponent(version)}` : ""));
  const met = (await api(`/v1/metrics?model=${encodeURIComponent(name)}`)).by_version, runs = (await api(`/v1/models/${encodeURIComponent(name)}/runs?limit=12`)).runs;
  const mv = met.find((x) => x.version === d.version) || {}, b = d.validity_bounds || {};
  let h = `<div class="panel"><div class="toolbar"><h2 style="margin:0">${esc(name)}</h2><select id="ver" style="width:auto">${d.versions.slice().reverse().map((v) => `<option ${v === d.version ? "selected" : ""}>${esc(v)}</option>`).join("")}</select></div>
    <dl class="kv"><dt>Kind</dt><dd>${esc(d.kind)}${d.method ? " · " + esc(d.method) : ""}</dd><dt>Stage</dt><dd>${esc(d.stage)}${d.version === d.latest ? " (latest)" : ""}</dd>
    <dt>OOD policy</dt><dd><b>${esc(d.ood_policy)}</b> — ${d.ood_policy === "refuse" ? "rows outside the validity range get no prediction" : "rows outside the validity range are predicted and flagged"}</dd>
    <dt>Trust</dt><dd>${esc(d.trust || "not stated by the model")}</dd><dt>Source</dt><dd>${esc(d.source || "")}${d.description ? " — " + esc(d.description) : ""}</dd>
    <dt>Endpoint</dt><dd><code>POST /v1/models/${esc(name)}/predict</code> · <code>/versions/${esc(d.version)}/predict</code></dd>
    ${d.remote ? `<dt>Forwards to</dt><dd><code>${esc(d.remote.base)}/v1/models/${esc(d.remote.model)}</code></dd>` : ""}</dl>
    <div class="files" style="margin-top:10px"><a href="/api/models/${encodeURIComponent(name)}/model_ref.json?version=${encodeURIComponent(d.version)}" download="${esc(name)}.model_ref.json">model_ref.json</a>${d.kind === "onnx" || d.kind === "pinneapple.surrogate" ? `<a href="/v1/models/${encodeURIComponent(name)}/versions/${encodeURIComponent(d.version)}/export/onnx">model.onnx</a>` : ""}</div></div>`;
  h += `<div class="kpis" style="margin-top:14px"><div class="kpi"><div class="l">Calls</div><div class="v">${mv.calls || 0}</div><div class="s">${mv.rows || 0} rows</div></div>
    <div class="kpi"><div class="l">Errors</div><div class="v">${mv.errors || 0}</div><div class="s">rejected requests</div></div>
    <div class="kpi"><div class="l">OOD rows</div><div class="v">${mv.ood_rows || 0}</div><div class="s">${mv.refused_rows || 0} refused</div></div>
    <div class="kpi"><div class="l">Latency p50</div><div class="v">${fmt(mv.latency_ms_p50, 3)} ms</div><div class="s">server side</div></div>
    <div class="kpi"><div class="l">Latency p95</div><div class="v">${fmt(mv.latency_ms_p95, 3)} ms</div><div class="s">this version</div></div></div>`;
  h += `<div class="grid2" style="margin-top:14px"><div class="panel"><h3 style="margin-top:0">Predict</h3><div class="predform">${d.inputs.map((k) => `<label>${esc(k)} <span class="u">${b[k] ? fmt(b[k][0], 3) + " .. " + fmt(b[k][1], 3) : ""}</span><input data-k="${esc(k)}" type="number" step="any" value="${b[k] ? +((b[k][0] + b[k][1]) / 2).toPrecision(4) : 0}"></label>`).join("")}</div>
      <label>OOD policy for this call<select id="pol"><option value="">the model's (${esc(d.ood_policy)})</option><option value="refuse">refuse</option></select></label>
      <button class="primary" id="go">Predict</button><div id="pout"></div></div>
    <div class="panel"><h3 style="margin-top:0">Validity range</h3><table><thead><tr><th>input</th><th class="num">from</th><th class="num">to</th></tr></thead><tbody>${d.inputs.map((k) => `<tr><td>${esc(k)}</td><td class="num">${b[k] ? fmt(b[k][0]) : "—"}</td><td class="num">${b[k] ? fmt(b[k][1]) : "—"}</td></tr>`).join("")}</tbody></table>
      <p class="meta">Outputs: ${d.outputs.map(esc).join(", ")}.</p></div></div>`;
  h += `<div class="panel" style="margin-top:14px"><h3 style="margin-top:0">Run log</h3><table><thead><tr><th>time (UTC)</th><th>version</th><th>caller</th><th class="num">rows</th><th class="num">OOD</th><th class="num">refused</th><th class="num">latency</th><th>error</th></tr></thead><tbody>
    ${runs.map((r) => `<tr><td>${esc(r.time.replace("T", " ").slice(0, 19))}</td><td>${esc(r.version || "")}</td><td>${esc(r.caller)}</td><td class="num">${r.rows}</td><td class="num">${r.ood_rows}</td><td class="num">${r.refused}</td><td class="num">${fmt(r.latency_ms, 3)} ms</td><td>${esc(r.error || "")}</td></tr>`).join("") || `<tr><td colspan="8" class="meta">no calls yet</td></tr>`}</tbody></table></div>`;
  $("#detail").innerHTML = h;
  $("#ver").onchange = () => open(name, $("#ver").value);
  $("#go").onclick = async () => {
    const inputs = Object.fromEntries($$(".predform input").map((x) => [x.dataset.k, +x.value]));
    const body = { inputs, version: d.version }; if ($("#pol").value) body.ood_policy = $("#pol").value;
    try {
      const r = await api(`/v1/models/${encodeURIComponent(name)}/predict`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      const o = r.ood[0];
      const html = `<table style="margin-top:10px"><tbody>${Object.entries(r.outputs).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num"><b>${v[0] === null ? "refused" : Array.isArray(v[0]) ? "field (" + v[0].flat(Infinity).length + " values)" : fmt(v[0])}</b></td></tr>`).join("")}</tbody></table>
        ${o.ood ? `<div class="ood ${r.outputs[d.outputs[0]][0] === null ? "refused" : ""}"><b>${r.outputs[d.outputs[0]][0] === null ? "Refused: outside the validity range" : "Outside the validity range: an extrapolation"}</b>${o.reasons.map((x) => `<div>${esc(x)}</div>`).join("")}</div>` : `<p class="meta">Inside the validity range.</p>`}
        <p class="meta">${fmt(r.latency_ms, 3)} ms server side · request ${esc(r.request_id)} · policy ${esc(r.ood_policy)}</p>`;
      const keep = $$(".predform input").map((x) => x.value), pol = $("#pol").value;
      await open(name, d.version);                       // refresh metrics and the run log with this call
      $$(".predform input").forEach((x, i) => (x.value = keep[i])); $("#pol").value = pol; $("#pout").innerHTML = html;
    } catch (e) { $("#pout").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  };
}
function fill(inputs) { $$(".predform input").forEach((x) => { if (x.dataset.k in inputs) x.value = inputs[x.dataset.k]; }); }

async function ops() {
  const m = await api("/v1/metrics");
  $("#ops").innerHTML = `<h2 style="margin-top:0">Operations</h2><p class="meta">From the run log: every call, including refused and failed ones.</p>
    <div class="kpis"><div class="kpi"><div class="l">Calls</div><div class="v">${m.calls}</div></div><div class="kpi"><div class="l">Errors</div><div class="v">${m.errors}</div></div></div>
    <table style="margin-top:12px"><thead><tr><th>model</th><th>version</th><th class="num">calls</th><th class="num">rows</th><th class="num">errors</th><th class="num">OOD rows</th><th class="num">refused</th><th class="num">p50</th><th class="num">p95</th><th>last call</th></tr></thead><tbody>
    ${m.by_version.map((r) => `<tr><td>${esc(r.model)}</td><td>${esc(r.version)}</td><td class="num">${r.calls}</td><td class="num">${r.rows}</td><td class="num">${r.errors}</td><td class="num">${r.ood_rows}</td><td class="num">${r.refused_rows}</td><td class="num">${fmt(r.latency_ms_p50, 3)} ms</td><td class="num">${fmt(r.latency_ms_p95, 3)} ms</td><td>${esc(r.last_call.slice(0, 19).replace("T", " "))}</td></tr>`).join("")}</tbody></table>`;
}

// ------------------------------------------------------------------ publish
async function who() { try { const m = await api("/api/meta"); $("#who").textContent = `You are ${m.you.principal}${m.you.admin ? " (admin)" : ""}.`; } catch (e) { /* ignore */ } }
$("#savecred").onclick = () => { try { sessionStorage.setItem("mdl_tok", $("#tok").value); sessionStorage.setItem("mdl_key", $("#key").value); } catch (e) { /* storage off */ } who(); };
async function publish(btn, st, fn) { btn.disabled = true; $(st).textContent = "Publishing…"; try { const d = await fn(); $(st).innerHTML = `Published <b>${esc(d.name)}</b> ${esc(d.version)}.`; await list(); } catch (e) { $(st).innerHTML = `<span class="err">${esc(e.message)}</span>`; } finally { btn.disabled = false; } }
$("#pubref").onclick = (e) => publish(e.currentTarget, "#refst", () => { const ref = JSON.parse($("#ref").value); return api("/api/models/ref", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model_ref: ref, ood_policy: $("#refpol").value, stage: $("#refstage").value }) }); });
$("#pubonnx").onclick = (e) => publish(e.currentTarget, "#onst", () => { const fd = new FormData(); fd.append("file", $("#ofile").files[0]); fd.append("name", $("#oname").value); fd.append("spec", $("#ospec").value); fd.append("ood_policy", $("#opol").value); return api("/api/models/onnx", { method: "POST", body: fd }); });
$("#pubsur").onclick = (e) => publish(e.currentTarget, "#sust", () => { const fd = new FormData(); fd.append("file", $("#sfile").files[0]); fd.append("name", $("#sname").value); fd.append("ood_policy", $("#spol").value); return api("/api/models/surrogate", { method: "POST", body: fd }); });

function snippets() {
  const o = location.origin;
  $("#c1").textContent = `curl -s ${o}/v1/models
curl -s -X POST ${o}/v1/models/zoo-airfoil-cst-rans/predict -H 'content-type: application/json' \\
  -d '{"inputs": {"T0": 0.1696, "T1": 0.1450, "T2": 0.1314, "M0": 0.0249, "M1": 0.0711, "M2": 0.0590, "alpha_deg": [4, 20]}}'
curl -s ${o}/v1/metrics
curl -s ${o}/api/models/zoo-airfoil-cst-rans/model_ref.json      # the shared ModelRef contract

# publish (admin): a Surrogate Studio endpoint by its ModelRef
curl -s -X POST ${o}/api/models/ref -H 'X-Admin-Token: …' -H 'content-type: application/json' \\
  -d '{"model_ref": {"name": "airfoil", "uri": "https://surrogate.pinneapple.org/v1/models/example-airfoil-cl-cd", "inputs": []}}'`;
  $("#c2").textContent = `from pinneapple_registry.gateway_client import GatewayClient

gw = GatewayClient("${o}", api_key=None)
print([m["name"] for m in gw.models()])
r = gw.predict("zoo-airfoil-cst-rans", [{"T0": 0.1696, "T1": 0.1450, "T2": 0.1314, "M0": 0.0249,
                                         "M1": 0.0711, "M2": 0.0590, "alpha_deg": 4.0}])
print(r["outputs"]["Cl"], r["ood"][0]["ood"], r["latency_ms"])`;
}
