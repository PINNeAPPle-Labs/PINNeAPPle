// Engineering GUI Agent UI: read a task, start a run, follow its step timeline, answer confirmations.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
let WORKFLOWS = [], META = null, RUN = null, TIMER = null;
const KIND = { screen: "screenshot of the virtual display", window: "window / 3D view", render: "exported by the application", plot: "chart from the log" };

async function api(url, opts = {}) {
  const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error(body?.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : r.statusText);
  return body;
}

// ------------------------------------------------------------------------------------------------ tabs
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(t) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + t));
  if (t === "runs") loadRuns();
}

// ------------------------------------------------------------------------------------------- workflows
function currentWf() { return WORKFLOWS.find((w) => w.id === $("wf").value); }
function renderInputs(values = {}) {
  const wf = currentWf();
  $("inputs").innerHTML = Object.entries(wf.inputs).map(([k, s]) => `
    <label for="in-${k}">${esc(s.description || k)} <span class="u">${esc(s.unit)} · ${s.min ?? ""}–${s.max ?? ""}</span></label>
    <input id="in-${k}" data-k="${k}" type="number" step="any" min="${s.min ?? ""}" max="${s.max ?? ""}" value="${values[k] ?? s.default}">`).join("");
  $("pipeline").innerHTML = wf.plan.map((p, i) => `<div><span class="mode">${i + 1} · ${esc(p.mode)}</span><b>${esc(p.title)}</b>
    <code>${esc(p.action)}</code>${p.confirm.length ? `<br><span class="mode">asks: ${esc(p.confirm.join(", "))}</span>` : ""}
    ${p.available ? "" : `<br><span class="miss">${esc(p.reason)}</span>`}</div>`).join("");
}
function inputValues() {
  const out = {};
  document.querySelectorAll("#inputs input").forEach((i) => { out[i.dataset.k] = Number(i.value); });
  return out;
}

$("read").addEventListener("click", async () => {
  $("interp").innerHTML = "Reading…";
  try {
    const it = await api("/api/interpret", { method: "POST", body: JSON.stringify({ task: $("task").value }) });
    $("wf").value = it.workflow;
    renderInputs(it.resolved_inputs);
    $("interp").innerHTML = `<b>${esc(currentWf().title)}</b> · read by ${esc(it.method === "llm" ? "the local LLM" : it.method === "rules" ? "rules (no LLM)" : "structure")}
      <div class="note">${it.notes.map(esc).join("<br>")}</div>`;
  } catch (e) { $("interp").innerHTML = `<span class="miss">${esc(e.message)}</span>`; }
});

$("start").addEventListener("click", async () => {
  $("start").disabled = true;
  $("status").textContent = "Starting…";
  try {
    const r = await api("/api/runs", { method: "POST", body: JSON.stringify({ workflow: $("wf").value, inputs: inputValues() }) });
    $("status").textContent = "";
    openRun(r.run_id);
  } catch (e) { $("status").textContent = e.message; }
  $("start").disabled = false;
});

// ------------------------------------------------------------------------------------------------ runs
function openRun(id) {
  showTab("run");
  history.replaceState(null, "", "#run=" + id);
  clearTimeout(TIMER);
  poll(id);
}
async function poll(id) {
  try {
    RUN = await api("/api/runs/" + id);
    renderRun(RUN);
    if (RUN.active || RUN.state === "running" || RUN.state === "pending") TIMER = setTimeout(() => poll(id), 1500);
  } catch (e) { $("view").innerHTML = `<div class="panel"><p class="miss">${esc(e.message)}</p></div>`; }
}

function shots(r, s) {
  if (!s.evidence.length) return "";
  return `<div class="shots">${s.evidence.map((e) => {
    const url = `/api/runs/${r.run_id}/files/${e.path}`;
    return `<figure data-src="${url}" data-cap="${esc(e.caption)} (${esc(KIND[e.kind] || e.kind)})"><img loading="lazy" src="${url}" alt="${esc(e.caption)}">
      <figcaption><span class="kind">${esc(e.kind)}</span> ${esc(e.caption)}</figcaption></figure>`;
  }).join("")}</div>`;
}

function stepCard(r, s, i) {
  const dur = s.duration_s != null ? `${s.duration_s.toFixed(1)} s` : "";
  const cur = r.current_step === s.id && r.active;
  const state = cur ? "running" : s.state;
  const checks = s.checks.length ? `<ul class="checks-list">${s.checks.map((c) =>
    `<li class="${c.passed ? "ok" : "ko"}">${esc(c.name)} <span class="d">— ${esc(c.detail)}</span></li>`).join("")}</ul>` : "";
  const acts = s.actions.length ? `<ul class="acts">${s.actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>` : "";
  const files = s.artifacts.length ? `<details class="more"><summary>${s.artifacts.length} file(s) produced</summary><pre>${s.artifacts.map((a) =>
    `${a.path}${a.kind === "dir" ? "/" : ""}  ${a.bytes.toLocaleString()} B  sha256 ${a.sha256.slice(0, 16)}…`).map(esc).join("\n")}</pre></details>` : "";
  const cmds = s.commands.length ? `<details class="more"><summary>commands and logs</summary><pre>${esc(s.commands.join("\n"))}</pre>
    ${s.logs.map((l) => `<a class="link" target="_blank" href="/api/runs/${r.run_id}/files/${esc(l)}">${esc(l)}</a>`).join(" · ")}</details>` : "";
  const err = s.error && s.state !== "awaiting_confirmation" ? `<div class="errbox">${esc(s.error)}</div>` : "";
  return `<div class="step"><div class="dot ${esc(state)}">${state === "passed" ? "✓" : state === "failed" ? "✗" : i + 1}</div>
    <div class="card${cur ? " current" : ""}"><div class="hd"><b>${esc(s.title)}</b><code>${esc(s.action)}</code>
      <span class="mode-pill ${esc(s.mode)}">${esc(s.mode === "gui" ? "GUI" : "API")}</span>
      <span class="state st-${esc(state)}">${esc(state.replaceAll("_", " "))}</span><span class="dur">${dur}</span></div>
      ${acts}${checks}${shots(r, s)}${err}${files}${cmds}</div></div>`;
}

function gate(r) {
  const s = r.steps.find((x) => x.state === "awaiting_confirmation");
  if (!s || r.active) return "";
  return `<div class="gate"><h3>“${esc(s.title)}” needs your confirmation</h3>
    <ul>${(s.confirmation?.reasons || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
    <div class="row"><label>Your name (kept in the evidence)<input id="who" maxlength="80" placeholder="e.g. J. Doe"></label>
      <button class="primary" onclick="decide('approve','${esc(s.id)}')">Approve and continue</button>
      <button class="danger" onclick="decide('decline','${esc(s.id)}')">Decline</button></div></div>`;
}
async function decide(kind, step) {
  try {
    await api(`/api/runs/${RUN.run_id}/${kind}`, { method: "POST", body: JSON.stringify({ step, by: $("who")?.value || "" }) });
    poll(RUN.run_id);
  } catch (e) { alert(e.message); }
}

function criteria(r) {
  if (!r.criteria.length) return "";
  return `<h3 style="margin:16px 0 0;font-size:15px">Acceptance criteria</h3><table class="crit">${r.criteria.map((c) =>
    `<tr><td class="${c.passed ? "met" : "notmet"}">${c.passed ? "MET" : "NOT MET"}</td><td>${esc(c.name)}</td><td>${esc(c.detail)}</td></tr>`).join("")}</table>`;
}

function renderRun(r) {
  const done = !r.active && !["running", "pending"].includes(r.state);
  const links = done ? `<span class="links"><a target="_blank" href="/api/runs/${r.run_id}/report">Evidence report</a>
    <a target="_blank" href="/api/runs/${r.run_id}/study_result">StudyResult</a><a href="/api/runs/${r.run_id}/bundle">Download all (zip)</a></span>` : "";
  const inputs = Object.entries(r.inputs).map(([k, v]) => `${esc(k)} = ${esc(v)} ${esc(r.workflow.inputs?.[k]?.unit || "")}`).join(" · ");
  const queued = r.job && r.job.status === "queued";
  const msg = queued ? "Waiting in the queue: this server runs one task at a time"
    : r.active ? `Working on “${esc((r.steps.find((s) => s.id === r.current_step) || {}).title || "…")}”` : esc(r.message);
  const scope = `<details class="limits"><summary>Limits of this workflow (${(r.workflow.limits || []).length})</summary>
    <ul>${(r.workflow.limits || []).map((l) => `<li>${esc(l)}</li>`).join("")}</ul>
    <p>Every number, image and log on this page comes from this run. A step is passed only if it ran and every check
      on it held. Meeting the acceptance criteria written in the workflow is not a validation of the model.</p></details>`;
  $("view").innerHTML = `<div class="panel">
    <div class="runhead"><div><h2>${esc(r.workflow.title)}</h2><div class="meta">run ${esc(r.run_id)} · ${inputs}</div></div>
      <div><span class="state st-${esc(r.active ? "running" : r.state)}">${esc(queued ? "queued" : (r.active ? "running" : r.state).replaceAll("_", " "))}</span>${links}</div></div>
    <div class="verdict ${esc(r.active ? "running" : r.state)}">${msg}</div>
    ${gate(r)}
    <div class="timeline">${r.steps.map((s, i) => stepCard(r, s, i)).join("")}</div>
    ${criteria(r)}${scope}</div>`;
  document.querySelectorAll(".shots figure").forEach((f) => f.addEventListener("click", () => {
    $("lb-img").src = f.dataset.src; $("lb-cap").textContent = f.dataset.cap; $("lightbox").showModal();
  }));
}

async function loadRuns() {
  const runs = await api("/api/runs");
  $("runs").innerHTML = runs.length ? `<table class="runs"><tr><th>Run</th><th>Workflow</th><th>Inputs</th><th>State</th><th></th></tr>${runs.map((r) =>
    `<tr><td>${esc(r.run_id)}</td><td>${esc(r.title)}</td><td>${Object.entries(r.inputs).map(([k, v]) => `${esc(k)}=${esc(v)}`).join(", ")}</td>
     <td><span class="state st-${esc(r.active ? "running" : r.state)}">${esc((r.active ? "running" : r.state).replaceAll("_", " "))}</span></td>
     <td><a class="link" href="#run=${esc(r.run_id)}" onclick="openRun('${esc(r.run_id)}');return false">open</a></td></tr>`).join("")}</table>`
    : "<p>No runs yet on this server.</p>";
}

// ------------------------------------------------------------------------------------------------ init
(async function init() {
  [META, WORKFLOWS] = await Promise.all([api("/api/meta"), api("/api/workflows")]);
  $("wf").innerHTML = WORKFLOWS.map((w) => `<option value="${esc(w.id)}">${esc(w.title)}</option>`).join("");
  $("wf").addEventListener("change", () => renderInputs());
  renderInputs();
  $("tools").innerHTML = Object.entries(META.tools).map(([k, v]) => `<span>${esc(k)}</span><span class="${v ? "" : "miss"}">${esc(v || "not installed")}</span>`).join("");
  if (META.mode === "selfhosted") $("mode-note").textContent = "Self-hosted mode: POST /api/workflows adds your own workflows (zip with workflow.yaml, project and macros).";
  const origin = location.origin;
  $("c1").textContent = `# start a run of the committed workflow (stops before the solver step)
curl -s -X POST ${origin}/api/runs -H 'Content-Type: application/json' \\
  -d '{"workflow":"channel_obstacle","inputs":{"obstacle_height_mm":55,"inlet_velocity_mm_s":20}}'
# follow it, approve the solver step, fetch the evidence
curl -s ${origin}/api/runs/<run_id>
curl -s -X POST ${origin}/api/runs/<run_id>/approve -H 'Content-Type: application/json' -d '{"step":"solve","by":"me"}'
curl -s ${origin}/api/runs/<run_id>/study_result
curl -s -o run.zip ${origin}/api/runs/<run_id>/bundle`;
  $("c2").textContent = `from pinneapple_tools.automation import Runner, Workflow, example_workflow

wf = Workflow.load(example_workflow("channel_obstacle"))
run = Runner(wf, "runs/r1", inputs={"obstacle_height_mm": 55}).run()          # stops before "solve"
run = Runner(wf, "runs/r1").run(approve=["solve"], approver="me")
print(run.state, run.message)                                                  # report.html, study_result.json`;
  const m = location.hash.match(/run=([\w-]+)/);
  if (m) openRun(m[1]);
})();
