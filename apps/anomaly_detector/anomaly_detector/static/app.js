"use strict";
// Physics Anomaly Detector UI: upload, configure, events with explanations, residual channels, benchmark.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 4) => (v === null || v === undefined || !isFinite(v) ? "—" : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(d) + "");
const pct = (v) => (v === null || v === undefined ? "—" : (100 * v).toFixed(1) + " %");
let META = null, UPLOAD = null, RES = null, JOB = null;

$$("#tabs button").forEach((b) => b.addEventListener("click", () => {
  $$("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${b.dataset.tab}`));
}));
async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail || j));
  return j;
}
async function waitJob(job, line) {
  for (;;) {
    const j = await api(`/api/jobs/${job.id}`);
    line.textContent = `${j.status} · ${Math.round(100 * (j.progress || 0))} % ${j.message ? "· " + j.message : ""}`;
    if (j.status === "done") return j;
    if (j.status === "failed" || j.status === "cancelled") throw new Error(j.error || j.status);
    await new Promise((r) => setTimeout(r, 400));
  }
}

// ---------------------------------------------------------------- input
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
$("#file").addEventListener("change", (e) => upload(e.target.files[0]));
async function upload(f) {
  if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  $("#fileinfo").textContent = `uploading ${f.name}…`;
  try {
    const r = await fetch("/api/uploads", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail);
    UPLOAD = j;
    const c = await api(`/api/columns/${j.id}`);
    $("#fileinfo").innerHTML = `${esc(f.name)} · ${c.rows.toLocaleString()} rows · columns: ${esc(c.columns.join(", "))}` +
      (c.t_range ? ` · t from ${fmt(c.t_range[0])} to ${fmt(c.t_range[1])} s` : ` · <span class="err">no column t</span>`);
    $("#sensors").value = c.numeric.filter((x) => x !== "t").join(", ");
    if (c.t_range) $("#normal").value = Math.round(c.t_range[0] + 0.25 * (c.t_range[1] - c.t_range[0]));
    $("#label").value = c.columns.includes("fault") ? "fault" : "";
    $("#run").disabled = false;
  } catch (e) { $("#fileinfo").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}
function config() {
  let extra;
  try { extra = JSON.parse($("#balances").value || "{}"); } catch (e) { throw new Error("balances: not valid JSON"); }
  const sensors = $("#sensors").value.split(",").map((s) => s.trim()).filter(Boolean);
  const methods = $$("#methods input:checked").map((i) => i.value);
  const model = $("#model").value;
  return { normal_until: +$("#normal").value, sensors: sensors.length ? sensors : null, methods, window: +$("#window").value,
    threshold: +$("#threshold").value, label_column: $("#label").value.trim() || null, balances: extra.balances || [],
    constants: extra.constants || {}, model, model_states: model === "none" ? [] : ["h", "T"], model_inputs: model === "none" ? [] : ["q_in", "Q", "T_in", "T_amb"] };
}
async function analyze(body) {
  const line = $("#status");
  $("#run").disabled = $("#example").disabled = true;
  try {
    const job = await api("/api/analyze", { method: "POST", body: JSON.stringify(body) });
    const j = await waitJob(job, line);
    JOB = j; RES = j.result;
    line.textContent = `done: ${RES.events.length} events on ${RES.n.toLocaleString()} samples`;
    render();
  } catch (e) { line.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  $("#run").disabled = !UPLOAD; $("#example").disabled = false;
}
$("#run").addEventListener("click", () => { try { analyze({ upload_id: UPLOAD.id, ...config() }); } catch (e) { $("#status").innerHTML = `<span class="err">${esc(e.message)}</span>`; } });
$("#example").addEventListener("click", () => {
  const p = META.preset;
  $("#normal").value = p.normal_until; $("#label").value = p.label_column; $("#sensors").value = p.sensors.join(", ");
  $("#balances").value = JSON.stringify({ balances: p.balances, constants: p.constants }, null, 1);
  analyze({ example: true });
});

// ---------------------------------------------------------------- results
function render() {
  const R = RES, sc = R.score;
  const k = (l, v, s) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s || ""}</div></div>`;
  let kp = k("Events", R.events.length, `${R.events.filter((e) => e.severity === "critical" || e.severity === "high").length} high or critical`) +
    k("Flagged samples", pct(R.flagged_fraction), `after ${R.normal_samples} normal samples`);
  if (sc) kp += k("Faults found", `${sc.per_fault.filter((p) => p.detected).length} / ${sc.per_fault.length}`, "labelled intervals with an event") +
    k("Point precision", pct(sc.point_precision), `recall ${pct(sc.point_recall)}`) + k("False alarms", `${fmt(sc.false_alarms_per_hour, 2)} /h`, `${sc.false_alarm_events} events in ${fmt(sc.evaluated_hours, 3)} h`) +
    k("Mean delay", sc.mean_delay_s === null ? "—" : `${fmt(sc.mean_delay_s / 60, 3)} min`, "fault start → first event");
  const sigs = Object.keys(R.signals);
  const physics = R.channels.filter((c) => c.method === "balance" || c.method === "model");
  $("#out").innerHTML = `<div class="panel"><div class="kpis">${kp}</div>
    <div class="charts2">${sigs.slice(0, 4).map((s, i) => `<div class="chartbox"><div class="cap">${esc(s)} <span class="meta">· red = flagged</span></div><div id="sig${i}"></div>${R.truth ? `<div class="truthbar" id="tb${i}"></div>` : ""}</div>`).join("")}</div>
    ${R.truth ? `<div class="meta">grey bar under each chart: labelled faults (${R.truth.length})</div>` : ""}
    <h3>Events</h3>
    ${R.events.length ? `<table class="ev"><tr><th class="num">start [min]</th><th class="num">duration</th><th>severity</th><th>methods</th><th>first explanation</th></tr>
      ${R.events.map((e, i) => `<tr data-i="${i}"><td class="num">${fmt(e.start / 60)}</td><td class="num">${fmt(e.duration / 60, 3)} min</td><td><span class="pill ${e.severity}">${e.severity}</span></td><td>${esc(e.methods.join(", "))}</td><td>${esc(e.explanations[0] || "")}</td></tr>`).join("")}</table>` : `<p class="hint">No event.</p>`}
    <div id="evdetail"></div>
    <h3>Physics residuals</h3><p class="hint">Score = (residual − normal median) / normal spread; dashed lines at ± threshold.</p>
    <div class="charts2">${physics.map((c, i) => `<div class="chartbox"><div class="cap">${esc(c.name)} <span class="meta">${esc(c.units)} · ${c.n_flagged} samples over the threshold</span></div><div id="res${i}"></div></div>`).join("")}</div>
    ${sc ? `<h3>Per labelled fault</h3><table><tr><th>fault</th><th class="num">start [min]</th><th>detected</th><th class="num">delay [min]</th><th>by</th></tr>${sc.per_fault.map((p) => `<tr><td>${esc(p.kind)}${p.target ? ` (${esc(p.target)})` : ""}</td><td class="num">${fmt(p.start / 60)}</td><td>${p.detected ? "✓" : "✗"}</td><td class="num">${p.delay_s === null ? "—" : fmt(p.delay_s / 60, 3)}</td><td>${esc(p.methods.join(", "))}</td></tr>`).join("")}</table>` : ""}
    <div class="toolbar" style="margin-top:12px"><span class="meta">Downloads</span><span>
      <a class="link" href="/api/jobs/${JOB.id}/files/events.csv">events (CSV)</a> · <a class="link" href="/api/jobs/${JOB.id}/files/anomaly.study.json">StudyResult (JSON)</a></span></div></div>`;
  const x = R.t.map((t) => +(t / 60).toFixed(1));
  sigs.slice(0, 4).forEach((s, i) => {
    PChart.mount($(`#sig${i}`), { x, series: [{ name: s, y: R.signals[s], color: "#334155", width: 1.1 }], flags: R.flags, height: 150 });
    if (R.truth) {
      const t0 = R.t[0], t1 = R.t[R.t.length - 1];
      $(`#tb${i}`).innerHTML = R.truth.map((iv) => `<div title="${esc(iv.kind)}" style="left:${(100 * (iv.start - t0)) / (t1 - t0)}%;width:${(100 * (iv.end - iv.start)) / (t1 - t0)}%">${esc(iv.kind)}</div>`).join("");
    }
  });
  physics.forEach((c, i) => PChart.mount($(`#res${i}`), { x, series: [{ name: "score", y: c.score, color: c.method === "balance" ? "#b45309" : "#7c3aed", width: 1.1 }], height: 150,
    hlines: [{ y: c.threshold, color: "#b42318", label: `+${c.threshold}` }, { y: -c.threshold, color: "#b42318", label: `−${c.threshold}` }] }));
  $$("table.ev tr[data-i]").forEach((tr) => tr.addEventListener("click", () => {
    $$("table.ev tr").forEach((x) => x.classList.toggle("sel", x === tr));
    const e = R.events[+tr.dataset.i];
    $("#evdetail").innerHTML = `<div class="evdetail"><b>Event ${+tr.dataset.i + 1}</b> · ${fmt(e.start / 60)}–${fmt(e.end / 60)} min · peak ${fmt(e.peak_ratio, 3)} × threshold · variables: ${esc(e.variables.join(", "))}
      <ul>${e.explanations.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`;
  }));
}

// ---------------------------------------------------------------- benchmark
$("#brun").addEventListener("click", async () => {
  const line = $("#bstatus");
  $("#brun").disabled = true;
  try {
    const job = await api("/api/benchmark", { method: "POST", body: "{}" });
    const j = await waitJob(job, line);
    const r = j.result, rows = Object.entries(r.results);
    const kinds = Object.keys(rows[0][1].per_fault_kind);
    line.textContent = `done: seeds ${r.seeds.join(", ")}, window ${r.window} samples, threshold ${r.threshold}`;
    const cls = (v) => { const [a, b] = v.split("/").map(Number); return a === b ? "ok" : a === 0 ? "no" : "part"; };
    $("#bench-out").innerHTML = `<table style="margin-top:12px"><tr><th>detector</th><th class="num">point precision</th><th class="num">point recall</th><th class="num">F1</th><th class="num">faults found</th><th class="num">false alarms /h</th><th class="num">mean delay [min]</th></tr>
      ${rows.map(([k, v]) => `<tr><td><b>${esc(k)}</b></td><td class="num">${pct(v.point_precision)}</td><td class="num">${pct(v.point_recall)}</td><td class="num">${fmt(v.point_f1, 3)}</td><td class="num">${pct(v.event_recall)}</td><td class="num">${fmt(v.false_alarms_per_hour, 2)}</td><td class="num">${v.mean_delay_s === null ? "—" : fmt(v.mean_delay_s / 60, 3)}</td></tr>`).join("")}</table>
      <h3>Which detector finds which fault (detected / runs)</h3>
      <table class="mat"><tr><th>detector</th>${kinds.map((k) => `<th class="num">${esc(k)}</th>`).join("")}</tr>
      ${rows.map(([k, v]) => `<tr><td><b>${esc(k)}</b></td>${kinds.map((q) => `<td class="num ${cls(v.per_fault_kind[q])}">${v.per_fault_kind[q]}</td>`).join("")}</tr>`).join("")}</table>
      <p class="hint" style="margin-top:10px">${esc(r.protocol)}. Point recall counts samples inside the true fault intervals; windowed detectors
      lag the fault start, so event recall (a fault with at least one overlapping event, 10 min grace) is the fairer figure. Balances do not see the
      drain-valve fault because the outflow is measured and the physics still holds: only the model residual catches it.</p>
      <div class="meta"><a class="link" href="/api/jobs/${j.id}/files/benchmark.json">benchmark.json</a></div>`;
  } catch (e) { line.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  $("#brun").disabled = false;
});

// ---------------------------------------------------------------- API + meta
const origin = location.origin;
$("#c1").textContent = `ID=$(curl -s -F file=@history.csv ${origin}/api/uploads | jq -r .id)
JOB=$(curl -s -X POST ${origin}/api/analyze -H 'Content-Type: application/json' -d '{
  "upload_id": "'$ID'", "normal_until": 7200, "sensors": ["h", "T", "q_in", "q_out", "Q"],
  "balances": [{"name": "mass", "accumulation": "A * h", "inflow": "q_in", "outflow": "q_out", "units": "m3/s"}],
  "constants": {"A": 1.5}, "label_column": "fault"}' | jq -r .id)
curl ${origin}/api/jobs/$JOB                       # status, then result.events and result.score
curl -O ${origin}/api/jobs/$JOB/files/anomaly.study.json`;
$("#c2").textContent = `from pinneapple_data.contracts import read_table
from pinneapple_systems.operations.anomaly import DetectorConfig, detect, tank_balance_rules, tank_constants

df, _ = read_table("tank_six_faults.csv")
cfg = DetectorConfig(normal_until=7200, sensors=["h", "T", "q_in", "q_out", "Q"], methods=("balance",),
                     balances=tank_balance_rules(), constants=tank_constants())
for e in detect({c: df[c].to_numpy() for c in df.columns if c != "fault"}, cfg).events:
    print(e.start, e.severity, e.explanations[0])`;
api("/api/meta").then((m) => {
  META = m; $("#scope").innerHTML = renderScope(m.scope);
  $("#balances").value = JSON.stringify({ balances: m.preset.balances, constants: m.preset.constants }, null, 1);
}).catch(() => {});
