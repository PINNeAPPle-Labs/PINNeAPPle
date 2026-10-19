"use strict";
// Physics-Based Control Lab UI: closed-loop comparison, MPC plans, recommendation mode, API examples.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 4) => (v === null || v === undefined || !isFinite(v) ? "—" : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(d) + "");
const COLORS = { pid: "#b45309", rules: "#cbd5e1", lmpc: "#0f766e", nmpc: "#4338ca", smpc: "#be185d" };
let META = null;

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
const v = (id) => +$(`#${id}`).value;

function study() {
  const spec = {
    setpoints: [[v("sp1t"), v("sp1h"), v("sp1T")], [v("sp2t"), v("sp2h"), v("sp2T")], [v("sp3t"), v("sp3h"), v("sp3T")]],
    h_bounds: [v("hmin"), v("hmax")], T_max: v("tmax"), q_in_bounds: [0, v("qmax") / 1000], Q_bounds: [0, v("Qmax") * 1000],
    dq_max: v("dq") / 1000, dQ_max: v("dQ") * 1000, horizon: v("hor"), w_energy: v("we"), w_move: v("wm"),
  };
  const faults = v("dem") ? [{ kind: "valve", start: v("dem_t"), end: null, magnitude: -v("dem") / 100, target: "" }] : [];
  const scenario = { duration: v("dur"), T_in_steps: [[0, 15], [v("tin_t"), v("tin")]], faults };
  return { spec, scenario, controllers: $$("#ctrls input:checked").map((i) => i.value) };
}

async function run(body) {
  const line = $("#status");
  $("#run").disabled = $("#example").disabled = true;
  try {
    const j = await waitJob(await api("/api/compare", { method: "POST", body: JSON.stringify(body) }), line);
    line.textContent = "done";
    render(j);
  } catch (e) { line.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  $("#run").disabled = $("#example").disabled = false;
}
$("#run").addEventListener("click", () => run(study()));
$("#example").addEventListener("click", () => run({ controllers: ["pid", "rules", "lmpc", "nmpc"] }));

function render(job) {
  const R = job.result, runs = Object.entries(R.runs), spec = R.spec;
  const keys = [["iae_T_K_s", "IAE T [K·s]"], ["iae_h_m_s", "IAE h [m·s]"], ["max_violation_T_K", "max T violation [K]"], ["max_violation_h_m", "max h violation [m]"],
    ["heater_energy_kWh", "heater energy [kWh]"], ["heater_travel_kW", "heater travel [kW]"], ["pump_travel_L_s", "pump travel [L/s]"], ["mean_solve_ms", "solve [ms/step]"]];
  const best = {};
  for (const [k] of keys) best[k] = Math.min(...runs.map(([, r]) => r.metrics[k]));
  const table = `<table><tr><th>controller</th>${keys.map(([, l]) => `<th class="num">${l}</th>`).join("")}<th>failed solves</th></tr>
    ${runs.map(([n, r]) => `<tr><td style="white-space:nowrap"><i class="sw" style="background:${COLORS[n]}"></i> <b>${esc(r.controller)}</b><div class="meta">${esc(r.backend || "")}</div></td>
    ${keys.map(([k]) => `<td class="num ${r.metrics[k] === best[k] && !k.startsWith("mean_solve") ? "best" : ""}">${fmt(r.metrics[k])}</td>`).join("")}<td class="num">${r.failed_solves ?? 0}</td></tr>`).join("")}</table>`;
  const checks = R.checks.map((c) => `<tr><td>${esc(c.name)}</td><td><span class="badge ${c.status === "PASS" ? "pass" : c.status === "WARN" ? "warn" : "fail"}">${esc(c.status)}</span></td><td class="num">${fmt(c.value)}</td></tr>`).join("");
  const mpcs = runs.filter(([, r]) => r.plans && r.plans.length);
  $("#out").innerHTML = `<div class="panel">
    <div class="banner ok" style="font-weight:600">${esc(R.mode)}. Same plant, noise and disturbances for every controller; numbers below are from this simulation.</div>
    <h3 style="margin-top:0">Performance (lower is better; green = best)</h3><div class="tablewrap">${table}</div>
    <div class="charts2">
      <div class="chartbox"><div class="cap">Temperature T [°C] and its reference</div><div id="cT"></div></div>
      <div class="chartbox"><div class="cap">Level h [m] and its reference</div><div id="ch"></div></div>
      <div class="chartbox"><div class="cap">Heater power Q [kW]</div><div id="cQ"></div></div>
      <div class="chartbox"><div class="cap">Pump flow q_in [L/s]</div><div id="cq"></div></div>
    </div>
    ${mpcs.length ? `<h3>What the MPC planned</h3><div class="planbar"><select id="pctrl" style="width:auto">${mpcs.map(([n, r]) => `<option value="${n}">${esc(r.controller)}</option>`).join("")}</select>
      <span>plan made at</span><input type="range" id="pidx" min="0" value="6"><b id="ptime"></b></div><div class="charts2"><div class="chartbox"><div class="cap">Predicted T over the horizon vs what happened</div><div id="pT"></div></div><div class="chartbox"><div class="cap">Planned heater power [kW]</div><div id="pQ"></div></div></div>` : ""}
    <h3>Checks recorded in the StudyResult</h3><table><tr><th>check</th><th>status</th><th class="num">value</th></tr>${checks}</table>
    <div class="toolbar" style="margin-top:10px"><span class="meta">Downloads</span><span><a class="link" href="/api/jobs/${job.id}/files/control.study.json">StudyResult (JSON)</a> ·
      <a class="link" href="/api/jobs/${job.id}/files/trajectories.csv">trajectories (CSV)</a></span></div>
    <p class="hint">Disturbances: demand ${esc(JSON.stringify(R.scenario.faults.map((f) => `${-100 * f.magnitude}% from ${f.start}s`)))}, inlet water ${esc(JSON.stringify(R.scenario.T_in_steps))}.
      The MPCs read the state from an ensemble Kalman filter (the twin runtime) that also estimates the unaccounted outflow and the heater efficiency.</p></div>`;
  const tr0 = runs[0][1].trajectory, x = tr0.t.map((t) => +(t / 60).toFixed(1));
  const line = (key, scale = 1) => runs.map(([n, r]) => ({ name: r.controller, y: r.trajectory[key].map((q) => q * scale), color: COLORS[n], width: 1.4 }));
  PChart.mount($("#cT"), { x, series: [...line("T"), { name: "reference", y: tr0.T_ref, color: "#111827", dash: "4 3", width: 1 }], height: 210, hlines: [{ y: spec.T_max, color: "#b42318", label: `T max ${spec.T_max}` }] });
  PChart.mount($("#ch"), { x, series: [...line("h"), { name: "reference", y: tr0.h_ref, color: "#111827", dash: "4 3", width: 1 }], height: 210, hlines: spec.h_bounds.map((b) => ({ y: b, color: "#b42318", label: `limit ${b}` })) });
  PChart.mount($("#cQ"), { x, series: line("Q", 1e-3), height: 170, hlines: [{ y: spec.Q_bounds[1] / 1000, color: "#b42318", label: "heater max" }] });
  PChart.mount($("#cq"), { x, series: line("q_in", 1e3), height: 170 });
  if (mpcs.length) {
    const draw = () => {
      const n = $("#pctrl").value, r = R.runs[n], plans = r.plans, i = Math.min(+$("#pidx").value, plans.length - 1), p = plans[i], tr = r.trajectory;
      $("#pidx").max = plans.length - 1; $("#ptime").textContent = `t = ${fmt(p.t / 60)} min`;
      const xs = p.plan_t.map((t) => +(t / 60).toFixed(1));
      const actual = p.plan_t.map((t) => { const k = tr.t.findIndex((q) => q >= t - 1e-6); return k < 0 ? null : tr.T[k]; });
      const ref = p.plan_t.map((t) => { const k = tr.t.findIndex((q) => q >= t - 1e-6); return k < 0 ? null : tr.T_ref[k]; });
      PChart.mount($("#pT"), { x: xs, series: [{ name: "predicted", y: p.pred_T, color: COLORS[n] }, { name: "simulated plant", y: actual, color: "#111827", width: 1 }, { name: "reference", y: ref, color: "#94a3b8", dash: "4 3" }], height: 170 });
      PChart.mount($("#pQ"), { x: xs, series: [{ name: "planned Q", y: p.plan_Q.map((q) => q / 1000), color: COLORS[n] }], height: 170 });
    };
    $("#pctrl").addEventListener("change", draw); $("#pidx").addEventListener("input", draw); draw();
  }
}

// ---------------------------------------------------------------- recommend
$("#rec").addEventListener("click", async () => {
  const line = $("#rstatus");
  try {
    let estimate = { h: v("r_h"), T: v("r_T"), q_leak: v("r_leak") / 1000, eta: v("r_eta") / 100 };
    if ($("#r_json").value.trim()) estimate = JSON.parse($("#r_json").value);
    const body = { estimate, reading: { q_in: v("r_q") / 1000, Q: v("r_Q") * 1000, T_in: v("r_tin"), T_amb: v("r_tamb") }, t: v("r_t"), spec: study().spec };
    const r = await api("/api/recommend", { method: "POST", body: JSON.stringify(body) });
    line.textContent = `${r.controller} · solved in ${fmt(r.solve_ms, 3)} ms${r.solver_ok ? "" : " · solver did not converge: review with care"}`;
    const k = (l, val, s) => `<div class="kpi"><div class="l">${l}</div><div class="v">${val}</div><div class="s">${s || ""}</div></div>`;
    const x = r.plan.plan_t.map((t) => +(t / 60).toFixed(1));
    $("#rout").innerHTML = `<div class="panel"><div class="banner ok">${esc(r.mode)}: review before acting.</div>
      <div class="kpis">${k("Proposed pump flow", `${fmt(1000 * r.proposed_next.q_in, 3)} L/s`, "next step")}${k("Proposed heater power", `${fmt(r.proposed_next.Q / 1000, 3)} kW`, "next step")}
      ${k("Predicted max T", `${fmt(r.predicted.max_T)} °C`, `limit ${r.constraints.T_max}`)}${k("Predicted h range", `${fmt(r.predicted.min_h, 3)}–${fmt(r.predicted.max_h, 3)} m`, `limits ${r.constraints.h_bounds.join("–")}`)}
      ${k("Heater energy", `${fmt(r.predicted.heater_energy_kWh, 3)} kWh`, "over the horizon")}</div>
      <div class="charts2"><div class="chartbox"><div class="cap">Predicted temperature [°C]</div><div id="rT"></div></div><div class="chartbox"><div class="cap">Predicted level [m]</div><div id="rh"></div></div>
      <div class="chartbox"><div class="cap">Planned heater power [kW]</div><div id="rQ"></div></div><div class="chartbox"><div class="cap">Planned pump flow [L/s]</div><div id="rq"></div></div></div></div>`;
    PChart.mount($("#rT"), { x, series: [{ name: "predicted", y: r.plan.pred_T, color: "#4338ca" }, { name: "reference", y: r.reference.T_ref, color: "#111827", dash: "4 3" }], height: 170, hlines: [{ y: r.constraints.T_max, color: "#b42318", label: "T max" }] });
    PChart.mount($("#rh"), { x, series: [{ name: "predicted", y: r.plan.pred_h, color: "#4338ca" }, { name: "reference", y: r.reference.h_ref, color: "#111827", dash: "4 3" }], height: 170 });
    PChart.mount($("#rQ"), { x, series: [{ name: "Q", y: r.plan.plan_Q.map((q) => q / 1000), color: "#4338ca" }], height: 150 });
    PChart.mount($("#rq"), { x, series: [{ name: "q_in", y: r.plan.plan_q_in.map((q) => q * 1000), color: "#4338ca" }], height: 150 });
  } catch (e) { line.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
});

// ---------------------------------------------------------------- API + meta
const origin = location.origin;
$("#c1").textContent = `# closed-loop comparison (job)
curl -X POST ${origin}/api/compare -H 'Content-Type: application/json' \\
  -d '{"controllers": ["pid", "lmpc", "nmpc"], "spec": {"T_max": 31, "setpoints": [[0, 1.2, 25], [600, 1.2, 29]]},
       "scenario": {"duration": 3600, "T_in_steps": [[0, 15], [2400, 11]]}}'
# recommendation from the Digital Twin Runtime's state
STATE=$(curl -s https://twin.pinneapple.org/api/assets/tank-1/state)
curl -X POST ${origin}/api/recommend -H 'Content-Type: application/json' \\
  -d '{"estimate": '"$STATE"', "reading": {"q_in": 0.0066, "Q": 300000, "T_in": 15, "T_amb": 20}, "t": 700}'`;
$("#c2").textContent = `from pinneapple_systems.operations.control import ControlScenario, ControlSpec, compare_controllers

result = compare_controllers(ControlSpec(), ControlScenario(), controllers=("pid", "rules", "lmpc", "nmpc"))
for name, run in result["runs"].items():
    print(name, run["metrics"]["iae_T_K_s"], run["metrics"]["max_violation_T_K"])`;
api("/api/meta").then((m) => { META = m; $("#scope").innerHTML = renderScope(m.scope); }).catch(() => {});
