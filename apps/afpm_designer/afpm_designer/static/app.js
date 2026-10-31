// Axial-Flux Motor Designer page: the design form, the two jobs (estimate, verify in 3-D), results and the 3-D view.
const $ = (s) => document.querySelector(s);
const LABELS = {
  poles: ["Poles", ""], slots: ["Slots", ""], r_in: ["Inner radius", "mm"], r_out: ["Outer radius", "mm"],
  gap: ["Air gap", "mm"], magnet_thickness: ["Magnet thickness", "mm"], magnet_arc: ["Magnet arc / pole pitch", ""],
  rotor_yoke: ["Rotor yoke", "mm"], tooth_height: ["Tooth height", "mm"], stator_yoke: ["Stator yoke", "mm"],
  slot_fraction: ["Slot opening / slot pitch", ""], Br: ["Remanence Br", "T"], topology: ["Topology", ""],
};
const MM = new Set(["r_in", "r_out", "gap", "magnet_thickness", "rotor_yoke", "tooth_height", "stator_yoke"]);
let last = { estimate: null, verify: null };
let chartData = {};

async function init() {
  const d = (await (await fetch("/api/defaults")).json()).design;
  $("#design").innerHTML = Object.entries(d).map(([k, v]) => {
    const [label, unit] = LABELS[k] || [k, ""];
    if (k === "topology") {
      return `<label>${label}<select name="${k}"><option value="single">single-sided</option>` +
        `<option value="double_rotor">double rotor</option></select></label>`;
    }
    const val = MM.has(k) ? +(v * 1e3).toPrecision(6) : v;
    return `<label>${label}${unit ? ` (${unit})` : ""}<input name="${k}" type="number" step="any" value="${val}"></label>`;
  }).join("");
  $("#btn-estimate").onclick = () => run("estimate");
  $("#btn-verify").onclick = () => run("verify");
}

function design() {
  const out = {};
  for (const el of $("#design").elements) {
    if (!el.name) continue;
    if (el.name === "topology") out.topology = el.value;
    else out[el.name] = MM.has(el.name) ? +el.value / 1e3 : +el.value;
  }
  out.poles = Math.round(out.poles); out.slots = Math.round(out.slots);
  return out;
}

async function run(kind) {
  $("#status").textContent = kind === "estimate" ? "estimating…" : "solving in 3-D…";
  for (const b of document.querySelectorAll(".actions button")) b.disabled = true;
  try {
    const r = await fetch(`/api/${kind}`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(design()) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail ? (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)) : r.statusText);
    const res = await poll(j.id || j.job_id);
    show(kind, res);
  } catch (e) {
    $("#status").textContent = `error: ${e.message}`;
  } finally {
    for (const b of document.querySelectorAll(".actions button")) b.disabled = false;
  }
}

async function poll(id) {
  for (;;) {
    const j = await (await fetch(`/api/jobs/${id}`)).json();
    $("#status").textContent = `${j.status}${j.message ? `: ${j.message}` : ""}`;
    if (j.status === "done" || j.status === "succeeded" || j.status === "finished") return j.result;
    if (["failed", "cancelled", "error"].includes(j.status)) throw new Error(j.error || j.status);
    await new Promise((r) => setTimeout(r, 700));
  }
}

const f = (v, k = 1, d = 4) => (v === undefined || v === null ? "—" : (+(v * k)).toPrecision(d));

function show(kind, r) {
  const set = (k, v) => { const el = document.querySelector(`[data-k=${k}]`); if (el) el.textContent = v; };
  const fill = (p, x) => {
    set(`${p}_pkpk`, f(x.cogging_pkpk_Nm)); set(`${p}_lam`, f(x.tooth_linkage_Wb, 1e3));
    set(`${p}_ke`, f(x.ke_tooth_V_s, 1e3));
    if (x.tets) set(`${p}_size`, `${x.tets.toLocaleString()}, ${x.seconds.toFixed(0)} s`);
  };
  if (kind === "estimate") {
    last.estimate = r; fill("e", r);
    chartData = { x: r.thetas_deg, est: r.cogging_Nm };
  } else {
    last.verify = r; fill("v", r); fill("e", r.estimate);
    set("r_pkpk", f(r.estimate_over_3d.cogging_pkpk)); set("r_lam", f(r.estimate_over_3d.tooth_linkage));
    const by = (x, what) => `${x >= 1 ? "overstates" : "understates"} ${what} by ${Math.abs(100 * (x - 1)).toFixed(1)} %`;
    $("#overstate").textContent = "On this design and grid the estimate "
      + `${by(r.estimate_over_3d.cogging_pkpk, "the cogging peak to peak")} and `
      + `${by(r.estimate_over_3d.tooth_linkage, "the flux linkage")}.`;
    chartData = { x: r.thetas_deg, est: r.estimate.cogging_Nm, v3: r.cogging_Nm };
  }
  const series = [{ name: "quasi-3D", y: chartData.est, color: "#eb6834", width: 2 }];
  if (chartData.v3) series.unshift({ name: "3-D", y: chartData.v3, color: "#2a78d6", width: 2 });
  window.PChart.mount($("#cogging"), { height: 240, x: chartData.x, series, yLabel: "cogging torque (N m)" });
  $("#status").textContent = `${kind === "estimate" ? "estimate" : "3-D check"} done`;
  view(r.view);
}

async function view(token) {
  if (!token) return;
  await new Promise((res) => (window.FieldView3D ? res() : window.addEventListener("fieldview3d-ready", res, { once: true })));
  const meta = await (await fetch(`/api/view/${token}`)).json();
  new window.FieldView3D($("#v3d"), { meta, glb: `/api/view/${token}.glb`, field: "B", height: 460 });
}

init();
