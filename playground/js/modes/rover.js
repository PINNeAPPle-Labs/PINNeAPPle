// Rover-wheel mode (UI only): the NASA VIPER wheel on GRC-1 lunar simulant. The zoo's wheel-soil contact PINN
// (one graph: forces and query checks, pinneapple_hub.zoo.web) and held-out points of the analytic Wong-Reece model.
import { LineChart } from "../charts.js";
import { writeState } from "../main.js";
import { load } from "../models.js";
import { renderModelTrust } from "../trust.js";
import { clamp, fmt, h, rafThrottle, renderVerdict, segmented, slider, svg, tile } from "../ui.js";

export const title = "Rover wheel";
const MODEL = "rover_contact_grc1_viper";
const S_SWEEP = Array.from({ length: 57 }, (_, i) => -0.6 + i * 0.03);
const GROUSERS = 24;

export async function mount(root, params) {
  const model = await load(MODEL);
  const phys = model.meta.reference.physics, ho = model.meta.reference.heldout, vr = model.meta.card.validity_range;
  const S_VALID = vr.slip, Z_VALID = vr.sinkage_m.map((v) => v * 1000);
  const st = { slip: 0.2, z: 20, chart: "Fx" };

  const sS = slider({ label: "Slip ratio", min: -0.6, max: 1.0, step: 0.01, value: st.slip, valid: S_VALID, digits: 2, testid: "slip", onInput: (v) => { st.slip = v; go(); } });
  const sZ = slider({ label: "Sinkage", unit: "mm", min: 0.5, max: 80, step: 0.5, value: st.z, valid: Z_VALID, digits: 1, testid: "sinkage", onInput: (v) => { st.z = v; go(); } });
  const controls = h("div", { class: "card" }, h("h3", {}, "Wheel state"), sS.el, sZ.el,
    h("p", { class: "hint" }, "Slip > 0: the wheel spins faster than it moves (driving). Slip < 0: it skids (braking)."),
    h("dl", { class: "kv", style: "margin-top:12px" }, h("dt", {}, "Wheel"), h("dd", {}, "VIPER, R 0.25 m, b 0.20 m"), h("dt", {}, "Grousers"), h("dd", {}, "24"), h("dt", {}, "Soil"), h("dd", {}, "GRC-1 lunar simulant"), h("dt", {}, "Slip angle"), h("dd", {}, "0°")));

  const drawing = svg("svg", { class: "draw", viewBox: "0 0 820 380", "data-testid": "rover-svg", "aria-label": "VIPER wheel in the soil" });
  const stage = h("div", { class: "card stage" }, h("div", { class: "overlay" }, h("span", { "data-testid": "wheel-state" }), h("span", {}, "side view, to scale")), drawing);
  const cC = h("div", { class: "card" });
  const head = h("h3", {}, h("span", {}, "Sweep over slip at this sinkage"), segmented([["Fx", "Drawbar pull"], ["Fz", "Normal load"], ["My", "Torque"]], st.chart, (k) => { st.chart = k; drawChart(); }));
  cC.append(head);
  const chart = new LineChart(cC, { xLabel: "slip", yLabel: "Fx (N)", testid: "rover-chart" });

  const tFx = tile("Drawbar pull Fx", { hero: true, testid: "Fx" }), tFz = tile("Normal load Fz", { testid: "Fz" }), tMy = tile("Driving torque My", { testid: "My" });
  const results = h("div", { class: "card" }, h("h3", {}, h("span", {}, "Prediction"), h("span", { class: "hint", style: "margin:0;text-transform:none;letter-spacing:0", "data-testid": "latency" })),
    h("div", { class: "tiles" }, tFx.el, tFz.el, tMy.el),
    h("p", { class: "hint" }, `Held-out error vs the analytic model (v${model.meta.card.version}): Fx RMSE ${fmt(model.meta.card.validation_metrics["test/Fx_rmse_N"], 2)} N, Fz ${fmt(model.meta.card.validation_metrics["test/Fz_rmse_N"], 2)} N. The Wong–Reece model itself is 4–14 N from published single-wheel tests.`));
  const verdict = h("div");
  const trustCard = h("div", { class: "card" }, h("h3", {}, "Trust verdict"), verdict, h("p", { class: "hint" }, h("a", { href: "#", onclick: (e) => { e.preventDefault(); document.getElementById("trust-open").click(); } }, "Why trust this? Model card and evidence →")));
  root.append(h("div", { class: "mode" }, h("div", { class: "col-left stack" }, controls), h("div", { class: "col-center stack" }, stage, cC), h("div", { class: "col-right stack" }, results, trustCard)));
  stage.classList.add("o1"); results.classList.add("o2"); controls.classList.add("o3"); trustCard.classList.add("o4"); cC.classList.add("o5");

  // ------------------------------------------------------------------ wheel drawing + animation
  const PX = 480;                                   // px per metre
  const ground = 292, cx = 410, R = phys.R_m * PX;
  const gSoil = svg("g"), gWheel = svg("g"), gForces = svg("g"), gDust = svg("g");
  drawing.append(svg("defs", {}, svg("marker", { id: "ahr", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, svg("path", { d: "M0,0 L10,5 L0,10 z", fill: "context-stroke" }))), gSoil, gWheel, gDust, gForces);
  let phase = 0, soilShift = 0, last = performance.now(), raf = 0, pred = null;
  const rnd = (i) => { const v = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return v - Math.floor(v); };
  const pebbles = Array.from({ length: 60 }, (_, i) => ({ x: rnd(i) * 820, y: ground + 10 + rnd(i + 100) * 70, r: 1 + rnd(i + 200) * 2 }));

  function drawWheel() {
    const zpx = st.z / 1000 * PX, cy = ground - R + zpx;
    gSoil.innerHTML = "";
    // undisturbed soil ahead of the wheel, the rut (depth = sinkage) behind it, a small bow wave at the front
    gSoil.append(svg("rect", { x: 0, y: ground + zpx, width: 820, height: 140, fill: "var(--soil)" }));
    gSoil.append(svg("path", { d: `M0,${ground} L${cx - R * 0.95},${ground} Q${cx - R * 0.55},${ground - zpx * 0.35} ${cx},${ground + zpx} L0,${ground + zpx} Z`, fill: "var(--soil)" }));
    for (const p of pebbles) {
      const x = ((p.x - soilShift) % 820 + 820) % 820;
      gSoil.append(svg("circle", { cx: x, cy: p.y + zpx, r: p.r, fill: "var(--soil-dark)" }));
    }
    gWheel.innerHTML = "";
    gWheel.setAttribute("transform", `translate(${cx} ${cy}) rotate(${phase})`);
    gWheel.append(svg("circle", { r: R, fill: "var(--metal)", stroke: "var(--ink-2)", "stroke-width": 2 }));
    gWheel.append(svg("circle", { r: R * 0.93, fill: "none", stroke: "var(--ink-2)", "stroke-width": 1, opacity: 0.4 }));
    for (let i = 0; i < GROUSERS; i++) {
      const a = (i / GROUSERS) * 2 * Math.PI;
      gWheel.append(svg("line", { x1: Math.cos(a) * R, y1: Math.sin(a) * R, x2: Math.cos(a) * (R + 9), y2: Math.sin(a) * (R + 9), stroke: "var(--ink-2)", "stroke-width": 3, "stroke-linecap": "round" }));
    }
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * 2 * Math.PI;
      gWheel.append(svg("line", { x1: Math.cos(a) * 22, y1: Math.sin(a) * 22, x2: Math.cos(a) * R * 0.9, y2: Math.sin(a) * R * 0.9, stroke: "var(--ink-2)", "stroke-width": 2.5, opacity: 0.55 }));
    }
    gWheel.append(svg("circle", { r: 22, fill: "var(--surface)", stroke: "var(--ink-2)", "stroke-width": 2 }));
    return { cy };
  }

  function drawForces(cy) {
    gForces.innerHTML = "";
    if (!pred) return;
    const fx = clamp(pred.Fx / 2.5, -220, 220);
    const by = cy + R;
    gForces.append(svg("line", { class: "force", stroke: "var(--series-3)", x1: cx, y1: by - 4, x2: cx + fx, y2: by - 4, "marker-end": "url(#ahr)" }));
    gForces.append(svg("text", { x: cx + fx + (fx >= 0 ? 8 : -8), y: by + 18, "text-anchor": fx >= 0 ? "start" : "end", fill: "var(--series-3)", "font-size": 13, "font-weight": 600 }, `Fx ${fmt(pred.Fx, 0)} N`));
    // the soil pushes up on the wheel at the contact patch
    const fzl = clamp(pred.Fz / 18, 6, 72);
    gForces.append(svg("line", { class: "force", stroke: "var(--series-1)", x1: cx - 30, y1: by + fzl + 4, x2: cx - 30, y2: by + 2, "marker-end": "url(#ahr)" }));
    gForces.append(svg("text", { x: cx - 40, y: by + 22, "text-anchor": "end", fill: "var(--ink)", "font-size": 13, "font-weight": 600 }, `Fz ${fmt(pred.Fz, 0)} N`));
    const tr = R + 26, sweep = clamp(pred.My / 300, -1, 1) * 70;
    if (Math.abs(sweep) > 2) {
      const a0 = -100, a1 = a0 + sweep, P = (a) => [cx + tr * Math.cos(a * Math.PI / 180), cy + tr * Math.sin(a * Math.PI / 180)];
      const [x0, y0] = P(a0), [x1, y1] = P(a1);
      gForces.append(svg("path", { class: "force", stroke: "var(--series-2)", d: `M${x0},${y0} A${tr},${tr} 0 0 ${sweep > 0 ? 1 : 0} ${x1},${y1}`, "marker-end": "url(#ahr)" }));
      gForces.append(svg("text", { x: x1 + 8, y: y1 - 6, fill: "var(--series-2)", "font-size": 13, "font-weight": 600 }, `My ${fmt(pred.My, 0)} N·m`));
    }
  }

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.05); last = now;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const omega = reduce ? 0 : 40;                  // deg/s, visual
    phase += omega * dt;
    // ground speed = omega R (1 - s); slip 1 = spinning in place, slip < 0 = skidding
    soilShift += (omega * Math.PI / 180) * R * (1 - clamp(st.slip, -1, 1)) * dt;
    const { cy } = drawWheel();
    drawForces(cy);
    gDust.innerHTML = "";
    if (!reduce && st.slip > 0.3) for (let i = 0; i < 10; i++) {
      const t = ((now / 600 + i / 10) % 1), x = cx - 30 - t * 140 * st.slip, y = ground - 10 - Math.sin(t * Math.PI) * 50 * st.slip;
      gDust.append(svg("circle", { cx: x, cy: y + st.z / 1000 * PX, r: 2.2, fill: "var(--soil-dark)", opacity: 1 - t }));
    }
    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);

  // ------------------------------------------------------------------ model + checks
  function drawChart() {
    if (!drawChart.sweep) return;
    const k = st.chart, idx = { Fx: 0, Fz: 1, My: 2 }[k], unit = k === "My" ? "N·m" : "N";
    const near = ho.slip.map((s, i) => [s, ho.sinkage_m[i], ho[`${k}_${k === "My" ? "Nm" : "N"}`][i]]).filter(([, z]) => Math.abs(z * 1000 - st.z) <= 1.5);
    chart.yLabel = `${k} (${unit})`;
    chart.update({
      series: [{ name: `${MODEL}`, x: S_SWEEP, y: drawChart.sweep.map((r) => r[idx]), color: "var(--series-1)" }],
      refs: [{ name: `Wong–Reece, held-out (sinkage ${st.z} ± 1.5 mm)`, x: near.map((n) => n[0]), y: near.map((n) => n[2]), color: "var(--series-2)" }],
      marker: { x: st.slip, y: pred[k] }, xValid: S_VALID, ariaLabel: `${k} against slip`,
    });
  }

  let seq = 0;
  async function update() {
    writeState("rover", { s: st.slip, z: st.z, ...(st.chart !== "Fx" ? { c: st.chart } : {}) });
    const my = ++seq, z = st.z / 1000;
    const out = await model.run([[st.slip, z], ...S_SWEEP.map((s) => [s, z])], [st.slip, z]);
    if (my !== seq) return;
    const y = out.rows.map((r) => [r.Fx_N, r.Fz_N, r.My_Nm]), p = { Fx: y[0][0], Fz: y[0][1], My: y[0][2] };
    pred = p; drawChart.sweep = y.slice(1);
    tFx.set(`${fmt(p.Fx, 1)} <small>N</small>`, "", p.Fx);
    tFz.set(`${fmt(p.Fz, 1)} <small>N</small>`, "", p.Fz);
    tMy.set(`${fmt(p.My, 1)} <small>N·m</small>`, "", p.My);
    root.querySelector("[data-testid=wheel-state]").textContent = st.slip > 0 ? `driving, slip ${st.slip.toFixed(2)}` : st.slip < 0 ? `skidding, slip ${st.slip.toFixed(2)}` : "free rolling";
    root.querySelector("[data-testid=latency]").textContent = `${y.length} evaluations in ${model.lastMs >= 1 ? model.lastMs.toFixed(0) : "<1"} ms`;
    drawChart();
    renderVerdict(verdict, out.query);
    renderModelTrust({ meta: model.meta, evaluation: out.query });
  }
  const go = rafThrottle(update);

  function setState(p) {
    const s = Number(p.get("s")), z = Number(p.get("z"));
    if (p.has("s") && Number.isFinite(s)) st.slip = clamp(s, -0.6, 1);
    if (p.has("z") && Number.isFinite(z)) st.z = clamp(z, 0.5, 80);
    if (["Fx", "Fz", "My"].includes(p.get("c"))) st.chart = p.get("c");
    head.querySelectorAll(".seg button").forEach((b, i) => b.setAttribute("aria-pressed", String(["Fx", "Fz", "My"][i] === st.chart)));
    sS.set(st.slip); sZ.set(st.z); go();
  }
  setState(params);
  window.__playground = { mode: "rover", state: st, get pred() { return pred; }, update: go };
  return { setState, unmount() { cancelAnimationFrame(raf); } };
}
