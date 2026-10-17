// Airfoil mode (UI only). The geometry (outline, thickness, handles, validity) comes from the library's CST graphs,
// the predictions and the trust verdict from the zoo model's graph (pp.zoo.export_web): see pinneapple_hub/zoo/web.
import { LineChart } from "../charts.js";
import { writeState } from "../main.js";
import { airfoilGeometry, load } from "../models.js";
import { renderModelTrust } from "../trust.js";
import { clamp, fmt, h, rafThrottle, renderVerdict, segmented, slider, svg, tile } from "../ui.js";

export const title = "Airfoil";
const MODEL = "airfoil_cst_rans_mlp";
const A_SLIDER = [-6, 18];
const SWEEP = Array.from({ length: 49 }, (_, i) => -6 + i * 0.5);
const DEFAULT = { preset: "NACA 2412", alpha: 4 };

// drawing frame: chord 0..1 -> 600 px, leading edge at x=110, chord line at y=190
const S = 600, OX = 110, OY = 190, PIV = 0.25;
const P = (x, y) => [OX + x * S, OY - y * S];

export async function mount(root, params) {
  const [model, geo] = await Promise.all([load(MODEL), airfoilGeometry()]);
  const card = model.meta.card, A_VALID = card.validity_range.alpha_deg;
  const presets = Object.fromEntries(Object.entries(geo.meta.presets).map(([k, v]) => [k, v.shape]));
  const st = { shape: presets[DEFAULT.preset].slice(), alpha: DEFAULT.alpha, chart: "cl" };

  // ------------------------------------------------------------------ layout
  const chips = h("div", { class: "chips", role: "group", "aria-label": "Presets" });
  for (const [name, p] of Object.entries(geo.meta.presets)) {
    const out = p.out_of_bounds.length > 0;
    chips.append(h("button", { class: `chip${out ? " out" : ""}`, type: "button", "data-preset": name, title: out ? `outside the model's CST bounds (${p.out_of_bounds.join(", ")})` : "", onclick: () => { st.shape = p.shape.slice(); syncSliders(); go(); } }, name));
  }
  const alpha = slider({ label: "Angle of attack α", unit: "deg", min: A_SLIDER[0], max: A_SLIDER[1], step: 0.25, value: st.alpha, valid: A_VALID, digits: 2, testid: "alpha", onInput: (v) => { st.alpha = v; go(); } });
  const weightSliders = geo.meta.params.map((p, i) => {
    const [lo, hi] = geo.meta.bounds[i], span = hi - lo;
    return slider({ label: p, min: +(lo - 0.6 * span).toFixed(3), max: +(hi + 0.6 * span).toFixed(3), step: 0.0005, value: st.shape[i], valid: [lo, hi], digits: 4, testid: `w-${p}`, onInput: (v) => { st.shape[i] = v; go(); } });
  });
  const syncSliders = () => { weightSliders.forEach((s, i) => s.set(st.shape[i])); alpha.set(st.alpha); };

  const controls = h("div", { class: "card" },
    h("h3", {}, "Shape"), chips,
    h("p", { class: "hint" }, "Drag the round handles on the airfoil, or pick a preset. Dashed presets sit outside the model's range."),
    alpha.el,
    h("details", { class: "more" }, h("summary", {}, "CST weights (thickness T, camber M)"), ...weightSliders.map((s) => s.el),
      h("p", { class: "hint" }, "Green: the bounds the model was trained on. upper = M + T, lower = M − T.")));

  const drawing = svg("svg", { class: "draw", viewBox: "0 0 820 380", "aria-label": "Airfoil shape; drag the handles to edit", "data-testid": "airfoil-svg" });
  const overlay = h("div", { class: "overlay" }, h("span", { "data-testid": "stage-alpha" }), h("span", {}, "Re 4·10⁶ · incompressible · k-ω SST"));
  const stage = h("div", { class: "card stage" }, overlay, drawing,
    h("div", { class: "note", "data-testid": "field-note" }, "Pressure and velocity fields: not available yet. This zoo model predicts integrated coefficients (Cl, Cd, Cm); a field model is planned for the next zoo release, and the field view will appear here when it ships."));

  const chartCard = h("div", { class: "card" });
  const chartHead = h("h3", {}, h("span", {}, "Polar sweep"), segmented([["cl", "Cl–α"], ["ld", "L/D–α"], ["polar", "Cl–Cd"], ["cm", "Cm–α"]], st.chart, (k) => { st.chart = k; drawChart(); }));
  chartCard.append(chartHead);
  const chart = new LineChart(chartCard, { xLabel: "α (deg)", yLabel: "Cl", testid: "polar-chart" });

  const tLD = tile("Lift-to-drag L/D", { hero: true, testid: "LD" }), tCl = tile("Lift Cl", { testid: "Cl" }), tCd = tile("Drag Cd", { testid: "Cd" }), tCm = tile("Moment Cm c/4", { testid: "Cm" }), tT = tile("Thickness t/c", { testid: "thickness" });
  const tCam = tile("Max camber", { testid: "camber" });
  const latency = h("span", { class: "hint", style: "margin:0;text-transform:none;letter-spacing:0", "data-testid": "latency" });
  const results = h("div", { class: "card" }, h("h3", {}, h("span", {}, "Prediction"), latency),
    h("div", { class: "tiles" }, tLD.el, tCl.el, tCd.el, tCm.el, tT.el, tCam.el),
    h("p", { class: "hint" }, "± is the spread of the 5-member ensemble (1σ)."));
  const verdict = h("div"), design = h("div");
  const trustCard = h("div", { class: "card" }, h("h3", {}, "Trust verdict"), verdict, h("h3", { style: "margin-top:14px" }, "Design constraints"), design,
    h("p", { class: "hint" }, h("a", { href: "#", onclick: (e) => { e.preventDefault(); document.getElementById("trust-open").click(); } }, "Why trust this? Model card and evidence →")));

  root.append(h("div", { class: "mode" },
    h("div", { class: "col-left stack" }, controls),
    h("div", { class: "col-center stack" }, stage, chartCard),
    h("div", { class: "col-right stack" }, results, trustCard)));
  stage.classList.add("o1"); results.classList.add("o2"); controls.classList.add("o3"); trustCard.classList.add("o4"); chartCard.classList.add("o5");

  // ------------------------------------------------------------------ drawing + drag
  let dragging = null, sweep = null, pred = null, shapeGeo = null, evaluation = null;
  const gFlow = svg("g"), gShape = svg("g"), gFixed = svg("g");
  drawing.append(svg("defs", {}, svg("marker", { id: "ah", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" }, svg("path", { d: "M0,0 L10,5 L0,10 z", fill: "context-stroke" }))), gFlow, gShape, gFixed);
  for (let k = 0; k < 6; k++) gFlow.append(svg("path", { class: "flow-arrow", d: `M14,${50 + k * 58} L64,${50 + k * 58}`, "marker-end": "url(#ah)" }));
  gFlow.append(svg("text", { x: 14, y: 368, fill: "var(--muted)", "font-size": 12 }, "V∞"));

  function localPoint(evt) {
    const m = gShape.getScreenCTM().inverse(), pt = new DOMPoint(evt.clientX, evt.clientY).matrixTransform(m);
    return [(pt.x - OX) / S, (OY - pt.y) / S];
  }
  async function moveHandle(side, i, y) {
    const yu = shapeGeo.handles.upper.map((p) => p[1]), yl = shapeGeo.handles.lower.map((p) => p[1]);
    if (side === "upper") yu[i] = clamp(y, yl[i] + 0.004, 0.3); else yl[i] = clamp(y, -0.3, yu[i] - 0.004);
    st.shape = await geo.fromHandles(yu, yl);
    syncSliders(); go();
  }
  drawing.addEventListener("pointermove", (e) => { if (dragging && shapeGeo) moveHandle(dragging.side, dragging.i, localPoint(e)[1]); });
  const endDrag = () => { if (dragging) { dragging.el.classList.remove("drag"); dragging = null; } };
  drawing.addEventListener("pointerup", endDrag); drawing.addEventListener("pointercancel", endDrag);

  function drawShape(rangeFail) {
    gShape.innerHTML = "";
    const [px, py] = P(PIV, 0);
    gShape.setAttribute("transform", `rotate(${st.alpha} ${px} ${py})`);
    const g = shapeGeo, n = g.x.length, pts = [];
    for (let i = n - 1; i >= 0; i--) pts.push(P(g.x[i], g.upper[i]));
    for (let i = 1; i < n; i++) pts.push(P(g.x[i], g.lower[i]));
    const out = rangeFail || !g.valid[0];
    gShape.append(svg("path", { d: `M${pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join("L")}Z`, fill: out ? "var(--bad-bg)" : "var(--shape-fill)", stroke: out ? "var(--bad)" : "var(--shape-line)", "stroke-width": 2, "stroke-linejoin": "round", "data-testid": "airfoil-path" }));
    gShape.append(svg("path", { d: `M${g.x.map((x, i) => P(x, (g.upper[i] + g.lower[i]) / 2).map((v) => v.toFixed(1)).join(",")).join("L")}`, fill: "none", stroke: "var(--shape-line)", "stroke-width": 1, "stroke-dasharray": "4 4", opacity: 0.6 }));
    gShape.append(svg("line", { x1: P(0, 0)[0], y1: OY, x2: P(1, 0)[0], y2: OY, stroke: "var(--axis)", "stroke-width": 1, "stroke-dasharray": "2 5" }));
    for (const side of ["upper", "lower"]) g.handles[side].forEach(([x, y], i) => {
      const [cx, cy] = P(x, y);
      const c = svg("circle", { class: "handle", cx, cy, r: 8, "data-handle": `${side}${i}`, tabindex: 0, role: "slider", "aria-label": `${side} surface handle ${i + 1}`, "aria-valuenow": y.toFixed(4) });
      const hit = svg("circle", { class: "handle-hit", cx, cy, r: 20 });
      const start = (e) => { e.preventDefault(); drawing.setPointerCapture(e.pointerId); dragging = { side, i, el: c }; c.classList.add("drag"); };
      hit.addEventListener("pointerdown", start); c.addEventListener("pointerdown", start);
      c.addEventListener("keydown", async (e) => {
        const d = { ArrowUp: 0.002, ArrowDown: -0.002 }[e.key]; if (!d) return;
        e.preventDefault();
        await moveHandle(side, i, y + d);
        requestAnimationFrame(() => drawing.querySelector(`[data-handle="${side}${i}"]`)?.focus());
      });
      gShape.append(hit, c);
    });
  }

  function drawForces() {
    gFixed.innerHTML = "";
    if (!pred) return;
    // forces at the quarter chord: lift normal to the free stream, drag along it (drawn 25x longer, it is small)
    const [px, py] = P(PIV, 0), L = clamp(pred.Cl, -1.2, 2.4) * 70, D = clamp(pred.Cd * 70 * 25, 0, 300);
    gFixed.append(svg("line", { class: "force lift", x1: px, y1: py, x2: px, y2: py - L, "marker-end": "url(#ah)" }));
    gFixed.append(svg("text", { x: px - 8, y: py - L + (L > 0 ? -8 : 18), "text-anchor": "end", fill: "var(--series-3)", "font-size": 14, "font-weight": 650 }, `Lift  Cl ${fmt(pred.Cl, 3)}`));
    gFixed.append(svg("line", { class: "force drag", x1: px, y1: py, x2: px + D, y2: py, "marker-end": "url(#ah)" }));
    gFixed.append(svg("text", { x: px + D + 8, y: py + 5, fill: "var(--series-2)", "font-size": 14, "font-weight": 650 }, `Drag ×25`));
    gFixed.append(svg("circle", { cx: px, cy: py, r: 4, fill: "var(--ink)" }));
  }

  const matchPreset = () => Object.keys(presets).find((k) => presets[k].every((w, i) => Math.abs(w - st.shape[i]) < 2e-4)) || null;

  function drawChart() {
    if (!sweep) return;
    const preset = matchPreset(), y = (k) => sweep.map((p) => p[k]);
    const band = (k, s) => ({ lo: y(k).map((v, i) => v - 2 * sweep[i][s]), hi: y(k).map((v, i) => v + 2 * sweep[i][s]) });
    const polars = model.meta.reference.polars.filter((r) => r.preset === preset);
    const refs = (key) => polars.map((r, j) => {
      const yv = key === "polar" ? r.Cl : key === "ld" ? r.Cl.map((c, i) => c / r.Cd[i]) : key === "cm" ? r.Cm : r.Cl;
      const xv = key === "polar" ? r.Cd : r.alpha, ok = yv.map(Number.isFinite);
      return { name: r.source, color: j ? "var(--series-3)" : "var(--series-2)", x: xv.filter((_, i) => ok[i]), y: yv.filter((_, i) => ok[i]) };
    });
    const cfg = { cl: ["Cl", "Cl", "Cl_std"], ld: ["L/D", "LD", "LD_std"], cm: ["Cm c/4", "Cm", "Cm_std"], polar: ["Cl", "Cl", null] }[st.chart];
    chart.yLabel = cfg[0]; chart.xLabel = st.chart === "polar" ? "Cd" : "α (deg)";
    const name = `${MODEL} (ensemble mean)`;
    if (st.chart === "polar") {
      const keep = SWEEP.map((a) => a >= A_VALID[0] && a <= A_VALID[1]);
      chart.update({ series: [{ name, x: y("Cd").filter((_, i) => keep[i]), y: y("Cl").filter((_, i) => keep[i]), color: "var(--series-1)" }], refs: refs("polar"), marker: { x: pred.Cd, y: pred.Cl }, xFmt: (v) => v.toFixed(4), ariaLabel: "Drag polar, Cl against Cd" });
    } else {
      chart.update({ series: [{ name, x: SWEEP, y: y(cfg[1]), ...band(cfg[1], cfg[2]), color: "var(--series-1)" }], refs: refs(st.chart), marker: { x: st.alpha, y: pred[cfg[1]] }, xValid: A_VALID, ariaLabel: `${cfg[0]} against angle of attack` });
    }
  }

  let seq = 0;
  async function update() {
    writeState("airfoil", { s: st.shape.map((v) => v.toFixed(5)).join(","), a: st.alpha, ...(st.chart !== "cl" ? { c: st.chart } : {}) });
    const my = ++seq, row = [...st.shape, st.alpha];
    const [g, res] = await Promise.all([geo.measure(st.shape), model.run([row, ...SWEEP.map((a) => [...st.shape, a])], row)]);
    if (my !== seq) return;
    shapeGeo = g; pred = res.rows[0]; sweep = res.rows.slice(1); evaluation = res.query;
    drawShape(evaluation.groups.some((c) => c.id === "cst-range" && c.status === "fail"));
    drawForces();
    overlay.firstChild.textContent = `α = ${st.alpha.toFixed(2)}°`;
    tLD.set(`${fmt(pred.LD, 1)} <small>± ${fmt(pred.LD_std, 1)}</small>`, "ensemble mean ± 1σ", pred.LD);
    tCl.set(fmt(pred.Cl, 3), `± ${fmt(pred.Cl_std, 4)}`, pred.Cl);
    tCd.set(`${(pred.Cd * 1e4).toFixed(1)} <small>counts</small>`, `Cd ${fmt(pred.Cd, 5)} ± ${(pred.Cd_std * 1e4).toFixed(1)} ct`, pred.Cd);
    tCm.set(fmt(pred.Cm, 4), `± ${fmt(pred.Cm_std, 4)}`, pred.Cm);
    tT.set(`${(100 * g.props.thickness).toFixed(1)} <small>%</small>`, `at x/c ${g.props.thickness_at.toFixed(2)}`, g.props.thickness);
    tCam.set(`${(100 * g.props.camber).toFixed(2)} <small>%</small>`, `at x/c ${g.props.camber_at.toFixed(2)}`, g.props.camber);
    latency.textContent = `${SWEEP.length + 1} evaluations in ${model.lastMs >= 1 ? model.lastMs.toFixed(0) : "<1"} ms`;
    renderVerdict(verdict, evaluation);
    renderVerdict(design, evaluation, "design", "design");
    chips.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.preset === matchPreset())));
    drawChart();
    renderModelTrust({ meta: model.meta, evaluation, extra: h("div", {}, h("h4", {}, "Independent reference on the polar"), h("p", {}, "Pick NACA 2412, 0012, 4412 or 4415: the polar shows OpenFOAM runs the model never saw and, for NACA 2412, an XFOIL polar from an independent solver.")) });
  }
  const go = rafThrottle(update);

  function setState(p) {
    const s = (p.get("s") || "").split(",").map(Number);
    if (s.length === 6 && s.every(Number.isFinite)) st.shape = s;
    else if (p.get("preset") && presets[p.get("preset")]) st.shape = presets[p.get("preset")].slice();
    const a = Number(p.get("a"));
    if (p.has("a") && Number.isFinite(a)) st.alpha = clamp(a, A_SLIDER[0], A_SLIDER[1]);
    if (["cl", "ld", "polar", "cm"].includes(p.get("c"))) st.chart = p.get("c");
    chartHead.querySelectorAll(".seg button").forEach((b, i) => b.setAttribute("aria-pressed", String(["cl", "ld", "polar", "cm"][i] === st.chart)));
    syncSliders(); go();
  }
  setState(params);
  const onTheme = () => drawChart();
  window.addEventListener("themechange", onTheme);
  // exposed for the browser tests and the demo recorder
  window.__playground = { mode: "airfoil", state: st, get pred() { return pred; }, get evaluation() { return evaluation; }, update: go };
  return { setState, unmount() { window.removeEventListener("themechange", onTheme); } };
}
