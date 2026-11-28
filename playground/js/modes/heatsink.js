// Heat-sink mode (UI only): plate-fin heat sink in ducted forced air. The zoo model's graph returns the resistance,
// its split-conformal 95 % bound, the pressure drop and the query checks (pinneapple_hub.zoo.checks).
import { LineChart } from "../charts.js";
import { writeState } from "../main.js";
import { load } from "../models.js";
import { renderModelTrust } from "../trust.js";
import { clamp, fmt, h, rafThrottle, renderVerdict, slider, svg, tile } from "../ui.js";

export const title = "Heat sink";
const MODEL = "heatsink_forced_mlp";
// card input order
const KEYS = ["base_width_mm", "base_depth_mm", "base_thickness_mm", "n_fins", "fin_thickness_mm", "fin_height_mm", "k_w_mk", "source_width_mm", "source_depth_mm", "t_ambient_c", "air_velocity_m_s", "power_w"];
const SHORT = { base_width_mm: "W", base_depth_mm: "D", base_thickness_mm: "tb", n_fins: "n", fin_thickness_mm: "t", fin_height_mm: "H", k_w_mk: "k", source_width_mm: "sw", source_depth_mm: "sd", t_ambient_c: "Ta", air_velocity_m_s: "V", power_w: "Q" };
const DEFAULTS = { base_width_mm: 80, base_depth_mm: 100, base_thickness_mm: 6, n_fins: 14, fin_thickness_mm: 1.5, fin_height_mm: 35, k_w_mk: 200, source_width_mm: 40, source_depth_mm: 40, t_ambient_c: 25, air_velocity_m_s: 2.5, power_w: 60 };
const MATERIALS = [["Aluminium 6063-T5", 209], ["Aluminium 1050", 229], ["Aluminium 6061-T6", 167], ["Copper C110", 388], ["Die-cast aluminium A380 (outside range)", 96]];
const V_SWEEP = Array.from({ length: 40 }, (_, i) => 0.2 + i * 0.2);

export async function mount(root, params) {
  const model = await load(MODEL);
  const vr = model.meta.card.validity_range;
  const st = { ...DEFAULTS };
  const S = {};
  const mk = (key, label, unit, min, max, step, digits, valid) => (S[key] = slider({ label, unit, min, max, step, value: st[key], valid: valid === undefined ? vr[key] : valid, digits, testid: SHORT[key], onInput: (v) => { st[key] = v; go(); } }));
  mk("n_fins", "Fin count", "", 2, 40, 1, 0, null);
  mk("fin_height_mm", "Fin height", "mm", 4, 100, 0.5, 1);
  mk("fin_thickness_mm", "Fin thickness", "mm", 0.3, 5, 0.05, 2);
  mk("air_velocity_m_s", "Air speed", "m/s", 0.2, 8, 0.05, 2);
  mk("power_w", "Power", "W", 5, 500, 1, 0);
  mk("base_width_mm", "Base width", "mm", 20, 240, 1, 0);
  mk("base_depth_mm", "Base depth (flow length)", "mm", 20, 240, 1, 0);
  mk("base_thickness_mm", "Base thickness", "mm", 1, 20, 0.5, 1);
  mk("source_width_mm", "Heat source width", "mm", 4, 240, 1, 0, null);
  mk("source_depth_mm", "Heat source depth", "mm", 4, 240, 1, 0, null);
  mk("t_ambient_c", "Ambient air", "°C", -10, 60, 1, 0);
  const mat = h("select", { class: "sel", "aria-label": "Material", "data-testid": "material", onchange: () => { st.k_w_mk = +mat.value; go(); } },
    MATERIALS.map(([n, k]) => h("option", { value: k }, `${n} · ${k} W/(m·K)`)));

  const controls = h("div", { class: "card" }, h("h3", {}, "Design"),
    S.n_fins.el, S.fin_height_mm.el, S.fin_thickness_mm.el, S.air_velocity_m_s.el,
    h("details", { class: "more" }, h("summary", {}, "Base, material, heat source, power"),
      h("label", { class: "slider" }, h("div", { class: "lab" }, h("span", {}, "Material")), mat),
      S.power_w.el, S.base_width_mm.el, S.base_depth_mm.el, S.base_thickness_mm.el, S.source_width_mm.el, S.source_depth_mm.el, S.t_ambient_c.el),
    h("p", { class: "hint" }, "Green: the range the model was trained on. Ducted flow (no bypass)."));

  const drawing = svg("svg", { class: "draw", viewBox: "0 0 820 360", "data-testid": "heatsink-svg", "aria-label": "Front view of the heat sink" });
  const stage = h("div", { class: "card stage" }, h("div", { class: "overlay" }, h("span", {}, "Front view, air flowing into the screen"), h("span", { "data-testid": "gap" })), drawing);
  const cR = h("div", { class: "card" }, h("h3", {}, "Thermal resistance vs air speed"));
  const chartR = new LineChart(cR, { xLabel: "air speed (m/s)", yLabel: "R (K/W)", testid: "r-chart", height: 220 });
  const cP = h("div", { class: "card" }, h("h3", {}, "Pressure drop vs air speed"));
  const chartP = new LineChart(cP, { xLabel: "air speed (m/s)", yLabel: "ΔP (Pa)", testid: "dp-chart", height: 200 });

  const tR = tile("Thermal resistance, base to air", { hero: true, testid: "R" }), tT = tile("Base temperature", { testid: "Tbase" }), tP = tile("Pressure drop", { testid: "dP" }), tG = tile("Fin gap", { testid: "fingap" });
  const results = h("div", { class: "card" }, h("h3", {}, "Prediction"), h("div", { class: "tiles" }, tR.el, tT.el, tP.el, tG.el),
    h("p", { class: "hint" }, "The 95 % bound is split-conformal: on 900 held-out designs the true resistance was below it 95.8 % of the time."));
  const verdict = h("div");
  const trustCard = h("div", { class: "card" }, h("h3", {}, "Trust verdict"), verdict, h("p", { class: "hint" }, h("a", { href: "#", onclick: (e) => { e.preventDefault(); document.getElementById("trust-open").click(); } }, "Why trust this? Model card and evidence →")));

  root.append(h("div", { class: "mode" }, h("div", { class: "col-left stack" }, controls), h("div", { class: "col-center stack" }, stage, cR, cP), h("div", { class: "col-right stack" }, results, trustCard)));
  stage.classList.add("o1"); results.classList.add("o2"); controls.classList.add("o3"); trustCard.classList.add("o4"); cR.classList.add("o5"); cP.classList.add("o6");

  const row = (o) => KEYS.map((k) => o[k]);
  let pred = null;

  function draw(Tb, gap) {
    drawing.innerHTML = "";
    const W = st.base_width_mm, H = st.fin_height_mm, tb = st.base_thickness_mm, n = st.n_fins, t = st.fin_thickness_mm;
    const sc = Math.min(560 / Math.max(W, 60), 230 / Math.max(H + tb + 10, 40)), x0 = 410 - (W * sc) / 2, yb = 300;
    // heat: 25 C -> 0, 125 C -> 1
    const heat = clamp(((Tb ?? 25) - 25) / 100, 0, 1), hot = `hsl(${30 - 30 * heat} ${55 + 35 * heat}% ${58 - 12 * heat}%)`;
    const gid = "hsg";
    drawing.append(svg("defs", {}, svg("linearGradient", { id: gid, x1: 0, y1: 1, x2: 0, y2: 0 }, svg("stop", { offset: 0, "stop-color": hot }), svg("stop", { offset: 1, "stop-color": "var(--metal)" }))));
    drawing.append(svg("rect", { x: x0, y: yb - tb * sc, width: W * sc, height: tb * sc, fill: hot, rx: 2 }));
    const pitch = t + gap;
    for (let i = 0; i < n; i++) {
      const fx = x0 + i * pitch * sc;
      if (fx + t * sc > x0 + W * sc + 0.5) break;
      drawing.append(svg("rect", { x: fx, y: yb - (tb + H) * sc, width: Math.max(t * sc, 1), height: H * sc, fill: `url(#${gid})` }));
      if (i < n - 1 && gap > 0) {
        const cx = fx + (t + gap / 2) * sc, dots = Math.max(1, Math.round(st.air_velocity_m_s));
        for (let k = 0; k < Math.min(dots, 6); k++) {
          const cy = yb - (tb + H * ((k + 0.5) / Math.min(dots, 6))) * sc;
          if (gap * sc > 5) drawing.append(svg("circle", { cx, cy, r: Math.min(3, gap * sc / 3), fill: "var(--series-1)", opacity: 0.55 }));
        }
      }
    }
    const sw = clamp(st.source_width_mm, 0, W);
    drawing.append(svg("rect", { x: 410 - (sw * sc) / 2, y: yb, width: sw * sc, height: 10, fill: "var(--series-2)", rx: 2 }));
    drawing.append(svg("text", { x: 410, y: yb + 28, "text-anchor": "middle", fill: "var(--muted)", "font-size": 12 }, `heat source ${st.power_w} W`));
    drawing.append(svg("text", { x: x0, y: yb - (tb + H) * sc - 10, fill: "var(--muted)", "font-size": 12 }, `${n} fins · H ${fmt(H, 1)} mm · t ${fmt(t, 2)} mm · W ${W} mm`));
    if (Tb !== undefined) drawing.append(svg("text", { x: x0 + W * sc + 10, y: yb - tb * sc / 2 + 4, fill: "var(--ink)", "font-size": 13, "font-weight": 600 }, `${fmt(Tb, 1)} °C`));
  }

  let seq = 0;
  async function update() {
    writeState("heatsink", Object.fromEntries(KEYS.map((k) => [SHORT[k], st[k]])));
    const my = ++seq, base = row(st);
    const out = await model.run([base, ...V_SWEEP.map((v) => row({ ...st, air_velocity_m_s: v }))], base);
    if (my !== seq) return;
    const y = out.rows.map((r) => [r.r_base_k_w, r.pressure_drop_pa, r.r_base_upper95_k_w]), p = y[0];
    pred = { R: p[0], dP: p[1], R95: p[2] };
    const Tb = st.t_ambient_c + st.power_w * p[0], Tb95 = st.t_ambient_c + st.power_w * p[2];
    tR.set(`${fmt(p[0], 3)} <small>K/W</small>`, `95 % upper bound ${fmt(p[2], 3)} K/W`, p[0]);
    tT.set(`${fmt(Tb, 1)} <small>°C</small>`, `≤ ${fmt(Tb95, 1)} °C (95 %)`, Tb);
    tP.set(`${fmt(p[1], 1)} <small>Pa</small>`, "", p[1]);
    const gap = out.query.checks.find((c) => c.id === "range:fin_gap_mm").value;   // as computed by the model's checks
    tG.set(`${fmt(gap, 2)} <small>mm</small>`, "", gap);
    root.querySelector("[data-testid=gap]").textContent = `${y.length} evaluations in ${model.lastMs >= 1 ? model.lastMs.toFixed(0) : "<1"} ms`;
    draw(Tb, gap);
    const sw = y.slice(1);
    chartR.update({ series: [{ name: "R (model)", x: V_SWEEP, y: sw.map((r) => r[0]), color: "var(--series-1)" }, { name: "95 % upper bound", x: V_SWEEP, y: sw.map((r) => r[2]), color: "var(--series-1)", dash: "5 4" }], marker: { x: st.air_velocity_m_s, y: p[0] }, xValid: vr.air_velocity_m_s, yZero: true, ariaLabel: "Thermal resistance against air speed" });
    chartP.update({ series: [{ name: "", x: V_SWEEP, y: sw.map((r) => r[1]), color: "var(--series-2)" }], marker: { x: st.air_velocity_m_s, y: p[1] }, xValid: vr.air_velocity_m_s, yZero: true, ariaLabel: "Pressure drop against air speed" });
    renderVerdict(verdict, out.query);
    renderModelTrust({ meta: model.meta, evaluation: out.query });
  }
  const go = rafThrottle(update);

  function setState(p) {
    for (const k of KEYS) { const v = Number(p.get(SHORT[k])); if (p.has(SHORT[k]) && Number.isFinite(v)) st[k] = v; }
    for (const k of KEYS) S[k]?.set(st[k]);
    if (![...mat.options].some((o) => +o.value === st.k_w_mk)) mat.append(h("option", { value: st.k_w_mk }, `Custom · ${st.k_w_mk} W/(m·K)`));
    mat.value = String(st.k_w_mk);
    go();
  }
  setState(params);
  window.__playground = { mode: "heatsink", state: st, get pred() { return pred; }, update: go };
  return { setState };
}
