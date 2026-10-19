// PINNeAPPle zoo demo (static Hugging Face Space). UI only: the sweeps, bands and check verdicts come from
// pinneapple_hub.zoo.demo.export_space (computed from the published models when the zoo is published); for the
// visitor's own inputs, the model's ONNX graph and its query checks run in the browser through the library's
// runtime (lib/zoo.js, lib/onnx.js). Charts and widgets are the Playground's (js/charts.js, js/ui.js).
import { grouped, loadModel } from "./lib/zoo.js";
import { LineChart } from "./js/charts.js";
import { h, renderVerdict, sig } from "./js/ui.js";

const view = document.getElementById("view");
const getJSON = async (url) => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json(); };
const index = await getJSON("index.json");
const pre = {}, metas = {}, graphs = {};
const load = async (name) => {
  pre[name] ??= await getJSON(`models/${name}.sweeps.json`);
  metas[name] ??= await getJSON(`models/${name}.json`);
  return pre[name];
};

/** Contiguous stretches of a sweep where the verdict is not "pass": [{x0, x1, status}]. */
function zones(x, status) {
  const out = [];
  let cur = null;
  status.forEach((s, i) => {
    const a = i ? (x[i - 1] + x[i]) / 2 : x[0], b = i < x.length - 1 ? (x[i] + x[i + 1]) / 2 : x[i];
    if (s === "pass") { cur = null; return; }
    if (cur && cur.status === s) cur.x1 = b;
    else { cur = { x0: a, x1: b, status: s }; out.push(cur); }
  });
  return out;
}

const state = { model: index.models[0].name, vary: null, values: null, result: null };

async function show(name, { keepValues = false } = {}) {
  const p = await load(name);
  state.model = name;
  if (!keepValues || !state.values) state.values = { ...p.default.inputs };
  state.vary = p.inputs.some((d) => d.name === state.vary) ? state.vary : p.inputs[p.inputs.length - 1].name;
  state.result = { source: "precomputed", sweep: p.sweeps[state.vary], point: { pred: p.default.pred, checks: p.default.checks } };
  render();
}

/** Run the sweep and the chosen point through the model's ONNX graph (prediction + query checks per row). */
async function evaluateHere() {
  const p = pre[state.model], names = p.inputs.map((d) => d.name), sw = p.sweeps[state.vary];
  graphs[state.model] ??= await loadModel("models", state.model);
  const m = graphs[state.model];
  const base = names.map((k) => state.values[k]);
  // the precomputed grid, stretched to include the chosen value when it lies outside it
  const v = state.values[state.vary], a = Math.min(sw.x[0], v), b = Math.max(sw.x[sw.x.length - 1], v);
  const xs = a === sw.x[0] && b === sw.x[sw.x.length - 1] ? sw.x : sw.x.map((_, i) => a + ((b - a) * i) / (sw.x.length - 1));
  const rows = xs.map((x) => base.map((u, j) => (names[j] === state.vary ? x : u)));
  const res = await m.evaluateRows([...rows, base]);
  const point = res.pop();
  const pred = Object.fromEntries(p.all_outputs.map((k) => [k, res.map((r) => r.outputs[k])]));
  state.result = { source: "browser", ms: m.lastMs, sweep: { x: xs, range: sw.range, pred, status: res.map((r) => r.verdict.status) },
    point: { pred: point.outputs, checks: { verdict: point.verdict, checks: point.checks } } };
  render();
}

function controls(p) {
  const card = h("div", { class: "card" }, h("h3", {}, "Model"));
  const sel = h("select", { class: "sel", "aria-label": "Model", onchange: (e) => show(e.target.value) },
    index.models.map((m) => h("option", { value: m.name, selected: m.name === state.model }, m.name)));
  card.append(sel, h("p", { class: "hint" }, p.task));
  const vary = h("select", { class: "sel", "aria-label": "Sweep this input", onchange: (e) => { state.vary = e.target.value; state.result.source === "browser" ? evaluateHere() : show(state.model, { keepValues: true }); } },
    p.inputs.map((d) => h("option", { value: d.name, selected: d.name === state.vary }, d.name)));
  const inputs = h("div", { class: "card" }, h("h3", {}, "Inputs"), h("label", { class: "hint" }, "Sweep this input"), vary);
  for (const d of p.inputs) {
    const inp = h("input", { type: "number", step: "any", value: state.values[d.name], "aria-label": d.name });
    const paint = () => inp.classList.toggle("oor", +inp.value < d.lo || +inp.value > d.hi);
    inp.addEventListener("input", () => { state.values[d.name] = +inp.value; paint(); });
    paint();
    inputs.append(h("div", { class: "field", title: d.description },
      h("label", {}, h("span", {}, d.name, d.unit && d.unit !== "-" ? h("span", { class: "u" }, ` ${d.unit}`) : null), h("span", { class: "rng" }, `${sig(d.lo, 4)} … ${sig(d.hi, 4)}`)), inp));
  }
  inputs.append(h("div", { class: "row" },
    h("button", { class: "btn trust-btn", type: "button", onclick: () => evaluateHere().catch(fail) }, "Evaluate in your browser"),
    h("button", { class: "btn", type: "button", onclick: () => show(state.model) }, "Reset")));
  return h("div", { class: "stack" }, card, inputs);
}

function charts(p) {
  const r = state.result, sw = r.sweep, d = p.inputs.find((i) => i.name === state.vary);
  const wrap = h("div", { class: "card" }, h("h3", {}, `Sweep of ${state.vary}`),
    h("p", { class: "src" }, r.source === "browser" ? `Computed in your browser (ONNX, ${sig(r.ms, 3)} ms).` : "Precomputed from the published model at a held-out test case."));
  const z = zones(sw.x, sw.status);
  for (const o of p.outputs) {
    const y = sw.pred[o.name];
    const s = { name: o.name, x: sw.x, y, color: "var(--brand)" };
    if (o.std) { const sd = sw.pred[o.std]; s.lo = y.map((v, i) => v - 2 * sd[i]); s.hi = y.map((v, i) => v + 2 * sd[i]); }
    if (o.upper) { s.lo = y; s.hi = sw.pred[o.upper]; }
    const c = new LineChart(wrap, { xLabel: `${state.vary}${d.unit && d.unit !== "-" ? ` [${d.unit}]` : ""}`, yLabel: `${o.name}${o.unit && o.unit !== "-" ? ` [${o.unit}]` : ""}` });
    c.update({ series: [s], zones: z, xValid: sw.range, marker: { x: state.values[state.vary], y: r.point.pred[o.name] } });
  }
  return wrap;
}

function verdict(p) {
  const r = state.result, spec = metas[state.model].checks, ev = { verdict: r.point.checks.verdict, groups: grouped(spec, r.point.checks.checks) };
  const box = h("div");
  renderVerdict(box, ev);
  const out = h("dl", { class: "kv" }, p.outputs.flatMap((o) => [h("dt", {}, o.name), h("dd", {}, `${sig(r.point.pred[o.name], 5)} ${o.unit && o.unit !== "-" ? o.unit : ""}`)]));
  return h("div", { class: "stack" },
    h("div", { class: "card" }, h("h3", {}, "This input"), out, box),
    h("div", { class: "card" }, h("h3", {}, "The model"),
      h("p", {}, h("b", {}, `${p.model} v${p.version}`), " · trust decision ", h("span", { class: `decision ${p.trust.decision}` }, p.trust.decision)),
      h("p", { class: "hint" }, "pp.zoo.validate: card metrics reproduced on the stored reference, acceptance limits and the model's physics checks."),
      h("p", {}, h("a", { href: p.hub, target: "_blank", rel: "noopener" }, "Model card on the Hub"), ` · revision ${p.revision}`)));
}

function render() {
  const p = pre[state.model];
  view.innerHTML = "";
  view.append(h("div", { class: "zoo" }, controls(p), charts(p), verdict(p)));
}

function fail(e) {
  view.prepend(h("div", { class: "verdict fail", role: "alert" }, h("div", {}, h("b", {}, "Something went wrong"), h("span", {}, String(e.message || e)))));
}

show(state.model).catch(fail);
