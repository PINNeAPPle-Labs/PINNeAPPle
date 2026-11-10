// PINNeAPPle web runtime: load the zoo models and the CST airfoil geometry exported by the library
// (`python -m pinneapple_hub.zoo.web DIR`), run them, and evaluate the query checks into a trust verdict exactly as
// pinneapple_hub.zoo.checks.evaluate does in Python (tests/js/zoo_web.test.mjs checks both against each other).
import { Session } from "./onnx.js";

export const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs";

// ------------------------------------------------------------------ graphs
async function ortSession(bytes) {
  const ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_URL.replace(/[^/]+$/, "");
  ort.env.wasm.numThreads = 1;
  const s = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  return {
    outputNames: s.outputNames, inputNames: s.inputNames,
    async run(feeds, types) {
      const f = {};
      for (const [k, v] of Object.entries(feeds)) {
        const T = types[k] === "float64" ? Float64Array : Float32Array;
        f[k] = new ort.Tensor(types[k] === "float64" ? "float64" : "float32", T.from(v.data), v.dims);
      }
      const r = await s.run(f), out = {};
      for (const k of s.outputNames) out[k] = { dims: [...r[k].dims], data: Float64Array.from(r[k].data, Number) };
      return out;
    },
  };
}

/** Load an .onnx file: {inputNames, outputNames, run({name: rows}) -> {output: rows}, lastMs}.
 *  engine "builtin" (default, onnx.js) or "ort" (onnxruntime-web from the CDN). */
export async function loadGraph(url, { engine = "builtin" } = {}) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  const builtin = new Session(bytes), types = builtin.inputTypes;
  const sess = engine === "ort" ? await ortSession(bytes) : builtin;
  const g = {
    engine, bytes: bytes.byteLength, inputNames: builtin.inputNames, outputNames: builtin.outputNames, lastMs: 0,
    /** feeds: {input: array of rows}. Returns {output: array of rows} (last axis = columns). */
    async run(feeds) {
      const f = {};
      for (const [k, rows] of Object.entries(feeds)) {
        const w = rows[0].length, data = new Float64Array(rows.length * w);
        rows.forEach((row, i) => data.set(row, i * w));
        f[k] = { dims: [rows.length, w], data };
      }
      const t0 = performance.now();
      const out = engine === "ort" ? await sess.run(f, types) : sess.run(f);
      g.lastMs = performance.now() - t0;
      const res = {};
      for (const [k, t] of Object.entries(out)) {
        const w = t.dims.length > 1 ? t.dims[t.dims.length - 1] : 1, n = t.data.length / w;
        res[k] = Array.from({ length: n }, (_, i) => Array.from(t.data.subarray(i * w, (i + 1) * w)));
      }
      return res;
    },
  };
  return g;
}

// ------------------------------------------------------------------ query checks -> verdict
const RANK = { pass: 0, warn: 1, fail: 2 };

export function compare(value, op, limit) {
  if (value === null || value === undefined || !Number.isFinite(value)) return false;
  if (op === "in") return limit[0] <= value && value <= limit[1];
  return { "<": value < limit, "<=": value <= limit, ">": value > limit, ">=": value >= limit }[op];
}

function fmtG(v, p) {                                          // Python format(v, ".pg")
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  const [m, e] = v.toExponential(Math.max(p - 1, 0)).split("e"), exp = Number(e);
  if (exp < -4 || exp >= p) {
    const mm = m.includes(".") ? m.replace(/0+$/, "").replace(/\.$/, "") : m;
    return `${mm}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  const f = v.toFixed(Math.max(p - 1 - exp, 0));
  return f.includes(".") ? f.replace(/0+$/, "").replace(/\.$/, "") : f;
}

/** Python str.format for the templates of the check spec: {name}, {name:.Ng}, {name:.Nf}, {name:.N%}. */
export function formatTemplate(tpl, fields) {
  return tpl.replace(/\{(\w+)(?::\.(\d+)([gf%]))?\}/g, (all, k, d, t) => {
    if (!(k in fields)) return all;
    const v = fields[k];
    if (!t) return String(v);
    const n = Number(d);
    if (t === "f") return v.toFixed(n);
    if (t === "%") return `${(v * 100).toFixed(n)}%`;
    return fmtG(v, n || 1);
  });
}

/** Statuses, details and verdict of one row of check values: the JS twin of pinneapple_hub.zoo.checks.evaluate. */
export function evaluate(spec, values, aux) {
  const fields = Object.fromEntries(spec.aux.map((k, i) => [k, aux[i]]));
  const checks = spec.checks.map((c, i) => {
    const v = values[i], ok = compare(v, c.op, c.limit), lim = c.op === "in" ? c.limit[0] : c.limit;
    return { ...c, value: v, status: ok ? "pass" : c.severity, detail_text: formatTemplate(c.detail, { ...fields, value: v, limit: lim }) };
  });
  return { checks, verdict: verdict(spec, checks) };
}

export function verdict(spec, checks) {
  let worst = "pass";
  for (const c of checks) if ((c.kind || "trust") === "trust" && RANK[c.status] > RANK[worst]) worst = c.status;
  return { status: worst, ...spec.verdicts[worst] };
}

/** Checks merged by group for display: one line per group (worst status, details of the members that fail). */
export function grouped(spec, checks) {
  const out = [], seen = new Map();
  for (const c of checks) {
    if (!c.group) { out.push(c); continue; }
    if (!seen.has(c.group)) {
      const g = { id: c.group, rule: spec.groups[c.group].rule, name: spec.groups[c.group].name, kind: c.kind, members: [], status: "pass" };
      seen.set(c.group, g); out.push(g);
    }
    const g = seen.get(c.group);
    g.members.push(c);
    if (RANK[c.status] > RANK[g.status]) g.status = c.status;
  }
  for (const g of seen.values()) {
    const bad = g.members.filter((m) => m.status !== "pass");
    g.detail_text = bad.length ? bad.map((m) => m.detail_text).join("; ") : `all ${g.members.length} inside the range`;
  }
  return out;
}

// ------------------------------------------------------------------ zoo models
/** A zoo model exported with pp.zoo.export_web: predictions and query checks in one call. */
export async function loadModel(base, name, opts = {}) {
  const meta = await fetch(`${base}/${name}.json`).then((r) => { if (!r.ok) throw new Error(`${name}.json: HTTP ${r.status}`); return r.json(); });
  const graph = await loadGraph(`${base}/${meta.onnx.file}`, opts);
  const outCols = meta.onnx.outputs;
  return {
    meta, spec: meta.checks, graph, inputs: meta.onnx.input_columns,
    get lastMs() { return graph.lastMs; },
    /** rows: inputs in card order; query: one row. -> {rows: [{output columns by name}], query: evaluation} */
    async run(rows, query) {
      const out = await graph.run({ x: rows, query: [query || rows[0]] });
      const named = rows.map((_, i) => {
        const o = {};
        for (const [k, cols] of Object.entries(outCols)) if (Array.isArray(cols)) cols.forEach((c, j) => { o[c] = out[k][i][j]; });
        return o;
      });
      const ev = evaluate(meta.checks, out.checks[0], out.aux[0]);
      return { rows: named, query: { ...ev, groups: grouped(meta.checks, ev.checks) } };
    },
    /** rows: inputs in card order -> [{outputs: {column: value}, ...evaluation of that row's query checks}]
     *  (a sweep: every row is predicted and checked, as pinneapple_hub.zoo.checks.query_checks does). */
    async evaluateRows(rows) {
      const out = await graph.run({ x: rows, query: rows });
      return rows.map((_, i) => {
        const o = {};
        for (const [k, cols] of Object.entries(outCols)) if (Array.isArray(cols)) cols.forEach((c, j) => { o[c] = out[k][i][j]; });
        return { outputs: o, ...evaluate(meta.checks, out.checks[i], out.aux[i]) };
      });
    },
  };
}

// ------------------------------------------------------------------ CST airfoil geometry
/** One row of the geometry graph's outputs -> named properties, validity (as geometry.valid) and handle points. */
export function geometryResult(meta, o) {
  const props = Object.fromEntries(meta.props.map((k, i) => [k, o.props[i]]));
  let valid = [true, ""];
  meta.validity.forEach((k, i) => { const [op, lim, why] = meta.validity_limits[k]; if (valid[0] && !compare(o.validity[i], op, lim)) valid = [false, why]; });
  const hx = meta.handle_x, h = o.handles;
  return { props, valid, upper: o.upper, lower: o.lower, x: meta.outline_x,
    handles: { upper: hx.map((x, i) => [x, h[i]]), lower: hx.map((x, i) => [x, h[3 + i]]) } };
}

/** The CST airfoil geometry graphs (pinneapple_design.aero.geometry_onnx). */
export async function loadAirfoilGeometry(base, opts = {}) {
  const meta = await fetch(`${base}/cst_geometry.json`).then((r) => r.json());
  const [geo, fh] = await Promise.all([loadGraph(`${base}/${meta.files.geometry}`, opts), loadGraph(`${base}/${meta.files.from_handles}`, opts)]);
  return {
    meta,
    /** shape (6 weights) -> {props, valid: [ok, reason], upper, lower, x, handles: {upper: [[x,y]], lower}} */
    async measure(shape) {
      const o = await geo.run({ x: [shape] });
      return geometryResult(meta, Object.fromEntries(Object.entries(o).map(([k, v]) => [k, v[0]])));
    },
    /** upper and lower handle heights (3 + 3) -> shape */
    async fromHandles(yUpper, yLower) { return (await fh.run({ x: [[...yUpper, ...yLower]] })).shape[0]; },
  };
}
