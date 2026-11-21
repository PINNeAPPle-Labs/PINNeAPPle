// PINNeAPPle web runtime: a small, dependency-free ONNX interpreter for the graphs the library exports
// (pinneapple_hub.zoo.export, pinneapple_design.aero.geometry_onnx): MLPs, their pre/post-processing and query checks,
// a few hundred kB of weights and ~30 operators. It reads the .onnx protobuf directly and evaluates in float64.
// onnxruntime-web runs the same files (zoo.js, engine "ort") but is ~10 MB of WebAssembly, too much for a page that
// should load in a second. Parity with onnxruntime is tested in tests/js/zoo_web.test.mjs.

// ------------------------------------------------------------------ protobuf (wire format)
class Reader {
  constructor(buf) { this.b = buf; this.p = 0; this.end = buf.length; }
  varint() {
    let r = 0, s = 1, byte;
    do { byte = this.b[this.p++]; r += (byte & 0x7f) * s; s *= 128; } while (byte & 0x80);
    return r; // int64 values here (dims, attrs, ids) fit in 2^53; negatives come out as 2^64 - x, fixed below
  }
  int64() {
    let r = 0n, s = 0n, byte;
    do { byte = this.b[this.p++]; r |= BigInt(byte & 0x7f) << s; s += 7n; } while (byte & 0x80);
    return Number(BigInt.asIntN(64, r));
  }
  bytes() { const n = this.varint(), s = this.b.subarray(this.p, this.p + n); this.p += n; return s; }
  fixed32() { const v = new DataView(this.b.buffer, this.b.byteOffset + this.p, 4).getFloat32(0, true); this.p += 4; return v; }
  skip(wt) {
    if (wt === 0) this.varint(); else if (wt === 1) this.p += 8; else if (wt === 2) { const n = this.varint(); this.p += n; }
    else if (wt === 5) this.p += 4; else throw new Error(`protobuf wire type ${wt}`);
  }
  *fields() { while (this.p < this.end) { const k = this.varint(); yield [Math.floor(k / 8), k & 7]; } }
}
const sub = (r) => new Reader(r.bytes());
const str = (r) => new TextDecoder().decode(r.bytes());

function packedOr(r, wt, one, out) {
  if (wt === 2) { const s = sub(r); while (s.p < s.end) out.push(one(s)); } else out.push(one(r));
}

const DT = { 1: "float32", 6: "int32", 7: "int64", 9: "bool", 11: "float64" };

function parseTensor(r) {
  const t = { dims: [], dtype: 1, name: "", data: null }, f = [], i64 = [], i32 = [], d = [];
  let raw = null;
  for (const [no, wt] of r.fields()) {
    if (no === 1) packedOr(r, wt, (s) => s.int64(), t.dims);
    else if (no === 2) t.dtype = r.varint();
    else if (no === 4) packedOr(r, wt, (s) => s.fixed32(), f);
    else if (no === 5) packedOr(r, wt, (s) => s.int64(), i32);
    else if (no === 7) packedOr(r, wt, (s) => s.int64(), i64);
    else if (no === 8) t.name = str(r);
    else if (no === 9) raw = r.bytes().slice();
    else if (no === 10) packedOr(r, wt, (s) => { const v = new DataView(s.b.buffer, s.b.byteOffset + s.p, 8).getFloat64(0, true); s.p += 8; return v; }, d);
    else r.skip(wt);
  }
  const n = t.dims.reduce((a, b) => a * b, 1);
  let vals;
  if (raw) {
    const dv = new DataView(raw.buffer);
    vals = new Array(n);
    for (let k = 0; k < n; k++) {
      if (t.dtype === 1) vals[k] = dv.getFloat32(4 * k, true);
      else if (t.dtype === 11) vals[k] = dv.getFloat64(8 * k, true);
      else if (t.dtype === 7) vals[k] = Number(dv.getBigInt64(8 * k, true));
      else if (t.dtype === 6) vals[k] = dv.getInt32(4 * k, true);
      else if (t.dtype === 9) vals[k] = raw[k];
      else throw new Error(`tensor dtype ${t.dtype}`);
    }
  } else vals = f.length ? f : i64.length ? i64 : i32.length ? i32 : d;
  return { ...tensor(t.dims, Float64Array.from(vals), DT[t.dtype] || "float32"), name: t.name };
}

function parseAttr(r) {
  const a = { name: "", ints: [], floats: [] };
  for (const [no, wt] of r.fields()) {
    if (no === 1) a.name = str(r);
    else if (no === 2) a.f = r.fixed32();
    else if (no === 3) a.i = r.int64();
    else if (no === 4) a.s = new TextDecoder().decode(r.bytes());
    else if (no === 5) a.t = parseTensor(sub(r));
    else if (no === 7) packedOr(r, wt, (s) => s.fixed32(), a.floats);
    else if (no === 8) packedOr(r, wt, (s) => s.int64(), a.ints);
    else r.skip(wt);
  }
  return a;
}

function parseNode(r) {
  const n = { inputs: [], outputs: [], op: "", attrs: {} };
  for (const [no, wt] of r.fields()) {
    if (no === 1) n.inputs.push(str(r));
    else if (no === 2) n.outputs.push(str(r));
    else if (no === 4) n.op = str(r);
    else if (no === 5) { const a = parseAttr(sub(r)); n.attrs[a.name] = a; }
    else r.skip(wt);
  }
  return n;
}

function parseValueInfo(r) {
  let name = "", elem = 1;
  for (const [no, wt] of r.fields()) {
    if (no === 1) name = str(r);
    else if (no === 2) {                                      // TypeProto.tensor_type.elem_type
      const t = sub(r);
      for (const [n2, w2] of t.fields()) {
        if (n2 === 1) { const tt = sub(t); for (const [n3, w3] of tt.fields()) { if (n3 === 1) elem = tt.varint(); else tt.skip(w3); } }
        else t.skip(w2);
      }
    } else r.skip(wt);
  }
  return { name, dtype: DT[elem] || "float32" };
}

function parseGraph(r) {
  const g = { nodes: [], init: {}, inputs: [], outputs: [], inputTypes: {} };
  for (const [no, wt] of r.fields()) {
    if (no === 1) g.nodes.push(parseNode(sub(r)));
    else if (no === 5) { const t = parseTensor(sub(r)); g.init[t.name] = t; }
    else if (no === 11) { const v = parseValueInfo(sub(r)); g.inputs.push(v.name); g.inputTypes[v.name] = v.dtype; }
    else if (no === 12) g.outputs.push(parseValueInfo(sub(r)).name);
    else r.skip(wt);
  }
  g.inputs = g.inputs.filter((n) => !(n in g.init));
  return g;
}

export function parseModel(bytes) {
  const r = new Reader(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));
  let graph = null, opset = 0;
  for (const [no, wt] of r.fields()) {
    if (no === 7) graph = parseGraph(sub(r));
    else if (no === 8) {
      const s = sub(r); let dom = "", v = 0;
      for (const [n2, w2] of s.fields()) { if (n2 === 1) dom = str(s); else if (n2 === 2) v = s.int64(); else s.skip(w2); }
      if (dom === "" || dom === "ai.onnx") opset = v;
    } else r.skip(wt);
  }
  if (!graph) throw new Error("no graph in ONNX model");
  return { graph, opset };
}

// ------------------------------------------------------------------ tensors
export function tensor(dims, data, dtype = "float32") { return { dims: [...dims], data, dtype }; }
const size = (d) => d.reduce((a, b) => a * b, 1);
const strides = (d) => { const s = new Array(d.length); let k = 1; for (let i = d.length - 1; i >= 0; i--) { s[i] = k; k *= d[i]; } return s; };
const normAxis = (a, r) => (a < 0 ? a + r : a);

function broadcast(a, b, f) {
  // fast paths: equal shapes, a scalar, or a vector along the last axis (bias / normalisation rows)
  const na = a.data.length, nb = b.data.length;
  if (na >= nb && a.dims.length >= b.dims.length) {
    const last = a.dims[a.dims.length - 1];
    const same = nb === na && a.dims.join() === b.dims.join();
    const row = nb === last && b.dims.slice(0, -1).every((d) => d === 1) && b.dims[b.dims.length - 1] === last;
    if (same || nb === 1 || row) {
      const out = new Float64Array(na), A = a.data, B = b.data;
      if (same) for (let i = 0; i < na; i++) out[i] = f(A[i], B[i]);
      else if (nb === 1) { const v = B[0]; for (let i = 0; i < na; i++) out[i] = f(A[i], v); }
      else for (let i = 0; i < na; i++) out[i] = f(A[i], B[i % last]);
      return tensor(a.dims, out, a.dtype);
    }
  }
  const r = Math.max(a.dims.length, b.dims.length);
  const da = [...Array(r - a.dims.length).fill(1), ...a.dims], db = [...Array(r - b.dims.length).fill(1), ...b.dims];
  const out = da.map((v, i) => {
    if (v !== db[i] && v !== 1 && db[i] !== 1) throw new Error(`broadcast ${a.dims} vs ${b.dims}`);
    return Math.max(v, db[i]);
  });
  const n = size(out), so = strides(out), sa = strides(da), sb = strides(db), res = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let ia = 0, ib = 0, rem = k;
    for (let d = 0; d < r; d++) {
      const c = Math.floor(rem / so[d]); rem -= c * so[d];
      if (da[d] !== 1) ia += c * sa[d];
      if (db[d] !== 1) ib += c * sb[d];
    }
    res[k] = f(a.data[ia], b.data[ib]);
  }
  return tensor(out, res, a.dtype);
}

const unary = (f) => (n, [x]) => [tensor(x.dims, x.data.map(f), x.dtype)];
const binary = (f) => (n, [a, b]) => [broadcast(a, b, f)];
const ints = (t) => Array.from(t.data);

function matmul2(a, b, m, k, n, transB = false) {
  const out = new Float64Array(m * n);
  if (transB) {                                     // b is (n, k): row-by-row dot products
    for (let i = 0; i < m; i++) {
      const ai = i * k;
      for (let j = 0; j < n; j++) {
        const bj = j * k;
        let s = 0;
        for (let p = 0; p < k; p++) s += a[ai + p] * b[bj + p];
        out[i * n + j] = s;
      }
    }
  } else {                                          // b is (k, n): accumulate rows of b
    for (let i = 0; i < m; i++) {
      const oi = i * n;
      for (let p = 0; p < k; p++) {
        const v = a[i * k + p], bp = p * n;
        if (v === 0) continue;
        for (let j = 0; j < n; j++) out[oi + j] += v * b[bp + j];
      }
    }
  }
  return out;
}

function transpose(x, perm) {
  const r = x.dims.length; perm = perm && perm.length ? perm : [...Array(r).keys()].reverse();
  const od = perm.map((p) => x.dims[p]), si = strides(x.dims), so = strides(od), out = new Float64Array(x.data.length);
  for (let k = 0; k < out.length; k++) {
    let rem = k, src = 0;
    for (let d = 0; d < r; d++) { const c = Math.floor(rem / so[d]); rem -= c * so[d]; src += c * si[perm[d]]; }
    out[k] = x.data[src];
  }
  return tensor(od, out, x.dtype);
}

function slice(x, starts, ends, axes, steps) {
  const r = x.dims.length; axes = axes || [...Array(starts.length).keys()];
  const st = new Array(r).fill(0), sp = new Array(r).fill(1), od = [...x.dims];
  axes.forEach((ax0, j) => {
    const ax = normAxis(ax0, r), dim = x.dims[ax], step = steps ? steps[j] : 1;
    if (step !== 1) throw new Error("Slice: only step 1 is supported");
    let s = starts[j], e = ends[j];
    s = s < 0 ? s + dim : s; e = e < 0 ? e + dim : e;
    s = Math.min(Math.max(s, 0), dim); e = Math.min(Math.max(e, 0), dim);
    st[ax] = s; od[ax] = Math.max(e - s, 0); sp[ax] = step;
  });
  const si = strides(x.dims), so = strides(od), out = new Float64Array(size(od));
  for (let k = 0; k < out.length; k++) {
    let rem = k, src = 0;
    for (let d = 0; d < r; d++) { const c = Math.floor(rem / so[d]); rem -= c * so[d]; src += (st[d] + c) * si[d]; }
    out[k] = x.data[src];
  }
  return tensor(od, out, x.dtype);
}

function reduce(x, axes, keep, f, init, fin) {
  const r = x.dims.length; axes = (axes && axes.length ? axes : [...Array(r).keys()]).map((a) => normAxis(a, r));
  const kd = x.dims.map((d, i) => (axes.includes(i) ? 1 : d)), out = new Float64Array(size(kd)).fill(init);
  const si = strides(x.dims), so = strides(kd), cnt = axes.reduce((a, ax) => a * x.dims[ax], 1);
  for (let k = 0; k < x.data.length; k++) {
    let rem = k, dst = 0;
    for (let d = 0; d < r; d++) { const c = Math.floor(rem / si[d]); rem -= c * si[d]; if (kd[d] !== 1) dst += c * so[d]; }
    out[dst] = f(out[dst], x.data[k]);
  }
  return tensor(keep ? kd : kd.filter((_, i) => !axes.includes(i)), fin ? out.map((v) => fin(v, cnt)) : out, x.dtype);
}

const attr = (n, name, dflt) => { const a = n.attrs[name]; if (!a) return dflt; if (a.ints.length) return a.ints; if (a.floats.length) return a.floats; return a.i ?? a.f ?? a.s ?? a.t ?? dflt; };
const listAttr = (n, name) => { const a = n.attrs[name]; return a ? (a.ints.length ? a.ints : a.i !== undefined ? [a.i] : []) : null; };

const OPS = {
  Identity: (n, [x]) => [x],
  Constant: (n) => [attr(n, "value")],
  Add: binary((a, b) => a + b), Sub: binary((a, b) => a - b), Mul: binary((a, b) => a * b), Div: binary((a, b) => a / b),
  Pow: binary((a, b) => a ** b), Max: binary(Math.max), Min: binary(Math.min),
  Sigmoid: unary((v) => 1 / (1 + Math.exp(-v))), Tanh: unary(Math.tanh), Sin: unary(Math.sin), Cos: unary(Math.cos),
  Exp: unary(Math.exp), Log: unary(Math.log), Sqrt: unary(Math.sqrt), Neg: unary((v) => -v), Abs: unary(Math.abs),
  Relu: unary((v) => Math.max(v, 0)), Softplus: unary((v) => Math.log1p(Math.exp(v))),
  Clip(n, [x, lo, hi]) {
    const a = lo ? lo.data[0] : attr(n, "min", -Infinity), b = hi ? hi.data[0] : attr(n, "max", Infinity);
    return [tensor(x.dims, x.data.map((v) => Math.min(Math.max(v, a), b)), x.dtype)];
  },
  Cast: (n, [x]) => [tensor(x.dims, x.data, DT[attr(n, "to", 1)] || "float32")],
  Gemm(n, [a, b, c]) {
    if (attr(n, "transA", 0)) a = transpose(a);
    const tb = attr(n, "transB", 0), al = attr(n, "alpha", 1), be = attr(n, "beta", 1);
    const [m, k] = a.dims, nn = tb ? b.dims[0] : b.dims[1];
    let y = tensor([m, nn], matmul2(a.data, b.data, m, k, nn, tb), a.dtype);
    if (al !== 1) y.data = y.data.map((v) => v * al);
    if (c) y = broadcast(y, be === 1 ? c : tensor(c.dims, c.data.map((v) => v * be)), (p, q) => p + q);
    return [y];
  },
  MatMul(n, [a, b]) {
    if (a.dims.length === 1) a = tensor([1, a.dims[0]], a.data, a.dtype);
    if (b.dims.length === 2 && a.dims.length >= 2) {
      const k = a.dims[a.dims.length - 1], m = a.data.length / k, nn = b.dims[1];
      return [tensor([...a.dims.slice(0, -1), nn], matmul2(a.data, b.data, m, k, nn), a.dtype)];
    }
    // batched: (..., m, k) x (..., k, n) with equal leading dims or a 2-D left operand
    const k = b.dims[b.dims.length - 2], nn = b.dims[b.dims.length - 1], bs = b.data.length / (k * nn);
    const m = a.dims[a.dims.length - 2], as = a.data.length / (m * k), out = new Float64Array(bs * m * nn);
    for (let q = 0; q < bs; q++) {
      const A = a.data.subarray((as === 1 ? 0 : q) * m * k, ((as === 1 ? 0 : q) + 1) * m * k);
      out.set(matmul2(A, b.data.subarray(q * k * nn, (q + 1) * k * nn), m, k, nn), q * m * nn);
    }
    return [tensor([...b.dims.slice(0, -2), m, nn], out, a.dtype)];
  },
  Concat(n, xs) {
    xs = xs.filter((x) => x && size(x.dims) > 0);
    const r = xs[0].dims.length, ax = normAxis(attr(n, "axis", 0), r);
    const od = [...xs[0].dims]; od[ax] = xs.reduce((s, x) => s + x.dims[ax], 0);
    const outer = size(od.slice(0, ax)), out = new Float64Array(size(od));
    let p = 0;
    for (let o = 0; o < outer; o++) for (const x of xs) {
      const blk = size(x.dims.slice(ax));
      out.set(x.data.subarray(o * blk, (o + 1) * blk), p); p += blk;
    }
    return [tensor(od, out, xs[0].dtype)];
  },
  Unsqueeze(n, [x, a]) {
    const axes = a ? ints(a) : listAttr(n, "axes"), r = x.dims.length + axes.length, d = [...x.dims];
    axes.map((v) => normAxis(v, r)).sort((p, q) => p - q).forEach((v) => d.splice(v, 0, 1));
    return [tensor(d, x.data, x.dtype)];
  },
  Squeeze(n, [x, a]) {
    const axes = (a ? ints(a) : listAttr(n, "axes")) || x.dims.map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0);
    const ax = axes.map((v) => normAxis(v, x.dims.length));
    return [tensor(x.dims.filter((_, i) => !ax.includes(i)), x.data, x.dtype)];
  },
  Reshape(n, [x, s]) {
    const shp = ints(s).map((v, i) => (v === 0 ? x.dims[i] : v)), known = shp.filter((v) => v !== -1).reduce((a, b) => a * b, 1);
    return [tensor(shp.map((v) => (v === -1 ? x.data.length / known : v)), x.data, x.dtype)];
  },
  Flatten(n, [x]) { const ax = normAxis(attr(n, "axis", 1), x.dims.length); return [tensor([size(x.dims.slice(0, ax)), size(x.dims.slice(ax))], x.data, x.dtype)]; },
  Transpose: (n, [x]) => [transpose(x, listAttr(n, "perm"))],
  Shape: (n, [x]) => [tensor([x.dims.length], Float64Array.from(x.dims), "int64")],
  Gather(n, [x, idx]) {
    const ax = normAxis(attr(n, "axis", 0), x.dims.length), outer = size(x.dims.slice(0, ax)), inner = size(x.dims.slice(ax + 1));
    const ids = ints(idx).map((i) => (i < 0 ? i + x.dims[ax] : i)), out = new Float64Array(outer * ids.length * inner);
    let p = 0;
    for (let o = 0; o < outer; o++) for (const i of ids) { out.set(x.data.subarray((o * x.dims[ax] + i) * inner, (o * x.dims[ax] + i + 1) * inner), p); p += inner; }
    return [tensor([...x.dims.slice(0, ax), ...idx.dims, ...x.dims.slice(ax + 1)], out, x.dtype)];
  },
  Slice(n, [x, s, e, a, st]) {
    if (s) return [slice(x, ints(s), ints(e), a && ints(a), st && ints(st))];
    return [slice(x, listAttr(n, "starts"), listAttr(n, "ends"), listAttr(n, "axes"))];
  },
  ReduceMin: (n, [x, a]) => [reduce(x, a ? ints(a) : listAttr(n, "axes"), attr(n, "keepdims", 1), Math.min, Infinity)],
  ReduceMax: (n, [x, a]) => [reduce(x, a ? ints(a) : listAttr(n, "axes"), attr(n, "keepdims", 1), Math.max, -Infinity)],
  ArgMax(n, [x]) {                                            // first maximum, as numpy / torch
    const r = x.dims.length, ax = normAxis(attr(n, "axis", 0), r), keep = attr(n, "keepdims", 1);
    if (attr(n, "select_last_index", 0)) throw new Error("ArgMax: select_last_index is not supported");
    const outer = size(x.dims.slice(0, ax)), len = x.dims[ax], inner = size(x.dims.slice(ax + 1));
    const out = new Float64Array(outer * inner);
    for (let o = 0; o < outer; o++) for (let i = 0; i < inner; i++) {
      let best = 0, bv = -Infinity;
      for (let k = 0; k < len; k++) { const v = x.data[(o * len + k) * inner + i]; if (v > bv) { bv = v; best = k; } }
      out[o * inner + i] = best;
    }
    const d = [...x.dims]; if (keep) d[ax] = 1; else d.splice(ax, 1);
    return [tensor(d, out, "int64")];
  },
  GatherElements(n, [x, idx]) {
    const r = x.dims.length, ax = normAxis(attr(n, "axis", 0), r), si = strides(x.dims), so = strides(idx.dims);
    const out = new Float64Array(idx.data.length);
    for (let k = 0; k < out.length; k++) {
      let rem = k, src = 0;
      for (let d = 0; d < r; d++) {
        const c = Math.floor(rem / so[d]); rem -= c * so[d];
        let j = d === ax ? idx.data[k] : c; if (j < 0) j += x.dims[d];
        src += j * si[d];
      }
      out[k] = x.data[src];
    }
    return [tensor(idx.dims, out, x.dtype)];
  },
  Acos: unary(Math.acos), Floor: unary(Math.floor),
  ConstantOfShape(n, [shp]) {
    const v = attr(n, "value"), d = ints(shp);
    return [tensor(d, new Float64Array(size(d)).fill(v ? v.data[0] : 0), v ? v.dtype : "float32")];
  },
  Expand(n, [x, shp]) {
    const d = ints(shp), r = Math.max(d.length, x.dims.length), pad = (v) => [...Array(r - v.length).fill(1), ...v];
    const xd = pad(x.dims), od = pad(d).map((v, i) => Math.max(v, xd[i]));
    return [broadcast(tensor(od, new Float64Array(size(od)), x.dtype), tensor(xd, x.data, x.dtype), (p, q) => q)];
  },
  Range(n, [s, l, dl]) {
    const a = s.data[0], b = l.data[0], st = dl.data[0], k = Math.max(Math.ceil((b - a) / st), 0);
    return [tensor([k], Float64Array.from({ length: k }, (_, i) => a + i * st), s.dtype)];
  },
  Where(n, [c, a, b]) {
    const r = Math.max(c.dims.length, a.dims.length, b.dims.length);
    const pad = (d) => [...Array(r - d.length).fill(1), ...d], dc = pad(c.dims), da = pad(a.dims), db = pad(b.dims);
    const od = dc.map((v, i) => Math.max(v, da[i], db[i])), so = strides(od), out = new Float64Array(size(od));
    const at = (t, d, k) => { let rem = k, idx = 0; const st = strides(d); for (let i = 0; i < r; i++) { const q = Math.floor(rem / so[i]); rem -= q * so[i]; if (d[i] !== 1) idx += q * st[i]; } return t.data[idx]; };
    for (let k = 0; k < out.length; k++) out[k] = at(c, dc, k) ? at(a, da, k) : at(b, db, k);
    return [tensor(od, out, a.dtype)];
  },
  Equal: binary((a, b) => (a === b ? 1 : 0)), Less: binary((a, b) => (a < b ? 1 : 0)), Greater: binary((a, b) => (a > b ? 1 : 0)),
  ReduceMean: (n, [x, a]) => [reduce(x, a ? ints(a) : listAttr(n, "axes"), attr(n, "keepdims", 1), (p, q) => p + q, 0, (v, c) => v / c)],
  ReduceSum: (n, [x, a]) => [reduce(x, a ? ints(a) : listAttr(n, "axes"), attr(n, "keepdims", 1), (p, q) => p + q, 0)],
};

export const SUPPORTED_OPS = Object.keys(OPS);

// ------------------------------------------------------------------ session
export class Session {
  constructor(bytes) {
    const { graph, opset } = parseModel(bytes);
    this.graph = graph; this.opset = opset;
    const missing = [...new Set(graph.nodes.map((n) => n.op))].filter((op) => !OPS[op]);
    if (missing.length) throw new Error(`unsupported ONNX ops: ${missing.join(", ")}`);
    this.inputNames = graph.inputs; this.outputNames = graph.outputs; this.inputTypes = graph.inputTypes;
  }
  static async fromUrl(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return new Session(new Uint8Array(await r.arrayBuffer()));
  }
  /** feeds: {name: {dims, data}} -> {output name: {dims, data (Float64Array)}} */
  run(feeds) {
    const env = { ...this.graph.init };
    for (const [k, v] of Object.entries(feeds)) env[k] = tensor(v.dims, Float64Array.from(v.data));
    for (const n of this.graph.nodes) {
      const outs = OPS[n.op](n, n.inputs.map((i) => (i === "" ? undefined : env[i])));
      n.outputs.forEach((o, i) => { env[o] = outs[i]; });
    }
    return Object.fromEntries(this.outputNames.map((o) => [o, env[o]]));
  }
}
