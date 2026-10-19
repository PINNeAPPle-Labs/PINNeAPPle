// PINNeAPPle studio core: colour scales, colour bars, field colouring, streamline tubes, slice textures and edge
// overlays shared by the studio viewer (viewer.js) and the apps (served at /studio/studio-core.js).
// No import of three.js: the functions that build three.js objects take it as their first argument, so the pure
// part runs in node (tests/js/studio_core.test.mjs).

// CFD convention: jet, blue = low, red = high. Stops: [position 0..1, [r, g, b] 0..255].
export const JET = [[0, [0, 0, 143]], [0.125, [0, 0, 255]], [0.375, [0, 255, 255]], [0.625, [255, 255, 0]], [0.875, [255, 0, 0]], [1, [128, 0, 0]]];

// diverging, neutral midpoint (use a range symmetric about zero): Moreland's coolwarm, plain blue-white-red and
// ColorBrewer RdBu reversed; the same stops as pinneapple_tools.visualization.studio.compose
export const COOLWARM = [[0, [59, 76, 192]], [0.25, [141, 176, 254]], [0.5, [221, 221, 221]], [0.75, [244, 154, 123]], [1, [180, 4, 38]]];
export const BWR = [[0, [0, 0, 255]], [0.5, [255, 255, 255]], [1, [255, 0, 0]]];
export const RDBU = [[0, [33, 102, 172]], [0.25, [103, 169, 207]], [0.5, [247, 247, 247]], [0.75, [239, 138, 98]], [1, [178, 24, 43]]];

// errors: sequential, white = no error, dark red = largest (YlOrRd); used for |prediction - reference|
export const ERR = [[0, [255, 255, 229]], [0.2, [255, 237, 160]], [0.4, [254, 178, 76]], [0.6, [252, 78, 42]], [0.8, [189, 0, 38]], [1, [103, 0, 13]]];

export function ramp(stops, t) {
  t = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
  for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], f = (t - a) / ((b - a) || 1);
    return ca.map((c, k) => (c + f * (cb[k] - c)) / 255);
  }
  return stops[stops.length - 1][1].map((c) => c / 255);
}
export const jet = (t) => ramp(JET, t);
export const lin = (c) => Math.pow(c, 2.2);              // vertex colours are linear in three.js

// percentile q (0..1) of the finite values
export function pct(vals, q) {
  const a = Array.from(vals).filter((v) => v !== null && Number.isFinite(v)).sort((x, y) => x - y);
  return a.length ? a[Math.min(a.length - 1, Math.max(0, Math.round(q * (a.length - 1))))] : 0;
}
// colour range: "auto" clips 1 % at each end (outliers do not wash the scale out), "full" is min..max
export function autoRange(vals, mode = "auto") {
  const q = mode === "full" ? 0 : 0.01;
  let lo = pct(vals, q), hi = pct(vals, 1 - q);
  if (hi === lo) { const d = Math.abs(lo) * 0.01 || 1; lo -= d; hi += d; }
  return [lo, hi];
}

// decimals that tell the five ticks apart
export function decimals(lo, hi) {
  const s = Math.abs(hi - lo) || Math.abs(hi) || 1;
  return Math.max(0, Math.min(6, 2 - Math.floor(Math.log10(s))));
}
export function fmt(v, d) {
  if (v === 0) return d > 3 ? "0" : (0).toFixed(d);
  return Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3 ? v.toExponential(2) : v.toFixed(d);
}
export function ticks(lo, hi, n = 5, nd = null) {
  const d = nd === null ? decimals(lo, hi) : nd;
  return Array.from({ length: n }, (_, k) => fmt(lo + ((hi - lo) * k) / (n - 1), d));
}

// colour bar as HTML (inline styles, works in any page): title, gradient, ticks, optional words under the ends
export function colorbarHTML({ title = "", lo = 0, hi = 1, nd = null, loTxt = "", hiTxt = "", stops = JET, width = 240 } = {}) {
  const g = Array.from({ length: 21 }, (_, k) => `rgb(${ramp(stops, k / 20).map((v) => Math.round(v * 255))}) ${k * 5}%`).join(",");
  const row = (cells, extra = "") => `<div style="display:flex;justify-content:space-between;gap:10px;font-size:12px;margin-top:2px${extra}">${cells}</div>`;
  return `<div class="sc-title" style="font-weight:600">${title}</div>`
    + `<div class="sc-bar" style="height:12px;width:${width}px;max-width:70vw;border-radius:3px;margin-top:5px;background:linear-gradient(90deg,${g})"></div>`
    + row(ticks(lo, hi, 5, nd).map((t) => `<span>${t}</span>`).join(""))
    + (loTxt || hiTxt ? row(`<span>${loTxt}</span><span>${hiTxt}</span>`, ";opacity:.75") : "");
}

// per-vertex colours (linear, for vertexColors) of the values on the scale lo..hi
export function colorArray(vals, lo, hi, stops = JET) {
  const c = new Float32Array(vals.length * 3), span = (hi - lo) || 1;
  for (let i = 0; i < vals.length; i++) {
    const rgb = ramp(stops, (vals[i] - lo) / span);
    c[3 * i] = lin(rgb[0]); c[3 * i + 1] = lin(rgb[1]); c[3 * i + 2] = lin(rgb[2]);
  }
  return c;
}
export const colorAttribute = (THREE, vals, lo, hi, stops = JET) => new THREE.BufferAttribute(colorArray(vals, lo, hi, stops), 3);

// RGBA pixels of a slice grid (rows along v, null/NaN = transparent); mirror "even" | "odd" adds the other half of a
// half-model slice on the left (odd: the value changes sign, e.g. streamwise vorticity)
export function sliceImage(grid, lo, hi, mirror = null, stops = JET, alpha = 235) {
  const nv = grid.length, nu = grid[0].length, w = mirror ? 2 * nu : nu, data = new Uint8ClampedArray(4 * w * nv);
  const put = (col, row, v) => {
    const o = 4 * ((nv - 1 - row) * w + col);
    if (v === null || v === undefined || !Number.isFinite(v)) return;
    const rgb = ramp(stops, (v - lo) / ((hi - lo) || 1));
    data[o] = rgb[0] * 255; data[o + 1] = rgb[1] * 255; data[o + 2] = rgb[2] * 255; data[o + 3] = alpha;
  };
  for (let r = 0; r < nv; r++) for (let k = 0; k < nu; k++) {
    const v = grid[r][k];
    if (mirror) { put(nu + k, r, v); put(nu - 1 - k, r, v === null || mirror !== "odd" ? v : -v); } else put(k, r, v);
  }
  return { width: w, height: nv, data };
}
export function sliceTexture(THREE, grid, lo, hi, mirror = null, stops = JET) {
  const im = sliceImage(grid, lo, hi, mirror, stops);
  const cv = document.createElement("canvas"); cv.width = im.width; cv.height = im.height;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(im.width, im.height);
  img.data.set(im.data); ctx.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; t.minFilter = THREE.LinearFilter;
  return t;
}
// textured quad origin + s u + t v (corners already mapped to three.js by toV3)
export function sliceMesh(THREE, origin, u, v, texture, toV3) {
  const P = [origin, origin.map((x, k) => x + u[k]), origin.map((x, k) => x + u[k] + v[k]), origin.map((x, k) => x + v[k])].map(toV3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(P.flatMap((p) => p.toArray()), 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2)); geo.setIndex([0, 1, 2, 0, 2, 3]);
  const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: texture, transparent: true, side: THREE.DoubleSide, toneMapped: false, depthWrite: false }));
  m.renderOrder = 2;
  return m;
}

// slices of one group form a stack; sorted along the plane normal, so a slider walks through them in order
export function sliceStacks(slices) {
  const out = {};
  for (const s of slices) (out[s.group || s.name] = out[s.group || s.name] || []).push(s);
  for (const k of Object.keys(out)) {
    const s0 = out[k][0], n = cross(s0.u, s0.v);
    out[k].sort((a, b) => dot(a.origin, n) - dot(b.origin, n));
  }
  return out;
}
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// a traced line that stops (e.g. at the end of the box) repeats its last point: drop the repeats
export function dropRepeats(P, S = null, eps = 1e-9) {
  const keep = P.map((p, k) => k === 0 || Math.hypot(p[0] - P[k - 1][0], p[1] - P[k - 1][1], p[2] - P[k - 1][2]) > eps);
  return [P.filter((_, k) => keep[k]), S ? S.filter((_, k) => keep[k]) : null];
}
// indices of the lines to draw at density 0..1 (1 = all): evenly spread, never empty
export function thin(n, density) {
  if (n === 0) return [];
  const k = Math.max(1, Math.round(n * Math.min(1, Math.max(0, density))));
  return Array.from({ length: k }, (_, i) => Math.min(n - 1, Math.floor((i * n) / k)));
}

// streamlines as tubes coloured by their values (lines: arrays of points, values: matching arrays or null).
// Returns a Group whose children carry userData {index, curve}; opts: lo, hi, radius, toV3, stops, eps.
export function tubeLines(THREE, lines, values, opts) {
  const { lo = 0, hi = 1, radius, toV3, stops = JET, eps = 1e-9 } = opts;
  const g = new THREE.Group();
  const plain = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0ea5e9, emissiveIntensity: 0.85, roughness: 0.4 });
  const vc = new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false });
  lines.forEach((P0, li) => {
    const [P, S] = dropRepeats(P0, values ? values[li] : null, eps);
    if (P.length < 4) return;
    const curve = new THREE.CatmullRomCurve3(P.map(toV3)), seg = Math.min(500, P.length * 2), rs = 6;
    const geo = new THREE.TubeGeometry(curve, seg, radius, rs, false);
    if (S) {
      const v = new Float32Array(geo.attributes.position.count);
      for (let i = 0; i <= seg; i++) {
        const f = (i / seg) * (S.length - 1), k = Math.min(S.length - 2, Math.floor(f)), s = S[k] + (S[k + 1] - S[k]) * (f - k);
        for (let j = 0; j <= rs; j++) v[i * (rs + 1) + j] = s;
      }
      geo.setAttribute("color", colorAttribute(THREE, v, lo, hi, stops));
      geo.userData.vals = v;                              // recolourLines() reuses them for a new range
    }
    const m = new THREE.Mesh(geo, S ? vc : plain);
    m.userData = { index: li, curve };
    g.add(m);
  });
  return g;
}

// new colour range for tubes built by tubeLines
export function recolourLines(THREE, group, lo, hi, stops = JET) {
  group.children.forEach((m) => { const v = m.geometry && m.geometry.userData.vals; if (v) m.geometry.setAttribute("color", colorAttribute(THREE, v, lo, hi, stops)); });
}

// edge overlay of a mesh: "feature" (creases above 30°), "wire" (every triangle edge) or "off"
export function setEdges(THREE, mesh, mode, color = 0x1e293b) {
  if (mesh.userData.edges) { mesh.remove(mesh.userData.edges); mesh.userData.edges.geometry.dispose(); mesh.userData.edges = null; }
  if (!mode || mode === "off") return;
  const geo = mode === "wire" ? new THREE.WireframeGeometry(mesh.geometry) : new THREE.EdgesGeometry(mesh.geometry, 30);
  const e = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: mode === "wire" ? 0.35 : 0.8 }));
  e.renderOrder = 1;
  mesh.add(e); mesh.userData.edges = e;
}

// ---------------------------------------------------------------- animation helpers (scene.json "animation")
// fractional frame index of time t in the increasing times (clamped to the ends)
export function frameAt(times, t) {
  const n = times.length;
  if (n < 2 || t <= times[0]) return 0;
  if (t >= times[n - 1]) return n - 1;
  let a = 0, b = n - 1;
  while (b - a > 1) { const m = (a + b) >> 1; if (times[m] <= t) a = m; else b = m; }
  return a + (t - times[a]) / ((times[b] - times[a]) || 1);
}
// the frame pair and weight of a fractional index: [k0, k1, w]
export function bracket(f, n) {
  const k0 = Math.max(0, Math.min(n - 1, Math.floor(f))), k1 = Math.min(n - 1, k0 + 1);
  return [k0, k1, k1 === k0 ? 0 : f - k0];
}
export const lerp = (a, b, w) => (a === null || b === null ? (w < 0.5 ? a : b) : a + (b - a) * w);
export const lerp3 = (a, b, w) => [lerp(a[0], b[0], w), lerp(a[1], b[1], w), lerp(a[2], b[2], w)];
// scene axes -> three.js as a matrix C (rows); a rotation R in scene axes is C R C^T in three.js
export const AXES_MATRIX = { z_up: [[1, 0, 0], [0, 0, 1], [0, -1, 0]], aircraft: [[0, 1, 0], [0, 0, 1], [1, 0, 0]], as_is: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] };
// quaternion [w, x, y, z] in scene axes -> three.js order [x, y, z, w]: C is a proper rotation, so C R C^T turns
// about the axis C a by the same angle, i.e. the vector part maps like a point
export function quatToThree(q, axes = "z_up") {
  const C = AXES_MATRIX[axes] || AXES_MATRIX.z_up, v = [q[1], q[2], q[3]];
  const r = C.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
  return [r[0], r[1], r[2], q[0]];
}
// one head-up line: "label: value unit" with the entry's decimals
export function hudLine(h, v) {
  const s = v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(h.nd ?? 1);
  return `${h.label}: ${s}${h.unit ? " " + h.unit : ""}`;
}

// colour map stops by name ("jet", "coolwarm", "bwr", "rdbu"; anything else: jet)
export const CMAPS = { jet: JET, coolwarm: COOLWARM, bwr: BWR, rdbu: RDBU };
export const cmapStops = (name) => CMAPS[name] || JET;

// first point index of a trail shown at frame `last` (its last `seconds`; null: all) and the opacity factor of point
// i: 1 at the current time t, falling linearly to 0 at the tail end when `fade`
export function trailWindow(times, last, seconds = null, fade = false, t = null) {
  const tn = t === null ? times[last] : t;
  let j0 = 0;
  if (seconds !== null && seconds !== undefined) while (j0 < last && times[j0] < tn - seconds) j0++;
  const span = seconds ?? ((tn - times[j0]) || 1);
  return [j0, (i) => (fade ? Math.min(1, Math.max(0, 1 - (tn - times[i]) / span)) : 1)];
}
