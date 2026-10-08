// PINNeAPPle studio viewer: any scene exported by pinneapple_tools.visualization.studio.web_viewer.
// scene.glb carries the surfaces (fields as vertex attributes _NAME), scene.json the lines, slices and labels.
// CFD colours: jet scale, blue = low, red = high, with a colour bar.
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { GLTFLoader } from "./vendor/three/loaders/GLTFLoader.js";
import { RoomEnvironment } from "./vendor/three/environments/RoomEnvironment.js";

const JET = [[0, [0, 0, 143]], [0.125, [0, 0, 255]], [0.375, [0, 255, 255]], [0.625, [255, 255, 0]], [0.875, [255, 0, 0]], [1, [128, 0, 0]]];
function jet(t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < JET.length; i++) if (t <= JET[i][0]) {
    const [a, ca] = JET[i - 1], [b, cb] = JET[i], f = (t - a) / (b - a);
    return ca.map((c, k) => (c + f * (cb[k] - c)) / 255);
  }
  return JET[JET.length - 1][1].map((c) => c / 255);
}
const lin = (c) => Math.pow(c, 2.2);                     // vertex colours are linear in three.js
const AX = { z_up: (p) => new THREE.Vector3(p[0], p[2], -p[1]), aircraft: (p) => new THREE.Vector3(p[1], p[2], p[0]) };
const fmt = (v, d) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d));
function pct(vals, q) { const a = Array.from(vals).filter(Number.isFinite).sort((x, y) => x - y); return a.length ? a[Math.min(a.length - 1, Math.max(0, Math.round(q * (a.length - 1))))] : 0; }
function decimals(lo, hi) { const s = Math.abs(hi - lo) || Math.abs(hi) || 1; return Math.max(0, Math.min(6, 2 - Math.floor(Math.log10(s)))); }

const $ = (s) => document.querySelector(s);
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 0.9;
renderer.outputColorSpace = THREE.SRGBColorSpace;
$("#stage").prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe9edf2);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
const sun = new THREE.DirectionalLight(0xffffff, 1.6); scene.add(sun);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.6));
const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 1e6);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;

let META = {}, label = (f) => f, model = null, centre = new THREE.Vector3(), radius = 1, mode = "real", fieldName = null;
const lineGroup = new THREE.Group(), sliceGroup = new THREE.Group(); scene.add(lineGroup, sliceGroup);
const meshes = [];

function colorbar(title, lo, hi, loTxt = "", hiTxt = "") {
  const g = Array.from({ length: 21 }, (_, k) => `rgb(${jet(k / 20).map((v) => Math.round(v * 255))}) ${k * 5}%`).join(",");
  const d = decimals(lo, hi);
  $("#legend").innerHTML = `<b>${title}</b><div class="bar" style="background:linear-gradient(90deg,${g})"></div>`
    + `<div class="ticks">${[0, 1, 2, 3, 4].map((k) => `<span>${fmt(lo + (hi - lo) * k / 4, d)}</span>`).join("")}</div>`
    + (loTxt || hiTxt ? `<div class="ticks dim"><span>${loTxt}</span><span>${hiTxt}</span></div>` : "");
  $("#legend").style.display = "block";
}

function fit(view = "iso") {
  // scene axes x (length / flow), y, z up -> three.js (x, z, -y); the aircraft convention is handled by AX
  const d = (META.axes === "aircraft"
    ? { iso: [-1.1, 0.75, -1], front: [0, 0.12, -1], top: [0, 1, 0.001], side: [-1, 0.12, 0], back: [0, 0.12, 1] }
    : { iso: [-1, 0.75, 1.1], front: [-1, 0.12, 0], top: [0.001, 1, 0], side: [0, 0.12, 1], back: [1, 0.12, 0] })[view];
  const v = new THREE.Vector3(...d).normalize();
  const dist = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05;
  camera.position.copy(centre).addScaledVector(v, dist); camera.near = dist / 1000; camera.far = dist * 100; camera.updateProjectionMatrix();
  controls.target.copy(centre); controls.update();
  sun.position.copy(centre).add(new THREE.Vector3(-1, 2, 1.2).multiplyScalar(radius * 3));
}

function setMode(m, arg) {
  mode = m; if (m === "field") fieldName = arg;
  document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("on", b.dataset.m === m && (m !== "field" || b.dataset.f === arg)));
  $("#legend").style.display = "none";
  lineGroup.visible = m === "lines"; sliceGroup.visible = m === "slice";
  sliceGroup.children.forEach((c) => (c.visible = m === "slice" && c.name === arg));
  for (const o of meshes) {
    if (m === "real" || m === "lines") { o.material = o.userData.mat; continue; }
    if (m === "slice") { o.material = new THREE.MeshStandardMaterial({ color: 0xb8c0c8, transparent: true, opacity: 0.3, depthWrite: false }); continue; }
    const a = o.geometry.getAttribute("_" + arg.toLowerCase());
    if (!a) { o.material = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.6 }); continue; }
    const [lo, hi] = META.ranges[arg];
    const c = new Float32Array(a.count * 3);
    for (let i = 0; i < a.count; i++) { const rgb = jet((a.array[i] - lo) / ((hi - lo) || 1)); c[3 * i] = lin(rgb[0]); c[3 * i + 1] = lin(rgb[1]); c[3 * i + 2] = lin(rgb[2]); }
    o.geometry.setAttribute("color", new THREE.BufferAttribute(c, 3));
    o.material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, metalness: 0, envMapIntensity: 0.4 });
  }
  if (m === "field") { const [lo, hi] = META.ranges[arg]; colorbar((META.labels || {})[arg] || label(arg), lo, hi); }
  if (m === "lines" && META.lineRange) colorbar(META.lineLabel || "value along the lines", META.lineRange[0], META.lineRange[1]);
  if (m === "slice") { const s = META.slices.find((x) => x.name === arg); colorbar(s.label || s.name, s.range[0], s.range[1]); }
}

function buildLines(ax) {
  const sets = META.lines || []; if (!sets.length) return;
  const all = sets.flatMap((L) => (L.values || []).flat());
  const lo = all.length ? pct(all, 0.02) : 0, hi = all.length ? pct(all, 0.98) : 1;
  META.lineRange = all.length ? [lo, hi] : null; META.lineLabel = (sets.find((L) => L.label) || {}).label;
  const rad = radius * 0.0035;
  const plain = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0ea5e9, emissiveIntensity: 0.8 });
  const vc = new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false });
  for (const L of sets) L.points.forEach((P0, li) => {
    const keep = P0.map((p, k) => k === 0 || Math.hypot(p[0] - P0[k - 1][0], p[1] - P0[k - 1][1], p[2] - P0[k - 1][2]) > 1e-9);
    const P = P0.filter((_, k) => keep[k]); if (P.length < 4) return;
    const S = L.values ? L.values[li].filter((_, k) => keep[k]) : null;
    const curve = new THREE.CatmullRomCurve3(P.map(ax)), seg = Math.min(500, P.length * 2), rs = 6;
    const geo = new THREE.TubeGeometry(curve, seg, rad, rs, false);
    if (S) {
      const col = new Float32Array(geo.attributes.position.count * 3);
      for (let i = 0; i <= seg; i++) {
        const f = (i / seg) * (S.length - 1), k = Math.min(S.length - 2, Math.floor(f)), v = S[k] + (S[k + 1] - S[k]) * (f - k);
        const rgb = jet((v - lo) / ((hi - lo) || 1));
        for (let j = 0; j <= rs; j++) { const o = 3 * (i * (rs + 1) + j); col[o] = lin(rgb[0]); col[o + 1] = lin(rgb[1]); col[o + 2] = lin(rgb[2]); }
      }
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    }
    lineGroup.add(new THREE.Mesh(geo, S ? vc : plain));
  });
}

function buildSlices(ax) {
  for (const s of META.slices || []) {
    const g = s.grid, nv = g.length, nu = g[0].length, vals = g.flat().filter((v) => v !== null);
    s.range = [pct(vals, 0.02), pct(vals, 0.98)];
    const cv = document.createElement("canvas"); cv.width = nu; cv.height = nv;
    const ctx = cv.getContext("2d"), img = ctx.createImageData(nu, nv);
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      const v = g[j][i], o = 4 * ((nv - 1 - j) * nu + i);
      if (v === null) { img.data[o + 3] = 0; continue; }
      const rgb = jet((v - s.range[0]) / ((s.range[1] - s.range[0]) || 1));
      img.data[o] = rgb[0] * 255; img.data[o + 1] = rgb[1] * 255; img.data[o + 2] = rgb[2] * 255; img.data[o + 3] = 235;
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
    const O = s.origin, U = s.u, V = s.v, P = [O, O.map((x, k) => x + U[k]), O.map((x, k) => x + U[k] + V[k]), O.map((x, k) => x + V[k])].map(ax);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(P.flatMap((p) => p.toArray()), 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2)); geo.setIndex([0, 1, 2, 0, 2, 3]);
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, toneMapped: false, depthWrite: false }));
    m.name = s.name; m.renderOrder = 2; sliceGroup.add(m);
  }
}

async function main() {
  META = await (await fetch("scene.json")).json();
  document.title = META.title || "PINNeAPPle"; $("#title").textContent = META.title || "";
  const gltf = await new GLTFLoader().loadAsync("scene.glb");
  model = gltf.scene; scene.add(model);
  const fields = new Set(); META.ranges = {};
  model.traverse((o) => { if (o.isMesh) { o.userData.mat = o.material; meshes.push(o); Object.keys(o.geometry.attributes).filter((k) => k.startsWith("_")).forEach((k) => fields.add(k.slice(1))); } });
  for (const f of fields) {
    const all = meshes.flatMap((o) => { const a = o.geometry.getAttribute("_" + f); return a ? Array.from(a.array) : []; });
    META.ranges[f] = [pct(all, 0.01), pct(all, 0.99)];
  }
  // three.js lowercases attribute names: show the original field name, and its label in the colour bar
  label = (f) => {
    const k = Object.keys(META.labels || {}).find((x) => x.toLowerCase() === f); if (k) META.labels[f] = META.labels[k];
    return (META.fields || []).find((x) => x.toLowerCase() === f) || k || f;
  };
  const box = new THREE.Box3().setFromObject(model); box.getCenter(centre); radius = box.getSize(new THREE.Vector3()).length() / 2;
  const ax = AX[META.axes] || AX.z_up;
  buildLines(ax); buildSlices(ax);
  if (META.lines && META.lines.length) box.expandByObject(lineGroup);
  const btn = (m, f, t) => `<button data-m="${m}" ${f ? `data-f="${f}"` : ""}>${t}</button>`;
  $("#modes").innerHTML = btn("real", "", "Realistic") + [...fields].map((f) => btn("field", f, label(f))).join("")
    + (META.lines && META.lines.length ? btn("lines", "", "Streamlines") : "") + (META.slices || []).map((s) => btn("slice", s.name, s.name)).join("");
  document.querySelectorAll("#modes button").forEach((b) => (b.onclick = () => setMode(b.dataset.m, b.dataset.f)));
  document.querySelectorAll("[data-v]").forEach((b) => (b.onclick = () => fit(b.dataset.v)));
  $("#shot").onclick = () => { const a = document.createElement("a"); a.href = renderer.domElement.toDataURL("image/png"); a.download = "view.png"; a.click(); };
  fit("iso"); const first = [...fields][0];
  setMode(first ? "field" : "real", first);
  $("#load").style.display = "none";
}

function resize() { const w = $("#stage").clientWidth, h = $("#stage").clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe($("#stage")); resize();
(function loop() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); })();
main().catch((e) => { $("#load").textContent = "Could not load the scene: " + e.message; });
