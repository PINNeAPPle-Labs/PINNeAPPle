// PINNeAPPle studio viewer: any scene exported by pinneapple_tools.visualization.studio.web_viewer.
// scene.glb carries the surfaces (fields as vertex attributes _NAME), scene.json the lines, slices and labels.
// CFD colours: jet scale, blue = low, red = high, with a colour bar. Controls: colour range (auto, full or typed),
// slice position (slices of one group), streamline density, edges (feature or wireframe).
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { GLTFLoader } from "./vendor/three/loaders/GLTFLoader.js";
import { RoomEnvironment } from "./vendor/three/environments/RoomEnvironment.js";
import * as core from "./studio-core.js";

const AX = { z_up: (p) => new THREE.Vector3(p[0], p[2], -p[1]), aircraft: (p) => new THREE.Vector3(p[1], p[2], p[0]), as_is: (p) => new THREE.Vector3(p[0], p[1], p[2]) };

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

// state: mode real | field | lines | slice; arg = field name or slice group
const S = { mode: "real", arg: null, range: {}, data: {}, stackPos: {}, density: 1, edges: "off" };
let META = {}, label = (f) => f, model = null, centre = new THREE.Vector3(), radius = 1, ax = AX.z_up, stacks = {};
const lineGroup = new THREE.Group(), sliceGroup = new THREE.Group(); scene.add(lineGroup, sliceGroup);
const meshes = [];
const grey = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.6 });
const ghost = new THREE.MeshStandardMaterial({ color: 0xb8c0c8, transparent: true, opacity: 0.3, depthWrite: false });
const fieldMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, metalness: 0, envMapIntensity: 0.4 });

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

// ---------------------------------------------------------------- colour range per field / lines / slice group
const key = () => (S.mode === "field" ? "f:" + S.arg : S.mode === "lines" ? "lines" : S.mode === "slice" ? "s:" + S.arg : null);
const hasScale = () => key() && !(S.mode === "lines" && !S.data.lines.length);
function currentSlice() { const st = stacks[S.arg] || []; return st[Math.min(st.length - 1, S.stackPos[S.arg] || 0)]; }
function values() {
  if (S.mode === "field") return S.data["f:" + S.arg];
  if (S.mode === "lines") return S.data.lines;
  if (S.mode === "slice") return (stacks[S.arg] || []).flatMap((s) => s.grid.flat());   // one scale for the stack
  return [];
}
function range() { const k = key(); if (!S.range[k]) S.range[k] = core.autoRange(values() || [], "auto"); return S.range[k]; }

function legend() {
  const L = $("#legend");
  if (!hasScale()) { L.style.display = "none"; return; }
  const [lo, hi] = range();
  let title = "";
  if (S.mode === "field") title = (META.labels || {})[S.arg] || label(S.arg);
  if (S.mode === "lines") title = META.lineLabel || "value along the lines";
  if (S.mode === "slice") { const s = currentSlice(); title = s.label || s.name; }
  L.innerHTML = core.colorbarHTML({ title, lo, hi, width: 260 });
  L.style.display = "block";
}

function paint() {
  const [lo, hi] = hasScale() ? range() : [0, 1];
  lineGroup.visible = S.mode === "lines"; sliceGroup.visible = S.mode === "slice";
  for (const o of meshes) {
    if (S.mode === "real" || S.mode === "lines") o.material = o.userData.mat;
    else if (S.mode === "slice") o.material = ghost;
    else {
      const a = o.geometry.getAttribute("_" + S.arg.toLowerCase());
      if (!a) { o.material = grey; continue; }
      o.geometry.setAttribute("color", core.colorAttribute(THREE, a.array, lo, hi)); o.material = fieldMat;
    }
  }
  if (S.mode === "lines") {
    core.recolourLines(THREE, lineGroup, lo, hi);
    const on = new Set(core.thin(lineGroup.userData.n || 0, S.density));
    lineGroup.children.forEach((m) => (m.visible = on.has(m.userData.index)));
  }
  if (S.mode === "slice") {
    const cur = currentSlice();
    sliceGroup.children.forEach((m) => (m.visible = m.userData.slice === cur));
    const tk = `${lo}:${hi}`;
    if (cur.mesh.userData.tk !== tk) {                    // texture for the current range
      const old = cur.mesh.material.map; cur.mesh.material.map = core.sliceTexture(THREE, cur.grid, lo, hi);
      cur.mesh.material.needsUpdate = true; if (old) old.dispose(); cur.mesh.userData.tk = tk;
    }
  }
  legend(); panel();
}

// ---------------------------------------------------------------- control panel
function panel() {
  const P = $("#ctl"), k = key(), parts = [];
  if (hasScale()) {
    const [lo, hi] = range(), d = core.decimals(lo, hi) + 1;
    parts.push(`<div class="row"><span>Colour range</span><input id="lo" type="number" step="any" value="${+lo.toFixed(d)}"><span>to</span>`
      + `<input id="hi" type="number" step="any" value="${+hi.toFixed(d)}"><button id="auto" title="1st to 99th percentile">Auto</button><button id="full" title="minimum to maximum">Full</button></div>`);
  }
  if (S.mode === "slice" && (stacks[S.arg] || []).length > 1) {
    const st = stacks[S.arg], i = Math.min(st.length - 1, S.stackPos[S.arg] || 0);
    parts.push(`<div class="row"><span>Slice position</span><input id="pos" type="range" min="0" max="${st.length - 1}" step="1" value="${i}"><span>${st[i].name}</span></div>`);
  }
  if (S.mode === "lines") {
    const n = lineGroup.userData.n || 0;
    parts.push(`<div class="row"><span>Line density</span><input id="dens" type="range" min="0.05" max="1" step="0.05" value="${S.density}"><span>${core.thin(n, S.density).length} / ${n}</span></div>`);
  }
  parts.push(`<div class="row"><span>Edges</span><select id="edges">${[["off", "Off"], ["feature", "Feature edges"], ["wire", "Wireframe"]].map(([v, t]) => `<option value="${v}" ${S.edges === v ? "selected" : ""}>${t}</option>`).join("")}</select></div>`);
  P.innerHTML = parts.join("");
  const num = (id) => parseFloat($(id).value);
  if ($("#lo")) {
    const set = () => { const lo = num("#lo"), hi = num("#hi"); if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) { S.range[k] = [lo, hi]; paint(); } };
    $("#lo").onchange = set; $("#hi").onchange = set;
    $("#auto").onclick = () => { S.range[k] = core.autoRange(values(), "auto"); paint(); };
    $("#full").onclick = () => { S.range[k] = core.autoRange(values(), "full"); paint(); };
  }
  if ($("#pos")) $("#pos").oninput = (e) => { S.stackPos[S.arg] = +e.target.value; paint(); };
  if ($("#dens")) $("#dens").oninput = (e) => { S.density = +e.target.value; paint(); };
  $("#edges").onchange = (e) => { S.edges = e.target.value; meshes.forEach((o) => core.setEdges(THREE, o, S.edges)); };
}

function setMode(m, arg) {
  S.mode = m; S.arg = arg || null;
  document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("on", b.dataset.m === m && (!b.dataset.f || b.dataset.f === arg)));
  paint();
}

// ---------------------------------------------------------------- scene parts
function buildLines() {
  const all = (META.lines || []).flatMap((L) => L.points.map((P, i) => [P, L.values ? L.values[i] : null]));
  S.data.lines = all.flatMap(([, v]) => v || []);
  META.lineLabel = ((META.lines || []).find((L) => L.label) || {}).label;
  const [lo, hi] = S.data.lines.length ? core.autoRange(S.data.lines) : [0, 1];
  const g = core.tubeLines(THREE, all.map(([P]) => P), all.some(([, v]) => v) ? all.map(([, v]) => v) : null,
    { lo, hi, radius: radius * 0.0035, toV3: ax });
  if (g.children.length) lineGroup.add(...g.children);
  lineGroup.userData.n = all.length;
}

function buildSlices() {
  for (const s of META.slices || []) {
    s.mesh = core.sliceMesh(THREE, s.origin, s.u, s.v, null, ax);
    s.mesh.userData.slice = s; s.mesh.name = s.name; sliceGroup.add(s.mesh);
  }
  stacks = core.sliceStacks(META.slices || []);
}

async function main() {
  META = await (await fetch("scene.json")).json();
  document.title = META.title || "PINNeAPPle"; $("#title").textContent = META.title || "";
  const gltf = await new GLTFLoader().loadAsync("scene.glb");
  model = gltf.scene; scene.add(model);
  const fields = new Set();
  model.traverse((o) => { if (o.isMesh) { o.userData.mat = o.material; meshes.push(o); Object.keys(o.geometry.attributes).filter((k) => k.startsWith("_")).forEach((k) => fields.add(k.slice(1))); } });
  for (const f of fields) S.data["f:" + f] = meshes.flatMap((o) => { const a = o.geometry.getAttribute("_" + f); return a ? Array.from(a.array) : []; });
  // three.js lowercases attribute names: show the original field name, and its label in the colour bar
  label = (f) => {
    const k = Object.keys(META.labels || {}).find((x) => x.toLowerCase() === f); if (k) META.labels[f] = META.labels[k];
    return (META.fields || []).find((x) => x.toLowerCase() === f) || k || f;
  };
  const box = new THREE.Box3().setFromObject(model); box.getCenter(centre); radius = box.getSize(new THREE.Vector3()).length() / 2;
  ax = AX[META.axes] || AX.z_up;
  buildLines(); buildSlices();
  const btn = (m, f, t) => `<button data-m="${m}" ${f ? `data-f="${f}"` : ""}>${t}</button>`;
  $("#modes").innerHTML = btn("real", "", "Realistic") + [...fields].map((f) => btn("field", f, label(f))).join("")
    + (lineGroup.userData.n ? btn("lines", "", "Streamlines") : "") + Object.keys(stacks).map((g) => btn("slice", g, g)).join("");
  document.querySelectorAll("#modes button").forEach((b) => (b.onclick = () => setMode(b.dataset.m, b.dataset.f)));
  document.querySelectorAll("[data-v]").forEach((b) => (b.onclick = () => fit(b.dataset.v)));
  $("#shot").onclick = () => { const a = document.createElement("a"); a.href = renderer.domElement.toDataURL("image/png"); a.download = "view.png"; a.click(); };
  fit("iso"); const first = [...fields][0];
  setMode(first ? "field" : "real", first);
  $("#load").style.display = "none";
  window.studio = { S, setMode, fit, paint };              // for scripted screenshots and tests
}

function resize() { const w = $("#stage").clientWidth, h = $("#stage").clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe($("#stage")); resize();
(function loop() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); })();
main().catch((e) => { $("#load").textContent = "Could not load the scene: " + e.message; });
