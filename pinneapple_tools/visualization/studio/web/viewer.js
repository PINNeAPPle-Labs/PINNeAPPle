// PINNeAPPle studio viewer: any scene exported by pinneapple_tools.visualization.studio.web_viewer.
// scene.glb carries the surfaces (fields as vertex attributes _NAME), scene.json the lines, slices and labels.
// CFD colours: jet scale, blue = low, red = high, with a colour bar. Controls: colour range (auto, full or typed),
// slice position (slices of one group), streamline density, edges (feature or wireframe).
// scene.json "animation" (pinneapple_tools.visualization.studio.animation): rigid tracks, arrows, trails, per-vertex
// fields and head-up values per frame, with a bottom bar (play, time slider, speed, follow camera).
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
const hemi = new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.6); scene.add(hemi);
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
  const fg = A.follow && A.groups[A.followGroup];
  const c = fg ? groupCentre(fg) : centre, rad = fg ? fg.radius : radius;
  // scene axes x (length / flow), y, z up -> three.js (x, z, -y); the aircraft convention is handled by AX
  const d = (META.axes === "aircraft"
    ? { iso: [-1.1, 0.75, -1], front: [0, 0.12, -1], top: [0, 1, 0.001], side: [-1, 0.12, 0], back: [0, 0.12, 1] }
    : { iso: [-1, 0.75, 1.1], front: [-1, 0.12, 0], top: [0.001, 1, 0], side: [0, 0.12, 1], back: [1, 0.12, 0] })[view];
  const v = new THREE.Vector3(...d).normalize();
  const dist = rad / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05;
  camera.position.copy(c).addScaledVector(v, dist); controls.target.copy(c); clip(); camera.updateProjectionMatrix();
  controls.update(); A.prev = null;
  if (!META.lighting) sun.position.copy(centre).add(new THREE.Vector3(-1, 2, 1.2).multiplyScalar(radius * 3));
}
// near / far planes from the distance to the target, far enough for the whole scene (a km-long runway)
function clip() {
  const d = Math.max(camera.position.distanceTo(controls.target), 1e-6);
  const n = d / 1000, f = Math.max(d * 100, (camera.position.distanceTo(centre) + radius) * 2);
  if (Math.abs(camera.near - n) > 1e-3 * n || Math.abs(camera.far - f) > 1e-3 * f) { camera.near = n; camera.far = f; camera.updateProjectionMatrix(); }
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
  L.innerHTML = core.colorbarHTML({ title, lo, hi, width: 260, stops: S.mode === "lines" ? META.lineStops : core.JET });
  L.style.display = "block";
}

function paint() {
  const [lo, hi] = hasScale() ? range() : [0, 1];
  lineGroup.visible = S.mode === "lines"; sliceGroup.visible = S.mode === "slice";
  for (const o of meshes) {
    if (o.userData.animField && S.mode !== "slice") { o.material = o.userData.mat; continue; }
    if (S.mode === "real" || S.mode === "lines") o.material = o.userData.mat;
    else if (S.mode === "slice") o.material = ghost;
    else {
      const a = o.geometry.getAttribute("_" + S.arg.toLowerCase());
      if (!a) { o.material = grey; continue; }
      o.geometry.setAttribute("color", core.colorAttribute(THREE, a.array, lo, hi)); o.material = fieldMat;
    }
  }
  if (S.mode === "lines") {
    core.recolourLines(THREE, lineGroup, lo, hi, META.lineStops);
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
  // colour map, range, radius and opacity of the first set that gives them (Lines.cmap / color_range / radius / alpha)
  const first = (k) => ((META.lines || []).find((L) => L[k] !== undefined && L[k] !== null) || {})[k];
  const cmap = first("cmap") || "jet", alpha = first("alpha") ?? 1, rad = first("radius");
  META.lineStops = core.cmapStops(cmap);
  let [lo, hi] = S.data.lines.length ? core.autoRange(S.data.lines) : [0, 1];
  if (first("range")) [lo, hi] = first("range");
  else if (cmap !== "jet") { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; [lo, hi] = [-m, m]; }
  if (S.data.lines.length) S.range.lines = [lo, hi];
  const g = core.tubeLines(THREE, all.map(([P]) => P), all.some(([, v]) => v) ? all.map(([, v]) => v) : null,
    { lo, hi, radius: rad || radius * 0.0035, toV3: ax, stops: META.lineStops });
  if (alpha < 1) g.children.forEach((m) => { m.material = m.material.clone(); m.material.transparent = true; m.material.opacity = alpha; m.material.depthWrite = false; });
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
  ax = AX[META.axes] || AX.z_up;
  if (META.animation) buildAnimation(gltf);
  const box = new THREE.Box3().setFromObject(model);            // static scene, bodies at frame 0, trajectories
  for (const P of A.paths) for (const p of P) box.expandByPoint(ax(p));
  box.getCenter(centre); radius = box.getSize(new THREE.Vector3()).length() / 2;
  if (META.lighting) applyLighting(META.lighting);
  buildLines(); buildSlices();
  const btn = (m, f, t) => `<button data-m="${m}" ${f ? `data-f="${f}"` : ""}>${t}</button>`;
  $("#modes").innerHTML = btn("real", "", "Realistic") + [...fields].map((f) => btn("field", f, label(f))).join("")
    + (lineGroup.userData.n ? btn("lines", "", "Streamlines") : "") + Object.keys(stacks).map((g) => btn("slice", g, g)).join("");
  document.querySelectorAll("#modes button").forEach((b) => (b.onclick = () => setMode(b.dataset.m, b.dataset.f)));
  document.querySelectorAll("[data-v]").forEach((b) => (b.onclick = () => fit(b.dataset.v)));
  $("#shot").onclick = () => { const a = document.createElement("a"); a.href = renderer.domElement.toDataURL("image/png"); a.download = "view.png"; a.click(); };
  if (A.on) animBar();
  fit("iso"); const first = [...fields][0];
  setMode(first ? "field" : "real", first);
  if (A.on && META.animation.camera === "follow") setFollow(true);
  $("#load").style.display = "none";
  window.studio = { S, setMode, fit, paint, setFrame, play, setFollow, anim: A };   // for scripted screenshots and tests
}


// ---------------------------------------------------------------- animation
const A = { on: false, f: 0, t: 0, playing: false, speed: 1, follow: false, vectors: true, groups: {}, arrows: [], trails: [], fields: [], paths: [], prev: null, followGroup: null };
const UP = new THREE.Vector3(0, 1, 0), _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3();
const toQ = (q) => new THREE.Quaternion(...core.quatToThree(q, META.axes));
const sRGB = (rgb) => new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
function groupCentre(G) { return G.local.clone().applyMatrix4(G.g.matrixWorld); }

function buildAnimation(gltf) {
  const an = META.animation; A.on = true; A.followGroup = an.follow_group || (an.tracks[0] || {}).group || null;
  const byNode = {};                                          // glTF node index -> three.js object
  model.traverse((o) => { const a = gltf.parser.associations.get(o); if (a && a.nodes !== undefined && !(a.nodes in byNode)) byNode[a.nodes] = o; });
  for (const tr of an.tracks) {
    const g = new THREE.Group(); g.name = "track:" + tr.group; model.add(g);
    for (const i of tr.surfaces || []) if (byNode[i]) g.add(byNode[i]);
    const box = new THREE.Box3().setFromObject(g);           // body frame (the group is still at the origin)
    const local = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
    const r = box.isEmpty() ? 1 : Math.max(1e-6, box.getSize(new THREE.Vector3()).length() / 2);
    A.groups[tr.group] = { g, tr, local, radius: r };
    A.paths.push(tr.positions);
  }
  for (const vs of an.vectors) {
    const parent = vs.attach && A.groups[vs.attach] ? A.groups[vs.attach].g : model;
    const M = vs.origins[0].length;
    let L = 0; for (const fr of vs.vectors) for (const v of fr) L = Math.max(L, Math.hypot(v[0], v[1], v[2]));
    L = L || 1;
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0, color: 0xffffff, envMapIntensity: 0.5, toneMapped: false });
    const shaftGeo = new THREE.CylinderGeometry(1, 1, 1, 10).translate(0, 0.5, 0), headGeo = new THREE.ConeGeometry(1, 1, 14).translate(0, 0.5, 0);
    const shaft = new THREE.InstancedMesh(shaftGeo, mat, M), head = new THREE.InstancedMesh(headGeo, mat, M);
    shaft.frustumCulled = head.frustumCulled = false;
    const base = sRGB(hexRGB(vs.color || "#f97316"));
    for (let i = 0; i < M; i++) { shaft.setColorAt(i, base); head.setColorAt(i, base); }
    parent.add(shaft, head);
    A.arrows.push({ vs, shaft, head, L, r: 0.022 * L, h: 0.22 * L, stops: core.cmapStops(vs.cmap) });
  }
  for (const tr of an.trails) {
    const n = tr.points.length, P = new Float32Array(n * 3); tr.points.forEach((p, i) => ax(p).toArray(P, 3 * i));
    const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(P, 3));
    // RGBA per point: colour from the values (e.g. the load factor) or the trail colour; alpha set per frame (fade)
    const rgb = tr.values
      ? core.colorArray(Float32Array.from(tr.values, (x) => (x === null ? NaN : x)), ...(tr.range || core.autoRange(tr.values, "full")), core.cmapStops(tr.cmap))
      : (() => { const c = new THREE.Color(tr.color || "#f59e0b"), a = new Float32Array(3 * n); for (let i = 0; i < n; i++) c.toArray(a, 3 * i); return a; })();
    const rgba = new Float32Array(4 * n); for (let i = 0; i < n; i++) { rgba.set(rgb.subarray(3 * i, 3 * i + 3), 4 * i); rgba[4 * i + 3] = tr.alpha ?? 1; }
    geo.setAttribute("color", new THREE.BufferAttribute(rgba, 4));
    const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false });
    const line = new THREE.Line(geo, mat);
    line.frustumCulled = false; model.add(line); A.trails.push({ tr, line }); A.paths.push(tr.points);
  }
  for (const fd of an.fields) {
    const o = byNode[fd.index] || meshes.find((m) => m.name === fd.surface);
    if (!o || !o.isMesh) continue;
    const n = o.geometry.attributes.position.count;
    o.geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(3 * n), 3));
    o.userData.mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, metalness: 0, envMapIntensity: 0.4, side: THREE.DoubleSide });
    o.userData.animField = true; A.fields.push({ fd, o, n });
  }
  applyFrame(0);
}
function hexRGB(h) { const c = new THREE.Color(h); c.convertLinearToSRGB(); return [c.r, c.g, c.b]; }

// put everything at fractional frame f (between frames: positions linear, rotations slerp, values linear)
function applyFrame(f) {
  const an = META.animation, n = an.times.length, [k0, k1, w] = core.bracket(f, n);
  A.f = f; A.t = core.lerp(an.times[k0], an.times[k1], w);
  for (const G of Object.values(A.groups)) {
    G.g.position.copy(ax(core.lerp3(G.tr.positions[k0], G.tr.positions[k1], w)));
    G.g.quaternion.slerpQuaternions(toQ(G.tr.quaternions[k0]), toQ(G.tr.quaternions[k1]), w);
    G.g.updateMatrixWorld(true);
  }
  for (const a of A.arrows) {
    const { vs } = a, M = vs.origins[0].length, [lo, hi] = vs.range || [0, 1];
    a.shaft.visible = a.head.visible = A.vectors;
    for (let i = 0; i < M; i++) {
      const o = ax(core.lerp3(vs.origins[k0][i], vs.origins[k1][i], w)), d = ax(core.lerp3(vs.vectors[k0][i], vs.vectors[k1][i], w));
      const len = d.length();
      if (!(len > 1e-9 * a.L)) { _m.makeScale(0, 0, 0); a.shaft.setMatrixAt(i, _m); a.head.setMatrixAt(i, _m); continue; }
      d.divideScalar(len); _q.setFromUnitVectors(UP, d);
      const h = Math.min(a.h, 0.45 * len), r = Math.min(a.r, 0.25 * len);
      a.shaft.setMatrixAt(i, _m.compose(o, _q, _s.set(r, len - h, r)));
      a.head.setMatrixAt(i, _m.compose(_p.copy(o).addScaledVector(d, len - h), _q, _s.set(2.6 * r, h, 2.6 * r)));
      if (vs.values) {
        const c = sRGB(core.ramp(a.stops, hi - lo > 1e-12 ? (core.lerp(vs.values[k0][i], vs.values[k1][i], w) - lo) / (hi - lo) : 0.5));
        a.shaft.setColorAt(i, c); a.head.setColorAt(i, c);
      }
    }
    for (const m of [a.shaft, a.head]) { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; }
  }
  A.trailLast = Math.min(an.times.length - 1, k0 + (w > 0 ? 1 : 0)); updateTrails();
  for (const { fd, o, n } of A.fields) {
    const v0 = fd.values[k0], v1 = fd.values[k1], v = new Float32Array(n);
    for (let i = 0; i < n; i++) v[i] = core.lerp(v0[i], v1[i], w) ?? NaN;
    o.geometry.attributes.color.array.set(core.colorArray(v, fd.range[0], fd.range[1], core.cmapStops(fd.cmap))); o.geometry.attributes.color.needsUpdate = true;
  }
  if (A.follow) chase();
  hud();
}

// trails: the last trail_seconds behind the body, fading out; never running through the camera (only the newest
// points at least 0.8 x the camera-target distance in front of the camera, fading out where cut). Called every frame.
const _cf = new THREE.Vector3(), _cp = new THREE.Vector3();
function updateTrails() {
  if (!A.trails.length || A.trailLast === undefined) return;
  const an = META.animation, last = A.trailLast;
  camera.getWorldDirection(_cf); const dmin = 0.8 * camera.position.distanceTo(controls.target);
  for (const { tr, line } of A.trails) {
    const L = Math.min(tr.points.length - 1, last), [j0, fade] = core.trailWindow(an.times, L, tr.seconds, tr.fade, A.t);
    const pos = line.geometry.attributes.position.array, col = line.geometry.attributes.color;
    let j = L;                                                // newest run of points ahead of the camera
    while (j > j0 && _cp.fromArray(pos, 3 * (j - 1)).sub(camera.position).dot(_cf) > dmin) j--;
    const cut = j > j0, n = L - j + 1;
    for (let i = j; i <= L; i++) col.array[4 * i + 3] = (tr.alpha ?? 1) * fade(i) * (cut ? (i - j) / Math.max(1, n - 1) : 1);
    col.needsUpdate = true; line.geometry.setDrawRange(j, Math.max(1, n)); line.visible = A.vectors;
  }
}

// chase camera: follows the body's translation (and its rotation when follow_frame is "body"); orbiting still works
function chase() {
  const G = A.groups[A.followGroup]; if (!G) return;
  const p = groupCentre(G), q = G.g.quaternion.clone();
  if (A.prev) {
    const off = camera.position.clone().sub(A.prev.p);
    if (META.animation.follow_frame !== "world") off.applyQuaternion(q.clone().multiply(A.prev.q.clone().invert()));
    camera.position.copy(p).add(off); controls.target.copy(p);
  }
  A.prev = { p, q };
}
function setFollow(on) {
  A.follow = !!on && !!A.groups[A.followGroup]; A.prev = null;
  const cb = $("#afollow"); if (cb) cb.checked = A.follow;
  if (!A.follow) { fit("iso"); return; }
  const G = A.groups[A.followGroup], an = META.animation, p = groupCentre(G), q = G.g.quaternion;
  let off;
  if (an.follow_offset) { off = ax(an.follow_offset); if (an.follow_frame !== "world") off.applyQuaternion(q); }
  else {                                                     // behind the direction of travel, a little above
    const [k0, k1] = core.bracket(A.f, an.times.length), P = G.tr.positions, kk = k1 > k0 ? [k0, k1] : [Math.max(0, k0 - 1), k0];
    const vel = ax(P[kk[1]]).sub(ax(P[kk[0]])); vel.y = 0;
    const back = vel.lengthSq() > 1e-12 ? vel.normalize().negate() : new THREE.Vector3(-1, 0, 1).normalize();
    const dist = G.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 2.2;
    off = back.multiplyScalar(dist * 0.9).add(new THREE.Vector3(0, dist * 0.42, 0));
  }
  camera.position.copy(p).add(off); controls.target.copy(p); controls.update(); clip(); A.prev = { p, q: q.clone() };
}
function setFrame(k) {
  const n = META.animation ? META.animation.times.length : 0; if (!n) return;
  applyFrame(Math.max(0, Math.min(n - 1, +k))); controls.update(); clip(); renderer.render(scene, camera);
}
function play(on = true) { A.playing = !!on; const b = $("#aplay"); if (b) b.textContent = A.playing ? "Pause" : "Play"; }

function animBar() {
  const an = META.animation, n = an.times.length;
  document.body.classList.add("anim");
  const speeds = [0.25, 0.5, 1, 2, 4];
  $("#anim").innerHTML = `<button id="aplay">Play</button><input id="aslider" type="range" min="0" max="${n - 1}" step="1" value="0">`
    + `<span id="atime"></span><select id="aspeed" title="playback speed">${speeds.map((s) => `<option value="${s}" ${s === 1 ? "selected" : ""}>${s}×</option>`).join("")}</select>`
    + (Object.keys(A.groups).length ? `<label><input id="afollow" type="checkbox"> Follow</label>` : "")
    + (A.arrows.length || A.trails.length ? `<label><input id="avec" type="checkbox" checked> Vectors</label>` : "");
  $("#anim").style.display = "flex";
  $("#aplay").onclick = () => play(!A.playing);
  $("#aslider").oninput = (e) => { play(false); setFrame(+e.target.value); };
  $("#aspeed").onchange = (e) => (A.speed = +e.target.value);
  if ($("#afollow")) $("#afollow").onchange = (e) => setFollow(e.target.checked);
  if ($("#avec")) $("#avec").onchange = (e) => { A.vectors = e.target.checked; applyFrame(A.f); };
  const bars = [...an.vectors.filter((v) => v.values && v.range && v.range[1] - v.range[0] > 1e-12).map((v) => [v.label || v.name, v.range, v.cmap]),
    ...(an.trails || []).filter((t) => t.values && t.range).map((t) => [t.label || t.name, t.range, t.cmap]),
    ...an.fields.map((f) => [f.label || f.surface, f.range, f.cmap])];
  if (bars.length) { $("#alegend").innerHTML = bars.map(([t, [lo, hi], c]) => core.colorbarHTML({ title: t, lo, hi, width: 220, stops: core.cmapStops(c) })).join('<div style="height:8px"></div>'); $("#alegend").style.display = "block"; }
  hud();
}
function hud() {
  if (!$("#hud") || !A.on) return;
  const an = META.animation, [k0, k1, w] = core.bracket(A.f, an.times.length);
  const lines = [`t = ${A.t.toFixed(2)} s`, ...an.hud.map((h) => core.hudLine(h, core.lerp(h.values[k0], h.values[k1], w)))];
  $("#hud").innerHTML = lines.map((l, i) => `<div${i ? "" : ' style="font-weight:700"'}>${l}</div>`).join(""); $("#hud").style.display = "block";
  if ($("#atime")) $("#atime").textContent = `${A.t.toFixed(2)} / ${an.times[an.times.length - 1].toFixed(2)} s`;
  const sl = $("#aslider"); if (sl && document.activeElement !== sl) sl.value = Math.round(A.f);
}

// lighting preset from pinneapple_tools.visualization.studio.lighting (e.g. a black lunar sky and one low Sun)
function applyLighting(L) {
  scene.background = new THREE.Color(L.background[1]);
  const d = ax(L.sun.direction).normalize();
  sun.position.copy(centre).addScaledVector(d, 4 * radius); sun.target.position.copy(centre); scene.add(sun.target);
  sun.intensity = 1.2 * L.sun.intensity; sun.color = new THREE.Color(...L.sun.color);
  hemi.intensity = L.ambient; hemi.groundColor = new THREE.Color(0x222222);
  if (L.background[1] === "#000000") scene.environment = null;
}

function resize() { const w = $("#stage").clientWidth, h = $("#stage").clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe($("#stage")); resize();
let tPrev = performance.now();
(function loop() {
  const now = performance.now(), dt = Math.min(0.25, (now - tPrev) / 1000); tPrev = now;
  if (A.on && A.playing) {
    const T = META.animation.times, t0 = T[0], t1 = T[T.length - 1];
    A.t += dt * A.speed; if (A.t > t1) A.t = t0 + ((A.t - t0) % ((t1 - t0) || 1));
    applyFrame(core.frameAt(T, A.t));
  }
  controls.update(); clip(); if (A.on) updateTrails(); renderer.render(scene, camera); requestAnimationFrame(loop);
})();
main().catch((e) => { $("#load").textContent = "Could not load the scene: " + e.message; });
