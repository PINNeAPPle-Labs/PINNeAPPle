// PINNeAPPle Twin3D viewer: loads scene.json + geometry.glb + fields.bin written by
// pinneapple_twin3d.Scene.export(). Static files only; three.js from jsdelivr.
//
// URL parameters:
//   ?scene=path/to/scene.json   (default: ./scene.json)
//   &live=ws://host:port/path   live sensor values: messages {"id": "...", "value": 1.23}
//                               or a list of them; alarm when outside the sensor envelope.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const sceneUrl = new URL(params.get("scene") || "scene.json", location.href);

// ── colormaps ──────────────────────────────────────────────────────────
const clamp01 = (x) => Math.min(1, Math.max(0, x));
function turbo(t) { // Google "Turbo" polynomial approximation (Mikhailov, 2019)
  t = clamp01(t);
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [clamp01(r), clamp01(g), clamp01(b)];
}
const VIRIDIS = [[0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530], [0.207, 0.372, 0.553],
  [0.164, 0.471, 0.558], [0.128, 0.567, 0.551], [0.135, 0.659, 0.518], [0.267, 0.749, 0.441],
  [0.478, 0.821, 0.318], [0.741, 0.873, 0.150], [0.993, 0.906, 0.144]];
const COOLWARM = [[0.230, 0.299, 0.754], [0.552, 0.690, 0.996], [0.865, 0.865, 0.865], [0.958, 0.604, 0.482], [0.706, 0.016, 0.150]];
function lut(table) {
  return (t) => {
    t = clamp01(t) * (table.length - 1);
    const i = Math.min(table.length - 2, Math.floor(t)), f = t - i;
    return [0, 1, 2].map((k) => table[i][k] * (1 - f) + table[i + 1][k] * f);
  };
}
const CMAPS = { turbo, viridis: lut(VIRIDIS), coolwarm: lut(COOLWARM) };

// ── state ──────────────────────────────────────────────────────────────
const state = { manifest: null, fields: null, parts: new Map(), sensors: [], step: 0, playing: null,
  field: "", cmap: "turbo", live: new Map(), clipPlane: new THREE.Plane(new THREE.Vector3(1, 0, 0), 0) };

// ── three.js setup ─────────────────────────────────────────────────────
const viewport = $("viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.localClippingEnabled = true;
viewport.prepend(renderer.domElement);
const scene3 = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1e7);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene3.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
scene3.add(sun);
const sensorGroup = new THREE.Group();
scene3.add(sensorGroup);

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function applyTheme() { scene3.background = new THREE.Color(cssVar("--viewport") || "#e9ecf1"); }

function resize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

function showError(msg) {
  const el = $("error");
  el.style.display = "grid";
  el.textContent = msg;
}

// ── loading ────────────────────────────────────────────────────────────
async function load() {
  const res = await fetch(sceneUrl);
  if (!res.ok) throw new Error(`Não foi possível abrir ${sceneUrl} (HTTP ${res.status}).`);
  const m = await res.json();
  if (m.format !== "pinneapple-twin3d/1") throw new Error(`Formato desconhecido: ${m.format}`);
  state.manifest = m;
  $("title").textContent = m.title;
  const nTri = m.parts.reduce((a, p) => a + p.triangles, 0);
  $("meta").textContent = `${m.parts.length} peças · ${nTri.toLocaleString("pt-BR")} triângulos · unidade ${m.length_unit}` +
    (m.source ? ` · fonte: ${m.source}` : "");

  const base = new URL(".", sceneUrl);
  const [gltf, buf] = await Promise.all([
    new GLTFLoader().loadAsync(new URL(m.geometry, base).href),
    m.fields.length ? fetch(new URL(m.fields_file, base)).then((r) => r.arrayBuffer()) : Promise.resolve(new ArrayBuffer(0)),
  ]);
  state.fields = buf;

  const byName = new Map();
  gltf.scene.traverse((o) => { if (o.isMesh) byName.set(o.name, o); });
  for (const p of m.parts) {
    const mesh = byName.get(p.name) || byName.get(p.name.replace(/\s/g, "_"));
    if (!mesh) continue;
    const base = new THREE.Color(...p.color);
    mesh.material = new THREE.MeshStandardMaterial({ color: base, metalness: 0.1, roughness: 0.7,
      side: THREE.DoubleSide, clippingPlanes: [], clipShadows: true });
    const count = mesh.geometry.attributes.position.count;
    mesh.geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    state.parts.set(p.name, { meta: p, mesh, base, fields: new Map() });
  }
  scene3.add(gltf.scene);
  for (const f of m.fields) {
    const part = state.parts.get(f.part);
    if (part) part.fields.set(f.name, new Float32Array(buf, f.offset, f.steps * f.count));
  }
  buildUi();
  fitCamera();
  update();
}

// ── UI ─────────────────────────────────────────────────────────────────
function fieldNames() { return [...new Set(state.manifest.fields.map((f) => f.name))]; }
function fieldUnit(name) { return (state.manifest.fields.find((f) => f.name === name) || {}).unit || ""; }
function fieldRange(name) {
  if (!$("autorange").checked) return [parseFloat($("vmin").value), parseFloat($("vmax").value)];
  const fs = state.manifest.fields.filter((f) => f.name === name);
  return [Math.min(...fs.map((f) => f.min)), Math.max(...fs.map((f) => f.max))];
}
const fmt = (v) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) ? v.toExponential(3) : v.toPrecision(4);

function buildUi() {
  const m = state.manifest;
  const sel = $("field");
  sel.innerHTML = `<option value="">(sem campo — cor da peça)</option>` +
    fieldNames().map((n) => `<option value="${n}">${n}${fieldUnit(n) ? ` [${fieldUnit(n)}]` : ""}</option>`).join("");
  state.field = fieldNames()[0] || "";
  sel.value = state.field;
  sel.onchange = () => { state.field = sel.value; syncRangeInputs(); update(); };
  $("cmap").onchange = (e) => { state.cmap = e.target.value; update(); };
  $("autorange").onchange = () => { syncRangeInputs(); update(); };
  $("vmin").oninput = $("vmax").oninput = () => { $("autorange").checked = false; update(); };
  syncRangeInputs();

  // parts, grouped
  const groups = new Map();
  for (const [name, p] of state.parts) {
    const g = p.meta.group || "Peças";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(name);
  }
  const box = $("parts");
  box.innerHTML = "";
  for (const [g, names] of groups) {
    if (groups.size > 1) box.insertAdjacentHTML("beforeend", `<div class="meta" style="margin-top:6px">${g}</div>`);
    for (const name of names) {
      const row = document.createElement("label");
      row.className = "row part-actions";
      row.innerHTML = `<input type="checkbox" checked><span></span><button title="Mostrar só esta">isolar</button>`;
      row.querySelector("span").textContent = name;
      const cb = row.querySelector("input");
      cb.onchange = () => { state.parts.get(name).mesh.visible = cb.checked; };
      row.querySelector("button").onclick = (e) => {
        e.preventDefault();
        for (const [n, p] of state.parts) {
          p.mesh.visible = n === name;
          box.querySelectorAll("label").forEach((l) => {
            if (l.querySelector("span").textContent === n) l.querySelector("input").checked = n === name;
          });
        }
      };
      box.appendChild(row);
    }
  }

  // time
  const nt = m.times.length;
  $("time").max = Math.max(nt - 1, 0);
  $("time").oninput = (e) => { state.step = +e.target.value; update(); };
  $("play").disabled = nt < 2;
  $("play").onclick = () => {
    if (state.playing) { clearInterval(state.playing); state.playing = null; $("play").textContent = "▶"; return; }
    $("play").textContent = "⏸";
    state.playing = setInterval(() => { state.step = (state.step + 1) % nt; $("time").value = state.step; update(); }, 250);
  };

  // clipping
  const [lo, hi] = m.bounds;
  const setClip = () => {
    const axis = $("clipAxis").value;
    for (const p of state.parts.values()) p.mesh.material.clippingPlanes = axis === "" ? [] : [state.clipPlane];
    if (axis === "") return;
    const a = +axis, pos = lo[a] + (hi[a] - lo[a]) * (+$("clipPos").value / 1000);
    const flip = $("clipFlip").checked ? -1 : 1;
    const n = new THREE.Vector3(); n.setComponent(a, flip);
    state.clipPlane.set(n, -flip * pos);  // keeps the side where flip * (x_a - pos) >= 0
  };
  $("clipAxis").onchange = $("clipPos").oninput = $("clipFlip").onchange = setClip;

  // sensors
  const diag = new THREE.Vector3(...hi).sub(new THREE.Vector3(...lo)).length();
  const geo = new THREE.SphereGeometry(diag * 0.008, 16, 12);
  $("sensors").innerHTML = m.sensors.length ? "" : `<span class="meta">nenhum sensor</span>`;
  for (const s of m.sensors) {
    const mat = new THREE.MeshBasicMaterial({ color: 0x1f9d55 });
    const ball = new THREE.Mesh(geo, mat);
    ball.position.set(...s.position);
    sensorGroup.add(ball);
    const row = document.createElement("div");
    row.className = "sensor";
    row.innerHTML = `<span class="dot"></span><span></span><span class="val"></span>`;
    row.children[1].textContent = s.label;
    $("sensors").appendChild(row);
    const tag = document.createElement("div");
    tag.className = "tag";
    viewport.appendChild(tag);
    state.sensors.push({ meta: s, ball, row, tag });
  }
  $("fit").onclick = fitCamera;
  $("theme").onclick = () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark" ||
      (!document.documentElement.hasAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
    applyTheme(); updateLegend();
  };
  connectLive();
}

function syncRangeInputs() {
  if (!state.field) return;
  const [a, b] = fieldRange(state.field);
  if ($("autorange").checked) { $("vmin").value = a; $("vmax").value = b; }
}

function fitCamera() {
  const [lo, hi] = state.manifest.bounds;
  const c = new THREE.Vector3(...lo).add(new THREE.Vector3(...hi)).multiplyScalar(0.5);
  const r = Math.max(new THREE.Vector3(...hi).sub(new THREE.Vector3(...lo)).length() / 2, 1e-6);
  const dist = r / Math.sin((camera.fov * Math.PI) / 360);
  camera.position.copy(c).add(new THREE.Vector3(0.6, 0.45, 0.66).normalize().multiplyScalar(dist));
  camera.near = dist / 1000; camera.far = dist * 100; camera.updateProjectionMatrix();
  controls.target.copy(c); controls.update();
  sun.position.copy(camera.position);
}

// ── per-step update ────────────────────────────────────────────────────
function valuesAt(part, name, step) {
  const arr = part.fields.get(name);
  if (!arr) return null;
  const count = part.mesh.geometry.attributes.position.count;
  const steps = arr.length / count;
  const s = Math.min(step, steps - 1);
  return arr.subarray(s * count, (s + 1) * count);
}

function update() {
  const m = state.manifest;
  const cmap = CMAPS[state.cmap];
  const [vmin, vmax] = state.field ? fieldRange(state.field) : [0, 1];
  const span = vmax - vmin || 1;
  for (const p of state.parts.values()) {
    const vals = state.field ? valuesAt(p, state.field, state.step) : null;
    const mat = p.mesh.material;
    if (!vals) { mat.vertexColors = false; mat.color.copy(p.base); mat.needsUpdate = true; continue; }
    const col = p.mesh.geometry.attributes.color;
    for (let i = 0; i < vals.length; i++) {
      const [r, g, b] = cmap((vals[i] - vmin) / span);
      col.array[3 * i] = r; col.array[3 * i + 1] = g; col.array[3 * i + 2] = b;
    }
    col.needsUpdate = true;
    if (!mat.vertexColors) { mat.vertexColors = true; mat.color.set(0xffffff); mat.needsUpdate = true; }
  }
  const t = m.times.length ? m.times[state.step] : null;
  $("timeLabel").textContent = t === null ? "regime permanente" : `t = ${fmt(t)} ${m.time_unit} (${state.step + 1}/${m.times.length})`;
  updateSensors();
  updateLegend();
}

function updateLegend() {
  const show = !!state.field;
  $("legend").style.display = show ? "block" : "none";
  if (!show) return;
  const [a, b] = fieldRange(state.field);
  const stops = Array.from({ length: 11 }, (_, i) => {
    const [r, g, bl] = CMAPS[state.cmap](i / 10);
    return `rgb(${(r * 255) | 0},${(g * 255) | 0},${(bl * 255) | 0}) ${i * 10}%`;
  });
  $("legendBar").style.background = `linear-gradient(90deg, ${stops.join(",")})`;
  $("legendName").textContent = state.field + (fieldUnit(state.field) ? ` [${fieldUnit(state.field)}]` : "");
  $("lmin").textContent = fmt(a); $("lmax").textContent = fmt(b);
}

function sensorValue(s) {
  if (state.live.has(s.meta.id)) return state.live.get(s.meta.id);
  return s.meta.series ? s.meta.series[Math.min(state.step, s.meta.series.length - 1)] : null;
}

function updateSensors() {
  for (const s of state.sensors) {
    const v = sensorValue(s);
    const env = s.meta.envelope;
    const alarm = v !== null && env && (v < env[0] || v > env[1]);
    const txt = v === null ? "—" : `${fmt(v)} ${s.meta.unit}`;
    s.row.classList.toggle("alarm", !!alarm);
    s.row.children[2].textContent = txt;
    s.row.title = env ? `Envelope válido: ${env[0]} … ${env[1]} ${s.meta.unit}` : "";
    s.tag.textContent = `${s.meta.label}: ${txt}`;
    s.tag.classList.toggle("alarm", !!alarm);
    s.ball.material.color.set(alarm ? 0xd6453d : 0x1f9d55);
  }
}

function connectLive() {
  const url = params.get("live");
  if (!url) return;
  const ws = new WebSocket(url);
  $("live").textContent = "ao vivo: conectando…";
  ws.onopen = () => { $("live").textContent = "ao vivo: conectado"; };
  ws.onclose = () => { $("live").textContent = "ao vivo: desconectado"; };
  ws.onerror = () => { $("live").textContent = "ao vivo: erro de conexão"; };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    for (const item of Array.isArray(msg) ? msg : [msg]) {
      if (item && typeof item.id === "string" && Number.isFinite(item.value)) state.live.set(item.id, item.value);
    }
    updateSensors();
  };
}

// ── probe (value under the cursor) ─────────────────────────────────────
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let probeQueued = false;
renderer.domElement.addEventListener("pointermove", (e) => {
  const r = renderer.domElement.getBoundingClientRect();
  pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  if (!probeQueued) { probeQueued = true; requestAnimationFrame(() => { probeQueued = false; probe(e.clientX - r.left, e.clientY - r.top); }); }
});
renderer.domElement.addEventListener("pointerleave", () => { $("probe").style.display = "none"; });

function probe(px, py) {
  raycaster.setFromCamera(pointer, camera);
  const meshes = [...state.parts.values()].filter((p) => p.mesh.visible).map((p) => p.mesh);
  const hits = raycaster.intersectObjects(meshes, false).filter((h) =>
    !h.object.material.clippingPlanes.length || state.clipPlane.distanceToPoint(h.point) >= 0);
  const el = $("probe");
  if (!hits.length) { el.style.display = "none"; return; }
  const h = hits[0];
  const part = [...state.parts.values()].find((p) => p.mesh === h.object);
  const pos = h.object.geometry.attributes.position;
  const local = h.object.worldToLocal(h.point.clone());
  let best = h.face.a, bestD = Infinity;
  for (const i of [h.face.a, h.face.b, h.face.c]) {
    const d = local.distanceToSquared(new THREE.Vector3().fromBufferAttribute(pos, i));
    if (d < bestD) { bestD = d; best = i; }
  }
  const vals = state.field ? valuesAt(part, state.field, state.step) : null;
  const u = state.manifest.length_unit;
  el.innerHTML = "";
  const lines = [part.meta.name,
    `(${fmt(h.point.x)}, ${fmt(h.point.y)}, ${fmt(h.point.z)}) ${u}`];
  if (vals) lines.push(`${state.field} = ${fmt(vals[best])} ${fieldUnit(state.field)}`);
  for (const l of lines) { const d = document.createElement("div"); d.textContent = l; el.appendChild(d); }
  el.style.display = "block";
  el.style.left = `${px + 14}px`;
  el.style.top = `${py + 14}px`;
}

// ── render loop ────────────────────────────────────────────────────────
const tmp = new THREE.Vector3();
function frame() {
  controls.update();
  renderer.render(scene3, camera);
  const w = viewport.clientWidth, h = viewport.clientHeight;
  for (const s of state.sensors) {
    tmp.copy(s.ball.position).project(camera);
    const visible = tmp.z < 1 && Math.abs(tmp.x) <= 1 && Math.abs(tmp.y) <= 1;
    s.tag.style.display = visible ? "block" : "none";
    if (visible) { s.tag.style.left = `${(tmp.x + 1) / 2 * w}px`; s.tag.style.top = `${(1 - tmp.y) / 2 * h}px`; }
  }
  requestAnimationFrame(frame);
}

applyTheme();
resize();
frame();
load().catch((e) => { console.error(e); showError(e.message); $("status").textContent = "erro ao carregar"; });
