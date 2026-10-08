// Realistic 3D aircraft viewer (three.js r169, vendored): physical sky lighting the paint (PMREM of the sky), sun with
// soft shadows, a runway on grass, spinning propeller; modes: realistic, surface pressure from OpenFOAM, stall map
// from the vortex lattice, animated streamlines. Axes: the API speaks aircraft axes (x aft, y right, z up); glTF/three
// is y-up with x = aircraft y, y = aircraft z, z = aircraft x.
import * as THREE from "three";
import { OrbitControls } from "/vendor/three/OrbitControls.js";
import { GLTFLoader } from "/vendor/three/loaders/GLTFLoader.js";
import { Sky } from "/vendor/three/objects/Sky.js";

const A2T = (p) => new THREE.Vector3(p[1], p[2], p[0]);
const DIVERGE = [[0, [33, 102, 172]], [0.35, [146, 197, 222]], [0.55, [247, 247, 247]], [0.75, [244, 165, 130]], [1, [178, 24, 43]]];
const STALL = [[0, [26, 152, 80]], [0.55, [166, 217, 106]], [0.75, [254, 224, 139]], [0.9, [244, 109, 67]], [1, [165, 0, 38]]];
function ramp(stops, t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], f = (t - a) / (b - a);
    return ca.map((c, k) => (c + f * (cb[k] - c)) / 255);
  }
  return stops[stops.length - 1][1].map((c) => c / 255);
}

function canvasTex(w, h, draw, repeat) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat[0], repeat[1]); }
  return t;
}
function noise(g, w, h, base, amp, n) {
  g.fillStyle = base; g.fillRect(0, 0, w, h);
  for (let i = 0; i < n; i++) { const v = (Math.random() - 0.5) * amp; g.fillStyle = `rgba(${v > 0 ? "255,255,255" : "0,0,0"},${Math.abs(v)})`; g.fillRect(Math.random() * w, Math.random() * h, 1 + Math.random() * 2, 1 + Math.random() * 2); }
}

class AircraftViewer {
  constructor(host) {
    this.host = host; this.mode = "real"; this.spin = true; this.flowOn = false;
    host.innerHTML = `<div class="av"><div class="av-bar">
        <div class="seg av-modes"><button data-m="real" class="on">Realistic</button><button data-m="cp">Pressure (OpenFOAM)</button><button data-m="stall">Stall map</button><button data-m="flow">Flow</button></div>
        <div class="av-right"><button data-c="hero">3/4</button><button data-c="front">Front</button><button data-c="side">Side</button><button data-c="top">Top</button><button data-c="rear">Rear</button>
          <label><input type="checkbox" data-k="spin" checked> prop</label><label><input type="checkbox" data-k="rot"> orbit</label><button data-k="shot">📷 PNG</button></div></div>
      <div class="av-stage"><div class="av-info"></div><div class="av-legend"></div><div class="av-load">loading the aircraft…</div></div></div>`;
    this.stage = host.querySelector(".av-stage");
    this.info = host.querySelector(".av-info");
    this.legend = host.querySelector(".av-legend");
    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 0.5;
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap;
    this.stage.prepend(r.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 5000);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true; this.controls.maxPolarAngle = Math.PI * 0.495; this.controls.minDistance = 4; this.controls.maxDistance = 60;
    this._sky();
    this._ground();
    host.querySelectorAll("[data-m]").forEach((b) => (b.onclick = () => this.setMode(b.dataset.m)));
    host.querySelectorAll("[data-c]").forEach((b) => (b.onclick = () => this.view(b.dataset.c)));
    host.querySelector("[data-k=spin]").onchange = (e) => (this.spin = e.target.checked);
    host.querySelector("[data-k=rot]").onchange = (e) => (this.controls.autoRotate = e.target.checked);
    host.querySelector("[data-k=shot]").onclick = () => { const a = document.createElement("a"); a.href = r.domElement.toDataURL("image/png"); a.download = "aircraft.png"; a.click(); };
    new ResizeObserver(() => this.resize()).observe(this.stage);
    this.resize();
    this.clock = new THREE.Clock();
    const loop = () => { this._tick(); requestAnimationFrame(loop); };
    loop();
  }

  _sky() {
    const sky = new Sky(); sky.scale.setScalar(4500);
    const u = sky.material.uniforms;
    u.turbidity.value = 4.5; u.rayleigh.value = 1.4; u.mieCoefficient.value = 0.004; u.mieDirectionalG.value = 0.86;
    const sun = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - 34), THREE.MathUtils.degToRad(145));
    u.sunPosition.value.copy(sun);
    this.scene.add(sky);
    const pm = new THREE.PMREMGenerator(this.renderer);
    const envScene = new THREE.Scene(); const sky2 = sky.clone(); envScene.add(sky2);
    this.scene.environment = pm.fromScene(envScene, 0.02).texture;
    this.scene.fog = new THREE.Fog(0xc9d6e2, 160, 1100);
    const light = this.sun = new THREE.DirectionalLight(0xfff4e5, 2.6);
    light.position.copy(sun).multiplyScalar(60);
    light.castShadow = true;
    light.shadow.mapSize.set(4096, 4096);
    const s = light.shadow.camera; s.left = -14; s.right = 14; s.top = 14; s.bottom = -14; s.near = 1; s.far = 160;
    light.shadow.bias = -0.0003; light.shadow.normalBias = 0.02; light.shadow.radius = 4;
    this.scene.add(light, light.target);
    this.scene.add(new THREE.HemisphereLight(0xdfe9f5, 0x5a6a45, 0.35));
  }

  _ground() {
    const grass = canvasTex(512, 512, (g, w, h) => noise(g, w, h, "#5d7342", 0.18, 26000), [90, 90]);
    const gm = new THREE.MeshStandardMaterial({ map: grass, roughness: 0.97, color: 0xffffff });
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(3000, 3000), gm);
    ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true;
    const rw = canvasTex(256, 2048, (g, w, h) => {
      noise(g, w, h, "#3d3f42", 0.08, 90000);
      g.fillStyle = "#e9e9e6";
      g.fillRect(6, 0, 5, h); g.fillRect(w - 11, 0, 5, h);
      for (let y = 0; y < h; y += 128) g.fillRect(w / 2 - 2.5, y + 20, 5, 70);
      for (let k = 0; k < 8; k++) { g.fillRect(28 + k * 13, h - 200, 8, 150); g.fillRect(w - 36 - k * 13, h - 200, 8, 150); }
    });
    const runway = new THREE.Mesh(new THREE.PlaneGeometry(30, 240), new THREE.MeshStandardMaterial({ map: rw, roughness: 0.82 }));
    runway.rotation.x = -Math.PI / 2; runway.position.set(0, 0.01, 60); runway.receiveShadow = true;
    this.groundGroup = new THREE.Group(); this.groundGroup.add(ground, runway);
    this.scene.add(this.groundGroup);
  }

  async load(d) {
    // d: {glb, ground_z, span, length, loading: {eta, ratio}, lines_vlm, lines_cfd, cp_available, title}
    this.d = d;
    this.host.querySelector(".av-load").style.display = "block";
    if (this.model) { this.scene.remove(this.model); this.model.traverse((o) => o.geometry && o.geometry.dispose()); }
    const gltf = await new GLTFLoader().loadAsync(d.glb);
    const m = this.model = gltf.scene;
    m.traverse((o) => {
      if (o.isMesh) {
        o.castShadow = o.receiveShadow = true;
        o.userData.mat = o.material;
        if (o.material.name === "glass") { o.material.envMapIntensity = 1.6; }
        if (o.material.clearcoat) { o.material.envMapIntensity = 1.2; }
      }
    });
    this.groundGroup.position.y = d.ground_z;
    this.scene.add(m);
    // propeller pivot at the spinner centre
    const prop = m.getObjectByName("propeller");
    if (prop) {
      const box = new THREE.Box3();
      prop.traverse((o) => { if (o.isMesh && o.name.startsWith("spinner")) box.expandByObject(o); });
      const hub = box.getCenter(new THREE.Vector3());
      const pivot = new THREE.Object3D(); pivot.position.copy(hub);
      m.add(pivot);
      prop.position.sub(hub); pivot.add(prop);
      this.prop = pivot;
    }
    const b = new THREE.Box3().setFromObject(m);
    this.centre = b.getCenter(new THREE.Vector3()); this.size = b.getSize(new THREE.Vector3()).length();
    this._buildLines();
    this.host.querySelector("[data-m=cp]").disabled = !d.cp_available;
    this.host.querySelector("[data-m=cp]").title = d.cp_available ? "" : "only for designs run in 3D OpenFOAM";
    this.host.querySelector(".av-load").style.display = "none";
    if (!this.viewed) { this.view("hero"); this.viewed = true; }
    this.setMode(this.mode === "cp" && !d.cp_available ? "real" : this.mode);
  }

  _buildLines() {
    if (this.lines) this.scene.remove(this.lines);
    const g = this.lines = new THREE.Group();
    const src = this.d.lines_cfd && this.d.lines_cfd.length ? this.d.lines_cfd : this.d.lines_vlm || [];
    this.lineSource = this.d.lines_cfd && this.d.lines_cfd.length ? "OpenFOAM" : "vortex lattice (potential flow)";
    this.particles = [];
    const mat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0ea5e9, emissiveIntensity: 0.9, roughness: 0.4, transparent: true, opacity: 0.85 });
    const pmat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const pg = new THREE.SphereGeometry(0.045, 8, 6);
    for (const L of src) {
      const pts = L.map(A2T);
      const curve = new THREE.CatmullRomCurve3(pts);
      g.add(new THREE.Mesh(new THREE.TubeGeometry(curve, Math.min(400, pts.length * 2), 0.018, 6, false), mat));
      for (let k = 0; k < 3; k++) { const s = new THREE.Mesh(pg, pmat); s.userData = { curve, t: k / 3 }; g.add(s); this.particles.push(s); }
    }
    g.visible = false;
    this.scene.add(g);
  }

  setMode(mode) {
    this.mode = mode;
    this.host.querySelectorAll("[data-m]").forEach((b) => b.classList.toggle("on", b.dataset.m === mode));
    const d = this.d || {};
    const grey = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.6, metalness: 0.0, envMapIntensity: 0.3 });
    this.model && this.model.traverse((o) => {
      if (!o.isMesh) return;
      const name = o.name || "";
      if (mode === "real" || mode === "flow") { o.material = o.userData.mat; return; }
      if (mode === "cp") {
        const a = o.geometry.getAttribute("_cp");
        if (!a) { o.material = grey; return; }
        o.geometry.setAttribute("color", this._colors(a.array, (v) => ramp(DIVERGE, (v + 1.2) / 2.0)));
      } else if (mode === "stall") {
        if (!/^wing_(left|right)/.test(name) || !d.loading) { o.material = grey; return; }
        const pos = o.geometry.getAttribute("position"), half = d.span / 2, L = d.loading;
        const vals = new Float32Array(pos.count);
        for (let i = 0; i < pos.count; i++) { const eta = Math.min(1, Math.abs(pos.getX(i)) / half); vals[i] = interp(L.eta, L.ratio, eta); }
        o.geometry.setAttribute("color", this._colors(vals, (v) => ramp(STALL, v)));
      }
      o.material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, metalness: 0.0, envMapIntensity: 0.3 });
    });
    if (this.lines) this.lines.visible = mode === "flow";
    const lg = this.legend;
    if (mode === "cp") lg.innerHTML = legendHTML(DIVERGE, "surface Cp (OpenFOAM)", "−1.2 suction", "+0.8 stagnation");
    else if (mode === "stall") lg.innerHTML = legendHTML(STALL, "local cl / cl max at stall onset", "0.0", "1.0 stalls first");
    else if (mode === "flow") lg.innerHTML = `<div class="av-lt">Streamlines</div><div class="meta" style="color:#334">${this.lineSource}</div>`;
    else lg.innerHTML = "";
    this.info.innerHTML = d.title ? `<b>${d.title}</b>` + (d.subtitle ? `<br>${d.subtitle}` : "") : "";
  }

  _colors(vals, fn) {
    const c = new Float32Array(vals.length * 3);
    const lin = (v) => Math.pow(v, 2.2);                       // vertex colours are linear in three.js
    for (let i = 0; i < vals.length; i++) { const rgb = fn(vals[i]); c[3 * i] = lin(rgb[0]); c[3 * i + 1] = lin(rgb[1]); c[3 * i + 2] = lin(rgb[2]); }
    return new THREE.BufferAttribute(c, 3);
  }

  view(name) {
    if (!this.centre) return;
    const c = this.centre, k = this.size / 13;
    const P = { hero: [-8.5, 1.7, -11.5], front: [0, 1.0, -17], side: [-17, 1.2, 0], top: [0.01, 19, 0], rear: [7.5, 3.2, 12.5] }[name];
    this.camera.position.set(c.x + P[0] * k, c.y + P[1] * k, c.z + P[2] * k);
    this.controls.target.copy(c);
    this.controls.update();
  }

  resize() {
    const w = this.stage.clientWidth, h = this.stage.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  _tick() {
    const dt = Math.min(0.05, this.clock.getDelta());
    if (this.prop && this.spin) this.prop.rotation.z += dt * 38;
    if (this.lines && this.lines.visible) for (const s of this.particles) {
      s.userData.t = (s.userData.t + dt * 0.12) % 1; s.position.copy(s.userData.curve.getPointAt(s.userData.t));
    }
    if (this.centre) { this.sun.target.position.copy(this.centre); this.sun.position.copy(this.centre).add(this.scene.children[0].material.uniforms.sunPosition.value.clone().multiplyScalar(60)); }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

function interp(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++) if (x <= xs[i]) return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[ys.length - 1];
}
function legendHTML(stops, title, lo, hi) {
  const g = Array.from({ length: 11 }, (_, k) => `rgb(${ramp(stops, k / 10).map((v) => Math.round(v * 255))}) ${k * 10}%`).join(",");
  return `<div class="av-lt">${title}</div><div style="height:10px;width:200px;border-radius:3px;background:linear-gradient(90deg,${g})"></div><div class="av-ll"><span>${lo}</span><span>${hi}</span></div>`;
}

window.ADO_AircraftViewer = AircraftViewer;
window.dispatchEvent(new Event("ado-viewer-ready"));
