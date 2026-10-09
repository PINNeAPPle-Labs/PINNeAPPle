// Realistic 3D aircraft viewer (three.js r169, vendored): physical sky lighting the paint (PMREM of the sky), sun with
// soft shadows, a runway on grass or a cloud deck in flight; modes: realistic, OpenFOAM skin fields (Cp, Cf), streamlines
// coloured by speed, OpenFOAM slices (symmetry plane, wake), stall map from the vortex lattice. CFD colours use the
// usual jet scale: blue = low, red = high. Axes: the API speaks aircraft axes (x aft, y right, z up); glTF/three
// is y-up with x = aircraft y, y = aircraft z, z = aircraft x. Colour scales, colour bars, tubes, slice textures and
// edges come from the PINNeAPPle studio core (/studio/studio-core.js), shared with pp.viz.web_viewer.
import * as THREE from "three";
import { OrbitControls } from "/vendor/three/OrbitControls.js";
import { GLTFLoader } from "/vendor/three/loaders/GLTFLoader.js";
import { Sky } from "/vendor/three/objects/Sky.js";
import * as core from "/studio/studio-core.js";

const A2T = (p) => new THREE.Vector3(p[1], p[2], p[0]);
const STALL = [[0, [26, 152, 80]], [0.55, [166, 217, 106]], [0.75, [254, 224, 139]], [0.9, [244, 109, 67]], [1, [165, 0, 38]]];

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
    this.range = {}; this.density = 1; this.edges = "off";       // viewer controls: colour range per field, lines, edges
    host.innerHTML = `<div class="av"><div class="av-bar">
        <div class="seg av-modes"><button data-m="real" class="on">Realistic</button><button data-m="cp">Surface (CFD)</button><button data-m="flow">Streamlines</button><button data-m="slice">Slices (CFD)</button><button data-m="stall">Stall map</button></div>
        <div class="av-right"><button data-c="hero">3/4</button><button data-c="front">Front</button><button data-c="side">Side</button><button data-c="top">Top</button><button data-c="rear">Rear</button>
          <label><input type="checkbox" data-k="spin" checked> prop</label><label><input type="checkbox" data-k="rot"> orbit</label><button data-k="shot">📷 PNG</button></div></div>
      <div class="av-stage"><div class="av-info"></div><div class="av-legend"></div><div class="av-field"></div><div class="av-load">loading the aircraft…</div></div></div>`;
    this.stage = host.querySelector(".av-stage");
    this.info = host.querySelector(".av-info");
    this.legend = host.querySelector(".av-legend");
    this.fieldBox = host.querySelector(".av-field");
    this.field = { cp: "cp", slice: "sym_cp" };
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

  _cloudDeck() {
    const tex = canvasTex(1024, 1024, (g, w, h) => {
      g.fillStyle = "#7d97b3"; g.fillRect(0, 0, w, h);
      for (let i = 0; i < 2600; i++) {
        const x = Math.random() * w, y = Math.random() * h, r = 8 + Math.random() * 46, a = 0.05 + Math.random() * 0.12;
        const gr = g.createRadialGradient(x, y, 0, x, y, r); gr.addColorStop(0, `rgba(255,255,255,${a * 2})`); gr.addColorStop(1, "rgba(255,255,255,0)");
        g.fillStyle = gr; g.beginPath(); g.arc(x, y, r, 0, 7); g.fill();
      }
    }, [6, 6]);
    tex.wrapS = tex.wrapT = THREE.MirroredRepeatWrapping;
    const deck = this.clouds = new THREE.Mesh(new THREE.PlaneGeometry(20000, 20000), new THREE.MeshStandardMaterial({ map: tex, roughness: 1, color: 0xffffff }));
    deck.rotation.x = -Math.PI / 2; deck.position.y = -650;
    this.scene.add(deck);
  }

  async load(d) {
    // d: {glb, ground_z, span, loading: {eta, ratio}, lines_vlm, cfd (OpenFOAM id or null), flight, title, subtitle}
    this.d = d;
    this.cfd = null;
    if (d.cfd) {
      try { this.cfd = await (await fetch(`/api/cfd3d/${d.cfd}`)).json(); } catch (e) { this.cfd = null; }
    }
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
    this.groundGroup.visible = !d.flight;
    if (d.flight && !this.clouds) this._cloudDeck();
    if (this.clouds) this.clouds.visible = !!d.flight;
    this.scene.fog = d.flight ? new THREE.Fog(0xb9cde0, 900, 6000) : new THREE.Fog(0xc9d6e2, 160, 1100);
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
    this.controls.minDistance = this.size * 0.25; this.controls.maxDistance = this.size * 5;
    this.controls.maxPolarAngle = d.flight ? Math.PI : Math.PI * 0.495;
    const sc = this.sun.shadow.camera, e = this.size * 0.75; sc.left = -e; sc.right = e; sc.top = e; sc.bottom = -e; sc.far = this.size * 6; sc.updateProjectionMatrix();
    this._buildLines();
    this._buildSlices();
    for (const m of ["cp", "slice"]) {
      const b = this.host.querySelector(`[data-m=${m}]`);
      b.disabled = !this.cfd; b.title = this.cfd ? "" : "only for designs run in OpenFOAM 3D (the A320 class and the balanced pick)";
    }
    this.range = {};                                            // a new design starts on the app's scales
    this._applyEdges();
    this.host.querySelector(".av-load").style.display = "none";
    if (!this.viewed) { this.view("hero"); this.viewed = true; }
    this.setMode(["cp", "slice"].includes(this.mode) && !this.cfd ? "real" : this.mode);
  }

  _buildLines() {
    // OpenFOAM streamlines (approach, both sides) coloured by |U|/U∞; otherwise the vortex-lattice ones at cruise
    if (this.lines) this.scene.remove(this.lines);
    const cfd = this.cfd && this.cfd.lines && this.cfd.lines.length;
    let src = this.d.lines_vlm || [], spd = null;
    if (cfd) {
      // each traced line and its mirror image (half model) sit side by side: 2i and 2i + 1
      src = []; spd = [];
      this.cfd.lines.forEach((L, i) => { src.push(L, L.map((p) => [p[0], -p[1], p[2]])); spd.push(this.cfd.line_speed[i], this.cfd.line_speed[i]); });
    }
    this.mirrored = !!cfd;
    this.lineSource = cfd ? `OpenFOAM, approach at ${this.cfd.speed} m/s, α ${this.cfd.alpha}°` : "vortex lattice (potential flow) at cruise";
    const [lo, hi] = cfd ? this.cfd.ranges.speed : [0, 1];
    this.lineRange = [lo, hi];
    this.lineVals = spd ? spd.flat() : [];
    const sz = this.size || 12;
    const g = this.lines = core.tubeLines(THREE, src, spd, { lo, hi, radius: 0.0018 * sz, toV3: A2T, eps: 1e-3 });
    g.userData.n = src.length;
    this.particles = [];
    const pmat = new THREE.MeshBasicMaterial({ color: 0xffffff }), pg = new THREE.SphereGeometry(0.0045 * sz, 8, 6);
    for (const tube of [...g.children]) for (let k = 0; k < 3; k++) {
      const p = new THREE.Mesh(pg, pmat); p.userData = { curve: tube.userData.curve, t: k / 3, index: tube.userData.index };
      g.add(p); this.particles.push(p);
    }
    g.visible = false;
    this.scene.add(g);
  }

  _lineDensity() {
    // thin the lines evenly; mirrored pairs stay together
    const n = this.lines.userData.n, pairs = this.mirrored ? n / 2 : n;
    const keep = core.thin(pairs, this.density);
    const on = new Set(keep.flatMap((i) => (this.mirrored ? [2 * i, 2 * i + 1] : [i])));
    this.lines.children.forEach((c) => (c.visible = on.has(c.userData.index)));
    return [keep.length, pairs];
  }

  _buildSlices() {
    // textured planes from the OpenFOAM slices: the symmetry plane (y = 0) and a crossflow plane in the wake
    if (this.slices) this.scene.remove(this.slices);
    this.slices = new THREE.Group(); this.slicePlanes = {};
    if (!this.cfd) return;
    const quad = (P) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(P.flatMap((p) => A2T(p).toArray()), 3));
      g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
      g.setIndex([0, 1, 2, 0, 2, 3]);
      return g;
    };
    const S = this.cfd.sym, W = this.cfd.wake;
    this.slicePlanes.sym = new THREE.Mesh(quad([[S.x[0], S.y, S.z[0]], [S.x[1], S.y, S.z[0]], [S.x[1], S.y, S.z[1]], [S.x[0], S.y, S.z[1]]]));
    this.slicePlanes.wake = new THREE.Mesh(quad([[W.x, -W.y[1], W.z[0]], [W.x, W.y[1], W.z[0]], [W.x, W.y[1], W.z[1]], [W.x, -W.y[1], W.z[1]]]));
    for (const m of Object.values(this.slicePlanes)) { m.renderOrder = 2; this.slices.add(m); }
    this.slices.visible = false;
    this.scene.add(this.slices);
  }

  _fieldButtons(mode) {
    const opts = mode === "cp" ? [["cp", "Pressure Cp"], ["cf", "Skin friction Cf"]]
      : mode === "slice" ? [["sym_cp", "Symmetry plane: Cp"], ["sym_speed", "Symmetry plane: speed"], ["wake_vort", "Wake: vorticity"], ["wake_speed", "Wake: speed"]] : [];
    this.fieldBox.innerHTML = (opts.length ? `<div class="seg">${opts.map(([k, l]) => `<button data-f="${k}" class="${this.field[mode] === k ? "on" : ""}">${l}</button>`).join("")}</div>` : "")
      + `<div class="av-ctl"></div>`;
    this.fieldBox.querySelectorAll("[data-f]").forEach((b) => (b.onclick = () => {
      const was = this.field[mode]; this.field[mode] = b.dataset.f; this.setMode(mode);
      if (mode === "slice" && was.slice(0, 3) !== b.dataset.f.slice(0, 3)) this.view(b.dataset.f.startsWith("sym") ? "side" : "rear");
    }));
  }

  // the values and the default (server) colour range shown in the current mode, for the range control
  _scale() {
    const R = (this.cfd && this.cfd.ranges) || {};
    if (this.mode === "cp" && this.cfd && this.model) {
      const fk = this.field.cp, vals = [];
      this.model.traverse((o) => { const a = o.isMesh && o.geometry.getAttribute("_" + fk); if (a) for (const v of a.array) vals.push(v); });
      return { key: "cp:" + fk, vals, def: R[fk] || [0, 1] };
    }
    if (this.mode === "slice" && this.cfd) {
      const sk = this.field.slice, S = this.cfd.sym, W = this.cfd.wake;
      const grid = { sym_cp: S.cp, sym_speed: S.speed, wake_vort: W.vorticity, wake_speed: W.speed }[sk];
      const def = { sym_cp: R.sym_cp, sym_speed: R.sym_speed, wake_vort: R.vorticity, wake_speed: R.wake_speed }[sk];
      return { key: "slice:" + sk, vals: grid.flat(), def };
    }
    if (this.mode === "flow" && this.lineVals && this.lineVals.length) return { key: "flow", vals: this.lineVals, def: this.lineRange };
    return null;
  }
  _range() { const sc = this._scale(); return sc ? this.range[sc.key] || sc.def : null; }

  _controls() {
    const box = this.fieldBox.querySelector(".av-ctl"), sc = this._scale(), rows = [];
    if (sc) {
      const [lo, hi] = this._range(), d = core.decimals(lo, hi) + 1;
      rows.push(`<label>Colour range <input data-r="lo" type="number" step="any" value="${+lo.toFixed(d)}"> to <input data-r="hi" type="number" step="any" value="${+hi.toFixed(d)}"></label>`
        + `<button data-r="auto" title="1st to 99th percentile">Auto</button><button data-r="full" title="minimum to maximum">Full</button><button data-r="reset" title="the app's scale">Reset</button>`);
    }
    if (this.mode === "flow" && this.lines) {
      const [k, n] = this._lineDensity();
      rows.push(`<label>Line density <input data-r="dens" type="range" min="0.05" max="1" step="0.05" value="${this.density}"></label><span data-r="count">${k} / ${n}</span>`);
    }
    rows.push(`<label>Edges <select data-r="edges">${[["off", "Off"], ["feature", "Feature edges"], ["wire", "Wireframe"]].map(([v, t]) => `<option value="${v}" ${this.edges === v ? "selected" : ""}>${t}</option>`).join("")}</select></label>`);
    box.innerHTML = rows.map((r) => `<div>${r}</div>`).join("");
    const q = (r) => box.querySelector(`[data-r=${r}]`);
    if (sc) {
      const set = () => { const lo = parseFloat(q("lo").value), hi = parseFloat(q("hi").value); if (hi > lo) { this.range[sc.key] = [lo, hi]; this.setMode(this.mode); } };
      q("lo").onchange = set; q("hi").onchange = set;
      q("auto").onclick = () => { this.range[sc.key] = core.autoRange(sc.vals, "auto"); this.setMode(this.mode); };
      q("full").onclick = () => { this.range[sc.key] = core.autoRange(sc.vals, "full"); this.setMode(this.mode); };
      q("reset").onclick = () => { delete this.range[sc.key]; this.setMode(this.mode); };
    }
    if (q("dens")) q("dens").oninput = (e) => { this.density = +e.target.value; const [k, n] = this._lineDensity(); q("count").textContent = `${k} / ${n}`; };
    q("edges").onchange = (e) => { this.edges = e.target.value; this._applyEdges(); };
  }

  _applyEdges() {
    this.model && this.model.traverse((o) => { if (o.isMesh) core.setEdges(THREE, o, this.edges); });
  }

  setMode(mode) {
    const prev = this.mode;
    this.mode = mode;
    if (this.centre && mode !== prev) {
      if (mode === "cp" || (mode === "flow" && !["cp", "slice"].includes(prev))) this.view("high");
      if (mode === "slice") this.view(this.field.slice.startsWith("sym") ? "side" : "rear");
    }
    this.host.querySelectorAll("[data-m]").forEach((b) => b.classList.toggle("on", b.dataset.m === mode));
    const d = this.d || {};
    const grey = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.6, metalness: 0.0, envMapIntensity: 0.3 });
    const ghost = new THREE.MeshStandardMaterial({ color: 0xb8c0c8, roughness: 0.6, transparent: true, opacity: 0.35, depthWrite: false });
    const fk = this.field.cp, [flo, fhi] = (mode === "cp" && this._range()) || [0, 1];
    this.model && this.model.traverse((o) => {
      if (!o.isMesh) return;
      const name = o.name || "";
      if (mode === "real" || mode === "flow") { o.material = o.userData.mat; return; }
      if (mode === "slice") { o.material = ghost; return; }
      if (mode === "cp") {
        const a = o.geometry.getAttribute("_" + fk);
        if (!a) { o.material = grey; return; }
        o.geometry.setAttribute("color", core.colorAttribute(THREE, a.array, flo, fhi));
      } else if (mode === "stall") {
        if (!/^wing_(left|right)/.test(name) || !d.loading) { o.material = grey; return; }
        const pos = o.geometry.getAttribute("position"), half = d.span / 2, L = d.loading;
        const vals = new Float32Array(pos.count);
        for (let i = 0; i < pos.count; i++) { const eta = Math.min(1, Math.abs(pos.getX(i)) / half); vals[i] = interp(L.eta, L.ratio, eta); }
        o.geometry.setAttribute("color", core.colorAttribute(THREE, vals, 0, 1, STALL));
      }
      o.material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, metalness: 0.0, envMapIntensity: 0.3 });
    });
    if (this.lines) this.lines.visible = mode === "flow";
    if (this.slices) this.slices.visible = mode === "slice";
    const lg = this.legend;
    this._fieldButtons(mode);
    if (this.lines && mode === "flow" && this.lineVals.length) { const [a, b] = this._range(); core.recolourLines(THREE, this.lines, a, b); }
    if (mode === "cp") lg.innerHTML = fk === "cp" ? colorbar("Pressure coefficient Cp (OpenFOAM)", flo, fhi, 2, "suction", "stagnation")
      : colorbar("Skin friction coefficient Cf (OpenFOAM)", flo, fhi, 4, "low shear", "high shear");
    else if (mode === "slice" && this.cfd) {
      const sk = this.field.slice, S = this.cfd.sym, W = this.cfd.wake;
      const on = sk.startsWith("sym") ? "sym" : "wake";
      const grid = { sym_cp: S.cp, sym_speed: S.speed, wake_vort: W.vorticity, wake_speed: W.speed }[sk];
      const [lo, hi] = this._range();
      const key = `${this.cfd.id}:${sk}:${lo}:${hi}`;
      if (this.sliceKey !== key) {
        const tex = core.sliceTexture(THREE, grid, lo, hi, on === "wake" ? (sk === "wake_vort" ? "odd" : "even") : null);
        const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, toneMapped: false, depthWrite: false });
        const m = this.slicePlanes[on]; if (m.material.map) m.material.map.dispose(); m.material = mat;
        this.sliceKey = key;
      }
      this.slicePlanes.sym.visible = on === "sym"; this.slicePlanes.wake.visible = on === "wake";
      lg.innerHTML = {
        sym_cp: colorbar("Pressure Cp on the symmetry plane", lo, hi, 2, "suction", "stagnation"),
        sym_speed: colorbar("Speed |U|/U∞ on the symmetry plane", lo, hi, 2, "slower", "faster"),
        wake_vort: colorbar(`Streamwise vorticity ωx·c/U∞, ${(W.x).toFixed(1)} m behind the nose`, lo, hi, 2, "clockwise", "anticlockwise"),
        wake_speed: colorbar("Speed |U|/U∞ in the wake plane", lo, hi, 3, "wake deficit", "faster"),
      }[sk];
    } else if (mode === "stall") lg.innerHTML = core.colorbarHTML({ title: "local cl / cl max at stall onset", lo: 0, hi: 1, nd: 2, hiTxt: "stalls first", stops: STALL, width: 220 });
    else if (mode === "flow") lg.innerHTML = this.cfd ? colorbar("Speed along the streamlines |U|/U∞", ...this._range(), 2, "slower", "faster") + `<div class="meta" style="color:#334">${this.lineSource}</div>`
      : `<div class="av-lt">Streamlines</div><div class="meta" style="color:#334">${this.lineSource}</div>`;
    else lg.innerHTML = "";
    this._controls();
    this.info.innerHTML = d.title ? `<b>${d.title}</b>` + (d.subtitle ? `<br>${d.subtitle}` : "") : "";
  }

  view(name) {
    if (!this.centre) return;
    const c = this.centre, k = this.size / 13;
    const P = this.d && this.d.flight
      ? { hero: [-8.5, -1.6, -11.0], high: [-9.0, 7.5, -9.5], front: [0, 0.6, -16.5], side: [-17, -0.5, 0], top: [0.01, 18, 0], rear: [8.0, 3.5, 12.0] }[name]
      : { hero: [-8.5, 1.7, -11.5], high: [-9.0, 7.5, -9.5], front: [0, 1.0, -17], side: [-17, 1.2, 0], top: [0.01, 19, 0], rear: [7.5, 3.2, 12.5] }[name];
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
      s.userData.t = (s.userData.t + dt * 0.08) % 1; s.position.copy(s.userData.curve.getPointAt(s.userData.t));
    }
    if (this.centre) { this.sun.target.position.copy(this.centre); this.sun.position.copy(this.centre).add(this.scene.children[0].material.uniforms.sunPosition.value.clone().multiplyScalar(this.size * 2)); }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

function interp(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++) if (x <= xs[i]) return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[ys.length - 1];
}
const colorbar = (title, lo, hi, nd, loTxt, hiTxt) => core.colorbarHTML({ title, lo, hi, nd, loTxt, hiTxt });

window.ADO_AircraftViewer = AircraftViewer;
window.dispatchEvent(new Event("ado-viewer-ready"));
