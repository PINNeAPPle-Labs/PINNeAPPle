// Preview of a field on the boundary of the dataset's mesh (three.js, vendored).
import * as THREE from "three";
import { OrbitControls } from "/vendor/three/OrbitControls.js";
const fmt = (v) => (v === null || v === undefined || !isFinite(v) ? "—" : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(4) + "");
const STOPS = [[0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]], [0.75, [94, 201, 98]], [1, [253, 231, 37]]];
function cmap(t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < STOPS.length; i++) if (t <= STOPS[i][0]) { const [a, ca] = STOPS[i - 1], [b, cb] = STOPS[i], f = (t - a) / (b - a); return ca.map((c, k) => (c + f * (cb[k] - c)) / 255); }
  return STOPS[4][1].map((c) => c / 255);
}
class PreviewViewer {
  constructor(host, pv, field) {
    this.pv = pv;
    host.innerHTML = `<div class="v3d-stage" style="position:relative"><div class="v3d-legend"><div class="v3d-lt"></div><canvas width="18" height="220"></canvas><div class="v3d-ticks"></div></div></div>`;
    this.stage = host.querySelector(".v3d-stage");
    const P = pv.points, T = pv.tris; this.nt = T.length / 3;
    let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], P[i + k]); mx[k] = Math.max(mx[k], P[i + k]); }
    const c0 = mn.map((a, k) => (a + mx[k]) / 2), sc = 100 / Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1e-30);
    const pos = new Float32Array(this.nt * 9);
    for (let t = 0; t < this.nt; t++) for (let j = 0; j < 3; j++) { const p = T[3 * t + j]; for (let k = 0; k < 3; k++) pos[9 * t + 3 * j + k] = (P[3 * p + k] - c0[k]) * sc; }
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute("position", new THREE.BufferAttribute(pos, 3)); this.geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(this.nt * 9), 3)); this.geo.computeVertexNormals();
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true }); this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); this.renderer.setClearColor(0, 0);
    this.stage.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene(); this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 5000); this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x9a9a9a, 1.9));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9); sun.position.set(1, -1.5, 2); this.scene.add(sun);
    this.scene.add(new THREE.Mesh(this.geo, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })));
    // mesh edges: always for a bare mesh, faint over a field when the surface is not too dense
    if (!field || this.nt < 12000) {
      const edges = new THREE.LineSegments(new THREE.WireframeGeometry(this.geo), new THREE.LineBasicMaterial({ color: 0x0f172a, transparent: true, opacity: field ? 0.12 : 0.35 }));
      this.scene.add(edges);
    }
    const flat = mx[2] - mn[2] < 0.12 * Math.min(mx[0] - mn[0], mx[1] - mn[1]);   // a 2D case (thin in z), not a slender beam
    this.camera.position.copy(new THREE.Vector3(...(flat ? [0, -0.02, 1] : [1, 0.8, 1.1])).normalize().multiplyScalar(flat ? 185 : 200)); this.controls.update();
    this.paint(field);
    new ResizeObserver(() => this.resize()).observe(this.stage); this.resize();
    const loop = () => { this.controls.update(); this.renderer.render(this.scene, this.camera); requestAnimationFrame(loop); }; loop();
  }
  paint(field) {
    const leg = this.stage.querySelector(".v3d-legend");
    if (!field || !this.pv.fields[field]) {                               // bare mesh: light grey
      this.geo.attributes.color.array.fill(0.78); this.geo.attributes.color.needsUpdate = true; leg.style.display = "none"; return;
    }
    leg.style.display = "";
    const f = this.pv.fields[field], v = f.values, fin = v.filter((x) => x !== null);
    const lo = Math.min(...fin), hi = Math.max(...fin), col = this.geo.attributes.color.array, T = this.pv.tris;
    for (let t = 0; t < this.nt; t++) for (let j = 0; j < 3; j++) {
      const x = f.where === "cell" ? v[this.pv.tri_cell[t]] : v[T[3 * t + j]];
      col.set((x === null ? [0.8, 0.8, 0.8] : cmap((x - lo) / (hi - lo || 1))).map((c) => Math.pow(c, 2.2)), 9 * t + 3 * j);   // vertex colours are linear
    }
    this.geo.attributes.color.needsUpdate = true;
    const lc = this.stage.querySelector(".v3d-legend canvas"), g = lc.getContext("2d");
    for (let p = 0; p < 220; p++) { const c = cmap(1 - p / 219).map((x) => Math.round(x * 255)); g.fillStyle = `rgb(${c})`; g.fillRect(0, p, 18, 1); }
    this.stage.querySelector(".v3d-lt").textContent = field + (f.kind && f.kind !== "value" ? ` (${f.kind})` : ""); this.stage.querySelector(".v3d-ticks").innerHTML = [hi, (lo + hi) / 2, lo].map((x) => `<span>${fmt(x)}</span>`).join("");
  }
  resize() { const w = this.stage.clientWidth, h = this.stage.clientHeight; if (!w || !h) return; this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); }
}
window.PreviewViewer = PreviewViewer;
window.dispatchEvent(new Event("iop-viewer-ready"));
