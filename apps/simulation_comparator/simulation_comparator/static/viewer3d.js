// Error / value map on the reference geometry (three.js, vendored): boundary surface coloured per cell or per node.
import * as THREE from "three";
import { OrbitControls } from "/vendor/three/OrbitControls.js";

const fmt = (v) => (v === null || v === undefined || !isFinite(v) ? "—" : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(4) + "");

class FieldViewer {
  constructor(host, view, field, show, unit) {
    const s = view.surface, f = view.fields[field];
    const { cmap, SEQ_ERR, SEQ_VAL } = window.CMP_CMAP;
    host.innerHTML = `<div class="v3d-stage" style="position:relative"><div class="v3d-info"></div>
      <div class="v3d-legend"><div class="v3d-lt"></div><canvas width="18" height="220"></canvas><div class="v3d-ticks"></div></div><div class="v3d-probe"></div></div>`;
    this.stage = host.querySelector(".v3d-stage");
    const P = s.points, T = s.tris, nt = T.length / 3;
    let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], P[i + k]); mx[k] = Math.max(mx[k], P[i + k]); }
    const c0 = mn.map((a, k) => (a + mx[k]) / 2), sc = 100 / Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1e-30);
    const vals = f[show];
    const fin = vals.filter((x) => x !== null);
    const fmax = Math.max(...fin);
    let lo = show === "error" ? 0 : Math.min(...fin), hi = show === "error" ? window.CMP_P99(fin) || fmax : fmax;
    const clipped = show === "error" && hi < fmax;
    if (show !== "error") { const o = f[show === "ref" ? "cand" : "ref"].filter((x) => x !== null); lo = Math.min(lo, ...o); hi = Math.max(hi, ...o); }
    const stops = show === "error" ? SEQ_ERR : SEQ_VAL;
    const pos = new Float32Array(nt * 9), col = new Float32Array(nt * 9);
    this.triVal = new Float32Array(nt);
    for (let t = 0; t < nt; t++) for (let j = 0; j < 3; j++) {
      const p = T[3 * t + j];
      for (let k = 0; k < 3; k++) pos[9 * t + 3 * j + k] = (P[3 * p + k] - c0[k]) * sc;
      const v = f.kind === "surface_cell" ? vals[s.tri_cell[t]] : vals[p];
      const c = v === null ? [200, 200, 200] : cmap(stops, (v - lo) / (hi - lo || 1));
      col.set(c.map((x) => x / 255), 9 * t + 3 * j);
      if (j === 0) this.triVal[t] = v === null ? NaN : (f.kind === "surface_cell" ? v : NaN);
    }
    this.f = f; this.s = s; this.show = show;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3)); geo.setAttribute("color", new THREE.BufferAttribute(col, 3)); geo.computeVertexNormals();
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); this.renderer.setClearColor(0, 0);
    this.stage.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene(); this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 5000); this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xbbbbbb, 2.6));
    this.mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    this.scene.add(this.mesh);
    const flat = mx[2] - mn[2] < 0.05 * Math.max(mx[0] - mn[0], mx[1] - mn[1]);
    const d = new THREE.Vector3(...(flat ? [0, -0.02, 1] : [1, 0.8, 1.1])).normalize().multiplyScalar(flat ? 120 : 190);
    this.camera.position.copy(d); this.controls.update();
    const lc = this.stage.querySelector(".v3d-legend canvas"), g = lc.getContext("2d");
    for (let p = 0; p < 220; p++) { const c = cmap(stops, 1 - p / 219); g.fillStyle = `rgb(${c})`; g.fillRect(0, p, 18, 1); }
    this.stage.querySelector(".v3d-lt").textContent = `${{ error: "|error|", ref: "reference", cand: "candidate" }[show]}${unit ? ` (${unit})` : ""}`;
    this.stage.querySelector(".v3d-ticks").innerHTML = [hi, (lo + hi) / 2, lo].map((x, i) => `<span>${i === 0 && clipped ? "≥ " : ""}${fmt(x)}</span>`).join("");
    this.stage.querySelector(".v3d-info").innerHTML = `<b>${{ error: "Difference |candidate − reference|", ref: "Reference", cand: "Candidate (interpolated)" }[show]}</b><br>${clipped ? `colour scale clipped at the 99th percentile (max ${fmt(fmax)})<br>` : ""}drag to rotate · scroll to zoom`;
    this.ray = new THREE.Raycaster(); this.probe = this.stage.querySelector(".v3d-probe");
    this.renderer.domElement.addEventListener("mousemove", (e) => this.hover(e));
    this.renderer.domElement.addEventListener("mouseleave", () => (this.probe.style.display = "none"));
    new ResizeObserver(() => this.resize()).observe(this.stage); this.resize();
    const loop = () => { this.controls.update(); this.renderer.render(this.scene, this.camera); requestAnimationFrame(loop); }; loop();
  }
  resize() { const w = this.stage.clientWidth, h = this.stage.clientHeight; if (!w || !h) return; this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); }
  hover(e) {
    if (this.f.kind !== "surface_cell") return;
    const r = this.renderer.domElement.getBoundingClientRect();
    this.ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.camera);
    const hit = this.ray.intersectObject(this.mesh)[0];
    if (!hit) { this.probe.style.display = "none"; return; }
    const c = this.s.tri_cell[hit.faceIndex];
    this.probe.innerHTML = `reference <b>${fmt(this.f.ref[c])}</b> · candidate <b>${fmt(this.f.cand[c])}</b> · |error| <b>${fmt(this.f.error[c])}</b>`;
    this.probe.style.display = "block"; this.probe.style.left = Math.min(r.width - 320, e.clientX - r.left + 14) + "px"; this.probe.style.top = e.clientY - r.top + 14 + "px";
  }
}
window.FieldViewer = FieldViewer;
window.dispatchEvent(new Event("cmp-viewer-ready"));
