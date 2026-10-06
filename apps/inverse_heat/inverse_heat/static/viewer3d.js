// 3D temperature field of the block (three.js, vendored): one coloured box per finite-volume cell, a cut-away to see
// inside, network / finite-volume / difference views, the chip footprint and the thermocouples, a probe under the cursor.
import * as THREE from "three";
import { OrbitControls } from "/vendor/three/OrbitControls.js";

const DIVERGE = [[0, [33, 102, 172]], [0.5, [247, 247, 247]], [1, [178, 24, 43]]];
function stopsColor(stops, t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], f = (t - a) / (b - a);
    return ca.map((c, k) => (c + f * (cb[k] - c)) / 255);
  }
  return stops[stops.length - 1][1].map((c) => c / 255);
}

class BlockViewer {
  constructor(host, d) {
    this.d = d; this.view = "pinn"; this.cut = true;
    host.innerHTML = `<div class="v3d"><div class="v3d-toolbar">
        <div class="v3d-group"><select data-k="view"><option value="pinn">Network (PINN)</option><option value="ref">Finite volumes</option><option value="diff">Difference</option></select>
          <label><input type="checkbox" data-k="cut" checked> cut-away</label></div>
        <div class="v3d-group"><button data-k="iso">Iso</button><button data-k="top">Top</button><button data-k="side">Side</button></div></div>
      <div class="v3d-stage"><div class="v3d-info"></div>
        <div class="v3d-legend"><div class="v3d-lt"></div><canvas width="18" height="220"></canvas><div class="v3d-ticks"></div></div>
        <div class="v3d-probe"></div></div></div>`;
    this.stage = host.querySelector(".v3d-stage");
    this.info = host.querySelector(".v3d-info");
    this.probe = host.querySelector(".v3d-probe");
    const [nz, ny, nx] = d.shape, [lx, ly, lz] = d.size;
    this.dx = lx / nx; this.dy = ly / ny; this.dz = lz / nz;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.stage.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 1, 1000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 0);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(30, 60, 40); this.scene.add(sun);

    // cells: three.js y is up = block z; block (x, y) map to three (x, -z) so the view matches the 2D slices
    const geo = new THREE.BoxGeometry(this.dx * 0.97, this.dz * 0.97, this.dy * 0.97);
    const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    this.mesh = new THREE.InstancedMesh(geo, mat, nx * ny * nz);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.mesh);
    this.ids = [];

    // chip footprint under the bottom face and thermocouples on top
    const [xc, yc, half] = d.device;
    const chip = new THREE.Mesh(new THREE.BoxGeometry(2 * half, 1.2, 2 * half), new THREE.MeshLambertMaterial({ color: 0x334155 }));
    chip.position.set(xc - lx / 2, -lz / 2 - 0.7, -(yc - ly / 2)); this.scene.add(chip);
    const sg = new THREE.SphereGeometry(0.9, 16, 12), sm = new THREE.MeshLambertMaterial({ color: 0xffffff, emissive: 0x222222 });
    for (const [sx, sy] of d.sensors) { const s = new THREE.Mesh(sg, sm); s.position.set(sx - lx / 2, lz / 2 + 0.6, -(sy - ly / 2)); this.scene.add(s); }
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(lx, lz, ly)), new THREE.LineBasicMaterial({ color: 0x1f2a3d }));
    this.scene.add(edges);

    host.querySelector("[data-k=view]").onchange = (e) => { this.view = e.target.value; this.update(); };
    host.querySelector("[data-k=cut]").onchange = (e) => { this.cut = e.target.checked; this.update(); };
    host.querySelector("[data-k=iso]").onclick = () => this.setView(1.0, 0.8, 1.3);
    host.querySelector("[data-k=top]").onclick = () => this.setView(0, 1, 0.001);
    host.querySelector("[data-k=side]").onclick = () => this.setView(0, 0.05, 1);
    this.ray = new THREE.Raycaster(); this.mouse = new THREE.Vector2();
    this.renderer.domElement.addEventListener("mousemove", (e) => this.onMove(e));
    this.renderer.domElement.addEventListener("mouseleave", () => (this.probe.style.display = "none"));
    new ResizeObserver(() => this.resize()).observe(this.stage);
    this.setView(1.0, 0.8, 1.3);
    this.update(); this.resize();
    const loop = () => { this.controls.update(); this.renderer.render(this.scene, this.camera); requestAnimationFrame(loop); };
    loop();
  }
  setView(x, y, z) {
    const r = 75, v = new THREE.Vector3(x, y, z).normalize();
    this.camera.position.copy(v.multiplyScalar(r)); this.controls.target.set(0, 0, 0); this.controls.update();
  }
  resize() {
    const w = this.stage.clientWidth, h = this.stage.clientHeight;
    this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  value(n) {
    const { pinn, ref } = this.d;
    return this.view === "pinn" ? pinn[n] : this.view === "ref" ? ref[n] : pinn[n] - ref[n];
  }
  update() {
    const d = this.d, [nz, ny, nx] = d.shape, [lx, ly, lz] = d.size;
    let lo = d.lo, hi = d.hi, stops = null;
    if (this.view === "diff") { let m = 0; for (let n = 0; n < d.pinn.length; n++) m = Math.max(m, Math.abs(d.pinn[n] - d.ref[n])); lo = -m; hi = m; stops = DIVERGE; }
    const m4 = new THREE.Matrix4(), col = new THREE.Color();
    let c = 0; this.ids = [];
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) * this.dx, y = (j + 0.5) * this.dy, z = (k + 0.5) * this.dz;
      if (this.cut && x > lx / 2 + 2 && y < ly / 2 - 2) continue;              // remove the front quarter, off-centre
      const n = (k * ny + j) * nx + i, t = (this.value(n) - lo) / (hi - lo);
      m4.makeTranslation(x - lx / 2, z - lz / 2, -(y - ly / 2));
      this.mesh.setMatrixAt(c, m4);
      const rgb = stops ? stopsColor(stops, t) : window.IHL_CMAP(t).map((v) => v / 255);
      this.mesh.setColorAt(c, col.setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace));
      this.ids.push(n); c++;
    }
    this.mesh.count = c;
    this.mesh.instanceMatrix.needsUpdate = true; this.mesh.instanceColor.needsUpdate = true;
    // legend
    const lc = this.stage.querySelector(".v3d-legend canvas"), g = lc.getContext("2d");
    for (let p = 0; p < 220; p++) {
      const t = 1 - p / 219, rgb = stops ? stopsColor(stops, t).map((v) => v * 255) : window.IHL_CMAP(t);
      g.fillStyle = `rgb(${rgb.map(Math.round).join(",")})`; g.fillRect(0, p, 18, 1);
    }
    const unit = this.view === "diff" ? "K" : "°C";
    this.stage.querySelector(".v3d-lt").textContent = this.view === "diff" ? "PINN − FV (K)" : "T (°C)";
    this.stage.querySelector(".v3d-ticks").innerHTML = [hi, (lo + hi) / 2, lo].map((v) => `<span>${v.toFixed(this.view === "diff" ? 2 : 1)}</span>`).join("");
    const label = { pinn: "Network (PINN)", ref: "Finite volumes (check)", diff: "Network − finite volumes" }[this.view];
    this.info.innerHTML = `<b>${label}</b><br>chip (dark) underneath · ○ thermocouples on top<br>drag to rotate · scroll to zoom${this.view !== "diff" ? "" : ` · max |Δ| ${hi.toFixed(2)} ${unit}`}`;
  }
  onMove(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(this.mouse, this.camera);
    const hit = this.ray.intersectObject(this.mesh)[0];
    if (!hit || hit.instanceId === undefined) { this.probe.style.display = "none"; return; }
    const n = this.ids[hit.instanceId], [, ny, nx] = this.d.shape;
    const i = n % nx, j = Math.floor(n / nx) % ny, k = Math.floor(n / (nx * ny));
    this.probe.textContent = `x ${((i + 0.5) * this.dx).toFixed(1)}, y ${((j + 0.5) * this.dy).toFixed(1)}, z ${((k + 0.5) * this.dz).toFixed(2)} mm: ` +
      `PINN ${this.d.pinn[n].toFixed(2)} °C · FV ${this.d.ref[n].toFixed(2)} °C`;
    this.probe.style.display = "block";
    this.probe.style.left = Math.min(r.width - 330, e.clientX - r.left + 12) + "px"; this.probe.style.top = e.clientY - r.top + 12 + "px";
  }
}

window.IHL_BlockViewer = BlockViewer;
window.dispatchEvent(new Event("ihl-viewer-ready"));
