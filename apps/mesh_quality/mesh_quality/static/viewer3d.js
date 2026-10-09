// 3D view of the mesh boundary coloured by a quality metric, with the problem regions labelled (three.js, vendored).
import * as THREE from "three";
import { OrbitControls } from "/vendor/three/OrbitControls.js";

const fmt = (v) => (v === null || v === undefined || !isFinite(v) ? "—" : Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(4) + "");
const STOPS_OK = [[0, [224, 236, 248]], [0.35, [158, 202, 225]], [0.7, [66, 146, 198]], [1, [8, 69, 148]]];

function lerp(stops, t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], f = (t - a) / (b - a);
    return ca.map((c, k) => (c + f * (cb[k] - c)) / 255);
  }
  return stops[stops.length - 1][1].map((c) => c / 255);
}

// colour for a value: good values in cool greys/blues (lighter = better), beyond warn amber, beyond fail red
function colourFor(v, info, range) {
  if (v === null || v === undefined || !isFinite(v)) return [0.8, 0.8, 0.8];
  const low = info.better === "low";
  const warn = info.warn, fail = info.fail;
  if (fail !== null && (low ? v > fail : v <= fail)) return [0.71, 0.14, 0.09];
  if (warn !== null && (low ? v > warn : v < warn)) return [0.96, 0.62, 0.04];
  // good values: spread over the data range (up to the threshold), light = best, dark blue = closest to the limit
  let t;
  if (warn !== null) {
    const top = low ? Math.min(range[1], warn) : Math.max(range[0], warn);
    const best = low ? range[0] : range[1];
    t = Math.abs(v - best) / Math.max(Math.abs(top - best), 1e-30);
  } else t = (Math.log10(Math.max(v, 1e-30)) - Math.log10(Math.max(range[0], 1e-30))) / Math.max(1e-9, Math.log10(Math.max(range[1], 1e-30)) - Math.log10(Math.max(range[0], 1e-30)));
  return lerp(STOPS_OK, t);
}

class MeshViewer {
  constructor(host, rep) {
    this.rep = rep;
    const v = rep.view, info = rep.metric_info;
    this.fields = Object.keys(v.fields);
    const graded = this.fields.filter((k) => rep.metrics[k] && rep.metrics[k].graded);
    const pref = ["non_orthogonality", "skewness", "scaled_jacobian", "element_skewness", "aspect_ratio", "edge_ratio", "volume"];
    const bad = graded.filter((k) => rep.metrics[k].status !== "pass");
    this.field = (bad.length ? bad : graded.length ? graded : this.fields).sort((a, b) => pref.indexOf(a) - pref.indexOf(b))[0];
    host.innerHTML = `<div class="v3d"><div class="v3d-toolbar">
        <div class="v3d-group"><select data-k="field">${this.fields.map((k) => `<option value="${k}" ${k === this.field ? "selected" : ""}>${info[k].label}</option>`).join("")}</select>
          <label><input type="checkbox" data-k="only"> problems only</label><label><input type="checkbox" data-k="wire"> edges</label></div>
        <div class="v3d-group"><button data-k="iso">Iso</button><button data-k="top">Top</button><button data-k="front">Front</button><button data-k="side">Side</button></div></div>
      <div class="v3d-stage"><div class="v3d-info"></div>
        <div class="v3d-legend"><div class="v3d-lt"></div><canvas width="18" height="220"></canvas><div class="v3d-ticks"></div></div>
        <div class="v3d-probe"></div></div></div>`;
    this.stage = host.querySelector(".v3d-stage");
    this.probe = host.querySelector(".v3d-probe");
    const P = v.points, T = v.tris;
    let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], P[i + k]); mx[k] = Math.max(mx[k], P[i + k]); }
    this.centre = mn.map((a, k) => (a + mx[k]) / 2);
    this.scale = 100 / Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1e-30);
    this.nt = T.length / 3;
    const pos = new Float32Array(this.nt * 9);
    for (let t = 0; t < this.nt; t++) for (let j = 0; j < 3; j++) {
      const p = T[3 * t + j];
      for (let k = 0; k < 3; k++) pos[9 * t + 3 * j + k] = (P[3 * p + k] - this.centre[k]) * this.scale;
    }
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(this.nt * 9), 3));
    this.geo.computeVertexNormals();
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.stage.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 5000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x99aabb, 2.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.0); sun.position.set(60, 100, 80); this.scene.add(sun);
    this.mat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true });
    this.mesh = new THREE.Mesh(this.geo, this.mat);
    this.scene.add(this.mesh);
    this.wire = new THREE.LineSegments(new THREE.WireframeGeometry(this.geo), new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.25 }));
    this.wire.visible = false;
    this.scene.add(this.wire);
    // problem regions: points + labels
    this.labels = [];
    for (const r of v.regions || []) {
      const p = r.points.map((x, i) => (x - this.centre[i % 3]) * this.scale);
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(p), 3));
      const pts = new THREE.Points(g, new THREE.PointsMaterial({ color: r.status === "fail" ? 0xb42318 : 0xd97706, size: 6, sizeAttenuation: false, depthTest: false }));
      pts.renderOrder = 2;
      this.scene.add(pts);
      const n = p.length / 3, c = [0, 0, 0];
      for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) c[k] += p[3 * i + k] / n;
      const el = document.createElement("div"); el.className = `v3d-label ${r.status}`; el.textContent = r.name.replace("Region ", "");
      this.stage.appendChild(el);
      this.labels.push({ el, pos: new THREE.Vector3(...c) });
    }
    host.querySelector("[data-k=field]").onchange = (e) => { this.field = e.target.value; this.paint(); };
    host.querySelector("[data-k=only]").onchange = (e) => { this.only = e.target.checked; this.paint(); };
    host.querySelector("[data-k=wire]").onchange = (e) => { this.wire.visible = e.target.checked; };
    host.querySelector("[data-k=iso]").onclick = () => this.view(1, 0.8, 1.2);
    host.querySelector("[data-k=top]").onclick = () => this.view(0, 0, 1);
    host.querySelector("[data-k=front]").onclick = () => this.view(0, -1, 0.001);
    host.querySelector("[data-k=side]").onclick = () => this.view(1, 0, 0.001);
    this.ray = new THREE.Raycaster();
    this.renderer.domElement.addEventListener("mousemove", (e) => this.hover(e));
    this.renderer.domElement.addEventListener("mouseleave", () => (this.probe.style.display = "none"));
    new ResizeObserver(() => this.resize()).observe(this.stage);
    this.camera.up.set(0, 0, 1);
    const flat = mx[2] - mn[2] < 0.05 * Math.max(mx[0] - mn[0], mx[1] - mn[1]);
    flat ? this.view(0, -0.15, 1) : this.view(1, 0.8, 1.2);
    this.paint(); this.resize();
    const loop = () => { this.controls.update(); this.renderer.render(this.scene, this.camera); this.placeLabels(); requestAnimationFrame(loop); };
    loop();
  }
  view(x, y, z, target = new THREE.Vector3(0, 0, 0), dist = 190) {
    const d = new THREE.Vector3(x, y, z).normalize().multiplyScalar(dist);
    this.controls.target.copy(target); this.camera.position.copy(target.clone().add(d)); this.controls.update();
  }
  focus(r) {
    if (!r || !r.centre) return;
    const c = new THREE.Vector3(...r.centre.map((x, k) => (x - this.centre[k]) * this.scale));
    const ext = r.bbox_max ? Math.max(...r.bbox_max.map((x, k) => (x - r.bbox_min[k]) * this.scale)) : 5;
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this.view(dir.x, dir.y, dir.z, c, Math.max(25, ext * 4));
  }
  resize() {
    const w = this.stage.clientWidth, h = this.stage.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  paint() {
    const info = this.rep.metric_info[this.field], vals = this.rep.view.fields[this.field];
    const finite = vals.filter((x) => x !== null && isFinite(x));
    const range = [Math.min(...finite), Math.max(...finite)];
    const col = this.geo.attributes.color.array;
    let hidden = 0;
    for (let t = 0; t < this.nt; t++) {
      const c = colourFor(vals[t], info, range);
      const isBad = c[0] > 0.9 || c[0] > 0.7 && c[1] < 0.2;
      for (let j = 0; j < 3; j++) col.set(this.only && !isBad ? [0.93, 0.94, 0.96] : c, 9 * t + 3 * j);
      if (this.only && !isBad) hidden++;
    }
    this.mat.opacity = 1;
    this.geo.attributes.color.needsUpdate = true;
    // legend: cool scale for good values, amber/red bands for warn/fail
    const lc = this.stage.querySelector(".v3d-legend canvas"), g = lc.getContext("2d");
    const lo = range[0], hi = range[1];
    for (let p = 0; p < 220; p++) {
      const v = hi - (p / 219) * (hi - lo);
      const c = colourFor(v, info, range).map((x) => Math.round(x * 255));
      g.fillStyle = `rgb(${c.join(",")})`; g.fillRect(0, p, 18, 1);
    }
    this.stage.querySelector(".v3d-lt").textContent = info.label + (info.unit && !info.unit.startsWith("m3") ? ` (${info.unit})` : "");
    this.stage.querySelector(".v3d-ticks").innerHTML = [hi, (lo + hi) / 2, lo].map((x) => `<span>${fmt(x)}</span>`).join("");
    const lim = info.warn !== null ? `amber beyond ${fmt(info.warn)}` : "";
    const lim2 = info.fail !== null ? `, red beyond ${fmt(info.fail)}` : "";
    this.stage.querySelector(".v3d-info").innerHTML = `<b>${info.label}</b> of the cell behind each boundary face<br>${lim}${lim2}<br>drag to rotate · scroll to zoom · right-drag to pan`;
  }
  placeLabels() {
    const w = this.stage.clientWidth, h = this.stage.clientHeight;
    for (const l of this.labels) {
      const p = l.pos.clone().project(this.camera);
      const vis = p.z < 1 && Math.abs(p.x) < 1.05 && Math.abs(p.y) < 1.05;
      l.el.style.display = vis ? "block" : "none";
      if (vis) { l.el.style.left = `${(p.x + 1) / 2 * w}px`; l.el.style.top = `${(1 - p.y) / 2 * h}px`; }
    }
  }
  hover(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.camera);
    const hit = this.ray.intersectObject(this.mesh)[0];
    if (!hit) { this.probe.style.display = "none"; return; }
    const t = hit.faceIndex, info = this.rep.metric_info;
    const rows = this.fields.map((k) => `${info[k].label}: <b>${fmt(this.rep.view.fields[k][t])}</b>`).join("<br>");
    this.probe.innerHTML = rows;
    this.probe.style.display = "block";
    this.probe.style.left = Math.min(r.width - 230, e.clientX - r.left + 14) + "px"; this.probe.style.top = e.clientY - r.top + 14 + "px";
  }
}

window.MeshViewer = MeshViewer;
window.dispatchEvent(new Event("mqa-viewer-ready"));
