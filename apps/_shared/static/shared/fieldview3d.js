// 3-D field view shared by the apps: a pinneapple-field-view/v1 description (fields, colour ranges, deformations,
// markers) plus its glTF from pinneapple_tools.visualization.studio.fieldview.field_view. Field switch, colour bar
// (jet for values, the error scale for errors), deformed shape with a scale factor, marked regions (boundary
// conditions, loads, error hotspots) with labels, probe under the cursor, edges, standard views.
// three.js is the copy vendored once in the library (/vendor); colours and colour bars come from /studio/studio-core.js.
import * as THREE from "three";
import { OrbitControls } from "/vendor/three/OrbitControls.js";
import { GLTFLoader } from "/vendor/three/loaders/GLTFLoader.js";
import * as core from "/studio/studio-core.js";

const MARK = { bc: 0x1d4ed8, load: 0xc2410c, hotspot: 0x7c3aed, region: 0x0f766e };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(4) + "");
const toV3 = (p) => new THREE.Vector3(p[0], p[2], -p[1]);          // engineering z up -> glTF y up (Scene axes z_up)

export class FieldView3D {
  // opts: {meta, glb: url, field: name, deformed: bool, height: px, view: iso | top | front | side, onField(field)}
  constructor(host, opts) {
    this.host = host; this.meta = opts.meta; this.opts = opts;
    const fs = this.meta.fields;
    this.field = fs.find((f) => f.name === opts.field) || fs[0];
    this.deformOn = opts.deformed !== false && Object.keys(this.meta.deform || {}).length > 0;
    this.scaleMode = "auto"; this.scale = null;
    const groups = [...new Set(fs.map((f) => f.kind === "error" ? "Error" : "Values"))];
    host.innerHTML = `<div class="fv3"><div class="fv3-bar">
        <label>Field <select data-k="field">${groups.map((g) => `<optgroup label="${g}">${fs.filter((f) => (f.kind === "error" ? "Error" : "Values") === g).map((f) => `<option value="${esc(f.name)}" ${f === this.field ? "selected" : ""}>${esc(f.label)}${f.unit ? ` [${esc(f.unit)}]` : ""}</option>`).join("")}</optgroup>`).join("")}</select></label>
        <span class="fv3-def" ${Object.keys(this.meta.deform || {}).length ? "" : "hidden"}><label><input type="checkbox" data-k="def" ${this.deformOn ? "checked" : ""}> deformed ×</label><input data-k="scale" type="number" min="0" step="any"><button data-k="auto" title="largest displacement drawn as 10 % of the body">auto</button><button data-k="true" title="true scale">1×</button></span>
        <label ${this.meta.markers.length ? "" : "hidden"}><input type="checkbox" data-k="marks" checked> regions</label>
        <select data-k="edges"><option value="off">no edges</option><option value="feature">feature edges</option><option value="wire">wireframe</option></select>
        <span class="fv3-views"><button data-v="iso">Iso</button><button data-v="top">Top</button><button data-v="front">Front</button><button data-v="side">Side</button></span></div>
      <div class="fv3-stage" style="height:${opts.height || 460}px"><div class="fv3-info"></div><div class="fv3-legend"></div><div class="fv3-probe"></div><div class="fv3-load">loading the mesh…</div></div></div>`;
    this.stage = host.querySelector(".fv3-stage");
    this.$ = (k) => host.querySelector(`[data-k=${k}]`);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.stage.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.001, 1e7);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.25));
    this.sun = new THREE.DirectionalLight(0xffffff, 1.0); this.scene.add(this.sun);
    this.mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75, metalness: 0, side: THREE.DoubleSide });
    this.ghost = new THREE.Mesh(undefined, new THREE.MeshBasicMaterial({ color: 0x94a3b8, wireframe: true, transparent: true, opacity: 0.18 }));
    this.marks = new THREE.Group(); this.scene.add(this.marks);
    this.labels = [];
    this.$("field").onchange = (e) => this.setField(e.target.value);
    this.$("def").onchange = (e) => { this.deformOn = e.target.checked; this.deformNow(); };
    this.$("scale").onchange = (e) => { const s = parseFloat(e.target.value); if (Number.isFinite(s) && s >= 0) { this.scaleMode = "user"; this.scale = s; this.deformNow(); } };
    this.$("auto").onclick = () => { this.scaleMode = "auto"; this.deformNow(); };
    this.$("true").onclick = () => { this.scaleMode = "user"; this.scale = 1; this.deformNow(); };
    this.$("marks").onchange = (e) => { this.marks.visible = e.target.checked; this.labels.forEach((l) => (l.on = e.target.checked)); };
    this.$("edges").onchange = (e) => this.mesh && core.setEdges(THREE, this.mesh, e.target.value);
    host.querySelectorAll("[data-v]").forEach((b) => (b.onclick = () => this.view(b.dataset.v)));
    this.ray = new THREE.Raycaster();
    this.probe = this.stage.querySelector(".fv3-probe");
    this.renderer.domElement.addEventListener("mousemove", (e) => this.hover(e));
    this.renderer.domElement.addEventListener("mouseleave", () => (this.probe.style.display = "none"));
    new ResizeObserver(() => this.resize()).observe(this.stage);
    this.ready = new Promise((res) => new GLTFLoader().load(opts.glb, (g) => { this.build(g); res(this); }, undefined,
      () => { this.stage.querySelector(".fv3-load").textContent = "could not load the mesh"; res(this); }));
    const loop = () => { if (!this.host.isConnected) return; this.controls.update(); this.renderer.render(this.scene, this.camera); this.placeLabels(); requestAnimationFrame(loop); };
    loop();
  }
  build(g) {
    g.scene.traverse((o) => { if (o.isMesh && !this.mesh) this.mesh = o; });
    this.stage.querySelector(".fv3-load").remove();
    if (!this.mesh) return;
    const geo = this.mesh.geometry;
    this.base = Float32Array.from(geo.attributes.position.array);
    this.mesh.material = this.mat;
    this.scene.add(this.mesh);
    this.ghost.geometry = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(this.base, 3));
    this.ghost.geometry.setIndex(geo.index); this.ghost.visible = false; this.scene.add(this.ghost);
    this.measure();
    for (const m of this.meta.markers) {
      const pts = new THREE.BufferGeometry().setFromPoints(m.points.map(toV3));
      pts.userData.vertices = m.vertices;
      const p = new THREE.Points(pts, new THREE.PointsMaterial({ color: MARK[m.kind] || MARK.region, size: m.kind === "hotspot" ? 5 : 4, sizeAttenuation: false, depthTest: false, transparent: true, opacity: 0.9 }));
      p.renderOrder = 3; p.userData.fields = m.fields; this.marks.add(p);
      if (m.label === false) continue;                    // points only (e.g. hotspots beyond the first few)
      const el = document.createElement("div"); el.className = `fv3-label ${m.kind}`; el.textContent = m.name; el.title = m.detail || m.name;
      this.stage.appendChild(el);
      this.labels.push({ el, pos: toV3(m.centre), on: true, fields: m.fields, anchor: m.anchor, rank: m.kind === "hotspot" ? 2 : m.kind === "region" ? 1 : 0 });
    }
    this.labels.sort((a, b) => a.rank - b.rank);         // conditions and loads first, then hotspots in their order
    this.setField(this.field.name); this.view(this.opts.view || "iso"); this.resize();
  }
  measure() {                                           // bounds of the shape as drawn (deformed or not)
    const geo = this.mesh.geometry;
    geo.computeBoundingSphere(); geo.computeBoundingBox();
    this.centre = geo.boundingSphere.center.clone(); this.radius = geo.boundingSphere.radius || 1;
    const bb = geo.boundingBox;
    this.corners = [0, 1, 2, 3, 4, 5, 6, 7].map((k) => new THREE.Vector3(k & 1 ? bb.max.x : bb.min.x, k & 2 ? bb.max.y : bb.min.y, k & 4 ? bb.max.z : bb.min.z));
  }
  shows(fields) { return !fields || fields.includes(this.field.name); }
  attr(key) { return this.mesh && this.mesh.geometry.getAttribute("_" + key); }
  setField(name) {
    this.field = this.meta.fields.find((f) => f.name === name) || this.field;
    this.$("field").value = this.field.name;
    if (!this.mesh) return;
    const f = this.field, a = this.attr(f.key);
    const stops = f.kind === "error" ? core.ERR : core.JET;
    this.mesh.geometry.setAttribute("color", core.colorAttribute(THREE, a.array, f.lo, f.hi, stops));
    this.marks.children.forEach((p) => (p.visible = this.shows(p.userData.fields)));
    this.stage.querySelector(".fv3-legend").innerHTML = core.colorbarHTML({ title: `${f.label}${f.unit ? ` [${f.unit}]` : ""}`, lo: f.lo, hi: f.hi, stops, width: 240,
      loTxt: f.kind === "error" ? "0" : "", hiTxt: f.clipped ? `≥ ${num(f.hi)} (max ${num(f.max)})` : "" });
    this.deformNow();
    if (this.opts.onField) this.opts.onField(f);
  }
  deformNow() {
    if (!this.mesh) return;
    const d = this.field.deform && this.meta.deform[this.field.deform];
    const ax = d && ["x", "y", "z"].map((c) => this.attr(d.key + c));
    if (this.scaleMode === "auto" && d) this.scale = +d.auto_scale.toPrecision(2);
    const s = this.deformOn && ax && ax[0] ? this.scale : 0;
    this.$("scale").value = d ? this.scale : "";
    const pos = this.mesh.geometry.attributes.position.array;
    for (let i = 0; i < pos.length; i += 3) for (let c = 0; c < 3; c++) pos[i + c] = this.base[i + c] + (s ? s * ax[c].array[i / 3] : 0);
    this.mesh.geometry.attributes.position.needsUpdate = true;
    this.mesh.geometry.computeVertexNormals();
    const at = (i) => new THREE.Vector3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    for (const p of this.marks.children) {               // markers sit on their surface vertices, deformed or not
      const vs = p.geometry.userData.vertices, a = p.geometry.attributes.position;
      if (!vs) continue;
      vs.forEach((v, k) => a.setXYZ(k, pos[3 * v], pos[3 * v + 1], pos[3 * v + 2]));
      a.needsUpdate = true; p.geometry.computeBoundingSphere();
    }
    for (const l of this.labels) if (l.anchor !== undefined && l.anchor !== null) l.pos = at(l.anchor);
    this.ghost.visible = s > 0;
    const f = this.field, m = this.meta, u = d && d.unit ? " " + esc(d.unit) : "";
    const lines = [`<b>${esc(f.label)}</b>${f.note ? ` · ${esc(f.note)}` : ""}`];
    if (s > 0) lines.push(`deformed shape (${esc(this.field.deform)}) ×${num(s)}: largest displacement ${num(d.max)}${u} drawn as ${num(d.max * s)}${u}; grey wire: undeformed`);
    else if (d) lines.push("undeformed shape");
    if (f.max_all !== null && f.max !== null && f.max_all > 1.0001 * f.max) lines.push(`surface max ${num(f.max)}; max over the whole mesh ${num(f.max_all)} (inside)`);
    lines.push(`${m.faces.toLocaleString("en-US")} triangles${m.decimated ? ` (decimated from ${m.faces_original.toLocaleString("en-US")} for the browser)` : ""} · drag to rotate, scroll to zoom`);
    this.stage.querySelector(".fv3-info").innerHTML = lines.join("<br>");
  }
  view(v) {
    if (!this.mesh) return;
    this.measure();
    const dir = { iso: [1, 0.8, 1.15], top: [0.001, 1, 0.001], front: [0.001, 0.05, 1], side: [1, 0.05, 0.001] }[v] || [1, 0.8, 1.15];
    const n = new THREE.Vector3(...dir).normalize();
    let dist = this.radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.1;
    this.controls.target.copy(this.centre);
    for (let it = 0; it < 3; it++) {                     // fit the bounding box (not its sphere) into 78 % of the view
      this.camera.position.copy(this.centre).addScaledVector(n, dist); this.camera.near = dist / 1000; this.camera.far = dist * 100;
      this.camera.lookAt(this.centre); this.camera.updateProjectionMatrix(); this.camera.updateMatrixWorld();
      const ext = Math.max(...this.corners.map((c) => { const p = c.clone().project(this.camera); return Math.max(Math.abs(p.x), Math.abs(p.y)); }));
      dist *= Math.max(0.3, ext / 0.78);
    }
    this.camera.position.copy(this.centre).addScaledVector(n, dist); this.camera.near = dist / 1000; this.camera.far = dist * 100;
    this.camera.updateProjectionMatrix(); this.controls.update();
    this.sun.position.copy(this.centre).add(new THREE.Vector3(1, 2, 1.5).multiplyScalar(this.radius * 3));
  }
  focus(p, ext) {
    if (!this.mesh) return;
    const c = toV3(p), dir = this.camera.position.clone().sub(this.controls.target).normalize();
    const dist = Math.max(this.radius * 0.6, (ext || this.radius * 0.3) * 3);
    this.controls.target.copy(c); this.camera.position.copy(c).addScaledVector(dir, dist); this.controls.update();
  }
  resize() {
    const w = this.stage.clientWidth, h = this.stage.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  placeLabels() {
    // labels never cover each other: in priority order, each takes its anchor or the nearest free slot above or
    // below it, else it is hidden (its points stay marked)
    const w = this.stage.clientWidth, h = this.stage.clientHeight, sr = this.stage.getBoundingClientRect();
    const placed = [...this.stage.querySelectorAll(".fv3-info, .fv3-legend")].map((el) => {   // keep them readable
      const r = el.getBoundingClientRect();
      return { x: r.left - sr.left + r.width / 2, y: r.top - sr.top + r.height / 2, w: r.width, h: r.height };
    });
    for (const l of this.labels) {
      const p = l.pos.clone().project(this.camera);
      let vis = l.on && this.shows(l.fields) && p.z < 1 && Math.abs(p.x) < 1.05 && Math.abs(p.y) < 1.05;
      if (vis) {
        l.el.style.display = "block";
        const bw = l.el.offsetWidth + 4, bh = l.el.offsetHeight + 2, x = ((p.x + 1) / 2) * w, y = ((1 - p.y) / 2) * h;
        const lift = 0.8 * l.el.offsetHeight;            // the CSS transform draws a label above its anchor
        const slot = [0, -1, 1, -2, 2, -3, 3].map((k) => y + k * bh).find((yy) => !placed.some((q) =>
          Math.abs(q.x - x) < (q.w + bw) / 2 && Math.abs(q.y - (yy - lift)) < (q.h + bh) / 2));
        if (slot === undefined) vis = false;
        else { placed.push({ x, y: slot - lift, w: bw, h: bh }); l.el.style.left = `${x}px`; l.el.style.top = `${slot}px`; }
      }
      l.el.style.display = vis ? "block" : "none";
    }
  }
  hover(e) {
    if (!this.mesh) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    this.ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.camera);
    const hit = this.ray.intersectObject(this.mesh)[0];
    if (!hit) { this.probe.style.display = "none"; return; }
    const v = hit.face.a;
    this.probe.innerHTML = this.meta.fields.map((f) => `${esc(f.label)}: <b>${num(this.attr(f.key).array[v])}</b>${f.unit ? ` ${esc(f.unit)}` : ""}`).join("<br>");
    this.probe.style.display = "block";
    this.probe.style.left = Math.min(r.width - 250, e.clientX - r.left + 14) + "px"; this.probe.style.top = e.clientY - r.top + 14 + "px";
  }
}

window.FieldView3D = FieldView3D;
window.dispatchEvent(new Event("fieldview3d-ready"));
