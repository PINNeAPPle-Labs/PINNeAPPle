// Journey mode (#723): fly from the observer's place on Earth to a planet, the Moon, the Sun or a star, in 3-D.
// The geometry (positions, distances, light time, travel times, the target's magnitude and size versus distance)
// comes from pinneapple_physics.astro.journey through /api/journey/*; this file draws. The camera stays at the
// origin and every object is placed in its true direction at a compressed distance with its true angular size, so
// what you see is right from 100 km to 1000 light-years.
import * as THREE from "three";
import { D2R, bvToTeff, interp, mul3, radecVec, rotZ, teffToRGB, toMatrix4 } from "./astro.js";
import { glowTexture } from "./sky.js";
import { num, t } from "./i18n.js";

const LY = 9.4607304725808e12, AU = 149597870.7, PC = 3.0856775814913673e13;
const T = [[1, 0, 0], [0, 0, 1], [0, -1, 0]];                     // ICRS (x, y, z) -> three (x, z, -y)
const tr = (M) => M[0].map((_, j) => M.map((r) => r[j]));
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const BODY_COL = { sun: "#fff3c4", mercury: "#cfc6b8", venus: "#fff1c8", earth: "#8fc0ff", moon: "#e0e0e6", mars: "#ff9a6a",
  jupiter: "#ffe2b8", saturn: "#f5dc9c", uranus: "#a8e6ee", neptune: "#7ea6ff" };
const SCALE_KEYS = ["moon", "au", "heliopause", "light_year", "nearest_star"];

// compressed scene distance: monotonic, ~log; objects keep their true angular size (radius x C(d) / d)
const C = (d) => 2 + 40 * Math.log10(1 + d / 1000);
const three = (v) => new THREE.Vector3(v[0], v[2], -v[1]);

export class JourneyView {
  constructor(renderer, labels) {
    this.renderer = renderer; this.labels = labels;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 5000);
    this.sunLight = new THREE.PointLight(0xffffff, 4, 0, 0);
    this.scene.add(this.sunLight, new THREE.AmbientLight(0x404858, 0.25));
    this.loader = new THREE.TextureLoader();
    this.bodies = {}; this.stars3d = null; this.plan = null; this.state = "idle";
    this.rate = 2.2; this.yaw = 0; this.pitch = 0; this.markersPassed = new Set();
    this.glow = glowTexture();
  }

  setScene(sc, cat) {
    this.sc = sc;
    for (const o of Object.values(this.bodies)) this.scene.remove(o.grp);
    this.bodies = {};
    for (const [b, d] of Object.entries(sc.bodies)) {
      const grp = new THREE.Group();
      let mat;
      if (b === "sun") mat = new THREE.MeshBasicMaterial({ map: this.tex("sun.jpg") });
      else mat = new THREE.MeshStandardMaterial({ map: this.tex(b === "earth" ? "earth_day.jpg" : `${b}.jpg`), roughness: 1, metalness: 0 });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 32), mat);
      if (b === "earth") mesh.matrixAutoUpdate = false;
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, color: BODY_COL[b], transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
      grp.add(mesh, glow); this.scene.add(grp);
      if (b === "saturn") {                          // rings 1.24-2.27 radii, perpendicular to the pole (RA 40.589, Dec 83.537)
        const rg = new THREE.RingGeometry(1.24, 2.27, 128, 1), uv = rg.attributes.uv, ps = rg.attributes.position;
        for (let i = 0; i < ps.count; i++) uv.setXY(i, (Math.hypot(ps.getX(i), ps.getY(i)) - 1.24) / 1.03, 0.5);
        const ring = new THREE.Mesh(rg, new THREE.MeshStandardMaterial({ map: this.tex("saturn_rings.png"), transparent: true, side: THREE.DoubleSide, roughness: 1 }));
        const p = radecVec(40.589, 83.537);
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(p[0], p[2], -p[1]));
        ring.rotation.x = -Math.PI / 2;            // the ring lies in the globe's equator
        mesh.add(ring);
      }
      this.bodies[b] = { grp, mesh, glow, pos: d.pos_km, r: d.radius_km };
    }
    // stars: 3-D ones (with a parallax) move with the camera; the rest of the catalogue stays as a background
    const n = sc.stars.length;
    this.starPos = sc.stars.map((s) => mul(radecVec(s[1], s[2]), s[3] * LY));
    this.starAbs = sc.stars.map((s) => s[4]);
    const col = new Float32Array(n * 3), pos = new Float32Array(n * 3), size = new Float32Array(n);
    sc.stars.forEach((s, i) => col.set(teffToRGB(bvToTeff(s[5] ?? 0.6)), i * 3));
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setAttribute("size", new THREE.BufferAttribute(size, 1));
    const m = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { pr: { value: this.renderer.getPixelRatio() } },
      vertexShader: `attribute float size; attribute vec3 color; uniform float pr; varying vec3 c; varying float a; void main(){ c = color; a = clamp(size / 3.0, 0.15, 1.0); gl_PointSize = max(size, 1.2) * pr; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);} `,
      fragmentShader: `varying vec3 c; varying float a; void main(){ float r = length(gl_PointCoord-0.5)*2.0; if(r>1.0) discard; gl_FragColor = vec4(c*exp(-r*r*5.0)*a,1.0);} ` });
    if (this.stars3d) this.scene.remove(this.stars3d);
    this.stars3d = new THREE.Points(g, m); this.stars3d.frustumCulled = false; this.scene.add(this.stars3d);
    if (cat && !this.bg) {
      const in3d = new Set(sc.stars.map((s) => s[0])), bp = [], bc = [];
      for (let i = 0; i < cat.ra.length; i++) {
        if (in3d.has(cat.hr[i]) || cat.v[i] > 6.5) continue;
        const v = radecVec(cat.ra[i], cat.dec[i]); bp.push(v[0] * 4000, v[2] * 4000, -v[1] * 4000);
        const c = teffToRGB(bvToTeff(cat.bv[i] ?? 0.6)), k = Math.max(0.15, Math.min(0.8, (7 - cat.v[i]) / 5));
        bc.push(c[0] * k, c[1] * k, c[2] * k);
      }
      const bg = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(bp, 3)).setAttribute("color", new THREE.Float32BufferAttribute(bc, 3));
      this.bg = new THREE.Points(bg, new THREE.PointsMaterial({ size: 1.4, sizeAttenuation: false, vertexColors: true, transparent: true, depthWrite: false }));
      this.scene.add(this.bg);
    }
  }
  tex(f) { const x = this.loader.load(`/static/textures/${f}`); x.colorSpace = THREE.SRGBColorSpace; return x; }

  // a new trip: up from the observer's site (2000 km along the local vertical, out of the atmosphere), then straight on
  start(plan) {
    this.plan = plan; this.state = "flying"; this.markersPassed = new Set(); this.yaw = 0; this.pitch = 0; this.lookBack = false;
    const s = plan.start_km, e = plan.end_km, a = plan.arrival_km, earth = this.bodies.earth.pos;
    const up = mul(sub(s, earth), 1 / len(sub(s, earth)));
    this.p0 = s; this.p1 = add(s, mul(up, 2000)); this.p2 = a;
    this.L1 = 2000; this.L2 = len(sub(a, this.p1)); this.total = this.L1 + this.L2; this.trav = 0;
    if (plan.target.kind === "star") {
      if (this.targetStar) this.scene.remove(this.targetStar.grp);
      const col = new THREE.Color(...teffToRGB(plan.target.teff_k || 5800));
      const grp = new THREE.Group(), mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), new THREE.MeshBasicMaterial({ color: col }));
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, color: col, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
      grp.add(mesh, glow); this.scene.add(grp);
      this.targetStar = { grp, mesh, glow, pos: e, r: plan.target.radius_km };
    } else if (this.targetStar) { this.scene.remove(this.targetStar.grp); this.targetStar = null; }
  }
  skip() { if (this.plan) this.trav = this.total; }

  camPos() {
    if (this.trav <= this.L1) return add(this.p0, mul(sub(this.p1, this.p0), this.trav / this.L1));
    return add(this.p1, mul(sub(this.p2, this.p1), Math.min(1, (this.trav - this.L1) / this.L2)));
  }
  counters() {
    const p = this.plan; if (!p) return null;
    const cam = this.camPos(), left = len(sub(p.end_km, cam));
    const prof = p.profile, lx = prof.distance_left_km.map(Math.log10), lv = Math.log10(Math.max(left, prof.distance_left_km[0]));
    return { travelled: Math.min(this.trav, this.total), left, light_left_s: left / 299792.458, progress: Math.min(1, this.trav / this.total),
      magnitude: interp(lx, prof.magnitude, lv), ang_deg: interp(lx, prof.angular_diameter_deg, lv), arrived: this.state === "arrived" };
  }

  update(dt, simJD) {
    if (!this.plan || !this.sc) return;
    if (this.state === "flying") {
      const left = this.total - this.trav;
      this.trav += this.rate * Math.max(10, Math.min(this.trav + 10, left)) * dt;
      for (const k of SCALE_KEYS) {
        const d = this.plan.markers_km[k];
        if (d !== undefined && !this.markersPassed.has(k) && this.trav >= d) { this.markersPassed.add(k); this.onMarker?.(k); }
      }
      if (this.trav >= this.total - 1e-6) { this.trav = this.total; this.state = "arrived"; this.onArrive?.(); }
    }
    const cam = this.camPos(), p = this.plan;
    const place = (o, pos, r, minPx) => {
      const rel = sub(pos, cam), d = len(rel), k = C(d) / d;
      o.grp.position.copy(three(mul(rel, k)));
      o.mesh.scale.setScalar(Math.max(r * k, 1e-6));
      o.glow.scale.setScalar(Math.max(r * k * 3, C(d) * minPx));
      return d;
    };
    for (const [b, o] of Object.entries(this.bodies)) {
      const d = place(o, o.pos, o.r, b === "sun" ? 0.02 : 0.012);
      if (b === "sun") { this.sunLight.position.copy(o.grp.position); o.glow.material.opacity = Math.min(1, Math.max(0.15, 1.6 - Math.log10(d / AU) * 0.35)); }
      else o.glow.material.opacity = 0.7;
    }
    // the Earth turns: G maps the terrestrial frame to three axes (true equator of date ~ ICRS here)
    const e = this.bodies.earth, gast = p.gast_deg + 360.98564736629 * (simJD - (Date.parse(p.t) / 864e5 + 2440587.5));
    const G = mul3(mul3(T, mul3(tr(p.npb), rotZ(-gast * D2R))), tr(T));
    const k = C(len(sub(e.pos, cam))) / len(sub(e.pos, cam));
    e.mesh.matrix.copy(toMatrix4(THREE, G)).scale(new THREE.Vector3(e.r * k, e.r * k, e.r * k)); e.mesh.matrixWorldNeedsUpdate = true;
    if (this.targetStar) {
      const d = place(this.targetStar, this.targetStar.pos, this.targetStar.r, 0.01);
      const m = p.target.abs_mag_v + 5 * Math.log10(d / (10 * PC));
      this.targetStar.glow.scale.setScalar(Math.max(this.targetStar.r * C(d) / d * 6, C(d) * 0.004 * Math.pow(10, -0.2 * (m - 2))));
    }
    // stars with a parallax: direction and magnitude from where we are
    const pos = this.stars3d.geometry.attributes.position.array, size = this.stars3d.geometry.attributes.size.array;
    for (let i = 0; i < this.starPos.length; i++) {
      const rel = sub(this.starPos[i], cam), d = len(rel), kk = Math.min(C(d), 3800) / d;
      pos[i * 3] = rel[0] * kk; pos[i * 3 + 1] = rel[2] * kk; pos[i * 3 + 2] = -rel[1] * kk;
      const m = this.starAbs[i] + 5 * Math.log10(d / (10 * PC));
      size[i] = Math.max(0, Math.min(14, 4.5 - 0.65 * m));
    }
    this.stars3d.geometry.attributes.position.needsUpdate = true; this.stars3d.geometry.attributes.size.needsUpdate = true;
    // look at the target (along the vertical during the lift-off), or orbit it after arrival
    const tgt = sub(p.end_km, cam), up = sub(cam, e.pos);
    // during the climb, look at the horizon towards the target (a little down, so the Earth fills the lower half),
    // then turn to the target over the first 60,000 km
    let dir = three(tgt).normalize();
    const vert = three(up).normalize(), F = 60000;
    if (this.trav < F) {
      const hz = dir.clone().sub(vert.clone().multiplyScalar(dir.dot(vert)));
      const look = (hz.lengthSq() > 1e-8 ? hz.normalize() : new THREE.Vector3(1, 0, 0)).multiplyScalar(Math.cos(0.25)).add(vert.clone().multiplyScalar(-Math.sin(0.25)));
      const f = Math.pow(Math.min(1, this.trav / F), 0.6);
      dir = look.lerp(dir, f).normalize();
    }
    // after the climb the camera's "up" is the celestial north pole (the local vertical points along the path)
    const north = three([0, 0, 1]), camUp = this.trav < F ? vert : (Math.abs(dir.dot(north)) > 0.98 ? three([1, 0, 0]) : north);
    if (this.lookBack) dir.negate();               // home is behind you
    if (this.state === "arrived") {                // look around: yaw about "up", pitch about the right axis
      const v = camUp.clone(), right = dir.clone().cross(v).normalize();
      dir.applyAxisAngle(right, this.pitch * D2R).applyAxisAngle(v, this.yaw * D2R).normalize();
    }
    this.camera.position.set(0, 0, 0);
    this.camera.up.copy(camUp);
    this.camera.lookAt(dir);
  }

  render() { this.renderer.render(this.scene, this.camera); }

  drawLabels(w, h, name) {
    const L = this.labels; L.begin();
    const v = new THREE.Vector3();
    const put = (p3, text, cls) => { v.copy(p3).project(this.camera); if (v.z > 1 || Math.abs(v.x) > 1.1 || Math.abs(v.y) > 1.1) return; L.add((v.x + 1) / 2 * w, (1 - v.y) / 2 * h, text, cls); };
    for (const [b, o] of Object.entries(this.bodies)) {
      if (b === "sun" && this.plan?.target.kind === "star" && this.state !== "idle") {
        const d = len(sub(o.pos, this.camPos()));
        if (d > 0.01 * LY) { put(o.grp.position, t("journey.sunAsStar", { mag: num(4.81 + 5 * Math.log10(d / (10 * PC)), 1) }), "body"); continue; }
      }
      put(o.grp.position, name(b), "body");
    }
    if (this.targetStar) put(this.targetStar.grp.position, name(this.plan.target.id), "body");
    L.end();
  }
}
