// The solar system in 3-D: heliocentric J2000-ecliptic positions from the JPL ephemeris (server), planets to scale
// in distance (linear or logarithmic), the Earth with its day/night terminator, rotation and the observer's pin,
// the Moon's orbit, and the real star background.
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { D2R, interp, mul3, radecVec, rotZ, teffToRGB, bvToTeff, toMatrix4 } from "./astro.js";
import { glowTexture } from "./sky.js";
import { t } from "./i18n.js";

const RADII = { mercury: 2439.4, venus: 6051.8, earth: 6371.0, mars: 3389.5, jupiter: 69911, saturn: 58232, uranus: 25362, neptune: 24622 };
const SPRITE_COLORS = { mercury: "#cfc6b8", venus: "#fff1c8", earth: "#8fc0ff", mars: "#ff9a6a", jupiter: "#ffe2b8", saturn: "#f5dc9c", uranus: "#a8e6ee", neptune: "#7ea6ff" };
const T = [[1, 0, 0], [0, 0, 1], [0, -1, 0]];                     // ecliptic (x, y, z) -> three (x, z, -y)
const tr = (M) => M[0].map((_, j) => M.map((r) => r[j]));

const EARTH_VS = `varying vec2 vUv; varying vec3 vN; varying vec3 vP;
void main(){ vUv = uv; vN = normalize(mat3(modelMatrix) * normal); vP = (modelMatrix * vec4(position,1.0)).xyz; gl_Position = projectionMatrix * viewMatrix * vec4(vP,1.0); }`;
const EARTH_FS = `uniform sampler2D day; uniform sampler2D night; uniform vec3 sunDir; varying vec2 vUv; varying vec3 vN; varying vec3 vP;
void main(){
  vec3 n = normalize(vN); float ndl = dot(n, normalize(sunDir));
  vec3 d = texture2D(day, vUv).rgb; vec3 ni = texture2D(night, vUv).rgb;
  float k = smoothstep(-0.12, 0.12, ndl);
  // day side: the surface plus the blue of the sunlit atmosphere (without it a sunlit Pacific reads as black)
  vec3 dayc = d * (0.08 + 1.3 * max(ndl, 0.0)) + vec3(0.05, 0.11, 0.26) * max(ndl, 0.0);
  vec3 col = mix(ni * vec3(1.0, 0.85, 0.6) * 1.6 + d * 0.02, dayc, k);
  vec3 v = normalize(cameraPosition - vP);
  float fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);
  col += fres * vec3(0.3, 0.55, 1.0) * (0.15 + 0.9 * smoothstep(-0.3, 0.4, ndl));
  // terminator: thin reddish twilight band
  col += vec3(0.25, 0.08, 0.02) * exp(-pow(ndl / 0.05, 2.0)) * 0.6;
  gl_FragColor = vec4(col, 1.0);
  #include <colorspace_fragment>
}`;

export class SystemView {
  constructor(renderer, labels) {
    this.renderer = renderer; this.labels = labels;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.005, 20000);
    this.camera.position.set(0, 60, 90);
    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.08; this.controls.enabled = false;
    this.logScale = true; this.focus = null; this.tween = null;
    const loader = new THREE.TextureLoader();
    const tex = (f) => { const t = loader.load(`/static/textures/${f}`); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t; };
    // Sun
    this.sun = new THREE.Mesh(new THREE.SphereGeometry(2.2, 64, 32), new THREE.MeshBasicMaterial({ map: tex("sun.jpg") }));
    const sg = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture("rgba(255,240,200,1)", "rgba(255,170,60,0.35)"), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    sg.scale.setScalar(16); this.sun.add(sg);
    this.scene.add(this.sun);
    // ambient floor: a planet seen at a large phase angle still shows its surface (the sunlit side stays far brighter)
    this.scene.add(new THREE.PointLight(0xffffff, 4.5, 0, 0), new THREE.AmbientLight(0x8a96aa, 0.9));
    // planets
    this.planets = {};
    const halo = glowTexture("rgba(255,255,255,1)", "rgba(255,255,255,0.6)");   // its core sits behind the sphere
    for (const b of Object.keys(RADII)) {
      const r = 0.32 * Math.sqrt(RADII[b] / 6371);
      let mat;
      if (b === "earth") {
        mat = new THREE.ShaderMaterial({ vertexShader: EARTH_VS, fragmentShader: EARTH_FS,
          uniforms: { day: { value: tex("earth_day.jpg") }, night: { value: tex("earth_night.jpg") }, sunDir: { value: new THREE.Vector3(1, 0, 0) } } });
      } else mat = new THREE.MeshStandardMaterial({ map: tex(`${b}.jpg`), roughness: 1, metalness: 0 });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 64, 32), mat);
      mesh.userData.body = b;
      const grp = new THREE.Group(); grp.add(mesh); this.scene.add(grp);
      // glow sprite: keeps the planet visible as a coloured point when it is a few pixels across
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: halo, color: SPRITE_COLORS[b], transparent: true,
        blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0.9 }));
      glow.userData.base = r * 6; glow.scale.setScalar(r * 6); grp.add(glow);
      if (b === "earth") mesh.matrixAutoUpdate = false;
      if (b === "saturn") {
        const rg = new THREE.RingGeometry(r * 1.24, r * 2.27, 128, 1);
        const pos = rg.attributes.position, uv = rg.attributes.uv;
        for (let i = 0; i < pos.count; i++) { const rr = Math.hypot(pos.getX(i), pos.getY(i)) / r; uv.setXY(i, (rr - 1.24) / 1.03, 0.5); }
        const ring = new THREE.Mesh(rg, new THREE.MeshStandardMaterial({ map: loader.load("/static/textures/saturn_rings.png"), transparent: true, side: THREE.DoubleSide, roughness: 1 }));
        // Saturn's pole (RA 40.589, Dec 83.537) in three axes; the ring plane is perpendicular to it
        this.saturnPole = radecVec(40.589, 83.537);
        grp.add(ring); this.ring = ring;
      }
      this.planets[b] = { grp, mesh, r, glow: grp.children[1], orbit: null };
    }
    // Moon
    this.moon = new THREE.Mesh(new THREE.SphereGeometry(0.32 * Math.sqrt(1737.4 / 6371), 48, 24), new THREE.MeshStandardMaterial({ map: tex("moon.jpg"), roughness: 1 }));
    this.moon.userData.body = "moon";
    this.scene.add(this.moon);
    // observer pin
    this.pin = new THREE.Mesh(new THREE.SphereGeometry(0.006, 16, 8), new THREE.MeshBasicMaterial({ color: 0xff4d4d }));
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.0012, 0.0012, 0.05, 6), new THREE.MeshBasicMaterial({ color: 0xff8080 }));
    beam.position.y = 0.025; this.pinBeam = beam; this.pinGrp = new THREE.Group(); this.pinGrp.add(this.pin, beam);
    this.scene.add(this.pinGrp);
    this.orbits = new THREE.Group(); this.scene.add(this.orbits);
  }

  setSizeScale(k) {
    this.sizeK = k;
    const f = k / 3;
    for (const o of Object.values(this.planets)) { o.mesh.scale.setScalar(f); o.glow.scale.setScalar(o.glow.userData.base * f); }
    if (this.ring) this.ring.scale.setScalar(f);
    this.moon.scale.setScalar(f);
  }

  setSmallBody(tr) {
    if (this.sbLine) { this.scene.remove(this.sbLine); this.scene.remove(this.sbDot); }
    this.sbTrack = tr;
    const pts = tr.pos_au.map((v) => this.map(v));
    this.sbLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: "#ff5c8a", transparent: true, opacity: 0.8 }));
    this.sbDot = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture("rgba(255,120,160,1)", "rgba(255,80,140,0.3)", 64), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.sbDot.scale.setScalar(1.2);
    this.scene.add(this.sbLine, this.sbDot);
  }

  scaleR(rAU) { return this.logScale ? 30 * Math.log(1 + rAU / 0.4) / Math.log(3.5) : 30 * rAU; }
  map(v) {                                       // ecliptic AU vector -> three position with the distance scale
    const r = Math.hypot(...v); if (r === 0) return new THREE.Vector3();
    const k = this.scaleR(r) / r;
    return new THREE.Vector3(v[0] * k, v[2] * k, -v[1] * k);
  }

  setCatalog(cat) {
    const E0 = [[1, 0, 0], [0, Math.cos(23.4392911 * D2R), Math.sin(23.4392911 * D2R)], [0, -Math.sin(23.4392911 * D2R), Math.cos(23.4392911 * D2R)]];
    const M = mul3(T, E0);
    const n = cat.ra.length, pos = [], col = [], size = [];
    for (let i = 0; i < n; i++) {
      if (cat.v[i] > 6.0) continue;
      const v = radecVec(cat.ra[i], cat.dec[i]);
      pos.push(...[0, 1, 2].map((r) => (M[r][0] * v[0] + M[r][1] * v[1] + M[r][2] * v[2]) * 8000));
      col.push(...teffToRGB(bvToTeff(cat.bv[i] ?? 0.6)));
      size.push(Math.max(1.0, 4.2 - cat.v[i] * 0.55));
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute("size", new THREE.Float32BufferAttribute(size, 1));
    const m = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { pr: { value: this.renderer.getPixelRatio() } },
      vertexShader: `attribute float size; attribute vec3 color; uniform float pr; varying vec3 c; void main(){ c = color; gl_PointSize = size * pr; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);} `,
      fragmentShader: `varying vec3 c; void main(){ float r = length(gl_PointCoord-0.5)*2.0; if(r>1.0) discard; gl_FragColor = vec4(c*exp(-r*r*6.0)*0.9,1.0);} ` });
    const pts = new THREE.Points(g, m); pts.frustumCulled = false;
    this.starPts = pts; this.scene.add(pts);
  }

  setData(d) {
    this.data = d;
    this.tdbOffset = d.jd_tdb - d.jd_utc;
    this.E = d.ecliptic_j2000; this.npb = d.npb;
    this._buildOrbits();
  }

  _buildOrbits() {
    this.orbits.clear();
    const d = this.data; if (!d) return;
    const colors = { mercury: "#a89f94", venus: "#e8cf9a", earth: "#6fa8ff", mars: "#ff8a5c", jupiter: "#e3bf8f", saturn: "#e9d39a", uranus: "#9fe3ea", neptune: "#7c9cff" };
    for (const b of Object.keys(RADII)) {
      const pts = d.orbits[b].map((v) => this.map(v));
      pts.push(pts[0]);
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: colors[b], transparent: true, opacity: 0.38, depthWrite: false }));
      this.orbits.add(line);
    }
    if (this.sbTrack) this.setSmallBody(this.sbTrack);
    this.moonOrbit = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: "#c8c8d8", transparent: true, opacity: 0.35 }));
    this.orbits.add(this.moonOrbit);
  }

  // units per km of the geocentric Moon (distance exaggerated, and pushed out with the planet size so the Moon
  // stays outside the enlarged Earth)
  moonScale() { return 1.15 / 384400 * Math.max(1, (this.sizeK || 3) / 3); }

  update(simJD) {
    const d = this.data; if (!d) return;
    const jd = simJD + this.tdbOffset;
    for (const [b, o] of Object.entries(this.planets)) {
      const arr = d.planets[b];
      const v = [0, 1, 2].map((k) => interp(d.jd, arr.map((p) => p[k]), jd));
      o.grp.position.copy(this.map(v));
      o.helio = v;
    }
    // Earth orientation: Earth-fixed -> three = T . E . NPB^T . R3(-GAST) . T^-1
    const gast = d.gast_deg + 360.98564736629 * (simJD - d.jd_utc);
    const G = mul3(mul3(mul3(T, this.E), mul3(tr(this.npb), rotZ(-gast * D2R))), tr(T));
    const em = this.planets.earth.mesh;
    const f = (this.sizeK || 3) / 3;               // the matrix is set by hand, so the size slider's scale goes in here
    em.matrix.copy(toMatrix4(THREE, G)).scale(new THREE.Vector3(f, f, f)); em.matrixWorldNeedsUpdate = true;
    this.G = G;
    const ep = this.planets.earth.grp.position;
    em.material.uniforms.sunDir.value.copy(ep).negate().normalize();
    // Saturn ring orientation
    if (this.ring) {
      const pole = this.saturnPole, M = mul3(T, this.E);
      const p = new THREE.Vector3(...[0, 1, 2].map((r) => M[r][0] * pole[0] + M[r][1] * pole[1] + M[r][2] * pole[2]));
      this.ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), p);
      this.planets.saturn.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), p);
    }
    // Moon
    const mp = d.moon.pos_km;
    const mv = [0, 1, 2].map((k) => interp(d.moon.jd, mp.map((p) => p[k]), jd));
    const s = this.moonScale();
    this.moon.position.set(ep.x + mv[0] * s, ep.y + mv[2] * s, ep.z - mv[1] * s);
    const i0 = Math.max(0, d.moon.jd.findIndex((x) => x >= jd) - 112);
    const pts = d.moon.pos_km.slice(i0, i0 + 224).map((p) => new THREE.Vector3(ep.x + p[0] * s, ep.y + p[2] * s, ep.z - p[1] * s));
    this.moonOrbit.geometry.setFromPoints(pts);
    // observer pin
    if (this.observer) {
      const la = this.observer.lat * D2R, lo = this.observer.lon * D2R;
      const itrs = [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)];
      const g = [itrs[0], itrs[2], -itrs[1]];                    // T . itrs
      const w = [0, 1, 2].map((r) => G[r][0] * g[0] + G[r][1] * g[1] + G[r][2] * g[2]);
      const up = new THREE.Vector3(...w).normalize();
      this.pinUp = up;
      this.pinGrp.position.copy(ep).add(up.clone().multiplyScalar(this.planets.earth.r * (this.sizeK || 3) / 3 * 1.005));
      this.pinGrp.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), up);
    }
    if (this.sbTrack) {
      const t = this.sbTrack, v = [0, 1, 2].map((k) => interp(t.jd, t.pos_au.map((p) => p[k]), jd));
      this.sbDot.position.copy(this.map(v));
    }
    if (this.focus && !this.tween) {
      const target = this.focusPos();
      const delta = target.clone().sub(this.controls.target);
      this.camera.position.add(delta); this.controls.target.copy(target);
    }
    this._tween();
    this.controls.update();
  }

  focusPos() {
    if (this.focus === "moon") return this.moon.position.clone();
    if (this.focus === "sun") return new THREE.Vector3();
    if (this.focus === "you") return this.pinGrp.position.clone();
    return this.planets[this.focus]?.grp.position.clone() ?? new THREE.Vector3();
  }

  flyTo(pos, target, ms = 2200, then) {
    this.tween = { p0: this.camera.position.clone(), t0: this.controls.target.clone(), p1: pos, t1: target, start: performance.now(), ms, then };
  }
  _tween() {
    const tw = this.tween; if (!tw) return;
    let k = Math.min(1, (performance.now() - tw.start) / tw.ms);
    const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
    // fly along a log-distance path so the zoom feels uniform
    const d0 = tw.p0.distanceTo(tw.t0), d1 = tw.p1.distanceTo(tw.t1);
    const tgt = tw.t0.clone().lerp(tw.t1, e);
    const dir = tw.p0.clone().sub(tw.t0).normalize().lerp(tw.p1.clone().sub(tw.t1).normalize(), e).normalize();
    const dist = Math.exp(Math.log(d0) + (Math.log(d1) - Math.log(d0)) * e);
    this.camera.position.copy(tgt).add(dir.multiplyScalar(dist));
    this.controls.target.copy(tgt);
    if (k >= 1) { const f = tw.then; this.tween = null; if (f) f(); }
  }

  // camera hovering just above the observer, looking up their local vertical (start of the zoom-out)
  observerShot() {
    const up = this.pinUp ?? new THREE.Vector3(0, 1, 0);
    const p = this.pinGrp.position.clone();
    // just above the observer's spot on the globe, looking down at it
    const f = (this.sizeK || 3) / 3;               // keep the framing when the Earth is enlarged
    const side = new THREE.Vector3(0, 1, 0).cross(up).normalize().multiplyScalar(0.12 * f);
    return { pos: p.clone().add(up.clone().multiplyScalar(0.35 * f)).add(side), target: this.planets.earth.grp.position.clone() };
  }
  // the whole planetary system, framed so that every planet lies inside the part of the canvas left of a panel
  // covering the right ``covered`` fraction of its width and above a HUD covering the bottom ``below`` fraction of
  // its height; seen from 50 deg above the ecliptic, from the side opposite the Earth (its day side faces us)
  overviewShot(covered = 0, below = 0) {
    const ep = this.planets.earth.grp.position;
    const az = new THREE.Vector3(-ep.x, 0, -ep.z).normalize();
    const el = 50 * D2R;
    const back = new THREE.Vector3(az.x * Math.cos(el), Math.sin(el), az.z * Math.cos(el));
    const f = (this.sizeK || 3) / 3;
    const pts = Object.values(this.planets).map((o) => [o.grp.position, o.r * f * 3]);   // centre, radius with halo
    const cam = this.camera.clone();
    const tanV = Math.tan(cam.fov * D2R / 2), tanH = tanV * cam.aspect;
    const m = 0.06, x0 = -1 + m, x1 = 1 - 2 * covered - m, y0 = -1 + 2 * below + m, y1 = 1 - m;     // target box in NDC
    let target = new THREE.Vector3(), d = 150;
    const v = new THREE.Vector3();
    for (let it = 0; it < 40; it++) {
      cam.position.copy(target).addScaledVector(back, d); cam.lookAt(target); cam.updateMatrixWorld();
      const inv = cam.matrixWorldInverse;
      let a = Infinity, b = -Infinity, c = Infinity, e = -Infinity;
      for (const [p, r] of pts) {
        v.copy(p).applyMatrix4(inv);
        const z = -v.z, rx = r / (z * tanH), ry = r / (z * tanV);
        a = Math.min(a, v.x / (z * tanH) - rx); b = Math.max(b, v.x / (z * tanH) + rx);
        c = Math.min(c, v.y / (z * tanV) - ry); e = Math.max(e, v.y / (z * tanV) + ry);
      }
      const s = Math.max((b - a) / (x1 - x0), (e - c) / (y1 - y0));
      // move the target so the planets' box centres on the free area, then scale the distance to fit it
      const right = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1);
      target.addScaledVector(right, ((a + b) / 2 - (x0 + x1) / 2 * s) * d * tanH)
        .addScaledVector(up, ((c + e) / 2 - (y0 + y1) / 2 * s) * d * tanV);
      d *= s;
      if (Math.abs(s - 1) < 1e-4) break;
    }
    return { pos: target.clone().addScaledVector(back, d), target };
  }

  drawLabels(w, h) {
    const L = this.labels; L.begin();
    const v = new THREE.Vector3();
    // hide labels of objects behind the Earth's disc
    const cam = this.camera.position, ep = this.planets.earth.grp.position;
    const toE = ep.clone().sub(cam), dE = toE.length(), rE = Math.asin(Math.min(1, this.planets.earth.r / dE));
    const put = (p, text, cls, id) => {
      if (id !== "earth" && id !== "you") {
        const toP = p.clone().sub(cam);
        if (toP.length() > dE && toP.angleTo(toE) < rE) return;
      }
      v.copy(p).project(this.camera);
      if (v.z > 1) return;
      L.add((v.x + 1) / 2 * w, (1 - v.y) / 2 * h, text, cls, id);
    };
    put(new THREE.Vector3(), t("body.sun"), "body", "sun");
    for (const [b, o] of Object.entries(this.planets)) put(o.grp.position, t("body." + b), "body", b);
    if (this.sbTrack) put(this.sbDot.position, this.sbTrack.name.replace(/\s*\(.*\)/, ""), "sat", "sb");
    if (this.camera.position.distanceTo(this.planets.earth.grp.position) < 12) {
      put(this.moon.position, t("body.moon"), "body", "moon");
      if (this.observer) put(this.pinGrp.position, t("system.you"), "you", "you");
    }
    L.end();
  }

  pick(ndcX, ndcY) {
    const rc = new THREE.Raycaster(); rc.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.camera);
    const objs = [this.sun, this.moon, ...Object.values(this.planets).map((o) => o.mesh)];
    const hit = rc.intersectObjects(objs, false)[0];
    if (hit) return { kind: "body", id: hit.object === this.sun ? "sun" : hit.object.userData.body };
    // fall back to the nearest label-sized target on screen
    let best = null;
    for (const [b, o] of Object.entries({ sun: { grp: this.sun }, moon: { grp: this.moon }, ...this.planets })) {
      const p = o.grp.position.clone().project(this.camera);
      const dd = Math.hypot(p.x - ndcX, p.y - ndcY);
      if (dd < 0.04 && (!best || dd < best.d)) best = { kind: "body", id: b, d: dd };
    }
    return best;
  }

  render() { this.renderer.render(this.scene, this.camera); }
}
