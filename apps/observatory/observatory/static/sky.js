// The observer's sky dome: stars (Yale BSC), constellations, the Milky Way, grids, the Sun, Moon and planets
// (server-computed topocentric apparent places), satellites, atmosphere (Preetham sky), refraction and extinction.
import * as THREE from "three";
import { Sky } from "three/addons/objects/Sky.js";
import { D2R, altazVec, compass, horizonMatrix, interpAltaz, mul3, radecVec, refractionDeg, teffToRGB, bvToTeff, toMatrix4 } from "./astro.js";
import { lang, t } from "./i18n.js";

const R_STARS = 1000, R_BODIES = 900;
const BODY_COLORS = { sun: "#fff3c4", moon: "#e8e8f0", mercury: "#cfc6b8", venus: "#fff6d8", mars: "#ff9a6a",
  jupiter: "#ffe2b8", saturn: "#f5dc9c", uranus: "#a8e6ee", neptune: "#7ea6ff" };
const NAMES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"];
// IAU WGCCRE north poles (J2000) for ring / disc orientation
const POLES = { saturn: [40.589, 83.537], jupiter: [268.057, 64.495] };

export function glowTexture(inner = "rgba(255,255,255,1)", mid = "rgba(255,255,255,0.25)", size = 256) {
  const c = document.createElement("canvas"); c.width = c.height = size;
  const g = c.getContext("2d"), r = size / 2;
  const grd = g.createRadialGradient(r, r, 0, r, r, r);
  grd.addColorStop(0, inner); grd.addColorStop(0.12, inner); grd.addColorStop(0.3, mid); grd.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grd; g.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

const STAR_VS = `
attribute float mag; attribute vec3 color; attribute float seed;
uniform float limitMag, pixelRatio, night, time, atmosphere, groundOn, sizeScale;
varying vec3 vColor; varying float vAlpha;
float refr(float h) { return h < -1.0 ? 0.0 : 1.02 / tan(radians(h + 10.3 / (h + 5.11))) / 60.0; }
void main() {
  vec3 d = normalize((modelMatrix * vec4(position, 1.0)).xyz);
  float h = degrees(asin(clamp(d.y, -1.0, 1.0)));
  float X = 1.0;
  if (atmosphere > 0.5) {
    float h2 = h + refr(h);
    vec2 hz = normalize(d.xz + 1e-9);
    d = vec3(hz.x * cos(radians(h2)), sin(radians(h2)), hz.y * cos(radians(h2)));
    h = h2;
    X = 1.0 / (sin(radians(max(h, -0.5))) + 0.50572 * pow(max(h, -0.5) + 6.07995, -1.6364));
  }
  float m = mag + (atmosphere > 0.5 ? 0.25 * (X - 1.0) : 0.0);
  float rel = limitMag - m;
  float tw = atmosphere > 0.5 ? 1.0 + 0.22 * sin(time * (7.0 + seed * 9.0) + seed * 40.0) * clamp((X - 1.0) / 6.0, 0.0, 1.0) : 1.0;
  vAlpha = clamp(0.3 + rel * 0.35, 0.0, 1.0) * night * tw * step(0.0, rel);
  if (groundOn > 0.5 && h < -0.3) vAlpha = 0.0;
  vColor = color;
  gl_PointSize = clamp((2.4 + max(rel, 0.0) * 1.9) * sizeScale, 0.0, 34.0) * pixelRatio;
  gl_Position = projectionMatrix * viewMatrix * vec4(d * ${R_STARS.toFixed(1)}, 1.0);
}`;
const STAR_FS = `
varying vec3 vColor; varying float vAlpha;
void main() {
  float r = length(gl_PointCoord - 0.5) * 2.0;
  if (r > 1.0) discard;
  float core = exp(-r * r * 6.0), halo = exp(-r * 3.5) * 0.35;
  gl_FragColor = vec4(vColor * (core + halo) * vAlpha * 1.7, 1.0);
}`;
const MW_VS = `varying vec3 vP; varying vec3 vW;
void main() { vP = position; vW = (modelMatrix * vec4(position, 1.0)).xyz; gl_Position = projectionMatrix * viewMatrix * vec4(vW, 1.0); }`;
const MW_FS = `uniform sampler2D tex; uniform float intensity; varying vec3 vP; varying vec3 vW;
void main() {
  vec3 p = normalize(vP);
  float u = atan(p.y, p.x) / 6.2831853; u = fract(u);
  float v = 0.5 + asin(clamp(p.z, -1.0, 1.0)) / 3.14159265;
  float g = texture2D(tex, vec2(u, v)).r;
  float h = normalize(vW).y;
  float fadeH = smoothstep(-0.02, 0.25, h);
  gl_FragColor = vec4(vec3(0.72, 0.78, 1.0) * pow(g, 1.8) * intensity * fadeH, 1.0);
}`;
const GROUND_FS = `uniform vec3 top; uniform vec3 bottom; varying vec3 vW;
void main() { float h = clamp(-normalize(vW).y, 0.0, 1.0); gl_FragColor = vec4(mix(top, bottom, pow(h, 0.35)), 1.0); }`;

export class SkyView {
  constructor(renderer, labels) {
    this.renderer = renderer;
    this.labels = labels;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(70, 1, 0.1, 6000);
    this.camera.rotation.order = "YXZ";
    this.yaw = 180; this.pitch = 25; this.fov = 70;     // look south by default (north in the southern hemisphere: set by app)
    this.layers = { constellations: true, names: true, altaz: false, equatorial: false, milkyway: true,
      atmosphere: true, ground: true, satellites: true, planetsBig: false };
    this.lat = 0; this.lst = 0; this.time = 0;
    this.eqJ2000 = new THREE.Group(); this.eqJ2000.matrixAutoUpdate = false;
    this.eqDate = new THREE.Group(); this.eqDate.matrixAutoUpdate = false;
    this.local = new THREE.Group();
    this.scene.add(this.eqJ2000, this.eqDate, this.local);
    this._sky(); this._ground(); this._grids(); this._bodies();
    this.satGroup = new THREE.Group(); this.local.add(this.satGroup);
    this.sats = [];
  }

  // ------------------------------------------------------------------------------------------- construction
  _sky() {
    const sky = new Sky(); sky.scale.setScalar(4500); sky.renderOrder = -10;
    const u = sky.material.uniforms;
    u.turbidity.value = 4; u.rayleigh.value = 1.6; u.mieCoefficient.value = 0.004; u.mieDirectionalG.value = 0.85;
    this.skyMesh = sky; this.scene.add(sky);
    // twilight / night sky: a simple single-scattering look-alike driven by the Sun's altitude and azimuth
    // (Preetham's model is only used with the Sun above the horizon)
    const ng = new THREE.Mesh(new THREE.SphereGeometry(4000, 64, 32), new THREE.ShaderMaterial({ side: THREE.BackSide, depthWrite: false,
      uniforms: { sunDir: { value: new THREE.Vector3(0, -1, 0) }, k: { value: 1 } },
      vertexShader: `varying vec3 vW; void main(){ vW = (modelMatrix*vec4(position,1.0)).xyz; gl_Position = projectionMatrix*viewMatrix*vec4(vW,1.0);} `,
      fragmentShader: `uniform vec3 sunDir; uniform float k; varying vec3 vW;
      void main(){
        vec3 d = normalize(vW); float h = max(d.y, 0.0);
        float sa = degrees(asin(clamp(sunDir.y, -1.0, 1.0)));            // Sun altitude
        float tw = clamp((sa + 18.0) / 18.0, 0.0, 1.0);                   // 0 at astronomical night, 1 at sunset
        vec3 zen = mix(vec3(0.004, 0.007, 0.018), vec3(0.10, 0.20, 0.42), pow(tw, 2.2));
        vec3 hor = mix(vec3(0.03, 0.045, 0.08), vec3(0.42, 0.40, 0.48), pow(tw, 2.0));
        vec3 col = mix(hor, zen, pow(h, 0.5));
        vec2 sh = normalize(sunDir.xz + 1e-6), dh = normalize(d.xz + 1e-6);
        float az = dot(sh, dh) * 0.5 + 0.5;                                 // 1 towards the Sun's azimuth
        float glow = pow(az, 3.0) * exp(-h * 7.0) * pow(tw, 2.5);
        col += glow * mix(vec3(0.9, 0.35, 0.08), vec3(1.0, 0.62, 0.30), smoothstep(-6.0, 0.0, sa)) * 1.3;
        col += exp(-h * 18.0) * pow(tw, 3.0) * vec3(0.25, 0.12, 0.10) * (1.0 - az);   // Belt of Venus / Earth shadow edge
        gl_FragColor = vec4(col * k, 1.0);
      }` }));
    ng.renderOrder = -11; this.nightSky = ng; this.scene.add(ng);
    const mw = new THREE.Mesh(new THREE.SphereGeometry(R_STARS * 1.02, 96, 48),
      new THREE.ShaderMaterial({ vertexShader: MW_VS, fragmentShader: MW_FS, side: THREE.BackSide, transparent: true,
        depthWrite: false, blending: THREE.AdditiveBlending,
        uniforms: { tex: { value: null }, intensity: { value: 0.3 } } }));
    mw.renderOrder = -5; this.mw = mw; this.eqJ2000.add(mw);
    new THREE.TextureLoader().load("/api/milkyway.jpg", (t) => { mw.material.uniforms.tex.value = t; });
  }

  _ground() {
    const g = new THREE.Mesh(new THREE.SphereGeometry(950, 64, 32, 0, Math.PI * 2, Math.PI / 2 + 0.0005, Math.PI / 2),
      new THREE.ShaderMaterial({ vertexShader: MW_VS, fragmentShader: GROUND_FS, side: THREE.BackSide,
        uniforms: { top: { value: new THREE.Color("#141a22") }, bottom: { value: new THREE.Color("#020304") } } }));
    g.renderOrder = 5; this.groundMesh = g; this.local.add(g);
    // horizon glow ring
    const ring = new THREE.Mesh(new THREE.CylinderGeometry(940, 940, 30, 128, 1, true),
      new THREE.ShaderMaterial({ transparent: true, depthWrite: false, side: THREE.BackSide, blending: THREE.AdditiveBlending,
        uniforms: { col: { value: new THREE.Color("#3a5a8a") }, k: { value: 0.25 } },
        vertexShader: `varying float vy; void main(){ vy = position.y; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0);} `,
        fragmentShader: `uniform vec3 col; uniform float k; varying float vy; void main(){ float a = exp(-abs(vy)/5.0)*k; gl_FragColor = vec4(col*a,1.0);} ` }));
    ring.renderOrder = 6; this.horizonGlow = ring; this.local.add(ring);
  }

  _grids() {
    const lines = (pts, color, opacity) => {
      const geo = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
    };
    const R = R_STARS * 0.99, a = [];
    for (let alt = 0; alt <= 80; alt += 10) for (let az = 0; az < 360; az += 2) {
      a.push(...altazVec(alt, az, R), ...altazVec(alt, az + 2, R));
    }
    for (let az = 0; az < 360; az += 15) for (let alt = 0; alt < 88; alt += 2) a.push(...altazVec(alt, az, R), ...altazVec(alt + 2, az, R));
    this.altazGrid = lines(a, "#4f7a5a", 0.45); this.local.add(this.altazGrid);
    const e = [];
    const v = (ra, dec) => radecVec(ra, dec).map((x) => x * R);
    for (let dec = -80; dec <= 80; dec += 10) for (let ra = 0; ra < 360; ra += 2) e.push(...v(ra, dec), ...v(ra + 2, dec));
    for (let ra = 0; ra < 360; ra += 15) for (let dec = -88; dec < 88; dec += 2) e.push(...v(ra, dec), ...v(ra, dec + 2));
    this.eqGrid = lines(e, "#5c79b8", 0.35); this.eqDate.add(this.eqGrid);
    const ecl = [], eps = 23.4392911 * D2R;
    for (let l = 0; l < 360; l += 1) {
      const p = (lam) => { const L = lam * D2R; return [Math.cos(L) * R, Math.sin(L) * Math.cos(eps) * R, Math.sin(L) * Math.sin(eps) * R]; };
      ecl.push(...p(l), ...p(l + 1));
    }
    this.ecliptic = lines(ecl, "#b9934a", 0.45); this.eqDate.add(this.ecliptic);
  }

  _bodies() {
    this.sunLight = new THREE.DirectionalLight(0xffffff, 2.2);
    this.scene.add(this.sunLight, new THREE.AmbientLight(0x334466, 0.35));   // earthshine on the Moon's night side
    this.bodies = {};
    const glow = glowTexture();
    const loader = new THREE.TextureLoader();
    for (const b of NAMES) {
      const grp = new THREE.Group();
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, color: BODY_COLORS[b], transparent: true,
        blending: THREE.AdditiveBlending, depthWrite: false }));
      sprite.renderOrder = b === "sun" ? 2 : 6;          // glow drawn over the (possibly dark) disc
      grp.add(sprite);
      let mesh = null;
      if (b !== "sun") {
        const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 });
        loader.load(`/static/textures/${b}.jpg`, (t) => { t.colorSpace = THREE.SRGBColorSpace; mat.map = t; mat.needsUpdate = true; });
        mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), mat);
        mesh.renderOrder = 3; grp.add(mesh);
      } else {
        const tex = loader.load("/static/textures/sun.jpg"); tex.colorSpace = THREE.SRGBColorSpace;
        mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshBasicMaterial({ map: tex, color: 0xffffff }));
        mesh.renderOrder = 3; grp.add(mesh);
      }
      if (b === "saturn") {
        const rg = new THREE.RingGeometry(1.24, 2.27, 96, 1);
        const pos = rg.attributes.position, uv = rg.attributes.uv;
        for (let i = 0; i < pos.count; i++) { const r = Math.hypot(pos.getX(i), pos.getY(i)); uv.setXY(i, (r - 1.24) / 1.03, 0.5); }
        const rt = loader.load("/static/textures/saturn_rings.png");
        const ring = new THREE.Mesh(rg, new THREE.MeshStandardMaterial({ map: rt, transparent: true, side: THREE.DoubleSide, roughness: 1 }));
        ring.renderOrder = 4; mesh.add(ring); this.saturnRing = ring;
      }
      this.local.add(grp);
      this.bodies[b] = { grp, sprite, mesh, alt: -90, az: 0, data: null };
    }
  }

  // ------------------------------------------------------------------------------------------------- data
  setCatalog(cat) {
    this.cat = cat;
    const n = cat.ra.length;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), mag = new Float32Array(n), seed = new Float32Array(n);
    this.starVec = pos;
    for (let i = 0; i < n; i++) {
      const v = radecVec(cat.ra[i], cat.dec[i]);
      pos.set(v.map((x) => x * R_STARS), i * 3);
      const c = teffToRGB(bvToTeff(cat.bv[i] ?? 0.6));
      // desaturate a little: the eye sees star colours weakly
      const l = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
      col.set([0.55 * c[0] + 0.45 * l, 0.55 * c[1] + 0.45 * l, 0.55 * c[2] + 0.45 * l], i * 3);
      mag[i] = cat.v[i]; seed[i] = Math.random();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geo.setAttribute("mag", new THREE.BufferAttribute(mag, 1));
    geo.setAttribute("seed", new THREE.BufferAttribute(seed, 1));
    this.starUniforms = { limitMag: { value: 6.2 }, pixelRatio: { value: this.renderer.getPixelRatio() }, night: { value: 1 },
      time: { value: 0 }, atmosphere: { value: 1 }, groundOn: { value: 1 }, sizeScale: { value: 1 } };
    const pts = new THREE.Points(geo, new THREE.ShaderMaterial({ vertexShader: STAR_VS, fragmentShader: STAR_FS,
      uniforms: this.starUniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    pts.frustumCulled = false; pts.renderOrder = 1;
    this.stars = pts; this.eqJ2000.add(pts);
    // constellation figures
    const seg = [];
    for (const c of cat.constellations) for (const line of c.lines)
      for (let k = 0; k + 1 < line.length; k++) seg.push(...radecVec(...line[k]).map((x) => x * R_STARS * 0.995),
        ...radecVec(...line[k + 1]).map((x) => x * R_STARS * 0.995));
    const cg = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(seg, 3));
    this.conLines = new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ color: "#5d7fbf", transparent: true, opacity: 0.42, depthWrite: false }));
    this.eqJ2000.add(this.conLines);
    this.conLabels = cat.constellations.map((c) => ({ names: c.names || { en: c.name }, v: radecVec(...c.label) }));
    this.namedStars = [];
    for (let i = 0; i < n; i++) if (cat.name[i] && (cat.v[i] < 2.1 || (cat.proper[i] && cat.v[i] < 3.2))) this.namedStars.push(i);
    this.namedStars.sort((a, b) => cat.v[a] - cat.v[b]);
  }

  setSky(d) {
    this.skyData = d;
    this.lat = d.observer.lat; this.lon = d.observer.lon;
    this.npb = d.npb;
  }

  setSats(d) { this.satData = d; this._buildSats(); }

  _buildSats() {
    this.satGroup.clear(); this.sats = [];
    if (!this.satData) return;
    const glow = glowTexture("rgba(200,255,230,1)", "rgba(120,255,200,0.2)", 64);
    for (const s of this.satData.satellites) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, color: "#bfffe0", transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
      sp.renderOrder = 7; this.satGroup.add(sp);
      const tg = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * 40), 3));
      const trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: "#6fe3b0", transparent: true, opacity: 0.35, depthWrite: false }));
      trail.frustumCulled = false; this.satGroup.add(trail);
      this.sats.push({ s, sp, trail, alt: -90, az: 0 });
    }
  }

  // ------------------------------------------------------------------------------------------------ frame
  update(simJD, wall) {
    if (!this.skyData) return;
    const d = this.skyData;
    this.time = wall;
    this.lst = (((d.gast_deg + d.sidereal_rate_deg_per_day * (simJD - d.jd_utc) + d.observer.lon) % 360) + 360) % 360;
    const H = horizonMatrix(this.lst, this.lat);
    this.Mdate = H; this.Mj2000 = mul3(H, this.npb);
    this.eqJ2000.matrix.copy(toMatrix4(THREE, this.Mj2000)); this.eqJ2000.matrixWorldNeedsUpdate = true;
    this.eqDate.matrix.copy(toMatrix4(THREE, H)); this.eqDate.matrixWorldNeedsUpdate = true;

    // bodies
    const S = d.samples, atm = this.layers.atmosphere;
    for (const [b, o] of Object.entries(this.bodies)) {
      const s = S.bodies[b];
      // samples are refracted apparent altitudes; with the atmosphere off use the geometric ones
      let [alt, az] = interpAltaz(S.jd, s.alt, s.az, simJD);
      if (!atm) alt -= refractionDeg(alt - refractionDeg(alt));
      o.alt = alt; o.az = az;
      const dist = s.dist_km[Math.min(s.dist_km.length - 1, Math.max(0, Math.round((simJD - S.jd[0]) / (S.jd[1] - S.jd[0]))))];
      const radiusKm = { sun: 695700, moon: 1737.4, mercury: 2439.4, venus: 6051.8, mars: 3389.5, jupiter: 69911, saturn: 58232, uranus: 25362, neptune: 24622 }[b];
      const ang = Math.asin(Math.min(1, radiusKm / dist));            // true angular radius
      const big = this.layers.planetsBig && b !== "sun" && b !== "moon" ? 60 : 1;
      const R = R_BODIES * Math.tan(ang) * big;
      const p = altazVec(alt, az, R_BODIES);
      o.grp.position.set(...p);
      o.mesh.scale.setScalar(Math.max(R, 0.0001));
      const now = d.now[b];
      const m = now?.magnitude ?? 0;
      const fovK = this.fov / 70;
      let gs;
      if (b === "sun") gs = 150 * fovK;
      else if (b === "moon") gs = R * 3.2 + 4 * fovK;
      else gs = Math.max(2.5, (3.0 - m * 0.5)) * 5.0 * fovK + R * 2.2;
      o.sprite.scale.setScalar(gs);
      o.sprite.material.opacity = b === "sun" ? 1 : Math.min(1, (b === "moon" ? 0.25 + 0.6 * (now?.illuminated ?? 0.5) : 0.95) * (alt > -1 || !this.layers.ground ? 1 : 0));
      o.grp.visible = alt > -2 || !this.layers.ground;
      if (b === "sun") this.sunDir = new THREE.Vector3(...altazVec(alt, az, 1));
    }
    this.sunLight.position.copy(this.sunDir).multiplyScalar(100);
    this._orientDiscs();

    // sky brightness from the Sun's altitude
    const sunAlt = this.bodies.sun.alt;
    const night = 1 - smooth(-16, -3, sunAlt);
    const lim = sunAlt < -18 ? 6.3 : sunAlt < -12 ? 4.5 + (6.3 - 4.5) * (-12 - sunAlt) / 6 : sunAlt < -6 ? 2.5 + 2 * (-6 - sunAlt) / 6 : sunAlt < 0 ? -1 + 3.5 * (-sunAlt) / 6 : -3;
    const zoomGain = 2.0 * Math.log10(70 / this.fov);
    if (this.starUniforms) {
      const u = this.starUniforms;
      u.limitMag.value = (atm ? lim : 6.8) + Math.max(0, zoomGain);
      u.night.value = atm ? Math.max(night, 0.04) : 1;
      u.time.value = wall; u.atmosphere.value = atm ? 1 : 0; u.groundOn.value = this.layers.ground ? 1 : 0;
      u.sizeScale.value = Math.min(1.6, Math.max(0.9, Math.sqrt(70 / this.fov) * 0.9));
      this.stars.visible = true;
    }
    this.skyMesh.visible = atm && sunAlt > -0.5;
    this.nightSky.visible = !atm || sunAlt <= -0.5;
    this.nightSky.material.uniforms.k.value = atm ? 1 : 0.35;
    this.nightSky.material.uniforms.sunDir.value.copy(atm ? this.sunDir : new THREE.Vector3(0, -1, 0));
    const su = this.skyMesh.material.uniforms;
    su.sunPosition.value.copy(this.sunDir);
    su.rayleigh.value = sunAlt > -3 ? 1.6 : 1.6 + (-3 - sunAlt) * 0.3;
    this.renderer.toneMappingExposure = atm ? 0.32 + 0.25 * night : 0.6;
    this.mw.visible = this.layers.milkyway;
    this.mw.material.uniforms.intensity.value = (atm ? Math.pow(night, 2) * 0.42 : 0.4) * (this.layers.milkyway ? 1 : 0);
    this.groundMesh.visible = this.layers.ground;
    const dayK = smooth(-12, 4, sunAlt);
    this.groundMesh.material.uniforms.top.value.setRGB(0.05 + 0.25 * dayK, 0.06 + 0.27 * dayK, 0.08 + 0.22 * dayK);
    this.horizonGlow.visible = this.layers.ground;
    this.horizonGlow.material.uniforms.k.value = atm ? 0.12 + 0.3 * (1 - night) : 0.1;
    this.conLines.visible = this.layers.constellations;
    this.conLines.material.opacity = 0.45 * (atm ? Math.max(night, 0.15) : 1);
    this.altazGrid.visible = this.layers.altaz;
    this.eqGrid.visible = this.layers.equatorial; this.ecliptic.visible = this.layers.equatorial;

    // satellites
    this.satGroup.visible = this.layers.satellites;
    if (this.satData && this.layers.satellites) {
      const jd = this.satData.jd;
      for (const o of this.sats) {
        const [alt, az] = interpAltaz(jd, o.s.alt, o.s.az, simJD);
        o.alt = alt; o.az = az;
        const k = Math.min(o.s.sunlit.length - 1, Math.max(0, Math.round((simJD - jd[0]) / (jd[1] - jd[0]))));
        const lit = o.s.sunlit[k] > 0.5, mag = o.s.mag[k];
        o.lit = lit; o.mag = mag;
        const vis = alt > 0 && simJD >= jd[0] && simJD <= jd[jd.length - 1];
        o.sp.visible = vis;
        o.sp.position.set(...altazVec(alt, az, R_BODIES * 0.98));
        o.sp.scale.setScalar((lit ? Math.max(3, 9 - (mag ?? 4) * 1.2) : 3) * this.fov / 70);
        o.sp.material.color.set(lit ? "#d8fff0" : "#553333");
        const arr = o.trail.geometry.attributes.position.array;
        for (let i = 0; i < 40; i++) {
          const [a2, z2] = interpAltaz(jd, o.s.alt, o.s.az, simJD - i * 6 / 86400);
          arr.set(altazVec(a2, z2, R_BODIES * 0.98), i * 3);
        }
        o.trail.geometry.attributes.position.needsUpdate = true;
        o.trail.visible = vis && lit;
      }
    }
    // camera
    this.camera.fov = this.fov;
    if (!this.gyro) { this.camera.rotation.set(this.pitch * D2R, -this.yaw * D2R, 0); }
  }

  _orientDiscs() {
    // Moon / planet discs: texture centre faces the observer, north up along the celestial north direction
    const ncp = new THREE.Vector3();
    const H = this.Mdate;
    ncp.set(H[0][2], H[1][2], H[2][2]);                                // the true pole of date in horizon axes
    for (const b of ["moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]) {
      const o = this.bodies[b];
      const toObs = o.grp.position.clone().normalize().negate();
      let pole = ncp.clone();
      if (POLES[b]) { const v = radecVec(...POLES[b]); const M = this.Mj2000; pole.set(M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2], M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2], M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]); }
      const up = pole.clone().sub(toObs.clone().multiplyScalar(pole.dot(toObs)));
      if (b === "saturn" && this.saturnRing) {
        // ring plane is perpendicular to the true pole (not its projection)
        const m = new THREE.Matrix4().makeBasis(toObs.clone().cross(pole).normalize(), pole.clone().normalize(),
          toObs.clone().cross(pole).normalize().cross(pole).normalize());
        o.mesh.quaternion.setFromRotationMatrix(m);
        this.saturnRing.rotation.set(-Math.PI / 2, 0, 0);
        continue;
      }
      up.normalize();
      const x = toObs, z = x.clone().cross(up).normalize();
      const m = new THREE.Matrix4().makeBasis(x, up, z);
      o.mesh.quaternion.setFromRotationMatrix(m);
    }
  }

  // -------------------------------------------------------------------------------------------- labels
  drawLabels(w, h) {
    const L = this.labels; L.begin();
    const cam = this.camera, v = new THREE.Vector3();
    const taken = [];
    const put = (vec, text, cls, extra, avoid) => {
      v.set(...vec).applyMatrix4(cam.matrixWorldInverse);
      if (v.z > -1) return;
      v.applyMatrix4(cam.projectionMatrix);
      const x = (v.x + 1) / 2 * w, y = (1 - v.y) / 2 * h;
      if (x < -50 || x > w + 50 || y < -20 || y > h + 20) return;
      if (avoid) { if (taken.some(([a, b]) => Math.abs(a - x) < 60 && Math.abs(b - y) < 13)) return; taken.push([x, y]); }
      L.add(x, y, text, cls, extra);
    };
    const ground = this.layers.ground;
    for (const az of [0, 90, 180, 270, 45, 135, 225, 315]) put(altazVec(1.5, az, 500), compass(az), "cardinal");
    if (this.layers.names && this.cat) {
      const M = this.Mj2000;
      const limit = this.fov < 25 ? 4.0 : this.fov < 45 ? 3.2 : 2.1;
      for (const i of this.namedStars) {
        if (this.cat.v[i] > limit && !(this.cat.v[i] < 2.6)) continue;
        if (this.cat.v[i] > this.starUniforms.limitMag.value - 0.8 || this.cat.name[i].startsWith("HR ")) continue;
        const s = [this.starVec[i * 3], this.starVec[i * 3 + 1], this.starVec[i * 3 + 2]];
        const p = [0, 1, 2].map((r) => M[r][0] * s[0] + M[r][1] * s[1] + M[r][2] * s[2]);
        if (ground && p[1] < 0) continue;
        if (this.layers.atmosphere && this.bodies.sun.alt > -4 && this.cat.v[i] > 0.5) continue;
        put(p, this.cat.name[i], "star", null, true);
      }
    }
    if (this.layers.constellations && this.conLabels) {
      const M = this.Mj2000;
      for (const c of this.conLabels) {
        const p = [0, 1, 2].map((r) => (M[r][0] * c.v[0] + M[r][1] * c.v[1] + M[r][2] * c.v[2]) * 1000);
        if (ground && p[1] < 0) continue;
        put(p, c.names[lang] || c.names.en, "con");
      }
    }
    for (const [b, o] of Object.entries(this.bodies)) {
      if (ground && o.alt < -1) continue;
      put(o.grp.position.toArray(), t("body." + b), "body", b);
    }
    if (this.layers.satellites) for (const o of this.sats) if (o.sp.visible && (o.lit || o.s.name.includes("ISS")))
      put(o.sp.position.toArray(), o.s.name.replace(" (ZARYA)", "").slice(0, 18), "sat");
    L.end();
  }

  // --------------------------------------------------------------------------------------------- picking
  pick(ndcX, ndcY) {
    const ray = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(this.camera).normalize();
    const pixDeg = this.fov / (this.renderer.domElement.clientHeight || 800);
    let best = null;
    const consider = (vec, obj, tolDeg) => {
      const u = new THREE.Vector3(...vec).normalize();
      const ang = Math.acos(Math.min(1, u.dot(ray))) / D2R;
      if (ang < tolDeg && (!best || ang - tolDeg < best.score)) best = { ...obj, score: ang - tolDeg };
    };
    for (const [b, o] of Object.entries(this.bodies)) if (o.grp.visible) consider(o.grp.position.toArray(), { kind: "body", id: b }, Math.max(14 * pixDeg, 0.3));
    for (const o of this.sats) if (o.sp.visible) consider(o.sp.position.toArray(), { kind: "sat", id: o.s.norad, sat: o }, 14 * pixDeg);
    if (this.cat) {
      const M = this.Mj2000, lim = this.starUniforms.limitMag.value;
      for (let i = 0; i < this.cat.v.length; i++) {
        if (this.cat.v[i] > lim) continue;
        const s = [this.starVec[i * 3], this.starVec[i * 3 + 1], this.starVec[i * 3 + 2]];
        const p = [0, 1, 2].map((r) => M[r][0] * s[0] + M[r][1] * s[1] + M[r][2] * s[2]);
        consider(p, { kind: "star", id: this.cat.hr[i], index: i }, 10 * pixDeg * (1 + Math.max(0, 3 - this.cat.v[i]) * 0.3));
      }
    }
    return best;
  }

  lookAt(alt, az) { this.yaw = az; this.pitch = Math.max(-10, Math.min(89, alt)); }
  render() { this.renderer.render(this.scene, this.camera); }
}

function smooth(a, b, x) { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }
