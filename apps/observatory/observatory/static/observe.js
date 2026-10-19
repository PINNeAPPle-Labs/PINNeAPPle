// Observe mode (#722): an observing session. Plan the night for a target, set up a telescope, slew to it with a GoTo
// mount, then track it in an eyepiece / camera view while time runs at 1x to 3600x. Every number comes from
// pinneapple_physics.astro.observing through /api/observe/*; this file draws.
import { lang, num, t } from "./i18n.js";
import { D2R, R2D, interp, radecVec, refractionDeg } from "./astro.js";

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const BODIES = ["moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"];
const BODY_COLORS = { moon: "#dcdcd2", mercury: "#cfc6b8", venus: "#fff1c8", mars: "#ff9a6a", jupiter: "#e9d2ad",
  saturn: "#e8d39a", uranus: "#a8e6ee", neptune: "#7ea6ff" };

export class Observe {
  constructor(ctx) {
    this.ctx = ctx;                // {api, iso, qs, place, sky, S, bname, conName, dfmt, tfmt, toast}
    this.target = null; this.plan = null; this.presets = null; this.optics = null;
    this.setup = { telescope: "newton200", eyepiece: "plossl25", camera: "", aperture: "", focal: "", mount: "altaz" };
    this.weather = { mode: "perfect", view: null }; this.seeing = 2.0;
    this.log = []; this.tracking = false; this.slewing = null;
    this.state = { offset_px: 0, max_offset_px: 0, rotation_deg: 0, hidden: false, frames: 0 };
    this.textures = {}; this.survey = null;
    const q = new URLSearchParams(location.search);          // ?scope=sct8&eye=plossl32&mount=equatorial&sky=real
    for (const [k, key] of [["scope", "telescope"], ["eye", "eyepiece"], ["cam", "camera"], ["mount", "mount"]]) if (q.get(k)) this.setup[key] = q.get(k);
    if (q.get("sky") === "real" || q.get("wx")) this.weather.mode = q.get("sky") === "perfect" ? "perfect" : (q.get("sky") || "perfect");
  }

  // ------------------------------------------------------------------------------------------- the panel
  async render(el) {
    this.el = el;
    if (!this.presets) this.presets = await this.ctx.api("/api/observe/presets");
    const P = this.presets;
    const opt = (o, sel) => Object.entries(o).map(([k, v]) => `<option value="${k}" ${k === sel ? "selected" : ""}>${esc(v.name)}</option>`).join("");
    el.innerHTML = `
      <h3>${t("observe.step1")}</h3>
      <div class="row"><select id="obTarget" style="flex:1"></select><button class="primary" id="obPlan">${t("observe.plan")}</button></div>
      <div id="obPlanOut"></div>
      <h3>${t("observe.step2")}</h3>
      <div class="row"><select id="obTel" style="flex:1">${opt(P.telescopes, this.setup.telescope)}<option value="custom">${t("observe.custom")}</option></select></div>
      <div class="row" id="obCustom" hidden><label class="small">${t("observe.aperture")} <input id="obAp" type="number" min="20" max="2000" value="150" style="width:70px"> mm</label>
        <label class="small">${t("observe.focal")} <input id="obFl" type="number" min="100" max="40000" value="750" style="width:80px"> mm</label></div>
      <div class="row"><select id="obEye" style="flex:1"><option value="">${t("observe.camera")}</option>${opt(P.eyepieces, this.setup.eyepiece)}</select>
        <select id="obCam"><option value="">—</option>${opt(P.cameras, this.setup.camera)}</select></div>
      <div id="obOptics"></div>
      <h3>${t("observe.step3")}</h3>
      <div class="row"><select id="obMount"><option value="altaz">${t("observe.mountAltaz")}</option><option value="equatorial">${t("observe.mountEq")}</option></select>
        <button class="primary" id="obSlew">${t("observe.slew")}</button></div><div id="obSlewOut" class="small"></div>
      <h3>${t("observe.step4")}</h3>
      <div class="row"><button class="primary" id="obTrack">${t("observe.track")}</button>
        <label class="small"><input type="radio" name="obWx" value="perfect" ${this.weather.mode === "perfect" ? "checked" : ""}> ${t("observe.perfectSky")}</label>
        <label class="small"><input type="radio" name="obWx" value="real" ${this.weather.mode === "real" ? "checked" : ""}> ${t("observe.realSky")}</label></div>
      <div id="obWx" class="small"></div>
      <div class="row small"><label>${t("observe.seeing")} <input id="obSeeing" type="range" min="0.5" max="6" step="0.1" value="${this.seeing}"></label><span id="obSeeingV" class="pill">${num(this.seeing, 1)}″</span></div>
      <h3>${t("observe.logTitle")}</h3><div class="row"><input id="obNote" type="text" placeholder="${esc(t("observe.notePh"))}" style="flex:1"><button class="ghost" id="obAddNote">${t("observe.addNote")}</button></div>
      <div id="obLog"></div>`;
    await this.fillTargets();
    const $ = (s) => el.querySelector(s);
    $("#obPlan").onclick = () => this.planTarget($("#obTarget").value);
    $("#obTarget").onchange = () => this.planTarget($("#obTarget").value);
    const optChange = () => {
      this.setup.telescope = $("#obTel").value; this.setup.eyepiece = $("#obEye").value; this.setup.camera = $("#obCam").value;
      $("#obCustom").hidden = this.setup.telescope !== "custom";
      this.loadOptics();
    };
    for (const s of ["#obTel", "#obEye", "#obCam", "#obAp", "#obFl"]) $(s).onchange = optChange;
    $("#obMount").value = this.setup.mount;
    $("#obMount").onchange = (e) => { this.setup.mount = e.target.value; };
    $("#obSlew").onclick = () => this.slew();
    $("#obTrack").onclick = () => this.startTracking();
    for (const r of el.querySelectorAll("input[name=obWx]")) r.onchange = (e) => { this.weather.mode = e.target.value; this.loadWeather(); };
    $("#obSeeing").oninput = (e) => { this.seeing = +e.target.value; $("#obSeeingV").textContent = num(this.seeing, 1) + "″"; };
    $("#obAddNote").onclick = () => { this.addLog($("#obNote").value || t("observe.snapshot"), true); $("#obNote").value = ""; };
    this.renderLog();
    this.loadOptics();
    this.loadWeather();
    if (this.target) { $("#obTarget").value = this.target; this.planTarget(this.target); }
  }

  async fillTargets() {
    const sel = this.el.querySelector("#obTarget");
    const [stars, messier] = await Promise.all([this.ctx.api(`/api/facts/list?kind=star&lang=${lang}`), this.ctx.api(`/api/facts/list?kind=messier&lang=${lang}`)]);
    const group = (label, rows) => `<optgroup label="${esc(label)}">${rows.map(([v, n]) => `<option value="${esc(v)}">${esc(n)}</option>`).join("")}</optgroup>`;
    this.names = Object.fromEntries([...stars.cards, ...messier.cards].map((c) => [c.id, c.name]));
    sel.innerHTML = group(t("observe.gPlanets"), BODIES.map((b) => [b, this.ctx.bname(b)])) +
      group(t("learn.group.star"), stars.cards.map((c) => [c.id, c.name])) +
      group(t("learn.group.messier"), messier.cards.map((c) => [c.id, c.name]));
    if (this.target) sel.value = this.target;
  }

  async loadOptics() {
    const s = this.setup, el = this.el?.querySelector("#obOptics");
    let q = s.telescope === "custom" ? `aperture_mm=${this.el.querySelector("#obAp").value}&focal_mm=${this.el.querySelector("#obFl").value}` : `telescope=${s.telescope}`;
    q += `&eyepiece=${s.eyepiece || ""}&camera=${s.camera || ""}`;
    try { this.optics = await this.ctx.api(`/api/observe/optics?${q}`); } catch (e) { if (el) el.textContent = e.message; return; }
    const o = this.optics;
    if (!el) return;
    el.innerHTML = `<div class="kv">${o.magnification ? `<span>${t("observe.mag")}</span><span>${num(o.magnification, 0)}×</span><span>${t("observe.tfov")}</span><span>${num(o.true_fov_deg * 60, 1)}′</span><span>${t("observe.exitPupil")}</span><span>${num(o.exit_pupil_mm, 1)} mm</span>` : ""}
      ${o.camera_fov_deg ? `<span>${t("observe.camFov")}</span><span>${num(o.camera_fov_deg[0] * 60, 1)}′ × ${num(o.camera_fov_deg[1] * 60, 1)}′</span><span>${t("observe.scale")}</span><span>${num(o.image_scale_arcsec_per_px, 2)}″/px</span>` : ""}
      <span>${t("observe.focalRatio")}</span><span>f/${num(o.focal_ratio, 1)}</span>
      <span>${t("observe.resolution")}</span><span>${num(o.dawes_arcsec, 2)}″ · ${num(o.rayleigh_arcsec, 2)}″</span>
      <span>${t("observe.limMag")}</span><span>${num(o.limiting_mag, 1)}</span></div><p class="small muted">${t("observe.opticsNote")}</p>`;
  }

  fov() {                                             // degrees across the view
    const o = this.optics;
    if (!o) return 1;
    if (o.true_fov_deg && this.setup.eyepiece) return o.true_fov_deg;
    if (o.camera_fov_deg) return o.camera_fov_deg[0];
    return o.true_fov_deg || 1;
  }

  async loadWeather() {
    const p = new URLSearchParams(location.search);
    const force = p.get("wx");
    const q = `mode=${this.weather.mode}&t=${this.ctx.iso()}&${this.ctx.qs()}` + (force && this.weather.mode === "real" ? `&force=${force}` : "");
    try { this.weather.view = await this.ctx.api(`/api/observe/weather?${q}`); } catch (e) { this.weather.view = null; }
    const v = this.weather.view, el = this.el?.querySelector("#obWx");
    if (!el || !v) return;
    el.innerHTML = v.fallback ? `<span class="warn-text">${t("observe.wxFallback")}</span>`
      : v.mode === "perfect" ? t("observe.wxPerfect") : t("observe.wxReal", { cc: num(v.cloud_cover, 0) });
  }

  // ------------------------------------------------------------------------------------------- plan
  async planTarget(id) {
    this.target = id; this.survey = null;
    const out = this.el.querySelector("#obPlanOut");
    out.innerHTML = `<p class="muted">${t("common.computing")}</p>`;
    let p;
    try { p = await this.ctx.api(`/api/observe/plan?target=${encodeURIComponent(id)}&t=${this.ctx.iso()}&step_min=2&${this.ctx.qs()}`); }
    catch (e) { out.textContent = e.message; return; }
    this.plan = p;
    const w = p.best_window, tf = this.ctx.tfmt;
    let h = `<div id="obChart"></div>`;
    if (p.never_rises) h += `<div class="card warn-text">${t("observe.neverRises", { place: esc(this.ctx.place()) })}</div>`;
    h += `<div class="kv"><span>${t("tonight.rise")}</span><span>${p.rise.length ? tf(p.rise[0]) : "—"}</span><span>${t("tonight.transit")}</span><span>${p.transit.length ? tf(p.transit[0]) : "—"}</span>
      <span>${t("tonight.set")}</span><span>${p.set.length ? tf(p.set[0]) : "—"}</span>
      <span>${t("observe.astroNight")}</span><span>${p.astronomical_dusk.length ? tf(p.astronomical_dusk[0]) : "—"} – ${p.astronomical_dawn.length ? tf(p.astronomical_dawn[0]) : "—"}</span>
      <span>${t("body.moon")}</span><span>${num(p.moon_illuminated * 100, 0)}%</span></div>`;
    h += w ? `<div class="card">${t("observe.best", { start: tf(w.start), end: tf(w.end), when: tf(w.best), alt: num(w.alt, 0), X: num(w.airmass, 2), sep: num(w.moon_sep, 0), sky: num(w.sky_mag, 1) })}</div>`
      : (!p.never_rises ? `<p class="muted">${t("plan.none", { min: p.min_alt })}</p>` : "");
    out.innerHTML = h;
    const x = p.jd.map((j) => tf(new Date((j - 2440587.5) * 864e5).toISOString()));
    PChart.mount(out.querySelector("#obChart"), { x, height: 190, yLabel: t("unit.deg"), yMin: -40, yMax: 90,
      flags: p.sun_alt.map((s) => s > -18),
      series: [{ name: this.ctx.targetName(p.target), y: p.alt, color: "#7fb4ff", width: 2 }, { name: t("body.sun"), y: p.sun_alt, color: "#ffcf6e" },
        { name: t("body.moon"), y: p.moon_alt, color: "#c8c8d8", dash: "4 3" }],
      hlines: [{ y: 0, color: "#55607a", label: t("plan.horizon") }, { y: p.min_alt, color: "#3a4a66", dash: "2 3" }] });
  }

  // the target's direction now (three.js horizon frame), from the live sky (ephemeris / catalogue)
  targetVec() {
    const sky = this.ctx.sky, id = this.target;
    if (!id) return null;
    if (BODIES.includes(id)) {                     // the sky draws refracted places; the field is geometric
      const b = sky.bodies[id]; if (!b) return null;
      return altazToVec(sky.layers.atmosphere ? geometric(b.alt) : b.alt, b.az);
    }
    const p = this.plan?.target;
    if (!p || p.ra_deg === undefined || !sky.Mj2000) return null;
    return j2000ToVec(sky.Mj2000, p.ra_deg, p.dec_deg);
  }
  // where the mount points: the plan's (library) alt/az samples, interpolated at the current time
  mountAltAz(jd) {
    const p = this.plan;
    if (!p) return null;
    const alt = interp(p.jd, p.alt, jd);
    const az = Math.atan2(interp(p.jd, p.az.map((a) => Math.sin(a * D2R)), jd), interp(p.jd, p.az.map((a) => Math.cos(a * D2R)), jd)) * R2D;
    return [geometric(alt), (az + 360) % 360];        // the plan's altitudes are refracted
  }

  // ------------------------------------------------------------------------------------------- slew
  async slew() {
    const out = this.el.querySelector("#obSlewOut");
    if (!this.plan) await this.planTarget(this.el.querySelector("#obTarget").value);
    const sky = this.ctx.sky, to = this.mountAltAz(this.ctx.S.simJD);
    if (!to) return;
    const from = [sky.pitch, sky.yaw];
    const s = await this.ctx.api(`/api/observe/slew?from_alt=${from[0]}&from_az=${from[1]}&to_alt=${to[0]}&to_az=${to[1]}&lat=${this.ctx.S.obs.lat}&mount=${this.setup.mount}`);
    if (!s.ok) { out.innerHTML = `<span class="warn-text">${t("observe.belowLimit", { min: s.min_alt })}</span>`; return false; }
    out.innerHTML = t("observe.slewing", { s: num(s.seconds, 1) }) + (s.meridian_flip ? " · " + t("observe.flip") : "");
    // animate the sky view along the slew, with the same trapezoid profile in real time (capped at 12 s)
    const dur = Math.min(12, s.seconds) * 1000, t0 = performance.now(), dAz = ((to[1] - from[1] + 540) % 360) - 180;
    this.slewing = { t0, dur };
    await new Promise((res) => {
      const step = () => {
        const k = Math.min(1, (performance.now() - t0) / dur), e = k < 0.5 ? 2 * k * k : 1 - 2 * (1 - k) * (1 - k);
        const now = this.mountAltAz(this.ctx.S.simJD) || to;
        sky.lookAt(from[0] + (now[0] - from[0]) * e, (from[1] + dAz * e + 360) % 360);
        sky.fov = Math.max(5, sky.fov + (Math.max(this.fov() * 6, 5) - sky.fov) * e * 0.08);
        if (k < 1) requestAnimationFrame(step); else res();
      };
      step();
    });
    this.slewing = null;
    out.innerHTML = t("observe.slewDone", { s: num(s.seconds, 1) }) + (s.meridian_flip ? " · " + t("observe.flip") : "");
    this.addLog(t("observe.logSlew", { s: num(s.seconds, 1) }));
    return true;
  }

  // ------------------------------------------------------------------------------------------- track
  async startTracking() {
    if (!this.plan || this.plan.target.id !== this.el.querySelector("#obTarget").value) await this.planTarget(this.el.querySelector("#obTarget").value);
    if (!this.plan) return;
    const box = document.querySelector("#eyepiece");
    box.hidden = false; this.tracking = true;
    this.state = { offset_px: 0, max_offset_px: 0, rotation_deg: 0, hidden: false, frames: 0, startJD: this.ctx.S.simJD };
    box.innerHTML = `<canvas id="epCanvas"></canvas><div class="ep-hud" id="epHud"></div><div class="ep-credit" id="epCredit"></div>
      <div class="ep-ctl"><span class="small">${t("observe.speed")}</span>${[1, 60, 600, 3600].map((s) => `<button class="ghost" data-speed="${s}">${s}×</button>`).join("")}
      <button class="ghost" id="epSnap">${t("observe.snapshot")}</button><button class="ghost" id="epClose">${t("learn.close")}</button></div>`;
    for (const b of box.querySelectorAll("[data-speed]")) b.onclick = () => { this.ctx.S.speed = +b.dataset.speed; this.ctx.S.playing = true; };
    box.querySelector("#epClose").onclick = () => this.stopTracking();
    box.querySelector("#epSnap").onclick = () => this.addLog(t("observe.snapshot"), true);
    this.loadSurvey();
    this.addLog(t("observe.logStart", { name: this.ctx.targetName(this.plan.target) }));
    await this.loadWeather();
  }
  stopTracking() {
    this.tracking = false;
    const box = document.querySelector("#eyepiece"); box.hidden = true; box.innerHTML = "";
    this.addLog(t("observe.logStop"));
  }
  loadSurvey() {
    const p = this.plan?.target;
    if (!p || p.ra_deg === undefined) { this.survey = null; return; }
    const fov = Math.max(0.1, Math.min(5, this.fov() * 1.5));
    const img = new Image();
    img.onload = () => { this.survey = { img, fov }; };
    img.onerror = () => { this.survey = { failed: true }; };
    img.src = `/api/observe/field.jpg?ra=${p.ra_deg.toFixed(4)}&dec=${p.dec_deg.toFixed(4)}&fov=${fov.toFixed(3)}&size=512`;
  }
  texture(b) {
    if (!this.textures[b]) { const im = new Image(); im.src = `/static/textures/${b}.jpg`; this.textures[b] = im; }
    return this.textures[b];
  }

  // one frame of the eyepiece / camera view: gnomonic projection around the mount's pointing
  frame() {
    if (!this.tracking) return;
    const cv = document.querySelector("#epCanvas");
    if (!cv || !this.plan) return;
    const S = this.ctx.S, sky = this.ctx.sky, p = this.plan, jd = S.simJD;
    const W = cv.clientWidth, H = cv.clientHeight, dpr = Math.min(2, window.devicePixelRatio || 1);
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const fov = this.fov(), pxPerDeg = Math.min(W, H) / fov, cx = W / 2, cy = H / 2;
    const point = this.mountAltAz(jd);
    const c = altazToVec(point[0], point[1]);
    // the view's "up": the zenith for an alt-az mount, the celestial pole for an equatorial one (no field rotation)
    const H0 = sky.Mdate;
    let up = this.setup.mount === "equatorial" && H0 ? [H0[0][2], H0[1][2], H0[2][2]] : [0, 1, 0];
    if (S.obs.lat < 0 && this.setup.mount === "equatorial") up = up.map((x) => -x);
    const eu = norm(sub(up, mulv(c, dot(up, c)))), er = cross(c, eu);
    const proj = (v) => { const d = dot(v, c); if (d <= 0) return null; return [cx + dot(v, er) / d * R2D * pxPerDeg, cy - dot(v, eu) / d * R2D * pxPerDeg]; };
    // conditions at this time, from the plan (library): sky brightness, extinction, air mass, field rotation
    const sm = interp(p.jd, p.sky_mag, jd), alt = interp(p.jd, p.alt, jd);
    const ext = (b) => interp(p.jd, p.extinction[b].map((x) => x ?? 40), jd);
    const eV = ext("V"), eB = ext("B"), eR = ext("R");
    const skyL = Math.min(1, Math.pow(10, -0.4 * (sm - 13.0)));          // 13 mag/arcsec^2 ~ a bright twilight
    g.fillStyle = `rgb(${Math.round(6 + 120 * skyL)},${Math.round(9 + 140 * skyL)},${Math.round(18 + 170 * skyL)})`;
    g.fillRect(0, 0, W, H);
    const lim = (this.optics?.limiting_mag ?? 12) - Math.max(0, (21.6 - sm) * 0.8) - eV;
    const blurPx = this.seeing / 3600 * pxPerDeg;
    g.save(); if (blurPx > 0.6) g.filter = `blur(${Math.min(8, blurPx / 2).toFixed(2)}px)`;
    // the target itself: survey image, planet disc, or a fuzzy catalogue shape
    const tv = this.targetVec();
    const tp = tv ? proj(tv) : null;
    if (this.survey?.img && p.target.ra_deg !== undefined) {
      const s = this.survey.fov * pxPerDeg;
      const q = this.setup.mount === "equatorial" ? 0 : interp(p.jd, p.parallactic_deg, jd) * D2R;
      g.save(); g.translate(tp ? tp[0] : cx, tp ? tp[1] : cy); g.rotate(-q); g.globalAlpha = Math.max(0.15, 1 - skyL) * Math.pow(10, -0.4 * eV * 0.5);
      g.drawImage(this.survey.img, -s / 2, -s / 2, s, s); g.restore();
    }
    // stars of the catalogue in the field (Yale BSC), dimmed and reddened by extinction
    const cat = sky.cat, M = sky.Mj2000, r = fov * 0.75;
    if (cat && M) {
      for (let i = 0; i < cat.ra.length; i++) {
        if (cat.v[i] > lim) continue;
        const v = j2000ToVec(M, cat.ra[i], cat.dec[i]);
        if (Math.acos(Math.min(1, dot(v, c))) * R2D > r) continue;
        const pp = proj(v); if (!pp) continue;
        const m = cat.v[i] + eV, size = Math.min(6, Math.max(0.8, 1.2 + (lim - m) * 0.35));
        const red = Math.pow(10, -0.4 * (eR - eV)), blue = Math.pow(10, -0.4 * (eB - eV));
        g.fillStyle = `rgba(${Math.min(255, Math.round(255 * red))},${Math.round(240)},${Math.min(255, Math.round(255 * blue))},${Math.min(1, 0.35 + (lim - m) * 0.25)})`;
        g.beginPath(); g.arc(pp[0], pp[1], size, 0, 7); g.fill();
      }
    }
    if (tp && BODIES.includes(this.target)) {
      const now = S.skyData?.now?.[this.target];
      const rad = Math.max(2, (now?.angular_diameter_arcsec ?? 10) / 7200 * pxPerDeg);
      const tex = this.texture(this.target);
      g.save(); g.beginPath(); g.arc(tp[0], tp[1], rad, 0, 7); g.clip();
      g.fillStyle = BODY_COLORS[this.target]; g.fill();
      if (tex.complete && tex.naturalWidth) { g.globalAlpha = 0.9; g.drawImage(tex, tex.naturalWidth * 0.25, 0, tex.naturalWidth * 0.5, tex.naturalHeight, tp[0] - rad, tp[1] - rad, 2 * rad, 2 * rad); }
      g.restore();
      if (this.target === "saturn") { g.strokeStyle = "rgba(232,211,154,.8)"; g.lineWidth = Math.max(1, rad * 0.25); g.beginPath(); g.ellipse(tp[0], tp[1], rad * 2.2, rad * 0.8, 0.3, 0, 7); g.stroke(); }
    } else if (tp && !this.survey?.img && !["oc", "pos", "star", "constellation"].includes(p.target.kind)) {   // diffuse objects only
      const size = (Math.max(...String(p.target.size_arcmin || "5").split("x").map(Number)) || 5) / 60 * pxPerDeg;
      const grd = g.createRadialGradient(tp[0], tp[1], 0, tp[0], tp[1], Math.max(4, size / 2));
      grd.addColorStop(0, "rgba(220,225,255,.55)"); grd.addColorStop(1, "rgba(220,225,255,0)");
      g.fillStyle = grd; g.beginPath(); g.arc(tp[0], tp[1], Math.max(4, size / 2), 0, 7); g.fill();
    }
    g.restore();
    // clouds
    const wv = this.weather.view;
    const hidden = !!(wv && wv.mode === "real" && wv.hidden);
    if (wv && wv.mode === "real" && wv.cloud_cover > 0) { g.fillStyle = `rgba(120,124,132,${Math.min(0.97, wv.cloud_cover / 100)})`; g.fillRect(0, 0, W, H); }
    if (hidden) { g.fillStyle = "#e6ecf5"; g.font = "600 16px system-ui"; g.textAlign = "center"; g.fillText(t("observe.cloudsHide"), cx, cy); }
    if (hidden !== this.state.hidden) { this.state.hidden = hidden; this.addLog(hidden ? t("observe.logClouds", { cc: num(wv.cloud_cover, 0) }) : t("observe.logClear")); }
    // crosshair, tracking error, field rotation
    g.strokeStyle = "rgba(255,90,90,.55)"; g.lineWidth = 1; g.beginPath(); g.moveTo(cx - 10, cy); g.lineTo(cx + 10, cy); g.moveTo(cx, cy - 10); g.lineTo(cx, cy + 10); g.stroke();
    this.state.offset_px = tp ? Math.hypot(tp[0] - cx, tp[1] - cy) : null;
    if (tp && !this.slewing) this.state.max_offset_px = Math.max(this.state.max_offset_px, this.state.offset_px);
    const rot = this.setup.mount === "equatorial" ? 0 : interp(p.jd, p.parallactic_deg, jd) - interp(p.jd, p.parallactic_deg, this.state.startJD);
    this.state.rotation_deg = rot; this.state.frames++; this.state.alt = alt; this.state.jd = jd;
    const fr = interp(p.jd, p.field_rotation_deg_s.map((x) => x ?? 0), jd);
    this.ctx.sky.lookAt(point[0], point[1]);                 // the finder (sky view) follows the mount
    document.querySelector("#epHud").innerHTML = `<b>${esc(this.ctx.targetName(p.target))}</b> · ${this.ctx.tfmt(new Date((jd - 2440587.5) * 864e5).toISOString())}<br>
      ${t("observe.hudAlt", { alt: num(alt, 1), X: alt > 0 ? num(interp(p.jd, p.airmass.map((x) => x ?? 40), jd), 2) : "—", ext: num(eV, 2), sky: num(sm, 1) })}<br>
      ${this.setup.mount === "equatorial" ? t("observe.hudNoRot") : t("observe.hudRot", { rate: num(fr * 60, 3), total: num(rot, 1) })} · ${t("observe.hudFov", { fov: num(fov * 60, 0) })}`;
    document.querySelector("#epCredit").textContent = this.survey?.img ? t("observe.credit", { credit: this.presets.survey_credit }) : t("observe.catalogueField");
    if (alt < 0 && !this.state.setLogged) { this.state.setLogged = true; this.addLog(t("observe.logSet")); }
  }

  // ------------------------------------------------------------------------------------------- session log
  addLog(note, snapshot = false) {
    const S = this.ctx.S, p = this.plan, jd = S.simJD;
    const e = { iso: new Date((jd - 2440587.5) * 864e5).toISOString(), note };
    if (p) {
      e.alt = interp(p.jd, p.alt, jd); e.airmass = interp(p.jd, p.airmass.map((x) => x ?? 40), jd); e.sky_mag = interp(p.jd, p.sky_mag, jd);
    }
    e.weather = this.weather.view ? { mode: this.weather.view.mode, cloud_cover: this.weather.view.cloud_cover, hidden: !!this.weather.view.hidden } : null;
    e.seeing = this.seeing;
    if (snapshot) { const cv = document.querySelector("#epCanvas"); if (cv) try { e.snapshot = cv.toDataURL("image/jpeg", 0.6); } catch (err) { /* tainted */ } }
    this.log.push(e);
    this.renderLog();
  }
  renderLog() {
    const el = this.el?.querySelector("#obLog");
    if (!el) return;
    el.innerHTML = this.log.length ? `<table class="small">${this.log.slice().reverse().map((e) => `<tr><td>${this.ctx.tfmt(e.iso)}</td><td>${esc(e.note)}${e.snapshot ? ` <img class="logsnap" src="${e.snapshot}" alt="">` : ""}<br><span class="muted">${e.alt !== undefined ? t("observe.logCond", { alt: num(e.alt, 1), X: e.alt > 0 ? num(e.airmass, 2) : "—", sky: num(e.sky_mag, 1), seeing: num(e.seeing, 1) }) : ""}${e.weather ? " · " + (e.weather.mode === "real" ? t("observe.logCloud", { cc: num(e.weather.cloud_cover, 0) }) : t("observe.perfectSky")) : ""}</span></td></tr>`).join("")}</table>
      <button class="ghost" id="obLogSave">${t("observe.download")}</button>` : `<p class="small muted">${t("observe.logEmpty")}</p>`;
    const b = el.querySelector("#obLogSave");
    if (b) b.onclick = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(this.log, null, 1)], { type: "application/json" })); a.download = "observing-log.json"; a.click(); };
  }
}

// --------------------------------------------------------------------------------------------- vectors
function geometric(alt) { return alt - refractionDeg(alt - refractionDeg(alt)); }
function altazToVec(alt, az) { const a = alt * D2R, z = az * D2R; return [Math.cos(a) * Math.sin(z), Math.sin(a), -Math.cos(a) * Math.cos(z)]; }
function j2000ToVec(M, ra, dec) { const s = radecVec(ra, dec); return [0, 1, 2].map((r) => M[r][0] * s[0] + M[r][1] * s[1] + M[r][2] * s[2]); }
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mulv = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const n = Math.hypot(...a) || 1; return mulv(a, 1 / n); };
