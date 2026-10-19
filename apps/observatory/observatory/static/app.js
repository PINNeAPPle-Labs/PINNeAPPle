import * as THREE from "three";
import { SkyView } from "./sky.js";
import { SystemView } from "./system.js";
import { D2R, R2D, compass, dateToJd, fmtDec, fmtRA, jdToDate, radecVec } from "./astro.js";
import { renderBlackHole, renderConditions, renderDecay, renderSmallBodies, renderSpaceWeather, renderStarsML, checkName, verdictBadge } from "./ai.js";
import { applyStatic, fmtDate, lang, num, onLangChange, setLang, sig, t, tOr } from "./i18n.js";
import { renderLearn, startupTip } from "./learn.js";
import { Observe } from "./observe.js";
import { JourneyView } from "./journey.js";

// ------------------------------------------------------------------------------------------------- setup
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const params = new URLSearchParams(location.search);
// [catalogue key, lat, lon, elevation m, time zone]
const CITIES = [
  ["saopaulo", -23.5505, -46.6333, 760, "America/Sao_Paulo"], ["rio", -22.9068, -43.1729, 10, "America/Sao_Paulo"],
  ["brasilia", -15.7939, -47.8828, 1170, "America/Sao_Paulo"], ["lisbon", 38.7223, -9.1393, 50, "Europe/Lisbon"],
  ["london", 51.5074, -0.1278, 20, "Europe/London"], ["newyork", 40.7128, -74.006, 10, "America/New_York"],
  ["sanfrancisco", 37.7749, -122.4194, 20, "America/Los_Angeles"], ["tokyo", 35.6762, 139.6503, 40, "Asia/Tokyo"],
  ["sydney", -33.8688, 151.2093, 50, "Australia/Sydney"], ["atacama", -23.0293, -67.7548, 5050, "America/Santiago"],
  ["maunakea", 19.8207, -155.4681, 4205, "Pacific/Honolulu"], ["reykjavik", 64.1466, -21.9426, 30, "Atlantic/Reykjavik"],
  ["capetown", -33.9249, 18.4241, 20, "Africa/Johannesburg"], ["equator", 0, 0, 0, "UTC"],
];
const canvas = $("#gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: params.has("capture") });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.outputColorSpace = THREE.SRGBColorSpace;

class Labels {
  constructor(el) { this.el = el; this.pool = []; this.n = 0; }
  begin() { this.n = 0; }
  add(x, y, text, cls, id) {
    let d = this.pool[this.n];
    if (!d) { d = document.createElement("div"); this.el.appendChild(d); this.pool.push(d); }
    if (d._t !== text) { d.textContent = text; d._t = text; }
    const c = "lbl " + cls; if (d.className !== c) d.className = c;
    d.style.left = x.toFixed(1) + "px"; d.style.top = y.toFixed(1) + "px"; d.style.display = "";
    this.n++;
  }
  end() { for (let i = this.n; i < this.pool.length; i++) this.pool[i].style.display = "none"; }
}
const labels = new Labels($("#labels"));
const sky = new SkyView(renderer, labels);
const sys = new SystemView(renderer, labels);
const journey = new JourneyView(renderer, labels);

const S = {
  obs: { name: "saopaulo", lat: -23.5505, lon: -46.6333, elev: 760, tz: "America/Sao_Paulo" },
  simJD: dateToJd(new Date()), speed: 1, playing: true, view: "sky", sliderBase: null,
  skyData: null, skyLoading: false, sysData: null, sysLoading: false, satData: null, satLoading: false,
  selected: null, lastNowRefresh: 0,
};
try { const saved = JSON.parse(localStorage.getItem("obs.loc") || "null"); if (saved && saved.lat !== undefined) S.obs = saved; } catch (e) { /* storage blocked */ }
if (params.has("lat")) S.obs = { name: params.get("name") || "custom", lat: +params.get("lat"), lon: +params.get("lon"), elev: +(params.get("elev") || 0), tz: params.get("tz") || Intl.DateTimeFormat().resolvedOptions().timeZone };
if (params.has("t")) S.simJD = dateToJd(new Date(params.get("t")));
if (params.has("speed")) S.speed = +params.get("speed");
if (params.has("fov")) sky.fov = +params.get("fov");
sky.lookAt(params.has("alt") ? +params.get("alt") : 30, params.has("az") ? +params.get("az") : (S.obs.lat < 0 ? 180 : 0));
if (params.get("layers")) for (const kv of params.get("layers").split(",")) { const [k, v] = kv.split(":"); sky.layers[k] = v !== "0"; }

// the observer's place in the interface language: a city from the list, "my location", or a name from the URL
function placeName(o = S.obs) {
  const c = CITIES.find((x) => Math.abs(x[1] - o.lat) < 1e-3 && Math.abs(x[2] - o.lon) < 1e-3);
  if (c) return t("city." + c[0]);
  if (o.name === "here" || o.name === "My location") return t("loc.here");
  if (!o.name || o.name === "custom" || o.name === "Custom") return t("loc.custom");
  return o.name;
}

// --------------------------------------------------------------------------------------------- data I/O
// API errors carry a message code (X-Error-Code) so they are shown in the interface language
async function api(path) {
  let r;
  try { r = await fetch(path); } catch (e) { throw new Error(t("err.network")); }
  if (!r.ok) {
    const code = r.headers.get("X-Error-Code");
    let args = {};
    try { args = JSON.parse(r.headers.get("X-Error-Args") || "{}"); } catch (e) { /* none */ }
    if (code) throw new Error(tOr("err." + code, t("err.generic", { status: r.status }), args));
    if (r.status === 429) throw new Error(t("err.busy"));
    throw new Error(t("err.generic", { status: r.status }));
  }
  return r.json();
}
const isoOf = (jd) => jdToDate(jd).toISOString();
const qs = () => `lat=${S.obs.lat}&lon=${S.obs.lon}&elev=${S.obs.elev || 0}`;
function viewRaDec() {                                   // J2000 RA/Dec of the centre of the sky view
  const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(sky.camera.quaternion);
  const M = sky.Mj2000; const v = [0, 1, 2].map((c) => M[0][c] * dir.x + M[1][c] * dir.y + M[2][c] * dir.z);
  return [((Math.atan2(v[1], v[0]) * R2D) + 360) % 360, Math.asin(v[2]) * R2D];
}
function aboveHorizon() {
  const out = new Set(), M = sky.Mj2000, cat = sky.cat;
  if (!M || !cat) return out;
  for (let i = 0; i < cat.hr.length; i++) { const s = radecVec(cat.ra[i], cat.dec[i]); if (M[1][0] * s[0] + M[1][1] * s[1] + M[1][2] * s[2] > 0) out.add(cat.hr[i]); }
  return out;
}
const ctx = { iso: () => isoOf(S.simJD), qs, viewRaDec, aboveHorizon, api,
  showSmallBody: async (key) => { const tr = await api(`/api/smallbodies/track?body=${key}&t=${isoOf(S.simJD)}&days=600`); sys.setSmallBody(tr); if (S.view !== "system") switchView("system"); } };

async function loadSky(force) {
  if (S.skyLoading) return;
  const d = S.skyData;
  const sp = Math.abs(S.speed);
  const span = sp > 600 ? 48 : sp > 30 ? 12 : 4;
  if (!force && d && d.observer.lat === S.obs.lat && d.observer.lon === ((S.obs.lon + 540) % 360) - 180) {
    const j = d.samples.jd, margin = (j[j.length - 1] - j[0]) * 0.2;
    if (S.simJD > j[0] + margin && S.simJD < j[j.length - 1] - margin) return;
  }
  S.skyLoading = true;
  try {
    const data = await api(`/api/sky?${qs()}&t=${isoOf(S.simJD)}&span_h=${span}&step_min=${(span * 60 / 72).toFixed(2)}`);
    S.skyData = data; sky.setSky(data); renderTonight(); if (S.selected) renderObject(S.selected);
  } catch (e) { toast(t("toast.sky", { msg: e.message })); }
  S.skyLoading = false;
}
async function loadSystem(force) {
  if (S.sysLoading) return;
  const d = S.sysData;
  if (!force && d && S.simJD + 0.0008 > d.jd[0] + 3 && S.simJD < d.jd[d.jd.length - 1] - 3) return;
  S.sysLoading = true;
  try { const data = await api(`/api/system?t=${isoOf(S.simJD)}&days=60`); S.sysData = data; sys.setData(data); }
  catch (e) { toast(t("toast.system", { msg: e.message })); }
  S.sysLoading = false;
}
async function loadSats(force) {
  if (S.satLoading || !sky.layers.satellites || Math.abs(S.speed) > 120) return;
  const d = S.satData;
  if (!force && d && S.simJD > d.jd[0] + 0.001 && S.simJD < d.jd[d.jd.length - 1] - 0.001) return;
  if (!force && d && d.satellites.length === 0 && performance.now() - (S.satTried || 0) < 60000) return;
  S.satLoading = true; S.satTried = performance.now();
  try { const data = await api(`/api/satellites?${qs()}&t=${isoOf(S.simJD)}&minutes=20&step_s=10&limit=60`); S.satData = data; sky.setSats(data); }
  catch (e) { /* satellites are optional */ }
  S.satLoading = false;
}

// ------------------------------------------------------------------------------------------------ the loop
let last = performance.now();
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  for (const c of [sky.camera, sys.camera, journey.camera]) { c.aspect = w / h; c.updateProjectionMatrix(); }
}
window.addEventListener("resize", resize); resize();

function frame() {
  const now = performance.now();               // (a capture script may drive this clock)
  const dt = Math.min(0.25, (now - last) / 1000); last = now;
  if (S.playing && !params.has("freeze")) S.simJD += dt * S.speed / 86400;
  const w = window.innerWidth, h = window.innerHeight;
  if (S.view === "sky") {
    loadSky(); loadSats();
    sky.update(S.simJD, now / 1000);
    sky.camera.updateProjectionMatrix(); sky.camera.updateMatrixWorld();
    sky.render(); sky.drawLabels(w, h);
  } else if (S.view === "journey") {
    journey.update(dt, S.simJD);
    journey.camera.updateMatrixWorld();
    journey.render(); journey.drawLabels(w, h, journeyName); journeyHud();
  } else {
    loadSystem();
    sys.observer = S.obs;
    sys.update(S.simJD);
    sys.render(); sys.drawLabels(w, h);
  }
  if (S.view === "sky" && S.skyData && now - S.lastNowRefresh > 20000 && Math.abs(S.speed) <= 60) {
    S.lastNowRefresh = now; if (Math.abs(S.simJD - S.skyData.jd_utc) * 86400 > 60) loadSky(true);
  }
  if (observe.tracking) observe.frame();
  updateClock();
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------------------------- the clock
function updateClock() {
  const d = jdToDate(S.simJD);
  const tz = S.obs.tz || undefined;
  $("#clock").textContent = fmtDate(d, { timeZone: tz, weekday: "short", day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const sun = sky.bodies?.sun?.alt;
  const lst = sky.lst / 15;
  const utc = fmtDate(d, { timeZone: "UTC", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  $("#clockSub").textContent = t("clock.sub", { utc, h: Math.floor(lst), m: String(Math.floor((lst % 1) * 60)).padStart(2, "0"), place: placeName(), lat: num(S.obs.lat, 3), lon: num(S.obs.lon, 3) }) +
    (sun !== undefined && S.view === "sky" ? " · " + t("clock.sun", { alt: num(sun, 1) }) : "");
  const sp = S.speed, a = Math.abs(sp);
  $("#speed").textContent = !S.playing ? t("hud.paused") : sp === 1 ? "×1" : "×" + (a >= 3600 ? t("hud.perSecond", { v: sp / 3600, u: t("unit.h") }) : a >= 60 ? t("hud.perSecond", { v: sp / 60, u: t("unit.min") }) : sp);
  $("#tPlay").textContent = S.playing ? "❚❚" : "▶";
}

// ------------------------------------------------------------------------------------------ the controls
const SPEEDS = [-86400, -3600, -600, -60, 1, 60, 600, 3600, 86400];
$("#tPlay").onclick = () => { S.playing = !S.playing; };
$("#tNow").onclick = () => { S.simJD = dateToJd(new Date()); S.speed = 1; S.playing = true; $("#tSlider").value = 0; S.sliderBase = null; loadSky(true); };
$("#tFwd").onclick = () => { const i = SPEEDS.indexOf(S.speed); S.speed = SPEEDS[Math.min(SPEEDS.length - 1, (i < 0 ? 4 : i) + 1)]; S.playing = true; };
$("#tBack").onclick = () => { const i = SPEEDS.indexOf(S.speed); S.speed = SPEEDS[Math.max(0, (i < 0 ? 4 : i) - 1)]; S.playing = true; };
$("#tSlider").oninput = (e) => {
  if (S.sliderBase === null) S.sliderBase = S.simJD;
  S.simJD = S.sliderBase + (+e.target.value) / 1440; S.playing = false;
};
$("#tSlider").onchange = () => { loadSky(); };
$("#tDate").onchange = (e) => { if (e.target.value) { S.simJD = dateToJd(new Date(e.target.value)); S.sliderBase = null; $("#tSlider").value = 0; loadSky(true); loadSystem(true); } };
$("#btnPanel").onclick = () => $("#panel").classList.toggle("open");
$("#tLayers").onclick = () => $("#hud").classList.toggle("showlayers");
if (window.innerWidth < 760 && !params.has("panel")) $("#panel").classList.remove("open");
if (params.get("panel") === "0") $("#panel").classList.remove("open");

const SIZE_K = params.has("size") ? +params.get("size") : 12;     // planet size exaggeration in the system view
sys.setSizeScale(SIZE_K);
const LAYERS = ["constellations", "names", "altaz", "equatorial", "milkyway", "atmosphere", "ground", "satellites", "planetsBig"];
function renderLayers() {
  const logOn = $("#logScale") ? $("#logScale").checked : true, k = $("#sizeK") ? $("#sizeK").value : SIZE_K;
  $("#layers").innerHTML = LAYERS.map((k) => `<label><input type="checkbox" data-k="${k}" ${sky.layers[k] ? "checked" : ""}>${t("layer." + k)}</label>`).join("") +
    `<label class="sysonly"><input type="checkbox" id="logScale" ${logOn ? "checked" : ""}>${t("layer.log")}</label>` +
    `<label class="sysonly">${t("layer.size")}<input type="range" id="sizeK" min="1" max="24" value="${k}" step="0.5" style="width:90px"><span id="sizeKv">${k}</span></label>`;
  for (const cb of document.querySelectorAll("#layers input[data-k]")) cb.onchange = (e) => { sky.layers[e.target.dataset.k] = e.target.checked; if (e.target.dataset.k === "satellites") loadSats(true); };
  $("#logScale").onchange = (e) => { sys.logScale = e.target.checked; sys._buildOrbits(); };
  $("#sizeK").oninput = (e) => { sys.setSizeScale(+e.target.value); $("#sizeKv").textContent = e.target.value; };
}
renderLayers();

// location
const sel = $("#city");
function renderCities() {
  sel.innerHTML = CITIES.map((c, i) => `<option value="${i}">${esc(t("city." + c[0]))}</option>`).join("") + `<option value="here">${esc(t("loc.here"))}</option>`;
  const ci = CITIES.findIndex((c) => Math.abs(c[1] - S.obs.lat) < 1e-3 && Math.abs(c[2] - S.obs.lon) < 1e-3);
  sel.value = ci >= 0 ? ci : "here";
}
renderCities();
sel.onchange = () => {
  if (sel.value === "here") return locate();
  const c = CITIES[+sel.value];
  setObserver({ name: c[0], lat: c[1], lon: c[2], elev: c[3], tz: c[4] });
};
$("#btnLocate").onclick = locate;
function locate() {
  if (!navigator.geolocation) return toast(t("toast.noGeo"));
  navigator.geolocation.getCurrentPosition((p) => {
    setObserver({ name: "here", lat: +p.coords.latitude.toFixed(4), lon: +p.coords.longitude.toFixed(4), elev: Math.max(0, Math.round(p.coords.altitude || 0)), tz: Intl.DateTimeFormat().resolvedOptions().timeZone });
    sel.value = "here";
  }, (e) => toast(t("toast.location", { msg: e.message })), { enableHighAccuracy: false, timeout: 10000 });
}
function setObserver(o) {
  S.obs = o;
  try { localStorage.setItem("obs.loc", JSON.stringify(o)); } catch (e) { /* ignore */ }
  $("#locText").textContent = `${num(o.lat, 3)}°, ${num(o.lon, 3)}°`;
  S.satData = null; loadSky(true); loadSats(true); renderSatsPanel();
}
$("#locText").textContent = `${num(S.obs.lat, 3)}°, ${num(S.obs.lon, 3)}°`;

function toast(msg) { const el = $("#toast"); el.textContent = msg; el.hidden = false; clearTimeout(el._h); el._h = setTimeout(() => (el.hidden = true), 4000); }

// language switcher
for (const b of document.querySelectorAll("#langSel button")) b.onclick = () => setLang(b.dataset.lang);
function setLangButtons() { for (const b of document.querySelectorAll("#langSel button")) b.classList.toggle("on", b.dataset.lang === lang); }

// --------------------------------------------------------------------------------------- pointer / touch
let drag = null; const pointers = new Map();
canvas.addEventListener("pointerdown", (e) => {
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  drag = { x: e.clientX, y: e.clientY, moved: 0, fov: sky.fov, dist: pinchDist() };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId) || !drag) return;
  const prev = pointers.get(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]);
  drag.moved += Math.abs(e.clientX - prev[0]) + Math.abs(e.clientY - prev[1]);
  if (S.view === "journey") { journey.yaw -= (e.clientX - prev[0]) * 0.2; journey.pitch = Math.max(-80, Math.min(80, journey.pitch - (e.clientY - prev[1]) * 0.2)); return; }
  if (S.view !== "sky" || sky.gyro) return;
  if (pointers.size === 2) { const d = pinchDist(); if (drag.dist) sky.fov = clampFov(drag.fov * drag.dist / d); return; }
  const k = sky.fov / window.innerHeight;
  sky.yaw = (sky.yaw - (e.clientX - prev[0]) * k + 360) % 360;
  sky.pitch = Math.max(-30, Math.min(89.5, sky.pitch + (e.clientY - prev[1]) * k));
});
canvas.addEventListener("pointerup", (e) => {
  pointers.delete(e.pointerId);
  if (drag && drag.moved < 6) onClick(e);
  if (!pointers.size) drag = null;
});
canvas.addEventListener("wheel", (e) => { if (S.view === "sky") { e.preventDefault(); sky.fov = clampFov(sky.fov * Math.exp(e.deltaY * 0.0012)); } }, { passive: false });
function pinchDist() { const p = [...pointers.values()]; return p.length === 2 ? Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]) : 0; }
function clampFov(f) { return Math.max(0.5, Math.min(110, f)); }
function onClick(e) {
  const x = (e.clientX / window.innerWidth) * 2 - 1, y = -(e.clientY / window.innerHeight) * 2 + 1;
  if (S.view === "journey") return;
  const hit = S.view === "sky" ? sky.pick(x, y) : sys.pick(x, y);
  if (!hit) return;
  S.selected = hit; renderObject(hit); openTab("object");
  if (S.view === "system" && hit.kind === "body") focusSystem(hit.id);
}
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { S.playing = !S.playing; e.preventDefault(); }
  if (e.key === "+") sky.fov = clampFov(sky.fov / 1.25);
  if (e.key === "-") sky.fov = clampFov(sky.fov * 1.25);
});

// ------------------------------------------------------------------------------------------------- gyro
const hasOrientation = "DeviceOrientationEvent" in window && /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
if (hasOrientation) $("#btnGyro").hidden = false;
const gyroLabel = () => { $("#btnGyro").textContent = sky.gyro ? t("top.drag") : t("top.point"); };
$("#btnGyro").onclick = async () => {
  if (sky.gyro) { sky.gyro = false; window.removeEventListener("deviceorientation", onOrient); gyroLabel(); $("#reticle").hidden = true; return; }
  try { if (typeof DeviceOrientationEvent.requestPermission === "function") { const r = await DeviceOrientationEvent.requestPermission(); if (r !== "granted") return toast(t("toast.motionDenied")); } }
  catch (e) { return toast(t("toast.motionNone")); }
  sky.gyro = true; gyroLabel(); $("#reticle").hidden = false;
  window.addEventListener("deviceorientationabsolute", onOrient); window.addEventListener("deviceorientation", onOrient);
  toast(t("toast.pointPhone"));
};
const zee = new THREE.Vector3(0, 0, 1), q0 = new THREE.Quaternion(), q1 = new THREE.Quaternion(-Math.sqrt(0.5), 0, 0, Math.sqrt(0.5));
function onOrient(e) {
  if (!sky.gyro || e.beta === null) return;
  if (e.type === "deviceorientation" && e.absolute === false && e.webkitCompassHeading === undefined && window.ondeviceorientationabsolute !== undefined) return;
  // alpha: rotation about the vertical, 0 = north for absolute events; iOS reports the compass heading separately
  const alpha = (e.webkitCompassHeading !== undefined ? (360 - e.webkitCompassHeading) : e.alpha) * D2R;
  const beta = e.beta * D2R, gamma = e.gamma * D2R, orient = (screen.orientation?.angle || window.orientation || 0) * D2R;
  const euler = new THREE.Euler(beta, alpha, -gamma, "YXZ");
  const q = new THREE.Quaternion().setFromEuler(euler).multiply(q1).multiply(q0.setFromAxisAngle(zee, -orient));
  sky.camera.quaternion.copy(q);
}

// ------------------------------------------------------------------------------------ view transitions
$("#vSky").onclick = () => switchView("sky");
$("#vSys").onclick = () => switchView("system");
$("#vJour").onclick = () => openJourney(journey.plan?.target?.body || journey.plan?.target?.id || "moon");
function setViewButtons() {
  for (const [id, v] of [["#vSky", "sky"], ["#vSys", "system"], ["#vJour", "journey"]]) $(id).classList.toggle("on", S.view === v);
  document.body.dataset.view = S.view;
  $("#journey").hidden = S.view !== "journey";
}
async function switchView(v, opts = {}) {
  if (v === S.view) return;
  const fade = $("#fade");
  if (v === "journey" || S.view === "journey") {                 // a plain cross-fade to and from the journey
    fade.style.opacity = 1; await sleep(450);
    if (S.view === "system") sys.controls.enabled = false;
    S.view = v; setViewButtons();
    if (v === "sky") loadSky(true);
    if (v === "system") { await loadSystem(true); sys.controls.enabled = true; const o = sys.overviewShot(); sys.camera.position.copy(o.pos); sys.controls.target.copy(o.target); }
    fade.style.opacity = 0;
    return;
  }
  if (v === "system") {
    await loadSystem(true);
    // widen the field of view in the sky, fade, then start from just above the observer and fly out
    const f0 = sky.fov; const t0 = performance.now();
    await new Promise((res) => { const step = () => { const k = Math.min(1, (performance.now() - t0) / 700); sky.fov = f0 + (110 - f0) * k * k; if (k < 1) requestAnimationFrame(step); else res(); }; step(); });
    fade.style.opacity = 1; await sleep(450);
    S.view = "system"; setViewButtons(); sys.controls.enabled = true; sys.focus = null;
    sys.observer = S.obs; sys.update(S.simJD);
    const a = sys.observerShot(); sys.camera.position.copy(a.pos); sys.controls.target.copy(a.target);
    sky.fov = f0;
    fade.style.opacity = 0;
    const mid = sys.planets.earth.grp.position.clone();
    sys.flyTo(mid.clone().add(a.pos.clone().sub(mid).normalize().multiplyScalar(2.2)).add(new THREE.Vector3(0, 1.2, 0)), mid, opts.ms1 || 2600, () => {
      const o = sys.overviewShot(panelCover(), hudCover()); sys.flyTo(o.pos, o.target, opts.ms2 || 3200);
    });
  } else {
    sys.focus = null;
    const a = sys.observerShot();
    await new Promise((res) => sys.flyTo(a.pos, a.target, 2200, res));
    fade.style.opacity = 1; await sleep(450);
    S.view = "sky"; setViewButtons(); sys.controls.enabled = false; loadSky(true);
    fade.style.opacity = 0;
  }
}
// fraction of the canvas width hidden by the side panel (0 when it is closed or docked at the bottom on phones)
function panelCover() {
  const p = $("#panel");
  if (!p.classList.contains("open") || window.innerWidth < 760) return 0;
  return Math.min(0.6, Math.max(0, (window.innerWidth - p.getBoundingClientRect().left) / window.innerWidth));
}
// fraction of the canvas height taken by the time / layer controls at the bottom left
function hudCover() {
  const r = $("#hud").getBoundingClientRect();
  return Math.min(0.5, Math.max(0, (window.innerHeight - r.top) / window.innerHeight));
}
function focusSystem(id) {
  sys.focus = id;
  const tp = sys.focusPos();
  const f = id === "sun" ? 1 : (sys.sizeK || 3) / 3;          // planets and Moon are drawn f times enlarged
  const r = f * (id === "sun" ? 8 : id === "moon" ? 0.4 : (sys.planets[id]?.r ?? 0.3) * 7);
  const dir = sys.camera.position.clone().sub(tp).normalize();
  sys.flyTo(tp.clone().add(dir.multiplyScalar(r)), tp, 1800);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
window.__obs = { S, sky, sys, switchView, focusSystem, loadSky, ctx, setLang, openLearn: (id) => openLearn(id), openObserve: (id) => openObserve(id), get observe() { return observe; }, journey, openJourney: (id) => openJourney(id) };          // for scripted captures (GIF / screenshots)

// ------------------------------------------------------------------------------------------------- panels
for (const b of document.querySelectorAll(".ptabs button")) b.onclick = () => openTab(b.dataset.tab);
function currentTab() { return document.querySelector(".ptabs button.on")?.dataset.tab; }
function openTab(tab) {
  for (const b of document.querySelectorAll(".ptabs button")) b.classList.toggle("on", b.dataset.tab === tab);
  for (const s of document.querySelectorAll(".pbody")) s.classList.toggle("on", s.id === "tab-" + tab);
  if (tab === "sats") renderSatsPanel();
  if (tab === "analyses" && !$("#tab-analyses").dataset.ready) renderAnalyses();
  if (tab === "engine" && !$("#tab-engine").dataset.ready) renderEngine();
  if (tab === "observe" && !$("#tab-observe").dataset.ready) { $("#tab-observe").dataset.ready = 1; observe.render($("#tab-observe")); }
  if (tab === "learn" && !$("#tab-learn").dataset.ready) { $("#tab-learn").dataset.ready = 1; renderLearn($("#tab-learn"), learnCtx); }
  $("#panel").classList.add("open");
}
// ------------------------------------------------------------------------------------------------- learn
function conName(key) {
  const c = sky.cat?.constellations?.find((x) => x.key === key || x.id === key);
  return c ? (c.names?.[lang] || c.name) : key;
}
function lookRaDec(ra, dec) {                    // J2000 RA/Dec -> look there in the sky view
  const M = sky.Mj2000; if (!M) return;
  const v = radecVec(ra, dec), p = [0, 1, 2].map((r) => M[r][0] * v[0] + M[r][1] * v[1] + M[r][2] * v[2]);
  const alt = Math.asin(p[1]) * R2D, az = ((Math.atan2(p[0], -p[2]) * R2D) + 360) % 360;
  if (S.view !== "sky") switchView("sky");
  sky.lookAt(alt, az); sky.fov = Math.min(sky.fov, 40);
}
const learnCtx = {
  api, qs, iso: () => isoOf(S.simJD), place: () => placeName(), conName,
  dfmt: (iso) => fmtDate(new Date(iso), { weekday: "short", hour: "2-digit", minute: "2-digit", timeZone: S.obs.tz }),
  canLook: (c) => !!(S.skyData?.now?.[c.id] || c.ra_deg !== undefined || c.hr || c.kind === "constellation"),
  look: (c) => {
    if (S.skyData?.now?.[c.id]) { const o = S.skyData.now[c.id]; if (S.view !== "sky") switchView("sky"); sky.lookAt(o.alt, o.az); return; }
    if (c.ra_deg !== undefined) return lookRaDec(c.ra_deg, c.dec_deg);
    if (c.hr && sky.cat) { const i = sky.cat.hr.indexOf(c.hr); if (i >= 0) lookRaDec(sky.cat.ra[i], sky.cat.dec[i]); return; }
    if (c.kind === "constellation" && sky.cat) { const k = c.id.slice(4), x = sky.cat.constellations.find((y) => y.key === k); if (x) lookRaDec(x.label[0], x.label[1]); }
  },
  openLearn: (id) => openLearn(id),
  observe: (c) => openObserve(c.observe || c.id),
  journey: (c) => openJourney(c.observe || c.id),
};
const observe = new Observe({ api, qs, iso: () => isoOf(S.simJD), place: () => placeName(), sky, S, bname: (b) => bname(b), conName,
  tfmt: (iso) => fmtDate(new Date(iso), { hour: "2-digit", minute: "2-digit", timeZone: S.obs.tz }),
  targetName: (info) => (info.kind === "body" ? bname(info.id) : observe.names?.[info.id] || info.id) });
function openObserve(id) {                       // e.g. from a Learn card: plan this target
  observe.target = id;
  $("#tab-observe").dataset.ready = 1;
  observe.render($("#tab-observe"));
  openTab("observe");
}
// ---------------------------------------------------------------------------------------------- journey
const JOURNEY_BODIES = ["moon", "sun", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"];
let journeyStars = null;
function journeyName(id) {
  if (JOURNEY_BODIES.includes(id) || id === "earth") return bname(id);
  const hr = +String(id).replace(/^star-/, ""), i = sky.cat ? sky.cat.hr.indexOf(hr) : -1;
  return i >= 0 ? sky.cat.name[i] : String(id);
}
const fmtKm = (km) => (km < 2e6 ? `${num(km, 0)} km` : km < 0.05 * 9.4607e12 ? `${num(km / 149597870.7, km < 1.5e9 ? 3 : 1)} ${t("unit.au")}` : `${num(km / 9.4607304725808e12, km < 9.46e13 ? 3 : 1)} ${t("unit.ly")}`);
function fmtDur(s) {
  const a = Math.abs(s);
  if (a < 120) return `${num(s, a < 10 ? 2 : 1)} s`;
  if (a < 7200) return `${num(s / 60, 1)} ${t("unit.min")}`;
  if (a < 3 * 86400) return `${num(s / 3600, 1)} ${t("unit.h")}`;
  if (a < 2 * 365.25 * 86400) return `${num(s / 86400, 0)} ${t("unit.f.d")}`;
  return `${num(s / 3.15576e7, s < 3.15576e9 ? 1 : 0)} ${t("unit.yr")}`;
}
async function openJourney(id) {
  const box = $("#journey");
  if (!journey.sc) journey.setScene(await api(`/api/journey/scene?t=${isoOf(S.simJD)}`), sky.cat);
  if (!journeyStars) journeyStars = (await api(`/api/facts/list?kind=star&lang=${lang}`)).cards;
  box.innerHTML = `<div class="jrow"><select id="jTarget">${JOURNEY_BODIES.map((b) => `<option value="${b}">${bname(b)}</option>`).join("")}
      <optgroup label="${esc(t("learn.group.star"))}">${journeyStars.map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join("")}</optgroup></select>
    <button class="primary" id="jGo">${t("journey.go")}</button><label class="small">${t("journey.speed")} <input id="jRate" type="range" min="-1" max="1.6" step="0.05" value="${Math.log10(journey.rate)}"></label>
    <button class="ghost" id="jSkip">${t("journey.skip")}</button><button class="ghost" id="jBack">${t("journey.lookBack")}</button></div><div id="jHud" class="small"></div><div id="jMarker" class="jmarker"></div>`;
  $("#jTarget").value = id;
  $("#jGo").onclick = () => startJourney($("#jTarget").value);
  $("#jRate").oninput = (e) => { journey.rate = Math.pow(10, +e.target.value); };
  $("#jSkip").onclick = () => journey.skip();
  $("#jBack").onclick = () => { journey.lookBack = !journey.lookBack; $("#jBack").classList.toggle("on", journey.lookBack); };
  if (S.view !== "journey") await switchView("journey");
  await startJourney(id);
}
async function startJourney(id) {
  let p;
  try { p = await api(`/api/journey/plan?target=${encodeURIComponent(id)}&t=${isoOf(S.simJD)}&${qs()}`); } catch (e) { toast(e.message); return; }
  p.target.id = p.target.id || p.target.body;
  journey.start(p);
  $("#jMarker").textContent = "";
}
journey.onMarker = (k) => { const el = $("#jMarker"); if (el) { el.textContent = t("journey.passed." + k); el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); } };
journey.onArrive = () => { const p = journey.plan; openLearn(p.target.body || p.target.id); };
function journeyHud() {
  const c = journey.counters(), p = journey.plan, el = $("#jHud");
  if (!c || !el) return;
  const tt = p.travel_times_s;
  const sunLine = p.target.kind === "star" && c.arrived ? `<br>${t("journey.sunBehind", { mag: num(p.sun_seen_from_target_mag, 2) })}` : "";
  const html = `<b>${esc(journeyName(p.target.id))}</b> · ${c.arrived ? t("journey.arrived") : t("journey.flying")}
    <div class="jbar"><i style="width:${(c.progress * 100).toFixed(1)}%"></i></div>
    ${t("journey.travelled", { d: fmtKm(c.travelled) })} · ${t("journey.left", { d: fmtKm(c.left) })} · ${t("journey.lightLeft", { tm: fmtDur(c.light_left_s) })}<br>
    ${t("journey.target", { mag: num(c.magnitude, 1), size: c.ang_deg >= 1 / 60 ? num(c.ang_deg, c.ang_deg < 1 ? 2 : 0) + "°" : num(c.ang_deg * 3600, c.ang_deg * 3600 < 1 ? 4 : 1) + "″" })}<br>
    ${t("journey.tripTimes", { total: fmtKm(p.distance_km), light: fmtDur(tt.light), voyager: fmtDur(tt.voyager1), nh: fmtDur(tt.new_horizons), plane: fmtDur(tt.airliner), walk: fmtDur(tt.walking) })}${sunLine}`;
  if (el._h !== html) { el.innerHTML = html; el._h = html; }
  el.dataset.arrived = c.arrived ? "1" : "0";
}
function openLearn(id) {
  $("#tab-learn").dataset.ready = 1;
  renderLearn($("#tab-learn"), learnCtx, id);
  openTab("learn");
}
// the object tab ends with the start of the object's info card and a link to the full card
const TEASERS = new Map();                       // cards already fetched, so redraws of the object tab are instant
async function learnTeaser(hit) {
  const id = hit.kind === "body" ? hit.id : hit.kind === "star" ? `star:${hit.id}` : hit.kind === "sat" ? `sat:${hit.sat.s.norad}` : null;
  const box = $("#oLearn");
  if (!id || !box) return;
  const k = id + "|" + lang;
  if (!TEASERS.has(k)) TEASERS.set(k, await api(`/api/facts/card?id=${encodeURIComponent(id)}&lang=${lang}`).catch(() => null));
  const c = TEASERS.get(k);
  if (!c || !box.isConnected) return;
  box.innerHTML = `<div class="card learn-teaser"><b>${esc(c.name)}</b> <span class="badge neutral">${esc(c.kind_label)}</span><p>${esc(c.summary)}</p>
    <p class="small">${esc(c.curiosities[0]?.text || "")}</p><button class="primary" id="oLearnGo">${t("learn.openCard")}</button></div>`;
  box.querySelector("#oLearnGo").onclick = () => openLearn(c.id);
}
const BODY_KEYS = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "earth"];
const bname = (b) => tOr("body." + b, b);
const tfmt = (iso) => { if (!iso) return "—"; return fmtDate(new Date(iso), { hour: "2-digit", minute: "2-digit", timeZone: S.obs.tz }); };
const dfmt = (iso) => fmtDate(new Date(iso), { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZone: S.obs.tz });
const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

function moonIcon(k, waxing, pa) {
  // illuminated fraction k, bright limb on the right when waxing (northern view), rotated by the position angle
  const r = 28, x = (2 * k - 1) * r;
  const sweep = waxing ? 1 : 0;
  const path = `M 32 ${32 - r} A ${r} ${r} 0 0 ${sweep} 32 ${32 + r} A ${Math.abs(x)} ${r} 0 0 ${k > 0.5 ? sweep : 1 - sweep} 32 ${32 - r} Z`;
  return `<svg class="moonicon" viewBox="0 0 64 64"><circle cx="32" cy="32" r="${r}" fill="#1b2130"/><g transform="rotate(${(90 - pa).toFixed(1)} 32 32)"><path d="${path}" fill="#e9e6da"/></g></svg>`;
}

function renderTonight() {
  const d = S.skyData; if (!d) return;
  const now = d.now, alm = d.almanac, m = d.moon;
  const tw = alm.twilight;
  const visible = Object.values(now).filter((b) => b.name !== "sun" && b.alt > 0).sort((a, b) => (a.magnitude ?? 9) - (b.magnitude ?? 9));
  let h = `<h3>${t("tonight.skyNow")}</h3><div class="kv"><span>${t("tonight.sky")}</span><span>${esc(tOr("skystate." + slug(d.sky_state), d.sky_state))}</span><span>${t("tonight.sunAlt")}</span><span>${num(d.sun_alt, 1)}°</span>
    <span>${t("tonight.lst")}</span><span>${num(d.lst_hours, 3)} ${t("unit.h")}</span><span>${t("tonight.observer")}</span><span>${num(d.observer.lat, 3)}°, ${num(d.observer.lon, 3)}°, ${num(d.observer.elevation_m, 0)} m</span></div>`;
  h += `<h3>${t("body.moon")}</h3><div class="card">${moonIcon(m.illuminated_fraction, m.phase_longitude_deg < 180, m.bright_limb_pa_deg)}
    <b>${esc(tOr("moonphase." + slug(m.name), m.name))}</b><br><span class="muted">${t("tonight.moonLine", { lit: num(m.illuminated_fraction * 100, 1), age: num(m.age_days, 1), dist: num(m.distance_km, 0) })}</span>
    <br><span class="small muted">${t("tonight.rts", { rise: tfmt(alm.rise_transit_set.moon.rise[0]), transit: tfmt(alm.rise_transit_set.moon.transit[0]), set: tfmt(alm.rise_transit_set.moon.set[0]) })}</span><div style="clear:both"></div></div>`;
  h += `<h3>${t("tonight.twilight")}</h3><table><tr><th></th><th class="r">${t("tonight.morning")}</th><th class="r">${t("tonight.evening")}</th></tr>` +
    ["sun", "civil", "nautical", "astronomical"]
      .map((k) => `<tr><td>${t("twilight." + k)}</td><td class="r">${tfmt(tw[k].rising[0])}</td><td class="r">${tfmt(tw[k].setting[0])}</td></tr>`).join("") + "</table>";
  h += `<h3>${t("tonight.visible")}</h3>` + (visible.length ? `<table><tr><th>${t("tonight.object")}</th><th class="r">${t("tonight.alt")}</th><th class="r">${t("tonight.az")}</th><th class="r">${t("tonight.mag")}</th></tr>` +
    visible.map((b) => `<tr data-b="${b.name}" style="cursor:pointer"><td>${bname(b.name)}</td><td class="r">${num(b.alt, 1)}°</td><td class="r">${num(b.az, 0)}° ${compass(b.az)}</td><td class="r">${num(b.magnitude, 1)}</td></tr>`).join("") + "</table>" : `<p class="muted">${t("tonight.noPlanet")}</p>`);
  h += `<h3>${t("tonight.rtsTitle")}</h3><table><tr><th></th><th class="r">${t("tonight.rise")}</th><th class="r">${t("tonight.transit")}</th><th class="r">${t("tonight.set")}</th></tr>` +
    Object.entries(alm.rise_transit_set).map(([b, r]) => `<tr><td>${bname(b)}</td><td class="r">${tfmt(r.rise[0])}</td><td class="r">${tfmt(r.transit[0])}</td><td class="r">${tfmt(r.set[0])}</td></tr>`).join("") + "</table>";
  h += `<p class="small muted">${t("tonight.note", { tz: esc(S.obs.tz || t("tonight.localTz")), day: dfmt(alm.window_start) })} ${t("method.sky")}.</p>`;
  $("#tab-tonight").innerHTML = h;
  for (const tr of $("#tab-tonight").querySelectorAll("tr[data-b]")) tr.onclick = () => { const b = tr.dataset.b; const o = d.now[b]; sky.lookAt(o.alt, o.az); S.selected = { kind: "body", id: b }; renderObject(S.selected); openTab("object"); };
}

const fmtDist = (o) => (o.distance_au > 0.1 ? num(o.distance_au, 5) + " " + t("unit.au") : num(o.distance_km, 0) + " km");
const fmtLight = (s) => (s > 120 ? num(s / 60, 2) + " " + t("unit.min") : num(s, 3) + " s");
async function renderObject(hit) {
  const el = $("#tab-object");
  const d = S.skyData;
  if (hit.kind === "body") {
    const b = hit.id;
    if (b === "earth") { el.innerHTML = `<h3>${t("body.earth")}</h3><p>${t("object.earth", { lat: num(S.obs.lat, 3), lon: num(S.obs.lon, 3) })}</p>` + elementsHtml(S.sysData?.elements?.earth); return; }
    const o = d?.now?.[b];
    if (!o) { el.innerHTML = `<h3>${bname(b)}</h3>`; return; }
    const r = d.almanac.rise_transit_set[b];
    let h = `<h3>${bname(b)} <span class="badge ${o.alt > 0 ? "pass" : "neutral"}">${o.alt > 0 ? t("object.above") : t("object.below")}</span></h3>
    <div class="kv"><span>${t("object.altRefr")}</span><span>${num(o.alt_apparent, 3)}°</span><span>${t("object.azimuth")}</span><span>${num(o.az, 3)}° ${compass(o.az)}</span>
    <span>${t("object.radecDate")}</span><span>${fmtRA(o.ra)} ${fmtDec(o.dec)}</span><span>${t("object.radecJ2000")}</span><span>${fmtRA(o.ra_j2000)} ${fmtDec(o.dec_j2000)}</span>
    <span>${t("object.hourAngle")}</span><span>${num(o.hour_angle / 15, 3)} ${t("unit.h")}</span>
    <span>${t("object.distance")}</span><span>${fmtDist(o)}</span>
    <span>${t("object.lightTime")}</span><span>${fmtLight(o.light_time_s)}</span>
    <span>${t("object.rangeRate")}</span><span>${num(o.range_rate_km_s, 3)} km/s</span>
    <span>${t("object.diameter")}</span><span>${o.angular_diameter_arcsec > 120 ? num(o.angular_diameter_arcsec / 60, 2) + "′" : num(o.angular_diameter_arcsec, 2) + "″"}</span>
    <span>${t("object.magnitude")}</span><span>${num(o.magnitude, 2)}</span>`;
    if (o.airmass) h += `<span>${t("object.airmass")}</span><span>${num(o.airmass, 3)} · ${num(o.extinction_mag, 2)} ${t("unit.mag")}</span>`;
    if (o.elongation !== undefined) h += `<span>${t("object.elongation")}</span><span>${num(o.elongation, 2)}°</span><span>${t("object.phase")}</span><span>${num(o.phase_angle, 1)}° · ${num(o.illuminated * 100, 1)}%</span>`;
    h += `<span>${t("object.rts")}</span><span>${tfmt(r.rise[0])} · ${tfmt(r.transit[0])} · ${tfmt(r.set[0])}</span></div>`;
    h += `<div class="row"><button class="primary" id="oLook">${t("object.look")}</button>${b !== "sun" ? `<button class="ghost" id="oOrbit">${t("object.orbit")}</button><button class="ghost" id="oPlan">${t("object.plan")}</button>` : ""}</div><div id="oMore"></div>`;
    el.innerHTML = h;
    $("#oLook").onclick = () => { if (S.view === "sky") { sky.lookAt(o.alt, o.az); sky.fov = Math.min(sky.fov, b === "moon" || b === "sun" ? 5 : 30); } else focusSystem(b); };
    if ($("#oOrbit")) $("#oOrbit").onclick = () => orbitInto($("#oMore"), b);
    if ($("#oPlan")) $("#oPlan").onclick = () => planInto($("#oMore"), b);
    el.insertAdjacentHTML("beforeend", `<div id="oLearn"></div>`); learnTeaser(hit);
  } else if (hit.kind === "star") {
    el.innerHTML = `<p class="muted">${t("object.loadingStar", { hr: hit.id })}</p>`;
    try {
      const s = await api(`/api/analysis/star?hr=${hit.id}&t=${isoOf(S.simJD)}&${qs()}`);
      const sw = s.rgb ? `<span class="swatch" style="color:rgb(${s.rgb.map((x) => Math.round(x * 255)).join(",")});background:currentColor"></span>` : "";
      const dist = s.distance_ly ? `${num(s.distance_ly, 1)} ${t("unit.ly")} · ${num(s.distance_pc, 2)} pc` : "—";
      el.innerHTML = `<h3>${sw}${esc(s.name)} <span class="muted">${esc(s.designation)} · HR ${s.hr}</span></h3>
      <div class="kv"><span>${t("object.altAz")}</span><span>${num(s.alt, 2)}° · ${num(s.az, 1)}° ${compass(s.az)}</span>
      <span>${t("object.radecDate")}</span><span>${fmtRA(s.ra_date)} ${fmtDec(s.dec_date)}</span>
      <span>${t("object.vBv")}</span><span>${num(s.v_mag, 2)} · ${num(s.b_v, 2)}</span><span>${t("object.spType")}</span><span>${esc(s.spectral_type)}</span>
      <span>${t("object.parallax")}</span><span>${s.parallax_mas ? num(s.parallax_mas, 1) + " mas" + (s.parallax_dynamical ? " " + t("object.dynamical") : "") : "—"}</span>
      <span>${t("object.distance")}</span><span>${dist}</span>
      <span>${t("object.lightLeft")}</span><span>${s.distance_ly ? Math.round(new Date().getFullYear() - s.distance_ly) : "—"}</span>
      <span>${t("object.absMag")}</span><span>${num(s.abs_mag_v, 2)}</span><span>${t("object.teff")}</span><span>${s.teff_k ? Math.round(s.teff_k) + " K" : "—"}</span>
      <span>${t("object.luminosity")}</span><span>${s.luminosity_lsun ? num(s.luminosity_lsun, s.luminosity_lsun < 10 ? 2 : 0) + " L☉" : "—"}</span>
      <span>${t("object.radius")}</span><span>${s.radius_rsun ? num(s.radius_rsun, s.radius_rsun < 10 ? 2 : 0) + " R☉" : "—"}</span>
      <span>${t("object.rv")}</span><span>${s.radial_velocity_km_s !== null ? num(s.radial_velocity_km_s, 0) + " km/s" : "—"}</span></div>
      <p class="small muted">${t("method.star")}. ${t("object.bscNote")}</p>
      <h3>${t("object.hr")}</h3><div id="oHR"></div><p class="small muted">${t("object.hrNote")}</p>`;
      drawHR($("#oHR"), hit.id);
      el.insertAdjacentHTML("beforeend", `<div id="oLearn"></div>`); learnTeaser(hit);
    } catch (e) { el.innerHTML = `<p>${esc(e.message)}</p>`; }
  } else if (hit.kind === "sat") {
    const o = hit.sat;
    el.innerHTML = `<h3>${esc(o.s.name)} <span class="muted">NORAD ${o.s.norad}</span></h3><div class="kv">
      <span>${t("object.altAz")}</span><span>${num(o.alt, 2)}° · ${num(o.az, 1)}°</span><span>${t("object.sunlit")}</span><span>${o.lit ? t("object.yes") : t("object.inShadow")}</span>
      <span>${t("object.magnitude")}</span><span>${num(o.mag, 1)}</span><span>${t("object.tleAge")}</span><span>${num(o.s.tle_age_days, 2)} d</span></div><div id="oPass"></div><div id="oLearn"></div>`;
    learnTeaser(hit);
    const p = await api(`/api/passes?${qs()}&t=${isoOf(S.simJD)}&days=2&norad=${o.s.norad}`).catch(() => null);
    if (p) $("#oPass").innerHTML = passesTable(p.passes);
  }
}

function elementsHtml(el) {
  if (!el) return "";
  return `<h3>${el.center === "earth" ? t("elements.titleGeo") : t("elements.title")}</h3><div class="kv">
    <span>${t("elements.a")}</span><span>${el.center === "earth" ? num(el.a_km, 0) + " km" : num(el.a_au, 6) + " " + t("unit.au")}</span><span>${t("elements.e")}</span><span>${num(el.e, 6)}</span>
    <span>${t("elements.i")}</span><span>${num(el.i_deg, 4)}°</span><span>${t("elements.node")}</span><span>${num(el.Omega_deg, 3)}°</span>
    <span>${t("elements.peri")}</span><span>${num(el.omega_deg, 3)}°</span><span>${t("elements.M")}</span><span>${num(el.M_deg, 3)}°</span>
    <span>${t("elements.period")}</span><span>${el.period_days > 800 ? num(el.period_years, 3) + " " + t("unit.yr") : num(el.period_days, 3) + " d"}</span>
    ${el.center !== "earth" ? `<span>${t("elements.periAph")}</span><span>${num(el.perihelion_au, 4)} · ${num(el.aphelion_au, 4)} ${t("unit.au")}</span>` : ""}</div>`;
}
async function orbitInto(el, b) {
  el.innerHTML = `<p class="muted">${t("common.computing")}</p>`;
  try {
    const o = await api(`/api/analysis/orbit?body=${b}&t=${isoOf(S.simJD)}&${qs()}`);
    el.innerHTML = elementsHtml(o.elements) + `<h3>${t("orbit.title")}</h3><div class="kv"><span>${t("object.distance")}</span><span>${num(o.distance_au, 6)} ${t("unit.au")}</span>
      <span>${t("orbit.oneWay")}</span><span>${num(o.light_time_s / 60, 3)} ${t("unit.min")}</span><span>${t("orbit.rangeRate")}</span><span>${num(o.range_rate_km_s, 4)} km/s (${o.range_rate_km_s > 0 ? t("orbit.receding") : t("orbit.approaching")})</span>
      <span>${t("orbit.halpha")}</span><span>${num(o.doppler.observed_nm, 5)} nm</span><span>${t("orbit.shift")}</span><span>${num(o.doppler.shift_pm, 2)} pm (β = ${sig(o.doppler.beta, 4)})</span></div><p class="small muted">${t("method.orbit")}</p>`;
  } catch (e) { el.textContent = e.message; }
}
async function planInto(el, name) {
  el.innerHTML = `<p class="muted">${t("common.computing")}</p>`;
  let p;
  try { p = await api(`/api/analysis/target?name=${encodeURIComponent(name)}&t=${isoOf(S.simJD)}&${qs()}`); } catch (e) { el.textContent = e.message; return; }
  const w = p.best_window;
  el.innerHTML = `<h3>${t("plan.title")}</h3><div id="pChart"></div>` + (w ? `<div class="card">${t("plan.best", { start: dfmt(w.start), end: tfmt(w.end), hours: num(w.hours, 1), min: p.min_alt })}<br>
    ${t("plan.highest", { when: tfmt(w.best), alt: num(w.max_alt, 1), X: num(w.airmass, 2), sep: num(w.moon_separation, 0), malt: num(w.moon_alt, 0) })}</div>`
    : `<p class="muted">${t("plan.none", { min: p.min_alt })}</p>`);
  const x = p.jd.map((j) => jdToDate(j).toISOString());
  PChart.mount($("#pChart"), { x, height: 200, yLabel: t("unit.deg"), yMin: -30, yMax: 90,
    series: [{ name: bname(name), y: p.alt, color: "#7fb4ff", width: 2 }, { name: t("body.sun"), y: p.sun_alt, color: "#ffcf6e" }, { name: t("body.moon"), y: p.moon_alt, color: "#c8c8d8", dash: "4 3" }],
    hlines: [{ y: 0, color: "#55607a", label: t("plan.horizon") }, { y: -18, color: "#3a4a66", label: t("plan.astroDark"), dash: "2 3" }] });
}

let HR = null;
async function drawHR(el, hrSel) {
  HR = HR || await api("/api/analysis/hr");
  const W = el.clientWidth || 360, H = 300, L = 36, B = 26;
  const M = sky.Mj2000, cat = sky.cat;
  const idx = new Map(cat.hr.map((h, i) => [h, i]));
  const X = (bv) => L + (bv + 0.4) / 2.4 * (W - L - 6), Y = (m) => 6 + (m + 8) / 24 * (H - 6 - B);
  let dots = "";
  for (let k = 0; k < HR.hr.length; k++) {
    const i = idx.get(HR.hr[k]); let up = false;
    if (i !== undefined && M) { const s = radecVec(cat.ra[i], cat.dec[i]); up = (M[1][0] * s[0] + M[1][1] * s[1] + M[1][2] * s[2]) > 0; }
    const c = HR.hr[k] === hrSel ? "#ffcf6e" : up ? "#ffffff" : "#3d5277";
    const r = HR.hr[k] === hrSel ? 5 : up ? 1.8 : 1.3;
    dots += `<circle cx="${X(HR.bv[k]).toFixed(1)}" cy="${Y(HR.abs_mag[k]).toFixed(1)}" r="${r}" fill="${c}" ${HR.hr[k] === hrSel ? 'stroke="#000"' : ""}/>`;
  }
  let ax = "";
  for (const bv of [-0.4, 0, 0.5, 1, 1.5, 2]) ax += `<text x="${X(bv)}" y="${H - 8}" text-anchor="middle">${num(bv, bv % 1 ? 1 : 0)}</text>`;
  for (const m of [-6, -3, 0, 3, 6, 9, 12, 15]) ax += `<text x="${L - 4}" y="${Y(m) + 3}" text-anchor="end">${m}</text>`;
  el.innerHTML = `<svg width="${W}" height="${H}" style="font:10px system-ui;fill:#8d9bb3"><defs><linearGradient id="hrg" x1="0" x2="1"><stop offset="0" stop-color="#9bb0ff" stop-opacity=".12"/><stop offset="1" stop-color="#ff7b3a" stop-opacity=".12"/></linearGradient></defs>
    <rect x="${L}" y="6" width="${W - L - 6}" height="${H - 6 - B}" fill="url(#hrg)"/>${dots}${ax}
    <text x="${W / 2}" y="${H}" text-anchor="middle">${t("object.hrAxis")}</text><text x="10" y="${H / 2}" transform="rotate(-90 10 ${H / 2})" text-anchor="middle">M_V</text></svg>`;
}

function passesTable(ps) {
  if (!ps || !ps.length) return `<p class="muted">${t("sats.none")}</p>`;
  return `<table><tr><th>${t("sats.satellite")}</th><th>${t("tonight.rise")}</th><th class="r">${t("sats.max")}</th><th>${t("tonight.set")}</th><th class="r">${t("tonight.mag")}</th></tr>` + ps.map((p) =>
    `<tr${p.visible ? ' style="color:#d8fff0"' : ' class="muted"'}><td>${esc(p.satellite.replace(" (ZARYA)", ""))}</td><td>${dfmt(p.rise_utc)} ${compass(p.rise_az ?? 0)}</td>
     <td class="r">${num(p.max_alt, 0)}° ${compass(p.culmination_az)}</td><td>${tfmt(p.set_utc)} ${compass(p.set_az ?? 0)}</td><td class="r">${p.visible ? num(p.brightest_mag, 1) : "—"}</td></tr>`).join("") +
    `</table><p class="small muted">${t("sats.note")}</p>`;
}
async function renderSatsPanel() {
  const el = $("#tab-sats");
  el.innerHTML = `<p class="muted">${t("sats.computing")}</p>`;
  try {
    const p = await api(`/api/passes?${qs()}&t=${isoOf(S.simJD)}&days=3`);
    const src = p.source || {};
    const err = src.groups ? Object.values(src.groups).find((g) => g.error) : null;
    let h = `<h3>${t("sats.title")}</h3>` + passesTable(p.passes);
    if (!p.passes.length && err) h = `<h3>${t("tab.sats")}</h3><div class="card">${t("sats.noTle", { err: esc(err.error) })}</div>`;
    h += `<p class="small muted">${t("method.passes")} ${t("sats.tleSource")}</p><div id="decayBox"></div>`;
    el.innerHTML = h;
    renderDecay($("#decayBox"), ctx);
  } catch (e) { el.innerHTML = `<p>${esc(e.message)}</p>`; }
}

// --------------------------------------------------------------------------------------------- analyses
function eventName(e) {
  if (e.kind === "planet") return t("ev.planet." + slug(e.type), { body: bname(e.body) });
  return tOr("ev." + slug(e.event), e.event);
}
function renderAnalyses() {
  const el = $("#tab-analyses"); el.dataset.ready = 1;
  const opts = BODY_KEYS.filter((k) => k !== "earth").map((k) => `<option value="${k}">${bname(k)}</option>`).join("");
  const starName = (hr) => { const i = sky.cat ? sky.cat.hr.indexOf(hr) : -1; return i >= 0 ? sky.cat.name[i] : "HR " + hr; };
  const stars = [2491, 7001, 5340, 2061, 1713, 6134].map((hr) => [hr, starName(hr)]);   // Sirius, Vega, Arcturus, ...
  el.innerHTML = `
  <h3>${t("an.sep")}</h3><div class="row"><input id="aA" type="text" value="moon" list="blist"> ↔ <input id="aB" type="text" value="jupiter" list="blist"><button class="primary" id="aSep">${t("an.measure")}</button></div>
  <datalist id="blist">${opts}${stars.map(([hr, n]) => `<option value="HR ${hr}">${n}</option>`).join("")}</datalist><div id="aSepOut" class="small"></div>
  <h3>${t("an.window")}</h3><div class="row"><select id="aTgt">${opts}</select><button class="primary" id="aPlan">${t("an.plan")}</button></div><div id="aPlanOut"></div>
  <h3>${t("an.orbit")}</h3><div class="row"><select id="aOrb">${opts.replace(/<option value="sun">[^<]*<\/option>/, "")}</select><button class="primary" id="aOrbit">${t("an.compute")}</button></div><div id="aOrbitOut"></div>
  <h3>${t("an.bb")}</h3><div class="row"><input id="aT" type="range" min="2500" max="30000" value="5772" step="10" style="flex:1"><span id="aTv" class="pill">5772 K</span></div><div id="aBB"></div>
  <h3>${t("an.events")}</h3><div class="row"><select id="aDays"><option value="90">${t("an.next3")}</option><option value="365" selected>${t("an.next12")}</option><option value="730">${t("an.next24")}</option></select><button class="primary" id="aEv">${t("an.list")}</button></div><div id="aEvOut"></div>
  <h3>${t("an.sw")}</h3><div id="aSW"><button class="ghost" id="aSWgo">${t("an.swGo")}</button></div>
  <h3>${t("an.wx")}</h3><div id="aWX"><button class="ghost" id="aWXgo">${t("an.wxGo")}</button></div>
  <h3>${t("an.sb")}</h3><div id="aSB"><button class="ghost" id="aSBgo">${t("an.sbGo")}</button></div>
  <h3>${t("an.ml")}</h3><div id="aML"><button class="ghost" id="aMLgo">${t("an.mlGo")}</button></div>
  <h3>${t("an.bh")}</h3><div id="aBH"></div>
  <h3>${t("an.lens")}</h3><p class="small muted">${t("an.lensNote")}</p>
  <div class="row"><label class="small">${t("an.camDist")} <input id="aRc" type="range" min="8" max="300" value="120"></label><select id="aDisc"><option value="auto">${t("an.discAuto")}</option><option value="off">${t("an.discOff")}</option><option value="on">${t("an.discOn")}</option></select><button class="primary" id="aLens">${t("an.renderHere")}</button></div><div id="aLensOut"></div>`;
  $("#aSWgo").onclick = () => renderSpaceWeather($("#aSW"), ctx);
  $("#aWXgo").onclick = () => renderConditions($("#aWX"), ctx);
  $("#aSBgo").onclick = () => renderSmallBodies($("#aSB"), ctx);
  $("#aMLgo").onclick = () => renderStarsML($("#aML"), ctx);
  renderBlackHole($("#aBH"), ctx);
  $("#aSep").onclick = async () => {
    try { const r = await api(`/api/analysis/separation?a=${encodeURIComponent($("#aA").value)}&b=${encodeURIComponent($("#aB").value)}&t=${isoOf(S.simJD)}&${qs()}`);
      $("#aSepOut").innerHTML = `<div class="card">${t("an.sepOut", { dms: esc(r.separation_dms), deg: num(r.separation_deg, 5), pa: num(r.position_angle_deg, 2) })}<br><span class="muted">${t("method.separation")}</span></div>`; }
    catch (e) { $("#aSepOut").textContent = e.message; }
  };
  $("#aPlan").onclick = () => planInto($("#aPlanOut"), $("#aTgt").value);
  $("#aOrbit").onclick = () => orbitInto($("#aOrbitOut"), $("#aOrb").value);
  const bb = async () => {
    const T = +$("#aT").value; $("#aTv").textContent = T + " K";
    const r = await api(`/api/analysis/blackbody?T=${T}`);
    const c = `rgb(${r.rgb.map((x) => Math.round(x * 255)).join(",")})`;
    const step = 5;
    PChart.mount($("#aBB"), { x: r.wavelength_nm.filter((_, i) => i % step === 0).map((x) => Math.round(x)), height: 170, yLabel: "B_λ", series: [{ name: `${T} K`, y: r.radiance.filter((_, i) => i % step === 0), color: c, width: 2 }] });
    $("#aBB").insertAdjacentHTML("beforeend", `<p class="small"><span class="swatch" style="color:${c};background:${c}"></span>${t("an.bbOut", { peak: num(r.peak_nm, 0), flux: sig(r.flux_w_m2, 4) })}</p>`);
  };
  $("#aT").onchange = bb; bb();
  $("#aEv").onclick = async () => {
    $("#aEvOut").innerHTML = `<p class="muted">${t("an.evSearching")}</p>`;
    try {
      const r = await api(`/api/analysis/events?t=${isoOf(S.simJD)}&days=${$("#aDays").value}&${qs()}`);
      const icon = { moon: "☾", season: "☀", planet: "♁", eclipse: "◐", transit: "●" };
      $("#aEvOut").innerHTML = `<table>${r.events.filter((e) => e.kind !== "moon" || /Full|New/.test(e.event)).map((e) =>
        `<tr${e.kind === "eclipse" || e.kind === "transit" ? ' style="color:#ffcf6e"' : ""}><td>${icon[e.kind] || ""}</td><td>${dfmt(e.utc)}</td><td>${esc(eventName(e))}${e.elongation_deg !== undefined && e.kind === "planet" ? ` <span class="muted">${num(e.elongation_deg, 1)}°, ${t("tonight.mag")} ${num(e.magnitude, 1)}</span>` : ""}${e.local ? `<br><span class="small">${t("an.evLocal", { when: dfmt(e.local.max_utc), pct: num(e.local.obscuration * 100, 0) })}${e.local.visible ? "" : " " + t("an.evSunDown")}</span>` : ""}</td></tr>`).join("")}</table><p class="small muted">${t("method.events")}</p>`;
    } catch (e) { $("#aEvOut").textContent = e.message; }
  };
  $("#aLens").onclick = async () => {
    const [ra, dec] = viewRaDec();
    $("#aLensOut").innerHTML = `<p class="muted">${t("an.lensTracing")}</p>`;
    const rc = $("#aRc").value;
    const url = `/api/analysis/lensing.png?ra=${ra.toFixed(3)}&dec=${dec.toFixed(3)}&fov=60&r_cam=${rc}&width=480&disc=${$("#aDisc").value}`;
    let res, info;
    try { [res, info] = await Promise.all([fetch(url), api("/api/analysis/lensing")]); if (!res.ok) throw new Error(res.statusText); }
    catch (e) { $("#aLensOut").textContent = t("err.render"); return; }
    const disc = res.headers.get("X-Disc") === "1", nb = res.headers.get("X-Bright-Stars"), bm = res.headers.get("X-Bright-Mag");
    const img = new Image(); img.className = "lens"; img.src = URL.createObjectURL(await res.blob());
    img.onload = () => {
      $("#aLensOut").innerHTML = ""; $("#aLensOut").appendChild(img);
      $("#aLensOut").insertAdjacentHTML("beforeend", `<p class="small">${t("an.lensOut", { ra: fmtRA(ra), dec: fmtDec(dec), rc, n: nb, mag: bm })}${disc ? " · " + t("an.lensDisc") : ""}.</p>
      <table><tr><th>${t("an.bhCol")}</th><th class="r">${t("an.shadowPred")}</th><th class="r">${t("an.ehtRing")}</th><th class="r">σ</th></tr>${info.eht.map((r) => `<tr><td>${esc(r.object)}</td><td class="r">${num(r.predicted_uas, 1)} µas</td><td class="r">${num(r.observed_uas, 1)} ± ${num(r.observed_err_uas, 1)}</td><td class="r">${num(r.sigma, 2)}</td></tr>`).join("")}</table>
      <p class="small muted">${t("an.sunDefl", { v: num(info.sun_deflection_arcsec, 3) })} ${t("method.lensing")}.</p>`);
    };
    img.onerror = () => { $("#aLensOut").textContent = t("err.render"); };
  };
}

// ----------------------------------------------------------------------------------------------- engine
async function renderEngine() {
  const el = $("#tab-engine"); el.dataset.ready = 1;
  const bodies = ["mars", "venus", "mercury", "jupiter", "saturn", "earth"];
  el.innerHTML = `<h3>${t("eng.title")}</h3>
  <p class="small">${t("eng.intro")}</p>
  <div class="row"><select id="eBody">${bodies.map((b) => `<option value="${b}">${bname(b)}</option>`).join("")}</select><button class="primary" id="eRun">${t("eng.run")}</button><span id="eStat" class="small muted"></span></div>
  <div id="eOut"></div>
  <h3>${t("eng.others")}</h3>
  <table class="small"><tr><th>${t("eng.model")}</th><th>${t("eng.checkedAgainst")}</th><th></th></tr>
  <tr><td>${t("eng.m.decay")}</td><td>${t("eng.m.decayRef")}</td><td><button class="ghost" data-go="sats">${t("tab.sats")}</button></td></tr>
  <tr><td>${t("eng.m.neo")}</td><td>${t("eng.m.neoRef")}</td><td><button class="ghost" data-go="aSBgo">${t("eng.open")}</button></td></tr>
  <tr><td>${t("eng.m.sb")}</td><td>${t("eng.m.sbRef")}</td><td><button class="ghost" data-go="aSBgo">${t("eng.open")}</button></td></tr>
  <tr><td>${t("eng.m.kp")}</td><td>${t("eng.m.kpRef")}</td><td><button class="ghost" data-go="aSWgo">${t("eng.open")}</button></td></tr>
  <tr><td>${t("eng.m.ml")}</td><td>${t("eng.m.mlRef")}</td><td><button class="ghost" data-go="aMLgo">${t("eng.open")}</button></td></tr>
  <tr><td>${t("eng.m.wx")}</td><td>${t("eng.m.wxRef")}</td><td><button class="ghost" data-go="aWXgo">${t("eng.open")}</button></td></tr></table>
  <h3>${t("eng.integrators")}</h3><div id="eInt"><p class="muted">${t("common.loading")}</p></div>
  <h3>${t("eng.validation")}</h3><div id="eVal"><p class="muted">${t("common.loading")}</p></div>`;
  $("#eRun").onclick = () => runSurrogate($("#eBody").value);
  for (const b of el.querySelectorAll("[data-go]")) b.onclick = () => {
    const g = b.dataset.go;
    if (g === "sats") return openTab("sats");
    openTab("analyses"); const btn = $("#" + g); if (btn) { btn.click(); btn.scrollIntoView({ block: "start" }); }
  };
  runSurrogate("mars"); loadIntegrators(); loadValidation();
}
async function runSurrogate(body) {
  $("#eStat").textContent = t("eng.training");
  try {
    const r = await api(`/api/engine/surrogate?body=${body}&t=${isoOf(S.simJD)}&days=365`);
    $("#eStat").textContent = t("eng.trained", { s: num(r.train_seconds, 1), n: r.model.params });
    const x = r.t_days.map((d) => Math.round(d));
    const h = `<div class="card">${verdictBadge(r.verdict)} &nbsp;${t("eng.summary", { km: num(r.summary.max_error_km_test, 0), arcsec: num(r.summary.max_angle_arcsec_test, 3), kep: num(r.summary.kepler_max_error_km, 0), cov: Math.round(r.summary.coverage * 100), level: Math.round((1 - r.summary.alpha) * 100), band: num(r.summary.band_km_median, 0) })}</div>
      <table><tr><th>${t("trust.check")}</th><th class="r">${t("trust.value")}</th><th class="r">${t("trust.limit")}</th><th></th></tr>${r.checks.map((c) => `<tr><td>${esc(checkName(c))}</td><td class="r">${sig(c.value)}</td><td class="r">${sig(c.threshold)}</td><td><span class="badge ${c.status}">${t("status." + c.status)}</span></td></tr>`).join("")}</table>
      <div id="eErr"></div><div id="eEn"></div><div id="eLoss"></div>
      <p class="small muted">${t("eng.reference", { method: t("method.surrogateRef"), e: sig(r.reference.energy_drift), l: sig(r.reference.angmom_drift) })}</p>`;
    $("#eOut").innerHTML = h;
    PChart.mount($("#eErr"), { x, height: 190, yLog: true, yLabel: "km", series: [
      { name: t("eng.s.pinn"), y: r.error_km.map((e) => Math.max(e, 1)), color: "#7fb4ff", width: 2 },
      { name: t("eng.s.kepler"), y: r.kepler_error_km.map((e) => Math.max(e, 1)), color: "#ff6b6b", dash: "4 3" },
      { name: t("eng.s.band"), y: r.band_km, color: "#3ccf8e", dash: "2 2" }] });
    const e0 = r.energy_ref[0];
    PChart.mount($("#eEn"), { x, height: 160, yLabel: "Δε/|ε|", series: [
      { name: t("eng.s.energyRef"), y: r.energy_ref.map((e) => (e - e0) / Math.abs(e0)), color: "#ffcf6e" },
      { name: t("eng.s.energyPinn"), y: r.energy_pinn.map((e) => (e - e0) / Math.abs(e0)), color: "#7fb4ff", dash: "4 3" }] });
    PChart.mount($("#eLoss"), { x: r.history.map((it) => it.iter), height: 150, yLog: true, yLabel: t("eng.loss"), series: [
      { name: t("eng.s.physics"), y: r.history.map((it) => it.physics), color: "#c58cff" }, { name: t("eng.s.anchors"), y: r.history.map((it) => it.data), color: "#3ccf8e" }] });
  } catch (e) { $("#eStat").textContent = e.message; }
}
async function loadIntegrators() {
  try {
    const r = await api("/api/engine/integrators?years=2&dt_hours=6");
    const x = r.t_days.map((d) => Math.round(d)), col = { leapfrog: "#ffcf6e", yoshida4: "#3ccf8e", rk4: "#ff6b6b" };
    $("#eInt").innerHTML = `<div id="eIntE"></div><div id="eIntP"></div><p class="small muted">${t("eng.intNote", { km: num(r.methods.yoshida4.error_vs_de421_km.mars.at(-1), 0) })}</p>`;
    PChart.mount($("#eIntE"), { x, height: 170, yLog: true, yLabel: "|ΔE/E|", series: Object.entries(r.methods).map(([m, d]) => ({ name: t("eng.s.evals", { m, n: d.force_evals }), y: d.rel_energy_error.map((v) => Math.max(Math.abs(v), 1e-16)), color: col[m] })) });
    PChart.mount($("#eIntP"), { x, height: 160, yLog: true, yLabel: "km", series: Object.entries(r.methods).map(([m, d]) => ({ name: t("eng.s.marsVsDe421", { m }), y: d.error_vs_de421_km.mars.map((v) => Math.max(v, 1e-3)), color: col[m] })) });
  } catch (e) { $("#eInt").textContent = e.message; }
}
async function loadValidation() {
  const r = await api("/api/validation").catch(() => null);
  if (!r || r.running || !r.rows) { $("#eVal").innerHTML = `<p class="muted">${t("eng.valRunning")}</p>`; return setTimeout(loadValidation, 4000); }
  $("#eVal").innerHTML = `<table><tr><th>${t("eng.method")}</th><th>${t("eng.ref")}</th><th class="r">${t("eng.result")}</th><th></th></tr>${r.rows.map((x) =>
    `<tr><td>${esc(tOr("val." + x.id, x.method))}</td><td class="small">${esc(tOr("valref." + x.id, x.reference))}</td><td class="r">${sig(x.value)} ${esc(tOr("unit." + x.unit, x.unit))}<br><span class="small muted">${t("eng.limitN", { lim: sig(x.threshold), n: x.n })}</span></td><td><span class="badge ${x.status}">${t("status." + x.status)}</span></td></tr>`).join("")}</table>
    <p class="small muted">${t("eng.valNote", { s: num(r.seconds, 1) })}</p>`;
}

// ------------------------------------------------------------------------------------------ language change
onLangChange(() => {
  setLangButtons(); renderLayers(); renderCities(); gyroLabel();
  renderTonight();
  if (S.selected) renderObject(S.selected);
  const tab = currentTab();
  const shown = $("#tab-learn .learn-card")?.dataset.card;
  for (const id of ["tab-analyses", "tab-engine", "tab-learn", "tab-observe"]) { $("#" + id).innerHTML = ""; delete $("#" + id).dataset.ready; }
  if (tab === "observe") { $("#tab-observe").dataset.ready = 1; observe.render($("#tab-observe")); }
  if (tab === "learn") openLearn(shown);
  if (tab === "sats") renderSatsPanel(); else $("#tab-sats").innerHTML = "";
  if (tab === "analyses") renderAnalyses();
  if (tab === "engine") renderEngine();
  if (!S.selected) $("#tab-object").innerHTML = `<p class="muted">${t("object.hint")}</p>`;
});

// ------------------------------------------------------------------------------------------------- start
applyStatic(); setLangButtons(); gyroLabel(); setViewButtons();
$("#tab-object").innerHTML = `<p class="muted">${t("object.hint")}</p>`;
(async () => {
  try {
    const cat = await api("/api/catalog");
    sky.setCatalog(cat); sys.setCatalog(cat);
  } catch (e) { toast(t("toast.catalogue", { msg: e.message })); }
  await loadSky(true);
  last = performance.now();
  requestAnimationFrame(frame);
  window.__ready = true;
  if (params.get("view") === "system") switchView("system").then(() => {
    if (params.get("focus")) setTimeout(() => focusSystem(params.get("focus")), +(params.get("focusDelay") || 7000));
  });
  if (params.get("tab")) openTab(params.get("tab"));
  if (params.get("observe")) openObserve(params.get("observe"));        // e.g. observe=saturn, observe=M31
  if (params.get("journey")) {                                          // e.g. journey=mars&jrate=1.3
    if (params.get("jrate")) journey.rate = Math.pow(10, +params.get("jrate"));
    openJourney(params.get("journey"));
  }
  if (params.get("learn")) openLearn(params.get("learn"));            // e.g. learn=M31, learn=star:2061, learn=saturn
  else if (params.get("tip") !== "0" && !params.has("capture") && !params.get("select") && !params.get("tab") && !params.get("observe") && !params.get("journey")) startupTip($("#tip"), learnCtx);
  if (params.get("select")) {                    // e.g. select=star:2491 or select=body:saturn
    const [kind, id] = params.get("select").split(":");
    S.selected = { kind, id: kind === "star" ? +id : id }; renderObject(S.selected); openTab("object");
  }
})();
