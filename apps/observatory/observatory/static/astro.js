// Small client-side helpers. Everything physical is computed on the server (pinneapple_physics.astro); the browser
// only interpolates samples, applies the sidereal rotation, and draws.
import { num, t } from "./i18n.js";

export const D2R = Math.PI / 180, R2D = 180 / Math.PI;

// alt/az (deg) -> three.js world: x = east, y = up, z = -north
export function altazVec(alt, az, r = 1) {
  const a = alt * D2R, z = az * D2R;
  return [r * Math.cos(a) * Math.sin(z), r * Math.sin(a), -r * Math.cos(a) * Math.cos(z)];
}
export function vecAltaz(x, y, z) {
  const r = Math.hypot(x, y, z);
  return [Math.asin(y / r) * R2D, ((Math.atan2(x, -z) * R2D) + 360) % 360];
}
// J2000 (ICRS) unit vector from RA/Dec in degrees
export function radecVec(ra, dec) {
  const a = ra * D2R, d = dec * D2R;
  return [Math.cos(d) * Math.cos(a), Math.cos(d) * Math.sin(a), Math.sin(d)];
}
// 3x3 row-major multiply
export function mul3(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}
export function rotZ(t) { const c = Math.cos(t), s = Math.sin(t); return [[c, s, 0], [-s, c, 0], [0, 0, 1]]; }
// true equator of date -> local horizon (three.js axes), for local sidereal time `lstDeg` and latitude `latDeg`
export function horizonMatrix(lstDeg, latDeg) {
  const p = latDeg * D2R, sp = Math.sin(p), cp = Math.cos(p);
  const L = [[0, 1, 0], [cp, 0, sp], [sp, 0, -cp]];
  return mul3(L, rotZ(lstDeg * D2R));
}
export function toMatrix4(THREE, M) {
  const m = new THREE.Matrix4();
  m.set(M[0][0], M[0][1], M[0][2], 0, M[1][0], M[1][1], M[1][2], 0, M[2][0], M[2][1], M[2][2], 0, 0, 0, 0, 1);
  return m;
}
// B-V -> Teff (Ballesteros 2012) -> sRGB (Tanner Helland fit, as in pinneapple_physics.blackhole.raytrace)
export function bvToTeff(bv) { return 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62)); }
export function teffToRGB(T) {
  const t = Math.min(Math.max(T, 1000), 40000) / 100;
  const r = t <= 66 ? 255 : 329.698727446 * Math.pow(t - 60, -0.1332047592);
  const g = t <= 66 ? 99.4708025861 * Math.log(t) - 161.1195681661 : 288.1221695283 * Math.pow(t - 60, -0.0755148492);
  const b = t >= 66 ? 255 : (t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307);
  return [r, g, b].map((v) => Math.min(Math.max(v, 0), 255) / 255);
}
// linear interpolation in a sample table (x ascending)
export function interp(xs, ys, x) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const f = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + f * (ys[hi] - ys[lo]);
}
// interpolate alt/az as unit vectors (avoids the 0/360 azimuth wrap)
export function interpAltaz(jd, alts, azs, x) {
  const n = jd.length;
  let lo = 0, hi = n - 1;
  if (x <= jd[0]) hi = 1; else if (x >= jd[n - 1]) lo = n - 2;
  else while (hi - lo > 1) { const m = (lo + hi) >> 1; if (jd[m] <= x) lo = m; else hi = m; }
  const f = Math.min(Math.max((x - jd[lo]) / (jd[hi] - jd[lo]), -0.5), 1.5);
  const a = altazVec(alts[lo], azs[lo]), b = altazVec(alts[hi], azs[hi]);
  const v = [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
  return vecAltaz(...v);
}
export function refractionDeg(h) { // Saemundsson, standard conditions (display only)
  if (h < -1) return 0;
  return 1.02 / Math.tan((h + 10.3 / (h + 5.11)) * D2R) / 60;
}
export function jdToDate(jd) { return new Date((jd - 2440587.5) * 86400000); }
export function dateToJd(d) { return d.getTime() / 86400000 + 2440587.5; }
export function fmtRA(deg) {
  const h = ((deg % 360) + 360) % 360 / 15, hh = Math.floor(h), m = (h - hh) * 60, mm = Math.floor(m), s = (m - mm) * 60;
  return `${hh}h ${String(mm).padStart(2, "0")}m ${(s < 10 ? "0" : "") + num(Math.floor(s * 10) / 10, 1)}s`;
}
export function fmtDec(deg) {
  const s = deg < 0 ? "−" : "+", a = Math.abs(deg), d = Math.floor(a), m = (a - d) * 60, mm = Math.floor(m), ss = (m - mm) * 60;
  return `${s}${d}° ${String(mm).padStart(2, "0")}′ ${ss.toFixed(0).padStart(2, "0")}″`;
}
// 16-point compass in the interface language (N, NNE, ... ; Portuguese uses L for east and O for west)
export function compass(az) { return t("compass.points").split(" ")[Math.round((((az % 360) + 360) % 360) / 22.5) % 16]; }
