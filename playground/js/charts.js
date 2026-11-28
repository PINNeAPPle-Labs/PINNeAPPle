// A small SVG line chart: one y-axis, thin lines, an uncertainty band, reference dots, the current point, the
// out-of-range zone shaded, and a crosshair tooltip. Redrawn in place on every update (cheap at these sizes).
import { h, svg, sig } from "./ui.js";

export function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
  const out = [];
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) out.push(Math.abs(k) < 1e-9 ? 0 : +(k * step).toPrecision(12));
  return out;
}

export class LineChart {
  constructor(parent, { width = 560, height = 250, xLabel = "", yLabel = "", testid } = {}) {
    this.W = width; this.H = height; this.m = { l: 52, r: 14, t: 14, b: 38 };
    this.xLabel = xLabel; this.yLabel = yLabel;
    this.svg = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "data-testid": testid });
    this.tip = h("div", { class: "tip" });
    this.legend = h("div", { class: "legend" });
    this.el = h("div", { class: "chart" }, this.svg, this.tip, this.legend);
    parent.append(this.el);
    this.svg.addEventListener("pointermove", (e) => this._hover(e));
    this.svg.addEventListener("pointerleave", () => { this.tip.classList.remove("on"); this.cross && this.cross.setAttribute("opacity", 0); });
  }

  /** opts: {series: [{name, x[], y[], lo?[], hi?[], color, dash?}], refs: [{name, x[], y[], color}],
   *        marker: {x, y}, xValid: [lo, hi], zones: [{x0, x1, status: "warn" | "fail"}], xDomain?, yDomain?, xFmt?,
   *        yFmt?, ariaLabel} */
  update(o) {
    this.o = o;
    const { W, H, m } = this, s = this.svg;
    s.innerHTML = "";
    s.setAttribute("aria-label", o.ariaLabel || `${this.yLabel} vs ${this.xLabel}`);
    const xs = [], ys = [];
    for (const se of o.series) { xs.push(...se.x); ys.push(...se.y, ...(se.lo || []), ...(se.hi || [])); }
    for (const r of o.refs || []) { xs.push(...r.x); ys.push(...r.y); }
    if (o.marker) ys.push(o.marker.y);
    const fin = (a) => a.filter(Number.isFinite);
    let [x0, x1] = o.xDomain || [Math.min(...fin(xs)), Math.max(...fin(xs))];
    let [y0, y1] = o.yDomain || [Math.min(...fin(ys)), Math.max(...fin(ys))];
    if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
    const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
    if (o.yZero && y0 > 0) y0 = 0;
    const xt = niceTicks(x0, x1, 6), yt = niceTicks(y0, y1, 5);
    y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
    const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r), Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    this.X = X; this.Y = Y; this.x0 = x0; this.x1 = x1;
    const g = svg("g");
    if (o.xValid) {
      const [a, b] = o.xValid;
      if (a > x0) g.append(svg("rect", { class: "oor-zone", x: X(x0), y: m.t, width: X(a) - X(x0), height: H - m.t - m.b }));
      if (b < x1) g.append(svg("rect", { class: "oor-zone", x: X(b), y: m.t, width: X(x1) - X(b), height: H - m.t - m.b }));
    }
    for (const z of o.zones || []) {                       // stretches where checks warn / fail: {x0, x1, status}
      const a = Math.max(z.x0, x0), b = Math.min(z.x1, x1);
      if (b > a) g.append(svg("rect", { class: `zone zone-${z.status}`, x: X(a), y: m.t, width: X(b) - X(a), height: H - m.t - m.b }));
    }
    for (const v of yt) if (v >= y0 && v <= y1) {
      g.append(svg("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
      g.append(svg("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, (o.yFmt || sig)(v)));
    }
    for (const v of xt) if (v >= x0 - 1e-9 && v <= x1 + 1e-9) g.append(svg("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, (o.xFmt || sig)(v)));
    g.append(svg("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    g.append(svg("text", { class: "title", x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, this.xLabel));
    g.append(svg("text", { class: "title", x: 12, y: (m.t + H - m.b) / 2, transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle" }, this.yLabel));
    const path = (x, y) => x.map((v, i) => (Number.isFinite(y[i]) ? `${i && Number.isFinite(y[i - 1]) ? "L" : "M"}${X(v).toFixed(1)},${Y(y[i]).toFixed(1)}` : "")).join("");
    for (const se of o.series) if (se.lo) {
      const d = path(se.x, se.hi) + se.x.slice().reverse().map((v, i) => `L${X(v).toFixed(1)},${Y(se.lo[se.x.length - 1 - i]).toFixed(1)}`).join("") + "Z";
      g.append(svg("path", { class: "band", d, fill: se.bandColor || "var(--band)" }));
    }
    for (const se of o.series) g.append(svg("path", { class: "line", d: path(se.x, se.y), stroke: se.color, "stroke-dasharray": se.dash || null }));
    for (const r of o.refs || []) r.x.forEach((v, i) => g.append(svg("circle", { class: "ref", cx: X(v), cy: Y(r.y[i]), r: 4.5, fill: r.color }, svg("title", {}, `${r.name}: ${sig(v)}, ${sig(r.y[i], 4)}`))));
    this.cross = svg("line", { class: "cross", y1: m.t, y2: H - m.b, opacity: 0 });
    g.append(this.cross);
    if (o.marker && Number.isFinite(o.marker.y)) {
      g.append(svg("circle", { class: "marker", cx: X(o.marker.x), cy: Y(o.marker.y), r: 6, fill: o.marker.color || o.series[0].color }));
    }
    s.append(g);
    this.legend.innerHTML = "";
    const items = [...o.series.filter((se) => se.name).map((se) => [se.name, se.color, se.lo ? "band" : "line"]), ...(o.refs || []).filter((r) => r.x.length).map((r) => [r.name, r.color, "dot"])];
    if (items.length > 1 || (o.refs || []).some((r) => r.x.length)) for (const [n, c, k] of items) this.legend.append(h("span", {}, h("i", { class: k === "dot" ? "dot" : "", style: `background:${c}` }), n + (k === "band" ? " (± 2σ band)" : "")));
  }

  _hover(e) {
    const o = this.o; if (!o || !o.series.length) return;
    const r = this.svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * this.W;
    const se = o.series[0];
    let best = 0;
    se.x.forEach((v, i) => { if (Math.abs(this.X(v) - px) < Math.abs(this.X(se.x[best]) - px)) best = i; });
    const xv = se.x[best];
    this.cross.setAttribute("x1", this.X(xv)); this.cross.setAttribute("x2", this.X(xv)); this.cross.setAttribute("opacity", 1);
    const rows = o.series.map((s2) => {
      const i = s2.x.indexOf(xv), y = s2.y[i];
      const pm = s2.lo && i >= 0 ? ` ± ${sig((s2.hi[i] - s2.lo[i]) / 4, 2)}` : "";
      return i >= 0 ? `${s2.name || this.yLabel}: <b>${(o.tipFmt || ((v) => sig(v, 4)))(y)}</b>${pm}` : "";
    }).filter(Boolean);
    this.tip.innerHTML = `${this.xLabel} ${(o.xFmt || sig)(xv)}<br>${rows.join("<br>")}`;
    const bx = (this.X(xv) / this.W) * r.width, by = (this.Y(se.y[best]) / this.H) * r.height;
    this.tip.style.left = `${Math.min(Math.max(bx, 70), r.width - 70)}px`; this.tip.style.top = `${by}px`;
    this.tip.classList.add("on");
  }
}
