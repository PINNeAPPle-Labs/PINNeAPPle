// Small DOM helpers shared by the modes.

export function h(tag, attrs = {}, ...kids) {
  const ns = tag.startsWith("svg:") ? "http://www.w3.org/2000/svg" : null;
  const e = ns ? document.createElementNS(ns, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
}
export const svg = (tag, attrs, ...kids) => h(`svg:${tag}`, attrs, ...kids);

export function fmt(v, d = 3) {
  if (v === undefined || v === null || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(d).replace(/^-0(\.0+)?$/, "0$1");
}
export const sig = (v, n = 3) => (Number.isFinite(v) ? (+v.toPrecision(n)).toString() : "–");
export const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

/** Range slider with the model's validity band drawn on the track; returns {el, set(v), get()}. */
export function slider({ label, unit = "", min, max, step, value, valid, digits = 2, onInput, testid }) {
  const out = h("output", { "data-testid": testid && `${testid}-value` });
  const input = h("input", { type: "range", min, max, step, value, "aria-label": label, "data-testid": testid });
  const band = h("span", { class: "valid", "aria-hidden": "true" });
  if (valid) {
    const l = ((valid[0] - min) / (max - min)) * 100, r = ((valid[1] - min) / (max - min)) * 100;
    band.style.left = `calc(${l}% + ${9 - (18 * l) / 100}px)`;
    band.style.width = `calc(${r - l}% - ${(18 * (r - l)) / 100}px)`;
  }
  const el = h("div", { class: "slider" }, h("div", { class: "lab" }, h("span", {}, label, unit ? h("span", { class: "u" }, ` ${unit}`) : null), out), h("div", { class: "track" }, band, input));
  const paint = (v) => {
    out.textContent = (+v).toFixed(digits);
    const bad = valid && (v < valid[0] - 1e-12 || v > valid[1] + 1e-12);
    el.classList.toggle("oor", !!bad); out.classList.toggle("oor", !!bad);
    out.title = bad ? `outside the validity range ${valid[0]}–${valid[1]}` : "";
  };
  input.addEventListener("input", () => { paint(+input.value); onInput && onInput(+input.value); });
  paint(value);
  return { el, input, set(v) { input.value = v; paint(+input.value); }, get: () => +input.value };
}

export function tile(key, { hero = false, testid } = {}) {
  const v = h("div", { class: "v", "data-testid": testid }), s = h("div", { class: "s" }), k = h("div", { class: "k" }, key);
  const el = h("div", { class: `tile${hero ? " hero" : ""}` }, k, v, s);
  return { el, set(value, sub = "", raw) { v.innerHTML = value; s.textContent = sub; if (raw !== undefined) v.dataset.raw = String(raw); } };
}

const ICONS = {
  pass: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  warn: '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17.5v.01"/>',
  fail: '<circle cx="12" cy="12" r="9"/><path d="M8.5 8.5l7 7M15.5 8.5l-7 7"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5v.01"/>',
};
export const icon = (k) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[k]}</svg>`;

/** Render a query evaluation of the library's runtime ({verdict, groups}: pinneapple_hub/zoo/web/zoo.js). */
export function renderVerdict(el, ev, testid = "verdict", kind = "trust") {
  const v = ev.verdict.status;
  el.innerHTML = "";
  if (kind === "trust") {
    el.append(h("div", { class: `verdict ${v}`, role: "status", "data-testid": testid, "data-verdict": v, html: icon(v) },
      h("div", {}, h("b", {}, ev.verdict.title), h("span", {}, ev.verdict.text))));
  }
  const ul = h("ul", { class: "checks" });
  for (const c of ev.groups.filter((g) => (g.kind || "trust") === kind)) {
    ul.append(h("li", { class: c.status, "data-check": c.id },
      h("span", { class: "ico", html: icon(c.status) }),
      h("span", {}, c.name, c.rule ? h("span", { class: "rule", title: "pinneapple check rule" }, c.rule) : null,
        c.detail_text ? h("span", { class: "det" }, c.detail_text) : null)));
  }
  el.append(ul);
  return v;
}

export function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.classList.add("on");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("on"), 1800);
}

/** Coalesce bursts of input events into one call per animation frame. */
export function rafThrottle(fn) {
  let queued = false;
  return () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; fn(); }); };
}

export function segmented(options, value, onChange) {
  const el = h("div", { class: "seg", role: "group" });
  const btns = options.map(([k, label]) => h("button", { type: "button", "aria-pressed": String(k === value), onclick: () => { btns.forEach((b) => b.setAttribute("aria-pressed", String(b === btn(k)))); onChange(k); } }, label));
  const btn = (k) => btns[options.findIndex((o) => o[0] === k)];
  el.append(...btns);
  return el;
}
