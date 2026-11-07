// Router, URL-hash state and the shell (theme, share, trust drawer, offline cache).
// The whole state of a mode lives in the hash: #/airfoil?s=0.17,0.15,...&a=4 is a shareable link.
import { setEngine, getEngine } from "./models.js";
import { openTrust } from "./trust.js";
import { h, toast } from "./ui.js";

const MODES = {
  airfoil: () => import("./modes/airfoil.js"),
  heatsink: () => import("./modes/heatsink.js"),
  rover: () => import("./modes/rover.js"),
  gallery: () => import("./modes/gallery.js"),
};

export function parseHash(hash = location.hash) {
  const m = hash.replace(/^#\/?/, "").split("?");
  const mode = MODES[m[0]] ? m[0] : "airfoil";
  return { mode, params: new URLSearchParams(m[1] || "") };
}

let current = null, currentMode = null, writing = false;

/** Modes call this when their state changes; replaces the hash without adding history entries. */
export function writeState(mode, params) {
  const p = new URLSearchParams(params);
  if (getEngine() === "ort") p.set("engine", "ort");
  const next = `#/${mode}${[...p].length ? `?${p}` : ""}`;
  if (next !== location.hash) { writing = true; history.replaceState(null, "", next); writing = false; }
}

async function route() {
  if (writing) return;
  const { mode, params } = parseHash();
  setEngine(params.get("engine"));
  document.querySelectorAll(".tabs a").forEach((a) => a.setAttribute("aria-selected", String(a.dataset.mode === mode)));
  const view = document.getElementById("view");
  if (mode === currentMode && current?.setState) { current.setState(params); return; }
  current?.unmount?.();
  currentMode = mode;
  view.innerHTML = "";
  view.append(h("div", { class: "loading" }, h("div", { class: "spinner" })));
  try {
    const mod = await MODES[mode]();
    if (currentMode !== mode) return;
    view.innerHTML = "";
    current = await mod.mount(view, params);
    document.title = `${mod.title} · PINNeAPPle Playground`;
    document.getElementById("engine-note").textContent = getEngine() === "ort" ? "Engine: onnxruntime-web." : "";
  } catch (e) {
    console.error(e);
    view.innerHTML = "";
    view.append(h("div", { class: "card error" }, h("b", {}, "Could not start this mode. "), String(e.message || e)));
  }
}

function theme() {
  const root = document.documentElement;
  let saved = null;
  try { saved = localStorage.getItem("pp-theme"); } catch { /* storage may be blocked */ }
  if (saved) root.dataset.theme = saved;
  document.getElementById("theme").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("pp-theme", root.dataset.theme); } catch { /* ignore */ }
    window.dispatchEvent(new Event("themechange"));
  });
}

function shell() {
  theme();
  document.getElementById("share").addEventListener("click", async () => {
    const url = location.href;
    try {
      if (navigator.share && matchMedia("(pointer: coarse)").matches) await navigator.share({ title: document.title, url });
      else { await navigator.clipboard.writeText(url); toast("Link copied: it opens exactly this state"); }
    } catch { toast(url); }
  });
  document.getElementById("trust-open").addEventListener("click", () => openTrust(true));
  document.getElementById("trust-close").addEventListener("click", () => openTrust(false));
  document.getElementById("scrim").addEventListener("click", () => openTrust(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") openTrust(false); });
  window.addEventListener("hashchange", route);
  if ("serviceWorker" in navigator && location.protocol !== "file:" && !new URLSearchParams(location.search).has("nosw")) {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
}

shell();
route();
