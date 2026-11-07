// Model loading for the modes: the library's web runtime (pinneapple_hub/zoo/web/zoo.js, served as lib/) does the
// work; this file only keeps one instance per model and the engine chosen in the URL (engine=ort).
import { loadAirfoilGeometry, loadModel } from "pinneapple/zoo.js";

const cache = new Map();
let engine = "builtin";

export function setEngine(e) { const next = e === "ort" ? "ort" : "builtin"; if (next !== engine) { engine = next; cache.clear(); } }
export const getEngine = () => engine;

const once = (key, make) => { if (!cache.has(key)) cache.set(key, make()); return cache.get(key); };
export const load = (name) => once(name, () => loadModel("models", name, { engine }));
export const airfoilGeometry = () => once("cst-geometry", () => loadAirfoilGeometry("models", { engine }));
