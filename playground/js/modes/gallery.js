// Geometry gallery: the pp.geometries library in 3-D (light glTF files), with an info card per geometry and the
// code that regenerates it. three.js (MIT, r169) is loaded only when this mode opens.
import { writeState } from "../main.js";
import { renderGeometryTrust } from "../trust.js";
import { h, sig } from "../ui.js";

export const title = "Geometry gallery";

export async function mount(root, params) {
  const idx = await fetch("data/geometries.json").then((r) => r.json());
  const items = idx.geometries;
  const st = { g: items[0].name, cat: "all" };
  const cats = ["all", ...new Set(items.map((g) => g.category))];

  const viewer = h("div", { class: "viewer", "data-testid": "viewer" }, h("div", { class: "vload" }, "Loading the 3-D viewer…"));
  const info = h("div", { class: "card", "data-testid": "geometry-info" });
  const grid = h("div", { class: "gallery", role: "list" });
  const filters = h("div", { class: "filters", role: "group", "aria-label": "Category" });
  for (const c of cats) filters.append(h("button", { class: "chip", type: "button", "data-cat": c, "aria-pressed": String(c === st.cat), onclick: () => { st.cat = c; renderGrid(); } }, c.replace(/_/g, " ")));
  for (const g of items) {
    grid.append(h("button", { class: "gcard", type: "button", role: "listitem", "data-name": g.name, "data-cat": g.category, onclick: () => select(g.name, true) },
      h("img", { src: `data/thumbs/${g.name}.jpg`, alt: g.title, loading: "lazy", width: 360, height: 240 }),
      h("div", { class: "gt" }, h("b", {}, g.title), h("span", {}, `${g.category.replace(/_/g, " ")} · ${(g.triangles / 1000).toFixed(1)}k triangles`))));
  }
  root.append(h("div", { class: "mode two" },
    h("div", { class: "col-left stack" }, info),
    h("div", { class: "col-center stack" }, h("div", { class: "card stage", style: "padding:0" }, viewer,
      h("div", { class: "note" }, "Drag to orbit, scroll or pinch to zoom. Light level of detail; the fine meshes are on the Hub.")),
      h("div", { class: "card" }, h("h3", {}, `The library: ${items.length} geometries`), filters, grid))));

  function renderGrid() {
    filters.querySelectorAll(".chip").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.cat === st.cat)));
    grid.querySelectorAll(".gcard").forEach((c) => { c.hidden = st.cat !== "all" && c.dataset.cat !== st.cat; c.setAttribute("aria-pressed", String(c.dataset.name === st.g)); });
  }

  function renderInfo(g) {
    const d = g.dimensions_m || {};
    info.innerHTML = "";
    info.append(h("h3", {}, g.category.replace(/_/g, " ")), h("h2", { style: "margin:0 0 6px;font-size:18px" }, g.title), h("p", { class: "hint", style: "color:var(--ink-2)" }, g.description),
      h("dl", { class: "kv", style: "margin-top:12px" },
        h("dt", {}, "Size L × W × H"), h("dd", {}, `${sig(d.length_x, 4)} × ${sig(d.width_y, 4)} × ${sig(d.height_z, 4)} m`),
        h("dt", {}, "Triangles (light)"), h("dd", {}, String(g.triangles)), h("dt", {}, "glTF file"), h("dd", {}, `${(g.glb_bytes / 1024).toFixed(0)} kB`)),
      h("p", { class: "hint" }, `Frame: ${g.frame}`),
      h("h3", { style: "margin-top:14px" }, "Physics it is built for"), h("ul", { style: "margin:0;padding-left:18px;font-size:13px;color:var(--ink-2)" }, g.physics.map((p) => h("li", {}, p))),
      h("h3", { style: "margin-top:14px" }, "Regenerate it"), h("pre", { class: "code" }, g.regenerate),
      h("p", { class: "hint" }, h("a", { href: g.code, target: "_blank", rel: "noopener" }, "Builder source"), " · ", h("a", { href: g.glb, download: `${g.name}.glb` }, "Download .glb"), " · ",
        h("a", { href: "#", onclick: (e) => { e.preventDefault(); document.getElementById("trust-open").click(); } }, "Where the dimensions come from →")));
    renderGeometryTrust(g);
  }

  // ------------------------------------------------------------------ three.js viewer
  let three = null, disposed = false;
  async function initThree() {
    const THREE = await import("three");
    const { OrbitControls } = await import("three/addons/OrbitControls.js");
    const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
    const { RoomEnvironment } = await import("three/addons/environments/RoomEnvironment.js");
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    const scene = new THREE.Scene();
    scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 0.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(3, 5, 4); scene.add(sun);
    const camera = new THREE.PerspectiveCamera(35, 1.6, 0.01, 1000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.autoRotate = !matchMedia("(prefers-reduced-motion: reduce)").matches; controls.autoRotateSpeed = 1.2;
    renderer.domElement.addEventListener("pointerdown", () => { controls.autoRotate = false; });
    viewer.innerHTML = ""; viewer.append(renderer.domElement);
    const resize = () => { const r = viewer.getBoundingClientRect(); renderer.setSize(r.width, r.height, false); camera.aspect = r.width / Math.max(r.height, 1); camera.updateProjectionMatrix(); };
    new ResizeObserver(resize).observe(viewer); resize();
    const loop = () => { if (disposed) return; controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); };
    loop();
    return { THREE, renderer, scene, camera, controls, loader: new GLTFLoader(), current: null };
  }

  let loadSeq = 0;
  async function show(g) {
    const my = ++loadSeq;
    try {
      three = three || await initThree();
    } catch (e) {
      viewer.innerHTML = ""; viewer.append(h("div", { class: "vload" }, h("img", { src: `data/thumbs/${g.name}.jpg`, alt: g.title, style: "max-width:100%" })));
      return;
    }
    const { THREE, scene, camera, controls, loader } = three;
    const note = h("div", { class: "vload", "data-testid": "viewer-loading" }, `Loading ${g.title}…`);
    viewer.append(note);
    const tryLoad = (url) => new Promise((res, rej) => loader.load(url, res, undefined, rej));
    let gltf;
    try { gltf = await tryLoad(g.glb); } catch { try { gltf = await tryLoad(g.glb_hub); } catch (e) { note.textContent = `Could not load the model (${e.message || e}).`; return; } }
    if (my !== loadSeq || disposed) return;
    note.remove();
    if (three.current) { scene.remove(three.current); three.current.traverse((o) => { o.geometry?.dispose(); }); }
    const obj = gltf.scene;
    const wrap = new THREE.Group(); wrap.add(obj); scene.add(wrap); three.current = wrap;
    const box = new THREE.Box3().setFromObject(wrap), c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
    wrap.position.sub(c);
    const r = Math.max(s.x, s.y, s.z);
    camera.near = r / 100; camera.far = r * 100; camera.updateProjectionMatrix();
    camera.position.set(r * 1.1, r * 0.55, r * 1.25);
    controls.target.set(0, 0, 0); controls.update();
    viewer.dataset.loaded = g.name;
  }

  function select(name, user = false) {
    const g = items.find((x) => x.name === name) || items[0];
    st.g = g.name;
    writeState("gallery", { g: st.g });
    renderGrid(); renderInfo(g); show(g);
    if (user && matchMedia("(max-width: 760px)").matches) viewer.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function setState(p) { if (p.get("g") !== st.g || !three) select(p.get("g") || st.g); }
  setState(params);
  window.__playground = { mode: "gallery", state: st };
  return { setState, unmount() { disposed = true; three?.renderer.dispose(); } };
}
