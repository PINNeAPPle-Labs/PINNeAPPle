// 3-D preview of a library geometry (its committed light glTF) with the three.js copy vendored in the library
// (/vendor) and the studio core's edges (/studio/studio-core.js), the same pieces as the studio viewer.
import * as THREE from "three";
import { OrbitControls } from "/vendor/three/OrbitControls.js";
import { GLTFLoader } from "/vendor/three/loaders/GLTFLoader.js";
import { RoomEnvironment } from "/vendor/three/environments/RoomEnvironment.js";
import * as core from "/studio/studio-core.js";

export function showGeometry(host, url, dims = {}) {
  const msg = host.querySelector(".vmsg");
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 0.95;
  host.prepend(renderer.domElement);
  renderer.domElement.setAttribute("aria-label", "3-D view: drag to rotate, scroll to zoom");
  renderer.domElement.setAttribute("role", "img");
  const scene = new THREE.Scene();
  scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
  const sun = new THREE.DirectionalLight(0xffffff, 1.4); scene.add(sun);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.5));
  const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 1e6);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
  let model = null, spin = !matchMedia("(prefers-reduced-motion: reduce)").matches, edges = "off";
  const meshes = [];
  const bar = document.createElement("div"); bar.className = "vbar";
  bar.innerHTML = `<button data-v="iso">Iso</button><button data-v="front">Front</button><button data-v="top">Top</button><button data-v="side">Side</button><button data-e>Edges</button><button data-s>${spin ? "Stop" : "Spin"}</button>`;
  host.append(bar);
  const dim = document.createElement("div"); dim.className = "vdim";
  dim.textContent = ["length_x", "width_y", "height_z"].filter((k) => dims[k] != null).map((k) => `${k[0].toUpperCase()} ${(+dims[k]).toPrecision(4)} m`).join(" · ");
  if (dim.textContent) host.append(dim);
  let radius = 1, center = new THREE.Vector3();
  function view(v) {
    const d = radius * 2.6, dir = { iso: [1, 0.7, 1], front: [0, 0, 1], top: [0, 1, 0.001], side: [1, 0, 0] }[v];
    camera.position.copy(center).add(new THREE.Vector3(...dir).normalize().multiplyScalar(d));
    camera.near = radius / 100; camera.far = radius * 100; camera.updateProjectionMatrix();
    controls.target.copy(center); controls.update();
  }
  bar.querySelectorAll("[data-v]").forEach((b) => (b.onclick = () => view(b.dataset.v)));
  bar.querySelector("[data-e]").onclick = () => { edges = edges === "off" ? "feature" : "off"; meshes.forEach((m) => core.setEdges(THREE, m, edges)); };
  bar.querySelector("[data-s]").onclick = (e) => { spin = !spin; e.target.textContent = spin ? "Stop" : "Spin"; };
  controls.addEventListener("start", () => { spin = false; bar.querySelector("[data-s]").textContent = "Spin"; });
  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, false); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(host); resize();
  new GLTFLoader().load(url, (g) => {
    model = g.scene;
    model.traverse((o) => {
      if (o.isMesh) {
        meshes.push(o);
        if (!o.material || !o.material.color) o.material = new THREE.MeshStandardMaterial({ color: 0x9fb3c8 });
        o.material.side = THREE.DoubleSide;
      }
    });
    scene.add(model);
    const box = new THREE.Box3().setFromObject(model), s = new THREE.Sphere();
    box.getBoundingSphere(s); radius = s.radius || 1; center = s.center;
    sun.position.copy(center).add(new THREE.Vector3(radius * 3, radius * 4, radius * 2));
    view("iso"); msg.remove();
  }, undefined, (err) => { msg.textContent = `could not load the mesh (${err?.message || err})`; });
  (function loop() {
    if (!host.isConnected) { renderer.dispose(); return; }
    requestAnimationFrame(loop);
    if (spin && model) model.rotation.y += 0.004;
    controls.update(); renderer.render(scene, camera);
  })();
}
