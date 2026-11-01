// Offline support. build.py fills VERSION and PRECACHE (the app shell, models, gallery index, thumbnails and
// three.js); the light glTF files are cached the first time they are viewed.
const VERSION = "b06386db3f7d";
const PRECACHE = ["./", "css/app.css", "data/geometries.json", "data/thumbs/airliner_a320.jpg", "data/thumbs/aortic_arch.jpg", "data/thumbs/caarc_building.jpg", "data/thumbs/centrifugal_impeller.jpg", "data/thumbs/container_ship_kcs.jpg", "data/thumbs/f1_car.jpg", "data/thumbs/falcon9_booster.jpg", "data/thumbs/helicopter_h145.jpg", "data/thumbs/high_speed_train.jpg", "data/thumbs/launch_vehicle_two_stage.jpg", "data/thumbs/pin_fin_heatsink.jpg", "data/thumbs/quadrotor_drone.jpg", "data/thumbs/road_car.jpg", "data/thumbs/turboprop_atr72.jpg", "data/thumbs/viper_rover.jpg", "data/thumbs/wigley_hull.jpg", "data/thumbs/wind_turbine_nrel5mw.jpg", "icon.svg", "js/charts.js", "js/main.js", "js/models.js", "js/modes/airfoil.js", "js/modes/gallery.js", "js/modes/heatsink.js", "js/modes/rover.js", "js/trust.js", "js/ui.js", "lib/onnx.js", "lib/zoo.js", "manifest.webmanifest", "models/airfoil_cst_rans_mlp.json", "models/airfoil_cst_rans_mlp.onnx", "models/cst_from_handles.onnx", "models/cst_geometry.json", "models/cst_geometry.onnx", "models/heatsink_forced_mlp.json", "models/heatsink_forced_mlp.onnx", "models/index.json", "models/rover_contact_grc1_viper.json", "models/rover_contact_grc1_viper.onnx", "vendor/three/LICENSE", "vendor/three/OrbitControls.js", "vendor/three/environments/RoomEnvironment.js", "vendor/three/loaders/GLTFLoader.js", "vendor/three/three.module.min.js", "vendor/three/utils/BufferGeometryUtils.js", "index.html"];
const CACHE = `playground-${VERSION}`;

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k.startsWith("playground-") && k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return;
  // cache first (files are versioned by the cache name), then network; store what the network returns
  e.respondWith(caches.open(CACHE).then(async (c) => {
    const hit = await c.match(req, { ignoreSearch: true });
    if (hit) return hit;
    const res = await fetch(req);
    if (res.ok && res.type === "basic") c.put(req, res.clone());
    return res;
  }));
});
