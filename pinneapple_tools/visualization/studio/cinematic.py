"""A cinematic, scroll-driven page for one 3-D result: the geometry with its field, a camera that walks around it
and one caption per beat.

    from pinneapple_tools.visualization.studio import Scene, cinematic_page
    sc = Scene.from_file("airliner.stl")
    sc.map_field("cp", points, cp)
    cinematic_page(sc, "site/demo", field="cp", title="A320 em cruzeiro",
                   sections=[{"kicker": "O resultado", "title": "Pressão na pele", "body": "..."}])

Writes ``out_dir/index.html`` (self-contained: the vendored three.js travels with it, no CDN, no server -- it opens
from ``file://``), ``scene.glb`` (the field baked into ``COLOR_0``, as the Unreal bridge does) and ``scene.json``
(the title, the field, its range and the captions, for anything that wants to read them back).

The page is built for a client-facing hero: black stage, exponential fog, a key/rim/fill light rig and a camera
whose azimuth, elevation and distance follow the scroll between the beats. It respects
``prefers-reduced-motion`` (no drift, no easing) and degrades to a readable page when WebGL is unavailable.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional, Sequence

from .compose import css_gradient, srgb_to_linear
from .scene import AXES, MATERIALS, Scene, write_glb
from .unreal import color_range, vertex_colors

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "web", "vendor")

#: (azimuth, elevation, distance in model radii) of the camera at each beat; repeated, with a nudge, when the page
#: has more captions than beats
BEATS = [(-2.10, 0.24, 3.70), (-0.95, 0.40, 3.10), (0.30, 0.20, 2.80), (1.45, 0.55, 3.40), (2.55, 0.28, 3.90)]


def beats_for(n: int) -> List[List[float]]:
    """``n`` camera beats: the fixed choreography, cycled with a small angular nudge past the fifth."""
    out = []
    for i in range(max(1, n)):
        az, el, d = BEATS[i % len(BEATS)]
        k = i // len(BEATS)
        out.append([round(az + k * 0.55, 3), round(max(0.05, el - k * 0.06), 3), round(d * (1 + k * 0.12), 3)])
    return out


def _page(scene: Scene, field: Optional[str], cmap: str, rng, sections: Sequence[Dict[str, str]],
          css: str = "", head: str = "", lang: str = "pt-BR", home: str = "../",
          scene_url: str = "scene.glb") -> tuple[str, Dict]:
    """The HTML and the ``scene.json`` it works from."""
    label = scene.labels.get(field, field) if field else ""
    lo, hi = (rng if field else (0.0, 1.0))
    cfg = {
        "title": scene.title,
        "scene": scene_url,
        "field": {"name": field, "label": label, "cmap": cmap, "range": [float(lo), float(hi)]},
        "sections": [{"kicker": s.get("kicker", ""), "title": s.get("title", ""), "body": s.get("body", "")}
                     for s in sections],
        "beats": beats_for(len(sections)),
    }
    caps = []
    for i, s in enumerate(cfg["sections"]):
        cls = "beat" + (" hero" if i == 0 else "")
        kicker = f'<p class="kicker">{s["kicker"]}</p>' if s["kicker"] else ""
        body = f'<p class="body">{s["body"]}</p>' if s["body"] else ""
        caps.append(f'<section class="{cls}" data-i="{i}">{kicker}<h{"1" if i == 0 else "2"}>{s["title"]}</h{"1" if i == 0 else "2"}>{body}</section>')
    cmap_css = css_gradient(cmap, 16) if field else ""
    hud = (f'<div id="hud"><p class="hud-title">{scene.title}</p><p class="hud-field">{label}</p>'
           f'<div class="cmap" style="background:{cmap_css}"></div>'
           f'<div class="range"><span>{lo:.4g}</span><span>{hi:.4g}</span></div></div>') if field else ""
    html = (_HTML.replace("__LANG__", lang).replace("__TITLE__", scene.title).replace("__CSS__", css)
            .replace("__HEAD__", head)
            .replace("__HOME__", home).replace("__CAPTIONS__", "\n".join(caps)).replace("__HUD__", hud)
            .replace("__JS__", _JS).replace("__CFG__", json.dumps(cfg, ensure_ascii=False)))
    return html, cfg


def cinematic_page(scene: Scene, out_dir: str, *, field: Optional[str] = None, cmap: str = "jet",
                   field_range: Optional[Sequence[float]] = None, title: Optional[str] = None,
                   sections: Optional[Sequence[Dict[str, str]]] = None, css: str = "", head: str = "",
                   lang: str = "pt-BR", home: str = "../", scene_url: Optional[str] = None,
                   filename: str = "index.html") -> str:
    """Write the page into ``out_dir`` and return the path of its HTML.

    ``field``: the field shown as vertex colours (default: the first the scene carries). ``sections``: one caption
    per beat, ``{"kicker", "title", "body"}``; the first is the hero. ``css``/``head``: extra rules and ``<head>``
    tags, so a site can bring its own fonts and colours. ``home``: where the page's back link points, relative to
    ``out_dir``. ``scene_url``: where the page loads the geometry from, when it is not the ``scene.glb`` written
    beside it (a second language's page can share one copy).
    """
    if not scene.surfaces:
        raise ValueError("the scene has no surfaces to show")
    if title:
        scene.title = title
    names = scene.field_names()
    if field is None:
        field = names[0] if names else None
    elif field not in names:
        raise KeyError(f"no surface carries the field {field!r} (fields: {names})")

    if sections is None:
        sections = [{"kicker": "PINNeAPPle", "title": scene.title, "body": "Um resultado de simulação em 3-D."}]
        for s in scene.surfaces[:3]:
            sections.append({"kicker": s.material, "title": s.name,
                             "body": f"{len(s.vertices):,} vértices, {len(s.faces):,} triângulos".replace(",", " ")})
    if not sections:
        raise ValueError("at least one section is needed")

    os.makedirs(out_dir, exist_ok=True)
    rng = color_range(scene, field, field_range, cmap) if field else (0.0, 1.0)
    colors = vertex_colors(scene, field, cmap=cmap, rng=field_range) if field else None
    local = scene_url is None or scene_url == "scene.glb"
    if local:
        with open(os.path.join(out_dir, "scene.glb"), "wb") as fh:
            fh.write(write_glb(scene.surfaces, AXES[scene.axes], {**MATERIALS, **scene.materials},
                               generator="PINNeAPPle (cinematic page)",
                               colors=None if colors is None else {k: srgb_to_linear(v) for k, v in colors.items()}))

    html, cfg = _page(scene, field, cmap, rng, sections, css=css, head=head, lang=lang, home=home,
                     scene_url=scene_url or "scene.glb")
    with open(os.path.join(out_dir, "scene.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1, ensure_ascii=False)
    page = os.path.join(out_dir, filename)
    with open(page, "w", encoding="utf-8") as fh:
        fh.write(html)

    dst = os.path.join(out_dir, "vendor")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(VENDOR, dst)
    return os.path.abspath(page)


_HTML = """<!doctype html>
<html lang="__LANG__">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
  :root { --ink: #eef2f6; --dim: #8f9aa6; --bg: #070a10; --accent: #3dd6c4; }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         -webkit-font-smoothing: antialiased; }
  #stage { position: fixed; inset: 0; width: 100%; height: 100%; display: block; z-index: 0; }
  .vignette { position: fixed; inset: 0; z-index: 1; pointer-events: none;
              background: radial-gradient(ellipse at 50% 45%, transparent 40%, rgba(3,5,9,.72) 100%); }
  .scrim { position: fixed; inset: 0; z-index: 1; pointer-events: none;
           background: linear-gradient(100deg, rgba(4,7,12,.80) 0%, rgba(4,7,12,.28) 30%,
                       rgba(4,7,12,.04) 50%, rgba(4,7,12,0) 62%); }
  .bar { position: fixed; z-index: 3; inset: 0 0 auto 0; display: flex; justify-content: space-between;
         align-items: center; padding: 18px clamp(16px, 4vw, 40px); font-size: 13px; letter-spacing: .12em;
         text-transform: uppercase; color: var(--dim); }
  .bar a { color: var(--dim); text-decoration: none; border-bottom: 1px solid rgba(143,154,166,.35); padding-bottom: 2px; }
  .bar a:hover { color: var(--accent); border-color: var(--accent); }
  main { position: relative; z-index: 2; }
  .beat { min-height: 100svh; display: flex; flex-direction: column; justify-content: center;
          padding: 0 clamp(20px, 7vw, 120px); max-width: 720px; opacity: 0; transform: translateY(14px);
          transition: opacity .5s ease, transform .5s ease; }
  .beat.on { opacity: 1; transform: none; }
  .kicker { margin: 0 0 10px; font-size: 12px; letter-spacing: .22em; text-transform: uppercase; color: var(--accent); }
  h1 { font-size: clamp(34px, 7vw, 76px); line-height: .98; margin: 0 0 18px; letter-spacing: -.03em; }
  h2 { font-size: clamp(26px, 4.4vw, 48px); line-height: 1.04; margin: 0 0 14px; letter-spacing: -.02em; }
  .body { margin: 0; font-size: clamp(15px, 1.5vw, 19px); line-height: 1.6; color: #c3ccd6; max-width: 46ch; }
  #hud { position: fixed; z-index: 3; right: clamp(16px, 4vw, 40px); bottom: clamp(18px, 4vw, 40px);
         width: 210px; font-size: 11px; letter-spacing: .06em; color: var(--dim); text-align: right; }
  #hud .hud-title { margin: 0 0 2px; color: var(--ink); font-size: 12px; letter-spacing: .02em; }
  #hud .hud-field { margin: 0 0 7px; }
  #hud .cmap { height: 8px; border-radius: 2px; border: 1px solid rgba(255,255,255,.14); }
  #hud .range { display: flex; justify-content: space-between; margin-top: 4px; }
  #boot { position: fixed; z-index: 4; inset: 0; display: grid; place-items: center; background: var(--bg);
          color: var(--dim); font-size: 13px; letter-spacing: .18em; text-transform: uppercase;
          transition: opacity .6s ease; }
  body.ready #boot { opacity: 0; pointer-events: none; }
  .no-webgl #stage { display: none; }
  .no-webgl body, body.no-webgl { background: var(--bg); }
  .no-webgl #boot { opacity: 0; }
  .no-webgl main { padding-bottom: 40vh; }
  .no-webgl .beat { opacity: 1; transform: none; }
  @media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
    .beat { transition: none; }
  }
  __CSS__
</style>
__HEAD__
<script type="importmap">{ "imports": { "three": "./vendor/three/three.module.min.js" } }</script>
</head>
<body>
<canvas id="stage"></canvas>
<div class="vignette"></div>
<div class="scrim"></div>
<div class="bar"><span>PINNeAPPle Labs</span><a href="__HOME__">&larr; site</a></div>
__HUD__
<div id="boot">carregando</div>
<main>
__CAPTIONS__
</main>
<script type="module">
__JS__
</script>
</body>
</html>
"""

_JS = r"""
import * as THREE from 'three';
import { GLTFLoader } from './vendor/three/loaders/GLTFLoader.js';

const CFG = __CFG__;
const canvas = document.getElementById('stage');
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const smooth = (t) => t * t * (3 - 2 * t);

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
} catch (err) {
  document.body.classList.add('no-webgl');
  document.body.classList.add('ready');
  throw err;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(innerWidth, innerHeight, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070a10);
const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.01, 1000);

scene.add(new THREE.HemisphereLight(0x93d4ff, 0x04070c, 1.05));
const key = new THREE.DirectionalLight(0xffffff, 2.7); key.position.set(1.3, 1.9, 1.0); scene.add(key);
const rim = new THREE.DirectionalLight(0x3dd6c4, 1.7); rim.position.set(-1.5, 0.6, -1.3); scene.add(rim);
const fill = new THREE.DirectionalLight(0xff8a5b, 0.75); fill.position.set(-0.7, -0.5, 1.1); scene.add(fill);

const pivot = new THREE.Group();
scene.add(pivot);

let radius = 1, loaded = false;
const target = new THREE.Vector3();
const wanted = new THREE.Vector3();
const clock = new THREE.Clock();

new GLTFLoader().load(CFG.scene || 'scene.glb', (gltf) => {
  const root = gltf.scene;
  root.traverse((o) => {
    if (!o.isMesh) return;
    const coloured = !!o.geometry.getAttribute('color');
    const m = new THREE.MeshStandardMaterial({
      vertexColors: coloured,
      color: coloured ? 0xffffff : 0x8d949c,
      metalness: coloured ? 0.05 : 0.4,
      roughness: coloured ? 0.4 : 0.5,
    });
    if (coloured) {
      // a little self-illumination from the vertex colours: the low end of the scale still reads on a black stage
      m.onBeforeCompile = (shader) => {
        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <emissivemap_fragment>',
          '#include <emissivemap_fragment>\n\ttotalEmissiveRadiance += vColor.rgb * 0.22;'
        );
      };
    }
    o.material = m;
  });
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  radius = Math.max(size.x, size.y, size.z) * 0.5 || 1;
  root.position.sub(centre);
  pivot.add(root);
  scene.fog = new THREE.FogExp2(0x070a10, 0.12 / radius);
  camera.near = radius / 200;
  camera.far = radius * 80;
  camera.updateProjectionMatrix();
  loaded = true;
  snap();                       // land on the current beat instead of flying in from the origin
  document.body.classList.add('ready');
}, undefined, (err) => {
  console.error(err);
  document.body.classList.add('no-webgl', 'ready');
});

function place(az, el, d) {
  return new THREE.Vector3(
    Math.cos(el) * Math.cos(az),
    Math.sin(el),
    Math.cos(el) * Math.sin(az)
  ).multiplyScalar(d * radius);
}

function progress() {
  const max = document.documentElement.scrollHeight - innerHeight;
  return max > 0 ? clamp(scrollY / max, 0, 1) : 0;
}

const caps = [...document.querySelectorAll('.beat')];

function beatAt(t) {
  const beats = CFG.beats;
  const n = beats.length - 1;
  const u = n > 0 ? clamp(t, 0, 1) * n : 0;
  const i = Math.min(Math.max(0, n - 1), Math.floor(u));
  const f = n > 0 ? smooth(u - i) : 0;
  const a = beats[i], b = beats[Math.min(n, i + 1)];
  return {
    u: u,
    az: a[0] + (b[0] - a[0]) * f,
    el: a[1] + (b[1] - a[1]) * f,
    d: a[2] + (b[2] - a[2]) * f,
  };
}

function snap() {
  const b = beatAt(progress());
  camera.position.copy(place(b.az, b.el, b.d));
  camera.lookAt(0, 0, 0);
}

function update() {
  if (!loaded) return;
  const b = beatAt(progress());
  const u = b.u;
  const drift = reduced ? 0 : Math.sin(clock.elapsedTime * 0.22) * 0.035;
  wanted.copy(place(b.az + drift, b.el, b.d));
  camera.position.lerp(wanted, reduced ? 1 : 0.09);
  target.set(0, 0, 0);
  camera.lookAt(target);
  const nearest = Math.round(clamp(u, 0, caps.length - 1));
  caps.forEach((c, k) => c.classList.toggle('on', k === nearest));
}

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight, false);
});

renderer.setAnimationLoop(() => {
  update();
  renderer.render(scene, camera);
});
"""
