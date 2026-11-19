"""The dashboard: move the geometry, see the surrogate answer, and check it against a real solve on demand.

    python -m pinneapple_simulation.pipelines.usd_structural_surrogate.dashboard run/surrogate --port 8131

Every slider move posts the four design variables, revolves the part's section (:func:`design.revolve_surface`,
microseconds -- gmsh takes seconds and would not be interactive), samples the predicted fields at the surface
vertices and returns one payload. Nothing is cached between designs, so what the page shows is the network's answer
for that geometry and not an interpolation of two solved cases.

Three buttons, and each one is a check rather than a flourish:

**optimise** -- descend on the surrogate: lightest pedestal whose predicted peak von Mises stays under a fraction of
yield, with a floor on the plate thickness and the column diameter. The surrogate makes the search free; the number
it lands on is only a proposal until it is solved.

**solve for real** -- run ``ccx`` on the geometry the sliders describe and put its peak stress and tip displacement
beside the prediction. This is the only number on the page that is not the model talking about itself.

**refine** -- add that solved design to a short fine-tuning set and update the operator on the spot (a few hundred
steps). It is the adapt-to-new-data loop: the surrogate is corrected by the solver at the point where the user
noticed the disagreement.
"""
from __future__ import annotations

import json
import math
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .design import REFERENCE_LOAD_N, profile, revolve_surface
from .grid import sample_field

__all__ = ["Predictor", "serve", "main"]

YIELD_STEEL_PA = 250e6
# every design variable is a length the sliders show in millimetres and the model works in metres
SLIDER = {"t": ("plate thickness", "mm", 1e3, 1),
          "dc": ("column diameter", "mm", 1e3, 1),
          "r": ("flare radius", "mm", 1e3, 1),
          "h": ("load height", "mm", 1e3, 0)}


class Predictor:
    """Turns a design into the payload the page draws: surface, displacement, von Mises, and the numbers."""

    def __init__(self, surrogate, *, n_theta: int = 96):
        self.s = surrogate
        self.n_theta = n_theta
        self.lock = threading.Lock()

    def design_from_query(self, q: dict[str, str]) -> dict[str, float]:
        """The query's design in **metres**: the sliders carry millimetres, the model does not."""
        space = self.s.space
        names = list(space.bounds)
        if all(k in q for k in names):
            return {k: float(q[k]) / SLIDER.get(k, (0, "", 1.0, 0))[2] for k in names}
        u = [float(q.get(f"u{i}", 0.5)) for i in range(len(names))]
        return space.denormalized(u)

    def payload(self, design: dict[str, float], load_n: float = REFERENCE_LOAD_N, *,
                scale: float | None = None) -> dict[str, Any]:
        if not self.s.valid(design):
            raise ValueError(f"design outside the trained envelope {sorted(self.s.space.bounds)}: {design}")
        V, F = revolve_surface(design, self.s.space.plate_diameter, self.n_theta)
        occ = self._occupancy(design, V)
        feats = self._features(occ)
        with self.lock:
            pred = self.s.predict_fields(feats)
        k = float(load_n) / REFERENCE_LOAD_N                 # linear elastic: exact, checked by checks.linearity
        u = np.stack([sample_field(pred[c], self.s.grid, V) for c in ("ux", "uy", "uz")], axis=-1) * k
        vm = sample_field(pred["vm"], self.s.grid, V) * k
        mag = np.linalg.norm(u, axis=1)
        # the peak is read over the solid's cells, not over the drawn surface: the stress concentration's hottest
        # cell is an interior one, and a maximum over surface vertices reports 7.65 MPa where the grid says 10.11
        # and ccx 13.30 -- a display under-report that made the surrogate look better than it is.
        peak_grid = float(pred["vm"][occ].max()) * k
        # the display exaggeration, as in an FEA post-processor: 6 % of the part's height at the largest displacement
        k_vis = scale if scale is not None else 0.06 * float(np.ptp(V[:, 2])) / max(float(mag.max()), 1e-12)
        deformed = V + k_vis * u
        return {
            "design": {k2: float(v2) for k2, v2 in design.items()},
            "load_n": float(load_n),
            "vertices": V.astype(float).reshape(-1).tolist(),
            "deformed": deformed.astype(float).reshape(-1).tolist(),
            "faces": F.astype(int).reshape(-1).tolist(),
            "vm": vm.astype(float).tolist(),
            "u": mag.astype(float).tolist(),
            "exaggeration": float(k_vis),
            "stats": {"peak_vm_pa": peak_grid, "peak_vm_mpa": peak_grid / 1e6,
                      "tip_mm": float(mag.max()) * 1e3,
                      "mass_kg": self._mass(design),
                      "utilisation": peak_grid / YIELD_STEEL_PA,
                      "yield_pa": YIELD_STEEL_PA},
            "envelope_ok": self.s.valid(design),
        }

    # ------------------------------------------------------------------ geometry -> grid
    def _occupancy(self, design: dict[str, float], V: np.ndarray | None = None) -> np.ndarray:
        """Voxelise the part: a cell is inside when its radius is under the section's radius at its height.

        The radius at height ``z`` is closed-form -- plate, cone, column -- rather than interpolated off
        :func:`design.profile`, because the profile begins with two points at ``z = 0`` (the axis and the plate's rim)
        and an interpolator over a repeated abscissa picks the wrong branch there, leaving the plate's own cells
        outside the part. The solid drawn by :func:`design.revolve_surface` is this same one.
        """
        g = self.s.grid
        C = g.centers()
        t, dc, r, h = (float(design[k]) for k in ("t", "dc", "r", "h"))
        R = self.s.space.plate_diameter / 2.0
        rc, rf = dc / 2.0, dc / 2.0 + r
        z = C[:, 2]
        radius_at = np.where(z < t, R,
                             np.where(z < t + r, rf - (rf - rc) * (z - t) / max(r, 1e-12), rc))
        radius_at = np.where((z < 0.0) | (z > h), 0.0, radius_at)
        return (np.hypot(C[:, 0], C[:, 1]) <= radius_at).reshape(g.n)

    def _features(self, occ: np.ndarray) -> np.ndarray:
        from .surrogate import features
        return features(occ, _sdf(occ, self.s.grid), self.s.grid)

    @staticmethod
    def _mass(design: dict[str, float], rho: float = 7850.0,
              plate_diameter: float | None = None) -> float:
        rz = profile(design, plate_diameter or 0.2719)
        vol = 0.0
        for (r0, z0), (r1, z1) in zip(rz[:-1], rz[1:], strict=False):
            vol += math.pi * (z1 - z0) * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0
        return vol * rho

    # ------------------------------------------------------------------ optimisation
    def optimise(self, start: dict[str, float], *, load_n: float = REFERENCE_LOAD_N, target: float = 0.6,
                 steps: int = 16, lr: float = 0.35) -> dict[str, Any]:
        """Lightest design whose predicted peak von Mises stays under ``target`` x yield, by projected descent.

        The objective is mass; the constraint is the predicted peak. The descent is on the surrogate, so it costs a
        forward pass per step -- and it is a *proposal*: the page's "solve for real" is what makes it a number.
        """
        from .surrogate import features
        u = np.asarray(self.s.space.normalized(start), float)
        rng = np.random.default_rng(0)
        best = {"util": float("inf"), "u": u.copy(), "mass": float("inf"), "design": dict(start)}
        trace = []
        eps = 1e-3
        for i in range(steps):
            design = self.s.space.denormalized(u)
            # the design and its one-sided perturbation in every variable, in one batched forward pass
            probes, masses = [design], [self._mass(design)]
            for j in range(len(u)):
                du = u.copy()
                du[j] = min(1.0, du[j] + eps)
                d2 = self.s.space.denormalized(du)
                probes.append(d2)
                masses.append(self._mass(d2))
            feats, occs = [], []
            for d in probes:
                occ_d = self._occupancy(d, revolve_surface(d, self.s.space.plate_diameter, 32)[0])
                occs.append(occ_d)
                feats.append(features(occ_d, _sdf(occ_d, self.s.grid), self.s.grid))
            out = self.s.predict_batch(feats)
            # the peak over the solid's cells, which is what the payload reports and what the checks score --
            # a maximum over the whole box would read the filled cells outside the part too
            utils = [float(o["vm"][occ_d].max()) * load_n / REFERENCE_LOAD_N / YIELD_STEEL_PA
                     for o, occ_d in zip(out, occs, strict=True)]
            util, m = utils[0], masses[0]
            trace.append({"step": i, "mass_kg": m, "util": util,
                          **{k: float(v) for k, v in design.items()}})
            # feasible and lighter than anything feasible so far?
            if util <= target and m < best["mass"]:
                best = {"util": util, "u": u.copy(), "mass": m, "design": dict(design)}
            # gradient by finite differences on the penalised objective (mass + stress violation)
            grad = np.array([(masses[j + 1] - m) / 0.05 + 400.0 * max(0.0, utils[j + 1] - target)
                             - 400.0 * max(0.0, util - target) for j in range(len(u))])
            u = np.clip(u - lr * grad / (np.linalg.norm(grad) + 1e-9) + 0.01 * rng.standard_normal(len(u)), 0.0, 1.0)
        return {**best, "trace": trace, "target": target, "yield_pa": YIELD_STEEL_PA}

    # ------------------------------------------------------------------ real solve
    def solve_real(self, design: dict[str, float], workdir: str, *, load_n: float = REFERENCE_LOAD_N,
                   element_size: float = 0.020, threads: int = 2) -> dict[str, Any]:
        """Run ``ccx`` on this geometry and return the numbers to set beside the prediction."""
        from .design import run_design
        r = run_design(design, workdir, element_size=element_size, threads=threads)
        k = float(load_n) / REFERENCE_LOAD_N
        return {"peak_vm_pa": float(r["von_mises"].max()) * k,
                "peak_vm_mpa": float(r["von_mises"].max()) * k / 1e6,
                "tip_mm": float(np.abs(r["u"]).max()) * k * 1e3,
                "n_nodes": r["info"]["n_nodes"], "n_elements": int(len(r["elements"])),
                "design": {k2: float(v2) for k2, v2 in design.items()},
                "fields": {"u": r["u"].tolist(), "vm": r["von_mises"].tolist(),
                           "nodes": r["nodes"].tolist(), "elements": r["elements"].tolist()}}

    def refine(self, design: dict[str, float], workdir: str, *, steps: int = 120, lr: float = 5e-4,
               element_size: float = 0.020, threads: int = 2) -> dict[str, Any]:
        """Solve this design for real, fine-tune the operator on it, and report the change in error."""
        import torch

        from .grid import fill_outside, rasterize_barycentric
        from .surrogate import features as _f

        r = self.solve_real(design, workdir, element_size=element_size, threads=threads)
        grid = self.s.grid
        nodes = np.array(r["fields"]["nodes"], float)
        V, _ = revolve_surface(design, self.s.space.plate_diameter, 48)
        occ = self._occupancy(design, V)
        elem = np.array(r["fields"]["elements"], int)          # same interpolation the campaign trained on
        u = fill_outside(rasterize_barycentric(nodes, np.array(r["fields"]["u"]), grid, elem, "C3D4", inside=occ),
                         occ, grid)
        vm = fill_outside(rasterize_barycentric(nodes, np.array(r["fields"]["vm"]), grid, elem, "C3D4", inside=occ),
                          occ, grid)
        x = torch.from_numpy(_f(occ, _sdf(occ, grid), grid)[None])
        y_raw = np.stack([u[..., k] for k in range(3)] + [vm])[None] / self.s.out_scale[None, :, None, None, None]
        y = torch.from_numpy(y_raw.astype(np.float32))
        net = self.s.net
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        losses = []
        with self.lock:
            for _ in range(steps):
                net.train()
                loss = torch.nn.functional.mse_loss(net(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            net.eval()
        return {"steps": steps, "first_loss": losses[0], "last_loss": losses[-1],
                "solved": {k: v for k, v in r.items() if k != "fields"}}


def _sdf(occ: np.ndarray, grid) -> np.ndarray:
    from .grid import signed_distance
    return signed_distance(occ, grid)


# ---------------------------------------------------------------------- the page
_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>UR10 pedestal — structural surrogate</title>
<style>
:root{--bg:#0e1013;--fg:#e8ecf1;--dim:#8b95a5;--acc:#4cc9f0;--warn:#f0932b;--bad:#e05555}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
#app{display:flex;height:100vh}
#view{flex:1;position:relative}
canvas{display:block}
#panel{width:360px;padding:18px;overflow-y:auto;border-left:1px solid #1d2228;background:#12151a}
h1{font-size:15px;margin:0 0 2px;font-weight:600}
.sub{color:var(--dim);font-size:12px;margin-bottom:16px}
label{display:flex;justify-content:space-between;font-size:12px;color:var(--dim);margin-top:13px}
label b{color:var(--fg);font-variant-numeric:tabular-nums}
input[type=range]{width:100%;margin:5px 0 0;accent-color:var(--acc)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:18px}
.stat{background:#171b21;border:1px solid #232a33;border-radius:8px;padding:9px 10px}
.stat span{display:block;font-size:11px;color:var(--dim)}
.stat b{font-size:17px;font-variant-numeric:tabular-nums}
button{width:100%;margin-top:9px;padding:9px;border-radius:8px;border:1px solid #2b3742;
background:#1b2530;color:var(--fg);font:13px inherit;cursor:pointer}
button:hover{background:#223040}button:disabled{opacity:.45;cursor:default}
#log{margin-top:14px;font-size:11.5px;color:var(--dim);white-space:pre-wrap;font-family:ui-monospace,monospace}
#bar{position:absolute;left:14px;bottom:12px;font-size:11px;color:var(--dim)}
.legend{height:8px;border-radius:4px;margin-top:5px;
background:linear-gradient(90deg,#2c5fb8,#20b2aa,#8fd14f,#ffd93d,#ff8b3d,#d7263d)}
.ok{color:var(--acc)}.no{color:var(--bad)}
</style></head><body><div id="app"><div id="view"><div id="bar">drag to orbit · scroll to zoom</div></div>
<div id="panel">
<h1>UR10 pedestal · structural surrogate</h1>
<div class="sub">geometry from NVIDIA's OpenUSD asset · CalculiX · FNO (PhysicsNeMo)</div>
<div id="sliders"></div>
<label>load <b><span id="vload">1000</span> N</b></label>
<input id="load" type="range" min="200" max="2000" step="50" value="1000">
<div class="grid">
<div class="stat"><span>σ<sub>vm</sub> peak</span><b id="svm">–</b></div>
<div class="stat"><span>tip deflection</span><b id="stip">–</b></div>
<div class="stat"><span>mass</span><b id="smass">–</b></div>
<div class="stat"><span>utilisation (σ/σy)</span><b id="sutil">–</b></div>
</div>
<div class="legend"></div>
<button id="bopt">optimise (lightest under σ ≤ 0.6 σy)</button>
<button id="bsolve">solve with the real ccx</button>
<button id="brefine" disabled>refine the model on this case</button>
<div id="log"></div>
</div></div>
<script type="importmap">{"imports":{"three":"https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js"}}</script>
<script type="module">
import * as THREE from 'three';
const NAMES=JSON.parse(document.getElementById('sliders').dataset.names||'[]');
let SPEC=null, lastPayload=null, lastReal=null;
const view=document.getElementById('view');
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0e1013);
const cam=new THREE.PerspectiveCamera(38,1,0.01,100);
cam.up.set(0,0,1);                       // the part is Z-up, as it is in the USD asset
const rnd=new THREE.WebGLRenderer({antialias:true}); rnd.setPixelRatio(devicePixelRatio);
view.appendChild(rnd.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x223044,1.15));
const key=new THREE.DirectionalLight(0xffffff,1.5); key.position.set(1,2,1.4); scene.add(key);
const fill=new THREE.DirectionalLight(0x88bbff,0.5); fill.position.set(-1.4,-0.6,0.8); scene.add(fill);
let mesh=null, target=new THREE.Vector3(0,0,0.25), dist=1.15, rotY=0.5, rotX=0.35;
function cmap(t){ t=Math.min(1,Math.max(0,t));
  const stops=[[0.17,0.37,0.72],[0.13,0.70,0.67],[0.56,0.82,0.31],[1.0,0.85,0.24],[1.0,0.55,0.24],[0.84,0.15,0.24]];
  const x=t*(stops.length-1), i=Math.min(stops.length-2,Math.floor(x)), f=x-i;
  return [0,1,2].map(k=>stops[i][k]+f*(stops[i+1][k]-stops[i][k])); }
function draw(p){
  const V=new Float32Array(p.vertices), Vd=new Float32Array(p.deformed), F=p.faces;
  if(!mesh){ const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.BufferAttribute(Vd,3));
    g.setIndex(F); g.computeVertexNormals();
    g.setAttribute('color',new THREE.BufferAttribute(new Float32Array(V.length),3));
    mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,metalness:0.25,roughness:0.55,
      flatShading:false,side:THREE.DoubleSide}));
    scene.add(mesh); fit(Vd);
  } else { mesh.geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(F),1)); }
  const g=mesh.geometry, pos=g.getAttribute('position'), col=g.getAttribute('color');
  pos.array.set(Vd); col.array.set(new Float32Array(Vd.length));
  const mx=Math.max(...p.vm), mn=Math.min(...p.vm);
  for(let i=0;i<p.vm.length;i++){const c=cmap((p.vm[i]-mn)/Math.max(mx-mn,1e-12));
    col.array[3*i]=c[0];col.array[3*i+1]=c[1];col.array[3*i+2]=c[2];}
  pos.needsUpdate=true; col.needsUpdate=true; g.computeBoundingSphere(); fit(Vd);
  lastPayload=p; renderStats(p);
}
function fit(Vd){ let lo=[1e9,1e9,1e9],hi=[-1e9,-1e9,-1e9];
  for(let i=0;i<Vd.length;i+=3)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],Vd[i+k]);hi[k]=Math.max(hi[k],Vd[i+k]);}
  target.set((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2);
  const span=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]);
  dist=0.88*span/Math.tan(0.5*cam.fov*Math.PI/180);  // room for the rotated view, not just the AABB
  cam.near=0.01*dist; cam.far=10*dist; cam.updateProjectionMatrix(); }
function renderStats(p){ const s=p.stats;
  document.getElementById('svm').textContent=s.peak_vm_mpa.toFixed(1)+' MPa';
  document.getElementById('stip').textContent=s.tip_mm.toFixed(3)+' mm';
  document.getElementById('smass').textContent=s.mass_kg.toFixed(2)+' kg';
  const e=document.getElementById('sutil'); e.textContent=(100*s.utilisation).toFixed(0)+'%';
  e.className=s.utilisation<0.6?'ok':(s.utilisation<1?'':'no'); }
function resize(){ const w=view.clientWidth,h=view.clientHeight;
  rnd.setSize(w,h); cam.aspect=w/h; cam.updateProjectionMatrix(); }
addEventListener('resize',resize);
let drag=null;
rnd.domElement.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];});
addEventListener('pointerup',()=>drag=null);
addEventListener('pointermove',e=>{ if(!drag)return;
  rotY+=(e.clientX-drag[0])*0.01; rotX=Math.max(-1.4,Math.min(1.4,rotX+(e.clientY-drag[1])*0.01));
  drag=[e.clientX,e.clientY]; });
rnd.domElement.addEventListener('wheel',e=>{e.preventDefault(); dist*=Math.exp(e.deltaY*0.0012);},{passive:false});
function loop(){ requestAnimationFrame(loop);
  const cy=Math.cos(rotY),sy=Math.sin(rotY),cx=Math.cos(rotX),sx=Math.sin(rotX);
  cam.position.set(target.x+dist*cy*cx, target.y+dist*sy*cx, target.z+dist*sx);
  cam.lookAt(target); rnd.render(scene,cam); }
function log(m){ document.getElementById('log').textContent=m; }
let t0=0;
async function fetchPredict(){
  const qs=slidersQuery();
  const r=await fetch('/api/predict?'+qs); const p=await r.json(); draw(p); return p; }
function slidersQuery(){
  const q=new URLSearchParams();
  for(const s of SPEC.variables) q.set(s.name, document.getElementById('sl_'+s.name).value);
  q.set('F', document.getElementById('load').value); return q.toString(); }
function buildSliders(spec){
  SPEC=spec; const box=document.getElementById('sliders'); box.innerHTML='';
  for(const s of spec.variables){
    const lab=document.createElement('label');
    lab.innerHTML='<span>'+s.label+'</span><b><span id="v_'+s.name+'"></span> '+s.unit+'</b>';
    const inp=document.createElement('input'); inp.type='range'; inp.id='sl_'+s.name;
    inp.min=s.min; inp.max=s.max; inp.step=(s.max-s.min)/200; inp.value=s.value;
    const upd=()=>{document.getElementById('v_'+s.name).textContent=(+inp.value).toFixed(s.decimals);};
    inp.addEventListener('input',()=>{upd(); clearTimeout(t0); t0=setTimeout(fetchPredict,60);});
    box.appendChild(lab); box.appendChild(inp); upd();
  }
  document.getElementById('load').addEventListener('input',()=>{
    document.getElementById('vload').textContent=document.getElementById('load').value;
    clearTimeout(t0); t0=setTimeout(fetchPredict,60); });
}
async function boot(){
  const r=await fetch('/api/spec'); const spec=await r.json(); buildSliders(spec); resize(); loop();
  await fetchPredict();
  document.getElementById('bopt').onclick=async()=>{
    log('optimising on the surrogate…'); const r=await fetch('/api/optimize?'+slidersQuery());
    const o=await r.json();
    for(const s of SPEC.variables){ const el=document.getElementById('sl_'+s.name);
      if(o.design[s.name]!==undefined){ const mm=o.design[s.name]*1000;   // the model answers in metres
        el.value=mm; document.getElementById('v_'+s.name).textContent=mm.toFixed(s.decimals);} }
    await fetchPredict(); log('proposal: mass '+o.mass.toFixed(2)+' kg, utilisation '+
      (100*o.util).toFixed(0)+'%  ('+o.steps+' steps on the surrogate)'); };
  document.getElementById('bsolve').onclick=async()=>{
    log('running ccx…'); const b=document.getElementById('bsolve'); b.disabled=true;
    try{ const r=await fetch('/api/solve?'+slidersQuery()); const o=await r.json();
      if(o.error){ log('ccx: '+o.error); } else {
        lastReal=o; const p=lastPayload;
        const dp=100*(Math.abs(p.stats.peak_vm_pa-o.peak_vm_pa)/o.peak_vm_pa);
        const du=100*(Math.abs(p.stats.tip_mm-o.tip_mm)/Math.max(o.tip_mm,1e-9));
        log('ccx: σ='+o.peak_vm_mpa.toFixed(1)+' MPa, tip='+o.tip_mm.toFixed(3)+' mm, '+o.n_nodes+' nodes\n'+
            'surrogate: σ='+p.stats.peak_vm_mpa.toFixed(1)+' MPa ('+dp.toFixed(1)+'% off), tip='+
            p.stats.tip_mm.toFixed(3)+' mm ('+du.toFixed(1)+'%)');
        document.getElementById('brefine').disabled=false; } }
    finally{ b.disabled=false; } };
  document.getElementById('brefine').onclick=async()=>{
    log('fine-tuning the model on the solved case…'); const b=document.getElementById('brefine'); b.disabled=true;
    try{ const r=await fetch('/api/refine?'+slidersQuery()); const o=await r.json();
      if(o.error){ log('refine: '+o.error); } else {
        await fetchPredict();
        const p=lastPayload, rr=o.solved;
        const dp=100*(Math.abs(p.stats.peak_vm_pa-rr.peak_vm_pa)/rr.peak_vm_pa);
        log('refined in '+o.steps+' steps (loss '+o.first_loss.toExponential(1)+' → '+
            o.last_loss.toExponential(1)+')\nnow: σ='+p.stats.peak_vm_mpa.toFixed(1)+' MPa vs ccx '+
            rr.peak_vm_mpa.toFixed(1)+' MPa ('+dp.toFixed(1)+'%)'); } }
    finally{ b.disabled=false; } };
}
boot();
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    predictor: Predictor = None
    workdir: str = ""

    def log_message(self, *a):                                        # keep the console for the run
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, _PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/spec":
                return self._send(200, json.dumps(self._spec()).encode())
            design = self.predictor.design_from_query(q)
            load = float(q.get("F", REFERENCE_LOAD_N))
            if u.path == "/api/predict":
                return self._send(200, json.dumps(self.predictor.payload(design, load)).encode())
            if u.path == "/api/optimize":
                out = self.predictor.optimise(design, load_n=load)
                return self._send(200, json.dumps({"design": out["design"], "mass": out["mass"],
                                                   "util": out["util"], "trace": out["trace"][-8:],
                                                   "steps": len(out["trace"])}).encode())
            if u.path == "/api/solve":
                out = self.predictor.solve_real(design, self.workdir, load_n=load)
                out.pop("fields", None)
                return self._send(200, json.dumps(out).encode())
            if u.path == "/api/refine":
                return self._send(200, json.dumps(self.predictor.refine(design, self.workdir)).encode())
            return self._send(404, b'{"error":"not found"}')
        except Exception as e:                                        # a bad design is a message, not a 500 page
            return self._send(200, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())

    def _spec(self) -> dict[str, Any]:
        space = self.predictor.s.space
        labels = {"t": ("plate thickness", "mm", 1e3, 1), "dc": ("column diameter", "mm", 1e3, 1),
                  "r": ("flare radius", "mm", 1e3, 1), "h": ("load height", "mm", 1e3, 0)}
        nominal = space.denormalized([0.5] * len(space.bounds))
        variables = []
        for k, (a, b) in space.bounds.items():
            lab, unit, scale, dec = labels.get(k, (k, "m", 1.0, 3))
            variables.append({"name": k, "label": lab, "unit": unit, "decimals": dec,
                              "min": a * scale, "max": b * scale, "value": nominal[k] * scale})
        return {"variables": variables, "plate_diameter_mm": space.plate_diameter * 1e3,
                "load_reference_n": space.reference_load,
                "metrics": self.predictor.s.metrics}


def serve(surrogate_path: str | Path, port: int = 8131, *, host: str = "127.0.0.1", workdir: str = "run/dashboard",
          block: bool = True):
    """Serve the dashboard. Returns the running server when ``block=False``."""
    from .surrogate import Surrogate
    _Handler.predictor = Predictor(Surrogate.load(surrogate_path))
    _Handler.workdir = workdir
    Path(workdir).mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{httpd.server_address[1]}/"
    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        httpd.url = url                                                 # type: ignore[attr-defined]
        return httpd
    print(f"PINNeAPPle structural surrogate: {url}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return None


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="pinneapple-structural-dashboard")
    ap.add_argument("surrogate", help="path to the trained surrogate (without .pt/.json)")
    ap.add_argument("--port", type=int, default=8131)
    ap.add_argument("--workdir", default="run/dashboard")
    a = ap.parse_args(argv)
    serve(a.surrogate, a.port, workdir=a.workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
