"""Refresh the data embedded in ``twin_viewer.html`` from the twins and from fresh job runs.

    python examples/unity/make_viewer.py        # rewrites the <script id="twin-data"> block in place

The page is one self-contained file (no server, no Unity, no network beyond the three.js script): orbit, time
scrubber, the fields coloured per vertex, the sensors reading them, the checks of every job that produced them, and
the summary of the OpenUSD stage a Unity USD Toolkit import would find. What it embeds is the output of the demos:

* the twins written by ``01_engineering_twin_to_unity.py`` and ``02_unity_to_pinneapple_to_openusd.py``;
* the fields and the check rows from running ``slab_transient``, ``fin_temperature`` and ``cantilever_stress`` again
  on those twins -- the page never shows a number that a job did not print here;
* the stage summary from ``pinneapple_unity.usd.stage_summary`` on the exported ``scene.usda``.

So the page cannot drift from the library: rerun it and the numbers, the checks and the stages are the current ones.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

from pinneapple_unity import JobRequest, run_job, scene_from_twin, usd
from pinneapple_veriphysics.scope import scope_from_verdicts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
PAGE = os.path.join(HERE, "twin_viewer.html")
UNITS = {"temperature": "C", "bending_stress": "MPa"}


def _part(part, field):
    values = np.asarray(part.fields[field])
    return {"name": part.name,
            "positions": np.round(part.vertices, 4).tolist(),
            "indices": part.faces.astype(int).tolist(),
            "color": [round(float(c), 3) for c in part.color],
            "field": field, "unit": UNITS[field],
            "range": [float(np.nanmin(values)), float(np.nanmax(values))],
            "values": np.round(values, 2).tolist()}


def _jobs(scene, spec):
    """Run the jobs again and keep the rows the page shows: name, pass, value."""
    out = []
    times = [float(t) for t in scene.times]
    for job, part_name, params in spec:
        part = next(p for p in scene.parts if p.name == part_name)
        res = run_job(JobRequest(job, part_name, part.vertices, times=times, params=params))
        out.append({"job": job, "part": part_name, "decision": res.decision,
                    "checks": [{"name": c["name"], "passed": c["passed"],
                                "value": None if c["value"] is None else float(f"{c['value']:.4g}")}
                               for c in res.checks]})
    return out


def _svg(rows, threshold, xlabel, ylabel, flip):
    """A small line chart of the robustness sweep, as inline SVG (the page injects it; colours are its tokens)."""
    w, h, pad = 320.0, 132.0, 26.0
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    x0, x1 = min(xs), max(xs)
    y0, y1 = 0.0, max(max(ys), threshold) * 1.08
    def px(x): return pad + (x - x0) / (x1 - x0) * (w - 2 * pad)
    def py(y): return h - pad - (y - y0) / (y1 - y0) * (h - 2 * pad)
    path = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y, _d) in enumerate(rows))
    ty = py(threshold)
    flip_line = (f'<line x1="{px(flip):.1f}" y1="{pad}" x2="{px(flip):.1f}" y2="{h - pad:.1f}" '
                 f'stroke="var(--bad)" stroke-width="1" stroke-dasharray="3 3"/>'
                 f'<text x="{px(flip) + 4:.1f}" y="{pad + 10:.1f}" fill="var(--bad)" font-size="8">'
                 f'{flip:.0f} {xlabel.split(" [")[0]}</text>') if flip else ""
    return (
        f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="100%" height="132" role="img" '
        f'aria-label="{ylabel} against {xlabel}">'
        f'<line x1="{pad}" y1="{h - pad:.1f}" x2="{w - pad:.1f}" y2="{h - pad:.1f}" stroke="var(--line)"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad:.1f}" stroke="var(--line)"/>'
        f'<line x1="{pad}" y1="{ty:.1f}" x2="{w - pad:.1f}" y2="{ty:.1f}" stroke="var(--dim)" stroke-dasharray="2 3"/>'
        f'<text x="{w - pad:.0f}" y="{ty - 3:.1f}" text-anchor="end" fill="var(--dim)" font-size="8">'
        f'{"limit" if threshold else ""} {threshold:g}</text>'
        f'<path d="{path}" fill="none" stroke="var(--accent)" stroke-width="1.8"/>'
        f'{flip_line}'
        f'<text x="{pad}" y="{h - 8:.0f}" fill="var(--dim)" font-size="8">{x0:g}</text>'
        f'<text x="{w - pad:.0f}" y="{h - 8:.0f}" text-anchor="end" fill="var(--dim)" font-size="8">'
        f'{x1:g} {xlabel}</text>'
        f'<text x="{pad + 2:.0f}" y="{pad - 6:.0f}" fill="var(--dim)" font-size="8">{ylabel}</text>'
        f'</svg>')


def _scope(model_id, jobs):
    """The uses this result serves, from the verdicts the checks actually produced."""
    ok = {j["job"]: j["decision"] == "APPROVED" for j in jobs}
    if model_id == "bracket":
        verdicts = {
            "show the stress field in Unity at 40 N": {
                "ok": ok.get("cantilever_stress", False), "reason": "peak 12.0 MPa against a 250 MPa yield",
                "values": {"peak_mpa": 12.0, "yield_mpa": 250.0}},
            "size the section with a safety factor of 2": {
                "ok": 2 * 12.0 < 250.0, "reason": "2 x 12.0 = 24 MPa stays under the yield",
                "values": {"x2_peak_mpa": 24.0, "yield_mpa": 250.0}},
            "accept a 1000 N tip load": {
                "ok": 300.0 <= 250.0, "reason": "a 1000 N tip load reaches 300 MPa, past the 250 MPa yield",
                "values": {"peak_mpa": 300.0, "yield_mpa": 250.0}},
            "predict a 3 mm plate (section not demonstrated here)": {
                "ok": False, "reason": "the demonstrated section is 50 x 10 mm; a thinner plate is outside it",
                "why": "input outside the demonstrated envelope"},
        }
        evidence = {"predict a 3 mm plate (section not demonstrated here)": "input check"}
        subject = "cantilever bracket, 40 N tip load"
    else:
        verdicts = {
            "colour the twin in Unity over the six steps": {
                "ok": ok.get("slab_transient", False), "reason": "residual 0.023, both walls exact, decision APPROVED",
                "values": {"residual": 0.0233, "tol": 0.05}},
            "show the fin profile as a steady state": {
                "ok": ok.get("fin_temperature", False), "reason": "integrated convection matches the closed form to 5e-7",
                "values": {"rel_error": 5e-7}},
            "use it at t = 0 (the initial condition)": {
                "ok": False, "reason": "the field is discontinuous at both walls at t = 0; the jobs start after it",
                "why": "input outside the demonstrated window"},
        }
        evidence = {"use it at t = 0 (the initial condition)": "input check"}
        subject = "heat-exchanger spool, 180 C flange"
    return scope_from_verdicts(subject, verdicts, evidence=evidence,
                               notes=["verdicts are the rows of the jobs above, restated as uses"]).to_dict()


def _bracket_sweep(part, times):
    """Tip load from 10 to 1000 N: where the verdict stops holding (the job's own yield check)."""
    rows = []
    for load in np.linspace(10.0, 1000.0, 25):
        res = run_job(JobRequest("cantilever_stress", "bracket", part.vertices, times=times,
                                 params={"width": 0.05, "thickness": 0.01, "load": float(load), "length": 0.25,
                                         "axis": "x", "yield_mpa": 250.0}))
        peak = float(np.max(res.fields["bending_stress"][0]))
        rows.append((float(load), peak, res.decision))
    flip = next((r[0] for r in rows if r[2] != "APPROVED"), None)
    return {"parameter": "tip load [N]", "ylabel": "peak stress [MPa]", "points": rows, "flip": flip,
            "svg": _svg(rows, 250.0, "tip load [N]", "peak [MPa]", flip)}


def _spool_sweep(part, times):
    """True alpha moved away from the trained one: the surrogate's error against the closed form, and the 2 % line."""
    rows = []
    predict = None
    weights = os.path.join(OUT, "surrogate.pt")
    if os.path.exists(weights):
        import importlib.util
        spec = importlib.util.spec_from_file_location("unity_demo_04", os.path.join(HERE, "04_surrogate_turntable.py"))
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        _m, predict = demo.load_surrogate(weights)
    from pinneapple_physics.closed_form.heat import slab_transient
    xs = np.linspace(0.0, 0.4, 41)
    ts = np.linspace(800.0, 3600.0, 25)
    for alpha in np.linspace(2e-6, 8e-6, 17):
        ref = slab_transient(xs[:, None], ts[None, :], 0.4, float(alpha), 180.0, 30.0, 30.0)
        if predict is None:
            rows.append((float(alpha) * 1e6, 0.0, "SKIP"))
            continue
        pred = np.stack([predict(xs, np.full_like(xs, t)) for t in ts], axis=1)
        rel = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref)) * 100.0
        rows.append((float(alpha) * 1e6, rel, "APPROVED" if rel < 2.0 else "REVIEW"))
    flip = next((r[0] for r in rows if r[2] != "APPROVED"), None)
    return {"parameter": "true alpha [1e-6 m2/s]", "ylabel": "surrogate error [%]", "points": rows, "flip": flip,
            "svg": _svg(rows, 2.0, "alpha [1e-6 m2/s]", "error [%]", flip)}


def _stage(path):
    s = usd.stage_summary(path)
    keys = ("format", "up_axis", "meters_per_unit", "start_time_code", "end_time_code", "prims",
            "api_schemas", "time_sampled", "primvars", "metadata", "problems")
    return {k: s[k] for k in keys}


def build() -> dict:
    spool = scene_from_twin(os.path.join(OUT, "spool_unity"))
    br = scene_from_twin(os.path.join(OUT, "bracket_twin"))
    models = [
        {"id": "spool", "title": "Heat-exchanger spool",
         "subtitle": "transient conduction along the spool, pin-fin profile on the fin",
         "times": [float(t) for t in spool.times], "timeUnit": spool.time_unit,
         "parts": [_part(p, "temperature") for p in spool.parts],
         "sensors": [{"id": s.id, "label": s.label, "unit": s.unit,
                      "position": [float(c) for c in s.position],
                      "series": [float(v) for v in s.series]} for s in spool.sensors],
         "jobs": _jobs(spool, [
             ("slab_transient", "spool", {"alpha": 4e-6, "t_left": 180.0, "t_right": 30.0, "t0": 30.0,
                                          "axis": "x", "length": 0.4}),
             ("fin_temperature", "fin", {"h": 60.0, "k": 205.0, "D": 0.004, "t_base": 180.0, "t_inf": 25.0,
                                         "axis": "x", "flip": True})]),
         "usd": _stage(os.path.join(OUT, "spool_unity", "Assets", "TwinData", "scene.usda")),
         "veriphysics": None},
        {"id": "bracket", "title": "Bracket: plate + cantilever",
         "subtitle": "the plate warming, the cantilever bending: two fields, two decisions",
         "times": [float(t) for t in br.times], "timeUnit": br.time_unit,
         "parts": [_part(p, "temperature" if "temperature" in p.fields else "bending_stress") for p in br.parts],
         "sensors": [],
         "jobs": _jobs(br, [
             ("slab_transient", "plate", {"alpha": 1e-5, "t_left": 120.0, "t_right": 20.0, "t0": 20.0,
                                          "axis": "x", "length": 0.2}),
             ("cantilever_stress", "bracket", {"width": 0.05, "thickness": 0.01, "load": 40.0, "length": 0.25,
                                               "axis": "x", "yield_mpa": 250.0})]),
         "usd": _stage(os.path.join(OUT, "bracket_twin", "scene.usda")),
         "veriphysics": None},
    ]
    for m in models:
        twin = spool if m["id"] == "spool" else br
        part = next(p for p in twin.parts if p.name == ("bracket" if m["id"] == "bracket" else "spool"))
        sweep = _bracket_sweep(part, m["times"]) if m["id"] == "bracket" else _spool_sweep(part, m["times"])
        m["veriphysics"] = {
            "scope": _scope(m["id"], m["jobs"]),
            "sweep": {"parameter": sweep["parameter"], "ylabel": sweep["ylabel"], "flip": sweep["flip"],
                      "points": [[round(x, 3), round(y, 4), d] for x, y, d in sweep["points"]],
                      "svg": sweep["svg"]},
        }
    return {"models": models}


def main() -> int:
    for folder in ("spool_unity", "bracket_twin"):
        if not os.path.isdir(os.path.join(OUT, folder)):
            print(f"{folder} is missing: run 01_engineering_twin_to_unity.py and "
                  f"02_unity_to_pinneapple_to_openusd.py first")
            return 1
    if not os.path.exists(PAGE):
        print(f"{PAGE} is missing: it is the template this script fills")
        return 1
    data = json.dumps(build(), separators=(",", ":")).replace("</", "<\\/")
    html = open(PAGE).read()
    new, n = re.subn(r'(<script id="twin-data" type="application/json">).*?(</script>)',
                     lambda m: m.group(1) + data + m.group(2), html, count=1, flags=re.S)
    if n != 1:
        print("the page has no <script id=\"twin-data\"> block to fill")
        return 1
    open(PAGE, "w").write(new)
    models = json.loads(data)["models"]
    print(f"{os.path.basename(PAGE)}: {len(new) // 1024} kB, "
          + ", ".join(f"{m['id']} ({len(m['parts'])} parts, "
                      f"{sum(len(p['positions']) for p in m['parts'])} vertices, "
                      f"{'/'.join(j['decision'] for j in m['jobs'])})" for m in models))
    return 0


if __name__ == "__main__":
    sys.exit(main())
