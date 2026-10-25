# Runs inside FreeCAD's own Python (freecadcmd fc_job.py, job file in PINNEAPPLE_FC_JOB); nothing from PINNeAPPle is
# imported here. Two jobs:
#   build:  {"job": "build", "template": name, "parameters": {alias: value}, "output": path.FCStd}
#           writes a parametric document: a spreadsheet "Params" (one alias per parameter, values in mm / deg) and
#           Part primitives whose dimensions and placements are expressions of it.
#   export: {"job": "export", "document": path, "parameters": {alias: value}, "solid": name, "output_dir": dir,
#            "tolerance_mm": 0.05}
#           sets the aliases, recomputes, checks every object, saves the modified document, exports the solid to
#           STEP and its faces (triangulated, outward oriented, with surface type, area and centre) to faces.json.
# The result is written to the job's "result" path: {"ok": bool, "error": traceback or None, ...}.
import json
import os
import sys
import traceback

import FreeCAD as App

JOB = json.load(open(os.environ["PINNEAPPLE_FC_JOB"]))


def sheet(doc, parameters):
    s = doc.addObject("Spreadsheet::Sheet", "Params")
    for i, (k, v) in enumerate(parameters.items()):
        s.set(f"A{i + 1}", k)
        s.set(f"B{i + 1}", repr(float(v)))
        s.setAlias(f"B{i + 1}", k)
    doc.recompute()
    return s


def cylinder(doc, name, radius, height, placement):
    c = doc.addObject("Part::Cylinder", name)
    c.setExpression("Radius", radius)
    c.setExpression("Height", height)
    c.setExpression("Placement", placement)
    return c


def build_pipe_bend(doc, p):
    """Straight inlet leg along +x ending at the origin, a bend of radius Rc (centre line) through `angle` degrees
    about the z axis, a straight outlet leg; diameter D."""
    sheet(doc, p)
    inlet = cylinder(doc, "Inlet", "Params.D / 2", "Params.L_in",
                     "placement(vector(-Params.L_in; 0; 0); rotation(vector(0; 1; 0); 90))")
    bend = doc.addObject("Part::Torus", "Bend")
    bend.setExpression("Radius1", "Params.Rc")
    bend.setExpression("Radius2", "Params.D / 2")
    bend.setExpression("Angle3", "Params.angle")
    bend.setExpression("Placement", "placement(vector(0; Params.Rc; 0); rotation(vector(0; 0; 1); -90))")
    outlet = cylinder(doc, "Outlet", "Params.D / 2", "Params.L_out",
                      "placement(vector(Params.Rc * sin(Params.angle); Params.Rc - Params.Rc * cos(Params.angle); 0);"
                      " rotation(vector(0; 0; 1); Params.angle) * rotation(vector(0; 1; 0); 90))")
    fluid = doc.addObject("Part::MultiFuse", "Fluid")
    fluid.Shapes = [inlet, bend, outlet]
    return fluid


def build_heated_pipe(doc, p):
    """Straight pipe along +x of diameter D: an unheated entry length L_entry from x = 0, then a heated length
    L_heated (two cylinders, so the wall has one face per section)."""
    sheet(doc, p)
    entry = cylinder(doc, "Entry", "Params.D / 2", "Params.L_entry", "placement(vector(0; 0; 0); "
                     "rotation(vector(0; 1; 0); 90))")
    heated = cylinder(doc, "Heated", "Params.D / 2", "Params.L_heated",
                      "placement(vector(Params.L_entry; 0; 0); rotation(vector(0; 1; 0); 90))")
    fluid = doc.addObject("Part::MultiFuse", "Fluid")
    fluid.Shapes = [entry, heated]
    return fluid


def build_spheroid(doc, p):
    """Spheroid centred at the origin, length L along x (the flow direction), diameter D across (L = D: sphere)."""
    sheet(doc, p)
    body = doc.addObject("Part::Ellipsoid", "Body")
    body.setExpression("Radius1", "Params.L / 2")
    body.setExpression("Radius2", "Params.D / 2")
    body.Radius3 = 0
    body.setExpression("Placement", "placement(vector(0; 0; 0); rotation(vector(0; 1; 0); 90))")
    return body


TEMPLATES = {"pipe_bend": build_pipe_bend, "heated_pipe": build_heated_pipe, "spheroid": build_spheroid}


def set_parameters(doc, parameters):
    sheets = [o for o in doc.Objects if o.TypeId == "Spreadsheet::Sheet"]
    known = {}
    for s in sheets:
        for cell in s.getUsedCells():
            a = s.getAlias(cell)
            if a:
                known[a] = (s, cell)
    missing = [k for k in parameters if k not in known]
    if missing:
        raise KeyError(f"no spreadsheet alias {missing}; the document has {sorted(known)}")
    for k, v in parameters.items():
        s, cell = known[k]
        s.set(cell, repr(float(v)))
    doc.recompute()
    values = {}
    for k, (s, _cell) in known.items():
        v = s.get(k)
        values[k] = float(getattr(v, "Value", v))
    return values


def faces_of(shape, tol):
    shape.tessellate(tol)                      # one triangulation of the solid: shared edges get shared points
    out = []
    for i, f in enumerate(shape.Faces):
        pts, tris = f.tessellate(tol)
        P = [[v.x, v.y, v.z] for v in pts]
        T = [list(t) for t in tris]
        if T:                                  # orient outward: a point just off the first triangle must be outside
            a, b, c = (App.Vector(*P[j]) for j in T[0])
            n = (b - a).cross(c - a)
            if n.Length > 0:
                n.normalize()
                cen = (a + b + c) * (1.0 / 3.0)
                eps = max(1e-3, 1e-4 * shape.BoundBox.DiagonalLength)
                if shape.isInside(cen + n * eps, eps * 0.1, False):
                    T = [[t[0], t[2], t[1]] for t in T]
        com = f.CenterOfMass
        out.append({"index": i, "type": f.Surface.__class__.__name__, "area": f.Area,
                    "centre": [com.x, com.y, com.z], "points": P, "triangles": T})
    return out


def main():
    res = {"ok": False, "error": None, "freecad": ".".join(App.Version()[:3])}
    try:
        if JOB["job"] == "build":
            doc = App.newDocument(JOB["template"])
            obj = TEMPLATES[JOB["template"]](doc, JOB["parameters"])
            doc.recompute()
            bad = [o.Name for o in doc.Objects if "Invalid" in o.State or "Error" in o.State]
            if bad:
                raise RuntimeError(f"objects failed to recompute: {bad}")
            doc.saveAs(JOB["output"])
            res.update(ok=True, document=JOB["output"], solid=obj.Name, volume_mm3=obj.Shape.Volume)
        elif JOB["job"] == "export":
            doc = App.openDocument(JOB["document"])
            doc.recompute()
            values = set_parameters(doc, JOB.get("parameters") or {})
            bad = [o.Name for o in doc.Objects if "Invalid" in o.State or "Error" in o.State]
            obj = doc.getObject(JOB["solid"])
            if obj is None:
                raise KeyError(f"no object {JOB['solid']!r} in {JOB['document']}")
            shape = obj.Shape
            out = JOB["output_dir"]
            os.makedirs(out, exist_ok=True)
            step = os.path.join(out, "geometry.step")
            shape.exportStep(step)
            fc = os.path.join(out, "model.FCStd")
            doc.saveAs(fc)
            faces = faces_of(shape, float(JOB.get("tolerance_mm", 0.05)))
            with open(os.path.join(out, "faces.json"), "w") as f:
                json.dump(faces, f)
            bb = shape.BoundBox
            res.update(ok=True, parameters=values, invalid_objects=bad, valid=shape.isValid(),
                       closed=all(s.isClosed() for s in shape.Shells), solids=len(shape.Solids),
                       volume_mm3=shape.Volume, area_mm2=shape.Area, faces=len(shape.Faces),
                       bound_box_mm=[bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax], step=step, document=fc,
                       faces_json=os.path.join(out, "faces.json"))
        else:
            raise ValueError(f"unknown job {JOB['job']!r}")
    except Exception:
        res["error"] = traceback.format_exc()
    with open(JOB["result"], "w") as f:
        json.dump(res, f, indent=1)


main()
sys.stdout.flush()
