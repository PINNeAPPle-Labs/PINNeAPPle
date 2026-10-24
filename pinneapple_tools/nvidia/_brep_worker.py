"""USD BRep worker: builds a part from a recipe and measures it. Standalone (usd_brep + numpy + stdlib only), so it
runs in whichever interpreter has ``usd-brep`` (its wheel is CPython 3.12):

    python _brep_worker.py request.json result.npz

The request is ``{"recipe": {"ops": [...], "result": name}, "options": {...}}`` (see ``pinneapple_tools.nvidia.brep``).
"""
import json
import sys

import numpy as np
import usd_brep as sm


def _build(recipe):
    objs = {}
    for op in recipe["ops"]:
        k, n = op["op"], op["name"]
        if k == "box":
            objs[n] = sm.create_box(tuple(op["origin"]), *op["size"])
        elif k == "cylinder":
            objs[n] = sm.create_cylinder(tuple(op["origin"]), op["radius"], op["height"])
        elif k == "sphere":
            objs[n] = sm.create_sphere(tuple(op["origin"]), op["radius"])
        elif k == "union":
            objs[n] = sm.boolean_union(objs[op["a"]], objs[op["b"]])
        elif k == "difference":
            objs[n] = sm.boolean_difference(objs[op["a"]], objs[op["b"]])
        elif k == "translate":
            objs[n] = sm.translate(sm.brep_copy(objs[op["src"]]), tuple(op["vector"]))
        elif k == "rotate":
            objs[n] = sm.rotate(sm.brep_copy(objs[op["src"]]), tuple(op["point"]), tuple(op["axis"]), op["angle_deg"])
        elif k == "fillet":
            objs[n] = sm.circular_fillet(objs[op["src"]], op["radius"])
        elif k == "fillet_edges_near":
            b = objs[op["src"]]
            edges = []
            for p in op["points"]:
                i, _ = sm.find_edge(b, tuple(p))
                edges.append(b.edges()[i])
            objs[n] = sm.fillet_edges(b, edges, radius=op["radius"])
        elif k == "extrude_polygon":
            curves = [sm.create_line_segment(tuple(a), tuple(b))
                      for a, b in zip(op["points"], op["points"][1:] + op["points"][:1], strict=True)]
            objs[n] = sm.linear_sweep(curves, tuple(op["direction"]), op["distance"])
        else:
            raise ValueError(f"unknown op {k!r}")
    return objs[recipe["result"]]


def _face_curvature(face, n=7, h=1e-4):
    """Largest normal curvature found on an n x n grid of the face's parameter domain (finite differences of the
    unit normal), and where."""
    s = face.surface()
    (u0, v0), (u1, v1) = face.uv_domain()                 # (min corner, max corner) of the parameter box
    best, where = 0.0, None
    for u in np.linspace(u0, u1, n + 2)[1:-1]:
        for v in np.linspace(v0, v1, n + 2)[1:-1]:
            if face.classify_uv(u, v) != "inside":
                continue
            p = np.array(s.evaluate(u, v))
            nn = np.array(s.normal(u, v))
            nn /= np.linalg.norm(nn)
            for du, dv in ((h * (u1 - u0), 0.0), (0.0, h * (v1 - v0))):
                p2 = np.array(s.evaluate(u + du, v + dv))
                n2 = np.array(s.normal(u + du, v + dv))
                n2 /= np.linalg.norm(n2)
                d = np.linalg.norm(p2 - p)
                if d > 0:
                    k = np.linalg.norm(n2 - nn) / d
                    if k > best:
                        best, where = k, p
    return best, where


def main(req_path, out_path):
    req = json.load(open(req_path))
    opt = req.get("options", {})
    b = _build(req["recipe"])
    acc = opt.get("relative_accuracy", 1e-6)
    out = {"faces": b.face_count(), "edges": b.edge_count(), "vertices": b.vertex_count(),
           "manifold": bool(b.is_manifold_solid()), "volume": b.volume(acc), "area": b.area(acc),
           "census": {str(k): int(v) for k, v in b.material_census().items()}}
    mp = b.mass_properties(acc, opt.get("density", 1.0))
    out["mass"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in mp.items()}
    lo, hi = b.bounding_box(True)
    out["bbox"] = [list(lo), list(hi)]
    arrays = {}
    curv = []
    for i, f in enumerate(b.faces()):
        k, where = _face_curvature(f)
        curv.append({"face": i, "area": f.area(acc), "max_curvature": k,
                     "where": None if where is None else where.tolist()})
    out["curvature"] = curv
    tess = []
    for j, tol in enumerate(opt.get("chord_tolerances", [1e-3])):
        pb = b.tessellate(chord_height_tolerance=tol, curve_angle_tolerance_deg=opt.get("angle_deg", 25.0),
                          surface_angle_tolerance_deg=opt.get("angle_deg", 25.0),
                          max_edge_length=opt.get("max_edge_length", 0.0))
        a = pb.to_numpy_arrays()
        V = np.asarray(a["points"], float).reshape(-1, 3)
        flat = np.asarray(a["faces"], int)                  # [n, i0, ..., i_{n-1}] per polygon
        F, k0 = [], 0
        while k0 < len(flat):
            c = flat[k0]
            poly = flat[k0 + 1:k0 + 1 + c]
            F += [[poly[0], poly[m], poly[m + 1]] for m in range(1, c - 1)]
            k0 += c + 1
        F = np.asarray(F, int)
        C = V[F].mean(1)                                   # triangle centroids: where the chord error lives
        sample = C if len(C) <= opt.get("max_samples", 4000) else C[np.linspace(0, len(C) - 1,
                                                                               opt.get("max_samples", 4000)).astype(int)]
        dist = np.array([sm.get_closest_point(b, tuple(map(float, p)))[1] for p in sample])
        pm = pb.mass_properties(1.0)
        tess.append({"chord_tolerance": tol, "triangles": int(len(F)), "points": int(len(V)),
                     "max_deviation": float(dist.max()), "rms_deviation": float(np.sqrt(np.mean(dist ** 2))),
                     "mesh_volume": float(pm["volume"]), "mesh_area": float(pm["area"]),
                     "manifold": bool(pb.is_manifold_solid())})
        arrays[f"V{j}"], arrays[f"F{j}"] = V, F
    out["tessellations"] = tess
    if opt.get("vertex_curvature") and arrays:
        V = arrays[f"V{len(tess) - 1}"]
        faces = b.faces()
        kv, fid = np.zeros(len(V)), np.zeros(len(V), int)
        for i, p in enumerate(V):
            f = sm.find_face(b, tuple(map(float, p)))
            fid[i] = next(j for j, g in enumerate(faces) if g == f)
            s = f.surface()
            _, uvs, _ = sm.surface_closest_point(s, tuple(map(float, p)))
            u, v = uvs[0].x, uvs[0].y
            (u0, v0), (u1, v1) = s.uv_domain()
            nn = np.array(s.normal(u, v))
            nn /= np.linalg.norm(nn)
            pp = np.array(s.evaluate(u, v))
            k = 0.0
            for du, dv in ((1e-4 * (u1 - u0), 0.0), (0.0, 1e-4 * (v1 - v0))):
                sg = -1.0 if (u + du > u1 or v + dv > v1) else 1.0       # step inwards at the domain's far edge
                p2 = np.array(s.evaluate(u + sg * du, v + sg * dv))
                n2 = np.array(s.normal(u + sg * du, v + sg * dv))
                d = np.linalg.norm(p2 - pp)
                if d > 0:
                    k = max(k, np.linalg.norm(n2 / np.linalg.norm(n2) - nn) / d)
            kv[i] = k
        arrays["vertex_curvature"], arrays["vertex_face"] = kv, fid
    if opt.get("usd_brep_path"):
        from usd_brep import usd as busd
        busd.export_brep(b.copy(), opt["usd_brep_path"])
        out["usd_brep_path"] = opt["usd_brep_path"]
    if opt.get("native_path"):
        b.write_to_file(opt["native_path"])
    arrays["report"] = np.frombuffer(json.dumps(out).encode(), dtype=np.uint8)
    np.savez(out_path, **arrays)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
