"""Auto-detect a project's lineage from its files: OpenFOAM cases (blockMeshDict → mesh → run), Gmsh .geo → .msh,
CalculiX decks and results, datasets exported by the Interoperability Hub (linked to their source files by sha256),
Comparator / Mesh Quality / Preflight reports, PINNeAPPle model cards, and a declared lineage.json that annotates or
overrides what was detected.
"""
from __future__ import annotations

import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .graph import Artifact, Lineage, sha256

LOG_DATE = re.compile(r"^Date\s*:\s*(\w{3})\s+(\d{1,2})\s+(\d{4})", re.M)
LOG_TIME = re.compile(r"^Time\s*:\s*(\d{2}:\d{2}:\d{2})", re.M)
LOG_VER = re.compile(r"^(?:Version|Build)\s*:\s*(\S+)", re.M)
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
SKIP = re.compile(r"(^|/)(__MACOSX|\.git|\.DS_Store)|(^|/)\._")


def _log_info(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    v = LOG_VER.search(text)
    if v:
        b = re.match(r"OPENFOAM=(\d+)", v.group(1))
        out["version"] = f"v{b.group(1)}" if b else v.group(1)
    d, t = LOG_DATE.search(text), LOG_TIME.search(text)
    if d:
        out["timestamp"] = f"{d.group(3)}-{MONTHS.get(d.group(1), 1):02d}-{int(d.group(2)):02d}T{t.group(1) if t else '00:00:00'}+00:00"
    return out


def _txt(b: Optional[bytes]) -> str:
    return b.decode("latin-1") if b else ""


def detect_project(fs: Dict[str, bytes], project: str = "") -> Lineage:
    fs = {k: v for k, v in fs.items() if not SKIP.search(k)}
    g = Lineage(project=project)
    owner_of: Dict[str, str] = {}                 # file path -> artifact id that contains it
    hashes = {k: sha256(v) for k, v in fs.items()}
    used: set = set()

    def mark(aid: str, paths: List[str]):
        for p in paths:
            owner_of[p] = aid
            used.add(p)

    # ---------------------------------------------------------------- OpenFOAM cases
    roots = sorted({k[: -len("system/controlDict")] for k in fs if k.endswith("system/controlDict")})
    for root in roots:
        name = root.rstrip("/") or (project or "case")
        files = [k for k in fs if k.startswith(root)]
        ctrl = _txt(fs.get(root + "system/controlDict"))
        app = (re.search(r"^\s*application\s+(\w+)\s*;", ctrl, re.M) or [None, None])[1]
        geo_id = mesh_id = None
        bmd = root + "system/blockMeshDict"
        if bmd in fs:
            geo_id = f"{name}/blockMeshDict"
            g.add(Artifact(id=geo_id, kind="geometry", name=f"{name}: geometry and blocking (blockMeshDict)", file=bmd,
                           hash=hashes[bmd], actual_hash=hashes[bmd], software="OpenFOAM blockMesh", item=f"{name} geometry",
                           detected="OpenFOAM system/blockMeshDict"))
            mark(geo_id, [bmd])
        stls = [k for k in files if "/triSurface/" in k and k.lower().endswith((".stl", ".obj"))]
        geo_inputs = [geo_id] if geo_id else []
        for s in stls:
            sid = os.path.basename(s)
            g.add(Artifact(id=sid, kind="geometry", name=f"surface {sid}", file=s, hash=hashes[s], actual_hash=hashes[s],
                           detected="snappyHexMesh triSurface"))
            mark(sid, [s])
            geo_inputs.append(sid)
        pm = [k for k in files if "/constant/polyMesh/" in "/" + k[len(root) - 1:] or k.startswith(root + "constant/polyMesh/")]
        if pm:
            mesh_id = f"{name}/polyMesh"
            core = b"".join(fs[k] for k in sorted(pm) if k.split("/")[-1].removesuffix(".gz") in ("points", "faces", "owner", "neighbour"))
            owner_txt = _txt(fs.get(root + "constant/polyMesh/owner", b"")[:2000])
            note = re.search(r'note\s+"([^"]*)"', owner_txt)
            params = dict(re.findall(r"(n\w+):(\d+)", note.group(1))) if note else {}
            params = {k: int(v) for k, v in params.items()}
            log = _txt(fs.get(root + "log.blockMesh")) or _txt(fs.get(root + "log.snappyHexMesh")) or _txt(fs.get(root + "log.gmshToFoam"))
            li = _log_info(log)
            src = geo_inputs[:]
            gm = re.search(r"gmshToFoam\s+(\S+\.msh)", log)
            if gm:
                src.append(os.path.basename(gm.group(1)))
            soft = "OpenFOAM snappyHexMesh" if root + "log.snappyHexMesh" in fs else "OpenFOAM gmshToFoam" if gm else "OpenFOAM blockMesh"
            g.add(Artifact(id=mesh_id, kind="mesh", name=f"{name}: mesh ({params.get('nCells', '?')} cells)",
                           file=root + "constant/polyMesh", hash=sha256(core), software=soft, version=li.get("version"),
                           timestamp=li.get("timestamp"), parameters=params, inputs=src, item=f"{name} mesh",
                           detected="OpenFOAM constant/polyMesh"))
            mark(mesh_id, pm + [root + "log.blockMesh", root + "log.checkMesh"])
        logs = [k for k in files if re.search(r"/?log\.\w+Foam$", k)]
        li = _log_info(_txt(fs.get(logs[0]))) if logs else {}
        params: Dict[str, Any] = {}
        try:
            from pinneapple_data.simulation_metadata import extract
            rec = extract({k[len(root):]: v for k, v in fs.items() if k.startswith(root)})
            an, cv = rec.get("analysis") or {}, rec.get("convergence") or {}
            tb = an.get("turbulence") or {}
            params = {k: v for k, v in {"application": app,
                                        "turbulence": " ".join(x for x in (tb.get("type"), tb.get("model")) if x) if isinstance(tb, dict) else tb,
                                        "algorithm": an.get("algorithm"), "steady": an.get("steady"),
                                        "convergence": cv.get("status"),
                                        "iterations": (rec.get("time") or {}).get("steps_run")}.items() if v not in (None, "")}
            for prm in rec.get("parameters") or []:
                if isinstance(prm, dict) and prm.get("name") and len(params) < 12:
                    params[prm["name"]] = prm.get("value")
        except Exception:                                             # metadata is a bonus; the lineage stands without it
            params = {"application": app} if app else {}
        times = sorted({k[len(root):].split("/")[0] for k in files if re.match(r"\d+(\.\d+)?/", k[len(root):]) and
                        not k[len(root):].startswith("0/")}, key=float)
        if times:
            params["result_times"] = times
            params["latest_time"] = times[-1]
        sim_files = [k for k in files if k not in owner_of]
        g.add(Artifact(id=name, kind="simulation", name=f"{name}: {app or 'OpenFOAM'} run", file=root or ".",
                       software=f"OpenFOAM {app}" if app else "OpenFOAM", version=li.get("version"),
                       timestamp=li.get("timestamp"), parameters=params, inputs=[mesh_id] if mesh_id else [],
                       hash=sha256(b"".join(fs[k] for k in sorted(sim_files))), detected="OpenFOAM case (system/controlDict)"))
        mark(name, sim_files)
    # ---------------------------------------------------------------- Gmsh / CAD
    for k in sorted(fs):
        if k in used:
            continue
        low = k.lower()
        base, ext = os.path.splitext(os.path.basename(k))
        if ext in (".step", ".stp", ".iges", ".igs", ".brep", ".stl", ".obj") or ext == ".geo":
            g.add(Artifact(id=os.path.basename(k), kind="geometry", name=base, file=k, hash=hashes[k], actual_hash=hashes[k],
                           software="Gmsh" if ext == ".geo" else None, item=f"{base} geometry", detected=f"{ext} file"))
            mark(os.path.basename(k), [k])
            if ext == ".geo":
                for ref in re.findall(r'(?:Merge|ShapeFromFile)\s*\(?\s*"([^"]+)"', _txt(fs[k])):
                    g.nodes[os.path.basename(k)].inputs.append(os.path.basename(ref))
    for k in sorted(fs):
        if k in used:
            continue
        base, ext = os.path.splitext(os.path.basename(k))
        if ext == ".msh":
            head = _txt(fs[k][:300])
            ver = (re.search(r"\$MeshFormat\s+(\S+)", head) or [None, None])[1]
            ins = [n for n in g.nodes if n == f"{base}.geo" or (g.nodes[n].kind == "geometry" and os.path.splitext(n)[0] == base)]
            g.add(Artifact(id=os.path.basename(k), kind="mesh", name=f"{base} mesh", file=k, hash=hashes[k], actual_hash=hashes[k],
                           software="Gmsh", version=f"MSH {ver}" if ver else None, inputs=ins, item=f"{base} mesh",
                           detected=".msh file"))
            mark(os.path.basename(k), [k])
    # ---------------------------------------------------------------- CalculiX / Abaqus
    for k in sorted(fs):
        if k in used or not k.lower().endswith(".inp"):
            continue
        txt = _txt(fs[k])
        if "*STEP" not in txt.upper():
            continue
        base = os.path.splitext(os.path.basename(k))[0]
        incs = [os.path.basename(x) for x in re.findall(r"\*INCLUDE\s*,\s*INPUT\s*=\s*\"?([^\"\n,]+)", txt, re.I)]
        steps = [s.strip().upper() for s in re.findall(r"^\*(STATIC|FREQUENCY|HEAT TRANSFER|DYNAMIC|BUCKLE)", txt, re.M | re.I)]
        log = _txt(fs.get(os.path.join(os.path.dirname(k), "log.ccx")) or b"")
        ver = (re.search(r"Version\s+([\d.]+)", log) or [None, None])[1]
        for inc in incs:
            if inc not in g.nodes:
                path = next((p for p in fs if os.path.basename(p) == inc), None)
                if path:
                    g.add(Artifact(id=inc, kind="mesh", name=f"{inc} (included)", file=path, hash=hashes[path],
                                   actual_hash=hashes[path], detected="*INCLUDE of the deck"))
                    mark(inc, [path])
        g.add(Artifact(id=os.path.basename(k), kind="simulation", name=f"{base}: CalculiX deck", file=k, hash=hashes[k],
                       actual_hash=hashes[k], software="CalculiX", version=ver, parameters={"steps": steps},
                       inputs=incs, detected=".inp deck with *STEP"))
        mark(os.path.basename(k), [k])
        for ext in (".frd", ".dat"):
            r = next((p for p in fs if p not in used and os.path.basename(p) == base + ext), None)
            if r:
                head = _txt(fs[r][:3000])
                rv = (re.search(r"1UVERSION\s+Version\s+([\d.]+)", head) or [None, None])[1]
                g.add(Artifact(id=os.path.basename(r), kind="result", name=f"{base} results ({ext[1:]})", file=r,
                               hash=hashes[r], actual_hash=hashes[r], software="CalculiX", version=rv,
                               inputs=[os.path.basename(k)], detected=f"{ext} next to the deck"))
                mark(os.path.basename(r), [r])
    # ---------------------------------------------------------------- datasets and reports from the PINNeAPPle apps
    by_prefix: Dict[str, str] = {}
    for p, h in hashes.items():
        if p in owner_of:
            by_prefix[h[:16]] = owner_of[p]
    for k in sorted(fs):
        if k in used:
            continue
        meta = _pinneapple_meta(k, fs[k])
        if not meta:
            continue
        kind, info = meta
        ins: List[str] = []
        for f in info.get("files", []) or []:
            src = by_prefix.get(str(f.get("sha256", ""))[:16])
            if src and src not in ins:
                ins.append(src)
        for nm in info.get("names", []):
            stem = nm.split(".")[0]
            hit = next((a for a in g.nodes if a == nm or a == stem or g.nodes[a].file == nm
                        or os.path.basename(str(g.nodes[a].file)) == nm), None)
            if hit is None:
                hit = next((a for a in g.nodes if a.split("/")[0] == stem), None)
            if hit and hit not in ins:
                ins.append(hit)
        anc = set()
        for i in ins:
            anc.update(g.upstream(i))
        ins = [i for i in ins if i not in anc]                         # keep the most derived sources only
        g.add(Artifact(id=os.path.basename(k), kind=kind, name=info.get("title") or os.path.basename(k), file=k,
                       hash=hashes[k], actual_hash=hashes[k], software=info.get("software"), timestamp=info.get("timestamp"),
                       parameters=info.get("parameters", {}), inputs=ins, detected=info.get("rule")))
        mark(os.path.basename(k), [k])
    # ---------------------------------------------------------------- declared lineage (annotates / overrides)
    for k in sorted(fs):
        if os.path.basename(k).lower() in ("lineage.json",) or k.lower().endswith(".lineage.json"):
            try:
                decl = Lineage.from_json(json.loads(fs[k]))
            except Exception as e:
                g.add(Artifact(id=os.path.basename(k), kind="file", notes=f"unreadable lineage file: {e}"))
                continue
            g.project = g.project or decl.project
            for a in decl.nodes.values():
                if a.file:                                             # verify the declared hash against the file
                    path = next((p for p in fs if p == a.file or p.endswith("/" + a.file)), None)
                    if path:
                        a.actual_hash = hashes[path]
                elif a.id in g.nodes and g.nodes[a.id].actual_hash:
                    a.actual_hash = g.nodes[a.id].actual_hash
                if a.id in g.nodes and a.hash and g.nodes[a.id].hash and g.nodes[a.id].detected and a.hash != g.nodes[a.id].hash:
                    a.actual_hash = g.nodes[a.id].hash                 # detected content vs declared record
                g.add(a)
            used.add(k)
    # ---------------------------------------------------------------- everything else: unlinked files
    declared_files = {str(a.file) for a in g.nodes.values() if a.file}
    for k in sorted(fs):
        if k in used or k in owner_of or k in declared_files or any(k.endswith("/" + f) for f in declared_files):
            continue
        if re.search(r"(^|/)(0|0\.orig)/", k):
            continue
        g.add(Artifact(id=os.path.basename(k), kind="file", file=k, hash=hashes[k], actual_hash=hashes[k],
                       detected="unclassified file"))
    return g


def _pinneapple_meta(path: str, b: bytes) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Recognise files written by the PINNeAPPle apps (and model cards) and read their source references."""
    low = path.lower()
    try:
        if low.endswith(".json"):
            d = json.loads(b)
            if not isinstance(d, dict):
                return None
            if d.get("schema") == "pinneapple.physical_dataset/1":
                md = d.get("metadata", {})
                return "dataset", {"files": md.get("files"), "software": md.get("converter"), "timestamp": md.get("converted_at"),
                                   "title": f"dataset ({md.get('solver') or md.get('source_format')})",
                                   "parameters": {"fields": list(d.get("fields", {})), "time": md.get("time")},
                                   "rule": "Interoperability Hub dataset (JSON)"}
            if "fields" in d and "global" in d and "reference" in d and "candidate" in d:
                return "report", {"names": [os.path.basename(str(d["reference"].get("file", ""))),
                                            os.path.basename(str(d["candidate"].get("file", "")))],
                                  "software": "PINNeAPPle Simulation Comparator",
                                  "title": "comparison " + " vs ".join(os.path.basename(str(d[s].get("file", ""))) for s in ("reference", "candidate")),
                                  "parameters": {f["reference"]: f"{100 * f['summary']['rel_l2']:.2f} %" for f in d["fields"]
                                                 if f.get("summary", {}).get("rel_l2") is not None},
                                  "rule": "Simulation Comparator export"}
            if "checks" in d and "metrics" in d and "summary" in d:
                return "report", {"names": [f.get("name") for f in d.get("files", [])], "software": "PINNeAPPle Mesh Quality",
                                  "title": f"mesh health: {d.get('status')}", "parameters": {"status": d.get("status")},
                                  "rule": "Mesh Quality export"}
            if "findings" in d and "sections" in d:
                return "report", {"names": [f.get("name") for f in d.get("files", [])], "software": "PINNeAPPle Simulation Preflight",
                                  "title": f"pre-flight: {d.get('status')}", "parameters": {"status": d.get("status")},
                                  "rule": "Simulation Preflight export"}
            if d.get("kind") == "model" or "trained_on" in d:
                return "model", {"names": [str(x) for x in d.get("trained_on", [])], "software": d.get("framework") or d.get("software"),
                                 "timestamp": d.get("timestamp"), "title": d.get("name") or "model",
                                 "parameters": {k: v for k, v in d.items() if k in ("architecture", "epochs", "loss", "metrics")},
                                 "rule": "model card (trained_on)"}
            return None
        if low.endswith(".parquet"):
            import pyarrow.parquet as pq
            md = pq.read_schema(io.BytesIO(b)).metadata or {}
            if b"pinneapple" in md:
                d = json.loads(md[b"pinneapple"])
                m = d.get("metadata", {})
                return "dataset", {"files": m.get("files"), "software": m.get("converter"), "timestamp": m.get("converted_at"),
                                   "title": f"dataset ({m.get('solver') or m.get('source_format')}, Parquet)",
                                   "parameters": {"fields": list(d.get("fields", {})), "time": m.get("time")},
                                   "rule": "Interoperability Hub dataset (Parquet)"}
        if low.endswith((".h5", ".hdf5")):
            import h5py
            with h5py.File(io.BytesIO(b), "r") as h:
                if h.attrs.get("schema") == "pinneapple.physical_dataset/1":
                    d = json.loads(h.attrs["metadata"])
                    m = d.get("metadata", {})
                    return "dataset", {"files": m.get("files"), "software": m.get("converter"), "timestamp": m.get("converted_at"),
                                       "title": f"dataset ({m.get('solver') or m.get('source_format')}, HDF5)",
                                       "parameters": {"fields": list(d.get("fields", {})), "time": m.get("time")},
                                       "rule": "Interoperability Hub dataset (HDF5)"}
    except Exception:                                                  # not ours, or unreadable: treat as a plain file
        return None
    return None
