"""CalculiX / Abaqus input decks (.inp) and CalculiX run files (.sta, .cvg, .dat, solver output)."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
DOF = {1: "ux", 2: "uy", 3: "uz", 4: "rx", 5: "ry", 6: "rz", 11: "T", 8: "p"}
PROCEDURES = {"STATIC": "static", "DYNAMIC": "implicit dynamic", "HEAT TRANSFER": "heat transfer",
              "FREQUENCY": "modal (eigenfrequencies)", "BUCKLE": "linear buckling",
              "COUPLED TEMPERATURE-DISPLACEMENT": "coupled thermo-mechanical", "VISCO": "viscoplastic / creep",
              "UNCOUPLED TEMPERATURE-DISPLACEMENT": "uncoupled thermo-mechanical", "MODAL DYNAMIC": "modal dynamic",
              "STEADY STATE DYNAMICS": "harmonic response", "GREEN": "Green functions", "SENSITIVITY": "sensitivity",
              "DYNAMIC, EXPLICIT": "explicit dynamic", "ELECTROMAGNETICS": "electromagnetics", "CFD": "CFD"}
ELEMENT_FAMILY = {"C3D": "3-D solid", "CPS": "plane stress", "CPE": "plane strain", "CAX": "axisymmetric",
                  "S3": "shell", "S4": "shell", "S6": "shell", "S8": "shell", "B3": "beam", "B2": "beam",
                  "T3D": "truss", "DC": "heat transfer", "SPRING": "spring", "DASHPOT": "dashpot", "MASS": "point mass"}


def _kw(line: str) -> tuple[str, dict[str, str]]:
    parts = [p.strip() for p in line[1:].split(",")]
    name = parts[0].upper()
    params = {}
    for p in parts[1:]:
        if not p:
            continue
        k, _, v = p.partition("=")
        params[k.strip().upper()] = v.strip()
    return name, params


def parse_inp(text: str, include: dict[str, str] | None = None) -> dict[str, Any]:
    """Parse an input deck into model / steps / materials / BCs / loads. ``include`` maps file names to
    the text of *INCLUDE'd files that were uploaded too."""
    lines = text.splitlines()
    i = 0
    nodes: list[tuple[float, float, float]] = []
    elem_types: dict[str, int] = {}
    nsets: dict[str, int] = {}
    elsets: dict[str, int] = {}
    materials: dict[str, dict[str, Any]] = {}
    sections: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    model_bcs: list[dict[str, Any]] = []
    includes: list[str] = []
    contact: list[str] = []
    other: dict[str, int] = {}
    cur_mat: str | None = None
    cur_step: dict[str, Any] | None = None
    heading = None
    expanded: list[str] = []
    for l in lines:                                                   # resolve *INCLUDE first
        if l.upper().startswith("*INCLUDE"):
            _, prm = _kw(l)
            fn = prm.get("INPUT", "").strip('"').split("/")[-1]
            includes.append(fn)
            if include and fn in include:
                expanded += include[fn].splitlines()
            continue
        expanded.append(l)
    lines = expanded
    n = len(lines)

    def data_block(j: int) -> tuple[list[str], int]:
        out = []
        while j < n and not (lines[j].startswith("*") and not lines[j].startswith("**")):
            s = lines[j].strip()
            if s and not s.startswith("**"):
                out.append(s)
            j += 1
        return out, j

    while i < n:
        l = lines[i]
        if not l.startswith("*") or l.startswith("**"):
            i += 1
            continue
        name, prm = _kw(l)
        data, j = data_block(i + 1)
        if name == "HEADING":
            heading = " ".join(data)[:200] or None
        elif name == "NODE" and "PRINT" not in name:
            for d in data:
                p = [x for x in d.split(",") if x.strip()]
                if len(p) >= 3:
                    try:
                        nodes.append((float(p[1]), float(p[2]), float(p[3]) if len(p) > 3 else 0.0))
                    except ValueError:
                        pass
            if prm.get("NSET"):
                nsets[prm["NSET"].upper()] = nsets.get(prm["NSET"].upper(), 0) + len(data)
        elif name == "ELEMENT":
            t = prm.get("TYPE", "?").upper()
            cnt = sum(1 for d in data if not d.rstrip().endswith(","))   # continuation lines end with ','
            elem_types[t] = elem_types.get(t, 0) + cnt
            if prm.get("ELSET"):
                elsets[prm["ELSET"].upper()] = elsets.get(prm["ELSET"].upper(), 0) + cnt
        elif name in ("NSET", "ELSET"):
            key = (prm.get("NSET") or prm.get("ELSET") or "?").upper()
            if "GENERATE" in prm:
                c = 0
                for d in data:
                    a = [int(float(x)) for x in d.split(",") if x.strip()]
                    if len(a) >= 2:
                        c += (a[1] - a[0]) // (a[2] if len(a) > 2 else 1) + 1
            else:
                c = sum(len([x for x in d.split(",") if x.strip()]) for d in data)
            (nsets if name == "NSET" else elsets)[key] = (nsets if name == "NSET" else elsets).get(key, 0) + c
        elif name == "MATERIAL":
            cur_mat = prm.get("NAME", f"material{len(materials) + 1}")
            materials[cur_mat] = {"name": cur_mat}
        elif name in ("ELASTIC", "DENSITY", "CONDUCTIVITY", "SPECIFIC HEAT", "EXPANSION", "PLASTIC", "HYPERELASTIC",
                      "DEFORMATION PLASTICITY", "CREEP", "DAMPING") and cur_mat:
            vals = [[float(x) for x in re.findall(_NUM, d)] for d in data]
            m = materials[cur_mat]
            if name == "ELASTIC" and vals and len(vals[0]) >= 2:
                m["young_modulus"], m["poisson_ratio"] = vals[0][0], vals[0][1]
                if len(vals) > 1:
                    m["elastic_temperature_points"] = len(vals)
            elif name == "DENSITY" and vals:
                m["density"] = vals[0][0]
            elif name == "CONDUCTIVITY" and vals:
                m["conductivity"] = vals[0][0]
            elif name == "SPECIFIC HEAT" and vals:
                m["specific_heat"] = vals[0][0]
            elif name == "EXPANSION" and vals:
                m["expansion"] = vals[0][0]
            elif name == "PLASTIC":
                m["plastic_points"] = len(vals)
                if vals:
                    m["yield_stress"] = vals[0][0]
            else:
                m[name.lower().replace(" ", "_")] = True
        elif name.endswith("SECTION"):
            sections.append({"kind": name.title(), "elset": prm.get("ELSET"), "material": prm.get("MATERIAL"),
                             "thickness": (re.findall(_NUM, data[0])[0] if data and name in ("SHELL SECTION", "SOLID SECTION")
                                           and re.findall(_NUM, data[0]) else None)})
        elif name == "STEP":
            cur_step = {"index": len(steps) + 1, "name": prm.get("NAME"), "nlgeom": "NLGEOM" in prm and prm["NLGEOM"].upper() != "NO",
                        "max_increments": int(prm["INC"]) if prm.get("INC", "").isdigit() else None,
                        "perturbation": "PERTURBATION" in prm, "procedure": None, "time": None,
                        "boundary_conditions": [], "loads": [], "outputs": []}
            steps.append(cur_step)
        elif name == "END STEP":
            cur_step = None
        elif name in PROCEDURES and cur_step is not None:
            cur_step["procedure"] = PROCEDURES[name] + (" (steady state)" if "STEADY STATE" in prm else "")
            nums = [float(x) for x in re.findall(_NUM, data[0])] if data else []
            if name in ("STATIC", "DYNAMIC", "HEAT TRANSFER", "COUPLED TEMPERATURE-DISPLACEMENT", "VISCO",
                        "UNCOUPLED TEMPERATURE-DISPLACEMENT"):
                keys = ("initial_increment", "period", "min_increment", "max_increment")
                cur_step["time"] = {k: v for k, v in zip(keys, nums)}
                if "DIRECT" in prm:
                    cur_step["time"]["fixed_increments"] = True
            elif name == "FREQUENCY" and nums:
                cur_step["modes"] = int(nums[0])
            elif name == "BUCKLE" and nums:
                cur_step["modes"] = int(nums[0])
        elif name == "BOUNDARY":
            for d in data:
                p = [x.strip() for x in d.split(",")]
                if len(p) >= 2:
                    try:
                        a = int(float(p[1]))
                        b = int(float(p[2])) if len(p) > 2 and p[2] else a
                        val = float(p[3]) if len(p) > 3 and p[3] else 0.0
                    except ValueError:
                        continue
                    rec = {"set": p[0], "dofs": [DOF.get(k, str(k)) for k in range(a, b + 1)], "value": val,
                           "kind": "fixed" if val == 0 else "prescribed"}
                    (cur_step["boundary_conditions"] if cur_step is not None else model_bcs).append(rec)
        elif name in ("CLOAD", "DLOAD", "DFLUX", "CFLUX", "FILM", "RADIATE", "TEMPERATURE", "DSLOAD") and cur_step is not None:
            agg: dict[tuple[str, str], float] = {}
            count: dict[tuple[str, str], int] = {}
            for d in data:
                p = [x.strip() for x in d.split(",")]
                if len(p) < 2:
                    continue
                kind = p[1].upper() if name in ("DLOAD", "FILM", "RADIATE", "DFLUX", "DSLOAD") else \
                    DOF.get(int(float(p[1])), p[1]) if re.fullmatch(_NUM, p[1]) else p[1]
                try:
                    val = float(p[2]) if len(p) > 2 and p[2] else (float(p[1]) if name == "TEMPERATURE" else 0.0)
                except ValueError:
                    val = 0.0
                key = ("(nodes)" if re.fullmatch(r"\d+", p[0]) else p[0], kind)
                agg[key] = agg.get(key, 0.0) + val
                count[key] = count.get(key, 0) + 1
            for (target, kind), val in agg.items():
                cur_step["loads"].append({"type": name.lower(), "target": target, "component": kind,
                                          "total" if name == "CLOAD" else "value": val if name == "CLOAD" else val / count[(target, kind)],
                                          "entries": count[(target, kind)]})
        elif name in ("NODE PRINT", "EL PRINT", "NODE FILE", "EL FILE", "NODE OUTPUT", "ELEMENT OUTPUT", "CONTACT FILE") \
                and cur_step is not None:
            cur_step["outputs"].append(f"{name.title()}: {' '.join(data)[:60]}")
        elif name in ("CONTACT PAIR", "TIE", "SURFACE INTERACTION", "EQUATION", "MPC", "RIGID BODY", "COUPLING"):
            contact.append(f"{name.title()} {prm.get('NAME', '') or ''}".strip())
        else:
            other[name] = other.get(name, 0) + 1
        i = j
    bbox = None
    if nodes:
        a = np.array(nodes)
        bbox = {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "size": (a.max(0) - a.min(0)).tolist()}
    return {"heading": heading, "nodes": len(nodes), "elements": int(sum(elem_types.values())), "element_types": elem_types,
            "bounding_box": bbox, "node_sets": nsets, "element_sets": elsets, "materials": list(materials.values()),
            "sections": sections, "steps": steps, "model_boundary_conditions": model_bcs, "includes": includes,
            "interactions": contact, "other_keywords": other}


def parse_sta(text: str) -> list[dict[str, Any]]:
    rows = []
    for l in text.splitlines():
        p = l.split()
        if len(p) >= 7 and p[0].isdigit():
            try:
                rows.append({"step": int(p[0]), "increment": int(p[1]), "attempts": int(p[2].rstrip("U")),
                             "iterations": int(p[3]), "total_time": float(p[4]), "step_time": float(p[5]),
                             "increment_size": float(p[6]), "cutback": p[2].endswith("U") or int(p[2].rstrip("U")) > 1})
            except ValueError:
                continue
    return rows


def parse_cvg(text: str) -> list[dict[str, Any]]:
    rows = []
    for l in text.splitlines():
        p = l.split()
        if len(p) >= 9 and p[0].isdigit():
            try:
                rows.append({"step": int(p[0]), "increment": int(p[1]), "attempt": int(p[2]), "iteration": int(p[3]),
                             "contact_elements": int(p[4]), "force_residual_pct": float(p[5]),
                             "displacement_correction_pct": float(p[6]), "flux_residual_pct": float(p[7]),
                             "temperature_correction_pct": float(p[8])})
            except ValueError:
                continue
    return rows


def parse_dat(text: str) -> dict[str, Any]:
    """Largest printed displacement (or min/max of other printed results) at the last time of each set."""
    out: dict[str, Any] = {}
    hdr = re.compile(r"^\s*(displacements|temperatures|stresses|forces|strains|heat flux)\s*\(([^)]*)\)\s*"
                     rf"for set (\S+) and time\s+({_NUM})", re.M)
    heads = list(hdr.finditer(text))
    for k, m in enumerate(heads):
        body = text[m.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)]
        rows = []
        for l in body.splitlines():
            p = re.findall(_NUM, l)
            if len(p) >= 2:
                rows.append([float(x) for x in p[1:]])
        if not rows:
            continue
        width = len(rows[0])
        a = np.array([r for r in rows if len(r) == width])
        kind, comps, nset, tim = m.group(1), m.group(2), m.group(3), float(m.group(4))
        entry: dict[str, Any] = {"set": nset, "time": tim, "components": comps}
        if kind == "displacements" and width >= 3:
            mag = np.linalg.norm(a[:, :3], axis=1)
            entry.update({"max_magnitude": float(mag.max()), "at_max": a[int(np.argmax(mag)), :3].tolist()})
        else:
            entry.update({"max": float(a.max()), "min": float(a.min())})
        key = f"{kind} {nset}"
        if key not in out or tim >= out[key]["time"]:
            out[key] = entry
    return out


def parse_ccx_log(text: str) -> dict[str, Any]:
    v = re.search(r"CalculiX Version ([\w.\-]+)", text)
    return {"version": v.group(1) if v else None, "finished": "Job finished" in text,
            "errors": re.findall(r"\*ERROR[^\n]*", text)[:5], "warnings": len(re.findall(r"\*WARNING", text)),
            "wall_time_s": float(m.group(1)) if (m := re.search(rf"Total CalculiX Time:\s*({_NUM})", text)) else None,
            "threads": int(m.group(1)) if (m := re.search(r"Using up to (\d+) cpu", text)) else None}


def assess(sta: list[dict[str, Any]], cvg: list[dict[str, Any]], deck: dict[str, Any], log: dict[str, Any] | None) -> dict[str, Any]:
    reasons, recs = [], []
    out: dict[str, Any] = {"increments": sta, "iterations": cvg}
    if log and log.get("errors"):
        status = "failed"
        reasons.append("Solver errors: " + "; ".join(log["errors"][:3]))
    elif not sta:
        status = "unknown" if not log else ("completed" if log.get("finished") else "incomplete")
        reasons.append("No .sta file: increment history unavailable." if not log else
                       ("The solver output says 'Job finished'." if log.get("finished") else "The solver output is incomplete."))
    else:
        steps = [s for s in deck.get("steps", []) if s.get("time")]
        status = "converged"
        for st in {r["step"] for r in sta}:
            rows = [r for r in sta if r["step"] == st]
            period = None
            if st - 1 < len(deck.get("steps", [])) and deck["steps"][st - 1].get("time"):
                period = deck["steps"][st - 1]["time"].get("period")
            reached = rows[-1]["step_time"]
            if period and reached < period * (1 - 1e-6):
                status = "failed"
                reasons.append(f"Step {st} stopped at step time {reached:g} of {period:g}.")
                recs.append("Allow smaller increments (minimum increment), add stabilisation, or check for "
                            "unconstrained rigid-body modes and contact definitions.")
            else:
                reasons.append(f"Step {st}: all {len(rows)} increments converged; the step reached its full time"
                               f"{f' ({period:g})' if period else ''}.")
        cut = [r for r in sta if r["cutback"]]
        if cut:
            reasons.append(f"{len(cut)} increment{'s' if len(cut) > 1 else ''} needed a cutback (smaller time increment).")
        its = [r["iterations"] for r in sta]
        reasons.append(f"Newton iterations per increment: {min(its)}–{max(its)} (mean {np.mean(its):.1f}).")
        if cvg:
            last = {}
            for r in cvg:
                last[(r["step"], r["increment"])] = r
            worst = max(v["force_residual_pct"] for v in last.values())
            out["final_force_residual_pct_max"] = worst
            reasons.append(f"Largest final force residual of any increment: {worst:.2g} % of the average force.")
        if steps and not any(s.get("nlgeom") for s in deck["steps"]) and len(sta) > len(steps):
            recs.append("The analysis is linear (no NLGEOM): several increments are not needed.")
    if log and log.get("finished") and status == "unknown":
        status = "completed"
    return {**out, "status": status, "reasons": reasons, "recommendations": recs}
