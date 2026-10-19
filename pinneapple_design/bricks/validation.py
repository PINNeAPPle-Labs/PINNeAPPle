"""Checks of the brick physics against closed forms, gathered into a trust report.

:func:`validate` runs the statics LP, the Newton replay and the voxel FEA on cases with known answers and returns
one row per check plus a :class:`~pinneapple_analysis.trust.TrustReport`. Disagreements are reported, not hidden:
at exactly the harmonic limit the closed form is neutral equilibrium while Newton's compliant contacts tip it.
"""
from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np

from . import closed_form as cf
from .assembly import Assembly, Ground, stack
from .parts import G, JointModel
from .statics import Load, Structure, analyse, analyse_assembly, capacity, from_assembly, top_load


def _row(name, simulated, reference, tol, unit="", source="", note=""):
    err = abs(simulated - reference) / max(abs(reference), 1e-300) if np.isfinite(reference) else float("nan")
    return {"check": name, "simulated": float(simulated), "reference": float(reference), "rel_error": float(err),
            "tolerance": tol, "passed": bool(err <= tol), "unit": unit, "source": source, "note": note}


def _harmonic_structure(n, f, L=63.8, m=0.00303):
    right = cf.harmonic_stack(n, L) + L
    right = np.cumsum(np.diff(np.concatenate([[0.0], right])) * f)
    boxes = [((r - L, 0, i * 9.6), (r, 7.8, (i + 1) * 9.6)) for i, r in enumerate(right)]
    return Structure.from_boxes(boxes, [m] * n, ground_rects=[(-1e4, 0, -1e4, 1e4)])


def harmonic_limit_lp(n: int = 6, L: float = 63.8, tol: float = 1e-4, joint: JointModel | None = None) -> float:
    """Largest factor on the harmonic offsets the LP still finds stable (bisection; closed form: 1 for sharp edges,
    ``joint.edge_inset_mm = 0``)."""
    joint = joint or JointModel(edge_inset_mm=0.0)
    lo, hi = 0.9, 1.1
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if analyse(_harmonic_structure(n, mid, L), joint).stable else (lo, mid)
    return lo


def validate(*, newton: bool | None = None, fea: bool = True, joint: JointModel | None = None) -> dict:
    """Run every closed-form check. ``newton``: include the Newton replay checks (default: when installed). The
    closed forms have sharp edges, so the statics checks use ``joint`` with ``edge_inset_mm = 0``.

    Returns ``{"rows": [...], "harmonic_newton": [...], "report": TrustReport}``."""
    from pinneapple_analysis.trust import Check, TrustReport
    joint = dataclasses.replace(joint or JointModel(), edge_inset_mm=0.0)
    rows = []
    # 1. harmonic stacking, unconnected blocks (LP): limit factor 1
    lim = harmonic_limit_lp(joint=joint)
    rows.append(_row("harmonic stack (6 blocks): stability limit / closed form", lim, 1.0, 1e-3, "x offsets",
                     "Johnson 1955; Paterson and Zwick 2009"))
    # 2. single block on a table edge: tips at L/2
    lo, hi = 0.3, 0.7
    while hi - lo > 1e-5:
        mid = 0.5 * (lo + hi)
        s = Structure.from_boxes([((-63.8 * (1 - mid), 0, 0), (63.8 * mid, 7.8, 9.6))], [0.003],
                                 ground_rects=[(-1e3, 0, -1e3, 1e3)])
        lo, hi = (mid, hi) if analyse(s, joint).stable else (lo, mid)
    rows.append(_row("single block overhang limit / length", lo, 0.5, 1e-3, "", "centre of mass over the edge"))
    # 3. stud cantilevers: one and two studs
    for part_name, region, d, studs, x1 in (("brick_1x4", (0, 1, 0, 1), 16.0, [4.0], 8.0),
                                            ("brick_1x8", (0, 2, 0, 1), 32.0, [4.0, 12.0], 16.0)):
        a = Assembly(ground=Ground("baseplate", [region]))
        a.add(part_name, 0, 0, 0)
        r = analyse_assembly(a, joint=joint)
        ref = cf.stud_cantilever_tension(a.bricks[0].part.mass * G, d, studs, x1)
        rows.append(_row(f"stud cantilever {part_name} on {len(studs)} stud(s): tension per stud",
                         r.utilisation * joint.clutch_n, ref, 1e-6, "N", "moment about the patch edge"))
    # 4. tower push: tip force W b / h
    t = stack("brick_2x2", 8)
    lam = capacity(from_assembly(t, [top_load(t, (1.0, 0.0, 0.0))]), joint)
    rows.append(_row("tower of 8 2x2 bricks: tipping push at the top", lam,
                     cf.tower_tip_force(t.mass() * G, 8.0, t.height_mm()), 1e-6, "N", "F = W b / h"))
    # 5. simply supported plate beam: reactions add up to weight + load
    b = Assembly(ground=Ground("table", [(0, 1, 0, 1), (15, 16, 0, 1)]))
    for x in (0, 8):
        b.add("plate_1x8", x, 0, 0)
    b.add("plate_1x4", 0, 0, 1)
    b.add("plate_1x8", 4, 0, 1)
    b.add("plate_1x4", 12, 0, 1)
    P = 0.05
    s = from_assembly(b, [top_load(b, (0, 0, -P), at=(64.0, 4.0))])
    r = analyse(s, joint)
    ground = [k for k, i in enumerate(s.interfaces) if i.lower < 0]
    rows.append(_row("plate beam: sum of support reactions", float(r.interface_compression[ground].sum()),
                     b.mass() * G + P, 1e-6, "N", "vertical equilibrium"))
    out = {"rows": rows, "harmonic_newton": []}
    # 6. voxel FEA beam vs Timoshenko
    if fea:
        from . import fea as _fea
        beam = Assembly(ground=Ground("table", [(0, 1, 0, 1), (15, 16, 0, 1)]))
        for k in range(3):
            if k % 2 == 0:
                beam.add("plate_1x8", 0, 0, k)
                beam.add("plate_1x8", 8, 0, k)
            else:
                beam.add("plate_1x4", 0, 0, k)
                beam.add("plate_1x8", 4, 0, k)
                beam.add("plate_1x4", 12, 0, k)
        m = _fea.voxel_mesh(beam, 4).mesh
        L, bw, h = 128.0, 8.0, 9.6
        ends = [n for n in m.nodes_on(z=0.0) if abs(m.nodes[n, 0]) < 1e-9 or abs(m.nodes[n, 0] - L) < 1e-9]
        fixed = {n: (2,) for n in ends}
        fixed[ends[0]] = (0, 1, 2)
        fixed[[n for n in ends if m.nodes[n, 0] > 1][0]] = (1, 2)
        res = _fea.solve(beam, [Load(0, (L / 2, 4, h), (0, 0, -1.0))], fixed_nodes=fixed, gravity=False, per_stud=4)
        w = -res.fem.u[m.nodes_on(x=L / 2, z=0.0), 2].mean()
        E, nu = _fea.E_ABS_MPA, _fea.NU_ABS
        I_ = bw * h ** 3 / 12
        tim = cf.beam_midspan_deflection(L, E, I_, P=1.0) + L / (4 * 5 / 6 * E / (2 * (1 + nu)) * bw * h)
        rows.append(_row("voxel FEA plate beam: mid-span deflection (Timoshenko)", w, tim, 0.02, "mm",
                         "Euler-Bernoulli + shear", f"Euler-Bernoulli alone: {cf.beam_midspan_deflection(L, E, I_, P=1.0):.4f} mm"))
    # 7. Newton: harmonic stack and table edge
    if newton is None:
        newton = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None
    if newton:
        from .dynamics import harmonic_threshold, replay
        hn = harmonic_threshold(n=4, factors=(0.9, 0.95, 0.98, 0.99, 1.0, 1.01, 1.02, 1.05))
        out["harmonic_newton"] = hn
        away = [r for r in hn if r["factor"] != 1.0]
        rows.append({"check": "Newton harmonic stack (4 blocks): stands iff factor < 1, factors != 1",
                     "simulated": float(sum(r["agree"] for r in away)), "reference": float(len(away)),
                     "rel_error": float(1 - sum(r["agree"] for r in away) / len(away)), "tolerance": 0.0,
                     "passed": all(r["agree"] for r in away), "unit": "cases", "source": "harmonic stacking",
                     "note": "at exactly factor 1 (neutral equilibrium) Newton " +
                             ("keeps" if next(r for r in hn if r["factor"] == 1.0)["newton_stands"] else "tips") +
                             " the stack"})
        a = Assembly(ground=Ground("table", [(-100, 0, -100, 100)]))
        a.add("brick_1x4", -3, 0, 0)
        a.add("brick_1x4", -1, 2, 0)
        rp = replay(a, duration=0.5, frames=6, joint=joint)
        ok = rp.moved.tolist() == [False, True]
        rows.append({"check": "Newton table edge: brick with centre of mass inside stays, outside falls",
                     "simulated": float(ok), "reference": 1.0, "rel_error": 0.0 if ok else 1.0, "tolerance": 0.0,
                     "passed": ok, "unit": "", "source": "tipping", "note": ""})
    checks = [Check(r["check"], "supports" if r["passed"] else "contradicts", critical=True,
                    value=r["rel_error"], limit=r["tolerance"], source=r["source"]) for r in rows]
    if not newton:
        checks.append(Check("Newton replay checks", "not_run", reason="newton/warp not installed"))
    out["report"] = TrustReport(checks, model="pinneapple_design.bricks statics, dynamics and FEA")
    return out
