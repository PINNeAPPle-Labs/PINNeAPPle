"""Plate-fin heat-sink designs evaluated two ways at once: the exact solid from NVIDIA USD BRep (volume, area, mass)
and PINNeAPPle's closed-form thermal and hydraulic model (``pinneapple_physics.closed_form.plate_fin_heatsink``:
Teertstra forced convection, fin efficiency, spreading resistance, pressure drop). The table feeds surrogates
(PhysicsNeMo) and design searches; :func:`verify` re-evaluates a proposed design with both before it is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pinneapple_physics.closed_form import plate_fin_heatsink as pfh

from .brep import evaluate_part
from .parts import heat_sink, heat_sink_geometry

__all__ = ["Duty", "evaluate_design", "design_table", "sample_designs", "verify", "FEATURES"]

FEATURES = ("n_fins", "fin_height", "fin_t")


@dataclass
class Duty:
    power_w: float = 60.0
    t_ambient_c: float = 25.0
    air_velocity_m_s: float = 3.0
    source: float = 0.04            # square heat source side [m]
    base_t: float = 0.005
    length: float = 0.1
    width: float = 0.1
    material: str = "al6063"
    root_fillet: float = 0.0        # m, inner fin-root radius of the extrusion (exact solid only)


def evaluate_design(n_fins: int, fin_height: float, fin_t: float, duty: Duty | None = None, *, brep: bool = True,
                    chord_tolerances=()) -> dict:
    """One design: closed-form thermal / hydraulic result and (``brep``) the exact solid's volume, area, mass."""
    duty = duty or Duty()
    g = pfh.HeatSinkGeometry(duty.width, duty.length, duty.base_t, int(n_fins), fin_t, fin_height, duty.material)
    op = pfh.OperatingPoint(duty.power_w, duty.t_ambient_c, duty.air_velocity_m_s, duty.source, duty.source)
    th = pfh.evaluate(g, op)
    cf = heat_sink_geometry(n_fins=int(n_fins), fin_height=fin_height, fin_t=fin_t, base_t=duty.base_t,
                            length=duty.length, width=duty.width, root_fillet=duty.root_fillet)
    out = {"n_fins": int(n_fins), "fin_height": float(fin_height), "fin_t": float(fin_t), "gap": cf["gap"],
           "r_total": th["r_total_k_w"], "t_source": th["t_source_max_c"], "dp": th["pressure_drop_pa"],
           "fin_efficiency": th["fin_efficiency"], "mass_closed_form": th["mass_kg"], "volume_closed_form": cf["volume"],
           "area_closed_form": cf["total_area"], "warnings": th["warnings"]}
    if brep:
        rho = pfh.MATERIALS[duty.material]["rho"]
        rep = evaluate_part(heat_sink(n_fins=int(n_fins), fin_height=fin_height, fin_t=fin_t, base_t=duty.base_t,
                                      length=duty.length, width=duty.width, root_fillet=duty.root_fillet),
                            chord_tolerances=list(chord_tolerances),
                            density=rho)
        out.update(volume=rep.volume, area=rep.area, mass=rep.raw["mass"]["mass"], manifold=rep.manifold,
                   faces=rep.raw["faces"], brep_s=rep.seconds, _rep=rep)
    return out


def sample_designs(n: int, *, n_range=(6, 20), h_range=(0.015, 0.045), t_range=(0.001, 0.003), min_gap=0.002,
                   width: float = 0.1, seed: int = 0) -> list[tuple[int, float, float]]:
    """Latin-hypercube designs inside the ranges with at least ``min_gap`` between fins."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        u = (rng.permutation(n) + rng.random(n)) / n
        v = (rng.permutation(n) + rng.random(n)) / n
        w = (rng.permutation(n) + rng.random(n)) / n
        for a, b, c in zip(u, v, w, strict=True):
            nf = int(round(n_range[0] + a * (n_range[1] - n_range[0])))
            h = h_range[0] + b * (h_range[1] - h_range[0])
            t = t_range[0] + c * (t_range[1] - t_range[0])
            if (width - nf * t) / (nf - 1) >= min_gap and len(out) < n:
                out.append((nf, h, t))
    return out


def design_table(designs, duty: Duty | None = None, *, brep: bool = True) -> list[dict]:
    rows = [evaluate_design(*d, duty or Duty(), brep=brep) for d in designs]
    for r in rows:
        r.pop("_rep", None)
    return rows


def verify(n_fins: int, fin_height: float, fin_t: float, duty: Duty | None = None, **kw) -> dict:
    """Closed form + exact solid for a design a surrogate proposed (the evaluation that decides)."""
    return evaluate_design(n_fins, fin_height, fin_t, duty, **kw)
