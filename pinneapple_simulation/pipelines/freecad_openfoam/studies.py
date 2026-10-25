"""The worked, validated studies of the pipeline, as :class:`~.pipeline.PipelineSpec` factories (generic geometry,
no client data). Each compares a measured quantity with a closed form or a published correlation
(:mod:`pinneapple_physics.closed_form`); the measured numbers are in ``examples/freecad_openfoam`` and the guide.

==============  =========================================================  ==========================================
study           what is measured                                           reference
==============  =========================================================  ==========================================
``pipe_bend``   laminar 180° bend (Re 200, Rc / D = 4, De = 70.7): the     Hagen-Poiseuille ``f = 64 / Re``; White
                fully developed friction factor of the straight inlet leg  (1929) ``f_c / f_s`` of a curved pipe
                and the ratio of the developed pressure gradient in the
                bend (120°-165°) to the straight one
``heated_pipe`` laminar thermal entry at constant wall temperature (Re     Hausen (1943) mean Nusselt number
                100, Pr 2, heated length 10 D after a 6 D unheated entry):  (Incropera eq. 8.57)
                mean Nusselt number from the bulk outlet temperature
``sphere``      sphere drag coefficient at Re 100 (laminar, steady wake;   Clift, Grace & Weber (1978) standard drag
                half model, symmetry plane)                                curve, Table 5.2
==============  =========================================================  ==========================================
"""
from __future__ import annotations

from .pipeline import PipelineSpec

__all__ = ["pipe_bend", "heated_pipe", "sphere", "STUDIES"]

_BEND_ANGLES = (60, 90, 105, 120, 135, 150, 165)


def pipe_bend(cells_across: float = 16, backend: str = "auto", procs: int | str = "auto",
              bend_window: bool = True) -> PipelineSpec:
    """Laminar flow through a pipe bend; inputs: the FreeCAD parameters (D, Rc, angle, L_in, L_out) and Re.
    ``bend_window=False`` drops the sections inside the bend and the White (1929) reference, which need a bend of at
    least 165° (for sweeps over the angle)."""
    angles = _BEND_ANGLES if bend_window else ()
    secs = [{"name": f"b{a}", "origin": [f"Rc * sin(radians({a}))", f"Rc - Rc * cos(radians({a}))", 0],
             "normal": [f"cos(radians({a}))", f"sin(radians({a}))", 0], "radius": "D / 2"} for a in angles]
    secs += [{"name": n, "origin": [x, 0, 0], "normal": [1, 0, 0], "radius": "D / 2"}
             for n, x in (("s1", "-6 * D"), ("s2", "-2 * D"))]
    spec = PipelineSpec(
        id="pipe_bend", title="Laminar pipe bend: pressure drop and Dean-flow friction", template="pipe_bend",
        backend=backend, procs=procs,
        parameters={"L_in": {"default": 200.0}},
        inputs={"Re": {"default": 200.0, "min": 10.0, "max": 1500.0, "unit": "-",
                       "description": "Reynolds number on the diameter (laminar)"}},
        physics={"nu": 1e-6, "rho": 1000.0, "velocity": "Re * nu / (D * 1e-3)", "iterations": 4000,
                 "residual_tol": 1e-6},
        mesh={"cells_across": float(cells_across), "length": "D", "level": 1},
        sections=secs,
        derived={"Dh": "sqrt(4 * area_inlet / pi)", "Um": "inflow_m3_s / area_inlet", "Re_m": "Um * Dh / nu",
                 "G_s": "(p_s1 - p_s2) / (4 * D * 1e-3)",
                 "G_b": "(p_b120 - p_b165) / (Rc * 1e-3 * radians(45))",
                 "f_s": "G_s * Dh / (0.5 * rho * Um ** 2)", "ratio": "G_b / G_s"},
        references={"straight": {"measured": "f_s", "reference": "laminar_f(Re_m)", "tol": 0.03,
                                 "source": "Hagen-Poiseuille, f = 64 / Re"},
                    "bend": {"measured": "ratio", "reference": "white_ratio(dean(Re_m, D, Rc))", "tol": 0.05,
                             "source": "White (1929) curved-pipe friction"}},
        plane={"origin": [0, 0, 0], "normal": [0, 0, 1]},
        description="FreeCAD pipe bend (spreadsheet D, Rc, angle, L_in, L_out) -> snappyHexMesh -> laminar simpleFoam. "
                    "Measured: total pressure drop, the developed straight friction factor and the curved/straight "
                    "pressure-gradient ratio in the second half of the bend.",
        limits=["Laminar, steady, incompressible; Re up to 1500.",
                "The bend ratio is compared with White's correlation for fully developed curved-pipe flow; the "
                "measuring window (120-165 degrees) must be developed, which holds for the 180 degree default.",
                "The validated configuration is D = 10 mm, Rc = 40 mm, angle 180 degrees, L_in = 200 mm, Re = 200."])
    if not bend_window:
        spec.references.pop("bend")
        for k in ("G_b", "ratio"):
            spec.derived.pop(k)
    return spec


def heated_pipe(cells_across: float = 16, backend: str = "auto", procs: int | str = "auto") -> PipelineSpec:
    """Laminar thermal entry at constant wall temperature (passive scalar T, inlet 0, heated wall 1)."""
    return PipelineSpec(
        id="heated_pipe", title="Heated pipe: laminar thermal entry, mean Nusselt number", template="heated_pipe",
        backend=backend, procs=procs,
        inputs={"Re": {"default": 100.0, "min": 10.0, "max": 1500.0, "unit": "-", "description": "Reynolds number"},
                "Pr": {"default": 2.0, "min": 0.5, "max": 20.0, "unit": "-", "description": "Prandtl number"}},
        physics={"nu": 1e-6, "rho": 1000.0, "velocity": "Re * nu / (D * 1e-3)", "iterations": 3000,
                 "residual_tol": 1e-6,
                 "scalar": {"name": "T", "D": "nu / Pr", "inlet": 0.0, "walls": {"wall_heated": 1.0}}},
        mesh={"cells_across": float(cells_across), "length": "D", "level": 1},
        derived={"Nu_m": "nu_from_T(Re, Pr, D / L_heated, 1.0, scalar_in, scalar_out)"},
        references={"Nu": {"measured": "Nu_m", "reference": "hausen_nu(Re, Pr, D / L_heated)", "tol": 0.05,
                           "source": "Hausen (1943), Incropera eq. 8.57"}},
        plane={"origin": [0, 0, 0], "normal": [0, 0, 1]},
        description="FreeCAD straight pipe with an unheated entry and a heated section -> snappyHexMesh -> "
                    "laminar simpleFoam, then scalarTransportFoam for the temperature (a passive scalar at "
                    "constant properties) -> mean Nusselt number from the bulk outlet temperature.",
        limits=["Constant properties, no buoyancy, no viscous heating: the temperature is a passive scalar.",
                "Hausen's correlation assumes a developed velocity profile at the start of heating; the 6 D "
                "unheated entry develops it for Re up to about 100 (0.06 Re D).",
                "Axial conduction is neglected by the correlation (Pe = Re Pr = 200 here)."])


def sphere(surface_level: int = 4, backend: str = "auto", procs: int | str = "auto") -> PipelineSpec:
    """Sphere (the spheroid template with L = D) in a uniform stream at low Reynolds number."""
    return PipelineSpec(
        id="sphere_drag", title="Sphere in a uniform stream: drag coefficient", template="spheroid",
        backend=backend, procs=procs,
        inputs={"Re": {"default": 100.0, "min": 20.0, "max": 200.0, "unit": "-",
                       "description": "Reynolds number on the diameter (steady axisymmetric wake below ~210)"}},
        physics={"nu": 1.5e-5, "rho": 1.2, "speed": "Re * nu / (D * 1e-3)", "turbulence": "laminar",
                 "half_model": True, "ref_area": "pi / 4 * (D * 1e-3) ** 2", "domain": [4, 10, 4, 4], "cells_per_length": 4,
                 "surface_level": [int(surface_level), int(surface_level) + 1], "iterations": 1500,
                 "residual_tol": 1e-5},
        references={"CD": {"measured": "CD", "reference": "clift_cd(Re)", "tol": 0.05,
                           "source": "Clift, Grace & Weber (1978), Table 5.2"}},
        plane={"origin": [0, 0.01, 0], "normal": [0, 1, 0]},
        description="FreeCAD spheroid (D = L) -> ExternalFlow (snappyHexMesh, half model) -> laminar simpleFoam "
                    "-> drag coefficient from the wall pressure and shear.",
        limits=["Steady laminar wake: Re below about 210; above it the wake is unsteady.",
                "Finite domain (4 D upstream, 10 D downstream, 4 D to the sides) adds blockage of order 1 %."])


STUDIES = {"pipe_bend": pipe_bend, "heated_pipe": heated_pipe, "sphere": sphere}
