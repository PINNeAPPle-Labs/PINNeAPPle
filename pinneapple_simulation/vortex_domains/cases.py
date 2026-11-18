"""The cases the module is built for: a cylinder (to check the solver against published numbers) and the reference
case of the video that motivated it -- a NACA 0012 with a deploying spoiler.

Both return a :class:`Case`: the geometry, the configuration, what the run should be compared with, and a note on
what that comparison means. Nothing is run until :func:`~.study.run_study` is called.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .bodies import (
    DEG,
    Assembly,
    Body,
    cylinder,
    deploying_spoiler,
    naca0012,
    naca4,
    resample_polygon,
)
from .solver import VVDConfig

__all__ = ["Case", "cylinder_case", "naca0012_spoiler_case", "spoiler_hinge"]


@dataclass
class Case:
    """A VVD problem: the bodies, the configuration, and what the result should be judged against."""

    name: str
    assembly: Assembly
    config: VVDConfig
    reference: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def build(self):
        """A :class:`~.solver.VVD2D` for this case."""
        from .solver import VVD2D

        return VVD2D(self.assembly, self.config, name=self.name)

    def describe(self) -> str:
        bodies = ", ".join(f"{b.name} ({len(b.vertices)} vertices)" for b in self.assembly.bodies)
        return (f"{self.name}: Re = {self.config.re:g}, U_inf = {self.config.u_inf}, "
                f"bodies: {bodies}, t_end = {self.config.t_end:g}")


def cylinder_case(re: float = 40.0, diameter: float = 1.0, panels: int = 96, t_end: float = 10.0,
                  reference_length: float | None = None) -> Case:
    """Flow past a circular cylinder -- the solver's verification case.

    At ``Re = 40`` the flow is steady and the drag coefficient is ``1.3375`` (Dennis & Chang, *J. Fluid Mech.* 42,
    1970); at ``Re = 200`` the wake sheds with a Strouhal number ``St = 0.1968`` (Williamson & Brown, *J. Fluids
    Struct.* 12, 1998, ``St = 0.2731 - 1.1129/sqrt(Re) + 0.4821/Re``). Those two numbers are the references the
    case carries; the tolerance is the one the verification run measured and reports, not a guess.
    """
    length = diameter if reference_length is None else reference_length
    config = VVDConfig(re=re, u_inf=(1.0, 0.0), t_end=t_end, reference_length=length)
    reference = {"drag_coefficient": 1.3375, "drag_source": "Dennis & Chang 1970, JFM 42:471 (Re=40, steady)",
                 "strouhal_law": "Williamson & Brown 1998: 0.2731 - 1.1129/sqrt(Re) + 0.4821/Re"}
    note = ("the drag of a cylinder in a steady stream; at Re = 40 the reference solution has no shedding, so a "
            "run that keeps shedding is telling you it is under-resolved")
    return Case(name=f"cylinder_re{re:g}", assembly=Assembly([cylinder(diameter, panels)]), config=config,
                reference=reference, note=note,
                metadata={"panels": panels, "diameter": diameter, "reference_length": length})


def spoiler_hinge(chord: float = 1.0, spoiler_length: float = 0.10, trailing_gap: float = 0.20) -> np.ndarray:
    """Where the spoiler is hinged, from the two lengths the reference case states.

    The spoiler is ``spoiler_length`` (10 % of the chord) long and its trailing edge sits ``trailing_gap`` (20 % of
    the chord) ahead of the airfoil's trailing edge, so the hinge is at ``x = c (1 - trailing_gap)`` on the upper
    surface: flush, the spoiler covers the last tenth of the chord; deployed, it stands up from that point.
    """
    x = chord * (1.0 - trailing_gap)
    return np.array([x, 0.0])


def naca0012_spoiler_case(re: float = 40_000.0, chord: float = 1.0, panels: int = 200,
                          spoiler_panels: int = 24, spoiler_length: float = 0.10,
                          deploy_angle_deg: float = 90.0, t_start: float = 0.4, deploy_time: float = 0.5,
                          t_end: float = 1.8, alpha_deg: float = 0.0, base_gap: float = 0.0,
                          max_domains: int = 20000) -> Case:
    """The reference case: a NACA 0012 at zero incidence, with a spoiler that deploys at ``t_start``.

    The numbers are the ones the reference video states: ``Re = 4e4``, spoiler of 10 % of the chord whose trailing
    edge is 20 % of the chord ahead of the airfoil's trailing edge, angle of attack 0, no-slip on the surface and
    an unperturbed free stream at infinity.

    Honest scope, in two parts.

    *Resolution.* At ``Re = 4e4`` the boundary layer is of order ``c/sqrt(Re) = 0.005c``, thinner than the panels a
    laptop-sized run can afford, so this case exercises the mechanism (moving geometry, vorticity creation,
    separation off the spoiler, force history) rather than resolving the boundary layer of the real flow. The
    solver's quantitative verification is on the cylinder at Re = 40 and 200, where the references exist.

    *The deployment.* This is the open case of the module, and the tests and the example say so: the moving wall asks
    for vorticity at a scale the panels cannot hold, so during the deployment the number of domains grows to
    thousands and the run leaves its stable envelope (the reported loads diverge). Flush, before it, the case gives
    the two numbers it owes -- `Cl = 0` and `Cd = 0.02` -- and those are reproducible. ``max_domains`` bounds the
    cost, and the ``displacement`` check is what reports the failure.

    *Trailing edge.* The panels are uniform in arclength (cosine spacing leaves panel next to the edge that are
    far shorter than the wall can resolve), but the section keeps its sharp trailing edge, where the potential flow
    is singular: the edge velocity reaches 15 to 50 times the free stream, and nothing here imposes a Kutta
    condition to remove it. What keeps the run finite is the cap on the wall flux (``flux_limit``), which is the
    honest description of the fix -- a numerical guard, not a Kutta condition. ``base_gap > 0`` blunts the base
    instead; measured, it makes the wall velocity physical (``U_e = 1.2``) but the base's own corners are then the
    demanding configuration, so it is not the default.
    """
    hinge = spoiler_hinge(chord, spoiler_length)
    # a blunt (open) trailing edge: a sharp one leaves a velocity singularity in the potential flow, and this
    # solver has no Kutta condition to remove it (see the note below)
    # uniform panels in arclength: cosine spacing leaves panels next to the trailing edge that are far shorter
    # than the wall can resolve. `base_gap > 0` cuts the last part of the chord off (a blunt base).
    section = naca4(m=0.0, p=0.4, t=0.12, chord=chord, n=max(200, 4 * panels), closed_te=False)
    stations = section.vertices[section.vertices[:, 0] <= chord * (1.0 - base_gap) + 1e-12]
    airfoil = Body(resample_polygon(stations, panels), name="naca0012")
    if alpha_deg:
        # rotate the airfoil about its leading edge by turning the whole body

        a = alpha_deg * DEG
        c, s = np.cos(a), np.sin(a)
        rot = np.array([[c, -s], [s, c]])
        airfoil.vertices = airfoil.vertices @ rot.T
        airfoil.area = float(abs(0.5 * np.dot(airfoil.vertices[:, 0], np.roll(airfoil.vertices[:, 1], -1))
                                  - 0.5 * np.dot(airfoil.vertices[:, 1], np.roll(airfoil.vertices[:, 0], -1))))
    # the hinge sits on the upper surface: use the airfoil's own surface point at that chord station
    surface = naca0012(n=2000, chord=chord)
    upper = surface.vertices[surface.vertices[:, 1] > -1e-9]
    station = hinge[0]
    y_upper = float(np.interp(station, upper[::-1, 0], upper[::-1, 1])) if upper.size else 0.0
    hinge = np.array([station, max(y_upper, 0.0)])

    spoiler = deploying_spoiler(hinge_at=hinge, length=spoiler_length * chord, thickness=0.004 * chord,
                               n=spoiler_panels, angle0_deg=0.0, angle1_deg=deploy_angle_deg,
                               t_start=t_start, duration=deploy_time)
    config = VVDConfig(re=re, u_inf=(1.0, 0.0), t_end=t_end, reference_length=chord,
                       max_domains=max_domains)
    return Case(
        name=f"naca0012_spoiler_re{re:g}",
        assembly=Assembly([airfoil, spoiler]),
        config=config,
        reference={"angle_of_attack_deg": 0.0,
                   "note": "symmetric section at zero incidence: the lift of the bare airfoil must stay at zero"},
        note=("a deploying spoiler: 10 % of the chord, hinged where its trailing edge is 20 % of the chord ahead of "
              "the airfoil trailing edge, rising to 90 degrees -- the case of the reference video"),
        metadata={"chord": chord, "hinge": hinge.tolist(), "spoiler_length": spoiler_length * chord,
                  "deploy_start": t_start, "deploy_time": deploy_time, "alpha_deg": alpha_deg,
                  "base_gap": base_gap, "panels": len(airfoil.vertices) + spoiler_panels},
    )
