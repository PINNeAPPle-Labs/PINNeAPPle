"""The arm in CalculiX: an external solver as the third model, and the tube that grades it.

The structural check of :mod:`.structure` is this library's own -- 8-node hexahedra with incompatible modes, and a beam
formula beside them. Both are ours, so a study that only ever ran the two could be right for the same wrong reason.
This module puts the same part through an independent code: gmsh builds the OpenCASCADE geometry and meshes it, and
``ccx`` solves it. The stress the corner is finally reported with has then been obtained twice, by two
discretisations and two programs.

Two cases:

* :func:`fem_tube_calculix` -- a hollow steel tube, clamped on its end face and loaded at the tip, against
  Timoshenko's cantilever (bending plus shear). The error it returns is the number
  :mod:`pinneapple_design.suspension.verification` quotes, and the difference between its first- and second-order
  answers is what the element order is worth;
* :func:`fem_lower_arm_calculix` -- the sizing case of :mod:`.loads` on the lower arm's leg: the same axial force and
  root moment :func:`pinneapple_design.suspension.structure.fem_lower_arm` applies to its hexahedra, so the two peaks
  and the beam formula are read side by side.

Requirements: the ``gmsh`` **Python** module (the ``gmsh`` executable is a Python entry point that may not see the
module) and ``ccx``, native or through the Docker runner (:func:`pinneapple_simulation.external_solvers.calculix.runner.ccx_backend`).
A leg is seconds to a couple of minutes, which is why the verification runs this and the Studio study does not.
"""
from __future__ import annotations

import math
import os

import numpy as np

from pinneapple_simulation.external_solvers.calculix.runner import ccx_backend
from pinneapple_simulation.external_solvers.calculix.study import FEModel, Material, Static, solve

from .hardpoints import SuspensionSpec
from .structure import TubeSection

__all__ = ["available", "fem_tube_calculix", "fem_lower_arm_calculix", "TUBE_SIZE", "TUBE_ORDER"]

#: the mesh the verification case uses: second-order tetrahedra are what make a coarse mesh agree with the beam, and
#: the numbers are in :mod:`pinneapple_design.suspension.verification` (first order at the same size is 3 % low)
TUBE_SIZE = 0.008
TUBE_ORDER = 2

#: the mesh the sizing case uses: second-order tetrahedra at a 10 mm target, which is where the peak stops moving
#: (measured on the nominal corner: 1.026 at 14 mm, 1.023 at 10 mm, 1.015 at 8 mm against beam theory). Linear
#: tetrahedra at 12 mm read the same peak 17 % low -- bending lock, which is the reason
#: :func:`pinneapple_design.suspension.structure.fem_lower_arm` carries incompatible modes at all
LEG_SIZE = 0.010
LEG_ORDER = 2


def available() -> bool:
    """True when ``ccx`` can be run (native binary or the Docker runner)."""
    return ccx_backend() is not None


def _add_tube(length_m: float, od_m: float, wall_m: float) -> float:
    """Add a hollow tube from the origin along ``+x`` to the current gmsh model; returns its B-rep volume [m^3]."""
    import gmsh

    occ = gmsh.model.occ
    outer = occ.addCylinder(0.0, 0.0, 0.0, float(length_m), 0.0, 0.0, 0.5 * od_m)
    # the bore is longer than the tube at both ends: a cut that stops exactly on the face leaves a sliver of material
    inner = occ.addCylinder(-0.01, 0.0, 0.0, float(length_m) + 0.02, 0.0, 0.0, 0.5 * od_m - wall_m)
    cut, _ = occ.cut([(3, outer)], [(3, inner)])
    return float(occ.getMass(3, cut[0][1]))


def _mesh_tube(length_m: float, section: TubeSection, size: float, order: int, workdir: str | None):
    """(the meshed tube, its exact volume): the volume is the B-rep's, so a mesh refinement does not move the mass."""
    box: dict[str, float] = {}

    def build() -> None:
        box["volume"] = _add_tube(length_m, section.od_m, section.wall_m)

    model = FEModel.from_gmsh_model(build, size, order=order, workdir=workdir)
    model.title = f"tube {section.describe()}"
    return model, box["volume"]


def _ends(model: FEModel, length_m: float) -> tuple[np.ndarray, np.ndarray]:
    """The nodes on the clamped face (``x = 0``) and on the loaded one (``x = length``)."""
    return (model.nodes_where(lambda X: X[:, 0] < 1e-9),
            model.nodes_where(lambda X: X[:, 0] > length_m - 1e-9))


def fem_tube_calculix(length_m: float = 0.40, od_m: float = 0.030, wall_m: float = 0.003,
                      load_n: float = 1000.0, *, size: float = TUBE_SIZE, order: int = TUBE_ORDER,
                      workdir: str | None = None, elastic_modulus_pa: float = 210e9,
                      poisson: float = 0.3) -> dict[str, object]:
    """A hollow tube cantilever in CalculiX against Timoshenko's beam, on the same geometry and the same load.

    The closed form, for outer diameter ``do``, wall ``t``, length ``L``, with the shear coefficient of a hollow circle
    ``k = 6 (1 + nu) (1 + m^2)^2 / ((7 + 6 nu) (1 + m^2)^2 + (20 + 12 nu) m^2)``, ``m = di / do``:

        delta = F L^3 / (3 E I) + F L / (k G A)

    The tube is fixed on its whole end face, which is not the beam's plane-section-at-the-support assumption (the
    clamp suppresses the shear warping there), so the agreement is reported as measured, not assumed.

    Returns
    -------
    out : dict
        ``deflection_m`` (finite elements), ``timoshenko_m`` (closed form), ``error_rel``, ``bending_share``,
        ``volume_m3``, ``mass_kg`` (B-rep volume times the density), ``elements``, ``nodes``, ``order``, ``size``.
    """
    section = TubeSection(od_m, wall_m)
    work = workdir or os.path.join(os.getcwd(), "_calculix_tube")
    model, volume = _mesh_tube(length_m, section, size, order, work)
    model.material = Material("steel", E=elastic_modulus_pa, nu=poisson, density=7850.0)
    root, tip = _ends(model, length_m)
    res = solve(model, Static(fix={tuple(root): (1, 2, 3)}, loads={tuple(tip): (0.0, 0.0, -float(load_n))}),
                os.path.join(work, "run"), threads=1, keep=False)
    E, nu = elastic_modulus_pa, poisson
    di = od_m - 2.0 * wall_m
    inertia = math.pi * (od_m ** 4 - di ** 4) / 64.0
    area = math.pi * (od_m ** 2 - di ** 2) / 4.0
    m = di / od_m
    k = 6.0 * (1.0 + nu) * (1.0 + m ** 2) ** 2 / ((7.0 + 6.0 * nu) * (1.0 + m ** 2) ** 2 + (20.0 + 12.0 * nu) * m ** 2)
    shear_modulus = E / (2.0 * (1.0 + nu))
    bending = load_n * length_m ** 3 / (3.0 * E * inertia)
    theory = bending + load_n * length_m / (k * shear_modulus * area)
    got = float(-np.mean(res.u[tip, 2]))
    return {"deflection_m": got, "timoshenko_m": float(theory), "error_rel": got / theory - 1.0,
            "bending_share": float(bending / theory), "shear_coefficient": float(k), "length_m": float(length_m),
            "volume_m3": float(volume), "mass_kg": float(volume) * 7850.0, "elements": int(len(model.elements)),
            "nodes": int(len(model.nodes)), "order": int(order), "size": float(size)}


def fem_lower_arm_calculix(spec: SuspensionSpec, section: TubeSection, loads: dict,
                           *, size: float = LEG_SIZE, order: int = LEG_ORDER, workdir: str | None = None) -> dict:
    """The lower arm's leg in CalculiX, under the sizing case :mod:`.loads` gives it.

    The two legs share the ball joint's force and root moment, so each is checked for half of it; the root moment
    enters as an equivalent transverse force at the tip (``M / L``), which produces the same root moment, and the sign
    of the force does not matter for von Mises. The peak is read over the nodes away from both ends (``0.15 L .. 0.85 L``)
    -- a stress reported at the node the load is applied to measures the load, not the part -- and beam theory is
    evaluated at the station where the finite elements peak, so the ratio means something.

    Returns
    -------
    out : dict
        ``sigma_peak_pa`` (finite elements), ``sigma_beam_pa`` (``N / A + M c / I`` at the same station), ``ratio``,
        ``station_m``, ``axial_n`` and ``root_moment_nm`` as applied, ``tip_deflection_m``, ``volume_m3``,
        ``mass_kg``, ``elements`` and ``nodes``.
    """
    leg = float(loads["leg_length_m"])
    axial = float(loads["lower_axial_n"]) / 2.0                       # compression positive
    moment = float(loads["lower_moment_nm"]) / 2.0
    transverse = moment / leg
    work = workdir or os.path.join(os.getcwd(), "_calculix_leg")
    model, volume = _mesh_tube(leg, section, size, order, work)
    model.material = Material("steel", E=spec.elastic_modulus_pa, nu=0.3, density=spec.density_kg_m3)
    root, tip = _ends(model, leg)
    res = solve(model, Static(fix={tuple(root): (1, 2, 3)}, loads={tuple(tip): (-axial, 0.0, -transverse)}),
                os.path.join(work, "run"), threads=1, keep=False)
    x = model.nodes[:, 0]
    band = np.flatnonzero((x > 0.15 * leg) & (x < 0.85 * leg))
    vm = np.asarray(res.von_mises, float)
    i = int(band[int(np.argmax(vm[band]))])
    station = float(x[i])
    sigma_beam = abs(axial) / section.area_m2 + transverse * (leg - station) * section.c_m / section.inertia_m4
    peak = float(vm[band].max())
    return {"sigma_peak_pa": peak, "sigma_beam_pa": float(sigma_beam),
            "ratio": float(peak / sigma_beam) if sigma_beam > 0 else float("nan"), "station_m": station,
            "axial_n": axial, "root_moment_nm": moment, "transverse_n": transverse, "length_m": leg,
            "tip_deflection_m": float(np.linalg.norm(res.u[tip], axis=1).max()),
            "volume_m3": float(volume), "mass_kg": float(volume) * spec.density_kg_m3,
            "elements": int(len(model.elements)), "nodes": int(len(model.nodes)), "order": int(order),
            "size": float(size)}
