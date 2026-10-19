"""Electromagnetics and electrodynamics in SI: references, materials, solvers and machines.

Statics and machines (2-D, P1 triangles):

* :mod:`.mesh` -- polar ring mesher with a moving band for rotation, structured rectangles; gmsh importer.
* :mod:`.materials` -- linear media, permanent magnets, nonlinear B-H curves.
* :mod:`.hysteresis` -- Froehlich-Kennelly saturation curve, Jiles-Atherton hysteresis.
* :mod:`.fem` -- A_z magnetostatics: assembly and Newton solver with Dirichlet and (anti-)periodic constraints.
* :mod:`.electrostatics` -- -div(eps grad phi) = rho with electrodes; capacitance from the field energy.
* :mod:`.eddy` -- time-harmonic eddy currents in solid conductors with imposed net currents.
* :mod:`.postprocess` -- Maxwell-stress and co-energy torque, flux linkage, back-EMF, iron and magnet losses.
* :mod:`.machine` -- surface-PM / reluctance machine model (cogging, torque vs current angle, losses,
  rotor design sensitivities).

Fields, waves and references:

* :mod:`.constants` -- CODATA 2022 constants; :mod:`.units` -- SI dimensions and a dimensional check of
  Maxwell's equations.
* :mod:`.media` -- isotropic, anisotropic and dispersive (Drude, Lorentz, Debye) media, ``exp(+j omega t)``.
* :mod:`.statics` -- capacitors, Biot-Savart, loops, solenoids, inductances in closed form.
* :mod:`.waves` -- Fresnel, multilayers, skin depth, waveguides, cavities, dipoles, cylinder scattering,
  surface plasmons.
* :mod:`.mie` -- Mie series for dielectric and conducting spheres.
* :mod:`.analytic` -- closed-form references of the machine solvers (wire, coax, magnetised cylinder,
  skin effect, subdomain solution of a slotless machine).

Physics checks of any field (Gauss, Faraday, Ampere-Maxwell, Poynting, reciprocity, passivity) are in
:mod:`pinneapple_physics.maxwell_checks` and ``pinneapple check`` (``physics = "maxwell"``). See
docs/guides/electromagnetics.md.
"""
from . import constants, hysteresis, media, mie, statics, units, waves
from .analytic import (
    Layer,
    RadialLayers,
    coax_field,
    magnetized_cylinder,
    skin_effect_wire,
    wire_field,
)
from .constants import C0, EPS0, ETA0
from .eddy import TimeHarmonicSolution, solve_time_harmonic
from .electrostatics import ElectrostaticSolution, capacitance, solve_electrostatic
from .fem import Geometry, MagnetostaticSolution, constraint_map, flux_density, solve_magnetostatic
from .hysteresis import FrohlichKennelly, JilesAtherton
from .materials import (
    AIR,
    MU0,
    NU0,
    BHCurve,
    BrauerCurve,
    LinearMaterial,
    Magnet,
    illustrative_steel,
)
from .media import Anisotropic, Debye, Drude, Lorentz, Medium
from .mesh import Mesh2D, Ring, mesh_from_gmsh, polar_mesh, rect_mesh, zipper

__all__ = [
    "AIR", "MU0", "NU0", "BHCurve", "BrauerCurve", "Geometry", "Layer", "LinearMaterial", "Magnet",
    "MagnetostaticSolution", "Mesh2D", "RadialLayers", "Ring", "TimeHarmonicSolution", "coax_field",
    "constraint_map", "flux_density", "illustrative_steel", "magnetized_cylinder", "mesh_from_gmsh",
    "polar_mesh", "skin_effect_wire", "solve_magnetostatic", "solve_time_harmonic", "wire_field", "zipper",
    "rect_mesh", "constants", "hysteresis", "media", "mie", "statics", "units", "waves", "C0", "EPS0", "ETA0",
    "ElectrostaticSolution", "capacitance", "solve_electrostatic", "FrohlichKennelly", "JilesAtherton",
    "Anisotropic", "Debye", "Drude", "Lorentz", "Medium",
]
