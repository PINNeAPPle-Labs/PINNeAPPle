"""Real, closed-form (non-PDE) engineering physics models.

``pinneapple_physics.pde_environment`` targets full PDE/FEM problems
(a ``ProblemSpec`` a real solver like FEniCS or a PINN then solves).
Not every real, citable engineering model needs that machinery --
classical lumped-parameter and 1D analytical results (fin-array
conduction, beam bending/fatigue, membrane diffusion, Helmholtz
resonance, ...) are real, standard, textbook physics in their own
right, useful as fast surrogates, training-data generators, or
independent verification ground truth without a PDE solve.

This subpackage exists so those closed-form models are shared,
tested, real PINNeAPPle code -- instead of being re-derived and
duplicated inside every downstream product that needs one. It started
from formulas originally written standalone inside five
``PINNeAPPle-apps`` products (heatsink/brakecool/ridetune/soundshape/
patchdose design tools) and was ported back into the platform so
future products can import them directly rather than re-deriving them.

Sub-modules
-----------
fin_array_conduction
    Extended-surface (fin) conduction + convection, real closed-form
    result (Incropera & DeWitt) generalized over any fin-array
    geometry (straight rectangular fins, radial vanes, ...).
cantilever_fatigue
    Cantilever-beam bending stress + Basquin stress-life fatigue.
membrane_diffusion
    Fick's-first-law membrane-controlled diffusive release/transport.
helmholtz_resonator
    Helmholtz-resonator (bass-reflex port) tuning equation.
plate_fin_heatsink
    Plate-fin heat sink: forced (Teertstra) / natural (Bar-Cohen)
    convection, spreading resistance (Lee), pressure drop.
pcb_thermal
    PCB stack-up conductivity, thermal vias, JEDEC two-resistor packages
    and board-surface convection + radiation.
burgers
    Viscous (Cole-Hopf) and inviscid (characteristics) Burgers solutions
    for the sine initial condition.
pipe_flow
    Darcy friction factors (laminar, Blasius, Colebrook, Haaland), pressure
    drop and first-cell height for a target y+.
heat
    Transient 1-D slab conduction (Fourier series, Carslaw & Jaeger).
compressible
    Isentropic ratios, normal shock, Pitot pressure, modified-Newtonian Cp_max.
particle_drag
    Sphere drag coefficient (Clift, Grace & Weber 1978; Schiller & Naumann 1933).
rocket_nozzle
    Ideal rocket nozzle: exit Mach from the area ratio, choked mass flow, c*, thrust coefficient, Isp.
ideal_reactors
    Ideal isothermal first-order reactors: plug-flow and stirred-tank conversion, tube volume and residence time.
"""
from __future__ import annotations

from . import (
    burgers,
    cantilever_fatigue,
    compressible,
    fin_array_conduction,
    heat,
    helmholtz_resonator,
    ideal_reactors,
    membrane_diffusion,
    particle_drag,
    pcb_thermal,
    pipe_flow,
    plate_fin_heatsink,
    rocket_nozzle,
)

__all__ = ["fin_array_conduction", "plate_fin_heatsink", "cantilever_fatigue", "membrane_diffusion",
           "helmholtz_resonator", "pcb_thermal", "burgers", "pipe_flow", "heat", "compressible", "particle_drag",
           "rocket_nozzle", "ideal_reactors"]
