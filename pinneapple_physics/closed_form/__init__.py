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
"""
from __future__ import annotations

from . import cantilever_fatigue
from . import fin_array_conduction
from . import helmholtz_resonator
from . import membrane_diffusion

__all__ = ["fin_array_conduction", "cantilever_fatigue", "membrane_diffusion", "helmholtz_resonator"]
