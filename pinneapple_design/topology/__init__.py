"""Density-based topology optimization: SIMP, density filter, Heaviside projection, OC and MMA.

Problems: structural compliance (MBB beam, cantilever), thermal compliance, Stokes flow with
Brinkman penalization (Borrvall-Petersson), and the iron of a machine rotor for torque
(:mod:`pinneapple_design.topology.magnetic`, built on ``pinneapple_simulation.electromagnetics``).
See docs/guides/topology_optimization.md.
"""
from .driver import DesignChain, TopOptResult, finite_difference_check, optimize  # noqa: F401
from .filters import DensityFilter, SymmetryMap, heaviside, heaviside_derivative  # noqa: F401
from .optimizers import MMA, oc_update  # noqa: F401
from .problems import (  # noqa: F401
    Compliance2D,
    Q4Grid,
    StokesDissipation2D,
    ThermalCompliance2D,
    conduction_ke,
    elasticity_ke,
)
