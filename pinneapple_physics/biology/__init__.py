"""Biology-informed modelling: well-mixed models as symbolic ODE systems (SBML import and export), and the
biological models used by the identifiability, inference and reproduction examples.

    from pinneapple_physics.biology import read_sbml
    sys_ = read_sbml("BIOMD0000000012.xml")          # Elowitz & Leibler repressilator
    traj = sys_.simulate(np.linspace(0, 1000, 501))  # SciPy BDF with the analytic Jacobian
"""
from .ode_system import TIME, ODESystem, Trajectory
from .sbml import (
           SBMLError,
           SBMLUnsupported,
           biomodel,
           list_biomodels,
           mathml_to_sympy,
           read_sbml,
           write_sbml,
)

__all__ = ["ODESystem", "Trajectory", "TIME", "read_sbml", "write_sbml", "mathml_to_sympy", "SBMLError",
           "SBMLUnsupported", "biomodel", "list_biomodels"]
