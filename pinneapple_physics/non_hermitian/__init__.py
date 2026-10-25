"""Non-Hermitian physics: PT-symmetric dimers and chains, exceptional points and how they can (or cannot) be tuned.

* :mod:`.pt_dimer`: :class:`PTDimer`, :class:`PTChain`, EP location in closed form and from the matrix alone,
  eigenvector coalescence, Petermann factor;
* :mod:`.ep_tuning`: symbolic check of an EP tuning law against the no-go theorem for PT-symmetric dimers
  (Zenodo 23287622), classification of couplings, the three routes around it, and the theorem's hypotheses as a
  :mod:`pinneapple_physics.physics_audit` profile;
* :mod:`.viz`: spectrum across the EP, EP laws by coupling class, the coupling-plane picture.

Plain numpy (sympy for the symbolic checks, matplotlib for the figures).
"""
from .ep_tuning import (
    THEOREM_ASSUMPTIONS,
    EPTuningReport,
    analyze_ep_tuning,
    classify_coupling,
    coupling_classes,
    design_features,
    feedback_coupling,
    route_a_sqrt_coupling,
    route_c_asymmetric,
    theorem_profile,
)
from .pt_dimer import PTChain, PTDimer, eigenvector_overlap, petermann_factor, pt_violation

__all__ = ["PTDimer", "PTChain", "pt_violation", "eigenvector_overlap", "petermann_factor", "EPTuningReport",
           "analyze_ep_tuning", "classify_coupling", "coupling_classes", "route_a_sqrt_coupling", "route_c_asymmetric",
           "feedback_coupling", "theorem_profile", "design_features", "THEOREM_ASSUMPTIONS"]
