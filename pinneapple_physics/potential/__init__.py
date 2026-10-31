"""Derivative trees from a potential (``pp.potential``): one scalar function of its natural variables generates every
property by differentiation, consistently.

* :class:`Potential`, :func:`tree` -- a potential (torch or SymPy) and its derivative tree (autodiff, plus symbolic
  where the potential is a SymPy expression);
* :func:`legendre` -- transforms between potentials with a convexity check (:func:`convexity`);
* :func:`maxwell_relations`, :func:`consistency_check` -- the mixed-partial identities, and how far data or a
  property-by-property model is from one potential;
* domain adapters: :mod:`.thermo` (Helmholtz / Gibbs, ideal gas, van der Waals, Peng-Robinson), :mod:`.hyperelastic`
  (Neo-Hookean), :mod:`.mechanics` (Lagrangian, double pendulum), :mod:`.machines` (co-energy of an electric machine
  from FEA samples).

Guide: ``docs/guides/derivative_trees.md``.
"""
from .core import DerivativeTree, Name, Node, Potential, Response, Var, tree
from .legendre import ConvexityReport, LegendreResult, convexity, legendre
from .relations import (
    FAMILIES,
    ConsistencyReport,
    Identity,
    MaxwellRelation,
    consistency_check,
    identities,
    maxwell_relations,
)

__all__ = ["Potential", "Var", "Name", "Response", "Node", "DerivativeTree", "tree", "legendre", "convexity",
           "ConvexityReport", "LegendreResult", "maxwell_relations", "consistency_check", "identities", "Identity",
           "MaxwellRelation", "ConsistencyReport", "FAMILIES"]
