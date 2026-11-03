"""Reduced-order modeling workflow (#1208): POD-Galerkin, DMD and operator inference built on one protocol from
full-order-model snapshots, and checked against the full model on parameters and times never used for training."""
from .fom import PeriodicTransportFOM, make_fom
from .workflow import reduced_order_workflow

__all__ = ["PeriodicTransportFOM", "make_fom", "reduced_order_workflow"]
