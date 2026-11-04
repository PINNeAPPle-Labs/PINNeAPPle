"""Satellite design (#1207): orbit, power, thermal, link, attitude and lifetime budgets from the mission, the mass
closed at the least value with margins, re-verified by an orbit propagation over the lifetime.

    from pinneapple_design.satellite import satellite_design_workflow
    r = satellite_design_workflow()          # the default 6U Earth-observation mission
"""
from . import budgets
from .workflow import Mission, contact_time, propagate_j2, satellite_design_workflow

__all__ = ["budgets", "Mission", "contact_time", "propagate_j2", "satellite_design_workflow"]
