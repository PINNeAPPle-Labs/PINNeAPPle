"""Chip design (#1206): floorplan thermal with the leakage loop, power-grid IR drop and electromigration, Elmore
delay of critical nets, placement and power-grid design, re-verified on a finer thermal model.

    from pinneapple_design.chip import chip_design_workflow
    r = chip_design_workflow()          # the default floorplan; r["checks"], r["summary"]
"""
from .floorplan import Block, Floorplan, default_floorplan
from .interconnect import WireSpec, elmore_delay, net_delay, transient_step, wire_rc
from .power_grid import GridSpec, black_mttf_years, solve_grid, uniform_grid_closed_form
from .thermal import DieStack, StackLayer, default_stack, solve_thermal, stack_closed_form
from .workflow import chip_design_workflow, floorplan_from_dict

__all__ = ["Block", "Floorplan", "default_floorplan", "WireSpec", "elmore_delay", "net_delay", "transient_step",
           "wire_rc", "GridSpec", "black_mttf_years", "solve_grid", "uniform_grid_closed_form", "DieStack",
           "StackLayer", "default_stack", "solve_thermal", "stack_closed_form", "chip_design_workflow",
           "floorplan_from_dict"]
